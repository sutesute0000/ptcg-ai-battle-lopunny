"""Learn the rollout's evaluation function from real games.

The first lookahead attempt ranked candidate turns with hand-written board
weights and lost 88% -> 40%. The simulator was fine (signal/noise 9:1); the
weights were guesses. This fits them instead: take board states out of real
replays, label each with whether that player went on to win, and regress.

Outputs plain float weights to paste into the agent, so the submitted code
carries no runtime dependency.

Usage: train_eval.py <episode_zip> [--arch "Mega Lopunny ex"] [--max-games 600]
"""
import sys, os, json, zipfile, argparse, math, random, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import to_observation_class  # noqa: E402

from cg.api import all_card_data, all_attack  # noqa: E402

LOPUNNY, BUNEARY, AIR_BALLOON = 849, 848, 1174
ENERGY_IDS = {11, 13, 14}
CT = {c.cardId: c for c in all_card_data()}
AT = {a.attackId: a for a in all_attack()}

FEATURES = [
    'prize_diff',        # their remaining prizes minus ours: + is us ahead
    'my_board_dmg',      # damage sitting on our Pokemon (normalised)
    'opp_board_dmg',
    'my_active_hp',
    'opp_active_hp',
    'my_bench',
    'opp_bench',
    'my_hand',
    'opp_hand',
    'my_fuelled_attackers',
    'my_deck',
    'turn',
    # --- added after the first model proved out on the ladder ---
    'my_active_is_attacker',   # a Lopunny up front, not a shield body
    'my_active_can_flee',      # Air Balloon on it = next turn's 230 is live
    'opp_active_prizes',       # 3 for a Mega ex, 1 for a lone single-prizer
    'energy_in_hand',
    'my_prizes_left',          # absolute, not just the difference
    'opp_prizes_left',
    # --- the opponent's reply, estimated statically ---
    # Full 2-ply would need their deck, their hand and a stand-in for their
    # policy. The one question that actually decides our turns is cheaper than
    # all of that: can they knock our Active out next turn? We deal 230 and
    # hand back 3 prizes when our Mega ex dies, so "do I survive the reply"
    # is the trade we keep getting wrong.
    'opp_can_ko',
    'opp_threat_ratio',
]


def opp_best_damage(state, mi):
    """Highest damage the opponent's board can put on our Active next turn.

    Energy-aware (they get one attachment), weakness-doubled, and with a
    special case for attacks that place counters per card in hand — printed
    damage reads 0 for those, which is exactly how Alakazam's 300 hides."""
    me, opp = state.players[mi], state.players[1 - mi]
    my_active = me.active[0] if me.active else None
    if my_active is None:
        return 0
    mc = CT.get(my_active.id)
    weak = getattr(mc, 'weakness', None) if mc else None
    best = 0
    board = ([x for x in (opp.active or []) if x is not None] + list(opp.bench or []))
    for p in board:
        c = CT.get(p.id)
        if c is None:
            continue
        is_active = bool(opp.active) and opp.active[0] is not None and opp.active[0].serial == p.serial
        avail = len(p.energies or []) + 1          # one attachment per turn
        for aid in (c.attacks or []):
            a = AT.get(aid)
            if a is None:
                continue
            if len(a.energies or []) > avail:
                continue                            # cannot pay for it
            d = a.damage or 0
            text = (a.text or '').lower()
            if not d and 'damage counter' in text and 'each card in your hand' in text:
                d = 20 * opp.handCount              # 2 counters per card
            if weak is not None and getattr(c, 'energyType', None) == weak:
                d *= 2
            if not is_active:
                d = int(d * 0.7)                    # they must promote it first
            best = max(best, d)
    return best


def board(pl):
    return ([x for x in (pl.active or []) if x is not None] + list(pl.bench or []))


def prize_value(poke):
    c = CT.get(poke.id) if poke is not None else None
    if c is None:
        return 1
    if getattr(c, 'megaEx', False):
        return 3
    if getattr(c, 'ex', False):
        return 2
    return 1


def features(state, mi, hand_ids=None):
    me, opp = state.players[mi], state.players[1 - mi]
    mb, ob = board(me), board(opp)
    ma_ = me.active[0] if me.active else None
    oa = opp.active[0] if opp.active else None
    if hand_ids is None:
        hand_ids = [c.id for c in (me.hand or [])]
    threat = opp_best_damage(state, mi)
    return [
        (len(opp.prize) - len(me.prize)) / 6.0,
        sum(max(0, p.maxHp - p.hp) for p in mb) / 500.0,
        sum(max(0, p.maxHp - p.hp) for p in ob) / 500.0,
        (ma_.hp / 350.0) if ma_ is not None else 0.0,
        (oa.hp / 350.0) if oa is not None else 0.0,
        len(me.bench or []) / 5.0,
        len(opp.bench or []) / 5.0,
        me.handCount / 15.0,
        opp.handCount / 15.0,
        len([p for p in mb if p.id in (LOPUNNY, BUNEARY) and p.energies]) / 3.0,
        me.deckCount / 40.0,
        min(state.turn or 0, 40) / 40.0,
        1.0 if (ma_ is not None and ma_.id == LOPUNNY) else 0.0,
        1.0 if (ma_ is not None
                and any(t.id == AIR_BALLOON for t in (ma_.tools or []))) else 0.0,
        prize_value(oa) / 3.0,
        sum(1 for c in hand_ids if c in ENERGY_IDS) / 4.0,
        len(me.prize) / 6.0,
        len(opp.prize) / 6.0,
        1.0 if (ma_ is not None and threat >= ma_.hp) else 0.0,
        min(1.5, threat / max(1.0, ma_.hp if ma_ is not None else 1.0)),
    ]


# The 12 features of the model that is live on the ladder, plus the one added
# feature that showed real signal. Adding all eight new ones diluted the model
# (test lift 11.4 -> 10.9 -> 10.1 points as features and data grew), so this
# keeps what worked and admits only the measured addition.
FEATURE_SETS = {
    'v1': list(range(12)),
    'all': list(range(20)),
    'v1+threat': list(range(12)) + [19],
}


def fit(X, y, epochs=260, lr=0.35, l2=1e-4):
    """Plain logistic regression, batch gradient descent — no dependencies."""
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for ep in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def accuracy(X, y, w, b):
    ok = 0
    for xi, yi in zip(X, y):
        z = b + sum(wj * xj for wj, xj in zip(w, xi))
        ok += int((z > 0) == (yi > 0.5))
    return ok / max(1, len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip', nargs='+', help='one or more episode zips')
    ap.add_argument('--arch', default='Mega Lopunny ex')
    ap.add_argument('--max-games', type=int, default=600)
    ap.add_argument('--stride', type=int, default=9, help='sample 1 state per N decisions')
    ap.add_argument('--featureset', default='all', choices=sorted(FEATURE_SETS))
    a = ap.parse_args()
    keep = FEATURE_SETS[a.featureset]

    X, y = [], []
    games = 0
    pairs = []
    for zp in a.zip:
        z = zipfile.ZipFile(zp)
        pairs += [(z, nm) for nm in sorted(x for x in z.namelist() if x.endswith('.json'))]
    for z, nm in pairs:
        if games >= a.max_games:
            break
        try:
            d = json.loads(z.read(nm))
            rw = d['rewards']
            if rw[0] == rw[1]:
                continue
            decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
            sides = [pi for pi in (0, 1)
                     if isinstance(decks[pi], list) and len(decks[pi]) == 60
                     and str(ma.dk(decks[pi])) == a.arch]
            if not sides:
                continue
            games += 1
            steps = d['steps']
            for pi in sides:
                won = 1.0 if rw[pi] > rw[1 - pi] else 0.0
                k = 0
                for t in range(1, len(steps)):
                    if pi >= len(steps[t]):
                        continue
                    e = steps[t][pi]
                    if e.get('status') != 'ACTIVE':
                        continue
                    od = e.get('observation') or {}
                    if not isinstance(od.get('current'), dict):
                        continue
                    k += 1
                    if k % a.stride:
                        continue
                    obs = to_observation_class(od)
                    st = obs.current
                    if st is None or st.turn is None:
                        continue
                    X.append([features(st, pi)[j] for j in keep])
                    y.append(won)
        except Exception:
            continue

    print(f"games={games}  samples={len(X)}  positive={sum(y)/max(1,len(y)):.1%}")
    idx = list(range(len(X)))
    random.Random(0).shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in tr]; ytr = [y[i] for i in tr]
    Xte = [X[i] for i in te]; yte = [y[i] for i in te]
    w, b = fit(Xtr, ytr)
    print(f"train acc={accuracy(Xtr, ytr, w, b):.3f}  test acc={accuracy(Xte, yte, w, b):.3f}")
    print(f"  (baseline: always predict the majority class = "
          f"{max(sum(yte), len(yte)-sum(yte))/max(1,len(yte)):.3f})")
    names = [FEATURES[j] for j in keep]
    print("\n重み(大きいほど勝ちに効く):")
    for name, wj in sorted(zip(names, w), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<24}{wj:+.3f}")
    print("\n# paste into the agent")
    print("EVAL_W = [" + ", ".join(f"{v:.4f}" for v in w) + "]")
    print(f"EVAL_B = {b:.4f}")


if __name__ == '__main__':
    main()
