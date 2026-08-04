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

LOPUNNY, BUNEARY = 849, 848

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
]


def board(pl):
    return ([x for x in (pl.active or []) if x is not None] + list(pl.bench or []))


def features(state, mi):
    me, opp = state.players[mi], state.players[1 - mi]
    mb, ob = board(me), board(opp)
    ma_ = me.active[0] if me.active else None
    oa = opp.active[0] if opp.active else None
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
        min(state.turn, 40) / 40.0,
    ]


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
    ap.add_argument('zip')
    ap.add_argument('--arch', default='Mega Lopunny ex')
    ap.add_argument('--max-games', type=int, default=600)
    ap.add_argument('--stride', type=int, default=9, help='sample 1 state per N decisions')
    a = ap.parse_args()

    X, y = [], []
    games = 0
    z = zipfile.ZipFile(a.zip)
    for nm in sorted(x for x in z.namelist() if x.endswith('.json')):
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
                    X.append(features(st, pi))
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
    print("\n重み(大きいほど勝ちに効く):")
    for name, wj in sorted(zip(FEATURES, w), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<24}{wj:+.3f}")
    print("\n# paste into the agent")
    print("EVAL_W = [" + ", ".join(f"{v:.4f}" for v in w) + "]")
    print(f"EVAL_B = {b:.4f}")


if __name__ == '__main__':
    main()
