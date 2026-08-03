"""Probe the engine's forward simulator so we know exactly what it gives us.

Answers, on real MAIN decisions:
  - does search_begin work with a dummy opponent deck?
  - can we roll a whole turn forward and see the end-of-turn board?
  - does the search branch stay ours, or does it hand us the opponent's turn?
  - how much wall-clock does one rollout cost?

Usage: search_probe.py <opponent_dir> [games]
"""
import os, sys, time, warnings
from collections import Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(ROOT, 'agents', 'lopunny')
opp_dir = os.path.abspath(sys.argv[1])   # resolve before we chdir
sys.path.insert(0, AGENT)
os.chdir(AGENT)
import main as M  # noqa: E402
from cg.api import (SelectContext, OptionType, search_begin, search_step,  # noqa: E402
                    search_end)

games = int(sys.argv[2]) if len(sys.argv) > 2 else 3

stats = Counter()
timings = []
DECK = M.read_deck_csv()


def unseen_own_cards(obs, mi):
    """The cards from our 60 that we cannot currently see = deck + prizes.

    The simulator draws from whatever `your_deck` we hand it, so passing a
    placeholder makes every simulated draw a lie. Deducing the real remainder
    is what makes a rollout worth anything."""
    seen = Counter()
    p = obs.current.players[mi]
    for c in (p.hand or []):
        seen[c.id] += 1
    for c in (p.discard or []):
        seen[c.id] += 1
    for c in (p.prize or []):
        if c is not None:
            seen[c.id] += 1
    board = list(p.bench or []) + [x for x in (p.active or []) if x is not None]
    for poke in board:
        seen[poke.id] += 1
        for c in (poke.energyCards or []) + (poke.tools or []) + (poke.preEvolution or []):
            seen[c.id] += 1
    stad = getattr(obs.current, 'stadium', None) or []
    for c in stad:
        if c is not None:
            seen[c.id] += 1
    rest = []
    for cid, n in Counter(DECK).items():
        rest += [cid] * max(0, n - seen[cid])
    return rest


def rollout(obs, first_pick, max_steps=40):
    """Take first_pick, then keep playing OUR policy until the turn ends.

    Returns (end_state, n_steps, reason)."""
    st = obs.current
    mi = st.yourIndex
    me, opp = st.players[mi], st.players[1 - mi]
    rest = unseen_own_cards(obs, mi)
    need = me.deckCount + len(me.prize)
    if len(rest) < need:                      # deduction came up short
        rest = rest + [DECK[0]] * (need - len(rest))
    stats['deck_exact'] += (len(rest) == need)
    s = search_begin(
        obs,
        your_deck=rest[:me.deckCount],
        your_prize=rest[me.deckCount:need],
        opponent_deck=[1] * opp.deckCount,
        opponent_prize=[1] * len(opp.prize),
        opponent_hand=[1] * opp.handCount,
        opponent_active=[],
    )
    cur = search_step(s.searchId, first_pick)
    steps = 1
    while steps < max_steps:
        o = cur.observation
        if o.select is None:
            return cur, steps, 'select-none'
        if o.current is not None and o.current.result != -1:
            return cur, steps, 'game-over'
        # once control passes to the opponent, stop: that is the end of our turn
        if o.current is not None and o.current.yourIndex != mi:
            return cur, steps, 'opponent-turn'
        try:
            pick = M.Policy(o).choose()
        except Exception:
            return cur, steps, 'policy-error'
        cur = search_step(cur.searchId, pick)
        steps += 1
    return cur, steps, 'max-steps'


def spy(obs_dict, *rest):
    obs = M.to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    live = M.Policy(obs).choose()
    if (obs.select.context == SelectContext.MAIN
            and obs.search_begin_input and stats['probed'] < 25
            and len(obs.select.option) > 1):
        stats['probed'] += 1
        try:
            t0 = time.time()
            end, steps, why = rollout(obs, live)
            timings.append(time.time() - t0)
            stats[f'end:{why}'] += 1
            stats['steps_total'] += steps
            if end is not None and end.observation.current is not None:
                ec = end.observation.current
                mi = obs.current.yourIndex
                before = obs.current.players[1 - mi]
                after = ec.players[1 - mi]
                b_act = before.active[0] if before.active else None
                a_act = after.active[0] if after.active else None
                if b_act is not None and a_act is not None and b_act.serial == a_act.serial:
                    stats['dmg_total'] += (b_act.hp - a_act.hp)
                    stats['dmg_samples'] += 1
                if len(after.prize) < len(before.prize):
                    stats['took_prize'] += 1
            search_end()
        except Exception as e:
            stats[f'ERR:{type(e).__name__}'] += 1
            stats['errmsg_' + str(e)[:70]] += 1
    return live


from kaggle_environments import make  # noqa: E402
os.chdir(opp_dir)
for g in range(games):
    make('cabt').run([spy, os.path.join(opp_dir, 'main.py')])

print(f"probed MAIN decisions: {stats['probed']}")
for k, v in sorted(stats.items()):
    if k.startswith(('end:', 'ERR', 'errmsg')):
        print(f"  {k}: {v}")
if stats['probed']:
    print(f"  平均ステップ数/ロールアウト: {stats['steps_total']/stats['probed']:.1f}")
if timings:
    print(f"  1ロールアウト: 平均 {sum(timings)/len(timings)*1000:.1f} ms / "
          f"最大 {max(timings)*1000:.0f} ms")
if stats['dmg_samples']:
    print(f"  ターン終了時の相手アクティブへの累計ダメージ: "
          f"平均 {stats['dmg_total']/stats['dmg_samples']:.0f} "
          f"({stats['dmg_samples']}件), サイド取得 {stats['took_prize']}回")
