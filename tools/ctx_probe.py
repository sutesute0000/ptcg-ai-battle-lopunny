"""Inspect the board state behind a pilot's choices in ONE SelectContext.

Answers "when does the top pilot pick X instead of Y?" by dumping, for every
decision in that context, the features that plausibly drive it alongside what
they actually chose.

Usage: ctx_probe.py <zip> --player NAME --arch ARCH --context TO_ACTIVE [--max-games 40]
"""
import sys, os, json, zipfile, argparse, warnings
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import SelectContext, all_card_data, to_observation_class, AreaType  # noqa: E402

CT = {c.cardId: c for c in all_card_data()}
LOPUNNY, BUNEARY, DUNSPARCE, DUDUNSPARCE, FAN_ROTOM = 849, 848, 305, 66, 174


def cname(cid):
    c = CT.get(cid)
    return c.name if c else f'#{cid}'


ap = argparse.ArgumentParser()
ap.add_argument('zip'); ap.add_argument('--player'); ap.add_argument('--arch')
ap.add_argument('--context', default='TO_ACTIVE'); ap.add_argument('--max-games', type=int, default=40)
a = ap.parse_args()

want_ctx = {int(c.value) for c in SelectContext if c.name == a.context}
rows = []
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
        win = 0 if rw[0] > rw[1] else 1
        names = [ag['Name'] for ag in d['info']['Agents']]
        if a.player and names[win] != a.player:
            continue
        decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
        if a.arch and str(ma.dk(decks[win])) != a.arch:
            continue
        games += 1
        steps = d['steps']
        pi = win
        for t in range(1, len(steps) - 1):
            if pi >= len(steps[t]):
                continue
            e = steps[t][pi]
            if e.get('status') != 'ACTIVE':
                continue
            obs_d = e.get('observation') or {}
            sel = obs_d.get('select')
            if not isinstance(sel, dict) or sel.get('context') not in want_ctx:
                continue
            if len(sel.get('option') or []) <= 1:
                continue
            nxt = steps[t + 1][pi] if pi < len(steps[t + 1]) else None
            if not nxt or nxt.get('action') is None:
                continue
            obs = to_observation_class(obs_d)
            me = obs.current.players[pi]
            opp = obs.current.players[1 - pi]
            opts = obs.select.option

            def opt_poke(o):
                seq = me.bench if o.area == AreaType.BENCH else me.active
                if seq and o.index is not None and 0 <= o.index < len(seq):
                    return seq[o.index]
                return None

            chosen = [opt_poke(opts[i]) for i in nxt['action'] if i < len(opts)]
            chosen = [c for c in chosen if c is not None]
            if not chosen:
                continue
            c0 = chosen[0]
            avail = [opt_poke(o) for o in opts]
            avail = [p for p in avail if p is not None]
            fueled_lop = [p for p in avail if p.id == LOPUNNY and p.energies]
            cheap = [p for p in avail if p.id in (DUNSPARCE, DUDUNSPARCE, BUNEARY, FAN_ROTOM)]
            rows.append(dict(
                pick=cname(c0.id),
                pick_fueled=bool(c0.energies),
                my_prizes=len(me.prize), opp_prizes=len(opp.prize),
                n_fueled_lop=len(fueled_lop), n_cheap=len(cheap),
                bench=len(me.bench), turn=obs.current.turn,
            ))
    except Exception:
        continue

print(f"{a.context}: {len(rows)} decisions over {games} games\n")
by_pick = Counter(r['pick'] for r in rows)
print("picks:", dict(by_pick.most_common()))

# For each candidate driver, compare its distribution per pick
print(f"\n{'pick':<20}{'n':>5}{'cheapAvail':>12}{'fueledLop':>11}{'bench':>8}{'myPrize':>9}{'oppPrize':>10}{'turn':>7}")
for pick in [p for p, _ in by_pick.most_common()]:
    rs = [r for r in rows if r['pick'] == pick]
    n = len(rs)
    avg = lambda k: sum(r[k] for r in rs) / n
    print(f"{pick:<20}{n:>5}{avg('n_cheap'):>12.2f}{avg('n_fueled_lop'):>11.2f}"
          f"{avg('bench'):>8.2f}{avg('my_prizes'):>9.2f}{avg('opp_prizes'):>10.2f}{avg('turn'):>7.1f}")

# The decisive question: when a cheap body WAS available, what did they pick?
print("\nwhen >=1 cheap body available:")
sub = [r for r in rows if r['n_cheap'] >= 1]
print(" ", dict(Counter(r['pick'] for r in sub).most_common()))
print("  of those, when a fuelled Lopunny was ALSO available:")
sub2 = [r for r in sub if r['n_fueled_lop'] >= 1]
print("  ", dict(Counter(r['pick'] for r in sub2).most_common()))
print("  when NO fuelled Lopunny available:")
sub3 = [r for r in sub if r['n_fueled_lop'] == 0]
print("  ", dict(Counter(r['pick'] for r in sub3).most_common()))
