"""Why did we attack for 60 instead of 230?

Gale Thrust only gets its +170 when the attacker moved bench->active that turn,
so every un-bonused attack is a turn the deck's engine did not fire. For each
one, this records the board state and buckets the cause.

Usage: weak_attack_why.py --team すてすて
"""
import os, sys, json, glob, argparse, warnings
from collections import Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
from cg.api import SelectContext, OptionType, all_attack, to_observation_class  # noqa: E402

CACHE = os.path.join(ROOT, '.cache', 'replays')
AT = {a.attackId: a for a in all_attack()}
MAIN = int(SelectContext.MAIN)
LOPUNNY, BUNEARY, AIR_BALLOON = 849, 848, 1174

ap = argparse.ArgumentParser()
ap.add_argument('--team', required=True)
a = ap.parse_args()

causes = Counter()
energy_in_play = Counter()
total = Counter()

for path in sorted(glob.glob(os.path.join(CACHE, '*.json'))):
    try:
        d = json.load(open(path))
        names = [ag['Name'] for ag in d['info']['Agents']]
        if a.team not in names:
            continue
        me = names.index(a.team)
        steps = d['steps']
        for t in range(1, len(steps) - 1):
            if me >= len(steps[t]):
                continue
            e = steps[t][me]
            if e.get('status') != 'ACTIVE':
                continue
            od = e.get('observation') or {}
            sel = od.get('select')
            if not isinstance(sel, dict) or sel.get('context') != MAIN:
                continue
            nxt = steps[t + 1][me] if me < len(steps[t + 1]) else None
            if not nxt or not nxt.get('action'):
                continue
            opts = sel.get('option') or []
            i = nxt['action'][0]
            if i >= len(opts) or opts[i].get('type') != int(OptionType.ATTACK):
                continue
            at = AT.get(opts[i].get('attackId'))
            if not at or at.name != 'Gale Thrust':
                continue
            obs = to_observation_class(od)
            st = obs.current
            if st.retreated:
                total['230 (bonus)'] += 1
                continue
            total['60 (no bonus)'] += 1

            mine = st.players[me]
            active = mine.active[0] if mine.active else None
            bench = mine.bench or []
            fuelled_bench = [p for p in bench if p.id == LOPUNNY and p.energies]
            n_energy = sum(len(p.energies or []) for p in
                           ([active] if active else []) + list(bench))
            energy_in_play[n_energy] += 1

            if not fuelled_bench:
                any_lop = [p for p in bench if p.id == LOPUNNY]
                causes['ベンチに準備済みミミロップが無い'
                       + ('(ミミロップ自体は居るがエネ無し)' if any_lop else '(ミミロップ自体が居ない)')] += 1
            else:
                free = active is not None and any(
                    tl.id == AIR_BALLOON for tl in (active.tools or []))
                if free:
                    causes['にげられたのに、にげなかった(かるいし有り)'] += 1
                elif active is not None and (active.energies or []):
                    causes['にげるにはエネルギーを捨てる必要があった'] += 1
                else:
                    causes['にげるコストを払えなかった(エネ0・かるいし無し)'] += 1
    except Exception:
        continue

n_weak = total['60 (no bonus)']
print(f"Gale Thrust: 230ダメージ {total['230 (bonus)']} 回 / "
      f"60ダメージ {n_weak} 回 "
      f"(弱攻撃の割合 {n_weak / max(1, sum(total.values())):.0%})\n")
print("60ダメージになった理由:")
for k, v in causes.most_common():
    print(f"  {v:>4} ({v/max(1,n_weak):>4.0%})  {k}")
print("\nそのときの場のエネルギー総数:",
      dict(sorted(energy_in_play.items())))
