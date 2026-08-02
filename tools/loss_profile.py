"""Profile our wins vs losses from cached ladder replays.

The gauntlet says 82% and the ladder says 42%, so something real opponents do
is not represented locally. This counts, per game, what our agent actually got
to do — attacks made, turns that ended with no attack, whether the attack was
the fuelled 230 or the bare 60 — and splits it by result.

Usage: loss_profile.py --team すてすて [--opp "Mega Lucario ex"]
"""
import os, sys, json, glob, argparse, warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import (SelectContext, OptionType, all_card_data, all_attack,  # noqa: E402
                    to_observation_class)

CACHE = os.path.join(ROOT, '.cache', 'replays')
CT = {c.cardId: c for c in all_card_data()}
AT = {a.attackId: a for a in all_attack()}
MAIN = int(SelectContext.MAIN)

ap = argparse.ArgumentParser()
ap.add_argument('--team', required=True)
ap.add_argument('--opp', default=None)
a = ap.parse_args()

stats = defaultdict(list)      # (won,) -> list of per-game dicts
attack_names = defaultdict(Counter)

for path in sorted(glob.glob(os.path.join(CACHE, '*.json'))):
    try:
        d = json.load(open(path))
        names = [ag['Name'] for ag in d['info']['Agents']]
        if a.team not in names:
            continue
        me = names.index(a.team)
        rw = d['rewards']
        if rw[0] == rw[1]:
            continue
        won = rw[me] > rw[1 - me]
        decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
        if not isinstance(decks[1 - me], list) or len(decks[1 - me]) != 60:
            continue
        opp = str(ma.dk(decks[1 - me]))
        if a.opp and opp != a.opp:
            continue

        steps = d['steps']
        attacks = 0
        big_hits = 0          # Gale Thrust with the +170 bonus
        my_turns = set()
        attacked_turns = set()
        last_turn = 0
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
            cur = od.get('current') or {}
            turn = cur.get('turn', 0)
            last_turn = max(last_turn, turn)
            my_turns.add(turn)
            nxt = steps[t + 1][me] if me < len(steps[t + 1]) else None
            if not nxt or nxt.get('action') is None:
                continue
            idx = nxt['action']
            if not idx:
                continue
            opts = sel.get('option') or []
            if idx[0] >= len(opts):
                continue
            o = opts[idx[0]]
            if o.get('type') == int(OptionType.ATTACK):
                attacks += 1
                attacked_turns.add(turn)
                at = AT.get(o.get('attackId'))
                nm = at.name if at else str(o.get('attackId'))
                attack_names[won][nm] += 1
                if nm == 'Gale Thrust' and cur.get('retreated'):
                    big_hits += 1

        idle = len(my_turns) - len(attacked_turns)
        stats[won].append(dict(opp=opp, attacks=attacks, big=big_hits,
                               turns=len(my_turns), idle=idle, last=last_turn))
    except Exception:
        continue


def show(won):
    rows = stats[won]
    if not rows:
        return
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    label = 'WINS ' if won else 'LOSSES'
    print(f"{label} n={n:<4} 自分のターン数 {avg('turns'):>5.1f} | "
          f"攻撃回数 {avg('attacks'):>4.1f} | うち230ダメージ {avg('big'):>4.1f} | "
          f"攻撃しなかったターン {avg('idle'):>4.1f}")


print(f"opponent filter: {a.opp or '(all)'}\n")
show(True)
show(False)
print("\n使った技の内訳:")
for won in (True, False):
    if attack_names[won]:
        lab = '勝ち' if won else '負け'
        tot = sum(attack_names[won].values())
        print(f"  {lab}: " + ', '.join(f"{k}×{v}" for k, v in attack_names[won].most_common(6))
              + f"  (計{tot})")

# which opponents dominate the losses
print("\n負けた相手の内訳:", dict(Counter(r['opp'] for r in stats[False]).most_common(8)))
