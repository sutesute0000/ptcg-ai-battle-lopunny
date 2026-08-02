"""Measure how often the deck's engine actually fires, in local games.

Win rate is noisy and the local bots are weak, but "what fraction of our Gale
Thrusts got the +170" measures the mechanism directly and converges fast. On
the real ladder this sits at 52% (179 of 342); every point is a turn the deck
did what it was built to do.

Usage: engine_rate.py <agent_dir> <opponent_dir> [games]
"""
import os, sys, warnings
from collections import Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
agent_dir = os.path.abspath(sys.argv[1])
opp_dir = os.path.abspath(sys.argv[2])
games = int(sys.argv[3]) if len(sys.argv) > 3 else 30

sys.path.insert(0, agent_dir)
os.chdir(agent_dir)
import main as M  # noqa: E402
from cg.api import SelectContext, OptionType  # noqa: E402

LOPUNNY, AIR_BALLOON = 849, 1174
tally = Counter()
causes = Counter()


def spy(obs_dict, *rest):
    obs = M.to_observation_class(obs_dict)
    if obs.select is None:
        return M.read_deck_csv()
    try:
        sel = M.Policy(obs).choose()
    except Exception:
        tally['agent_error'] += 1
        return M._legal_fallback(obs)
    if obs.select.context == SelectContext.MAIN and sel:
        o = obs.select.option[sel[0]]
        if o.type == OptionType.ATTACK:
            at = M.ATTACKS.get(o.attackId)
            if at and at.name == 'Gale Thrust':
                st = obs.current
                if st.retreated:
                    tally['big'] += 1
                else:
                    tally['weak'] += 1
                    mine = st.players[st.yourIndex]
                    active = mine.active[0] if mine.active else None
                    bench = mine.bench or []
                    fuelled = [p for p in bench if p.id == LOPUNNY and p.energies]
                    if not fuelled:
                        has_lop = any(p.id == LOPUNNY for p in bench)
                        causes['ベンチにミミロップが居ない' if not has_lop
                               else 'ベンチのミミロップにエネが無い'] += 1
                    elif active is not None and any(
                            t.id == AIR_BALLOON for t in (active.tools or [])):
                        causes['かるいし有りなのに、にげなかった'] += 1
                    elif active is not None and (active.energies or []):
                        causes['にげるとエネを捨てることになる'] += 1
                    else:
                        causes['にげるコストを払えない'] += 1
    return sel


os.chdir(opp_dir)
from kaggle_environments import make  # noqa: E402

wins = losses = 0
for g in range(games):
    res = make('cabt').run([spy, os.path.join(opp_dir, 'main.py')])
    ra, rb = res[-1][0]['reward'], res[-1][1]['reward']
    if ra is None or rb is None:
        continue
    wins += ra > rb
    losses += ra < rb

big, weak = tally['big'], tally['weak']
tot = big + weak
print(f"{os.path.basename(agent_dir)} vs {os.path.basename(opp_dir)}, {games} games")
print(f"  win rate      : {wins}/{wins+losses} = {wins/max(1,wins+losses):.0%}")
print(f"  Gale Thrust   : 230dmg {big}  /  60dmg {weak}")
print(f"  ENGINE RATE   : {big/max(1,tot):.1%}   <- higher is better")
if tally['agent_error']:
    print(f"  agent errors  : {tally['agent_error']}")
if causes:
    print("  60ダメージになった理由:")
    for k, v in causes.most_common():
        print(f"      {v:>4} ({v/max(1,weak):>4.0%})  {k}")
