"""Run N local cabt games between two agent dirs (each: main.py + deck.csv + cg/).

Usage: eval_local.py <dir_a> <dir_b> [games]
Sides alternate each game. Reports A's win rate and any error statuses.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a_dir, b_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
games = int(sys.argv[3]) if len(sys.argv) > 3 else 20

os.chdir(b_dir)  # lets a cwd-based opponent find its deck.csv
from kaggle_environments import make  # noqa: E402

a_main, b_main = os.path.join(a_dir, 'main.py'), os.path.join(b_dir, 'main.py')
wins = losses = draws = 0
errors = []
for g in range(games):
    env = make('cabt')
    agents = [a_main, b_main] if g % 2 == 0 else [b_main, a_main]
    res = env.run(agents)
    a_i = 0 if g % 2 == 0 else 1
    ra, rb = res[-1][a_i]['reward'], res[-1][1 - a_i]['reward']
    sa, sb = res[-1][a_i]['status'], res[-1][1 - a_i]['status']
    if sa != 'DONE' or sb != 'DONE':
        errors.append((g, sa, sb))
    if ra is None or rb is None:
        errors.append((g, 'none-reward', (ra, rb)))
        continue
    if ra > rb:
        wins += 1
    elif ra < rb:
        losses += 1
    else:
        draws += 1
    print(f"game {g+1}: {'A' if ra > rb else 'B' if rb > ra else '-'}  (running A: {wins}W-{losses}L-{draws}D)", flush=True)

print(f"\nA={os.path.basename(a_dir)} vs B={os.path.basename(b_dir)}: "
      f"{wins}W-{losses}L-{draws}D  ({wins / max(1, wins + losses):.0%})")
if errors:
    print("errors:", errors[:10])
