"""List top-N leaderboard teams with the archetype(s) they actually played in an episode zip."""
import sys, os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
from meta_analyze import dk, load_elo, iter_games  # noqa: E402

zip_path, lb_dir = sys.argv[1], sys.argv[2]
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 30

elo = load_elo(lb_dir)  # TeamName -> score
top = sorted(elo.items(), key=lambda kv: -kv[1])[:topn]
top_names = {n for n, _ in top}

played = defaultdict(Counter)          # team -> archetype counter
record = defaultdict(lambda: [0, 0])   # team -> [wins, games]
for deck_a, deck_b, winner, names in iter_games(zip_path, max_n=None):
    decks = (deck_a, deck_b)
    for pi in (0, 1):
        name = names[pi]
        if name in top_names:
            arch = dk(decks[pi])
            played[name][arch] += 1
            record[name][1] += 1
            if winner == pi:
                record[name][0] += 1

print(f"{'rank':>4} {'elo':>7}  {'team':<28} {'games':>5} {'wr':>5}  decks")
for i, (name, score) in enumerate(top, 1):
    w, n = record[name]
    decks = ', '.join(f"{a}×{c}" for a, c in played[name].most_common(3)) or '(no games this day)'
    wr = f"{w/n:.0%}" if n else '-'
    print(f"{i:>4} {score:>7.1f}  {name[:28]:<28} {n:>5} {wr:>5}  {decks}")
