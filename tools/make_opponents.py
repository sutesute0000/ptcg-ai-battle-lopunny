"""Build local sparring partners for the current meta.

Pulls the most-played winning decklist per archetype out of an episode zip and
wraps it in the reference repo's GenericPolicy, giving a competent (not
top-tier) opponent for each deck we actually face on the ladder.

Usage: make_opponents.py <episode_zip> [--min-games 12]
Creates agents/_opp/<archetype>/ with main.py + deck.csv + cg/.
"""
import sys, os, shutil, argparse, warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
from meta_analyze import dk, iter_games  # noqa: E402

REF_BASE = ROOT + '/reference/ptcg-abc/agents/_base'
ENGINE = ROOT + '/docs/official/starter/sample_submission/sample_submission/cg'

MAIN_PY = '''"""Generic sparring opponent: pilots deck.csv with the shared GenericPolicy."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, _HERE)
from generic_policy import make_generic_agent  # noqa: E402
my_deck = [int(x) for x in open(os.path.join(_HERE, "deck.csv")) if x.strip()]
_impl = make_generic_agent(my_deck)


def agent(obs_dict):
    return _impl(obs_dict)
'''


def slug(name):
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name)).strip('_')


ap = argparse.ArgumentParser()
ap.add_argument('zip')
ap.add_argument('--min-games', type=int, default=12)
a = ap.parse_args()

builds = defaultdict(Counter)   # archetype -> Counter(decklist tuple) of WINS
for deck_a, deck_b, winner, names in iter_games(a.zip, max_n=None):
    if winner is None:
        continue
    d = (deck_a, deck_b)[winner]
    if len(d) != 60:
        continue
    builds[str(dk(d))][tuple(sorted(d))] += 1

out_root = os.path.join(ROOT, 'agents', '_opp')
os.makedirs(out_root, exist_ok=True)
made = []
for arch, lists in sorted(builds.items(), key=lambda kv: -sum(kv[1].values())):
    total = sum(lists.values())
    if total < a.min_games:
        continue
    # deterministic: most wins, then lexicographically smallest list
    best, n = max(lists.items(), key=lambda kv: (kv[1], [-x for x in kv[0]]))
    d = os.path.join(out_root, slug(arch))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'deck.csv'), 'w') as f:
        f.write('\n'.join(str(c) for c in best) + '\n')
    with open(os.path.join(d, 'main.py'), 'w') as f:
        f.write(MAIN_PY)
    for src in ('policy_base.py', 'generic_policy.py'):
        shutil.copy(os.path.join(REF_BASE, src), d)
    if not os.path.exists(os.path.join(d, 'cg')):
        shutil.copytree(ENGINE, os.path.join(d, 'cg'))
    made.append((arch, total, n))

print(f"{'archetype':<32}{'wins in zip':>12}{'chosen list used':>18}")
for arch, total, n in made:
    print(f"{arch:<32}{total:>12}{n:>18}")
print(f"\nwrote {len(made)} opponents to agents/_opp/")
