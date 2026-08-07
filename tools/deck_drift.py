"""Is our decklist still the one winning with this archetype?

We copied a build on 08-01 and have been tuning the pilot ever since, but the
list itself has never been revisited while the field moved underneath it. This
ranks the builds actually being played now and diffs them against ours.

Usage: deck_drift.py <episode_zip> [--arch "Mega Lopunny ex"] [--deck agents/lopunny/deck.csv]
"""
import sys, os, json, zipfile, argparse, warnings
from collections import Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import all_card_data  # noqa: E402

CT = {c.cardId: c for c in all_card_data()}


def name(cid):
    c = CT.get(cid)
    return c.name if c else f'#{cid}'


ap = argparse.ArgumentParser()
ap.add_argument('zip')
ap.add_argument('--arch', default='Mega Lopunny ex')
ap.add_argument('--deck', default='agents/lopunny/deck.csv')
ap.add_argument('--top', type=int, default=6)
a = ap.parse_args()

ours = tuple(sorted(int(x) for x in open(os.path.join(ROOT, a.deck)) if x.strip()))

builds, wins = Counter(), Counter()
z = zipfile.ZipFile(a.zip)
for nm in sorted(x for x in z.namelist() if x.endswith('.json')):
    try:
        d = json.loads(z.read(nm))
        rw = d['rewards']
        if rw[0] == rw[1]:
            continue
        w = 0 if rw[0] > rw[1] else 1
        for pi in (0, 1):
            deck = d['steps'][1][pi]['action']
            if not isinstance(deck, list) or len(deck) != 60:
                continue
            if str(ma.dk(deck)) != a.arch:
                continue
            k = tuple(sorted(deck))
            builds[k] += 1
            if pi == w:
                wins[k] += 1
    except Exception:
        continue

print(f"{a.arch}: {len(builds)} distinct builds over {sum(builds.values())} appearances\n")
print(f"{'games':>6}{'wr':>6}  diff vs ours")
for k, n in builds.most_common(a.top):
    add = Counter(k) - Counter(ours)
    rem = Counter(ours) - Counter(k)
    s_add = ', '.join(f"+{v} {name(c)}" for c, v in add.items()) or '-'
    s_rem = ', '.join(f"-{v} {name(c)}" for c, v in rem.items()) or '-'
    mark = '   <<< OURS' if k == ours else ''
    print(f"{n:>6}{wins[k]/n:>6.0%}  {s_add} | {s_rem}{mark}")

o = builds.get(ours, 0)
print(f"\nour exact list: {o} appearances"
      + (f", {wins[ours]/o:.0%} win rate" if o else "  <-- nobody is running it any more"))

# Which cards are gaining and losing across the whole archetype, weighted by wins
card_w, card_n = Counter(), Counter()
for k, n in builds.items():
    for cid, c in Counter(k).items():
        card_w[cid] += c * wins[k]
        card_n[cid] += c * n
print(f"\n{'card':<28}{'avg copies (winners)':>22}{'ours':>6}")
rows = []
for cid in set(card_n) | set(ours):
    avg = card_w[cid] / max(1, sum(wins.values()))
    rows.append((avg - Counter(ours)[cid], cid, avg, Counter(ours)[cid]))
for diff, cid, avg, mine in sorted(rows, key=lambda r: -abs(r[0]))[:12]:
    print(f"{name(cid):<28}{avg:>22.2f}{mine:>6}")
