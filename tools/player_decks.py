"""Extract a named player's decklist(s) from an episode zip, with card names.

Usage: player_decks.py <episode_zip> <player_name> [--arch <substring>]
Prints each distinct 60-card list the player used (count of games), as cardId x copies + name.
"""
import sys, os, argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
from meta_analyze import dk, iter_games  # noqa: E402
from cg.api import all_card_data  # noqa: E402

CT = {c.cardId: c for c in all_card_data()}

ap = argparse.ArgumentParser()
ap.add_argument('zip'); ap.add_argument('player'); ap.add_argument('--arch', default=None)
a = ap.parse_args()

lists = Counter()   # tuple(sorted deck) -> games
wins = Counter()
for deck_a, deck_b, winner, names in iter_games(a.zip, max_n=None):
    decks = (deck_a, deck_b)
    for pi in (0, 1):
        if names[pi] == a.player:
            arch = dk(decks[pi])
            if a.arch and a.arch.lower() not in str(arch).lower():
                continue
            key = tuple(sorted(decks[pi]))
            lists[key] += 1
            if winner == pi:
                wins[key] += 1

for key, n in lists.most_common(3):
    print(f"\n=== decklist used in {n} games, wr {wins[key]/n:.0%} (archetype: {dk(list(key))}) ===")
    for cid, c in sorted(Counter(key).items()):
        card = CT.get(cid)
        nm = card.name if card else '?'
        print(f"  {cid:>5} x{c}  {nm}")
