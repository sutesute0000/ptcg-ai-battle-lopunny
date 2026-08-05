"""Emit a compact archetype -> decklist table to embed in the agent.

Simulating the opponent's reply needs their deck: hand the simulator a
placeholder and they draw sixty Basic Grass Energy and do nothing, which makes
the whole 2-ply exercise a lie. We cannot see their list, but we can recognise
the archetype from what they have shown and substitute the consensus build.

Reads the sparring decks already derived from winning ladder lists.
Usage: emit_meta_decks.py > /tmp/decks.py
"""
import os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
from cg.api import all_card_data  # noqa: E402

CT = {c.cardId: c for c in all_card_data()}
OPP = os.path.join(ROOT, 'agents', '_opp')

SKIP = {'alakazam', 'dragapult_ex'}          # superseded by the *_strong dirs

print("# Consensus decklists per archetype, from winning lists on the 2026-08-02")
print("# ladder (tools/emit_meta_decks.py). Used to give the opponent a real deck")
print("# when simulating their reply — a placeholder deck makes them harmless and")
print("# the whole 2-ply read worthless.")
# Matched at runtime by overlap with whatever the opponent has revealed, not by
# a signature card: overlap degrades gracefully when we have seen only a Basic
# or two, and does not care which printing of a support Pokemon they run.
print("META_DECKS = [")
for d in sorted(os.listdir(OPP)):
    if d in SKIP or not os.path.isfile(os.path.join(OPP, d, 'deck.csv')):
        continue
    deck = [int(x) for x in open(os.path.join(OPP, d, 'deck.csv')) if x.strip()]
    if len(deck) != 60:
        continue
    print(f"    # {d}")
    print("    [", end="")
    ids = sorted(deck)
    for i in range(0, len(ids), 15):        # wrap on entries, never mid-number
        print("\n     " + ",".join(str(c) for c in ids[i:i + 15]) + ",", end="")
    print("],")
print("]")
