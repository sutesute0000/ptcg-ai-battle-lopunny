"""Run an agent against the real meta field, weighted by ladder prevalence.

Local mirror A/B says almost nothing about ladder strength — the field is 39%
Grimmsnarl, not 100% us. This plays the agent against each meta deck and
reports a prevalence-weighted win rate.

Usage: gauntlet.py <agent_dir> [games_per_opponent]
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
agent_dir = os.path.abspath(sys.argv[1])
per = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# Top-tier field share on 2026-08-01 (Elo>=1100 slice).
# `*_strong` dirs are bespoke pilots lifted from the reference repo; the rest are
# GenericPolicy, which is far weaker than a real ladder agent. The difference is
# not cosmetic: GenericPolicy Alakazam reads as a 55% matchup for us, the bespoke
# one reads 33% — and the real ladder says 35%. Prefer a bespoke pilot whenever
# one exists, and treat a GenericPolicy result as an upper bound.
# Shares are from 2026-08-02, Elo>=1100. Refresh them: between 08-01 and 08-02
# Ogerpon fell 23.6%->13.2% while Mega Lucario went 0.1%->4.1%, and a field
# list that omits the decks currently beating us reads far too high.
FIELD = {
    'marnie_s_grimmsnarl_ex': 0.335,
    'mega_lopunny_ex': 0.157,
    'alakazam_strong': 0.146,           # 7.7% -> 14.6% in four days
    'mega_kangaskhan_ex': 0.115,
    'teal_mask_ogerpon_ex': 0.089,
    'dragapult_strong': 0.078,          # 2.7% -> 7.8%, and our worst matchup
    'mega_lucario_ex': 0.023,           # 63% WR; our bot for it reads far too weak
    'thwackey': 0.015,
    'cynthia_s_garchomp_ex': 0.008,
}

# Real top-tier matchup rates for our deck on 2026-08-06, for calibration:
#   Grimmsnarl 57%, Alakazam 48%, Kangaskhan 72%, Ogerpon 61%,
#   Dragapult 44%, Lucario 14%, Thwackey 78%.
# The Alakazam bot tracks reality well (46-50% local vs 48% real). The Lucario
# bot does not (73-83% local vs 14% real) — treat that column as meaningless.

from kaggle_environments import make  # noqa: E402

a_main = os.path.join(agent_dir, 'main.py')
rows = []
for opp, share in FIELD.items():
    opp_dir = os.path.join(ROOT, 'agents', '_opp', opp)
    if not os.path.isdir(opp_dir):
        continue
    b_main = os.path.join(opp_dir, 'main.py')
    os.chdir(opp_dir)
    w = l = 0
    for g in range(per):
        res = make('cabt').run([a_main, b_main] if g % 2 == 0 else [b_main, a_main])
        ai = 0 if g % 2 == 0 else 1
        ra, rb = res[-1][ai]['reward'], res[-1][1 - ai]['reward']
        if ra is None or rb is None:
            continue
        if ra > rb:
            w += 1
        elif ra < rb:
            l += 1
    wr = w / max(1, w + l)
    rows.append((opp, share, w, l, wr))
    print(f"{opp:<28} {w:>3}W-{l:<3}L  {wr:>5.0%}  (field {share:.1%})", flush=True)

tot_share = sum(r[1] for r in rows)
weighted = sum(r[1] * r[4] for r in rows) / max(1e-9, tot_share)
raw = sum(r[2] for r in rows) / max(1, sum(r[2] + r[3] for r in rows))
print(f"\n{os.path.basename(agent_dir)}: weighted WR {weighted:.1%} "
      f"(unweighted {raw:.1%}, {sum(r[2]+r[3] for r in rows)} games)")
