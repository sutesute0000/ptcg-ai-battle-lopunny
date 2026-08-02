"""Autopsy our OWN ladder games: fetch replays, then report what actually beats us.

Top-pilot divergence tells us what a good pilot does in general; this tells us
where WE lose on the real ladder, which is the higher-signal input once we have
games. Replays are cached under .cache/replays/ so re-runs are cheap.

Usage:
  autopsy.py --team すてすて [--subs 55178313,55178163] [--max-fetch 60]
  autopsy.py --team すてすて --replay-losses   # also replay losses through the agent
"""
import os, sys, json, glob, argparse, subprocess, warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import all_card_data  # noqa: E402

CACHE = os.path.join(ROOT, '.cache', 'replays')
KAGGLE = os.path.join(ROOT, '.venv', 'bin', 'kaggle')
COMP = 'pokemon-tcg-ai-battle'
CT = {c.cardId: c for c in all_card_data()}


def sh(args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def my_submissions():
    out = sh([KAGGLE, 'competitions', 'submissions', COMP])
    ids = []
    for line in out.splitlines()[2:]:
        tok = line.split()
        if tok and tok[0].isdigit():
            ids.append((tok[0], ' '.join(tok[3:-2])[:44]))
    return ids


def episode_ids(sub_id):
    out = sh([KAGGLE, 'competitions', 'episodes', str(sub_id)])
    ids = []
    for line in out.splitlines()[2:]:
        tok = line.split()
        if tok and tok[0].isdigit() and 'VALIDATION' not in line and 'COMPLETED' in line:
            ids.append(tok[0])
    return ids


def fetch(ep_id):
    path = os.path.join(CACHE, f'episode-{ep_id}-replay.json')
    if os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    sh([KAGGLE, 'competitions', 'replay', str(ep_id), '-p', CACHE])
    return path if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team', required=True)
    ap.add_argument('--subs', default=None, help='comma-separated submission ids')
    ap.add_argument('--max-fetch', type=int, default=80)
    a = ap.parse_args()

    subs = ([(s, '') for s in a.subs.split(',')] if a.subs else my_submissions())
    print(f"submissions: {[s for s, _ in subs]}")

    fetched = 0
    per_sub = {}
    for sub, desc in subs:
        eps = episode_ids(sub)
        per_sub[sub] = eps
        for ep in eps:
            if fetched >= a.max_fetch:
                break
            if fetch(ep):
                fetched += 1
    print(f"replays cached: {len(glob.glob(os.path.join(CACHE, '*.json')))} "
          f"(fetched {fetched} this run)\n")

    rec = defaultdict(lambda: [0, 0])       # opp archetype -> [wins, games]
    by_sub = defaultdict(lambda: [0, 0])
    turns = defaultdict(list)
    ep_to_sub = {ep: sub for sub, eps in per_sub.items() for ep in eps}

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
            rec[opp][1] += 1
            rec[opp][0] += int(won)
            ep = os.path.basename(path).split('-')[1]
            sub = ep_to_sub.get(ep, '?')
            by_sub[sub][1] += 1
            by_sub[sub][0] += int(won)
            turns[(opp, won)].append(len(d['steps']))
        except Exception:
            continue

    print(f"{'opponent archetype':<30}{'W':>4}{'G':>5}{'WR':>7}{'avg steps W/L':>16}")
    for opp, (w, g) in sorted(rec.items(), key=lambda kv: -kv[1][1]):
        tw = turns.get((opp, True)) or [0]
        tl = turns.get((opp, False)) or [0]
        print(f"{opp:<30}{w:>4}{g:>5}{w/max(1,g):>6.0%}"
              f"{sum(tw)/len(tw):>8.0f}/{sum(tl)/len(tl):<7.0f}")
    tot_w = sum(w for w, _ in rec.values())
    tot_g = sum(g for _, g in rec.values())
    print(f"\n{'TOTAL':<30}{tot_w:>4}{tot_g:>5}{tot_w/max(1,tot_g):>6.0%}")
    print("\nby submission:")
    for sub, (w, g) in by_sub.items():
        print(f"  {sub}: {w}/{g} = {w/max(1,g):.0%}")


if __name__ == '__main__':
    main()
