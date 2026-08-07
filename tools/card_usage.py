"""Find the cards in our own deck that we are failing to use.

The Spiky Hopper discovery came from noticing that top pilots reached for an
attack we essentially never played. This looks for that shape everywhere at
once: replay real games of our archetype, count how often the winner actually
plays each card of the 60, and count how often our agent would have. A card
they use far more than we do is a mechanism we have not wired up — which is a
different and much more productive finding than a difference of taste.

Usage: card_usage.py <episode_zip> <agent_dir> [--max-games 40]
"""
import sys, os, json, zipfile, argparse, importlib.util, warnings
from collections import Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import (AreaType, OptionType, SelectContext, all_card_data,  # noqa: E402
                    all_attack, to_observation_class)

CT = {c.cardId: c for c in all_card_data()}
AT = {a.attackId: a for a in all_attack()}


def load_agent(agent_dir):
    d = os.path.join(ROOT, agent_dir)
    cur = os.getcwd()
    sys.path.insert(0, d)
    os.chdir(d)
    try:
        spec = importlib.util.spec_from_file_location('our_agent', os.path.join(d, 'main.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cur)
    return m


def label(o, obs, pi):
    """What resource does this MAIN option actually spend?"""
    t = o.type
    if t == OptionType.ATTACK:
        a = AT.get(o.attackId)
        return f"[attack] {a.name if a else o.attackId}"
    if t in (OptionType.PLAY, OptionType.ABILITY, OptionType.ATTACH, OptionType.EVOLVE):
        try:
            p = obs.current.players[pi]
            area = o.area if o.area is not None else AreaType.HAND
            seq = {AreaType.HAND: p.hand, AreaType.DISCARD: p.discard,
                   AreaType.ACTIVE: p.active, AreaType.BENCH: p.bench}.get(area)
            if seq and o.index is not None and 0 <= o.index < len(seq) and seq[o.index]:
                c = CT.get(seq[o.index].id)
                kind = {OptionType.PLAY: 'play', OptionType.ABILITY: 'ability',
                        OptionType.ATTACH: 'attach', OptionType.EVOLVE: 'evolve'}[t]
                return f"[{kind}] {c.name if c else seq[o.index].id}"
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('agent_dir')
    ap.add_argument('--arch', default='Mega Lopunny ex')
    ap.add_argument('--opp-arch', default=None)
    ap.add_argument('--max-games', type=int, default=40)
    a = ap.parse_args()

    agent = load_agent(a.agent_dir).agent
    theirs, ours = Counter(), Counter()
    games = 0

    z = zipfile.ZipFile(a.zip)
    for nm in sorted(x for x in z.namelist() if x.endswith('.json')):
        if games >= a.max_games:
            break
        try:
            d = json.loads(z.read(nm))
            rw = d['rewards']
            if rw[0] == rw[1]:
                continue
            win = 0 if rw[0] > rw[1] else 1
            decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
            if str(ma.dk(decks[win])) != a.arch:
                continue
            if a.opp_arch and str(ma.dk(decks[1 - win])) != a.opp_arch:
                continue
            games += 1
            steps = d['steps']
            pi = win
            for t in range(1, len(steps) - 1):
                if pi >= len(steps[t]):
                    continue
                e = steps[t][pi]
                if e.get('status') != 'ACTIVE':
                    continue
                od = e.get('observation') or {}
                sel = od.get('select')
                if not isinstance(sel, dict):
                    continue
                if sel.get('context') != int(SelectContext.MAIN):
                    continue
                opts = sel.get('option') or []
                if len(opts) <= 1:
                    continue
                nxt = steps[t + 1][pi] if pi < len(steps[t + 1]) else None
                if not nxt or nxt.get('action') is None:
                    continue
                obs = to_observation_class(od)
                objs = obs.select.option
                for i in nxt['action']:
                    if i < len(objs):
                        lb = label(objs[i], obs, pi)
                        if lb:
                            theirs[lb] += 1
                try:
                    mine = agent(od)
                except Exception:
                    continue
                for i in mine:
                    if i < len(objs):
                        lb = label(objs[i], obs, pi)
                        if lb:
                            ours[lb] += 1
        except Exception:
            continue

    print(f"replayed {games} games"
          + (f" vs {a.opp_arch}" if a.opp_arch else "") + "\n")
    keys = set(theirs) | set(ours)
    rows = sorted(keys, key=lambda k: (ours[k] - theirs[k]))
    print(f"{'resource':<40}{'they':>6}{'we':>6}{'diff':>7}")
    print("--- 使えていない(彼らが多用、うちが使わない) ---")
    for k in rows[:12]:
        if theirs[k] - ours[k] < 3:
            break
        print(f"{k:<40}{theirs[k]:>6}{ours[k]:>6}{ours[k]-theirs[k]:>+7}")
    print("--- 使いすぎ ---")
    for k in reversed(rows[-8:]):
        if ours[k] - theirs[k] < 3:
            break
        print(f"{k:<40}{theirs[k]:>6}{ours[k]:>6}{ours[k]-theirs[k]:>+7}")


if __name__ == '__main__':
    main()
