"""Replay a top pilot's real games through our agent and decode every disagreement.

For each decision the pilot faced, we feed the same observation to our agent and
compare picks. Disagreements are decoded into card/attack names and aggregated,
which is how concrete piloting rules get derived (rather than guessed).

Usage:
  tools/divergence.py <episode_zip> <agent_dir> --player Majkel1337 --arch "Mega Lopunny ex"
                      [--context MAIN] [--max-games 60] [--show 12]
"""
import sys, os, json, zipfile, argparse, importlib.util, warnings
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/reference/ptcg-abc/tools')
sys.path.insert(0, ROOT + '/reference/ptcg-abc/docs/official/models/cg-lib')
import meta_analyze as ma  # noqa: E402
from cg.api import (SelectContext, OptionType, AreaType, all_card_data,  # noqa: E402
                    all_attack, to_observation_class)

CTX_NAME = {int(c.value): c.name for c in SelectContext}
OPT_NAME = {int(o.value): o.name for o in OptionType}
CT = {c.cardId: c for c in all_card_data()}
AT = {a.attackId: a for a in all_attack()}


def cname(cid):
    c = CT.get(cid)
    return c.name if c else f'#{cid}'


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


def decode_opt(o, obs, pi):
    t = getattr(o, 'type', None)
    tn = OPT_NAME.get(int(t), str(t)) if t is not None else '?'

    def card_at(area, idx, owner):
        try:
            p = obs.current.players[owner]
            seq = {AreaType.HAND: p.hand, AreaType.ACTIVE: p.active, AreaType.BENCH: p.bench,
                   AreaType.DISCARD: p.discard, AreaType.DECK: getattr(obs.select, 'deck', None),
                   AreaType.LOOKING: getattr(obs.current, 'looking', None)}.get(area)
            if seq and idx is not None and 0 <= idx < len(seq) and seq[idx] is not None:
                return cname(seq[idx].id)
        except Exception:
            pass
        return None

    owner = o.playerIndex if o.playerIndex is not None else pi
    if tn == 'ATTACK':
        a = AT.get(o.attackId)
        return f"ATTACK:{a.name if a else o.attackId}"
    if tn in ('PLAY', 'ABILITY', 'CARD', 'DISCARD'):
        nm = card_at(o.area if o.area is not None else AreaType.HAND, o.index, owner)
        return f"{tn}:{nm or '?'}"
    if tn in ('ATTACH', 'EVOLVE'):
        src = card_at(o.area if o.area is not None else AreaType.HAND, o.index, owner)
        tgt = card_at(o.inPlayArea, o.inPlayIndex, pi)
        where = 'ACTIVE' if o.inPlayArea == AreaType.ACTIVE else 'BENCH'
        return f"{tn}:{src or '?'}->{tgt or '?'}@{where}"
    if tn in ('ENERGY', 'ENERGY_CARD', 'TOOL_CARD'):
        return f"{tn}:{card_at(o.area, o.index, owner) or '?'}"
    if tn == 'NUMBER':
        return f"NUMBER:{o.number}"
    return tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('agent_dir')
    ap.add_argument('--player', default=None)
    ap.add_argument('--arch', default=None)
    ap.add_argument('--opp-arch', default=None, help='only games against this archetype')
    ap.add_argument('--context', default=None)
    ap.add_argument('--max-games', type=int, default=60)
    ap.add_argument('--show', type=int, default=10)
    a = ap.parse_args()

    agent = load_agent(a.agent_dir).agent
    only = {int(c.value) for c in SelectContext if c.name == a.context} if a.context else None

    human_pick = defaultdict(Counter)
    our_pick = defaultdict(Counter)
    pairs = defaultdict(list)
    agree = defaultdict(lambda: [0, 0])
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
            names = [ag['Name'] for ag in d['info']['Agents']]
            if a.player and names[win] != a.player:
                continue
            decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
            if a.arch and str(ma.dk(decks[win])) != a.arch:
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
                obs_d = e.get('observation') or {}
                sel = obs_d.get('select')
                if not isinstance(sel, dict):
                    continue
                if len(sel.get('option') or []) <= 1:
                    continue
                ctx = sel.get('context')
                if only and ctx not in only:
                    continue
                nxt = steps[t + 1][pi] if pi < len(steps[t + 1]) else None
                if not nxt or nxt.get('status') != 'ACTIVE' or nxt.get('action') is None:
                    continue
                human = sorted(set(nxt['action']))
                try:
                    ours = sorted(set(agent(obs_d)))
                    obs = to_observation_class(obs_d)
                except Exception:
                    continue
                agree[ctx][1] += 1
                if ours == human:
                    agree[ctx][0] += 1
                    continue
                opts = obs.select.option
                hlab = [decode_opt(opts[i], obs, pi) for i in human if i < len(opts)]
                olab = [decode_opt(opts[i], obs, pi) for i in ours if i < len(opts)]
                for l in hlab:
                    human_pick[ctx][l] += 1
                for l in olab:
                    our_pick[ctx][l] += 1
                if len(pairs[ctx]) < a.show:
                    pairs[ctx].append((hlab, olab))
        except Exception:
            continue

    print(f"replayed {games} games of {a.player or a.arch}\n")
    print(f"{'context':<28} {'agree':>7} {'decisions':>10}")
    for ctx in sorted(agree, key=lambda c: agree[c][0] / max(1, agree[c][1])):
        ag, tot = agree[ctx]
        print(f"{CTX_NAME.get(ctx, ctx):<28} {ag/max(1,tot):>6.0%} {tot:>10}")

    for ctx in sorted(agree, key=lambda c: agree[c][0] / max(1, agree[c][1])):
        ag, tot = agree[ctx]
        if tot - ag < 5:
            continue
        print(f"\n=== {CTX_NAME.get(ctx, ctx)} — {tot-ag} disagreements ({ag/max(1,tot):.0%} agree) ===")
        print("  THEY chose:", ', '.join(f"{k}×{v}" for k, v in human_pick[ctx].most_common(8)))
        print("  WE  chose:", ', '.join(f"{k}×{v}" for k, v in our_pick[ctx].most_common(8)))
        for h, o in pairs[ctx][:a.show]:
            print(f"    they={h} | we={o}")


if __name__ == '__main__':
    main()
