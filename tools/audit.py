"""Hunt for defects in the agent, not for improvements.

Two changes have ever gained ground on the real ladder: supplying an exact
fact (verified knockouts) and fixing an actual bug (Boss's Orders was gusting
up the opponent's healthiest Pokemon). Five that widened a model's discretion
all lost. So this looks only for things that are wrong.

Three checks:

1. OWNERSHIP — the Boss's Orders bug class. A context can offer options about
   our cards and theirs in the same list; a ranker that assumes one owner
   silently mis-scores the other. Reports, per context, how often options of
   each owner appear and whether the pick agrees with the ranker's assumption.
2. UNRANKED — contexts with no ranker at all, decided by the generic
   fallback. Frequency tells us which are worth writing.
3. INVARIANTS — decisions that are wasteful on their face: attaching energy a
   Pokemon cannot use, passing on a free ability, ending a turn with an
   unspent attack.

Usage: audit.py <opponent_dir> [games]
"""
import os, sys, warnings, importlib.util
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(ROOT, 'agents', 'lopunny')
sys.path.insert(0, AGENT)
os.chdir(AGENT)
import main as M  # noqa: E402
from cg.api import AreaType, OptionType, SelectContext  # noqa: E402

opp_dir = os.path.join(ROOT, 'agents', '_opp',
                       sys.argv[1] if len(sys.argv) > 1 else 'marnie_s_grimmsnarl_ex')
games = int(sys.argv[2]) if len(sys.argv) > 2 else 10

owners = defaultdict(Counter)      # ctx -> Counter(owner of offered options)
picked_owner = defaultdict(Counter)
unranked = Counter()
invariants = Counter()
RANKED = None


def owner_of(o, mi):
    if o.playerIndex is None:
        return 'unspecified'
    return 'ours' if o.playerIndex == mi else 'THEIRS'


def audit(obs_dict, *rest):
    obs = M.to_observation_class(obs_dict)
    if obs.select is None:
        return M.read_deck_csv()
    pol = M.Policy(obs)
    sel = pol.choose()
    st = obs.current
    mi = st.yourIndex
    ctx = SelectContext(sel_ctx := obs.select.context)
    opts = obs.select.option

    global RANKED
    if RANKED is None:
        RANKED = {c for c in SelectContext
                  if c in _ranker_contexts()}

    if len(opts) > 1:
        for o in opts:
            owners[ctx.name][owner_of(o, mi)] += 1
        for i in sel:
            if i < len(opts):
                picked_owner[ctx.name][owner_of(opts[i], mi)] += 1
        if ctx not in RANKED and ctx not in (SelectContext.IS_FIRST,
                                             SelectContext.MULLIGAN):
            unranked[ctx.name] += 1

    # --- invariants -------------------------------------------------------
    if ctx == SelectContext.MAIN and sel:
        o = opts[sel[0]]
        me = st.players[mi]
        if o.type == OptionType.END:
            # ending while an attack was still on the table
            if any(x.type == OptionType.ATTACK for x in opts):
                invariants['攻撃できるのにターン終了'] += 1
            if any(x.type == OptionType.ABILITY for x in opts):
                invariants['特性を使わずにターン終了'] += 1
            act = me.active[0] if me.active else None
            if act is not None and not (act.energies or []) and any(
                    x.type == OptionType.ATTACH for x in opts):
                invariants['前のポケモンが無エネのままターン終了'] += 1
        if o.type == OptionType.ATTACH:
            tgt = M.get_card(obs, o.inPlayArea, o.inPlayIndex, mi)
            src = pol.opt_card(o)
            if (tgt is not None and src is not None and src.id in M.ENERGY_IDS
                    and tgt.id in (M.DUNSPARCE, M.DUDUNSPARCE, M.FAN_ROTOM)):
                invariants['攻撃に使わないポケモンへのエネ付け'] += 1
    return sel


def _ranker_contexts():
    """The contexts Policy.choose() has an explicit ranker for."""
    src = open(os.path.join(AGENT, 'main.py')).read()
    block = src.split('rankers = {', 1)[1].split('}', 1)[0]
    out = set()
    for line in block.splitlines():
        line = line.strip()
        if line.startswith('SelectContext.'):
            out.add(getattr(SelectContext, line.split('.')[1].split(':')[0]))
    return out


from kaggle_environments import make  # noqa: E402
os.chdir(opp_dir)
for g in range(games):
    M._spent[0] = 0.0
    make('cabt').run([audit, os.path.join(opp_dir, 'main.py')])

print(f"=== 1. 所有者の取り違え (相手のカードが混ざる文脈) ===")
print(f"{'context':<30}{'ours':>7}{'THEIRS':>8}{'unspec':>8}  picked")
for ctx, c in sorted(owners.items(), key=lambda kv: -kv[1]['THEIRS']):
    if not c['THEIRS']:
        continue
    p = picked_owner[ctx]
    print(f"{ctx:<30}{c['ours']:>7}{c['THEIRS']:>8}{c['unspecified']:>8}"
          f"  {dict(p)}")

print(f"\n=== 2. ranker が無く汎用フォールバックで処理した文脈 ===")
for ctx, n in unranked.most_common():
    print(f"  {n:>4}  {ctx}")
if not unranked:
    print("  なし")

print(f"\n=== 3. 見るからに無駄な判断 ===")
for k, n in invariants.most_common():
    print(f"  {n:>4}  {k}")
if not invariants:
    print("  なし")
