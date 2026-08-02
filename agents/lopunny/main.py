"""Mega Lopunny ex hit-and-run agent.

Game plan: set up multiple Mega Lopunny ex (330HP, retreat 1), attack with
Gale Thrust (1 colorless, 60 + 170 when the attacker moved from bench to
active this turn) every turn, cycling attackers via free retreat (Air
Balloon) and healing with Wally's Compassion.

Every selection scores each legal option and returns the best; on any
error a legal fallback is returned so the agent never crashes.
"""
import os

from cg.api import (
    AreaType, Observation, OptionType, SelectContext, all_attack,
    to_observation_class,
)

# --- our deck's card ids ---
MIST = 11
ENRICHING = 13
SPIKY = 14
DUDUNSPARCE = 66
FAN_ROTOM = 174
DUNSPARCE = 305
BUNEARY = 848
LOPUNNY = 849
POFFIN = 1086
ULTRA_BALL = 1121
POKEGEAR = 1122
POKE_PAD = 1152
AIR_BALLOON = 1174
BOSS = 1182
XEROSIC = 1197
HILDA = 1225
LILLIE = 1227
WALLY = 1229

ENERGY_IDS = {MIST, ENRICHING, SPIKY}
BASIC_POKE = {BUNEARY, DUNSPARCE, FAN_ROTOM}

ATTACKS = {a.attackId: a for a in all_attack()}


def _agent_dir() -> str | None:
    """Directory containing this source file.

    Works both as a normal module (__file__) and under kaggle_environments'
    exec-based loader, where the compiled code object carries the source path.
    """
    import inspect
    here = globals().get("__file__")
    if not here:
        frame = inspect.currentframe()
        here = frame.f_code.co_filename if frame else None
    if here and os.path.exists(here):
        return os.path.dirname(os.path.abspath(here))
    return None


def read_deck_csv() -> list[int]:
    import sys
    candidates = []
    d = _agent_dir()
    if d:
        candidates.append(os.path.join(d, "deck.csv"))
    candidates += ["/kaggle_simulations/agent/deck.csv", "deck.csv"]
    candidates += [os.path.join(p, "deck.csv") for p in sys.path if p]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                rows = f.read().split("\n")
            return [int(rows[i]) for i in range(60)]
    raise FileNotFoundError("deck.csv")


def get_card(obs, area, index, player_index):
    """Resolve an option's (area, index, playerIndex) to a Card/Pokemon."""
    st = obs.current
    if player_index is None:
        player_index = st.yourIndex  # options about our own cards may omit it
    pl = st.players[player_index] if 0 <= player_index < 2 else None
    if area is None:
        area = AreaType.HAND  # PLAY options carry only a hand index

    def safe(arr, i):
        return arr[i] if arr is not None and i is not None and 0 <= i < len(arr) else None

    if area == AreaType.DECK:
        return safe(getattr(obs.select, "deck", None), index)
    if area == AreaType.HAND:
        return safe(getattr(pl, "hand", None), index)
    if area == AreaType.DISCARD:
        return safe(getattr(pl, "discard", None), index)
    if area == AreaType.ACTIVE:
        return safe(getattr(pl, "active", None), index)
    if area == AreaType.BENCH:
        return safe(getattr(pl, "bench", None), index)
    if area == AreaType.LOOKING:
        return safe(getattr(obs.current, "looking", None), index)
    return None


class Policy:
    def __init__(self, obs: Observation):
        self.obs = obs
        st = obs.current
        self.me_i = st.yourIndex
        self.me = st.players[self.me_i]
        self.opp = st.players[1 - self.me_i]
        self.st = st
        self.hand_ids = [c.id for c in (self.me.hand or [])]
        self.active = self.me.active[0] if self.me.active else None
        self.bench = self.me.bench or []

    # --- helpers -------------------------------------------------------
    def fueled_bench_lopunny(self):
        return [b for b in self.bench if b.id == LOPUNNY and b.energies]

    def board_count(self, cid):
        n = sum(1 for b in self.bench if b.id == cid)
        if self.active is not None and self.active.id == cid:
            n += 1
        return n

    def lopunny_line_on_board(self):
        return self.board_count(BUNEARY) + self.board_count(LOPUNNY)

    def opt_card(self, o):
        return get_card(self.obs, o.area, o.index, o.playerIndex)

    # --- per-context scoring -------------------------------------------
    def score_main(self, o) -> float:
        t = o.type
        if t == OptionType.END:
            return -1000
        if t == OptionType.EVOLVE:
            src = self.opt_card(o)
            if src is not None and src.id == LOPUNNY:  # evolving INTO Lopunny
                return 9000
            return 8900
        if t == OptionType.ABILITY:
            card = self.opt_card(o)
            cid = card.id if card else -1
            if cid == FAN_ROTOM:
                return 8500  # first-turn Fan Call: fetch Buneary
            if cid == DUDUNSPARCE:
                return 6000  # Run Away Draw
            return 500
        if t == OptionType.ATTACH:
            src = self.opt_card(o)  # the card in hand (energy or tool)
            tgt = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me_i)
            if tgt is None:
                return 100
            src_id = src.id if src is not None else -1
            is_active_tgt = o.inPlayArea == AreaType.ACTIVE
            if src_id == AIR_BALLOON:
                # free retreat is the engine of the hit-and-run loop
                if tgt.id == LOPUNNY:
                    return 8700 if is_active_tgt else 7400
                if tgt.id == BUNEARY:
                    return 6800
                return 300
            # energy: fuel the NEXT attacker — a benched Lopunny line without
            # energy — so we can retreat-promote and hit 230 every turn
            s = 0.0
            active_fueled = (self.active is not None
                             and self.active.id == LOPUNNY and self.active.energies)
            if tgt.id == LOPUNNY and not tgt.energies:
                if is_active_tgt:
                    s = 8600
                else:
                    s = 8650 if active_fueled else 8450
            elif tgt.id == BUNEARY and not tgt.energies:
                s = 6500 if active_fueled else 5500
            elif tgt.id == DUDUNSPARCE:
                s = 250
            else:
                s = 150
            if src_id == ENRICHING:
                s += 220  # attaching Enriching draws 4
            if src_id == MIST and is_active_tgt:
                s += 60  # attack-effect protection is best on the active
            return s
        if t == OptionType.PLAY:
            card = self.opt_card(o)
            cid = card.id if card else -1
            return self.play_score(cid)
        if t == OptionType.RETREAT:
            return self.retreat_score()
        if t == OptionType.ATTACK:
            return self.attack_score(o)
        if t == OptionType.DISCARD:
            return -500
        return 0

    def play_score(self, cid) -> float:
        hand = self.hand_ids
        if cid in BASIC_POKE:
            # bench bodies early; keep at least 2 Lopunny lines going
            if cid == BUNEARY and self.lopunny_line_on_board() < 3:
                return 8800
            if len(self.bench) < 3:
                return 7800
            return 3000
        if cid == POFFIN:
            if self.lopunny_line_on_board() < 3 and len(self.bench) < self.me.benchMax:
                return 8300
            return 1000
        if cid == AIR_BALLOON:
            return 7200  # free retreat enables the hit-and-run loop
        if cid == ULTRA_BALL:
            if LOPUNNY not in hand and self.board_count(BUNEARY) > 0 and len(hand) >= 3:
                return 7600
            return 900
        if cid == POKE_PAD:
            return 5200
        if cid == POKEGEAR:
            return 5600
        # supporters: engine only offers when legal (one per turn)
        if cid == HILDA:
            if LOPUNNY not in hand and self.board_count(BUNEARY) > 0:
                return 7900
            return 4200
        if cid == LILLIE:
            return 7000 if len(hand) <= 4 else 1200
        if cid == WALLY:
            worst = self.most_damaged_mega()
            if worst is not None and (worst.maxHp - worst.hp) >= 150 and not self.st.energyAttached:
                return 8700
            return -200
        if cid == BOSS:
            return 4000
        if cid == XEROSIC:
            return 3600 if self.opp.handCount >= 6 else 800
        return 1000

    def most_damaged_mega(self):
        megas = [p for p in [self.active] + list(self.bench)
                 if p is not None and p.id == LOPUNNY and p.hp < p.maxHp]
        return max(megas, key=lambda p: p.maxHp - p.hp) if megas else None

    def retreat_score(self) -> float:
        if self.st.retreated or self.active is None:
            return -900
        fueled = self.fueled_bench_lopunny()
        if not fueled:
            return -900
        free = any(t.id == AIR_BALLOON for t in (self.active.tools or []))
        cost_ok = free or len(self.active.energies or []) >= 1
        if not cost_ok:
            return -900
        # swap out a non-attacker or a damaged/spent Lopunny for a fresh one
        if self.active.id != LOPUNNY:
            return 8850
        if self.active.hp < self.active.maxHp or not self.active.energies:
            return 8850
        # healthy fueled Lopunny already active: still worth the +170 bonus
        return 8820 if free else 4800

    def attack_score(self, o) -> float:
        atk = ATTACKS.get(o.attackId)
        if atk is None:
            return 5000
        dmg = atk.damage or 0
        name = (atk.name or "")
        if name == "Gale Thrust" and self.st.retreated:
            dmg += 170
        opp_active = self.opp.active[0] if self.opp.active else None
        if opp_active is not None and dmg >= opp_active.hp:
            dmg += 120  # KO bonus
        return 5000 + dmg

    # --- card-pick contexts --------------------------------------------
    def to_hand_rank(self, o) -> float:
        card = self.opt_card(o)
        cid = card.id if card is not None else -1
        hand = self.hand_ids
        if cid == LOPUNNY:
            return 100 - 20 * hand.count(LOPUNNY)
        if cid == BUNEARY:
            return 90 - 25 * (self.lopunny_line_on_board() + hand.count(BUNEARY))
        if cid in ENERGY_IDS:
            n_energy = sum(1 for h in hand if h in ENERGY_IDS)
            base = 70 - 30 * n_energy
            return base + (5 if cid == MIST else 0)
        if cid == DUNSPARCE:
            return 55 - 20 * hand.count(DUNSPARCE)
        if cid == DUDUNSPARCE:
            return 50 - 15 * hand.count(DUDUNSPARCE)
        if cid in (HILDA, LILLIE, BOSS):
            return 45
        if cid == FAN_ROTOM:
            return 30
        return 20

    def discard_rank(self, o) -> float:
        """Higher = more willing to discard."""
        card = self.opt_card(o)
        cid = card.id if card is not None else -1
        keep = {LOPUNNY: -100, WALLY: -40, BOSS: -35, HILDA: -30, LILLIE: -25}
        if cid in keep:
            n = self.hand_ids.count(cid)
            return keep[cid] + 10 * max(0, n - 1)
        if cid in ENERGY_IDS:
            n_energy = sum(1 for h in self.hand_ids if h in ENERGY_IDS)
            return 50 if n_energy >= 3 else -10
        if cid in (POKEGEAR, POKE_PAD, XEROSIC):
            return 40
        if cid in (POFFIN, ULTRA_BALL, AIR_BALLOON):
            return 20
        if cid in BASIC_POKE:
            return 5
        return 10

    def promote_rank(self, o) -> float:
        p = self.opt_card(o)
        if p is None:
            return 0
        if p.id == LOPUNNY:
            return 100 + (50 if p.energies else 0) + p.hp / 10
        if p.id == DUDUNSPARCE:
            return 80  # 140HP speed bump that recycles itself
        if p.id == DUNSPARCE:
            return 40
        if p.id == BUNEARY:
            return 35
        return 30 + p.hp / 10

    def setup_rank(self, o) -> float:
        c = self.opt_card(o)
        cid = c.id if c is not None else -1
        return {BUNEARY: 100, DUNSPARCE: 60, FAN_ROTOM: 40}.get(cid, 10)

    def opp_target_rank(self, o) -> float:
        """Boss's Orders / damage targets: prefer the easiest KO."""
        p = self.opt_card(o)
        if p is None:
            return 0
        return 200 - p.hp / 5

    def heal_rank(self, o) -> float:
        p = self.opt_card(o)
        if p is None:
            return 0
        return (p.maxHp - p.hp) if p.hp < p.maxHp else 0

    # --- dispatch -------------------------------------------------------
    def choose(self) -> list[int]:
        sel = self.obs.select
        ctx = sel.context
        opts = sel.option
        n = len(opts)

        if ctx == SelectContext.IS_FIRST:
            # go second: attack from turn 1
            want = OptionType.NO
            return [next((i for i, o in enumerate(opts) if o.type == want), 0)]
        if ctx == SelectContext.MULLIGAN:
            has_basic = any(c in BASIC_POKE for c in self.hand_ids)
            want = OptionType.NO if has_basic else OptionType.YES
            return [next((i for i, o in enumerate(opts) if o.type == want), 0)]

        rankers = {
            SelectContext.MAIN: self.score_main,
            SelectContext.SETUP_ACTIVE_POKEMON: self.setup_rank,
            SelectContext.SETUP_BENCH_POKEMON: self.setup_rank,
            SelectContext.SWITCH: self.promote_rank,
            SelectContext.TO_ACTIVE: self.promote_rank,
            SelectContext.TO_BENCH: self.setup_rank,
            SelectContext.TO_HAND: self.to_hand_rank,
            SelectContext.DISCARD: self.discard_rank,
            SelectContext.DISCARD_CARD_OR_ATTACHED_CARD: self.discard_rank,
            SelectContext.TO_DECK: self.discard_rank,
            SelectContext.TO_DECK_BOTTOM: self.discard_rank,
            SelectContext.HEAL: self.heal_rank,
            SelectContext.REMOVE_DAMAGE_COUNTER: self.heal_rank,
            SelectContext.EFFECT_TARGET: self.opp_target_rank,
            SelectContext.DAMAGE: self.opp_target_rank,
            SelectContext.DAMAGE_COUNTER: self.opp_target_rank,
            SelectContext.DAMAGE_COUNTER_ANY: self.opp_target_rank,
            SelectContext.ATTACK: self.attack_score,
        }
        ranker = rankers.get(ctx)

        if ranker is None:
            # YES/NO effects: default to YES; counts: take the max; else first legal
            yes = next((i for i, o in enumerate(opts) if o.type == OptionType.YES), None)
            if yes is not None and sel.maxCount == 1:
                return [yes]
            if n and opts[0].type == OptionType.NUMBER and sel.maxCount == 1:
                best = max(range(n), key=lambda i: opts[i].number or 0)
                return [best]
            k = max(sel.minCount, min(1, sel.maxCount)) if sel.maxCount else sel.minCount
            return list(range(k))

        scored = sorted(range(n), key=lambda i: -ranker(opts[i]))
        if ctx == SelectContext.MAIN or sel.maxCount == 1:
            return [scored[0]]
        # multi-pick: take options while they look worthwhile, at least minCount
        picks = []
        for i in scored:
            if len(picks) >= sel.maxCount:
                break
            if len(picks) < sel.minCount or ranker(opts[i]) > 0:
                picks.append(i)
        return picks if len(picks) >= sel.minCount else scored[: sel.minCount]


def _legal_fallback(obs: Observation) -> list[int]:
    try:
        k = obs.select.minCount or (1 if obs.select.maxCount else 0)
        k = min(max(k, obs.select.minCount), obs.select.maxCount)
        return list(range(k))
    except Exception:
        return [0]


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    try:
        sel = Policy(obs).choose()
        # validate before returning
        n = len(obs.select.option)
        if (len(set(sel)) == len(sel)
                and obs.select.minCount <= len(sel) <= obs.select.maxCount
                and all(isinstance(i, int) and 0 <= i < n for i in sel)):
            return sel
        return _legal_fallback(obs)
    except Exception:
        return _legal_fallback(obs)
