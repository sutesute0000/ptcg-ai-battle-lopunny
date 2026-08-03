"""Mega Lopunny ex hit-and-run agent.

Game plan: set up multiple Mega Lopunny ex (330HP, retreat 1), attack with
Gale Thrust (1 colorless, 60 + 170 when the attacker moved from bench to
active this turn) every turn, cycling attackers via free retreat (Air
Balloon) and healing with Wally's Compassion.

Every selection scores each legal option and returns the best; on any
error a legal fallback is returned so the agent never crashes.
"""
import os

import time
from collections import Counter

from cg.api import (
    AreaType, Observation, OptionType, SelectContext, all_attack,
    all_card_data, search_begin, search_end, search_step, to_observation_class,
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

# Pokémon whose attack deals no damage but places damage counters instead.
# That is an *effect*, and Mist Energy prevents all effects of the opponent's
# attacks on its holder — so a Lopunny carrying Mist is simply immune to them.
# Alakazam's Powerful Hand (2 counters per card in the attacker's hand, i.e.
# 300+) is the meta example.
def _build_effect_damage_ids():
    cards = all_card_data()
    ids = {
        c.cardId
        for c in cards
        if c.hp and any(
            ATTACKS.get(aid) is not None
            and not (ATTACKS[aid].damage or 0)
            and 'damage counter' in (ATTACKS[aid].text or '').lower()
            for aid in (c.attacks or [])
        )
    }
    # Include the pre-evolutions: an Abra on the bench already tells us an
    # Alakazam is coming, and the energy we attach now is what will be holding
    # the line by then.
    by_name = {}
    for c in cards:
        if c.hp:
            by_name.setdefault(c.name, []).append(c.cardId)
    for _ in range(2):  # Stage 2 lines are at most two steps back
        for c in cards:
            if c.cardId in ids or not c.hp:
                continue
            evolves_into = [d for d in cards
                            if d.cardId in ids and d.evolvesFrom == c.name]
            if evolves_into:
                ids.add(c.cardId)
    return ids


EFFECT_DAMAGE_IDS = _build_effect_damage_ids()


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


# --- lookahead budget ------------------------------------------------------
# One game allows 10 minutes of thinking and running out loses instantly, so
# spend at most a third of it on search and fall back to static scoring after
# that. _IN_ROLLOUT stops the rollout's own decisions from recursing.
TIME_BUDGET_S = 200.0
_spent = [0.0]
_IN_ROLLOUT = [False]


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

    def opp_board(self):
        board = list(self.opp.bench or [])
        if self.opp.active:
            board += [p for p in self.opp.active if p is not None]
        return board

    def facing_mirror(self):
        return any(p.id in (LOPUNNY, BUNEARY) for p in self.opp_board())

    def facing_effect_damage(self):
        """Opponent fields an attacker that places damage counters instead of
        dealing damage — the thing Mist Energy shuts off completely."""
        return any(p.id in EFFECT_DAMAGE_IDS for p in self.opp_board())

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
                # Run Away Draw: free 3 cards, then it shuffles itself back —
                # the deck's main engine, used every turn by the top pilot.
                return 8200
            return 500
        if t == OptionType.ATTACH:
            src = self.opt_card(o)  # the card in hand (energy or tool)
            tgt = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me_i)
            if tgt is None:
                return 100
            src_id = src.id if src is not None else -1
            is_active_tgt = o.inPlayArea == AreaType.ACTIVE
            if src_id == AIR_BALLOON:
                # Free retreat is the engine of the loop: without it, retreating
                # discards the attacker's energy. Tools survive evolution, so a
                # Balloon on Buneary carries over to Mega Lopunny ex.
                if tgt.tools:
                    return 100  # already has a tool
                if tgt.id == LOPUNNY:
                    return 8700 if is_active_tgt else 7400
                if tgt.id == BUNEARY:
                    return 5000
                if tgt.id in (DUNSPARCE, DUDUNSPARCE):
                    # shield bodies must be able to flee after they are promoted
                    return 5200 if not tgt.energies else 3800
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
            if src_id == MIST:
                # Every Lopunny we fuel is the one that will be Active when the
                # opponent swings back, so against an effect-damage attacker
                # Mist goes on it. Verified in-engine: while Mist is attached,
                # Powerful Hand places no counters at all.
                #
                # This does NOT rescue the Alakazam matchup (28% -> 30% over
                # 150 games) and it was never going to: their whole deck is
                # single-prize while our attacker is worth 3, so they need two
                # knockouts to our six. Blanking some of their attacks cannot
                # close a 3x prize deficit — that is a deck problem, not a
                # piloting one. Kept because it is gated to this matchup and
                # blanking a 300-damage attack cannot be worse than not.
                if tgt.id in (LOPUNNY, BUNEARY) and self.facing_effect_damage():
                    s += 900
                elif is_active_tgt:
                    s += 60
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
            if cid == BUNEARY and self.lopunny_line_on_board() < 3:
                return 8800
            if cid == DUNSPARCE:
                # Feeds the Dudunsparce draw loop and restocks shields, but it
                # goes down AFTER the attacker line: raising it above Buneary
                # (which the pilots' play counts appear to suggest) measured
                # 86.3% -> 79.8% in the gauntlet, mostly from Grimmsnarl.
                if len(self.bench) < self.me.benchMax:
                    return 8100
                return 3000
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
            # costs two cards from hand — only worth it to find a missing piece
            if LOPUNNY not in hand and self.board_count(BUNEARY) > 0 and len(hand) >= 4:
                return 6200
            return 900
        if cid == POKE_PAD:
            return 5200
        if cid == POKEGEAR:
            return 5600
        # supporters: only one per turn, so these compete with each other
        if cid == HILDA:
            # Hilda is the only reliable way to find Mega Lopunny ex (Poké Pad
            # can't touch a Rule Box card). Half our attacks land for 60
            # instead of 230 because no second Lopunny is on the bench, so
            # when the loop has no spare attacker this outranks a blind draw.
            if LOPUNNY not in hand and self.board_count(BUNEARY) > 0:
                return 7600 if self.board_count(LOPUNNY) < 2 else 6600
            return 3000
        if cid == LILLIE:
            return 7000 if len(hand) <= 4 else 1200
        if cid == WALLY:
            # heal the Mega back to 330 and recycle its energy to hand — the
            # top pilot's most-played supporter, so don't hold out for a
            # near-death Lopunny.
            worst = self.most_damaged_mega()
            if worst is None:
                return -200
            dmg = worst.maxHp - worst.hp
            if dmg >= 150:
                return 8700
            if dmg >= 80:
                return 6800
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
        # Paid retreat discards energy equal to the retreat cost. On a shield
        # body that is a fine price for the 230 swing; on a fuelled Lopunny it
        # throws away the very energy that makes it an attacker.
        if self.active.id != LOPUNNY:
            return 8850
        if not free and self.active.energies:
            # Paying the retreat cost discards this Lopunny's only energy, but
            # the alternative is attacking for 60 (scored ~5060) instead of
            # 230. One energy for +170 is worth it while a fuelled attacker
            # waits on the bench.
            return 5500
        if self.active.hp < self.active.maxHp or not self.active.energies:
            return 8850
        return 8820

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
        """Search targets. Dudunsparce is the repeatable draw engine (play
        Dunsparce -> evolve -> Run Away Draw -> it shuffles itself back), so
        the top pilot fetches it about as often as the attacker line."""
        card = self.opt_card(o)
        cid = card.id if card is not None else -1
        hand = self.hand_ids
        # Until a second Lopunny line exists the loop cannot run, so the
        # attacker outranks the draw engine; after that the ordering flips.
        need_attacker = self.board_count(LOPUNNY) < 2
        if cid == LOPUNNY:
            base = 110 if need_attacker else 86
            return base - 25 * (hand.count(LOPUNNY) + self.board_count(LOPUNNY))
        if cid == BUNEARY:
            base = 100 if need_attacker else 90
            return base - 25 * (self.lopunny_line_on_board() + hand.count(BUNEARY))
        if cid == DUNSPARCE:
            # doubles as the sacrificial shield and the draw engine's base
            return 92 - 18 * (hand.count(DUNSPARCE) + self.board_count(DUNSPARCE))
        if cid == DUDUNSPARCE:
            return 62 - 20 * (hand.count(DUDUNSPARCE) + self.board_count(DUDUNSPARCE))
        if cid in ENERGY_IDS:
            n_energy = sum(1 for h in hand if h in ENERGY_IDS)
            # An unfuelled bench Lopunny is the biggest remaining reason an
            # attack lands for 60, but searching harder for energy does NOT
            # convert into wins: prioritising it lifted the Alakazam engine
            # rate 24% -> 41% while that matchup stayed at 28%, and cost ~2pt
            # weighted overall. The engine metric is necessary, not sufficient.
            base = 70 - 30 * n_energy
            # Enriching refunds itself (draw 4); Spiky punishes the attacker
            base += {ENRICHING: 12, SPIKY: 6}.get(cid, 0)
            return base
        if cid == WALLY:
            worst = self.most_damaged_mega()
            return 60 if worst is not None else 35
        if cid == AIR_BALLOON:
            return 50
        if cid in (HILDA, LILLIE):
            return 40
        if cid == BOSS:
            return 30
        if cid == FAN_ROTOM:
            return 25 if self.st.turn <= 2 else 5
        return 20

    def discard_rank(self, o) -> float:
        """Higher = more willing to discard.

        Supporters are the expendable resource: only one can be played per
        turn and the deck runs 4 copies of each, so a second Hilda/Lillie in
        hand is dead weight. The item engine (Poffin/Ultra Ball/Pokégear/Poké
        Pad) and Air Balloon are what keep the loop running, so they stay.
        """
        card = self.opt_card(o)
        cid = card.id if card is not None else -1
        dup = max(0, self.hand_ids.count(cid) - 1)

        if cid == LOPUNNY:
            return -100
        if cid == AIR_BALLOON:
            spare = self.hand_ids.count(AIR_BALLOON) + sum(
                1 for p in [self.active] + list(self.bench)
                if p is not None and any(t.id == AIR_BALLOON for t in (p.tools or [])))
            return -60 if spare <= 2 else 15
        if cid == BUNEARY:
            return -40 if self.lopunny_line_on_board() < 3 else 5
        if cid in ENERGY_IDS:
            # Each attacker needs exactly one energy and Wally's Compassion
            # returns it to hand, so spare energy is the cheapest thing to pitch.
            n_energy = sum(1 for h in self.hand_ids if h in ENERGY_IDS)
            if n_energy >= 2:
                return 60 - (15 if cid == ENRICHING else 0)
            return -20
        if cid == LILLIE:
            return 42 + 12 * dup
        if cid in (HILDA, BOSS, WALLY, XEROSIC):
            return 33 + 12 * dup
        if cid == FAN_ROTOM:
            return 45 if self.st.turn > 2 else -30  # Fan Call is first-turn only
        if cid in (POKEGEAR, POKE_PAD, POFFIN, ULTRA_BALL):
            return 28 + 12 * dup
        if cid in (DUNSPARCE, DUDUNSPARCE):
            return 20
        return 15

    def switch_rank(self, o) -> float:
        """Retreat destination, chosen during OUR turn: this is the Pokémon
        that "moved from the Bench to the Active Spot this turn", so it must
        be a fuelled Mega Lopunny ex to collect Gale Thrust's +170."""
        p = self.opt_card(o)
        if p is None:
            return 0
        if p.id == LOPUNNY:
            return 200 + (80 if p.energies else 0) + p.hp / 10
        if p.id == BUNEARY:
            return 60 + (20 if p.energies else 0)
        return {DUDUNSPARCE: 50, DUNSPARCE: 45, FAN_ROTOM: 30}.get(p.id, 25) + p.hp / 20

    def shield_rank(self, o) -> float:
        """Replace a knocked-out Active (forced promotion between turns).

        Promoting Lopunny here WASTES its attack: Gale Thrust only gets +170
        when the attacker moves bench->active during our own turn, and a
        Pokémon promoted after a KO is already Active when the turn starts.
        Sending up a cheap body instead keeps the 230 loop intact (retreat it,
        promote a fuelled Lopunny, attack) and costs 1 prize instead of 3.
        """
        p = self.opt_card(o)
        if p is None:
            return 0
        can_flee = (any(t.id == AIR_BALLOON for t in (p.tools or []))
                    or len(p.energies or []) >= 1)
        if p.id == LOPUNNY:
            # An unfuelled Lopunny promoted here is a 3-prize wall that cannot
            # attack: top pilots did this once in 57 promotions. A fuelled one,
            # though, is the right answer against everything except the mirror
            # — measured over decisions where both a fuelled Lopunny AND a
            # cheap body were available, they promoted the Lopunny 68-100% of
            # the time (Grimmsnarl 75%, Kangaskhan 75%, Ogerpon 68%), and only
            # 51% in the mirror, where the opponent can actually punch through
            # 330 HP and a 3-prize gift decides the game.
            if not p.energies:
                return 5
            return 120 if self.facing_mirror() else 200
        base = {DUNSPARCE: 100, FAN_ROTOM: 60, DUDUNSPARCE: 55, BUNEARY: 40}.get(p.id, 50)
        return base + (40 if can_flee else 0) + p.hp / 20

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
    # --- forward search -------------------------------------------------
    def _unseen_own_cards(self):
        """Our 60 minus everything we can see = what is still in deck+prizes.

        The simulator draws from whatever deck we hand it, so a placeholder
        would make every simulated draw a lie."""
        seen = Counter()
        p = self.me
        for c in (p.hand or []):
            seen[c.id] += 1
        for c in (p.discard or []):
            seen[c.id] += 1
        for c in (p.prize or []):
            if c is not None:
                seen[c.id] += 1
        for poke in [x for x in (p.active or []) if x is not None] + list(p.bench or []):
            seen[poke.id] += 1
            for c in ((poke.energyCards or []) + (poke.tools or [])
                      + (poke.preEvolution or [])):
                seen[c.id] += 1
        for c in (getattr(self.st, 'stadium', None) or []):
            if c is not None:
                seen[c.id] += 1
        rest = []
        for cid, n in Counter(read_deck_csv()).items():
            rest += [cid] * max(0, n - seen[cid])
        return rest

    @staticmethod
    def _board(pl):
        return ([x for x in (pl.active or []) if x is not None] + list(pl.bench or []))

    def _turn_value(self, end_state) -> float:
        """Score the board as our turn hands over to the opponent."""
        a = end_state.observation.current
        if a is None:
            return -1e9
        am, ao = a.players[self.me_i], a.players[1 - self.me_i]

        before = {p.serial: p for p in self._board(self.opp)}
        after = {p.serial: p for p in self._board(ao)}

        # A knocked-out Pokémon LEAVES the board, taking its accumulated damage
        # with it. Scoring raw board damage therefore punishes the knockout we
        # were aiming for, so count the kills explicitly and only diff the HP
        # of the Pokémon that are still standing.
        kos = sum(1 for s in before if s not in after)
        chip = sum(before[s].hp - after[s].hp for s in before if s in after)

        v = 60000.0 * kos
        v += 100000.0 * (len(self.me.prize) - len(am.prize))
        v += 10.0 * chip
        # a fuelled Lopunny waiting on the bench is next turn's 230
        v += 300.0 * len([p for p in (am.bench or [])
                          if p.id == LOPUNNY and p.energies])
        act = am.active[0] if am.active else None
        if act is not None:
            v += 2.0 * act.hp + (400.0 if act.id == LOPUNNY else 0.0)
        v += 15.0 * am.handCount
        # losing our own bodies is the mirror of the above
        my_before = {p.serial for p in self._board(self.me)}
        my_after = {p.serial for p in self._board(am)}
        v -= 40000.0 * len(my_before - my_after)
        return v

    def _rollout(self, i, rest, need):
        """Take option i, then finish the turn with the static policy."""
        s = search_begin(
            self.obs,
            your_deck=rest[:self.me.deckCount],
            your_prize=rest[self.me.deckCount:need],
            opponent_deck=[1] * self.opp.deckCount,
            opponent_prize=[1] * len(self.opp.prize),
            opponent_hand=[1] * self.opp.handCount,
            opponent_active=[],
        )
        cur = search_step(s.searchId, [i])
        for _ in range(40):
            o = cur.observation
            if o.select is None or o.current is None:
                break
            if o.current.result != -1:
                break
            if o.current.yourIndex != self.me_i:
                break                        # our turn is over
            cur = search_step(cur.searchId, Policy(o).choose())
        return cur

    def _lookahead_main(self, opts, static_best):
        """Use the simulator only for what it judges without ambiguity: which
        first actions actually end the turn with a knockout.

        Ranking whole turns by a hand-written board score was tried first and
        lost badly to plain static scoring (88% -> 40% vs Grimmsnarl): the
        scores are tuned against real top-pilot play, and a crude end-of-turn
        heuristic is not qualified to overrule them. Verifying a kill, though,
        is not a judgement call — so the rollout only gets to speak when it
        finds a knockout the static choice misses.
        """
        rest = self._unseen_own_cards()
        need = self.me.deckCount + len(self.me.prize)
        if len(rest) < need:
            return None                      # deduction failed; do not guess
        _IN_ROLLOUT[0] = True
        try:
            if self._rollout_kos(static_best, rest, need):
                return None                  # static line already kills
            for i in range(len(opts)):
                if i == static_best:
                    continue
                if self._rollout_kos(i, rest, need):
                    return [i]
        finally:
            _IN_ROLLOUT[0] = False
            try:
                search_end()
            except Exception:
                pass
        return None

    def _rollout_kos(self, i, rest, need) -> bool:
        """Does taking option i end our turn having knocked something out?"""
        try:
            end = self._rollout(i, rest, need)
        except Exception:
            return False
        a = end.observation.current
        if a is None:
            return False
        am, ao = a.players[self.me_i], a.players[1 - self.me_i]
        before = {p.serial for p in self._board(self.opp)}
        after = {p.serial for p in self._board(ao)}
        took_prize = len(am.prize) < len(self.me.prize)
        # ...without trading one of ours away. A Pokémon leaving our board is
        # NOT proof of that — Dudunsparce's Run Away Draw shuffles itself back
        # into the deck every turn. The opponent taking a prize is.
        gave_prize = len(ao.prize) < len(self.opp.prize)
        return bool((before - after or took_prize) and not gave_prize)

    def choose(self) -> list[int]:
        sel = self.obs.select
        ctx = sel.context
        opts = sel.option
        n = len(opts)

        if (ctx == SelectContext.MAIN and n > 1 and not _IN_ROLLOUT[0]
                and getattr(self.obs, 'search_begin_input', None)
                and _spent[0] < TIME_BUDGET_S):
            static_best = max(range(n), key=lambda i: self.score_main(opts[i]))
            t0 = time.time()
            try:
                picked = self._lookahead_main(opts, static_best)
            except Exception:
                picked = None
            _spent[0] += time.time() - t0
            if picked is not None:
                return picked

        if ctx == SelectContext.IS_FIRST:
            # go FIRST: the deck wins by setting up two fuelled Lopunny lines
            # before it ever needs to attack (top pilot: 14/14 games first).
            want = OptionType.YES
            return [next((i for i, o in enumerate(opts) if o.type == want), 0)]
        if ctx == SelectContext.MULLIGAN:
            has_basic = any(c in BASIC_POKE for c in self.hand_ids)
            want = OptionType.NO if has_basic else OptionType.YES
            return [next((i for i, o in enumerate(opts) if o.type == want), 0)]

        rankers = {
            SelectContext.MAIN: self.score_main,
            SelectContext.SETUP_ACTIVE_POKEMON: self.setup_rank,
            SelectContext.SETUP_BENCH_POKEMON: self.setup_rank,
            SelectContext.SWITCH: self.switch_rank,
            SelectContext.TO_ACTIVE: self.shield_rank,
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
