"""D-MEM-SCHED: the KV stage of D BETWEEN wakes (29.09.).

User law (29.09. ~10:20Z): "Ungenutzter Speicher wird, bis er genutzt werden
müsste, durch MoE-Experten befüllt. IMMER. Falls durch irgendwas mehr VRAM
gebraucht wird, werden MoE-Experten in den Systemram verlagert. IMMER."
Design: /spinning/gpu-arb/docs/D-ELASTIK-EXPERTEN-0929.md, section "Deckung
mit #251c/d" -- the one gap left: the #251c/d stage moves only at the wake.

This module is the DECISION half only -- a replicated state machine, no
CUDA. The pages move through the one existing mechanism: the stage form of
#251c/d (``d_seat_vram.StageForm``, the launcher's stages 262144 / 393216 /
524288 -- the sum above 262k IS the ladder) applied by
``d_seat_vram.SeatVram.apply_stage`` (KV prefix of stage j plus the expert
rows of its cell; rows OFF coldest first, ``expert_pool_device.set_seat_rows_on``).
The expert bytes live 1:1 in the store (SlotLedger), so a displaced row writes
nothing back and a row turned on again is filled lazily by the ordinary LRU
fetch on its next miss. The runtime tick is ``d_seat_vram.runtime_tick``.

The mapped KV is the stage that holds ``used + incoming + air`` global tokens:

* grow: the smallest stage holding the need; above the top stage only the
  ADMISSION waits (``admit_wait_stage``) -- never an OOM, never an abort;
* shrink on an END event (finish / abort / park) in the next round -- the
  event is not noise (user 10:40Z) -- but never below a live page: pages an
  unbacked tree node still holds (the L2 write-through is not acked, 27B
  "Lesen statt Rechnen") hold the stage (``stage_down_waited_backup``) and the
  shrink stays PENDING until they drain;
* shrink WITHOUT an event only after ``hysteresis_rounds`` rounds in a row AND
  with one full stage step between the new stage and the need (27B condition
  1: bs swinging 3<->4 at a boundary does not flap).

Every input is GLOBAL (the running requests' tokens, the queue head, the
running rids -- identical on every D rank under uneven DCP 1,2,2): the stage
is the same on every rank without a collective (27B condition 2). The one
collective -- the group's highest live page, MAX -- is asked only when the
replicated machine may shrink, so every rank enters it alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

MARKER = "WEG2 D-MEM-SCHED"
#: 27B condition 1: rounds a no-event shrink condition must hold in a row
HYSTERESIS_ROUNDS = 32


@dataclass
class StageStep:
    stage: int
    tokens: int
    changed: bool
    admit_wait: bool
    reason: str


@dataclass
class MemSched:
    """REPLICATED: the KV stage of the D group, fed with global inputs only."""

    stage_tokens: Tuple[int, ...]
    air_tokens: int
    hysteresis_rounds: int = HYSTERESIS_ROUNDS
    stage: int = 0
    counters: Dict[str, int] = field(default_factory=lambda: {
        "stage_up": 0, "stage_down": 0, "stage_down_on_end": 0,
        "stage_down_waited_backup": 0, "stage_flap": 0, "admit_wait_stage": 0})
    _below: int = 0
    _round: int = 0
    _last_down_round: Optional[int] = None
    #: a decided shrink its live floor still blocks (the stage wanted)
    pending: Optional[int] = None
    #: the runtime tick's bookkeeping: the wake epoch the machine belongs to
    #: and the running rids of the last iteration (their loss = an end event)
    _epoch: object = None
    _rids: Optional[frozenset] = None

    def __post_init__(self):
        toks = tuple(int(t) for t in self.stage_tokens)
        if not toks or any(b <= a for a, b in zip(toks, toks[1:])):
            raise ValueError("%s: stage tokens %s must be strictly rising" % (MARKER, toks))
        self.stage_tokens = toks

    def _smallest_holding(self, need: int) -> Optional[int]:
        for j, t in enumerate(self.stage_tokens):
            if t >= need:
                return j
        return None

    def shrink_candidate(self, used_tokens: int, incoming_tokens: int = 0) -> bool:
        """REPLICATED: may this iteration shrink at all? Only then does the
        caller ask the group for its live-page floor (one collective, entered
        by every rank alike -- 27B condition 1: nothing on the hot path)."""
        need = int(used_tokens) + int(incoming_tokens) + int(self.air_tokens)
        j = self._smallest_holding(need)
        return j is not None and j < self.stage

    def step(self, used_tokens: int, incoming_tokens: int = 0, *, ended: bool = False,
             unbacked_tokens: int = 0, floor_tokens: int = 0) -> StageStep:
        """One scheduler iteration (between rounds / at the admission point).

        ``used_tokens``: the global KV tokens the group holds now;
        ``incoming_tokens``: what the admission wants to add this iteration;
        ``ended``: a request finished / aborted / parked since the last call;
        ``unbacked_tokens``: of the ended requests' pages, those whose L2
        write-through is not acked yet -- they stay mapped until it is;
        ``floor_tokens``: the group's highest live KV page (in tokens) -- a
        stage never ends below a page a request or an unbacked tree node
        holds. A shrink it blocks stays PENDING (``pending``): the caller
        caps new pages below the wanted stage, and the next iteration takes
        it as soon as the floor has drained (no second hysteresis)."""
        self._round += 1
        top = len(self.stage_tokens) - 1
        need = int(used_tokens) + int(incoming_tokens) + int(self.air_tokens)
        cur = self.stage_tokens[self.stage]
        if need > cur:
            self._below = 0
            self.pending = None
            j = self._smallest_holding(need)
            if j is None:
                if int(incoming_tokens) > 0:
                    self.counters["admit_wait_stage"] += 1
                changed = self.stage != top
                if changed:
                    self._count_up()
                self.stage = top
                return StageStep(top, self.stage_tokens[top], changed, int(incoming_tokens) > 0,
                                 "need %d above the top stage: the admission waits" % need)
            self._count_up()
            self.stage = j
            return StageStep(j, self.stage_tokens[j], True, False, "grow to hold %d" % need)
        target = self._smallest_holding(need)
        if target is None or target >= self.stage:
            self._below = 0
            self.pending = None
            return StageStep(self.stage, cur, False, False, "holds")
        if self.pending is not None:
            # a decided shrink waits for its floor only -- the need still fits
            return self._shrink_to(max(self.pending, target), floor_tokens, end=False,
                                   why="pending shrink")
        if ended:
            hold = need + int(unbacked_tokens)
            j = self._smallest_holding(hold)
            j = self.stage if j is None else min(self.stage, j)
            if int(unbacked_tokens) > 0 and j > target:
                self.counters["stage_down_waited_backup"] += 1
            if j < self.stage:
                return self._shrink_to(j, floor_tokens, end=True, why="end event")
            return StageStep(self.stage, cur, False, False, "end event: unbacked pages hold")
        # no event: one full stage step between the new stage and the need
        step = self.stage_tokens[1] - self.stage_tokens[0] if top >= 1 else 0
        gap = self._smallest_holding(need + step)
        if gap is None or gap >= self.stage:
            self._below = 0
            return StageStep(self.stage, cur, False, False, "within one step of the need")
        self._below += 1
        if self._below < int(self.hysteresis_rounds):
            return StageStep(self.stage, cur, False, False,
                             "below for %d/%d rounds" % (self._below, self.hysteresis_rounds))
        self._below = 0
        return self._shrink_to(gap, floor_tokens, end=False,
                               why="below for %d rounds" % self.hysteresis_rounds)

    def _shrink_to(self, want: int, floor_tokens: int, *, end: bool, why: str) -> StageStep:
        """Shrink toward ``want``, never below the live floor; the blocked
        rest stays pending (``stage_down_waited_backup`` once per decision)."""
        want, before = int(want), self.stage
        f = self._smallest_holding(int(floor_tokens))
        j = before if f is None else min(before, max(want, f))
        blocked = j > want
        if blocked and self.pending is None:
            self.counters["stage_down_waited_backup"] += 1
        if j < before:
            self._down(j, end=end)
        self.pending = want if blocked else None
        why = ("%s: live pages up to %d hold S%d (wanted S%d)" % (why, int(floor_tokens), j, want)
               if blocked else "%s: shrink to %d" % (why, self.stage_tokens[j]))
        return StageStep(j, self.stage_tokens[j], j != before, False, why)

    def _count_up(self) -> None:
        self.counters["stage_up"] += 1
        if (self._last_down_round is not None
                and self._round - self._last_down_round < int(self.hysteresis_rounds)):
            self.counters["stage_flap"] += 1

    def _down(self, j: int, *, end: bool) -> None:
        self.stage = j
        self.counters["stage_down"] += 1
        if end:
            self.counters["stage_down_on_end"] += 1
        self._last_down_round = self._round

    def line(self) -> str:
        c = self.counters
        return ("%s stage=%d tokens=%d stage_up=%d stage_down=%d stage_down_on_end=%d "
                "stage_down_waited_backup=%d stage_flap=%d admit_wait_stage=%d"
                % (MARKER, self.stage, self.stage_tokens[self.stage], c["stage_up"],
                   c["stage_down"], c["stage_down_on_end"], c["stage_down_waited_backup"],
                   c["stage_flap"], c["admit_wait_stage"]))


def air_tokens(chunk_tokens: int, seats: int, verify_tokens: int) -> int:
    """What the group can grow by before the next reaction: one extend chunk
    plus one decode round of every seat (n x (k+1) with the draft's k)."""
    return max(0, int(chunk_tokens)) + max(0, int(seats)) * max(1, int(verify_tokens))



