# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#791: PP ranks agree on a batch by DECISION, not by luck.

THE DEFECT (measured, see #788's `_trace_pp_admission_verdict` and the
uniformity-floor scope note next to it in scheduler.py). Under Pipeline
Parallelism every stage is an independent scheduler that re-derives its own
admission verdict from its OWN local radix/eviction state
(`_get_new_batch_prefill_raw`). The #616g rank-uniformity floors that already
solve this for Tensor Parallelism are scoped to `tp_cpu_group` and are a
no-op whenever that group has one member -- true on every rank of a
TP=1/PP=N boot. Requests are chain-forwarded to every stage unconditionally,
but nothing forwards the ADMISSION DECISION alongside them, so two stages can
disagree about which requests are in the batch, or agree on membership but
disagree on how much prefix each request reuses. Either divergence corrupts
the cross-stage activation tensor: `ScheduleBatch.prepare_for_extend` derives
`extend_num_tokens` (the tensor's row count) directly from `len(prefix_indices)`
per request (schedule_batch.py), so a prefix-length disagreement is a SHAPE
disagreement, not an accounting one.

WHAT CROSSES THE WIRE. Never `req.prefix_indices` -- those are literal KV-pool
slot pointers into THIS rank's own pool (schedule_batch.py, `match_prefix`);
shipping them off-rank is meaningless and would itself be a form of the
corruption this module exists to prevent. What crosses is the DECISION: an
ordered list of `(rid, prefix_len, extend_len, admitted)`, built once by the
rank that owns admission truth for the request stream (PP0, the sole rank
that reads from the tokenizer socket -- see
`SchedulerRequestReceiver.recv_requests`, request_receiver.py:117-121,143) and
carried forward through the chain by whichever mechanism the caller wires in
(see CARRIER below). This module has no wire code of its own; it is the pure
decision/reconcile logic, deliberately kept out of `scheduler_pp_mixin.py`
(another strand owns that file's receive path for #789 -- see the module-level
SCOPE FENCE note below) and out of the hot admission path in scheduler.py.

CARRIER (design note, not wired here). The typed tensor-dict/proxy channel
(`pp_typed_channel.py`) is the preferred carrier: it is already keyed per
`(src, kind)` and already documents that "a non-tensor entry travelling in
this dict is established practice on this channel, not a new risk"
(pp_typed_channel.py, module docstring). `kind` is a plain string there, not a
closed enum (`stash_typed`/`take_typed` take any `str`), so a new kind such as
`"admission_decision"` can ride alongside the `"proxy"` tensors for the same
`mb_id` without modifying `pp_typed_channel.py`. It also travels with the
exact activation tensor whose shape the decision determines, which is the
property that matters: a decision and its tensor can never be observed
out of sync. A separate per-pass pyobj message (modelled on the disagg
loops' consensus sends, `_pp_send_pyobj_to_next_stage` /
`point_to_point_pyobj`, scheduler_pp_mixin.py:721-736,902-910) was considered
and rejected for this reason -- it is a second, independently-timed channel,
and nothing pins it to arrive paired with the tensor it describes.

TWO FAILURE SHAPES, ASYMMETRIC (peer-established, HANDOFF quality).
  local match >= told  -> truncate to told. SAFE. This discards some of this
      rank's own legitimate local reuse; it is the identical slack trade
      #616g already makes on the TP axis (MIN-over-ranks), taken here on the
      PP axis instead. `reconcile_pp_admission_decision` takes this path
      silently -- it is not an anomaly, it is the expected common case
      whenever a downstream rank's cache is, or ever was, warmer than PP0's.
  local match <  told   -> UNSAFE, and it is physically un-fixable in the
      SAME pass. `told` was computed by an upstream rank BEFORE this rank's
      shortfall was knowable; the activation tensor already in flight was
      already sized against `told`, and only the FIRST stage does the
      embedding lookup, so a downstream rank that lacks KV for
      `[local, told)` has no token-level input from which to backfill it --
      the hidden states for that range were never computed by any stage,
      because the upstream rank that owns them decided they were reusable
      and skipped their forward entirely. There is no in-pass recomputation
      that closes this gap without an upstream redo, and an upstream redo
      mid-pass is exactly the blocking round-trip
      (scheduler.py:6391-6407's 2026-08-17 deadlock family) this module is
      built to avoid adding to the admission path. So the request cannot be
      honoured THIS pass. `reconcile_pp_admission_decision` therefore never
      fabricates a prefix length for it: it marks the request RETRACTED
      (removed from `effective`, so no caller can accidentally admit it with
      a corrupt length), emits exactly one bounded WARNING naming rank, rid,
      told and local, and carries the retraction forward in the amended
      decision so every remaining downstream rank makes the SAME membership
      decision about that one rid -- membership, not per-token shape, is
      what changes, which is a change PP already has a mechanism for
      (retraction, used today by the disagg decode loop's
      `send_retract_work`) rather than a new kind of failure. The request is
      not lost: it is expected to be re-queued and re-admitted on a LATER
      pass, at which point PP0 builds a fresh decision from current state
      and `told` for it starts at 0 (full recompute, correct, merely
      slower). This module does not implement the re-queue itself -- that is
      scheduler-loop wiring, out of this module's and this phase's scope --
      it only guarantees that what it hands back never lets a caller treat
      an unhonourable length as safe.

THE CONGRUENCE GUARD IS WHAT MAKES THE DEGRADE RARE, NOT WHAT REPLACES IT.
Bounding `told` at admission time against a downstream rank's ACTUAL current
local match would need that rank's state at the moment of the decision, which
is exactly the blocking collective this module must not add
(scheduler.py:6391-6407). A non-blocking, previously-published floor (each
rank's local match from its last completed pass, piggybacked on the existing
output-tensor return trip) narrows the window in which a downstream rank's
cache can still have moved between "last published" and "this pass" -- but it
cannot close that window to zero, so the degrade path above is not optional
scaffolding; it is the thing that makes the guard's staleness survivable
instead of silently wrong. Wiring that publish/consult loop is future work
(also out of scope for scheduler_pp_mixin.py under the current #789 scope
fence); this module's contract holds with or without it.

NO COLLECTIVE. This module performs no `torch.distributed` call of any kind,
by construction -- see `test_reconcile_never_touches_torch_distributed` in
the paired test file for a source-level pin on that property. Every function
here is a pure, rank-local computation over already-local inputs.

DEFAULT PATH. `pp_size <= 1` (today's only shipped configuration) must be
byte-identical to not having this module at all. `reconcile_pp_admission_decision`
and `build_pp_admission_decision` both take `pp_size` and short-circuit to an
identity pass-through when it is `<= 1`; see
`test_pp_size_one_is_a_pure_pass_through` for the pin.

NO HAND-PINNED NUMBERS. There are none in this module: every comparison below
is between two already-materialised local integers (`told` vs `local`), never
a heuristic constant.

#630: THE RETRY LIVELOCK, AND WHY THE DEGRADE ABOVE IS NOT BY ITSELF ENOUGH.
The degrade in "TWO FAILURE SHAPES" excludes an unhonourable request and
expects it to be re-queued and re-admitted on a LATER pass. Nothing in that
sentence forces the later pass to ask for less. Left alone: PP0 re-derives
`told` from its own unchanged local state, tells the downstream rank the
same too-long length, that rank retracts again -- a deterministic cycle with
no forward progress. That IS #630's family (see
`test_pp_disk_hicache_guard_630.py`'s history of this exact codebase's prior
livelock: "a bounded/degrading path that makes no forward progress is the
livelock defect," rooted there in a bounded wait that polled a REPORT
(`is_completed()`) instead of ever consuming a DRIVING signal, so two
polling peers never advanced). "It degrades gracefully" is not sufficient;
the degrade must strictly advance the request's state. `PPAdmissionCongruenceGuard`
below closes this: it remembers each retraction's ACTUAL observed local
match (`observed_local`, always `< told` by construction, since a retraction
only fires on that branch) and clamps the NEXT `told` PP0 offers for that
rid to at most that remembered value, so the second pass cannot repeat the
identical failure -- it can only repeat a STRICTER one, and a strictly
decreasing sequence of non-negative integers is well-founded. It is
rank-local state, mutated only from the same `amended_decision` values
`reconcile_pp_admission_decision` already produces for forwarding, so it
adds no new send/recv and no new synchronisation point -- see NO COLLECTIVE
above, which continues to hold with this class in play.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


#: #944: "this rank could not FIND the request", which is not the same fact as
#: "this rank has no prefix for it" and must never be spelled the same way.
#:
#: THE CLASS, and this is its THIRD instance. `_pp_reconcile_incoming_admission`
#: resolves a rid through a chain of lookups and, on a total miss, used to write
#: 0 into `local_match_lens`. The consumer below then read that 0 as a
#: MEASUREMENT and voided the pass. #797c fixed the chunked_req miss, #798 fixed
#: the wrong-slot miss -- two symptom patches of one class, each adding a lookup
#: while the miss kept answering with a number. #944 is the running-batch miss,
#: measured: 2106 unhonourable-prefix events, 2107 voided passes and a
#: three-rank hang in 35 s under real agent load.
#:
#: The idiom is not invented here. `tp_head_congruence._ABSENT_MATCH` (-1)
#: already distinguishes "does not hold this rid" from a length, one file over,
#: and MIN-reduces to itself for exactly this reason.
#:
#: SCOPE: this value lives INSIDE the resolution path (`local_match_lens` and
#: the reader immediately below it) and never crosses the wire. A rank reports
#: a miss to its peers as `PPAdmissionEntry.unresolved`, a boolean, because
#: `observed_local` has readers that treat it as a length -- see that field.
UNKNOWN_MATCH = -1

#: #944: how many consecutive rounds a rid may be DEFERRED for being
#: unresolvable before PP0 stops asking, refuses loudly, and pins the next
#: offer to `told=0`.
#:
#: WHY A CAP IS LOAD-BEARING AND NOT HOUSEKEEPING. `_learned_floor` is what
#: damped the re-offer before, and it is fed from `observed_local` -- which a
#: lookup miss must not set, because nothing was measured. So `UNKNOWN_MATCH`
#: ALONE makes the measured 2106-void loop WORSE than the old 0 did: the 0 was
#: a false measurement, but it at least clamped. The cap replaces that
#: accidental clamp with a bound that terminates for a stated reason.
#:
#: 3, because a defer exists to let rank-local state SETTLE (a request landing
#: in the running batch, a slot's chunked req being restored), and that settles
#: within one ring lap; `pp_size - 1` rounds is the longest a decision takes to
#: visit every rank, so three gives that a full round trip plus one. A larger
#: number does not buy a better outcome -- it buys more voided passes before
#: the same escape.
UNRESOLVED_DEFER_CAP = 3


class PPScheduleRefused(Exception):
    """#791 CORE: this rank cannot EXECUTE the forwarded pass geometry.

    Raised out of the admission adder and caught by the scheduler, which
    voids the pass through #797's existing machinery. It is deliberately an
    exception and not a return code: every `AddReqResult` a caller can return
    means "build a batch without this request", and building a batch without a
    request the upstream already launched hidden states for is precisely the
    corruption this refusal exists to prevent. There is no narrower batch to
    fall back to.
    """


@dataclass(frozen=True)
class PPAdmissionEntry:
    """One request's admission decision, as it crosses the wire.

    Deliberately does NOT carry `prefix_indices`, `last_node`, or any other
    rank-local pool handle -- see the module docstring's "WHAT CROSSES THE
    WIRE" section. `prefix_len`/`extend_len` are plain integers; a receiver
    reconstructs its own local `prefix_indices` from `prefix_len` against its
    OWN pool (or, if it cannot honour `prefix_len`, retracts -- it never
    borrows the sender's pointers).
    """

    rid: str
    prefix_len: int
    extend_len: int
    admitted: bool = True
    retracted: bool = False
    retracted_by_rank: Optional[int] = None
    observed_local: Optional[int] = None
    """Set only on a newly-retracted entry (see `reconcile_pp_admission_decision`):
    the retracting rank's ACTUAL local match length, i.e. the value that made
    `told` unhonourable (`observed_local < prefix_len` always holds when this
    is set -- that inequality is exactly what triggered the retraction).
    `None` for every non-retracted entry, and for an entry that was already
    retracted by an EARLIER rank (that rank's `observed_local` is preserved
    unchanged; a later rank must not overwrite it with its own, unrelated,
    local state -- see `reconcile_pp_admission_decision`'s pass-through
    branch). This is the signal `PPAdmissionCongruenceGuard` (#630) learns
    from -- never a guess, never a timeout, always a real observed shortfall."""

    unresolved: bool = False
    """#944: this rank could not LOCATE the request at all, so it measured
    nothing and `observed_local` is None.

    A SEPARATE FIELD, NOT A SPECIAL VALUE OF `observed_local`, and that is the
    whole point of this ticket. "I looked and found N tokens" and "I could not
    find the request" are different facts with opposite correct responses -- the
    first must void the pass, the second must let the group retry -- and #797c,
    #798 and #944 are three instances of one class in which they shared a
    spelling and no reader could tell them apart. Encoding the miss as a
    sentinel `observed_local` would be that same class one level up: THIS
    field's readers treat it as a LENGTH and feed it to `_learned_floor`, which
    clamps the next round's offer, and a floor learned from a number nobody
    measured is exactly the defect.

    ON THE WIRE because the rank that OBSERVES the miss is never the rank that
    can ACT on it: only PP0 chooses `told`. Downstream ranks report; PP0 owns
    the count and the escalation (`UNRESOLVED_DEFER_CAP`). A defer that only
    one rank takes IS the next divergence, so no rank takes one alone."""


@dataclass(frozen=True)
class PPAdmissionDecision:
    """An ordered, per-pass admission decision for one PP microbatch slot.

    `mb_id` matches the microbatch-slot keying already used by the typed
    proxy channel (`pp_typed_channel.py`'s `(src, kind)` demux, and
    `_pp_proxy_stamp`'s per-`mb_id` stamping in scheduler_pp_mixin.py) --
    carried here so a caller that wires this onto that channel has the same
    key available, without this module importing anything from it.
    """

    mb_id: int
    entries: Tuple[PPAdmissionEntry, ...]

    def by_rid(self) -> Dict[str, PPAdmissionEntry]:
        return {e.rid: e for e in self.entries}


class PPAdmissionCongruenceGuard:
    """PP0-side memory of downstream shortfalls -- closes the #630 retry
    livelock without a collective. See the module docstring's "#630: THE
    RETRY LIVELOCK" section for the defect and why this shape was chosen
    over a one-shot `told=0` pin.

    WHY "PP0 LEARNS" OVER "ONE-SHOT told=0 PIN" (the two shapes the task
    required evaluating). A pin that forces `told=0` on the very next retry
    also terminates the loop -- an empty prefix is always `<= any local
    match`, so a second retraction for that rid becomes impossible -- but it
    throws away every byte of that rid's reuse, on every rank, forever after
    a single one-token shortfall. Learning the shortfall's ACTUAL value
    (`observed_local`, never zero) and clamping `told` to it is the same
    shape of state (one int per outstanding rid) and no harder to build; the
    difference is entirely in what it buys: the degrade stays RARE (a rid
    that was one token short of `told` loses exactly one token of reuse, the
    next pass, not all of it) instead of merely TERMINATING. Chosen:
    PP0-learns-coverage.

    RID-SCOPED, ONE-SHOT, CLEARS ON SUCCESS. `_learned_floor[rid]` exists
    only while `rid` has an outstanding, unresolved shortfall. The instant a
    pass admits `rid` with NO retraction anywhere in the chain -- observed
    via `record_return_trip` on the fully chain-reconciled decision, the same
    value already threaded rank-to-rank as `amended_decision` -- the entry is
    deleted. A rid that was once short is not permanently capped once its
    cache state (on whichever rank was short) has moved on; this is the
    "clears once served" requirement the one-shot-pin shape would also have
    had to satisfy, met here for a learned value instead of a boolean.

    NO COLLECTIVE. Rank-local mutable state, read and written only by the
    rank that calls `build_pp_admission_decision` (PP0) and fed only by
    decisions that already crossed the wire through the EXISTING retraction
    mechanism. No new send, no new recv, no new synchronisation point -- see
    NO COLLECTIVE in the module docstring.

    TERMINATION (well-founded, not merely bounded). A retraction only fires
    when `local < told` (by construction of `reconcile_pp_admission_decision`),
    so every NEW retraction for `rid` sets `_learned_floor[rid]` to a value
    strictly less than the `told` that just failed. The next cycle's `told`
    for that rid is clamped to `<= _learned_floor[rid]`, so it is strictly
    less than the `told` that just failed -- a strictly decreasing sequence
    of non-negative integers, which cannot cycle and cannot decrease forever.
    It terminates in at most `told_initial` cycles, and in practice within at
    most `pp_size - 1` (each cycle can only discover a floor from a rank that
    has not yet reconciled this rid this pass).
    """

    def __init__(self, unresolved_defer_cap: int = UNRESOLVED_DEFER_CAP) -> None:
        self._learned_floor: Dict[str, int] = {}
        #: #944: consecutive rounds `rid` came back UNRESOLVED (no rank could
        #: locate it). Distinct from `_learned_floor` because it counts a
        #: different population -- see `PPAdmissionEntry.unresolved`.
        self._unresolved_rounds: Dict[str, int] = {}
        #: rids whose loud refusal has already been emitted, so the escalation
        #: is one line and not one line per pass. A bounded refusal that
        #: re-logs every pass is the 2106-line log this ticket is about, in a
        #: different colour.
        self._escalated: set = set()
        #: <= 0 disables the #944 bound entirely, restoring the pre-cap
        #: behaviour exactly -- the same neuterable escape `#552`'s
        #: `defer_limit` carries, so a can-fail proof can take the cap down on
        #: its own without taking the sentinel down with it.
        self._unresolved_defer_cap = int(unresolved_defer_cap)

    def unresolved_rounds(self, rid: str) -> int:
        """#944 diagnostic/test hook: consecutive unresolved rounds for `rid`."""
        return self._unresolved_rounds.get(rid, 0)

    def learned_floor(self, rid: str) -> Optional[int]:
        """Diagnostic/test hook: the outstanding floor for `rid`, or None.

        The SIBLING of `unresolved_rounds` above, and both exist so a test can
        tell the two populations apart from the outside -- reading them off one
        accessor, or off `prefix_len_for`, would fold them back together at the
        exact seam this ticket is about.
        """
        return self._learned_floor.get(rid)

    def prefix_len_for(self, rid: str, candidate_prefix_len: int) -> int:
        """What PP0 should tell downstream ranks for `rid` this pass.

        `candidate_prefix_len` is PP0's own fresh local match -- what
        `build_pp_admission_decision` would tell without this guard.
        Clamped to the learned floor if one is outstanding for `rid`;
        returned unchanged otherwise (a rid with no retraction history is
        not constrained by this guard at all).

        #944 ESCALATION, AND IT IS DECIDED HERE BECAUSE ONLY PP0 CAN ACT ON IT.
        A rid that has come back UNRESOLVED `UNRESOLVED_DEFER_CAP` times in a
        row is pinned to `told=0` and refused loudly, once. Downstream ranks
        observe the miss but cannot fix it -- they do not choose `told`, and a
        rank that rewrote the geometry mid-ring would be the instr20 crash --
        so the report rides the wire and the single decision is taken here.

        WHY told=0 WHEN THIS CLASS'S OWN DOCSTRING REJECTS A told=0 PIN. It
        rejects it as the GENERAL policy for an ordinary shortfall, where
        learning the real observed value costs the same and keeps the degrade
        rare. That argument does not reach here: a lookup miss measured
        NOTHING, so there is no value to learn, and `told=0` is the only offer
        that is honourable without a measurement (`reconcile_pp_admission_
        decision` admits it unconditionally). It is therefore not the rejected
        shape creeping back in -- it is the only terminator available once the
        measurable one is gone. The cost is one request's prefix reuse, once,
        after three voided rounds; the alternative is the unbounded re-offer,
        which is the #858 livelock shape.
        """
        if self._unresolved_defer_cap > 0:
            rounds = self._unresolved_rounds.get(rid, 0)
            if rounds >= self._unresolved_defer_cap:
                if rid not in self._escalated:
                    self._escalated.add(rid)
                    logger.error(
                        "#944 PP-ADMISSION UNRESOLVABLE rid=%s: %d consecutive "
                        "rounds in which no PP rank could locate this request "
                        "in any of the four places a request can live -- the "
                        "waiting queue, `chunked_req`, the named slot's "
                        "chunked req, or the running batch. No rank measured a "
                        "prefix, so no floor can be learned and the offer "
                        "cannot be damped by measurement. Offering told=0 "
                        "instead: it is honourable without a measurement, so "
                        "the pass runs, at the cost of this request's prefix "
                        "reuse for one pass. If this line repeats for many "
                        "rids, the lookup chain has a FIFTH gap and that is "
                        "the bug -- not this refusal.",
                        rid,
                        rounds,
                    )
                return 0
        floor = self._learned_floor.get(rid)
        if floor is None:
            return candidate_prefix_len
        return min(candidate_prefix_len, floor)

    def record_return_trip(self, decision: PPAdmissionDecision) -> None:
        """Consume a fully chain-reconciled decision on its way back to PP0.

        For every entry:
          * retracted this pass -> learn `_learned_floor[rid] = observed_local`,
            tightening (never widening) any floor already outstanding -- a
            stricter rank's finding must never be overwritten by a looser one.
          * admitted, not retracted -> the rid was served this pass with no
            shortfall anywhere in the chain; clear any outstanding floor.
          * neither (excluded before this module ever saw it, e.g. by PP0's
            own local admission control) -> not this guard's concern, left
            untouched.

        Performs no distributed communication and never blocks: this is a
        plain dict update over already-local data, the same
        `amended_decision` value the caller already has in hand for
        forwarding.
        """
        for entry in decision.entries:
            if entry.retracted:
                if entry.unresolved:
                    # #944 THE OTHER POPULATION, COUNTED SEPARATELY. No rank
                    # could locate this rid, so there is nothing to learn a
                    # floor from -- but the round still happened and still cost
                    # a voided pass, and something has to bound how many times
                    # that may repeat. Counted here rather than at the reporting
                    # rank because the count has to be the GROUP's: every rank
                    # reports independently, only PP0 acts (`prefix_len_for`).
                    #
                    # INCREMENT-ONLY ON THIS PATH, cleared only by a pass that
                    # actually served the rid (below). A defer that resets its
                    # own counter makes the bound unreachable, which is #552's
                    # measured lesson: "the bug itself wearing a fix".
                    self._unresolved_rounds[entry.rid] = (
                        self._unresolved_rounds.get(entry.rid, 0) + 1
                    )
                    continue
                observed = entry.observed_local
                if observed is None:
                    # A retraction without an observed value cannot safely
                    # tighten a floor -- leave any existing floor as-is rather
                    # than guess.
                    #
                    # #944 MADE THIS AN EXPECTED PATH, and the comment is
                    # corrected rather than left to mislead. It used to read
                    # "every retraction this module itself produces always sets
                    # observed_local, so this branch guards a malformed/foreign
                    # decision, not an expected path". The #944 UNRESOLVED
                    # branch now retracts with `observed_local=None` on purpose:
                    # a LOOKUP MISS taught this pass nothing about the rank's
                    # prefix, so there is no lesson to learn, and inventing one
                    # (0, or the -1 sentinel) would clamp the next round's offer
                    # from a number that was never measured. The guard was
                    # already the correct behaviour; only its reachability
                    # changed.
                    continue
                existing = self._learned_floor.get(entry.rid)
                self._learned_floor[entry.rid] = (
                    observed if existing is None else min(existing, observed)
                )
            elif entry.admitted:
                self._learned_floor.pop(entry.rid, None)
                # #944: the rid was SERVED, so both kinds of outstanding state
                # are stale -- the learned floor and the unresolved streak. The
                # streak must clear here and nowhere else: one bad minute of
                # lookups must not cap a request's reuse for the rest of its
                # life, and the escalation must be able to fire again if the
                # rid genuinely becomes unresolvable a second time.
                self._unresolved_rounds.pop(entry.rid, None)
                self._escalated.discard(entry.rid)

    def outstanding_rids(self) -> Tuple[str, ...]:
        """Diagnostic/test hook: rids currently carrying a learned floor."""
        return tuple(self._learned_floor)


def _executed_extent(req) -> Optional[Tuple[int, int]]:
    """`(start, length)` of the extend range this rank ACTUALLY BUILT, or None.

    THE BATCH IS THE SCHEDULE -- this reads the very field
    `ScheduleBatch.prepare_for_extend` sizes the cross-stage tensor from
    (`extend_range.start` == `len(prefix_indices)`, `extend_range.length`;
    schedule_batch.py:2261-2281), rather than recomputing a value that ought
    to agree with it. Two expressions that must agree is the defect one level
    up, so there is exactly one expression.

    `None` ONLY for a request the adder never touched. Every one of the eight
    `can_run_list.append(req)` sites in schedule_policy.py is immediately
    preceded by a `set_extend_range` (:1168/:1170, :1199/:1200, :1297/:1298,
    :1420/:1421, :1535/:1538, :1555/:1558, :1742/:1745, :1785/:1789), so a
    member of a real `can_run_list` always has one and the fallback in
    `build_pp_admission_decision` is reachable only from a bare test stand-in.
    That was worth establishing rather than assuming: a `getattr` default
    that silently produced 0, or silently produced the full length, would be
    the instr21 defect in new clothing.

    A ZERO-LENGTH RANGE IS REPORTED, NOT SUPPRESSED. `add_chunked_req`'s #679
    park (schedule_policy.py:1399) sets `extend_range(prefix, prefix)` and
    returns WITHOUT appending, so a parked chunk is not in `can_run_list` at
    all; but a chunk that lands exactly on its last token can be appended with
    `new_len == 0` (:1420-1421). If the first rank ran zero rows for a
    request, every rank must run zero rows for it -- reporting that faithfully
    is the contract, and inventing a length for it would be the same defect
    again. Downstream, `schedule_refusal_reason` therefore refuses a NEGATIVE
    extend and executes a zero verbatim.
    """
    extend_range = getattr(req, "extend_range", None)
    if extend_range is None:
        return None
    start = getattr(extend_range, "start", None)
    end = getattr(extend_range, "end", None)
    if start is None or end is None:
        return None
    start, end = int(start), int(end)
    return start, max(0, end - start)


def build_pp_admission_decision(
    mb_id: int,
    reqs: Sequence,
    *,
    pp_size: int,
    guard: Optional[PPAdmissionCongruenceGuard] = None,
    require_executed_geometry: bool = False,
) -> PPAdmissionDecision:
    """PP0's (or, under `pp_size<=1`, the only rank's) committed decision.

    Reads `req.rid`, `len(req.prefix_indices)`, and the request's own extend
    length (`req.extend_input_len` if present, else derived from
    `full_untruncated_fill_ids` minus the prefix) -- all values this rank
    already computed locally while building its batch. Emits only integers;
    `prefix_indices` itself never leaves this function.

    `guard`, when given and `pp_size > 1`: clamps `prefix_len` to any
    learned floor outstanding for that rid (#630 -- see
    `PPAdmissionCongruenceGuard`), and correspondingly INCREASES `extend_len`
    by the same amount so `prefix_len + extend_len` still equals the
    request's true total length. This is the same physical token count
    either way; clamping `told` down does not shorten the request, it only
    reclassifies some of its tokens from "reused prefix" to "freshly
    computed extend" -- exactly the same accounting
    `reconcile_pp_admission_decision`'s safe-truncate branch already relies
    on implicitly (there, the caller's own downstream scheduling absorbs the
    difference; here, the wire decision must state it explicitly, since it
    is computed before any downstream rank sees it).

    `guard=None` (the default) or `pp_size<=1`: behaviour is byte-identical
    to before this parameter existed -- see DEFAULT PATH above, still true
    with a guard object in play as long as it is never passed a `pp_size<=1`
    caller's decisions to learn from.

    #791 CORE: THE DECISION REPORTS WHAT THIS RANK RAN, NOT WHAT IT WAS
    OFFERED -- and until boot instr21 it reported neither.

    `req.extend_input_len` DOES NOT EXIST. There is no assignment to it
    anywhere in the tree; it survives only as a doc comment
    (schedule_batch.py:1847) and two stale references (:2379-2380). So the
    `getattr` below has ALWAYS returned `None` and the fallback has ALWAYS
    run -- and the fallback computes `len(full_untruncated_fill_ids) -
    prefix`, i.e. the WHOLE remaining prompt. Under chunked prefill that is
    never the batch: `PrefillAdder.add_one_req` caps the chunk at
    `rem_chunk_tokens` AFTER this value would have been read
    (schedule_policy.py:1738-1766), and `add_chunked_req` does the same
    (:1415-1419).

    Nothing noticed for as long as nothing read `extend_len`. The first boot
    that executed it died in 37 seconds (instr21, PP1 10:42:12): a ~17000-
    token drive prompt, one 512-row chunk on the wire, and

        ValueError: #631 PP proxy/batch mismatch: received hidden_states with
                    512 row(s) for a 1 batch of 16983 token(s)

    16983 is exactly `len(fill_ids) - prefix`. The disagreement had moved
    INSIDE the schedule: it named a pass no rank had run.

    `req.extend_range` is the executed geometry. `PrefillAdder` writes it via
    `set_extend_range` on every path that appends to `can_run_list`
    (schedule_policy.py:1719, :1743, :1762, :1202, :1428, and this feature's
    own :1287), and this function is called from scheduler.py AFTER that loop,
    over `can_run_list` itself -- so the value is always already there, and
    the producer needs no move. `prepare_for_extend` derives the tensor's row
    count from the same pair (`extend_range.start` == `len(prefix_indices)`,
    `extend_range.length`; schedule_batch.py:2261-2281), which is what makes
    reporting it -- rather than recomputing an equivalent -- the point.

    THE GUARD IS NOT RE-APPLIED ON THAT PATH, and the old comment claiming it
    was idempotent is now false. The admission loop clamps `prefix_indices` to
    the learned floor BEFORE `add_one_req` (scheduler.py), and
    `add_one_req`'s host load-back then GROWS it again
    (schedule_policy.py:1707-1717) -- so re-running `prefix_len_for` here
    would clamp a second time and move the difference into `extend_len`,
    manufacturing a third geometry nobody ran. The guard's job is upstream, on
    the request, where it belongs; this function's job is to report.

    `require_executed_geometry` (True from scheduler.py, the one production
    call site): a request with NO `extend_range` is a LOUD REFUSAL naming the
    rid, never a default. A silent 0 or a silent full length would be the
    instr21 defect in new clothing, and `None` is REACHABLE here --
    `Req.reset_for_retract` sets `extend_range = None`
    (schedule_batch.py:1588), which is how boot instr19 died at
    scheduler.py:5572 with `AttributeError: 'NoneType' object has no
    attribute 'end'`. Refusing here reaches that same state three frames
    earlier and names the request; the alternative is not "no refusal", it is
    the AttributeError we have already lost a boot to.

    WHEN THE None CASE AND THE RETRACTION CASE CO-OCCUR -- and they can, since
    `reset_for_retract` is ON the retraction path, which this rebuild makes
    rare but cannot eliminate (physical impossibility is real): the refusal
    fires FIRST, on the first rank, before any decision is sent. That is
    strictly the better order. The pass is voided everywhere by the ordinary
    mechanism (the first rank builds no batch, so `launched` is False and the
    empty decision admits nothing downstream), and no rank ever sees a
    geometry for a request whose geometry had been torn down.

    `require_executed_geometry=False` (the default) keeps the pre-#791-core
    arithmetic for a stand-in that carries no adder output -- the reqs in
    test_pp_admission_retry_livelock_630.py and
    test_pp_admission_prefix_indices_tensor_796.py, which exist to pin the
    guard and the #796 tensor read and never reach a real batch.
    """
    entries = []
    for req in reqs:
        executed = _executed_extent(req)
        if executed is not None:
            entries.append(
                PPAdmissionEntry(
                    rid=req.rid,
                    prefix_len=executed[0],
                    extend_len=executed[1],
                    admitted=True,
                )
            )
            continue
        if require_executed_geometry:
            raise PPScheduleRefused(
                f"#791 SCHEDULE UNBUILDABLE for rid={getattr(req, 'rid', '?')}: "
                f"this rank admitted the request but its extend_range is "
                f"{getattr(req, 'extend_range', None)!r}, so there is no "
                f"executed geometry to forward. Reporting a length derived "
                f"from anything else would name a pass no rank ran. "
                f"`reset_for_retract` is the known producer of this state "
                f"(schedule_batch.py:1588); boot instr19 met it one frame "
                f"later as an AttributeError at scheduler.py:5572."
            )
        # #796: `prefix_indices` is a TENSOR of KV-pool slot pointers, so it
        # must never reach a boolean context. `x or []` evaluates `bool(x)`,
        # and torch raises on both ends of the range that matters here: an
        # EMPTY tensor gives "Boolean value of Tensor with no values is
        # ambiguous" and a multi-element one gives the "more than one
        # element" variant. Only the single-element case would have gone
        # through silently -- so the original spelling was broken for very
        # nearly every request, and it took the boot that first got this far
        # to find out (PP0 aborted here on its first real prefill, boot
        # instr5, once the #796 send-handle fix stopped the ring wedging at
        # idle before any request was ever admitted).
        #
        # `len()` reads the shape only and does not synchronise, which is
        # what #790 requires of anything on this path; it is also the
        # spelling the rest of this feature already uses
        # (scheduler_pp_mixin.py:2652, :3508).
        prefix_indices = getattr(req, "prefix_indices", None)
        raw_prefix_len = 0 if prefix_indices is None else len(prefix_indices)
        raw_extend_len = getattr(req, "extend_input_len", None)
        if raw_extend_len is None:
            fill_ids = getattr(req, "full_untruncated_fill_ids", None)
            raw_extend_len = (
                max(0, len(fill_ids) - raw_prefix_len) if fill_ids is not None else 0
            )
        raw_prefix_len = int(raw_prefix_len)
        raw_extend_len = int(raw_extend_len)

        if guard is not None and pp_size > 1:
            told = guard.prefix_len_for(req.rid, raw_prefix_len)
        else:
            told = raw_prefix_len

        # told <= raw_prefix_len always (prefix_len_for only ever clamps
        # down): the shortfall moves from "prefix" to "extend", the total
        # (the only quantity the request's own token count constrains)
        # stays put.
        extend_len = raw_extend_len + (raw_prefix_len - told)

        entries.append(
            PPAdmissionEntry(
                rid=req.rid,
                prefix_len=told,
                extend_len=extend_len,
                admitted=True,
            )
        )
    return PPAdmissionDecision(mb_id=mb_id, entries=tuple(entries))


def reconcile_pp_admission_decision(
    decision: PPAdmissionDecision,
    local_match_lens: Dict[str, int],
    *,
    rank: int,
    pp_size: int,
    log: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, int], PPAdmissionDecision]:
    """A downstream rank's rank-local reconciliation of a received decision.

    Returns `(effective_prefix_len_by_rid, amended_decision)`:
      * `effective_prefix_len_by_rid` contains exactly the rids this rank may
        safely admit THIS pass, each mapped to the prefix length it must use
        (always `<= told`, and `<= this rank's own local match`). A retracted
        rid is simply ABSENT here -- callers must never default a missing rid
        to 0-is-safe-to-proceed-with; absence means "do not admit this pass".
      * `amended_decision` is what this rank forwards to the next stage: the
        same entries, except any newly-retracted rid has `admitted=False`,
        `retracted=True`, `retracted_by_rank=rank`. An already-retracted
        entry (set by an earlier rank) is passed through unchanged and does
        NOT get re-logged or re-evaluated -- see the "exactly one WARNING"
        pin in the paired test.

    `pp_size<=1`: pure identity pass-through (every `told` becomes
    `effective` unconditionally, no entry is ever retracted, nothing is
    logged) -- see DEFAULT PATH above.

    Raises nothing. An unhonourable entry is data, not an exception -- see
    the module docstring's "TWO FAILURE SHAPES" section for why a raise here
    would turn an ordinary, expected cache-topology fact (a downstream rank's
    cache is colder than PP0's) into a crash on every such admission.
    """
    log = log or logger
    if pp_size <= 1:
        effective = {e.rid: e.prefix_len for e in decision.entries if e.admitted}
        return effective, decision

    effective: Dict[str, int] = {}
    amended: List[PPAdmissionEntry] = []
    for entry in decision.entries:
        if not entry.admitted or entry.retracted:
            # Already excluded upstream (by PP0's own verdict, or by an
            # earlier rank's shortfall). Pass through verbatim: do not
            # re-derive, do not re-log, do not resurrect it.
            amended.append(entry)
            continue

        raw_local = local_match_lens.get(entry.rid, UNKNOWN_MATCH)
        local = int(raw_local)
        told = entry.prefix_len
        unknown = local == UNKNOWN_MATCH

        if told <= 0:
            # #944 A ZERO OFFER DEMANDS NOTHING, so no lookup result -- not
            # even a failed one -- can make it unhonourable. The rank is being
            # asked to reuse no prefix at all and to compute the request from
            # its first token; that is executable whether or not the rank has
            # ever heard of the rid.
            #
            # BEFORE THE SENTINEL THIS FELL OUT OF THE ARITHMETIC and needed no
            # branch: a miss answered 0 and `0 >= 0` held. `UNKNOWN_MATCH` is
            # -1, so it stops falling out, and leaving it implicit retracts the
            # FIRST, congruent round of every request -- strictly worse than
            # the defect being fixed here, and measured as exactly that by
            # `test_the_park_alone_leaves_a_flat_livelock`.
            #
            # It is also load-bearing for the bound: `UNRESOLVED_DEFER_CAP`'s
            # escape offers told=0 precisely because this branch honours it
            # without a measurement. Remove this and the cap stops terminating.
            #
            # A NO-OP for every rank that DID resolve (local >= 0 >= told), so
            # the known path below is unchanged, not merely equivalent.
            effective[entry.rid] = told
            amended.append(entry)
            continue

        if local >= told:
            # SAFE: truncate any extra local reuse to `told`. Same slack
            # trade #616g already makes on the TP axis -- take it.
            effective[entry.rid] = told
            amended.append(entry)
            continue

        if unknown:
            # #944: A MISS IS NOT A ZERO, AND IT IS NOT A VERDICT EITHER.
            #
            # The producer could not locate this request in any of the places it
            # knows to look. That is a statement about the LOOKUP, not about
            # this rank's cache, and the two have opposite correct responses:
            # a genuine shortfall must void the pass, whereas a miss on a
            # request this rank is very likely holding must not.
            #
            # Reported distinctly -- in the log AND in a field of its own on
            # the wire -- so the two populations can never again be read as
            # one, which is the whole reason #797c and #798 each looked like a
            # fresh defect instead of the same one twice.
            #
            # THE GROUP DEFERS, NOT THIS RANK. Excluding the rid from
            # `effective` and retracting the entry puts the pass through the
            # existing #797 void, and that void is already group-uniform:
            # `pp_pass_should_void` ORs the incoming flag and never clears it,
            # so a retraction anywhere on the ring stops every rank. This rank
            # therefore does not decide anything -- it reports. Whether the
            # group defers AGAIN or gives up is decided once, by PP0, from the
            # count `unresolved` feeds (`PPAdmissionCongruenceGuard`,
            # `UNRESOLVED_DEFER_CAP`). A defer that only one rank takes IS the
            # next divergence.
            log.warning(
                "#944 PP-ADMISSION UNRESOLVED prefix on rank %d: rid=%s "
                "told=%d local=UNKNOWN -- the reconcile could not find this "
                "request in the waiting queue, in chunked_req, in the slot's "
                "chunked req, or in the running batch. This is a LOOKUP MISS "
                "reported as such, NOT a measured zero: the rank may well hold "
                "the prefix. Excluded from this pass.",
                rank,
                entry.rid,
                told,
            )
            # #944 THE SENTINEL MUST NOT REACH THE WIRE, and a test caught
            # this: `observed_local` feeds `_learned_floor[rid]` (:293), which
            # clamps the NEXT round's `told`, and this field is documented at
            # :237 as "never zero". Writing -1 there makes the learned floor
            # -1 and changes the offer sequence -- which is exactly what
            # `test_the_park_alone_leaves_a_flat_livelock` went red on, on
            # BEHAVIOUR rather than on a number.
            #
            # The field is Optional[int], so None is the honest value: this
            # pass learned NOTHING about the rank's prefix, which is a
            # different statement from "it has -1 of it". The sentinel stays
            # where it belongs, inside the resolution map.
            #
            # Same discipline the sentinel exists for, one level up: a value's
            # meaning is a property of its READERS, and `observed_local` has
            # readers `local_match_lens` does not.
            amended.append(
                replace(
                    entry,
                    admitted=False,
                    retracted=True,
                    retracted_by_rank=rank,
                    observed_local=None,
                    unresolved=True,
                )
            )
            continue

        # UNSAFE and physically un-fixable this pass (see module docstring).
        # Fail loudly and boundedly: exactly one WARNING, never a raise, and
        # the request is excluded from `effective` rather than handed a
        # length it cannot honour. The BOOT and every OTHER request in this
        # decision are unaffected.
        log.warning(
            "#791 PP-ADMISSION unhonourable prefix on rank %d: rid=%s "
            "told=%d local=%d -- serving this request without prefix reuse "
            "on a later pass instead of corrupting the cross-stage tensor",
            rank,
            entry.rid,
            told,
            local,
        )
        amended.append(
            replace(
                entry,
                admitted=False,
                retracted=True,
                retracted_by_rank=rank,
                observed_local=local,
            )
        )
        # Deliberately absent from `effective`: see the docstring above on
        # why "missing" must mean "do not admit", not "assume 0 is safe".

    return effective, PPAdmissionDecision(mb_id=decision.mb_id, entries=tuple(amended))


def forwarded_schedule(
    decision: Optional[PPAdmissionDecision],
) -> Dict[str, Tuple[int, int]]:
    """#791 CORE: the pass geometry a downstream rank must EXECUTE.

    `rid -> (prefix_len, extend_len)` for exactly the entries that are still
    admitted and not retracted -- i.e. exactly the rids
    `reconcile_pp_admission_decision` put in `effective`, with the SECOND
    number that function drops on the floor.

    WHY THIS EXISTS AT ALL, AND IT IS NOT A NEW DATUM. `extend_len` has
    crossed the wire since #791's first commit (see `PPAdmissionEntry` above
    and the module docstring's "WHAT CROSSES THE WIRE"), and until this
    function nothing read it. `reconcile_pp_admission_decision` returns
    `Dict[rid, prefix_len]`, so a downstream rank received the chunk length
    the first rank committed and then RE-DERIVED its own from
    `PrefillAdder.add_one_req`'s rank-local `rem_chunk_tokens` /
    `rem_total_tokens` / host-load-back. That re-derivation is the boot
    instr20 crash, in full:

        PP0  verdict=ADMIT rid=6cbe2733 prefix_lens=0  chunked=1  ->  512 rows
        PP1  verdict=ADMIT rid=6cbe2733 prefix_lens=512 chunked=0  ->  333 tokens

    -- one 845-token prompt, two different splits of it, taken in the same
    second on the same forwarded decision. PP0 was told nothing new; PP1
    simply asked its own state a question the schedule had already answered.
    A HiCache prefetch had landed on PP1 (and PP2) between the retraction and
    the re-admit, `needs_host_load_back()` went true there and stayed false on
    PP0, and `add_one_req`'s load-back put the 512 prefix tokens back on a
    `prefix_indices` the admission loop had just clamped to the schedule's 0.

    So the fix is not to make the local derivation agree -- it is to stop
    deriving. A rank holding a forwarded schedule reads BOTH numbers off it.

    `None`, or a decision with nothing admitted (a #797 void), yields an empty
    mapping, which every consumer reads as "no schedule is being executed this
    pass" -- byte-identically to the behaviour that shipped before this
    function existed. Pure; no rank-local input of any kind.
    """
    if decision is None:
        return {}
    return {
        e.rid: (int(e.prefix_len), int(e.extend_len))
        for e in decision.entries
        if e.admitted and not e.retracted
    }


def order_batch_by_schedule(reqs: Sequence, schedule: Dict[str, Tuple[int, int]]):
    """#791 CORE: put this rank's batch into the FORWARDED order.

    ORDER IS GEOMETRY, and it is the one divergence every width check on this
    branch is blind to. `ScheduleBatch.prepare_for_extend` concatenates each
    request's tokens in `can_run_list` order (schedule_batch.py:2261-2262), so
    the same rid set in a different order gives the SAME row count and a
    different meaning for every row. `model_runner.py`'s `_hs.shape[0] !=
    _want` counts rows; the #757 stamp counts rows; neither can see a
    permutation. Only an identity can.

    The two orders are independently derived and there is no reason for them
    to agree: `can_run_list` follows THIS rank's `waiting_queue`, the schedule
    follows the first rank's, and the queues are fed by separate
    chain-forward arrivals.

    SORTED, NOT MERELY CHECKED. The decision is an ORDERED list --
    `build_pp_admission_decision` iterates the first rank's own
    `can_run_list`, so `decision.entries` IS that rank's batch order, and
    `forwarded_schedule` preserves it (dicts keep insertion order). A caller
    reaching this function has already proven membership identical, so the
    permutation is total and loses nothing; refusing would void a pass that
    is perfectly runnable once the rows line up.

    Returns a NEW list. An empty schedule returns the input order unchanged,
    which is the default path (no forwarded geometry) byte for byte.
    """
    if not schedule:
        return list(reqs)
    order = {rid: i for i, rid in enumerate(schedule)}
    return sorted(reqs, key=lambda req: order[req.rid])


def schedule_refusal_reason(
    *,
    rid: str,
    scheduled_prefix_len: int,
    scheduled_extend_len: int,
    local_prefix_len: int,
    local_fill_len: int,
) -> Optional[str]:
    """`None` when this rank can execute the forwarded geometry verbatim;
    otherwise the LOUD REFUSAL text, quoting the forwarded decision.

    THE THREE WAYS A FORWARDED GEOMETRY CAN BE LOCALLY IMPOSSIBLE, and none
    of them may be answered by adjusting the geometry:

      * the rank's own prefix is not the scheduled one. The admission loop
        clamps `prefix_indices` to the schedule before the adder ever sees the
        request, and `reconcile_pp_admission_decision` retracts anything it
        cannot reach, so this is unreachable on a healthy pass -- which is
        exactly why it is checked. A silent re-clamp here would be the same
        local narrowing this module exists to abolish.
      * the request does not have the tokens. `prefix + extend` past
        `full_untruncated_fill_ids` cannot be filled from anything this rank
        holds.
      * a NEGATIVE extend. Not merely unrunnable -- unrepresentable, so it can
        only mean a malformed decision.

    A ZERO EXTEND IS EXECUTABLE, and the change of mind is #791-core's own
    doing. Before the producer reported the EXECUTED geometry, a zero could
    only have been a fabrication; now it is a faithful report of a first rank
    that ran zero rows for that request (a chunk landing exactly on its last
    token, schedule_policy.py:1420-1421). Refusing it would void a pass the
    upstream ran perfectly well -- and substituting a length of this rank's
    own is precisely what this module abolishes. See `_executed_extent`.

    The caller turns this into a voided pass (the #797 path), never into a
    narrower batch. Pure: five integers in, a string or None out.
    """
    if scheduled_extend_len < 0:
        return (
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={rid}: the decision "
            f"names extend_len={scheduled_extend_len}, which is not a length. "
            f"A downstream rank may not substitute one of its own."
        )
    if local_prefix_len != scheduled_prefix_len:
        return (
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={rid}: the decision "
            f"names prefix_len={scheduled_prefix_len}, this rank holds "
            f"{local_prefix_len}. The batch geometry is the upstream's to "
            f"decide; narrowing or widening it here is what pairs one "
            f"microbatch's hidden states with another's metadata."
        )
    if scheduled_prefix_len + scheduled_extend_len > local_fill_len:
        return (
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={rid}: the decision "
            f"names prefix_len={scheduled_prefix_len} + "
            f"extend_len={scheduled_extend_len}, past this rank's "
            f"{local_fill_len} fill token(s). The two ranks disagree about the "
            f"request itself, not merely about its cache."
        )
    return None


def entries_retracted_by_rank(
    decision: Optional[PPAdmissionDecision], rank: int
) -> Tuple[PPAdmissionEntry, ...]:
    """The entries THIS rank newly retracted from a decision it received.

    #791c. `reconcile_pp_admission_decision` above is the only writer of
    `retracted_by_rank`, and it writes it ONLY on the hop that first finds
    the prefix unhonourable -- an entry another rank already retracted takes
    the pass-through branch and keeps that rank's number. So a non-empty
    result here means exactly one thing, with no inference: THIS RANK'S BATCH
    FOR THIS PASS IS NARROWER THAN ITS UPSTREAM'S, because the upstream built
    and forwarded its own batch from the decision as it stood BEFORE this
    retraction, and nothing can amend a batch that is already in flight.

    That is the whole content of the 2026-08-21 07:12:49 mispair (boot
    instr17): PP0 admitted two requests (126 extend tokens), PP1 found
    rid=5e744c29's told=16896 prefix unhonourable against its own local=0,
    excluded it from `effective`, and built a one-request batch of 22 tokens
    -- then paired PP0's 126 rows of hidden states with it. Every identity
    the proxy carries (slot, sequence, flip epoch) was CORRECT: the message
    was the current pass's, from the current ring, in the current epoch. Only
    the WIDTH disagreed, and the width disagreed because of this function's
    subject.

    PURE, AND KEPT IN THIS MODULE ON PURPOSE. The consumer is the PP proxy
    receive guard in scheduler_pp_mixin.py, but the fact is a property of a
    decision, and this module already owns every other such property
    (`congruent_rids` below is the same shape). `None` -- a caller with no
    decision recorded for the slot -- yields an empty tuple, which every
    consumer reads as "nothing known against this pass", i.e. exactly the
    behaviour that shipped before this function existed.
    """
    if decision is None:
        return ()
    return tuple(
        e
        for e in decision.entries
        if e.retracted
        and e.retracted_by_rank is not None
        and int(e.retracted_by_rank) == int(rank)
    )


def void_pp_admission_decision(
    decision: PPAdmissionDecision,
) -> PPAdmissionDecision:
    """#797: drop every STILL-ADMITTED entry, leaving the retractions alone.

    THE PASS, NOT THE REQUEST, IS WHAT A RETRACTION COSTS. `reconcile_pp_
    admission_decision` above narrows a decision by ONE rid and leaves the
    rest admitted, which is correct as a statement about that rid and wrong
    as a plan for the pass: the upstream has already built and launched a
    batch containing BOTH rids, so a batch of "the rest" is a strict subset
    of the one whose hidden states are already on the wire. Boot instr17
    computed exactly that subset -- 22 tokens against 126 rows -- and boots
    instr15/16/17 between them logged 661, 1651 and 1718 of these narrowings
    while the width check caught ONE. The other ~4000 were same-width
    pairings, which are silent wrong output. See `entries_retracted_by_rank`.

    So the retracting rank drops the whole pass instead, and this is the
    membership half of that: every entry that is still admitted becomes
    `admitted=False`, which `reconcile_pp_admission_decision`'s own
    pass-through branch then carries to every rank after this one, and which
    scheduler.py's admission loop reads as "not named this pass" -- the
    existing requeue-for-free path, not a new one.

    admitted=False AND retracted=False, WHICH IS THE THIRD STATE AND NOT AN
    OVERSIGHT. `PPAdmissionCongruenceGuard.record_return_trip` reads exactly
    these two flags: `retracted` teaches a floor, `admitted` clears one, and
    neither ("excluded before this module ever saw it") leaves the rid
    untouched. A collaterally-dropped rid must land in that third state: it
    suffered no shortfall, so it must teach no floor, and it was never
    served, so it must clear none either. Marking it `retracted` would clamp
    a prefix that was perfectly honourable; leaving it `admitted` would clear
    a floor another rank had just paid to learn.

    Pure and idempotent: a decision with nothing left to drop is returned
    with equal content, and a `retracted` entry is never touched.
    """
    if not any(e.admitted and not e.retracted for e in decision.entries):
        return decision
    return PPAdmissionDecision(
        mb_id=decision.mb_id,
        entries=tuple(
            replace(e, admitted=False) if e.admitted and not e.retracted else e
            for e in decision.entries
        ),
    )


def congruent_rids(decisions: Iterable[PPAdmissionDecision]) -> bool:
    """True iff every decision in `decisions` agrees on membership AND on
    every admitted rid's `(prefix_len, extend_len)`. Test/diagnostic helper:
    the property this whole module exists to make true across PP ranks."""
    decisions = list(decisions)
    if len(decisions) <= 1:
        return True
    reference = {
        e.rid: (e.prefix_len, e.extend_len)
        for e in decisions[0].entries
        if e.admitted and not e.retracted
    }
    for other in decisions[1:]:
        seen = {
            e.rid: (e.prefix_len, e.extend_len)
            for e in other.entries
            if e.admitted and not e.retracted
        }
        if seen != reference:
            return False
    return True


# How many consecutive VACUOUS admission verdicts the #788 trace swallows
# before it emits a roll-up line. Chosen against the measured idle rate of
# boot instr10 (146023 trace lines per rank), which puts a roll-up roughly
# every minute or two of idling: frequent enough that a reader never has to
# wonder whether the instrument is still alive, rare enough that three hours
# of idling costs kilobytes instead of the 5.9 GB boot instr11 wrote.
#
# It is a COUNT, never a duration. A wall-clock cadence would make the
# emission points rank-local and destroy the congruence property below.
PP_ADMISSION_VACUOUS_ROLLUP_EVERY = 1024


def pp_admission_verdict_is_vacuous(
    n_reqs: int, queue: int, running: int, chunked: int
) -> bool:
    """True iff this admission verdict carries no congruence evidence.

    A pass with nothing admitted, nothing queued, nothing running and nothing
    chunked was taken over an empty scheduler. Two ranks cannot disagree
    about an empty scheduler, so the line cannot show the divergence the #788
    trace exists to catch -- and at idle it is essentially every line: boot
    instr11 wrote 5.9 GB in three hours, of which the informative fraction
    was a rounding error.

    THE PROPERTY THIS FUNCTION MUST KEEP, and the reason it takes four plain
    ints and nothing else. The acceptance gate for rank congruence
    (evidence-665-f1/verdict_790.sh, step 3) diffs the emitted trace payloads
    across PP0/PP1/PP2 and requires ONE group per event. That diff is only
    meaningful if every rank decides to speak or stay silent from data every
    rank has identically. So the predicate is a pure function of the
    CONGRUENT payload fields -- `n_reqs`, `queue`, `running`, `chunked` --
    and must never consult anything rank-local: not wall-clock time, not a
    per-rank log volume, not a random sample, and not `avail`/`evictable`
    (which are this rank's own pool accounting and legitimately differ). If
    rank 0 suppressed a line rank 1 emitted, the gate would report a
    divergence that never happened, and an instrument that manufactures
    false positives is worse than no instrument.
    """
    return n_reqs == 0 and queue == 0 and running == 0 and chunked == 0
