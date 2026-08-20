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

    def __init__(self) -> None:
        self._learned_floor: Dict[str, int] = {}

    def prefix_len_for(self, rid: str, candidate_prefix_len: int) -> int:
        """What PP0 should tell downstream ranks for `rid` this pass.

        `candidate_prefix_len` is PP0's own fresh local match -- what
        `build_pp_admission_decision` would tell without this guard.
        Clamped to the learned floor if one is outstanding for `rid`;
        returned unchanged otherwise (a rid with no retraction history is
        not constrained by this guard at all).
        """
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
                observed = entry.observed_local
                if observed is None:
                    # Defensive: a retraction without an observed value
                    # cannot safely tighten a floor -- leave any existing
                    # floor as-is rather than guess. Every retraction this
                    # module itself produces always sets observed_local, so
                    # this branch guards a malformed/foreign decision, not
                    # an expected path.
                    continue
                existing = self._learned_floor.get(entry.rid)
                self._learned_floor[entry.rid] = (
                    observed if existing is None else min(existing, observed)
                )
            elif entry.admitted:
                self._learned_floor.pop(entry.rid, None)

    def outstanding_rids(self) -> Tuple[str, ...]:
        """Diagnostic/test hook: rids currently carrying a learned floor."""
        return tuple(self._learned_floor)


def build_pp_admission_decision(
    mb_id: int,
    reqs: Sequence,
    *,
    pp_size: int,
    guard: Optional[PPAdmissionCongruenceGuard] = None,
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
    """
    entries = []
    for req in reqs:
        raw_prefix_len = len(getattr(req, "prefix_indices", None) or [])
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

        local = int(local_match_lens.get(entry.rid, 0))
        told = entry.prefix_len

        if local >= told:
            # SAFE: truncate any extra local reuse to `told`. Same slack
            # trade #616g already makes on the TP axis -- take it.
            effective[entry.rid] = told
            amended.append(entry)
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
