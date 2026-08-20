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


def build_pp_admission_decision(
    mb_id: int,
    reqs: Sequence,
    *,
    pp_size: int,
) -> PPAdmissionDecision:
    """PP0's (or, under `pp_size<=1`, the only rank's) committed decision.

    Reads `req.rid`, `len(req.prefix_indices)`, and the request's own extend
    length (`req.extend_input_len` if present, else derived from
    `full_untruncated_fill_ids` minus the prefix) -- all values this rank
    already computed locally while building its batch. Emits only integers;
    `prefix_indices` itself never leaves this function.

    `pp_size<=1`: still returns a decision (there is no reason not to -- it
    is cheap and harmless), but nothing downstream is obligated to consult
    it, and `reconcile_pp_admission_decision` is a pure pass-through in that
    regime regardless of what this returns. See DEFAULT PATH above.
    """
    entries = []
    for req in reqs:
        prefix_len = len(getattr(req, "prefix_indices", None) or [])
        extend_len = getattr(req, "extend_input_len", None)
        if extend_len is None:
            fill_ids = getattr(req, "full_untruncated_fill_ids", None)
            extend_len = max(0, len(fill_ids) - prefix_len) if fill_ids is not None else 0
        entries.append(
            PPAdmissionEntry(
                rid=req.rid,
                prefix_len=int(prefix_len),
                extend_len=int(extend_len),
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
            replace(entry, admitted=False, retracted=True, retracted_by_rank=rank)
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
