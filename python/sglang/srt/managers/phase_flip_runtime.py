# SPDX-License-Identifier: Apache-2.0
"""Phase-flip KV mover for #631 Route A (PP=3 prefill <-> TP=3 decode).

Moves the full-attention paged KV between the PP layout (stage owns whole
layers, pool row = global slot id) and the TP layout (rank owns tokens
under the weighted DCP vector, pool row = compact row), on the #297
envelope, carried over LITERALLY from ``managers/kv_reshard.py``:

* CONSENSUS FIRST, BYTES SECOND: every ``consensus_interval``-th round --
  gated by the replicated round counter, never local state -- every rank
  enters ONE bounded MIN-reduction with
  ``(armed, ready, epoch, direction, config_fp, vector...)``. ``armed``
  and ``ready`` are MIN-semantics (skew is legal and uniformly resolves
  to "wait"); ``epoch``, ``direction`` (once armed), the layer-map/vector
  fingerprint (ALWAYS -- it is boot config, divergence is fatal armed or
  not) and the vector are equality-checked with the same loud
  :class:`KvReshardError` on every rank.
* PACK -> EXCHANGE -> CHECKSUM -> WRITE with the pool untouched through
  pack, exchange and checksum verification; only the write phase is the
  no-return region. Source and destination are DIFFERENT pools here (the
  PP pool and the TP pool coexist), so the #297 aliasing hazard cannot
  arise inside one buffer -- the write order (local first, then incoming,
  disjoint injective targets) is kept anyway.
* Pools are pre-sized at boot for BOTH layouts: no growth, no address
  change, no CUDA-graph recapture. Bounds are checked loudly before any
  byte moves.

Payload layout per (stage s, dcp rank r) pair, identical on both ends by
convention (a checksum trailer keeps it falsifiable at runtime): layer
ordinals ascending, slots ascending within a layer, K bytes then V; one
row list per pair, reused for every layer (token ownership is
layer-independent). The receiver derives the expected byte count from ITS
OWN pool's per-layer row width -- a sender whose row format diverges is a
loud size/checksum error, which is the runtime pin of the "PP and TP rows
are byte-compatible" claim.

Weights-arena refill and GDN state movement are separate steps of the
flip protocol (DESIGN_631 section 3.6); ``pre_cutover_fns`` is their
injection seam so the scheduler wiring can order them inside the same
no-return region.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.phase_flip_plan import (
    PP_TO_TP,
    TP_TO_PP,
    PhaseFlipTransition,
    build_phase_flip_transition,
    validate_layer_map,
)
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.model_executor.weights_arena import uint8_checksum
from sglang.srt.utils.common import ceil_align
from sglang.srt.managers.kv_reshard import (
    _CHECKSUM_BYTES,
    KvPoolView,
    _checksum,
    _encode,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP"


class PhaseFlipJoinTimeout(RuntimeError):
    """#631(c): the consensus round did not assemble within the bound.

    Caught inside ``on_round`` and turned into a loud abandonment; it never
    escapes to the event loop, because the flip is optional and the parked
    requests are not.
    """


PHASE_PP = "pp"
PHASE_TP = "tp"

_DIR_ID = {PP_TO_TP: 1, TP_TO_PP: 2}
_DIR_OF_PHASE = {PHASE_PP: PP_TO_TP, PHASE_TP: TP_TO_PP}

#: How long an ARMED flip may wait for a group-wide quiescent boundary
#: before it gives up -- seconds, wall clock, measured on whichever rank is
#: still unparked.
#:
#: An armed flip withholds new work so the in-flight state drains; that is
#: what makes the flip interposable BETWEEN a request's prefill and its
#: decode instead of only after every stream has finished. The cost is that
#: a rank which never reaches quiescence withholds work forever, and the
#: requests it is holding never resume. This deadline bounds that: when it
#: expires the FLIP is abandoned, loudly, and serving continues. The user's
#: requests are never aborted -- they are the thing being protected.
#:
#: 30 s is chosen against the legitimate worst case: a drain is a handful of
#: iterations plus, at most, the continuation of one already-half-written
#: chunked prefill (exempt from parking, because a chunk that stops mid-way
#: could never satisfy the quiescence predicate at all).
DEFAULT_PARK_DEADLINE_S = 30.0
# #631(c): how long a rank waits INSIDE the flip's consensus reduction for
# the rest of the group. Generous on purpose -- a peer draining a long
# prefill is normal and must not trip it. This is a wedge breaker, not a
# latency control: without it, a rank that enters and finds no peers waits
# for ever, because every other flip deadline is checked BEFORE entry.
DEFAULT_JOIN_DEADLINE_S = 45.0
# #631 option 2(b): how long an armed rank polls for the whole group to
# reach the flip entry before giving up. Generous: a peer finishing a long
# prefill chunk is normal. This bound is PRE-ENTRY and therefore legal --
# abandoning a poll costs nothing, whereas abandoning an ENTERED
# all_reduce aborts every rank (see the withdrawn (c)).
DEFAULT_PRESENCE_DEADLINE_S = 60.0
# #631: how long the armed spin sleeps between flag reads. Small enough
# that assembly is prompt at idle, large enough not to burn a core while
# a peer finishes draining. The spin touches no channel, so this is a
# pacing knob only -- it cannot affect correctness, just latency.
DEFAULT_PRESENCE_POLL_INTERVAL_S = 0.005

#: Env override for the above. Non-positive disables the deadline, which
#: restores the old unbounded wait -- available deliberately for debugging a
#: slow drain, and named so a reader sees that "no deadline" is a choice.
ENV_PARK_DEADLINE = "SGLANG_PHASE_FLIP_PARK_DEADLINE_S"


def park_deadline_s() -> float:
    try:
        return float(os.environ.get(ENV_PARK_DEADLINE, DEFAULT_PARK_DEADLINE_S))
    except ValueError:
        return DEFAULT_PARK_DEADLINE_S


_PHASE_AFTER = {PP_TO_TP: PHASE_TP, TP_TO_PP: PHASE_PP}


def _config_fingerprint(
    layer_map: Tuple[Tuple[int, ...], ...], vector: Tuple[int, ...]
) -> int:
    """31-bit stable fingerprint of the replicated flip configuration.

    Folded into every consensus payload and equality-checked ALWAYS: a
    rank booted with a different layer map or vector must die loudly at
    the first consensus round, not at the first wrong byte."""
    acc = 0
    for s, layers in enumerate(layer_map):
        for f in layers:
            acc = (acc * 1_000_003 + (s + 1) * 8191 + f * 131) % (2**31 - 1)
    for v in vector:
        acc = (acc * 1_000_003 + v * 65_537) % (2**31 - 1)
    return acc


class AbortDeferralWindow:
    """Pin 4 (DESIGN_631 3.6a): client disconnects during a flip.

    A parked request whose client vanishes mid-flip must not mutate the
    live slot set between the plan derivation and the write phase -- an
    abort applied on one rank before its peers diverges the replicated
    live set, which the runtime can only answer with a LOUD size/desync
    error (clean abort of the attempt, but a lost flip). Deferral makes
    the window airtight instead: while a flip is pending or executing,
    abort work is QUEUED; it drains in the first round after cutover (or
    after disarm). The queue preserves order. Slots are never leaked --
    the deferred abort frees them under the NEW layout, which is
    equivalent by the global-slot-id property (metadata never rewrites).
    """

    def __init__(self):
        self._deferred: List[Callable[[], None]] = []
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def deferred_count(self) -> int:
        return len(self._deferred)

    def activate(self) -> None:
        self._active = True

    def submit(self, work: Callable[[], None]) -> bool:
        """Run ``work`` now (returns False) or defer it (returns True)."""
        if self._active:
            self._deferred.append(work)
            return True
        work()
        return False

    def deactivate_and_drain(self) -> int:
        """Close the window and run everything deferred, in order."""
        self._active = False
        drained = 0
        while self._deferred:
            work = self._deferred.pop(0)
            work()
            drained += 1
        return drained


def _flip_spec_algo(scheduler):
    """The algorithm the TP DECODE phase will run, or a NONE sentinel.

    ``scheduler.spec_algorithm`` is NONE in the PP phase by design -- the
    configured algorithm is parked in ``flip_spec_algorithm`` at boot and
    swapped in at the cutover -- so the PP-phase question "will the phase
    I am about to enter speculate?" has to read the parked one.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = getattr(scheduler, "flip_spec_algorithm", None)
    if algo is None:
        return SpeculativeAlgorithm.from_string(None)
    return algo


def _harvest(scheduler):
    from sglang.srt.managers.phase_flip_resident_carry import (
        harvest_resident_batches,
    )

    try:
        return harvest_resident_batches(scheduler)
    except Exception:  # noqa: BLE001 - a readiness probe never breaks a flip
        return []


def build_flip_quiescence_fn(scheduler) -> Callable[[], bool]:
    """The flip ready predicate (DESIGN_631 3.5) -- NOT #297 fully-idle.

    True when no forward is in flight and no chunk is half-written, with
    requests PARKED: no partial chunk, previous batch drained, overlap
    result queue empty, PP micro-batches drained. Deliberately does NOT
    require an empty waiting queue or an empty running batch -- the flip
    exists to run between a request's prefill and its decode."""

    def _why_not() -> Optional[str]:
        """The reason this rank is not quiescent, or None if it is.

        SEPARATED OUT so a rank can SAY why it is holding. Defect I was
        diagnosed from three py-spy stacks because "ready=0" is all the
        log ever carried, and the interesting question -- WHICH rank is
        holding WHAT, and is it the same thing every epoch -- was
        unanswerable from the log alone. It costs one string on a path
        that only runs while a flip is armed.
        """
        # #631 DEFECT O, and it is the SAME CATEGORY ERROR for the third
        # time in this function: a term that refuses because WORK EXISTS
        # rather than because work is IN FLIGHT.
        #
        # This used to be "chunked_req is not None -> not quiescent". A
        # long chunked prefill occupies that attribute for its ENTIRE
        # duration, so a flip armed BECAUSE of pending prefill could only
        # commit once the prefill it was meant to accelerate had already
        # finished. Measured 2026-08-09 04:23:35-04:23:54Z: armed
        # "pending prefill 26624 tok > N", NOT QUIESCENT "a chunked
        # prefill is half-written" on all three ranks for 19 s, cutover to
        # pp at 04:23:54 -- at which instant the policy immediately armed
        # pp_to_tp again because prefill was "down to 0 tok". The whole
        # 32768-token prefill ran in the TP layout at 1525 tok/s against
        # 4553 tok/s measured in PP, and the instance paid two cutovers
        # for nothing.
        #
        # BETWEEN CHUNKS IS A SETTLED BOUNDARY. get_next_batch_to_run
        # caches ("stashes") the computed prefix every round, so a chunked
        # request that is not mid-forward holds committed KV and a fully
        # accounted extend_range -- exactly the state the carry can move.
        # What must be quiet is the FORWARD, which ``mbs`` and
        # ``result_queue`` below already answer.
        #
        # This is only sound because _live_reqs now enumerates
        # scheduler.chunked_req: the request is in NO batch, so without
        # that the flip would move the layout out from under a request
        # whose KV stayed behind. The two changes are one change.
        chunked = getattr(scheduler, "chunked_req", None)
        if chunked is not None and getattr(chunked, "req_pool_idx", None) is None:
            return "a chunked prefill has no pool row yet (mid-admission)"
        # #631 DEFECT L, and it is the SAME CATEGORY ERROR as the
        # _pp_microbatches_drained one two paragraphs down -- found the
        # same way, by a leg that could never commit.
        #
        # This used to read "last_batch is not empty". Under
        # event_loop_normal (the TP decode phase) the result is processed
        # in the SAME iteration as the forward and ``last_batch = batch``
        # is set afterwards, so at the hook a non-empty last_batch means
        # "requests are resident", NOT "work is in flight". A decoding
        # request makes it non-empty on every iteration for ever, so
        # tp_to_pp could never reach a quiescent boundary: armed at
        # 03:11:22Z and 03:12:52Z on all three ranks, "NOT QUIESCENT:
        # last_batch is not empty (1 req(s) visible)", abandoned at the
        # park deadline both times, while pp_to_tp had just carried the
        # same request across the other way without trouble.
        #
        # The genuine evidence of pending work is already checked: the
        # overlap loop's result_queue below, and the PP loop's in-flight
        # ``mbs`` further down. What remains -- and it is narrower -- is
        # whether every live request is reachable through the handle the
        # CARRY harvests. Right after a prefill the new requests are still
        # only in last_batch and are merged into the running batch by the
        # next get_next_batch_to_run: a real reason to wait, self-clearing
        # in one iteration.
        from sglang.srt.managers.phase_flip_resident_carry import (
            orphan_resident_reqs,
        )

        orphans = orphan_resident_reqs(scheduler)
        if orphans:
            return (
                f"{len(orphans)} request(s) are still only in "
                f"last_batch/last_mbs ({orphans[:4]}) and not yet merged "
                f"into the resident set the carry harvests"
            )
        result_queue = getattr(scheduler, "result_queue", None)
        if result_queue is not None and len(result_queue) > 0:
            return f"result_queue holds {len(result_queue)} result(s)"
        # IN-FLIGHT MICROBATCHES ONLY -- deliberately NOT
        # Scheduler._pp_microbatches_drained, which this used to call.
        #
        # That helper is the FULLY-IDLE predicate (is_fully_idle, on_idle)
        # and it also requires every ``running_mbs`` slot to be empty.
        # ``running_mbs`` is the RESIDENT DECODE SET, not work in flight:
        # it holds the requests currently being decoded and empties only
        # when they FINISH. Borrowing it made this function contradict its
        # own contract two lines up ("does NOT require an empty running
        # batch") and, worse, contradict the policy that drives it: the
        # policy arms pp_to_tp precisely BECAUSE requests are decoding, so
        # the arming condition and the quiescence condition could never
        # hold at the same time and every automatic flip abandoned at the
        # park deadline.
        #
        # MEASURED, 2026-08-09 01:29:50Z, POLICY=auto with one request
        # decoding: "NOT QUIESCENT: PP microbatches not drained (live mb
        # slots [], running_mbs slots [0])" on ranks 0 and 1 -- nothing in
        # flight, the resident decode set alone holding the flip. The gate
        # assembled, all three ranks entered the reduction and agreed to
        # abandon on the park deadline, with ready=0 everywhere.
        #
        # Carrying a resident decode set across the flip is what the rest
        # of the design already assumes: build_flip_live_slots_fn exists
        # to move exactly those requests' KV rows ("the flip runs with
        # requests parked, whose KV rows live in req_to_token"). What must
        # be quiet is the PIPELINE -- no forward in flight, no half-written
        # chunk -- which is what ``mbs`` answers.
        mbs = getattr(scheduler, "mbs", None)
        if mbs is not None:
            live = [
                i for i, mb in enumerate(mbs) if mb is not None and not mb.is_empty()
            ]
            if live:
                return f"PP microbatches still in flight (mb slots {live})"
        # #631: SPECULATION AND THE CARRIED REQUEST -- honest by waiting.
        #
        # A request that prefills in the PP phase has NO DRAFT STATE: the
        # PP phase carries no draft worker by design, so nothing ever ran
        # the draft_extend that a spec instance gives a request after its
        # target extend. Carrying such a request into a SPECULATING TP
        # phase kills the instance one pass later -- measured 03:32:14Z on
        # all three ranks:
        #
        #   eagle_worker_v2.draft -> eagle_draft_cuda_graph_runner.execute
        #   -> foreach_copy: output with shape [1, 1] doesn't match the
        #      broadcast shape [0, 1]        -> SIGQUIT
        #
        # the draft input having no rows for the carried request. Until
        # the draft state is built at the flip (HANDOFF_656 v3 section 3),
        # this rank is simply NOT READY to flip into speculation while
        # anything is resident.
        #
        # WAITING, not refusing at arm time. A rank-local refusal inside
        # arm() would let one rank decline while its peers armed, and
        # diverging epochs is corpse H -- fatal. Readiness runs through
        # the bounded park/abandon machinery, which is unanimous by
        # construction, and it has the right meaning anyway: the flip
        # happens as soon as nothing has to survive it, which is exactly
        # the regime every flip before this one ran in.
        runtime = getattr(scheduler, "phase_flip_runtime", None)
        pending = getattr(runtime, "pending", None) if runtime is not None else None
        if pending == PP_TO_TP and not _flip_spec_algo(scheduler).is_none():
            n_resident = sum(
                len(getattr(b, "reqs", []) or [])
                for b in _harvest(scheduler)
            )
            if n_resident:
                return (
                    f"{n_resident} resident request(s) would enter a "
                    f"SPECULATING TP phase with no draft state (they "
                    f"prefilled in the PP phase, which has no draft "
                    f"worker); waiting for them to finish rather than "
                    f"crashing the draft graph runner"
                )
        return None

    def _ready() -> bool:
        return _why_not() is None

    _ready.why_not = _why_not
    return _ready


def build_flip_live_slots_fn(scheduler) -> Callable[[], torch.Tensor]:
    """Live slots = radix tree values UNION parked requests' rows.

    #297 Stage A enumerates the tree only, correct at fully-idle. The
    flip runs with requests parked, whose KV rows live in req_to_token
    and are NOT all in the tree yet -- omitting them would silently drop
    the freshest prefix KV at the flip (DESIGN_631 3.5). Replicated: the
    tree and the batch state are rank-replicated between rounds."""

    def _live() -> torch.Tensor:
        parts: List[torch.Tensor] = []
        values = scheduler.tree_cache.all_values_flatten()
        if values is not None and values.numel():
            parts.append(values.detach().to("cpu", torch.int64))
        # ALL RESIDENT SLOTS, not scheduler.running_batch (#631 J). That
        # attribute is the CURRENT microbatch slot under event_loop_pp, and
        # the flip's hook fires at the end of an arbitrary slot iteration,
        # so reading it sampled an empty slot while a request sat resident
        # in another one -- and that request's rows were then never moved.
        # See _live_reqs for the measurement.
        #
        # The ROW EXTENT below is still req.seqlen, deliberately: the
        # allocator-owned extent is kv_allocated_len and the two differ
        # under #486's spec reserve, but that delta has not yet been
        # measured on a flip where a resident request was actually
        # enumerated (it could not be -- none ever was). _probe_allocated_
        # extent reports it every flip; change this only on that evidence.
        reqs = _live_reqs(scheduler)
        req_to_token = scheduler.req_to_token_pool.req_to_token
        for req in reqs:
            n = int(req.seqlen)
            if n <= 0:
                continue
            rows = req_to_token[req.req_pool_idx, :n]
            parts.append(rows.detach().to("cpu", torch.int64))
        _probe_allocated_extent(scheduler, reqs)
        if not parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.unique(torch.cat(parts))

    return _live


def _live_reqs(scheduler) -> List:
    """Every request RESIDENT on this rank, across ALL microbatch slots.

    SLOT SCOPE IS THE DEFECT THIS EXISTS FOR (#631 J, measured 2026-08-09
    02:21:03Z). Under ``event_loop_pp``, ``scheduler.running_batch`` and
    ``scheduler.last_batch`` are rebound to ``running_mbs[mb_id]`` and
    ``last_mbs[mb_id]`` at the TOP of every slot iteration. They therefore
    describe ONE microbatch slot -- whichever slot's iteration happens to
    be running -- and NOT the rank's resident set. The flip's round hook
    fires at the END of a slot iteration, so reading ``running_batch``
    there samples an arbitrary slot, and an empty one is indistinguishable
    from "no requests resident".

    Measured at a real cutover:

        at-arm       cur_slot_reqs=1 resident_reqs=1 resident_slots=[1]
        pre-cutover  cur_slot_reqs=0 resident_reqs=1 resident_slots=[1]

    The request was resident in slot 1 throughout; the hook simply ran for
    a different, empty slot. Enumerating from ``running_batch`` alone
    therefore missed its rows entirely.

    THIS IS NOT MERELY AN ACCOUNTING BUG. Rows that are not enumerated are
    not MOVED, so the resident request's freshest KV is left behind in the
    source pool and never written into the destination layout. The leak
    detector notices the arithmetic; the request's CONTEXT would simply be
    wrong, silently. That is the failure class this feature must never
    ship.

    ``running_mbs`` is the per-slot resident set and is the authority
    here; ``running_batch``/``last_batch`` are unioned in for the non-PP
    event loop, where ``running_mbs`` does not exist. Deduplicated by
    identity because the same Req object appears in several of these.
    """
    seen = set()
    out: List = []

    def _take(batch) -> None:
        for req in list(getattr(batch, "reqs", []) or []) if batch else []:
            if id(req) in seen:
                continue
            seen.add(id(req))
            out.append(req)

    for mb in getattr(scheduler, "running_mbs", []) or []:
        _take(mb)
    for name in ("running_batch", "last_batch"):
        _take(getattr(scheduler, name, None))
    # THE CHUNKED PREFILL IS RESIDENT AND IS IN NO BATCH (#631 defect O).
    # get_next_batch_to_run deliberately moves it out ("Move the chunked
    # request out of the batch so that we can merge only finished requests
    # to running_batch"), so every batch-based enumeration misses it --
    # while it holds committed KV for everything computed so far, and a
    # mamba slot. Enumerating it here is what lets the flip happen BETWEEN
    # chunks instead of waiting for the whole prefill to finish; without
    # it, relaxing the quiescence term would move the layout out from
    # under a request whose KV stayed behind, which is J.3 again.
    chunked = getattr(scheduler, "chunked_req", None)
    if chunked is not None and id(chunked) not in seen:
        seen.add(id(chunked))
        out.append(chunked)
    return out


def _probe_allocated_extent(scheduler, reqs) -> None:
    """#631 defect J: MEASURE the gap between what the allocator owns and
    what this enumeration covers. Does not change what is moved.

    VERIFY BEFORE FIXING. A wrong guess about which KV rows belong to a
    request does not fail loudly -- it silently moves the wrong bytes, or
    silently leaves the right ones behind, and the request's context is
    then quietly corrupt. So the delta is measured on a real flip before
    the enumeration is changed to close it.

    ``req.kv_allocated_len`` is the AUTHORITATIVE extent: the number of KV
    slots the allocator has handed this request, and precisely what the
    invariant checker charges to the pool (invariant_checker._check reads
    the same field, page-aligned). ``req.seqlen`` is
    ``len(origin_input_ids) + len(output_ids)`` -- a property of the
    SEQUENCE, not of the allocation, and under speculative decoding the
    two are structurally different: task #486 reserves W + L slots ahead
    of ``kv_committed_len`` every decode step, where W is the draft/verify
    write footprint (topk * num_steps, or num_draft_tokens) and L is the
    commit lag. On this rig's NEXTN config that reserve is several slots,
    NOT one -- so any fix built on "seqlen + 1" would be wrong in general
    even where it happens to balance the books on a quiet flip.
    """
    if not reqs:
        return
    try:
        page_size = int(getattr(scheduler.token_to_kv_pool_allocator, "page_size", 1))
        rows = []
        for req in reqs:
            alloc = int(getattr(req, "kv_allocated_len", -1))
            aligned = (
                ceil_align(alloc, page_size) if (page_size > 1 and alloc > 0) else alloc
            )
            rows.append(
                f"rid={getattr(req, 'rid', '?')} seqlen={int(req.seqlen)} "
                f"kv_allocated_len={alloc} aligned={aligned} "
                f"kv_committed_len={int(getattr(req, 'kv_committed_len', -1))} "
                f"cache_protected_len={int(getattr(req, 'cache_protected_len', -1))} "
                f"delta_vs_seqlen={aligned - int(req.seqlen)}"
            )
        logger.warning(
            "%s FLIP EXTENT PROBE (page_size=%d): %s. delta_vs_seqlen is the "
            "number of allocator-owned rows this enumeration does NOT move; "
            "nonzero means they are left owned by nobody in the destination "
            "stack, which is defect J.",
            LOG_PREFIX,
            page_size,
            " | ".join(rows),
        )
    except Exception as exc:  # noqa: BLE001 - a probe must never break a flip
        logger.warning("%s flip extent probe failed: %s", LOG_PREFIX, exc)


def flip_blocking_guards(scheduler) -> List[str]:
    """Features that refuse flip arming (DESIGN_631 3.7). Mirrors the
    #297 Stage-A guard shape, plus the #630 PP x disk-HiCache wedge."""
    guards: List[str] = []
    server_args = scheduler.server_args
    try:
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if scheduler.disaggregation_mode != DisaggregationMode.NULL:
            guards.append("PD disaggregation")
    except ImportError:
        pass
    if getattr(server_args, "enable_hierarchical_cache", False):
        guards.append(
            "hierarchical cache (#630: PP x disk HiCache wedges at warmup)"
        )
    if getattr(scheduler, "kv_session_offload", None) is not None:
        guards.append("kv-session-offload")
    if getattr(scheduler, "is_dual_group_lane", False) or getattr(
        server_args, "dual_group_lane", None
    ):
        guards.append("dual-group lane")
    if not hasattr(scheduler.tree_cache, "all_values_flatten"):
        guards.append(
            f"tree cache {type(scheduler.tree_cache).__name__} (no "
            f"all_values_flatten enumeration)"
        )
    return guards


class PhaseFlipLoopExit(Exception):
    """Control-flow signal: a flip COMMITTED this round; the current event
    loop must exit to the re-dispatching wrapper (dispatch_event_loop picks
    its loop ONCE from pp_size, so a changed topology needs a fresh
    dispatch). Raised by the scheduler's on_round hook AFTER
    PhaseFlipRuntime.on_round returned commit stats -- never from inside
    the runtime, whose epoch/phase bookkeeping must complete first. The
    quiescence predicate guarantees the loop holds no half-processed batch
    state when this propagates."""

    def __init__(self, direction: str):
        super().__init__(direction)
        self.direction = direction


def derive_pp_full_attn_layer_map(
    full_attention_layer_ids: Sequence[int],
    num_hidden_layers: int,
    pp_size: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Per-stage FULL-ATTENTION ORDINALS from the global layer geometry.

    A pure replicated function of (the model's global full-attention layer
    ids, the layer count, the PP stage split) -- every rank derives the
    same map, which the consensus fingerprint then pins at runtime. The
    stage split comes from get_pp_indices, the SAME function the PP model
    build used (env-uniform SGLANG_PP_LAYER_PARTITION included), so the
    map cannot drift from the actual stage windows.

    IMPORTANT SOURCE RULE: ``full_attention_layer_ids`` must be the
    UNMUTATED global list (e.g. from the TP stack's model_config, whose
    pp_size=1 adjust is the identity) -- the PP stack's model_config was
    rewritten in place to its stage-local slice
    (adjust_hybrid_swa_layers_for_pp)."""
    from sglang.srt.distributed.utils import get_pp_indices

    ids = [int(x) for x in full_attention_layer_ids]
    if ids != sorted(set(ids)):
        raise KvReshardError(
            f"full_attention_layer_ids must be strictly ascending, got {ids}"
        )
    if ids and not (0 <= ids[0] and ids[-1] < num_hidden_layers):
        raise KvReshardError(
            f"full_attention_layer_ids {ids} outside [0, {num_hidden_layers})"
        )
    bounds = [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]
    flat = [b for pair in bounds for b in pair]
    if flat != sorted(flat) or bounds[0][0] != 0 or bounds[-1][1] != num_hidden_layers:
        raise KvReshardError(
            f"PP stage bounds {bounds} do not partition [0, {num_hidden_layers})"
        )
    layer_map = []
    for start, end in bounds:
        layer_map.append(
            tuple(i for i, gid in enumerate(ids) if start <= gid < end)
        )
    covered = sorted(o for stage in layer_map for o in stage)
    if covered != list(range(len(ids))):
        raise KvReshardError(
            f"stage map {layer_map} does not cover every full-attention "
            f"ordinal exactly once (bounds {bounds}, ids {ids})"
        )
    return tuple(layer_map)


def build_gdn_flip_guard(scheduler) -> Callable[[str], None]:
    """5.3 PLACEHOLDER for the GDN state mover, honest by refusal.

    The full mover (layer-axis -> head-axis re-shard of conv/ssm state via
    MambaPool blobs, DESIGN_631 3.4) lands as slice 5.3b. Until then a
    flip with LIVE linear-attention state must refuse LOUDLY inside the
    no-return region's first step -- before any pool byte moved -- never
    proceed and silently truncate GDN state (the #212 Store-Route lesson).
    The 5.5 validation ladder's first rung (flip empty -> flip back) is
    exactly what this permits."""

    def _guard(direction: str) -> None:
        running = getattr(scheduler, "running_batch", None)
        reqs = list(getattr(running, "reqs", []) or []) if running else []
        if reqs:
            raise KvReshardError(
                f"{LOG_PREFIX} flip {direction} refused: {len(reqs)} live "
                f"request(s) hold GDN conv/ssm state and the GDN state "
                f"mover is not wired yet (slice 5.3b); flipping now would "
                f"silently truncate linear-attention state. Drain or wait."
            )

    return _guard


def build_production_flip_cutover(scheduler) -> Callable[[str], None]:
    """The cutover leg (DESIGN_631 3.6 step 5): everything the scheduler
    snapshotted from the boot topology is rebuilt for the target phase.
    Runs inside PhaseFlipRuntime._execute after KV/GDN/arena moves; the
    loop exit is raised LATER by the on_round hook (the runtime's
    epoch/phase bookkeeping must finish first)."""
    import dataclasses as _dc

    # Boot-phase snapshot for the return trip, taken ONCE at build time
    # (the scheduler's ps still holds the boot topology then).
    boot_ps = scheduler.ps
    boot_model_worker = scheduler.tp_worker

    def _cutover(direction: str) -> None:
        from sglang.srt.distributed import parallel_state as _ps
        from sglang.srt.distributed.utils import set_cp_token_ratios
        from sglang.srt.layers.dcp.owner import refresh_all_owner_bounds
        from sglang.srt.runtime_context import get_server_args

        stacks = scheduler.phase_flip_stacks
        tp_phase = direction == PP_TO_TP
        n = len(stacks.vector)
        world_rank = _ps.get_world_group().rank_in_group

        # The phase's speculation state, decided ONCE here because two
        # separate steps below need the same answer: the component rebuild
        # (4b) and the scheduler's own swap (7). Speculation belongs to the
        # TP DECODE phase (#631) -- the draft worker was built on the flip's
        # TP stack at boot and is armed with it; the PP phase carries none,
        # which is bit-for-bit the state of an instance without speculation.
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        want_draft = stacks.draft_worker if tp_phase else None
        want_spec_algo = (
            scheduler.flip_spec_algorithm
            if (tp_phase and want_draft is not None)
            else SpeculativeAlgorithm.from_string(None)
        )
        if tp_phase:
            # Mirrors the boot dispatch: with speculation the model worker
            # IS the draft worker, which drives the target through it.
            want_model_worker = (
                want_draft if want_draft is not None else stacks.tp_worker
            )
        else:
            want_model_worker = boot_model_worker

        # 1. Module-level group routing (forward collectives resolve
        # through the parallel_state getters; see phase_flip_boot).
        _ps.set_phase_flip_tp_active(tp_phase)

        # 2. Owner rule: the vector is boot-constant; refresh the bounds
        # consumers so the TP backends read the (re)installed vector.
        # This is a TOKEN-space quantity, so it must be the token vector,
        # not the weight shard vector. They are equal unless
        # SGLANG_UNEVEN_TOKEN_VECTOR overrides the token side; reinstalling
        # the weight vector here would leave the owner rule splitting rows
        # under a different vector than the pools were SIZED under, which
        # is an out-of-bounds slot id, not a slow path.
        set_cp_token_ratios(list(stacks.token_vector))
        refresh_all_owner_bounds()

        # 3. Scheduler topology snapshot (frozen dataclass -> new instance).
        if tp_phase:
            scheduler.ps = _dc.replace(
                boot_ps,
                tp_rank=world_rank,
                tp_size=n,
                pp_rank=0,
                pp_size=1,
                attn_tp_rank=world_rank,
                attn_tp_size=n,
            )
        else:
            scheduler.ps = boot_ps

        # 4. Cached group handles, re-derived through the ROUTED getters.
        scheduler.tp_group = _ps.get_tp_group()
        scheduler.tp_cpu_group = scheduler.tp_group.cpu_group
        scheduler.attn_tp_group = _ps.get_attn_tp_group()
        scheduler.attn_tp_cpu_group = scheduler.attn_tp_group.cpu_group
        scheduler.pp_group = _ps.get_pp_group()
        # dp-attention is a flip arming guard; the dp routing group is tp.
        scheduler.dp_tp_group = scheduler.tp_group

        # 4b. Scheduler COMPONENTS holding ps / group snapshots (found on
        # the first post-flip serving attempt, 2026-08-08): the request
        # receiver kept the boot ps and relayed requests PP-chain-style
        # while rank 0 ran TP semantics -- one rank in the pool-budget
        # all_reduce, another in the relay's point_to_point recv, wedge.
        # The output streamer's stale ps mis-gated the detokenizer send
        # (heartbeat loss). Both are plain dataclasses over ps + group
        # handles; rebuild them against the freshly-routed handles. The
        # completeness self-check (step 9) pins each one.
        import dataclasses as _dc2

        scheduler.request_receiver = _dc2.replace(
            scheduler.request_receiver,
            ps=scheduler.ps,
            tp_group=scheduler.tp_group,
            tp_cpu_group=scheduler.tp_cpu_group,
            attn_tp_group=scheduler.attn_tp_group,
            attn_tp_cpu_group=scheduler.attn_tp_cpu_group,
        )
        scheduler.output_streamer = _dc2.replace(
            scheduler.output_streamer, ps=scheduler.ps
        )
        if getattr(scheduler, "load_inquirer", None) is not None:
            scheduler.load_inquirer = _dc2.replace(
                scheduler.load_inquirer, ps=scheduler.ps
            )
        # The batch result processor caches the WORKERS, and it is on the
        # decode hot path: with speculation it calls back into the spec
        # worker (on_verify_complete_cpu) to resolve verified tokens. Built
        # at boot, when a phase-flip instance deliberately has no draft
        # worker, so without this rebuild the first post-flip decode died
        # with "'TpModelWorker' object has no attribute
        # 'on_verify_complete_cpu'" -- the boot-cached target being asked
        # to behave like the draft stack that had just been armed around
        # it (measured, boot 21, 2026-08-08).
        if getattr(scheduler, "batch_result_processor", None) is not None:
            scheduler.batch_result_processor = _dc2.replace(
                scheduler.batch_result_processor,
                draft_worker=want_draft,
                model_worker=want_model_worker,
            )
        # 4c. Census round realign: the detector's cadence counter drifted
        # per-rank under the pp loop; the cutover is group-aligned, so
        # re-zero here or the post-flip detector fires its gloo
        # all_gather_object at per-rank rounds and mispairs with the
        # request broadcasts on the same group FIFO (measured wedge,
        # window-2 boot 13). See CollectiveCensus.realign_round.
        from sglang.srt.distributed.collective_census import census as _census

        _census().realign_round()

        # 5. pp_max_micro_batch_size for the new pp_size (boot formula).
        get_server_args().override(
            "phase_flip.pp_max_micro_batch_size",
            pp_max_micro_batch_size=max(
                scheduler.max_running_requests // scheduler.ps.pp_size, 1
            ),
        )

        # 6. PP loop arrays: re-initialized for the new topology (reads the
        # NEW ps.pp_size).
        #
        # #631 J.3: this step USED TO DESTROY THE RESIDENT DECODE SET, and
        # that is the whole reason a flip under load was impossible. The
        # carry now lives inside init_pp_loop_state (it has three callers,
        # and the TP->PP leg re-enters event_loop_pp, which calls it again);
        # here we only bracket it with the evidence.
        #
        # The orphan check runs BEFORE the swap on purpose: a request
        # reachable only through last_mbs would mean the quiescence
        # predicate admitted a boundary that is not quiescent, and that is
        # a predicate bug to be raised, not a carry to be widened.
        from sglang.srt.managers.phase_flip_resident_carry import (
            assert_no_orphan_resident_reqs,
            promote_slot_zero_to_running_batch,
            resident_req_identity,
        )

        assert_no_orphan_resident_reqs(scheduler)
        resident_before = resident_req_identity(scheduler)
        scheduler.init_pp_loop_state()
        # 6b. The TP loops read ``running_batch``, not the slot array, so
        # the TP leg moves the re-seeded set over (and empties the slots,
        # or the next flip's harvest would resurrect a stale view of it).
        if tp_phase:
            promote_slot_zero_to_running_batch(scheduler)
        # 6c. MEMBERSHIP PIN, before the deferred aborts of step 8 are
        # allowed to change the set legitimately. Identity is (rid,
        # req_pool_idx): the slot ARRANGEMENT changes at a flip by design,
        # the MEMBERSHIP may not. A dropped request must fail here, loudly,
        # not surface a pass later as a stranded page and a stranded mamba
        # lock with the evidence already stale.
        resident_after = resident_req_identity(scheduler)
        if resident_after != resident_before:
            lost = [r for r in resident_before if r not in resident_after]
            gained = [r for r in resident_after if r not in resident_before]
            raise KvReshardError(
                f"{LOG_PREFIX} CUTOVER DROPPED THE RESIDENT DECODE SET "
                f"({direction}): {len(resident_before)} request(s) before, "
                f"{len(resident_after)} after; lost {lost[:8]}, gained "
                f"{gained[:8]}. Every request resident at a cutover must "
                f"survive it -- a dropped one strands its KV rows and its "
                f"mamba slot lock and its answer is simply never finished."
            )

        # 7. Active stack swap: the forward path follows model_worker.
        #
        # Speculation belongs to the TP DECODE phase (#631). The draft
        # worker was built on the flip's TP stack at boot and is armed
        # HERE, with the stack it targets; the PP phase runs with
        # spec_algorithm NONE and draft_worker None, which is bit-for-bit
        # the state an instance without speculation has. Mirrors the boot
        # dispatch in Scheduler.init_model_worker: with speculation the
        # model worker IS the draft worker, which drives the target
        # through it.
        scheduler.spec_algorithm = want_spec_algo
        scheduler.draft_worker = want_draft
        scheduler.model_worker = want_model_worker
        scheduler.phase_flip_active_stack = PHASE_TP if tp_phase else PHASE_PP

        # 8. Deferred aborts drain in the first post-flip round.
        window = getattr(scheduler, "phase_flip_abort_window", None)
        if window is not None and window.active:
            drained = window.deactivate_and_drain()
            if drained:
                logger.info(
                    "%s drained %d deferred abort(s) after cutover",
                    LOG_PREFIX,
                    drained,
                )

        # 9. Completeness self-check: every snapshot the rebuild list names
        # is verified against the routed source of truth, HERE, before the
        # first post-flip round can touch a stale handle. A missed rebuild
        # is a loud KvReshardError, never later corruption.
        verify_flip_cutover(scheduler, tp_phase)
        # 10. Publish the active layout for the API process (#631): the
        # log line below is the authoritative RECORD, but it is not
        # QUERYABLE, and utilisation cannot substitute for it -- a
        # pipelined PP prefill saturates all three cards exactly as TP
        # does. Published after verify, so what is advertised is a
        # cutover that passed its completeness check.
        from sglang.srt.managers.phase_flip_presence import publish_active_phase

        publish_active_phase(world_rank, scheduler.phase_flip_active_stack)
        logger.warning(
            "%s cutover complete: active stack %s, ps tp=%d pp=%d",
            LOG_PREFIX,
            scheduler.phase_flip_active_stack,
            scheduler.ps.tp_size,
            scheduler.ps.pp_size,
        )

    return _cutover


def verify_flip_cutover(scheduler, tp_phase: bool) -> None:
    """Post-cutover invariants (the coordinator's completeness pin): every
    scheduler snapshot on the 5.3 rebuild list must AGREE with the routed
    source of truth for the now-active phase. Any single stale reference
    -- a cached group handle still pointing at the other phase's group, a
    ps that kept the old topology, a model_worker from the wrong stack --
    fails HERE, loudly, before any round runs on it."""
    from sglang.srt.distributed import parallel_state as _ps

    stale = []
    if _ps.phase_flip_tp_routing_active() != tp_phase:
        stale.append(
            f"module routing active={_ps.phase_flip_tp_routing_active()} "
            f"but tp_phase={tp_phase}"
        )
    expect_tp = _ps.get_tp_group()
    expect_attn = _ps.get_attn_tp_group()
    expect_pp = _ps.get_pp_group()
    if scheduler.tp_group is not expect_tp:
        stale.append("tp_group")
    if scheduler.tp_cpu_group is not expect_tp.cpu_group:
        stale.append("tp_cpu_group")
    if scheduler.attn_tp_group is not expect_attn:
        stale.append("attn_tp_group")
    if scheduler.attn_tp_cpu_group is not expect_attn.cpu_group:
        stale.append("attn_tp_cpu_group")
    if scheduler.pp_group is not expect_pp:
        stale.append("pp_group")
    if scheduler.dp_tp_group is not scheduler.tp_group:
        stale.append("dp_tp_group")
    stacks = scheduler.phase_flip_stacks
    n = len(stacks.vector)
    want_tp_size = n if tp_phase else 1
    want_pp_size = 1 if tp_phase else n
    if scheduler.ps.tp_size != want_tp_size or scheduler.ps.pp_size != want_pp_size:
        stale.append(
            f"ps topology (tp={scheduler.ps.tp_size}, "
            f"pp={scheduler.ps.pp_size}; want tp={want_tp_size}, "
            f"pp={want_pp_size})"
        )
    if scheduler.ps.attn_tp_size != want_tp_size:
        stale.append(f"ps.attn_tp_size ({scheduler.ps.attn_tp_size})")
    # Speculation state, pinned per phase (#631). The TP phase must carry
    # the configured algorithm AND the draft worker built on the TP stack;
    # the PP phase must carry neither. A half-armed cutover -- the
    # algorithm swapped in without its draft worker, or a draft worker
    # left armed against the PP stack it was never built for -- is the
    # silent-corruption shape this check exists to refuse.
    want_draft = stacks.draft_worker if tp_phase else None
    if getattr(scheduler, "draft_worker", None) is not want_draft:
        stale.append("draft_worker (wrong phase)")
    want_algo_none = not tp_phase or stacks.draft_worker is None
    if scheduler.spec_algorithm.is_none() != want_algo_none:
        stale.append(
            f"spec_algorithm ({scheduler.spec_algorithm}; want "
            f"{'none' if want_algo_none else 'the configured algorithm'})"
        )
    if tp_phase:
        want_worker = (
            stacks.draft_worker
            if stacks.draft_worker is not None
            else stacks.tp_worker
        )
    else:
        want_worker = scheduler.tp_worker
    if scheduler.model_worker is not want_worker:
        stale.append("model_worker (wrong stack)")
    # Component ps/group snapshots (step 4b): each holder rebuilt at
    # cutover must reference the CURRENT ps object and routed groups --
    # a stale receiver relays requests in the other phase's topology
    # (measured wedge, first post-flip serving attempt 2026-08-08).
    receiver = getattr(scheduler, "request_receiver", None)
    if receiver is not None:
        if receiver.ps is not scheduler.ps:
            stale.append("request_receiver.ps")
        if receiver.attn_tp_group is not scheduler.attn_tp_group:
            stale.append("request_receiver.attn_tp_group")
        if receiver.tp_cpu_group is not scheduler.tp_cpu_group:
            stale.append("request_receiver.tp_cpu_group")
    streamer = getattr(scheduler, "output_streamer", None)
    if streamer is not None and streamer.ps is not scheduler.ps:
        stale.append("output_streamer.ps")
    brp = getattr(scheduler, "batch_result_processor", None)
    if brp is not None:
        # On the decode hot path, and it calls into the SPEC worker.
        if brp.model_worker is not scheduler.model_worker:
            stale.append("batch_result_processor.model_worker")
        if brp.draft_worker is not scheduler.draft_worker:
            stale.append("batch_result_processor.draft_worker")
    inquirer = getattr(scheduler, "load_inquirer", None)
    if inquirer is not None and inquirer.ps is not scheduler.ps:
        stale.append("load_inquirer.ps")
    window = getattr(scheduler, "phase_flip_abort_window", None)
    if window is not None and window.active:
        stale.append("abort window still active (drain missed)")
    # The resident decode set must live in the handle the ACTIVE phase's
    # event loop reads, and nowhere else (#631 J.3). The TP loops read
    # ``running_batch``; ``event_loop_pp`` reads the slot array and rebinds
    # ``running_batch`` per slot. A set left in the other phase's handle is
    # not merely untidy: it is invisible to the loop that is now running,
    # so those requests never decode again, and it is a second ageing view
    # that the next flip's harvest would resurrect.
    slots = list(getattr(scheduler, "running_mbs", []) or [])
    slot_resident = [i for i, mb in enumerate(slots) if len(getattr(mb, "reqs", []) or [])]
    running = getattr(scheduler, "running_batch", None)
    running_n = len(getattr(running, "reqs", []) or [])
    if tp_phase and slot_resident:
        stale.append(
            f"resident requests left in the PP slot array {slot_resident} "
            f"while the TP loop reads running_batch"
        )
    if not tp_phase and running_n and not any(running is mb for mb in slots):
        stale.append(
            f"running_batch holds {running_n} resident request(s) that are "
            f"in no PP slot, so event_loop_pp will never see them"
        )
    if stale:
        raise KvReshardError(
            f"{LOG_PREFIX} CUTOVER INCOMPLETE ({'tp' if tp_phase else 'pp'} "
            f"phase): stale after rebuild: {', '.join(stale)}. A stale "
            f"snapshot surviving cutover is the silent-corruption class "
            f"this check exists to catch -- refusing to run a round on it."
        )


def build_phase_flip_runtime(scheduler) -> "PhaseFlipRuntime":
    """Factory mirroring build_kv_reshard_runtime (kv_reshard.py): wires
    the scheduler's real state into PhaseFlipRuntime. Called lazily from
    the first scheduler round (house pattern); by then the boot builder
    has installed scheduler.phase_flip_stacks."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        get_world_group,
    )
    from sglang.srt.managers.kv_pressure_runtime import default_collective_min
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence
    from sglang.srt.managers.kv_reshard import _dist_exchange

    stacks = scheduler.phase_flip_stacks
    if stacks is None:
        raise KvReshardError(
            "build_phase_flip_runtime before build_phase_flip_tp_stack "
            "(the boot builder owns pools, arena and images)"
        )
    server_args = scheduler.server_args
    flip_tp = get_phase_flip_group("tp")
    world = get_world_group()

    pp_pool = scheduler.tp_worker.model_runner.token_to_kv_pool
    tp_pool = stacks.tp_worker.model_runner.token_to_kv_pool
    for name, pool in (("PP", pp_pool), ("TP", tp_pool)):
        if not hasattr(pool, "full_kv_pool"):
            raise KvReshardError(
                f"the {name} stack's pool {type(pool).__name__} has no "
                f"full_kv_pool; the flip moves hybrid-model full-attention "
                f"KV only (DESIGN_631 scope)"
            )
    pp_full = pp_pool.full_kv_pool
    tp_full = tp_pool.full_kv_pool
    pp_view = KvPoolView(pp_full.k_buffer, pp_full.v_buffer)
    tp_view = KvPoolView(tp_full.k_buffer, tp_full.v_buffer)

    # Global full-attention geometry from the TP stack's config (pp=1 ->
    # unmutated; the PP stack's was rewritten to its stage-local slice).
    # full_attention_layer_ids is a property of the HYBRID HF text config
    # (Qwen3NextConfig etc.), not of sglang's ModelConfig wrapper -- the
    # attention registry reads it via runner.mambaish_config, mirror that
    # (first real-metal flip boot, 2026-08-08).
    tp_model_config = stacks.tp_worker.model_config
    full_ids = list(tp_model_config.hf_text_config.full_attention_layer_ids)
    layer_map = derive_pp_full_attn_layer_map(
        full_ids,
        int(tp_model_config.num_hidden_layers),
        int(server_args.pp_size),
    )

    runtime = PhaseFlipRuntime(
        n_ranks=world.world_size,
        rank=world.rank_in_group,
        layer_map=layer_map,
        n_layers=len(full_ids),
        # build_phase_flip_transition documents this as "the weighted DCP
        # token vector of the TP layout" -- which rank OWNS which rows, a
        # token-space question. The weight shard vector answers a different
        # one.
        tp_vector=stacks.token_vector,
        boot_phase=PHASE_PP,
        consensus_interval=int(
            getattr(server_args, "kv_reshard_consensus_interval", 8)
        ),
        park_deadline_s=park_deadline_s(),
        # Label it as OURS: a shared helper reporting under its own
        # module's name sent a live wedge into the wrong subsystem.
        collective_min=default_collective_min(
            flip_tp.cpu_group, label="phase_flip.consensus"
        ),
        # #631 option 2(b): the pollable entry gate, and the non-blocking
        # pump that delivers this rank's arm forward while it waits.
        presence=PhaseFlipPresence(
            n_ranks=world.world_size,
            rank=world.rank_in_group,
        ),
        # #631 G: ONE mechanism where there used to be two half-working
        # ones. pump_fn and drain_fn are gone from the wiring -- both were
        # built on is_completed(), which never fires on this transport in
        # either direction (corpse F), so neither ever moved a byte. The
        # service turn does their job with a predicate the transport can
        # honour: a counter published on /dev/shm strictly after each
        # isend is posted.
        #
        # It is wired on EVERY rank, unlike the pair it replaces, which
        # were gated on the chain receiver and were therefore off on rank
        # 0 -- the intake rank, the one whose starvation defined corpse G.
        service_fn=getattr(scheduler, "pp_flip_service", None),
        channels_empty_fn=getattr(scheduler, "pp_flip_channels_empty", None),
        # (i) withhold presence until this rank's own forward is flushed,
        # so the flag means "I owe no send" rather than merely "I am
        # armed". This is now a condition that can be REACHED: the service
        # turn reaps the handle once the downstream's counter proves the
        # message consumed, where the pump could only ever fail to.
        owes_send_fn=getattr(scheduler, "pp_owes_chain_send", None),
        exchange=_dist_exchange(flip_tp.device_group, pp_view.device),
        pp_pool_view=pp_view,
        tp_pool_view=tp_view,
        live_slots_fn=build_flip_live_slots_fn(scheduler),
        ready_fn=build_flip_quiescence_fn(scheduler),
        cutover_fn=build_production_flip_cutover(scheduler),
        # DESIGN_631 3.6 order inside the no-return region: GDN state move
        # (5.3b mover -- its preconditions re-validate on every flip and
        # refuse loudly, the reachable-refusal contract), then the arena
        # refill. The full-attn KV move ran before these by the runtime.
        pre_cutover_fns=(
            _build_gdn_leg(scheduler),
            stacks.refill,
        ),
        pre_write_fns=(_build_kv_backing_swap(scheduler, stacks),),
        guards=flip_blocking_guards(scheduler),
    )
    # #631 J: read-only handle for the pool census straddling the cutover.
    runtime._census_scheduler = scheduler
    return runtime


def _build_kv_backing_swap(scheduler, stacks) -> Callable[[str], None]:
    """The runtime half of exclusive KV backing (#631).

    Runs at the read/write seam, the one instant where the source pool has
    been fully drained and the destination not yet touched, so the physical
    pages may move from one layout to the other. The VA reservations are
    untouched, so every address the TP stack's decode graphs baked in stays
    valid across any number of flips.

    Inert unless both pools were built VA-backed (swappable_backing, set
    under --enable-phase-flip): without it there is nothing to swap and the
    old both-resident behaviour stands.
    """
    pp_pool = scheduler.tp_worker.model_runner.token_to_kv_pool
    tp_pool = stacks.tp_worker.model_runner.token_to_kv_pool

    def _swap(direction: str) -> None:
        src, dst = (
            (pp_pool, tp_pool) if direction == PP_TO_TP else (tp_pool, pp_pool)
        )
        if not hasattr(src, "release_backing"):
            return
        # SOURCE FIRST, and the order is the whole point. Restoring the
        # destination first would hold both layouts' pages for the width of
        # the swap, which is precisely the residency being removed -- and
        # the corridor floor is a CONTINUOUS minimum, so a peak that lasts
        # only a few milliseconds still counts against it.
        #
        # Releasing first is safe because every row this transition owes has
        # already been read into the payloads above; nothing reads the
        # source pool again. The window where neither layout is backed is
        # bounded by these two calls and no kernel runs inside it.
        #
        # The restore cannot fail for want of memory: boot sized the budget
        # for max(PP, TP) and the source's pages were just handed back, so
        # the destination's span is covered. If it raises anyway it raises
        # loudly here, inside the flip, rather than corrupting anything.
        src.release_backing()
        dst.restore_backing()

    return _swap


def _build_gdn_leg(scheduler) -> Callable[[str], None]:
    from sglang.srt.managers.gdn_flip_mover import build_gdn_flip_mover

    return build_gdn_flip_mover(scheduler)


class PhaseFlipRuntime:
    """Drives one group's PP<->TP KV layout flip at a quiescent boundary.

    Injectables mirror ``KvReshardRuntime`` so the hermetic tests drive
    REAL threads through mock channels: ``collective_min`` is the
    consensus channel, ``exchange`` the pairwise byte channel,
    ``pp_pool_view``/``tp_pool_view`` the two resident pools (PP view
    layers = this stage's ordinals ascending; TP view layers = ALL
    ordinals ascending), ``live_slots_fn`` the replicated live slot
    enumeration (tree values UNION parked requests' rows -- DESIGN_631
    section 3.5), ``ready_fn`` the flip quiescence predicate,
    ``cutover_fn(direction)`` the snapshot-cache installer,
    ``pre_write_fns`` run at the read/write seam (cross-phase KV backing
    swap); ``pre_cutover_fns`` the ordered extra movers (weights arena, GDN
    state) executed inside the no-return region before cutover.
    """

    def __init__(
        self,
        *,
        n_ranks: int,
        rank: int,
        layer_map: Sequence[Sequence[int]],
        n_layers: int,
        tp_vector: Sequence[int],
        boot_phase: str = PHASE_PP,
        consensus_interval: int = 8,
        park_deadline_s: float = DEFAULT_PARK_DEADLINE_S,
        presence=None,
        pump_fn: Optional[Callable[[], None]] = None,
        # #631 clause (ii): consume whatever the upstream has already sent,
        # without blocking, so no peer can block on this armed rank.
        drain_fn: Optional[Callable[[], None]] = None,
        # #631 clause (i): True while this rank still owes a chain send.
        # Presence is withheld until it reads False, so the flag means
        # "my chain is flushed; I owe no send".
        owes_send_fn: Optional[Callable[[], bool]] = None,
        # #631 G: ONE TURN OF THE ARMED SERVICE LOOP. Consume every inbound
        # message the upstream's counter accounts for, then reap this
        # rank's own sends the downstream's counter proves consumed. It
        # subsumes pump_fn and drain_fn, which were the same intent built
        # on is_completed() -- a predicate this transport never satisfies
        # (corpse F), so they moved nothing.
        service_fn: Optional[Callable[[], None]] = None,
        # #631 G: returns None when every channel of this rank is empty,
        # else a human-readable reason. Flip-commit hygiene: a message in
        # flight across the re-formation misframes the post-flip stream.
        channels_empty_fn: Optional[Callable[[], Optional[str]]] = None,
        presence_deadline_s: float = DEFAULT_PRESENCE_DEADLINE_S,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        exchange: Optional[
            Callable[[Dict[int, torch.Tensor], Dict[int, int]], Dict[int, torch.Tensor]]
        ] = None,
        pp_pool_view: Optional[KvPoolView] = None,
        tp_pool_view: Optional[KvPoolView] = None,
        live_slots_fn: Optional[Callable[[], torch.Tensor]] = None,
        ready_fn: Optional[Callable[[], bool]] = None,
        cutover_fn: Optional[Callable[[str], None]] = None,
        pre_cutover_fns: Sequence[Callable[[str], None]] = (),
        pre_write_fns: Sequence[Callable[[str], None]] = (),
        guards: Sequence[str] = (),
        clock: Callable[[], float] = time.perf_counter,
        # #631: injected so the spin can be driven deterministically in
        # tests. The spin blocks on no channel; this only paces it.
        sleep: Callable[[float], None] = time.sleep,
        presence_poll_interval_s: float = DEFAULT_PRESENCE_POLL_INTERVAL_S,
    ):
        if n_ranks < 2:
            raise KvReshardError(
                f"a phase flip needs a multi-rank group, got n_ranks={n_ranks}"
            )
        if not (0 <= int(rank) < n_ranks):
            raise KvReshardError(f"rank {rank} out of range for {n_ranks} ranks")
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if collective_min is None or exchange is None:
            raise KvReshardError(
                "a phase flip needs both a consensus channel (collective_min) "
                "and a pairwise byte channel (exchange); running without them "
                "would turn the first honest divergence into a hang instead "
                "of a loud error."
            )
        missing = [
            name
            for fn, name in (
                (pp_pool_view, "pp_pool_view"),
                (tp_pool_view, "tp_pool_view"),
                (live_slots_fn, "live_slots_fn"),
                (ready_fn, "ready_fn"),
                (cutover_fn, "cutover_fn"),
            )
            if fn is None
        ]
        if missing:
            raise KvReshardError(f"PhaseFlipRuntime needs {', '.join(missing)}")
        if boot_phase not in (PHASE_PP, PHASE_TP):
            raise KvReshardError(f"unknown boot phase {boot_phase!r}")

        self._n = int(n_ranks)
        self._rank = int(rank)
        self._map = validate_layer_map(layer_map, n_layers)
        self._n_layers = int(n_layers)
        self._vec = tuple(int(x) for x in tp_vector)
        if len(self._map) != self._n or len(self._vec) != self._n:
            raise KvReshardError(
                f"layer map has {len(self._map)} stages and the vector "
                f"{self._vec} has {len(self._vec)} entries, but the group "
                f"has {self._n} ranks -- the flip reuses the SAME ranks"
            )
        my_layers = self._map[self._rank]
        if pp_pool_view.num_layers != len(my_layers):
            raise KvReshardError(
                f"PP pool view has {pp_pool_view.num_layers} layers but "
                f"stage {self._rank} owns {len(my_layers)} "
                f"({my_layers}); the view must cover exactly this stage's "
                f"ordinals, ascending"
            )
        if tp_pool_view.num_layers != self._n_layers:
            raise KvReshardError(
                f"TP pool view has {tp_pool_view.num_layers} layers but the "
                f"model has {self._n_layers} full-attention layers; the TP "
                f"layout holds every ordinal on every rank"
            )
        self._fp = _config_fingerprint(self._map, self._vec)
        self._phase = boot_phase
        self._interval = int(consensus_interval)
        self._collective_min = collective_min
        # #631 option 2(b): the pollable entry gate.
        self._presence = presence
        self._pump_fn = pump_fn
        self._drain_fn = drain_fn
        self._owes_send_fn = owes_send_fn
        self._service_fn = service_fn
        self._channels_empty_fn = channels_empty_fn
        #: Diagnostics: how often presence was withheld because a channel
        #: was not yet empty, and how often the entry check actually
        #: caught a non-empty channel at the gate (which should be never).
        self.presence_withheld_channels = 0
        self.entry_channel_violations = 0
        self._last_withhold_log = None
        self._last_not_ready_log = None
        #: #631 J: read-only handle for the pool census. Set by the
        #: builder; absent in unit stubs, where the census is a no-op.
        self._census_scheduler = None
        self._presence_deadline_s = float(presence_deadline_s)
        self._presence_wait_started = None
        self._gate_open_epoch = None
        #: #631 THE ROUND STAMP. Counts the consensus reductions this arm
        #: has COMPLETED, and is the second half of the presence marker's
        #: identity. The ranks agree on it without exchanging it: a
        #: completed reduction is a synchronisation point they all leave
        #: together, so they all enter the next one carrying the same
        #: count. Never a local loop counter -- those diverge under
        #: event_loop_pp, which is the very reason the gate exists.
        self._entry_round = 0
        self._sleep = sleep
        self._presence_poll_interval_s = float(presence_poll_interval_s)
        #: The (epoch, round) whose pre-entry wait is currently being
        #: timed. The bound is PER ROUND: a fresh round is a fresh
        #: question and gets its own budget.
        self._presence_wait_stamp = None
        self.presence_timeouts = 0
        #: Diagnostics: rounds in which presence was WITHHELD because this
        #: rank still owed a chain send (clause (i)). A non-zero count on a
        #: healthy boot is normal -- it is the flush being waited out.
        self.presence_withheld_rounds = 0
        self._join_deadline_s = DEFAULT_JOIN_DEADLINE_S
        self.join_deadline_aborts = 0
        self._exchange = exchange
        self._pp = pp_pool_view
        self._tp = tp_pool_view
        self._live_slots_fn = live_slots_fn
        self._ready_fn = ready_fn
        self._cutover_fn = cutover_fn
        self._pre_cutover_fns = tuple(pre_cutover_fns)
        # #631: run at the read/write seam, where the source pool is fully
        # drained and the destination not yet touched -- the only safe
        # instant to move physical backing between the two layouts.
        self._pre_write_fns = tuple(pre_write_fns)
        self.blocking_guards = tuple(guards)
        self._clock = clock

        self._round = 0
        self._epoch = 0
        self._pending: Optional[str] = None
        self._last_hold_reason: Optional[str] = None
        self.desync_checks = 0
        self.completed = 0
        self.last_stats: Optional[dict] = None
        #: Wall-clock bound on the parked wait; see DEFAULT_PARK_DEADLINE_S.
        self._park_deadline_s = float(park_deadline_s)
        #: Clock reading of the moment this rank armed, or None when idle.
        self._armed_at: Optional[float] = None
        #: Flips abandoned because the park deadline expired. A counter, so
        #: "this never happens in practice" stops being an assumption.
        self.park_deadline_aborts = 0
        #: Flips abandoned because the live set did not fit the target
        #: pool. Same reason for counting it.
        self.fit_aborts = 0

        logger.info(
            "%s armed at boot: rank %d/%d, phase %s, layer map %s, vector "
            "%s, consensus every %d rounds%s",
            LOG_PREFIX,
            self._rank,
            self._n,
            self._phase,
            self._map,
            self._vec,
            self._interval,
            (
                "; guards BLOCKING arming: " + ", ".join(self.blocking_guards)
                if self.blocking_guards
                else ""
            ),
        )

    # -- state ---------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- #631 THE RESUME GATE ------------------------------------------
    #
    # THE DEFECT IT CLOSES (reproduced 2026-08-09 06:26:34-35Z, specimen
    # /spinning/evidence-631/pp_proxy_mispair_20260809T0626Z). An abandon is
    # RANK-LOCAL: every rank times out on its own clock, so the ranks stop
    # being armed at different instants. The rank that disarms FIRST resumes
    # launching and sends its proxy hidden states. A peer still armed is
    # still withholding, so its ``cur_batch`` is None -- and the proxy recv
    # in _event_loop_pp_body is guarded by THAT rank's own batch, never by
    # whether the upstream sent. The message is not taken. It strands in
    # _pp_tensor_dict_inbox, and because the proxy stream is PURELY
    # POSITIONAL (no mb_id, no sequence number, no length; demultiplexed
    # only on __msg_type__) every later receive on that rank is off by one,
    # silently, for the rest of the loop is life. Observed: PP0 and PP2
    # abandoned, PP1 abandoned LAST, PP1 faulted.
    #
    # A COMMIT never showed this because the cutover re-enters
    # event_loop_pp and init_pp_loop_state resets every buffer, the inbox
    # included. The abandon paths reset nothing -- which is correct, and
    # must stay correct: unlike a commit, an abandon has NO quiescence
    # guarantee, so resetting loop state there would discard in-flight
    # microbatches. The fix therefore belongs at the RESUME, not at the
    # abandon.
    #
    # WHY THIS CANNOT RECREATE THE DEADLOCK CLASS, which is the first thing
    # to check when adding anything group-wide to this feature. THE DESIGN
    # LAW IS "no rank may block on any channel while a peer may be in a
    # different blocking channel". This gate BLOCKS ON NOTHING. It is a
    # PREDICATE, re-evaluated on each pass by the same code that already
    # decides whether to launch a batch: a gated rank keeps cycling its
    # loop, keeps servicing its channels, keeps answering health -- it
    # merely declines to LAUNCH NEW WORK, which is exactly what it was
    # already doing one pass earlier while armed. There is no rendezvous,
    # no barrier primitive and no wait. Marker writes are single-writer
    # /dev/shm files, reads are os.path.exists. Nothing here can be waited
    # on, so nothing here can deadlock.
    #
    # IT IS STILL BOUNDED, AND THE BOUND IS LOUD. A rank that dies while
    # armed would otherwise hold its peers out of service for ever, which
    # would be a worse failure than the one being fixed. On expiry the gate
    # opens anyway and says so at error level.

    #: How long a disarmed rank waits for its peers to publish their own
    #: disarm before resuming regardless. Deliberately short: the skew this
    #: closes is milliseconds (all three ranks abandoned inside the same
    #: logged second), so a second is already generous, and the cost of
    #: over-waiting is withheld throughput on a live server.
    RESUME_GATE_S = 1.0

    def _open_resume_gate(self) -> None:
        """Publish this rank's disarm and start gating its own resume.

        TOTALLY DEFENSIVE, and that is a requirement rather than caution.
        This runs on the ABANDON path, whose whole point is that the flip
        is the optional thing and the requests are not: an abandon must
        never raise, or the parked requests die with the instance --
        precisely what the park deadline exists to prevent. A gate that
        cannot arm degrades to "no gate" (the old behaviour), never to an
        exception.
        """
        try:
            clock = getattr(self, "_clock", None)
            if clock is None:
                return
            epoch = getattr(self, "_epoch", 0)
            self._resume_gate_epoch = epoch
            self._resume_gate_until = clock() + self.RESUME_GATE_S
            self._resume_gate_warned = False
            presence = getattr(self, "_presence", None)
            if presence is None:
                return
            presence.declare_disarmed(epoch)
        except Exception as exc:  # noqa: BLE001 - an abandon may never raise
            logger.error("%s could not arm the resume gate: %s", LOG_PREFIX, exc)

    def resume_withheld(self) -> Optional[str]:
        """Must this rank keep withholding new work? Reason if so.

        Called from the scheduler is parking test on every pass. Returns
        None the moment the group has all disarmed, or when the bound
        expires.
        """
        until = getattr(self, "_resume_gate_until", None)
        if until is None:
            return None
        epoch = getattr(self, "_resume_gate_epoch", 0)
        presence = getattr(self, "_presence", None)

        if presence is not None:
            try:
                if presence.all_disarmed(epoch):
                    self._resume_gate_until = None
                    return None
                seen = sorted(presence.disarmed(epoch))
            except Exception:  # noqa: BLE001 - a probe may not break serving
                self._resume_gate_until = None
                return None
        else:
            self._resume_gate_until = None
            return None

        if getattr(self, "_clock", lambda: until)() >= until:
            self._resume_gate_until = None
            if not getattr(self, "_resume_gate_warned", False):
                self._resume_gate_warned = True
                logger.error(
                    "%s RESUME GATE EXPIRED after %gs: epoch %d disarm seen "
                    "from ranks %s of %d. Resuming anyway -- a peer that "
                    "never publishes is either dead or wedged, and holding "
                    "this rank out of service for ever would be worse than "
                    "the mispairing this gate prevents. If the missing rank "
                    "is alive, expect a stranded proxy message and a "
                    "PP proxy/batch mismatch shortly.",
                    LOG_PREFIX,
                    self.RESUME_GATE_S,
                    epoch,
                    seen,
                    getattr(presence, "n_ranks", -1),
                )
            return None

        return (
            f"waiting for the group to disarm before resuming (epoch "
            f"{epoch}: ranks {seen} have disarmed)"
        )

    @property
    def pending(self) -> Optional[str]:
        return self._pending

    # -- arming (replicated callers) -----------------------------------------
    def arm(self, direction: str, source: str) -> Tuple[bool, str]:
        """Arm a flip. Replicated call; the consensus round commits it once
        every rank is armed AND ready. Returns (ok, msg)."""
        if self.blocking_guards:
            msg = (
                f"phase flip refused (guards): "
                f"{', '.join(self.blocking_guards)}"
            )
            logger.warning("%s %s", LOG_PREFIX, msg)
            return False, msg
        if direction not in _DIR_ID:
            return False, f"unknown flip direction {direction!r}"
        want = _DIR_OF_PHASE[self._phase]
        if direction != want:
            return False, (
                f"flip {direction} refused: current phase is {self._phase}, "
                f"the only legal transition is {want}"
            )
        if self._pending is not None and self._pending != direction:
            logger.warning(
                "%s re-arming %s -> %s (source %s)",
                LOG_PREFIX,
                self._pending,
                direction,
                source,
            )
        self._pending = direction
        # A fresh arm starts a fresh round sequence. The epoch already
        # distinguishes this arm from any earlier one, so the round
        # simply restarts at 0 rather than having to be globally unique.
        self._entry_round = 0
        self._presence_wait_stamp = None
        # The park clock starts at ARMING, not at the first unparked round:
        # the deadline bounds how long the requests are held, and they are
        # held from the moment this rank starts withholding work.
        self._armed_at = self._clock()
        # #631 J: census AT ARM. The pre/post-cutover pair proved the move
        # and the cutover innocent (identical unaccounted set on both
        # sides), and a no-flip control boot stayed clean, so the page goes
        # missing somewhere in the ARMED window. This bracket closes it.
        self._pool_census("at-arm", direction)
        msg = (
            f"phase flip armed: {direction} (source {source}); commits at "
            f"the next consensus boundary where every rank is quiescent, or "
            f"is abandoned after {self._park_deadline_s:g}s parked"
        )
        logger.warning("%s %s", LOG_PREFIX, msg)
        return True, msg

    # -- the per-round hook ---------------------------------------------------
    def on_round(self, require_armed_and_parked: bool = False) -> Optional[dict]:
        """One scheduler round; see KvReshardRuntime.on_round. Returns move
        stats when a flip executed this round, else ``None``.

        ``require_armed_and_parked`` is the PP-phase entry gate (measured
        wedges 2026-08-08, boots 9 and 10): under event_loop_pp the local
        round counters of the ranks diverge in ABSOLUTE value (pipeline
        fill, conditional per-slot ops), so ANY blocking reduction entered
        at a local cadence can pair with a peer blocked in a pipeline recv
        whose satisfying send sits behind this rank's reduction -- moving
        the hook inside the iteration only moved the wedge. With the gate,
        an UNARMED rank performs NO collective at all (there is nothing to
        agree on; arming state arrives via the broadcast RPC on every
        rank), and an armed rank enters only once it is locally PARKED
        (ready_fn: drained microbatches, no partial chunk) -- a parked
        rank owes no pipeline send, so no recv/reduction cycle can close.
        Peers converge on their own arm+drain, MIN-skew is legal, and the
        liveness bound turns a lost peer into a loud error. A flip under
        continuous load needs the posted-async two-phase consensus -- a
        named follow-up, not this gate.

        The wait is BOUNDED (see DEFAULT_PARK_DEADLINE_S): a rank armed
        past the deadline without parking joins the reduction anyway
        carrying ``expired``, and every participating rank abandons the
        flip on the reduced maximum. Abandoning the flip is the whole
        point -- the parked requests are never abandoned."""
        self._round += 1
        armed = 1 if self._pending is not None else 0
        ready = 1 if (armed and self._ready_fn()) else 0
        expired = 1 if self._park_expired(armed, ready) else 0
        # #631 QUIESCENT-ANNOUNCE. An armed rank that is NOT yet quiescent
        # must go back around the pass loop -- that is how it drains -- and
        # must NOT announce on the way. Announcing before quiescence is
        # what made the flag mean "I was at the entry once": the rank
        # published presence, returned to the loop, and met its top-of-pass
        # commit before it could come back and ENTER. The last announcer
        # then entered the reduction and every earlier one blocked behind
        # it (measured 23:39Z, three stacks).
        #
        # An EXPIRED rank is exempt: it has been armed past the park
        # deadline without ever draining, and it must be allowed to reach
        # the reduction to carry that fact into a group-agreed
        # abandonment. It owes no fresh work by then, having withheld
        # admissions for the whole deadline.
        # Scoped to a wired presence channel ON PURPOSE. This early
        # return exists solely to stop a rank ANNOUNCING before it is
        # quiescent; with no presence channel there is no announce, so
        # applying it would change the readiness-skew behaviour of the
        # plain consensus path (which holds uniformly inside the
        # reduction) for no gain. Caught by
        # TestConsensusDiscipline::test_readiness_skew_holds_uniformly.
        if armed and self._presence is not None and not ready and not expired:
            # SAY WHAT IS HOLDING THIS RANK, periodically. Without it the
            # only evidence is "ready=0" in an abandonment 30 s later, and
            # defect I had to be read off three py-spy stacks instead.
            self._log_not_ready()
            return None
        # #631 THE ENTRY GATE, evaluated AFTER park expiry now that
        # announcing requires quiescence. It SPINS: once this rank has
        # announced it does not return to the pass loop at all, because
        # that interval is exactly what kills it. The spin blocks on no
        # channel -- it reads flags and sleeps -- and is bounded per round,
        # so a group that never assembles abandons loudly instead of
        # hanging.
        if armed:
            gate = self._spin_for_group_presence()
            if gate is not True:
                return gate  # a pre-entry abandonment, or None if disarmed
        # The PP-phase entry gate, widened by the deadline: an armed rank
        # enters once it is PARKED, or -- if it has been armed past the
        # deadline without ever parking -- to carry that fact into the
        # consensus. Entering unparked is what makes the abandonment
        # GROUP-AGREED: the peers are already blocked in this reduction
        # waiting for exactly this rank, so the flag reaches them, every
        # rank abandons the same flip in the same round, and nobody is left
        # armed against a disarmed peer. It is safe here because an armed
        # rank has been withholding new work for the whole deadline, so it
        # owes no fresh pipeline send.
        if require_armed_and_parked and not (armed and (ready or expired)):
            return None
        if not require_armed_and_parked and self._round % self._interval != 0:
            return None
        dir_id = _DIR_ID[self._pending] if self._pending is not None else 0
        payload = _encode(
            [armed, ready, expired, self._epoch, dir_id, self._fp, *self._vec]
        )
        self.desync_checks += 1
        # #631(c) WITHDRAWN -- measured fatal, kept as a warning.
        #
        # Bounding this join and abandoning from inside CANNOT work on a
        # gloo collective. Measured 2026-08-08: the 45 s bound fired
        # (CollectiveTimeoutError), and the moment this rank walked away
        # its peers saw "gloo/transport/tcp/pair.cc:547 Connection closed
        # by peer" and every rank died with "Fatal Python error: Aborted".
        # A rank that has ENTERED an all_reduce owes that all_reduce; the
        # group has no way to un-enter it. So a wedge here cannot be
        # broken from inside the collective -- any bound has to be
        # applied BEFORE entry (do not enter unless the peers are known
        # to be joining), or the reduction has to become a non-blocking
        # poll that a rank re-enters, which is a different design.
        reduced = self._collective_min(payload)
        # THE ROUND ADVANCES HERE, and only here. Reaching this line
        # means every participant completed the SAME reduction, so this
        # is the one instant at which the ranks provably agree -- which
        # is exactly what makes the count usable as a shared stamp
        # without ever being exchanged. Incrementing anywhere else (a
        # local loop counter, a clock) would reintroduce the absolute
        # divergence between ranks that the gate exists to tolerate.
        self._entry_round += 1
        if len(reduced) != len(payload):
            raise KvReshardError(
                f"consensus channel returned {len(reduced)} values for a "
                f"{len(payload)}-value payload; the channel contract is "
                f"element-wise MIN of the packed proposal."
            )
        fields = [
            "armed",
            "ready",
            "expired",
            "epoch",
            "direction",
            "config_fp",
        ] + [f"vector[{i}]" for i in range(self._n)]
        lo = {f: reduced[2 * i] for i, f in enumerate(fields)}
        hi = {f: -reduced[2 * i + 1] for i, f in enumerate(fields)}

        # Equality family: epoch + config fingerprint + vector ALWAYS
        # (boot config); direction once every rank is armed.
        eq_checked = ["epoch", "config_fp"] + [
            f"vector[{i}]" for i in range(self._n)
        ]
        if lo["armed"] == 1:
            eq_checked.append("direction")
        mismatches = [
            f"{f}: min={lo[f]} max={hi[f]}" for f in eq_checked if lo[f] != hi[f]
        ]
        if mismatches:
            raise KvReshardError(
                f"{LOG_PREFIX} DESYNC at round {self._round}: the ranks "
                f"disagree on the flip state ({'; '.join(mismatches)}; this "
                f"rank: armed={armed} pending={self._pending} "
                f"epoch={self._epoch} phase={self._phase}). A flip that "
                f"disagrees across ranks must fail loudly HERE, before any "
                f"rank moves a byte under the wrong layout."
            )
        # Park deadline, decided on the MAX: one rank out of time is enough
        # to abandon the flip, and every rank in this reduction reads the
        # same max, so the abandonment is unanimous by construction.
        # Checked before the armed/ready holds -- those are the states the
        # deadline exists to stop waiting in.
        if hi["expired"] == 1:
            return self._abandon_parked_flip(ready)

        if lo["armed"] == 0:
            if hi["armed"] == 1:
                self._hold("waiting for every rank to arm (delivery skew)")
            return None
        if lo["ready"] == 0:
            self._hold(
                f"armed ({self._pending}), waiting for a group-wide "
                f"quiescent boundary (this rank ready={ready})"
            )
            return None
        self._last_hold_reason = None
        return self._execute()

    def _pool_census(self, when: str, direction: str) -> None:
        """#631 defect J: the allocator's own view, straddling the cutover.

        Reproduces the invariant checker's leak arithmetic
        (expected - free - cached) at a point where the flip can still be
        reasoned about, because by the time on_idle raises it the stacks
        have already been swapped and the evidence is one pass stale.

        Read-only and best effort: a census must never be able to affect a
        flip it is only watching.
        """
        try:
            scheduler = self._census_scheduler
            if scheduler is None:
                return
            alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
            tree = getattr(scheduler, "tree_cache", None)
            if alloc is None or tree is None:
                return
            free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())
            cached = set(tree.all_values_flatten().tolist())
            size = int(alloc.size)
            leaked = set(range(1, size + 1)) - free - cached
            reqs = _live_reqs(scheduler)
            # SLOT SCOPE MATTERS AND IS EASY TO MISREAD. Under
            # event_loop_pp, scheduler.running_batch / last_batch are
            # rebound to running_mbs[mb_id] / last_mbs[mb_id] at the TOP of
            # every slot iteration, so they describe ONE microbatch slot --
            # the one whose iteration is running -- not the rank's resident
            # set. A census that reports only those can read 0 while
            # requests sit in other slots, which is exactly how "the
            # request finished" got inferred from a slot that was merely
            # empty. Report both scopes so the two can never be confused.
            resident = 0
            slots_with_reqs = []
            for i, mb in enumerate(getattr(scheduler, "running_mbs", []) or []):
                n = len(getattr(mb, "reqs", []) or [])
                if n:
                    resident += n
                    slots_with_reqs.append(i)
            logger.warning(
                "%s POOL CENSUS %s %s: size=%d free=%d cached=%d "
                "available=%s cur_slot_reqs=%d resident_reqs=%d "
                "resident_slots=%s unaccounted=%d %s",
                LOG_PREFIX,
                when,
                direction,
                size,
                len(free),
                len(cached),
                getattr(alloc, "available_size", lambda: "?")(),
                len(reqs),
                resident,
                slots_with_reqs,
                len(leaked),
                sorted(leaked)[:12],
            )
        except Exception as exc:  # noqa: BLE001 - a census never breaks a flip
            logger.warning("%s pool census (%s) failed: %s", LOG_PREFIX, when, exc)

    def _log_not_ready(self) -> None:
        """Report what is holding this rank out of quiescence.

        Throttled to a quarter of the park deadline: a rank that drains in
        a pass or two stays silent, and one that never drains is on the
        record BEFORE the abandonment that names it.
        """
        why = None
        probe = getattr(self._ready_fn, "why_not", None)
        if probe is not None:
            try:
                why = probe()
            except Exception as exc:  # noqa: BLE001
                why = f"(quiescence probe failed: {exc})"
        if not why:
            return
        now = self._clock()
        if (
            self._last_not_ready_log is not None
            and (now - self._last_not_ready_log) < max(self._park_deadline_s / 4.0, 1.0)
        ):
            return
        self._last_not_ready_log = now
        logger.warning(
            "%s armed (%s) but NOT QUIESCENT: %s. This rank is holding the "
            "flip; it has not announced and is not at the entry.",
            LOG_PREFIX,
            self._pending,
            why,
        )

    def _spin_for_group_presence(self):
        """#631: announce, then SPIN here until the group assembles.

        THE POINT, and the whole reason this is a loop rather than one poll
        per pass: a rank that announces and then returns to the pass loop
        meets its top-of-pass commit before it can come back and ENTER, and
        that commit blocks behind whichever rank has already entered. The
        announce-to-entry interval must contain NO blocking channel
        operation, so the rank simply does not leave.

        WHY LEAVING THE LOOP IS SAFE HERE, and only here: this is reached
        only once ready_fn holds (or the park deadline has expired), i.e.
        the rank is drained -- no in-flight microbatches, no admissions, no
        owed payload. It therefore owes its peers neither hidden states nor
        chain data, and the only per-pass message it stops producing is the
        empty keep-alive forward. Peers that need nothing are either
        quiescent and spinning here too, or not yet quiescent -- and that
        second case is a BOUNDED RETRY, not a wedge: a mid-drain rank that
        stalls on its recv is released when the spinners' per-round bound
        expires, they abandon loudly, return to the loop and resume
        forwarding, it drains, and a later epoch retries with everyone
        genuinely quiescent. At true idle every rank is quiescent at once,
        so the gate opens on live evidence and the flip commits in the
        first epoch.

        Delegates each iteration to _await_group_presence, which stays the
        single-shot primitive: same announce, same round-scoped read, same
        pre-entry bound, same abandonment. Nothing new is invented here --
        this only stops the rank from going away between iterations.
        """
        while True:
            gate = self._await_group_presence()
            if gate is not True:
                if gate is not None:
                    return gate  # pre-entry abandonment: loud, nothing entered
                if self._pending is None:
                    # Disarmed underneath us (abandonment elsewhere).
                    return None
                # Not assembled yet. Sleep briefly and ask again WITHOUT
                # touching any channel. The pre-entry deadline inside
                # _await_group_presence is what ends this loop if the group
                # never arrives.
                self._sleep(self._presence_poll_interval_s)
                continue
            return True

    def is_armed(self) -> bool:
        """#631: is a flip armed on this rank right now?

        The scheduler's intake asks this every pass to decide whether it
        may admit new work and whether it may block on the chain. It is a
        read of ``_pending``, the one authority for arming -- deliberately
        not a mirrored flag, which would be a second state to keep in
        sync.
        """
        return self._pending is not None

    def _park_expired(self, armed: int, ready: int) -> bool:
        """Has this rank been armed-but-unparked past the deadline?

        Wall clock, not a round count: rounds are what the PP loop makes
        incomparable across ranks in the first place, and the quantity the
        operator cares about is how long a request may be held. The reading
        is rank-local and does NOT need to be replicated -- one rank
        raising the flag is enough, because the DECISION to abandon is
        taken from the reduced maximum in on_round, which every
        participating rank reads identically.
        """
        if not armed or ready or self._park_deadline_s <= 0:
            return False
        if self._armed_at is None:
            return False
        return (self._clock() - self._armed_at) >= self._park_deadline_s

    def _await_group_presence(self):
        """#631 option 2(b): the non-blocking armed wait, and the gate.

        Returns True when every rank is at the entry and the caller may
        safely enter the blocking reduction; None to keep polling on later
        rounds; or the result of an abandonment when the pre-entry bound
        expires.

        NOTHING IN THIS LOOP BLOCKS, which is the whole point -- it is the
        construction that satisfies the design law (no rank blocks on any
        channel while a peer may be in a different one). Concretely, per
        iteration this rank only:
          * PUMPS its outstanding arm-forward -- progresses it
            non-blockingly, never a blocking commit. A blocking commit
            here is corpse B' (boot 13): rank 0 blocked in
            _pp_commit_comm_work while its peers sat in the hidden-states
            exchange, because "the peer is waiting for the arm" is simply
            not true -- it may be in another channel entirely,
          * DRAINS its incoming chain non-blockingly (clause (ii)),
            buffering what arrives. An armed rank that stops consuming
            makes its UPSTREAM block on the ordinary top-of-pass commit,
            upstream of the gate, where no gate can help it (boot 18),
          * announces its own presence for this epoch (a file create) --
            but ONLY once it owes no send (clause (i)), so the flag means
            "my chain is flushed", not merely "I am armed",
          * polls peers' flags (file existence).

        Once all flags are up, entering is safe by CONSTRUCTION rather
        than by argument: every rank that will participate is in this
        same loop -- not blocked elsewhere -- flags are monotone so every
        rank observes the same all-ready fact, and each rank's own chain
        send was pumped to completion before it announced.

        THE THREE CLAUSES ARE ONE MECHANISM, not three precautions. (i)
        alone is unsatisfiable: a rank cannot flush a forward to a peer
        that has stopped reading. (ii) alone leaves the flag a lie, which
        is what let the peers enter on a rank that was still blocked.
        Together they close boot 18: every announced rank owes nothing,
        and no rank can be prevented from announcing.

        The bound is PRE-ENTRY and therefore legal, unlike the withdrawn
        (c): abandoning a poll costs nothing, because nothing has been
        entered and no peer is owed anything. Abandoning an ENTERED
        all_reduce aborts the whole group, which is why that bound was
        withdrawn and pinned.
        """
        if self._presence is None:
            # No presence channel wired (unit tests, or a builder that
            # predates the gate): fall through to the old behaviour rather
            # than silently never flipping.
            return True

        epoch = self._epoch
        entry_round = self._entry_round
        # PER-ROUND PRE-ENTRY BOUND. A new round is a new question, so it
        # gets its own budget; carrying the previous round's elapsed time
        # forward would abandon a perfectly healthy later round for time
        # spent waiting on an earlier one.
        stamp = (epoch, entry_round)
        if self._presence_wait_stamp != stamp:
            self._presence_wait_stamp = stamp
            self._presence_wait_started = self._clock()
        if self._presence_wait_started is None:
            self._presence_wait_started = self._clock()

        # #631 G: THE SERVICE TURN, and the reason this loop is no longer a
        # starvation source. A spinning rank used to stop issuing its
        # per-pass chain forward, and its downstream reached the hook ONLY
        # by returning from the blocking recv that forward satisfied -- so
        # the first rank to quiesce blocked every rank behind it, every
        # epoch, identically. The answer is not to keep sending (an armed
        # rank has nothing to forward) but to make the downstream not NEED
        # the send: it services its channels here and reaches the hook by
        # its own poll. Blocking inside this call is bounded by transfer
        # time, never by peer scheduling, because a counter proved the
        # message exists before the receive was made.
        if self._service_fn is not None:
            try:
                self._service_fn()
            except Exception as exc:  # noqa: BLE001 - servicing is best effort
                logger.warning("%s service turn failed: %s", LOG_PREFIX, exc)

        if self._pump_fn is not None:
            # Progress our own arm forward WITHOUT blocking on it. This is
            # what actually delivers the arm to the next stage while we
            # wait, and it is the difference between this design and
            # corpse B'.
            try:
                self._pump_fn()
            except Exception as exc:  # noqa: BLE001 - pumping is best effort
                logger.warning("%s pump failed: %s", LOG_PREFIX, exc)

        # #631 CLAUSE (ii), and the boot-18 fix. An armed rank must keep
        # servicing EVERY channel obligation it has, or a peer blocks on
        # it. Pumping alone covers only what this rank SENDS; the other
        # half of the obligation is what it RECEIVES. Boot 18: rank 2
        # armed and stopped consuming the chain, so rank 1's ordinary
        # top-of-pass commit of the previous pass's forward blocked in
        # work.wait() -- a blocking point that PRECEDES the gate, which is
        # why the gate could never cover it. Rank 1 therefore never
        # announced, the gate never assembled, and rank 0 waited in the
        # reduction. Draining here is what keeps the upstream free to
        # reach its own announce.
        if self._drain_fn is not None:
            try:
                self._drain_fn()
            except Exception as exc:  # noqa: BLE001 - draining is best effort
                logger.warning("%s drain failed: %s", LOG_PREFIX, exc)

        # #631 CLAUSE (i). ANNOUNCE ONLY ONCE THIS RANK OWES NO SEND, so a
        # raised flag means "my chain is flushed", not merely "I am armed".
        # Announcing while a forward is still outstanding is what made the
        # boot-18 flag a lie: the rank announced, went back around the
        # pass, and blocked on the top-of-pass commit of that very send
        # before it could reach the reduction its flag had promised. The
        # peers, seeing a full quorum, entered and waited for a rank that
        # was blocked elsewhere.
        #
        # Withholding is safe: presence is monotone, so a later announce is
        # simply a later fact, and the pre-entry deadline still bounds the
        # wait. A rank that can never flush abandons LOUDLY instead of
        # dragging the group into a reduction it cannot join.
        #
        # WITHHOLDING MUST FALL THROUGH TO THE DEADLINE, never return
        # early. Returning from here skips the pre-entry bound below and
        # turns "wait until flushed" into a NEW unbounded wait -- the same
        # shape as the wedge this clause exists to remove. Caught by
        # test_can_fail_a_rank_that_never_flushes_abandons_instead_of_wedging
        # while building it.
        owes = False
        if self._owes_send_fn is not None:
            try:
                owes = bool(self._owes_send_fn())
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s owes-send probe failed: %s", LOG_PREFIX, exc)
                owes = False

        # #631 G, FLIP-COMMIT HYGIENE. Quiescent AND fully serviced implies
        # every channel is empty; a rank that is not there yet withholds
        # presence exactly as a rank that owes a send does. Withholding
        # rather than abandoning is what keeps this CONVERGENT: a message
        # still in flight is normally reaped by the next service turn, and
        # the pre-entry deadline below still bounds the wait, so a rank
        # that can never empty its channels abandons loudly instead of
        # dragging the group into a reduction it cannot join.
        unclean = None
        if not owes and self._channels_empty_fn is not None:
            try:
                unclean = self._channels_empty_fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s channel probe failed: %s", LOG_PREFIX, exc)
                unclean = None
        if unclean:
            self.presence_withheld_channels += 1

        if owes or unclean:
            self.presence_withheld_rounds += 1
            # SAY WHY, PERIODICALLY. A withholding rank is invisible in the
            # log -- it simply does not announce -- and the only symptom is
            # an abandonment 60 s later naming it as "never reached the
            # entry", which points at the wrong place: it DID reach the
            # entry and chose not to announce. The first metal run of this
            # design cost a log-dig for exactly that reason. Throttled by
            # the presence deadline so a healthy withhold of a few rounds
            # stays silent and a stuck one is on the record before the
            # abandonment that follows it.
            now = self._clock()
            if (
                self._last_withhold_log is None
                or (now - self._last_withhold_log) >= self._presence_deadline_s / 4.0
            ):
                self._last_withhold_log = now
                logger.warning(
                    "%s epoch %d round %d: WITHHOLDING presence (%d rounds so "
                    "far) -- %s. This rank is AT the entry and declining to "
                    "announce; it is not blocked upstream of it.",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                    self.presence_withheld_rounds,
                    "still owes a chain send" if owes else unclean,
                )
        else:
            self._last_withhold_log = None
            self._presence.announce(
                epoch, note=f"pending={self._pending}", round_=entry_round
            )

        # A rank that is withholding cannot be part of a full quorum (its
        # own flag is down), so this is skipped rather than merely false.
        # #631 H: the predicate is now "everyone present AND nobody
        # withdrawn". A stale presence flag from a rank that has since
        # abandoned must not form a quorum -- that is corpse H.
        if not owes and not unclean and self._presence.quorum(epoch, round_=entry_round):
            # #631 G, THE ASSERT. Re-checked HERE, at the instant of entry,
            # because the withholding check above proves nothing about the
            # moment a quorum forms: a peer's message can land in between.
            # This is the cheap catch for the nastiest silent failure this
            # change can introduce -- a half-consumed two-step
            # point_to_point_pyobj message, or an unreaped isend, crossing
            # the re-formation and misframing the post-flip stream long
            # after the flip is forgotten. It also catches a sender that
            # died between posting its message and publishing its counter.
            #
            # Loud, and pre-entry: nothing has been entered, so abandoning
            # costs nothing and no peer is owed a collective. Crossing the
            # re-formation with a live channel would cost everything.
            late = None
            if self._channels_empty_fn is not None:
                try:
                    late = self._channels_empty_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s entry channel probe failed: %s", LOG_PREFIX, exc)
            if late:
                self.entry_channel_violations += 1
                logger.error(
                    "%s CHANNELS NOT EMPTY AT ENTRY for epoch %d round %d: "
                    "%s. A quiescent, fully serviced rank owes nothing on "
                    "any channel, so this is a framing or quiescence bug, "
                    "not a slow peer. Abandoning BEFORE entry -- nothing "
                    "was entered and no request was touched.",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                    late,
                )
                waited = self._clock() - self._presence_wait_started
                self._presence_wait_started = None
                return self._abandon_no_quorum(epoch, [], waited)
            self._commit_to_entering(epoch, entry_round)
            waited = self._clock() - self._presence_wait_started
            self._presence_wait_started = None
            # RE-BASE THE PARK CLOCK ON GROUP ASSEMBLY. The park deadline
            # measures "armed but never reached quiescence" -- a question
            # that is only meaningful once the group is actually
            # assembled. Left measuring from the arm, it races this gate:
            # a rank whose peers took longer than park_deadline_s to
            # arrive would abandon on the park deadline while those peers
            # were still polling, and the ranks would then disagree
            # around a gloo collective -- which is fatal, not merely
            # wrong ("Connection closed by peer" -> every rank aborts).
            # Measured 2026-08-08, boot 14: all three ranks announced
            # presence correctly, then abandoned at exactly 30.0s and the
            # group died.
            #
            # Re-basing keeps the two bounds from overlapping at all: the
            # presence bound governs assembly, the park bound governs
            # quiescence, and they now run in sequence rather than
            # concurrently.
            # ONCE PER ARM, never per round. Re-basing on every gate
            # opening makes the park deadline unreachable: the gate opens
            # each round, the clock resets each round, and a flip that can
            # never reach quiescence holds FOR EVER with its requests
            # parked -- measured 2026-08-08 (boot 17): repeated "group
            # present after 0.00s" on every rank, cutovers=0,
            # abandoned=0, and the server answering nothing.
            if self._gate_open_epoch != epoch:
                self._gate_open_epoch = epoch
                self._armed_at = self._clock()
                logger.warning(
                    "%s group present for epoch %d after %.2fs; park clock "
                    "re-based once, entering the consensus round",
                    LOG_PREFIX,
                    epoch,
                    waited,
                )
            return True

        waited = self._clock() - self._presence_wait_started
        if waited >= self._presence_deadline_s:
            # #631 H, THE WITHDRAWAL SIDE. Leaving is only permitted while
            # no peer has committed on this rank's presence. If one has,
            # this rank is still at the hook and owes it the reduction --
            # so it follows through instead of stranding it. That is the
            # invariant: any commit converts a withdrawing rank into an
            # enterer, and there is no interleaving where one enters and
            # another stays out.
            if not self._presence.may_withdraw(epoch, round_=entry_round):
                logger.warning(
                    "%s pre-entry bound expired for epoch %d round %d, but a "
                    "peer is already ENTERING on this rank's presence -- "
                    "following through into the reduction rather than "
                    "stranding it",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                )
                self._commit_to_entering(epoch, entry_round)
                self._presence_wait_started = None
                return True
            self._presence.declare_withdrawn(epoch, round_=entry_round)
            # Re-check AFTER publishing: a peer may have committed in the
            # window between the check and the write.
            if not self._presence.may_withdraw(epoch, round_=entry_round):
                logger.warning(
                    "%s withdrawal raced a peer's entry for epoch %d round "
                    "%d -- following through into the reduction",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                )
                self._commit_to_entering(epoch, entry_round)
                self._presence_wait_started = None
                return True
            missing = self._presence.missing(epoch, round_=entry_round)
            self._presence_wait_started = None
            return self._abandon_no_quorum(epoch, missing, waited)
        return None

    def _commit_to_entering(self, epoch: int, entry_round: int) -> None:
        """#631 H phase one: publish the intent to enter, then settle.

        Written BEFORE this rank actually enters, so a peer at its own
        pre-entry bound can see that someone has committed on its presence
        and is therefore forbidden from withdrawing. If a withdrawal is
        already visible, waiting is SAFE and terminates by construction:
        this rank's ENTERING marker forces that withdrawer to follow
        through, at which point it stops counting as withdrawn.
        """
        self._presence.declare_entering(epoch, round_=entry_round)
        if not self._presence.withdrawn(epoch, round_=entry_round):
            return
        deadline = self._clock() + self._presence_deadline_s
        while self._presence.withdrawn(epoch, round_=entry_round):
            if self._clock() >= deadline:
                logger.error(
                    "%s epoch %d round %d: a peer stayed WITHDRAWN despite "
                    "this rank ENTERING. The tie-break should have forced "
                    "it in; entering anyway would strand this rank.",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                )
                return
            self._sleep(self._presence_poll_interval_s)

    def _abandon_no_quorum(self, epoch: int, missing, waited: float):
        """Pre-entry abandonment: loud, safe, and retryable.

        Safe precisely because nothing was entered -- no peer is waiting
        on a collective this rank owes. Disarms and returns to normal
        cycling; the policy may re-arm, which mints a NEW epoch, so the
        stale flags of this one are never consulted again.
        """
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._last_hold_reason = None
        self.presence_timeouts += 1
        # Defensive: the abandon path is borrowed by duck-typed unit
        # fakes, and an abandon may never raise. See _open_resume_gate.
        getattr(self, "_open_resume_gate", lambda: None)()
        logger.error(
            "%s FLIP ABANDONED (no quorum): %s waited %.1fs for epoch %d "
            "and rank(s) %s never reached the flip entry (deadline %gs). "
            "NOTHING was entered and no request was touched -- serving "
            "continues on the %s stack and the policy may re-arm, which "
            "mints a new epoch. A rank that never reaches the entry is "
            "blocked upstream of it: look there, not at the flip.",
            LOG_PREFIX,
            direction,
            waited,
            epoch,
            missing,
            self._presence_deadline_s,
            self._phase,
        )
        return None

    def _join_bounded(self, payload):
        """#631(c): the consensus reduction under a deadline.

        Raises ``PhaseFlipJoinTimeout`` when peers do not join in time,
        instead of blocking for ever. The bound is deliberately generous
        (``DEFAULT_JOIN_DEADLINE_S``): a slow peer draining a long
        prefill is normal and must NOT trip it -- this is a wedge
        breaker, not a latency control.
        """
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            CollectiveTimeoutError,
            PeerLostError,
        )

        try:
            return self._collective_min(payload, timeout_s=self._join_deadline_s)
        except TypeError:
            # An injected channel from a test/older builder that does not
            # take a deadline. Do not silently drop the bound: the whole
            # point is that this wait cannot be unbounded.
            logger.warning(
                "%s consensus channel takes no timeout; joining unbounded "
                "(a wedge here cannot be broken from inside)",
                LOG_PREFIX,
            )
            return self._collective_min(payload)
        except (CollectiveTimeoutError, PeerLostError) as exc:
            # BOTH, because the channel raises CollectiveTimeoutError and
            # catching only PeerLostError let it escape as a bare
            # "Fatal Python error: Aborted" (measured 2026-08-08).
            raise PhaseFlipJoinTimeout(
                f"no group-wide join within {self._join_deadline_s:g}s "
                f"({exc})"
            ) from exc

    def _abandon_unjoined_flip(self, why: str) -> None:
        """Give up on a flip whose consensus round never assembled.

        Same contract as ``_abandon_parked_flip``: disarm, log loudly,
        return to serving, and NEVER raise -- the flip is optional, the
        requests are not. The policy may re-arm at its next evaluation,
        so a transient skew costs one logged retry.
        """
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._last_hold_reason = None
        self.join_deadline_aborts += 1
        # Defensive: the abandon path is borrowed by duck-typed unit
        # fakes, and an abandon may never raise. See _open_resume_gate.
        getattr(self, "_open_resume_gate", lambda: None)()
        logger.error(
            "%s FLIP ABANDONED (join): %s never assembled a consensus "
            "round -- %s. Serving continues on the %s stack and no request "
            "was touched; the arm is dropped and may be retried. A peer "
            "that never joins is a peer that never reached its round hook: "
            "look for a rank blocked upstream of it, not for a slow flip.",
            LOG_PREFIX,
            direction,
            why,
            self._phase,
        )
        return None

    def _abandon_parked_flip(self, ready: int) -> None:
        """Give up on an armed flip that never reached quiescence.

        Disarms and returns to serving. Deliberately NOT an exception: the
        flip is the optional thing here, the requests are not. A raise
        would climb into the event loop and take the instance down with it,
        which is precisely the outcome this deadline exists to prevent --
        the parked requests would die with it.
        """
        waited = (
            self._clock() - self._armed_at if self._armed_at is not None else float("nan")
        )
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._last_hold_reason = None
        self.park_deadline_aborts += 1
        # Defensive: the abandon path is borrowed by duck-typed unit
        # fakes, and an abandon may never raise. See _open_resume_gate.
        getattr(self, "_open_resume_gate", lambda: None)()
        logger.error(
            "%s FLIP ABANDONED: %s was armed for %.1fs without the group "
            "reaching a quiescent boundary (deadline %gs; this rank "
            "ready=%d). The requests are NOT affected -- they were parked, "
            "not aborted, and serving resumes on the %s stack now. A rank "
            "that cannot park is holding work that never drains: look for a "
            "microbatch or a chunked prefill that never completes. Re-arm "
            "to try again.",
            LOG_PREFIX,
            direction,
            waited,
            self._park_deadline_s,
            ready,
            self._phase,
        )
        return None

    def _hold(self, reason: str) -> None:
        if reason != self._last_hold_reason:
            logger.info("%s hold: %s", LOG_PREFIX, reason)
            self._last_hold_reason = reason

    # -- pool/layer adapters --------------------------------------------------
    def _src_dst(self, direction: str) -> Tuple[KvPoolView, KvPoolView]:
        return (self._pp, self._tp) if direction == PP_TO_TP else (self._tp, self._pp)

    def _src_layer_idx(self, direction: str, ordinal: int) -> int:
        """Pool-local layer index of a global ordinal in MY sending pool."""
        if direction == PP_TO_TP:
            return self._map[self._rank].index(ordinal)
        return ordinal

    def _dst_layer_idx(self, direction: str, ordinal: int) -> int:
        if direction == PP_TO_TP:
            return ordinal
        return self._map[self._rank].index(ordinal)

    # -- the move -------------------------------------------------------------
    def _execute(self) -> Optional[dict]:
        direction = self._pending
        assert direction is not None
        t0 = self._clock()
        slots = self._live_slots_fn()
        slots = torch.unique(slots.detach().to("cpu", torch.int64))
        tr: PhaseFlipTransition = build_phase_flip_transition(
            slots, self._map, self._n_layers, self._vec, self._rank, direction
        )

        src, dst = self._src_dst(direction)
        # Bounds BEFORE any byte moves, and GROUP-AGREED before acting on
        # them. Both pools are pre-sized at boot, but whether the live set
        # FITS is a runtime quantity: it grows with the resident prefix
        # cache, and the TP pool shrank once speculation put a draft KV
        # allocation inside the same budget (#631 window 3, boot 19 --
        # "needs TP row 10896 but the TP pool holds 7719").
        #
        # Two things were wrong with raising here. First, the reading is
        # RANK-LOCAL (each rank has its own pool sizes and its own compact
        # rows), so a rank that raised while a peer proceeded would leave
        # the group half-flipped -- the same rank-local-state-feeds-
        # collective shape this file keeps having to fix. Second, raising
        # climbs into the event loop and takes the INSTANCE down; it killed
        # a healthy server that was serving fine in its current phase.
        #
        # Nothing has been mutated at this point -- the transition is a
        # plan, not a move -- so the safe answer is unanimous: reduce the
        # local verdict, and if ANY rank does not fit, every rank abandons
        # the flip and keeps serving. The flip is the optional thing here.
        too_small = []
        if tr.max_pp_row() >= self._pp.num_rows:
            too_small.append(
                f"PP row {tr.max_pp_row()} vs pool {self._pp.num_rows} rows "
                f"(the PP pool must cover every live global slot id)"
            )
        if tr.max_tp_row() >= self._tp.num_rows:
            too_small.append(
                f"TP row {tr.max_tp_row()} vs pool {self._tp.num_rows} rows "
                f"(the TP pool must cover the compact rows of vector "
                f"{self._vec})"
            )
        fits = 0 if too_small else 1
        reduced_fit = self._collective_min([fits, -fits])
        if reduced_fit[0] == 0:
            self._pending = None
            self._armed_at = None
            self._last_hold_reason = None
            self.fit_aborts += 1
            # Defensive: the abandon path is borrowed by duck-typed unit
            # fakes, and an abandon may never raise. See _open_resume_gate.
            getattr(self, "_open_resume_gate", lambda: None)()
            logger.error(
                "%s FLIP ABANDONED (pool too small for the live set): %s. "
                "This rank: %s. No bytes were moved -- the bound is checked "
                "before the plan is executed -- and serving continues on the "
                "%s stack with every request intact. The live set grows with "
                "the resident prefix cache, so flushing the cache or sizing "
                "the target pool up are both real answers; a smaller TP pool "
                "is also what a draft-KV allocation inside the same budget "
                "produces.",
                LOG_PREFIX,
                direction,
                "; ".join(too_small) if too_small else "fits (a peer did not)",
                self._phase,
            )
            return None

        # PACK (reads only): per peer, layers ascending, one row list.
        t_read0 = self._clock()
        outgoing_payloads: Dict[int, torch.Tensor] = {}
        for peer in tr.send_layers:
            # read_rows returns [n, row_nbytes] uint8, K bytes then V.
            parts = [
                src.read_rows(
                    self._src_layer_idx(direction, f), tr.send_rows[peer]
                ).reshape(-1)
                for f in tr.send_layers[peer]
            ]
            flat = torch.cat(parts)
            outgoing_payloads[peer] = torch.cat([flat, _checksum(flat)])
        read_ms = (self._clock() - t_read0) * 1000.0

        # Expected incoming sizes from MY OWN pool's row widths -- the
        # runtime pin of row byte-compatibility across layouts.
        incoming_nbytes: Dict[int, int] = {}
        for peer in tr.recv_layers:
            n = int(tr.recv_rows[peer].numel())
            nbytes = sum(
                dst.row_nbytes(self._dst_layer_idx(direction, f)) * n
                for f in tr.recv_layers[peer]
            )
            incoming_nbytes[peer] = nbytes + _CHECKSUM_BYTES

        # EXCHANGE (pools still untouched): failure up to and including
        # checksum verification aborts with both pools byte-identical.
        t_xfer0 = self._clock()
        received = self._exchange(outgoing_payloads, incoming_nbytes)
        xfer_ms = (self._clock() - t_xfer0) * 1000.0
        incoming_data: Dict[int, torch.Tensor] = {}
        for peer, rows in tr.recv_rows.items():
            payload = received.get(peer)
            if payload is None or payload.numel() != incoming_nbytes[peer]:
                got = 0 if payload is None else payload.numel()
                raise KvReshardError(
                    f"{LOG_PREFIX} exchange returned {got} bytes from peer "
                    f"{peer}, expected {incoming_nbytes[peer]} -- size "
                    f"mismatch means the layouts' row formats or the "
                    f"payload convention diverged"
                )
            data = payload[:-_CHECKSUM_BYTES]
            want = int(payload[-_CHECKSUM_BYTES:].clone().view(torch.int64).item())
            have = uint8_checksum(data)
            if want != have:
                raise KvReshardError(
                    f"{LOG_PREFIX} payload checksum mismatch from peer "
                    f"{peer}: sender {want}, receiver {have} -- refusing to "
                    f"scatter."
                )
            incoming_data[peer] = data

        # WRITE (no-return region): local first, then incoming. Targets are
        # disjoint (injective row map), so the write order is free -- kept
        # deterministic anyway.
        #
        # INVARIANT, load-bearing: every SOURCE READ completes before the
        # first DESTINATION WRITE. The local leg used to read and write per
        # layer in one loop, which is safe only while the two pools are
        # disjoint allocations. That is the #297 reads-before-writes hazard,
        # and it becomes reachable the moment the phases share backing
        # (one arena / mutually exclusive VMM backing sized max(PP, TP)
        # instead of their sum) -- then a destination write can land on a
        # source row that has not been read yet. Hoisting the reads costs
        # one list of already-materialised payloads, which the peer legs
        # above hold anyway.
        t_write0 = self._clock()
        local_src = (
            tr.local_pp_rows if direction == PP_TO_TP else tr.local_tp_rows
        )
        local_dst = (
            tr.local_tp_rows if direction == PP_TO_TP else tr.local_pp_rows
        )
        local_data = [
            src.read_rows(self._src_layer_idx(direction, f), local_src)
            for f in tr.local_layers
        ]

        # #631 CROSS-PHASE BACKING SWAP. Every byte this transition owes has
        # now been read: the peer legs are materialised above and the local
        # leg immediately before this line. Nothing further reads the SOURCE
        # pool, and the DESTINATION pool is written from here on -- so this
        # instant, and only this instant, is where the physical pages may
        # move from one layout to the other.
        #
        # Without it the two layouts' pools both hold pages for the whole
        # process life, which is what forced each of them to be sized against
        # half the per-rank budget. The VA reservations do not move, so
        # addresses baked into captured decode graphs stay valid.
        for fn in self._pre_write_fns:
            fn(direction)

        for f, data in zip(tr.local_layers, local_data):
            dst.write_rows(self._dst_layer_idx(direction, f), local_dst, data)
        for peer, rows in tr.recv_rows.items():
            n = int(rows.numel())
            offset = 0
            for f in tr.recv_layers[peer]:
                li = self._dst_layer_idx(direction, f)
                width = dst.row_nbytes(li)
                chunk = incoming_data[peer][offset : offset + n * width]
                dst.write_rows(li, rows, chunk.view(n, width))
                offset += n * width
        write_ms = (self._clock() - t_write0) * 1000.0

        # EXTRA MOVERS (weights arena, GDN state) then CUTOVER.
        #
        # #631 defect J: census the pool on BOTH sides of the cutover. The
        # leak the checker raises one pass later cannot distinguish "the
        # enumeration missed a row" from "the cutover mis-registers the
        # destination allocator", and those have opposite fixes. Straddling
        # the cutover answers it directly: if the unaccounted page is
        # already there BEFORE, the enumeration is innocent.
        self._pool_census("pre-cutover", direction)
        for fn in self._pre_cutover_fns:
            fn(direction)
        self._cutover_fn(direction)
        self._pool_census("post-cutover", direction)
        self._phase = _PHASE_AFTER[direction]
        self._pending = None
        self._armed_at = None
        self._epoch += 1
        self.completed += 1
        total_ms = (self._clock() - t0) * 1000.0
        stats = {
            "direction": direction,
            "phase": self._phase,
            "epoch": self._epoch,
            "live_slots": tr.total_slots,
            "outgoing_cells": tr.outgoing_cells,
            "incoming_cells": tr.incoming_cells,
            "sent_bytes": sum(int(t.numel()) for t in outgoing_payloads.values()),
            "received_bytes": sum(incoming_nbytes.values()),
            "read_ms": read_ms,
            "exchange_ms": xfer_ms,
            "write_ms": write_ms,
            "total_ms": total_ms,
        }
        self.last_stats = stats
        logger.warning(
            "%s DONE %s (epoch %d) in %.1f ms: %d live slots, sent %d "
            "cells / %.2f MiB, received %d cells / %.2f MiB (read %.1f ms, "
            "exchange %.1f ms, write %.1f ms)",
            LOG_PREFIX,
            direction,
            self._epoch,
            total_ms,
            tr.total_slots,
            tr.outgoing_cells,
            stats["sent_bytes"] / 1048576.0,
            tr.incoming_cells,
            stats["received_bytes"] / 1048576.0,
            read_ms,
            xfer_ms,
            write_ms,
        )
        return stats
