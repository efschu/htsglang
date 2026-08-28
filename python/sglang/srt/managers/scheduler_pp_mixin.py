from __future__ import annotations

import logging
import math
import os
import time
from array import array
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.distributed.pp_typed_channel import (
    recv_typed_tensor_dict,
    resolve_src,
    stash_typed,
    typed_inbox,
)
from sglang.srt.distributed.utils import pp_gapped_ownership_active
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.io_struct import PhaseFlipReqInput
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.phase_flip_counters import (
    CHAN_DICT,
    CHAN_PASS,
    CHAN_REQ,
    CHAN_SLOT,
)
from sglang.srt.managers.pp_admission_congruence import (
    UNKNOWN_MATCH,
    PPAdmissionDecision,
    PPAdmissionEntry,
    entries_retracted_by_rank,
    forwarded_schedule,
    order_batch_by_schedule,
    reconcile_pp_admission_decision,
    void_pp_admission_decision,
)
from sglang.srt.managers.pp_stash_disposition import (
    PP_LOOP_ONLY,
    UNDECLARED,
    census_stash,
    stash_keys_with_disposition,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, release_req
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj
from sglang.srt.utils.common import Range, get_device_module, is_xpu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


_PP_STATS_UNSET = object()

# #791 PP ADMISSION UNIFORMITY.
#
# Each PP rank is an independent scheduler that re-derives its own admission
# verdict from its own local radix-cache state (scheduler.py's
# `_get_new_batch_prefill_raw`). Requests are chain-forwarded unconditionally
# regardless of what any rank decided, and the proxy send/receive that carries
# the actual hidden-state tensor is gated on the RECEIVING rank's OWN
# independently-derived `cur_batch` -- so two ranks disagreeing about which
# requests are admitted, or how much prefix a given request reuses, is not a
# quality issue: `ScheduleBatch.prepare_for_extend` sizes the cross-stage
# tensor directly off `len(req.prefix_indices)` per request, so a length
# disagreement is a SHAPE disagreement, and an admission disagreement is a
# ROW-COUNT disagreement -- either wedges the pipeline or corrupts the wire.
#
# `pp_admission_congruence.py` (#791 + #630) is the pure decision/reconcile
# logic; this module is its only wiring. See that module's docstring for the
# full design authority (WHAT CROSSES THE WIRE, THE TWO FAILURE SHAPES, THE
# CONGRUENCE GUARD, NO COLLECTIVE, NO HAND-PINNED NUMBERS).
#
# Rides the SAME typed tensor-dict channel as "proxy"/"output" -- `kind` is a
# plain string there, not a closed enum (pp_typed_channel.py), so this needs
# zero changes to that file. It shares the wire's single CHAN_DICT
# sent/consumed counters with those two kinds by construction (both are
# always demultiplexed by `__msg_type__` AFTER coming off the wire -- see
# `_pp_recv_typed_dict`), which is correct: a rank calling the wire "empty"
# while an admission_decision message was still on it would be the same bug
# class as miscounting a proxy or output message.
ADMISSION_DECISION_KIND = "admission_decision"

# The single non-tensor payload key the decision travels under. Riding a
# plain python object inside the tensor dict is established practice on this
# wire -- `__msg_type__` and `__stamp__` (`_pp_send_dict_to_next_stage`) do
# exactly this already.
_ADMISSION_DECISION_PAYLOAD_KEY = "__admission_decision__"

# #791b: the FIRST rank's own output-ring verdict for the slot this decision
# belongs to, carried on the decision message that already travels 0 -> 1 ->
# ... -> last in the same pass.
#
# WHY IT HAS TO TRAVEL. The output ring has two gates and they read two
# DIFFERENT ranks' `mbs`: the last rank sends when ITS slot holds a runnable
# batch (`_pp_send_output_to_next_stage`), and PP0 receives when ITS slot does
# (`_pp_send_recv_and_preprocess_output_tensors`'s `_do_recv`). #753 made them
# the same EXPRESSION (`_pp_output_exchange_due`) but could not make them the
# same FACT, because each rank still asks it of its own batch. That is sound
# only while every rank's membership for a slot is identical -- which is
# exactly what a #791 retraction breaks: a downstream rank that cannot honour
# a told prefix drops the rid, the amended decision carries the drop to every
# rank AFTER it, and the last rank ends up with an empty slot while PP0, which
# is upstream of every retracting rank and therefore never sees the amendment,
# still holds the batch it launched. PP0 then blocks for ever in a receive no
# rank is required to satisfy (specimen: boot instr11, 2026-08-21 05:00:33,
# rid 2f5e25a1... told=512 local=0 retracted on rank 1; PP0 wedged two passes
# later in `_pp_recv_dict_from_prev_stage` while PP1 and PP2 sat at the top of
# their next pass in `pp_chain_receiver.recv`, waiting for the chain send PP0
# could no longer reach -- a three-way ring).
#
# So the verdict is decided ONCE, by the rank that will apply it to the
# receive, and carried to the rank that must honour it on the send. Same law
# as #791 itself ("decide admission on rank 0 and carry the decision") and as
# #753 ("send and receive must be the SAME QUESTION asked of the SAME batch"),
# applied to the one gate pair those two left rank-local.
#
# Absent key means False, so a message from a stand-in or from a path that
# never sets it produces exactly today's behaviour.
_PP_OUTPUT_EXPECTED_KEY = "__pp_output_expected__"

# #791b: marks an output message the last rank sent ONLY to keep the ring
# matched -- the slot PP0 expects an output for produced none, because the
# pipeline retracted it. Never carries tokens; see `_pp_absorb_void_output`
# for what PP0 does with it and why it may not be turned into a placeholder
# result.
_PP_VOID_OUTPUT_KEY = "__pp_void_output__"

# #797: THIS PASS IS OFF, ON EVERY RANK FROM HERE DOWN. Set by the rank that
# performs a #791 retraction and then FORWARDED (OR-ed, never cleared) by
# every rank after it, on the decision message that already travels
# 0 -> 1 -> ... -> last within the pass.
#
# The membership half of a void travels on the entries themselves
# (`void_pp_admission_decision` marks them `admitted=False`, which
# scheduler.py's admission loop already reads as "not named this pass"). This
# key carries the half the entries cannot express: a rank whose prefill
# admission is emptied does not therefore run NOTHING -- `get_next_batch_to_
# run` falls through to its running decode batch, and a downstream decode
# batch paired with the upstream's prefill batch is the same mispairing one
# forward further on. So the fact that the ROUND is withheld has to be a fact
# about the pass, not an inference each rank makes from its own emptiness.
#
# Absent key means False: a stand-in, a sender that predates this key, or
# `pp_size <= 1` all behave exactly as they did before it existed.
_PP_PASS_VOIDED_KEY = "__pp_pass_voided__"

# #797: "I have a runnable batch for this slot, so a proxy from me is coming."
# PER-HOP, not forwarded: every sender OVERWRITES it with its own slot's
# state, because the only rank a receiver needs this about is its IMMEDIATE
# upstream. (`_PP_OUTPUT_EXPECTED_KEY` above is the opposite -- PP0's verdict,
# carried verbatim to the last rank -- and the two must not be confused.)
#
# WHY A RECEIVER NEEDS IT. A rank that voids its pass has no `cur_batch`, so
# `_event_loop_pp_body` never calls `_pp_recv_proxy_tensors` -- while its
# upstream, which voided nothing, has already launched and posted its proxy
# isend. That is the bounded-recv corpse: one unmatched message per voided
# pass, and the upstream blocks for ever in the NEXT pass's
# `_pp_commit_comm_work(self.send_proxy_work)`. The voiding rank therefore
# drains exactly one proxy and discards it -- but only when a proxy really is
# coming, which is a fact only the sender has (its batch can be empty for
# capacity reasons that never reach the decision). So the sender states it.
_PP_UPSTREAM_LAUNCHED_KEY = "__pp_upstream_launched__"

# 2026-08-20, fourth deadlock of the "a rank blocked on a peer for something
# not required for this iteration's forward progress" family. PP0's ring
# wraparound receive for the admission decision (feeding
# `PPAdmissionCongruenceGuard.record_return_trip`, a pure learning step) was
# a plain blocking p2p recv inside the per-iteration loop: once PP0 had
# `pp_size` sends outstanding it stopped sending entirely until a full ring
# lap returned, and the downstream ranks that would have had to complete
# that lap were themselves blocked at the top of their own pass waiting for
# PP0's NEXT forward send. Closed ring, zero GPU utilisation, no
# ADMISSION-WEDGE marker (the detector did not know this shape). See
# `_pp_try_recv_admission_decision`'s docstring for the fix and
# `_PP_ADMISSION_PENDING_SENDS_CAP` below for the one bookkeeping
# consequence of making the receive opportunistic instead of blocking.
#
# PURE MEMORY/STALENESS BOUND, NEVER A CORRECTNESS ONE. The PP0-only
# `self._pp_admission_pending_sends` deque (see its docstring in
# `init_pp_loop_state`) exists only to gate WHEN `_event_loop_pp_body`
# opportunistically peeks for an already-arrived wraparound lap -- it holds
# no data that is read back out, so forgetting its oldest entry loses no
# information about what is actually in flight on the wire (which is
# unbounded and always was). Without a cap, a run where wraparounds stay
# unavailable for a long stretch would grow this deque by one entry every
# pass forever; this caps it instead.
_PP_ADMISSION_PENDING_SENDS_CAP = 64


def pp_admission_decision_to_wire(decision: PPAdmissionDecision) -> Dict[str, object]:
    """#791: serialize a PPAdmissionDecision for the typed tensor-dict wire.

    Plain tuples, not the dataclasses themselves -- keeps the wire payload
    independent of dataclass identity/pickling details and easy to inspect
    on the receiving side without importing anything beyond this module.
    """
    return {
        _ADMISSION_DECISION_PAYLOAD_KEY: (
            int(decision.mb_id),
            tuple(
                (
                    e.rid,
                    int(e.prefix_len),
                    int(e.extend_len),
                    bool(e.admitted),
                    bool(e.retracted),
                    e.retracted_by_rank,
                    e.observed_local,
                    # #944: the miss travels as its OWN field, never as a
                    # sentinel value of `observed_local` -- PP0 is the rank
                    # that must act on it and is never the rank that observes
                    # it. Both ends of this codec ship together, so the tuple
                    # width is a single-version fact.
                    bool(e.unresolved),
                )
                for e in decision.entries
            ),
        )
    }


def pp_admission_decision_from_wire(message: Dict[str, object]) -> PPAdmissionDecision:
    """#791: inverse of pp_admission_decision_to_wire."""
    mb_id, raw_entries = message[_ADMISSION_DECISION_PAYLOAD_KEY]
    entries = tuple(
        PPAdmissionEntry(
            rid=rid,
            prefix_len=prefix_len,
            extend_len=extend_len,
            admitted=admitted,
            retracted=retracted,
            retracted_by_rank=retracted_by_rank,
            observed_local=observed_local,
            unresolved=unresolved,
        )
        for (
            rid,
            prefix_len,
            extend_len,
            admitted,
            retracted,
            retracted_by_rank,
            observed_local,
            unresolved,
        ) in raw_entries
    )
    return PPAdmissionDecision(mb_id=int(mb_id), entries=entries)


class PPBoundaryStats:
    """Byte and wall-time accounting for the pipeline stage boundary.

    Every crossing goes through ``_pp_send_dict_to_next_stage`` or
    ``_pp_recv_typed_dict``, so these two chokepoints see the whole boundary:
    the activation proxy (``hidden_states`` + ``residual``) forward, and the
    ``next_token_ids`` feedback from the last stage back to rank 0.

    The two timings mean different things and are reported separately on
    purpose:

    * ``send`` is the ENQUEUE cost. With ``async_send=True`` the transfer is
      handed to ``isend`` and waited on later, so this is not wire time.
    * ``recv`` is a BLOCKING wait, i.e. pipeline bubble plus wire time. It is
      an upper bound on the transfer, never the transfer alone.

    The wire cost on its own has to come from a p2p ping-pong on the same
    group and the same shapes (``scripts/pp/pp_link_pingpong.py``); no
    in-server counter can separate it from the bubble.
    """

    __slots__ = ("every", "n", "nbytes", "seconds", "since_log")

    def __init__(self, every: int):
        self.every = every
        self.n = {"send": 0, "recv": 0}
        self.nbytes = {"send": 0, "recv": 0}
        self.seconds = {"send": 0.0, "recv": 0.0}
        self.since_log = 0

    @staticmethod
    def _dict_bytes(tensor_dict: Dict[str, torch.Tensor]) -> int:
        return sum(
            v.numel() * v.element_size()
            for v in tensor_dict.values()
            if isinstance(v, torch.Tensor)
        )

    def record(
        self, kind: str, tensor_dict: Dict[str, torch.Tensor], seconds: float
    ) -> None:
        self.n[kind] += 1
        self.nbytes[kind] += self._dict_bytes(tensor_dict)
        self.seconds[kind] += seconds
        self.since_log += 1
        if self.since_log >= self.every:
            self.log()

    def log(self) -> None:
        self.since_log = 0
        parts = []
        for kind in ("send", "recv"):
            n = self.n[kind]
            if not n:
                continue
            parts.append(
                f"{kind} n={n} "
                f"{self.nbytes[kind] / n / 1024:.1f} KiB/crossing "
                f"{self.seconds[kind] / n * 1e6:.1f} us/crossing "
                f"(total {self.nbytes[kind] / 1024 / 1024:.1f} MiB, "
                f"{self.seconds[kind]:.3f} s)"
            )
        logger.info(
            "PP boundary: %s [send=enqueue only, recv=bubble+wire]",
            "; ".join(parts) or "no crossings yet",
        )


def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch."""
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
    )


#: The escape hatch for the refusal below. Set it to investigate the defect;
#: it is the only way to reach the gapped forward, and it says in its own name
#: that what it produces is not to be trusted.
PP_GAPPED_KNOWN_WRONG_ENV = "SGLANG_PP_GAPPED_ALLOW_KNOWN_WRONG"


def _refuse_known_wrong_gapped_forward() -> None:
    """#753: a gapped layer set may not be SERVED while its forward is wrong.

    MEASURED, not suspected. On 2026-08-18 the gapped cut booted, served, and
    answered the determined-answer probe -- "The capital of France is", temp 0,
    seed 735000001 -- with '\\n\\n' where the same checkpoint under the
    contiguous layout answers 'Paris'. Longer generations degenerate into a
    repeated token. The error is in the FORWARD: the very first prefill token
    is already wrong, before any decode step or KV reuse.

    This is the worst class of defect this corpus tracks, because nothing about
    it looks like a failure -- the boot is clean, health is 200, throughput is
    plausible, and the text is confidently wrong. The layout also has no
    performance case left to justify the risk: measured per-GPU utilization is
    16.6/34.7/15.0 % against 80.5/55.5/47.0 % contiguous, and 2.5x the
    wall-clock on the same prompt, because the crossings serialize the stages
    by construction.

    So the gate is a REFUSAL rather than a warning. A warning in a boot log is
    not a control; this configuration must not be reachable by accident.
    """
    import os

    if os.getenv(PP_GAPPED_KNOWN_WRONG_ENV, "") not in ("", "0", "false", "False"):
        logger.warning(
            "#753: serving a gapped PP layer set with a KNOWN-WRONG forward "
            "because %s is set. Output from this instance is not correct and "
            "must not be used for anything but debugging the defect itself.",
            PP_GAPPED_KNOWN_WRONG_ENV,
        )
        return
    raise ValueError(
        "REFUSED: a gapped PP layer set (SGLANG_PP_LAYER_SET with "
        "non-contiguous stage ownership) produces a NUMERICALLY WRONG forward "
        "and may not be served. Measured 2026-08-18: the determined-answer "
        "probe 'The capital of France is' at temperature 0 returns '\\n\\n' "
        "where the same checkpoint under a contiguous layout returns 'Paris', "
        "and longer generations degenerate into a repeated token. The first "
        "prefill token is already wrong, so the fault is in the forward and "
        "not in decode or KV reuse. The crossing wire, the entry protocol, the "
        "typed channel and the lockstep exchange are all in place and the boot "
        "is otherwise clean -- which is exactly why this refuses rather than "
        "warns: nothing about the failure is visible from outside. The layout "
        "has no performance case either (per-GPU utilization 16.6/34.7/15.0 % "
        "against 80.5/55.5/47.0 % contiguous, 2.5x the wall-clock), so there "
        f"is nothing to trade against correctness. Set {PP_GAPPED_KNOWN_WRONG_ENV}=1 "
        "to reach the forward anyway while debugging it."
    )


#: "the caller did not supply a value", so that an explicit ``None`` can mean
#: "I received nothing this iteration" instead of collapsing into the default.
#: A plain ``None`` default could not express both -- see ``_do_send``.
_NOT_SUPPLIED = object()


def _pp_output_exchange_due(batch: Optional[ScheduleBatch]) -> bool:
    """Does this slot owe an output exchange? ONE predicate, both sides.

    #753. The output ring had two gates that were never the same expression:
    the last rank sent when its slot held a runnable batch, and a middle rank
    received when ITS slot did -- a DIFFERENT slot, ``(mb_id + 1) %
    pp_loop_size`` against ``(mb_id + pp_size) % pp_loop_size``. Those agree
    only because the pipeline stagger puts the ranks on different slots at the
    same instant, so the two indices name the same batch from two positions.

    A gapped forward removes the stagger -- every stage owns layers inside
    every other stage's span, so all three are in the same pass or none of them
    progresses -- and the moment the offsets go to zero the two gates start
    naming DIFFERENT batches. One rank then sends while another declines to
    receive, and the ring starves. That is the v7pp12 specimen: the iteration
    barrier timed out after 120s with 'no peer could be proven dead', because
    the peers were not dead, they had simply decided the exchange was not due.

    So the fix is not an index. It is that send and receive must be the SAME
    QUESTION asked of the SAME batch, which is what this function is.
    """
    return (
        batch is not None
        and not batch.forward_mode.is_prebuilt()
        and not _pp_can_skip_output_comm(batch)
    )


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool


def _release_dynamic_chunk_probe(scheduler, req) -> None:
    """Hand back everything one dynamic-chunk probe took: KV, MAMBA, req slot.

    THE MAMBA SLOT WAS THE ONE NOBODY FREED, and it is why
    ``--enable-dynamic-chunking`` could not boot the phase-flip stack.
    ``ReqToTokenPool.free`` releases the REQ slot only; the mamba state has its
    own allocator and its own call. The profiler freed KV and the req slot and
    nothing else, so every probe consumed one mamba slot permanently. With
    ``--max-mamba-cache-size 12`` the twelfth probe found the pool empty:

        Profiling prefill latency for dynamic chunking:  9%| | 12/128
        mamba state slot pool exhausted and nothing evictable ... pool=12
        [PP Dynamic Chunk] [PP0] profiling failed (alloc_req_slots runs out
          of memory ...)
        ValueError: pool memory leak detected!
          [full] total=509621, available=509621   <- KV came back
          [mamba] total=12, available=0, leaked_mamba_pages={1,...,12}

    It failed at exactly the pool size, and the full pool came back untouched.
    That is the whole diagnosis: not a chunk-size change racing the seam, and
    nothing to do with the seam at all -- the instance died in ``on_idle`` at
    boot, before it ever served a request.

    ORDER IS LOAD-BEARING: the mamba release reads
    ``req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]``,
    and ``free`` sets ``req_pool_idx`` to None. Freeing the req slot first
    makes the mamba release either raise or free the wrong buffer.

    Idempotent and never raises: it runs on the abort path too, where the
    request may hold some slots and not others, and an instrument that can
    raise while cleaning up after a failure turns one defect into two.

    #969 -- FOR THE PROBE ONLY, AND THAT IS NOT AN OVERSIGHT. This releases
    memory WITHOUT `cache_finished_req`, so it never reaches the
    `dec_lock_ref(req.last_node)` those methods end in. That is correct here
    and only here: the request this frees is the synthetic
    ``_dyn_chunk_probe_req``, built inside the profiling loop below, never
    matched against the prefix tree, and therefore holding no lock ref to
    give back. It is WRONG for any admitted request, and applying it to the
    PP voids is precisely the defect #969 closes -- the rows went back to the
    allocator while the tree kept its lock on them, `reset_for_retract` then
    cleared `last_node`, and the ref became unreachable for the life of the
    process. Admitted requests go through `_release_voided_request` below.

    Do not "unify" the two. Routing the probe through the disciplined path
    would decrement a ref it never took --
    `UnifiedRadixCache.cache_finished_req` decrements `req.last_node` with no
    None guard -- which is the same accounting defect in the other direction
    (#929). The two functions are not one discipline written twice; they are
    a request that holds a tree lock and a request that does not.
    """
    if req is None:
        return
    pool = getattr(scheduler, "req_to_token_pool", None)
    if pool is None:
        return
    try:
        if getattr(req, "req_pool_idx", None) is not None:
            end = getattr(getattr(req, "extend_range", None), "end", None)
            if end:
                kv_indices = pool.req_to_token[req.req_pool_idx, :end]
                scheduler.token_to_kv_pool_allocator.free(kv_indices)
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning("[PP Dynamic Chunk] probe KV release failed: %s", exc)
    try:
        # MAMBA BEFORE THE REQ SLOT. See the ordering note above.
        if getattr(req, "mamba_pool_idx", None) is not None and hasattr(
            pool, "free_mamba_cache"
        ):
            pool.free_mamba_cache(req)
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning("[PP Dynamic Chunk] probe mamba release failed: %s", exc)
    try:
        if getattr(req, "req_pool_idx", None) is not None:
            pool.free(req)
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning("[PP Dynamic Chunk] probe req-slot release failed: %s", exc)


def _release_voided_request(scheduler, req, remaining_req_count: int = 0) -> None:
    """#969: a voided request is a RETRACTED request, and is released as one.

    THE LEAK THIS CLOSES. Both void sites used to hand their non-resident
    requests to `_release_dynamic_chunk_probe`, which returns KV rows, the
    mamba slot and the req-pool row by calling the allocator and the pool
    directly. Nothing on that route reaches `cache_finished_req`, and
    `cache_finished_req` is where `dec_lock_ref(req.last_node)` lives (the
    last two lines of every implementation of it: radix_cache.py,
    pure_swa_radix_cache.py, radix_cache_cpp.py, unified_radix_cache.py). So
    the pages went back to the allocator while the radix tree kept its lock
    on them; `reset_for_retract` then cleared `last_node`, and the ref could
    no longer even be found, let alone returned. One leaked lock per voided
    request, for the life of the process.

    WHAT THAT LOOKED LIKE FROM OUTSIDE. `evictable_size_` counts UNLOCKED
    tokens, so a tree whose nodes are all pinned reports nothing evictable at
    all -- and every funding rung downstream reads its reclaim off that
    number. The boots logged "reclaimed 0 MiB" against a full tree, which is
    not a pressure reading but the statement that nothing in it was ever
    unlocked (#813/#694 family).

    NOT A THIRD MECHANISM. `release_req` (schedule_batch.py) already IS this
    discipline, and both correct retraction paths -- `retract_decode` and
    `retract_all` -- are its only callers. This function holds no release
    logic of its own: it supplies the scheduler's collaborators and the
    never-raise contract this path has always carried, and delegates the
    rest. The probe keeps its own release for the reason given in
    `_release_dynamic_chunk_probe`'s docstring, which is a different premise
    rather than a second expression of this one.

    THE CALLER MUST NOT RESET. `release_req` calls `reset_for_retract` as its
    LAST act -- last because that method clears `last_node` and
    `mamba_pool_idx`, so a release sequenced after it has nothing left to
    release and no node left to unlock. A call site that resets again
    double-counts `retraction_count`, which feeds `retract_decode`'s
    solo-OOM abort ladder: the request would be aborted at half the retries
    the operator configured. The fallback below covers only the case where
    the release raised before reaching that point -- the caller re-queues
    this request either way, and the next pass's `get_next_batch_to_run`
    dereferences `extend_range` on it with no guard of its own (#797b/#798).
    """
    if req is None:
        return
    try:
        release_req(
            req=req,
            remaing_req_count=remaining_req_count,
            server_args=scheduler.server_args,
            req_to_token_pool=scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            tree_cache=scheduler.tree_cache,
            hisparse_coordinator=getattr(scheduler, "hisparse_coordinator", None),
        )
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning(
            "#969 voided-request release failed for %s: %s",
            getattr(req, "rid", None),
            exc,
        )
    if not getattr(req, "is_retracted", False):
        try:
            req.reset_for_retract()
        except Exception as exc:  # noqa: BLE001 - a retract may not raise
            logger.warning("#969 voided-request retract failed: %s", exc)


def _park_chunked_prefill_chunk(scheduler, req) -> bool:
    """#797b: un-do ONE prepared-but-never-run chunk. True iff it parked one.

    THE CRASH THIS CLOSES, boot instr19 08:47:27, all three ranks within one
    second of each other:

        AttributeError: 'NoneType' object has no attribute 'end'
          scheduler.py  get_next_batch_to_run
            if self.chunked_req.extend_range.end > len(...prefix_indices):

    `self.chunked_req` is SCHEDULER-owned and lives across rounds, but it is
    also a member of the batch the round builds, so #797's void handed it to
    `_release_dynamic_chunk_probe` + `reset_for_retract` like any other
    admitted request. `reset_for_retract` sets `extend_range = None`
    (schedule_batch.py) and the next round dereferences it. With
    `--chunked-prefill-size 512` against ~17000-token prompts a chunked
    request is in flight essentially always, so the void hit it on its first
    try: health at 08:46:48, dead at 08:47:41.

    A VOID IS A PARK, NOT A RETRACTION, and the park is not a new concept
    here -- `PrefillAdder.add_chunked_req` already has it (schedule_policy.py,
    the #679 zero-budget branch: "the request stays the chunked request, is
    retried next round, and nothing leaks"), and the scheduler already
    documents its shape at the stash site: "a parked chunk leaves
    `extend_range.end == len(prefix_indices)`, so there is nothing new to
    cache and stashing would be a no-op". That is exactly the state a voided
    pass must leave behind, so this reconstructs it:

      * FREE ONLY WHAT THIS ROUND ALLOCATED, `[len(prefix_indices):
        extend_range.end]`. Never `[:end]`, which is what
        `_release_dynamic_chunk_probe` frees -- that helper was written for a
        synthetic dynamic-chunk PROBE whose whole range is its own, whereas a
        chunked request's `[:len(prefix_indices)]` is the RADIX TREE's, held
        under a lock ref by every chunk already stashed. Returning those to
        the allocator is a double free, not a leak.
      * PARK `extend_range`, so the next round's stash is a no-op on a chunk
        no rank downstream of the retraction ever computed. Leaving the
        prepared range in place would insert it into THIS rank's tree only,
        and the next offer for that prefix would be unhonourable -- the #630
        livelock, one voided pass per chunk.
      * GIVE BACK THE ADMISSION'S `inflight_middle_chunks` INCREMENT
        (scheduler.py, right after `new_chunked_req`). Its matching decrement
        lives in `process_batch_result_prefill`, which never runs for a
        voided pass, and the counter gates `req.finished()` -- a leaked
        increment is a request that can never report finished.

    NOT re-queued and NOT reset: the chunked request is not in
    `waiting_queue` (``add_chunked_req`` re-admits it from
    ``self.chunked_req`` directly), so appending it would put it in the batch
    twice, and resetting it would throw away the `prefix_indices` / `last_node`
    handles for every chunk already stashed.

    Idempotent and never raises, on the same argument
    `_release_dynamic_chunk_probe` makes: this runs while cleaning up after a
    divergence, and an instrument that can raise there turns one defect into
    two.
    """
    if req is None:
        return False
    extend_range = getattr(req, "extend_range", None)
    end = getattr(extend_range, "end", None)
    if end is None:
        # Already parked, already reset, or never prepared -- nothing this
        # round allocated, so nothing to give back.
        return False
    prefix_indices = getattr(req, "prefix_indices", None)
    start = 0 if prefix_indices is None else len(prefix_indices)
    pool = getattr(scheduler, "req_to_token_pool", None)
    # #961 ONE PREDICATE FOR BOTH GIVE-BACKS, because they answer the same
    # question and had drifted into two expressions of it. `end > start` is
    # "this round actually prepared a chunk": the KV release below already
    # tests it, and the `inflight_middle_chunks` give-back below did NOT --
    # it ran whenever the function got past the `end is None` gate.
    #
    # THAT GAP IS ALREADY REACHABLE, and not only through #961's change. A
    # chunk parked by `add_chunked_req`'s #679 branch carries
    # `Range(prefix, prefix)` (schedule_policy.py:1434-1436), so `end` is not
    # None, nothing was prepared, and the increment this line hands back was
    # never taken -- `process_batch_result_prefill` owns the matching
    # decrement and no pass ran. Handing back an increment that was not made
    # is the same accounting defect as leaking one, in the other direction:
    # this function's own third bullet says a leaked increment is "a request
    # that can never report finished"; a spurious give-back is a request that
    # reports finished while a chunk is still owed.
    #
    # #961 makes the same shape arrive from a second producer -- a truncated
    # request now carries `Range(told, told)` rather than `None` -- so the
    # predicate is made exact here rather than relying on the None gate above
    # to keep catching it by accident.
    try:
        prepared = int(end) > int(start)
    except Exception:  # noqa: BLE001 - cleanup must not raise
        prepared = False
    try:
        if (
            pool is not None
            and getattr(req, "req_pool_idx", None) is not None
            and prepared
        ):
            kv_indices = pool.req_to_token[req.req_pool_idx, int(start) : int(end)]
            scheduler.token_to_kv_pool_allocator.free(kv_indices)
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning("#797b parked-chunk KV release failed: %s", exc)
    try:
        req.extend_range = Range(int(start), int(start))
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning("#797b parked-chunk extend_range reset failed: %s", exc)
    try:
        if prepared and int(getattr(req, "inflight_middle_chunks", 0) or 0) > 0:
            req.inflight_middle_chunks -= 1
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise
        logger.warning("#797b parked-chunk inflight accounting failed: %s", exc)
    return True


#: #757: what the ARMED drain may do with one message off the wire.
DRAIN_STASH = "stash"
DRAIN_DISCARD = "discard"

#: #787: bounded settle window for ``pp_flip_drain_leftover_dicts``.
#:
#: These are ENGINEERING TOLERANCES, not planner/capacity quantities -- do
#: not fold them into the #770 pins-audit as if they sized anything about
#: KV, VRAM, or batching. They exist to absorb ordinary local-fabric /
#: shared-counter arrival latency (the gap between a peer's isend and this
#: rank's next poll of the shared /dev/shm sent-counter) AFTER the #787
#: sender-side ordering guarantee (``_abandon_no_quorum`` /
#: ``_abandon_unjoined_flip`` flushing pending sends before local flip
#: state clears) is in place. The settle window is NOT a substitute for
#: that guarantee and is not sound on its own: it cannot observe a send
#: that has not been issued yet, only one that has been issued but has not
#: yet been counted. Widening it papers over a missing sender-side flush
#: rather than fixing it.
DRAIN_SETTLE_BUDGET_S = 0.75
DRAIN_SETTLE_STEP_S = 0.02

#: #789: bounded backstop for the proxy-receive readiness gate,
#: ``_pp_wait_for_proxy_readiness`` (used by ``_pp_recv_proxy_tensors``
#: immediately below it). See that function's docstring for the full
#: contract; in short, THIS IS NOT THE DECISION MECHANISM -- the decision
#: is a positive reading of the CHAN_DICT sent/consumed counters (the
#: upstream provably posted a message this rank has not yet consumed),
#: checked on every poll and acted on the instant it becomes true. This
#: budget only bounds how long the gate is willing to wait for that
#: counter to move at all before concluding, loudly, that no upstream
#: scheduled work for this slot.
#:
#: HAND PIN #789: unlike DRAIN_SETTLE_BUDGET_S above (whose 0.75s is
#: chosen to absorb ordinary counter-publish latency, not compute time),
#: this number has to be generous enough to survive a LEGITIMATELY slow
#: upstream forward pass (a large prefill chunk, heavy batching, spec
#: decode verification) without mistaking it for "upstream chose not to
#: schedule this slot" -- and no specimen on this rig has independently
#: measured that worst case. It is set by analogy to the nearest existing
#: precedents for the same shape of question -- "how long may a
#: legitimately slow peer operation run before a wedge-shaped wait gives
#: up" -- DEFAULT_PARK_DEADLINE_S (30s) and DEFAULT_JOIN_DEADLINE_S (45s)
#: in phase_flip_runtime.py. Override with SGLANG_PP_PROXY_READINESS_
#: BUDGET_S if a future specimen shows this is wrong in either direction.
DEFAULT_PROXY_READINESS_BUDGET_S = 30.0
#: Poll interval for the same gate. Matches DRAIN_SETTLE_STEP_S's pacing
#: rationale exactly: prompt when the counter moves, cheap while it does
#: not.
PROXY_READINESS_POLL_STEP_S = 0.02
#: Env override for DEFAULT_PROXY_READINESS_BUDGET_S. Unset or a
#: non-positive value falls back to the documented default, mirroring
#: phase_flip_runtime.py's ENV_PARK_DEADLINE convention.
ENV_PROXY_READINESS_BUDGET = "SGLANG_PP_PROXY_READINESS_BUDGET_S"


def _pp_proxy_readiness_budget_s() -> float:
    raw = os.environ.get(ENV_PROXY_READINESS_BUDGET)
    if raw is None:
        return DEFAULT_PROXY_READINESS_BUDGET_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_PROXY_READINESS_BUDGET_S
    return value if value > 0 else DEFAULT_PROXY_READINESS_BUDGET_S


#: Position of the phase-flip epoch inside a proxy ``__stamp__``, appended by
#: ``_pp_proxy_stamp``. A value below zero, or a stamp too short to have the
#: element at all, means "this sender could not name its epoch" -- see
#: ``pp_proxy_stamp_epoch``.
PP_PROXY_STAMP_EPOCH_INDEX = 3


def pp_proxy_stamp_epoch(stamp) -> Optional[int]:
    """The phase-flip epoch a proxy stamp names, or None if it names none.

    None for a stamp written before this field existed, for a stand-in that
    carries a shorter tuple, and for a boot with no phase-flip runtime (where
    ``_pp_flip_epoch`` has nothing to read and the sender writes -1). Every
    consumer below treats None as "fall back to the slot-only comparison",
    which is exactly the behaviour that shipped before this field.
    """
    try:
        epoch = int(stamp[PP_PROXY_STAMP_EPOCH_INDEX])
    except Exception:  # noqa: BLE001 - an unreadable epoch names no epoch
        return None
    return epoch if epoch >= 0 else None


def pp_flip_epoch_of(holder) -> Optional[int]:
    """``holder._pp_flip_epoch()``, or None if the holder has no such method.

    MODULE-LEVEL AND getattr-BASED BY THIS FILE'S OWN CONVENTION (#787), not
    as a defensive habit. Several methods here are bound UNBOUND onto stand-in
    holders that carry only what the method under test touched when the holder
    was written -- `pp_flip_drain_tensor_dicts` reaches `_pp_ran_mb_ids` the
    same way, and `_pp_note_output_expectation`'s docstring states the rule.
    A holder written before #795 must keep working, and "no accessor" reads as
    "no epoch", which is exactly the slot-only comparison that shipped then.

    A free function rather than a mixin method because a holder that binds
    methods by NAME would not have the mixin method either -- which is the
    whole failure this avoids.
    """
    fn = getattr(holder, "_pp_flip_epoch", None)
    return fn() if fn is not None else None


def pp_flip_forget_ring_scoped_slots(holder) -> None:
    """Drop every slot number recorded against a ring that no longer exists.

    #829. A microbatch slot index is an index into a ring of
    ``pp_loop_size`` slots that a cutover REBUILDS FROM ZERO. It is
    therefore meaningful only together with the ring it indexes -- the
    same argument ``_pp_proxy_stamp`` makes for proxies (#795), applied to
    the only other places a slot number outlives the iteration that
    produced it:

      ``_pp_flip_arm_mb_id``   the slot an armed window began on
      ``_pp_flip_arm_epoch``   the ring generation that slot belongs to
      ``_pp_flip_resume_slot`` a slot the pass loop has been asked to
                               jump to but has not consumed yet

    ``pp_loop_size`` is re-derived on every rebuild and CHANGES across a
    cutover (the TP phase gets ``pp_size=1``, and a gapped wire pins the
    ring to a single slot), so a carried index can be not merely wrong but
    out of range.

    WHAT THIS COST WHEN IT WAS MISSING, boot_window2_0823_1554 @
    f9d7637f04: a committed cutover carried ``_pp_flip_arm_mb_id = 2``
    across the rebuild, #824 W4b's restore applied it to the fresh ring,
    and PP0 re-entered epoch 6 on slot 2 while its downstream was on slot
    0. One second later PP1 raised ``#631 PROXY LEFTOVER REFUSED`` on a
    proxy stamped ``mb_id=2`` in that same epoch, and the group died.

    ``_pp_flip_armed_passes`` IS DELIBERATELY NOT CLEARED HERE, and that
    omission is the point rather than an oversight. It is what makes the
    next tick take the falling edge, and the falling edge is the only
    place every disarm passes through -- #757's
    ``pp_flip_drain_leftover_dicts`` lives there. Clearing it would take
    that drain off the commit path silently. #829 removes the SLOT
    decision from that path and nothing else.

    Written with plain assignment rather than ``delattr`` so a holder that
    never had the attributes gains them as None, which is what every
    reader in this file already treats as "nothing recorded".
    """
    holder._pp_flip_arm_mb_id = None
    holder._pp_flip_arm_epoch = None
    holder._pp_flip_resume_slot = None


def pp_proxy_stamp_names_pass(stamp, mb_id: int, epoch: Optional[int]) -> bool:
    """Does ``stamp`` name the pass ``(epoch, mb_id)`` this rank is running?

    THE STAMP'S NAMESPACE IS (FLIP EPOCH, SLOT), NOT SLOT ALONE, and that is
    the whole content of this predicate.

    WHY SLOT ALONE IS NOT AN IDENTITY. ``mb_id`` is an index into the
    microbatch slot ring, so it lives in ``range(pp_loop_size)`` -- three or
    six values on this rig. Worse, the ring is not merely cyclic: a phase
    flip's cutover calls ``init_pp_loop_state`` (phase_flip_runtime.py:1580),
    which rebuilds ``mbs``/``running_mbs``/``last_mbs`` for the new topology
    and restarts the slot numbering from zero. So after a cutover the SAME
    slot numbers are handed out again to passes that have nothing to do with
    the ones before it, and a proxy stranded on the wire across that cutover
    matches a slot-only comparison with probability ``1/pp_loop_size`` rather
    than never. That is the specimen of 2026-08-21 06:10:48 (boot instr15,
    /spinning/evidence-665-f1/SPECIMEN_795_proxy_batch_mismatch.txt): the
    ``PROXY LEFTOVER REFUSED`` guard read a matching ``mb_id``, passed the
    message into model compute, and the only thing that caught it was the
    WIDTH check 30 layers down -- "119 row(s) for a batch of 27 token(s)".

    WHY THE EPOCH IS THE RIGHT SECOND COMPONENT, and not a longer timeout or
    a wider drain. ``PhaseFlipRuntime._epoch`` (phase_flip_runtime.py:7398)
    increments exactly once per COMPLETED cutover, immediately after
    ``_cutover_fn`` -- i.e. once per rebuild of the very ring ``mb_id``
    indexes. It is therefore not a heuristic clock but the generation number
    of the namespace itself, and the runtime already refuses to flip at all
    when ranks disagree on it (the equality family at
    phase_flip_runtime.py:3438-3440, corpse H). An ABANDONED flip does not
    advance it and does not rebuild the ring, so pre-arm proxies stay
    legitimately owed and this predicate keeps saying so -- which is exactly
    the distinction ``pp_flip_drain_leftover_dicts`` needs and could not make.

    ABSENT EPOCH MEANS TODAY'S BEHAVIOUR, on both sides. A stamp with no
    epoch, or a caller with no epoch to compare against, degrades to the
    slot-only test that shipped before. Same convention as
    ``_PP_OUTPUT_EXPECTED_KEY`` (#791b).

    HONEST RESIDUAL. Within ONE epoch a leftover a whole ring-cycle stale
    still names this slot and is still accepted; ``seq`` remains stamped and
    unconsulted because FIFO delivery already makes it monotone, so it
    discriminates nothing a receiver can predict. Closing that case needs a
    per-pass sequence the RECEIVER can derive, which is a different change.
    See ``test_a_coinciding_slot_within_one_epoch_is_the_residual_hole``.
    """
    try:
        stamp_mb = int(stamp[0])
    except Exception:  # noqa: BLE001 - an unreadable stamp names no pass
        return False
    if stamp_mb != int(mb_id):
        return False
    if epoch is None:
        return True
    stamp_epoch = pp_proxy_stamp_epoch(stamp)
    if stamp_epoch is None:
        return True
    return stamp_epoch == int(epoch)


def pp_proxy_pass_retraction_reason(
    amended: Optional[PPAdmissionDecision], rank: Optional[int]
) -> Optional[str]:
    """#791c: why the proxy about to be paired belongs to a WIDER batch.

    THE IDENTITY IS NOT THE PROBLEM ANY MORE; THE MEMBERSHIP IS. `mb_id`,
    `seq` and the flip `epoch` all answer "WHICH PASS is this message from",
    and by 2026-08-21 all three answered correctly and the instance still
    died. Boot instr17, 07:12:49 PP1:

        PP0  #788 PP-ADMISSION verdict=ADMIT n_reqs=2
             rids=51a294650b...,5e744c29f8... prefix_lens=0,16896
        PP1  #791 PP-ADMISSION unhonourable prefix on rank 1:
             rid=5e744c29f8... told=16896 local=0
        PP1  #788 PP-ADMISSION verdict=ADMIT n_reqs=1
             rids=51a294650b... prefix_lens=0
        PP1  ValueError: #631 PP proxy/batch mismatch: received hidden_states
             with 126 row(s) for a 1 batch of 22 token(s)

    126 = 22 + 104: PP0's batch is PP1's batch PLUS the request PP1 retracted.
    Same pass, same slot, same flip epoch, `PROXY LEFTOVER REFUSED` correctly
    0 on the whole boot. Nothing was stale, nothing was stranded, no cutover
    was involved -- and no sharper pass identity, per-slot generation or
    receiver-derived sequence could ever have discriminated it, because the
    message WAS this pass's.

    WHAT ACTUALLY DIVERGED. `reconcile_pp_admission_decision`
    (pp_admission_congruence.py) drops a rid whose `told` prefix this rank
    cannot honour from `effective`, and scheduler.py's admission loop then
    omits it from this rank's batch. The UPSTREAM rank built and forwarded
    its own batch from the decision as it stood before that retraction and
    has already launched: a batch in flight cannot be amended. So the
    retraction, which is rank-local and correct in itself, makes this rank's
    batch a strict subset of the one whose hidden states are on the wire.

    WHY THIS PREDICATE AND NOT A WIDTH COMPARISON. The width is already
    checked, at model_runner.py's `_hs.shape[0] != _want` -- 30 layers into
    compute, naming no cause, and BLIND WHENEVER THE TWO BATCHES HAPPEN TO
    HAVE THE SAME TOKEN COUNT. Chunked prefill caps a chunk at
    `chunked_prefill_size`, so two ranks running DIFFERENT requests routinely
    present the SAME width; that pair is not a shape error, it is silent
    wrong output, and it is the same hazard the #631 stamp exists for ("a
    leftover of the SAME width, which is silent wrong output rather than a
    shape error"). This predicate reads the retraction itself, so it fires on
    membership, not on arithmetic.

    RECEIVER-PREDICTABLE BY CONSTRUCTION, which is the property `seq` never
    had. The receiver does not infer this from anything the sender wrote: it
    IS the rank that performed the retraction, it did so at the top of this
    same pass (`_event_loop_pp_body`'s #791 block, strictly before the proxy
    receive), and #791b already records the amended decision per slot in
    `_pp_admission_amended_by_slot`. Nothing new crosses the wire.

    NONE MEANS "NOTHING KNOWN AGAINST THIS PASS", on both arguments -- a rank
    with no decision recorded for the slot, a stand-in that never ran the
    #791 block, `pp_size <= 1`, the first rank. Every such case reproduces
    exactly the behaviour that shipped before this function, per this file's
    `_pp_note_output_expectation` / `pp_flip_epoch_of` convention (#787).
    """
    if amended is None or rank is None:
        return None
    retracted = entries_retracted_by_rank(amended, rank)
    if not retracted:
        return None
    first = retracted[0]
    return (
        f"this rank retracted {len(retracted)} request(s) from the admission "
        f"decision for this pass (first: rid={first.rid} told="
        f"{first.prefix_len} local={first.observed_local} extend="
        f"{first.extend_len}), so its batch is a strict SUBSET of the one the "
        f"upstream had already launched when it forwarded these hidden states"
    )


def pp_first_retracting_rank(decision: Optional[PPAdmissionDecision]) -> Optional[int]:
    """#797: the lowest rank that retracted anything from this decision.

    THE RANKS ABOVE IT RAN THE PASS AND THE RANKS FROM IT DOWN DID NOT, and
    that split is the whole content of this number. A rank voids when it
    retracts (`pp_pass_should_void`) and every rank after it inherits the
    void on the wire, while every rank BEFORE it built and launched its batch
    from the decision as it stood earlier -- they cannot join a void decided
    after they had already gone.

    `None` when nothing in the decision names a retracting rank: an ordinary
    pass, or a void produced by something other than a #791 retraction.

    #801: WHAT A CONSUMER MAY CONCLUDE FROM THAT `None` IS NOT "do not
    forward", and reading it that way was the intermediate-hop wedge. It
    says only that this decision names no retraction; on a void it therefore
    says every rank still RAN the pass, which is the case in which the void
    must travel FURTHEST. `pp_void_relay_stop_rank` turns the two states
    into a ring position instead of into silence.
    """
    if decision is None:
        return None
    ranks = [
        int(e.retracted_by_rank)
        for e in decision.entries
        if e.retracted and e.retracted_by_rank is not None
    ]
    return min(ranks) if ranks else None


def pp_void_relay_stop_rank(
    first_retracting_rank: Optional[int], pp_size: Optional[int]
) -> Optional[int]:
    """#801: the first ring position that will NOT receive this void.

    THE DEFAULT IS THE WHOLE FIX, and the defect it closes is a DEFAULT and
    not a missing branch. `pp_first_retracting_rank` answers `None` for two
    unrelated states -- "this decision names no retraction" and "this void
    carries no decision at all" -- and #797 read both as "stop here". On a
    ring, "stop" is the one answer that cannot be taken back: the successor's
    blocking receive is already posted, and no later pass can satisfy it.
    Specimen wedge_802f_1712 is that shape, with PP1 parked in
    `_pp_recv_dict_from_prev_stage` while PP0 held a void it forwarded
    nowhere.

    WHAT THE ABSENCE ACTUALLY MEANS, and it points the other way. The void
    arm of `_pp_send_output_to_next_stage` fires when the LAST rank has no
    output to give for a slot the first rank published an expectation for.
    If no rank retracted, then no rank voided its own pass, so every rank
    except that last one still holds a launched batch for the slot and has a
    receive posted for it. The void must then reach `pp_size - 2` and stop
    one hop short of its own source -- the widest travel, not the narrowest.

    THE STOP IS STILL A STOP, and that half is unchanged: when a retraction
    IS named, ranks from it down have empty slots and their receives
    early-return, so a hop past `first_retracting_rank - 1` would leave an
    unmatched message on the channel -- the bounded-recv corpse, from the
    sender's side, which is exactly what #796's law forbids. This function
    widens the default; it does not remove the rule.

    `None` only when `pp_size` is unknown. No production `Scheduler` is in
    that state (`ps.pp_size` is set at init), and a stand-in that lacks it
    keeps the pre-#801 behaviour rather than having a ring size guessed for
    it.
    """
    if first_retracting_rank is not None:
        return int(first_retracting_rank)
    if pp_size is None:
        return None
    return max(int(pp_size) - 1, 0)


def pp_upstream_void_pending(scheduler) -> bool:
    """#951: is this pass ALREADY void because the upstream did not launch?

    THE ONE PREDICATE, READ AT TWO MOMENTS. `_pp_void_pass_without_upstream_
    launch` asks this question AFTER `get_next_batch_to_run`, which is where
    the void must be FORWARDED from -- that placement is #798's and it is
    correct. scheduler.py's void guard asks it BEFORE, which is where the pass
    must be REFUSED. Two moments, one fact; a second copy of these four
    conditions is how the two would drift.

    WHY IT IS ANSWERABLE THIS EARLY, which is the whole reason #951 is a guard
    and not a bigger unwind. `_pp_upstream_launched_incoming` is written by
    `_pp_recv_admission_decision`, and that method's own docstring records its
    placement: "Positioned in `_event_loop_pp_body` strictly BEFORE
    `get_next_batch_to_run`". So by the time the guard runs, this rank has
    already been told, by the sender itself, whether its upstream launched.
    Nothing here is derived from local state and nothing here is a new
    collective -- it is a per-hop fact that already crossed the wire.

    WHAT IT DELIBERATELY DOES NOT ASK: whether this rank derived a batch.
    #798's own site adds that term (`self.mbs[mb_id] is not None`) because it
    runs after the plan and uses it to scope the #801-spin streak. It is not
    part of the decision to REFUSE the pass -- a pass whose upstream posted no
    hidden states may not run whether or not this rank could have filled it.

    NO-OP ON EVERY PATH THAT CANNOT HAVE THE PROBLEM, and this list is the
    dangerous half. The first rank never receives an admission decision, so
    `_pp_upstream_launched_incoming` is False on EVERY pass it takes; a guard
    that skipped the `is_first_rank` term would void every pass on PP0 and
    serve nothing at all. `pp_size <= 1` has no upstream hop, and a gapped set
    has no stage-boundary proxy for the void to protect.

    Tolerant of a stand-in that never ran `init_pp_loop_state`, as this file's
    convention requires (#787): every read is defaulted rather than asserted,
    and every default is the INERT answer.
    """
    ps = getattr(scheduler, "ps", None)
    if ps is None or int(getattr(ps, "pp_size", 1) or 1) <= 1:
        return False
    pp_group = getattr(scheduler, "pp_group", None)
    if pp_group is None or getattr(pp_group, "is_first_rank", True):
        return False
    if getattr(scheduler, "_pp_gapped_wire", False):
        return False
    return not getattr(scheduler, "_pp_upstream_launched_incoming", False)


def pp_idle_void_should_report(streak: int) -> bool:
    """#801-spin: report a no-progress void streak on powers of two only.

    THE FLOOD IS PART OF THE DEFECT, not merely a symptom of it. In specimen
    boot_802f_staged1_0822_1716 PP1 emitted 2353 `#798` warnings and 2353
    `#797d` warnings -- 4706 multi-line records in five seconds, from ONE rank,
    while the other two emitted none. Nothing in that volume is information:
    every record after the first says what the first already said, and the one
    fact a reader needs -- that the rank is repeating a pass and getting
    nowhere -- is the one fact no single line states.

    Powers of two, so the count itself carries the duration: a streak of n
    costs floor(log2(n)) + 1 lines, twelve for that specimen instead of 2353,
    and the LAST line printed names the order of magnitude reached. This is
    the "said once" shape #823/#824 applied to a state that changes, adapted
    to a state that only ever grows.
    """
    return streak > 0 and (streak & (streak - 1)) == 0


def pp_idle_void_streak_exceeded(streak: int, bound: int) -> bool:
    """#801-spin: True once a no-progress void streak has reached its bound.

    A NON-POSITIVE BOUND DISABLES THE REFUSAL and is the escape hatch: it
    restores the pre-#801-spin behaviour exactly (spin for ever, silently),
    which is what an operator who suspects a false positive needs to be able
    to ask for without editing code. It is deliberately not the default,
    because a rank that has taken 512 consecutive passes without its upstream
    launching once is not serving anybody.
    """
    return bound > 0 and streak >= bound


def pp_void_forward_payload(
    holder, message: Dict[str, object]
) -> Optional[Dict[str, object]]:
    """#797: the void this rank must pass on, or None if it is the last one.

    THE WEDGE THIS CLOSES, and it is one #797 would otherwise CREATE. #791b's
    void keeps the output ring matched for the FIRST rank, because that is
    where boot instr11 died. It stops there. When the retraction happens on
    rank r, every rank in 1..r-1 also holds a launched batch for that slot and
    also has an output receive posted for it -- and PP0, having absorbed the
    void, forwards nothing, so they block on a message no rank will send.

    Before #797 that shape produced a MISPAIR rather than a wedge (the
    retracting rank narrowed its batch and the last rank still sent a real
    output), so trading one for the other would not be a fix. The void
    therefore travels the whole way the real output would have: last -> 0 ->
    1 -> ... -> r-1, each rank absorbing it, releasing its own copies of the
    requests and passing it on.

    IT MUST STOP AT r-1. Rank r and everything after it voided, so their slots
    are empty and their receives early-return (`_do_recv`'s `target is None`);
    one more hop would be an unmatched message, the bounded-recv corpse. The
    stopping point is not guessed -- it is `pp_first_retracting_rank` of the
    decision the void already carries.

    #801 -- THE RELAY INVARIANT, AND IT IS WHY THIS FUNCTION IS THE FIX SITE.
    A non-last rank that took exactly one message off this wire for a ring
    generation must put exactly one back on it, void included; the ring is
    then a strict relay and the message count on hop r -> r+1 equals the
    count the last rank emitted. #797 held that for the one state it could
    name -- a decision with a retraction in it -- and fell SILENT for the two
    it could not: a void carrying no decision, and a void whose decision
    names no retracting rank (the last rank simply had no batch for a slot
    PP0 expected an output for). Both are states in which every intermediate
    rank still holds a launched batch, so both are states in which silence
    parks the successor for ever. `pp_void_relay_stop_rank` supplies the
    ring position for those two, and the argument for the value it picks is
    in its own docstring.

    Returns the payload VERBATIM (a copy), decision included, so the next hop
    can apply the identical rule. None whenever the rule does not fire, which
    is every ordinary pass and every rank at or past the stop position.
    """
    if not isinstance(message, dict):
        return None
    group = getattr(holder, "pp_group", None)
    if group is None or getattr(group, "is_last_rank", False):
        return None
    rank = getattr(getattr(holder, "ps", None), "pp_rank", None)
    if rank is None:
        return None
    raw = message.get(_ADMISSION_DECISION_PAYLOAD_KEY)
    first = (
        pp_first_retracting_rank(
            pp_admission_decision_from_wire({_ADMISSION_DECISION_PAYLOAD_KEY: raw})
        )
        if raw is not None
        else None
    )
    stop = pp_void_relay_stop_rank(
        first, getattr(getattr(holder, "ps", None), "pp_size", None)
    )
    if stop is None or int(rank) + 1 >= int(stop):
        return None
    return dict(message)


def pp_output_payload_with_return_trip(
    holder, payload: Dict[str, object], mb_id: int
) -> Dict[str, object]:
    """#797: put this slot's chain-reconciled decision on the output.

    THE ONE CHANNEL THAT SURVIVES #796'S LAW. `record_return_trip` is what
    `PPAdmissionCongruenceGuard` learns from, and #796 removed its only
    feeder when it deleted the ring wraparound -- correctly, because a
    rank must not post a send no peer is required to take. #791b restored
    HALF of it: the void output carries the decision back, so a floor is
    learned on a pass that produced no output at all. The other half was
    still missing, and it is the half that CLEARS a floor.

    WHY THE MISSING HALF MATTERS. `record_return_trip` learns on
    `retracted` and clears on `admitted and not retracted`; with only the
    void feeding it, no successful pass ever reaches it, so a rid that was
    once one token short keeps its clamp for the rest of its life (#796
    named that residual cost precisely, and this is what pays it back).
    The floor is what makes the retraction TERMINATE; the clearing is what
    keeps a single cold-cache moment from suppressing that request's
    prefix reuse for ever.

    NOT A NEW MESSAGE. This rides the output message the last rank already
    sends and PP0 is already required to receive, exactly as
    `_PP_OUTPUT_EXPECTED_KEY` rides the decision message. PER HOP, and
    that is not decoration: PP0 POPS the payload (`pp_absorb_admission_
    return`) before the dict becomes a `PPProxyTensors` and is forwarded
    on, because that class maps `v[key]` over EVERY entry and a tuple left
    in it would slice to nonsense rather than raise -- the same hazard
    `_pp_recv_proxy_tensors` pops `__stamp__` for.

    COPIES rather than mutates: the caller's dict belongs to a live
    `PPProxyTensors` on the sending rank, and adding a non-tensor entry to
    it would put that same hazard on the SENDER's copy.
    """
    carried = getattr(holder, "_pp_admission_amended_by_slot", None)
    decision = carried[mb_id] if carried and 0 <= mb_id < len(carried) else None
    if decision is None:
        return payload
    out = dict(payload)
    out.update(pp_admission_decision_to_wire(decision))
    return out


def pp_absorb_admission_return(holder, message: Dict[str, object]) -> bool:
    """#797: take the decision riding home on an output and learn from it.

    True iff one was there. POPS it, for the reason
    `pp_output_payload_with_return_trip` gives: what is left of this dict
    becomes a `PPProxyTensors` and is forwarded round the ring.

    Every guard is a getattr: a rank with no `_pp_admission_guard` (a
    stand-in, `pp_size <= 1`, a boot predating #630) simply drops the
    payload, which is the behaviour that shipped before this existed.
    """
    if not isinstance(message, dict):
        return False
    raw = message.pop(_ADMISSION_DECISION_PAYLOAD_KEY, None)
    if raw is None:
        return False
    guard = getattr(holder, "_pp_admission_guard", None)
    if guard is None:
        return False
    guard.record_return_trip(
        pp_admission_decision_from_wire({_ADMISSION_DECISION_PAYLOAD_KEY: raw})
    )
    return True


#: #947: env flip for the RELOCATED dead-premise actuator. Default OFF, and
#: that default is the whole discipline: five compensators in this family were
#: placed on paths the failure starves, and each cost a boot to discover. This
#: one ships INERT. The boot MEASURES the ring first (`pp_ring_note`), and the
#: actuator is flipped only once the markers show its site actually runs per
#: void iteration and lies before the follow-on pass is parametrised. Flipping
#: on suspicion would be placement number six with extra steps.
_ACT_AT_RING_ENV = "SGLANG_946_ACT_AT_RING"


def pp_act_at_ring_enabled() -> bool:
    """True iff the relocated actuator is armed. Read at the call site, never
    cached, so a boot can be compared against itself in a log."""
    return os.environ.get(_ACT_AT_RING_ENV, "") == "1"


def pp_ring_note(holder, site: str, voided: bool) -> None:
    """#947 THE VOID-ITERATION INSTRUMENT: counting only, no behaviour.

    THE QUESTION THIS ANSWERS, and it is the one five failed placements could
    not: not "did my compensator do the right thing when called" but "does the
    path it sits on RUN AT ALL under the failure it is meant to fix". The
    reachability ratchet cannot ask that -- it can only pin a path once the
    path is known -- so the ring gets measured instead of reasoned about.

    TWO OBSERVABLES, both per-iteration and both cheap:

      * `voids_between_admissions` -- how many passes void between two entries
        of the admission path. Boot window-946 could only offer the QUOTIENT
        (9471 voids / 6 prefill batches = ~1578), which cannot distinguish "a
        steady 1578 every time" from "one burst of 9000 and five healthy
        rounds". A quotient is not a curve, and the fix differs between those
        two shapes.
      * `site` -- which site was reached on THIS iteration. A site that never
        appears while voids climb is a site no actuator may be placed on,
        proven rather than assumed.

    Emitted on a cadence, never per line: this runs in the hot loop of a wedged
    instance and 9471 log lines is the defect it is investigating, in a new
    colour.
    """
    counts = getattr(holder, "_pp_ring_site_counts", None)
    if counts is None:
        counts = {}
        holder._pp_ring_site_counts = counts
    counts[site] = counts.get(site, 0) + 1
    if voided:
        holder._pp_ring_voids_run = getattr(holder, "_pp_ring_voids_run", 0) + 1
        return
    # A NON-VOIDED iteration is the "admission path was entered" event, so the
    # streak that ends here is exactly the quantity in question.
    run = getattr(holder, "_pp_ring_voids_run", 0)
    hist = getattr(holder, "_pp_ring_void_runs", None)
    if hist is None:
        hist = []
        holder._pp_ring_void_runs = hist
    if run:
        hist.append(run)
    holder._pp_ring_voids_run = 0
    holder._pp_ring_admissions = getattr(holder, "_pp_ring_admissions", 0) + 1
    every = int(os.environ.get("SGLANG_947_RING_EVERY", "25") or 25)
    if every > 0 and holder._pp_ring_admissions % every == 0:
        runs = list(hist)
        logger.warning(
            "#947 VOID-RING CENSUS: %d admission-path entries so far; void "
            "runs between them (most recent last, up to 20): %s; max=%s "
            "mean=%.1f; sites reached this boot: %s. A site absent from that "
            "map does NOT run under this failure and no actuator may be "
            "placed on it.",
            holder._pp_ring_admissions,
            runs[-20:],
            max(runs) if runs else 0,
            (sum(runs) / len(runs)) if runs else 0.0,
            dict(sorted(counts.items())),
        )


def pp_premise_probe(holder, kind: str, **fields) -> None:
    """#948: the three counters that separate why the armed actuator never acts.

    window-947 established what is NOT the answer. Both candidate sites run
    under the failure (`prefill:chunked_block` 4117, `ring:pre_plan` 12483), the
    escalation fires on exactly the frozen rid, and the actuator is armed -- yet
    `pp_premise_is_dead(self.chunked_req)` is never true at the ring. Three
    explanations survived that boot and it could not separate them, so each gets
    ONE counter rather than a fourth guess:

      mark_hit / mark_miss   (a) was the escalated rid in `can_run_list` when
                                 the marking loop ran? A miss means no mark was
                                 ever written.
      act_rid_mismatch       (b) at the ring, is `self.chunked_req` the request
                                 that was marked? A mismatch means the mark
                                 exists on a request the act never looks at --
                                 and `where=` then says which of the four places
                                 the marked rid actually lives in.
      gen_mismatch           (c) did the generation stamp move between mark and
                                 act? Counted WITH BOTH VALUES, because "they
                                 differ" without the numbers cannot distinguish
                                 a cutover from an unreadable stamp.

    ABSENT IS NOT ZERO, the same rule the #947 census carries: a kind that never
    fires does not appear in the map at all, so "this counter is missing" and
    "this counter measured zero" can never be confused. That distinction is what
    makes a null reading admissible evidence rather than a shrug.

    Rate-limited on the same pattern and for the same reason: this runs in the
    hot loop of a wedged instance, and 9471 log lines is the defect under
    investigation, not a licence.
    """
    counts = getattr(holder, "_pp_premise_probe_counts", None)
    if counts is None:
        counts = {}
        holder._pp_premise_probe_counts = counts
    counts[kind] = counts.get(kind, 0) + 1
    samples = getattr(holder, "_pp_premise_probe_samples", None)
    if samples is None:
        samples = {}
        holder._pp_premise_probe_samples = samples
    if fields and kind not in samples:
        # FIRST OCCURRENCE ONLY, kept verbatim. The counts say how often; one
        # concrete sample says what, and a sample per occurrence would be the
        # log flood this instrument exists to avoid.
        samples[kind] = dict(fields)
    total = sum(counts.values())
    every = int(os.environ.get("SGLANG_948_PROBE_EVERY", "500") or 500)
    if every > 0 and total % every == 0:
        logger.warning(
            "#948 PREMISE PROBE: counts=%s first-samples=%s. A kind ABSENT "
            "from counts did not occur at all -- that is not the same as zero, "
            "and it is the reading that separates the three candidates: "
            "mark_miss means no mark was written, act_rid_mismatch means the "
            "mark is on a request the act never inspects, gen_mismatch means "
            "the stamp moved between writing and reading.",
            dict(sorted(counts.items())),
            samples,
        )


def pp_request_locations(holder) -> Dict[str, object]:
    """#946: every place a request can live, as one rid -> req mapping.

    THE FOUR PLACES, and they are the SAME enumeration `_pp_reconcile_incoming_
    admission` already walks for #944: the waiting queue, `chunked_req`, the
    named slot's chunked req, and the running batch. Sharing the list rather
    than re-deriving it is the point -- a FIFTH place must not have to be
    discovered once per consumer, which is exactly how #797c, #798 and #944
    became three tickets for one class.

    WHY IT EXISTS AT ALL, measured. `scheduler.py`'s #943b re-issue built its
    candidate set as `{r.rid: r for r in self.waiting_queue}`. A chunked
    continuation is never in the waiting queue -- #797b deliberately RESTORES
    it as `self.chunked_req`, because that state outlives the round and putting
    it back is what stopped boot instr19 dying on `extend_range.end`. So the
    re-issue could not nominate the one request that was stuck (`PREFETCH
    RE-ISSUED` 0 across six boots), and the same request never re-entered
    through the waiting-queue path where `prefix_len_for` applies (the offer
    froze at `told=8192` for thousands of passes). ONE structural fact, both
    symptoms -- the fourth instance of the compensator-reachability class.

    EVERY LOOKUP IS A getattr (#787 convention), and that is load-bearing here
    rather than defensive: this mapping feeds `take_agreed_reissue`, which is a
    COLLECTIVE. A rank that raised while building its candidate set would skip
    an all_reduce its peers had already entered -- the deadlock family this
    whole feature is a list of. A holder that carries none of the fields yields
    an empty mapping, which the vote is already built to answer 0 for.

    WAITING QUEUE FIRST, so a request that is somehow in two places resolves to
    the one the admission loop would act on.
    """
    out: Dict[str, object] = {}
    for req in getattr(holder, "waiting_queue", None) or ():
        rid = getattr(req, "rid", None)
        if rid is not None:
            out.setdefault(rid, req)
    chunked = getattr(holder, "chunked_req", None)
    rid = getattr(chunked, "rid", None)
    if rid is not None:
        out.setdefault(rid, chunked)
    by_slot = getattr(holder, "_pp_chunked_req_before_by_slot", None) or {}
    try:
        slot_reqs = list(by_slot.values())
    except AttributeError:  # a stand-in that is not a mapping
        slot_reqs = []
    for req in slot_reqs:
        rid = getattr(req, "rid", None)
        if rid is not None:
            out.setdefault(rid, req)
    running = getattr(holder, "running_batch", None)
    for req in getattr(running, "reqs", None) or ():
        rid = getattr(req, "rid", None)
        if rid is not None:
            out.setdefault(rid, req)
    return out


#: #946: the attribute the dead-premise mark lives under. On the REQUEST and
#: not in a guard-side dict, because #797b restores the same object and the
#: rank that must ACT on the mark is not always the rank that saw the void.
_PREMISE_DEAD_STAMP = "_pp_premise_dead_generation"

#: #949: how many DECLINED re-fetches a request may collect before the escape
#: gives up on the cheap path and spends the expensive one. Bounded because an
#: unbounded retry is the #858 livelock shape; small because each round costs a
#: pass. The retry only exists at all now that a decline can be TOLD APART from
#: a success -- before window-946fix-0828 every decline was reported as
#: "refetch" and there was nothing to retry on.
_REFETCH_DECLINE_STREAK = "_pp_refetch_decline_streak"
REFETCH_DECLINE_CAP = 3


def pp_mark_premise_dead(req) -> None:
    """#946 DECIDE AT THE VOID: record that this request's prefix premise is
    dead, stamped with the binding generation it was decided under.

    STAMPED, AND THAT IS WHAT MAKES IT SAFE TO LEAVE ON A LIVE OBJECT. The mark
    must survive the #797b chunked restore (same generation -- the whole point)
    and must NOT survive a retract or a cutover, which move the binding. A
    stamp does both with no `reset_for_retract` special case and no cleanup
    path that could be forgotten: the reader compares generations and a
    superseded mark simply reads as absent. Same rule #911 uses for completion
    routing.

    DECIDED HERE, NOT ACTED ON HERE. The void fires AFTER this pass's geometry
    was committed and launched; changing the premise now would rewrite a pass
    in flight, which is the instr20 crash. The acting half is
    `pp_apply_dead_premise_at_chunk_boundary`, at the one point where nothing
    is committed yet.
    """
    if req is None:
        return
    from sglang.srt.mem_cache import hicache_phase_binding as _hpb

    try:
        setattr(req, _PREMISE_DEAD_STAMP, int(_hpb.current_generation()))
    except Exception:  # noqa: BLE001 - a mark may never break the void path
        pass


def pp_premise_is_dead(req) -> bool:
    """True iff `req` carries a dead-premise mark from the CURRENT generation.

    A mark from a superseded binding reads as absent, which is the
    self-invalidation `pp_mark_premise_dead` exists for.
    """
    if req is None:
        return False
    stamped = getattr(req, _PREMISE_DEAD_STAMP, None)
    if stamped is None:
        return False
    from sglang.srt.mem_cache import hicache_phase_binding as _hpb

    try:
        return int(stamped) == int(_hpb.current_generation())
    except Exception:  # noqa: BLE001 - unreadable generation means "not live"
        return False


def pp_apply_dead_premise_at_chunk_boundary(holder, req) -> str:
    """#946 ACT AT THE CHUNK BOUNDARY. Returns "none" / "refetch" / "recompute".

    THE ONE LEGAL POINT, and the criterion it satisfies is #944c's: the point
    must lie BEFORE the commitment it changes, never after. `add_chunked_req`
    derives everything from `len(req.prefix_indices)` and only THEN calls
    `set_extend_range`, so before that call no geometry for the next chunk
    exists and changing the premise cannot contradict a report -- the report
    has not been made. The previous chunk is finished or voided; nothing is in
    flight. `#906 / #890 hole 2` already gates this exact site and its own
    comment establishes the safety: the request stays `self.chunked_req`, is
    not dropped, and resumes mid-chunk unharmed.

    RE-FETCH IS PREFERRED AND IT IS A LAW, NOT A TASTE. `told=0` on the
    measured rid discards 8192 tokens -- two full `--chunked-prefill-size 4096`
    chunks -- against a standing user law that allows at most ONE chunk of
    loss. The store still holds the pages, so re-fetching under the current
    generation pays a fetch instead of a full recompute and keeps the premise.
    `told=0` is the bounded last resort for when no fetch can be issued, and it
    names the tokens it throws away: a law breach may be necessary, it may
    never be silent.

    CONSUMED BY AN OUTCOME, NOT BY AN ATTEMPT (#949, corrected after
    window-946fix-0828). The mark used to be cleared before the re-fetch was
    even tried, so a DECLINED re-fetch spent the mark and the request never got
    the escape again. That was invisible while every decline was reported as
    "refetch". Now: ISSUED clears it, the terminator clears it, and a decline
    KEEPS it and retries -- bounded by `REFETCH_DECLINE_CAP`, after which the
    terminator runs. The escape still cannot re-fire for ever, which is what
    the old blanket clear was protecting against; it simply no longer pays for
    that with a silently wasted mark.

    Returns "none" / "refetch" / "refetch-declined" / "recompute".
    """
    # #948 COUNTER (c): did the stamp move between mark and act? Counted with
    # BOTH values, because "they differ" without the numbers cannot tell a
    # cutover from an unreadable stamp.
    stamped = getattr(req, _PREMISE_DEAD_STAMP, None) if req is not None else None
    if stamped is not None and not pp_premise_is_dead(req):
        from sglang.srt.mem_cache import hicache_phase_binding as _hpb

        try:
            _now = int(_hpb.current_generation())
        except Exception:  # noqa: BLE001 - unreadable generation is itself the datum
            _now = None
        pp_premise_probe(
            holder,
            "gen_mismatch",
            rid=getattr(req, "rid", "?"),
            stamped=stamped,
            now=_now,
        )
    if not pp_premise_is_dead(req):
        return "none"

    # #949 THE MARK IS NO LONGER CONSUMED BEFORE THE ATTEMPT. It used to be
    # cleared here, unconditionally, so a re-fetch that was DECLINED still
    # burned the mark and the request was never offered the escape again --
    # invisible, because the decline was reported as "refetch". Now the mark is
    # spent only by an outcome that actually changed something: an ISSUED
    # re-fetch, or the terminator.
    prefetch = getattr(holder, "_prefetch_kvcache", None)
    if callable(prefetch):
        verdict = "declined:raised"
        try:
            verdict = prefetch(req)
        except Exception as exc:  # noqa: BLE001 - fall through to the terminator
            logger.warning(
                "#946 re-fetch could not be issued for rid=%s (%s); falling "
                "back to the recompute terminator",
                getattr(req, "rid", "?"),
                exc,
            )
        # A tree cache that predates the honest verdict returns None. Treat
        # that as UNOBSERVABLE, never as success -- reading a missing answer as
        # a good one is the exact defect this ticket closes.
        if verdict is None:
            verdict = "declined:unobservable"
        if verdict == "issued":
            try:
                delattr(req, _PREMISE_DEAD_STAMP)
            except AttributeError:
                pass
            try:
                delattr(req, _REFETCH_DECLINE_STREAK)
            except AttributeError:
                pass
            pp_premise_probe(holder, "refetch_issued", rid=getattr(req, "rid", "?"))
            return "refetch"

        streak = int(getattr(req, _REFETCH_DECLINE_STREAK, 0) or 0) + 1
        setattr(req, _REFETCH_DECLINE_STREAK, streak)
        reason = str(verdict).split(":", 1)[-1]
        pp_premise_probe(
            holder,
            "refetch_declined",
            rid=getattr(req, "rid", "?"),
            reason=reason,
            streak=streak,
        )
        if streak < REFETCH_DECLINE_CAP:
            # KEEP THE MARK and retry next round. The premise is still dead and
            # the cheap escape has not been shown impossible -- only unavailable
            # on this pass. Bounded by the cap directly below.
            logger.warning(
                "#946 REFETCH DECLINED rid=%s reason=%s streak=%d/%d: the "
                "dead-premise escape asked for a re-fetch and NONE WAS ISSUED. "
                "The mark is kept and the escape retries next round. If "
                "reason=rate_limited persists, the root is the #915 host-tier "
                "capacity asymmetry across the flip (prefetch_capacity_limit "
                "is 0.5 * mem_pool_host.size, cache_controller.py:841) and "
                "NOT this escape.",
                getattr(req, "rid", "?"),
                reason,
                streak,
                REFETCH_DECLINE_CAP,
            )
            return "refetch-declined"
        logger.warning(
            "#946 REFETCH DECLINED rid=%s reason=%s streak=%d/%d -- CAP "
            "REACHED, spending the terminator below.",
            getattr(req, "rid", "?"),
            reason,
            streak,
            REFETCH_DECLINE_CAP,
        )

    # The terminator consumes the mark: this branch does change something, and
    # a request that reaches it must not arrive again on the same premise.
    try:
        delattr(req, _PREMISE_DEAD_STAMP)
    except AttributeError:
        pass
    try:
        delattr(req, _REFETCH_DECLINE_STREAK)
    except AttributeError:
        pass
    # #955 AND THE OTHER HALF OF THE SAME LIFECYCLE, which the sentence above
    # ("must not arrive again on the same premise") asserted and nothing
    # enforced. Spending the request-side marks is correct and was never the
    # gap: the request is RE-MARKED from scheduler.py:9324-9326, which reads
    # `PPAdmissionCongruenceGuard.is_escalated` -- durable guard state this
    # function never touched. That flag was discarded at exactly one site, the
    # `elif entry.admitted:` branch of `record_return_trip`, i.e. only after a
    # completed ring round-trip; and the escalation's own consequence (this
    # terminator, and the voided pass it produces) is what prevents that
    # round-trip. So the only operation that could end the escalation was
    # unreachable from the path the escalation creates, and the escape ran
    # once every `REFETCH_DECLINE_CAP + 1` passes for the life of the
    # instance: 85 recomputes of ONE rid at 8192 tokens each in 14 s
    # (window-951-boot, 2/2 boots, byte-identical).
    #
    # TOLD, NOT ASKED. The guard is notified rather than consulted, and it
    # reads its own last offer for the value -- the scheduler passing a length
    # in would be a second expression for a quantity that already has one.
    #
    # EVERY LOOKUP A getattr (#787), and load-bearing rather than defensive
    # here: this function is called with stand-in holders by the registered
    # tests, and a holder that carries no guard must lose nothing but this
    # bookkeeping.
    _guard = getattr(holder, "_pp_admission_guard", None)
    _note_spent = getattr(_guard, "note_terminator_spent", None)
    if callable(_note_spent):
        try:
            _note_spent(getattr(req, "rid", None))
        except Exception:  # noqa: BLE001 - bookkeeping may never break the escape
            logger.warning(
                "#955 could not record the terminator spend for rid=%s; the "
                "recompute below still happens, but the escalation may re-arm",
                getattr(req, "rid", "?"),
            )
    # #796 CLASS, and this line is its third instance -- it killed all three
    # ranks in window-946rf-0828 the first time a boot ever REACHED the
    # terminator. `prefix_indices` is a torch tensor; `x or ()` asks `bool(x)`,
    # which torch refuses from two elements up. Fine for an empty prefix, fine
    # for one token, fatal for a real one -- which is why it survived every
    # earlier boot: the escape always returned "refetch" above it and this code
    # never ran. #949 made the terminator deliverable and the landmine went off
    # on the same pass.
    #
    # ONE DEFINITION. `prefix_len` exists because this exact crash already
    # happened at phase_flip_draft_bootstrap.py (W37-B, 2026-08-25) and
    # scheduler.py:8108 carries the same warning in a comment. #946 spelled a
    # third copy regardless. Use the canonical one rather than adding a fourth.
    from sglang.srt.managers.phase_flip_draft_bootstrap import (
        prefix_len as _prefix_len,
    )

    discarded = _prefix_len(req)
    logger.warning(
        "#946 PREMISE RECOMPUTE rid=%s: no re-fetch could be issued, so the "
        "dead prefix premise is dropped and %d token(s) will be recomputed. "
        "THIS IS A DOUBLE PREFILL and the standing law allows at most one "
        "chunk of loss -- it is spent here only because the bounded "
        "alternative is an unterminated re-offer. If this line is frequent, "
        "the re-fetch path is unreachable and THAT is the bug, not this "
        "fallback.",
        getattr(req, "rid", "?"),
        discarded,
    )
    truncate = getattr(req, "truncate_prefix_to", None)
    if callable(truncate):
        truncate(0)
    return "recompute"


def pp_apply_dead_premise_anywhere(holder) -> Dict[str, str]:
    """#948: apply the dead-premise escape wherever the marked request LIVES.

    THE FIX FOR THE ONE THING window-947 LEFT STANDING, and the desk separated
    it before any metal was spent: the actuator was correct and its ARGUMENT was
    wrong. The mark is written over `can_run_list`; the act read a single field
    (`self.chunked_req`). A request marked anywhere else was marked for ever and
    inspected never -- which is exactly why an ARMED actuator produced
    `#946 PREMISE RECOMPUTE` 0 while `UNRESOLVABLE` fired on the right rid.

    Proven hermetically rather than argued: handed `chunked_req` the actuator
    returns "none"; handed the request the mark is actually on, the SAME
    actuator returns "recompute". Same code, different argument.

    THE ENUMERATION IS NOT NEW. `pp_request_locations` already lists the four
    places a request can live, and this is its second consumer -- which is why
    it was factored out of the #943b candidate set instead of inlined there. A
    fifth place is now one edit for both consumers rather than a fifth ticket.

    Returns `rid -> outcome` for every request acted on, empty when nothing is
    marked. The empty case walks four short sequences and touches nothing,
    which is the default path on every healthy pass.
    """
    out: Dict[str, str] = {}
    for rid, req in pp_request_locations(holder).items():
        if getattr(req, _PREMISE_DEAD_STAMP, None) is None:
            continue
        outcome = pp_apply_dead_premise_at_chunk_boundary(holder, req)
        if outcome != "none":
            out[rid] = outcome
    return out


def pp_chunked_local_match(req) -> Optional[int]:
    """#797c: how much of THIS request this rank has already computed.

    THE FALSE NEGATIVE THAT CAUSED THE LIVELOCK. Boot instr19 retracted the
    same rid three times in one second with `told` GROWING by exactly one
    chunk each time and `local` pinned at 0:

        08:47:26 PP1 unhonourable prefix: rid=d4f59edf... told=512  local=0
        08:47:27 PP1 unhonourable prefix: rid=d4f59edf... told=1024 local=0
        08:47:27 PP1 unhonourable prefix: rid=d4f59edf... told=1536 local=0

    PP1 had computed that prefix. It admitted the request's FIRST chunk
    itself, congruently, one second earlier (`verdict=ADMIT n_reqs=1
    rids=d4f59edf... prefix_lens=0` on all three ranks). `local=0` was never a
    measurement: `_pp_reconcile_incoming_admission` looks a rid up in
    `self.waiting_queue`, scheduler.py drops every admitted request out of
    that queue (`self.waiting_queue = [x for x in self.waiting_queue if x not
    in can_run_set]`), and a CHUNKED request then lives in `self.chunked_req`
    instead -- so the lookup misses by construction on every round after the
    first, and the miss defaults to 0. The reconcile's own docstring rests on
    "not in this rank's waiting_queue is physically indistinguishable from
    'this rank's cache has nothing for it'", which is true for every request
    EXCEPT the one that is guaranteed to be mid-prefill.

    So the retraction was spurious, and with `--chunked-prefill-size 512`
    against ~17000-token prompts it is spurious for essentially every large
    request, on every round but its first. Before #797 that produced a
    narrowed pass -- a silent same-width mispair, which is very likely where
    the bulk of instr15/16/17's 661/1651/1718 events came from. After #797 it
    produces a voided pass, which is what instr19 measured.

    WHY `extend_range.end` AND NOT A RADIX RE-MATCH. `told` asks "how many of
    this request's leading tokens do you have KV for". For a chunked request
    that is exactly `extend_range.end` -- `Req.get_fill_ids` is
    `full_untruncated_fill_ids[:extend_range.end]`, so the range's end IS the
    absolute index this rank has computed up to. A radix match would answer a
    DIFFERENT question and answer it late: the completed chunk is not handed
    to the tree until the top of the next round (`stash_chunked_request`),
    which on a downstream rank has not run yet when this reconcile happens.
    The rank holds the KV in `req_to_token` either way; a number that says
    otherwise is the same false negative one stash later.

    CONGRUENT BY CONSTRUCTION, which is what makes the retraction stop firing
    rather than merely fire less. Every rank advances the chunked request
    through the same `add_chunked_req` sequence, so after every COMPLETED
    round every rank's `extend_range.end` is the same number; and a round
    that does not complete is parked on the ranks that prepared it
    (`_park_chunked_prefill_chunk`), which restores that same number. PP0's
    `told` is its own count of the same quantity, so `local >= told` holds --
    the predicate is an equality test between two ranks' progress on one
    request, not a cache-warmth guess.

    `None` when the request is not usable as a chunked source (no request, no
    range), which every caller reads as "nothing known", i.e. the behaviour
    that shipped before this function.
    """
    if req is None:
        return None
    end = getattr(getattr(req, "extend_range", None), "end", None)
    if end is None:
        return None
    return max(0, int(end))


def pp_chunked_req_for_slot(holder, mb_id) -> Optional[object]:
    """#798: the chunked request the NAMED microbatch slot carries, or None.

    A module-level function looked up through this module's globals at call
    time, like `pp_chunked_local_match` and `pp_void_keeps_request` beside it,
    and for the same two reasons. It is the single point at which the slot
    ring enters the admission reconcile, so it is the single point a can-fail
    proof can neuter to show the rest of the machinery depends on it; and a
    module global is neuterable WITHOUT every existing holder having to bind a
    new method, which a mixin method is not.

    Never raises and never guesses. A holder with no ring (a stand-in, a
    fixture, a boot predating the per-slot snapshot) and an `mb_id` outside it
    both answer None -- which is precisely the shipped behaviour, the lookup
    that existed before #798. Raising on the admission path would turn one
    defect into two on the path least able to afford it.
    """
    ring = getattr(holder, "_pp_chunked_req_before_by_slot", None)
    if not ring:
        return None
    try:
        slot = int(mb_id)
    except (TypeError, ValueError):
        return None
    if slot < 0 or slot >= len(ring):
        return None
    return ring[slot]


def pp_void_keeps_request(req, resident_rids, chunked_before) -> bool:
    """#797/#797b: is this batch member SCHEDULER-owned rather than round-owned?

    A voided pass hands its batch's requests back -- KV, mamba slot, req-pool
    row -- and re-queues them, which is right for a request the round itself
    admitted and wrong, in two different ways, for a request that outlives the
    round:

      * RESIDENT DECODE REQUEST (`running_mbs[mb_id]`). It keeps decoding from
        the very pages `_release_voided_request` would return to the
        allocator, so freeing them corrupts another request's cache, and
        re-queueing it puts it in the batch twice. The pass simply did not
        run; it decodes again next pass from the state it still holds.
      * THE CHUNKED REQUEST (`self.chunked_req`). `reset_for_retract` sets
        `extend_range = None` and the next round dereferences
        `self.chunked_req.extend_range.end` -- boot instr19, 53 seconds, all
        three ranks. It is also not in `waiting_queue` (``add_chunked_req``
        re-admits it from `self.chunked_req` directly), so appending it
        duplicates it, and resetting it discards the tree handles for every
        chunk already stashed. It is parked instead, by
        `_park_chunked_prefill_chunk`.

    A SEPARATE, NAMED PREDICATE for the same reason `pp_pass_should_void` is:
    it is the one thing a can-fail proof must be able to neuter on its own.
    Blinding it to False is exactly the disposal that shipped before these two
    guards existed, while the put-back and the park still run their own
    bodies -- so the proof shows which guard did the saving instead of
    reverting the lot.
    """
    if req is None:
        return False
    if chunked_before is not None and req is chunked_before:
        return True
    return getattr(req, "rid", None) in (resident_rids or ())


def pp_pass_should_void(
    amended: Optional[PPAdmissionDecision],
    rank: Optional[int],
    incoming_voided: bool,
) -> bool:
    """#797: must this rank run NOTHING for this pass?

    TWO WAYS IN, AND THEY ARE DIFFERENT FACTS. Either this rank performed a
    #791 retraction against the decision it received -- in which case its own
    batch would be a strict subset of the one its upstream already launched,
    which is the mispair -- or a rank BEFORE it did, and said so on the wire
    (`_PP_PASS_VOIDED_KEY`), in which case the pass is already off and this
    rank must not restart it with a decode batch of its own.

    ORed, NEVER CLEARED. A rank downstream of a retraction retracts nothing
    itself (`reconcile_pp_admission_decision` passes an already-retracted
    entry through verbatim, which is what `entries_retracted_by_rank` is
    `retracted_by_rank ==`-keyed for), so a rank that consulted only its own
    verdict would un-void the pass and be the only rank running it.

    A SEPARATE, NAMED PREDICATE because it is the one thing a can-fail proof
    has to be able to neuter WITHOUT taking `entries_retracted_by_rank` down
    with it -- that lookup also feeds #791c's receive guard, and a proof that
    blinds both cannot show which of the two did the preventing. Looked up
    through this module's globals at call time, like every other function
    here.

    `None` on either argument means "nothing known", the same convention
    `pp_proxy_pass_retraction_reason` and `pp_flip_epoch_of` use.
    """
    if bool(incoming_voided):
        return True
    if amended is None or rank is None:
        return False
    return bool(entries_retracted_by_rank(amended, rank))


def pp_pass_retraction_reason_of(holder, mb_id: int) -> Optional[str]:
    """``holder._pp_pass_retraction_reason(mb_id)``, or None if it has none.

    MODULE-LEVEL AND getattr-BASED BY THIS FILE'S OWN CONVENTION (#787), for
    the identical reason ``pp_flip_epoch_of`` above is: several methods here
    are bound UNBOUND onto stand-in holders that carry only what the method
    under test touched when the holder was written, and a holder written
    before #791c must keep working. "No accessor" reads as "nothing known
    against this pass", which is exactly the behaviour that shipped then.

    A free function rather than a mixin method because a holder that binds
    methods by NAME would not have the mixin method either -- which is the
    whole failure this avoids.
    """
    fn = getattr(holder, "_pp_pass_retraction_reason", None)
    return fn(mb_id) if fn is not None else None


def classify_armed_drain_message(msg, ran_mb_ids, epoch: Optional[int] = None) -> tuple:
    """``(action, kind, why)`` for one message taken while ARMED.

    #757. The armed drain used to be kind-BLIND and discard everything, which
    is corpse S: the upstream wire MULTIPLEXES the proxy forward and the output
    return, an output belongs to work launched BEFORE the arm, and eating one
    blocked PP1 for ever. Disabling the drain then left only the receive-side
    guard, and comp4 hit it under load:

        #631 PROXY LEFTOVER REFUSED: proxy mb_id=2 seq=151 rows=512 arrived
        while this rank is on mb_id=1 -- sent by an upstream that resumed
        while this rank was still armed

    Both failures are the same missing distinction, so this function makes it
    once and the drain applies it:

    * ``output`` -- ALWAYS stashed. It is owed to a real consumer that already
      looks in the inbox. Discarding it destroys a microbatch's results and
      strands every rank behind it. This is the corpse-S half.
    * ``proxy`` for a microbatch this rank NEVER RAN -- discarded. There is no
      batch it could pair with, now or later, so leaving it on the wire is what
      strands it and puts every later receive off by one. This is the #757 half.
    * ``proxy`` for a microbatch this rank DID run -- stashed. It was launched
      before the arm and is still owed. The design note flagged exactly this
      case as "the same question a second time"; the stamp answers it.
    * ``proxy`` from ANOTHER FLIP EPOCH -- discarded, and checked BEFORE the
      slot test above, because the slot test is what it defeats. The cutover
      rebuilds the slot ring (``init_pp_loop_state``), so a slot number from
      the previous epoch names nothing in this one however well it matches;
      believing it is the 2026-08-21 mispair. See
      ``pp_proxy_stamp_names_pass``. ``epoch=None`` (a caller with no runtime
      to ask) skips this test entirely, leaving the pre-#795 behaviour.
    * anything unstamped or of unknown kind -- stashed. An unidentifiable
      message is not evidence of a void pass, and discarding on absence of
      evidence is how the corpse-S class is re-entered.

    Pure and module-level so the decision is testable without a scheduler, a
    process group or a boot -- which is what the previous inline form was not.
    """
    if not isinstance(msg, dict):
        return (DRAIN_STASH, "default", "not a dict; unidentifiable, so kept")
    kind = msg.get("__msg_type__", "default")
    if kind != "proxy":
        return (DRAIN_STASH, kind, f"kind={kind} is owed to a real consumer")
    stamp = msg.get("__stamp__")
    if stamp is None:
        return (DRAIN_STASH, kind, "proxy carries no stamp; cannot prove it void")
    try:
        mb_id = int(stamp[0])
    except Exception:  # noqa: BLE001 - a malformed stamp proves nothing
        return (DRAIN_STASH, kind, "proxy stamp unreadable; cannot prove it void")
    stamp_epoch = pp_proxy_stamp_epoch(stamp)
    if epoch is not None and stamp_epoch is not None and stamp_epoch != int(epoch):
        # BEFORE the slot test below, because a cross-epoch slot number is
        # exactly what defeats it: the cutover rebuilt the ring this rank
        # indexes, so "this rank DID run slot N" is true of a DIFFERENT slot N.
        return (
            DRAIN_DISCARD,
            kind,
            f"proxy mb_id={mb_id} is from flip epoch {stamp_epoch} while this "
            f"rank is on epoch {epoch}; the cutover rebuilt the slot ring, so "
            f"no batch of this rank's can ever pair with it",
        )
    if mb_id in set(ran_mb_ids or ()):
        return (
            DRAIN_STASH,
            kind,
            f"proxy mb_id={mb_id} names a pass this rank DID run; still owed",
        )
    return (
        DRAIN_DISCARD,
        kind,
        f"proxy mb_id={mb_id} names a pass this rank never ran; void while armed",
    )


class SchedulerPPMixin:
    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        # #631: the phase-flip consensus must not run inside
        # get_next_batch_to_run here -- that is the TOP of the iteration,
        # before this rank's sends are issued, and a blocking world-
        # reduction there deadlocks against the pipeline p2p chain
        # (measured wedge 2026-08-08: two ranks in the reduction, one in
        # recv-from-prev). It runs at the END of each microbatch iteration
        # instead (all sends flushed -> every peer can reach its own
        # reduction of the same round). The flag is reset on ANY exit so a
        # post-flip event_loop_normal runs the hook inline again.
        self._defer_flip_round_to_pp_loop = True
        try:
            self._event_loop_pp_body()
        finally:
            self._defer_flip_round_to_pp_loop = False

    def _event_loop_pp_body(self):
        while True:
            server_is_idle = True
            # #631 DEFECT Q, CLOSED HERE. The slot index is a WHILE loop and
            # not a ``for`` because an armed, fully parked rank must HOLD it
            # -- see _pp_flip_hold_slot for the measurement and the argument.
            # With the flip disabled the two are the same loop: the hold is
            # never taken and mb_id increments once per iteration exactly as
            # ``for mb_id in range(self.pp_loop_size)`` did.
            mb_id = 0
            while mb_id < self.pp_loop_size:
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                self._pp_flip_pass_tick(mb_id)
                # #824 W4(b): honour a slot restore requested by the falling
                # edge above. Restart the body on that slot rather than
                # advancing, so this rank re-enters the pipeline where it
                # armed -- the same guarantee _pp_flip_hold_slot gives a
                # window long enough to reach it. Cleared before the jump,
                # and set only on a falling edge, so it fires once per
                # abandoned window and cannot loop.
                resume_slot = getattr(self, "_pp_flip_resume_slot", None)
                if resume_slot is not None:
                    self._pp_flip_resume_slot = None
                    if int(resume_slot) != mb_id:
                        mb_id = int(resume_slot)
                        continue
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.request_receiver.recv_requests()
                self._pp_forward_and_process_input_requests(recv_reqs)
                # #791 PP ADMISSION UNIFORMITY. Consume THIS pass's inbound
                # admission decision before this rank derives its own batch,
                # so the admission loop below (scheduler.py's
                # `_get_new_batch_prefill_raw`) can be DRIVEN by
                # `self._pp_admission_incoming_effective` instead of
                # independently re-deriving a verdict that might disagree
                # with PP0's. Unconditional -- never gated on this rank's
                # own local state, see `_pp_recv_admission_decision`'s
                # docstring -- and strictly BEFORE `get_next_batch_to_run` /
                # `_pp_recv_proxy_tensors` / `_pp_wait_for_proxy_readiness`
                # (#789), so an ordinary prefix-length divergence is
                # degraded here and the #789 contract's raise path stays
                # unfired on a healthy pass (the #791 task's ordering
                # requirement). PP0 has nothing to consume here -- it is the
                # one BUILDING the decision, inside the call below.
                self._pp_admission_incoming_effective = None
                # #791 CORE: reset on the same argument and in the same
                # breath as `effective` -- the two are one fact split in two,
                # and a pass that receives nothing must inherit neither half.
                self._pp_admission_incoming_schedule = None
                self._pp_admission_amended_to_forward = None
                # #791b: reset before the receive, so a pass that receives
                # nothing (pp_size <= 1, or the first rank) cannot inherit the
                # previous pass's expectation.
                self._pp_output_expected_incoming = False
                # #797: reset on the same argument, and on EVERY rank -- PP0
                # and `pp_size <= 1` never receive, so without this they would
                # carry a void decided passes ago into a pass that has no
                # retraction in it at all.
                self._pp_pass_voided_incoming = False
                self._pp_upstream_launched_incoming = False
                self._pp_admission_pass_voided = False
                # #951: reset in the same breath and for the same reason. This
                # one is written by scheduler.py's void guard DURING the pass
                # and read by `_pp_void_pass_without_upstream_launch` after it,
                # so its whole life is one pass; a stale True would let a rank
                # that is now idle keep incrementing the #801-spin streak it is
                # no longer earning.
                self._pp_upstream_void_withheld_work = False
                if self.ps.pp_size > 1 and not self.pp_group.is_first_rank:
                    with torch.profiler.record_function("pp_admission_decision_recv"):
                        incoming_decision = self._pp_recv_admission_decision()
                        effective, amended = self._pp_reconcile_incoming_admission(
                            incoming_decision
                        )
                        # #797 PREVENTION, and this is the line the whole
                        # change turns on. `effective` above is the pass
                        # NARROWED by this rank's own retraction, and a
                        # narrowed pass is precisely what may not be run: the
                        # upstream built and launched its batch from the
                        # decision as it stood BEFORE the retraction, so this
                        # rank's batch would be a strict subset of the one
                        # whose hidden states are already on the wire. #791c
                        # detects that pairing at the proxy boundary; this
                        # stops it being created. See
                        # `_pp_void_retracted_pass`.
                        effective, amended = self._pp_void_retracted_pass(
                            effective, amended
                        )
                        self._pp_admission_incoming_effective = effective
                        # #791 CORE: the SECOND number off the same decision.
                        # Taken from `amended` and not from the raw incoming
                        # decision, so a void empties it exactly when it
                        # empties `effective` -- the two can never name
                        # different rid sets.
                        self._pp_admission_incoming_schedule = (
                            self._pp_forwarded_schedule_from(amended)
                        )
                        self._pp_admission_amended_to_forward = amended
                        # #791b: record what PP0 said it will expect back for
                        # this slot, and what this rank is forwarding, BEFORE
                        # anything below can raise or return early. The last
                        # rank reads both again in
                        # `_pp_send_output_to_next_stage`.
                        self._pp_note_output_expectation(
                            mb_id, self._pp_output_expected_incoming, amended
                        )
                # #797b: the chunked request as it stands BEFORE this round's
                # admission can touch it. `get_next_batch_to_run` below is
                # where `add_chunked_req` advances it (a finished chunk clears
                # it, `adder.new_chunked_req` starts a new one), and a voided
                # pass has to put it back -- a rank downstream of the
                # retraction never ran this round at all, so ITS chunked state
                # is the pre-admission one and that is what every rank must
                # agree on. See `_pp_absorb_void_output`.
                self._pp_note_chunked_req_before_admission(mb_id)
                # #947 THE RING SITE, and the whole reason this instrument
                # exists. This line runs on EVERY iteration of the PP body --
                # voided ones included -- and `get_next_batch_to_run` directly
                # below is where the follow-on pass is parametrised. So it is
                # the one point that satisfies both halves of the criterion
                # five previous placements failed: it RUNS under the failure,
                # and it lies BEFORE the commitment it would change.
                #
                # Boot window-946 proved the previous act point does not: the
                # dead-premise escape sat inside `_get_new_batch_prefill_raw`'s
                # chunked block, which was entered ~6 times while 9471 passes
                # voided (`#946 PREMISE RECOMPUTE` 0 with the mark correctly
                # set on the frozen rid). Legal, and unreached.
                #
                # MEASURE BEFORE ACTING. The census below is counting only; the
                # actuator is env-gated and OFF by default, so this boot
                # answers "does this site run per void iteration" before
                # anything is moved onto it.
                pp_ring_note(
                    self, "ring:pre_plan", bool(self._pp_admission_pass_voided)
                )
                if pp_act_at_ring_enabled():
                    # #948 COUNTER (b): is the request the act inspects the one
                    # that was marked? The mark is written over `can_run_list`;
                    # the act reads ONE field. If a marked rid lives anywhere
                    # else, the mark is real and permanently invisible here.
                    # `where=` names which of the four places it actually is,
                    # using the enumeration #946 already built.
                    _ck = getattr(self, "chunked_req", None)
                    _marked = [
                        r
                        for r in pp_request_locations(self).values()
                        if getattr(r, _PREMISE_DEAD_STAMP, None) is not None
                    ]
                    if _marked:
                        _ck_rid = getattr(_ck, "rid", None)
                        _hit = any(getattr(r, "rid", None) == _ck_rid for r in _marked)
                        if _hit:
                            pp_premise_probe(self, "act_rid_hit", rid=_ck_rid)
                        else:
                            _where = {}
                            for _name, _src in (
                                ("waiting_queue", getattr(self, "waiting_queue", None)),
                                (
                                    "running_batch",
                                    getattr(
                                        getattr(self, "running_batch", None),
                                        "reqs",
                                        None,
                                    ),
                                ),
                            ):
                                for _r in _src or ():
                                    if getattr(_r, _PREMISE_DEAD_STAMP, None):
                                        _where[_name] = _where.get(_name, 0) + 1
                            pp_premise_probe(
                                self,
                                "act_rid_mismatch",
                                chunked_rid=_ck_rid,
                                marked_rids=[getattr(r, "rid", "?") for r in _marked][
                                    :3
                                ],
                                where=_where or "neither_queue_nor_running",
                            )
                    # #948 SWEEP THE FOUR PLACES, not one field. The mark is
                    # written over `can_run_list`; reading only `chunked_req`
                    # is why the armed actuator never fired on metal.
                    pp_apply_dead_premise_anywhere(self)
                with torch.profiler.record_function("get_next_batch_to_run"):
                    plan = self.get_next_batch_to_run(
                        running_batch=self.running_batch, last_batch=self.last_batch
                    )
                    self.running_batch = plan.running_batch
                    self.mbs[mb_id] = plan.batch_to_run
                self.running_mbs[mb_id] = self.running_batch

                # #797d: THIS RANK'S OWN VOID MUST REACH THE BATCH, NOT ONLY
                # THE ADMISSION DICT. `_pp_void_retracted_pass` above already
                # set `self._pp_admission_pass_voided`; `get_next_batch_to_
                # run` is expected to honour it on its own (scheduler.py's
                # `_pp_admission_pass_voided` guard), but its local
                # continuation logic (chunked_req, the resident running
                # batch) runs BEFORE that guard and can still leave
                # `self.mbs[mb_id]` non-empty -- see `_pp_void_own_batch`'s
                # docstring for the exact deadlock this closes
                # (SPECIMEN_wedge_19-02.txt). Placed strictly before the
                # admission-decision send below, so a voided slot's
                # `launched=self.mbs[mb_id] is not None` is never sent True,
                # and strictly before the `cur_batch = self.mbs[mb_id]`
                # branch further down, so a voided rank takes the drain
                # branch there instead of blocking in a proxy receive nobody
                # will satisfy.
                if self._pp_admission_pass_voided:
                    self._pp_void_own_batch(mb_id)

                # #798: AND THE MIRROR OF IT -- a pass this rank's UPSTREAM
                # did not launch cannot run here either. `_pp_drain_voided_
                # proxy` was gated on the sender's own `launched` statement;
                # the `if cur_batch:` RECEIVE branch below never was, so a
                # rank holding resident work while its upstream ran nothing
                # entered a blocking receive for a message nobody posted and
                # consumed the next pass's proxy instead -- the 2026-08-21
                # specimens, which are a surplus receive and not a slot-ring
                # divergence. Placed HERE, strictly before the admission
                # decision is forwarded below, so the `launched=` this rank
                # sends is already the voided value and the ranks below take
                # the same decision off the same per-hop fact. See
                # `_pp_void_pass_without_upstream_launch`.
                self._pp_void_pass_without_upstream_launch(mb_id)

                # #795 PP ADMISSION UNIFORMITY, RELOCATED: emit/forward this
                # pass's admission decision HERE, immediately after
                # `get_next_batch_to_run`, and strictly BEFORE `_pp_launch_
                # batch` below -- not after it, as #791 originally placed it.
                #
                # WHY THIS MOVED AGAIN, MEASURED (test_pp_admission_chain_
                # flush_deadlock_795.py). #791 already established "send
                # before the chain flush" and that fix is real and still
                # correct -- but it left this send positioned after `_pp_
                # launch_batch`, and under a gapped layer set (`--pp-stage-
                # ratio`) `_pp_launch_batch` is not merely local compute: it
                # is where the mid-forward CROSSING wire (pp_crossing_wire.py)
                # rendezvous-exchanges activations with peer ranks, on the
                # SAME `pp_group` typed-channel demultiplexer this decision
                # travels on (pp_crossing_wire.py:270-277). A rank that owns
                # a crossing TARGET (this rig: PP0, per #753's "crossings
                # from BOTH PP1 (after layer 31) and PP2 (after layer 35)")
                # blocks inside `_pp_launch_batch` waiting for a peer's
                # crossing SEND -- and that peer cannot reach its own
                # crossing send without first deriving a non-empty batch from
                # THIS rank's admission decision, which (before this move)
                # this rank had not sent yet, because the send was still
                # ordered after its own `_pp_launch_batch`. PP0 blocked
                # waiting on a peer that is blocked waiting on PP0: the
                # fifth deadlock of the "never block on a peer for something
                # not required for this iteration's forward progress" family,
                # one channel below the one #791 fixed, sharing its
                # demultiplexer. Reproduced hermetically (three real gloo
                # processes, the shipped `_pp_send_admission_decision` /
                # `_pp_recv_admission_decision` and the shipped
                # `send_typed_tensor_dict` / `recv_typed_tensor_dict` the
                # crossing wire itself uses) in test_fixed_ordering_with_
                # crossing_channel; test_send_before_launch_fixes_crossing_
                # deadlock proves this relocation removes it. No collective
                # is introduced -- still the same point-to-point decision
                # send/forward #791 built, just moved earlier. The decision
                # CONTENT is unchanged and already fully known at this point:
                # PP0's `self._pp_admission_last_built_decision` is built
                # inside `get_next_batch_to_run` above (scheduler.py:6896),
                # and downstream's `self._pp_admission_amended_to_forward`
                # was set even earlier, at line 621-629, before `get_next_
                # batch_to_run` even ran. Nothing below this point that used
                # to run before the old send site (`cur_batch` derivation,
                # proxy receive, `_pp_launch_batch`, output processing, the
                # proxy send) is read by this block or writes anything this
                # block reads, so moving it here changes WHEN the message
                # goes out, never WHAT it contains.
                if self.ps.pp_size > 1:
                    if self.pp_group.is_first_rank:
                        raw = getattr(self, "_pp_admission_last_built_decision", None)
                        self._pp_admission_last_built_decision = None
                        fresh_decision = (
                            replace(raw, mb_id=mb_id)
                            if raw is not None
                            else PPAdmissionDecision(mb_id=mb_id, entries=())
                        )
                        # #791b: PP0's own output-ring verdict for this slot,
                        # taken from the batch `get_next_batch_to_run` just
                        # installed and published on the decision so the last
                        # rank applies the SAME fact rather than re-deriving
                        # it from a slot a retraction may have emptied. It is
                        # the identical expression, on the identical object,
                        # that PP0's own `_do_recv` will apply to this slot
                        # `pp_size - 1` passes from now -- `self.mbs[mb_id]`
                        # is not rewritten again until this slot comes round.
                        expects_output = _pp_output_exchange_due(self.mbs[mb_id])
                        self._pp_note_output_expectation(mb_id, expects_output, None)
                        self._pp_send_admission_decision(
                            fresh_decision,
                            expects_output=expects_output,
                            # #797: PP0 builds the decision, so it can never
                            # be the rank that retracts against it -- the void
                            # can only ever start downstream of here.
                            pass_voided=False,
                            launched=self.mbs[mb_id] is not None,
                        )
                        self._pp_admission_pending_sends.append(mb_id)
                        # Capped, not unbounded -- see
                        # _PP_ADMISSION_PENDING_SENDS_CAP's docstring. What
                        # falls off here is only a bookkeeping marker for
                        # WHEN to opportunistically peek below; nothing on
                        # the wire is touched or forgotten by dropping it.
                        while (
                            len(self._pp_admission_pending_sends)
                            > _PP_ADMISSION_PENDING_SENDS_CAP
                        ):
                            self._pp_admission_pending_sends.popleft()
                        # Attempt the wraparound only once at least one full
                        # ring lap could plausibly have completed -- see the
                        # field's docstring in init_pp_loop_state -- and,
                        # critically, ONLY OPPORTUNISTICALLY: this used to be
                        # an unconditional BLOCKING receive here, which is
                        # the fourth deadlock of the "a rank must never block
                        # on a peer for something not required for this
                        # iteration's forward progress" family (see
                        # `_PP_ADMISSION_PENDING_SENDS_CAP`'s docstring
                        # above for the measured specimen).
                        # `record_return_trip` is a pure learning step
                        # (teaches `PPAdmissionCongruenceGuard` the coverage
                        # a lap observed; never required for this iteration
                        # to make progress), so the receive that feeds it
                        # must never be allowed to block this iteration on a
                        # peer. `_pp_try_recv_admission_decision` only
                        # consumes a lap that is ALREADY in hand and returns
                        # None immediately otherwise -- see its docstring.
                        if len(self._pp_admission_pending_sends) >= self.ps.pp_size:
                            with torch.profiler.record_function(
                                "pp_admission_decision_return_trip"
                            ):
                                returned = self._pp_try_recv_admission_decision()
                            if returned is not None:
                                self._pp_admission_pending_sends.popleft()
                                self._pp_admission_guard.record_return_trip(returned)
                            # else: no lap already in hand this pass -- skip
                            # it. `_pp_admission_pending_sends` is left at
                            # its current (capped) length, still >= pp_size,
                            # so every following pass retries this same
                            # cheap opportunistic check (one inbox peek, no
                            # wire touch) until a lap does show up. The
                            # skipped lap itself is never lost: see
                            # `_pp_try_recv_admission_decision`'s docstring
                            # for why it stays consumable later. The one
                            # user-visible consequence is that the learned
                            # floor `record_return_trip` clears LATER --
                            # whenever a lap next happens to already be in
                            # hand -- instead of at a fixed
                            # pass; this only delays reuse recovery for a
                            # retracted rid, nothing else.
                    else:
                        amended = self._pp_admission_amended_to_forward
                        if amended is None:
                            amended = PPAdmissionDecision(mb_id=mb_id, entries=())
                        # #791b: forwarded VERBATIM. This rank is not entitled
                        # to an opinion about the ring -- PP0 owns the verdict,
                        # every rank in between only carries it.
                        self._pp_send_admission_decision(
                            amended,
                            expects_output=self._pp_output_expected_incoming,
                            # #797: the void is OR-ed, never cleared -- a rank
                            # downstream of a retraction must not be able to
                            # un-void a pass by having retracted nothing of its
                            # own. `launched` is the opposite: it is THIS
                            # rank's own slot, overwritten for the next hop.
                            pass_voided=self._pp_admission_pass_voided,
                            launched=self.mbs[mb_id] is not None,
                        )

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors(mb_id)
                else:
                    # #797: a voided pass has no batch, so the branch above
                    # never runs -- and the upstream, which voided nothing,
                    # has already posted its proxy isend. Take it and drop it,
                    # or the upstream blocks for ever on next pass's
                    # `_pp_commit_comm_work(self.send_proxy_work)`.
                    self._pp_drain_voided_proxy(mb_id)
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                # #631 defect R: OUTSIDE the block above, deliberately. See
                # _pp_record_slot_last_batch -- nesting this under "did the
                # slot run something" is the resident-carry leak.
                self._pp_record_slot_last_batch(next_mb_id)
                if not self.pp_group.is_last_rank:
                    if cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                                stamp=self._pp_proxy_stamp(mb_id, result),
                            )

                self.pp_outputs = next_pp_outputs

                # #788: flush THIS pass's request-chain send only after
                # everything else this rank owed its peers this iteration is
                # already posted -- proxy isend, output commit, AND the #791/
                # #795 admission decision (sent earlier THIS pass, right
                # after `get_next_batch_to_run`, strictly before `_pp_launch_
                # batch` -- see the comment there for why it moved). This is
                # the last outbound act of the iteration precisely because it
                # is the one that blocks on a peer reaching the top of its
                # next pass; anything a peer might still be waiting on must
                # already be on the wire by the time we get here. See
                # _pp_commit_pending_req_work and the #788 send site in
                # _pp_forward_and_process_input_requests for why the old
                # top-of-pass position closed a cycle. Guarded the same way
                # the send site is: the last rank never posts a chain send,
                # so it has nothing to flush here either.
                # #796: reap this pass's admission-decision send BEFORE the
                # chain flush. Retaining the handle is what makes the
                # decision actually reach the peer at all (see
                # _pp_send_admission_decision); reaping it here, rather than
                # never, is what keeps the channel from growing an
                # ever-longer list of un-waited isends. Ordered before the
                # chain flush because it is the strictly weaker wait -- see
                # _pp_commit_admission_send_work's docstring.
                self._pp_commit_admission_send_work()
                if not self.pp_group.is_last_rank:
                    self._pp_commit_pending_req_work()

                # #631 phase-flip round hook, deferred from
                # get_next_batch_to_run: every send of this iteration is
                # flushed above (output dict committed, proxy isend issued,
                # request forward sent at the top), so the bounded
                # consensus is now the LAST blocking op of the iteration
                # and cannot close a cycle with a peer's pending recv. A
                # commit raises PhaseFlipLoopExit here -- a quiescent
                # boundary by construction (ready_fn gates on drained
                # microbatches).
                if self.server_args.enable_phase_flip:
                    self._phase_flip_on_round(require_armed_and_parked=True)
                    # #631 DEFECT Q. Do NOT advance the slot while the armed
                    # window is running dry: every rank must re-enter the
                    # pipeline on the slot it left it on.
                    if self._pp_flip_hold_slot():
                        continue

                # #753: THE LOCKSTEP THE GAPPED LAYOUT REQUIRES, MADE EXPLICIT.
                #
                # A gapped forward is not a pipeline. Every stage owns layers
                # INSIDE every other stage's span, so a pass can only be
                # computed with all three ranks inside it at once, handing the
                # activation back and forth. The crossings enforce that WITHIN
                # a pass -- each one is a rendezvous -- but nothing enforced it
                # ACROSS passes, and the stages do not finish together: PP1's
                # last owned layer is 31 and PP0's is 62, so both leave the
                # forward well before PP2 computes 63. Whoever leaves first
                # reaches the next pass first.
                #
                # Four boots died of that drift, each one stage further along:
                # v7pp5/v7pp6 with PP0 already at layer 4 of the next pass
                # while the others sat in the output receive, v7pp9 with all
                # three flushing sends nobody had posted a receive for, and
                # v7pp10 with PP0 a whole pass ahead, blocked in a crossing
                # SEND to peers that were still in the previous pass's output
                # exchange. Each fix removed one way to drift; none removed
                # drift itself, because the loop simply has no statement that
                # the passes are collective.
                #
                # This is that statement. One barrier per iteration, on the
                # path that already cannot overlap passes, so it costs a
                # rendezvous the layout was paying for anyway -- and it makes
                # the boundary a place where a peer cannot be a pass behind.
                if self._pp_gapped_wire:
                    with torch.profiler.record_function("pp_gapped_lockstep"):
                        self.pp_group.barrier()

                mb_id += 1

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        self.init_pp_loop_state()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self._pp_forward_and_process_input_requests(recv_reqs)

                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk(
                    last_batch=self.last_batch, running_batch=self.running_batch
                )
                prefill_plan = self.get_new_batch_prefill(self.running_batch)
                batch = prefill_plan.batch_to_run
                self.running_batch = prefill_plan.running_batch
                batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors(mb_id)

                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if bmbs[next_mb_id] is not None:
                    next_consensus_bootstrapped_rids = (
                        self._pp_recv_pyobj_from_prev_stage()
                    )
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                            stamp=self._pp_proxy_stamp(mb_id, result),
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        self.init_pp_loop_state()

        # PD additional state initialization
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.request_receiver.recv_requests()
                self._pp_forward_and_process_input_requests(recv_reqs)

                # reaching consensus through PP ranks
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work)

                # get batch to run and proxy tensors if needed
                plan = self.get_next_disagg_decode_batch_to_run(
                    running_batch=self.running_batch
                )
                self.running_batch = plan.running_batch
                batch = plan.batch_to_run
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors(mb_id)

                # early send output if possible
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)

                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch and not cur_batch.forward_mode.is_prebuilt():
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                            stamp=self._pp_proxy_stamp(mb_id, result),
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.on_idle()

    def _pp_forward_and_process_input_requests(
        self: Scheduler, recv_reqs: List
    ) -> None:
        """Forward PP requests before running handlers that may block on peers.

        Ported from upstream sgl-project/sglang#33934 (task #633).

        The order matters and is not cosmetic. Some control requests block
        until every rank joins -- ``InitWeightsUpdateGroupReqInput`` waits for
        the new process group to come up. If a stage runs its LOCAL handler
        before forwarding the request onward, it starts waiting for peers that
        have not been told to join yet, and the downstream stage never sees
        the request because its upstream is parked inside the handler. That is
        a circular wait between adjacent stages, and it hangs the boot rather
        than failing it.

        Forwarding first costs nothing (the send is async and its completion
        is committed on the next pass) and removes the cycle: every stage has
        the request in flight before any stage blocks on it.
        """
        # #631 ARM-DELIVERED-THEN-SAME-PASS-JOIN.
        #
        # THE INVARIANT, in three parts:
        #  (i)  a rank that receives or originates a flip arm must arm AND
        #       reach the consensus reduction within the SAME pass, never
        #       returning to the chain recv in between;
        #  (ii) before joining, a rank that owes an arm forward must have
        #       THAT SPECIFIC send completed -- a targeted commit of the
        #       one work handle, never a blanket synchronous send;
        #  (iii) the last stage owes no forward and joins directly.
        #
        # WHY THE BLOCKING REDUCTION IS THEN SAFE, by induction: when rank
        # k enters the reduction, (ii) guarantees rank k+1 already HAS the
        # arm in its recv buffer. Rank k+1 processes it at the top of its
        # current or next pass and, by (i)+(ii), arms, completes its own
        # forward and joins within that pass. No rank inside the reduction
        # is owed anything by a rank outside it except the join itself,
        # which arrives by induction. The worst case is LATENCY (a peer
        # mid-prefill-chunk finishes its pass first), never a deadlock.
        #
        # Each clause is a measured failure, not a precaution:
        #  * omitting (ii) is variant A -- rank 0 issued an async forward,
        #    armed, and blocked in the reduction before the send was ever
        #    progressed (_pp_commit_comm_work runs at the TOP of the next
        #    pass, which never came). py-spy: rank 0 in bounded_collective,
        #    ranks 1-2 in _pull_raw_reqs, 0 % GPU.
        #  * deferring the arm by a pass to satisfy (ii) breaks (i) and is
        #    a GUARANTEED miss: the downstream stage must re-enter the
        #    chain recv to reach the pass where it acts, and that recv
        #    blocks because upstream is already inside the reduction
        #    (boot 12: all three ranks armed, cutovers=0, dead at 40 s).
        #  * satisfying (ii) with async_send=False is variant B and
        #    deadlocks against the HIDDEN-STATES exchange, because a peer
        #    need not be in the chain recv at all. Hence "targeted": commit
        #    the request-chain handle only, leaving every other channel
        #    async.
        #
        # This applies to the MANUAL flip too, not only the policy's. It is
        # strictly safer, and it removes manual's latent at-idle deadlock:
        # manual has only ever been exercised UNDER TRAFFIC, where the loop
        # keeps cycling and the commit happens naturally.
        carries_flip_arm = bool(recv_reqs) and any(
            isinstance(r, PhaseFlipReqInput) for r in recv_reqs
        )

        if not self.pp_group.is_last_rank:
            if self.pp_phase_flip_armed():
                # #631 THE BOOT-18 FIX, downstream half. This commit is
                # the ORDINARY top-of-pass flush of the previous pass's
                # forward, and it is where rank 1 was found blocked while
                # rank 0 sat in the consensus reduction. It blocks
                # whenever the downstream stage has stopped consuming --
                # which is exactly what that stage does once IT is armed
                # and waiting at the gate. Because this point PRECEDES
                # the gate, no gate could ever have covered it.
                #
                # An armed rank issues no new forward -- it is admitting no
                # new work, so it has nothing to forward -- and reaps the
                # outstanding one through the SERVICE TURN instead: consume
                # whatever the upstream posted, then flush this rank's own
                # send once the downstream's counter proves it is consumed.
                #
                # This line used to call pp_pump_send_req_work, which
                # reaped NOTHING on this build (corpse F): is_completed()
                # never fires for an isend here, so send_req_work stayed
                # non-empty, presence was withheld for ever, and every flip
                # abandoned at the presence deadline. The service turn is
                # the same intent with a predicate the transport can
                # actually honour. Presence is still withheld until the
                # handle is reaped, so the flag still means "my chain is
                # flushed" (clause (i), PhaseFlipRuntime) -- it is now a
                # condition that can be REACHED.
                self.pp_flip_service()
            else:
                # #788: THIS LINE IS UNCHANGED, AND THAT IS THE POINT.
                #
                # The wedge was this commit blocking BEFORE the rank had
                # posted anything else it owed its peers for the iteration:
                # the proxy tensor-dict send and the output-ring send both
                # come later in _event_loop_pp_body, so a downstream whose
                # progress needs one of them was waiting on a rank that was
                # itself waiting on that downstream. Specimen WEDGE_788: two
                # ranks parked in this exact commit while the last rank
                # waited on a proxy neither had sent yet.
                #
                # The fix is NOT to move this call. It is to make it a
                # no-op in the loop where the hazard lives, by flushing the
                # handle at the END of the iteration instead -- see
                # _pp_commit_pending_req_work and its call site in
                # _event_loop_pp_body. By the time that loop returns here,
                # send_req_work has already been waited on and cleared, so
                # this commit finds an empty list.
                #
                # Leaving the call in place is what keeps the other two
                # callers -- event_loop_pp_disagg_prefill (:618) and
                # event_loop_pp_disagg_decode (:765), neither of which runs
                # the end-of-iteration commit -- on exactly their previous
                # behaviour, and it is what keeps the #633 ordering contract
                # this function's docstring documents literally intact:
                # commit, then forward, then process, with send_req_work
                # holding THIS pass's handle on return. Staging the send
                # into a second slot instead broke that contract and the
                # test that pins it (test_scheduler_pp_request_order_633).
                #
                # #753's a7ff250dc8 made the same "flush after the exchange,
                # not before it" move for the OUTPUT channel. This is that
                # fix for the REQUEST-CHAIN channel, and it is not gated to
                # the gapped layout: the hazard is not gapped-specific.
                self._pp_commit_comm_work(self.send_req_work)
                with torch.profiler.record_function("send_reqs_to_next_stage"):
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs,
                        async_send=True,
                    )
            # NOTE: no blocking commit here, deliberately. Committing the
            # arm-carrying send in-pass is corpse B' (boot 13): this rank
            # blocked in _pp_commit_comm_work while its peers sat in the
            # HIDDEN-STATES exchange, because "the peer is waiting for the
            # arm" is false -- it may be in another channel entirely. The
            # arm forward is instead PUMPED non-blockingly by the armed
            # poll loop (PhaseFlipRuntime._await_group_presence), which
            # delivers it without this rank blocking on anything.

        # (i): arm in this same pass; the flip hook at the end of this
        # microbatch iteration then joins without an intervening recv.
        self.process_input_requests(recv_reqs)

    def _pp_commit_pending_req_work(self: Scheduler) -> None:
        """#788: flush the outstanding request-chain send from the END of
        the iteration, not the top of the next pass.

        Called once per iteration from ``_event_loop_pp_body``, after the
        proxy send and the output commit, before the phase-flip round hook.
        That position is load-bearing, not incidental: everything this rank
        owed a peer for this pass is already posted by the time this blocks,
        so the wait here can only be on a peer that has no reason left to be
        waiting on THIS rank -- the same property the round hook's own
        comment (a few lines below this call site) already relies on for
        the collective it runs.

        IT FLUSHES ``send_req_work`` ITSELF, and introduces no second slot.
        Staging the send into a separate ``_pp_pending_req_work`` was the
        first shape of this fix and it was wrong twice over: it broke the
        #633 ordering contract (``send_req_work`` must hold THIS pass's
        handle on return from _pp_forward_and_process_input_requests, pinned
        by test_scheduler_pp_request_order_633), and it left the two
        disaggregation loops -- which never call this method -- overwriting
        a live Work handle every pass. With no second slot there is nothing
        to leak and nothing for a scheduler stand-in to grow a field for.

        The list is safe to wait on unconditionally: ``_pp_commit_comm_work``
        clears it, an armed pass reaps it through ``pp_flip_service``
        instead, and ``init_pp_loop_state`` seeds it empty -- so on any pass
        that posted nothing this is a wait on an empty list, and the commit
        at the top of the next pass then finds it already cleared -- which
        is precisely what makes that older call site harmless rather than
        the place the group deadlocks.
        """
        self._pp_commit_comm_work(self.send_req_work)

    def pp_pump_send_req_work(self: Scheduler) -> None:
        """#631: MEASURED DEAD. Reaps nothing on this build, ever.

        The intent was to progress the outstanding request-chain send
        without blocking, by polling the handle and reaping it once it
        reported complete. Measured 2026-08-08: ``is_completed()`` NEVER
        fires for an isend here -- not even after the peer has fully
        consumed the message -- so this never clears ``send_req_work``.
        The only thing that has ever reaped a chain send is the BLOCKING
        ``_pp_commit_comm_work``. Pinned by
        test_measured_the_send_side_pump_can_never_reap.

        Consequence worth stating plainly, because several design notes
        assumed otherwise: arms reach downstream stages via those stages'
        own blocking chain recv -- the recv side's wait() is what
        progresses the transfer -- never because an armed rank pumped the
        arm forward while it waited.

        Kept rather than deleted: it is harmless (it cannot mutate state),
        it is where a working predicate would go if the transport ever
        gains one, and deleting it would silently erase the record of what
        was tried. It is NOT a mechanism anything may rely on. A blocking
        commit here would be corpse B'; an unpumped async send is corpse A.
        """
        works = getattr(self, "send_req_work", None)
        if not works:
            return
        try:
            pending = []
            for w in works:
                handle = getattr(w, "work", w)
                done = getattr(handle, "is_completed", None)
                if done is None or done():
                    continue
                pending.append(w)
            if not pending:
                self.send_req_work = None
        except Exception:  # noqa: BLE001 - pumping is strictly best effort
            return

    def pp_phase_flip_armed(self: Scheduler) -> bool:
        """#631: does the ARMED forward rule apply on this rank?

        False on every boot without the flip, so the default PP path keeps
        its exact shape.

        It used to require a chain receiver as well, because the armed
        rule was only safe while something still CONSUMED the chain and
        the poll-based consumer was dead. THAT CONDITION WAS ALSO A BUG
        (#631 G): rank 0 has no upstream and therefore no receiver, so the
        armed rule was off on exactly the rank that most needs it -- the
        intake rank kept admitting new work while armed and could never
        drain to quiescence under load. The service turn is safe on every
        rank now: on rank 0 its consume half is a no-op and its flush half
        is real.
        """
        if not self.server_args.enable_phase_flip:
            return False
        return self.phase_flip_is_armed()

    def pp_owes_chain_send(self: Scheduler) -> bool:
        """#631 clause (i): does this rank still owe a request-chain send?

        The last stage never forwards, so it owes nothing and announces
        immediately. Everyone else owes until the pump has reaped the
        handle -- and ``send_req_work`` is precisely the handle the
        ordinary top-of-pass commit would have blocked on, which is what
        makes emptying it the right definition of "flushed".
        """
        return bool(getattr(self, "send_req_work", None))

    # -- #631 G: the ARMED SERVICE LOOP ----------------------------------
    #
    # THE DEFECT (measured 2026-08-09 00:06-00:09Z, corpse G): a rank that
    # became quiescent and spun at the gate stopped issuing its per-pass
    # chain forward. Its downstream reached the hook ONLY by returning
    # from a blocking chain recv that this very forward satisfied, so the
    # first rank to quiesce -- rank 0, the intake rank, always -- prevented
    # every rank behind it from ever becoming ready. Bounded (the spinner
    # abandoned at its deadline) but NOT convergent: the same rank drained
    # first every epoch, so the starvation reproduced identically.
    #
    # THE FIX is not to resume sending -- an armed rank has nothing to
    # forward and the unconsumed sends would pile up. It is to stop the
    # downstream from NEEDING that forward: while armed, a rank does not
    # block on any inbound channel, it SERVICES them. It reaches the hook
    # by its own poll, so no rank's readiness depends on a peer's traffic.
    #
    # WHY THIS IS NOT THE BOUNDED-RECV CORPSE (it will look like one).
    # That corpse completed iterations WITHOUT consuming while the
    # upstream kept sending: the rates decoupled, unmatched sends
    # accumulated, and the SENDERS blocked. Two differences, each
    # sufficient on its own. (1) This loop is GREEDY -- it consumes every
    # message the sender's counter accounts for and never skips an
    # available one. (2) It exists ONLY in the armed state, where
    # admissions are held and armed upstreams issue no new forwards, so
    # the accumulation driver is absent by construction rather than by
    # tuning.
    #
    # Spin-at-the-hook is this same loop with the gate already open, which
    # is why there is one mechanism here and not two.

    # -- #631 defect Q: THE ARMED WINDOW HAS NO PASS CLOCK ---------------
    #
    # THE INSTRUMENT, and the hypothesis it exists to KILL OR CONFIRM.
    #
    # What normally holds the PP stages in phase is not a shared counter --
    # there is none. ``mb_id`` is the inner loop index of
    # ``_event_loop_pp_body`` and is PURELY RANK-LOCAL. The synchroniser is
    # the BLOCKING chain receive: a rank cannot begin its next slot
    # iteration until its upstream has sent one message, so pass counts
    # cannot drift. Every rank then runs the same deterministic
    # ``get_next_batch_to_run`` over the same request stream and the stages
    # build the same batches -- a property this tree RELIES ON and, as of
    # the audit on 2026-08-09, states nowhere and asserts nowhere.
    #
    # THE ARMED INTAKE RULE REMOVES THAT CLOCK. While a flip is armed,
    # ``request_receiver._pull_raw_reqs`` returns ``[]`` immediately on
    # every rank -- rank 0 does not read zmq, and rank k>0 does not take
    # the blocking ``chain_receiver.recv()``. Nothing blocks, so for the
    # whole armed window (up to the 30 s park deadline) each rank
    # free-runs its slot loop at its own rate. If the ranks run different
    # numbers of passes in that window, their ``mb_id`` phase relationship
    # is destroyed, and NOTHING re-establishes it when the flip ABANDONS
    # and the ranks return to the ordinary loop.
    #
    # WHY A COMMIT IS SAFE AND AN ABANDON IS NOT -- the asymmetry the
    # specimen shows and this hypothesis explains. A committed flip
    # re-forms the topology and rebuilds the loop state
    # (``init_pp_loop_state``), so the phase is reset by construction. An
    # abandon returns to the SAME loop with drifted counters.
    #
    # WHAT IT COSTS WHEN IT GOES WRONG (specimen
    # /spinning/evidence-631/oom_and_abandon_20260809T0521Z, 05:21:16Z):
    # a microbatch's hidden states meet a DIFFERENT microbatch's batch. On
    # that boot a 2048-row chunked-prefill chunk met a 1-request decode
    # batch's cache indices, and one kernel BEFORE the guarded
    # ``packed_decode`` that pair reaches ``causal_conv1d_update``, whose
    # batch-vs-indices assert sits behind ``validate_data`` (default
    # False). It launched 2048 programs against a 1-element index tensor:
    # an out-of-bounds READ of the indices and an out-of-bounds WRITE into
    # conv_state. That is the "illegal memory access" the abandon path was
    # blamed for. It surfaced a second later in barlink's BAR1 status poll
    # only because a sticky CUDA fault reports at the next synchronising
    # call -- the poll is where it SHOWS, never where it happens.
    #
    # THE MEASUREMENT. Count slot iterations per rank across one armed
    # window and publish the count where peers can read it. If the
    # hypothesis is right the three counts differ, and the spread is the
    # phase error. If they agree, the hypothesis is dead and the
    # divergence is somewhere else -- which is exactly why this is an
    # instrument and not yet a fix.
    #
    # A NOTE FOR WHOEVER FIXES IT, so the instrument is not mistaken for
    # the remedy: restoring a per-pass chain message while armed is NOT
    # obviously the answer -- presence is withheld while a rank owes a
    # send, so a forward every pass could withhold presence every round.
    # The likelier shape is that an armed rank must not ADVANCE its slot
    # loop while it is doing no pipeline work.

    def _pp_flip_pass_tick(self: Scheduler, mb_id: int) -> None:
        """One slot iteration. Counts only while a flip is armed.

        Zero cost on a boot without the flip: it returns on the server-args
        test before touching anything. While armed it writes one small
        /dev/shm file per pass, which is the same discipline and the same
        directory the presence gate already uses.
        """
        # #824 W4a: the slot this rank is CURRENTLY on, published for the
        # chain-stall drain, which needs a live mb_id to tell an owed proxy
        # from a leftover one. Set before the enable test so it is accurate
        # for every caller that can reach it.
        self._pp_live_mb_id = mb_id
        if not getattr(self.server_args, "enable_phase_flip", False):
            return
        try:
            armed = self.pp_phase_flip_armed()
        except Exception:  # noqa: BLE001 - an instrument may never break the loop
            return

        passes = getattr(self, "_pp_flip_armed_passes", None)
        counters = getattr(self, "pp_flip_counters", None)

        if armed:
            if passes is None:
                self._pp_flip_armed_passes = 0
                self._pp_flip_arm_mb_id = mb_id
                # #829: WHICH RING THAT SLOT NUMBER BELONGS TO, recorded at
                # the same instant as the slot itself. A slot index is an
                # index into a ring of `pp_loop_size` slots that a cutover
                # rebuilds from zero, so on its own it is not a name -- the
                # same argument `_pp_proxy_stamp` makes for proxies (#795),
                # applied to the one other place a slot number outlives the
                # iteration that produced it. Read through
                # `pp_flip_epoch_of` so a holder with no accessor keeps its
                # pre-#795 behaviour.
                self._pp_flip_arm_epoch = pp_flip_epoch_of(self)
            else:
                self._pp_flip_armed_passes = passes + 1
            if counters is not None:
                try:
                    counters.publish_gauge(CHAN_PASS, self._pp_flip_armed_passes)
                    # THE QUANTITY THE PIPELINE ACTUALLY PAIRS ON. The pass
                    # count above measures the divergence; this one is the
                    # thing that must NOT diverge, and publishing it is what
                    # lets the falling edge below return a verdict instead of
                    # a number nobody can act on.
                    counters.publish_gauge(CHAN_SLOT, mb_id)
                except Exception:  # noqa: BLE001
                    pass
            return

        # Falling edge: the flip committed or -- the interesting case --
        # abandoned. Report the whole group's pass counts from ONE rank,
        # because correlating three log streams by timestamp is how the
        # last two diagnoses in this feature lost a boot each.
        if passes is None:
            return
        self._pp_flip_armed_passes = None
        arm_mb = getattr(self, "_pp_flip_arm_mb_id", None)

        # #824 W4(b): AN ARMED WINDOW THAT RAN NO ITERATIONS MUST NOT MOVE
        # THE SLOT. This is the case the note above predicted -- "an armed
        # rank must not ADVANCE its slot loop while it is doing no pipeline
        # work" -- and boot_827 is its measurement:
        #
        #   rank 0 ran 0 slot iteration(s) (armed at mb_id=0,
        #                                   disarmed at mb_id=1)
        #
        # _pp_flip_hold_slot is what normally guarantees every rank resumes
        # on the slot it armed on, but it only engages once the armed window
        # has run the pipeline DRY, which takes pp_loop_size parked
        # iterations. A window that ends sooner never reaches it. The rank
        # then advances its slot for an iteration in which it admitted
        # nothing -- an armed rank's _pull_raw_reqs returns [] without
        # touching the request chain -- so the slot moved while the chain
        # that PACES it did not. Ranks that abandoned on a different clock
        # re-enter the pipeline on different slots, and from then on stage
        # k's hidden states pair with stage k+1's batch by an index the two
        # no longer agree on (#631 defect Q).
        #
        # Restoring the arm slot is safe precisely in this case and is not
        # claimed beyond it: with passes == 0 the rank launched no work
        # (get_next_batch_to_run returns None while a flip is pending) and
        # made no drain progress, so the slot it is sent back to is the one
        # it was parked on. Longer windows keep going through the hold.
        #
        # #829: BUT ONLY IF THE RING IT NAMES STILL EXISTS. W4b's argument
        # holds for an ABANDON, which returns to the same loop with the
        # same ring -- and the falling edge does not distinguish an abandon
        # from a COMMIT, which does not. A commit raises PhaseFlipLoopExit
        # from `_phase_flip_on_round` (the round hook at the bottom of
        # `_event_loop_pp_body`) while `_pp_flip_armed_passes` is still 0;
        # the loop unwinds, the cutover swaps the topology,
        # `init_pp_loop_state` builds a NEW ring, the body restarts at
        # `mb_id = 0`, and the first tick of that new ring takes THIS
        # branch with an `arm_mb` recorded against the ring that was just
        # retired. Nothing cleared it: `_pp_flip_arm_mb_id` was written on
        # the rising edge and read here, and the rebuild does not touch it.
        #
        # This tree already states the law (see the note above this method,
        # "WHY A COMMIT IS SAFE AND AN ABANDON IS NOT"), and the #631
        # guard's own message names the hazard: "a pass from before a
        # cutover that rebuilt this rank's whole slot ring, whose slot
        # number therefore names nothing here however well it matches".
        #
        # MEASURED, boot_window2_0823_1554 @ f9d7637f04. One second after
        # the sixth cutover COMPLETED, PP0 logged this very restore --
        #   "ran 0 slot iteration(s) (armed at mb_id=2, disarmed at
        #    mb_id=0) [...] group RESUME SLOTS [2, 1, 1] -- DIVERGED"
        # -- where mb_id=0 is not one slot later but the first slot of the
        # new ring. PP1 then refused a proxy stamped mb_id=2 while sitting
        # on mb_id=0 OF THE SAME EPOCH (#631 PROXY LEFTOVER REFUSED) and
        # the group died. The epoch-4 cutover took the identical path 21 s
        # earlier and survived only because all three ranks happened to
        # carry the same retired slot ([1, 1, 1] -- AGREED). Agreement
        # there was luck; this term is the guarantee.
        #
        # Refused POSITIVELY and by name rather than by omission, because
        # the absence of a log line is what made the epoch-4 crossing look
        # healthy. The rank simply stays where the rebuilt ring started it,
        # which is the ring's own defined entry slot rather than a number
        # inherited from a ring that no longer exists.
        #
        # WHAT IS NOT CLAIMED HERE, because #737 makes the loose version
        # false: this is NOT "all ranks must sit on the same mb_id". PP
        # stages legitimately run at different microbatch offsets, and the
        # ring arithmetic says so itself -- `next_first_rank_mb_id` and
        # `next_mb_id` are deliberately different slots. The claim is
        # narrower and does not depend on any offset: a slot index recorded
        # against ring generation N must not be applied to generation N+1,
        # where it indexes a different array that a different cutover built
        # and whose `pp_loop_size` may not even contain it.
        now_epoch = pp_flip_epoch_of(self)
        arm_epoch = getattr(self, "_pp_flip_arm_epoch", None)
        ring_rebuilt = (
            arm_epoch is not None
            and now_epoch is not None
            and int(arm_epoch) != int(now_epoch)
        )
        # Consumed once, like the slot it qualifies: a stale epoch must not
        # be able to answer for a later window.
        self._pp_flip_arm_epoch = None
        would_restore = passes == 0 and arm_mb is not None and int(arm_mb) != int(mb_id)
        # #838 CLASS 1, THE W12 FORM. Keyed on the SHAPE (zero armed passes
        # across a ring rebuild) and NOT on `would_restore`, deliberately:
        # the branch below speaks only when a jump was actually averted, and
        # this file's own comment records that the epoch-4 crossing "survived
        # only because all three ranks happened to carry the same retired
        # slot -- agreement there was luck". A detector that stayed silent on
        # the lucky crossing would report the same hazard as healthy exactly
        # when it did no damage, and say nothing until the day it did.
        try:
            from sglang.srt.managers import layout_conformance

            alarm, detail = layout_conformance.stale_ring_restore_verdict(
                passes,
                arm_epoch,
                now_epoch,
                arm_mb,
                getattr(getattr(self, "phase_policy_state", None), "last_reason", None),
            )
            if alarm:
                layout_conformance.note_conformance_violation(
                    detail, time.perf_counter()
                )
        except Exception:  # noqa: BLE001 - a detector may never break the loop
            pass
        # Reported only when a jump was actually averted. A commit whose
        # retired arm slot happens to equal the new ring's first slot
        # changes nothing and should say nothing -- which is every commit
        # at pp_loop_size 1, where the TP phase and the gapped wire both
        # put the ring, and where the only slot there has ever been is 0.
        if would_restore and ring_rebuilt:
            logger.warning(
                "%s SLOT RESTORE REFUSED: the armed window ran 0 slot "
                "iterations and ended in a COMMIT, not an abandon -- it "
                "armed at mb_id=%s in flip epoch %s and this tick is the "
                "first of the ring epoch %s rebuilt. That slot number "
                "names nothing in this ring, so it is not restored; this "
                "rank stays on mb_id=%d, the slot the rebuilt ring itself "
                "started it on. Restoring it is what put PP0 "
                "on slot 2 of a fresh ring on boot_window2_0823_1554 and "
                "killed the group one second later (#829, #824 W4b, "
                "#631 defect Q).",
                "PHASE-FLIP",
                arm_mb,
                arm_epoch,
                now_epoch,
                mb_id,
            )
        elif would_restore:
            self._pp_flip_resume_slot = int(arm_mb)
            logger.warning(
                "%s SLOT RESTORE: the armed window ran 0 slot iterations and "
                "abandoned at mb_id=%d after arming at mb_id=%s, so the hold "
                "never engaged. Returning this rank to the slot it armed on; "
                "without it the slot advances while the request chain that "
                "paces it does not, and the ranks resume on different slots "
                "(#824 W4b, #631 defect Q).",
                "PHASE-FLIP",
                mb_id,
                arm_mb,
            )

        # #757: THE FALLING EDGE IS THE ONLY PLACE EVERY DISARM PASSES THROUGH.
        # Disarm has three routes and two of them are purely rank-local --
        # `_abandon_no_quorum` (phase_flip_runtime.py:3885) and
        # `_abandon_unjoined_flip` (:3949) both clear `_pending` with no
        # collective and no channel re-check, then hand control straight back
        # to the ordinary loop. `pp_flip_channels_empty` is consulted only
        # BEFORE this rank's own entry (:3682, :3748), never on the way out.
        # So an upstream can abandon on its own clock, resume launching, and
        # post a proxy into a downstream that is still armed -- and the
        # emptiness proof that was supposed to prevent it is a SAMPLE taken
        # earlier, not a barrier. Draining here catches the in-flight
        # leftover on the way back to the pass loop, on every route out.
        try:
            self.pp_flip_drain_leftover_dicts(mb_id)
        except Exception as exc:  # noqa: BLE001 - a drain may never break the loop
            logger.error("%s #757 leftover drain failed: %s", "PHASE-FLIP", exc)

        if counters is None:
            return
        try:
            per_rank = [counters.sent(CHAN_PASS, r) for r in range(counters.n_ranks)]
            slots = [counters.sent(CHAN_SLOT, r) for r in range(counters.n_ranks)]
        except Exception:  # noqa: BLE001
            return
        spread = max(per_rank) - min(per_rank) if per_rank else 0
        # #631 DEFECT Q, AS A VERDICT. The spread is expected and harmless:
        # ranks spin at their own rate and abandon on their own clock. The
        # SLOT is what may not diverge, because it is what the proxy stamp,
        # the mbs occupancy and the output pairing are all indexed by. A
        # rank reads its peers' last published armed slot here; with the
        # hold in place (_pp_flip_hold_slot) every rank is parked on the
        # slot it armed on, so these agree.
        agreed = len(set(slots)) <= 1
        logger.log(
            logging.WARNING if agreed else logging.ERROR,
            "%s PASS-CLOCK across the armed window: rank %d ran %d slot "
            "iteration(s) (armed at mb_id=%s, disarmed at mb_id=%d); group "
            "passes %s, SPREAD %d; group RESUME SLOTS %s -- %s. The spread "
            "is not the defect: ranks spin at their own rate and abandon on "
            "their own clock. The RESUME SLOT is, because stage k's hidden "
            "states pair with stage k+1's batch by that index and by "
            "nothing else (#631 defect Q).",
            "PHASE-FLIP",
            counters.rank,
            passes,
            arm_mb,
            mb_id,
            per_rank,
            spread,
            slots,
            (
                "AGREED"
                if agreed
                else "DIVERGED, so every later proxy on this instance is "
                "mispaired and the slot hold did not do its job"
            ),
        )

    def _pp_flip_hold_slot(self: Scheduler) -> bool:
        """Must this rank stay on the SAME microbatch slot for another turn?

        #631 DEFECT Q, and this is its fix rather than another instrument
        for it.

        THE MEASUREMENT THAT NAMES THE DEFECT (2026-08-09 07:19:23Z, the
        boot that produced corpse R):

            rank 0 ran 44477 slot iteration(s) (armed at mb_id=2,
                   disarmed at mb_id=2)
            rank 1 ran 33690 slot iteration(s) (armed at mb_id=2,
                   disarmed at mb_id=0)
            rank 2 ran 38069 slot iteration(s) (armed at mb_id=2,
                   disarmed at mb_id=2)   SPREAD 10787

        ALL THREE RANKS ARM ON THE SAME SLOT AND LEAVE ON DIFFERENT ONES.
        That is the whole defect, and it is not a message defect.

        WHY THE ARMED WINDOW DRIFTS AT ALL. In steady state the pass loop
        is PACED BY THE REQUEST CHAIN: every slot iteration makes exactly
        one blocking chain receive, so rank k's i-th iteration is rank
        k-1's i-th iteration and the slot indices cannot diverge. An armed
        rank admits nothing (``_pull_raw_reqs`` returns [] before touching
        the chain) and launches nothing (``get_next_batch_to_run`` returns
        ``batch_to_run=None`` while a flip is pending), so its iterations
        are pure spin -- roughly 8 kHz here -- and the pacing is gone. Each
        rank then abandons on its OWN park deadline, having spun a
        different number of times, and re-enters the pipeline on a
        different slot.

        WHAT THAT COSTS, and why it looked like a stranded message. Nothing
        is left on the wire: a parked rank neither sends nor receives a
        proxy, so the one-message-per-pass contract is never broken and the
        counts stay balanced. What breaks is the LABEL. Stage k computes
        the hidden states of its slot-s batch while stage k+1 applies them
        to its slot-s' batch, for ever after, because both indices simply
        advance from wherever their rank happened to stop. The proxy stamp
        detects the first such message ("stamp mb_id=2 ... while this rank
        is on mb_id=1") and both disposals died trying to treat a standing
        phase offset as one stale message: corpse R took a second message
        against a debt of one and wedged; corpse S drained a wire that had
        nothing surplus on it and ate an output.

        THE FIX IS TO STOP THE INDEX, NOT TO REPAIR ITS CONSEQUENCES. Hold
        the slot once the armed window has run the pipeline dry; the rank
        keeps spinning, keeps servicing its channels, keeps polling the
        gate -- it simply does so on ONE slot. Every rank then resumes
        where it armed, whatever its spin count was, and the spread becomes
        irrelevant instead of fatal.

        WHY THE HOLD IS REACHED ON THE SAME SLOT ON EVERY RANK, which is
        the only property that makes this correct. A parked iteration sets
        ``mbs[mb_id] = None``. The arm itself is slot-uniform because it
        rides the request chain, which is 1:1 and ordered, so it lands on
        the same ordinal iteration everywhere (measured above: "armed at
        mb_id=2" on all three ranks). From that shared slot every rank
        needs exactly ``pp_loop_size`` parked iterations to null every
        slot, so ``all(mb is None)`` first holds at the same slot index on
        every rank. ``is None`` and not ``is_empty()`` deliberately: the
        stricter test is the one that is reached after a FIXED number of
        iterations rather than whenever a slot happens to be empty.

        A HALF-WRITTEN CHUNK IS NOT A HOLD. ``chunked_req`` is exempt from
        the park (its continuation must complete or quiescence is
        unreachable), so those iterations launch real work, are chain-paced
        like any other, and are lockstep across ranks. Holding there would
        stop a rank the pipeline is still driving.

        ONE AUTHORITY FOR "ARMED", checked rather than assumed. The park in
        ``get_next_batch_to_run`` keys on ``phase_flip_runtime.pending is
        not None``; ``pp_phase_flip_armed`` -> ``is_armed()`` is a read of
        that same ``_pending`` and nothing else. So "armed" here and
        "parked" there cannot disagree, and this predicate can never hold a
        rank that is still being handed batches.

        NO LAUNCH TIMING MOVES -- the refined design law. Every iteration
        this suppresses launches nothing, sends nothing and receives
        nothing; it is a spin the loop was already doing, on a different
        index. No rank waits on any peer to decide whether to hold, so no
        synchronisation point is added at arm time either.
        """
        if not getattr(self.server_args, "enable_phase_flip", False):
            return False
        try:
            if not self.pp_phase_flip_armed():
                return False
        except Exception:  # noqa: BLE001 - never let a probe break the loop
            return False
        if getattr(self, "chunked_req", None) is not None:
            return False
        mbs = getattr(self, "mbs", None)
        if not mbs:
            return False
        return all(mb is None for mb in mbs)

    def _pp_flip_bump_sent(self: Scheduler, chan: str) -> None:
        counters = getattr(self, "pp_flip_counters", None)
        if counters is not None:
            counters.bump_sent(chan)

    def _pp_flip_bump_attempted(self: Scheduler, chan: str) -> None:
        counters = getattr(self, "pp_flip_counters", None)
        if counters is not None:
            counters.bump_attempted(chan)

    def _pp_flip_bump_consumed(self: Scheduler, chan: str) -> None:
        counters = getattr(self, "pp_flip_counters", None)
        if counters is not None:
            counters.bump_consumed(chan)

    def _pp_flip_ring(self: Scheduler) -> Tuple[int, int]:
        """(this rank, ring size) of the PP CHAIN -- not of the live ps.

        #631 DEFECT M. These two helpers used to read ``self.ps``
        directly, and the cutover REWRITES ps per phase: the TP phase gets
        pp_rank=0, pp_size=1. The ring then degenerated to
        ``(0 - 1) % 1 == 0`` on every rank, i.e. UPSTREAM == SELF, and the
        flip-commit hygiene check compared a rank's own dict SEND counter
        against its own dict CONSUME counter -- two different wires. Rank
        0 is the first PP stage: it sends proxy dicts downstream and
        consumes none, so its imbalance was permanent and grew with the PP
        phase's traffic.

        Measured 2026-08-09 03:21:08-03:22:08Z: rank 0 WITHHELD presence
        for 8889 rounds with "tensor-dict wire has 24 unconsumed
        message(s) from rank 0" -- itself -- and tp_to_pp abandoned for
        want of a quorum it could never form. No message was ever
        unconsumed; the ring was.

        The counters are built ONCE from the PP topology at boot, so they
        are the ring's one authority; everything that needs it reads it
        from there rather than from a ps the flip rewrites.
        """
        counters = getattr(self, "pp_flip_counters", None)
        if counters is not None:
            return counters.rank, counters.n_ranks
        return self.ps.pp_rank, self.ps.pp_size

    def _pp_flip_upstream(self: Scheduler) -> int:
        rank, n = self._pp_flip_ring()
        return (rank - 1) % n

    def _pp_flip_downstream(self: Scheduler) -> int:
        rank, n = self._pp_flip_ring()
        return (rank + 1) % n

    def pp_flip_consume_inbound(self: Scheduler) -> int:
        """Greedily take every inbound message the upstream says it posted.

        The request chain only. The tensor-dict wire is deliberately NOT
        consumed here, and that is a positive design decision rather than
        an omission: a dict message is a microbatch's hidden states or its
        outputs, and a rank that has one inbound is BY DEFINITION not
        quiescent (its own ``mbs`` slot is live, which is what
        ``_pp_microbatches_drained`` reads). Such a rank is still in the
        ordinary pass loop, where that recv happens normally. Consuming a
        dict here would mean buffering a microbatch across a layout change
        that discards it -- so instead the entry check below PROVES the
        wire is empty, and a violation abandons the flip rather than
        crossing the re-formation with a message in flight.
        """
        receiver = getattr(self, "pp_chain_receiver", None)
        counters = getattr(self, "pp_flip_counters", None)
        if receiver is None or counters is None:
            return 0
        posted = counters.sent(CHAN_REQ, self._pp_flip_upstream())
        return receiver.consume_up_to(posted)

    def pp_flip_flush_drained_sends(self: Scheduler) -> None:
        """Reap this rank's outstanding sends -- but ONLY once consumed.

        ``_pp_commit_comm_work`` is a blocking ``wait()``, and blocking is
        exactly what an armed rank may not do speculatively: the measured
        wire fact is that a sender's ``wait()`` blocks when the receiver
        has posted no matching irecv. Gated on the downstream's published
        CONSUMED count, the same call is bounded -- the message is already
        off the wire, so the wait returns immediately.

        This is the working replacement for ``pp_pump_send_req_work``,
        which reaped nothing because ``is_completed()`` never fires for an
        isend here (corpse F). Reaping matters beyond tidiness: while
        ``send_req_work`` is non-empty this rank "owes a send", presence is
        withheld, and the flip abandons at the presence deadline.
        """
        counters = getattr(self, "pp_flip_counters", None)
        if counters is None:
            return
        downstream = self._pp_flip_downstream()
        for chan, attr in (
            (CHAN_REQ, "send_req_work"),
            (CHAN_DICT, "send_output_work"),
            (CHAN_DICT, "send_proxy_work"),
        ):
            work = getattr(self, attr, None)
            if not work:
                continue
            if counters.consumed(chan, downstream) < counters.local_sent(chan):
                continue
            self._pp_commit_comm_work(work)

    def pp_flip_flush_pending_dict_sends(self: Scheduler) -> None:
        """#787 SENDER-SIDE HALF: last-chance reap/count before an abandon.

        Wired as ``flush_pending_sends_fn`` into ``PhaseFlipRuntime`` and
        called synchronously by ``_abandon_no_quorum`` and
        ``_abandon_unjoined_flip``, strictly BEFORE either clears this
        rank's local flip state (``self._pending = None``). It is the
        complementary half of the receiver-side settle window added to
        ``pp_flip_drain_leftover_dicts`` above: that window can only find
        a message that was already sent and merely not yet counted; it
        is powerless against a send this rank had not yet issued when a
        downstream peer's settle window expired. This function closes
        that gap from the sending side -- it reaps and counts anything
        THIS rank has already posted, so the downstream's own
        ``counters.sent()`` reading reflects reality before this rank is
        free to resume and race ahead.

        Deliberately just ``pp_flip_flush_drained_sends()``: that call is
        already bounded (gated on the downstream's published CONSUMED
        count, so it never blocks on an unfinished peer) and already the
        correct reap primitive for CHAN_DICT and CHAN_REQ alike. This
        wrapper exists so the call site in ``phase_flip_runtime.py`` names
        the #787 contract explicitly, independent of whatever
        ``pp_flip_flush_drained_sends`` is used for elsewhere.

        HONEST SCOPE: this cannot and does not force an in-flight forward
        computation to complete early -- a send that has not been issued
        yet still lands whenever it actually completes, unaffected by
        this call. What it guarantees is that any send ALREADY POSTED by
        this rank is reaped and counted before disarm, closing exactly
        the ordering gap #787 exploits (a downstream's one-shot drain
        snapshot running before an upstream's already-completed send was
        counted).
        """
        try:
            self.pp_flip_flush_drained_sends()
        except Exception as exc:  # noqa: BLE001 - best effort, abandon must not raise
            logger.warning(
                "%s #787 pre-abandon send flush failed: %s", "PHASE-FLIP", exc
            )

    def pp_flip_drain_tensor_dicts(self: Scheduler) -> int:
        """Consume the tensor-dict wire while ARMED, and discard what comes.

        THIS IS THE PREVENTION HALF OF THE MISPAIRING FIX, and it is where
        the strand is actually killed.

        THE STRAND. A flip abandon is rank-local: each rank times out on
        its own clock. The first rank to disarm resumes launching and sends
        its proxy hidden states. Its downstream is still armed, so it has
        no ``cur_batch`` -- and the proxy recv was guarded by THIS rank's
        batch, never by whether the upstream sent. The message was left on
        the wire, and every later receive on that rank was off by one,
        silently, for the rest of the loop's life.

        WHY DISCARDING IS RIGHT HERE, AND ONLY HERE. The open question in
        the design note was "discarding loses a microbatch". Metal answered
        it: at THIS point there is no microbatch to lose. An armed rank is
        withholding admissions and launching nothing, so the message names
        a pass this rank never ran and there is no batch it could ever pair
        with -- now or later. That is precisely NOT true at the receive
        site, where discarding a message you do have a batch for wedged the
        instance (corpse R).

        WHY THE BLOCKING RECV IS SAFE -- the whole safety argument. It is
        made ONLY when the upstream's published counter exceeds this rank's
        consumed count, so the message provably exists and the call is
        bounded by transfer time rather than by peer scheduling. The
        publish-after-post ordering makes the only possible skew
        counter-lags-send, which under-reports and can never invent a
        message that was not sent. Corpse F rules out polling here; this
        counter is what replaces it.

        WHY IT MOVES NO LAUNCH TIMING -- the refined design law. It adds
        consumption during the armed window only, when this rank launches
        nothing anyway. No rank's decision about WHEN to proceed changes,
        which is what the resume gate got wrong by holding ranks out of
        launching (HANDOFF §7).

        DISABLED -- CORPSE S. The paragraph that stood here said the
        demultiplexer was bypassed deliberately because "both kinds are
        equally void while armed". METAL FALSIFIED THAT SENTENCE within one
        arm: the upstream wire MULTIPLEXES the proxy forward and the output
        return, and an output belongs to work launched BEFORE the arm. This
        function ate one (kind=output, PP1, 07:33:30Z) and PP1 then blocked
        for ever waiting for it, with PP2 behind it. See the call site in
        ``pp_flip_service`` for the full specimen and for what a correct
        version would have to do instead. It is left here, uncalled, so the
        next reader inherits the measurement rather than the idea.
        """
        counters = getattr(self, "pp_flip_counters", None)
        if counters is None:
            return 0
        drained = 0
        # Bounded so a counter that runs away cannot pin the rank here.
        for _ in range(64):
            posted = counters.sent(CHAN_DICT, self._pp_flip_upstream())
            if counters.local_consumed(CHAN_DICT) >= posted:
                break
            raw = self.pp_group.recv_tensor_dict(
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                )
            )
            self._pp_flip_bump_consumed(CHAN_DICT)
            drained += 1
            stamp = raw.get("__stamp__") if isinstance(raw, dict) else None
            # #757: DEMULTIPLEX, then decide. Kind-blind discarding is corpse
            # S -- it ate an `output` owed to a real consumer and blocked PP1
            # for ever. The message stays off the wire either way (the
            # upstream's blocking commit waits on exactly that), but only a
            # provably void proxy is dropped; everything else is handed to the
            # inbox its consumer already reads.
            # getattr: this method is bound UNBOUND against stubs by
            # test_pp_proxy_stamp_631, which carries only what the old drain
            # touched. A missing accessor means "no slots known", which is the
            # conservative reading anyway.
            ran_fn = getattr(self, "_pp_ran_mb_ids", None)
            # #795: same getattr convention for the epoch -- a stand-in with
            # no accessor yields None, which turns the epoch test off and
            # leaves the slot-only classification that shipped before.
            action, kind, why = classify_armed_drain_message(
                raw,
                ran_fn() if ran_fn is not None else set(),
                pp_flip_epoch_of(self),
            )
            if action == DRAIN_STASH:
                # Never discard for want of somewhere to put it -- that is
                # corpse S. The inbox is created on demand by stash_typed.
                stash_typed(self.pp_group, None, kind, raw)
                logger.info(
                    "%s armed drain took a tensor dict off the wire and "
                    "STASHED it: kind=%s stamp=%s -- %s. (%d this window)",
                    "#757",
                    kind,
                    stamp,
                    why,
                    drained,
                )
            else:
                logger.info(
                    "%s armed drain took a tensor dict off the wire and "
                    "discarded it: kind=%s stamp=%s -- %s. Leaving it on the "
                    "wire is what used to strand it and put every later "
                    "receive off by one. (%d this window)",
                    "#757",
                    kind,
                    stamp,
                    why,
                    drained,
                )
        if drained:
            self._pp_flip_drained_total = (
                getattr(self, "_pp_flip_drained_total", 0) + drained
            )
        return drained

    def pp_flip_drain_leftover_dicts(self: Scheduler, live_mb_id: int) -> int:
        """#757: clear pre-arm leftovers at DISARM, without eating an output.

        THIS IS CORPSE S DONE CORRECTLY, and the difference is not courage --
        it is that two things exist now which did not on 2026-08-09.

        WHAT WENT WRONG BEFORE. ``pp_flip_drain_tensor_dicts`` above is
        kind-blind: it logs ``kind=`` and ``stamp=`` and then discards
        whatever it took. The upstream wire MULTIPLEXES the proxy forward
        and the output return, so it ate an OUTPUT (PP1, 07:33:30Z) that
        belonged to work launched BEFORE the arm and was still owed; PP1
        then blocked for ever waiting for what it had itself destroyed.
        Its own comment names the repair: a drain that discards "must
        demultiplex first (stash 'output' in the inbox, where its consumer
        already looks) and then decide about 'proxy' alone".

        WHAT MAKES THAT REPAIR EXPRESSIBLE NOW.
          1. ``_pp_tensor_dict_inbox`` (a defaultdict(deque)) and the
             demultiplexing ``_pp_recv_typed_dict``, which POPS from that
             inbox before touching the wire. So a message stashed here is
             not discarded -- it is delivered to its real consumer later.
          2. ``__stamp__`` on every proxy. Corpse S had no way to tell a
             void proxy from an owed one; the stamp names the pass, so
             "belongs to a pass this rank is not on" is now a FACT rather
             than the assumption that sank the first attempt.

        WHY THIS IS THE #757 FIX. Measured on 2026-08-18 (comp4, crash at
        06:36:29, specimen SPECIMEN-2026-08-18T0636Z-comp4-proxy-leftover-
        underload.log): all six PASS-CLOCK verdicts read AGREED with group
        RESUME SLOTS [1,1,1], so the ranks did NOT diverge and #631 defect Q
        did not happen. The proxy that killed it was stamped mb_id=2 and
        arrived one second after a CLEAN tp_to_pp commit -- an in-flight
        message from before the arm, exactly the case the corpse-S comment
        flagged as unresolved ("in-flight microbatches launched before the
        arm raise the same question a second time"). Nothing drained it,
        because the only drain was switched off.

        NOTHING IS DISCARDED EXCEPT A PROVABLY VOID PROXY: wrong-kind
        messages are stashed for their own consumer, and a proxy whose
        stamp matches the pass this rank is about to run is stashed too --
        it is owed, not leftover. The #631 receive guard is untouched and
        stays the backstop; if this drain is ever wrong, the guard still
        refuses rather than mispairs.

        #787 -- THE RECEIVER-SIDE HALF, ADDED HERE. The paragraphs above
        describe a ONE-SHOT snapshot of ``counters.sent()``: it fires
        exactly once at the falling edge of the pass tick, on this rank's
        own clock, with no collective re-check against the upstream. #787
        is that snapshot missing a message the upstream posts strictly
        AFTER this rank's single sweep already ran and returned -- the
        rank disarms, resumes launching, and later receives that stale
        proxy at the ordinary #631-guarded receive site instead of here,
        where it would have been recognised as leftover and dropped.
        The fix is a BOUNDED SETTLE WINDOW: when the snapshot currently
        reads "nothing new", re-poll the SAME local SHM counter a few
        more times (``DRAIN_SETTLE_STEP_S`` apart, up to
        ``DRAIN_SETTLE_BUDGET_S`` total) before actually breaking out of
        the loop, so a send that is landing right now has a chance to be
        seen and drained here instead of downstream. This is deliberately
        a poll of the counter this loop already reads -- NEVER a
        timing-out ``Work.wait()`` on the gloo handle, which would tear
        apart the sender/receiver pairing on the wire (the corpse-F
        lesson threaded through this module: ``is_completed()``/timed
        waits do not behave sanely on this transport). The settle window
        alone is NOT a fix -- it only closes a receiver-side race BY
        finding a message that was already sent; it cannot see one that
        has not been sent yet. That is why #787 also adds a sender-side
        ordering guarantee (``_abandon_no_quorum`` /
        ``_abandon_unjoined_flip`` in ``phase_flip_runtime.py``, via
        ``pp_flip_flush_pending_dict_sends`` below): flush and count
        every pending old-pass send BEFORE this rank's own peer flips to
        abandoned/resumed. Neither half ships alone.
        """
        counters = getattr(self, "pp_flip_counters", None)
        if counters is None:
            return 0
        discarded = 0
        drained_messages = 0
        settle_deadline = None
        # Bounded for the same reason the corpse is: a counter that runs
        # away must not pin the rank here. The cap counts actual messages
        # taken off the wire, not settle-window polls -- a long settle
        # wait must not eat into the 64-message drain budget.
        while drained_messages < 64:
            posted = counters.sent(CHAN_DICT, self._pp_flip_upstream())
            if counters.local_consumed(CHAN_DICT) >= posted:
                # #787: nothing new by this reading -- but that is exactly
                # the one-shot snapshot #787 exploits. Give a bounded
                # settle window for an in-flight send to land and be
                # counted before actually giving up.
                now = time.monotonic()
                if settle_deadline is None:
                    settle_deadline = now + DRAIN_SETTLE_BUDGET_S
                if now >= settle_deadline:
                    break
                time.sleep(DRAIN_SETTLE_STEP_S)
                continue
            # A message is provably available now; the settle window only
            # ever applies to the gap AFTER the last provably real one.
            settle_deadline = None
            raw = self.pp_group.recv_tensor_dict(
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                )
            )
            self._pp_flip_bump_consumed(CHAN_DICT)
            drained_messages += 1
            kind = raw.get("__msg_type__", "default") if isinstance(raw, dict) else None
            if kind != "proxy":
                # THE CORPSE-S LESSON, ENCODED. An output belongs to work
                # launched before the arm and is still owed to a real
                # consumer; stash it where `_pp_recv_typed_dict` already
                # looks instead of destroying it.
                stash_typed(
                    self.pp_group, None, kind if kind is not None else "default", raw
                )
                continue
            stamp = raw.get("__stamp__") if isinstance(raw, dict) else None
            # #795: (epoch, slot), not slot alone. THIS LINE IS THE 2026-08-21
            # MISPAIR. When the disarm follows a COMMITTED cutover, the ring
            # this rank resumes on was rebuilt from zero by
            # `init_pp_loop_state` (phase_flip_runtime.py:1580), so a
            # pre-cutover proxy whose slot number coincides with `live_mb_id`
            # was stashed here as "owed" and delivered straight into model
            # compute -- with the #631 receive guard, the intended backstop,
            # agreeing for the same reason. An ABANDONED flip does not advance
            # the epoch and does not rebuild the ring, so the pre-arm proxies
            # this drain was written for still read as owed. See
            # `pp_proxy_stamp_names_pass`.
            epoch = pp_flip_epoch_of(self)
            if (
                stamp is None
                or live_mb_id < 0
                or pp_proxy_stamp_names_pass(stamp, live_mb_id, epoch)
            ):
                # Owed, not leftover: this is the pass the rank is about to
                # run. Stash so the ordinary receive returns it.
                stash_typed(self.pp_group, None, "proxy", raw)
                continue
            discarded += 1
            logger.info(
                "%s #757/#795 drained a LEFTOVER proxy at disarm: stamp=%s while "
                "this rank resumes on mb_id=%s in flip epoch %s. It names a pass "
                "from another slot or another flip epoch, so no batch of this "
                "rank's can ever pair with it. (%d this window)",
                "PHASE-FLIP",
                stamp,
                live_mb_id,
                epoch,
                discarded,
            )
        if discarded:
            self._pp_flip_leftovers_dropped = (
                getattr(self, "_pp_flip_leftovers_dropped", 0) + discarded
            )
        return discarded

    # #757 FOLD (review verdict b, measured): the disarm-time drain above and
    # the armed-time classifier below are COMPLEMENTARY halves. Gloo isend
    # posts instantly but wait() blocks until the peer recv (measured 3000.3
    # ms stall for a 3 s armed window, size-independent, 4 KiB and 8 MiB) --
    # so the armed drain below keeps the upstream's blocking commit live
    # DURING the window, and the disarm sweep above catches anything that
    # arrived between its last poll and the falling edge.
    def _pp_ran_mb_ids(self: Scheduler) -> set:
        """Microbatch ids this rank has a slot for -- i.e. passes it can own.

        A proxy stamped with one of these was launched before the arm and is
        still owed; one stamped with anything else names a pass this rank never
        ran. ``mbs`` is the per-slot resident set, so its indices are exactly
        the ids this rank can legitimately pair with.
        """
        try:
            mbs = getattr(self, "mbs", None) or ()
            return {i for i, b in enumerate(mbs) if b is not None}
        except Exception:  # noqa: BLE001 - an unreadable slot set proves nothing
            return set()

    def pp_flip_service(self: Scheduler) -> None:
        """One turn of the armed service loop: consume, then flush.

        CONSUME FIRST, and the order is load-bearing. Taking the upstream's
        message off the wire is what lets the UPSTREAM flush its own send
        on its next turn; flushing first would leave every rank waiting on
        a peer that is waiting on it. Consuming is also the half that can
        never block for peer reasons -- the counter proved the message
        exists -- so doing it first means a rank always makes the group's
        progress possible before asking anything of it.
        """
        try:
            self.pp_flip_consume_inbound()
        except Exception as exc:  # noqa: BLE001
            logger.error("%s armed consume failed: %s", "#631", exc)
        # CORPSE S -- THE ARMED DRAIN IS DISABLED, and must not be re-enabled
        # in this shape. Metal, 2026-08-09 07:33:30Z, specimen
        # /spinning/evidence-631/wedge_20260809T073423Z_armed_drain_ate_output_20260809T0733Z:
        #
        #     PP1] #631 armed drain took a tensor dict off the wire and
        #          discarded it: kind=OUTPUT stamp=None
        #
        # and 20 s later PP1 AND PP2 were both blocked in
        # _pp_recv_dict_from_prev_stage while PP0 spun at the flip gate.
        # PP1 was waiting for the output its own drain had eaten.
        #
        # THE FALSE SENTENCE, verbatim from the docstring below: "both kinds
        # are equally void while armed". They are NOT. The upstream wire
        # MULTIPLEXES two streams -- the proxy forward and the output return,
        # relayed stage by stage and demultiplexed by __msg_type__ AFTER
        # coming off the wire. A proxy for a pass an armed rank never ran is
        # void. An OUTPUT belongs to work that was launched BEFORE the arm
        # and is still owed to a real consumer. Discarding it destroys a
        # microbatch's results and strands every rank waiting behind it.
        #
        # The repair is not a bigger hammer: a drain that wants to be
        # kind-blind cannot be, and one that discards must demultiplex first
        # (stash 'output' in the inbox, where its consumer already looks) and
        # then decide about 'proxy' alone -- for which in-flight microbatches
        # launched before the arm raise the same question a second time.
        # #757: RE-ENABLED. Corpse S killed the KIND-BLIND drain, and the note
        # above states the repair exactly -- "one that discards must
        # demultiplex first (stash 'output' in the inbox, where its consumer
        # already looks) and then decide about 'proxy' alone". That is now
        # `classify_armed_drain_message`, and the in-flight-proxy question the
        # note raised a second time is answered by the stamp: a proxy for a
        # microbatch this rank DID run is stashed, not dropped.
        #
        # The guard at `_pp_recv_proxy_tensors` STAYS. It refused correctly on
        # comp4 and it is the only thing standing between this race and silent
        # cross-microbatch corruption; this drain is the prevention half that
        # should keep it from ever firing.
        try:
            self.pp_flip_drain_tensor_dicts()
        except Exception as exc:  # noqa: BLE001 - a drain may never kill a flip
            logger.error("%s armed drain failed: %s", "#757", exc)
        try:
            self.pp_flip_flush_drained_sends()
        except Exception as exc:  # noqa: BLE001
            logger.error("%s armed flush failed: %s", "#631", exc)
        # #800: THE ESCAPE HATCH, and it belongs HERE rather than in the
        # presence probe. `pp_flip_channels_empty` is consulted twice per round
        # and is documented as reporting rather than acting; retiring from
        # inside it would make a probe mutate the thing it measures. This is
        # the armed loop's one service turn, it already acts, and it runs
        # immediately BEFORE the probe in the same round.
        try:
            self.pp_flip_retire_undeclared_stash()
        except Exception as exc:  # noqa: BLE001 - an escape may never kill a flip
            logger.error("%s undeclared-stash escape failed: %s", "#800", exc)

    def pp_flip_retire_undeclared_stash(self: Scheduler) -> int:
        """#800: bound how long an UNDECLARED kind may hold the flip gate.

        THE STATE THIS CREATES, and it is the one the specimen did not have.
        A stashed message with no declared disposition used to have no timeout,
        no discard rule and no way to say "I cannot place this". It simply sat
        there and the rank went quiet -- 57922 withheld rounds over five
        minutes, six abandon/re-arm cycles, an instance that answered nothing
        while its port stayed open. A message this rank cannot place is now a
        NAMED, BOUNDED state: held conservatively at first (it might be an owed
        payload), reported by kind from the first round by
        ``pp_flip_channels_empty``, and retired loudly here once the escape
        deadline expires.

        DECLARED KINDS ARE NEVER TOUCHED. ``output``, ``proxy`` and
        ``crossing`` are owed to a consumer that looks for them after the
        cutover and must keep blocking; ``admission_decision`` does not block
        at all and must survive an abandon intact, because the resumed PP body
        pops exactly one per pass and dropping one puts every later receive off
        by one. Retiring either group here would be corpse S with a clock on
        it.

        The clock is per ``(src, kind)`` and is reset when the key empties, so
        a kind that is consumed and stashed again starts fresh rather than
        inheriting a stranger's age.

        ``SGLANG_PP_STASH_ESCAPE_S <= 0`` disables the retirement while leaving
        the census and its log line intact -- the off-switch a guard needs in
        order to be provable in both directions.
        """
        inbox = getattr(self, "_pp_tensor_dict_inbox", None)
        seen: Dict[tuple, float] = getattr(self, "_pp_stash_first_seen", None)
        if seen is None:
            seen = {}
            self._pp_stash_first_seen = seen
        keys = stash_keys_with_disposition(inbox, (UNDECLARED,))
        # Forget keys that have emptied, so the next occupant is timed from its
        # own arrival rather than from a predecessor's.
        for stale in [k for k in seen if k not in keys]:
            del seen[stale]
        if not keys:
            return 0
        # One named seam for the clock, so the escape's DEADLINE can be tested
        # without a five-minute test. Absent on the shipped scheduler, which is
        # the point: production reads the real monotonic clock and nothing has
        # to be injected for it to do so.
        now = getattr(self, "_pp_stash_clock", time.monotonic)()
        deadline = envs.SGLANG_PP_STASH_ESCAPE_S.get()
        retired = 0
        for key in keys:
            first = seen.setdefault(key, now)
            age = now - first
            if deadline <= 0 or age < deadline:
                continue
            queue = inbox.get(key)
            depth = len(queue or ())
            if not depth:
                continue
            queue.clear()
            del seen[key]
            retired += depth
            logger.error(
                "%s RETIRED %d stashed message(s) of UNDECLARED kind=%s from "
                "rank %s after %.1fs (escape deadline %.1fs). No consumer is "
                "registered for this kind at the flip gate, so holding it was "
                "blocking this rank's presence and no amount of waiting could "
                "clear it. Declare its disposition in pp_stash_disposition "
                "instead of relying on this escape: a retired message is a "
                "LOST payload if any consumer did in fact owe it.",
                "#800",
                depth,
                key[1] if isinstance(key, tuple) and len(key) >= 2 else key,
                key[0] if isinstance(key, tuple) and len(key) >= 2 else "?",
                age,
                deadline,
            )
        if retired:
            self._pp_stash_retired_total = (
                getattr(self, "_pp_stash_retired_total", 0) + retired
            )
        return retired

    def pp_flip_retire_pp_loop_stash(self: Scheduler) -> int:
        """#800: retire the PP-loop-only stash at a CUTOVER, by name.

        Called from the cutover, which is the one moment these messages stop
        being owed: they name a pass in a slot ring that the cutover destroys
        (``init_pp_loop_state`` rebuilds it), and the phase they are consumed
        in no longer exists after this point.

        WHY THIS IS NOT ALREADY IMPLICIT. The cutover's own instrument states
        that ``init_pp_loop_state`` "clears ... the tensor-dict inbox with no
        drain and no carry" and reports ``inbox=%d`` under the heading CUTOVER
        DISCARDS IN-FLIGHT OUTPUT. That stopped being true at #753, which moved
        the inbox off the scheduler and onto the ``pp_group`` so the crossing
        wire could share it; ``init_pp_loop_state`` explicitly does not touch it
        any more. Nothing has cleared it since, so a message parked here
        outlives the cutover, outlives the whole TP phase, and is handed to the
        NEXT PP epoch's receive -- a mispair of exactly the class #795's epoch
        stamp exists to catch. This makes the discard real and names what it
        discarded.

        THE THREE DISPOSITIONS ARE TREATED DIFFERENTLY HERE, and the split is
        the point:

        ``PP_LOOP_ONLY``  retired, at INFO. Expected, routine, and the reason
                          this method exists.
        ``UNDECLARED``    retired, at ERROR. An undeclared entry BLOCKS
                          presence, so a rank cannot announce while holding
                          one and finding it here means the gate was bypassed
                          -- but leaving it is the very outliving this method
                          prevents, and nothing is known to be owed it. So it
                          goes, loudly. (This branch was missing from the
                          first cut of #800: the sweep took only PP_LOOP_ONLY
                          and an undeclared entry would have survived into the
                          next phase, which is the defect being fixed, one
                          disposition over.)
        ``BLOCKS_FLIP``   NOT retired. A real consumer may still be owed the
                          payload, so sweeping it would destroy a token to
                          tidy up a predicate bug. Reported and left, so the
                          defect stays visible.
        """
        inbox = getattr(self, "_pp_tensor_dict_inbox", None)
        census = census_stash(inbox)
        if census.blocking:
            logger.error(
                "%s CUTOVER FOUND A BLOCKING STASH: %s. The presence gate is "
                "supposed to make this impossible -- a rank does not announce "
                "while it holds one -- so this is a quiescence-predicate bug. "
                "Left in place deliberately; sweeping it here would hide it, "
                "and its payload may still be owed to a real consumer.",
                "#800",
                ", ".join(f"{n} x {kind}" for kind, n in census.blocking),
            )
        if census.undeclared:
            logger.error(
                "%s CUTOVER FOUND AN UNDECLARED STASH: %s. This also blocks "
                "presence, so reaching a cutover with one held means the gate "
                "was bypassed. Retired anyway -- nothing is registered to "
                "consume it and leaving it would hand it to the next epoch's "
                "receive -- but declare its disposition in "
                "pp_stash_disposition rather than relying on this sweep.",
                "#800",
                ", ".join(f"{n} x {kind}" for kind, n in census.undeclared),
            )
        retired = 0
        for key in stash_keys_with_disposition(inbox, (PP_LOOP_ONLY, UNDECLARED)):
            queue = inbox.get(key)
            depth = len(queue or ())
            if not depth:
                continue
            queue.clear()
            retired += depth
            logger.info(
                "%s cutover retired %d stashed message(s) of kind=%s from rank "
                "%s: they name a pass in the slot ring this cutover rebuilds, "
                "so no consumer can ever take them again",
                "#800",
                depth,
                key[1] if isinstance(key, tuple) and len(key) >= 2 else key,
                key[0] if isinstance(key, tuple) and len(key) >= 2 else "?",
            )
        seen = getattr(self, "_pp_stash_first_seen", None)
        if seen:
            seen.clear()
        return retired

    def pp_flip_channels_empty(self: Scheduler) -> Optional[str]:
        """Are ALL of this rank's channels empty? None if yes, else why not.

        THE FLIP-COMMIT HYGIENE CHECK. Quiescent plus fully serviced
        implies every channel is empty; anything else is a framing or
        quiescence bug, and it is the nastiest one this change can
        introduce. A half-consumed ``point_to_point_pyobj`` message, or an
        unreaped isend, crossing the re-formation would misframe the
        post-flip stream -- silently, and long after the flip.

        Cheap, and it also catches a sender that died between posting its
        message and publishing its counter: the wire then holds a message
        nobody will ever account for, and this is where that shows up.

        Reported rather than raised: the caller runs this BEFORE entering
        the reduction, where abandoning is free and safe.
        """
        counters = getattr(self, "pp_flip_counters", None)
        if counters is None:
            return None
        reasons: List[str] = []

        receiver = getattr(self, "pp_chain_receiver", None)
        if receiver is not None:
            posted = counters.sent(CHAN_REQ, self._pp_flip_upstream())
            if receiver.consumed < posted:
                reasons.append(
                    f"request chain has {posted - receiver.consumed} "
                    f"unconsumed message(s) from rank "
                    f"{self._pp_flip_upstream()}"
                )
            if receiver.mid_message:
                reasons.append("request chain is HALF-RECEIVED (mid-message)")
            if receiver.pending():
                reasons.append(
                    f"request-chain inbox holds {receiver.pending()} "
                    f"unhandled message(s)"
                )

        dict_posted = counters.sent(CHAN_DICT, self._pp_flip_upstream())
        dict_taken = counters.local_consumed(CHAN_DICT)
        if dict_taken < dict_posted:
            reasons.append(
                f"tensor-dict wire has {dict_posted - dict_taken} "
                f"unconsumed message(s) from rank {self._pp_flip_upstream()}"
            )
        # #800: BY DISPOSITION, not by a blanket sum. A blanket sum counts a
        # message whose only consumer is the PP loop body -- which cannot run
        # while the presence gate holds this rank -- and the gate then waits
        # for a consumer the gate itself is preventing. That closed cycle
        # killed serving twice on 2026-08-22, both times on PP1, with
        # `kind=admission_decision` in the inbox and 57922 withheld rounds.
        # See `pp_stash_disposition` for the full measurement and for why the
        # message must be KEPT (an abandon resets nothing, so discarding it
        # would put every later decision receive off by one).
        census = census_stash(getattr(self, "_pp_tensor_dict_inbox", {}))
        stash_reason = census.block_reason()
        if stash_reason:
            reasons.append(stash_reason)

        for attr in ("send_req_work", "send_output_work", "send_proxy_work"):
            if getattr(self, attr, None):
                reasons.append(f"{attr} is not reaped")
        if getattr(self, "last_rank_comm_queue", None):
            reasons.append("last_rank_comm_queue is not empty")
        # THE ONE-SLOT BUFFER BETWEEN THE WIRE AND THE PROCESSING, and the
        # only in-flight output state this predicate did not name (#631).
        #
        # Every other check above is about a WIRE or a QUEUE: unconsumed
        # messages, stashed inbox entries, unreaped sends, the last rank's
        # comm queue. ``pp_outputs`` is none of those. It holds an output
        # tensor dict that has ALREADY BEEN RECEIVED off the ring and is
        # waiting for the NEXT pass to turn it into tokens -- so with it
        # set, every wire is legitimately empty and quiescence passed while
        # a sampled token had not yet reached anyone's output_ids.
        # ``init_pp_loop_state`` then assigns ``pp_outputs = None`` at the
        # cutover, and that token is gone: the KV has it (the model goes on
        # writing the right continuation) but rank 0 -- the ONLY rank with a
        # detokenizer socket, in both phases -- never appends it, so the
        # client is short exactly one token.
        #
        # Measured: a request crossing pp_to_tp under load lost one token
        # per crossing, in the IDS as well as the text (' 18 19 2 21', the
        # '0' of "20" absent from output_ids itself), while the no-flip
        # control was clean 3/3.
        #
        # WAITING is the whole fix, and it cannot wedge: this only defers
        # the flip to a boundary where the pending output has been
        # processed, and the park/abandon machinery already bounds that
        # wait -- an output that never drains costs a loud abandonment,
        # which is the behaviour this feature had under load anyway, not a
        # new deadlock class. Draining it here instead was NOT chosen:
        # pp_flip_drain_tensor_dicts is corpse S precisely because a drain
        # on this path ate an output.
        if getattr(self, "pp_outputs", None) is not None:
            reasons.append(
                "pp_outputs holds a received-but-unprocessed output "
                "(its sampled token has reached no output_ids yet)"
            )

        return "; ".join(reasons) if reasons else None

    def _pp_record_slot_last_batch(self: Scheduler, slot: int) -> None:
        """``last_mbs[slot]`` names the batch that slot ran LAST ITERATION.

        #631 DEFECT R -- THE RESIDENT-CARRY LEAK, and it is one indentation
        --------------------------------------------------------------------
        This assignment used to live INSIDE ``if self.mbs[slot] is not
        None:``, one level deeper, sharing a block with the D2H sync and
        ``_pp_process_batch_result``. Sharing that block silently changed
        what the name MEANS, from

            "the batch this slot ran in its previous iteration"   (correct)

        to

            "the last non-empty batch this slot EVER ran"          (a leak)

        and the two only differ when a slot legitimately runs nothing while
        requests are still resident in it. Before strict phase purity that
        combination barely existed: a slot holding a non-empty
        ``running_batch`` produced a decode batch, so ``mbs[slot]`` was
        non-None and the entry was refreshed every cycle. STRICT PURITY
        CREATED IT ON PURPOSE -- ``get_next_batch_to_run`` returns
        ``batch_to_run = None`` for a resident decode batch in the PP layout
        (``scheduler.py``, ``phase_decode_blocked_here`` -> ``ret = None``),
        and that None is the intended signal that the PP phase has no work.

        WHAT THE STALE ENTRY THEN DOES, once per visit to the slot, forever
        --------------------------------------------------------------------
        The stale entry is an EXTEND batch, so on every later visit to this
        slot ``get_next_batch_to_run`` takes its
        ``last_batch.forward_mode.is_extend()`` branch and reaches
        ``running_batch.merge_batch(last_batch)``. ``merge_batch`` extends
        ``reqs`` IN PLACE, so the same requests are appended again, and
        again, once per cycle:

            claims 5 -> 13338 -> 26671 -> ... -> 868447   (+1 per round)

        Neither existing defence can see it, and that is not an oversight in
        either of them:

          * the self-merge guard (``scheduler.py``, "SELF-MERGE REFUSED")
            compares ``last_batch is running_batch`` -- but the stale entry
            is a DISTINCT object; and
          * ``harvest_resident_batches`` dedupes by ``id(batch)`` -- but the
            duplication is INSIDE one batch's ``reqs``, not across batches.

        A distinct object holding already-resident Reqs is the one shape
        that defeats both at once, which is exactly the hazard entry K of
        ``phase_flip_presence`` predicted for a non-idempotent merge.

        WHY UNCONDITIONAL IS THE CORRECT RULE, not merely the fixed one
        --------------------------------------------------------------------
        The non-PP loops have always written ``self.last_batch = batch``
        unconditionally, with ``batch`` possibly None. This restores that
        same semantics per slot, so both loop families now answer "what did
        the previous iteration run" the same way, and "nothing" is a real
        answer rather than a hole that preserves the previous answer.

        The ordering makes the consumption exactly-once: iteration
        ``slot - 1`` publishes ``last_mbs[slot]`` from the cycle's earlier
        ``mbs[slot]``, iteration ``slot`` consumes it, and the next cycle
        overwrites it with whatever that consumption produced -- None
        included.

        The default path is unchanged: with purity off, a resident slot
        always produces a batch, so ``mbs[slot]`` is non-None on exactly the
        iterations that previously reached this line.
        """
        self.last_mbs[slot] = self.mbs[slot]

    def init_pp_loop_state(self: Scheduler):
        # #631 J.3: THIS REBIND IS WHERE THE RESIDENT DECODE SET DIED.
        #
        # ``running_mbs`` is not scratch space -- under event_loop_pp it IS
        # the rank's resident set (running_batch/last_batch are per-slot
        # aliases, #631 J.1). Rebinding it to fresh empty batches drops
        # every resident request: unreachable Req objects whose KV rows
        # stay allocated (the leaked page the idle checker reports) and
        # whose mamba slot locks stay held (x_lru.full_lock_ref=1 ->
        # SIGQUIT). Two symptoms, one omission, measured at the cutover of
        # 2026-08-09 02:36Z.
        #
        # The rule is stated HERE and not at the cutover because this
        # function has three callers -- boot, the cutover's topology swap,
        # and event_loop_pp's own entry -- and the TP->PP leg re-dispatches
        # into that loop immediately after the cutover, so a carry
        # installed only there would be wiped by the loop it was for.
        # At boot nothing is resident and the harvest is empty, so the
        # default path is bit-for-bit unchanged.
        from sglang.srt.managers.phase_flip_resident_carry import (
            carry_across_pp_loop_init,
            harvest_resident_batches,
        )

        carried = harvest_resident_batches(self)

        # #753: which HANDOFF PROTOCOL this run uses, decided once.
        #
        # A gapped layer set inverts the pipeline's central assumption. With
        # contiguous ownership a stage's layers form one uninterrupted span,
        # so a rank can wait for its predecessor's finished hidden states and
        # only then enter the forward loop. On a gapped cut the spans
        # INTERLEAVE -- PP0 owns 0-2, PP1 owns 3, PP0 owns 4-6 -- so PP0's
        # forward cannot finish until PP1 has computed layer 3, while PP1's
        # stage-boundary receive cannot return until PP0's forward has
        # finished. That is a closed cycle, and it is not a tuning problem:
        # boot v7pp4 (2026-08-18) wedged in exactly it, with PP0 blocked in the
        # wire's receive for layer 4 and PP1/PP2 blocked in
        # `_pp_recv_typed_dict` waiting for a proxy that could never be sent.
        #
        # So under a gapped set the stage-boundary handoff is not merely
        # unnecessary, it is the deadlock. Every rank enters the loop together
        # and the crossing wire carries every activation, including each
        # stage's entry.
        self._pp_gapped_wire: bool = pp_gapped_ownership_active(self.ps.pp_size)
        if self._pp_gapped_wire:
            _refuse_known_wrong_gapped_forward()
            if self.server_args.pp_async_batch_depth > 0:
                raise ValueError(
                    "a gapped PP layer set cannot be combined with "
                    f"--pp-async-batch-depth {self.server_args.pp_async_batch_depth}. "
                    "Layer execution under a gapped set is a strict ping-pong: "
                    "every stage must be inside the SAME forward at the same "
                    "time, because each one's next layer is another's previous "
                    "one. The ordinary ring is safe -- it still launches "
                    "exactly one forward per iteration -- but async depth "
                    "commits the output stage BEFORE that launch so a rank can "
                    "enter the next forward while a peer is still in the last "
                    "one's output exchange. Their crossings then share one "
                    "ordered channel with nothing to tell the passes apart."
                )
            if getattr(self.server_args, "enable_phase_flip", False):
                raise ValueError(
                    "a gapped PP layer set cannot yet be combined with the "
                    "phase flip. The flip's armed drain reads the tensor-dict "
                    "wire on the assumption that a 'proxy' message is the only "
                    "kind it may consume mid-pass; a gapped run replaces those "
                    "with 'crossing' messages, and reconciling the two is a "
                    "separate slice. Refused here rather than allowed to strand "
                    "a crossing at the first arm."
                )

        self.pp_loop_size: int = self.ps.pp_size + self.server_args.pp_async_batch_depth
        if self._pp_gapped_wire:
            # ONE SLOT, and the reason is the offsets rather than the count.
            #
            # I set this to 1 once before (v7pp5), watched the output ring
            # starve, and blamed the ring's "fill slack" -- the unfilled slot
            # that lets a receive return early during pipeline fill. That
            # reading was wrong, and d139c463cc reverting it was wrong with it.
            # The slack was never the mechanism; it was a symptom of the two
            # gates being different expressions. A middle rank sent only
            # ``if pp_outputs`` -- what it received LAST iteration -- while
            # receiving whenever its slot held a batch, so during fill it
            # received without ever sending. Fixing the LAG (4b7ce2a81f) made a
            # middle rank forward what it just received, which turned its send
            # gate into "I received", and _pp_output_exchange_due now makes
            # that the same question on both sides.
            #
            # With symmetric gates the slot count stops being a correctness
            # knob and becomes what it always should have been: how many passes
            # may be in flight. A gapped layout permits exactly one, because
            # every stage must be inside the same pass. Keeping the ordinary
            # ring size is what broke v7pp12 -- next_first_rank_mb_id and
            # next_mb_id then name DIFFERENT slots, the two gates disagree, and
            # the iteration barrier starves with no peer provably dead.
            #
            # At one slot both indices collapse to 0, so the send gate and the
            # receive gate cannot reference different batches even in
            # principle. _pp_assert_gapped_slots_coincide checks that rather
            # than trusting the arithmetic to stay this way.
            self.pp_loop_size = 1
        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
        self.require_attn_tp_allgather = (
            not self.server_args.enable_dsa_prefill_context_parallel
        )
        # #829: the ring is being rebuilt, so every slot number recorded
        # against the old one dies here. See the function's docstring for
        # the argument and for why `_pp_flip_armed_passes` is deliberately
        # left alone.
        pp_flip_forget_ring_scoped_slots(self)

        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]] = deque()

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.launch_event = None

        # #791 PP ADMISSION UNIFORMITY. Per-pass scratch, filled and drained
        # every iteration of _event_loop_pp_body -- not part of the resident
        # carry above, since a request excluded on the pass this loop is
        # (re-)entered on simply reappears on the next pass's admission loop
        # (self.waiting_queue is never mutated for a skipped req).
        # `_pp_admission_pending_sends` is PP0-only: a rolling count of
        # decisions PP0 has sent but not yet opportunistically consumed the
        # ring's wraparound for. It exists so the wraparound CHECK is
        # deferred until at least one full lap could plausibly have
        # completed, rather than PP0 checking for THIS pass's own wraparound
        # immediately -- see _event_loop_pp_body's #791 block for why the
        # latter would re-serialize the pipeline one pass at a time. Capped
        # at `_PP_ADMISSION_PENDING_SENDS_CAP` (see that constant's
        # docstring) so it cannot grow unboundedly on a run where
        # wraparounds stay unavailable for a long stretch -- a memory bound
        # only, never a correctness one.
        self._pp_admission_incoming_effective: Optional[Dict[str, int]] = None
        # #791 CORE: the forwarded pass geometry this rank EXECUTES --
        # `rid -> (prefix_len, extend_len)`, filled from the same amended
        # decision as `_pp_admission_incoming_effective` above and None on
        # every rank that owns its own admission truth.
        self._pp_admission_incoming_schedule: Optional[Dict[str, Tuple[int, int]]] = (
            None
        )
        self._pp_admission_amended_to_forward: Optional[PPAdmissionDecision] = None
        self._pp_admission_pending_sends: deque = deque()
        # #796: the admission decision's outstanding async send, held here
        # for the same reason `send_req_work` / `send_proxy_work` /
        # `send_output_work` are held on the scheduler -- a gloo isend whose
        # handle and buffers are local temporaries is dropped when they go
        # out of scope, and the message never reaches the peer. Drained
        # every iteration by `_pp_commit_admission_send_work`.
        self._pp_admission_send_work: List[P2PWork] = []
        # #797: PASS-SCOPED, rewritten at the top of every pass by
        # `_event_loop_pp_body` before `get_next_batch_to_run` reads them.
        # Cleared here too, because a cutover re-enters the loop and a void
        # decided in the epoch that just ended names nothing in the new one --
        # the same argument `_pp_output_expected_by_slot` makes above.
        self._pp_admission_pass_voided: bool = False
        self._pp_pass_voided_incoming: bool = False
        self._pp_upstream_launched_incoming: bool = False
        # #951: pass-scoped like the three above -- scheduler.py's void guard
        # writes it, `_pp_void_pass_without_upstream_launch` reads it once, and
        # a cutover must not carry either across.
        self._pp_upstream_void_withheld_work: bool = False
        # #801-spin: consecutive no-progress voids. Cleared here for the same
        # reason as the flags above -- a cutover rebuilds the slot ring, so a
        # streak counted against the old one names nothing in the new one.
        self._pp_upstream_idle_void_streak: int = 0
        self._pp_idle_void_suppress_log: bool = False
        # #791b: PER SLOT, refreshed every pass by
        # `_pp_note_output_expectation`, read on the last rank by
        # `_pp_send_output_to_next_stage`. Per slot rather than one scalar
        # because with `pp_async_batch_depth > 0` the last rank's send gate
        # reads `next_first_rank_mb_id`, which is the slot whose decision
        # arrived `pp_async_batch_depth` passes ago, not this pass's.
        # All-False at re-entry is the correct start: no slot has an
        # expectation until PP0 publishes one, so nothing can be voided on
        # the strength of a pre-flip pass.
        self._pp_output_expected_by_slot: List[bool] = [False] * self.pp_loop_size
        # #791b: the fully chain-reconciled decision this rank forwarded for
        # each slot. The last rank rides it back to PP0 inside a void output
        # so `PPAdmissionCongruenceGuard.record_return_trip` -- which #796
        # left with no feeder at all when it removed the wraparound -- learns
        # the observed shortfall and #630's termination argument holds again.
        self._pp_admission_amended_by_slot: List[Optional[PPAdmissionDecision]] = [
            None
        ] * self.pp_loop_size
        # #797b: `self.chunked_req` as it stood before each slot's admission,
        # written every pass by `_pp_note_chunked_req_before_admission` and
        # read only when a void has to put it back. All-None at re-entry is
        # correct: a cutover rebuilds the ring, so no slot has a pre-admission
        # value until this epoch publishes one.
        self._pp_chunked_req_before_by_slot: List[Optional[Req]] = [
            None
        ] * self.pp_loop_size
        # #753: NOT assigned here any more. The inbox moved onto the pp_group
        # so the crossing wire -- a second consumer of the same channel --
        # shares it; ``_pp_tensor_dict_inbox`` below is now a read-only view of
        # that one store, which keeps every existing reader (the flip's pending
        # counts, the drain) working against the messages that actually exist.
        # Re-seed what the rebind above destroyed (#631 J.3). No-op unless
        # requests were resident, i.e. always a no-op at boot.
        carry_across_pp_loop_init(self, carried)

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            try:
                model_runner = self.tp_worker.model_runner
                model_config = model_runner.model_config
                input_ids_list: List[array[int]] = []
                for i in range(128):
                    chunk_size = int(
                        self.chunked_prefill_size * 1.25
                        - i * (self.chunked_prefill_size * 1.25 // 128)
                    )
                    if chunk_size <= 0:
                        break
                    input_ids = array(
                        "q",
                        np.random.randint(
                            0, 10000, size=chunk_size, dtype=np.int64
                        ).tobytes(),
                    )
                    input_ids_list.append(input_ids)

                sampling_params = SamplingParams(
                    temperature=0,
                    max_new_tokens=1,
                )
                # Create and profile requests
                for i, input_ids in enumerate(
                    tqdm(
                        input_ids_list,
                        desc="Profiling prefill latency for dynamic chunking",
                    )
                ):
                    req = Req(
                        rid=str(i),
                        origin_input_text="",
                        origin_input_ids=input_ids,
                        sampling_params=sampling_params,
                    )
                    req.full_untruncated_fill_ids = req.origin_input_ids
                    req.logprob_start_len = -1
                    # The abort path below releases whatever the in-flight
                    # probe is holding; without this the request that RAISED
                    # keeps its slots for the life of the process.
                    self._dyn_chunk_probe_req = req
                    req.set_extend_range(
                        len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                    )

                    # Prepare batch
                    batch = ScheduleBatch.init_new(
                        [req],
                        self.req_to_token_pool,
                        self.token_to_kv_pool_allocator,
                        self.tree_cache,
                        self.model_config,
                        False,
                        self.spec_algorithm,
                    )

                    current_seq_len = req.extend_range.end

                    if is_dp_attention_enabled():
                        # For profiling, we only have one request on PP0
                        # Set global_num_tokens to indicate this rank has tokens, others have 0
                        dp_size = get_attention_dp_size()
                        global_num_tokens = [0] * dp_size
                        dp_rank = get_attention_dp_rank()
                        global_num_tokens[dp_rank] = current_seq_len
                        batch.global_num_tokens = global_num_tokens
                        batch.global_num_tokens_for_logprob = global_num_tokens

                    hs = (
                        getattr(model_config, "hc_hidden_size", None)
                        or model_config.hidden_size
                    )
                    proxy_tensors = {
                        "hidden_states": torch.zeros(
                            (current_seq_len, hs),
                            dtype=model_config.dtype,
                            device=self.device,
                        ),
                        "residual": torch.zeros(
                            (current_seq_len, model_config.hidden_size),
                            dtype=model_config.dtype,
                            device=self.device,
                        ),
                    }
                    pp_proxy_topk_size = model_runner.get_pp_proxy_topk_size()
                    if pp_proxy_topk_size is not None:
                        proxy_tensors["topk_indices"] = torch.zeros(
                            (current_seq_len, pp_proxy_topk_size),
                            dtype=torch.int32,
                            device=self.device,
                        )

                    pp_proxy = PPProxyTensors(proxy_tensors)

                    # Measure latency with device synchronization for accurate timing
                    device_module = get_device_module()
                    # Synchronize before starting timing to ensure clean measurement
                    device_module.synchronize()

                    start = time.perf_counter()
                    batch.prepare_for_extend()

                    # Resolve deferred H2D: prepare_for_extend now leaves input_ids=None
                    if (
                        batch.input_ids is None
                        and batch.prefill_input_ids_cpu is not None
                    ):
                        batch.input_ids = batch.prefill_input_ids_cpu.to(
                            self.device, non_blocking=True
                        )
                        batch.prefill_input_ids_cpu = None

                    forward_batch = ForwardBatch.init_new(batch, model_runner)
                    set_is_extend_in_batch(batch.forward_mode.is_extend())

                    _ = model_runner.forward(
                        forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                    )

                    # Synchronize after forward to ensure GPU operations complete
                    device_module.synchronize()

                    latency_seconds = time.perf_counter() - start
                    latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                    seq_lens.append(len(input_ids))
                    latencies.append(latency_ms)

                    # Release everything this probe took. KV, MAMBA, req slot.
                    _release_dynamic_chunk_probe(self, req)
                    self._dyn_chunk_probe_req = None

                logger.info(
                    f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                    f"seq_lens={seq_lens}, latencies_ms={latencies}"
                )

                if self.ps.attn_tp_size > 1:
                    data_to_sync_tp = [seq_lens, latencies]
                    data_to_sync_tp = broadcast_pyobj(
                        data_to_sync_tp,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                    seq_lens, latencies = data_to_sync_tp

                if self.ps.attn_cp_size > 1:
                    data_to_sync_tp = [seq_lens, latencies]
                    data_to_sync_tp = broadcast_pyobj(
                        data_to_sync_tp,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            except Exception as e:
                # #661: THE PROFILE MAY FAIL, AND THE FAILURE MUST NOT
                # ESCAPE PAST THE BROADCAST BELOW.
                #
                # This block runs on PP0 only, and the function ends in a
                # pp_group.broadcast_object_list every rank enters
                # unconditionally. Letting the exception propagate leaves
                # PP0 in the caller's rank-local `except` (scheduler.py
                # init_chunked_prefill, which sets enable_dynamic_chunking
                # = False for ITSELF and walks on into the event loop)
                # while every other rank blocks forever in a broadcast
                # whose src will never arrive. Measured on this rig
                # 2026-08-11: PP0 raised `alloc_req_slots runs out of
                # memory ... available_size()=4` under
                # --max-running-requests 4, and PP1/PP2 sat in
                # broadcast_object_list until the boot was killed -- the
                # HTTP port never opened.
                #
                # So the failure is caught HERE, on the rank that can have
                # it, and is published as DATA: empty samples. Every rank
                # still enters the broadcast, every rank receives the same
                # empty lists, and every rank then declines dynamic
                # chunking together in the check below. A rank-local
                # `except` around a collective is a deadlock; a broadcast
                # verdict is not.
                logger.warning(
                    "[PP Dynamic Chunk] [PP0] profiling failed (%s); "
                    "publishing an EMPTY sample set so every rank "
                    "declines dynamic chunking together instead of "
                    "blocking in the broadcast below.",
                    e,
                )
                # AND HAND BACK WHAT THE FAILED PROBE HOLDS. Publishing empty
                # samples makes every rank decline together, which is right --
                # but the probe that raised is still holding a req slot, a
                # mamba slot and its KV, and nothing else will ever free them.
                _release_dynamic_chunk_probe(
                    self, getattr(self, "_dyn_chunk_probe_req", None)
                )
                self._dyn_chunk_probe_req = None
                seq_lens, latencies = [], []
        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # #661: THE VERDICT, AFTER THE COLLECTIVE AND THEREFORE UNIFORM.
        #
        # Empty samples mean PP0's profiling failed (see the except above). It
        # is raised HERE rather than on PP0 at the point of failure, because
        # every rank now holds the same empty lists and every rank raises the
        # same error in the same place. The caller's `except` in
        # init_chunked_prefill then disables dynamic chunking on ALL ranks --
        # which is what it always intended to do, and could not, while the
        # failure was rank-local.
        #
        # Fitting an empty sample set would be the other bug: ChunkSizePredictor
        # would carry meaningless coefficients and every predicted width would
        # be a guess dressed as a measurement.
        if not seq_lens or not latencies:
            raise RuntimeError(
                "[PP Dynamic Chunk] profiling produced no samples on PP0; "
                "every rank declines dynamic chunking together. The usual "
                "cause is that the profile's request slots exceed "
                "--max-running-requests."
            )

        # Quadratic model: f(l) = al^2 + bl + c
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = self.max_prefill_tokens
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.ps.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    rids_to_check=good_consensus_bootstrapped_rids
                    + bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [[req.rid for req in good_reqs], [req.rid for req in failed_reqs]]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            good_bootstrapped_rids, bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_bootstrapped_rids = list(
                set(prev_good_bootstrapped_rids) & set(curr_good_bootstrapped_rids)
            )
            bad_bootstrapped_rids = list(
                set(prev_bad_bootstrapped_rids) | set(curr_bad_bootstrapped_rids)
            )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_prefill_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
    ]:
        # #753: UNDER A GAPPED SET THE EXCHANGE RUNS BEFORE THE FLUSH.
        #
        # The shipped order flushes last iteration's isends and only then posts
        # this iteration's receives. That is safe while every rank sends before
        # it receives, because the peer's matching receive was already posted
        # in the same iteration as the send. The gapped ordering breaks that
        # symmetry -- a middle rank receives first so it can forward what it
        # just got -- and the flush then waits for a receive that this
        # iteration has not reached yet. Every rank flushing a send that no
        # rank has posted a receive for is a closed cycle: boot v7pp9 sat in
        # _pp_commit_comm_work on all three ranks, three works deep, stuck on
        # the first.
        #
        # Flushing AFTER the exchange keeps the property the flush actually
        # needs -- that a send is not waited on until its peer has had the
        # chance to take it -- rather than the accident of send-before-receive
        # that used to supply it. The pending list is captured first because
        # the exchange rebinds send_output_work to this iteration's work.
        pending_output_work = self.send_output_work
        gapped = getattr(self, "_pp_gapped_wire", False)
        if not gapped:
            self._pp_commit_comm_work(work=pending_output_work)
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        if gapped:
            # Now that every rank has posted this iteration's receives, last
            # iteration's sends can be waited on without closing a cycle.
            self._pp_commit_comm_work(work=pending_output_work)
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = (
                self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
            )
            p2p_work = point_to_point_pyobj(
                data,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                ((self.ps.pp_rank + 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                async_send=async_send,
            )
            # #631 G: STRICTLY AFTER THE POST, never before. The isend is
            # on the wire by the time this line runs, so the only skew a
            # peer can observe is counter-lags-send -- a real message seen
            # one poll late. Publishing first would advertise a message
            # that does not exist yet and send a peer into an UNBOUNDED
            # blocking recv, which is the wedge class this whole feature
            # removes. See phase_flip_counters for the full argument.
            self._pp_flip_bump_sent(CHAN_REQ)
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = (
                self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
            )
            data = point_to_point_pyobj(
                [],
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.ps.pp_rank - 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
            )
        else:
            data = None

        if self.ps.attn_tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )

        if self.ps.attn_cp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_cp_group.rank,
                self.attn_cp_cpu_group,
                src=self.attn_cp_group.ranks[0],
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_boundary_stats(self: Scheduler) -> Optional[PPBoundaryStats]:
        """The boundary accountant, or None when SGLANG_PP_BOUNDARY_STATS=0.

        Resolved once per scheduler process and cached, so the default path
        costs one attribute read per crossing and nothing else.
        """
        stats = getattr(self, "_pp_boundary_stats_obj", _PP_STATS_UNSET)
        if stats is _PP_STATS_UNSET:
            every = envs.SGLANG_PP_BOUNDARY_STATS.get()
            stats = PPBoundaryStats(every) if every > 0 else None
            self._pp_boundary_stats_obj = stats
        return stats

    @property
    def _pp_tensor_dict_inbox(self):
        """The shared ``(src, kind)`` inbox for this rank's PP tensor-dict wire.

        A view, not a second store. Readers that only sum queue lengths -- the
        flip's pending-message counts -- are unaffected by the richer key; the
        two consumers that index it (this mixin's receive and the armed drain)
        go through ``pp_typed_channel`` so the key stays in one place.
        """
        from sglang.srt.distributed.pp_typed_channel import typed_inbox

        return typed_inbox(self.pp_group)

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
        msg_type: str = "default",
        stamp: Optional[tuple] = None,
    ):
        # #753: the counterpart of the receive-side skip. A gapped run has
        # already delivered this rank's last owned layer to whoever owns the
        # next one, over the wire, inside the loop; the hidden states sitting
        # in the result at the end of the forward are that same tensor and the
        # next stage has no receive posted for them. Sending anyway would leave
        # one unmatched message per pass on the channel -- the bounded-recv
        # corpse, from the sender's side.
        if msg_type == "proxy" and getattr(self, "_pp_gapped_wire", False):
            return []
        # Warn once if using default untyped messages
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        # #631 VARIANT B: make the proxy pairing NON-POSITIONAL.
        #
        # The consumer used to pair "whatever came off the wire this slot
        # iteration" with "whatever batch I have this slot iteration".
        # PPProxyTensors carried no identity, so ONE stranded message put
        # every later receive off by one, silently and for ever -- the
        # mispairing root cause (specimen pp_proxy_mispair_20260809T0626Z).
        #
        # The stamp gives the message its own identity so a leftover from an
        # abandoned window matches nothing and is dropped LOUDLY instead of
        # being computed on. No rank's launch timing moves and no
        # synchronisation point is added, which is what the refined design
        # law demands after the resume gate wedged the instance (HANDOFF §7).
        #
        # It rides as a dict entry because that is what crosses the wire, and
        # that is SAFE BY PRECEDENT rather than by hope: __msg_type__ above is
        # itself a non-tensor entry that has always travelled in this dict and
        # into PPProxyTensors without a consumer tripping over it.
        if stamp is not None:
            tensor_dict["__stamp__"] = stamp
        p2p_work = []
        stats = self._pp_boundary_stats()
        started = time.perf_counter() if stats else 0.0
        # #789: publish "a send for you is now under way" BEFORE the call,
        # because the call itself can be a RENDEZVOUS. The first p2p op on
        # a torch NCCL group creates its 2-rank communicator lazily, and
        # `isend` then does not return until the downstream enters the
        # matching receive -- so a counter published only AFTER the post
        # cannot exist in the window where the downstream is deciding
        # whether entering that receive is safe. Boots instr7/instr8
        # (2026-08-21) both died there: PP0 inside this isend, PP1 and PP2
        # in `_pp_wait_for_proxy_readiness` refusing to collect it, the gate
        # raising after 30 s over "posted 1681, consumed 1681".
        #
        # This does NOT relax the counter module's ordering rule. `sent`
        # below still means "on the wire" and is still published strictly
        # after the post; this is a second counter with a second meaning.
        # See PhaseFlipCounters.bump_attempted for the full argument.
        self._pp_flip_bump_attempted(CHAN_DICT)
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(
                    self.attn_tp_group if self.require_attn_tp_allgather else None
                ),
                async_send=async_send,
            )
        )
        if stats:
            stats.record("send", tensor_dict, time.perf_counter() - started)
        # #631 G: after the post, for the same reason as the request
        # chain. ONE counter for this wire, shared by 'proxy' and 'output'
        # -- they are demultiplexed by __msg_type__ AFTER coming off it,
        # so counting them apart would let a rank call a wire empty while
        # a message of the other kind was still on it.
        self._pp_flip_bump_sent(CHAN_DICT)
        return p2p_work

    def _pp_recv_typed_dict(
        self: Scheduler,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
    ) -> Dict[str, torch.Tensor]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind.
        """
        # #753: the inbox lives on the GROUP, not on this scheduler, because
        # the crossing wire is a SECOND consumer of the same channel. Two
        # private stashes on one wire is not a demultiplexer -- each side would
        # hold messages the other is blocking for. See pp_typed_channel.
        stats = self._pp_boundary_stats()

        def _off_the_wire(tensor_dict) -> None:
            if stats:
                # The elapsed time is charged to the receive that actually
                # blocked; a message served from the inbox never reaches here.
                stats.record("recv", tensor_dict, time.perf_counter() - started[0])
            # #631 G: counted here, off the WIRE, before the demultiplex --
            # a stashed message has still left the wire, and the upstream's
            # blocking commit is waiting on exactly that fact.
            self._pp_flip_bump_consumed(CHAN_DICT)
            started[0] = time.perf_counter()

        started = [time.perf_counter()]
        # #821: MARK THE ONE PLACE A PP RANK CAN DISAPPEAR SILENTLY.
        #
        # Every blocking dict receive in this file funnels through the call
        # below -- proxy, output, and the mandatory admission decision alike.
        # A rank parked here is doing nothing a counter can see: `forward_ct`
        # is frozen and `cur_batch_for_debug` still holds whatever the PREVIOUS
        # pass left, which on an idle-then-wedged rank is None. The scheduler
        # watchdog's own activity predicate reads exactly those two
        # (invariant_checker.py, `create_scheduler_watchdog`), so it concludes
        # "not active" and never fires: the wedge produces no SIGQUIT, no
        # py-spy dump, no log line -- only silence until a launcher tears the
        # tree down. That is why the #816 specimen has no watchdog line in it.
        #
        # This timestamp is the missing input, and it is deliberately a
        # TIMESTAMP rather than a boolean: a healthy PP rank is blocked here
        # for a few milliseconds on almost every pass (that is how the ring
        # paces itself), and an idle server is blocked here nearly all the
        # time, so a boolean would make `is_active` permanently true on a
        # perfectly healthy idle boot and the watchdog would SIGQUIT it. The
        # reader decides what is too long; this only records since when.
        #
        # Cleared in `finally` so a receive that RAISES -- the gloo
        # "Connection closed by peer" a dead peer produces, see
        # test_pp_dead_peer_is_not_the_wedge_801.py -- does not leave a stale
        # timestamp behind that would later read as a wedge.
        self._pp_blocked_recv_since = time.monotonic()
        # #824 W5(b): stamp WHICH arm, not merely that some arm is blocked.
        # Three channels ride this one call (proxy / output / admission) and
        # a fourth blocking receive lives in the request-relay chain; the
        # watchdog line named none of them on boot_827.
        self._pp_blocked_recv_arm = f"typed-dict/{expected_kind}"
        try:
            tensor_dict = recv_typed_tensor_dict(
                self.pp_group,
                expected_kind,
                src=None,
                all_gather_group=all_gather_group,
                on_message=_off_the_wire,
            )
        finally:
            self._pp_blocked_recv_since = None
            self._pp_blocked_recv_arm = None
        if expected_kind == "default":
            logger.warning_once(
                f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        return tensor_dict

    def _pp_note_output_expectation(
        self: Scheduler,
        mb_id: int,
        expects_output: bool,
        amended: Optional[PPAdmissionDecision],
    ) -> None:
        """#791b: record this pass's output-ring verdict for slot ``mb_id``.

        Written on EVERY pass by every rank, so a slot can never be voided on
        a stale expectation: the value the last rank reads was published for
        that same slot, in the same generation, by the rank that will apply
        it on the receiving side.

        Tolerant of a stand-in that never ran ``init_pp_loop_state``, as this
        file's own convention requires (#787): a holder without the arrays
        grows them here rather than raising.
        """
        size = max(int(getattr(self, "pp_loop_size", 0) or 0), int(mb_id) + 1)
        flags = getattr(self, "_pp_output_expected_by_slot", None)
        if flags is None or len(flags) < size:
            flags = list(flags or []) + [False] * (size - len(flags or []))
            self._pp_output_expected_by_slot = flags
        flags[mb_id] = bool(expects_output)
        carried = getattr(self, "_pp_admission_amended_by_slot", None)
        if carried is None or len(carried) < size:
            carried = list(carried or []) + [None] * (size - len(carried or []))
            self._pp_admission_amended_by_slot = carried
        carried[mb_id] = amended

    def _pp_note_chunked_req_before_admission(self: Scheduler, mb_id: int) -> None:
        """#797b: remember this slot's chunked request as it stands NOW.

        Written on EVERY pass by every rank, immediately before
        `get_next_batch_to_run`, which is the only place `self.chunked_req`
        moves. PER SLOT and not one scalar, for the same reason
        `_pp_output_expected_by_slot` is: the void for a slot is absorbed
        `pp_size - 1` passes after that slot was admitted, and by then the
        scalar would name a different round.

        Tolerant of a stand-in that never ran `init_pp_loop_state`, as this
        file's convention requires (#787): a holder without the array grows
        one here rather than raising.
        """
        size = max(int(getattr(self, "pp_loop_size", 0) or 0), int(mb_id) + 1)
        carried = getattr(self, "_pp_chunked_req_before_by_slot", None)
        if carried is None or len(carried) < size:
            carried = list(carried or []) + [None] * (size - len(carried or []))
            self._pp_chunked_req_before_by_slot = carried
        carried[int(mb_id)] = getattr(self, "chunked_req", None)

    def _pp_output_expected_for_slot(self: Scheduler, mb_id: int) -> bool:
        """#791b: did the FIRST rank say it will receive an output for this
        slot? False whenever nothing was ever published for it."""
        flags = getattr(self, "_pp_output_expected_by_slot", None)
        if not flags or mb_id >= len(flags):
            return False
        return bool(flags[mb_id])

    def _pp_pass_retraction_reason(self: Scheduler, mb_id: int) -> Optional[str]:
        """#791c: did THIS rank narrow its own batch for slot ``mb_id``?

        Reads the amended decision `_pp_note_output_expectation` already
        records per slot (#791b) -- written on EVERY pass by every non-first
        rank, at the top of the pass, strictly before the proxy receive -- so
        this asks a question that is fully answered before the message it
        judges has even arrived. That is what makes it a discriminator the
        RECEIVER CAN PREDICT rather than another label on the wire.

        getattr-based and None-tolerant throughout, by this file's stand-in
        convention (#787): a holder with no slot array, a rank with no
        `ps.pp_rank`, `pp_size <= 1` and the first rank all yield None, which
        `_pp_recv_proxy_tensors` reads as "nothing known against this pass" --
        exactly the behaviour that shipped before #791c.
        """
        carried = getattr(self, "_pp_admission_amended_by_slot", None)
        if not carried or int(mb_id) < 0 or int(mb_id) >= len(carried):
            return None
        rank = getattr(getattr(self, "ps", None), "pp_rank", None)
        return pp_proxy_pass_retraction_reason(carried[int(mb_id)], rank)

    def _pp_send_admission_decision(
        self: Scheduler,
        decision: PPAdmissionDecision,
        *,
        expects_output: bool = False,
        pass_voided: bool = False,
        launched: bool = False,
    ) -> None:
        """#791: forward this pass's admission decision to the next stage.

        UNLIKE the proxy send (`_pp_send_dict_to_next_stage(..., msg_type=
        "proxy")` above, gated on `not self.pp_group.is_last_rank`), this is
        never gated on rank position. The whole point of this wire is that
        it keeps travelling past the last stage and wraps back to PP0 --
        `pp_group.send_tensor_dict`'s default `dst=None` already resolves to
        "next rank in the ring", which lands on PP0 from the last rank the
        same way the output ring wraps -- so PP0 can close the #630 learning
        loop via `PPAdmissionCongruenceGuard.record_return_trip`.

        SENT EVERY PASS, EVEN WHEN EMPTY (`entries=()`). Withholding it
        whenever a rank's own verdict is empty would recreate exactly the
        defect this module exists to close: a downstream rank's receive
        gated on ITS OWN cur_batch rather than on what upstream actually
        decided (`_event_loop_pp_body`'s `if cur_batch: ... _pp_recv_proxy_
        tensors(mb_id)`, the root defect this whole feature targets).

        NOT SENT BY THE LAST RANK -- #796, and this is a CORRECTION to the
        ring described above. The wraparound edge (last rank -> PP0) has no
        matching receive on PP0 by construction: PP0 never issues a blocking
        receive for it (that closed a ring, the fourth specimen -- see
        `_pp_try_recv_admission_decision`), it only PEEKS an inbox that is
        filled solely as a side effect of the per-iteration OUTPUT receive
        -- and that receive early-returns whenever the slot is empty
        (`_pp_send_recv_and_preprocess_output_tensors`'s `_do_recv`: `if
        target is None ... return`), which is every idle pass. So the
        wraparound was one unmatched message per pass on the channel: the
        exact corpse `_pp_send_dict_to_next_stage` already refuses for the
        proxy under a gapped wire ("Sending anyway would leave one unmatched
        message per pass on the channel -- the bounded-recv corpse, from the
        sender's side"). The same law is now applied here: A RANK MUST NOT
        POST A SEND NO PEER IS REQUIRED TO TAKE.

        WHAT THAT COSTS, STATED PLAINLY: `PPAdmissionCongruenceGuard.
        record_return_trip` no longer runs, because no lap can arrive. It
        was never load-bearing for TERMINATION -- #630's retry livelock
        terminates on the floor the guard learns from the RETRACTION
        (`prefix_len_for`, strictly decreasing and non-negative), and
        `record_return_trip` only CLEARED that floor early. The floors are
        keyed by rid and a rid dies with its request, so the residual cost
        is that one request's reuse stays suppressed for the rest of ITS
        life instead of recovering mid-flight. A bounded, per-request
        performance effect -- traded for a channel with no unmatched sends
        on it.

        THE WORK HANDLE IS RETAINED, NOT DISCARDED -- #796, the defect that
        wedged six boots. This used to call `_pp_send_dict_to_next_stage`
        and throw the returned `P2PWork` list away, which made it the ONLY
        async channel in `_event_loop_pp_body` that did not keep its handle
        alive on the scheduler (`send_req_work`, `send_proxy_work` and
        `send_output_work` all do, and all are committed later). A gloo
        isend whose handle and backing buffers are local temporaries is
        dropped when they go out of scope: the message never reaches the
        peer, and every downstream rank then blocks for ever at
        `_event_loop_pp_body`'s decision receive waiting for something that
        was never actually on the wire. That is why three successive
        re-orderings of this send (#791, #795, and the wraparound fix) each
        corrected a real defect and none of them saved a boot -- the
        ordering of a message that does not exist cannot matter.
        Reproduced hermetically in
        test_pp_admission_send_handle_dropped_796.py, whose dropped-handle
        arm reproduces the six-boot py-spy signature exactly (PP0 in the
        chain flush one iteration ahead, PP1 and PP2 both in this channel's
        receive). Committed at the end of the iteration by
        `_pp_commit_admission_send_work`.

        Still fire-and-forget WITHIN the pass: nothing on the admission path
        may block on a peer here (scheduler.py's `_get_new_batch_prefill_raw`
        NO COLLECTIVE note -- the 2026-08-17 deadlock family this must not
        repeat). Retaining a handle is not blocking on one.
        """
        if self.ps.pp_size <= 1:
            return
        if self.pp_group.is_last_rank:
            return
        tensor_dict = pp_admission_decision_to_wire(decision)
        # #791b: PP0's output-ring verdict for this slot rides along. See
        # `_PP_OUTPUT_EXPECTED_KEY` for why the fact has to travel instead of
        # being re-derived per rank, and `_pp_send_output_to_next_stage` for
        # what the last rank does with it.
        tensor_dict[_PP_OUTPUT_EXPECTED_KEY] = bool(expects_output)
        # #797: the two facts a #791 retraction makes the next rank need. See
        # each key's own comment; `pass_voided` is OR-ed along the chain and
        # `launched` is overwritten by every hop.
        tensor_dict[_PP_PASS_VOIDED_KEY] = bool(pass_voided)
        tensor_dict[_PP_UPSTREAM_LAUNCHED_KEY] = bool(launched)
        works = self._pp_send_dict_to_next_stage(
            tensor_dict, async_send=True, msg_type=ADMISSION_DECISION_KIND
        )
        # Tolerant of a stand-in that never ran `init_pp_loop_state`, as this
        # file's own convention requires (#787): a holder without the field
        # simply grows one here.
        pending = getattr(self, "_pp_admission_send_work", None)
        if pending is None:
            pending = []
            self._pp_admission_send_work = pending
        pending.extend(works or [])

    def _pp_commit_admission_send_work(self: Scheduler) -> None:
        """#796: reap this pass's admission-decision send.

        Called once per iteration from `_event_loop_pp_body`, immediately
        BEFORE `_pp_commit_pending_req_work`. That order is deliberate and
        the weaker of the two waits comes first: this one is satisfied as
        soon as the next rank reaches its decision receive at the TOP of the
        same pass, whereas the chain flush is not satisfied until that rank
        reaches the top of the NEXT pass. A wait that is already implied by
        one placed after it can never be the wait that closes a cycle.

        Safe to wait on unconditionally. Every remaining send on this
        channel is matched by construction: rank r sends to rank r+1 exactly
        once per pass, rank r+1 issues exactly one blocking receive for it
        at the top of that same pass, and the last rank now posts nothing
        (see `_pp_send_admission_decision`). So this is a wait on a message
        the peer is REQUIRED to take -- the same property
        `_pp_commit_pending_req_work` relies on -- or, on a pass that sent
        nothing, a wait on an empty list.
        """
        pending = getattr(self, "_pp_admission_send_work", None)
        if pending:
            self._pp_commit_comm_work(pending)

    def _pp_recv_admission_decision(self: Scheduler) -> Optional[PPAdmissionDecision]:
        """#791: this pass's inbound admission decision (blocking receive).

        Always issued, never gated on this rank's own local state -- see
        `_pp_send_admission_decision`'s docstring for why that gating would
        be the defect, not the fix. Positioned in `_event_loop_pp_body`
        strictly BEFORE `get_next_batch_to_run`, so this rank's own
        admission loop can be DRIVEN by the decision instead of
        independently re-deriving one that might disagree with it -- and
        strictly before `_pp_recv_proxy_tensors` / `_pp_wait_for_proxy_
        readiness` (#789), so an ordinary prefix-length divergence is
        degraded here and never has to reach that contract's raise path
        (the #791 task's ordering requirement).
        """
        if self.ps.pp_size <= 1:
            return None
        message = self._pp_recv_typed_dict(expected_kind=ADMISSION_DECISION_KIND)
        # #791b: PP0's output-ring verdict for this slot, published on the
        # same message. Absent means False -- an older sender, or a stand-in,
        # then behaves exactly as it did before this key existed.
        self._pp_output_expected_incoming = bool(
            message.get(_PP_OUTPUT_EXPECTED_KEY, False)
        )
        # #797: the two facts a retraction upstream of this rank makes it
        # need. Absent means False on both, for the same reason.
        self._pp_pass_voided_incoming = bool(message.get(_PP_PASS_VOIDED_KEY, False))
        self._pp_upstream_launched_incoming = bool(
            message.get(_PP_UPSTREAM_LAUNCHED_KEY, False)
        )
        return pp_admission_decision_from_wire(message)

    def _pp_try_recv_admission_decision(
        self: Scheduler,
    ) -> Optional[PPAdmissionDecision]:
        """PP0's OPPORTUNISTIC counterpart to `_pp_recv_admission_decision`,
        used ONLY for the ring wraparound check in `_event_loop_pp_body`
        (never for the ordinary forward receive non-first ranks issue at
        the top of the loop -- those genuinely need to block to make
        progress, and are unaffected by this method).

        THE DEADLOCK THIS CLOSES -- the fourth of the "a rank must never
        block on a peer for something not required for this iteration's
        forward progress" family, measured live 2026-08-20: boot reached
        health, froze on the first request, zero GPU utilisation on all
        ranks. py-spy on all three: PP0 blocked in exactly the wraparound
        receive this method replaces; PP1 and PP2 both blocked at the
        forward receive (`_event_loop_pp_body`'s #791 block, non-first-rank
        branch) waiting for PP0's NEXT decision. Closed ring: PP0 would not
        send again until its wraparound receive returned, and the only
        ranks that could complete that lap were the ones waiting on PP0's
        next send.

        `record_return_trip` (`PPAdmissionCongruenceGuard`) is a pure
        LEARNING step -- it teaches the guard the downstream coverage a lap
        actually observed and clears a rid's learned floor early. Nothing
        about THIS iteration's forward progress depends on it, so the
        receive that feeds it must never be allowed to block this
        iteration on a peer.

        NON-BLOCKING BY CONSTRUCTION, NOT BY TIMEOUT. This never calls the
        underlying blocking gloo receive itself. It only PEEKS
        `pp_typed_channel.typed_inbox` for an already-stashed
        `(src, ADMISSION_DECISION_KIND)` message -- the exact precedent
        `_pp_wait_for_proxy_readiness` set (see its docstring: "AN
        ALREADY-STASHED MESSAGE IS ALSO A POSITIVE SIGNAL") for gating a
        blocking receive on inbox presence rather than trusting silence --
        and only when one is already there does it call
        `_pp_recv_admission_decision`, whose own fast path
        (`recv_typed_tensor_dict`'s `take_typed` at the top, in
        pp_typed_channel.py) then pops it with no further wire activity.
        The peek and the pop use the identical `(src, kind)` key, computed
        the same way (`resolve_src`), so there is no window between them
        for a second message to arrive and be mistaken for this one.

        WHY A LAP CAN ALREADY BE PRESENT WITHOUT THIS RANK EVER HAVING
        ISSUED A DEDICATED BLOCKING RECEIVE FOR IT. PP0's mandatory
        per-iteration "output" receive (`_pp_recv_dict_from_prev_stage`)
        reads from the SAME directed pair -- the last rank, ring-wrapped to
        PP0 -- that the amended admission decision also travels on.
        `recv_typed_tensor_dict` drains messages off that wire until it
        finds the kind it asked for ("output"), stashing every other kind
        it passes over -- including ADMISSION_DECISION_KIND -- into the
        inbox along the way. So a lap that arrived interleaved with an
        "output" message is typically already sitting here by the time
        this runs, discovered as a side effect of a receive the loop was
        going to issue anyway, at no extra wire cost. If it is not there
        yet, this iteration simply carries on without it -- see the call
        site in `_event_loop_pp_body` for what that costs.

        #796 UPDATE, AND IT MAKES THIS METHOD DORMANT RATHER THAN WRONG:
        the last rank no longer EMITS the wraparound at all, because PP0 was
        never required to take it and one unmatched message per pass is the
        bounded-recv corpse (see `_pp_send_admission_decision` for the full
        argument and for what the lost `record_return_trip` learning costs).
        No lap can therefore arrive any more, and this returns None every
        time. It is kept, not deleted, for the same reason
        `pp_pump_send_req_work` is kept: it is exactly where a working
        wraparound would be consumed if PP0 ever gains a receive it is
        genuinely required to issue, and deleting it would erase the record
        of what the ring was for. It is NOT a mechanism anything may rely
        on, and nothing does -- the call site treats a None as an ordinary
        "not this pass".

        Returns None immediately, touching nothing, when `pp_size <= 1`
        (matching `_pp_recv_admission_decision`'s own no-op contract) or
        when no lap is already in hand.
        """
        if self.ps.pp_size <= 1:
            return None
        src = resolve_src(self.pp_group, None)
        if not typed_inbox(self.pp_group).get((src, ADMISSION_DECISION_KIND)):
            return None
        return self._pp_recv_admission_decision()

    def _pp_reconcile_incoming_admission(
        self: Scheduler, decision: PPAdmissionDecision
    ) -> Tuple[Dict[str, int], PPAdmissionDecision]:
        """#791: this (downstream) rank's reconciliation of a received
        decision against its OWN local radix-cache state.

        For every named, still-admitted rid, freshly matches it against
        THIS rank's own tree_cache (`req.init_next_round_input` -- safe to
        call again here even though the admission loop below calls it again
        too, per the idempotency already relied on for this exact call by
        schedule_policy.py's pre-admission priority pass) to get a real
        local candidate length, then hands everything to the pure
        `reconcile_pp_admission_decision` (#791/#630) for the actual
        safe-truncate / unsafe-retract verdict and the exactly-once warning.

        FOUR PLACES A REQUEST CAN LIVE, and a rid found in NONE of them is
        `UNKNOWN_MATCH`, never 0.

        The sentence that stood here said a rid missing from `waiting_queue`
        "is treated as a local match of 0 -- physically indistinguishable from
        'this rank's cache has nothing for it' ... it needs no special case".
        That was wrong three times over, and each time it cost a boot: #797c
        (the rid is in `chunked_req`), #798 (it is in ANOTHER slot's chunked
        req), #944 (it is in the RUNNING BATCH). The two facts are not
        indistinguishable at all -- one is a statement about this rank's cache,
        the other about this lookup -- and spelling them the same way is what
        made three instances of one class read as three separate defects. The
        lookup chain below is exhaustive as far as it is known to be; what has
        changed is that running out of places to look now says so.

        #797c, THE FIRST OF THE THREE, IN FULL -- kept because it is the
        clearest specimen of the false negative and the reason the second and
        third lookups exist at all.

        The CHUNKED request is dropped from `waiting_queue` the moment
        it is first admitted (scheduler.py's `self.waiting_queue = [x for x in
        self.waiting_queue if x not in can_run_set]`) and lives in
        `self.chunked_req` from then on -- so the lookup above misses it by
        construction on every round but its first, and the miss defaults to a
        local match of 0 for a request this rank is DEMONSTRABLY mid-prefill
        on. That false negative retracted the same rid three times in one
        second on boot instr19 while `told` grew by one chunk each time. It is
        answered from the request's own progress instead; see
        `pp_chunked_local_match` for why `extend_range.end` is the right
        quantity and a radix re-match is not.
        """
        pp_size = self.ps.pp_size
        if pp_size <= 1:
            return {}, decision
        by_rid = {req.rid: req for req in self.waiting_queue}
        chunked = getattr(self, "chunked_req", None)
        chunked_rid = getattr(chunked, "rid", None)
        slot_chunked = pp_chunked_req_for_slot(self, decision.mb_id)
        slot_chunked_rid = getattr(slot_chunked, "rid", None)
        local_match_lens: Dict[str, int] = {}
        for entry in decision.entries:
            if not entry.admitted or entry.retracted:
                continue
            req = by_rid.get(entry.rid)
            if req is None:
                if chunked_rid is not None and entry.rid == chunked_rid:
                    computed = pp_chunked_local_match(chunked)
                    if computed is not None:
                        local_match_lens[entry.rid] = computed
                        continue
                # #798: AND THE SLOT THE DECISION NAMES, which is the lookup
                # whose absence livelocked boot_798_0822_0829 for 13 minutes
                # without serving one request. `self.chunked_req` is a single
                # scheduler-wide field, but `_pp_void_own_batch` restores it
                # PER SLOT out of `_pp_chunked_req_before_by_slot[mb_id]`, so
                # with more than one microbatch slot in flight the value
                # standing there belongs to whichever slot wrote it last --
                # not to the slot this decision is about. The miss then
                # defaulted to 0 and was consumed as a MEASUREMENT: 2212
                # consecutive `told=98304 local=0` retractions for a rank
                # holding all 98304 tokens, voids alternating slot 2, 0, 2, 0
                # with no exception, each void restoring the other slot's
                # snapshot and so re-arming the next one.
                #
                # ONLY THE NAMED SLOT MAY ANSWER. Scanning the whole ring for
                # a rid would be the same defect with a wider blast radius --
                # answering a question about slot 2 with slot 0's progress.
                # `PPAdmissionDecision.mb_id` is documented as "one PP
                # microbatch slot" and has been on the wire since #791; this
                # is an unread field, not a missing one.
                if slot_chunked_rid is not None and entry.rid == slot_chunked_rid:
                    computed = pp_chunked_local_match(slot_chunked)
                    if computed is not None:
                        local_match_lens[entry.rid] = computed
                        continue
                # #944 THE FOURTH PLACE A REQUEST CAN LIVE, and the one that
                # wedged the instance. Once a request finishes its chunked
                # prefill it is in NEITHER the waiting queue NOR `chunked_req`
                # NOR the slot's chunked req -- it is in the RUNNING BATCH.
                # `self.running_batch` is on this object and is read elsewhere
                # in this file; this resolution chain never consulted it, so a
                # rank holding the whole prefix answered 0.
                #
                # Measured under real agent-shaped load (long SHARED prefixes,
                # which is what puts requests into the running batch while new
                # admissions for the same prefix keep arriving): 2106
                # unhonourable-prefix events, 2107 voided passes, and a
                # three-rank hang 35 s after health.
                running = getattr(self, "running_batch", None)
                running_reqs = getattr(running, "reqs", None) or ()
                for r in running_reqs:
                    if getattr(r, "rid", None) == entry.rid:
                        local_match_lens[entry.rid] = int(
                            getattr(r, "kv_committed_len", 0) or 0
                        )
                        break
                else:
                    # #944 A MISS IS NOT A ZERO. Writing 0 here is what made
                    # #797c and #798 look like two unrelated defects instead of
                    # one class seen twice: the consumer cannot tell "no prefix"
                    # from "not found", and it voids the pass on both. The
                    # sentinel makes the two separable at the reader; the reader
                    # decides what to do about it.
                    local_match_lens[entry.rid] = UNKNOWN_MATCH
                continue
            req.init_next_round_input(self.tree_cache)
            local_match_lens[entry.rid] = len(req.prefix_indices)
        return reconcile_pp_admission_decision(
            decision,
            local_match_lens,
            rank=self.ps.pp_rank,
            pp_size=pp_size,
            log=logger,
        )

    def _pp_forwarded_schedule_from(
        self: Scheduler, amended: Optional[PPAdmissionDecision]
    ) -> Dict[str, Tuple[int, int]]:
        """#791 CORE: the pass geometry this rank will EXECUTE, off `amended`.

        A one-line method and not a one-line expression inline in
        `_event_loop_pp_body`, for the same reason `_pp_pass_retraction_reason`
        and `pp_pass_should_void` are methods: it is the single point at which
        the forwarded chunk lengths enter this rank, so it is the single point
        a test can neuter to prove the rest of the machinery actually depends
        on them. Resolves `forwarded_schedule` through this module's globals,
        which is what makes that neuter a real one.
        """
        return forwarded_schedule(amended)

    def _pp_order_batch_by_schedule(
        self: Scheduler, reqs, schedule: Dict[str, Tuple[int, int]]
    ):
        """#791 CORE: put this rank's batch into the forwarded ORDER.

        A method for the same reason `_pp_forwarded_schedule_from` above is
        one: it is the single point at which the forwarded order is applied,
        so it is the single point a test can neuter -- and it resolves
        `order_batch_by_schedule` through this module's globals, which is what
        makes that neuter real. scheduler.py calls it through here rather than
        importing the function directly, so the test and production share one
        resolution path.
        """
        return order_batch_by_schedule(reqs, schedule)

    def _pp_void_retracted_pass(
        self: Scheduler,
        effective: Dict[str, int],
        amended: PPAdmissionDecision,
    ) -> Tuple[Dict[str, int], PPAdmissionDecision]:
        """#797: a retraction drops the PASS, not just the rid.

        THE NUMBERS THAT DECIDED THIS. Boots instr15/16/17 logged 661, 1651
        and 1718 `#791 unhonourable prefix` retractions and died on ONE width
        mismatch each. Every other narrowing computed: `reconcile_pp_
        admission_decision` dropped the unhonourable rid, this rank built a
        batch from what was left, and the upstream's hidden states -- for a
        batch containing that rid -- were paired with it. Chunked prefill caps
        every chunk at the same size, so two ranks running DIFFERENT request
        sets routinely present EQUAL widths; `model_runner.forward`'s
        `_hs.shape[0] != _want` sees nothing at all in that case. So the
        ~4000 survivals in that series are not survivals, they are silent
        wrong output, and every uptime number measured on them is void.

        WHY THE PASS AND NOT THE RID. There are only three membership
        outcomes and two of them are unavailable. The rank cannot ADMIT the
        rid -- it has no KV for the prefix its upstream reused, and the
        upstream sent hidden states only for the extend tokens, so there is
        nothing to reconstruct the missing prefix from. The upstream cannot be
        AMENDED -- it sent its decision and launched its batch earlier in this
        same pass, and a batch in flight cannot be recalled. What is left is
        to run the pass NOWHERE, which restores uniform membership in the one
        direction that is physically available.

        THE COST, NAMED: the upstream's forward for this pass is wasted, once
        per prefix first offered that a downstream cannot honour. It is not
        once per pass: the void rides back to PP0 inside #791b's void output,
        `PPAdmissionCongruenceGuard.record_return_trip` learns the observed
        local match as a floor, and `prefix_len_for` clamps the next offer for
        that rid to it -- a strictly decreasing, non-negative sequence, so the
        rid stops being re-offered an unhonourable prefix. The requests are
        not lost either: `_pp_absorb_void_output` releases and re-queues them
        on PP0, which is the same requeue-for-free path a capacity rejection
        already uses.

        WHAT IT REPLACES, STATED PLAINLY: #791c's boundary raise. That
        converted the mispair into a named refusal thirty layers earlier and
        killed the boot anyway; it stays in place as a tripwire, and a boot
        carrying this change should count zero of it.

        Returns `(effective, amended)` UNCHANGED when this rank retracted
        nothing of its own and nothing upstream voided -- the default path is
        not merely equivalent, it is the same objects.
        """
        rank = getattr(getattr(self, "ps", None), "pp_rank", None)
        voided = pp_pass_should_void(
            amended, rank, getattr(self, "_pp_pass_voided_incoming", False)
        )
        self._pp_admission_pass_voided = voided
        if not voided:
            return effective, amended
        mine = entries_retracted_by_rank(amended, rank) if rank is not None else ()
        if mine:
            self._pp_pass_voids = getattr(self, "_pp_pass_voids", 0) + 1
            first = mine[0]
            logger.warning(
                "#797 PP-ADMISSION pass voided on rank %s: this rank retracted "
                "%d request(s) (first: rid=%s told=%d local=%s), so its batch "
                "would have been a strict SUBSET of the one the upstream "
                "already launched. Running the whole pass nowhere instead: "
                "rank 0's requests are released and re-queued by the void "
                "output, and the observed local match is fed back as a prefix "
                "floor so the next offer for this rid is honourable.",
                rank,
                len(mine),
                first.rid,
                first.prefix_len,
                first.observed_local,
            )
        # Both halves of the void. The dict is what scheduler.py's admission
        # loop reads (a rid it does not name is simply not admitted, the
        # existing requeue-for-free path); the decision is what the next rank
        # and the return trip read.
        return {}, void_pp_admission_decision(amended)

    def _pp_drain_voided_proxy(self: Scheduler, mb_id: int) -> bool:
        """#797: take the proxy a voided pass will never pair, and drop it.

        True iff a message was taken. THE WIRE OWES EXACTLY ONE MESSAGE PER
        PASS and that debt does not disappear because this rank decided not to
        run: the upstream posted its proxy isend before anything downstream
        could tell it otherwise, and `_pp_commit_comm_work(self.send_proxy_
        work)` on its NEXT pass is a blocking wait on that message being
        taken. Leaving it is the bounded-recv corpse (see `_pp_send_dict_to_
        next_stage`), and it wedges the upstream, not this rank.

        GATED ON THE SENDER'S OWN STATEMENT, never on an inference from this
        rank's state. `_PP_UPSTREAM_LAUNCHED_KEY` is written per hop by the
        rank that will or will not send, because "did my upstream launch" is
        not derivable here: its batch can be empty for capacity reasons that
        never appear in the decision, and a blocking receive for a message
        nobody sent is the deadlock family this whole feature is a list of.

        Not the #631 guard's business: this is a deliberate discard of a
        message whose pairing has already been refused, so it reads no stamp
        and raises nothing. Under a gapped set there is no proxy on this
        channel at all (`_pp_recv_proxy_tensors`' own early return), so this
        is a no-op there for the same reason.
        """
        if not getattr(self, "_pp_admission_pass_voided", False):
            return False
        if not getattr(self, "_pp_upstream_launched_incoming", False):
            return False
        if self.ps.pp_size <= 1 or self.pp_group.is_first_rank:
            return False
        if getattr(self, "_pp_gapped_wire", False):
            return False
        self._pp_recv_typed_dict(
            expected_kind="proxy",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )
        self._pp_voided_proxy_drains = getattr(self, "_pp_voided_proxy_drains", 0) + 1
        logger.warning(
            "#797 PP-ADMISSION voided proxy drained on slot %d: the upstream "
            "had already launched when this pass was voided, so its hidden "
            "states are on the wire with no batch to pair them with. Taken "
            "and discarded -- an unmatched message here blocks the UPSTREAM "
            "in its next pass's proxy commit, not this rank.",
            mb_id,
        )
        return True

    def _pp_void_own_batch(self: Scheduler, mb_id: int) -> bool:
        """#797d: THIS rank's own voided pass must build nothing for `mb_id`,
        even when `get_next_batch_to_run` still handed back a batch.

        `_pp_void_retracted_pass` (above) empties the ADMISSION DICT
        (`effective`) and the forwarded DECISION (`amended`), but neither of
        those is what `get_next_batch_to_run` schedules from: it also runs
        its own local continuation logic -- `self.chunked_req` (the #797b
        stash at scheduler.py's `if self.chunked_req.extend_range.end >
        len(prefix_indices): self.stash_chunked_request(...)`, unconditional
        and reached BEFORE that function's own `_pp_admission_pass_voided`
        early return) and the resident `running_batch` -- so it can still
        return a non-empty `plan.batch_to_run` for a pass this rank has
        already decided to run nowhere.

        Downstream of that, uncleared: `_event_loop_pp_body`'s `if
        cur_batch:` branch is taken and blocks in `_pp_recv_proxy_tensors` on
        a proxy the voided UPSTREAM rank never sends (it took the
        `_pp_drain_voided_proxy` branch instead, having voided the same
        pass) -- the exact deadlock in SPECIMEN_wedge_19-02.txt: PP2 wedged
        in `_pp_recv_proxy_tensors` <- `_event_loop_pp_body:1503` with
        `effective={}` (the void executed) but `cur_batch` non-None (the
        batch did not). Clearing the slot here, before that branch is
        reached, also fixes the LAST rank's own output-send decision for
        free: `_pp_commit_send_output_work_and_preprocess_output_tensors`
        reads this same `self.mbs[...]` entry to choose between sending real
        output and the `#791b` void-output fallback, so no separate flag is
        needed for that side of the ring.

        Called immediately after `get_next_batch_to_run`, strictly before
        this pass's admission decision is sent (so a voided slot's
        `launched=self.mbs[mb_id] is not None` is never sent True) and
        strictly before `cur_batch = self.mbs[mb_id]` is read.

        Mirrors `_pp_absorb_void_output`'s restore idiom -- chunked_req
        restored to its pre-admission value and parked, never retracted;
        resident decode requests kept, not released -- but for THIS rank's
        own decision rather than an incoming wire message, so it does not
        touch `pp_absorb_admission_return` / `_pp_void_forward_payload`:
        those are PP0-only return-trip bookkeeping that has no counterpart
        here.

        `self.running_batch` / `self.running_mbs[mb_id]` are deliberately
        NOT touched. Both were already reassigned this pass from `plan.
        running_batch`, but that value is the carry-over of the PRIOR pass's
        already-finished `last_batch` (scheduler.py's `last_batch.forward_
        mode.is_extend()` merge, reached and completed before that
        function's void check), not a product of THIS pass's now-voided
        admission -- undoing it would discard already-validated resident
        state on the same reasoning `_pp_absorb_void_output` itself gives for
        never releasing a resident request: "the pass simply did not run,
        and it decodes again next pass from the state it still holds".

        True iff there was a batch to void. False (a no-op) when
        `get_next_batch_to_run` already honoured the void on its own --
        scheduler.py's own `_pp_admission_pass_voided` guard is meant to
        guarantee exactly that, so idempotence here is what makes this call
        safe defense-in-depth rather than a second, competing source of
        truth.
        """
        # #801-spin: CONSUMED FIRST, and #951 is why the position matters.
        #
        # The #798 site sets this for the single pass it is suppressing and
        # this method is documented as consuming it, "so it can never outlive
        # this pass". That held only while the #798 path always arrived here
        # with a batch to void. Since #951 refuses that pass before it builds
        # anything, this method now takes the empty-slot return below on
        # exactly those passes -- and a consume that sits after the return is
        # not a consume. The flag would then survive into the NEXT void, which
        # is an ordinary #797 retraction, and silence a record that nothing
        # asked to be silenced. Read and cleared before any early exit; the
        # local is what the log below reads.
        suppress = getattr(self, "_pp_idle_void_suppress_log", False)
        self._pp_idle_void_suppress_log = False

        batch = self.mbs[mb_id] if mb_id < len(self.mbs) else None
        if batch is None:
            return False
        self.mbs[mb_id] = None
        if mb_id < len(self.mb_metadata):
            self.mb_metadata[mb_id] = None

        carried_slots = getattr(self, "_pp_chunked_req_before_by_slot", None)
        chunked_before = (
            carried_slots[mb_id]
            if carried_slots and 0 <= mb_id < len(carried_slots)
            else None
        )
        if getattr(self, "chunked_req", None) is not chunked_before:
            self.chunked_req = chunked_before
        parked = _park_chunked_prefill_chunk(self, chunked_before)

        # THE INSTR19 STATE, DEFENDED AGAINST RATHER THAN PROPAGATED.
        # `_park_chunked_prefill_chunk` treats `extend_range is None` on
        # its way in as "already parked, already reset, or never
        # prepared -- nothing to give back" and leaves it untouched (see
        # its own docstring) -- which is the right call for the KV-release
        # question it answers, but wrong for this call site's OWN
        # obligation: whatever it leaves in `self.chunked_req` is what the
        # NEXT pass's `get_next_batch_to_run` dereferences unconditionally
        # (`self.chunked_req.extend_range.end`, scheduler.py) the instant
        # `self.chunked_req is not None`, with no guard of its own. Under
        # this rank's own restore above, that state should be unreachable:
        # `chunked_before` is a snapshot taken at the top of THIS pass, and
        # if its `extend_range` had already been None going into THIS
        # pass, `get_next_batch_to_run`'s identical top-of-function read
        # would already have raised earlier in this very pass, before this
        # gate ever ran -- so nothing upstream of here is known to produce
        # it. The check costs one attribute read and turns "should not
        # happen" into an explicit, coherent "no carried chunk" rather than
        # a silent bet that the argument keeps holding across every future
        # edit to the functions on this path.
        if (
            self.chunked_req is not None
            and getattr(self.chunked_req, "extend_range", None) is None
        ):
            logger.warning(
                "#797d own-void: the chunked request for slot %d already had "
                "extend_range=None (the reset_for_retract shape "
                "_park_chunked_prefill_chunk cannot repair) -- clearing "
                "self.chunked_req instead of carrying a request the next "
                "pass's get_next_batch_to_run cannot read.",
                mb_id,
            )
            self.chunked_req = None

        running_mbs = getattr(self, "running_mbs", None) or ()
        running = running_mbs[mb_id] if 0 <= mb_id < len(running_mbs) else None
        resident = {r.rid for r in (getattr(running, "reqs", None) or ())}
        reqs = list(getattr(batch, "reqs", None) or ())
        released = 0
        for req in reqs:
            if pp_void_keeps_request(req, resident, chunked_before):
                continue
            # #969: THE RETRACTION PATH, NOT THE PROBE PATH. These are
            # admitted requests -- they were matched against the prefix tree
            # at admission and hold a lock ref on `req.last_node`. The probe
            # helper that used to stand here frees rows without ever reaching
            # `cache_finished_req`, so it left that ref behind on pages it had
            # already returned to the allocator, and `reset_for_retract`
            # cleared the only handle to it one line later. `release_req`
            # resets the request as its last act, so this site does not.
            _release_voided_request(self, req, len(reqs) - released)
            self.waiting_queue.append(req)
            released += 1

        # #801-spin: CONSUMED, never merely read -- read and cleared at the top
        # of this method (see there for why the position is load-bearing), so
        # every other caller -- the #797 retraction path above all -- logs
        # exactly as it always did, and a stale True can never survive into a
        # pass that did not ask for it.
        if not suppress:
            logger.warning(
                "#797d PP-ADMISSION own pass voided on slot %d: get_next_batch_to_run "
                "still returned a batch for a pass this rank had already voided, so "
                "it is being cleared here instead of launched. %d of %d request(s) "
                "released and re-queued (the rest are resident in the running batch, "
                "or are the chunked request, and keep their pages -- #797/#797b; "
                "chunk parked=%s).",
                mb_id,
                released,
                len(reqs),
                parked,
            )
        return True

    def _pp_void_pass_without_upstream_launch(self: Scheduler, mb_id: int) -> bool:
        """#798: this rank may not run a pass its UPSTREAM did not launch.

        True iff this pass was voided here. THE DEFECT THIS CLOSES, stated as
        the wire contract it breaks: whether a proxy message exists for a
        given pass is decided independently on the two ends of the wire.

            sender    (:1567-1580)   sends iff ``self.mbs[mb_id]``
            receiver  (:1518-1529)   receives iff ``self.mbs[mb_id]``, else
                                     drains iff voided AND
                                     ``_pp_upstream_launched_incoming``

        Three of those four combinations are reconciled. ``_pp_drain_voided_
        proxy`` was given an explicit gate on the sender's own statement, and
        its docstring states the rule: "GATED ON THE SENDER'S OWN STATEMENT,
        never on an inference from this rank's state ... a blocking receive
        for a message nobody sent is the deadlock family this whole feature is
        a list of." The RECEIVE branch never got that gate --
        ``_PP_UPSTREAM_LAUNCHED_KEY`` was read in exactly one place in this
        module, inside the drain -- so the fourth combination, THIS RANK HAS A
        BATCH AND ITS UPSTREAM DID NOT LAUNCH, entered a blocking receive for
        a message that was never posted, took the NEXT pass's proxy instead,
        and left every later receive one message ahead for ever. The first
        stamp compared after that reads exactly one slot ahead in the same
        flip epoch.

        BOTH #798 SPECIMENS ARE THAT READING, and neither is a slot-ring
        divergence: specimen 1 (boot_seed796, 22:44:52Z) refused a proxy
        stamped mb_id=2 while on mb_id=1; specimen 2 (boot_seed797b,
        22:52:20Z) one stamped mb_id=1 while on mb_id=0. In both the receiver
        is BEHIND in slot index and AHEAD in messages consumed, which is the
        signature of a surplus receive, not of a rank that spun ahead. A
        voided pass keeps every per-hop rendezvous that paces the loop -- the
        chain receive (:1263), the decision receive (:1299), the decision-send
        reap (:1607) and the chain flush (:1609) all run on a voided pass --
        so a void cannot desynchronise the ring on its own, and
        test_pp_void_slot_advance_798.py's genuine-void case proves it does
        not.

        WHY THE COMBINATION ARISES AFTER A RETRACTION, which is what puts this
        defect one to two passes downstream of every #797 void rather than
        anywhere else. The void is deliberately asymmetric in what it
        preserves: ``_pp_void_own_batch`` does NOT touch ``running_mbs``,
        because "the pass simply did not run, and it decodes again next pass
        from the state it still holds", while PP0's #791b void output RELEASES
        and re-queues its own requests (both specimens log "1 of rank 0's 2
        request(s) have been released and re-queued"). So right after a void
        the upstream can have nothing to launch for a slot on which this rank
        still holds resident work.

        VOIDING, NOT WAITING, AND NOT COMPUTING. A non-first rank under a
        non-gapped set has no entry activations of its own -- the hidden
        states it needs are exactly the ones the upstream did not produce -- so
        the pass is unrunnable here in the same physical sense #797 describes.
        Running it nowhere is the one direction available, and it restores
        uniform membership rather than repairing a consequence.

        RANK-AGREED BY CONSTRUCTION, which is the property that makes this
        safe where a rank-local hold would not be. The decision is taken from
        ``_pp_upstream_launched_incoming`` -- a per-hop fact written by the
        rank that will or will not send -- and it is taken HERE, immediately
        after ``_pp_void_own_batch`` and strictly BEFORE this pass's admission
        decision is forwarded (:1371-1516). So the ``launched=self.mbs[mb_id]
        is not None`` this rank forwards is already the voided value, and the
        next rank down learns the pass is off by the same mechanism a #797
        void travels on, one hop at a time, with no collective and no new
        synchronisation point. Ordering this after the send instead would
        forward ``launched=True`` while sending no proxy, which is the same
        defect one hop further down.

        THE SLOT CURSOR IS DELIBERATELY NOT TOUCHED. An earlier reading of
        #798 proposed holding ``mb_id`` for a voided pass, mirroring #631
        defect Q's arming hold at :1625. That would have been wrong twice
        over: the ring does not diverge here (measured above), and the hold's
        ``continue`` skips the #753 gapped-lockstep barrier at :1656, so a
        hold taken on a condition that is not rank-uniform converts a
        harmless state into a barrier hang. This fix adds no ``continue`` and
        changes no index.

        NO-OP ON EVERY PATH THAT CANNOT HAVE THE PROBLEM: the first rank (no
        upstream), ``pp_size <= 1``, a gapped set (there is no stage-boundary
        proxy at all -- ``_pp_recv_proxy_tensors`` returns None immediately
        there, so the receive this prevents is not made in the first place),
        an upstream that did launch, and a slot that is already empty.
        """
        if not pp_upstream_void_pending(self):
            # #801-spin: THE RESET, and it is the whole definition of progress
            # here. The upstream launched, so this pass can run and this rank's
            # resident state can change. Every other shape the predicate
            # excludes is one in which the spin cannot occur at all (no
            # upstream, no wire), and the empty-slot return below is the shape
            # in which this rank has nothing to re-derive -- all of them clear
            # the streak, so a streak counts consecutive passes of exactly one
            # state: THIS RANK HAD WORK AND ITS UPSTREAM HAD NOTHING TO GIVE
            # IT.
            self._pp_upstream_idle_void_streak = 0
            return False
        batch = self.mbs[mb_id] if mb_id < len(self.mbs) else None
        if batch is None and not getattr(
            self, "_pp_upstream_void_withheld_work", False
        ):
            self._pp_upstream_idle_void_streak = 0
            return False
        # #951: AND THE SLOT IS EMPTY BECAUSE THE GUARD EMPTIED IT. Once the
        # void is decided before the plan, `self.mbs[mb_id]` is None on exactly
        # the passes this method used to find non-empty, so reading emptiness
        # alone would retire the #801-spin detector at the moment it started
        # mattering -- a livelock that no longer counts is a livelock that no
        # longer raises. scheduler.py's guard therefore records what it
        # withheld: True only when this rank HELD WORK (a non-empty waiting
        # queue, or a running batch) and was refused anyway, which is the same
        # "this rank had work and its upstream had nothing to give it" the
        # streak has always counted. It is a pre-plan superset of "derived a
        # batch" -- work the pass could not have fitted also counts now -- and
        # that is the safe direction here: it can only make a genuine
        # no-progress streak visible sooner, never hide one, and an idle rank
        # (empty queue, empty running batch) still clears the streak exactly as
        # before.

        self._pp_upstream_idle_voids = getattr(self, "_pp_upstream_idle_voids", 0) + 1
        streak = getattr(self, "_pp_upstream_idle_void_streak", 0) + 1
        self._pp_upstream_idle_void_streak = streak

        # #801-spin: the twin record this void is about to produce
        # (`_pp_void_own_batch`'s `#797d`) is suppressed on the same passes
        # this one is, so the pair stays a pair and the specimen's 4706 lines
        # become 24. Consumed by that method, so it can never outlive this pass.
        report = pp_idle_void_should_report(streak)
        self._pp_idle_void_suppress_log = not report
        if report:
            logger.warning(
                "#798 PP-ADMISSION pass voided on slot %d: the upstream reported "
                "launched=False for this pass, so no hidden states were posted for "
                "it, but this rank still derived a batch from its own resident "
                "state. Receiving here would take the NEXT pass's proxy and leave "
                "every later receive one message ahead (the 2026-08-21 specimens). "
                "Running the pass nowhere instead, and forwarding the void so the "
                "ranks below take the same decision. This is consecutive "
                "no-progress void %d on this rank (reported on powers of two; a "
                "streak of 1 is the ordinary post-retraction case #798 was "
                "written for, a growing one is the #801-spin livelock).",
                mb_id,
                streak,
            )

        bound = envs.SGLANG_PP_IDLE_VOID_STREAK_BOUND.get()
        if pp_idle_void_streak_exceeded(streak, bound):
            # #801-spin: A VOID THAT NEITHER BLOCKS NOR PROGRESSES IS STILL A
            # WEDGE, and until here it was the only one this file could not
            # see. #801 and #802 closed the two ways the ring could BLOCK; this
            # closes the way it could SPIN. Specimen
            # boot_802f_staged1_0822_1716: PP1 alone took 2353 of these passes
            # in five seconds -- ~470 per second, no forward, no output, and
            # not one line saying so -- while PP0 and PP2 logged nothing at all.
            # Refusing by name converts an instance that is silently not
            # serving into one that says why it is not, which is the only
            # difference this rank is in a position to make: the reason its
            # upstream never launches is upstream state (see the ticket), and
            # a rank cannot repair a peer's admission from here.
            self._pp_upstream_idle_void_streak = 0
            raise RuntimeError(
                f"#801-spin PP IDLE-VOID LIVELOCK REFUSED: this rank (pp_rank="
                f"{self.ps.pp_rank}) has voided {streak} CONSECUTIVE passes "
                f"because its upstream reported launched=False for every one of "
                f"them, while still deriving a batch of its own each time "
                f"(currently slot {mb_id}). A voided pass runs no forward, so "
                f"this is {streak} passes in which this rank served nothing and "
                f"its state did not change -- it will re-derive the same batch "
                f"from the same resident work next pass and void it again. The "
                f"defect is NOT on this rank: the void handling did exactly what "
                f"#798 requires each time. It is whatever leaves the upstream "
                f"unable to launch while this rank holds resident work (in the "
                f"1716 specimen, a chunked request parked here across every one "
                f"of the 2353 passes). Raise or disable the bound with "
                f"SGLANG_PP_IDLE_VOID_STREAK_BOUND (0 disables) if this streak "
                f"is legitimate on your workload."
            )

        # Both halves of the void, exactly as `_pp_void_retracted_pass`
        # arranges them, so that what this rank FORWARDS names the same empty
        # pass its own batch now is. Set before `_pp_void_own_batch` below so
        # the clearing runs under the same flag the #797 path sets.
        self._pp_admission_pass_voided = True
        self._pp_admission_incoming_effective = {}
        amended = getattr(self, "_pp_admission_amended_to_forward", None)
        if amended is not None:
            amended = void_pp_admission_decision(amended)
            self._pp_admission_amended_to_forward = amended
            self._pp_admission_incoming_schedule = self._pp_forwarded_schedule_from(
                amended
            )

        # The clearing itself is `_pp_void_own_batch`'s, unchanged: chunked
        # request restored to its pre-admission value and parked, resident
        # decode requests kept, everything else released and re-queued. This
        # void has exactly the same obligation, so it uses exactly the same
        # code rather than a second implementation of it.
        self._pp_void_own_batch(mb_id)
        return True

    def _pp_flip_epoch(self: Scheduler) -> Optional[int]:
        """#795: the generation number of the microbatch slot ring.

        ``PhaseFlipRuntime._epoch`` (phase_flip_runtime.py:7398) increments
        exactly once per COMPLETED cutover, and the cutover is precisely what
        rebuilds the ring ``mb_id`` indexes (``init_pp_loop_state``, called at
        phase_flip_runtime.py:1580). So this is not a clock bolted onto the
        wire -- it is the version of the namespace the proxy stamp names, read
        from the one authority for it rather than mirrored into a second copy
        (the same reason ``phase_flip_is_armed`` reads ``_pending`` directly).

        Every rank increments it independently at its own cutover completion,
        and the runtime's own consensus reduction raises DESYNC if two ranks
        ever disagree (phase_flip_runtime.py:3438-3454), so a sender's epoch
        and a receiver's epoch are comparable quantities by construction.

        None -- not 0 -- when there is no runtime: a boot without the phase
        flip has no epochs to tell apart, and None is what every consumer
        below reads as "fall back to the slot-only comparison", i.e. exactly
        the behaviour that shipped before this field existed.
        """
        runtime = getattr(self, "phase_flip_runtime", None)
        if runtime is None:
            return None
        try:
            return int(runtime.epoch)
        except Exception:  # noqa: BLE001 - an unreadable epoch names no epoch
            return None

    def _pp_proxy_stamp(self: Scheduler, mb_id: int, result) -> tuple:
        """#631 VARIANT B / #795: the identity a proxy message carries.

        (mb_id, monotone seqno, row count, flip epoch).

        The seqno is per rank and never resets, so it distinguishes two
        messages for the SAME slot -- which is exactly the pair a stranded
        leftover creates.

        #795 APPENDED THE EPOCH, and that is the element the consumers
        actually compare. ``mb_id`` alone is not an identity: it indexes a
        ring of ``pp_loop_size`` slots that a cutover rebuilds from zero, so
        a leftover stranded across a flip matches a slot-only test one time in
        ``pp_loop_size``. Measured, 2026-08-21 06:10:48, boot instr15 -- the
        guard passed a 119-row prefill proxy into a 27-token decode batch and
        only the width check downstream caught it. See
        ``pp_proxy_stamp_names_pass`` for the full argument.

        -1 when this rank has no phase-flip runtime to ask. Readers map that
        to "no epoch named" and fall back to the slot-only comparison, so a
        boot without the flip keeps its exact previous behaviour.
        """
        self._pp_proxy_seq = getattr(self, "_pp_proxy_seq", 0) + 1
        rows = -1
        try:
            hs = result.pp_hidden_states_proxy_tensors.tensors.get("hidden_states")
            if hs is not None:
                rows = int(hs.shape[0])
        except Exception:  # noqa: BLE001 - a stamp may never break a send
            pass
        epoch = pp_flip_epoch_of(self)
        return (
            int(mb_id),
            int(self._pp_proxy_seq),
            rows,
            -1 if epoch is None else int(epoch),
        )

    def _pp_wait_for_dict_readiness(
        self: Scheduler, mb_id: int, kind: str = "proxy"
    ) -> None:
        """#789: refuse to enter a blocking CHAN_DICT receive until there is
        POSITIVE evidence a message is actually coming.

        #802-ring: PARAMETERISED BY WIRE KIND, AND THAT IS THE WHOLE OF
        STAGE A. This gate was written for the proxy receive and shipped
        guarding only that one, while `_pp_recv_dict_from_prev_stage` -- the
        OUTPUT receive -- entered its blocking call with nothing in front of
        it. `_pp_recv_proxy_tensors`' own docstring names the shape that
        costs: CORPSE R ends with "PP0 `_pp_recv_dict_from_prev_stage` -- the
        OUTPUT wire" as one arc of a closed cycle, so the file already knew
        this wire could be the blocking one and guarded the other.

        MEASURED, specimen /spinning/evidence-665-f1/wedge_802f_1712/ (PP=3,
        --enable-phase-flip, py-spy of all three ranks):

            PP0  _pp_commit_comm_work <- _pp_commit_pending_req_work
                 (scheduler_pp_mixin.py:4071/2262) -- flushing the request
                 chain PP1 can only take at the top of its NEXT pass
            PP1  _pp_recv_dict_from_prev_stage <- _do_recv
                 (scheduler_pp_mixin.py:5608/6069) -- blocked HERE, in the
                 unguarded output receive, so it never reaches that top
            PP2  PpChainReceiver.recv <- recv_requests
                 (pp_chain_receiver.py:329) -- waiting on PP1's chain send

        A closed three-arc ring, and the arc that can be cut without a new
        protocol is PP1's: it is the only one waiting on a message that was
        never posted at all, rather than on a peer's scheduling.

        WHY THE OUTPUT WIRE CAN OWE NOTHING WHILE A RECEIVER EXPECTS ONE.
        The two ends of the intermediate hop apply unrelated predicates.
        `_do_recv` decides to receive from THIS rank's own slot state
        (`mbs[next_mb_id]` non-None, not prebuilt, not
        `_pp_can_skip_output_comm`); the non-last sender at
        `_pp_send_output_to_next_stage` decides to forward on `if
        pp_outputs:`, which is whatever it received LAST iteration. The
        last-rank hop is matched by construction because it consults
        `_pp_output_expected_for_slot`, but that flag answers "did the FIRST
        rank say it will receive an output for this slot" -- PP0's verdict,
        published for PP0's ring arc. No rank publishes the same thing for
        the intermediate hop, so an intermediate receiver's expectation is
        an independent variable and the two ends can disagree. Closing that
        disagreement is a protocol extension (a per-slot expectation every
        non-first rank publishes to its predecessor, two hops around this
        ring) and is deliberately NOT what this function does. This makes
        the resulting wait BOUNDED and LOUD instead of permanent; it does
        not make the missing send appear.

        THE FALSE-POSITIVE DIRECTION IS THE SAFE ONE, which is what makes
        widening the gate to a second wire defensible at all. Every exit
        that is not the timeout is an early `return` into the caller's
        ordinary blocking receive -- byte for byte the behaviour that
        shipped before this gate existed. The function can only CHANGE an
        outcome by raising, and it raises only when the counters never moved
        for the entire budget, which is precisely the state in which the
        receive would otherwise have blocked for ever.

        THE COUNTER IS NOT PROXY-SPECIFIC, only its old name was.
        `_pp_send_dict_to_next_stage` bumps CHAN_DICT for EVERY dict it
        posts, proxy and output alike, and `_pp_recv_typed_dict` bumps
        `local_consumed` off the wire before the demultiplex -- deliberately
        one counter for one wire, as its own comment says: "ONE counter for
        this wire, shared by 'proxy' and 'output' -- they are demultiplexed
        by __msg_type__ AFTER coming off it, so counting them apart would
        let a rank call a wire empty while a message of the other kind was
        still on it." So this gate reads the same true statement about the
        same wire whichever kind its caller is about to ask for. `kind` is
        used for ONE thing only: the non-destructive inbox peek, which is
        per-`(src, kind)` and must match the key the caller's own receive
        will use a moment later.

        THAT SHARED COUNTER IS ALSO THIS GATE'S PRECISION LIMIT, and saying
        so is not a caveat but the reason the budget exists. A message of
        the OTHER kind in flight reads as positive evidence here, so the
        caller may proceed into a receive whose own message is still not
        posted. That is not a regression -- it is exactly the unguarded
        behaviour, and the caller's `_pp_recv_typed_dict` stashes a
        wrong-kind message and receives again, which is the demultiplexer
        working as designed. The gate's guarantee is therefore one-sided and
        precisely stated: a rank never waits FOR EVER on a wire its upstream
        has posted nothing to, and never refuses a wire its upstream has.

        THE DEFECT THIS CLOSES. Measured py-spy specimen (evidence-665-f1,
        2026-08-20, PP=3 with --enable-phase-flip): PP0 and PP1 both idle
        (cur_batch=None, server_is_idle=True), parked in
        ``_pp_commit_pending_req_work``; PP2 (cur_batch not None) blocked in
        the plain gloo receive inside ``_pp_recv_typed_dict``, forever -- an
        unbounded wait for a proxy that no upstream ever scheduled. WHY the
        two upstreams never scheduled a batch for this slot is a
        request-admission question, out of scope here (a separate
        investigation thread owns it); this function's only job is to make
        the wait BOUNDED and LOUD instead of silent, without changing
        anything about a healthy pass.

        THE DEFECT THIS CLOSES. Measured py-spy specimen (evidence-665-f1,
        2026-08-20, PP=3 with --enable-phase-flip): PP0 and PP1 both idle
        (cur_batch=None, server_is_idle=True), parked in
        ``_pp_commit_pending_req_work``; PP2 (cur_batch not None) blocked in
        the plain gloo receive inside ``_pp_recv_typed_dict``, forever -- an
        unbounded wait for a proxy that no upstream ever scheduled. WHY the
        two upstreams never scheduled a batch for this slot is a
        request-admission question, out of scope here (a separate
        investigation thread owns it); this function's only job is to make
        the wait BOUNDED and LOUD instead of silent, without changing
        anything about a healthy pass.

        NOT A TIMING-OUT WAIT ON THE TRANSPORT. A timed-out gloo
        ``Work.wait()`` on this build DESTROYS the pair (measured: the peer
        then sees "Connection closed by peer") -- corpse F, see the module
        docstring of ``phase_flip_counters`` and ``pp_flip_drain_leftover_
        dicts`` above. This function never touches the tensor-dict wire at
        all; it only polls the SAME pollable, out-of-band ``/dev/shm`` side
        channel ``pp_flip_drain_leftover_dicts`` already uses for exactly
        this kind of question -- ``PhaseFlipCounters``' CHAN_DICT
        sent/consumed counters, which are populated on EVERY ordinary
        proxy/output send (``_pp_send_dict_to_next_stage``), not only while
        a flip is armed.

        THE POSITIVE SIGNAL, NOT A GUESS FROM SILENCE. ``counters.sent(
        CHAN_DICT, upstream)`` is published by the upstream STRICTLY AFTER
        its isend is posted (the ordering law in ``phase_flip_counters``'s
        module docstring); once it exceeds this rank's own ``counters.
        local_consumed(CHAN_DICT)``, a message is PROVABLY already posted,
        and this function returns immediately so the caller's ordinary
        blocking receive -- now bounded by transfer time alone, not by peer
        scheduling -- proceeds exactly as it always has. That
        immediate-return-on-presence is what keeps a healthy boot
        unchanged: on any pass where the upstream has already posted (the
        common case, since this rank only reaches here after deciding it
        has a batch to run), the very first poll already sees the counter
        ahead and this function costs one counter read.

        THE BOUNDED BACKSTOP IS NOT THE DECISION (#630 lesson). If the
        counter shows nothing new, this polls again -- up to
        ``DEFAULT_PROXY_READINESS_BUDGET_S``, at
        ``PROXY_READINESS_POLL_STEP_S`` intervals -- because the upstream
        may simply still be inside a legitimately slow forward pass and has
        not posted its isend yet: the counter cannot lie, but it also
        cannot show a message that has not been sent. Only once that
        budget is entirely exhausted with the counter having NEVER moved
        does this raise. This is deliberately different from "wait N
        seconds then give up regardless": the loop acts the INSTANT the
        counter proves a message is in flight, on every single poll, and
        the deadline exists purely to bound how long it is willing to wait
        for that proof to arrive -- never as the trigger for the decision
        itself.

        ONLY GATES WHEN THE SIGNAL EXISTS. If ``pp_flip_counters`` is
        ``None`` (the ordinary non-phase-flip boot, including the
        reference regression launch command), there is no side channel to
        poll and this function is a no-op -- the caller's receive is
        unchanged from before this function existed. This is deliberate
        scope, not laziness: without ``--enable-phase-flip`` there is no
        per-rank publish/consume identity this function could read without
        new production wiring, and the defect this closes is only measured
        on boots that DO have ``--enable-phase-flip`` set (see
        WEDGE_788_specimen.note).

        AN ALREADY-STASHED MESSAGE IS ALSO A POSITIVE SIGNAL (#753
        interaction, found by test_pp_drain_completeness_787.py's own
        cutover case going red under this gate). ``pp_flip_drain_leftover_
        dicts`` can take a "proxy" message fully off the wire and, if it is
        the one this rank actually owes, ``stash_typed`` it in
        ``pp_typed_channel``'s per-``(src, kind)`` inbox for its real
        consumer -- this function's caller -- to pick up next, with no
        further wire activity. ``_pp_flip_bump_consumed`` already fired
        when that message left the wire (#631 G: counted off the wire,
        before the demultiplex), so by the time this gate runs,
        ``local_consumed`` has ALREADY caught up to ``sent`` for that
        message -- the counter-only check above would misread that as
        "nothing new" and wait out the whole budget for a message that is
        already sitting here, ready. So: check the inbox FIRST, exactly
        the same ``(src, kind)`` key ``recv_typed_tensor_dict`` itself will
        check a moment later (non-destructively -- this only peeks; the
        real, consuming read stays in ``_pp_recv_typed_dict``, unchanged).
        A non-empty queue is checked before, and takes priority over, the
        counter poll below.

        TRANSPORT COVERAGE, AND ITS LIMIT. This gate sits BEFORE
        ``_pp_recv_typed_dict`` is called at all -- before the gloo
        metadata phase AND before whatever device-payload phase would
        follow it for a matched message. A rank that never observes
        "upstream posted" never enters ``_pp_recv_typed_dict``, so it can
        never reach either transport's blocking call on THIS wire, for
        THIS slot. And because ``bump_sent`` fires only after the
        upstream's ``send_tensor_dict`` call returns, a sender that is
        itself stuck mid-send (metadata not yet posted, or a device-level
        payload send not yet posted) also never advances its counter --
        this rank's poll sees the same "nothing new" either way.

        THAT LAST SENTENCE USED TO END "so a stuck SENDER is covered
        identically to a sender that never intended to send at all", AND
        THAT WAS THE #789 DEFECT, not a description of it. Treating the
        two identically is safe only when the sender's progress does not
        depend on this rank. When the send is a RENDEZVOUS -- the lazy
        creation of a torch NCCL 2-rank p2p communicator, which every boot
        performs on its first real prefill -- the sender is stuck
        PRECISELY BECAUSE this rank has not entered the receive, and this
        gate refusing to enter it closes the cycle rather than avoiding
        one. Boots instr7 and instr8 (2026-08-21) both died there, with
        PP0 in `isend` inside `_pp_send_dict_to_next_stage` while PP1 and
        PP2 polled here and then raised over "posted 1681, consumed 1681".
        The two cases are now distinguished by a second counter,
        `attempted`, published before the send rather than after it: a
        sender inside a send has entered it, a sender that scheduled
        nothing has not. See PhaseFlipCounters.bump_attempted.

        What this gate does NOT cover: a send
        whose ``isend`` call has already returned (so ``bump_sent`` already
        fired and this gate already let the receive proceed) but whose
        underlying transfer then stalls mid-flight on the wire -- e.g. a
        device kernel that has started but not completed. That is a
        different failure shape (a message provably WAS posted, not "no
        message was ever posted"), needs a receipt/completion signal this
        rank does not have today, and is out of scope for this first
        slice, which targets the specific, measured, currently-reproducible
        defect: "the last rank enters an unbounded blocking wait for a
        proxy that will never be sent" (never posted at all, not posted
        and stalled).
        """
        counters = getattr(self, "pp_flip_counters", None)
        if counters is None:
            return
        upstream = self._pp_flip_upstream()
        src = resolve_src(self.pp_group, None)
        deadline = None
        while True:
            if typed_inbox(self.pp_group).get((src, kind)):
                # Already fully off the wire and stashed for this exact
                # consumer -- the ordinary receive below will find it with
                # no further wire activity. Return immediately rather than
                # trust the counters, which -- for a stashed message --
                # already read "caught up" (see the docstring section
                # above) and would otherwise wait out the whole budget for
                # something already delivered.
                return
            posted = counters.sent(CHAN_DICT, upstream)
            consumed = counters.local_consumed(CHAN_DICT)
            if consumed < posted:
                # Positive presence signal: the upstream provably posted a
                # message this rank has not yet taken off the wire. Return
                # immediately -- the caller's ordinary receive is next, and
                # from here it is bounded by transfer time, not by whether
                # the upstream ever schedules anything.
                return
            attempted = counters.attempted(CHAN_DICT, upstream)
            if consumed < attempted:
                # #789: the upstream is INSIDE a send for this rank right
                # now. That is just as positive a signal as "posted", and
                # it is the ONLY one available while the send in question
                # is a rendezvous -- the first p2p op on a torch NCCL group
                # creates its communicator lazily, and that isend cannot
                # return, so `posted` above cannot bump, until THIS rank
                # enters the receive. Gating on `posted` alone therefore
                # made this gate one arc of a three-arc cycle and killed
                # boots instr7 and instr8 on their first real prefill.
                #
                # Waiting here is bounded for the same reason the `posted`
                # branch is: the decision to send is already taken and
                # unconditional (the counter is bumped on the line before
                # the send call), so the only remaining wait is transfer
                # and rendezvous time. See PhaseFlipCounters.bump_attempted.
                return
            now = time.monotonic()
            if deadline is None:
                deadline = now + _pp_proxy_readiness_budget_s()
            if now < deadline:
                time.sleep(PROXY_READINESS_POLL_STEP_S)
                continue
            budget = _pp_proxy_readiness_budget_s()
            label = str(kind).upper()
            logger.error(
                "%s #789 %s READINESS TIMEOUT: mb_id=%s -- this rank's "
                "upstream (rank %s) has posted %d dict message(s) on CHAN_DICT "
                "(entered %d) and this rank has consumed %d; no new message "
                "appeared within %.1fs. No upstream scheduled work for this "
                "slot -- refusing to enter the blocking %s receive rather "
                "than wedge.",
                "PHASE-FLIP",
                label,
                mb_id,
                upstream,
                posted,
                attempted,
                consumed,
                budget,
                kind,
            )
            # #802-ring: the OUTPUT spelling names the asymmetry that
            # produces it, because the counters alone read identically for
            # both wires and the next reader would otherwise re-derive it.
            hint = (
                (
                    " This is the intermediate output hop: this rank decided "
                    "to receive from its OWN slot state while its upstream "
                    "decided to forward on `if pp_outputs:`, and nothing "
                    "publishes this rank's per-slot expectation to that "
                    "upstream (`_pp_output_expected_for_slot` is the FIRST "
                    "rank's verdict, not this one's). The missing send is "
                    "upstream of this line; see #802-ring."
                )
                if kind == "output"
                else ""
            )
            raise RuntimeError(
                f"#789 {label} READINESS TIMEOUT: mb_id={mb_id}: this rank's "
                f"upstream (rank {upstream}) posted {posted} dict message(s) "
                f"on CHAN_DICT (entered {attempted}), this rank has consumed "
                f"{consumed}, and no new "
                f"message appeared within {budget:.1f}s. No upstream scheduled "
                f"work for this slot; refusing to enter an unbounded blocking "
                f"receive. See #789.{hint}"
            )

    #: #802-ring: AN ALIAS, NOT A WRAPPER, and the difference is the whole
    #: reason this line exists. The gate was born proxy-only and roughly ten
    #: stand-in holders across the #631/#757/#787/#789/#791/#795/#797/#798
    #: test family bind it BY THIS NAME, one method at a time, via
    #: `setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))`.
    #: A wrapper that delegated to `_pp_wait_for_dict_readiness` would resolve
    #: that second name on the HOLDER, which binds only what its own tuple
    #: lists -- so every one of those holders would raise AttributeError the
    #: moment it entered a receive. Measured, not predicted: the wrapper form
    #: turned 9 green tests in test_pp_drain_completeness_787.py and
    #: test_pp_proxy_readiness_rendezvous_789.py into 5 failures.
    #: Binding the SAME function object under both names keeps every existing
    #: holder working untouched, while the honest name is the one new code
    #: reads. `kind` defaults to "proxy", so the old call signature
    #: `self._pp_wait_for_proxy_readiness(mb_id)` is unchanged in meaning.
    _pp_wait_for_proxy_readiness = _pp_wait_for_dict_readiness

    def _pp_recv_proxy_tensors(
        self: Scheduler, mb_id: int = -1
    ) -> Optional[PPProxyTensors]:
        """Receive this slot's hidden states, and PROVE they are this slot's.

        #631 VARIANT B, second cut. The stamp is the DETECTION half of the
        mispairing fix; the prevention half is ``pp_flip_drain_tensor_dicts``,
        which stops a message stranding in the first place.

        EXACTLY ONE MESSAGE PER PASS. That is the invariant this function
        may not break, and breaking it is corpse R:

        CORPSE R -- THE DROP-AND-RETRY DISPOSAL, metal-falsified
        2026-08-09 07:19:29Z, specimen
        /spinning/evidence-631/stamp_drop_wedge_20260809T0719Z.
        The first cut dropped a non-matching message and looped to take
        "the next one", bounded at 8. The DETECTION was exactly right --
        "stamp mb_id=2 seq=2811 rows=1 arrived while this rank is on
        mb_id=1", a one-row decode hidden state, the mispair specimen's own
        signature, refused instead of computed on. Six seconds later the
        instance wedged:

            PP1  here, in the SECOND recv the loop makes after a drop
            PP2  here, same
            PP0  _pp_recv_dict_from_prev_stage -- the OUTPUT wire

        a closed cycle: PP0 waits on PP2's output, PP2 on PP1's proxy, PP1
        on a proxy from PP0 that was never sent and never will be. The
        argument "dropping is safe because the sender is not waiting on us"
        was true and beside the point: the sender is not what blocks, the
        RECEIVER is. The wire owes one message per pass, so a second
        blocking call is made against a debt of one.

        THIS IS THE BOUNDED-RECV CORPSE READ BACKWARDS. That corpse:
        complete a pass without consuming, unmatched SENDS pile up, the
        senders block. Its dual: consume TWICE in one pass and the surplus
        RECV blocks for ever. ``for _ in range(8)`` reads like a bound on a
        spin; it was a licence to make eight blocking calls where the
        contract permits one.

        SO WHAT DOES A MISMATCH DO NOW? It REFUSES, loudly, and does not
        touch the wire again. That is not a repair -- it is a strictly
        better failure, and it is the same choice the shipped
        ``model_runner.forward`` shape check already makes for the same
        pair. It also catches what that check cannot: a leftover of the
        SAME width, which is silent wrong output rather than a shape error.
        With the armed drain in place this is not expected to fire at all;
        if it ever does, the message names itself and the next reader
        starts from an identity instead of a GDN kernel 30 layers deep.
        """
        if self.pp_group.is_first_rank:
            return None
        if getattr(self, "_pp_gapped_wire", False):
            # #753: under a gapped set there IS no stage-boundary handoff --
            # every activation, this rank's entry included, crosses inside the
            # forward loop. Returning None here is what lets all stages enter
            # the loop together; waiting for a proxy instead is the v7pp4
            # deadlock (see the protocol note at pp_loop_size). The model's
            # entry branch accepts None only when its wire reports
            # provides_entry_activations, so a set that is gapped without a
            # crossing into some stage's first layer still fails loudly there
            # rather than forwarding uninitialised hidden states.
            return None
        # #789: prove a message is actually coming before entering the
        # blocking receive below. See _pp_wait_for_proxy_readiness's
        # docstring for the full contract; it is a no-op when
        # pp_flip_counters is None (unchanged behaviour on such boots).
        self._pp_wait_for_proxy_readiness(mb_id)
        raw = self._pp_recv_typed_dict(
            expected_kind="proxy",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )
        # POPPED, not read: the identity has done its entire job the moment
        # the message is accepted, and what remains travels on into model
        # compute. PPProxyTensors' slice path maps v[key] over EVERY entry
        # and cuda-graph buffer copies iterate the dict, so a tuple left in
        # here would slice to nonsense rather than raise -- the worst
        # available outcome. The __msg_type__ precedent shows a non-tensor
        # entry SURVIVES THE WIRE; it does not show one is safe to compute
        # on.
        stamp = raw.pop("__stamp__", None) if isinstance(raw, dict) else None
        # #795: the comparison is (epoch, slot). A slot-only test is what let
        # the 2026-08-21 mispair through this exact line -- see
        # `pp_proxy_stamp_names_pass` for why the slot number alone is not an
        # identity across a cutover.
        epoch = pp_flip_epoch_of(self)
        if stamp is None or mb_id < 0 or pp_proxy_stamp_names_pass(stamp, mb_id, epoch):
            # #791c: THE STAMP IS RIGHT AND THE MESSAGE IS STILL WRONG. Every
            # identity above answers "which PASS is this from"; none answers
            # "which BATCH is it of". A retraction this rank performed at the
            # top of THIS pass makes its batch a strict subset of the one the
            # upstream had already launched, so the current pass's own proxy
            # is the wrong width -- or, worse, the right width and the wrong
            # requests. See `pp_proxy_pass_retraction_reason`.
            diverged = pp_pass_retraction_reason_of(self, mb_id)
            if diverged is None:
                return PPProxyTensors(raw)
            self._pp_proxy_batch_divergences = (
                getattr(self, "_pp_proxy_batch_divergences", 0) + 1
            )
            raise RuntimeError(
                f"#791c PROXY BATCH DIVERGED: a proxy stamped {stamp} arrived "
                f"for mb_id={mb_id} in flip epoch {epoch} and its identity is "
                f"CORRECT -- it is this pass, this slot, this ring. It is "
                f"still unpairable, because {diverged}. Computing on it pairs "
                f"one request set's hidden states with another's metadata; "
                f"when the two sets happen to have the same token count "
                f"(chunked prefill caps every chunk at the same size, so this "
                f"is common) that is silent wrong output, which the width "
                f"check in model_runner.forward cannot see at all. The "
                f"upstream cannot be told: it built and launched its batch "
                f"from the decision as it stood BEFORE this rank's "
                f"retraction, and a batch in flight cannot be amended. The "
                f"defect to chase is therefore upstream of this line -- "
                f"either PP0 must not offer a prefix a downstream rank cannot "
                f"honour (PPAdmissionCongruenceGuard.prefix_len_for is the "
                f"floor built for that, and #796 removed the return trip that "
                f"fed it), or a retraction must void the whole pass on every "
                f"rank rather than only narrow this one's batch."
            )

        self._pp_proxy_drops = getattr(self, "_pp_proxy_drops", 0) + 1
        raise RuntimeError(
            f"#631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id={stamp[0]} "
            f"seq={stamp[1]} rows={stamp[2]} epoch={pp_proxy_stamp_epoch(stamp)} "
            f"arrived while this rank is on mb_id={mb_id} in flip epoch "
            f"{epoch}. It belongs to a pass this rank did not run -- either "
            f"another slot of this epoch (in practice one sent by an upstream "
            f"that resumed while this rank was still armed), or, when the "
            f"epochs differ, a pass from before a cutover that rebuilt this "
            f"rank's whole slot ring, whose slot number therefore names "
            f"nothing here however well it matches (#795). Computing on it "
            f"would pair one microbatch's hidden states with another's "
            f"metadata and corrupt memory rather than merely fail; taking "
            f"another message instead wedges the pipeline (corpse R), because "
            f"the wire owes exactly one message per pass. The drains "
            f"(pp_flip_drain_tensor_dicts while armed, "
            f"pp_flip_drain_leftover_dicts at disarm) are what is supposed to "
            f"prevent this from ever being reached -- if you are reading this, "
            f"they did not, and THAT is the defect to chase."
        )

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
        mb_id: int = -1,
    ) -> Dict[str, torch.Tensor]:
        # #802-ring: the OUTPUT wire gets the gate the PROXY wire has had
        # since #789. This receive is where PP1 sat, for ever, in specimen
        # wedge_802f_1712 while PP0 held a request-chain flush PP1 could only
        # take at the top of its next pass and PP2 waited on PP1's own chain
        # send -- a closed ring whose only cuttable arc is this one. A no-op
        # when `pp_flip_counters` is None, and on every pass where the
        # upstream has posted; see `_pp_wait_for_dict_readiness`.
        self._pp_wait_for_dict_readiness(mb_id, kind="output")
        return self._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=(
                self.attn_tp_group if self.require_attn_tp_allgather else None
            ),
        )

    def _pp_make_skip_output_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: Optional[PPBatchMetadata],
    ):
        bs = len(batch.reqs)
        placeholder = torch.zeros(bs, dtype=torch.int64, device=self.device)
        # next_pp_outputs = None so non-last ranks skip forwarding
        # (pp_outputs is None gate). Placeholder carried in
        # batch_result.next_token_ids for process_batch_result_prefill.
        batch.output_ids = placeholder
        batch_result = GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=placeholder,
            can_run_cuda_graph=(
                mb_metadata.can_run_cuda_graph if mb_metadata else False
            ),
            skipped_output_comm=True,
        )
        d2h_event = self.device_module.Event()
        d2h_event.record(self.device_module.current_stream())
        return None, batch_result, d2h_event

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        batch.input_ids = pp_outputs["next_token_ids"].to(torch.int64)
        # PP rank 0 also relays into output_tokens_buf so the next iter's
        # resolve_forward_inputs finds these tokens for the decode portion
        # of mixed-chunk batches (which gather via mix_running_indices).
        self.future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=batch.input_ids)
        )
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
        )
        return output_result

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque,
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        # #753: SYNCHRONOUS UNDER A GAPPED SET, so there is nothing left to
        # flush later. An isend defers its completion to a flush in a LATER
        # iteration, which is a dependency on the peer reaching its next pass.
        # Under lockstep that is precisely what cannot be assumed: v7pp11 hung
        # with two ranks already at the iteration barrier and the third still
        # waiting on an isend those two could only take by moving past it.
        #
        # The gapped exchange is already a serialized chain -- last rank sends,
        # rank 0 receives and forwards, rank 1 receives and forwards, last rank
        # receives -- so each send has its matching receive posted within the
        # same iteration and a blocking send simply completes. Nothing is
        # queued, nothing is flushed, and the barrier has no debt to wait on.
        # The overlap an isend buys is worth nothing here anyway: the stages
        # cannot compute concurrently.
        async_output = not getattr(self, "_pp_gapped_wire", False)
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            sent_real_output = False
            target = mbs[next_first_rank_mb_id]
            if target is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                # #753: the SAME predicate the receiving side applies, so the
                # ring cannot have one rank sending while another declines.
                if _pp_output_exchange_due(target):
                    self.device_module.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_output_payload_with_return_trip(
                                self,
                                pp_outputs_to_send.tensors,
                                next_first_rank_mb_id,
                            ),
                            async_send=async_output,
                            msg_type="output",
                        )
                    sent_real_output = True
            # #791b: THE PREDICATE ABOVE IS THIS RANK'S, AND THAT IS THE HOLE.
            # It reads this rank's own slot, while the receiving side reads
            # PP0's -- and a #791 retraction empties this one without touching
            # that one, because the amendment only ever travels DOWNSTREAM and
            # PP0 is upstream of every rank that can make it. When that
            # happens PP0 is already blocked, or about to block, in a receive
            # nothing will satisfy; boot instr11 died there with all three
            # ranks in a ring. So the ring is kept matched by construction:
            # if PP0 said it expects an output for this slot and this rank has
            # none to give, it still puts exactly one message on the wire.
            # This is NOT the bounded-recv corpse in reverse -- the message is
            # one PP0 is REQUIRED to take, by the same verdict that put it on
            # the wire.
            if not sent_real_output and self._pp_output_expected_for_slot(
                next_first_rank_mb_id
            ):
                with torch.profiler.record_function("send_void_res_dict"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        self._pp_void_output_payload(next_first_rank_mb_id),
                        async_send=async_output,
                        msg_type="output",
                    )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            # #801 -- THE INTERMEDIATE HOP, AND WHAT `if pp_outputs:` NOW
            # MEANS. This predicate is not a decision about whether the
            # successor expects a message; it is the relay invariant of
            # `pp_void_forward_payload`: this rank forwards exactly when it
            # took a message off the wire for this ring generation, and a
            # void it absorbed reaches here as a truthy `pp_outputs` because
            # `_do_recv` republishes it. Before #801 a void could be absorbed
            # and forwarded NOWHERE whenever its decision named no retracting
            # rank or carried no decision at all, and the successor -- whose
            # receive is unconditional once its own slot is non-empty -- sat
            # in `_pp_recv_dict_from_prev_stage` for ever (wedge_802f_1712).
            #
            # WHAT THIS PREDICATE DOES NOT COVER, AND WHAT DOES. The case this
            # one cannot see is the inverse of the relay hole: this rank
            # received NOTHING because its OWN slot was empty while the
            # successor's was not, so `pp_outputs` is None and the successor's
            # unconditional receive has nothing coming. No rank publishes its
            # per-slot expectation to its PREDECESSOR, so this end genuinely
            # cannot answer it.
            #
            # THE RING DOES NOT NEED A WIRE AGAINST ITS OWN DIRECTION FOR IT,
            # and the note that once said so here was read as an open posting
            # for long enough to be ordered built. #951 closes the same
            # disagreement FORWARDS: `_pp_send_admission_decision` carries
            # `launched=self.mbs[mb_id] is not None`, overwritten by every hop,
            # `_pp_recv_admission_decision` reads it into
            # `_pp_upstream_launched_incoming`, and `pp_upstream_void_pending`
            # REFUSES a pass whose upstream posted nothing. The successor gives
            # up its receive instead of the predecessor acquiring an obligation
            # to send -- one hop of an existing message rather than a fifth
            # collective pair on a ring that has already buried four.
            #
            # REACHABLE AND MEASURED, not argued: the state is produced by a
            # request lost from an intermediate rank's four lookup places
            # (#944) meeting the zero offer `UNRESOLVED_DEFER_CAP` escapes
            # with, which `reconcile_pp_admission_decision` honours WITHOUT a
            # lookup -- so nothing retracts, that rank admits nothing, and its
            # successor still does. A retraction can never produce it: it
            # spreads emptiness only AWAY from rank 0.
            # `test_pp_reverse_wire_reachability_801.py` drives that path to a
            # wedge with the posting absent and to twelve clean passes with it.
            #
            # Bounded and loud rather than silent either way since #802-ring:
            # `_pp_recv_dict_from_prev_stage` enters through
            # `_pp_wait_for_dict_readiness(kind="output")`, whose timeout text
            # names this exact asymmetry.
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=async_output,
                        msg_type="output",
                    )
        return send_output_work

    def _pp_void_output_payload(self: Scheduler, mb_id: int) -> Dict[str, object]:
        """#791b: the message the last rank sends when PP0 expects an output
        for a slot the pipeline retracted out from under it.

        Carries no tensors -- a dict of pure metadata is established practice
        on this channel (`__msg_type__`, `__stamp__`, the whole admission
        decision), and fabricating a zero token tensor here is precisely what
        must not happen: `_pp_make_skip_output_result`'s placeholder is
        legitimate only because a mid-prefill chunk really produced no token,
        whereas THESE requests were never run past this rank's stage at all.

        The fully chain-reconciled decision rides back with it, which is the
        return trip #796 removed when it deleted the wraparound. Unlike that
        wraparound this one is not an unmatched message: it exists only on a
        pass where PP0's own published verdict obliges it to receive.
        """
        return pp_output_payload_with_return_trip(
            self, {_PP_VOID_OUTPUT_KEY: True}, mb_id
        )

    def _pp_absorb_void_output(
        self: Scheduler,
        mb_id: int,
        message: Dict[str, object],
        mbs: List[ScheduleBatch],
        mb_metadata: List[Optional[PPBatchMetadata]],
    ) -> bool:
        """#791b: consume a void output on the first rank. True iff it was one.

        THREE THINGS HAPPEN, AND THE ORDER IS NOT ARBITRARY.

        1. The slot is EMPTIED. `_event_loop_pp_body` guards result processing
           on `self.mbs[next_mb_id] is not None`, not on whether a result
           arrived, so leaving the batch in place would run
           `d2h_event.synchronize()` on None. Emptying it also makes
           `_pp_record_slot_last_batch` record None, which keeps the
           un-processed prefill out of `get_next_batch_to_run`'s
           `last_batch.filter_batch(...)` / `running_batch.merge_batch(...)`
           path -- where it would have become a resident decode request whose
           sampled token never existed.

        2. Every request in it is RELEASED and RE-QUEUED. This is the wiring
           `pp_admission_congruence`'s docstring names and explicitly leaves
           out of its own scope ("expected to be re-queued and re-admitted on
           a LATER pass ... that is scheduler-loop wiring"). Without it the
           batch's KV pages, mamba slot and req-pool slot have no owner at
           all: `process_batch_result_prefill` is what normally hands them to
           the tree cache, and it never runs for a voided pass.
           The release goes through `_release_voided_request`, i.e. through
           the same `release_req` both retraction paths use. It was
           `_release_dynamic_chunk_probe` until #969, "reused verbatim
           because it already frees the three in the one order that works",
           and that reasoning was itself the defect: freeing the three is not
           the whole obligation. The probe helper never reaches
           `cache_finished_req`, so it returned the pages to the allocator
           and left the tree's lock ref standing on them. A voided request is
           a RETRACTED request and is released as one.

        3. The carried decision teaches `PPAdmissionCongruenceGuard` the
           shortfall. Without it PP0 re-offers the identical `told` next pass,
           the same rank retracts again, and the void repeats for ever -- a
           degrade that makes no forward progress, which is #630's livelock
           rather than a fix. With it, `told` is clamped to the observed local
           match, and a strictly decreasing sequence of non-negative integers
           terminates.
        """
        if not isinstance(message, dict) or not message.pop(_PP_VOID_OUTPUT_KEY, False):
            return False

        # #797: decided BEFORE the pops below strip the decision out, and
        # published for `_do_recv` to forward. See `pp_void_forward_payload`
        # for why a void that stops at the first rank turns #797 into a wedge
        # whenever the retraction happens on a rank other than the first
        # downstream one.
        self._pp_void_forward_payload = pp_void_forward_payload(
            self, {_PP_VOID_OUTPUT_KEY: True, **message}
        )

        batch = mbs[mb_id] if mb_id < len(mbs) else None
        reqs = list(getattr(batch, "reqs", None) or ())
        if mb_id < len(mbs):
            mbs[mb_id] = None
        if mb_id < len(mb_metadata):
            mb_metadata[mb_id] = None

        pp_absorb_admission_return(self, message)

        # #797b: PUT THE CHUNKED REQUEST BACK WHERE THE ROUND FOUND IT, and do
        # it BEFORE the disposal loop, because that loop is what killed boot
        # instr19 (53 s, all three ranks, `'NoneType' object has no attribute
        # 'end'` at `get_next_batch_to_run`'s `self.chunked_req.extend_range.
        # end`). `self.chunked_req` is SCHEDULER state that outlives the
        # round, and `get_next_batch_to_run` is the only thing that moves it:
        # `add_chunked_req` clears it when a chunk finishes, `adder.
        # new_chunked_req` starts a new one. A rank downstream of the
        # retraction never ran that code this pass at all, so ITS chunked
        # state is the pre-admission one -- which makes the pre-admission
        # value the only one every rank can agree on, and restoring it here
        # the only disposal that leaves the ranks congruent.
        #
        # Both directions matter. A chunk STARTED this round is un-started, so
        # its request goes back to being an ordinary waiting-queue member and
        # is retracted by the loop below like any other. A chunk CARRIED into
        # this round stays carried, is excluded from that loop, and has this
        # round's prepared-but-never-run chunk parked -- see
        # `_park_chunked_prefill_chunk` for why a park and not a retraction,
        # and why only `[len(prefix_indices):extend_range.end]` may be freed.
        carried_slots = getattr(self, "_pp_chunked_req_before_by_slot", None)
        chunked_before = (
            carried_slots[mb_id]
            if carried_slots and 0 <= mb_id < len(carried_slots)
            else None
        )
        if getattr(self, "chunked_req", None) is not chunked_before:
            self.chunked_req = chunked_before
        parked = _park_chunked_prefill_chunk(self, chunked_before)

        # THE INSTR19 STATE, AND HERE IT IS REACHED RATHER THAN MERELY
        # GUARDED AGAINST. The own-void twin of this site (`#797d own-void`,
        # earlier in this file) carries the identical eight lines above and
        # then this identical check, and its comment argues the state should
        # be unreachable: `chunked_before` is a snapshot taken at the top of
        # THIS pass, so if its `extend_range` were already None going into the
        # pass, `get_next_batch_to_run`'s own unconditional read of
        # `self.chunked_req.extend_range.end` would have raised earlier in the
        # same pass. THAT ARGUMENT DOES NOT CARRY OVER TO THIS SITE, and the
        # difference is the whole reason this guard has to exist here too:
        # void-OUTPUT is driven by the retraction arriving from downstream and
        # fires ONCE PER SLOT, several times in a row, with no
        # `get_next_batch_to_run` in between. Each call restores ITS OWN
        # slot's `chunked_before` and then, in the loop below, calls
        # `reset_for_retract` on every batch member that is not kept for THAT
        # slot. A request that is slot B's carried chunk but only an ordinary
        # member of slot A's batch is therefore reset by slot A's loop and
        # then reinstated, already reset, as `self.chunked_req` by slot B's
        # restore. `pp_void_keeps_request` cannot prevent it: it is asked per
        # slot, and per slot it answers correctly.
        #
        # MEASURED, boot_798_0822_0646.log, commit 9478e774b6: rank 0 logged
        # three `#791b PP-ADMISSION void output` messages back to back on
        # slots 2, 0 and 1 at 06:51:15-06:51:16 (releasing 1 of 2, 1 of 2, and
        # then 0 of 1 -- the last request kept precisely because it was that
        # slot's chunked request), and the very next line is the scheduler
        # exception: `AttributeError: 'NoneType' object has no attribute
        # 'end'` at `self.chunked_req.extend_range.end`, raised on the
        # following pass out of `_event_loop_pp_body`. PP1 and PP2 then died
        # on gloo `Connection closed by peer`, a cascade rather than three
        # independent faults. This path only became reachable once the seam
        # could fund a cutover at all (#796): nothing survived a committed
        # `tp_to_pp` before, because the flip never committed.
        #
        # `_park_chunked_prefill_chunk` cannot repair it -- it treats
        # `extend_range is None` on the way in as "already parked, already
        # reset, or never prepared, nothing to give back" and leaves it
        # untouched, which is right for the KV-release question it answers and
        # silent on this site's obligation. Whatever is left in
        # `self.chunked_req` is what the next pass dereferences with no guard
        # of its own. Clearing it is the correct disposal and not a loss: the
        # request keeps its pages, is re-admitted from the waiting queue by
        # the ordinary path, and a chunk that was reset has no stashed tree
        # handles left to honour. The rid and slot are logged so the next boot
        # names the cross-slot pair rather than leaving it to be re-derived.
        #
        # THE CONDITION NAMES THE CLASS, NOT THE SYMPTOM, and that is
        # deliberate. `extend_range` is merely the field the next reader
        # happens to touch FIRST; `reset_for_retract` clears seventeen more in
        # the same breath (`last_node`, `swa_uuid_for_lock`, `mamba_pool_idx`,
        # `routed_experts`, `prefix_indices` back to empty, ...), and a
        # None-check on one of them would leave the next edit to rediscover
        # this bug through whichever field a future reader dereferences first.
        # `is_retracted` is the marker of the shape itself: `reset_for_retract`
        # sets it (schedule_batch.py:1590) and only `prepare_for_extend` clears
        # it (schedule_batch.py:2439), so it is True over exactly the window in
        # which the request has been reset and not yet re-prepared -- which is
        # exactly the window in which it must not be carried as the scheduler's
        # chunked request. The `extend_range` test is kept beside it as the
        # narrower belt: it is what actually raised, and a request could in
        # principle reach the reset shape by a path that does not set the flag.
        # `getattr` rather than a plain attribute read, matching the idiom the
        # eight lines above already use. A scheduler that has never set
        # `chunked_req` is a real shape here -- the restore above only assigns
        # the attribute when the carried value DIFFERS from the current one, so
        # a holder that never had the attribute and carries None still does not
        # have it by this line. `test_ring_survives_the_retraction_with_the_fix`
        # is the case that proves it, and a plain read raised there.
        chunked_carry = getattr(self, "chunked_req", None)
        if chunked_carry is not None and (
            getattr(chunked_carry, "is_retracted", False)
            or getattr(chunked_carry, "extend_range", None) is None
        ):
            logger.warning(
                "#791b void-output: the chunked request for slot %d (rid=%s) "
                "is in the reset_for_retract shape that "
                "_park_chunked_prefill_chunk cannot repair "
                "(is_retracted=%s, extend_range=%s), produced by an earlier "
                "slot's disposal loop in this same retraction -- clearing "
                "self.chunked_req instead of carrying a request the next "
                "pass's get_next_batch_to_run cannot read.",
                mb_id,
                getattr(chunked_carry, "rid", None),
                getattr(chunked_carry, "is_retracted", False),
                getattr(chunked_carry, "extend_range", None),
            )
            self.chunked_req = None

        # #797: A RESIDENT REQUEST MUST NOT BE RELEASED HERE, and this is not
        # a hardening detail -- it is a double-free. `_release_dynamic_chunk_
        # probe` hands back the request's KV pages, mamba slot and req-pool
        # row; a request that is also in `running_mbs[mb_id]` keeps decoding
        # from those same pages on the next pass, so freeing them corrupts
        # another request's cache and re-queueing it duplicates it. It was
        # unreachable while only a FULL decline could void a slot (the void
        # then carried a freshly built prefill batch, whose requests are not
        # merged into the running batch until the next visit) and #797 makes
        # it reachable, because a mixed chunked-prefill batch voids the same
        # way and carries resident decode requests in it. A resident request
        # needs nothing done to it: the pass simply did not run, and it
        # decodes again next pass from the state it still holds.
        running_mbs = getattr(self, "running_mbs", None) or ()
        running = running_mbs[mb_id] if 0 <= mb_id < len(running_mbs) else None
        resident = {r.rid for r in (getattr(running, "reqs", None) or ())}
        released = 0
        for req in reqs:
            if pp_void_keeps_request(req, resident, chunked_before):
                continue
            # #969: THE RETRACTION PATH, NOT THE PROBE PATH -- same reason as
            # the twin site in `_pp_void_own_batch`. The probe helper never
            # reaches `dec_lock_ref`, so every request voided here used to
            # leave a permanent lock on its prefix and the tree reported
            # nothing evictable. `release_req` resets the request as its last
            # act, so this site does not.
            _release_voided_request(self, req, len(reqs) - released)
            self.waiting_queue.append(req)
            released += 1

        logger.warning(
            "#791b PP-ADMISSION void output on slot %d: the pipeline retracted "
            "this microbatch downstream, so the last rank produced nothing for "
            "it, and %d of rank 0's %d request(s) have been released and "
            "re-queued (the rest are resident in the running batch, or are the "
            "chunked request, and keep their pages -- #797/#797b; chunk "
            "parked=%s). The message itself is what keeps the output ring "
            "matched -- without it rank 0 blocks for ever in a receive no rank "
            "is required to satisfy (boot instr11, 2026-08-21).",
            mb_id,
            released,
            len(reqs),
            parked,
        )
        return True

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # On CUDA, isend is async: it enqueues to the stream and returns,
        # so every rank can send first safely. On some backends isend is
        # effectively blocking and does not return until the peer posts a
        # matching recv; if every PP rank sends first, all ranks block
        # waiting for a receiver and the ring deadlocks. Order send/recv
        # by pp_rank parity (even: send->recv, odd: recv->send) so each
        # adjacent pair has one sender and one receiver posted at the
        # same time.

        # CUDA: send first
        # XPU: even ranks send first, odd ranks recv first.
        send_first = (not is_xpu()) or ((self.ps.pp_rank % 2) == 0)

        def _do_send(forward_now=_NOT_SUPPLIED):
            # ``forward_now`` is the gapped path's lag removal: forward what
            # this rank received THIS iteration instead of last iteration's
            # ``pp_outputs``.
            #
            # THE SENTINEL IS THE WHOLE POINT, and its absence cost boot
            # v7pp17. Defaulting to None made "no exchange was due, I received
            # nothing" indistinguishable from "not supplied", so the gapped
            # path fell back to the STALE pp_outputs from the last pass that
            # did exchange. On an idle tick that is an unmatched send -- the
            # peer's receive early-returns because its slot is empty -- and
            # since gapped sends are synchronous it blocks for ever. The rank
            # never reaches the iteration barrier, and the barrier reports what
            # it can see: 'made no progress for 120s and no peer could be
            # proven dead'. The peer was not dead, it was holding a send that
            # nobody had any reason to take.
            #
            # With the sentinel, receiving nothing means sending nothing, which
            # is the same predicate the receiving side used to decide it had
            # nothing to receive.
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs if forward_now is _NOT_SUPPLIED else forward_now,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            target = mbs[next_mb_id]
            if target is None or target.forward_mode.is_prebuilt():
                return
            if _pp_can_skip_output_comm(target):
                next_pp_outputs, batch_result, d2h_event = (
                    self._pp_make_skip_output_result(target, mb_metadata[next_mb_id])
                )
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                # #802-ring: the slot is passed so the readiness gate can name
                # it. It is the slot this rank's own gate above just proved
                # non-empty -- exactly the expectation no upstream is told.
                raw_output = self._pp_recv_dict_from_prev_stage(next_mb_id)
            # #791b: a void carries no tokens and must not be turned into one.
            # `_pp_absorb_void_output` empties the slot, so the loop's "slot
            # non-empty => a result was received for it" invariant (the guard
            # at `_event_loop_pp_body`'s `if self.mbs[next_mb_id] is not None`)
            # holds with all three of `next_pp_outputs`, `batch_result` and
            # `d2h_event` left as None.
            if self._pp_absorb_void_output(next_mb_id, raw_output, mbs, mb_metadata):
                # #797: PASS THE VOID ON when a rank between this one and the
                # retraction still holds a launched batch for this slot. It
                # rides the ordinary one-iteration output lag -- this is the
                # same `pp_outputs` a real output would have been forwarded
                # by, and `_do_send`'s `if pp_outputs:` is the same gate.
                # `pp_void_forward_payload` returns None on every ordinary
                # pass and at the last rank that needs it, so nothing is put
                # on the wire that no peer must take.
                forward = getattr(self, "_pp_void_forward_payload", None)
                if forward is not None:
                    next_pp_outputs = PPProxyTensors(forward)
                    self._pp_void_forward_payload = None
                return
            # #797: a SUCCESSFUL pass carries its chain-reconciled decision
            # home too, not only a voided one, and it is popped here for the
            # same reason `__stamp__` is popped in `_pp_recv_proxy_tensors` --
            # what is left becomes a PPProxyTensors and is forwarded on.
            # Learning only from voids leaves `record_return_trip` with no way
            # to CLEAR a floor, which is #796's named residual cost.
            pp_absorb_admission_return(self, raw_output)
            next_pp_outputs = PPProxyTensors(raw_output)
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    target, mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        if getattr(self, "_pp_gapped_wire", False):
            # #753: the send gate reads mbs[next_first_rank_mb_id] and the
            # receive gate reads mbs[next_mb_id]. Under lockstep those must be
            # the SAME slot or the two gates can answer differently about
            # different batches -- the v7pp12 starve. At pp_loop_size 1 both
            # collapse to 0; this refuses rather than trusting that they will.
            if next_first_rank_mb_id != next_mb_id:
                raise AssertionError(
                    f"gapped output ring needs one slot: the send gate reads "
                    f"slot {next_first_rank_mb_id} and the receive gate reads "
                    f"slot {next_mb_id}. Those indices agree only under the "
                    f"pipeline stagger, which a gapped set removes -- one rank "
                    f"then sends while another declines to receive and the "
                    f"ring starves with no peer provably dead (boot v7pp12). "
                    f"pp_loop_size must be 1 on this path."
                )
            # THE RING'S ONE-ITERATION LAG IS WHAT DEADLOCKS LOCKSTEP.
            #
            # Outputs originate at the last rank and travel 2 -> 0 -> 1 -> 2.
            # A middle rank forwards ``pp_outputs``, which is what it received
            # LAST iteration -- so on the first pass rank 0 has nothing to
            # forward and rank 1 blocks on a send that will not come until rank
            # 0 has been round again. The contiguous path hides this because
            # the proxy chain gates each stage's forward, so a downstream rank
            # cannot reach its output receive before the upstream has been
            # through its own. Take the chain away and the lag is exposed:
            # boot v7pp6 wedged with PP0 already at layer 4 of the next decode
            # pass while PP1 and PP2 sat in the output receive.
            #
            # Under a gapped set the stages are in lockstep by construction --
            # the crossings ARE a rendezvous -- so there is no stagger to
            # absorb a lag, and the fix is to remove the lag rather than to
            # reintroduce a stagger that the layout cannot have. A middle rank
            # receives FIRST and forwards what it just received, in the same
            # iteration. The last rank still sends first: it is the source, it
            # already holds this pass's outputs, and its send is what starts
            # the chain.
            #
            # 2 sends -> 0 receives -> 0 forwards -> 1 receives -> 1 forwards
            # -> 2 receives. One pass, no debt carried across iterations.
            if self.pp_group.is_last_rank:
                send_output_work = _do_send()
                _do_recv()
            else:
                _do_recv()
                send_output_work = _do_send(forward_now=next_pp_outputs)
        elif send_first:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        cur_batch: ScheduleBatch,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(cur_batch, pp_proxy_tensors)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, cur_batch)
                            ),
                        )
                    )
        return result, event

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                rids_to_check=good_consensus_prealloc_rids
                + bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            if self.enable_hisparse:
                for req in released_reqs:
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(length: float) -> float:
            """Total latency function: f(length) = a*length^2 + b*length + c."""
            return (
                self.quadratic_coeff_a * length * length
                + self.linear_coeff_b * length
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
