from __future__ import annotations

import logging
import math
import os
import time
from array import array
from collections import deque
from dataclasses import dataclass
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
from sglang.srt.managers.phase_flip_counters import (
    CHAN_DICT,
    CHAN_PASS,
    CHAN_REQ,
    CHAN_SLOT,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
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
from sglang.srt.utils.common import get_device_module, is_xpu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


_PP_STATS_UNSET = object()


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


def classify_armed_drain_message(msg, ran_mb_ids) -> tuple:
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
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.request_receiver.recv_requests()
                self._pp_forward_and_process_input_requests(recv_reqs)
                with torch.profiler.record_function("get_next_batch_to_run"):
                    plan = self.get_next_batch_to_run(
                        running_batch=self.running_batch, last_batch=self.last_batch
                    )
                    self.running_batch = plan.running_batch
                    self.mbs[mb_id] = plan.batch_to_run
                self.running_mbs[mb_id] = self.running_batch
                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors(mb_id)
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
                # already posted (proxy isend above, output commit above).
                # See _pp_commit_pending_req_work and the comment at the
                # #788 send site in _pp_forward_and_process_input_requests
                # for why the old top-of-pass position could close a cycle.
                # Guarded the same way the send site is: the last rank never
                # posts a chain send, so it has nothing to flush here either.
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
            action, kind, why = classify_armed_drain_message(
                raw, ran_fn() if ran_fn is not None else set()
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
            if stamp is None or live_mb_id < 0 or int(stamp[0]) == int(live_mb_id):
                # Owed, not leftover: this is the pass the rank is about to
                # run. Stash so the ordinary receive returns it.
                stash_typed(self.pp_group, None, "proxy", raw)
                continue
            discarded += 1
            logger.info(
                "%s #757 drained a LEFTOVER proxy at disarm: stamp=%s while this "
                "rank resumes on mb_id=%s. It names a pass launched before the arm "
                "that this rank never ran, so no batch can ever pair with it. "
                "(%d this window)",
                "PHASE-FLIP",
                stamp,
                live_mb_id,
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
        stashed = sum(
            len(q) for q in getattr(self, "_pp_tensor_dict_inbox", {}).values()
        )
        if stashed:
            reasons.append(f"tensor-dict inbox holds {stashed} stashed message(s)")

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
        tensor_dict = recv_typed_tensor_dict(
            self.pp_group,
            expected_kind,
            src=None,
            all_gather_group=all_gather_group,
            on_message=_off_the_wire,
        )
        if expected_kind == "default":
            logger.warning_once(
                f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        return tensor_dict

    def _pp_proxy_stamp(self: Scheduler, mb_id: int, result) -> tuple:
        """#631 VARIANT B: the identity a proxy message carries.

        (mb_id, monotone seqno, row count). The seqno is per rank and never
        resets, so it distinguishes two messages for the SAME slot -- which
        is exactly the pair a stranded leftover creates.
        """
        self._pp_proxy_seq = getattr(self, "_pp_proxy_seq", 0) + 1
        rows = -1
        try:
            hs = result.pp_hidden_states_proxy_tensors.tensors.get("hidden_states")
            if hs is not None:
                rows = int(hs.shape[0])
        except Exception:  # noqa: BLE001 - a stamp may never break a send
            pass
        return (int(mb_id), int(self._pp_proxy_seq), rows)

    def _pp_wait_for_proxy_readiness(self: Scheduler, mb_id: int) -> None:
        """#789: refuse to enter the blocking proxy receive until there is
        POSITIVE evidence a message is actually coming.

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
        this rank's poll sees the same "nothing new" either way, so a
        stuck SENDER is covered identically to a sender that never
        intended to send at all. What this gate does NOT cover: a send
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
            if typed_inbox(self.pp_group).get((src, "proxy")):
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
            now = time.monotonic()
            if deadline is None:
                deadline = now + _pp_proxy_readiness_budget_s()
            if now < deadline:
                time.sleep(PROXY_READINESS_POLL_STEP_S)
                continue
            budget = _pp_proxy_readiness_budget_s()
            logger.error(
                "%s #789 PROXY READINESS TIMEOUT: mb_id=%s -- this rank's "
                "upstream (rank %s) has posted %d dict message(s) on CHAN_DICT "
                "and this rank has consumed %d; no new message appeared within "
                "%.1fs. No upstream scheduled work for this slot -- refusing "
                "to enter the blocking proxy receive rather than wedge.",
                "PHASE-FLIP",
                mb_id,
                upstream,
                posted,
                consumed,
                budget,
            )
            raise RuntimeError(
                f"#789 PROXY READINESS TIMEOUT: mb_id={mb_id}: this rank's "
                f"upstream (rank {upstream}) posted {posted} dict message(s) "
                f"on CHAN_DICT, this rank has consumed {consumed}, and no new "
                f"message appeared within {budget:.1f}s. No upstream scheduled "
                f"work for this slot; refusing to enter an unbounded blocking "
                f"receive. See #789."
            )

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
        if stamp is None or mb_id < 0 or int(stamp[0]) == int(mb_id):
            return PPProxyTensors(raw)

        self._pp_proxy_drops = getattr(self, "_pp_proxy_drops", 0) + 1
        raise RuntimeError(
            f"#631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id={stamp[0]} "
            f"seq={stamp[1]} rows={stamp[2]} arrived while this rank is on "
            f"mb_id={mb_id}. It belongs to a pass this rank did not run -- "
            f"in practice one sent by an upstream that resumed while this "
            f"rank was still armed. Computing on it would pair one "
            f"microbatch's hidden states with another's metadata and "
            f"corrupt memory rather than merely fail; taking another "
            f"message instead wedges the pipeline (corpse R), because the "
            f"wire owes exactly one message per pass. The armed drain "
            f"(pp_flip_drain_tensor_dicts) is what is supposed to prevent "
            f"this from ever being reached -- if you are reading this, it "
            f"did not, and THAT is the defect to chase."
        )

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
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
            target = mbs[next_first_rank_mb_id]
            if target is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                # #753: the SAME predicate the receiving side applies, so the
                # ring cannot have one rank sending while another declines.
                if _pp_output_exchange_due(target):
                    self.device_module.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=async_output,
                            msg_type="output",
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=async_output,
                        msg_type="output",
                    )
        return send_output_work

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
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
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
