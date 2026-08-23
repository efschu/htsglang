from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Union,
)

import zmq
from torch.distributed import barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.utils import (
    broadcast_pyobj,
    point_to_point_pyobj,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy]
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: ParallelState
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_forward_mode: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None
    # #631: returns a PhaseFlipReqInput when the automatic phase policy
    # wants a flip, else None. Optional and defaulted so every existing
    # construction is unchanged and the default path costs one compare.
    phase_policy_hook: Optional[Callable[..., Any]] = None
    # #631: the single owner of this rank's request-chain receive stream
    # (PpChainReceiver), installed only when the phase flip is enabled.
    # None restores the direct point_to_point_pyobj call, i.e. the exact
    # upstream behaviour for every boot without the flip.
    chain_receiver: Optional[Any] = None
    # #631: True while a phase flip is armed on this rank. An armed rank
    # must not take on new work and must not block on the chain -- see
    # _pull_raw_reqs.
    phase_flip_armed_hook: Optional[Callable[[], bool]] = None
    # #631 G: one turn of the ARMED SERVICE LOOP -- consume every inbound
    # message the upstream's counter accounts for, then reap the sends the
    # downstream's counter proves consumed. Replaces the poll-based drain,
    # which absorbed nothing (corpse F).
    phase_flip_service_hook: Optional[Callable[[], None]] = None
    # #824 W5(b): called as on_blocked_recv(arm, since) around the DIRECT
    # chain receive below, and with (None, None) when it returns. Without
    # it that call is a blocking PP receive that records nothing, so the
    # watchdog cannot name it -- the same blind spot #821 left on the
    # chain_receiver path, which is where two of three ranks wedged on
    # boot_827. The PpChainReceiver path stamps its own marker instead.
    on_blocked_recv: Optional[Callable[[Optional[str], Optional[float]], None]] = None
    # #824 W4a: run one drain turn when the chain receive reports a CLOSED
    # RING (PpChainRecvStalled), then resume the SAME posted receive. None
    # keeps the pre-#824 behaviour of letting the stall propagate.
    pp_chain_stall_service: Optional[Callable[[], Any]] = None

    #: How many drain turns one chain receive may trigger before the stall
    #: is allowed to propagate. A closed ring is cut by the FIRST turn; a
    #: second means the drain did not release the peer, and spinning on it
    #: would replace a loud wedge with a quiet one.
    MAX_STALL_SERVICE_TURNS = 2

    def _recv_chain_breaking_closed_rings(self):
        """#824 W4a: the blocking chain receive, plus the ring-cut.

        ``PpChainRecvStalled`` does not mean "give up". It means the chain
        receive has PROVEN the ring is closed -- this rank's CHAN_DICT
        upstream has entered a send it cannot finish until this rank drains
        it -- and the receive is still posted and still framed. So the
        answer is to drain that wire and come back to the same receive,
        which is what the resumability of ParkedWait exists for.

        The stall is re-raised rather than swallowed if servicing does not
        clear it, because at that point the ring is closed for a reason this
        code does not model, and a caller that keeps retrying would turn a
        diagnosable wedge into an invisible one.
        """
        from sglang.srt.managers.pp_chain_receiver import PpChainRecvStalled

        for turn in range(self.MAX_STALL_SERVICE_TURNS + 1):
            try:
                return self.chain_receiver.recv()
            except PpChainRecvStalled:
                if self.pp_chain_stall_service is None or turn >= self.MAX_STALL_SERVICE_TURNS:
                    raise
                logger.warning(
                    "PP-CHAIN-RECV closed ring detected on the request "
                    "chain; running drain turn %d to release the upstream's "
                    "dict send, then resuming the same posted receive.",
                    turn + 1,
                )
                self.pp_chain_stall_service()

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_forward_mode()):
                return []

        recv_reqs = self._pull_raw_reqs()

        # #631 automatic phase policy. The arm rides the SAME chain a
        # manual POST /phase_flip uses, because forwarding it is what
        # WAKES the downstream stages out of their blocking chain recv --
        # see maybe_arm_phase_policy. The ordering that makes it safe is
        # DELIVERY-BEFORE-BLOCK, enforced in scheduler_pp_mixin.
        #
        # THE GUARD IS THE ZMQ-INTAKE TEST, never "did I get a list". In
        # the PP phase _pull_raw_reqs returns a list on EVERY stage --
        # rank 0 from zmq, ranks 1..n-1 from point_to_point_pyobj -- so a
        # list-based guard is true everywhere and every stage injects its
        # own arm. Measured 2026-08-08: 1/2/3 arms on PP0/PP1/PP2, a
        # 12765-line capture-census flood, and a self-kill.
        is_request_origin = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        if (
            is_request_origin
            and recv_reqs is not None
            and self.phase_policy_hook is not None
        ):
            # #713: hand the policy the batch it is riding in. It evaluates
            # BEFORE these requests are queued (scheduler.py:4089), so without
            # them it asks "is there prefill work?" of a queue that has not
            # been told yet, reads 0, and declines the flip toward the very
            # work that woke it -- measured at 31.64 s TTFT for a ten-token
            # prompt on an idle box. The ordering itself is load-bearing
            # (DELIVERY-BEFORE-BLOCK) and is deliberately NOT changed here.
            policy_req = self.phase_policy_hook(recv_reqs)
            if policy_req is not None:
                recv_reqs.append(policy_req)

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _phase_flip_armed(self) -> bool:
        if self.phase_flip_armed_hook is None:
            return False
        try:
            return bool(self.phase_flip_armed_hook())
        except Exception:  # noqa: BLE001 - never let a probe break intake
            return False

    def _pull_raw_reqs(self) -> Optional[List]:
        # #631 THE ARMED INTAKE RULE. A rank with a flip armed admits NO
        # new work and BLOCKS ON NOTHING:
        #
        #   rank 0    leaves its requests in the zmq socket. Not reading
        #             them is what buffers them -- there is no queue to
        #             manage and nothing can be lost.
        #   rank k>0  keeps CONSUMING the chain -- greedily, and bounded by
        #             transfer time rather than by peer scheduling, because
        #             the upstream's published counter proves each message
        #             exists before the blocking receive is made (#631 G).
        #             Consuming is not optional: boot 18 measured what
        #             happens when an armed rank stops. Rank 2 armed and
        #             stopped reading, so rank 1's ordinary top-of-pass
        #             commit of the previous forward blocked in work.wait()
        #             BEFORE rank 1 could announce presence. The gate never
        #             assembled and rank 0 sat in the reduction alone.
        #             The blocking point preceded the gate, so the gate
        #             could never have covered it -- the fix has to be
        #             here, at the obligation itself.
        #
        # THE SERVICE TURN RUNS ON EVERY RANK, rank 0 included. Its consume
        # half is a no-op there (no upstream chain) but its flush half is
        # not: rank 0 still owes the reaping of the forward it issued in
        # the pass it armed, and while that handle is unreaped it "owes a
        # send" and withholds presence for ever.
        #
        # Both halves return an EMPTY list rather than None: empty means
        # "no new work this pass", which every downstream step already
        # handles, whereas None means "not the intake rank".
        if self._phase_flip_armed():
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                if self.phase_flip_service_hook is not None:
                    self.phase_flip_service_hook()
                return []
            return None

        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = sock_recv(self.recv_from_rpc, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                dp_offset = (
                    self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
                )
                if self.chain_receiver is not None:
                    # #631: ONE owner of this stream. The receiver drains
                    # its inbox first, so messages it absorbed while a
                    # flip was armed are handed over here in arrival
                    # order before anything new is taken off the wire.
                    # Routing the blocking path through it as well is
                    # what keeps a half-received message from being
                    # misframed by a second, competing irecv.
                    recv_reqs = self._recv_chain_breaking_closed_rings()
                else:
                    src = (self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset
                    if self.on_blocked_recv is not None:
                        self.on_blocked_recv(
                            f"request-chain/point_to_point<-{src}", time.monotonic()
                        )
                    try:
                        recv_reqs = point_to_point_pyobj(
                            [],
                            self.ps.pp_rank * self.ps.tp_size + dp_offset,
                            self.world_group.cpu_group,
                            src,
                            self.ps.pp_rank * self.ps.tp_size + dp_offset,
                        )
                    finally:
                        # Cleared even when the receive RAISES, so a dead
                        # peer's "Connection closed by peer" cannot leave a
                        # stale timestamp that later reads as a wedge.
                        if self.on_blocked_recv is not None:
                            self.on_blocked_recv(None, None)
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
        if self.server_args.enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.ps.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.ps.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = self.server_args.enable_dp_attention_local_control_broadcast
            if _local_ctrl:
                if self.ps.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.ps.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            elif self.ps.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.ps.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return recv_reqs

    def unwrap_pickle_wrapper(self, recv_reqs: Optional[List]) -> None:
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                req.unwrap_pickle_fields()
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                for sub_req in req:
                    sub_req.unwrap_pickle_fields()

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        if (
            self.ps.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        if recv_reqs:
            if self.model_config.is_multimodal and has_shm_features(recv_reqs):
                # The broadcast source returns with its original objects while
                # peer ranks may still be unpickling ShmPointerMMData
                # (-> shm_open).  Synchronize the same CPU groups that carried
                # SHM-backed work requests before materialize() unlinks them.
                if self.server_args.enable_dp_attention:
                    if self.ps.attn_tp_size > 1:
                        barrier(group=self.attn_tp_cpu_group)
                    if self.ps.attn_cp_size > 1:
                        barrier(group=self.attn_cp_cpu_group)
                elif self.ps.tp_size > 1:
                    barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs
