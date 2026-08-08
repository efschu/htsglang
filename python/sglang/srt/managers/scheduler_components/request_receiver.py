from __future__ import annotations

from dataclasses import dataclass, field
import time
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
    # #631(a): seconds to wait for a chain batch before returning empty so
    # this stage can re-evaluate its own round hooks. 0.0 = the upstream
    # unbounded wait, which is the default and every non-policy boot.
    chain_recv_poll_s: float = 0.0
    # In-flight chain receive, carried ACROSS iterations. A posted irecv
    # cannot be abandoned -- the sender will still deliver into it -- so a
    # bounded poll must keep the same Work and re-poll it next time, never
    # post a second one. dict (not a plain attr) because this dataclass is
    # frozen.
    chain_recv_state: dict = field(default_factory=dict)

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

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _chain_recv_bounded(self, src: int, dst: int) -> List:
        """#631(a): one bounded step of the PP request-chain receive.

        Returns the batch when one has arrived, or ``[]`` when the poll
        window expired with nothing in flight yet. Returning empty is a
        NORMAL outcome, not an error: it hands control back to the event
        loop so this stage can run its round hooks (notably the phase
        flip's consensus join) and come back here.

        THE INVARIANT: a posted ``irecv`` is a standing promise to the
        sender, which will deliver into that exact tensor. It can never
        be abandoned on a timeout. So the Work and its tensor live in
        ``chain_recv_state`` and are re-polled on later calls; a second
        irecv is never posted while one is outstanding. Getting this
        wrong desynchronises the chain rather than merely slowing it.

        Only the SIZE header is polled. Once it lands, the sender is
        already committed to the payload, so the body is read with the
        ordinary blocking wait -- it is in flight by construction and
        bounding it would buy nothing.
        """
        import pickle

        import numpy as np
        import torch
        import torch.distributed as dist

        st = self.chain_recv_state
        work = st.get("work")
        if work is None:
            tensor_size = torch.tensor([0], dtype=torch.long)
            work = dist.irecv(
                tensor_size, src=src, group=self.world_group.cpu_group
            )
            st["work"] = work
            st["tensor_size"] = tensor_size

        deadline = time.perf_counter() + self.chain_recv_poll_s
        while not work.is_completed():
            if time.perf_counter() >= deadline:
                # Still outstanding: keep it for the next call and let
                # this stage get on with its round.
                return []
            time.sleep(0.001)
        work.wait()

        size = int(st["tensor_size"].item())
        st.clear()
        if size == 0:
            return []
        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.irecv(
            tensor_data, src=src, group=self.world_group.cpu_group
        ).wait()
        return pickle.loads(tensor_data.numpy().tobytes())

    def _pull_raw_reqs(self) -> Optional[List]:
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
                if self.chain_recv_poll_s:
                    # #631(a): BOUNDED chain recv, so an idle downstream
                    # stage is a CYCLING stage.
                    #
                    # The unbounded wait below is what makes an automatic
                    # flip impossible at idle. Measured 2026-08-08 (boot
                    # 10): all three ranks armed identically, rank 0's
                    # park predicate went true first and it entered the
                    # flip's blocking reduction -- and entering it stops
                    # rank 0 forwarding the very batch ranks 1 and 2 are
                    # blocked on here. They had already evaluated
                    # on_round once while NOT yet parked, returned, and
                    # come back here to block forever. No one can join,
                    # and rank 0 cannot time out because the park
                    # deadline is checked before entry, not from inside.
                    #
                    # An unbounded wait in a coordination path is also
                    # against this fork's bounded-wait canon (#610/#615/
                    # #653). With a bound, a stage that has nothing to
                    # receive returns empty, re-reaches its own
                    # on_round -- by then parked -- and joins. Idle
                    # coordination then needs no chain message at all,
                    # which is what lets the policy stay message-free.
                    recv_reqs = self._chain_recv_bounded(
                        src=(self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
                        dst=self.ps.pp_rank * self.ps.tp_size + dp_offset,
                    )
                else:
                    recv_reqs = point_to_point_pyobj(
                        [],
                        self.ps.pp_rank * self.ps.tp_size + dp_offset,
                        self.world_group.cpu_group,
                        (self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
                        self.ps.pp_rank * self.ps.tp_size + dp_offset,
                    )
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
