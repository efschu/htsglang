# Copyright 2023-2026 SGLang Team
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
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators import (
    barlink_abort_gate,
    barlink_capture_census,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.moe import offload_capture_gate
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.speculative import adaptive_graph_memory
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.jit_cold_build import run_capture_warmups
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        # Draft-solo placement: the solo host's DRAFT graphs are captured
        # rank-LOCALLY (weight-TP=1 model, collective-free forward) while the
        # shadow ranks skip draft capture entirely — a TP-group barrier here
        # would wait on ranks that never enter it and deadlock the boot.
        self._skip_warmup_barrier = getattr(
            cuda_graph_runner.model_runner, "spec_solo_rank_local_graphs", False
        )
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        # Adaptive graph-memory offload (Stage 2): a state build in progress
        # supplies its own private, pauseable capture pool; the shared global
        # pool is neither created nor touched. Backend instances are
        # per-runner, and adaptive runners are per-state, so caching the
        # override on self._pool is safe.
        pool_override = adaptive_graph_memory.capture_pool_override()
        if pool_override is not None:
            self._pool = pool_override
        elif self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        #
        # "kernels are loaded" is where the cold-build collision lives: several
        # jit_kernel modules BUILD on first call, so on an empty cache this
        # loop can sit minutes in nvcc while a peer rank is already waiting in
        # a deadline-bearing device collective. run_capture_warmups runs the
        # identical loop inside the cold-build window, which relaxes that
        # deadline for exactly these forwards -- the recorded pass below stays
        # outside it, so the captured graph keeps the steady-state deadline.
        run_capture_warmups(
            forward_fn,
            repeats=2,
            device_module=self._device_module,
            tp_group=self._tp_group,
            skip_barrier=self._skip_warmup_barrier,
            post_warmup_hook=post_warmup_hook,
            reason="full cuda-graph capture warmup",
        )

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        # Stage-2 adaptive offload routes the capture through the build tag's
        # torch_memory_saver region (pauseable capture pool); a no-op
        # passthrough to graph_ctx otherwise.
        # #603b: everything a replay of THIS graph will do is decided in the
        # recorded pass below, and no instrument observes a replay -- the
        # collective census counts host calls (a replay makes none) and the
        # launch record's unchecked counter deliberately does not advance
        # under capture. So the collectives are recorded here, keyed by the
        # shape, and diffed across ranks once at the first scheduler census
        # tick. The key is the ShapeKey, which carries no rank-local
        # component, so the segments line up across ranks by construction.
        with barlink_capture_census.segment(f"full/{shape_key}"):
            with adaptive_graph_memory.capture_graph_ctx(
                graph_ctx,
                cuda_graph=graph,
                pool=self._pool,
                stream=self._capture_stream,
            ):
                out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        # #622: name the window BEFORE the launch. If the kernels in this
        # graph abort, the check three lines below is the host point that
        # reports it, and without this it can only say "some replayed graph".
        # Host-side, outside any capture, five stores, no allocation.
        barlink_abort_gate.note_replay("full", shape_key)
        self._graphs[shape_key].replay()
        # #431: a captured graph contains the barlink BAR1 spin kernels but no
        # host code between them, so the per-collective abort check cannot
        # fire during a replay -- this is the next host point after it. Costs
        # one truth test on an empty list unless a BAR1 transport is live in
        # this process; see barlink_abort_gate for what it costs when one is.
        barlink_abort_gate.check_after_graph_replay()
        # Same boundary, same reason (#443): the capturable MoE expert-offload
        # counts an unreachable cold row on device because testing for it in
        # the step would be a host read inside the capture. Empty registry
        # unless a #394 shared cold tier is live.
        offload_capture_gate.check_after_graph_replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
