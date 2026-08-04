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
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import offload_capture_gate
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.jit_cold_build import run_capture_warmups
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


#: ``LogitsProcessorOutput`` fields the BCG buffer layer carries through a
#: captured segment (#462). Every one of them is indexed by the batch's TOKEN
#: count on this path, and the contract is read off the code, not assumed:
#:
#: * ``DecodeCudaGraphRunner.execute`` -- the only consumer of a decode replay
#:   output -- slices ``next_token_logits``, ``full_logits``, ``hidden_states``
#:   and ``cross_aux_hidden_states`` with the SAME ``[: raw_num_token]``
#:   (``decode_cuda_graph_runner.py:2112-2144``). That is the downstream
#:   statement that the four share a leading dimension here.
#: * ``LogitsProcessor._get_pruned_states`` is why they do: in
#:   ``decode_or_idle`` / ``target_verify`` / ``draft_extend_v2`` -- the three
#:   modes a decode graph captures -- it sets ``pruned_states = hidden_states``
#:   and ``sample_indices = None`` (``logits_processor.py:585-596``), so the
#:   sampled logits are not gathered down to one row per sequence and both the
#:   logits and the stored hidden states keep one row per token.
#: * ``mm_input_embeds`` is a pass-through of ``forward_batch.mm_input_embeds``
#:   (``logits_processor.py:519``), which is ``[#token, hidden]``
#:   (``mm_utils.py:1140``). The prefill runner slices it by ``raw_num_tokens``
#:   (``prefill_cuda_graph_runner.py:1192``); the decode runner drops it. It is
#:   buffered rather than dropped here so nothing depends on which of those two
#:   a future reader copies.
#:
#: The one place in the tree where these fields do NOT share a leading
#: dimension is PREFILL: ``prefill_cuda_graph_runner.py:1194`` slices
#: ``next_token_logits`` by ``raw_bs`` (per SEQUENCE, from the last-token prune
#: above) while ``hidden_states`` goes by ``raw_num_tokens``. That case cannot
#: reach this branch: with a prefill BCG the runner captures the LAYER MODEL
#: body, which returns a bare tensor, and runs the LM head and the logits
#: processor eagerly outside the graph
#: (``prefill_cuda_graph_runner.py:1103-1130``). Everything outside this tuple
#: is refused by name rather than mapped -- see :func:`_refuse_unbuffered_lpo`.
_LPO_TOKEN_DIM_FIELDS = (
    "next_token_logits",
    "hidden_states",
    "cross_aux_hidden_states",
    "full_logits",
    "mm_input_embeds",
)


def _refuse_unbuffered_lpo(output: LogitsProcessorOutput) -> None:
    """Refuse a captured output carrying a field this layer cannot replay.

    Deliberately a refusal and not a pass-through. A replay buffer holds
    whatever the CAPTURE wrote; a field that the graph does not rewrite on
    every replay would be served frozen at capture-time content, and for the
    host-side fields on this dataclass -- the sampler's ``next_token_*``
    logprob lists, the prefill-only ``input_*`` logprob lists,
    ``customized_info`` -- that is stale data with the right type and no
    exception. This decode path produces none of them (parts 2 and 3 of
    ``LogitsProcessorOutput`` are filled by ``Sampler`` and by the
    extend-with-logprob branch, both outside a captured decode segment), so
    the refusal is unreachable on the routes that exist today and exists to
    keep the next one honest.
    """
    unsupported = [
        field.name
        for field in dataclasses.fields(output)
        if field.name not in _LPO_TOKEN_DIM_FIELDS
        and getattr(output, field.name, None) is not None
    ]
    if unsupported:
        raise TypeError(
            "BCG cannot buffer LogitsProcessorOutput fields "
            f"{sorted(unsupported)}: this layer replays a per-token tensor "
            "buffer, and these are either host-side objects or not indexed by "
            "the token count, so a replay would serve capture-time values "
            "with no error. Add them to _LPO_TOKEN_DIM_FIELDS only with the "
            "consumer that reads them and its leading dimension named."
        )


def _lpo_tensors(output: LogitsProcessorOutput) -> Dict[str, torch.Tensor]:
    """The present, buffered fields of ``output``, after refusing the rest."""
    _refuse_unbuffered_lpo(output)
    present: Dict[str, torch.Tensor] = {}
    for name in _LPO_TOKEN_DIM_FIELDS:
        value = getattr(output, name, None)
        if value is None:
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                f"BCG expected LogitsProcessorOutput.{name} to be a tensor, "
                f"got {type(value)}"
            )
        present[name] = value
    return present


def _lpo_rows(tensors: Dict[str, torch.Tensor]) -> Optional[int]:
    """The single leading-dim row count shared by ``tensors``.

    Refuses a mixture instead of picking one. The whole hazard this branch was
    held back for is that the fields might not agree: ``_slice_output`` takes
    one row count, so a disagreement silently truncates or mis-attributes rows
    of a spec-path logits tensor without raising anywhere.
    """
    rows = {name: int(t.shape[0]) for name, t in tensors.items()}
    if not rows:
        return None
    distinct = set(rows.values())
    if len(distinct) > 1:
        raise ValueError(
            "BCG cannot buffer a LogitsProcessorOutput whose fields disagree "
            f"on their leading dimension: {rows}. On the decode path they are "
            "all per-token (decode_cuda_graph_runner.py:2112-2144 slices them "
            "identically); a mixture means this output came from a mode this "
            "buffer layer has no mapping for, and guessing one would return "
            "correctly-shaped wrong rows."
        )
    return distinct.pop()


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        # Draft-solo placement: solo-host draft graphs capture rank-locally
        # (shadows never enter the matching barrier — see
        # full_cuda_graph_backend for the deadlock this avoids).
        self._skip_warmup_barrier = getattr(
            cuda_graph_runner.model_runner, "spec_solo_rank_local_graphs", False
        )
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        #: Leading rows the shared buffer holds; see the row-budget note in
        #: capture_one for why this is not simply ``shape_key.size``.
        self._buffer_rows: int = 0
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self._buffer_rows = 0
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # Same cold-build window as the full and piecewise backends: several
        # jit_kernel modules build on FIRST CALL, so on an empty cache these
        # forwards can sit minutes in nvcc while a peer rank already waits in a
        # deadline-bearing device collective. The capture below stays outside
        # the window.
        warmup_out = run_capture_warmups(
            forward_fn,
            repeats=2,
            device_module=self._device_module,
            tp_group=self._tp_group,
            skip_barrier=self._skip_warmup_barrier,
            post_warmup_hook=post_warmup_hook,
            reason="breakable cuda-graph capture warmup",
        )

        graph = BreakableCUDAGraph()
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        size = shape_key.size
        if self._shared_output_buffer is None:
            # Row budget: ``shape_key.size`` is the graph key, and for a decode
            # runner that is the BATCH size -- while the body's output is
            # indexed by TOKENS, i.e. ``bs * num_tokens_per_bs`` under a
            # non-ragged speculative verify (`_capture_graph_size`,
            # decode_cuda_graph_runner.py:682). Sizing the shared buffer from
            # the key alone would hold one row per sequence for a per-token
            # output and truncate every draft position but the first, with no
            # error. Take the body's own leading dimension where it is larger;
            # captures run largest-first (`reversed(self.capture_bs)`), so the
            # first one sets the high-water mark for all of them. Identical to
            # the previous behaviour whenever rows == size, which is every
            # plain-decode capture.
            self._buffer_rows = max(size, self._max_leading_rows(warmup_out) or size)
            self._shared_output_buffer = self._alloc_full_buffer(
                warmup_out, self._buffer_rows
            )
        with BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=self._pool,
            stream=self._capture_stream,
        ):
            out = captured_fn()
            out_rows = self._output_rows(out, self._buffer_rows)
            produced = self._max_leading_rows(out)
            if produced is not None and produced > self._buffer_rows:
                raise ValueError(
                    "BCG shared output buffer holds "
                    f"{self._buffer_rows} rows but the capture at "
                    f"{shape_key} produced {produced}. Capturing "
                    "largest-first is what keeps this buffer big enough; a "
                    "larger shape arriving later would be silently truncated."
                )
            self._copy_output_to_buffer(out, self._shared_output_buffer, out_rows)

        stored = self._slice_output(self._shared_output_buffer, out_rows)
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored

    def _max_leading_rows(self, output: Any) -> Optional[int]:
        """Largest leading-dim row count anywhere in ``output``; ``None`` if it
        holds no tensor (the weightless-worker sentinel, an empty container)."""
        if torch.is_tensor(output):
            return int(output.shape[0])
        if isinstance(output, LogitsProcessorOutput):
            return _lpo_rows(_lpo_tensors(output))
        if isinstance(output, PPProxyTensors):
            rows = [int(t.shape[0]) for t in output.tensors.values()]
            return max(rows) if rows else None
        if isinstance(output, (list, tuple)):
            rows = [
                r
                for r in (self._max_leading_rows(item) for item in output)
                if r is not None
            ]
            return max(rows) if rows else None
        return None

    def _output_rows(self, output: Any, cap: int) -> int:
        """Leading-dim row count actually produced by the body, clamped to ``cap``.

        A body that shards or prunes its output along dim 0 returns fewer than
        ``cap`` rows; everything else returns exactly ``cap``.
        """
        if torch.is_tensor(output):
            return min(cap, output.shape[0])
        if isinstance(output, LogitsProcessorOutput):
            rows = _lpo_rows(_lpo_tensors(output))
            return cap if rows is None else min(cap, rows)
        if isinstance(output, PPProxyTensors):
            rows = [t.shape[0] for t in output.tensors.values()]
            return min([cap, *rows])
        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        """A same-structure buffer as ``output`` but with ``size`` leading rows."""
        if output is None:
            return None
        if torch.is_tensor(output):
            return output.new_empty((size, *output.shape[1:]))
        if isinstance(output, LogitsProcessorOutput):
            # Field-for-field, and ONLY the fields present at capture: an
            # absent field stays absent so the structure check in
            # _copy_output_to_buffer can see a body that changed its mind
            # between capture sizes instead of writing into a buffer nobody
            # filled.
            tensors = _lpo_tensors(output)
            buffers = {
                name: t.new_empty((size, *t.shape[1:])) for name, t in tensors.items()
            }
            buffers.setdefault("next_token_logits", None)
            return LogitsProcessorOutput(**buffers)
        if isinstance(output, PPProxyTensors):
            return PPProxyTensors(
                {
                    key: t.new_empty((size, *t.shape[1:]))
                    for key, t in output.tensors.items()
                }
            )
        if isinstance(output, tuple):
            return tuple(self._alloc_full_buffer(o, size) for o in output)
        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, LogitsProcessorOutput):
            sliced = {
                name: t[:num_tokens] for name, t in _lpo_tensors(output).items()
            }
            sliced.setdefault("next_token_logits", None)
            return LogitsProcessorOutput(**sliced)
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, LogitsProcessorOutput) and isinstance(
            output_buffer, LogitsProcessorOutput
        ):
            tensors = _lpo_tensors(output)
            buffers = _lpo_tensors(output_buffer)
            if tensors.keys() != buffers.keys():
                # A body that produces hidden_states at one capture size and
                # not at another would otherwise leave a buffer field holding
                # the previous size's rows, replayed forever.
                raise ValueError(
                    "BCG logits-output structure changed between capture "
                    f"sizes: {sorted(tensors)} != {sorted(buffers)}"
                )
            for name, tensor in tensors.items():
                self._copy_output_to_buffer(tensor, buffers[name], num_tokens)
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, (list, tuple)) and isinstance(
            output_buffer, type(output)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        # #431: same reason as in FullCudaGraphBackend.replay -- a replayed
        # graph runs the barlink BAR1 spin kernels with no host code between
        # them, so this boundary is the next place their status word can be
        # read at all.
        barlink_abort_gate.check_after_graph_replay()
        # Same boundary, same reason (#443): the capturable MoE expert-offload
        # counts an unreachable cold row on device because testing for it in
        # the step would be a host read inside the capture. Empty registry
        # unless a #394 shared cold tier is live.
        offload_capture_gate.check_after_graph_replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
        self._shared_output_buffer = None
        self._buffer_rows = 0
