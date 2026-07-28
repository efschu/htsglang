"""CUDA graph capture context manager + forward-context propagation.

Owns two pieces of cross-cutting state used by *every* piecewise-style
backend (currently breakable + tc_piecewise):

* _in_tc_piecewise_cuda_graph — a process-global flag set true while we
  are inside the capture or replay window of a piecewise CUDA graph.
  Read by model code that needs to take the static-buffer / fixed-shape
  branch. See refactor/plan.md §6.5 for the full semantics.
* TcPiecewiseForwardContext — a dataclass propagated across attention/MoE
  layers during capture and replay so that submodules can reach the
  current ForwardBatch and per-layer metadata without threading
  arguments through every call site. Named TcPiecewise… (matches
  Backend.TC_PIECEWISE + enable_tc_piecewise_cuda_graph) to
  disambiguate from the per-forward-call
  sglang.srt.model_executor.forward_context.ForwardContext.

This module deliberately does **not** own torch.compile-specific state
(warmup flag, capture stream); those live in compilation/compile_phase.py.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

_in_tc_piecewise_cuda_graph: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tc_piecewise.in_cuda_graph", default=False
)


def is_in_tc_piecewise_cuda_graph() -> bool:
    """True while inside tc_piecewise CUDA graph capture/replay."""
    return _in_tc_piecewise_cuda_graph.get()


@contextmanager
def enable_tc_piecewise_cuda_graph():
    """Mark the enclosed scope as inside a tc_piecewise CUDA graph
    capture/replay. Any exception raised inside is logged with the
    PCG-specific failure hint, then re-raised for the caller to handle.
    """
    token = _in_tc_piecewise_cuda_graph.set(True)
    try:
        yield
    except Exception as exc:
        msg = PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG.format(
            backend=Backend.TC_PIECEWISE, suggestions=TCPCG_FAILURE_HINT
        )
        logger.error(f"{type(exc).__name__}: {exc}\n{msg}")
        raise
    finally:
        _in_tc_piecewise_cuda_graph.reset(token)


@dataclass
class TcPiecewiseForwardContext:
    forward_batch: Optional[ForwardBatch] = None
    attention_layers: Optional[List[Any]] = field(default=None)
    quant_config: Any = None
    moe_layers: Optional[List[Any]] = field(default=None)
    moe_fusions: Optional[List[Any]] = field(default=None)
    dsa_indexers: Optional[List[Any]] = field(default=None)
    num_tokens: Optional[int] = None
    raw_num_tokens: Optional[int] = None


# Context-local, for the same reason as model_executor.forward_context: with
# the multi-group runtime's concurrent lane (#274 slice C) two forwards are in
# flight in one process, and the split-op boundary resolves its forward_batch
# and attention_layers through this slot. A plain global hands the serving
# group the lane's layers.
_tc_piecewise_forward_context: contextvars.ContextVar[
    Optional[TcPiecewiseForwardContext]
] = contextvars.ContextVar("tc_piecewise.forward_context", default=None)


def get_tc_piecewise_forward_context() -> Optional[TcPiecewiseForwardContext]:
    return _tc_piecewise_forward_context.get()


@contextmanager
def set_tc_piecewise_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
    moe_fusions: List[Any],
    dsa_indexers: Optional[List[Any]] = None,
    num_tokens: Optional[int] = None,
    raw_num_tokens: Optional[int] = None,
):
    token = _tc_piecewise_forward_context.set(
        TcPiecewiseForwardContext(
            forward_batch=forward_batch,
            attention_layers=attention_layers,
            quant_config=quant_config,
            moe_layers=moe_layers,
            moe_fusions=moe_fusions,
            dsa_indexers=dsa_indexers,
            num_tokens=num_tokens,
            raw_num_tokens=raw_num_tokens,
        )
    )
    try:
        yield
    finally:
        _tc_piecewise_forward_context.reset(token)


TCPCG_FAILURE_HINT = (
    "1. change to breakable by --cuda-graph-backend-prefill=breakable\n"
    "2. disable the prefill CUDA graph by --cuda-graph-backend-prefill=disabled\n"
    "3. if it is an OOM problem, set --mem-fraction-static to a smaller value "
    "(e.g., 0.8 or 0.7) or set --cuda-graph-max-bs-prefill to a smaller value "
    "(e.g., 2048)\n"
)
