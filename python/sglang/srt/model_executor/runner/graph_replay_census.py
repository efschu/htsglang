"""23.09. (fnFL2x46): D's verify graph dies on its FIRST replay for a new
request after that request's extend -- TP0 'illegal memory access' 4325.7 ms
(x44), 4316.4 ms (x45) and 4352.9 ms (x46) into the replay -- and survives
every replay once one eager verify round ran in between (x46: the control
request's round 1 eager, rounds 2-4 graphed, CODE MATCH). Whatever the eager
round sets right is an INPUT the replay reads: a static buffer, attention
metadata, or the MoE expert pool tables.

``SGLANG_GRAPH_REPLAY_CENSUS=N`` logs, for the first N TARGET_VERIFY replays
whose batch context reaches ``MIN_CONTEXT`` tokens, one line per tensor of
those inputs -- shape, dtype, min, max -- taken after ``load_batch`` and
before the replay, so a dying replay still writes its census. A surviving and
a dying replay of the same boot become comparable line by line. Tensors above
``MAX_NUMEL`` elements (KV pools) are named, not reduced. Off by default;
never raises.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Set, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "SGLANG_GRAPH_REPLAY_CENSUS"
MIN_CONTEXT = 64
MAX_NUMEL = 1 << 24
#: Nested objects walked below a backend's forward metadata.
MAX_DEPTH = 2
_STATE = {"budget": None, "n": 0}
_SKIP_TYPES = ("Pool", "Allocator", "Runner", "Worker", "Config")
_POOL_FIELDS = ("hot_phys", "host_row", "row_key", "row_use", "error", "staging_rows")


def _budget() -> int:
    if _STATE["budget"] is None:
        try:
            _STATE["budget"] = max(0, int(os.environ.get(ENV, "0") or 0))
        except ValueError:
            _STATE["budget"] = 0
    return _STATE["budget"]


def _range(t: torch.Tensor) -> str:
    if t.numel() == 0:
        return "empty"
    if t.numel() > MAX_NUMEL:
        return "large"
    if t.dtype == torch.bool:
        t = t.to(torch.int32)
    if t.is_complex():
        return "complex"
    return f"{t.min().item()}..{t.max().item()}"


def _fields(obj: Any) -> List[Tuple[str, Any]]:
    """Attribute pairs of a plain object, a msgspec Struct or a __slots__
    class. x47: the QSA metadata has no __dict__ and was skipped silently."""
    if hasattr(obj, "__dict__"):
        return list(vars(obj).items())
    names = getattr(type(obj), "__struct_fields__", None) or getattr(
        type(obj), "__slots__", None
    )
    if isinstance(names, str):
        names = (names,)
    return [(n, getattr(obj, n, None)) for n in (names or ())]


def _walk(prefix: str, obj: Any, depth: int, out: List[Tuple[str, torch.Tensor]],
          seen: Set[int]) -> None:
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    for key, value in _fields(obj):
        name = f"{prefix}.{key}"
        if isinstance(value, torch.Tensor):
            out.append((name, value))
        elif (
            depth > 0
            and value is not None
            and not isinstance(value, (torch.nn.Module, str, bytes, int, float, bool))
            and not any(s in type(value).__name__ for s in _SKIP_TYPES)
            and _fields(value)
        ):
            _walk(name, value, depth - 1, out, seen)


def _backend_inputs(attn_backend) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    seen: Set[int] = set()
    for label, backend in (
        ("attn", attn_backend),
        ("attn.full", getattr(attn_backend, "full_attn_backend", None)),
        ("attn.linear", getattr(attn_backend, "linear_attn_backend", None)),
    ):
        metadata = getattr(backend, "forward_metadata", None)
        _walk(f"{label}.forward_metadata", metadata, MAX_DEPTH, out, seen)
    return out


def _pool_lines(model) -> List[str]:
    """The expert pool tables over all MoE layers: per field the range across
    layers, plus the layers whose sticky error word is set."""
    ranges = {f: [] for f in _POOL_FIELDS}
    error_layers = []
    layers = 0
    for module in model.modules():
        cache = getattr(module, "_expert_offload", None)
        tables = getattr(cache, "_pool_tables", None)
        if tables is None:
            continue
        layers += 1
        for field in _POOL_FIELDS:
            t = getattr(tables, field, None)
            if isinstance(t, torch.Tensor) and t.numel():
                ranges[field].append((int(t.min().item()), int(t.max().item())))
        if int(tables.error[0]) != 0:
            error_layers.append(getattr(getattr(cache, "layer", None), "layer_id", "?"))
    lines = [f"pool layers={layers} error_layers={error_layers[:8]}"]
    for field, rs in ranges.items():
        if rs:
            lines.append(
                f"pool.{field} min={min(r[0] for r in rs)} max={max(r[1] for r in rs)}"
            )
    return lines


def maybe_census(runner, forward_batch) -> None:
    """One census of the replay's inputs, while the budget lasts."""
    try:
        budget = _budget()
        if budget <= 0 or _STATE["n"] >= budget:
            return
        if not forward_batch.forward_mode.is_target_verify():
            return
        context = int(forward_batch.seq_lens.max().item())
        if context < MIN_CONTEXT:
            return
        _STATE["n"] += 1
        n = _STATE["n"]
        head = f"GRAPH-REPLAY-CENSUS n={n} ctx={context} bs={forward_batch.batch_size}"
        tensors: List[Tuple[str, torch.Tensor]] = []
        _walk("buffers", runner.buffers, 0, tensors, set())
        tensors.extend(_backend_inputs(runner.attn_backend))
        for name, t in tensors:
            logger.info(
                "%s %s shape=%s dtype=%s range=%s",
                head, name, tuple(t.shape), str(t.dtype).replace("torch.", ""),
                _range(t),
            )
        for line in _pool_lines(runner.model_runner.model):
            logger.info("%s %s", head, line)
        logger.info("%s done tensors=%d", head, len(tensors))
    except Exception as exc:  # noqa: BLE001 - instrument must not raise
        logger.warning("GRAPH-REPLAY-CENSUS failed: %s: %s", type(exc).__name__, exc)
