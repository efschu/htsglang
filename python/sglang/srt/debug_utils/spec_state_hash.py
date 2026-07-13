"""Persistent-state hash dumps for the speculative-decoding path.

#50 campaign, round 7: with NEXTN the eager pipeline is fully deterministic
PER RUN INDEX (run1 always produces output A, run2 always output B, bit-exact
across boots), i.e. some process-persistent state outside the poisoned pools
is mutated by every request and feeds the next request's math. Under CUDA
graphs the same evolution lives in static replay buffers and converges to a
degenerate attractor.

This module is the falsifier/locator for that class: after every finished
request it walks all objects reachable from the target/draft workers and
logs one line per reachable tensor (sha256 of the raw bytes) and per plain
int/float/bool attribute. Diffing the dump of request N against request N+1
(same prompt) lists exactly the state that evolved.

Usage (server): SGLANG_SPEC_STATE_HASH=1, then grep the scheduler log for
"SPEC_STATE_HASH". Lines are sorted by path within a dump and framed by
BEGIN/END markers carrying the dump tag, so:

    grep 'SPEC_STATE_HASH .*tag=req_end_1 ' log > run1.txt
    grep 'SPEC_STATE_HASH .*tag=req_end_2 ' log > run2.txt
    diff run1.txt <(sed 's/tag=req_end_2/tag=req_end_1/' run2.txt)

Cost: one full walk hashes every pool byte; on a loaded rank expect tens of
seconds. SGLANG_SPEC_STATE_HASH_MAX_MB=N switches tensors above N MiB to a
strided-sample fingerprint (still change-sensitive, much faster).

Deliberate exclusions:
* nn.Parameter tensors (model weights — immutable in inference, huge); the
  END line reports how many were skipped.
* Objects whose type is defined outside sglang/torch.nn (tokenizers, HF
  configs, ZMQ sockets, ...) are not traversed into.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import time
from typing import Any, Iterable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Only objects whose type lives in these module prefixes are traversed into.
# torch.nn covers container modules (ModuleList/Sequential) between sglang
# layers; everything else torch-internal stays a leaf. flashinfer/sgl_kernel
# wrapper objects are included because their internal plan/workspace buffers
# are persistent, partially rewritten per plan() and read by every kernel
# run — exactly the state class this locator exists for. (Round 8 coverage
# hole: the attention wrappers were skipped as foreign modules, so their
# plan/workspace state was invisible to the dump.)
DEFAULT_TRAVERSE_PREFIXES = ("sglang.", "torch.nn.", "flashinfer", "sgl_kernel")

_MAX_DEPTH = 20
_MAX_ENTRIES = 500_000
_CHUNK_ELEMS = 1 << 24  # elements per device->host hashing chunk
_SAMPLE_ELEMS = 1 << 20  # elements hashed for over-limit tensors


def _tensor_bytes_view(t: torch.Tensor) -> torch.Tensor:
    """Contiguous 1-D uint8 view of a tensor's storage bytes (bit-exact)."""
    flat = t.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    try:
        return flat.view(torch.uint8)
    except (RuntimeError, TypeError):
        # e.g. torch.bool on some builds: same itemsize, safe to reinterpret
        # via an explicit cast (deterministic either way).
        return flat.to(torch.uint8)


def hash_tensor(t: torch.Tensor, max_bytes: int = 0) -> str:
    """sha256 (first 16 hex chars) of a tensor's raw bytes.

    max_bytes > 0: tensors larger than that are fingerprinted from a strided
    sample of ~_SAMPLE_ELEMS elements plus the first and last chunk-aligned
    elements — deterministic and change-sensitive, but not exhaustive.
    """
    try:
        if t.is_meta:
            return "meta"
        n = t.numel()
        if n == 0:
            return "empty"
        h = hashlib.sha256()
        nbytes = n * t.element_size()
        if max_bytes > 0 and nbytes > max_bytes:
            flat = t.detach().reshape(-1)
            stride = max(1, n // _SAMPLE_ELEMS)
            sample = flat[::stride]
            h.update(_tensor_bytes_view(sample).cpu().numpy().tobytes())
            h.update(_tensor_bytes_view(flat[:64]).cpu().numpy().tobytes())
            h.update(_tensor_bytes_view(flat[-64:]).cpu().numpy().tobytes())
            return "~" + h.hexdigest()[:15]  # "~" marks a sampled fingerprint
        flat = t.detach().reshape(-1)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        for i in range(0, n, _CHUNK_ELEMS):
            piece = flat[i : i + _CHUNK_ELEMS]
            h.update(_tensor_bytes_view(piece).cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    except Exception as e:  # never break serving for a debug dump
        return f"ERR:{type(e).__name__}"


def _is_traversable(obj: Any, prefixes: Tuple[str, ...]) -> bool:
    if isinstance(obj, (list, tuple, dict)):
        return True
    mod = getattr(type(obj), "__module__", "") or ""
    return any(mod.startswith(p) for p in prefixes)


def _object_items(obj: Any) -> Iterable[Tuple[str, Any]]:
    """(child-path-suffix, child) pairs of one object, deterministic order."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"[{k!r}]", v
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield f"[{i}]", v
    else:
        d = getattr(obj, "__dict__", None)
        if d is not None:
            for k, v in d.items():
                yield f".{k}", v
        for slot in getattr(type(obj), "__slots__", ()) or ():
            if isinstance(slot, str) and hasattr(obj, slot):
                yield f".{slot}", getattr(obj, slot)


def collect_state_entries(
    roots: dict,
    traverse_prefixes: Tuple[str, ...] = DEFAULT_TRAVERSE_PREFIXES,
    max_bytes: int = 0,
) -> Tuple[List[str], int, int]:
    """Walk ``roots`` and return (sorted entry lines, n_tensors, n_params_skipped).

    Each line is "path=... kind=... ..." without the tag/rank framing.
    """
    entries: List[str] = []
    n_tensors = 0
    n_params_skipped = 0
    visited = set()
    stack: List[Tuple[str, Any, int]] = [
        (name, obj, 0) for name, obj in sorted(roots.items(), reverse=True)
    ]

    while stack and len(entries) < _MAX_ENTRIES:
        path, obj, depth = stack.pop()
        if obj is None or depth > _MAX_DEPTH:
            continue

        if isinstance(obj, torch.Tensor):
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            if isinstance(obj, torch.nn.Parameter):
                n_params_skipped += 1
                continue
            n_tensors += 1
            entries.append(
                f"path={path} kind=tensor shape={tuple(obj.shape)} "
                f"dtype={obj.dtype} dev={obj.device} "
                f"hash={hash_tensor(obj, max_bytes=max_bytes)}"
            )
            continue

        if isinstance(obj, (bool, int, float)):
            # bool before int: bool is an int subclass.
            entries.append(f"path={path} kind=scalar value={obj!r}")
            continue

        if isinstance(obj, (str, bytes, type)) or inspect.isroutine(obj):
            continue

        if not _is_traversable(obj, traverse_prefixes):
            continue
        if id(obj) in visited:
            continue
        visited.add(id(obj))

        children = list(_object_items(obj))
        # LIFO stack: push reversed to keep a stable, declaration-order walk.
        for suffix, child in reversed(children):
            stack.append((path + suffix, child, depth + 1))

    entries.sort()
    return entries, n_tensors, n_params_skipped


def dump_spec_state_hashes(scheduler, tag: str) -> None:
    """Log one SPEC_STATE_HASH line per persistent tensor/scalar of the
    target and draft workers. Never raises."""
    try:
        rank = getattr(scheduler, "tp_rank", -1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        roots = {}
        if getattr(scheduler, "draft_worker", None) is not None:
            roots["draft_worker"] = scheduler.draft_worker
        if getattr(scheduler, "tp_worker", None) is not None:
            roots["tp_worker"] = scheduler.tp_worker
        max_bytes = envs.SGLANG_SPEC_STATE_HASH_MAX_MB.get() * (1 << 20)
        t0 = time.perf_counter()
        entries, n_tensors, n_params = collect_state_entries(
            roots, max_bytes=max_bytes
        )
        elapsed = time.perf_counter() - t0
        logger.info("SPEC_STATE_HASH BEGIN tag=%s rank=%s", tag, rank)
        for line in entries:
            logger.info("SPEC_STATE_HASH tag=%s rank=%s %s", tag, rank, line)
        logger.info(
            "SPEC_STATE_HASH END tag=%s rank=%s entries=%d tensors=%d "
            "params_skipped=%d elapsed=%.1fs",
            tag,
            rank,
            len(entries),
            n_tensors,
            n_params,
            elapsed,
        )
    except Exception:
        logger.exception("SPEC_STATE_HASH dump failed (tag=%s)", tag)


_finished_request_count = 0


def maybe_dump_on_request_finish(scheduler, batch) -> None:
    """Hook: called at the end of Scheduler.process_batch_result. Emits one
    dump per batch that finished at least one request."""
    global _finished_request_count
    try:
        reqs = getattr(batch, "reqs", None) or []
        n_finished = sum(1 for req in reqs if req.finished())
    except Exception:
        return
    if n_finished == 0:
        return
    _finished_request_count += n_finished
    dump_spec_state_hashes(scheduler, tag=f"req_end_{_finished_request_count}")
