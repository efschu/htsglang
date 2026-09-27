"""WEG2-GRAPH-UPLOAD: pay a CUDA graph's first-launch upload at capture, not in the first replay under load.

What the 27B b1 boot showed (27.09.2026, dkr27breleasedraftbar1w109270652, tree 8f0bf40c2f, D TP0 on the
RTX 5090). D died at 06:57:17 with "CUDA error: out of memory" in ``torch.cuda.CUDAGraph.replay`` of the
full decode graph (``decode_cuda_graph_runner.execute`` -> ``full_cuda_graph_backend.replay``), forward
mode TARGET_VERIFY, bs=3 -- the FIRST bs=3 replay of that D process (it had run only bs1 and bs2). The
card had 5 MiB free (WEG2-VRAM-PEAK card_free_mib=5) while the torch allocator held ~1.5 GiB of free
cache (#1028c alloc_cache), so the failing allocation was NOT a torch allocation (those retry from the
cache; alloc ooms=0): it was the driver's. torch (2.11) instantiates with ``cudaGraphInstantiateWithFlags``
and never calls ``cudaGraphUpload`` (the symbol is absent from libtorch_cuda.so), and ``capture_one``
never replays -- so every shape uploads its executable graph at its FIRST ``replay()``, whenever that
happens. The rc11a agent-load boot ran its first bs=3 at 617 MiB free and survived every later bs=3 at
>= 71 MiB.

The fix: right after a shape is captured -- at boot, with the card still roomy -- upload the graph exec
explicitly (driver API ``cuGraphUpload(hGraphExec, hStream)``, libcuda.so.1, the one driver every
runtime copy shares) and print what it cost:

    WEG2-GRAPH-UPLOAD shape=<key> mib=<card free before - after> ms=<..> rc=0

``mib`` is the driver-side cost the replay would otherwise have paid at run time (measured with
``torch.cuda.mem_get_info`` around a synchronized upload). A failure is a line, never an exception: the
graph then uploads lazily at its first replay, exactly as before.

Switch ``SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE`` (environ.py, code default off; the weg2 launcher turns it
on for group D in ``build_env``, an explicit value wins). Model-neutral: every profile's D uses the full
graph backend.

OPEN (to read on the metal): whether the uploaded exec survives a weg2 sleep (the saver pauses the
cuda_graph TAG's pool, the exec's driver resources are not in that pool). The first replay per shape
after the first wake shows it: no OOM and no card_free drop at that replay.
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

MARKER = "WEG2-GRAPH-UPLOAD"
MIB = float(1 << 20)

_LIB: dict = {"lib": None, "tried": False}


def enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE.get())
    except Exception:  # noqa: BLE001 -- a guard never kills a capture
        return False


def _libcuda():
    """libcuda.so.1 (the driver), loaded once; None where it is absent."""
    if not _LIB["tried"]:
        _LIB["tried"] = True
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                fn = lib.cuGraphUpload
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                fn.restype = ctypes.c_int
                _LIB["lib"] = lib
                break
            except (OSError, AttributeError):
                continue
    return _LIB["lib"]


def reset_for_tests() -> None:
    _LIB["lib"] = None
    _LIB["tried"] = False


def _exec_handle(graph: Any) -> Optional[int]:
    """The cudaGraphExec_t (== CUgraphExec) of a torch CUDAGraph, or None
    (older torch without ``raw_cuda_graph_exec``, or nothing instantiated)."""
    raw = getattr(graph, "raw_cuda_graph_exec", None)
    if raw is None:
        return None
    h = raw()
    h = int(h) if h is not None else 0
    return h or None


def upload_after_capture(
    graph: Any,
    shape_key: Any,
    stream: Any = None,
    *,
    lib: Any = None,
    cuda: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Optional[Tuple[int, float]]:
    """Upload one captured graph; returns ``(rc, mib)`` or ``None`` when it did
    nothing. Never raises. ``lib``/``cuda`` are injectable for tests
    (``cuda`` defaults to ``torch.cuda``)."""
    try:
        if cuda is None:
            import torch

            cuda = torch.cuda
        try:
            if cuda.is_current_stream_capturing():
                return None  # never inside a capture
        except Exception:  # noqa: BLE001
            pass
        h = _exec_handle(graph)
        if h is None:
            logger.info("%s shape=%s skipped: no graph exec handle (torch without raw_cuda_graph_exec?)",
                        MARKER, shape_key)
            return None
        lib = lib if lib is not None else _libcuda()
        if lib is None:
            logger.info("%s shape=%s skipped: libcuda.so.1 not loadable", MARKER, shape_key)
            return None
        if stream is None:
            stream = cuda.current_stream()
        s = int(getattr(stream, "cuda_stream", 0) or 0)
        cuda.synchronize()
        free0, _ = cuda.mem_get_info()
        t = clock()
        rc = int(lib.cuGraphUpload(ctypes.c_void_p(h), ctypes.c_void_p(s)))
        cuda.synchronize()
        ms = (clock() - t) * 1000.0
        free1, _ = cuda.mem_get_info()
        mib = (float(free0) - float(free1)) / MIB
        if rc == 0:
            logger.info("%s shape=%s mib=%.1f ms=%.1f rc=0 card_free_after=%.0f", MARKER, shape_key, mib, ms,
                        free1 / MIB)
        else:
            logger.warning("%s shape=%s FAILED rc=%d mib=%.1f card_free=%.0f -- the graph uploads lazily at its "
                           "first replay (the pre-fix behaviour)", MARKER, shape_key, rc, mib, free1 / MIB)
        return rc, mib
    except Exception as exc:  # noqa: BLE001 -- a guard never kills a capture
        logger.warning("%s shape=%s skipped: %s: %s", MARKER, shape_key, type(exc).__name__, exc)
        return None
