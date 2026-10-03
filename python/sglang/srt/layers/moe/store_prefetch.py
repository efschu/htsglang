"""BOOTZEIT 5c (29.09.): the next layer's expert-store files open while this
layer's shards are still being consumed.

z30w-park, [ct-stream-presplit] summed per rank: ``store_open`` 5.17 s on PP0
(29 layers x 4 files, 23.78 GiB pinned) and 2.90 s on D TP0 (48 x 4). The time
is ``open_store`` -> ``shared_pinned_empty``: ftruncate + mmap of the tmpfs
file, then ``cudaHostRegister``, which faults in (P: allocates and zeroes) and
pins every page. It runs on the loader thread inside the presplit, where the
consumers are idle and nothing else moves.

Every MoE layer of a rank has the SAME store geometry: the same four expert
tensors with the same row shapes and dtypes, and the same slot count (under
the Karte it is ``karte["slots"]``, one number for the model). So when layer N
has opened its files, the files of layer N+1 are known -- path
``store_path(dir, "L<N+1>", attr)``, shape ``(slots, *row)`` -- and a single
background thread opens them while the loader thread feeds layer N+1's shards
to the consumers. ``cudaHostRegister`` releases the GIL (torch's cudart
binding), so the pinning runs beside the load instead of in its critical path.

What makes it safe:

* Only a layer this process ARMED for the presplit is prefetched
  (``note_armed`` from the compressed-tensors WNA16 create_weights) -- never a
  layer of another pipeline stage.
* ``open_store`` takes a prefetched file only if path, shape and dtype are the
  ones it was asked for; otherwise it drops the mapping and -- if the prefetch
  created the file -- unlinks it, so the real open creates it at the right
  size. A failed prefetch is logged and the real open runs as before.
* One layer of look-ahead, started after layer N's presplit freed its host
  stack: the tmpfs pages of N+1 exist a little earlier, while N+1's host stack
  fills, but never beyond what the baseline holds at N+1's own presplit (that
  stack plus that store, both alive until the presplit ends) -- the host peak
  does not move.
* The worker thread binds the loader's CUDA device, so the registration uses
  the same primary context as before (never device 0 by accident).

``SGLANG_OPT_LOAD_STORE_PREFETCH`` (default off until the first metal series).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_ARMED: set = set()  # layer ids armed for the presplit in this process
_READY: Dict[str, Future] = {}  # store path -> Future[(tensor, created)]
_EXEC: Optional[ThreadPoolExecutor] = None
_EXEC_DEVICE: Optional[int] = None
#: census for the load's summary line: prefetched / taken / dropped / failed,
#: and the seconds the loader thread still waited on a running prefetch
STATS = {"prefetched": 0, "taken": 0, "dropped": 0, "failed": 0, "wait_s": 0.0,
         "bg_open_s": 0.0}


def enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OPT_LOAD_STORE_PREFETCH.get())


def note_armed(layer_id) -> None:
    """create_weights armed this layer's presplit (this process loads it)."""
    if layer_id is None:
        return
    with _LOCK:
        _ARMED.add(int(layer_id))


def _executor(device_index: Optional[int]) -> ThreadPoolExecutor:
    global _EXEC, _EXEC_DEVICE
    with _LOCK:
        if _EXEC is None:
            def _bind():
                if device_index is not None and torch.cuda.is_available():
                    torch.cuda.set_device(int(device_index))

            _EXEC = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="store-prefetch", initializer=_bind
            )
            _EXEC_DEVICE = device_index
        return _EXEC


def _open(directory, layer_key, attr, slots, row_shape, dtype):
    from sglang.srt.layers.moe import expert_store as _es

    t0 = time.perf_counter()
    out = _es.open_store_uncached(
        directory, layer_key, attr, slots, tuple(row_shape), dtype, num_slots=slots
    )
    with _LOCK:
        STATS["bg_open_s"] += time.perf_counter() - t0
    return out


def prefetch_next(
    layer_id,
    *,
    directory: str,
    slots: int,
    geometry: Sequence[Tuple[str, Tuple[int, ...], torch.dtype]],
    device_index: Optional[int],
) -> int:
    """Open layer ``layer_id + 1``'s store files in the background with this
    layer's geometry. Returns how many opens were queued (0 = nothing to do)."""
    if not enabled() or layer_id is None or not geometry:
        return 0
    from sglang.srt.layers.moe import expert_store as _es

    nxt = int(layer_id) + 1
    with _LOCK:
        if nxt not in _ARMED:
            return 0
    key = f"L{nxt}"
    ex = _executor(device_index)
    n = 0
    for attr, row_shape, dtype in geometry:
        path = _es.store_path(directory, key, attr)
        with _LOCK:
            if path in _READY:
                continue
            _READY[path] = ex.submit(_open, directory, key, attr, int(slots), row_shape, dtype)
            STATS["prefetched"] += 1
        n += 1
    return n


def take(path: str, shape: Tuple[int, ...], dtype: torch.dtype):
    """The prefetched ``(tensor, created)`` for exactly this path/shape/dtype,
    or None (then the caller opens as before)."""
    with _LOCK:
        fut = _READY.pop(path, None)
    if fut is None:
        return None
    t0 = time.perf_counter()
    try:
        tensor, created = fut.result()
    except Exception as e:  # noqa: BLE001 -- the real open runs and raises if it must
        with _LOCK:
            STATS["failed"] += 1
            STATS["wait_s"] += time.perf_counter() - t0
        logger.warning("BOOTZEIT5c STORE-PREFETCH %s failed (%s: %s); opening in place",
                       path, type(e).__name__, e)
        return None
    with _LOCK:
        STATS["wait_s"] += time.perf_counter() - t0
    if tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype:
        with _LOCK:
            STATS["taken"] += 1
        return tensor, created
    # Not the file the presplit wants: drop the mapping, and a file the
    # prefetch itself created at the wrong size goes, so the real open
    # creates it right (an existing file keeps its size -- as before).
    del tensor
    if created:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    with _LOCK:
        STATS["dropped"] += 1
    logger.warning("BOOTZEIT5c STORE-PREFETCH %s dropped: prefetched geometry is not "
                   "the one asked for (%s %s); opening in place", path, tuple(shape), dtype)
    return None


def drain(what: str = "load") -> None:
    """End of a load: wait for what is still running, drop what nobody took,
    print the census. The worker thread ends with it."""
    global _EXEC
    with _LOCK:
        pending = list(_READY.items())
        _READY.clear()
        ex, _EXEC = _EXEC, None
        armed = len(_ARMED)
        _ARMED.clear()
    for _path, fut in pending:
        try:
            fut.result()
        except Exception:  # noqa: BLE001 -- nothing depends on an untaken prefetch
            pass
    if ex is not None:
        ex.shutdown(wait=True)
    if STATS["prefetched"] or pending:
        logger.info(
            "BOOTZEIT5c STORE-PREFETCH %s: armed=%d prefetched=%d taken=%d dropped=%d "
            "failed=%d untaken=%d | background open %.2f s, loader waited %.2f s",
            what, armed, STATS["prefetched"], STATS["taken"], STATS["dropped"],
            STATS["failed"], len(pending), STATS["bg_open_s"], STATS["wait_s"],
        )
    for k in STATS:
        STATS[k] = 0.0 if isinstance(STATS[k], float) else 0


def geometry_of(presplit_attrs: List[Tuple[str, torch.Tensor]]):
    """(attr, row_shape, dtype) of the tensors a presplit opened stores for."""
    return [(a, tuple(t.shape[1:]), t.dtype) for a, t in presplit_attrs]
