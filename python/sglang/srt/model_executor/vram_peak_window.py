"""WEG2-VRAM-PEAK: the allocator peak of EVERY P chunk, D round window and flip leg (fnFL2 H55).

What existed before and why it is not enough. ``[vram-peak]`` and
``WEG2-GRAPH-POOL`` read ``max_memory_allocated`` as the peak SINCE THE POOLS
(one reset at ``after pools``) and print only on a new high-water of
>= 0.25 GiB. So the planner's chunk transient (3697 MiB on PP0) is the worst
chunk of all boots it was measured on, and no line says which chunk, how the
chunks differ, or what ``reserved`` did. The per-forward tracker
(``forward_peak.py``) is off unless ``SGLANG_FORWARD_PEAK_PATH`` is set and
writes a JSON file at exit, not a timed line.

This module closes ONE window at a time and prints it:

    WEG2-VRAM-PEAK rank=0 phase=chunk rows=16384 n=1 t0_unix_ms=.. t_unix_ms=..
      window_ms=.. peak_allocated_mib=.. peak_reserved_mib=.. start_allocated_mib=..
      transient_mib=.. allocated_mib=.. reserved_mib=.. card_free_start_mib=..
      card_free_mib=.. card_total_mib=..

* ``phase=chunk`` -- every extend forward of the target runner (P: one line
  per chunk; D: its extend after a flip).
* ``phase=round`` -- decode/verify forwards, ``n`` of them per line
  (``SGLANG_WEG2_VRAM_PEAK_ROUNDS``, default 64, or 5 s, whichever first).
* ``phase=flip leg=release|resume`` -- one flip RPC on this rank, both groups.
  Before the leg the open window is closed as ``phase=round`` (if it holds
  rounds) or ``phase=idle``.

A window runs from the previous record to this one (``t0_unix_ms`` ..
``t_unix_ms``, wall clock in ms so the NVML probe ``weg2/tools/vram_hires.py``
can lay it on its 5-ms trace). ``transient_mib = peak_allocated -
start_allocated``: what the window added on top of what was allocated when it
opened (the planner's ``peak - allocated``).

COST. One ``memory_stats()`` read, one ``reset_peak_memory_stats()`` (host-side
counter re-base, no device sync) and one ``cudaMemGetInfo`` (driver call, no
sync) per record. Per decode round only a counter increment and a clock read.
The allocator peak is exact for torch allocations even though the forward is
asynchronous: the caching allocator books every block at launch time on the
host, which is the order the peak is taken over.

THE CUMULATIVE PEAK STAYS WHAT IT WAS. Re-basing the counter per window would
silently turn ``[vram-peak]``'s and ``WEG2-GRAPH-POOL``'s ``peak_mib`` ("since
the pools", read by ``planner.graph_pool_ledger`` and the KARTE) into "since
the last window". So every re-base first folds the counter into a shadow, and
those two readers take ``cum_peak_allocated()`` = max(shadow, counter). The
``after pools`` reset clears the shadow with the counter.

WHAT IT CANNOT SEE. Memory outside torch's caching allocator (CUDA context,
NCCL, TMS-unmapped tag regions, cuMemMap'd arenas) and anything that happens
between two driver reads. ``card_free_*`` are the two boundary reads of the
window, not its minimum; the minimum is the NVML probe's job.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MARKER = "WEG2-VRAM-PEAK"
MIB = float(1 << 20)
#: A round window closes after this many seconds even with fewer rounds.
ROUND_WINDOW_MAX_S = 5.0

_STATE: Dict[str, Any] = {
    "shadow_alloc": 0,       # folded allocator peak since the pools
    "shadow_reserved": 0,
    "t0": None,              # window start, unix seconds
    "start_alloc": None,     # allocated bytes at window start
    "free_start": None,      # card free bytes at window start
    "rounds": 0,
    "rank": None,
}


def _enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_VRAM_PEAK.get())
    except Exception:  # noqa: BLE001 -- an instrument never kills a forward
        return False


def _round_every() -> int:
    try:
        from sglang.srt.environ import envs

        return max(1, int(envs.SGLANG_WEG2_VRAM_PEAK_ROUNDS.get()))
    except Exception:  # noqa: BLE001
        return 64


def _peaks(cuda):
    """(peak_alloc, peak_reserved, alloc_now, reserved_now) in bytes, one
    ``memory_stats()`` read."""
    st = cuda.memory_stats()
    return (
        int(st.get("allocated_bytes.all.peak", 0)),
        int(st.get("reserved_bytes.all.peak", 0)),
        int(st.get("allocated_bytes.all.current", 0)),
        int(st.get("reserved_bytes.all.current", 0)),
    )


def cum_peak_allocated(cuda) -> int:
    """The allocator peak since the pools, in bytes, across every window
    re-base this module made. Replaces a bare ``max_memory_allocated()``."""
    return max(int(_STATE["shadow_alloc"]), int(cuda.max_memory_allocated()))


def reset_since_pools() -> None:
    """``after pools`` re-based the counter: the shadow starts over with it,
    and the first window opens now."""
    _STATE["shadow_alloc"] = 0
    _STATE["shadow_reserved"] = 0
    _STATE["t0"] = None
    _STATE["start_alloc"] = None
    _STATE["free_start"] = None
    _STATE["rounds"] = 0


def _close(cuda, rank: int, phase: str, extra: str = "", clock=time.time) -> Optional[str]:
    """Close the open window: read, fold, re-base, print. Returns the line."""
    try:
        pa, pr, a, r = _peaks(cuda)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s skipped: %s", MARKER, exc)
        return None
    _STATE["shadow_alloc"] = max(int(_STATE["shadow_alloc"]), pa)
    _STATE["shadow_reserved"] = max(int(_STATE["shadow_reserved"]), pr)
    try:
        cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001 -- then the next window is cumulative, and says nothing wrong
        logger.debug("%s reset failed: %s", MARKER, exc)
    try:
        free, total = cuda.mem_get_info()
        free, total = int(free), int(total)
    except Exception:  # noqa: BLE001
        free = total = -1
    now = clock()
    t0 = _STATE["t0"]
    start_alloc = _STATE["start_alloc"]
    free_start = _STATE["free_start"]
    n = int(_STATE["rounds"])
    _STATE["t0"] = now
    _STATE["start_alloc"] = a
    _STATE["free_start"] = free
    _STATE["rounds"] = 0

    def mib(x):
        return "na" if x is None or x < 0 else f"{x / MIB:.0f}"

    line = (
        f"{MARKER} rank={rank} phase={phase}{extra} n={n if phase == 'round' else 1 if phase == 'chunk' else 0} "
        f"t0_unix_ms={'na' if t0 is None else int(t0 * 1000)} t_unix_ms={int(now * 1000)} "
        f"window_ms={'na' if t0 is None else f'{(now - t0) * 1000:.0f}'} "
        f"peak_allocated_mib={mib(pa)} peak_reserved_mib={mib(pr)} "
        f"start_allocated_mib={mib(start_alloc)} "
        f"transient_mib={'na' if start_alloc is None else mib(pa - start_alloc)} "
        f"allocated_mib={mib(a)} reserved_mib={mib(r)} "
        f"card_free_start_mib={mib(free_start)} card_free_mib={mib(free)} card_total_mib={mib(total)}"
    )
    logger.info(line)
    return line


def _capturing(cuda) -> bool:
    from sglang.srt.model_executor.vram_family_census import _capturing as cap

    return cap(cuda)


def on_forward_end(runner, forward_batch, cuda) -> Optional[str]:
    """Called at every forward end (``maybe_log_vram_peak``). Extend of the
    target runner -> ``phase=chunk``; decode/verify -> count, close as
    ``phase=round`` every N rounds or 5 s. The draft runner never closes a
    window (it shares the allocator; its forwards belong to the round)."""
    if not _enabled() or getattr(runner, "is_draft_model_runner", False):
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or _capturing(cuda):
        return None
    from sglang.srt.model_executor.vram_family_census import _rank_of

    rank = _rank_of(runner)
    _STATE["rank"] = rank
    is_verify = bool(getattr(mode, "is_target_verify", lambda: False)())
    if mode.is_extend() and not is_verify:
        ids = getattr(forward_batch, "input_ids", None)
        rows = int(ids.shape[0]) if ids is not None else -1
        if _STATE["rounds"]:
            _close(cuda, rank, "round")
        return _close(cuda, rank, "chunk", f" rows={rows}")
    if mode.is_decode() or is_verify:
        _STATE["rounds"] += 1
        t0 = _STATE["t0"]
        if _STATE["rounds"] >= _round_every() or (t0 is not None and time.time() - t0 >= ROUND_WINDOW_MAX_S):
            return _close(cuda, rank, "round")
    return None


def _scheduler_rank(owner) -> int:
    """Rank of the leg's owner -- the weight-updater manager (``.tp_worker``,
    ``.scheduler``) or a scheduler (``.ps``): the same index the forward
    records print (``_rank_of`` of the target runner), else the PP rank on a
    PP group and the TP rank otherwise, else the last forward's rank."""
    runner = getattr(getattr(owner, "tp_worker", None), "model_runner", None)
    if runner is not None:
        try:
            from sglang.srt.model_executor.vram_family_census import _rank_of

            return int(_rank_of(runner))
        except Exception:  # noqa: BLE001
            pass
    ps = getattr(owner, "ps", None) or getattr(getattr(owner, "scheduler", None), "ps", None)
    try:
        if ps is not None:
            if int(getattr(ps, "pp_size", 1) or 1) > 1:
                return int(ps.pp_rank)
            return int(getattr(ps, "tp_rank", 0) or 0)
    except Exception:  # noqa: BLE001
        pass
    r = _STATE.get("rank")
    return int(r) if r is not None else -1


def _leg_cuda():
    """``torch.cuda`` when the instrument is on and CUDA is up in this
    process, else ``None`` (never initializes CUDA)."""
    if not _enabled():
        return None
    if _STATE.get("cuda_override") is not None:  # hermetic tests
        return _STATE["cuda_override"]
    try:
        import torch

        return torch.cuda if torch.cuda.is_initialized() else None
    except Exception:  # noqa: BLE001
        return None


def flip_leg(leg: str):
    """Decorator for the scheduler's release/resume RPC handlers: close the
    open window before the leg (``round`` or ``idle``), then the leg's own
    window after it (``phase=flip leg=<leg> ms=<rpc>``), even when the leg
    raises. Off (``SGLANG_WEG2_VRAM_PEAK=0``) = a plain call."""

    def deco(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            cuda = _leg_cuda()
            if cuda is None:
                # off, or no CUDA in this process (hermetic tests, CPU tools):
                # nothing to read, and a read must not initialize it
                return fn(self, *args, **kwargs)
            rank = -1
            try:
                rank = _scheduler_rank(self)
                _close(cuda, rank, "round" if _STATE["rounds"] else "idle")
            except Exception as exc:  # noqa: BLE001 -- the leg runs regardless
                logger.debug("%s pre-leg skipped: %s", MARKER, exc)
            t = time.perf_counter()
            ok = "raised"
            try:
                out = fn(self, *args, **kwargs)
                ok = "ok"
                return out
            finally:
                try:
                    _close(cuda, rank, "flip",
                           f" leg={leg} rpc_ms={(time.perf_counter() - t) * 1000:.0f} rpc={ok}")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("%s leg skipped: %s", MARKER, exc)

        return wrapper

    return deco
