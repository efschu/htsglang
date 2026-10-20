"""S104 L3-FINAL-DRAIN: at SIGTERM the storage owner copies what L2 completed since its last write-behind pass.

THE METAL (rc22 boot ..._9c40fe40a2_1020_094747 -> rc23 ..._d76cae6851_1020_105621, the 10:50 stop).
pdflip-24-473 finished on D at 10:50:13 (``#59 FINISH-TRACK cache_len=37120``, front ``SEQ-MARK depth=37120
prompt=35908``): its decoded tail [35904, 37120) went into the three trees. The D->P flip's flush published it one
second later on every rank (TP0 ``#924D station=backup rid=node1290``, TP1/TP2 ``#1470 FLUSH-PUBLISH issued=15``
and a last sweep whose only leftovers were byteless worker anchors, ``in_flight_after=0``): the 19 pages were
COMPLETE in the L2 arena. SIGTERM came at 10:50:15 (D log, last line), the parent's ``kill_process_tree`` at
10:50:20 (P log). The L2 arena lives in the container's /dev/shm under the boot tag and dies with it; the only copy
that survives a restart is L3, and L3 receives a page from the write-behind thread (``HiCacheFile.
l3_write_behind_pass``, every ``FLLIPER_PDFLIP_L3_WRITE_BEHIND_S`` = 2 s, a 256 MiB budget per arena and cycle
that one 58.8 MB mamba anchor slot already eats four times -- ``budget=OVER pending=22``). The thread is a
daemon: SIGTERM's default action ended the process before it reached the pages. rc23 read the chain to 35904
(``L3-INDEX-PRESENCE depth=35904 kv_pages=561``), front ``PREFIX-DIVERGE verdict=store_short ... lost=1216`` for
pdflip-0-5, ``PDFLIP-USAGE-TRUE p_computed=10745`` = 9529 new + 1216 recomputed. The ``refused_by=
anchor_only_no_arena:5`` of that flush was NOT the loss: those were byteless Form A worker anchors (S104 F2).

THE RULE. The process that owns an L3 write-behind (one per group: D TP0, P PP0 -- ``_l3_write_behind_start``
registers it here) handles SIGTERM: it stops its write-behind thread, lets a drain thread run the same pass
without the slice budget's byte cap (its own read priority, QS pair gate and once-per-(slot, generation) rule are
unchanged) until a full cycle writes nothing or ``FINAL_DRAIN_S`` has passed, then restores the default action and
re-sends SIGTERM to itself -- the process ends exactly as before, only later by at most the bound. A rank without
a write-behind (TP1/TP2, PP1/PP2) re-sends at once: identical to the old behaviour. The drain runs in its own
thread and the handler only joins it with a timeout, so a lock the interrupted main thread holds can delay the
drain but never the exit. ``FINAL_DRAIN_S`` = 3 s: the measured window between the group's SIGTERM and the
parent's ``kill_process_tree`` is 5 s (10:50:15 -> 10:50:20, "Sleeping 5 seconds before crash diagnostics"); a
process still alive then would be taken by the parent's py-spy/coredump diagnostics instead of exiting.

What this cannot cover, named: a rank death (W86 at rc23 11:05) kills without SIGTERM, and a SIGKILL ends the
owner before any handler; the pages completed inside the last write-behind tick are then lost as before.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LINE = "S104 L3-FINAL-DRAIN"
#: seconds the SIGTERM handler gives the drain (see the module head: 5 s measured to the parent's kill)
FINAL_DRAIN_S = 3.0
#: no byte cap for the final cycle -- the bound is the wall clock, not a budget
_UNBOUNDED_BYTES = 1 << 50

_BACKENDS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_STATE: Dict[str, Any] = {"installed": False, "fired": 0}


def register(backend: Any) -> None:
    """Called by ``HiCacheFile._l3_write_behind_start`` once its thread runs (the storage owner only). On the main
    thread (the boot's cache attach) it arms the handler at once; else ``install`` at boot-complete arms it."""
    try:
        _BACKENDS.add(backend)
    except TypeError:  # noqa: BLE001 - an object without weakref support (test scaffolds): not registered
        return
    if threading.current_thread() is threading.main_thread():
        install()


def registered() -> List[Any]:
    return list(_BACKENDS)


def drain_backend(backend: Any, deadline: float, *, clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Stop ``backend``'s write-behind thread and run its pass until a full cycle writes nothing or ``deadline``.

    Returns the summed counters. Never raises (a pass failure ends the drain, named)."""
    out: Dict[str, Any] = {"passes": 0, "cycles": 0, "written": 0, "bytes": 0, "pending": 0, "refused": 0,
                           "stopped": "", "ms": 0.0}
    t0 = clock()
    stop = getattr(backend, "_l3wb_stop", None)
    thr = getattr(backend, "_l3wb_thread", None)
    if stop is not None:
        stop.set()
    if thr is not None and thr is not threading.current_thread() and thr.is_alive():
        thr.join(timeout=max(0.0, min(0.5, deadline - clock())))
        if thr.is_alive():
            # its pass is still inside one slice: two passes over the same arena cursors are not allowed
            out["stopped"] = "thread_busy"
            out["ms"] = (clock() - t0) * 1000.0
            return out
    pass_fn = getattr(backend, "l3_write_behind_pass", None)
    if not callable(pass_fn):
        out["stopped"] = "no_pass"
        return out
    cont = False
    cycle_written = 0
    while True:
        left = deadline - clock()
        if left <= 0.0:
            out["stopped"] = "deadline"
            break
        try:
            tot = pass_fn(budget_bytes=_UNBOUNDED_BYTES, quiet=lambda: None, cont=cont,
                          slice_s=max(0.01, min(0.5, left))) or {}
        except Exception as exc:  # noqa: BLE001 - the exit must come; the failure is named
            logger.warning("%s pass raised: %r", LINE, exc)
            out["stopped"] = "raised"
            break
        out["passes"] += 1
        w = int(tot.get("written", 0) or 0)
        out["written"] += w
        out["bytes"] += int(tot.get("bytes", 0) or 0)
        out["pending"] = int(tot.get("pending", 0) or 0)
        out["refused"] += int(tot.get("refused", 0) or 0)
        cycle_written += w
        if tot.get("paused"):
            out["stopped"] = "paused:%s" % tot.get("paused")
            break
        if tot.get("sliced"):
            cont = True
            continue
        out["cycles"] += 1
        if cycle_written == 0:
            out["stopped"] = "done"
            break
        cont = False
        cycle_written = 0
    out["ms"] = (clock() - t0) * 1000.0
    return out


def drain_all(deadline: float) -> List[Dict[str, Any]]:
    res = []
    for b in registered():
        r = drain_backend(b, deadline)
        r["store"] = str(getattr(b, "file_path", "?"))
        res.append(r)
    return res


def _handler(signum, frame) -> None:  # noqa: ARG001 - signal signature
    _STATE["fired"] += 1
    try:
        backends = registered()
        if backends:
            deadline = time.monotonic() + FINAL_DRAIN_S
            box: Dict[str, Any] = {}

            def _run():
                box["res"] = drain_all(deadline)

            t = threading.Thread(target=_run, name="s104-l3-final-drain", daemon=True)
            t.start()
            t.join(timeout=FINAL_DRAIN_S + 0.25)
            for r in box.get("res", []) or []:
                logger.warning(
                    "%s pid=%d store=%s stopped=%s passes=%d cycles=%d written=%d bytes=%d pending=%d refused=%d "
                    "ms=%.0f (SIGTERM: the pages L2 completed since the last write-behind pass get their L3 copy "
                    "before the process ends; the arena dies with the boot, L3 does not)",
                    LINE, os.getpid(), r.get("store"), r.get("stopped"), r["passes"], r["cycles"], r["written"],
                    r["bytes"], r["pending"], r["refused"], r["ms"])
            if t.is_alive():
                logger.warning("%s pid=%d stopped=timeout after %.1f s: the process ends now (the pass in flight "
                               "is abandoned)", LINE, os.getpid(), FINAL_DRAIN_S)
    except BaseException as exc:  # noqa: BLE001 - the exit must come
        try:
            logger.warning("%s handler raised: %r", LINE, exc)
        except BaseException:  # noqa: BLE001
            pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)


def install() -> bool:
    """Install the SIGTERM handler in THIS process -- only in a process that OWNS a write-behind (D TP0, P PP0),
    main thread only, never over a handler someone else set. A rank without one keeps SIGTERM's default action:
    a Python handler runs only between bytecodes, so it would delay the death of a rank whose main thread is
    blocked in C (a collective whose peer is gone) -- that is never traded for nothing."""
    if _STATE["installed"]:
        return True
    if not registered():
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    try:
        prev = signal.getsignal(signal.SIGTERM)
    except Exception:  # noqa: BLE001
        return False
    if prev not in (signal.SIG_DFL, None):
        logger.info("%s not installed: SIGTERM already has a handler (%r)", LINE, prev)
        return False
    signal.signal(signal.SIGTERM, _handler)
    _STATE["installed"] = True
    logger.info("%s armed pid=%d stores=%s (SIGTERM: the write-behind's L2 tail goes to L3 first, at most %.1f s)",
                LINE, os.getpid(), [str(getattr(b, "file_path", "?")) for b in registered()], FINAL_DRAIN_S)
    return True
