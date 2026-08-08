# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#603b: on-demand, per-rank dump of the BAR1 launch record (SIGUSR1).

WHY THIS EXISTS
---------------
The 2026-08-06 crash family has a second shape the existing instruments cannot
resolve. The collective census compares COUNTS and reports them byte-identical;
py-spy shows all three ranks stuck at the same Python line. Both are true and
neither names the defect, because the ranks are not disagreeing about how MANY
collectives they ran -- they are disagreeing about WHICH ONE they are currently
in, and that position lives in the transport, not in the stack.

``BarlinkBar1Transport`` already records exactly that (``_note_launch``:
``_last_op``, ``_last_nbytes``, ``_unchecked_launches``). It is only ever
printed by the rank that RAISES, and exactly one rank raises per crash -- so
the comparison that would name the offset has never been available. This makes
it available on demand, from outside, on every rank at once.

THE HARD CONSTRAINT: NO DEVICE SYNC
-----------------------------------
This handler runs while the GPU is wedged -- a spin kernel is burning at 100%
util waiting for a peer flag that is not coming. Any read of device memory
(``.item()``, ``.cpu()``, ``torch.cuda.synchronize()``) would enqueue behind
that kernel and hang the handler, destroying the evidence it exists to collect
and adding a second wedge on top of the first.

So: HOST-SIDE STATE ONLY. Plain Python attributes, and tensors only when they
are already CPU-resident. Device-resident flag/epoch words are deliberately
NOT read, and are reported as "device-resident (not read: would sync)" rather
than omitted, so a reader is never left wondering whether the field was empty
or skipped.

USE
---
``kill -USR1 <each scheduler pid>``; every rank appends one line to its log.
Diff the three lines. Equal ``unchecked`` with a different ``last_op`` is a
POSITION offset; different ``unchecked`` is a COUNT divergence (which the
census would also have caught).
"""

from __future__ import annotations

import logging
import signal
import threading
import weakref
from typing import Any, List

logger = logging.getLogger(__name__)

__all__ = ["register", "dump", "install_sigusr1_handler"]

#: Weak refs, so a dead transport cannot keep itself alive for a diagnostic.
_TRANSPORTS: "List[weakref.ref]" = []
_LOCK = threading.Lock()
_INSTALLED = False


def register(transport: Any) -> None:
    """Register a live transport for the signal dump. Never raises."""
    try:
        with _LOCK:
            _TRANSPORTS.append(weakref.ref(transport))
    except Exception:  # noqa: BLE001 - a diagnostic must not be the cause
        pass


def _describe(t: Any) -> str:
    """One transport's host-side launch record.

    Every field is read with a default: this runs in a signal handler on a
    process that is already in trouble, and an AttributeError here would
    replace the evidence with a traceback.
    """
    parts = [
        f"group={getattr(t, 'group_label', None) or getattr(t, '_group_label', '?')}",
        f"rank={getattr(t, 'rank', '?')}/{getattr(t, 'world_size', '?')}",
        f"last_op={getattr(t, '_last_op', '') or '(none)'}",
        f"last_nbytes={getattr(t, '_last_nbytes', -1)}",
        f"unchecked={getattr(t, '_unchecked_launches', -1)}",
        f"captured_launches={getattr(t, '_captured_launches', None)}",
        f"last_op_captured={getattr(t, '_last_op_captured', None)}",
        f"result_i={getattr(t, '_result_i', None)}",
        f"result_counter={getattr(t, '_result_counter', None)}",
    ]
    # #622: (round, mesh watermark, a2a watermark), from the PINNED mirror the
    # watchdog poll refreshes on its private stream — CPU-resident by
    # construction, so printing it here keeps the no-device-sync constraint.
    # Two spaced SIGUSR1 probes under replay load showing strict growth of all
    # three words are the ack barrier's live capture-safety proof.
    _rm = getattr(t, "_round_mirror", None)
    if _rm is not None:
        try:
            parts.append(
                f"round_mirror=[round={int(_rm[0])}, mesh_consumed={int(_rm[1])}, "
                f"a2a_consumed={int(_rm[2])}]"
            )
        except Exception:  # noqa: BLE001 - a diagnostic must not break the dump
            parts.append("round_mirror=<unreadable>")
    # CPU-resident mirrors only. See the module docstring: a device read here
    # would queue behind the wedged spin kernel and hang the handler.
    for name in ("_status_stage_host", "_status_host", "_step_host"):
        obj = getattr(t, name, None)
        if obj is None:
            continue
        try:
            if getattr(obj, "device", None) is not None and obj.device.type == "cpu":
                parts.append(f"{name}={obj.flatten()[:8].tolist()}")
            else:
                parts.append(f"{name}=device-resident (not read: would sync)")
        except Exception:  # noqa: BLE001
            parts.append(f"{name}=<unreadable>")
    return " ".join(parts)


def dump(reason: str = "signal") -> None:
    """Log every live transport's launch record. Never raises."""
    try:
        with _LOCK:
            refs = list(_TRANSPORTS)
        live = [r() for r in refs]
        live = [t for t in live if t is not None]
        if not live:
            logger.error("BARLINK-LAUNCH-DUMP (%s): no live BAR1 transport", reason)
            return
        for t in live:
            logger.error("BARLINK-LAUNCH-DUMP (%s): %s", reason, _describe(t))
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not be the cause
        try:
            logger.error("BARLINK-LAUNCH-DUMP failed: %r", exc)
        except Exception:  # noqa: BLE001
            pass


def _handler(signum, frame):  # noqa: ARG001
    dump(reason="SIGUSR1 pid-local")


#: Where the sampler thread writes. One file per rank, truncated each tick, so
#: reading it during a wedge always yields the CURRENT record without parsing.
SAMPLE_DIR = "/spinning/wedge-catch-603b"


def start_sampler(rank: int, interval: float = 1.0) -> bool:
    """Write this rank's launch record to a file every ``interval`` seconds.

    WHY A THREAD AND NOT THE SIGNAL HANDLER. Measured on-card, window boot
    441242: SIGUSR1 was delivered to all three ranks during a live wedge and
    NOTHING was logged. Python runs signal handlers only between bytecodes, and
    during the wedge every rank's main thread is parked inside a C-level CUDA
    sync (``.item()`` behind a spinning kernel), so the interpreter never
    reaches a check point and the handler cannot run. The signal path is
    therefore useless for exactly the state it was built to observe.

    A daemon thread does run: CPython releases the GIL around blocking CUDA
    calls, which is why the barlink peer-watchdog thread keeps ticking through
    the same wedge. So the record is SAMPLED continuously instead of requested
    on demand -- and that is strictly better evidence anyway, because the file
    then also holds the record from BEFORE the wedge, which is where the offset
    is introduced.

    Never raises; a diagnostic must not be the reason a boot fails.
    """
    try:
        import os

        os.makedirs(SAMPLE_DIR, exist_ok=True)
        path = os.path.join(SAMPLE_DIR, f"launch_rank{rank}.log")

        def _run():
            import time

            while True:
                try:
                    with _LOCK:
                        refs = list(_TRANSPORTS)
                    live = [r() for r in refs]
                    live = [t for t in live if t is not None]
                    stamp = time.strftime("%H:%M:%S", time.gmtime())
                    with open(path, "a") as fh:
                        for t in live:
                            fh.write(f"{stamp} {_describe(t)}\n")
                        fh.flush()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(interval)

        th = threading.Thread(
            target=_run, name="barlink-launch-sampler", daemon=True
        )
        th.start()
        logger.info(
            "BARLINK-LAUNCH-DUMP sampler started: rank %d -> %s every %.1fs "
            "(thread, not signal: a signal handler cannot run while the main "
            "thread is inside a CUDA sync).",
            rank,
            path,
            interval,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def install_sigusr1_handler() -> bool:
    """Install the SIGUSR1 handler once, from the main thread.

    Returns True when this call installed it. Signal handlers can only be set
    from the main thread, and the transport is constructed there in a scheduler
    process; a call from anywhere else is a no-op rather than an error, because
    a diagnostic that raises during model init would be far worse than a
    diagnostic that is quietly unavailable.
    """
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return False
        if threading.current_thread() is not threading.main_thread():
            return False
        try:
            signal.signal(signal.SIGUSR1, _handler)
            _INSTALLED = True
        except Exception:  # noqa: BLE001
            return False
    logger.info(
        "BARLINK-LAUNCH-DUMP armed: send SIGUSR1 to this pid to log the BAR1 "
        "launch record (host-side only; device flag words are deliberately not "
        "read, they would sync behind a wedged spin kernel)."
    )
    return True
