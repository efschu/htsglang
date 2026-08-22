# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#799: carry the #699/#739 admission-wedge verdict OUT of the scheduler.

The detector already knows. On boot 0822_0829 it wrote ``ADMISSION-WEDGE`` 146
times across thirteen minutes in which the lane produced zero decode batches
and zero finished requests -- and the watchdog restarted nothing, because the
verdict never left the scheduler process. This module is the missing edge, and
nothing more: the detection logic stays in
``scheduler_components.invariant_checker``.

WHY A FILE AND NOT AN HTTP FIELD. The consumer is a watchdog whose entire
remit is a server that answers HTTP while serving nobody. A signal fetched by
asking that same server is a detector sharing the failure mode it detects
(``turnkey/probe.py`` states the same rule for the generation probe). Worse,
the natural implementation -- tokenizer process queries scheduler over IPC --
routes the query through the very loop under suspicion. A file written by the
process that computes the verdict is readable when HTTP is gone, cannot block
the reader, and needs no new endpoint.

WHY NOT A GENERATION PROBE. Proving liveness by generating on a timer was
retired by user order 2026-08-14 (see ``turnkey/watchdog.py``,
``Policy.generation_probe_enabled``). This signal is passive: the scheduler
publishes what it already computed, and the watchdog reads a file.

TRI-STATE, DELIBERATELY. :class:`WedgeSignal.verdict` is ``True`` (a rank
reports a wedge), ``False`` (fresh files, none wedged) or ``None`` (no
measurement: export off, no file yet, or every file stale). ``None`` must
never be read as "fine" -- a missing measurement is not a measurement, the
same rule ``Observation.generation`` already follows.

INHERITED BLIND SPOT, NAMED RATHER THAN HIDDEN. The verdict this module
carries is exactly ``admission_wedge_verdict``'s, including its documented
coverage gap: ``invariant_checker.py`` returns "not wedged" whenever
``running > 0``, so the #536 fast-lane starvation class -- a request starved
behind a co-tenant that IS running -- produces no alarm here and therefore no
watchdog action either. Transporting a verdict does not widen it. Closing that
class needs a second signal in the detector, which is #536's own work.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import time
from typing import Callable, Optional

__all__ = [
    "DEFAULT_STATUS_DIR",
    "STALE_AFTER_S",
    "WedgeSignal",
    "publish_verdict",
    "read_wedge_signal",
    "status_dir",
]

logger = logging.getLogger(__name__)

#: tmpfs, cleared by the kernel on reboot, and not on /spinning where a
#: 10-second cadence would add write traffic to the evidence volume.
DEFAULT_STATUS_DIR = "/run/htsglang/wedge"

#: A verdict older than this is no longer a measurement. Sized against the
#: publisher's own cadence: ``ADMISSION_WEDGE_POLL_SECONDS`` is 10 s, so this
#: tolerates four consecutive missed publishes before the reader downgrades to
#: "no measurement". Tight enough that a dead publisher stops being believed
#: within a minute; loose enough that one slow poll is not an outage.
STALE_AFTER_S: float = 45.0

#: Kill switch. The export is ON by default and that is a deliberate reversal
#: of this repository's recurring defect -- a detector built, tested, and left
#: unconsumed (the "counter without an actuator" form, catalogued three times
#: over). A ~120-byte atomic write every 10 s from a daemon thread that is
#: already awake cannot plausibly cost a boot, and a signal that must be
#: switched on to work is a signal that will be off during the next outage.
ENV_DISABLE = "SGLANG_WEDGE_STATUS_DISABLE"
ENV_DIR = "SGLANG_WEDGE_STATUS_DIR"


def status_dir(env: Optional[dict] = None) -> Optional[str]:
    """The directory to publish into, or ``None`` when the export is off."""
    env = os.environ if env is None else env
    if str(env.get(ENV_DISABLE, "")).strip().lower() in ("1", "true", "yes", "on"):
        return None
    return env.get(ENV_DIR) or DEFAULT_STATUS_DIR


@dataclasses.dataclass(frozen=True)
class WedgeSignal:
    """What the watchdog gets to see.

    ``verdict`` is the tri-state above. ``stale`` says a file existed but was
    too old to believe: that is NOT a wedge conviction, but it IS worth a log
    line, because a publisher that stopped publishing while the API still
    answers is its own kind of sick.
    """

    verdict: Optional[bool]
    detail: str
    ranks_seen: int = 0
    stale: bool = False

    @property
    def wedged(self) -> Optional[bool]:
        return self.verdict


def _payload(rank: str, alarm: bool, detail: str, wall: float) -> str:
    return json.dumps(
        {
            "wedged": bool(alarm),
            "detail": str(detail)[:2000],
            "wall": float(wall),
            "rank": str(rank),
            "pid": os.getpid(),
        }
    )


def publish_verdict(
    rank: str,
    alarm: bool,
    detail: str,
    directory: Optional[str] = None,
    now: Optional[Callable[[], float]] = None,
) -> Optional[str]:
    """Write this rank's current verdict. Returns the path, or ``None``.

    One file PER RANK, never one shared file: the detector runs in every
    scheduler process, and several processes rewriting one path would race and
    would also lose the information that matters most -- WHICH rank is wedged.

    Wall clock, not ``perf_counter``: the reader is a different process, and
    ``perf_counter`` epochs are per-process. Comparing them across the process
    boundary would produce an arbitrary age.

    Atomic via write-temp-then-rename, so a reader never sees a half-written
    file. Every failure is swallowed after one log line: a scheduler that dies
    because its telemetry sink is read-only is strictly worse than a watchdog
    that cannot see.
    """
    directory = status_dir() if directory is None else directory
    if not directory:
        return None
    clock = now or time.time
    try:
        os.makedirs(directory, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(rank))
        path = os.path.join(directory, f"wedge.{safe or 'rank'}.json")
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".wedge.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(_payload(rank, alarm, detail, clock()))
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path
    except Exception as e:  # noqa: BLE001 - telemetry must never kill serving
        logger.warning("wedge-status publish to %s failed: %s", directory, e)
        return None


def read_wedge_signal(
    directory: Optional[str] = None,
    stale_after_s: float = STALE_AFTER_S,
    now: Optional[Callable[[], float]] = None,
) -> WedgeSignal:
    """Read every rank's verdict and reduce them to one.

    ANY fresh rank reporting a wedge makes the lane wedged. The reduction is
    deliberately OR and not majority: under pure TP the ranks are one
    collective, and a single rank that cannot admit work stalls the group. A
    majority rule would need the wedge to spread before anyone acted.
    """
    directory = status_dir() if directory is None else directory
    if not directory:
        return WedgeSignal(None, "wedge-status export is disabled")
    clock = now or time.time
    t = clock()
    try:
        names = sorted(
            n
            for n in os.listdir(directory)
            if n.startswith("wedge.") and n.endswith(".json")
        )
    except OSError as e:
        return WedgeSignal(None, f"wedge-status dir unreadable: {e}")
    if not names:
        return WedgeSignal(None, f"no wedge-status files in {directory}")

    fresh: list = []
    stale_ranks: list = []
    for name in names:
        try:
            with open(os.path.join(directory, name), "r") as fh:
                rec = json.load(fh)
            wall = float(rec.get("wall", 0.0))
            wedged = bool(rec.get("wedged", False))
            rank = str(rec.get("rank", name))
            detail = str(rec.get("detail", ""))
        except (OSError, ValueError, TypeError) as e:
            stale_ranks.append(f"{name} unreadable ({e})")
            continue
        age = t - wall
        if age > float(stale_after_s):
            stale_ranks.append(f"{rank} {age:.0f}s old")
            continue
        fresh.append((rank, wedged, detail, age))

    if not fresh:
        return WedgeSignal(
            None,
            "every wedge-status file is stale or unreadable ("
            + "; ".join(stale_ranks[:4])
            + "): the publisher stopped publishing, which is not itself a "
            "wedge verdict",
            ranks_seen=0,
            stale=True,
        )

    wedged = [f for f in fresh if f[1]]
    if wedged:
        rank, _, detail, age = wedged[0]
        return WedgeSignal(
            True,
            f"rank {rank} reports ADMISSION-WEDGE ({age:.0f}s old): {detail}",
            ranks_seen=len(fresh),
            stale=bool(stale_ranks),
        )
    return WedgeSignal(
        False,
        f"{len(fresh)} rank(s) report no wedge",
        ranks_seen=len(fresh),
        stale=bool(stale_ranks),
    )
