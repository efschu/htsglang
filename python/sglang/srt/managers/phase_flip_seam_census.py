# Copyright 2023-2024 SGLang Team
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
"""#631: per-STAGE GPU memory attribution across one PP<->TP cutover.

WHY AN AGGREGATE WAS NOT ENOUGH
-------------------------------
The cost of a cutover was measured from OUTSIDE the process by sampling
NVML at 100 ms across one driven flip (2026-08-09, pool 253528):

    card 0   free 5705 -> 2745 MiB     transient 2960 MiB
    card 1   free 6248 -> 4872 MiB     transient 1376 MiB
    card 2   free 4507 -> 2679 MiB     transient 1828 MiB

That number is REAL and it is a TRANSIENT, not a phase steady-state
difference: free memory returns to 5685/6248/4487 MiB afterwards, within
20 MiB of the pre-flip baseline, and stays there. (Checking that the
trajectory RECOVERS is what separates the two readings, and it is the
first thing to re-check if this number is ever re-measured -- a peak that
does not recover is a phase asymmetry and has an entirely different fix.)

But an aggregate cannot say WHICH stage spends it, and the candidate
stages have entirely different fixes. Two candidates are already
excluded, both by evidence rather than by argument:

  * the CARRIED KV PAYLOAD is not it. The runtime's own per-flip stats
    report 0.5-0.7 MiB sent per rank (epochs 1 and 2, boot 22:23Z) --
    four orders of magnitude below the transient. The flip carries almost
    nothing; it is the LAYOUT change that is expensive, not the data.
  * ``arena_refill`` is not it by construction: it is one contiguous
    ``copy_`` from the host image into an arena that already exists, with
    no device-side staging, and its only transient (the post-copy
    checksum) is bounded by ``_checksum_chunk_bytes`` and was measured at
    <=128 MiB.

What remains is the CROSS-PHASE BACKING SWAP (``release_backing`` /
``restore_backing``, which really do call ``cuMemRelease`` /
``cuMemCreate``) and the GDN state mover. This census settles it by
measurement instead of a fourth round of deduction.

WHAT IT MEASURES, AND WHY EACH COLUMN IS THERE
----------------------------------------------
At every stage boundary it records three numbers that disagree with each
other in informative ways:

  ``free``        NVML/driver free bytes. THE observable that matters:
                  it is exactly what refuses a ``cuMemCreate``, and it is
                  what the 1024 MiB corridor floor is denominated in.
  ``reserved``    torch's caching-allocator reserve. Memory torch holds
                  that the driver has already handed out.
  ``allocated``   the live subset of ``reserved``.

``reserved - allocated`` is torch's SLACK, and the slack is why the
driver-visible peak scales faster than the pool: the driver is asked for
(what the stage wants) - (slack torch happens to be sitting on), so a
configuration that shrinks the slack raises the driver-visible cost of an
unchanged stage. Recording all three is what makes that separable; a
census of ``free`` alone would show the symptom and hide the mechanism.

DISCIPLINE
----------
* Marking outside an open census is a NO-OP, never an error. Stage marks
  sit on the cutover path, which is the no-return region: this instrument
  may not be the reason a flip dies. Every failure mode here degrades to
  "no census line".
* One census at a time per process, enforced. The flip protocol runs one
  cutover per rank at a time, so a second open census means a caller bug
  and is worth a loud line -- but it still does not raise.
* Cost is a handful of ``mem_get_info`` calls per flip (microseconds
  each), so this stays ON by default. An instrument that only runs when
  someone remembers to enable it is an instrument that is off during the
  incident worth measuring.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#631 seam-census]"

_MIB = 1024 * 1024


def _default_probe() -> Optional[Tuple[int, int, int]]:
    """(free, reserved, allocated) bytes on the current device, or None.

    Returns None rather than raising when CUDA is unavailable or the
    driver call fails: on the cutover path a broken probe must cost a
    missing line and nothing else.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return (
            int(free),
            int(torch.cuda.memory_reserved()),
            int(torch.cuda.memory_allocated()),
        )
    except Exception:  # pragma: no cover - defensive on the no-return path
        return None


class SeamCensus:
    """Stage-by-stage memory record for ONE cutover on ONE rank."""

    def __init__(
        self,
        direction: str,
        rank: int,
        probe: Callable[[], Optional[Tuple[int, int, int]]] = _default_probe,
    ) -> None:
        self.direction = str(direction)
        self.rank = int(rank)
        self._probe = probe
        # (label, free, reserved, allocated); free==-1 marks a failed probe,
        # which is kept as a row so a gap in the record is visible rather
        # than silently closed up.
        self.stages: List[Tuple[str, int, int, int]] = []

    def mark(self, label: str) -> None:
        """Record one stage boundary.

        The guard sits HERE, at the call to the probe, and not at the
        module-level ``mark()`` -- which is where it was first written,
        and which left ``begin()``'s own "entry" mark unprotected because
        ``begin`` calls this method directly. A guard at one of two call
        sites is a guard that does not run on the other; on the cutover's
        no-return path that is an instance, not a missing log line.
        (Caught by test_a_probe_that_raises_does_not_escape.)
        """
        try:
            sample = self._probe()
        except Exception:
            sample = None
        if sample is None:
            self.stages.append((str(label), -1, -1, -1))
            return
        free, reserved, allocated = sample
        self.stages.append((str(label), free, reserved, allocated))

    @property
    def baseline_free(self) -> Optional[int]:
        """Free bytes at the first successful probe."""
        for _label, free, _r, _a in self.stages:
            if free >= 0:
                return free
        return None

    def trough(self) -> Optional[Tuple[str, int]]:
        """(label, free) of the stage with the LEAST free memory."""
        valid = [(lbl, free) for lbl, free, _r, _a in self.stages if free >= 0]
        if not valid:
            return None
        return min(valid, key=lambda row: row[1])

    def peak_bytes(self) -> Optional[int]:
        """Driver-visible transient: baseline free minus the trough.

        Negative would mean the flip ENDED with more free memory than it
        started with at its deepest point, which cannot happen by
        definition of a minimum; it is clamped at 0 only so a caller can
        treat the result as a size.
        """
        base = self.baseline_free
        low = self.trough()
        if base is None or low is None:
            return None
        return max(0, base - low[1])

    def format_line(self) -> str:
        """One line per flip per rank: per-stage DELTAS against baseline.

        Deltas rather than absolutes because the question this answers is
        "which stage spent the memory", and three columns of 5-digit
        absolutes make that arithmetic the reader's job.
        """
        base = self.baseline_free
        parts = []
        prev_free: Optional[int] = None
        for label, free, reserved, allocated in self.stages:
            if free < 0:
                parts.append(f"{label}=probe-failed")
                continue
            step = "" if prev_free is None else f" step{(free - prev_free) // _MIB:+d}"
            parts.append(
                f"{label} free={free // _MIB}{step} "
                f"slack={(reserved - allocated) // _MIB}"
            )
            prev_free = free
        low = self.trough()
        peak = self.peak_bytes()
        head = (
            f"{LOG_PREFIX} {self.direction} rank {self.rank}: "
            f"transient {(peak or 0) // _MIB} MiB"
        )
        if low is not None and base is not None:
            head += (
                f" (baseline free {base // _MIB} MiB, trough {low[1] // _MIB} MiB "
                f"at '{low[0]}')"
            )
        return head + " | " + " | ".join(parts)


_active: Optional[SeamCensus] = None


def begin(
    direction: str,
    rank: int,
    probe: Callable[[], Optional[Tuple[int, int, int]]] = _default_probe,
) -> Optional[SeamCensus]:
    """Open a census for one cutover. ``probe`` is injectable so the pins
    can drive this off CUDA -- the stage bookkeeping is the part under
    test, and it must not need a GPU to be tested."""
    global _active
    if _active is not None:
        logger.warning(
            "%s begin(%s) while a census for %s is still open; the earlier "
            "one is dropped. One cutover runs at a time per rank, so this "
            "means a flip did not reach its end() -- look for an exception "
            "on the cutover path.",
            LOG_PREFIX,
            direction,
            _active.direction,
        )
    _active = SeamCensus(direction, rank, probe)
    _active.mark("entry")
    return _active


def mark(label: str) -> None:
    """Record a stage boundary. No-op when no census is open."""
    census = _active
    if census is None:
        return
    try:
        census.mark(label)
    except Exception:  # pragma: no cover - the no-return path owns this
        pass


def end() -> Optional[SeamCensus]:
    """Close the census, log its line, and return it. None if none open."""
    global _active
    census = _active
    _active = None
    if census is None:
        return None
    try:
        census.mark("done")
        logger.info("%s", census.format_line())
    except Exception:  # pragma: no cover
        pass
    return census


def active() -> Optional[SeamCensus]:
    return _active


def reset() -> None:
    """Drop any open census. For tests and for a flip that aborted."""
    global _active
    _active = None
