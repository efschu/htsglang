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

**IT DID SETTLE IT, AND THE ANSWER WAS THE EXCLUDED CANDIDATE (#485,
2026-08-14).** The two paragraphs above are kept verbatim because the
deduction they record is exactly what this instrument was built to
overrule, and deleting a falsified prediction hides how the mistake was
made. The measurement, pooled over the 196 ``tp_to_pp`` flips of the two
#485 certification windows (s50 + s51):

    stage at the trough        weights_refill, 86 of 86 flips on rank 0
    gdn_state                  step +0 MiB in every flip
    backing_restore_span       -5280 MiB, netted by backing_release_span
                               +3584 -- identical to the MiB in the modal
                               flip AND in the one that broke the law
    weights_refill             -4278 MiB modal, -5536 once

So the backing swap and the GDN mover -- "what remains" -- are not what
spends it; the refill is, by a factor of three.

WHERE THE DEDUCTION WENT WRONG, because the error is instructive and
survives in other reasoning about this path: "an arena that ALREADY
EXISTS" is false on the ``tp_to_pp`` leg. ``PhaseFlipStacks.refill``
calls ``_commit_refill_high_water()`` BEFORE the copy, and on a rank
whose PP layout is the larger one that grows the arena from the TP
layout back to the high-water through ``arena_carrier.set_active_prefix``
-- a real ``cuMemCreate`` of ~4150 MiB, which is the term the exclusion
never considered. The copy and the checksum were priced correctly; the
commit that precedes them was not priced at all. The 128 MiB checksum
bound is also confirmed by this data rather than contradicted: the step
takes exactly two modal values, -4150 and -4278, and the 128 MiB
difference between them IS the checksum.

Full derivation and the amended certification criterion:
``docs/dev/485/EXCURSION_ANALYSIS_485.md``.

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


def law_floor_bytes() -> int:
    """The corridor law floor in bytes, or 0 when the check is disabled.

    THE ONE-LINE HISTORY: for three shifts this module sampled the exact
    observable the corridor law is written against, named the 1024 MiB floor in
    its own docstring above, and never compared the two. On 2026-08-12 a
    ``pp_to_tp`` cutover on PP1 sat at 940 MiB free for 1.5 s -- 84 MiB under
    the law -- and the process recorded the number in a line it emitted AFTER
    the flip had already completed, so the breach was discovered hours later
    from an external 100 ms CSV. A recorder that holds the evidence and does
    not read it is not neutral: it is the reason a breach looks like a clean
    run until someone else measures it.

    The LAW floor (1024 MiB), not ``SGLANG_CORRIDOR_FLOOR_MIB`` (1536), which
    is the admission gate's ARMING floor and carries a margin on top of the
    law. Reporting a "breach" on the arming floor would cry wolf on every
    cutover that legally spends its margin.
    """
    # 656: through the ONE declaration, not a private env read with its own
    # "1024" fallback. See corridor_guard.LAW_ENV.
    from sglang.srt.managers.corridor_guard import corridor_law_bytes

    return corridor_law_bytes()


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
        #: Stages whose free reading was BELOW the corridor law floor. Kept as
        #: a list rather than a flag so the line can say how long the walk
        #: stayed under, which is the difference between a sampling artefact
        #: and the 1.5 s plateau this was built for.
        self.below_law: List[Tuple[str, int]] = []
        self._law_announced = False

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
        self._check_law(str(label), free, reserved, allocated)

    def _check_law(self, label: str, free: int, reserved: int, allocated: int) -> None:
        """Compare this stage's free reading against the corridor law.

        Announced ONCE per census, at the stage that first crosses, and then
        counted silently: a cutover that spends 60 stages under the floor must
        not emit 60 identical ERROR lines on the no-return path, but it must
        also not be indistinguishable from one that dipped for a single sample.
        The count goes on the summary line.

        Reports the SLACK alongside, because the slack is what decides whether
        the crossing was fundable. At the 2026-08-12 trough it read 1054 MiB --
        i.e. the walk could have stayed legal at any moment and nothing asked.
        Guarded like every other statement in this module: an instrument on the
        cutover path may not be the reason a flip dies.
        """
        floor = law_floor_bytes()
        if floor <= 0 or free >= floor:
            return
        self.below_law.append((label, free))
        if self._law_announced:
            return
        self._law_announced = True
        try:
            logger.error(
                "%s CORRIDOR LAW BROKEN during %s rank %d at stage '%s': free "
                "%d MiB is below the %d MiB floor. torch is holding %d MiB of "
                "slack (reserved %d MiB, allocated %d MiB) -- if that number is "
                "large, the walk was fundable and the reclaim did not run; if "
                "it is small, this configuration cannot fund this cutover and "
                "the budget is the finding. Reported HERE, at the stage that "
                "crossed, because the summary line is emitted only after the "
                "flip completes.",
                LOG_PREFIX,
                self.direction,
                self.rank,
                label,
                free // _MIB,
                floor // _MIB,
                (reserved - allocated) // _MIB,
                reserved // _MIB,
                allocated // _MIB,
            )
        except Exception:  # pragma: no cover - the no-return path owns this
            pass

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
            # #485: ``allocated`` IS PRINTED, and it is not decoration.
            #
            # The three suspects for the one seam that broke the corridor law
            # (docs/dev/485/EXCURSION_ANALYSIS_485.md) are separated by
            # exactly one column, and it is this one. ``allocated`` is live
            # torch bytes, physically backed by definition, so:
            #
            #   allocated FLAT while free drops  -> the bytes went somewhere
            #                                       outside torch (a VMM
            #                                       commit, the arena carrier)
            #   allocated RISES with free falling -> a torch allocation
            #
            # ``slack`` alone cannot carry that: it is a DIFFERENCE, so an
            # allocation that raises reserved and allocated together leaves it
            # unchanged -- which is precisely what the 11:44:06 breach did
            # (slack 652 both before and after the anomalous refill).
            #
            # And ``reserved`` cannot carry it either, on this rig: measured
            # at rest on the 5090, reserved=32040.0 against nvml_used=30050.1
            # is 1990 MiB of reserve that no physical page backs, so a
            # virtual reserve can grow without costing a byte of NVML free.
            # Both are printed anyway -- the point of a census is that the
            # reader does not have to have guessed the right column in
            # advance -- but ``allocated`` is the one that decides.
            parts.append(
                f"{label} free={free // _MIB}{step} "
                f"slack={(reserved - allocated) // _MIB} "
                f"alloc={allocated // _MIB} res={reserved // _MIB}"
            )
            prev_free = free
        low = self.trough()
        peak = self.peak_bytes()
        # NAME IT FOR WHAT IT IS. If the least-free stage is the LAST one,
        # the flip simply ENDED lower than it started and the number is the
        # PHASE STEP -- the destination layout holding more than the source
        # -- not a transient the cutover spends and gives back. Calling
        # both "transient" is exactly the conflation that put a withdrawn
        # 1.4-3.0 GiB "cutover cost" into three handoffs; the instrument
        # must not re-seed it.
        # Compared against the last stage with a VALID reading, not simply
        # the last row: end() appends a "done" mark whose probe can fail,
        # and a failed row at the end would otherwise mask a flip that
        # really did finish at its lowest point.
        last_valid = None
        for label_i, free_i, _r, _a in self.stages:
            if free_i >= 0:
                last_valid = label_i
        ended_at_the_low = low is not None and last_valid == low[0]
        label = "phase step" if ended_at_the_low else "transient"
        head = (
            f"{LOG_PREFIX} {self.direction} rank {self.rank}: "
            f"{label} {(peak or 0) // _MIB} MiB"
        )
        if low is not None and base is not None:
            head += (
                f" (baseline free {base // _MIB} MiB, trough {low[1] // _MIB} MiB "
                f"at '{low[0]}')"
            )
        # The verdict belongs in the HEAD, not buried among the stage parts:
        # every consumer of this corpus greps the head, and a breach that is
        # only visible to whoever reads to the end of a 60-stage line is a
        # breach that gets missed -- which is exactly what happened once.
        if self.below_law:
            head += (
                f" *** CORRIDOR LAW BROKEN: {len(self.below_law)} stage(s) below "
                f"{law_floor_bytes() // _MIB} MiB, deepest "
                f"{min(free for _lbl, free in self.below_law) // _MIB} MiB "
                f"at '{self.below_law[0][0]}' ***"
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
