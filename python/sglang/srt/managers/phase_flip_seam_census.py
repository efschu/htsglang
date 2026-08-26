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
import time
from typing import Callable, Dict, List, Optional, Tuple

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
        #: (label, monotonic seconds) per stage, parallel to ``stages``.
        #:
        #: TIME, BECAUSE THE MEMORY WALK ALREADY PROVED IT IS NOT THE PAYLOAD.
        #: Measured 2026-08-16 across five flips spanning 123 -> 102538 live
        #: slots (834x), the FLIP DONE receipt's own read+exchange+write
        #: accounted for only 458-953 ms of a 2459-3186 ms flip. The
        #: remaining 2001-2233 ms was UNNAMED, and it barely moved (231 ms
        #: spread) while the payload grew 834x -- so it is a fixed per-flip
        #: cost, and a fixed cost cannot be found by looking at transfer
        #: sizes. It has to be timed where it is spent, which is here: the
        #: census already stops at every stage boundary that matters.
        #:
        #: Kept PARALLEL to ``stages`` rather than widening that tuple, whose
        #: shape ``format_line`` and the #485 excursion analysis both read.
        self.times: List[Tuple[str, float]] = []
        #: #873: per-segment decompositions, keyed by the label that CLOSES the
        #: segment, in seconds. Registered by whoever ran the work, because
        #: that is the only code that knows what the work was made of.
        #:
        #: WHY A LINK AND NOT A NEW MARK. The decomposition of this walk's
        #: dominant segment ALREADY EXISTED -- `rotation_phase_report`
        #: (#809/W28) emits it -- as a separate log line that this line never
        #: mentioned. Two instruments over one segment with nothing joining
        #: them, so the reader of the dominant term could not find the
        #: decomposition and fitted a model to the bar instead. A third
        #: instrument would have left that unchanged.
        self.explained: Dict[str, Tuple[Tuple[str, float], ...]] = {}
        #: #917: (label, error) of the FIRST stage boundary whose probe failed
        #: with a POISON-class CUDA error, and the last label whose probe came
        #: back clean. Together they bracket the interval the context died in.
        #:
        #: WHY THIS PAIR IS THE WHOLE INSTRUMENT. `mark()` runs at ~19 named
        #: boundaries of the cutover and its probe is a DRIVER CALL
        #: (`mem_get_info`). On a poisoned context that call raises, so the
        #: census has been touching the driver at every candidate boundary all
        #: along -- and swallowing the answer as "probe-failed". Classifying it
        #: turns a record that already exists into an origin bracket, without
        #: adding a single call site or a second taxonomy of stage names.
        self.probe_poison: Optional[Tuple[str, str]] = None
        self.last_clean_label: Optional[str] = None
        #: Probe failures that were NOT poison-class, counted separately. A
        #: transient `mem_get_info` hiccup and a dead context must never again
        #: be one row -- that conflation is #867's class, and this module held
        #: the second instance of it.
        self.probe_failures = 0

    def _classify_probe_failure(self, label: str, exc: BaseException) -> None:
        """Split a failed stage probe into "hiccup" and "the context is gone".

        #917, AND IT IS #867's CLASS ARRIVING IN A SECOND MODULE. `mark()`
        caught bare `Exception` around a driver call and recorded every outcome
        as ``probe-failed``. That is right for a transient and wrong for a
        sticky CUDA fault, and it could not tell them apart -- the same defect
        `barlink_abort_gate.poll_status_words` carried until #867 narrowed its
        handler. The sweep that found the first instance never reached here,
        so the cutover's own instrument was quietly discarding the earliest
        host-visible evidence of a poisoned context at the exact boundaries
        that would have named the interval it was born in.

        WHAT THIS BUYS, CONCRETELY. In the 0826 rerun (boot #2, 21:53:36) the
        first three reports of the fault were the barlink watchdog poll, the
        #760 stream quiesce, and a scheduler traceback in `get_cpu_copy` --
        three sites, one sticky fault, and NOTE_867's standing verdict that
        earliest-in-time is evidence for origin and not proof. The watchdog
        thread polls on a timer, so it wins the race to observe an
        already-poisoned context no matter who poisoned it; that ordering
        cannot discriminate and never could. These marks run on the SCHEDULER
        thread, in the cutover's own order, so the bracket
        ``[last_clean_label, probe_poison label]`` names ONE segment of the
        walk -- and each segment has one candidate in it.

        Registered with `barlink_abort_gate.record_poison`, which is
        FIRST-WINS across the process: if a census boundary records the origin,
        the watchdog's later report says so instead of claiming it.

        DEGRADES, NEVER RAISES. An instrument on the no-return path may not be
        the reason a flip dies -- the rule this module already states for
        `_check_law`. The import is local because `managers` importing
        `distributed` at module scope is a cycle this file does not need.
        """
        try:
            from sglang.srt.distributed.device_communicators import (
                barlink_abort_gate,
            )

            if not barlink_abort_gate.is_poison_error(exc):
                self.probe_failures += 1
                return
            if self.probe_poison is None:
                self.probe_poison = (str(label), str(exc))
            barlink_abort_gate.record_poison(
                f"seam-census probe at stage '{label}'", exc
            )
        except Exception:  # pragma: no cover - the no-return path owns this
            self.probe_failures += 1

    def explain(self, label: str, terms) -> None:
        """Register what the segment ENDING at ``label`` was made of.

        Keyed on the CLOSING mark because that is the mark the work's own code
        stamps when it finishes: `stacks.refill` runs, then the walk stamps
        `weights_refill`. Keying on the opening mark would ask the caller to
        know what ran before it, which it does not.

        TOTALLY DEFENSIVE, and not as a habit. This is called from inside the
        flip's no-return region, where an instrument that raises is an instance
        rather than a missing log line -- the contract `mark()` above already
        carries. Anything that is not a sequence of (name, seconds) pairs is
        dropped here rather than at format time, so a bad registration costs
        the line its decomposition and nothing else.
        """
        try:
            clean = tuple(
                (str(name), float(secs))
                for name, secs in terms
                if secs is not None and float(secs) == float(secs)  # rejects NaN
            )
        except Exception:  # noqa: BLE001 - an instrument may never break a flip
            return
        if clean:
            self.explained[str(label)] = clean

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
        # BEFORE the probe: the probe itself calls into the driver
        # (mem_get_info), so stamping after it would charge the driver call to
        # the segment that just ended and make every segment look slower than
        # it is.
        self.times.append((str(label), time.monotonic()))
        try:
            sample = self._probe()
        except Exception as exc:  # noqa: BLE001 - classified (#917), then degraded
            sample = None
            self._classify_probe_failure(str(label), exc)
        if sample is None:
            self.stages.append((str(label), -1, -1, -1))
            return
        free, reserved, allocated = sample
        # #917: a boundary whose driver call ANSWERED proves the context was
        # still alive when the walk passed it. That is the lower end of the
        # bracket, and it is only ever set here.
        self.last_clean_label = str(label)
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

    def format_timing_line(self) -> str:
        """One line per flip per rank: per-segment WALL MILLISECONDS.

        THE COMPANION TO format_line, AND THE REASON IT WAS NOT ENOUGH. That
        line answers "which stage spent the MEMORY". This one answers "which
        stage spent the TIME", and until it existed the answer was simply not
        in the log: the FLIP DONE receipt times read, exchange and write --
        the three phases of the wave transfer -- and nothing else, so
        everything between the arm and the cutover was one unnamed block.

        Measured 2026-08-16, five flips, 123 -> 102538 live slots (834x):
        read+exchange+write accounted 458-953 ms of a 2459-3186 ms flip. The
        unnamed remainder was 2001-2233 ms and moved by only 231 ms across the
        whole 834x range. That is a FIXED per-flip cost roughly 4x the size of
        the transfer it surrounds, and it is what an idle window is made of.

        Sorted by cost DESCENDING, because a decomposition is read to find the
        dominating segment and that segment should not have to be hunted for.
        """
        if len(self.times) < 2:
            return (
                f"{LOG_PREFIX} timing {self.direction} rank {self.rank}: "
                f"only {len(self.times)} stage mark(s), no segment to time"
            )
        segs = []
        for (a_label, a_t), (b_label, b_t) in zip(self.times, self.times[1:]):
            segs.append((f"{a_label}->{b_label}", (b_t - a_t) * 1000.0))
        total_ms = (self.times[-1][1] - self.times[0][1]) * 1000.0
        ranked = sorted(segs, key=lambda kv: kv[1], reverse=True)
        body = " | ".join(f"{name} {ms:.1f}" for name, ms in ranked)
        top_name, top_ms = ranked[0]
        share = (100.0 * top_ms / total_ms) if total_ms > 0 else 0.0
        return (
            f"{LOG_PREFIX} timing {self.direction} rank {self.rank}: "
            f"{total_ms:.1f} ms across {len(segs)} segment(s), worst "
            f"'{top_name}' {top_ms:.1f} ms ({share:.0f}% of the walk)"
            f"{self._worst_attribution(top_name, top_ms)}. "
            f"ms by segment, descending -- {body}"
        )

    def _worst_attribution(self, top_name: str, top_ms: float) -> str:
        """#873: the dominant segment either shows its parts or says it has none.

        THE DOMINANT ONE AND NO OTHER. A decomposition printed for every segment
        is a wall of numbers; the segment a reader ACTS on is the worst one, and
        that is the one that has to be either attributed or declared
        unattributable. The other twelve keep their bare millisecond, which is
        all a bar chart needs to be.

        THE REMAINDER IS NAMED, NEVER DISTRIBUTED (#846, and the rule
        `RotationPhases.residual_s` already states for its own sum): a
        decomposition that silently absorbs its leftover into its largest term
        is how a reconciliation is made to pass without meaning anything. An
        OVERRUN -- terms wider than the segment, which two clocks and a
        late-stamped mark can genuinely produce -- is reported for the same
        reason and is not clamped to zero, because a clamp reads as perfect
        agreement.
        """
        terms = self.explained.get(str(top_name).split("->")[-1])
        if not terms:
            return (
                " -- NOT DECOMPOSED: this is a boundary between two marks, not "
                "a mechanism. Whatever ran here may be several kinds of work "
                "with different cost drivers, so do not fit a rate-plus-"
                "constant model to it; register a decomposition with "
                "SeamCensus.explain() from the code that runs it"
            )
        try:
            body = " + ".join(f"{name} {secs * 1000.0:.1f}" for name, secs in terms)
            accounted_ms = sum(float(secs) for _n, secs in terms) * 1000.0
            gap = float(top_ms) - accounted_ms
            tail = (
                f" + UNEXPLAINED {gap:.1f}"
                if gap >= 0
                else f" -- OVERRUN by {-gap:.1f} ms (the terms are wider than "
                "the segment: two clocks, or a mark stamped after the work it "
                "closes)"
            )
            return f" = {body}{tail}"
        except Exception:  # noqa: BLE001 - an instrument may never break a flip
            return " -- decomposition present but unformattable"

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
        head += self._poison_clause()
        return head + " | " + " | ".join(parts)

    def _poison_clause(self) -> str:
        """The origin BRACKET, or nothing at all. #917.

        DECLARED AS AN INSTRUMENT, NOT A FIX. It does not stop a fault; it
        answers the one question the 0826 specimens could not -- which segment
        of the cutover the context died in -- and it answers it with the
        cutover's own stage names rather than a taxonomy invented for the
        occasion. Read it as: everything up to and including
        ``last clean`` executed against a live context; the fault was born in
        the segment that ends at ``poisoned``.

        TWO SHAPES, BECAUSE THERE ARE TWO WAYS TO LEARN THIS.

        * This census caught it: it names both ends itself.
        * Another thread caught it first -- in practice the barlink watchdog,
          which polls on a timer and therefore almost always wins the race to
          observe an already-poisoned context. Its record carries no position
          in the walk, so the clause supplies one: the last boundary this rank
          passed cleanly. That converts "somebody saw poison" into "poison
          appeared after <stage>", which is the reading the W40 and 0826
          specimens each lacked, and it is exactly NOTE_867's open question 2
          made answerable WITHOUT silencing the watchdog. `pause_polling()`
          would have answered it by removing the fastest detector from the
          no-return path; this answers it by giving that detector a coordinate.

        Silent when nothing was poisoned, so a healthy flip's line is
        byte-identical to before.
        """
        if self.probe_poison is not None:
            label, err = self.probe_poison
            clean = self.last_clean_label or "<none>"
            return (
                f" *** #917 CONTEXT POISONED between '{clean}' and '{label}': "
                f"{err.splitlines()[0] if err else '?'} ***"
            )
        try:
            from sglang.srt.distributed.device_communicators import (
                barlink_abort_gate,
            )

            record = barlink_abort_gate.poison_record()
        except Exception:  # pragma: no cover - the no-return path owns this
            record = None
        if not record:
            return ""
        clean = self.last_clean_label or "<none>"
        return (
            f" *** #917 CONTEXT POISONED, first reported by "
            f"{record.get('source')!r}; this rank's last clean stage "
            f"boundary was '{clean}' ***"
        )


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


def explain(label: str, terms) -> None:
    """#873: register a decomposition for the segment ending at ``label``.

    No-op when no census is open, exactly like ``mark()`` above and for the
    same reason: the work that knows the decomposition also runs on boot paths
    where no flip census exists, and it may not have to ask.
    """
    census = _active
    if census is None:
        return
    try:
        census.explain(label, terms)
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
        logger.info("%s", census.format_timing_line())
    except Exception:  # pragma: no cover
        pass
    return census


def active() -> Optional[SeamCensus]:
    return _active


def reset() -> None:
    """Drop any open census. For tests and for a flip that aborted."""
    global _active
    _active = None
