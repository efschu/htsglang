# SPDX-License-Identifier: Apache-2.0
"""#1284: the host ring's NEED series, and the refusal that reads it.

WHAT KILLED weg2sb5e, in one sentence: the sleeping group's D2H leg walked into
a ring that its peer was not emptying, sat out the whole 110 s acquire budget
inside ``host_ring.cpp`` and then died with ``W31 Weg2HostRingExhausted -- the
launch check (L6) was violated``, having emitted NOTHING in between.

TWO THINGS ARE WRONG THERE AND THIS MODULE FIXES THE SECOND, MEASURABLY.

1.  L6 ARMED A RING WITH NEGATIVE SERIAL SLACK ON EVERY CARD.  weg2sb5e's own
    launch lines say it: ``SERIAL FORM need_d2p=19149 need_p2d=20050
    slack=-1608/-2509 MiB`` (nvml1), ``-881/-2986`` (nvml0), ``-982/-2622``
    (nvml2).  The serial requirement ``H(c) >= image_W(c) + max_tag_S(c)`` was
    COMPUTED, PRINTED as failing on all three cards, and then waived, because
    the per-card ``WEG2-HOST-RING CHECK`` line resolved ``leg_form=interleave``
    and checked "the R5 corridor (W32) alone".  R5's corridor is only sound
    under a premise the front's own code denies in writing -- "there is no
    happens-before between S's begin_leg and W's first read"
    (``front.py`` at the ``asyncio.gather`` of the two legs).  So the ring runs
    on a race, and 30 flips were that race being won 30 times.

2.  WHEN THE RACE IS LOST THERE IS NO INSTRUMENT AND NO EARLY REFUSAL.  The
    C++ waiter's only output is the death line 110 s later, and its text blames
    L6 -- which sends every reader to the launch arithmetic even when, as here,
    the launch arithmetic is not what changed.  Nothing in the log says how
    close any earlier flip came.

THE MEASUREMENT THAT SETTLES CREEP-VS-ARITHMETIC, AND WHY IT HAD TO BE ADDED.
The question "does the per-tag need GROW per flip" was answerable for weg2sb5e
only by reconstructing it from ``WEG2-FLIP-TAG`` lines after the fact; the
answer was NO -- every ``(rank, tag, direction)`` carried exactly ONE distinct
byte value across all 15 saves and 15 restores.  The need is flat; what is not
logged at all is the FREE side.  :meth:`RingNeedGuard.guard_tag` emits one
``WEG2-RING NEED`` line per tag per leg carrying need, free and their
difference, so the next boot answers that question from its own log instead of
from an archaeologist.

THE REFUSAL IS A TREND TEST, NOT A THRESHOLD, and it has to be.  A shortfall at
the start of a leg is NORMAL and is the whole design: S's acquires are funded by
W's releases as the two legs interleave.  weg2sb5e's D leg opened every one of
its 15 saves needing ~13.9 GiB against ~3.7 GiB free on nvml1 and completed
each time.  A guard that refused on ``need > free`` would have refused all 30
good flips.  What separates the good flips from the wedge is whether free is
RISING: the peer either is emptying the ring or it is not.  So the guard waits
only while it sees progress, and refuses as soon as progress stops -- in
``stall_probe_s`` (default 2.0 s), not in 110 s, and with the series printed.

WHAT THIS DOES NOT DO, stated so no reader assumes otherwise:

* It does not make the ring bigger and must not.  A strict serial L6 on
  weg2sb5e's numbers demands +2509 / +2986 / +2622 MiB = +8117 MiB total, which
  the host ledger cannot fund: the chosen arm sits ON its ``--store-min-gib``
  floor (``store=8 GiB``, ``reap-bound 8.64``), so paying the ring in shmem
  pushes the store under the floor and the next boot into a W21 refusal.
* It does not repair the missing wake leg.  On weg2sb5e's 31st flip the front
  logged ``WEG2-FLIP begin epoch=30 sleep=D wake=P`` and reached its
  ``gathered-legs`` stage, and group P then ran 1809 idle event-loop rounds
  without a single ``/resume_memory_occupation`` reaching its scheduler.  Why
  that leg did not arrive is a SEPARATE defect (the #1274 flip-stall family).
  This guard turns its consequence from a 110 s silent hang plus a misdirecting
  death line into a 2 s refusal that names the peer and prints the series.
* It never invents a free reading.  ``ring_stats()`` returning ``None`` means
  this boot published no ring at all -- an ABSENCE, not a zero -- and the guard
  stands down and says so rather than refusing on a number it does not have.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

#: The ring's granule, mirrored from ``TMS_RING_GRANULE_BYTES`` in
#: ``tms_csrc/host_ring.h``.  Every acquire is rounded UP to this, so a need
#: expressed in bytes and a need expressed in granules differ by up to one
#: granule per allocation -- which is exactly the rounding term a byte-level
#: comparison would miss.
RING_GRANULE_BYTES = 2 * 1024 * 1024

MIB = 1024 * 1024

#: How long the guard tolerates NO rise in free before calling the leg
#: unfunded.  Each ``host_ring.cpp`` wait slice is 250 ms and each peer tag
#: release frees hundreds of MiB at once, so 2.0 s is eight slices and several
#: whole tags -- long enough that a merely slow peer is never mistaken for an
#: absent one, short enough that the 110 s budget is never reached.
DEFAULT_STALL_PROBE_S = 2.0

#: The poll slice.  Matches the C++ waiter's own 250 ms so the two instruments
#: sample the same shared header at the same rate.
DEFAULT_SLICE_S = 0.25

#: The hard ceiling on one tag's wait, whatever the trend says.  A peer that
#: keeps freeing a granule at a time would otherwise extend the deadline
#: forever; this bounds the guard itself.  Well under the C++ acquire budget,
#: which is the point -- the refusal must land BEFORE the waiter's, or the
#: waiter's misdirecting text wins the race to the log.
DEFAULT_MAX_WAIT_S = 20.0


class Weg2HostRingUnfunded(RuntimeError):
    """W51: the sleeping leg needs host granules the peer is not releasing.

    Raised BEFORE the blocking acquire, so no waiter has been parked yet and
    the VRAM state of this tag is untouched.  Distinct from W31
    ``Weg2HostRingExhausted``, which is the C++ backstop reached only after the
    full acquire budget: reaching W31 means this guard was bypassed or absent.

    W51 was picked by enumerating the assigned set (W1..W50 are taken, see
    ``test_weg2_wcode_uniqueness_1263``), not by choosing a round number --
    which is how W31 came to name two different things.
    """


@dataclass(frozen=True)
class RingNeedSample:
    """One ``WEG2-RING NEED`` observation.

    ``delta_mib`` is ``need_mib - free_mib``: POSITIVE is the shortfall the
    peer still has to fund, NEGATIVE (or zero) means the acquire fits right now
    and cannot block.  The sign convention is stated on the log line itself
    because a bare ``delta`` reads either way.
    """

    tag: str
    card: str
    need_mib: int
    free_mib: int
    t_s: float

    @property
    def delta_mib(self) -> int:
        return self.need_mib - self.free_mib

    @property
    def fits(self) -> bool:
        return self.need_mib <= self.free_mib


def need_granules(nbytes: int, granule_bytes: int = RING_GRANULE_BYTES) -> int:
    """Granules an acquire of ``nbytes`` takes -- rounded UP, as the ring does.

    ``bytes_to_granules`` in ``host_ring.cpp`` rounds up per ALLOCATION, so a
    tag made of many allocations pays the rounding many times.  This function
    is the single-allocation form; callers that hold only a per-tag total get a
    LOWER BOUND from it, and the docstring of :meth:`RingNeedGuard.guard_tag`
    says so rather than letting the caller assume exactness.
    """
    if nbytes <= 0:
        return 0
    return (int(nbytes) + granule_bytes - 1) // granule_bytes


def need_mib(nbytes: int, granule_bytes: int = RING_GRANULE_BYTES) -> int:
    """``nbytes`` as the MiB the ring will actually charge (granule-rounded)."""
    return need_granules(nbytes, granule_bytes) * granule_bytes // MIB


class RingNeedGuard:
    """Per-leg tracker: emits the NEED series and refuses an unfunded leg.

    One instance per D2H leg.  It holds the leg's samples so a refusal can
    print the WHOLE series -- the flat-need / falling-free shape is only
    legible across tags, and a refusal that showed one row would send the
    reader back to the log to assemble the rest by hand.
    """

    def __init__(
        self,
        card_uuid: str,
        group: str = "",
        rank: int = -1,
        *,
        stall_probe_s: float = DEFAULT_STALL_PROBE_S,
        slice_s: float = DEFAULT_SLICE_S,
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
        granule_bytes: int = RING_GRANULE_BYTES,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Optional[logging.Logger] = None,
        # #1350d: WHICH LEG THIS SAMPLE BELONGS TO. Without it the NEED series
        # of a boot cannot be summed "per sleep leg" at all -- measured on
        # weg2xsn24, whose 60 P-lines and 30 D-lines carry no epoch and no
        # direction, so the per-card sums double-count across legs and are
        # useless as an image size. Set by the caller that opens the leg
        # (`set_leg`); "unknown" until it is, and printed either way.
        leg: str = "",
    ) -> None:
        self.card_uuid = card_uuid or "unknown"
        self.group = group
        self.rank = rank
        self.leg = str(leg or "unknown")
        self.stall_probe_s = float(stall_probe_s)
        self.slice_s = float(slice_s)
        self.max_wait_s = float(max_wait_s)
        self.granule_bytes = int(granule_bytes)
        self._clock = clock
        self._sleep = sleep
        self._log = log if log is not None else logger
        self.samples: List[RingNeedSample] = []
        self._explained = False

    # -- reading the ring ---------------------------------------------------

    def _free_mib(self, stats: Optional[dict]) -> Optional[int]:
        """Free MiB from a ``ring_stats()`` dict, or ``None`` for ABSENCE.

        ``None`` in, ``None`` out -- a boot with no ring published has no free
        reading, and turning that into 0 would make every leg look unfunded.
        A dict missing the counters is the same absence, not a zero.
        """
        if not stats:
            return None
        try:
            free = int(stats["granules_free"])
            granule = int(stats.get("granule_bytes") or self.granule_bytes)
        except (KeyError, TypeError, ValueError):
            return None
        return free * granule // MIB

    # -- the instrument -----------------------------------------------------

    def set_leg(self, epoch: object, src: str = "", dst: str = "") -> str:
        """#1350d: name the leg every following NEED line belongs to.

        ``leg=<epoch>/<src>-><dst>``.  Called once by whoever opens a leg; the
        guard never derives it, because deriving a leg identity from a counter
        this class owns would be a second bookkeeping beside the front's epoch.
        """
        self.leg = f"{epoch}/{src or '?'}->{dst or '?'}"
        return self.leg

    def record(self, tag: str, need_mib_: int, free_mib_: int) -> RingNeedSample:
        """Append one sample and emit its ``WEG2-RING NEED`` line."""
        sample = RingNeedSample(
            tag=tag,
            card=self.card_uuid,
            need_mib=int(need_mib_),
            free_mib=int(free_mib_),
            t_s=self._clock(),
        )
        self.samples.append(sample)
        # The five fixed fields are on EVERY line, so the series parses without
        # the prose.  The prose is on the FIRST line of a leg only: a leg emits
        # one line per tag plus one per observed release, and repeating 600
        # characters of definition ~27 times per rank per flip would bury the
        # series it exists to make readable.
        if self._explained:
            self._log.info(
                "WEG2-RING NEED tag=%s card=%s leg=%s need_mib=%d free_mib=%d "
                "delta_mib=%d (group=%s rank=%d; fields defined on this leg's "
                "first WEG2-RING NEED line)",
                tag, self.card_uuid, self.leg, sample.need_mib, sample.free_mib,
                sample.delta_mib, self.group or "?", self.rank,
            )
            return sample
        self._explained = True
        self._log.info(
            "WEG2-RING NEED tag=%s card=%s leg=%s need_mib=%d free_mib=%d delta_mib=%d "
            "(group=%s rank=%d; need is THIS tag's bytes rounded up to the ring's "
            "%d MiB granule -- a LOWER bound, because the ring rounds per "
            "ALLOCATION and a tag is many; free is granules_free x granule read "
            "from the shared header at this instant, NOT a device reading; "
            "delta = need - free, POSITIVE = the shortfall the waking peer still "
            "has to release, and a positive delta is NORMAL under the "
            "interleaved leg form -- it is a delta that stops SHRINKING that is "
            "the fault, which is what W51 tests)",
            tag, self.card_uuid, self.leg, sample.need_mib, sample.free_mib,
            sample.delta_mib, self.group or "?", self.rank, self.granule_bytes // MIB,
        )
        return sample

    def format_series(self) -> str:
        """The whole leg's series, one row per sample, for a refusal message."""
        if not self.samples:
            return "(no samples: the leg refused before its first tag)"
        t0 = self.samples[0].t_s
        rows = [
            "tag=%s need_mib=%d free_mib=%d delta_mib=%d t+%.2fs"
            % (s.tag, s.need_mib, s.free_mib, s.delta_mib, s.t_s - t0)
            for s in self.samples
        ]
        return " | ".join(rows)

    # -- the refusal --------------------------------------------------------

    def guard_tag(
        self,
        tag: str,
        tag_bytes: int,
        stats_fn: Callable[[], Optional[dict]],
        peer_hint: str = "",
    ) -> None:
        """Emit this tag's NEED line and refuse if the leg is unfunded.

        Called immediately before the ``pause(tag)`` that enters the blocking
        C++ acquire.  ``tag_bytes`` is the tag's TOTAL host bytes, so the need
        derived from it is a lower bound on what the acquire will charge (the
        ring rounds per allocation); the guard is therefore conservative in the
        safe direction -- it can under-state a shortfall, never invent one.

        Returns normally when the acquire can proceed: either it fits now, or
        free is still rising and the interleave is doing its job.  Raises
        :class:`Weg2HostRingUnfunded` when free has not risen for
        ``stall_probe_s``, or when ``max_wait_s`` is reached with the need still
        unmet.  Stands down silently (no refusal, no wait) when this boot
        published no ring.
        """
        stats = stats_fn()
        free = self._free_mib(stats)
        if free is None:
            # #1350d: NOT-APPLICABLE, NOT `need_mib=0`. On the `exchange` arm
            # there is no ring, and a 0 here is indistinguishable from "this tag
            # has nothing to save" -- measured on weg2xsn24, where 32 of group
            # P's 60 NEED lines read `need_mib=0` for that reason alone and
            # would have sized an anchor at half the P group. The SAVED BYTES
            # are printed instead, under their own field name, so the line is
            # still a measurement of the tag and never of the ring.
            self._log.info(
                "WEG2-RING NEED tag=%s card=%s leg=%s need_mib=NOT-APPLICABLE "
                "saved_mib=%d free_mib=n/a delta_mib=n/a "
                "-- no host ring published on this boot (ring_stats() returned no "
                "counters), so there is no free reading to compare against and the "
                "guard stands down; this is an ABSENCE, never a zero. `saved_mib` "
                "is THIS tag's own bytes and IS a measurement -- it is what the leg "
                "puts away for this tag whether or not a ring exists",
                tag, self.card_uuid, self.leg,
                int(tag_bytes) // MIB,
            )
            return

        want = need_mib(tag_bytes, self.granule_bytes)
        sample = self.record(tag, want, free)
        if sample.fits:
            return

        # It will block.  The only question worth asking is whether the peer is
        # emptying the ring.  Wait while free RISES, refuse when it stops.
        t_start = self._clock()
        best = free
        last_progress = t_start
        while True:
            now = self._clock()
            if now - t_start >= self.max_wait_s:
                self._refuse(tag, want, best, peer_hint, now - t_start,
                             "the ceiling on one tag's wait")
            if now - last_progress >= self.stall_probe_s:
                self._refuse(tag, want, best, peer_hint, now - t_start,
                             "free has not risen for %.2f s" % (now - last_progress))
            self._sleep(self.slice_s)
            free_now = self._free_mib(stats_fn())
            if free_now is None:
                # The ring went away under us.  Not this guard's verdict to
                # give -- let the acquire path report what it sees.
                return
            if free_now > best:
                best = free_now
                last_progress = self._clock()
                self.record(tag, want, free_now)
                if want <= best:
                    return

    def _refuse(
        self,
        tag: str,
        want: int,
        best: int,
        peer_hint: str,
        waited_s: float,
        why: str,
    ) -> None:
        raise Weg2HostRingUnfunded(
            "W51 Weg2HostRingUnfunded card=%s tag=%s need_mib=%d free_mib=%d "
            "delta_mib=%d waited_s=%.2f group=%s rank=%d peer=%s -- %s, so the "
            "waking peer is not releasing into this ring and no further waiting "
            "can fund this acquire. REFUSED BEFORE the blocking acquire: no "
            "waiter was parked, this tag's device bytes are untouched, and the "
            "C++ acquire budget (W31 Weg2HostRingExhausted, 110 s on weg2sb5e) "
            "was NOT entered. The need series for this leg: %s. Read the series "
            "before the launch arithmetic: a FLAT need across flips with free "
            "not rising is the peer failing to release (the flip's wake leg did "
            "not run), while a RISING need is a real per-tag creep and belongs "
            "to L6."
            % (
                self.card_uuid, tag, want, best, want - best, waited_s,
                self.group or "?", self.rank, peer_hint or "unknown", why,
                self.format_series(),
            )
        )
