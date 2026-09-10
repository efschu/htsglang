"""The #721 host ledger for the Weg-2 six-process shape, priced at BOTH moments.

Every term below is NAMED with its provenance, printed by :func:`format_lines`,
and none of them is a spec number copied blind: they are the measured posts of
the campaigns of 2026-09-06 (WEG2_BUILD_DECISIONS_0906.md section 1d/1e and
CAMPAIGN_b0_0906.md / CAMPAIGN_a_0906.md) or the standing #721 constants the
line already runs under (``planner/weg1_host_sizing.py``).

Two moments (record section 1c B2, spec section 4.2.3):

* ``launch`` -- group D is loading (LOAD_TRANSIENT charged) while group P is
  already dormant with its cpu-backup image resident.
* ``run`` -- both groups exist, the DORMANT group's cpu backup is resident and
  the awake group runs at its serving heap.  With the #1233 one-backup flip
  (record 1h: the patched torch_memory_saver frees a chunk's host image on
  resume, and the front interleaves ``src.pause(weights_k)`` with
  ``dst.resume(weights_k)``) the run moment used to be charged ONE image plus
  ONE chunk.  C19 (2026-09-07, WEG2_FLIPCOST_SPEC_0907 R7/section 7) REPLACES
  that with the SHARED HOST RING, and the replacement is what deletes three
  terms rather than adding a fourth:

  * ``BACKUP_P_BYTES`` / ``BACKUP_D_BYTES`` -- two #809-census constants,
    measured stale by ~3.4 GiB against the per-card table the boots actually
    log, and encoding a model (one private pinned image per group) that the
    ring removes.  DELETED.
  * ``chunk_gib`` (``image / N``) -- the residency proxy for "one chunk in
    flight".  With one shared region there is no chunk beside the image: the
    region IS the whole host weights cost and it is charged once.  DELETED.

  What is charged instead, per :mod:`sglang.srt.weg2.ring_table`:

  * launch = ``Sigma span1`` = ``Sigma image_P(c)``, because the launcher's
    ``sleep(P)`` is the first pause and only span 1 is registered then (R7:
    registering ``Sigma H`` at the launch moment costs the M=1200 arm 2.98 GiB
    it does not have and the ladder falls to M=600).
  * run = ``Sigma H`` = ``Sigma max_g image_g(c)``.

  Both come from the PREVIOUS boot's own lines with a provenance string; there
  is no fallback constant, because a constant is exactly what the campaign
  measured wrong twice.

FIX 5 (2026-09-07, after boot weg2dk5's host cgroup OOM) contributed two of
those deletions' siblings, and both survive the move onto the ring:

* the DENOMINATOR: ``base`` is now the tighter of the meminfo reading and the
  CGROUP reading (``ceiling - memory.current - cli_reserve``), and the printed
  line names which one bound.  The reaper watches ``memory.current``; weg2dk5
  was reaped with ``MemAvailable`` at 23.96 GB, above the #721 floor, so the
  quantity this ledger measured was never the quantity that governs;

  FIX 6 (2026-09-07) repairs that SAME denominator once more, in the other
  direction: ``memory.current`` counts the page cache, which the kernel hands
  back under pressure instead of killing for, so charging the whole reading as
  spent refuses room nobody occupies.  Measured twice on metal: at weg2dk5's
  OWN launch readings the whole ladder refused (the branch could not launch
  from the state its last boot launched from), and on the live box the chosen
  arm moved S=1 M=1200 -> M=600 unannounced.  What is charged now is the
  NON-RECLAIMABLE part of the reading -- ``memory.current`` minus
  :func:`cg_reclaimable_bytes` -- with both the reclaimable term and the
  resulting base printed by name, and the same correction runs inside
  :meth:`Arm.predicted_run_peak_gib`: an origin that carries cache and a
  watermark that does not are not comparable quantities;
* ``HOST_HEADROOM_GIB`` (#1232) is DELETED, not shrunk: it was a constant
  fitted to the gap between the #721 floor and the level at which this box
  actually OOMs, i.e. a compensation layer for the wrong denominator, and the
  upstream-minimal law makes a defect found in a compensation layer a deletion
  candidate rather than a repair order.

Fix 5's THIRD change -- the run moment priced as ``one image + the measured
FLIP TRANSIENT`` -- does NOT survive, and its removal is C19's, not a
regression.  It priced the per-allocation flip, where the interleave's peak was
both images partially resident at once.  The shared host ring removes that
quantity: the region is preallocated at ``Sigma H`` and the legs copy THROUGH
it, so ``run`` charges ``Sigma H`` ONCE and that IS the peak.  Charging both
would be the same bytes twice.  :data:`FLIP_HOST_TRANSIENT_GIB` survives as the
MEASURED DATUM of the old form (it is what A1-3 compares against), never again
as a term.

FIX 8 (2026-09-08, after boot weg2dk7's Q2 measurement) changes WHERE the image
term comes from and WHICH quantity picks the arm:

* the IMAGE TERM IS MEASURED, not summed from the weight tags.  weg2dk7 sampled
  the dormant group's per-rank ``RssShmem`` with P asleep and D awake:
  **38.63 GiB against the 28.83 GiB this ledger charged**, a +9.80 GiB (+34 %)
  under-charge on a single image.  The cause is named rather than fitted:
  everything with ``enable_cpu_backup`` is in the image (draft weights, graph
  pools, workspaces, embeddings), not only the ``weights_*`` tags the #809
  census counted.  :func:`resolve_image_terms` takes the previous boot's own
  measurement when one exists, falls back to that named dk7 reading for P, and
  REFUSES to claim a smaller image for D than the one measured for P.  ON THE
  RING that measurement is not charged directly: it is what SIZES H(c), and
  :func:`price` charges the resulting ``Sigma H`` once.  Same measurement, one
  charge -- :mod:`sglang.srt.weg2.ring_table` reads this module's sidecar for
  it, so there is ONE reader of the dormant image on the line;
* the RUN PEAK IS THE ARM SELECTOR (W21 ``Weg2HostRunPeakRefused``), not an
  advisory beside it.  Two boots died and one could not flip while the advisory
  said "below";
* the ORIGIN of that prediction is the RUN moment, not the launch moment.
  ``memory.current`` at launch is a FLOOR: weg2dk7 launched into 15.36 GiB, was
  handed the bigger arm for it, and idled at 90.1-94.6 GiB with an EMPTY store,
  while weg2dk6 launched into 46.80 GiB, took the smaller arm and died at 96.06.
  A quieter launch bought the tighter boot -- the selector was anti-correlated
  with safety.  :func:`run_origin_gib` charges the LARGER of the launch reading
  and the measured run-moment residual (what the box holds that this ledger's
  own term list does not name).

The arm ladder (record section 1c B2: "if the ledger refuses at S=2 the
launcher sizes S=1 and prints why; if it refuses at S=1 the boot REFUSES") is
extended by the mamba host pool ``M`` in the same spirit: every arm is printed,
the first fundable one is taken, and a shrink is never silent.  If no arm funds
the store floor the launch is refused by name (W20 ``Weg2HostLedgerRefused``);
if an arm funds both moments but its predicted run peak is not below the
observed reap point, the refusal is W21 (:class:`Weg2HostRunPeakRefused`).

USER RULING 2026-09-07 06:2xZ (record section 1g): the canonical page store --
the ONE carrier -- lives on a RAM-backed filesystem sized by this ledger's
leftover.  The operator's own count was "118 GiB minus 55 GiB of backups, the
L2 pools, ~10 GiB of Claude CLIs and the 16 GiB floor -> ~20 GiB".  That count
carries no heap term; the measured heaps (b0: 3.385 GiB per awake rank, (a):
2.36 GiB per dormant rank) are 17.2 GiB for six ranks and this ledger charges
them, which is why the printed leftover is smaller than the expectation.  A
ledger that omitted a measured term to match an expectation would be the
indicator-law violation the record forbids.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

GIB = float(2**30)
GB = 1e9

#: #721 floor (host_ledger_preflight.sh FLOOR_G, weg1_host_sizing.FLOOR_BYTES).
FLOOR_GIB = 16.0
#: HOST_HEADROOM_GIB (#1232, 16 GiB) IS DELETED HERE, fix 5, and the deletion
#: is the fix -- not a shrink.  Its own docstring named it "the measured gap"
#: between the #721 floor and the level at which the box actually OOMs, i.e. a
#: compensation constant for the fact that this ledger was reading the wrong
#: denominator: /proc/meminfo, which the reaper does not watch.  Boot weg2dk5
#: (2026-09-07) reaped six processes with MemAvailable still at 23.96 GB --
#: 7.9 GiB ABOVE the #721 floor alone -- so the floor never bound and the
#: headroom on top of it was fitted to a quantity that does not govern.  What
#: governs is ``memory.current`` against the cgroup's ceiling, and both are
#: readable (:func:`read_cgroup`).  Upstream-minimal law: a compensation layer
#: whose defect is found is a deletion candidate, not a repair order -- the
#: gap is now measured directly instead of estimated once and carried.
#: Operator list, record section 1g: "the Claude CLIs (~10 GiB)".  Charged
#: ONCE against MemTotal (they are live RSS, so MemAvailable already nets out
#: whatever they hold right now; the term keeps their room when they grow).
CLI_RESERVE_GIB = 10.0
#: b0: awake steady RssAnon peaks 3.385 / 3.038 / 2.996 GiB (L-bf2), max taken.
#:
#: RE-CHECKED 2026-09-08 against the live sb4 base and CONFIRMED CORRECT -- the
#: ~6.5 GiB/rank figure that circulated in the weg-2 briefings is NOT in this
#: ledger and must not be put back. That figure was an INSTRUMENT ERROR: CUDA
#: pinned host memory maps `/dev/zero` MAP_SHARED and the kernel counts it as
#: RssShmem, NOT RssAnon, so anon + pinned + heap were being summed as "anon".
#: Measured on the six live ranks: RssAnon 3395 / 3014 / 2346 MiB (P) and
#: 3240 / 2959 / 2896 MiB (D) -- i.e. 2.3-3.4 GiB, which is exactly what this
#: constant already carries.
#:
#: NO TERM DOUBLE-COUNTS, checked rather than assumed: the pinned half lives in
#: the `/dev/zero` shmem mappings and is priced HERE as `rings` + `anchors`
#: (RING_*_MULT_GB_PER_S, ANCHORS_AT_2400_BYTES), and the host weights image as
#: `Sigma H` (ring_bytes). Those are shmem currency; this constant is anon
#: currency. Summing them is correct; replacing this one with 6.5 would charge
#: the pinned buffers twice.
HEAP_AWAKE_GIB = 3.385
#: Campaign (a): RssAnon "flat at 2.36 GiB" on the dormant tp=1 rank.
HEAP_DORMANT_GIB = 2.36
#: BACKUP_P_BYTES / BACKUP_D_BYTES ARE DELETED HERE (C19), and the deletion is
#: the fix.  They were #809-census constants for a model -- one private pinned
#: image per group -- that the shared host ring removes; they were also stale by
#: ~3.4 GiB against the per-card table every boot logs for itself.  The host
#: weights cost now enters :func:`price` as ``ring_bytes`` / ``ring_span1_bytes``,
#: solved by :mod:`sglang.srt.weg2.ring_table` from the previous boot's own
#: WEG2-CHUNK-BYTES / WEG2-FLIP-TAG lines and printed with its provenance.
#: There is deliberately NO fallback constant: a boot with no measured table
#: refuses (W20) rather than pricing the flip from a number nobody measured.

#: #1233 boot weg2dk5 (2026-09-07): THE FLIP'S OWN HOST TRANSIENT OF THE OLD,
#: PER-ALLOCATION FORM.  NOT A TERM OF :func:`price` -- C19's shared host ring
#: removed the quantity it measures, and charging it beside ``Sigma H`` would
#: charge the same bytes twice.  It is kept because it is a MEASUREMENT of this
#: box with its population named, and because it is the figure FLIPCOST A1-3
#: compares the ring against (image + transient ~48.7 GiB vs Sigma H ~36 GiB).

#: Spec section 2.6, #809 FLIP IMAGE PREFETCH census: PP image 30.96 GB,
#: TP image 29.15 GB per group.
#:
#: FIX 8 RENAME, and the rename IS the finding: these are the WEIGHT-TAG byte
#: sums, and the dormant image is NOT that.  Boot weg2dk7 measured the sleeping
#: group's per-rank ``RssShmem`` at 38.63 GiB against the 28.83 GiB this pair
#: priced -- everything built with ``enable_cpu_backup`` is in the image (draft
#: weights, cuda-graph pools, workspaces, embeddings), while the census counted
#: the ``weights_*`` tags alone.  They are KEPT (the derivation is a real
#: census, not a hand number) and they are no longer CHARGED as the image: they
#: are the reference the measured ``extra`` term is stated against, so a reader
#: sees both halves of the correction instead of one replaced number.
WEIGHT_TAGS_P_BYTES = 30.96 * GB
WEIGHT_TAGS_D_BYTES = 29.15 * GB
#: MEASURED, boot weg2dk7 (`/spinning/gpu-arb/weg2/BOOT_weg2dk7_0907.md`, the
#: attributed quiet rows 00:29:02Z and 00:31:07Z, tip 402a2df856): with group P
#: ASLEEP and group D AWAKE, the sum of ``RssShmem`` over P's five worker pids
#: was **38.63 GiB** in both rows.  The same rows carry cgroup ``shmem``
#: 48.33 GiB of which **44.56 GiB has no backing file** -- torch_memory_saver's
#: CPU backup is anonymous ``MAP_SHARED``, so ``df``/``du`` on /dev/shm see
#: nothing and an attribution that looks for the image AS A FILE concludes it is
#: not resident.  This is the dormant image of a PP group on this checkpoint.
DK7_DORMANT_IMAGE_P_GIB = 38.63
#: The same boot's quiet ``memory.current`` (row 00:29:02Z), with the store
#: tmpfs holding 0.00 GiB of its 9 GiB and ZERO flips run.  Used to derive the
#: RUN-MOMENT residual below.  The row carries no reclaimable column; charging
#: the whole reading as non-reclaimable makes the derived residual LARGER, i.e.
#: the conservative direction, and that is stated rather than assumed.
DK7_QUIET_CG_CURRENT_GIB = 90.10
DK7_QUIET_STORE_USED_GIB = 0.00
#: The arm weg2dk7 actually ran while measuring the two figures above.
DK7_ARM_S_GB = 1
DK7_ARM_M_MIB = 1200
DK7_PROVENANCE = (
    "BOOT_weg2dk7_0907.md quiet row 2026-09-08T00:29:02Z (boot weg2dk7 @ 402a2df856), "
    "per-rank RssShmem of the SLEEPING group"
)
#:
#: The old term was ``backup_resident / weight_chunks`` -- "one chunk in flight"
#: -- a RESIDENCY model of an interleave whose PEAK is both images partially
#: resident plus the wake's anon working set.  Same "peak is not residency"
#: class (CONTRADICTIONS_REGISTER C32, #631 seam-peak table) that fix 4 closed
#: on the DEVICE axis; this closes it on the HOST axis.
#:
#: MEASURED, with its population and its denominator named: for each of the TEN
#: ``WEG2-FLIP begin`` windows of boot weg2dk5, ``cg_current`` at the interleave
#: peak minus the minimum ``cg_current`` in the 30 s quiet window before that
#: begin line (``/spinning/gpu-arb/memts_weg2_weg2dk5.csv``, 5 s cadence):
#:   0.03 / 9.20 / 3.39 / 6.12 / 6.83 / 9.97 / 4.32 / 5.21 / 3.81 / 8.58 GiB.
#: A PEAK term takes the MAX of its population, never the mean -> 9.97 GiB.
#: (Flip 0's 0.03 is the 5 s sampler missing that interleave's peak, not a free
#: flip; it is left in the population rather than dropped, and it cannot move a
#: max.)
#:
#: THE FIXED-vs-CUMULATIVE DISCRIMINATOR IS SETTLED BY THIS SERIES, so the next
#: boot does not have to spend itself on it (BOOT_weg2dk5_0907.md fix shape 3):
#: the transient does NOT grow with flip index (flip 5 is the max, flips 6-9 are
#: 4.32/5.21/3.81/8.58) -- it is per-flip FIXED.  What DOES creep is the quiet
#: baseline, 83.72 -> 87.98 GiB over the boot, and that creep is the store tmpfs
#: filling (quiet Shmem 47.73 -> 52.68 GiB = +4.95, against 5.1 GiB of store
#: content at the death): a standing cost the ledger budgeted as ``store_gib``
#: at the time, not a leak.  #1236 RETIRED that budget line: the store is on
#: disk now, so this creep is a property of the tmpfs form that no longer runs
#: -- the reading is kept because the FLIP TRANSIENT it measures is still the
#: subject of this constant, but the store half of its explanation is history.
FLIP_HOST_TRANSIENT_GIB = 9.97
#: ...AND IT IS THE PRE-RING VALUE. With the shared registered host ring the
#: transient collapses, because the legs copy THROUGH a preallocated region
#: instead of each tag allocating a fresh cudaMallocHost:
#:   weg2rg2 (2026-09-08 03:00Z, 8c7f5d8d00)  +1.10 GiB  "drops from +9.97 to +1.1"
#:   weg2rg3 (2026-09-08 04:5xZ)              +2.88 GiB  13 flips; the record calls
#:                                                       its 10 s sampler a LOWER bound
#: The MAX is taken, not the mean: the margin must cover the worst recorded flip
#: of this form, and rg3's is explicitly a lower bound, so this term is itself a
#: lower bound on the true worst case.
RING_ERA_FLIP_TRANSIENT_GIB = {"weg2rg2": 1.10, "weg2rg3": 2.88}

#: RUN-PEAK UNDER-PREDICTION, the term the pre-ring transient was accidentally
#: standing in for. The estimator does not miss the flip; it misses the STEADY
#: STATE, and that is what actually killed the boots:
#:   weg2dk5  predicted 92.89, reached 95.90               +3.01   (pre-ring)
#:   weg2dk6  predicted 91.23, sampled 96.06 / peak 98.90  +4.83 / +7.67 (pre-ring)
#:   weg2rg6  predicted 93.66, peak 93.55                  -0.11   (ring era)
#:   weg2sb4  predicted 91.44, idled 96.03-96.60           +5.16   (ring era)
#: Only the RING-ERA rows are used -- the form is what the bound is for -- so the
#: term is sb4's +5.16. Measured over sb4's own idle window (boot 16:29Z, 96.60
#: reached ~17:30Z, ~60 min), which is why the drift term below is charged only
#: for the window length BEYOND that.
#:
#: THIS TERM IS AN EMPIRICAL CATCH-ALL. It is (measured - predicted), so it
#: already contains the idle anon drift of those boots, the foreign desk load
#: those boots carried, and the estimator's own error. Re-adding any of those in
#: full would double-charge; that is why the drift term is charged only on the
#: EXCESS window and the foreign term is measured and PRINTED but not re-added.
#: FIX 4 adds the FIRST HONEST RING-ERA SAMPLE: weg2sb5c predicted 86.18 and
#: measured 87.72 non-reclaimable = +1.54, and unlike every earlier row BOTH
#: sides are in the same currency (sb4's +5.16 compares a prediction against a
#: raw memory.current that happened to carry ~0 file cache at idle -- true, but
#: true by luck rather than by construction). The rule is MAX-over-samples, so
#: the term does not move: 5.16 still binds. It is recorded because the next
#: in-currency sample is what will eventually retire the sb4 row, and because a
#: provenance table that only keeps the maximum cannot show that the estimator
#: is improving.
RUN_PEAK_RESIDUAL_GIB = {"weg2sb4": 5.16, "weg2rg6": -0.11, "weg2sb5c": 1.54}
RESIDUAL_WINDOW_MIN = 60.0

#: #1269 / user order 2026-09-08 ("kein uebertreten mehr der schwelle. fuehrt
#: nur zum absturz"): THE REAP WATERMARK IS A HARD BOUND WITH A NAMED MARGIN.
#:
#: WHICH SAMPLES COUNT. A reap sample is a reading where the KERNEL reaped --
#: an `oom_kill` delta in the same row. Two qualify and one does not:
#:   weg2dk5  95.90 GiB  oom_kill 18 -> 24, /proc/vmstat 60 -> 66   COUNTS
#:   weg2dk6  96.06 GiB sampled / 98.90 kernel peak, "died the same death"
#:                                                                 COUNTS
#:   weg2sb4  96.60 GiB  NO kernel OOM -- the OPERATOR killed it    DOES NOT
#: sb4 is evidence that the box is unhappy above the mark, not evidence of
#: where the kernel reaps; counting a human's patience as a kernel threshold
#: would move the watermark UP on the strength of a boot that never reaped.
#: The bound is the LOWEST kernel sample, so dk5 remains authoritative.
REAP_SAMPLES_GIB = {
    "weg2dk5": 95.90,   # oom_kill 18 -> 24 in the same memts row
    "weg2dk6": 96.06,   # sampled; 98.90 kernel peak; same death
}
REAP_SAMPLE_EXCLUDED = {
    "weg2sb4": (96.60, "operator kill, no kernel OOM -- not a reap sample"),
}

#: One ring granule: the smallest unit a flip copy moves (ring_table's C3/C4
#: form issues 2 MiB granules). It is the FLOOR of the transient term, so the
#: margin can never collapse to zero when a record is missing.
RING_GRANULE_GIB = 2.0 / 1024.0

#: Idle anon drift, MiB/min, used until a boot carries WEG2-IDLE-CENSUS
#: `d_anon_mib_per_min`. Conservative default from the sb4 table measured
#: 2026-09-08 over two /proc samples 140 s apart: P PP0 +5.4, PP1 +5.4,
#: PP2 +0.0, each D rank +2.8 => +19.0 MiB/min over the six ranks, against an
#: independently observed +21.3 MiB/min. The SUM is the right term: the drift
#: is charged to one cgroup, not per rank.
IDLE_ANON_DRIFT_MIB_PER_MIN_DEFAULT = 19.0

#: Planned window length in minutes that the drift is integrated over. The
#: operator hands out 90-minute windows (GPU-Fenster-Rotation), so a boot is
#: expected to stand for at least that long without crossing the mark.
PLANNED_WINDOW_MIN_DEFAULT = 90.0


@dataclass(frozen=True)
class Margin:
    """The named margin between the predicted peak and the reap watermark.

    NOT a hand number and NOT a safety factor. Three measured terms, and the
    care is in what is NOT added twice:

    ``transient``  the worst recorded flip transient OF THIS FORM. Ring-era, so
                   ~2.9 GiB, not the 9.97 GiB of the per-allocation form.
    ``residual``   max (measured peak - predicted peak) over the ring-era boots.
                   An EMPIRICAL CATCH-ALL: it already contains those boots' idle
                   drift, their foreign desk load, and the estimator's error.
    ``drift``      charged ONLY on the window BEYOND the one the residual was
                   measured over, because inside that window it is already in
                   the residual.
    ``foreign``    reserve for desk load ARRIVING AFTER arm time. Default 0, and
                   that is deliberate -- see below.

    WHY ``foreign`` DEFAULTS TO ZERO, which is the opposite of a hand-wave. The
    cgroup that reaps is shared with the Claude sessions, their pytest runs,
    worktrees and dry-runs, and that load is real (measured on the sb4 base:
    cgroup anon 33.49 GiB against 21.78 GiB of sglang RssAnon = 11.71 GiB
    foreign, which independently validates the 10 GiB CLI_RESERVE_GIB the
    ledger already carried -- in the WRONG denominator, against MemTotal rather
    than against the cgroup the reaper watches).

    But that load AT ARM TIME IS ALREADY IN THE ORIGIN: :func:`run_origin_gib`
    returns a ``memory.current`` reading, and a cgroup reading counts every
    process in the cgroup, Claude included. Re-adding it as a margin term would
    charge it twice. What is NOT in the origin is desk work that arrives LATER,
    and that is unbounded by construction -- a boot cannot reserve against how
    much an operator will run tomorrow. So it is MEASURED and PRINTED (so a
    breach can be attributed) and left to the RUNTIME guard, which is what a
    guard is for: the boot bound reserves what is predictable, W22 catches what
    is not.
    """

    transient_gib: float
    residual_gib: float
    drift_gib: float
    foreign_gib: float
    drift_mib_per_min: float
    window_min: float
    transient_source: str
    residual_source: str
    drift_source: str
    foreign_source: str

    @property
    def total_gib(self) -> float:
        """The BOOT margin. Kept as the default name because W21 and the store
        sizing -- the two callers that grade a PREDICTION -- are its users."""
        return self.boot_total_gib

    @property
    def boot_total_gib(self) -> float:
        return (
            self.transient_gib + self.residual_gib + self.drift_gib + self.foreign_gib
        )

    @property
    def runtime_total_gib(self) -> float:
        """The RUNTIME margin: the boot margin MINUS the model-error term.

        #1269 FIX 4, and it is a design correction rather than a relaxation.
        ``residual`` is (measured - predicted): it exists to cover how wrong the
        PREDICTION can be. W21 grades a prediction, so it must carry it. W22
        grades a MEASUREMENT -- `memory.current` net of reclaimable cache, the
        real number -- and a measurement has already realised whatever error the
        residual was reserving for. Charging it there subtracts the same error
        twice and refuses a box that is nowhere near the reap point.

        Boot weg2sb5c is what made this concrete: it was refused at
        87.67 GiB non-reclaimable against the 87.30 boot bound -- crossed by
        0.37 -- while the actual reap point sat **8.2 GiB** away. The transient
        and the drift still belong here: neither has happened yet at the moment
        of the reading, so both are still future spend the box must have room
        for.
        """
        return self.transient_gib + self.drift_gib + self.foreign_gib

    def _drift_note(self) -> str:
        return (
            f"{self.drift_mib_per_min:.1f} MiB/min x "
            f"{max(0.0, self.window_min - RESIDUAL_WINDOW_MIN):.0f} min beyond the "
            f"residual's own {RESIDUAL_WINDOW_MIN:.0f} min, {self.drift_source}"
        )

    def terms(self, scope: str = "boot") -> str:
        """The terms of one bound. ``scope`` is 'boot' or 'runtime'; the runtime
        form OMITS the residual and says so, so a reader of a W22 line can see
        that the omission was deliberate and not a missing term."""
        head = (
            f"transient {self.transient_gib:.2f} [{self.transient_source}] "
            f"+ drift {self.drift_gib:.2f} [{self._drift_note()}] "
            f"+ foreign {self.foreign_gib:.2f} [{self.foreign_source}]"
        )
        if scope == "runtime":
            return (
                head + f" GiB; residual {self.residual_gib:.2f} DELIBERATELY NOT "
                "CHARGED here -- it is model error and this bound grades a "
                "MEASUREMENT, not a prediction (#1269 fix 4)"
            )
        return (
            f"transient {self.transient_gib:.2f} [{self.transient_source}] "
            f"+ residual {self.residual_gib:.2f} [{self.residual_source}] "
            f"+ drift {self.drift_gib:.2f} [{self._drift_note()}] "
            f"+ foreign {self.foreign_gib:.2f} [{self.foreign_source}] GiB"
        )


def measure_foreign_anon(
    cgroup_anon_bytes: Optional[int], own_pids: Sequence[int]
) -> Tuple[Optional[float], float, str]:
    """Non-sglang anon in the shared cgroup: (foreign GiB, sglang GiB, source).

    ``cgroup anon - sum(RssAnon of the boot's own pids)``. The cgroup that reaps
    is shared with the Claude sessions and their desk work, and that share is
    NOT small: the boot agent counted 9 ``.lxc`` oom_kills that were Claude
    CLIs, and a single desk probe reading the checkpoint added +815 MiB inside
    sb4's own idle window. A breach that this term explains is a breach the
    operator caused, and it must be named as such rather than charged to sglang.
    """
    own = 0
    seen = 0
    for pid in own_pids:
        try:
            with open(f"/proc/{pid}/status", "rb") as fh:
                for raw in fh:
                    if raw.startswith(b"RssAnon:"):
                        own += int(raw.split()[1]) * 1024
                        seen += 1
                        break
        except OSError:
            continue
    own_gib = own / GIB
    if cgroup_anon_bytes is None:
        return None, own_gib, "cgroup memory.stat anon unreadable -- foreign share unknown"
    foreign = float(cgroup_anon_bytes) / GIB - own_gib
    return (
        foreign,
        own_gib,
        f"cgroup anon {float(cgroup_anon_bytes) / GIB:.2f} GiB - sglang RssAnon "
        f"{own_gib:.2f} GiB over {seen}/{len(own_pids)} readable pids",
    )


def split_by_baseline(
    anon_now_bytes: Optional[int], anon_preboot_bytes: Optional[int]
) -> Tuple[Optional[float], Optional[float], str]:
    """(sglang GiB, foreign GiB, source) from the PRE-BOOT anon baseline.

    #1269 fix 3. The previous form, ``cgroup anon - sum(RssAnon of own pids)``,
    is ARITHMETICALLY INVALID, and boot weg2sb5b printed the proof:

        SPLIT sglang=61.48 foreign=-30.91 GiB anon
              [cgroup anon 30.57 - sglang RssAnon 61.48 over 111/111 pids]

    ``cgroup anon`` counts each physical page ONCE; ``sum(RssAnon)`` counts a
    shared page once PER PROCESS that maps it, and six ranks plus ~105 forked
    workers share a great deal. The two are not subtractable, the difference
    went negative -- which is impossible -- and so the line's conclusion
    ("sglang dominates") was unsupported even if it happened to be true.

    The pre-boot reading IS subtractable, being the same quantity at an earlier
    time: whatever anon the cgroup held before this boot existed is foreign to
    it, and the rise since is the boot's. sb5b measured 10.05 GiB pre-boot,
    exactly as its checklist asked, and the guard did not use it.

    HONEST LIMIT, stated rather than papered over: this attributes by TIME, not
    by OWNER. Desk work started after the baseline is charged to sglang. That is
    the conservative direction for a guard whose job is to tear down -- it never
    under-reports the boot's own share -- but it is not an ownership
    measurement, and a later foreign spike cannot be separated from the boot's
    own growth by this instrument.
    """
    if anon_now_bytes is None or anon_preboot_bytes is None:
        return None, None, "no pre-boot anon baseline -- split not computable"
    foreign = float(anon_preboot_bytes) / GIB
    sglang = float(anon_now_bytes) / GIB - foreign
    if sglang < 0.0:
        return (
            0.0,
            float(anon_now_bytes) / GIB,
            f"anon fell BELOW the pre-boot baseline {foreign:.2f} GiB (foreign "
            "work exited); the boot's own share is not separable here",
        )
    return (
        sglang,
        foreign,
        f"cgroup anon now {float(anon_now_bytes) / GIB:.2f} GiB vs pre-boot "
        f"baseline {foreign:.2f} GiB (attribution by TIME, not by owner)",
    )


def resolve_margin(
    flip_transient_gib: Optional[float] = None,
    drift_mib_per_min: Optional[float] = None,
    window_min: float = PLANNED_WINDOW_MIN_DEFAULT,
    residual_gib: Optional[float] = None,
    foreign_headroom_gib: float = 0.0,
    foreign_source: str = "",
) -> Margin:
    """Build the margin from measurements, naming every source.

    The transient is read from the RING-ERA records of this form (max over the
    recorded flips); the pre-ring :data:`FLIP_HOST_TRANSIENT_GIB` is used only
    when no ring-era sample exists, and the line says which one was used.
    """
    if flip_transient_gib is not None:
        transient, t_src = float(flip_transient_gib), "caller-supplied measured transient"
    elif RING_ERA_FLIP_TRANSIENT_GIB:
        boot, transient = max(RING_ERA_FLIP_TRANSIENT_GIB.items(), key=lambda kv: kv[1])
        t_src = f"RING-ERA max over {sorted(RING_ERA_FLIP_TRANSIENT_GIB)}, binding {boot}"
    else:
        transient, t_src = FLIP_HOST_TRANSIENT_GIB, "PRE-RING fallback, no ring-era sample exists"
    if transient < RING_GRANULE_GIB:
        transient, t_src = RING_GRANULE_GIB, f"floored at one ring granule ({t_src} was smaller)"

    if residual_gib is not None:
        residual, r_src = float(residual_gib), "caller-supplied measured residual"
    else:
        rb, residual = max(RUN_PEAK_RESIDUAL_GIB.items(), key=lambda kv: kv[1])
        residual = max(0.0, residual)
        r_src = f"RING-ERA max (measured peak - predicted) over {sorted(RUN_PEAK_RESIDUAL_GIB)}, binding {rb}"

    if drift_mib_per_min is None:
        drift_rate, d_src = IDLE_ANON_DRIFT_MIB_PER_MIN_DEFAULT, "sb4 default, no WEG2-IDLE-CENSUS yet"
    else:
        drift_rate, d_src = float(drift_mib_per_min), "WEG2-IDLE-CENSUS d_anon_mib_per_min"
    excess_min = max(0.0, float(window_min) - RESIDUAL_WINDOW_MIN)
    return Margin(
        transient_gib=transient,
        residual_gib=residual,
        drift_gib=drift_rate * excess_min / 1024.0,
        foreign_gib=float(foreign_headroom_gib),
        drift_mib_per_min=drift_rate,
        window_min=float(window_min),
        transient_source=t_src,
        residual_source=r_src,
        drift_source=d_src,
        foreign_source=foreign_source
        or "0 by design: arm-time foreign load is already in the origin; later "
        "desk load is unbounded and is the RUNTIME guard's job (W22)",
    )


def watermark_provenance(margin: Optional[Margin] = None,
                         watermark_gib: Optional[float] = None) -> str:
    """`WEG2-HOST WATERMARK=<v> source=<events> margin=<v> (terms)`."""
    m = margin if margin is not None else resolve_margin()
    w = watermark_gib if watermark_gib is not None else OBSERVED_REAP_NONRECLAIM_BYTES / GIB
    events = ", ".join(f"{k} {v:.2f}" for k, v in sorted(REAP_SAMPLES_GIB.items()))
    excl = ", ".join(f"{k} {v:.2f} EXCLUDED ({why})"
                     for k, (v, why) in sorted(REAP_SAMPLE_EXCLUDED.items()))
    live = read_cgroup_pressure()
    if live.get("nonreclaim_gib") is not None:
        cur = (
            f" LIVE nonreclaim={live['nonreclaim_gib']:.2f} "
            f"raw_current={live['current_gib']:.2f} "
            f"file_reclaimable={live['file_reclaimable_gib']:.2f} GiB "
            f"[{live['source']}]"
        )
    else:
        cur = f" LIVE unreadable [{live.get('source')}]"
    return (
        f"WEG2-HOST WATERMARK={w:.2f} GiB CURRENCY=non-reclaimable "
        f"(= memory.current at a reap, because the kernel has ALREADY reclaimed the "
        f"file cache by then: dk5's own reap row carries 0.03 GiB of it; dk6's "
        f"memory.stat was not captured, so that half is inference from one row) "
        f"source=[{events}] "
        f"BOOT bound = {w - m.boot_total_gib:.2f} GiB (margin {m.boot_total_gib:.2f}: "
        f"{m.terms()}) | RUNTIME bound = {w - m.runtime_total_gib:.2f} GiB "
        f"(margin {m.runtime_total_gib:.2f}: {m.terms('runtime')}); "
        f"excluded=[{excl}].{cur}"
    )


class Weg2HostWatermarkBreached(RuntimeError):
    """W22: the RUNTIME breach. cgroup memory.current crossed the hard bound
    while serving. The response is a CONTROLLED teardown down the killer path,
    never "accept the risk" and never a silent restart -- the user's standing
    order of 2026-09-08, given after exactly that offer was taken and the box
    went into the OOM anyway."""


def watermark_breach_verdict(
    current_bytes: int,
    margin: Optional[Margin] = None,
    watermark_gib: Optional[float] = None,
    cgroup_anon_bytes: Optional[int] = None,
    own_pids: Sequence[int] = (),
    nonreclaim_gib: Optional[float] = None,
    file_reclaimable_gib: Optional[float] = None,
    anon_preboot_bytes: Optional[int] = None,
    composition: Optional[Dict[str, Optional[float]]] = None,
) -> Optional[str]:
    """The runtime check, in NON-RECLAIMABLE currency (#1269 fix 3).

    Returns the verdict LINE when the bound is crossed, else None. Pure, so it
    is testable against a synthetic memory.stat with no processes involved.

    ``nonreclaim_gib`` is what the bound is tested against; the raw reading is
    printed beside it so the line carries BOTH currencies and the next boot can
    see which one bites. With no non-reclaimable reading available the raw one
    is used and the line SAYS SO -- conservative (it can only over-report
    pressure), but it is exactly the comparison that refused weg2sb5b 28 GiB
    below danger, so it is never silent about which currency it used.

    The split comes from the PRE-BOOT anon baseline (:func:`split_by_baseline`),
    never from ``cgroup anon - sum(RssAnon)``: that subtraction is invalid and
    printed a negative foreign term on sb5b. ``cgroup_anon_bytes`` /
    ``own_pids`` are kept in the signature for callers that still pass them,
    but ``own_pids`` no longer participates in the arithmetic.
    """
    m = margin if margin is not None else resolve_margin()
    w = watermark_gib if watermark_gib is not None else OBSERVED_REAP_NONRECLAIM_BYTES / GIB
    # #1269 FIX 4: the RUNTIME bound, which carries no model-error term. This
    # call grades a MEASUREMENT; the residual reserves for how wrong a
    # PREDICTION can be, and the measurement has already realised that error.
    # Boot weg2sb5c was refused at 87.67 against the boot bound 87.30 with the
    # reap point 8.2 GiB away -- that refusal was the double-charge, not danger.
    bound = w - m.runtime_total_gib
    boot_bound = w - m.boot_total_gib
    current = float(current_bytes) / GIB
    if nonreclaim_gib is not None:
        tested = float(nonreclaim_gib)
        currency = "non-reclaimable"
    else:
        tested = current
        currency = "RAW memory.current -- INCLUDES reclaimable cache, no memory.stat"
    if tested <= bound:
        return None
    cache = (
        f" file_reclaimable={file_reclaimable_gib:.2f}"
        if file_reclaimable_gib is not None
        else ""
    )
    if composition:
        cache += " STAT " + " ".join(
            f"{k}={v:.2f}" for k, v in sorted(composition.items()) if v is not None
        )
    split = ""
    if cgroup_anon_bytes is not None and anon_preboot_bytes is not None:
        sg, fo, src = split_by_baseline(cgroup_anon_bytes, anon_preboot_bytes)
        if sg is not None and fo is not None:
            split = (
                f" SPLIT sglang={sg:.2f} foreign={fo:.2f} GiB anon [{src}]"
                f" -- {'FOREIGN desk load' if fo > sg else 'sglang'} dominates."
            )
    return (
        f"W22 Weg2HostWatermarkBreached current={tested:.2f} watermark={w:.2f} "
        f"margin={m.runtime_total_gib:.2f} ({m.terms('runtime')}); "
        f"graded against the RUNTIME bound {bound:.2f} GiB "
        f"(the BOOT bound is {boot_bound:.2f}, and it is NOT what this line "
        f"grades: W21 grades predictions, W22 grades measurements) "
        f"crossed by {tested - bound:.2f} GiB in {currency} currency "
        f"[raw memory.current={current:.2f}{cache}].{split} CONTROLLED TEARDOWN "
        "down the killer path. Standing order 2026-09-08: the threshold is never "
        "crossed; crossing only ends in a crash, so there is no 'accept the risk'."
    )

#: #1233 boot weg2dk5: the OBSERVED REAP POINT of this box's cgroup -- the
#: memts row at 21:15:30Z, ``cg_current_b=102,998,904,832`` with ``oom_kill``
#: 18 -> 24 in the same row (and /proc/vmstat 60 -> 66 independently, same
#: delta +6).  ADVISORY ONLY, never a refusal: one boot's death is a watermark,
#: not a limit, and this cgroup publishes no finite ``memory.max`` to check
#: against.  It is printed beside each arm's PREDICTED run-peak ``cg_current``
#: so a configuration that is about to repeat weg2dk5 says so BEFORE the boot.
OBSERVED_REAP_CURRENT_BYTES = 102_998_904_832
#: The same watermark in the currency fix 6 budgets in: the NON-RECLAIMABLE
#: part of that reading.  The memts row of the reap (21:15:30Z) carries
#: ``cached_kb=55,231,500`` and ``shmem_kb=55,202,584``, i.e.
#: ``pagecache_ex_shmem_kb=28,916`` -- 0.03 GiB.  At the moment of the kill the
#: box's page cache was ALREADY almost pure shmem (the store tmpfs and the
#: shm-backed cpu images), so the watermark barely moves; stating that is what
#: makes the comparison against a non-reclaimable origin legitimate rather than
#: lucky.
#:
#: THE DIRECTION OF THE ONE TERM THAT IS MISSING (#1233 fix 7 -- fix 6 wrote
#: "tighter" here and had the sign backwards).  ``slab_reclaimable`` is not in
#: the sampler's columns, so it is charged as spent and NOT subtracted.  Not
#: subtracting a reclaimable term leaves the watermark HIGHER, and a higher
#: watermark makes the advisory's "ABOVE" verdict LESS likely: this constant is
#: an UPPER BOUND on the reap point in fix 6's currency, and the advisory
#: UNDER-warns by at most the unsampled slab term.  How big that is, measured
#: on this box rather than guessed -- ``/sys/fs/cgroup/memory.stat
#: slab_reclaimable`` = 564,459,544 B = 0.53 GiB (2026-09-07T23:57:35Z, idle
#: box) and 0.73 GiB at the pre-fix-6 reading recorded in
#: :func:`cg_reclaimable_bytes`.  So the under-warn is bounded by ~0.5-0.7 GiB
#: against a 95.90 GiB watermark, an order of magnitude under the advisory's
#: own 3.01 GiB under-prediction of weg2dk5 -- it does not move that boot's
#: verdict, and it is stated rather than corrected by a hand constant, because
#: the reap row itself carries no slab column to correct it WITH.
OBSERVED_REAP_NONRECLAIM_BYTES = OBSERVED_REAP_CURRENT_BYTES - 28_916 * 1024
#: cgroup-v2 semantics MEASURED on this box (2026-09-07, before fix 6) rather
#: than recalled -- the two readings a wrong formula would silently invert:
#:   /sys/fs/cgroup/memory.stat file  = 51,171,528,704 B
#:   /proc/meminfo Cached  = 49,972,196 kB = 51,171,528,704 B  (equal, to the byte)
#:   /sys/fs/cgroup/memory.stat shmem = 20,425,609,216 B
#:   /proc/meminfo Shmem   = 19,946,884 kB = 20,425,609,216 B  (equal, to the byte)
#:   /proc/meminfo SwapTotal = 0 kB
#: So v2's ``file`` INCLUDES ``shmem`` exactly as ``Cached`` includes ``Shmem``,
#: and with no swap shmem cannot be evicted at all -- it can only be deleted.
#: ``current - file`` would therefore hand back the tmpfs page store, THE
#: CARRIER, as if it were free memory.
CGROUP_V2_FILE_INCLUDES_SHMEM_PROOF = (
    "memory.stat file=51,171,528,704 B == /proc/meminfo Cached 49,972,196 kB; "
    "memory.stat shmem=20,425,609,216 B == /proc/meminfo Shmem 19,946,884 kB; "
    "SwapTotal=0 kB (this box, 2026-09-07): v2 `file` includes `shmem`, and "
    "shmem is unevictable without swap"
)
#: The loader's transient while group D loads next to dormant P, charged at
#: the launch moment only.  #721's constant is 27 GiB
#: (weg1_host_sizing.LOAD_TRANSIENT_BYTES, page cache + staging); on THIS
#: launcher it is MEASURED smaller: memts of boot weg2ls1b2 (2026-09-07
#: 07:09:44Z, first sample, right after D READY) read MemAvailable 48.0 GB
#: against 60.0 GB once the load's page cache had been reclaimed (07:10:1xZ),
#: i.e. 12 GB of the load's cache was not counted as available.  The in-load
#: PEAK was not sampled (the sampler started after D was ready; this
#: launcher now starts it before group P) -- what IS metal-proven is that the
#: launch moment of that boot (same shape: P image resident, D loading,
#: memavail 107.6 GiB) passed with no OOM while the 27 GiB constant plus the
#: #1232 headroom would price it at -7.4 GiB, so the constant over-charges
#: this loader by at least that much.  12 GiB is the measured residual; the
#: next boot's memts series replaces it if the in-load peak reads higher.
LOAD_TRANSIENT_GIB = 12.0
#: #1317n THE TRANSIENT, SPLIT BY MEMORY CLASS -- because the ledger already
#: has two currencies and this constant was being spent in the wrong one.
#:
#: THE TWO CURRENCIES, both already in this module. The reap watermark counts
#: NON-RECLAIMABLE memory only: :func:`cg_reclaimable_bytes` is
#: ``(file - shmem) + slab_reclaimable`` and its docstring states "anon and
#: unevictable are never subtracted: both are exactly what the reaper kills to
#: recover". MemAvailable is the other currency, and it genuinely UNDER-REPORTS
#: active page cache.
#:
#: WHICH CLASS THE 12 GiB IS, from this constant's own provenance above: "12 GB
#: of THE LOAD'S PAGE CACHE was not counted as available", reclaimed ~30 s
#: after D READY (weg2ls1b2: MemAvailable 48.0 GB during the load against 60.0
#: GB once it had been reclaimed). Page cache. Not anon, not shmem.
#:
#: WHAT THAT MEANS FOR EACH BARRIER, one currency per barrier and no mixing:
#: against a MemAvailable-derived base the whole transient IS charged -- the
#: launcher samples BEFORE the groups start, so the depression is a real
#: forward-looking cost of that reading. Against the REAP-derived base
#: (``ceiling - non-reclaimable memory.current``) the page-cache half may NOT
#: be charged, because that base does not contain page cache in the first
#: place; charging it there books a quantity the reaper never kills for.
#:
#: MEASURED ON boot weg2sn6r, both forms, so the difference is a number and not
#: an argument: launch leftover **-2.78 GiB** charged as a SUM (what refused
#: the boot) against **+9.2 GiB** charged by class. The run moment funds either
#: way (leftover 7.75 GiB).
LOAD_TRANSIENT_PAGECACHE_GIB = 12.0
#: The anon/shmem half. ZERO, and the zero is asserted with provenance rather
#: than assumed: the ls1b2 measurement does NOT separate anon from page cache
#: -- it reads MemAvailable before and after a reclaim -- so what is proven is
#: that the residual DISAPPEARED on reclaim, and only page cache does that.
#: An anon staging buffer would have survived it. If a later memts series
#: separates the two, this is where the anon part goes, and it stays charged
#: against BOTH barriers.
LOAD_TRANSIENT_ANON_GIB = 0.0
LOAD_TRANSIENT_SPLIT_PROVENANCE = (
    "weg2ls1b2 (#721): MemAvailable 48.0 GB during D's load against 60.0 GB "
    "once the load's PAGE CACHE had been reclaimed ~30 s after READY. The "
    "residual vanished on reclaim, which only page cache does -- an anon "
    "staging buffer would have survived it -- so the split is 12.0 page cache "
    "/ 0.0 anon. The measurement does not separate the two directly; this line "
    "IS the provenance the zero rests on."
)
if abs(
    (LOAD_TRANSIENT_PAGECACHE_GIB + LOAD_TRANSIENT_ANON_GIB) - LOAD_TRANSIENT_GIB
) > 1e-9 or not LOAD_TRANSIENT_SPLIT_PROVENANCE:
    raise RuntimeError(
        "#1317n the load transient's class split must sum to the measured "
        f"total ({LOAD_TRANSIENT_GIB}) and must carry a provenance line: got "
        f"{LOAD_TRANSIENT_PAGECACHE_GIB} + {LOAD_TRANSIENT_ANON_GIB}. An "
        "unproven split would let the launch moment be priced in a currency "
        "nobody measured, which is the defect this split exists to fix."
    )
#: b0: 10.19 GB MEASURED for six (rank x phase) anchor pools at m_mib=2400
#: (2.55/1.49/1.06 + 2.55/1.27/1.27 GB).  Scaled linearly with M.
ANCHORS_AT_2400_BYTES = 10.19 * GB
#: The SAME measurement, split the way it was measured: three (rank x phase)
#: pools per group.  Kept beside the total so the two can never say different
#: things -- ``non_backup_host_bytes`` needs the per-group half and
#: :func:`price` needs the sum, and a second hand-typed total is how they drift.
ANCHORS_P_AT_2400_BYTES = (2.55 + 1.49 + 1.06) * GB
ANCHORS_D_AT_2400_BYTES = (2.55 + 1.27 + 1.27) * GB
ANCHORS_REFERENCE_M_MIB = 2400
#: #1318 THE RING MULTIPLIERS ARE DERIVED, NOT MEASURED TWICE.
#:
#: WHAT WAS WRONG.  b0 read "8xS ring = 16.00 GB at S=2" off a boot log and
#: split it by eye into "PP pool 2.00/1.00/1.00 GB (2xS)" for group P and
#: "TP pool 4.00 GB x3 (6xS)" for group D.  Boot ``weg2sn5w`` measured the
#: pools themselves and the D half is 2x over-booked: D's log prints
#: ``HiCache host pool (30518 tokens)`` on all three ranks at 32,768 B per
#: token, so the group holds 30,518 x 32,768 x 3 = 3.00 GB per S -- against
#: 6.0 charged.  A 3.0 GB/S over-charge at S=2 is 6.0 GB the store never got,
#: and it is what held the shipped store at 9 GiB with a 244,140-token bound
#: still in force below the 262,144 context.
#:
#: WHY A FORMULA AND NOT A SECOND PAIR OF CONSTANTS.  A hand-typed pair is a
#: second bookkeeping beside a quantity the pool already knows, and it drifts
#: the moment the cell size or the rank layout moves (which is exactly what
#: uneven DCP does to group P).  The ring cost of one GB of
#: ``--hicache-size`` is
#:
#:      rows_per_rank x sum(cell_bytes over the group's ranks)
#:
#: because ``pool_host/base.py sync_fixed_hicache_size`` MIN-syncs the ROW
#: COUNT across the group, so every rank holds the same number of rows and
#: the row count is set by the rank with the LARGEST cell:
#:      rows_per_rank(S=1 GB) = GB / max(cell_bytes)
#: and each rank then pays its own cell size for those rows.  Linear in S.
#:
#: PROVENANCE, both groups, boot ``weg2sn5w`` (front/P/D logs under
#: /spinning/evidence-665-f1, pattern ``boot_weg2_weg2sn5w_*``):
#:   * group P, ``--tp-size 1 --pp-size 3``, uneven DCP: cells 18,432 /
#:     8,192 / 6,144 B; rows MIN-sync to 1e9/18432 = 54,254 (P log:
#:     ``host KV pool 54254``); ring = 54,254 x 32,768 = 1.78 GB per S.
#:     The 2.0 charged before over-charges by 0.22 GB/S -- small, and in the
#:     conservative direction, which is why it was never caught.
#:   * group D, ``--tp-size 3`` even: cells 32,768 B x 3; rows =
#:     1e9/32768 = 30,518 (D log: ``HiCache host pool (30518 tokens)``);
#:     ring = 30,518 x 98,304 = 3.00 GB per S.  This is the 2x.
#: Derived total 4.78xS against b0's 8xS reading: the derivation may never
#: come out ABOVE the reading it replaces, and :func:`_ring_mult_gb_per_s`
#: refuses if it does rather than silently under-charging the host.
CELL_BYTES_P_PER_RANK = (18432, 8192, 6144)
CELL_BYTES_D_PER_RANK = (32768, 32768, 32768)
#: The b0 reading the derivation replaces, kept ONLY as the refusal ceiling
#: above.  Never charged.
RING_B0_TOTAL_MULT_GB_PER_S = 8.0


def _ring_mult_gb_per_s(cells: tuple) -> float:
    """GB of host ring per GB of ``--hicache-size``, for one group.

    ``rows_per_rank = GB / max(cells)`` is ``sync_fixed_hicache_size``'s
    MIN-sync stated as arithmetic; every rank then pays its own cell size for
    those rows, so the group's bytes are ``rows * sum(cells)``.
    """
    if not cells or min(cells) <= 0:
        raise ValueError(f"cell byte sizes must all be positive, got {cells!r}")
    rows_per_rank = GB / float(max(cells))
    return rows_per_rank * float(sum(cells)) / GB


RING_P_MULT_GB_PER_S = _ring_mult_gb_per_s(CELL_BYTES_P_PER_RANK)
RING_D_MULT_GB_PER_S = _ring_mult_gb_per_s(CELL_BYTES_D_PER_RANK)
if RING_P_MULT_GB_PER_S + RING_D_MULT_GB_PER_S > RING_B0_TOTAL_MULT_GB_PER_S:
    raise RuntimeError(
        "#1318 host ring derivation came out ABOVE the b0 reading it replaces "
        f"({RING_P_MULT_GB_PER_S:.3f}+{RING_D_MULT_GB_PER_S:.3f} = "
        f"{RING_P_MULT_GB_PER_S + RING_D_MULT_GB_PER_S:.3f} > "
        f"{RING_B0_TOTAL_MULT_GB_PER_S:.3f} xS): a derivation that charges the "
        "host LESS than the measurement is a free store, and one that charges "
        "MORE means the cell sizes or the rank layout above no longer describe "
        "this boot. Re-measure the pools; do not relax this check."
    )
#: b0 U14: the unattributed residual is 1.006 GiB = ~4 % on top of the
#: host-pool posts (rings + anchors).  Charged as +4 % on those posts.
HOST_POOL_OVERHEAD = 0.04

#: #1233 draft KV across the flip (C17). Pinned host DRAFT pools, one row per
#: target host slot (``kv_cache_builder._build_draft_host_pool``: "same slot
#: count as the target host pool"), 2048 B/token for the NEXTN head (1 layer
#: x 4 kv heads x 256 head_dim x 2 (K,V) x 1 B fp8).  Group P: the producer
#: on the last stage, 61,037 slots (P log weg2zr2: host KV pool 61,036 + the
#: page-alignment slot) -> 119.2 MiB.  Group D: 30,519 slots x this rank's
#: head share {1024, 512, 512} B (2/1/1 heads over the three ranks) -> 59.6
#: MiB in total.  Both are pinned at launch and charged at BOTH moments; D's
#: term used to sit implicitly inside RING_D_MULT_GB_PER_S and is explicit
#: from here on.
#: #1264 ``--draft-kv-on-p off`` DOES NOT REACH THIS TERM, deliberately, and
#: the reason is named here rather than left for a reader to rediscover.  Under
#: ``off`` group P builds no draft host pool at all, so ``draft_host_p_gib``
#: (119.2 MiB = 0.116 GiB) is charged against a pool that does not exist.  It
#: is left charged because the error is CONSERVATIVE in the only direction that
#: matters: an over-charge makes the ledger stricter (a smaller store, an
#: earlier refusal), never looser, and 0.116 GiB against a ~90 GiB idle boot is
#: below the resolution of every arm decision the ladder makes.  Threading the
#: predicate through ``charge_terms`` -> ``price`` -> ``choose`` for it would
#: put a boolean in three signatures that ``predicted_run_peak_gib`` and
#: ``dk7_run_residual_gib`` must then agree about -- the second-bookkeeping
#: shape, for a term smaller than the rounding.  The ledger DOES follow the
#: switch where it is material, and through the ring rather than a flag: an
#: ``off`` boot's ring is 643 MiB smaller (Sigma H 38306 -> 37663 MiB, measured
#: dry-run 2026-09-08), and the ledger hands that back as store 8 -> 9 GiB.
#: If this term is ever made switch-aware, make it aware in ``charge_terms``
#: only, so those three call sites keep their single authority.
DRAFT_PAGE_BYTES = 2048
DRAFT_HOST_SLOTS_P = 61037
DRAFT_HOST_SLOTS_D = 30519
DRAFT_HOST_P_MIB = DRAFT_HOST_SLOTS_P * DRAFT_PAGE_BYTES / float(2**20)
DRAFT_HOST_D_MIB = DRAFT_HOST_SLOTS_D * (1024 + 512 + 512) / float(2**20)
#: The draft tier of the STORE: 2048 B per token beside the 32768 B canonical
#: KV page (16 attention layers x 2048 B) = 1/16.  Unchanged in bytes versus
#: the three per-rank shards it replaces (1024+512+512 = 2048), inodes / 3.
STORE_DRAFT_FRACTION = DRAFT_PAGE_BYTES / 32768.0

#: The arm ladder: (S GB per --hicache-size, M MiB per --hicache-mamba-host-mib).
DEFAULT_ARMS: Tuple[Tuple[int, int], ...] = ((1, 2400), (1, 1200), (1, 600))


class Weg2HostLedgerRefused(RuntimeError):
    """W20: no arm of the ladder funds both moments plus the store floor."""


class Weg2HostRunPeakRefused(RuntimeError):
    """W21 (fix 8): an arm funds both moments, and its RUN PEAK does not.

    The term this class exists for is the one two boots died on while the
    advisory said "below": the predicted non-reclaimable ``memory.current`` at
    the run peak against :data:`OBSERVED_REAP_NONRECLAIM_BYTES`.  It is a
    SEPARATE class from W20 so the log says which quantity refused -- a store
    floor and a reap watermark are different findings with different levers.
    """


# --------------------------------------------------------------------------
# FIX 8: the image term, measured
# --------------------------------------------------------------------------


@dataclass
class ImageTerms:
    """The dormant host image of each group, with WHERE each number came from.

    ``p_measured`` / ``d_measured`` are False for a value that is a recorded
    reading of ANOTHER boot or a BOUND -- the printed line says so, because a
    bound that reads like a measurement is exactly how 28.83 GiB survived three
    boots.
    """

    p_gib: float
    d_gib: float
    p_source: str
    d_source: str
    p_measured: bool
    d_measured: bool
    extra_p_gib: float
    extra_d_gib: float


def resolve_image_terms(record: Optional[Dict[str, dict]] = None) -> ImageTerms:
    """The dormant image per group, in the fix-8 precedence order.

    (a) THIS LINE'S OWN PREVIOUS MEASUREMENT -- an entry written by
        :func:`dormant_image_sample` at that group's first sleep (the launcher
        for P, the front for D), carrying ``rss_shmem_gib`` plus the commit,
        the boot tag and the timestamp it was taken at;
    (b) absent a measurement for P: the named dk7 reading
        (:data:`DK7_DORMANT_IMAGE_P_GIB`, :data:`DK7_PROVENANCE`);
    (c) absent a measurement for D: a BOUND, never a claim.  D's dormant image
        has NEVER been measured (dk7's 6.71 GiB is D AWAKE -- its live host
        rings and anchors, not a backup), so the ledger refuses to price it
        below the one image it HAS measured:
        ``max(image_P, weight_tags_D + extra_P)``.  Both candidates are stated;
        the ``extra`` term is what the weight-tag census misses, measured once
        on P and carried to D because the mechanism (every ``enable_cpu_backup``
        buffer, not only ``weights_*``) is the same on both groups.
    """
    wt_p = WEIGHT_TAGS_P_BYTES / GIB
    wt_d = WEIGHT_TAGS_D_BYTES / GIB
    rec = record or {}
    p_entry = rec.get("P") or {}
    d_entry = rec.get("D") or {}
    p_meas = p_entry.get("rss_shmem_gib")
    d_meas = d_entry.get("rss_shmem_gib")

    if p_meas is not None:
        p_gib = float(p_meas)
        p_source = (
            f"MEASURED by this line: boot {p_entry.get('boot_tag', '?')} @ "
            f"{p_entry.get('commit', '?')} at {p_entry.get('at', '?')} "
            f"(RssShmem sum of the sleeping group's {len(p_entry.get('pids') or [])} pids)"
        )
        p_measured = True
    else:
        p_gib = DK7_DORMANT_IMAGE_P_GIB
        p_source = f"RECORDED MEASUREMENT of another boot: {DK7_PROVENANCE}"
        p_measured = False
    extra_p = p_gib - wt_p

    if d_meas is not None:
        d_gib = float(d_meas)
        d_source = (
            f"MEASURED by this line: boot {d_entry.get('boot_tag', '?')} @ "
            f"{d_entry.get('commit', '?')} at {d_entry.get('at', '?')} "
            f"(RssShmem sum of the sleeping group's {len(d_entry.get('pids') or [])} pids)"
        )
        d_measured = True
    else:
        candidate = wt_d + extra_p
        d_gib = max(p_gib, candidate)
        d_source = (
            f"BOUND, NOT A MEASUREMENT: group D's dormant image has never been measured "
            f"(dk7's 6.71 GiB is D AWAKE -- live host rings and anchors, not a backup). "
            f"max(measured_P {p_gib:.2f}, weight_tags_D {wt_d:.2f} + extra_P {extra_p:.2f} "
            f"= {candidate:.2f}) = {max(p_gib, candidate):.2f} GiB -- the ledger will not "
            f"claim a SMALLER image for D than the one it has measured for P"
        )
        d_measured = False
    return ImageTerms(
        p_gib=p_gib,
        d_gib=d_gib,
        p_source=p_source,
        d_source=d_source,
        p_measured=p_measured,
        d_measured=d_measured,
        extra_p_gib=extra_p,
        extra_d_gib=d_gib - wt_d,
    )


#: #1317n D's OWN L2 BUDGET, IN GB, DERIVED FROM THE REQUEST CAP.
#:
#: WHY THIS EXISTS AT ALL, and it is a deletion story rather than a feature.
#: Upstream stages EVERY L3 read through the host pool (``batch_get`` into host
#: pages; there is no disk->device path), and it sizes that pool so residency
#: can never bind a read: ``pool_host/base.py`` takes ``ratio x device_pool``
#: when ``--hicache-size`` is unset, with the ratio defaulting to 2.0, so
#: L2 >= L1 by construction. This fork shipped ``--hicache-size 1`` -- a fixed
#: 1 GB, 30,518 rows on group D -- which BROKE that invariant, and the whole
#: #1317 window/chain/anchor layer was the compensation for the break. The user
#: decided 2026-09-10 to delete the compensation and restore the invariant.
#:
#: WHY NOT SIMPLY UNSET THE FLAG, which would be the true upstream form:
#: priced, and it does not fit. Group D's profiled per-rank device capacity on
#: boot weg2sn6p was [188788, 262358, 260566] tokens; ``ratio 2.0`` against the
#: largest is 524,716 host rows at 32,768 B = **17.2 GB on one rank, 51.5 GB
#: across three**, against a box with 39 GiB MemAvailable and a reap mark at
#: 95.90 GiB non-reclaimable. The ratio path is unaffordable on this rig, so
#: the absolute flag stays -- but it is DERIVED from what one cap-sized read
#: needs instead of pinned at a constant that no quantity justified.
#:
#: THE INVARIANT THAT IS ACTUALLY RESTORED, stated as the thing to test: a
#: single store read of ``--max-kv-per-request`` tokens must fit in one
#: prefetch, on every rank, without a window. That is
#:     rows_needed = cap_tokens x max_rank_share
#: and the pool must hold it BEHIND the load-pool fraction the controller
#: applies (``HICACHE_LOAD_POOL_USAGE_FRACTION`` = 0.9), hence the division.
#:
#: WORKED, boot weg2sn6p: cap 262,144 x share 0.375 x 32,768 B / 0.9 / 1e9 =
#: 3.58 -> **4 GB per rank** = 122,070 rows, of which the controller will lend
#: 0.9 x 122,070 = 109,863 >= the 98,304 a cap-sized read needs. The old
#: pinned 1 GB gave 30,518 rows and a limit of 27,466 -- which is where the
#: 413 band and every window came from.
def derive_d_hicache_size_gb(
    cap_tokens: int,
    max_rank_share: float,
    cell_bytes: int,
    fraction: float,
) -> Dict[str, float]:
    """S_D in GB plus every term that produced it. Never a bare number.

    Returns the terms as well as the result so the launcher can print a
    provenance line: a derived budget whose inputs are not printed is
    indistinguishable from a pinned one to the next reader, and this fork has
    paid for that confusion (a launcher census read 19 mamba slots as the KV
    carrier and shipped a 17-token bound, boot weg2rg5).
    """
    cap_tokens = max(0, int(cap_tokens))
    cell_bytes = max(1, int(cell_bytes))
    share = float(max_rank_share)
    fraction = float(fraction)
    if not (0.0 < share <= 1.0):
        raise ValueError(
            f"max_rank_share must be in (0, 1]: {share!r}. It is max(v)/sum(v) "
            "of group D's KV-token ownership vector, so it is a fraction by "
            "construction; a value outside the interval means the vector was "
            "misread, and sizing L2 against a misread share is how the 1 GB "
            "constant survived unquestioned."
        )
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f"fraction must be in (0, 1]: {fraction!r} "
            "(HICACHE_LOAD_POOL_USAGE_FRACTION)"
        )
    rows_needed = int(-(-cap_tokens * share // 1))  # ceil, share is a fraction
    gb_exact = rows_needed * cell_bytes / fraction / GB
    s_gb = int(-(-gb_exact // 1))
    rows = int(s_gb * GB // cell_bytes)
    return {
        "s_gb": float(s_gb),
        "gb_exact": gb_exact,
        "cap_tokens": float(cap_tokens),
        "max_rank_share": share,
        "cell_bytes": float(cell_bytes),
        "fraction": fraction,
        "rows_needed": float(rows_needed),
        "rows": float(rows),
        "rows_lendable": float(int(rows * fraction)),
    }


def d_hicache_provenance(terms: Dict[str, float], share_source: str) -> str:
    """The one line that makes S_D auditable at the boot log."""
    return (
        f"WEG2-L2 D hicache_size={int(terms['s_gb'])} GB derived: "
        f"cap={int(terms['cap_tokens'])} share={terms['max_rank_share']:.4f} "
        f"({share_source}) fraction={terms['fraction']:.2f} "
        f"cell={int(terms['cell_bytes'])} "
        f"rows_needed={int(terms['rows_needed'])} rows={int(terms['rows'])} "
        f"rows_lendable={int(terms['rows_lendable'])} "
        f"exact={terms['gb_exact']:.2f} GB -- one cap-sized store read must fit "
        f"in ONE prefetch on every rank, with no window and no chain (#1317n). "
        f"Upstream sizes this as ratio x device_pool (pool_host/base.py, ratio "
        f"2.0 -> L2 >= L1); that path costs 17.2 GB/rank on this rig's profiled "
        f"D capacity and is unaffordable, so the absolute flag stays and is "
        f"DERIVED here instead of pinned"
    )


def largest_fundable_d_cap(
    priced_ok, cap_tokens: int, max_rank_share: float,
    cell_bytes: int, fraction: float,
) -> Optional[Dict[str, float]]:
    """The biggest ``--max-kv-per-request`` some fundable S_D could serve.

    ORDER ITEM 1, THE HALF THAT WAS MISSING: *"ist S_D nicht fundierbar ->
    benannte Refusal am Launch, die den GROESSTEN FUNDIERBAREN CAP nennt (kein
    stilles Verkleinern)"*. The first build shipped the refusal without the
    number, so boot weg2sn6r got a correct W20 that named Sigma H and the rung
    as its levers and could not say what cap WOULD fit -- leaving the operator
    to invert the arithmetic by hand.

    ``priced_ok`` is a callable ``s_gb_d -> bool`` (does an arm at that D budget
    fund BOTH moments), so this function does no pricing of its own and cannot
    drift from the ladder that refused.

    Returns None when not even S_D=1 funds the moments -- then the D budget is
    not the binding term and saying otherwise would send the operator after the
    wrong lever.
    """
    share = float(max_rank_share)
    cell = max(1, int(cell_bytes))
    frac = float(fraction)
    if share <= 0.0 or frac <= 0.0:
        return None
    best = None
    for s_d in range(max(1, int(round(cap_tokens * share * cell / frac / GB))), 0, -1):
        if priced_ok(s_d):
            best = s_d
            break
    if best is None:
        return None
    rows = int(best * GB // cell)
    lendable = int(rows * frac)
    return {
        "s_gb_d": float(best),
        "rows": float(rows),
        "rows_lendable": float(lendable),
        "cap_tokens": float(int(lendable / share)),
        "asked_cap": float(int(cap_tokens)),
    }


def _arm_s_d(arm) -> str:
    """#1317n D's own L2 budget AS PRICED, or "=S" when it equals P's.

    On the ARM line because the two budgets stopped being one number: D carries
    a whole cap-sized read (4 GB) where P carries a staging tier (1 GB), which
    is +8.38 GiB of rings. An ARM line printing one S while the boot runs two
    is how an arm gets believed against a budget it was never scored on, and
    the reap mark is what sits on the other side of that error.
    """
    v = (getattr(arm, "terms", None) or {}).get("s_gb_d")
    return "=S" if v is None else str(int(v))

def xchg_bounce_bytes_per_card(slots: Optional[int] = None,
                               slot_bytes: Optional[int] = None) -> int:
    """ONE card's store-and-forward deposit on the host, in bytes (#1273 S6).

    THE EXCHANGE'S OWN CARRIER, PRICED.  The on-card lane's ``host`` arm keeps
    its bounce in ``/dev/shm`` (``oncard_host_path``), ``cudaHostRegister``ed,
    and S6 sizes it ``slots >= batches`` so the source can deposit and return
    without a live consumer.  Those are pinned, non-reclaimable bytes that land
    directly on ``memory.current`` for the span of a flip, and until this
    function existed they appeared in NO ledger term -- the omission is named
    verbatim in ``oncard_host_path``'s own docstring ("192 MiB of host that
    spec 0.2's ledger term does NOT carry ... the number belongs in the
    record").

    ``kv-1m-kein-lossy-kein-hostram`` is about KV and does not reach here: this
    is the exchange's own carrier, not hot KV parked in host RAM.  What DOES
    reach here is ``host-schwelle-nie-uebertreten`` -- which is exactly why the
    bytes become a TERM (so the reap bound sees them and the store shrinks by
    them) instead of a note.

    IT IS NOT "FOR THE SPAN OF A FLIP", AND THE EARLIER WORDING WAS WRONG (S6
    refuter, finding 9).  ``HostBounce.close`` deliberately does not unlink, so
    a deposited file lives until the boot's ``/dev/shm`` residue sweep -- which
    is what makes the destination's later leg able to read it, and what makes
    charging the bytes at BOTH the launch and the run moment correct rather
    than conservative.

    THE GEOMETRY IS READ FROM ITS OWNER, and now actually is (S6 refuter,
    must_fix 3).  This read ``ONCARD_SLOTS_MAX x ONCARD_SLOT_BYTES`` -- the
    slot count's CEILING beside the slot size's FLOOR -- while claiming to read
    the owner's bound, and understated the shape's maximum by 4x: 8 x 32 MiB =
    256 MiB against the 8 x 128 MiB the planner may actually derive
    (``plan_oncard_slot_bytes`` clamps to ``ONCARD_SLOT_BYTES_MAX``).  The
    consequence was not an overspend but a systematic FALSE REFUSAL, because
    ``deposit_refusal_reason`` grades the deposit against this same number: a
    per-card diagonal above ~256 MiB refused ``ledger-cannot-fund-deposit`` --
    exactly the band the geometry exists for.  The owner's own name for the
    bound is :data:`tp.ONCARD_DEPOSIT_BYTES_MAX`, and this reads THAT.

    The worst case is charged, not the case a particular flip happens to
    derive, because the arm is chosen once at launch and the derivation runs
    per leg -- a charge that tracked the derivation would fund the smallest
    flip and refuse none.
    """
    from sglang.srt.weg2 import weight_exchange_transport as tp

    if slots is None and slot_bytes is None:
        return int(tp.ONCARD_DEPOSIT_BYTES_MAX)
    return (int(tp.ONCARD_SLOTS_MAX if slots is None else slots)
            * int(tp.ONCARD_SLOT_BYTES_MAX if slot_bytes is None else slot_bytes))


def xchg_bounce_bytes(cards: Optional[int] = None,
                      slot_bytes: Optional[int] = None) -> int:
    """Every card's deposit -- the term :func:`charge_terms` carries.

    ``cards`` DEFAULTED TO A TYPED ``3`` (S6 refuter, finding 6): a hand number
    at the one call site, derived from nothing, in a module whose whole subject
    is numbers with provenance.  It now comes from the region's own rank
    layout -- one bounce file per CARD, and a card is a co-located pair of the
    six rows -- which is the same arithmetic
    :data:`tp.ONCARD_HOST_DEGRADE_MIB` announces the degrade with.
    """
    from sglang.srt.weg2 import weight_exchange_region as xr_

    if cards is None:
        cards = int(xr_.N_RANKS) // 2
    # S6 fix E: ``slot_bytes`` is the launcher passing the value of
    # ``--weg2-xchg-oncard-slot-mib`` it parsed, because ITS process imported
    # the transport before that flag existed in any environment.  ``None`` is
    # the rank path, where the module constant already resolved from the
    # published env.  One arithmetic function, two ways in.
    return max(0, int(cards)) * xchg_bounce_bytes_per_card(
        slot_bytes=slot_bytes)


def charge_terms(
    s_gb: int, m_mib: int, ranks_per_group: int, images: ImageTerms,
    s_gb_d: Optional[int] = None,
    xchg_bounce_host_bytes: int = 0,
) -> Dict[str, float]:
    """Everything the BOOT ITSELF adds to ``memory.current``, per term.

    One authority for "what this arm charges": :func:`price` builds the arm
    from it, :meth:`Arm.predicted_run_peak_gib` adds it to the run origin, and
    :func:`dk7_run_residual_gib` subtracts it from a measured reading.  Three
    call sites that used to be able to disagree about the same sum.

    The image terms are NOT in here: which image is resident is a property of
    the MOMENT (launch charges P's, the run moment charges the dormant one),
    not of the arm.

    ``xchg_bounce_host_bytes`` IS AN ARM PROPERTY and therefore is (#1273 S6):
    it is 0 on ``--weg2-weight-source ring`` -- the default and every boot that
    has run -- and :func:`xchg_bounce_bytes` on the arms that create the file.
    The KEY IS ALWAYS PRESENT, value 0.0 when unarmed, because the three
    consumers of this dict must not be able to disagree about whether a term
    exists; an unarmed boot prints ``xchg_bounce=0.00`` and says so rather than
    leaving a reader to wonder whether it was priced.  The parameter is
    deliberately NOT named after the module function that produces the value:
    :func:`price`'s own fix-8 note records what a shadowed name costs.
    """
    anchors_gib = (ANCHORS_AT_2400_BYTES * (m_mib / ANCHORS_REFERENCE_M_MIB)) / GIB
    # #1317n THE TWO GROUPS ARE PRICED SEPARATELY, because they no longer carry
    # the same budget: P only WRITES the store (its 1 GB staging tier is all it
    # needs) while D must hold a whole cap-sized read. `s_gb_d` defaults to
    # `s_gb`, so every pre-existing caller and every recorded arm is
    # byte-identical; only a caller that passes a different D budget changes.
    _s_d = s_gb if s_gb_d is None else int(s_gb_d)
    rings_gib = (
        RING_P_MULT_GB_PER_S * s_gb + RING_D_MULT_GB_PER_S * _s_d
    ) * GB / GIB
    return {
        "heaps_gib": ranks_per_group * (HEAP_AWAKE_GIB + HEAP_DORMANT_GIB),
        "anchors_gib": anchors_gib,
        "rings_gib": rings_gib,
        # #1317n the D budget the rings were priced from, so the ARM line
        # prints what it CHARGED rather than what it was asked for.
        "s_gb_d": float(_s_d),
        "overhead_gib": HOST_POOL_OVERHEAD * (anchors_gib + rings_gib),
        "draft_host_p_gib": DRAFT_HOST_P_MIB / 1024.0,
        "draft_host_d_gib": DRAFT_HOST_D_MIB / 1024.0,
        "image_p_gib": images.p_gib,
        "image_d_gib": images.d_gib,
        "weight_tags_p_gib": WEIGHT_TAGS_P_BYTES / GIB,
        "weight_tags_d_gib": WEIGHT_TAGS_D_BYTES / GIB,
        "image_extra_p_gib": images.extra_p_gib,
        "image_extra_d_gib": images.extra_d_gib,
        "xchg_bounce_gib": max(0, int(xchg_bounce_host_bytes)) / GIB,
    }


def _boot_charges_gib(terms: Dict[str, float]) -> float:
    """The sum of the arm's own charges, image and store EXCLUDED."""
    return (
        terms["heaps_gib"] + terms["anchors_gib"] + terms["rings_gib"]
        + terms["overhead_gib"] + terms["draft_host_p_gib"] + terms["draft_host_d_gib"]
        # #1273 S6: 0.00 on every ring-arm boot, so every existing number is
        # unchanged; on an armed boot it is charged at BOTH moments, which is
        # what makes `size_store_gib`'s leftover -- and therefore the store --
        # shrink by exactly the deposit rather than by a note in a docstring.
        + terms["xchg_bounce_gib"]
    )


def dk7_run_residual_gib() -> float:
    """What boot weg2dk7 held at the RUN moment that this ledger does not name.

    DERIVED, not a constant: the measured quiet reading
    (:data:`DK7_QUIET_CG_CURRENT_GIB`) minus everything the ledger charges for
    the arm that boot ran (:data:`DK7_ARM_S_GB` / :data:`DK7_ARM_M_MIB`), minus
    the MEASURED image of the dormant group, minus the store's MEASURED content
    (0.00 GiB -- the store was empty).  No flip transient: zero flips ran.

    What is in it: the two server processes, the front, the launcher, the
    sampler and every foreign process on the box -- the boot's 13 processes'
    non-shmem working set beyond the six ranks' heaps this ledger charges, plus
    whatever else shares the cgroup.  The ledger does not name those terms, so
    the honest form is to measure the remainder instead of extending the term
    list with estimates.
    """
    images = resolve_image_terms(None)
    terms = charge_terms(DK7_ARM_S_GB, DK7_ARM_M_MIB, 3, images)
    return (
        DK7_QUIET_CG_CURRENT_GIB
        - _boot_charges_gib(terms)
        - images.p_gib
        - DK7_QUIET_STORE_USED_GIB
    )


def record_run_residual_gib(
    entry: Dict[str, object], ranks_per_group: int = 3
) -> Tuple[Optional[float], float]:
    """One record's run residual RE-PRICED in the arm's own currency.

    Returns ``(residual, correction)``; ``correction`` is what was removed, so
    a caller can print it and 0.0 means "this record needed nothing".

    #1325 -- WHY THE READER AND NOT ONLY THE SAMPLER. The sampler's own
    subtraction was in the wrong currency (it omitted ``s_gb_d``, so D's rings
    stayed inside the residual and the ledger then added them again at
    prediction time). Fixing the sampler corrects records written from now on
    -- and the record that BLOCKS the next boot is already on disk. The sidecar
    is APPEND-ONLY because history is evidence, so it may not be rewritten;
    the derivation therefore moves to the reader, where it corrects old and new
    records alike from one place. ONE reader, ONE derivation, and the stored
    ``run_residual_gib`` stays exactly as the boot wrote it.

    THE CORRECTION IS COMPUTABLE FROM THE RECORD ALONE, which is what makes
    this a re-derivation and not a guess. The stored residual is
    ``nonreclaim - charges(arm without s_gb_d) - image``, so
    ``nonreclaim = stored + charges_old + image`` and the corrected value is
    ``stored - (charges_new - charges_old)``. Both charge figures come from
    ``charge_terms`` with the record's OWN stored ``arm`` and ``rss_shmem_gib``
    -- no new measurement, no constant.

    MEASURED against the live sidecar
    (``/spinning/evidence-665-f1/...``, 2026-09-10): every boot up to weg2sn6p
    stored an arm of ``{s_gb: 1, m_mib: 600|1200}`` with NO ``s_gb_d`` and their
    P residuals sit at 7.42-9.01 GiB. **weg2sn6s is the first and only record
    carrying ``s_gb_d: 4``** -- it is the first boot on this line to give D its
    own budget -- and its P residual jumps to **16.34**. Corrected: 16.34 -
    8.7172 = **7.62**, i.e. back inside the band every other boot occupies, and
    below sn6p's 9.01. That is the record whose 16.34 predicts ``run_peak
    92.53`` against the 87.30 boot bound and fires ``W21
    Weg2HostRunPeakRefused`` on every dry run of this form.

    ``None`` when the record has no residual, which stays an absence.
    """
    stored = entry.get("run_residual_gib")
    if stored is None:
        return None, 0.0
    arm = entry.get("arm")
    if not isinstance(arm, dict):
        return float(stored), 0.0
    # EVERY read of the arm is inside the guard, including this comparison.
    # An arm carrying a non-numeric `s_gb` raised ValueError here while the
    # try-block below was already written to swallow exactly that -- the guard
    # was one line short of the thing it guarded (found by its own test).
    try:
        s_d = arm.get("s_gb_d")
        bounce_gib = float(arm.get("xchg_bounce_gib", 0.0) or 0.0)
        img_gib = float(entry.get("rss_shmem_gib") or 0.0)
        images = ImageTerms(
            p_gib=img_gib, d_gib=img_gib, p_source="record", d_source="record",
            p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
        )
        # THE CORRECT SUM for the arm this record's boot actually ran: both
        # currency terms, one signature, one call (#1325 rings + #1326 bounce).
        new = _boot_charges_gib(
            charge_terms(
                int(arm["s_gb"]), int(arm["m_mib"]), ranks_per_group, images,
                s_gb_d=None if s_d is None else int(s_d),
                xchg_bounce_host_bytes=int(round(bounce_gib * GIB)),
            )
        )
        # WHAT THE WRITING SAMPLER SUBTRACTED. Read from the record when it
        # says so (#1326 `residual_charges_gib`), and only RECONSTRUCTED for a
        # record written before that field existed -- where the sampler
        # provably passed neither term, so the pre-#1325 sum is exact rather
        # than inferred. Preferring the stored figure is what keeps this sound
        # across the NEXT currency change: a record that already subtracted
        # correctly yields correction 0 without this function knowing anything
        # about which terms exist.
        stored_charges = entry.get("residual_charges_gib")
        if stored_charges is None:
            old = _boot_charges_gib(
                charge_terms(int(arm["s_gb"]), int(arm["m_mib"]),
                             ranks_per_group, images)
            )
        else:
            old = float(stored_charges)
    except (KeyError, TypeError, ValueError):
        # A record whose arm cannot be re-priced keeps its stored value. It is
        # the CONSERVATIVE direction (the stored figure is the larger one) and
        # never a silent 0.
        return float(stored), 0.0
    correction = new - old
    if abs(correction) < 1e-9:
        # Already in its own currency: every single-budget, unarmed record, and
        # every record written by the corrected sampler. Byte-identical, and
        # reported as exactly 0.0 so a caller's "was this re-priced" test is a
        # comparison and not a tolerance.
        return float(stored), 0.0
    return float(stored) - correction, correction


def run_origin_gib(
    cg_nonreclaim_gib: Optional[float], record: Optional[Dict[str, dict]] = None
) -> Tuple[Optional[float], str]:
    """The origin the RUN PEAK is predicted from, and where it came from.

    THE FIX-8 CORRECTION: the launch-moment reading is a FLOOR, not the origin.
    Measured on two boots one night apart -- weg2dk6 launched into 46.80 GiB and
    took M=600, weg2dk7 launched into 15.36 GiB, was handed M=1200 and a 9 GiB
    store for it, and idled at 90.1-94.6 GiB with the store EMPTY.  The quieter
    launch bought the tighter boot, because the box at the run moment holds
    terms that were simply not there yet at the launch moment.

    So the origin is ``max(launch reading, measured run-moment residual)``: a
    loaded box at launch still charges what it holds, and a quiet box may not
    buy an arm the run moment cannot carry.  ``None`` (and the reason) when no
    cgroup sample was passed -- a prediction without an origin is the reading
    that made weg2dk5 look fundable.
    """
    if cg_nonreclaim_gib is None:
        return None, "no cgroup sample passed -- no origin to add this arm's charges to"
    # #1325: RE-PRICED ON READ, never taken as stored. `record_run_residual_gib`
    # removes the double charge a two-budget record carries (D's rings sat
    # inside the residual because the sampler's subtraction omitted `s_gb_d`),
    # and is byte-identical for every single-budget record. The sidecar is
    # append-only, so the reader is the only place this can be corrected for a
    # record already on disk -- and the record blocking the next boot is one.
    _repriced = [
        (record_run_residual_gib(e), g, e)
        for g, e in (record or {}).items()
        if isinstance(e, dict) and e.get("run_residual_gib") is not None
    ]
    residuals = [
        (v, g, e, corr) for (v, corr), g, e in _repriced if v is not None
    ]
    if residuals:
        floor, group, entry, _corr = max(residuals, key=lambda r: r[0])
        # #1325: the chosen floor NAMES ITS LOAD CLASS and its form. Not a
        # selector -- the max() is unchanged -- but a boot whose origin came
        # from a sample taken under load must SAY so on its own ARM line, or
        # the next reader cannot tell an idle floor from a load peak and will
        # re-derive the sn6s confusion from scratch. `?` for a pre-#1325
        # record, which is a fact about the record and not a claim about the
        # box.
        floor_src = (
            f"MEASURED run-moment residual of boot {entry.get('boot_tag', '?')} @ "
            f"{entry.get('commit', '?')} ({entry.get('at', '?')}, group {group}, "
            f"load_class={entry.get('load_class', '?')}, "
            f"form={entry.get('form_key', '?')}, stored="
            f"{float(entry['run_residual_gib']):.2f}"
            + (
                f", RE-PRICED -{_corr:.2f} for S_D (#1325)" if _corr else ""
            )
            + ")"
        )
    else:
        floor = dk7_run_residual_gib()
        floor_src = (
            f"DERIVED from {DK7_PROVENANCE}: quiet memory.current "
            f"{DK7_QUIET_CG_CURRENT_GIB:.2f} GiB minus that boot's own charges at "
            f"S={DK7_ARM_S_GB} M={DK7_ARM_M_MIB} minus the measured image minus the "
            f"store's measured content {DK7_QUIET_STORE_USED_GIB:.2f} GiB"
        )
    if cg_nonreclaim_gib >= floor:
        return cg_nonreclaim_gib, (
            f"the launch-moment non-reclaimable reading {cg_nonreclaim_gib:.2f} GiB, which is "
            f"AT OR ABOVE the run-moment residual floor {floor:.2f} GiB [{floor_src}]"
        )
    return floor, (
        f"the RUN-MOMENT RESIDUAL FLOOR {floor:.2f} GiB [{floor_src}] -- the launch-moment "
        f"reading {cg_nonreclaim_gib:.2f} GiB is only a floor and a quieter launch does not "
        f"buy a bigger arm (weg2dk6 46.80 GiB launch -> M=600 -> died at 96.06; weg2dk7 "
        f"15.36 GiB launch -> M=1200 + 9 GiB store -> 90.1-94.6 GiB at IDLE, store EMPTY)"
    )


@dataclass
class Arm:
    s_gb: int
    m_mib: int
    ranks_per_group: int
    memtotal_bytes: int
    memavail_bytes: int
    terms: Dict[str, float] = field(default_factory=dict)
    launch_leftover_gib: float = 0.0
    run_leftover_gib: float = 0.0

    @property
    def launch_worst_case_gib(self) -> float:
        """The launch leftover with the WHOLE load transient charged as
        non-reclaimable -- i.e. as if the class split (#1317n) were wrong.

        Kept as a first-class term rather than a comment because it is the
        safety net of that split: the split says 12 GiB of the transient is
        page cache and may not be booked against the reap bound, and this is
        the number that answers "and if it isn't?".
        """
        t = self.terms or {}
        v = t.get("launch_leftover_sum_gib")
        return self.launch_leftover_gib if v is None else float(v)

    @property
    def fundable_moments(self) -> bool:
        """Both moments fund, AND the worst case stays inside the margin.

        #1317n THE THIRD CONJUNCT IS THE SPLIT'S SAFETY NET. Pricing the launch
        moment in the reap currency (page cache not charged against a bound the
        reaper does not enforce on page cache) turns weg2sn6r's -2.78 GiB into
        +9.2 GiB. If that class assignment is ever wrong, the honest fallback is
        not "boot anyway": the SUM form must still sit inside the named margin,
        so a configuration that would breach the reap watermark even once the
        page cache is reclaimed keeps being REFUSED. On sn6r the sum form is
        -2.78 against a margin of 8.60, so it holds with 5.82 GiB to spare --
        the split changes the verdict, and the worst case still has a floor.
        """
        if not (self.launch_leftover_gib >= 0.0 and self.run_leftover_gib >= 0.0):
            return False
        margin = (self.terms or {}).get("margin_total_gib")
        if margin is None:
            return True
        return self.launch_worst_case_gib >= -float(margin)

    def predicted_run_peak_gib(self) -> Optional[float]:
        """What ``memory.current`` this arm reaches at the RUN PEAK, or None.

        #1236: THERE IS NO STORE TERM IN THIS SUM ANY MORE, and its absence is
        the point rather than an omission.  The canonical page store was a
        tmpfs, so every byte of it was ``shmem`` inside this cgroup -- charged
        here in full, un-evictable (``SwapTotal`` 0 on this box), and therefore
        a direct subtrahend of the room the reaper leaves.  It is a plain
        directory on the ZFS dataset now (``launcher.STORE_ROOT``), so its cost
        to the host is PAGE CACHE and ARC: reclaimable, outside this cgroup's
        ``anon``, and not what the reaper kills for.  Charging it here would be
        a phantom RAM post for bytes that are on a disk.

        The sum of everything this arm actually charges to the cgroup -- the
        reserves (``floor``) are deliberately NOT in it, because a reserve is
        room kept free, not memory spent.  ``None`` when no cgroup sample was
        passed: the prediction has no origin to add to, and a number without
        its origin is exactly the reading that made weg2dk5 look fundable.

        THE HOST WEIGHTS TERM HERE IS ``Sigma H`` AND IT APPEARS ONCE.  Fix 5
        wrote this sum for the per-allocation form, where the run peak was one
        resident image PLUS the interleave's transient; C19's shared host ring
        makes those the same bytes -- the region is preallocated at ``Sigma H``
        and the legs copy through it -- so summing both would double-charge the
        very term this prediction exists to check.

        FIX 6: the origin is the NON-RECLAIMABLE reading, the same denominator
        :func:`price` budgets against.  Adding this arm's charges to a
        ``memory.current`` that carries page cache compares an inflated origin
        with :data:`OBSERVED_REAP_NONRECLAIM_BYTES`, a watermark that carries
        almost none -- two different quantities wearing one unit.

        FIX 8: that origin is the RUN-moment one (:func:`run_origin_gib`), and
        this number now REFUSES arms (W21) instead of advising about them.
        """
        origin = self.terms.get("run_origin_gib")
        if origin is None:
            return None
        t = self.terms
        return (
            float(origin)
            + _boot_charges_gib(t)
            + t["host_ring_gib"]
        )


#: #1236: ``size_store_gib`` and ``StoreSizing`` ARE DELETED, not shrunk and
#: not flagged.  They existed to answer "how much of the host RAM leftover may
#: the store tmpfs take, bounded by the reap point" -- train fix 3's residual.
#: The store is a directory on the ZFS dataset now and its size is the P KV
#: pool's own bytes (``launcher.plan_store``), so that question has no subject
#: left: there is no RAM residual to hand out, and a function that kept
#: computing one would be a second bookkeeping beside a size the pool already
#: fixes.  ``--store-min-gib`` went with it: a FLOOR on a RAM residual is
#: meaningless once the store is not funded from RAM.  What replaced the floor
#: is a HARD LAW instead of a knob -- ``max_size >= P pool bytes`` by
#: construction (#1236), checked against the disk and refused by name (W52).


def read_meminfo(path: str = "/proc/meminfo") -> Dict[str, int]:
    """MemTotal/MemAvailable/Shmem in BYTES from /proc/meminfo."""
    with open(path) as f:
        text = f.read()
    out: Dict[str, int] = {}
    for key, val in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M):
        out[key] = int(val) * 1024
    return out


def non_backup_host_bytes(group: str, s_gb: int, m_mib: int) -> int:
    """The host bytes ONE group holds that are NOT the flip backup image.

    FIX 1 (round 1) finding 3.  A sleeping group's RssShmem is its whole
    shared-memory residency: the TMS backup image the ring must hold PLUS the
    mamba anchor pools PLUS that group's half of the HiCache rings.  The last
    two are posted by name in :func:`price` (``anchors_gib``, ``rings_gib``) and
    charged against the same host budget, so a ring sized to an un-netted
    RssShmem charges them twice -- Sigma H walks from ~32 to ~42 GiB and the
    ledger W20-refuses at every rung, which is exactly A1-3's state for the old
    form.  This is not a second bookkeeping: it reads the SAME constants
    :func:`price` reads, and the split between the groups is the one those
    constants were measured as.

    * anchors: b0 measured six (rank x phase) pools at M=2400 as
      ``2.55/1.49/1.06 + 2.55/1.27/1.27 GB`` -- three per group, so a group
      holds its own triple, scaled linearly with M like :func:`price` does.
    * rings: record 1e, "group P owns the 2xS half, group D the 6xS half".

    ``group`` is ``"P"`` or ``"D"``; anything else raises rather than guessing.
    """
    if group not in ("P", "D"):
        raise ValueError(f"group must be 'P' or 'D', not {group!r}")
    scale = m_mib / ANCHORS_REFERENCE_M_MIB
    anchors = (ANCHORS_P_AT_2400_BYTES if group == "P" else ANCHORS_D_AT_2400_BYTES) * scale
    ring_mult = RING_P_MULT_GB_PER_S if group == "P" else RING_D_MULT_GB_PER_S
    rings = ring_mult * s_gb * GB
    # The pool overhead price() posts on top of anchors+rings is charged against
    # the same bytes, so it belongs to the same subtrahend.
    return int(round((anchors + rings) * (1.0 + HOST_POOL_OVERHEAD)))


def cg_reclaimable_bytes(stat: Dict[str, int]) -> Optional[int]:
    """The part of ``memory.current`` the kernel reclaims instead of killing for.

    ``reclaimable = (file - shmem) + slab_reclaimable``, from a parsed
    ``memory.stat``.  ``None`` -- never a guess and never 0 -- when any of the
    three terms is absent: a missing reading must reach :func:`price` as an
    absence so it can charge the WHOLE reading and say so.

    Why not ``current - file``: in cgroup v2 ``file`` INCLUDES ``shmem``, so
    that form gives back the tmpfs page store -- the canonical carrier -- as if
    it were free.  Measured on this box rather than recalled, see
    :data:`CGROUP_V2_FILE_INCLUDES_SHMEM_PROOF`: ``memory.stat file`` equals
    /proc/meminfo ``Cached`` to the byte and ``memory.stat shmem`` equals
    ``Shmem`` to the byte, with ``SwapTotal`` 0 -- so shmem cannot be evicted at
    all here and is charged as spent, while ``slab_reclaimable`` (0.73 GiB at
    that reading) is reclaimable by the same shrinker path as the page cache.
    ``anon`` and ``unevictable`` are never subtracted: both are exactly what the
    reaper kills to recover.
    """
    keys = ("file", "shmem", "slab_reclaimable")
    if any(stat.get(k) is None for k in keys):
        return None
    page_cache_ex_shmem = max(0, int(stat["file"]) - int(stat["shmem"]))
    return page_cache_ex_shmem + int(stat["slab_reclaimable"])


def read_cgroup(root: str = "/sys/fs/cgroup") -> Dict[str, Optional[int]]:
    """The cgroup2 memory facts the REAPER acts on, in BYTES.

    ``current`` / ``peak`` from ``memory.current`` / ``memory.peak``,
    ``oom_kill`` from ``memory.events``, and ``max`` from ``memory.max`` --
    ``None`` when that file says ``max``, i.e. when this cgroup publishes NO
    finite ceiling.  Inside this LXC container it does say ``max``, and that
    absence is a fact the caller must NAME (:func:`price` falls back to
    MemTotal and says so) rather than paper over with a constant.

    FIX 6 adds the ``memory.stat`` terms (``anon``, ``file``, ``shmem``,
    ``unevictable``, ``slab_reclaimable``) and the derived ``reclaimable``:
    ``memory.current`` alone cannot tell memory that is HELD from cache the
    kernel will hand back, and the ledger must charge only the former.

    Every key is ``None`` when its file is unreadable; an unreadable cgroup is
    never silently priced as an empty one.
    """

    def _int(name: str) -> Optional[int]:
        try:
            with open(f"{root}/{name}") as f:
                text = f.read().strip()
        except OSError:
            return None
        if text == "max":
            return None
        try:
            return int(text)
        except ValueError:
            return None

    oom: Optional[int] = None
    try:
        with open(f"{root}/memory.events") as f:
            m = re.search(r"^oom_kill (\d+)", f.read(), re.M)
        oom = int(m.group(1)) if m else None
    except OSError:
        oom = None

    stat: Dict[str, int] = {}
    try:
        with open(f"{root}/memory.stat") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    stat[parts[0]] = int(parts[1])
    except OSError:
        stat = {}

    out: Dict[str, Optional[int]] = {
        "current": _int("memory.current"),
        "peak": _int("memory.peak"),
        "max": _int("memory.max"),
        "oom_kill": oom,
        "reclaimable": cg_reclaimable_bytes(stat),
    }
    for key in ("anon", "file", "shmem", "unevictable", "slab_reclaimable"):
        out[key] = stat.get(key)
    return out


def price(
    memtotal_bytes: int,
    memavail_bytes: int,
    s_gb: int,
    m_mib: int,
    *,
    ranks_per_group: int = 3,
    ring_bytes: int = 0,
    ring_span1_bytes: int = 0,
    ring_absent_by_design: bool = False,
    cg_current_bytes: Optional[int] = None,
    reclaimable_bytes: Optional[int] = None,
    cg_ceiling_bytes: Optional[int] = None,
    # #1317n D's own L2 budget. None = "same as P", which is every recorded
    # arm and every pre-existing caller, so those are byte-identical.
    s_gb_d: Optional[int] = None,
    measured_record: Optional[Dict[str, dict]] = None,
    xchg_bounce_host_bytes: int = 0,
) -> Arm:
    """Price one arm at both moments.  Pure.

    ``ring_bytes`` = ``Sigma_c H(c)`` and ``ring_span1_bytes`` = ``Sigma_c
    image_P(c)`` are C19's replacement for the deleted backup constants, solved
    from the previous boot by :func:`sglang.srt.weg2.ring_table.solve`.  Both
    must be positive: a zero is not "a free flip", it is a missing measurement,
    and pricing it as zero is the shape that made boot weg2ls1b2 look fundable.

    THE FLIP TRANSIENT IS NOT A TERM HERE, and its absence is the ring (C19 +
    FLIPCOST A1-1/A1-3), not an omission.  Fix 5 replaced the "one chunk in
    flight" endpoint proxy with ``FLIP_HOST_TRANSIENT_GIB`` = 9.97 GiB, the
    measured PEAK of the old per-allocation flip.  The shared host ring removes
    the quantity that constant measured: the region is preallocated at
    ``Sigma H`` and the legs copy THROUGH it, so there is no transient stacked
    on top of an image -- ``run`` charges ``Sigma H`` once and that IS the peak.
    A1-3 is the same statement from the other side: the old form is infeasible
    on this host budget (image + transient ~48.7 GiB against Sigma H ~36 GiB).

    ``cg_current_bytes`` / ``cg_ceiling_bytes`` are the DENOMINATOR SWAP of
    fix 5 (boot weg2dk5): the reaper watches ``memory.current`` against the
    cgroup ceiling, not /proc/meminfo, so ``base`` is the TIGHTER of the two
    readings and ``base_source`` names which one bound.  Both ``None`` (a
    hermetic caller, or an unreadable cgroup) prices the meminfo arm alone and
    says so in ``base_source`` -- it never invents a ceiling.

    ``reclaimable_bytes`` is fix 6: what of that reading is page cache and
    reclaimable slab (:func:`cg_reclaimable_bytes`).  Only ``current`` MINUS
    that is charged, because only that is memory the reaper has to kill for.
    ``None`` means the ``memory.stat`` terms were unreadable: the whole reading
    is then charged -- the conservative direction -- and ``base_source`` names
    the absence rather than assuming a cache share.  (FIX 8 renamed this
    parameter: it used to be ``cg_reclaimable_bytes`` and SHADOWED the
    module-level function of that name inside the very scope whose docstring
    refers to it -- inert, and a loaded gun in the next edit.)

    ``measured_record`` is fix 8: this line's own previously measured dormant
    images and run-moment residuals (:func:`read_measured_record`).  ``None``
    prices the named dk7 reading for P and a BOUND for D, and says which is
    which -- see :func:`resolve_image_terms`.

    ``xchg_bounce_host_bytes`` is #1273 S6: the weight exchange's own pinned
    host carrier (:func:`xchg_bounce_bytes`), 0 on the ring arm and therefore
    on every boot that has run to date.  It is charged like any other term --
    the flip transient's absence above is about the RING and says nothing about
    a carrier the ring never had.
    """
    if s_gb < 1 or m_mib < 1:
        raise ValueError(f"arm terms must be >= 1: S={s_gb} M={m_mib}")
    # #1327 (S6 slice 2): ABSENCE OF A RING AND ABSENCE OF A MEASUREMENT ARE
    # TWO FACTS, and until now they shared one spelling (`ring_bytes == 0`).
    #
    # Under `--weg2-weight-source exchange` there IS no host weights ring: the
    # launcher publishes no `TMS_HOST_RING_*`, Sigma H is 0 BY DESIGN, and the
    # arming line says so (`ring_H_mib=0`). The refusal below is about a boot
    # whose PREDECESSOR logged no ring table, i.e. a number the planner would
    # have to guess -- and it must keep refusing that, because guessing is what
    # C19 deleted. Loosening the guard to `< 0` would let the stale-table case
    # through silently, which is the same defect one layer down.
    #
    # So the caller DECLARES the absence instead: `ring_absent_by_design` is
    # decided at the ONE launcher call site that already knows the arm, exactly
    # as `xchg_bounce_host_bytes` is ("a ledger that knew about arm names would
    # be a second reader of a decision that already has one"). The ledger never
    # reads the arm string.
    #
    # WHAT THIS RELEASES, and it is 2.7x what the S6 planning assumed: the
    # weights ring is the `host_ring_gib` term, MEASURED 32.19 GiB on this rig
    # at the run moment (`RUN MOMENT = the host weights term 32.19 GiB`), NOT
    # the 12.83 GiB `rings_gib` term -- that one is the HiCache host pools
    # (`RING_P_MULT_GB_PER_S * s_gb + RING_D_MULT_GB_PER_S * s_gb_d`) and the
    # exchange does not touch it. Two different rings, and the S6 headroom
    # claim was attached to the wrong one.
    if ring_absent_by_design:
        if ring_bytes or ring_span1_bytes:
            raise ValueError(
                "ring_absent_by_design with a non-zero ring "
                f"(ring_bytes={ring_bytes}, ring_span1_bytes={ring_span1_bytes}): "
                "the two inputs contradict each other, and a contradiction "
                "resolved silently is how a charged term becomes invisible"
            )
    elif ring_bytes <= 0 or ring_span1_bytes <= 0:
        raise Weg2HostLedgerRefused(
            "W20 Weg2HostLedgerRefused: the host weights term has no measured source "
            f"(ring_bytes={ring_bytes}, ring_span1_bytes={ring_span1_bytes}).  C19 "
            "deleted BACKUP_P_BYTES / BACKUP_D_BYTES because they were stale "
            "constants; the replacement is the previous boot's own per-card table "
            "(ring_table.solve), and there is no third option.  A boot whose "
            "predecessor logged no WEG2-CHUNK-BYTES / WEG2-FLIP-TAG lines cannot be "
            "priced -- the planner REFUSES to guess (R22) rather than inventing a "
            "number, and THIS BOOT REFUSES.  (FIX 2: this sentence used to name the "
            "OLD serial form as the thing that would still run, which was false -- "
            "that form is priced from the SAME table, one image plus one tag in "
            "flight, so with no table there is nothing left to fall back to and the "
            "launch stops here by name.)"
        )
    if ring_span1_bytes > ring_bytes:
        raise ValueError(
            f"span1 {ring_span1_bytes} > ring {ring_bytes}: span 1 is a PREFIX of the "
            "region (image_P(c) <= H(c) by construction)"
        )
    base_meminfo_gib = min(memavail_bytes / GIB, memtotal_bytes / GIB - CLI_RESERVE_GIB)
    base_cgroup_gib: Optional[float] = None
    cg_nonreclaim_bytes: Optional[int] = None
    stat_note = ""
    if cg_current_bytes is not None:
        # FIX 6: page cache inside ``memory.current`` is not spent memory.  The
        # reclaimable share is CLAMPED into [0, current]: the two files are read
        # microseconds apart, and a stat that momentarily exceeds the reading
        # must not turn into free memory this box never had.
        reclaim = 0
        if reclaimable_bytes is None:
            stat_note = (
                " [memory.stat unreadable: the WHOLE reading is charged, the "
                "conservative direction]"
            )
        else:
            reclaim = max(0, min(int(reclaimable_bytes), int(cg_current_bytes)))
        cg_nonreclaim_bytes = int(cg_current_bytes) - reclaim
    if cg_nonreclaim_bytes is not None and cg_ceiling_bytes is not None:
        # The CLI reserve is charged here too and for the same reason as on the
        # meminfo arm: ``memory.current`` nets out what the CLIs hold RIGHT NOW,
        # this term keeps the room they grow into.
        base_cgroup_gib = (
            (cg_ceiling_bytes - cg_nonreclaim_bytes) / GIB - CLI_RESERVE_GIB
        )
    if base_cgroup_gib is None:
        base_gib = base_meminfo_gib
        base_source = "meminfo (no cgroup sample passed)"
    elif base_cgroup_gib <= base_meminfo_gib:
        base_gib = base_cgroup_gib
        base_source = (
            "cgroup (ceiling - non-reclaimable memory.current - cli_reserve)"
            + stat_note
        )
    else:
        base_gib = base_meminfo_gib
        base_source = "meminfo (min(memavail, memtotal-cli))"
    # FIX 8's ONE AUTHORITY for what this arm charges beside the image -- the
    # same dict predicted_run_peak_gib adds and dk7_run_residual_gib subtracts,
    # so those three can no longer disagree about the same sum.  The draft tier
    # is in it: a HOST tier of its own, charged at BOTH moments, allocated at
    # load and outliving every flip (that is the point of carrying draft KV
    # across the flip).  It is NOT part of the flip image -- the ring carries
    # the weights, this carries the draft pages, never the same bytes.
    images = resolve_image_terms(measured_record)
    charges = charge_terms(s_gb, m_mib, ranks_per_group, images, s_gb_d=s_gb_d,
                           xchg_bounce_host_bytes=xchg_bounce_host_bytes)
    heaps_gib = charges["heaps_gib"]
    anchors_gib = charges["anchors_gib"]
    rings_gib = charges["rings_gib"]
    overhead_gib = charges["overhead_gib"]
    draft_host_p_gib = charges["draft_host_p_gib"]
    draft_host_d_gib = charges["draft_host_d_gib"]
    host_ring_gib = ring_bytes / GIB
    host_ring_span1_gib = ring_span1_bytes / GIB
    # THE MEASURED IMAGE IS REPORTED, NOT CHARGED A SECOND TIME (C19 x fix 8).
    # fix 8's measurement is right and it is what A1-2 rules the image must come
    # from -- but on the ring it enters through H(c): ring_table sizes each
    # card's granule from this module's own sidecar and price charges the
    # resulting Sigma H once.  Charging images.p/d here as well would charge the
    # same RssShmem bytes twice, which is precisely the double-charge ring fix 1
    # finding 3 removed.  They stay in the term list because A1-2 requires both
    # numbers (measured image and weight-tag census) to print.
    image_p_gib = images.p_gib
    image_d_gib = images.d_gib
    common = base_gib - FLOOR_GIB - _boot_charges_gib(charges)
    # R7: at the launch moment only span 1 is registered (P's first pause is the
    # launcher's sleep(P)); span 2 lands at D's first pause, when D's load
    # transient is gone.  Charging Sigma H at launch is what turns the M=1200
    # arm's leftover from +1.05 into -1.93 GiB.
    # #1317n ONE CURRENCY PER BARRIER. `base_source` names which reading won
    # the base, and that reading decides which half of the transient is a real
    # cost against it. `launch_sum` is kept and printed either way as the
    # WORST CASE (the whole transient non-reclaimable), and it is a hard gate
    # below -- so this is a class assignment, never a sum->max trick.
    launch_sum = common - host_ring_span1_gib - LOAD_TRANSIENT_GIB
    launch_reap = common - host_ring_span1_gib - LOAD_TRANSIENT_ANON_GIB
    _reap_currency = str(base_source or "").startswith("cgroup")
    launch = launch_reap if _reap_currency else launch_sum
    # Sigma H IS the run peak; there is no flip transient beside it (A1-1/A1-3).
    run = common - host_ring_gib
    origin_gib, origin_source = run_origin_gib(
        None if cg_nonreclaim_bytes is None else cg_nonreclaim_bytes / GIB,
        measured_record,
    )
    arm = Arm(
        s_gb=s_gb,
        m_mib=m_mib,
        ranks_per_group=ranks_per_group,
        memtotal_bytes=memtotal_bytes,
        memavail_bytes=memavail_bytes,
    )
    arm.terms = {
        "memtotal_gib": memtotal_bytes / GIB,
        "memavail_gib": memavail_bytes / GIB,
        "cli_reserve_gib": CLI_RESERVE_GIB,
        "base_gib": base_gib,
        "base_meminfo_gib": base_meminfo_gib,
        "base_cgroup_gib": base_cgroup_gib,
        "base_source": base_source,
        "cg_current_gib": (
            None if cg_current_bytes is None else cg_current_bytes / GIB
        ),
        "cg_reclaimable_gib": (
            None if reclaimable_bytes is None or cg_current_bytes is None
            else max(0, min(int(reclaimable_bytes), int(cg_current_bytes))) / GIB
        ),
        "cg_nonreclaim_gib": (
            None if cg_nonreclaim_bytes is None else cg_nonreclaim_bytes / GIB
        ),
        "cg_ceiling_gib": (
            None if cg_ceiling_bytes is None else cg_ceiling_bytes / GIB
        ),
        "floor_gib": FLOOR_GIB,
        # #1317n THE LABEL DEFECT, and it was in the very line this ticket
        # repaired. `arm.terms` is built here as a FRESH literal -- it is NOT
        # `charge_terms()`'s return -- so the `s_gb_d` key added there never
        # reached the ARM line, which then printed `S_D==S` while the ring
        # term proved S_D=4 (12.83 GiB is unreachable at S_D=1: (1.7778+3.0)
        # x0.931323 = 4.45). A correct price with a wrong label, i.e. exactly
        # the instrument-text class this fork keeps paying for: the boot agent
        # could verify the VALUE only by inverting the ring arithmetic.
        "s_gb_d": float(s_gb if s_gb_d is None else s_gb_d),
        "heaps_gib": heaps_gib,
        "host_ring_gib": host_ring_gib,
        "host_ring_span1_gib": host_ring_span1_gib,
        # #1327: WHY the term is 0, so a reader never has to guess whether
        # a zero means "no ring" or "no measurement". An unarmed gate must
        # never be readable as a passed one.
        "ring_absent_by_design": bool(ring_absent_by_design),
        "image_p_gib": image_p_gib,
        "image_d_gib": image_d_gib,
        "image_p_source": images.p_source,
        "image_d_source": images.d_source,
        "image_p_measured": images.p_measured,
        "image_d_measured": images.d_measured,
        "weight_tags_p_gib": charges["weight_tags_p_gib"],
        "weight_tags_d_gib": charges["weight_tags_d_gib"],
        "image_extra_p_gib": images.extra_p_gib,
        "image_extra_d_gib": images.extra_d_gib,
        "run_origin_gib": origin_gib,
        "run_origin_source": origin_source,
        # #1317n the margin the worst-case launch gate measures against. The
        # DEFAULT resolution, so `price()` alone can enforce the gate; `choose`
        # may pass a different Margin, and it re-reads the same field.
        "margin_total_gib": resolve_margin().boot_total_gib,
        "load_transient_gib": LOAD_TRANSIENT_GIB,
        "load_transient_pagecache_gib": LOAD_TRANSIENT_PAGECACHE_GIB,
        "load_transient_anon_gib": LOAD_TRANSIENT_ANON_GIB,
        "launch_leftover_sum_gib": launch_sum,
        "launch_leftover_reap_gib": launch_reap,
        "launch_currency": "reap" if _reap_currency else "memavailable",
        "anchors_gib": anchors_gib,
        "rings_gib": rings_gib,
        "overhead_gib": overhead_gib,
        "draft_host_p_gib": draft_host_p_gib,
        "draft_host_d_gib": draft_host_d_gib,
        # #1273 S6.  THE KEY IS ALWAYS HERE, 0.0 on the ring arm, because
        # ``_boot_charges_gib`` indexes it and ``predicted_run_peak_gib`` sums
        # that -- a key that can be absent is how the three consumers this
        # dict exists to reconcile would start disagreeing again.
        "xchg_bounce_gib": charges["xchg_bounce_gib"],
        "store_draft_fraction": STORE_DRAFT_FRACTION,
    }
    arm.launch_leftover_gib = launch
    arm.run_leftover_gib = run
    return arm


# --------------------------------------------------------------------------
# FIX 8: measuring the dormant image instead of summing tags for it
# --------------------------------------------------------------------------

#: The sidecar this line writes its own measurements into, under the launcher's
#: log directory.  There was NO such mechanism before fix 8: the fix-5/6/7
#: constants are literals in this module carrying their boot and timestamp in a
#: comment, which is honest but cannot close the loop -- a boot could not hand
#: its successor a number.  The sidecar is that loop, and it stores WHO measured
#: (commit, boot tag, timestamp), never a bare figure.
MEASURED_RECORD_NAME = "weg2_measured_record.json"


def read_cgroup_pressure(root: str = "/sys/fs/cgroup") -> Dict[str, Optional[float]]:
    """NON-RECLAIMABLE PRESSURE in GiB, with the raw reading beside it.

    #1269 fix 3 -- the defect that refused boot weg2sb5b 28 GiB below danger.
    ``memory.current`` counts RECLAIMABLE page cache, which the kernel drops
    before it ever OOMs; the run-peak model is built from anon + shmem. The two
    cannot be compared during a load, and sb5b is the proof (launcher memts,
    host-wide /proc/meminfo terms):

        peak 18:57:01Z  memory.current 96.94 GiB
                      = anon 27.72 + shmem 40.50 (rings+store)
                      + page cache excl. shmem 28.08 + ~0.64 slab/kernel
        NON-RECLAIMABLE 68.86 GiB; max over the boot 78.26 GiB

    and that 28 GiB of cache was ALREADY RESIDENT BEFORE THE BOOT -- pre-boot
    28.53 -> peak 28.08, it FELL. Foreign, reclaimable, and nothing the
    checkpoint load put there; the boot's own growth was 57.4 GiB and entirely
    anon + shmem. Against the 87.30 GiB bound the honest figure had 9.04 GiB of
    room while the raw reading breached by 9.64.

    WHY THE WATERMARK CARRIES OVER UNCHANGED, as a derivation with its
    assumption rather than an assertion: at a real reap the kernel has ALREADY
    reclaimed the file cache -- that is what reclaim IS, and it runs before the
    OOM killer -- so ``memory.current`` at the moment of the kill is already
    ~pure non-reclaimable. dk5's own reap row bears this out: its
    ``pagecache_ex_shmem`` is 28,916 kB = 0.03 GiB. So 95.90 / 96.06 are
    already non-reclaimable readings and need no restatement.
    ASSUMPTION: that dk6's row behaves as dk5's did. dk6's memory.stat was not
    captured, so this is inference from ONE measured row, not two.

    Preferred formula ``current - inactive_file - active_file`` (the two file
    LRUs are exactly what reclaim walks). The sum form
    ``anon + shmem + slab_unreclaimable + unevictable`` is the fallback when
    those fields are absent, and is reported as such: it can differ from the
    first by kernel-internal terms.
    """
    out: Dict[str, Optional[float]] = {
        "current_gib": None,
        "nonreclaim_gib": None,
        "file_reclaimable_gib": None,
        "anon_gib": None,
        "shmem_gib": None,
        "source": None,
    }
    st: Dict[str, int] = {}
    try:
        with open(f"{root}/memory.stat") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    st[parts[0]] = int(parts[1])
    except OSError:
        st = {}
    try:
        with open(f"{root}/memory.current") as f:
            out["current_gib"] = int(f.read().strip()) / GIB
    except (OSError, ValueError):
        pass
    if st:
        out["anon_gib"] = st.get("anon", 0) / GIB
        out["shmem_gib"] = st.get("shmem", 0) / GIB
        # The memts sampler dumps these column names per row from the next boot
        # on; carrying them VERBATIM is what lets acceptance diff the guard's
        # line against the sampler without a translation table.
        for k in (
            "file", "slab_reclaimable", "slab_unreclaimable",
            "inactive_file", "active_file", "unevictable",
        ):
            out[f"{k}_gib"] = st[k] / GIB if k in st else None
    if out["current_gib"] is not None and "inactive_file" in st and "active_file" in st:
        filec = (st["inactive_file"] + st["active_file"]) / GIB
        out["file_reclaimable_gib"] = filec
        out["nonreclaim_gib"] = out["current_gib"] - filec
        out["source"] = "memory.current - inactive_file - active_file"
    elif st:
        out["nonreclaim_gib"] = (
            st.get("anon", 0)
            + st.get("shmem", 0)
            + st.get("slab_unreclaimable", 0)
            + st.get("unevictable", 0)
        ) / GIB
        if out["current_gib"] is not None:
            out["file_reclaimable_gib"] = out["current_gib"] - out["nonreclaim_gib"]
        out["source"] = (
            "anon+shmem+slab_unreclaimable+unevictable FALLBACK "
            "(inactive_file/active_file absent; may differ by kernel-internal terms)"
        )
    else:
        out["source"] = "memory.stat unreadable -- no non-reclaimable reading"
    return out


def read_cgroup_anon_bytes(root: str = "/sys/fs/cgroup") -> Optional[int]:
    """``memory.stat anon`` in bytes, or ``None`` when unreadable.

    The denominator :func:`measure_foreign_anon` splits: anon is the currency
    the idle grower (#1269) and the desk load both move in, while the store and
    the host ring sit in ``shmem`` and are flat.
    """
    try:
        with open(f"{root}/memory.stat") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] == "anon" and parts[1].isdigit():
                    return int(parts[1])
    except OSError:
        return None
    return None


def read_cgroup_shmem_bytes(root: str = "/sys/fs/cgroup") -> Optional[int]:
    """``memory.stat shmem`` in bytes, or ``None`` when unreadable.

    This is the whole-cgroup figure the dk7 sampler measured 48.33 GiB of, and
    it equals ``/proc/meminfo Shmem`` to the byte on this box (see
    :data:`CGROUP_V2_FILE_INCLUDES_SHMEM_PROOF`).
    """
    try:
        with open(f"{root}/memory.stat") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] == "shmem" and parts[1].isdigit():
                    return int(parts[1])
    except OSError:
        return None
    return None


def rss_shmem_bytes(pids: Iterable[int]) -> Tuple[int, List[int]]:
    """Sum ``RssShmem`` over the given pids; return (bytes, the pids that answered).

    ``RssShmem`` is where the TMS CPU backup IS visible: it is anonymous
    ``MAP_SHARED``, so it has NO backing file and ``df``/``du`` on /dev/shm show
    nothing (dk7: 44.56 GiB of 48.33 GiB of cgroup shmem had no file).  A pid
    that has exited between the listing and the read is skipped and NOT counted
    as zero -- the returned pid list is the denominator of the sum.
    """
    total = 0
    seen: List[int] = []
    for pid in pids:
        try:
            with open(f"/proc/{int(pid)}/status") as f:
                text = f.read()
        except OSError:
            continue
        m = re.search(r"^RssShmem:\s+(\d+) kB", text, re.M)
        if m is None:
            continue
        total += int(m.group(1)) * 1024
        seen.append(int(pid))
    return total, seen


def dormant_image_sample(
    *,
    group: str,
    shmem_before_bytes: Optional[int],
    shmem_after_bytes: Optional[int],
    pids: Sequence[int],
    weight_tags_gib: float,
    interleaved: bool,
    boot_tag: str,
    commit: str,
    at: Optional[str] = None,
    cg_current_bytes: Optional[int] = None,
    reclaimable_bytes: Optional[int] = None,
    arm: Optional[Dict[str, float]] = None,
    ranks_per_group: int = 3,
    load_witness: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """One group's dormant image, measured at its FIRST sleep.  Pure but for /proc.

    TWO INSTRUMENTS, and they are not interchangeable:

    * ``rss_shmem_gib`` -- the sum of the now-sleeping group's per-rank
      ``RssShmem``.  This is the AUTHORITY and the term the ledger charges;
    * ``shmem_delta_gib`` -- the cgroup ``shmem`` delta across the sleep.  A
      CROSS-CHECK, and ``interleaved`` says when it is confounded: during a
      flip the destination group is RESUMING (its own image is being freed by
      the patched saver) while the source sleeps, so the delta is the
      difference of two images and NOT this group's image.  The launcher's
      first sleep of P is un-interleaved (D does not exist yet) and there the
      two instruments are comparable.

    ``run_residual_gib`` is derived only when the caller supplies the run
    moment (a cgroup reading and the arm): what the box holds that this
    ledger's term list does not name.  ``None`` with a stated reason
    otherwise -- never 0.

    #1236 REMOVED THE STORE FROM THIS SUBTRACTION, and it is the same finding
    as in :meth:`Arm.predicted_run_peak_gib`, one layer down.  The store used
    to be a tmpfs, so its content WAS inside ``cg_current_bytes`` as
    un-evictable ``shmem`` and subtracting it was correct.  It is a directory
    on the ZFS dataset now: its bytes are not in this cgroup at all, so
    subtracting them would credit the residual with memory the reading never
    contained and make an unexplained term look explained.  The caller's
    ``store_used_bytes`` argument went with it -- on a shared dataset the
    ``statvfs`` that produced it reports the WHOLE root filesystem's usage,
    which is not this store and never was this quantity.
    """
    delta = (
        None
        if shmem_before_bytes is None or shmem_after_bytes is None
        else (int(shmem_after_bytes) - int(shmem_before_bytes)) / GIB
    )
    rss, seen = rss_shmem_bytes(pids)
    rss_gib = rss / GIB
    residual: Optional[float] = None
    # Initialised here and not only in the branch below: the record's
    # `residual_charges_gib` reads it, and a name that exists on exactly one
    # path is a NameError waiting for the next editor (#1326).
    _charges_gib: Optional[float] = None
    residual_note = ""
    if cg_current_bytes is None or arm is None:
        residual_note = (
            "not the run moment (need a cgroup reading and the arm); NOT derived, "
            "and not 0"
        )
    else:
        reclaim = 0 if reclaimable_bytes is None else max(
            0, min(int(reclaimable_bytes), int(cg_current_bytes))
        )
        nonreclaim_gib = (int(cg_current_bytes) - reclaim) / GIB
        images = ImageTerms(
            p_gib=rss_gib, d_gib=rss_gib, p_source="this sample", d_source="this sample",
            p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
        )
        # #1325: THE SUBTRACTION MUST BE IN THE CURRENCY THE BOOT ACTUALLY RAN.
        # This call omitted `s_gb_d`, so on a two-budget boot it subtracted the
        # rings of an S_D == S arm while the boot held S_D=4 of them -- and the
        # unsubtracted difference landed in `run_residual_gib`, i.e. in the one
        # quantity whose entire definition is "what the box holds that this
        # ledger's term list does NOT name". The ledger then adds those same
        # rings back at prediction time, so they are charged twice.
        #
        # MEASURED with charge_terms itself, sn6s's arm (S=1 M=600 S_D=4,
        # ranks=3): boot charges 24.5047 GiB without `s_gb_d` against 33.2219
        # with it -- 8.7172 GiB under-subtracted. sn6s's recorded run residual
        # of 16.34 GiB corrects to 7.62, BELOW sn6p's idle 9.01, so
        # `run_origin_gib`'s max() keeps 9.01 and the arm prices exactly as it
        # did when (g) passed. The uncorrected 16.34 is what predicts
        # run_peak 92.53 against the 87.30 boot bound and fires W21
        # Weg2HostRunPeakRefused on EVERY dry run of this form.
        #
        # `arm` carries the key already (`price()` writes `s_gb_d` into its
        # terms, host_ledger.py:1778), so nothing new is measured or plumbed --
        # a value that was present at the call site was simply not passed.
        # `.get` and not `[...]`: a pre-#1325 arm dict has no such key and must
        # keep its single-budget subtraction, which is byte-identical because
        # `charge_terms` defaults `_s_d` to `s_gb`.
        #
        # #1326 (the S6c merge): THE XCHG BOUNCE IS THE SECOND TERM OF THE SAME
        # DEFECT, and it had to be added here in the same breath rather than
        # left for a later ticket. `xchg_bounce_gib` is an ARM charge -- it is
        # summed by `_boot_charges_gib` and added back by
        # `predicted_run_peak_gib` -- so an armed boot whose sampler did not
        # subtract it leaves the deposit inside `run_residual_gib` and the
        # ledger charges it TWICE at the run moment, exactly as it did with D's
        # rings. Nothing about the bounce being small makes that safe: the
        # error is structural, not proportional. ONE currency, both terms, one
        # signature (`charge_terms(..., s_gb_d=, xchg_bounce_host_bytes=)`) and
        # one call path -- there is deliberately no second sampler arm for the
        # armed case.
        _arm_s_d = arm.get("s_gb_d")
        _arm_bounce_bytes = int(round(float(arm.get("xchg_bounce_gib", 0.0) or 0.0) * GIB))
        charges = charge_terms(
            int(arm["s_gb"]), int(arm["m_mib"]), ranks_per_group, images,
            s_gb_d=None if _arm_s_d is None else int(_arm_s_d),
            xchg_bounce_host_bytes=_arm_bounce_bytes,
        )
        # STORED, so the reader never has to GUESS which currency wrote a
        # record (#1326). `record_run_residual_gib` re-prices records that are
        # already on disk, and it can only do that soundly if it knows the sum
        # the writing sampler actually subtracted. Reconstructing that from the
        # arm means encoding "which version of this function ran", which is a
        # guess that goes stale on the next currency change -- and this is the
        # second currency change in one day. A record that carries the figure
        # needs no version inference at all: correction = stored_charges -
        # correct_charges, and it is 0 for a record already in its own currency.
        _charges_gib = _boot_charges_gib(charges)
        residual = nonreclaim_gib - _charges_gib - rss_gib
        residual_note = (
            f"nonreclaimable {nonreclaim_gib:.2f} minus this boot's own charges "
            f"{_charges_gib:.2f} (at S={int(arm['s_gb'])} "
            f"S_D={int(_arm_s_d) if _arm_s_d is not None else int(arm['s_gb'])} "
            f"M={int(arm['m_mib'])}, rings={charges['rings_gib']:.2f}, "
            f"xchg_bounce={charges['xchg_bounce_gib']:.2f}) minus the "
            f"measured image {rss_gib:.2f} (#1325/#1326: the subtraction carries "
            f"S_D AND the xchg bounce; omitting either left an arm charge inside "
            f"the residual, to be charged again at the run moment)"
        )
    return {
        "group": group,
        "at": at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boot_tag": boot_tag,
        "commit": commit,
        "shmem_before_bytes": shmem_before_bytes,
        "shmem_after_bytes": shmem_after_bytes,
        "shmem_delta_gib": delta,
        "rss_shmem_gib": rss_gib,
        "weight_tags_gib": weight_tags_gib,
        "extra_gib": rss_gib - weight_tags_gib,
        "pids": seen,
        "pids_asked": [int(p) for p in pids],
        "interleaved": bool(interleaved),
        "cg_current_bytes": cg_current_bytes,
        "arm": arm,
        "run_residual_gib": residual,
        "run_residual_note": residual_note,
        # #1326: THE SUM THIS SAMPLER SUBTRACTED, so a reader can re-price the
        # record without inferring which version of this function wrote it.
        # None when no run moment was available (the residual is None too).
        "residual_charges_gib": _charges_gib,
        # #1325: WHAT THE BOX WAS DOING WHEN THIS WAS SAMPLED, and WHICH
        # SERVING FORM it speaks for. Both RECORDED, neither yet a selector of
        # a different number -- and that ordering is deliberate. The reading
        # itself was wrong (the S_D subtraction above), and a load-class or
        # form-key rule layered on top of a wrong reading would have priced a
        # corrected sample by a rule built for the uncorrected one.
        #
        # `load_class` is MEASURED, never inferred from the clock: the front
        # passes the queue depth and the in-flight count it already holds at
        # the sampling moment, and anything above zero is `loaded`. A sample
        # taken at a flip under load is `loaded` by construction, which is
        # exactly the sn6s case (its record was taken at 15:21:55Z, mid-load).
        # `unknown` when no witness was passed -- never `idle`, because an
        # absent witness is not a quiet box.
        #
        # `form_key` names the terms that fix the HOST posten and are
        # independent of the arm being priced: the ranks per group and the
        # measured weight-tag sum. It deliberately does NOT include the arm --
        # the residual is already arm-normalised by the subtraction above, so
        # keying selection on the arm would be circular (you would need the arm
        # to find the record that prices the arm). Shadow/exchange forms do not
        # change either term, so they share a key, which is what the operator's
        # reading of the form requires.
        "load_class": (
            "unknown"
            if not load_witness
            else (
                "loaded"
                if sum(int(v or 0) for v in load_witness.values()) > 0
                else "idle"
            )
        ),
        "load_witness": dict(load_witness or {}),
        "form_key": f"ranks={int(ranks_per_group)};wtags={weight_tags_gib:.2f}",
    }


def format_dormant_image(rec: Dict[str, object]) -> str:
    """The one log line the next boot's ledger reads its image term from."""
    delta = rec.get("shmem_delta_gib")
    return (
        f"WEG2 DORMANT-IMAGE group={rec['group']} "
        f"shmem_delta_gib={'unreadable' if delta is None else f'{float(delta):.2f}'} "
        f"rss_shmem_gib={float(rec['rss_shmem_gib']):.2f} "
        f"weight_tags_gib={float(rec['weight_tags_gib']):.2f} "
        f"extra_gib={float(rec['extra_gib']):.2f} "
        f"(pids {rec['pids']} of {rec['pids_asked']}; "
        + (
            "INTERLEAVED: the shmem delta is confounded -- the destination group was "
            "resuming (freeing its own image) while this one slept, so only "
            "rss_shmem_gib measures THIS group's image"
            if rec.get("interleaved")
            else "un-interleaved: no other group was resuming, so the two instruments "
            "are comparable"
        )
        + "; run_residual_gib="
        + (
            f"{float(rec['run_residual_gib']):.2f}"
            if rec.get("run_residual_gib") is not None
            else f"none ({rec.get('run_residual_note', '')})"
        )
        + f"; boot {rec['boot_tag']} @ {rec['commit']} at {rec['at']})"
    )


def read_measured_record(
    path: str, boot_tag: Optional[str] = None
) -> Dict[str, dict]:
    """The newest entry per group from the sidecar, or ``{}``.

    A malformed or missing file is an ABSENCE -- the ledger then prices the
    named dk7 reading and says so -- never a silent zero.

    ``boot_tag`` RESTRICTS THE ANSWER TO ONE BOOT'S SAMPLES, and #1264 (B) is
    why the parameter exists. This sidecar is APPEND-ONLY across boots, so
    "the newest entry" is a sample from whatever booted last -- while the
    consumer that corrects it (:func:`non_backup_host_bytes`, fed by the arm
    that :func:`ring_table.parse_chosen_arm` reads out of the SOURCE boot's own
    front log) is keyed to a different boot entirely. Measured drift from that
    mispaired join, three consecutive solves of the SAME source boot weg2rg6:
    the nvml2 residual read 118.2 -> 901.2 -> 1344.2 MiB.

    It is a RATCHET, not noise. The sample is the sleeping group's whole
    ``RssShmem``, which includes the host ring's own MAP_SHARED pages; a bigger
    ring makes a bigger sample, which sizes a bigger ring for the next boot.
    Boot weg2t2a measured ``extra_gib`` 7.474, weg2t2b 8.995, on identical
    ``weight_tags_gib`` 28.834. That is what walked Sigma span1 +2755 MiB
    between two boots that were supposed to differ only in an idle census, took
    the ledger from M=1200 to M=600, and left 0.74 GiB to the reap watermark.

    Passing the tag makes sample and correction ONE boot's pair, which is what
    the correction was always documented to be, and makes a re-solve from the
    same source deterministic forever.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    entries = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return {}
    out: Dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("rss_shmem_gib") is None:
            continue
        g = str(e.get("group", ""))
        if not g:
            continue
        if boot_tag is not None and str(e.get("boot_tag", "")) != boot_tag:
            continue
        if g not in out or str(e.get("at", "")) >= str(out[g].get("at", "")):
            out[g] = e
    return out


def append_measured_record(path: str, rec: Dict[str, object]) -> None:
    """Append one sample to the sidecar (append-only: history is evidence)."""
    try:
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples") if isinstance(data, dict) else None
    except (OSError, ValueError):
        samples = None
    if not isinstance(samples, list):
        samples = []
    samples.append(rec)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"samples": samples}, f, indent=1, default=str)
    os.replace(tmp, path)


def resolve_cg_ceiling(
    cgroup: Dict[str, Optional[int]], memtotal_bytes: int
) -> Tuple[Optional[int], str]:
    """The ceiling ``memory.current`` is budgeted against, and WHERE it came from.

    A finite ``memory.max`` is the ceiling and says so.  When there is none --
    the case inside this LXC container, where ``memory.max`` reads ``max`` --
    the ceiling falls back to the container's lxcfs ``MemTotal`` and the source
    string SAYS it is a fallback, because that fallback is knowably loose: boot
    weg2dk5 was reaped at memory.current 95.93 GiB against a MemTotal of
    118.05 GiB, i.e. ~22 GiB below this ceiling.  That looseness is why the
    run-peak advisory (which compares against the OBSERVED reap point) exists
    beside it rather than being folded into the ceiling as a fudge.
    """
    if cgroup.get("max") is not None:
        return int(cgroup["max"]), "cgroup memory.max"
    return (
        int(memtotal_bytes),
        "FALLBACK lxcfs MemTotal (memory.max is 'max': this cgroup publishes no "
        "finite ceiling; loose by ~22 GiB against weg2dk5's observed reap point)",
    )


def _gib_or_none(value: Optional[float]) -> str:
    """An absent reading prints as ``unreadable``, never as 0.00 GiB."""
    return "unreadable" if value is None else f"{value:.2f} GiB"


def _advisory_line(arm: Arm, chosen: bool) -> str:
    """The RUN-PEAK line, printed for the CHOSEN arm or -- on a total refusal --
    for the most frugal arm on the ladder.

    FIX 8 prints it in BOTH cases on purpose: everything this line states about
    the estimator (its residual on three boots, the watermark's own unsampled
    slab term, the direction of both) is exactly what a reader of a REFUSAL
    needs, and a refusal is the outcome on this box today.  Emitting it only on
    success would delete the explanation at the moment it is wanted.
    """
    watermark_gib = OBSERVED_REAP_NONRECLAIM_BYTES / GIB
    predicted = arm.predicted_run_peak_gib()
    subject = "this arm" if chosen else f"the most frugal arm (S={arm.s_gb} M={arm.m_mib})"
    if predicted is None:
        return (
            "WEG2-HOST-LEDGER RUN-PEAK ADVISORY: not computed -- no cgroup sample was "
            "passed, so there is no origin to add this arm's charges to. An absent "
            "prediction is stated, never printed as a pass."
        )
    verdict = "ABOVE" if predicted > watermark_gib else "below"
    return (
        f"WEG2-HOST-LEDGER RUN-PEAK ADVISORY: {subject} predicts non-reclaimable memory.current="
        f"{predicted:.2f} GiB at the run peak (run origin {_gib_or_none(arm.terms['run_origin_gib'])} "
        f"[{arm.terms['run_origin_source']}] + heaps + anchors + rings + "
        f"overhead + draft pools + the host ring Sigma H; the {FLOOR_GIB:.0f} GiB floor "
        f"is a reserve and is NOT in this sum, and #1236 REMOVED the store term -- the "
        f"page store is a directory on the ZFS dataset now, page cache and ARC, "
        f"reclaimable, outside this cgroup's anon), "
        f"which is {verdict} the OBSERVED REAP POINT {watermark_gib:.2f} GiB "
        "(boot weg2dk5 21:15:30Z, memory.current 102,998,904,832 B minus the 28,916 kB "
        "of that row that was still reclaimable, with oom_kill 18 -> 24 in the same row; "
        "that row has NO slab_reclaimable column, so this watermark is an UPPER bound and "
        "this verdict UNDER-warns by that unsampled term -- 0.53 GiB live on this box "
        "2026-09-07T23:57:35Z, 0.73 GiB at the fix-6 reading). "
        "FIX 8: THIS LINE IS NO LONGER ONLY ADVISORY -- an arm whose predicted peak is "
        "not below the watermark is REFUSED by name (W21 Weg2HostRunPeakRefused); the "
        "sentence below is what that refusal is calibrated against. THE RESIDUAL SERIES, "
        "all three boots side by side: weg2dk5 predicted 92.89 against 95.90 reached "
        "(3.01 GiB UNDER-prediction); weg2dk6 predicted 91.23 against 96.06 sampled / 98.90 "
        "kernel peak (4.83 / 7.67 GiB UNDER) and died the same death; weg2dk7 predicted "
        "91.92 for the RUN peak and measured 90.10-94.57 GiB AT IDLE with an EMPTY store "
        "and zero flips. dk7 also EXPLAINED that series rather than extending it: the "
        "dormant image measures 38.63 GiB against the 28.83 GiB the weight-tag census "
        "priced (+9.80 GiB on ONE image), which covers dk6's 4.83-7.67 GiB under-"
        "prediction on its own and with no co-residency needed. That term is priced here "
        "from this boot on, so this residual is the image correction's own test."
    )


def choose(
    memtotal_bytes: int,
    memavail_bytes: int,
    *,
    arms: Sequence[Tuple[int, int]] = DEFAULT_ARMS,
    ranks_per_group: int = 3,
    ring_bytes: int = 0,
    ring_span1_bytes: int = 0,
    ring_absent_by_design: bool = False,
    ring_provenance: str = "",
    cg_current_bytes: Optional[int] = None,
    reclaimable_bytes: Optional[int] = None,
    slab_reclaimable_bytes: Optional[int] = None,
    cg_ceiling_bytes: Optional[int] = None,
    cg_ceiling_source: str = "",
    cg_oom_kill: Optional[int] = None,
    # #1317n D's own L2 budget. It rides the SAME kwargs block the launcher
    # hands to both `choose` and `price` -- boot weg2sn6a died of a price()
    # call that had drifted from choose()'s keywords, so a term reaching only
    # one of them is exactly the defect this signature exists to prevent.
    s_gb_d: Optional[int] = None,
    # #1317n the terms D's L2 was derived from, so a W20 can name the LARGEST
    # FUNDABLE CAP instead of leaving the operator to invert the arithmetic
    # (order item 1: "kein stilles Verkleinern" -- but also no silent silence).
    d_cap_terms: Optional[Dict[str, float]] = None,
    measured_record: Optional[Dict[str, dict]] = None,
    margin: Optional[Margin] = None,
    xchg_bounce_host_bytes: int = 0,
) -> Tuple[Arm, Optional[float], List[str]]:
    """Walk the ladder; return (arm, reap headroom GiB, printed lines) or W20/W21.

    #1236 CHANGED THE MIDDLE OF THAT TUPLE, and the change is the whole point
    of this pass: it used to be ``store_gib``, the size of the tmpfs the store
    lived on, computed here as ``min(run leftover, reap bound)``.  The store is
    a directory on the ZFS dataset now (``launcher.STORE_ROOT``) and is sized
    from the P KV pool, not from host RAM, so this function no longer sizes it,
    no longer charges it in :meth:`Arm.predicted_run_peak_gib`, and no longer
    carries a ``--store-min-gib`` floor.  What comes back instead is the REAP
    HEADROOM -- ``hard bound - predicted run peak`` for the chosen arm, or
    ``None`` when no cgroup sample let a peak be predicted at all.  That is a
    quantity this function actually computes, and it is what the caller needs
    to price the one host-RAM risk the disk store does carry (the ZFS ARC).

    ``slab_reclaimable_bytes`` is the LIVE ``memory.stat slab_reclaimable``
    reading -- the term the reap watermark's own row lacked.  It is subtracted
    from the watermark wherever the watermark is used as a bound, and an
    unreadable one subtracts nothing and says so.

    The cgroup arguments are fix 5's denominator (see :func:`price`).
    ``cg_oom_kill`` is printed as the PRE-BOOT BASELINE of a counter that is
    cumulative and carries no timestamps: a death can only ever be attributed
    by baseline-vs-after diff, and taking that baseline here means no boot can
    start without one.
    """
    lines: List[str] = []
    # TRAIN FIX 3: the term the reap watermark's own row LACKED, read live by
    # the caller from memory.stat and never a constant.  It is the only
    # quantity subtracted when the store's reap bound is computed.
    unsampled_gib = (
        None if slab_reclaimable_bytes is None else slab_reclaimable_bytes / GIB
    )
    priced = [
        price(
            memtotal_bytes,
            memavail_bytes,
            s,
            m,
            ranks_per_group=ranks_per_group,
            ring_bytes=ring_bytes,
            ring_span1_bytes=ring_span1_bytes,
            ring_absent_by_design=ring_absent_by_design,
            cg_current_bytes=cg_current_bytes,
            reclaimable_bytes=reclaimable_bytes,
            cg_ceiling_bytes=cg_ceiling_bytes,
            s_gb_d=s_gb_d,
            measured_record=measured_record,
            xchg_bounce_host_bytes=xchg_bounce_host_bytes,
        )
        for s, m in arms
    ]
    t = priced[0].terms
    lines.append(
        "WEG2-HOST-LEDGER TERMS "
        f"memtotal={t['memtotal_gib']:.2f} GiB memavail={t['memavail_gib']:.2f} GiB "
        f"(live /proc/meminfo) cli_reserve={CLI_RESERVE_GIB:.0f} GiB (record 1g, "
        f"charged once against MemTotal) "
        f"cgroup memory.current={_gib_or_none(t['cg_current_gib'])} ceiling={_gib_or_none(t['cg_ceiling_gib'])}"
        f"{(' [' + cg_ceiling_source + ']') if cg_ceiling_source else ''} "
        f"oom_kill_baseline={'unreadable' if cg_oom_kill is None else cg_oom_kill} "
        f"(#1233 fix 5, boot weg2dk5: the REAPER watches memory.current, not /proc/meminfo -- "
        "it killed six ranks with MemAvailable still at 23.96 GB) "
        f"of which reclaimable={_gib_or_none(t['cg_reclaimable_gib'])} "
        "(memory.stat: (file - shmem) + slab_reclaimable; cgroup-v2 `file` INCLUDES `shmem` "
        f"-- {CGROUP_V2_FILE_INCLUDES_SHMEM_PROOF} -- so `current - file` would hand back the "
        "page store's own tmpfs as free) "
        f"-> non-reclaimable={_gib_or_none(t['cg_nonreclaim_gib'])} charged "
        "(#1233 fix 6: cache the kernel hands back is not memory the reaper kills for; "
        "charging it refused weg2dk5's own launch state) "
        f"unsampled_slab={_gib_or_none(unsampled_gib)} (train fix 3: the LIVE memory.stat "
        "slab_reclaimable, the one column the reap row lacked -- the store's reap bound "
        "subtracts it because the watermark over-states the reap point by exactly it; "
        "unreadable means NOT subtracted and the bound is optimistic by that term) "
        f"base_meminfo={t['base_meminfo_gib']:.2f} GiB base_cgroup={_gib_or_none(t['base_cgroup_gib'])} "
        f"-> base={t['base_gib']:.2f} GiB bound by {t['base_source']} "
        f"floor={FLOOR_GIB:.0f} GiB (#721; the #1232 host_headroom term is DELETED, "
        "it was a compensation constant for this very denominator) "
        f"heaps={t['heaps_gib']:.2f} GiB ({ranks_per_group}x{HEAP_AWAKE_GIB} awake b0 + "
        f"{ranks_per_group}x{HEAP_DORMANT_GIB} dormant campaign (a)) "
        f"UNPRICED RESIDUAL, named rather than folded in: `(file - shmem)` also credits file "
        f"pages in the UNEVICTABLE LRU as reclaimable (measured on this box, fix-6 review: "
        f"cgroup unevictable 16,384 B against /proc/meminfo Mlocked 309,682,176 B = 0.29 GiB) "
        f"-- sub-GiB today and in the OPTIMISTIC direction, and this design keeps adding "
        f"pinned host memory, so it is stated here rather than left to be discovered. "
        f"image_P={t['image_p_gib']:.2f} GiB [{t['image_p_source']}] "
        f"image_D={t['image_d_gib']:.2f} GiB [{t['image_d_source']}] "
        f"weight_tags_P={t['weight_tags_p_gib']:.2f} GiB weight_tags_D={t['weight_tags_d_gib']:.2f} GiB "
        f"(#809 census -- the tag byte sums, NOT the image) extra_P={t['image_extra_p_gib']:.2f} GiB "
        f"extra_D={t['image_extra_d_gib']:.2f} GiB (#1233 fix 8: everything with enable_cpu_backup "
        "is in the image -- draft weights, graph pools, workspaces, embeddings -- and boot weg2dk7 "
        "measured the dormant group at 38.63 GiB against the 28.83 GiB the tag sum priced) "
        f"run_origin={_gib_or_none(t['run_origin_gib'])} bound by {t['run_origin_source']} "
        f"(the image terms above SIZE H(c); they are NOT charged here -- the ring is) "
        + ("RING ABSENT BY DESIGN (Sigma H=0, reason=exchange): the weights ring "
           "is not allocated on this arm, so both moments charge 0 for it -- "
           "this is a DECLARED absence, not a missing measurement (#1327). " 
           if t.get("ring_absent_by_design") else "")
        + f"RUN MOMENT = the host weights term {t['host_ring_gib']:.2f} GiB "
        f"(C19; BACKUP_P_BYTES / BACKUP_D_BYTES / chunk_gib are DELETED, not shrunk) "
        f"LAUNCH MOMENT = ring span 1, Sigma image_P = {t['host_ring_span1_gib']:.2f} GiB (R7) + "
        f"load_transient={LOAD_TRANSIENT_GIB:.0f} GiB (MEASURED residual of boot weg2ls1b2, #721 constant was 27; D loading while P is dormant) "
        f"ring provenance: {ring_provenance or 'NOT NAMED -- caller passed none'} "
        f"anchors@2400={ANCHORS_AT_2400_BYTES / GIB:.2f} GiB (b0 measured, scaled by M) "
        f"rings=({RING_P_MULT_GB_PER_S:.0f}+{RING_D_MULT_GB_PER_S:.0f})xS GB (b0) "
        f"overhead={HOST_POOL_OVERHEAD:.0%} of host-pool posts (b0 U14) "
        f"draft_host_P={DRAFT_HOST_P_MIB:.1f} MiB draft_host_D={DRAFT_HOST_D_MIB:.1f} MiB "
        f"(#1233 pinned draft host pools, both moments) "
        f"store_draft_fraction={STORE_DRAFT_FRACTION:.4f} (2048 of 32768 B per token)"
    )
    # FIX 6: origin and watermark in ONE currency -- both non-reclaimable.
    watermark_gib = OBSERVED_REAP_NONRECLAIM_BYTES / GIB
    # #1269 / standing order 2026-09-08: the watermark alone is not the bound.
    # The bound is the watermark MINUS a named margin -- the flip transient the
    # box will actually spend and the idle anon drift it will actually
    # accumulate over the planned window. Both WILL happen during the window,
    # so an arm that has not left room for them is already lost.
    margin = margin if margin is not None else resolve_margin()
    hard_bound_gib = watermark_gib - margin.total_gib
    lines.append(watermark_provenance(margin, watermark_gib))
    chosen: Optional[Arm] = None
    peak_bound_any = False
    # #1236: the STORE TERM IS GONE FROM THIS LOOP.  Train fix 3 sized it here
    # as min(run leftover, reap bound) because it was a tmpfs and therefore
    # un-evictable shmem inside this cgroup -- the one free term of the run
    # peak.  It is a directory on the ZFS dataset now, so there is no RAM
    # residual to hand out and no store floor to bind on: the arm table prices
    # RAM, the store is priced against the DISK by launcher.plan_store, and the
    # two no longer share a currency.  What the store USED to cost this sum is
    # printed on the CHOSEN line as the freed delta, so the change is visible
    # in the ledger's own output and not only in this comment.
    for arm in priced:
        moments_ok = arm.fundable_moments
        predicted = arm.predicted_run_peak_gib()
        # FIX 8: the run peak REFUSES.  Two boots died and one could not flip
        # while this quantity was printed as an advisory beside the arm it had
        # already condemned.
        # #1269: against the HARD BOUND (watermark - margin), not the raw mark.
        peak_ok = predicted is None or predicted <= hard_bound_gib
        binding: List[str] = []
        if arm.launch_leftover_gib < 0:
            binding.append(f"launch moment ({arm.launch_leftover_gib:.2f} GiB)")
        if arm.run_leftover_gib < 0:
            binding.append(f"run moment ({arm.run_leftover_gib:.2f} GiB)")
        if not peak_ok:
            binding.append(
                f"RUN PEAK ({predicted:.2f} > {hard_bound_gib:.2f} GiB hard bound "
                f"= {watermark_gib:.2f} reap point - {margin.total_gib:.2f} margin "
                f"[{margin.terms()}])"
            )
        ok = moments_ok and peak_ok
        if moments_ok and not peak_ok:
            peak_bound_any = True
        lines.append(
            # #1317n S_D IS APPENDED AFTER M, NOT INSERTED BETWEEN S AND M.
            # Three suites match the literal substring `ARM S=<s> M=<m>` (two
            # by `in`, one by a list comprehension that then indexes [0] and
            # raised IndexError), so splitting that pair is a text change
            # masquerading as a semantic one. The new term is additive and the
            # old reading is byte-identical -- which is also the honest shape:
            # nothing about S or M changed, D simply gained its own budget.
            f"WEG2-HOST-LEDGER ARM S={arm.s_gb} M={arm.m_mib} S_D={_arm_s_d(arm)}: "
            f"anchors={arm.terms['anchors_gib']:.2f} rings={arm.terms['rings_gib']:.2f} "
            f"overhead={arm.terms['overhead_gib']:.2f} "
            # #1273 S6: THE EXCHANGE'S PINNED HOST CARRIER, NAMED ON THE ARM
            # LINE.  0.00 on the ring arm -- an unarmed boot says the term was
            # priced at zero instead of leaving a reader to ask whether it was
            # priced at all -- and `slots x slot_bytes x cards` of /dev/shm on
            # the arms that create the file, charged at both moments.
            f"xchg_bounce={arm.terms['xchg_bounce_gib']:.2f} -> "
            f"leftover launch={arm.launch_leftover_gib:.2f} GiB "
            # #1317n BOTH FORMS ON THE LINE, because the class assignment is
            # the whole change and a reader must be able to see it rather than
            # trust it: `launch` is priced in the currency `launch_cur` names,
            # `launch_sum` is the worst case with the entire load transient
            # charged as non-reclaimable, and the gap between them IS the page
            # cache (boot weg2sn6r: -2.78 sum against +9.22 by class).
            f"launch_sum={arm.launch_worst_case_gib:.2f} GiB "
            f"launch_cur={(arm.terms or {}).get('launch_currency', '?')} "
            f"run={arm.run_leftover_gib:.2f} GiB "
            f"store=NOT CHARGED HERE (#1236: on disk under launcher.STORE_ROOT, "
            f"page_cache=reclaimable, sized from the P KV pool -- see WEG2-STORE) "
            f"unsampled_slab={_gib_or_none(unsampled_gib)} "
            "run_peak="
            + ("unreadable (no cgroup sample)" if predicted is None else f"{predicted:.2f} GiB")
            + f" vs hard bound {hard_bound_gib:.2f} GiB (reap {watermark_gib:.2f} - "
            f"margin {margin.total_gib:.2f}) => "
            + ("FUNDABLE" if ok else "refused (binding: " + ", ".join(binding) + ")")
        )
        if ok and chosen is None:
            chosen = arm
    if chosen is None:
        # The advisory's explanatory half belongs in the refusal too -- see
        # :func:`_advisory_line`.  The most frugal arm is the ladder's last.
        frugal = priced[-1]
        lines.append(_advisory_line(frugal, chosen=False))
        table = "\n".join(lines)
        # THE HONEST OUTCOME (fix 8): on today's box every arm may refuse, and
        # that IS the answer for this tree.  No term is shrunk to get an arm
        # through -- the term that has to move is the flip transient, and it
        # moves in the ring slice, not here.
        outcome = (
            "no arm funds a flip on this host budget.  The flip transient this sentence "
            "used to name as the thing that had to move is ALREADY gone: the host ring "
            "landed (C19), so the run moment charges Sigma H once and there is no "
            "transient left to cut"
        )
        levers = (
            "The levers are NAMED so this refusal is actionable, and #1236 changed which "
            "ones exist: THE STORE IS NOT A LEVER ANY MORE IN EITHER DIRECTION. It is a "
            "directory on the ZFS dataset, it is not charged to this sum, and "
            "--store-min-gib is deleted -- a floor on a RAM residual is meaningless once "
            "the store is not funded from RAM. The one lever left is the peak itself: cut "
            "Sigma H -- the per-card host ring table, solved from the previous boot's own "
            "measured dormant image (ring_table.solve), so a smaller image is a smaller "
            "ring -- or take a lower arm of the ladder. Note the direction #1236 moves "
            "this refusal: every arm's predicted peak is now LOWER by exactly the store "
            "term it used to carry (6 GiB on boot weg2sb5g), so an arm that refused for "
            "the store's sake alone no longer refuses at all."
        )
        if peak_bound_any:
            raise Weg2HostRunPeakRefused(
                "W21 Weg2HostRunPeakRefused: an arm funds both moments and the HARD BOUND "
                f"{hard_bound_gib:.2f} GiB (reap point {watermark_gib:.2f} GiB minus margin "
                f"{margin.total_gib:.2f} GiB = {margin.terms()}) still binds "
                "it -- its predicted RUN PEAK is not below the hard bound, and #1236 "
                "already removed the store term from that peak, so no store size can "
                f"repair it and none is asked to. {outcome}. The binding "
                "term is printed per arm below; the image term is now "
                "MEASURED (boot weg2dk7: 38.63 GiB dormant against the 28.83 GiB the tag "
                "census priced) and the origin is the RUN moment, not the launch moment. "
                f"{levers}\n" + table
            )
        cap_advice = ""
        if s_gb_d is not None and d_cap_terms:
            def _ok(sd: int) -> bool:
                try:
                    a = price(
                        memtotal_bytes, memavail_bytes, arms[0][0], arms[0][1],
                        ranks_per_group=ranks_per_group, ring_bytes=ring_bytes,
                        ring_span1_bytes=ring_span1_bytes,
                        cg_current_bytes=cg_current_bytes,
                        reclaimable_bytes=reclaimable_bytes,
                        cg_ceiling_bytes=cg_ceiling_bytes, s_gb_d=sd,
                        ring_absent_by_design=ring_absent_by_design,
                        measured_record=measured_record,
                    )
                except Exception:  # noqa: BLE001 - advice may never mask the refusal
                    return False
                return bool(a.fundable_moments)

            fit = largest_fundable_d_cap(
                _ok,
                int(d_cap_terms.get("cap_tokens", 0)),
                float(d_cap_terms.get("max_rank_share", 0.0)),
                int(d_cap_terms.get("cell_bytes", 1)),
                float(d_cap_terms.get("fraction", 1.0)),
            )
            if fit is None:
                cap_advice = (
                    " LARGEST FUNDABLE CAP: none -- not even S_D=1 funds the "
                    "moments, so group D's L2 budget is NOT the binding term "
                    "here and lowering --max-kv-per-request will not help. "
                    "Cut the term the binding moment names."
                )
            else:
                cap_advice = (
                    f" LARGEST FUNDABLE CAP: **{int(fit['cap_tokens'])} tokens** "
                    f"at S_D={int(fit['s_gb_d'])} GB "
                    f"({int(fit['rows'])} rows, {int(fit['rows_lendable'])} "
                    f"lendable behind the load fraction), against the "
                    f"{int(fit['asked_cap'])} asked for. So either lower "
                    f"--max-kv-per-request to that, or free the term the "
                    f"binding moment names below -- but do NOT lower "
                    f"--hicache-size for D on its own: that reintroduces the "
                    f"shortfall the #1317 window layer existed to compensate, "
                    f"which is what this ticket deleted."
                )
        raise Weg2HostLedgerRefused(
            "W20 Weg2HostLedgerRefused: no arm of the ladder funds both moments on this "
            "box (the store is NOT in this test any more -- #1236 put it on disk) "
            f"(host weights term = {priced[0].terms['host_ring_gib']:.2f} GiB at the run "
            f"moment, span 1 = {priced[0].terms['host_ring_span1_gib']:.2f} GiB at the launch "
            f"moment; {ring_provenance or 'no provenance passed'}). "
            "The INT8 checkpoint has only the cpu-backup wake path (W4), and the ledger "
            f"will not shrink another term silently. {outcome}. {levers}"
            f"{cap_advice}\n"
            + table
        )
    chosen_peak = chosen.predicted_run_peak_gib()
    headroom = None if chosen_peak is None else hard_bound_gib - chosen_peak
    lines.append(
        f"WEG2-HOST-LEDGER CHOSEN S={chosen.s_gb} GB (--hicache-size, both groups) "
        f"M={chosen.m_mib} MiB (--hicache-mamba-host-mib, both groups) "
        f"store=ON DISK, NOT A RAM POST (#1236: launcher.STORE_ROOT on the ZFS dataset; "
        f"its host cost is page_cache=reclaimable plus the ZFS ARC, neither of which the "
        f"reaper kills for, and neither of which is in this arm's sum) "
        f"launch_leftover={chosen.launch_leftover_gib:.2f} GiB "
        f"run_leftover={chosen.run_leftover_gib:.2f} GiB "
        f"reap_headroom={_gib_or_none(headroom)} (hard bound {hard_bound_gib:.2f} - "
        f"predicted run peak {_gib_or_none(chosen_peak)}) -- provenance: every term "
        "above; expectation from the operator (record 1g) was ~20 GiB without the "
        f"heap term ({chosen.terms['heaps_gib']:.2f} GiB measured). THE #1236 DELTA: this "
        "headroom is larger than the pre-#1236 form's by exactly the store term that "
        "form charged -- 6 GiB on boot weg2sb5g, whose store was RAM-backed -- and "
        "the launcher prints the delta against THIS boot's own store budget on the "
        "WEG2-STORE line rather than repeating a recalled number here."
    )
    lines.append(_advisory_line(chosen, chosen=True))
    return chosen, headroom, lines


class WindowUnsolvable(RuntimeError):
    """W21: no window fits this host pool, and the arithmetic says why.

    A launch refusal, never a runtime fallback: a group that cannot hold one
    window plus the write-through floor cannot read the store back in windows
    at all, and discovering that at the first long prompt means the boot is
    already serving.
    """


def solve_window(
    pool_rows: int,
    chunk: int,
    chains: int,
    ring_floor: int,
    anchor_slots: int,
) -> int:
    """#1317 C10: the largest window, in tokens, one store read may hold.

    THE QUANTITY.  A windowed store read holds, at any moment, ONE window of
    host rows per concurrent chain plus whatever the write-through ring has
    staged.  So

        W = chunk * floor((pool_rows - ring_floor) / (chains * chunk))

    ``chunk`` is the STATIC ``--chunked-prefill-size`` and never
    ``dynamic_chunked_prefill_size``: a C that varies per rank at runtime is
    the rank-divergent form the group vote exists to delete, and ``C | W`` is
    load-bearing for the GDN anchors (one anchor per chunk node).

    ``pool_rows`` is the group's OWN MIN-synced ``mem_pool_host.size``, read
    in-process.  It is deliberately NOT transported from the launcher: group P
    holds 54,254 rows and group D 30,518, and one launcher-solved W would
    truncate every P prefetch to a D-sized window.

    ``chains`` is ``--max-running-requests``: the pool is shared, and sizing W
    as if one request existed is how a second concurrent read finds no room
    and votes the first one down.

    Raises :class:`WindowUnsolvable` (W21) with the arithmetic when no window
    fits, or when the windows would need more resume anchors than the anchor
    pool holds -- an anchor-starved chain stalls at window 2, which is a
    launch-time fact and must be a launch-time refusal.

    Worked, group D on boot ``weg2sn5w``: pool_rows=30,518, chunk=4,096,
    chains=1, ring_floor=4,096, anchor_slots=37 ->
    W = 4,096 * floor(26,422/4,096) = 4,096 * 6 = 24,576; live W + ring_floor
    = 28,672 <= 30,518 (slack 1,846); anchors W//C + 1 = 7 of 37.
    """
    pool_rows = int(pool_rows)
    chunk = int(chunk)
    chains = max(1, int(chains))
    ring_floor = int(ring_floor)
    anchor_slots = int(anchor_slots)
    if chunk <= 0:
        raise WindowUnsolvable(
            f"W21 window unsolvable: chunk (--chunked-prefill-size) is {chunk}, "
            "which is not a length; a window is a whole number of chunks "
            "because the GDN anchors are published one per chunk node."
        )
    room = pool_rows - ring_floor
    windows = room // (chains * chunk) if room > 0 else 0
    w = chunk * windows
    if w < chunk:
        raise WindowUnsolvable(
            f"W21 window unsolvable: host pool {pool_rows} rows minus the "
            f"write-through floor {ring_floor} leaves {room} rows, which is "
            f"less than one chunk of {chunk} for each of {chains} concurrent "
            f"chains (needs {chains * chunk}). This group cannot read the "
            f"store back in windows at all. Raise --hicache-size, lower "
            f"--max-running-requests, or lower --chunked-prefill-size."
        )
    anchors_live = w // chunk + 1
    if anchors_live > anchor_slots:
        raise WindowUnsolvable(
            f"W21 window unsolvable: a window of {w} tokens needs "
            f"{anchors_live} live resume anchors (W//C + 1: one per chunk node "
            f"plus the head's own anchor, which must survive its window's "
            f"recycle) but the anchor pool holds {anchor_slots}. The chain "
            f"would stall at window 2 with an unmatchable node. Raise the "
            f"anchor pool (--mamba-host-pool-mib) or lower "
            f"--chunked-prefill-size."
        )
    return w


def window_provenance(
    pool_rows: int,
    chunk: int,
    chains: int,
    ring_floor: int,
    anchor_slots: int,
    prompt_tokens: int = 0,
) -> str:
    """One line naming every term that produced W, for the launcher to print.

    Prints the terms, not just the number: two groups printing the same W from
    different pools would be a coincidence, and the reader of a boot log has
    no other way to tell which pool answered.
    """
    w = solve_window(pool_rows, chunk, chains, ring_floor, anchor_slots)
    n = -(-int(prompt_tokens) // w) if prompt_tokens > 0 else 0
    return (
        f"#1317 WINDOW W={w} chunk={chunk} chains={chains} "
        f"pool_rows={pool_rows} ring_floor={ring_floor} "
        f"live={w + ring_floor}<={pool_rows} slack={pool_rows - w - ring_floor} "
        f"anchors={w // chunk + 1}/{anchor_slots} windows_for_{prompt_tokens}={n}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meminfo", default="/proc/meminfo")
    ap.add_argument("--ring-bytes", type=int, default=0, help="Sigma H, from ring_table.solve")
    ap.add_argument("--ring-span1-bytes", type=int, default=0, help="Sigma image_P")
    ap.add_argument("--ring-provenance", default="")
    ap.add_argument("--cgroup", default="/sys/fs/cgroup")
    ap.add_argument("--weight-chunks", type=int, default=0)
    ap.add_argument("--measured-record", default="")
    ns = ap.parse_args(argv)
    mi = read_meminfo(ns.meminfo)
    cg = read_cgroup(ns.cgroup)
    ceiling, ceiling_source = resolve_cg_ceiling(cg, mi["MemTotal"])
    try:
        arm, headroom, lines = choose(
            mi["MemTotal"],
            mi["MemAvailable"],
            ring_bytes=ns.ring_bytes,
            ring_span1_bytes=ns.ring_span1_bytes,
            ring_provenance=ns.ring_provenance,
            cg_current_bytes=cg["current"],
            reclaimable_bytes=cg["reclaimable"],
            slab_reclaimable_bytes=cg["slab_reclaimable"],
            cg_ceiling_bytes=ceiling,
            cg_ceiling_source=ceiling_source,
            cg_oom_kill=cg["oom_kill"],
            measured_record=read_measured_record(ns.measured_record) if ns.measured_record else None,
        )
    except (Weg2HostLedgerRefused, Weg2HostRunPeakRefused) as e:
        print(str(e))
        return 2
    print("\n".join(lines))
    print(f"WEG2_S_GB={arm.s_gb}")
    print(f"WEG2_M_MIB={arm.m_mib}")
    # #1236: WEG2_STORE_GIB is GONE from this output rather than printed as 0.
    # A key whose value became meaningless is deleted; a 0 would read as "the
    # ledger sized a zero store", which is a different and false statement.
    print("WEG2_REAP_HEADROOM_GIB=" + ("unreadable" if headroom is None else f"{headroom:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
