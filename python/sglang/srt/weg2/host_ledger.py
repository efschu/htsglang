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

The arm ladder (record section 1c B2: "if the ledger refuses at S=2 the
launcher sizes S=1 and prints why; if it refuses at S=1 the boot REFUSES") is
extended by the mamba host pool ``M`` in the same spirit: every arm is printed,
the first fundable one is taken, and a shrink is never silent.  If no arm funds
the store floor the launch is refused by name (W20 ``Weg2HostLedgerRefused``).

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

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
#: content at the death): a standing cost this ledger already budgets as
#: ``store_gib``, not a leak.
FLIP_HOST_TRANSIENT_GIB = 9.97
#: #1233 boot weg2dk5: the OBSERVED REAP POINT of this box's cgroup -- the
#: memts row at 21:15:30Z, ``cg_current_b=102,998,904,832`` with ``oom_kill``
#: 18 -> 24 in the same row (and /proc/vmstat 60 -> 66 independently, same
#: delta +6).  ADVISORY ONLY, never a refusal: one boot's death is a watermark,
#: not a limit, and this cgroup publishes no finite ``memory.max`` to check
#: against.  It is printed beside each arm's PREDICTED run-peak ``cg_current``
#: so a configuration that is about to repeat weg2dk5 says so BEFORE the boot.
OBSERVED_REAP_CURRENT_BYTES = 102_998_904_832
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
#: b0: "8xS ring = 16.00 GB at S=2" = PP pool 2.00/1.00/1.00 GB (2xS) plus
#: TP pool 4.00 GB x3 (6xS).  Per-process pools (record 1e): group P owns the
#: 2xS half, group D the 6xS half.
RING_P_MULT_GB_PER_S = 2.0
RING_D_MULT_GB_PER_S = 6.0
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
    def fundable_moments(self) -> bool:
        return self.launch_leftover_gib >= 0.0 and self.run_leftover_gib >= 0.0

    def predicted_run_peak_gib(self, store_gib: float) -> Optional[float]:
        """What ``memory.current`` this arm reaches at the RUN PEAK, or None.

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
        """
        cg = self.terms.get("cg_current_gib")
        if cg is None:
            return None
        t = self.terms
        return (
            float(cg)
            + t["heaps_gib"] + t["anchors_gib"] + t["rings_gib"] + t["overhead_gib"]
            + t["draft_host_p_gib"] + t["draft_host_d_gib"]
            + t["host_ring_gib"]
            + float(store_gib)
        )


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


def read_cgroup(root: str = "/sys/fs/cgroup") -> Dict[str, Optional[int]]:
    """The cgroup2 memory facts the REAPER acts on, in BYTES.

    ``current`` / ``peak`` from ``memory.current`` / ``memory.peak``,
    ``oom_kill`` from ``memory.events``, and ``max`` from ``memory.max`` --
    ``None`` when that file says ``max``, i.e. when this cgroup publishes NO
    finite ceiling.  Inside this LXC container it does say ``max``, and that
    absence is a fact the caller must NAME (:func:`price` falls back to
    MemTotal and says so) rather than paper over with a constant.

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
    return {
        "current": _int("memory.current"),
        "peak": _int("memory.peak"),
        "max": _int("memory.max"),
        "oom_kill": oom,
    }


def price(
    memtotal_bytes: int,
    memavail_bytes: int,
    s_gb: int,
    m_mib: int,
    *,
    ranks_per_group: int = 3,
    ring_bytes: int = 0,
    ring_span1_bytes: int = 0,
    cg_current_bytes: Optional[int] = None,
    cg_ceiling_bytes: Optional[int] = None,
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
    """
    if s_gb < 1 or m_mib < 1:
        raise ValueError(f"arm terms must be >= 1: S={s_gb} M={m_mib}")
    if ring_bytes <= 0 or ring_span1_bytes <= 0:
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
    if cg_current_bytes is not None and cg_ceiling_bytes is not None:
        # The CLI reserve is charged here too and for the same reason as on the
        # meminfo arm: ``memory.current`` nets out what the CLIs hold RIGHT NOW,
        # this term keeps the room they grow into.
        base_cgroup_gib = (
            (cg_ceiling_bytes - cg_current_bytes) / GIB - CLI_RESERVE_GIB
        )
    if base_cgroup_gib is None:
        base_gib = base_meminfo_gib
        base_source = "meminfo (no cgroup sample passed)"
    elif base_cgroup_gib <= base_meminfo_gib:
        base_gib = base_cgroup_gib
        base_source = "cgroup (ceiling - memory.current - cli_reserve)"
    else:
        base_gib = base_meminfo_gib
        base_source = "meminfo (min(memavail, memtotal-cli))"
    heaps_gib = ranks_per_group * (HEAP_AWAKE_GIB + HEAP_DORMANT_GIB)
    anchors_gib = (ANCHORS_AT_2400_BYTES * (m_mib / ANCHORS_REFERENCE_M_MIB)) / GIB
    rings_gib = (RING_P_MULT_GB_PER_S + RING_D_MULT_GB_PER_S) * s_gb * GB / GIB
    overhead_gib = HOST_POOL_OVERHEAD * (anchors_gib + rings_gib)
    host_ring_gib = ring_bytes / GIB
    host_ring_span1_gib = ring_span1_bytes / GIB
    # #1233: the draft tier is a HOST tier of its own, one budget per group, and
    # it is charged at BOTH moments -- it is allocated at load and outlives every
    # flip (that is the point of carrying draft KV across the flip).  It is NOT
    # part of the flip image: the ring carries the weights, this carries the
    # draft pages, and the two are never the same bytes.
    draft_host_p_gib = DRAFT_HOST_P_MIB / 1024.0
    draft_host_d_gib = DRAFT_HOST_D_MIB / 1024.0
    common = (
        base_gib - FLOOR_GIB - heaps_gib - anchors_gib - rings_gib
        - overhead_gib - draft_host_p_gib - draft_host_d_gib
    )
    # R7: at the launch moment only span 1 is registered (P's first pause is the
    # launcher's sleep(P)); span 2 lands at D's first pause, when D's load
    # transient is gone.  Charging Sigma H at launch is what turns the M=1200
    # arm's leftover from +1.05 into -1.93 GiB.
    launch = common - host_ring_span1_gib - LOAD_TRANSIENT_GIB
    # Sigma H IS the run peak; there is no flip transient beside it (see price's
    # docstring and FLIPCOST A1-1/A1-3).
    run = common - host_ring_gib
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
        "cg_ceiling_gib": (
            None if cg_ceiling_bytes is None else cg_ceiling_bytes / GIB
        ),
        "floor_gib": FLOOR_GIB,
        "heaps_gib": heaps_gib,
        "host_ring_gib": host_ring_gib,
        "host_ring_span1_gib": host_ring_span1_gib,
        "load_transient_gib": LOAD_TRANSIENT_GIB,
        "anchors_gib": anchors_gib,
        "rings_gib": rings_gib,
        "overhead_gib": overhead_gib,
        "draft_host_p_gib": draft_host_p_gib,
        "draft_host_d_gib": draft_host_d_gib,
        "store_draft_fraction": STORE_DRAFT_FRACTION,
    }
    arm.launch_leftover_gib = launch
    arm.run_leftover_gib = run
    return arm


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


def choose(
    memtotal_bytes: int,
    memavail_bytes: int,
    *,
    store_min_gib: float,
    arms: Sequence[Tuple[int, int]] = DEFAULT_ARMS,
    ranks_per_group: int = 3,
    ring_bytes: int = 0,
    ring_span1_bytes: int = 0,
    ring_provenance: str = "",
    cg_current_bytes: Optional[int] = None,
    cg_ceiling_bytes: Optional[int] = None,
    cg_ceiling_source: str = "",
    cg_oom_kill: Optional[int] = None,
) -> Tuple[Arm, float, List[str]]:
    """Walk the ladder; return (arm, store_gib, printed lines) or raise W20.

    ``store_gib`` is the RUN leftover floored to whole GiB: the tmpfs the
    canonical page store lives on.  A leftover below ``store_min_gib`` on
    every arm is a refusal -- a carrier that cannot hold one agent prefix is
    not a carrier, and the boot would pass R1 and fail R2/R4 for a reason the
    ledger already knew.

    The cgroup arguments are fix 5's denominator (see :func:`price`).
    ``cg_oom_kill`` is printed as the PRE-BOOT BASELINE of a counter that is
    cumulative and carries no timestamps: a death can only ever be attributed
    by baseline-vs-after diff, and taking that baseline here means no boot can
    start without one.
    """
    lines: List[str] = []
    priced = [
        price(
            memtotal_bytes,
            memavail_bytes,
            s,
            m,
            ranks_per_group=ranks_per_group,
            ring_bytes=ring_bytes,
            ring_span1_bytes=ring_span1_bytes,
            cg_current_bytes=cg_current_bytes,
            cg_ceiling_bytes=cg_ceiling_bytes,
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
        f"base_meminfo={t['base_meminfo_gib']:.2f} GiB base_cgroup={_gib_or_none(t['base_cgroup_gib'])} "
        f"-> base={t['base_gib']:.2f} GiB bound by {t['base_source']} "
        f"floor={FLOOR_GIB:.0f} GiB (#721; the #1232 host_headroom term is DELETED, "
        "it was a compensation constant for this very denominator) "
        f"heaps={t['heaps_gib']:.2f} GiB ({ranks_per_group}x{HEAP_AWAKE_GIB} awake b0 + "
        f"{ranks_per_group}x{HEAP_DORMANT_GIB} dormant campaign (a)) "
        f"RUN MOMENT = the host weights term {t['host_ring_gib']:.2f} GiB "
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
    chosen: Optional[Arm] = None
    store_gib = 0.0
    for arm in priced:
        run_store = math.floor(arm.run_leftover_gib) if arm.run_leftover_gib > 0 else 0.0
        ok = arm.fundable_moments and run_store >= store_min_gib
        lines.append(
            f"WEG2-HOST-LEDGER ARM S={arm.s_gb} M={arm.m_mib}: "
            f"anchors={arm.terms['anchors_gib']:.2f} rings={arm.terms['rings_gib']:.2f} "
            f"overhead={arm.terms['overhead_gib']:.2f} -> "
            f"leftover launch={arm.launch_leftover_gib:.2f} GiB "
            f"run={arm.run_leftover_gib:.2f} GiB store={run_store:.0f} GiB "
            f"(floor {store_min_gib:.0f}) => {'FUNDABLE' if ok else 'refused'}"
        )
        if ok and chosen is None:
            chosen = arm
            store_gib = float(run_store)
    if chosen is None:
        table = "\n".join(lines)
        raise Weg2HostLedgerRefused(
            "W20 Weg2HostLedgerRefused: no arm of the ladder funds both moments "
            f"plus a {store_min_gib:.0f} GiB store floor on this box "
            f"(host weights term = {priced[0].terms['host_ring_gib']:.2f} GiB at the run "
            f"moment, span 1 = {priced[0].terms['host_ring_span1_gib']:.2f} GiB at the launch "
            f"moment; {ring_provenance or 'no provenance passed'}). "
            "The INT8 checkpoint has only the cpu-backup wake path (W4), and "
            "the ledger will not shrink another term silently. The two levers are "
            "NAMED so this refusal is actionable: lower --store-min-gib to the run "
            "leftover the last arm above actually prints, or cut Sigma H itself "
            "(the per-card host ring table, ring_table.solve) -- shrinking the store "
            "tmpfs is NOT a lever, its Shmem was flat across the fatal window.\n"
            + table
        )
    lines.append(
        f"WEG2-HOST-LEDGER CHOSEN S={chosen.s_gb} GB (--hicache-size, both groups) "
        f"M={chosen.m_mib} MiB (--hicache-mamba-host-mib, both groups) "
        f"store={store_gib:.0f} GiB tmpfs (the run leftover, floored) "
        f"launch_leftover={chosen.launch_leftover_gib:.2f} GiB "
        f"run_leftover={chosen.run_leftover_gib:.2f} GiB -- provenance: every term "
        "above; expectation from the operator (record 1g) was ~20 GiB without the "
        f"heap term ({chosen.terms['heaps_gib']:.2f} GiB measured)"
    )
    predicted = chosen.predicted_run_peak_gib(store_gib)
    watermark_gib = OBSERVED_REAP_CURRENT_BYTES / GIB
    if predicted is None:
        lines.append(
            "WEG2-HOST-LEDGER RUN-PEAK ADVISORY: not computed -- no cgroup sample was "
            "passed, so there is no origin to add this arm's charges to. An absent "
            "prediction is stated, never printed as a pass."
        )
    else:
        verdict = "ABOVE" if predicted > watermark_gib else "below"
        lines.append(
            "WEG2-HOST-LEDGER RUN-PEAK ADVISORY: this arm predicts memory.current="
            f"{predicted:.2f} GiB at the run peak (cg_current now "
            f"{_gib_or_none(chosen.terms['cg_current_gib'])} + heaps + anchors + rings + "
            f"overhead + draft pools + dormant image + flip_transient + store {store_gib:.0f} "
            f"GiB; the {FLOOR_GIB:.0f} GiB floor is a reserve and is NOT in this sum), "
            f"which is {verdict} the OBSERVED REAP POINT {watermark_gib:.2f} GiB "
            "(boot weg2dk5 21:15:30Z, memory.current 102,998,904,832 B with oom_kill "
            "18 -> 24 in the same row). ADVISORY, not a refusal: one boot's death is a "
            "watermark and this cgroup publishes no finite memory.max to check against. "
            "weg2dk5's own arm (S=1 M=1200 store=9) prices at 99.05 GiB here, i.e. this "
            "line would have named that boot ABOVE the watermark before it started."
        )
    return chosen, store_gib, lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-min-gib", type=float, default=4.0)
    ap.add_argument("--meminfo", default="/proc/meminfo")
    ap.add_argument("--ring-bytes", type=int, default=0, help="Sigma H, from ring_table.solve")
    ap.add_argument("--ring-span1-bytes", type=int, default=0, help="Sigma image_P")
    ap.add_argument("--ring-provenance", default="")
    ap.add_argument("--cgroup", default="/sys/fs/cgroup")
    ns = ap.parse_args(argv)
    mi = read_meminfo(ns.meminfo)
    cg = read_cgroup(ns.cgroup)
    ceiling, ceiling_source = resolve_cg_ceiling(cg, mi["MemTotal"])
    try:
        arm, store, lines = choose(
            mi["MemTotal"],
            mi["MemAvailable"],
            store_min_gib=ns.store_min_gib,
            ring_bytes=ns.ring_bytes,
            ring_span1_bytes=ns.ring_span1_bytes,
            ring_provenance=ns.ring_provenance,
            cg_current_bytes=cg["current"],
            cg_ceiling_bytes=ceiling,
            cg_ceiling_source=ceiling_source,
            cg_oom_kill=cg["oom_kill"],
        )
    except Weg2HostLedgerRefused as e:
        print(str(e))
        return 2
    print("\n".join(lines))
    print(f"WEG2_S_GB={arm.s_gb}")
    print(f"WEG2_M_MIB={arm.m_mib}")
    print(f"WEG2_STORE_GIB={store:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
