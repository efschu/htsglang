"""The #721 host ledger for the Weg-2 six-process shape, priced at BOTH moments.

Every term below is NAMED with its provenance, printed by :func:`format_lines`,
and none of them is a spec number copied blind: they are the measured posts of
the campaigns of 2026-09-06 (WEG2_BUILD_DECISIONS_0906.md section 1d/1e and
CAMPAIGN_b0_0906.md / CAMPAIGN_a_0906.md) or the standing #721 constants the
line already runs under (``planner/weg1_host_sizing.py``).

Two moments (record section 1c B2, spec section 4.2.3):

* ``launch`` -- group D is loading (LOAD_TRANSIENT charged) while group P is
  already dormant with its cpu-backup image resident.
* ``run`` -- both groups exist, BOTH cpu backups are resident (DR-1: the
  torch_memory_saver cpu backup is shm-backed and never returns), the awake
  group runs at its serving heap.

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
#: Operator list, record section 1g: "the Claude CLIs (~10 GiB)".  Charged
#: ONCE against MemTotal (they are live RSS, so MemAvailable already nets out
#: whatever they hold right now; the term keeps their room when they grow).
CLI_RESERVE_GIB = 10.0
#: b0: awake steady RssAnon peaks 3.385 / 3.038 / 2.996 GiB (L-bf2), max taken.
HEAP_AWAKE_GIB = 3.385
#: Campaign (a): RssAnon "flat at 2.36 GiB" on the dormant tp=1 rank.
HEAP_DORMANT_GIB = 2.36
#: Spec section 2.6, #809 FLIP IMAGE PREFETCH census: PP image 30.96 GB,
#: TP image 29.15 GB per group.  Campaign (a) / S1 boot: the cpu backup is
#: shm-backed, 1x the image, persistent (DR-1) -- charged 1x per group.
BACKUP_P_BYTES = 30.96 * GB
BACKUP_D_BYTES = 29.15 * GB
#: #721 LOAD_TRANSIENT (weg1_host_sizing.LOAD_TRANSIENT_BYTES): the loader's
#: page-cache + staging transient while a group loads.  Launch moment only.
LOAD_TRANSIENT_GIB = 27.0
#: b0: 10.19 GB MEASURED for six (rank x phase) anchor pools at m_mib=2400
#: (2.55/1.49/1.06 + 2.55/1.27/1.27 GB).  Scaled linearly with M.
ANCHORS_AT_2400_BYTES = 10.19 * GB
ANCHORS_REFERENCE_M_MIB = 2400
#: b0: "8xS ring = 16.00 GB at S=2" = PP pool 2.00/1.00/1.00 GB (2xS) plus
#: TP pool 4.00 GB x3 (6xS).  Per-process pools (record 1e): group P owns the
#: 2xS half, group D the 6xS half.
RING_P_MULT_GB_PER_S = 2.0
RING_D_MULT_GB_PER_S = 6.0
#: b0 U14: the unattributed residual is 1.006 GiB = ~4 % on top of the
#: host-pool posts (rings + anchors).  Charged as +4 % on those posts.
HOST_POOL_OVERHEAD = 0.04

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


def read_meminfo(path: str = "/proc/meminfo") -> Dict[str, int]:
    """MemTotal/MemAvailable/Shmem in BYTES from /proc/meminfo."""
    with open(path) as f:
        text = f.read()
    out: Dict[str, int] = {}
    for key, val in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M):
        out[key] = int(val) * 1024
    return out


def price(
    memtotal_bytes: int,
    memavail_bytes: int,
    s_gb: int,
    m_mib: int,
    *,
    ranks_per_group: int = 3,
) -> Arm:
    """Price one arm at both moments.  Pure."""
    if s_gb < 1 or m_mib < 1:
        raise ValueError(f"arm terms must be >= 1: S={s_gb} M={m_mib}")
    base_gib = min(memavail_bytes / GIB, memtotal_bytes / GIB - CLI_RESERVE_GIB)
    heaps_gib = ranks_per_group * (HEAP_AWAKE_GIB + HEAP_DORMANT_GIB)
    anchors_gib = (ANCHORS_AT_2400_BYTES * (m_mib / ANCHORS_REFERENCE_M_MIB)) / GIB
    rings_gib = (RING_P_MULT_GB_PER_S + RING_D_MULT_GB_PER_S) * s_gb * GB / GIB
    overhead_gib = HOST_POOL_OVERHEAD * (anchors_gib + rings_gib)
    backup_p_gib = BACKUP_P_BYTES / GIB
    backup_d_gib = BACKUP_D_BYTES / GIB
    common = base_gib - FLOOR_GIB - heaps_gib - anchors_gib - rings_gib - overhead_gib
    launch = common - backup_p_gib - LOAD_TRANSIENT_GIB
    run = common - backup_p_gib - backup_d_gib
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
        "floor_gib": FLOOR_GIB,
        "heaps_gib": heaps_gib,
        "backup_p_gib": backup_p_gib,
        "backup_d_gib": backup_d_gib,
        "load_transient_gib": LOAD_TRANSIENT_GIB,
        "anchors_gib": anchors_gib,
        "rings_gib": rings_gib,
        "overhead_gib": overhead_gib,
    }
    arm.launch_leftover_gib = launch
    arm.run_leftover_gib = run
    return arm


def choose(
    memtotal_bytes: int,
    memavail_bytes: int,
    *,
    store_min_gib: float,
    arms: Sequence[Tuple[int, int]] = DEFAULT_ARMS,
    ranks_per_group: int = 3,
) -> Tuple[Arm, float, List[str]]:
    """Walk the ladder; return (arm, store_gib, printed lines) or raise W20.

    ``store_gib`` is the RUN leftover floored to whole GiB: the tmpfs the
    canonical page store lives on.  A leftover below ``store_min_gib`` on
    every arm is a refusal -- a carrier that cannot hold one agent prefix is
    not a carrier, and the boot would pass R1 and fail R2/R4 for a reason the
    ledger already knew.
    """
    lines: List[str] = []
    priced = [
        price(memtotal_bytes, memavail_bytes, s, m, ranks_per_group=ranks_per_group)
        for s, m in arms
    ]
    t = priced[0].terms
    lines.append(
        "WEG2-HOST-LEDGER TERMS "
        f"memtotal={t['memtotal_gib']:.2f} GiB memavail={t['memavail_gib']:.2f} GiB "
        f"(live /proc/meminfo) cli_reserve={CLI_RESERVE_GIB:.0f} GiB (record 1g, "
        f"charged once against MemTotal) base=min(memavail, memtotal-cli)="
        f"{t['base_gib']:.2f} GiB floor={FLOOR_GIB:.0f} GiB (#721) "
        f"heaps={t['heaps_gib']:.2f} GiB ({ranks_per_group}x{HEAP_AWAKE_GIB} awake b0 + "
        f"{ranks_per_group}x{HEAP_DORMANT_GIB} dormant campaign (a)) "
        f"backup_P={t['backup_p_gib']:.2f} GiB backup_D={t['backup_d_gib']:.2f} GiB "
        "(#809 census images, 1x each, DR-1 shm-backed) "
        f"load_transient={LOAD_TRANSIENT_GIB:.0f} GiB (#721, launch moment only) "
        f"anchors@2400={ANCHORS_AT_2400_BYTES / GIB:.2f} GiB (b0 measured, scaled by M) "
        f"rings=({RING_P_MULT_GB_PER_S:.0f}+{RING_D_MULT_GB_PER_S:.0f})xS GB (b0) "
        f"overhead={HOST_POOL_OVERHEAD:.0%} of host-pool posts (b0 U14)"
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
            f"plus a {store_min_gib:.0f} GiB store floor on this box. Both groups "
            "need a cpu backup (the INT8 checkpoint has only the cpu-backup wake "
            "path, W4), and the ledger will not shrink another term silently.\n"
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
    return chosen, store_gib, lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-min-gib", type=float, default=4.0)
    ap.add_argument("--meminfo", default="/proc/meminfo")
    ns = ap.parse_args(argv)
    mi = read_meminfo(ns.meminfo)
    try:
        arm, store, lines = choose(
            mi["MemTotal"], mi["MemAvailable"], store_min_gib=ns.store_min_gib
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
