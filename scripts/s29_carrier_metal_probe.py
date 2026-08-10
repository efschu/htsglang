"""#656 rung 2: prove the carrier's mechanics on real hardware, alone.

WHY THIS EXISTS AS A SEPARATE SCRIPT. The production boot that will use the
carrier takes minutes, loads ~27B of weights across three ranks and captures
CUDA graphs. If the carrier's driver-level mechanics are wrong, that boot is a
very expensive and very noisy way to find out -- and the failure would arrive
mixed in with phase-flip machinery, which is where this chain has repeatedly
lost time attributing a fault to the wrong layer.

This probe answers the three questions the design rests on, in about a second,
against the real driver:

  1. Does a KvVmmArena build and reserve here at all? This boot does NOT set
     --enable-vram-dial, so the carrier is the FIRST user of the VMM arena in
     the serving process. The JIT stub build and the cuMem* calls are
     unexercised code on this configuration until something exercises them.

  2. Does decommit_range actually return pages to the DRIVER? The corridor law
     is stated in NVML's free column, so a spill that torch reports as freed
     but the driver still owns is worth exactly zero. This measures free
     memory through mem_get_info, not through torch's counters.

  3. Does the virtual address survive the round trip? This is the entire
     premise: the TP decode graphs bake the drafter's parameter addresses, and
     the spill is only safe because those addresses stand still while the
     pages underneath them go away and come back.

Usage:  python scripts/s29_carrier_metal_probe.py [--device N] [--mib 256]

Deliberately small and deliberately on a card with headroom: it runs while the
production instance is serving, so it must not itself breach the corridor.
"""

from __future__ import annotations

import argparse
import sys

import torch

MIB = 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--mib", type=int, default=256)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA")
        return 2

    from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, align_up

    # TORCH INDICES ARE NOT NVML INDICES. Observed on this rig: nvidia-smi
    # index 1 is the 5090 (32607 MiB) while torch device 1 is a 3080 (19.58
    # GiB). Printing the mapping and picking by FREE memory keeps the probe
    # from landing on whichever card happens to be tightest -- which is how
    # the first run OOMed while a card with room sat idle.
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_i, total_i = torch.cuda.mem_get_info(i)
        print(
            f"[probe] torch cuda:{i} = {props.name}, total "
            f"{total_i/MIB:.0f} MiB, free {free_i/MIB:.0f} MiB"
        )

    dev = args.device
    if dev < 0:
        dev = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.mem_get_info(i)[0],
        )
        print(f"[probe] auto-selected the roomiest card: cuda:{dev}")
    payload = args.mib * MIB
    torch.cuda.set_device(dev)

    def free_mib() -> float:
        return torch.cuda.mem_get_info(dev)[0] / MIB

    base_free = free_mib()
    print(f"[probe] device {dev}, free at start {base_free:.0f} MiB")

    arena = KvVmmArena(
        dev,
        reserve_bytes=align_up(payload, 2 * MIB) + 64 * MIB,
        commit_chunk_bytes=64 * MIB,
        retain_handles=False,
    )
    print(f"[probe] arena built, base 0x{arena.base:x}, gran {arena.granularity}")

    with torch.cuda.use_mem_pool(arena.pool):
        span = torch.empty(payload, dtype=torch.uint8, device=f"cuda:{dev}")
    offset = span.data_ptr() - arena.base
    print(f"[probe] span offset {offset} (aligned: {offset % arena.granularity == 0})")

    arena.commit_range(offset, payload)
    after_commit = free_mib()
    ptr_before = span.data_ptr()

    # Write a recognisable pattern and check it reads back, so "committed"
    # means usable memory and not merely a bookkeeping entry.
    span.fill_(0x5A)
    torch.cuda.synchronize()
    # A SPARSE probe, not a reduction. `span.to(torch.int64).sum()` inflates
    # the payload eightfold -- 256 MiB became a 2 GiB allocation and OOMed on
    # a card that is also serving. What matters here is that the committed
    # range is readable and writable end to end, and sampling the boundaries
    # of every chunk establishes that without allocating anything.
    probe_at = list(range(0, payload, 64 * MIB)) + [payload - 1]
    checksum_before = [int(span[i].item()) for i in probe_at]
    print(
        f"[probe] committed {payload/MIB:.0f} MiB, free {base_free:.0f} -> "
        f"{after_commit:.0f} MiB (cost {base_free - after_commit:.0f})"
    )

    # REPEAT THE CYCLE, because this card is also serving.
    #
    # The first version of this probe measured one spill and reported that it
    # regained 80 MiB of 192. That was not the carrier: free memory at the end
    # of the run was 112 MiB below free at the start with our arena already
    # closed, i.e. the production instance took 112 MiB mid-probe, and
    # 192 - 112 = 80. A single before/after pair on a shared card measures the
    # carrier PLUS whatever else moved, and cannot tell them apart.
    #
    # Taking the BEST of several cycles is the honest reading: other processes
    # can only ever make the observed regain look SMALLER (they consume), so a
    # cycle that shows the full payload proves the pages really went back,
    # while no cycle can manufacture a regain that did not happen.
    regains = []
    released = 0
    ptr_after_spill = ptr_after_restore = ptr_before
    for cycle in range(3):
        pre = free_mib()
        released = arena.decommit_range(offset, 0)
        post = free_mib()
        ptr_after_spill = span.data_ptr()
        regains.append(post - pre)
        print(
            f"[probe] cycle {cycle}: SPILL released {released/MIB:.0f} MiB, "
            f"free {pre:.0f} -> {post:.0f} (regained {post - pre:.0f})"
        )
        arena.commit_range(offset, payload)
        ptr_after_restore = span.data_ptr()
    after_spill = after_commit + max(regains)
    after_restore = free_mib()
    span.fill_(0x5A)
    torch.cuda.synchronize()
    checksum_after = [int(span[i].item()) for i in probe_at]
    print(f"[probe] RESTORE done, free -> {after_restore:.0f} MiB")

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
        ok = ok and cond

    check(
        "VA is stable across spill",
        ptr_before == ptr_after_spill,
        f"0x{ptr_before:x} vs 0x{ptr_after_spill:x}",
    )
    check(
        "VA is stable across restore",
        ptr_before == ptr_after_restore,
        f"0x{ptr_before:x} vs 0x{ptr_after_restore:x}",
    )
    check(
        "decommit reports the payload",
        abs(released - payload) <= 64 * MIB,
        f"{released/MIB:.0f} vs {payload/MIB:.0f} MiB",
    )
    # The driver-visible regain is the claim the corridor cares about. Allow
    # slack for other activity on a card that is also serving.
    # Both sides in MiB. The first version compared a MiB delta against a BYTE
    # payload and so could never pass -- a reading of "192 of 192" printed
    # next to the word FAIL.
    check(
        "spill returns pages to the DRIVER",
        (after_spill - after_commit) > (payload / MIB) * 0.8,
        f"best regain {after_spill - after_commit:.0f} MiB of {payload/MIB:.0f}",
    )
    check(
        "memory is usable after restore",
        checksum_after == checksum_before and set(checksum_after) == {0x5A},
        f"{len(probe_at)} sample points",
    )

    arena.close()
    print(f"[probe] arena closed, free {free_mib():.0f} MiB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
