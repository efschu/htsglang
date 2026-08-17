#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Turnkey falsifier for #464 -- coalesced VMM resume, call count and wall time.

WHAT #464 CHANGES. ``KvVmmArena.commit_range`` splits a gap into
``self._chunk``-sized extents and issues one map + one setAccess per extent.
When the run is CONTIGUOUS those extents describe one VA region, so one handle
suffices and the resume becomes ~3 driver calls (map, setAccess, memset). The
coalescer is built and default-off; ``SGLANG_VMM_COALESCE_RESUME=1`` turns it
on. This script measures whether that is worth anything.

TWO CORRECTIONS TO THE TICKET'S PREMISE ARE ENCODED HERE, because a runner that
carried them silently would produce a number nobody could interpret.

**(1) The 40-85 ms band is NOT this path's baseline.** That band is the
graph-state swap cost, measured at ``adaptive_graph_memory.py:207-214`` as "the
price of remapping+zeroing ~1 GB per swap". That path swaps through
``torch_memory_saver`` pause/resume -- a third-party package -- and references
none of ``KvVmmArena``, ``commit_range``, ``cuMemMap`` or ``vmm_utils``.
``KvVmmArena`` backs the draft-weight carrier, the weights-arena tail and the
KV pool; no graph state is routed through it. So "beat 40-85 ms" is not a
like-for-like acceptance criterion for this change. This runner therefore
compares ON against OFF **on the same path, in the same process**, and reports
the band only as the analogous prior measurement it is. Reporting a delta
against a band from a different mechanism would be the #709 mistake: an
acceptance rule that cannot see what it claims to judge.

**(2) "~500 x 2 MiB calls" is chunk-dependent, and no default produces it.**
The extent count is ``ceil(nbytes / chunk)``, and the chunks actually in use
are ``SGLANG_FLIP_SEAM_CHUNK_MIB`` (default **8 MiB**, ``memory_pool.py:2477``)
and ``CARRIER_COMMIT_CHUNK`` (**64 MiB**, ``phase_flip_spill.py:219``). For
~1 GiB that is ~128 extents and ~16 extents respectively -- ~257 and ~33
driver calls, not ~1001. The 2 MiB figure is the allocation-granularity
fallback, not a chunk any default sets. So the runner computes the expected
count from the REAL chunk instead of asserting a number from the ticket.

The call count is ANALYTIC (it follows from the plan and is exact); only the
wall time needs a card. That split is deliberate: the count half is checkable
at the desk and is checked below, so the window only has to buy the timing.

USAGE

    # hermetic; no GPU, no window. Proves the harness works and can fail.
    python bench/464/run_464_resume.py --self-test

    # inside a claimed window (/spinning/gpu-arb/; this script claims nothing)
    python bench/464/run_464_resume.py --run --device 0 --mib 1024

Exit: 0 = measured, 1 = a check failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

MIB = 1024 * 1024

#: The chunks really in use, so the runner reports against reality rather than
#: against the ticket's 2 MiB illustration.
KNOWN_CHUNKS_MIB = {
    "kv_seam (SGLANG_FLIP_SEAM_CHUNK_MIB default)": 8,
    "carriers (CARRIER_COMMIT_CHUNK)": 64,
    "granularity fallback (the ticket's illustration)": 2,
}

#: Graph-state swap, `adaptive_graph_memory.py:207-214`. Carried for context
#: ONLY -- see correction (1). Never an acceptance threshold for this path.
GRAPH_STATE_BAND_MS = (40.0, 85.0)


def plan_extents(nbytes: int, chunk_bytes: Optional[int]) -> List[Tuple[int, int]]:
    """Mirror ``commit_range``'s split of one contiguous gap into extents."""
    if nbytes <= 0:
        return []
    if not chunk_bytes:
        return [(0, nbytes)]
    out: List[Tuple[int, int]] = []
    pos = 0
    while pos < nbytes:
        step = min(chunk_bytes, nbytes - pos)
        out.append((pos, step))
        pos += step
    return out


def driver_calls(extents: int) -> int:
    """One map + one setAccess per extent, plus one memset for the region."""
    return 2 * extents + 1 if extents else 0


@dataclass
class Arm:
    label: str
    extents: int
    calls: int
    ms: Optional[float] = None


@dataclass
class Result:
    nbytes: int = 0
    chunk_bytes: int = 0
    off: Optional[Arm] = None
    on: Optional[Arm] = None
    notes: List[str] = field(default_factory=list)

    @property
    def calls_saved(self) -> int:
        if not (self.off and self.on):
            return 0
        return max(0, self.off.calls - self.on.calls)

    def render(self) -> str:
        lines = [
            "## #464 coalesced VMM resume",
            "",
            f"payload      : {self.nbytes / MIB:.0f} MiB",
            f"commit chunk : {self.chunk_bytes / MIB:.0f} MiB",
            "",
            "arm        extents   driver calls   wall ms",
        ]
        for arm in (self.off, self.on):
            if arm is None:
                continue
            ms = "PENDING" if arm.ms is None else f"{arm.ms:.2f}"
            lines.append(f"{arm.label:9s} {arm.extents:8d} {arm.calls:14d}   {ms}")
        lines.append("")
        lines.append(f"driver calls saved: {self.calls_saved}")
        if self.off and self.on and self.off.ms and self.on.ms:
            delta = self.off.ms - self.on.ms
            lines.append(f"wall-time delta   : {delta:+.2f} ms (off - on)")
        lines += [
            "",
            "CONTEXT, not a threshold: the graph-state swap band is "
            f"{GRAPH_STATE_BAND_MS[0]:.0f}-{GRAPH_STATE_BAND_MS[1]:.0f} ms "
            "(adaptive_graph_memory.py:207-214). That path swaps through "
            "torch_memory_saver, NOT through KvVmmArena, so it is an analogue "
            "and never this path's acceptance criterion.",
        ]
        lines += self.notes
        return "\n".join(lines)


def analytic_arms(nbytes: int, chunk_bytes: int) -> Tuple[Arm, Arm]:
    """The exact call counts both ways. No card needed; this is arithmetic."""
    off_extents = len(plan_extents(nbytes, chunk_bytes))
    off = Arm("off", off_extents, driver_calls(off_extents))
    # Coalescing a CONTIGUOUS run yields exactly one extent.
    on = Arm("on", 1, driver_calls(1))
    return off, on


def self_test() -> int:
    """Hermetic proof the harness works, and that its checks can fail."""
    failures: List[str] = []
    ran: List[str] = []

    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    gib = 1024 * MIB

    # -- extent arithmetic against the chunks really in use
    check("8 MiB chunk gives 128 extents per GiB", len(plan_extents(gib, 8 * MIB)) == 128)
    check("64 MiB chunk gives 16 extents per GiB", len(plan_extents(gib, 64 * MIB)) == 16)
    check("2 MiB chunk gives 512 extents per GiB", len(plan_extents(gib, 2 * MIB)) == 512)
    check("no chunk is one monolithic extent", len(plan_extents(gib, None)) == 1)
    check("an empty payload plans nothing", plan_extents(0, 8 * MIB) == [])

    # An uneven tail must still be covered exactly -- a resume that maps a
    # different number of bytes than asked for is a bug, not an optimisation.
    uneven = plan_extents(3 * MIB + 7, 2 * MIB)
    check("uneven tail is covered", sum(step for _, step in uneven) == 3 * MIB + 7)
    check("uneven tail has a short last extent", uneven[-1][1] == MIB + 7)

    # -- driver-call model
    check("calls = 2*extents + 1", driver_calls(128) == 257)
    check("one extent is three calls", driver_calls(1) == 3)
    check("no extents is no calls", driver_calls(0) == 0)

    # -- the headline the ticket asserts, recomputed rather than assumed
    off, on = analytic_arms(gib, 8 * MIB)
    check("seam default: off is 257 calls", off.calls == 257)
    check("seam default: on is 3 calls", on.calls == 3)
    r = Result(nbytes=gib, chunk_bytes=8 * MIB, off=off, on=on)
    check("seam default saves 254 calls", r.calls_saved == 254)

    off64, on64 = analytic_arms(gib, 64 * MIB)
    r64 = Result(nbytes=gib, chunk_bytes=64 * MIB, off=off64, on=on64)
    check("carriers save only 30 calls", r64.calls_saved == 30)

    # -- THE CORRECTION, asserted rather than described: the 40-85 ms band must
    #    not be used as a pass/fail threshold anywhere in the report.
    text = r.render()
    check("report names the band as context", "not a threshold" in text)
    check("report says which module the band came from", "torch_memory_saver" in text)
    check("report shows PENDING wall time before a card runs", "PENDING" in text)

    # -- a saving must never be reported when nothing was compared
    check("no arms means no saving claimed", Result().calls_saved == 0)
    # -- and coalescing must never claim to save on a single-extent plan
    off1, on1 = analytic_arms(4 * MIB, 64 * MIB)
    check(
        "a single-extent payload saves nothing",
        Result(off=off1, on=on1).calls_saved == 0,
    )

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"self-test: OK ({len(ran)} checks)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mib", type=int, default=1024, help="payload size")
    ap.add_argument("--chunk-mib", type=int, default=8, help="commit chunk")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="/tmp/464_resume")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.run:
        ap.print_help()
        return 2

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"cannot run: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("cannot run: no CUDA device. The wall-time half of this gate is GPU-only;")
        print("the call-count half is arithmetic and is covered by --self-test.")
        return 2

    from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, align_up

    nbytes = args.mib * MIB
    chunk = args.chunk_mib * MIB
    os.makedirs(args.out, exist_ok=True)
    result = Result(nbytes=nbytes, chunk_bytes=chunk)
    result.off, result.on = analytic_arms(nbytes, chunk)

    for enabled, arm in ((False, result.off), (True, result.on)):
        arena = KvVmmArena(
            args.device,
            reserve_bytes=align_up(nbytes, 2 * MIB) + 64 * MIB,
            commit_chunk_bytes=chunk,
            coalesce_resume=enabled,
        )
        samples: List[float] = []
        for _ in range(args.repeats):
            # decommit_range's second argument is KEEP bytes, not the amount to
            # release -- so a full release is keep=0. Passing nbytes here would
            # keep everything, and every repeat after the first would time a
            # no-op commit and report a flattering zero.
            arena.decommit_range(0, 0)
            torch.cuda.synchronize(args.device)
            t0 = time.perf_counter()
            arena.commit_range(0, nbytes)
            torch.cuda.synchronize(args.device)
            samples.append((time.perf_counter() - t0) * 1e3)
        # Median, not mean: one driver hiccup must not set the number.
        samples.sort()
        arm.ms = samples[len(samples) // 2]
        result.notes.append(f"{arm.label}: samples ms = {[round(s, 2) for s in samples]}")
        del arena

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(
            {
                "nbytes": nbytes,
                "chunk_bytes": chunk,
                "off": {"extents": result.off.extents, "calls": result.off.calls, "ms": result.off.ms},
                "on": {"extents": result.on.extents, "calls": result.on.calls, "ms": result.on.ms},
                "calls_saved": result.calls_saved,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.out, "TICKET_BLOCK.md"), "w") as f:
        f.write(result.render() + "\n")
    print(result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
