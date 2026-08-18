"""#702 frontier solve on the current lineage (post-#723 enumeration).

Anchors verbatim from test_prefill_frontier_702.py (the instrument boot
c3e94878ff + the #707 seam closed form). Incumbent re-based to the ACTUAL
composite boot argv: --pp-stage-ratio 31,17,16 (argv_735_composite.txt:13).
"""

import sys

sys.path.insert(0, "/spinning/wt-602-slot2/python")

from sglang.srt.planner.prefill_frontier import solve_prefill_frontier
from sglang.srt.planner.seam_holdback import (
    SeamHoldbackError,
    SeamRecord,
    available_bytes_for_cut,
)

MIB = 1024 * 1024
CELL = 2048
INSTRUMENT = (28, 20, 16)  # the boot the seam record is anchored at
INSTRUMENT_ATTN = (7, 5, 4)
LIVE_INCUMBENT = (31, 17, 16)  # argv_735_composite.txt
MS_PER_LAYER = (1.7571, 7.740, 7.275)
X8 = 13.0e9 / MIB
X4 = 6.4e9 / MIB
LINKS = (X8, X8, X4)
GATHER_MIB = 24.09
FAM = tuple((i + 1) % 4 == 0 for i in range(64))
NOISE_FLOOR = 0.141

_ALLOWED = tuple(
    a * MIB / c for a, c in zip((8129.5, 4520.7, 3408.4), (14336, 10240, 8192))
)
RECORD = SeamRecord(
    id_space_tokens=min(_ALLOWED),
    bracket_mib=tuple(
        (_ALLOWED[i] - min(_ALLOWED)) * (14336, 10240, 8192)[i] / MIB
        for i in range(3)
    ),
    cell_bytes=(14336, 10240, 8192),
)


def attn_for(counts):
    out, s = [], 0
    for c in counts:
        out.append(sum(1 for i in range(s, s + c) if FAM[i]))
        s += c
    return tuple(out)


def avail_for(counts, attn):
    return available_bytes_for_cut(RECORD, INSTRUMENT, INSTRUMENT_ATTN, counts, attn)


def pipelined(counts):
    return max(MS_PER_LAYER[i] * counts[i] for i in range(3))


f = solve_prefill_frontier(
    total_layers=64,
    n_stages=3,
    incumbent=INSTRUMENT,
    incumbent_pool_tokens=436_278.0,
    ms_per_layer=MS_PER_LAYER,
    attn_counts_for=attn_for,
    available_bytes_for=avail_for,
    kv_bytes_per_token_per_attn_layer=CELL,
    total_attn_layers=16,
    gather_mib_per_attn_layer=GATHER_MIB,
    link_mib_per_s=LINKS,
    measured_for=lambda c: tuple(c) == INSTRUMENT,
    noise_floor=NOISE_FLOOR,
)

base_live = pipelined(LIVE_INCUMBENT)
print(f"live incumbent {LIVE_INCUMBENT}: pipelined {base_live:.2f} ms, "
      f"attn {attn_for(LIVE_INCUMBENT)}")

# Live incumbent's own pool row via the provider:
try:
    la = avail_for(LIVE_INCUMBENT, attn_for(LIVE_INCUMBENT))
    lc = attn_for(LIVE_INCUMBENT)
    live_pool = min(la[i] / (lc[i] * CELL) for i in range(3))
    print(f"live incumbent coupled pool (closed form): {live_pool:,.0f} tokens")
except SeamHoldbackError as e:
    live_pool = None
    print("live incumbent REFUSED:", e)

print()
hdr = (f"{'cut':>14} {'attn':>9} {'ms':>7} {'vs-live':>8} {'net-noPL':>9} "
       f"{'coupled-pool':>13} {'vs-live-pool':>12} {'ovh%':>6} flags")
print(hdr)
for p in sorted(f.points, key=lambda p: pipelined(p.counts)):
    ms = pipelined(p.counts)
    flags = []
    if p.needs_pipelining:
        flags.append("NEEDS-PIPELINING")
    if tuple(p.counts) == INSTRUMENT:
        flags.append("MEASURED-BOOT")
    if tuple(p.counts) == LIVE_INCUMBENT:
        flags.append("LIVE-INCUMBENT")
    vs_live = base_live / ms
    if 0 < (vs_live - 1.0) < NOISE_FLOOR and ms < base_live:
        flags.append("below-noise")
    vs_pool = (p.coupled_pool_tokens / live_pool) if live_pool else float("nan")
    print(f"{str(p.counts):>14} {str(p.attn_counts):>9} {ms:7.2f} {vs_live:8.3f} "
          f"{p.net_no_pipelining* base_live/pipelined(INSTRUMENT)/ (base_live/pipelined(INSTRUMENT)):9.3f} "
          f"{p.coupled_pool_tokens:13,.0f} {vs_pool:12.3f} {p.overhead*100:6.1f} "
          f"{','.join(flags)}")

# Arming-floor refusals, per lead depth (mandate (2): price it in, don't hide it)
print()
refused_total = 0
for n0 in range(28, 62):
    refused = 0
    for n1 in range(1, 64 - n0):
        counts = (n0, n1, 64 - n0 - n1)
        if counts[2] < 1:
            continue
        attn = attn_for(counts)
        if any(a <= 0 for a in attn):
            continue
        try:
            avail_for(counts, attn)
        except SeamHoldbackError:
            refused += 1
    if refused:
        print(f"lead {n0}: {refused} tail(s) REFUSED by the arming floor")
        refused_total += refused
print(f"total arming-floor refusals across enumeration: {refused_total}")
