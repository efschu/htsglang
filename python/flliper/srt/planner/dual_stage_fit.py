"""Dual layout (27B NVFP4): P stage cost fitted from a boot's own ``Prefill rank batch`` lines, and the P cut priced with it.

WHY THIS EXISTS. ``planner/pgap_stage_fit.py`` fits the stage cost from ``#PGAP`` lines, but the dual layout
runs without ``#PGAP`` (launcher: ``_pgap = p_hostgap and not dual_layout``, user order 20.10. "the dual layout stays
as it was"), so there is no fit source for the dual cut. Every P rank prints one ``Prefill rank batch ... gpu-ms:``
line per forward at INFO anyway; this module reads those and nothing else. DESK TOOL, no launcher wiring: the dual cut
is pinned (``--pp-stage-ratio``), this prices candidates for the operator.

THE MODEL (per P chunk of ``chunk`` tokens, depth ``x`` = mean prefix in 1k tokens, measured on boots 27.10.):

    T_s(x) = a_s + b_s * x          (one line per stage, fitted on full 1024-token chunks at rung 0)

A candidate cut ``(n0, n1, n2)`` is priced from the MEASURED cut's fits with the same assumptions as
``pgap_stage_fit``: the per-layer rate of a card class comes from its middle stage (3080: PP1), the last stage keeps
the remainder as a per-forward term, and the depth slope is per FULL-ATTENTION layer (every 4th layer, index % 4 == 3).
The pipeline cycle of a chunk is the slowest stage; a prompt's time is the sum of its chunks' cycles (+ pipeline fill),
times a calibration factor fitted on the measured cut's own TTFT.
"""

from __future__ import annotations

import argparse
import datetime
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec

#: ``[2026-10-27 14:20:55 PP1] Prefill rank batch, #new-token: 1024, #cached-token: 0, #chunks: 1, gpu-ms: 284.7 ... rung=0``
_LINE = re.compile(
    r"\[(?P<date>\d{4}-\d\d-\d\d) (?P<time>\d\d:\d\d:\d\d) PP(?P<rank>\d)\] Prefill rank batch, "
    r"#new-token: (?P<new>\d+), #cached-token: \d+, #chunks: \d+, gpu-ms: (?P<ms>[0-9.]+)(?P<rest>.*)"
)
_RUNG = re.compile(r"rung=(\d)")

FA_PERIOD = 4          # Qwen3.8 hybrid: every 4th layer is full attention (index % 4 == 3)
FULL_CHUNK = 1024      # the dual P ladder's wide chunk (profile --p-chunk-max 1024)
SAME_PROMPT_GAP_S = 2.5
MIN_FIT_POINTS = 30


class RankBatch(msgspec.Struct, frozen=True):
    t_s: float
    rank: int
    new_tokens: int
    gpu_ms: float
    rung: int


class StageFit(msgspec.Struct, frozen=True):
    a_ms: float
    b_ms_per_1k: float
    sd_ms: float
    n: int


class Costs(msgspec.Struct, frozen=True):
    """Per-layer and per-forward terms of one measured cut (all per 1024-token chunk)."""
    layer_ms_fast: float      # PP0 (5090) incl. its embedding, folded into the per-layer rate
    layer_ms_slow: float      # middle 3080 stage
    last_stage_ms: float      # what the last stage costs beyond its layers at the slow rate
    attn_ms_per_1k_fast: float
    attn_ms_per_1k_slow: float


def _epoch(date: str, clock: str) -> float:
    return datetime.datetime.fromisoformat(f"{date} {clock}").timestamp()


def parse_rank_batches(lines: Iterable[str]) -> List[RankBatch]:
    out = []
    for line in lines:
        m = _LINE.search(line)
        if m is None:
            continue
        rung = _RUNG.search(m.group("rest"))
        out.append(RankBatch(t_s=_epoch(m.group("date"), m.group("time")), rank=int(m.group("rank")),
                             new_tokens=int(m.group("new")), gpu_ms=float(m.group("ms")),
                             rung=int(rung.group(1)) if rung else -1))
    return out


def depth_points(batches: Sequence[RankBatch], *, rank: int, chunk: int = FULL_CHUNK, rung: int = 0
                 ) -> List[Tuple[float, float]]:
    """``(mean prefix in 1k, gpu-ms)`` of every full ``chunk`` forward of one rank. Depth is rebuilt by summing the
    chunk widths of consecutive forwards; a gap above SAME_PROMPT_GAP_S starts a new prompt (the log carries no depth:
    ``#cached-token`` is the prefix-cache hit, 0 here)."""
    pts: List[Tuple[float, float]] = []
    depth, last = 0, None
    for b in sorted((b for b in batches if b.rank == rank and b.new_tokens >= 400), key=lambda b: b.t_s):
        if last is None or b.t_s - last > SAME_PROMPT_GAP_S:
            depth = 0
        if b.new_tokens == chunk and b.rung == rung:
            pts.append(((depth + chunk / 2) / 1000.0, b.gpu_ms))
        depth += b.new_tokens
        last = b.t_s
    return pts


def _ols(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


def fit_line(pts: Sequence[Tuple[float, float]]) -> StageFit:
    """Least squares ``a + b*x``, refitted once without the worst decile (a host hiccup is not stage cost)."""
    if len(pts) < MIN_FIT_POINTS:
        raise ValueError(f"{len(pts)} full chunks < {MIN_FIT_POINTS}: no fit")
    a, b = _ols(pts)
    resid = sorted(abs(y - (a + b * x)) for x, y in pts)
    keep_below = resid[int(0.9 * (len(resid) - 1))]
    kept = [(x, y) for x, y in pts if abs(y - (a + b * x)) <= keep_below]
    a, b = _ols(kept)
    sd = (sum((y - (a + b * x)) ** 2 for x, y in kept) / len(kept)) ** 0.5
    return StageFit(a_ms=a, b_ms_per_1k=b, sd_ms=sd, n=len(kept))


def fa_layers(cut: Sequence[int]) -> Tuple[int, ...]:
    """Full-attention layers per stage of a contiguous cut: layers with index % 4 == 3."""
    out, lo = [], 0
    for n in cut:
        out.append(sum(1 for i in range(lo, lo + n) if i % FA_PERIOD == FA_PERIOD - 1))
        lo += n
    return tuple(out)


def costs_from_fits(fits: Sequence[StageFit], cut: Sequence[int]) -> Costs:
    """The measured cut's three line fits -> per-layer / per-forward / per-attention-layer terms."""
    if len(fits) != 3 or len(cut) != 3:
        raise ValueError("dual P has three stages")
    fa = fa_layers(cut)
    slow = fits[1].a_ms / cut[1]
    return Costs(layer_ms_fast=fits[0].a_ms / cut[0], layer_ms_slow=slow,
                 last_stage_ms=fits[2].a_ms - cut[2] * slow,
                 attn_ms_per_1k_fast=fits[0].b_ms_per_1k / fa[0],
                 attn_ms_per_1k_slow=(fits[1].b_ms_per_1k + fits[2].b_ms_per_1k) / (fa[1] + fa[2]))


def stage_ms(cut: Sequence[int], costs: Costs, *, depth_1k: float) -> Tuple[float, float, float]:
    fa = fa_layers(cut)
    return (cut[0] * costs.layer_ms_fast + fa[0] * costs.attn_ms_per_1k_fast * depth_1k,
            cut[1] * costs.layer_ms_slow + fa[1] * costs.attn_ms_per_1k_slow * depth_1k,
            cut[2] * costs.layer_ms_slow + costs.last_stage_ms + fa[2] * costs.attn_ms_per_1k_slow * depth_1k)


def ttft_s(cut: Sequence[int], costs: Costs, *, prompt_tokens: int, chunk: int = FULL_CHUNK,
           calibration: float = 1.0) -> float:
    """Sum of the slowest-stage time of every chunk, plus the fill of the first chunk, times ``calibration``."""
    total = 0.0
    for i in range(prompt_tokens // chunk):
        total += max(stage_ms(cut, costs, depth_1k=(i * chunk + chunk / 2) / 1000.0))
    first = stage_ms(cut, costs, depth_1k=chunk / 2000.0)
    return (total + sum(first) - max(first)) * calibration / 1000.0


def calibration(cut: Sequence[int], costs: Costs, measured_ttft_s: Dict[int, float]) -> float:
    """Mean ratio measured/model over the given ``{prompt_tokens: TTFT}`` points of the MEASURED cut."""
    ratios = [t / ttft_s(cut, costs, prompt_tokens=n) for n, t in measured_ttft_s.items()]
    return sum(ratios) / len(ratios)


def fit_log(path: str, *, cut: Sequence[int]) -> Tuple[List[StageFit], Costs]:
    with open(path, errors="replace") as fh:
        batches = parse_rank_batches(fh)
    fits = [fit_line(depth_points(batches, rank=r)) for r in range(3)]
    return fits, costs_from_fits(fits, cut)


def _table(costs: Costs, cuts: Sequence[Sequence[int]], base: Sequence[int], cal: float) -> List[str]:
    rows = ["cut        tok/s at 16k / 64k / 128k / 256k prompt (model x calibration)"]
    for cut in cuts:
        rates = [n / ttft_s(cut, costs, prompt_tokens=n, calibration=cal) for n in (16384, 65536, 131072, 262144)]
        rows.append(f"{','.join(str(n) for n in cut):<10} " + " / ".join(f"{r:6.0f}" for r in rates)
                    + ("   (measured cut)" if tuple(cut) == tuple(base) else ""))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("p_log", help="a dual boot's P log (``Prefill rank batch`` lines)")
    ap.add_argument("--fit-cut", default="31,17,16", help="the cut the log was measured on")
    ap.add_argument("--cuts", default="31,17,16;32,16,16;33,16,15;34,15,15;35,15,14", help="candidates, ';'-separated")
    ap.add_argument("--ttft", default="65536:30.0,131072:68.0,262144:177.4",
                    help="measured TTFT of the fit cut as tokens:seconds list (calibration)")
    args = ap.parse_args(argv)
    base = tuple(int(v) for v in args.fit_cut.split(","))
    fits, costs = fit_log(args.p_log, cut=base)
    for r, f in enumerate(fits):
        print(f"PP{r}: a={f.a_ms:.1f} ms  b={f.b_ms_per_1k:.3f} ms/1k  sd={f.sd_ms:.1f}  n={f.n}")
    measured = {int(k): float(v) for k, v in (p.split(":") for p in args.ttft.split(","))}
    cal = calibration(base, costs, measured)
    print(f"costs {costs}  calibration {cal:.3f}")
    cuts = [tuple(int(v) for v in c.split(",")) for c in args.cuts.split(";")]
    print("\n".join(_table(costs, cuts, base, cal)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
