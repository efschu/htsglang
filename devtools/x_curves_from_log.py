#!/usr/bin/env python3
"""x_curves_from_log: boot logs (P.log, D.log, front.log) -> weg2-x-curves/1 (S3 h16u 17.10., SOLL R10).

Re-uses what exists: the file schema/validation/encoding is ``flliper.srt.pdflip.x_curves`` and the forward collapsing
(``chunks==1``, max over the ranks of one forward) and the flip-price pairing are ``x_curves_build.forward_points`` /
``build_flip_price`` -- only the log reader and the per-depth LINEAR smoothing (ms = a + b*n, a,b >= 0, robust
Theil-Sen over n-cell medians) are new. Depth raster and n raster come from a template curve file (nf22a), so the new
file is a drop-in for the profile.

  python devtools/x_curves_from_log.py --p-log P.log --d-log D.log --front-log front.log \
      --template x_curves.nf22a.json --out x_curves.nf-h16u.json [--source TAG]
  python devtools/x_curves_from_log.py --selftest

A raster depth without enough points (< 3 distinct n cells spanning >= 512 tokens inside +-half a raster step) takes the
coefficients of its nearest fitted depths (linear in depth) and says so in the notes; ms beyond the largest observed n
is an extrapolation of the line and is counted per curve (never silent).
"""
from __future__ import annotations

import argparse
import calendar
import collections
import json
import os
import re
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))

from flliper.srt.pdflip import x_curves as xc  # noqa: E402
from flliper.srt.pdflip import x_curves_build as xb  # noqa: E402

BATCH_RE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([A-Z]+\d+)\] Prefill rank batch, #new-token: (\d+), #cached-token: (\d+), "
    r"#chunks: (\d+), gpu-ms: ([\d.]+)"
)
USERZEIT_RE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)\] .*PDFLIP-FLIP-USERZEIT dir=([DP]>[DP]) epoch=(\d+) user_flipzeit_ms=(\d+)"
    r"(?:.*? first_p_chunk_ts=([\d.]+))?"
)
N_MIN = 150  # raw points with fewer new tokens are not a forward cost (nf22a: n >= 150)


def _epoch(s: str, ms: int = 0) -> float:
    return calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M:%S")) + ms / 1000.0


def read_batches(path: str, role: str) -> List[dict]:
    rows = []
    with open(path, "r", errors="replace") as f:
        for ln in f:
            if "Prefill rank batch" not in ln:
                continue
            m = BATCH_RE.match(ln)
            if not m:
                continue
            rows.append({
                "role": role, "t": _epoch(m.group(1)), "rank": m.group(2), "new_tokens": int(m.group(3)),
                "cached_depth": int(m.group(4)), "chunks": int(m.group(5)), "gpu_ms": float(m.group(6)),
            })
    return rows


def read_flips(path: str, p_rows: Sequence[dict]) -> List[dict]:
    """PDFLIP-FLIP-USERZEIT lines -> role=flip rows; the D>P flip's depth = the cached depth of the first P batch at or
    after its ``first_p_chunk_ts`` (the request the flip was for)."""
    lead = sorted((r for r in p_rows if r["rank"] == "PP0" and r["chunks"] == 1), key=lambda r: r["t"])
    out, seen = [], set()
    with open(path, "r", errors="replace") as f:
        for ln in f:
            if "PDFLIP-FLIP-USERZEIT" not in ln:
                continue
            m = USERZEIT_RE.match(ln)
            if not m or (m.group(3), m.group(4)) in seen:
                continue
            seen.add((m.group(3), m.group(4)))
            t = _epoch(m.group(1), int(m.group(2)))
            row = {"role": "flip", "t": t, "direction": m.group(3), "flip_user_ms": float(m.group(5))}
            if m.group(3) == "D>P" and m.group(6):
                t_first = float(m.group(6))
                nxt = next((r for r in lead if r["t"] >= int(t_first) - 1), None)
                if nxt is not None:
                    row["depth"] = nxt["cached_depth"]
            out.append(row)
    return out


def theil_sen(cells: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(cells) for (x2, y2) in cells[i + 1:] if x2 > x1
    ]
    b = max(0.0, statistics.median(slopes)) if slopes else 0.0
    a = max(0.0, statistics.median(y - b * x for x, y in cells))
    return a, b


def fit_depths(points: Sequence[Tuple[int, int, float]], depths: Sequence[int], half_step: Sequence[int],
               n_bucket: int = 256) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int], Dict[int, int]]:
    """Per raster depth: (a, b) from the points inside [d - lo, d + hi]; counts of points; max observed n."""
    coef, cnt, nmax = {}, {}, {}
    for d, lo_hi in zip(depths, half_step):
        lo, hi = lo_hi
        sel = [(n, ms) for (dd, n, ms) in points if d - lo <= dd <= d + hi and n >= N_MIN]
        cells: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
        for n, ms in sel:
            cells[int(round(n / n_bucket))].append((n, ms))
        pts = sorted((statistics.median(x for x, _ in v), statistics.median(m for _, m in v)) for v in cells.values())
        cnt[d] = len(sel)
        nmax[d] = int(max((n for n, _ in sel), default=0))
        if len(pts) >= 3 and pts[-1][0] - pts[0][0] >= 512:
            coef[d] = theil_sen(pts)
    return coef, cnt, nmax


def fill_missing(coef: Dict[int, Tuple[float, float]], depths: Sequence[int]) -> Tuple[Dict[int, Tuple[float, float]], List[int]]:
    have = sorted(coef)
    if not have:
        raise xc.XCurvesRefused(xc.W_X_CURVES_MALFORMED, "no raster depth has enough points to fit a line")
    out, filled = dict(coef), []
    for d in depths:
        if d in coef:
            continue
        lo = max((h for h in have if h < d), default=None)
        hi = min((h for h in have if h > d), default=None)
        if lo is None or hi is None:
            a, b = coef[hi if lo is None else lo]
        else:
            w = (d - lo) / (hi - lo)
            a = coef[lo][0] * (1 - w) + coef[hi][0] * w
            b = coef[lo][1] * (1 - w) + coef[hi][1] * w
        out[d] = (a, b)
        filled.append(d)
    return out, filled


def build_curve(points, template: xc.PrefillCurve, label: str, notes: List[str]) -> xc.PrefillCurve:
    depths = [r.depth for r in template.rows]
    spans = []
    for i, d in enumerate(depths):
        lo = (d - depths[i - 1]) // 2 if i > 0 else 0
        hi = (depths[i + 1] - d) // 2 if i + 1 < len(depths) else 0
        spans.append((lo, max(hi, 1)))
    coef, cnt, nmax = fit_depths(points, depths, spans)
    coef, filled = fill_missing(coef, depths)
    rows, extrap = [], 0
    for r in template.rows:
        a, b = coef[r.depth]
        rows.append(xc.CurveRow(depth=r.depth, n=tuple(r.n), ms=tuple(round(max(a + b * n, 1.0), 2) for n in r.n)))
        extrap += sum(1 for n in r.n if n > nmax.get(r.depth, 0))
    notes.append(
        f"{label}: {len(points)} forwards; fitted depths {len(depths) - len(filled)}/{len(depths)}; "
        f"interpolated from neighbours: {filled or 'none'}; table cells beyond the observed max n (extrapolated): "
        f"{extrap}/{sum(len(r.n) for r in template.rows)}; points per depth: {[cnt[d] for d in depths]}"
    )
    return xc.PrefillCurve(
        rows=tuple(rows), chunk_tokens=template.chunk_tokens,
        instrument=f"{label} Prefill rank batch gpu_ms, max over ranks, chunks==1 -- per-depth linear fit ms = a + b*n "
                   f"(Theil-Sen over n-cell medians, a,b >= 0), raw n >= {N_MIN}; from boot logs (x_curves_from_log)",
        depth_interp=template.depth_interp, points=len(points),
    )


def build(p_rows, d_rows, flip_rows, template: xc.XCurves, source: str, flip_bucket: int = 50000):
    notes: List[str] = []
    p_pts, p_fold = xb.forward_points(p_rows, "p_chunk", "max")
    d_pts, d_fold = xb.forward_points(d_rows, "d_forward", "max")
    notes.append(f"folded lines skipped: P {p_fold}, D {d_fold}")
    p_curve = build_curve(p_pts, template.p_curve, "P", notes)
    d_curve = build_curve(d_pts, template.d_curve, "D", notes)
    fnotes: List[str] = []
    flip = xb.build_flip_price(flip_rows, depth_bucket=flip_bucket, notes=fnotes)
    notes += [f"flip: {flip.pairs} pair(s), attribution={flip.attribution}"] + fnotes
    ident = xc.CurveIdentity(
        model=template.identity.model, form=template.identity.form, hardware=template.identity.hardware,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source=source,
    )
    cur = xc.XCurves(format=xc.X_CURVES_FORMAT, identity=ident, p_curve=p_curve, d_curve=d_curve, flip_price=flip)
    xc.validate(cur)
    return cur, notes


def selftest() -> int:
    import tempfile

    # synthetic: P depth 0 line ms = 1000 + 0.5 n ; depth 16384 ms = 1200 + 0.5 n; D flat 800 + 1.0 n
    pr, dr = [], []
    t = 1_000_000
    for d, a in ((0, 1000.0), (16384, 1200.0)):
        for n in (300, 700, 1100, 1500, 2100, 3000, 4000):
            t += 5
            for rk, f in (("PP0", 1.0), ("PP1", 0.5)):
                pr.append({"role": "p_chunk", "t": t, "rank": rk, "new_tokens": n, "cached_depth": d, "chunks": 1,
                           "gpu_ms": f * (a + 0.5 * n)})
            dr.append({"role": "d_forward", "t": t, "rank": "TP0", "new_tokens": n, "cached_depth": d, "chunks": 1,
                       "gpu_ms": 800.0 + n})
    tmpl_curve = xc.PrefillCurve(
        rows=(xc.CurveRow(0, (256, 4096), (1.0, 2.0)), xc.CurveRow(16384, (256, 4096), (1.0, 2.0))),
        chunk_tokens=4096, instrument="t")
    tmpl = xc.XCurves(format=xc.X_CURVES_FORMAT,
                      identity=xc.CurveIdentity("m", "f", "h", "c", "s"), p_curve=tmpl_curve, d_curve=tmpl_curve,
                      flip_price=xc.FlipPrice(depth=(0,), seconds=(1.0,), attribution="none", pairs=1))
    flips = [
        {"role": "flip", "t": 100.0, "direction": "D>P", "flip_user_ms": 4000.0, "depth": 0},
        {"role": "flip", "t": 110.0, "direction": "P>D", "flip_user_ms": 3000.0},
    ]
    cur, notes = build(pr, dr, flips, tmpl, "selftest")
    p0 = dict(zip(cur.p_curve.rows[0].n, cur.p_curve.rows[0].ms))
    p1 = dict(zip(cur.p_curve.rows[1].n, cur.p_curve.rows[1].ms))
    d0 = dict(zip(cur.d_curve.rows[0].n, cur.d_curve.rows[0].ms))
    assert abs(p0[256] - 1128.0) < 1.0 and abs(p0[4096] - 3048.0) < 1.0, p0  # max over ranks = PP0
    assert abs(p1[256] - 1328.0) < 1.0, p1
    assert abs(d0[4096] - 4896.0) < 1.0, d0
    assert abs(cur.flip_price.seconds[0] - 7.0) < 1e-6, cur.flip_price
    m = BATCH_RE.match("[2026-10-16 17:25:56 PP0] Prefill rank batch, #new-token: 25, #cached-token: 0, #chunks: 1, gpu-ms: 1620.4 (compute")
    assert m and m.group(6) == "1620.4"
    m = USERZEIT_RE.match("[2026-10-16 17:25:56,640] INFO pdflip.front: PDFLIP-FLIP-USERZEIT dir=D>P epoch=1 user_flipzeit_ms=4242 missing=None start=x last_d_token_ts=1.0(x) waiter_arrival_ts=2.0(y) first_p_chunk_ts=1792171553.521 | z")
    assert m and m.group(5) == "4242" and m.group(6) == "1792171553.521"
    print("x_curves_from_log selftest OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-log")
    ap.add_argument("--d-log")
    ap.add_argument("--front-log")
    ap.add_argument("--template")
    ap.add_argument("--out")
    ap.add_argument("--source")
    ap.add_argument("--flip-depth-bucket", type=int, default=50000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for k in ("p_log", "d_log", "front_log", "template", "out"):
        if not getattr(a, k):
            ap.error(f"--{k.replace('_', '-')} is required")
    p_rows = read_batches(a.p_log, "p_chunk")
    d_rows = read_batches(a.d_log, "d_forward")
    flips = read_flips(a.front_log, p_rows)
    try:
        cur, notes = build(p_rows, d_rows, flips, xc.load(a.template),
                           a.source or os.path.basename(a.p_log).rsplit(".P.log", 1)[0], a.flip_depth_bucket)
    except xc.XCurvesRefused as e:
        print(f"X-CURVES FROM-LOG REFUSED {e}", file=sys.stderr)
        return 2
    with open(a.out, "wb") as f:
        f.write(xc.encode(cur))
    for n in notes:
        print(f"X-CURVES FROM-LOG {n}")
    print(xc.text_table(cur))
    print(f"X-CURVES FROM-LOG wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
