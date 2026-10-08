"""X-CURVES builder (06.10.): ``calib_x_<Boot>.jsonl`` -> one curve file.

The raw source is the calibration harvest of /spinning/gpu-arb/deskq/bl/
calib_dextend_nf.py, one JSON object per line:

* ``role=p_chunk``   -- P's ``Prefill rank batch`` lines (every PP rank):
  ``t, rank, new_tokens, cached_depth, chunks, gpu_ms, compute_ms, wait_ms``
* ``role=d_forward`` -- the same for D (every TP rank)
* ``role=d_extend``  -- one sweep request served on D: ``cached_depth,
  new_tokens, gpu_ms, gpu_ms_max, d_direct, ambiguous, error``
* ``role=flip``      -- one flip: ``t, epoch, direction (D>P|P>D), flip_user_ms``
  (the USER flip time, user order 02.10.: last token of the leaving phase ->
  first token of the arriving phase, end to end incl. lead-in and tail; D>P
  starts at ``held`` (the later of D's last token and the waiting request's
  arrival, 03.10.) and ends at P's first prefill forward, P>D runs from the
  end of P's last chunk to the first decode token on D); optional ``t_start``
  (unix s of that interval's start) and ``rid_prompt_tokens`` (informational,
  never read as a depth). ``flip_user_ms: null`` = the endpoint is MISSING and
  the flip is skipped. ``flip_user_ms`` is the only source of the flip price:
  a flip row without it is refused (write ``flip_user_ms`` per flip).
  Flipzeit = letztes Token abgehende Phase bis erstes Token ankommende Phase
  (flip_user_ms).
* ``role=calib_prefix`` -- the sweep's prefix request of one depth (the one
  that caused a flip pair): ``depth_target, prompt_tokens, t_send, t_end``

What the builder does, and every choice is named in the curve file:

1. A forward is ONE line with ``chunks == 1`` (a folded line carries several
   forwards in one gpu-ms and is skipped, counted). The ranks of one forward
   (same second, same ``new_tokens`` / ``cached_depth``) collapse to their
   ``max`` gpu-ms (``--p-rank-agg``: ``max`` = the slowest stage, ``pp0`` =
   PP0 / TP0 only, ``sum`` = the stages summed).
2. Points are bucketed by cached depth (``--depth-bucket`` tokens) and, inside
   a bucket, by new tokens (``--n-bucket``); a cell is the median.
3. D's points are its ``d_forward`` forwards; a ``d_extend`` point is added
   only where no ``d_forward`` forward of the same shape exists (the sweep's
   attribution of the same log lines, never counted twice).
4. The flip price is one D->P + P->D pair of USER flip times (``flip_user_ms``;
   D>P followed by the next P>D),
   attributed to a depth by the ``calib_prefix`` whose window holds the D>P
   flip's start, else by a ``depth`` field on the flip row, else depth-less
   (one price, ``attribution=none``, named as clamped on every lookup).
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from flliper.srt.pdflip import x_curves as xc

RANK_AGG = ("max", "pp0", "sum")
#: seconds of slack around a calib_prefix window when attributing a flip
FLIP_ATTRIBUTION_SLACK_S = 3.0


def read_rows(path: str) -> List[dict]:
    """The jsonl rows; a line that is not a JSON object is skipped (counted
    by the caller through the returned length vs the file)."""
    out: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _is_lead_rank(rank: str) -> bool:
    return str(rank) in ("PP0", "TP0")


def forward_points(rows: Iterable[dict], role: str, rank_agg: str
                   ) -> Tuple[List[Tuple[int, int, float]], int]:
    """``[(cached_depth, new_tokens, ms)]`` of one role's forwards (ranks
    collapsed), and the number of folded lines skipped."""
    groups: Dict[Tuple[int, int, int], List[Tuple[str, float]]] = collections.defaultdict(list)
    folded = 0
    for r in rows:
        if r.get("role") != role:
            continue
        if int(r.get("chunks", 1) or 1) != 1:
            folded += 1
            continue
        try:
            key = (int(float(r["t"])), int(r["cached_depth"]), int(r["new_tokens"]))
            ms = float(r["gpu_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if key[2] > 0 and ms > 0.0:
            groups[key].append((str(r.get("rank", "")), ms))
    pts: List[Tuple[int, int, float]] = []
    for (_t, depth, n), ranks in groups.items():
        if rank_agg == "pp0":
            lead = [ms for rk, ms in ranks if _is_lead_rank(rk)]
            if not lead:
                continue
            ms = lead[0]
        elif rank_agg == "sum":
            ms = sum(ms for _rk, ms in ranks)
        else:
            ms = max(ms for _rk, ms in ranks)
        pts.append((depth, n, ms))
    return pts, folded


def extend_points(rows: Iterable[dict], known: Iterable[Tuple[int, int, float]]
                  ) -> List[Tuple[int, int, float]]:
    """The sweep's D points (``d_extend``) whose shape no forward already has."""
    have = {(d, n) for d, n, _ms in known}
    out: List[Tuple[int, int, float]] = []
    for r in rows:
        if r.get("role") != "d_extend" or not r.get("d_direct") or r.get("ambiguous") or r.get("error"):
            continue
        d, n = r.get("cached_depth"), r.get("new_tokens")
        ms = r.get("gpu_ms_max") if r.get("gpu_ms_max") is not None else r.get("gpu_ms")
        if d is None or n is None or ms is None or int(n) <= 0 or float(ms) <= 0.0:
            continue
        if (int(d), int(n)) not in have:
            out.append((int(d), int(n), float(ms)))
    return out


def bucket_curve(points: Sequence[Tuple[int, int, float]], *, depth_bucket: int, n_bucket: int,
                 chunk_tokens: int, instrument: str,
                 depth_interp: str = xc.DEPTH_INTERP_LINEAR) -> xc.PrefillCurve:
    """Points -> rows: depth buckets, n buckets inside, medians."""
    if not points:
        raise xc.XCurvesRefused(xc.W_X_CURVES_MALFORMED, f"{instrument}: no points to build a curve from")
    by_depth: Dict[int, List[Tuple[int, int, float]]] = collections.defaultdict(list)
    for d, n, ms in points:
        by_depth[int(d) // max(1, depth_bucket)].append((d, n, ms))
    rows: List[xc.CurveRow] = []
    last_depth = -1
    for b in sorted(by_depth):
        pts = by_depth[b]
        depth = int(statistics.median(d for d, _n, _ms in pts))
        if depth <= last_depth:
            depth = last_depth + 1
        cells: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
        for _d, n, ms in pts:
            cells[int(round(n / max(1, n_bucket)))].append((n, ms))
        ns: List[int] = []
        mss: List[float] = []
        for c in sorted(cells):
            n = int(statistics.median(x for x, _ in cells[c]))
            if ns and n <= ns[-1]:
                continue
            ns.append(n)
            mss.append(float(statistics.median(m for _, m in cells[c])))
        rows.append(xc.CurveRow(depth=depth, n=tuple(ns), ms=tuple(mss)))
        last_depth = depth
    return xc.PrefillCurve(rows=tuple(rows), chunk_tokens=int(chunk_tokens), instrument=instrument,
                           depth_interp=depth_interp, points=len(points))


def _flip_ms(r: dict) -> Tuple[Optional[float], str]:
    """``(ms, basis)`` of one flip row. ``flip_user_ms`` is the only source; a row
    that carries the key but no positive number is a MISSING endpoint (``None``,
    ``negative`` = clock artefact), a row without the key is ``none``."""
    if "flip_user_ms" in r:
        v = r["flip_user_ms"]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None, "missing"
        return (fv, "flip_user_ms") if fv > 0.0 else (None, "missing")
    return None, "none"


def flip_pairs(rows: Sequence[dict], *, stats: Optional[Dict[str, int]] = None
               ) -> List[Tuple[float, float, Optional[int]]]:
    """``[(t_start_of_D>P, round_trip_s, row_depth)]``: each D>P flip with the
    next P>D flip after it, priced in USER flip time (``flip_user_ms``). A pair
    needs both directions; a missing endpoint drops the flip (and a D>P that has
    no P>D after it). ``stats`` counts the rows by basis (``flip_user_ms``,
    ``missing``, ``none``) and the priced pairs (``pairs_user``)."""
    st = stats if stats is not None else {}
    flips = sorted((r for r in rows if r.get("role") == "flip"), key=lambda r: float(r.get("t", 0.0)))
    out: List[Tuple[float, float, Optional[int]]] = []
    pending: Optional[Tuple[dict, float]] = None
    for r in flips:
        ms, basis = _flip_ms(r)
        st[basis] = st.get(basis, 0) + 1
        if ms is None:
            if r.get("direction") == "D>P":
                pending = None          # the D>P of this round trip is unpriced: no pair from it
            continue
        if r.get("direction") == "D>P":
            pending = (r, ms)
        elif r.get("direction") == "P>D" and pending is not None:
            prow, pms = pending
            t0 = float(prow["t_start"]) if prow.get("t_start") is not None else float(prow["t"]) - pms / 1000.0
            depth = prow.get("depth", r.get("depth"))
            out.append((t0, (pms + ms) / 1000.0, None if depth is None else int(depth)))
            st["pairs_user"] = st.get("pairs_user", 0) + 1
            pending = None
    return out


def _prefix_depth_at(t: float, prefixes: Sequence[dict]) -> Optional[int]:
    for p in prefixes:
        t0, t1 = p.get("t_send"), p.get("t_end") or p.get("t_emit")
        if t0 is None or t1 is None:
            continue
        if float(t0) - FLIP_ATTRIBUTION_SLACK_S <= t <= float(t1) + FLIP_ATTRIBUTION_SLACK_S:
            d = p.get("prompt_tokens") or p.get("depth_target")
            return None if d is None else int(d)
    return None


def build_flip_price(rows: Sequence[dict], *, depth_bucket: int,
                     notes: Optional[List[str]] = None) -> xc.FlipPrice:
    st: Dict[str, int] = {}
    pairs = flip_pairs(rows, stats=st)
    if not pairs:
        why = ""
        if st.get("none"):
            why += (f" ({st['none']} flip row(s) carry no flip_user_ms: write flip_user_ms per flip; "
                    f"Flipzeit = letztes Token abgehende Phase bis erstes Token ankommende Phase)")
        if st.get("missing"):
            why += f" ({st['missing']} flip row(s) have flip_user_ms null/non-positive: endpoint missing)"
        raise xc.XCurvesRefused(xc.W_X_CURVES_MALFORMED,
                                "no D>P + P>D flip pair of user flip times in the rows" + why)
    if notes is not None:
        notes.append(f"flip: user flip time (flip_user_ms): {st.get('pairs_user', 0)} pair(s); "
                     f"{st.get('missing', 0)} row(s) skipped (endpoint missing)"
                     + (f", {st['none']} row(s) skipped (no flip_user_ms)" if st.get("none") else ""))
    prefixes = [r for r in rows if r.get("role") == "calib_prefix" and not r.get("error")]
    placed: List[Tuple[int, float]] = []
    how = set()
    for t0, rt, row_depth in pairs:
        d = _prefix_depth_at(t0, prefixes)
        if d is not None:
            how.add("calib_prefix")
        elif row_depth is not None:
            d = row_depth
            how.add("row")
        if d is not None:
            placed.append((d, rt))
    if not placed:
        med = float(statistics.median(rt for _t, rt, _d in pairs))
        return xc.FlipPrice(depth=(0,), seconds=(med,),
                            attribution="none", pairs=len(pairs))
    by: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    for d, rt in placed:
        by[d // max(1, depth_bucket)].append((d, rt))
    depths: List[int] = []
    secs: List[float] = []
    for b in sorted(by):
        d = int(statistics.median(x for x, _ in by[b]))
        if depths and d <= depths[-1]:
            d = depths[-1] + 1
        depths.append(d)
        secs.append(float(statistics.median(s for _, s in by[b])))
    return xc.FlipPrice(depth=tuple(depths), seconds=tuple(secs),
                        attribution="+".join(sorted(how)),
                        pairs=len(placed))


def build(rows: Sequence[dict], *, model: str, form: str, hardware: str, source: str,
          p_chunk_tokens: int, d_chunk_tokens: int, depth_bucket: int = 8192,
          n_bucket: int = 256, rank_agg: str = "max",
          depth_interp: str = xc.DEPTH_INTERP_LINEAR, created: Optional[str] = None
          ) -> Tuple[xc.XCurves, List[str]]:
    """The curve file of ``rows`` and the builder's notes (what it skipped)."""
    if rank_agg not in RANK_AGG:
        raise xc.XCurvesRefused(xc.W_X_CURVES_MALFORMED, f"--p-rank-agg {rank_agg!r} not in {RANK_AGG}")
    p_pts, p_folded = forward_points(rows, "p_chunk", rank_agg)
    d_pts, d_folded = forward_points(rows, "d_forward", rank_agg)
    d_ext = extend_points(rows, d_pts)
    agg = {"max": "max over ranks", "pp0": "lead rank", "sum": "sum over ranks"}[rank_agg]
    p_curve = bucket_curve(p_pts, depth_bucket=depth_bucket, n_bucket=n_bucket,
                           chunk_tokens=p_chunk_tokens, depth_interp=depth_interp,
                           instrument=f"P Prefill rank batch gpu_ms, {agg}, chunks==1")
    d_curve = bucket_curve(d_pts + d_ext, depth_bucket=depth_bucket, n_bucket=n_bucket,
                           chunk_tokens=d_chunk_tokens, depth_interp=depth_interp,
                           instrument=f"D Prefill rank batch gpu_ms, {agg}, chunks==1 (+{len(d_ext)} d_extend)")
    flip_notes: List[str] = []
    flip = build_flip_price(rows, depth_bucket=depth_bucket, notes=flip_notes)
    ident = xc.CurveIdentity(model=model, form=form, hardware=hardware,
                             created=created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             source=source)
    curves = xc.XCurves(format=xc.X_CURVES_FORMAT, identity=ident, p_curve=p_curve,
                        d_curve=d_curve, flip_price=flip)
    xc.validate(curves)
    notes = [f"P: {len(p_pts)} forwards, {p_folded} folded lines skipped",
             f"D: {len(d_pts)} forwards + {len(d_ext)} d_extend points, {d_folded} folded lines skipped",
             f"flip: {flip.pairs} attributed pair(s), attribution={flip.attribution}"] + flip_notes
    return curves, notes


def _source_of(rows: Sequence[dict]) -> str:
    tags = sorted({str(r["tag"]) for r in rows if r.get("tag")})
    return ",".join(tags) or "untagged"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", help="calib_x_<Boot>.jsonl file(s)")
    ap.add_argument("--out", required=True, help="the curve file to write (JSON, weg2-x-curves/1)")
    ap.add_argument("--model", required=True, help="checkpoint key (the model directory name)")
    ap.add_argument("--form", required=True,
                    help="residue axes of the PDFLIP-FORM line: arch=..,experts=..,draft=..,kv=..,flip=..")
    ap.add_argument("--hardware", required=True, help="ordered card class labels, e.g. RTX5090,RTX3080,RTX3080")
    ap.add_argument("--source", default=None, help="boot tag(s); default: the rows' tag field")
    ap.add_argument("--p-chunk-tokens", type=int, required=True, help="P's --chunked-prefill-size of the boot")
    ap.add_argument("--d-chunk-tokens", type=int, required=True, help="D's --chunked-prefill-size of the boot")
    ap.add_argument("--depth-bucket", type=int, default=8192)
    ap.add_argument("--n-bucket", type=int, default=256)
    ap.add_argument("--p-rank-agg", choices=RANK_AGG, default="max")
    ap.add_argument("--depth-interp", choices=(xc.DEPTH_INTERP_LINEAR, xc.DEPTH_INTERP_LOG),
                    default=xc.DEPTH_INTERP_LINEAR)
    ap.add_argument("--cap", type=int, default=None, help="the text check's X cap (e.g. 12288)")
    a = ap.parse_args(argv)
    rows: List[dict] = []
    for path in a.jsonl:
        rows += read_rows(path)
    try:
        curves, notes = build(rows, model=a.model, form=a.form, hardware=a.hardware,
                              source=a.source or _source_of(rows), p_chunk_tokens=a.p_chunk_tokens,
                              d_chunk_tokens=a.d_chunk_tokens, depth_bucket=a.depth_bucket,
                              n_bucket=a.n_bucket, rank_agg=a.p_rank_agg, depth_interp=a.depth_interp)
    except xc.XCurvesRefused as e:
        print(f"X-CURVES BUILD REFUSED {e}", file=sys.stderr)
        return 2
    with open(a.out, "wb") as f:
        f.write(xc.encode(curves))
    for n in notes:
        print(f"X-CURVES BUILD {n}")
        if n.startswith("WARNING"):
            print(f"X-CURVES BUILD {n}", file=sys.stderr)
    print(xc.text_table(curves, cap=a.cap))
    print(f"X-CURVES BUILD wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
