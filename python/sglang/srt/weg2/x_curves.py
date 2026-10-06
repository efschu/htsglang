"""X-CURVES (06.10., NF seat, user go ~10:15Z): X from profile curves, not a
fixed or a live-solved X.

User decisions (binding, AUFTRAG-x-kurven-1006):

1. The fixed X as a standing profile value goes, in NF and 27B (flip forms).
   No X rework for Dual (both phases always awake; dual profiles carry no X).
2. NOT computed live. Once per model x form x hardware two curves are measured
   -- P's prefill and D's prefill over the whole context -- plus the flip price
   per depth as a third table. From them the front sets X PER REQUEST: at the
   request's pending-token context depth, does the flip pay or not; more
   requests open at the same time make the flip pay more.
3. One curve file per model x form x hardware; ONE profile line points at it.
4. (CHANGE 06.10. ~11:00Z, replaces the first decision 4) ``--x-mode`` has
   ONLY ``fixed`` and ``curve`` -- "es soll kein live geben und kein
   curve-capped". Without ``--x-mode`` the front behaves exactly as before
   the flag (not changed silently; see :data:`X_MODE_LIVE`).
5. (NF seat, user decision 06.10.) ``--x-ceiling-tokens`` MAY be set beside
   ``--x-mode curve``: the manual ceiling clamps X from above, the curve
   decides below it; D's W50 riegel stays on the LOWER of the curves'
   envelope and the manual ceiling (never below the start X, H84). Without
   the ceiling everything is as before (riegel = envelope).

This module is PURE (msgspec + stdlib): the schema, the loader with its named
refusals, the identity check, the interpolation, :func:`x_from_curves` and the
launcher's resolution of the mode. The front holds the loaded curves and asks
:func:`x_verdict_from_curves` once per arrival; nothing here keeps state.

THE MODEL. Both curves are tables of ONE prefill forward (one chunk, as the
``Prefill rank batch`` line logs it): ``(cached_depth, new_tokens) -> ms``.
A request of ``n`` new tokens at cached depth ``d`` costs the sum over its
chunks of ``chunk_tokens`` (each at its own depth ``d + i*chunk``). The flip
price is the measured D->P + P->D round trip in seconds, keyed by the context
of the request that rode it (``d + n``). The break-even of one request:

    D(d, n)  <=  price(d + n) / k  +  P(d, n)

``k`` = the requests one flip carries (the arrival + those queued for P now,
the X-K-FLIP amortisation). X is the FIRST crossing over the D curve's
measured ``n`` range: up to X the request is cheaper on D. More open requests
share the round trip, so the flip pays more and X never grows with ``k``.

Extrapolation is never silent: a lookup outside the measured range is clamped
to the edge and NAMED in the verdict's ``notes``; a request deeper than the
curves reach is refused by name (``beyond="refuse"``) or clamped to the
deepest row with a note (``beyond="clamp"``).
"""

from __future__ import annotations

import bisect
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import msgspec

# ---------------------------------------------------------------------------
# modes, format, refusal codes
# ---------------------------------------------------------------------------

#: NOT a ``--x-mode`` value: the name of the state WITHOUT ``--x-mode`` --
#: the front exactly as before the flag (the live re-solve wherever
#: SGLANG_WEG2_ENABLE_X_COST_LINE / the r_D-r_P-flip samples drive it). The
#: user removed ``live`` as a selectable mode (06.10. ~11:00Z); what the
#: unflagged default should become is left to the user, not changed here.
X_MODE_LIVE = "live"
#: ``--x-mode fixed``: ``--tp-prefill-max-tokens`` for the whole boot, no
#: live samples, no re-solve, no hysteresis.
X_MODE_FIXED = "fixed"
#: ``--x-mode curve``: X per request from ``--x-curves`` (bounded by the
#: curves' measured range and by D's W50 riegel, which the launcher sizes to
#: the curves' envelope).
X_MODE_CURVE = "curve"
#: The ``--x-mode`` values (user 06.10. ~11:00Z: only fixed and curve).
X_MODES: Tuple[str, ...] = (X_MODE_FIXED, X_MODE_CURVE)
CURVE_MODES: Tuple[str, ...] = (X_MODE_CURVE,)

#: ``--x-curves-beyond``: a request deeper than the curves reach.
BEYOND_CLAMP = "clamp"
BEYOND_REFUSE = "refuse"
BEYOND_POLICIES: Tuple[str, ...] = (BEYOND_CLAMP, BEYOND_REFUSE)

#: The curve file's format tag; a file of another format is refused (W191).
X_CURVES_FORMAT = "weg2-x-curves/1"

#: curve* mode without a curve file, or the file is absent / unreadable.
W_X_CURVES_MISSING = "W190 Weg2XCurvesMissing"
#: the file does not decode to the schema or fails its validation.
W_X_CURVES_MALFORMED = "W191 Weg2XCurvesMalformed"
#: the file was measured on another model, form or hardware.
W_X_CURVES_FOREIGN = "W192 Weg2XCurvesForeign"
#: a request deeper than the curves reach, under ``--x-curves-beyond refuse``.
W_X_CURVE_DEPTH_BEYOND = "W193 Weg2XCurveDepthBeyond"
#: ``--x-curves`` / ``--x-curves-beyond`` given while the mode reads no curve.
W_X_CURVES_WITHOUT_MODE = "W194 Weg2XCurvesWithoutCurveMode"
#: an unknown ``--x-mode`` (incl. the removed ``live`` / ``curve-capped``)
#: or ``--x-curves-beyond`` value.
W_X_MODE_UNKNOWN = "W196 Weg2XModeUnknown"

#: 27B PORT (06.10.): any X-curve flag in the DUAL layout. Both groups stay awake
#: there, nothing flips, X is the fixed dual allowance (1 + --dual-d-prefill-tokens);
#: user decision "kein X-Umbau fuer Dual", so the flags are refused, never ignored.
W_X_MODE_IN_DUAL = "W197 Weg2XModeInDualLayout"

REFUSAL_CODES: Tuple[str, ...] = (
    W_X_CURVES_MISSING, W_X_CURVES_MALFORMED, W_X_CURVES_FOREIGN, W_X_CURVE_DEPTH_BEYOND,
    W_X_CURVES_WITHOUT_MODE, W_X_MODE_UNKNOWN, W_X_MODE_IN_DUAL,
)

DEPTH_INTERP_LINEAR = "linear"
DEPTH_INTERP_LOG = "log"


class XCurvesRefused(RuntimeError):
    """A named refusal of the curve path; ``code`` is one of :data:`REFUSAL_CODES`."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# schema (msgspec, frozen)
# ---------------------------------------------------------------------------


class CurveIdentity(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Who the curves were measured on. ``model`` = the checkpoint key (its
    directory name, form.model_key); ``form`` = the residue axes of the WEG2
    form (``arch=..,experts=..,draft=..,kv=..,flip=..``, the X-COST-LINE /
    PARK-RT record key); ``hardware`` = the ordered card class labels
    (``RTX5090,RTX3080,RTX3080``, card_identity.inventory_signature)."""

    model: str
    form: str
    hardware: str
    created: str
    source: str


class CurveRow(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One cached depth: ``ms[i]`` is one forward of ``n[i]`` new tokens."""

    depth: int
    n: Tuple[int, ...]
    ms: Tuple[float, ...]


class PrefillCurve(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One group's prefill-forward table over (cached depth, new tokens).

    ``chunk_tokens`` > 0: a request is summed chunk by chunk (each chunk at
    its own depth); 0: a row prices a whole request in one forward.
    ``instrument`` names what ``ms`` is (e.g. ``gpu_ms max over ranks``)."""

    rows: Tuple[CurveRow, ...]
    chunk_tokens: int
    instrument: str
    depth_interp: str = DEPTH_INTERP_LINEAR
    points: int = 0


class FlipPrice(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The flip round trip (D->P + P->D) in seconds by request context depth.
    ``attribution`` names how a flip got its depth (``calib_prefix`` /
    ``row`` / ``none`` = one depth-less price)."""

    depth: Tuple[int, ...]
    seconds: Tuple[float, ...]
    attribution: str
    pairs: int = 0


class XCurves(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    format: str
    identity: CurveIdentity
    p_curve: PrefillCurve
    d_curve: PrefillCurve
    flip_price: FlipPrice


class XCurveVerdict(msgspec.Struct, frozen=True):
    """One request's X and everything the ROUTE-VERDICT line prints."""

    x: int
    x_curve: int
    depth: int
    depth_used: int
    k: float
    why: str
    clamp: str
    price_s: float
    notes: Tuple[str, ...]


# ---------------------------------------------------------------------------
# validation and loading
# ---------------------------------------------------------------------------


def _finite_pos(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0


def _strictly_increasing(xs: Sequence[float]) -> bool:
    return all(b > a for a, b in zip(xs, xs[1:]))


def _curve_problems(name: str, curve: PrefillCurve) -> List[str]:
    out: List[str] = []
    if not curve.rows:
        return [f"{name}: no rows"]
    if curve.chunk_tokens < 0:
        out.append(f"{name}: chunk_tokens {curve.chunk_tokens} < 0")
    if curve.depth_interp not in (DEPTH_INTERP_LINEAR, DEPTH_INTERP_LOG):
        out.append(f"{name}: depth_interp {curve.depth_interp!r} not linear|log")
    depths = [r.depth for r in curve.rows]
    if depths[0] < 0 or not _strictly_increasing(depths):
        out.append(f"{name}: row depths {depths} not >= 0 and strictly increasing")
    for r in curve.rows:
        if not r.n or len(r.n) != len(r.ms):
            out.append(f"{name} depth {r.depth}: n/ms empty or of unequal length")
        elif r.n[0] <= 0 or not _strictly_increasing(r.n):
            out.append(f"{name} depth {r.depth}: n {list(r.n)} not > 0 and strictly increasing")
        elif not all(_finite_pos(m) for m in r.ms):
            out.append(f"{name} depth {r.depth}: ms {list(r.ms)} not all finite > 0")
    return out


def validate(curves: XCurves) -> None:
    """Refuse (W191) a file whose content cannot price a request."""
    problems: List[str] = []
    if curves.format != X_CURVES_FORMAT:
        problems.append(f"format {curves.format!r} != {X_CURVES_FORMAT!r}")
    ident = curves.identity
    for field in ("model", "form", "hardware"):
        if not str(getattr(ident, field)).strip():
            problems.append(f"identity.{field} empty")
    problems += _curve_problems("p_curve", curves.p_curve)
    problems += _curve_problems("d_curve", curves.d_curve)
    fp = curves.flip_price
    if not fp.depth or len(fp.depth) != len(fp.seconds):
        problems.append("flip_price: depth/seconds empty or of unequal length")
    elif fp.depth[0] < 0 or not _strictly_increasing(fp.depth):
        problems.append(f"flip_price: depths {list(fp.depth)} not >= 0 and strictly increasing")
    elif not all(_finite_pos(s) for s in fp.seconds):
        problems.append(f"flip_price: seconds {list(fp.seconds)} not all finite > 0")
    if problems:
        raise XCurvesRefused(W_X_CURVES_MALFORMED, "; ".join(problems))


def decode(data: bytes, *, path: str = "<bytes>") -> XCurves:
    try:
        curves = msgspec.json.decode(data, type=XCurves)
    except (msgspec.ValidationError, msgspec.DecodeError) as e:
        raise XCurvesRefused(W_X_CURVES_MALFORMED, f"{path}: {e}") from e
    validate(curves)
    return curves


def encode(curves: XCurves) -> bytes:
    validate(curves)
    return msgspec.json.format(msgspec.json.encode(curves), indent=1)


def load(path: Optional[str]) -> XCurves:
    """The curve file at ``path``; absent / unreadable = W190, bad = W191."""
    if not path:
        raise XCurvesRefused(W_X_CURVES_MISSING, "no --x-curves file given")
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise XCurvesRefused(W_X_CURVES_MISSING, f"{path}: {type(e).__name__}: {e}") from e
    return decode(data, path=path)


def form_axes_of(boot_form) -> str:
    """The residue-axes string of a ``form.Weg2Form`` (the curve's ``form``)."""
    from sglang.srt.weg2 import form as _form

    return ",".join(f"{a}={getattr(boot_form, a)}" for a in _form.RESIDUE_AXES)


def split_form_key(form_key: str) -> Tuple[str, str]:
    """``model|axes`` (front.park_form_key) -> ``(model, axes)``; "" -> ("", "")."""
    model, _, axes = str(form_key or "").partition("|")
    return model, axes


def check_identity(curves: XCurves, *, model: str, form: str,
                   hardware: Optional[str]) -> str:
    """Refuse (W192) curves of another model, form or hardware; return the
    one-span description of what was checked. ``hardware=None`` = the caller
    cannot see the cards (the front: the launcher checked them) -- named."""
    ident = curves.identity
    wrong: List[str] = []
    if not model or not form:
        raise XCurvesRefused(W_X_CURVES_FOREIGN, (
            f"this boot publishes no model/form identity (model={model!r} form={form!r}), "
            f"so curves of model={ident.model} form={ident.form} cannot be shown to be its own"))
    if ident.model != model:
        wrong.append(f"model {ident.model!r} != boot {model!r}")
    if ident.form != form:
        wrong.append(f"form {ident.form!r} != boot {form!r}")
    if hardware is not None and ident.hardware != hardware:
        wrong.append(f"hardware {ident.hardware!r} != boot {hardware!r}")
    if wrong:
        raise XCurvesRefused(W_X_CURVES_FOREIGN, (
            "; ".join(wrong) + f" (curves source={ident.source} created={ident.created}). "
            "A curve file is measured once per model x form x hardware and prices no other"))
    hw = ident.hardware if hardware is not None else f"{ident.hardware} (checked by the launcher)"
    return f"model={ident.model} form={ident.form} hardware={hw}"


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------


def _row_ms(row: CurveRow, n: float, notes: Set[str], tag: str) -> float:
    """Piecewise linear in ``n``. Above the row's range: the edge value
    (named ``-n-edge``). Below it (the tail piece of a chunked request):
    the first segment's line, never below the proportional share of the
    first point (a forward's fixed cost is >= 0), or the edge value of a
    one-point row -- named ``-n-floor`` either way."""
    ns, ms = row.n, row.ms
    if n <= ns[0]:
        if n == ns[0]:
            return float(ms[0])
        notes.add(f"{tag}-n-floor")
        if len(ns) < 2:
            return float(ms[0])
        slope = (float(ms[1]) - float(ms[0])) / float(ns[1] - ns[0])
        return max(float(ms[0]) - slope * (ns[0] - n), float(ms[0]) * n / float(ns[0]))
    if n >= ns[-1]:
        if n > ns[-1]:
            notes.add(f"{tag}-n-edge")
        return float(ms[-1])
    i = bisect.bisect_right(ns, n)
    t = (n - ns[i - 1]) / float(ns[i] - ns[i - 1])
    return float(ms[i - 1]) + t * float(ms[i] - ms[i - 1])


def _bracket(curve: PrefillCurve, depth: float, notes: Set[str],
             tag: str) -> Tuple[CurveRow, CurveRow, float]:
    """The two rows around ``depth`` and the blend weight of the upper one;
    outside the rows the edge row, named (``{tag}-depth-floor|edge``)."""
    rows = curve.rows
    if depth <= rows[0].depth:
        if depth < rows[0].depth:
            notes.add(f"{tag}-depth-floor")
        return rows[0], rows[0], 0.0
    if depth >= rows[-1].depth:
        if depth > rows[-1].depth:
            notes.add(f"{tag}-depth-edge")
        return rows[-1], rows[-1], 0.0
    i = bisect.bisect_right([r.depth for r in rows], depth)
    lo, hi = rows[i - 1], rows[i]
    if curve.depth_interp == DEPTH_INTERP_LOG:
        w = (math.log1p(depth) - math.log1p(lo.depth)) / (math.log1p(hi.depth) - math.log1p(lo.depth))
    else:
        w = (depth - lo.depth) / float(hi.depth - lo.depth)
    return lo, hi, w


def forward_ms(curve: PrefillCurve, *, depth: float, n: float,
               notes: Optional[Set[str]] = None, tag: str = "") -> float:
    """One forward of ``n`` new tokens at cached ``depth``, in ms."""
    sink: Set[str] = set() if notes is None else notes
    lo, hi, w = _bracket(curve, depth, sink, tag)
    a = _row_ms(lo, n, sink, tag)
    if w == 0.0:
        return a
    return (1.0 - w) * a + w * _row_ms(hi, n, sink, tag)


def request_ms(curve: PrefillCurve, *, depth: float, n: float,
               notes: Optional[Set[str]] = None, tag: str = "") -> float:
    """A whole request of ``n`` new tokens at cached ``depth``: the sum over
    its chunks (each at its own depth); one forward when ``chunk_tokens`` is 0."""
    c = int(curve.chunk_tokens)
    if n <= 0:
        return 0.0
    if c <= 0:
        return forward_ms(curve, depth=depth, n=n, notes=notes, tag=tag)
    total, done = 0.0, 0.0
    while done < n:
        piece = min(float(c), n - done)
        total += forward_ms(curve, depth=depth + done, n=piece, notes=notes, tag=tag)
        done += piece
    return total


def flip_seconds(price: FlipPrice, *, depth: float, notes: Optional[Set[str]] = None) -> float:
    """The round trip at context ``depth``: piecewise linear, edges named."""
    sink: Set[str] = set() if notes is None else notes
    ds, ss = price.depth, price.seconds
    if depth <= ds[0]:
        if depth < ds[0] and len(ds) > 1:
            sink.add("flip-depth-floor")
        return float(ss[0])
    if depth >= ds[-1]:
        if depth > ds[-1]:
            sink.add("flip-depth-edge" if len(ds) > 1 else "flip-depthless")
        return float(ss[-1])
    i = bisect.bisect_right(ds, depth)
    t = (depth - ds[i - 1]) / float(ds[i] - ds[i - 1])
    return float(ss[i - 1]) + t * float(ss[i] - ss[i - 1])


def reach_depth(curves: XCurves) -> int:
    """The deepest cached depth both prefill curves were measured at."""
    return int(min(curves.d_curve.rows[-1].depth, curves.p_curve.rows[-1].depth))


# ---------------------------------------------------------------------------
# X per request
# ---------------------------------------------------------------------------


def _d_domain(curve: PrefillCurve, depth: float, reach: int) -> Tuple[int, int, Tuple[int, ...]]:
    """The ``n`` range D's curve prices at ``depth`` without extrapolating, and
    the breakpoints of its first chunk. One forward: the bracketing rows'
    common ``n`` range. Chunked: as many chunks as START within ``reach``
    (each priced at its own measured depth), the last one up to the rows'
    ``n`` edge."""
    lo, hi, _w = _bracket(curve, depth, set(), "")
    n_lo = max(lo.n[0], hi.n[0])
    n_hi = max(n_lo, min(lo.n[-1], hi.n[-1]))
    c = int(curve.chunk_tokens)
    if c > 0:
        n_hi = max(n_lo, int(max(0.0, reach - depth) // c) * c + min(n_hi, c))
    return int(n_lo), int(n_hi), tuple(lo.n) + tuple(hi.n)


def _candidates(curves: XCurves, depth: float, n_lo: int, n_hi: int,
                d_points: Iterable[int]) -> List[int]:
    """Every ``n`` where the break-even's terms change slope or jump: D's
    breakpoints, P's chunk edges and breakpoints (per chunk depth) and the flip
    price's depths -- between two of them the difference is linear."""
    cand: Set[int] = {n_lo, n_hi} | set(int(n) for n in d_points)
    for curve in (curves.d_curve, curves.p_curve):
        c = int(curve.chunk_tokens)
        for i in range(0, (n_hi // c + 1) if c > 0 else 1):
            base = i * c
            lo, hi, _w = _bracket(curve, depth + base, set(), "")
            cand.update(base + int(n) for n in lo.n + hi.n)
            cand.update((base, base + 1))
    cand.update(int(d - depth) for d in curves.flip_price.depth)
    return sorted(n for n in cand if n_lo <= n <= n_hi)


def _excess_ms(curves: XCurves, depth: float, n: float, k: float, notes: Set[str]) -> float:
    """D's cost of the request minus its cost through the flip, in ms: <= 0
    means D is the cheaper route (the request is SHORT)."""
    d_ms = request_ms(curves.d_curve, depth=depth, n=n, notes=notes, tag="d")
    p_ms = request_ms(curves.p_curve, depth=depth, n=n, notes=notes, tag="p")
    price_ms = 1000.0 * flip_seconds(curves.flip_price, depth=depth + n, notes=notes) / k
    return d_ms - p_ms - price_ms


def _first_crossing(curves: XCurves, depth: float, k: float, notes: Set[str]
                    ) -> Tuple[float, str, str]:
    """``(X, why, clamp)``: the first ``n`` of D's measured range where D
    becomes dearer than the flip, linear between candidates."""
    n_lo, n_hi, d_pts = _d_domain(curves.d_curve, depth, reach_depth(curves))
    cand = _candidates(curves, depth, n_lo, n_hi, d_pts)
    prev_n, prev_g = None, None
    for n in cand:
        g = _excess_ms(curves, depth, float(n), k, notes)
        if g > 0.0:
            if prev_n is None:
                return float(n_lo), "p-cheaper-at-n_lo", "n_lo"
            frac = -prev_g / (g - prev_g) if g != prev_g else 0.0
            return prev_n + frac * (n - prev_n), "crossing", "none"
        prev_n, prev_g = float(n), g
    return float(n_hi), "d-cheaper-to-n_hi", "n_hi"


def x_verdict_from_curves(*, depth: int, open_requests: int, curves: XCurves,
                          cap: Optional[int] = None, max_k: Optional[float] = None,
                          beyond: str = BEYOND_REFUSE) -> XCurveVerdict:
    """The X of one request at cached ``depth`` with ``open_requests`` riding
    one flip (>= 1), stateless. ``cap`` bounds X from above (the ceiling /
    D's riegel), ``max_k`` bounds the amortisation (a P phase's request cap).
    A depth beyond :func:`reach_depth` is refused (W193) or, with
    ``beyond="clamp"``, priced at the deepest row and named."""
    if beyond not in BEYOND_POLICIES:
        raise XCurvesRefused(W_X_MODE_UNKNOWN, f"--x-curves-beyond {beyond!r} not in {BEYOND_POLICIES}")
    notes: Set[str] = set()
    reach = reach_depth(curves)
    depth_used = max(0, int(depth))
    if depth_used > reach:
        if beyond == BEYOND_REFUSE:
            raise XCurvesRefused(W_X_CURVE_DEPTH_BEYOND, (
                f"cached depth {depth_used} > {reach}, the deepest depth both prefill curves "
                f"were measured at (source={curves.identity.source}); a deeper request is "
                f"refused by --x-curves-beyond refuse, never extrapolated"))
        notes.add(f"depth-clamped({depth_used}->{reach})")
        depth_used = reach
    k = max(1.0, float(open_requests))
    if max_k is not None and max_k >= 1.0:
        k = min(k, float(max_k))
    x_star, why, clamp = _first_crossing(curves, float(depth_used), k, notes)
    x_curve = int(math.floor(x_star + 1e-6))  # a crossing ON a token is that token (float noise)
    x = x_curve
    if cap is not None and cap > 0 and x > int(cap):
        x, clamp = int(cap), ("cap" if clamp == "none" else f"{clamp}+cap")
    price_s = flip_seconds(curves.flip_price, depth=float(depth_used + max(x, 0)))
    return XCurveVerdict(x=x, x_curve=x_curve, depth=int(depth), depth_used=depth_used,
                         k=k, why=why, clamp=clamp, price_s=price_s,
                         notes=tuple(sorted(notes)))


def x_from_curves(*, depth: int, open_requests: int, curves: XCurves,
                  cap: Optional[int] = None, max_k: Optional[float] = None,
                  beyond: str = BEYOND_REFUSE) -> int:
    """B (AUFTRAG-x-kurven-1006): the X of one request, an int (see
    :func:`x_verdict_from_curves`)."""
    return x_verdict_from_curves(depth=depth, open_requests=open_requests, curves=curves,
                                 cap=cap, max_k=max_k, beyond=beyond).x


def x_envelope(curves: XCurves, *, cap: Optional[int] = None) -> int:
    """The lone request's best X (k=1) over every measured D depth, capped:
    the boot-level upper bound every per-request X stays under -- the front's
    ``tp_prefill_max_tokens`` in a curve mode (the role X_1 has under X-K-FLIP),
    and what D's W50 riegel must admit."""
    best = 0
    for row in curves.d_curve.rows:
        if row.depth > reach_depth(curves):
            continue
        v = x_verdict_from_curves(depth=row.depth, open_requests=1, curves=curves,
                                  cap=cap, beyond=BEYOND_CLAMP)
        best = max(best, v.x)
    return int(best)


# ---------------------------------------------------------------------------
# launcher / front plumbing (pure)
# ---------------------------------------------------------------------------


class XModeLaunch(msgspec.Struct, frozen=True):
    """The launcher's resolution of ``--x-mode`` (see :func:`resolve_launch`)."""

    mode: str
    given: bool
    ceiling_for_d: int
    front_argv: Tuple[str, ...]
    line: str
    provenance: str
    #: appended to the X CEILING line under --x-mode (no "live X" there)
    ceiling_note: str = ""


def refuse_in_dual(*, mode: Optional[str], curves_path: Optional[str],
                   beyond: Optional[str]) -> None:
    """W197: the dual layout takes no X-curve flag (user: no X rework for Dual;
    dual profiles carry no X flags). Called by the launcher and by the front's
    main() with ``--dual-layout`` on."""
    given = [f"{flag} {val}" for flag, val in (("--x-mode", mode), ("--x-curves", curves_path),
                                               ("--x-curves-beyond", beyond)) if val is not None]
    if given:
        raise XCurvesRefused(W_X_MODE_IN_DUAL, (
            f"{', '.join(given)} in the --dual-layout: both groups stay awake, nothing flips and X "
            f"is the fixed dual allowance (1 + --dual-d-prefill-tokens), so there is no X for a "
            f"curve to set. Drop the X-curve flags (user decision 06.10.: no X rework for Dual)"))


def _check_mode_words(mode: Optional[str], curves_path: Optional[str],
                      beyond: Optional[str], ceiling: int) -> str:
    m = X_MODE_LIVE if mode is None else str(mode)
    if mode is not None and m not in X_MODES:
        raise XCurvesRefused(W_X_MODE_UNKNOWN, (
            f"--x-mode {m!r} not in {X_MODES} (user 06.10.: only fixed and curve -- "
            f"live and curve-capped were removed)"))
    if beyond is not None and beyond not in BEYOND_POLICIES:
        raise XCurvesRefused(W_X_MODE_UNKNOWN, f"--x-curves-beyond {beyond!r} not in {BEYOND_POLICIES}")
    if m not in CURVE_MODES and (curves_path or beyond is not None):
        raise XCurvesRefused(W_X_CURVES_WITHOUT_MODE, (
            f"--x-curves {curves_path or '(unset)'} / --x-curves-beyond {beyond or '(unset)'} "
            f"under --x-mode {m}: that mode reads no curve, so the file would be ignored "
            f"silently. Use --x-mode curve, or drop the curve flags"))
    if m in CURVE_MODES and not curves_path:
        raise XCurvesRefused(W_X_CURVES_MISSING, f"--x-mode {m} without --x-curves <file>")
    return m


def front_argv(*, mode: Optional[str], curves_path: Optional[str],
               beyond: Optional[str]) -> List[str]:
    """The front's flags, each only when given (no flag = argv byte-identical)."""
    argv: List[str] = []
    if mode is not None:
        argv += ["--x-mode", str(mode)]
    if curves_path:
        argv += ["--x-curves", os.path.abspath(str(curves_path))]
    if beyond is not None:
        argv += ["--x-curves-beyond", str(beyond)]
    return argv


def resolve_launch(*, mode: Optional[str], curves_path: Optional[str], beyond: Optional[str],
                   ceiling_flag: Optional[int], x_tokens: int, model: str, form: str,
                   hardware: str, dual: bool = False) -> XModeLaunch:
    """The launcher's half: check the words, load and identity-check the
    curves (model, form AND the live cards), and say which number D's W50
    riegel needs. Raises :class:`XCurvesRefused` by name.

    ``ceiling_for_d`` is what the launcher hands :func:`resolve_x_ceiling`:
    the flag unchanged, except under ``curve`` where D's riegel is the
    curves' envelope under the manual ceiling (user 06.10.: the LOWER of the
    two; the ceiling clamps X from above, the curve decides below it)."""
    if dual:
        refuse_in_dual(mode=mode, curves_path=curves_path, beyond=beyond)
    ceiling = int(ceiling_flag or 0)
    m = _check_mode_words(mode, curves_path, beyond, ceiling)
    argv = tuple(front_argv(mode=mode, curves_path=curves_path, beyond=beyond))
    if mode is None:
        line = "X MODE: none (no --x-mode: the front's X exactly as before the flag)"
        return XModeLaunch(m, False, ceiling, argv, line, "")
    if m == X_MODE_FIXED:
        line = (f"X MODE: --x-mode fixed -- X={int(x_tokens)} (--tp-prefill-max-tokens) for the whole "
                f"boot: no live X samples, no re-solve, no hysteresis")
        return XModeLaunch(m, True, ceiling, argv, line, "; x-mode=fixed (no live re-solve)",
                           " [x-mode=fixed: no live re-solve -- the front's X stays the start X]")
    curves = load(curves_path)
    ident = check_identity(curves, model=model, form=form, hardware=hardware)
    env_free = x_envelope(curves)
    cap = ceiling if ceiling > 0 else None
    env = x_envelope(curves, cap=cap)
    ceiling_for_d = env
    if cap is None:
        cap_txt = "no manual ceiling (--x-ceiling-tokens unset)"
    else:
        cap_txt = (f"manual ceiling --x-ceiling-tokens {ceiling} "
                   + ("BINDS" if ceiling < env_free else "does not bind")
                   + f" (curves' own envelope {env_free})")
    line = (f"X MODE: --x-mode {m} --x-curves {os.path.abspath(str(curves_path))} "
            f"--x-curves-beyond {beyond or BEYOND_CLAMP} -- {describe(curves)} {ident}; X PER "
            f"REQUEST from the D/P prefill curves and the flip price, k = 1 + requests queued for "
            f"P; {cap_txt}; envelope X={env} (lone request, max over the measured depths, under the "
            f"ceiling); D's W50 riegel = {max(int(x_tokens), ceiling_for_d)} (the LOWER of envelope "
            f"and ceiling, never below the start X): the hard bound no per-request X crosses; no "
            f"live X samples, no re-solve")
    prov = (f"; x-mode={m} curves={os.path.basename(str(curves_path))} "
            f"source={curves.identity.source} envelope X={env}"
            + ("" if cap is None else f" ceiling={ceiling}")
            + " (the start X above is the X-SOLO band floor only)")
    note = (f" [x-mode={m}: no live re-solve -- 'the live X' above is the per-request curve X; "
            + ("this ceiling is the curves' envelope, set by the launcher, not a flag]" if cap is None
               else f"this ceiling is min(envelope {env_free}, --x-ceiling-tokens {ceiling})]"))
    return XModeLaunch(m, True, int(ceiling_for_d), argv, line, prov, note)


def describe(curves: XCurves) -> str:
    """One span naming the file's content and identity (the X MODE line)."""
    d, p, fp = curves.d_curve, curves.p_curve, curves.flip_price
    return (f"[{curves.format} source={curves.identity.source} created={curves.identity.created} "
            f"D rows={len(d.rows)} depth {d.rows[0].depth}..{d.rows[-1].depth} chunk={d.chunk_tokens} "
            f"({d.instrument}); P rows={len(p.rows)} depth {p.rows[0].depth}..{p.rows[-1].depth} "
            f"chunk={p.chunk_tokens} ({p.instrument}); flip price {len(fp.depth)} depth(s) "
            f"{min(fp.seconds):.2f}..{max(fp.seconds):.2f}s attribution={fp.attribution}; "
            f"reach depth {reach_depth(curves)}]")


def verdict_note(*, mode: str, verdict: XCurveVerdict, source: str,
                 cap: Optional[int] = None) -> str:
    """The ROUTE-VERDICT suffix of a curve-mode arrival (instrument text
    names the mode, the ceiling in force, the curve source and THIS
    request's X)."""
    notes = ",".join(verdict.notes) or "none"
    return (f"; x_mode={mode} X_req={verdict.x} (curve {verdict.x_curve}, {verdict.why}, "
            f"clamp={verdict.clamp}, ceiling={cap if cap else 'none'}) depth={verdict.depth}"
            + (f"->{verdict.depth_used}" if verdict.depth_used != verdict.depth else "")
            + f" k={verdict.k:g} price={verdict.price_s:.2f}s notes={notes} curves={source}")


def fixed_note() -> str:
    """The ROUTE-VERDICT suffix under ``--x-mode fixed``."""
    return "; x_mode=fixed (no live re-solve)"


def text_table(curves: XCurves, *, depths: Optional[Sequence[int]] = None,
               ks: Sequence[int] = (1, 2, 4), cap: Optional[int] = None) -> str:
    """The builder's plot check as text: every row of both curves (ms and
    ms/token), the flip price, and X by depth x k."""
    lines: List[str] = [describe(curves)]
    for name, curve in (("D", curves.d_curve), ("P", curves.p_curve)):
        lines.append(f"{name} curve ({curve.instrument}, chunk {curve.chunk_tokens}):")
        for r in curve.rows:
            cells = " ".join(f"{n}:{m:.1f}ms({m / n:.3f}/tok)" for n, m in zip(r.n, r.ms))
            mono = "" if all(b >= a for a, b in zip(r.ms, r.ms[1:])) else "  NON-MONOTONE"
            lines.append(f"  depth {r.depth:>7}: {cells}{mono}")
    fp = curves.flip_price
    lines.append("flip price: " + " ".join(f"{d}:{s:.2f}s" for d, s in zip(fp.depth, fp.seconds))
                 + f" (attribution={fp.attribution}, pairs={fp.pairs})")
    ds = list(depths) if depths is not None else [r.depth for r in curves.d_curve.rows]
    lines.append("X by depth x k" + (f" (cap {cap})" if cap else "") + ":")
    for d in ds:
        cells: List[str] = []
        for k in ks:
            v = x_verdict_from_curves(depth=d, open_requests=k, curves=curves, cap=cap,
                                      beyond=BEYOND_CLAMP)
            cells.append(f"k={k}:{v.x}({v.why}{'' if not v.notes else ' ' + ','.join(v.notes)})")
        lines.append(f"  depth {d:>7}: " + "  ".join(cells))
    return "\n".join(lines)


def state_block(*, mode: str, source: str, envelope: int,
                last: Optional[XCurveVerdict]) -> Dict[str, object]:
    """``/weg2/state`` keys under ``--x-mode`` (the dashboard's interface)."""
    out: Dict[str, object] = {"x_mode": mode}
    if mode in CURVE_MODES:
        out["x_curves_source"] = source
        out["x_curves_envelope"] = int(envelope)
        if last is not None:
            out["x_curves_last"] = {"x": last.x, "depth": last.depth, "k": last.k,
                                    "why": last.why, "clamp": last.clamp}
    return out


class XFrontSetup(msgspec.Struct, frozen=True):
    """The front's resolution of its X mode (see :func:`front_setup`)."""

    mode: str
    curves: Optional[XCurves]
    source: str
    beyond: str
    cap: Optional[int]
    envelope: int
    line: str


def front_setup(*, mode: Optional[str], curves_path: Optional[str], beyond: Optional[str],
                form_key: str, x_start: int, x_ceiling: int) -> XFrontSetup:
    """The front's half: the same word checks as the launcher, the curves
    loaded and checked against the form this boot published (model + form;
    the cards were checked by the launcher). In a curve mode the cap is
    ``x_ceiling`` (D's W50 riegel, H84: no X above what D admits; the
    launcher sets it to the curves' envelope) and the boot-level X is the
    envelope. ``line`` is "" when no ``--x-mode`` was given (the front's log
    stays byte-identical)."""
    m = _check_mode_words(mode, curves_path, beyond, 0)
    pol = beyond or BEYOND_CLAMP
    if m not in CURVE_MODES:
        line = ("" if mode is None else
                f"WEG2 X MODE {m}: X={int(x_start)} for the whole boot, no live samples, "
                f"no re-solve, no hysteresis")
        return XFrontSetup(m, None, "", pol, None, int(x_start), line)
    curves = load(curves_path)
    model, form = split_form_key(form_key)
    ident = check_identity(curves, model=model, form=form, hardware=None)
    cap = int(x_ceiling) if int(x_ceiling) > 0 else None
    env = max(1, x_envelope(curves, cap=cap))
    source = f"{os.path.basename(str(curves_path))}@{curves.identity.source}"
    line = (f"WEG2 X MODE {m}: X PER REQUEST from {os.path.abspath(str(curves_path))} "
            f"{describe(curves)} {ident}; cap={cap} (D's W50 riegel: --x-ceiling-tokens as the "
            f"launcher sets it = min(envelope, manual ceiling), else the start X) envelope X={env} (the boot-level X: lone request, max over the measured "
            f"depths); beyond the reach: {pol}; no live X samples, no re-solve")
    return XFrontSetup(m, curves, source, pol, cap, env, line)
