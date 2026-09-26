"""Prefill CUDA graph only where it pays (``--p-prefill-graph-policy``, 27B line).

User order 26.09. ~05:15Z: "cuda graphen im prefill nur anschalten wenn es was
bringt, wenn es negativ ist, ausschalten". PURE (stdlib + p_chunk_policy):
the launcher hands in the configured buckets and a CALIBRATION TABLE, this
module returns which buckets group P captures and the lines that say why.

THE TABLE (JSON, ``--p-prefill-graph-calibration PATH`` or the key
``graph_calibration`` of a ``--p-chunk-model`` JSON)::

    {"ref_prefix": 16384,
     "widths": {"512":  {"graph_ms": [g0, g1, g2], "eager_ms": [e0, e1, e2],
                         "graph_attn": [...], "eager_attn": [...]},   # optional
                "1024": {...}, "2048": {...}},
     "source": "graphcal boots ..."}

``graph_ms`` / ``eager_ms`` = the prefix-free cost of ONE forward of that
width per P stage (the ``a`` of the #PGAP line ``gpu_fwd = a + attn * w *
(prefix + w/2)/1000``), ``*_attn`` = the stage's attention coefficient in the
``p_chunk_policy.StageModel.attn_ms_per_tok_1k`` unit (optional; absent =
the same for both modes, i.e. it cancels).

THE RULE (``decide``). Per width ``w`` the stage times at ``ref_prefix``
(``T = ms + attn * w * (ref_prefix + w/2) / 1000``) are compared by their
SLOWEST STAGE -- a pipeline runs at its bottleneck, so a graph that speeds up
an idle stage and slows the bottleneck is a loss -- and the graph is kept
only when ``max_r T_graph < (1 - min_gain) * max_r T_eager``. A width with no
complete measurement keeps the SAFE DEFAULT: the main bucket (the
``--p-prefill-graph`` value, 512 on the 27B) captured, every extra bucket
eager. So a boot without a table captures exactly what it captured before
this switch (argv byte-identical).

MODES: ``auto`` (default) = the rule; ``on`` = capture every configured
bucket regardless (the metal calibration arm); ``off`` = the switch is not
consulted at all (exactly the pre-switch behaviour; extra buckets refused).
Extra buckets additionally have to pass the VRAM gate in the launcher (the P
cut's pool with their capture pool must still clear the pool floor).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import p_chunk_policy as _pcp

POLICY_AUTO = "auto"
POLICY_ON = "on"
POLICY_OFF = "off"
POLICIES = (POLICY_AUTO, POLICY_ON, POLICY_OFF)
DEFAULT_MIN_GAIN = 0.01
DEFAULT_REF_PREFIX = 16384
LOG_TAG = "P-PREFILL-GRAPH-POLICY"


class GraphPolicyError(ValueError):
    """An invalid calibration table or bucket set."""


@dataclasses.dataclass(frozen=True)
class WidthCal:
    graph_ms: Tuple[float, ...] = ()
    eager_ms: Tuple[float, ...] = ()
    graph_attn: Tuple[float, ...] = ()
    eager_attn: Tuple[float, ...] = ()

    def complete(self, stages: int) -> bool:
        return len(self.graph_ms) == stages and len(self.eager_ms) == stages


@dataclasses.dataclass(frozen=True)
class Calibration:
    widths: Dict[int, WidthCal]
    ref_prefix: int = DEFAULT_REF_PREFIX
    source: str = ""

    @classmethod
    def from_json(cls, d) -> "Calibration":
        if isinstance(d, str):
            d = json.loads(d)
        if not isinstance(d, dict) or not isinstance(d.get("widths"), dict):
            raise GraphPolicyError("graph calibration: a JSON object with 'widths' expected")
        out: Dict[int, WidthCal] = {}
        for k, v in d["widths"].items():
            try:
                w = int(k)
            except ValueError:
                raise GraphPolicyError(f"graph calibration: width key {k!r} is not an int")
            if w <= 0:
                raise GraphPolicyError(f"graph calibration: width {w} must be positive")
            vals = {}
            for fld in ("graph_ms", "eager_ms", "graph_attn", "eager_attn"):
                raw = (v or {}).get(fld, ()) or ()
                try:
                    t = tuple(float(x) for x in raw)
                except (TypeError, ValueError):
                    raise GraphPolicyError(f"graph calibration: width {w} {fld} is not a number list")
                if any(x < 0 for x in t):
                    raise GraphPolicyError(f"graph calibration: width {w} {fld} has a negative value")
                vals[fld] = t
            out[w] = WidthCal(**vals)
        ref = int(d.get("ref_prefix", DEFAULT_REF_PREFIX) or 0)
        if ref < 0:
            raise GraphPolicyError("graph calibration: ref_prefix must be >= 0")
        return cls(out, ref, str(d.get("source", "")))

    def to_json(self) -> Dict[str, object]:
        ws = {}
        for w, c in sorted(self.widths.items()):
            e = {"graph_ms": list(c.graph_ms), "eager_ms": list(c.eager_ms)}
            if c.graph_attn:
                e["graph_attn"] = list(c.graph_attn)
            if c.eager_attn:
                e["eager_attn"] = list(c.eager_attn)
            ws[str(w)] = e
        return {"ref_prefix": self.ref_prefix, "widths": ws, "source": self.source}


def load_calibration(path: str) -> Calibration:
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, dict) and "graph_calibration" in d and "widths" not in d:
        d = d["graph_calibration"]
    return Calibration.from_json(d)


def _stage_times(ms: Sequence[float], attn: Sequence[float], w: int, ref: int) -> List[float]:
    work = float(w) * (float(ref) + 0.5 * float(w)) / 1000.0
    return [float(m) + (float(attn[r]) if r < len(attn) else 0.0) * work for r, m in enumerate(ms)]


@dataclasses.dataclass(frozen=True)
class WidthVerdict:
    width: int
    graph: bool
    reason: str
    graph_max_ms: Optional[float] = None
    eager_max_ms: Optional[float] = None


def verdict(width: int, main_bucket: int, cal: Optional[Calibration], stages: int,
            min_gain: float = DEFAULT_MIN_GAIN) -> WidthVerdict:
    """The auto rule for one width (see the module docstring)."""
    c = cal.widths.get(int(width)) if cal is not None else None
    if c is None or not c.complete(stages):
        is_main = int(width) == int(main_bucket)
        return WidthVerdict(int(width), is_main,
                            "no calibration -> safe default (%s)" % ("main bucket: graph" if is_main else "extra bucket: eager"))
    ga = c.graph_attn or c.eager_attn
    ea = c.eager_attn or c.graph_attn
    tg = _stage_times(c.graph_ms, ga, width, cal.ref_prefix)
    te = _stage_times(c.eager_ms, ea, width, cal.ref_prefix)
    gmax, emax = max(tg), max(te)
    win = gmax < (1.0 - float(min_gain)) * emax
    return WidthVerdict(
        int(width), win,
        "calibrated at prefix %d: slowest stage graph %.1f ms vs eager %.1f ms (%+.1f%%) -> %s"
        % (cal.ref_prefix, gmax, emax, 100.0 * (1.0 - gmax / emax) if emax > 0 else 0.0,
           "graph" if win else "eager"),
        gmax, emax)


def decide(policy: str, main_bucket: int, tiny: Sequence[int], extras: Sequence[int],
           cal: Optional[Calibration], stages: int, min_gain: float = DEFAULT_MIN_GAIN,
           ) -> Tuple[Tuple[int, ...], Tuple[WidthVerdict, ...]]:
    """``(captured buckets ascending, per-width verdicts)``.

    ``off``: ``[main] + tiny`` (the pre-switch set; extras refused). ``on``:
    ``[main] + tiny + extras``. ``auto``: per width in ``[main] + extras``
    the rule; tiny buckets ride with the main bucket (they exist only to spare
    the main bucket's padding, so without it they are dropped too)."""
    if policy not in POLICIES:
        raise GraphPolicyError(f"--p-prefill-graph-policy {policy!r}: one of {list(POLICIES)}")
    main = int(main_bucket)
    extras = tuple(sorted({int(x) for x in extras}))
    tiny = tuple(sorted({int(x) for x in tiny}))
    if not main:
        if extras:
            raise GraphPolicyError("--p-prefill-graph-buckets needs --p-prefill-graph N (the main bucket)")
        return (), ()
    bad = [x for x in extras if x <= main]
    if bad:
        raise GraphPolicyError(
            f"--p-prefill-graph-buckets {list(extras)}: every extra bucket must be above the main "
            f"bucket {main} (smaller ones are --p-prefill-graph-tiny), got {bad}")
    if policy == POLICY_OFF:
        if extras:
            raise GraphPolicyError(
                "--p-prefill-graph-buckets with --p-prefill-graph-policy off: 'off' is the pre-switch "
                "form (one main bucket + tiny); use 'on' (forced) or 'auto' (calibrated)")
        return tuple(sorted(set(tiny) | {main})), ()
    if policy == POLICY_ON:
        vs = tuple(WidthVerdict(w, True, "policy on (forced)") for w in (main,) + extras)
        return tuple(sorted(set(tiny) | {main} | set(extras))), vs
    vs = tuple(verdict(w, main, cal, stages, min_gain) for w in (main,) + extras)
    keep = {v.width for v in vs if v.graph}
    if main in keep:
        keep |= set(tiny)
    return tuple(sorted(keep)), vs


def overlay_stage_models(stages: Sequence[_pcp.StageModel], cal: Optional[Calibration],
                         captured: Sequence[int]) -> Tuple[_pcp.StageModel, ...]:
    """The chunk policy's stage models with the calibrated graph / eager cost
    per width written in -- the SAME numbers :func:`decide` compared, so the
    plan prices each width in the mode it will run in. No table = unchanged
    (the spec stays byte-identical)."""
    if cal is None or not cal.widths:
        return tuple(stages)
    S = len(stages)
    out = []
    for r, st in enumerate(stages):
        ws = sorted(set(m for m, _ in st.points if m > 0) | set(cal.widths))
        g = {m: st.base_ms(m, False) for m in ws}
        e = {m: st.base_ms(m, True) for m in ws}
        ga = {m: st.attn_coeff(m, False) for m in ws}
        ea = {m: st.attn_coeff(m, True) for m in ws}
        for w, c in cal.widths.items():
            if len(c.graph_ms) == S:
                g[w] = c.graph_ms[r]
            if len(c.eager_ms) == S:
                e[w] = c.eager_ms[r]
            if len(c.graph_attn) == S:
                ga[w] = c.graph_attn[r]
            if len(c.eager_attn) == S:
                ea[w] = c.eager_attn[r]
        zero = [(0, st.base_ms(0, False))] if st.points and st.points[0][0] == 0 else []
        out.append(dataclasses.replace(
            st,
            points=tuple(zero + [(m, g[m]) for m in ws]),
            eager_points=tuple([(m, e[m]) for m in ws]),
            attn_points=tuple((m, ga[m]) for m in ws),
            eager_attn_points=tuple((m, ea[m]) for m in ws),
            name=(st.name + " +graphcal").strip(),
        ))
    return tuple(out)


def decision_lines(policy: str, captured: Sequence[int], verdicts: Sequence[WidthVerdict],
                   cal: Optional[Calibration]) -> List[str]:
    head = (f"{LOG_TAG} policy={policy} captured={list(captured) or 'none (P prefills eager)'} "
            f"calibration={(cal.source or 'given') if cal is not None else 'none'}")
    return [head] + [f"{LOG_TAG} width={v.width} -> {'graph' if v.graph else 'eager'}: {v.reason}"
                     for v in verdicts]


__all__ = [
    "POLICY_AUTO", "POLICY_ON", "POLICY_OFF", "POLICIES", "DEFAULT_MIN_GAIN", "LOG_TAG",
    "GraphPolicyError", "WidthCal", "Calibration", "WidthVerdict", "load_calibration", "verdict",
    "decide", "overlay_stage_models", "decision_lines",
]
