"""Per-LAYER-TYPE cost model of a pipelined prefill group (27B line, 26.09.).

User orders 26.09. ~05:15Z: "der eigentliche Hebel ist jetzt der
Schichtschnitt" and "cuda graphen im prefill nur anschalten wenn es was
bringt"; ~05:20Z (NVFP4): more layers belong on the 5090. This module is the
model both questions are answered on, and the one the chunk-to-chunk moving
cut (desk/27b-dynsplit-0926, DYN_LAYER_SPLIT.md) reads. PURE: stdlib plus
``p_chunk_policy``, every number handed in as JSON, no sglang import.

THE MODEL. One forward of ``w`` tokens at prefix ``p`` on a stage that holds
``n`` layers, ``a`` of them full attention, on card ``c``, executed in mode
``m`` (``graph`` = a captured prefill CUDA graph, ``eager``):

    T = n * gemm_ms[c][m](w)                      prefix-INDEPENDENT part of
                                                  every layer (GEMMs, GDN
                                                  recurrence, norms)
      + a * attn[c][m](w) * w * (p + w/2) / 1e6   the full-attention kernels,
                                                  quadratic in the context
      + first_fixed[c][m](w)  (stage 0 only)      embedding / stage input
      + last_fixed[c][m](w)   (last stage only)   final norm, logits, outputs

``gemm_ms`` is ms PER LAYER PER FORWARD at width ``w`` (piecewise linear in
``w`` through its points, proportional below the first, the last segment
extended above); ``attn`` is ms per attention layer per unit of
``attn_work = w * (p + w/2) / 1e6`` (ms per Mtok^2; divide by 1000 for the
``attn_ms_per_tok_1k`` unit of :class:`p_chunk_policy.StageModel`), piecewise
linear in ``w`` and FLAT beyond its points (kernel efficiency saturates; it
is not extrapolated).

WHAT IS MEASURED AND WHAT IS ASSUMED (rc9j #PGAP, see the JSON sources):
  * the attention coefficient is format-independent to 3 % (INT8 and NVFP4
    agree on both cards: 5090 0.28 graph-512 / 0.17-0.18 eager-2048, 3080
    0.67-0.71 at every width) -- the attention kernels run on bf16/fp8 KV;
  * ASSUMPTION, stated: an attention layer's prefix-independent part costs
    the same as a GDN layer's on the same card (one cut per format measured,
    so the two cannot be separated; FLOPs differ by ~3 %);
  * a card that carries only ONE stage folds that stage's fixed part into its
    per-layer rate (the 5090 carries the embedding that way); a card with a
    middle stage takes its rate from it and books the remainder of its first
    or last stage as fixed (``card_costs_from_fit``).

CONTIGUITY. A PP stage owns a contiguous layer range, so its attention count
is FIXED by its boundaries (``attn_counts``): on the 27B (full attention every
4th layer) 52 layers on stage 0 carry 13 attention layers, never 8. A cut
"more layers but fewer attention layers on the 5090" does not exist as a
contiguous map (gapped maps are refused at runtime as numerically wrong).
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import p_chunk_policy as _pcp

MODE_GRAPH = "graph"
MODE_EAGER = "eager"
MODES = (MODE_GRAPH, MODE_EAGER)
FULL_ATTENTION = "full_attention"

Curve = Tuple[Tuple[int, float], ...]


class StageModelError(ValueError):
    """An invalid layer-type model, cut or fit input."""


def _curve(raw) -> Curve:
    pts = tuple((int(w), float(v)) for w, v in (raw or ()))
    ws = [w for w, _ in pts]
    if any(w <= 0 for w in ws) or ws != sorted(ws) or len(set(ws)) != len(ws):
        raise StageModelError(f"curve widths must be ascending, unique, > 0: {pts}")
    return pts


def curve_linear(pts: Curve, w: int) -> float:
    """Piecewise linear; proportional below the first point, last segment extended."""
    if not pts:
        return 0.0
    if len(pts) == 1 or w <= pts[0][0]:
        w0, v0 = pts[0]
        if len(pts) == 1 or w < w0:
            return v0 * float(w) / float(w0)
        return v0
    if w >= pts[-1][0]:
        (w0, v0), (w1, v1) = pts[-2], pts[-1]
    else:
        i = 1
        while pts[i][0] < w:
            i += 1
        (w0, v0), (w1, v1) = pts[i - 1], pts[i]
    return max(0.0, v0 + (v1 - v0) * (float(w) - w0) / float(w1 - w0))


def curve_flat(pts: Curve, w: int) -> float:
    """Piecewise linear, FLAT beyond both ends."""
    if not pts:
        return 0.0
    if w <= pts[0][0]:
        return pts[0][1]
    if w >= pts[-1][0]:
        return pts[-1][1]
    i = 1
    while pts[i][0] < w:
        i += 1
    (w0, v0), (w1, v1) = pts[i - 1], pts[i]
    return v0 + (v1 - v0) * (float(w) - w0) / float(w1 - w0)


def attn_work(w: int, prefix: int) -> float:
    """``w * (prefix + w/2) / 1e6``: the attention work unit of one chunk."""
    return float(w) * (float(prefix) + 0.5 * float(w)) / 1.0e6


# ---------------------------------------------------------------------------
# the card and the format model


@dataclasses.dataclass(frozen=True)
class CardCost:
    """One card class under one weight format (see the module docstring).

    Every field maps a mode (``graph`` / ``eager``) to a width curve. A mode
    that is absent falls back to the other (a missing measurement is priced
    as the measured mode, and :meth:`has` says which were measured)."""

    name: str
    gemm_ms: Dict[str, Curve]
    attn: Dict[str, Curve]
    first_fixed_ms: Dict[str, Curve] = dataclasses.field(default_factory=dict)
    last_fixed_ms: Dict[str, Curve] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        for fld in ("gemm_ms", "attn", "first_fixed_ms", "last_fixed_ms"):
            d = {str(k): _curve(v) for k, v in (getattr(self, fld) or {}).items()}
            bad = [k for k in d if k not in MODES]
            if bad:
                raise StageModelError(f"card {self.name}: {fld} modes {bad} not in {MODES}")
            object.__setattr__(self, fld, d)
        if not self.gemm_ms or not self.attn:
            raise StageModelError(f"card {self.name}: gemm_ms and attn need at least one mode")

    @staticmethod
    def _pick(d: Dict[str, Curve], mode: str) -> Curve:
        if mode in d and d[mode]:
            return d[mode]
        other = MODE_EAGER if mode == MODE_GRAPH else MODE_GRAPH
        return d.get(other, ())

    def has(self, mode: str) -> bool:
        return bool(self.gemm_ms.get(mode)) and bool(self.attn.get(mode))

    def gemm(self, w: int, mode: str) -> float:
        return curve_linear(self._pick(self.gemm_ms, mode), w)

    def attn_coeff(self, w: int, mode: str) -> float:
        return curve_flat(self._pick(self.attn, mode), w)

    def first_fixed(self, w: int, mode: str) -> float:
        return curve_linear(self._pick(self.first_fixed_ms, mode), w)

    def last_fixed(self, w: int, mode: str) -> float:
        return curve_linear(self._pick(self.last_fixed_ms, mode), w)

    def to_json(self) -> Dict[str, object]:
        out: Dict[str, object] = {"name": self.name}
        for fld in ("gemm_ms", "attn", "first_fixed_ms", "last_fixed_ms"):
            d = getattr(self, fld)
            if d:
                out[fld] = {k: [list(p) for p in v] for k, v in sorted(d.items())}
        return out

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "CardCost":
        return cls(str(d["name"]), dict(d.get("gemm_ms", {})), dict(d.get("attn", {})),
                   dict(d.get("first_fixed_ms", {}) or {}), dict(d.get("last_fixed_ms", {}) or {}))


def layer_types_every(n_layers: int, period: int = 4, offset: int = 3) -> Tuple[str, ...]:
    """The 27B layout: full attention at ``i % period == offset`` (3, 7, ..., 63)."""
    return tuple(FULL_ATTENTION if i % period == offset else "linear_attention" for i in range(int(n_layers)))


@dataclasses.dataclass(frozen=True)
class LayerCostModel:
    """A format's model: card classes, the stage-to-card order, the layer layout."""

    cards: Dict[str, CardCost]
    stage_cards: Tuple[str, ...]
    layer_types: Tuple[str, ...]
    source: str = ""
    eager_floor_ms: Tuple[float, ...] = ()

    def __post_init__(self):
        cards = {str(k): (v if isinstance(v, CardCost) else CardCost.from_json(v)) for k, v in self.cards.items()}
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "stage_cards", tuple(str(c) for c in self.stage_cards))
        object.__setattr__(self, "layer_types", tuple(str(t) for t in self.layer_types))
        object.__setattr__(self, "eager_floor_ms", tuple(float(x) for x in (self.eager_floor_ms or ())))
        missing = [c for c in self.stage_cards if c not in cards]
        if missing:
            raise StageModelError(f"stage cards {missing} have no CardCost")
        if not self.stage_cards or not self.layer_types:
            raise StageModelError("need at least one stage and one layer")
        if self.eager_floor_ms and len(self.eager_floor_ms) != len(self.stage_cards):
            raise StageModelError("eager_floor_ms needs one value per stage")

    @property
    def n_stages(self) -> int:
        return len(self.stage_cards)

    @property
    def n_layers(self) -> int:
        return len(self.layer_types)

    def attn_counts(self, counts: Sequence[int]) -> Tuple[int, ...]:
        """Full-attention layers per stage of the CONTIGUOUS cut ``counts``."""
        counts = [int(c) for c in counts]
        if len(counts) != self.n_stages or sum(counts) != self.n_layers or min(counts) < 1:
            raise StageModelError(
                f"cut {counts}: need {self.n_stages} stages of >= 1 layer summing to {self.n_layers}")
        out, lo = [], 0
        for c in counts:
            out.append(sum(1 for t in self.layer_types[lo:lo + c] if t == FULL_ATTENTION))
            lo += c
        return tuple(out)

    def stage_ms(self, counts: Sequence[int], stage: int, w: int, prefix: int, mode: str,
                 attn: Optional[Sequence[int]] = None) -> float:
        """One forward of stage ``stage`` of the cut ``counts`` (ms)."""
        a = (attn if attn is not None else self.attn_counts(counts))[stage]
        card = self.cards[self.stage_cards[stage]]
        t = int(counts[stage]) * card.gemm(w, mode) + int(a) * card.attn_coeff(w, mode) * attn_work(w, prefix)
        if stage == 0:
            t += card.first_fixed(w, mode)
        if stage == self.n_stages - 1:
            t += card.last_fixed(w, mode)
        if mode == MODE_EAGER and self.eager_floor_ms:
            t = max(t, self.eager_floor_ms[stage])
        return t

    def stage_models(self, counts: Sequence[int], widths: Sequence[int] = (256, 512, 1024, 2048, 4096),
                     ) -> Tuple[_pcp.StageModel, ...]:
        """The cut as :class:`p_chunk_policy.StageModel` s -- graph and eager
        priced apart, the attention coefficient per executed width -- so the
        chunk plan runs on exactly this model."""
        attn = self.attn_counts(counts)
        ws = sorted({int(w) for w in widths})
        out = []
        for r in range(self.n_stages):
            card = self.cards[self.stage_cards[r]]

            def base(w, mode, r=r):
                t = int(counts[r]) * card.gemm(w, mode)
                if r == 0:
                    t += card.first_fixed(w, mode)
                if r == self.n_stages - 1:
                    t += card.last_fixed(w, mode)
                return round(t, 4)

            a = attn[r]
            out.append(_pcp.StageModel(
                points=tuple((w, base(w, MODE_GRAPH)) for w in ws),
                attn_ms_per_tok_1k=round(a * card.attn_coeff(ws[0], MODE_GRAPH) / 1000.0, 9),
                eager_floor_ms=self.eager_floor_ms[r] if self.eager_floor_ms else 0.0,
                name=f"PP{r} {card.name} n={counts[r]} a={a}",
                eager_points=tuple((w, base(w, MODE_EAGER)) for w in ws),
                attn_points=tuple((w, round(a * card.attn_coeff(w, MODE_GRAPH) / 1000.0, 9)) for w in ws),
                eager_attn_points=tuple((w, round(a * card.attn_coeff(w, MODE_EAGER) / 1000.0, 9)) for w in ws),
            ))
        return tuple(out)

    def to_json(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "cards": {k: v.to_json() for k, v in sorted(self.cards.items())},
            "stage_cards": list(self.stage_cards),
            "layer_types": list(self.layer_types),
            "source": self.source,
        }
        if self.eager_floor_ms:
            d["eager_floor_ms"] = list(self.eager_floor_ms)
        return d

    @classmethod
    def from_json(cls, d) -> "LayerCostModel":
        if isinstance(d, str):
            d = json.loads(d)
        lt = d.get("layer_types")
        if not lt:
            spec = d.get("layer_layout") or {}
            lt = layer_types_every(int(spec["n_layers"]), int(spec.get("period", 4)), int(spec.get("offset", 3)))
        return cls({k: CardCost.from_json(v) for k, v in d["cards"].items()}, tuple(d["stage_cards"]),
                   tuple(lt), str(d.get("source", "")), tuple(d.get("eager_floor_ms", ()) or ()))


def load_model(path: str) -> LayerCostModel:
    with open(path) as fh:
        return LayerCostModel.from_json(json.load(fh))


# ---------------------------------------------------------------------------
# cuts and their evaluation


def contiguous_cuts(n_layers: int, n_stages: int, min_per_stage: int = 1) -> List[Tuple[int, ...]]:
    """Every contiguous cut (compositions of ``n_layers`` into ``n_stages``)."""
    out: List[Tuple[int, ...]] = []

    def rec(prefix: List[int], left: int, k: int):
        if k == 1:
            if left >= min_per_stage:
                out.append(tuple(prefix + [left]))
            return
        for c in range(min_per_stage, left - min_per_stage * (k - 1) + 1):
            rec(prefix + [c], left - c, k - 1)

    rec([], int(n_layers), int(n_stages))
    return out


@dataclasses.dataclass(frozen=True)
class RungResult:
    tokens: int
    chunks: Tuple[int, ...]
    makespan_ms: float
    stage_busy_ms: Tuple[float, ...]
    candidate: str


def plan_for(model_stages: Sequence[_pcp.StageModel], limits: _pcp.ChunkLimits, tokens: int
             ) -> Tuple[Tuple[int, ...], str]:
    """The chunk plan the runtime would execute for a fresh request of ``tokens``:
    the fixed width under the bypass, else the dynamic plan on this model."""
    if limits.dynamic_min_tokens and tokens <= limits.dynamic_min_tokens:
        f = limits.fixed_tokens
        return tuple([f] * (tokens // f) + ([tokens % f] if tokens % f else [])), "bypass"
    res = _pcp.plan_detail(tokens, len(model_stages), model_stages, limits)
    return res.chunks, res.candidate


def busy_ms(chunks: Sequence[int], stages: Sequence[_pcp.StageModel], limits: _pcp.ChunkLimits,
            start: int = 0) -> Tuple[float, ...]:
    out = []
    for st in stages:
        p, b = int(start), 0.0
        for c in chunks:
            m_exec, eager = limits.exec_shape(int(c))
            b += st.forward_ms(m_exec, int(c), p, eager)
            p += int(c)
        out.append(b)
    return tuple(out)


def evaluate_cut(model: LayerCostModel, counts: Sequence[int], limits: _pcp.ChunkLimits,
                 tokens: Sequence[int], chunks_override: Optional[Dict[int, Sequence[int]]] = None,
                 ) -> Tuple[RungResult, ...]:
    """Every rung of ``tokens`` on the cut, each with the plan the dynamic
    policy picks ON THIS CUT's model (or ``chunks_override[n]``)."""
    stages = model.stage_models(counts)
    out = []
    for n in tokens:
        if chunks_override and n in chunks_override:
            ch, cand = tuple(int(c) for c in chunks_override[n]), "override"
        else:
            ch, cand = plan_for(stages, limits, int(n))
        out.append(RungResult(int(n), ch, _pcp.makespan_ms(ch, 0, stages, limits),
                              busy_ms(ch, stages, limits), cand))
    return tuple(out)


def mix_score(results: Sequence[RungResult], weights: Dict[int, float],
              reference: Optional[Dict[int, float]] = None) -> float:
    """``sum_n w_n * T_n / ref_n`` (``ref`` = 1 when absent): with a reference
    every rung counts by its RELATIVE time, so a 128k rung does not outvote
    a 2k rung by its length alone."""
    s = 0.0
    for r in results:
        w = float(weights.get(r.tokens, 0.0))
        ref = float(reference.get(r.tokens, 1.0)) if reference else 1.0
        s += w * r.makespan_ms / ref
    return s


def drifting_cut_makespan(model: LayerCostModel, cuts: Sequence[Sequence[int]], chunks: Sequence[int],
                          limits: _pcp.ChunkLimits, start: int = 0,
                          stage_weights: Optional[Sequence[float]] = None,
                          ) -> Tuple[float, List[Tuple[int, ...]]]:
    """A cut that moves from chunk to chunk, priced on the same flow shop:
    every chunk runs on the cut (from ``cuts``) that minimises its slowest
    stage -- or, with ``stage_weights`` (lambda), its ``sum_r lambda_r T_r``
    (the Lagrangian of "minimise the busiest stage's total", which is what a
    pipeline's makespan follows; see :func:`best_drifting_makespan`). Moving
    layers/KV/state costs NOTHING here (DYN_LAYER_SPLIT prices the BAR1
    traffic; this is the ceiling it works against). Returns the makespan and
    the cut per chunk."""
    S = model.n_stages
    k = limits.max_inflight or S
    attn = {tuple(c): model.attn_counts(c) for c in cuts}
    fin = [0.0] * S
    last_done: List[float] = []
    chosen: List[Tuple[int, ...]] = []
    p = int(start)
    for i, c in enumerate(chunks):
        m_exec, eager = limits.exec_shape(int(c))
        mode = MODE_EAGER if eager else MODE_GRAPH
        best, best_t, best_v = None, None, None
        for cut in cuts:
            cut = tuple(cut)
            ts = [model.stage_ms(cut, r, m_exec, p, mode, attn[cut]) for r in range(S)]
            v = max(ts) if stage_weights is None else sum(w * t for w, t in zip(stage_weights, ts))
            if best_v is None or v < best_v - 1e-9:
                best, best_t, best_v = cut, ts, v
        chosen.append(best)
        ready = last_done[i - k] if i >= k else 0.0
        prev = ready
        for r in range(S):
            f = max(fin[r], prev) + best_t[r]
            fin[r] = f
            prev = f
        last_done.append(prev)
        p += int(c)
    return (fin[-1] if S else 0.0), chosen


def best_drifting_makespan(model: LayerCostModel, cuts: Sequence[Sequence[int]], chunks: Sequence[int],
                           limits: _pcp.ChunkLimits, start: int = 0, grid: int = 20,
                           ) -> Tuple[float, List[Tuple[int, ...]], Optional[Tuple[float, ...]]]:
    """The best :func:`drifting_cut_makespan` over the per-chunk min-max rule
    and a ``grid`` of stage weights on the simplex (step ``1/grid``) -- a
    cheap, deterministic stand-in for the min-max assignment; never worse than
    the best STATIC cut in ``cuts`` (a static cut is one lambda's answer at
    most, so it is added as a candidate explicitly)."""
    S = model.n_stages
    best_ms, best_cuts = drifting_cut_makespan(model, cuts, chunks, limits, start)
    best_w: Optional[Tuple[float, ...]] = None
    for cut in cuts:
        ms, cs = drifting_cut_makespan(model, [cut], chunks, limits, start)
        if ms < best_ms - 1e-9:
            best_ms, best_cuts, best_w = ms, cs, None
    if S >= 2 and grid > 0:
        def simplex(k, left):
            if k == 1:
                yield (left,)
                return
            for i in range(left + 1):
                for rest in simplex(k - 1, left - i):
                    yield (i,) + rest
        for ws in simplex(S, int(grid)):
            if min(ws) == 0 and max(ws) == grid:
                continue
            lam = tuple(float(x) / grid for x in ws)
            ms, cs = drifting_cut_makespan(model, cuts, chunks, limits, start, lam)
            if ms < best_ms - 1e-9:
                best_ms, best_cuts, best_w = ms, cs, lam
    return best_ms, best_cuts, best_w


# ---------------------------------------------------------------------------
# fitting from #PGAP (lines in, numbers out -- no file access here)

_ADMIT_RE = re.compile(
    r"\b(PP\d+)\] #969N ADMIT slot=\S+ fwd_ct=(-?\d+) bs=(\d+) extend=(\d+) input_ids=\S+ rids=\[([^\]]*)\]")
_PGAP_RE = re.compile(
    r"\b(PP\d+)\] #PGAP pp_rank=\S+ fwd=(-?\d+) tokens=(\d+) gpu_gap_ms=(\S+) gpu_fwd_ms=([0-9.]+)")
_BUCKETS_RE = re.compile(r"PREFILL-GRAPH captured .*?buckets=\[([^\]]*)\]")


@dataclasses.dataclass(frozen=True)
class WidthSample:
    rank: int
    width: int
    prefix: int
    gpu_ms: float
    gap_ms: float
    bs: int
    mode: str


def samples_from_lines(lines: Iterable[str], graph_buckets: Optional[Sequence[int]] = None
                       ) -> List[WidthSample]:
    """Every #PGAP forward joined to its request prefix (the ``#969N ADMIT``
    pass with the same forward counter; the prefix is the sum of the rid's
    earlier extends -- a width-agnostic version of
    ``pgap_stage_fit.read_pgap_log``). The mode is ``graph`` when the width
    fits the largest captured bucket (the runner's rule), else ``eager``;
    the buckets are read from the rank line ``PREFILL-GRAPH captured`` unless
    given."""
    state: Dict[str, Tuple[str, int]] = {}
    adm: Dict[Tuple[str, int], Tuple[int, int, int]] = {}
    raw: List[Tuple[int, int, int, int, float, float]] = []
    buckets = list(graph_buckets) if graph_buckets is not None else None
    seen_buckets: List[int] = []
    for ln in lines:
        if "#969N ADMIT" in ln:
            m = _ADMIT_RE.search(ln)
            if not m:
                continue
            rk, fct, bs, ext, rid = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
            prid, acc = state.get(rk, ("", 0))
            if rid != prid:
                acc = 0
            adm[(rk, fct + 1)] = (acc, ext, bs)
            state[rk] = (rid, acc + ext)
        elif "#PGAP pp_rank" in ln:
            m = _PGAP_RE.search(ln)
            if not m:
                continue
            info = adm.get((m.group(1), int(m.group(2))))
            if not info or info[1] != int(m.group(3)):
                continue
            try:
                gap = float(m.group(4))
            except ValueError:
                gap = float("inf")
            raw.append((int(m.group(1)[2:]), int(m.group(3)), info[0], info[2], float(m.group(5)), gap))
        elif "PREFILL-GRAPH captured" in ln and not seen_buckets:
            m = _BUCKETS_RE.search(ln)
            if m:
                seen_buckets = [int(x) for x in m.group(1).split(",") if x.strip()]
    if buckets is None:
        buckets = seen_buckets
    top = max(buckets) if buckets else 0
    return [WidthSample(r, w, p, g, gap, bs, MODE_GRAPH if (top and w <= top) else MODE_EAGER)
            for (r, w, p, bs, g, gap) in raw]


@dataclasses.dataclass(frozen=True)
class WidthLine:
    """``gpu_ms = a + s * w * (prefix + w/2) / 1e6`` for one (rank, width, mode)."""

    a_ms: float
    s: float
    sd_ms: float
    n: int
    trimmed: int
    prefix_lo: int
    prefix_hi: int


def _lsq(xs, ys):
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    a = my - b * mx
    return a, b, math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / n)


def fit_width_lines(samples: Sequence[WidthSample], widths: Sequence[int] = (512, 1024, 2048, 4096),
                    min_samples: int = 8, trim_floor_ms: float = 3.0, trim_mads: float = 5.0,
                    ) -> Dict[Tuple[int, int, str], WidthLine]:
    """Robust line per (rank, width, mode) over the bs-1 FULL-width forwards."""
    groups: Dict[Tuple[int, int, str], List[WidthSample]] = {}
    for s in samples:
        if s.bs == 1 and s.width in widths:
            groups.setdefault((s.rank, s.width, s.mode), []).append(s)
    out: Dict[Tuple[int, int, str], WidthLine] = {}
    for key, v in sorted(groups.items()):
        if len(v) < min_samples:
            continue
        xs = [attn_work(s.width, s.prefix) for s in v]
        ys = [s.gpu_ms for s in v]
        if max(xs) - min(xs) <= 0:
            continue
        keep = list(range(len(v)))
        a, b, sd = _lsq(xs, ys)
        for _ in range(3):
            res = sorted(abs(ys[i] - a - b * xs[i]) for i in keep)
            cut = max(trim_floor_ms, trim_mads * 1.4826 * res[len(res) // 2])
            nk = [i for i in keep if abs(ys[i] - a - b * xs[i]) <= cut]
            if len(nk) == len(keep) or len(nk) < 5:
                break
            keep = nk
            a, b, sd = _lsq([xs[i] for i in keep], [ys[i] for i in keep])
        out[key] = WidthLine(a, b, sd, len(keep), len(v) - len(keep),
                             min(s.prefix for s in v), max(s.prefix for s in v))
    return out


def card_costs_from_fit(lines: Dict[Tuple[int, int, str], WidthLine], counts: Sequence[int],
                        attn: Sequence[int], stage_cards: Sequence[str]) -> Dict[str, CardCost]:
    """Per-card curves from ONE cut's width lines (the rule in the module
    docstring: a middle stage sets its card's per-layer rate, a single-stage
    card folds its fixed part in; attention = the stage slope / its attention
    layers, averaged over the card's stages)."""
    S = len(counts)
    out: Dict[str, CardCost] = {}
    keys = sorted({(w, m) for (_, w, m) in lines})
    for name in dict.fromkeys(stage_cards):
        stages = [r for r, c in enumerate(stage_cards) if c == name]
        middle = [r for r in stages if 0 < r < S - 1]
        gemm: Dict[str, List[Tuple[int, float]]] = {}
        att: Dict[str, List[Tuple[int, float]]] = {}
        first: Dict[str, List[Tuple[int, float]]] = {}
        last: Dict[str, List[Tuple[int, float]]] = {}
        for (w, mode) in keys:
            have = [r for r in stages if (r, w, mode) in lines]
            if len(have) != len(stages):
                continue
            if middle:
                rate = lines[(middle[0], w, mode)].a_ms / float(counts[middle[0]])
            else:
                rate = sum(lines[(r, w, mode)].a_ms for r in stages) / float(sum(counts[r] for r in stages))
            gemm.setdefault(mode, []).append((w, round(rate, 4)))
            slopes = [lines[(r, w, mode)].s / float(attn[r]) for r in stages if attn[r] > 0]
            if slopes:
                att.setdefault(mode, []).append((w, round(sum(slopes) / len(slopes), 5)))
            if middle:
                for r in stages:
                    rem = lines[(r, w, mode)].a_ms - rate * counts[r]
                    if r == 0:
                        first.setdefault(mode, []).append((w, round(rem, 3)))
                    if r == S - 1:
                        last.setdefault(mode, []).append((w, round(rem, 3)))
        if not gemm:
            raise StageModelError(f"card {name}: no width is measured on all of its stages {stages}")
        out[name] = CardCost(name, gemm, att or {m: [(w, 0.0) for w, _ in v] for m, v in gemm.items()},
                             first, last)
    return out


__all__ = [
    "MODE_GRAPH", "MODE_EAGER", "MODES", "FULL_ATTENTION", "StageModelError", "CardCost",
    "LayerCostModel", "RungResult", "WidthSample", "WidthLine", "curve_linear", "curve_flat",
    "attn_work", "layer_types_every", "load_model", "contiguous_cuts", "plan_for", "busy_ms",
    "evaluate_cut", "mix_score", "drifting_cut_makespan", "best_drifting_makespan", "samples_from_lines", "fit_width_lines",
    "card_costs_from_fit",
]


def pchunk_json(model: LayerCostModel, counts: Sequence[int]) -> Dict[str, object]:
    """A ``--p-chunk-model`` JSON ({"stages": [...]}) for the cut ``counts``."""
    return {"stages": [s.to_json() for s in model.stage_models(counts)],
            "source": f"p_stage_model cut {','.join(map(str, counts))} attn "
                      f"{','.join(map(str, model.attn_counts(counts)))} from: {model.source[:300]}"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m sglang.srt.weg2.p_stage_model MODEL.json --cut 41,12,11 [--out F]``:
    write the cut's ``--p-chunk-model`` JSON (stdout without --out)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("model")
    ap.add_argument("--cut", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    model = load_model(a.model)
    doc = pchunk_json(model, [int(x) for x in a.cut.split(",")])
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
