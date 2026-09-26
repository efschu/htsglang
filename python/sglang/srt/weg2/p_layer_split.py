"""Dynamic layer split of a pipelined prefill group (``--p-layer-split``).

User order 26.09. ~05:35Z: the stage boundaries of group P (27B: PP0 5090,
PP1/PP2 3080) move WITHIN a request, chunk by chunk, without new ranks or
processes; KV, GDN state and weights of the moving layers travel over BAR1;
non-standard cuts run eager. Design and evidence: /spinning/gpu-arb/docs/
DYN_LAYER_SPLIT.md. This module is the PURE core (stdlib only at import; torch
only inside the tensor helpers at the bottom, imported lazily), every model
number handed in, like ``p_chunk_policy``.

WHY THE CUT DRIFTS. The two layer families scale differently between the
cards: per 512 chunk the 5090 runs a GDN/MLP layer 3.1x (INT8) / 6.5x (NVFP4,
native FP4 against W4A8) faster than a 3080, but an attention layer's
prefix term only ~2.4-2.5x faster (rc9 #PGAP fits, pgap_stage_fit). The
attention share grows with the prefix, so the balanced cut moves PP0 DOWN as
the prefix grows (balanced cut over 0..128k: NVFP4 PP0 47 -> 41 layers,
INT8 41 -> 39; see DYN_LAYER_SPLIT.md).

THE DESIGN (HOME MIRROR, UPSTREAM TAKES). Every layer keeps ONE home rank:
the home cut (``SplitGeometry.home_cuts``). Everything persistent stays keyed
to home -- radix tree, req_to_token, allocator, HiCache arena extents, mamba
anchors, flip weight exchange, prefill graphs. A boundary may only move
DOWNSTREAM of its home position (``home_b <= cut_b <= home_b + window_b``):
stage s additionally EXECUTES the first layers of stage s+1's home span. For
those "swing" layers stage s keeps a MIRROR (KV rows in extra per-layer pool
buffers addressed by its OWN slot indices, GDN state in extra per-layer state
buffers addressed by its OWN mamba index, the layer's weights resident) and
WRITES BACK every chunk's new KV rows and the GDN state to home. The
write-back travels DOWNSTREAM with the chunk's proxy frame, so home holds a
complete copy before it runs, publishes or anchors that chunk -- no pending
ledger, no upstream wait.

A mirror is valid for a request only if the upstream stage executed the
layer for ALL of that request's chunks from position 0. Hence the two rules
(V1): a cut may rise above home only on a FRESH request's first forward
(start == 0: nothing to pull), and afterwards only FALL (swing-out: free,
home already holds everything). ``LeaderCursor`` tracks this per request as
one frontier cut (the "Besitzkarte"), PP0 decides each forward's cut as the
minimum over the batch, and the cut rides the #791 decision row downstream
(``ForwardRow``/``FollowerCursor``: ranks never derive it themselves; a
missing, stale or foreign row is a crash-stop, raenge-nie-uneins).

THE PLAN (``plan_split``) prices every chunk candidate of ``p_chunk_policy``
jointly with a per-chunk cut: per chunk the balanced cut under the frontier
rule, then the exact flow-shop makespan over the per-chunk stage times
(non-home chunks priced EAGER). The home-cut plan is always a candidate and
wins unless the dynamic one is ``min_gain`` faster.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import p_chunk_policy as _pcp

POLICY_STATIC = "static"
POLICY_DYNAMIC = "dynamic"
POLICIES = (POLICY_STATIC, POLICY_DYNAMIC)

POLICY_ENV = "SGLANG_P_LAYER_SPLIT"
SPEC_ENV = "SGLANG_P_LAYER_SPLIT_SPEC"
LOG_TAG = "P-LAYER-SPLIT"

FAMILY_ATTENTION = "attention"
FAMILY_LINEAR = "linear"

#: Keys of the write-back tensors inside a proxy frame (never collide with
#: hidden_states / residual / aux_layer_<id>).
WB_K = "swing_k:"
WB_V = "swing_v:"
WB_CONV = "swing_conv:"
WB_SSM = "swing_ssm:"
#: The row key inside the #791 decision row / proxy frame meta.
ROW_KEY = "__p_layer_split__"

DEFAULT_MIN_GAIN = 0.01


class LayerSplitError(ValueError):
    """An invalid geometry, model or spec."""


class LayerSplitDivergence(RuntimeError):
    """Two ranks disagree about a forward's layer ownership. Crash-stop."""


# ---------------------------------------------------------------------------
# stage model (per layer FAMILY, the same model as planner.pgap_stage_fit)


def _interp(points: Tuple[Tuple[int, float], ...], m: int) -> float:
    if not points:
        return 1.0
    if m <= points[0][0]:
        return float(points[0][1])
    if m >= points[-1][0]:
        return float(points[-1][1])
    for (m0, f0), (m1, f1) in zip(points, points[1:]):
        if m0 <= m <= m1:
            return f0 + (f1 - f0) * (m - m0) / float(m1 - m0)
    return 1.0


@dataclasses.dataclass(frozen=True)
class FamilyStageModel:
    """Time of ONE forward of stage ``s`` executing ``n`` layers of which
    ``a`` are full attention, for a chunk of ``m_real`` tokens (executing at
    width ``m_exec``) starting at ``prefix``, in ms::

        n * layer_ms[s] * (m_exec / C) * width_scale_s(m_exec) + stage_fixed_ms[s]
      + a * attn_ms_per_1k[s] * (m_real / C) * (prefix + m_real / 2) / 1000

    and an EAGER forward is at least ``n * eager_ms_per_layer[s]`` (host
    launch pace). ``C`` = ``chunk_tokens``. These are exactly the fields of
    ``planner.pgap_stage_fit.DepthLinearStageCost`` (the per-family model the
    P cut solver and the cut-graph work use), plus the width curve and the
    eager floor ``p_chunk_policy`` prices chunks with -- one model for the
    chunk plan, the static cut and the dynamic cut. ``deep_*`` is the fit's
    second segment from ``deep_from_prefix`` on (FI wave split)."""

    layer_ms: Tuple[float, ...]
    attn_ms_per_1k: Tuple[float, ...]
    stage_fixed_ms: Tuple[float, ...]
    chunk_tokens: int
    width_scale: Tuple[Tuple[Tuple[int, float], ...], ...] = ()
    eager_ms_per_layer: Tuple[float, ...] = ()
    deep_from_prefix: int = 0
    deep_layer_ms: Tuple[float, ...] = ()
    deep_attn_ms_per_1k: Tuple[float, ...] = ()
    deep_stage_fixed_ms: Tuple[float, ...] = ()

    def __post_init__(self):
        S = len(self.layer_ms)
        if S < 1 or len(self.attn_ms_per_1k) != S or len(self.stage_fixed_ms) != S:
            raise LayerSplitError("layer_ms, attn_ms_per_1k and stage_fixed_ms need one entry per stage")
        if self.chunk_tokens <= 0:
            raise LayerSplitError(f"chunk_tokens must be > 0, got {self.chunk_tokens}")
        for name in ("width_scale", "eager_ms_per_layer"):
            v = getattr(self, name)
            if v and len(v) != S:
                raise LayerSplitError(f"{name} needs one entry per stage or none")
        if self.deep_from_prefix and not (
            len(self.deep_layer_ms) == len(self.deep_attn_ms_per_1k) == len(self.deep_stage_fixed_ms) == S
        ):
            raise LayerSplitError("a deep segment needs deep_layer_ms/attn/fixed per stage")
        vals = list(self.layer_ms) + list(self.attn_ms_per_1k) + list(self.stage_fixed_ms)
        if any(x < 0 for x in vals):
            raise LayerSplitError("stage model rates must be >= 0")
        object.__setattr__(self, "width_scale", tuple(
            tuple(sorted((int(m), float(f)) for m, f in ws)) for ws in self.width_scale))

    @property
    def stages(self) -> int:
        return len(self.layer_ms)

    @classmethod
    def from_depth_linear(cls, cost, *, width_scale=(), eager_ms_per_layer=()) -> "FamilyStageModel":
        """Adopt a ``pgap_stage_fit.DepthLinearStageCost`` (duck-typed)."""
        return cls(
            tuple(float(x) for x in cost.layer_ms),
            tuple(float(x) for x in cost.attn_ms_per_1k),
            tuple(float(x) for x in cost.stage_fixed_ms),
            int(cost.chunk_tokens),
            tuple(width_scale), tuple(float(x) for x in eager_ms_per_layer),
            int(getattr(cost, "deep_from_prefix", 0) or 0),
            tuple(float(x) for x in getattr(cost, "deep_layer_ms", ()) or ()),
            tuple(float(x) for x in getattr(cost, "deep_attn_ms_per_1k", ()) or ()),
            tuple(float(x) for x in getattr(cost, "deep_stage_fixed_ms", ()) or ()),
        )

    def forward_ms(self, s: int, n: int, a: int, m_exec: int, m_real: int, prefix: int,
                   eager: bool) -> float:
        deep = self.deep_from_prefix > 0 and prefix >= self.deep_from_prefix
        lm = (self.deep_layer_ms if deep else self.layer_ms)[s]
        am = (self.deep_attn_ms_per_1k if deep else self.attn_ms_per_1k)[s]
        fx = (self.deep_stage_fixed_ms if deep else self.stage_fixed_ms)[s]
        C = float(self.chunk_tokens)
        ws = _interp(self.width_scale[s], m_exec) if self.width_scale else 1.0
        t = n * lm * (m_exec / C) * ws + fx + a * am * (m_real / C) * (prefix + 0.5 * m_real) / 1000.0
        if eager and self.eager_ms_per_layer:
            t = max(t, n * self.eager_ms_per_layer[s])
        return t

    def stage_ms(self, counts: Sequence[int], stage: int, w: int, prefix: int, mode: str,
                 attn: Optional[Sequence[int]] = None) -> float:
        """``p_stage_model.LayerCostModel.stage_ms``'s signature, so either
        model prices a cut (``attn`` is required here: this model does not
        know the layer layout)."""
        if attn is None:
            raise LayerSplitError("FamilyStageModel.stage_ms needs the attention counts")
        return self.forward_ms(int(stage), int(counts[stage]), int(attn[stage]), int(w), int(w),
                               int(prefix), mode == "eager")

    def to_json(self) -> Dict[str, object]:
        return {k: (list(map(list, v)) if k == "width_scale" else
                    (list(v) if isinstance(v, tuple) else v))
                for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_json(cls, d: Mapping[str, object]) -> "FamilyStageModel":
        return cls(
            tuple(d["layer_ms"]), tuple(d["attn_ms_per_1k"]), tuple(d["stage_fixed_ms"]),
            int(d["chunk_tokens"]),
            tuple(tuple((int(m), float(f)) for m, f in ws) for ws in d.get("width_scale", ()) or ()),
            tuple(d.get("eager_ms_per_layer", ()) or ()),
            int(d.get("deep_from_prefix", 0) or 0),
            tuple(d.get("deep_layer_ms", ()) or ()), tuple(d.get("deep_attn_ms_per_1k", ()) or ()),
            tuple(d.get("deep_stage_fixed_ms", ()) or ()),
        )


# ---------------------------------------------------------------------------
# geometry: home cut, swing windows, resident sets


def hybrid_families(num_layers: int, full_attention_interval: int) -> Tuple[str, ...]:
    """Qwen3.5/3.8 hybrid: every ``interval``-th layer (id % interval ==
    interval - 1) is full attention, the rest GDN."""
    k = int(full_attention_interval)
    return tuple(FAMILY_ATTENTION if (i % k) == k - 1 else FAMILY_LINEAR for i in range(int(num_layers)))


Cut = Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SplitGeometry:
    """``home_cuts`` = cumulative stage ends except the last (42,53 for
    42/11/11). ``window[b]`` = how many head layers of stage b+1's home span
    stage b may additionally execute; ``max_attn[b]`` caps the attention
    layers among them (each costs a pool-shaped KV mirror)."""

    families: Tuple[str, ...]
    home_cuts: Cut
    window: Tuple[int, ...]
    max_attn: Tuple[int, ...] = ()

    def __post_init__(self):
        L = len(self.families)
        cuts = tuple(int(c) for c in self.home_cuts)
        object.__setattr__(self, "home_cuts", cuts)
        object.__setattr__(self, "window", tuple(int(w) for w in self.window))
        if not self.max_attn:
            object.__setattr__(self, "max_attn", tuple(10 ** 6 for _ in cuts))
        if any(f not in (FAMILY_ATTENTION, FAMILY_LINEAR) for f in self.families):
            raise LayerSplitError("families must be 'attention' or 'linear'")
        if len(self.window) != len(cuts) or len(self.max_attn) != len(cuts):
            raise LayerSplitError("window and max_attn need one entry per boundary")
        bounds = (0,) + cuts + (L,)
        for s in range(len(bounds) - 1):
            if bounds[s + 1] - bounds[s] < 1:
                raise LayerSplitError(f"home cut {cuts} leaves stage {s} empty")
        for b, w in enumerate(self.window):
            if w < 0:
                raise LayerSplitError("window must be >= 0")
            # stage b+1 keeps at least one home layer it executes itself
            if cuts[b] + w > bounds[b + 2] - 1:
                raise LayerSplitError(
                    f"window {w} at boundary {b} would take stage {b + 1}'s whole home span "
                    f"[{cuts[b]}, {bounds[b + 2]})")

    @property
    def num_layers(self) -> int:
        return len(self.families)

    @property
    def stages(self) -> int:
        return len(self.home_cuts) + 1

    def bounds(self, cut: Cut) -> Tuple[int, ...]:
        return (0,) + tuple(cut) + (self.num_layers,)

    def executed(self, cut: Cut, s: int) -> range:
        b = self.bounds(cut)
        return range(b[s], b[s + 1])

    def counts(self, cut: Cut) -> Tuple[Tuple[int, int], ...]:
        b = self.bounds(cut)
        out = []
        for s in range(self.stages):
            ids = range(b[s], b[s + 1])
            out.append((len(ids), sum(1 for i in ids if self.families[i] == FAMILY_ATTENTION)))
        return tuple(out)

    def home(self, s: int) -> range:
        return self.executed(self.home_cuts, s)

    def home_rank(self, layer: int) -> int:
        for s in range(self.stages):
            if layer in self.home(s):
                return s
        raise LayerSplitError(f"layer {layer} out of range")

    def swing_window(self, s: int) -> range:
        """Layers stage ``s`` may execute beyond its home (the head of s+1)."""
        if s >= len(self.home_cuts):
            return range(0)
        c = self.home_cuts[s]
        return range(c, c + self.window[s])

    def resident(self, s: int) -> Tuple[int, ...]:
        """Layers stage ``s`` must BUILD (modules, weights, KV/state buffers):
        home plus swing window. The persistent layer set stays ``home``."""
        return tuple(self.home(s)) + tuple(self.swing_window(s))

    def swing(self, cut: Cut, s: int) -> Tuple[int, ...]:
        return tuple(i for i in self.executed(cut, s) if i not in self.home(s))

    def valid(self, cut: Cut) -> bool:
        if len(cut) != len(self.home_cuts):
            return False
        for b, c in enumerate(cut):
            h = self.home_cuts[b]
            if not (h <= c <= h + self.window[b]):
                return False
            if sum(1 for i in range(h, c) if self.families[i] == FAMILY_ATTENTION) > self.max_attn[b]:
                return False
        return True

    def cut_options(self) -> Tuple[Cut, ...]:
        opts: List[Cut] = [()]
        for b in range(len(self.home_cuts)):
            h = self.home_cuts[b]
            nxt = []
            for pre in opts:
                for c in range(h, h + self.window[b] + 1):
                    nxt.append(pre + (c,))
            opts = nxt
        return tuple(c for c in opts if self.valid(c))

    def to_json(self) -> Dict[str, object]:
        return {"families": "".join("A" if f == FAMILY_ATTENTION else "L" for f in self.families),
                "home_cuts": list(self.home_cuts), "window": list(self.window),
                "max_attn": list(self.max_attn)}

    @classmethod
    def from_json(cls, d: Mapping[str, object]) -> "SplitGeometry":
        fam = tuple(FAMILY_ATTENTION if ch == "A" else FAMILY_LINEAR for ch in str(d["families"]))
        return cls(fam, tuple(d["home_cuts"]), tuple(d["window"]), tuple(d.get("max_attn", ()) or ()))


def dominated(a: Cut, b: Cut) -> bool:
    """``a <= b`` componentwise (a moved no boundary further than b)."""
    return all(x <= y for x, y in zip(a, b))


def cut_min(cuts: Iterable[Cut]) -> Cut:
    cuts = list(cuts)
    return tuple(min(c[b] for c in cuts) for b in range(len(cuts[0])))


# ---------------------------------------------------------------------------
# pricing and the joint plan


@dataclasses.dataclass(frozen=True)
class WritebackPrice:
    """The mirror stage's per-chunk write-back, priced CONSERVATIVELY as
    added sender time (the proxy send is async on the rig; the planner does
    not count on the overlap). ``gbps[b]`` = the measured BAR1 rate of
    boundary b's link (card probe: the x4 3080 bounds both 27B P edges at
    ~6.5 GB/s). ``gdn_every_chunk``: the state of every swing GDN layer rides
    every chunk (the safe default; 'needed' sends it only on anchor/last/
    swing-out chunks)."""

    kv_bytes_per_token_layer: int
    state_bytes_per_layer: int
    gbps: Tuple[float, ...]
    gdn_every_chunk: bool = True

    def ms(self, geom: "SplitGeometry", cut: Cut, s: int, tokens: int) -> float:
        if s >= len(self.gbps) or self.gbps[s] <= 0:
            return 0.0
        b = writeback_bytes(geom, cut, s, tokens, 1, self.kv_bytes_per_token_layer,
                            self.state_bytes_per_layer, self.gdn_every_chunk)
        return b / (self.gbps[s] * 1e9) * 1000.0

    def to_json(self) -> Dict[str, object]:
        return {"kv_bytes_per_token_layer": self.kv_bytes_per_token_layer,
                "state_bytes_per_layer": self.state_bytes_per_layer, "gbps": list(self.gbps),
                "gdn_every_chunk": self.gdn_every_chunk}

    @classmethod
    def from_json(cls, d: Mapping[str, object]) -> "WritebackPrice":
        return cls(int(d["kv_bytes_per_token_layer"]), int(d["state_bytes_per_layer"]),
                   tuple(float(x) for x in d["gbps"]), bool(d.get("gdn_every_chunk", True)))


@dataclasses.dataclass(frozen=True)
class ExtraCosts:
    """What the dynamic split pays on top of the stage model.

    ``writeback``: the mirror's per-chunk BAR1 write-back (conservative:
    added to the sender). ``eager_host_ms_per_layer``: the host launch pace
    of an EAGER forward per executed layer (a non-home cut always runs
    eager); the forward costs at least ``n * floor``. rc9-era #PGAP (xsn422,
    before the graph): 63 / 42 / 74 ms for 42 / 11 / 11 layers -> 1.5 (5090)
    / 3.8 (3080) ms per layer. SG's layer-type model prices eager 512 by
    proportion from its 1024 point and carries no host floor, so without
    this an eager 512 chunk would look CHEAPER than the graph it replaces."""

    writeback: Optional[WritebackPrice] = None
    eager_host_ms_per_layer: Tuple[float, ...] = ()
    #: Per boundary: BAR1 rate of a PULL (home -> upstream mirror: the KV
    #: prefix of every newly swung attention layer and the GDN state of every
    #: newly swung GDN layer) when a cut RISES mid-request. Empty = rises only
    #: on a fresh request's first chunk (V1, nothing to pull). Priced as
    #: waiting time of the pulling stage (conservative: no prefetch).
    pull_gbps: Tuple[float, ...] = ()

    def pull_ms(self, geom: "SplitGeometry", prev: Cut, cut: Cut, s: int, prefix: int) -> float:
        if not self.pull_gbps or s >= len(self.pull_gbps) or self.writeback is None:
            return 0.0
        new = [i for i in geom.swing(cut, s) if i not in geom.swing(prev, s)]
        if not new:
            return 0.0
        na = sum(1 for i in new if geom.families[i] == FAMILY_ATTENTION)
        b = na * int(prefix) * self.writeback.kv_bytes_per_token_layer + (
            (len(new) - na) * self.writeback.state_bytes_per_layer)
        return b / (self.pull_gbps[s] * 1e9) * 1000.0

    def to_json(self) -> Dict[str, object]:
        return {"writeback": None if self.writeback is None else self.writeback.to_json(),
                "eager_host_ms_per_layer": list(self.eager_host_ms_per_layer),
                "pull_gbps": list(self.pull_gbps)}

    @classmethod
    def from_json(cls, d: Optional[Mapping[str, object]]) -> "ExtraCosts":
        if not d:
            return cls()
        wb = d.get("writeback")
        return cls(None if not wb else WritebackPrice.from_json(wb),
                   tuple(float(x) for x in d.get("eager_host_ms_per_layer", ()) or ()),
                   tuple(float(x) for x in d.get("pull_gbps", ()) or ()))


def stage_times(geom: SplitGeometry, model, cut: Cut, m_exec: int, m_real: int,
                prefix: int, eager: bool, extra: Optional[ExtraCosts] = None) -> Tuple[float, ...]:
    """Per-stage ms of one forward under ``cut``. ``model`` is SG's
    ``p_stage_model.LayerCostModel`` (the per-layer-type model, the default)
    or a :class:`FamilyStageModel` -- anything with
    ``stage_ms(counts, stage, w, prefix, mode, attn)``."""
    na = geom.counts(cut)
    counts = tuple(n for n, _ in na)
    attn = tuple(a for _, a in na)
    mode = "eager" if eager else "graph"
    out = []
    for s in range(geom.stages):
        t = model.stage_ms(counts, s, int(m_exec), int(prefix), mode, attn)
        if extra is not None:
            if eager and extra.eager_host_ms_per_layer:
                t = max(t, counts[s] * extra.eager_host_ms_per_layer[s])
            if extra.writeback is not None:
                t += extra.writeback.ms(geom, cut, s, m_real)
        out.append(t)
    return tuple(out)


_MODEL_KEYS: Dict[int, Tuple[object, str]] = {}


def model_key(model) -> str:
    """A stable identity for caching (SG's model holds dicts: unhashable)."""
    hit = _MODEL_KEYS.get(id(model))
    if hit is not None and hit[0] is model:
        return hit[1]
    raw = json.dumps(model.to_json(), sort_keys=True, default=str)
    key = type(model).__name__ + ":" + hashlib.sha256(raw.encode()).hexdigest()[:16]
    if len(_MODEL_KEYS) > 64:
        _MODEL_KEYS.clear()
    _MODEL_KEYS[id(model)] = (model, key)
    return key


def model_stages(model) -> int:
    n = getattr(model, "n_stages", None)
    return int(n) if n is not None else int(model.stages)


def check_layout(geom: "SplitGeometry", model) -> None:
    """SG's model knows the layer layout; it must be the geometry's."""
    lt = getattr(model, "layer_types", None)
    if lt is None:
        return
    mine = tuple(f == FAMILY_ATTENTION for f in geom.families)
    theirs = tuple(t == "full_attention" for t in lt)
    if mine != theirs:
        raise LayerSplitError("stage model layer layout differs from the split geometry's")


def makespan(chunks: Sequence[int], cuts: Sequence[Cut], start: int, geom: SplitGeometry,
             model: FamilyStageModel, limits: _pcp.ChunkLimits,
             extra: Optional[ExtraCosts] = None) -> float:
    """Exact flow shop (``p_chunk_policy.makespan_ms``) with a per-chunk cut.
    A chunk whose cut is not home executes EAGER (no captured graph)."""
    S = geom.stages
    k = limits.max_inflight or S
    fin = [0.0] * S
    last: List[float] = []
    p = int(start)
    before: Optional[Cut] = None if int(start) == 0 else geom.home_cuts
    for i, (c, cut) in enumerate(zip(chunks, cuts)):
        m_exec, eager = limits.exec_shape(int(c))
        if tuple(cut) != geom.home_cuts:
            m_exec, eager = int(c), True
        ts = stage_times(geom, model, cut, m_exec, int(c), p, eager, extra)
        if extra is not None and before is not None and extra.pull_gbps:
            ts = tuple(t + extra.pull_ms(geom, before, cut, s, p) for s, t in enumerate(ts))
        before = tuple(cut)
        prev = last[i - k] if i >= k else 0.0
        for s in range(S):
            f = max(fin[s], prev) + ts[s]
            fin[s] = f
            prev = f
        last.append(prev)
        p += int(c)
    return fin[-1] if S else 0.0


def balanced_cut(geom: SplitGeometry, model: FamilyStageModel, c: int, prefix: int,
                 limits: _pcp.ChunkLimits, ceiling: Optional[Cut] = None,
                 extra: Optional[ExtraCosts] = None, before: Optional[Cut] = None) -> Cut:
    """The cut minimising the chunk's bottleneck stage (ties: closest to
    home), among cuts dominated by ``ceiling`` (the frontier rule); with
    ``before`` (the previous chunk's cut) a rise pays its pull."""
    best, best_key = geom.home_cuts, None
    m_home, eager_home = limits.exec_shape(int(c))
    for cut in geom.cut_options():
        if ceiling is not None and not dominated(cut, ceiling):
            continue
        home = cut == geom.home_cuts
        ts = stage_times(geom, model, cut, m_home if home else int(c), int(c), prefix,
                         eager_home if home else True, extra)
        if extra is not None and before is not None and extra.pull_gbps:
            ts = tuple(t + extra.pull_ms(geom, before, cut, s, prefix) for s, t in enumerate(ts))
        key = (round(max(ts), 6), sum(cut))
        if best_key is None or key < best_key:
            best, best_key = cut, key
    return best


def trajectory(chunks: Sequence[int], start: int, geom: SplitGeometry, model: FamilyStageModel,
               limits: _pcp.ChunkLimits, allow_rise: bool,
               extra: Optional[ExtraCosts] = None) -> Tuple[Cut, ...]:
    """Per-chunk cuts. V1 (no ``extra.pull_gbps``): the frontier rule --
    above home only from a fresh first chunk (``allow_rise``), then never
    rising again. With pulls: any cut per chunk, a rise paying its pull. Then
    the last ``stages`` chunks are refined against the EXACT makespan (the
    drain is where a flow shop and a per-chunk bottleneck disagree)."""
    pulls = extra is not None and bool(extra.pull_gbps)
    ceiling: Optional[Cut] = None if (allow_rise or pulls) else geom.home_cuts
    out: List[Cut] = []
    p = int(start)
    mk = model_key(model)
    before: Optional[Cut] = None if int(start) == 0 else geom.home_cuts
    for c in chunks:
        key = (geom, mk, limits.key(), int(c), p, ceiling, extra, before if pulls else None)
        cut = _BALANCED.get(key)
        if cut is None:
            cut = balanced_cut(geom, model, int(c), p, limits, ceiling, extra,
                               before if pulls else None)
            if len(_BALANCED) > 200000:
                _BALANCED.clear()
            _BALANCED[key] = cut
        out.append(cut)
        if not pulls:
            ceiling = cut
        before = cut
        p += int(c)
    if not out:
        return ()
    best = makespan(chunks, out, start, geom, model, limits, extra)
    for i in range(max(0, len(out) - geom.stages), len(out)):
        lo = out[i - 1] if i > 0 else None
        for cut in geom.cut_options():
            if cut == out[i]:
                continue
            if not pulls:
                if lo is not None and not dominated(cut, lo):
                    continue
                if lo is None and not allow_rise and cut != geom.home_cuts:
                    continue
                if i + 1 < len(out) and not dominated(out[i + 1], cut):
                    continue
            trial = out[:i] + [cut] + out[i + 1:]
            ms = makespan(chunks, trial, start, geom, model, limits, extra)
            if ms < best - 1e-9:
                best, out = ms, trial
    return tuple(out)


_BALANCED: Dict[tuple, Cut] = {}
_PLANS: Dict[tuple, "SplitPlan"] = {}


@dataclasses.dataclass(frozen=True)
class SplitPlan:
    start: int
    chunks: Tuple[int, ...]
    cuts: Tuple[Cut, ...]
    predicted_ms: float
    home_ms: float
    home_chunks: Tuple[int, ...]
    candidate: str

    @property
    def gain(self) -> float:
        return 0.0 if self.home_ms <= 0 else 1.0 - self.predicted_ms / self.home_ms

    def distinct_cuts(self) -> Tuple[Cut, ...]:
        seen: List[Cut] = []
        for c in self.cuts:
            if not seen or seen[-1] != c:
                seen.append(c)
        return tuple(seen)


def _chunk_candidates(start: int, end: int, limits: _pcp.ChunkLimits, ramps: bool) -> List[Tuple[str, List[int]]]:
    """The same closed candidate family ``p_chunk_policy`` prices (fixed,
    mid x head-ramp x tail-ramp), in the same order."""
    out: List[Tuple[str, List[int]]] = []
    fixed = _pcp._apply_hook(_pcp._fixed(start, end, limits), start, end, limits)
    out.append(("fixed", fixed))
    seen = {tuple(fixed)}
    ladder = limits.ladder()
    for mid in ladder:
        opts = ([None] + [s for s in ladder if s < mid]) if ramps else [None]
        for head in opts:
            for tail in opts:
                cand = _pcp._apply_hook(_pcp._build(start, end, mid, head, tail, ladder, limits),
                                        start, end, limits)
                if tuple(cand) in seen:
                    continue
                seen.add(tuple(cand))
                out.append((f"mid={mid} head={head or '-'} tail={tail or '-'}", cand))
    return out


def plan_split(prompt_len: int, geom: SplitGeometry, model: FamilyStageModel,
               limits: _pcp.ChunkLimits, *, start: int = 0, min_gain: float = DEFAULT_MIN_GAIN,
               ramps: Optional[bool] = None, allow_rise: Optional[bool] = None,
               extra: Optional[ExtraCosts] = None) -> SplitPlan:
    """Joint chunk widths + per-chunk cut for ``prompt_len`` tokens from
    ``start``. ``allow_rise`` defaults to ``start == 0`` (V1: no prefix pull)."""
    n = int(prompt_len)
    if n <= 0:
        return SplitPlan(int(start), (), (), 0.0, 0.0, (), "empty")
    if model_stages(model) != geom.stages:
        raise LayerSplitError(f"model has {model_stages(model)} stages, geometry {geom.stages}")
    check_layout(geom, model)
    use_ramps = limits.ramps if ramps is None else bool(ramps)
    rise = (int(start) == 0) if allow_rise is None else bool(allow_rise)
    key = (int(start), int(start) + n, geom, model_key(model), limits.key(), use_ramps, rise,
           float(min_gain), extra)
    hit = _PLANS.get(key)
    if hit is None:
        hit = _plan(int(start), int(start) + n, geom, model, limits, use_ramps, rise, float(min_gain), extra)
        if len(_PLANS) > 512:
            _PLANS.clear()
        _PLANS[key] = hit
    return hit


def _plan(start, end, geom, model, limits, ramps, rise, min_gain, extra) -> SplitPlan:
    home = geom.home_cuts
    best_home: Optional[Tuple[float, str, List[int]]] = None
    best_dyn: Optional[Tuple[float, str, List[int], Tuple[Cut, ...]]] = None
    for name, chunks in _chunk_candidates(start, end, limits, ramps):
        hms = makespan(chunks, [home] * len(chunks), start, geom, model, limits, extra)
        if best_home is None or hms < best_home[0] - 1e-9:
            best_home = (hms, name, chunks)
        cuts = trajectory(chunks, start, geom, model, limits, rise, extra)
        if all(c == home for c in cuts):
            continue
        dms = makespan(chunks, cuts, start, geom, model, limits, extra)
        if best_dyn is None or dms < best_dyn[0] - 1e-9:
            best_dyn = (dms, name, chunks, cuts)
    hms, hname, hchunks = best_home
    if best_dyn is None or best_dyn[0] > hms * (1.0 - min_gain):
        return SplitPlan(start, tuple(hchunks), tuple([home] * len(hchunks)), hms, hms, tuple(hchunks),
                         "home" if best_dyn is None else "home(hysteresis)")
    dms, dname, dchunks, dcuts = best_dyn
    return SplitPlan(start, tuple(dchunks), tuple(dcuts), dms, hms, tuple(hchunks), dname)


def best_static_cut(prompt_len: int, geom_all: SplitGeometry, model,
                    limits: _pcp.ChunkLimits, *, start: int = 0,
                    extra: Optional[ExtraCosts] = None) -> Tuple[Cut, float, Tuple[int, ...]]:
    """The best CONSTANT cut over every contiguous cut of ``geom_all``'s
    layer count (brute force; the yardstick the dynamic plan is measured
    against). Each candidate cut is priced as its own home (graph where
    the chunk fits a bucket), with its best chunk plan."""
    L, S = geom_all.num_layers, geom_all.stages
    best: Optional[Tuple[float, Cut, Tuple[int, ...]]] = None

    def rec(pre: Tuple[int, ...], lo: int, left: int):
        nonlocal best
        if left == 0:
            g = SplitGeometry(geom_all.families, pre, tuple(0 for _ in pre))
            res = plan_split(prompt_len, g, model, limits, start=start, extra=extra)
            if best is None or res.home_ms < best[0]:
                best = (res.home_ms, pre, res.home_chunks)
            return
        for c in range(lo, L - left + 1):
            rec(pre + (c,), c + 1, left - 1)

    rec((), 1, S - 1)
    return best[1], best[0], best[2]


# ---------------------------------------------------------------------------
# spec, env, digest


@dataclasses.dataclass(frozen=True)
class SplitSpec:
    geometry: SplitGeometry
    model: object                      # p_stage_model.LayerCostModel (default) or FamilyStageModel
    limits: _pcp.ChunkLimits
    min_gain: float = DEFAULT_MIN_GAIN
    gdn_writeback: str = "every"      # every | needed
    source: str = ""
    extra: Optional[ExtraCosts] = None

    def to_json(self) -> str:
        return json.dumps({"geometry": self.geometry.to_json(), "model": model_to_json(self.model),
                           "limits": self.limits.to_json(), "min_gain": self.min_gain,
                           "gdn_writeback": self.gdn_writeback, "source": self.source,
                           "extra": None if self.extra is None else self.extra.to_json()},
                          separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "SplitSpec":
        d = json.loads(raw)
        wb = str(d.get("gdn_writeback", "every"))
        if wb not in ("every", "needed"):
            raise LayerSplitError(f"gdn_writeback {wb!r}: every|needed")
        return cls(SplitGeometry.from_json(d["geometry"]), model_from_json(d["model"]),
                   _pcp.ChunkLimits.from_json(d["limits"]), float(d.get("min_gain", DEFAULT_MIN_GAIN)),
                   wb, str(d.get("source", "")),
                   None if not d.get("extra") else ExtraCosts.from_json(d["extra"]))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def model_to_json(model) -> Dict[str, object]:
    kind = "family" if isinstance(model, FamilyStageModel) else "layer_cost"
    return {"kind": kind, "data": model.to_json()}


def model_from_json(d: Mapping[str, object]):
    kind = str(d.get("kind", ""))
    if kind == "family":
        return FamilyStageModel.from_json(d["data"])
    if kind == "layer_cost":
        from sglang.srt.weg2 import p_stage_model as _psm

        return _psm.LayerCostModel.from_json(d["data"])
    raise LayerSplitError(f"unknown stage model kind {kind!r}")


def split_from_env(env: Mapping[str, str]) -> Optional[SplitSpec]:
    """The group's spec, or None under static/unset (byte-identical default).
    ``dynamic`` without a readable spec RAISES (a boot that asked for dynamic
    must never silently measure static)."""
    pol = str(env.get(POLICY_ENV, "") or "").strip().lower()
    if pol in ("", POLICY_STATIC):
        return None
    if pol != POLICY_DYNAMIC:
        raise LayerSplitError(f"{POLICY_ENV}={pol!r}: expected one of {POLICIES}")
    raw = str(env.get(SPEC_ENV, "") or "").strip()
    if not raw:
        raise LayerSplitError(f"{POLICY_ENV}=dynamic without {SPEC_ENV}")
    return SplitSpec.from_json(raw)


# ---------------------------------------------------------------------------
# the Besitzkarte: PP0 decides per forward, everyone else follows the row


@dataclasses.dataclass(frozen=True)
class ForwardRow:
    """What crosses the wire per forward (inside the #791 decision row)."""

    version: int
    digest: str
    cut: Cut
    gdn_wb: bool

    def encode(self) -> List[object]:
        return [int(self.version), str(self.digest), [int(c) for c in self.cut], int(bool(self.gdn_wb))]

    @classmethod
    def decode(cls, raw: Sequence[object]) -> "ForwardRow":
        if raw is None or len(raw) != 4:
            raise LayerSplitDivergence(f"{LOG_TAG}: malformed row {raw!r}")
        return cls(int(raw[0]), str(raw[1]), tuple(int(c) for c in raw[2]), bool(int(raw[3])))


@dataclasses.dataclass
class _ReqState:
    end: int
    frontier: Cut
    plan: SplitPlan
    steps: Dict[int, Tuple[int, Cut]]


class LeaderCursor:
    """PP0 only. Plans each request once (joint chunks + cuts), keeps its
    frontier (the one monotone cut its mirrors are valid under) and decides
    each forward's cut as the minimum over the batch."""

    def __init__(self, spec: SplitSpec, on_plan=None, max_keys: int = 256):
        self.spec = spec
        self.digest = spec.digest()
        self.on_plan = on_plan
        self._reqs: Dict[object, _ReqState] = {}
        self._version = 0
        self._max_keys = int(max_keys)

    def _make(self, key, pos: int, end: int) -> _ReqState:
        sp = self.spec
        plan = plan_split(end - pos, sp.geometry, sp.model, sp.limits, start=pos, min_gain=sp.min_gain,
                          extra=sp.extra)
        steps: Dict[int, Tuple[int, Cut]] = {}
        p = pos
        for c, cut in zip(plan.chunks, plan.cuts):
            steps[p] = (c, cut)
            p += c
        frontier = plan.cuts[0] if (pos == 0 and plan.cuts) else sp.geometry.home_cuts
        st = _ReqState(end, frontier, plan, steps)
        if len(self._reqs) >= self._max_keys:
            self._reqs.pop(next(iter(self._reqs)))
        self._reqs[key] = st
        if self.on_plan is not None:
            self.on_plan(key, plan, pos, end)
        return st

    def next_width(self, key, pos: int, end: int) -> int:
        """The planned chunk width (the ``ChunkPlanner.next_width`` twin).
        An off-plan ``pos`` (narrowed by corridor/park/load-back) replans the
        rest; the frontier is kept (it can only fall)."""
        pos, end = int(pos), int(end)
        if end <= pos:
            return 0
        st = self._reqs.get(key)
        if st is None or st.end != end:
            st = self._make(key, pos, end)
        elif pos not in st.steps:
            frontier = st.frontier
            st = self._make(key, pos, end)
            st.frontier = frontier
        return int(st.steps[pos][0])

    def wanted_cut(self, key, pos: int) -> Cut:
        st = self._reqs.get(key)
        if st is None or pos not in st.steps:
            return self.spec.geometry.home_cuts
        return st.steps[pos][1]

    def decide(self, batch: Sequence[Tuple[object, int, int]], *, anchor_or_last: bool = True) -> ForwardRow:
        """``batch`` = ``(key, pos, width)`` of every request in this forward,
        in batch order. The forward's cut = min over the batch of each
        request's wanted cut capped by its frontier; a request whose first
        forward this is (pos == 0, fresh) may lift its frontier to its plan.
        Every frontier then falls to the forward's cut."""
        geom = self.spec.geometry
        home = geom.home_cuts
        caps: List[Cut] = []
        for key, pos, width in batch:
            st = self._reqs.get(key)
            if st is None:
                caps.append(home)
                continue
            want = self.wanted_cut(key, int(pos))
            cap = st.frontier
            caps.append(want if dominated(want, cap) else cut_min([want, cap]))
        cut = cut_min(caps) if caps else home
        if not geom.valid(cut):
            raise LayerSplitError(f"{LOG_TAG}: decided cut {cut} outside the geometry")
        for key, pos, width in batch:
            st = self._reqs.get(key)
            if st is not None:
                st.frontier = cut_min([st.frontier, cut])
                if int(pos) + int(width) >= st.end:
                    self._reqs.pop(key, None)
        self._version += 1
        swing_gdn = any(geom.families[i] == FAMILY_LINEAR
                        for s in range(geom.stages) for i in geom.swing(cut, s))
        gdn_wb = swing_gdn and (self.spec.gdn_writeback == "every" or anchor_or_last)
        return ForwardRow(self._version, self.digest, cut, gdn_wb)

    def forget(self, key) -> None:
        self._reqs.pop(key, None)


class FollowerCursor:
    """Every rank (PP0 included, on its own row): adopt the forwarded row and
    derive what this stage executes, mirrors and writes back. Any
    disagreement -- missing row, foreign spec, version not advancing, a cut
    outside the geometry -- is a crash-stop."""

    def __init__(self, spec: SplitSpec, stage: int):
        self.spec = spec
        self.stage = int(stage)
        self.digest = spec.digest()
        self._last_version = 0

    def adopt(self, raw) -> ForwardRow:
        if raw is None:
            raise LayerSplitDivergence(
                f"{LOG_TAG}: stage {self.stage} got a forward without a layer-split row while "
                f"dynamic is armed (PP0 decides every forward's cut)")
        row = raw if isinstance(raw, ForwardRow) else ForwardRow.decode(raw)
        if row.digest != self.digest:
            raise LayerSplitDivergence(
                f"{LOG_TAG}: stage {self.stage} spec digest {self.digest} != row {row.digest}")
        if row.version <= self._last_version:
            raise LayerSplitDivergence(
                f"{LOG_TAG}: stage {self.stage} row version {row.version} does not advance "
                f"past {self._last_version}")
        if not self.spec.geometry.valid(row.cut):
            raise LayerSplitDivergence(f"{LOG_TAG}: row cut {row.cut} outside the geometry")
        self._last_version = row.version
        return row

    def executed(self, row: ForwardRow) -> Tuple[int, ...]:
        return tuple(self.spec.geometry.executed(row.cut, self.stage))

    def swing(self, row: ForwardRow) -> Tuple[int, ...]:
        """Layers this stage executes for its downstream neighbour (mirror,
        written back after the forward)."""
        return self.spec.geometry.swing(row.cut, self.stage)

    def incoming(self, row: ForwardRow) -> Tuple[int, ...]:
        """Home layers the upstream neighbour executed (written back to us)."""
        if self.stage == 0:
            return ()
        return self.spec.geometry.swing(row.cut, self.stage - 1)

    def graph_ok(self, row: ForwardRow) -> bool:
        return row.cut == self.spec.geometry.home_cuts


# ---------------------------------------------------------------------------
# swing slab pricing (the planner's post)


@dataclasses.dataclass(frozen=True)
class SwingSlab:
    stage: int
    attn_layers: int
    gdn_layers: int
    weight_bytes: int
    kv_bytes: int
    state_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.state_bytes


def swing_slab(geom: SplitGeometry, stage: int, *, pool_tokens: int, kv_bytes_per_token_layer: int,
               state_slots: int, state_bytes_per_layer: int, weight_bytes_attn: int,
               weight_bytes_gdn: int) -> SwingSlab:
    """VRAM a stage spends on its swing window: the window's weights, a
    pool-shaped KV mirror per attention layer ((pool+1) rows addressed by the
    stage's own slots) and a state-pool-shaped mirror per GDN layer."""
    win = tuple(geom.swing_window(stage))
    na = sum(1 for i in win if geom.families[i] == FAMILY_ATTENTION)
    ng = len(win) - na
    return SwingSlab(int(stage), na, ng, na * int(weight_bytes_attn) + ng * int(weight_bytes_gdn),
                     na * (int(pool_tokens) + 1) * int(kv_bytes_per_token_layer),
                     ng * (int(state_slots) + 1) * int(state_bytes_per_layer))


def writeback_bytes(geom: SplitGeometry, cut: Cut, stage: int, tokens: int, requests: int,
                    kv_bytes_per_token_layer: int, state_bytes_per_layer: int, gdn_wb: bool) -> int:
    sw = geom.swing(cut, stage)
    na = sum(1 for i in sw if geom.families[i] == FAMILY_ATTENTION)
    ng = len(sw) - na
    return na * int(tokens) * int(kv_bytes_per_token_layer) + (
        ng * int(requests) * int(state_bytes_per_layer) if gdn_wb else 0)


# ---------------------------------------------------------------------------
# log lines


def plan_line(plan: SplitPlan, *, key: object = "", end: int = 0) -> str:
    runs: List[str] = []
    for i, c in enumerate(plan.cuts):
        tag = "/".join(map(str, c))
        if runs and runs[-1].split("x")[0] == tag:
            head, _, cnt = runs[-1].partition("x")
            runs[-1] = f"{head}x{int(cnt or 1) + 1}"
        else:
            runs.append(tag)
    if len(runs) > 12:
        runs = runs[:6] + ["..."] + runs[-6:]
    return (f"{LOG_TAG} plan policy=dynamic key={key} start={plan.start} tokens={end - plan.start} "
            f"chunks={len(plan.chunks)} cuts=[{','.join(runs)}] widths=[{_pcp.format_plan(_pcp.PlanResult(plan.chunks, 0, 0, 0, ''))}] "
            f"predicted_ms={plan.predicted_ms:.1f} home_ms={plan.home_ms:.1f} gain={100.0 * plan.gain:+.1f}% "
            f"pick={plan.candidate}")


def armed_line(spec: SplitSpec, where: str = "") -> str:
    g = spec.geometry
    return (f"{LOG_TAG} armed policy=dynamic{(' ' + where) if where else ''} home_cuts={list(g.home_cuts)} "
            f"window={list(g.window)} max_attn={list(g.max_attn)} "
            f"resident={[len(g.resident(s)) for s in range(g.stages)]} min_gain={spec.min_gain:g} "
            f"gdn_writeback={spec.gdn_writeback} digest={spec.digest()} source={spec.source or '-'}")


# ---------------------------------------------------------------------------
# tensor helpers (torch imported lazily; the runtime binds them to its pools)


def gather_rows(buf, loc):
    """``buf[loc]`` along dim 0, contiguous (the sender's gather)."""
    return buf.index_select(0, loc).contiguous()


def scatter_rows(buf, loc, rows) -> None:
    """``buf[loc] = rows`` along dim 0 (the receiver's scatter)."""
    buf.index_copy_(0, loc, rows.to(buf.dtype))


def request_loc(req_to_token, row: int, start: int, end: int):
    """A request's slot indices for positions [start, end) on THIS rank --
    each rank reads its OWN req_to_token row; no index crosses the wire."""
    return req_to_token[int(row), int(start):int(end)].long()


def writeback_payload(swing_attn: Sequence[int], swing_gdn: Sequence[int], kv_of, state_of,
                      out_cache_loc, state_idx, gdn_wb: bool) -> Dict[str, object]:
    """After the forward on the MIRROR stage: this chunk's new KV rows of each
    swing attention layer (at this rank's ``out_cache_loc``) and, when
    ``gdn_wb``, the state of each swing GDN layer (at this rank's
    ``state_idx``). ``kv_of(l) -> (k, v)``, ``state_of(l) -> tuple of
    tensors`` indexed by the rank's own slot / state index."""
    out: Dict[str, object] = {}
    for l in swing_attn:
        k, v = kv_of(l)
        out[f"{WB_K}{l}"] = gather_rows(k, out_cache_loc)
        out[f"{WB_V}{l}"] = gather_rows(v, out_cache_loc)
    if gdn_wb:
        for l in swing_gdn:
            conv, ssm = state_of(l)
            out[f"{WB_CONV}{l}"] = gather_rows(conv, state_idx)
            out[f"{WB_SSM}{l}"] = gather_rows(ssm, state_idx)
    return out


def apply_writeback(payload: Mapping[str, object], incoming_attn: Sequence[int],
                    incoming_gdn: Sequence[int], kv_of, state_of, out_cache_loc, state_idx,
                    gdn_wb: bool) -> None:
    """On the HOME stage, before its own forward of the same chunk: scatter
    the upstream mirror's rows into the home pool at THIS rank's slots. A
    missing entry is a crash-stop (home would publish/anchor stale data)."""
    for l in incoming_attn:
        kk, vk = f"{WB_K}{l}", f"{WB_V}{l}"
        if kk not in payload or vk not in payload:
            raise LayerSplitDivergence(f"{LOG_TAG}: write-back of attention layer {l} missing")
        k, v = kv_of(l)
        scatter_rows(k, out_cache_loc, payload[kk])
        scatter_rows(v, out_cache_loc, payload[vk])
    if gdn_wb:
        for l in incoming_gdn:
            ck, sk = f"{WB_CONV}{l}", f"{WB_SSM}{l}"
            if ck not in payload or sk not in payload:
                raise LayerSplitDivergence(f"{LOG_TAG}: write-back of GDN layer {l} missing")
            conv, ssm = state_of(l)
            scatter_rows(conv, state_idx, payload[ck])
            scatter_rows(ssm, state_idx, payload[sk])


def is_writeback_key(name: str) -> bool:
    return name.startswith((WB_K, WB_V, WB_CONV, WB_SSM))


__all__ = [
    "POLICY_STATIC", "POLICY_DYNAMIC", "POLICIES", "POLICY_ENV", "SPEC_ENV", "LOG_TAG", "ROW_KEY",
    "LayerSplitError", "LayerSplitDivergence", "FamilyStageModel", "SplitGeometry", "SplitPlan",
    "SplitSpec", "ForwardRow", "LeaderCursor", "FollowerCursor", "SwingSlab", "hybrid_families",
    "plan_split", "best_static_cut", "makespan", "balanced_cut", "trajectory", "swing_slab",
    "writeback_bytes", "WritebackPrice", "ExtraCosts", "split_from_env", "plan_line", "armed_line", "gather_rows", "scatter_rows",
    "request_loc", "writeback_payload", "apply_writeback", "is_writeback_key", "dominated", "cut_min",
]
