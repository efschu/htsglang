"""Dynamic prefill chunk plan for a PIPELINED prefill group (``--p-chunk-policy``).

User order 25.09. ~21:40Z: "dynamische chunkgroesse brauchen wir. umsetzen" (27B),
~21:41Z: "auch fuers nf brauchen wir dynamische chunkgroesse" -- one interface
for both lines. This module is that interface: PURE (stdlib only, no sglang
import, nothing model-specific), every model parameter handed in.

THE PROBLEM. A P group is a pipeline of ``S`` stages. With ``n`` equal chunks
of stage time ``t`` its wall is ~``(n + S - 1) * t``: the ``S - 1`` term is the
fill/drain bubble (8k at 512: n=16, ~11 %; at 2048: n=4, ~33 %). Larger chunks
run the kernels more efficiently (5090 NVFP4 GEMM 828 TFLOPS at M=512 against
1086 at M=4096, INT8 470 -> 535 TOPS) but lengthen the bubble; every extra
chunk also pays the stage's per-forward fixed cost (27B graph replay: a few ms;
NF: the expert stream, ~1.7 s). There is no single right size: it depends on
the prompt length, the stage count and the per-stage cost curve.

THE RULE (``chunk_plan``). Price every candidate chunk sequence EXACTLY on the
handed-in stage model as a flow shop -- stage ``s`` starts chunk ``i`` once it
finished chunk ``i-1`` and stage ``s-1`` finished chunk ``i`` (plus the
in-flight cap) -- and take the cheapest. The candidates are a small, closed,
deterministic family: a middle size ``m`` from the allowed ladder, optionally
reached by a DOUBLING head ramp from ``h`` (small first chunks fill the
pipeline early) and left by a HALVING tail ramp down to ``t`` (small last
chunks drain it early). The fixed plan (``fixed_tokens`` repeated, today's
behaviour) is always a candidate, and it WINS unless another plan is predicted
at least ``min_gain`` faster -- so under its own model the dynamic plan is
never worse than fixed, and noise in the model cannot flip-flop the width.

THE LIMITS (``ChunkLimits``) are parameters, never constants here:
  * ``max_tokens``  -- the largest chunk; the planner sets it together with the
    VRAM/activation corridor it priced (27B: argv --chunked-prefill-size of
    group P under the dynamic policy; NF: with the residency). The caller
    hands ``max_tokens`` and the stage models from the SAME residency /
    planner state (NF: a_s depends on FR_P) -- a model that does not belong
    to the ceiling plans on the wrong curve.
  * ``min_tokens``  -- the smallest chunk the plan chooses (a final rest chunk
    may be shorter); the ladder is ``min_tokens * 2**k`` up to ``max_tokens``
    plus ``max_tokens`` itself and every graph bucket inside the range.
  * ``page``        -- every non-final chunk END is an absolute page multiple.
  * ``grid``        -- no chunk crosses an absolute multiple of ``grid`` (0 =
    off). NF: 4096 (its publish window H74 and tail fold H63d; NF-P runs no
    --mamba-checkpoint-interval).
    27B: its P anchors are DISTANCE-based (``weg2_anchor_step``: >= 4096 since
    the last anchor, at any chunk boundary), so 0 is correct there.
  * ``graph_buckets`` -- captured prefill graph sizes (() = none, NF). A chunk
    of ``c`` tokens runs at the smallest bucket >= ``c`` (padded); above the
    largest bucket it runs EAGER, priced with the stage's ``eager_floor_ms``
    (host launch) -- or is not allowed at all when ``eager`` is False.
  * ``max_inflight`` -- micro-batches in flight (default = stage count).
  * ``tail_hook``   -- optional ``f(chunks, start, end) -> chunks`` applied to
    every candidate (NF's tail-fold rule). It must keep the sum and every chunk
    within ``max_tokens``; a violation raises (a hook that breaks the plan is a
    bug, not a fallback).

BATCH LEVEL (NF item 4). Nothing here knows what a "request" is: the plan is
over a TOKEN STREAM ``[start, start + n_tokens)``. Per request (27B, P bs 1)
``start`` is the request's prefix and ``n_tokens`` its rest; for a forward
budget over several bundled prompts (NF UNDIVIDED_MICRO_BATCH / P_BURST_
ASSEMBLY) pass the total pending tokens and use the first entry as the
forward's budget -- the bundling inside that budget stays the caller's.

``ChunkPlanner`` is the stateful per-request cursor the scheduler uses
(``next_width``); it replans deterministically when the executed width
deviated from the plan (corridor narrowing, park, host load-back).

Every input of a plan is an argument, so the plan is a pure function of
them: a PP group whose ranks call it with the same arguments gets the same
widths (on the 27B only PP0's widths are executed anyway -- downstream ranks
run PP0's forwarded extents, #791).
"""

from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: The one flag value set (identical on both lines).
POLICY_FIXED = "fixed"
POLICY_DYNAMIC = "dynamic"
POLICIES = (POLICY_FIXED, POLICY_DYNAMIC)

#: Environment the launcher hands to a P group under ``dynamic`` (both lines).
POLICY_ENV = "SGLANG_P_CHUNK_POLICY"
SPEC_ENV = "SGLANG_P_CHUNK_SPEC"

#: Default hysteresis: another plan must be predicted >= 1 % faster than fixed.
DEFAULT_MIN_GAIN = 0.01

#: The log line prefix (both lines).
LOG_TAG = "P-CHUNK-POLICY"


class ChunkPolicyError(ValueError):
    """An invalid stage model, limit set or hook result."""


# ---------------------------------------------------------------------------
# stage model


@dataclasses.dataclass(frozen=True)
class StageModel:
    """Time of ONE forward of ONE pipeline stage, in ms.

    ``forward_ms(m_exec, m_real, prefix, eager)`` =
        ``base(m_exec)``                      -- prefix-free cost at the width
                                                 that actually executes (a
                                                 padded graph bucket runs its
                                                 GEMMs at the bucket width)
      + ``attn_ms_per_tok_1k * m_real * (prefix + m_real / 2) / 1000``
                                              -- the attention of the real
                                                 tokens against the prefix and
                                                 their own causal half
    and an eager forward is at least ``eager_floor_ms`` (host launch pace).

    ``base`` is the piecewise-linear interpolation of ``points`` =
    ``((m, ms), ...)`` ascending in ``m``; beyond the ends it extends the
    outer segment; ONE point means proportional (``ms * m / m0``). The
    NF form ``a + b * M`` is ``StageModel.linear(a, b)``.
    """

    points: Tuple[Tuple[int, float], ...]
    attn_ms_per_tok_1k: float = 0.0
    eager_floor_ms: float = 0.0
    name: str = ""

    def __post_init__(self):
        pts = tuple((int(m), float(t)) for m, t in self.points)
        if not pts:
            raise ChunkPolicyError("StageModel needs at least one (tokens, ms) point")
        ms = [m for m, _ in pts]
        if any(m < 0 for m in ms) or ms != sorted(ms) or len(set(ms)) != len(ms):
            raise ChunkPolicyError(f"StageModel points must be ascending, unique, >= 0: {pts}")
        if any(t < 0 for _, t in pts):
            raise ChunkPolicyError(f"StageModel point times must be >= 0: {pts}")
        if len(pts) == 1 and pts[0][0] == 0:
            raise ChunkPolicyError("a single StageModel point needs tokens > 0")
        if self.attn_ms_per_tok_1k < 0 or self.eager_floor_ms < 0:
            raise ChunkPolicyError("attn_ms_per_tok_1k and eager_floor_ms must be >= 0")
        object.__setattr__(self, "points", pts)

    @classmethod
    def linear(cls, a_ms: float, b_ms_per_token: float, *, attn_ms_per_tok_1k: float = 0.0,
               eager_floor_ms: float = 0.0, name: str = "") -> "StageModel":
        """``t(M) = a + b * M`` (NF form: a large fixed a_s per forward)."""
        return cls(((0, float(a_ms)), (1, float(a_ms) + float(b_ms_per_token))),
                   attn_ms_per_tok_1k, eager_floor_ms, name)

    def base_ms(self, m: int) -> float:
        pts = self.points
        if len(pts) == 1:
            m0, t0 = pts[0]
            return t0 * float(m) / float(m0)
        if m <= pts[0][0]:
            (m0, t0), (m1, t1) = pts[0], pts[1]
        elif m >= pts[-1][0]:
            (m0, t0), (m1, t1) = pts[-2], pts[-1]
        else:
            i = 1
            while pts[i][0] < m:
                i += 1
            (m0, t0), (m1, t1) = pts[i - 1], pts[i]
        return max(0.0, t0 + (t1 - t0) * (float(m) - m0) / float(m1 - m0))

    def forward_ms(self, m_exec: int, m_real: int, prefix: int, eager: bool) -> float:
        t = self.base_ms(m_exec) + self.attn_ms_per_tok_1k * m_real * (prefix + 0.5 * m_real) / 1000.0
        if eager and self.eager_floor_ms > t:
            t = self.eager_floor_ms
        return t

    def to_json(self) -> Dict[str, object]:
        return {"points": [list(p) for p in self.points], "attn_ms_per_tok_1k": self.attn_ms_per_tok_1k,
                "eager_floor_ms": self.eager_floor_ms, "name": self.name}

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "StageModel":
        if "a_ms" in d:
            return cls.linear(float(d["a_ms"]), float(d.get("b_ms_per_token", 0.0)),
                              attn_ms_per_tok_1k=float(d.get("attn_ms_per_tok_1k", 0.0)),
                              eager_floor_ms=float(d.get("eager_floor_ms", 0.0)),
                              name=str(d.get("name", "")))
        return cls(tuple((int(m), float(t)) for m, t in d["points"]),
                   float(d.get("attn_ms_per_tok_1k", 0.0)), float(d.get("eager_floor_ms", 0.0)),
                   str(d.get("name", "")))


# ---------------------------------------------------------------------------
# limits


TailHook = Callable[[List[int], int, int], List[int]]


@dataclasses.dataclass(frozen=True)
class ChunkLimits:
    """The allowed chunk set and the fixed baseline. See the module docstring."""

    max_tokens: int
    min_tokens: int
    fixed_tokens: int
    page: int = 1
    grid: int = 0
    graph_buckets: Tuple[int, ...] = ()
    eager: bool = True
    max_inflight: int = 0
    min_gain: float = DEFAULT_MIN_GAIN
    ramps: bool = True
    tail_hook: Optional[TailHook] = dataclasses.field(default=None, compare=False, hash=False)
    #: Short-prompt bypass (rc9j metal, 26.09.): a request whose rest is at
    #: most this many tokens when the cursor first sees it is NOT planned --
    #: every forward gets ``fixed_tokens``, exactly the fixed policy's width.
    #: 0 = off (every request is planned). A request that started above it
    #: keeps following its plan to the end.
    dynamic_min_tokens: int = 0

    def __post_init__(self):
        object.__setattr__(self, "graph_buckets", tuple(sorted({int(b) for b in self.graph_buckets})))
        if self.page < 1:
            raise ChunkPolicyError(f"page must be >= 1, got {self.page}")
        if not (0 < self.min_tokens <= self.max_tokens):
            raise ChunkPolicyError(
                f"need 0 < min_tokens <= max_tokens, got {self.min_tokens}/{self.max_tokens}")
        if not (0 < self.fixed_tokens <= self.max_tokens):
            raise ChunkPolicyError(
                f"fixed_tokens {self.fixed_tokens} must be in (0, max_tokens={self.max_tokens}]")
        for name in ("max_tokens", "min_tokens", "fixed_tokens"):
            if int(getattr(self, name)) % self.page:
                raise ChunkPolicyError(f"{name}={getattr(self, name)} is not a multiple of page {self.page}")
        if self.grid < 0 or (self.grid and self.grid % self.page):
            raise ChunkPolicyError(f"grid {self.grid} must be 0 or a positive multiple of page {self.page}")
        if self.grid and self.max_tokens > self.grid:
            raise ChunkPolicyError(
                f"max_tokens {self.max_tokens} exceeds grid {self.grid}: no chunk may cross the grid")
        if any(b <= 0 for b in self.graph_buckets):
            raise ChunkPolicyError(f"graph buckets must be positive: {self.graph_buckets}")
        if not self.eager:
            if not self.graph_buckets:
                raise ChunkPolicyError("eager=False needs graph buckets")
            if self.max_tokens > self.graph_buckets[-1]:
                raise ChunkPolicyError(
                    f"eager=False: max_tokens {self.max_tokens} above the largest graph bucket "
                    f"{self.graph_buckets[-1]}")
        if self.max_inflight < 0 or not (0.0 <= self.min_gain < 1.0):
            raise ChunkPolicyError("max_inflight must be >= 0 and 0 <= min_gain < 1")
        if self.dynamic_min_tokens < 0:
            raise ChunkPolicyError(f"dynamic_min_tokens must be >= 0, got {self.dynamic_min_tokens}")

    def ladder(self) -> Tuple[int, ...]:
        """The chunk sizes a plan chooses from, ascending."""
        sizes = set()
        s = self.min_tokens
        while s <= self.max_tokens:
            sizes.add(s)
            s *= 2
        sizes.add(self.max_tokens)
        sizes.add(self.fixed_tokens)
        sizes.update(b for b in self.graph_buckets if self.min_tokens <= b <= self.max_tokens)
        return tuple(sorted(sizes))

    def exec_shape(self, c: int) -> Tuple[int, bool]:
        """``(width that executes, eager?)`` for a chunk of ``c`` tokens."""
        for b in self.graph_buckets:
            if b >= c:
                return b, False
        return c, True

    def key(self) -> Tuple:
        return (self.max_tokens, self.min_tokens, self.fixed_tokens, self.page, self.grid,
                self.graph_buckets, self.eager, self.max_inflight, self.min_gain, self.ramps,
                self.dynamic_min_tokens)

    def to_json(self) -> Dict[str, object]:
        return {"max_tokens": self.max_tokens, "min_tokens": self.min_tokens,
                "fixed_tokens": self.fixed_tokens, "page": self.page, "grid": self.grid,
                "graph_buckets": list(self.graph_buckets), "eager": self.eager,
                "max_inflight": self.max_inflight, "min_gain": self.min_gain, "ramps": self.ramps,
                "dynamic_min_tokens": self.dynamic_min_tokens}

    @classmethod
    def from_json(cls, d: Dict[str, object], tail_hook: Optional[TailHook] = None) -> "ChunkLimits":
        return cls(int(d["max_tokens"]), int(d["min_tokens"]), int(d["fixed_tokens"]),
                   int(d.get("page", 1)), int(d.get("grid", 0)),
                   tuple(int(b) for b in d.get("graph_buckets", ()) or ()),
                   bool(d.get("eager", True)), int(d.get("max_inflight", 0)),
                   float(d.get("min_gain", DEFAULT_MIN_GAIN)), bool(d.get("ramps", True)), tail_hook,
                   int(d.get("dynamic_min_tokens", 0) or 0))


# ---------------------------------------------------------------------------
# pricing


def makespan_ms(chunks: Sequence[int], start: int, stage_time_model: Sequence[StageModel],
                limits: ChunkLimits) -> float:
    """Wall of ``chunks`` (tokens from ``start``) through the pipeline, ms.

    Exact flow-shop recursion: stage ``s`` starts chunk ``i`` at
    ``max(own finish of i-1, stage s-1 finish of i)``; stage 0 additionally
    waits for the LAST stage to finish chunk ``i - max_inflight``.
    """
    S = len(stage_time_model)
    k = limits.max_inflight or S
    fin = [0.0] * S
    last_done: List[float] = []
    p = int(start)
    for i, c in enumerate(chunks):
        m_exec, eager = limits.exec_shape(int(c))
        ready = last_done[i - k] if i >= k else 0.0
        prev = ready
        for s, st in enumerate(stage_time_model):
            f = max(fin[s], prev) + st.forward_ms(m_exec, int(c), p, eager)
            fin[s] = f
            prev = f
        last_done.append(prev)
        p += int(c)
    return fin[-1] if S else 0.0


def _largest_le(ladder: Sequence[int], x: int) -> Optional[int]:
    best = None
    for s in ladder:
        if s <= x:
            best = s
    return best


def _place(p: int, start: int, end: int, want: int, ladder: Sequence[int], limits: ChunkLimits) -> int:
    """The chunk that starts at ``p`` when ``want`` tokens are asked for."""
    rem = end - p
    c = min(int(want), rem)
    if c < rem:
        # Natural alignment: a ladder size s starts only at an offset that is
        # a multiple of s (a doubling ramp then stays on its own grid). The
        # origin is the stream start -- or absolute 0 under a grid, so the
        # chunks tile the grid instead of leaving a gap after every grid
        # point. An offset on no ladder multiple (an unaligned resume) first
        # closes the gap to the next multiple of the smallest size.
        off = p - (0 if limits.grid else start)
        aligned = [s for s in ladder if s <= c and off % s == 0]
        c = aligned[-1] if aligned else min(ladder[0] - off % ladder[0], rem)
    if limits.grid:
        to_grid = (p // limits.grid + 1) * limits.grid - p
        c = min(c, to_grid)
    if c < end - p and limits.page > 1:
        e = (p + c) // limits.page * limits.page
        if e <= p:
            e = min(end, (p // limits.page + 1) * limits.page)
        c = e - p
    return max(1, min(c, end - p))


def _build(start: int, end: int, mid: int, head: Optional[int], tail: Optional[int],
           ladder: Sequence[int], limits: ChunkLimits) -> List[int]:
    out: List[int] = []
    p = start
    while p < end:
        want = mid
        if head is not None:
            done = p - start
            want = min(want, max(head, _largest_le(ladder, done) or head))
        if tail is not None:
            want = min(want, max(tail, _largest_le(ladder, (end - p) // 2) or tail))
        c = _place(p, start, end, want, ladder, limits)
        out.append(c)
        p += c
    return out


def _fixed(start: int, end: int, limits: ChunkLimits) -> List[int]:
    """Today's behaviour: ``fixed_tokens`` per forward, the rest last -- held
    to the same hard limits as every candidate (grid, page); with ``grid`` 0
    and ``page`` 1 exactly the fixed scheduler's split."""
    out: List[int] = []
    p = start
    while p < end:
        c = min(limits.fixed_tokens, end - p)
        if limits.grid:
            c = min(c, (p // limits.grid + 1) * limits.grid - p)
        if c < end - p and limits.page > 1:
            e = (p + c) // limits.page * limits.page
            c = (e - p) if e > p else min(end, (p // limits.page + 1) * limits.page) - p
        out.append(c)
        p += c
    return out


def _apply_hook(chunks: List[int], start: int, end: int, limits: ChunkLimits) -> List[int]:
    if limits.tail_hook is None:
        return chunks
    got = [int(c) for c in limits.tail_hook(list(chunks), start, end)]
    if sum(got) != end - start or any(c <= 0 or c > limits.max_tokens for c in got):
        raise ChunkPolicyError(
            f"tail_hook broke the plan: {chunks} -> {got} (sum must stay {end - start}, "
            f"every chunk in (0, {limits.max_tokens}])")
    return got


@dataclasses.dataclass(frozen=True)
class PlanResult:
    chunks: Tuple[int, ...]
    predicted_ms: float
    fixed_ms: float
    fixed_chunks: int
    candidate: str

    @property
    def gain(self) -> float:
        return 0.0 if self.fixed_ms <= 0 else 1.0 - self.predicted_ms / self.fixed_ms


@lru_cache(maxsize=512)
def _plan_cached(start: int, end: int, stages: Tuple[StageModel, ...], lkey: Tuple,
                 limits: ChunkLimits, ramps: bool, hook_id: int) -> PlanResult:
    fixed = _apply_hook(_fixed(start, end, limits), start, end, limits)
    fixed_ms = makespan_ms(fixed, start, stages, limits)
    best = PlanResult(tuple(fixed), fixed_ms, fixed_ms, len(fixed), "fixed")
    ladder = limits.ladder()
    seen = {tuple(fixed)}
    for mid in ladder:
        opts = ([None] + [s for s in ladder if s < mid]) if ramps else [None]
        for head in opts:
            for tail in opts:
                cand = _apply_hook(_build(start, end, mid, head, tail, ladder, limits), start, end, limits)
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                ms = makespan_ms(cand, start, stages, limits)
                if ms < best.predicted_ms - 1e-9:
                    best = PlanResult(key, ms, fixed_ms, len(fixed),
                                      f"mid={mid} head={head or '-'} tail={tail or '-'}")
    if best.candidate != "fixed" and best.predicted_ms > fixed_ms * (1.0 - limits.min_gain):
        return PlanResult(tuple(fixed), fixed_ms, fixed_ms, len(fixed), "fixed(hysteresis)")
    return best


def plan_detail(prompt_len: int, stages: int, stage_time_model: Sequence[StageModel],
                limits: ChunkLimits, *, start: int = 0, ramps: Optional[bool] = None) -> PlanResult:
    """``chunk_plan`` with its prediction (for the log line and the dry run).

    ``prompt_len`` = tokens to prefill from absolute position ``start`` (for a
    request with a cached prefix: its rest, with ``start`` = the prefix).
    """
    n = int(prompt_len)
    if n <= 0:
        return PlanResult((), 0.0, 0.0, 0, "empty")
    if int(stages) != len(stage_time_model) or int(stages) < 1:
        raise ChunkPolicyError(
            f"stages={stages} but the stage model has {len(stage_time_model)} entries")
    use_ramps = limits.ramps if ramps is None else bool(ramps)
    return _plan_cached(int(start), int(start) + n, tuple(stage_time_model), limits.key(), limits,
                        use_ramps, id(limits.tail_hook))


def chunk_plan(prompt_len: int, stages: int, stage_time_model: Sequence[StageModel],
               limits: ChunkLimits, *, start: int = 0) -> List[int]:
    """THE interface: the chunk widths for ``prompt_len`` tokens from ``start``."""
    return list(plan_detail(prompt_len, stages, stage_time_model, limits, start=start).chunks)


# ---------------------------------------------------------------------------
# spec (what the launcher hands a P group) and the per-request cursor


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    stages: Tuple[StageModel, ...]
    limits: ChunkLimits
    source: str = ""

    def to_json(self) -> str:
        return json.dumps({"stages": [s.to_json() for s in self.stages],
                           "limits": self.limits.to_json(), "source": self.source},
                          separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str, tail_hook: Optional[TailHook] = None) -> "PolicySpec":
        d = json.loads(raw)
        stages = tuple(StageModel.from_json(s) for s in d["stages"])
        if not stages:
            raise ChunkPolicyError("policy spec has no stages")
        return cls(stages, ChunkLimits.from_json(d["limits"], tail_hook), str(d.get("source", "")))


def policy_from_env(env: Dict[str, str], tail_hook: Optional[TailHook] = None) -> Optional[PolicySpec]:
    """The group's spec, or None when the policy is fixed/unset (byte-identical default).

    A dynamic policy without a readable spec RAISES: silently running fixed
    under a boot that asked for dynamic would measure the wrong arm.
    """
    pol = str(env.get(POLICY_ENV, "") or "").strip().lower()
    if pol in ("", POLICY_FIXED):
        return None
    if pol != POLICY_DYNAMIC:
        raise ChunkPolicyError(f"{POLICY_ENV}={pol!r}: expected one of {POLICIES}")
    raw = str(env.get(SPEC_ENV, "") or "").strip()
    if not raw:
        raise ChunkPolicyError(f"{POLICY_ENV}=dynamic without {SPEC_ENV}")
    return PolicySpec.from_json(raw, tail_hook)


class ChunkPlanner:
    """Per-request cursor: ``next_width(key, pos, end)``.

    The first call for ``key`` plans ``[pos, end)`` with ramps. Later calls
    whose ``pos`` is one of the plan's boundaries take the next planned
    width; a ``pos`` OFF the plan (the executed width was narrowed by the
    corridor, a park or a host load-back) replans ``[pos, end)`` WITHOUT a
    head ramp -- the pipeline is already running. Deterministic in its
    call sequence. ``on_plan(key, result, pos, end, replan)`` is called once
    per (re)plan (the log line).
    """

    def __init__(self, spec: PolicySpec, on_plan: Optional[Callable[..., None]] = None,
                 max_keys: int = 256):
        self.spec = spec
        self.on_plan = on_plan
        self._plans: Dict[object, Tuple[int, Dict[int, int]]] = {}
        self._max_keys = int(max_keys)

    def _make(self, key, pos: int, end: int, ramps: bool) -> Dict[int, int]:
        res = plan_detail(end - pos, len(self.spec.stages), self.spec.stages, self.spec.limits,
                          start=pos, ramps=ramps and self.spec.limits.ramps)
        steps: Dict[int, int] = {}
        p = pos
        for c in res.chunks:
            steps[p] = c
            p += c
        if len(self._plans) >= self._max_keys:
            self._plans.pop(next(iter(self._plans)))
        self._plans[key] = (end, steps)
        if self.on_plan is not None:
            self.on_plan(key, res, pos, end, not ramps)
        return steps

    def next_width(self, key, pos: int, end: int) -> int:
        pos, end = int(pos), int(end)
        if end <= pos:
            return 0
        known = self._plans.get(key)
        if known is None or known[0] != end:
            lim = self.spec.limits
            if lim.dynamic_min_tokens and end - pos <= lim.dynamic_min_tokens:
                # Short-prompt bypass: no plan, no log line, the fixed width.
                return min(lim.fixed_tokens, end - pos)
            steps = self._make(key, pos, end, ramps=True)
        else:
            steps = known[1]
            if pos not in steps:
                steps = self._make(key, pos, end, ramps=False)
        return int(steps[pos])

    def forget(self, key) -> None:
        self._plans.pop(key, None)


def forward_budget(planner: ChunkPlanner, key, pos: int, end: int) -> int:
    """The forward's token budget for the request ``key`` at ``pos`` of ``end``.

    The planned width -- except when that width FINISHES the request: then
    the budget is at least ``fixed_tokens``, so the room a short prompt
    leaves in the forward stays available to bundle others exactly as under
    the fixed policy (a plan never shrinks the forward below fixed just
    because its head request is short). 0 when nothing is left.
    """
    w = planner.next_width(key, pos, end)
    if w and w >= end - pos:
        w = max(w, planner.spec.limits.fixed_tokens)
    return int(w)


def armed_line(spec: PolicySpec, where: str = "") -> str:
    """The ONE line a rank prints when the dynamic policy is armed."""
    lim = spec.limits
    return (
        f"{LOG_TAG} armed policy={POLICY_DYNAMIC}{(' ' + where) if where else ''} "
        f"stages={len(spec.stages)} ladder={list(lim.ladder())} max={lim.max_tokens} "
        f"min={lim.min_tokens} fixed={lim.fixed_tokens} page={lim.page} grid={lim.grid} "
        f"graph_buckets={list(lim.graph_buckets)} eager={lim.eager} "
        f"min_gain={lim.min_gain:g} ramps={lim.ramps} "
        f"dynamic_min_tokens={lim.dynamic_min_tokens} source={spec.source or '-'}"
    )


def planner_from_env(env: Dict[str, str], log: Optional[Callable[[str], None]] = None,
                     tail_hook: Optional[TailHook] = None) -> Optional[ChunkPlanner]:
    """The scheduler's entry: a ``ChunkPlanner`` under ``dynamic``, else None.

    ``log`` receives the armed line once and one plan line per request;
    replans (the executed width left the plan) are rate-limited to
    occurrences 1-16 and every 64th.
    """
    spec = policy_from_env(env, tail_hook)
    if spec is None:
        return None
    counts = {"plans": 0, "replans": 0}

    def _on_plan(key, res: PlanResult, pos: int, end: int, replan: bool = False) -> None:
        if log is None:
            return
        if replan:
            counts["replans"] += 1
            n = counts["replans"]
            if n > 16 and n % 64:
                return
            log(plan_line(res, key=key, start=pos, end=end) + f" replan=1 replans={n}")
            return
        counts["plans"] += 1
        log(plan_line(res, key=key, start=pos, end=end))

    if log is not None:
        log(armed_line(spec))
    return ChunkPlanner(spec, on_plan=_on_plan)


def format_plan(res: PlanResult, max_items: int = 12) -> str:
    """Run-length form of a plan: ``512x2,1024,2048x14,1024,512,300``."""
    runs: List[str] = []
    for c in res.chunks:
        if runs and runs[-1].split("x")[0] == str(c):
            head, _, cnt = runs[-1].partition("x")
            runs[-1] = f"{head}x{int(cnt or 1) + 1}"
        else:
            runs.append(str(c))
    if len(runs) > max_items:
        runs = runs[: max_items // 2] + ["..."] + runs[-max_items // 2:]
    return ",".join(runs)


def plan_line(res: PlanResult, *, key: object = "", start: int = 0, end: int = 0,
              policy: str = POLICY_DYNAMIC) -> str:
    """The ONE log line of a plan (same text on both lines)."""
    return (
        f"{LOG_TAG} plan policy={policy} key={key} start={start} tokens={end - start} "
        f"chunks={len(res.chunks)} fixed_chunks={res.fixed_chunks} widths=[{format_plan(res)}] "
        f"predicted_ms={res.predicted_ms:.1f} fixed_ms={res.fixed_ms:.1f} "
        f"gain={100.0 * res.gain:+.1f}% pick={res.candidate}"
    )


def bubble_share(chunks: Sequence[int], start: int, stage_time_model: Sequence[StageModel],
                 limits: ChunkLimits) -> float:
    """``1 - (busiest stage's busy time) / wall`` under the model."""
    wall = makespan_ms(chunks, start, stage_time_model, limits)
    if wall <= 0:
        return 0.0
    busy = []
    for st in stage_time_model:
        p, b = start, 0.0
        for c in chunks:
            m_exec, eager = limits.exec_shape(int(c))
            b += st.forward_ms(m_exec, int(c), p, eager)
            p += int(c)
        busy.append(b)
    return max(0.0, 1.0 - max(busy) / wall)


__all__ = [
    "POLICY_FIXED", "POLICY_DYNAMIC", "POLICIES", "POLICY_ENV", "SPEC_ENV", "LOG_TAG",
    "ChunkPolicyError", "StageModel", "ChunkLimits", "PlanResult", "PolicySpec", "ChunkPlanner",
    "chunk_plan", "plan_detail", "makespan_ms", "bubble_share", "policy_from_env", "format_plan",
    "plan_line", "armed_line", "planner_from_env", "forward_budget",
]
