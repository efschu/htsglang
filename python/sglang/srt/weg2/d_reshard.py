# SPDX-License-Identifier: Apache-2.0
"""--d-reshard: the D layout's weight shards chosen per load class (27B, 26.09.).

User order 26.09. ~08:40Z: "... soll auch ein dynamisches sharding implementiert
werden, sodass je nach anstehender last die layer auch im D layout dynamisch zur
laufzeit resharded werden ...", clarified ~08:45Z: "das bezieht sich natuerlich
nur auf TP1< - also nicht fuers nf".  Design and evidence:
``/spinning/gpu-arb/docs/DYN_D_RESHARD.md``.

SCOPE.  D layouts with TP > 1 in the 27B style: uneven TP (``--rank-tp-ratio``)
plus uneven DCP (KV split by tokens).  The NF D form (Form A: TP0 holds attention
and host, the 3080 ranks are expert workers) is not a TP>1 shard layout and is
refused by the profile field (:data:`PROFILE_SUPPORT`), never silently ignored.

WHAT A PRESET MOVES (stage "wake", this module's first stage).  Only the dense
MLP family vector (``tp_family="mlp"``, ``distributed/utils.py`` family plans,
the same lever ``--rank-mlp-ratio`` / ``SGLANG_UNEVEN_MLP_VECTOR`` already pins
per boot).  Held fixed, because moving them is where a second ledger would start:

* attention q/o heads: with 4 KV heads over 3 ranks the only split with >= 1 KV
  head per rank is [2, 1, 1] groups (12/6/6 q heads) -- there is nothing to move;
* GDN heads: a different head split re-shapes the per-rank mamba pool tensors
  and the arena GDN blob extents (stage 2 candidate, priced here, not built);
* the DCP token vector: the virtual pool size (``max_total_num_tokens``) is a
  function of it and is pinned at boot in the scheduler/allocator (stage 3).

WHY THE WAKE.  Every P->D flip re-writes all D weight bytes from the P holders
(``weight_exchange`` S1: the plan is pure arithmetic over both groups' shard
vectors), and the cutover re-enters every request through a HiCache READ of the
canonical, geometry-neutral page.  A different MLP vector at the wake moves no
extra byte; what it needs is (a) a per-preset view of the D MLP storage, (b) one
CUDA-graph set per preset, (c) the per-preset D rows of the flip manifest (pure
``partition_sizes`` arithmetic), and (d) ONE planner posten: the per-rank spread
``max_p bytes_r(p) - bytes_r(boot)``, which the KV pool cannot use.

RANKS NEVER DISAGREE.  One process decides (the front at the wake, it knows the
coming epoch's batch), every D rank adopts the :class:`ReshardRow` of that epoch
(:class:`FollowerCursor`); a missing, foreign, replayed or unknown row is a
crash-stop (:class:`ReshardDivergence`).

Everything here is pure: no CUDA, no allocation, no process state beyond the
module-level policy read by the launcher.  The rank-side executor (views, graph
sets, manifest rows, planner posten) is NOT wired: ``wake``/``live`` refuse the
boot in the launcher until it is (DYN_D_RESHARD.md sec. 7).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.distributed.utils import partition_units

POLICY_OFF = "off"
POLICY_WAKE = "wake"
POLICY_LIVE = "live"
POLICIES = (POLICY_OFF, POLICY_WAKE, POLICY_LIVE)

POLICY_ENV = "SGLANG_WEG2_D_RESHARD"
SPEC_ENV = "SGLANG_WEG2_D_RESHARD_SPEC"

#: The profile field (UNIFY_PLAN "Profil-Registry"; on desk/27b-unified-0926 this
#: becomes ``ModelProfile.d_reshard``).  Operator order 26.09. ~08:45Z.
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
PROFILE_SUPPORT: Mapping[str, str] = {"qwen27b": SUPPORTED, "nextflash": UNSUPPORTED}

#: A preset is taken only when predicted at least this much faster than the one
#: in force (hysteresis; the boot-to-boot spread on this rig is 2.6-4.2 %).
DEFAULT_MIN_GAIN = 0.02

KV_CELL_BYTES = 32768  # 16 attn layers x 4 kv heads x 256 x (K,V) x fp8: every DCP rank, per owned token


class ReshardError(ValueError):
    """A spec, preset or load class that cannot be priced or applied."""


class ReshardDivergence(RuntimeError):
    """Ranks would run different shard maps: crash-stop (raenge-nie-uneins)."""


class ReshardProfileRefused(RuntimeError):
    """--d-reshard requested for a profile whose D form is not a TP>1 shard layout."""


# ---------------------------------------------------------------------------
# Profile gate
# ---------------------------------------------------------------------------


def profile_of_config(cfg: Mapping) -> Optional[str]:
    """'qwen27b' | 'nextflash' | None from a checkpoint config.json (text_config
    aware).  The NF line is the MoE one (num_experts > 0); the 27B is the dense
    Qwen3.5/3.8 hybrid.  None = unknown architecture (refused under wake/live)."""
    text = cfg.get("text_config", cfg) if isinstance(cfg, Mapping) else {}
    experts = int(text.get("num_experts", 0) or 0)
    mtype = str(text.get("model_type", "") or cfg.get("model_type", ""))
    if experts > 0:
        return "nextflash"
    if mtype.startswith("qwen3_5") or ("full_attention_interval" in text and "linear_num_key_heads" in text):
        return "qwen27b"
    return None


def check_profile(profile: Optional[str], policy: str) -> None:
    """Refuse wake/live for any profile not marked supported (NF: Form A)."""
    if policy == POLICY_OFF:
        return
    support = PROFILE_SUPPORT.get(str(profile or ""), UNSUPPORTED)
    if support != SUPPORTED:
        raise ReshardProfileRefused(
            f"--d-reshard {policy}: REFUSED for profile {profile!r} (d_reshard={support}). "
            f"Dynamic D resharding exists only for TP>1 D layouts with uneven TP + uneven "
            f"DCP (qwen27b). The NF D form (Form A: TP0 attention/host, expert workers) "
            f"has no shard vector to move; the flag is refused, not ignored.")


# ---------------------------------------------------------------------------
# Geometry: which bytes follow which vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DGeometry:
    """The dense hybrid (Qwen3.5/3.8 27B) as the D shard plan sees it.

    Parameter counts per family; ``*_bpp`` = streamed bytes per parameter of the
    checkpoint format.  ``mlp_units`` is the family's shard granularity
    (INT8: 1088 x 16 elements; NVFP4: 136 x 128, the swizzled scale tile)."""

    hidden: int
    layers: int
    attn_layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    gdn_k_heads: int
    gdn_k_dim: int
    gdn_v_heads: int
    gdn_v_dim: int
    inter: int
    vocab: int
    mlp_units: int
    linear_bpp: float
    lm_head_bpp: float = 2.0
    draft_params: float = 1.6e9
    draft_bpp: float = 1.0

    @classmethod
    def from_config(cls, cfg: Mapping, *, mlp_units: int, linear_bpp: float,
                    draft_params: float = 1.6e9, draft_bpp: float = 1.0) -> "DGeometry":
        t = cfg.get("text_config", cfg)
        n = int(t["num_hidden_layers"])
        interval = int(t.get("full_attention_interval", 4))
        return cls(hidden=int(t["hidden_size"]), layers=n, attn_layers=n // interval,
                   q_heads=int(t["num_attention_heads"]), kv_heads=int(t["num_key_value_heads"]),
                   head_dim=int(t["head_dim"]), gdn_k_heads=int(t["linear_num_key_heads"]),
                   gdn_k_dim=int(t["linear_key_head_dim"]), gdn_v_heads=int(t["linear_num_value_heads"]),
                   gdn_v_dim=int(t["linear_value_head_dim"]), inter=int(t["intermediate_size"]),
                   vocab=int(t["vocab_size"]), mlp_units=int(mlp_units), linear_bpp=float(linear_bpp),
                   draft_params=float(draft_params), draft_bpp=float(draft_bpp))

    # -- parameter counts per family (whole model) ----------------------------
    def mlp_params(self) -> float:
        return 3.0 * self.hidden * self.inter * self.layers

    def attn_sharded_params(self) -> float:
        qg = self.q_heads * self.head_dim * 2  # q + output gate (attn_output_gate)
        return float(self.hidden * qg + self.q_heads * self.head_dim * self.hidden) * self.attn_layers

    def attn_replicated_params(self) -> float:
        # uneven DCP: every rank recomputes all kv heads (replicated K/V proj)
        return float(self.hidden * 2 * self.kv_heads * self.head_dim) * self.attn_layers

    def gdn_params(self) -> float:
        kd, vd = self.gdn_k_heads * self.gdn_k_dim, self.gdn_v_heads * self.gdn_v_dim
        qkvz = self.hidden * (2 * kd + 2 * vd)
        ba = self.hidden * 2 * self.gdn_v_heads
        return float(qkvz + ba + vd * self.hidden) * (self.layers - self.attn_layers)

    def lm_head_params(self) -> float:
        return float(self.vocab * self.hidden)

    def mlp_bytes(self) -> float:
        return self.mlp_params() * self.linear_bpp


@dataclass(frozen=True)
class Vector:
    """A D shard map: the base vector plus the MLP family vector (stage 1) and the
    GDN head units (stage 2, None = follow base)."""

    base: Tuple[int, ...]
    mlp: Optional[Tuple[int, ...]] = None
    gdn: Optional[Tuple[int, ...]] = None  # explicit per-rank GDN k-head units

    def validate(self, geom: DGeometry) -> None:
        n = len(self.base)
        if n < 2:
            raise ReshardError("d-reshard needs a TP>1 D layout")
        for label, v in (("base", self.base), ("mlp", self.mlp)):
            if v is not None and (len(v) != n or any(int(w) <= 0 for w in v)):
                raise ReshardError(f"{label} vector {v} invalid for {n} ranks")
        if self.gdn is not None and (len(self.gdn) != n or sum(self.gdn) != geom.gdn_k_heads
                                     or min(self.gdn) < 1):
            raise ReshardError(f"gdn units {self.gdn} must be {n} entries >=1 summing to {geom.gdn_k_heads}")


def family_units(geom: DGeometry, vec: Vector) -> Dict[str, List[int]]:
    """Per-rank units of every family, by the SAME largest-remainder rule the
    layers use (``distributed/utils.partition_units``)."""
    vec.validate(geom)
    n = len(vec.base)
    mlp = partition_units(geom.mlp_units, list(vec.mlp or vec.base))
    gdn = list(vec.gdn) if vec.gdn is not None else partition_units(geom.gdn_k_heads, list(vec.base))
    attn = partition_units(geom.kv_heads, list(vec.base)) if geom.kv_heads >= n else [1] * n
    draft = partition_units(8, list(vec.base))
    return {"mlp": mlp, "gdn": gdn, "attn": attn, "draft": draft}


def family_shares(geom: DGeometry, vec: Vector) -> Dict[str, List[float]]:
    u = family_units(geom, vec)
    tot = {"mlp": geom.mlp_units, "gdn": geom.gdn_k_heads,
           "attn": sum(u["attn"]), "draft": 8}
    return {k: [x / tot[k] for x in v] for k, v in u.items()}


def rank_bytes(geom: DGeometry, vec: Vector) -> List[float]:
    """Per-rank weight bytes streamed per decode round (target + draft + lm_head),
    i.e. the resident D weight footprint that moves with the vector."""
    s = family_shares(geom, vec)
    n = len(vec.base)
    out = []
    lm = geom.lm_head_params() * geom.lm_head_bpp / n  # vocab stays even (M22)
    rep = geom.attn_replicated_params() * geom.linear_bpp
    for r in range(n):
        b = geom.mlp_bytes() * s["mlp"][r]
        b += geom.attn_sharded_params() * geom.linear_bpp * s["attn"][r] + rep
        b += geom.gdn_params() * geom.linear_bpp * s["gdn"][r]
        b += geom.draft_params * geom.draft_bpp * s["draft"][r]
        b += lm
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Cost model (calibrated on the D logs, see DYN_D_RESHARD.md sec. 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DCalib:
    """Per-rank constants of one checkpoint format on this rig.

    decode compute_r = F_r + (1 + beta_r*(bs-1)) * bytes_r / E_r
                       + attn_r * tokens_r * bs * ctx/1000       [ms per round]
    decode round     = max_r compute_r + W(bs),  W = w1 + w_slope*(bs-1)
    prefill chunk    = max_r (pf_fixed_r + pf_k_r * byte_share_r) + pf_wait  [4096 tokens]
    """

    name: str
    e_gbs: Tuple[float, ...]
    f_ms: Tuple[float, ...]
    beta: Tuple[float, ...]
    attn_ms_per_ktok: Tuple[float, ...]
    w1_ms: float
    w_slope_ms: float
    pf_fixed_ms: Tuple[float, ...]
    pf_k_ms: Tuple[float, ...]
    pf_wait_ms: float
    provenance: str = ""


def fit_two_point(bytes_a: float, ms_a: float, bytes_b: float, ms_b: float) -> Tuple[float, float]:
    """(E in GB/s, F in ms) of ms = F + bytes/E through two operating points."""
    if abs(bytes_a - bytes_b) < 1e6:
        raise ReshardError("two-point fit needs two different byte loads")
    slope = (ms_a - ms_b) / ((bytes_a - bytes_b) / 1e9)  # ms per GB
    if slope <= 0:
        raise ReshardError(f"non-physical fit: slope {slope:.3f} ms/GB")
    return 1000.0 / slope, ms_a - slope * bytes_a / 1e9


@dataclass(frozen=True)
class LoadClass:
    kind: str  # "decode" | "prefill"
    bs: int = 1
    ctx: int = 0  # per request, tokens
    label: str = ""

    def key(self) -> str:
        return self.label or (f"{self.kind}-bs{self.bs}-{self.ctx // 1024}k" if self.kind == "decode"
                              else f"prefill-{self.ctx // 1024}k")


def decode_compute_ms(geom: DGeometry, cal: DCalib, vec: Vector, load: LoadClass,
                      token_share: Sequence[float]) -> List[float]:
    b = rank_bytes(geom, vec)
    out = []
    for r in range(len(b)):
        t = cal.f_ms[r] + (1.0 + cal.beta[r] * (load.bs - 1)) * (b[r] / 1e9) * 1000.0 / cal.e_gbs[r]
        t += cal.attn_ms_per_ktok[r] * token_share[r] * load.bs * load.ctx / 1000.0
        out.append(t)
    return out


def round_ms(geom: DGeometry, cal: DCalib, vec: Vector, load: LoadClass,
             token_share: Sequence[float]) -> float:
    """Wall ms of one D round (decode) or of one 4096-token D-prefill chunk."""
    if load.kind == "decode":
        return max(decode_compute_ms(geom, cal, vec, load, token_share)) + cal.w1_ms + cal.w_slope_ms * (load.bs - 1)
    if load.kind == "prefill":
        b = rank_bytes(geom, vec)
        tot = sum(b)
        return max(cal.pf_fixed_ms[r] + cal.pf_k_ms[r] * b[r] / tot for r in range(len(b))) + cal.pf_wait_ms
    raise ReshardError(f"unknown load kind {load.kind!r}")


# ---------------------------------------------------------------------------
# Presets and the per-class choice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    name: str
    mlp: Tuple[int, ...]

    def vector(self, base: Sequence[int]) -> Vector:
        return Vector(tuple(int(x) for x in base), tuple(int(x) for x in self.mlp))


def mlp_vector_for_share(geom: DGeometry, share0: float, n: int = 3) -> Tuple[int, ...]:
    """MLP units with rank 0 at ``share0`` and the rest split evenly (unit vector,
    the form SGLANG_UNEVEN_MLP_VECTOR takes)."""
    u0 = max(1, min(geom.mlp_units - (n - 1), int(round(share0 * geom.mlp_units))))
    rest = geom.mlp_units - u0
    tail = [rest // (n - 1)] * (n - 1)
    for i in range(rest - sum(tail)):
        tail[i] += 1
    return tuple([u0] + tail)


def spread_bytes(geom: DGeometry, base: Sequence[int], presets: Sequence[Preset],
                 boot: Optional[Preset] = None) -> List[float]:
    """Per-rank bytes the KV pool cannot use when every preset must fit: max over
    presets minus the boot preset (the one posten the planner must carry)."""
    ref = rank_bytes(geom, (boot.vector(base) if boot else Vector(tuple(base))))
    per = [rank_bytes(geom, p.vector(base)) for p in presets]
    return [max(0.0, max(pb[r] for pb in per) - ref[r]) for r in range(len(ref))]


def kv_tokens_after(capacity: Sequence[int], delta_bytes: Sequence[float],
                    token_vector: Optional[Sequence[int]] = None) -> Tuple[List[int], int]:
    """Per-rank local KV capacity after ``delta_bytes`` more weights, and the
    virtual pool (min_r cap_r / tv_r * sum tv; capacity-proportional tv if None)."""
    caps = [max(0, int(c - d / KV_CELL_BYTES)) for c, d in zip(capacity, delta_bytes)]
    if token_vector is None:
        return caps, sum(caps)
    unit = min(c / t for c, t in zip(caps, token_vector))
    return caps, int(unit * sum(token_vector))


def best_preset(geom: DGeometry, cal: DCalib, base: Sequence[int], presets: Sequence[Preset],
                load: LoadClass, token_share: Sequence[float]) -> Tuple[Preset, float]:
    """The fastest preset for one load class; ties -> earlier preset (deterministic)."""
    best = None
    for p in presets:
        t = round_ms(geom, cal, p.vector(base), load, token_share)
        if best is None or t < best[1] - 1e-9:
            best = (p, t)
    return best


def choose(geom: DGeometry, cal: DCalib, base: Sequence[int], presets: Sequence[Preset],
           load: LoadClass, token_share: Sequence[float], current: Optional[str],
           min_gain: float = DEFAULT_MIN_GAIN) -> str:
    """The preset for the coming epoch: the best one, unless the one in force is
    within ``min_gain`` of it (hysteresis -- a switch costs a graph-set change)."""
    p, t = best_preset(geom, cal, base, presets, load, token_share)
    if current is not None:
        cur = {q.name: q for q in presets}.get(current)
        if cur is None:
            raise ReshardError(f"preset in force {current!r} is not in the spec")
        tc = round_ms(geom, cal, cur.vector(base), load, token_share)
        if tc <= t * (1.0 + min_gain):
            return cur.name
    return p.name


# ---------------------------------------------------------------------------
# Spec (what the launcher hands group D) and the per-epoch row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReshardSpec:
    policy: str
    base: Tuple[int, ...]
    presets: Tuple[Preset, ...]
    boot: str
    min_gain: float = DEFAULT_MIN_GAIN
    fmt: str = ""

    def validate(self, geom: DGeometry) -> None:
        if self.policy not in POLICIES:
            raise ReshardError(f"policy {self.policy!r} not in {POLICIES}")
        names = [p.name for p in self.presets]
        if len(set(names)) != len(names) or not names:
            raise ReshardError(f"preset names must be unique and non-empty: {names}")
        if self.boot not in names:
            raise ReshardError(f"boot preset {self.boot!r} not among {names}")
        for p in self.presets:
            p.vector(self.base).validate(geom)
            # the jit activation kernel's widest vector (distributed/utils.py
            # assert_activation_aligned_shards) must hold for every preset
            sizes = [u * (geom.inter // geom.mlp_units) for u in partition_units(geom.mlp_units, list(p.mlp))]
            if any(s % 16 for s in sizes):
                raise ReshardError(f"preset {p.name}: MLP shards {sizes} not 16-aligned")

    def to_json(self) -> str:
        d = asdict(self)
        d["presets"] = [{"name": p.name, "mlp": list(p.mlp)} for p in self.presets]
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "ReshardSpec":
        d = json.loads(raw)
        return cls(policy=str(d["policy"]), base=tuple(int(x) for x in d["base"]),
                   presets=tuple(Preset(str(p["name"]), tuple(int(x) for x in p["mlp"])) for p in d["presets"]),
                   boot=str(d["boot"]), min_gain=float(d.get("min_gain", DEFAULT_MIN_GAIN)),
                   fmt=str(d.get("fmt", "")))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def preset(self, name: str) -> Preset:
        for p in self.presets:
            if p.name == name:
                return p
        raise ReshardDivergence(f"D-RESHARD unknown preset {name!r} (spec {self.digest()})")


@dataclass(frozen=True)
class ReshardRow:
    """The one decision of one flip epoch; the same row reaches every D rank."""

    epoch: int
    preset: str
    spec_digest: str
    reason: str = ""

    def line(self) -> str:
        return (f"D-RESHARD row epoch={self.epoch} preset={self.preset} "
                f"spec={self.spec_digest} reason={self.reason}")


def shard_map(geom: DGeometry, spec: ReshardSpec, preset: str) -> Dict[str, List[Tuple[int, int]]]:
    """(offset, size) per rank of every resharded dimension under ``preset`` --
    the "small map per epoch": MLP intermediate (gate_up rows / down cols)."""
    p = spec.preset(preset)
    elems = geom.inter // geom.mlp_units
    sizes = [u * elems for u in partition_units(geom.mlp_units, list(p.mlp))]
    offs, acc = [], 0
    for s in sizes:
        offs.append((acc, s))
        acc += s
    return {"mlp.intermediate": offs}


class LeaderCursor:
    """The one decider (front / TP0 at the wake).  Epochs strictly increase."""

    def __init__(self, geom: DGeometry, cal: DCalib, spec: ReshardSpec,
                 token_share: Sequence[float]):
        spec.validate(geom)
        self.geom, self.cal, self.spec = geom, cal, spec
        self.token_share = tuple(token_share)
        self.current = spec.boot
        self.epoch = -1

    def decide(self, epoch: int, load: LoadClass) -> ReshardRow:
        if epoch <= self.epoch:
            raise ReshardDivergence(f"D-RESHARD leader epoch {epoch} not after {self.epoch}")
        if self.spec.policy == POLICY_OFF:
            name, why = self.spec.boot, "off"
        else:
            name = choose(self.geom, self.cal, self.spec.base, self.spec.presets, load,
                          self.token_share, self.current, self.spec.min_gain)
            why = load.key()
        self.epoch, self.current = epoch, name
        return ReshardRow(epoch, name, self.spec.digest(), why)


class FollowerCursor:
    """Every D rank: adopt the leader's row or crash-stop."""

    def __init__(self, geom: DGeometry, spec: ReshardSpec, rank: int):
        spec.validate(geom)
        self.geom, self.spec, self.rank = geom, spec, int(rank)
        self.epoch = -1
        self.current = spec.boot

    def adopt(self, row: Optional[ReshardRow]) -> Tuple[int, int]:
        if row is None:
            raise ReshardDivergence(f"D-RESHARD rank {self.rank}: no row for the wake after epoch {self.epoch}")
        if row.spec_digest != self.spec.digest():
            raise ReshardDivergence(f"D-RESHARD rank {self.rank}: foreign spec {row.spec_digest} "
                                    f"(mine {self.spec.digest()})")
        if row.epoch <= self.epoch:
            raise ReshardDivergence(f"D-RESHARD rank {self.rank}: replayed row epoch {row.epoch} "
                                    f"(adopted {self.epoch})")
        self.spec.preset(row.preset)  # unknown -> divergence
        self.epoch, self.current = row.epoch, row.preset
        return shard_map(self.geom, self.spec, row.preset)["mlp.intermediate"][self.rank]


# ---------------------------------------------------------------------------
# The calibrated model of the rc9 D group (DYN_D_RESHARD.md sec. 1)
# ---------------------------------------------------------------------------

#: rc9 D base vector and DCP token vector (dkr27bbar1final09260145 / dkr27bnvfp4bar1final09260231)
RC9_BASE = (58, 25, 25)
RC9_TOKEN_VECTOR = {"int8": (26, 19, 19), "nvfp4": (14, 9, 9)}
#: per-rank local KV capacity at RC9_BASE (INT8 / NVFP4 "KV pool sizing" lines)
RC9_KV_CAPACITY = {"int8": (312994, 231297, 223873), "nvfp4": (443170, 294145, 288961)}
#: the auto vector of weg2xsn420 (24.09., INT8, DFLASH): the second operating point
XSN420_BASE = (3465, 2154, 2128)

_QWEN38_27B = {"hidden_size": 5120, "num_hidden_layers": 64, "full_attention_interval": 4,
               "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
               "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 48,
               "linear_value_head_dim": 128, "intermediate_size": 17408, "vocab_size": 248320,
               "model_type": "qwen3_5_text"}


def rc9_geometry(fmt: str) -> DGeometry:
    if fmt == "int8":
        return DGeometry.from_config(_QWEN38_27B, mlp_units=1088, linear_bpp=1.0, draft_bpp=1.0)
    if fmt == "nvfp4":
        return DGeometry.from_config(_QWEN38_27B, mlp_units=136, linear_bpp=0.5625, draft_bpp=0.5625)
    raise ReshardError(f"no rc9 geometry for format {fmt!r}")


def _rc9_int8_calib() -> DCalib:
    g = rc9_geometry("int8")
    a = rank_bytes(g, Vector(RC9_BASE))
    b = rank_bytes(g, Vector(XSN420_BASE))
    # bs=1 decode compute medians (Decode rank batch ... compute/wait):
    # A = dkr27bbar1final09260145 n=415: 18.3/21.9/21.8; B = weg2xsn420 n=646: 16.3/23.8/23.0
    e0, f0 = fit_two_point(a[0], 18.3, b[0], 16.3)
    b12a, b12b = (a[1] + a[2]) / 2, (b[1] + b[2]) / 2
    e1, f1 = fit_two_point(b12a, (21.9 + 21.8) / 2, b12b, (23.8 + 23.0) / 2)
    s = [x / sum(a) for x in a]
    # D-prefill 4096 chunk at A: compute 256.8/365.2/368.4, wait floor 1584.7 (3080);
    # invariant share 0.35 (uneven_perf._PREDICT_PREFILL_INVARIANT_FRACTION)
    pc = (256.8, 365.2, 368.4)
    return DCalib(
        name="int8", e_gbs=(e0, e1, e1), f_ms=(f0, f1, f1), beta=(0.05, 0.07, 0.07),
        attn_ms_per_ktok=(0.0264, 0.0468, 0.0468), w1_ms=9.1, w_slope_ms=3.8,
        pf_fixed_ms=tuple(0.35 * c for c in pc), pf_k_ms=tuple(0.65 * c / s[r] for r, c in enumerate(pc)),
        pf_wait_ms=1584.7,
        provenance="decode: two-point fit dkr27bbar1final09260145 [58,25,25] + weg2xsn420 auto "
                   "[3465,2154,2128]; beta from bs1..6 slopes; attn from #296 depth terms (131k); "
                   "prefill: one point, invariant 0.35")


def _rc9_nvfp4_calib(int8: DCalib) -> DCalib:
    g = rc9_geometry("nvfp4")
    a = rank_bytes(g, Vector(RC9_BASE))
    s = [x / sum(a) for x in a]
    # one point (dkr27bnvfp4bar1final09260231 n=109): 14.6/17.9/17.6; E borrowed from INT8
    c = (14.6, 17.9, 17.6)
    f = tuple(c[r] - (a[r] / 1e9) * 1000.0 / int8.e_gbs[r] for r in range(3))
    pc = (294.9, 807.4, 802.3)
    return DCalib(
        name="nvfp4", e_gbs=int8.e_gbs, f_ms=f, beta=int8.beta, attn_ms_per_ktok=int8.attn_ms_per_ktok,
        w1_ms=9.2, w_slope_ms=3.8,
        pf_fixed_ms=tuple(0.35 * x for x in pc), pf_k_ms=tuple(0.65 * x / s[r] for r, x in enumerate(pc)),
        pf_wait_ms=1585.6,
        provenance="decode: one point dkr27bnvfp4bar1final09260231, E borrowed from INT8 (LOW "
                   "confidence: W4A8 on sm86 is compute-heavier than its bytes); prefill one point")


def rc9_calib(fmt: str) -> DCalib:
    i8 = _rc9_int8_calib()
    if fmt == "int8":
        return i8
    if fmt == "nvfp4":
        return _rc9_nvfp4_calib(i8)
    raise ReshardError(f"no rc9 calibration for format {fmt!r}")


def token_share(tv: Sequence[int]) -> Tuple[float, ...]:
    s = float(sum(tv))
    return tuple(t / s for t in tv)


#: MLP share of rank 0 per preset (DYN_D_RESHARD.md sec. 1.4): 'dec' = the best
#: STATIC vector over the measured D load mix (and every decode class within
#: 1 %), 'pf' = the D-prefill optimum (4096-token chunk).
RC9_PRESET_SHARES = {"int8": {"dec": 0.65, "pf": 0.77}, "nvfp4": {"dec": 0.72, "pf": 0.92}}


def rc9_presets(fmt: str, names: Sequence[str] = ("dec",)) -> Tuple[Preset, ...]:
    """Named desk presets: 'rc9' (the boot today), 'dec', 'pf'."""
    g = rc9_geometry(fmt)
    out = []
    for n in names:
        if n == "rc9":
            out.append(Preset("rc9", tuple(partition_units(g.mlp_units, list(RC9_BASE)))))
        elif n in RC9_PRESET_SHARES[fmt]:
            out.append(Preset(n, mlp_vector_for_share(g, RC9_PRESET_SHARES[fmt][n])))
        else:
            raise ReshardError(f"unknown desk preset {n!r} (rc9, dec, pf)")
    return tuple(out)


def parse_presets(raw: str, fmt: str) -> Tuple[Preset, ...]:
    """'auto' -> ('dec',); 'dec,pf' -> desk presets; 'a=707:191:190;b=...' -> explicit
    unit vectors (':'-separated, one entry per rank)."""
    raw = str(raw or "auto").strip()
    if raw == "auto":
        return rc9_presets(fmt, ("dec",))
    if "=" not in raw:
        return rc9_presets(fmt, tuple(x.strip() for x in raw.split(",") if x.strip()))
    out = []
    for item in raw.split(";"):
        if not item.strip():
            continue
        name, _, vec = item.partition("=")
        try:
            units = tuple(int(x) for x in vec.split(":"))
        except ValueError:
            raise ReshardError(f"preset {item!r}: expected name=u0:u1:u2")
        out.append(Preset(name.strip(), units))
    return tuple(out)


def armed_line(spec: ReshardSpec, where: str = "") -> str:
    ps = " ".join(f"{p.name}={','.join(map(str, p.mlp))}" for p in spec.presets)
    return (f"D-RESHARD armed policy={spec.policy} base={','.join(map(str, spec.base))} "
            f"boot={spec.boot} presets[{ps}] min_gain={spec.min_gain} fmt={spec.fmt} "
            f"spec={spec.digest()} {where}".rstrip())


def boot_capacity(fmt: str, spec: ReshardSpec) -> List[int]:
    """Per-rank local KV capacity when group D boots on the spec's boot preset: the
    measured rc9 capacity moved by the boot preset's byte delta and by the spread
    every other preset must keep free."""
    g = rc9_geometry(fmt)
    boot = spec.preset(spec.boot).vector(spec.base)
    d_boot = [b - a for a, b in zip(rank_bytes(g, Vector(spec.base)), rank_bytes(g, boot))]
    sp = spread_bytes(g, spec.base, spec.presets, spec.preset(spec.boot))
    caps, _ = kv_tokens_after(RC9_KV_CAPACITY[fmt], [a + b for a, b in zip(d_boot, sp)])
    return caps


def plan_lines(spec: ReshardSpec, fmt: str, loads: Sequence[LoadClass]) -> List[str]:
    """Desk plan per load class against TODAY's D (base vector, no MLP family
    vector).  The DCP token vector is the boot's (measured install ~ capacity)."""
    g, cal = rc9_geometry(fmt), rc9_calib(fmt)
    caps = boot_capacity(fmt, spec)
    ts = tuple(c / float(sum(caps)) for c in caps)
    ts_rc9 = token_share(RC9_TOKEN_VECTOR[fmt])
    out = []
    for ld in loads:
        t0 = round_ms(g, cal, Vector(spec.base), ld, ts_rc9)
        name = choose(g, cal, spec.base, spec.presets, ld, ts, None, spec.min_gain)
        t1 = round_ms(g, cal, spec.preset(name).vector(spec.base), ld, ts)
        out.append(f"D-RESHARD plan {ld.key()} preset={name} ms={t1:.1f} rc9_ms={t0:.1f} "
                   f"gain={100.0 * (t0 - t1) / t0:+.1f}%")
    sp = spread_bytes(g, spec.base, spec.presets, spec.preset(spec.boot))
    out.append(f"D-RESHARD kv spread_MiB={[round(x / 2**20) for x in sp]} capacity={caps} "
               f"(sum {sum(caps)}, rc9 sum {sum(RC9_KV_CAPACITY[fmt])})")
    return out


# ---------------------------------------------------------------------------
# Server advisory under uneven DCP (operator/user 26.09. ~09:35Z)
# ---------------------------------------------------------------------------
#
# The old hint ``uneven TP: restart with SGLANG_UNEVEN_MLP_VECTOR=... to raise
# the KV pool from X to ~Y`` (model_executor/model_runner_kv_cache_mixin.py
# ``_maybe_suggest_mlp_rebalance`` -> distributed/utils.py
# ``suggest_unit_rebalance_multi``) maximises the MIN-synced local capacity --
# the pool of uneven TP WITHOUT a token split.  Under uneven DCP the token
# vector follows each card's capacity and the KV cell is the same on every rank
# (all kv heads, owned tokens), so moving MLP units conserves the SUM of the
# capacities (the solver's own ceiling note says so) and only moves WHERE the KV
# lives.  There is no maxkv objective left, only speed optima: this table.

#: formats with a D cost model; 'fp8' borrows INT8's constants (same 1 B/param,
#: sm86 runs it as W8A16 Marlin: labelled BORROWED on every line).
ADVISORY_FORMATS = {"compressed-tensors": "int8", "modelopt": "nvfp4", "modelopt_fp4": "nvfp4",
                    "fp8": "fp8"}
ADVISORY_BS = (1, 2, 4, 6)
ADVISORY_CTX = (2048, 32768, 131072)
ADVISORY_SHARES = tuple(x / 100.0 for x in range(40, 96))
#: the D time mix the one-preset 'wake' vector is optimised over (sec. 1.3:
#: dkr27bbar1final09260145 agent boot, TP0 gpu-ms by class)
ADVISORY_MIX = ((LoadClass("decode", 1, 32768), 0.887), (LoadClass("decode", 2, 32768), 0.021),
                (LoadClass("decode", 4, 32768), 0.015), (LoadClass("prefill", 1, 4096), 0.077))


def advisory_format(quant_method: Optional[str]) -> Optional[str]:
    return ADVISORY_FORMATS.get(str(quant_method or "").strip().lower())


def advisory_geometry(text_cfg: Mapping, fmt: str, mlp_units: int) -> DGeometry:
    bpp = {"int8": 1.0, "nvfp4": 0.5625, "fp8": 1.0}[fmt]
    return DGeometry.from_config(text_cfg, mlp_units=mlp_units, linear_bpp=bpp, draft_bpp=bpp)


def advisory_calib(fmt: str) -> DCalib:
    return rc9_calib("int8" if fmt == "fp8" else fmt)


def _share_vectors(geom: DGeometry, n: int) -> List[Tuple[int, ...]]:
    seen, out = set(), []
    for s in ADVISORY_SHARES:
        v = mlp_vector_for_share(geom, s, n)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def best_static_mlp(geom: DGeometry, cal: DCalib, base: Sequence[int], ts: Sequence[float],
                    mix=ADVISORY_MIX) -> Tuple[int, ...]:
    """The one MLP vector a single-preset --d-reshard wake takes: minimal
    time-weighted cost over the D mix (same grid as the table)."""
    best = None
    for v in _share_vectors(geom, len(base)):
        vec = Vector(tuple(base), v)
        c = sum(w * round_ms(geom, cal, vec, ld, ts) / round_ms(geom, cal, Vector(tuple(base)), ld, ts)
                for ld, w in mix)
        if best is None or c < best[0] - 1e-12:
            best = (c, v)
    return best[1]


def speed_advisory_lines(text_cfg: Mapping, quant_method: Optional[str], base: Sequence[int],
                         current_mlp: Sequence[int], mlp_units: int,
                         token_vector: Optional[Sequence[int]]) -> List[str]:
    """The uneven-DCP replacement of the KV restart hint: per load class the
    speed-optimal MLP unit vector, its predicted gain against the RUNNING vector,
    and the vector --d-reshard wake would take.  Never raises for an unmodelled
    boot: it says why there is no table instead (one line)."""
    n = len(base)
    head = "uneven DCP: KV sum is conserved when MLP units move (token vector follows capacity)"
    fmt = advisory_format(quant_method)
    why = None
    if fmt is None:
        why = f"no D cost model for quant method {quant_method!r} (calibrated: INT8, NVFP4; FP8 borrowed)"
    elif n != 3:
        why = f"cost model is calibrated for TP=3 (5090 + 2x3080), this group has {n} ranks"
    else:
        try:
            geom = advisory_geometry(text_cfg, fmt, mlp_units)
        except (KeyError, TypeError, ValueError) as exc:
            geom, why = None, f"config is not a Qwen3.5/3.8 dense hybrid ({exc!r})"
        if geom is not None and (geom.hidden, geom.layers, geom.inter) != (5120, 64, 17408):
            why = (f"cost model is calibrated for the Qwen3.8-27B geometry, this model is "
                   f"hidden={geom.hidden} layers={geom.layers} inter={geom.inter}")
    if why is not None:
        return [f"{head}; the old 'restart with SGLANG_UNEVEN_MLP_VECTOR' KV hint does not apply. "
                f"No speed table: {why}."]
    cal = advisory_calib(fmt)
    ts = token_share(token_vector) if token_vector and len(token_vector) == n else (1.0 / n,) * n
    cur = Vector(tuple(base), tuple(int(u) for u in current_mlp))
    wake = best_static_mlp(geom, cal, base, ts)
    tag = " BORROWED(INT8 constants)" if fmt == "fp8" else ""
    lines = [f"{head}; the old 'restart with SGLANG_UNEVEN_MLP_VECTOR' KV hint does not apply. "
             f"D speed optima (HOCHRECHNUNG, weg2/d_reshard, fmt={fmt}{tag}, base={','.join(map(str, base))}, "
             f"running mlp={','.join(map(str, current_mlp))}, --d-reshard wake takes "
             f"mlp={','.join(map(str, wake))}):"]
    loads = [LoadClass("decode", b, c) for b in ADVISORY_BS for c in ADVISORY_CTX] + [LoadClass("prefill", 1, 4096)]
    vecs = _share_vectors(geom, n)
    for ld in loads:
        t_cur = round_ms(geom, cal, cur, ld, ts)
        t_best, v_best = min((round_ms(geom, cal, Vector(tuple(base), v), ld, ts), v) for v in vecs)
        t_wake = round_ms(geom, cal, Vector(tuple(base), wake), ld, ts)
        lines.append(f"  D-SPEED {ld.key():16s} mlp={','.join(map(str, v_best)):14s} "
                     f"gain={100.0 * (t_cur - t_best) / t_cur:+5.1f}% (wake mlp {100.0 * (t_cur - t_wake) / t_cur:+5.1f}%) "
                     f"ms {t_cur:.1f}->{t_best:.1f}")
    return lines


def quant_method_of(hf_config, override: Optional[str] = None) -> Optional[str]:
    """--quantization if set, else quantization_config.quant_method (dict or object,
    top level or text_config)."""
    if override:
        return str(override)
    for holder in (hf_config, getattr(hf_config, "text_config", None)):
        qc = getattr(holder, "quantization_config", None) if holder is not None else None
        if isinstance(qc, Mapping) and qc.get("quant_method"):
            return str(qc["quant_method"])
        if qc is not None and getattr(qc, "quant_method", None):
            return str(qc.quant_method)
    return None


def config_dict(cfg) -> Mapping:
    if isinstance(cfg, Mapping):
        return cfg
    to_dict = getattr(cfg, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(vars(cfg))


def uneven_dcp_token_split(dcp_size: int, tp_size: int, uneven_dcp_flag: bool, vector_active: bool) -> bool:
    """The token split is (or will be) in force: DCP over the whole TP group and
    either a non-uniform vector installed or --uneven-dcp requested.  Every input
    is rank-uniform, so the KV hint's collective is skipped on all ranks or none."""
    return int(tp_size) > 1 and int(dcp_size) == int(tp_size) and (bool(vector_active) or bool(uneven_dcp_flag))
