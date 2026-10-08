"""HW-AP0 1525: the MODEL-PROFILE-DRIVEN fit bound ("passt rechnerisch?").

User orders 05.10. (~20:4xZ and ~20:5xZ, verbatim): "das muss generisch sein
bzw. per flags oder env gesetzt werden" and "es muss ja nicht das int8 werden
oder das nf. es kann ja auch ein gguf werden oder das nvfp4 nur im flip modus
ohne dual". So this module knows NO model name and NO card name: it computes
from

* a MODEL PROFILE (``fit_profiles_data/<profile>.json``: layers, per-layer
  dense / expert bytes, embedding / head / draft bytes, KV heads x head dim,
  attention vs linear layers, mamba state) -- derived by :func:`derive_profile`
  from a ``config.json`` with ``model_profile.estimate_weights_from_config``,
  never typed in;
* the profile's RECORDS (``profile_records``: stage fixed posts, activations,
  D fixed / growth, mamba per slot) -- measured on the reference rig, a record
  the profile borrowed from another profile is printed as borrowed;
* the launch argv/env (positional vectors, host/worker role, LRU rows, foreign
  / non-torch context);
* the live cards (VRAM total, arch, class).

THE BOUND (a NECESSARY condition, never "it runs well"): weights that must sit
on the cards + the KV of ``kv_tokens`` tokens + the mamba slots + the posts of
the records, against ``total - residue`` per card.

* **weights**: a dense model keeps every layer on the cards; a MoE model
  (``expert_mib > 0``) streams its experts from the host store -- the floor is
  FR = 0 with the LRU rows resident (``--pp-cut-expert-lru-rows``); tables the
  checkpoint keeps on DISK (PLE / n-gram) never count.
* **P (pipeline)**: stage ``i`` on card ``i`` in the launcher's card order; its
  layer capacity is ``(total - residue - fixed(role) - activation(role)) /
  cost per layer``; the plan fits iff every stage holds at least one layer and
  the capacities add up to the layer count (roles first / middle / last, the
  middle one repeated: ``inventory_view.ROLE``).
* **D (decode)**: with a host/worker role split (``--rank-tp-ratio`` has ONE
  non-zero entry: the host holds all dense weights) the host card needs fixed
  + activation + extend growth + KV + context; every other card its worker
  posts. Without it D is tensor-parallel: the sum bound.

THREE LEVELS: ``ja`` = the strict bound (all posts) holds, ``tight`` = only
the FLOOR (the same without the mamba slots: the "17,3 GiB without / 19,3 GiB
with mamba" of the D host in 1520 A) holds, ``nein`` = even the floor fails.

WHAT IT DOES NOT PROVE (HOCHRECHNUNG != MESSUNG): every post of a card or a
model the records never measured is BORROWED and printed as such
(``HW-BORROWED/unverified``): the residue of a card class other than RTX5090 /
RTX3080 (by arch from the reference classes), the activation of a 27B run
(no record), the Dual mode (P and D resident together: not modelled, the
Flip-mode bound is printed). PCIe, host RAM, BAR1, speed: not here.

PURE: stdlib + ``model_profile`` (derive only) + ``profile_records``. No
launcher import, no NVML, CPU only.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

MIB = 1 << 20
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_profiles_data")

JA, TIGHT, NEIN = "ja", "tight", "nein"

#: Tokens of KV the release requires (user order 05.10. ~05:40Z "P muss 262k
#: Prefill-Kontext"). A flag of the CLI (``--kv-tokens``), an argument here.
KV_TOKENS_DEFAULT = 262144
#: GDN/mamba slots held by P (``P 32 Slots``, 1520 A) and by D (D-SITZE line of
#: the reference dry-run: 38 slots at n = 6). Reference-rig figures -> overridable.
P_MAMBA_SLOTS_DEFAULT = 32
D_MAMBA_SLOTS_DEFAULT = 38
#: LRU rows per MoE layer of a stage when the argv names none.
LRU_ROWS_DEFAULT = 32
#: KV dtype of the release (fp8_e4m3).
KV_DTYPE_DEFAULT = "fp8_e4m3"

#: Per card CLASS the residue a card holds that the pipeline stage can not use:
#: total minus the PP-CUT budget the launcher printed in the reference dry-run
#: (5090 32607 -> 28760, 3080 20480 -> 17512; 1520 A1). The ONLY card-bound
#: measured numbers of this module; every other class takes the value of the
#: reference class of its ARCH and says BORROWED.
P_RESIDUE_BY_CLASS: Dict[str, int] = {"RTX5090": 3847, "RTX3080": 2968}
P_RESIDUE_PROVENANCE = ("PP-CUT budget line of the reference dry-run 1520 (total - budget: "
                        "5090 3847, 3080 2968 MiB)")
#: arch -> reference class the residue is borrowed from.
RESIDUE_ARCH_TWIN: Dict[str, str] = {"sm120": "RTX5090", "sm86": "RTX3080", "sm89": "RTX3080"}


class FitProfile(msgspec.Struct, frozen=True, kw_only=True):
    """What the bound needs of a model (``fit_profiles_data/<profile>.json``)."""

    profile: str
    weight_format: str
    derived_from: str
    n_layers: int
    layer_families: Tuple[str, ...]           # "attn" | "gdn" per layer
    layer_dense_mib: Tuple[float, ...]
    layer_expert_mib: Tuple[float, ...]
    n_experts: int
    embed_mib: float
    lm_head_mib: float
    draft_mib: float
    visual_mib: float
    #: tables the checkpoint keeps on disk (PLE / n-gram): never VRAM
    disk_mib: float
    kv_bytes_per_token_per_attn_layer: Dict[str, float]
    mamba_mib_per_slot_per_linear_layer: float
    extend_rate_mib_per_row: Optional[float] = None

    @property
    def attn_layers(self) -> int:
        return sum(1 for f in self.layer_families if f == "attn")

    @property
    def linear_layers(self) -> int:
        return self.n_layers - self.attn_layers

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0 and sum(self.layer_expert_mib) > 0.0

    def resident_dense_mib(self) -> float:
        return sum(self.layer_dense_mib) + self.embed_mib + self.lm_head_mib


class FitCard(msgspec.Struct, frozen=True):
    """One live card in the launcher's card order."""

    total_mib: int
    arch: str             # "sm86" ...
    cls: str = ""         # calibration class label ("RTX5090"), "" = none
    label: str = ""


class Assumptions(msgspec.Struct, frozen=True, kw_only=True):
    """The knobs (all overridable by flags of the CLI)."""

    kv_tokens: int = KV_TOKENS_DEFAULT
    kv_dtype: str = KV_DTYPE_DEFAULT
    p_mamba_slots: int = P_MAMBA_SLOTS_DEFAULT
    d_mamba_slots: int = D_MAMBA_SLOTS_DEFAULT
    lru_rows: int = LRU_ROWS_DEFAULT
    #: class -> residue MiB overrides (``--p-residue-mib RTX5090=3847``)
    residue_mib: Dict[str, int] = {}
    dual: bool = False


class Verdict(msgspec.Struct, kw_only=True):
    level: str = JA
    #: the first thing that fails the strict bound ("" when it holds)
    first: str = ""
    margin_mib: Optional[float] = None
    lines: List[str] = []
    #: provenance marks of every borrowed / unmodelled post this verdict used
    marks: List[str] = []

    def mark(self, text: str) -> None:
        if text not in self.marks:
            self.marks.append(text)


# ---------------------------------------------------------------------------
# profile derivation (data, not code): config.json -> fit_profiles_data/*.json
# ---------------------------------------------------------------------------


def derive_profile(config_path: str, *, profile: str, derived_from: str = "") -> FitProfile:
    """The FitProfile of a ``config.json`` (``model_profile`` formulas only)."""
    from flliper.srt.pdflip import model_profile as M

    cfg, _src = M.load_config(config_path)
    t = M.text_config(cfg)
    w = M.estimate_weights_from_config(cfg)
    fams = M.layer_families(t)
    kv_heads = M._int(t.get("num_key_value_heads"), M._int(t.get("num_attention_heads")))
    head_dim = M._int(t.get("head_dim"), M._int(t.get("hidden_size")) // max(1, M._int(t.get("num_attention_heads"))))
    kv: Dict[str, float] = {}
    for name, dtype_b in (("fp8_e4m3", 1.0), ("bf16", 2.0)):
        cell, scale = M.kv_cell_bytes(kv_heads, head_dim, head_dim, dtype_b)
        kv[name] = float(cell + scale) if name.startswith("fp8") else float(cell)
    ms = M.mamba_state_bytes(t)
    n_exp = max((M._int(t.get(k)) for k in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")),
                default=0)
    return FitProfile(
        profile=profile, weight_format=str(w["format"]), derived_from=derived_from or config_path,
        n_layers=len(fams), layer_families=tuple("attn" if f == M.FAM_ATTN else "gdn" for f in fams),
        layer_dense_mib=tuple(round(b / MIB, 4) for b in w["layer_bytes"]),
        layer_expert_mib=tuple(round(b / MIB, 4) for b in w["layer_expert_bytes"]),
        n_experts=n_exp, embed_mib=round(w["embed_bytes"] / MIB, 4), lm_head_mib=round(w["lm_head_bytes"] / MIB, 4),
        draft_mib=round(w["mtp_bytes"] / MIB, 4), visual_mib=round(w["visual_bytes"] / MIB, 4),
        disk_mib=round((w["ple_bytes"] + w["ngram_bytes"]) / MIB, 4), kv_bytes_per_token_per_attn_layer=kv,
        mamba_mib_per_slot_per_linear_layer=round((ms["total"] / MIB) if ms else 0.0, 4),
        extend_rate_mib_per_row=M.extend_rate_mib_per_row(cfg))


def dump_profile(p: FitProfile) -> str:
    return json.dumps(msgspec.to_builtins(p), indent=1, sort_keys=True) + "\n"


_PROFILE_CACHE: Dict[str, Optional[FitProfile]] = {}


def load_profile(profile: str, data_dir: str = DATA_DIR) -> Optional[FitProfile]:
    """The profile's fit profile, or None ("Profil fehlt")."""
    key = f"{data_dir}|{profile}"
    if key not in _PROFILE_CACHE:
        path = os.path.join(data_dir, f"{profile}.json")
        if not os.path.isfile(path):
            _PROFILE_CACHE[key] = None
        else:
            with open(path) as fh:
                _PROFILE_CACHE[key] = msgspec.convert(json.load(fh), FitProfile)
    return _PROFILE_CACHE[key]


def with_checkpoint_mib(p: FitProfile, ckpt_mib: int) -> Tuple[FitProfile, str]:
    """``p`` with every weight post scaled so that the resident posts + draft +
    visual add up to ``ckpt_mib`` (the checkpoint size of ANOTHER weight format
    of the same architecture: layer geometry borrowed, bytes measured on disk)."""
    have = sum(p.layer_dense_mib) + sum(p.layer_expert_mib) + p.embed_mib + p.lm_head_mib + p.draft_mib + p.visual_mib
    s = float(ckpt_mib) / have if have > 0 else 1.0
    q = msgspec.structs.replace(
        p, layer_dense_mib=tuple(round(x * s, 4) for x in p.layer_dense_mib),
        layer_expert_mib=tuple(round(x * s, 4) for x in p.layer_expert_mib),
        embed_mib=round(p.embed_mib * s, 4), lm_head_mib=round(p.lm_head_mib * s, 4),
        draft_mib=round(p.draft_mib * s, 4), visual_mib=round(p.visual_mib * s, 4))
    return q, (f"HW-BORROWED/unverified: layer geometry of the {p.weight_format} profile, weight bytes scaled x{s:.3f} "
               f"to the checkpoint size {ckpt_mib} MiB")


# ---------------------------------------------------------------------------
# argv / records readers
# ---------------------------------------------------------------------------


def _scoped_tokens(argv: Sequence[str], scope: str) -> List[str]:
    """Flat tokens of ``argv`` for the group ``scope`` ("p" | "d"): the shared
    tokens plus the words inside ``--extra-<scope>=...``."""
    out: List[str] = []
    for a in argv:
        if a.startswith("--extra-p=") or a.startswith("--extra-d="):
            if a.startswith(f"--extra-{scope}="):
                out.extend(a.split("=", 1)[1].split())
            continue
        out.append(a)
    return out


def flag_vector(argv: Sequence[str], flag: str, scope: str = "d") -> Optional[List[float]]:
    toks = _scoped_tokens(argv, scope)
    if flag not in toks:
        return None
    i = toks.index(flag)
    if i + 1 >= len(toks):
        return None
    try:
        return [float(x) for x in toks[i + 1].split(",") if x != ""]
    except ValueError:
        return None


def role_value(vec: Optional[Sequence[float]], i: int, n: int, default: float = 0.0) -> float:
    """``inventory_view.ROLE`` for stage ``i`` of ``n``: first, last, and the
    middle one repeated (a vector of the reference length 3 is the table)."""
    if not vec:
        return default
    if len(vec) == n:
        return float(vec[i])
    if i == 0:
        return float(vec[0])
    if i == n - 1:
        return float(vec[-1])
    return float(vec[1] if len(vec) > 2 else vec[-1])


def _record_vec(rec) -> Optional[List[float]]:
    v = rec.value if rec is not None else None
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return [float(x) for x in v.split(",") if x != ""]
        except ValueError:
            return None
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) if x is not None else float("nan") for x in v]


def _clean(vec: Optional[List[float]]) -> Optional[List[float]]:
    if vec is None:
        return None
    return [0.0 if x != x else x for x in vec]


class _Posts:
    """The record posts of a profile + how each is marked."""

    def __init__(self, profile: str, verdict: Verdict, records_dir: Optional[str] = None):
        from flliper.srt.pdflip import profile_records as PR

        kw = {} if records_dir is None else {"records_dir": records_dir}
        try:
            self._rec = PR.constants_of(profile, **kw)
        except Exception as exc:  # noqa: BLE001 - a missing records file is a named gap
            self._rec = {}
            verdict.mark(f"records of {profile!r} unreadable ({type(exc).__name__}): all posts unmodelled")
        self._v = verdict
        self._profile = profile

    def vec(self, name: str) -> Optional[List[float]]:
        rec = self._rec.get(name)
        if rec is None:
            self._v.mark(f"{name}: no record in {self._profile!r} (post not modelled)")
            return None
        if rec.borrowed:
            self._v.mark(f"{name}: HW-BORROWED from {rec.measured_on} (record)")
        return _clean(_record_vec(rec))

    def scalar(self, name: str) -> Optional[float]:
        v = self.vec(name)
        return None if not v else v[0]


# ---------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------


def residue_mib(card: FitCard, asm: Assumptions, verdict: Verdict) -> float:
    if card.cls in asm.residue_mib:
        return float(asm.residue_mib[card.cls])
    if card.cls in P_RESIDUE_BY_CLASS:
        return float(P_RESIDUE_BY_CLASS[card.cls])
    twin = RESIDUE_ARCH_TWIN.get(card.arch)
    if twin is None:
        verdict.mark(f"residue of {card.label or card.arch}: no twin for {card.arch}, 0 MiB assumed (UNVERIFIED)")
        return 0.0
    verdict.mark(f"HW-BORROWED/unverified: P-Residuum {(card.label or card.arch).split('/')[0]} <- {twin} "
                 f"{P_RESIDUE_BY_CLASS[twin]} MiB")
    return float(P_RESIDUE_BY_CLASS[twin])


def _lru_rows(argv: Sequence[str], asm: Assumptions) -> float:
    v = flag_vector(argv, "--pp-cut-expert-lru-rows", "p")
    return float(max(v)) if v else float(asm.lru_rows)


def mamba_slot_mib(p: FitProfile, posts: "_Posts") -> float:
    """MiB of one mamba slot of one linear layer: the profile's RECORD (measured
    dtype) wins over the config's state size."""
    rec = posts.scalar("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT")
    return rec if rec is not None else p.mamba_mib_per_slot_per_linear_layer


def layer_costs_mib(p: FitProfile, asm: Assumptions, argv: Sequence[str], *, mamba: bool,
                    slot_mib: Optional[float] = None) -> List[float]:
    """MiB one layer costs a P stage: dense + (MoE: the LRU rows | dense model:
    nothing more) + the layer's KV of ``kv_tokens`` tokens + (``mamba``) its
    slots."""
    kv_b = p.kv_bytes_per_token_per_attn_layer[asm.kv_dtype]
    kv_mib = kv_b * asm.kv_tokens / MIB
    rows = _lru_rows(argv, asm)
    out: List[float] = []
    for i, fam in enumerate(p.layer_families):
        c = p.layer_dense_mib[i]
        if p.is_moe and p.n_experts:
            c += rows * (p.layer_expert_mib[i] / p.n_experts)
        elif p.layer_expert_mib[i]:
            c += p.layer_expert_mib[i]
        if fam == "attn":
            c += kv_mib
        elif mamba:
            c += (p.mamba_mib_per_slot_per_linear_layer if slot_mib is None else slot_mib) * asm.p_mamba_slots
        out.append(c)
    return out


def _fmt(x: float) -> str:
    return f"{x:.0f}"


def _check_p(p: FitProfile, cards: Sequence[FitCard], asm: Assumptions, argv: Sequence[str], posts: _Posts,
             verdict: Verdict, *, strict: bool) -> Tuple[bool, str, float]:
    """The P bound. Returns (holds, first failing text, spare MiB)."""
    n = len(cards)
    # the stage posts (fixed + activation) count in both levels; record vectors are [first, mid, last].
    fixed = posts.vec("P_PP_STAGE_FIXED_MIB")
    act = posts.vec("P_ACTIVATION_MIB")
    costs = layer_costs_mib(p, asm, argv, mamba=strict, slot_mib=mamba_slot_mib(p, posts) if strict else None)
    mean_cost = sum(costs) / len(costs)
    caps: List[float] = []
    for i, c in enumerate(cards):
        avail = c.total_mib - residue_mib(c, asm, verdict)
        avail -= role_value(fixed, i, n) + role_value(act, i, n)
        caps.append(avail / mean_cost)
    why = ""
    ok = True
    for i, cap in enumerate(caps):
        if cap < 1.0:
            ok, why = False, (f"P stage {i} on {cards[i].label or cards[i].arch} holds {cap:.1f} layers (< 1) "
                              f"at {_fmt(mean_cost)} MiB/layer")
            break
    if ok and sum(int(c) for c in caps) < p.n_layers:
        ok, why = False, (f"P stages hold {sum(int(c) for c in caps)} of {p.n_layers} layers "
                          f"(caps {', '.join(f'{c:.1f}' for c in caps)}; {_fmt(mean_cost)} MiB/layer)")
    spare = sum(caps) * mean_cost - p.n_layers * mean_cost
    return ok, why, spare


def _d_host_split(argv: Sequence[str]) -> bool:
    """The D group has a host/worker split iff exactly one rank carries the dense weights."""
    v = flag_vector(argv, "--rank-tp-ratio", "d")
    return bool(v) and sum(1 for x in v if x > 0) == 1


def _check_d(p: FitProfile, cards: Sequence[FitCard], asm: Assumptions, argv: Sequence[str], posts: _Posts,
             verdict: Verdict, *, strict: bool) -> Tuple[bool, str, float]:
    n = len(cards)
    kv_mib = p.attn_layers * p.kv_bytes_per_token_per_attn_layer[asm.kv_dtype] * asm.kv_tokens / MIB
    mamba = (p.linear_layers * mamba_slot_mib(p, posts) * asm.d_mamba_slots) if strict else 0.0
    foreign = flag_vector(argv, "--d-foreign-context-mib", "p") or flag_vector(argv, "--d-foreign-context-mib", "d")
    nontorch = flag_vector(argv, "--d-nontorch-mib", "p") or flag_vector(argv, "--d-nontorch-mib", "d")
    if foreign is None or nontorch is None:
        verdict.mark("D context (foreign + non-torch): no vector in the launch argv, 0 MiB assumed (UNVERIFIED)")

    def ctx(i: int) -> float:
        return role_value(foreign, i, n) + role_value(nontorch, i, n)

    if _d_host_split(argv):
        d_fixed = posts.vec("D_FIXED_MIB")
        if d_fixed is None:
            # no record: the host holds every dense post + embedding/head + draft
            host_w = p.resident_dense_mib() + p.draft_mib
            verdict.mark("D_FIXED_MIB: derived from the model profile (dense + embed/head + draft)")
            d_fixed = [host_w, 0.0, 0.0]
        d_act = posts.vec("D_ACTIVATION_MIB")
        d_grow = posts.vec("D_EXTEND_GROWTH_MIB")
        # worker posts from the record's worker entries (repeated for more workers)
        for i, c in enumerate(cards):
            if i == 0:
                need = d_fixed[0] + kv_mib + mamba + ctx(0) + (d_act[0] if d_act else 0.0) + (d_grow[0] if d_grow else 0.0)
                role = "D host"
            else:
                wi = min(i, len(d_fixed) - 1)
                need = d_fixed[wi] + ctx(i)
                if d_grow:
                    need += d_grow[min(i, len(d_grow) - 1)]
                role = "D worker"
            if need > c.total_mib:
                return False, (f"{role} {c.label or c.arch}: needs {_fmt(need)} MiB (fixed + "
                               f"{'KV ' + _fmt(kv_mib) + ' + ' if i == 0 else ''}context {_fmt(ctx(i))} + "
                               f"activation/growth{' + mamba ' + _fmt(mamba) if strict else ''}), "
                               f"card has {c.total_mib}"), c.total_mib - need
        host_spare = cards[0].total_mib - (d_fixed[0] + kv_mib + mamba + ctx(0)
                                           + (d_act[0] if d_act else 0.0) + (d_grow[0] if d_grow else 0.0))
        return True, "", host_spare
    # tensor-parallel D: the sum bound
    w = p.resident_dense_mib() + p.draft_mib
    if p.is_moe:
        w += _lru_rows(argv, asm) / max(1, p.n_experts) * sum(p.layer_expert_mib)
    else:
        w += sum(p.layer_expert_mib)
    need = w + kv_mib + mamba + sum(ctx(i) for i in range(n))
    have = float(sum(c.total_mib for c in cards))
    if need > have:
        return False, f"D (TP{n}): weights + KV {_fmt(kv_mib)} + context need {_fmt(need)} MiB, cards hold {_fmt(have)}", have - need
    return True, "", have - need


def evaluate(p: FitProfile, cards: Sequence[FitCard], *, argv: Sequence[str] = (),
             asm: Optional[Assumptions] = None, records_profile: Optional[str] = None,
             records_dir: Optional[str] = None) -> Verdict:
    """The verdict for ``cards`` (launcher card order). ``records_profile``
    defaults to ``p.profile``."""
    asm = asm or Assumptions()
    v = Verdict()
    if not cards:
        v.level, v.first = NEIN, "no cards"
        return v
    posts = _Posts(records_profile or p.profile, v, records_dir)
    if asm.dual:
        v.mark("Dual (P and D resident together): NOT modelled, the Flip-mode bound is printed")
    if p.disk_mib:
        v.lines.append(f"{_fmt(p.disk_mib)} MiB of tables stay on disk (not VRAM)")
    results = {}
    for strict in (True, False):
        okp, whyp, sp = _check_p(p, cards, asm, argv, posts, v, strict=strict)
        okd, whyd, sd = _check_d(p, cards, asm, argv, posts, v, strict=strict)
        results[strict] = (okp, whyp, sp, okd, whyd, sd)
    okp, whyp, sp, okd, whyd, sd = results[True]
    fp_ok, fp_why, _fsp, fd_ok, fd_why, _fsd = results[False]
    v.margin_mib = min(sp, sd)
    if okp and okd:
        v.level = JA
        return v
    v.first = whyp if not okp else whyd
    if fp_ok and fd_ok:
        v.level = TIGHT
        return v
    v.level = NEIN
    v.first = fp_why if not fp_ok else fd_why
    return v
