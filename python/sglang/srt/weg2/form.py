"""WEG2-FORM: the boot's FORM AXES -- one resolver, one line, one env, one accessor.

User order 2026-09-24 (verbatim): "da muessen schalter rein, in den code, dense,
moe, vollstaendig im vram, mit offload - oder sowas aehnliches oder mehr oder
weniger (ergruende/begruende). sonst knallts doch an jeder stelle".

WHY THIS MODULE EXISTS.  The Qwen3.8-27B form never booted on the Next-Flash
line, and every probe of 24.09. died on an NF assumption in SHARED code that
keyed on an INCIDENTAL fact instead of a declared form: a group-env pop hanging
on the else of the expert-map branch (xsn412), P's draft flags read from the
NEXTN constants (xsn414/415), a #66 instrument only the NEXTN producer prints
(xsn417) -- and, worse because silent, the NF MEASUREMENTS (the #114 prefill
transient, the #1444 D residue record, the P logs the cut and the depth are
calibrated from) read by the 27B because "newest file" was the only key
(FORM_AXES_INVENTORY.md, 24.09.).

THE AXES (each derived from the flags that actually build the argv, never
typed twice):

``arch``     dense | moe            -- the checkpoint (config ``num_experts``)
``experts``  none | resident | offload
                                    -- dense -> none; moe -> offload as soon as
                                       an expert store or a resident fraction
                                       < 1 is stated, else resident.  Not named
                                       ``weights``: ``--flip-weights resident``
                                       already means "weights never flip".
``draft``    dflash | mtp | none    -- ``--spec-form`` (DFLASH | NEXTN)
``p_draft``  compute | cold | none  -- ``--draft-kv-on-p`` x ``--dflash-produce-on-p``
                                       x ``--weg2-disable-hicache``
``kv``       paged_dcp | qsa_forma  -- Form A (a ``worker`` rank role on D)
``flip``     family | resident      -- ``--flip-weights``
``vision``   off | resident | transient -- ``--weg2-vision``

A PROFILE (``--profile``) is a named bundle of EXPECTATIONS; a ``--form-*``
flag overrides the expectation; a derived value outside the expectation is
REFUSED by name (W140 Weg2FormContradiction).  ``--form-draft`` and
``--form-p-draft`` additionally DRIVE the legacy flags they map onto when those
were not given explicitly (one switch, one writer: the resolver writes ``ns``
before :func:`launcher.apply_spec_form` reads it); an explicit legacy flag
that disagrees is refused, never silently overridden.

THE CONTRACT TO THE RANKS: ONE environment variable, :data:`FORM_ENV`,
published by the launcher into its own ``os.environ`` (so ``build_env`` and
the front inherit the same value -- the B4k pattern), and ONE accessor,
:func:`current_form`.  ``None`` means "not a weg2 boot" and every form
declaration then applies, so non-weg2 paths never change behaviour.

THE MODEL-PROFILE REGISTRY (UNIFY S3): :class:`ModelProfile` rows in
:data:`PROFILES` carry what is KNOWN about a model -- its form expectations
(:data:`PROFILE_EXPECT` is derived from them), the rank switches whose default
follows the model (:data:`PROFILE_SWITCH_DEFAULTS`, derived), per-group env
rows, and its measured constants with provenance (:func:`profile_constant`).
Ranks read the row through the published form (:func:`current_profile`).
:func:`calibration_identity` is the ONE acceptor for measured sources
(checkpoint AND form, AND the 27B line term where the row names it).

Torch-free and launcher-free on purpose: ranks import this module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.name_compat import has_marker, tolerant_compile, tolerant_rx

FORM_ENV = "SGLANG_WEG2_FORM"
FORM_LINE_TAG = "WEG2-FORM"
W_CONTRADICTION = "W140 Weg2FormContradiction"

AXES: Tuple[str, ...] = ("arch", "experts", "draft", "p_draft", "kv", "flip", "vision")
AXIS_VALUES: Dict[str, Tuple[str, ...]] = {
    "arch": ("dense", "moe"),
    "experts": ("none", "resident", "offload"),
    "draft": ("dflash", "mtp", "none"),
    "p_draft": ("compute", "cold", "none"),
    "kv": ("paged_dcp", "qsa_forma"),
    "flip": ("family", "resident"),
    "vision": ("off", "resident", "transient"),
}
#: The axes an operator can STATE (``--form-<axis>``). ``flip`` and ``vision``
#: already are single flags of their own; a second spelling would be the
#: #1358 class ("one predicate, two call sites").
STATED_AXES: Tuple[str, ...] = ("arch", "experts", "draft", "p_draft", "kv")

PROFILE_QWEN27B = "qwen27b"
PROFILE_NEXTFLASH = "nextflash"
#: The profile a caller without a published form gets -- the launcher's own
#: ``--profile`` default, so a desk caller reads what a bare launcher boots.
DEFAULT_PROFILE = PROFILE_QWEN27B


# ==========================================================================
# UNIFY S3: THE MODEL-PROFILE REGISTRY
# ==========================================================================
#
# User order 2026-09-25 ~22:15Z: unify the 27B and Next-Flash lines "so, dass
# man andere modelle auch einfach einhaengen kann" (UNIFY_PLAN.md, Schritt 3).
#
# The AXES above say what a boot BUILDS (derived from its flags, W140 on a
# contradiction). A :class:`ModelProfile` says what is KNOWN about a model:
# which forms it is expected in, which rank mechanics it runs by default
# (END anchor, mamba anchor, repack, draft sharing), its per-group env rows,
# and its MEASURED constants with their provenance. One row per model; a new
# model is a new row (plus its calibration boots), not a new ``if``.
#
# RULES OF THE TABLE
# * Values stay per model. A constant the NF line never measured is NOT
#   silently the 27B number: its NF row names the 27B measurement it borrows
#   (``measured_on=qwen27b``) and :func:`borrowed_constants` lists it, so the
#   launcher can print it (UNIFY_PLAN "Risiko (c)": P_OVERSHOOT_MIB and
#   D_OVERSHOOT_MIB act on the NF argv today).
# * Rank switches whose default follows the profile are DERIVED from the
#   profile's fields through the two tables below (:data:`END_ANCHOR_SWITCHES`,
#   :data:`MAMBA_ANCHOR_SWITCHES`) plus three direct fields -- never typed a
#   second time. An explicitly set env var always wins
#   (:func:`profile_switch_default` is only the default).
# * A field that names a mechanism not yet in this tree (27B ``trim``,
#   ``grid4096``, dynamic P chunk ...) declares the model's FORM; until its
#   migration step lands, what it does here is switch the OTHER model's
#   mechanism off. UNIFY_PLAN Schritt 7/8 bring the mechanics.

END_ANCHOR_VALUES: Tuple[str, ...] = ("tail_handoff", "trim", "none")
MAMBA_ANCHOR_VALUES: Tuple[str, ...] = ("deepest", "grid4096", "none")
DRAFT_KINDS: Tuple[str, ...] = ("dflash2", "mtp", "none")
D_LAYOUT_VALUES: Tuple[str, ...] = AXIS_VALUES["kv"]

_TAIL_SWITCHES: Tuple[str, ...] = (
    "SGLANG_WEG2_TAIL_HANDOFF",
    "SGLANG_WEG2_TAIL_ADOPT",
    "SGLANG_WEG2_TAIL_VERIFY",
    "SGLANG_WEG2_TAIL_SKIP_EXTEND",
)

#: ``end_anchor`` -> the rank switches it owns. ``tail_handoff`` = the NF
#: hand-off (H18 E1, H21 adopt + verify, H24 skip-extend; weg2/tail_handoff.py,
#: tail_adopt.py). The fold (H63, SGLANG_WEG2_ENABLE_P_TAIL_FOLD) and the multi
#: tails (H42) stay code-default OFF and arm-set, as on the NF line. ``trim`` is
#: the 27B form (P-TRIM N-1, --p-trim-end-anchor, Schritt 8); ``none`` = neither.
END_ANCHOR_SWITCHES: Dict[str, Dict[str, object]] = {
    "tail_handoff": {k: True for k in _TAIL_SWITCHES},
    "trim": {k: False for k in _TAIL_SWITCHES},
    "none": {k: False for k in _TAIL_SWITCHES},
}

#: ``mamba_anchor`` -> the rank switches it owns. ``deepest`` = NF H19
#: (weg2/mamba_arena_displace.py via unified_radix_cache._weg2_mamba_claim,
#: arena_mamba_pool.settled_anchor_slots/drop_unreferenced, arena.c
#: arena_drop_unreferenced): -1 = auto max(2, slots // 4). ``0`` = no
#: displacement -- the cap is 0, the claim is first-come and neither the
#: displacement nor arena_drop_unreferenced is reached. ``grid4096`` is the 27B
#: form (anchor every 4096, max 4 per path, inner anchors released; Schritt 8).
MAMBA_ANCHOR_SWITCHES: Dict[str, Dict[str, object]] = {
    "deepest": {"SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS": -1},
    "grid4096": {"SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS": 0},
    "none": {"SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS": 0},
}


@dataclass(frozen=True)
class Measured:
    """A constant MEASURED on one model: its value, where it was measured, and
    WHICH profile's checkpoint it was measured on (``measured_on`` differing
    from the row that carries it = a borrowed value, listed by
    :func:`borrowed_constants`)."""

    value: object
    provenance: str
    measured_on: str


@dataclass(frozen=True)
class Experts:
    """``experts.store`` none | resident | offload and the arm's residency
    vectors (per P stage / per D rank). The vectors are the ARM's values,
    printed for the record; their source is the planner (P card, D
    FRACTION-SOLVE), never this table."""

    store: str = "none"
    swap: str = ""
    store_dir: str = ""
    residency_p: Tuple[float, ...] = ()
    residency_d: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Draft:
    kind: str
    steps: int = 0
    topk: int = 0
    tokens: int = 0
    block: int = 0
    window: int = 0
    path: str = ""
    #: card | host (NF draft_park.py: D's draft parks in system RAM during P)
    park: str = "card"
    #: NF H1b (SGLANG_WEG2_DRAFT_SHARE_EMBED): the MTP head shares embed/lm_head
    share_embed: bool = False


@dataclass(frozen=True)
class Chunk:
    #: 0 = no grid (NF: the anchor grain is page_size under QSA)
    grid: int
    #: fixed | dynamic (27B --p-chunk-policy, Schritt 7)
    policy: str
    #: the step model's name (27B builtin-int8) or its source
    model: str
    #: the P chunk the profile's arm runs (tokens)
    tokens: int


@dataclass(frozen=True)
class WeightFormat:
    """One checkpoint format of a model and the kernel form per card class."""

    name: str
    sm8x: str = "native"
    sm12x: str = "native"
    note: str = ""


@dataclass(frozen=True)
class RecordKey:
    """What a measured source (record sample, P log) must share with this
    boot to count (UNIFY_PLAN L1): ``checkpoint`` (model_key), ``form`` (the
    RESIDUE_AXES of its WEG2-FORM line, else its draft), ``line`` (the boot's
    commit is an ancestor of the commit this launcher runs -- the 27B
    line_identity 76e87ac3b2), ``power_limit`` (planner/power_limit.py scales
    rates by the NVML limit; not a FILTER yet, Schritt 9)."""

    fields: Tuple[str, ...]


RECORD_KEY_FIELDS: Tuple[str, ...] = ("checkpoint", "form", "line", "power_limit")


@dataclass(frozen=True)
class ModelProfile:
    id: str
    #: WEG2-FORM expectations per stated axis (W140)
    expect: Mapping[str, Tuple[str, ...]]
    arch: str
    experts: Experts
    draft: Draft
    p_draft: str
    replayssm: bool
    ple: bool
    #: model fact (config): full | qsa
    attn: str
    #: boot choice for D: paged_dcp | qsa_forma (form axis ``kv``)
    d_layout: str
    page_size: int
    kv_dtype: str
    chunk: Chunk
    end_anchor: str
    mamba_anchor: str
    #: H81 END-anchor carrier hold (SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD)
    mamba_carrier_hold: bool
    #: H39 dense repack outside the tag pools (SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL)
    repack_outside_pool: bool
    formats: Mapping[str, WeightFormat]
    #: the P cut: pinned ratio (NF) or solved from the constants (27B)
    p_cut: str
    x_start_tokens: int
    x_ceiling_tokens: int
    idle_layout: str
    vision: str
    context_tokens: int
    records: RecordKey
    #: the #1235 early-read facts' FLAG half in both groups' argv (27B TP-D);
    #: NF drops it (Form A collapses DCP), its env half is owned per group.
    early_read_flags: bool
    #: per-group env rows (NF uneven-DCP axis, Scheibe 6a); {} = none
    group_env: Mapping[str, Mapping[str, str]]
    #: checkpoints the #114 P prefill transient support points were measured on
    prefill_transient_checkpoints: Tuple[str, ...]
    constants: Mapping[str, Measured]

    def switch_defaults(self) -> Dict[str, object]:
        """The rank switches whose default this profile sets, DERIVED."""
        out: Dict[str, object] = {}
        out.update(END_ANCHOR_SWITCHES[self.end_anchor])
        out.update(MAMBA_ANCHOR_SWITCHES[self.mamba_anchor])
        out["SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD"] = bool(self.mamba_carrier_hold)
        out["SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL"] = bool(self.repack_outside_pool)
        out["SGLANG_WEG2_DRAFT_SHARE_EMBED"] = bool(self.draft.share_embed)
        return out

    def constant(self, name: str) -> object:
        try:
            return self.constants[name].value
        except KeyError:
            raise KeyError(f"profile {self.id!r} carries no constant {name!r}") from None


def _m(value: object, provenance: str, on: str = PROFILE_QWEN27B) -> Measured:
    return Measured(value=value, provenance=provenance, measured_on=on)


#: THE 27B MEASUREMENTS, values unchanged from the launcher module head (the
#: provenance comments stay there, beside the name every reader knows).
_QWEN27B_CONSTANTS: Dict[str, Measured] = {
    "DC_MEASURED_D_5090_MIB": _m(2228, "boot weg2ls1b2 (#1233 D dormant residue, 5090)"),
    "DC_MEASURED_D_3080_MIB": _m(1922, "boot weg2ls1b2 (#1233 D dormant residue, 3080)"),
    "DC_MEASURED_D_XCHG_MIB": _m((2588, 3084, 2588), "boot weg2xsn14 (exchange arm, B4h/B4k)"),
    "P_OVERSHOOT_MIB": _m((920, 0, 512), "boot weg2ls2b2 (P awake overshoot per P ordinal)"),
    "D_OVERSHOOT_MIB": _m((489, 0, 0), "boot weg2ls4b1 (D awake overshoot, NEXTN TP-D)"),
    "P_DRAFT_RESIDENT_BUDGET_MIB": _m(
        405.2 + 1213.0, "27B NEXTN head: mtp 405.2 + embed 1213.0 MiB (#1233 fix 3, L2 resident_mib)"),
    "CALIBRATION_LAYERS": _m(64, "boot bsscale (the cut MEASURED_MS_PER_LAYER was taken under)"),
    "MEASURED_MS_PER_LAYER": _m(
        "8.10,35.16,33.59", "boot bsscale 2026-09-07 (BSSCALE_0907.md, chunk 4096 bs6, cut 32,18,14)"),
    "P_PP_STAGE_FIXED_MIB": _m("2342.0,1105.5,3518.0", "boots weg2sb5f + weg2rg6 (#1286)"),
    "P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT": _m(1.5588, "boots weg2sb5f + weg2rg6 (#1286, six rank readings)"),
    "X_RECORDED_R_D_TOKS": _m(690.0, "record 1l/1o weg2zr2 (PRE-BARLINK)"),
    "X_RECORDED_R_P_TOKS": _m(3640.0, "record 1l/1o weg2zr2 (PRE-BARLINK)"),
    "X_RECORDED_FLIP_S": _m(13.247, "record 1l/1o weg2zr2 (PRE-BARLINK)"),
    "STORE_CENSUS_PROVENANCE": _m("boot weg2sb5g W9 store census 2026-09-09T07:12:38Z", "boot weg2sb5g"),
    "STORE_CENSUS_KV_PAGES": _m(50651, "boot weg2sb5g W9 store census"),
    "STORE_CENSUS_MAMBA_BLOBS": _m(42, "boot weg2sb5g W9 store census"),
    "STORE_CENSUS_DRAFT_PAGES": _m(26040, "boot weg2sb5g W9 store census"),
    "STORE_CENSUS_KV_PAGE_BYTES": _m(32768, "boot weg2sb5g W9 store census"),
}

#: THE NF ROW. Its OWN measurement is P_DRAFT_RESIDENT_BUDGET_MIB (fnFL2v71,
#: 1da8f29f12). Every other entry is the 27B measurement the NF line has read
#: since its base 76f8debf2c, BORROWED by name -- kept (the NF argv must stay
#: byte-identical, P_OVERSHOOT/D_OVERSHOOT shape it) and listed, never mixed in
#: silently. An NF record replaces a borrowed row (UNIFY_PLAN Risiko (c)).
_NEXTFLASH_CONSTANTS: Dict[str, Measured] = dict(_QWEN27B_CONSTANTS)
_NEXTFLASH_CONSTANTS["P_DRAFT_RESIDENT_BUDGET_MIB"] = _m(
    615.7 + 1522.7,
    "boot fnFL2v71 (21.09.): packed vocab 615.7 (d84f1394fe) + INT4 g32 mtp 1522.7 MiB "
    "(a905902f47), 1da8f29f12",
    on=PROFILE_NEXTFLASH,
)


PROFILES: Dict[str, ModelProfile] = {
    PROFILE_QWEN27B: ModelProfile(
        id=PROFILE_QWEN27B,
        expect={
            "arch": ("dense",),
            "experts": ("none",),
            # the 27B runs DFLASH today and ran NEXTN for weeks: both are its forms
            "draft": ("dflash", "mtp"),
            "p_draft": ("compute", "cold", "none"),
            "kv": ("paged_dcp",),
        },
        arch="dense",
        experts=Experts(store="none"),
        draft=Draft(kind="dflash2", block=8, window=2048,
                    path="/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-DFlash2-W8-lued",
                    park="card", share_embed=False),
        # --dflash-produce-on-p off (c60a5d387f); the flag comes with Schritt 6
        p_draft="none",
        replayssm=True,
        ple=False,
        attn="full",
        d_layout="paged_dcp",
        page_size=1,
        kv_dtype="auto",
        chunk=Chunk(grid=0, policy="dynamic", model="builtin-int8", tokens=2048),
        end_anchor="trim",
        mamba_anchor="grid4096",
        mamba_carrier_hold=False,
        repack_outside_pool=False,
        formats={
            "int8": WeightFormat("int8", note="compressed-tensors W8A8"),
            "fp8": WeightFormat("fp8", sm8x="marlin", note="--fp8-uniform-marlin"),
            "nvfp4": WeightFormat("nvfp4", sm8x="w4a8", sm12x="native", note="--fp4-native-mixed (78c2f16a90)"),
            "gguf": WeightFormat("gguf", note=".gguf detection (644de86ef9)"),
        },
        p_cut="solved (MEASURED_MS_PER_LAYER, P_PP_STAGE_FIXED_MIB, CALIBRATION_LAYERS)",
        x_start_tokens=4096,
        x_ceiling_tokens=12288,
        idle_layout="pp",
        vision="transient",
        context_tokens=262144,
        records=RecordKey(fields=("checkpoint", "form", "line")),
        early_read_flags=True,
        group_env={},
        prefill_transient_checkpoints=(),
        constants=_QWEN27B_CONSTANTS,
    ),
    PROFILE_NEXTFLASH: ModelProfile(
        id=PROFILE_NEXTFLASH,
        expect={
            "arch": ("moe",),
            "experts": ("resident", "offload"),
            # Memory DRAFT-ZUORDNUNG: DFlash2 is the 27B's draft only, NF = MTP.
            "draft": ("mtp",),
            "p_draft": ("compute", "none"),
            "kv": ("qsa_forma",),
        },
        arch="moe",
        experts=Experts(store="offload", swap="platztausch", store_dir="/mnt/nf-experts",
                        residency_p=(0.332, 0.64, 0.39), residency_d=(0.06, 0.51, 0.48)),
        draft=Draft(kind="mtp", steps=3, topk=1, tokens=4, park="host", share_embed=True),
        # H25: no draft head on P, D's draft parks in system RAM
        p_draft="none",
        replayssm=True,
        ple=True,
        attn="qsa",
        d_layout="qsa_forma",
        page_size=64,
        kv_dtype="fp8_e4m3",
        chunk=Chunk(grid=0, policy="fixed", model="linear from measurement (P card)", tokens=16384),
        end_anchor="tail_handoff",
        mamba_anchor="deepest",
        mamba_carrier_hold=True,
        repack_outside_pool=True,
        formats={
            "int4-mixed": WeightFormat("int4-mixed", note="compressed-tensors AutoRound (Minachist)"),
            "nvfp4": WeightFormat("nvfp4", sm8x="w4a8", sm12x="native",
                                  note="ModelOpt; 3080 W4A8 planned (user 25.09.)"),
        },
        p_cut="pinned --pp-stage-ratio 29,11,8 --pp-attn-stage-ratio 7,3,2 (arm)",
        x_start_tokens=4096,
        x_ceiling_tokens=12288,
        idle_layout="",
        vision="off",
        context_tokens=262144,
        records=RecordKey(fields=("checkpoint", "form")),
        early_read_flags=False,
        # Scheibe 6a: the uneven-DCP axis is GROUP-OWNED on Next Flash (seam D):
        # P (PP3, tp 1) keeps 1/1 (inert), D (Form A) gets 0/0 -- an inherited 1
        # lands in RankRoleError (rank_role.py resolve_dcp_under_host_kv).
        group_env={
            "P": {"SGLANG_UNEVEN_DCP": "1", "SGLANG_UNEVEN_DCP_WEIGHTED": "1"},
            "D": {"SGLANG_UNEVEN_DCP": "0", "SGLANG_UNEVEN_DCP_WEIGHTED": "0"},
        },
        prefill_transient_checkpoints=("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",),
        constants=_NEXTFLASH_CONSTANTS,
    ),
}


#: Profile = named bundle of EXPECTED values per axis (a set). Only the stated
#: axes carry expectations; flip/vision are printed, never expected. DERIVED
#: from the registry rows.
PROFILE_EXPECT: Dict[str, Dict[str, Tuple[str, ...]]] = {
    pid: {a: tuple(v) for a, v in prof.expect.items()} for pid, prof in PROFILES.items()
}

#: UNIFY S2/S3: switches whose DEFAULT differs by model profile -- one
#: environ.py entry each, its default resolved from the published form's
#: profile (:func:`profile_switch_default`). An explicitly set env var always
#: wins; no form / a profile not listed -> the fallback the environ entry names
#: (the NF line's code default). DERIVED from the rows (``switch_defaults``):
#: SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL (H39; 27B port 3c9bfeff95 off, NF
#: d6b7d4a1d3 on), SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD (H81; the 27B alias
#: SGLANG_WEG2_MAMBA_INNER_ANCHOR_RELEASE is read in environ.py), the four NF
#: tail switches (``end_anchor``), SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS
#: (``mamba_anchor``), SGLANG_WEG2_DRAFT_SHARE_EMBED (``draft.share_embed``).
PROFILE_SWITCH_DEFAULTS: Dict[str, Dict[str, object]] = {
    pid: prof.switch_defaults() for pid, prof in PROFILES.items()
}


def profile_row(profile: Optional[str]) -> Optional[ModelProfile]:
    """The registry row of ``profile``; ``None`` for no/unknown profile."""
    return PROFILES.get(str(profile or ""))


def borrowed_constants(profile: str) -> Tuple[Tuple[str, str], ...]:
    """``(name, measured_on)`` of every constant ``profile`` carries that was
    measured on ANOTHER profile's checkpoint -- the visible form of risk (c)."""
    prof = PROFILES[profile]
    return tuple(
        (n, m.measured_on) for n, m in sorted(prof.constants.items()) if m.measured_on != prof.id
    )


def borrowed_constants_line(profile: str) -> Optional[str]:
    """One launcher line naming the borrowed constants; None when there are none."""
    got = borrowed_constants(profile)
    if not got:
        return None
    by: Dict[str, List[str]] = {}
    for n, on in got:
        by.setdefault(on, []).append(n)
    parts = "; ".join(f"from {on}: {', '.join(ns)}" for on, ns in sorted(by.items()))
    return (f"WEG2-PROFILE {profile}: {len(got)} constant row(s) BORROWED from another "
            f"model's measurement ({parts}); where the launcher reads a record or census of "
            f"THIS model (#1444, xchg census, flags) that supersedes the row")


#: Checkpoint config keys that name routed experts (top level or text_config).
_EXPERT_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
#: Launcher/extra flags whose presence states an expert layout (MoE only).
_EXPERT_FLAGS = (
    "--rank-moe-ratio",
    "--rank-moe-resident-fraction",
    "--pp-cut-expert-device-fraction",
    "--pp-cut-expert-lru-rows",
)
_EXPERT_FRACTION_FLAGS = ("--rank-moe-resident-fraction", "--pp-cut-expert-device-fraction")
_EXPERT_STORE_ENV = "SGLANG_MOE_EXPERT_STORE_DIR"
_EXPERT_FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"


class Weg2FormContradiction(RuntimeError):
    """W140: the boot's stated/expected form contradicts what its flags build."""


# --------------------------------------------------------------------------
# the value object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Weg2Form:
    arch: str
    experts: str
    draft: str
    p_draft: str
    kv: str
    flip: str
    vision: str
    profile: str = ""
    model: str = ""
    #: axis -> where the value came from (printed only, never published)
    sources: Tuple[Tuple[str, str], ...] = field(default=(), compare=False)

    def axes(self) -> Dict[str, str]:
        return {a: getattr(self, a) for a in AXES}

    def describe(self) -> str:
        """``arch=dense experts=none ...`` -- the form in one span."""
        return " ".join(f"{a}={getattr(self, a)}" for a in AXES)

    def env_value(self) -> str:
        """The :data:`FORM_ENV` payload: ``,``-separated, no spaces -- NOT
        ``;``, because the launcher's WEG2-GROUP-ENV line and ``--env-p`` /
        ``--env-d`` already use ``;`` between variables."""
        parts = [f"{a}={getattr(self, a)}" for a in AXES]
        parts.append(f"profile={self.profile}")
        parts.append(f"model={self.model}")
        return ",".join(parts)

    def line(self) -> str:
        src = "; ".join(f"{a} <- {s}" for a, s in self.sources)
        return (
            f"{FORM_LINE_TAG} {self.describe()} profile={self.profile} "
            f"model={self.model}" + (f" (sources: {src})" if src else "")
        )

    def matches(self, **want: Iterable[str]) -> bool:
        for axis, allowed in want.items():
            if getattr(self, axis) not in tuple(allowed):
                return False
        return True


def parse_form(value: str) -> Optional[Weg2Form]:
    """Inverse of :meth:`Weg2Form.env_value`; ``None`` for an unusable value."""
    if not value:
        return None
    kv: Dict[str, str] = {}
    for item in str(value).split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            kv[k.strip()] = v.strip()
    if any(a not in kv for a in AXES):
        return None
    if any(kv[a] not in AXIS_VALUES[a] for a in AXES):
        return None
    return Weg2Form(
        **{a: kv[a] for a in AXES},
        profile=kv.get("profile", ""),
        model=kv.get("model", ""),
    )


def current_form(environ: Optional[Mapping[str, str]] = None) -> Optional[Weg2Form]:
    """THE rank-side accessor. ``None`` = no weg2 form published (not a weg2
    boot, or a desk test): callers then keep their pre-form behaviour."""
    env = os.environ if environ is None else environ
    return parse_form(env.get(FORM_ENV, ""))


def profile_switch_default(name: str, fallback, environ: Optional[Mapping[str, str]] = None):
    """The default of switch ``name`` for the profile in the published form
    (:data:`PROFILE_SWITCH_DEFAULTS`); ``fallback`` without a form, without a
    profile in it, or for a profile that does not list the switch. Reads only
    the form -- whether ``name`` itself is set is the caller's question. The
    value takes ``fallback``'s type (bool switches stay bool, int stay int)."""
    form = current_form(environ)
    prof = form.profile if form is not None else ""
    val = PROFILE_SWITCH_DEFAULTS.get(prof, {}).get(name, fallback)
    if isinstance(fallback, bool):
        return bool(val)
    if isinstance(fallback, int):
        return int(val)
    return val


def current_profile(environ: Optional[Mapping[str, str]] = None) -> Optional[ModelProfile]:
    """The registry row of the published form's profile; ``None`` without a
    form or for a profile the registry does not know."""
    form = current_form(environ)
    return profile_row(form.profile) if form is not None else None


def profile_constant(
    name: str, profile: Optional[str] = None, environ: Optional[Mapping[str, str]] = None
) -> object:
    """A measured constant of ``profile`` -- by default the published form's
    profile, else :data:`DEFAULT_PROFILE` (the launcher's ``--profile``
    default). THE one reader of :attr:`ModelProfile.constants`."""
    if profile is None:
        form = current_form(environ)
        profile = form.profile if form is not None and form.profile in PROFILES else DEFAULT_PROFILE
    row = profile_row(profile)
    if row is None:
        raise KeyError(f"unknown model profile {profile!r}; known: {sorted(PROFILES)}")
    return row.constant(name)


# --------------------------------------------------------------------------
# gate declarations
# --------------------------------------------------------------------------

#: Every gate/solve that only means something for some forms, and WHICH forms.
#: A gate outside its forms logs ONE line ``<name> SKIPPED (form ...)`` and
#: does not evaluate -- instead of evaluating incidental facts (a vector
#: present or not) and printing "ENTFAELLT" for a form it was never built for.
#: A gate inside its forms prints nothing new, so NF logs stay as they are.
FORM_GATES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "#106 WEG2-STORE-GEOMETRY": {"arch": ("moe",)},
    "#107 WEG2-EXPERT-MAP": {"arch": ("moe",)},
    "#134 WEG2-EXPERT-BAND": {"arch": ("moe",)},
    "#140 PP-CUT FRACTION-SOLVE": {"arch": ("moe",)},
    "#145 D-RANK FRACTION-SOLVE": {"arch": ("moe",)},
    "H14 WAKE-CREDIT": {"arch": ("moe",)},
    "#114 P-PREFILL-TRANSIENT": {},  # keyed on the calibration model, see launcher
}


def gate_applies(name: str, form: Optional[Weg2Form]) -> bool:
    if form is None:
        return True
    want = FORM_GATES.get(name)
    if not want:
        return True
    return form.matches(**want)


def gate_skip_line(name: str, form: Optional[Weg2Form]) -> Optional[str]:
    """``None`` if the gate applies to ``form``; else the one SKIPPED line."""
    if gate_applies(name, form):
        return None
    want = " ".join(f"{a}={'|'.join(v)}" for a, v in FORM_GATES[name].items())
    return f"{name} SKIPPED (form {form.describe()}) -- declared for {want} only"


# --------------------------------------------------------------------------
# argv helpers (no argparse: the extra strings are shell-split words)
# --------------------------------------------------------------------------


def flag_values(words: Sequence[str], flag: str) -> List[str]:
    """Every value of ``flag`` in ``words`` (``--f v`` and ``--f=v``), in order."""
    out: List[str] = []
    words = list(words)
    for i, w in enumerate(words):
        if w == flag and i + 1 < len(words):
            out.append(words[i + 1])
        elif w.startswith(flag + "="):
            out.append(w.split("=", 1)[1])
    return out


def flag_given(words: Sequence[str], flag: str) -> bool:
    return any(w == flag or w.startswith(flag + "=") for w in words)


def _fractions(value: str) -> List[float]:
    out: List[float] = []
    for tok in str(value).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------
# derivations
# --------------------------------------------------------------------------


def model_key(model: str) -> str:
    """The calibration identity of a checkpoint: its directory name."""
    return os.path.basename(str(model or "").rstrip("/").strip("'\""))


def checkpoint_arch(model: str) -> Tuple[Optional[str], str]:
    """``("moe"|"dense", source)`` from the checkpoint config; ``(None, why)``
    when it cannot be read (the profile's expectation then stands, and the
    source says so)."""
    path = os.path.join(str(model or ""), "config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"checkpoint config unreadable ({path}: {type(e).__name__})"
    if not isinstance(cfg, dict):
        return None, f"checkpoint config is not an object ({path})"
    for scope, d in (("", cfg), ("text_config.", cfg.get("text_config"))):
        if not isinstance(d, dict):
            continue
        for k in _EXPERT_CONFIG_KEYS:
            v = d.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                return "moe", f"checkpoint {scope}{k}={v}"
    return "dense", "checkpoint config names no routed experts"


def expert_evidence(
    launcher_words: Sequence[str],
    extra_p: Sequence[str],
    extra_d: Sequence[str],
    env_p: Mapping[str, str],
    env_d: Mapping[str, str],
) -> Tuple[List[str], bool]:
    """``(stated expert facts, offload?)`` from the arm's OWN statements (the
    launcher argv, the extra strings, the per-group env) -- never from the
    inherited shell, which a dense boot does not own."""
    facts: List[str] = []
    offload = False
    for words, where in (
        (launcher_words, "launcher"), (extra_p, "extra_p"), (extra_d, "extra_d")
    ):
        for flag in _EXPERT_FLAGS:
            vals = flag_values(words, flag)
            if not vals:
                continue
            facts.append(f"{flag} ({where})")
            if flag in _EXPERT_FRACTION_FLAGS and any(
                x < 1.0 for v in vals for x in _fractions(v)
            ):
                offload = True
    for env, where in ((env_p, "env_p"), (env_d, "env_d")):
        if env.get(_EXPERT_STORE_ENV):
            facts.append(f"{_EXPERT_STORE_ENV} ({where})")
            offload = True
        if env.get(_EXPERT_FRACTION_ENV):
            facts.append(f"{_EXPERT_FRACTION_ENV} ({where})")
            if any(x < 1.0 for x in _fractions(env[_EXPERT_FRACTION_ENV])):
                offload = True
    return facts, offload


def derive_kv(extra_d: Sequence[str]) -> Tuple[str, str]:
    roles = flag_values(extra_d, "--rank-role")
    if roles and "worker" in [r.strip() for r in roles[-1].split(",")]:
        return "qsa_forma", f"--rank-role {roles[-1]} (extra_d): Form A"
    return "paged_dcp", "no worker rank role on D"


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------

_DRAFT_TO_SPEC = {"dflash": "DFLASH", "mtp": "NEXTN"}
_SPEC_TO_DRAFT = {v: k for k, v in _DRAFT_TO_SPEC.items()}


def _refuse(msg: str) -> None:
    raise Weg2FormContradiction(f"{W_CONTRADICTION}: {msg}")


def _apply_stated_draft(ns, words: Sequence[str], sources: Dict[str, str]) -> None:
    stated = getattr(ns, "form_draft", None)
    if not stated:
        return
    if stated == "none":
        _refuse(
            "--form-draft none is not a form this launcher builds: argv_d always "
            "carries a drafter (spec_flags) and P's producer/cold forms are keyed "
            "on it. State --form-draft dflash or mtp."
        )
    want = _DRAFT_TO_SPEC[stated]
    if flag_given(words, "--spec-form"):
        have = str(getattr(ns, "spec_form", "") or "").upper()
        if have != want:
            _refuse(
                f"--form-draft {stated} contradicts the EXPLICIT --spec-form "
                f"{have} (the two name one switch; drop one of them)"
            )
    else:
        ns.spec_form = want
        sources["draft"] = f"--form-draft {stated} (drives --spec-form {want})"


def _apply_stated_p_draft(ns, words: Sequence[str], sources: Dict[str, str]) -> None:
    stated = getattr(ns, "form_p_draft", None)
    if not stated:
        return
    spec = str(getattr(ns, "spec_form", "") or "NEXTN").upper()
    if stated == "cold" and spec != "DFLASH":
        _refuse(
            f"--form-p-draft cold needs --spec-form DFLASH (this boot resolves "
            f"{spec}): only the DFlash producer has a cold form "
            "(--dflash-produce-on-p off); the NEXTN producer computes or is absent"
        )
    want_on_p = "off" if stated == "none" else "on"
    if flag_given(words, "--draft-kv-on-p"):
        if str(getattr(ns, "draft_kv_on_p", "")) != want_on_p:
            _refuse(
                f"--form-p-draft {stated} contradicts the EXPLICIT --draft-kv-on-p "
                f"{getattr(ns, 'draft_kv_on_p', '')}"
            )
    else:
        ns.draft_kv_on_p = want_on_p
    if spec == "DFLASH" and stated in ("compute", "cold"):
        want_produce = "on" if stated == "compute" else "off"
        if flag_given(words, "--dflash-produce-on-p"):
            if str(getattr(ns, "dflash_produce_on_p", "")) != want_produce:
                _refuse(
                    f"--form-p-draft {stated} contradicts the EXPLICIT "
                    f"--dflash-produce-on-p {getattr(ns, 'dflash_produce_on_p', '')}"
                )
        else:
            ns.dflash_produce_on_p = want_produce
    sources["p_draft"] = f"--form-p-draft {stated}"


def resolve_form(
    ns,
    argv_words: Sequence[str],
    *,
    parse_group_env: Callable[[str], Dict[str, str]],
    shlex_split: Callable[[str], List[str]],
    arch_probe: Callable[[str], Tuple[Optional[str], str]] = checkpoint_arch,
) -> Weg2Form:
    """THE resolver. Called ONCE by ``launcher.main`` right after parsing and
    BEFORE ``apply_spec_form`` (``--form-draft``/``--form-p-draft`` may write
    ``ns.spec_form``/``ns.draft_kv_on_p``/``ns.dflash_produce_on_p``).

    Raises :class:`Weg2FormContradiction` (W140) by name; never guesses.
    """
    words = list(argv_words)
    profile = str(getattr(ns, "profile", PROFILE_QWEN27B) or PROFILE_QWEN27B)
    expect = {a: tuple(v) for a, v in PROFILE_EXPECT.get(profile, {}).items()}
    expect_src = {a: f"--profile {profile}" for a in expect}
    for axis in STATED_AXES:
        stated = getattr(ns, f"form_{axis}", None)
        if stated:
            if stated not in AXIS_VALUES[axis]:
                _refuse(f"--form-{axis.replace('_', '-')} {stated!r} is not one of {AXIS_VALUES[axis]}")
            expect[axis] = (stated,)
            expect_src[axis] = f"--form-{axis.replace('_', '-')} {stated}"

    sources: Dict[str, str] = {}
    _apply_stated_draft(ns, words, sources)
    _apply_stated_p_draft(ns, words, sources)

    model = str(getattr(ns, "model", "") or "")
    extra_p = shlex_split(str(getattr(ns, "extra_p", "") or ""))
    extra_d = shlex_split(str(getattr(ns, "extra_d", "") or ""))
    env_p = parse_group_env(str(getattr(ns, "env_p", "") or ""))
    env_d = parse_group_env(str(getattr(ns, "env_d", "") or ""))

    # arch: the checkpoint is the truth; unreadable -> the expectation stands.
    arch, arch_src = arch_probe(model)
    if arch is None:
        fallback = expect.get("arch", ("dense",))[0]
        arch, arch_src = fallback, f"{expect_src.get('arch', 'default')} ({arch_src})"
    sources["arch"] = arch_src

    facts, offload = expert_evidence(words, extra_p, extra_d, env_p, env_d)
    if arch == "dense":
        if facts:
            _refuse(
                f"the checkpoint is DENSE ({arch_src}) but this arm states an "
                f"expert layout: {', '.join(facts)}. Those flags drive the MoE "
                "solves (#106/#107/#140/#145) and the expert store; on a dense "
                "model they are a wrong-profile arm, not a no-op"
            )
        experts, experts_src = "none", "dense checkpoint"
    elif offload:
        experts, experts_src = "offload", ", ".join(facts)
    else:
        experts, experts_src = "resident", (
            "moe checkpoint, no expert store and no resident fraction < 1 stated"
        )
    sources["experts"] = experts_src

    spec = str(getattr(ns, "spec_form", "NEXTN") or "NEXTN").upper()
    draft = _SPEC_TO_DRAFT.get(spec)
    if draft is None:
        _refuse(f"--spec-form {spec!r} maps to no draft axis value")
    sources.setdefault("draft", f"--spec-form {spec}")

    hicache_disabled = bool(getattr(ns, "weg2_disable_hicache", False))
    on_p = str(getattr(ns, "draft_kv_on_p", "on")) == "on" and not hicache_disabled
    if not on_p:
        p_draft = "none"
        why = "--weg2-disable-hicache" if hicache_disabled else "--draft-kv-on-p off"
    elif draft == "dflash":
        produce = str(getattr(ns, "dflash_produce_on_p", "off")).lower() == "on"
        p_draft = "compute" if produce else "cold"
        why = f"--dflash-produce-on-p {'on' if produce else 'off'}"
    else:
        p_draft, why = "compute", "NEXTN producer on P (--draft-kv-on-p on)"
    sources.setdefault("p_draft", why)

    kv, kv_src = derive_kv(extra_d)
    sources["kv"] = kv_src

    flip = str(getattr(ns, "flip_weights", "family") or "family")
    vision = str(getattr(ns, "weg2_vision", "off") or "off")
    sources["flip"] = f"--flip-weights {flip}"
    sources["vision"] = f"--weg2-vision {vision}"

    derived = {"arch": arch, "experts": experts, "draft": draft, "p_draft": p_draft, "kv": kv}
    for axis in STATED_AXES:
        allowed = expect.get(axis)
        if allowed and derived[axis] not in allowed:
            _refuse(
                f"axis {axis}: this boot BUILDS {axis}={derived[axis]} "
                f"({sources.get(axis, '?')}) but {expect_src[axis]} expects "
                f"{'|'.join(allowed)}. Fix the arm, or state the form you mean "
                f"with --form-{axis.replace('_', '-')} (a stated form overrides "
                "the profile's expectation; it never rewrites what the flags build)"
            )

    return Weg2Form(
        arch=arch, experts=experts, draft=draft, p_draft=p_draft, kv=kv,
        flip=flip, vision=vision, profile=profile, model=model_key(model),
        sources=tuple((a, sources[a]) for a in AXES if a in sources),
    )


def add_form_arguments(ap) -> None:
    """The ``--form-*`` switches (all optional; unset = the profile's expectation)."""
    for axis in STATED_AXES:
        ap.add_argument(
            f"--form-{axis.replace('_', '-')}",
            dest=f"form_{axis}",
            choices=list(AXIS_VALUES[axis]),
            default=None,
            help=(
                f"WEG2-FORM axis '{axis}' ({'|'.join(AXIS_VALUES[axis])}). Unset: "
                "the --profile's expectation. Set: overrides it; the boot is "
                f"REFUSED ({W_CONTRADICTION}) when its flags build a different "
                "value. --form-draft/--form-p-draft also DRIVE --spec-form / "
                "--draft-kv-on-p / --dflash-produce-on-p when those are not given."
            ),
        )


# --------------------------------------------------------------------------
# calibration identity: which boot measured which model
# --------------------------------------------------------------------------

_FORM_LINE_MODEL_RE = re.compile(tolerant_rx(re.escape(FORM_LINE_TAG)) + r" .*?\bmodel=(\S+)")
_FORM_LINE_AXES_RE = re.compile(
    tolerant_rx(re.escape(FORM_LINE_TAG)) + r" (arch=\S+ experts=\S+ draft=\S+ p_draft=\S+ kv=\S+ "
    r"flip=\S+ vision=\S+) profile=(\S*) model=(\S+)")
_MODEL_PATH_RE = re.compile(r"--model-path[= ]'?([^\s']+)")
_SPEC_ALGO_RE = re.compile(r"--speculative-algorithm[= ]'?([A-Z_]+)")
_SERVER_ARGS_MODEL_RE = re.compile(r"\bmodel_path='([^']+)'")
#: A front log states its groups' argv within its first few hundred lines; a
#: P log its ServerArgs within the first few thousand. Bounded either way.
_SCAN_MAX_BYTES = 16 << 20


@dataclass(frozen=True)
class BootIdentity:
    """What a boot log says it ran: the checkpoint, and -- when it can tell --
    its form (a WEG2-FORM line) or at least its draft (the groups' argv)."""
    model: Optional[str]
    form: Optional[Weg2Form] = None
    draft: Optional[str] = None


def log_identity(path: str) -> BootIdentity:
    """Scan a boot log's head for its checkpoint and draft (and form line)."""
    model = draft = None
    try:
        with open(path, "rb") as f:
            seen = 0
            for raw in f:
                seen += len(raw)
                if seen > _SCAN_MAX_BYTES:
                    break
                line = raw.decode("utf-8", "replace")
                if has_marker(line, FORM_LINE_TAG):
                    m = _FORM_LINE_AXES_RE.search(line)
                    if m:
                        kv = dict(x.split("=", 1) for x in m.group(1).split())
                        form = Weg2Form(**kv, profile=m.group(2), model=m.group(3))
                        return BootIdentity(model=form.model, form=form, draft=form.draft)
                    m = _FORM_LINE_MODEL_RE.search(line)
                    if m and model is None:
                        model = m.group(1)
                if "argv" in line or "model_path=" in line:
                    if model is None:
                        m = _MODEL_PATH_RE.search(line) or _SERVER_ARGS_MODEL_RE.search(line)
                        if m:
                            model = model_key(m.group(1))
                    if draft is None and model is not None:
                        m = _SPEC_ALGO_RE.search(line)
                        if m:
                            draft = _SPEC_TO_DRAFT.get(m.group(1))
                if model is not None and draft is not None:
                    break
    except OSError:
        return BootIdentity(model=None)
    return BootIdentity(model=model, draft=draft)


def log_model(path: str) -> Optional[str]:
    """The checkpoint (``model_key``) a boot log says it ran, or ``None``."""
    return log_identity(path).model


@lru_cache(maxsize=4096)
def _group_log_model_cached(path: str, mtime: float) -> Optional[str]:
    front = None
    for suffix in (".P.log", ".D.log"):
        if path.endswith(suffix):
            front = path[: -len(suffix)] + ".front.log"
    if front and os.path.exists(front):
        got = log_model(front)
        if got:
            return got
    return log_model(path)


def group_log_model(path: str) -> Optional[str]:
    """Model of a ``*.P.log``/``*.D.log``: its sibling front log first (it
    carries the argv), the group log's own ServerArgs line second."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return _group_log_model_cached(path, mtime)


_FRONT_LOG_RE = tolerant_compile(
    r"^boot_weg2_(?P<tag>.+)_(?P<tip>[0-9a-f]{7,40})_(?P<day>\d{4})_(?P<time>\d{6})\.front\.log$"
)


@lru_cache(maxsize=8)
def _front_log_index(evidence_dir: str, _stamp: int) -> Dict[str, Tuple[str, ...]]:
    """tag -> its front logs, newest first. ONE directory listing per
    evidence-dir state instead of one glob per record sample."""
    by_tag: Dict[str, List[Tuple[str, str]]] = {}
    try:
        names = os.listdir(evidence_dir)
    except OSError:
        return {}
    for n in names:
        m = _FRONT_LOG_RE.match(n)
        if m:
            by_tag.setdefault(m.group("tag"), []).append(
                (m.group("day") + m.group("time"), os.path.join(evidence_dir, n)))
    return {t: tuple(p for _, p in sorted(v, reverse=True)) for t, v in by_tag.items()}


@lru_cache(maxsize=4096)
def _front_log_identity_cached(path: str, _mtime: float) -> BootIdentity:
    return log_identity(path)


def boot_tag_identity(tag: str, evidence_dir: str) -> BootIdentity:
    """What the boot ``tag`` (a measured-record ``boot_tag``) ran, read from
    that boot's own front log; ``model=None`` when no log names one."""
    if not tag:
        return BootIdentity(model=None)
    try:
        stamp = int(os.path.getmtime(evidence_dir))
    except OSError:
        return BootIdentity(model=None)
    for path in _front_log_index(evidence_dir, stamp).get(str(tag), ()):
        try:
            got = _front_log_identity_cached(path, os.path.getmtime(path))
        except OSError:
            continue
        if got.model:
            return got
    return BootIdentity(model=None)


def boot_tag_model(tag: str, evidence_dir: str) -> Optional[str]:
    return boot_tag_identity(tag, evidence_dir).model


#: The axes that change what a group leaves on the device and in the host
#: image (the D residue, the dormant image, the run peak): a sample measured
#: under another value of any of them is not this form's measurement.
RESIDUE_AXES: Tuple[str, ...] = ("arch", "experts", "draft", "kv", "flip")


def same_model_sample(
    model: str, evidence_dir: str, form: Optional[Weg2Form] = None
) -> Callable[[dict], bool]:
    """Predicate for measured-record samples: measured on ``model`` (and, with
    ``form``, under the same residue axes)?

    A sample whose boot names no model is NOT accepted -- an unproven
    provenance is exactly the mix-up this predicate exists to stop. A boot
    with a WEG2-FORM line is compared on :data:`RESIDUE_AXES`; an older boot
    (no form line) on its draft when its argv names one, else on its model.
    """
    want = model_key(model)

    def accept(sample: dict) -> bool:
        ident = boot_tag_identity(str(sample.get("boot_tag", "")), evidence_dir)
        if ident.model != want:
            return False
        if form is None:
            return True
        if ident.form is not None:
            return all(getattr(ident.form, a) == getattr(form, a) for a in RESIDUE_AXES)
        return ident.draft is None or ident.draft == form.draft

    return accept


def same_model_log(model: str) -> Callable[[str], bool]:
    """Predicate for ``*.P.log`` calibration sources: this boot's checkpoint?"""
    want = model_key(model)

    def accept(path: str) -> bool:
        return group_log_model(path) == want

    return accept


# --------------------------------------------------------------------------
# UNIFY S3: ONE calibration identity (UNIFY_PLAN L1)
# --------------------------------------------------------------------------
#
# Two lines built the same intent twice: the NF line filters measured sources
# by checkpoint + form (same_model_sample / same_model_log above, 9310d2893a),
# the 27B line by checkpoint + LINE (weg2/line_identity.py 76e87ac3b2: the
# boot's commit is an ancestor of the commit this launcher runs -- user order
# 2026-09-24 11:4xZ "getrennt von nf und separat fuers 27b"). Here they are ONE
# acceptor; WHICH terms apply is the profile's ``records`` row, not a second
# module: accept(source) = checkpoint AND form AND (line, when the row names
# it). ``power_limit`` is part of the declared key (planner/power_limit.py
# SCALES rates by the NVML limit) but not a filter yet -- no front log states
# its boot's limits in a form a record sample can be judged by (Schritt 9).

_BOOT_LOG_RE = tolerant_compile(
    r"^boot_weg2_(?P<tag>.+)_(?P<tip>[0-9a-f]{7,40})_(?P<day>\d{4})_(?P<time>\d{6})"
    r"\.(?P<kind>front|P|D)\.log$"
)


@lru_cache(maxsize=16)
def _repo_head(repo: str) -> Optional[str]:
    """The commit the tree ``repo`` has checked out; None = git unreadable."""
    import subprocess

    try:
        r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


@lru_cache(maxsize=16)
def _ancestor_index(repo: str, head: str) -> Optional[Dict[str, Tuple[str, ...]]]:
    """Every commit reachable from ``head``, indexed by its 7-char prefix. ONE
    git call per (repo, head) -- 27B line_identity measured 0.12 s against
    3.74 s for one ``merge-base --is-ancestor`` per boot tag (the front's
    sleep-leg gate sits inside the first flip). None = git unreadable."""
    import subprocess

    try:
        r = subprocess.run(["git", "-C", repo, "rev-list", head],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    idx: Dict[str, List[str]] = {}
    for sha in r.stdout.split():
        idx.setdefault(sha[:7], []).append(sha)
    return {k: tuple(v) for k, v in idx.items()}


def is_line_ancestor(repo: str, commit: str, head: Optional[str] = None) -> bool:
    """``commit`` (a boot's short tip, >= 7 hex) is ``head`` (default: the
    tree's HEAD) or one of its ancestors. Unreadable git or an unknown commit
    -> False: an unproven provenance is not accepted."""
    head = head or _repo_head(repo)
    c = str(commit or "").lower()
    if not head or len(c) < 7:
        return False
    idx = _ancestor_index(repo, head)
    if idx is None:
        return False
    return any(full.startswith(c) for full in idx.get(c[:7], ()))


def _boot_tag_tip(tag: str, evidence_dir: str) -> Optional[str]:
    """The commit of the newest front log of boot ``tag``; None = no such log."""
    try:
        stamp = int(os.path.getmtime(evidence_dir))
    except OSError:
        return None
    for path in _front_log_index(evidence_dir, stamp).get(str(tag), ()):
        m = _FRONT_LOG_RE.match(os.path.basename(path))
        if m:
            return m.group("tip")
    return None


@dataclass(frozen=True)
class CalibrationIdentity:
    """What a measured source must share with this boot to count: the terms
    are the profile's ``records.fields`` (:class:`RecordKey`)."""

    model: str
    evidence_dir: str
    form: Optional[Weg2Form] = None
    fields: Tuple[str, ...] = ("checkpoint", "form")
    #: the tree this launcher runs (the ``line`` term's repository)
    repo: str = ""

    @property
    def uses_line(self) -> bool:
        return "line" in self.fields and bool(self.repo)

    def describe(self) -> str:
        parts = [f"checkpoint {model_key(self.model)}"]
        if "form" in self.fields and self.form is not None:
            parts.append("form " + " ".join(f"{a}={getattr(self.form, a)}" for a in RESIDUE_AXES))
        if self.uses_line:
            parts.append(f"a boot commit that is an ancestor of {self.repo} HEAD (the line)")
        declared = [f for f in self.fields if f not in ("checkpoint", "form", "line")]
        tail = f" [declared, not filtered: {', '.join(declared)}]" if declared else ""
        return " AND ".join(parts) + tail

    def accepts_sample(self, sample: dict) -> bool:
        """A measured-record sample, judged by its ``boot_tag``'s own front log."""
        base = same_model_sample(
            self.model, self.evidence_dir, self.form if "form" in self.fields else None)
        if not base(sample):
            return False
        if not self.uses_line:
            return True
        tip = _boot_tag_tip(str((sample or {}).get("boot_tag", "") or ""), self.evidence_dir)
        return tip is not None and is_line_ancestor(self.repo, tip)

    def accepts_log(self, path: str) -> bool:
        """A ``boot_weg2_*.{front,P,D}.log``: this checkpoint (and, with the
        ``line`` term, a commit of this line, read off the log's own name)."""
        if not same_model_log(self.model)(path):
            return False
        if not self.uses_line:
            return True
        m = _BOOT_LOG_RE.match(os.path.basename(str(path)))
        return m is not None and is_line_ancestor(self.repo, m.group("tip"))


def calibration_identity(
    model: str, evidence_dir: str, form: Optional[Weg2Form], repo: str = ""
) -> CalibrationIdentity:
    """The identity of this boot, its terms from the form's profile row
    (``records.fields``); a profile the registry does not know keeps the
    checkpoint + form terms (the NF-line form of 24.09.)."""
    if not isinstance(form, Weg2Form):
        form = None  # a desk stand-in: checkpoint term only, as before
    row = profile_row(form.profile) if form is not None else None
    fields = row.records.fields if row is not None else ("checkpoint", "form")
    return CalibrationIdentity(model=str(model or ""), evidence_dir=str(evidence_dir),
                               form=form, fields=tuple(fields), repo=str(repo or ""))
