"""Guided configuration wizard (task #270), v1.

Three steps and an expert view. The user says what they are serving and what
they want most of; the wizard answers with every deployment family this rig
can carry, every family it cannot, and the reason in both cases.

The five target quantities
--------------------------
``max_kv``, ``max_decode``, ``max_prefill``, ``max_parallel`` and ``ttft``.
The first four come from the working points that already exist
(``lever_profiles``); the fifth has its own logic per family, because time to
first token is the one quantity a topology change moves for a reason the other
four cannot express.

TTFT is always a PAIR
---------------------
Idle and under load. On this rig the same monolithic server answers in
0.257 s with nothing else running and 0.604 s with three decode streams in
flight (#212). Load on the serving cards is the actual adversary, so a single
TTFT number would hide the term that decides whether moving prefill elsewhere
is worth anything. Every family therefore reports both, and the wizard never
prints one without the other.

Undisturbedness is its own quantity
-----------------------------------
Moving prefill off the decode cards removes the decode spike (worst
inter-token time 6.54 -> 3.22 ms, #212). That is a real gain and it is not a
TTFT gain — on this rig the satellite arm COSTS 2.29 s of TTFT while removing
the spike. Reporting them separately is what lets a reader choose.

Nothing here measures
---------------------
Every number is either read from a study already on disk (``measured``),
derived by arithmetic over such numbers (``estimate``), or reported as
``absent`` with the study that would produce it. The vocabulary is
``bench_factors``' — this module does not invent a fourth word. The roofline
supplies the level of a throughput figure and never ranks one split against
another (#216); the cost model supplies the move.

The rejected register (``planner/rejected.py``) is consulted before a family
is offered. A combination the project already settled is never proposed, and
a blocked one is shown with its verdict rather than omitted — a family that
disappears teaches nothing.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner import rejected as rejmod
from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED

__all__ = [
    "TARGETS",
    "TARGET_KEYS",
    "FAMILIES",
    "FAMILY_KEYS",
    "SPILL_STATES",
    "LOCALITIES",
    "USAGE_PATTERNS",
    "ANCHORS",
    "TargetSpec",
    "FamilySpec",
    "cell",
    "ttft_pair",
    "link_gate",
    "build_matrix",
    "build_command",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TargetSpec:
    """One thing a user can want most of."""

    key: str
    label: str
    unit: str
    higher_is_better: bool
    #: The working point in ``lever_profiles.PROFILES`` that maximises it, or
    #: None when the quantity is not one of the slider's axes.
    profile_key: Optional[str]
    #: The metric key inside that working point, when there is one.
    metric_key: Optional[str]
    question: str


TARGETS: Tuple[TargetSpec, ...] = (
    TargetSpec(
        key="max_kv",
        label="context that fits",
        unit="tokens",
        higher_is_better=True,
        profile_key="max_context",
        metric_key="kv_tokens",
        question="How long a conversation can one session hold?",
    ),
    TargetSpec(
        key="max_decode",
        label="decode",
        unit="tok/s",
        higher_is_better=True,
        profile_key="max_decode",
        metric_key="decode_tok_s",
        question="How fast do tokens come out once generation is running?",
    ),
    TargetSpec(
        key="max_prefill",
        label="prefill",
        unit="tok/s",
        higher_is_better=True,
        profile_key="max_prefill",
        metric_key="prefill_tok_s",
        question="How fast is a prompt read?",
    ),
    TargetSpec(
        key="max_parallel",
        label="parallel requests",
        unit="sessions",
        higher_is_better=True,
        profile_key="balanced",
        metric_key="sessions",
        question="How many requests can be in flight at the target context?",
    ),
    TargetSpec(
        key="ttft",
        label="time to first token",
        unit="s",
        higher_is_better=False,
        profile_key=None,
        metric_key=None,
        question="How long before the first token appears -- idle, and under load?",
    ),
)

TARGET_KEYS: Tuple[str, ...] = tuple(t.key for t in TARGETS)

#: The spill axis of the matrix.
SPILL_STATES: Tuple[str, ...] = ("off", "on")
#: The locality axis of the matrix.
LOCALITIES: Tuple[str, ...] = ("local", "network")

#: What the prompts look like. Warm prefixes beat every topology lever on
#: TTFT, so the answer changes which families are worth showing first.
USAGE_PATTERNS: Tuple[str, ...] = ("fresh", "recurring")


# ---------------------------------------------------------------------------
# Measured anchors
# ---------------------------------------------------------------------------
#
# Every figure below was measured on THIS rig pair in the #212 satellite
# study (Qwen3.5-2B fp16, rig 1 = RTX 5090 decode, rig 2 = RTX 2080 Ti
# prefill, 40G RoCE). They are the inputs the TTFT arithmetic runs over; the
# arithmetic's output is an estimate, not a measurement, and is labelled that
# way. The study is named at every use so a number measured on a 2B model is
# never mistaken for a number measured on the model in the form.

#: Prefill rate the serving card reached with three decode streams in flight,
#: as a fraction of its idle rate (10 850 / 25 400 tok/s).
LOADED_PREFILL_RATE_FRACTION = 10850.0 / 25400.0

#: Monolithic TTFT, idle and under the same three streams.
ANCHOR_TTFT_IDLE_S = 0.257
ANCHOR_TTFT_LOADED_S = 0.604

#: Worst inter-token time of the running decodes, with prefill on the serving
#: cards and with prefill moved off them.
ANCHOR_DECODE_SPIKE_MS = 6.54
ANCHOR_DECODE_SPIKE_OFFLOADED_MS = 3.22
#: Median inter-token time, same two cases.
ANCHOR_DECODE_MEDIAN_MS = 3.52
ANCHOR_DECODE_MEDIAN_OFFLOADED_MS = 3.18

#: Bootstrap handshake + scheduling + the first decode step of a PD handover.
ANCHOR_PD_HANDSHAKE_S = 0.136

#: The satellite card's measured prefill rate, and the line it was measured on.
ANCHOR_SATELLITE_PREFILL_TOK_S = 2385.0
ANCHOR_LINK_GBS_40G = 1930.0 / 1000.0  # 1930 MB/s, one iperf3 stream
ANCHOR_LINK_GBS_1GBE = 0.105  # 105 MB/s, the discarded LAN route

ANCHOR_STUDY = (
    "the #212 prefill-satellite study (Qwen3.5-2B fp16, 5090 decode, "
    "2080 Ti prefill, 40G RoCE, 2026-07-28)"
)

ANCHORS: Dict[str, Any] = {
    "study": ANCHOR_STUDY,
    "ttft_idle_s": ANCHOR_TTFT_IDLE_S,
    "ttft_loaded_s": ANCHOR_TTFT_LOADED_S,
    "loaded_prefill_rate_fraction": LOADED_PREFILL_RATE_FRACTION,
    "decode_spike_ms": ANCHOR_DECODE_SPIKE_MS,
    "decode_spike_offloaded_ms": ANCHOR_DECODE_SPIKE_OFFLOADED_MS,
    "decode_median_ms": ANCHOR_DECODE_MEDIAN_MS,
    "decode_median_offloaded_ms": ANCHOR_DECODE_MEDIAN_OFFLOADED_MS,
    "pd_handshake_s": ANCHOR_PD_HANDSHAKE_S,
    "satellite_prefill_tok_s": ANCHOR_SATELLITE_PREFILL_TOK_S,
    "link_gbs_40g": ANCHOR_LINK_GBS_40G,
    "link_gbs_1gbe": ANCHOR_LINK_GBS_1GBE,
    "satellite_ttft_s": 2.892,
    "satellite_compute_share": 0.935,
    "satellite_transport_share": 0.018,
}


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    """One way of laying a model out over the hardware.

    ``spill`` and ``locality`` list the axis values this family can take at
    all. A value outside them is not hidden: the matrix emits the cell with
    the reason it cannot exist, because "this combination is impossible" is
    an answer and an empty square is not.
    """

    key: str
    label: str
    #: One sentence: what this family is for.
    purpose: str
    #: The longer explanation, shown when the family is opened.
    explain: str
    spill: Tuple[str, ...]
    locality: Tuple[str, ...]
    #: Register tags this family contributes to the blocklist check.
    tags: Tuple[str, ...] = ()
    #: "built" (code exists and has booted), "partial" (some slice exists),
    #: "design" (a design only). A non-built family is shown with what is
    #: missing rather than offered.
    status: str = "built"
    status_reason: str = ""
    #: Minimum number of local cards.
    min_local_cards: int = 1
    #: The ``flags.profiles`` preset this family launches from, when there is
    #: one. None means the command is assembled from a recipe instead.
    profile_kind: Optional[str] = None
    #: Runbook section holding the literal recipe.
    recipe: str = ""

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "purpose": self.purpose,
            "explain": self.explain,
            "spill": list(self.spill),
            "locality": list(self.locality),
            "tags": list(self.tags),
            "status": self.status,
            "status_reason": self.status_reason,
            "min_local_cards": self.min_local_cards,
            "profile_kind": self.profile_kind,
            "recipe": self.recipe,
        }


FAMILIES: Tuple[FamilySpec, ...] = (
    FamilySpec(
        key="solo_tp",
        label="Solo TP (even shards)",
        purpose="One rank per card, every rank the same size. Stock sglang.",
        explain=(
            "Every rank holds an identical shard, so the smallest card sets "
            "the budget for all of them. On a rig whose cards differ this is "
            "the configuration that throws capacity away, and seeing how "
            "much is the point of showing it. At one card it is simply the "
            "single-GPU server, and there it is the reference arm every "
            "byte-identity check is run against."
        ),
        spill=("off", "on"),
        locality=("local", "network"),
        tags=("solo-tp",),
        profile_kind="normal-tp",
        recipe="runbook 4.5 (TP=1) / 4.3 (cross-rig)",
    ),
    FamilySpec(
        key="uneven_tp_dcp",
        label="Uneven TP + weighted DCP",
        purpose=(
            "Shards sized to each card, KV split by token ownership. The "
            "fork's main path."
        ),
        explain=(
            "Weight shards follow the cards' budgets and the KV pool is "
            "split by tokens rather than by heads, so a larger card carries "
            "proportionally more of both. This is the only path on which "
            "speculation and DCP are supported together. The slowest rank "
            "still sets the clock, so the split loads the weakest card the "
            "most and per-rank timing is what says whether it worked."
        ),
        spill=("off", "on"),
        locality=("local", "network"),
        tags=("uneven-tp", "uneven-dcp"),
        min_local_cards=2,
        profile_kind="uneven-max-tokens",
        recipe="runbook 4.1",
    ),
    FamilySpec(
        key="pd_disjoint",
        label="PD disaggregation, disjoint cards",
        purpose=(
            "A prefill server and a decode server on separate cards, KV "
            "handed over."
        ),
        explain=(
            "Prefill runs on its own cards and pushes the KV (and, for a "
            "hybrid model, the GDN state) to the decode server, which "
            "continues without recomputing a token. The decode cards stop "
            "being interrupted by prefill chunks, which is the whole point. "
            "It costs speculation: this fork disables the speculative "
            "algorithm in disaggregation mode, so every PD figure is a "
            "no-spec figure and must not be put beside a NEXTN one."
        ),
        spill=("off",),
        locality=("local", "network"),
        tags=("pd",),
        min_local_cards=2,
        profile_kind="normal-tp",
        recipe="runbook 4.8",
    ),
    FamilySpec(
        key="pd_rank_reuse",
        label="PD with rank reuse (overlapping groups)",
        purpose=(
            "One rank belongs to both groups, so the fast group costs no "
            "second copy of its weights."
        ),
        explain=(
            "Two groups share a rank. On this rig the sketch is a FAST group "
            "{tp1, tp2} holding the whole model on the 5090 and a BIG group "
            "{tp1, tp3, tp4} spanning all three cards; tp1 is a member of "
            "both. The ratios nest -- BIG 6,1,1 against FAST 6,2 -- so tp1's "
            "weight shard is byte-identical in both groups and only tp2's "
            "2/8 is duplicated, which is exactly tp3+tp4 together. What tp1 "
            "does pay twice is GDN state: it needs a state pool for each "
            "group, about 1.5 MiB per head share per session per layer set. "
            "That is VRAM, not a correctness problem. A session moving from "
            "FAST to BIG leaves tp1's 6/8 share where it is and migrates "
            "only tp2's 2/8 to tp3 and tp4; KV still needs the token "
            "re-sharder, which is the piece that does not exist yet."
        ),
        spill=("off",),
        locality=("local",),
        tags=("pd", "rank-reuse"),
        status="design",
        status_reason=(
            "a design sketch, not commissioned: the prerequisite is the "
            "geometry re-sharder (KV token split + GDN head split), which is "
            "not built. Two further conditions are already known: a second "
            "process on the shared card is co-location and needs an NCCL "
            "runtime >= 2.30, and the 5090 at 27B-FP8 is already near 31.2 "
            "GiB with a 6/8 shard before a second state pool"
        ),
        min_local_cards=2,
        recipe="DESIGN_107",
    ),
    FamilySpec(
        key="extra_solo_session",
        label="Extra solo server beside the main group",
        purpose=(
            "A second, independent server co-resident on one of the cards."
        ),
        explain=(
            "A separate process with its own weights, its own KV pool and no "
            "share in the main group's collective clock. Nothing it does "
            "waits on a barrier, which is what makes it attractive for a "
            "small side model. It pays for that out of the main group's VRAM "
            "budget on the card it lands on, and the co-residence ledger "
            "charges it honestly: rank 0's own posts are weights, the mamba "
            "state pool, the speculative intermediate state, the prefill "
            "activation reserve and the dequant scratch, and only what is "
            "left after all of them is KV. A budget derived from the weight "
            "shard alone is about 2.5 GiB optimistic."
        ),
        spill=("off", "on"),
        locality=("local",),
        tags=("co-residence",),
        min_local_cards=1,
        profile_kind="single",
        recipe="runbook 4.5 / 4.6 on the chosen card",
    ),
    FamilySpec(
        key="satellite_prefill",
        label="Prefill satellite on a second rig",
        purpose=(
            "A second machine does the prefill and ships the KV over the "
            "network."
        ),
        explain=(
            "The satellite is a PD prefill arm on another host. It buys "
            "undisturbedness on the serving cards; whether it buys TTFT "
            "depends entirely on how its prefill rate compares with the "
            "serving cards' rate UNDER LOAD. Measured here, it does not: the "
            "2080 Ti spends 2.716 s of a 2.892 s answer computing, 93.5 % of "
            "the total, against 53 ms of transport. A stronger satellite "
            "card is the field where this turns positive; a faster line is "
            "not, because the line is already 1.8 % of the answer."
        ),
        spill=("off",),
        locality=("network",),
        tags=("pd", "satellite"),
        min_local_cards=1,
        profile_kind="single",
        recipe="runbook 4.8",
    ),
    FamilySpec(
        key="pp_cross_rig",
        label="Pipeline parallelism across rigs",
        purpose="Layer ranges split over two machines.",
        explain=(
            "Stage 0 holds the first layer range and the embedding, the last "
            "stage holds the tail and the lm_head, and activations cross the "
            "stage boundary once per microbatch instead of a collective per "
            "layer. Intra-rig this works and has booted: 44,20 over two "
            "cards runs at 44.2 tok/s with full decode CUDA graphs. Two "
            "things ride along -- --disable-overlap-schedule is mandatory "
            "and speculation is refused by a hard assert, so every PP figure "
            "is a no-spec figure -- and one is still open: the token count "
            "is min-reduced across the world, so the deeper stage sets it "
            "for everyone and a short stage's spare capacity is discarded."
        ),
        spill=("off",),
        locality=("network",),
        tags=("pipeline-parallel",),
        profile_kind="single",
        status="partial",
        status_reason=(
            "slice 1 (intra-rig, uneven layer split) is merged and has "
            "booted; the cross-rig stage boundary is slice 2 and is not "
            "built"
        ),
        min_local_cards=1,
        recipe="runbook 4.7 (intra-rig)",
    ),
)

FAMILY_KEYS: Tuple[str, ...] = tuple(f.key for f in FAMILIES)

_FAMILY_BY_KEY: Dict[str, FamilySpec] = {f.key: f for f in FAMILIES}


# ---------------------------------------------------------------------------
# Labelled cells
# ---------------------------------------------------------------------------


def cell(
    value: Optional[float],
    provenance: str,
    basis: str,
    *,
    unit: str = "",
    study: Optional[str] = None,
) -> dict:
    """One number with the label that says where it came from.

    An ``absent`` cell never carries a value: the whole point of the third
    label is to distinguish "nobody measured this" from "this is zero". A
    caller that passes a value with ``ABSENT`` gets the value dropped, so the
    invariant holds at the one place it is written rather than at every call
    site.
    """
    if provenance == ABSENT:
        value = None
    return {
        "value": value,
        "available": value is not None,
        "provenance": provenance,
        "basis": basis,
        "unit": unit,
        "study": study,
    }


def _absent(basis: str, unit: str = "", study: Optional[str] = None) -> dict:
    return cell(None, ABSENT, basis, unit=unit, study=study)


# ---------------------------------------------------------------------------
# TTFT
# ---------------------------------------------------------------------------


def ttft_pair(
    prefill_tok_s: Optional[float],
    context_tokens: int,
    *,
    loaded_fraction: float = LOADED_PREFILL_RATE_FRACTION,
    added_s: float = 0.0,
    basis_extra: str = "",
) -> Dict[str, dict]:
    """Time to first token, idle and under load, as the pair it has to be.

    ``prefill_tok_s`` is the idle rate; the loaded rate is that rate times
    ``loaded_fraction``, which on this rig is the measured 10 850 / 25 400
    ratio at three concurrent decode streams. ``added_s`` is whatever the
    family adds on top of compute (a handover, a transport), already summed.

    Both halves come back as ``estimate``: the inputs are measured, the
    division is not a measurement.
    """
    basis_head = (
        "context / prefill rate, plus what the family adds. The loaded rate "
        f"is the idle rate x {loaded_fraction:.3f}, the ratio measured in "
        + ANCHOR_STUDY
    )
    basis = basis_head + ((". " + basis_extra) if basis_extra else "")
    if not prefill_tok_s or prefill_tok_s <= 0:
        miss = (
            "no prefill rate for this configuration, so neither half of the "
            "pair can be stated. The rate comes from the working points; "
            "measure it directly with the split_probe job."
        )
        return {"idle": _absent(miss, "s"), "loaded": _absent(miss, "s")}
    ctx = max(int(context_tokens), 1)
    idle = ctx / float(prefill_tok_s) + added_s
    loaded = ctx / (float(prefill_tok_s) * loaded_fraction) + added_s
    return {
        "idle": cell(idle, ESTIMATE, basis, unit="s", study=ANCHOR_STUDY),
        "loaded": cell(loaded, ESTIMATE, basis, unit="s", study=ANCHOR_STUDY),
    }


def _satellite_ttft(
    context_tokens: int,
    satellite_tok_s: Optional[float],
    transport_s: Optional[float],
) -> Dict[str, dict]:
    """TTFT of a satellite arm: the satellite's own compute, plus the wire,
    plus the handover. The serving cards' load does not enter it -- that is
    the property being bought."""
    if not satellite_tok_s or satellite_tok_s <= 0:
        miss = (
            "the satellite's prefill rate is not measured. It is the term "
            "that decides this family (93.5 % of the answer in "
            + ANCHOR_STUDY
            + "), so without it no TTFT is stated. Run the pair matrix "
            "against that host, or supply the rate."
        )
        return {"idle": _absent(miss, "s"), "loaded": _absent(miss, "s")}
    ctx = max(int(context_tokens), 1)
    compute = ctx / float(satellite_tok_s)
    total = compute + float(transport_s or 0.0) + ANCHOR_PD_HANDSHAKE_S
    basis = (
        "satellite compute (context / its prefill rate) + transport over the "
        "measured link + the "
        f"{ANCHOR_PD_HANDSHAKE_S:.3f} s handover measured in " + ANCHOR_STUDY
        + ". The pair does not split idle from loaded because the serving "
        "cards' load no longer enters it -- that is what this family buys."
    )
    c = cell(total, ESTIMATE, basis, unit="s", study=ANCHOR_STUDY)
    return {"idle": dict(c), "loaded": dict(c)}


# ---------------------------------------------------------------------------
# The link gate
# ---------------------------------------------------------------------------


def link_gate(
    kv_bytes_per_token: Optional[float],
    state_bytes: Optional[float],
    link_gbs: Optional[float],
    contexts: Sequence[int],
    *,
    link_source: str = "",
) -> dict:
    """What a handover costs on the wire, per context length.

    A KV handover moves ``kv_bytes_per_token x context`` plus a fixed state
    block that does not grow with the prompt. Divided by the measured link
    rate that is a transport time; set against the family's total it is the
    share of the answer the network is responsible for. On the 40G line here
    that share was 1.8 %; the same 98 MiB over 1 GbE would have been about
    30 %, which is the number that says whether a slower line changes the
    verdict or merely the arithmetic.
    """
    if not kv_bytes_per_token or not link_gbs or link_gbs <= 0:
        return {
            "available": False,
            "provenance": ABSENT,
            "reason": (
                "needs the model's KV bytes per token and a measured link "
                "rate. The KV cell comes from the plan's balance report, the "
                "link rate from the card-probe pair matrix; run the pair "
                "matrix against the remote host to fill it."
            ),
            "rows": [],
        }
    rows = []
    fixed = float(state_bytes or 0.0)
    rate = float(link_gbs) * 1e9
    for ctx in contexts:
        payload = float(kv_bytes_per_token) * int(ctx) + fixed
        rows.append(
            {
                "context_tokens": int(ctx),
                "bytes": payload,
                "mib": payload / (2**20),
                "seconds": payload / rate,
            }
        )
    return {
        "available": True,
        "provenance": ESTIMATE,
        "link_gbs": float(link_gbs),
        "link_source": link_source or "unstated",
        "kv_bytes_per_token": float(kv_bytes_per_token),
        "state_bytes": fixed,
        "basis": (
            "model geometry (KV bytes per token from the plan's balance "
            "report, plus the length-independent state block) divided by the "
            "measured link rate"
        ),
        "counterfactual_1gbe_gbs": ANCHOR_LINK_GBS_1GBE,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def _profile_row(profiles: dict, key: str) -> Optional[dict]:
    for row in profiles.get("profiles") or []:
        if row.get("key") == key and row.get("resolved"):
            return row
    return None


def _metric(row: Optional[dict], key: str) -> Optional[dict]:
    if not row:
        return None
    for m in row.get("metrics") or []:
        if m.get("key") == key:
            return m
    return None


def _base_targets(profiles: dict) -> Dict[str, dict]:
    """The four capacity/throughput targets of the base topology, taken from
    the working points that already priced them.

    Every one of them is a planner figure, so every one is an ``estimate``;
    the working point's own ``basis`` prose is carried through unchanged so
    the wizard and the working-point table cannot disagree about what a
    number is.
    """
    out: Dict[str, dict] = {}
    for t in TARGETS:
        if t.profile_key is None:
            continue
        m = _metric(_profile_row(profiles, t.profile_key), t.metric_key or "")
        if not m or not m.get("available"):
            out[t.key] = _absent(
                "the working point for this quantity did not resolve on this "
                "rig; the plan below says why",
                t.unit,
            )
            continue
        out[t.key] = cell(
            m.get("value"), ESTIMATE, str(m.get("basis") or ""), unit=t.unit
        )
    return out


def _scaled(base: dict, factor: float, why: str) -> dict:
    """A base cell moved by a stated factor. Stays an estimate; the reason is
    appended rather than replacing the original basis, so the chain from the
    measured input to the printed number stays readable."""
    if not base.get("available"):
        return dict(base)
    return cell(
        float(base["value"]) * factor,
        ESTIMATE,
        (base.get("basis") or "") + " " + why,
        unit=base.get("unit", ""),
        study=base.get("study"),
    )


def _variant_reasons(
    fam: FamilySpec,
    spill: str,
    locality: str,
    ctx: "MatrixContext",
) -> List[dict]:
    """Why this cell of the matrix cannot exist. Empty means it can.

    Each reason names its source, because a refusal without a citation is
    indistinguishable from an opinion.
    """
    out: List[dict] = []
    if spill not in fam.spill:
        if spill == "on" and "pd" in fam.tags:
            e = rejmod.by_key("spill_with_pd")
            out.append(
                {
                    "reason": e.verdict if e else "spill and PD cannot combine",
                    "source": (e.evidence if e else ""),
                    "register_key": "spill_with_pd",
                }
            )
        elif spill == "on" and "pipeline-parallel" in fam.tags:
            e = rejmod.by_key("spill_with_pp_dp")
            out.append(
                {
                    "reason": e.verdict if e else "spill needs single-node TP/DCP",
                    "source": (e.evidence if e else ""),
                    "register_key": "spill_with_pp_dp",
                }
            )
        else:
            out.append(
                {
                    "reason": (
                        f"this family does not take spill {spill!r}"
                    ),
                    "source": "family definition",
                }
            )
    if locality not in fam.locality:
        if locality == "network":
            out.append(
                {
                    "reason": (
                        "this family is a single-host layout: it has no "
                        "stage or arm that could sit on another machine"
                    ),
                    "source": "family definition",
                }
            )
        else:
            out.append(
                {
                    "reason": (
                        "this family is defined by having a second machine "
                        "in it; without one it collapses into a local family "
                        "already listed above"
                    ),
                    "source": "family definition",
                }
            )
    if locality == "network" and not ctx.remotes:
        out.append(
            {
                "reason": (
                    "no remote host is known. Pair one on the hardware step, "
                    "or add it by address; nothing about a machine that has "
                    "not answered can be stated"
                ),
                "source": "pairing store",
            }
        )
    if fam.status == "design":
        out.append(
            {"reason": fam.status_reason, "source": fam.recipe or "design note"}
        )
    if fam.status == "partial" and locality == "network":
        out.append(
            {"reason": fam.status_reason, "source": fam.recipe or "task status"}
        )
    if len(ctx.gpus) < fam.min_local_cards:
        out.append(
            {
                "reason": (
                    f"needs at least {fam.min_local_cards} local cards, "
                    f"{len(ctx.gpus)} detected"
                ),
                "source": "hardware detection",
            }
        )
    # The bootability gate: a split that was tried and did not boot is a
    # measured finding, and it refuses the family it describes. It applies
    # only where the probe measures -- the local uneven path.
    if (
        fam.key == "uneven_tp_dcp"
        and locality == "local"
        and (ctx.measured or {}).get("unbootable")
    ):
        m = ctx.measured or {}
        out.append(
            {
                "reason": str(m.get("unbootable")),
                "source": (
                    "split_probe row for candidate "
                    f"{m.get('candidate')!r}, measured "
                    f"{m.get('timestamp')}"
                ),
            }
        )
    if spill == "on":
        for key in ("attention_backend_not_flashinfer", "page_size_not_1"):
            if key in ctx.spill_blockers:
                out.append(
                    {
                        "reason": ctx.spill_blockers[key],
                        "source": "server_args, hard reject at arg parse",
                    }
                )
    # The register, consulted on the combined tag set.
    tags = list(fam.tags) + list(ctx.config_tags)
    if spill == "on":
        tags.append("spill")
    for entry in rejmod.check_combination(tags):
        if entry.level != rejmod.BLOCKED:
            continue
        if entry.key in {r.get("register_key") for r in out}:
            continue
        out.append(
            {
                "reason": entry.verdict,
                "source": entry.evidence,
                "register_key": entry.key,
            }
        )
    return out


@dataclasses.dataclass
class MatrixContext:
    """Everything the matrix is evaluated against, gathered once."""

    gpus: List[dict] = dataclasses.field(default_factory=list)
    remotes: List[dict] = dataclasses.field(default_factory=list)
    capabilities: Dict[str, dict] = dataclasses.field(default_factory=dict)
    #: Tags describing the model + flags in the form (``"gguf"``, ``"moe"``,
    #: ``"hybrid"``, ``"moe-offload"``, ``"sm75"``, ...). These join the
    #: family's own tags for the register check.
    config_tags: Tuple[str, ...] = ()
    spill_blockers: Dict[str, str] = dataclasses.field(default_factory=dict)
    hybrid: bool = False
    usage_pattern: str = "fresh"
    context_tokens: int = 8192
    link_gbs: Optional[float] = None
    link_source: str = ""
    satellite_prefill_tok_s: Optional[float] = None
    kv_bytes_per_token: Optional[float] = None
    state_bytes: Optional[float] = None
    #: The split-probe row for this model on this rig, when one exists: the
    #: only source here that turns a target from ``estimate`` into
    #: ``measured``. It also carries the bootability verdict -- a row with a
    #: non-empty ``unbootable`` is a finding, and the family it describes is
    #: refused with that text rather than offered with a number.
    measured: Optional[dict] = None


def _undisturbedness(moves_prefill_off: bool) -> dict:
    """The decode spike, as its own quantity.

    Kept out of TTFT deliberately: on this rig the same change that removes
    the spike costs 2.29 s of TTFT, so folding one into the other would hide
    a real gain behind a real loss.
    """
    if not moves_prefill_off:
        return cell(
            ANCHOR_DECODE_SPIKE_MS,
            ESTIMATE,
            "prefill chunks share the decode cards, so a long prompt "
            "interrupts the running decodes. The figure is the worst "
            "inter-token time measured under that condition in "
            + ANCHOR_STUDY,
            unit="ms",
            study=ANCHOR_STUDY,
        )
    return cell(
        ANCHOR_DECODE_SPIKE_OFFLOADED_MS,
        ESTIMATE,
        "prefill no longer runs on the decode cards. The spike does not get "
        "smaller, it disappears: worst inter-token time "
        f"{ANCHOR_DECODE_SPIKE_MS} -> {ANCHOR_DECODE_SPIKE_OFFLOADED_MS} ms, "
        f"median {ANCHOR_DECODE_MEDIAN_MS} -> "
        f"{ANCHOR_DECODE_MEDIAN_OFFLOADED_MS} ms, measured in " + ANCHOR_STUDY,
        unit="ms",
        study=ANCHOR_STUDY,
    )


def _warm_prefix_note(ctx: MatrixContext, fam: FamilySpec) -> Optional[str]:
    """What recurring prefixes do to this family's TTFT.

    Prefix reuse removes prefill work rather than moving it, so on repeated
    prefixes it beats every topology lever here. For a hybrid model the
    caveat is sharp: the HiCache store route matches zero tokens on a Mamba
    checkpoint, because the match is truncated to the deepest node owning a
    Mamba snapshot and there is none, so only the PD path carries a handover.
    """
    if ctx.usage_pattern != "recurring":
        return None
    if ctx.hybrid:
        if "pd" in fam.tags:
            return (
                "Recurring prefixes: this family carries them. PD ships the "
                "GDN state alongside the KV, which is why the handover works "
                "on a hybrid checkpoint at all -- a decode arm receiving a "
                "6464-token prefix prefilled not one token of it."
            )
        return (
            "Recurring prefixes: inside one server the device radix cache "
            "reuses them normally. Across servers it does not on this "
            "checkpoint -- the HiCache store route truncates the match to "
            "the deepest node holding a Mamba snapshot, and on a hybrid GDN "
            "model there is none, so the match length is zero and the prompt "
            "is recomputed from token 0. Only the PD path carries a hybrid "
            "handover."
        )
    return (
        "Recurring prefixes: prefix reuse removes the prefill instead of "
        "moving it, so on this workload it dominates every choice on this "
        "page. The store route is viable on a dense checkpoint."
    )


def _family_variant(
    fam: FamilySpec,
    spill: str,
    locality: str,
    ctx: MatrixContext,
    base: Dict[str, dict],
) -> dict:
    reasons = _variant_reasons(fam, spill, locality, ctx)
    moves_prefill_off = fam.key in ("pd_disjoint", "satellite_prefill")
    targets: Dict[str, dict] = {}
    notes: List[str] = []

    if fam.key == "solo_tp":
        totals = [int(g.get("total_mib") or 0) for g in ctx.gpus] or [0]
        if len(set(totals)) > 1 and len(totals) > 1:
            factor = (min(totals) * len(totals)) / float(sum(totals) or 1)
            for t in TARGETS:
                if t.profile_key is None:
                    continue
                if t.key in ("max_kv", "max_parallel"):
                    targets[t.key] = _scaled(
                        base.get(t.key, _absent("", t.unit)),
                        factor,
                        (
                            "Scaled to even shards: every rank is the size "
                            f"the smallest card funds, so the usable budget "
                            f"is {min(totals)} MiB x {len(totals)} against "
                            f"{sum(totals)} MiB installed."
                        ),
                    )
                else:
                    targets[t.key] = dict(base.get(t.key, _absent("", t.unit)))
            notes.append(
                f"Even shards on unequal cards discard "
                f"{sum(totals) - min(totals) * len(totals)} MiB of installed "
                "VRAM. That is the cost of the stock path here, and it is "
                "the reason the uneven family exists."
            )
        else:
            for t in TARGETS:
                if t.profile_key is not None:
                    targets[t.key] = dict(base.get(t.key, _absent("", t.unit)))
    else:
        for t in TARGETS:
            if t.profile_key is not None:
                targets[t.key] = dict(base.get(t.key, _absent("", t.unit)))

    # A split-probe row replaces the estimate with the measurement, and only
    # for the path the probe actually boots: the local uneven one. A measured
    # number from a boot of a different topology would be a measurement of
    # something else wearing this family's label.
    if fam.key == "uneven_tp_dcp" and locality == "local" and ctx.measured:
        m = ctx.measured
        probe_basis = (
            "measured by a split_probe boot of candidate "
            f"{m.get('candidate')!r} on this rig ({m.get('timestamp')})"
        )
        for key, field, unit in (
            ("max_decode", "decode_tok_s", "tok/s"),
            ("max_prefill", "prefill_tok_s", "tok/s"),
            ("max_kv", "max_total_num_tokens", "tokens"),
        ):
            v = m.get(field)
            if v:
                targets[key] = cell(
                    float(v), MEASURED, probe_basis, unit=unit
                )
                notes.append(
                    f"{key.replace('_', ' ')}: measured, not estimated -- "
                    + probe_basis
                )

    # A family that re-divides the budget between arms cannot borrow the base
    # plan's capacity figures: those describe one group holding every card.
    # How much of the budget the decode arm keeps is the split control, and
    # the split control is v2. An estimate here would be an estimate of a
    # configuration nobody has chosen yet, so the cells stay absent and name
    # what would fill them.
    if fam.key in ("pd_disjoint", "satellite_prefill", "pd_rank_reuse"):
        for key, unit in (("max_kv", "tokens"), ("max_parallel", "sessions")):
            targets[key] = _absent(
                "this family divides the cards between a prefill arm and a "
                "decode arm, so the base plan -- one group holding every "
                "card -- does not describe it. The share each arm keeps is "
                "the split control, which is v2 of this wizard; set it there "
                "and these fill in.",
                unit,
            )
    if fam.key == "extra_solo_session":
        for key, unit in (("max_kv", "tokens"), ("max_parallel", "sessions")):
            targets[key] = _absent(
                "the co-resident server's own posts come out of the card it "
                "lands on before the main group gets any KV, and which side "
                "model it runs is not part of this step. Pick the side model "
                "and its card, and the co-residence ledger prices both.",
                unit,
            )
    if fam.key == "pp_cross_rig":
        for key, unit in (("max_kv", "tokens"), ("max_parallel", "sessions")):
            targets[key] = _absent(
                "the token count under pipeline parallelism is min-reduced "
                "across the world, so the deeper stage sets it for every "
                "stage and a layer split the wizard has not chosen cannot be "
                "priced. It is also the known open limitation of this family.",
                unit,
            )

    if spill == "on":
        notes.append(
            "Spill does not enlarge the resident KV pool -- the context and "
            "parallel figures above are the resident ones and are unchanged "
            "by it. What it buys is sessions PARKED in host RAM and waved "
            "back on demand, at the cost of a restore over the host link. "
            "How many, and what the restore costs at a given context, is not "
            "measured for this model; the spill lane's own boot study is "
            "what fills it."
        )
        targets["max_decode"] = _absent(
            "the spill lane refuses a speculative algorithm at arg parse "
            "unless the KVSO_ALLOW_SPEC bring-up gate is set, so this "
            "configuration decodes without speculation and the plan's "
            "speculative figure does not describe it. No no-spec figure has "
            "been measured for this model with spill on.",
            "tok/s",
        )
        notes.append(
            "No speculation in this configuration either. Its decode numbers "
            "are no-spec numbers and must not be compared against a NEXTN one."
        )

    # PD and PP both run without speculation. The decode figure of a
    # speculative plan cannot be carried across that boundary.
    if "pd" in fam.tags or "pipeline-parallel" in fam.tags:
        targets["max_decode"] = _absent(
            "this family runs without speculation -- PD disables the "
            "speculative algorithm and PP refuses it with a hard assert -- "
            "so the plan's speculative decode figure does not describe it "
            "and no no-spec figure has been measured for this model. "
            "Measure it with a split_probe run of the same shape.",
            "tok/s",
        )
        notes.append(
            "No speculation in this family. Every decode number it ever "
            "produces is a no-spec number and must not be compared against a "
            "NEXTN one."
        )

    if fam.key == "extra_solo_session":
        notes.append(
            "The co-resident server's budget comes out of the card it lands "
            "on, and the ledger charges it before KV: weights and runtime "
            "state, the mamba state pool, the speculative intermediate "
            "state, the prefill activation reserve and the dequant scratch. "
            "A figure derived from the weight shard alone is about 2.5 GiB "
            "optimistic. Where the main group already fills the card, the "
            "bracket closes from both sides and no allocation satisfies "
            "both -- the paths that open it are a smaller side model or "
            "lower graph/spec posts on the main group."
        )

    # TTFT, always as the pair.
    prefill_cell = base.get("max_prefill") or {}
    prefill_rate = prefill_cell.get("value") if prefill_cell.get("available") else None
    if fam.key == "satellite_prefill":
        gate = link_gate(
            ctx.kv_bytes_per_token,
            ctx.state_bytes,
            ctx.link_gbs,
            [ctx.context_tokens],
            link_source=ctx.link_source,
        )
        transport_s = gate["rows"][0]["seconds"] if gate.get("rows") else None
        pair = _satellite_ttft(
            ctx.context_tokens, ctx.satellite_prefill_tok_s, transport_s
        )
    elif "pd" in fam.tags:
        pair = ttft_pair(
            prefill_rate,
            ctx.context_tokens,
            added_s=ANCHOR_PD_HANDSHAKE_S,
            basis_extra=(
                "plus the "
                f"{ANCHOR_PD_HANDSHAKE_S:.3f} s handover measured in "
                + ANCHOR_STUDY
            ),
        )
    else:
        pair = ttft_pair(prefill_rate, ctx.context_tokens)
    targets["ttft"] = {
        "pair": True,
        "idle": pair["idle"],
        "loaded": pair["loaded"],
        "label": "time to first token",
        "unit": "s",
    }

    warm = _warm_prefix_note(ctx, fam)
    if warm:
        notes.append(warm)

    # Advisory register rows: measured, lost, still available on request.
    tags = list(fam.tags) + list(ctx.config_tags) + (["spill"] if spill == "on" else [])
    advisories = [
        e.to_json()
        for e in rejmod.check_combination(tags)
        if e.level == rejmod.NOT_DEFAULT
    ]

    return {
        "spill": spill,
        "locality": locality,
        "feasible": not reasons,
        "reasons": reasons,
        "targets": targets,
        "undisturbedness": _undisturbedness(moves_prefill_off),
        "notes": notes,
        "advisories": advisories,
    }


def build_matrix(
    profiles: dict,
    *,
    context: MatrixContext,
    plan: Optional[dict] = None,
    families: Sequence[str] = FAMILY_KEYS,
    contexts_for_link_gate: Sequence[int] = (4096, 16384, 32768, 131072),
) -> dict:
    """The family matrix: every family, every axis value, feasible or not.

    ``profiles`` is exactly what ``/api/lever_profiles`` returned. Nothing is
    recomputed from it -- the four capacity targets are the working points'
    own figures, carried through with their own basis text, so this page and
    the working-point table cannot disagree about the same configuration.
    """
    if not profiles.get("ok"):
        return {
            "ok": False,
            "reasons": list(profiles.get("reasons") or ["the plan did not resolve"]),
            "families": [],
        }
    base = _base_targets(profiles)
    rows: List[dict] = []
    for fam in FAMILIES:
        if fam.key not in families:
            continue
        variants = [
            _family_variant(fam, spill, locality, context, base)
            for spill in SPILL_STATES
            for locality in LOCALITIES
        ]
        rows.append(
            {
                **fam.to_json(),
                "feasible": any(v["feasible"] for v in variants),
                "variants": variants,
            }
        )
    gate = link_gate(
        context.kv_bytes_per_token,
        context.state_bytes,
        context.link_gbs,
        contexts_for_link_gate,
        link_source=context.link_source,
    )
    return {
        "ok": True,
        "model": profiles.get("model"),
        "targets": [dataclasses.asdict(t) for t in TARGETS],
        "spill_states": list(SPILL_STATES),
        "localities": list(LOCALITIES),
        "usage_pattern": context.usage_pattern,
        "context_tokens": int(context.context_tokens),
        "base_targets": base,
        "families": rows,
        "link_gate": gate,
        "anchors": dict(ANCHORS),
        "modifiers": _modifiers(context),
        "blocked": rejmod.register_json(rejmod.BLOCKED),
        "caveats": list(profiles.get("caveats") or []),
        "plan_fits": (plan or {}).get("fits"),
    }


def _modifiers(ctx: MatrixContext) -> List[dict]:
    """Things that ride on a family rather than being one.

    Expert offload is the clear case: it is selected by an environment
    variable, it is registered as an ordinary flag with a model-compat
    predicate, and it composes with every topology above instead of replacing
    one. It gets a row here so it is visible, and so the one combination the
    engine will NOT refuse on its own is refused here.
    """
    out: List[dict] = []
    gguf = "gguf" in ctx.config_tags
    moe = "moe" in ctx.config_tags
    entry = rejmod.by_key("gguf_moe_expert_offload")
    out.append(
        {
            "key": "moe_expert_offload",
            "label": "MoE expert offload",
            "applies": moe,
            "available": moe and not gguf,
            "reason": (
                entry.verdict
                if (moe and gguf and entry)
                else (
                    ""
                    if moe
                    else "dense checkpoint: there are no experts to offload"
                )
            ),
            "source": (entry.evidence if (moe and gguf and entry) else ""),
            "note": (
                "Composes with every family above. Requires "
                "--disable-cuda-graph unless SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1, "
                "and moves the budget to host RAM."
            ),
        }
    )
    out.append(
        {
            "key": "speculation",
            "label": "Speculative decoding",
            "applies": True,
            "available": True,
            "reason": "",
            "source": "",
            "note": (
                "Available on the local TP families. PD disables it with a "
                "warning and PP refuses it with a hard assert, so both are "
                "no-spec by construction."
            ),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------

#: Extra flags a family needs on top of its base preset, each with the reason
#: it is there. The wizard never emits a flag it cannot explain.
_FAMILY_FLAGS: Dict[str, List[Tuple[str, Optional[str], str]]] = {
    "pd_disjoint": [
        (
            "--disaggregation-mode",
            "prefill",
            "names this arm; the decode arm carries the same flag with "
            "'decode'",
        ),
        (
            "--disaggregation-transfer-backend",
            "mooncake",
            "the transfer backend the handover runs on",
        ),
        ("--page-size", "1", "the arms compare page size on handshake"),
    ],
    "satellite_prefill": [
        (
            "--disaggregation-mode",
            "prefill",
            "the satellite is a prefill arm on another host",
        ),
        (
            "--disaggregation-transfer-backend",
            "mooncake_tcp",
            "TCP over the fast line; rewritten to mooncake with MC_FORCE_TCP=1",
        ),
        ("--page-size", "1", "the arms compare page size on handshake"),
    ],
    "pp_cross_rig": [
        (
            "--disable-overlap-schedule",
            None,
            "mandatory under pp_size > 1; the boot dies at arg parse without it",
        ),
    ],
}

#: On every command this wizard emits, with no topology exception. It changes
#: nothing about inference; leaving it off blinds the dashboard, which reads
#: everything it draws from the endpoint this flag opens.
_ALWAYS_FLAGS: List[Tuple[str, Optional[str], str]] = [
    (
        "--enable-metrics",
        None,
        "mandatory on every launch here: without it the round times and the "
        "per-rank compute/wait split have nothing to read",
    ),
]

#: Flags a PD family must NOT carry. The engine turns speculation off in
#: disaggregation mode with a warning rather than an error, so a command that
#: still names it launches a server that quietly differs from the flag. The
#: wizard removes them instead and says why.
_PD_DROP_FLAGS: Tuple[str, ...] = (
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
    "--speculative-draft-model-path",
    "--speculative-adaptive",
)

#: Flags the wizard adds when spill is switched on.
_SPILL_FLAGS: List[Tuple[str, Optional[str], str]] = [
    ("--enable-kv-session-offload", None, "turns the spill lane on"),
    (
        "--attention-backend",
        "flashinfer",
        "spill accepts no other attention backend",
    ),
    ("--page-size", "1", "spill accepts no other page size"),
]


def _render(argv: Sequence[str], env: Dict[str, str], python_exe: str) -> str:
    """One copy-pasteable line in the runbook's shape."""
    parts = [f"{k}={v}" for k, v in sorted(env.items())]
    head = " ".join(parts)
    body = f'setsid "{python_exe}" -u -m sglang.launch_server ' + " ".join(
        str(a) for a in argv
    )
    return (head + " " + body).strip()


def build_command(
    family_key: str,
    *,
    profile: Optional[dict] = None,
    spill: str = "off",
    overrides: Optional[Dict[str, Any]] = None,
    python_exe: str = "$VENV/bin/python",
) -> dict:
    """The launch command for one family, plus the diff against it.

    ``profile`` is a ``flags.Profile`` rendered to JSON (``argv``,
    ``launch_env``) -- the argv is never assembled here from a flag catalog of
    its own, so the wizard and the runner tab launch the same server.
    ``overrides`` is the expert view's edited flag map; the answer carries
    both commands and the difference between them, so an edit is visible as
    an edit rather than as a new recommendation.
    """
    fam = _FAMILY_BY_KEY.get(family_key)
    if fam is None:
        return {"ok": False, "error": f"unknown family {family_key!r}"}
    profile = profile or {}
    argv: List[str] = [str(a) for a in (profile.get("argv") or [])]
    env: Dict[str, str] = {
        str(k): str(v) for k, v in (profile.get("launch_env") or {}).items()
    }
    provenance: List[dict] = [
        {
            "flag": "(base preset)",
            "value": profile.get("name") or fam.profile_kind or "none",
            "why": (
                "generated by the planner's profile generator for this model "
                "and this card set"
            ),
        }
    ]

    drop_why = ""
    if "pd" in fam.tags:
        drop_why = (
            "this fork disables speculation in disaggregation mode with a "
            "warning, not an error. Leaving the flags on the line would "
            "launch a server that quietly differs from the command that "
            "describes it"
        )
    elif spill == "on":
        drop_why = (
            "the spill lane refuses a speculative algorithm at arg parse "
            "unless the KVSO_ALLOW_SPEC bring-up gate is set. This command "
            "would not boot with them on it"
        )
    if drop_why:
        dropped = []
        i = 0
        while i < len(argv):
            if argv[i] in _PD_DROP_FLAGS:
                dropped.append(argv[i])
                n = 2 if i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
                del argv[i : i + n]
                continue
            i += 1
        if dropped:
            provenance.append(
                {
                    "flag": ", ".join(dropped),
                    "value": "removed",
                    "why": drop_why,
                }
            )

    added: List[Tuple[str, Optional[str], str]] = list(
        _FAMILY_FLAGS.get(family_key, [])
    ) + list(_ALWAYS_FLAGS)
    if spill == "on":
        added += _SPILL_FLAGS
    for flag, value, why in added:
        if flag in argv:
            i = argv.index(flag)
            if value is not None and i + 1 < len(argv):
                argv[i + 1] = value
        else:
            argv.append(flag)
            if value is not None:
                argv.append(value)
        provenance.append({"flag": flag, "value": value, "why": why})

    guided_argv = list(argv)
    guided = _render(guided_argv, env, python_exe)

    edited_argv = list(argv)
    diff: List[dict] = []
    for flag, value in sorted((overrides or {}).items()):
        name = flag if flag.startswith("--") else "--" + flag.replace("_", "-")
        # ``before`` is what the guided command had: a value string, True for
        # a bare store_true flag, or None when the flag was not there at all.
        before: Any = None
        if name in edited_argv:
            i = edited_argv.index(name)
            if i + 1 < len(edited_argv) and not str(edited_argv[i + 1]).startswith("--"):
                before = edited_argv[i + 1]
                if value in (None, False):
                    del edited_argv[i : i + 2]
                else:
                    edited_argv[i + 1] = str(value)
            else:
                before = True
                if value in (None, False):
                    del edited_argv[i]
        else:
            if value not in (None, False):
                edited_argv.append(name)
                if value is not True:
                    edited_argv.append(str(value))
        after = None if value in (None, False) else value
        if before != after:
            diff.append({"flag": name, "guided": before, "yours": after})

    return {
        "ok": True,
        "family": family_key,
        "label": fam.label,
        "recipe": fam.recipe,
        "spill": spill,
        "env": env,
        "argv": guided_argv,
        "command": guided,
        "edited_argv": edited_argv,
        "edited_command": _render(edited_argv, env, python_exe),
        "diff": diff,
        "provenance": provenance,
        "arms": _extra_arms(fam),
        "notes": _command_notes(fam, spill),
    }


def _extra_arms(fam: FamilySpec) -> List[dict]:
    """The commands that are not this one.

    A PD or satellite family is two servers; a cross-rig family is two hosts.
    Emitting only one of them would be a command that cannot serve.
    """
    if fam.key == "pd_disjoint":
        return [
            {
                "role": "decode arm",
                "note": (
                    "Same model and page size, "
                    "--disaggregation-mode decode, its own cards. The "
                    "handshake compares page size and KV dtype and nothing "
                    "else, so two different checkpoints will pair happily "
                    "and produce fluent nonsense -- check the model path on "
                    "both arms."
                ),
                "recipe": "runbook 4.8",
            }
        ]
    if fam.key == "satellite_prefill":
        return [
            {
                "role": "decode arm (this rig)",
                "note": (
                    "--disaggregation-mode decode on the serving cards, same "
                    "dtype as the satellite. A Turing satellite has no "
                    "bfloat16, so the whole pair runs fp16 -- including the "
                    "monolithic baseline it is compared against."
                ),
                "recipe": "runbook 4.8",
            },
            {
                "role": "satellite (remote host)",
                "note": (
                    "On sm75 use --attention-backend triton and "
                    "--sampling-backend pytorch: the flashinfer paged prefill "
                    "asks 65616 bytes of shared memory at head_dim 256 and "
                    "Turing has 65536, and it fires on the first real prefill "
                    "rather than at boot."
                ),
                "recipe": "runbook 4.8",
            },
        ]
    if fam.key == "pp_cross_rig":
        return [
            {
                "role": "second stage",
                "note": (
                    "--pp-layer-ratio must sum to the BACKBONE depth, not to "
                    "a GGUF block_count that includes the MTP head."
                ),
                "recipe": "runbook 4.7",
            }
        ]
    return []


def _command_notes(fam: FamilySpec, spill: str) -> List[str]:
    out = [
        "--enable-metrics is on every recipe here; without it the round "
        "times and the per-rank split have nothing to read."
    ]
    if "pd" in fam.tags:
        out.append(
            "Speculation is disabled for this server. Leaving "
            "--speculative-algorithm on the command line does not fail -- it "
            "is turned off with a warning, so the server quietly differs "
            "from the one the flag describes."
        )
    if "pipeline-parallel" in fam.tags:
        out.append(
            "--disable-overlap-schedule is mandatory and speculation is "
            "refused by a hard assert under pp_size > 1."
        )
    if spill == "on":
        out.append(
            "Spill is single-node pure TP/DCP: it refuses PD, pipeline and "
            "data parallelism, hierarchical cache, unified memory and the "
            "weightless lane at arg parse."
        )
    return out
