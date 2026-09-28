# SPDX-License-Identifier: Apache-2.0
"""The rank form: ONE mechanism for where weights and KV live (28.09.).

User, 28.09. ~21:30Z: "aufjedenfall gibt es hardwarekonfigurationen bei denen
das so waere, also muss es quasi einen flag dafuer geben" -- no form is wired
in; every form is a VALUE of two per-rank vectors:

  weights  the rank's weight share (``--rank-tp-ratio``); 0 = the rank holds
           no weight shard at all and only KV (a KV-only / weightless rank)
  tokens   the rank's full-attention KV token share (``--uneven-token-vector``);
           None = the serve chain's own default split

From them follow the TP group (ranks with weights > 0) and the DCP group (all
ranks; a rank with token share 0 still joins the collectives, it attends over
nothing). The three forms of the 27B-NVFP4 question
(docs 27B-NVFP4-5090-GEWICHTE-KV-3080-0928.md) are values:

  C  every rank has weights           today's TP (uneven or not)
  A  exactly one rank has weights     the weightless-KV lane
                                      (--weightless-kv-fastlane, #115/#131/#143):
                                      head = that rank, TP=1 collective-free,
                                      every other rank KV-only
  B  >= 2 ranks have weights and      TP over a SUBSET of the DCP group; needs
     >= 1 rank has none               rank_role seam F6 (collectives over a
                                      subgroup), not wired -> refused by name

NO flag of its own (user 28.09.: one mechanism with NF's #239): the vectors
ARE NF's #239 flags -- ``--rank-tp-ratio`` (0 = no dense share), ``--rank-role``
host/worker and ``--uneven-token-vector`` -- and form A is checked with #239's
own ``RankRolePlan.check_dense_ratio``. Below them only the BACKEND path is
chosen: an MoE/QSA model runs #239's Form A workers (experts + F14 KV rows),
a DENSE GQA model runs the existing weightless-KV lane (the same collectives
of layers/dcp/comm.py, flashinfer/triton worker dispatch).

The planner picks the form (``choose_form``) from measured prices
(``weg2.form_measures/2``, written by devtools/dcp_exchange_bench.py); a
missing price is UNMEASURED and keeps today's form C. An explicit form (the
vectors) overrides the planner by name (``FORM-OVERRIDE``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

BACKEND_TP = "tp"                    # form C: classic (uneven) TP
BACKEND_FORM_A_239 = "form_a_239"    # NF #239: --rank-role host/worker
BACKEND_LANE = "weightless_lane"     # dense GQA: --weightless-kv-fastlane

FORM_C = "C"
FORM_A = "A"
FORM_B = "B"

MEASURES_SCHEMA = "weg2.form_measures/2"
#: The transport the serve path runs (barlink BAR1, user 28.09. "DOCH die haben
#: barlink bar1"). ``nccl_control`` prices are a CONTROL ARM: loadable for a
#: side-by-side, never what the planner decides on by default.
SERVE_TRANSPORT = "bar1"
FORM_CHOICE_MARKER = "RANK-FORM"


class RankFormError(ValueError):
    """A rank form that cannot run, named (W18x)."""


class RankFormNoWeightRank(RankFormError):
    code = "W180 Weg2RankFormNoWeightRank"


class RankFormShapeMismatch(RankFormError):
    code = "W181 Weg2RankFormShapeMismatch"


class RankFormSubgroupTpNotWired(RankFormError):
    code = "W182 Weg2RankFormSubgroupTpNotWired"


class RankFormVisionNoTail(RankFormError):
    code = "W183 Weg2RankFormVisionNoTail"


class RankFormLaneSpec(RankFormError):
    code = "W184 Weg2RankFormLaneSpec"


def _refuse(cls, text: str) -> RankFormError:
    return cls(f"{cls.code}: {text}")


def _ints(xs: Sequence, what: str) -> Tuple[int, ...]:
    try:
        out = tuple(int(x) for x in xs)
    except (TypeError, ValueError):
        raise _refuse(RankFormShapeMismatch, f"{what} {list(xs)!r} is not a list of integers")
    if any(x < 0 for x in out):
        raise _refuse(RankFormShapeMismatch, f"{what} {list(out)} has a negative share")
    return out


@dataclass(frozen=True)
class RankForm:
    kind: str
    weights: Tuple[int, ...]
    tokens: Optional[Tuple[int, ...]]
    weight_ranks: Tuple[int, ...]
    kv_only_ranks: Tuple[int, ...]
    backend: str = BACKEND_TP

    @property
    def world(self) -> int:
        return len(self.weights)

    @property
    def head_rank(self) -> Optional[int]:
        return self.weight_ranks[0] if self.kind == FORM_A else None

    def serve_argv(self) -> List[str]:
        """The serve-chain flags that run this form (group sizes and the form's
        own switches; model, KV dtype and the rest stay the caller's)."""
        n = str(self.world)
        argv: List[str] = ["--tp-size", n]
        if self.backend == BACKEND_LANE:
            # the #239 vectors too (NF review 28.09.): launcher.d_kv_worker_ranks
            # reads them, without them the lane workers fall out of the F14 gate
            argv += ["--dcp-size", n, "--weightless-kv-fastlane",
                     "--weightless-kv-head-rank", str(self.head_rank),
                     "--rank-tp-ratio", ",".join(str(w) for w in self.weights)]
        elif self.backend == BACKEND_FORM_A_239:
            argv += ["--rank-role", ",".join("host" if w > 0 else "worker" for w in self.weights),
                     "--rank-tp-ratio", ",".join(str(w) for w in self.weights)]
        elif self.kind == FORM_C and len(set(self.weights)) > 1:
            argv += ["--rank-tp-ratio", ",".join(str(w) for w in self.weights)]
        if self.tokens is not None:
            argv += ["--uneven-token-vector", ",".join(str(t) for t in self.tokens)]
        return argv

    def line(self) -> str:
        return (f"{FORM_CHOICE_MARKER} {self.kind} [{self.backend}]: weights {list(self.weights)} "
                f"(TP ranks {list(self.weight_ranks)}, KV-only {list(self.kv_only_ranks)}), "
                f"tokens {list(self.tokens) if self.tokens is not None else 'default'}")


def f6_wired() -> bool:
    """Form B's precondition, read from THE seam registry (rank_role F6), never
    from a caller's flag: a TP group over a subset of the ranks."""
    from sglang.srt.rank_role import SEAMS

    return bool(SEAMS["F6"].wired)


def resolve_rank_form(
    weights: Sequence,
    tokens: Optional[Sequence] = None,
    *,
    rank_role: Optional[str] = None,
    dense: bool = True,
    subgroup_tp_wired: Optional[bool] = None,
) -> RankForm:
    """The form NF's #239 flags describe, or a named refusal.

    ``weights`` = ``--rank-tp-ratio``, ``tokens`` = ``--uneven-token-vector``,
    ``rank_role`` = ``--rank-role`` (checked with #239's own
    ``RankRolePlan.check_dense_ratio`` when given), ``dense`` = the model has
    no routed experts (picks the backend of form A).

    Refusals: no weight rank (W180); vectors of different length (W181); a
    KV-only rank without token share -- it would hold nothing and still sit in
    every collective (W181); form B while seam F6 is not wired (W182)."""
    w = _ints(weights, "weight shares")
    if not w:
        raise _refuse(RankFormShapeMismatch, "no ranks")
    t = None if tokens is None else _ints(tokens, "token shares")
    if t is not None and len(t) != len(w):
        raise _refuse(RankFormShapeMismatch,
                      f"weight shares {list(w)} name {len(w)} ranks, token shares {list(t)} name {len(t)}")
    weight_ranks = tuple(r for r, x in enumerate(w) if x > 0)
    kv_only = tuple(r for r, x in enumerate(w) if x == 0)
    if not weight_ranks:
        raise _refuse(RankFormNoWeightRank, f"weight shares {list(w)}: no rank holds weights")
    if t is not None:
        idle = [r for r in kv_only if t[r] == 0]
        if idle:
            raise _refuse(RankFormShapeMismatch,
                          f"KV-only rank(s) {idle} have token share 0 (tokens {list(t)}): "
                          f"a rank with neither weights nor KV holds nothing")
        if sum(t) == 0:
            raise _refuse(RankFormShapeMismatch, f"token shares {list(t)} are all 0")
    if rank_role is not None:
        from sglang.srt.rank_role import RankRoleError, RankRolePlan

        try:
            RankRolePlan.parse(rank_role, len(w)).check_dense_ratio(w)
        except RankRoleError as e:
            raise _refuse(RankFormShapeMismatch, f"--rank-role {rank_role} vs weights {list(w)}: {e}")
    backend = BACKEND_TP
    if not kv_only:
        kind = FORM_C
    elif len(weight_ranks) == 1:
        kind = FORM_A
        backend = BACKEND_LANE if dense else BACKEND_FORM_A_239
        if backend == BACKEND_LANE and t is not None and t[weight_ranks[0]] == 0:
            # the lane head writes and attends KV too and the pool is pinned
            # on local/ratio per rank (distributed.utils.resolve_cp_token_ratios)
            raise _refuse(
                RankFormShapeMismatch,
                f"tokens {list(t)}: on the weightless lane the head rank {weight_ranks[0]} "
                f"needs a token share >= 1 (host 0 is a #239 Form A layout for MoE/QSA)")
    else:
        kind = FORM_B
        if not (f6_wired() if subgroup_tp_wired is None else subgroup_tp_wired):
            raise _refuse(
                RankFormSubgroupTpNotWired,
                f"weights {list(w)}: TP over ranks {list(weight_ranks)} with KV-only rank(s) "
                f"{list(kv_only)} needs collectives over a subgroup (rank_role seam F6, "
                f"wired=False) -- the TP all-reduce would include the KV-only ranks")
    return RankForm(kind=kind, weights=w, tokens=t, weight_ranks=weight_ranks,
                    kv_only_ranks=kv_only, backend=backend)


def check_form_vision(form: RankForm, *, vision: bool, head_lease_mib: int = 0) -> Optional[str]:
    """The vision stage leases KV-tail views on the rank that runs the tower
    (the weight head). Under a form whose head holds no KV tokens there is no
    tail to lease: refused by name unless an explicit own lease (MiB on the
    head, outside the KV pool) is budgeted. Returns the line to log, or None."""
    if not vision:
        return None
    head = form.weight_ranks[0]
    head_tokens = None if form.tokens is None else form.tokens[head]
    if head_tokens == 0:
        if head_lease_mib <= 0:
            raise _refuse(
                RankFormVisionNoTail,
                f"vision on, but the head rank {head} holds no KV tokens (tokens "
                f"{list(form.tokens)}) -- the tower has no KV tail to lease; give the "
                f"head a token share or budget an own vision lease")
        return (f"{FORM_CHOICE_MARKER} vision: head rank {head} holds no KV, own lease "
                f"{int(head_lease_mib)} MiB outside the KV pool")
    return None


def check_form_spec(form: RankForm, algorithm: Optional[str]) -> None:
    """Form A runs on the weightless-KV lane, whose worker dispatch mirrors the
    EAGLE-family chain and (G-A1) the DFLASH chain -- solo draft on the head,
    one draft-block and one accept broadcast per round
    (server_args._reject_unsupported_weightless_spec). Anything else is
    refused here by name, before the serve chain would."""
    if form.backend != BACKEND_LANE or not algorithm:
        return
    a = str(algorithm).upper()
    if a not in ("EAGLE", "EAGLE3", "NEXTN", "DFLASH"):
        raise _refuse(
            RankFormLaneSpec,
            f"form A runs on the weightless-KV lane, which mirrors the EAGLE/EAGLE3/NEXTN "
            f"and DFLASH chains only; got --speculative-algorithm {algorithm}")


# --------------------------------------------------------------------------
# The planner's time model (decode round, bs rows) over measured prices.
# --------------------------------------------------------------------------

@dataclass
class FormMeasures:
    """``weg2.form_measures/2``: the prices the model needs, each keyed so a
    missing one is visible (never a default). Collective tables are keyed by
    transport first; ``load`` reads ONE transport (default the serve path's)."""

    allreduce_us: Dict[str, Dict[int, float]] = field(default_factory=dict)
    dcp_exchange_us: Dict[str, Dict[int, float]] = field(default_factory=dict)
    #: whole decode rounds measured on metal: form label -> cards key -> bs -> ms
    rounds_ms: Dict[str, Dict[str, Dict[int, float]]] = field(default_factory=dict)
    source: str = ""
    transport: str = SERVE_TRANSPORT

    @classmethod
    def load(cls, path: Optional[str], transport: str = SERVE_TRANSPORT) -> "FormMeasures":
        if not path or not os.path.exists(path):
            return cls(source="", transport=transport)
        with open(path) as f:
            doc = json.load(f)
        if doc.get("schema") != MEASURES_SCHEMA:
            raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {MEASURES_SCHEMA!r}")

        def table(key: str) -> Dict[str, Dict[int, float]]:
            out: Dict[str, Dict[int, float]] = {}
            for cards, per_bs in ((doc.get(key) or {}).get(transport) or {}).items():
                out[cards] = {int(bs): float(v["graph_median_us"] if v.get("graph_median_us")
                                             is not None else v["eager_median_us"])
                              for bs, v in per_bs.items()
                              if v.get("graph_median_us") is not None
                              or v.get("eager_median_us") is not None}
            return out

        rounds: Dict[str, Dict[str, Dict[int, float]]] = {}
        for form, per_cards in (doc.get("rounds") or {}).items():
            for cards, per_bs in per_cards.items():
                rounds.setdefault(form, {})[cards] = {
                    int(bs): float(v["median_ms"]) for bs, v in per_bs.items()
                    if v.get("median_ms") is not None
                    and v.get("transport", SERVE_TRANSPORT) == transport}
        return cls(allreduce_us=table("allreduce"), dcp_exchange_us=table("dcp_exchange"),
                   rounds_ms=rounds, source=path, transport=transport)


def cards_key(uuids: Sequence[str]) -> str:
    """The key a price is stored under: the card UUIDs in rank order."""
    return "+".join(str(u) for u in uuids)


@dataclass(frozen=True)
class FormCost:
    """One form's decode-round estimate, or why it has none."""

    form: str
    round_ms: Optional[float]
    terms: Mapping[str, float]
    missing: Tuple[str, ...]
    #: the model's number even where an input is only calculated (never used
    #: to CHOOSE -- a Hochrechnung is not a measurement), None if a collective
    #: price is missing entirely
    estimate_ms: Optional[float] = None
    measured_round: bool = False


def form_round_cost(
    form: RankForm,
    *,
    bs: int,
    weight_read_ms: Sequence[float],
    attn_ms: Sequence[float],
    fixed_ms: float,
    n_allreduce: int,
    n_attn_layers: int,
    uuids: Sequence[str],
    measures: FormMeasures,
    const_ms: Optional[Sequence[float]] = None,
    compute_measured: bool = False,
    label: Optional[str] = None,
) -> FormCost:
    """T_D(form) = max_r(weight_read_r * share_r + attn_r * tokshare_r)
                   + n_allreduce * t_ar(TP ranks) + n_attn_layers * t_dcp(DCP ranks)
                   + fixed.

    ``weight_read_ms[r]``: rank r reading ALL weights (its share scales it);
    ``const_ms[r]``: rank r's non-GEMM/launch constant, paid only where the rank
    holds weights (a KV-only rank runs no model layer); ``attn_ms[r]``: rank r
    attending over ALL KV of the round (its token share scales it). Collective
    prices come ONLY from ``measures``; a price the form needs and the file
    lacks is named in ``missing``. A whole round measured on metal
    (``measures.rounds_ms[label]``) replaces the model. Unless the compute
    inputs are measured (``compute_measured``), ``round_ms`` stays None and
    only ``estimate_ms`` carries the number."""
    lab = label or form.kind
    meas = measures.rounds_ms.get(lab, {}).get(cards_key(uuids), {}).get(int(bs))
    if meas is not None:
        return FormCost(form=lab, round_ms=meas, terms={"measured_round_ms": meas}, missing=(),
                        estimate_ms=meas, measured_round=True)
    missing: List[str] = []
    wsum = float(sum(form.weights))
    tok = form.tokens if form.tokens is not None else tuple(1 for _ in form.weights)
    tsum = float(sum(tok))
    const = list(const_ms) if const_ms is not None else [0.0] * form.world
    per_rank = [(float(const[r]) if form.weights[r] > 0 else 0.0)
                + float(weight_read_ms[r]) * form.weights[r] / wsum
                + float(attn_ms[r]) * tok[r] / tsum for r in range(form.world)]
    terms: Dict[str, float] = {"compute_ms": max(per_rank), "fixed_ms": float(fixed_ms)}
    tp_uuids = [uuids[r] for r in form.weight_ranks]
    if len(tp_uuids) > 1:
        ar = measures.allreduce_us.get(cards_key(tp_uuids), {}).get(int(bs))
        if ar is None:
            missing.append(f"allreduce[{measures.transport}][{cards_key(tp_uuids)}][bs{bs}]")
        else:
            terms["allreduce_ms"] = n_allreduce * ar / 1000.0
    if form.world > 1:
        # every candidate token-shards the full-attention KV over all ranks
        # (today's C runs uneven DCP too); the bench's dcp mode prices the
        # host-broadcast shape of A, an upper bound for C's head-split gather
        dk = measures.dcp_exchange_us.get(cards_key(uuids), {}).get(int(bs))
        if dk is None:
            missing.append(f"dcp_exchange[{measures.transport}][{cards_key(uuids)}][bs{bs}]")
        else:
            terms["dcp_ms"] = n_attn_layers * dk / 1000.0
    est = None if missing else sum(terms.values())
    if not compute_measured:
        missing.append(f"compute[{lab}] (calculated, not measured)")
    total = None if missing else est
    return FormCost(form=lab, round_ms=total, terms=terms, missing=tuple(missing),
                    estimate_ms=est)


def choose_form(
    candidates: Mapping[str, RankForm],
    costs: Mapping[str, FormCost],
    *,
    today: str,
    override: Optional[str] = None,
) -> Tuple[str, str]:
    """(label, line). An explicit override wins by name; otherwise the cheapest
    fully MEASURED candidate; if any candidate is unmeasured the planner keeps
    ``today`` (UNMEASURED never picks a new form)."""
    if override:
        if override not in candidates:
            raise ValueError(f"FORM-OVERRIDE {override!r} is not a candidate ({sorted(candidates)})")
        return override, (f"{FORM_CHOICE_MARKER} FORM-OVERRIDE {override}: "
                          f"{candidates[override].line()} (the planner does not decide)")
    unmeasured = {k: c.missing for k, c in costs.items() if c.round_ms is None}
    if unmeasured:
        est = ", ".join(f"{k} ~{costs[k].estimate_ms:.2f} ms" for k in sorted(costs)
                        if costs[k].estimate_ms is not None)
        return today, (f"{FORM_CHOICE_MARKER} UNMEASURED -- keeps today's form {today}; missing "
                       + "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(unmeasured.items()))
                       + (f" (estimate, not a decision: {est})" if est else ""))
    best = min(costs, key=lambda k: costs[k].round_ms)
    table = ", ".join(f"{k} {costs[k].round_ms:.2f} ms" for k in sorted(costs))
    return best, f"{FORM_CHOICE_MARKER} CHOSEN {best} (decode round: {table})"


# --------------------------------------------------------------------------
# Form B (FORM-B-F6-ENTWURF-0928.md v2 §1-2). Its own plan, NOT a second host:
# RankRolePlan keeps exactly one host. The scheduler group `tp` stays ALL
# ranks (NF objection 1); the model's linear/vocab collectives run on a
# separate `model_tp` group over the weight ranks only.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FormBPlan:
    weight_ranks: Tuple[int, ...]
    kv_ranks: Tuple[int, ...]
    weights: Tuple[int, ...]

    @property
    def lead(self) -> int:
        """Draft (solo), QSA index and the vision lease live here."""
        return self.weight_ranks[0]

    @property
    def world(self) -> int:
        return len(self.weights)

    def model_tp_partition(self) -> List[List[int]]:
        """The `model_tp` group list every rank passes to new_group (torch
        rule: all ranks build every group): W as one group, each KV-only rank
        in a group of its own -- it never enters a model_tp collective."""
        return [list(self.weight_ranks)] + [[r] for r in self.kv_ranks]

    def model_tp_ratio(self) -> List[int]:
        """The uneven TP vector INSIDE model_tp (weight ranks only)."""
        return [self.weights[r] for r in self.weight_ranks]

    def scheduler_tp_ranks(self) -> List[int]:
        """NF objection 1: request/host exchange, prefetch/claim and the
        tp_match_floor MINs run over ALL ranks -- a KV-only rank must join."""
        return list(range(self.world))

    def dcp_ranks(self) -> List[int]:
        """A, T, Q, M and the spec channel (draft_block, accept, spec_k)."""
        return list(range(self.world))

    def index_ranks(self) -> List[int]:
        """NF answer 3: the QSA index only on lead; W minus lead behaves like K."""
        return [self.lead]


def form_b_plan(form: RankForm) -> FormBPlan:
    if form.kind != FORM_B:
        raise ValueError(f"form_b_plan: form {form.kind} is not B ({form.line()})")
    return FormBPlan(weight_ranks=form.weight_ranks, kv_ranks=form.kv_only_ranks,
                     weights=form.weights)


# --------------------------------------------------------------------------
# F6 step 6: the BAR1 window riegel of Form B (draft v2 §5 "Fenster").
#
# Every rank process pins one BAR1 receive window per barlink group it is a
# member of (world size > 1), on ITS card
# (device_communicators/barlink_matrix_transport.py: _requested, the per-group
# key from the live group name, e.g. ``dcp:0`` -> DCP_0). Form B adds one
# group, ``model_tp:0``, and only the weight ranks W are in it (each KV-only
# rank sits alone -> no communicator, no window). The flip lane (weight
# exchange, 32-MiB slots) runs only where weights live. So a W card carries
# dcp:0 + model_tp:0 + flip lane on top of the group's other windows, a K
# card only the shared ones -- and the W-3080 is where the aperture binds
# (#1234 C1: 224 of 256 MiB usable per 3080, measured Used 224/256 with D at
# 16+32+40 and P at 24+96 -- already exhausted before Form B adds a window).
# A configuration that does not fit is refused BY NAME before any rank loads,
# instead of a Bar1WindowRefused at the model_tp build on metal.
# --------------------------------------------------------------------------

class RankFormBar1Window(RankFormError):
    code = "W187 Weg2RankFormBar1Window"


#: #1234 C1 (launcher, D argv): dcp:0's measured window.
FORM_B_DCP_WINDOW_MIB = 40
#: The flip lane's slot (draft v2 §5; the bar1 weight-exchange lanes carry
#: 32-MiB slots). Charged once per WEIGHT rank.
FORM_B_FLIP_LANE_MIB = 32
#: #1234 C1: usable BAR1 per RTX 3080 (256 gross minus the RM carve-out,
#: evaluated by barlink as NVML free minus RESERVE_MIB_DEFAULT 32).
BAR1_USABLE_MIB_3080 = 224
#: The window barlink requests for a group nobody configured
#: (barlink_matrix_transport.WINDOW_MIB_DEFAULT).
BARLINK_WINDOW_MIB_DEFAULT = 96


def _group_key(group: str) -> str:
    """Same spelling as barlink_matrix_transport._group_key (``dcp:0`` -> DCP_0)."""
    return "".join(c if c.isalnum() else "_" for c in group).upper()


def parse_window_spec(spec: Optional[str]) -> Tuple[int, Dict[str, int]]:
    """``--barlink-bar1-window-mib``: a bare default plus GROUP=MiB overrides
    (server_args publishes exactly these keys). None = barlink's own default."""
    default, own = BARLINK_WINDOW_MIB_DEFAULT, {}
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "=" in part:
                g, _, v = part.partition("=")
                own[g.strip().upper()] = int(v)
            else:
                default = int(part)
        except ValueError:
            raise _refuse(RankFormBar1Window, f"window spec {spec!r}: {part!r} is not GROUP=MiB or MiB")
    return default, own


def form_b_window_requirement(
    plan: FormBPlan,
    card_of_rank: Sequence[str],
    *,
    window_spec: Optional[str],
    groups_all_ranks: Sequence[str] = ("world:0", "tp:0", "dcp:0"),
    flip_lane_mib: int = FORM_B_FLIP_LANE_MIB,
    resident_mib_by_card: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> Dict[str, Dict[str, int]]:
    """Per card: every BAR1 window post this Form B group pins there, named.

    ``groups_all_ranks`` are the barlink groups every rank is a member of (the
    D group today: world, the scheduler tp, dcp); ``model_tp:0`` is added for
    each weight rank when |W| >= 2, the flip lane likewise. A card hosting two
    ranks pins every window twice (one region per process). ``resident`` are
    the other tenants' posts on a card (group P's 24 + PP_0 96 on a 3080 in
    the Weg-2 layout), taken as given."""
    if len(card_of_rank) != plan.world:
        raise _refuse(RankFormBar1Window,
                      f"card map {list(card_of_rank)} names {len(card_of_rank)} ranks, the form {plan.world}")
    default, own = parse_window_spec(window_spec)

    def window(group: str) -> int:
        return own.get(_group_key(group), default)

    out: Dict[str, Dict[str, int]] = {}
    for card, posts in (resident_mib_by_card or {}).items():
        out.setdefault(str(card), {}).update({f"resident {k}": int(v) for k, v in posts.items()})
    for r in range(plan.world):
        card = str(card_of_rank[r])
        posts = out.setdefault(card, {})
        mine = list(groups_all_ranks)
        if r in plan.weight_ranks and len(plan.weight_ranks) >= 2:
            mine.append("model_tp:0")
        for g in mine:
            posts[f"rank{r} {g}"] = window(g)
        if r in plan.weight_ranks and flip_lane_mib:
            posts[f"rank{r} flip_lane"] = int(flip_lane_mib)
    return out


def check_form_b_windows(
    plan: FormBPlan,
    card_of_rank: Sequence[str],
    usable_mib_by_card: Mapping[str, int],
    **kw,
) -> Dict[str, Dict[str, int]]:
    """The riegel: every card's posts must fit its usable BAR1 aperture, or a
    named W187 listing card, ranks, posts, sum and usable. Returns the posts
    (for the launch line) when everything fits."""
    posts = form_b_window_requirement(plan, card_of_rank, **kw)
    over = []
    for card, p in posts.items():
        if card not in usable_mib_by_card:
            raise _refuse(RankFormBar1Window, f"no usable BAR1 size for card {card!r}")
        need, have = sum(p.values()), int(usable_mib_by_card[card])
        if need > have:
            ranks = sorted({k.split()[0] for k in p if k.startswith("rank")})
            over.append(f"card {card} ({', '.join(ranks)}): "
                        + " + ".join(f"{k} {v}" for k, v in p.items())
                        + f" = {need} MiB > usable {have} MiB")
    if over:
        raise _refuse(RankFormBar1Window,
                      "Form B's BAR1 windows do not fit: " + "; ".join(over)
                      + ". Shrink a window (e.g. TP_0: under Form B the scheduler tp group carries "
                      "control traffic only; MODEL_TP_0 sized to the o_proj/MLP all-reduce) or move a rank.")
    return posts


# --------------------------------------------------------------------------
# F6 step 2c: the construction context of a Form B rank.
#
# Every parallel Linear caches tp_size / tp_rank from the parallel context
# when it is BUILT (layers/linear.py: get_parallel().tp_size), and the
# attention projections cache attn_tp_size / attn_tp_rank; RowParallelLinear
# gates its all-reduce on the cached value. Under Form B a weight rank must
# therefore be built as one of |W| ranks with its index in W -- never as one
# of all N (which would size its shard against the KV-only ranks' zero and
# hand the all-reduce a group of N). The shard plan is re-scoped the same way:
# ``tp_partition_sizes`` applies a vector only when len == tp_size, so the
# weight vector restricted to W is installed for the build. The model's
# collectives then go to model_tp (2b), the attention layers' to
# get_attn_tp_layer_group() (2c). Same mechanism as the weightless lane's
# construction override (model_runner: _wl_build_ctx), with |W| instead of 1.
# --------------------------------------------------------------------------

class RankFormKvRankBuild(RankFormError):
    code = "W188 Weg2RankFormKvRankBuildNotWired"


def form_b_build_override(
    partition: Sequence[Sequence[int]],
    rank: int,
    base_ratios: Optional[Sequence[int]] = None,
    families: Optional[Mapping[str, Sequence[int]]] = None,
) -> Tuple[Dict[str, int], Optional[List[int]], Dict[str, List[int]]]:
    """(parallel override, W-restricted base vector, W-restricted families)
    for a WEIGHT rank of the installed model_tp partition.

    ``base_ratios`` / ``families`` are the process plan over ALL ranks (a
    KV-only rank's entry is 0); None = even split, and it stays even over W.
    A KV-only rank is refused by name (W188): its construction (meta model, no
    weight load) and its attention-only forward belong to seam F15."""
    multi = [list(p) for p in partition if len(p) >= 2]
    if len(multi) != 1:
        raise _refuse(RankFormShapeMismatch,
                      f"model_tp partition {[list(p) for p in partition]} has no single weight group")
    w = sorted(multi[0])
    if rank not in w:
        raise _refuse(
            RankFormKvRankBuild,
            f"rank {rank} is a KV-only rank of Form B (weight ranks {w}); building it needs the "
            "KV-only construction path (meta model, attention-only forward) -- rank_role seam F15, "
            "not wired. Refused before any layer is built.")
    i = w.index(rank)
    override = dict(tp_size=len(w), tp_rank=i, attn_tp_size=len(w), attn_tp_rank=i)
    world = sum(len(p) for p in partition)

    def restrict(vec: Optional[Sequence[int]], what: str) -> Optional[List[int]]:
        if vec is None:
            return None
        vec = list(vec)
        if len(vec) != world:
            raise _refuse(RankFormShapeMismatch, f"{what} {vec} names {len(vec)} ranks, the form {world}")
        bad = [r for r in range(world) if (r in w) != (vec[r] > 0)]
        if what == "weight vector" and bad:
            raise _refuse(RankFormShapeMismatch,
                          f"weight vector {vec} disagrees with the weight ranks {w} at rank(s) {bad}")
        return [vec[r] for r in w]

    return (override, restrict(base_ratios, "weight vector"),
            {k: restrict(v, f"family {k!r}") for k, v in (families or {}).items() if v})


def form_b_build_context(rank: int):
    """The context a Form B rank builds its model under (model_runner), or
    None when no Form B partition is installed (the caller keeps its own)."""
    import contextlib

    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.utils import (
        get_tp_partition_families,
        get_tp_partition_ratios,
        scoped_tp_partition_ratios,
    )
    from sglang.srt.runtime_context import get_parallel

    partition = ps.get_model_tp_partition()
    if partition is None:
        return None
    override, base, fams = form_b_build_override(
        partition, rank, get_tp_partition_ratios(), get_tp_partition_families())

    @contextlib.contextmanager
    def _ctx():
        with get_parallel().override(**override), \
                scoped_tp_partition_ratios(base, fams or None, allow_zero=False):
            yield

    return _ctx()
