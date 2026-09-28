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
(``weg2.form_measures/1``, written by devtools/dcp_exchange_bench.py); a
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

MEASURES_SCHEMA = "weg2.form_measures/1"
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
            argv += ["--dcp-size", n, "--weightless-kv-fastlane",
                     "--weightless-kv-head-rank", str(self.head_rank)]
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


def resolve_rank_form(
    weights: Sequence,
    tokens: Optional[Sequence] = None,
    *,
    rank_role: Optional[str] = None,
    dense: bool = True,
    subgroup_tp_wired: bool = False,
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
    else:
        kind = FORM_B
        if not subgroup_tp_wired:
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
    EAGLE-family chain only (server_args._reject_unsupported_weightless_spec).
    DFLASH (the 27B's DFlash2) has its own round shape the workers do not
    mirror yet (gap G-A1): refused here by name, before the serve chain would."""
    if form.backend != BACKEND_LANE or not algorithm:
        return
    a = str(algorithm).upper()
    if a not in ("EAGLE", "EAGLE3", "NEXTN"):
        raise _refuse(
            RankFormLaneSpec,
            f"form A runs on the weightless-KV lane, which mirrors the EAGLE/EAGLE3/NEXTN "
            f"chain only; got --speculative-algorithm {algorithm} (DFLASH on the lane is gap G-A1)")


# --------------------------------------------------------------------------
# The planner's time model (decode round, bs rows) over measured prices.
# --------------------------------------------------------------------------

@dataclass
class FormMeasures:
    """``weg2.form_measures/1``: the prices the model needs, each keyed so a
    missing one is visible (never a default)."""

    allreduce_us: Dict[str, Dict[int, float]] = field(default_factory=dict)
    dcp_exchange_us: Dict[str, Dict[int, float]] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def load(cls, path: Optional[str]) -> "FormMeasures":
        if not path or not os.path.exists(path):
            return cls(source="")
        with open(path) as f:
            doc = json.load(f)
        if doc.get("schema") != MEASURES_SCHEMA:
            raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {MEASURES_SCHEMA!r}")

        def table(key: str) -> Dict[str, Dict[int, float]]:
            out: Dict[str, Dict[int, float]] = {}
            for cards, per_bs in (doc.get(key) or {}).items():
                out[cards] = {int(bs): float(v["graph_median_us"] if v.get("graph_median_us")
                                             is not None else v["eager_median_us"])
                              for bs, v in per_bs.items()
                              if v.get("graph_median_us") is not None
                              or v.get("eager_median_us") is not None}
            return out

        return cls(allreduce_us=table("allreduce"), dcp_exchange_us=table("dcp_exchange"),
                   source=path)


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
) -> FormCost:
    """T_D(form) = max_r(weight_read_r * share_r + attn_r * tokshare_r)
                   + n_allreduce * t_ar(TP ranks) + n_attn_layers * t_dcp(DCP ranks)
                   + fixed.

    ``weight_read_ms[r]``: rank r reading ALL weights (its share scales it);
    ``attn_ms[r]``: rank r attending over ALL KV of the round (its token share
    scales it). Collective prices come ONLY from ``measures``; a price the form
    needs and the file lacks is named in ``missing`` and the round is None."""
    missing: List[str] = []
    wsum = float(sum(form.weights))
    tok = form.tokens if form.tokens is not None else tuple(1 for _ in form.weights)
    tsum = float(sum(tok))
    per_rank = [float(weight_read_ms[r]) * form.weights[r] / wsum
                + float(attn_ms[r]) * tok[r] / tsum for r in range(form.world)]
    terms: Dict[str, float] = {"compute_ms": max(per_rank), "fixed_ms": float(fixed_ms)}
    tp_uuids = [uuids[r] for r in form.weight_ranks]
    if len(tp_uuids) > 1:
        ar = measures.allreduce_us.get(cards_key(tp_uuids), {}).get(int(bs))
        if ar is None:
            missing.append(f"allreduce[{cards_key(tp_uuids)}][bs{bs}]")
        else:
            terms["allreduce_ms"] = n_allreduce * ar / 1000.0
    if form.world > 1:
        # every candidate token-shards the full-attention KV over all ranks
        # (today's C runs uneven DCP too); the bench's dcp mode prices the
        # host-broadcast shape of A, an upper bound for C's head-split gather
        dk = measures.dcp_exchange_us.get(cards_key(uuids), {}).get(int(bs))
        if dk is None:
            missing.append(f"dcp_exchange[{cards_key(uuids)}][bs{bs}]")
        else:
            terms["dcp_ms"] = n_attn_layers * dk / 1000.0
    total = None if missing else sum(terms.values())
    return FormCost(form=form.kind, round_ms=total, terms=terms, missing=tuple(missing))


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
        return today, (f"{FORM_CHOICE_MARKER} UNMEASURED -- keeps today's form {today}; missing "
                       + "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(unmeasured.items())))
    best = min(costs, key=lambda k: costs[k].round_ms)
    table = ", ".join(f"{k} {costs[k].round_ms:.2f} ms" for k in sorted(costs))
    return best, f"{FORM_CHOICE_MARKER} CHOSEN {best} (decode round: {table})"
