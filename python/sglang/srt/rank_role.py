# SPDX-License-Identifier: Apache-2.0
"""Form A rank ROLES, and a named refusal at every seam that is not yet wired.

Under Form A a rank is one of two things and the difference is total, not
gradual:

  HOST    -- runs every dense part (attention + QSA indexer, GDN/linear_attn,
             o_proj, the hyper-connection mixer, norms, embeddings, PLE,
             lm_head), holds the whole KV for the full context, the GDN
             recurrent states, the MTP draft and the graphs, plus its own
             share of the experts.
  WORKER  -- holds its own experts and NOTHING else. No dense weights, no KV
             pool, no draft, no dense-family graphs.

The role is what makes the zero legible. `--rank-tp-ratio 1,0,0` on its own
is ambiguous: a zero could be an arithmetic accident (and in every caller
that predates Form A it IS one). `--rank-role host,worker,worker` says the
zero is a LAYOUT, and the two must agree -- which is checked here.

WHY THE SEAM REGISTRY EXISTS. Form A touches eleven places -- the nine the
design note started with plus two the slice-2 survey of the call sites found
-- and only some are built. The failure mode this module is written against is the one that costs
a GPU window: an unwired seam that does not refuse, so the boot gets to layer
three of forty-eight before it dies with an `IndexError` on an empty tensor,
and the log says nothing about Form A. Every seam therefore has an entry with
its file:line and a `wired` flag, and the guards below raise
`FormASeamNotWired` naming the seam, what it would take, and where. A seam
that is wired loses its guard; a seam that is not keeps it until it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "HOST",
    "WORKER",
    "ROLES",
    "RankRoleError",
    "FormASeamNotWired",
    "RankRolePlan",
    "parse_rank_roles",
    "SEAMS",
    "Seam",
    "require_wired",
    "guard_kv_pool",
    "guard_draft_worker",
    "guard_dense_weights",
    "guard_collective_subgroup",
    "guard_graph_mode",
    "guard_zero_width_linear",
    "guard_dcp_merge",
    "UNWIRED_ORDER",
    "resolve_dcp_under_host_kv",
]

HOST = "host"
WORKER = "worker"
ROLES = (HOST, WORKER)


class RankRoleError(ValueError):
    """The role vector is not a Form A role vector."""


class FormASeamNotWired(NotImplementedError):
    """A Form A seam this configuration needs has not been built yet.

    Deliberately loud and deliberately early: the alternative is a boot that
    dies forty-five layers later with a shape error that names nothing.
    """


# ---------------------------------------------------------------------------
# The seams, as data. file:line are against this worktree (546dd2bd36 plus
# the slice-1/2 commits); F10 and F11 came out of the slice-2 survey.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Seam:
    id: str
    what: str
    where: str
    wired: bool
    note: str = ""

    def refuse(self, context: str = "") -> "FormASeamNotWired":
        tail = f" Context: {context}" if context else ""
        return FormASeamNotWired(
            f"Form A seam {self.id} is not wired: {self.what}. "
            f"It lives at {self.where}. {self.note}{tail}"
        )


SEAMS: Dict[str, Seam] = {
    s.id: s
    for s in (
        Seam(
            "F1",
            "a zero entry in the dense ratio vector, admitted by the flag layer",
            "server_args.py:11866 (--rank-tp-ratio entries must be positive), "
            "distributed/utils.py:_normalize_partition_plan",
            wired=True,
            note="Slice 2: admitted only together with an explicit "
            "--rank-role vector, so an accidental zero still raises.",
        ),
        Seam(
            "F2",
            "a zero-width rank in the unit partition (q packets, kv heads, "
            "GDN heads, o_proj, mixer, vocab)",
            "distributed/utils.py:_partition_units_with_empty_ranks, "
            "partition_units(allow_zero=), tp_partition_sizes",
            wired=True,
            note="Slice 1 opened the arithmetic, slice 2 threaded it through "
            "the plan; the kv-group alignment composes.",
        ),
        Seam(
            "F3",
            "a worker rank that never LOADS the dense weights (not merely "
            "idles through them)",
            "models/qwen4_exp.py load_weights / _build_embed_tokens",
            wired=False,
            note="Needs a load-time filter keyed on the role, and the "
            "census must then show 'experts' alone on a worker. This is the "
            "slice with the largest VRAM gain (2 x 1.84 GiB).",
        ),
        Seam(
            "F4",
            "KV only on the host: a worker builds no KV pool and owns no "
            "context tokens",
            "distributed/utils.py:1430-1431 cp_token_context_budget "
            "(assert all(v > 0), then capacities[r] // vector[r]), "
            ":1180 the token-vector validation, "
            "model_executor/pool_configurator.py:167 (if ratio_r > 0: skips "
            "the draft-KV cell correction SILENTLY)",
            wired=False,
            note="resolve_dcp_under_host_kv() below resolves the POLICY "
            "(dcp off, replication off); the pool-construction half is open. "
            "The pool_configurator line is the dangerous one -- it does not "
            "refuse, it mis-scales.",
        ),
        Seam(
            "F5",
            "the DCP LSE merge and head gather with a zero-head rank",
            "layers/dcp/comm.py:196-199 (assertion on head_counts[rank]), "
            ":222 cp_local_head_bounds",
            wired=False,
            note="Under Form A the merge should not run at all (the host "
            "holds every head); the refusal exists so that a configuration "
            "that still reaches it says so instead of asserting.",
        ),
        Seam(
            "F6",
            "a collective over a SUBSET of the ranks",
            "distributed/parallel_state.py:1042-1068 (one communicator per "
            "GroupCoordinator; barlink knows no subgroups)",
            wired=False,
            note="Form A's decode path needs no subgroup as long as the MoE "
            "exchange spans all ranks; it becomes necessary when the dense "
            "layers want a collective the workers must not join.",
        ),
        Seam(
            "F7",
            "a DIRECTED reduce to the host instead of an all-reduce",
            "layers/moe/host_moe_exchange.py (built), over "
            "distributed/device_communicators/barlink_bar1.py:3270 (put)",
            wired=True,
            note="Slice 3. barlink's facade still has no reduce; the "
            "exchange builds one from put and NAMES the all_reduce fallback "
            "rather than degrading silently.",
        ),
        Seam(
            "F8",
            "planner VRAM posts booked per ROLE, not symmetrically",
            "form_a_plan.py (built), against uneven_perf.py:196-197 "
            "(_SOLO_HOST_* is the precedent) and :201 "
            "(_PREDICT_MIN_RANK_TOKENS=4096 declares a KV-less rank "
            "infeasible)",
            wired=True,
            note="Slice 1 built the solve; the predictor's minimum-token "
            "rule still has to learn about worker ranks.",
        ),
        Seam(
            "F9",
            "CUDA-graph mode chosen per role (host captures the dense "
            "families, a worker captures only its expert route)",
            "layers/moe/offload_capture_gate.py:236/249 (process-wide env "
            "decision)",
            wired=False,
            note="Deferrable: the first Form A probe boot can run decode "
            "EAGER. That costs throughput but measures the per-layer chain, "
            "which is what the probe is for.",
        ),
        # ---- found by the slice-2 seam survey, not in the original nine ----
        Seam(
            "F10",
            "the W62 saturation refusal rests on the written assumption "
            "that a zero-head rank CANNOT happen",
            "weg2/launcher.py:7203-7207 (\"A rank owning zero heads cannot "
            "happen; asserting against zero heads would be a guard that can "
            "never fire\"), refusal at :7217-7285",
            wired=False,
            note="Form A makes that assumption false. Until the refusal "
            "learns about worker ranks it will reject every Form A vector as "
            "'axis switched off' -- a guard that was correct becoming a "
            "blocker is exactly the class that eats a GPU window.",
        ),
        Seam(
            "F11",
            "a zero-width parallel Linear COMPUTES instead of being skipped",
            "layers/linear.py:2069-2092 (row-parallel "
            "input_size_per_partition can be a true 0 on the element path), "
            ":670-699 (column-parallel output_partition_sizes=[0]), "
            "distributed/utils.py:1790 assert_activation_aligned_shards "
            "(0 % 8 == 0, so the only activation guard passes a zero "
            "silently)",
            wired=False,
            note="THE DANGEROUS ONE. F.linear with K=0 does not raise, it "
            "returns zeros, and the all-reduce adds them -- a wrong result "
            "that is only visible at the output. Under Form A a worker must "
            "not CONSTRUCT these layers at all (that is F3); this seam is "
            "the backstop for the case where one slips through.",
        ),
    )
}

#: The seams that must be wired before a Form A boot can be believed, in the
#: order the survey found them knocking. Kept as data so a report can print
#: the remaining work without re-deriving it.
UNWIRED_ORDER: Tuple[str, ...] = ("F3", "F11", "F4", "F10", "F5", "F9", "F6")


def require_wired(seam_id: str, context: str = "") -> None:
    """Raise unless seam `seam_id` is built. The one-line guard."""
    try:
        seam = SEAMS[seam_id]
    except KeyError:
        raise RankRoleError(
            f"unknown Form A seam {seam_id!r}; known: {sorted(SEAMS)}"
        ) from None
    if not seam.wired:
        raise seam.refuse(context)


# ---------------------------------------------------------------------------
# The role vector
# ---------------------------------------------------------------------------
def parse_rank_roles(value: str, tp_size: Optional[int] = None) -> Tuple[str, ...]:
    """Parse ``--rank-role host,worker,worker``."""
    parts = [p.strip().lower() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise RankRoleError("--rank-role is empty.")
    bad = [p for p in parts if p not in ROLES]
    if bad:
        raise RankRoleError(
            f"--rank-role entries must be one of {list(ROLES)}, got {bad}."
        )
    if tp_size is not None and len(parts) != tp_size:
        raise RankRoleError(
            f"--rank-role has {len(parts)} entries but --tp-size is "
            f"{tp_size}; one role per rank."
        )
    return tuple(parts)


@dataclass(frozen=True)
class RankRolePlan:
    """Which rank is the attention host, and what follows from that."""

    roles: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.roles) < 2:
            raise RankRoleError(
                f"Form A needs at least two ranks, got {list(self.roles)}. "
                "With one rank there is no worker and the layout is the "
                "ordinary single-GPU one."
            )
        hosts = [r for r, role in enumerate(self.roles) if role == HOST]
        if len(hosts) != 1:
            raise RankRoleError(
                f"Form A has exactly ONE attention host; {list(self.roles)} "
                f"names {len(hosts)} ({hosts}). Two hosts would mean the "
                "dense side is sharded again, which is the layout Form A "
                "replaces; none means nobody computes attention."
            )

    @classmethod
    def parse(cls, value: str, tp_size: Optional[int] = None) -> "RankRolePlan":
        return cls(parse_rank_roles(value, tp_size))

    @property
    def tp_size(self) -> int:
        return len(self.roles)

    @property
    def host_rank(self) -> int:
        return self.roles.index(HOST)

    @property
    def worker_ranks(self) -> List[int]:
        return [r for r, role in enumerate(self.roles) if role == WORKER]

    def is_host(self, rank: int) -> bool:
        return self.role_of(rank) == HOST

    def is_worker(self, rank: int) -> bool:
        return self.role_of(rank) == WORKER

    def role_of(self, rank: int) -> str:
        if not 0 <= rank < self.tp_size:
            raise RankRoleError(
                f"rank {rank} is outside the role vector {list(self.roles)}."
            )
        return self.roles[rank]

    # -- what the role IMPLIES ------------------------------------------
    def dense_ratio(self) -> List[int]:
        """The base ``--rank-tp-ratio`` the roles mean: the host owns every
        dense unit, a worker owns none."""
        return [1 if role == HOST else 0 for role in self.roles]

    def check_dense_ratio(self, ratio: Sequence[int]) -> None:
        """Refuse a ratio vector that disagrees with the roles.

        The two must say the same thing, and the direction of the check
        matters: a worker with a non-zero dense share would silently load
        dense weights onto a card that Form A budgeted for experts alone,
        and the first sign of it would be an OOM on a 3080.
        """
        if len(ratio) != self.tp_size:
            raise RankRoleError(
                f"--rank-tp-ratio has {len(ratio)} entries but --rank-role "
                f"has {self.tp_size}."
            )
        wrong_worker = [
            r for r in self.worker_ranks if int(ratio[r]) != 0
        ]
        if wrong_worker:
            raise RankRoleError(
                f"rank(s) {wrong_worker} are --rank-role worker but "
                f"--rank-tp-ratio {list(ratio)} gives them a dense share. A "
                "worker holds experts and nothing else; give it 0, or make "
                "it a host."
            )
        if int(ratio[self.host_rank]) <= 0:
            raise RankRoleError(
                f"rank {self.host_rank} is the --rank-role host but "
                f"--rank-tp-ratio {list(ratio)} gives it no dense share; "
                "then nobody computes attention."
            )


# ---------------------------------------------------------------------------
# Guards -- called from the places that would otherwise fail late and mutely
# ---------------------------------------------------------------------------
def guard_dense_weights(plan: RankRolePlan, rank: int) -> None:
    """Called where a rank is about to LOAD dense weights (F3)."""
    if plan.is_worker(rank):
        require_wired(
            "F3",
            f"rank {rank} is a Form A worker and must not load dense "
            "weights, but the load path has no role filter yet.",
        )


def guard_kv_pool(plan: RankRolePlan, rank: int, tokens: int) -> None:
    """Called where a rank sizes its KV pool (F4)."""
    if plan.is_worker(rank) and tokens:
        require_wired(
            "F4",
            f"rank {rank} is a Form A worker but was handed {tokens} KV "
            "tokens; under Form A the host owns the whole context.",
        )


def guard_draft_worker(plan: RankRolePlan, rank: int) -> None:
    """Called where a rank builds the MTP draft model (F3, draft half)."""
    if plan.is_worker(rank):
        require_wired(
            "F3",
            f"rank {rank} is a Form A worker and must not build the MTP "
            "draft; the draft is unsharded on the host.",
        )


def guard_collective_subgroup(plan: RankRolePlan, name: str) -> None:
    """Called where a collective must span only the dense ranks (F6)."""
    require_wired(
        "F6",
        f"collective {name!r} would have to run over the host alone, but "
        "the transport has no subgroup.",
    )


def guard_graph_mode(plan: RankRolePlan, rank: int, mode: str) -> None:
    """Called where a rank picks its CUDA-graph mode (F9).

    Eager is explicitly allowed and is the first probe boot's form -- what
    is refused is a CAPTURED dense graph on a rank that holds no dense
    weights, because that captures nothing and hides it.
    """
    if plan.is_worker(rank) and mode not in ("eager", "disabled", None):
        require_wired(
            "F9",
            f"rank {rank} is a Form A worker but was given graph mode "
            f"{mode!r}; the graph decision is still process-wide.",
        )


def guard_zero_width_linear(
    plan: RankRolePlan, rank: int, name: str, width: int
) -> None:
    """Called where a parallel Linear is CONSTRUCTED with its shard width.

    The backstop for F11. A width of 0 is not an error in itself -- under
    Form A it is the correct answer for a worker -- but it must lead to the
    layer being SKIPPED, not built: ``F.linear`` with K=0 returns zeros
    without raising, and the all-reduce adds them, so the only symptom is a
    wrong output forty-eight layers later.
    """
    if width > 0:
        return
    require_wired(
        "F11",
        f"rank {rank} would build the parallel layer {name!r} with shard "
        f"width 0. Under Form A a worker must not construct it at all "
        f"(F3); a zero-width layer that runs is a silent wrong answer.",
    )


def guard_dcp_merge(plan: RankRolePlan, rank: int) -> None:
    """Called if a rank reaches the DCP LSE merge under Form A (F5)."""
    require_wired(
        "F5",
        f"rank {rank} reached the DCP LSE merge, but under Form A the host "
        "holds every head and there is nothing to merge.",
    )


# ---------------------------------------------------------------------------
# DCP under a host-held KV
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DcpResolution:
    dcp_size: int
    uneven_dcp_kv_replicated: bool
    reason: str


def resolve_dcp_under_host_kv(
    plan: RankRolePlan,
    requested_dcp_size: Optional[int],
    requested_replicated: Optional[bool],
    forced: bool = False,
) -> DcpResolution:
    """Under Form A the host holds the WHOLE KV, so DCP has nothing to do.

    Decode context parallelism exists to spread the KV of one context over
    several ranks and merge the partial attentions by LSE. Form A removes
    its premise: there is one rank with heads and one rank with KV, and it
    is the same rank. So `dcp_size` collapses to 1 and the replicated-KV
    geometry is off -- and that is a RESOLUTION, not a default, which is why
    it is returned with its reason attached rather than written somewhere.

    `forced` (an explicit --dcp-size / SGLANG_UNEVEN_DCP from the operator)
    is refused rather than overridden: silently ignoring an explicit flag is
    how a measurement ends up comparing two different layouts.
    """
    if forced and requested_dcp_size not in (None, 1):
        raise RankRoleError(
            f"--dcp-size {requested_dcp_size} was set explicitly, but under "
            f"Form A rank {plan.host_rank} holds the whole KV and every "
            "attention head -- there is no second rank to merge with. Drop "
            "the flag, or drop the worker roles."
        )
    if forced and requested_replicated:
        raise RankRoleError(
            "the replicated-KV DCP geometry was requested explicitly "
            "(SGLANG_UNEVEN_DCP / --uneven-dcp-kv-replicated), but under "
            "Form A there is nothing to replicate across: one rank owns "
            "every head and every token."
        )
    return DcpResolution(
        dcp_size=1,
        uneven_dcp_kv_replicated=False,
        reason=(
            f"Form A: rank {plan.host_rank} is the attention host and holds "
            f"the whole KV; ranks {plan.worker_ranks} hold none, so there is "
            "no token axis to split and no LSE merge to run."
        ),
    )
