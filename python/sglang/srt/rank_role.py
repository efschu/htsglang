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

import re
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
    "guard_zero_width_linear_shard",
    "form_a_dense_is_unsharded",
    "FormAZeroWidthLinear",
    "FormAWorkerBuildsDraft",
    "worker_keeps_parameter",
    "guard_dcp_merge",
    "UNWIRED_ORDER",
    "resolve_dcp_under_host_kv",
]

HOST = "host"
WORKER = "worker"
ROLES = (HOST, WORKER)


class RankRoleError(ValueError):
    """The role vector is not a Form A role vector."""


class FormAZeroWidthLinear(RankRoleError):
    """A parallel Linear was constructed with a shard width of zero.

    Its OWN class, and deliberately not a FormASeamNotWired: the refusal IS
    the feature here, not a placeholder for missing work. Tying it to the
    seam's wired flag would have made the backstop evaporate the moment the
    seam was marked built -- which is exactly what happened on the first
    attempt and is the reason this class exists.
    """


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
    #: (path under python/sglang/srt, line, a substring that must be AT or
    #: within a few lines of it). This is what makes the registry
    #: self-checking: `where` is prose a reader trusts, `anchors` is the same
    #: claim in a form a test can falsify. Measured need for it: in slice 5
    #: the F4 entry still cited utils.py:1430 and :1180, both of which had
    #: become other code -- a seam list whose file:line have drifted is worse
    #: than no seam list, because it is believed.
    anchors: Tuple[Tuple[str, int, str], ...] = ()

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
            "server_args.py:12023 (--rank-tp-ratio entries must be positive), "
            ":11995 (--rank-role requires an explicit vector), "
            "distributed/utils.py:_normalize_partition_plan",
            wired=True,
            note="Slice 2: admitted only together with an explicit "
            "--rank-role vector, so an accidental zero still raises.",
            anchors=(
                ("server_args.py", 12023, "--rank-tp-ratio entries must be"),
                ("server_args.py", 11995, "--rank-role requires an explicit"),
            ),
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
            anchors=(
                ("distributed/utils.py", 1357,
                 "def _partition_units_with_empty_ranks"),
                ("distributed/utils.py", 124, "def set_tp_partition_ratios"),
            ),
        ),
        Seam(
            "F3",
            "a worker rank that never LOADS the dense weights (not merely "
            "idles through them)",
            "models/qwen4_exp.py weight_name_needed (LOAD veto), :1497 / "
            ":1523 / :1531 (ple, both hyper_connections), qwen3_5.py "
            "(linear_attn, qkv_proj, o_proj, attn), qwen2_moe.py "
            "(shared_expert), qwen3_vl.py (lm_head)",
            wired=True,
            note="BUILT, all three parts. (1) The loader veto: a worker "
            "never READS a dense tensor (checked against the real weight "
            "map -- 221.184 of 225.300 names kept, no shared expert, no "
            "draft, no vision; the ROUTER was added to the kept set in "
            "slice 6a, see _ROUTER_MARKER). (2) The CONSTRUCTION skip at "
            "the posts that cost most -- the two per-layer hyper-connection "
            "mixers and the model-level one (not sharded at all, so every "
            "rank held them in full: 0.63 GiB per rank in INT8) and the "
            "PLE. (3) The rest of the construction skip: linear_attn, "
            "qkv_proj, o_proj and the RadixAttention layer in qwen3_5.py, "
            "the QSA indexer in qwen4_exp.py, the SHARED expert in "
            "qwen2_moe.py, embed_tokens and lm_head. Acceptance is "
            "form_a_construction.expected_census_categories('worker'), "
            "which slice 6a widened from ('experts',) to "
            "('experts', 'moe_gate') with its reason attached.",
            anchors=(
                ("models/qwen4_exp.py", 2280, "this_rank_is_form_a_worker()"),
                ("models/qwen4_exp.py", 1539, 'skip_on_worker("ple"'),
                ("models/qwen4_exp.py", 1565,
                 'skip_on_worker("hyper_connection"'),
                ("models/qwen3_5.py", 807, 'skip_on_worker("linear_attn"'),
                ("models/qwen3_5.py", 1120, 'skip_on_worker("self_attn"'),
            ),
        ),
        Seam(
            "F13",
            "the MODEL-LEVEL vocab collectives must fall with the vocab "
            "sharding, exactly as F12's per-layer ones fall with the dense "
            "sharding",
            "layers/vocab_parallel_embedding.py:730-732 (the embedding "
            "all-reduce), layers/logits_processor.py (the logits "
            "all-gather), models/qwen3_vl.py:1358-1368 (the host's lm_head "
            "built unsharded), against distributed/utils.py:1734 "
            "tp_vocab_ratios "
            "(\"vocab always even\" -- the vocab family deliberately does "
            "NOT inherit the base ratio vector)",
            wired=True,
            note="FOUND BY THE WORKER FORWARD, not by the seam survey. F12 "
            "silenced the collectives INSIDE a decoder layer; these two sit "
            "outside it, once per forward, and the survey missed them "
            "because they are not per-layer. They would have hung the boot "
            "in exactly the same way and one op earlier: tp_vocab_ratios "
            "keeps the vocab EVEN under a plain uneven-TP plan, so without "
            "this the host would hold one third of the rows and all-reduce "
            "the embedding with two ranks that hold none. Built as the same "
            "answer F12 gives: the sharding goes (enable_tp=False on the "
            "host's VocabParallelEmbedding / ParallelLMHead, so tp_size=1 "
            "there -- full vocab, no mask, no collective) and the gather "
            "goes with it (skip_all_gather in LogitsProcessor).",
            anchors=(
                ("models/qwen3_vl.py", 1367, "form_a_dense_is_unsharded"),
                ("layers/logits_processor.py", 385, "form_a_dense_is_unsharded"),
                ("distributed/utils.py", 1743, "def tp_vocab_ratios"),
            ),
        ),
        Seam(
            "F4",
            "KV only on the host: a worker builds no KV pool and owns no "
            "context tokens",
            "distributed/utils.py:1487 cp_token_context_budget "
            "(assert all(v > 0 ...)), :1488 (capacities[r] // vector[r]), "
            ":1554 (the search already skips v <= 0), "
            "model_executor/pool_configurator.py:170-180 (the ratio_r == 0 "
            "branch)",
            wired=True,
            note="Three halves, all built: the POLICY "
            "(resolve_dcp_under_host_kv -- dcp off, replication off), the "
            "ARITHMETIC (cp_token_context_budget excludes a rank that funds "
            "no unit instead of dividing by zero, and still refuses an "
            "all-zero vector), and the MIS-SCALE (pool_configurator returned "
            "1.0 silently for a rank with no pool; it now returns 0.0, the "
            "same answer it already gives a shadow rank at :152-153). The "
            "pool ALLOCATION needs nothing: cell_size == 0 already has the "
            "_KVLESS_STAGE_TOKENS path at :256 / :553-557.",
            anchors=(
                ("distributed/utils.py", 1487, "def cp_token_context_budget"),
                ("model_executor/pool_configurator.py", 173, "FORM A (F4)"),
            ),
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
            anchors=(
                ("layers/dcp/comm.py", 197, "assert counts[rank] == local_heads"),
                ("layers/dcp/comm.py", 222, "def cp_local_head_bounds"),
            ),
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
            anchors=(
                ("distributed/parallel_state.py", 618, "class GroupCoordinator"),
            ),
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
            anchors=(
                ("distributed/device_communicators/barlink_bar1.py", 3270,
                 "def put"),
                ("layers/moe/host_moe_exchange.py", 14, "host-centric"),
            ),
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
            anchors=(
                ("uneven_perf.py", 196, "_SOLO_HOST_WORKSPACE_MIB"),
            ),
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
            anchors=(
                ("layers/moe/offload_capture_gate.py", 236, "def env_graph_mode"),
                ("layers/moe/offload_capture_gate.py", 249,
                 "def resolve_offload_graph_mode"),
            ),
        ),
        # ---- found by the slice-6a symmetry probe (form_a_symmetry.py) ----
        Seam(
            "F12",
            "the HOST's own per-layer dense collectives must disappear with "
            "the sharding",
            "models/qwen4_exp.py:1100-1101 (o_proj all-reduce), :1574 "
            "(attn_tp_all_reduce), :1640 (attn_tp_all_gather); verdict from "
            "form_a_symmetry.py probe_form_a_boot",
            wired=True,
            note="BUILT. Three sites now return early under a Form A plan: "
            "LinearBase.reduce (o_proj), the attn_tp_all_reduce in the "
            "layer postprocess, and the attn_tp_all_gather in the q scatter "
            "-- form_a_dense_is_unsharded(). THE ONE THE SLICE PLAN DID NOT "
            "HAVE, and the reason slice "
            "6a must not ship alone. Those collectives exist only because "
            "the dense side is SHARDED; under Form A rank 0 owns every head, "
            "so they have no second participant. Silencing the worker's "
            "dense path while the host still issues them is not a "
            "half-built feature, it is a DEADLOCK -- the host blocks on "
            "ranks that already left the forward, on all three cards, with "
            "no log line, until the deadman fires. Measured by the probe: "
            "worker_skips_dense alone diverges at collective #0.",
            anchors=(
                ("models/qwen4_exp.py", 1144, "form_a_dense_is_unsharded"),
                ("form_a_symmetry.py", 2, "do the ranks still AGREE"),
            ),
        ),
        # ---- found by the slice-2 seam survey, not in the original nine ----
        Seam(
            "F10",
            "the W62 saturation refusal rests on the written assumption "
            "that a zero-head rank CANNOT happen",
            "weg2/launcher.py:7203-7207 (\"A rank owning zero heads cannot "
            "happen; asserting against zero heads would be a guard that can "
            "never fire\"), predicate _saturated at :7226-7232",
            wired=True,
            note="Form A made that written assumption false, so both the "
            "prose and the predicate were corrected: a weight of 0 is not a "
            "saturated axis, it is a rank that was never on the axis, and "
            "_saturated now skips it. Left as it was it fired for EVERY "
            "Form A worker (0 / total * units = 0.0 < 1.0 always) and the "
            "refusal would have rejected every Form A vector as 'axis "
            "switched off'.",
            anchors=(
                ("weg2/launcher.py", 7226, "def _saturated"),
                ("weg2/launcher.py", 7230,
                 "Form A worker: not on this axis, not saturated"),
            ),
        ),
        Seam(
            "F11",
            "a zero-width parallel Linear COMPUTES instead of being skipped",
            "layers/linear.py:2069-2092 (row-parallel "
            "input_size_per_partition can be a true 0 on the element path), "
            ":670-699 (column-parallel output_partition_sizes=[0]), "
            "distributed/utils.py:1806 assert_activation_aligned_shards "
            "(0 % 8 == 0, so the only activation guard passes a zero "
            "silently); backstop guard_zero_width_linear_shard at "
            "layers/linear.py",
            wired=True,
            note="THE DANGEROUS ONE, now closed by refusal rather than by "
            "prevention: both construction sites call "
            "guard_zero_width_linear_shard, which raises FormAZeroWidthLinear "
            "-- its OWN class, not a seam-not-wired, because the refusal is "
            "the feature and would otherwise evaporate the moment this seam "
            "was marked built. Inert on a classic boot (no role plan "
            "installed). Prevention is still F3's construction half; this "
            "only guarantees that a layer which slips through is LOUD.",
            anchors=(
                ("layers/linear.py", 2093, "guard_zero_width_linear_shard"),
                ("layers/linear.py", 677, "guard_zero_width_linear_shard"),
                ("distributed/utils.py", 1815,
                 "def assert_activation_aligned_shards"),
            ),
        ),
    )
}

#: The seams that must be wired before a Form A boot can be believed, in the
#: order the survey found them knocking. Kept as data so a report can print
#: the remaining work without re-deriving it.
UNWIRED_ORDER: Tuple[str, ...] = ("F5", "F9", "F6")


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
#: The ONE substring that distinguishes a routed expert from everything else
#: in this checkpoint's parameter names. Measured against the real weight map
#: of Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist (225.300 entries):
#: 221.184 of them are ``model.language_model.layers.<n>.mlp.experts.<e>.*``.
#: The near-misses this must NOT match are real and adjacent --
#: ``mlp.shared_expert.*`` (dense, runs for every token) and
#: ``mlp.shared_expert_gate`` -- which is why the marker carries its dots.
_ROUTED_EXPERT_MARKER = ".mlp.experts."

#: The draft's experts live under this prefix. Under Form A the MTP draft is
#: UNSHARDED on the host, so a worker keeps none of them even though their
#: names carry the expert marker. This is the one case where "is it an
#: expert" and "does a worker want it" disagree.
_DRAFT_PREFIX = "mtp."


#: The SHARED expert is dense -- it runs for every token -- so it belongs to
#: the host. Its parameters are easy to mistake for routed ones twice over:
#: by name (``mlp.shared_expert``, ``mlp.shared_expert_gate``), and because
#: the fused-shared-expert path REWRITES ``mlp.shared_expert.`` into
#: ``mlp.experts.<num_routed>.`` (models/qwen3_5.py:2448-2452), after which
#: the name is indistinguishable from a routed expert's. That is what
#: `num_routed_experts` is for below.
_SHARED_EXPERT_MARKERS = (".mlp.shared_expert.", ".mlp.shared_expert_gate")

_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")

#: The ROUTER. Slice 6a moved it onto the worker: a worker picks its own
#: experts out of the broadcast MoE input rather than being told which ones
#: to run, because the router is a replicated [hidden, num_experts] matmul
#: (0.12 GiB over 48 layers, boot fn8ah) and the alternative -- shipping
#: topk_ids/topk_weights per layer -- is a wider payload in GLOBAL expert
#: ids that each rank would have to re-filter anyway. Reasoning in
#: form_a_worker_forward's module docstring.
#:
#: The dots matter here exactly as they do for the expert marker: the near
#: miss is ``mlp.shared_expert_gate``, which is dense and host-only, and
#: which `_SHARED_EXPERT_MARKERS` rejects first.
_ROUTER_MARKER = ".mlp.gate."


def worker_keeps_parameter(
    name: str, num_routed_experts: Optional[int] = None
) -> bool:
    """Does a Form A WORKER need this checkpoint parameter? (F3)

    A worker holds the ROUTED experts of the language model AND ITS ROUTER,
    and nothing else: no attention, no GDN, no hyper-connection mixer, no
    embeddings, no lm_head, no PLE, no norms, no SHARED expert, no vision
    tower -- and no draft, because the draft is unsharded on the host.

    The router (``mlp.gate``) was on the veto list until slice 6a and is
    now kept: the worker runs it on the broadcast MoE input to pick its own
    experts. See ``_ROUTER_MARKER`` above for why that is cheaper than
    shipping the host's choice.

    Stated as a predicate over NAMES rather than over modules because that
    is where the load path can veto a tensor without constructing anything
    first (models/qwen4_exp.py:2138 `weight_name_needed`), and because a
    name predicate is checkable against the real weight map.

    `num_routed_experts`, when known, rejects the FUSED shared expert: it
    arrives as ``mlp.experts.<num_routed>.*`` and is otherwise
    indistinguishable from a routed expert. Without it that one module
    would land on a worker, which is wrong but silent -- it would compute
    for every token on a rank that only ever sees broadcast rows.
    """
    if name.startswith(_DRAFT_PREFIX) or ".mtp." in name:
        return False
    if any(marker in name for marker in _SHARED_EXPERT_MARKERS):
        return False
    if _ROUTER_MARKER in name:
        return True
    if _ROUTED_EXPERT_MARKER not in name:
        return False
    if num_routed_experts is not None:
        hit = _EXPERT_ID_RE.search(name)
        if hit is not None and int(hit.group(1)) >= num_routed_experts:
            return False
    return True


# ---------------------------------------------------------------------------
# The INSTALLED role plan for this worker process.
#
# Symmetric with distributed.utils.set_tp_partition_ratios: the scheduler
# process installs it once before any model code runs, and everything
# downstream asks here instead of threading a server_args through the model.
# ---------------------------------------------------------------------------
_INSTALLED_PLAN: Optional["RankRolePlan"] = None
_INSTALLED_RANK: int = 0


def set_form_a_role_plan(plan: Optional["RankRolePlan"], rank: int = 0) -> None:
    """Install this process's Form A role plan (or None for a classic boot)."""
    global _INSTALLED_PLAN, _INSTALLED_RANK
    if plan is not None:
        plan.role_of(rank)  # refuses a rank outside the vector, here and now
    _INSTALLED_PLAN = plan
    _INSTALLED_RANK = int(rank)


def installed_role_plan() -> Optional["RankRolePlan"]:
    return _INSTALLED_PLAN


def this_rank_is_form_a_worker() -> bool:
    """True only on a Form A worker rank. False on a classic boot, so every
    caller's default path is untouched by construction."""
    return _INSTALLED_PLAN is not None and _INSTALLED_PLAN.is_worker(
        _INSTALLED_RANK
    )


def this_rank_is_form_a_host() -> bool:
    """True only on the Form A host rank (a plan is installed and this rank
    is not a worker). False on a classic boot."""
    return _INSTALLED_PLAN is not None and not _INSTALLED_PLAN.is_worker(
        _INSTALLED_RANK
    )


def form_a_token_src_rank() -> Optional[int]:
    """The rank whose sampled tokens every other rank adopts under Form A
    (the host: it alone has lm_head + hidden states), or None on a classic
    boot."""
    return None if _INSTALLED_PLAN is None else int(_INSTALLED_PLAN.host_rank)


def guard_dense_weights(plan: RankRolePlan, rank: int) -> None:
    """Called where a rank is about to LOAD dense weights (F3).

    F3 is wired, so this is no longer "the filter does not exist yet". What
    it checks now is that the filter is actually IN FORCE on this rank: the
    loader veto and the construction skip both ask
    `this_rank_is_form_a_worker()`, which reads the INSTALLED plan, not the
    plan object passed around. A rank that is a worker by the plan but has
    no plan installed in its own process would load and build everything,
    silently, and only announce itself as an OOM on a 3080.
    """
    if plan.is_worker(rank) and not (
        _INSTALLED_PLAN is not None
        and _INSTALLED_PLAN.is_worker(_INSTALLED_RANK)
        and _INSTALLED_RANK == rank
    ):
        raise RankRoleError(
            f"rank {rank} is a Form A worker in the plan {plan.roles}, but "
            f"this process has role plan {_INSTALLED_PLAN} installed for "
            f"rank {_INSTALLED_RANK}. Every F3 site (the loader veto in "
            "weight_name_needed, the construction skip in "
            "form_a_construction.skip_on_worker) reads the INSTALLED plan, "
            "so without it this rank would load and build the whole dense "
            "side and report it as an OOM, not as a misconfiguration."
        )


def guard_kv_pool(plan: RankRolePlan, rank: int, tokens: int) -> None:
    """Called where a rank sizes its KV pool (F4)."""
    if plan.is_worker(rank) and tokens:
        require_wired(
            "F4",
            f"rank {rank} is a Form A worker but was handed {tokens} KV "
            "tokens; under Form A the host owns the whole context.",
        )


class FormAWorkerBuildsDraft(RankRoleError):
    """A Form A worker was about to build the MTP draft. Its own class, not
    a seam-not-wired: the refusal IS the feature and must not evaporate
    when a seam is marked built."""


def guard_draft_worker(plan: RankRolePlan, rank: int) -> None:
    """Called where a rank builds the MTP draft model.

    Under Form A the draft is the HOST's alone (+0.84 GiB there, the price
    of un-sharding it -- DESIGN_FORM_A_0920 §3.1), and the mechanism for
    that already exists and is not ours:
    ``--speculative-draft-placement solo`` builds the draft on the meta
    device on every other rank and skips their draft forward entirely.
    Without it the host runs a SHARDED draft whose per-layer collectives
    the workers would have to join with draft weights they do not hold --
    a second hang, in a second model, of exactly the F12 shape.

    Note the standing rule this sits against (memory
    `draft-zuordnung-27b-dflash2-nextflash-mtp`): every draft is SHARDED,
    solo is "nur ein explizites Opt-in fuer A/B, nie Default". Form A is
    that explicit opt-in -- its whole layout is "the dense side lives on
    one card" and the draft is part of the dense side -- so the flag has to
    be named in the boot line, deliberately, not inherited.
    """
    if plan.is_worker(rank):
        raise FormAWorkerBuildsDraft(
            f"rank {rank} is a Form A worker and must not build the MTP "
            "draft: under Form A the draft is unsharded on rank "
            f"{plan.host_rank}. Pass --speculative-draft-placement solo "
            "(with --speculative-draft-gpu pointing at the host's device) "
            "so the other ranks build it on the meta device and skip the "
            "draft forward."
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
    raise FormAZeroWidthLinear(
        f"rank {rank} ({plan.role_of(rank)}) built the parallel layer "
        f"{name!r} with shard width 0. F.linear with K=0 does not raise -- "
        f"it returns zeros, and the all-reduce adds them, so the only "
        f"symptom would be a wrong output many layers later. Under Form A a "
        f"worker must not CONSTRUCT this layer at all (seam F3, the "
        f"construction half); this refusal is the backstop for one that "
        f"slipped through."
    )


def form_a_dense_is_unsharded() -> bool:
    """F12: under Form A the dense side lives on ONE rank, so the per-layer
    dense collectives have no second participant.

    The q all-gather and the o_proj all-reduce exist only because the dense
    dimension is SPLIT across ranks. With `--rank-role host,worker,worker`
    and `--rank-tp-ratio 1,0,0` the host owns every head and every dense
    weight; there is nothing to gather from anyone and nothing to sum with
    anyone. Issuing them anyway means the host blocks on ranks that are not
    coming -- a hang, not an error, which is why this predicate exists and
    why the boot gate (form_a_boot_gate) checks that all ranks agree about
    it before the first forward.

    False on every classic boot, so the default path is untouched.
    """
    return _INSTALLED_PLAN is not None


def guard_zero_width_linear_shard(name: str, width: int) -> None:
    """F11 at a construction site that has no plan object to hand.

    The shape every hot call site needs: no arguments beyond what it already
    has, and a fast exit on the classic path. `this_rank_is_form_a_worker()`
    is False whenever no role plan is installed, so on every boot that is not
    Form A this costs one attribute read and returns.
    """
    if width > 0 or not this_rank_is_form_a_worker():
        return
    plan = _INSTALLED_PLAN
    assert plan is not None  # implied by this_rank_is_form_a_worker()
    guard_zero_width_linear(plan, _INSTALLED_RANK, name, width)


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
