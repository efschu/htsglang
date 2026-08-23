"""The rejected register, as data (task #270).

Every row here is a combination that was **tried on this project and settled**.
The register itself lives in the operator's notes; this module is its
machine-readable half, so the guided wizard cannot propose something the
project already measured and put down.

Two levels, and the difference matters:

``BLOCKED``
    The combination is wrong or impossible, not merely unprofitable. It is
    never offered, and a configuration that names it is refused with the
    verdict text. Silent output corruption, a missing instruction on the
    target architecture, or a guard already in the engine.

``NOT_DEFAULT``
    The combination works and produces correct output; it was measured and
    lost. It is offered, but never proposed on its own, and it always carries
    the counter-number that decided it. A user who asks for it explicitly gets
    it with the measurement attached.

Each row keeps its ``evidence`` string verbatim from the register, because the
point of a rejection is the number that produced it. A row with no evidence
would be an opinion, and an opinion cannot bind a later attempt.

The register's own rule is that verdicts are re-checked before every feature
order — several rows are days old and some may be reopened by different
hardware. So a row also names ``scope``: ``"rig"`` means the verdict is a
statement about *this* hardware and does not generalise (see the standing
"the rig is a lower bound" rule), ``"general"`` means the finding is a
property of the code or of an architecture and travels.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BLOCKED",
    "NOT_DEFAULT",
    "LEVELS",
    "RejectedEntry",
    "REGISTER",
    "by_key",
    "blocked_keys",
    "check_combination",
    "register_json",
]

#: Never offered. Correctness, or a hard architectural wall.
BLOCKED = "blocked"
#: Offered on request only, always with the counter-number.
NOT_DEFAULT = "not-default"

LEVELS: Tuple[str, ...] = (BLOCKED, NOT_DEFAULT)


@dataclasses.dataclass(frozen=True)
class RejectedEntry:
    """One settled verdict.

    ``tags`` is what the wizard matches against: a configuration whose tag set
    is a superset of an entry's tags has hit that entry. Tags are the coarse
    vocabulary of the family matrix (``"tree-spec"``, ``"uneven-dcp"``,
    ``"gguf"``, ``"moe-offload"``, ...), deliberately small — an entry that
    needs a subtler condition than a tag set states it in ``note`` and is not
    matched automatically.
    """

    key: str
    what: str
    verdict: str
    level: str
    evidence: str
    tags: Tuple[str, ...] = ()
    scope: str = "general"
    note: Optional[str] = None
    #: The three short lines a reader actually needs, and they are three
    #: rather than one because a verdict paragraph answers none of them
    #: directly. What would this have BOUGHT; what does it COST; and WHY does
    #: the cost arise -- the mechanism, not the measurement. One sentence
    #: each: a rejection nobody finishes reading is a rejection nobody learns
    #: from, and this register's whole purpose is that the next person does
    #: not repeat the work.
    gain: str = ""
    cost: str = ""
    why: str = ""
    #: For a NOT_DEFAULT row: the exact flag that turns it on. The register
    #: says a thing is available on request, so the request has to be one
    #: click -- an entry that says "available" and leaves the reader to work
    #: out how is not offering anything. Empty for BLOCKED rows: there is
    #: nothing to hand over.
    unlock: str = ""

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "what": self.what,
            "verdict": self.verdict,
            "level": self.level,
            "evidence": self.evidence,
            "tags": list(self.tags),
            "scope": self.scope,
            "note": self.note,
            "gain": self.gain,
            "cost": self.cost,
            "why": self.why,
            "unlock": self.unlock,
            # A blocked row has no switch by definition; a not-default one is
            # only genuinely "available on request" when the request exists.
            "unlockable": bool(self.unlock) and self.level == NOT_DEFAULT,
        }


#: The register. Order follows the source list so the two can be diffed.
REGISTER: Tuple[RejectedEntry, ...] = (
    RejectedEntry(
        key="tree_spec_uneven_dcp",
        what="tree speculation (--speculative-eagle-topk > 1) under uneven DCP",
        verdict=(
            "BLOCKED: silently non-greedy (tree-mask verify logits) AND "
            "performance-negative, 80 -> 68 tok/s"
        ),
        gain="a wider draft tree, so more tokens can be accepted per verify step",
        cost="wrong output, and it is slower as well: 80 -> 68 tok/s",
        why=(
            "the tree mask is not applied to the verify logits under uneven "
            "DCP, so the accept decision is taken on the wrong distribution. "
            "Nothing crashes -- the output is simply no longer the greedy one."
        ),
        level=BLOCKED,
        evidence="guard restored in 46b55054f; the engine fail-fasts on it",
        tags=("tree-spec", "uneven-dcp"),
        scope="general",
    ),
    RejectedEntry(
        key="tree_spec_tp1",
        what="tree speculation (topk > 1) on TP=1 / non-DCP",
        verdict=(
            "not a default: -36.8 % throughput for +7.7 % acceptance "
            "(9x the noise floor). Correctness is NOT refuted here — it is "
            "self-deterministic and shows no #76 pattern — so it stays "
            "available, it is just never proposed"
        ),
        gain="+7.7 % acceptance: more of each draft survives verification",
        cost="-36.8 % throughput, nine times the noise floor",
        why=(
            "every branch is drafted AND verified whether or not it is taken. "
            "At this acceptance rate the extra verify work costs more than the "
            "extra accepted tokens save. Correctness is not in question here."
        ),
        unlock="--speculative-eagle-topk 2",
        level=NOT_DEFAULT,
        evidence="#139 A/B 2026-07-28 (189945d618)",
        tags=("tree-spec", "solo-tp"),
        scope="rig",
    ),
    RejectedEntry(
        key="vocab_ratio_7_3_3",
        what="--rank-vocab-ratio 7,3,3 as a default",
        verdict=(
            "not a default: +3.9-6.0 % measured, but inside the scatter of "
            "the same measurement"
        ),
        gain="+3.9-6.0 % in the measurement that found it",
        cost="nothing measurable -- and no evidence the gain is real",
        why=(
            "the difference is inside the scatter of the same measurement "
            "repeated, so it cannot be told apart from noise. Not refused; just "
            "never proposed on evidence this thin."
        ),
        unlock="--rank-vocab-ratio 7,3,3",
        level=NOT_DEFAULT,
        evidence="task #203, longer run",
        tags=("vocab-ratio",),
        scope="rig",
    ),
    RejectedEntry(
        key="spec_k4_mixed",
        what="speculative k=4 on mixed cards",
        verdict="not a default: lost the #103 matrix",
        gain="a longer draft, so more tokens per draft pass",
        cost="it lost the comparison against shorter drafts on mixed cards",
        why=(
            "the slowest rank sets the verify clock, and a longer draft "
            "lengthens exactly that step. On equal cards the trade looks "
            "different."
        ),
        unlock="--speculative-num-steps 4",
        level=NOT_DEFAULT,
        evidence="task #103 matrix",
        tags=("spec-k4", "mixed-cards"),
        scope="rig",
    ),
    RejectedEntry(
        key="mlp_concentration_611",
        what="MLP concentration 6,1,1 as a prefill lever at the 27B-FP8 point",
        verdict=(
            "measured NET NEGATIVE: prefill +8.2 %, but decode -13.7 % (well "
            "over the noise floor) and KV -47.9 %; the collective floor "
            "(69-75 % of the window) is untouched — the floor is the lever, "
            "not the split"
        ),
        gain="+8.2 % prefill: dense MLP work moves onto the strong card",
        cost="-13.7 % decode and -47.9 % KV",
        why=(
            "decode is gated by the slowest rank, so loading the strong card "
            "makes the weak ranks the clock, and the units they gave up were "
            "holding KV. The verdict is about ONE vector serving BOTH phases; "
            "per phase the prefill half carries (#354: FP8 16,1,1 is +22.6 % "
            "prefill, -20.0 % decode, -79 % context), which is what "
            "--rank-perf-tune phase-prefill asks for (#357)."
        ),
        unlock="--rank-mlp-ratio 6,1,1",
        level=NOT_DEFAULT,
        evidence=(
            "#264 A/B 2026-07-28, INTEGRATION_R3_VALIDATION (539154288d); "
            "per-phase re-measurement #354 2026-07-31 (4 boots x 16 points)"
        ),
        tags=("mlp-concentration",),
        scope="rig",
    ),
    RejectedEntry(
        key="reserve_2200_3080",
        what="--rank-auto-reserve-mib 2200 on the 3080s (27B-FP8, auto-performance)",
        verdict=(
            "BLOCKED: tips into a GDN prefill-scratch OOM on the first real "
            "chunk. 2700 carries. Warmup passes and fakes success"
        ),
        gain="about 500 MiB more KV pool on each 3080",
        cost="an out-of-memory abort on the first real prefill chunk",
        why=(
            "the GDN prefill scratch is sized per chunk and is larger than "
            "anything warmup allocates, so the boot looks healthy right up to "
            "the first long prompt. 2700 leaves enough room."
        ),
        level=BLOCKED,
        evidence="task #250, base and branch identical",
        tags=("reserve-2200",),
        scope="rig",
    ),
    RejectedEntry(
        key="triton_fp8_sm86",
        what="Triton fp8 tuning on sm86 (RTX 3080)",
        verdict=(
            "IMPOSSIBLE: Triton has no fp8e4nv below sm89 ('type fp8e4nv not "
            "supported'); on sm86 the runtime takes a different kernel path "
            "entirely"
        ),
        gain="the Triton fp8 kernel path on the 3080s",
        cost="it does not compile at all",
        why=(
            "Triton has no fp8e4nv type below sm89. On sm86 the runtime takes a "
            "different kernel path entirely, so there is nothing here to tune."
        ),
        level=BLOCKED,
        evidence="idle tuner log 2026-07-28",
        tags=("triton-fp8", "sm86"),
        scope="general",
    ),
    RejectedEntry(
        key="gguf_on_sm75",
        what="a GGUF checkpoint on a Turing (sm75) rank",
        verdict=(
            "BLOCKED: sgl-kernel is cubin-only with a gencode floor of sm_80 "
            "and no PTX, so it holds no code a 2080 Ti can execute. The rank "
            "loads weights, then dies mid-forward on a missing symbol — the "
            "promised loud failure at GGUFConfig is not wired (open bug #269)"
        ),
        gain="a GGUF checkpoint on a Turing rank",
        cost="the rank loads its weights and then dies mid-forward",
        why=(
            "sgl-kernel ships cubins with a gencode floor of sm_80 and no PTX, "
            "so it holds no code a 2080 Ti can execute. The loud failure this "
            "should produce at load time is not wired up yet (open bug #269)."
        ),
        level=BLOCKED,
        evidence="#212 satellite bring-up; gguf.py:40-62 comment vs the unread flag",
        tags=("gguf", "sm75"),
        scope="general",
    ),
    RejectedEntry(
        key="gguf_moe_expert_offload",
        what="GGUF MoE checkpoint with SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0",
        verdict=(
            "BLOCKED: the expert-offload installer has no quantization guard "
            "and GGUF has no presplit half at all "
            "(GGUFUninitializedParameter materialises only in the "
            "postprocess). The engine will not refuse it; it fails late and "
            "confusingly. Fail-fast is task #268"
        ),
        gain="host-RAM expert offload for a GGUF MoE checkpoint",
        cost="a late and confusing failure rather than a refusal",
        why=(
            "the offload installer slices expert stacks with no quantisation "
            "guard, and GGUF has no load-time half at all -- its parameters "
            "only materialise in the postprocess. The fail-fast guard is task "
            "#268."
        ),
        level=BLOCKED,
        evidence="#123 finding; expert_offload.py has no GGUF/MoeWNA16 path",
        tags=("gguf", "moe-offload"),
        scope="general",
    ),
    RejectedEntry(
        key="spill_with_pd",
        what="session KV spill together with PD disaggregation",
        verdict=(
            "BLOCKED by the engine: --enable-kv-session-offload refuses a "
            "non-null --disaggregation-mode. The two re-interpret KV slot "
            "identity differently"
        ),
        gain="parked sessions and a separate prefill arm at the same time",
        cost="refused at argument parse",
        why=(
            "the two assign KV slot identity differently: a slot a spilled "
            "session expects to reclaim means something else to a handover."
        ),
        level=BLOCKED,
        evidence="server_args.py:4992, hard reject at arg parse",
        tags=("spill", "pd"),
        scope="general",
    ),
    RejectedEntry(
        key="spill_with_pp_dp",
        what="session KV spill with --pp-size > 1 or --dp-size > 1",
        verdict=(
            "BLOCKED by the engine: spill S1 supports single-node pure TP/DCP only"
        ),
        gain="session spill on a pipeline- or data-parallel server",
        cost="refused at argument parse",
        why=(
            "spill addresses one node's pure TP/DCP pool. There is no owner "
            "rule for a slot whose tokens live on another stage."
        ),
        level=BLOCKED,
        evidence="server_args.py:5015, hard reject at arg parse",
        tags=("spill", "pipeline-parallel"),
        scope="general",
    ),
    RejectedEntry(
        key="pp_with_spec",
        what="pipeline parallelism together with speculative decoding",
        verdict=(
            "BLOCKED for PLAIN pipeline parallelism: under pp_size > 1 a hard "
            "assert refuses a speculative algorithm, and the boot dies at arg "
            "parse. EXEMPT under --enable-phase-flip, which the assert now "
            "names explicitly: a phase-flip instance runs PP for prefill with "
            "no draft worker built at all and arms speculation on the TP "
            "decode stack at cutover, so the combination is enforced by "
            "construction rather than refused. A plain-PP number is therefore "
            "still a no-spec number and must not be put next to a NEXTN one; "
            "a phase-flip number is NOT covered by this row"
        ),
        gain="speculative decoding across pipeline stages",
        cost="the boot dies at argument parse, unless --enable-phase-flip",
        why=(
            "pp_size > 1 asserts no speculative algorithm unless "
            "enable_phase_flip -- still a hard assert, so PLAIN pipeline "
            "figures stay no-spec figures. The sibling half went the other "
            "way: disable_overlap_schedule is set by a post-process pass "
            "before its assert can fire, making that half an auto-disable."
        ),
        level=BLOCKED,
        evidence=(
            "server_args.py:19042 (if pp_size > 1) and server_args.py:19057 "
            "(spec assert, "
            "'or self.enable_phase_flip'); the overlap half is auto-disabled "
            "first at arg_groups/overrides.py:2163 "
            "(_pipeline_parallel_overlap_disable). Re-verified #704b; the "
            "earlier :16240-16245 cite had drifted and its verdict text "
            "predated both the phase-flip exemption and the auto-disable"
        ),
        tags=("pipeline-parallel", "speculation"),
        scope="general",
    ),
    RejectedEntry(
        key="cuda_ipc_weight_import_pd",
        what="CUDA-IPC weight import between PD processes",
        verdict=(
            "abandoned: no purchase in the loader (the postprocess rewrites "
            "the tensors), and the process boundary loses its purpose once "
            "weights are shared"
        ),
        gain="one copy of the weights serving both PD processes",
        cost="no saving in practice, and the design loses its point",
        why=(
            "the IMPORTING process runs its own postprocess, which rewrites "
            "the tensors after import, so the shared pages stop being shared; "
            "and once the weights are shared the process boundary is no "
            "longer buying isolation."
        ),
        level=BLOCKED,
        evidence="DESIGN_107, the fork in the road",
        tags=("pd", "cuda-ipc-weights"),
        scope="general",
        note=(
            "REPRICED #444d against upstream PR #33279 (weight-daemon "
            "transport plug-in, adds a vmm_fd/SCM_RIGHTS backend next to "
            "torch_ipc). The entry STANDS, and the re-read sharpened why. The "
            "daemon loads through the full pipeline -- disk, TP shard, "
            "quantize -- and exports only afterwards, so its consumers never "
            "rewrite what they mapped. This design has no such exporter: both "
            "PD processes are ordinary engines, each runs "
            "process_weights_after_loading itself (model_loader/loader.py:928 "
            "and four further sites), and that is precisely the rewrite the "
            "verdict names. So the daemon does not reopen this key -- it is a "
            "different construction that happens not to have this key's "
            "defect, and the register should not be read as 'sharing weights "
            "across processes is dead'. The daemon shape belongs to #305 as a "
            "concrete WARM_GPU rung (registry/ledger.py TenantState.WARM_GPU, "
            "registry/rungs.py: post-quantized weights resident while no "
            "engine owns them), where the open question is hot-switch resume "
            "time. It does not help #329: its export is TP-sharded, so an "
            "elastic membership change invalidates exactly what it holds. "
            "Cost of pursuing the daemon shape here is a PORT, not a flip -- "
            "weight_cache/ does not exist in this tree at all. No build in "
            "this round; taking it up is a planner decision under a new key."
        ),
    ),
    RejectedEntry(
        key="intra_rig_collective_overlap",
        what="intra-rig collective overlap (a second NCCL lane)",
        verdict=(
            "measured, NOT built: 15.8-23.6 % of the budget is structurally "
            "non-overlappable and the single overlappable pair is 1.2 %, "
            "below the detection floor"
        ),
        gain="collective time hidden behind compute",
        cost="the build cost, for nothing measurable",
        why=(
            "15.8-23.6 % of the budget is structurally non-overlappable, and "
            "the one pair that could overlap is 1.2 % -- below the detection "
            "floor of the measurement that would have to prove it."
        ),
        level=BLOCKED,
        evidence="feat/intrarig-collective-overlap (documentation branch)",
        tags=("collective-overlap",),
        scope="rig",
    ),
    RejectedEntry(
        key="dcp_overlap_fusion",
        what="DCP overlap / fusion in the CUDA-graph regime",
        verdict=(
            "performance-neutral: collectives that look expensive under eager "
            "are not the bottleneck inside a graph (a ~15 % phantom win, "
            "caught before shipping)"
        ),
        gain="about 15 %, when measured under eager execution",
        cost="nothing at all under CUDA graphs",
        why=(
            "collectives that look expensive in eager mode are not the "
            "bottleneck inside a captured graph. The win was an artefact of the "
            "eager harness and was caught before it shipped."
        ),
        level=BLOCKED,
        evidence="full-perf-testing rule, extension 2026-07-20",
        tags=("dcp-overlap",),
        scope="general",
    ),
    RejectedEntry(
        key="barlink_ring_bidir",
        what="splitting the barlink ring bidirectionally (two workers per direction)",
        verdict=(
            "REGRESSION: +17 % at 80 KiB. A ring step is two requests in "
            "lockstep; there is no concurrency to expose. "
            "SGLANG_BARLINK_UCX_RING_BIDIR stays 0"
        ),
        gain="both directions of the ring in flight at once",
        cost="+17 % time at 80 KiB -- a regression",
        why=(
            "a ring step is two requests in lockstep, so there is no "
            "concurrency to expose; the second worker only adds "
            "synchronisation."
        ),
        level=BLOCKED,
        evidence="#266 A/B 2026-07-28, d6d4231e5a",
        tags=("barlink-ring-bidir",),
        scope="general",
    ),
    RejectedEntry(
        key="path_bundling",
        what="path bundling (one transfer striped over both lines)",
        verdict=(
            "not built before somebody with full lanes shows a need; only "
            "the CHOICE of line per message class is released (#240)"
        ),
        gain="one transfer striped across both lines",
        cost="unbuilt complexity",
        why=(
            "nobody with full lanes has shown a need. The part that does pay -- "
            "choosing which line carries which message class -- is released "
            "separately (#240)."
        ),
        level=BLOCKED,
        evidence="task #240 pre-check",
        tags=("path-bundling",),
        scope="rig",
    ),
    RejectedEntry(
        key="int8_mamba_sizing",
        what="int8 Mamba checkpoint sizing",
        verdict=(
            "dropped: the benefit does not justify the work, and it collides "
            "logically with HiCache's quantise-exactly-once invariant"
        ),
        gain="a smaller Mamba state pool",
        cost="the work, and a cache invariant",
        why=(
            "the saving is small, and it collides with quantise-exactly-once: a "
            "state quantised on its way into the hierarchical cache would be "
            "quantised twice."
        ),
        level=BLOCKED,
        evidence="task #193",
        tags=("int8-mamba",),
        scope="general",
    ),
    RejectedEntry(
        key="gdn_fla_autotune",
        what="re-enabling the GDN/fla autotune",
        verdict=(
            "DO NOT BUILD: the ceiling is measured at 0.00 % of decode and at "
            "most 0.14 % of prefill wall (~10x under the noise floor); the "
            "decode GDN path lies entirely in kernels without the decorator"
        ),
        gain="autotuned GDN kernels",
        cost="nothing -- the ceiling itself is zero",
        why=(
            "the measured ceiling is 0.00 % of decode and at most 0.14 % of "
            "prefill, ten times under the noise floor, because the decode GDN "
            "path lies entirely in kernels that carry no autotune decorator."
        ),
        level=BLOCKED,
        evidence="INTEGRATION_R3_VALIDATION.md, #200 section (5b42c4d859)",
        tags=("gdn-autotune",),
        scope="general",
    ),
    RejectedEntry(
        key="ucc_transport",
        what="moving the ucx layer to UCC",
        verdict=(
            "abandoned: c10d-UCC is not compiled into torch 2.11 and would "
            "need USE_UCC source builds on both hosts, for code hygiene at "
            "best. The floor is not in the transport"
        ),
        gain="UCC in place of the ucx layer",
        cost="source builds of torch on both hosts",
        why=(
            "c10d-UCC is not compiled into torch 2.11, so this is a build "
            "project for code hygiene -- and the floor is not in the transport "
            "to begin with."
        ),
        level=BLOCKED,
        evidence="#247 check + #266 finding",
        tags=("ucc",),
        scope="general",
    ),
    RejectedEntry(
        key="crossrig_tp_push",
        what="pushing cross-rig TP further (a third attempt)",
        verdict=(
            "EXHAUSTED on this hardware: ~89 us decode / ~202 us verify "
            "all_reduce at world 4. Without GDR on GeForce, host staging is "
            "the wall (~86 % of the cost). The one untested lever left is a "
            "dedicated progress thread pinned to a core"
        ),
        gain="more cards in one tensor-parallel group, across two machines",
        cost="about 89 us decode and 202 us verify per all_reduce at world 4",
        why=(
            "without GDR on GeForce cards every collective stages through host "
            "memory, which is around 86 % of the cost. It works and it is "
            "available; a third round of optimisation is what has been ruled "
            "out. The one untested lever left is a dedicated progress thread "
            "pinned to a core."
        ),
        level=NOT_DEFAULT,
        evidence="the #244 + #263 + #266 chain",
        tags=("crossrig-tp-push",),
        scope="rig",
    ),
    RejectedEntry(
        key="c4_indexer_head_fold",
        what="folding the C4 indexer's head axis into one effective query",
        verdict=(
            "WRONG OPERATOR: the fold needs the per-head product to enter the "
            "head sum linearly, and the C4 indexer applies a per-head ReLU "
            "between the two. Measured 0.94 relative divergence and 0.41-0.56 "
            "top-k overlap against the production torch path; the same fold "
            "reproduces a relu-free operator to 3.6e-07"
        ),
        gain=(
            "the quadratic term would lose the index_n_heads=64 factor, "
            "reported 37-99x on 8xA800 at KV 16K-64K"
        ),
        cost=(
            "silently different page selection on every long-context prefill, "
            "with no error bound"
        ),
        why=(
            "logits[i,j] = sum_h w[i,h] * relu(q[i,h,:] . k[j,:]) * "
            "k_scale[j]. ReLU is not linear, so the head sum cannot cross it "
            "and the head axis is irreducible -- MQA is necessary for the "
            "fold and is genuinely present here, but it is not sufficient. "
            "Reopen only if the GPU arm in NOTE_440 shows the ReLU is inert "
            "on the real checkpoint, and then only as a labelled "
            "approximation, never as an identity."
        ),
        level=BLOCKED,
        evidence=(
            "docs/dev/NOTE_440_c4_indexer_head_fold.md; "
            "test_dsv4_indexer_head_fold_440.py (18 tests); "
            "sgl-project/sglang#33246 comment 5159510149 and PR #33271, "
            "whose folded Triton kernel carries no ReLU while the per-head "
            "fallback in the same file does"
        ),
        tags=("c4-indexer-head-fold",),
        scope="general",
    ),
    RejectedEntry(
        key="slot_keyed_verify_intermediates",
        what=(
            "re-keying the speculative intermediate caches (intermediate_ssm, "
            "intermediate_conv_window) by mamba slot instead of batch position"
        ),
        verdict=(
            "DOES NOT FIT, AND FIXES NOTHING OPEN: the caches are sized "
            "mamba_spec_state_size = max_num_reqs while the slot space is "
            "max_mamba_cache_size, and max_num_reqs = max_mamba_cache_size // "
            "mamba_ratio -- slot ids index those caches out of bounds by the "
            "whole mamba ratio"
        ),
        gain=(
            "self-owning rows: two concurrently verifying runners could share "
            "one pool without colliding on row 0"
        ),
        cost=(
            "the two largest speculative scratch buffers grow by the mamba "
            "ratio, for a collision no configuration can reach"
        ),
        why=(
            "The collision is real as a mechanism (measured in "
            "test_verify_intermediate_row_ownership_450.py) but its "
            "precondition is not: the #274 lane is its own ModelRunner with "
            "its own req_to_token_pool and attention backend, and inside one "
            "runner verifies are sequential on one stream. A per-lane offset "
            "partition pays the same cost in miniature. Reopen if a second "
            "verifier is ever handed an existing pool."
        ),
        level=BLOCKED,
        evidence=(
            "test/registered/unit/spec/test_verify_intermediate_row_ownership_450.py "
            "(12 tests, 6 subtests: collision measured on a shared cache, clean "
            "on disjoint ones, both instruments shown to report the wrong "
            "configuration); model_runner_kv_cache_mixin.py "
            "'if self.req_to_token_pool is None'; dual_group_lane.py "
            "_build_lane_under_scope and _lane_server_args_view"
        ),
        tags=("verify-intermediate-rows", "dual-group-lane"),
        scope="general",
    ),
    RejectedEntry(
        key="moe_link_calibrated_coefficients",
        what=(
            "calibrating the link-proportional cold-expert compute placement "
            "with per-rank cold-traffic coefficients measured on a prior boot "
            "(--rank-moe-ratio link-calibrated)"
        ),
        verdict=(
            "FALSIFIED MODEL on exactly TWO load-bearing legs, and the "
            "transfer term is NOT one of them. (1) END TO END: -0.94 % "
            "against the baseline, inside the same-window A-vs-A floor of "
            "4.09 % spread, while the uncalibrated arm is well outside it at "
            "-7.67 %. (2) MECHANISM: see below. Work-matched, it reads "
            "1.4573x on the transfer term against the uncalibrated 1.4253x "
            "and slightly WINS that term; the earlier '1.439x against 1.496x' "
            "divided two counters sampled at different work points and is "
            "withdrawn (#482/#523)"
        ),
        gain=(
            "removing the first-order traffic model's residual; its predicted "
            "1.498x is withdrawn with the pre-teardown revision"
        ),
        cost=(
            "a prior boot of the same recipe, spent on overloading whichever "
            "rank that boot left the smallest expert range"
        ),
        why=(
            "the coefficient treats the cache hit rate as a property of the "
            "RANK; it is a property of the owned RANGE SIZE, because a smaller "
            "range fits the cache better. Measured in the window that produced "
            "the coefficients: tp1 0.8450 -> 0.9050 as its range shrank "
            "72 -> 58, tp2 0.8474 -> 0.7814 as its grew 72 -> 89. The solve "
            "read tp2's 0.9586 as spare capacity and made it the new clock."
        ),
        level=NOT_DEFAULT,
        evidence=(
            "/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md "
            "sections 5 and 6 (three boots, 900 tokens x 3 runs x 1 warmup, "
            "DeepSeek-V4-Flash UD-IQ3_XXS, TP=3 on 1x RTX 5090 + 2x RTX 3080); "
            "the falsifier was named in advance in "
            "scripts/dev/394_s2_proof/ARM3_COMPUTE.md, readout item 4"
        ),
        tags=("moe-offload", "moe-link-calibrated"),
        scope="general",
        note=(
            "The MECHANISM is not rejected -- --rank-moe-ratio link, the "
            "uncalibrated solve, is the confirmed one and is what the flag "
            "resolves to. Only the coefficient model is. The symbol stays in "
            "tree so a range-size-aware replacement can be measured against "
            "these three arms rather than against a memory of them."
        ),
        unlock=(
            "--rank-moe-ratio link-calibrated plus one coefficient per rank in "
            "SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS (both required; plain 'link' "
            "refuses while that variable is set)"
        ),
    ),
    RejectedEntry(
        key="draft_kv_dcp_below_kv_threshold",
        what=(
            "putting the DRAFT KV cache on the token-sharded layout "
            "(--draft-kv-layout dcp) at tp <= num_kv_heads, to extend the "
            "#492 replication+token axis from the target to the draft"
        ),
        verdict=(
            "NOT A DEFAULT below the threshold: the applicability rule is "
            "two-sided and both sides are measured. Above tp > kv_heads 'dcp' "
            "is right and 'replicated' is the DEGRADED layout (accept 1.05, "
            "61 verify rounds for 64 tokens); at or below it 'replicated' is "
            "right and 'dcp' is VRAM-neutral and costs 10-16 % acceptance."
        ),
        gain=(
            "the target's token-vector attention lever would also apply to "
            "the draft's attention barrier instead of only the target's"
        ),
        cost="10-16 % acceptance, for no VRAM saving at this threshold",
        why=(
            "below the threshold every rank can hold a real kv-head shard, so "
            "the draft has a head split to run on; token-sharding it splits "
            "the draft's own context instead and buys nothing, while the "
            "verify round still pays the full speculative cost."
        ),
        level=NOT_DEFAULT,
        evidence=(
            "docs/dev/TASK_108_DRAFT_KV_DCP.md, 'TP > num_kv_heads window' "
            "Q2 table and the two-sided rule below it "
            "(/spinning/gpu-battery-results/2026-08-02_108-tpgtkv/)"
        ),
        tags=("uneven-dcp", "draft-kv-layout", "spec"),
        scope="general",
        note=(
            "This is the SPEC cross-charge of the #492 replication axis. The "
            "axis itself is not rejected -- the target's KV is already "
            "replicated-heads + token-sharded under uneven DCP and costs "
            "nothing extra there. Only dragging the draft onto it below the "
            "threshold is."
        ),
        unlock="--draft-kv-layout dcp",
    ),
    RejectedEntry(
        key="tts_talker_tensor_parallel",
        what=(
            "splitting the Qwen3-TTS talker across the three cards with "
            "tensor parallelism (--rank-tp-ratio over --rank-gpu-id 0,1,2)"
        ),
        verdict=(
            "NOT_DEFAULT: representable but LOSES to a single 5090 by 2-5x. "
            "Predicted RTF 0.043 (at an optimistic 10 us per collective) to "
            "0.117 (at 40 us) against 0.022 solo, and worse than a solo 3080 "
            "(0.052) from ~20 us upward. Not a geometry refusal -- the uneven "
            "plan carries the shard fine (see gain)"
        ),
        gain=(
            "a third of the per-rank weight read: 37.0 GiB per audio-second "
            "split as q [8,4,4], kv [4,2,2], MLP [1392,840,840]"
        ),
        cost=(
            "2472 extra all-reduces per audio-second, each a 2 KiB round trip "
            "on a rig with no P2P and no NVLink"
        ),
        why=(
            "the talker is LATENCY-bound, not bandwidth-bound: 192 forward "
            "passes per audio-second at batch 1 over a 0.6 B model leave a "
            "per-step bandwidth term of 0.09-0.50 ms on the 5090 against a "
            "measured 6.4 ms step. TP divides the negligible term and "
            "multiplies the dominant one. The three-card win here is one "
            "INSTANCE per card serving independent sessions."
        ),
        level=NOT_DEFAULT,
        evidence=(
            "docs/dev/ANALYSE_488_talker_lane_layout.md SS2-5: weight census "
            "of the 478-tensor checkpoint, the sequential frame loop at "
            "qwen_tts modeling_qwen3_tts.py:1671/:1684, the head/MLP vectors "
            "executed against distributed/utils.py and pinned in "
            "test/registered/unit/models/test_qwen3_tts_talker_lane_488.py, "
            "and the no-P2P rig fact in FEATURE_CATALOG SS7. FIXED-COST "
            "CALCULATION, not a GPU measurement -- the falsifying window is "
            "specified in ANALYSE_488 SS6 and has not run"
        ),
        tags=("tts-talker", "uneven-tp"),
        scope="rig",
        note=(
            "The arithmetic is rig-scoped because the collective term is: on "
            "a fabric with sub-microsecond small-message latency the balance "
            "moves. The 192-steps-per-audio-second shape is a property of the "
            "architecture and travels; the verdict is the two together."
        ),
        unlock="--rank-tp-ratio 5,3,3 --rank-gpu-id <5090>,<3080a>,<3080b>",
    ),
    RejectedEntry(
        key="cpu_expert_lane_per_event_dequant",
        what=(
            "computing cold MoE experts on the LOCAL CPU with a per-event "
            "int4 -> fp32 dequant of the expert's weights"
        ),
        verdict=(
            "BLOCKED: the dequant costs 6.177 ms per expert against 0.36 ms "
            "of headroom (17x), and ~10x the entire H2D fetch it was meant to "
            "replace -- the lane would be roughly ten times SLOWER than just "
            "streaming the weights"
        ),
        gain=(
            "activations cross the link, not weights: ~800:1 less traffic "
            "per expert event in decode, worth ~2.3x on compute"
        ),
        cost="6.177 ms/expert of dequant against 0.36 ms of headroom (17x)",
        why=(
            "half precision on this CPU is a 400x cliff (~1.0 GFLOP/s bf16 "
            "and fp16 vs 23-432 fp32), so the lane must compute in fp32; fp32 "
            "masters need 129 GB against 98 GB with no swap, so they cannot be "
            "held; which forces the dequant per event, where it costs more "
            "than the transfer. The three only conflict together."
        ),
        level=BLOCKED,
        evidence=(
            "docs/dev/ANALYSE_CPU_EXPERT_LANE.md SS2 (CPU GEMM on the real "
            "expert shapes) and SS6a (dequant on REAL tensors from the "
            "delivered Qwen3.5-35B-A3B-GPTQ-Int4 shards, layer 23 expert 0, "
            "all three projections, cold source, 16 threads), measured "
            "2026-08-05. Hermetic CPU measurements, no card. Robust to kernel "
            "quality: even a 5x faster dequant leaves it 3.4x over budget"
        ),
        tags=("moe-offload", "cpu-lane"),
        scope="rig",
        note=(
            "TWO ROUTES REMAIN, and neither is this one. (a) INTEGER CPU "
            "KERNELS (llama.cpp-style K-quant GEMM) have no dequant step at "
            "all; that is the natural route for the DSV4F/K3 vehicle, whose "
            "transfer wall makes the upside largest, and it stays OPEN -- "
            "nothing here argues for or against it, it is simply a much "
            "larger build. (b) a PRE-DEQUANTISED PARTIAL fp32 TIER: an fp32 "
            "expert is 12.6 MB, so ~40 GB of spare RAM holds ~31 % of the "
            "model's experts with no dequant in the path. That is a DIFFERENT "
            "feature (a second host tier plus an admission policy, #407 being "
            "the right home) and it is gated on a measurement nobody has: how "
            "skewed cold-expert routing actually is. That measurement is "
            "task #390 (expert-heat statistics) -- answerable from existing "
            "instrumentation without a line of CPU-lane code, and it is the "
            "FIRST question on any revisit, not a CPU-GEMM question at all. "
            "The rig scope is because both the 98 GB RAM ceiling and the "
            "half-precision cliff are properties of this box."
        ),
    ),
)


_BY_KEY: Dict[str, RejectedEntry] = {e.key: e for e in REGISTER}


def by_key(key: str) -> Optional[RejectedEntry]:
    """One entry, or ``None``."""
    return _BY_KEY.get(key)


def blocked_keys() -> Tuple[str, ...]:
    """The keys that are never offered."""
    return tuple(e.key for e in REGISTER if e.level == BLOCKED)


def check_combination(tags: Sequence[str]) -> List[RejectedEntry]:
    """Every register entry whose tags are all present in ``tags``.

    An entry with no tags never matches automatically: it is documentation
    for a reader, not a predicate. An entry matches only when the
    configuration carries *all* of its tags, so ``("tree-spec",)`` alone does
    not fire the tree-spec-under-uneven-DCP row — the DCP tag has to be there
    too. Both levels come back; the caller decides whether a
    :data:`NOT_DEFAULT` row blocks a proposal (it does) or blocks the
    configuration (it does not).
    """
    have = set(tags)
    return [e for e in REGISTER if e.tags and set(e.tags).issubset(have)]


def register_json(level: Optional[str] = None) -> dict:
    """The whole register as JSON, optionally filtered to one level."""
    rows = [e for e in REGISTER if level is None or e.level == level]
    return {
        "ok": True,
        "levels": list(LEVELS),
        "count": len(rows),
        "entries": [e.to_json() for e in rows],
    }
