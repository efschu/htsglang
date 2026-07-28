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
        level=NOT_DEFAULT,
        evidence="task #203, longer run",
        tags=("vocab-ratio",),
        scope="rig",
    ),
    RejectedEntry(
        key="spec_k4_mixed",
        what="speculative k=4 on mixed cards",
        verdict="not a default: lost the #103 matrix",
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
        level=NOT_DEFAULT,
        evidence="#264 A/B 2026-07-28, INTEGRATION_R3_VALIDATION (539154288d)",
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
        level=BLOCKED,
        evidence="server_args.py:4992, hard reject at arg parse",
        tags=("spill", "pd"),
        scope="general",
    ),
    RejectedEntry(
        key="spill_with_pp_dp",
        what="session KV spill with --pp-size > 1 or --dp-size > 1",
        verdict=(
            "BLOCKED by the engine: spill S1 supports single-node pure "
            "TP/DCP only"
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
            "BLOCKED by the engine: pp_size > 1 asserts "
            "disable_overlap_schedule and speculative_algorithm is None. It "
            "is a hard assert, not an auto-disable — the boot dies at arg "
            "parse. Every PP number on this rig is therefore a no-spec "
            "number and must not be put next to a NEXTN one"
        ),
        level=BLOCKED,
        evidence="server_args.py:11214",
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
        level=BLOCKED,
        evidence="DESIGN_107, the fork in the road",
        tags=("pd", "cuda-ipc-weights"),
        scope="general",
    ),
    RejectedEntry(
        key="intra_rig_collective_overlap",
        what="intra-rig collective overlap (a second NCCL lane)",
        verdict=(
            "measured, NOT built: 15.8-23.6 % of the budget is structurally "
            "non-overlappable and the single overlappable pair is 1.2 %, "
            "below the detection floor"
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
        level=BLOCKED,
        evidence="full-perf-testing rule, extension 2026-07-20",
        tags=("dcp-overlap",),
        scope="general",
    ),
    RejectedEntry(
        key="htccl_ring_bidir",
        what="splitting the HTCCL ring bidirectionally (two workers per direction)",
        verdict=(
            "REGRESSION: +17 % at 80 KiB. A ring step is two requests in "
            "lockstep; there is no concurrency to expose. "
            "SGLANG_HTCCL_UCX_RING_BIDIR stays 0"
        ),
        level=BLOCKED,
        evidence="#266 A/B 2026-07-28, d6d4231e5a",
        tags=("htccl-ring-bidir",),
        scope="general",
    ),
    RejectedEntry(
        key="path_bundling",
        what="path bundling (one transfer striped over both lines)",
        verdict=(
            "not built before somebody with full lanes shows a need; only "
            "the CHOICE of line per message class is released (#240)"
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
        level=NOT_DEFAULT,
        evidence="the #244 + #263 + #266 chain",
        tags=("crossrig-tp-push",),
        scope="rig",
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
