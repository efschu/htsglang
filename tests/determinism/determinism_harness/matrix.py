# SPDX-License-Identifier: Apache-2.0
"""Declarative test matrix: (feature/config) -> expected byte-identity class.

Each row names the run-under-test config, the reference it is compared
against, the expected :class:`ByteIdentityClass`, and the class parameters
(fp band / near-tie margin / gate length). The GPU runner (r2 window)
consumes these rows verbatim; the CPU test suite validates the matrix's
internal consistency -- including the guard that the retracted FP8
overclaim ("offload == MACHINE_ZERO vs no-offload") can never re-enter.

SEED-PINNING DISCIPLINE (memory ``hetero-spec-determinismus``): every gate
run -- reference, test, and rerun -- boots with the SAME pinned
``--random-seed`` (:data:`PINNED_SEED`). sglang otherwise randomizes the
seed per boot (server_args: ``random.randint`` when unset), which silently
degrades every comparison. Cross-boot near-tie sensitivity is pre-existing
and expected; the pinned seed removes the avoidable part. Reruns for
self-determinism must be same-boot where possible, or identically-seeded
fresh boots (recorded as such in the gate log).

BAND CALIBRATION: numeric ``band`` / ``near_tie_margin`` values below are
PROVISIONAL, anchored to measured magnitudes in the memory files (marlin
~1e-2 intrinsic; MoE capture floor ~3.3e-2; FP8 sub-ULP). They are to be
calibrated against real trajectories at the GPU window before the harness
gates CI -- tighten, don't loosen, and record the measured floor per row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .classes import ByteIdentityClass

__all__ = ["PINNED_SEED", "CaseSpec", "TEST_MATRIX", "EXCLUDED_CASES", "get_case", "validate_matrix"]

#: The single pinned seed for ALL gate runs (ref, test, rerun). The value is
#: arbitrary; what matters is that it is fixed, shared by every boot in a
#: comparison, and never silently defaulted (sglang randomizes when unset).
PINNED_SEED = 1234


@dataclass(frozen=True)
class CaseSpec:
    """One matrix row: config under test + reference + expected class."""

    case_id: str
    description: str
    #: which model vehicle the runner should use (mapped to a concrete
    #: local model path at the GPU window): "dense_lane" = the weightless-KV
    #: lane vehicle (dense/GQA, TP=2 head+worker), "moe_fp8" / "moe_marlin"
    #: = the offload vehicles.
    model_role: str
    #: sglang server args (server_args field names) for the run under test.
    test_config: Mapping[str, Any]
    #: env vars for the run under test. MUST be baked into the MAIN process
    #: (pickled server args path) -- sglang scrubs custom env for scheduler
    #: TP workers, a worker-side env toggle is a silent no-op (memory
    #: ``full-perf-testen``).
    test_env: Mapping[str, str]
    #: sglang server args for the reference run.
    reference_config: Mapping[str, Any]
    reference_env: Mapping[str, str]
    expected_class: ByteIdentityClass
    #: decode-gate length (number of greedy decode steps to compare);
    #: 1 for single-shot-prefill gates.
    num_decode_tokens: int
    #: fp-reassociation band / near-tie margin (None for MACHINE_ZERO rows).
    band: Optional[float] = None
    near_tie_margin: Optional[float] = None
    seed: int = PINNED_SEED
    #: whether SELF_DET_NEAR_TIE reruns are needed (run==run evidence).
    needs_rerun: bool = False
    #: GPU-window-only evidence beyond the trajectory assertion.
    quality_gate: Optional[str] = None
    notes: str = ""


_LANE_BASE: Dict[str, Any] = {
    "tp_size": 2,
    "dcp_size": 2,
    "weightless_kv_fastlane": True,
    "weightless_kv_head_rank": 0,
    "random_seed": PINNED_SEED,
}

_SOLO_BASE: Dict[str, Any] = {
    "tp_size": 1,
    "random_seed": PINNED_SEED,
}


TEST_MATRIX: List[CaseSpec] = [
    CaseSpec(
        case_id="head_local_prefill_single_shot",
        description=(
            "Weightless-KV lane, single-shot prefill with empty prefix: the head "
            "computes the full forward locally (workers never embed), so the "
            "last-position prefill logits must be bit-exact vs. a TP=1 solo boot."
        ),
        model_role="dense_lane",
        test_config=dict(_LANE_BASE),
        test_env={},
        reference_config=dict(_SOLO_BASE),
        reference_env={},
        expected_class=ByteIdentityClass.MACHINE_ZERO,
        num_decode_tokens=1,
        notes="HEAD-LOCAL path spec: machine-zero vs TP=1 solo (weightless-kv-lane).",
    ),
    CaseSpec(
        case_id="weightless_decode",
        description=(
            "Weightless-KV lane greedy decode: cross-rank LSE merge "
            "(cp_lse_ag_out) reassociates fp -- argmax-clean trajectory vs. TP=1 "
            "solo, bounded per-step delta, non-compounding."
        ),
        model_role="dense_lane",
        test_config=dict(_LANE_BASE),
        test_env={},
        reference_config=dict(_SOLO_BASE),
        reference_env={},
        expected_class=ByteIdentityClass.DECODE_CLASS,
        num_decode_tokens=96,
        band=5e-2,
        near_tie_margin=3e-2,
        notes="CROSS-RANK-MERGE path spec; 96-token window matches the B1 gate.",
    ),
    CaseSpec(
        case_id="graph_vs_eager_weightless",
        description=(
            "Weightless lane with decode CUDA graphs vs. the SAME lane config "
            "eager: this path has no capture-gated dual-stream branches, so "
            "graph == eager bit-exact (#133 machine-null; B2 part-2 captured "
            "streaming max|d|=0)."
        ),
        model_role="dense_lane",
        test_config=dict(_LANE_BASE),  # graphs on = default (full-perf discipline)
        test_env={},
        reference_config={**_LANE_BASE, "disable_cuda_graph": True},
        reference_env={},
        expected_class=ByteIdentityClass.MACHINE_ZERO,
        num_decode_tokens=96,
        notes=(
            "Reference here is eager-weightless, NOT solo. Contrast: MoE-offload "
            "graph-vs-eager is NOT machine-zero (see EXCLUDED_CASES)."
        ),
    ),
    CaseSpec(
        case_id="chunked_prefill_with_prefix",
        description=(
            "Weightless lane, chunked prefill over a warm radix prefix "
            "(has_prefix path, rank-uniform collectives): decode-class vs. TP=1 "
            "solo -- the prefix merge goes through the same cross-rank LSE merge."
        ),
        model_role="dense_lane",
        test_config={**_LANE_BASE, "weightless_kv_chunked_block_size": 64},
        test_env={},
        reference_config=dict(_SOLO_BASE),
        reference_env={},
        expected_class=ByteIdentityClass.DECODE_CLASS,
        num_decode_tokens=96,
        band=5e-2,
        near_tie_margin=3e-2,
        notes="Runner must warm the prefix (two-request protocol) before the gate run.",
    ),
    CaseSpec(
        case_id="streaming_host_spill",
        description=(
            "Weightless lane with the tiered host-spill KV fabric (B1/B2: "
            "static slot->tier map, H2D block staging, per-rank pool shrink) vs. "
            "the SAME lane all-resident: streaming must add nothing over the B0 "
            "fp floor (measured 96/96 argmax-identical)."
        ),
        model_role="dense_lane",
        test_config={
            **_LANE_BASE,
            "weightless_kv_chunked_block_size": 64,
            "weightless_kv_host_spill_tokens": 8192,
            "weightless_kv_spill_device_cap": 2048,
        },
        test_env={},
        reference_config={**_LANE_BASE, "weightless_kv_chunked_block_size": 64},
        reference_env={},
        expected_class=ByteIdentityClass.DECODE_CLASS,
        num_decode_tokens=96,
        band=5e-2,
        near_tie_margin=3e-2,
        notes=(
            "Reference = all-resident lane (streaming's own baseline), not solo; "
            "the lane-vs-solo delta is already covered by weightless_decode. "
            "spill_device_cap forces genuine host streaming on a small vehicle."
        ),
    ),
    CaseSpec(
        case_id="fp8_offload",
        description=(
            "FP8-Triton MoE expert offload (token-wave processing, sorted-slot "
            "deterministic residency) vs. no-offload: self-deterministic, "
            "divergence only at near-ties (sub-ULP magnitude). NOT bit-exact to "
            "no-offload -- the compacted resident+scratch layout changes the fp "
            "reduction order (author retraction 0fb3d8007)."
        ),
        model_role="moe_fp8",
        test_config=dict(_SOLO_BASE),
        test_env={"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": "0.25"},
        reference_config=dict(_SOLO_BASE),
        reference_env={},  # fraction unset = no offload
        expected_class=ByteIdentityClass.SELF_DET_NEAR_TIE,
        num_decode_tokens=256,
        band=1e-3,
        near_tie_margin=1e-3,
        needs_rerun=True,
        quality_gate="ppl within noise vs. no-offload (#120 bar); coherence; self-det 5/5",
        notes=(
            "256-token window deliberately matches the rerun that exposed the "
            "overclaim (118/256, fork at token 115 near-tie). Band is sub-ULP-"
            "scale, PROVISIONAL -- calibrate at GPU window."
        ),
    ),
    CaseSpec(
        case_id="marlin_offload",
        description=(
            "GPTQ/AWQ marlin-Int4 MoE expert offload (fixed-resident+scratch) "
            "vs. no-offload: buffer shrink changes moe_align tiling -> intrinsic "
            "~1e-2 logits delta that does NOT scale with spill (f=0.99 "
            "falsifier); confident tokens byte-identical, only genuine near-ties "
            "flip. Bit-exactness is UNREACHABLE on this kernel (by design)."
        ),
        model_role="moe_marlin",
        test_config=dict(_SOLO_BASE),
        test_env={"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": "0.25"},
        reference_config=dict(_SOLO_BASE),
        reference_env={},
        expected_class=ByteIdentityClass.SELF_DET_NEAR_TIE,
        num_decode_tokens=64,
        band=5e-2,
        near_tie_margin=3e-2,
        needs_rerun=True,
        quality_gate="ppl within noise vs. no-offload (#120 bar); coherence; self-det 5/5",
        notes="Magnitude anchor: measured ~1e-2 intrinsic; #77/#120 acceptance bar.",
    ),
]


#: Deliberately-NOT-encoded cases, with the reason. These document negative
#: knowledge so nobody "helpfully" adds them back as gates.
EXCLUDED_CASES: Dict[str, str] = {
    "moe_offload_graph_vs_eager": (
        "MoE-offload captured-vs-eager is NOT machine-zero and NOT gateable as "
        "such: even at fraction=1.0 with zero offload code active, graph vs "
        "eager shows ~3.3e-2 (capture-gated dual-stream branches + capture-"
        "context kernel selection in the MoE stack). The valid gate is the "
        "ISOLATION proof that the offload machinery contributes exactly zero "
        "on top of that pre-existing floor (tests/moe_offload/"
        "test_capturable_gpu.py), not an end-to-end graph==eager comparison. "
        "Contrast graph_vs_eager_weightless, which IS machine-zero because the "
        "weightless lane has no capture-gated branches."
    ),
    "fp8_offload_machine_zero": (
        "RETRACTED OVERCLAIM (0fb3d8007): FP8 offload is NOT byte-identical to "
        "no-offload. The 64-token 'proof' was confident-prompt luck; the "
        "256-token rerun matched 118/256 with a near-tie fork at token 115. "
        "torch.equal held for the kernel-level tensor contract, not for "
        "server token-ID identity. The correct class is SELF_DET_NEAR_TIE "
        "(see the fp8_offload row). test_matrix.py enforces this exclusion."
    ),
}


def get_case(case_id: str) -> CaseSpec:
    for case in TEST_MATRIX:
        if case.case_id == case_id:
            return case
    raise KeyError(f"no matrix row {case_id!r}")


def validate_matrix() -> None:
    """Structural consistency of the matrix; raises AssertionError on breach.

    Called by the CPU test suite; the GPU runner calls it before any boot.
    """
    ids = [c.case_id for c in TEST_MATRIX]
    assert len(ids) == len(set(ids)), f"duplicate case ids: {ids}"
    for c in TEST_MATRIX:
        assert c.num_decode_tokens >= 1, c.case_id
        assert c.seed == PINNED_SEED, (
            f"{c.case_id}: seed {c.seed} != PINNED_SEED {PINNED_SEED} -- all "
            "gates share one pinned seed"
        )
        for cfg in (c.test_config, c.reference_config):
            assert cfg.get("random_seed") == PINNED_SEED, (
                f"{c.case_id}: boot config missing pinned random_seed "
                "(sglang randomizes when unset)"
            )
        if c.expected_class is ByteIdentityClass.MACHINE_ZERO:
            assert c.band is None and c.near_tie_margin is None, (
                f"{c.case_id}: MACHINE_ZERO admits no band -- a band on a "
                "machine-zero row silently weakens the contract"
            )
            assert not c.needs_rerun, c.case_id
        else:
            assert c.band is not None and c.near_tie_margin is not None, (
                f"{c.case_id}: {c.expected_class.value} requires explicit band "
                "and near_tie_margin"
            )
        if c.expected_class is ByteIdentityClass.SELF_DET_NEAR_TIE:
            assert c.needs_rerun, (
                f"{c.case_id}: SELF_DET_NEAR_TIE without a rerun cannot "
                "establish run==run"
            )
        # The overclaim guard: no offload case may claim bit-exactness or
        # even argmax-clean identity vs. the no-offload reference.
        if "offload" in c.case_id or "SGLANG_MOE_RESIDENT_EXPERT_FRACTION" in c.test_env:
            assert c.expected_class is ByteIdentityClass.SELF_DET_NEAR_TIE, (
                f"{c.case_id}: offload vs no-offload is SELF_DET_NEAR_TIE, "
                "never MACHINE_ZERO/DECODE_CLASS -- see EXCLUDED_CASES"
                "['fp8_offload_machine_zero'] (retracted overclaim 0fb3d8007)"
            )
        assert (c.test_config, c.test_env) != (c.reference_config, c.reference_env), (
            f"{c.case_id}: test and reference configs are identical -- the row "
            "compares nothing"
        )
