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
    #: lane vehicle (dense/GQA; on this box TP=3: 5090 head + 2 weightless
    #: 3080 workers), "moe_fp8" / "moe_marlin" = the offload vehicles.
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
    #: True when the row's fp band has NOT been measured on hardware yet. Such
    #: a row is fully specified (configs, reference, class, runner
    #: obligations) but must NOT gate: the #124 calibration lesson was that a
    #: guessed band can be two orders of magnitude wrong, and a wrong band is
    #: worse than no row. ``validate_matrix`` therefore admits ``band is
    #: None`` only together with this flag, and the runner must refuse to
    #: return a verdict for such a row until the band is filled in.
    pending_calibration: bool = False
    #: GPU-window-only evidence beyond the trajectory assertion.
    quality_gate: Optional[str] = None
    notes: str = ""


_LANE_BASE: Dict[str, Any] = {
    # TP=3/DCP=3 is the lane geometry every prior manual gate (#131/#133/
    # #134/#136a/#136b) validated on this box: head on the 5090 + two
    # weightless 3080 workers. The fastlane requires dcp_size == tp_size.
    "tp_size": 3,
    "dcp_size": 3,
    "weightless_kv_fastlane": True,
    "weightless_kv_head_rank": 0,
    "random_seed": PINNED_SEED,
}

_SOLO_BASE: Dict[str, Any] = {
    "tp_size": 1,
    "random_seed": PINNED_SEED,
}

#: The solo reference arm for the LANE rows is the TRUSTED-NUMERICS baseline,
#: so it runs eager: r2's upstream sync gave the standard TP=1 path
#: breakable/piecewise PREFILL CUDA graphs plus captured decode, and
#: graph-context kernel selection on that path is NOT part of the head-local
#: machine-zero contract (measured live at the GPU window: solo-with-graphs
#: prefill differs from the lane head's eager prefill by ~5.7e-2 across 90%
#: of the vocab, and solo graph decode adds ~0.3-0.9 per-step deltas).
#: Full-perf discipline is preserved: graphs stay ON for every TEST arm;
#: --disable-cuda-graph is the designated REFERENCE arm exactly as the #124
#: briefing allows. The lane's own graph path is separately proven
#: machine-zero (max|d| = 0 over 96 steps) by the graph_vs_eager_weightless
#: row. The MoE offload rows also boot eager on BOTH arms -- not by this
#: rationale but by necessity: the offload machinery's per-forward residency
#: plan is data-dependent and hard-gated against CUDA graphs in
#: fused_moe_triton/layer.py, so eager-vs-eager is the only common-mode
#: offload comparison that exists.
_SOLO_EAGER_REF: Dict[str, Any] = {
    **_SOLO_BASE,
    "disable_cuda_graph": True,
}

#: The chain-speculation configuration shared by BOTH arms of every spec row.
#: Matching it across arms is the substance of the oracle, not a formality --
#: an unmatched reference puts the verify forward's shape change in one arm
#: only, and then the comparison measures speculation rather than the feature
#: under test (Window 5's Gate 1). ``topk == 1`` is the only mode the lane
#: admits (tree verify stays rejected, #76/#139), and ``k`` must be fixed per
#: boot -- adaptive k is rejected on the lane precisely because it would make
#: the verify capture multi-shaped.
_CHAIN_SPEC: Dict[str, Any] = {
    "speculative_algorithm": "EAGLE3",
    "speculative_eagle_topk": 1,
    "speculative_num_steps": 3,
    "speculative_num_draft_tokens": 4,
    "speculative_draft_placement": "solo",
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
        reference_config=dict(_SOLO_EAGER_REF),
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
        reference_config=dict(_SOLO_EAGER_REF),
        reference_env={},
        expected_class=ByteIdentityClass.DECODE_CLASS,
        num_decode_tokens=96,
        band=24.0,
        near_tie_margin=1.0,
        notes=(
            "CROSS-RANK-MERGE path spec; 96-token window matches the B1 gate. "
            "CALIBRATED 2026-07-20 (run4, counting prompt, fp32 logits row "
            "dump): argmax-clean 96/96, per-step full-vocab max|d| measured "
            "median 0.84 / p95 6.4 / max 14.34; band = ~1.7x measured max. "
            "The 14.34 outlier (step 72) is a mid-rank token swing (ref "
            "+13.0 -> lane -1.4) with argmax margin ~17 at that step -- "
            "flagged in the activation report, does not affect the "
            "trajectory and does not compound. Gate prompt min top-2 margin "
            "3.2 (>> per-step deltas at the top ranks)."
        ),
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
        test_config={**_LANE_BASE, "weightless_kv_chunked_block_size": 1024},
        test_env={},
        reference_config=dict(_SOLO_EAGER_REF),
        reference_env={},
        expected_class=ByteIdentityClass.DECODE_CLASS,
        num_decode_tokens=96,
        band=10.0,
        near_tie_margin=1.0,
        notes=(
            "Runner must warm the prefix (two-request protocol) before the gate "
            "run. Block size 1024 = the graph-rung-ladder-validated value "
            "(#136a); B0's block=64 control was eager-only, and full-perf "
            "discipline runs this row with decode graphs ON. CALIBRATED "
            "2026-07-20 (run4, copy-task suffix): argmax-clean 96/96, "
            "measured per-step max|d| 5.02 (3.35 on the first-draft prompt); "
            "band = ~2x measured."
        ),
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
            "weightless_kv_chunked_block_size": 1024,
            "weightless_kv_host_spill_tokens": 16384,
            "weightless_kv_spill_device_cap": 2048,
        },
        test_env={},
        reference_config={**_LANE_BASE, "weightless_kv_chunked_block_size": 1024},
        reference_env={},
        expected_class=ByteIdentityClass.DECODE_CLASS,
        num_decode_tokens=96,
        band=4.5,
        near_tie_margin=1.0,
        notes=(
            "Reference = all-resident lane (streaming's own baseline), not solo; "
            "the lane-vs-solo delta is already covered by weightless_decode. "
            "spill_device_cap forces genuine host streaming on a small vehicle "
            "(runner asserts the prompt actually exceeds the per-rank device "
            "cap). CALIBRATED 2026-07-20 (run4): argmax-clean 96/96, measured "
            "per-step max|d| 2.22 (1.19 on the first-draft prompt); band = "
            "~2x measured."
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
        # The offload machinery is eager-only BY DESIGN (data-dependent
        # per-forward residency plan; hard-gated in fused_moe_triton/layer.py
        # against SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1 with graphs on), so
        # the valid comparison is eager-offload vs eager-no-offload
        # (common execution mode on both arms). The offload-x-graph question
        # is covered by the isolation gate tests/moe_offload/
        # test_capturable_gpu.py, see EXCLUDED_CASES.
        test_config=dict(_SOLO_EAGER_REF),
        test_env={"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": "0.25"},
        reference_config=dict(_SOLO_EAGER_REF),
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
            "scale, PROVISIONAL -- calibrate at GPU window. STATUS 2026-07-20 "
            "(activation): VEHICLE-BLOCKED on this box -- no FP8 MoE fits "
            "TP=1 (Qwen3.6-35B-A3B-FP8 is 31 GB vs the 5090's 32 GB: the "
            "no-offload REFERENCE arm cannot boot; the Qwen3.6-27B-FP8 "
            "checkpoint is a DENSE qwen3_5 model with no experts). Runner "
            "reports this instead of guessing (MODEL_ROLES['moe_fp8']=None)."
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
        # Eager on both arms: offload is eager-only by design (see the
        # fp8_offload row comment).
        test_config=dict(_SOLO_EAGER_REF),
        test_env={"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": "0.25"},
        reference_config=dict(_SOLO_EAGER_REF),
        reference_env={},
        expected_class=ByteIdentityClass.SELF_DET_NEAR_TIE,
        num_decode_tokens=64,
        band=1e-1,
        near_tie_margin=5e-2,
        needs_rerun=True,
        quality_gate="ppl within noise vs. no-offload (#120 bar); coherence; self-det 5/5",
        notes=(
            "Magnitude anchor: measured ~1e-2 intrinsic; #77/#120 acceptance "
            "bar. STATUS 2026-07-20 (activation, vehicle Qwen3.5-35B-A3B-"
            "GPTQ-Int4, FLAGGED FOR HUMAN JUDGMENT): run==run LOGIT "
            "bit-identity is unestablishable on this vehicle -- the falsifier "
            "probe showed the BASE no-offload server is itself non-bit-"
            "identical run-to-run (same boot, same prompt: per-step max|d| "
            "median 0.71 / max 2.02, 0 token flips; offload-on run-to-run "
            "median 0.69 / max 5.5, also 0 flips), i.e. the noise is "
            "intrinsic to the marlin-MoE/hybrid kernel path (atomics "
            "reduction order), NOT an offload regression, and offload adds "
            "nothing beyond that floor (ref-vs-test same magnitude, 0 flips, "
            "min margin 2.5). The #77 'self-det 5/5' provenance was the "
            "122B vehicle at token level. TOKEN-level self-determinism and "
            "near-tie-only divergence hold 64/64 here; the LOGIT-level "
            "run==run contract this row encodes does not. Left encoded "
            "as-is: whether to relax the class to token-level self-det for "
            "marlin vehicles is a contract decision, not a calibration."
        ),
    ),
    CaseSpec(
        case_id="weightless_spec_matched",
        description=(
            "Weightless-KV lane WITH chain speculation (#143) vs. a TP=1 solo "
            "boot carrying the IDENTICAL speculative configuration. The "
            "matched reference is the whole point: speculation changes the "
            "target forward's shape (k+1 rows in one verify instead of one "
            "decode row per token), so an unmatched reference measures "
            "speculation, not the lane."
        ),
        model_role="dense_lane",
        test_config={**_LANE_BASE, **_CHAIN_SPEC},
        test_env={},
        reference_config={**_SOLO_EAGER_REF, **_CHAIN_SPEC},
        reference_env={},
        expected_class=ByteIdentityClass.SPEC_NEAR_TIE,
        num_decode_tokens=256,
        band=None,  # MEASURE FIRST -- see notes
        near_tie_margin=1.0,
        needs_rerun=True,
        pending_calibration=True,
        quality_gate=(
            "mean accept length (meta_info.spec_accept_length) clears its "
            "floor and is recorded beside the reference arm's; Window 5 saw "
            "the lane run 8-12% below the plain path on the same prompts, "
            "unexplained -- this row is the content-pinned setting in which "
            "that number is finally comparable"
        ),
        notes=(
            "PRIMARY lane x spec oracle. Runner obligations: (1) capture the "
            "target VERIFY logits, not ModelRunner.sample's decode row -- the "
            "spec path never reaches that dump hook (bs*(k+1) rows fail its "
            "shape[0] == 1 guard); use --determinism-logits-dump-dir with the "
            "spec verify dump. (2) Project both arms with "
            "SpecRun.to_trajectory() before comparing. (3) The draft is "
            "bit-identical across the arms BY GEOMETRY -- the lane forces "
            "--speculative-draft-placement solo with an UNSHARDED draft on "
            "the head rank, and a TP=1 reference's draft is unsharded too -- "
            "so the whole draft side is common-mode. Verify that assumption "
            "at the window with the cheap TP=1 solo-vs-split A/B rather than "
            "assuming it. BAND IS DELIBERATELY None: it must be MEASURED at "
            "the GPU window before this row can gate, and it will NOT equal "
            "weightless_decode's 24.0 -- the projected rows come from a "
            "verify-shaped forward, and once the arms' accept lengths differ "
            "the same token index is scored at different slot positions in "
            "the two arms, which is a further fp difference. The #124 "
            "calibration lesson applies verbatim: a guessed band is worse "
            "than no row."
        ),
    ),
    CaseSpec(
        case_id="spec_replay_teacher_forced",
        description=(
            "Teacher-forced replay: the token sequence the lane+spec arm "
            "emitted is fed back as a plain prompt to a NON-speculative boot "
            "of the same geometry, and that boot's per-position argmax must "
            "reproduce the sequence. This is the only row that reaches the "
            "question the token-level checks are blind to -- did the verify "
            "forward attend over the RIGHT KV context."
        ),
        model_role="dense_lane",
        test_config={**_LANE_BASE, **_CHAIN_SPEC},
        test_env={},
        reference_config=dict(_SOLO_EAGER_REF),
        reference_env={},
        expected_class=ByteIdentityClass.SPEC_NEAR_TIE,
        num_decode_tokens=256,
        band=None,  # MEASURE FIRST -- see notes
        near_tie_margin=1.0,
        needs_rerun=True,
        pending_calibration=True,
        notes=(
            "TEACHER-FORCED REPLAY, not a free-running A/B. The reference arm "
            "runs ONE prefill over the test arm's emitted sequence and yields "
            "one logits row per position; no reference generation happens, so "
            "the two arms cannot fork and the comparison stays step-aligned "
            "for the whole window. That is what makes this the sharpest "
            "instrument available: an accept plumbing bug is caught by "
            "check_accept_rule_exactness, but a verify reading the wrong KV "
            "slots emits its own argmax perfectly consistently and is only "
            "visible against an independently-computed continuation. Still "
            "near-tie-gated, not 0-flip: a single prefill over N positions is "
            "a third fp path, different from both decode and verify. BAND "
            "DELIBERATELY None -- measure at the window."
        ),
    ),
]


#: Deliberately-NOT-encoded cases, with the reason. These document negative
#: knowledge so nobody "helpfully" adds them back as gates.
EXCLUDED_CASES: Dict[str, str] = {
    "spec_vs_nospec_token_identity": (
        "MEASURED-FALSE PREMISE (#143 Window 5), not an oversight. "
        "docs_new/weightless_chain_spec.md §5.2 originally asserted that a "
        "lane WITH chain spec at temperature 0 must be TOKEN-IDENTICAL to the "
        "lane WITHOUT spec -- 'a hard equality, not a similarity'. It is not. "
        "Two control arms on plain TP=2 with the lane switched OFF show "
        "speculation alone breaking strict token identity at temperature 0 on "
        "the default path (common prefixes 1 / 172 / 16 / 256-of-256 across "
        "four content classes). The greedy verify's ACCEPT RULE is exact -- "
        "integer equality against the target argmax, no threshold, no "
        "tolerance (eagle_utils.verify_tree_greedy_func; the "
        "--speculative-accept-threshold-* knobs are consumed by the sampling "
        "branch only) -- so a divergence cannot originate in the accept "
        "decision. It originates in the target LOGITS: a verify scores k+1 "
        "positions in one forward, a different batch shape and attention "
        "kernel path from a one-row decode, hence a different fp reduction "
        "order and a different argmax at near-ties. Note this is a claim "
        "about the observed configuration, and 'plausible' is not 'proven': "
        "the DISCRIMINATOR is the margin at the divergence, which is what the "
        "weightless_spec_matched and spec_replay_teacher_forced rows measure. "
        "Independently of fp there is also a STRUCTURAL losslessness caveat "
        "that no reassociation argument covers: with repetition / presence / "
        "frequency penalties active, a verify applies the round's pre-step "
        "penalty vector to all k+1 draft positions (eagle_utils.py:919-941, "
        "upstream's 'relaxed version of penalties for speculative decoding') "
        "and only ONE token per round reaches the penalizers "
        "(eagle_prepare_for_decode -> cumulate_penalty_output_tokens takes "
        "req.output_ids[-1]), so the intermediate accepted tokens are never "
        "counted. Inert at default sampling params, and therefore NOT the "
        "explanation for Window 5 -- but it means 'greedy speculation is "
        "lossless' is false by construction whenever penalties are set. "
        "Do not add a spec-on-vs-spec-off token-identity gate."
    ),
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
    "weightless_worker_fp8_kv": (
        "#127 per-ROLE KV precision (--weightless-kv-worker-cache-dtype "
        "fp8_e5m2: workers store their KV token-shard in fp8, the head keeps "
        "its own format). NOT YET A GATED ROW -- deliberately, not by "
        "oversight. The delta here is QUANTIZATION on ~(dcp_size-1)/dcp_size "
        "of the tokens, not fp reassociation, so it needs its own measured "
        "band; the #124 calibration lesson was that a guessed band can be two "
        "orders of magnitude wrong, and a wrong band is worse than no row. "
        "Two rows belong here once a GPU window has measured them: (a) "
        "worker-fp8 lane vs the SAME lane inheriting --kv-cache-dtype -> "
        "DECODE_CLASS with a measured band, and (b) head_local_prefill under "
        "a role split -> MACHINE_ZERO, which is the concrete claim the design "
        "makes for splitting by ROLE rather than switching the whole group "
        "(the head's KV format is untouched, so its head-local path must stay "
        "bit-exact). Recipe in docs_new/weightless_kv_role_precision.md §8."
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
        elif c.pending_calibration:
            assert c.band is None, (
                f"{c.case_id}: pending_calibration rows carry NO band -- a "
                "value here would read as measured. Clear the flag when the "
                "band is measured on hardware."
            )
            assert c.near_tie_margin is not None, (
                f"{c.case_id}: the near-tie margin is a contract choice, not a "
                "measurement, and is required even while the band is pending"
            )
        else:
            assert c.band is not None and c.near_tie_margin is not None, (
                f"{c.case_id}: {c.expected_class.value} requires explicit band "
                "and near_tie_margin (or pending_calibration=True)"
            )
        if c.expected_class in (
            ByteIdentityClass.SELF_DET_NEAR_TIE,
            ByteIdentityClass.SPEC_NEAR_TIE,
        ):
            assert c.needs_rerun, (
                f"{c.case_id}: {c.expected_class.value} without a rerun cannot "
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
        # The Window-5 guard: a speculative arm may never be gated as
        # bit-exact or argmax-clean against a NON-speculative reference. See
        # EXCLUDED_CASES["spec_vs_nospec_token_identity"] -- that premise was
        # measured false with the feature under test switched off.
        spec_on = c.test_config.get("speculative_algorithm") is not None
        spec_off = c.reference_config.get("speculative_algorithm") is None
        if spec_on and spec_off:
            assert c.expected_class is ByteIdentityClass.SPEC_NEAR_TIE, (
                f"{c.case_id}: a speculative arm against a non-speculative "
                "reference is SPEC_NEAR_TIE, never MACHINE_ZERO/DECODE_CLASS "
                "-- see EXCLUDED_CASES['spec_vs_nospec_token_identity']"
            )
        if c.expected_class is ByteIdentityClass.SPEC_NEAR_TIE:
            assert spec_on, (
                f"{c.case_id}: SPEC_NEAR_TIE requires speculation on the arm "
                "under test"
            )
        assert (c.test_config, c.test_env) != (c.reference_config, c.reference_env), (
            f"{c.case_id}: test and reference configs are identical -- the row "
            "compares nothing"
        )
