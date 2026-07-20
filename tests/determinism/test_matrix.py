# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the declarative test matrix + the runner seam.

Locks the (feature/config) -> class contract: the expected rows exist with
the right classes and references, the pinned-seed discipline is enforced on
every boot config, and -- permanently -- the retracted FP8 overclaim
("offload == machine-zero vs no-offload") cannot re-enter the matrix.
"""

import pytest

from determinism_harness import (
    EXCLUDED_CASES,
    PINNED_SEED,
    TEST_MATRIX,
    ByteIdentityClass,
    Trajectory,
    evaluate_case,
    get_case,
    run_case_config,
)
from determinism_harness.matrix import validate_matrix
from test_primitives import confident_logits, make_near_tie_pair, traj, with_noise

EXPECTED_ROWS = {
    "head_local_prefill_single_shot": ByteIdentityClass.MACHINE_ZERO,
    "weightless_decode": ByteIdentityClass.DECODE_CLASS,
    "graph_vs_eager_weightless": ByteIdentityClass.MACHINE_ZERO,
    "chunked_prefill_with_prefix": ByteIdentityClass.DECODE_CLASS,
    "streaming_host_spill": ByteIdentityClass.DECODE_CLASS,
    "fp8_offload": ByteIdentityClass.SELF_DET_NEAR_TIE,
    "marlin_offload": ByteIdentityClass.SELF_DET_NEAR_TIE,
}


def test_matrix_validates():
    validate_matrix()


def test_matrix_covers_exactly_the_specified_gates():
    assert {c.case_id: c.expected_class for c in TEST_MATRIX} == EXPECTED_ROWS


def test_seed_pinning_discipline():
    for c in TEST_MATRIX:
        assert c.seed == PINNED_SEED
        assert c.test_config["random_seed"] == PINNED_SEED
        assert c.reference_config["random_seed"] == PINNED_SEED


def test_offload_rows_can_never_claim_bit_exactness():
    """Permanent guard against the retracted overclaim (0fb3d8007)."""
    for c in TEST_MATRIX:
        offloady = "offload" in c.case_id or "SGLANG_MOE_RESIDENT_EXPERT_FRACTION" in c.test_env
        if offloady:
            assert c.expected_class is ByteIdentityClass.SELF_DET_NEAR_TIE, c.case_id
            assert c.needs_rerun, c.case_id
            assert c.quality_gate, c.case_id
    assert "fp8_offload_machine_zero" in EXCLUDED_CASES
    assert "0fb3d8007" in EXCLUDED_CASES["fp8_offload_machine_zero"]
    assert "moe_offload_graph_vs_eager" in EXCLUDED_CASES


def test_machine_zero_rows_have_no_band():
    for c in TEST_MATRIX:
        if c.expected_class is ByteIdentityClass.MACHINE_ZERO:
            assert c.band is None and c.near_tie_margin is None, c.case_id


def test_graph_row_references_eager_same_lane_not_solo():
    c = get_case("graph_vs_eager_weightless")
    assert c.reference_config.get("disable_cuda_graph") is True
    assert c.reference_config.get("weightless_kv_fastlane") is True  # NOT solo
    assert "disable_cuda_graph" not in c.test_config  # full-perf: graphs on


def test_streaming_row_references_resident_lane_not_solo():
    c = get_case("streaming_host_spill")
    assert c.test_config["weightless_kv_host_spill_tokens"] > 0
    assert c.test_config["weightless_kv_spill_device_cap"] > 0
    assert "weightless_kv_host_spill_tokens" not in c.reference_config
    assert c.reference_config.get("weightless_kv_fastlane") is True


def test_lane_rows_use_the_lane_and_solo_reference():
    for cid in ("head_local_prefill_single_shot", "weightless_decode", "chunked_prefill_with_prefix"):
        c = get_case(cid)
        assert c.test_config.get("weightless_kv_fastlane") is True
        assert c.reference_config.get("tp_size") == 1
        assert "weightless_kv_fastlane" not in c.reference_config


def test_evaluate_case_dispatch_on_synthetic_data():
    """End-to-end wiring proof without a GPU: matrix row -> class check."""
    mz = get_case("head_local_prefill_single_shot")
    ref = traj(confident_logits(T=1))
    assert evaluate_case(mz, ref=ref, test=traj(ref.logits.clone())).ok
    assert not evaluate_case(mz, ref=ref, test=traj(with_noise(ref.logits, 1e-4))).ok

    dec = get_case("weightless_decode")
    ref = traj(confident_logits())
    assert evaluate_case(dec, ref=ref, test=traj(with_noise(ref.logits, 1e-4))).ok

    sd = get_case("fp8_offload")
    a, b, _ = make_near_tie_pair()
    assert evaluate_case(sd, ref=a, test=b, rerun=traj(b.logits.clone())).ok
    assert not evaluate_case(sd, ref=a, test=b, rerun=None).ok


def test_runner_is_stubbed_with_explicit_seam():
    """The GPU boot wiring is the ONLY stub, and it says so."""
    with pytest.raises(NotImplementedError) as ei:
        run_case_config(get_case("weightless_decode"), "test")
    assert "r2" in str(ei.value)


def test_trajectory_rejects_malformed_input():
    import torch

    with pytest.raises(ValueError):
        Trajectory(token_ids=[1, 2], logits=torch.zeros(3, 8), seed=PINNED_SEED)
    with pytest.raises(ValueError):
        Trajectory(token_ids=[1], logits=torch.zeros(8), seed=PINNED_SEED)
