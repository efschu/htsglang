# SPDX-License-Identifier: Apache-2.0
"""#483: ``check_regressions`` must price prefill on the RESOLVED rates.

STATUS OF THE TICKET. The defect #483 describes is REAL and is already FIXED.
``key_solver.check_regressions`` built its cost model from ``rates`` and then
read ``rates.require_gemm_tflops()`` off the PRE-resolution object, so both
shipped anchors -- FP8 checkpoints whose 3080 ranks dispatch to weight-only
Marlin -- were priced on the dense bf16 fallback, i.e. on a 3.5:1 rank spread
where the hardware ran 9.7:1. The fix landed with #475 (``resolved =
model.rates``, ``key_solver.py:4777-4780``) and is written up in
``NOTE_475_phase_prefill_prediction.md`` SS5.

WHY THIS FILE EXISTS ANYWAY. The only test that exercises
``check_regressions`` at all is ``TestRegressionAnchors`` in
``test_key_solver.py``, and it is ``@unittest.skipUnless(_have(_FP8))`` --
it needs the real 27B-FP8 checkpoint on disk and SKIPS in every hermetic run.
Nothing pins the fixed behaviour where it is actually run, so a revert of
those four lines is invisible. This file is that pin, on inlined fixtures: a
synthetic FP8 checkpoint plus a hardware profile carrying the fp8 Marlin lanes
the card probe cannot measure, so RESOLVED and UNRESOLVED rates differ by
construction and the two prices are 7.9 percentage points apart.

CAN-FAIL ARM (executed, 2026-08-03): reverting ``key_solver.py:4778-4780`` to
``link = rates.link_bw_gbs`` / ``gemm = rates.require_gemm_tflops()`` /
``families = rates.gemm_family_tflops or None`` turns exactly two of these red,
and they are the two that read the arithmetic:

    test_prefill_is_priced_on_the_resolved_rates
        assert 2.85 == 10.703037772554701 +- 0.01
    test_without_a_profile_the_fallback_is_still_the_resolved_object
        assert 2.85 == 10.03463116349257 +- 0.01

Note what stays GREEN under the revert: ``test_the_row_reports_the_lane_it
_priced_on``. The row's ``gemm_format`` / ``gemm_lanes`` fields are read off
``resolved`` at ``key_solver.py:4795-4796`` no matter what priced the numbers,
so the provenance kept saying "fp8 Marlin" while the arithmetic ran on bf16.
That is precisely why the defect survived: the artifact named the right lane.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_cost_model_open_items as fx  # noqa: E402  one fixture rig, one place
from sglang.srt.planner import key_solver as ks  # noqa: E402

#: ``check_regressions`` picks the geometry off the anchor KEY: this one takes
#: the #264 branch -- budgets 29607/17780/17780, ranks in cuda order, and the
#: 2,1,1 -> 6,1,1 concentration. That is the vector pair where the lane
#: resolution is worth 7.9 points of predicted prefill, which is what makes it
#: the discriminating case. The measured numbers are the shipped anchor's; this
#: file asserts nothing about them, only about WHICH rates priced the row.
_KEY = "264_611_net_negative"
_BUDGETS = [32607 - 3000, 20480 - 2700, 20480 - 2700]
_RANK_GPU_ID = [0, 1, 2]
_REF_VEC = [2, 1, 1]
_CAND_VEC = [6, 1, 1]

_ANCHOR = ks.RegressionAnchor(
    key=_KEY,
    what="synthetic anchor, present only to drive the prefill pricing path",
    source="test_check_regressions_pricing_483.py",
    measured={"enc": +8.2, "dec": -13.7},
    tolerance_pct={"enc": 4.0, "dec": 4.0},
    tolerance_reason="not asserted here; this file checks WHICH rates priced it",
)


@pytest.fixture(scope="module")
def fp8_checkpoint(tmp_path_factory):
    return fx._write_checkpoint(str(tmp_path_factory.mktemp("fp8ckpt")), fx._FP8_QUANT)


@pytest.fixture(scope="module")
def profile():
    return fx._profile_with_lanes()


@pytest.fixture(autouse=True)
def _mamba_dtype(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")


def _row(model_dir, hardware_profile):
    (row,) = ks.check_regressions(
        model_dir, fx._PROBE, anchors=[_ANCHOR], hardware_profile=hardware_profile
    )
    return row


def _enc_from(rates, model_dir):
    """``enc`` as ``check_regressions`` would report it off ``rates``.

    Same terms in the same order as the function under test; the only degree
    of freedom left is which rate object the three prefill inputs come from.
    """
    from sglang.srt.uneven_perf import PlanInputs

    pi = PlanInputs(
        tp_size=3,
        model_path=model_dir,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        max_running_requests=16,
        rank_gpu_id=list(_RANK_GPU_ID),
        effective_vram_mib=list(_BUDGETS),
    )
    model = ks.build_cost_model(pi, list(_BUDGETS), list(_BUDGETS), rates)
    gemm = rates.require_gemm_tflops()
    link = rates.link_bw_gbs
    families = rates.gemm_family_tflops or None
    ref = model.perf.prefill_time_model(list(_REF_VEC), gemm, link, families)
    cand = model.perf.prefill_time_model(list(_CAND_VEC), gemm, link, families)
    return (ref / cand - 1.0) * 100.0


def _raw(hardware_profile):
    return ks.rates_from_probe(
        fx._PROBE, list(_RANK_GPU_ID), hardware_profile=hardware_profile
    )


class TestTheTwoRateObjectsReallyDiffer:
    """The precondition. Without it the file would pass on a tautology: an
    instrument that cannot tell the two prices apart cannot pin which one is
    used (spread precondition, CLAUDE.md)."""

    def test_the_probe_alone_cannot_see_the_marlin_lane(self, profile):
        raw = _raw(profile)
        # Pre-resolution: dense bf16 on the 3080s, native fp8 on the 5090 --
        # the mixed read the card probe produces, and the checkpoint's own
        # format has not entered the numbers at all yet.
        assert raw.gemm_format == "bf16"
        assert [round(x, 2) for x in raw.gemm_tflops] == [231.97, 65.57, 65.59]
        assert max(raw.gemm_tflops) / min(raw.gemm_tflops) == pytest.approx(
            3.54, abs=0.02
        )

    def test_resolution_moves_every_rank_onto_the_checkpoint_lane(self, profile):
        resolved = _raw(profile).resolve_gemm_format("fp8", None)
        assert resolved.gemm_format == "fp8"
        # 5090 native fp8, both 3080s on weight-only Marlin: the 9.7:1 spread
        # the hardware ran, against the 3.5:1 the probe alone reports.
        assert [round(x, 2) for x in resolved.gemm_tflops] == [566.88, 58.44, 59.15]
        assert max(resolved.gemm_tflops) / min(resolved.gemm_tflops) == pytest.approx(
            9.70, abs=0.02
        )

    def test_the_two_prices_are_far_apart(self, fp8_checkpoint, profile):
        """7.9 points on this vector pair. If they were close the rest of this
        file could not falsify anything."""
        raw = _raw(profile)
        unresolved_enc = _enc_from(raw, fp8_checkpoint)
        resolved_enc = _enc_from(raw.resolve_gemm_format("fp8", None), fp8_checkpoint)
        assert unresolved_enc == pytest.approx(2.85, abs=0.05)
        assert resolved_enc == pytest.approx(10.70, abs=0.05)


class TestPrefillIsPricedOnTheResolvedRates:
    def test_prefill_is_priced_on_the_resolved_rates(self, fp8_checkpoint, profile):
        """THE FALSIFIER. Red on the pre-#475 form of key_solver.py:4777-4780."""
        raw = _raw(profile)
        resolved_enc = _enc_from(raw.resolve_gemm_format("fp8", None), fp8_checkpoint)
        unresolved_enc = _enc_from(raw, fp8_checkpoint)
        reported = _row(fp8_checkpoint, profile)["predicted_pct"]["enc"]
        assert reported == pytest.approx(resolved_enc, abs=0.01)
        assert reported != pytest.approx(unresolved_enc, abs=0.5)

    def test_the_row_reports_the_lane_it_priced_on(self, fp8_checkpoint, profile):
        """The row's own provenance has to agree with its arithmetic. A row
        that names the fp8 lane while pricing on bf16 is the shape that kept
        #483 invisible for as long as it was."""
        row = _row(fp8_checkpoint, profile)
        assert row["gemm_format"] == "fp8"
        assert row["gemm_lanes"] == [
            "fp8 native (_scaled_mm)",
            "fp8 Marlin (weight-only)",
            "fp8 Marlin (weight-only)",
        ]

    def test_without_a_profile_the_fallback_is_still_the_resolved_object(
        self, fp8_checkpoint
    ):
        """With no hardware profile the Marlin lanes are unknown and the
        resolution falls back to dense WITH #324's warning. The pricing must
        still read the RESOLVED object: the fallback is a labelled rate, not a
        reason to read a different object."""
        row = _row(fp8_checkpoint, None)
        assert row["gemm_format"] == "fp8"
        expected = _enc_from(
            _raw(None).resolve_gemm_format("fp8", None), fp8_checkpoint
        )
        assert row["predicted_pct"]["enc"] == pytest.approx(expected, abs=0.01)


class TestDecodeUsesTheSameObject:
    """The other half of #483's claim: the decode price already went through
    ``model``. It does -- pinned here so the two halves cannot drift apart
    again."""

    def test_decode_is_priced_through_the_model(self, fp8_checkpoint, profile):
        from sglang.srt.uneven_perf import PlanInputs

        pi = PlanInputs(
            tp_size=3,
            model_path=fp8_checkpoint,
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            rank_gpu_id=list(_RANK_GPU_ID),
            effective_vram_mib=list(_BUDGETS),
        )
        model = ks.build_cost_model(pi, list(_BUDGETS), list(_BUDGETS), _raw(profile))
        f = model.perf.calibration.nonweight_fraction
        tw_ref = model.decode_weight_time(model.perf.mlp_unit_partition(list(_REF_VEC)))
        tw_cand = model.decode_weight_time(
            model.perf.mlp_unit_partition(list(_CAND_VEC))
        )
        step_ratio = f + (1.0 - f) * (tw_cand / tw_ref)
        expected = (1.0 / step_ratio - 1.0) * 100.0
        row = _row(fp8_checkpoint, profile)
        assert row["predicted_pct"]["dec"] == pytest.approx(
            round(expected, 2), abs=0.01
        )
