"""Task #49 level-2 probe (2026-09-20, after fn8c2 stayed red with the barrier).

The existing VERDICT can only say "the row was written and is not finite". It
never looks at the INPUT, so "the kernel computed garbage" and "the kernel
faithfully multiplied a NaN it was handed" produce the SAME verdict. That is
the gap these tests pin shut: the level-2 probe walks
A -> GEMM1 -> ACT -> GEMM2 and names the first stage that carries a bad row,
plus whether the bad output rows are a subset of the bad input rows.

Everything here is CPU-only.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/spinning/wt-nan49c-0920/python \\
    /spinning/htsglang-gpu/.venv/bin/python -m pytest -q \\
    tests/moe_offload/test_marlin_stage_probe_0920.py
"""

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    STAGE_ORDER,
    _bad_rows,
    _first,
    _n,
    _weights_scales_finite,
    classify_nonfinite_origin,
    classify_transport,
    marlin_c_sentinel_on,
    marlin_probe_level,
    marlin_stage_probe_on,
)
import sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe as fmm


@pytest.fixture(autouse=True)
def _reset_memos():
    fmm._PROBE_LEVEL["n"] = None
    fmm._C_SENTINEL["on"] = None
    fmm._STAGE_LOGGED["n"] = 0
    yield
    fmm._PROBE_LEVEL["n"] = None
    fmm._C_SENTINEL["on"] = None
    fmm._STAGE_LOGGED["n"] = 0


# --- the level ladder -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,level", [(None, 0), ("0", 0), ("1", 1), ("true", 1), ("on", 1), ("2", 2)]
)
def test_probe_level_ladder(monkeypatch, raw, level):
    if raw is None:
        monkeypatch.delenv("SGLANG_MOE_MARLIN_C_SENTINEL", raising=False)
    else:
        monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", raw)
    assert marlin_probe_level() == level
    assert marlin_c_sentinel_on() is (level >= 1)
    assert marlin_stage_probe_on() is (level >= 2)


def test_level_1_still_means_exactly_what_it_meant(monkeypatch):
    """Every fn8ap/fn8aq/fn8ar/fn8as/fn8c* boot line said '=1'. Those boots must
    keep their exact meaning, or their logs stop being comparable."""
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "1")
    assert marlin_c_sentinel_on() is True
    assert marlin_stage_probe_on() is False


# --- the origin classifier --------------------------------------------------


def test_origin_names_the_first_bad_stage():
    assert (
        classify_nonfinite_origin(
            {"INPUT-A": 0, "GEMM1": 0, "ACT": 0, "GEMM2": 7}
        )
        == "ENTERS-AT-GEMM2"
    )
    assert (
        classify_nonfinite_origin({"INPUT-A": 0, "GEMM1": 4, "ACT": 4, "GEMM2": 4})
        == "ENTERS-AT-GEMM1"
    )
    assert (
        classify_nonfinite_origin({"INPUT-A": 3, "GEMM1": 3, "ACT": 3, "GEMM2": 3})
        == "ENTERS-AT-INPUT-A"
    )


def test_all_clean_is_clean():
    assert (
        classify_nonfinite_origin(dict.fromkeys(STAGE_ORDER, 0)) == "CLEAN"
    )


def test_an_unmeasured_stage_is_not_a_clean_stage():
    # The #49 trap in one assertion: a scan that threw must never be reported
    # as evidence of absence.
    assert (
        classify_nonfinite_origin({"INPUT-A": 0, "GEMM1": 0}) == "UNMEASURED-AT-ACT"
    )
    assert classify_nonfinite_origin({}) == "UNMEASURED-AT-INPUT-A"
    assert (
        classify_nonfinite_origin({"GEMM1": 5}) == "UNMEASURED-AT-INPUT-A"
    ), "a bad later stage must not mask a missing earlier one"


# --- transport vs production ------------------------------------------------


def test_transport_when_every_bad_output_row_was_already_bad_on_input():
    assert classify_transport(n_bad_in=5, n_bad_out=5, n_bad_out_also_in=5) == "TRANSPORT"
    assert classify_transport(n_bad_in=9, n_bad_out=5, n_bad_out_also_in=5) == "TRANSPORT"


def test_produced_here_when_the_input_was_clean():
    assert classify_transport(0, 806, 0) == "PRODUCED-HERE"


def test_produced_here_despite_bad_input_is_its_own_name():
    assert classify_transport(3, 5, 0) == "PRODUCED-HERE-DESPITE-BAD-INPUT"


def test_mixed_is_not_rounded_to_either_side():
    assert classify_transport(4, 6, 2) == "MIXED"


def test_clean_output_is_clean_whatever_the_input():
    assert classify_transport(7, 0, 0) == "CLEAN"


def test_impossible_overlap_raises():
    with pytest.raises(ValueError):
        classify_transport(1, 2, 3)
    with pytest.raises(ValueError):
        classify_transport(1, 5, 2)


# --- the row scan -----------------------------------------------------------


def test_bad_rows_flags_the_row_not_the_element():
    t = torch.tensor([[1.0, 2.0], [float("nan"), 1.0], [3.0, float("inf")]])
    m = _bad_rows(t)
    assert m.tolist() == [False, True, True]
    assert _n(m) == 2
    assert _first(m) == [1, 2]


def test_bad_rows_handles_a_3d_stage_tensor():
    t = torch.zeros(4, 3, 2)
    t[2, 1, 0] = float("nan")
    assert _bad_rows(t).tolist() == [False, False, True, False]


def test_absent_tensor_is_none_not_zero():
    assert _bad_rows(None) is None
    assert _n(None) == 0
    assert _first(None) is None


def test_first_bad_is_capped_but_ordered():
    m = torch.zeros(400, dtype=torch.bool)
    for i in (6, 7, 8, 10, 180, 181, 182, 184, 300):
        m[i] = True
    # The fn8c2 shape: runs WITH HOLES (9 and 183 are fine). A warp-fragment
    # corruption does not leave holes; this is what the report must show.
    assert _first(m) == [6, 7, 8, 10, 180, 181, 182, 184]


# --- the scale check --------------------------------------------------------


def test_scales_finite_ignores_integer_tensors_and_none():
    assert _weights_scales_finite(None, torch.tensor([1, 2], dtype=torch.int32)) is True


def test_scales_finite_catches_a_nan_scale():
    assert _weights_scales_finite(torch.tensor([1.0, float("nan")])) is False
    assert _weights_scales_finite(torch.tensor([1.0, 2.0])) is True


# --- the report -------------------------------------------------------------


def test_report_is_silent_when_everything_is_clean(caplog):
    masks = {k: torch.zeros(8, dtype=torch.bool) for k in STAGE_ORDER}
    with caplog.at_level("ERROR"):
        fmm._stage_probe_report(masks, True, 8, 1, 64)
    assert "[nan-probe-in]" not in caplog.text


def test_report_names_the_input_when_the_call_was_handed_the_nan(caplog):
    bad = torch.zeros(8, dtype=torch.bool)
    bad[3] = True
    masks = {k: bad.clone() for k in STAGE_ORDER}
    with caplog.at_level("ERROR"):
        fmm._stage_probe_report(masks, True, 8, 1, 64)
    assert "ORIGIN ENTERS-AT-INPUT-A" in caplog.text
    assert "'GEMM2': 'TRANSPORT'" in caplog.text


def test_report_names_the_kernel_when_the_input_was_clean(caplog):
    clean = torch.zeros(8, dtype=torch.bool)
    bad = torch.zeros(8, dtype=torch.bool)
    bad[5] = True
    masks = {"INPUT-A": clean, "GEMM1": bad, "ACT": bad, "GEMM2": bad}
    with caplog.at_level("ERROR"):
        fmm._stage_probe_report(masks, True, 8, 1, 64)
    assert "ORIGIN ENTERS-AT-GEMM1" in caplog.text
    assert "'GEMM1': 'PRODUCED-HERE'" in caplog.text


def test_report_budget_stops_the_residual_flood(caplog):
    bad = torch.ones(4, dtype=torch.bool)
    masks = {k: bad.clone() for k in STAGE_ORDER}
    with caplog.at_level("ERROR"):
        for _ in range(fmm._STAGE_LOG_BUDGET + 10):
            fmm._stage_probe_report(masks, True, 4, 10, 64)
    # fn8c2 emitted the same downstream finding 1290 times; the budget is what
    # keeps the interesting first calls readable.
    assert caplog.text.count("[nan-probe-in] ORIGIN") == fmm._STAGE_LOG_BUDGET


def test_report_survives_a_broken_mask(caplog):
    with caplog.at_level("ERROR"):
        fmm._stage_probe_report({"INPUT-A": "not a tensor"}, True, 8, 1, 64)
    assert "[nan-probe-in] ORIGIN" not in caplog.text
