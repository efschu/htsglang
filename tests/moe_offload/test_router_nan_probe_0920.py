"""Task #49 after fn8c8b: the router-level probe and the renorm guard.

fn8c8b's verdict was ENTERS-AT-ROUTER-W with bad_gemm2 == bad_routerw on the
same rows -- the Marlin kernel carries a NaN it was handed. These tests pin the
next question shut: WHICH half of the router made it, and what the repair would
change. CPU-only.
"""

import pytest
import torch

import sglang.srt.layers.moe.expert_offload as eo
from sglang.srt.layers.moe.router_nan_probe import (
    ENV_RENORM_GUARD,
    RENORM_EPS,
    ROUTER_LEVELS,
    classify_router_origin,
    guarded_denominator,
    renorm_guard_on,
    renorm_rows_that_would_nan,
)


# --- the three-level classifier --------------------------------------------


def test_gate_logits_win_over_everything_downstream():
    assert (
        classify_router_origin(
            {"GATE-LOGITS": 2, "TOPK-WEIGHTS": 9, "GATHERED": 9}
        )
        == "ENTERS-AT-GATE-LOGITS"
    )


def test_finite_logits_plus_bad_weights_is_the_renorm_case():
    """The fn8c8b hypothesis, as a name."""
    assert (
        classify_router_origin(
            {"GATE-LOGITS": 0, "TOPK-WEIGHTS": 383, "GATHERED": 383}
        )
        == "ENTERS-AT-TOPK-WEIGHTS"
    )


def test_only_the_gather_dirty_would_be_an_index_defect():
    assert (
        classify_router_origin({"GATE-LOGITS": 0, "TOPK-WEIGHTS": 0, "GATHERED": 5})
        == "ENTERS-AT-GATHERED"
    )


def test_clean_is_clean():
    assert classify_router_origin(dict.fromkeys(ROUTER_LEVELS, 0)) == "CLEAN"


def test_an_unmeasured_level_is_never_read_as_clean():
    assert (
        classify_router_origin({"GATE-LOGITS": 0, "GATHERED": 3})
        == "UNMEASURED-AT-TOPK-WEIGHTS"
    )
    assert classify_router_origin({}) == "UNMEASURED-AT-GATE-LOGITS"


# --- the defect itself, reproduced in three lines of torch ------------------


def test_the_unguarded_renorm_nans_a_whole_token_row():
    """Why the damage is a CONTIGUOUS BLOCK of pairs and not scattered weights:
    one zero row destroys all K of that token's weights at once."""
    w = torch.tensor([[0.4, 0.6], [0.0, 0.0], [0.3, 0.7]])
    out = w / w.sum(dim=-1, keepdim=True)
    bad = ~torch.isfinite(out).all(dim=1)
    assert bad.tolist() == [False, True, False]
    assert torch.isnan(out[1]).all(), "ALL K weights of that token, not one"


def test_the_guard_removes_exactly_those_rows_and_nothing_else():
    w = torch.tensor([[0.4, 0.6], [0.0, 0.0], [0.3, 0.7]])
    den = w.sum(dim=-1, keepdim=True)
    guarded = w / guarded_denominator(den)
    assert torch.isfinite(guarded).all()
    # The rows that were already fine must come out BIT-IDENTICAL, or a guarded
    # boot is no longer comparable to an unguarded one.
    plain = w / den
    assert torch.equal(guarded[0], plain[0])
    assert torch.equal(guarded[2], plain[2])
    assert torch.equal(guarded[1], torch.zeros(2))


def test_the_guard_counts_how_often_it_fired():
    den = torch.tensor([[1.0], [0.0], [2.0], [0.0]])
    assert renorm_rows_that_would_nan(den) == 2


def test_eps_is_below_any_real_routing_sum():
    """The clamp may only ever touch a genuinely zero/denormal sum."""
    assert RENORM_EPS < 1e-30
    den = torch.tensor([[1e-20], [1e-8], [1.0]])
    assert renorm_rows_that_would_nan(den) == 0
    assert torch.equal(guarded_denominator(den), den)


def test_guarded_denominator_works_on_a_plain_float():
    assert guarded_denominator(0.0) == RENORM_EPS
    assert guarded_denominator(2.0) == 2.0


# --- the switch -------------------------------------------------------------


def test_renorm_guard_is_opt_in():
    """On by default it would MASK the defect: a 0/0 token would silently be
    routed nowhere and the boot would look healthy."""
    assert renorm_guard_on({}) is False
    assert renorm_guard_on({ENV_RENORM_GUARD: "0"}) is False


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes"])
def test_renorm_guard_switches_on(raw):
    assert renorm_guard_on({ENV_RENORM_GUARD: raw}) is True


def test_topk_module_actually_uses_the_guard():
    """Both unguarded division sites must go through it, or the switch is a
    decoration."""
    import pathlib

    src = pathlib.Path(
        eo.__file__
    ).parent.joinpath("topk.py").read_text()
    assert src.count("renorm_guard_on()") == 2
    assert src.count("guarded_denominator(") == 2
    assert "topk_weights / topk_weights.sum(dim=-1, keepdim=True)" not in src


# --- the probe --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    import sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe as fmm

    fmm._PROBE_LEVEL["n"] = None
    fmm._C_SENTINEL["on"] = None
    eo._ROUTER_PROBE_LOGGED["n"] = 0
    monkeypatch.setattr(fmm, "_is_cuda", False)
    yield
    fmm._PROBE_LEVEL["n"] = None
    fmm._C_SENTINEL["on"] = None
    eo._ROUTER_PROBE_LOGGED["n"] = 0


def test_probe_is_silent_below_level_2(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "1")
    w = torch.full((4, 2), float("nan"))
    with caplog.at_level("ERROR"):
        eo._router_probe(None, w, w.reshape(-1, 1), 4, 2)
    assert "[nan-probe-rw]" not in caplog.text


def test_probe_names_the_renorm_case(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "2")
    logits = torch.zeros(4, 8)
    w = torch.tensor([[0.5, 0.5], [float("nan")] * 2, [0.5, 0.5], [0.5, 0.5]])
    with caplog.at_level("ERROR"):
        eo._router_probe(logits, w, w.reshape(-1, 1), 4, 2)
    assert "ROUTER-ORIGIN ENTERS-AT-TOPK-WEIGHTS" in caplog.text
    # K of K weights gone for that one token -- the 0/0 signature.
    assert "bad_weight_tokens=1" in caplog.text
    assert "bad_flat_pairs=2" in caplog.text
    assert "pairs_per_bad_token=2.00" in caplog.text


def test_probe_names_the_gate_when_the_logits_are_already_gone(
    monkeypatch, caplog
):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "2")
    logits = torch.zeros(3, 8)
    logits[1, 0] = float("inf")
    w = torch.tensor([[0.5, 0.5], [float("nan")] * 2, [0.5, 0.5]])
    with caplog.at_level("ERROR"):
        eo._router_probe(logits, w, w.reshape(-1, 1), 3, 2)
    assert "ROUTER-ORIGIN ENTERS-AT-GATE-LOGITS" in caplog.text


def test_probe_is_silent_when_everything_is_finite(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "2")
    w = torch.full((4, 2), 0.5)
    with caplog.at_level("ERROR"):
        eo._router_probe(torch.zeros(4, 8), w, w.reshape(-1, 1), 4, 2)
    assert "[nan-probe-rw]" not in caplog.text


def test_probe_respects_its_budget(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "2")
    w = torch.full((2, 2), float("nan"))
    with caplog.at_level("ERROR"):
        for _ in range(eo._ROUTER_PROBE_BUDGET + 5):
            eo._router_probe(None, w, w.reshape(-1, 1), 2, 2)
    assert caplog.text.count("ROUTER-ORIGIN") == eo._ROUTER_PROBE_BUDGET


def test_probe_survives_junk(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_MOE_MARLIN_C_SENTINEL", "2")
    with caplog.at_level("ERROR"):
        eo._router_probe(None, "not a tensor", None, 4, 2)
    assert "ROUTER-ORIGIN" not in caplog.text
