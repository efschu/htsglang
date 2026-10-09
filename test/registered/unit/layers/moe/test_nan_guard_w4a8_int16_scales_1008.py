"""H88 SCALEFIX follow-up 1008: the fetch nan-guard on W4A8 (int16-container) scales.

``CompressedTensorsWNA16A8MoE`` keeps ``w13_weight_scale`` / ``w2_weight_scale`` as int16 BIT PATTERNS in the
model-dtype container (NF serves bf16). The AutoRound group scales are signed (about half negative), and a
negative int16 in [-128, -1] is an -inf/NaN bit pattern in bf16 ([-1024, -1] in fp16). The old
``MoEExpertOffloadCache._nan_guard_fetched`` ran ``torch.isfinite`` over every float resident tensor and so
raised a false "[nan-guard] fetched slot(s) with non-finite ..." for valid W4A8 scales. Fixed: for a scale
attribute whose layer carries the int16 factor (``moe_w4a8_layout.FACTOR_OF_SCALE``), the guard checks the
int16 band |v| <= W4A8_SCALE_INT_RANGE (4096) instead; every other tensor / layer is checked as before.

CPU only: the method is called unbound on a SimpleNamespace stand-in for the cache.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import nan_guard
from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

LOGGER = "sglang.srt.layers.moe.expert_offload"
S, G, N = 4, 2, 8


def _int16_as_bf16(rows):
    return torch.tensor(rows, dtype=torch.int16).view(torch.bfloat16)


def _valid_signed_rows():
    # every slot carries negative int16 values that are NaN / -inf bit patterns in bf16, plus the band edges
    base = [[[-1, -100, -128, 4096, -4096, 7, -7, 1000] for _ in range(G)] for _ in range(S)]
    return base


@pytest.fixture
def guard_env(monkeypatch):
    def arm(on: bool):
        v = "1" if on else "0"
        monkeypatch.setenv("SGLANG_NAN_GUARD", v)
        monkeypatch.setenv("SGLANG_NAN_GUARD_FETCH", v)
        nan_guard._STATE["on"] = None

    yield arm
    nan_guard._STATE["on"] = None


def _fake(resident, **layer_attrs):
    return SimpleNamespace(_resident=resident, layer=SimpleNamespace(layer_id=23, **layer_attrs))


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno >= logging.ERROR]


FETCH_PLAN = [(130, 0), (131, 1), (140, 2), (151, 3)]


def test_a_valid_negative_int16_scales_raise_no_alarm(guard_env, caplog):
    guard_env(True)
    dst = _int16_as_bf16(_valid_signed_rows())
    assert not bool(torch.isfinite(dst.float()).all())  # the old check WOULD see non-finite values here
    fake = _fake({"w13_weight_scale": dst}, w13_act_scale_factor=torch.tensor(1e-5))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        MoEExpertOffloadCache._nan_guard_fetched(fake, FETCH_PLAN)
    assert _errors(caplog) == []


def test_b_out_of_band_int16_names_attr_and_expert_slot(guard_env, caplog):
    guard_env(True)
    rows = _valid_signed_rows()
    rows[2][1][5] = -5000  # slot 2 = expert 140
    fake = _fake({"w13_weight_scale": _int16_as_bf16(rows)}, w13_act_scale_factor=torch.tensor(1e-5))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        MoEExpertOffloadCache._nan_guard_fetched(fake, FETCH_PLAN)
    errs = _errors(caplog)
    assert len(errs) == 1, errs
    assert "w13_weight_scale" in errs[0]
    assert "out of band" in errs[0]
    assert "(140, 2)" in errs[0]
    assert "1 of 4 slots" in errs[0]


def test_c_float_scale_layer_without_factor_still_flags_nan(guard_env, caplog):
    guard_env(True)
    dst = torch.ones(S, G, N, dtype=torch.bfloat16)
    dst[1] = float("nan")  # slot 1 = expert 131, the whole row non-finite
    # A16 layer: no factor attribute at all; channelwise W4A8 (G == 1): factor None -- both keep the old check
    for layer_attrs in ({}, {"w13_act_scale_factor": None}):
        caplog.clear()
        fake = _fake({"w13_weight_scale": dst}, **layer_attrs)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            MoEExpertOffloadCache._nan_guard_fetched(fake, FETCH_PLAN)
        errs = _errors(caplog)
        assert len(errs) == 1, (layer_attrs, errs)
        assert "non-finite w13_weight_scale" in errs[0]
        assert "(131, 1)" in errs[0]


def test_d_guard_off_logs_nothing(guard_env, caplog):
    guard_env(False)
    rows = _valid_signed_rows()
    rows[0][0][0] = 5000
    nanbuf = torch.full((S, G, N), float("nan"), dtype=torch.bfloat16)
    fake = _fake(
        {"w13_weight_scale": _int16_as_bf16(rows), "w2_weight_scale": nanbuf},
        w13_act_scale_factor=torch.tensor(1e-5),
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        MoEExpertOffloadCache._nan_guard_fetched(fake, FETCH_PLAN)
    assert [r for r in caplog.records if r.name == LOGGER] == []
