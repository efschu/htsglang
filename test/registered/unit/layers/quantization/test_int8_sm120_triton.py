"""SGLANG_INT8_SM120_TRITON (27b-int8tri 26.09.): the Triton INT8 W8A8
small-M GEMM for sm_120 behind a switch.

Desk-only (no GPU, CUDA_VISIBLE_DEVICES=""). Pins:
  (a) the table is exactly the documented rule applied to the measured
      sweep.json, and its lookup (M buckets, unknown shapes);
  (b) the dispatch: sm_120 only (monkeypatched capability), M <= 16,
      bias-free bf16, layout, one armed line, one line per decision;
  (c) apply_weights: switch off -> the sgl call with the same arguments and
      nothing else; switch on -> Triton result or the same sgl call;
  (d) numerics in the Triton CPU interpreter against an int64 reference
      (exact accumulator; the interpreter truncates fp32->bf16 instead of
      RTNE, so a full-range output must equal exactly one of RTNE/RTZ);
  (e) no host sync in the dispatch/launch source (CUDA-graph capture).
"""

import inspect
import json
import logging
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

from sglang.srt.layers.quantization import int8_sm120_triton as tri
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as w8a8,
)

SWEEP = Path("/spinning/evidence-665-f1/int8_mm_sweep_20260926T171511Z/sweep.json")
HIDDEN = 5120


def _gate_up(u):
    return (32 * u, HIDDEN)


def _down(u):
    return (HIDDEN, 16 * u)


# ------------------------------------------------------------------ (a) table
@pytest.mark.skipif(not SWEEP.exists(), reason="evidence sweep.json not on this box")
def test_table_is_the_rule_applied_to_the_measured_sweep():
    sweep = json.loads(SWEEP.read_text())
    assert tri.derive_table(sweep) == tri.SM120_TABLE
    assert sweep["device"]["name"].endswith("RTX 5090")
    # Every entry is bit-identical to sgl in the sweep (the rule demands it).
    checked = 0
    for cell in sweep["cells"]:
        if cell["phase"] == 1:  # phase 1 = sgl only (wave test)
            continue
        for m_str, mv in cell["m"].items():
            cfg = tri.SM120_TABLE.get((cell["N"], cell["K"]), {}).get(int(m_str))
            if cfg is not None:
                assert mv["lanes"][tri.cfg_name(cfg)]["bitwise_eq_vs_sgl"] == 1.0
                checked += 1
    assert checked == sum(len(row) for row in tri.SM120_TABLE.values()) == 31


def test_table_rule_rejects_overlap_and_non_bitwise_lanes():
    def lane(us_all, eq=1.0):
        return {
            "us": sorted(us_all)[len(us_all) // 2],
            "us_all": us_all,
            "bitwise_eq_vs_sgl": eq,
        }

    sweep = {
        "cells": [
            {
                "N": 5120,
                "K": 10432,
                "m": {
                    "8": {
                        "lanes": {
                            "sgl": lane([10.0, 10.1, 10.2]),
                            "tri_n32_k256_s1_w4_st3": lane(
                                [9.0, 9.1, 10.05]
                            ),  # overlaps
                            "tri_n64_k256_s4_w4_st3": lane([9.5, 9.6, 9.7]),
                            "tri_n32_k128_s4_w4_st4": lane([8.0, 8.1, 8.2], eq=0.99),
                        }
                    },
                    "16": {
                        "lanes": {
                            "sgl": lane([10.0, 10.1, 10.2]),
                            "tri_n32_k256_s1_w4_st3": lane([10.0, 10.1, 10.2]),
                        }
                    },
                },
            }
        ]
    }
    assert tri.derive_table(sweep) == {(5120, 10432): {8: (64, 256, 4, 4, 3)}}


def test_table_covers_the_drq_5090_shapes_and_the_wave_edge():
    # drq = 652:218:218 -> the 5090 holds u = 652.
    assert tri.lookup(*_gate_up(652), 8) == (64, 256, 1, 4, 3)
    assert tri.lookup(*_gate_up(652), 16) is None  # measured: no separated lane
    assert tri.lookup(*_down(652), 8) == (64, 256, 4, 4, 3)
    assert tri.lookup(*_down(652), 16) == (32, 256, 4, 4, 3)
    # Above the wave edge (684+) gate_up has an entry for both buckets.
    for u in (684, 704, 707):
        assert tri.lookup(*_gate_up(u), 8) is not None
        assert tri.lookup(*_gate_up(u), 16) is not None
    # down 680: sgl was best / not separated -> no entry at all.
    assert (HIDDEN, 16 * 680) not in tri.SM120_TABLE


def test_lookup_buckets_and_unknown_shapes():
    n, k = _down(707)
    for m in range(1, 9):
        assert tri.lookup(n, k, m) == tri.SM120_TABLE[(n, k)][8]
    for m in range(9, 17):
        assert tri.lookup(n, k, m) == tri.SM120_TABLE[(n, k)][16]
    assert tri.lookup(n, k, 0) is None
    assert tri.lookup(n, k, 17) is None
    assert tri.lookup(n, k, 4096) is None
    # 3080 shards (218 units) and an unmeasured 5090 u are not in the table.
    assert tri.lookup(*_gate_up(218), 8) is None
    assert tri.lookup(*_down(218), 8) is None
    assert tri.lookup(*_gate_up(660), 8) is None


def test_every_config_splits_k_without_an_empty_split():
    for (n, k), row in tri.SM120_TABLE.items():
        for cfg in row.values():
            bn, bk, s, nw, ns = cfg
            kps = tri.k_per_split(k, bk, s)
            assert kps % bk == 0
            assert kps * s >= k and kps * (s - 1) < k, (n, k, cfg)
            assert bn in (32, 64, 128) and bk in (128, 256) and s in (1, 2, 4)


# --------------------------------------------------------------- (b) dispatch
class _Launch:
    def __init__(self):
        self.calls = []

    def __call__(self, x_q, weight, x_scale, weight_scale, out, cfg):
        self.calls.append((x_q, weight, x_scale, weight_scale, out, cfg))
        return out


@pytest.fixture
def dispatch(monkeypatch):
    tri._reset_for_tests()
    launch = _Launch()
    state = {"cap": (12, 0)}
    monkeypatch.setattr(tri, "_on_cuda", lambda t: True)
    monkeypatch.setattr(tri, "_device_capability", lambda d: state["cap"])
    monkeypatch.setattr(tri, "launch", launch)
    yield types.SimpleNamespace(launch=launch, state=state)
    tri._reset_for_tests()


def _operands(m, n, k, *, col_major=True):
    x_q = torch.zeros(m, k, dtype=torch.int8)
    w = (
        torch.zeros(n, k, dtype=torch.int8).t()
        if col_major
        else torch.zeros(k, n, dtype=torch.int8)
    )
    x_scale = torch.ones(m, 1, dtype=torch.float32)
    w_scale = torch.ones(n, 1, dtype=torch.float32)
    return x_q, w, x_scale, w_scale


def test_dispatch_sm120_in_table_launches_the_table_config(dispatch, caplog):
    n, k = _down(707)
    ops = _operands(8, n, k)
    with caplog.at_level(logging.INFO, logger=tri.logger.name):
        y = tri.maybe_int8_scaled_mm(*ops, torch.bfloat16, None)
        y2 = tri.maybe_int8_scaled_mm(*_operands(8, n, k), torch.bfloat16, None)
    assert y is not None and y2 is not None
    assert y.shape == (8, n) and y.dtype == torch.bfloat16
    assert [c[-1] for c in dispatch.launch.calls] == [tri.SM120_TABLE[(n, k)][8]] * 2
    assert dispatch.launch.calls[0][:4] == ops
    text = caplog.text
    assert text.count("INT8-SM120-TRITON armed") == 1
    assert text.count(f"INT8-SM120-TRITON use N={n} K={k} M<=8") == 1
    assert tri.counters()["triton"] == 2


@pytest.mark.parametrize("cap", [(8, 6), (8, 9), (9, 0), (10, 0), (12, 1)])
def test_dispatch_other_devices_stay_on_sgl(dispatch, caplog, cap):
    dispatch.state["cap"] = cap
    n, k = _down(707)
    with caplog.at_level(logging.INFO, logger=tri.logger.name):
        assert (
            tri.maybe_int8_scaled_mm(*_operands(8, n, k), torch.bfloat16, None) is None
        )
        assert (
            tri.maybe_int8_scaled_mm(*_operands(8, n, k), torch.bfloat16, None) is None
        )
    assert dispatch.launch.calls == []
    assert caplog.text.count("INT8-SM120-TRITON inactive") == 1
    assert "INT8-SM120-TRITON armed" not in caplog.text
    assert tri.counters() == {"triton": 0, "sgl_fallback": 0, "inactive": 2}


def test_dispatch_falls_back_outside_the_measured_scope(dispatch):
    n, k = _down(707)
    bias = torch.zeros(n, dtype=torch.bfloat16)
    cases = [
        (_operands(8, n, k), torch.bfloat16, bias),  # bias
        (_operands(8, n, k), torch.float16, None),  # fp16 out
        (_operands(17, n, k), torch.bfloat16, None),  # M > 16
        (_operands(8, *_gate_up(660)), torch.bfloat16, None),  # unknown shape
        (_operands(8, n, k, col_major=False), torch.bfloat16, None),  # row-major B
        (_operands(8, *_down(684)), torch.bfloat16, None),  # bucket without entry
    ]
    for ops, dt, b in cases:
        assert tri.maybe_int8_scaled_mm(*ops, dt, b) is None
    x_q, w, xs, ws = _operands(8, n, k)
    assert (
        tri.maybe_int8_scaled_mm(
            x_q, w, xs.to(torch.bfloat16), ws, torch.bfloat16, None
        )
        is None
    )
    assert (
        tri.maybe_int8_scaled_mm(x_q, w, xs, ws[: n - 1], torch.bfloat16, None) is None
    )
    assert dispatch.launch.calls == []
    assert tri.counters()["sgl_fallback"] == len(cases) + 2


def test_dispatch_skips_non_cuda_tensors(monkeypatch):
    tri._reset_for_tests()
    monkeypatch.setattr(
        tri,
        "_device_capability",
        lambda d: pytest.fail("capability read for a CPU tensor"),
    )
    assert (
        tri.maybe_int8_scaled_mm(*_operands(8, *_down(707)), torch.bfloat16, None)
        is None
    )


# ---------------------------------------------------------- (c) apply_weights
class _Recorder:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _scheme_and_layer(monkeypatch, n, k, m):
    x = torch.zeros(m, k, dtype=torch.bfloat16)
    x_q = torch.zeros(m, k, dtype=torch.int8)
    x_scale = torch.ones(m, 1, dtype=torch.float32)
    layer = types.SimpleNamespace(
        weight=torch.zeros(n, k, dtype=torch.int8).t(),
        weight_scale=torch.ones(n, 1, dtype=torch.float32),
    )
    monkeypatch.setattr(w8a8, "per_token_quant_int8", lambda t: (x_q, x_scale))
    sgl_out = torch.full((m, n), 3.0, dtype=torch.bfloat16)
    sgl = _Recorder(sgl_out)
    monkeypatch.setattr(w8a8, "int8_scaled_mm", sgl, raising=False)
    scheme = object.__new__(w8a8.CompressedTensorsW8A8Int8)
    return scheme, layer, x, x_q, x_scale, sgl, sgl_out


class _Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"switch off touched int8_sm120_triton.{name}")


def _expected_sgl_call(layer, x, x_q, x_scale, bias):
    return (
        (x_q, layer.weight, x_scale, layer.weight_scale),
        {"out_dtype": x.dtype, "bias": bias},
    )


def _same_call(got, want):
    (ga, gk), (wa, wk) = got, want
    return (
        len(ga) == len(wa)
        and all(a is b for a, b in zip(ga, wa))
        and gk.keys() == wk.keys()
        and all(gk[key] is wk[key] for key in gk)
    )


def test_switch_off_is_the_sgl_call_only(monkeypatch):
    n, k = _down(707)
    scheme, layer, x, x_q, x_scale, sgl, sgl_out = _scheme_and_layer(
        monkeypatch, n, k, 8
    )
    monkeypatch.setattr(w8a8, "_int8_sm120_triton", False)
    monkeypatch.setattr(w8a8, "_i8tri", _Forbidden(), raising=False)
    for bias in (None, torch.zeros(n, dtype=torch.bfloat16)):
        sgl.calls.clear()
        y = scheme.apply_weights(layer, x, bias)
        assert y is sgl_out
        assert len(sgl.calls) == 1
        assert _same_call(
            sgl.calls[0], _expected_sgl_call(layer, x, x_q, x_scale, bias)
        )


def test_switch_on_uses_triton_or_the_identical_sgl_call(monkeypatch):
    n, k = _down(707)
    scheme, layer, x, x_q, x_scale, sgl, sgl_out = _scheme_and_layer(
        monkeypatch, n, k, 8
    )
    tri_out = torch.full((8, n), 7.0, dtype=torch.bfloat16)
    decide = {"y": tri_out}
    seen = []

    def fake_maybe(*args):
        seen.append(args)
        return decide["y"]

    monkeypatch.setattr(w8a8, "_int8_sm120_triton", True)
    monkeypatch.setattr(
        w8a8,
        "_i8tri",
        types.SimpleNamespace(maybe_int8_scaled_mm=fake_maybe),
        raising=False,
    )
    assert scheme.apply_weights(layer, x, None) is tri_out
    assert sgl.calls == []
    assert all(
        a is b
        for a, b in zip(seen[0][:4], (x_q, layer.weight, x_scale, layer.weight_scale))
    )
    assert seen[0][4] is x.dtype and seen[0][5] is None
    decide["y"] = None
    assert scheme.apply_weights(layer, x, None) is sgl_out
    assert len(sgl.calls) == 1
    assert _same_call(sgl.calls[0], _expected_sgl_call(layer, x, x_q, x_scale, None))


def test_switch_resolution(monkeypatch):
    monkeypatch.delenv("SGLANG_INT8_SM120_TRITON", raising=False)
    monkeypatch.setattr(w8a8, "_is_cuda", True)
    assert w8a8._resolve_int8_sm120_triton() is False
    monkeypatch.setenv("SGLANG_INT8_SM120_TRITON", "1")
    assert w8a8._resolve_int8_sm120_triton() is True
    monkeypatch.setattr(w8a8, "_is_cuda", False)
    assert w8a8._resolve_int8_sm120_triton() is False
    monkeypatch.setattr(w8a8, "_is_cuda", True)
    monkeypatch.setenv("SGLANG_INT8_SM120_TRITON", "0")
    assert w8a8._resolve_int8_sm120_triton() is False


# ------------------------------------------------------ (d) interpreter numerics
_INTERP = textwrap.dedent("""
    import sys, torch
    from sglang.srt.layers.quantization import int8_sm120_triton as tri
    g = torch.Generator().manual_seed(1234)
    bad = []

    def operands(m, n, k, lo, hi):
        x = torch.randint(lo, hi, (m, k), generator=g, dtype=torch.int8)
        w = torch.randint(lo, hi, (n, k), generator=g, dtype=torch.int8).t()
        return x, w

    # 1) exact accumulator: |acc| <= 2*2*K <= 256 is exact in bf16 whatever the
    #    rounding, and unit scales -> out == int64 reference exactly.
    for (m, n, k) in [(8, 48, 40), (16, 80, 72), (5, 33, 64), (1, 16, 16)]:
        x, w = operands(m, n, k, -2, 3)
        ref = (x.long() @ w.long()).to(torch.bfloat16)
        ones_m, ones_n = torch.ones(m, 1), torch.ones(n, 1)
        for cfg in [(32, 16, 1, 4, 1), (32, 16, 2, 4, 1), (16, 16, 3, 4, 2),
                    (64, 32, 4, 4, 1), (16, 16, 5, 4, 1)]:
            out = torch.empty(m, n, dtype=torch.bfloat16)
            tri.launch(x, w, ones_m, ones_n, out, cfg)
            if not torch.equal(out, ref):
                bad.append(("exact", m, n, k, cfg))

    # 2) full range, table configs, awkward N/K: acc * (sb * sa) rounded; the
    #    interpreter truncates (RTZ), the GPU emits cvt.rn (RTNE) -- exactly
    #    one of the two, never a mix.
    cfgs = sorted({c for row in tri.SM120_TABLE.values() for c in row.values()})
    for (m, n, k) in [(8, 200, 1000), (16, 136, 520), (3, 72, 272)]:
        x, w = operands(m, n, k, -128, 128)
        sa = torch.rand(m, 1, generator=g) * 0.02 + 1e-3
        sb = torch.rand(n, 1, generator=g) * 0.02 + 1e-3
        f32 = (x.long() @ w.long()).float() * (sb.view(1, -1) * sa.view(-1, 1))
        rtne = f32.to(torch.bfloat16)
        rtz = (f32.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
        for cfg in cfgs:
            out = torch.empty(m, n, dtype=torch.bfloat16)
            tri.launch(x, w, sa, sb, out, cfg)
            if not (torch.equal(out, rtne) or torch.equal(out, rtz)):
                bad.append(("round", m, n, k, cfg))
            # 3) the epilogue value itself, before any bf16 rounding: an fp32
            #    output must be float(acc) * (sb * sa) bit for bit (CUTLASS's
            #    association order; the GPU sweep measured bf16 bit-equality
            #    to sgl with exactly this order).
            out32 = torch.empty(m, n, dtype=torch.float32)
            tri.launch(x, w, sa, sb, out32, cfg)
            if not torch.equal(out32, f32):
                bad.append(("fp32-epilogue", m, n, k, cfg))
    print("CFGS", len(cfgs), "BAD", bad)
    sys.exit(1 if bad else 0)
    """)


def test_interpreter_numerics_exact_int32_split_k():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-c", _INTERP],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "BAD []" in r.stdout


# ----------------------------------------------------- (e) graph-capture safety
def test_no_host_sync_in_the_dispatch_path():
    src = inspect.getsource(tri.maybe_int8_scaled_mm) + inspect.getsource(tri.launch)
    for word in (
        ".item(",
        ".cpu(",
        ".tolist(",
        "synchronize",
        ".numpy(",
        "autotune",
        "nonzero(",
    ):
        assert word not in src, word
