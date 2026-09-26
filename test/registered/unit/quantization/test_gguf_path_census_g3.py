"""#GGUFPATH -- the G3 decision instrument (27B line, 2026-09-25), on the desk.

What the first GGUF boot must answer: how much of a D verify round goes to MMVQ,
MMQ and dequant+cuBLAS, per ggml type, per card, per M bucket (bs1 verify = M 8,
bs2 = M 16). These tests drive the REAL dispatch (``gguf.fused_mul_mat_gguf``)
with its kernels stubbed on CPU and a fake device backend in place of the two
Triton kernels, and check: the branch each call is attributed to is the one the
dispatch took; off (the default) leaves the dispatch untouched; the host never
waits (a queued snapshot is read only once ready); the window line.

RED on RC4 060d97dcdd: the census module and the dispatch's path report do not
exist.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.quantization import gguf as G  # noqa: E402
from sglang.srt.layers.quantization import gguf_path_census as C  # noqa: E402

IQ4_XS, Q5_K, Q8_0, F16 = 23, 13, 8, 1


class FakeOps:
    """The device half on the host: a fake clock advances 1000 ns per stamp
    pair, the snapshot becomes ready only when the test says so."""

    def __init__(self):
        self.now = 0
        self.ready = False
        self.stamps = 0
        self.accums = 0
        self.snapshots = 0

    def alloc(self, device):
        return torch.zeros((C.MAX_KEYS, 3), dtype=torch.int64), torch.zeros(
            (C.MAX_KEYS,), dtype=torch.int64
        )

    def stamp(self, ts, slot):
        self.stamps += 1
        ts[slot] = self.now

    def accum(self, ts, acc, slot, nbytes):
        self.accums += 1
        self.now += 1000
        acc[slot, 0] += self.now - ts[slot]
        acc[slot, 1] += nbytes
        acc[slot, 2] += 1

    def snapshot(self, acc):
        self.snapshots += 1
        host = acc.clone()
        return host, (lambda: self.ready)

    def capturing(self):
        return False


@pytest.fixture
def stub_kernels(monkeypatch):
    """The four kernels as CPU stand-ins; the dispatch logic itself is real."""

    def out(x, qweight):
        return torch.zeros(x.shape[0], qweight.shape[0])

    # (imported only on a CUDA build -- raising=False lets the CPU desk add them)
    monkeypatch.setattr(
        G, "ggml_mul_mat_vec_a8", lambda w, x, t, n: out(x, w), raising=False
    )
    monkeypatch.setattr(
        G, "ggml_mul_mat_a8", lambda w, x, t, n: out(x, w), raising=False
    )
    monkeypatch.setattr(
        G, "_ggml_dequantize_ws", lambda w, t, n, k, dt: torch.zeros(n, k)
    )
    monkeypatch.setattr(G, "_mmvq_safe_for_device", lambda: 2)
    monkeypatch.setattr(G, "_is_cuda", False)
    yield


@pytest.fixture
def census(stub_kernels):
    c = C.PathCensus(rounds=2, ops=FakeOps())
    C.reset_for_tests(c)
    yield c
    C.reset_for_tests(None)


def _w(qtype, n, k_bytes, dtype=torch.uint8):
    return torch.zeros(n, k_bytes, dtype=dtype)


def _x(m, k=256):
    return torch.zeros(m, k)


def _attributed(c):
    return {c._key_of[s]: c._label[s] for s in c._label}


def test_every_branch_is_attributed_to_the_path_the_dispatch_took(census):
    """RED on RC4: no census, no path report."""
    #: one superblock per row (K = 256): IQ4_XS 136 B, Q5_K 176 B, Q8_0 8 x 34 B
    row = {IQ4_XS: 136, Q5_K: 176, Q8_0: 272}
    calls = [
        (IQ4_XS, 8, 6144),  # i-quant, N > 5120, M <= 8   -> MMVQ
        (IQ4_XS, 16, 6144),  # i-quant, N > 5120, M = 16   -> DEQ (the G3 cliff)
        (IQ4_XS, 16, 5120),  # i-quant, N <= 5120, M = 16  -> MMVQ
        (Q5_K, 8, 4096),  # K-quant, M <= MMQ cap (8)   -> MMQ
        (Q5_K, 16, 4096),  # K-quant, M = 16 > cap       -> DEQ
        (Q8_0, 2, 4096),  # standard, M <= mmvq_safe    -> MMVQ
    ]
    for qtype, m, n in calls:
        G.fused_mul_mat_gguf(_x(m), _w(qtype, n, row[qtype]), qtype)
    G.fused_mul_mat_gguf(
        _x(16).to(torch.bfloat16), _w(F16, 4096, 256, torch.bfloat16), F16
    )
    got = _attributed(census)
    assert got[(IQ4_XS, 8, 6144, 136)] == "mmvq"
    assert got[(IQ4_XS, 16, 6144, 136)] == "deq"
    assert got[(IQ4_XS, 16, 5120, 136)] == "mmvq"
    assert got[(Q5_K, 8, 4096, 176)] == "mmq"
    assert got[(Q5_K, 16, 4096, 176)] == "deq"
    assert got[(Q8_0, 2, 4096, 272)] == "mmvq"
    assert got[(F16, 16, 4096, 256)] == "dense"
    assert census._ops.stamps == census._ops.accums == 7


def test_the_mmq_cap_moves_the_k_quants_off_dequant(census, monkeypatch):
    """(c) SGLANG_GGUF_MMQ_MAX_TOKENS=16, as the census sees it."""
    monkeypatch.setattr(G, "_MMQ_MAX_TOKENS", 16)
    G.fused_mul_mat_gguf(_x(16), _w(Q5_K, 4096, 176), Q5_K)
    G.fused_mul_mat_gguf(_x(16), _w(IQ4_XS, 6144, 136), IQ4_XS)
    got = _attributed(census)
    assert got[(Q5_K, 16, 4096, 176)] == "mmq"
    assert got[(IQ4_XS, 16, 6144, 136)] == "deq"  # no i-quant MMQ: the G3 gap


def test_off_is_the_default_and_touches_nothing(stub_kernels, monkeypatch):
    monkeypatch.delenv(C.ENV, raising=False)
    C.reset_for_tests(None)
    assert C.census() is None
    y = G.fused_mul_mat_gguf(_x(16), _w(IQ4_XS, 6144, 136), IQ4_XS)
    assert tuple(y.shape) == (16, 6144)
    C.reset_for_tests(None)


def test_prefill_shapes_and_empty_batches_are_not_measured(census):
    G.fused_mul_mat_gguf(_x(C.M_MAX + 1), _w(IQ4_XS, 6144, 136), IQ4_XS)
    G.fused_mul_mat_gguf(_x(0), _w(IQ4_XS, 6144, 136), IQ4_XS)
    assert census._ops.stamps == 0 and not census._label


def test_the_host_never_waits_for_a_window(census):
    """A queued snapshot is read only once ready; until then no line and no
    second snapshot -- and nothing ever synchronizes."""
    for _ in range(2):
        G.fused_mul_mat_gguf(_x(16), _w(IQ4_XS, 6144, 136), IQ4_XS)
        assert census.end_round(bs=2) is None
    assert census._ops.snapshots == 1
    for _ in range(3):
        assert census.end_round(bs=2) is None  # not ready: keep serving
    assert census._ops.snapshots == 1
    census._ops.ready = True
    lines = census.end_round(bs=1)
    assert lines and lines[0].startswith("#GGUFPATH rounds=2 bs={2: 2}")
    assert "deq=100.0%" in lines[0]
    assert any("m9-16 path=deq calls=2" in ln and "IQ4_XS:2/" in ln for ln in lines[1:])


def test_a_window_reports_deltas_not_totals(census):
    """One call per round, a window of 2: each window's line counts the calls
    between its two snapshots, never the cumulative total since boot."""
    census._ops.ready = True
    for _ in range(5):
        G.fused_mul_mat_gguf(_x(8), _w(IQ4_XS, 6144, 136), IQ4_XS)
        census.end_round(bs=1)
    per_path = [ln for ln in census.lines if "path=mmvq" in ln]
    assert len(per_path) == 2, census.lines
    assert all("calls=2 " in ln for ln in per_path), per_path


def test_a_malformed_switch_is_loud_and_off(monkeypatch, caplog):
    monkeypatch.setenv(C.ENV, "lots")
    C.reset_for_tests(None)
    assert C.census() is None
    assert "#GGUFPATH instrument OFF" in caplog.text
    C.reset_for_tests(None)


def test_a_first_call_inside_a_capture_stays_unmeasured(stub_kernels):
    ops = FakeOps()
    ops.capturing = lambda: True
    c = C.PathCensus(rounds=2, ops=ops)
    C.reset_for_tests(c)
    try:
        G.fused_mul_mat_gguf(_x(8), _w(IQ4_XS, 6144, 136), IQ4_XS)
        assert c._acc is None and c._unarmed == 1 and ops.stamps == 0
    finally:
        C.reset_for_tests(None)


def test_the_scheduler_closes_a_census_round_where_dgap_does():
    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S)
    i = src.index("_dgap.end_round(deferred=")
    j = src.index("_gpc.end_round(bs=batch.batch_size())")
    assert abs(j - i) < 600


def test_arm_allocates_once_before_capture_and_is_a_no_op_when_off(monkeypatch):
    c = C.PathCensus(rounds=2, ops=FakeOps())
    C.reset_for_tests(c)
    try:
        assert C.arm("cpu") is True and c._acc is not None
        acc = c._acc
        assert C.arm("cpu") is True and c._acc is acc  # idempotent
    finally:
        C.reset_for_tests(None)
    monkeypatch.delenv(C.ENV, raising=False)
    assert C.arm("cpu") is False
    C.reset_for_tests(None)


def test_the_model_runner_arms_before_any_graph_is_captured():
    from sglang.srt.model_executor import model_runner as MR

    src = inspect.getsource(MR)
    arm = src.index("_gguf_path_census_arm(self.device)")
    assert arm < src.index("self.init_prefill_cuda_graph()", arm - 2000)
    assert src.index("self.eager_runner = EagerRunner(self)") < arm
    assert 'if self.model_config.quantization == "gguf":' in src[arm - 600 : arm]
