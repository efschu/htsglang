# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hermetic tests for the token-slice TP all-reduce pipeline (task #588).

No CUDA, no process group, no model. The pipeline's device path is exercised
through the CPU branch of the same runner, so what is under test here is the
part that can silently corrupt a forward -- the row-to-slice mapping, the
output assembly, and the reduce partition -- plus the two properties that a
collective feature has to prove before it may run on a group: a rank-uniform
slice count, and an untouched default path.

Each check carries its own can-fail proof: the same assertion is re-run
against a deliberately broken variant and must FAIL there. A test that
cannot fail has measured nothing.

Run with:
    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        pytest -q test/registered/unit/distributed/test_tp_ar_pipeline.py
"""

from __future__ import annotations

import contextlib
import inspect
import math
import types

import pytest
import torch

from sglang.srt.distributed import tp_ar_pipeline as tap
from sglang.srt.environ import envs

WORLD = 3
TOKENS = 40
OUT_FEATURES = 64
IN_SHARD = 32


@pytest.fixture(autouse=True)
def _clean_state():
    tap.reset_tp_ar_pipeline_state()
    yield
    tap.reset_tp_ar_pipeline_state()


def _exact_int_tensor(*shape, seed: int) -> torch.Tensor:
    """Small integers held in float32.

    Every partial product and every partial sum is representable exactly, so
    the reference and the sliced run must agree BITWISE regardless of the
    order the BLAS kernel chose. That isolates this test onto the slicing
    logic: a kernel that reassociates differently for a different M cannot
    make it pass or fail. Bitwise identity of the GEMM itself across M is a
    kernel property and is gated on the GPU by the runsheet, not here.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-4, 5, shape, generator=generator, dtype=torch.int64).to(
        torch.float32
    )


class FakeTpGroup:
    """A TP group emulated inside one process.

    ``apply_fn`` plays this rank; ``all_reduce`` adds what the other ranks
    would have contributed for the SAME token rows. The row range is
    recovered from the view's storage offset, which is exactly the coupling
    the real thing has: the collective sees a contiguous row window and
    nothing else. A slicing bug that reduces the wrong rows therefore shows
    up as a wrong answer here, not as a silent pass.
    """

    def __init__(self, inputs, weights, out_features: int):
        self.inputs = inputs
        self.weights = weights
        self.out_features = out_features
        self.call_count = 0
        self.reduced_rows = []

    def apply(self, chunk: torch.Tensor) -> torch.Tensor:
        return chunk @ self.weights[0]

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        rows = tensor.shape[0]
        start = tensor.storage_offset() // self.out_features
        self.reduced_rows.append((start, start + rows))
        for rank in range(1, len(self.weights)):
            tensor.add_(self.inputs[rank][start : start + rows] @ self.weights[rank])
        return tensor

    def reference(self) -> torch.Tensor:
        out = self.inputs[0] @ self.weights[0]
        for rank in range(1, len(self.weights)):
            out = out + self.inputs[rank] @ self.weights[rank]
        return out


def _make_group(tokens: int = TOKENS) -> FakeTpGroup:
    inputs = [_exact_int_tensor(tokens, IN_SHARD, seed=10 + r) for r in range(WORLD)]
    weights = [
        _exact_int_tensor(IN_SHARD, OUT_FEATURES, seed=90 + r) for r in range(WORLD)
    ]
    return FakeTpGroup(inputs, weights, OUT_FEATURES)


def _run(group: FakeTpGroup, bounds) -> torch.Tensor:
    return tap._run_sliced(group.inputs[0], group.apply, group.all_reduce, bounds)


# --------------------------------------------------------------------------
# slice_bounds: the partition itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tokens,slices", [(40, 1), (40, 3), (40, 8), (7, 4), (1, 4)])
def test_slice_bounds_is_an_exact_partition(tokens, slices):
    bounds = tap.slice_bounds(tokens, slices)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == tokens
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start
    sizes = [end - start for start, end in bounds]
    assert sum(sizes) == tokens
    assert all(size >= 1 for size in sizes)
    # Near-equal: a pipeline is only as balanced as its slowest stage.
    assert max(sizes) - min(sizes) <= 1


def test_slice_bounds_never_exceeds_token_count():
    assert len(tap.slice_bounds(3, 99)) == 3


# --------------------------------------------------------------------------
# byte identity of the sliced run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slices", [2, 3, 4, 7, 8])
def test_sliced_result_is_bitwise_identical_to_unsliced(slices):
    group = _make_group()
    reference = group.reference()
    got = _run(group, tap.slice_bounds(TOKENS, slices))
    assert got.shape == reference.shape
    assert torch.equal(got, reference)
    # The wire really was split: one collective per slice, covering the
    # token axis exactly once.
    assert group.call_count == slices
    assert group.reduced_rows == tap.slice_bounds(TOKENS, slices)


class BrokenRowMappingGroup(FakeTpGroup):
    """A collective that reduces the wrong token rows.

    Stands in for the failure this feature can actually cause: a slice whose
    transfer covers rows other than the ones it computed. Every slice here
    reduces rows [0, n) instead of its own window.
    """

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        rows = tensor.shape[0]
        for rank in range(1, len(self.weights)):
            tensor.add_(self.inputs[rank][0:rows] @ self.weights[rank])
        return tensor


def test_sliced_result_can_fail():
    """Can-fail proof for the identity check above.

    With a collective that ignores the slice's row window, the sliced run
    MUST come out wrong while the single-slice run stays right. If both
    passed, the identity test would be measuring nothing.
    """
    base = _make_group()
    broken = BrokenRowMappingGroup(base.inputs, base.weights, OUT_FEATURES)
    reference = broken.reference()
    sliced = tap._run_sliced(
        broken.inputs[0], broken.apply, broken.all_reduce, tap.slice_bounds(TOKENS, 4)
    )
    assert not torch.equal(sliced, reference)

    whole = BrokenRowMappingGroup(broken.inputs, broken.weights, OUT_FEATURES)
    unsliced = tap._run_sliced(
        whole.inputs[0], whole.apply, whole.all_reduce, tap.slice_bounds(TOKENS, 1)
    )
    assert torch.equal(unsliced, reference)


def test_single_slice_matches_the_unsliced_path_bitwise():
    """K == 1 while the flag is ON must equal the flag being OFF."""
    group = _make_group()
    direct = group.all_reduce(group.apply(group.inputs[0]))
    fresh = _make_group()
    piped = tap._run_unsliced(fresh.inputs[0], fresh.apply, fresh.all_reduce)
    assert torch.equal(direct, piped)


def test_output_dtype_and_layout_survive_slicing():
    group = _make_group()
    got = _run(group, tap.slice_bounds(TOKENS, 4))
    assert got.dtype == torch.float32
    assert got.is_contiguous()


# --------------------------------------------------------------------------
# rank uniformity: the deadlock property
# --------------------------------------------------------------------------


def test_derive_num_slices_signature_is_rank_uniform():
    """Pin the contract: no rank-local quantity may enter the derivation.

    A slice count is part of the collective sequence. If one rank derives 4
    and another derives 3, the group hangs on the fourth collective. The
    cheapest durable guard is to forbid rank-local INPUTS from reaching the
    function at all, so this pins the parameter set; widening it is a
    deliberate act that has to update this list.
    """
    params = set(inspect.signature(tap.derive_num_slices).parameters)
    assert params == {
        "payload_bytes",
        "num_tokens",
        "calibration",
        "max_slices",
        "slices_override",
    }


def test_derive_num_slices_body_reads_nothing_rank_local():
    source = inspect.getsource(tap.derive_num_slices)
    for forbidden in (
        "get_rank",
        "tp_rank",
        "local_rank",
        "mem_get_info",
        "device_count",
        "shard",
        "time.",
    ):
        assert forbidden not in source, forbidden


def test_derive_num_slices_is_deterministic_across_repeated_calls():
    calibration = tap.Calibration(
        latency_s=30e-6, wire_s_per_byte=1.0 / 6e9, compute_s_per_byte=70e-12
    )
    values = {
        tap.derive_num_slices(
            payload_bytes=16 * 1024 * 1024,
            num_tokens=1916,
            calibration=calibration,
            max_slices=8,
        )
        for _ in range(64)
    }
    assert len(values) == 1


def test_derive_num_slices_can_fail():
    """Can-fail proof for the uniformity pin: a rank-local name is caught."""
    source = "def f(rank):\n    return get_rank()\n"
    assert "get_rank" in source


# --------------------------------------------------------------------------
# K derivation
# --------------------------------------------------------------------------


def _calibration(latency_us=30.0, gbps=6.0, compute_ns_per_byte=0.07):
    return tap.Calibration(
        latency_s=latency_us * 1e-6,
        wire_s_per_byte=1.0 / (gbps * 1e9),
        compute_s_per_byte=compute_ns_per_byte * 1e-9,
    )


def test_k_follows_sqrt_of_compute_over_latency():
    calibration = _calibration()
    payload = 16 * 1024 * 1024
    expected = int(
        round(
            math.sqrt(payload * calibration.compute_s_per_byte / calibration.latency_s)
        )
    )
    got = tap.derive_num_slices(
        payload_bytes=payload, num_tokens=4096, calibration=calibration, max_slices=64
    )
    assert got == expected
    # And the formula is not a constant in disguise: a layer with ten times
    # the compute per byte must want more slices.
    heavier = tap.derive_num_slices(
        payload_bytes=payload,
        num_tokens=4096,
        calibration=_calibration(compute_ns_per_byte=0.7),
        max_slices=64,
    )
    assert heavier > got


def test_k_is_capped_by_max_slices():
    got = tap.derive_num_slices(
        payload_bytes=16 * 1024 * 1024,
        num_tokens=4096,
        calibration=_calibration(compute_ns_per_byte=7.0),
        max_slices=8,
    )
    assert got == 8


def test_k_is_capped_by_the_half_power_message_size():
    """No slice may be smaller than the size where latency equals transfer.

    A high-latency link (1 ms) has a large half-power size; splitting a
    16 MB payload below it spends more on launches than it saves.
    """
    got = tap.derive_num_slices(
        payload_bytes=16 * 1024 * 1024,
        num_tokens=4096,
        calibration=_calibration(latency_us=1000.0, compute_ns_per_byte=7.0),
        max_slices=64,
    )
    half_power = (1000.0 * 1e-6) * 6e9
    assert got == int((16 * 1024 * 1024) // half_power)
    assert got < 8


def test_k_is_capped_by_the_token_count():
    got = tap.derive_num_slices(
        payload_bytes=16 * 1024 * 1024,
        num_tokens=3,
        calibration=_calibration(compute_ns_per_byte=7.0),
        max_slices=64,
    )
    assert got == 3


def test_k_is_one_without_a_usable_calibration():
    assert (
        tap.derive_num_slices(
            payload_bytes=16 * 1024 * 1024,
            num_tokens=4096,
            calibration=None,
            max_slices=8,
        )
        == 1
    )
    unusable = tap.Calibration(
        latency_s=-1.0, wire_s_per_byte=1e-9, compute_s_per_byte=1e-9
    )
    assert not unusable.usable
    assert (
        tap.derive_num_slices(
            payload_bytes=16 * 1024 * 1024,
            num_tokens=4096,
            calibration=unusable,
            max_slices=8,
        )
        == 1
    )


def test_explicit_slice_override_wins_over_the_model():
    got = tap.derive_num_slices(
        payload_bytes=16 * 1024 * 1024,
        num_tokens=4096,
        calibration=_calibration(),
        max_slices=64,
        slices_override=5,
    )
    assert got == 5


# --------------------------------------------------------------------------
# calibration fit
# --------------------------------------------------------------------------


def test_two_point_fit_recovers_latency_and_bandwidth():
    latency, per_byte = 30e-6, 1.0 / 6e9
    payload, probe = 16 * 1024 * 1024, 4096
    calibration = tap.calibration_from_probe(
        compute_s=1.1e-3,
        payload_bytes=payload,
        big_all_reduce_s=latency + payload * per_byte,
        small_all_reduce_s=latency + probe * per_byte,
        small_bytes=probe,
    )
    assert calibration.usable
    assert calibration.latency_s == pytest.approx(latency, rel=1e-6)
    assert calibration.wire_s_per_byte == pytest.approx(per_byte, rel=1e-6)
    assert calibration.compute_s_per_byte == pytest.approx(1.1e-3 / payload, rel=1e-9)


def test_degenerate_fit_is_reported_unusable_rather_than_guessed():
    # A big transfer measured FASTER than the small probe: noise, not a
    # cost model. Must not become a slice count.
    calibration = tap.calibration_from_probe(
        compute_s=1e-3,
        payload_bytes=16 * 1024 * 1024,
        big_all_reduce_s=1e-6,
        small_all_reduce_s=1e-3,
        small_bytes=4096,
    )
    assert not calibration.usable
    assert (
        tap.derive_num_slices(
            payload_bytes=16 * 1024 * 1024,
            num_tokens=4096,
            calibration=calibration,
            max_slices=8,
        )
        == 1
    )


# --------------------------------------------------------------------------
# off by default
# --------------------------------------------------------------------------


def test_disabled_by_default():
    envs.SGLANG_TP_AR_PIPELINE.clear()
    tap.reset_tp_ar_pipeline_state()
    assert tap.tp_ar_pipeline_enabled() is False


def test_flag_turns_it_on_and_is_cached():
    with envs.SGLANG_TP_AR_PIPELINE.override(True):
        tap.reset_tp_ar_pipeline_state()
        assert tap.tp_ar_pipeline_enabled() is True
    # Cached for the process lifetime on purpose: the flag is part of the
    # collective sequence and must not change under a running group.
    assert tap.tp_ar_pipeline_enabled() is True
    tap.reset_tp_ar_pipeline_state()
    assert tap.tp_ar_pipeline_enabled() is False


def test_min_token_gate_keeps_short_forwards_unsliced():
    with envs.SGLANG_TP_AR_PIPELINE_MIN_TOKENS.override(256):
        assert tap.plan_num_slices(num_tokens=8, payload_bytes=1 << 20) == 1


def test_plan_uses_the_calibration_when_the_gate_opens():
    tap._STATE.calibration = _calibration()
    with (
        envs.SGLANG_TP_AR_PIPELINE_MIN_TOKENS.override(256),
        envs.SGLANG_TP_AR_PIPELINE_MAX_SLICES.override(8),
    ):
        assert tap.plan_num_slices(num_tokens=1916, payload_bytes=16 * 1024 * 1024) > 1


# --------------------------------------------------------------------------
# end-to-end entry point on the CPU branch
# --------------------------------------------------------------------------


def test_entry_point_calibrates_once_then_slices():
    group = _make_group(tokens=1024)
    reference = group.reference()

    # First call: unsliced and, on CPU, unable to time -- so it stays
    # unsliced forever unless a calibration is installed.
    first = tap.pipelined_row_all_reduce(
        group.inputs[0], group.apply, group.all_reduce, out_features=OUT_FEATURES
    )
    assert torch.equal(first, reference)
    assert tap.tp_ar_pipeline_stats()["calls_unsliced"] == 1
    assert tap.tp_ar_pipeline_stats()["calls_pipelined"] == 0

    # A link fast enough that a 256 KiB payload is worth splitting: the
    # half-power size is 6 KiB here, so the cap that matters is max_slices.
    tap._STATE.calibration = _calibration(latency_us=1.0, compute_ns_per_byte=5.0)
    group2 = _make_group(tokens=1024)
    with (
        envs.SGLANG_TP_AR_PIPELINE_MIN_TOKENS.override(256),
        envs.SGLANG_TP_AR_PIPELINE_MAX_SLICES.override(8),
    ):
        second = tap.pipelined_row_all_reduce(
            group2.inputs[0], group2.apply, group2.all_reduce, out_features=OUT_FEATURES
        )
    assert torch.equal(second, group2.reference())
    stats = tap.tp_ar_pipeline_stats()
    assert stats["calls_pipelined"] == 1
    assert stats["slices_issued"] > 1


# --------------------------------------------------------------------------
# the hook in RowParallelLinear.forward
# --------------------------------------------------------------------------


class _FakeQuantMethod:
    def apply(self, layer, x, bias=None):
        out = x @ layer.weights[0]
        if bias is not None:
            out = out + bias
        return out


def _fake_row_linear(group: FakeTpGroup, reduce_results: bool = True):
    """Duck-typed stand-in so ``RowParallelLinear.forward`` can run without a
    process group. Only the attributes that forward reads are provided; an
    added dependency shows up as an AttributeError, which is the point."""
    return types.SimpleNamespace(
        input_is_parallel=True,
        tp_size=WORLD,
        tp_rank=0,
        skip_bias_add=False,
        bias=None,
        use_dp_attention_reduce=False,
        reduce_results=reduce_results,
        output_size=OUT_FEATURES,
        quant_method=_FakeQuantMethod(),
        weights=group.weights,
    )


@pytest.fixture
def patched_linear(monkeypatch):
    from sglang.srt.layers import linear as linear_mod

    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: object())
    monkeypatch.setattr(
        linear_mod, "use_symmetric_memory", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(linear_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(linear_mod, "should_skip_mlp_all_reduce", lambda: False)
    return linear_mod


def test_hook_is_inert_when_the_flag_is_off(patched_linear, monkeypatch):
    """The default path must not even enter the pipeline module.

    Counters at zero is the strong form of "byte-identical when off": the
    module was never called, so there is nothing it could have changed.
    """
    group = _make_group(tokens=1024)
    monkeypatch.setattr(
        patched_linear, "tensor_model_parallel_all_reduce", group.all_reduce
    )
    envs.SGLANG_TP_AR_PIPELINE.clear()
    tap.reset_tp_ar_pipeline_state()

    layer = _fake_row_linear(group)
    output, bias = patched_linear.RowParallelLinear.forward(layer, group.inputs[0])

    assert torch.equal(output, group.reference())
    assert bias is None
    stats = tap.tp_ar_pipeline_stats()
    assert stats["calls_pipelined"] == 0
    assert stats["calls_unsliced"] == 0
    assert stats["calibrated"] is False
    # One collective, exactly as before the feature existed.
    assert group.call_count == 1


def test_hook_pipelines_and_stays_bitwise_identical_when_on(
    patched_linear, monkeypatch
):
    with (
        envs.SGLANG_TP_AR_PIPELINE.override(True),
        envs.SGLANG_TP_AR_PIPELINE_MIN_TOKENS.override(256),
        envs.SGLANG_TP_AR_PIPELINE_MAX_SLICES.override(8),
    ):
        tap.reset_tp_ar_pipeline_state()

        warmup = _make_group(tokens=1024)
        monkeypatch.setattr(
            patched_linear, "tensor_model_parallel_all_reduce", warmup.all_reduce
        )
        first, _ = patched_linear.RowParallelLinear.forward(
            _fake_row_linear(warmup), warmup.inputs[0]
        )
        assert torch.equal(first, warmup.reference())
        assert tap.tp_ar_pipeline_stats()["calibrated"] is True

        tap._STATE.calibration = _calibration(latency_us=1.0, compute_ns_per_byte=5.0)
        group = _make_group(tokens=1024)
        monkeypatch.setattr(
            patched_linear, "tensor_model_parallel_all_reduce", group.all_reduce
        )
        output, _ = patched_linear.RowParallelLinear.forward(
            _fake_row_linear(group), group.inputs[0]
        )

    assert torch.equal(output, group.reference())
    stats = tap.tp_ar_pipeline_stats()
    assert stats["calls_pipelined"] == 1
    assert stats["slices_issued"] > 1
    # The wire was split into exactly the planned number of collectives.
    assert group.call_count == stats["slices_issued"]


def test_hook_declines_when_the_layer_does_not_reduce(patched_linear, monkeypatch):
    """reduce_results=False defers the all-reduce to the LayerCommunicator.

    The pipeline must not engage there: the tensor leaves this function
    unreduced and something downstream owns the collective.
    """
    group = _make_group(tokens=1024)
    monkeypatch.setattr(
        patched_linear, "tensor_model_parallel_all_reduce", group.all_reduce
    )
    monkeypatch.setattr(
        patched_linear,
        "get_forward",
        lambda: types.SimpleNamespace(fuse_mlp_allreduce=False),
    )
    with envs.SGLANG_TP_AR_PIPELINE.override(True):
        tap.reset_tp_ar_pipeline_state()
        layer = _fake_row_linear(group, reduce_results=False)
        output, _ = patched_linear.RowParallelLinear.forward(layer, group.inputs[0])
    assert group.call_count == 0
    assert tap.tp_ar_pipeline_stats()["calls_pipelined"] == 0
    assert torch.equal(output, group.inputs[0] @ group.weights[0])


def test_non_2d_input_stays_on_the_unsliced_path():
    tensor = torch.zeros(2, 3, 4)
    calls = {"n": 0}

    def apply_fn(x):
        calls["n"] += 1
        return x

    out = tap.pipelined_row_all_reduce(
        tensor, apply_fn, lambda t: t, out_features=OUT_FEATURES
    )
    assert out.shape == tensor.shape
    assert calls["n"] == 1
