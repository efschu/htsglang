# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the #391c streaming GGUF-MoE staging door.

No CUDA, no model file, no kernel. A synthetic multi-layer GGUF weight stream is
pushed through the REAL ``FusedMoE._load_gguf_weight`` /
``materialize_gguf_weights`` pair, in the order the real iterator emits
(``gguf_quant_weights_iterator``: one whole ``ffn_gate_exps`` tensor, then
``ffn_up_exps``, then ``ffn_down_exps``, per layer, and every ``qweight_type``
marker ahead of all of them).

What is pinned here:

  * the PEAK LEDGER -- host bytes retained at the worst instant stay under
    [pinned tier + one layer's incomplete set + slack] while several layers'
    worth of bytes flow through, and the same bound applied to the old
    accumulate-then-materialize path FAILS (the can-fail companion, without
    which the bound proves nothing);
  * byte identity -- the tiers the streaming door builds are the tiers the
    materialization door builds from the same input, tensor for tensor;
  * idempotency -- a second ``process_weights_after_loading`` after streaming
    is a no-op, tensors and release tally included;
  * default unchanged -- with no resident fraction the stream takes the old
    accumulate path and produces the same full ``[E, ...]`` stack;
  * #394 -- a delegated cold expert is released IN STREAM (never copied into
    this rank's pinned tier), and without ``SGLANG_MOE_HOST_SHARD_RATIO`` the
    door constructs no context at all.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
    python -m pytest tests/moe_offload/test_gguf_streaming_staging.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
from torch.nn.parameter import UninitializedParameter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe import expert_offload  # noqa: E402
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    plan_load_time_staging,
    reset_expert_offload_release,
    reset_host_shard_log_latch,
    reset_streaming_staging_ledger,
    streaming_staging_ledger,
)

FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"
SCRATCH_ENV = "SGLANG_MOE_SCRATCH_SLOTS"
STREAM_ENV = "SGLANG_MOE_GGUF_STREAM_STAGING"
TRACE_ENV = "SGLANG_MOE_STAGING_TRACE"
RATIO_ENV = "SGLANG_MOE_HOST_SHARD_RATIO"
CARD_UUIDS_ENV = "SGLANG_RANK_CARD_UUIDS"
UNSAFE_DELEGATE_ENV = "SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE"

# Toy geometry. Row byte counts are whole ggml K-quant blocks (Q4_K 144 B,
# Q6_K 210 B) so the byte arithmetic is the arithmetic of a real checkpoint;
# nothing here decodes them, they are opaque payload on purpose.
LAYERS = 4
E = 8
GATE_ROWS = 2  # ffn_gate_exps rows per expert; up is the same width
W13_ROWS = 2 * GATE_ROWS
W2_ROWS = 3
W13_ROW_BYTES = 144 * 2
W2_ROW_BYTES = 210 * 2
W13_TYPE = 12  # Q4_K
W2_TYPE = 14  # Q6_K

W13_EXPERT_BYTES = W13_ROWS * W13_ROW_BYTES
W2_EXPERT_BYTES = W2_ROWS * W2_ROW_BYTES
EXPERT_BYTES = W13_EXPERT_BYTES + W2_EXPERT_BYTES


def _shard_bytes(rows, row_bytes, seed):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        np.ascontiguousarray(
            rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
        )
    )


def _stream_plan(num_layers=LAYERS, owned=range(E)):
    """The synthetic checkpoint: (layer, expert, shard) -> deterministic bytes.

    Emitted in iterator order, which is what makes the peak claim meaningful:
    an expert's ``w13`` row is only complete once ``ffn_up_exps`` arrives, i.e.
    after the layer's whole gate tensor has gone past, so "one layer's
    incomplete set" is the real bound, not "one expert".
    """
    events = []
    for layer in range(num_layers):
        for shard, rows, row_bytes in (
            ("w1", GATE_ROWS, W13_ROW_BYTES),
            ("w3", GATE_ROWS, W13_ROW_BYTES),
            ("w2", W2_ROWS, W2_ROW_BYTES),
        ):
            for expert in owned:
                seed = (layer * 1000) + (expert * 10) + "w1w3w2".index(shard)
                events.append(
                    (layer, expert, shard, _shard_bytes(rows, row_bytes, seed))
                )
    return events


# --------------------------------------------------------------------------
# a stub layer that borrows the real methods under test
# --------------------------------------------------------------------------


def _stub_layer_type():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    members = {
        name: value
        for name, value in vars(FusedMoE).items()
        if name.lower().startswith("_gguf")
    }
    for name in (
        "_load_gguf_weight",
        "_drain_gguf_stream_stagers",
        "_maybe_close_gguf_stream_layer",
        "_finish_gguf_moe_offload_staging",
        "materialize_gguf_weights",
        "_new_gguf_stream_stager",
    ):
        members[name] = vars(FusedMoE)[name]
    return type("_StreamGGUFMoELayer", (torch.nn.Module,), members)


class _QType:
    def __init__(self, weight_type):
        self.weight_type = int(weight_type)


class _FakeGGUFMoEMethod:
    pass


_FakeGGUFMoEMethod.__name__ = "GGUFMoEMethod"


def _make_layer(
    layer_id,
    fraction,
    expert_shard=False,
    owned=E,
    moe_tp_size=1,
    moe_tp_rank=0,
):
    from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter

    layer = _stub_layer_type()()
    layer.layer_id = layer_id
    layer.quant_method = _FakeGGUFMoEMethod()
    layer.num_experts = E
    layer.num_local_experts = E
    layer.moe_tp_size = moe_tp_size
    layer.moe_tp_rank = moe_tp_rank
    layer._expert_offload_fraction = fraction
    layer._gguf_expert_shard = expert_shard
    layer._gguf_expert_range = (0, owned)
    layer.w13_qweight_type = _QType(W13_TYPE)
    layer.w2_qweight_type = _QType(W2_TYPE)
    for attr, rows, row_bytes in (
        ("w13_qweight", W13_ROWS, W13_ROW_BYTES),
        ("w2_qweight", W2_ROWS, W2_ROW_BYTES),
    ):
        param = GGUFUninitializedParameter(requires_grad=False)
        param.is_gguf_weight = True
        param.output_dim = 0
        param.tensor_shape = (E, rows, row_bytes)
        param.data_container = []
        param.expert_data_map = {}
        layer.register_parameter(attr, param)
    return layer


# --------------------------------------------------------------------------
# the peak ledger
# --------------------------------------------------------------------------


def _host_bytes_retained(layers):
    """HOST bytes the load is holding right now, deduplicated by storage.

    Counted: the loader's per-expert holders (``expert_data_map``,
    ``data_container``), the stager's in-flight shards, and the pinned spill
    tier -- every host allocation the load owns.

    NOT counted: the ``[R+C]`` resident buffer. On the rig that buffer is
    device memory (``param.materialize`` inherits the parameter's device, which
    is the load target); on a desk box without CUDA it lands on the host and
    would flatter neither arm honestly. Excluding it measures the same quantity
    in both arms, which is what a comparison needs.
    """
    seen = {}

    def add(tensor):
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        if tensor.device.type != "cpu":
            return
        storage = tensor.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()

    for layer in layers:
        for attr in ("w13_qweight", "w2_qweight"):
            param = getattr(layer, attr, None)
            if param is None:
                continue
            for tensor in getattr(param, "data_container", None) or []:
                add(tensor)
            for tensor in (getattr(param, "expert_data_map", None) or {}).values():
                add(tensor)
        for stager in (getattr(layer, "_gguf_stream_stagers", None) or {}).values():
            add(stager.spill)
            for parts in stager._pending.values():
                for tensor in parts.values():
                    add(tensor)
        for _buf, spill in (
            getattr(layer, "_moe_offload_presplit", None) or {}
        ).values():
            add(spill)
    return sum(seen.values())


def _run_stream(layers, events, materialize_at_end=True):
    """Drive the synthetic stream and return the peak host bytes retained.

    The materialization hook is called for EVERY layer only after the whole
    stream, exactly as ``DefaultModelLoader.load_model`` does (loader.py: the
    ``process_weights_after_loading`` loop runs after ``model.load_weights``).
    That ordering is the defect this change is about, so the test must not
    quietly improve on it.
    """
    peak = 0
    for layer_id, expert, shard, tensor in events:
        layer = layers[layer_id]
        param = getattr(layer, "w2_qweight" if shard == "w2" else "w13_qweight")
        layer._load_gguf_weight(param, tensor, shard, expert, tp_rank=0)
        del tensor
        peak = max(peak, _host_bytes_retained(layers))
    if materialize_at_end:
        for layer in layers:
            layer.materialize_gguf_weights()
            peak = max(peak, _host_bytes_retained(layers))
    return peak


def _bound(num_layers, plan):
    """[pinned tier over all layers] + [one layer's incomplete set] + slack."""
    pinned = num_layers * len(plan.spill_ids) * EXPERT_BYTES
    one_layer_incomplete = E * GATE_ROWS * W13_ROW_BYTES
    return pinned + one_layer_incomplete, pinned, one_layer_incomplete


def test_streamed_peak_stays_under_pinned_plus_one_layer(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    reset_expert_offload_release()
    reset_streaming_staging_ledger()

    plan = plan_load_time_staging(E, fraction=0.5)
    assert plan.resident_count == 4 and plan.buffer_slots == 5
    assert len(plan.spill_ids) == 4

    layers = [_make_layer(i, 0.5) for i in range(LAYERS)]
    peak = _run_stream(layers, _stream_plan())

    bound, pinned, incomplete = _bound(LAYERS, plan)
    total_streamed = LAYERS * E * EXPERT_BYTES
    assert peak <= bound, (
        f"streamed peak {peak} exceeds pinned {pinned} + one layer's "
        f"incomplete set {incomplete}"
    )
    # ... while several layers' worth of bytes went past. The bound itself is
    # 56% of the flow at fraction 0.5 (pinned 4/8 of every layer + one gate
    # tensor), so a peak inside it is a peak the old path could not reach.
    assert peak < 0.60 * total_streamed
    assert bound < 0.60 * total_streamed

    # The code's own accounting agrees with the external measurement.
    ledger = streaming_staging_ledger()
    assert ledger.streamed_bytes == total_streamed
    assert ledger.pinned_bytes == pinned
    assert ledger.layers == LAYERS
    assert ledger.peak_host_bytes <= bound
    assert ledger.inflight_bytes == 0  # nothing left in flight at the end


def test_the_peak_bound_can_fail_on_the_old_accumulate_path(monkeypatch):
    """Can-fail companion: the same bound, the same stream, the old door.

    ``SGLANG_MOE_GGUF_STREAM_STAGING=0`` restores accumulate-then-materialize.
    Its ledger peak is the FULL streamed set -- every layer's every expert alive
    at once -- which is precisely the 126.19 GiB against 98.5 GiB of host RAM
    that OOM-killed boot attempt 5.
    """
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    monkeypatch.setenv(STREAM_ENV, "0")
    reset_expert_offload_release()
    reset_streaming_staging_ledger()

    plan = plan_load_time_staging(E, fraction=0.5)
    layers = [_make_layer(i, 0.5) for i in range(LAYERS)]
    peak = _run_stream(layers, _stream_plan())

    bound, _pinned, _incomplete = _bound(LAYERS, plan)
    total_streamed = LAYERS * E * EXPERT_BYTES

    assert peak > bound, "the bound must not hold for the accumulate path"
    assert peak >= total_streamed, "every expert of every layer was held at once"
    # The staging ledger never saw a byte: the streaming door was not taken.
    assert streaming_staging_ledger().streamed_bytes == 0
    # It did still produce the tiers, at the old door.
    for layer in layers:
        assert layer._moe_offload_gguf_staged is True


def test_trace_logs_cumulative_host_bytes_at_layer_boundaries(monkeypatch, caplog):
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    monkeypatch.setenv(TRACE_ENV, "1")
    reset_expert_offload_release()
    reset_streaming_staging_ledger()

    layers = [_make_layer(i, 0.5) for i in range(LAYERS)]
    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.expert_offload"):
        # NO materialization pass: the boundaries must be reached DURING the
        # stream. A line printed at the drain would print for all N layers at
        # once, after the load, and could not be lined up against a
        # time-series RAM monitor -- which is the whole use for it.
        _run_stream(layers, _stream_plan(), materialize_at_end=False)

    lines = [
        r.getMessage()
        for r in caplog.records
        if "[moe-staging-trace]" in r.getMessage()
    ]
    assert len(lines) == LAYERS  # one boundary per layer, not per tensor
    assert lines[0].startswith("[moe-staging-trace] layer 0 staged (#1):")
    assert "peak host held" in lines[-1]
    # Cumulative, so the last line's pinned figure is the whole pinned tier.
    plan = plan_load_time_staging(E, fraction=0.5)
    pinned = LAYERS * len(plan.spill_ids) * EXPERT_BYTES
    assert f"pinned(host)={pinned / 1024:.2f} KiB" in lines[-1]
    # ... and closing the layer stays a once-per-layer event when the
    # materialization hook does eventually run.
    for layer in layers:
        layer.materialize_gguf_weights()
    assert (
        len([r for r in caplog.records if "[moe-staging-trace]" in r.getMessage()])
        == LAYERS
    )
    assert streaming_staging_ledger().layers == LAYERS


def test_trace_is_silent_without_the_env(monkeypatch, caplog):
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    monkeypatch.delenv(TRACE_ENV, raising=False)
    reset_expert_offload_release()
    reset_streaming_staging_ledger()

    layers = [_make_layer(0, 0.5)]
    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.expert_offload"):
        _run_stream(layers, _stream_plan(num_layers=1))

    assert not [r for r in caplog.records if "[moe-staging-trace]" in r.getMessage()]
    # The ledger still counts -- only the printing is gated.
    assert streaming_staging_ledger().layers == 1


# --------------------------------------------------------------------------
# byte identity against the materialization door
# --------------------------------------------------------------------------


@pytest.mark.parametrize("expert_shard", [False, True])
def test_streamed_tiers_are_byte_identical_to_the_materialized_ones(
    monkeypatch, expert_shard
):
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    owned = E - 1 if expert_shard else E
    events = _stream_plan(num_layers=1, owned=range(owned))

    reset_expert_offload_release()
    reset_streaming_staging_ledger()
    monkeypatch.setenv(STREAM_ENV, "1")
    streamed = [_make_layer(0, 0.5, expert_shard=expert_shard, owned=owned)]
    _run_stream(streamed, events)

    reset_expert_offload_release()
    monkeypatch.setenv(STREAM_ENV, "0")
    pulled = [_make_layer(0, 0.5, expert_shard=expert_shard, owned=owned)]
    _run_stream(pulled, events)

    a, b = streamed[0], pulled[0]
    assert a._moe_offload_gguf_staged and b._moe_offload_gguf_staged
    assert a._moe_offload_full_experts == b._moe_offload_full_experts
    assert getattr(a, "_moe_offload_frozen_layout", None) == getattr(
        b, "_moe_offload_frozen_layout", None
    )
    plan_ids = plan_load_time_staging(
        owned + (1 if expert_shard else 0),
        fraction=0.5,
        pinned_experts=(owned,) if expert_shard else (),
    )
    for attr in ("w13_qweight", "w2_qweight"):
        buf_a, spill_a = a._moe_offload_presplit[attr]
        buf_b, spill_b = b._moe_offload_presplit[attr]
        assert buf_a.shape == buf_b.shape
        for slot in range(plan_ids.resident_count):
            assert torch.equal(buf_a[slot], buf_b[slot]), f"{attr} resident slot {slot}"
        assert torch.equal(spill_a, spill_b), f"{attr} pinned tier"


def test_the_byte_identity_pin_can_fail(monkeypatch):
    """A stream fed in a DIFFERENT expert order must not match -- otherwise the
    comparison above would pass for any placement at all."""
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    monkeypatch.setenv(STREAM_ENV, "1")
    events = _stream_plan(num_layers=1)
    shuffled = [
        (layer, (expert + 3) % E, shard, tensor)
        for layer, expert, shard, tensor in events
    ]

    reset_expert_offload_release()
    reset_streaming_staging_ledger()
    a = [_make_layer(0, 0.5)]
    _run_stream(a, events)
    reset_expert_offload_release()
    b = [_make_layer(0, 0.5)]
    _run_stream(b, shuffled)

    spill_a = a[0]._moe_offload_presplit["w2_qweight"][1]
    spill_b = b[0]._moe_offload_presplit["w2_qweight"][1]
    assert not torch.equal(spill_a, spill_b)


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


def test_process_weights_after_loading_is_idempotent_after_streaming(monkeypatch):
    from sglang.srt.layers.moe.expert_offload import expert_offload_release_totals

    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    reset_expert_offload_release()
    reset_streaming_staging_ledger()

    layers = [_make_layer(0, 0.5)]
    _run_stream(layers, _stream_plan(num_layers=1))
    layer = layers[0]
    before = {
        attr: (buf, spill) for attr, (buf, spill) in layer._moe_offload_presplit.items()
    }
    tally_before = expert_offload_release_totals()
    ledger_before = streaming_staging_ledger().pinned_bytes

    layer.materialize_gguf_weights()
    layer.materialize_gguf_weights()

    for attr, (buf, spill) in before.items():
        buf2, spill2 = layer._moe_offload_presplit[attr]
        assert buf2 is buf and spill2 is spill  # same objects, not re-staged
    assert expert_offload_release_totals() == tally_before
    assert streaming_staging_ledger().pinned_bytes == ledger_before
    assert streaming_staging_ledger().layers == 1  # one trace boundary, not three


# --------------------------------------------------------------------------
# default unchanged
# --------------------------------------------------------------------------


def test_default_gguf_boot_takes_the_old_accumulate_path(monkeypatch):
    """No resident fraction -> no staging door is even opened."""
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    reset_streaming_staging_ledger()

    layers = [_make_layer(0, 1.0)]
    _run_stream(layers, _stream_plan(num_layers=1))
    layer = layers[0]

    assert not hasattr(layer, "_moe_offload_presplit")
    assert not getattr(layer, "_moe_offload_gguf_staged", False)
    assert layer.w13_qweight.shape == (E, W13_ROWS, W13_ROW_BYTES)
    assert layer.w2_qweight.shape == (E, W2_ROWS, W2_ROW_BYTES)
    assert streaming_staging_ledger().streamed_bytes == 0
    assert not isinstance(layer.w13_qweight, UninitializedParameter)


def test_default_path_bytes_match_the_stream(monkeypatch):
    """The full stack the default path builds is the stream, expert by expert."""
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    events = _stream_plan(num_layers=1)
    layers = [_make_layer(0, 1.0)]
    _run_stream(layers, events)

    by_key = {(expert, shard): tensor for _l, expert, shard, tensor in events}
    for expert in range(E):
        expected = torch.cat([by_key[(expert, "w1")], by_key[(expert, "w3")]], dim=0)
        assert torch.equal(layers[0].w13_qweight.data[expert], expected)
        assert torch.equal(layers[0].w2_qweight.data[expert], by_key[(expert, "w2")])


# --------------------------------------------------------------------------
# #394 wiring
# --------------------------------------------------------------------------


def test_no_ratio_means_no_cold_shard_context(monkeypatch):
    monkeypatch.delenv(RATIO_ENV, raising=False)
    monkeypatch.delenv(CARD_UUIDS_ENV, raising=False)
    layer = _make_layer(0, 0.5, expert_shard=True, owned=E - 1, moe_tp_size=3)
    assert layer._gguf_cold_shard_context() is None


def test_the_published_card_vector_activates_the_context(monkeypatch):
    """#407 cut 2 -> #394: the layer reads the launcher's vector, not a peer.

    The point of the assertion below is the ABSENCE of a collective: the only
    thing that changed between this test and the one above is an environment
    variable, and the context came alive.
    """
    monkeypatch.delenv(RATIO_ENV, raising=False)
    monkeypatch.setenv(CARD_UUIDS_ENV, "GPU-x4,GPU-x8a,GPU-x8b")
    monkeypatch.setenv(UNSAFE_DELEGATE_ENV, "1")
    monkeypatch.setattr(
        expert_offload,
        "_measured_h2d_gbps_by_uuid",
        lambda uuid: {"GPU-x4": 6.4, "GPU-x8a": 13.0, "GPU-x8b": 13.0}[uuid],
    )
    layer = _make_layer(0, 0.5, expert_shard=True, owned=E - 1, moe_tp_size=3)

    context = layer._gguf_cold_shard_context()

    assert context is not None
    assert context.ratio.source == "card-probe-h2d"
    assert context.ratio.provenance == "measured"
    assert context.ratio.weights[0] == min(context.ratio.weights)


def test_a_vector_for_a_wider_group_is_not_read_by_a_narrower_one(monkeypatch):
    """A 4-entry vector against a 3-rank MoE group describes another group."""
    monkeypatch.delenv(RATIO_ENV, raising=False)
    monkeypatch.setenv(CARD_UUIDS_ENV, "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setenv(UNSAFE_DELEGATE_ENV, "1")
    layer = _make_layer(0, 0.5, expert_shard=True, owned=E - 1, moe_tp_size=3)
    assert layer._gguf_cold_shard_context() is None


def test_a_disjoint_expert_shard_refuses_to_delegate_by_default(monkeypatch):
    """Measured 2026-08-02: delegation on a disjoint shard loses the expert.

    The ranks hold disjoint expert ranges under #82, so a delegated cold expert
    is not relocated to a peer -- it is absent, and the first token routed to it
    dies in ExpertResidencyPlanner.resolve. The refusal must hold even with a
    ratio AND a card vector both present, which is what this pins.
    """
    monkeypatch.setenv(RATIO_ENV, "1,2,2")
    monkeypatch.setenv(CARD_UUIDS_ENV, "GPU-x4,GPU-x8a,GPU-x8b")
    monkeypatch.delenv(UNSAFE_DELEGATE_ENV, raising=False)
    layer = _make_layer(0, 0.5, expert_shard=True, owned=E - 1, moe_tp_size=3)
    assert layer._gguf_cold_shard_context() is None


def test_an_intermediate_dim_layer_never_delegates(monkeypatch):
    """The ColdShardContext precondition, enforced at the construction site."""
    monkeypatch.setenv(RATIO_ENV, "1,2,2")
    monkeypatch.setenv(UNSAFE_DELEGATE_ENV, "1")
    layer = _make_layer(0, 0.5, expert_shard=False, moe_tp_size=3)
    assert layer._gguf_cold_shard_context() is None


def test_delegated_experts_are_released_in_stream(monkeypatch):
    monkeypatch.setenv(UNSAFE_DELEGATE_ENV, "1")
    monkeypatch.setenv(FRACTION_ENV, "0.5")
    monkeypatch.setenv(SCRATCH_ENV, "1")
    monkeypatch.setenv(RATIO_ENV, "1,2,2")
    reset_expert_offload_release()
    reset_streaming_staging_ledger()
    reset_host_shard_log_latch()

    owned = E - 1
    layers = [_make_layer(0, 0.5, expert_shard=True, owned=owned, moe_tp_size=3)]
    _run_stream(layers, _stream_plan(num_layers=1, owned=range(owned)))
    layer = layers[0]

    reference = plan_load_time_staging(E, fraction=0.5, pinned_experts=(owned,))
    delegated = layer._moe_offload_delegated_experts
    assert delegated, "the ratio should have handed cold experts to the peers"
    # This rank's pinned tier holds ONLY its own share of the cold pool.
    for attr in ("w13_qweight", "w2_qweight"):
        _buf, spill = layer._moe_offload_presplit[attr]
        assert spill.shape[0] == len(reference.spill_ids) - len(delegated)
    # Residency is untouched by the ratio (#394's central claim).
    assert layer._moe_offload_frozen_layout[0] == list(reference.resident_ids)

    ledger = streaming_staging_ledger()
    # Delegated bytes crossed the stream and were dropped without a copy: they
    # are in ``streamed`` and in ``delegated``, and in no tier.
    expected_pinned = sum(
        layer._moe_offload_presplit[attr][1].numel()
        for attr in ("w13_qweight", "w2_qweight")
    )
    assert ledger.pinned_bytes == expected_pinned
    assert ledger.delegated_bytes == len(delegated) * EXPERT_BYTES
    assert ledger.inflight_bytes == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
