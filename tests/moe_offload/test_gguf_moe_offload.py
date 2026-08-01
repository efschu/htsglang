# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the GGUF-MoE load-time expert offload (#123, GGUF half).

No CUDA, no model file, no kernel. The expert tensors are synthetic GGUF
QUANTIZED blocks -- real Q4_K (144 B / 256 values) and Q6_K (210 B / 256
values) byte layouts -- and ``gguf.quants.dequantize`` is the reference
decoder, so "the bytes survived the tiering" is checked against ggml's own
semantics rather than against our own copy of them.

What is pinned here:

  * ``plan_load_time_staging`` -- residency decided before any allocation,
    including the pinned padding expert of the uneven-TP expert-dim shard
    (#82), whose id is ``E-1`` and would otherwise be the ONE expert every
    foreign token routes to AND the one guaranteed to be in the spill tier;
  * ``stage_experts_into_tiers`` -- resident slots and spill rows hold the
    expert the plan says, byte-for-byte, and the scratch region is left alone;
  * whole-expert granularity: an expert's row bytes are an exact multiple of
    its ggml block size and no staging step ever cuts inside a row (#82/#109);
  * ``materialize_gguf_weights`` -- with a fraction < 1 it never builds the
    ``[E, ...]`` stack; with fraction 1.0 it builds exactly the stack it always
    did (default path unchanged);
  * a full round trip: install -> resolve -> fetch -> remap -> apply over the
    two tiers reproduces the all-resident reference BIT-FOR-BIT after dequant;
  * the #268 guard both ways: a staged layer is admitted, an unstaged one
    (uncovered ggml type, e.g. MXFP4) is still refused;
  * the #390 expert_stats instrument counts on the GGUF path.

The fetch in the round-trip test is a host->host copy (no CUDA context), so it
proves the ADDRESSING (which row lands in which slot) and the byte identity,
not the PCIe transfer itself. That part is the same ``_fetch`` code the fp8 /
GPTQ / AWQ halves already run on the card.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
    python -m pytest tests/moe_offload/test_gguf_moe_offload.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from gguf import GGMLQuantizationType as WeightType  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    MoEExpertOffloadCache,
    assert_expert_offload_quant_supported,
    plan_load_time_staging,
    reset_expert_offload_release,
    stage_experts_into_tiers,
)

FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"
SCRATCH_ENV = "SGLANG_MOE_SCRATCH_SLOTS"

# Toy geometry. Only two things have to be true of it for the tiering claims:
# every row is a whole number of ggml blocks, and the expert axis is dim 0.
E = 12
W13_ROWS = 4  # gate+up rows per expert
W2_ROWS = 3  # down rows per expert
BLOCKS_PER_ROW = 2
W13_TYPE = WeightType.Q4_K
W2_TYPE = WeightType.Q6_K


def _row_bytes(qtype):
    _, block_bytes = GGML_QUANT_SIZES[qtype]
    return block_bytes * BLOCKS_PER_ROW


# Byte offsets of the fp16 super-block scale fields inside one ggml block:
#   block_q4_K { half d; half dmin; uint8 scales[12]; uint8 qs[128]; }  (144 B)
#   block_q6_K { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; } (210 B)
# Random bytes are a valid ggml payload everywhere EXCEPT here: an fp16 whose
# exponent field is all ones decodes to inf/NaN, and NaN != NaN would make the
# byte-identity comparison below meaningless. Clearing the top exponent bit of
# the high byte keeps the field finite without touching a single quant bit.
_F16_SCALE_BYTES = {
    WeightType.Q4_K: (1, 3),
    WeightType.Q6_K: (209,),
}


def _expert_bytes(qtype, rows, seed):
    """One expert's quantized tensor: uint8 [rows, row_bytes].

    Raw bytes rather than a quantizer output on purpose: gguf-py 0.19.0
    implements no Q4_K / Q6_K *encoder* (``quantize`` raises
    NotImplementedError), only the decoder -- which is the direction this test
    needs, and the one the CUDA kernels mirror.
    """
    rng = np.random.default_rng(seed)
    _, block_bytes = GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, size=(rows, _row_bytes(qtype)), dtype=np.uint8)
    blocks = raw.reshape(rows, BLOCKS_PER_ROW, block_bytes)
    for off in _F16_SCALE_BYTES[qtype]:
        blocks[:, :, off] &= 0xBF
    return torch.from_numpy(np.ascontiguousarray(blocks.reshape(raw.shape)))


def _dequant(t):
    """Reference decode of one expert's Q4_K/Q6_K bytes to float32 rows."""
    qtype = W13_TYPE if t.shape[-1] == _row_bytes(W13_TYPE) else W2_TYPE
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.numpy(), qtype)))


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def test_plan_is_none_without_offload(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    assert plan_load_time_staging(E) is None


def test_plan_static_layout(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    plan = plan_load_time_staging(E)
    assert plan.num_experts == E
    assert plan.resident_count == 3  # ceil(0.25 * 12)
    assert plan.buffer_slots == 5  # R + scratch
    assert plan.resident_ids == (0, 1, 2)
    assert plan.spill_ids == tuple(range(3, E))
    assert plan.is_static_layout


def test_plan_pins_the_padding_expert(monkeypatch):
    """#82: the shard's zero-pad expert is id E-1 and must stay resident."""
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    plan = plan_load_time_staging(E, pinned_experts=(E - 1,))
    assert plan.resident_ids == (E - 1, 0, 1)
    assert plan.spill_ids == tuple(range(2, E - 1))
    assert not plan.is_static_layout
    # every expert accounted for, exactly once
    assert sorted(plan.resident_ids + plan.spill_ids) == list(range(E))


def test_plan_refuses_more_pinned_than_resident_slots(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.1")  # R = ceil(1.2) = 2
    with pytest.raises(ValueError, match="must stay resident"):
        plan_load_time_staging(E, pinned_experts=(0, 1, 2))


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------


def _stage(plan, qtype, rows):
    experts = [
        _expert_bytes(qtype, rows, seed=100 + e) for e in range(plan.num_experts)
    ]
    out = torch.zeros((plan.buffer_slots, rows, _row_bytes(qtype)), dtype=torch.uint8)
    spill = stage_experts_into_tiers(plan, lambda e: experts[e], out)
    return experts, out, spill


def test_staging_places_every_expert_exactly_where_the_plan_says(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    plan = plan_load_time_staging(E, pinned_experts=(E - 1,))
    experts, out, spill = _stage(plan, W13_TYPE, W13_ROWS)

    for slot, expert_id in enumerate(plan.resident_ids):
        assert torch.equal(out[slot], experts[expert_id]), f"slot {slot}"
    for row, expert_id in enumerate(plan.spill_ids):
        assert torch.equal(spill[row], experts[expert_id]), f"pool row {row}"
    assert spill.shape[0] == E - plan.resident_count
    # Scratch region stays untouched by staging (the fetch path owns it).
    for slot in range(plan.resident_count, plan.buffer_slots):
        assert torch.equal(out[slot], torch.zeros_like(out[slot]))


def test_staging_never_holds_the_full_stack(monkeypatch):
    """``source`` is called once per expert and ``release`` fires immediately.

    That is the property that keeps host peak at (two tiers + one expert)
    rather than at (two tiers + the whole loaded set).
    """
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    plan = plan_load_time_staging(E)
    live = {e: _expert_bytes(W13_TYPE, W13_ROWS, seed=e) for e in range(E)}
    calls = []

    def get(e):
        calls.append(e)
        return live[e]

    out = torch.zeros(
        (plan.buffer_slots, W13_ROWS, _row_bytes(W13_TYPE)), dtype=torch.uint8
    )
    stage_experts_into_tiers(plan, get, out, release=live.pop)

    assert calls == list(plan.resident_ids) + list(plan.spill_ids)
    assert len(calls) == E  # exactly once each
    assert live == {}  # every source tensor released as it was consumed


def test_expert_rows_are_whole_ggml_blocks():
    """#82/#109: per-expert accounting must respect block boundaries.

    The offload only ever slices dim 0 (the expert axis), which carries no
    block structure. This pins the arithmetic that makes that safe: a row is an
    integral number of blocks, so an expert's byte count is too, and no tier
    boundary can land inside a block.
    """
    for qtype, rows in ((W13_TYPE, W13_ROWS), (W2_TYPE, W2_ROWS)):
        elems_per_block, block_bytes = GGML_QUANT_SIZES[qtype]
        assert elems_per_block == 256  # K-quant super-block
        assert _row_bytes(qtype) % block_bytes == 0
        expert = _expert_bytes(qtype, rows, seed=1)
        assert expert.numel() % block_bytes == 0
        # ... and the reference decoder agrees on the element count.
        assert _dequant(expert).shape == (rows, BLOCKS_PER_ROW * elems_per_block)


# --------------------------------------------------------------------------
# materialize_gguf_weights: the interception point
# --------------------------------------------------------------------------


class _QType:
    def __init__(self, weight_type):
        self.weight_type = int(weight_type)


class _FakeGGUFMoEMethod:
    pass


_FakeGGUFMoEMethod.__name__ = "GGUFMoEMethod"


def _gguf_layer(num_owned=E, expert_shard=False, declared_w2_type=None):
    """A minimal stand-in for a loaded GGUF FusedMoE layer.

    It borrows the real ``FusedMoE`` methods under test rather than
    reimplementing them, and carries only the attributes they read: the two
    uninitialized GGUF parameters with their loader-populated
    ``expert_data_map``, the ggml type holders, and the shard flags.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter

    class _StubGGUFMoELayer(torch.nn.Module):
        materialize_gguf_weights = FusedMoE.materialize_gguf_weights
        # ``FusedMoE._gguf_expert_source`` unwraps to a plain function on
        # attribute access; re-wrap so the stub keeps it a staticmethod.
        _gguf_expert_source = staticmethod(FusedMoE._gguf_expert_source)
        _gguf_moe_offload_eligible = FusedMoE._gguf_moe_offload_eligible
        _gguf_moe_offload_eligible_uncached = (
            FusedMoE._gguf_moe_offload_eligible_uncached
        )
        _finish_gguf_moe_offload_staging = FusedMoE._finish_gguf_moe_offload_staging
        # #391c streaming door. Present on the stub so the pull-path tests
        # below exercise the SAME materialize_gguf_weights the server runs,
        # drain call included.
        _GGUF_SHARD_TO_ATTR = FusedMoE._GGUF_SHARD_TO_ATTR
        _drain_gguf_stream_stagers = FusedMoE._drain_gguf_stream_stagers
        _gguf_cold_shard_context = FusedMoE._gguf_cold_shard_context

    layer = _StubGGUFMoELayer()
    layer.layer_id = 7
    layer.quant_method = _FakeGGUFMoEMethod()
    layer.num_experts = num_owned
    layer.num_local_experts = num_owned
    layer._gguf_expert_shard = expert_shard
    layer._gguf_expert_range = (0, num_owned)
    layer.w13_qweight_type = _QType(W13_TYPE)
    # ``declared_w2_type`` decouples the ggml type the layer ADVERTISES from
    # the byte layout the synthetic tensors use, so an uncovered type (MXFP4 /
    # 39, which gguf-py cannot even size) can be exercised without inventing
    # its block format.
    layer.w2_qweight_type = _QType(
        W2_TYPE if declared_w2_type is None else declared_w2_type
    )

    half = W13_ROWS // 2
    sources = {"w13_qweight": {}, "w2_qweight": {}}
    for attr, rows, qtype in (
        ("w13_qweight", W13_ROWS, W13_TYPE),
        ("w2_qweight", W2_ROWS, W2_TYPE),
    ):
        param = GGUFUninitializedParameter(requires_grad=False)
        param.is_gguf_weight = True
        param.tensor_shape = (num_owned, rows, _row_bytes(qtype))
        param.data_container = []
        param.expert_data_map = {}
        for e in range(num_owned):
            full = _expert_bytes(
                qtype, rows, seed=1000 + e + (0 if attr[1] == "1" else 500)
            )
            sources[attr][e] = full
            if attr == "w13_qweight":
                param.expert_data_map[(e, "w1")] = full[:half].clone()
                param.expert_data_map[(e, "w3")] = full[half:].clone()
            else:
                param.expert_data_map[(e, "w2")] = full
            param.data_container.append(full)
        layer.register_parameter(attr, param)
    layer._test_sources = sources
    return layer


def test_expert_source_drop_releases_both_holders():
    """``drop`` must clear the parameter's map too, or nothing is freed.

    ``expert_weights`` is a by-expert VIEW built from
    ``param.expert_data_map``; both hold a reference to the same tensor, so
    releasing one of them keeps the bytes alive in the other and the staging's
    "one expert above the two tiers" host bound silently becomes "the whole
    loaded set above the two tiers".
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    w = _expert_bytes(W2_TYPE, W2_ROWS, seed=42)
    expert_data_map = {(0, "w2"): w, (1, "w2"): w.clone()}
    expert_weights = {
        0: {"w2": expert_data_map[(0, "w2")]},
        1: {"w2": expert_data_map[(1, "w2")]},
    }

    count, _, _, get, drop = FusedMoE._gguf_expert_source(
        "w2_qweight", expert_weights, 2, False, expert_data_map=expert_data_map
    )
    assert count == 2
    assert torch.equal(get(0), w)
    drop(0)
    assert 0 not in expert_weights
    assert (0, "w2") not in expert_data_map
    assert (1, "w2") in expert_data_map  # untouched


def test_materialize_default_path_still_builds_the_full_stack(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    layer = _gguf_layer()
    layer.materialize_gguf_weights()

    assert not hasattr(layer, "_moe_offload_presplit")
    assert not getattr(layer, "_moe_offload_gguf_staged", False)
    assert layer.w13_qweight.shape == (E, W13_ROWS, _row_bytes(W13_TYPE))
    assert layer.w2_qweight.shape == (E, W2_ROWS, _row_bytes(W2_TYPE))
    for e in range(E):
        assert torch.equal(
            layer.w13_qweight.data[e], layer._test_sources["w13_qweight"][e]
        )
        assert torch.equal(
            layer.w2_qweight.data[e], layer._test_sources["w2_qweight"][e]
        )


def test_materialize_stages_into_two_tiers(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    reset_expert_offload_release()
    layer = _gguf_layer()
    layer.materialize_gguf_weights()

    assert layer._moe_offload_gguf_staged is True
    assert layer._moe_offload_full_experts == E
    assert set(layer._moe_offload_presplit) == {"w13_qweight", "w2_qweight"}
    # The parameter itself is the [R+C] buffer -- the [E] stack never existed.
    assert layer.w13_qweight.shape == (5, W13_ROWS, _row_bytes(W13_TYPE))
    assert layer.w2_qweight.shape == (5, W2_ROWS, _row_bytes(W2_TYPE))

    for attr in ("w13_qweight", "w2_qweight"):
        buf, spill = layer._moe_offload_presplit[attr]
        src = layer._test_sources[attr]
        for slot in range(3):
            assert torch.equal(buf[slot], src[slot])
        for row in range(E - 3):
            assert torch.equal(spill[row], src[3 + row])


def test_materialize_pins_the_padding_expert_under_expert_shard(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    reset_expert_offload_release()
    layer = _gguf_layer(num_owned=E - 1, expert_shard=True)
    layer.materialize_gguf_weights()

    resident_ids, spill_ids = layer._moe_offload_frozen_layout
    assert layer._moe_offload_full_experts == E  # E-1 owned + 1 zero pad
    assert resident_ids[0] == E - 1  # the pad expert took slot 0
    buf, _ = layer._moe_offload_presplit["w13_qweight"]
    assert torch.equal(buf[0], torch.zeros_like(buf[0]))
    assert sorted(resident_ids + spill_ids) == list(range(E))


def test_materialize_declines_uncovered_ggml_type(monkeypatch):
    """MXFP4 (type 39) has no GGUF MoE kernel: no staging, no marker."""
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    layer = _gguf_layer(declared_w2_type=39)
    layer.materialize_gguf_weights()

    assert not getattr(layer, "_moe_offload_gguf_staged", False)
    assert not hasattr(layer, "_moe_offload_presplit")
    assert layer.w13_qweight.shape[0] == E  # fell back to the full stack


# --------------------------------------------------------------------------
# #268 guard, both directions
# --------------------------------------------------------------------------


def test_guard_admits_a_staged_gguf_layer(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv(SCRATCH_ENV, "2")
    layer = _gguf_layer()
    layer.materialize_gguf_weights()
    assert (
        assert_expert_offload_quant_supported(
            layer.quant_method, layer_id=7, layer=layer
        )
        is None
    )


def test_guard_still_refuses_an_unstaged_gguf_layer(monkeypatch):
    """Can-fail proof: same quant method, uncovered ggml type -> still hard."""
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    layer = _gguf_layer(declared_w2_type=39)
    layer.materialize_gguf_weights()
    with pytest.raises(RuntimeError) as excinfo:
        assert_expert_offload_quant_supported(
            layer.quant_method, layer_id=7, layer=layer
        )
    msg = str(excinfo.value)
    assert "GGUFMoEMethod" in msg
    assert "_moe_offload_gguf_staged" in msg
    assert "MXFP4" in msg


def test_guard_refuses_gguf_without_a_layer():
    """No layer to inspect => not staged => refuse (the pre-#123 behavior)."""
    with pytest.raises(RuntimeError):
        assert_expert_offload_quant_supported(_FakeGGUFMoEMethod(), layer_id=1)


def test_guard_still_refuses_the_ascend_gguf_method():
    class _FakeGGUFMoEAscendMethod:
        pass

    _FakeGGUFMoEAscendMethod.__name__ = "GGUFMoEAscendMethod"
    layer = torch.nn.Module()
    layer._moe_offload_gguf_staged = True  # marker must NOT help here
    with pytest.raises(RuntimeError, match="GGUFMoEAscendMethod"):
        assert_expert_offload_quant_supported(
            _FakeGGUFMoEAscendMethod(), layer_id=2, layer=layer
        )


# --------------------------------------------------------------------------
# round trip: two tiers -> fetch -> remap -> dequant == all-resident reference
# --------------------------------------------------------------------------


class _TopK:
    def __init__(self, topk_ids, topk_weights):
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.router_logits = None

    def _replace(self, **kw):
        out = _TopK(self.topk_ids, self.topk_weights)
        out.router_logits = self.router_logits
        for k, v in kw.items():
            setattr(out, k, v)
        return out

    def __iter__(self):
        return iter((self.topk_weights, self.topk_ids, self.router_logits))


class _Dispatch:
    def __init__(self, hidden_states, topk_output):
        self.hidden_states = hidden_states
        self.hidden_states_scale = None
        self.topk_output = topk_output

    def _replace(self, **kw):
        out = _Dispatch(self.hidden_states, self.topk_output)
        out.hidden_states_scale = self.hidden_states_scale
        for k, v in kw.items():
            setattr(out, k, v)
        return out


class _Combine:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

    def _replace(self, hidden_states):
        return _Combine(hidden_states)


def _reference_apply(w13, w2):
    """A stand-in for ``fused_moe_gguf`` that reads the SAME expert axis.

    It dequantizes the routed expert's blocks with gguf-py and contracts them
    against the token, which is all the round trip needs: if a single byte of a
    tiered expert landed in the wrong slot, the dequantized rows differ and the
    output differs.
    """

    def apply_fn(dispatch):
        x = dispatch.hidden_states
        ids = dispatch.topk_output.topk_ids
        weights = dispatch.topk_output.topk_weights
        # Same dtype as the hidden states, like every real MoE apply: the
        # multi-wave path scatters into a torch.empty_like(hidden) buffer, so a
        # wider accumulator here would show up as a wave-order artifact rather
        # than as the byte-identity this test is about.
        out = torch.zeros(x.shape, dtype=x.dtype)
        for t in range(x.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if e < 0:
                    continue
                up = _dequant(w13()[e])
                down = _dequant(w2()[e])
                rows = torch.cat([up, down], dim=0) * x[t]
                out[t] += rows.sum(0) * float(weights[t, k])
        return _Combine(out)

    return apply_fn


def _routing(num_tokens, top_k, num_experts, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int64)
    weights = torch.rand(num_tokens, top_k, generator=g, dtype=torch.float32)
    return ids, weights


def _reference_output(num_owned, expert_shard, x, ids, weights, monkeypatch):
    """All-resident arm: fraction 1.0, one full ``[E]`` stack, no offload."""
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    ref_layer = _gguf_layer(num_owned=num_owned, expert_shard=expert_shard)
    ref_layer.materialize_gguf_weights()
    assert not hasattr(ref_layer, "_moe_offload_presplit")
    return _reference_apply(
        lambda: ref_layer.w13_qweight.data, lambda: ref_layer.w2_qweight.data
    )(_Dispatch(x, _TopK(ids, weights))).hidden_states


@pytest.mark.parametrize("expert_shard", [False, True])
def test_round_trip_matches_the_all_resident_reference(monkeypatch, expert_shard):
    """Single wave (decode shape): tiered bytes reproduce the reference exactly."""
    monkeypatch.setenv(SCRATCH_ENV, "6")
    num_owned = E - 1 if expert_shard else E
    total_experts = num_owned + (1 if expert_shard else 0)

    x = torch.randn(
        3,
        BLOCKS_PER_ROW * 256,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(3),
    )
    ids, weights = _routing(3, 2, total_experts, seed=4)
    reference = _reference_output(num_owned, expert_shard, x, ids, weights, monkeypatch)

    # Offload arm: fraction 0.25, two tiers, run_waves drives fetch + remap.
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    reset_expert_offload_release()
    layer = _gguf_layer(num_owned=num_owned, expert_shard=expert_shard)
    layer.materialize_gguf_weights()
    assert layer._moe_offload_gguf_staged

    cache = MoEExpertOffloadCache(layer, 0.25)
    cache.install()
    assert cache.num_local_experts == total_experts
    got = cache.run_waves(
        _Dispatch(x, _TopK(ids, weights)),
        _reference_apply(lambda: layer.w13_qweight.data, lambda: layer.w2_qweight.data),
    ).hidden_states

    assert torch.equal(got, reference)
    assert cache.planner.stats.overflow_forwards == 0  # single wave
    # The offload actually did work (otherwise the equality is vacuous).
    assert cache.planner.stats.fetches > 0
    assert cache.planner.stats.h2d_bytes > 0
    assert layer.w13_qweight.shape[0] < total_experts


def test_round_trip_multi_wave(monkeypatch):
    """More unique experts per forward than scratch slots -> token waves."""
    monkeypatch.setenv(SCRATCH_ENV, "2")
    x = torch.randn(
        8,
        BLOCKS_PER_ROW * 256,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(5),
    )
    ids, weights = _routing(8, 2, E, seed=6)
    reference = _reference_output(E, False, x, ids, weights, monkeypatch)

    monkeypatch.setenv(FRACTION_ENV, "0.25")
    reset_expert_offload_release()
    layer = _gguf_layer()
    layer.materialize_gguf_weights()
    cache = MoEExpertOffloadCache(layer, 0.25)
    cache.install()
    got = cache.run_waves(
        _Dispatch(x, _TopK(ids, weights)),
        _reference_apply(lambda: layer.w13_qweight.data, lambda: layer.w2_qweight.data),
    ).hidden_states

    assert cache.planner.stats.overflow_forwards >= 1
    assert torch.equal(got, reference)


# --------------------------------------------------------------------------
# #390 instrument
# --------------------------------------------------------------------------


def test_expert_stats_count_on_the_gguf_path(monkeypatch):
    from sglang.srt.layers.moe import expert_stats

    monkeypatch.setenv(SCRATCH_ENV, "4")
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    monkeypatch.setenv("SGLANG_EXPERT_STATS", "1")
    expert_stats.reset_for_tests()
    reset_expert_offload_release()
    try:
        layer = _gguf_layer()
        layer.materialize_gguf_weights()
        cache = MoEExpertOffloadCache(layer, 0.25)
        cache.install()
        assert cache._router_stats is not None, "stats not wired on the GGUF path"

        x = torch.randn(
            4, BLOCKS_PER_ROW * 256, generator=torch.Generator().manual_seed(7)
        )
        ids, weights = _routing(4, 3, E, seed=8)
        cache.run_waves(
            _Dispatch(x, _TopK(ids, weights)),
            _reference_apply(
                lambda: layer.w13_qweight.data, lambda: layer.w2_qweight.data
            ),
        )

        stats = cache._router_stats
        assert stats.forwards == 1
        assert stats.tokens == 4
        assert stats.activations == ids.numel()
        # Residency is the plan's, so hits are exactly the routed ids < R.
        expected_hits = int((ids < cache.resident_count).sum())
        assert stats.hit_activations == expected_hits
        assert stats.miss_activations == ids.numel() - expected_hits
        assert sum(stats.expert_activations) == ids.numel()
        # The planner's own fetch tally is handed to the dump (#390).
        assert stats.residency is cache.planner.stats
    finally:
        expert_stats.reset_for_tests()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
