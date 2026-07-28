# SPDX-License-Identifier: Apache-2.0
"""Host-side gate for the AWQ MoE load-time offload (#123-AWQ).

``AWQMoEScheme.create_weights`` committed the whole ``[E, ...]`` AWQ expert
stack -- qweight, scales AND qzeros -- on the ambient cuda device, so a
4-bit MoE checkpoint that does not fit a card OOM'd during load, before the
expert offload ever got a chance to spill anything. The presplit half already
existed (``process_weights_after_loading`` ends in
``presplit_expert_offload_after_repack``); only the allocation side was
missing, exactly as it was for fp8 (#256) and GPTQ.

These tests pin the fix without loading a model or running a marlin kernel:

  * ``create_weights`` puts all six expert-major tensors on the host when the
    offload fraction is set, and on the default device when it is not;
  * the AWQ-specific part: ``qzeros`` follow the weights. AWQ is asymmetric, so
    the zero-points carry real checkpoint data that the repack consumes and the
    marlin apply reads per expert -- unlike GPTQ, where they stay empty;
  * ``process_weights_after_loading`` runs the repack first and ends in the
    presplit;
  * the presplit stages the post-repack AWQ tensor names and puts every tensor
    of one expert -- qweight, scales, qzeros -- on the SAME spill row, which is
    what makes the shared fetch plan correct.

The presplit uses ``pin_memory()`` and therefore needs a CUDA context; those
tests are skipped without one. No GPU memory is allocated and no kernel runs
(all tensors stay on the host).

Run:
  PYTHONPATH=python python -m pytest tests/moe_offload/test_awq_presplit.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    MoEExpertOffloadCache,
    presplit_expert_offload_after_repack,
    reset_expert_offload_release,
    resident_slot_count,
)

FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the pinned spill pool needs a CUDA context (host tensors only)",
)

# Qwen3.6-35B-A3B-AWQ-4bit geometry, scaled down on every axis so the test is
# cheap. group_size must divide both hidden and intermediate; pack_factor 8 is
# the 4-bit int32 packing.
E = 32
HIDDEN = 256
INTER = 128
GROUP = 64
PACK = 8

# The six expert-major tensors create_weights allocates.
AWQ_ATTRS = (
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
)


class _StubAWQConfig:
    weight_bits = 4
    group_size = GROUP
    pack_factor = PACK


def _scheme():
    """An AWQMoEScheme with just the fields create_weights reads.

    ``__new__`` skips ``_init_kernel``, which would import the GPU marlin
    kernels; create_weights never touches ``self.kernel``.
    """
    from sglang.srt.layers.quantization.awq.schemes.awq_moe import AWQMoEScheme

    scheme = AWQMoEScheme.__new__(AWQMoEScheme)
    scheme.quant_config = _StubAWQConfig()
    return scheme


def _create_weights(scheme, num_experts=E, ambient="meta"):
    """Run create_weights with a non-CPU ambient default device.

    The real loader builds the model inside a ``torch.device("cuda")`` context,
    so ``device=None`` means "on the card". A test that just calls
    create_weights on a CPU-only host would assert ``device.type == "cpu"``
    against a default that is already cpu -- vacuously green on the unfixed
    code. ``meta`` is a device that is not cpu and costs no memory, so it makes
    the host/default distinction observable without a GPU; the cuda variant
    below runs the same check against the real thing when a card is present.
    """
    layer = torch.nn.Module()
    layer.num_local_experts = num_experts
    with torch.device(ambient):
        scheme.create_weights(
            layer,
            num_experts=num_experts,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTER,
            params_dtype=torch.float16,
            weight_loader=lambda *a, **k: None,
        )
    return layer


def _repacked_awq_layer(num_experts=E):
    """A stub FusedMoE layer carrying post-repack AWQ marlin expert tensors.

    Shapes do not have to be the true marlin layout -- the presplit only cares
    that dim 0 is the expert axis. Rows are filled with the expert id so a
    misrouted row is visible as a value, not just as a shape.
    """
    layer = torch.nn.Module()
    layer.num_local_experts = num_experts

    def _p(*shape, dtype=torch.float16):
        t = torch.empty(shape, dtype=dtype)
        for e in range(num_experts):
            t[e] = e
        return torch.nn.Parameter(t, requires_grad=False)

    layer.register_parameter(
        "w13_qweight", _p(num_experts, HIDDEN // 16, 2 * INTER * 2, dtype=torch.int32)
    )
    layer.register_parameter(
        "w2_qweight", _p(num_experts, INTER // 16, HIDDEN * 2, dtype=torch.int32)
    )
    layer.register_parameter("w13_scales", _p(num_experts, HIDDEN // GROUP, 2 * INTER))
    layer.register_parameter("w2_scales", _p(num_experts, INTER // GROUP, HIDDEN))
    layer.register_parameter(
        "w13_qzeros",
        _p(num_experts, HIDDEN // GROUP, 2 * INTER // PACK, dtype=torch.int32),
    )
    layer.register_parameter(
        "w2_qzeros",
        _p(num_experts, INTER // GROUP, HIDDEN // PACK, dtype=torch.int32),
    )
    return layer


@pytest.fixture
def fraction_025(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    reset_expert_offload_release()
    yield 0.25
    reset_expert_offload_release()


# --------------------------------------------------------------------------
# create_weights residence -- the actual #123-AWQ fix
# --------------------------------------------------------------------------


def test_create_weights_puts_the_expert_stack_on_the_host_when_offloading(
    monkeypatch, fraction_025
):
    layer = _create_weights(_scheme())
    for attr in AWQ_ATTRS:
        assert getattr(layer, attr).device.type == "cpu", attr
    # shapes are the stock ones -- only the residence changed
    assert layer.w13_qweight.shape == (E, HIDDEN, 2 * INTER // PACK)
    assert layer.w2_qweight.shape == (E, INTER, HIDDEN // PACK)
    assert layer.w13_scales.shape == (E, HIDDEN // GROUP, 2 * INTER)
    assert layer.w2_scales.shape == (E, INTER // GROUP, HIDDEN)


def test_qzeros_follow_the_weights_to_the_host(monkeypatch, fraction_025):
    """AWQ-specific. GPTQ leaves qzeros on the default device because they stay
    empty (symmetric, desc_act=False). AWQ zero-points hold real checkpoint data
    of full [E] size and are read per expert by the marlin apply, so leaving
    them behind would keep an expert-major tensor on GPU."""
    layer = _create_weights(_scheme())
    assert layer.w13_qzeros.device.type == "cpu"
    assert layer.w2_qzeros.device.type == "cpu"
    assert layer.w13_qzeros.shape[0] == E
    assert layer.w2_qzeros.shape[0] == E
    # and the offload cache does stage them, so their residence matters
    assert "w13_qzeros" in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS
    assert "w2_qzeros" in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS


def test_create_weights_default_path_uses_the_default_device(monkeypatch):
    monkeypatch.delenv(FRACTION_ENV, raising=False)
    layer = _create_weights(_scheme())
    for attr in AWQ_ATTRS:
        assert getattr(layer, attr).device.type == "meta", attr


def test_create_weights_at_fraction_one_is_the_stock_path(monkeypatch):
    """fraction == 1.0 must not take the host branch: the gate is `< 1.0`."""
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    layer = _create_weights(_scheme())
    for attr in AWQ_ATTRS:
        assert getattr(layer, attr).device.type == "meta", attr


@needs_cuda
def test_create_weights_on_a_real_cuda_ambient_device(monkeypatch):
    """Same distinction as the meta tests, against the device the loader really
    sets. Allocates the (small) stack once per branch and frees it."""
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    layer = _create_weights(_scheme(), ambient="cuda")
    for attr in AWQ_ATTRS:
        assert getattr(layer, attr).device.type == "cpu", attr
    del layer

    monkeypatch.delenv(FRACTION_ENV, raising=False)
    layer = _create_weights(_scheme(), ambient="cuda")
    for attr in AWQ_ATTRS:
        assert getattr(layer, attr).device.type == "cuda", attr
    del layer
    torch.cuda.empty_cache()


def test_weight_attrs_survive_the_device_change(monkeypatch, fraction_025):
    """The host allocation must not cost the loader metadata -- weight_loader
    and the transposed/group markers are what routes checkpoint shards."""
    layer = _create_weights(_scheme())
    for attr in AWQ_ATTRS:
        p = getattr(layer, attr)
        assert getattr(p, "is_transposed", None) is True, attr
        assert getattr(p, "quant_method", None) == "group", attr
        assert getattr(p, "weight_loader", None) is not None, attr


# --------------------------------------------------------------------------
# process_weights_after_loading wiring
# --------------------------------------------------------------------------


def test_repack_runs_before_the_presplit(monkeypatch):
    """Order gate: the marlin repack replaces every expert tensor, so it has to
    be finished when the presplit stages them."""
    from sglang.srt.layers.moe import expert_offload as eo_mod
    from sglang.srt.layers.quantization.awq.schemes.awq_moe import AWQMoEScheme

    scheme = AWQMoEScheme.__new__(AWQMoEScheme)
    scheme.quant_config = _StubAWQConfig()
    order = []

    class _Kernel:
        def process_weights_after_loading(self, layer):
            order.append("repack")

    scheme.kernel = _Kernel()
    monkeypatch.setattr(
        eo_mod,
        "presplit_expert_offload_after_repack",
        lambda lyr: order.append("presplit"),
    )

    scheme.process_weights_after_loading(torch.nn.Module())
    assert order == ["repack", "presplit"]


# --------------------------------------------------------------------------
# presplit mechanics on AWQ tensor names
# --------------------------------------------------------------------------


@needs_cuda
def test_presplit_stages_all_six_awq_tensors(fraction_025):
    layer = _repacked_awq_layer()
    before = {a: getattr(layer, a).data.clone() for a in AWQ_ATTRS}

    presplit_expert_offload_after_repack(layer)

    R = resident_slot_count(E, 0.25)
    presplit = layer._moe_offload_presplit
    assert set(presplit) == set(AWQ_ATTRS)
    assert layer._moe_offload_full_experts == E

    for attr, (buf, spill) in presplit.items():
        assert spill.shape[0] == E - R, attr
        assert spill.is_pinned(), attr
        assert torch.equal(
            buf[:R].view(torch.uint8), before[attr][:R].view(torch.uint8)
        ), attr
        # the registered param is a placeholder: it must not keep the stack alive
        assert getattr(layer, attr).shape[0] == 0, attr


@needs_cuda
def test_qzeros_share_the_expert_row_with_their_weight(fraction_025):
    """The spill pool is indexed by one shared fetch plan (expert -> row). An
    AWQ expert's qweight, scales and zero-points must land on the same row, or
    the marlin apply pairs a weight with a foreign expert's zero-points."""
    layer = _repacked_awq_layer()
    before = {a: getattr(layer, a).data.clone() for a in AWQ_ATTRS}

    presplit_expert_offload_after_repack(layer)

    R = resident_slot_count(E, 0.25)
    presplit = layer._moe_offload_presplit
    for expert in range(R, E):
        row = expert - R  # the index _fetch uses, identically for every attr
        for attr, (_buf, spill) in presplit.items():
            assert torch.equal(
                spill[row].view(torch.uint8), before[attr][expert].view(torch.uint8)
            ), f"{attr}: expert {expert} did not land on pool row {row}"


@needs_cuda
def test_presplit_is_a_noop_without_the_offload_flag(monkeypatch):
    monkeypatch.delenv(FRACTION_ENV, raising=False)
    layer = _repacked_awq_layer()
    ptrs = {a: getattr(layer, a).data_ptr() for a in AWQ_ATTRS}
    shapes = {a: getattr(layer, a).shape for a in AWQ_ATTRS}

    presplit_expert_offload_after_repack(layer)

    assert not hasattr(layer, "_moe_offload_presplit")
    for attr, ptr in ptrs.items():
        assert getattr(layer, attr).data_ptr() == ptr, attr
        assert getattr(layer, attr).shape == shapes[attr], attr


@needs_cuda
def test_cache_reads_the_full_expert_count_from_the_stash(fraction_025):
    """After the presplit, the AWQ params are 0-row placeholders; the cache must
    size itself from _moe_offload_full_experts, not from the placeholder."""
    layer = _repacked_awq_layer()
    layer.moe_runner_config = None
    presplit_expert_offload_after_repack(layer)
    layer.num_local_experts = 0  # what a shape-derived reader would see

    cache = MoEExpertOffloadCache(layer, 0.25)
    assert cache.num_local_experts == E
    assert cache.resident_count == resident_slot_count(E, 0.25)
