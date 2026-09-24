"""H68b (d): ``prepare_moe_nvfp4_layer_for_marlin_inplace`` -- the NVFP4 MoE
Marlin repack PER EXPERT into one preallocated output, Parameter identity
kept, global scales as ``[E, 1]``.

What is pinned, hermetically (CPU, the CUDA repack kernel replaced by a
deterministic stand-in -- the SAME stand-in for old and new, so the comparison
is of the surrounding logic, which is what changed):

* old (list + ``torch.stack``, fresh Parameters) and new produce the SAME
  bytes for every tensor; the global scales differ only in shape ``[E]`` vs
  ``[E, 1]`` and :func:`nvfp4_marlin_global_scale_1d` gives the kernel the old
  ``[E]`` values back as a view;
* the new path keeps every Parameter OBJECT (upstream #38074: a fresh
  Parameter left the old [E] stack alive in the loader's params_dict);
* the new path reads host-staged experts expert by expert (source on the
  host, outputs on the compute device) and never builds an [E] list.
"""

import types

import pytest

try:
    import torch

    from sglang.srt.layers.quantization import marlin_utils_fp4 as M
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

E, H, I = 3, 128, 64  # experts, hidden, moe intermediate


def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
    """Shape-true stand-in for gptq_marlin_repack: [K/8, N] -> [K/16, 2N] int32,
    content-dependent so a mixed-up expert shows."""
    assert num_bits == 4 and b_q_weight.shape == (size_k // 8, size_n)
    return (b_q_weight.reshape(size_k // 16, size_n * 2) * 3 + 1).contiguous()


def fake_workspace(device, max_blocks_per_sm=1):
    return torch.zeros(8, dtype=torch.int, device=device)


def make_layer(seed=0):
    g = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    layer.quant_config = types.SimpleNamespace(group_size=16)
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
    layer.intermediate_size_per_partition = I
    layer.params_dtype = torch.bfloat16

    def P(t):
        return torch.nn.Parameter(t, requires_grad=False)

    layer.w13_weight = P(torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, dtype=torch.uint8))
    layer.w2_weight = P(torch.randint(0, 256, (E, H, I // 2), generator=g, dtype=torch.uint8))
    layer.w13_weight_scale = P(
        (torch.rand(E, 2 * I, H // 16, generator=g) * 4 + 0.25).to(torch.float8_e4m3fn))
    layer.w2_weight_scale = P(
        (torch.rand(E, H, I // 16, generator=g) * 4 + 0.25).to(torch.float8_e4m3fn))
    # already collapsed to the gate column by process_weights_after_loading
    layer.w13_weight_scale_2 = P(torch.rand(E, generator=g) * 1e-3 + 1e-5)
    layer.w2_weight_scale_2 = P(torch.rand(E, generator=g) * 1e-3 + 1e-5)
    return layer


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(M, "gptq_marlin_repack", fake_repack, raising=False)
    monkeypatch.setattr(M, "marlin_make_workspace", fake_workspace)


def test_same_bytes_as_the_stacked_path(cpu_kernels):
    old, new = make_layer(), make_layer()
    M.prepare_moe_nvfp4_layer_for_marlin(old)
    M.prepare_moe_nvfp4_layer_for_marlin_inplace(new, device=torch.device("cpu"))
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        a, b = getattr(old, name).data, getattr(new, name).data
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
        a, b = getattr(old, name).data, getattr(new, name).data
        assert tuple(a.shape) == (E,) and tuple(b.shape) == (E, 1)
        view = M.nvfp4_marlin_global_scale_1d(b)
        assert view.data_ptr() == b.data_ptr(), "a view, never a copy"
        assert torch.equal(a, view)
    assert new.workspace is not None


def test_parameter_objects_are_kept(cpu_kernels):
    layer = make_layer()
    before = {n: id(p) for n, p in layer.named_parameters()}
    M.prepare_moe_nvfp4_layer_for_marlin_inplace(layer, device=torch.device("cpu"))
    after = {n: id(p) for n, p in layer.named_parameters()}
    assert before == after
    # and the stacked path is the one that did NOT keep them (#38074)
    old = make_layer()
    before_old = {n: id(p) for n, p in old.named_parameters()}
    M.prepare_moe_nvfp4_layer_for_marlin(old)
    assert any(before_old[n] != id(p) for n, p in old.named_parameters())


def test_reads_the_source_expert_by_expert(cpu_kernels, monkeypatch):
    """A host-staged source is indexed one expert at a time and each expert is
    moved to the compute device on its own -- no [E] list, no torch.stack."""
    layer = make_layer()
    stacks = []
    real_stack = torch.stack
    monkeypatch.setattr(torch, "stack", lambda *a, **k: stacks.append(1) or real_stack(*a, **k))
    M.prepare_moe_nvfp4_layer_for_marlin_inplace(layer, device=torch.device("cpu"))
    assert stacks == []


def test_one_dim_view_passes_other_shapes_through():
    t = torch.arange(4, dtype=torch.bfloat16)
    assert M.nvfp4_marlin_global_scale_1d(t) is t
    t2 = torch.ones(4, 2)
    assert M.nvfp4_marlin_global_scale_1d(t2) is t2


def test_group_size_other_than_16_is_refused(cpu_kernels):
    layer = make_layer()
    layer.quant_config = types.SimpleNamespace(group_size=32)
    with pytest.raises(ValueError, match="group_size=16"):
        M.prepare_moe_nvfp4_layer_for_marlin_inplace(layer, device=torch.device("cpu"))
