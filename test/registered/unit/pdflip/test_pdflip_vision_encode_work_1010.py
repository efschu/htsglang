"""VISION-WORK (10.10.): the Qwen3-VL tower's forward drops dead activations,
and the work model follows -- 4096x4096 fits the flip's air.

The bug (metal 10.10., INT8 flip, P PP0 on nvml1): a 4096x4096 still image
(65536 patches, the model's own limit) was refused with W105b
PdFlipVisionVictimShort -- "needs 2402 MiB of work memory, the card's air is
1960 MiB (free 1471 + idle cache 489)". The dual measured that encode at
+2150 MiB, which is EXACTLY the old forward's live tensors at its MLP peak
(hidden-wide row blocks, 144 MiB each at 65536 patches): residual, norm1
rows, attention output (both dead, still named in the block), norm2, fc1 out
and act out (3.74 each, fc1 still named while fc2 ran), fc2 out = 12.47, plus
the pos rows (named through all blocks), the pixel rows and the rope rows.
The victims cannot help: the tower lives as VIEWS in still-allocated weight
storages, so their bytes are never the caching allocator's.

Pinned (hermetic, CPU; the numbers are untouched -- only lifetimes change):
  * the MLP: fc1's rows are dead when fc2 runs, output bitwise as before;
  * the block: norm1's rows and the attention output are dead when the MLP
    runs, output bitwise as before;
  * VisionAttention: the fused qkv is dead at the kernel, q/k/v are dead at
    the projection, output bitwise as before;
  * the tower: the pos rows are dead before the first block;
  * the work model: the metal image now fits the metal air.
All five are RED on 23c85f1d20.
"""

import types
import weakref
from unittest import mock

import torch
import torch.nn.functional as F

from flliper.srt.layers.attention import vision as vision_mod
from flliper.srt.layers.attention.vision import VisionAttention
from flliper.srt.models import qwen3_vl
from flliper.srt.models.qwen3_vl import Qwen3_VisionBlock, Qwen3_VisionMLP, Qwen3VLMoeVisionModel
from flliper.srt.pdflip import vision_victim as vv

MIB = 1 << 20
S, HID, INTER, HEADS = 12, 8, 20, 2
HEAD_DIM = HID // HEADS


class _Lin(torch.nn.Module):
    """An flliper linear's call shape ``(out, bias or None)``; keeps a weak
    reference to every output's STORAGE (alive while any view of it is) and
    runs ``probe`` first."""

    def __init__(self, n_in: int, n_out: int, seed: int, probe=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(n_out, n_in, generator=g), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.randn(n_out, generator=g), requires_grad=False)
        self.probe = probe
        self.outs = []

    def forward(self, x):
        if self.probe is not None:
            self.probe()
        out = F.linear(x, self.weight, self.bias)
        self.outs.append(_storage_ref(out))
        return out, None


def _storage_ref(t: torch.Tensor) -> weakref.ref:
    """A weak reference to ``t``'s storage object: torch keeps one python
    object per storage, alive exactly while some tensor or view uses it."""
    return weakref.ref(t.untyped_storage())


def _bare(cls):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    return obj


def _x(*shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(1010))


def _mlp(probe=None):
    mlp = _bare(Qwen3_VisionMLP)
    mlp.linear_fc1 = _Lin(HID, INTER, 1)
    mlp.act = lambda t: F.gelu(t, approximate="tanh")
    mlp.linear_fc2 = _Lin(INTER, HID, 2, probe=probe)
    return mlp


def test_mlp_fc1_rows_are_dead_when_fc2_runs():
    dead = []
    mlp = _mlp(probe=lambda: dead.append(mlp.linear_fc1.outs[-1]() is None))
    x = _x(S, HID)
    got = mlp(x)
    w1, w2 = mlp.linear_fc1, mlp.linear_fc2
    ref = F.linear(F.gelu(F.linear(x, w1.weight, w1.bias), approximate="tanh"), w2.weight, w2.bias)
    assert dead == [True]
    assert torch.equal(got, ref)


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = _Lin(HID, HID, 3)

    def forward(self, hidden_states, **kw):
        return self.lin(hidden_states)[0]


class _Norm(torch.nn.LayerNorm):
    def __init__(self):
        super().__init__(HID)
        self.outs = []

    def forward(self, x):
        out = super().forward(x)
        self.outs.append(_storage_ref(out))
        return out


def test_block_norm1_rows_and_attention_output_are_dead_when_the_mlp_runs():
    blk = _bare(Qwen3_VisionBlock)
    blk.norm1, blk.norm2, blk.attn = _Norm(), torch.nn.LayerNorm(HID), _Attn()
    seen = []
    blk.mlp = _mlp(probe=lambda: seen.append((blk.norm1.outs[-1]() is None,
                                              blk.attn.lin.outs[-1]() is None)))
    x = _x(S, 1, HID)
    ref_x = x.clone()
    got = blk(x, cu_seqlens=None, rotary_pos_emb_cos=None, rotary_pos_emb_sin=None)
    a = F.linear(F.layer_norm(ref_x, (HID,), blk.norm1.weight, blk.norm1.bias, blk.norm1.eps),
                 blk.attn.lin.weight, blk.attn.lin.bias)
    ref_x += a
    w1, w2 = blk.mlp.linear_fc1, blk.mlp.linear_fc2
    n2 = F.layer_norm(ref_x, (HID,), blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
    ref_x += F.linear(F.gelu(F.linear(n2, w1.weight, w1.bias), approximate="tanh"), w2.weight, w2.bias)
    assert seen == [(True, True)]
    assert torch.equal(got, ref_x)


class _Backend:
    def __init__(self, on_call):
        self.on_call = on_call
        self.qkv = []

    def forward(self, q, k, v, bsz, **kw):
        self.on_call()
        self.qkv = [_storage_ref(t) for t in (q, k, v)]
        return F.scaled_dot_product_attention(*(t.transpose(0, 1) for t in (q, k, v))).transpose(0, 1)


def test_attention_qkv_is_dead_at_the_kernel_and_qkv_copies_at_the_projection():
    att = _bare(VisionAttention)
    att.num_attention_heads_per_partition = att.num_attention_kv_heads_per_partition = HEADS
    att.head_size, att.q_size, att.kv_size = HEAD_DIM, HID, HID
    att.use_qkv_parallel, att.qk_normalization, att.qk_normalization_by_head_size = True, False, False
    att.customized_position_embedding_applier, att.sinks, att.softmax_scale = None, None, None
    att.qkv_proj = _Lin(HID, 3 * HID, 4)
    at_kernel, at_proj = [], []
    att.qkv_backend = _Backend(lambda: at_kernel.append(att.qkv_proj.outs[-1]() is None))
    att.proj = _Lin(HID, HID, 5, probe=lambda: at_proj.append([r() is None for r in att.qkv_backend.qkv]))
    x = _x(1, S, HID)
    with mock.patch.object(vision_mod, "get_server_args",
                           return_value=types.SimpleNamespace(rl_on_policy_target=None)):
        got = att(x)
    qkv = F.linear(x, att.qkv_proj.weight, att.qkv_proj.bias)
    q, k, v = (t.reshape(S, HEADS, HEAD_DIM).contiguous().transpose(0, 1) for t in qkv.split([HID, HID, HID], -1))
    o = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(1, S, HID)
    ref = F.linear(o, att.proj.weight, att.proj.bias)
    assert at_kernel == [True] and at_proj == [[True, True, True]]
    assert torch.equal(got, ref)


class _Patch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, HID)

    def forward(self, x):
        return self.proj(x)


class _ProbeBlock(torch.nn.Module):
    def __init__(self, probe):
        super().__init__()
        self.probe = probe

    def forward(self, x, **kw):
        self.probe()
        return x


def test_tower_pos_rows_are_dead_before_the_first_block():
    tower = _bare(Qwen3VLMoeVisionModel)
    tower.patch_embed = _Patch()
    pos = []

    def interpolate(grid_thw_list):
        out = torch.ones(S, HID)
        pos.append(_storage_ref(out))
        return out

    tower.fast_pos_embed_interpolate_from_list = interpolate
    tower.rot_pos_emb = lambda grid: (torch.zeros(S, 1), torch.zeros(S, 1))
    seen = []
    tower.blocks = torch.nn.ModuleList([_ProbeBlock(lambda: seen.append(pos[-1]() is None))])
    tower.deepstack_visual_indexes = []
    tower.merger = torch.nn.Identity()
    with mock.patch.object(qwen3_vl, "get_server_args",
                           return_value=types.SimpleNamespace(mm_attention_backend="sdpa")):
        tower(_x(S, 3), grid_thw=torch.tensor([[1, 3, 4]]))
    assert seen == [True]


#: the 27B tower of the metal W105b (Qwen3_5ForConditionalGeneration, no deepstack)
_VC = types.SimpleNamespace(hidden_size=1152, intermediate_size=4304, num_heads=16, out_hidden_size=5120,
                            spatial_merge_size=2, deepstack_visual_indexes=[], patch_size=16,
                            in_channels=3, temporal_patch_size=2)


def test_the_metal_4096_image_fits_the_metal_flip_air():
    """W105b of 10.10.: 65536 patches against free 1471 + idle cache 489 MiB.
    The model books the leaner forward: pixels 192 + rope 18 + the MLP phase
    (residual, norm2, fc1, act = 2*1152 + 2*4304 wide) 1364 = 1574 MiB."""
    still = types.SimpleNamespace(image_grid_thw=torch.tensor([[1, 256, 256]]))
    work = vv.encode_work_for(_VC, [still], "sdpa")
    assert work == 1574 * MIB
    assert vv.encode_air_refusal(work, 1471 * MIB, 489 * MIB, 65536) == ""
