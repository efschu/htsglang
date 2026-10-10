"""VISION-ENCODER (09.10.): the sdpa vision attention drops its s*s mask when
cu_seqlens is ONE segment over all of s, and the work model follows.

The bug (metal, 09.10.): a 4096x4096 image (65536 patches) was refused with
W105b PdFlipVisionVictimShort -- ``encode_work_bytes`` booked 14690 MiB, 12288 MiB
of it the ``quad`` term 3*p*p: ``VisionSdpaAttention.forward`` always built the
block mask from cu_seqlens (a [1, s, s] bool on the HOST via
``_generate_mask_cache``, its additive form on the device), even for a still
image, where the one segment [0, s] makes that mask all True -- the same
attention as no mask, which lets torch take an O(s) kernel.

Pinned (hermetic, CPU):
  * the all-True claim itself for both ``_generate_mask_cache`` branches, and
    that the maskless result equals the masked one (fp32 and bf16);
  * the call shape: one segment -> ``attn_mask=None`` and no host mask built;
    two segments, a segment short of s, or the single-precision softmax path
    keep the mask exactly as before;
  * the work model: a still image (one segment) has no quad term, a clip
    (t=2) or two grid rows in one item keep it, triton_attn never had it.
"""

import types
from unittest import mock

import pytest
import torch

from flliper.srt.layers.attention import vision as vision_mod
from flliper.srt.layers.attention.vision import VisionSdpaAttention
from flliper.srt.pdflip import vision_victim as vv

HEADS, HEAD_DIM = 4, 72  # head_dim of the Qwen-VL tower (1152 / 16)


def _qkv(s: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(1009)
    return [torch.randn(s, HEADS, HEAD_DIM, generator=g).to(dtype) for _ in range(3)]


def _attn(flatten_batch: bool, softmax_in_single_precision: bool = False) -> VisionSdpaAttention:
    return VisionSdpaAttention(head_dim=HEAD_DIM, num_heads=HEADS, num_kv_heads=HEADS,
                               flatten_batch=flatten_batch,
                               softmax_in_single_precision=softmax_in_single_precision)


@pytest.mark.parametrize("flatten_batch", [True, False])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_one_segment_mask_is_all_true_and_dropping_it_changes_nothing(flatten_batch, dtype, tol):
    """Derived property: for cu_seqlens == [0, s] both branches of
    ``_generate_mask_cache`` return an all-True mask, so attention without a
    mask is the same attention."""
    s = 96
    full = VisionSdpaAttention._generate_mask_cache(s, flatten_batch, (0, s))
    assert bool(full.all()) and tuple(full.shape[-2:]) == (s, s)
    q, k, v = _qkv(s, dtype)
    cu = torch.tensor([0, s], dtype=torch.int32)
    attn = _attn(flatten_batch)
    got = attn(q, k, v, bsz=1, cu_seqlens=cu)
    ref = attn(q, k, v, bsz=1, cu_seqlens=cu, attention_mask=full)
    assert got.shape == (s, HEADS, HEAD_DIM) and got.dtype == dtype
    torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol)


def _sdpa_calls(attn: VisionSdpaAttention, s: int, cu: list):
    """(attn_mask handed to SDPA, or 'no-sdpa'; host-mask builds) of one forward."""
    real = VisionSdpaAttention._generate_mask_cache
    q, k, v = _qkv(s, torch.float32)
    seen = []

    def fake_sdpa(q, k, v, attn_mask=None, **kw):
        seen.append(attn_mask)
        return torch.zeros_like(q)

    with mock.patch.object(VisionSdpaAttention, "_generate_mask_cache", side_effect=real) as gen, \
            mock.patch.object(vision_mod.F, "scaled_dot_product_attention", side_effect=fake_sdpa):
        attn(q, k, v, bsz=1, cu_seqlens=torch.tensor(cu, dtype=torch.int32))
    return (seen[0] if seen else "no-sdpa"), gen.call_count


def test_one_segment_passes_no_mask_and_builds_no_host_mask():
    for flatten_batch in (True, False):
        mask, host_builds = _sdpa_calls(_attn(flatten_batch), 64, [0, 64])
        assert mask is None and host_builds == 0


def test_masks_that_are_not_all_true_stay():
    """Negative branches: two segments (a clip, t=2) keep the block mask; a
    single segment short of s keeps its mask; the single-precision softmax
    path needs a mask and keeps it even for one segment."""
    mask, host_builds = _sdpa_calls(_attn(True), 8, [0, 3, 8])
    want = torch.zeros(1, 8, 8, dtype=torch.bool)
    want[:, :3, :3] = True
    want[:, 3:, 3:] = True
    assert host_builds == 1 and torch.equal(mask, want)

    mask, host_builds = _sdpa_calls(_attn(True), 8, [0, 5])
    assert host_builds == 1 and tuple(mask.shape) == (1, 8, 8) and not bool(mask.all())

    mask, host_builds = _sdpa_calls(_attn(True, softmax_in_single_precision=True), 8, [0, 8])
    assert mask == "no-sdpa" and host_builds == 1


# --------------------------------------------------------------- work model --

#: the tower of the metal W105b (no deepstack mergers: its 4096x4096 booking
#: was 14690 MiB = 2402 linear + 12288 quad)
_VC = types.SimpleNamespace(hidden_size=1152, intermediate_size=4304, num_heads=16, out_hidden_size=5120,
                            spatial_merge_size=2, deepstack_visual_indexes=[], patch_size=16,
                            in_channels=3, temporal_patch_size=2)
_KW = dict(hidden=1152, intermediate=4304, heads=16, out_hidden=5120, merge=2, deepstack=0, in_dim=1536)
MIB = 1 << 20


def _item(grid):
    return types.SimpleNamespace(image_grid_thw=torch.tensor(grid))


def test_work_model_books_the_quad_term_only_for_more_than_one_segment():
    """Bug regression (W105b on metal for 4096x4096): a still image is one
    segment, sdpa then runs without the 3*p*p mask, so the model books none;
    a clip (t=2) or two grid rows in one item still carry the mask, and
    triton_attn never booked it."""
    still = _item([[1, 256, 256]])  # 4096x4096 at patch 16: 65536 patches
    assert vv.item_segments(still) == 1
    assert vv.encode_work_bytes(65536, quadratic=True, **_KW) == 14690 * MIB  # the metal booking
    assert vv.encode_work_for(_VC, [still], "sdpa") == 2402 * MIB

    clip = _item([[2, 128, 128]])
    two_rows = _item([[1, 64, 64], [1, 64, 64]])
    assert vv.item_segments(clip) == 2 and vv.item_segments(two_rows) == 2
    assert vv.encode_work_for(_VC, [clip], "sdpa") == vv.encode_work_bytes(32768, quadratic=True, **_KW)
    assert vv.encode_work_for(_VC, [two_rows], "sdpa") == vv.encode_work_bytes(8192, quadratic=True, **_KW)
    assert vv.encode_work_for(_VC, [clip], "triton_attn") == vv.encode_work_bytes(32768, quadratic=False, **_KW)
