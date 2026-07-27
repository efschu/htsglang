# SPDX-License-Identifier: Apache-2.0
"""GPU gate for the expert-major prefill wave order (#254).

The CPU test (test_wave_order.py) proves the k-slot buffer makes the summation
order independent of the wave split. This test proves the other half on the
real kernel: that the per-(token, k-slot) contribution can be produced OUTSIDE
the fused reduction bit-for-bit, by submitting each routed pair as its own
pseudo-token with top_k == 1.

Gates (all against ONE unsplit apply over the full batch):
  * token-major waves (today's split)                       -> bit-identical
  * pair-flattened contributions + one fixed-order combine   -> bit-identical
  * the same contributions produced EXPERT-MAJOR             -> bit-identical
  * per-wave partial sums (the naive expert-major)           -> NOT identical
    (falsifier: the gate above is not vacuous)

Covers bf16 and fp8-blockwise, up to T=2048 / E=64 / top_k=8, including
duplicate expert ids in a row and -1 padded slots.

Run (needs an fp8-capable GPU for the fp8 cases):
  LD_LIBRARY_PATH=<venv nvidia libs> python -m pytest \
      tests/moe_offload/test_wave_order_gpu.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="expert-major wave order needs CUDA"
)

if torch.cuda.is_available():
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    if get_context()._server_args is None:
        get_context().set_server_args(ServerArgs(model_path="dummy"))

    from sglang.srt.layers.moe.expert_offload import combine_topk_partials
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )

DEV = "cuda"
FP8_OK = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@pytest.fixture(autouse=True)
def _free_between_cases():
    """The 2048-token case allocates ~GiB; do not carry it into the next test."""
    yield
    torch.cuda.empty_cache()


def _make_case(T, H, INTER, E, K, seed=0, fp8=False):
    g = torch.Generator().manual_seed(seed)
    # Sample on CPU, then move: on-GPU randn is not arch-identical.
    hidden = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    if fp8:
        w13 = (
            (torch.randn(E, 2 * INTER, H, generator=g) * 0.1)
            .to(DEV)
            .to(torch.float8_e4m3fn)
        )
        w2 = (
            (torch.randn(E, H, INTER, generator=g) * 0.1)
            .to(DEV)
            .to(torch.float8_e4m3fn)
        )
        kw = dict(
            use_fp8_w8a8=True,
            w1_scale=(
                torch.rand(E, (2 * INTER + 127) // 128, (H + 127) // 128, generator=g)
                * 0.02
                + 0.01
            ).to(DEV),
            w2_scale=(
                torch.rand(E, (H + 127) // 128, (INTER + 127) // 128, generator=g)
                * 0.02
                + 0.01
            ).to(DEV),
            block_shape=[128, 128],
        )
    else:
        w13 = (
            (torch.randn(E, 2 * INTER, H, generator=g) * 0.1).to(torch.bfloat16).to(DEV)
        )
        w2 = (torch.randn(E, H, INTER, generator=g) * 0.1).to(torch.bfloat16).to(DEV)
        kw = {}
    tw, tid = torch.topk(torch.softmax(torch.randn(T, E, generator=g), dim=-1), K, -1)
    return hidden, w13, w2, tw.float().to(DEV), tid.int().to(DEV), kw


def _apply(hidden, w13, w2, tw, tid, kw):
    return fused_experts_impl(
        hidden, w13, w2, tw, tid, inplace=False, filter_expert=True, **kw
    )


def _bit_eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _pair_contributions(hidden, w13, w2, tw, tid, kw, idx=None):
    """V[t,k] via one pseudo-token per routed (t,k) pair (top_k == 1)."""
    T, K = tid.shape
    flat_tw = tw.reshape(-1, 1).contiguous()
    flat_tid = tid.reshape(-1, 1).contiguous()
    if idx is None:
        rows = torch.arange(T, device=DEV).repeat_interleave(K)
        idx = torch.arange(T * K, device=DEV)
    else:
        rows = torch.div(idx, K, rounding_mode="floor")
    part = _apply(
        hidden.index_select(0, rows).contiguous(),
        w13,
        w2,
        flat_tw.index_select(0, idx),
        flat_tid.index_select(0, idx),
        kw,
    )
    return idx, part


CASES = [
    ("bf16-small", 64, 512, 256, 16, 4, False),
    ("bf16-big", 512, 2048, 768, 32, 8, False),
    ("fp8-blk", 512, 2048, 768, 32, 8, True),
    ("fp8-chunk", 2048, 2048, 768, 64, 8, True),
]


@pytest.mark.parametrize("tag,T,H,INTER,E,K,fp8", CASES)
def test_pair_contributions_reproduce_the_unsplit_apply(tag, T, H, INTER, E, K, fp8):
    if fp8 and not FP8_OK:
        pytest.skip("fp8e4nv unsupported on this GPU")
    hidden, w13, w2, tw, tid, kw = _make_case(T, H, INTER, E, K, seed=T, fp8=fp8)
    ref = _apply(hidden, w13, w2, tw, tid, kw)

    idx, part = _pair_contributions(hidden, w13, w2, tw, tid, kw)
    V = torch.zeros((T * K, ref.shape[-1]), dtype=ref.dtype, device=DEV)
    V.index_copy_(0, idx, part)
    out = torch.empty_like(ref)
    combine_topk_partials(V.view(T, K, -1), out, 1.0)

    assert _bit_eq(ref, out), f"{tag}: pair-flattened contributions diverged"


@pytest.mark.parametrize("tag,T,H,INTER,E,K,fp8", CASES)
def test_expert_major_assembly_is_bit_identical(tag, T, H, INTER, E, K, fp8):
    """Produce the SAME buffer wave by wave over disjoint expert groups."""
    if fp8 and not FP8_OK:
        pytest.skip("fp8e4nv unsupported on this GPU")
    hidden, w13, w2, tw, tid, kw = _make_case(T, H, INTER, E, K, seed=T, fp8=fp8)
    ref = _apply(hidden, w13, w2, tw, tid, kw)

    V = torch.zeros((T * K, ref.shape[-1]), dtype=ref.dtype, device=DEV)
    flat = tid.reshape(-1)
    per = max(1, E // 8)
    for s in range(0, E, per):
        mask = (flat >= s) & (flat < min(s + per, E))
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        _, part = _pair_contributions(hidden, w13, w2, tw, tid, kw, idx=idx)
        V.index_copy_(0, idx, part)
    out = torch.empty_like(ref)
    combine_topk_partials(V.view(T, K, -1), out, 1.0)

    assert _bit_eq(ref, out), f"{tag}: expert-major assembly diverged"


def test_kernel_config_is_m_invariant_only_for_blockwise_fp8():
    """Both wave orders inherit their bit-identity from ONE property: the fused
    kernel config must not depend on the per-apply token count M, or the K
    blocking (and with it the accumulation order) changes between a wave and
    the unsplit reference.

    Blockwise fp8 pins BLOCK_SIZE_K to the quantization block, so it is
    M-invariant and every split is bit-exact. The unquantized path carries an
    ``M <= E`` heuristic that flips BLOCK_SIZE_K 32 <-> 64 -- a byte-identity
    hole that predates this feature and already affects the token-major split
    (whose waves sit at ~C*T/(top_k*...) ~= a few dozen tokens, i.e. right at
    the boundary). Expert-major waves are far larger in M (one row per routed
    pair, not per token), so they land on the same side as the reference more
    often, but the guarantee comes from the quantization format, not the split.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_default_config,
    )

    E, N, Kdim, topk = 32, 768, 2048, 8
    blk = [128, 128]
    small = get_default_config(32, E, N, Kdim, topk, "fp8_w8a8", False, blk)
    large = get_default_config(2048, E, N, Kdim, topk, "fp8_w8a8", False, blk)
    assert small == large, "blockwise fp8 config must not depend on M"

    small = get_default_config(32, E, N, Kdim, topk, None, False, None)
    large = get_default_config(2048, E, N, Kdim, topk, None, False, None)
    assert small["BLOCK_SIZE_K"] != large["BLOCK_SIZE_K"], (
        "the M<=E heuristic on the unquantized path is expected to flip "
        "BLOCK_SIZE_K; if this ever stops being true, the bf16 wave-split "
        "gates below can be tightened to bit-identity"
    )


@pytest.mark.parametrize("tag,T,H,INTER,E,K,fp8", [c for c in CASES if c[-1]])
def test_token_major_waves_are_bit_identical(tag, T, H, INTER, E, K, fp8):
    """The path this feature is measured against (regression guard). Restricted
    to blockwise fp8 -- see test_kernel_config_is_m_invariant_only_for_blockwise_fp8
    for why the unquantized path cannot carry this gate."""
    if fp8 and not FP8_OK:
        pytest.skip("fp8e4nv unsupported on this GPU")
    hidden, w13, w2, tw, tid, kw = _make_case(T, H, INTER, E, K, seed=T, fp8=fp8)
    ref = _apply(hidden, w13, w2, tw, tid, kw)
    out = torch.empty_like(ref)
    step = max(1, T // 16)
    for s in range(0, T, step):
        rows = torch.arange(s, min(s + step, T), device=DEV)
        out.index_copy_(
            0,
            rows,
            _apply(
                hidden.index_select(0, rows),
                w13,
                w2,
                tw.index_select(0, rows),
                tid.index_select(0, rows),
                kw,
            ),
        )
    assert _bit_eq(ref, out), f"{tag}: token-major waves diverged"


def test_per_wave_partial_sums_are_not_bit_identical():
    """Falsifier: accumulating already-reduced per-wave partial sums (the naive
    expert-major) re-associates the top-k reduction and does NOT match."""
    T, H, INTER, E, K = 512, 2048, 768, 32, 8
    hidden, w13, w2, tw, tid, kw = _make_case(T, H, INTER, E, K, seed=T)
    ref = _apply(hidden, w13, w2, tw, tid, kw)
    acc = torch.zeros_like(ref)
    per = max(1, E // 8)
    for s in range(0, E, per):
        mask = (tid >= s) & (tid < min(s + per, E))
        acc += _apply(hidden, w13, w2, torch.where(mask, tw, 0.0), tid, kw)
    assert not _bit_eq(ref, acc)


@pytest.mark.parametrize("mutate", ["dup", "pad"])
def test_pair_contributions_with_duplicate_and_padded_slots(mutate):
    if not FP8_OK:
        pytest.skip("fp8e4nv unsupported on this GPU")
    T, H, INTER, E, K = 256, 2048, 768, 32, 8
    hidden, w13, w2, tw, tid, kw = _make_case(T, H, INTER, E, K, seed=7, fp8=True)
    if mutate == "dup":
        tid[:, 1] = tid[:, 0]
    else:
        tid[::3, -1] = -1
    ref = _apply(hidden, w13, w2, tw, tid, kw)

    idx, part = _pair_contributions(hidden, w13, w2, tw, tid, kw)
    V = torch.zeros((T * K, ref.shape[-1]), dtype=ref.dtype, device=DEV)
    V.index_copy_(0, idx, part)
    # A padded slot is never assigned to a wave; its buffer row stays zero,
    # which is what the kernel writes for it.
    V.view(T, K, -1)[tid.long() < 0] = 0
    out = torch.empty_like(ref)
    combine_topk_partials(V.view(T, K, -1), out, 1.0)
    assert _bit_eq(ref, out)
