"""#616c: the weighted-DCP index build must not do a blocking D2H read.

`build_dcp_weighted_kv_indices` derives `total` as `int(full_indptr[bs].item())`
unless the caller supplies `total_tokens`. That `.item()` is an UNBOUNDED
blocking device-to-host sync sitting inside the collective window, and it is the
line the 2026-08-07 02:00 wedge died on: all three ranks stopped on it, each
holding a BAR1 spin kernel in its own device queue, none able to enqueue the work
that would release the others. A host parked in a CUDA sync cannot poll, cannot
time out and cannot enqueue; barlink's own staged-status wait is a BOUNDED poll,
which is why wedges that park there recover and this one did not.

The channel to skip the read existed but no caller ever passed it, so the branch
was dead code. These tests pin the wiring rather than the channel: each one FAILS
on the pre-fix tree.

Hermetic: no CUDA, no process group, no model. CPU tensors only.
"""

import types
from typing import Optional

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
    _dcp_host_total_tokens,
)
from sglang.srt.layers.dcp.owner import build_dcp_weighted_kv_indices


class _Sentinel(Exception):
    """Raised from the patched index builder to stop before any device work."""


def _capturing_builder(captured):
    def _builder(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _Sentinel()

    return _builder


def _fake_prefill_updater(req_to_token):
    """Minimal stand-in carrying only what call_begin_forward touches first."""
    attn_backend = types.SimpleNamespace(
        uneven_dcp=True,
        uneven_dcp_weighted=True,
        dcp_size=3,
        dcp_rank=0,
        # The real rig's weighted split for rank 0 (5090): 30/64 of the tokens.
        cp_S=64,
        cp_lo=0,
        cp_hi=30,
        cp_ratio=30,
        active_ragged_wrapper=None,
    )
    return types.SimpleNamespace(attn_backend=attn_backend, req_to_token=req_to_token)


# ---------------------------------------------------------------------------
# 1. The host sum is the SAME number the device read would have produced.
# ---------------------------------------------------------------------------


def test_host_total_tokens_equals_the_device_sum():
    prefix_lens_cpu = [41214 - 2048, 1337, 0, 9]
    prefix_lens = torch.tensor(prefix_lens_cpu, dtype=torch.int32)

    assert _dcp_host_total_tokens(prefix_lens_cpu) == int(prefix_lens.sum())


def test_host_total_tokens_is_none_without_a_mirror():
    # No mirror => None => build_dcp_weighted_kv_indices keeps the old read.
    # This is what preserves every caller that has no CPU-side lengths.
    assert _dcp_host_total_tokens(None) is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "needs a real device: the .item() branch this guards is only reached "
        "when paged_kernel_lens.is_cuda, and the index build launches Triton"
    ),
)
def test_supplying_total_tokens_does_not_change_the_result():
    """The channel must be a pure sync removal, not a behaviour change."""
    torch.manual_seed(0)
    bs = 3
    lens = torch.tensor([7, 0, 11], dtype=torch.int32)
    req_to_token = torch.randint(0, 512, (bs, 64), dtype=torch.int32)
    req_pool_indices = torch.arange(bs, dtype=torch.int32)

    def _run(total_tokens: Optional[int]):
        return build_dcp_weighted_kv_indices(
            req_to_token,
            req_pool_indices,
            lens,
            torch.zeros(bs + 1, dtype=torch.int32),
            None,
            64,
            0,
            30,
            30,
            total_tokens=total_tokens,
        )

    base_indptr, base_indices = _run(None)
    wired_indptr, wired_indices = _run(int(lens.sum()))

    assert torch.equal(base_indptr, wired_indptr)
    assert torch.equal(base_indices, wired_indices)


# ---------------------------------------------------------------------------
# 2. The wedging call site actually receives the mirror. Both FAIL pre-fix.
# ---------------------------------------------------------------------------


def test_call_begin_forward_passes_total_tokens_to_the_index_build(monkeypatch):
    """The exact site of the 02:00 wedge: flashinfer_backend call_begin_forward,
    uneven_dcp_weighted branch. Pre-fix it forwards no total_tokens at all and
    the build falls through to the blocking read."""
    captured = {}
    monkeypatch.setattr(
        "sglang.srt.layers.attention.flashinfer_backend._build_dcp_weighted_kv_indices",
        _capturing_builder(captured),
    )

    bs = 1
    # The real wedge batch: bs=1, seq_lens_sum 41214, a 2048-token extend chunk.
    prefix_lens_cpu = [41214 - 2048]
    req_to_token = torch.zeros((bs, 8), dtype=torch.int32)
    updater = _fake_prefill_updater(req_to_token)

    with pytest.raises(_Sentinel):
        FlashInferIndicesUpdaterPrefill.call_begin_forward(
            updater,
            None,  # wrapper_ragged
            None,  # wrapper_paged
            torch.zeros(bs, dtype=torch.int32),  # req_pool_indices
            torch.tensor([41214], dtype=torch.int32),  # paged_kernel_lens
            41214,  # paged_kernel_lens_sum (sum of seq_lens, NOT of prefix_lens)
            torch.tensor([41214], dtype=torch.int32),  # seq_lens
            torch.tensor(prefix_lens_cpu, dtype=torch.int32),  # prefix_lens
            None,  # kv_start_idx
            torch.zeros(bs + 1, dtype=torch.int32),  # kv_indptr
            torch.zeros(bs + 1, dtype=torch.int32),  # qo_indptr
            False,  # use_ragged (multimodal forces this False; DCP splits anyway)
            None,  # spec_info
            extend_prefix_lens_cpu=prefix_lens_cpu,
        )

    total_tokens = captured["kwargs"].get("total_tokens")
    assert total_tokens is not None, (
        "call_begin_forward reached the weighted-DCP index build without a "
        "host-side total_tokens; the build will fall back to the blocking "
        "full_indptr[bs].item() D2H that wedged all three ranks at 02:00."
    )
    # It must be sum(prefix_lens), NOT the readily-available but wrong
    # paged_kernel_lens_sum (41214), which is the sum of seq_lens.
    assert total_tokens == sum(prefix_lens_cpu)
    assert total_tokens != 41214


def test_update_single_wrapper_forwards_the_host_mirror(monkeypatch):
    """The missing link pre-fix: update_single_wrapper consumed
    extend_prefix_lens_cpu only on the use_ragged=True branch and never handed
    it down, so call_begin_forward's parameter was always None."""
    seen = {}

    def _fake_call_begin_forward(self, *args, **kwargs):
        seen.update(kwargs)

    updater = _fake_prefill_updater(torch.zeros((1, 8), dtype=torch.int32))
    updater.call_begin_forward = types.MethodType(_fake_call_begin_forward, updater)
    updater.kv_indptr = [torch.zeros(2, dtype=torch.int32)]
    updater.qo_indptr = [torch.zeros(2, dtype=torch.int32)]

    prefix_lens_cpu = [39166]
    FlashInferIndicesUpdaterPrefill.update_single_wrapper(
        updater,
        torch.zeros(1, dtype=torch.int32),  # req_pool_indices
        torch.tensor([41214], dtype=torch.int32),  # seq_lens
        torch.tensor([41214], dtype=torch.int32),  # seq_lens_cpu
        41214,  # seq_lens_sum
        torch.tensor(prefix_lens_cpu, dtype=torch.int32),  # prefix_lens
        [None],  # prefill_wrappers
        False,  # use_ragged -- the multimodal path that skipped the mirror
        None,  # encoder_lens
        None,  # spec_info
        extend_prefix_lens_cpu=prefix_lens_cpu,
    )

    assert seen.get("extend_prefix_lens_cpu") == prefix_lens_cpu, (
        "update_single_wrapper did not forward the host prefix-length mirror; "
        "on the use_ragged=False (multimodal) path it is the only way the "
        "weighted-DCP branch can avoid the blocking device read."
    )
