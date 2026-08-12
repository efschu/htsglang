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

"""#629: the cuda-graph REPLAY-PREP fills take the host mirror too.

#616h/#623 removed the unbounded blocking device-to-host read from the eager
DCP index builds. They left the Triton cuda-graph fills alone, on a reason
recorded in the #623 commit message -- "the Triton cuda-graph buffer fills,
which have no host mirror in scope". That was true of the signatures of
``_update_decode_kv_buffers`` / ``_update_target_verify_buffers``, and false of
the path that calls them: ``init_forward_metadata_out_graph`` receives the
ForwardBatch, which carries ``seq_lens_cpu`` and ``seq_lens_sum``.

WHICH READ IS ACTUALLY REMOVED, precisely. Under the EVEN owner rule the graph
fills hand the builder an address-stable ``cuda_graph_kv_indices`` buffer, so
the ``torch.empty(int(dcp_lens.sum().item()))`` sizing branch is never entered
-- that path has no read to remove. Under the WEIGHTED rule
``build_dcp_weighted_kv_indices`` derives ``total_tokens`` from
``int(full_indptr[bs].item())`` whenever it is not supplied, buffer or no
buffer. So the weighted replay-prep fill is the site with the sync, and it is
the rule this rig runs.

SECOND DEFECT, same family (the ``dcp_fresh_host_lens`` tests below). A mirror
handed to ``dcp_host_lens`` with no ``expected_sum`` is accepted UNCHECKED.
``seq_lens_cpu`` is a non-None but STALE slice exactly when ``seq_lens_sum`` is
None, so the eager sites wired by #623 sized from a stale vector on gpu_only
batches -- a silent mis-size, which is a worse failure than the stall it
replaced. The freshness signal has to be applied, not merely relied upon.

Hermetic: no CUDA, no process group, no model.
"""

import types

import pytest
import torch

from sglang.srt.layers.dcp.layout import (
    dcp_fresh_host_lens,
    dcp_host_lens,
    get_dcp_lens,
)
from sglang.srt.layers.dcp.owner import dcp_verify_paged_lens

_TRITON = "sglang.srt.layers.attention.triton_backend"

# The rig's weighted split for rank 0 (the 5090): 30/64 of the tokens.
_CP_S, _CP_LO, _CP_HI, _CP_RATIO = 64, 0, 30, 30
_DCP_SIZE, _DCP_RANK = 3, 0

# The device vector and the host mirror are given DIFFERENT numbers throughout.
# Asserting the result follows the MIRROR is what proves provenance; had they
# been equal, every test here would pass on the unfixed tree by coincidence.
_DEVICE_LEN = 41214
_MIRROR_LEN = 39166


class _Sentinel(Exception):
    """Raised from the patched builder to stop before any device work."""


def _capturing_builder(captured):
    def _builder(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _Sentinel()

    return _builder


def _forward_mode(mode: str):
    return types.SimpleNamespace(
        is_decode_or_idle=lambda: mode == "decode",
        is_target_verify=lambda: mode == "verify",
        is_draft_extend_v2=lambda: mode == "draft_extend",
    )


def _graph_backend(weighted: bool = True, bs: int = 1):
    """A TritonAttnBackend stand-in carrying only what the fills touch."""
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    fake = types.SimpleNamespace(
        uneven_dcp_weighted=weighted,
        dcp_size=_DCP_SIZE,
        dcp_rank=_DCP_RANK,
        cp_S=_CP_S,
        cp_lo=_CP_LO,
        cp_hi=_CP_HI,
        cp_ratio=_CP_RATIO,
        device="cpu",
        num_draft_tokens=4,
        sliding_window_size=None,
        req_to_token=torch.zeros((bs, 8), dtype=torch.int32),
        kv_indptr=torch.zeros(bs + 1, dtype=torch.int32),
        qo_indptr=torch.zeros(bs + 1, dtype=torch.int32),
        cuda_graph_kv_indices=torch.zeros(4096, dtype=torch.int64),
        cuda_graph_window_kv_indices=None,
        mask_indptr=torch.zeros(bs + 1, dtype=torch.int32),
        cuda_graph_custom_mask=None,
    )
    fake._dcp_lens = lambda lens, start=None: get_dcp_lens(
        lens, _DCP_SIZE, _DCP_RANK, start
    )
    for name in (
        "_dcp_kv_indices",
        "_dcp_weighted_kv_indices",
        "_update_decode_kv_buffers",
        "_update_target_verify_buffers",
        "_apply_cuda_graph_metadata",
    ):
        setattr(fake, name, types.MethodType(getattr(TritonAttnBackend, name), fake))
    return fake


def _capture_dcp_kv_indices(fake, captured):
    """Replace the shared index build with a recorder that stops the fill."""

    def _recorder(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _Sentinel()

    fake._dcp_kv_indices = _recorder


# ---------------------------------------------------------------------------
# 1. The freshness guard: "no sum" must degrade to "no mirror", never to
#    "unchecked mirror".
# ---------------------------------------------------------------------------


def test_a_mirror_without_its_sum_is_dropped_rather_than_trusted():
    mirror = torch.tensor([_MIRROR_LEN], dtype=torch.int32)
    assert dcp_fresh_host_lens(mirror, None) is None


def test_a_mirror_with_its_sum_is_kept():
    mirror = torch.tensor([_MIRROR_LEN], dtype=torch.int32)
    assert dcp_fresh_host_lens(mirror, _MIRROR_LEN) is mirror


def test_the_unguarded_helper_is_what_accepts_the_stale_slice():
    """The defect the guard exists for, stated as an executable fact.

    dcp_host_lens with no expected_sum hands the vector straight back. That is
    correct for a vector nobody knows the sum of, and wrong for seq_lens_cpu,
    which is stale precisely in that case. If this assertion ever flips,
    dcp_fresh_host_lens has become redundant and should go.
    """
    stale = torch.tensor([_DEVICE_LEN], dtype=torch.int32)
    assert dcp_host_lens(stale, None) is not None
    assert int(dcp_host_lens(stale, None).sum()) == _DEVICE_LEN


# ---------------------------------------------------------------------------
# 2. The entry point: it holds the ForwardBatch, so it is where the mirror
#    enters the cuda-graph path.
# ---------------------------------------------------------------------------


def _run_entry(*, seq_lens_cpu, seq_lens_sum, in_capture, captured, bs=1):
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    fake = _graph_backend()

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _Sentinel()

    fake._apply_cuda_graph_metadata = _recorder
    forward_batch = types.SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.zeros(bs, dtype=torch.int32),
        seq_lens=torch.tensor([_DEVICE_LEN] * bs, dtype=torch.int32),
        forward_mode=_forward_mode("decode"),
        spec_info=None,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=seq_lens_sum,
        encoder_lens=None,
    )
    with pytest.raises(_Sentinel):
        TritonAttnBackend.init_forward_metadata_out_graph(
            fake, forward_batch, in_capture=in_capture
        )


@pytest.mark.parametrize("in_capture", [False, True])
def test_replay_prep_entry_takes_the_mirror_from_the_forward_batch(in_capture):
    """Both legs -- capture and replay -- forward it; capture builds the same
    buffers replay refills, so wiring only one would leave the other syncing."""
    captured = {}
    _run_entry(
        seq_lens_cpu=torch.tensor([_MIRROR_LEN], dtype=torch.int32),
        seq_lens_sum=_MIRROR_LEN,
        in_capture=in_capture,
        captured=captured,
    )
    assert captured["seq_lens_sum"] == _MIRROR_LEN
    assert int(captured["seq_lens_cpu"].sum()) == _MIRROR_LEN


def test_a_gpu_only_batch_hands_the_fills_no_mirror_at_all():
    """seq_lens_sum None is the gpu_only signal; the non-None seq_lens_cpu
    beside it is a stale slice and must not reach the fills."""
    captured = {}
    _run_entry(
        seq_lens_cpu=torch.tensor([_DEVICE_LEN], dtype=torch.int32),
        seq_lens_sum=None,
        in_capture=False,
        captured=captured,
    )
    assert captured["seq_lens_cpu"] is None
    assert captured["seq_lens_sum"] is None


# ---------------------------------------------------------------------------
# 3. The dispatcher and the fills forward it onwards.
# ---------------------------------------------------------------------------


def test_decode_replay_fill_forwards_the_mirror_to_the_index_build():
    fake = _graph_backend()
    captured = {}
    _capture_dcp_kv_indices(fake, captured)

    with pytest.raises(_Sentinel):
        fake._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.tensor([_DEVICE_LEN], dtype=torch.int32),
            forward_mode=_forward_mode("decode"),
            spec_info=None,
            seq_lens_cpu=torch.tensor([_MIRROR_LEN], dtype=torch.int32),
            seq_lens_sum=_MIRROR_LEN,
        )

    assert int(captured["kwargs"]["lens_cpu"].sum()) == _MIRROR_LEN
    assert captured["kwargs"]["lens_sum"] == _MIRROR_LEN


def test_verify_replay_fill_forwards_the_mirror_through_the_length_function():
    """The verify PAGED length is seq_lens -- emphatically NOT
    seq_lens + num_draft_tokens. The mirror goes through the same
    dcp_verify_paged_lens the device vector does, so a future change to that
    rule cannot move one side without the other."""
    fake = _graph_backend()
    captured = {}
    _capture_dcp_kv_indices(fake, captured)
    spec_info = types.SimpleNamespace(draft_token_num=fake.num_draft_tokens)

    with pytest.raises(_Sentinel):
        fake._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.tensor([_DEVICE_LEN], dtype=torch.int32),
            forward_mode=_forward_mode("verify"),
            spec_info=spec_info,
            seq_lens_cpu=torch.tensor([_MIRROR_LEN], dtype=torch.int32),
            seq_lens_sum=_MIRROR_LEN,
        )

    mirror = captured["kwargs"]["lens_cpu"]
    assert int(mirror.sum()) == _MIRROR_LEN
    # Same function, same number -- not the identity assumed inline.
    expected = dcp_verify_paged_lens(
        torch.tensor([_MIRROR_LEN], dtype=torch.int32), fake.num_draft_tokens
    )
    assert torch.equal(mirror, expected)
    assert captured["kwargs"]["lens_sum"] == _MIRROR_LEN


def test_a_fill_without_a_mirror_keeps_the_old_device_read():
    """The default path is unchanged: no mirror in, no mirror out."""
    fake = _graph_backend()
    captured = {}
    _capture_dcp_kv_indices(fake, captured)

    with pytest.raises(_Sentinel):
        fake._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.tensor([_DEVICE_LEN], dtype=torch.int32),
            forward_mode=_forward_mode("decode"),
            spec_info=None,
        )

    assert captured["kwargs"]["lens_cpu"] is None
    assert captured["kwargs"]["lens_sum"] is None


# ---------------------------------------------------------------------------
# 4. End to end: the number the builder receives is the mirror's, and the
#    device read is not taken. This is the assertion that the sync is gone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["decode", "verify"])
def test_weighted_replay_prep_takes_total_tokens_from_the_host_mirror(
    monkeypatch, mode
):
    """The weighted builder is the site that syncs even when handed a buffer:
    total_tokens falls back to int(full_indptr[bs].item()) when absent."""
    captured = {}
    monkeypatch.setattr(
        f"{_TRITON}.build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )
    fake = _graph_backend(weighted=True)
    spec_info = (
        types.SimpleNamespace(draft_token_num=fake.num_draft_tokens)
        if mode == "verify"
        else None
    )

    with pytest.raises(_Sentinel):
        fake._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.tensor([_DEVICE_LEN], dtype=torch.int32),
            forward_mode=_forward_mode(mode),
            spec_info=spec_info,
            seq_lens_cpu=torch.tensor([_MIRROR_LEN], dtype=torch.int32),
            seq_lens_sum=_MIRROR_LEN,
        )

    assert captured["kwargs"].get("total_tokens") == _MIRROR_LEN


@pytest.mark.parametrize("mode", ["decode", "verify"])
def test_weighted_replay_prep_keeps_the_device_read_without_a_mirror(monkeypatch, mode):
    captured = {}
    monkeypatch.setattr(
        f"{_TRITON}.build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )
    fake = _graph_backend(weighted=True)
    spec_info = (
        types.SimpleNamespace(draft_token_num=fake.num_draft_tokens)
        if mode == "verify"
        else None
    )

    with pytest.raises(_Sentinel):
        fake._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.tensor([_DEVICE_LEN], dtype=torch.int32),
            forward_mode=_forward_mode(mode),
            spec_info=spec_info,
        )

    assert captured["kwargs"].get("total_tokens") is None


# ---------------------------------------------------------------------------
# 5. Census pin: a replay-prep fill added later must not skip the mirror.
# ---------------------------------------------------------------------------


def test_no_replay_prep_index_build_is_left_unwired():
    """Every _dcp_kv_indices call inside the cuda-graph fills passes lens_cpu.

    The #623 census pins the flashinfer/Triton call sites of the weighted
    BUILDER. This pins the layer above it: the Triton graph fills, which reach
    that builder through _dcp_kv_indices and were the ones #623 left out.
    """
    import inspect
    import re

    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    unwired = []
    for name in ("_update_decode_kv_buffers", "_update_target_verify_buffers"):
        src = inspect.getsource(getattr(TritonAttnBackend, name))
        for m in re.finditer(r"self\._dcp_kv_indices\(", src):
            depth, i = 1, m.end()
            while depth:
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                i += 1
            if "lens_cpu=" not in src[m.end() : i]:
                unwired.append(f"{name}+{src[: m.start()].count(chr(10)) + 1}")

    assert not unwired, (
        f"cuda-graph replay-prep index build at {unwired} passes no lens_cpu; "
        f"that fill is still on the unbounded D2H read inside the collective "
        f"window"
    )


def test_the_census_can_actually_fail():
    """A pin that cannot fail proves nothing -- so fail it on purpose."""
    import re

    src = "def f(self):\n    self._dcp_kv_indices(a, b, c)\n"
    unwired = []
    for m in re.finditer(r"self\._dcp_kv_indices\(", src):
        depth, i = 1, m.end()
        while depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        if "lens_cpu=" not in src[m.end() : i]:
            unwired.append(m.start())
    assert unwired, "the census regex stopped detecting an unwired call site"


# ---------------------------------------------------------------------------
# 6. The two PrefillWrapper updaters that were never put on the channel.
#
# #616h wired call_begin_forward to accept the mirrors; #623 wired
# update_single_wrapper to supply them. update_sliding_window and
# update_cross_attention -- the other two arms of the same exclusive dispatch
# -- were left passing nothing, so a backend running either arm still reached
# int(full_indptr[bs].item()). Both already TOOK seq_lens_cpu as a parameter
# and dropped it on the floor, which is the #616h failure mode exactly: a
# wired channel with no caller supplying it.
# ---------------------------------------------------------------------------


def _prefill_updater(sliding_window_size=None, swa_kv_pool=None):
    return types.SimpleNamespace(
        sliding_window_size=sliding_window_size,
        _swa_kv_pool=swa_kv_pool,
        prefill_wrapper_ragged=None,
        kv_indptr=[torch.zeros(2, dtype=torch.int32) for _ in range(2)],
        qo_indptr=[torch.zeros(2, dtype=torch.int32) for _ in range(2)],
        _build_swa_prefix_custom_mask=lambda *a, **k: None,
    )


def _capture_begin_forward(updater):
    calls = []

    def _cbf(*args, **kwargs):
        calls.append(kwargs)

    updater.call_begin_forward = _cbf
    return calls


def test_cross_attention_updater_forwards_the_mirrors():
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterPrefill,
    )

    updater = _prefill_updater()
    calls = _capture_begin_forward(updater)
    seq_lens_cpu = torch.tensor([_MIRROR_LEN], dtype=torch.int32)

    FlashInferIndicesUpdaterPrefill.update_cross_attention(
        updater,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([_DEVICE_LEN], dtype=torch.int32),
        seq_lens_cpu,
        _MIRROR_LEN,
        None,
        [None, None],
        False,
        torch.tensor([7], dtype=torch.int32),  # encoder_lens
        None,
    )

    assert len(calls) == 2
    for call in calls:
        assert call["seq_lens_cpu"] is seq_lens_cpu
    # wrapper 0 reads seq_lens, so its mirror is exact...
    assert calls[0]["paged_kernel_lens_cpu"] is seq_lens_cpu
    # ...wrapper 1 reads encoder_lens, which has no mirror; none is invented.
    assert calls[1]["paged_kernel_lens_cpu"] is None


def test_sliding_window_updater_forwards_the_mirrors():
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterPrefill,
    )

    window = 100
    updater = _prefill_updater(sliding_window_size=window)
    calls = _capture_begin_forward(updater)
    seq_lens_cpu = torch.tensor([_MIRROR_LEN], dtype=torch.int32)
    prefix_cpu = [_MIRROR_LEN - 8]

    FlashInferIndicesUpdaterPrefill.update_sliding_window(
        updater,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([_DEVICE_LEN], dtype=torch.int32),
        seq_lens_cpu,
        _MIRROR_LEN,
        torch.tensor([_DEVICE_LEN - 8], dtype=torch.int32),  # prefix_lens
        [None, None],
        True,  # use_ragged
        None,
        None,
        extend_prefix_lens_cpu=prefix_cpu,
    )

    assert len(calls) == 2
    for call in calls:
        assert call["seq_lens_cpu"] is seq_lens_cpu
        assert call["extend_prefix_lens_cpu"] is prefix_cpu
    # wrapper 0, ragged: min(prefix, window) taken on the host, and the sum
    # with it -- so the .sum().item() that used to sit here is gone.
    assert torch.equal(
        calls[0]["paged_kernel_lens_cpu"], torch.tensor([window], dtype=torch.int64)
    )
    # wrapper 1 is full attention: paged_kernel_lens IS seq_lens.
    assert calls[1]["paged_kernel_lens_cpu"] is seq_lens_cpu


def test_sliding_window_updater_claims_no_mirror_it_cannot_justify():
    """prefix_lens=None is re-derived on the device from seq_lens (and maybe a
    device-only num_accept_tokens), so the incoming prefix mirror no longer
    describes it and must be dropped rather than paired with it."""
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterPrefill,
    )

    updater = _prefill_updater(sliding_window_size=100)
    calls = _capture_begin_forward(updater)

    FlashInferIndicesUpdaterPrefill.update_sliding_window(
        updater,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([_DEVICE_LEN], dtype=torch.int32),
        torch.tensor([_MIRROR_LEN], dtype=torch.int32),
        _MIRROR_LEN,
        None,  # prefix_lens -> re-derived below
        [None, None],
        True,
        None,
        None,
        extend_prefix_lens_cpu=[_MIRROR_LEN - 8],
    )

    assert calls[0]["paged_kernel_lens_cpu"] is None


def test_no_prefill_updater_arm_is_left_off_the_mirror_channel():
    """Census pin: all three arms of the exclusive updater dispatch supply the
    mirrors. A fourth arm added later, or a regression on one of these, is a
    new unbounded D2H in the collective window."""
    import inspect

    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterPrefill,
    )

    arms = ("update_single_wrapper", "update_sliding_window", "update_cross_attention")
    missing = {}
    for name in arms:
        src = inspect.getsource(getattr(FlashInferIndicesUpdaterPrefill, name))
        absent = [
            kw
            for kw in ("seq_lens_cpu=", "paged_kernel_lens_cpu=")
            if f"{kw}" not in src.split("call_begin_forward(", 1)[-1]
        ]
        if absent:
            missing[name] = absent

    assert not missing, (
        f"prefill updater arm(s) do not forward the host mirror to "
        f"call_begin_forward: {missing}; that arm still reaches "
        f"int(full_indptr[bs].item()) inside the collective window, and trips "
        f"fast_prefill_plan's seq_lens_cpu assert under a captured graph"
    )


def test_the_prefill_arms_do_not_size_from_a_stale_slice():
    """gpu_only: seq_lens_sum is None and seq_lens_cpu beside it is stale.

    The forwarded mirror stays raw (fast_prefill_plan asserts it is not None),
    but the mirror used for SIZING must be dropped -- sizing an index buffer
    from a stale vector is a silent mis-size.
    """
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterPrefill,
    )

    stale = torch.tensor([_DEVICE_LEN], dtype=torch.int32)

    updater = _prefill_updater()
    calls = _capture_begin_forward(updater)
    FlashInferIndicesUpdaterPrefill.update_cross_attention(
        updater,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([_DEVICE_LEN], dtype=torch.int32),
        stale,
        None,  # seq_lens_sum -> gpu_only
        None,
        [None, None],
        False,
        torch.tensor([7], dtype=torch.int32),
        None,
    )
    assert calls[0]["paged_kernel_lens_cpu"] is None
    assert calls[0]["seq_lens_cpu"] is stale

    updater = _prefill_updater(sliding_window_size=100)
    calls = _capture_begin_forward(updater)
    FlashInferIndicesUpdaterPrefill.update_sliding_window(
        updater,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([_DEVICE_LEN], dtype=torch.int32),
        stale,
        None,  # seq_lens_sum -> gpu_only
        torch.tensor([_DEVICE_LEN - 8], dtype=torch.int32),
        [None, None],
        False,  # non-ragged wrapper-0 branch needs both host vectors
        None,
        None,
        extend_prefix_lens_cpu=[_DEVICE_LEN - 8],
    )
    assert calls[0]["paged_kernel_lens_cpu"] is None
    assert calls[1]["paged_kernel_lens_cpu"] is None
    assert calls[1]["seq_lens_cpu"] is stale
