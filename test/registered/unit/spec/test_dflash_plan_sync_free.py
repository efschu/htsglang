"""SGLANG_DFLASH_PLAN_SYNC_FREE: the DFLASH decode round plans without host reads.

Measured on the 27B D group (xsn421/xsn422, D TP0, bs 1): the host waited in
every round for the draft forward -- owner.py ``compact[owned]`` (83-105
py-spy samples) and ``repeat_interleave`` in the verify's weighted-DCP index
build -- and FlashInfer's ``plan()`` read qo/kv indptr back from the device for
the draft and the verify (prefill.py 1963/1974/1975/3054/3055), plus a device
sum for the draft's window wrapper (flashinfer_backend 6439).

What is pinned here, all on CPU, no CUDA, no process group:

1. the fixed-shape index build gives the SAME kv_indptr and the same leading
   kv_indices as the old build, for exact and over-estimated host bounds, and
   it has no data-dependent shape at all (it runs on META tensors, the old
   ``compact[owned]`` does not);
2. the worker-side prebuild writes the very buffer the verify graph reads and
   stages the exact counts to the host;
3. call_begin_forward takes the prebuilt index and hands HOST metadata to both
   verify plans -- and with no prebuilt it runs the old build unchanged;
4. the draft's host plan metadata equals what generate_attn_arg_prefill
   computes on the device, is only produced for a round flagged exact, and a
   mismatch is refused on the device;
5. the draft's window wrapper gets a host length mirror when prefix == seq, and
   only with the switch on.
"""

import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import flashinfer_backend as fib
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.dcp import owner
from sglang.srt.layers.dcp.owner import (
    build_dcp_weighted_kv_indices,
    build_dcp_weighted_kv_indices_sync_free,
    dcp_weighted_pack_owned,
    dcp_weighted_read_slots,
)
from sglang.srt.layers.dcp.verify_preplan import (
    DcpVerifyPrebuilt,
    host_arange_indptr,
    host_indptr_from_lens,
    host_ones,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput

# The rig's D split (--rank-tp-ratio 58,25,25) plus shapes that stress the rule.
PLANS = ([58, 25, 25], [2, 1, 1], [1, 1, 1], [30, 17, 17])


def _bounds(plan, rank):
    prefix = [0]
    for w in plan:
        prefix.append(prefix[-1] + w)
    return prefix[-1], prefix[rank], prefix[rank + 1], prefix[rank + 1] - prefix[rank]


class _CpuKvIndicesKernel:
    """CPU stand-in for create_flashinfer_kv_indices_triton[(bs,)](...)."""

    def __init__(self):
        self.launches = 0

    def __getitem__(self, grid):
        def _launch(req_to_token, req_pool_indices, lens, indptr, start, out, stride):
            self.launches += 1
            for b in range(int(grid[0])):
                n = int(lens[b])
                o = int(indptr[b])
                s = 0 if start is None else int(start[b])
                row = int(req_pool_indices[b])
                out[o : o + n] = req_to_token[row, s : s + n].to(out.dtype)

        return _launch


@pytest.fixture
def cpu_kernel(monkeypatch):
    k = _CpuKvIndicesKernel()
    monkeypatch.setattr(owner, "create_flashinfer_kv_indices_triton", k)
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_info.create_flashinfer_kv_indices_triton", k
    )
    return k


def _case(seed, bs, width=384, max_slot=5000):
    g = torch.Generator().manual_seed(seed)
    r2t = torch.randint(0, max_slot, (bs + 3, width), generator=g, dtype=torch.int32)
    rpi = torch.randperm(bs + 3, generator=g)[:bs].to(torch.int32)
    lens = torch.randint(0, width, (bs,), generator=g, dtype=torch.int32)
    if bs > 1:
        lens[1] = 0  # an empty request must keep its row in kv_indptr
    return r2t, rpi, lens


# ---------------------------------------------------------------------------
# 1. The fixed-shape build: same answer, no data-dependent shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("slack", [0, 1, 53])
def test_sync_free_build_equals_the_old_build(cpu_kernel, plan, slack):
    for rank in range(len(plan)):
        for seed, bs in ((0, 1), (1, 3), (2, 6)):
            r2t, rpi, lens = _case(seed, bs)
            cp = _bounds(plan, rank)
            old_indptr, old_indices = build_dcp_weighted_kv_indices(
                r2t,
                rpi,
                lens,
                torch.zeros(bs + 1, dtype=torch.int32),
                None,
                *cp,
                pad=256,
                total_tokens=int(lens.sum()),
            )
            buf = torch.zeros(bs + 1, dtype=torch.int32)
            new_indptr, new_indices = build_dcp_weighted_kv_indices_sync_free(
                r2t,
                rpi,
                lens,
                buf,
                None,
                *cp,
                total_tokens_bound=int(lens.sum()) + slack,
                pad=256,
            )
            assert torch.equal(new_indptr, old_indptr)
            n = int(old_indptr[-1])
            # Same leading entries + the same 256 zero pad, a longer zero tail.
            assert new_indices.numel() == int(lens.sum()) + slack + 256
            assert torch.equal(new_indices[: n + 256], old_indices)
            assert not bool(new_indices[n:].any())
            # Written in place into the buffer the verify graph reads.
            assert new_indptr.data_ptr() == buf.data_ptr()


def test_sync_free_build_with_a_start_offset(cpu_kernel):
    """kv_start_idx is part of the kernel's contract; the pack must not care."""
    r2t, rpi, lens = _case(7, 4)
    lens = torch.clamp(lens, max=200)
    start = torch.tensor([3, 0, 17, 100], dtype=torch.int32)
    cp = _bounds([58, 25, 25], 1)
    old = build_dcp_weighted_kv_indices(
        r2t, rpi, lens, torch.zeros(5, dtype=torch.int32), start, *cp,
        pad=0, total_tokens=int(lens.sum()),
    )
    new = build_dcp_weighted_kv_indices_sync_free(
        r2t, rpi, lens, torch.zeros(5, dtype=torch.int32), start, *cp,
        total_tokens_bound=int(lens.sum()) + 9, pad=0,
    )
    assert torch.equal(new[0], old[0])
    assert torch.equal(new[1][: int(old[0][-1])], old[1])


def test_an_undersized_bound_is_refused(cpu_kernel):
    """The bound sizes the kernel's output; below the device sum it would be
    an out-of-bounds write. CPU raises; CUDA asserts on the device."""
    r2t, rpi, lens = _case(3, 3)
    with pytest.raises(ValueError):
        build_dcp_weighted_kv_indices_sync_free(
            r2t, rpi, lens, torch.zeros(4, dtype=torch.int32), None,
            *_bounds([58, 25, 25], 0),
            total_tokens_bound=int(lens.sum()) - 1,
        )


def test_pack_ignores_the_tail_past_the_device_total():
    """Past full_indptr[-1] the slot buffer holds whatever the allocator left
    (the host bound over-sized it). Poison that tail with slots this rank
    OWNS: none of them may leak into kv_indices or the counts."""
    cp = _bounds([58, 25, 25], 0)
    real = torch.tensor([5, 200, 7, 58, 116, 3], dtype=torch.int32)
    poison = torch.full((40,), 1, dtype=torch.int32)  # slot 1: owned by rank 0
    full_kv = torch.cat([real, poison])
    full_indptr = torch.tensor([0, 2, 6], dtype=torch.int32)
    kv_indices, owned_prefix = dcp_weighted_pack_owned(full_kv, full_indptr, *cp, pad=4)
    compact, owned = dcp_weighted_read_slots(real, *cp)
    ref = compact[owned]
    n = int(ref.numel())
    assert torch.equal(owned_prefix, torch.tensor([0, int(owned[:2].sum()), n]))
    assert torch.equal(kv_indices[:n], ref)
    assert not bool(kv_indices[n:].any())


def test_pack_has_no_data_dependent_shape():
    """Runs on META tensors (no data), so no step can need a host read. The
    old boolean compaction cannot, which is exactly the owner.py:566 stall."""
    full_kv = torch.empty(1000, dtype=torch.int32, device="meta")
    full_indptr = torch.empty(4, dtype=torch.int32, device="meta")
    kv_indices, owned_prefix = dcp_weighted_pack_owned(
        full_kv, full_indptr, *_bounds([58, 25, 25], 0), pad=256
    )
    assert tuple(kv_indices.shape) == (1256,)
    assert tuple(owned_prefix.shape) == (4,)

    compact, owned = dcp_weighted_read_slots(full_kv, *_bounds([58, 25, 25], 0))
    with pytest.raises(Exception):
        compact[owned]


# ---------------------------------------------------------------------------
# 2. The worker-side prebuild.
# ---------------------------------------------------------------------------


def _fake_backend(r2t, *, weighted=True, cap=8, plan=(58, 25, 25), rank=0):
    cp_S, cp_lo, cp_hi, cp_ratio = _bounds(list(plan), rank)
    upd = types.SimpleNamespace(
        kv_indptr=[torch.zeros(cap + 1, dtype=torch.int32)], req_to_token=r2t
    )
    return types.SimpleNamespace(
        uneven_dcp=True,
        uneven_dcp_weighted=weighted,
        cp_S=cp_S,
        cp_lo=cp_lo,
        cp_hi=cp_hi,
        cp_ratio=cp_ratio,
        indices_updater_prefill=upd,
    )


def test_prebuild_fills_the_verify_buffer_and_stages_exact_counts(cpu_kernel):
    r2t, rpi, lens = _case(11, 5)
    fb = _fake_backend(r2t)
    pre = FlashInferAttnBackend.dcp_verify_prebuild(fb, rpi, lens, lens.clone())
    assert isinstance(pre, DcpVerifyPrebuilt)
    old_indptr, old_indices = build_dcp_weighted_kv_indices(
        r2t, rpi, lens, torch.zeros(6, dtype=torch.int32), None,
        *_bounds([58, 25, 25], 0), pad=256, total_tokens=int(lens.sum()),
    )
    host = pre.host_kv_indptr()
    assert host.device.type == "cpu"
    assert torch.equal(host, old_indptr)
    assert pre.kv_indptr.data_ptr() == fb.indices_updater_prefill.kv_indptr[0].data_ptr()
    assert torch.equal(pre.kv_indptr, old_indptr)
    n = int(host[-1])
    assert torch.equal(pre.kv_indices[: n + 256], old_indices)


def test_prebuild_sizes_from_an_overestimate_too(cpu_kernel):
    """The reservation bound (nxt_kv_lens) over-estimates; only the size moves."""
    r2t, rpi, lens = _case(12, 3)
    fb = _fake_backend(r2t)
    pre = FlashInferAttnBackend.dcp_verify_prebuild(fb, rpi, lens, lens + 16)
    old_indptr, _ = build_dcp_weighted_kv_indices(
        r2t, rpi, lens, torch.zeros(4, dtype=torch.int32), None,
        *_bounds([58, 25, 25], 0), pad=256, total_tokens=int(lens.sum()),
    )
    assert torch.equal(pre.host_kv_indptr(), old_indptr)


def test_prebuild_declines_without_weighted_dcp_or_mirror(cpu_kernel):
    r2t, rpi, lens = _case(13, 2)
    assert FlashInferAttnBackend.dcp_verify_prebuild(
        _fake_backend(r2t, weighted=False), rpi, lens, lens
    ) is None
    assert FlashInferAttnBackend.dcp_verify_prebuild(
        _fake_backend(r2t), rpi, lens, None
    ) is None
    # A CUDA mirror would itself be the read being removed -> refused; a row
    # count that does not match the batch is refused as well.
    assert FlashInferAttnBackend.dcp_verify_prebuild(
        _fake_backend(r2t), rpi, lens, torch.tensor([1, 2, 3])
    ) is None
    assert cpu_kernel.launches == 0


# ---------------------------------------------------------------------------
# 3. The verify plan consumes it.
# ---------------------------------------------------------------------------


class _Wrapper:
    def __init__(self, bs):
        self.calls = []
        self._paged_kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32)

    def begin_forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _verify_updater(r2t, kv_indptr_buf, plan=(58, 25, 25), rank=0):
    cp_S, cp_lo, cp_hi, cp_ratio = _bounds(list(plan), rank)
    attn_backend = types.SimpleNamespace(
        uneven_dcp=True,
        uneven_dcp_weighted=True,
        dcp_size=3,
        dcp_rank=rank,
        cp_S=cp_S,
        cp_lo=cp_lo,
        cp_hi=cp_hi,
        cp_ratio=cp_ratio,
        dcp_tree_mask=False,
        active_ragged_wrapper=None,
    )
    return types.SimpleNamespace(
        attn_backend=attn_backend,
        req_to_token=r2t,
        dcp_local_qo_heads=8,
        dcp_local_kv_heads=2,
        num_qo_heads=24,
        num_kv_heads=4,
        head_dim=256,
        q_data_type=torch.bfloat16,
        data_type=torch.float8_e4m3fn,
        kv_last_page_len=torch.ones(16, dtype=torch.int32),
        _swa_kv_pool=None,
        kv_indptr=[kv_indptr_buf],
    )


def _verify_input(draft_num=8, prebuilt=None):
    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long),
        positions=torch.empty((0,), dtype=torch.int64),
        draft_token_num=draft_num,
        dcp_verify_prebuilt=prebuilt,
    )


def _run_verify_plan(updater, rpi, lens, spec_info, kv_indptr_buf, bs):
    ragged, paged = _Wrapper(bs), _Wrapper(bs)
    FlashInferIndicesUpdaterPrefill.call_begin_forward(
        updater,
        ragged,
        paged,
        rpi,
        lens,  # paged_kernel_lens == committed seq_lens
        int(lens.sum()) + 8 * bs,  # the verify host bound the worker sets
        lens,  # seq_lens
        None,  # prefix_lens
        None,  # kv_start_idx
        kv_indptr_buf,
        torch.zeros(bs + 1, dtype=torch.int32),  # qo_indptr buffer
        False,  # use_ragged (the DCP verify split forces it)
        spec_info,
        paged_kernel_lens_cpu=lens + 8,
    )
    return ragged, paged


def test_verify_plan_takes_the_prebuilt_index_and_host_metadata(cpu_kernel, monkeypatch):
    def _no_old_build(*a, **k):
        raise AssertionError("the prebuilt path must not run the old build")

    monkeypatch.setattr(fib, "_build_dcp_weighted_kv_indices", _no_old_build)
    bs, draft_num = 3, 8
    r2t, rpi, lens = _case(21, bs)
    kv_buf = torch.zeros(17, dtype=torch.int32)
    fb = _fake_backend(r2t)
    fb.indices_updater_prefill.kv_indptr = [kv_buf]
    pre = FlashInferAttnBackend.dcp_verify_prebuild(fb, rpi, lens, lens)

    upd = _verify_updater(r2t, kv_buf)
    ragged, paged = _run_verify_plan(
        upd, rpi, lens, _verify_input(draft_num, pre), kv_buf, bs
    )

    host_qo = host_arange_indptr(bs, draft_num)
    (r_args, _), = ragged.calls
    assert r_args[0] is host_qo and r_args[1] is host_qo
    (p_args, _), = paged.calls
    qo, kv_indptr, kv_indices, last = p_args[:4]
    assert qo is host_qo
    assert kv_indptr.device.type == "cpu" and torch.equal(kv_indptr, pre.host_kv_indptr())
    assert torch.equal(last, host_ones(bs))
    n = int(pre.host_kv_indptr()[-1])
    # Same length as the old build's kv_indices: owned slots + the 256 pad.
    assert kv_indices.numel() == n + 256
    assert torch.equal(kv_indices, pre.kv_indices[: n + 256])


def test_verify_plan_without_prebuilt_is_the_old_path(cpu_kernel, monkeypatch):
    """Switch off (no prebuilt): the old build runs with the old arguments and
    the plans get the tensors it returned, not host copies."""
    seen = {}
    real = fib._build_dcp_weighted_kv_indices

    def _spy(*a, **k):
        seen["kwargs"] = k
        out = real(*a, **k)
        seen["out"] = out
        return out

    monkeypatch.setattr(fib, "_build_dcp_weighted_kv_indices", _spy)
    bs = 2
    r2t, rpi, lens = _case(22, bs)
    kv_buf = torch.zeros(9, dtype=torch.int32)
    upd = _verify_updater(r2t, kv_buf)
    ragged, paged = _run_verify_plan(upd, rpi, lens, _verify_input(8, None), kv_buf, bs)

    assert seen["kwargs"]["total_tokens"] == int(lens.sum()) + 8 * bs
    (p_args, _), = paged.calls
    assert p_args[1] is seen["out"][0]
    assert p_args[2] is seen["out"][1]
    assert p_args[0] is not host_arange_indptr(bs, 8)
    (r_args, _), = ragged.calls
    assert r_args[0] is not host_arange_indptr(bs, 8)


def test_a_prebuilt_for_another_shape_is_ignored(cpu_kernel, monkeypatch):
    """A padded graph bucket (bs differs) or another kv_indptr storage must
    fall back to the old build instead of planning a mismatched index."""
    bs = 2
    r2t, rpi, lens = _case(23, bs)
    kv_buf = torch.zeros(9, dtype=torch.int32)
    fb = _fake_backend(r2t)
    fb.indices_updater_prefill.kv_indptr = [torch.zeros(9, dtype=torch.int32)]
    pre_other_buf = FlashInferAttnBackend.dcp_verify_prebuild(fb, rpi, lens, lens)
    assert fib._usable_verify_prebuilt(
        _verify_input(8, pre_other_buf), bs, kv_buf, True
    ) is None
    fb.indices_updater_prefill.kv_indptr = [kv_buf]
    pre = FlashInferAttnBackend.dcp_verify_prebuild(fb, rpi, lens, lens)
    assert fib._usable_verify_prebuilt(_verify_input(8, pre), bs + 1, kv_buf, True) is None
    assert fib._usable_verify_prebuilt(_verify_input(8, pre), bs, kv_buf, False) is None
    assert fib._usable_verify_prebuilt(_verify_input(8, pre), bs, kv_buf, True) is pre


# ---------------------------------------------------------------------------
# 4. The draft plan: exact host metadata, only when flagged, checked on device.
# ---------------------------------------------------------------------------


def _draft_spec(draft_num=8, exact=True):
    spec = _verify_input(draft_num, None)
    spec.host_lens_exact = exact
    return spec


def test_draft_host_plan_equals_the_device_layout(cpu_kernel):
    bs, draft_num = 3, 8
    r2t, rpi, _ = _case(31, bs, width=3000)
    lens = torch.tensor([5, 2048, 0], dtype=torch.int32)
    spec = _draft_spec(draft_num)
    kv_indices, kv_indptr, qo_indptr, mask = spec.generate_attn_arg_prefill(
        rpi, lens, int(lens.sum()), r2t
    )
    hp = fib._draft_host_plan(spec, bs, lens.clone(), mask, False)
    assert hp is not None and hp.check_device
    assert torch.equal(hp.qo_indptr, qo_indptr)
    assert torch.equal(hp.kv_indptr, kv_indptr)
    assert torch.equal(hp.last_page_len, torch.ones(bs, dtype=torch.int32))


def test_draft_host_plan_needs_the_exact_flag_and_a_plain_layout():
    lens = torch.tensor([7, 9], dtype=torch.int32)
    assert fib._draft_host_plan(_draft_spec(exact=False), 2, lens, None, False) is None
    assert fib._draft_host_plan(_draft_spec(), 2, lens, torch.ones(3), False) is None
    assert fib._draft_host_plan(_draft_spec(), 2, lens, None, True) is None
    assert fib._draft_host_plan(_draft_spec(), 2, None, None, False) is None
    assert fib._draft_host_plan(_draft_spec(), 3, lens, None, False) is None
    # The spec input's own default is off: nothing but the worker raises it.
    assert _verify_input().host_lens_exact is False
    assert _verify_input().dcp_verify_prebuilt is None


def test_plan_indptr_mismatch_is_refused_on_the_device():
    w = _Wrapper(2)
    w._paged_kv_indptr_buf = torch.tensor([0, 13, 21], dtype=torch.int32)
    fib._assert_plan_indptr_matches(w, torch.tensor([0, 13, 21], dtype=torch.int32), 2)
    with pytest.raises(RuntimeError):
        fib._assert_plan_indptr_matches(
            w, torch.tensor([0, 13, 22], dtype=torch.int32), 2
        )


def test_host_indptr_helpers():
    assert torch.equal(
        host_indptr_from_lens(torch.tensor([3, 0, 5]), add=8),
        torch.tensor([0, 11, 19, 32], dtype=torch.int32),
    )
    a = host_arange_indptr(4, 8)
    assert a is host_arange_indptr(4, 8)  # cached, never rewritten
    assert torch.equal(a, torch.arange(0, 40, 8, dtype=torch.int32))


# ---------------------------------------------------------------------------
# 5. The draft's window wrapper: the 6439 device sum.
# ---------------------------------------------------------------------------


def _window_updater(window=2048):
    seen = []

    def _cbf(*args, **kwargs):
        seen.append((args, kwargs))

    upd = types.SimpleNamespace(
        sliding_window_size=window,
        prefill_wrapper_ragged=None,
        kv_indptr=[torch.zeros(9, dtype=torch.int32), torch.zeros(9, dtype=torch.int32)],
        qo_indptr=[torch.zeros(9, dtype=torch.int32), torch.zeros(9, dtype=torch.int32)],
        _swa_kv_pool=None,
        call_begin_forward=_cbf,
    )
    return upd, seen


def _run_window(upd, seq_lens):
    FlashInferIndicesUpdaterPrefill.update_sliding_window(
        upd,
        torch.arange(seq_lens.numel(), dtype=torch.int32),
        seq_lens,
        seq_lens.clone(),  # seq_lens_cpu
        int(seq_lens.sum()),
        None,  # prefix_lens: the DFLASH draft block forward
        [None, None],
        False,
        None,
        _draft_spec(),
    )


def test_window_wrapper_gets_a_host_mirror_with_the_switch(monkeypatch):
    monkeypatch.setenv("SGLANG_DFLASH_PLAN_SYNC_FREE", "1")

    def _no_device_sum(lens_cpu, lens):
        assert lens_cpu is not None, "window wrapper fell back to the device sum"
        return int(lens_cpu.sum())

    monkeypatch.setattr(fib, "_host_sum_or_device", _no_device_sum)
    upd, seen = _window_updater(window=16)
    seq = torch.tensor([10, 40, 16], dtype=torch.int32)
    _run_window(upd, seq)
    (a0, k0), (a1, k1) = seen
    assert torch.equal(k0["paged_kernel_lens_cpu"], torch.minimum(seq.to(torch.int64), torch.tensor(16)))
    assert a0[4] == int(torch.clamp(seq, max=16).sum())
    assert torch.equal(k1["paged_kernel_lens_cpu"].to(torch.int64), seq.to(torch.int64))


def test_window_wrapper_is_unchanged_without_the_switch(monkeypatch):
    monkeypatch.delenv("SGLANG_DFLASH_PLAN_SYNC_FREE", raising=False)
    assert envs.SGLANG_DFLASH_PLAN_SYNC_FREE.get() is False
    upd, seen = _window_updater(window=16)
    _run_window(upd, torch.tensor([10, 40], dtype=torch.int32))
    (a0, k0), _ = seen
    assert k0["paged_kernel_lens_cpu"] is None


# ---------------------------------------------------------------------------
# 6. The worker's side of the contract.
# ---------------------------------------------------------------------------


def _worker(*, switch=True, compact=True, page_size=1, target_backend=None):
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    w = types.SimpleNamespace(
        _plan_sync_free=switch,
        use_compact_draft_cache=compact,
        page_size=page_size,
        target_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(attn_backend=target_backend)
        ),
    )
    for name in (
        "_compact_draft_host_lens_exact",
        "_target_dcp_verify_backend",
        "_dcp_verify_prebuild",
    ):
        setattr(w, name, types.MethodType(getattr(DFlashWorkerV2, name), w))
    return w


def test_worker_flags_exact_draft_lengths_only_in_the_exact_case():
    assert _worker()._compact_draft_host_lens_exact() is True
    assert _worker(switch=False)._compact_draft_host_lens_exact() is False
    assert _worker(compact=False)._compact_draft_host_lens_exact() is False
    # page_size > 1: the host value is the page-aligned ENVELOPE, not exact.
    assert _worker(page_size=16)._compact_draft_host_lens_exact() is False


class _PrebuildRecorder:
    uneven_dcp = True
    uneven_dcp_weighted = True

    def __init__(self):
        self.calls = []

    def dcp_verify_prebuild(self, rpi, seq_lens, host):
        self.calls.append(host)
        return "prebuilt"


def test_worker_prebuild_reaches_the_hybrid_full_attention_backend():
    fi = _PrebuildRecorder()
    hybrid = types.SimpleNamespace(full_attn_backend=fi)
    w = _worker(target_backend=hybrid)
    exact = torch.tensor([10, 20])
    batch = types.SimpleNamespace(
        req_pool_indices=torch.tensor([1, 2]), seq_lens=exact.clone(), seq_lens_cpu=exact
    )
    draft_input = types.SimpleNamespace(nxt_kv_lens_cpu=torch.tensor([26, 36]))
    assert w._dcp_verify_prebuild(batch, draft_input) == "prebuilt"
    assert fi.calls[-1] is exact  # the published mirror when there is one
    batch.seq_lens_cpu = None
    assert w._dcp_verify_prebuild(batch, draft_input) == "prebuilt"
    assert fi.calls[-1] is draft_input.nxt_kv_lens_cpu  # else the reservation bound
    draft_input.nxt_kv_lens_cpu = None
    assert w._dcp_verify_prebuild(batch, draft_input) is None


def test_worker_prebuild_skips_a_target_without_weighted_dcp():
    fi = _PrebuildRecorder()
    fi.uneven_dcp_weighted = False
    w = _worker(target_backend=fi)
    batch = types.SimpleNamespace(
        req_pool_indices=torch.tensor([1]), seq_lens=torch.tensor([4]),
        seq_lens_cpu=torch.tensor([4]),
    )
    assert w._dcp_verify_prebuild(batch, types.SimpleNamespace()) is None
    assert fi.calls == []
    assert _worker(target_backend=object())._target_dcp_verify_backend() is None


# ---------------------------------------------------------------------------
# 7. (d) The [vram-peak] high-water read: same number, no flatten.
# ---------------------------------------------------------------------------


def test_vram_peak_fast_read_is_the_same_key(monkeypatch):
    from sglang.srt.model_executor import vram_family_census as vfc

    nested = {"allocated_bytes": {"all": {"peak": 7 * 2**30, "current": 1}}}
    monkeypatch.setattr(torch._C, "_cuda_memoryStats", lambda dev: nested, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 123)

    monkeypatch.delenv("SGLANG_VRAM_PEAK_FAST_READ", raising=False)
    assert vfc._max_allocated_bytes(torch.cuda) == 123  # off: the public call
    monkeypatch.setenv("SGLANG_VRAM_PEAK_FAST_READ", "1")
    assert vfc._max_allocated_bytes(torch.cuda) == 7 * 2**30

    # A stand-in module (the tests' injection seam) is never bypassed.
    fake = types.SimpleNamespace(max_memory_allocated=lambda: 5)
    assert vfc._max_allocated_bytes(fake) == 5

    # Any surprise in the private layout falls back to the public call.
    monkeypatch.setattr(torch._C, "_cuda_memoryStats", lambda dev: {}, raising=False)
    assert vfc._max_allocated_bytes(torch.cuda) == 123


def test_prebuilt_host_read_polls_the_event_and_reads_once():
    class _Ev:
        def __init__(self):
            self.queries = 0

        def query(self):
            self.queries += 1
            return self.queries >= 3

        def synchronize(self):  # the unbounded wait must not be used
            raise AssertionError("host_kv_indptr must poll, not synchronize")

    ev = _Ev()
    buf = torch.tensor([0, 4, 9, 9, 77], dtype=torch.int32)
    pre = DcpVerifyPrebuilt(
        bs=3, kv_indptr=buf[:4], kv_indices=torch.zeros(3), host_buf=buf, event=ev
    )
    out = pre.host_kv_indptr()
    assert ev.queries == 3
    assert torch.equal(out, torch.tensor([0, 4, 9, 9], dtype=torch.int32))
    buf.zero_()  # the pinned buffer is reused next round; the read is a copy
    assert pre.host_kv_indptr() is out and ev.queries == 3
