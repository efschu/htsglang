"""#623 / #618: every DCP index-build site must derive its size on the host.

#616h wired the ``total_tokens`` bypass at ONE of the five call sites of
``build_dcp_weighted_kv_indices`` (the plain extend site). The remaining four
kept ``int(full_indptr[bs].item())`` -- an UNBOUNDED blocking device-to-host
read inside the collective window -- and one of them, the target-verify site,
is the one standing in the production wedge stack (02:02 crash, and the 03:23
wedge dump pinned the identical owner.py frame). #618 is the same hazard in the
sibling ``else`` branches, which size their kv_indices with
``int(dcp_lens.sum().item())`` and are reachable whenever
``uneven_dcp_weighted`` is off.

What a fix at these sites buys, stated exactly: across the 239 recorded wedge
events, 195 park all ranks in barlink's BOUNDED poll and recover, while the
unbounded sync sites are the fatal-capable minority -- a host inside a CUDA
sync can neither poll, time out, nor enqueue the work that releases its peers.
Converting a site from an unbounded sync to none removes that failure mode ON
THAT PATH. It is NOT a claim to have cured the graph-replay family root.

Test shape. Each seam test makes the host mirror carry a DIFFERENT number from
the length tensor handed to the builder, then asserts the size follows the
MIRROR. That is the only way to prove provenance rather than coincidence: on
the pre-fix tree the number can only come from the device tensor, so every one
of these fails there.

Hermetic: no CUDA, no process group, no model. CPU tensors only.
"""

import types

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterDecode,
    FlashInferIndicesUpdaterPrefill,
    _dcp_host_total_tokens,
)
from sglang.srt.layers.dcp.layout import (
    dcp_host_even_total,
    dcp_host_lens,
    dcp_host_total_tokens,
    get_dcp_lens,
)
from sglang.srt.speculative.spec_info import SpecInputType

_FLASHINFER = "sglang.srt.layers.attention.flashinfer_backend"
_TRITON = "sglang.srt.layers.attention.triton_backend"

# The rig's weighted split for rank 0 (the 5090): 30/64 of the tokens.
_CP_S, _CP_LO, _CP_HI, _CP_RATIO = 64, 0, 30, 30
# The even owner rule used by the #618 branches.
_DCP_SIZE, _DCP_RANK = 3, 0


class _Sentinel(Exception):
    """Raised from the patched builder/kernel to stop before any device work."""


def _capturing_builder(captured):
    def _builder(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _Sentinel()

    return _builder


class _CapturingKernel:
    """Stand-in for create_triton_kv_indices_for_dcp_triton[(grid,)](...)."""

    def __init__(self, captured):
        self._captured = captured

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            self._captured["kernel_args"] = args
            raise _Sentinel()

        return _launch


def _capture_empty_sizes(monkeypatch, captured):
    """Record every torch.empty size and keep the allocation on the host.

    The even (#618) decode branch hardcodes ``device="cuda"``; forcing CPU here
    is what lets the branch run at all without a device. The recorded size IS
    the assertion subject: it is the number the branch would otherwise have
    taken off the device.
    """
    real_empty = torch.empty

    def _fake_empty(*args, **kwargs):
        kwargs.pop("device", None)
        captured.setdefault("empty_sizes", []).append(
            args[0] if args else kwargs.get("size")
        )
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", _fake_empty)


def _attn_backend(weighted: bool):
    return types.SimpleNamespace(
        uneven_dcp=True,
        uneven_dcp_weighted=weighted,
        dcp_size=_DCP_SIZE,
        dcp_rank=_DCP_RANK,
        cp_S=_CP_S,
        cp_lo=_CP_LO,
        cp_hi=_CP_HI,
        cp_ratio=_CP_RATIO,
        active_ragged_wrapper=None,
    )


def _fake_prefill_updater(weighted: bool = True, bs: int = 1):
    return types.SimpleNamespace(
        attn_backend=_attn_backend(weighted),
        req_to_token=torch.zeros((bs, 8), dtype=torch.int32),
    )


def _fake_decode_updater(weighted: bool = True, bs: int = 1):
    return types.SimpleNamespace(
        attn_backend=_attn_backend(weighted),
        req_to_token=torch.zeros((bs, 8), dtype=torch.int32),
    )


def _run_prefill(updater, *, paged_lens, paged_sum, spec_info, mirror, bs=1):
    """Drive FlashInferIndicesUpdaterPrefill.call_begin_forward to its DCP branch."""
    return FlashInferIndicesUpdaterPrefill.call_begin_forward(
        updater,
        None,  # wrapper_ragged
        None,  # wrapper_paged
        torch.zeros(bs, dtype=torch.int32),  # req_pool_indices
        torch.tensor(paged_lens, dtype=torch.int32),  # paged_kernel_lens
        paged_sum,  # paged_kernel_lens_sum
        torch.tensor(paged_lens, dtype=torch.int32),  # seq_lens
        None,  # prefix_lens
        None,  # kv_start_idx
        torch.zeros(bs + 1, dtype=torch.int32),  # kv_indptr
        torch.zeros(bs + 1, dtype=torch.int32),  # qo_indptr
        False,  # use_ragged
        spec_info,
        paged_kernel_lens_cpu=mirror,
    )


def _verify_spec_info(draft_num=4):
    return types.SimpleNamespace(
        spec_input_type=SpecInputType.EAGLE_VERIFY,
        draft_token_num=draft_num,
        custom_mask=None,
        kv_indptr=None,
    )


def _draft_extend_spec_info(num_tokens_per_req=2):
    return types.SimpleNamespace(
        spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND,
        num_tokens_per_req=num_tokens_per_req,
        kv_indptr=None,
    )


# ---------------------------------------------------------------------------
# 0. The host math itself: same integers as the device formula, and a mirror
#    that cannot be trusted is refused rather than guessed at.
# ---------------------------------------------------------------------------


def test_host_even_total_matches_the_tensor_formula():
    """#618: sum(get_dcp_lens(...)) on the host == the device sum, exactly."""
    torch.manual_seed(0)
    for _ in range(64):
        bs = int(torch.randint(1, 9, (1,)))
        lens = torch.randint(0, 4096, (bs,), dtype=torch.int32)
        for size in (1, 2, 3, 5):
            for rank in range(size):
                device_sum = int(get_dcp_lens(lens, size, rank).sum())
                assert dcp_host_even_total(lens.tolist(), size, rank) == device_sum
                assert dcp_host_even_total(lens, size, rank) == device_sum


def test_host_even_total_matches_with_a_start_offset():
    """The kv_start_idx form is part of the formula, not a special case."""
    torch.manual_seed(1)
    lens = torch.randint(0, 512, (6,), dtype=torch.int32)
    start = torch.randint(0, 64, (6,), dtype=torch.int32)
    for size in (2, 3, 4):
        for rank in range(size):
            assert dcp_host_even_total(lens, size, rank, start=start) == int(
                get_dcp_lens(lens, size, rank, start).sum()
            )


def test_a_mirror_that_disagrees_with_the_known_sum_is_refused():
    """The staleness guard: gpu_only batches carry a non-None but STALE
    seq_lens_cpu, and sizing an index buffer from it would be a silent
    mis-size, not a crash. A caller that knows the real sum gets None back."""
    assert dcp_host_lens([10, 10], expected_sum=20) is not None
    assert dcp_host_lens([10, 10], expected_sum=21) is None
    assert dcp_host_total_tokens([10, 10], 21) is None
    assert dcp_host_even_total([10, 10], 3, 0, expected_sum=21) is None
    # No claimed sum -> no check; the mirror is taken as given.
    assert dcp_host_total_tokens([10, 10]) == 20


def test_absent_mirror_keeps_the_old_device_read():
    """Default path: nothing supplied -> None -> callers keep .item()."""
    assert dcp_host_lens(None) is None
    assert dcp_host_total_tokens(None) is None
    assert dcp_host_even_total(None, 3, 0) is None
    assert _dcp_host_total_tokens(None) is None


def test_a_cuda_mirror_is_refused_rather_than_read():
    """A 'mirror' that lives on the device would reintroduce the very sync."""

    class _FakeCudaTensor(torch.Tensor):
        @property
        def is_cuda(self):  # pragma: no cover - trivial
            return True

    fake = torch.tensor([1, 2, 3]).as_subclass(_FakeCudaTensor)
    assert dcp_host_lens(fake) is None


# ---------------------------------------------------------------------------
# 1. Site A -- decode. flashinfer_backend FlashInferIndicesUpdaterDecode.
# ---------------------------------------------------------------------------


def test_decode_weighted_site_takes_total_tokens_from_the_host_mirror(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        f"{_FLASHINFER}._build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )
    updater = _fake_decode_updater(weighted=True)
    # The mirror deliberately differs from the length tensor: a total of 39166
    # can then only have come from the host side.
    mirror = torch.tensor([39166], dtype=torch.int32)

    with pytest.raises(_Sentinel):
        FlashInferIndicesUpdaterDecode.call_begin_forward(
            updater,
            None,  # wrapper
            torch.zeros(1, dtype=torch.int32),  # req_pool_indices
            torch.tensor([41214], dtype=torch.int32),  # paged_kernel_lens
            39166,  # paged_kernel_lens_sum (the mirror's sum)
            torch.zeros(2, dtype=torch.int32),  # kv_indptr
            None,  # kv_start_idx
            None,  # spec_info
            mirror,  # seq_lens_cpu
        )

    assert captured["kwargs"].get("total_tokens") == 39166, (
        "the uneven-DCP decode index build got no host total_tokens and will "
        "fall back to the blocking full_indptr[bs].item() D2H"
    )


def test_decode_even_site_sizes_kv_indices_from_the_host_mirror(monkeypatch):
    """#618 sibling of the site above."""
    captured = {}
    _capture_empty_sizes(monkeypatch, captured)
    monkeypatch.setattr(
        f"{_FLASHINFER}.create_triton_kv_indices_for_dcp_triton",
        _CapturingKernel(captured),
    )
    updater = _fake_decode_updater(weighted=False)

    with pytest.raises(_Sentinel):
        FlashInferIndicesUpdaterDecode.call_begin_forward(
            updater,
            None,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([12], dtype=torch.int32),  # device lens -> 12//3 = 4
            9,  # the mirror's sum
            torch.zeros(2, dtype=torch.int32),
            None,  # kv_start_idx: absent, so the host formula applies
            None,
            torch.tensor([9], dtype=torch.int32),  # mirror -> 9//3 = 3
        )

    assert captured["empty_sizes"] == [3], (
        "the even-DCP decode branch sized kv_indices from the device "
        "dcp_lens.sum().item() instead of the host mirror"
    )


def test_decode_even_site_falls_back_when_kv_start_idx_is_device_only(monkeypatch):
    """kv_start_idx is part of the length formula. No host copy -> no guess."""
    assert (
        dcp_host_even_total(
            [9], _DCP_SIZE, _DCP_RANK, start=torch.tensor([2]).as_subclass(
                type("_C", (torch.Tensor,), {"is_cuda": property(lambda self: True)})
            )
        )
        is None
    )


# ---------------------------------------------------------------------------
# 2. Site D -- target verify. THE production wedge stack.
# ---------------------------------------------------------------------------


def test_verify_weighted_site_takes_total_tokens_from_the_host_mirror(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        f"{_FLASHINFER}._build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )

    with pytest.raises(_Sentinel):
        _run_prefill(
            _fake_prefill_updater(weighted=True),
            paged_lens=[41214],  # what the device tensor says
            paged_sum=39166,  # what the host knows
            spec_info=_verify_spec_info(),
            mirror=torch.tensor([39166], dtype=torch.int32),
        )

    total_tokens = captured["kwargs"].get("total_tokens")
    assert total_tokens == 39166, (
        "the target-verify index build -- the site in the production wedge "
        "stack -- reached build_dcp_weighted_kv_indices without a host total"
    )
    assert total_tokens != 41214, "the total must not come from the device tensor"


def test_verify_even_site_sizes_kv_indices_from_the_host_mirror(monkeypatch):
    captured = {}
    _capture_empty_sizes(monkeypatch, captured)
    monkeypatch.setattr(
        f"{_FLASHINFER}.create_triton_kv_indices_for_dcp_triton",
        _CapturingKernel(captured),
    )

    with pytest.raises(_Sentinel):
        _run_prefill(
            _fake_prefill_updater(weighted=False),
            paged_lens=[12],  # device -> 4
            paged_sum=9,
            spec_info=_verify_spec_info(),
            mirror=torch.tensor([9], dtype=torch.int32),  # host -> 3
        )

    # +256 is the branch's own pad, unchanged by this fix.
    assert captured["empty_sizes"] == [3 + 256]


# ---------------------------------------------------------------------------
# 3. Site C -- draft extend. The length vector is DERIVED, so the host side
#    must run the same subtraction (and the same clamp).
# ---------------------------------------------------------------------------


def test_draft_extend_weighted_site_applies_the_subtraction_on_the_host(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        f"{_FLASHINFER}._build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )

    with pytest.raises(_Sentinel):
        _run_prefill(
            _fake_prefill_updater(weighted=True),
            paged_lens=[500],  # device -> 500 - 2 = 498
            paged_sum=100,
            spec_info=_draft_extend_spec_info(num_tokens_per_req=2),
            mirror=torch.tensor([100], dtype=torch.int32),  # host -> 98
        )

    assert captured["kwargs"].get("total_tokens") == 98, (
        "draft-extend must subtract num_tokens_per_req from the HOST mirror; "
        "the committed prefix is shorter than seq_lens on this path"
    )


def test_draft_extend_host_total_is_clamped_at_zero(monkeypatch):
    """A request whose whole sequence is this step's tokens has no prefix.
    Skipping the clamp would over-size the buffer (and mis-state the length)."""
    captured = {}
    monkeypatch.setattr(
        f"{_FLASHINFER}._build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )

    with pytest.raises(_Sentinel):
        _run_prefill(
            _fake_prefill_updater(weighted=True, bs=2),
            paged_lens=[1, 40],
            paged_sum=41,
            spec_info=_draft_extend_spec_info(num_tokens_per_req=4),
            mirror=torch.tensor([1, 40], dtype=torch.int32),
            bs=2,
        )

    # max(1-4, 0) + max(40-4, 0) == 36, not 37.
    assert captured["kwargs"].get("total_tokens") == 36


def test_draft_extend_even_site_sizes_kv_indices_from_the_host_mirror(monkeypatch):
    captured = {}
    _capture_empty_sizes(monkeypatch, captured)
    monkeypatch.setattr(
        f"{_FLASHINFER}.create_triton_kv_indices_for_dcp_triton",
        _CapturingKernel(captured),
    )

    with pytest.raises(_Sentinel):
        _run_prefill(
            _fake_prefill_updater(weighted=False),
            paged_lens=[500],  # device -> (500-2)//3 = 166
            paged_sum=11,
            spec_info=_draft_extend_spec_info(num_tokens_per_req=2),
            mirror=torch.tensor([11], dtype=torch.int32),  # host -> (11-2)//3 = 3
        )

    assert captured["empty_sizes"] == [3 + 256]


# ---------------------------------------------------------------------------
# 4. Site B's even sibling -- plain extend (#618). The weighted half was wired
#    in #616h; this is the branch reachable with uneven_dcp_weighted off.
# ---------------------------------------------------------------------------


def test_extend_even_site_sizes_kv_indices_from_the_host_mirror(monkeypatch):
    captured = {}
    _capture_empty_sizes(monkeypatch, captured)
    monkeypatch.setattr(
        f"{_FLASHINFER}.create_triton_kv_indices_for_dcp_triton",
        _CapturingKernel(captured),
    )
    updater = _fake_prefill_updater(weighted=False)

    with pytest.raises(_Sentinel):
        FlashInferIndicesUpdaterPrefill.call_begin_forward(
            updater,
            None,
            None,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([41214], dtype=torch.int32),  # paged_kernel_lens
            41214,
            torch.tensor([41214], dtype=torch.int32),  # seq_lens
            torch.tensor([12], dtype=torch.int32),  # prefix_lens -> device 4
            None,
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            False,  # use_ragged
            None,  # spec_info -> the plain extend branch
            extend_prefix_lens_cpu=[9],  # host -> 3
        )

    assert captured["empty_sizes"] == [3 + 256]


def test_extend_branch_default_path_is_unchanged_without_a_mirror(monkeypatch):
    """No mirror anywhere -> total_tokens None -> the pre-#616h behaviour."""
    captured = {}
    monkeypatch.setattr(
        f"{_FLASHINFER}._build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )
    updater = _fake_prefill_updater(weighted=True)

    with pytest.raises(_Sentinel):
        FlashInferIndicesUpdaterPrefill.call_begin_forward(
            updater,
            None,
            None,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([41214], dtype=torch.int32),
            41214,
            torch.tensor([41214], dtype=torch.int32),
            torch.tensor([39166], dtype=torch.int32),
            None,
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            False,
            None,
        )

    assert captured["kwargs"].get("total_tokens") is None


def test_verify_site_falls_back_when_the_mirror_is_stale(monkeypatch):
    """A mirror whose sum contradicts paged_kernel_lens_sum is not used."""
    captured = {}
    monkeypatch.setattr(
        f"{_FLASHINFER}._build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )

    with pytest.raises(_Sentinel):
        _run_prefill(
            _fake_prefill_updater(weighted=True),
            paged_lens=[41214],
            paged_sum=41214,  # the truth
            spec_info=_verify_spec_info(),
            mirror=torch.tensor([999], dtype=torch.int32),  # stale slice
        )

    assert captured["kwargs"].get("total_tokens") is None, (
        "a stale host mirror must be refused, not used to size the index build"
    )


# ---------------------------------------------------------------------------
# 5. Site E -- the Triton backend's owner-rule builders.
# ---------------------------------------------------------------------------


def _fake_triton_backend(weighted: bool):
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
        req_to_token=torch.zeros((1, 8), dtype=torch.int32),
    )
    fake._dcp_lens = lambda lens, start=None: get_dcp_lens(
        lens, _DCP_SIZE, _DCP_RANK, start
    )
    fake._dcp_weighted_kv_indices = types.MethodType(
        TritonAttnBackend._dcp_weighted_kv_indices, fake
    )
    return TritonAttnBackend._dcp_kv_indices, fake


def test_triton_weighted_site_takes_total_tokens_from_the_host_mirror(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        f"{_TRITON}.build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )
    entry, fake = _fake_triton_backend(weighted=True)

    with pytest.raises(_Sentinel):
        entry(
            fake,
            torch.zeros(1, dtype=torch.int32),  # req_pool_indices
            torch.tensor([41214], dtype=torch.int32),  # lens
            torch.zeros(2, dtype=torch.int32),  # kv_indptr
            lens_cpu=torch.tensor([39166], dtype=torch.int32),
            lens_sum=39166,
        )

    assert captured["kwargs"].get("total_tokens") == 39166


def test_triton_even_site_sizes_kv_indices_from_the_host_mirror(monkeypatch):
    captured = {}
    _capture_empty_sizes(monkeypatch, captured)
    monkeypatch.setattr(
        f"{_TRITON}.create_triton_kv_indices_for_dcp_triton", _CapturingKernel(captured)
    )
    entry, fake = _fake_triton_backend(weighted=False)

    with pytest.raises(_Sentinel):
        entry(
            fake,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([12], dtype=torch.int32),  # device -> 4
            torch.zeros(2, dtype=torch.int32),
            lens_cpu=torch.tensor([9], dtype=torch.int32),  # host -> 3
            lens_sum=9,
        )

    assert captured["empty_sizes"] == [3]


def test_triton_default_path_is_unchanged_without_a_mirror(monkeypatch):
    """The cuda-graph callers pass no mirror and must keep the old read."""
    captured = {}
    monkeypatch.setattr(
        f"{_TRITON}.build_dcp_weighted_kv_indices", _capturing_builder(captured)
    )
    entry, fake = _fake_triton_backend(weighted=True)

    with pytest.raises(_Sentinel):
        entry(
            fake,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([41214], dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )

    assert captured["kwargs"].get("total_tokens") is None


# ---------------------------------------------------------------------------
# 6. Callers actually forward the mirrors (the #616h failure mode was a wired
#    channel with no caller supplying it).
# ---------------------------------------------------------------------------


def test_update_single_wrapper_forwards_the_paged_kernel_lens_mirror():
    """use_ragged=False: paged_kernel_lens IS seq_lens, so its mirror is
    seq_lens_cpu -- NOT extend_prefix_lens_cpu, which describes a different
    vector on this path. The two DCP spec branches index over the former."""
    seen = {}

    def _fake_call_begin_forward(self, *args, **kwargs):
        seen.update(kwargs)

    updater = _fake_prefill_updater()
    updater.call_begin_forward = types.MethodType(_fake_call_begin_forward, updater)
    updater.kv_indptr = [torch.zeros(2, dtype=torch.int32)]
    updater.qo_indptr = [torch.zeros(2, dtype=torch.int32)]

    seq_lens_cpu = torch.tensor([41214], dtype=torch.int32)
    FlashInferIndicesUpdaterPrefill.update_single_wrapper(
        updater,
        torch.zeros(1, dtype=torch.int32),  # req_pool_indices
        torch.tensor([41214], dtype=torch.int32),  # seq_lens
        seq_lens_cpu,
        41214,  # seq_lens_sum
        None,  # prefix_lens
        [None],  # prefill_wrappers
        False,  # use_ragged
        None,  # encoder_lens
        None,  # spec_info
    )

    mirror = seen.get("paged_kernel_lens_cpu")
    assert mirror is not None, (
        "update_single_wrapper did not forward the paged_kernel_lens mirror; "
        "the verify and draft-extend DCP branches then have no host total and "
        "fall back to the blocking device read"
    )
    assert torch.equal(mirror, seq_lens_cpu)


def test_update_single_wrapper_forwards_the_prefix_mirror_on_the_ragged_path():
    """use_ragged=True: paged_kernel_lens IS prefix_lens, so the mirror is the
    extend one. Getting these two the wrong way round is the #616h trap."""
    seen = {}

    def _fake_call_begin_forward(self, *args, **kwargs):
        seen.update(kwargs)

    updater = _fake_prefill_updater()
    updater.call_begin_forward = types.MethodType(_fake_call_begin_forward, updater)
    updater.kv_indptr = [torch.zeros(2, dtype=torch.int32)]
    updater.qo_indptr = [torch.zeros(2, dtype=torch.int32)]

    FlashInferIndicesUpdaterPrefill.update_single_wrapper(
        updater,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([41214], dtype=torch.int32),
        torch.tensor([41214], dtype=torch.int32),
        41214,
        torch.tensor([39166], dtype=torch.int32),  # prefix_lens
        [None],
        True,  # use_ragged
        None,
        None,
        extend_prefix_lens_cpu=[39166],
    )

    assert seen.get("paged_kernel_lens_cpu") == [39166]
    assert seen.get("extend_prefix_lens_cpu") == [39166]


@pytest.mark.parametrize(
    "source",
    [
        "sglang/srt/layers/attention/flashinfer_backend.py",
        "sglang/srt/layers/attention/triton_backend.py",
    ],
)
def test_no_owner_rule_builder_call_is_left_unwired(source):
    """Census pin: every call of the weighted builder in these two backends
    passes a total_tokens (even if it evaluates to None at runtime). A new
    unwired call site is a new unbounded sync in the collective window."""
    import pathlib
    import re

    import sglang

    root = pathlib.Path(sglang.__file__).resolve().parents[1]
    text = (root / source).read_text()
    # Call sites only: the alias assignment and the import carry no "(".
    starts = [
        m.end()
        for m in re.finditer(r"(?<![\w.])_?build_dcp_weighted_kv_indices\(", text)
    ]
    assert starts, f"{source}: no weighted-DCP index build found at all"

    unwired = []
    for start in starts:
        depth, i = 1, start
        while depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if "total_tokens=" not in text[start:i]:
            unwired.append(text[:start].count("\n") + 1)

    assert not unwired, (
        f"{source}: weighted-DCP index build at line(s) {unwired} passes no "
        f"total_tokens; that call site is still on the unbounded D2H read"
    )
