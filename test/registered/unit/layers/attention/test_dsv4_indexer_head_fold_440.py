"""#440 -- the C4 indexer head axis cannot be folded out, and why.

Upstream sgl-project/sglang#33246 (comment 5159510149, Hakureirm) proposes
collapsing the DSpark indexer's head axis before the GEMM. The identity given
there is

    logits[i, j] = sum_h w[i, h] * (sum_d q[i, h, d] * k[j, d]) * k_scale[j]

which is linear in the per-head product, so by linearity the head sum moves
inside and one folded query ``q_eff[i, :] = sum_h w[i, h] * q[i, h, :]`` replaces
the whole per-head stage. The quadratic term drops the ``index_n_heads = 64``
factor. Measured 37-99x on 8xA800 in that comment.

THE IDENTITY IS NOT THIS OPERATOR. Every implementation of the C4 indexer, in
this fork and upstream, applies a per-head ``ReLU`` between the dot product and
the weighted head sum:

    logits[i, j] = sum_h w[i, h] * relu(sum_d q[i, h, d] * k[j, d]) * k_scale[j]

``relu`` is not linear, so the head sum cannot cross it. The head axis is
irreducible for this operator regardless of the MQA structure of K -- MQA is
necessary for the fold and is genuinely present here, but it is not sufficient,
and the missing condition is the one the upstream derivation does not mention.

The reference implementation, sgl-project/sglang#33271, confirms this from its
own diff: its folded Triton kernel ``_folded_paged_logits`` is a ``tl.dot``, a
causal ``tl.where`` and a store, with no ReLU anywhere, while the per-head
fallback it keeps twenty lines below still applies ``F.relu``. See
docs/dev/NOTE_440_c4_indexer_head_fold.md.

This file exists so the proposal cannot be adopted later by anyone reading only
the derivation. It pins four things:

* the ALGEBRA is right -- on a relu-free operator the fold reproduces the
  per-head result to fp32 rounding, so nothing here disputes the mathematics;
* the MQA PREMISE holds in this fork -- the index cache carries one shared K
  per position and the weights are indexed ``(row, head)`` only, so the fold's
  own stated precondition is satisfied and is not what rules it out;
* the OPERATOR is not linear -- the production torch path disagrees with the
  folded model grossly, not within any dtype tolerance, and the top-k it feeds
  overlaps the folded top-k at roughly a coin flip;
* the ReLU is the WHOLE of the difference -- with ``F.relu`` neutered inside the
  module the two agree again. That is this file's can-fail: it proves the
  divergence is caused by the term the derivation omits, and it fires the day
  somebody deletes the ``relu`` from the production path.

GPU-free: everything here is CPU float32.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import indexer as indexer_mod
from sglang.srt.layers.attention.dsv4 import indexer_arch
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.srt.layers.attention.dsv4.indexer_arch import deepgemm_indexer_supported
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

PAGE_SIZE = 64
HEAD_DIM = 128
PAGE_BYTES = PAGE_SIZE * HEAD_DIM + PAGE_SIZE * 4

INDEXER_MOD = "sglang.srt.layers.attention.dsv4.indexer"


def _build_inputs(seq_lens, *, num_heads=8, seed=0):
    """Paged FP8 index cache in the production layout, on CPU.

    Same construction as test_dsv4_indexer_seq_chunk_426.py; kept local so the
    files can drift apart without silently coupling their fixtures.

    Note the shape of the cache: ``[pages, PAGE_SIZE, 1, HEAD_DIM + 4]``. That
    ``1`` is the MQA premise in the layout -- one K vector per position, shared
    by all query heads. See :class:`TestMqaPremiseHolds`.
    """
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages_total = batch_size * max_pages + 1

    g = torch.Generator().manual_seed(seed)
    raw = torch.empty(num_pages_total, PAGE_BYTES, dtype=torch.uint8)
    values = (torch.randn(num_pages_total, PAGE_SIZE * HEAD_DIM, generator=g) * 0.5).to(
        FP8_DTYPE
    )
    raw[:, : PAGE_SIZE * HEAD_DIM] = values.view(dtype=torch.uint8)
    scales = torch.rand(num_pages_total, PAGE_SIZE, generator=g) + 0.5
    raw[:, PAGE_SIZE * HEAD_DIM :] = scales.contiguous().view(dtype=torch.uint8)

    kvcache = raw.view(num_pages_total, PAGE_SIZE, 1, HEAD_DIM + 4)

    page_table = torch.arange(1, batch_size * max_pages + 1, dtype=torch.int32).view(
        batch_size, max_pages
    )

    q = (torch.randn(batch_size, 1, num_heads, HEAD_DIM, generator=g) * 0.5).to(
        FP8_DTYPE
    )
    weight = torch.rand(batch_size, num_heads, generator=g) + 0.1
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

    return dict(
        q_fp8=q,
        kvcache_fp8=kvcache,
        weight=weight,
        seq_lens=seq_lens_t,
        page_table=page_table,
        deep_gemm_metadata=None,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )


def _unpack_cache(inputs):
    """Dequantise the paged index cache the way the production path does.

    Returns ``k[B, padded_seq, HEAD_DIM]`` float32 and ``k_scale[B, padded_seq]``,
    gathered through the page table so both references below see exactly the
    tensors the function under test sees.
    """
    kvcache = inputs["kvcache_fp8"]
    page_table = inputs["page_table"]
    batch_size = page_table.shape[0]
    max_pages = (inputs["max_seq_len"] + PAGE_SIZE - 1) // PAGE_SIZE
    scale_offset = PAGE_SIZE * HEAD_DIM

    flat = kvcache.view(-1, PAGE_SIZE * (HEAD_DIM + 4))
    gathered = flat[page_table[:, :max_pages]]

    k = (
        gathered[..., :scale_offset]
        .contiguous()
        .view(dtype=FP8_DTYPE)
        .to(torch.float32)
        .view(batch_size, max_pages * PAGE_SIZE, HEAD_DIM)
    )
    k_scale = (
        gathered[..., scale_offset:]
        .contiguous()
        .view(dtype=torch.float32)
        .view(batch_size, max_pages * PAGE_SIZE)
    )
    return k, k_scale


def _length_mask(logits, seq_lens):
    positions = torch.arange(logits.shape[1])
    return logits.masked_fill(
        positions.unsqueeze(0) >= seq_lens.unsqueeze(1), float("-inf")
    )


def _per_head_reference(inputs, *, relu: bool):
    """The definition, written out per head. ``relu=False`` is the linear twin."""
    k, k_scale = _unpack_cache(inputs)
    q = inputs["q_fp8"][:, 0].to(torch.float32)  # [B, H, D]
    score = torch.bmm(k, q.transpose(1, 2))  # [B, S, H]
    if relu:
        score = torch.relu(score)
    score = score * inputs["weight"].unsqueeze(1)
    score = score.sum(dim=2) * k_scale
    return _trim(score, inputs)


def _folded_reference(inputs):
    """The upstream fold: collapse the head axis into one effective query first.

    ``q_eff[i, :] = sum_h w[i, h] * q[i, h, :]``, then a plain matrix-vector
    product against the shared K. ``k_scale`` factors out per column and is
    applied afterwards exactly as the per-head path applies it -- the scale is
    not what the fold changes.
    """
    k, k_scale = _unpack_cache(inputs)
    q = inputs["q_fp8"][:, 0].to(torch.float32)  # [B, H, D]
    q_eff = (inputs["weight"].unsqueeze(-1) * q).sum(dim=1)  # [B, D]
    score = torch.bmm(k, q_eff.unsqueeze(-1)).squeeze(-1) * k_scale
    return _trim(score, inputs)


def _trim(score, inputs):
    max_seq_len = inputs["max_seq_len"]
    if score.shape[1] < max_seq_len:
        score = torch.nn.functional.pad(
            score, (0, max_seq_len - score.shape[1]), value=float("-inf")
        )
    else:
        score = score[:, :max_seq_len]
    return _length_mask(score, inputs["seq_lens"])


def _finite(logits, seq_lens):
    return torch.cat([logits[i, : int(n)] for i, n in enumerate(seq_lens)])


def _topk_overlap(a, b, k):
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    return len(ia & ib) / k


class _NoRelu:
    """Neuter ``F.relu`` inside the indexer module only.

    Used for the can-fail below. It turns the production operator into the
    linear one the upstream derivation assumes, without touching any other
    arithmetic in the function.
    """

    def __enter__(self):
        self._patch = mock.patch.object(
            indexer_mod.F, "relu", lambda x, *a, **kw: x, create=False
        )
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class TestFoldAlgebraIsCorrect(CustomTestCase):
    """Nothing here disputes the mathematics -- only its applicability.

    On a relu-free operator the fold is exact up to fp32 rounding, and the cost
    argument that motivates it is real: the folded form contracts the head axis
    once per query row instead of once per (query, KV) pair.
    """

    def test_fold_reproduces_the_linear_operator(self):
        inputs = _build_inputs([256, 192], num_heads=64, seed=3)
        linear = _finite(_per_head_reference(inputs, relu=False), inputs["seq_lens"])
        folded = _finite(_folded_reference(inputs), inputs["seq_lens"])
        # fp32 accumulation over 64 heads x 128 dims in a different order; the
        # tolerance is rounding, not model error.
        torch.testing.assert_close(folded, linear, atol=1e-3, rtol=1e-4)

    def test_folded_form_touches_the_head_axis_once_per_row(self):
        """The cost claim, in the shapes the two forms actually contract.

        Per head: one [S, D] x [D, H] product per KV position. Folded: one
        [H, D] reduction per query row, then [S, D] x [D, 1]. This is why the
        idea is attractive enough to need a written refusal.
        """
        inputs = _build_inputs([1024], num_heads=64, seed=4)
        k, _ = _unpack_cache(inputs)
        seq, heads = k.shape[1], inputs["weight"].shape[1]
        per_head_macs = seq * heads * HEAD_DIM
        folded_macs = heads * HEAD_DIM + seq * HEAD_DIM
        self.assertLess(folded_macs * 32, per_head_macs)


class TestMqaPremiseHolds(CustomTestCase):
    """The fold's stated precondition is satisfied here -- and is not the blocker.

    Upstream calls the MQA premise load-bearing, and it is: with per-head K the
    fold is false before any question of linearity arises. This fork does have
    one shared K per position, so if the operator were linear the fold would
    apply. Pinning it keeps the refusal honest about which premise fails.
    """

    def test_index_cache_carries_one_shared_k_per_position(self):
        inputs = _build_inputs([128], num_heads=8)
        self.assertEqual(inputs["kvcache_fp8"].shape[2], 1)
        # The production function asserts the same layout itself.
        fp8_paged_mqa_logits_torch_sm120(**inputs)

    def test_per_head_k_cache_is_rejected_not_silently_reduced(self):
        inputs = _build_inputs([128], num_heads=8)
        cache = inputs["kvcache_fp8"]
        inputs["kvcache_fp8"] = cache.expand(-1, -1, 2, -1)
        with self.assertRaises(AssertionError):
            fp8_paged_mqa_logits_torch_sm120(**inputs)

    def test_weights_are_indexed_by_row_and_head_only(self):
        inputs = _build_inputs([128], num_heads=8)
        self.assertEqual(inputs["weight"].dim(), 2)
        self.assertEqual(
            inputs["weight"].shape, (inputs["q_fp8"].shape[0], inputs["q_fp8"].shape[2])
        )


class TestOperatorIsNotLinearInTheHeadAxis(CustomTestCase):
    """The production paths apply a per-head ReLU, so the fold is not this operator.

    Both torch implementations are covered: the ``_sm120`` variant is what
    ``select_paged_mqa_logits_fn`` hands to every card without DeepGEMM since
    #417 Cut 3 (the sm86 3080s and the sm120 5090 on this rig), and
    ``fp8_paged_mqa_logits_torch`` is the reference twin the kernel tests
    compare against. They must refuse the fold identically.
    """

    def _both_paths(self, inputs):
        squeezed = dict(inputs)
        return [
            ("sm120", fp8_paged_mqa_logits_torch_sm120(**inputs)),
            ("reference", fp8_paged_mqa_logits_torch(**squeezed)),
        ]

    def test_production_output_matches_the_per_head_definition(self):
        """Anchor: the reference used below IS what the production path computes."""
        inputs = _build_inputs([320, 256], num_heads=64, seed=5)
        expected = _per_head_reference(inputs, relu=True)
        for name, got in self._both_paths(inputs):
            with self.subTest(path=name):
                torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-4)

    def test_fold_is_not_within_any_dtype_tolerance(self):
        inputs = _build_inputs([320, 256], num_heads=64, seed=5)
        folded = _finite(_folded_reference(inputs), inputs["seq_lens"])
        for name, got in self._both_paths(inputs):
            with self.subTest(path=name):
                actual = _finite(got, inputs["seq_lens"])
                rel = (actual - folded).abs().max() / actual.abs().max()
                # Not a precision question. bf16 round-off is ~1e-2 relative;
                # this is a different function.
                self.assertGreater(rel.item(), 0.1)

    def test_fold_loses_the_top_k_it_exists_to_produce(self):
        """These logits only ever feed a top-k, so score the thing that matters.

        Upstream reports 0.9967 top-k overlap at a 0.8% selection ratio for the
        folded kernel against a per-head baseline. Against the relu'd operator
        this fork runs, the overlap at a comparable ratio is near chance.
        """
        inputs = _build_inputs([4096], num_heads=64, seed=6)
        actual = fp8_paged_mqa_logits_torch_sm120(**inputs)[0, :4096]
        folded = _folded_reference(inputs)[0, :4096]
        overlap = _topk_overlap(actual, folded, 64)  # 1.6% selection ratio
        self.assertLess(overlap, 0.8)


class TestReluIsTheWholeDifference(CustomTestCase):
    """Can-fail: neuter the ReLU and the fold becomes exact again.

    This is the falsifier for everything above. If the divergence came from the
    fixture, the scale placement, or the summation order, removing the ReLU
    would not close it. It closes completely, which identifies the ReLU as the
    single term the upstream derivation omits -- and makes this test fail the
    day the ``F.relu`` disappears from the production path.
    """

    def test_without_relu_the_production_path_equals_the_fold(self):
        inputs = _build_inputs([320, 256], num_heads=64, seed=5)
        folded = _finite(_folded_reference(inputs), inputs["seq_lens"])
        with _NoRelu():
            got = _finite(
                fp8_paged_mqa_logits_torch_sm120(**inputs), inputs["seq_lens"]
            )
        torch.testing.assert_close(got, folded, atol=1e-3, rtol=1e-4)

    def test_with_relu_the_same_comparison_fails(self):
        inputs = _build_inputs([320, 256], num_heads=64, seed=5)
        folded = _finite(_folded_reference(inputs), inputs["seq_lens"])
        got = _finite(fp8_paged_mqa_logits_torch_sm120(**inputs), inputs["seq_lens"])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(got, folded, atol=1e-3, rtol=1e-4)

    def test_relu_actually_bites_on_this_fixture(self):
        """Guards the guard: a fixture with no negative products proves nothing."""
        inputs = _build_inputs([320, 256], num_heads=64, seed=5)
        k, _ = _unpack_cache(inputs)
        q = inputs["q_fp8"][:, 0].to(torch.float32)
        raw = torch.bmm(k, q.transpose(1, 2))
        self.assertGreater((raw < 0).float().mean().item(), 0.2)


class TestNonPagedCouplingIsAlreadyArchGuarded(CustomTestCase):
    """#440 item 4 -- the ``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`` coupling.

    Upstream ``_can_use_nonpaged_indexer`` disables the non-paged DeepGEMM fast
    path whenever that env is set, with no architecture guard ahead of it
    (upstream indexer.py:490-508 on main at the time of writing). On Ampere that
    env was the only way to reach a working paged path, so asking for it also
    cost the non-paged prefill route -- a coupling worth breaking there.

    In this fork the coupling is defanged rather than present-and-harmful, and
    the decoupling upstream needs an arch guard for is already the arch guard
    this fork added in #417 Cut 3. ``indexer.py:596`` rejects the non-paged
    branch on any card without DeepGEMM and runs BEFORE the env clause at
    ``indexer.py:612``, so an sm8x rank can never be offered a route into
    ``deep_gemm.fp8_mqa_logits``. And since #417 nobody has to set the env on
    such a card at all: ``resolve_paged_mqa_logits_backend`` picks the torch
    paged path from the capability.

    What survives is the documented doctrine of ``indexer_arch.py`` -- an
    explicit env selection is a statement about the launch, not a probe, so
    "do not use DeepGEMM for indexer logits" is honoured on both routes. These
    tests pin the ordering that makes that safe.
    """

    def setUp(self):
        super().setUp()
        deepgemm_indexer_supported.cache_clear()
        self.addCleanup(deepgemm_indexer_supported.cache_clear)

    def _is_eligible(self, capability, *, torch_env: bool):
        from types import SimpleNamespace

        from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_parallel

        deepgemm_indexer_supported.cache_clear()
        with (
            mock.patch.multiple(
                indexer_arch,
                is_cuda=lambda: True,
                get_device_capability_no_init=lambda device_id: capability,
            ),
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(True),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(torch_env),
            mock.patch(f"{INDEXER_MOD}.is_cuda", return_value=True),
            mock.patch(f"{INDEXER_MOD}.is_hip", return_value=False),
            get_parallel().override(attn_cp_size=1),
            mock.patch(
                f"{INDEXER_MOD}.is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch(f"{INDEXER_MOD}.is_in_breakable_cuda_graph", return_value=False),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            return C4IndexerBackendMixin._can_use_nonpaged_indexer(
                SimpleNamespace(hisparse_coordinator=None),
                c4_indexer=SimpleNamespace(use_fp4_indexer=False),
                forward_batch=SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    _original_forward_mode=None,
                    tbo_parent_token_range=None,
                    batch_size=1,
                ),
                indexer_metadata=SimpleNamespace(use_prefill_cuda_graph=False),
            )

    def test_no_deepgemm_card_is_refused_whatever_the_env_says(self):
        """The refusal is architectural, so decoupling the env cannot expose it."""
        for capability in ((8, 0), (8, 6), (8, 9), (12, 0)):
            for torch_env in (False, True):
                with self.subTest(sm=capability, env=torch_env):
                    self.assertFalse(self._is_eligible(capability, torch_env=torch_env))

    def test_the_env_clause_only_bites_where_deepgemm_exists(self):
        for capability in ((9, 0), (10, 0)):
            with self.subTest(sm=capability):
                self.assertTrue(self._is_eligible(capability, torch_env=False))
                self.assertFalse(self._is_eligible(capability, torch_env=True))

    def test_torch_paged_path_needs_no_env_on_a_card_without_deepgemm(self):
        """Why the upstream coupling has no victim here: nobody must set it."""
        from sglang.srt.layers.attention.dsv4.indexer_arch import (
            BACKEND_TORCH,
            resolve_paged_mqa_logits_backend,
        )

        for capability in ((8, 6), (12, 0)):
            with self.subTest(sm=capability):
                deepgemm_indexer_supported.cache_clear()
                with (
                    mock.patch.multiple(
                        indexer_arch,
                        is_cuda=lambda: True,
                        get_device_capability_no_init=lambda device_id: capability,
                    ),
                    envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
                    envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
                    envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
                ):
                    self.assertEqual(resolve_paged_mqa_logits_backend(0), BACKEND_TORCH)


class TestRowZeroPageTableReuseIsGuarded(CustomTestCase):
    """#440 -- the two traps the upstream fold hit, audited against this fork.

    Upstream's folded kernel reuses row 0's page table for a whole tile to save
    registers (sgl-project/sglang#33246 comment 5159716067). That is sound for
    single-sequence chunked prefill and silently reads the wrong KV for a batch
    of different requests; the uniformity check added to guard it was a
    device-to-host sync that invalidated CUDA graph capture on first call.

    This fork makes the same row-0 assumption in a different place:
    ``_get_nonpaged_indexer_plan`` builds its plan from ``page_table[:1]``
    (indexer.py:690). It is guarded structurally rather than by a runtime probe
    -- ``_can_use_nonpaged_indexer`` refuses any batch that is not a single
    request (indexer.py:604) and refuses capture outright (indexer.py:622) --
    so neither trap is reachable. Both properties are pinned here because they
    are load-bearing and invisible at the point where the assumption is made.
    """

    @staticmethod
    def _backend():
        """A real mixin instance -- ``_get_nonpaged_indexer_plan`` calls back
        into ``_can_use_nonpaged_indexer`` through ``self``."""
        from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin

        class _Backend(C4IndexerBackendMixin):
            def __init__(self):
                super().__init__()
                self.hisparse_coordinator = None

        return _Backend()

    @staticmethod
    def _ctx(*, capturing=False):
        from sglang.srt.runtime_context import get_parallel

        return (
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(True),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            mock.patch(f"{INDEXER_MOD}.is_cuda", return_value=True),
            mock.patch(f"{INDEXER_MOD}.is_hip", return_value=False),
            mock.patch(f"{INDEXER_MOD}.deepgemm_indexer_supported", return_value=True),
            mock.patch(
                f"{INDEXER_MOD}.is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch(f"{INDEXER_MOD}.is_in_breakable_cuda_graph", return_value=False),
            mock.patch(
                "torch.cuda.is_current_stream_capturing", return_value=capturing
            ),
            get_parallel().override(attn_cp_size=1),
        )

    def _eligible(self, *, batch_size, capturing=False):
        from types import SimpleNamespace

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        with contextlib.ExitStack() as stack:
            for ctx in self._ctx(capturing=capturing):
                stack.enter_context(ctx)
            return self._backend()._can_use_nonpaged_indexer(
                c4_indexer=SimpleNamespace(use_fp4_indexer=False),
                forward_batch=SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    _original_forward_mode=None,
                    tbo_parent_token_range=None,
                    batch_size=batch_size,
                ),
                indexer_metadata=SimpleNamespace(use_prefill_cuda_graph=False),
            )

    def test_a_multi_request_batch_never_reaches_the_row_zero_plan(self):
        self.assertTrue(self._eligible(batch_size=1))
        for batch_size in (2, 3, 8):
            with self.subTest(batch_size=batch_size):
                self.assertFalse(self._eligible(batch_size=batch_size))

    def test_capture_never_reaches_it_either(self):
        self.assertFalse(self._eligible(batch_size=1, capturing=True))

    def _plan(self, *, seq_lens_cpu, extend_seq_lens_cpu, query_rows=8192):
        from types import SimpleNamespace

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
            batch_size=1,
            seq_lens=torch.tensor([query_rows * 4], dtype=torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(
            use_prefill_cuda_graph=False, nonpaged_plan=None, c4_page_size=64
        )
        with contextlib.ExitStack() as stack:
            for ctx in self._ctx():
                stack.enter_context(ctx)
            return self._backend()._get_nonpaged_indexer_plan(
                c4_indexer=SimpleNamespace(use_fp4_indexer=False),
                forward_batch=forward_batch,
                indexer_metadata=metadata,
                page_table=torch.zeros((query_rows, 64), dtype=torch.int32),
                c4_seq_lens=torch.ones(query_rows, dtype=torch.int32),
                query_rows=query_rows,
            )

    def test_the_plan_builder_forces_no_device_to_host_sync(self):
        """The guard upstream needed is one this fork never has to run.

        Eligibility is decided from python-side batch metadata, and the plan is
        built from tensors that are already on the host -- nothing here reads a
        device tensor back, so there is no ``.item()`` to invalidate a capture.
        """
        calls = []
        real_item = torch.Tensor.item

        def probed(self, *a, **kw):
            calls.append(tuple(self.shape))
            return real_item(self, *a, **kw)

        with mock.patch.object(torch.Tensor, "item", probed):
            plan = self._plan(
                seq_lens_cpu=torch.tensor([32768], dtype=torch.int32),
                extend_seq_lens_cpu=torch.tensor([8192], dtype=torch.int32),
            )
        self.assertIsNotNone(plan)
        self.assertEqual(calls, [])

    def test_a_device_resident_length_bails_instead_of_syncing(self):
        """Can-fail for the line above: make the metadata non-host and the plan
        refuses itself rather than pulling it across."""
        plan = self._plan(
            seq_lens_cpu=torch.zeros(1, dtype=torch.int32, device="meta"),
            extend_seq_lens_cpu=torch.tensor([8192], dtype=torch.int32),
        )
        self.assertIsNone(plan)


if __name__ == "__main__":
    unittest.main()
