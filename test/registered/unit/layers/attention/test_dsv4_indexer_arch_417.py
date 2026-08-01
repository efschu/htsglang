"""#417 Cut 3 -- the DSpark indexer needs a route on cards without DeepGEMM.

The DSpark layers of DeepSeek-V4-Flash (layers 40-42, `compress_ratio` 4/128)
run a paged-MQA-logits step whose default implementation is DeepGEMM: Hopper
and datacenter Blackwell only. A torch implementation has existed since #24692
but was reachable only by setting `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`, so an
Ampere or consumer-Blackwell rank went to DeepGEMM anyway and died there --
one step past where Cuts 1 and 2 leave it.

Two properties have to hold and are pinned here:

* the kernel choice and the schedule-metadata choice can never disagree. A
  rank with a DeepGEMM schedule and no DeepGEMM, or a DeepGEMM kernel and no
  schedule, is a crash either way;
* the substitution is NOT bit-identical to DeepGEMM and must be named. These
  logits feed a top-k, so a near-tie between two KV positions can be broken
  differently. Silence there would look like a mysterious quality difference
  between ranks.

GPU-free: capability is mocked, never sniffed.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import indexer_arch
from sglang.srt.layers.attention.dsv4.indexer_arch import (
    BACKEND_AITER,
    BACKEND_DEEPGEMM,
    BACKEND_TILELANG,
    BACKEND_TORCH,
    deepgemm_indexer_metadata_needed,
    deepgemm_indexer_supported,
    reset_substitution_warnings,
    resolve_paged_mqa_logits_backend,
    warn_torch_indexer_substitution_once,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# (capability, DeepGEMM's paged-MQA-logits kernel exists here)
_CAPABILITY_MATRIX = [
    ((8, 0), False),  # A100
    ((8, 6), False),  # RTX 3080 -- TP1/TP2 of this rig
    ((8, 9), False),  # L20 / Ada
    ((9, 0), True),  # H100
    ((10, 0), True),  # B200
    ((10, 3), True),  # GB300
    ((12, 0), False),  # RTX 5090 -- TP0 of this rig; DeepGEMM declined SM12x
    ((12, 1), False),  # DGX Spark GB10
]

_ENV_COMBINATIONS = [
    # (tilelang, aiter, torch)
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
]


class _ArchMixin:
    def setUp(self):
        super().setUp()
        self._clear()
        self.addCleanup(self._clear)

    @staticmethod
    def _clear():
        deepgemm_indexer_supported.cache_clear()
        reset_substitution_warnings()

    def _with_capability(self, capability, cuda=True):
        self._clear()
        return mock.patch.multiple(
            indexer_arch,
            is_cuda=lambda: cuda,
            get_device_capability_no_init=lambda device_id: capability,
        )

    def _with_capabilities(self, per_device, cuda=True):
        self._clear()
        return mock.patch.multiple(
            indexer_arch,
            is_cuda=lambda: cuda,
            get_device_capability_no_init=lambda device_id: per_device[device_id],
        )

    @staticmethod
    def _with_envs(tilelang=False, aiter=False, torch_impl=False):
        return (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(tilelang),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(aiter),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(torch_impl),
        )


class TestDeepgemmIndexerGate(_ArchMixin, CustomTestCase):
    def test_capability_domain(self):
        for capability, expected in _CAPABILITY_MATRIX:
            with self.subTest(sm=capability), self._with_capability(capability):
                self.assertEqual(deepgemm_indexer_supported(0), expected)

    def test_hopper_and_datacenter_blackwell_are_unchanged(self):
        """Backward compatibility: only devices that currently crash may move."""
        for capability in ((9, 0), (10, 0), (10, 3)):
            with self.subTest(sm=capability), self._with_capability(capability):
                self.assertTrue(deepgemm_indexer_supported(0))
                self.assertEqual(resolve_paged_mqa_logits_backend(0), BACKEND_DEEPGEMM)
                self.assertTrue(deepgemm_indexer_metadata_needed(0))

    def test_gate_answers_per_device(self):
        """#343: the 5090 and a 3080 in one process. Here they happen to agree
        that DeepGEMM is absent, so the discriminating pair is Hopper+Ampere."""
        with self._with_capabilities({0: (9, 0), 1: (8, 6)}):
            self.assertTrue(deepgemm_indexer_supported(0))
            self.assertFalse(deepgemm_indexer_supported(1))
            self.assertTrue(deepgemm_indexer_supported(0))


class TestBackendResolution(_ArchMixin, CustomTestCase):
    def test_ampere_and_consumer_blackwell_auto_select_torch(self):
        for capability in ((8, 0), (8, 6), (8, 9), (12, 0), (12, 1)):
            with self.subTest(sm=capability), self._with_capability(capability):
                self.assertEqual(resolve_paged_mqa_logits_backend(0), BACKEND_TORCH)

    def test_explicit_selections_win_everywhere(self):
        cases = [
            (dict(tilelang=True), BACKEND_TILELANG),
            (dict(aiter=True), BACKEND_AITER),
            (dict(torch_impl=True), BACKEND_TORCH),
        ]
        for env_kwargs, expected in cases:
            for capability in ((8, 6), (9, 0), (12, 0)):
                with self.subTest(env=env_kwargs, sm=capability):
                    with self._with_capability(capability):
                        overrides = self._with_envs(**env_kwargs)
                        with overrides[0], overrides[1], overrides[2]:
                            self.assertEqual(
                                resolve_paged_mqa_logits_backend(0), expected
                            )

    def test_metadata_and_kernel_can_never_disagree(self):
        """The safety invariant, over the whole env x capability matrix.

        Never a DeepGEMM kernel without its schedule; never a schedule built
        on a card that has no DeepGEMM to build it with.
        """
        for capability, deepgemm_here in _CAPABILITY_MATRIX:
            for tilelang, aiter, torch_impl in _ENV_COMBINATIONS:
                with self.subTest(
                    sm=capability, tilelang=tilelang, aiter=aiter, torch=torch_impl
                ):
                    with self._with_capability(capability):
                        overrides = self._with_envs(tilelang, aiter, torch_impl)
                        with overrides[0], overrides[1], overrides[2]:
                            backend = resolve_paged_mqa_logits_backend(0)
                            needs = deepgemm_indexer_metadata_needed(0)
                            if backend == BACKEND_DEEPGEMM:
                                self.assertTrue(
                                    needs,
                                    "DeepGEMM kernel selected without its schedule",
                                )
                            if needs:
                                self.assertTrue(
                                    deepgemm_here,
                                    "schedule built on a card without DeepGEMM",
                                )


class TestSubstitutionIsNamed(_ArchMixin, CustomTestCase):
    def test_warning_names_the_device_and_the_consequence(self):
        with self._with_capability((8, 6)):
            with self.assertLogs(indexer_arch.logger, level="WARNING") as logs:
                warn_torch_indexer_substitution_once(1)
        text = "\n".join(logs.output)
        self.assertIn("8.6", text)
        self.assertIn("top-k", text)
        self.assertIn("not identical", text)

    def test_warning_fires_once_per_device_not_once_per_call(self):
        with self._with_capabilities({0: (8, 6), 1: (8, 6)}):
            with self.assertLogs(indexer_arch.logger, level="WARNING") as logs:
                warn_torch_indexer_substitution_once(0)
                warn_torch_indexer_substitution_once(0)
                warn_torch_indexer_substitution_once(0)
            self.assertEqual(len(logs.output), 1)

            with self.assertLogs(indexer_arch.logger, level="WARNING") as logs:
                warn_torch_indexer_substitution_once(1)
            self.assertEqual(len(logs.output), 1)

    def test_it_can_fail(self):
        """Can-fail proof: with the warning suppressed, the assertion breaks."""
        reset_substitution_warnings()
        warn_torch_indexer_substitution_once(0)  # consume the one-shot
        with self.assertRaises(AssertionError):
            with self.assertLogs(indexer_arch.logger, level="WARNING"):
                warn_torch_indexer_substitution_once(0)


class TestPagedIndexerMetadataBypass(_ArchMixin, CustomTestCase):
    """The integration half: constructing the metadata on a card without
    DeepGEMM must not import or call DeepGEMM."""

    def _build(self, capability):
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

        with self._with_capability(capability), envs.SGLANG_OPT_USE_TOPK_V2.override(
            False
        ):
            return PagedIndexerMetadata(
                page_size=256,
                page_table=torch.zeros(2, 4, dtype=torch.int32),
                c4_seq_lens=torch.ones(2, dtype=torch.int32),
            )

    def test_no_deepgemm_schedule_on_ampere(self):
        metadata = self._build((8, 6))
        self.assertIsNone(metadata.deep_gemm_metadata)

    def test_no_deepgemm_schedule_on_consumer_blackwell(self):
        metadata = self._build((12, 0))
        self.assertIsNone(metadata.deep_gemm_metadata)

    def test_hopper_still_tries_to_build_one(self):
        """The bypass must not have swallowed the Hopper path. There is no
        DeepGEMM in this CPU environment, so 'tries' is all that can be
        observed -- but that is exactly the branch that must be taken, and a
        bypass that fired everywhere would return None instead of raising.
        """
        with self.assertRaises(Exception) as ctx:
            self._build((9, 0))
        self.assertNotIsInstance(ctx.exception, AssertionError)


class TestTorchVariantChoice(_ArchMixin, CustomTestCase):
    """Which of the two torch implementations the production call site can use.

    They are not interchangeable, and the difference is not architectural. The
    paged call site passes `seq_lens` with a trailing dim of 1; only the
    trimmed variant squeezes it. Routing a card at `fp8_paged_mqa_logits_torch`
    -- which is what the old `is_sm120_supported()` split did for anything that
    was not SM120 -- therefore asserts rather than falls back.
    """

    @staticmethod
    def _inputs(batch=1, heads=2, head_dim=128, block=64, pages=1):
        q = torch.zeros(batch, 1, heads, head_dim, dtype=torch.uint8).view(
            dtype=torch.float8_e4m3fn
        )
        kv = torch.zeros(pages, block, 1, head_dim + 4, dtype=torch.uint8)
        weight = torch.ones(batch, heads, dtype=torch.float32)
        page_table = torch.zeros(batch, pages, dtype=torch.int64)
        return q, kv, weight, page_table

    def test_trimmed_variant_accepts_the_shape_the_call_site_passes(self):
        from sglang.srt.layers.attention.dsv4.indexer import (
            fp8_paged_mqa_logits_torch_sm120,
        )

        q, kv, weight, page_table = self._inputs()
        seq_lens_2d = torch.ones(1, 1, dtype=torch.int32)
        out = fp8_paged_mqa_logits_torch_sm120(
            q, kv, weight, seq_lens_2d, page_table, None, 64, False
        )
        self.assertEqual(out.shape, (1, 64))

    def test_untrimmed_variant_rejects_it(self):
        """The reason the dispatch may not send a non-SM120 card there."""
        from sglang.srt.layers.attention.dsv4.indexer import (
            fp8_paged_mqa_logits_torch,
        )

        q, kv, weight, page_table = self._inputs()
        seq_lens_2d = torch.ones(1, 1, dtype=torch.int32)
        with self.assertRaises(AssertionError):
            fp8_paged_mqa_logits_torch(
                q, kv, weight, seq_lens_2d, page_table, None, 64, False
            )


class TestDispatchSelectsTheImplementation(_ArchMixin, CustomTestCase):
    """`select_paged_mqa_logits_fn` is the whole of Cut 3; pin what it returns."""

    def _select(self, capability, use_fp4=False, **env_kwargs):
        from sglang.srt.layers.attention.dsv4.indexer import (
            select_paged_mqa_logits_fn,
        )

        with self._with_capability(capability):
            overrides = self._with_envs(**env_kwargs)
            with overrides[0], overrides[1], overrides[2]:
                return select_paged_mqa_logits_fn(
                    device=torch.device("cuda", 0), use_fp4_indexer=use_fp4
                )

    def test_ampere_gets_the_trimmed_torch_implementation(self):
        from sglang.srt.layers.attention.dsv4.indexer import (
            fp8_paged_mqa_logits_torch_sm120,
        )

        for capability in ((8, 0), (8, 6), (8, 9), (12, 0)):
            with self.subTest(sm=capability):
                self.assertIs(
                    self._select(capability), fp8_paged_mqa_logits_torch_sm120
                )

    def test_it_is_never_the_untrimmed_one(self):
        """The old `is_sm120_supported()` split sent everything non-SM120 here,
        where the 2-D `seq_lens` this call site passes trips an assert."""
        from sglang.srt.layers.attention.dsv4.indexer import (
            fp8_paged_mqa_logits_torch,
        )

        for capability in ((8, 6), (12, 0)):
            for env_kwargs in ({}, {"torch_impl": True}):
                with self.subTest(sm=capability, env=env_kwargs):
                    self.assertIsNot(
                        self._select(capability, **env_kwargs),
                        fp8_paged_mqa_logits_torch,
                    )

    def test_hopper_still_reaches_for_deepgemm(self):
        """No DeepGEMM in this CPU environment, so the import raising is the
        observable proof that the DeepGEMM branch was taken."""
        with self.assertRaises(Exception) as ctx:
            self._select((9, 0))
        self.assertNotIsInstance(ctx.exception, AssertionError)

    def test_fp4_indexer_refuses_by_name_instead_of_falling_back(self):
        """There is no non-DeepGEMM FP4 indexer. Routing an FP4 checkpoint at
        the FP8 torch path would read the index cache with the wrong layout and
        return plausible numbers, which is worse than stopping."""
        with self.assertRaises(RuntimeError) as ctx:
            self._select((8, 6), use_fp4=True)
        message = str(ctx.exception)
        self.assertIn("FP4", message)
        self.assertIn("8.6", message)
        self.assertIn("fp8_fp4_paged_mqa_logits", message)

    def test_the_warning_fires_only_when_nobody_asked(self):
        with self._with_capability((8, 6)):
            with self.assertLogs(indexer_arch.logger, level="WARNING"):
                self._select((8, 6))
        # Explicitly requested: the same substitution, but not a surprise.
        self._clear()
        with self.assertRaises(AssertionError):
            with self.assertLogs(indexer_arch.logger, level="WARNING"):
                self._select((8, 6), torch_impl=True)


class TestNonPagedIndexerNeedsDeepgemm(_ArchMixin, CustomTestCase):
    """The non-paged branch calls `deep_gemm.fp8_mqa_logits` with no torch twin,
    so a card without DeepGEMM must be sent back to the paged path."""

    def _is_eligible(self, capability):
        from types import SimpleNamespace

        from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_parallel

        indexer_mod = "sglang.srt.layers.attention.dsv4.indexer"
        with self._with_capability(
            capability
        ), envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(
            True
        ), envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(
            False
        ), envs.SGLANG_OPT_USE_AITER_INDEXER.override(
            False
        ), envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(
            False
        ), mock.patch(
            f"{indexer_mod}.is_cuda", return_value=True
        ), mock.patch(
            f"{indexer_mod}.is_hip", return_value=False
        ), get_parallel().override(
            attn_cp_size=1
        ), mock.patch(
            f"{indexer_mod}.is_in_tc_piecewise_cuda_graph", return_value=False
        ), mock.patch(
            f"{indexer_mod}.is_in_breakable_cuda_graph", return_value=False
        ), mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
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

    def test_hopper_still_eligible(self):
        self.assertTrue(self._is_eligible((9, 0)))

    def test_ampere_and_consumer_blackwell_are_not(self):
        for capability in ((8, 0), (8, 6), (12, 0)):
            with self.subTest(sm=capability):
                self.assertFalse(self._is_eligible(capability))


if __name__ == "__main__":
    unittest.main()
