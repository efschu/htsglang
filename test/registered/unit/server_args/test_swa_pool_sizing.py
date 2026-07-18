"""Unit tests for --swa-pool-sizing (task #91 Stage A): constant window-cap
SWA pool sizing for hybrid sliding-window models. CPU only."""

import argparse
import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.pool_configurator import (
    SWAChunkCapPoolConfigurator,
    swa_pool_token_cap,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def make_mr(
    disable_radix_cache,
    swa_pool_sizing,
    max_running_requests=4,
    chunked_prefill_size=2048,
    sliding_window_size=1024,
    full_layers=10,
):
    """Minimal fake ModelRunner for the configurator applicability check."""
    sa = SimpleNamespace(
        max_running_requests=max_running_requests,
        disable_radix_cache=disable_radix_cache,
        chunked_prefill_size=chunked_prefill_size,
        swa_pool_sizing=swa_pool_sizing,
        swa_full_tokens_ratio=0.8,
        speculative_num_draft_tokens=None,
        speculative_algorithm=None,
        disable_overlap_schedule=True,
        disaggregation_mode="null",
    )
    mc = SimpleNamespace(full_attention_layer_ids=list(range(full_layers)))
    return SimpleNamespace(
        server_args=sa,
        model_config=mc,
        sliding_window_size=sliding_window_size,
        page_size=1,
        dp_size=1,
    )


class TestIsApplicable(CustomTestCase):
    def test_ratio_mode_radix_on_not_applicable(self):
        # Default path byte-identity: radix cache on + ratio mode keeps the
        # ratio configurator (legacy behavior).
        self.assertFalse(
            SWAChunkCapPoolConfigurator.is_applicable(make_mr(False, "ratio"))
        )

    def test_legacy_radix_off_route_unchanged(self):
        self.assertTrue(
            SWAChunkCapPoolConfigurator.is_applicable(make_mr(True, "ratio"))
        )

    def test_cap_mode_allows_radix_cache(self):
        self.assertTrue(
            SWAChunkCapPoolConfigurator.is_applicable(make_mr(False, "cap"))
        )

    def test_cap_mode_requires_full_layers(self):
        self.assertFalse(
            SWAChunkCapPoolConfigurator.is_applicable(
                make_mr(False, "cap", full_layers=0)
            )
        )

    def test_cap_mode_requires_sliding_window(self):
        self.assertFalse(
            SWAChunkCapPoolConfigurator.is_applicable(
                make_mr(False, "cap", sliding_window_size=None)
            )
        )

    def test_cap_mode_requires_max_running_requests(self):
        self.assertFalse(
            SWAChunkCapPoolConfigurator.is_applicable(
                make_mr(False, "cap", max_running_requests=None)
            )
        )


class TestSwaPoolTokenCap(CustomTestCase):
    def test_cap_is_window_bounded_not_context_bounded(self):
        # The pin must scale with window/reqs/chunks, never with context
        # length (there is no context term in the formula at all).
        mr = make_mr(False, "cap")
        cap = swa_pool_token_cap(mr, 4)
        # 4 * (window 1024 + eviction lag + decode alloc) + chunks + page:
        # well under any long-context full-pool need.
        self.assertGreater(cap, 4 * 1024)
        self.assertLess(cap, 4 * 1024 + 4 * 2048 + 4096)


class TestCliAndValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def parse(self, *extra):
        return self.parser.parse_args(["--model-path", "m", *extra])

    def test_default_is_ratio(self):
        self.assertEqual(self.parse().swa_pool_sizing, "ratio")

    def test_cap_parses(self):
        self.assertEqual(
            self.parse("--swa-pool-sizing", "cap").swa_pool_sizing, "cap"
        )

    def test_invalid_choice_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--swa-pool-sizing", "bogus")

    def _validate(self, **kwargs):
        # model_path='dummy' short-circuits __post_init__ (same pattern as
        # test_uneven_tp_args), so the validation helper runs in isolation.
        args = ServerArgs(model_path="dummy", **kwargs)
        args._handle_cache_compatibility()
        return args

    def test_cap_requires_max_running_requests(self):
        with self.assertRaisesRegex(ValueError, "max-running-requests"):
            self._validate(swa_pool_sizing="cap", chunked_prefill_size=2048)

    def test_cap_requires_chunked_prefill(self):
        with self.assertRaisesRegex(ValueError, "chunked prefill"):
            self._validate(
                swa_pool_sizing="cap",
                max_running_requests=4,
                chunked_prefill_size=-1,
            )

    def test_cap_accepted_with_preconditions(self):
        args = self._validate(
            swa_pool_sizing="cap",
            max_running_requests=4,
            chunked_prefill_size=2048,
        )
        self.assertEqual(args.swa_pool_sizing, "cap")

    def test_ratio_mode_never_validates_cap_rules(self):
        args = self._validate(swa_pool_sizing="ratio")
        self.assertEqual(args.swa_pool_sizing, "ratio")


class TestSwaHybridCapModeCeiling(CustomTestCase):
    """#90 cap in cap mode: full_need only (no ceil(swa_need/ratio) term)."""

    def _cap_for(self, sizing):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        sa = SimpleNamespace(
            max_running_requests=4,
            swa_full_tokens_ratio=0.8,
            swa_pool_sizing=sizing,
            chunked_prefill_size=2048,
            disable_radix_cache=False,
            speculative_num_draft_tokens=None,
            max_speculative_num_draft_tokens=None,
            speculative_algorithm=None,
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            disable_overlap_schedule=True,
            disaggregation_mode="null",
        )
        mr = SimpleNamespace(
            is_hybrid_swa=True,
            mambaish_config=None,
            server_args=sa,
            model_config=SimpleNamespace(
                context_len=8192,
                full_attention_layer_ids=list(range(10)),
                hf_config=SimpleNamespace(architectures=["Gemma4ForCausalLM"]),
            ),
            sliding_window_size=1024,
            page_size=1,
            dp_size=1,
        )
        return ModelRunnerKVCacheMixin._swa_hybrid_kv_token_cap(mr)

    def test_cap_mode_returns_full_need_only(self):
        ratio_cap = self._cap_for("ratio")
        cap_cap = self._cap_for("cap")
        # full_need = 4 * (8192 + headroom); both modes share it.
        self.assertIsNotNone(cap_cap)
        self.assertGreaterEqual(cap_cap, 4 * 8192)
        # ratio mode >= cap mode always (max(full_need, swa term) vs full_need)
        self.assertGreaterEqual(ratio_cap, cap_cap)
        # At this shape (small swa_need) both bind at full_need: byte-identity
        # of the ratio formula is asserted indirectly by equality here.
        self.assertEqual(ratio_cap, cap_cap)


if __name__ == "__main__":
    unittest.main()
