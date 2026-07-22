"""Unit tests for the T156 deterministic drafter policy (force=policy).

CPU-only tests of the pure pieces: policy-table parsing/validation, the
default switch-point derivation from the drafter config (training context =
2 x sliding window -- deliberately below the 4 x window ctx gate), the
table lookup, and the gate-as-safety-filter fallback in policy_select. GPU
behavior is covered by the live validation protocol, not here.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.speculative.cross_algo_utils import (
    derive_policy_switch_ctx,
    policy_lookup_index,
    policy_select,
    resolve_drafter_policy_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

DFLASH16 = ("dflash", 16)
NEXTN3 = ("nextn", 3)
NEXTN2 = ("nextn", 2)
ARM_KS = {1, 2, 3, 4, 5}

TWO_STAGE = [(0, DFLASH16), (4096, NEXTN3)]
THREE_STAGE = [(0, DFLASH16), (4096, NEXTN3), (65536, NEXTN2)]


def _resolve(raw, **kw):
    base = dict(
        arm_ks=ARM_KS,
        dflash_block=16,
        primary_k=3,
        draft_model_path="/nonexistent",
    )
    base.update(kw)
    return resolve_drafter_policy_table(raw, **base)


class TestSwitchCtxDerivation(CustomTestCase):
    def _write_cfg(self, tmpdir, cfg):
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f)

    def test_zlab_swa_drafter_derives_training_ctx_4096(self):
        # The z-lab drafter: window 2048, trained at ctx 4096 = 2 * window.
        # The GATE derives 8192 (4 * window) from the same config -- the
        # policy switch point must be the training context, not the gate.
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(
                d,
                {
                    "max_position_embeddings": 262144,
                    "sliding_window": 2048,
                    "use_sliding_window": True,
                },
            )
            ctx, source = derive_policy_switch_ctx(d)
        self.assertEqual(ctx, 4096)
        self.assertIn("training context", source)

    def test_factor_env_override(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(d, {"sliding_window": 2048})
            os.environ["SGLANG_CROSS_POLICY_CTX_FACTOR"] = "3"
            try:
                ctx, _ = derive_policy_switch_ctx(d)
            finally:
                del os.environ["SGLANG_CROSS_POLICY_CTX_FACTOR"]
        self.assertEqual(ctx, 6144)

    def test_capped_at_mpe(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(
                d, {"max_position_embeddings": 3000, "sliding_window": 2048}
            )
            ctx, _ = derive_policy_switch_ctx(d)
        self.assertEqual(ctx, 3000)

    def test_no_swa_uses_mpe(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(d, {"max_position_embeddings": 65536})
            ctx, source = derive_policy_switch_ctx(d)
        self.assertEqual(ctx, 65536)
        self.assertIn("max_position_embeddings", source)

    def test_unreadable_config_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, source = derive_policy_switch_ctx(d)
        self.assertIsNone(ctx)
        self.assertIn("unreadable", source)


class TestTableResolution(CustomTestCase):
    def test_auto_default_two_stages(self):
        # The derived default: DFLASH below the training ctx, nextn:auto
        # (task-A analytic k, sentinel 0) above.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"sliding_window": 2048}, f)
            for raw in (None, "auto", " AUTO "):
                table, source = _resolve(raw, draft_model_path=d)
                self.assertEqual(
                    table, [(0, DFLASH16), (4096, ("nextn", 0))]
                )
                self.assertIn("auto", source)

    def test_explicit_nextn_auto_stage(self):
        table, _ = _resolve("0:dflash:16,4096:nextn:auto")
        self.assertEqual(table, [(0, DFLASH16), (4096, ("nextn", 0))])
        # 'auto' is a nextn-only value.
        with self.assertRaises(ValueError):
            _resolve("0:dflash:auto,4096:nextn:3")

    def test_policy_select_passes_auto_stage_through(self):
        from sglang.srt.speculative.cross_algo_utils import policy_select

        table = [(0, DFLASH16), (4096, ("nextn", 0))]
        rung, _, _ = policy_select(table, [(5000, 64)], 8192, 0.8, NEXTN3)
        self.assertEqual(rung, ("nextn", 0))  # resolved later by the worker

    def test_auto_without_derivable_switch_point_fails(self):
        with self.assertRaises(ValueError):
            _resolve(None)  # draft_model_path has no config.json

    def test_explicit_table_parses(self):
        table, source = _resolve("0:dflash:16,4096:nextn:3,65536:nextn:2")
        self.assertEqual(table, THREE_STAGE)
        self.assertIn("explicit", source)

    def test_explicit_rejects_bad_entries(self):
        for raw in (
            "",  # empty
            "0:dflash",  # not 3 parts
            "0:dflash:16:x",  # not 3 parts
            "x:dflash:16",  # non-int ctx
            "0:dflash:big",  # non-int value
            "0:eagle:3",  # unknown family
            "-1:dflash:16",  # negative ctx
        ):
            with self.assertRaises(ValueError, msg=raw):
                _resolve(raw)

    def test_explicit_rejects_unknown_rungs(self):
        # dflash block must match the resident rung's block size.
        with self.assertRaises(ValueError):
            _resolve("0:dflash:8,4096:nextn:3")
        # nextn k must be in the configured arm set.
        with self.assertRaises(ValueError):
            _resolve("0:dflash:16,4096:nextn:7")
        self.assertIn(
            "arm set",
            str(
                self._exc("0:dflash:16,4096:nextn:7")
            ),
        )

    def _exc(self, raw):
        try:
            _resolve(raw)
        except ValueError as e:
            return e
        self.fail(f"{raw!r} did not raise")

    def test_explicit_rejects_bad_ordering(self):
        # First stage must start at 0.
        with self.assertRaises(ValueError):
            _resolve("100:dflash:16,4096:nextn:3")
        # Strictly ascending.
        with self.assertRaises(ValueError):
            _resolve("0:dflash:16,4096:nextn:3,4096:nextn:2")
        with self.assertRaises(ValueError):
            _resolve("0:dflash:16,8192:nextn:3,4096:nextn:2")


class TestLookup(CustomTestCase):
    def test_lookup_index(self):
        self.assertEqual(policy_lookup_index(THREE_STAGE, 0), 0)
        self.assertEqual(policy_lookup_index(THREE_STAGE, 4095), 0)
        self.assertEqual(policy_lookup_index(THREE_STAGE, 4096), 1)
        self.assertEqual(policy_lookup_index(THREE_STAGE, 65535), 1)
        self.assertEqual(policy_lookup_index(THREE_STAGE, 65536), 2)
        self.assertEqual(policy_lookup_index(THREE_STAGE, 10**9), 2)


class TestPolicySelect(CustomTestCase):
    GATE = 8192
    NEAR = 0.8

    def _select(self, table, ctx_and_remaining, gate=GATE):
        return policy_select(
            table, ctx_and_remaining, gate, self.NEAR, ("nextn", 3)
        )

    def test_below_switch_point_selects_dflash(self):
        rung, max_ctx, fell_back = self._select(TWO_STAGE, [(2000, 512)])
        self.assertEqual(rung, DFLASH16)
        self.assertEqual(max_ctx, 2000)
        self.assertFalse(fell_back)

    def test_above_switch_point_selects_nextn(self):
        rung, _, fell_back = self._select(TWO_STAGE, [(4096, 512)])
        self.assertEqual(rung, NEXTN3)
        self.assertFalse(fell_back)

    def test_crossing_is_deterministic_at_stage_start(self):
        # The mid-stream transition: 4095 -> dflash, 4096 -> nextn.
        self.assertEqual(self._select(TWO_STAGE, [(4095, 8)])[0], DFLASH16)
        self.assertEqual(self._select(TWO_STAGE, [(4096, 8)])[0], NEXTN3)

    def test_bs2_max_over_requests(self):
        rung, max_ctx, _ = self._select(TWO_STAGE, [(2000, 64), (5000, 64)])
        self.assertEqual(rung, NEXTN3)
        self.assertEqual(max_ctx, 5000)

    def test_gate_fallback_to_next_stage_above(self):
        # An explicit table putting DFLASH above the gate: lookup says
        # dflash, the gate (safety filter) says no -> next stage above.
        table = [(0, DFLASH16), (65536, NEXTN2)]
        rung, _, fell_back = self._select(table, [(20000, 512)])
        self.assertEqual(rung, NEXTN2)
        self.assertTrue(fell_back)

    def test_gate_near_preemption_falls_back(self):
        # 7000 > 0.8 * 8192 and budget crosses the gate: the gate would
        # pre-empt DFLASH at decode start; policy honors it.
        table = [(0, DFLASH16), (8192, NEXTN3)]
        rung, _, fell_back = self._select(table, [(7000, 2000)])
        self.assertEqual(rung, NEXTN3)
        self.assertTrue(fell_back)
        # Budget cannot cross -> no fallback.
        rung, _, fell_back = self._select(table, [(7000, 1000)])
        self.assertEqual(rung, DFLASH16)
        self.assertFalse(fell_back)

    def test_gate_fallback_without_stage_above_uses_fallback_rung(self):
        table = [(0, DFLASH16)]  # dflash-only table
        rung, _, fell_back = self._select(table, [(50000, 64)])
        self.assertEqual(rung, NEXTN3)  # the passed fallback (primary k)
        self.assertTrue(fell_back)

    def test_gate_off_never_falls_back(self):
        table = [(0, DFLASH16)]
        rung, _, fell_back = self._select(table, [(50000, 64)], gate=None)
        self.assertEqual(rung, DFLASH16)
        self.assertFalse(fell_back)

    def test_default_table_switch_below_gate_no_fallback(self):
        # The default table's switch point (4096) sits BELOW the gate
        # (8192), so in the dflash region the gate never triggers on ctx
        # alone -- only near-preemption can.
        rung, _, fell_back = self._select(TWO_STAGE, [(4000, 64)])
        self.assertEqual(rung, DFLASH16)
        self.assertFalse(fell_back)

    def test_empty_batch(self):
        rung, max_ctx, fell_back = self._select(TWO_STAGE, [])
        self.assertEqual(rung, DFLASH16)  # stage 0
        self.assertEqual(max_ctx, 0)
        self.assertFalse(fell_back)


if __name__ == "__main__":
    unittest.main()
