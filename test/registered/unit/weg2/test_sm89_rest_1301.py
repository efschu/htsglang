# SPDX-License-Identifier: Apache-2.0
"""Auftrag 1301 (sm89-Rest, NF-Sitz): the two sm_89 code gaps that
SM89-DURCHSPIEL-1002 + HW-P0 1003 left open, pinned CPU-hermetically.

(1) Determinism certificate (HW-P0 report "offen: determinism_certificate.
    has_sm8x_rank liest noch den festen Bereich"): an sm89 rank WITHOUT sm_89
    SASS in the sgl_kernel wheel runs its fp8 GEMM through gptq_marlin_gemm
    (FP8-SM89-FALLBACK) -- the kernel #190 measured as run-to-run
    nondeterministic -- and fp8_utils.deterministic_fp8_marlin_disabled already
    switches it off for the flag. The certificate must say the same: the
    ``fp8_marlin_sm8x`` exclusion and the ``SGLANG_DETERMINISTIC_FP8_GEMM``
    forced env. sm86 / sm120 decide exactly as before and never touch the wheel.

(2) QSA rows kernel (SM89-DURCHSPIEL row "QSA SGLANG_FORCE_QSA_ROWS_CONFIG"):
    an sm_89 card had no entry in ``_ARCH_ROWS_DEFAULTS`` and so took the
    device-name L20 table, whose (16, 1, 2) above 512 rows spills on sm86 and
    sm120 (H58/H101). Offline compile for sm89 gives the identical spill
    (REG 255 / STACK 472 at the measured shape), so sm89 takes the H101 table.
    sm86 and sm120 stay exactly as they were.
"""

from __future__ import annotations

import itertools
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt import determinism_certificate as dc
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.srt.utils import wheel_sass
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _facts(archs, **kw):
    kw.setdefault("has_fp8_weights", True)
    return dc.DeterminismFacts(rank_archs=tuple(archs), **kw)


def _old_has_sm8x_rank(archs):
    """The pre-1301 definition, verbatim."""
    return any(80 <= a < 89 for a in archs)


class TestCertificateSm89Route(unittest.TestCase):
    def test_sm89_without_wheel_sass_is_on_the_marlin_route(self):
        for wheel in (False, None):
            f = _facts((89,), wheel_sm89_sass=wheel)
            self.assertTrue(f.has_sm8x_rank, wheel)
            c = dc.resolve_certificate(f)
            self.assertTrue(c.excluded("fp8_marlin_sm8x"), wheel)
            self.assertEqual(c.forced_env.get("SGLANG_DETERMINISTIC_FP8_GEMM"), "1")
            self.assertTrue(any("sm89 rank" in n for n in c.notes))

    def test_unknown_wheel_is_named_unknown_known_absent_is_named_absent(self):
        n_none = dc.resolve_certificate(_facts((89,), wheel_sm89_sass=None)).notes
        n_false = dc.resolve_certificate(_facts((89,), wheel_sm89_sass=False)).notes
        self.assertTrue(any("was not shown to carry" in n for n in n_none))
        self.assertTrue(any("carries no" in n for n in n_false))

    def test_sm89_with_wheel_sass_keeps_cutlass_and_no_exclusion(self):
        f = _facts((89,), wheel_sm89_sass=True)
        self.assertFalse(f.has_sm8x_rank)
        c = dc.resolve_certificate(f)
        self.assertFalse(c.excluded("fp8_marlin_sm8x"))
        self.assertNotIn("SGLANG_DETERMINISTIC_FP8_GEMM", c.forced_env)

    def test_mixed_sm120_sm89_group_follows_its_sm89_rank(self):
        c = dc.resolve_certificate(_facts((120, 89), wheel_sm89_sass=False))
        self.assertTrue(c.excluded("fp8_marlin_sm8x"))
        c = dc.resolve_certificate(_facts((120, 89), wheel_sm89_sass=True))
        self.assertFalse(c.excluded("fp8_marlin_sm8x"))

    def test_no_fp8_no_exclusion_whatever_the_arch(self):
        c = dc.resolve_certificate(_facts((89,), has_fp8_weights=False))
        self.assertFalse(c.excluded("fp8_marlin_sm8x"))
        self.assertNotIn("SGLANG_DETERMINISTIC_FP8_GEMM", c.forced_env)


class TestCertificateUnchangedWithoutSm89(unittest.TestCase):
    """Test 'unveraendert': every group without an sm89 rank decides as before,
    for every wheel fact, and never probes the wheel."""

    ARCHS = (70, 75, 80, 86, 88, 90, 100, 120)

    def test_has_sm8x_rank_identical_to_the_old_formula(self):
        for n in (0, 1, 2, 3):
            for combo in itertools.combinations_with_replacement(self.ARCHS, n):
                for wheel in (None, False, True):
                    f = _facts(combo, wheel_sm89_sass=wheel)
                    self.assertEqual(f.has_sm8x_rank, _old_has_sm8x_rank(combo), (combo, wheel))
                    self.assertFalse(f.has_sm89_marlin_rank)

    def test_reference_rig_certificate_is_identical_with_and_without_the_wheel_fact(self):
        rig = (120, 86, 86)
        base = dc.resolve_certificate(_facts(rig))
        for wheel in (None, False, True):
            c = dc.resolve_certificate(_facts(rig, wheel_sm89_sass=wheel))
            self.assertEqual(c.render(), base.render())
            self.assertEqual(dict(c.forced_env), dict(base.forced_env))
            self.assertEqual(c.notes, base.notes)
        self.assertTrue(base.excluded("fp8_marlin_sm8x"))
        self.assertEqual(base.forced_env.get("SGLANG_DETERMINISTIC_FP8_GEMM"), "1")

    def test_sm120_only_has_no_fp8_exclusion(self):
        c = dc.resolve_certificate(_facts((120, 120)))
        self.assertFalse(c.excluded("fp8_marlin_sm8x"))
        self.assertNotIn("SGLANG_DETERMINISTIC_FP8_GEMM", c.forced_env)

    def test_the_wheel_is_never_probed_without_an_sm89_rank(self):
        args = types.SimpleNamespace(quantization="fp8", kv_cache_dtype="auto")
        with mock.patch.object(wheel_sass, "wheel_carries_sass",
                               side_effect=AssertionError("wheel probed")):
            for rig in ((120, 86, 86), (86,), (120,), ()):
                f = dc.facts_from_server_args(args, rig)
                self.assertIsNone(f.wheel_sm89_sass)

    def test_the_wheel_is_probed_once_for_an_sm89_rank(self):
        args = types.SimpleNamespace(quantization="fp8", kv_cache_dtype="auto")
        for carries in (True, False, None):
            with mock.patch.object(wheel_sass, "wheel_carries_sass",
                                   return_value=carries) as probe:
                f = dc.facts_from_server_args(args, (89, 89))
            probe.assert_called_once_with((8, 9))
            self.assertIs(f.wheel_sm89_sass, carries)

    def test_a_failing_probe_is_unknown_not_a_crash(self):
        args = types.SimpleNamespace(quantization="fp8", kv_cache_dtype="auto")
        with mock.patch.object(wheel_sass, "wheel_carries_sass", side_effect=OSError("x")):
            f = dc.facts_from_server_args(args, (89,))
        self.assertIsNone(f.wheel_sm89_sass)
        self.assertTrue(f.has_sm8x_rank)  # unknown = conservative = Marlin route


class TestCertificateAgreesWithTheFp8Gate(unittest.TestCase):
    """The certificate and fp8_utils.deterministic_fp8_marlin_disabled use one
    rule: the same (sm, wheel fact) -> the same Marlin-route decision."""

    def tearDown(self):
        from sglang.srt.layers.quantization import fp8_utils as fu

        fu.deterministic_fp8_marlin_disabled.cache_clear()

    def test_same_decision_for_every_arch_and_wheel_fact(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import fp8_utils as fu

        old = envs.SGLANG_DETERMINISTIC_FP8_GEMM.get()
        envs.SGLANG_DETERMINISTIC_FP8_GEMM.set(True)
        try:
            for sm, wheel in itertools.product((86, 89, 90, 120), (True, False, None)):
                fu.deterministic_fp8_marlin_disabled.cache_clear()
                with mock.patch.object(fu, "is_cuda", lambda: True), \
                        mock.patch.object(fu, "get_device_capability",
                                          lambda *a, **k: (sm // 10, sm % 10)), \
                        mock.patch.object(fu, "wheel_carries_sass", lambda cc: wheel):
                    gate = fu.deterministic_fp8_marlin_disabled()
                cert = _facts((sm,), wheel_sm89_sass=wheel).has_sm8x_rank
                self.assertEqual(cert, gate, (sm, wheel))
        finally:
            envs.SGLANG_DETERMINISTIC_FP8_GEMM.set(old)


# -- (2) QSA rows default for sm89 -------------------------------------------------

L20 = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


class TestQsaRowsSm89Default(unittest.TestCase):
    def setUp(self):
        sa._ROWS_CONFIG_CACHE.clear()

    def tearDown(self):
        sa._ROWS_CONFIG_CACHE.clear()

    def _on(self, cc, name):
        return mock.patch.multiple(
            sa.torch.cuda,
            get_device_capability=lambda *a: cc,
            get_device_name=lambda *a: name,
        )

    def test_sm89_takes_the_h101_table_only_the_top_entry_differs_from_l20(self):
        t = sa.rows_table_for_arch(89)
        self.assertEqual(t, sa._SM120_ROWS_CONFIGS)
        self.assertEqual(t[:-1], L20[:-1])
        self.assertEqual(t[-1], (float("inf"), (64, 8, 2)))

    def test_sm86_and_sm120_tables_are_exactly_what_they_were(self):
        self.assertIsNone(sa.rows_table_for_arch(86))
        self.assertEqual(sa.rows_table_for_arch(120), sa._SM120_ROWS_CONFIGS)
        self.assertEqual(set(sa._ARCH_ROWS_DEFAULTS), {89, 120})
        for arch in (70, 75, 80, 90, 100, 121):
            self.assertIsNone(sa.rows_table_for_arch(arch), arch)

    def test_launch_config_per_card(self):
        with self._on((8, 9), "NVIDIA GeForce RTX 4090"):
            self.assertEqual(sa._get_rows_config(600), (64, 8, 2))
            self.assertEqual(sa._get_rows_config(512), (32, 4, 2))
            self.assertEqual(sa._get_rows_config(1), (32, 8, 2))
        with self._on((8, 6), "NVIDIA GeForce RTX 3080"):  # unchanged: the L20 row
            self.assertEqual(sa._get_rows_config(600), (16, 1, 2))
            self.assertEqual(sa._get_rows_config(512), (32, 4, 2))
        with self._on((12, 0), "NVIDIA GeForce RTX 5090"):
            self.assertEqual(sa._get_rows_config(600), (64, 8, 2))

    def test_an_env_table_for_sm89_still_wins_a_foreign_arch_group_is_ignored(self):
        from sglang.srt.environ import envs

        with self._on((8, 9), "NVIDIA GeForce RTX 4090"):
            with mock.patch.object(envs.SGLANG_FORCE_QSA_ROWS_CONFIG, "get",
                                   return_value="sm89:inf=32/8/2"):
                self.assertEqual(sa._get_rows_config(600), (32, 8, 2))
            sa._ROWS_CONFIG_CACHE.clear()
            # the release D form names only sm120: -> sm89 keeps its default
            with mock.patch.object(envs.SGLANG_FORCE_QSA_ROWS_CONFIG, "get",
                                   return_value="sm120:32=32/8/2,inf=64/8/2"):
                self.assertEqual(sa._get_rows_config(600), (64, 8, 2))
            sa._ROWS_CONFIG_CACHE.clear()
            # the P form (generic group) applies to sm89 as to every arch
            with mock.patch.object(envs.SGLANG_FORCE_QSA_ROWS_CONFIG, "get",
                                   return_value="inf=64/8/2"):
                self.assertEqual(sa._get_rows_config(100000), (64, 8, 2))

    def test_prewarm_forms_follow_the_sm89_table(self):
        with self._on((8, 9), "NVIDIA GeForce RTX 4090"):
            forms = [cfg for _lo, cfg in sa.rows_launch_forms(89)]
        self.assertIn((64, 8, 2), forms)
        self.assertNotIn((16, 1, 2), forms)
        with self._on((8, 6), "NVIDIA GeForce RTX 3080"):
            self.assertIn((16, 1, 2), [cfg for _lo, cfg in sa.rows_launch_forms(86)])


if __name__ == "__main__":
    unittest.main()
