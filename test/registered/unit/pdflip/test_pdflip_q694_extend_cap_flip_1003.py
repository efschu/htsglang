# SPDX-License-Identifier: Apache-2.0
"""Q-694: the 27B INT8 FLIP line's D extend chunk follows the card (rc12g vote), not only under P0.

METAL (INT8 y8va, boot dkr27browauthoritybar1fs10031830, b2744f5043, D.log _1003_183056 +
debug_hold rc12gbf10031828_rank0_pid11529): D-TP0 (5090) 18:39:36Z, pdflip-30-101 routed SHORT by the
front (uncached 11328 < X 12288, presence 70169), X-GATE admit, ``#969 EXTENT ... 70169, 74265 ...
4096``, ``PDFLIP-EXTEND-CACHE-TRIM rank=0 released=264 card_free_before=265 card_free_after=529
threshold=1200``, then ``OutOfMemoryError: Tried to allocate 20.00 MiB ... 3.44 MiB is free`` in
barlink_bar1._all_reduce_one_round (linear.py:2411). The D group env carried
``FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,0,0`` but no ``FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB``: the
launcher wrote the chunk-cap rate only under the P0 torch cache cap (off on the qwen27b row) or from
the #145 ledger (Next Flash only), so ``extend_trim.width_vote`` never voted and the trim's 529 MiB
met a 4096-row deep extend (TP0 transient 518-1001 MiB at 4096 rows, 40 boots).
"""
import os
import types
import unittest
from unittest import mock

try:
    from flliper.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from flliper.srt.pdflip import extend_trim as ET
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20
RATE = 0.3091                 # D_EXTEND_CAP_PER_ROW_MIB (qwen27b)
RATE_ENV = "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
TRIM_ENV_D = "FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,0,0"   # the y8va D group env (front.log:247)
FREE_BEFORE = 265             # PDFLIP-EXTEND-CACHE-TRIM card_free_before
RELEASED = 264                # ... released
FREE_AFTER = 529              # ... card_free_after
DEATH_ROWS = 4096             # #969 EXTENT 70169 -> 74265
START_ALLOCATED = 27627       # PDFLIP-VRAM-PEAK rank=0 phase=round 18:39:36 allocated_mib (before the extend)
DEATH_ALLOCATED = 28518       # the OOM: "27.85 GiB is allocated by PyTorch" (+ 20 MiB requested)
MEASURED_RATE_MAX = 0.2690    # max transient/rows over 642 D extend windows of >= 2048 rows (40 boots)


def _cards():
    from flliper.srt.pdflip import launcher as L

    return [L.Card(1, "u1", "RTX 5090", 32607), L.Card(0, "u0", "RTX 3080", 20480),
            L.Card(2, "u2", "RTX 3080", 20480)]


def _ns(profile="qwen27b", env_d=TRIM_ENV_D, **kw):
    base = dict(profile=profile, model="/nonexistent/Qwen3.8-27B-INT8-gdncov", extra_d="", env_d=env_d,
                d_foreign_context_mib="", d_nontorch_mib="", d_reserve_mib="",
                d_residency_reference_logs="", d_card_reference_logs="",
                wake_credit_reference_logs="", dual_layout=False, dual_share=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _solve(ns):
    from flliper.srt.pdflip import launcher as L

    lines = []
    L.log_d_rank_vram_solve(ns, _cards(), [27792, 17384, 17168], lines.append, "D")
    return lines


class FlipLauncherWritesTheRate(CustomTestCase):
    def test_flip_solve_arms_the_chunk_cap(self):
        """RED on b2744f5043: env_d stayed 'FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,0,0'."""
        ns = _ns()
        lines = _solve(ns)
        self.assertIn(f"{RATE_ENV}={RATE},{RATE},{RATE}", ns.env_d)
        self.assertTrue(ns.env_d.startswith(TRIM_ENV_D + ";"))
        self.assertTrue(any("Q694 EXTEND-CAP-FLIP" in ln and "card_free_post" in ln for ln in lines), lines)

    def test_second_pass_is_idempotent(self):
        ns = _ns()
        _solve(ns)
        first = ns.env_d
        lines = _solve(ns)
        self.assertEqual(ns.env_d, first)
        self.assertFalse(any("Vorrang" in ln for ln in lines if "Q694" in ln))

    def test_operator_value_wins(self):
        ns = _ns(env_d=TRIM_ENV_D + f";{RATE_ENV}=0.5,0.5,0.5")
        lines = _solve(ns)
        self.assertIn(f"{RATE_ENV}=0.5,0.5,0.5", ns.env_d)
        self.assertNotIn(f"{RATE_ENV}={RATE}", ns.env_d)
        self.assertTrue(any("Q694" in ln and "Vorrang" in ln for ln in lines))


class DualUnchanged(CustomTestCase):
    """User order 03.10.: dual fixes never touch flip, and this flip fix never touches dual."""

    def test_dual_layout_writes_nothing(self):
        for kw in ({"dual_layout": True}, {"dual_share": True}, {"dual_layout": True, "dual_share": True}):
            ns = _ns(**kw)
            lines = _solve(ns)
            self.assertEqual(ns.env_d, TRIM_ENV_D, kw)
            self.assertFalse(any("Q694" in ln for ln in lines), kw)
            self.assertNotIn(RATE_ENV, ns.env_d)

    def test_dual_arm_function_is_inert(self):
        from flliper.srt.pdflip import launcher as L

        ns = _ns(dual_layout=True)
        lines = []
        self.assertIsNone(L._d_extend_flip_rate_env(ns, lines.append, "D"))
        self.assertEqual((ns.env_d, lines), (TRIM_ENV_D, []))


class OtherFormsUnchanged(CustomTestCase):
    def test_nextflash_has_no_record_and_writes_nothing(self):
        from flliper.srt.pdflip import launcher as L

        ns = _ns(profile="nextflash", env_d="")
        lines = []
        self.assertIsNone(L._d_extend_flip_rate_env(ns, lines.append, "D"))
        self.assertEqual((ns.env_d, lines), ("", []))

    def test_p0_branch_keeps_its_own_line(self):
        from flliper.srt.pdflip import launcher as L

        ns = _ns(env_d="FLLIPER_PDFLIP_TORCH_CACHE_CAP=1")
        lines = []
        self.assertEqual(L._d_extend_cap_rate_env(ns, lines.append, "D"), f"{RATE},{RATE},{RATE}")
        self.assertTrue(any("EXTEND-CAP rows_cap aus min(card_free, torch_cap - reserved)" in ln
                            for ln in lines))
        self.assertFalse(any("Q694" in ln for ln in lines))


class _Cuda:
    """D-TP0 at 18:39:36Z: 265 MiB free, 264 MiB of releasable cache."""

    def __init__(self):
        self.free = FREE_BEFORE
        self.reserved = 28000
        self.emptied = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return self.free * MIB, 32088 * MIB

    def memory_reserved(self, *a):
        return self.reserved * MIB

    def synchronize(self):
        pass

    def empty_cache(self):
        self.emptied += 1
        self.reserved -= RELEASED
        self.free += RELEASED


def _rank_env():
    return {"FLLIPER_PDFLIP_EXTEND_TRIM_MIB": "1200,0,0", RATE_ENV: f"{RATE},{RATE},{RATE}"}


class RankVoteAtTheDeath(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def _vote(self, env):
        cuda = _Cuda()
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("FLLIPER_PDFLIP_TORCH_CACHE_CAP", None)
            return ET.width_vote(cuda, 0, 4096, 1, True), cuda

    def test_fixed_env_cuts_to_what_529_mib_funds(self):
        vote, cuda = self._vote(_rank_env())
        self.assertEqual(cuda.emptied, 1)              # the trim runs in the vote, before the chunk forms
        self.assertEqual(cuda.free, FREE_AFTER)
        self.assertEqual(vote, 740)                    # floor((529 - 300) / 0.3091)
        self.assertLessEqual(vote * RATE + ET.CHUNKING_FLOOR_MIB, FREE_AFTER)
        # the metal's own 4096-row chunk had taken 911 MiB when the 20 MiB all_reduce output was
        # refused -- more than the 529 the trim left -- and its per-row cost is under the record,
        # so the cut chunk at the same per-row cost keeps the 300 MiB line
        death_transient = DEATH_ALLOCATED + 20 - START_ALLOCATED
        self.assertGreater(death_transient, FREE_AFTER)
        self.assertLessEqual(death_transient / DEATH_ROWS, RATE)
        self.assertLessEqual(vote * death_transient / DEATH_ROWS + ET.CHUNKING_FLOOR_MIB, FREE_AFTER)

    def test_record_covers_every_measured_extend(self):
        """The profile record the flip arm reads must stay above every measured D extend's
        allocated transient per row (a lowered record re-opens the y8va death)."""
        from flliper.srt.pdflip import launcher as L

        rec = [float(v) for v in L._pconst("D_EXTEND_CAP_PER_ROW_MIB", "qwen27b")]
        self.assertEqual(rec, [RATE, RATE, RATE])
        self.assertGreaterEqual(min(rec), MEASURED_RATE_MAX)


if __name__ == "__main__":
    unittest.main()
