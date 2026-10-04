# SPDX-License-Identifier: Apache-2.0
"""Q-710 EXTEND-CAP-FLOOR: the D extend chunk vote never crawls (INT8 y8vb, 03.10. 19:34-19:51Z).

Boot ``boot_weg2_dkr27browauthoritybar1fs10031930_76f5bbde19_1003_193056`` (27B INT8 flip line,
Q-694 EXTEND-CAP-FLIP armed the rc12g width vote there): the D-TP1 3080 held 72-290 MiB free in
the D phase, ``rows_cap = floor((post - 300) / 0.3091)`` is page-clamped to ONE row, the MIN
reduce cut the group to 1 (``#794 GROUP-NARROWED this prefill chunk from 4096 to 1`` x11), and
3905 one-token target extends of ~152 ms each ran in 17 minutes (76 requests; a 5-token
hand-off tail took 5 forwards, the 1546-token SHORT 14-78 took 860, five running streams had no
token for 133 s). Eager extends are latency-bound (129 tp.all_reduce per forward), one row costs
what a hundred cost.

Fix, flip line only (``SGLANG_WEG2_EXTEND_CAP_FLOOR``, written by the launcher where it writes
the Q-694 rate): the vote keeps a floor chunk (configured / 16, bounded by what the free card
funds at the priced rate), and the X gate refuses a fresh D-direct request longer than a group
width that is under the floor, so the front routes it through P.
"""
import os
import types
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import extend_trim as ET  # noqa: E402

MIB = 1 << 20
RATE = 0.3091            # D_EXTEND_CAP_PER_ROW_MIB (qwen27b)
CONFIGURED = 4096        # --chunked-prefill-size of the y8vb boot
FLOOR_ENV = "SGLANG_WEG2_EXTEND_CAP_FLOOR"
RATE_ENV = "SGLANG_WEG2_EXTEND_GROWTH_PER_ROW_MIB"


class _Cuda:
    def __init__(self, free):
        self.free = float(free)

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * MIB), int(20055 * MIB)

    def memory_reserved(self):
        return int(18000 * MIB)

    def synchronize(self):
        pass

    def empty_cache(self):
        pass


def _vote(free, *, armed, page=1, rank=1):
    env = {RATE_ENV: str(RATE)}
    if armed:
        env[FLOOR_ENV] = "1"
    with mock.patch.dict(os.environ, env, clear=False):
        if not armed:
            os.environ.pop(FLOOR_ENV, None)
        ET.reset_for_tests()
        try:
            return ET.width_vote(_Cuda(free), rank, CONFIGURED, page, True)
        finally:
            ET.reset_for_tests()


class VoteFloor(CustomTestCase):
    def test_y8vb_tp1_post_108_voted_one_row_and_now_a_floor_chunk(self):
        """RED on 1de15e82ca: D:57198 cap=1 post=108 -> GROUP-NARROWED 4096 to 1."""
        self.assertEqual(_vote(108, armed=False), 1)           # the y8vb form, flag off
        w = _vote(108, armed=True)
        self.assertEqual(w, CONFIGURED // ET.CAP_FLOOR_DIV)    # 256
        self.assertLessEqual(w * RATE, 108)                    # funded by the card itself

    def test_floor_is_only_what_the_free_card_funds(self):
        """post 72 (the boot's minimum): 232 rows, not the 256 the card cannot fund."""
        w = _vote(72, armed=True)
        self.assertEqual(w, int(72 // RATE))
        self.assertLessEqual(w * RATE, 72)
        self.assertGreater(w, 1)

    def test_line_respecting_cap_above_the_floor_is_untouched(self):
        for free in (460, 962, 1182):                          # TP2/TP1 posts of the same boot
            self.assertEqual(_vote(free, armed=True), _vote(free, armed=False), free)

    def test_no_card_no_floor_raise_beyond_the_page(self):
        self.assertEqual(_vote(0.1, armed=True), 1)            # nothing funded: the old cap
        self.assertEqual(_vote(0.1, armed=True, page=64), 64)

    def test_floor_is_page_aligned_and_derived_from_the_configured_width(self):
        self.assertEqual(ET.floor_rows(4096, 64), 256)
        self.assertEqual(ET.floor_rows(8192, 1), 512)
        self.assertEqual(ET.floor_rows(100, 64), 64)           # never under one page
        self.assertEqual(ET.floored_width(1, 108.0, RATE, 4096, 1), 256)

    def test_flag_off_is_byte_identical_for_every_post(self):
        for free in (0, 5, 72, 108, 290, 300, 301, 330, 460, 1182, 5000):
            self.assertEqual(_vote(free, armed=False), ET.rows_cap(free, RATE, 1)
                             if ET.rows_cap(free, RATE, 1) < CONFIGURED else None, free)


class GateRefusal(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()
        self._p = mock.patch.dict(os.environ, {FLOOR_ENV: "1"})
        self._p.start()

    def tearDown(self):
        self._p.stop()
        ET.reset_for_tests()

    def test_fresh_long_request_under_a_starved_width_is_refused(self):
        """14-78: 1546 uncached at width 1 crawled 860 forwards; now it goes through P."""
        self.assertTrue(ET.refuses_over_width(1, 1546, CONFIGURED, 1, False))
        self.assertTrue(ET.refuses_over_width(232, 1546, CONFIGURED, 1, False))

    def test_handoff_tail_and_short_fitting_one_chunk_are_admitted(self):
        self.assertFalse(ET.refuses_over_width(232, 5, CONFIGURED, 1, False))     # d_uncached=5
        self.assertFalse(ET.refuses_over_width(232, 161, CONFIGURED, 1, False))   # d_uncached=161
        self.assertFalse(ET.refuses_over_width(232, 232, CONFIGURED, 1, False))

    def test_chunk_continuation_is_never_refused(self):
        self.assertFalse(ET.refuses_over_width(1, 1545, CONFIGURED, 1, True))

    def test_a_width_at_or_above_the_floor_never_refuses(self):
        self.assertFalse(ET.refuses_over_width(256, 9000, CONFIGURED, 1, False))
        self.assertFalse(ET.refuses_over_width(4096, 9000, CONFIGURED, 1, False))

    def test_no_group_width_no_verdict(self):
        self.assertFalse(ET.refuses_over_width(None, 9000, CONFIGURED, 1, False))
        self.assertFalse(ET.refuses_over_width(0, 9000, CONFIGURED, 1, False))
        self.assertFalse(ET.refuses_over_width(1, 9000, 0, 1, False))

    def test_flag_off_never_refuses(self):
        os.environ.pop(FLOOR_ENV, None)
        ET.reset_for_tests()
        self.assertFalse(ET.refuses_over_width(1, 1546, CONFIGURED, 1, False))


class SchedulerVerdict(CustomTestCase):
    """``Scheduler._weg2_cap_floor_verdict`` on a stand-in: the marks, the W31 string, the line."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.managers.scheduler import Scheduler
        except Exception as exc:  # pragma: no cover -- no scheduler import on this box
            raise unittest.SkipTest(f"scheduler not importable: {exc}")
        cls.fn = staticmethod(Scheduler._weg2_cap_floor_verdict)

    def setUp(self):
        ET.reset_for_tests()
        self._p = mock.patch.dict(os.environ, {FLOOR_ENV: "1"})
        self._p.start()

    def tearDown(self):
        self._p.stop()
        ET.reset_for_tests()

    def _sched(self, width):
        return types.SimpleNamespace(uniform_corridor_width=lambda: width, page_size=1,
                                     chunked_prefill_size=CONFIGURED)

    def test_refused_then_a_tail_admitted_and_marked(self):
        s = self._sched(1)
        req = types.SimpleNamespace(rid="weg2-14-78")
        with self.assertLogs("sglang.srt.managers.scheduler", level="WARNING") as cm:
            self.assertEqual(self.fn(s, req, 1546, "admit"), "W31")
        self.assertTrue(any("EXTEND-CAP-FLOOR route=P rid=weg2-14-78 uncached=1546" in m for m in cm.output))
        tail = types.SimpleNamespace(rid="weg2-0-1")
        self.assertEqual(self.fn(s, tail, 1, "admit"), "admit")
        self.assertIn("weg2-0-1", s._weg2_ecf_admitted)
        # a continuation of an admitted rid is not priced again
        self.assertEqual(self.fn(s, tail, 5000, "admit"), "admit")

    def test_healthy_width_admits_and_marks(self):
        s = self._sched(4096)
        req = types.SimpleNamespace(rid="weg2-2-16")
        self.assertEqual(self.fn(s, req, 2760, "admit"), "admit")
        self.assertIn("weg2-2-16", s._weg2_ecf_admitted)

    def test_gate_is_skipped_when_the_flag_is_off(self):
        """The call site only enters the verdict under cap_floor_armed()."""
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._weg2_x_refuses)
        self.assertIn("_weg2_extend_trim.cap_floor_armed()", src)
        self.assertIn("self._weg2_cap_floor_verdict(", src)


class LauncherArming(CustomTestCase):
    """The flag is written where, and only where, the flip arm wrote the Q-694 rate."""

    def _solve(self, ns):
        from sglang.srt.weg2 import launcher as L

        lines = []
        cards = [L.Card(1, "u1", "RTX 5090", 32607), L.Card(0, "u0", "RTX 3080", 20480),
                 L.Card(2, "u2", "RTX 3080", 20480)]
        L.log_d_rank_vram_solve(ns, cards, [27792, 17384, 17168], lines.append, "D")
        return lines

    def _ns(self, **kw):
        base = dict(profile="qwen27b", model="/nonexistent/Qwen3.8-27B-INT8-gdncov", extra_d="",
                    env_d="SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0", d_foreign_context_mib="",
                    d_nontorch_mib="", d_reserve_mib="", d_residency_reference_logs="",
                    d_card_reference_logs="", wake_credit_reference_logs="", dual_layout=False,
                    dual_share=False)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_flip_arm_writes_the_floor_flag(self):
        """RED on 1de15e82ca: the flip arm wrote the rate and the measurement, no floor."""
        ns = self._ns()
        lines = self._solve(ns)
        self.assertIn(f"{FLOOR_ENV}=1", ns.env_d)
        self.assertIn(f"{RATE_ENV}=", ns.env_d)
        self.assertTrue(any("EXTEND-CAP-FLOOR floor=on" in ln for ln in lines), lines)

    def test_second_pass_is_idempotent_and_the_operator_wins(self):
        ns = self._ns()
        self._solve(ns)
        first = ns.env_d
        self._solve(ns)
        self.assertEqual(ns.env_d, first)
        ns = self._ns(env_d="SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0;" + FLOOR_ENV + "=0")
        lines = self._solve(ns)
        self.assertIn(f"{FLOOR_ENV}=0", ns.env_d)
        self.assertTrue(any("EXTEND-CAP-FLOOR floor=off" in ln and "Vorrang" in ln for ln in lines), lines)

    def test_dual_layout_never_gets_the_flag(self):
        for kw in ({"dual_layout": True}, {"dual_share": True}):
            ns = self._ns(**kw)
            self._solve(ns)
            self.assertNotIn(FLOOR_ENV, ns.env_d, kw)

    def test_p0_cap_arm_never_gets_the_flag(self):
        ns = self._ns(env_d="SGLANG_WEG2_TORCH_CACHE_CAP=1")
        self._solve(ns)
        self.assertNotIn(FLOOR_ENV, ns.env_d)


if __name__ == "__main__":
    unittest.main()
