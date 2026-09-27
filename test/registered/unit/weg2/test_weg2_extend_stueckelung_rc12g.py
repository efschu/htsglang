# SPDX-License-Identifier: Apache-2.0
"""rc12g (NF rc12f 26af2ccdca, D-TP0, 27.09. 04:27Z): the trim is spent, the chunk has to follow the card.

rc12f trimmed before every D extend under 3071 MiB free; the trim never brought
the card back above ~2730 (cache only -- the untagged rest, the draft and the
KV fill stay). Two 4-k chunks of one request then grew reserved by the
measured per-row rate, WEG2-VRAM-PEAK / WEG2-EXTEND-CACHE-TRIM verbatim:

  04:27:45 post 2385, 4096 rows, reserved +1992 -> card_free 393
  04:27:50 post 2325, 4029 rows, reserved +2116 -> card_free 207  (HALT < 300)

rc12g caps the chunk before it is formed: rows_cap = floor((post - 300) /
D_EXTEND_GROWTH_PER_ROW_MIB), down to the page, as this rank's vote in the
scheduler's existing packed MIN reduce (#794 corridor width) -- so every rank
cuts to the same width and no collective is added.
"""

import os
import unittest

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import extend_trim as ET  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

MIB = 1 << 20
PAGE = 64
RATE = 0.5625

#: (boot, time, rows, reserved growth) -- the extend windows of rc12b..f on
#: D-TP0 with >= 512 rows (tmp/r989/rate_windows.py over the five D-logs)
WINDOWS = (
    ("09270007", "00:18:02", 603, 776),     # rc12b: decode/draft inside the window
    ("09270007", "00:23:27", 3750, 842),
    ("09270103", "01:19:54", 4096, 768),
    ("09270212", "02:37:41", 4096, 1974),
    ("09270212", "02:35:49", 4059, 1784),
    ("09270311", "03:39:07", 4096, 2304),   # the maximum
    ("09270311", "03:39:12", 4072, 2272),
    ("09270311", "03:40:51", 4096, 2272),
    ("09270402", "04:21:25", 4096, 1992),
    ("09270402", "04:22:03", 3073, 1656),
    ("09270402", "04:27:45", 4096, 1992),
    ("09270402", "04:27:50", 4029, 2116),
)


class _Cuda:
    """mem_get_info / reserved / empty_cache of one card, in MiB."""

    def __init__(self, free, cache=0.0):
        self.free, self.cache, self.trims = float(free), float(cache), 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * MIB), int(32088 * MIB)

    def memory_reserved(self):
        return int((28000 + self.cache) * MIB)

    def synchronize(self):
        pass

    def empty_cache(self):
        self.trims += 1
        self.free += self.cache
        self.cache = 0.0


def _env(rates, trim="3071,1724,1725"):
    ET.reset_for_tests()
    ET._CACHE["rates"] = ET.parse_thresholds(rates)
    ET._CACHE["thresholds"] = ET.parse_thresholds(trim)


class TestRecord(CustomTestCase):
    def test_rate_is_the_measured_maximum_of_the_big_extends(self):
        rate = ET.extend_growth_per_row_mib([(r, g) for _b, _t, r, g in WINDOWS])
        self.assertEqual(rate, RATE)  # rc12e 03:39:07, 2304 / 4096

    def test_small_window_does_not_price_the_rate(self):
        # 603 rows, +776 MiB = 1.29/row: the window holds decode/draft rounds
        self.assertIsNone(ET.extend_growth_per_row_mib([(603, 776)]))
        self.assertEqual(ET.GROWTH_PER_ROW_MIN_ROWS, 2048)

    def test_profile_carries_the_record_nextflash_only(self):
        vals, boots = L.d_extend_growth_per_row_record("nextflash")
        self.assertEqual(vals, [RATE, None, None])
        self.assertIn("dkrnfh91bar1dauer09270402", boots)
        self.assertEqual(L.d_extend_growth_per_row_record("qwen27b"), (None, ""))

    def test_fixpoint_the_rate_does_not_move_with_the_cut(self):
        before = ET.extend_growth_per_row_mib([(4096, 2304)])
        cut = ET.rows_cap(2385, before, PAGE)
        after = ET.extend_growth_per_row_mib([(4096, 2304), (cut, cut * before)])
        self.assertEqual(before, after)
        # and three rounds of "measure, cut, measure" stay put
        rate = before
        for _ in range(3):
            rate = ET.extend_growth_per_row_mib([(ET.rows_cap(2325, rate, PAGE),
                                                  ET.rows_cap(2325, rate, PAGE) * rate), (4096, 2304)])
        self.assertEqual(rate, before)


class TestCap(CustomTestCase):
    def test_metal_0427_45_is_cut(self):
        # floor((2385 - 300) / 0.5625) = 3706 -> page 3648 < 4096
        self.assertEqual(ET.rows_cap(2385, RATE, PAGE), 3648)
        self.assertGreaterEqual(2385 - 3648 * RATE, 300)

    def test_metal_0427_50_is_cut(self):
        # floor((2325 - 300) / 0.5625) = 3600 -> page 3584 < 4029 (card_free 207 at the metal)
        self.assertEqual(ET.rows_cap(2325, RATE, PAGE), 3584)
        self.assertGreaterEqual(2325 - 3584 * RATE, 300)

    def test_metal_0422_03_is_not_cut(self):
        # floor((2569 - 300) / 0.5625) = 4033 -> 4032 >= 3073 rows: the request
        # extends whole (at the metal it left 913 MiB)
        cap = ET.rows_cap(2569, RATE, PAGE)
        self.assertEqual(cap, 4032)
        self.assertGreaterEqual(cap, 3073)

    def test_the_boundary_for_a_full_chunk(self):
        # a 4096-row chunk is cut exactly when post < 300 + 4096 * 0.5625 = 2604
        self.assertEqual(ET.rows_cap(2604, RATE, PAGE), 4096)
        self.assertLess(ET.rows_cap(2603, RATE, PAGE), 4096)
        # the trim threshold itself (3071) funds 4926 rows: never cut there
        self.assertGreater(ET.rows_cap(3071, RATE, PAGE), 4096)

    def test_never_below_one_page(self):
        self.assertEqual(ET.rows_cap(310, RATE, PAGE), PAGE)
        self.assertEqual(ET.rows_cap(100, RATE, PAGE), PAGE)


class TestVote(CustomTestCase):
    def tearDown(self):
        ET.reset_for_tests()

    def test_off_without_rate_is_no_vote(self):
        _env(None)
        self.assertIsNone(ET.width_vote(_Cuda(2000, 400), 0, 4096, PAGE, True))

    def test_no_pending_prefill_no_card_read(self):
        _env("0.5625,0,0")
        cuda = _Cuda(2000, 400)
        self.assertIsNone(ET.width_vote(cuda, 0, 4096, PAGE, False))
        self.assertEqual(cuda.trims, 0)

    def test_vote_reads_the_card_after_the_trim(self):
        _env("0.5625,0,0")
        cuda = _Cuda(1765, 620)  # 04:27:45: 1765 before, trim to 2385
        self.assertEqual(ET.width_vote(cuda, 0, 4096, PAGE, True), 3648)
        self.assertEqual(cuda.trims, 1)

    def test_trims_once_per_extend_not_per_iteration(self):
        _env("0.5625,0,0")
        cuda = _Cuda(1765, 620)
        for _ in range(5):  # a queue waiting for seats: five decode iterations
            ET.width_vote(cuda, 0, 4096, PAGE, True)
            cuda.cache += 100
        self.assertEqual(cuda.trims, 1)

        class _Mode:
            def is_extend(self):
                return True

            def is_target_verify(self):
                return False

        class _Batch:
            forward_mode = _Mode()

        class _Worker:
            tp_rank = 0

        import sys
        import types

        real = sys.modules.get("torch")
        fake = types.SimpleNamespace(cuda=cuda)
        sys.modules["torch"] = fake
        try:
            ET.before_extend(_Worker(), _Batch())  # an extend ran: re-armed
        finally:
            sys.modules["torch"] = real
        ET.width_vote(cuda, 0, 4096, PAGE, True)
        self.assertGreaterEqual(cuda.trims, 2)

    def test_wide_card_casts_no_vote(self):
        _env("0.5625,0,0")
        self.assertIsNone(ET.width_vote(_Cuda(3500), 0, 4096, PAGE, True))

    def test_3080_ranks_without_rate_do_not_vote(self):
        _env("0.5625,0,0")
        self.assertIsNone(ET.width_vote(_Cuda(900), 1, 4096, PAGE, True))
        self.assertIsNone(ET.width_vote(_Cuda(900), 2, 4096, PAGE, True))


class TestGroupMin(CustomTestCase):
    """The vote rides `_local_corridor_width_ceiling`, the payload term of the
    packed MIN reduce: every rank contributes an int every iteration (the
    payload width never depends on the vote) and the group takes the MIN."""

    def tearDown(self):
        ET.reset_for_tests()

    def _ceiling(self, rank, cuda):
        import torch

        from sglang.srt.managers.scheduler import Scheduler

        class _S:
            chunked_prefill_size = 4096
            page_size = PAGE
            chunked_req = object()
            waiting_queue = []

        s = _S()
        s.tp_rank = rank
        real = torch.cuda
        torch.cuda = cuda
        try:
            return Scheduler._local_corridor_width_ceiling(s)
        finally:
            torch.cuda = real

    def test_tp1_tp2_follow_tp0(self):
        _env("0.5625,0,0")
        votes = [self._ceiling(0, _Cuda(1765, 620)),
                 self._ceiling(1, _Cuda(900)),
                 self._ceiling(2, _Cuda(1400))]
        for v in votes:
            self.assertIsInstance(v, int)
        self.assertEqual(votes[0], 3648)
        self.assertEqual(votes[1:], [4096, 4096])
        self.assertEqual(min(votes), 3648)  # the reduce: every rank cuts to 3648

    def test_payload_is_an_int_on_every_rank_with_or_without_the_feature(self):
        _env(None)
        for r in range(3):
            self.assertEqual(self._ceiling(r, _Cuda(500)), 4096)


class TestLauncher(CustomTestCase):
    def test_env_value_from_the_record(self):
        self.assertEqual(ET.launcher_rates([RATE, None, None]), "0.5625,0,0")
        self.assertEqual(ET.launcher_rates([None, None, None]), "")
        self.assertEqual(ET.launcher_rates([]), "")

    def test_parse_zero_is_no_rate(self):
        _env("0.5625,0,0")
        self.assertEqual(ET.threshold_for(1, ET.rates()), 0.0)
        ET.reset_for_tests()


if __name__ == "__main__":
    unittest.main()
