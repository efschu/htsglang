# SPDX-License-Identifier: Apache-2.0
"""rc12e (NF Dauerlauf rc12d, D TP0, 27.09. 02:33-02:41Z): the extend grew reserved by more than it used.

rc12d's PDFLIP-VRAM-PEAK lines of D-TP0 (5090), verbatim numbers:
* 02:33:10 phase=round transient 621 MiB, peak_reserved 28740 -> 29914
  (+1174), card_free 619 -- before it 983 MiB were cached (reserved 28740 -
  allocated 27757) that the extend could not reuse;
* 02:40:00 phase=round transient 980, peak_reserved 30456, card_free 729,
  alloc_retries=1; 02:41:14 phase=chunk transient 911, card_free 589,
  alloc_retries=1 -- two awake phases in a row, the rc12c death pattern.

The fix (pdflip/extend_trim.py): before a D extend, a card with less free than
floor + booked activation empties the allocator cache (PDFLIP-EXTEND-CACHE-TRIM),
so the extend's segments come from free card bytes, not on top of a cache it
cannot use. The fake allocator below reproduces the shape: a stale cache of
another stream plus an extend whose blocks do not fit its own freed blocks.
"""

import importlib.util
import os
import types
import unittest
from pathlib import Path

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

MIB = 1 << 20
TOTAL = 32089       # torch's view of the 5090 ("total capacity of 31.34 GiB")
FLOOR = 767         # MEASURED-D corridor floor TP0 (rc12c/rc12d front.log)
ACT = 1024          # booked activation of the D form (#145 posts)


class FakeCuda:
    """A caching allocator in MiB: segments are cudaMalloc'd blocks; a freed
    block stays cached and serves only a request it can hold on its own
    stream. Out of card bytes -> free every unused cached segment and retry
    (``num_alloc_retries``), then OOM -- the CUDACachingAllocator's order."""

    def __init__(self, *, nontorch, live, stale_cache):
        self.nontorch = nontorch
        self.segs = [{"size": live, "used": True, "stream": 0}]
        if stale_cache:
            self.segs.append({"size": stale_cache, "used": False, "stream": 7})
        self.retries = 0
        self.ooms = 0
        self.empty_calls = 0
        self.min_free = self.free_mib()

    # -- the torch.cuda surface extend_trim reads --------------------------
    def reserved_mib(self):
        return sum(s["size"] for s in self.segs)

    def free_mib(self):
        return TOTAL - self.nontorch - self.reserved_mib()

    def mem_get_info(self):
        return self.free_mib() * MIB, TOTAL * MIB

    def memory_reserved(self):
        return self.reserved_mib() * MIB

    def is_current_stream_capturing(self):
        return False

    def synchronize(self):
        pass

    def empty_cache(self):
        self.empty_calls += 1
        self.segs = [s for s in self.segs if s["used"]]

    # -- the extend's own allocations ---------------------------------------
    def malloc(self, size):
        for s in self.segs:
            if not s["used"] and s["stream"] == 0 and s["size"] >= size:
                s["used"] = True
                return s
        if self.free_mib() < size:
            self.retries += 1
            self.empty_cache()
            self.empty_calls -= 1  # the allocator's own release, not a trim
            if self.free_mib() < size:
                self.ooms += 1
                raise MemoryError(size)
        s = {"size": size, "used": True, "stream": 0}
        self.segs.append(s)
        self.min_free = min(self.min_free, self.free_mib())
        return s

    @staticmethod
    def free(s):
        s["used"] = False


def _extend(cuda):
    """rc12d 02:33:10's shape: two blocks live together (621 MiB), then one
    larger block neither freed segment can hold -> 1174 MiB new segments."""
    a = cuda.malloc(300)
    b = cuda.malloc(321)
    cuda.free(a)
    cuda.free(b)
    c = cuda.malloc(553)
    cuda.free(c)
    return 621


def _rc12d_card():
    # reserved 28740 = allocated 27757 + 983 cached of another stream;
    # non-torch 1600 -> card free 1749 before the extend
    return FakeCuda(nontorch=1600, live=27757, stale_cache=983)


class TheExtendRecreatesSegmentsOnAShortCard(CustomTestCase):
    def test_without_the_trim_reserved_grows_past_the_transient(self):
        cuda = _rc12d_card()
        r0 = cuda.reserved_mib()
        transient = _extend(cuda)
        growth = cuda.reserved_mib() - r0
        self.assertEqual(growth, 1174)             # rc12d 02:33:10: +1174
        self.assertGreater(growth, transient)      # 1174 > 621
        self.assertGreater(growth, ACT)            # past the booked activation
        self.assertLess(cuda.min_free, FLOOR)      # 575 < 767 (rc12d: 619)

    def test_the_trim_keeps_growth_under_the_booked_activation(self):
        cuda = _rc12d_card()
        r0 = cuda.reserved_mib()
        line = ET.maybe_trim(cuda, 0, float(FLOOR + ACT), clock=iter([1.0, 1.0042]).__next__)
        self.assertIsNotNone(line)
        self.assertIn("PDFLIP-EXTEND-CACHE-TRIM rank=0 ms=4.2 released=983 "
                      "card_free_before=1749 card_free_after=2732 threshold=1791", line)
        _extend(cuda)
        growth = cuda.reserved_mib() - r0          # against the reserved BEFORE trim + extend
        self.assertLessEqual(growth, ACT)          # 1174 - 983 = 191
        self.assertGreaterEqual(cuda.min_free, FLOOR)
        self.assertEqual((cuda.retries, cuda.ooms), (0, 0))

    def test_the_hook_trims_a_target_extend_only(self):
        ET._CACHE["thresholds"] = [float(FLOOR + ACT), 1700.0, 1700.0]
        self.addCleanup(ET.reset_for_tests)
        cuda = _rc12d_card()
        import torch

        orig = torch.cuda
        torch.cuda = cuda
        self.addCleanup(setattr, torch, "cuda", orig)
        worker = types.SimpleNamespace(tp_rank=0, is_draft_worker=False)
        mode = types.SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False)
        decode = types.SimpleNamespace(is_extend=lambda: False, is_target_verify=lambda: False)
        self.assertIsNone(ET.before_extend(worker, types.SimpleNamespace(forward_mode=decode)))
        draft = types.SimpleNamespace(tp_rank=0, is_draft_worker=True, is_phase_flip_tp_stack=False)
        self.assertIsNone(ET.before_extend(draft, types.SimpleNamespace(forward_mode=mode)))
        self.assertEqual(cuda.empty_calls, 0)
        self.assertIn("released=983", ET.before_extend(worker, types.SimpleNamespace(forward_mode=mode)))
        self.assertEqual(cuda.empty_calls, 1)


class TheTrimStaysOutOfTheWayWithRoom(CustomTestCase):
    def test_no_trim_above_the_threshold(self):
        cuda = FakeCuda(nontorch=1000, live=27757, stale_cache=983)   # free 2349 >= 1791
        self.assertIsNone(ET.maybe_trim(cuda, 0, float(FLOOR + ACT)))
        self.assertEqual(cuda.empty_calls, 0)
        self.assertEqual(cuda.reserved_mib(), 27757 + 983)

    def test_no_threshold_no_cuda_call(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"cuda.{name} read without a threshold")

        self.assertIsNone(ET.maybe_trim(Untouchable(), 0, None))

    def test_malformed_env_is_off(self):
        self.assertIsNone(ET.parse_thresholds("1791,abc"))
        self.assertIsNone(ET.parse_thresholds(""))
        self.assertIsNone(ET.parse_thresholds(None))
        self.assertEqual(ET.parse_thresholds("1791, 1724,1725"), [1791.0, 1724.0, 1725.0])
        self.assertEqual(ET.threshold_for(2, [1791.0, 1724.0, 1725.0]), 1725.0)
        self.assertEqual(ET.threshold_for(5, [1791.0]), 1791.0)
        self.assertIsNone(ET.threshold_for(5, [1791.0, 1.0]))


def _rc12c_test_module():
    p = Path(__file__).with_name("test_pdflip_d_awake_rest_rc12c.py")
    spec = importlib.util.spec_from_file_location("_rc12c_terms", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TheLauncherWritesTheThresholdForNFOnly(CustomTestCase):
    def test_nf_threshold_is_floor_plus_booked_activation(self):
        T = _rc12c_test_module()
        terms = []
        _c, budgets = T._d_pass("nextflash", [], terms=terms)
        led = L.d_card_ledger(terms, budgets, "D")
        fits = T._fits(budgets, (91, 48, 48))
        self.assertEqual(L.d_extend_trim_env(led, fits), "1791,1724,1725")

    def test_27b_gets_no_threshold_and_no_hook_work(self):
        # qwen27b has no D_AWAKE_REST record -> no card_terms -> no ledger ->
        # log_d_rank_vram_solve never writes FLLIPER_PDFLIP_EXTEND_TRIM_MIB
        T = _rc12c_test_module()
        self.assertEqual(L.d_awake_rest(T._cards(), "qwen27b"), (None, ""))
        ET.reset_for_tests()
        self.addCleanup(ET.reset_for_tests)
        from flliper.srt.environ import envs

        with envs.FLLIPER_PDFLIP_EXTEND_TRIM_MIB.override(None):
            class NoBatchRead:
                def __getattr__(self, name):
                    raise AssertionError(f"batch.{name} read with the trim off")

            self.assertIsNone(ET.before_extend(types.SimpleNamespace(tp_rank=0), NoBatchRead()))
            self.assertIsNone(ET.thresholds())

    def test_a_missing_rank_writes_nothing(self):
        led = types.SimpleNamespace(floor_mib=(767.0, 700.0, 701.0))
        fits = [types.SimpleNamespace(rank=0, activation_mib=1024.0)]
        self.assertEqual(L.d_extend_trim_env(led, fits), "")


if __name__ == "__main__":
    unittest.main()
