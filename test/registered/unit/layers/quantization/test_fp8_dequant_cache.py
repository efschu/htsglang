"""Budgeted dequant cache (#179 candidate 3).

The fp8 dequant fallback expands the weight ONCE PER FORWARD, so its cost is
paid per forward, not per token. Measured on this rig (Qwen3.6-27B fp8
block-scaled, TP=3, forced fallback):

    decode   91.53 -> 27.59 tok/s   (-69.9%; 10.93 -> 36.25 ms per token)
    prefill  1472.5 -> 1353.1 tok/s (-8.1%)

Caching the expansion is only legitimate because the dequantised weight is a
PURE function of (weight, scale, block_size, out_dtype) -- established by
falsification before the cache was written. These tests pin the properties the
cache depends on, and gate it on BYTE EQUALITY against the uncached path rather
than on "looks about right".
"""

import unittest
import unittest.mock

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    cached_dequant,
    dequant_cache_clear,
    dequant_cache_stats,
    dequant_fp8_block_weight,
    dequant_fp8_weight,
)


def _mib(n):
    return str(int(n))


class TestDequantPurity(unittest.TestCase):
    """The property the whole design rests on. If this fails, the cache is
    invalid and must be removed, not patched."""

    def setUp(self):
        torch.manual_seed(0)
        self.n, self.k, self.bn, self.bk = 256, 128, 128, 128
        self.w = torch.randn(self.n, self.k).to(torch.float8_e4m3fn)
        self.s = torch.rand(self.n // self.bn, self.k // self.bk) + 0.5

    def _blk(self, dtype=torch.bfloat16):
        return dequant_fp8_block_weight(self.w, self.s, [self.bn, self.bk], dtype)

    def test_repeated_calls_are_bit_identical(self):
        self.assertTrue(torch.equal(self._blk(), self._blk()))

    def test_intervening_other_dtype_does_not_perturb(self):
        a = self._blk()
        self._blk(torch.float16)
        self.assertTrue(torch.equal(a, self._blk()))

    def test_dtype_changes_the_result_so_it_belongs_in_the_key(self):
        self.assertNotEqual(self._blk().dtype, self._blk(torch.float16).dtype)

    def test_inputs_are_not_mutated(self):
        w0, s0 = self.w.clone(), self.s.clone()
        self._blk()
        self.assertTrue(torch.equal(self.w, w0))
        self.assertTrue(torch.equal(self.s, s0))

    def test_ragged_partial_block_is_also_pure(self):
        n, k = 200, 100
        w = torch.randn(n, k).to(torch.float8_e4m3fn)
        s = torch.rand((n + 127) // 128, (k + 127) // 128) + 0.5
        a = dequant_fp8_block_weight(w, s, [128, 128], torch.bfloat16)
        b = dequant_fp8_block_weight(w, s, [128, 128], torch.bfloat16)
        self.assertTrue(torch.equal(a, b))


class TestCacheIsByteExact(unittest.TestCase):
    """A cache that returns *nearly* the right weight is a silent accuracy bug,
    so the gate is byte equality against the uncached path."""

    def setUp(self):
        dequant_cache_clear()
        torch.manual_seed(1)
        self.w = torch.randn(256, 128).to(torch.float8_e4m3fn)
        self.s = torch.rand(2, 1) + 0.5

    def _make(self, dtype=torch.bfloat16):
        return dequant_fp8_block_weight(self.w, self.s, [128, 128], dtype)

    def test_cached_equals_uncached_bytes(self):
        with unittest.mock.patch.dict(
            "os.environ", {"SGLANG_FP8_DEQUANT_CACHE_MIB": _mib(64)}
        ):
            dequant_cache_clear()
            from sglang.srt.layers.quantization import fp8_utils

            fp8_utils._dequant_cache._budget = None  # re-read env
            uncached = self._make()
            first = cached_dequant(self.w, torch.bfloat16, self._make)
            second = cached_dequant(self.w, torch.bfloat16, self._make)
            self.assertTrue(torch.equal(uncached, first))
            self.assertTrue(torch.equal(uncached, second))
            self.assertIs(first, second)  # second call really came from cache

    def test_off_by_default_is_a_passthrough(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("SGLANG_FP8_DEQUANT_CACHE_MIB", None)
            from sglang.srt.layers.quantization import fp8_utils

            fp8_utils._dequant_cache._budget = None
            dequant_cache_clear()
            a = cached_dequant(self.w, torch.bfloat16, self._make)
            b = cached_dequant(self.w, torch.bfloat16, self._make)
            self.assertTrue(torch.equal(a, b))
            self.assertIsNot(a, b)  # no caching happened
            self.assertEqual(dequant_cache_stats()["entries"], 0)


class TestCacheInvalidationAndBudget(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.quantization import fp8_utils

        self.fp8_utils = fp8_utils
        dequant_cache_clear()
        torch.manual_seed(2)

    def _budgeted(self, mib):
        self.fp8_utils._dequant_cache._budget = mib * 1024 * 1024
        dequant_cache_clear()

    def test_inplace_weight_change_invalidates(self):
        """Weights are NOT immutable (weight loading, LoRA, checkpoint engine).
        Binding to data_ptr alone would serve a stale weight; the entry is bound
        to _version so an in-place write discards it."""
        self._budgeted(64)
        w = torch.randn(128, 128).to(torch.float8_e4m3fn)
        s = torch.rand(1, 1) + 0.5
        mk = lambda: dequant_fp8_block_weight(w, s, [128, 128], torch.bfloat16)
        first = cached_dequant(w, torch.bfloat16, mk)
        with torch.no_grad():
            w.copy_(torch.randn(128, 128).to(torch.float8_e4m3fn))  # bumps _version
        second = cached_dequant(w, torch.bfloat16, mk)
        self.assertIsNot(first, second, "stale weight served after in-place write")

    def test_budget_evicts_lru_and_stays_within_budget(self):
        # Budget 1 MiB; each entry is 512*512*2 B = 0.5 MiB dequantised, so
        # eight entries are 4 MiB and eviction MUST occur. (Sized deliberately:
        # an earlier version used 256x256, whose eight entries fit inside the
        # budget, so it asserted eviction that correctly never happened.)
        self._budgeted(1)
        ws = [torch.randn(512, 512).to(torch.float8_e4m3fn) for _ in range(8)]
        s = torch.rand(4, 4) + 0.5
        for w in ws:
            cached_dequant(
                w,
                torch.bfloat16,
                lambda w=w: dequant_fp8_block_weight(w, s, [128, 128], torch.bfloat16),
            )
        st = dequant_cache_stats()
        self.assertLessEqual(st["bytes"], st["budget_bytes"])
        self.assertGreater(st["evictions"], 0)

    def test_entry_larger_than_budget_is_never_cached(self):
        """and must not evict the whole cache trying to make room for itself."""
        self._budgeted(1)
        small = torch.randn(64, 64).to(torch.float8_e4m3fn)
        s_small = torch.rand(1, 1) + 0.5
        cached_dequant(
            small,
            torch.bfloat16,
            lambda: dequant_fp8_block_weight(small, s_small, [128, 128], torch.bfloat16),
        )
        kept = dequant_cache_stats()["entries"]
        big = torch.randn(2048, 2048).to(torch.float8_e4m3fn)
        s_big = torch.rand(16, 16) + 0.5
        cached_dequant(
            big,
            torch.bfloat16,
            lambda: dequant_fp8_block_weight(big, s_big, [128, 128], torch.bfloat16),
        )
        self.assertEqual(dequant_cache_stats()["entries"], kept)

    def test_dtype_is_part_of_the_key(self):
        self._budgeted(64)
        w = torch.randn(128, 128).to(torch.float8_e4m3fn)
        s = torch.rand(1, 1) + 0.5
        a = cached_dequant(
            w, torch.bfloat16,
            lambda: dequant_fp8_block_weight(w, s, [128, 128], torch.bfloat16),
        )
        b = cached_dequant(
            w, torch.float16,
            lambda: dequant_fp8_block_weight(w, s, [128, 128], torch.float16),
        )
        self.assertEqual(a.dtype, torch.bfloat16)
        self.assertEqual(b.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
