"""#641: total_kv_head_num must not be derived from per_rank * tp_size.

Under uneven TP each rank holds a different number of KV heads.
Multiplying one rank's share by the rank count produces a wrong total.

The fix uses model_config.get_total_num_kv_heads() (already correct on the
prefill side) instead of the broken multiplication on the decode side.

Four latent fallbacks in staging_buffer.py and the conn.py backends repeat the
same multiplication; they are replaced with named hard errors after the primary
fix makes total_kv_head_num always present.

Reachability: the code path is gated behind
SGLANG_DISAGG_STAGING_BUFFER (default False, environ.py:919).
With staging disabled the multiplication is never reached -- LATENT, not LIVE.
"""

import types
import unittest

from sglang.srt.disaggregation.common.staging_buffer import (
    compute_head_slice_params,
    resolve_total_kv_heads,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_args(**kw):
    """Build a kv_args-like namespace for tests."""
    defaults = dict(
        total_kv_head_num=0,
        kv_head_num=0,
        engine_rank=0,
        kv_item_lens=[256],
        kv_data_ptrs=[0x1000],
        kv_data_lens=[1024],
        page_size=16,
        gpu_id=0,
        ib_device="",
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _even_kv_args(total: int, per_rank: int, tp_size: int):
    """Even TP: per_rank * tp_size == total."""
    return _make_kv_args(
        total_kv_head_num=total,
        kv_head_num=per_rank,
    )


def _uneven_kv_args(total: int, rank0_heads: int, tp_size: int):
    """Uneven TP: rank0_heads * tp_size != total.

    total        = true model total (e.g. 4 KV heads)
    rank0_heads  = what this rank actually holds (e.g. 3)
    tp_size      = number of ranks (e.g. 2)

    With even TP and total=4, tp_size=2, per_rank would be 2.
    Here rank0 holds 3 heads, rank1 holds 1 head.
    rank0_heads * tp_size = 3*2 = 6, but the correct total is 4.
    """
    return _make_kv_args(
        total_kv_head_num=total,
        kv_head_num=rank0_heads,
    )


# ---------------------------------------------------------------------------
# Tests: primary fix -- decode.py should set total from model config
# ---------------------------------------------------------------------------

class TestResolveTotalKvHeadsFromArgs(unittest.TestCase):
    """resolve_total_kv_heads must honour the explicit total_kv_head_num
    over the fallback multiplication, which is incorrect under uneven TP."""

    def test_even_tp_prefill_sets_correct_total(self):
        """When prefill sets total correctly, resolve_total_kv_heads returns it."""
        kv_args = _even_kv_args(total=4, per_rank=2, tp_size=2)
        self.assertEqual(resolve_total_kv_heads(kv_args, 2), 4)

    def test_uneven_tp_uses_total_not_fallback(self):
        """Under uneven TP, total_kv_head_num (4) takes precedence over
        the broken fallback (kv_head_num * tp_size = 3*2 = 6)."""
        kv_args = _uneven_kv_args(total=4, rank0_heads=3, tp_size=2)
        self.assertEqual(resolve_total_kv_heads(kv_args, 2), 4)

    def test_head_slice_with_even_total(self):
        """Even TP: head slice params use the correct total."""
        src_head_start, num_src, dst_head_start, num_dst = compute_head_slice_params(
            src_attn_tp_size=2,
            dst_attn_tp_size=2,
            src_tp_rank=0,
            dst_tp_rank=0,
            total_kv_heads=4,
        )
        self.assertEqual(src_head_start, 0)
        self.assertEqual(dst_head_start, 0)

    def test_head_slice_with_uneven_total_produces_wrong_ranges_when_fallback(self):
        """Mechanical proof that the broken fallback (per_rank*tp_size)
        feeds compute_head_slice_params with the wrong total, shifting
        head ranges and producing silently corrupt attention."""
        # Simulate: 4 true total, tp=2, rank0 has 3 heads, rank1 has 1 head.
        true_total = 4
        buggy_total = 3 * 2  # per_rank=3 * tp_size=2 = 6 (WRONG)

        _, num_heads_correct, _, _ = compute_head_slice_params(
            src_attn_tp_size=2, dst_attn_tp_size=2,
            src_tp_rank=0, dst_tp_rank=0,
            total_kv_heads=true_total,
        )
        _, num_heads_buggy, _, _ = compute_head_slice_params(
            src_attn_tp_size=2, dst_attn_tp_size=2,
            src_tp_rank=0, dst_tp_rank=0,
            total_kv_heads=buggy_total,
        )

        # With correct total=4:  heads_per_rank = 4//2 = 2
        # With buggy total=6:    heads_per_rank = 6//2 = 3
        # -> the wrong total shifts head boundaries.
        self.assertNotEqual(num_heads_correct, num_heads_buggy)


# ---------------------------------------------------------------------------
# Tests: fallback hard-errors
# ---------------------------------------------------------------------------

class TestFallbackBecomesHardError(unittest.TestCase):
    """After the primary fix, total_kv_head_num is always set upstream.
    The fallback path (total missing but kv_head_num present) must error
    rather than silently returning a plausible-but-wrong number."""

    def test_missing_total_with_per_rank_raises(self):
        """total_kv_head_num=0 but kv_head_num present -> ValueError."""
        kv_args = _make_kv_args(
            total_kv_head_num=0,  # not set upstream
            kv_head_num=3,
        )
        with self.assertRaises(ValueError) as cm:
            resolve_total_kv_heads(kv_args, 2)
        # The error message should explain that total_kv_head_num is required.
        exc_text = str(cm.exception)
        self.assertIn("total_kv_head_num", exc_text)

    def test_both_missing_raises(self):
        """Neither total nor per-rank set -> ValueError."""
        kv_args = _make_kv_args(
            total_kv_head_num=0,
            kv_head_num=0,
        )
        with self.assertRaises(ValueError):
            resolve_total_kv_heads(kv_args, 2)

    def test_explicit_total_always_used(self):
        """Even when kv_head_num is also present, explicit total wins."""
        kv_args = _make_kv_args(
            total_kv_head_num=4,
            kv_head_num=3,  # different value -- should NOT be used
        )
        self.assertEqual(resolve_total_kv_heads(kv_args, 2), 4)


# ---------------------------------------------------------------------------
# Divisibility audit (informational -- not fixed in this branch)
# ---------------------------------------------------------------------------

class TestDivisibilityObservations(unittest.TestCase):
    """Document that // operations on total_kv_heads carry no divisibility
    guard.  These are listed for awareness; fixing them is out of scope."""

    def test_uneven_division_truncates(self):
        """Compute: total_kv_heads // tp_size with non-divisible total.

        File: staging_buffer.py:698-699
        Operation: total_kv_heads // src_attn_tp_size

        When total_kv_heads=7, src_attn_tp_size=3:
          7 // 3 = 2 per rank, but 2*3 = 6, so 1 head is unaccounted for.
        The function uses max(1, ...) to avoid zero, but truncation is silent.
        """
        src_head_start, num_heads, _, _ = compute_head_slice_params(
            src_attn_tp_size=3,
            dst_attn_tp_size=3,
            src_tp_rank=0,
            dst_tp_rank=0,
            total_kv_heads=7,
        )
        heads_per_rank = 7 // 3  # 2, but 2*3 = 6 < 7
        self.assertEqual(heads_per_rank, 2)
        self.assertEqual(num_heads, 2)
        # 3 ranks * 2 heads = 6, but there are 7 total -- 1 head lost.
        self.assertEqual(src_head_start, 0)

    def test_mooncake_conn_division_path(self):
        """mooncake/conn.py:840-841 (and nixl/conn.py:744-745,
        mori/conn.py:772-773) use the same unguarded division pattern:
          src_heads_per_rank = max(1, total_kv_heads // self.attn_tp_size)

        Listed here as documentation.  No guard exists; non-divisible
        values truncate silently.
        """
        total = 7
        tp_size = 3
        self.assertEqual(total // tp_size, 2)
        self.assertNotEqual(2 * tp_size, total)  # truncation


if __name__ == "__main__":
    unittest.main()
