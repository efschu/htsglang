# SPDX-License-Identifier: Apache-2.0
"""A non-divisible prefill/decode TP pair must be REFUSED, not staged (#643).

``compute_head_slice_params`` (``common/staging_buffer.py:686-717``) and the
three transport copies of the same arithmetic (``mooncake/conn.py:816-870``,
``nixl/conn.py``, ``mori/conn.py``) split KV heads with a bare floor division::

    src_heads_per_rank = max(1, total_kv_heads // src_attn_tp_size)
    dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)

That is only a partition when the two TP sizes stand in an integer-multiple
relationship. When neither divides the other -- prefill TP=3 against decode
TP=2 is the smallest real case -- three independent defects fire at once, and
every one of them is silent:

1. OVERLAP. Two source ranks compute the same destination head offset and
   write the same bytes. Last writer wins; the other rank's heads are lost.
2. OUT-OF-RANGE. A source rank computes ``dst_head_start + num_heads`` beyond
   the destination rank's own head capacity ``total // dst_attn_tp_size``,
   i.e. a write past the end of that rank's slice.
3. UNDERSIZED STAGING. ``compute_staging_layout`` sizes the staging region for
   ``src_attn_tp_size // dst_attn_tp_size`` writers (``staging_buffer.py:736``,
   mirrored at ``mooncake/conn.py:1366``). Floor division rounds the writer
   count DOWN, so more ranks write into the region than it was sized for.

``HazardTest`` is the point of this file, and it follows #642's rule: a guard
whose test only checks that the guard raises proves nothing about the hazard
it claims to prevent. The corruption is therefore demonstrated FIRST, from the
shipped arithmetic itself, in pure integers -- no GPU, no tensors, no
transport. ``GuardTest`` then shows the configuration can no longer be reached.

Relationship to the neighbours: #641 fixed the total (``per_rank * tp_size``
was the wrong global count) and #642 fixed the coordinate system the draft
pool is addressed in. Both left the SPLIT itself alone -- #641's own
``TestDivisibilityObservations`` records the truncation and declares it out of
scope. This file closes that item.

Why refusal rather than a general split: a correct many-to-many split is
expressible (intersect each source rank's head interval with each destination
rank's), but it changes the transfer loop structure in three transport
backends whose correctness cannot be established without a two-instance PD
boot. Shipping a loud refusal is the part that can be validated at desk. The
general split is written up as a recipe in
``docs/dev/HANDOFF_643_BUNDLE.md`` rather than shipped unwired.
"""

import unittest

from sglang.srt.disaggregation.common.staging_buffer import (
    compute_head_slice_params,
    compute_staging_layout,
)
from sglang.srt.disaggregation.common.tp_pair import (
    HeadSplitNotRepresentable,
    validate_tp_pair_divisible,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# The shipped arithmetic, restated once, so the hazard is provable without
# calling the (now guarded) function.
# ---------------------------------------------------------------------------


def _unguarded_head_slice(src_tp, dst_tp, src_rank, total):
    """``compute_head_slice_params`` as it read BEFORE this fix.

    Copied verbatim from ``staging_buffer.py:698-717`` at aca5037531 so the
    hazard tests keep demonstrating the real defect after the guard makes the
    live function refuse. Kept in the test, never in the shipped path.
    """
    src_heads_per_rank = max(1, total // src_tp)
    dst_heads_per_rank = max(1, total // dst_tp)
    local_tp_rank = src_rank % src_tp
    if src_tp > dst_tp:
        num_heads_to_send = src_heads_per_rank
        src_replication = max(1, src_tp // total)
        unique_head_idx = local_tp_rank // src_replication
        dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
    else:
        num_heads_to_send = dst_heads_per_rank
        dst_head_start = 0
    return dst_head_start, num_heads_to_send


def _coverage(src_tp, dst_tp, total):
    """Map destination head index -> number of source ranks writing it."""
    cover = {}
    for r in range(src_tp):
        start, n = _unguarded_head_slice(src_tp, dst_tp, r, total)
        for h in range(start, start + n):
            cover[h] = cover.get(h, 0) + 1
    return cover


class HazardTest(CustomTestCase):
    """The corruption, shown from the arithmetic, before any guard exists."""

    def test_non_divisible_pair_overlaps_destination_heads(self):
        """TP 3 -> 2, 6 KV heads: two source ranks target the same heads.

        src rank 0 -> dst heads [0,2)
        src rank 1 -> dst heads [2,4)
        src rank 2 -> dst heads [1,3)

        Heads 1 and 2 are each written by two ranks. One rank's KV is
        silently destroyed by the other's.
        """
        cover = _coverage(src_tp=3, dst_tp=2, total=6)
        overwritten = sorted(h for h, c in cover.items() if c > 1)
        self.assertEqual(
            overwritten,
            [1, 2],
            "expected heads 1 and 2 to be double-written under TP 3->2",
        )

    def test_non_divisible_pair_writes_past_destination_capacity(self):
        """TP 3 -> 2, 6 KV heads: a write lands outside the dst rank's slice.

        Each destination rank owns ``6 // 2 == 3`` heads, valid indices
        [0,3). Source rank 1 computes dst_head_start=2 and sends 2 heads,
        i.e. [2,4) -- head 3 is past the end.
        """
        capacity = 6 // 2
        cover = _coverage(src_tp=3, dst_tp=2, total=6)
        out_of_range = sorted(h for h in cover if h >= capacity)
        self.assertEqual(
            out_of_range,
            [3],
            "expected a write past the destination rank's head capacity",
        )

    def test_non_divisible_pair_undersizes_the_staging_region(self):
        """TP 4 -> 3: staging is sized for 1 writer, 4 ranks write.

        ``num_writers = src_attn_tp_size // dst_attn_tp_size`` = 4 // 3 = 1.
        Four prefill ranks nonetheless produce KV for this decode rank, so
        the region is undersized by a factor of four.
        """
        src_tp, dst_tp = 4, 3
        num_writers = src_tp // dst_tp
        self.assertEqual(num_writers, 1)
        self.assertLess(
            num_writers,
            src_tp,
            "floor division rounds the writer count down; the region is "
            "sized for fewer writers than actually write into it",
        )

    def test_divisible_pairs_stay_inside_capacity(self):
        """The contrast case: integer-multiple pairs never exceed capacity.

        This is what makes out-of-range a sound discriminator rather than an
        artefact of the probe.
        """
        for total, src_tp, dst_tp in [(8, 4, 2), (8, 2, 4), (6, 3, 1), (6, 2, 2)]:
            with self.subTest(total=total, src=src_tp, dst=dst_tp):
                capacity = max(1, total // dst_tp)
                cover = _coverage(src_tp, dst_tp, total)
                self.assertEqual(
                    [h for h in cover if h >= capacity],
                    [],
                    f"divisible pair {src_tp}->{dst_tp} must stay in range",
                )


class GuardTest(CustomTestCase):
    """After the fix the configuration cannot be reached silently."""

    def test_validate_refuses_non_divisible_pair(self):
        with self.assertRaises(HeadSplitNotRepresentable) as cm:
            validate_tp_pair_divisible(3, 2, total_kv_heads=6, where="unit-test")
        msg = str(cm.exception)
        # The message must name both sizes and the site, so an operator can
        # act on it without reading the source.
        self.assertIn("3", msg)
        self.assertIn("2", msg)
        self.assertIn("unit-test", msg)

    def test_validate_admits_divisible_pairs(self):
        for src_tp, dst_tp in [(4, 2), (2, 4), (3, 1), (1, 3), (2, 2), (6, 3)]:
            with self.subTest(src=src_tp, dst=dst_tp):
                validate_tp_pair_divisible(
                    src_tp, dst_tp, total_kv_heads=12, where="unit-test"
                )

    def test_compute_head_slice_params_refuses_non_divisible_pair(self):
        """The chokepoint itself refuses -- not just a separate validator."""
        with self.assertRaises(HeadSplitNotRepresentable):
            compute_head_slice_params(
                src_attn_tp_size=3,
                dst_attn_tp_size=2,
                src_tp_rank=0,
                dst_tp_rank=0,
                total_kv_heads=6,
            )

    def test_compute_staging_layout_refuses_non_divisible_pair(self):
        with self.assertRaises(HeadSplitNotRepresentable):
            compute_staging_layout(
                src_attn_tp_size=3,
                dst_attn_tp_size=2,
                dst_tp_rank=0,
                total_kv_heads=6,
                num_tokens=1,
                bytes_per_head_token=1,
                num_layers=1,
            )

    def test_divisible_pair_still_computes(self):
        """Backward compatibility: the supported cases are untouched."""
        src_start, n, dst_start, n2 = compute_head_slice_params(
            src_attn_tp_size=4,
            dst_attn_tp_size=2,
            src_tp_rank=0,
            dst_tp_rank=0,
            total_kv_heads=8,
        )
        self.assertEqual(n, n2)
        self.assertEqual(src_start, 0)
        self.assertEqual(dst_start, 0)


class HandshakeRefusalTest(CustomTestCase):
    """The refusal an operator actually meets: the decode arm's handshake.

    Modelled on ``test_pd_model_identity_guard_631a.HandshakeIdentityGuardTest``
    -- same stub shape, same ``requests.get`` patch. This is the point where
    the pair first exists; a PD arm's ServerArgs never names the peer's TP
    size, so no parse-time gate could have caught it.
    """

    def _manager(self, decode_attn_tp_size):
        from types import SimpleNamespace

        # Same field set as the #631a handshake stub, which this mirrors.
        server_args = SimpleNamespace(
            model_path="/models/qwen3.6-27b",
            revision=None,
            dtype="auto",
            quantization=None,
            kv_cache_dtype="fp8_e4m3",
            rank_tp_ratio=None,
            rank_kv_ratio=None,
        )
        return SimpleNamespace(
            prefill_info_table={},
            kv_args=SimpleNamespace(page_size=1, total_kv_head_num=6),
            server_args=server_args,
            attn_tp_size=decode_attn_tp_size,
            _resolve_rank_mapping=lambda info: None,
        )

    def _run(self, prefill_attn_tp_size, decode_attn_tp_size):
        from unittest import mock

        from sglang.srt.disaggregation.common.conn import CommonKVManager

        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": prefill_attn_tp_size,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 1,
            "kv_cache_dtype": "fp8_e4m3",
            "follow_bootstrap_room": False,
        }
        with mock.patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            return CommonKVManager.try_ensure_parallel_info(
                self._manager(decode_attn_tp_size), "127.0.0.1:8998"
            )

    def test_non_divisible_pair_is_refused_at_handshake(self):
        """Prefill TP=3 against decode TP=2 must not pair."""
        with self.assertRaises(HeadSplitNotRepresentable) as cm:
            self._run(prefill_attn_tp_size=3, decode_attn_tp_size=2)
        msg = str(cm.exception)
        self.assertIn("non-divisible TP pair", msg)
        self.assertIn("127.0.0.1:8998", msg, "refusal must name the peer")
        self.assertIn("attn_tp_size=3", msg)
        self.assertIn("attn_tp_size=2", msg)

    def test_divisible_pairs_still_pair(self):
        """Differing-but-divisible TP is supported and must remain so.

        This is the backward-compatibility pin: #643 must not refuse the
        heterogeneous pairs the engine transfers correctly today.
        """
        for prefill_tp, decode_tp in [(1, 1), (4, 2), (2, 4), (3, 3), (6, 2)]:
            with self.subTest(prefill=prefill_tp, decode=decode_tp):
                self.assertTrue(self._run(prefill_tp, decode_tp))


if __name__ == "__main__":
    unittest.main()
