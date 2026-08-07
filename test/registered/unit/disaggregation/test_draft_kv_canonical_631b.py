# SPDX-License-Identifier: Apache-2.0
"""The canonical full-head draft-KV layout (#631b, variant iii).

Three properties, and each exists because getting it wrong is silent:

1. **The layout is versioned and a mismatch REFUSES.** If one arm can
   reinterpret the other's bytes under a different layout, the silent
   wrong-output failure that variant (iii) was chosen to avoid has been
   rebuilt inside variant (iii).
2. **The full-head shipment justifies itself from RUNTIME geometry.** Shipping
   every KV head beat local recompute on one quantitative ground -- the whole
   draft KV is smaller than the hidden states recompute would need. That is a
   property of the checkpoint. A wide-KV model inverts it, and the code must
   refuse there instead of moving more bytes than the option it replaced.
3. **Head dealing loses no head.** 4 KV heads over 3 ranks is the case the
   whole variant exists for. ``num_kv_heads // tp_size`` gives 1 and drops
   head 3, silently, because nothing downstream counts heads. That truncation
   is the arithmetic wall general reslicing hits, and it is pinned here.
"""

import unittest

from sglang.srt.disaggregation.draft_kv_canonical import (
    CANONICAL_LAYOUT_VERSION,
    DraftKvCanonicalLayout,
    DraftKvLayoutMismatch,
    check_full_head_shipment_is_justified,
    local_head_window,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The reference checkpoint: Qwen3.6-27B, GQA, fp8 KV.
_REF = dict(
    version=CANONICAL_LAYOUT_VERSION,
    num_kv_heads=4,
    head_dim=256,
    element_size=1,
    num_draft_layers=1,
)
_REF_HIDDEN_SIZE = 5120
_REF_HIDDEN_ELEMENT_SIZE = 2  # bf16 activations


def _layout(**over):
    return DraftKvCanonicalLayout(**{**_REF, **over})


class LayoutVersionTest(CustomTestCase):
    def test_matching_layouts_are_compatible(self):
        _layout().assert_compatible(_layout(), peer="decode")

    def test_version_mismatch_refuses_and_says_so(self):
        with self.assertRaises(DraftKvLayoutMismatch) as ctx:
            _layout().assert_compatible(_layout(version=99), peer="decode-arm")
        msg = str(ctx.exception)
        self.assertIn("version mismatch", msg)
        self.assertIn("decode-arm", msg, "refusal must name the peer")
        self.assertIn(str(CANONICAL_LAYOUT_VERSION), msg)
        self.assertIn("99", msg)

    def test_geometry_mismatch_at_same_version_refuses(self):
        """Same version, different meaning, is the more dangerous case."""
        for field, value in (
            ("num_kv_heads", 8),
            ("head_dim", 128),
            ("element_size", 2),
            ("num_draft_layers", 2),
        ):
            with self.subTest(field=field):
                with self.assertRaises(DraftKvLayoutMismatch) as ctx:
                    _layout().assert_compatible(_layout(**{field: value}), peer="p")
                self.assertIn(field, str(ctx.exception))


class ShipmentJustificationTest(CustomTestCase):
    def test_reference_checkpoint_is_justified(self):
        layout = _layout()
        self.assertEqual(layout.bytes_per_token(), 2048)
        check_full_head_shipment_is_justified(
            layout, _REF_HIDDEN_SIZE, _REF_HIDDEN_ELEMENT_SIZE
        )

    def test_wide_kv_checkpoint_is_refused(self):
        """An MHA-ish checkpoint inverts the comparison, so it must refuse.

        This is the test that stops the reference model's numbers from being
        baked in as a standing assumption.
        """
        wide = _layout(num_kv_heads=64, head_dim=128)
        self.assertGreater(
            wide.bytes_per_token(), _REF_HIDDEN_SIZE * _REF_HIDDEN_ELEMENT_SIZE
        )
        with self.assertRaises(DraftKvLayoutMismatch) as ctx:
            check_full_head_shipment_is_justified(
                wide, _REF_HIDDEN_SIZE, _REF_HIDDEN_ELEMENT_SIZE
            )
        msg = str(ctx.exception)
        self.assertIn("not justified", msg)
        self.assertIn("B/token", msg, "refusal must show the arithmetic")

    def test_bound_is_the_alternative_not_a_constant(self):
        """Widening the hidden state must widen what is admissible.

        A layout refused against a small hidden size must be accepted against
        a large one -- which is only true if the bound really is the
        alternative's cost rather than a hidden constant.
        """
        layout = _layout(num_kv_heads=16)  # 8192 B/token
        with self.assertRaises(DraftKvLayoutMismatch):
            check_full_head_shipment_is_justified(layout, 2048, 2)  # 4096 B/token
        check_full_head_shipment_is_justified(layout, 8192, 2)  # 16384 B/token


class HeadWindowTest(CustomTestCase):
    def test_four_heads_over_three_ranks_loses_nothing(self):
        """The case the variant exists for."""
        windows = [local_head_window(4, 3, r) for r in range(3)]
        self.assertEqual(windows, [(0, 2), (2, 3), (3, 4)])

        covered = [h for start, end in windows for h in range(start, end)]
        self.assertEqual(sorted(covered), [0, 1, 2, 3], "every head owned once")
        self.assertEqual(len(covered), len(set(covered)), "no head owned twice")

    def test_naive_floor_division_would_drop_a_head(self):
        """Pin the wall, so nobody 'simplifies' back into it.

        This is what staging_buffer.compute_head_slice_params does, and why
        this module does not reuse it.
        """
        naive_per_rank = 4 // 3
        self.assertEqual(naive_per_rank * 3, 3, "naive split covers only 3 of 4 heads")
        self.assertLess(naive_per_rank * 3, 4)

    def test_partition_is_exact_for_many_shapes(self):
        for num_heads in range(0, 17):
            for tp_size in range(1, 9):
                with self.subTest(heads=num_heads, tp=tp_size):
                    windows = [
                        local_head_window(num_heads, tp_size, r) for r in range(tp_size)
                    ]
                    covered = [h for s, e in windows for h in range(s, e)]
                    self.assertEqual(sorted(covered), list(range(num_heads)))
                    # Contiguous and non-overlapping: each window starts where
                    # the previous ended.
                    for (_, prev_end), (start, _) in zip(windows, windows[1:]):
                        self.assertEqual(start, prev_end)

    def test_more_ranks_than_heads_yields_empty_windows(self):
        """Replicated-KV layouts legitimately do this; it must not raise."""
        windows = [local_head_window(2, 4, r) for r in range(4)]
        self.assertEqual(windows, [(0, 1), (1, 2), (2, 2), (2, 2)])

    def test_invalid_inputs_are_rejected(self):
        for args in ((4, 0, 0), (4, 3, 3), (4, 3, -1), (-1, 3, 0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    local_head_window(*args)


if __name__ == "__main__":
    unittest.main()
