"""The weighted-DCP READ side, pinned against an independent numpy reference.

``test_dcp_weighted_owner_rule.py`` pins the WRITE side (inverse, partition,
collision, compactness). This file pins the pieces the *readers* use --
``dcp_weighted_read_slots`` and ``dcp_weighted_owned_lengths`` -- because
those are what a backend has to reproduce to build a paged ``kv_indptr`` /
``kv_indices`` over its owned slots. Both were extracted out of
``build_dcp_weighted_kv_indices`` for exactly this reason: everything in that
function except one ``create_flashinfer_kv_indices_triton`` launch (which only
materialises ``req_to_token[req, start:start+len]``) is pure integer tensor
math, so the part that decides whether uneven DCP is CORRECT is testable on
CPU with no device and no collectives.

The reference below is written from the RULE as prose, not transcribed from
the implementation:

    rank r owns global slot L  <=>  (L mod S) in [prefix[r], prefix[r+1])
    it stores L at row          (L div S) * ratio_r + (L mod S - prefix[r])

so a shared bug cannot hide in both.
"""

import unittest

import numpy as np
import torch

from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.layers.dcp.owner import (
    dcp_weighted_owned_lengths,
    dcp_weighted_owner_bounds,
    dcp_weighted_read_slots,
    dcp_weighted_write_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# [1,1,1] is deliberate: the weighted rule must degenerate to even modulo.
PLANS = ([2, 1, 1], [3, 2, 1], [4, 2, 2], [1, 1, 1], [5, 3], [13, 30, 21])


def _ref_owner(locs: np.ndarray, plan, rank):
    """Independent numpy reference for (owned, compact), from the rule as prose."""
    prefix = np.concatenate([[0], np.cumsum(plan)])
    S = int(prefix[-1])
    lo, hi = int(prefix[rank]), int(prefix[rank + 1])
    ratio = hi - lo
    off = locs % S
    owned = (off >= lo) & (off < hi)
    compact = (locs // S) * ratio + (off - lo)
    return owned, compact


class TestDcpWeightedIndexMath(CustomTestCase):
    def setUp(self):
        self._saved = get_cp_token_ratios()

    def tearDown(self):
        set_cp_token_ratios(self._saved)

    def _bounds(self, plan, rank):
        set_cp_token_ratios(plan)
        return dcp_weighted_owner_bounds(len(plan), rank)

    # ------------------------------------------------------------ read side

    def test_read_slots_match_the_numpy_reference(self):
        locs_np = np.arange(1000, dtype=np.int64)
        locs = torch.from_numpy(locs_np)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    cp_S, cp_lo, cp_hi, cp_ratio = self._bounds(plan, rank)
                    compact, owned = dcp_weighted_read_slots(
                        locs, cp_S, cp_lo, cp_hi, cp_ratio
                    )
                    ref_owned, ref_compact = _ref_owner(locs_np, plan, rank)
                    np.testing.assert_array_equal(owned.numpy(), ref_owned)
                    # unowned entries are documented as meaningless; compare
                    # only where the rule says the row is real.
                    np.testing.assert_array_equal(
                        compact.numpy()[ref_owned], ref_compact[ref_owned]
                    )

    def test_read_and_write_sides_agree_slot_for_slot(self):
        """The single property that makes the cache coherent: a token is read
        from the row it was written to. Read and write are two call sites of
        the same expression -- this is what stops them drifting apart."""
        locs = torch.arange(1000, dtype=torch.int64)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    cp_S, cp_lo, cp_hi, cp_ratio = self._bounds(plan, rank)
                    w_loc, w_mask = dcp_weighted_write_slots(
                        locs, cp_S, cp_lo, cp_hi, cp_ratio
                    )
                    r_loc, r_mask = dcp_weighted_read_slots(
                        locs, cp_S, cp_lo, cp_hi, cp_ratio
                    )
                    self.assertTrue(torch.equal(w_mask, r_mask))
                    self.assertTrue(
                        torch.equal(
                            w_loc[w_mask].to(torch.int64), r_loc[r_mask].to(torch.int64)
                        )
                    )

    def test_read_slots_accept_an_int32_slot_list(self):
        """req_to_token rows arrive as int32 from the kv-index kernel; the
        modulo must not be done in int32-overflow territory or on the wrong
        dtype silently."""
        locs32 = torch.arange(500, dtype=torch.int32)
        cp_S, cp_lo, cp_hi, cp_ratio = self._bounds([13, 30, 21], 1)
        compact, owned = dcp_weighted_read_slots(locs32, cp_S, cp_lo, cp_hi, cp_ratio)
        self.assertEqual(compact.dtype, torch.int32)
        ref_owned, ref_compact = _ref_owner(
            np.arange(500, dtype=np.int64), [13, 30, 21], 1
        )
        np.testing.assert_array_equal(owned.numpy(), ref_owned)
        np.testing.assert_array_equal(compact.numpy()[ref_owned], ref_compact[ref_owned])

    # -------------------------------------------------------- owned lengths

    def test_owned_lengths_are_a_segmented_sum(self):
        """kv_indptr is built from these; an off-by-one request boundary makes
        one request read another's rows."""
        rng = np.random.default_rng(1731)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    lens_np = rng.integers(0, 40, size=7).astype(np.int64)
                    locs_np = rng.integers(0, 4000, size=int(lens_np.sum())).astype(
                        np.int64
                    )
                    cp_S, cp_lo, cp_hi, cp_ratio = self._bounds(plan, rank)
                    _, owned = dcp_weighted_read_slots(
                        torch.from_numpy(locs_np), cp_S, cp_lo, cp_hi, cp_ratio
                    )
                    got = dcp_weighted_owned_lengths(
                        owned, torch.from_numpy(lens_np)
                    ).numpy()
                    ref_owned, _ = _ref_owner(locs_np, plan, rank)
                    bounds = np.concatenate([[0], np.cumsum(lens_np)])
                    ref = np.array(
                        [
                            int(ref_owned[bounds[i] : bounds[i + 1]].sum())
                            for i in range(len(lens_np))
                        ],
                        dtype=np.int64,
                    )
                    np.testing.assert_array_equal(got, ref)

    def test_owned_lengths_on_an_all_empty_batch(self):
        """Every request has zero context (first prefill chunk). Must be zeros,
        not a crash on the empty repeat_interleave."""
        lens = torch.zeros(5, dtype=torch.int64)
        owned = torch.zeros(0, dtype=torch.bool)
        got = dcp_weighted_owned_lengths(owned, lens)
        self.assertEqual(got.tolist(), [0, 0, 0, 0, 0])

    # ------------------------------------------------------- group property

    def test_the_group_reconstructs_every_slot_exactly_once(self):
        """Across the whole DCP group the owned slices must partition the
        request's token list, in order -- otherwise a rank's attention silently
        misses (or double-counts) part of the context."""
        rng = np.random.default_rng(173)
        for plan in PLANS:
            with self.subTest(plan=plan):
                locs_np = rng.permutation(2000)[:600].astype(np.int64)
                locs = torch.from_numpy(locs_np)
                seen = np.zeros(len(locs_np), dtype=np.int64)
                for rank in range(len(plan)):
                    cp_S, cp_lo, cp_hi, cp_ratio = self._bounds(plan, rank)
                    _, owned = dcp_weighted_read_slots(
                        locs, cp_S, cp_lo, cp_hi, cp_ratio
                    )
                    seen += owned.numpy().astype(np.int64)
                np.testing.assert_array_equal(seen, np.ones(len(locs_np), dtype=np.int64))

    def test_owned_rows_of_one_rank_never_collide(self):
        """Two distinct global slots owned by the same rank must map to two
        distinct physical rows, or one token overwrites the other."""
        locs = torch.arange(3000, dtype=torch.int64)
        for plan in PLANS:
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    cp_S, cp_lo, cp_hi, cp_ratio = self._bounds(plan, rank)
                    compact, owned = dcp_weighted_read_slots(
                        locs, cp_S, cp_lo, cp_hi, cp_ratio
                    )
                    rows = compact[owned].to(torch.int64)
                    self.assertEqual(rows.unique().numel(), rows.numel())

    def test_a_uniform_plan_reproduces_the_even_modulo_read(self):
        """The even rule (own L iff L % N == rank, stored at L // N) is the
        all-ones case; if this ever diverged, switching a config from an even
        to a uniform-weighted vector would silently move every row."""
        locs = torch.arange(600, dtype=torch.int64)
        dcp = 3
        for rank in range(dcp):
            cp_S, cp_lo, cp_hi, cp_ratio = self._bounds([1] * dcp, rank)
            compact, owned = dcp_weighted_read_slots(locs, cp_S, cp_lo, cp_hi, cp_ratio)
            self.assertTrue(torch.equal(owned, (locs % dcp) == rank))
            self.assertTrue(
                torch.equal(compact[owned].to(torch.int64), (locs // dcp)[owned])
            )


if __name__ == "__main__":
    unittest.main()
