# SPDX-License-Identifier: Apache-2.0
"""#631 phase flip (PP<->TP layout) plan arithmetic: hermetic contract tests.

CPU-only, no torch.distributed, no GPU. The load-bearing gates:

* layer-map validation -- holes, duplicates and out-of-range ordinals are
  refused loudly (with the can-fail proof that the checker checks);
* coverage -- every (layer, slot) cell is handled exactly once on the
  sending layout and exactly once on the receiving layout, and the
  sender/receiver lists of every pair agree cell-for-cell;
* byte identity -- a reference executor that builds each pair payload
  from the SENDER's plan and consumes it with the RECEIVER's plan (the
  shared enumeration convention: layers ascending, slots ascending, one
  row list per pair) reassembles the TP layout byte-identically from the
  PP layout, and the reverse flip round-trips byte-identically;
* falsifier -- the same harness with a deliberately broken layer map on
  one receiver goes RED, proving the byte-identity gate can fail.
"""

import itertools
import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import (
    PP_TO_TP,
    TP_TO_PP,
    build_phase_flip_transition,
    default_wave_count,
    layer_waves,
    ordered_layer_waves,
    restore_first_wave_count,
    seam_transient_peaks,
    validate_layer_map,
)
from sglang.srt.layers.dcp.reshard_plan import KvReshardError, owner_of, rows_of
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# A #625-shaped small world: 16 full-attn ordinals split 8/4/4, uneven
# token vector. Plus jagged and degenerate variants.
MAP_625 = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
VEC_UNEVEN = (3, 2, 2)
MAP_JAGGED = ((5, 0, 9), (1, 2, 3, 4, 6), (7, 8,))
N_JAGGED = 10
MAP_EMPTY_STAGE = ((0, 1, 2, 3), (), (4, 5))
N_EMPTY = 6


def _slots(n, hi, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.unique(torch.randint(0, hi, (n,), generator=g, dtype=torch.int64))


class TestLayerMapValidation(CustomTestCase):
    def test_partition_accepted_and_normalized(self):
        norm = validate_layer_map([[0, 1], [2], [3]], 4)
        self.assertEqual(norm, ((0, 1), (2,), (3,)))

    def test_empty_stage_is_legal(self):
        validate_layer_map(MAP_EMPTY_STAGE, N_EMPTY)

    def test_duplicate_refused(self):
        with self.assertRaisesRegex(KvReshardError, "both stage"):
            validate_layer_map([[0, 1], [1, 2], [3]], 4)

    def test_hole_refused(self):
        with self.assertRaisesRegex(KvReshardError, "no stage for"):
            validate_layer_map([[0], [2], [3]], 4)

    def test_out_of_range_refused(self):
        with self.assertRaisesRegex(KvReshardError, "outside"):
            validate_layer_map([[0, 4], [1, 2], [3]], 4)


class TestTransitionArithmetic(CustomTestCase):
    def _all(self, slots, layer_map, n_layers, vec, direction):
        return [
            build_phase_flip_transition(
                slots, layer_map, n_layers, vec, r, direction
            )
            for r in range(len(vec))
        ]

    def _assert_coverage(self, trs, slots, layer_map, n_layers, vec):
        """Every (layer, slot) cell exactly once per layout side."""
        all_cells = {
            (f, int(s)) for f in range(n_layers) for s in slots.tolist()
        }
        for side in ("send", "recv"):
            seen = {}
            for tr in trs:
                # local cells count on both sides.
                local_slot_ids = tr.local_pp_rows.tolist()
                for f in tr.local_layers:
                    for s in local_slot_ids:
                        key = (f, s)
                        self.assertNotIn(key, seen, f"{side}: cell {key} twice")
                        seen[key] = tr.rank
                layers_map = tr.send_layers if side == "send" else tr.recv_layers
                rows_map = tr.send_rows if side == "send" else tr.recv_rows
                for peer, layers in layers_map.items():
                    # Rows on the PP side are slot ids; on the TP side they
                    # are compact rows -- recover slot ids for identity.
                    rows = rows_map[peer]
                    if (side == "send") == (tr.direction == PP_TO_TP):
                        slot_ids = rows.tolist()  # PP side: rows ARE slots
                    else:
                        owner = owner_of(slots, vec)
                        mine = slots[owner == tr.rank]
                        self.assertTrue(
                            torch.equal(rows, rows_of(mine, vec, tr.rank))
                        )
                        slot_ids = mine.tolist()
                    for f in layers:
                        for s in slot_ids:
                            key = (f, s)
                            self.assertNotIn(
                                key, seen, f"{side}: cell {key} twice"
                            )
                            seen[key] = tr.rank
            self.assertEqual(set(seen), all_cells, f"{side}: coverage hole")

    def test_coverage_exactly_once_625_shape(self):
        slots = _slots(300, 900, seed=1)
        for direction in (PP_TO_TP, TP_TO_PP):
            trs = self._all(slots, MAP_625, 16, VEC_UNEVEN, direction)
            self._assert_coverage(trs, slots, MAP_625, 16, VEC_UNEVEN)

    def test_coverage_jagged_map(self):
        slots = _slots(120, 400, seed=2)
        trs = self._all(slots, MAP_JAGGED, N_JAGGED, VEC_UNEVEN, PP_TO_TP)
        self._assert_coverage(trs, slots, MAP_JAGGED, N_JAGGED, VEC_UNEVEN)

    def test_pairwise_agreement(self):
        slots = _slots(200, 700, seed=3)
        for direction in (PP_TO_TP, TP_TO_PP):
            trs = self._all(slots, MAP_625, 16, VEC_UNEVEN, direction)
            for a in trs:
                for peer, layers in a.send_layers.items():
                    b = trs[peer]
                    self.assertEqual(layers, b.recv_layers[a.rank])
                    self.assertEqual(
                        int(a.send_rows[peer].numel()),
                        int(b.recv_rows[a.rank].numel()),
                    )

    def test_empty_live_set(self):
        slots = torch.empty(0, dtype=torch.int64)
        tr = build_phase_flip_transition(
            slots, MAP_625, 16, VEC_UNEVEN, 0, PP_TO_TP
        )
        self.assertEqual(tr.outgoing_cells, 0)
        self.assertEqual(tr.incoming_cells, 0)
        self.assertEqual(tr.max_pp_row(), -1)
        self.assertEqual(tr.max_tp_row(), -1)

    def test_unsorted_slots_refused(self):
        bad = torch.tensor([5, 3, 9], dtype=torch.int64)
        with self.assertRaisesRegex(KvReshardError, "sorted ascending"):
            build_phase_flip_transition(bad, MAP_625, 16, VEC_UNEVEN, 0, PP_TO_TP)

    def test_negative_slot_refused(self):
        bad = torch.tensor([-2, 3, 9], dtype=torch.int64)
        with self.assertRaisesRegex(KvReshardError, "negative"):
            build_phase_flip_transition(bad, MAP_625, 16, VEC_UNEVEN, 0, PP_TO_TP)

    def test_bad_direction_refused(self):
        with self.assertRaisesRegex(KvReshardError, "direction"):
            build_phase_flip_transition(
                _slots(10, 50), MAP_625, 16, VEC_UNEVEN, 0, "sideways"
            )

    def test_stage_count_vector_mismatch_refused(self):
        with self.assertRaisesRegex(KvReshardError, "must match"):
            build_phase_flip_transition(
                _slots(10, 50), MAP_625, 16, (3, 2), 0, PP_TO_TP
            )


# ---------------------------------------------------------------------------
# Reference executor: byte identity through the shared payload convention.
# ---------------------------------------------------------------------------

ROW_BYTES = 24  # arbitrary; same for K-analog cells in every mock layer


def _ref_bytes(layer, slot, seed=11):
    g = torch.Generator().manual_seed(seed * 1_000_003 + layer * 4093 + slot)
    return torch.randint(0, 256, (ROW_BYTES,), generator=g, dtype=torch.uint8)


def _make_pp_pools(layer_map, slots, n_rows):
    """rank -> {ordinal: [n_rows, ROW_BYTES] uint8}, filled from ref."""
    pools = []
    for layers in layer_map:
        pool = {f: torch.zeros(n_rows, ROW_BYTES, dtype=torch.uint8) for f in layers}
        for f in layers:
            for s in slots.tolist():
                pool[f][s] = _ref_bytes(f, s)
        pools.append(pool)
    return pools


def _make_tp_pools(n_ranks, n_layers, n_rows):
    return [
        {f: torch.zeros(n_rows, ROW_BYTES, dtype=torch.uint8) for f in range(n_layers)}
        for _ in range(n_ranks)
    ]


def _apply_flip(trs, src_pools, dst_pools, src_side_of):
    """Reference executor: local copies, then pair payloads built strictly
    from the SENDER's plan and consumed strictly from the RECEIVER's plan
    (layers ascending, one row list per pair). ``src_side_of(tr)`` names
    which of the rank's row tensors index the source pool."""
    # local moves
    for tr in trs:
        src, dst = src_pools[tr.rank], dst_pools[tr.rank]
        s_rows, d_rows = src_side_of(tr)
        for f in tr.local_layers:
            dst[f][d_rows] = src[f][s_rows]
    # pair payloads
    for tr in trs:
        for peer in sorted(tr.send_layers):
            sender_pool = src_pools[tr.rank]
            payload = torch.cat(
                [
                    sender_pool[f][tr.send_rows[peer]]
                    for f in sorted(tr.send_layers[peer])
                ],
                dim=0,
            )
            rx = trs[peer]
            rx_layers = sorted(rx.recv_layers[tr.rank])
            rx_rows = rx.recv_rows[tr.rank]
            n = int(rx_rows.numel())
            parts = payload.split(n, dim=0)
            assert len(parts) == len(rx_layers)
            for f, part in zip(rx_layers, parts):
                dst_pools[peer][f][rx_rows] = part


class TestByteIdentity(CustomTestCase):
    def _run(self, layer_map, n_layers, vec, seed, poison_receiver_map=None):
        slots = _slots(160, 500, seed=seed)
        n_pp_rows = 500
        owner = owner_of(slots, vec)
        n_tp_rows = 0
        for r in range(len(vec)):
            rr = rows_of(slots[owner == r], vec, r)
            if rr.numel():
                n_tp_rows = max(n_tp_rows, int(rr.max().item()) + 1)
        pp = _make_pp_pools(layer_map, slots, n_pp_rows)
        tp = _make_tp_pools(len(vec), n_layers, max(n_tp_rows, 1))

        trs = [
            build_phase_flip_transition(
                slots, layer_map, n_layers, vec, r, PP_TO_TP
            )
            for r in range(len(vec))
        ]
        if poison_receiver_map is not None:
            # Falsifier: ONE rank believes a different (still valid) layer
            # map -- its recv layer labels shift, bytes land on the wrong
            # layers, and the identity check below must go red.
            r = 1
            trs[r] = build_phase_flip_transition(
                slots, poison_receiver_map, n_layers, vec, r, PP_TO_TP
            )
        try:
            _apply_flip(
                trs, pp, tp, lambda tr: (tr.local_pp_rows, tr.local_tp_rows)
            )
        except (AssertionError, KeyError, IndexError):
            # A convention mismatch may surface before the byte check: a
            # payload-shape assert, a reference to a layer the pool never
            # owned (KeyError), or an out-of-range row (IndexError). All
            # are detected divergence -- the runtime analog is the loud
            # checksum/size/bounds error family.
            return False, (pp, tp, trs, slots)

        # verify TP layout against the reference rule
        for r, tr in enumerate(trs):
            mine = slots[owner == r]
            rows = rows_of(mine, vec, r)
            for f in range(n_layers):
                for s, row in zip(mine.tolist(), rows.tolist()):
                    if not torch.equal(tp[r][f][row], _ref_bytes(f, s)):
                        return False, (pp, tp, trs, slots)
        return True, (pp, tp, trs, slots)

    def test_pp_to_tp_byte_identity_625_shape(self):
        ok, _ = self._run(MAP_625, 16, VEC_UNEVEN, seed=21)
        self.assertTrue(ok)

    def test_byte_identity_jagged_and_empty_stage(self):
        ok, _ = self._run(MAP_JAGGED, N_JAGGED, VEC_UNEVEN, seed=22)
        self.assertTrue(ok)
        ok, _ = self._run(MAP_EMPTY_STAGE, N_EMPTY, VEC_UNEVEN, seed=23)
        self.assertTrue(ok)

    def test_falsifier_broken_receiver_layer_map_goes_red(self):
        # Same partition SHAPE, shifted stage boundary: still a valid
        # partition (passes validate_layer_map), but disagrees with the
        # senders -- the byte-identity gate MUST fail. Proves the gate
        # can fail; guards against a future "plan metadata in consensus"
        # regression silently masking map divergence.
        shifted = ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15))
        ok, _ = self._run(MAP_625, 16, VEC_UNEVEN, seed=24, poison_receiver_map=shifted)
        self.assertFalse(ok, "broken layer map went undetected -- gate cannot fail")

    def test_roundtrip_pp_tp_pp_byte_identical(self):
        slots = _slots(140, 450, seed=25)
        vec = VEC_UNEVEN
        owner = owner_of(slots, vec)
        n_tp_rows = 1
        for r in range(len(vec)):
            rr = rows_of(slots[owner == r], vec, r)
            if rr.numel():
                n_tp_rows = max(n_tp_rows, int(rr.max().item()) + 1)
        pp = _make_pp_pools(MAP_625, slots, 450)
        orig = [{f: t.clone() for f, t in pool.items()} for pool in pp]
        tp = _make_tp_pools(len(vec), 16, n_tp_rows)

        fwd = [
            build_phase_flip_transition(slots, MAP_625, 16, vec, r, PP_TO_TP)
            for r in range(len(vec))
        ]
        _apply_flip(fwd, pp, tp, lambda tr: (tr.local_pp_rows, tr.local_tp_rows))

        # wipe PP pools, flip back
        pp2 = [
            {f: torch.zeros_like(t) for f, t in pool.items()} for pool in pp
        ]
        back = [
            build_phase_flip_transition(slots, MAP_625, 16, vec, r, TP_TO_PP)
            for r in range(len(vec))
        ]
        _apply_flip(back, tp, pp2, lambda tr: (tr.local_tp_rows, tr.local_pp_rows))

        for r, pool in enumerate(pp2):
            for f, t in pool.items():
                # only live rows round-trip; compare on live slots
                for s in slots.tolist():
                    self.assertTrue(
                        torch.equal(t[s], orig[r][f][s]),
                        f"roundtrip mismatch rank {r} layer {f} slot {s}",
                    )


RIG_MAP = ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11), (12, 13, 14, 15))
RIG_VEC = (14, 10, 8)


class TestOrderedLayerWaves(CustomTestCase):
    """The restore-first wave planner (#631 section 2.1b, move 2 and 3)."""

    def test_the_cap_rises_from_the_smallest_stage_to_the_layer_count(self):
        self.assertEqual(default_wave_count(RIG_MAP), 4)
        self.assertEqual(restore_first_wave_count(RIG_MAP), 16)

    def test_every_layer_lands_exactly_once_in_both_directions(self):
        for direction in (PP_TO_TP, TP_TO_PP):
            for n_waves in (1, 3, 4, 16):
                waves = ordered_layer_waves(RIG_MAP, RIG_VEC, n_waves, direction)
                self.assertEqual(len(waves), n_waves)
                flat = [f for w in waves for f in w]
                self.assertEqual(
                    sorted(flat),
                    list(range(16)),
                    f"{direction} W={n_waves} does not partition the layers",
                )

    def test_the_two_directions_want_opposite_orders(self):
        """Not cosmetic: one order cannot serve both (see the peaks below)."""
        fwd = ordered_layer_waves(RIG_MAP, RIG_VEC, 16, PP_TO_TP)
        rev = ordered_layer_waves(RIG_MAP, RIG_VEC, 16, TP_TO_PP)
        self.assertNotEqual(fwd, rev)

    def test_using_one_direction_order_for_the_other_costs_real_memory(self):
        """The falsifier for making the order direction-agnostic.

        If someone 'simplifies' this by computing the order once, the
        binding peak in the wrong direction must be visibly worse -- that
        is the whole justification for threading ``direction`` through
        ``_flip_waves``.
        """
        for direction, other in ((PP_TO_TP, TP_TO_PP), (TP_TO_PP, PP_TO_TP)):
            right = [
                f for w in ordered_layer_waves(RIG_MAP, RIG_VEC, 16, direction)
                for f in w
            ]
            wrong = [
                f for w in ordered_layer_waves(RIG_MAP, RIG_VEC, 16, other)
                for f in w
            ]
            pk_right = seam_transient_peaks(right, RIG_MAP, RIG_VEC, direction)
            pk_wrong = seam_transient_peaks(wrong, RIG_MAP, RIG_VEC, direction)
            self.assertLess(
                max(pk_right[1:]),
                max(pk_wrong[1:]),
                f"{direction}: the other direction's order is no worse "
                f"({pk_right} vs {pk_wrong}), so the split is unjustified",
            )

    def test_the_order_beats_the_proportional_split_on_binding_ranks(self):
        for direction in (PP_TO_TP, TP_TO_PP):
            chosen = [
                f for w in ordered_layer_waves(RIG_MAP, RIG_VEC, 16, direction)
                for f in w
            ]
            baseline = [f for w in layer_waves(RIG_MAP, 4) for f in w]
            pk = seam_transient_peaks(chosen, RIG_MAP, RIG_VEC, direction)
            base = seam_transient_peaks(baseline, RIG_MAP, RIG_VEC, direction)
            self.assertLessEqual(max(pk[1:]), max(base[1:]))

    def test_the_chosen_order_is_exhaustively_optimal(self):
        """Brute force on a map small enough to enumerate completely.

        The planner is a dynamic program, and a DP that optimises the wrong
        recurrence still returns a confident answer. ``[3, 2, 2]`` has 210
        distinct rank sequences, so the true minimum is computable directly
        and the DP has to match it -- not merely beat the baseline.
        """
        small_map = ((0, 1, 2), (3, 4), (5, 6))
        small_vec = (3, 2, 2)
        for direction in (PP_TO_TP, TP_TO_PP):
            best = None
            for perm in set(itertools.permutations([0, 0, 0, 1, 1, 2, 2])):
                cursor = {0: 0, 1: 0, 2: 0}
                order = []
                for r in perm:
                    order.append(small_map[r][cursor[r]])
                    cursor[r] += 1
                pk = seam_transient_peaks(order, small_map, small_vec, direction)
                binding = max(pk[1:])
                if best is None or binding < best:
                    best = binding
            got = seam_transient_peaks(
                [
                    f
                    for w in ordered_layer_waves(small_map, small_vec, 7, direction)
                    for f in w
                ],
                small_map,
                small_vec,
                direction,
            )
            self.assertAlmostEqual(
                max(got[1:]),
                best,
                places=9,
                msg=f"{direction}: DP returned {max(got[1:])}, optimum is {best}",
            )

    def test_the_planner_is_a_pure_function_of_its_inputs(self):
        """Both ends of every pair derive the split independently."""
        for direction in (PP_TO_TP, TP_TO_PP):
            a = ordered_layer_waves(RIG_MAP, RIG_VEC, 16, direction)
            b = ordered_layer_waves(
                tuple(tuple(s) for s in RIG_MAP), tuple(RIG_VEC), 16, direction
            )
            self.assertEqual(a, b)

    def test_an_empty_stage_does_not_break_the_planner(self):
        """A degenerate split is legal per ``validate_layer_map``."""
        degenerate = ((0, 1, 2, 3), (), (4, 5))
        for direction in (PP_TO_TP, TP_TO_PP):
            waves = ordered_layer_waves(degenerate, (4, 0, 2), 6, direction)
            self.assertEqual(sorted(f for w in waves for f in w), list(range(6)))


if __name__ == "__main__":
    unittest.main()
