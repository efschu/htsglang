# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The BAR1 point-to-point seam: the algebra a p2p kernel must obey.

WHY THIS EXISTS. #732 established that barlink BAR1 carries no point-to-point
traffic, and that this — not cost — is the only remaining ground for the PP
family-placement foreclosure. The cost argument was withdrawn: 29 extra
crossings at a measured 7.30 us host ping-pong is ~0.7 % of a 30 ms round.

WHAT IS AND IS NOT BUILDABLE AT THE DESK. `put()` writes into a peer's mapped
BAR window (`barlink_bar1.py:2562`), and the three existing kernels
(`bar1_mesh_kernel`, `bar1_ring_kernel`, `bar1_a2a_kernel`) are all collectives
— there is no p2p kernel and no standalone device-side wait. A real send/recv
therefore needs either a new CUDA kernel or a host spin, and either way it needs
TWO cards with dmabuf peer mapping to exercise. That half is window-gated and is
named as such rather than faked.

What IS provable here is the part a kernel cannot get wrong later: where a
directed pair writes, how many flag lines it costs, how a payload larger than a
slot is chunked, and which asks are refused. That algebra is this module.

DISCIPLINE INHERITED FROM THE HOST TRANSPORT. `barlink_host.send/recv`
(`barlink_host.py:1100`, `:1120`) is the precedent: a per-pair address slot, a
flags address, a per-peer sequence counter, a bounded timeout, and a NAMED
refusal when p2p is disabled. This seam keeps that shape so a future BAR1
implementation is a transport swap, not a second protocol.

DISCIPLINE INHERITED FROM THE WINDOW LAYOUT. `geometry()` appends every new
region at the END and uses `-1` for "does not exist", explicitly so existing
offsets stay byte-for-byte. `flags_requirement()` allocates one 256-byte line
per (topology, step, sender) to avoid false sharing. Both are honoured; the
tests below would fail if either were broken.
"""

from __future__ import annotations

import unittest

from sglang.srt.distributed.device_communicators.barlink_bar1 import geometry
from sglang.srt.distributed.device_communicators.barlink_bar1_p2p import (
    P2P_LINE_BYTES,
    P2pUnavailable,
    capture_safety,
    check_p2p_payload,
    p2p_flags_extra,
    p2p_layout,
    p2p_plan,
    p2p_region_bytes,
    p2p_slot_index,
)
from sglang.test.test_utils import CustomTestCase


class TestDirectedSlotAlgebra(CustomTestCase):
    """One slot per DIRECTED pair: (a->b) and (b->a) must never collide."""

    def test_every_directed_pair_gets_a_distinct_slot(self):
        for world in (2, 3, 4, 8):
            with self.subTest(world=world):
                seen = {}
                for src in range(world):
                    for dst in range(world):
                        if src == dst:
                            continue
                        idx = p2p_slot_index(src, dst, world)
                        self.assertNotIn(
                            idx, seen, f"{src}->{dst} collides with {seen.get(idx)}"
                        )
                        seen[idx] = (src, dst)
                self.assertEqual(len(seen), world * (world - 1))

    def test_the_two_directions_differ(self):
        self.assertNotEqual(p2p_slot_index(0, 1, 3), p2p_slot_index(1, 0, 3))

    def test_slots_are_dense_from_zero(self):
        world = 4
        idx = sorted(
            p2p_slot_index(s, d, world)
            for s in range(world)
            for d in range(world)
            if s != d
        )
        self.assertEqual(idx, list(range(world * (world - 1))))

    def test_a_self_send_is_refused(self):
        with self.assertRaises(P2pUnavailable):
            p2p_slot_index(2, 2, 4)

    def test_a_rank_outside_the_world_is_refused(self):
        with self.assertRaises(P2pUnavailable):
            p2p_slot_index(0, 9, 4)


class TestFlagLines(CustomTestCase):
    """One 256-byte line per directed pair -- no false sharing between senders."""

    def test_one_line_per_directed_pair(self):
        for world in (2, 3, 8):
            with self.subTest(world=world):
                self.assertEqual(
                    p2p_flags_extra(world), world * (world - 1) * P2P_LINE_BYTES
                )

    def test_the_line_size_matches_the_modules_own_discipline(self):
        self.assertEqual(P2P_LINE_BYTES, 256)

    def test_a_degenerate_world_costs_nothing(self):
        self.assertEqual(p2p_flags_extra(1), 0)


class TestLayoutIsAppendOnly(CustomTestCase):
    """geometry() keeps existing offsets byte-for-byte; so must this."""

    def _base(self):
        return geometry(3, 1 << 20)

    def test_the_p2p_region_starts_after_every_existing_region(self):
        base = self._base()
        laid = p2p_layout(base, world=3, slot_bytes=4096)
        existing_end = max(
            base["region_bytes"],
            base["off_ring"],
            base["off_a2a"] if base["off_a2a"] >= 0 else 0,
        )
        self.assertGreaterEqual(laid["off_p2p"], existing_end)

    def test_no_existing_offset_moves(self):
        base = self._base()
        laid = p2p_layout(base, world=3, slot_bytes=4096)
        for key in ("off_mesh", "off_ring", "off_a2a", "chunk_max", "max_bytes"):
            self.assertEqual(laid[key], base[key], key)

    def test_absent_means_minus_one_not_zero(self):
        base = self._base()
        laid = p2p_layout(base, world=3, slot_bytes=0)
        self.assertEqual(laid["off_p2p"], -1, "0 would be the mesh region")

    def test_the_region_holds_every_directed_slot(self):
        laid = p2p_layout(self._base(), world=4, slot_bytes=4096)
        self.assertEqual(laid["p2p_region_bytes"], 4 * 3 * 4096)


class TestChunkPlan(CustomTestCase):
    """A payload larger than one slot is chunked; put() refuses to re-map."""

    def test_a_fitting_payload_is_one_chunk(self):
        self.assertEqual(p2p_plan(1000, 4096), [(0, 1000)])

    def test_an_oversized_payload_is_split_at_the_slot(self):
        self.assertEqual(p2p_plan(10000, 4096), [(0, 4096), (4096, 4096), (8192, 1808)])

    def test_the_chunks_cover_exactly_the_payload(self):
        for nbytes in (1, 4095, 4096, 4097, 65536):
            with self.subTest(nbytes=nbytes):
                plan = p2p_plan(nbytes, 4096)
                self.assertEqual(sum(n for _o, n in plan), nbytes)
                self.assertEqual(plan[0][0], 0)

    def test_an_empty_payload_plans_nothing(self):
        self.assertEqual(p2p_plan(0, 4096), [])

    def test_a_zero_slot_is_refused(self):
        with self.assertRaises(P2pUnavailable):
            p2p_plan(10, 0)


class TestRefusals(CustomTestCase):
    """The failure this replaces is a silent wrong write, so every ask refuses."""

    def test_a_payload_over_the_window_is_refused_by_name(self):
        with self.assertRaises(P2pUnavailable) as caught:
            check_p2p_payload(nbytes=1 << 24, slot_bytes=4096, world=3, window_bytes=8192)
        message = str(caught.exception)
        self.assertIn("window", message.lower())
        # The refusal must carry the arithmetic, like put()'s does.
        self.assertIn("8192", message)

    def test_a_world_below_two_is_refused(self):
        with self.assertRaises(P2pUnavailable):
            check_p2p_payload(nbytes=16, slot_bytes=4096, world=1, window_bytes=1 << 20)

    def test_a_fitting_ask_passes(self):
        self.assertIsNone(
            check_p2p_payload(
                nbytes=10240, slot_bytes=4096, world=3, window_bytes=1 << 20
            )
        )


class TestCaptureSafetyIsStatedNotAssumed(CustomTestCase):
    """The PP crossing sits in the decode path, so this must be explicit."""

    def test_the_send_half_is_capturable(self):
        verdict = capture_safety()
        self.assertTrue(verdict["send_capturable"])

    def test_the_recv_half_is_not_capturable_today(self):
        verdict = capture_safety()
        self.assertFalse(verdict["recv_capturable"])

    def test_it_names_why_and_what_would_change_it(self):
        verdict = capture_safety()
        self.assertIn("device-side", verdict["reason"].lower())
        self.assertTrue(verdict["breakable_required"])


if __name__ == "__main__":
    unittest.main()


class TestByteCorrectnessOverASimulatedWindow(CustomTestCase):
    """CONTENT verified, not just delivered.

    The failure a p2p geometry can have is aliasing: two directed pairs whose
    slots overlap deliver successfully and corrupt each other. A delivery check
    would pass; only reading the bytes back catches it. This drives the real
    slot algebra over a bytearray standing in for the mapped window, so the
    proof is of the arithmetic a kernel will use, not of a mock.
    """

    def _write_all(self, world, slot_bytes, payload):
        window = bytearray(p2p_region_bytes(world, slot_bytes))
        expected = {}
        for src in range(world):
            for dst in range(world):
                if src == dst:
                    continue
                # A pattern unique to the DIRECTED pair, so a swap is visible.
                pattern = bytes([(src * 31 + dst * 7 + i) & 0xFF for i in range(payload)])
                base = p2p_slot_index(src, dst, world) * slot_bytes
                for off, length in p2p_plan(payload, slot_bytes):
                    window[base + off : base + off + length] = pattern[off : off + length]
                expected[(src, dst)] = pattern
        return window, expected

    def test_no_directed_pair_corrupts_another(self):
        for world in (2, 3, 4):
            with self.subTest(world=world):
                slot_bytes, payload = 512, 300
                window, expected = self._write_all(world, slot_bytes, payload)
                for (src, dst), pattern in expected.items():
                    base = p2p_slot_index(src, dst, world) * slot_bytes
                    got = bytes(window[base : base + payload])
                    self.assertEqual(
                        got, pattern, f"{src}->{dst} was overwritten by another pair"
                    )

    def test_a_full_slot_payload_still_does_not_bleed(self):
        world, slot_bytes = 3, 256
        window, expected = self._write_all(world, slot_bytes, slot_bytes)
        for (src, dst), pattern in expected.items():
            base = p2p_slot_index(src, dst, world) * slot_bytes
            self.assertEqual(bytes(window[base : base + slot_bytes]), pattern)

    def test_the_falsifier_can_fail(self):
        """Guard the guard: a deliberately aliasing index must be caught."""
        world, slot_bytes, payload = 3, 512, 300
        window = bytearray(p2p_region_bytes(world, slot_bytes))
        # Collapse the direction: a->b and b->a share a slot.
        def bad_index(src, dst, w):
            lo, hi = min(src, dst), max(src, dst)
            return lo * (w - 1) + (hi - 1)

        written = {}
        for src in range(world):
            for dst in range(world):
                if src == dst:
                    continue
                pattern = bytes([(src * 31 + dst * 7 + i) & 0xFF for i in range(payload)])
                base = bad_index(src, dst, world) * slot_bytes
                window[base : base + payload] = pattern
                written[(src, dst)] = pattern
        corrupted = [
            pair
            for pair, pattern in written.items()
            if bytes(
                window[
                    bad_index(*pair, world) * slot_bytes : bad_index(*pair, world)
                    * slot_bytes
                    + payload
                ]
            )
            != pattern
        ]
        self.assertTrue(
            corrupted, "the aliasing check itself is vacuous if nothing corrupts"
        )
