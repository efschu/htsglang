# SPDX-License-Identifier: Apache-2.0
"""Q-650 DUAL-ANCHOR-RELEASE-D (27B dual y8v, boot dkr27bnvfp4dual1mpsleepbar1fs10031504, 15:18-15:21).

The mamba arena holds 112 anchor slots shared by P and D. Q-610 made P give its
references back; group D never did -- every D rank held 111-112 of the 112 slots
(``ARENA-REF-CENSUS complete=112``, D own_held 111-112). P's END anchor of pdflip-0-121
found no slot (``PDFLIP-PUBLISH-CHUNK stopped=mamba_full``), D refused the tail (W50),
the front re-routed through P leg 1 into the same full arena, and the second refusal
ended W53 -- every long request after 15:20:18; D decoded nothing from 15:20:53.

Guarded here:
  * D gives back its least recently used SETTLED anchors (lock 0) until P finds room,
    so P's claim succeeds (without the fix: refused).
  * D never gives back an anchor a running request holds.
  * the front's ANCHOR-OWED hold waits for that room before the second P leg, bounded.
"""

import asyncio
import os
import tempfile
import unittest

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import dual_anchor_release as DAR  # noqa: E402

SLOTS = 112


class FakeArena:
    """Slot -> reader references (all ranks); a claim needs a slot nobody references."""

    def __init__(self, slots: int):
        self.refs = [0] * slots

    @property
    def pinned(self) -> int:
        return sum(1 for r in self.refs if r > 0)

    def claim(self) -> bool:
        for i, r in enumerate(self.refs):
            if r == 0:
                self.refs[i] = 1
                return True
        return False


class Node:
    def __init__(self, slot: int, age: int, locked: bool = False):
        self.slot, self.last_access_time, self.locked = slot, age, locked
        self.released = False


def _d_tree_holds_every_slot(arena: FakeArena, locked=()):
    nodes = []
    for s in range(SLOTS):
        arena.refs[s] += 1
        nodes.append(Node(s, age=s, locked=s in locked))
    return nodes


def _d_tick(arena: FakeArena, nodes: list) -> int:
    """One Q-650 D tick against the fake arena, wired like the radix cache does."""
    held = [n for n in nodes if not n.released]

    def release(n):
        n.released = True
        arena.refs[n.slot] -= 1

    need = DAR.d_release_need(slots=SLOTS, pinned=arena.pinned, d_held=len(held))
    return DAR.d_release_pass(held, need=need, releasable=lambda n: not n.locked,
                              release=release, age=lambda n: n.last_access_time)


class TestDReleaseGivesPRoom(CustomTestCase):
    def test_y8v_full_arena_p_claim_succeeds_after_the_d_tick(self):
        arena = FakeArena(SLOTS)
        nodes = _d_tree_holds_every_slot(arena)
        self.assertFalse(FakeArena.claim(_copy(arena)), "precondition: the y8v arena refuses P's claim")
        released = _d_tick(arena, nodes)
        self.assertGreater(released, 0)
        self.assertLessEqual(len(nodes) - released, DAR.d_cap(SLOTS))
        self.assertGreaterEqual(SLOTS - arena.pinned, DAR.p_room(SLOTS))
        self.assertTrue(arena.claim(), "P's END-anchor claim must find a slot after D's tick")

    def test_lru_first_and_locked_anchors_are_kept(self):
        arena = FakeArena(SLOTS)
        locked = set(range(0, 40))   # the oldest 40 are held by running requests
        nodes = _d_tree_holds_every_slot(arena, locked=locked)
        _d_tick(arena, nodes)
        self.assertFalse(any(n.released for n in nodes if n.locked))
        freed = sorted(n.last_access_time for n in nodes if n.released)
        self.assertEqual(freed[0], 40, "the oldest UNLOCKED anchor goes first")

    def test_under_cap_with_room_releases_nothing(self):
        self.assertEqual(DAR.d_release_need(slots=SLOTS, pinned=50, d_held=50), 0)
        # P's own refs fill the arena but D is under its cap: D gives what P lacks,
        # never more than it holds
        self.assertEqual(DAR.d_release_need(slots=SLOTS, pinned=SLOTS, d_held=30), DAR.p_room(SLOTS))
        self.assertEqual(DAR.d_release_need(slots=SLOTS, pinned=SLOTS, d_held=10), 10)


class TestArmedD(CustomTestCase):
    def test_group_d_of_the_dual_layout_only(self):
        self.assertTrue(DAR.armed_d({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}))
        self.assertFalse(DAR.armed_d({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}))
        self.assertFalse(DAR.armed_d({"FLLIPER_PDFLIP_DUAL_LAYOUT": "0", "FLLIPER_PDFLIP_GROUP": "D"}))


class TestAnchorOwedHold(CustomTestCase):
    def _run(self, readings, max_s=60.0):
        clock = [0.0]
        it = iter(readings)
        last = [None]

        def read():
            last[0] = next(it, last[0])
            return last[0]

        async def sleep(s):
            clock[0] += s

        return asyncio.run(DAR.wait_anchor_room(read, sleep=sleep, clock=lambda: clock[0],
                                                max_s=max_s, poll_s=0.5))

    def test_full_arena_holds_until_room(self):
        full = {"slots": SLOTS, "pinned": SLOTS}
        room = {"slots": SLOTS, "pinned": SLOTS - DAR.p_room(SLOTS)}
        outcome, held = self._run([full, full, full, room])
        self.assertEqual(outcome, "room")
        self.assertAlmostEqual(held, 1.5)

    def test_no_reading_or_room_never_waits(self):
        self.assertEqual(self._run([None])[0], "free")
        self.assertEqual(self._run([{"slots": SLOTS, "pinned": 10}])[0], "free")

    def test_bounded(self):
        outcome, held = self._run([{"slots": SLOTS, "pinned": SLOTS}], max_s=3.0)
        self.assertEqual(outcome, "timeout")
        self.assertGreaterEqual(held, 3.0)

    def test_room_file_roundtrip_and_staleness(self):
        with tempfile.TemporaryDirectory() as d:
            DAR.publish_room(slots=SLOTS, pinned=111, d_held=110, tag="t", root=d, now=100.0)
            r = DAR.read_room(tag="t", root=d, now=101.0)
            self.assertTrue(DAR.room_blocked(r))
            self.assertIsNone(DAR.read_room(tag="t", root=d, now=100.0 + DAR.ROOM_MAX_AGE_S + 1))


class TestOncePer(CustomTestCase):
    def test_first_and_every_nth(self):
        once = DAR.OncePer(every=4)
        out = [once("k") for _ in range(9)]
        self.assertEqual(out, [True, False, False, True, False, False, False, True, False])
        self.assertEqual(once.count("k"), 9)


def _copy(arena: FakeArena) -> FakeArena:
    a = FakeArena(len(arena.refs))
    a.refs = list(arena.refs)
    return a


if __name__ == "__main__":
    unittest.main()
