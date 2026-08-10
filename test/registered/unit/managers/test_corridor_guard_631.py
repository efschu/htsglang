# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656 spec item 15a/15b: the spill-before-alloc gate.

WHAT THESE PIN, and why each is a test rather than a comment:

* **The check is against the allocation, not the current state.** A guard
  that only asks "is free above the floor right now" passes immediately
  before an allocation that will breach. The corridor law is a CONTINUOUS
  minimum, so the only useful question is `free - want >= floor`.

* **Two watermarks.** Freeing exactly to the floor guarantees the next
  allocation spills again. `test_frees_past_the_floor_by_delta` pins that the
  gate frees to `floor + delta`, and `test_a_second_ask_does_not_respill`
  pins the consequence, which is the actual point.

* **Cheapest first.** The spend order is the reclaim-ordering law. A gate
  that reached for kvso before the idle drafter would pay latency it did not
  need to.

* **Refusal is a refusal.** When providers are exhausted the gate must say
  no. A gate that let the allocation proceed would launder a corridor breach
  as a check that passed -- and this chain has lost an instance to exactly
  the allocation this gate exists to guard.

* **Freed bytes are re-probed, never trusted.** A provider that hands memory
  back to torch's cache rather than the driver has freed nothing the
  corridor law can see, and the guard must notice.

Hermetic: an injected free-memory probe, no CUDA.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import corridor_guard as cg

MIB = 1024 * 1024


class _Card:
    """A fake device whose free memory only moves when a provider frees."""

    def __init__(self, free_mib: int):
        self.free = free_mib * MIB

    def probe(self) -> int:
        return self.free

    def provider(self, pool_mib: int, *, to_driver: bool = True):
        """A payload of `pool_mib` that can be given back once."""
        state = {"left": pool_mib * MIB}

        def free_up_to(nbytes: int) -> int:
            take = min(state["left"], max(0, int(nbytes)))
            state["left"] -= take
            if to_driver:
                self.free += take
            # A provider that returns bytes to torch's cache instead of the
            # driver still REPORTS them; the guard must not believe it.
            return take

        return free_up_to


def _guard(card, floor=1024, delta=256):
    return cg.CorridorGuard(
        0, floor_mib=floor, delta_mib=delta, probe=card.probe
    )


class ChecksAgainstTheAllocationTest(unittest.TestCase):
    def test_no_reclaim_when_the_allocation_already_fits(self):
        card = _Card(4096)
        g = _guard(card)
        r = g.ensure_headroom(1000 * MIB)
        self.assertTrue(r.ok)
        self.assertEqual(r.reclaimed, 0)
        self.assertEqual(g.arm_count, 0)

    def test_arms_when_the_allocation_would_breach_even_though_free_is_high(self):
        # 3000 MiB free is far above the 1024 floor, but a 2500 MiB
        # allocation lands at 500. A threshold observer sees nothing wrong
        # until after the fact; this is the whole reason the gate exists.
        card = _Card(3000)
        g = _guard(card)
        g.register("draft", 10, card.provider(600))
        r = g.ensure_headroom(2500 * MIB)
        self.assertEqual(g.arm_count, 1)
        self.assertTrue(r.ok)
        self.assertGreaterEqual(card.free - 2500 * MIB, 1024 * MIB)


class TwoWatermarksTest(unittest.TestCase):
    def test_frees_past_the_floor_by_delta(self):
        card = _Card(1200)
        g = _guard(card, floor=1024, delta=256)
        g.register("draft", 10, card.provider(2000))
        r = g.ensure_headroom(400 * MIB)
        self.assertTrue(r.ok)
        # floor + delta + want = 1024 + 256 + 400 = 1680
        self.assertGreaterEqual(card.free, 1680 * MIB)

    def test_a_second_ask_does_not_respill(self):
        # The point of the upper watermark: the next allocation is free.
        card = _Card(1200)
        g = _guard(card, floor=1024, delta=256)
        g.register("draft", 10, card.provider(2000))
        g.ensure_headroom(400 * MIB)
        armed_after_first = g.arm_count
        g.ensure_headroom(200 * MIB)
        self.assertEqual(g.arm_count, armed_after_first)


class SpendOrderTest(unittest.TestCase):
    def test_cheapest_provider_first(self):
        card = _Card(1100)
        g = _guard(card)
        order = []

        def mk(name, pool_mib):
            inner = card.provider(pool_mib)

            def f(n):
                order.append(name)
                return inner(n)

            return f

        g.register("kvso", 90, mk("kvso", 4000))
        g.register("draft", 10, mk("draft", 4000))
        g.register("gdn", 50, mk("gdn", 4000))
        g.ensure_headroom(500 * MIB)
        self.assertEqual(order[0], "draft")
        self.assertNotIn("kvso", order)

    def test_expensive_provider_is_reached_when_cheap_ones_are_exhausted(self):
        # Item 15c: when everything cheap is gone, kvso keeps computing over
        # the host tier. The price is tempo -- never a corridor breach.
        card = _Card(1100)
        g = _guard(card)
        g.register("draft", 10, card.provider(100))
        g.register("kvso", 90, card.provider(4000))
        r = g.ensure_headroom(900 * MIB)
        self.assertTrue(r.ok)
        self.assertIn("kvso", r.used_providers)


class RefusalTest(unittest.TestCase):
    def test_refuses_when_providers_cannot_cover_it(self):
        card = _Card(1100)
        g = _guard(card)
        g.register("draft", 10, card.provider(50))
        r = g.ensure_headroom(4000 * MIB)
        self.assertFalse(r.ok)
        self.assertEqual(g.refuse_count, 1)

    def test_refusal_can_raise_for_callers_that_want_it(self):
        card = _Card(1100)
        g = _guard(card)
        with self.assertRaises(cg.CorridorBreachRefused):
            g.ensure_headroom(4000 * MIB, raise_on_refusal=True)

    def test_no_providers_at_all_refuses_rather_than_passing(self):
        card = _Card(1100)
        g = _guard(card)
        self.assertFalse(g.ensure_headroom(4000 * MIB).ok)


class HonestAccountingTest(unittest.TestCase):
    def test_a_provider_that_frees_to_torchs_cache_does_not_count(self):
        # It REPORTS bytes but the driver's free column does not move, so the
        # guard must still refuse. This is the failure this chain has already
        # made once: crediting a release that NVML could not see.
        card = _Card(1100)
        g = _guard(card)
        g.register("liar", 10, card.provider(8000, to_driver=False))
        r = g.ensure_headroom(4000 * MIB)
        self.assertFalse(r.ok)

    def test_a_raising_provider_does_not_take_the_allocation_down(self):
        card = _Card(1100)
        g = _guard(card)

        def boom(_n):
            raise RuntimeError("provider exploded")

        g.register("boom", 10, boom)
        g.register("draft", 20, card.provider(4000))
        r = g.ensure_headroom(900 * MIB)
        self.assertTrue(r.ok)
        self.assertIn("draft", r.used_providers)

    def test_rejects_a_non_callable_provider_at_registration(self):
        g = _guard(_Card(4096))
        with self.assertRaises(TypeError):
            g.register("nope", 1, "not a function")


class DraftCarrierProviderTest(unittest.TestCase):
    class _Carrier:
        def __init__(self):
            self.spilled = False

        def spill(self):
            if self.spilled:
                return 0.0
            self.spilled = True
            return 285.5

    def test_adapts_the_carrier_and_is_idempotent(self):
        c = self._Carrier()
        f = cg.draft_carrier_provider(c)
        self.assertAlmostEqual(f(10 * MIB) / MIB, 285.5, places=3)
        self.assertTrue(c.spilled)
        self.assertEqual(f(10 * MIB), 0)

    def test_none_carrier_is_free(self):
        self.assertEqual(cg.draft_carrier_provider(None)(MIB), 0)


if __name__ == "__main__":
    unittest.main()
