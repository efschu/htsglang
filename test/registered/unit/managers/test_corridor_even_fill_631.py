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
"""#656 spec item 16: the cards fill EVENLY, and host RAM is the last resort.

The user's order, in their terms:

    "CARDS MUST FILL EVENLY BEFORE ANY HOST SPILL. Host spill happens ONLY
     when ALL three cards are at the floor -- never while one card binds and
     the others have headroom."

and "even" is defined on FREE HEADROOM, not on bytes held: the cards are
32/20/20 GiB, so equal bytes would be permanently unequal pressure. The
objective is therefore a WATER-FILLING one over the per-card free column.

WHAT THESE PIN

* **A tier is not a cost.** Item 15's cheapest-first ordering still decides
  the order WITHIN a tier, but it must not be able to promote a host spill
  ahead of a rebalance merely because the host spill is cheaper to execute.
  Those are different questions and this file keeps them separate.

* **The host gate is a fleet predicate, not a local one.** A guard that only
  sees its own card cannot tell "everything is full" from "I am the only one
  that is full", and those two situations have opposite correct answers. The
  gate therefore reads the whole free column.

* **Refusal beats a premature host spill.** When rebalancing cannot help and
  the fleet is not yet level, the correct answer is to refuse and let the
  caller take its refusal path -- not to quietly page to RAM. That is the
  same principle as item 15's refusal: the gate never launders a policy
  breach as a check that passed.

* **The levelling objective is computable.** "Evenly filled" has to be a
  number in a CSV, so the water-filling targets are a pure function and are
  tested as one.

Hermetic: injected probes, no CUDA, no NVML.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import corridor_guard as cg

MIB = 1024 * 1024


class _Fleet:
    """Three cards whose free column moves only when a provider frees."""

    def __init__(self, free_mib):
        self.free = [m * MIB for m in free_mib]

    def probe(self, index: int):
        return lambda: self.free[index]

    def column(self):
        return list(self.free)

    def provider(self, index: int, pool_mib: int):
        state = {"left": pool_mib * MIB}

        def free_up_to(nbytes: int) -> int:
            take = min(state["left"], max(0, int(nbytes)))
            state["left"] -= take
            self.free[index] += take
            return take

        return free_up_to


def _guard(fleet, index, floor=1024, delta=256):
    return cg.CorridorGuard(
        index,
        floor_mib=floor,
        delta_mib=delta,
        probe=fleet.probe(index),
        fleet_probe=fleet.column,
    )


class HostSpillIsGatedOnTheWholeFleetTest(unittest.TestCase):
    def test_host_tier_is_refused_while_another_card_has_headroom(self):
        # Card 0 binds at 1100; the 5090 is sitting on 6 GiB. Paging KV to
        # host here is exactly what item 16 forbids -- the bytes belong on
        # card 1, not in RAM.
        fleet = _Fleet([1100, 6000, 3000])
        g = _guard(fleet, 0)
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        r = g.ensure_headroom(900 * MIB, reason="kv grow")
        self.assertFalse(r.ok)
        self.assertEqual(r.used_providers, ())
        self.assertEqual(g.host_blocked_count, 1)
        self.assertIn("not level", r.detail)

    def test_host_tier_is_admitted_once_every_card_is_at_the_floor(self):
        # Now nothing can be rebalanced anywhere: this is the ONLY state in
        # which item 16 permits host RAM.
        fleet = _Fleet([1100, 1200, 1150])
        g = _guard(fleet, 0)
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        r = g.ensure_headroom(900 * MIB, reason="kv grow")
        self.assertTrue(r.ok)
        self.assertEqual(r.used_providers, ("kvso",))
        self.assertEqual(g.host_blocked_count, 0)

    def test_a_single_peer_with_headroom_is_enough_to_block(self):
        # Two of three are at the floor. "ALL three" means all three.
        fleet = _Fleet([1100, 1150, 5000])
        g = _guard(fleet, 0)
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        self.assertFalse(g.ensure_headroom(900 * MIB).ok)

    def test_without_a_fleet_probe_the_host_tier_stays_shut(self):
        # A guard that cannot see the fleet cannot prove the fleet is level,
        # and item 16 is a permission that must be PROVEN, not assumed.
        fleet = _Fleet([1100, 1200, 1150])
        g = cg.CorridorGuard(0, probe=fleet.probe(0))
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        r = g.ensure_headroom(900 * MIB)
        self.assertFalse(r.ok)
        self.assertEqual(g.host_blocked_count, 1)


class TierBeatsCostTest(unittest.TestCase):
    def test_a_cheap_host_spill_does_not_overtake_an_expensive_rebalance(self):
        # The host spill is cheaper to EXECUTE and covers the ask on its own.
        # Item 16 still says the rebalance goes first. Cost orders within a
        # tier; it may not cross one.
        fleet = _Fleet([1100, 1150, 1150])
        g = _guard(fleet, 0)
        order = []

        def spy(name, inner):
            def f(n):
                order.append(name)
                return inner(n)

            return f

        g.register(
            "kvso", 1, spy("kvso", fleet.provider(0, 4000)), tier=cg.RELIEF_HOST
        )
        g.register(
            "rebalance",
            99,
            spy("rebalance", fleet.provider(0, 4000)),
            tier=cg.RELIEF_REBALANCE,
        )
        r = g.ensure_headroom(900 * MIB)
        self.assertTrue(r.ok)
        self.assertEqual(order[0], "rebalance")

    def test_park_sits_between_rebalance_and_host(self):
        fleet = _Fleet([1100, 1150, 1150])
        g = _guard(fleet, 0)
        order = []

        def spy(name, inner):
            def f(n):
                order.append(name)
                return inner(n)

            return f

        # Each provider is deliberately too small to satisfy the ask alone,
        # so the gate must walk the whole ladder and the order is observable.
        g.register("kvso", 1, spy("kvso", fleet.provider(0, 400)), tier=cg.RELIEF_HOST)
        g.register(
            "park", 1, spy("park", fleet.provider(0, 400)), tier=cg.RELIEF_PARK
        )
        g.register(
            "rebalance",
            1,
            spy("rebalance", fleet.provider(0, 400)),
            tier=cg.RELIEF_REBALANCE,
        )
        g.ensure_headroom(1500 * MIB)
        self.assertEqual(order, ["rebalance", "park", "kvso"])

    def test_rebalance_alone_can_settle_it_and_host_is_never_touched(self):
        fleet = _Fleet([1100, 6000, 3000])
        g = _guard(fleet, 0)
        touched = []
        g.register(
            "rebalance",
            10,
            fleet.provider(0, 2000),
            tier=cg.RELIEF_REBALANCE,
        )
        g.register(
            "kvso",
            90,
            lambda n: touched.append(n) or 0,
            tier=cg.RELIEF_HOST,
        )
        r = g.ensure_headroom(900 * MIB)
        self.assertTrue(r.ok)
        self.assertEqual(touched, [])
        self.assertEqual(r.used_providers, ("rebalance",))

    def test_rebalance_is_not_gated_on_being_below_the_mean(self):
        # A card can be above the fleet mean and still need headroom for a
        # big allocation. Levelling must stay available to it; only the HOST
        # tier is gated.
        fleet = _Fleet([5000, 1100, 1100])
        g = _guard(fleet, 0)
        g.register("rebalance", 10, fleet.provider(0, 2000), tier=cg.RELIEF_REBALANCE)
        r = g.ensure_headroom(4500 * MIB)
        self.assertTrue(r.ok)
        self.assertEqual(r.used_providers, ("rebalance",))


class DefaultTierTest(unittest.TestCase):
    def test_an_unlabelled_provider_is_a_rebalance_not_a_host_spill(self):
        # Item 15's providers predate tiers. Defaulting them to HOST would
        # switch off relief that already works; defaulting them to REBALANCE
        # keeps the previous behaviour and forces the host class to be
        # declared deliberately, which is the direction an error should fall.
        fleet = _Fleet([1100, 6000, 3000])
        g = _guard(fleet, 0)
        g.register("draft", 10, fleet.provider(0, 2000))
        r = g.ensure_headroom(900 * MIB)
        self.assertTrue(r.ok)
        self.assertEqual(r.used_providers, ("draft",))


class WaterFillingTest(unittest.TestCase):
    def test_equal_free_targets_ignore_card_size(self):
        # 32/20/20 GiB cards. The objective is equal FREE, so the targets are
        # equal to each other, not proportional to the totals.
        targets = cg.water_fill_targets([6000 * MIB, 1200 * MIB, 1800 * MIB])
        self.assertEqual(len(set(targets)), 1)
        self.assertEqual(sum(targets), 9000 * MIB)

    def test_spread_is_the_levelness_metric_booked_in_the_csv(self):
        self.assertEqual(cg.free_spread_mib([1100 * MIB, 1150 * MIB]), 50)
        self.assertEqual(cg.free_spread_mib([1100 * MIB, 6000 * MIB]), 4900)

    def test_a_level_fleet_is_level_within_the_delta(self):
        # "At the floor" is a band, not a point: 1024 exactly would make the
        # predicate unreachable in practice.
        self.assertTrue(cg.fleet_is_level([1100 * MIB, 1200 * MIB], 1024, 256))
        self.assertFalse(cg.fleet_is_level([1100 * MIB, 3000 * MIB], 1024, 256))

    def test_an_empty_column_is_not_level(self):
        # No evidence is not evidence of levelness.
        self.assertFalse(cg.fleet_is_level([], 1024, 256))


class WhenRefusingIsFatalTest(unittest.TestCase):
    """Item 16 is a preference, not a suicide pact.

    The gate normally withholds host RAM while any card still has headroom,
    because the bytes belong on that card. But on the pp->tp leg a refusal is
    not survivable: strict purity forbids decode in PP, so a refused pp->tp
    means decode never runs again, and nothing the PP phase holds can free the
    memory that would end the refusal. Measured on 2026-08-10: 411 abandons,
    0 requests in 6 minutes, /health 503, every rank alive.

    Spec item 15c already answers this in the user's own words -- when
    everything resident is hot, kvso keeps computing over the host tier, and
    "the price is tempo, NEVER a corridor breach". Refusing forever does not
    protect the corridor; the corridor is fine in that state. It kills
    serving instead. So the host tier is admitted when the alternative is a
    deadlock, and the event is counted and logged with the free column so the
    missing levelling is loud rather than silent.
    """

    def test_an_unlevel_fleet_still_gets_host_when_refusal_is_fatal(self):
        fleet = _Fleet([1100, 6000, 3000])
        g = _guard(fleet, 0)
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        r = g.ensure_headroom(900 * MIB, refusal_is_fatal=True)
        self.assertTrue(r.ok)
        self.assertEqual(r.used_providers, ("kvso",))
        self.assertEqual(g.host_forced_count, 1)
        self.assertIn("refusal would deadlock", r.detail)

    def test_the_default_still_withholds_host_on_an_unlevel_fleet(self):
        # The escape must be opt-in per call. A default-on escape would make
        # item 16 unenforceable everywhere.
        fleet = _Fleet([1100, 6000, 3000])
        g = _guard(fleet, 0)
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        self.assertFalse(g.ensure_headroom(900 * MIB).ok)
        self.assertEqual(g.host_forced_count, 0)

    def test_rebalance_is_still_spent_first_even_when_refusal_is_fatal(self):
        # The escape opens the host tier; it does not reorder the ladder.
        fleet = _Fleet([1100, 6000, 3000])
        g = _guard(fleet, 0)
        order = []

        def spy(name, inner):
            def f(n):
                order.append(name)
                return inner(n)

            return f

        g.register("kvso", 1, spy("kvso", fleet.provider(0, 4000)), tier=cg.RELIEF_HOST)
        g.register(
            "rebalance", 99, spy("rebalance", fleet.provider(0, 4000)),
            tier=cg.RELIEF_REBALANCE,
        )
        r = g.ensure_headroom(900 * MIB, refusal_is_fatal=True)
        self.assertTrue(r.ok)
        self.assertEqual(order[0], "rebalance")
        # Rebalance covered it, so the escape was never needed.
        self.assertEqual(g.host_forced_count, 0)

    def test_a_level_fleet_does_not_count_as_a_forced_host_spill(self):
        # When the fleet IS level the host tier was permitted anyway, so
        # nothing was forced and the counter must not inflate.
        fleet = _Fleet([1100, 1200, 1150])
        g = _guard(fleet, 0)
        g.register("kvso", 90, fleet.provider(0, 4000), tier=cg.RELIEF_HOST)
        r = g.ensure_headroom(900 * MIB, refusal_is_fatal=True)
        self.assertTrue(r.ok)
        self.assertEqual(g.host_forced_count, 0)


if __name__ == "__main__":
    unittest.main()
