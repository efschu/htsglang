# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#553 cut 2, second half: a mover that can address "tenant X".

WHAT THE ANALYSIS FOUND MISSING (§2, tenant COLD row): "works for the translator
(`ledger.park_all`) and for GDN slots (#364). **No generic per-tenant mover; no
single caller that can address 'tenant X'**." Cut 2's first half wired the two
byte probes so the bridge stops answering all-unavailable; this is the rest --
the thing a cold event can actually call once the bridge has ranked a plan.

WHY THIS IS NOT A SECOND LEDGER. `AudioAssetLedger` already parks and restores,
and its `ParkRoute` protocol (device / park / restore / size_bytes) is already
generic -- the plan says "what makes it translator-specific is only its
wake-rank vocabulary". So this does not reimplement parking. It registers
TENANTS over that same protocol and supplies the ordering as CONFIGURATION, so
a second tenant with a different vocabulary works without the translator's
need-order being hard-coded anywhere. Building a parallel asset ledger would be
the two-authorities defect this module exists downstream of.

REPORTS, NOT PLANS (#694). Bytes come from what a route RETURNED. A route that
was asked and reported nothing is carried as stranded -- never counted as zero,
never dropped -- the same distinction `cold_event` already makes for sources,
for the same reason: bytes that left one ledger and entered none go unnoticed
for weeks.

REFUSALS. An unknown tenant raises rather than returning zero: "no such tenant"
and "that tenant had nothing to give" are different answers, and a cold event
that cannot tell them apart will keep asking the wrong one.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers.tenant_mover import (
    TenantMover,
    UnknownTenant,
)
from sglang.test.test_utils import CustomTestCase


#: Distinct from None, which this stub uses for "not specified". A route that
#: REPORTS NOTHING is the case the stranding rule is about, and it has to be
#: expressible separately from "defaulted to its size" -- a first version of
#: this stub conflated them and two stranding tests failed against correct code.
REPORTS_NOTHING = object()


class _Route:
    """A ParkRoute stand-in: the protocol the translator ledger already uses."""

    def __init__(self, name, size, device="cuda:0", parked_bytes=None, hot=False):
        self.name = name
        self._size = size
        self._device = device
        self._parked = size if parked_bytes is None else parked_bytes
        self.hot = hot
        self.restored = False
        self.park_calls = 0

    def device(self):
        return self._device

    def size_bytes(self):
        return self._size

    def park(self):
        self.park_calls += 1
        if self._parked is REPORTS_NOTHING:
            return None
        return self._parked

    def restore(self):
        self.restored = True


class TestAddressingATenant(CustomTestCase):
    def test_parking_a_tenant_reports_what_its_routes_returned(self):
        mover = TenantMover()
        mover.register("translator", [_Route("asr", 100), _Route("codec", 50)])
        result = mover.park_tenant("translator")
        self.assertEqual(result.released_bytes, 150)
        self.assertTrue(result.ok)

    def test_an_unknown_tenant_refuses_by_name(self):
        mover = TenantMover()
        mover.register("translator", [_Route("asr", 100)])
        with self.assertRaises(UnknownTenant) as caught:
            mover.park_tenant("nobody")
        message = str(caught.exception)
        self.assertIn("nobody", message)
        # The refusal must name who IS registered, or the caller cannot correct.
        self.assertIn("translator", message)

    def test_tenants_are_independent(self):
        mover = TenantMover()
        a = _Route("a", 10)
        b = _Route("b", 20)
        mover.register("t1", [a])
        mover.register("t2", [b])
        mover.park_tenant("t1")
        self.assertEqual(a.park_calls, 1)
        self.assertEqual(b.park_calls, 0)


class TestStrandingIsNotZero(CustomTestCase):
    def test_a_route_that_reports_nothing_is_stranded(self):
        mover = TenantMover()
        mover.register("t", [_Route("good", 100), _Route("silent", 40, parked_bytes=REPORTS_NOTHING)])
        # REPORTS_NOTHING makes park() return None: asked, reported nothing.
        result = mover.park_tenant("t")
        self.assertEqual(result.released_bytes, 100)
        self.assertEqual([s.name for s in result.stranded], ["silent"])
        self.assertFalse(result.ok, "a stranded route must not read as success")

    def test_a_route_reporting_zero_is_an_accounting_not_stranding(self):
        mover = TenantMover()
        mover.register("t", [_Route("empty", 0, parked_bytes=0)])
        result = mover.park_tenant("t")
        self.assertEqual(result.released_bytes, 0)
        self.assertEqual(result.stranded, ())
        self.assertTrue(result.ok, "a delivered zero is a measurement")


class TestTheVocabularyIsConfiguration(CustomTestCase):
    """The translator's need order must not be hard-coded here."""

    def test_restore_follows_the_supplied_ranks(self):
        mover = TenantMover()
        order = []
        routes = [_Route("codec", 10), _Route("asr", 10), _Route("talker", 10)]
        for r in routes:
            r.restore = (lambda n=r.name: order.append(n))
        mover.register("t", routes, ranks={"asr": 0, "talker": 1, "codec": 2})
        mover.park_tenant("t")
        mover.restore_tenant("t")
        self.assertEqual(order, ["asr", "talker", "codec"])

    def test_a_second_tenant_may_use_a_different_vocabulary(self):
        mover = TenantMover()
        order = []
        routes = [_Route("x", 10), _Route("y", 10)]
        for r in routes:
            r.restore = (lambda n=r.name: order.append(n))
        mover.register("other", routes, ranks={"y": 0, "x": 1})
        mover.park_tenant("other")
        mover.restore_tenant("other")
        self.assertEqual(order, ["y", "x"], "the ordering must not be translator-shaped")

    def test_unranked_routes_restore_last_in_registration_order(self):
        mover = TenantMover()
        order = []
        routes = [_Route("a", 1), _Route("b", 1), _Route("c", 1)]
        for r in routes:
            r.restore = (lambda n=r.name: order.append(n))
        mover.register("t", routes, ranks={"c": 0})
        mover.park_tenant("t")
        mover.restore_tenant("t")
        self.assertEqual(order, ["c", "a", "b"])


class TestParkedBytesByDevice(CustomTestCase):
    def test_it_sums_per_device_from_reports(self):
        mover = TenantMover()
        mover.register(
            "t",
            [
                _Route("a", 100, device="cuda:0"),
                _Route("b", 50, device="cuda:1"),
                _Route("c", 25, device="cuda:0"),
            ],
        )
        mover.park_tenant("t")
        self.assertEqual(mover.parked_bytes_by_device(), {"cuda:0": 125, "cuda:1": 50})

    def test_nothing_parked_is_an_empty_map_not_zeros(self):
        mover = TenantMover()
        mover.register("t", [_Route("a", 100)])
        self.assertEqual(mover.parked_bytes_by_device(), {})


class TestItComposesWithColdEvent(CustomTestCase):
    """The mover must be usable as cold_event's release_fn without adaptation."""

    def test_release_fn_returns_the_cold_event_triple(self):
        mover = TenantMover()
        mover.register("t", [_Route("a", 70)])
        ok, delivered, detail = mover.release_fn()("t")
        self.assertTrue(ok)
        self.assertEqual(delivered, 70)
        self.assertIsInstance(detail, str)

    def test_a_stranded_release_reports_none_not_zero(self):
        mover = TenantMover()
        mover.register("t", [_Route("a", 70, parked_bytes=REPORTS_NOTHING)])
        ok, delivered, _ = mover.release_fn()("t")
        self.assertFalse(ok)
        self.assertIsNone(delivered, "cold_event reads None as stranded")


if __name__ == "__main__":
    unittest.main()
