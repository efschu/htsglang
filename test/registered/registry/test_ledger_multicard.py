"""The multi-card layer of the VRAM ledger: all-or-nothing, and UUID keying.

Hermetic. Card totals are supplied explicitly, the clock is hand-advanced and
pid liveness is injected, so nothing here depends on a driver, a device or a
wall clock. The single-card behaviour is covered by
``test/registered/video_enhance/test_reservation.py``, which is the same store:
M2 wrote it, M1 moved it into the registry, and both suites run against one
implementation on purpose.

    python -m pytest test/registered/registry/test_ledger_multicard.py -v
"""

import tempfile
import unittest
from pathlib import Path

from sglang.srt.registry.ledger import (
    MIB,
    CardDemand,
    MultiCardReservation,
    ReservationRejected,
    ReservationStore,
    TenantState,
    adopt,
    plan_reservation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CARD_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
CARD_B = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
GIB = 1024 * MIB


class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class MultiCardLedgerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.clock = _Clock()
        self.totals = {CARD_A: 20 * GIB, CARD_B: 32 * GIB}
        self.store = ReservationStore(
            self.root,
            clock=self.clock,
            total_bytes_resolver=lambda uuid: self.totals[uuid],
        )

    def hold(self, tenant_id, klass=1):
        return MultiCardReservation(
            self.store, tenant_id=tenant_id, klass=klass, totals=self.totals
        )

    # -- all-or-nothing ----------------------------------------------------

    def test_two_card_acquire_writes_both(self):
        hold = self.hold("tp2")
        hold.acquire(
            [
                CardDemand(CARD_A, 15 * GIB),
                CardDemand(CARD_B, 15 * GIB),
            ]
        )
        self.assertEqual(hold.cards, (CARD_A, CARD_B))
        for card in (CARD_A, CARD_B):
            entry = self.store.read(card).tenant("tp2")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.reserved_bytes, 15 * GIB)

    def test_partial_failure_rolls_the_first_card_back(self):
        """The whole reason the layer exists: no half-acquired engine."""
        other = self.hold("incumbent")
        other.acquire([CardDemand(CARD_B, 30 * GIB)])

        hold = self.hold("tp2")
        with self.assertRaises(ReservationRejected):
            hold.acquire([CardDemand(CARD_A, 8 * GIB), CardDemand(CARD_B, 8 * GIB)])

        self.assertIsNone(self.store.read(CARD_A).tenant("tp2"))
        self.assertIsNone(self.store.read(CARD_B).tenant("tp2"))
        self.assertEqual(hold.cards, ())
        # The incumbent is untouched: a failed admission never costs a holder
        # anything.
        self.assertEqual(
            self.store.read(CARD_B).tenant("incumbent").reserved_bytes, 30 * GIB
        )

    def test_promotion_failure_leaves_the_tenant_demoted_everywhere(self):
        hold = self.hold("tp2")
        hold.acquire(
            [CardDemand(CARD_A, 8 * GIB), CardDemand(CARD_B, 8 * GIB)],
            state=TenantState.COLD,
        )
        blocker = self.hold("blocker")
        blocker.acquire([CardDemand(CARD_B, 31 * GIB)])

        with self.assertRaises(ReservationRejected):
            hold.set_state(TenantState.HOT)

        for card in (CARD_A, CARD_B):
            entry = self.store.read(card).tenant("tp2")
            self.assertEqual(
                entry.state,
                TenantState.COLD,
                f"card {card} left the tenant half promoted",
            )

    def test_release_frees_every_card(self):
        hold = self.hold("tp2")
        hold.acquire([CardDemand(CARD_A, 4 * GIB), CardDemand(CARD_B, 4 * GIB)])
        hold.release()
        for card in (CARD_A, CARD_B):
            self.assertIsNone(self.store.read(card).tenant("tp2"))

    def test_two_ranks_on_one_card_are_one_entry_of_double_size(self):
        """Co-location is expressed as bytes, not as two rows.

        One tenant holds one reservation per card. Two ranks of the same engine
        sharing a card is one entry sized for both -- which is exactly what the
        invariant needs to see.
        """
        hold = self.hold("tp2-colocated")
        hold.acquire([CardDemand(CARD_B, 2 * 15 * GIB)])
        self.assertEqual(len(self.store.read(CARD_B).entries), 1)
        self.assertEqual(
            self.store.read(CARD_B).tenant("tp2-colocated").reserved_bytes, 30 * GIB
        )
        # 2 x 15 GiB + 400 MiB corridor against 32 GiB fits; a third rank does
        # not, and that is the physical-impossibility check, not a margin.
        with self.assertRaises(ReservationRejected):
            self.hold("third").acquire([CardDemand(CARD_B, 15 * GIB)])

    # -- identity ----------------------------------------------------------

    def test_uuid_keying_survives_enumeration_renumbering(self):
        """The card is the UUID. Index order is not an input anywhere.

        This is the failure the ledger was built against: NVML index 0 and 1
        swap between boots and driver states, so a store keyed on the index
        hands a tenant the other card's budget.
        """
        hold = self.hold("tenant")
        hold.acquire([CardDemand(CARD_B, 30 * GIB)])

        # Simulate a reboot that swaps enumeration: the same physical cards,
        # opposite indices, and the *totals resolver* now returns the other
        # order for indices -- which the ledger never consults.
        renumbered = {CARD_B: 32 * GIB, CARD_A: 20 * GIB}
        store2 = ReservationStore(
            self.root,
            clock=self.clock,
            total_bytes_resolver=lambda uuid: renumbered[uuid],
        )
        self.assertEqual(store2.read(CARD_B).reserved_bytes, 30 * GIB)
        self.assertEqual(store2.read(CARD_A).reserved_bytes, 0)
        # And a 20 GiB card is still refused a 30 GiB tenant after the swap.
        with self.assertRaises(ReservationRejected):
            store2.acquire(
                card_uuid=CARD_A,
                tenant_id="big",
                klass=1,
                reserved_bytes=30 * GIB,
            )

    def test_ledger_file_is_named_for_the_uuid(self):
        self.hold("t").acquire([CardDemand(CARD_A, GIB)])
        names = sorted(p.name for p in self.root.glob("*.json"))
        self.assertIn(CARD_A + ".json", names)

    # -- planning ----------------------------------------------------------

    def test_plan_reservation_is_read_only_and_specific(self):
        self.hold("incumbent").acquire([CardDemand(CARD_A, 18 * GIB)])
        report = plan_reservation(
            self.store, [CardDemand(CARD_A, 4 * GIB)], self.totals
        )
        self.assertFalse(report.fits)
        (shortfall,) = report.shortfalls
        self.assertEqual(shortfall.card_uuid, CARD_A)
        self.assertEqual(shortfall.held_bytes, 18 * GIB)
        self.assertEqual(shortfall.total_bytes, 20 * GIB)
        self.assertEqual(
            shortfall.shortfall_bytes, 18 * GIB + 4 * GIB + 400 * MIB - 20 * GIB
        )
        self.assertIn("incumbent", shortfall.holders)
        self.assertIn("incumbent", shortfall.render())
        # Nothing was written.
        self.assertIsNone(self.store.read(CARD_A).tenant("planned"))

    def test_plan_can_price_an_eviction_before_performing_one(self):
        self.hold("victim").acquire([CardDemand(CARD_A, 18 * GIB)])
        blocked = plan_reservation(
            self.store, [CardDemand(CARD_A, 10 * GIB)], self.totals
        )
        self.assertFalse(blocked.fits)
        freed = plan_reservation(
            self.store,
            [CardDemand(CARD_A, 10 * GIB)],
            self.totals,
            ignoring_tenants=["victim"],
        )
        self.assertTrue(freed.fits)

    def test_replanning_a_tenant_does_not_count_it_against_itself(self):
        self.hold("t").acquire([CardDemand(CARD_A, 18 * GIB)])
        self.assertTrue(
            plan_reservation(
                self.store,
                [CardDemand(CARD_A, 19 * GIB)],
                self.totals,
                excluding_tenant="t",
            ).fits
        )

    # -- leases ------------------------------------------------------------

    def test_heartbeat_pushes_every_cards_lease(self):
        hold = self.hold("tp2")
        hold.acquire(
            [CardDemand(CARD_A, GIB), CardDemand(CARD_B, GIB)], lease_seconds=100.0
        )
        self.clock.advance(90.0)
        hold.heartbeat(lease_seconds=100.0)
        for card in (CARD_A, CARD_B):
            entry = self.store.read(card).tenant("tp2")
            self.assertEqual(entry.lease_expiry_ts, self.clock.now + 100.0)

    def test_orphaned_multi_card_tenant_is_reaped_on_every_card(self):
        """A crashed tenant must not hold bytes on any card it touched."""
        store = ReservationStore(
            self.root,
            clock=self.clock,
            total_bytes_resolver=lambda uuid: self.totals[uuid],
            pid_alive=lambda pid: False,
        )
        hold = MultiCardReservation(
            store, tenant_id="crashed", klass=1, totals=self.totals
        )
        hold.acquire(
            [CardDemand(CARD_A, 18 * GIB), CardDemand(CARD_B, 30 * GIB)],
            lease_seconds=60.0,
            pid=999_999,
        )
        self.clock.advance(61.0)
        for card in (CARD_A, CARD_B):
            self.assertEqual(len(store.reap(card)), 1)
            self.assertIsNone(store.read(card).tenant("crashed"))

    def test_a_live_pid_keeps_its_bytes_through_an_expired_lease(self):
        """A starved heartbeat thread is not a dead tenant.

        Reclaiming from a running process hands the same bytes to two owners,
        which is worse than a stale lease line.
        """
        store = ReservationStore(
            self.root,
            clock=self.clock,
            total_bytes_resolver=lambda uuid: self.totals[uuid],
            pid_alive=lambda pid: True,
        )
        store.acquire(
            card_uuid=CARD_A,
            tenant_id="busy",
            klass=1,
            reserved_bytes=GIB,
            lease_seconds=10.0,
        )
        self.clock.advance(1000.0)
        self.assertEqual(store.reap(CARD_A), [])
        self.assertIsNotNone(store.read(CARD_A).tenant("busy"))

    # -- reattachment ------------------------------------------------------

    def test_adopt_reattaches_to_files_that_outlived_the_process(self):
        self.hold("tp2").acquire([CardDemand(CARD_A, GIB), CardDemand(CARD_B, GIB)])
        again = adopt(self.store, "tp2", 1, [CARD_A, CARD_B], totals=self.totals)
        self.assertEqual(again.cards, (CARD_A, CARD_B))
        again.release()
        self.assertIsNone(self.store.read(CARD_A).tenant("tp2"))

    def test_adopt_ignores_cards_the_tenant_never_held(self):
        self.hold("tp1").acquire([CardDemand(CARD_A, GIB)])
        again = adopt(self.store, "tp1", 1, [CARD_A, CARD_B], totals=self.totals)
        self.assertEqual(again.cards, (CARD_A,))


if __name__ == "__main__":
    unittest.main()
