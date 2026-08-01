"""A tier reservation is a NAMED post in the ledger that owns the bytes (#407).

DESIGN_305 §6, in capitals: *one ledger, no second accounting*. Two properties
carry that constraint and are pinned here:

1. **Named or refused.** The #260/#400 Posten convention -- every declared byte
   is booked under a name a refusal table can print. An unnamed post, or one
   spelled so it would collide with the ``TierId`` grammar, is refused at the
   door.

2. **The card's own ledger keeps the invariant.** ``VramLedgerHook`` writes
   through ``registry.ledger.ReservationStore``; the corridor, the lease and
   the ``sum(reserved) + corridor <= total`` check stay there. The test drives
   a real store in a temp directory -- injected totals, injected clock -- so
   the forwarding path is executed rather than described.

Hermetic: no driver, no card, no ``/run`` write.

    python -m pytest test/registered/unit/memtier/test_tier_reservations.py -v
"""

import tempfile
import unittest
from pathlib import Path

from sglang.srt.memtier.reservations import (
    InMemoryTierLedger,
    TierPost,
    TierReservationRejected,
    UnnamedPost,
    VramLedgerHook,
    ledger_post_name,
    summarise,
)
from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
)
from sglang.srt.planner.cost_model import Rate
from sglang.srt.registry.ledger import MIB, ReservationStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CARD = "GPU-11111111-2222-3333-4444-555555555555"
GIB = 1024 * MIB


def _caps(ledger_key):
    return TierCaps(
        latency_us=Rate.absent("fixture", unit="us"),
        bandwidth_gbs=Rate.measured(1.0, "fixture", unit="GB/s"),
        aperture_bytes=Rate.absent("fixture", unit="bytes"),
        ledger_key=ledger_key,
    )


def _host_tier(total=1000, floor=0):
    return TierDescriptor(
        id="host:rig-1",
        kind=TierKind.HOST,
        host="rig-1",
        capacity=TierCapacity(
            total=(
                Rate.absent("fixture: unknown", unit="bytes")
                if total is None
                else Rate.measured(float(total), "fixture", unit="bytes")
            ),
            floor=(
                Rate.absent("fixture: floor not measured", unit="bytes")
                if floor is None
                else Rate.measured(float(floor), "fixture", unit="bytes")
            ),
        ),
        volatility=Volatility.EXPENSIVE_OK,
        admits=frozenset({"experts"}),
        caps=_caps("host_ram"),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name="pcie"),
        profile_id="fixture",
    )


def _card_tier(uuid=CARD, total=20 * GIB):
    return TierDescriptor(
        id=f"vram:{uuid}",
        kind=TierKind.DEVICE,
        host="rig-1",
        capacity=TierCapacity(
            total=Rate.measured(float(total), "fixture NVML", unit="bytes"),
            floor=Rate.measured(0.0, "fixture", unit="bytes"),
            corridor=400 * MIB,
        ),
        volatility=Volatility.DEVICE_BOUND_ONLY,
        admits=frozenset({"experts"}),
        caps=_caps("vram"),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name="cuda-local"),
        profile_id="fixture",
        card_model="NVIDIA GeForce RTX 3080",
    )


class PostNamingTest(unittest.TestCase):
    def test_a_post_name_is_namespaced_by_tier_and_bucket(self):
        """A memtier post shares one JSON file with a tenant's own posts
        (``vram_dial``, ``serving_boot``); the namespace is what keeps them
        from colliding when cut 4 makes #286's ledger a view of this one."""
        self.assertEqual(
            ledger_post_name(_card_tier(), "expert_spill"),
            "memtier:vram:expert_spill",
        )

    def test_an_unnamed_or_unprintable_post_is_refused(self):
        """Can-fail proof: a bucket called "" or "Expert Spill" or one with a
        colon would break the refusal table and the id grammar respectively."""
        tier = _card_tier()
        for bad in ("", "Expert Spill", "expert:spill", "9lives", "expert-spill"):
            with self.assertRaises(UnnamedPost, msg=bad):
                ledger_post_name(tier, bad)
        with self.assertRaises(UnnamedPost):
            TierPost(name="Expert", nbytes=1, holder="tenant")
        with self.assertRaises(UnnamedPost):
            TierPost(name="expert", nbytes=1, holder="")


class InMemoryLedgerTest(unittest.TestCase):
    def setUp(self):
        self.ledger = InMemoryTierLedger()
        self.tier = _host_tier(total=1000, floor=100)

    def test_a_reservation_is_readable_back_by_name(self):
        self.ledger.reserve(
            tier=self.tier, post=TierPost(name="kv_spill", nbytes=300, holder="t1")
        )
        self.assertEqual(
            self.ledger.posts(self.tier.id), {"memtier:host_ram:kv_spill": 300}
        )
        self.assertEqual(self.ledger.reserved_bytes(self.tier.id), 300)

    def test_re_reserving_one_post_replaces_rather_than_accumulates(self):
        """A holder that raises its own budget must not be charged twice --
        the ``acquire`` semantics of the store this mirrors."""
        for size in (300, 500):
            self.ledger.reserve(
                tier=self.tier,
                post=TierPost(name="kv_spill", nbytes=size, holder="t1"),
            )
        self.assertEqual(self.ledger.reserved_bytes(self.tier.id), 500)

    def test_the_second_holder_is_refused_with_an_itemised_table(self):
        """#400: a rejection names the tier, the holders and the arithmetic --
        the failure it replaces is an OOM three minutes later with no table."""
        self.ledger.reserve(
            tier=self.tier, post=TierPost(name="kv_spill", nbytes=800, holder="t1")
        )
        with self.assertRaises(TierReservationRejected) as ctx:
            self.ledger.reserve(
                tier=self.tier,
                post=TierPost(name="experts", nbytes=500, holder="t2"),
            )
        message = str(ctx.exception)
        self.assertIn("host:rig-1", message)
        self.assertIn("memtier:host_ram:kv_spill", message)
        self.assertEqual(ctx.exception.requested_bytes, 500)
        self.assertEqual(self.ledger.reserved_bytes(self.tier.id), 800)

    def test_an_unbounded_headroom_refuses_instead_of_guessing(self):
        """Can-fail proof: a tier whose floor was never measured has no
        headroom. Treating "unknown" as "plenty" is exactly the #400 defect
        (arm L accepted, then dead at 31.14 GiB in use on a 31.34 GiB card)."""
        tier = _host_tier(total=1000, floor=None)
        with self.assertRaises(TierReservationRejected) as ctx:
            self.ledger.reserve(
                tier=tier, post=TierPost(name="kv_spill", nbytes=1, holder="t1")
            )
        self.assertIn("unbounded", str(ctx.exception))

    def test_release_requires_the_right_holder(self):
        """Negative branch: releasing somebody else's post would hand out bytes
        that are still in use."""
        self.ledger.reserve(
            tier=self.tier, post=TierPost(name="kv_spill", nbytes=300, holder="t1")
        )
        self.assertFalse(
            self.ledger.release(tier=self.tier, post_name="kv_spill", holder="t2")
        )
        self.assertTrue(
            self.ledger.release(tier=self.tier, post_name="kv_spill", holder="t1")
        )
        self.assertEqual(self.ledger.reserved_bytes(self.tier.id), 0)

    def test_summarise_lists_only_tiers_with_posts(self):
        self.ledger.reserve(
            tier=self.tier, post=TierPost(name="kv_spill", nbytes=300, holder="t1")
        )
        lines = summarise(self.ledger, (self.tier, _card_tier()))
        self.assertEqual(len(lines), 2)
        self.assertIn("host:rig-1", lines[0])


class VramLedgerHookTest(unittest.TestCase):
    """Device tiers forward into the card ledger; nothing is re-implemented."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ReservationStore(
            Path(self._tmp.name),
            total_bytes_resolver=lambda uuid: 20 * GIB,
            pid_alive=lambda pid: True,
            clock=lambda: 1_000_000.0,
        )
        self.hook = VramLedgerHook(self.store, tenant_id="srt-0", klass=1)
        self.tier = _card_tier()

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_device_reservation_lands_in_the_card_file_as_a_named_post(self):
        self.hook.reserve(
            tier=self.tier,
            post=TierPost(name="expert_spill", nbytes=GIB, holder="srt-0"),
        )
        entry = self.store.read(CARD).tenant("srt-0")
        self.assertEqual(entry.posts, {"memtier:vram:expert_spill": GIB})
        self.assertEqual(entry.reserved_bytes, GIB)
        self.assertEqual(self.hook.reserved_bytes(self.tier.id), GIB)

    def test_two_posts_sum_into_one_entry(self):
        """One tenant, one entry, several named items -- the shape
        ``ReservationEntry.posts`` already has. A second entry per post would
        be the "second accounting" §6 forbids."""
        self.hook.reserve(
            tier=self.tier,
            post=TierPost(name="expert_spill", nbytes=GIB, holder="srt-0"),
        )
        self.hook.reserve(
            tier=self.tier,
            post=TierPost(name="kv_spill", nbytes=2 * GIB, holder="srt-0"),
        )
        entry = self.store.read(CARD).tenant("srt-0")
        self.assertEqual(len(entry.posts), 2)
        self.assertEqual(entry.reserved_bytes, 3 * GIB)

    def test_the_card_invariant_still_rejects_and_the_reason_survives(self):
        """The corridor and the total belong to the card ledger. The hook must
        surface its rejection, not swallow or re-derive it."""
        with self.assertRaises(TierReservationRejected) as ctx:
            self.hook.reserve(
                tier=self.tier,
                post=TierPost(name="expert_spill", nbytes=20 * GIB, holder="srt-0"),
            )
        self.assertIn("corridor", str(ctx.exception).lower())
        self.assertIsNone(self.store.read(CARD).tenant("srt-0"))

    def test_release_removes_the_post_and_lowers_the_entry(self):
        self.hook.reserve(
            tier=self.tier,
            post=TierPost(name="expert_spill", nbytes=GIB, holder="srt-0"),
        )
        self.hook.reserve(
            tier=self.tier, post=TierPost(name="kv_spill", nbytes=GIB, holder="srt-0")
        )
        self.assertTrue(
            self.hook.release(tier=self.tier, post_name="kv_spill", holder="srt-0")
        )
        entry = self.store.read(CARD).tenant("srt-0")
        self.assertEqual(entry.posts, {"memtier:vram:expert_spill": GIB})
        self.assertEqual(entry.reserved_bytes, GIB)
        self.assertFalse(
            self.hook.release(tier=self.tier, post_name="kv_spill", holder="srt-0")
        )

    def test_a_non_device_tier_is_not_booked_in_a_card_file(self):
        with self.assertRaises(TypeError):
            self.hook.reserve(
                tier=_host_tier(),
                post=TierPost(name="kv_spill", nbytes=1, holder="srt-0"),
            )

    def test_an_unenumerated_card_has_no_ledger_to_book_in(self):
        tier = _card_tier().evolve(id="vram:unenumerated@rig-2", host="rig-2")
        with self.assertRaises(TierReservationRejected) as ctx:
            self.hook.reserve(
                tier=tier, post=TierPost(name="kv_spill", nbytes=1, holder="srt-0")
            )
        self.assertIn("never been enumerated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
