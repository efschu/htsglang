"""#410 slice 2: pins, and their honest cost.

Slice 1 refused to branch a checkpoint whose references had been evicted. That
is the right failure and still a failure: the store is an LRU cache, a
conversation checkpoint is not cache-shaped, and its whole value is staying
restorable after its pages stop being hot. So a checkpoint pins what it
references -- and a pin is a promise the eviction path has to keep AND account
for.

The three named red-first cases:

* ``test_the_pinned_sibling_survives_eviction_pressure`` -- two pages, one
  pinned, real eviction pressure on the real evictor: the unpinned one dies and
  the pinned one is still there. Without the skip, LRU order alone decides.
* ``test_pinned_bytes_are_not_reported_as_reclaimable`` -- the #715 lesson as
  arithmetic. A capacity decision that reads ``used_bytes`` believes it can
  reclaim what eviction will skip; ``reclaimable_bytes`` is the number the
  actuator can actually deliver.
* ``test_an_orphan_pin_is_reaped`` -- a pin whose checkpoint is gone protects
  nothing and blocks eviction forever, which is the ``.part706`` leak shape one
  layer up. Age-gated for the same reason: a checkpoint being written has pins
  before it has a manifest.

Plus the property that makes branching work at all: pins are REF-COUNTED, so
two checkpoints sharing a prefix share its pages and unpinning one does not
strip the other.
"""

import os
import tempfile
import unittest

import torch

try:
    from sglang.srt.mem_cache.canonical_page_store import sweep_partials
except ImportError:  # canonical_page_store is not in this lineage (#411 recon)
    # The pinned-partial protection is an integration with the canonical page
    # store, which the adopted #410 lineage does not carry. Skipped rather than
    # deleted so it lights up by itself the day that store is ported, instead
    # of being a gap nobody remembers to re-test.
    sweep_partials = None
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pin_ledger import (
    PinBudgetExceeded,
    PinLedger,
    stems_with_sizes,
)
from sglang.test.test_utils import CustomTestCase

IDENTITY = "0123456789abcdef"
PAGE = torch.arange(64, dtype=torch.uint8)


def _backend(root, *, extra=None):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen3.6-27B",
            model_identity_hash=IDENTITY,
            extra_config=extra,
        ),
        file_path=root,
    )


class TestPinsUnderPressure(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.page_cost = self._measure_page_cost()

    def _measure_page_cost(self) -> int:
        """What ONE page costs the evictor's accounting.

        Measured, not assumed: the evictor charges the ALLOCATED size, so a
        64-byte page costs a filesystem block. A cap computed from the tensor
        length would evict the page before a test could pin it -- and the pin
        would then correctly charge nothing, so the test would fail for a
        reason that has nothing to do with pinning."""
        probe_dir = tempfile.mkdtemp(dir=self.root)
        # The probe needs a cap of its own: byte accounting only runs when
        # eviction is CONFIGURED, so an uncapped probe reports 0 used bytes and
        # every cap derived from it would be 0.
        probe = _backend(probe_dir, extra={"max_size": str(1 << 30)})
        probe.set("aa_probe", PAGE)
        cost = int(probe.capacity_stats()["used_bytes"])
        assert cost > 0, "page cost probe measured nothing"
        return cost

    def test_the_pinned_sibling_survives_eviction_pressure(self):
        """The red-first case: identical pages, one pinned, real pressure."""
        # A cap that fits two pages but not three forces a real eviction.
        store = _backend(
            self.root,
            extra={"max_size": str(3 * self.page_cost), "eviction_ratio": 1.0},
        )
        store.set("aa_keepme", PAGE)
        store.set("bb_dropme", PAGE)
        store.pin_checkpoint("ckpt-1", ["aa_keepme"])

        for i in range(6):  # push well past the cap
            store.set(f"cc_filler{i:02d}", PAGE)

        self.assertTrue(store.exists("aa_keepme"), "pinned page was evicted")
        self.assertFalse(store.exists("bb_dropme"), "unpinned page survived")

    def test_pinned_bytes_are_not_reported_as_reclaimable(self):
        """#715: never report as deliverable what the actuator cannot deliver."""
        store = _backend(self.root, extra={"max_size": str(64 * self.page_cost)})
        store.set("aa_pinned", PAGE)
        store.set("bb_free", PAGE)
        before = store.capacity_stats()
        self.assertEqual(before["pinned_bytes"], 0)
        self.assertEqual(before["reclaimable_bytes"], before["used_bytes"])

        result = store.pin_checkpoint("ckpt-1", ["aa_pinned"])
        after = store.capacity_stats()
        self.assertEqual(after["pinned_entries"], 1)
        self.assertGreater(after["pinned_bytes"], 0)
        self.assertEqual(
            after["reclaimable_bytes"], after["used_bytes"] - after["pinned_bytes"]
        )
        self.assertEqual(result.bytes_added, after["pinned_bytes"])

    def test_a_store_of_only_pins_cannot_spin(self):
        """Eviction reuses the in-flight skip budget, so an all-pinned store
        runs out of attempts instead of looping forever."""
        # Cap fits exactly the three pages, so they are all still present when
        # they are pinned; the NEXT write is the one with nowhere to go.
        store = _backend(
            self.root,
            extra={"max_size": str(3 * self.page_cost), "eviction_ratio": 1.0},
        )
        for i in range(3):
            store.set(f"aa_p{i}", PAGE)
        store.pin_checkpoint("ckpt-1", [f"aa_p{i}" for i in range(3)])
        self.assertEqual(store.capacity_stats()["pinned_entries"], 3)

        # Completes rather than hangs: every candidate is pinned, so the loop
        # exhausts its skip budget and returns. Whether the write is admitted
        # or refused is the evictor's business; not spinning is this test's.
        store.set("bb_new", PAGE)
        self.assertTrue(all(store.exists(f"aa_p{i}") for i in range(3)))
        stats = store.capacity_stats()
        self.assertEqual(stats["reclaimable_bytes"], 0)


class TestLedgerSemantics(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.ledger = PinLedger(self.root)

    def test_pins_are_ref_counted_across_checkpoints(self):
        """Branching exists to SHARE a prefix, so unpinning one holder must not
        strip its sibling."""
        self.ledger.pin("ckpt-1", {"shared": 100, "only-1": 10})
        self.ledger.pin("ckpt-2", {"shared": 100, "only-2": 20})
        self.assertEqual(self.ledger.pinned_bytes(), 130)

        freed = self.ledger.unpin("ckpt-1")
        self.assertEqual(freed, 10)  # not the shared page
        self.assertTrue(self.ledger.is_pinned("shared"))
        self.assertFalse(self.ledger.is_pinned("only-1"))

        freed = self.ledger.unpin("ckpt-2")
        self.assertEqual(freed, 120)  # now the shared page goes too
        self.assertFalse(self.ledger.is_pinned("shared"))
        self.assertEqual(self.ledger.pinned_bytes(), 0)

    def test_the_ledger_is_visible_per_checkpoint(self):
        self.ledger.pin("ckpt-1", {"a": 100, "b": 200})
        self.ledger.pin("ckpt-2", {"c": 50})
        view = self.ledger.ledger()
        self.assertEqual(view["checkpoints"]["ckpt-1"]["entries"], 2)
        self.assertEqual(view["checkpoints"]["ckpt-1"]["bytes"], 300)
        self.assertEqual(view["pinned_entries"], 3)
        self.assertEqual(view["pinned_bytes"], 350)

    def test_pins_survive_a_restart(self):
        """A checkpoint outlives the process; a memory-only pin would silently
        stop protecting anything at the next boot."""
        self.ledger.pin("ckpt-1", {"a": 100})
        reloaded = PinLedger(self.root)
        self.assertEqual(reloaded.load(), 1)
        self.assertTrue(reloaded.is_pinned("a"))
        self.assertEqual(reloaded.pinned_bytes(), 100)

    def test_the_budget_refuses_with_the_numbers(self):
        ledger = PinLedger(self.root, budget_bytes=250)
        ledger.pin("ckpt-1", {"a": 200})
        with self.assertRaises(PinBudgetExceeded) as cm:
            ledger.pin("ckpt-2", {"b": 100})
        message = str(cm.exception)
        for number in ("100", "200", "250", "50"):
            self.assertIn(number, message)
        self.assertIn("pin museum", message)
        # Refused means NOT pinned: the ledger is unchanged.
        self.assertFalse(ledger.is_pinned("b"))
        self.assertEqual(ledger.pinned_bytes(), 200)

    def test_re_pinning_shared_pages_does_not_recharge_the_budget(self):
        """A branch that shares its parent's entire prefix costs no new bytes,
        which is the case the budget must not punish."""
        ledger = PinLedger(self.root, budget_bytes=250)
        ledger.pin("ckpt-1", {"a": 200})
        result = ledger.pin("ckpt-2", {"a": 200})  # same page, new holder
        self.assertEqual(result.bytes_added, 0)
        self.assertEqual(result.bytes_shared, 200)
        self.assertEqual(ledger.pinned_bytes(), 200)

    def test_pinning_a_page_that_is_not_there_charges_nothing(self):
        missing = os.path.join(self.root, "nope.bin")
        self.assertEqual(stems_with_sizes([("nope", missing)]), {})

    # -- orphans ------------------------------------------------------------

    def test_an_orphan_pin_is_reaped(self):
        self.ledger.pin("ckpt-gone", {"a": 100})
        self.ledger.pin("ckpt-live", {"b": 50})
        alive = {"ckpt-live"}
        reaped = self.ledger.reap_orphans(
            lambda cid: cid in alive, older_than_s=0, now=lambda: 1e12
        )
        self.assertEqual(reaped, ("ckpt-gone",))
        self.assertFalse(self.ledger.is_pinned("a"))
        self.assertTrue(self.ledger.is_pinned("b"))
        self.assertEqual(self.ledger.pinned_bytes(), 50)

    def test_a_young_orphan_is_left_alone(self):
        """A checkpoint being written has pins before it has a manifest;
        reaping those would delete the protection exactly when it is needed."""
        self.ledger.pin("ckpt-new", {"a": 100})
        reaped = self.ledger.reap_orphans(lambda cid: False, older_than_s=3600)
        self.assertEqual(reaped, ())
        self.assertTrue(self.ledger.is_pinned("a"))


class TestSweeperRespectsPins(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _old_partial(self, stem):
        shard = os.path.join(self.root, stem[:2])
        os.makedirs(shard, exist_ok=True)
        path = os.path.join(shard, f"{stem}.bin.part706")
        with open(path, "wb") as f:
            f.write(b"\x00")
        stamp = os.stat(path).st_mtime - 7200
        os.utime(path, (stamp, stamp))
        return path

    @unittest.skipIf(sweep_partials is None, "canonical_page_store not in this lineage")
    def test_a_pinned_pages_partial_is_not_reaped(self):
        """Age cannot tell an abandoned partial from one a checkpoint is
        waiting on; a pin can."""
        pinned = self._old_partial("aa_pinned")
        loose = self._old_partial("bb_loose")
        ledger = PinLedger(self.root)
        ledger.pin("ckpt-1", {"aa_pinned": 1})
        reaped = sweep_partials(
            self.root, older_than_s=3600, is_pinned=ledger.is_pinned
        )
        self.assertEqual(reaped, 1)
        self.assertTrue(os.path.exists(pinned))
        self.assertFalse(os.path.exists(loose))

    @unittest.skipIf(sweep_partials is None, "canonical_page_store not in this lineage")
    def test_without_pins_the_sweeper_is_unchanged(self):
        loose = self._old_partial("bb_loose")
        self.assertEqual(sweep_partials(self.root, older_than_s=3600), 1)
        self.assertFalse(os.path.exists(loose))


if __name__ == "__main__":
    unittest.main()
