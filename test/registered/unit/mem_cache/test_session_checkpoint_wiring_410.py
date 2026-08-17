"""#410 slice 2 wiring: the checkpoint lifecycle that actually takes the pins.

The pin ledger was complete and had NO CALLER. ``pin_checkpoint`` existed,
the evictor honoured it, the budget refused -- and no checkpoint ever pinned
anything, so slice 1's "a reference was evicted, refusing to branch" still
governed every checkpoint in practice. A protection never taken looks finished
in the diff and does nothing at runtime, which is the #698 lesson from the
other side.

So every case here drives the REAL store and the REAL evictor under REAL
pressure. None of them assert on the ledger's own bookkeeping: a test that asks
the ledger whether it pinned something would have passed against the unwired
tree, which is exactly the failure being closed.

The named cases from the brief:

* ``test_a_pinned_checkpoint_still_branches_after_eviction_pressure`` -- create,
  flood the store past its cap, branch. The plan comes back whole.
* ``test_an_unpinned_checkpoint_still_refuses_after_eviction`` -- slice 1's
  refusal is NOT superseded. A checkpoint nobody paid to keep still fails at
  branch time, and it fails the same way it did before.
* ``test_the_budget_refuses_the_create_and_leaves_nothing_behind`` -- over
  budget means no manifest, no pins, no half-created checkpoint.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pin_ledger import PinBudgetExceeded
from sglang.srt.mem_cache.session_checkpoint import (
    CheckpointExists,
    CheckpointNotFound,
    branch_plan,
    create_checkpoint,
    delete_checkpoint,
    list_checkpoints,
    load_checkpoint,
)
from sglang.srt.mem_cache.session_manifest import ManifestIncomplete, build_manifest
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


def _manifest(session_id, keys):
    """A non-hybrid checkpoint over ``keys``: no GDN anchor, no grid."""
    return build_manifest(
        model_identity=IDENTITY,
        session_id=session_id,
        page_hashes=list(keys),
        requested_token_count=len(keys),
        gdn_blob_key=None,
        checkpoint_interval=None,
        is_hybrid_model=False,
    )


class TestCheckpointWiring(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.page_cost = self._measure_page_cost()

    def _measure_page_cost(self) -> int:
        """What ONE page costs the evictor's accounting -- measured.

        Same trap as the ledger suite: the evictor charges ALLOCATED size, so a
        cap computed from the tensor length evicts pages before they can be
        pinned and the test then fails for a reason unrelated to pinning. The
        probe needs a cap of its own, because byte accounting only runs when
        eviction is configured.
        """
        probe_dir = tempfile.mkdtemp(dir=self.root)
        probe = _backend(probe_dir, extra={"max_size": str(1 << 30)})
        probe.set("aa_probe", PAGE)
        cost = int(probe.capacity_stats()["used_bytes"])
        assert cost > 0, "page cost probe measured nothing"
        return cost

    def _pressured_store(self, pages_that_fit=3, **kw):
        return _backend(
            self.root,
            extra={
                "max_size": str(pages_that_fit * self.page_cost),
                "eviction_ratio": 1.0,
                **kw,
            },
        )

    def _flood(self, store, n=8):
        """Push well past the cap so eviction genuinely runs."""
        for i in range(n):
            store.set(f"zz_filler{i:02d}", PAGE)

    # ---------------------------------------------------------------- named

    def test_a_pinned_checkpoint_still_branches_after_eviction_pressure(self):
        """Create -> pressure -> branch. The whole point of the slice."""
        store = self._pressured_store()
        keys = ["aa_page0", "aa_page1"]
        for k in keys:
            store.set(k, PAGE)

        create_checkpoint(store, "ckpt-pinned", _manifest("sess-1", keys))
        self._flood(store)

        # The branch plan is the operation a user actually performs, and it
        # verifies against the live store -- so this is not asking the ledger
        # what it thinks, it is asking whether the pages are there.
        plan = branch_plan(store, "ckpt-pinned", model_identity=IDENTITY)
        self.assertEqual([s.key for s in plan], keys)
        for k in keys:
            self.assertTrue(store.exists(k), f"pinned reference {k} was evicted")

    def test_an_unpinned_checkpoint_still_refuses_after_eviction(self):
        """Slice 1's refusal survives, for checkpoints nobody paid to keep.

        Both behaviours are live at once. Pinning did not replace the refusal;
        it made the refusal avoidable for those who pay for it.
        """
        store = self._pressured_store()
        keys = ["bb_page0", "bb_page1"]
        for k in keys:
            store.set(k, PAGE)

        create_checkpoint(store, "ckpt-cheap", _manifest("sess-2", keys), pin=False)
        self._flood(store)

        with self.assertRaises(ManifestIncomplete) as cm:
            branch_plan(store, "ckpt-cheap", model_identity=IDENTITY)
        self.assertTrue(cm.exception.missing, "refusal named no missing reference")

    def test_the_budget_refuses_the_create_and_leaves_nothing_behind(self):
        """Over budget: no pins, no manifest, no half-created checkpoint."""
        # The budget is read ONCE, when the store builds its ledger, so it has
        # to be in the environment before the store exists.
        with mock.patch.dict(
            os.environ, {"SGLANG_HICACHE_PIN_BUDGET_BYTES": "1"}, clear=False
        ):
            store = _backend(self.root, extra={"max_size": str(1 << 30)})
        self.assertEqual(
            int(store.pins.budget_bytes), 1, "the budget did not reach the ledger"
        )
        keys = ["cc_page0"]
        for k in keys:
            store.set(k, PAGE)

        with self.assertRaises(PinBudgetExceeded) as cm:
            create_checkpoint(store, "ckpt-toobig", _manifest("sess-3", keys))

        message = str(cm.exception)
        self.assertIn("ckpt-toobig", message)
        for number in ("budget", "exceeds"):
            self.assertIn(number, message)
        self.assertEqual(int(store.pin_stats()["pinned_bytes"]), 0)
        self.assertEqual(list_checkpoints(store), ())
        with self.assertRaises(CheckpointNotFound):
            load_checkpoint(store, "ckpt-toobig")

    # ------------------------------------------------------------ lifecycle

    def test_delete_releases_the_pin_and_the_pages_become_evictable(self):
        """Unpin-on-delete, proven by eviction rather than by the ledger."""
        store = self._pressured_store()
        keys = ["dd_page0"]
        for k in keys:
            store.set(k, PAGE)
        create_checkpoint(store, "ckpt-temp", _manifest("sess-4", keys))

        delete_checkpoint(store, "ckpt-temp")
        self._flood(store)

        self.assertFalse(
            store.exists("dd_page0"),
            "the page stayed protected after its checkpoint was deleted",
        )
        self.assertEqual(list_checkpoints(store), ())

    def test_a_branch_sharing_the_whole_prefix_costs_no_new_pinned_bytes(self):
        """Ref-counting, at the layer that creates the second holder."""
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        keys = ["ee_page0", "ee_page1"]
        for k in keys:
            store.set(k, PAGE)

        first = create_checkpoint(store, "ckpt-parent", _manifest("sess-5", keys))
        self.assertGreater(first.bytes_added, 0)

        second = create_checkpoint(store, "ckpt-branch", _manifest("sess-6", keys))
        self.assertEqual(second.bytes_added, 0, "a shared prefix was charged twice")
        self.assertGreater(second.bytes_shared, 0)

    def test_deleting_a_branch_does_not_strip_its_parent(self):
        """The property branching exists for, driven through eviction."""
        store = self._pressured_store()
        keys = ["ff_page0"]
        for k in keys:
            store.set(k, PAGE)
        create_checkpoint(store, "ckpt-p", _manifest("sess-7", keys))
        create_checkpoint(store, "ckpt-b", _manifest("sess-8", keys))

        delete_checkpoint(store, "ckpt-b")
        self._flood(store)

        self.assertTrue(
            store.exists("ff_page0"),
            "deleting the branch stripped the parent's protection",
        )
        branch_plan(store, "ckpt-p", model_identity=IDENTITY)

    def test_a_create_over_missing_references_is_refused_before_pinning(self):
        """Pinning a key that is not there protects nothing while looking
        like success, so an already-incomplete prefix is refused at create."""
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        store.set("gg_page0", PAGE)

        with self.assertRaises(ManifestIncomplete):
            create_checkpoint(
                store, "ckpt-gap", _manifest("sess-9", ["gg_page0", "gg_missing"])
            )
        self.assertEqual(int(store.pin_stats()["pinned_bytes"]), 0)
        self.assertEqual(list_checkpoints(store), ())

    def test_a_failed_manifest_write_releases_the_pins(self):
        """Rollback: a create that cannot persist leaves no pinned bytes.

        Without it the pins outlive the failure and only the age-gated orphan
        reaper would ever collect them -- protection held for a checkpoint that
        will never exist.
        """
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        store.set("hh_page0", PAGE)

        with mock.patch(
            "sglang.srt.mem_cache.session_checkpoint._write_atomic",
            side_effect=OSError("disk went away"),
        ):
            with self.assertRaises(OSError):
                create_checkpoint(store, "ckpt-doomed", _manifest("s-10", ["hh_page0"]))

        self.assertEqual(int(store.pin_stats()["pinned_bytes"]), 0)
        self.assertEqual(list_checkpoints(store), ())

    def test_the_pin_is_taken_before_the_manifest_is_written(self):
        """The ordering claim, made falsifiable.

        Between "the references exist" and "the checkpoint is durable" the
        pages are ordinary LRU entries. This drives eviction INSIDE that
        window: if the pin were taken after the manifest write, the flood
        would take the pages and the checkpoint would be born unbranchable.
        """
        store = self._pressured_store()
        keys = ["ll_page0"]
        for k in keys:
            store.set(k, PAGE)

        real_write = None
        from sglang.srt.mem_cache import session_checkpoint as sc

        real_write = sc._write_atomic

        def evict_then_write(path, blob):
            self._flood(store)  # the crash window, made real
            return real_write(path, blob)

        with mock.patch.object(sc, "_write_atomic", evict_then_write):
            create_checkpoint(store, "ckpt-order", _manifest("s-14", keys))

        plan = branch_plan(store, "ckpt-order", model_identity=IDENTITY)
        self.assertEqual([s.key for s in plan], keys)

    def test_creating_over_an_existing_id_is_refused(self):
        """Overwriting would unpin the old references as a side effect of a
        call that reads like a create."""
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        store.set("ii_page0", PAGE)
        create_checkpoint(store, "ckpt-dup", _manifest("s-11", ["ii_page0"]))

        with self.assertRaises(CheckpointExists):
            create_checkpoint(store, "ckpt-dup", _manifest("s-11", ["ii_page0"]))
        self.assertEqual(list_checkpoints(store), ("ckpt-dup",))

    def test_a_checkpoint_survives_a_restart(self):
        """Durability at the lifecycle layer: manifest and pins both reload."""
        store = self._pressured_store()
        keys = ["jj_page0"]
        for k in keys:
            store.set(k, PAGE)
        create_checkpoint(store, "ckpt-durable", _manifest("s-12", keys))

        reopened = self._pressured_store()
        self.assertEqual(list_checkpoints(reopened), ("ckpt-durable",))
        self._flood(reopened)
        plan = branch_plan(reopened, "ckpt-durable", model_identity=IDENTITY)
        self.assertEqual([s.key for s in plan], keys)

    def test_deleting_an_unknown_checkpoint_is_refused(self):
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        with self.assertRaises(CheckpointNotFound):
            delete_checkpoint(store, "ckpt-nope")

    def test_the_manifest_round_trips(self):
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        store.set("kk_page0", PAGE)
        manifest = _manifest("s-13", ["kk_page0"])
        create_checkpoint(store, "ckpt-rt", manifest)

        loaded = load_checkpoint(store, "ckpt-rt")
        self.assertEqual(loaded.references(), manifest.references())
        self.assertEqual(loaded.model_identity, IDENTITY)


if __name__ == "__main__":
    unittest.main()
