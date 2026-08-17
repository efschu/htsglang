"""#410 slice 1: the session manifest, against the real canonical store.

The payload format is not under test here -- #706 already proved it. What is
under test is the layer above it, and specifically the three ways a checkpoint
can lie about what it can restore:

1. IT CAN OMIT THE RECURRENT STATE. On a hybrid checkpoint a KV-only prefix
   resolves to ZERO usable tokens, not to fewer (the storage hit is the MINIMUM
   across pools and the device match advances only at nodes carrying mamba
   state -- ``test_mamba_gates_the_hit_706.py``). So a stateless manifest is
   refused where it is BUILT.
2. IT CAN CLAIM A POSITION WHERE NO STATE EXISTS. With a checkpoint interval,
   states live on a grid; a checkpoint is snapped DOWN to it and records both
   positions, because quietly covering fewer tokens than asked is a wrong
   answer rather than a rounding.
3. ITS REFERENCES CAN HAVE BEEN EVICTED. A manifest holds references, not
   bytes. ``test_a_manifest_whose_page_was_evicted_refuses_at_branch`` is the
   red-first case named in the slice: it must refuse AT BRANCH, before seeding,
   rather than produce a silent partial prefix.

The store side is real: pages are written through ``HiCacheFile`` in canonical
mode by three PP stages, and existence is asked the way a read would ask it,
so #706's invisible-until-complete rule is part of what is being tested.
"""

import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.session_manifest import (
    FORMAT_VERSION,
    GdnAnchor,
    ManifestError,
    ManifestIncomplete,
    SessionManifest,
    build_manifest,
    dumps,
    loads,
    seed_plan,
    verify_against_store,
    verify_model_identity,
)
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_CUT = [(0, 28), (28, 48), (48, 64)]
IDENTITY = "0123456789abcdef"
INTERVAL = 4
PAGES = [f"tok{i:02d}" for i in range(8)]


def _window(lo, hi):
    return window_for_layers(
        SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _backend(root, pp_rank):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=pp_rank,
            pp_size=3,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen3.6-27B",
            model_identity_hash=IDENTITY,
            canonical_kv_page=_window(*PP_CUT[pp_rank]),
        ),
        file_path=root,
    )


def _manifest(**over):
    kwargs = dict(
        model_identity=IDENTITY,
        session_id="sess-1",
        page_hashes=PAGES,
        requested_token_count=8,
        gdn_blob_key="tok07.mamba",
        checkpoint_interval=INTERVAL,
        is_hybrid_model=True,
    )
    kwargs.update(over)
    return build_manifest(**kwargs)


class TestManifestRules(CustomTestCase):
    def test_a_hybrid_checkpoint_without_state_is_refused_at_build(self):
        with self.assertRaises(ManifestError) as cm:
            _manifest(gdn_blob_key=None)
        self.assertIn("ZERO usable tokens", str(cm.exception))

    def test_a_dense_model_may_omit_the_anchor(self):
        m = _manifest(gdn_blob_key=None, is_hybrid_model=False)
        self.assertIsNone(m.gdn)
        self.assertEqual(m.references(), tuple(PAGES))

    def test_the_position_snaps_down_to_the_grid_and_says_so(self):
        m = _manifest(requested_token_count=7)
        self.assertEqual(m.token_count, 4)
        self.assertEqual(m.requested_token_count, 7)
        self.assertTrue(m.snapped)
        # And it names only the pages it actually covers.
        self.assertEqual(m.page_hashes, tuple(PAGES[:4]))
        self.assertEqual(m.gdn.token_position, 4)

    def test_an_on_grid_position_is_untouched(self):
        m = _manifest(requested_token_count=8)
        self.assertFalse(m.snapped)
        self.assertEqual(m.token_count, 8)

    def test_a_checkpoint_before_the_first_grid_point_is_refused(self):
        """Snapping to 0 would produce a checkpoint anchored to nothing."""
        with self.assertRaises(ManifestError) as cm:
            _manifest(requested_token_count=3)
        self.assertIn("no state to anchor", str(cm.exception))

    def test_claiming_more_tokens_than_pages_is_refused(self):
        with self.assertRaises(ManifestError):
            _manifest(page_hashes=PAGES[:2], requested_token_count=8)

    def test_without_an_interval_every_position_is_valid(self):
        m = _manifest(requested_token_count=7, checkpoint_interval=None)
        self.assertEqual(m.token_count, 7)
        self.assertFalse(m.snapped)

    # -- the versioned format (the #411 contract) ---------------------------

    def test_round_trip_is_deterministic(self):
        m = _manifest()
        once, twice = dumps(m), dumps(m)
        self.assertEqual(once, twice)
        back = loads(once)
        self.assertEqual(back.as_dict(), m.as_dict())
        self.assertEqual(back.gdn.blob_key, m.gdn.blob_key)

    def test_an_unknown_version_is_refused_not_best_effort_parsed(self):
        raw = dumps(_manifest()).replace(
            f'"format_version":{FORMAT_VERSION}', '"format_version":99'
        )
        with self.assertRaises(ManifestError) as cm:
            loads(raw)
        self.assertIn("#411", str(cm.exception))

    def test_a_foreign_model_identity_is_refused(self):
        m = _manifest()
        verify_model_identity(m, IDENTITY)  # no raise
        with self.assertRaises(ManifestError) as cm:
            verify_model_identity(m, "ffffffffffffffff")
        self.assertIn("different byte format", str(cm.exception))


class TestAgainstTheRealStore(CustomTestCase):
    """Completeness, asked of the real backend in canonical mode."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.stages = [_backend(self.root, r) for r in range(3)]
        self.reader = self.stages[0]

    def _store_page(self, key, stages=(0, 1, 2)):
        for rank in stages:
            stage = self.stages[rank]
            stage.set(key, _payload(stage.canonical_kv_page))

    def _store_blob(self, key):
        self.stages[0].set(key, torch.arange(48, dtype=torch.uint8))

    def _store_everything(self, manifest):
        for page in manifest.page_hashes:
            self._store_page(page)
        if manifest.gdn is not None:
            self._store_blob(manifest.gdn.blob_key)

    def test_a_complete_checkpoint_verifies_and_plans(self):
        m = _manifest()
        self._store_everything(m)
        self.assertEqual(verify_against_store(m, self.reader.exists), ())
        plan = seed_plan(m, exists=self.reader.exists, model_identity=IDENTITY)
        self.assertEqual(len(plan), len(PAGES) + 1)
        self.assertEqual([s.kind for s in plan][-1], "gdn")
        self.assertEqual([s.kind for s in plan][:-1], ["kv"] * len(PAGES))

    def test_a_manifest_whose_page_was_evicted_refuses_at_branch(self):
        """The red-first case: seed nothing rather than a partial prefix."""
        m = _manifest()
        self._store_everything(m)
        # Evict for real: remove the file the store would serve.
        import os

        evicted = m.page_hashes[3]
        path = self.reader._existing_path(self.reader._get_suffixed_key(evicted))
        os.remove(path)

        self.assertEqual(verify_against_store(m, self.reader.exists), (evicted,))
        with self.assertRaises(ManifestIncomplete) as cm:
            seed_plan(m, exists=self.reader.exists)
        self.assertIn(evicted, cm.exception.missing)
        self.assertIn("resolves to zero usable tokens", str(cm.exception))

    def test_a_missing_gdn_blob_refuses_too(self):
        """The half that #212 is about: every page present, state gone."""
        m = _manifest()
        for page in m.page_hashes:
            self._store_page(page)
        self.assertEqual(verify_against_store(m, self.reader.exists), (m.gdn.blob_key,))
        with self.assertRaises(ManifestIncomplete):
            seed_plan(m, exists=self.reader.exists)

    def test_a_half_written_page_counts_as_missing(self):
        """#706's invisible-until-complete rule, inherited: a page only two of
        three stages wrote is not a partial page, it is not a page."""
        m = _manifest()
        self._store_everything(m)
        partial = "tok99"
        self._store_page(partial, stages=(0, 1))
        m2 = _manifest(page_hashes=list(PAGES[:3]) + [partial], requested_token_count=4)
        self.assertEqual(verify_against_store(m2, self.reader.exists), (partial,))

    def test_the_plan_is_the_branch_order(self):
        """The recurrent state is fetched LAST, so its absence can never look
        like a completed seed."""
        m = _manifest()
        self._store_everything(m)
        plan = seed_plan(m, exists=self.reader.exists)
        self.assertEqual(plan[-1].key, m.gdn.blob_key)
        self.assertEqual([s.key for s in plan[:-1]], list(m.page_hashes))

    def test_branch_lineage_is_recorded(self):
        m = _manifest()
        child = _manifest(
            session_id="sess-2",
            parent={"checkpoint_id": "ckpt-1", "branch_position": m.token_count},
        )
        self.assertEqual(child.parent["branch_position"], 8)
        self.assertEqual(loads(dumps(child)).parent, child.parent)


class TestManifestShape(CustomTestCase):
    def test_references_are_pages_then_state(self):
        m = SessionManifest(
            model_identity=IDENTITY,
            session_id="s",
            token_count=2,
            page_hashes=("a", "b"),
            gdn=GdnAnchor(blob_key="b.mamba", token_position=2),
        )
        self.assertEqual(m.references(), ("a", "b", "b.mamba"))

    def test_a_manifest_carries_no_geometry(self):
        """What makes it portable (#411) and what #706 earned: no rank, no
        world size, no layer cut anywhere in the serialised form."""
        blob = dumps(_manifest())
        for forbidden in ("tp_rank", "tp_size", "pp_rank", "pp_size", "layer"):
            self.assertNotIn(forbidden, blob)


if __name__ == "__main__":
    unittest.main()
