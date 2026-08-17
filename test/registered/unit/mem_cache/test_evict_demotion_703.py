"""#703: a warm prefix must survive eviction by going to disk, not by luck.

THE PREMISE, established at the code before any of this was built:

* the DEFAULT ``write_through`` policy writes to storage on the write-through
  ACK (``hiradix_cache._finish_write_through_ack`` -> ``write_backup_storage``),
  i.e. INSERT-driven, not eviction-driven;
* device eviction either DEMOTES a node that already has a host copy
  (``_evict_backuped``) or DROPS one that does not (``_evict_regular``);
* host eviction (``evict_host``) frees the host rows and unlinks the node.

Neither eviction path writes to storage. So a prefix that never reached disk --
staged below the write-through threshold, refused by the evictor's cap,
attached to storage late, or dropped when the host pool was full under
``write_back`` -- is LOST at eviction. That gap is what the demotion hook
closes, and it is what these tests pin.

The round trip is exercised against the REAL canonical store: pages are written
through ``HiCacheFile`` in canonical mode by three PP stages, so the key, the
bytes and #706's invisible-until-complete rule are all real. Only the tree and
the controller are fakes, and they mirror exactly the surface the hook touches
(``enable_storage``, ``ongoing_backup``, ``write_backup_storage``).

RED FIRST: ``test_without_demotion_the_evicted_prefix_is_gone`` is today's
behaviour -- evict, re-request, nothing on disk. The green twin differs only in
whether the hook ran.
"""

import tempfile
import unittest

import torch

from sglang.srt.mem_cache import hicache_demotion as demotion
from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_CUT = [(0, 28), (28, 48), (48, 64)]
IDENTITY = "0123456789abcdef"
KEY = "warmprefix01"


def _window(lo, hi):
    return window_for_layers(
        SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _backend(root, *, pp_rank=0, pp_size=3, tp_rank=0, tp_size=1, window, dcp=False):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen3.6-27B",
            model_identity_hash=IDENTITY,
            dcp_owner_mode=dcp,
            canonical_kv_page=window,
        ),
        file_path=root,
    )


class _Node:
    """The parts of a TreeNode the demotion hook reads."""

    def __init__(self, key, payload):
        self.hash_value = [key]
        self.host_value = payload


class _FakeTree:
    """The surface the hook touches, and nothing else.

    ``write_backup_storage`` is the real method's contract: take the node's
    host bytes and put them in the store under its content key. Here it writes
    through the REAL backend, so the key and the bytes are real.
    """

    def __init__(self, backend, *, enable_storage=True):
        self.enable_storage = enable_storage
        self.ongoing_backup = {}
        self._backend = backend
        self.writes = []

    def write_backup_storage(self, node, backup_len=None):
        key = node.hash_value[-1]
        self.writes.append(key)
        self._backend.set(key, node.host_value)


class TestDemotionRoundTrip(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        demotion.reset_stats()
        self.addCleanup(demotion.reset_stats)
        self.stages = [
            _backend(self.root, pp_rank=r, window=_window(*PP_CUT[r])) for r in range(3)
        ]

    def _enable(self, cap="4"):
        import os

        os.environ["SGLANG_HICACHE_DEMOTE_ON_EVICT"] = cap
        self.addCleanup(os.environ.pop, "SGLANG_HICACHE_DEMOTE_ON_EVICT", None)

    def _evict_the_prefix(self, tree, stage_rank):
        """The moment the prefix leaves memory. Its host bytes are still valid
        here -- that is exactly why the hook runs BEFORE the free."""
        stage = self.stages[stage_rank]
        node = _Node(KEY, _payload(stage.canonical_kv_page))
        return demotion.demote_before_host_evict(tree, node)

    # -- red: today's behaviour ---------------------------------------------

    def test_without_demotion_the_evicted_prefix_is_gone(self):
        tree = _FakeTree(self.stages[0])
        self.assertFalse(self._evict_the_prefix(tree, 0))
        self.assertEqual(tree.writes, [])
        for stage in self.stages:
            self.assertFalse(stage.exists(KEY))

    # -- green: the hook -----------------------------------------------------

    def test_demotion_puts_the_prefix_on_disk(self):
        self._enable()
        tree = _FakeTree(self.stages[0])
        self.assertTrue(self._evict_the_prefix(tree, 0))
        self.assertEqual(tree.writes, [KEY])
        self.assertEqual(demotion.stats().demoted, 1)

    def test_the_round_trip_returns_identical_bytes(self):
        """The whole point: evicted in one geometry, served in another, byte
        for byte, through the canonical key."""
        self._enable()
        # All three stages demote their slice as the prefix leaves memory.
        for rank in range(3):
            tree = _FakeTree(self.stages[rank])
            self.assertTrue(self._evict_the_prefix(tree, rank))

        # Re-requested by the OTHER geometry: the TP decode phase, all 16 slots.
        reader = _backend(
            self.root,
            pp_rank=0,
            pp_size=1,
            tp_rank=1,
            tp_size=3,
            dcp=True,
            window=window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS),
        )
        self.assertTrue(reader.exists(KEY))
        out = torch.zeros(reader.canonical_kv_page.byte_length, dtype=torch.uint8)
        self.assertIsNotNone(reader.get(KEY, out))
        self.assertTrue(torch.equal(out, _payload(reader.canonical_kv_page)))

    def test_a_partially_demoted_prefix_is_not_served(self):
        """#706's rule, inherited: two of three stages is not a page."""
        self._enable()
        for rank in (0, 1):
            tree = _FakeTree(self.stages[rank])
            self._evict_the_prefix(tree, rank)
        self.assertFalse(self.stages[0].exists(KEY))

    # -- the guards ----------------------------------------------------------

    def test_off_by_default(self):
        """And OFF AT THE GATE, not merely at the cap.

        Found by mutation: with the gate removed the call still returned False,
        because an unset cap is 0 and everything then drops. That is
        fail-closed and welcome, but it made the gate untested -- so this also
        asserts nothing was COUNTED as backpressure, which only holds if the
        gate short-circuited first."""
        tree = _FakeTree(self.stages[0])
        self.assertFalse(demotion.enabled())
        self.assertFalse(self._evict_the_prefix(tree, 0))
        self.assertEqual(demotion.stats().dropped_backpressure, 0)
        self.assertEqual(demotion.stats().skipped_no_storage, 0)
        self.assertEqual(tree.writes, [])

    def test_the_host_seam_demotes_before_the_bytes_are_freed(self):
        """STRUCTURAL pin, and labelled as such.

        ``evict_host`` frees the host rows the storage write reads, so the
        demotion must precede it. The live loop needs a real tree and real
        pools, which this hermetic suite has no way to build -- so the order is
        pinned at the source. Weaker than behavioural, and better than the
        nothing that let a mutation swap the two lines undetected."""
        import pathlib
        import re

        import sglang.srt.mem_cache.hiradix_cache as hr

        text = pathlib.Path(hr.__file__).read_text()
        body = text[text.index("    def evict_host(") :]
        body = body[: body.index("\n    def ", 1)]
        demote_at = body.index("demote_before_host_evict")
        free_at = body.index("cache_controller.evict_host(")
        self.assertLess(
            demote_at,
            free_at,
            "the demotion hook runs AFTER evict_host frees the host rows; it "
            "would then read freed bytes or skip a node with no host_value",
        )
        self.assertIsNotNone(re.search(r"demote_before_host_evict\(self, x\)", body))

    def test_backpressure_drops_and_counts_instead_of_queueing(self):
        """Eviction runs under memory pressure: queueing without limit is how
        that becomes an OOM. Dropping is a later miss, never corruption."""
        self._enable(cap="2")
        tree = _FakeTree(self.stages[0])
        tree.ongoing_backup = {1: object(), 2: object()}  # at the cap
        self.assertFalse(self._evict_the_prefix(tree, 0))
        self.assertEqual(demotion.stats().dropped_backpressure, 1)
        self.assertEqual(demotion.stats().demoted, 0)
        self.assertEqual(tree.writes, [])

    def test_a_node_without_a_key_or_bytes_is_not_a_candidate(self):
        self._enable()
        tree = _FakeTree(self.stages[0])
        no_bytes = _Node(KEY, None)
        no_key = _Node(None, _payload(self.stages[0].canonical_kv_page))
        no_key.hash_value = None
        self.assertFalse(demotion.demote(tree, no_bytes, reason="t"))
        self.assertFalse(demotion.demote(tree, no_key, reason="t"))
        self.assertEqual(demotion.stats().skipped_not_persistable, 2)

    def test_without_a_storage_tier_it_is_a_no_op(self):
        self._enable()
        tree = _FakeTree(self.stages[0], enable_storage=False)
        self.assertFalse(self._evict_the_prefix(tree, 0))
        self.assertEqual(demotion.stats().skipped_no_storage, 1)

    def test_a_failing_write_never_escapes_the_eviction_path(self):
        """A raise here would turn 'the cache lost an entry' into 'the request
        failed', inside the allocator's pressure path."""
        self._enable()

        class _Hostile(_FakeTree):
            def write_backup_storage(self, node, backup_len=None):
                raise RuntimeError("disk gone")

        tree = _Hostile(self.stages[0])
        self.assertFalse(self._evict_the_prefix(tree, 0))
        self.assertEqual(demotion.stats().failed, 1)

    def test_the_device_evict_seam_uses_the_same_bound(self):
        self._enable(cap="1")
        tree = _FakeTree(self.stages[0])
        node = _Node(KEY, _payload(self.stages[0].canonical_kv_page))
        self.assertTrue(demotion.demote_on_device_evict(tree, node))
        tree.ongoing_backup = {1: object()}
        self.assertFalse(demotion.demote_on_device_evict(tree, node))
        self.assertEqual(demotion.stats().dropped_backpressure, 1)


if __name__ == "__main__":
    unittest.main()
