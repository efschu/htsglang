"""#703: does a warm prefix survive a phase flip?

Today it does not, and the reason is not the key format (that was #706) but
TIMING. This suite is written red-first around that: the same flip-shaped
sequence is run twice, once without the hook and once with it, and the only
difference is whether the prefix is findable afterwards.

THE SHAPE BEING SIMULATED, and why a fake is honest here. A flip is:

  1. requests park at a quiescent seam;
  2. [the hook's slot -- nothing occupies it today];
  3. cutover: the scheduler swaps to the other stack, whose device KV pool is
     a DIFFERENT pool (the TP stack builds with pp_size=1 and spans all 16
     attention layers; a PP stage spans 7/5/4);
  4. serving resumes in the new geometry.

``HiRadixCache`` binds its host pool to the pool that existed when it was
built -- the boot one -- and the scheduler allocator is likewise assigned once.
So step 3 leaves the host tier describing a pool that is no longer live, and a
prefix's only route across is the geometry-free store. Building the real pools
needs a GPU; this suite is hermetic (CUDA_VISIBLE_DEVICES=""). What it does
instead is exercise the REAL canonical store through the REAL file backend on
both sides of the seam, with the tree/controller replaced by fakes whose API
mirrors the parts the hook actually calls (``write_backup``,
``writing_check``, ``ongoing_backup``, the local ack drain). The store, the
keys and the bytes are real; the scheduling around them is not.

What each test pins:

* ``test_without_the_hook_the_prefix_is_gone`` -- the red case. Under the
  write_back policy nothing stages a warm prefix, so the store is empty at the
  seam and the post-flip lookup misses. This is today's behaviour.
* ``test_with_the_hook_the_prefix_survives`` -- the same sequence with the hook
  in slot 2: the post-flip geometry reads the prefix back, byte for byte.
* the refusals -- no storage, or storage without the canonical format -- which
  are the difference between "retention is off" and "retention looks on and
  silently does nothing".
* the deadline -- a backup that never acknowledges must not hold the flip.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_flip_writeback import (
    FlipWritebackRefused,
    canonical_store_of,
    flip_writeback,
    maybe_flip_writeback,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_CUT = [(0, 28), (28, 48), (48, 64)]
IDENTITY = "0123456789abcdef"
PREFIX_KEYS = ["warm01", "warm02", "warm03"]


def _kv_window(lo, hi):
    return window_for_layers(
        SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _backend(root, *, pp_rank, pp_size, tp_rank, tp_size, window, dcp=False):
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
    """The parts of a radix TreeNode the hook reads."""

    def __init__(self, key, hash_value):
        self.key = key
        self.hash_value = [hash_value]
        self.host_value = None
        self.children = {}

    @property
    def backuped(self):
        return self.host_value is not None


class _FakeTree:
    """A radix cache with the write-back policy's timing, and nothing else.

    Mirrors what the hook touches: staging is a no-op until someone asks for
    it (that IS the write_back policy), the storage stage runs when a staged
    node is checked, and acknowledgements arrive through a drain.
    """

    def __init__(self, backend, window, *, ack=True):
        self.cache_controller = type("_CC", (), {"storage_backend": backend})()
        self.enable_storage = True
        self.root_node = _Node(key=[], hash_value=None)
        self.root_node.hash_value = None
        self._backend = backend
        self._window = window
        self.ongoing_backup = {}
        self._pending_storage = []
        self._ack = ack
        self.writing_checks = 0
        for i, key in enumerate(PREFIX_KEYS):
            node = _Node(key=[i], hash_value=key)
            self.root_node.children[i] = node

    # -- the API the hook calls ---------------------------------------------
    def write_backup(self, node, write_back=False):
        if node.host_value is not None:
            return 0
        node.host_value = _payload(self._window)
        return 1

    def writing_check(self, write_back=False):
        """Device->host acks land, and each one starts a storage backup --
        the real class does exactly this in _finish_write_through_ack."""
        self.writing_checks += 1
        for node in self.root_node.children.values():
            if node.host_value is None or node in self._pending_storage:
                continue
            op_id = len(self.ongoing_backup) + len(self._pending_storage) + 1
            self.ongoing_backup[op_id] = node
            self._pending_storage.append(node)

    def _drain_storage_control_queues_local(self):
        if not self._ack:
            return  # a backup that never acknowledges
        for op_id, node in list(self.ongoing_backup.items()):
            self._backend.set(node.hash_value[0], node.host_value)
            self.ongoing_backup.pop(op_id, None)


class TestFlipWriteback(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        # Pre-flip: the PP prefill phase, this rank owning stage 0's layers.
        self.pp_window = _kv_window(*PP_CUT[0])
        self.pre = _backend(
            self.root, pp_rank=0, pp_size=3, tp_rank=0, tp_size=1, window=self.pp_window
        )

    def _complete_the_page_from_the_other_stages(self, key):
        """The other two PP stages deposit their slots, as they would live."""
        for rank in (1, 2):
            stage = _backend(
                self.root,
                pp_rank=rank,
                pp_size=3,
                tp_rank=0,
                tp_size=1,
                window=_kv_window(*PP_CUT[rank]),
            )
            stage.set(key, _payload(stage.canonical_kv_page))

    def _post_flip_reader(self):
        """After the cutover: the TP decode phase, all 16 slots, new geometry.

        A different backend object on purpose -- the flip does not carry the
        host tier, so the only thing shared with the pre-flip side is the
        store on disk.
        """
        return _backend(
            self.root,
            pp_rank=0,
            pp_size=1,
            tp_rank=1,
            tp_size=3,
            dcp=True,
            window=window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS),
        )

    # -- red first ----------------------------------------------------------

    def test_without_the_hook_the_prefix_is_gone(self):
        """Today's behaviour. Under write_back nothing stages a warm prefix,
        the flip does not touch HiCache, and the seam passes with the store
        still empty -- so the new geometry has nothing to find."""
        tree = _FakeTree(self.pre, self.pp_window)
        # (no hook here: this is slot 2 left empty, as in the live code)
        for key in PREFIX_KEYS:
            self._complete_the_page_from_the_other_stages(key)

        reader = self._post_flip_reader()
        for key in PREFIX_KEYS:
            self.assertFalse(reader.exists(key))
        self.assertEqual(len(tree.ongoing_backup), 0)

    def test_with_the_hook_the_prefix_survives(self):
        tree = _FakeTree(self.pre, self.pp_window)
        report = flip_writeback(tree, deadline_s=1.0)
        self.assertEqual(report.eligible, len(PREFIX_KEYS))
        self.assertEqual(report.staged, len(PREFIX_KEYS))
        self.assertTrue(report.complete)

        for key in PREFIX_KEYS:
            self._complete_the_page_from_the_other_stages(key)

        reader = self._post_flip_reader()
        expected = _payload(reader.canonical_kv_page)
        for key in PREFIX_KEYS:
            self.assertTrue(reader.exists(key))
            out = torch.zeros(reader.canonical_kv_page.byte_length, dtype=torch.uint8)
            self.assertIsNotNone(reader.get(key, out))
            self.assertTrue(torch.equal(out, expected))

    def test_the_hook_is_what_made_the_difference(self):
        """Same store, same keys, same stages: the ONLY difference between the
        two cases above is whether slot 2 was occupied."""
        tree = _FakeTree(self.pre, self.pp_window)
        self.assertEqual(len([f for f in os.listdir(self.root)]), 0)
        flip_writeback(tree, deadline_s=1.0)
        self.assertGreater(len([f for f in os.listdir(self.root)]), 0)

    # -- the refusals -------------------------------------------------------

    def test_refused_without_a_storage_tier(self):
        tree = _FakeTree(self.pre, self.pp_window)
        tree.enable_storage = False
        with self.assertRaises(FlipWritebackRefused) as cm:
            flip_writeback(tree, deadline_s=1.0)
        self.assertIn("nowhere", str(cm.exception))

    def test_refused_without_the_canonical_format(self):
        """The refusal that matters most: a store whose keys still carry this
        phase's geometry would take the IO at the seam and deliver a page the
        other phase cannot name."""
        plain = _backend(
            self.root, pp_rank=0, pp_size=3, tp_rank=0, tp_size=1, window=None
        )
        tree = _FakeTree(plain, self.pp_window)
        self.assertIsNone(canonical_store_of(tree))
        with self.assertRaises(FlipWritebackRefused) as cm:
            flip_writeback(tree, deadline_s=1.0)
        self.assertIn("canonical", str(cm.exception))

    # -- the bound ----------------------------------------------------------

    def test_a_backup_that_never_acknowledges_does_not_hold_the_flip(self):
        """The deadline is a hard bound. An unbounded wait here parks the
        instance with requests already parked -- the #630 wedge shape."""
        tree = _FakeTree(self.pre, self.pp_window, ack=False)
        clock = {"t": 0.0}

        def now():
            return clock["t"]

        def sleep(dt):
            clock["t"] += dt

        report = flip_writeback(tree, deadline_s=0.05, now=now, sleep=sleep)
        self.assertFalse(report.complete)
        self.assertEqual(report.outstanding, len(PREFIX_KEYS))
        self.assertLessEqual(report.elapsed_s, 0.2)

    def test_already_staged_nodes_are_not_rewritten(self):
        tree = _FakeTree(self.pre, self.pp_window)
        first = flip_writeback(tree, deadline_s=1.0)
        second = flip_writeback(tree, deadline_s=1.0)
        self.assertEqual(first.staged, len(PREFIX_KEYS))
        self.assertEqual(second.staged, 0)
        self.assertEqual(second.already_staged, len(PREFIX_KEYS))

    # -- the gate -----------------------------------------------------------

    def test_gate_is_off_by_default(self):
        class _Args:
            phase_flip_writeback = False

        class _Sched:
            server_args = _Args()
            tree_cache = None

        self.assertIsNone(maybe_flip_writeback(_Sched()))

    def test_no_scheduler_is_a_safe_no_op(self):
        """The call site passes whatever handle it has. A missing one must be a
        no-op, not an exception inside a cutover."""
        self.assertIsNone(maybe_flip_writeback(None))

    def test_gate_on_runs_the_hook(self):
        tree = _FakeTree(self.pre, self.pp_window)

        class _Args:
            phase_flip_writeback = True
            phase_flip_writeback_deadline_s = 1.0

        class _Sched:
            server_args = _Args()
            tree_cache = tree

        report = maybe_flip_writeback(_Sched())
        self.assertIsNotNone(report)
        self.assertTrue(report.complete)


if __name__ == "__main__":
    unittest.main()
