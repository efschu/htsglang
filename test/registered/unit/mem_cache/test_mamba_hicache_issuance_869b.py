# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#869b: the mamba state has to come out of HiCache the way the KV does.

TWO DEFECTS, MEASURED BEFORE THEY WERE FIXED, one per half of this file.

(1) ISSUANCE. `store_presence_pages` is the ONE gate that decides whether a
    storage re-fetch is issued (#950). Its docstring claimed it asked "the
    identical question with the identical helper" as the real fetch. It did
    not: the fetch asks `batch_exists_v2` WITH the tree's component transfers,
    which min-clamps the usable KV prefix to the last page that also carries a
    mamba anchor; the probe asked plain `batch_exists`, which sees KV pages
    alone. Measured on the real file backend with four KV pages present and the
    anchor only at page 1: the probe answered 4, the fetch could use 2. The
    gate admitted a fetch whose recurrent half could not land at the promised
    depth, the KV bytes arrived, and the match walk then refused the node for
    want of a state -- the #873 census reading `refusers=MambaComponent` on 671
    of 675 walks WITH the KV bytes present.

    The honest answer is the capped one, and it is what the store already
    computes: prefix usable up to the last anchor, the remainder re-prefilled.
    That bounded loss is what the "no double prefill" law permits; a KV-only
    promise that later collapses to a zeroed match is not.

(2) GEOMETRY. `MambaPoolHost.set_from_flat_data_page` reshapes each slice of an
    incoming blob to `tensor.shape` -- the READER's shape -- so a blob written
    by a differently-shaped mamba pool was silently reinterpreted. On a
    phase-flip boot the PP and TP stacks hold different geometry on both the
    layer axis and the shard axis, and the host page inherits whichever pool
    built it. Making mamba come back out of HiCache is precisely what makes a
    PP-written blob reachable by a TP-shaped reader, so this ticket had to
    close that direction in the same change.

Both halves run hermetically: no GPU, no server, real backend, real method
bodies (the stubs below BORROW the production functions rather than imitating
them).
"""

import tempfile
import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MambaBlobGeometryError,
    MambaPoolHost,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGES = 4


def _backend(tmpdir):
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test/869b",
    )
    be = HiCacheFile(cfg, file_path=tmpdir)
    be.file_path = tmpdir
    return be


def _write(be, key, component=None, nbytes=64):
    stem = be._get_component_key(key, component)
    path = be._sharded_path(stem)
    be._ensure_shard_dir(path)
    with open(path, "wb") as fh:
        fh.write(b"\x00" * nbytes)


class _Probe:
    """Carries the REAL methods under test, with the minimum around them.

    `store_presence_pages` and `_presence_pool_transfers` are the production
    functions, bound to this object. Nothing here re-implements them, so a
    regression in either shows up as a failure rather than as a stub that
    quietly kept agreeing with the test.
    """

    store_presence_pages = HiCacheController.store_presence_pages

    def _presence_pool_transfers(self):
        # Resolved at CALL time, not at class-definition time, and that is
        # load-bearing for the red-first proof: binding it as a class attribute
        # made the module fail to COLLECT when the fix was reverted, so every
        # arm went red for an import reason and none of them measured the
        # behaviour they name. A collection error is indistinguishable from a
        # genuine failure in a summary count.
        return HiCacheController._presence_pool_transfers(self)

    def __init__(self, backend, hashes):
        self.storage_backend = backend
        self.page_size = 1
        self._hashes = hashes

    def get_hash_str(self, token_ids, last_hash, page_size=None):
        return list(self._hashes)


class TestPresenceProbeCoversBothComponents(unittest.TestCase):
    """The one issuance decision must cover KV AND the mamba anchor."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="t869b_")
        self.addCleanup(lambda: None)
        self.keys = [f"page{i}" for i in range(PAGES)]
        self.be = _backend(self.tmp)
        for k in self.keys:
            _write(self.be, k)
        self.be.register_mem_host_pool_v2(object(), PoolName.MAMBA)

    def _probe(self):
        return _Probe(self.be, self.keys).store_presence_pages([1, 2, 3, 4], None)

    def test_capped_to_last_anchor_not_to_the_kv_tail(self):
        """Anchor at page 1 of 4 -> the gate must answer 2, not 4.

        THE RED ARM. Before the fix this returned 4 (plain `batch_exists`),
        which is the KV-only promise that later collapses into a refused match.
        """
        _write(self.be, self.keys[1], PoolName.MAMBA)
        self.assertEqual(self._probe(), 2)

    def test_no_anchor_anywhere_is_never_a_full_match_lie(self):
        """KV fully present, no mamba blob at all -> 0, never PAGES.

        THE COUNTER-ARM the ticket demands: without a covering anchor the
        answer must not be a full match. Before the fix this returned 4.
        """
        self.assertEqual(self._probe(), 0)

    def test_anchor_at_the_tail_still_reports_the_full_prefix(self):
        """No regression: when the anchor DOES cover the tail, nothing is lost."""
        _write(self.be, self.keys[PAGES - 1], PoolName.MAMBA)
        self.assertEqual(self._probe(), PAGES)

    def test_loss_is_bounded_by_the_anchor_spacing(self):
        """The capped answer is the last anchor, not the first.

        Two anchors present (pages 0 and 2): the gate must report the DEEPER
        one, so the re-prefill costs at most the span since the last anchor.
        Reporting page 0 would be safe-but-wasteful and would silently turn one
        interval of loss into the whole prefix.
        """
        _write(self.be, self.keys[0], PoolName.MAMBA)
        _write(self.be, self.keys[2], PoolName.MAMBA)
        self.assertEqual(self._probe(), 3)


class TestPresenceProbeFallback(unittest.TestCase):
    """A backend that cannot be asked must not be read as an answer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="t869bf_")
        self.keys = [f"page{i}" for i in range(PAGES)]
        self.be = _backend(self.tmp)
        for k in self.keys:
            _write(self.be, k)

    def test_no_mamba_pool_registered_keeps_the_pre_869b_call(self):
        """Non-hybrid deployments are bit-for-bit unchanged."""
        probe = _Probe(self.be, self.keys)
        self.assertIsNone(probe._presence_pool_transfers())
        self.assertEqual(probe.store_presence_pages([1, 2, 3, 4], None), PAGES)

    def test_backend_without_v2_falls_back_rather_than_reporting_absent(self):
        """`NotImplementedError` -> the KV-only answer, not 0.

        Reporting 0 would decline every fetch on a backend that simply lacks
        the v2 interface. Reporting the KV-only count is the pre-#869b
        behaviour, and the code says so in the log rather than silently.
        """

        class _NoV2:
            registered_pools = {PoolName.MAMBA: object()}

            def batch_exists_v2(self, *a, **k):
                raise NotImplementedError

            def batch_exists(self, keys, extra_info=None):
                return len(keys)

        probe = _Probe(_NoV2(), self.keys)
        self.assertEqual(probe.store_presence_pages([1, 2, 3, 4], None), PAGES)

    def test_probe_transfer_matches_the_component_it_stands_in_for(self):
        """The rebuilt transfer must be the one MambaComponent would build."""
        self.be.register_mem_host_pool_v2(object(), PoolName.MAMBA)
        (transfer,) = _Probe(self.be, self.keys)._presence_pool_transfers()
        self.assertEqual(transfer.name, PoolName.MAMBA)
        self.assertEqual(transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(len(transfer.keys), 1)


class _GeometryReader:
    """Real `set_from_flat_data_page`, minimal pool state around it.

    The production method is borrowed, not imitated -- the guard under test is
    the shipped one.
    """

    set_from_flat_data_page = MambaPoolHost.set_from_flat_data_page

    def __init__(self, num_layers, temporal_shape, conv_shapes):
        self.page_size = 1
        self.num_mamba_layers = num_layers
        self.temporal_state_shape = torch.Size(temporal_shape)
        self.conv_state_shapes = [torch.Size(s) for s in conv_shapes]
        self._tensors = [
            torch.zeros((num_layers,) + tuple(temporal_shape), dtype=torch.uint8)
        ] + [
            torch.zeros((num_layers,) + tuple(s), dtype=torch.uint8)
            for s in conv_shapes
        ]
        self.size_per_token = sum(
            t.numel() * t.element_size() for t in self._tensors
        )

    def _iter_page_tensors(self, index):
        return iter(self._tensors)

    def page_bytes(self):
        return self.page_size * self.size_per_token


class TestMambaBlobGeometryIsRefusedNotReinterpreted(unittest.TestCase):
    """The danger direction: a foreign-geometry blob must never be accepted."""

    def _tp_shaped(self):
        # "TP" stack: all layers, this rank's narrow shard.
        return _GeometryReader(16, (4, 2), [(4, 2)])

    def _pp_shaped(self):
        # "PP" stack: this stage's layers only, full width. BOTH axes differ.
        return _GeometryReader(7, (8, 2), [(8, 2)])

    def test_own_geometry_round_trips(self):
        """The green arm: a correctly shaped blob is still accepted."""
        pool = self._tp_shaped()
        blob = torch.arange(pool.page_bytes(), dtype=torch.uint8) % 251
        pool.set_from_flat_data_page(0, blob)
        self.assertEqual(int(pool._tensors[0].reshape(-1)[0]), int(blob[0]))

    def test_foreign_geometry_blob_is_refused(self):
        """THE MUTANT THIS TICKET REQUIRES: wrong state must go red.

        A blob written by the other phase's mamba pool is offered to a
        TP-shaped reader. Before the guard, the reader sliced it against its
        OWN shapes and copied whatever landed -- the request would then decode
        from another phase's recurrent state. That is the worst failure form
        available here, so it is the one pinned by a test.
        """
        writer = self._pp_shaped()
        reader = self._tp_shaped()
        self.assertNotEqual(writer.page_bytes(), reader.page_bytes())
        foreign = torch.zeros(writer.page_bytes(), dtype=torch.uint8)
        with self.assertRaises(MambaBlobGeometryError):
            reader.set_from_flat_data_page(0, foreign)

    def test_refusal_names_the_geometry_it_expected(self):
        """The message has to be actionable in a boot log, not just an abort."""
        reader = self._tp_shaped()
        foreign = torch.zeros(reader.page_bytes() + 8, dtype=torch.uint8)
        with self.assertRaises(MambaBlobGeometryError) as ctx:
            reader.set_from_flat_data_page(0, foreign)
        msg = str(ctx.exception)
        self.assertIn("num_mamba_layers=16", msg)
        self.assertIn(str(reader.page_bytes()), msg)

    def test_a_short_blob_is_refused_before_it_can_half_write_a_slot(self):
        """Undersized input must not partially fill the slot."""
        reader = self._tp_shaped()
        short = torch.zeros(reader.page_bytes() - 8, dtype=torch.uint8)
        with self.assertRaises(MambaBlobGeometryError):
            reader.set_from_flat_data_page(0, short)
        self.assertEqual(int(reader._tensors[0].abs().sum()), 0)


class TestPerNodeGranularityIsTheDefault(unittest.TestCase):
    """The grid is a throttle, never a design assumption (user order, 28.08).

    "der mamba muss auch aus dem hicache kommen, so wie der kv" -- and the
    follow-up: do NOT build around the 8192 checkpoint interval, write the
    mamba state per NODE the way KV pages are written, however often that is.

    These pin that the per-node route is what the default configuration
    already takes, so nothing downstream may quietly reintroduce a grid
    assumption. `--mamba-checkpoint-interval` defaults to None
    (`server_args.py`), and at None every grid test in the write path is the
    identity -- the three gates below are the ones that would otherwise make
    an anchor sparse:

      * `commit_insert_component_data` (the insert backstop),
      * `prepare_for_caching_req` (both retention branches),
      * `create_match_validator` / `is_resume_candidate` (the read rule).

    All three route through `is_on_interval`, so pinning it at None pins all
    three at once -- and that shared routing is itself the #747 property that
    keeps the two match lineages from drifting.
    """

    def test_every_position_is_a_legal_anchor_without_an_interval(self):
        from sglang.srt.mem_cache.mamba_ckpt_utils import is_on_interval

        # Deliberately including the #873 census depths (45-49), the ones that
        # a grid of 8192 makes structurally impossible to anchor.
        for pos in (1, 45, 46, 47, 48, 49, 4095, 8191, 12345):
            self.assertTrue(is_on_interval(pos, None), f"position {pos}")

    def test_a_present_state_off_any_grid_is_a_resume_candidate(self):
        from sglang.srt.mem_cache.mamba_ckpt_utils import is_resume_candidate

        self.assertTrue(
            is_resume_candidate(47, None, has_device_value=True),
            "a node carrying state at token 47 must be resumable with no grid",
        )

    def test_a_host_only_state_is_resumable_which_is_the_whole_ticket(self):
        """The HiCache half: an evicted anchor with a host copy still matches."""
        from sglang.srt.mem_cache.mamba_ckpt_utils import is_resume_candidate

        self.assertTrue(
            is_resume_candidate(
                47,
                None,
                has_device_value=False,
                has_host_value=True,
                device_only=False,
            )
        )

    def test_absence_is_reported_as_absent_not_as_a_grid_refusal(self):
        """Without a grid the only refusal left is a genuinely stateless node.

        This is the distinction #913 exists to draw: ABSENT sends a reader to
        the write side, OFF_GRID to the read-side policy. With the grid off,
        OFF_GRID must be unreachable -- otherwise a reader is sent to a policy
        that is not in play.
        """
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            RESUME_REFUSAL_ABSENT,
            resume_refusal_reason,
        )

        self.assertEqual(
            resume_refusal_reason(47, None, has_device_value=False),
            RESUME_REFUSAL_ABSENT,
        )
        self.assertIsNone(
            resume_refusal_reason(47, None, has_device_value=True),
            "a stated position can only be refused by a grid, and there is none",
        )


if __name__ == "__main__":
    unittest.main()
