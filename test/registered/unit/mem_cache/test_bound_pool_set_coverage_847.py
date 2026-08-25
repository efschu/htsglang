"""#718/#847: a phase rebind narrowed the bound host tier, and the consumers split.

SPECIMEN (boot_w38b_0825_1722.log, 17:25:59, all three PP ranks):

    hybrid_cache_controller.py:583  start_loading
    hybrid_cache_controller.py:729  move_hybrid_indices
    cache_controller.py:1217        device_indices = device_indices.cpu()
    AttributeError: 'NoneType' object has no attribute 'cpu'

CHAIN. At 17:25:53 the controller was rebound to the 'tp' pools (generation 3;
`[#719 hicache-rebind] rebound 3 reader(s)`), and the TP host tier is built at
`phase_flip_boot.py:2019-2032` with EXACTLY ONE entry, ``PoolName.KV`` -- on a
model whose boot-time tier was ``pools=KV + MAMBA``. Six seconds later a mamba
state that lived only on the host was matched (``MAMBA-HOST-RESUME``), so
``load_back`` built a ``PoolTransfer(name=MAMBA, host_indices=...)`` with
``device_indices`` unset by contract, and
``_resolve_pool_transfers_allocation`` -- whose contract is "auto-alloc where
they are None" -- hit ``entry_map.get(MAMBA) is None`` and ``continue``d,
returning the transfer UNRESOLVED inside the same list as the resolved ones.

ROOT, and it is structural: ``check_shapes`` guards the rebind by comparing ONE
SCALAR (layer count). A KV-only group passes that check while dropping the MAMBA
pool entirely, so the invariant that was actually violated -- the incoming tier
must describe every pool the outgoing one did -- was never asked about.

THE CLASS, not the instance: a narrower binding installed under a scalar guard,
with the consumers splitting into two kinds --

  * THE ONES THAT CRASH -- ``move_indices`` dereferences the unresolved index
                 set (this specimen), and its write-side twin dies one line
                 later at ``cache_controller.py:1218`` on ``host_indices.sort()``.
  * SILENT SKIPPERS -- ``HostPoolGroup.load_to_device_per_layer`` skips a
                 transfer whose pool it does not know, so the KV moves and the
                 recurrent state does NOT, while the tree reports the prefix
                 resident. That is a wrong ANSWER, which is worse than the
                 crash, and it is LIVE (not merely latent) on the
                 ``kernel``/``page_first``/write-back-JIT path, which bypasses
                 ``move_hybrid_indices`` entirely
                 (``hybrid_cache_controller.py:468-475``) and so would never
                 have produced the crash that exposed this.
  * LEAKERS   -- the two host-slot release paths drop an unknown pool's slots
                 instead of freeing them.

REFUSE, NEVER SKIP. Skipping moves the KV while the state stays behind and the
tree calls the prefix resident: a wrong answer. Refusing costs a recompute,
which is merely slow -- the same trade the tree already makes at
``mamba_component.py:986-991`` (mamba slot starvation -> re-prefill the
segment) and at ``hicache_phase_guard`` (a refused prefetch is a miss now).

Hermetic: mocks only, no CUDA.
"""

import logging
import types
import unittest
from queue import Queue
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_phase_binding import (
    PhasePools,
    RebindRefused,
    check_pool_coverage,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup


def _entry(name, *, anchor=False, alloc_result=None):
    """A PoolEntry stand-in with the two callables the resolver reaches for."""
    entry = MagicMock()
    entry.name = name
    entry.is_primary_index_anchor = anchor
    entry.host_pool.page_size = 1
    entry.host_pool.layout = "layer_first"
    entry.host_pool.device = "cpu"
    entry.host_pool.size = 16
    entry.host_pool.can_use_write_back_jit = False
    entry.host_pool.layer_num = 4
    entry.device_alloc_fn = MagicMock(return_value=alloc_result)
    entry.device_free_fn = MagicMock()
    entry.device_evict_fn = None
    entry.host_evict_fn = None
    entry.layer_mapper = lambda layer_id: layer_id
    return entry


def _controller_with(entry_map):
    """A `self` carrying only what `_resolve_pool_transfers_allocation` touches.

    The refusal helper is BOUND from the real class rather than stubbed: its
    rate limit is part of what is under test.
    """
    ctrl = types.SimpleNamespace()
    ctrl.mem_pool_host = types.SimpleNamespace(entry_map=entry_map)
    ctrl._refuse_unresolvable_transfer = types.MethodType(
        HybridCacheController._refuse_unresolvable_transfer, ctrl
    )
    return ctrl


def _resolve(ctrl, pools, alloc_host=False, **kw):
    return HybridCacheController._resolve_pool_transfers_allocation(
        ctrl, pools, alloc_host, **kw
    )


class TestResolverRefusesUnknownPool(unittest.TestCase):
    """(a) The resolver must refuse, not hand back a half-resolved list."""

    def test_unknown_pool_returns_none_instead_of_unresolved_transfer(self):
        # The bound tier is the post-cutover KV-only one; the transfer is the
        # mamba load-back the tree just asked for.
        ctrl = _controller_with({PoolName.KV: _entry(PoolName.KV, anchor=True)})
        mamba = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([7], dtype=torch.int64),
        )

        out = _resolve(ctrl, [mamba])

        self.assertIsNone(
            out,
            "a pool the bound tier cannot describe must refuse the whole "
            "transfer set; returning the list leaves device_indices=None for "
            "move_indices to dereference",
        )
        self.assertIsNone(mamba.device_indices)

    def test_refusal_rolls_back_slots_already_allocated(self):
        # SWA resolves first and MAMBA is unknown: the SWA slots must come back,
        # or every refused load-back leaks device slots.
        swa_slots = torch.tensor([3], dtype=torch.int64)
        swa_entry = _entry(PoolName.SWA, alloc_result=swa_slots)
        ctrl = _controller_with(
            {PoolName.KV: _entry(PoolName.KV, anchor=True), PoolName.SWA: swa_entry}
        )
        swa = PoolTransfer(
            name=PoolName.SWA, host_indices=torch.tensor([1], dtype=torch.int64)
        )
        mamba = PoolTransfer(
            name=PoolName.MAMBA, host_indices=torch.tensor([2], dtype=torch.int64)
        )

        out = _resolve(ctrl, [swa, mamba])

        self.assertIsNone(out)
        swa_entry.device_free_fn.assert_called_once()
        self.assertIsNone(swa.device_indices, "rollback must un-set what it freed")

    def test_unknown_pool_refuses_before_allocating_anything_else(self):
        """The early branch is not merely redundant with the post-condition.

        The post-condition is a backstop and would also catch this, but only
        AFTER every later pool has been allocated and rolled back. Under load
        that churn runs on the device allocator once per scheduler tick for as
        long as the state sits on the host. The refusal belongs at the first
        pool that cannot be resolved.
        """
        swa_entry = _entry(
            PoolName.SWA, alloc_result=torch.tensor([3], dtype=torch.int64)
        )
        ctrl = _controller_with(
            {PoolName.KV: _entry(PoolName.KV, anchor=True), PoolName.SWA: swa_entry}
        )
        unknown = PoolTransfer(
            name=PoolName.MAMBA, host_indices=torch.tensor([2], dtype=torch.int64)
        )
        later = PoolTransfer(
            name=PoolName.SWA, host_indices=torch.tensor([1], dtype=torch.int64)
        )

        self.assertIsNone(_resolve(ctrl, [unknown, later]))
        swa_entry.device_alloc_fn.assert_not_called()

    def test_known_pool_still_resolves(self):
        """The control: refusing must not break the path that works."""
        slots = torch.tensor([5], dtype=torch.int64)
        ctrl = _controller_with(
            {
                PoolName.KV: _entry(PoolName.KV, anchor=True),
                PoolName.MAMBA: _entry(PoolName.MAMBA, alloc_result=slots),
            }
        )
        mamba = PoolTransfer(
            name=PoolName.MAMBA, host_indices=torch.tensor([9], dtype=torch.int64)
        )

        out = _resolve(ctrl, [mamba])

        self.assertIsNotNone(out)
        self.assertIs(out[0].device_indices, slots)

    def test_both_index_sets_missing_is_refused(self):
        """Nothing to allocate FROM and nothing to move: refuse, do not carry."""
        ctrl = _controller_with(
            {
                PoolName.KV: _entry(PoolName.KV, anchor=True),
                PoolName.MAMBA: _entry(PoolName.MAMBA, alloc_result=None),
            }
        )
        self.assertIsNone(_resolve(ctrl, [PoolTransfer(name=PoolName.MAMBA)]))

    def test_post_condition_catches_a_derived_transfer_left_unresolved(self):
        """(d) The check that turns a boot-second-20 crash into a unit test.

        A DERIVED transfer never enters the allocation loop -- it copies its
        source's index sets -- so no per-branch refusal covers it. If the source
        it copies is itself unresolved, only the post-condition stands between
        that None and `move_indices`. This input reaches the post-condition and
        nothing else, which is what makes it a test OF the post-condition.
        """
        ctrl = _controller_with({PoolName.KV: _entry(PoolName.KV, anchor=True)})
        sidecar = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)

        out = _resolve(
            ctrl,
            [sidecar],
            kv_host_indices=torch.tensor([1], dtype=torch.int64),
            kv_device_indices=None,  # the anchor never resolved
        )

        self.assertIsNone(
            out,
            "a returned list must contain no None index set; the derived pool "
            "copied a None and no earlier branch inspects it",
        )

    def test_refusal_log_is_rate_limited(self):
        """The residual: match_prefix re-derives the hit every tick, so a
        refused load-back repeats. It must not print forever (the 449 MB flood
        class)."""
        ctrl = _controller_with({PoolName.KV: _entry(PoolName.KV, anchor=True)})
        with self.assertLogs(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
            level=logging.ERROR,
        ) as caught:
            for _ in range(40):
                _resolve(
                    ctrl,
                    [
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            host_indices=torch.tensor([1], dtype=torch.int64),
                        )
                    ],
                )
        self.assertLess(
            len(caught.output),
            40,
            "40 refusals must not produce 40 log lines",
        )
        self.assertGreaterEqual(len(caught.output), 1, "and must not go silent")

    def test_write_side_twin_refuses_too(self):
        """The same `continue`, reached from write(): host stays None and the
        write path dies at cache_controller.py:1218 on host_indices.sort()."""
        ctrl = _controller_with({PoolName.KV: _entry(PoolName.KV, anchor=True)})
        backup = PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=torch.tensor([4], dtype=torch.int64),
        )

        out = _resolve(ctrl, [backup], alloc_host=True)

        self.assertIsNone(out)
        self.assertIsNone(backup.host_indices)


class TestRebindPoolCoverage(unittest.TestCase):
    """(b) check_shapes compared a scalar where the invariant is structural."""

    def _reader(self, names):
        return types.SimpleNamespace(
            mem_pool_host=types.SimpleNamespace(entry_map={n: object() for n in names})
        )

    def _incoming(self, names):
        return PhasePools(
            phase="tp",
            device_pool=types.SimpleNamespace(layer_num=16),
            host_pool=types.SimpleNamespace(
                layer_num=16, entry_map={n: object() for n in names}
            ),
            allocator=object(),
        )

    def test_narrowing_rebind_is_refused(self):
        readers = {"controller": self._reader([PoolName.KV, PoolName.MAMBA])}
        with self.assertRaises(RebindRefused) as ctx:
            check_pool_coverage(readers, self._incoming([PoolName.KV]))
        self.assertIn("mamba", str(ctx.exception).lower())

    def test_equal_pool_set_is_admitted(self):
        readers = {"controller": self._reader([PoolName.KV, PoolName.MAMBA])}
        check_pool_coverage(
            readers, self._incoming([PoolName.KV, PoolName.MAMBA])
        )  # must not raise

    def test_wider_pool_set_is_admitted(self):
        readers = {"controller": self._reader([PoolName.KV])}
        check_pool_coverage(readers, self._incoming([PoolName.KV, PoolName.MAMBA]))

    def test_rebind_itself_refuses_a_narrowing_tier(self):
        """The wiring, not just the helper: a check nobody calls is inert."""
        from sglang.srt.mem_cache import hicache_phase_binding as binding

        readers = {"controller": self._reader([PoolName.KV, PoolName.MAMBA])}
        before = binding.binding_state().generation
        with self.assertRaises(RebindRefused):
            binding.rebind(readers, self._incoming([PoolName.KV]))
        self.assertEqual(
            binding.binding_state().generation,
            before,
            "a refused rebind must not mint a generation",
        )

    def test_reader_without_a_host_tier_is_not_invented(self):
        """A reader that names no host pool (the scheduler names the allocator
        only) must not be turned into a coverage claim."""
        readers = {"scheduler": types.SimpleNamespace()}
        check_pool_coverage(readers, self._incoming([PoolName.KV]))


class TestExecutorRefusesUnknownPool(unittest.TestCase):
    """(c) The silent skipper -- the site that would HIDE the bug."""

    def _group(self):
        anchor = _entry(PoolName.KV, anchor=True)
        return HostPoolGroup([anchor])

    def test_load_raises_on_a_pool_the_tier_cannot_describe(self):
        group = self._group()
        stray = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([1], dtype=torch.int64),
            device_indices=torch.tensor([2], dtype=torch.int64),
        )
        with self.assertRaises(ValueError) as ctx:
            group.load_to_device_per_layer(
                MagicMock(),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                0,
                "direct",
                pool_transfers=[stray],
            )
        self.assertIn("mamba", str(ctx.exception).lower())

    def test_backup_raises_on_a_pool_the_tier_cannot_describe(self):
        group = self._group()
        stray = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([1], dtype=torch.int64),
            device_indices=torch.tensor([2], dtype=torch.int64),
        )
        with self.assertRaises(ValueError):
            group.backup_from_device_all_layer(
                MagicMock(),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                "direct",
                pool_transfers=[stray],
            )

    def test_unmapped_layer_is_still_skipped(self):
        """The control: a layer this pool does not cover is a legitimate skip
        and must stay one."""
        anchor = _entry(PoolName.KV, anchor=True)
        extra = _entry(PoolName.MAMBA)
        extra.layer_mapper = lambda layer_id: None
        group = HostPoolGroup([anchor, extra])
        group.load_to_device_per_layer(
            MagicMock(),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            0,
            "direct",
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.tensor([1], dtype=torch.int64),
                    device_indices=torch.tensor([2], dtype=torch.int64),
                )
            ],
        )
        extra.host_pool.load_to_device_per_layer.assert_not_called()


class TestReleasePathsDoNotLeak(unittest.TestCase):
    """(c) Both leak paths: an unknown pool's host slots must not vanish."""

    def _controller(self, entry_map, owning):
        ctrl = types.SimpleNamespace()
        ctrl.mem_pool_host = types.SimpleNamespace(entry_map=entry_map, page_size=1)
        ctrl.host_mem_release_queue = Queue()
        ctrl.extra_host_mem_release_queues = {PoolName.MAMBA: Queue()}
        ctrl.extra_host_mem_release_entries = owning
        for name in (
            "_append_host_mem_release_pages",
            "entry_for_extra_release",
        ):
            setattr(
                ctrl, name, types.MethodType(getattr(HybridCacheController, name), ctrl)
            )
        return ctrl

    def test_release_uses_the_owning_entry_after_a_narrowing_rebind(self):
        owning_entry = _entry(PoolName.MAMBA)
        ctrl = self._controller(
            {PoolName.KV: _entry(PoolName.KV, anchor=True)},
            {PoolName.MAMBA: owning_entry},
        )
        HybridCacheController.append_host_mem_release(
            ctrl,
            extra_pools=[
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.tensor([11], dtype=torch.int64),
                )
            ],
        )
        self.assertEqual(
            ctrl.extra_host_mem_release_queues[PoolName.MAMBA].qsize(),
            1,
            "the slots must reach the release queue even though the currently "
            "bound tier no longer names the pool",
        )


if __name__ == "__main__":
    unittest.main()
