"""#1206 / R1: the transfer domain is the ENTRIES' KEY SPACE, not a count.

WHAT IS BROKEN, AND WHERE
-------------------------
The host->device restore loop's iteration domain is a boot-frozen COUNT
(``transfer_layer_num``) while the pool mappings are keyed by GLOBAL layer
id (``memory_pool.py:1932``
``self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}``).
``_make_layer_mapper`` (``hybrid_pool_assembler.py:44-53``) bounds a GLOBAL id
by that COUNT at ``:49`` ``if not 0 <= layer_id < transfer_layer_num:``, so on
a PP stage whose ids start above the count the mapper answers ``None`` for
EVERY key it holds: with ``tln=18`` and keys ``32..49`` (PP1) nothing is
restored at all, and the tree still calls the prefix resident.

The counter that the restore loop drives is a SECOND consumer of the same
count, and the two halves of the tree wait on it in two different index
spaces: ``HybridReqToTokenPool._wait_for_mamba_layer`` waits on the GLOBAL id
(``memory_pool.py:2001``) while ``HybridLinearKVPool._wait_for_layer`` waits on
``self.local_slot(layer_id)`` (``:5004``). One counter, two index spaces.

WHAT THESE TESTS PIN (spec rows T-1, T-2, T-2b, T-3, T-10, T-11, T-12)
----------------------------------------------------------------------
* T-1  -- a ``PoolEntry`` keeps its MAPPING and can answer for every key it
         holds; the group's domain is ``1 + max(key)`` and equals the device
         side's key space (the ``owned == driven`` predicate at group level).
* T-2  -- the group's per-layer load visits every mapped layer of EVERY entry
         exactly once, anchor included.
* T-2b -- the draft tier's own bound is MEMBERSHIP in its key space, not a
         count compared against a global id.
* T-3  -- the driven domain follows the rebind, because the loop reads it
         through ``self.mem_pool_host`` and ``_stamp`` moves that attribute.
* T-10 -- the index a consumer waits on IS the index at which the producer
         completed that layer's copy. This is the wrong-answer hazard the
         wider domain CREATES if only one of the two halves is landed.
* T-11 -- the mamba wait's bound is the counter's own width.
* T-12 -- the counter is resized to EXACTLY the driven domain, in place, and
         never grow-only: an unrecorded event queries True, so a counter wider
         than the domain acks a load whose copies have not landed.

Hermetic: CPU only, no CUDA. ``device_module.Event()`` constructs and
``query()`` answers without a device; ``record()`` does not, so no test here
records one.
"""

import ast
import inspect
import textwrap
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.hicache_phase_binding import PhasePools, _stamp
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

# Qwen3.8-27B, 64 layers, ``full_attention_interval = 4``: full attention sits
# at [3, 7, 11, ..., 63] and every other layer is GDN (48 GDN + 16 full).
# Config: /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8/config.json
# The three PP stages of this boot are [0,32), [32,50) and [50,64); each has
# its own test method below rather than a subTest loop, so a FAILED name line
# and the summary count cannot disagree.


def _stage_maps(start, end):
    """The two device maps for one PP stage, built as the device pools build
    them: GLOBAL layer id -> dense local index (``memory_pool.py:1932`` for
    mamba, ``:4841-4843`` for KV)."""
    full_ids = [i for i in range(start, end) if i % 4 == 3]
    mamba_ids = [i for i in range(start, end) if i % 4 != 3]
    full_map = {layer_id: i for i, layer_id in enumerate(full_ids)}
    mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_ids)}
    return full_map, mamba_map


def _host_pool_stub(layer_num=4):
    pool = MagicMock()
    pool.page_size = 1
    pool.layout = "layer_first"
    pool.device = "cpu"
    pool.size = 16
    pool.can_use_write_back_jit = False
    pool.layer_num = layer_num
    return pool


def _group_for_stage(start, end):
    full_map, mamba_map = _stage_maps(start, end)
    anchor = PoolEntry(
        name=PoolName.KV,
        host_pool=_host_pool_stub(len(full_map)),
        device_pool=MagicMock(),
        layer_mapping=full_map,
        is_primary_index_anchor=True,
    )
    extra = PoolEntry(
        name=PoolName.MAMBA,
        host_pool=_host_pool_stub(len(mamba_map)),
        device_pool=MagicMock(),
        layer_mapping=mamba_map,
    )
    return HostPoolGroup([anchor, extra]), full_map, mamba_map


class TestPoolEntryKeySpace(unittest.TestCase):
    """T-1."""

    def _check_stage(self, start, end):
        group, full_map, mamba_map = _group_for_stage(start, end)

        # (a) per entry: the entry answers for every key it holds.
        for entry in group.entries:
            for key in entry.layer_mapping:
                self.assertIsNotNone(
                    entry.local_layer(key),
                    f"{entry.name} cannot answer for its own key {key}",
                )

        # (b) owned == driven at group level: the DEVICE maps' key space
        # against the HOST group's entries' key space. Two objects, one kind.
        owned = set(mamba_map) | set(full_map)
        driven = set().union(*(set(e.layer_mapping) for e in group.entries))
        self.assertEqual(owned, driven)

        # (c) the same equality one level up, scalar against scalar.
        self.assertEqual(group.transfer_layer_domain, 1 + max(owned))

    def test_pool_entry_key_space_is_inside_its_transfer_domain_pp0(self):
        self._check_stage(0, 32)

    def test_pool_entry_key_space_is_inside_its_transfer_domain_pp1(self):
        self._check_stage(32, 50)

    def test_pool_entry_key_space_is_inside_its_transfer_domain_pp2(self):
        self._check_stage(50, 64)

    def test_an_empty_group_has_a_zero_domain(self):
        """The ``max()`` of an empty key set is not an exception."""
        entry = PoolEntry(
            name=PoolName.KV,
            host_pool=_host_pool_stub(0),
            device_pool=MagicMock(),
            layer_mapping={},
            is_primary_index_anchor=True,
        )
        self.assertEqual(HostPoolGroup([entry]).transfer_layer_domain, 0)


class TestGroupLoadCoverage(unittest.TestCase):
    """T-2."""

    def _check_stage(self, start, end):
        group, full_map, mamba_map = _group_for_stage(start, end)
        seen = {PoolName.KV: [], PoolName.MAMBA: []}

        def _record(name):
            def load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                seen[name].append(layer_id)

            return load_to_device_per_layer

        for entry in group.entries:
            entry.host_pool.load_to_device_per_layer = _record(entry.name)

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([1], dtype=torch.int64),
            device_indices=torch.tensor([2], dtype=torch.int64),
        )
        for i in range(group.transfer_layer_domain):
            group.load_to_device_per_layer(
                MagicMock(),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                i,
                "direct",
                pool_transfers=[transfer],
            )

        self.assertEqual(sorted(seen[PoolName.KV]), sorted(full_map.values()))
        self.assertEqual(sorted(seen[PoolName.MAMBA]), sorted(mamba_map.values()))

    def test_group_load_visits_every_mapped_layer_exactly_once_pp0(self):
        self._check_stage(0, 32)

    def test_group_load_visits_every_mapped_layer_exactly_once_pp1(self):
        self._check_stage(32, 50)

    def test_group_load_visits_every_mapped_layer_exactly_once_pp2(self):
        self._check_stage(50, 64)


class TestDraftTierDomain(unittest.TestCase):
    """T-2b."""

    def test_draft_tier_loop_uses_a_domain_not_a_frozen_count(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            _host_pool_covers_layer,
        )

        covered = {0: 0, 2: 1, 5: 2}
        draft = types.SimpleNamespace(
            entries=[types.SimpleNamespace(layer_mapping=covered)],
            layer_num=len(covered),
        )
        visited = [i for i in range(8) if _host_pool_covers_layer(draft, i)]
        self.assertEqual(visited, [0, 2, 5])

        # The defect, stated as an inequality between the two candidates: the
        # frozen count and the key space disagree on this tier, so a count
        # bound visits 1 (uncovered) and skips 5 (covered).
        frozen = [i for i in range(8) if i < draft.layer_num]
        self.assertNotEqual(visited, frozen)

    def test_a_plain_draft_tier_is_dense_from_zero(self):
        """A non-composite host pool: membership IS ``range(layer_num)``, so
        the conversion is byte-identical there."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            _host_pool_covers_layer,
        )

        plain = types.SimpleNamespace(layer_num=4)
        self.assertEqual(
            [i for i in range(8) if _host_pool_covers_layer(plain, i)],
            [0, 1, 2, 3],
        )


class TestDomainFollowsRebind(unittest.TestCase):
    """T-3."""

    def test_controller_layer_domain_follows_the_rebind(self):
        boot_group, _, _ = _group_for_stage(32, 50)
        tp_full = {i: i for i in range(64) if i % 4 == 3}
        tp_mamba = {i: i for i in range(64) if i % 4 != 3}
        tp_anchor = PoolEntry(
            name=PoolName.KV,
            host_pool=_host_pool_stub(len(tp_full)),
            device_pool=MagicMock(),
            layer_mapping=tp_full,
            is_primary_index_anchor=True,
        )
        tp_extra = PoolEntry(
            name=PoolName.MAMBA,
            host_pool=_host_pool_stub(len(tp_mamba)),
            device_pool=MagicMock(),
            layer_mapping=tp_mamba,
        )
        tp_group = HostPoolGroup([tp_anchor, tp_extra])

        controller = types.SimpleNamespace(mem_pool_host=boot_group)
        self.assertEqual(controller.mem_pool_host.transfer_layer_domain, 50)

        _stamp(
            controller,
            PhasePools(
                phase="tp",
                device_pool=None,
                allocator=None,
                host_pool=tp_group,
            ),
            7,
        )
        self.assertEqual(
            set(range(controller.mem_pool_host.transfer_layer_domain)),
            set(range(64)),
        )

    def test_the_loop_reads_the_domain_through_the_swapped_attribute(self):
        """The half that makes the rebind assertion load-bearing: a domain
        cached on the CONTROLLER would survive the swap above and still be
        wrong. The loop must read it through ``self.mem_pool_host``."""
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(HybridCacheController.start_loading))
        )
        bounds = [
            node.iter
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and getattr(node.iter.func, "id", None) == "range"
        ]
        self.assertEqual(len(bounds), 1)
        arg = bounds[0].args[0]
        self.assertEqual(ast.unparse(arg), "self.mem_pool_host.transfer_layer_domain")


class TestCounterIndexSpace(unittest.TestCase):
    """T-10 and T-11."""

    @staticmethod
    def _recording_counter():
        counter = types.SimpleNamespace(completed=[], waited=[], num_layers=50)
        counter.wait_until = counter.waited.append
        return counter

    def test_layer_done_counter_index_space_matches_its_waiters(self):
        full_map, mamba_map = _stage_maps(32, 50)
        domain = 1 + max(set(full_map) | set(mamba_map))
        counter = self._recording_counter()

        # PRODUCER: the loop completes step i for every i of the domain.
        completed_at = {}
        for i in range(domain):
            completed_at[i] = i

        # CONSUMER, KV half: the real method, on a stand-in that owns this
        # stage's full-attention ids.
        kv = types.SimpleNamespace(
            layer_transfer_counter=counter,
            local_slot=lambda layer_id: full_map[layer_id],
        )
        for global_id in full_map:
            HybridLinearKVPool._wait_for_layer(kv, global_id)

        # CONSUMER, mamba half: the real method.
        mamba = types.SimpleNamespace(
            layer_transfer_counter=counter,
            _mamba_transfer_frame=len(full_map) + len(mamba_map),
        )
        for global_id in mamba_map:
            HybridReqToTokenPool._wait_for_mamba_layer(mamba, global_id)

        self.assertEqual(
            counter.waited,
            [completed_at[g] for g in list(full_map) + list(mamba_map)],
        )
        # The finish sentinel is load_events[-1], so the last index of the
        # domain must be one the producer actually completes.
        self.assertIn(domain - 1, completed_at)

    def test_mamba_layer_wait_bound_is_the_counter_width(self):
        counter = self._recording_counter()
        counter.num_layers = 64
        pool = types.SimpleNamespace(
            layer_transfer_counter=counter,
            _mamba_transfer_frame=18,
        )
        HybridReqToTokenPool._wait_for_mamba_layer(pool, 49)
        self.assertEqual(counter.waited, [49])

    def test_no_counter_or_no_frame_is_still_a_silent_no_wait(self):
        """The ``None`` semantics ``swa_memory_pool`` relies on: the presence
        test survives; only the BOUND moves to the counter's width."""
        counter = self._recording_counter()
        no_counter = types.SimpleNamespace(
            layer_transfer_counter=None, _mamba_transfer_frame=18
        )
        HybridReqToTokenPool._wait_for_mamba_layer(no_counter, 3)
        no_frame = types.SimpleNamespace(
            layer_transfer_counter=counter, _mamba_transfer_frame=None
        )
        HybridReqToTokenPool._wait_for_mamba_layer(no_frame, 3)
        self.assertEqual(counter.waited, [])


class TestCounterResize(unittest.TestCase):
    """T-12."""

    def test_counter_width_equals_the_driven_domain(self):
        counter = LayerDoneCounter(18)
        counter_id = id(counter)
        event_ids = [id(e) for e in counter.events]

        counter.resize(64)
        self.assertEqual(counter.num_layers, 64)

        counter.resize(50)
        self.assertEqual(counter.num_layers, 50)
        self.assertEqual(id(counter), counter_id)
        self.assertEqual([id(e) for e in counter.events], event_ids)
        for event in counter.events:
            self.assertEqual(len(event.load_events), 50)
            self.assertEqual(event._num_layers, 50)
            # DANGER DIRECTION: a grow-only resize leaves the finish sentinel
            # on load_events[63], an event the 50-domain loop never records.
            # An unrecorded event queries True, so the ACK fires before the
            # copies land.
            self.assertIs(event.finish_event, event.load_events[49])
            self.assertTrue(event.finish_event.query())


if __name__ == "__main__":
    unittest.main()
