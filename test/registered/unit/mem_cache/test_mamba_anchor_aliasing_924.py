"""#924 SIBLING: a mamba slot returned to the free list while a tree node still names it.

THE DEFECT THE NAMED #924 GUARD CANNOT SEE. ``MambaSlotAllocator._refuse_double_free``
(``allocator/mamba.py``) catches a slot returned TWICE -- a duplicate in a free
list that is a bare ``torch.cat``. This is the sibling: nothing is returned
twice. The slot is returned ONCE, and a radix node goes on referencing it as a
resume anchor. The free list and the tree then both own it, ``alloc()`` hands it
to the next request, and the tree keeps offering it -- one GDN state read by two
requests, which is a wrong answer that never raises.

MEASURED, boot 10 (``/spinning/evidence-665-f1/boot_855_weg1b10_2126a4a1d2_0904_211702.log``,
21:29:28Z, all three ranks, the raise that killed the instance)::

    [mamba] total=20, available=20, evictable=4, withheld=0,
            double_owned_src=live, free_list_duplicates=0,
            duplicate_slot_ids=None, free_and_cached=4

``free_list_duplicates=0`` is why the #924 instrument stayed silent, and
``free_and_cached=4`` with ``available == total`` is the signature these tests
pin: every slot on the free list, four of them still named by tree nodes
(``TREE CENSUS nodes=5 | MAMBA: tracked_evictable=4 recomputed_evictable=4``,
same log line 364223).

ROOT, at file:line: ``UnifiedRadixCache.reclaim_rows_for_drop``
(``unified_radix_cache.py``) walks every node of the tree and returns the rows
and mamba slots it still holds -- the #1050 contract, "rows the drop returns
that ``evict`` could not, because the node was locked" -- but it returned them
without ever taking the reference away from the node. Every other release of a
node-held anchor in this file goes through ``MambaComponent.evict_component``,
which frees AND nulls ``cd.value`` AND corrects the size book; this one pass had
only the first third of that. The fix factors the other two thirds out as
``UnifiedRadixCache._disown_reclaimed_value`` and calls it on both halves.

THE CLASS, and the sibling sweep: "a component's rows are released by a walker
over tree nodes instead of through the component's own owner-transfer
primitive". Both halves of this function had it -- the FULL half frees through
``_free_full`` and left ``node.component_data[FULL].value`` standing too -- so
both are fixed and both are pinned below. The future check is the post-condition
these tests assert directly: after any pass that returns rows the tree held,
``free_set & tree_set`` must be empty for every component.

THE TESTS DRIVE THE REAL FUNCTIONS against a real ``UnifiedRadixCache`` and a
real ``MambaSlotAllocator`` on CPU, and assert on the ledger the boot died on --
deliberately not on the source text.
"""

import unittest
from array import array
from dataclasses import dataclass

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


MAMBA_SLOTS = 20
NUM_LAYERS = 24
FULL_LAYER_IDS = (3, 7, 11, 15, 19, 23)
NON_FULL_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in set(FULL_LAYER_IDS)]


@dataclass
class _Fixture:
    cache: UnifiedRadixCache
    allocator: TokenToKVPoolAllocator
    pool: HybridReqToTokenPool


def _build_cpu_fixture() -> _Fixture:
    """A FULL+MAMBA ``UnifiedRadixCache`` on CPU.

    Its own fixture rather than an import from
    ``test_unified_radix_cache_unittest``: that module resolves its device with
    ``get_device()``, which raises on a box with no accelerator, and these tests
    must run at the desk with ``CUDA_VISIBLE_DEVICES=""``. Everything else
    mirrors that module's ``build_fixture`` for the plain no_buffer / no-int8
    configuration -- which is the one boot 10 ran
    (``mamba_radix_cache_strategy='no_buffer'``,
    ``enable_int8_mamba_checkpoint=False``).
    """
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)

    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=256,
            n_groups=1,
            num_heads=2,
            head_dim=16,
            state_size=16,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=NON_FULL_LAYER_IDS)

    pool = HybridReqToTokenPool(
        size=10,
        mamba_size=MAMBA_SLOTS,
        mamba_spec_state_size=10,
        max_context_len=512,
        device="cpu",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=NON_FULL_LAYER_IDS,
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=256,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS),
        device="cpu",
        enable_memory_saver=False,
        mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=256,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False,
        enable_kv_cache_events=False,
        eviction_policy="lru",
        is_eagle=False,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return _Fixture(cache=cache, allocator=allocator, pool=pool)


def _run_one_request(fx: _Fixture, i: int, *, lock: bool = False) -> None:
    """Prompt -> ``cache_unfinished_req`` -> output -> ``cache_finished_req``.

    The production shape of a served request, and the one that leaves TWO mamba
    anchors in the tree: the unfinished insert donates a freshly allocated slot,
    the finished insert donates the request's own (relinquished) one. Boot 10
    shows exactly two ``#969H BACKUP mamba_value=has_value`` lines per finished
    request, which is that pair.
    """
    prompt = list(range(1000 * (i + 1), 1000 * (i + 1) + 8))
    out = list(range(5000 + 100 * i, 5000 + 100 * i + 4))
    sp = SamplingParams(temperature=0, max_new_tokens=1)
    req = Req(
        rid=f"aliasing-{i}",
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=sp,
    )
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", prompt)
    req.set_extend_range(0, len(prompt))
    kv = fx.allocator.alloc(len(prompt))
    fx.pool.write((req.req_pool_idx, slice(0, len(prompt))), kv)
    req.kv_committed_len = len(prompt)
    req.last_node = fx.cache.root_node
    req.cache_protected_len = 0
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.mamba_last_track_seqlen = len(prompt)
    fx.cache.cache_unfinished_req(req)

    total = len(prompt) + len(out)
    req.output_ids = array("q", out)
    req.full_untruncated_fill_ids = array("q", prompt + out)
    kv2 = fx.allocator.alloc(len(out))
    fx.pool.write((req.req_pool_idx, slice(len(prompt), total)), kv2)
    req.kv_committed_len = total
    req.set_extend_range(0, total)
    req.mamba_last_track_seqlen = total
    fx.cache.cache_finished_req(req, is_insert=True)
    if lock:
        # The #1050 premise: the drop's own `evict` REFUSES a locked node, so
        # the reclaim is the only thing that can return its rows.
        fx.cache.inc_lock_ref(req.last_node)
    fx.pool.free(req)


def _mamba_free_and_cached(fx: _Fixture) -> list:
    free = {int(v) for v in fx.pool.mamba_allocator.free_slots.tolist()}
    cached = {int(v) for v in fx.cache.all_mamba_values_flatten().tolist()}
    return sorted(free & cached)


def _full_free_and_cached(fx: _Fixture) -> list:
    free = set(fx.allocator.free_pages.tolist()) | set(
        fx.allocator.release_pages.tolist()
    )
    cached = set(fx.cache.all_values_flatten().tolist())
    return sorted(free & cached)


class TheReclaimMustNotLeaveAnAliasedAnchor(CustomTestCase):
    """RED before the fix, on the exact ledger line boot 10 died on."""

    def test_reclaim_leaves_no_mamba_slot_both_free_and_tree_held(self):
        fx = _build_cpu_fixture()
        for i in range(2):
            _run_one_request(fx, i)

        held_before = fx.cache.mamba_evictable_size()
        self.assertGreater(
            held_before, 0, "fixture precondition: the tree must hold anchors"
        )

        report = fx.cache.reclaim_rows_for_drop()
        self.assertTrue(report["reclaimed"], report)
        self.assertGreater(report["mamba_slots"], 0, report)

        self.assertEqual(
            _mamba_free_and_cached(fx),
            [],
            "a slot the reclaim returned is still named by a tree node: "
            "alloc() will hand it to the next request while the tree offers "
            "it as a resume anchor (boot 10, free_and_cached=4)",
        )

    def test_reclaim_reproduces_the_boot10_ledger_line_when_broken(self):
        """The three terms of the killer line, asserted together.

        ``available == total`` AND ``evictable > 0`` AND
        ``free_list_duplicates == 0`` is the whole boot-10 signature; asserting
        only ``free_and_cached`` would also pass for a build that leaked the
        slots instead of aliasing them.
        """
        fx = _build_cpu_fixture()
        for i in range(2):
            _run_one_request(fx, i)
        fx.cache.reclaim_rows_for_drop()

        available = fx.pool.mamba_allocator.available_size()
        evictable = fx.cache.mamba_evictable_size()
        free_list = [int(v) for v in fx.pool.mamba_allocator.free_slots.tolist()]

        self.assertEqual(
            len(free_list),
            len(set(free_list)),
            "free list must stay duplicate-free (this is NOT the #924 shape)",
        )
        self.assertLessEqual(
            available + evictable,
            fx.pool.mamba_pool.size,
            f"available({available}) + evictable({evictable}) exceeds the "
            f"{fx.pool.mamba_pool.size}-slot pool: that surplus IS the "
            "negative `mamba usage` boot 10 printed five times before dying",
        )

    def test_reclaim_leaves_no_kv_row_both_free_and_tree_held(self):
        """SIBLING SWEEP, same function, other component.

        The FULL half of ``reclaim_rows_for_drop`` frees through
        ``_free_full`` and had the identical omission. It is pinned here so the
        class cannot come back on the pool the boot happened not to die on.
        """
        fx = _build_cpu_fixture()
        for i in range(2):
            _run_one_request(fx, i)
        fx.cache.reclaim_rows_for_drop()

        self.assertEqual(
            _full_free_and_cached(fx),
            [],
            "a KV row the reclaim returned is still named by a tree node",
        )

    def test_reclaim_of_a_locked_node_keeps_the_size_book_straight(self):
        """A LOCKED node's rows are counted PROTECTED, not evictable.

        The reclaim exists for locked nodes (``evict`` refuses them), so
        disowning one must debit ``component_protected_size_``. Debiting the
        evictable book instead balances one term by breaking another, and
        ``sanity_check`` PART 4 is what would say so.
        """
        fx = _build_cpu_fixture()
        _run_one_request(fx, 0, lock=True)
        self.assertGreater(fx.cache.mamba_protected_size(), 0)

        fx.cache.reclaim_rows_for_drop()

        self.assertEqual(_mamba_free_and_cached(fx), [])
        self.assertEqual(fx.cache.mamba_protected_size(), 0)
        self.assertEqual(fx.cache.mamba_evictable_size(), 0)
        self.assertEqual(fx.cache.full_protected_size(), 0)
        self.assertEqual(fx.cache.full_evictable_size(), 0)

    def test_reclaim_is_idempotent(self):
        """A second reclaim must find nothing and free nothing.

        Before the fix the second pass saw the same values again and reported
        them as ``mamba_already_free`` -- the ledger difference (#1055) held,
        but only because the aliasing was still there to be re-detected.
        """
        fx = _build_cpu_fixture()
        for i in range(2):
            _run_one_request(fx, i)
        fx.cache.reclaim_rows_for_drop()
        second = fx.cache.reclaim_rows_for_drop()

        self.assertEqual(second["mamba_held"], 0, second)
        self.assertEqual(second["full_held"], 0, second)
        self.assertEqual(second["mamba_slots"], 0, second)
        self.assertEqual(second["reason"], "tree held nothing", second)


class TheAliasingMustBeANamedStop(CustomTestCase):
    """#924 named STOP: a negative mamba occupancy is a verdict, not a log line.

    Boot 10 printed ``mamba usage: -0.10 ... -0.30`` five times over eight
    minutes and nothing acted on it. The reading is rank-uniform and derivable
    from terms the idle invariant already holds, so it is raised THERE -- on
    every rank, with nothing in flight -- and never from a batch line inside the
    no-return region between collectives (#969 §W3).
    """

    @staticmethod
    def _checker(fx):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        class _Observer:
            @staticmethod
            def session_held_mamba_slots():
                return 0

        checker = SchedulerInvariantChecker.__new__(SchedulerInvariantChecker)
        checker.req_to_token_pool = fx.pool
        checker.tree_cache = fx.cache
        checker.pool_stats_observer = _Observer()
        checker.server_args = ServerArgs(model_path="dummy", page_size=1)
        checker.token_to_kv_pool_allocator = fx.allocator
        checker.get_token_to_kv_pool_allocator = None
        return checker

    @staticmethod
    def _stats(available, evictable):
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            PoolStats,
        )

        return PoolStats(
            full_num_used=0,
            full_token_usage=0.0,
            full_available_size=0,
            full_evictable_size=0,
            is_hybrid_ssm=True,
            mamba_num_used=MAMBA_SLOTS - (available + evictable),
            mamba_usage=0.0,
            mamba_available_size=available,
            mamba_evictable_size=evictable,
        )

    def test_negative_occupancy_is_a_leak_verdict(self):
        fx = _build_cpu_fixture()
        checker = self._checker(fx)
        leak, msg = checker._check_mamba_pool(
            self._stats(MAMBA_SLOTS, 4)  # boot 10: available == total, 4 held
        )
        self.assertTrue(leak, msg)
        self.assertIn("#924 MAMBA SLOT ALIASING", msg)
        self.assertIn("mamba_num_used=-4", msg)

    def test_a_free_list_longer_than_the_pool_is_a_leak_verdict(self):
        fx = _build_cpu_fixture()
        checker = self._checker(fx)
        leak, msg = checker._check_mamba_pool(self._stats(MAMBA_SLOTS + 3, 0))
        self.assertTrue(leak, msg)
        self.assertIn("#924 MAMBA SLOT ALIASING", msg)

    def test_a_balanced_pool_is_not_a_verdict(self):
        """The can-fail proof's other half: the STOP must stay silent when the
        ledger balances, or it says nothing when it fires."""
        fx = _build_cpu_fixture()
        checker = self._checker(fx)
        leak, msg = checker._check_mamba_pool(self._stats(MAMBA_SLOTS, 0))
        self.assertFalse(leak, msg)
        self.assertNotIn("#924 MAMBA SLOT ALIASING", msg)


class TheDiscriminatorMustBeBoundedAndNameTheSlot(CustomTestCase):
    """#924D: the per-request slot trail Boot 11 decides #1190 with.

    One line per (rid, station), never per step. The cap is asserted because a
    trail that grows with load is a trail that gets turned off.
    """

    def test_one_line_per_rid_and_station(self):
        from sglang.srt.mem_cache.allocator import mamba as mamba_alloc

        with self.assertLogs(mamba_alloc.__name__, level="INFO") as captured:
            for _ in range(5):
                mamba_alloc.note_924d("alloc", rid="probe-A", slot=torch.tensor([7]))
            mamba_alloc.note_924d("free", rid="probe-A", slot=torch.tensor([7]))
            mamba_alloc.note_924d("alloc", rid="probe-B", slot=torch.tensor([9]))

            # No rid: the subject falls back to the node, so two different
            # nodes are two lines -- `write_backup` carries no request.
            mamba_alloc.note_924d("backup", node_id=41, slot=torch.tensor([3]))
            mamba_alloc.note_924d("backup", node_id=41, slot=torch.tensor([3]))
            mamba_alloc.note_924d("backup", node_id=42, slot=torch.tensor([4]))

        lines = [line for line in captured.output if "#924D" in line]
        self.assertEqual(len(lines), 5, lines)
        self.assertEqual(
            len([line for line in lines if "station=backup" in line]), 2, lines
        )
        self.assertTrue(any("rid=node41" in line for line in lines))
        self.assertTrue(any("station=alloc" in line and "probe-A" in line for line in lines))
        self.assertTrue(any("station=free" in line and "probe-A" in line for line in lines))
        self.assertTrue(any("mamba_slot=[7]" in line for line in lines))

    def test_it_never_raises_on_an_unreadable_argument(self):
        from sglang.srt.mem_cache.allocator import mamba as mamba_alloc

        class _Hostile:
            def reshape(self, *_a, **_k):
                raise RuntimeError("no")

            def __str__(self):
                raise RuntimeError("no")

        mamba_alloc.note_924d("alloc", rid=_Hostile(), slot=_Hostile())


if __name__ == "__main__":
    unittest.main()
