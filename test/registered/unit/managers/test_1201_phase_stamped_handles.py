# SPDX-License-Identifier: Apache-2.0
"""#1201 -- a handle stamped with the boot phase, consumed by the other phase.

CLASS.  ``ReqToTokenPool`` is not the scheduler's alone.  Four holders cache a
reference to it, all of them at CONSTRUCTION:

  * ``UnifiedRadixCache.req_to_token_pool``   (unified_radix_cache.py:429)
  * the pool's own back-reference, ``tree_cache`` (memory_pool.py:1871), set
    only by ``bind_tree_cache`` -- whose two callers are both tree-cache
    constructors (unified_radix_cache.py:514, mamba_radix_cache.py:502)
  * the pool's ``layer_transfer_counter`` (memory_pool.py:1876), set only by
    ``register_layer_transfer_counter`` from the assembler, at boot
  * ``FutureMap.pool`` (overlap_utils.py:339), built once in ``init_overlap``

``phase_req_pool_binding.rebind_req_pool_for_cutover`` (:180) moves exactly ONE
of them: the scheduler's.  From the first cutover the other three go on naming
the OUTGOING phase's pool, and because both pools hold the same number of rows
the divergence does not fail -- it lands IN RANGE on the wrong pool.  This is
the hazard ``kv_session_offload.py:2921-2924`` names in its own docstring and
answers with a read-at-use property; that answer is what this suite pins for
the tree cache.

RANK-UNIFORM, which is why no ballot, digest or MIN can catch it: every rank is
wrong the same way.

THE DANGEROUS DIRECTION IS THE SILENT ONE.  The free path has a loud half and a
silent half, and the silent one runs FIRST:

  * loud   -- ``common.py:1849`` ``tree_cache.req_to_token_pool.free(req)``
              reaches ``ReqToTokenPool.free_slot`` (memory_pool.py:468), whose
              double-return guard (:492-497) refuses a row that is already free
              on the outgoing pool;
  * silent -- ``common.py:1836``
              ``tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx]``
              reads the WRONG pool's tensor and hands whatever it finds to
              ``token_to_kv_pool_allocator.free()``.

So the tests that matter most here assert the READ lands on the pool that
minted the row, not merely that some guard fires.

Hermetic: CPU tensors, no accelerator, no scheduler process.
"""

from __future__ import annotations

import types
import unittest

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase


class _HandleHoldingPool(ReqToTokenPool):
    """A pool carrying the two back-references ``HybridReqToTokenPool`` has.

    The production type is ``HybridReqToTokenPool`` (memory_pool.py:1871/1876),
    which cannot be built without a mamba pool.  What the rebind actually uses
    is the DUCK-TYPED surface -- ``bind_tree_cache`` and
    ``register_layer_transfer_counter`` -- so that is what this pins.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tree_cache = None
        self.layer_transfer_counter = None
        self._mamba_transfer_frame = None

    def bind_tree_cache(self, tree_cache) -> None:
        self.tree_cache = tree_cache

    def register_layer_transfer_counter(
        self, layer_transfer_counter, mamba_transfer_frame=None
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter
        self._mamba_transfer_frame = mamba_transfer_frame


def _pool(size=4, ctx=8, cls=ReqToTokenPool):
    return cls(
        size=size, max_context_len=ctx, device="cpu", enable_memory_saver=False
    )


def _cache(pool):
    return UnifiedRadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )


def _scheduler(pp_pool, tp_pool, tree_cache=None):
    """Minimal stand-in carrying exactly what the rebind walks."""
    pp_runner = types.SimpleNamespace(req_to_token_pool=pp_pool)
    tp_runner = types.SimpleNamespace(req_to_token_pool=tp_pool)
    return types.SimpleNamespace(
        req_to_token_pool=pp_pool,
        tree_cache=tree_cache,
        tp_worker=types.SimpleNamespace(model_runner=pp_runner),
        phase_flip_stacks=types.SimpleNamespace(
            tp_worker=types.SimpleNamespace(model_runner=tp_runner)
        ),
        server_args=types.SimpleNamespace(phase_flip_rebind_hicache=False),
        running_batch=None,
        waiting_queue=[],
    )


def _rebind(scheduler, phase):
    from sglang.srt.managers.phase_req_pool_binding import (
        rebind_req_pool_for_cutover,
    )

    return rebind_req_pool_for_cutover(scheduler, phase)


class TestTheTreeCacheReadsThePoolAtUse(CustomTestCase):
    """B1: the cache's handle must be the scheduler's, read at use."""

    def test_an_unowned_cache_still_answers_with_its_constructor_pool(self):
        """No owner registered -> byte-identical to the old behaviour."""
        pp = _pool()
        cache = _cache(pp)
        self.assertIs(cache.req_to_token_pool, pp)

    def test_a_bound_cache_follows_the_owner(self):
        pp, tp = _pool(), _pool()
        cache = _cache(pp)
        sched = _scheduler(pp, tp, tree_cache=cache)
        cache.bind_req_pool_owner(sched)
        self.assertIs(cache.req_to_token_pool, pp)
        sched.req_to_token_pool = tp
        self.assertIs(
            cache.req_to_token_pool,
            tp,
            "the cache cached a reference the cutover moved out from under it",
        )

    def test_the_cutover_leaves_cache_and_scheduler_on_one_pool(self):
        pp, tp = _pool(), _pool()
        cache = _cache(pp)
        sched = _scheduler(pp, tp, tree_cache=cache)
        _rebind(sched, "tp")
        self.assertIs(sched.req_to_token_pool, tp)
        self.assertIs(
            sched.tree_cache.req_to_token_pool,
            sched.req_to_token_pool,
            "#1201: the cutover moved the scheduler and left the tree cache "
            "on the outgoing phase's pool",
        )


class TestTheSilentHalfOfTheFreePath(CustomTestCase):
    """The read at common.py:1836 must land on the pool that minted the row.

    DANGEROUS DIRECTION: this one does not raise when it is wrong.  It returns
    another pool's row contents, in range, and those indices go straight to
    ``token_to_kv_pool_allocator.free()``.
    """

    def test_the_kv_read_sees_the_rows_the_running_phase_wrote(self):
        pp, tp = _pool(), _pool()
        cache = _cache(pp)
        sched = _scheduler(pp, tp, tree_cache=cache)
        _rebind(sched, "tp")

        # What the TP phase writes after the cutover, through the pool the
        # scheduler now owns.
        row = 1
        tp.req_to_token[row, :4] = torch.tensor([11, 12, 13, 14])

        # What `cache_finished_req` reads (common.py:1836), verbatim shape.
        seen = sched.tree_cache.req_to_token_pool.req_to_token[row][:4]
        self.assertTrue(
            torch.equal(seen, torch.tensor([11, 12, 13, 14])),
            f"#1201 SILENT: the free path read {seen.tolist()} from the "
            f"outgoing pool instead of the running phase's rows; those "
            f"indices are handed to token_to_kv_pool_allocator.free()",
        )


class TestTheIncomingPoolGetsItsHandlesBack(CustomTestCase):
    """(b)+(c): every handle the constructor stamped is re-stamped at the seam."""

    def test_the_incoming_pool_is_bound_to_the_tree_cache(self):
        """Without this the pool's ``tree_cache`` stays None for the whole TP
        phase, disarming the evict-then-retry at memory_pool.py:2019-2024 and
        reintroducing the #581/#773 regression that
        unified_radix_cache.py:507-513 was written to close."""
        pp = _pool(cls=_HandleHoldingPool)
        tp = _pool(cls=_HandleHoldingPool)
        cache = _cache(pp)
        sched = _scheduler(pp, tp, tree_cache=cache)
        self.assertIs(pp.tree_cache, cache)  # done at construction
        _rebind(sched, "tp")
        self.assertIs(
            tp.tree_cache,
            cache,
            "#1201: nothing re-registers the tree cache on the incoming "
            "phase's pool, so its REQUIRED mamba allocs cannot evict",
        )

    def test_the_layer_transfer_counter_is_carried_to_the_incoming_pool(self):
        """A join on a pool nothing re-registered is a join that never happens
        (memory_pool.py:1995-1998)."""
        pp = _pool(cls=_HandleHoldingPool)
        tp = _pool(cls=_HandleHoldingPool)
        counter = object()
        pp.register_layer_transfer_counter(counter, 32)
        sched = _scheduler(pp, tp, tree_cache=_cache(pp))
        _rebind(sched, "tp")
        self.assertIs(tp.layer_transfer_counter, counter)
        self.assertEqual(tp._mamba_transfer_frame, 32)

    def test_no_counter_to_carry_is_not_an_error(self):
        """The default path: no hierarchical cache, no counter, on either pool.

        ``layer_transfer_counter is None`` is the NORMAL steady state for every
        boot without a HiCache controller -- ``swa_memory_pool.py:124-125``
        even registers None deliberately -- so an unconditional refusal here
        would refuse the default path.
        """
        pp = _pool(cls=_HandleHoldingPool)
        tp = _pool(cls=_HandleHoldingPool)
        sched = _scheduler(pp, tp, tree_cache=_cache(pp))
        _rebind(sched, "tp")
        self.assertIsNone(tp.layer_transfer_counter)

    def test_a_plain_pool_without_the_hooks_is_not_an_error(self):
        pp, tp = _pool(), _pool()
        sched = _scheduler(pp, tp, tree_cache=_cache(pp))
        _rebind(sched, "tp")
        self.assertIs(sched.req_to_token_pool, tp)


class TestTheSeamRefusesADivergentHandle(CustomTestCase):
    """(d): a fifth holder is a loud stop at the seam, not a wrong answer."""

    def _assert_identity(self, scheduler):
        from sglang.srt.managers.phase_req_pool_binding import (
            assert_req_pool_identity,
        )

        return assert_req_pool_identity(scheduler)

    def test_a_clean_cutover_passes(self):
        pp = _pool(cls=_HandleHoldingPool)
        tp = _pool(cls=_HandleHoldingPool)
        sched = _scheduler(pp, tp, tree_cache=_cache(pp))
        _rebind(sched, "tp")
        self._assert_identity(sched)

    def test_a_cache_left_on_the_outgoing_pool_is_refused(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            ReqPoolRebindRefused,
        )

        pp, tp = _pool(), _pool()
        cache = _cache(pp)
        sched = _scheduler(pp, tp, tree_cache=cache)
        _rebind(sched, "tp")
        # A fifth holder, simulated: something re-points the cache after the
        # rebind (or a new cache is built from the outgoing pool).
        cache.bind_req_pool_owner(None)
        cache.req_to_token_pool = pp
        with self.assertRaises(ReqPoolRebindRefused):
            self._assert_identity(sched)

    def test_a_pool_left_bound_to_nothing_is_refused(self):
        from sglang.srt.managers.phase_req_pool_binding import (
            ReqPoolRebindRefused,
        )

        pp = _pool(cls=_HandleHoldingPool)
        tp = _pool(cls=_HandleHoldingPool)
        sched = _scheduler(pp, tp, tree_cache=_cache(pp))
        _rebind(sched, "tp")
        tp.tree_cache = None
        with self.assertRaises(ReqPoolRebindRefused):
            self._assert_identity(sched)

    def test_the_assertion_is_wired_into_the_cutover(self):
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[4]
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "phase_flip_runtime.py"
        ).read_text()
        self.assertIn(
            "assert_req_pool_identity",
            src,
            "an identity assertion nothing calls is a document",
        )


class TestTheRegistryNamesThePhaseStampedHandles(CustomTestCase):
    """(d) structural: 'what this list forgets, a boot finds'."""

    def test_the_request_pool_is_declared_mutated_state(self):
        from sglang.srt.managers.cutover_participants import (
            MUTATED_STATE,
            ReadWindow,
        )

        self.assertEqual(
            MUTATED_STATE.get("req_to_token_pool"),
            ReadWindow.OUTSIDE_CUTOVER,
        )

    def test_the_request_pool_rebind_is_a_registered_participant(self):
        from sglang.srt.managers.cutover_participants import REGISTRY

        named = {p.name: p for p in REGISTRY}
        self.assertIn("request_pool_phase_ownership", named)
        p = named["request_pool_phase_ownership"]
        self.assertTrue(p.hook and p.probe, p)

    def test_the_pool_back_references_are_registered(self):
        from sglang.srt.managers.cutover_participants import REGISTRY

        named = {p.name: p for p in REGISTRY}
        self.assertIn("req_pool_back_references", named)
        self.assertTrue(named["req_pool_back_references"].hook)

    def test_the_future_map_holder_is_registered_and_no_longer_a_gap(self):
        """FutureMap caches the pool OBJECT (overlap_utils.py:339), built once
        in ``init_overlap``.  THIS cut did not rebind it and filed it on the
        backlog by name; the B3 cut that followed built the rebuild and its
        probe, so the entry now names both obligations.  The gap assertion that
        stood here is RETRACTED rather than deleted, so the sequence is
        readable: filed here, closed one commit later.  See
        ``test_1201_future_map_phase.py`` for the closure's own suite."""
        from sglang.srt.managers.cutover_participants import (
            REGISTRY,
            participants_with_gaps,
        )

        named = {p.name: p for p in REGISTRY}
        self.assertIn("future_map_req_pool_holder", named)
        entry = named["future_map_req_pool_holder"]
        self.assertTrue(entry.hook and entry.probe, entry)
        gaps = {p.name for p in participants_with_gaps()}
        self.assertNotIn("future_map_req_pool_holder", gaps)


class TestEveryTreeCacheClassTheFlipCanRunUnderHasTheHook(CustomTestCase):
    """REPAIR: the property was added to one tree cache, the probe to both.

    ``mem_cache/registry.py`` picks ``UnifiedRadixCache`` for a hybrid-SSM
    model under hierarchical cache and ``MambaRadixCache`` for the same model
    WITHOUT it. #1201 gave the owner property to the first only, so on the
    second ``_restamp_phase_handles``' ``hasattr(tree_cache,
    "bind_req_pool_owner")`` guard was False, the cached handle stayed on the
    outgoing pool -- and ``assert_req_pool_identity``, which was added to both,
    then raised ``ReqPoolRebindRefused`` at the end of every cutover. The
    property is the fix; the probe is for holders that have neither.
    """

    CLASSES = ("UnifiedRadixCache", "MambaRadixCache")

    def _cls(self, name):
        if name == "UnifiedRadixCache":
            return UnifiedRadixCache
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        return MambaRadixCache

    def test_both_classes_expose_the_rebind_hook(self):
        for name in self.CLASSES:
            cls = self._cls(name)
            self.assertTrue(
                hasattr(cls, "bind_req_pool_owner"),
                f"{name} is skipped by _restamp_phase_handles' hasattr guard, "
                "so its request handle never follows the cutover",
            )

    def test_the_handle_follows_the_owner_and_falls_back_without_one(self):
        for name in self.CLASSES:
            cls = self._cls(name)
            cache = cls.__new__(cls)
            cache.req_to_token_pool = "constructor-pool"
            self.assertEqual("constructor-pool", cache.req_to_token_pool, name)
            cache.bind_req_pool_owner(
                types.SimpleNamespace(req_to_token_pool="incoming-pool")
            )
            self.assertEqual(
                "incoming-pool",
                cache.req_to_token_pool,
                f"{name} still names the outgoing phase's pool after the "
                "cutover bound the owner",
            )
            cache.bind_req_pool_owner(None)
            self.assertEqual(
                "constructor-pool",
                cache.req_to_token_pool,
                f"{name}: an unbound cache must be byte-identical to pre-#1201",
            )


if __name__ == "__main__":
    unittest.main()
