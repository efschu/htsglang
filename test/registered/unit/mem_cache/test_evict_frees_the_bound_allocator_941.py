"""#941: the tree evicts into the allocator it was BUILT with, not the bound one.

THE MEASUREMENT THAT LOCATES THIS (window 2k, boot_2k_dd0e3bc224_0827_1237.log,
PP0, one cutover pair, and it is arithmetic rather than correlation)::

    12:42:00 at-arm      tp_to_pp  free=470650 cached=267 unacc=332  alloc=Paged
    12:42:01 RESIDENTS RELEASED for tp_to_pp: ... the prefix tree dropped
             returning 267 row(s) to the allocator
    12:42:01 pre-cutover tp_to_pp  free=470650 cached=0   unacc=599  alloc=Paged

The drop reports 267 rows returned and the allocator's free count DOES NOT MOVE.
The 267 rows leave ``cached`` and land in ``unaccounted``, exactly. Five cycles,
five exact matches (65, 332, 599, 867, 1135 -- each the previous plus the TP
phase's whole allocation). And the rows are not destroyed: the OTHER phase's
allocator reports them, as DUPLICATES in its free list -- its ``available_size``
(a raw length) exceeds the distinct free set the census enumerates by exactly the
same running total (143360 free vs 143425 / 143692 / 143959 / 144227 / 144495
available). One number leaving one allocator's books and arriving on another's.

THE ROOT, AND IT IS A BINDING THE REBIND MODULE NEVER ENUMERATED.
``hicache_phase_binding`` states the law itself: "THREE READERS, AND WHY A
PARTIAL REBIND IS WORSE THAN NONE ... If one moves and another does not, the
readers disagree about which pool a row id names, and the disagreement is
invisible: every call still succeeds, against different memory." It then moves
``token_to_kv_pool_allocator`` on the scheduler, the tree and the controller.

``FullComponent.__init__`` captures a BOUND METHOD off that attribute::

    allocator = cache.token_to_kv_pool_allocator
    self._free_full = allocator.free          # <- the fourth reader

A bound method carries its instance. Rebinding the tree's ATTRIBUTE cannot reach
it, and ``coherence_check`` cannot see it either: it compares
``hicache_binding_generation`` on the three NAMED readers, and the component is
not one of them. So the check passes while the free path is stale -- the
indicator is green for the same reason the defect is invisible.

WHICH DROP LEAKS, AND WHY ONLY THAT ONE. The seam retracts and drops BEFORE the
cutover rebinds, so at a ``pp_to_tp`` drop the current binding is still PP -- the
phase that minted the rows -- and the stale capture happens to agree. At a
``tp_to_pp`` drop the binding is TP and the capture is still PP: every row the TP
phase minted is freed into the PP allocator. That is why the boot leaks once per
cutover PAIR and why the amount is the TP phase's allocation, not the PP one's.

WHAT THE FIX IS AND IS NOT. It is not a free at the seam and not a lock release:
no row is freed here that was not already being freed, at the same moment, by the
same call. Only the ALLOCATOR the free is addressed to changes -- from the one
captured at construction to the one currently bound. So there is no new free to
pair with a later drop, and no double-free to build.
"""

import types
import unittest

import torch

from sglang.srt.managers.phase_flip_runtime import drop_prefix_tree_returning_rows
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_phase_binding import PhasePools, _stamp
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~3s: two tiny CPU-only KV pools, no accelerator, no group, no boot.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

POOL = 64
PAGE_SIZE = 1
ROWS = 6  # what the "TP phase" mints and the drop then has to give back


def _kv_pool():
    return MHATokenToKVPool(
        size=POOL,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        head_num=2,
        head_dim=4,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )


def _pp_allocator():
    """The boot phase's allocator -- the class this rig's PP stack builds."""
    return TokenToKVPoolAllocator(
        size=POOL,
        dtype=torch.float16,
        device="cpu",
        kvcache=_kv_pool(),
        need_sort=False,
    )


def _tp_allocator():
    """The flip phase's allocator -- the class this rig's TP stack builds.

    Deliberately the production PAIR (Token vs Paged) and not two copies of one
    class: the boot's census prints exactly these two type names on the two
    sides of the cutover, and a double shaped differently from production is
    this module's own recorded way of surviving the boot it kills (W29).
    """
    return PagedTokenToKVPoolAllocator(
        size=POOL,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=_kv_pool(),
        need_sort=False,
    )


def _cache(allocator):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=64, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE_SIZE,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )


def _rebind(cache, allocator):
    """The PRODUCTION rebinder, not a hand-written setattr.

    ``_stamp`` is what ``hicache_phase_binding`` runs on each of its three
    readers at the cutover (123 times on the 2k boot, "rebound 3 reader(s)").
    Driving the real one is what makes the red below a statement about the
    shipping path rather than about this file's idea of one. ``device_pool`` and
    ``host_pool`` are None so that only the allocator binding moves -- ``_stamp``
    sets an attribute only where the value is not None AND the reader already
    has it, so the None halves are skipped exactly as they are for a reader that
    does not carry them.
    """
    _stamp(
        cache,
        PhasePools(phase="tp", device_pool=None, host_pool=None, allocator=allocator),
        1,
    )


def _tree_holding(cache, allocator, n=ROWS):
    """Rows MINTED BY ``allocator`` and handed to the tree, evictable."""
    rows = allocator.alloc(n)
    assert rows is not None and len(rows) == n
    cache.insert(
        InsertParams(
            key=RadixKey(list(range(100, 100 + n))),
            value=rows.to(dtype=torch.int64, copy=True),
        )
    )
    return rows


class TestTheHarnessIsTheProductionShape(CustomTestCase):
    """Before asserting anything about the free, prove the scenario is real:
    the rebind moves the tree's attribute, and the tree really holds the rows."""

    def test_the_production_rebinder_moves_the_trees_attribute(self):
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        self.assertIs(cache.token_to_kv_pool_allocator, pp)
        _rebind(cache, tp)
        self.assertIs(cache.token_to_kv_pool_allocator, tp)

    def test_the_tree_holds_the_rows_the_flip_phase_minted(self):
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _rebind(cache, tp)
        _tree_holding(cache, tp)
        self.assertEqual(int(cache.full_evictable_size()), ROWS)


class TestEvictionPaysBackTheBoundAllocator(CustomTestCase):
    """THE RED. Today the eviction pays back the allocator captured at
    construction, so the bound one never sees its rows again and the other one
    receives ids it never minted."""

    def test_the_bound_allocator_gets_its_rows_back(self):
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _rebind(cache, tp)
        _tree_holding(cache, tp)

        before = int(tp.available_size())
        cache.evict(EvictParams(num_tokens=ROWS))
        self.assertEqual(
            int(tp.available_size()) - before,
            ROWS,
            "the eviction did not return the rows to the allocator that minted "
            "them; this is the boot's 'drop returned N row(s)' beside an "
            "unchanged free count",
        )

    def test_the_unbound_allocator_receives_nothing(self):
        """The other half, and the one that makes this more than a leak.

        The rows do not vanish: they are concatenated onto the free list of an
        allocator that never minted them, where they sit as DUPLICATES of ids it
        already calls free. That is a second writer waiting for the first
        allocation that reaches them.
        """
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _rebind(cache, tp)
        _tree_holding(cache, tp)

        before = int(pp.available_size())
        cache.evict(EvictParams(num_tokens=ROWS))
        self.assertEqual(
            int(pp.available_size()),
            before,
            "an allocator that minted nothing was paid back; on the boot this "
            "is the PP allocator's available_size drifting above its own "
            "enumerated free set by exactly the leaked total",
        )

    def test_the_cutover_drop_returns_the_rows_to_the_bound_allocator(self):
        """The same property through the seam's own entry point, because that
        is where the boot measured it -- and the drop's RETURNED COUNT is not
        the evidence. It reported 267 while nothing came back."""
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _rebind(cache, tp)
        _tree_holding(cache, tp)

        before = int(tp.available_size())
        returned = drop_prefix_tree_returning_rows(cache)
        self.assertEqual(returned, ROWS)
        self.assertEqual(int(tp.available_size()) - before, ROWS)


class TestNothingIsFreedTwiceAndNothingNewIsFreed(CustomTestCase):
    """THE SAFETY PIN. This changes WHERE a free is addressed, never WHETHER one
    happens. If it ever added a free, the pairing with a later drop is the
    double-free the #938 instrument refused to build."""

    def test_the_free_happens_exactly_once_and_only_for_evicted_rows(self):
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _rebind(cache, tp)
        rows = _tree_holding(cache, tp)

        seen = []
        real_free = tp.free

        def counting_free(idx):
            seen.append([int(v) for v in idx.tolist()])
            return real_free(idx)

        tp.free = counting_free
        try:
            cache.evict(EvictParams(num_tokens=ROWS))
        finally:
            tp.free = real_free

        self.assertEqual(len(seen), 1, f"expected one free call, got {seen}")
        self.assertEqual(sorted(seen[0]), sorted(int(v) for v in rows.tolist()))

    def test_an_unrebound_cache_is_byte_identical(self):
        """The default path -- no flip, no rebind -- must not move at all."""
        pp = _pp_allocator()
        cache = _cache(pp)
        _tree_holding(cache, pp)
        before = int(pp.available_size())
        cache.evict(EvictParams(num_tokens=ROWS))
        self.assertEqual(int(pp.available_size()) - before, ROWS)


class TestNoComponentHoldsABindingTheRebindCannotReach(CustomTestCase):
    """THE FUTURE CHECK, and it is the class and not this instance.

    The defect was not "``_free_full`` was wrong". It was that a tree component
    may hold a bound method of a REBINDABLE allocator, and nothing in the rebind
    path or its coherence check can see that it did. The next component to do it
    reproduces this exactly, and it would again be invisible.

    So this asserts over EVERY component the cache builds, by walking their
    attributes after a rebind and refusing any callable still carrying the old
    allocator as its ``__self__``. Behavioural, not a source-text match: a
    capture spelled differently is caught the same way.
    """

    def test_no_component_attribute_still_carries_the_old_allocator(self):
        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _rebind(cache, tp)

        # The old allocator and anything reachable from it that a component
        # could plausibly have bound to. NEVER None: `getattr(x, "__self__",
        # None) is None` is true of every non-method attribute, so a None in
        # this set would flag the whole object graph and read as a finding.
        old = {id(pp)}
        inner = getattr(pp, "full_attn_allocator", None)
        if inner is not None:
            old.add(id(inner))

        stale = []
        for comp in cache._components_tuple:
            for name, value in vars(comp).items():
                owner = getattr(value, "__self__", None)
                if owner is not None and id(owner) in old:
                    stale.append(f"{type(comp).__name__}.{name}")
        self.assertEqual(
            stale,
            [],
            "a tree component still holds a callable bound to the allocator the "
            "cutover rebound away from; hicache_phase_binding cannot move it "
            "and coherence_check cannot see it (#941)",
        )


class TestTheUnownedProbeCanReachTheOtherPhasesPool(CustomTestCase):
    """THE INSTRUMENT THAT SHOULD HAVE NAMED THIS, AND ABSTAINED INSTEAD.

    ``#919 UNOWNED-BLOCK`` exists to answer "is an un-enumerated pool object
    holding these rows?". On the 2k boot it printed ``NO-SECOND-POOL: ... no
    second pool object is reachable from here`` at every census -- the
    ``not candidates`` branch of ``unenumerated_owner_verdict``, i.e. it had
    NOTHING to test, not a tested negative.

    ``PhaseFlipStacks`` carries ``tp_worker`` and no ``pp_worker``; the PP stack
    is the scheduler's own ``tp_worker``. The candidate list looked for
    ``stacks.pp_worker``, found None, and skipped ``stacks.tp_worker`` as the
    already-enumerated allocator -- so in the TP phase, the only phase this rig
    ever reports unowned rows in, the list was empty by construction.
    """

    @staticmethod
    def _candidates(sched):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        probe = PhaseFlipRuntime._unenumerated_owner_candidates
        holder = types.SimpleNamespace(_census_scheduler=sched)
        return probe(holder)

    def test_the_other_phases_allocator_is_a_candidate_in_both_phases(self):
        pp, tp = _pp_allocator(), _tp_allocator()
        pp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(token_to_kv_pool_allocator=pp)
        )
        tp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(token_to_kv_pool_allocator=tp)
        )

        # TP phase: the scheduler enumerates the TP allocator, so the PP stack's
        # is the un-enumerated owner -- the one the boot's rows went to.
        names = [
            c.name
            for c in self._candidates(
                types.SimpleNamespace(
                    token_to_kv_pool_allocator=tp,
                    tp_worker=pp_worker,
                    phase_flip_stacks=types.SimpleNamespace(tp_worker=tp_worker),
                    draft_token_to_kv_pool_allocator=None,
                    draft_token_to_kv_pool=None,
                )
            )
        ]
        self.assertIn("pp_stack_allocator", names)
        self.assertNotIn("tp_stack_allocator", names)

        # PP phase: the mirror image, from the same list and the same identity
        # skip -- the probe never has to know which phase it is in.
        names = [
            c.name
            for c in self._candidates(
                types.SimpleNamespace(
                    token_to_kv_pool_allocator=pp,
                    tp_worker=pp_worker,
                    phase_flip_stacks=types.SimpleNamespace(tp_worker=tp_worker),
                    draft_token_to_kv_pool_allocator=None,
                    draft_token_to_kv_pool=None,
                )
            )
        ]
        self.assertIn("tp_stack_allocator", names)
        self.assertNotIn("pp_stack_allocator", names)

    def test_the_verdict_stops_saying_no_second_pool_is_reachable(self):
        from sglang.srt.mem_cache.kv_row_ownership import unenumerated_owner_verdict

        pp, tp = _pp_allocator(), _tp_allocator()
        cands = self._candidates(
            types.SimpleNamespace(
                token_to_kv_pool_allocator=tp,
                tp_worker=types.SimpleNamespace(
                    model_runner=types.SimpleNamespace(token_to_kv_pool_allocator=pp)
                ),
                phase_flip_stacks=types.SimpleNamespace(
                    tp_worker=types.SimpleNamespace(
                        model_runner=types.SimpleNamespace(
                            token_to_kv_pool_allocator=tp
                        )
                    )
                ),
                draft_token_to_kv_pool_allocator=None,
                draft_token_to_kv_pool=None,
            )
        )
        # The boot's sample, verbatim: "#919 UNOWNED-BLOCK NO-SECOND-POOL:
        # 65 row(s), sample=[1, 2, 3, 4, 5, 6, 7, 8]".
        verdict, detail = unenumerated_owner_verdict([1, 2, 3, 4, 5, 6, 7, 8], cands)
        self.assertNotIn("no second pool object is reachable", detail)
        self.assertIn("pp_stack_allocator", detail)


class TestTheCoherenceCheckCannotSeeThisBinding(CustomTestCase):
    """WHY IT SURVIVED. ``hicache_phase_binding`` verifies a rebind "by
    generation, not by inspection" -- over three NAMED readers. The component
    is a fourth, so the check is green in precisely the state this file reds."""

    def test_the_named_readers_are_coherent_while_the_free_path_is_not(self):
        from sglang.srt.mem_cache.hicache_phase_binding import _STATE, coherence_check

        pp, tp = _pp_allocator(), _tp_allocator()
        cache = _cache(pp)
        _stamp(
            cache,
            PhasePools(phase="tp", device_pool=None, host_pool=None, allocator=tp),
            _STATE.generation,
        )
        # The reader dict the module itself would check: the component is absent
        # from it by construction, which is the whole finding.
        self.assertEqual(coherence_check({"tree_cache": cache}), _STATE.generation)


if __name__ == "__main__":
    unittest.main()
