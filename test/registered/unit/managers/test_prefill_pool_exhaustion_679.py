# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#679: A SERVING INSTANCE MUST DEGRADE ON POOL PRESSURE, NEVER DIE.

THE DEATH, 2026-08-15 23:41:01, unpinned boot 1, pool 454039, five agent lanes.
The KV pool reached available 0 / evictable 0 and ``alloc_token_slots`` raised
"Out of memory. Try to lower your batch size" out of
``get_new_batch_prefill -> prepare_for_extend -> alloc_for_extend`` on ALL THREE
ranks at once, followed by "terminate called without an active exception".

WHY NOTHING CAUGHT IT, established by tracing every relief this tree owns:

  retract_decode      exists, and is reachable ONLY from update_running_batch,
                      which runs AFTER get_new_batch_prefill in the same
                      iteration. The prefill path cannot reach it.
  the #287 ladder     needs --kv-pressure-ladder to exist at all, commits
                      transitions only every consensus_interval (8) rounds
                      behind a bounded collective, and its actuators reshape
                      FUTURE admission. None of it acts on the batch that is
                      about to allocate.
  kvso try_spill      wired into the decode-OOM branch and into the ladder.
                      Not reachable from the prefill allocation.
  evict_from_tree_cache  the one relief the alloc site does attempt -- and with
                      nothing evictable it is a guaranteed no-op that reports
                      nothing back, so the raise is reached having tried
                      literally nothing else.

AND THE ADMISSION SIDE HANDED IT THE WORK. ``PrefillAdder.add_chunked_req``
carried an explicit override::

    if _rem_tokens <= 0:
        _rem_tokens = self.rem_chunk_tokens

i.e. when the budget said the pool had NOTHING, schedule a full chunk anyway.
The comment above it is right that the request must not be dropped -- it leaks
if it leaves unhandled -- but "admit it anyway" was never the only way to keep
it: PARKING is already a first-class state in this scheduler, produced by the
hybrid-SWA branch three lines up and documented at the chunked-request stash
("a parked chunk leaves extend_range.end == len(prefix_indices), so there is
nothing new to cache").

So the fix is two-layered, and the layers are not interchangeable:

  PREVENTION, admission side, group-uniform. A chunk is scheduled only for
  tokens the pool can actually fund; below one page it parks. The predicate
  reads the PUBLISHED availability floor, not this rank's own size, so every
  rank parks on the same iteration -- a rank-local branch here would split the
  group across different batches, which is a hang, not a stall.

  A NET, alloc site, rank-local. One bounded relief, one retry, then the
  original error unchanged -- the shape ``_mem_create_reclaiming`` already uses
  for driver OOM. Providers are rank-local by contract because by then the
  group has committed to a batch and a collective would hang.

THE 45s WINDOW IS THE AMPLIFIER. SGLANG_PHASE_POLICY_PP_WINDOW_S=45 is live and
admits roughly three times the concurrent prefills the 15s regime did, so the
pool reaches zero far more often. Nothing here depends on the window length:
the guard is a function of what the pool can fund at the moment of scheduling,
which is exactly the quantity a longer window drives to zero.
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache import common as mc

PAGE = 64
CHUNK = 512


class _Allocator:
    def __init__(self, available, alloc_succeeds_after=None):
        self._available = available
        self.alloc_calls = []
        self._succeeds_after = alloc_succeeds_after

    def available_size(self):
        return self._available

    def alloc(self, n):
        self.alloc_calls.append(n)
        if self._succeeds_after is not None and len(self.alloc_calls) > (
            self._succeeds_after
        ):
            return list(range(n))
        return None if self._available < n else list(range(n))


class _TreeCache:
    """The exhausted pool of the crash: available 0, evictable 0."""

    def __init__(self, available=0, evictable=0, uniform_floor=None, **kw):
        self.token_to_kv_pool_allocator = _Allocator(available, **kw)
        self._evictable = evictable
        self.uniform_avail_floor = uniform_floor
        self.evicted = []

    def evictable_size(self):
        return self._evictable

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return True

    def evict(self, params):
        # With nothing evictable this is the no-op the crash walked through.
        self.evicted.append(params)

    def pretty_print(self):
        return ""

    def available_and_evictable_str(self):
        return (
            f"available={self.token_to_kv_pool_allocator.available_size()}, "
            f"evictable={self._evictable}"
        )


class TheAdmissionDecisionTest(unittest.TestCase):
    """The line that scheduled a chunk onto an empty pool."""

    def test_an_exhausted_pool_PARKS_the_chunk(self):
        """THE CRASH, as one assertion. available 0 + evictable 0 must yield a
        park, not a 512-token chunk."""
        self.assertEqual(mc.chunk_tokens_the_pool_can_fund(0, PAGE, CHUNK), 0)

    def test_less_than_one_page_parks_too(self):
        """Below a page nothing useful can be allocated, so scheduling it only
        moves the failure into the allocator."""
        self.assertEqual(mc.chunk_tokens_the_pool_can_fund(PAGE - 1, PAGE, CHUNK), 0)

    def test_exactly_one_page_is_granted(self):
        self.assertEqual(mc.chunk_tokens_the_pool_can_fund(PAGE, PAGE, CHUNK), PAGE)

    def test_a_tight_pool_grants_what_it_HAS_not_the_nominal_chunk(self):
        """The old override took the nominal chunk size regardless of the
        pool. Taking what exists is the difference between a short chunk and a
        dead instance."""
        self.assertEqual(mc.chunk_tokens_the_pool_can_fund(200, PAGE, CHUNK), 200)

    def test_a_healthy_pool_is_unchanged(self):
        """The common path must be byte-identical: full chunk, no throttling."""
        self.assertEqual(mc.chunk_tokens_the_pool_can_fund(1 << 20, PAGE, CHUNK), CHUNK)

    def test_a_negative_or_absurd_input_never_grants_tokens(self):
        for bad in (-1, -(1 << 30)):
            with self.subTest(fundable=bad):
                self.assertEqual(mc.chunk_tokens_the_pool_can_fund(bad, PAGE, CHUNK), 0)


class TheAdmissionPredicateIsGroupUniformTest(unittest.TestCase):
    """A rank-local branch here splits the group across different batches,
    which is a hang rather than a stall. The availability term therefore comes
    from the PUBLISHED floor, exactly as eviction's does."""

    def test_the_published_floor_wins_over_this_ranks_own_size(self):
        tc = _TreeCache(available=9999, evictable=0, uniform_floor=0)
        self.assertEqual(
            mc.fundable_extend_tokens(tc),
            0,
            "a rank with a roomy local pool must still see the group's floor, "
            "or it admits work its peers are parking",
        )

    def test_without_a_floor_the_local_value_is_used(self):
        tc = _TreeCache(available=1234, evictable=0, uniform_floor=None)
        self.assertEqual(mc.fundable_extend_tokens(tc), 1234)

    def test_evictable_tokens_count_as_fundable(self):
        """Eviction is what alloc_token_slots attempts before allocating, so
        tokens the tree can give back are genuinely fundable."""
        tc = _TreeCache(available=100, evictable=400, uniform_floor=None)
        self.assertEqual(mc.fundable_extend_tokens(tc), 500)

    def test_every_rank_reaches_the_same_verdict_on_the_same_floor(self):
        """Three ranks, wildly different local pools, one published floor."""
        verdicts = {
            local: mc.chunk_tokens_the_pool_can_fund(
                mc.fundable_extend_tokens(
                    _TreeCache(available=local, evictable=0, uniform_floor=0)
                ),
                PAGE,
                CHUNK,
            )
            for local in (0, 5000, 100000)
        }
        self.assertEqual(set(verdicts.values()), {0}, f"ranks disagreed: {verdicts}")

    def test_a_missing_tree_cache_funds_nothing(self):
        self.assertEqual(mc.fundable_extend_tokens(None), 0)

    def test_an_unreadable_pool_funds_nothing_rather_than_guessing(self):
        class _Broken:
            token_to_kv_pool_allocator = None

        self.assertEqual(mc.fundable_extend_tokens(_Broken()), 0)


class TheAllocSiteIsANetNotAGraveTest(unittest.TestCase):
    def setUp(self):
        mc.clear_extend_relief_providers()

    def tearDown(self):
        mc.clear_extend_relief_providers()

    def test_with_no_provider_it_still_raises__failloud_is_preserved(self):
        """The net must not become a silent stall: with nothing to reclaim the
        original error is still the last word."""
        tc = _TreeCache(available=0, evictable=0)
        with self.assertRaises(RuntimeError) as cm:
            mc.alloc_token_slots(tc, 128)
        self.assertIn("Out of memory", str(cm.exception))

    def test_a_provider_that_frees_lets_the_retry_succeed(self):
        """DEGRADE, NOT DIE: one bounded relief, one retry."""
        tc = _TreeCache(available=0, evictable=0, alloc_succeeds_after=1)
        freed = []

        def _provider(n):
            freed.append(n)
            return n

        mc.register_extend_relief_provider(_provider)
        out = mc.alloc_token_slots(tc, 128)
        self.assertIsNotNone(out)
        self.assertEqual(freed, [128], "the provider is asked for what is needed")
        self.assertEqual(len(tc.token_to_kv_pool_allocator.alloc_calls), 2)

    def test_it_retries_exactly_ONCE(self):
        """Bounded. A retry loop on an exhausted pool is a hot spin inside the
        scheduler, which is a different way to lose the instance."""
        tc = _TreeCache(available=0, evictable=0)
        mc.register_extend_relief_provider(lambda n: n)
        with self.assertRaises(RuntimeError):
            mc.alloc_token_slots(tc, 128)
        self.assertEqual(len(tc.token_to_kv_pool_allocator.alloc_calls), 2)

    def test_a_provider_that_frees_NOTHING_is_not_retried(self):
        """No point re-asking an allocator when nothing was returned."""
        tc = _TreeCache(available=0, evictable=0)
        mc.register_extend_relief_provider(lambda n: 0)
        with self.assertRaises(RuntimeError):
            mc.alloc_token_slots(tc, 128)
        self.assertEqual(len(tc.token_to_kv_pool_allocator.alloc_calls), 1)

    def test_a_provider_that_RAISES_does_not_mask_the_OOM(self):
        """A relief bug must not replace the diagnosis with its own traceback."""
        tc = _TreeCache(available=0, evictable=0)

        def _broken(n):
            raise ValueError("provider is broken")

        mc.register_extend_relief_provider(_broken)
        with self.assertRaises(RuntimeError) as cm:
            mc.alloc_token_slots(tc, 128)
        self.assertIn("Out of memory", str(cm.exception))

    def test_a_healthy_pool_never_consults_relief(self):
        """The common path must not pay for the net."""
        tc = _TreeCache(available=4096, evictable=0)
        asked = []
        mc.register_extend_relief_provider(lambda n: asked.append(n) or n)
        out = mc.alloc_token_slots(tc, 128)
        self.assertIsNotNone(out)
        self.assertEqual(asked, [])

    def test_providers_are_deduplicated(self):
        calls = []

        def _p(n):
            calls.append(n)
            return 0

        mc.register_extend_relief_provider(_p)
        mc.register_extend_relief_provider(_p)
        tc = _TreeCache(available=0, evictable=0)
        with self.assertRaises(RuntimeError):
            mc.alloc_token_slots(tc, 64)
        self.assertEqual(len(calls), 1)


class TheWiringIsPinnedTest(unittest.TestCase):
    """Pinned on the source: a guard that exists but is not called is the
    failure mode this whole ticket is about."""

    def test_add_chunked_req_consults_the_pool_before_scheduling(self):
        import inspect

        from sglang.srt.managers.schedule_policy import PrefillAdder

        src = inspect.getsource(PrefillAdder.add_chunked_req)
        self.assertIn("chunk_tokens_the_pool_can_fund", src)
        self.assertIn("fundable_extend_tokens", src)

    def test_the_zero_budget_override_is_GONE(self):
        """The exact line that killed the instance must not come back."""
        import inspect

        from sglang.srt.managers.schedule_policy import PrefillAdder

        src = inspect.getsource(PrefillAdder.add_chunked_req)
        self.assertNotIn(
            "_rem_tokens = self.rem_chunk_tokens",
            src,
            "scheduling the nominal chunk on an exhausted pool is the crash",
        )

    def test_a_parked_chunk_schedules_no_new_tokens(self):
        """The park must produce the state the scheduler already documents:
        extend_range.end == len(prefix_indices), i.e. nothing new to cache."""
        import inspect

        from sglang.srt.managers.schedule_policy import PrefillAdder

        src = inspect.getsource(PrefillAdder.add_chunked_req)
        self.assertIn("set_extend_range(", src)
        self.assertIn("len(req.prefix_indices), len(req.prefix_indices)", src)

    def test_the_alloc_site_attempts_relief_before_raising(self):
        import inspect

        src = inspect.getsource(mc.alloc_token_slots)
        self.assertIn("_attempt_extend_relief", src)
        self.assertLess(
            src.index("_attempt_extend_relief"),
            src.index("raise RuntimeError"),
            "relief after the raise is relief that never runs",
        )


if __name__ == "__main__":
    unittest.main()


class TheEvictionReceiptIsReadTest(unittest.TestCase):
    """#681: the allocation failed with 65,766 evictable tokens on the books.

    THE CRASH, 2026-08-16 01:46:10, all three ranks reporting identically:

        Try to allocate 512 tokens.
        Available full tokens: 66039 (available=273 + evictable=65766)

    512 needed, 65,766 reported evictable, allocation failed anyway -- and pool
    usage was 0.85, so this was never exhaustion. The pools agreed across ranks,
    so `uniform_avail_floor` was None, so the trigger used the local value
    (273 < 512) and eviction DID run. It simply did not deliver.

    WHY: `evict` walks the LEAF FRONTIER -- it pops evictable leaves and
    re-pushes a parent only once all its children are gone and it is unlocked --
    while `evictable_size_` counts unlocked tokens ANYWHERE in the tree. Tokens
    behind a locked chain are counted and unreachable. The counter promises what
    the actuator cannot pay.

    AND NOBODY LOOKED AT THE RECEIPT. `evict` returns num_tokens_evicted;
    `evict_from_tree_cache` discarded it, so an under-delivery was
    indistinguishable from success and the error three lines later reported
    plenty of memory. That is the defect this class pins.
    """

    def test_the_evicted_count_is_returned_not_discarded(self):
        class _Cache(_TreeCache):
            def __init__(self, delivers):
                super().__init__(available=0, evictable=65766)
                self.delivers = delivers

            def evict(self, params):
                self.evicted.append(params)
                return type("R", (), {"num_tokens_evicted": self.delivers})()

        self.assertEqual(mc.evict_from_tree_cache(_Cache(512), 512), 512)
        self.assertEqual(mc.evict_from_tree_cache(_Cache(0), 512), 0)

    def test_no_eviction_needed_reports_zero_rather_than_lying(self):
        tc = _TreeCache(available=4096, evictable=0)
        self.assertEqual(mc.evict_from_tree_cache(tc, 512), 0)
        self.assertEqual(tc.evicted, [], "nothing to evict, nothing evicted")

    def test_the_error_NAMES_the_under_delivery(self):
        """The operator must be able to read, from the failure itself, that
        eviction under-delivered -- not infer it from a number that looks
        healthy."""
        note = mc._eviction_shortfall_note(
            _TreeCache(available=273, evictable=65766), 512, 0
        )
        self.assertIn("UNDER-DELIVERED", note)
        self.assertIn("512", note)
        self.assertIn("65766", note)
        # The first version of this assertion pinned the wording "LEAF
        # FRONTIER", i.e. the #681 explanation that tokens behind a LOCKED
        # chain were counted but unreachable. That explanation is false --
        # locking is ancestor-closed, so an unlocked node never has a locked
        # descendant (test_evictable_reachability_681). Pinning a wrong
        # mechanism in a diagnostic is worse than pinning none, so what is
        # pinned now is the property the note must carry: that firing at all
        # is a REGRESSION, because the frontier can pay what the counter
        # promises.
        self.assertIn("REGRESSION SIGNAL", note)
        # #790 retired the "THIS LINE SHOULD BE UNREACHABLE" clause, and the
        # reason is the same one that retired "LEAF FRONTIER" above: it pinned
        # a claim that turned out to be false. A residency cap
        # (managers/kv_backing_relief.py, ``KvRowCap``) confiscates freed ids
        # above its cap at the allocator's free listener, so a shortfall here
        # has a LAWFUL cause and the note now names it when one is engaged.
        # "REGRESSION SIGNAL" survives as the verdict for the case where no
        # confiscator is named, which is what this assertion covers.
        self.assertNotIn(
            "behind a locked chain",
            note,
            "the falsified mechanism must not come back",
        )

    def test_a_delivering_eviction_adds_no_note(self):
        self.assertEqual(
            mc._eviction_shortfall_note(_TreeCache(available=0, evictable=0), 512, 512),
            "",
        )

    def test_the_note_survives_a_cache_that_cannot_report_evictable(self):
        class _Mute(_TreeCache):
            def evictable_size(self):
                raise RuntimeError("no accessor")

        note = mc._eviction_shortfall_note(_Mute(), 512, 0)
        self.assertIn("UNDER-DELIVERED", note)


class EveryPrefillAllocPathHasTheNetTest(unittest.TestCase):
    """#681 closes rule 3 of DESIGN_679: parking/relief must be reachable from
    EVERY alloc path, not the one that happened to crash first.

    The audit of prepare_for_extend -> alloc_for_extend found THREE raise sites
    and only one covered:

        alloc_req_slots                 request slots (and mamba states)
        alloc_token_slots               page_size == 1   <- the one covered
        alloc_paged_token_slots_extend  page_size > 1
    """

    def test_all_three_prefill_raise_sites_ask_the_net(self):
        import inspect

        for fn in (
            mc.alloc_token_slots,
            mc.alloc_paged_token_slots_extend,
            mc.alloc_req_slots,
        ):
            with self.subTest(site=fn.__name__):
                src = inspect.getsource(fn)
                self.assertIn(
                    "_attempt_extend_relief",
                    src,
                    f"{fn.__name__} can raise on the prefill admission path "
                    "with no relief attempted -- rule 3 violated",
                )

    def test_the_net_is_asked_BEFORE_each_raise(self):
        import inspect

        for fn in (mc.alloc_token_slots, mc.alloc_paged_token_slots_extend):
            with self.subTest(site=fn.__name__):
                src = inspect.getsource(fn)
                self.assertLess(
                    src.index("_attempt_extend_relief"),
                    src.index("raise RuntimeError"),
                    "relief after the raise is relief that never runs",
                )
