"""#681: new-request admission must decide from the GROUP-uniform fundable pool.

THE GAP THIS PINS. #679 taught the CHUNKED path to park instead of scheduling
work the pool cannot fund, and it did so on a group-uniform predicate
(`fundable_extend_tokens` -> `uniform_avail_for_evict`). The NEW-request path
never learned either half: `PrefillAdder.rem_total_tokens` reads THIS RANK's
own `available_size() + evictable_size()`, so

  * a rank whose pool is roomier than the group's binding rank admits work the
    group cannot fund, and the allocation dies at `alloc_for_extend` -- the
    2026-08-16 01:46:10 death, all three ranks at once; and
  * the admission BRANCH is taken on a rank-local number, which is the
    rank-local-test-before-a-collective family this tree keeps paying for
    (#583, #603, #616g, #639): two ranks can build different batches and the
    group hangs rather than stalls.

`fundable_extend_tokens`' own docstring names the first half and was written
for the chunked gate; these tests hold the new-request gate to the same rule.

WHY A CAP AND NOT A REPLACEMENT. `rem_total_tokens` subtracts
`rem_total_token_offset` -- the running batch's reservation plus everything
this iteration has already promised to admit -- and reservations the fundable
floor knows nothing about (mamba gap, page overhead). The floor is therefore an
ADDITIONAL ceiling, never a substitute: the budget is the MIN of what this rank
may spend and what the group can actually pay.

INERT ON THE DEFAULT PATH, BY CONSTRUCTION. `uniform_avail_for_evict` returns
the live local value when no floor is published (single rank, or pools that
agree), so on the reference boot the cap equals the local term and nothing
moves. That is asserted here rather than assumed.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _tree_cache(*, evictable: int = 0, floor=None) -> MagicMock:
    """A stand-in whose two availability terms can be set independently.

    `supports_mamba` is pinned False so `rem_total_tokens` takes the plain
    `available_size() + evictable_size()` branch: a bare MagicMock answers
    truthily and would silently route these assertions through the hybrid-SSM
    branch instead, testing a different property than the one named.
    """
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = evictable
    tc.full_evictable_size.return_value = evictable
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    #: the published group MIN; None means "no reduce ran", the default path.
    tc.uniform_avail_floor = floor
    return tc


def _allocator(*, available: int = 0) -> MagicMock:
    alloc = MagicMock()
    alloc.available_size.return_value = available
    alloc.full_available_size.return_value = available
    alloc.swa_available_size.return_value = 0
    return alloc


def _running_batch() -> MagicMock:
    batch = MagicMock()
    batch.reqs = []
    return batch


def _adder(tree_cache, allocator, **kwargs) -> PrefillAdder:
    defaults = dict(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=_running_batch(),
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=None,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    defaults.update(kwargs)
    return PrefillAdder(**defaults)


class TheAdmissionBudgetRespectsTheGroupFloorTest(unittest.TestCase):
    """The core of #681: what this rank HAS is not what the group can PAY."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_a_group_floor_below_this_ranks_pool_caps_the_budget(self):
        """THE 01:46 SHAPE. This rank sees 100000 free; the binding rank has 0.

        Admitting against the local number is what let a batch be built that
        `alloc_for_extend` could not fund. The budget must collapse to the
        floor, not to what this rank happens to hold.
        """
        tc = _tree_cache(evictable=0, floor=0)
        adder = _adder(tc, _allocator(available=100_000), fundable_extend_floor=0)
        self.assertLessEqual(
            adder.rem_total_tokens,
            0,
            "the budget still reports this rank's own pool, so a batch the "
            "group cannot fund is admissible",
        )

    def test_a_partial_floor_grants_exactly_the_floor(self):
        tc = _tree_cache(evictable=0, floor=512)
        adder = _adder(tc, _allocator(available=100_000), fundable_extend_floor=512)
        self.assertEqual(adder.rem_total_tokens, 512)

    def test_evictable_tokens_the_group_agrees_on_still_count(self):
        """Fundable is availability PLUS evictable: eviction is what
        `alloc_token_slots` attempts before allocating, so tokens the radix
        tree can give back are genuinely payable."""
        tc = _tree_cache(evictable=300, floor=200)
        adder = _adder(tc, _allocator(available=100_000), fundable_extend_floor=500)
        self.assertEqual(adder.rem_total_tokens, 500)


class TheFloorIsChargedAsWorkIsAdmittedTest(unittest.TestCase):
    """A floor consulted per-request without accounting is not a bound.

    N requests each comparing themselves against the SAME untouched free pool
    is precisely the over-admission this ticket is about, one level up. The cap
    must share `rem_total_token_offset` with the local term.
    """

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_committed_tokens_are_subtracted_from_the_floor(self):
        tc = _tree_cache(evictable=0, floor=1000)
        adder = _adder(tc, _allocator(available=100_000), fundable_extend_floor=1000)
        self.assertEqual(adder.rem_total_tokens, 1000)
        adder.rem_total_token_offset += 600
        self.assertEqual(
            adder.rem_total_tokens,
            400,
            "the floor must be spent down as work is admitted, or every "
            "request in the round passes against the same free pool",
        )

    def test_an_over_committed_round_reports_no_budget_not_a_negative_grant(self):
        tc = _tree_cache(evictable=0, floor=1000)
        adder = _adder(tc, _allocator(available=100_000), fundable_extend_floor=1000)
        adder.rem_total_token_offset += 4000
        self.assertLessEqual(adder.rem_total_tokens, 0)


class TheVerdictIsGroupUniformTest(unittest.TestCase):
    """Every rank must reach the same admission verdict on the same iteration.

    This is the hang-versus-stall property: a rank-local branch here splits the
    group across different batches and different collective shapes.
    """

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_ranks_with_different_pools_agree_on_the_same_floor(self):
        """THE PRODUCTION CONTRACT, and the local values respect it.

        The floor is the group MIN of `available_size()`, so `floor <= every
        rank's own availability` holds by construction -- a rank below the
        floor is not a state the reduce can produce. The local pools here are
        therefore all >= the floor, which is the only case worth asserting
        uniformity over; the clamp's behaviour in the contradictory case is
        pinned separately (`test_the_local_term_still_binds_...`), where
        refusing to be talked UP by a stale floor is the wanted behaviour.
        """
        floor = 768
        verdicts = []
        for local in (768, 5_000, 250_000):
            self.assertGreaterEqual(
                local, floor, "the test's own premise must hold the invariant"
            )
            tc = _tree_cache(evictable=0, floor=floor)
            adder = _adder(tc, _allocator(available=local), fundable_extend_floor=floor)
            verdicts.append(adder.rem_total_tokens)
        self.assertEqual(
            len(set(verdicts)),
            1,
            f"ranks disagreed on the admission budget: {verdicts}",
        )

    def test_the_floor_binds_before_the_local_term_whenever_the_invariant_holds(self):
        """Why the MIN resolves to the floor in production, stated as a test.

        `floor = uniform_avail + evictable` and
        `local = local_avail + evictable - offset`, with
        `uniform_avail <= local_avail`. So `floor - offset <= local` always,
        and the group term is the binding one on every rank. If this ever goes
        red, the two gates have stopped agreeing and #681's new-request half
        is open again in a shape the uniformity test above cannot see.
        """
        evictable, offset = 400, 250
        for local_avail, uniform_avail in ((10_000, 900), (900, 900)):
            tc = _tree_cache(evictable=evictable, floor=uniform_avail)
            adder = _adder(
                tc,
                _allocator(available=local_avail),
                fundable_extend_floor=uniform_avail + evictable,
            )
            adder.rem_total_token_offset += offset
            self.assertEqual(
                adder.rem_total_tokens,
                uniform_avail + evictable - offset,
                "the group floor must be the binding term",
            )


class TheDefaultPathIsUntouchedTest(unittest.TestCase):
    """Backward compatibility, asserted rather than asserted-about.

    With no floor published -- single rank, or pools that agree -- the cap must
    not exist at all. This is the reference boot.
    """

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_without_a_floor_the_budget_is_the_local_value(self):
        tc = _tree_cache(evictable=1_000, floor=None)
        adder = _adder(tc, _allocator(available=9_000), fundable_extend_floor=None)
        self.assertEqual(adder.rem_total_tokens, 10_000)

    def test_a_roomy_floor_never_tightens_a_healthy_pool(self):
        tc = _tree_cache(evictable=1_000, floor=10**9)
        adder = _adder(tc, _allocator(available=9_000), fundable_extend_floor=10**9)
        self.assertEqual(
            adder.rem_total_tokens,
            10_000,
            "a floor above the local pool must be inert, not a new ceiling",
        )

    def test_the_local_term_still_binds_when_it_is_the_smaller_one(self):
        """The cap is a MIN, not a replacement: a rank that is itself short
        must not be talked UP by a roomy group floor."""
        tc = _tree_cache(evictable=0, floor=10**9)
        adder = _adder(tc, _allocator(available=64), fundable_extend_floor=10**9)
        self.assertEqual(adder.rem_total_tokens, 64)


class TheCeilingCannotWedgeTheInstanceTest(unittest.TestCase):
    """A ceiling that mis-reads 0 is worse than the crash it prevents.

    `fundable_extend_tokens` returns 0 for BOTH "the pool is empty" and "the
    pool could not be read". The chunked gate can live with that -- 0 parks a
    chunk and the next round retries. As a budget ceiling it would refuse every
    request forever. So the ceiling is applied only where a floor was actually
    published.
    """

    def test_no_published_floor_means_no_ceiling(self):
        from sglang.srt.mem_cache.common import published_fundable_floor

        tc = _tree_cache(evictable=10, floor=None)
        tc.token_to_kv_pool_allocator = _allocator(available=10)
        self.assertIsNone(published_fundable_floor(tc))

    def test_a_missing_tree_cache_means_no_ceiling_rather_than_a_zero_one(self):
        from sglang.srt.mem_cache.common import published_fundable_floor

        self.assertIsNone(published_fundable_floor(None))

    def test_a_published_floor_is_returned(self):
        from sglang.srt.mem_cache.common import published_fundable_floor

        tc = _tree_cache(evictable=100, floor=400)
        tc.token_to_kv_pool_allocator = _allocator(available=999)
        self.assertEqual(published_fundable_floor(tc), 500)

    def test_a_published_ZERO_floor_is_honoured_not_discarded(self):
        """The distinction is 'was a floor published', not 'is it non-zero'.
        A genuine group-wide zero must still bind, or the ticket's own crash
        state is exempt from the fix."""
        from sglang.srt.mem_cache.common import published_fundable_floor

        tc = _tree_cache(evictable=0, floor=0)
        tc.token_to_kv_pool_allocator = _allocator(available=0)
        self.assertEqual(published_fundable_floor(tc), 0)


class TheWiringIsPinnedTest(unittest.TestCase):
    """A budget the scheduler never hands a floor to is a dead parameter.

    The other cases bind the adder directly, which is the honest way to test
    the ARITHMETIC and structurally incapable of testing that the PRODUCTION
    call site supplies the number. Only the source shows that, and the
    invariant is static, so a source-level assertion is the right tool -- the
    same instrument #677's ordering guard uses, for the same reason.
    """

    def _scheduler_source(self) -> str:
        import sglang.srt.managers.scheduler as scheduler_mod

        with open(scheduler_mod.__file__, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_the_prefill_adder_is_constructed_with_a_fundable_floor(self):
        src = self._scheduler_source()
        self.assertIn(
            "fundable_extend_floor=",
            src,
            "the scheduler builds the PrefillAdder without a group-uniform "
            "fundable floor, so the cap is dead code and #681's new-request "
            "half is open again",
        )

    def test_the_floor_comes_from_the_guarded_helper(self):
        src = self._scheduler_source()
        self.assertIn(
            "published_fundable_floor",
            src,
            "the ceiling must come through published_fundable_floor; calling "
            "fundable_extend_tokens directly reinstates the wedge risk (a "
            "mis-read 0 becomes a permanent refusal rather than a park)",
        )

    def test_the_guarded_helper_still_delegates_to_the_group_uniform_one(self):
        """The second hop of the contract: the guard must not become its own
        floor computation, or the rank-local branch this ticket closes comes
        back through the back door."""
        import inspect

        from sglang.srt.mem_cache.common import published_fundable_floor

        self.assertIn(
            "fundable_extend_tokens",
            inspect.getsource(published_fundable_floor),
        )

    def test_the_adder_still_exposes_the_parameter(self):
        """If a refactor drops the parameter, these guards should go red
        rather than keep passing while watching nothing."""
        import inspect

        sig = inspect.signature(PrefillAdder.__init__)
        self.assertIn("fundable_extend_floor", sig.parameters)


if __name__ == "__main__":
    unittest.main()
