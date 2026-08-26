# SPDX-License-Identifier: Apache-2.0
"""#905: the census-scheduler double is only evidence if it models the seam.

THE FAILURE MODE THIS FILE EXISTS TO PREVENT, stated by the ticket that
ordered the double: "a double that only satisfies the `is None` check turns 32
reds into 32 greens that measure nothing" (#630, where unfaithful stubs
modelled a deadline-ignoring wait and hid a livelock; W29, where the suite's
own tree double carried `full_evictable_size_` as an ATTRIBUTE while production
reads `full_evictable_size()` as a METHOD, so ten green tests survived the boot
that defect killed).

So every property `seam_census_double` claims is falsified here, in the
direction that would make it decorative. Two kinds of test:

  * FAITHFULNESS -- the double reaches the same state the shipped seam code
    puts a real scheduler in, through the shipped functions;
  * CAN-FAIL -- the contract violation is PLANTED and the double is required
    to notice. A property nobody has ever seen fail is not pinned.

The order this pins is the one #825 died on: RETRACT, then RESET. Reversing it
here does not fail an assertion, it CRASHES with the metal's own signature.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    SeamOrderError,
    _live_reqs,
    build_cutover_release,
    consume_retracted_from_live_universe,
    drop_prefix_tree_returning_rows,
    release_residents_for_cutover,
    tree_evictable_full_rows,
)
from sglang.test.test_utils import CustomTestCase

from seam_census_double import (  # noqa: E402 (sibling)
    ROWS_PER_RESIDENT,
    FaithfulBatch,
    FaithfulCensusScheduler,
)


def _run_the_seam(sched):
    """The #856 release step, driven through the SHIPPED functions only.

    This is deliberately a transcription of `_release_residents_for_cutover`'s
    body rather than a call to it, because that method needs a whole
    `PhaseFlipRuntime`. Every function it names is the production one.
    """
    built = build_cutover_release(sched)
    assert built is not None
    retract, drop = built

    def _retract_and_consume(rs):
        out = retract(rs)
        sched.retired_refs = consume_retracted_from_live_universe(sched, rs)
        return out

    reqs = list(_live_reqs(sched))
    released = release_residents_for_cutover(
        reqs, retract=_retract_and_consume, reset_tree=drop
    )
    sched.readmitted = sched.readmit_seam_residents(list(released or ()))
    return released


class TestTheDoubleIsNotAnIsNoneStub(CustomTestCase):
    """It has to have residents, or the retraction leg never runs."""

    def test_the_default_fixture_has_a_non_empty_resident_set(self):
        sched = FaithfulCensusScheduler()
        self.assertEqual(sched.live_req_count(), 3)
        self.assertGreater(sched.ledger.outstanding(), 0)

    def test_the_resident_set_spans_more_than_one_batch(self):
        # `_live_reqs` deduplicates by identity across running_mbs,
        # running_batch, last_batch and chunked_req. A double that put each
        # request in exactly one place would never exercise that, and the
        # #631-J slot-scope defect it exists for would be invisible.
        sched = FaithfulCensusScheduler()
        places = sum(len(b.reqs) for b in sched.batches())
        self.assertGreater(places, sched.live_req_count())

    def test_the_tree_answers_the_reader_production_actually_uses(self):
        # THE W29 DIRECTION. `tree_evictable_full_rows` calls
        # `full_evictable_size()`; a double carrying only the trailing-
        # underscore attribute reads as None -> the drop skips its eviction.
        sched = FaithfulCensusScheduler()
        self.assertIsNotNone(tree_evictable_full_rows(sched.tree_cache))
        self.assertGreater(tree_evictable_full_rows(sched.tree_cache), 0)

    def test_can_fail_a_tree_with_only_the_attribute_reads_as_None(self):
        import types

        attr_only = types.SimpleNamespace(full_evictable_size_=12)
        self.assertIsNone(
            tree_evictable_full_rows(attr_only),
            "the reader accepted an attribute-only tree, so the W29 defect "
            "would pass through this suite again",
        )


class TestTheSeamReachesTheStateItPromises(CustomTestCase):
    def test_every_resident_is_retracted_readmitted_and_off_the_live_set(self):
        sched = FaithfulCensusScheduler()
        released = _run_the_seam(sched)
        self.assertEqual(len(released), 3)
        self.assertEqual(sched.live_req_count(), 0)
        # SIX references for three requests: `_live_reqs` deduplicates by
        # identity, `consume_retracted_from_live_universe` does not -- it has
        # to retire the reference in EVERY structure that names it. Three
        # requests spread over running_mbs[0] (1), running_mbs[1] (2),
        # running_batch (2) and last_batch (1) is six. A count of three would
        # mean a structure was left holding a freed request, which is W27.
        self.assertEqual(sched.retired_refs, 6)
        self.assertEqual(sched.readmitted, 3)
        self.assertEqual(
            [r.rid for r in sched.waiting_queue],
            ["r0-0", "r0-1", "r0-2"],
            "re-admission must restore kv_arrival_seq order",
        )

    def test_no_row_belongs_to_nobody_afterwards(self):
        # W27-retry's 152-rows-per-cycle leak, as an invariant.
        sched = FaithfulCensusScheduler()
        _run_the_seam(sched)
        self.assertEqual(sched.orphaned_rows(), 0)
        self.assertEqual(sched.ledger.outstanding(), 0)
        self.assertEqual(sched.ledger.available_size(), sched.ledger.total)
        self.assertEqual(sched.req_to_token_pool.outstanding(), 0)

    def test_the_drop_evicted_before_it_reset(self):
        sched = FaithfulCensusScheduler()
        _run_the_seam(sched)
        self.assertEqual(sched.tree_cache.evict_calls, 1)
        self.assertEqual(sched.tree_cache.resets, 1)

    def test_the_state_copy_ran_while_the_rows_were_still_held(self):
        # `seam_copy_state` runs BEFORE `release_kv_cache`, "while the rows
        # still hold live bytes". A double that released first would record a
        # copy over zero rows and every #783 assertion above it would be a
        # copy of nothing.
        sched = FaithfulCensusScheduler()
        _run_the_seam(sched)
        for req in sched.residents:
            self.assertEqual(req.rows_at_offload, ROWS_PER_RESIDENT)
            self.assertEqual(req.offloaded_at_extent, req.kv_committed_len)

    def test_the_seam_stamped_its_own_retractions(self):
        # W30: the seam stamps `seam_readmit_epoch` at exactly one site, so
        # its re-admissions are separable from ordinary OOM preemption.
        sched = FaithfulCensusScheduler()
        sched.phase_flip_runtime = type("R", (), {"epoch": 11})()
        _run_the_seam(sched)
        self.assertEqual([r.seam_readmit_epoch for r in sched.residents], [11] * 3)

    def test_an_idle_instance_still_drops_its_tree(self):
        sched = FaithfulCensusScheduler(n_residents=0)
        released = _run_the_seam(sched)
        self.assertEqual(released, [])
        self.assertEqual(sched.tree_cache.resets, 1)
        self.assertEqual(sched.orphaned_rows(), 0)


class TestTheOrderIsTheLaw(CustomTestCase):
    """PLANTED VIOLATIONS. Each one must be caught by the double."""

    def test_can_fail_reset_before_retract_reproduces_the_825_crash(self):
        # #825, three ranks down, 2026-08-23: `reset()` installs a NEW root,
        # so a parked request's parent chain no longer terminates and
        # `dec_lock_ref` runs off the top into None. The double must
        # reproduce that, or it cannot tell the lawful order from the fatal
        # one and every ordering assertion in the suite is decorative.
        sched = FaithfulCensusScheduler()
        sched.tree_cache.reset()  # the #825 ordering
        with self.assertRaises(AttributeError) as caught:
            sched.tree_cache.cache_finished_req(sched.residents[0], is_insert=False)
        self.assertIn("NoneType", str(caught.exception))

    def test_can_fail_a_drop_that_only_resets_orphans_the_trees_rows(self):
        # W27-retry: a bare `tree.reset()` is a BOOKKEEPING reset. Planting it
        # in place of `drop_prefix_tree_returning_rows` must leave rows that
        # belong to nobody, or the ledger cannot see the leak it exists for.
        sched = FaithfulCensusScheduler()
        held_by_tree = len(sched.ledger.held.get("tree", ()))
        self.assertGreater(held_by_tree, 0)
        sched.tree_cache.reset()  # the leaking drop
        self.assertEqual(sched.orphaned_rows(), held_by_tree)

    def test_the_shipped_drop_returns_those_same_rows(self):
        # ... and the real one does not leak them.
        sched = FaithfulCensusScheduler()
        held_by_tree = len(sched.ledger.held.get("tree", ()))
        returned = drop_prefix_tree_returning_rows(sched.tree_cache)
        self.assertEqual(returned, held_by_tree)
        self.assertEqual(sched.orphaned_rows(), 0)

    def test_can_fail_a_retraction_that_frees_without_retiring_stays_live(self):
        # THE W27 ROOT. `retract_all` releases rows, mamba slot and tree lock,
        # and leaves the Req referenced by every batch structure. Skipping
        # `consume_retracted_from_live_universe` must leave the request
        # enumerable with its resources gone -- which is how W27 died in
        # `resident_mamba_slots`, one second after the retraction.
        sched = FaithfulCensusScheduler()
        retract, _drop = build_cutover_release(sched)
        reqs = list(_live_reqs(sched))
        retract(reqs)  # freed, but never consumed out of the live universe
        self.assertEqual(sched.live_req_count(), 3)
        self.assertEqual(sched.seam_carried_bytes(), 0)

    def test_a_missing_step_is_refused_rather_than_skipped(self):
        sched = FaithfulCensusScheduler()
        with self.assertRaises(SeamOrderError):
            release_residents_for_cutover(
                sched.residents, retract=None, reset_tree=sched.tree_cache.reset
            )
        with self.assertRaises(SeamOrderError):
            release_residents_for_cutover(
                sched.residents, retract=lambda rs: rs, reset_tree=None
            )

    def test_can_fail_editing_reqs_instead_of_filtering_desynchronises(self):
        # `consume_retracted_from_live_universe` uses `filter_batch(
        # keep_indices=...)` "because a batch carries per-request tensors
        # alongside the list and a raw list edit desynchronises them". The
        # double has to be able to SEE that, or the choice is untested.
        batch = FaithfulBatch(FaithfulCensusScheduler().residents)
        batch.assert_in_sync()
        batch.reqs = batch.reqs[1:]  # the raw edit
        with self.assertRaises(AssertionError):
            batch.assert_in_sync()

    def test_the_shipped_consume_keeps_the_batch_in_sync(self):
        sched = FaithfulCensusScheduler()
        consume_retracted_from_live_universe(sched, sched.residents[:1])
        sched.assert_batches_in_sync()


class TestTheRefusalsAreStillReachable(CustomTestCase):
    """The double must not accidentally satisfy a guard it should trip."""

    def test_a_scheduler_without_a_resettable_tree_is_refused(self):
        import types

        sched = FaithfulCensusScheduler()
        sched.tree_cache = types.SimpleNamespace()  # a ChunkCache has no reset
        self.assertIsNone(build_cutover_release(sched))

    def test_the_real_double_is_accepted(self):
        self.assertIsNotNone(build_cutover_release(FaithfulCensusScheduler()))


if __name__ == "__main__":
    unittest.main()
