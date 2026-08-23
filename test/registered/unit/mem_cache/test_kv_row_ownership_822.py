"""#822: the KV row ownership invariant, red-first against four measured specimens.

Every number in this file was measured on this rig. None is illustrative.

SOURCES
=======

#816 -- /spinning/evidence-665-f1/boot_816_core_0823_0608.log,
    "KV-BACKING exposure clamp (after recovery)". TWELVE firings, four per
    rank, at five distinct second-marks (06:14:21, 06:14:22, 06:18:15,
    06:19:43, 06:32:05). The brief for this task said three; three was the
    number of cited log POSITIONS, not the rate. Each rank reported the SAME
    three numbers on all four of its firings::

        rank   exposed   committed   unbacked
        PP0    466994    212992      254002
        PP1    466994    124928      342066
        PP2    466994    133120      333874

    Provenance note, because it matters for how much these numbers prove: the
    4-tuple appears in that BOOT LOG, not anywhere in the source tree. The
    in-tree #816 specimen is a different boot (417850 exposed vs 105413
    committed, ``kv_backing_relief.py:2674-2679``), and the three caps
    212992/133120/124928 also appear in commit 689161de77, which is tagged
    #802, not #796. Both are covered below so the suite does not rest on one
    reading of one log.

#814 -- 340384 rows unaccounted, "73% of the PP1 pool". The in-tree companion
    figures are "capped at 124928 of 465190" (``phase_flip_spill.py:1344,1347``)
    and 340262 of 465190 (``FEATURE_CATALOG.md:585-596``).

#796/#802 -- PP-space id 344009 still live against a TP cap of 212992 after a
    cutover. Fix 689161de77 cleared ONE site (``flip_pending_from_live_fn``);
    what makes the shape possible is that nothing retires the SPACE.

#717/#722 -- backing shrank to 69054 rows under a highest live row of 233289
    (commit 675793cdc8, describing the crash left by the reverted first attempt
    c4e557963e). #722 is this fork's retroactive label for that failure SHAPE;
    there is no #722 commit. The shape is what the law has to catch.

WHY THE MUTANT CLASS AT THE BOTTOM IS NOT OPTIONAL
==================================================

A checker that cannot fail is not a checker, and this whole task exists because
four checks that each looked right were each blind to the next specimen. So for
every law there is a test that DISABLES it and asserts the matching specimen
goes undetected. That is the can-fail proof in the danger direction: it shows
the specimen is caught BY that law and not incidentally by another one.
"""

import unittest

from sglang.srt.mem_cache.kv_row_ownership import (
    Law,
    RowOwnershipAuthority,
    RowSpace,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


# Measured, boot 2026-08-23 06:08: identical on all four firings of each rank.
# That the numbers do not drift across the boot is itself the finding -- a leak
# accumulates, a structural over-exposure does not.
SPECIMEN_816 = (
    ("PP0", 466994, 212992, 254002),
    ("PP1", 466994, 124928, 342066),
    ("PP2", 466994, 133120, 333874),
)

# The in-tree #816 specimen, from a different boot (kv_backing_relief.py:2674).
SPECIMEN_816_INTREE = (417850, 105413, 312437)

# #717/#722: the counter-form.
SPECIMEN_722_BACKING = 69054
SPECIMEN_722_HIGHEST_LIVE = 233289

# #796/#802: the id that outlived its space.
SPECIMEN_796_STALE_ID = 344009
SPECIMEN_796_TP_CAP = 212992


def laws(violations):
    """The set of laws broken. Tests assert on THIS, never on message text.

    A reworded log line must not be able to turn a red test green -- that is
    the #505 silent-wrongness class applied to the suite itself.
    """
    return {v.law for v in violations}


def only(violations, law):
    """The single violation of ``law``, asserting there is exactly one."""
    hits = [v for v in violations if v.law is law]
    assert len(hits) == 1, f"expected exactly one {law}, got {hits}"
    return hits[0]


def full_owner(authority, space, *, owner="pool"):
    """Give every committed row exactly one owner, so only the law under test fires."""
    authority.declare(owner, range(space.reserved, space.committed))


class TestExposureLaw(CustomTestCase):
    """#816: exposed id space must not exceed the committed backing."""

    def test_three_measured_ranks(self):
        """All three ranks, with the exact unbacked delta each clamp reported."""
        for rank, exposed, committed, unbacked in SPECIMEN_816:
            with self.subTest(rank=rank):
                space = RowSpace(exposed=exposed, committed=committed)
                auth = RowOwnershipAuthority(space)
                full_owner(auth, space)

                v = only(auth.audit(), Law.EXPOSURE)
                self.assertEqual(v.rows, unbacked)
                # The delta is the arithmetic, not a remembered constant.
                self.assertEqual(v.rows, exposed - committed)

    def test_in_tree_specimen(self):
        """The other #816 boot, so the law does not rest on one log reading."""
        exposed, committed, unbacked = SPECIMEN_816_INTREE
        space = RowSpace(exposed=exposed, committed=committed)
        auth = RowOwnershipAuthority(space)
        full_owner(auth, space)

        self.assertEqual(only(auth.audit(), Law.EXPOSURE).rows, unbacked)

    def test_definition_is_shared_with_816(self):
        """There must be exactly ONE definition of over-exposed in the tree.

        A second copy here is precisely the failure this module exists to end
        (#345/#352/#355: "one consumer never got the treatment"). Pin the reuse
        so a later refactor cannot quietly fork the definition.
        """
        from sglang.srt.managers.kv_backing_relief import exposure_over_backing

        for _, exposed, committed, unbacked in SPECIMEN_816:
            self.assertEqual(exposure_over_backing(exposed, committed), unbacked)

    def test_sound_state_is_silent(self):
        """The other direction: a lawful space must produce ZERO violations.

        Without this the suite could pass by always returning a violation, which
        is the same defect as never returning one.
        """
        space = RowSpace(exposed=124928, committed=124928)
        auth = RowOwnershipAuthority(space)
        full_owner(auth, space)

        self.assertEqual(auth.audit(), [])


class TestOwnershipLaw(CustomTestCase):
    """#814: every committed row has exactly one owner."""

    def test_unaccounted_is_exposure_not_ownership(self):
        """THE #814 FINDING, as a test: the census asked the wrong range.

        #814 reported 340384 rows unaccounted = "73% of the PP1 pool", and
        340384 / 466994 = 0.7289. The denominator is the EXPOSED space, not the
        124928-row backing. Those rows were not owned by nobody -- they did not
        exist. They are #816's unbacked rows, counted a second time under the
        wrong law.

        So with the exact PP1 numbers and every COMMITTED row properly owned,
        a correct authority reports EXPOSURE and says nothing about ownership.
        """
        space = RowSpace(exposed=466994, committed=124928)
        auth = RowOwnershipAuthority(space)
        full_owner(auth, space)

        found = auth.audit()
        self.assertEqual(laws(found), {Law.EXPOSURE})
        self.assertEqual(only(found, Law.EXPOSURE).rows, 342066)
        # The 73% that made #814 look like a leak.
        self.assertAlmostEqual(342066 / 466994, 0.7325, places=3)

    def test_unenumerated_second_owner(self):
        """The real ownership defect: rows inside the backing with no owner.

        ``_census_owner_probe`` (phase_flip_runtime.py:3773) found this on
        metal -- ~94000 rows, 21% of a 448698-row pool, held by a pool object
        the census never enumerated, FLAT across four censuses (a leak
        accumulates; an unenumerated owner does not). The probe reports it.
        Nothing gated on it. This is that gate.
        """
        committed = 448698
        missing = 94000
        space = RowSpace(exposed=committed, committed=committed)
        auth = RowOwnershipAuthority(space)
        # The census's own world: one allocator, one tree -- and a hole where
        # the second pool object's rows are.
        auth.declare("free_list", range(space.reserved, committed - missing))

        found = auth.audit()
        self.assertEqual(laws(found), {Law.EXCLUSIVITY})
        self.assertEqual(only(found, Law.EXCLUSIVITY).rows, missing)

    def test_double_ownership_is_caught_even_on_a_partial_view(self):
        """Two owners on one row is silent corruption, not a crash.

        A partial view can MISS an owner; it can never INVENT one. So the
        unowned half of the law is suppressible and this half never is.
        """
        space = RowSpace(exposed=1000, committed=1000)
        auth = RowOwnershipAuthority(space)
        auth.declare("radix_cache", range(1, 700))
        auth.declare("free_list", range(650, 1000))  # 50 rows overlap

        found = auth.audit(expect_full_coverage=False)
        self.assertEqual(laws(found), {Law.EXCLUSIVITY})
        self.assertEqual(only(found, Law.EXCLUSIVITY).rows, 50)

    def test_padding_row_is_not_a_leak(self):
        """Row 0 is the cuda-graph padding row and belongs to no one by design.

        ``free_pages`` is ``arange(1, size + 1)`` (allocator/token.py:39). If the
        law counted row 0 it would report a one-row leak on every healthy boot,
        and a checker that is always red hides every real regression (#380/#585).
        """
        space = RowSpace(exposed=1000, committed=1000)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, 1000))

        self.assertEqual(auth.audit(), [])


class TestCoverageLaw(CustomTestCase):
    """#717/#722: no live row may sit at or above the committed backing."""

    def test_backing_below_the_highest_live_row(self):
        """The counter-form a downward clamp is blind to by construction."""
        space = RowSpace(exposed=SPECIMEN_722_BACKING, committed=SPECIMEN_722_BACKING)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(space.reserved, SPECIMEN_722_BACKING))
        # One request still holding a row far above what is now backed.
        auth.declare("resident:req-a", [SPECIMEN_722_HIGHEST_LIVE])

        found = auth.audit()
        self.assertIn(Law.COVERAGE, laws(found))
        v = only(found, Law.COVERAGE)
        self.assertEqual(v.rows, 1)
        self.assertIn(SPECIMEN_722_HIGHEST_LIVE, v.sample)

    def test_clamping_does_not_repair_this_shape(self):
        """A cap cannot make an unmapped live row addressable again.

        ``clamp_exposure_to_backing`` says this in prose (kv_backing_relief.py
        :2688-2693: "a grow, not a cap, is what fixes this"). Here it is a test:
        lowering exposure to the backing clears EXPOSURE and leaves COVERAGE
        standing, which is the honest outcome.
        """
        space = RowSpace(exposed=233290, committed=SPECIMEN_722_BACKING)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(space.reserved, SPECIMEN_722_BACKING))
        auth.declare("resident:req-a", [SPECIMEN_722_HIGHEST_LIVE])
        self.assertEqual(laws(auth.audit()), {Law.EXPOSURE, Law.COVERAGE})

        auth.set_backing(exposed=SPECIMEN_722_BACKING)  # what the clamp does
        self.assertEqual(laws(auth.audit()), {Law.COVERAGE})


class TestRetirementLaw(CustomTestCase):
    """#796/#802: a cutover retires the whole old id space in ONE step."""

    def test_id_that_outlived_its_space(self):
        """PP-space id 344009 against a TP cap of 212992."""
        space = RowSpace(exposed=466994, committed=466994)
        auth = RowOwnershipAuthority(space)
        auth.declare("flip_live_extent", [SPECIMEN_796_STALE_ID])

        # The cutover. One call.
        dropped = auth.retire(
            exposed=SPECIMEN_796_TP_CAP, committed=SPECIMEN_796_TP_CAP
        )
        self.assertEqual(dropped, 1)

        # A holder that did not get the memo redeclares under the OLD epoch --
        # which is exactly what "one writer and no clearer" produces.
        auth.declare("flip_live_extent", [SPECIMEN_796_STALE_ID], epoch=0)

        found = auth.audit(expect_full_coverage=False)
        self.assertEqual(laws(found), {Law.RETIREMENT})
        v = only(found, Law.RETIREMENT)
        self.assertIn(SPECIMEN_796_STALE_ID, v.sample)

    def test_retirement_needs_no_enumeration_of_holders(self):
        """The generalization of 689161de77, and the whole point of the epoch.

        That fix visited ONE holder. The shape recurs wherever ANY holder is
        not visited, and enumerating holders is the thing that kept being
        incomplete. So retirement invalidates the SPACE: here five holders are
        retired by a single call that never looks at any of them.
        """
        space = RowSpace(exposed=466994, committed=466994)
        auth = RowOwnershipAuthority(space)
        for i in range(5):
            auth.declare(f"holder-{i}", [300000 + i])

        self.assertEqual(auth.retire(exposed=212992, committed=212992), 5)
        self.assertEqual(auth.owners(), ())
        self.assertEqual(auth.epoch, 1)

    def test_stale_claim_is_named_stale_not_out_of_range(self):
        """Order matters: a retired claim's ids are meaningless in the new space.

        Id 344009 is also above the 212992 backing, so a checker that ran
        COVERAGE first would report "row out of range" and send the reader
        hunting a backing bug. The defect is that the id survived a cutover.
        """
        space = RowSpace(exposed=466994, committed=466994)
        auth = RowOwnershipAuthority(space)
        auth.retire(exposed=SPECIMEN_796_TP_CAP, committed=SPECIMEN_796_TP_CAP)
        auth.declare("stale", [SPECIMEN_796_STALE_ID], epoch=0)

        found = auth.audit(expect_full_coverage=False)
        self.assertEqual(laws(found), {Law.RETIREMENT})
        self.assertNotIn(Law.COVERAGE, laws(found))

    def test_a_backing_dial_move_is_not_a_cutover(self):
        """Growing or shrinking inside one id space must NOT retire anyone.

        The #330 dial moves the backing at runtime many times per boot. If that
        bumped the epoch, every live request would read as a #796 survivor and
        the law would be noise.
        """
        space = RowSpace(exposed=1000, committed=1000)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, 900))
        auth.declare("resident:req-a", range(900, 1000))

        auth.set_backing(committed=1000, exposed=1000)
        self.assertEqual(auth.epoch, 0)
        self.assertEqual(auth.audit(), [])

    def test_retirement_is_one_step(self):
        """No reader can see a bumped epoch with stale claims still attached.

        Single-process atomicity only. Whether all ranks cross the cutover
        together under load is a property of the flip's commit protocol, is only
        provable on metal, and is carried as an 18-lane window item -- it is NOT
        asserted here.
        """
        space = RowSpace(exposed=466994, committed=466994)
        auth = RowOwnershipAuthority(space)
        auth.declare("holder", [SPECIMEN_796_STALE_ID])

        auth.retire(exposed=212992, committed=212992)
        # Epoch advanced AND claims gone, observed together.
        self.assertEqual((auth.epoch, auth.owners()), (1, ()))


class TestCensusBridge(CustomTestCase):
    """The census stops producing a number without a verdict."""

    def test_census_sets_become_named_laws(self):
        """``_pool_census`` derives ``unaccounted`` from exactly these sets.

        Fed through the authority, the same three sets yield a named law
        instead of one ambiguous integer that could mean a leak, an
        unenumerated owner, or an over-exposed id space.
        """
        space = RowSpace(exposed=466994, committed=124928)
        auth = RowOwnershipAuthority(space)

        found = auth.observe_census(
            free_rows=range(1, 100000),
            cached_rows=range(100000, 124928),
            withheld_rows=(),
        )
        self.assertEqual(laws(found), {Law.EXPOSURE})

    def test_violation_counts_are_a_regression_metric(self):
        """#822 item 5: the #816 clamp's firing rate must trend to zero.

        The clamp stays as belt-and-suspenders, but under the authority it must
        have nothing left to correct. Counting per law is only meaningful
        because the laws do not overlap -- see the #814 range choice.
        """
        space = RowSpace(exposed=466994, committed=124928)
        auth = RowOwnershipAuthority(space)
        full_owner(auth, space)

        self.assertEqual(auth.violation_counts[Law.EXPOSURE], 0)
        auth.audit()
        auth.audit()
        self.assertEqual(auth.violation_counts[Law.EXPOSURE], 2)
        self.assertEqual(auth.violation_counts[Law.EXCLUSIVITY], 0)


class TestCallSiteAdapters(CustomTestCase):
    """The wiring, smoke-tested. A slice nobody executed is not a slice (#585)."""

    def test_authority_is_attached_once(self):
        """One authority per rank. A privately constructed one is not authority."""
        from sglang.srt.mem_cache.kv_row_ownership import AUTHORITY_ATTR, authority_for

        class Host:
            pass

        host = Host()
        first = authority_for(host, exposed=1000, committed=1000)
        second = authority_for(host, exposed=7, committed=7)
        self.assertIs(first, second)
        self.assertIs(getattr(host, AUTHORITY_ATTR), first)
        # The second call must not silently re-open the space behind the first.
        self.assertEqual(second.space.exposed, 1000)

    def test_unmeasurable_backing_is_unanswerable_not_clean(self):
        """``committed=None`` must never be reported as a healthy pool.

        Substituting the id space for the committed span is what cost a boot on
        2026-08-11 (kv_backing_relief.py:1180) and it would report the exact
        #816 state as sound.
        """
        from sglang.srt.mem_cache.kv_row_ownership import audit_pool_census

        auth = RowOwnershipAuthority(RowSpace(exposed=466994, committed=0))
        with self.assertLogs("sglang.srt.mem_cache.kv_row_ownership", "WARNING") as log:
            found = audit_pool_census(
                auth,
                exposed=466994,
                committed=None,
                free_rows=(),
                cached_rows=(),
                why="post-cutover",
            )
        self.assertEqual(found, [])
        self.assertIn("SKIPPED", "".join(log.output))
        # And it must not have moved the space to make itself look answerable.
        self.assertEqual(auth.space.committed, 0)

    def test_census_shaped_input_reproduces_the_816_verdict(self):
        """Census sets in, named law out -- the PP1 numbers end to end."""
        from sglang.srt.mem_cache.kv_row_ownership import audit_pool_census

        auth = RowOwnershipAuthority(RowSpace(exposed=0, committed=0))
        found = audit_pool_census(
            auth,
            exposed=466994,
            committed=124928,
            free_rows=range(1, 100000),
            cached_rows=range(100000, 124928),
            withheld_rows=(),
            why="post-cutover pp_to_tp",
        )
        self.assertEqual(laws(found), {Law.EXPOSURE})
        self.assertEqual(only(found, Law.EXPOSURE).rows, 342066)

    def test_runtime_helpers_are_best_effort(self):
        """The audit and the retirement must not be able to raise into the seam.

        They run inside the flip's no-return region. A checker that can crash
        the flip it is watching would just be an eighth crash root, so both are
        exercised here against a runtime whose scheduler is missing entirely.
        """
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        class Bare:
            """Only the attributes the two helpers touch."""

            _census_scheduler = None

            _committed_backing_rows = PhaseFlipRuntime._committed_backing_rows
            _census_ownership_audit = PhaseFlipRuntime._census_ownership_audit
            _retire_row_id_space = PhaseFlipRuntime._retire_row_id_space

        bare = Bare()
        self.assertIsNone(bare._committed_backing_rows())
        # Neither may raise, and neither may leave the flip in a new state.
        bare._census_ownership_audit("smoke", None, 1000, set(), set(), set())
        bare._retire_row_id_space("pp_to_tp")

    def test_runtime_helpers_reach_the_authority_on_the_happy_path(self):
        """The wiring's GOOD path, not only its guarded one.

        The other smoke test proves the helpers cannot raise when nothing is
        wired -- which a pair of no-ops would also pass. This one gives them a
        scheduler shaped like the real one and asserts the audit actually
        reached the authority and the cutover actually retired the space.
        """
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR
        from sglang.srt.mem_cache.kv_row_ownership import AUTHORITY_ATTR

        class Relief:
            def _current_rows(self):
                return 124928

        class Alloc:
            size = 466994

        scheduler = type("Sched", (), {})()
        setattr(scheduler, KV_BACKING_RELIEF_ATTR, Relief())
        scheduler.token_to_kv_pool_allocator = Alloc()

        class Runtime:
            _census_scheduler = scheduler

            _committed_backing_rows = PhaseFlipRuntime._committed_backing_rows
            _census_ownership_audit = PhaseFlipRuntime._census_ownership_audit
            _retire_row_id_space = PhaseFlipRuntime._retire_row_id_space

        rt = Runtime()
        self.assertEqual(rt._committed_backing_rows(), 124928)

        # The census leg: the PP1 specimen, end to end through the adapter.
        rt._census_ownership_audit(
            "post-cutover pp_to_tp",
            Alloc(),
            466994,
            set(range(1, 100000)),
            set(range(100000, 124928)),
            set(),
        )
        auth = getattr(rt, AUTHORITY_ATTR)
        self.assertEqual(auth.violation_counts[Law.EXPOSURE], 1)
        self.assertEqual(auth.space.committed, 124928)

        # The cutover leg: one call, and the space is a new epoch with no claims.
        rt._retire_row_id_space("pp_to_tp")
        self.assertEqual(auth.epoch, 1)
        self.assertEqual(auth.owners(), ())

    def test_cutover_retires_before_the_post_cutover_census(self):
        """Order is load-bearing, so pin it against a reordering refactor.

        If retirement ran after the post-cutover census, a surviving
        pre-cutover claim would be audited under the OLD epoch and report as an
        out-of-range row -- sending the reader after a backing bug instead of
        after the id that outlived its space.
        """
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._execute)
        retire = src.index('"_retire_row_id_space"')
        census = src.index('self._pool_census("post-cutover", direction)')
        cutover = src.index("self._cutover_fn(direction)")
        self.assertLess(cutover, retire)
        self.assertLess(retire, census)

    def test_the_auditor_cannot_break_the_census_it_rides_on(self):
        """A REGRESSION THIS BUILD ACTUALLY CAUSED, pinned so it cannot return.

        `_pool_census` is exercised elsewhere bound to a plain namespace
        (test_census_withheld_term_814.py). The first version of the wiring
        called `self._census_ownership_audit(...)` directly: the ATTRIBUTE
        LOOKUP raised AttributeError before the method's own try/except could
        run, the census's outer handler caught it, and its line became
        "pool census (at-arm) failed: ..." instead of the census. Six #814
        tests went red -- and the census's whole reason to exist is to still
        say something when the thing around it is broken.

        The rule "a census must never affect the flip it is watching" extends
        one level down. So the lookup is guarded too, and this pins it.
        """
        import types

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        class Alloc:
            size = 64
            page_size = 1
            residency_withheld_slots = 0
            free_pages = types.SimpleNamespace(tolist=lambda: list(range(1, 65)))
            release_pages = types.SimpleNamespace(tolist=lambda: [])

            def available_size(self):
                return 64

        tree = types.SimpleNamespace(
            all_values_flatten=lambda: types.SimpleNamespace(tolist=lambda: [])
        )
        scheduler = types.SimpleNamespace(
            token_to_kv_pool_allocator=Alloc(), tree_cache=tree
        )
        # Deliberately WITHOUT _census_ownership_audit / _census_owner_probe,
        # which is the shape the #814 tests use.
        bare = types.SimpleNamespace(
            _census_scheduler=scheduler,
            _census_owner_probe=lambda *a, **k: None,
        )

        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", "WARNING"
        ) as log:
            PhaseFlipRuntime._pool_census(bare, "at-arm", "pp_to_tp")
        text = "".join(log.output)
        self.assertIn("POOL CENSUS", text)
        self.assertNotIn("pool census (at-arm) failed", text)


class TestMutantsInTheDangerDirection(CustomTestCase):
    """Each law is proven load-bearing by removing it and losing its specimen.

    This is the can-fail proof. Without it, four checks that never fire look
    exactly like four checks that cannot fire -- which is how this task got
    seven crash roots in six days.
    """

    @staticmethod
    def _without(violations, law):
        """The mutant: the authority with one law deleted."""
        return [v for v in violations if v.law is not law]

    def test_dropping_exposure_loses_816(self):
        space = RowSpace(exposed=466994, committed=124928)
        auth = RowOwnershipAuthority(space)
        full_owner(auth, space)
        found = auth.audit()

        self.assertEqual(laws(found), {Law.EXPOSURE})  # armed
        self.assertEqual(self._without(found, Law.EXPOSURE), [])  # blind

    def test_dropping_exclusivity_loses_814(self):
        space = RowSpace(exposed=448698, committed=448698)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, 448698 - 94000))
        found = auth.audit()

        self.assertEqual(laws(found), {Law.EXCLUSIVITY})
        self.assertEqual(self._without(found, Law.EXCLUSIVITY), [])

    def test_dropping_coverage_loses_722(self):
        space = RowSpace(exposed=SPECIMEN_722_BACKING, committed=SPECIMEN_722_BACKING)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, SPECIMEN_722_BACKING))
        auth.declare("resident:req-a", [SPECIMEN_722_HIGHEST_LIVE])
        found = auth.audit()

        self.assertEqual(laws(found), {Law.COVERAGE})
        self.assertEqual(self._without(found, Law.COVERAGE), [])

    def test_dropping_retirement_loses_796(self):
        """The sharpest mutant: WITHOUT retirement the state looks LAWFUL.

        Not merely unreported -- lawful. Id 344009 under the pre-cutover space
        breaks no other law, so the three remaining checks all pass and the
        crash arrives anyway. That is why retirement had to be built and could
        not be audited for after the fact.
        """
        space = RowSpace(exposed=466994, committed=466994)
        auth = RowOwnershipAuthority(space)
        auth.retire(exposed=SPECIMEN_796_TP_CAP, committed=SPECIMEN_796_TP_CAP)
        auth.declare("stale", [SPECIMEN_796_STALE_ID], epoch=0)
        found = auth.audit(expect_full_coverage=False)

        self.assertEqual(laws(found), {Law.RETIREMENT})
        self.assertEqual(self._without(found, Law.RETIREMENT), [])

    def test_no_law_is_redundant(self):
        """Every law has a state only IT catches -- the union is minimal.

        If two laws always fired together, one of them would be decoration and
        the suite would be measuring itself.
        """
        seen = set()
        for law, build in (
            (Law.EXPOSURE, self._state_816),
            (Law.EXCLUSIVITY, self._state_814),
            (Law.COVERAGE, self._state_722),
            (Law.RETIREMENT, self._state_796),
        ):
            auth, kwargs = build()
            self.assertEqual(laws(auth.audit(**kwargs)), {law})
            seen.add(law)
        self.assertEqual(seen, set(Law))

    @staticmethod
    def _state_816():
        space = RowSpace(exposed=466994, committed=124928)
        auth = RowOwnershipAuthority(space)
        full_owner(auth, space)
        return auth, {}

    @staticmethod
    def _state_814():
        space = RowSpace(exposed=448698, committed=448698)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, 448698 - 94000))
        return auth, {}

    @staticmethod
    def _state_722():
        space = RowSpace(exposed=SPECIMEN_722_BACKING, committed=SPECIMEN_722_BACKING)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, SPECIMEN_722_BACKING))
        auth.declare("resident:req-a", [SPECIMEN_722_HIGHEST_LIVE])
        return auth, {}

    @staticmethod
    def _state_796():
        space = RowSpace(exposed=466994, committed=466994)
        auth = RowOwnershipAuthority(space)
        auth.retire(exposed=SPECIMEN_796_TP_CAP, committed=SPECIMEN_796_TP_CAP)
        auth.declare("stale", [SPECIMEN_796_STALE_ID], epoch=0)
        return auth, {"expect_full_coverage": False}


if __name__ == "__main__":
    unittest.main()


class TestClampFiringRateIsAMetric(CustomTestCase):
    """#822 item 5: the #816 clamp stays, and its firing rate is the metric.

    A law with no actuator under it is a comment, so `clamp_exposure_to_backing`
    keeps its job. What changes is that its firing rate now has a PARSER and a
    MEASURED baseline, so "under the authority the clamp has nothing left to
    correct" is a claim a later boot can falsify instead of a promise.
    """

    LOG = "/spinning/evidence-665-f1/boot_816_core_0823_0608.log"

    def test_parser_reads_a_firing(self):
        """Always runs, log or no log: the parser itself must be exercised."""
        from sglang.srt.mem_cache.kv_row_ownership import parse_clamp_firings

        line = (
            "[2026-08-23 06:14:21 PP1] KV-BACKING exposure clamp (after "
            "recovery): the allocator could hand out 466994 rows while only "
            "124928 are committed, so 342066 rows had no page behind them. "
            "Capped at 124928."
        )
        (firing,) = parse_clamp_firings([line])
        self.assertEqual(firing.rank, "PP1")
        self.assertEqual((firing.exposed, firing.committed), (466994, 124928))
        self.assertEqual(firing.unbacked, 342066)
        self.assertTrue(firing.is_self_consistent)

    def test_parser_ignores_everything_else(self):
        """A counter that counts the wrong lines is worse than no counter.

        The INDIKATOR-GESETZ, applied to this indicator: proven in BOTH
        directions -- it finds the firing above, and it finds nothing in lines
        that merely mention the same subsystem.
        """
        from sglang.srt.mem_cache.kv_row_ownership import parse_clamp_firings

        noise = [
            "[2026-08-23 06:14:21 PP1] KV-BACKING proposal rows current=407051",
            "[2026-08-23 06:14:21 PP1] PHASE-FLIP POOL CENSUS at-arm: unaccounted=340384",
            "",
        ]
        self.assertEqual(parse_clamp_firings(noise), [])

    def test_measured_baseline_against_the_real_boot_log(self):
        """The baseline is READ OFF the boot, never asserted.

        Twelve firings, four per rank. The brief for this task said three --
        that was the number of cited log POSITIONS, not the rate. A metric
        seeded from three would have scored a nine-firing boot as an
        improvement.
        """
        import os

        from sglang.srt.mem_cache.kv_row_ownership import (
            CLAMP_BASELINE_0823,
            CLAMP_BASELINE_ROWS_0823,
            clamp_firing_census,
            parse_clamp_firings,
        )

        if not os.path.exists(self.LOG):
            self.skipTest(f"specimen boot log not present: {self.LOG}")

        with open(self.LOG, errors="replace") as fh:
            firings = parse_clamp_firings(fh)

        self.assertEqual(len(firings), 12)
        with open(self.LOG, errors="replace") as fh:
            self.assertEqual(clamp_firing_census(fh), CLAMP_BASELINE_0823)

        # Every firing self-consistent, and every rank's numbers IDENTICAL on
        # all four of its firings: a structural defect, not a drifting leak.
        for firing in firings:
            self.assertTrue(firing.is_self_consistent, firing)
            self.assertEqual(
                (firing.exposed, firing.committed, firing.unbacked),
                CLAMP_BASELINE_ROWS_0823[firing.rank],
            )

    def test_the_law_reproduces_every_logged_firing(self):
        """Close the loop: the authority's verdict must equal what the clamp saw.

        If the law and the actuator under it disagreed on a single one of the
        twelve, one of them would be wrong and there would be no way to tell
        which -- which is the state this whole task exists to leave.
        """
        from sglang.srt.mem_cache.kv_row_ownership import CLAMP_BASELINE_ROWS_0823

        for rank, (exposed, committed, unbacked) in CLAMP_BASELINE_ROWS_0823.items():
            with self.subTest(rank=rank):
                space = RowSpace(exposed=exposed, committed=committed)
                auth = RowOwnershipAuthority(space)
                full_owner(auth, space)
                self.assertEqual(only(auth.audit(), Law.EXPOSURE).rows, unbacked)
