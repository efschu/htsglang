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
"""#770 -- the funding authority, pinned to the numbers that produced the TP lock.

Every quantity in this file is READ OFF a real log, never invented. Source:
``/spinning/evidence-665-f1/boot_816_core_0823_0608.log``, the tp_to_pp
abandons at 06:14:25 / 06:18:15 / 06:18:41 / 06:19:43 and the group shrink
verdicts at 06:28:58 / 06:32:05 / 06:52:31.

Hermetic: no CUDA, no NVML, no pool, no torch. That is deliberate -- the whole
reason the authority is a pure function of a snapshot is so the specimen's
arithmetic can be re-run at a desk.
"""

import unittest

from sglang.srt.managers.funding_authority import (
    CAUSE_FUNDED,
    CAUSE_GRANULARITY,
    CAUSE_PEER_VETO,
    CAUSE_PHANTOM,
    CAUSE_SCARCITY,
    GROUP_NONE,
    GROUP_UNIFORM_CAP,
    MIB,
    RELIEF_LOCAL,
    RELIEF_REBALANCE,
    BreakEvenProvenance,
    FundingAuthority,
    PROV_ENV,
    PROV_FROZEN,
    PROV_MEASURED,
    authority_from_seam_snapshot,
    solve_arming_floor,
    FundingError,
    Post,
    diagnose_floor_band,
    slack_above_uniform_floor,
    uniform_absolute_floor,
)

# -- the specimen, 06:19:43 PP0 ----------------------------------------------
# corridor gate refused: want 2533 MiB, free 2444 -> gap 89 MiB
# census: staging needs 2021, driver free 2444, allocator cache 313,
#         staging reserve kept free 819, seam fixed 228, arena due 437,
#         => SPENDABLE 1625        (2444 - 819 = 1625, checked below)
# KV rung: current=212992 rows, floor=122912, slack=90080, deficit=+1676 MiB
SPEC_WANT_MIB = 2533
SPEC_FREE_MIB = 2444
SPEC_STAGING_NEED_MIB = 2021
SPEC_SPENDABLE_MIB = 1625
SPEC_ALLOCATOR_CACHE_MIB = 313
SPEC_STAGING_RESERVE_MIB = 819
SPEC_SEAM_FIXED_MIB = 228
SPEC_ARENA_DUE_MIB = 437

# The KV row facts. 32 KiB per row, granule 8192 rows (= 256 MiB), from
# "commit chunk 8 MiB across 32 buffers, 32 KiB per row".
ROW_BYTES = 32 * 1024
KV_GRANULE_ROWS = 8192
KV_GRANULE_BYTES = KV_GRANULE_ROWS * ROW_BYTES

# The 06:32:05 group verdict, all three ranks.
PP_CAPS = (212992, 124928, 133120)
PP_FLOOR = 128549


class TestSpecimenArithmetic(unittest.TestCase):
    """Guard the constants themselves: a test built on a misread log is worse
    than no test. Each of these reproduces a figure the log PRINTED."""

    def test_spendable_is_free_minus_staging_reserve(self):
        self.assertEqual(SPEC_FREE_MIB - SPEC_STAGING_RESERVE_MIB, SPEC_SPENDABLE_MIB)

    def test_the_gap_the_briefing_named(self):
        self.assertEqual(SPEC_WANT_MIB - SPEC_FREE_MIB, 89)

    def test_the_real_staging_shortfall(self):
        self.assertEqual(SPEC_STAGING_NEED_MIB - SPEC_SPENDABLE_MIB, 396)

    def test_kv_granule_is_256_mib(self):
        self.assertEqual(KV_GRANULE_BYTES // MIB, 256)


class TestSpecimenTpLock(unittest.TestCase):
    """Form (a): the 06:19:43 abandon must be FUNDED from a named post."""

    def _authority_as_the_specimen_had_it(self):
        """Exactly the posts that existed at 06:19:43, with their real figures."""
        auth = FundingAuthority(rank=0)
        # Registered with the guard, and it paid zero: "reclaimed 0 MiB".
        auth.declare_post(
            Post(
                "allocator-cache",
                available_bytes=SPEC_ALLOCATOR_CACHE_MIB * MIB,
                tier=RELIEF_LOCAL,
                cost=10,
            )
        )
        # Registered with the guard, spilled for the whole PP phase -> dry.
        auth.declare_post(
            Post(
                "draft-weights",
                available_bytes=0,
                tier=RELIEF_REBALANCE,
                cost=20,
                unavailable_reason="already spilled for the PP phase",
            )
        )
        # THE FUNDER THE LOG NAMED AND THE LADDER COULD NOT REACH.
        # slack=90080 rows above the rung floor.
        auth.declare_post(
            Post(
                "kv-slack",
                available_bytes=90080 * ROW_BYTES,
                tier=RELIEF_REBALANCE,
                cost=30,
                granule_bytes=KV_GRANULE_BYTES,
                group_scope=GROUP_UNIFORM_CAP,
            )
        )
        return auth

    def test_the_396_mib_shortfall_is_funded_from_kv_slack(self):
        """The whole point. The log said '[nothing]'; there were 2815 MiB."""
        auth = self._authority_as_the_specimen_had_it()
        want = 396 * MIB
        v = auth.can_fund(
            want,
            # A correct absolute reduction: max(floors) leaves PP0 its slack.
            group_floor_bytes={"kv-slack": 90080 * ROW_BYTES},
            local_slack_bytes={"kv-slack": 90080 * ROW_BYTES},
        )
        self.assertTrue(v.ok, v.describe())
        self.assertEqual(v.cause, CAUSE_FUNDED)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertGreater(drawn["kv-slack"], 0)
        # Rounded UP to one whole granule, never silently down to zero.
        self.assertEqual(drawn["kv-slack"] % KV_GRANULE_BYTES, 0)

    def test_a_refusal_never_says_nothing(self):
        """Law 1. Even a total refusal enumerates every post and its reason."""
        auth = FundingAuthority(rank=0)
        auth.declare_post(Post("allocator-cache", 0, tier=RELIEF_LOCAL, cost=10))
        auth.declare_post(
            Post(
                "draft-weights",
                0,
                cost=20,
                unavailable_reason="already spilled for the PP phase",
            )
        )
        v = auth.can_fund(396 * MIB)
        self.assertFalse(v.ok)
        self.assertEqual(v.cause, CAUSE_SCARCITY)
        text = v.describe()
        # The two failure strings the old line could not tell apart.
        self.assertNotIn("[nothing]", text)
        self.assertIn("allocator-cache", text)
        self.assertIn("draft-weights", text)
        self.assertIn("already spilled", text)
        # Genuine scarcity IS the case where backing off is right.
        self.assertTrue(v.retry_is_pointless)

    def test_every_post_appears_even_when_not_needed(self):
        auth = self._authority_as_the_specimen_had_it()
        v = auth.can_fund(
            1 * MIB,
            group_floor_bytes={"kv-slack": 90080 * ROW_BYTES},
            local_slack_bytes={"kv-slack": 90080 * ROW_BYTES},
        )
        self.assertTrue(v.ok)
        self.assertEqual({d.post for d in v.draws}, set(auth.posts))


class TestPeerVetoForm(unittest.TestCase):
    """Form (b): #812 / kein-bindender-rang. A MIN-style reduction against
    DESIGNED inequality is a defect, and must never be reported as scarcity."""

    def test_absolute_reduction_leaves_the_roomy_rank_its_slack(self):
        floor = uniform_absolute_floor(PP_CAPS, (PP_FLOOR,) * 3)
        self.assertEqual(floor, PP_FLOOR)
        # PP0 has room; PP1's floor exceeds its own cap so it has none. That is
        # a fact about PP1 and must not propagate.
        self.assertEqual(slack_above_uniform_floor(PP_CAPS[0], floor), 84443)
        self.assertEqual(slack_above_uniform_floor(PP_CAPS[1], floor), 0)
        self.assertEqual(slack_above_uniform_floor(PP_CAPS[2], floor), 4571)

    def test_the_agreed_floor_must_clear_the_HIGHEST_live_set(self):
        """Unequal floors are the general case; the specimen's three happened
        to be equal at one instant (128549) and equal floors cannot tell a
        MAX-reduce from a MIN-reduce. This case can.

        DANGER DIRECTION: agreeing the SMALLEST floor shrinks the pool below
        the live set of the rank that needs the most, which is not a lost
        funding opportunity but a correctness failure -- rows still in use.
        The agreed absolute must therefore be the MAXIMUM of the floors, and
        every rank's live set must clear it.
        """
        caps = (212992, 124928, 133120)
        floors = (122912, 128549, 126000)
        agreed = uniform_absolute_floor(caps, floors)
        self.assertEqual(agreed, 128549)
        for f in floors:
            self.assertGreaterEqual(
                agreed, f, "the agreed floor must clear every rank's live set"
            )
        # And it still leaves the roomy rank real slack to fund with.
        self.assertEqual(slack_above_uniform_floor(caps[0], agreed), 84443)

    def test_row_size_model_reconstructs_the_logs_own_deficit(self):
        """A two-way check on ROW_BYTES before any conclusion leans on it.

        The 06:32:05 line printed 'current=212992 ... deficit=+1684 MiB ->
        SHRINK to 159102'. If 32 KiB/row is the right cell, those three
        printed numbers must be mutually consistent -- and they are, to the
        MiB. This is the assert that would catch a wrong cell size before it
        silently rescaled every other figure in this file.
        """
        shrink_rows = 212992 - 159102
        self.assertEqual(shrink_rows * ROW_BYTES // MIB, 1684)

    def test_pp0_slack_alone_covers_the_shortfall_six_times_over(self):
        floor = uniform_absolute_floor(PP_CAPS, (PP_FLOOR,) * 3)
        slack_mib = slack_above_uniform_floor(PP_CAPS[0], floor) * ROW_BYTES // MIB
        # 84443 rows at 32 KiB. NOT the log's 'deficit=+1684 MiB', which is the
        # smaller figure the rung ASKED for; the slack available exceeds it.
        self.assertEqual(slack_mib, 2638)
        self.assertGreater(slack_mib, 1684)  # covers what the rung asked
        self.assertGreater(slack_mib, 6 * 396)  # and the staging shortfall 6x

    def test_a_peer_veto_is_named_as_such_and_not_as_scarcity(self):
        """The specimen's reduction granted the no-change point. The authority
        must say 'your own slack was withheld', not 'there is nothing'."""
        auth = FundingAuthority(rank=0)
        mine = 84443 * ROW_BYTES
        auth.declare_post(
            Post(
                "kv-slack",
                available_bytes=mine,
                granule_bytes=KV_GRANULE_BYTES,
                group_scope=GROUP_UNIFORM_CAP,
            )
        )
        v = auth.can_fund(
            396 * MIB,
            # What the proportion-of-own-cap reduction actually released: nothing.
            group_floor_bytes={"kv-slack": 0},
            local_slack_bytes={"kv-slack": mine},
        )
        self.assertFalse(v.ok)
        self.assertEqual(v.cause, CAUSE_PEER_VETO)
        self.assertIn("withheld by a cross-rank reduction", v.veto_detail)
        # THE LOAD-BEARING ASSERT: a veto is not scarcity, so the exponential
        # stand-down must NOT be armed by it.
        self.assertFalse(v.retry_is_pointless)

    def test_unequal_vectors_are_rejected_not_silently_zipped(self):
        with self.assertRaises(FundingError):
            uniform_absolute_floor((1, 2, 3), (1, 2))


class TestGranularityForm(unittest.TestCase):
    """Law 3: the 3437-rows-against-an-8192-row-granule silent zero."""

    def test_sub_granule_ask_rounds_up_when_the_post_can_afford_it(self):
        post = Post(
            "kv-slack",
            available_bytes=90080 * ROW_BYTES,
            granule_bytes=KV_GRANULE_BYTES,
        )
        drawn, reason = post.creditable(3437 * ROW_BYTES)
        self.assertEqual(reason, "")
        self.assertEqual(drawn, KV_GRANULE_BYTES)  # one whole granule, not 0

    def test_sub_granule_ask_is_a_NAMED_refusal_when_it_cannot(self):
        post = Post(
            "kv-slack",
            available_bytes=100 * MIB,  # less than one 256 MiB granule
            granule_bytes=KV_GRANULE_BYTES,
        )
        drawn, reason = post.creditable(50 * MIB)
        self.assertEqual(drawn, 0)
        self.assertIn("granule", reason)

    def test_authority_reports_granularity_as_its_own_cause(self):
        auth = FundingAuthority()
        auth.declare_post(Post("kv-slack", 100 * MIB, granule_bytes=KV_GRANULE_BYTES))
        v = auth.can_fund(50 * MIB)
        self.assertFalse(v.ok)
        self.assertEqual(v.cause, CAUSE_GRANULARITY)
        self.assertFalse(v.retry_is_pointless)


class TestPhantomCapacity(unittest.TestCase):
    """Law 2: 'APPLIED shrink ... (107 MiB returned)' against claimed=0."""

    def test_a_post_that_underdelivered_is_derated(self):
        post = Post("kv-slack", 1000 * MIB, derate_num=0, derate_den=1)
        self.assertEqual(post.derated_bytes, 0)
        drawn, reason = post.creditable(100 * MIB)
        self.assertEqual(drawn, 0)
        self.assertIn("derated", reason)

    def test_phantom_is_its_own_cause_not_scarcity(self):
        auth = FundingAuthority()
        auth.declare_post(Post("kv-slack", 1000 * MIB, derate_num=0, derate_den=1))
        v = auth.can_fund(100 * MIB)
        self.assertEqual(v.cause, CAUSE_PHANTOM)

    def test_an_unobserved_post_is_trusted_once(self):
        post = Post("fresh", 500 * MIB, derate_den=0)
        self.assertEqual(post.derated_bytes, 500 * MIB)

    def test_a_post_cannot_claim_to_overdeliver(self):
        with self.assertRaises(FundingError):
            Post("impossible", 100 * MIB, derate_num=3, derate_den=2)


class TestUserReserveIsNeverFundable(unittest.TestCase):
    """Law 4 / DESIGN_584 R2, carried from #582."""

    def test_declaring_the_user_reserve_raises(self):
        auth = FundingAuthority()
        with self.assertRaises(FundingError) as ctx:
            auth.declare_post(Post("user-reserve", 1024 * MIB, is_user_reserve=True))
        self.assertIn("USER RESERVE", str(ctx.exception))

    def test_the_seam_staging_reserve_is_a_DIFFERENT_object_and_is_declarable(self):
        """The trap: two things called 'reserve'. 819 MiB here is the seam
        staging reserve from phase_flip_seam_reserve, not the user's 1024."""
        auth = FundingAuthority()
        auth.declare_post(Post("seam-staging-reserve", SPEC_STAGING_RESERVE_MIB * MIB))
        self.assertIn("seam-staging-reserve", auth.posts)

    def test_duplicate_declaration_raises(self):
        auth = FundingAuthority()
        auth.declare_post(Post("a", MIB))
        with self.assertRaises(FundingError):
            auth.declare_post(Post("a", MIB))


class TestWantGrowsPerRetry(unittest.TestCase):
    """Form (c): #808. want climbs 2501 -> 2517 -> 2533 -> 2579 across retries
    while free barely moves. The verdict's CAUSE must stay stable, because the
    backoff keys on it."""

    WANTS_MIB = (2501, 2517, 2533, 2579)

    def test_cause_is_stable_across_a_growing_want_when_slack_exists(self):
        causes = set()
        for want in self.WANTS_MIB:
            auth = FundingAuthority()
            mine = 84443 * ROW_BYTES
            auth.declare_post(
                Post(
                    "kv-slack",
                    mine,
                    granule_bytes=KV_GRANULE_BYTES,
                    group_scope=GROUP_UNIFORM_CAP,
                )
            )
            v = auth.can_fund(
                want * MIB,
                group_floor_bytes={"kv-slack": 0},
                local_slack_bytes={"kv-slack": mine},
            )
            causes.add(v.cause)
            self.assertFalse(v.retry_is_pointless)
        self.assertEqual(causes, {CAUSE_PEER_VETO})

    def test_shortfall_grows_with_want(self):
        prev = -1
        for want in self.WANTS_MIB:
            auth = FundingAuthority()
            auth.declare_post(Post("dry", 0, unavailable_reason="empty"))
            v = auth.can_fund(want * MIB)
            self.assertGreater(v.shortfall_bytes, prev)
            prev = v.shortfall_bytes


class TestUnsatisfiableFloor(unittest.TestCase):
    """Defect A from BUG_planner_corridor_capacity.md, recomputed from the
    SHIPPED constants rather than quoted."""

    def test_shipped_defaults_make_the_arming_floor_unreachable(self):
        from sglang.srt.managers import corridor_guard as cg
        from sglang.srt.managers import phase_flip_seam_reserve as sr

        d = diagnose_floor_band(
            cg.arming_floor_mib(),
            cg.corridor_band_ceiling_mib(),
            sr.DEFAULT_ARMING_MARGIN_MIB,
        )
        self.assertEqual(d.arming_floor_mib, 1331)
        self.assertEqual(d.band_ceiling_mib, 1229)
        self.assertFalse(d.satisfiable)
        self.assertEqual(d.overshoot_mib, 294)
        self.assertIn("UNSATISFIABLE", d.detail)

    def test_a_satisfiable_configuration_is_reported_as_such(self):
        d = diagnose_floor_band(900, 1229, 100)
        self.assertTrue(d.satisfiable)
        self.assertEqual(d.overshoot_mib, 0)

    def test_the_boundary_is_inclusive(self):
        self.assertTrue(diagnose_floor_band(1129, 1229, 100).satisfiable)
        self.assertFalse(diagnose_floor_band(1130, 1229, 100).satisfiable)


class TestSeamSnapshotBuilder(unittest.TestCase):
    """The builder the refusal path calls, fed the specimen's own figures."""

    def test_specimen_snapshot_funds_the_shortfall_and_names_kv_slack(self):
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=SPEC_ALLOCATOR_CACHE_MIB * MIB,
            kv_slack_rows=90080,
            row_bytes=ROW_BYTES,
            kv_granule_rows=KV_GRANULE_ROWS,
        )
        v = auth.can_fund(396 * MIB)
        self.assertTrue(v.ok, v.describe())
        self.assertIn("kv-slack", v.describe())
        # And the string it produces is the one that replaces "[nothing]".
        self.assertNotIn("[nothing]", v.describe())

    def test_an_empty_snapshot_still_names_all_three_posts(self):
        auth = authority_from_seam_snapshot()
        v = auth.can_fund(396 * MIB)
        self.assertFalse(v.ok)
        for name in ("allocator-cache", "draft-weights", "kv-slack"):
            self.assertIn(name, v.describe())
        self.assertIn("at or below its rung floor", v.describe())


class TestRungAccessorContract(unittest.TestCase):
    """CAN-FAIL PROOF for the refusal-path census.

    ``_funding_post_census`` reads the rung through a broad ``except`` so that
    a refusal can never crash. That safety has a cost: a WRONG attribute name
    is swallowed and the census goes permanently silent, which in a log is
    indistinguishable from a census that was simply never needed. This exact
    mistake was made while writing it -- ``current_rows`` / ``floor_rows`` /
    ``row_bytes`` were guessed and none of the three exist.

    These asserts pin the real names, so a rename upstream fails HERE loudly
    instead of silently disarming the census.
    """

    def test_rung_exposes_the_accessors_the_census_uses(self):
        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        self.assertTrue(callable(getattr(KvBackingRelief, "_min_release_rows", None)))
        import inspect

        src = inspect.getsource(KvBackingRelief)
        self.assertIn("_bytes_per_row", src)
        self.assertIn("_last_proposal_terms", src)

    def test_proposal_terms_carry_the_two_keys_the_census_reads(self):
        import inspect

        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        src = inspect.getsource(KvBackingRelief)
        # The census computes slack as current - floor_rows.
        self.assertIn('t["current"]', src)
        self.assertIn('t["floor_rows"]', src)

    def test_census_is_reachable_and_returns_a_named_string(self):
        """The census logic itself, exercised on a stand-in rung.

        Not the full runtime object -- that needs a scheduler -- but the exact
        read pattern the census performs, so a change to it fails here.
        """

        class _StandInRung:
            _last_proposal_terms = {"current": 212992, "floor_rows": 122912}
            _bytes_per_row = ROW_BYTES

            def _min_release_rows(self):
                return KV_GRANULE_ROWS

        rung = _StandInRung()
        terms = getattr(rung, "_last_proposal_terms", None)
        slack = max(0, int(terms["current"]) - int(terms["floor_rows"]))
        self.assertEqual(slack, 90080)  # the specimen's own figure
        auth = authority_from_seam_snapshot(
            kv_slack_rows=slack,
            row_bytes=int(rung._bytes_per_row),
            kv_granule_rows=int(rung._min_release_rows()),
        )
        self.assertTrue(auth.can_fund(396 * MIB).ok)


class TestVerdictHygiene(unittest.TestCase):
    def test_negative_availability_rejected(self):
        with self.assertRaises(FundingError):
            Post("bad", -1)

    def test_unknown_tier_rejected(self):
        with self.assertRaises(FundingError):
            Post("bad", MIB, tier=99)

    def test_unknown_scope_rejected(self):
        with self.assertRaises(FundingError):
            Post("bad", MIB, group_scope="whatever")

    def test_unnamed_post_rejected(self):
        with self.assertRaises(FundingError):
            Post("", MIB)

    def test_empty_authority_refuses_without_crashing(self):
        v = FundingAuthority().can_fund(100 * MIB)
        self.assertFalse(v.ok)
        self.assertIn("no posts declared", v.describe())

    def test_zero_want_is_trivially_funded(self):
        self.assertTrue(FundingAuthority().can_fund(0).ok)

    def test_group_scope_none_is_not_reduced(self):
        """Law 5: a rank-local draw changes no branch input, so a group floor
        for some OTHER post must not touch it."""
        auth = FundingAuthority()
        auth.declare_post(Post("local-thing", 500 * MIB, group_scope=GROUP_NONE))
        v = auth.can_fund(400 * MIB, group_floor_bytes={"local-thing": 0})
        self.assertTrue(v.ok, v.describe())


if __name__ == "__main__":
    unittest.main()


class TestSolveArmingFloor(unittest.TestCase):
    """B: the arming floor is the free variable; the corridor band is not."""

    def test_shipped_defaults_are_unsatisfiable_and_name_the_max_reserve(self):
        from sglang.srt.managers import corridor_guard as cg
        from sglang.srt.managers import phase_flip_seam_reserve as sr

        sol = solve_arming_floor(
            cg.corridor_band_floor_mib(),
            cg.corridor_band_ceiling_mib(),
            cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB,
            sr.DEFAULT_ARMING_MARGIN_MIB,
        )
        self.assertFalse(sol.satisfiable)
        self.assertEqual(sol.arming_floor_mib, 1331)
        # 1229 - 819 - 192 = 218, against the 512 shipped.
        self.assertEqual(sol.max_seam_entry_reserve_mib, 218)
        self.assertIn("UNSATISFIABLE", sol.detail)
        self.assertIn("CANNOT help", sol.detail)

    def test_a_reserve_within_the_headroom_is_satisfiable(self):
        sol = solve_arming_floor(819, 1229, 218, 192)
        self.assertTrue(sol.satisfiable)
        self.assertEqual(sol.arming_floor_mib, 1037)

    def test_one_mib_over_the_headroom_refuses(self):
        self.assertFalse(solve_arming_floor(819, 1229, 219, 192).satisfiable)

    def test_the_band_is_never_moved_to_make_it_fit(self):
        """The solution may only change the reserve; the band is a user law."""
        sol = solve_arming_floor(819, 1229, 512, 192)
        self.assertEqual(sol.max_seam_entry_reserve_mib, 1229 - 819 - 192)


class TestBreakEvenProvenance(unittest.TestCase):
    """C/#819: 7004 is not a literal; the staleness is one level down."""

    def test_all_frozen_is_reported_and_named(self):
        p = BreakEvenProvenance(
            3.2, PROV_FROZEN, 1681.0, PROV_FROZEN, 7245.5, PROV_FROZEN
        )
        self.assertFalse(p.is_fully_solved)
        self.assertEqual(p.frozen_inputs, ("flip_cost_s", "tp_tok_s", "pp_tok_s"))
        self.assertIn("STILL FROZEN", p.describe())

    def test_the_real_shipped_shape_one_measured_two_frozen(self):
        """Flip cost self-corrects via FlipCostEstimator.observe(); the two
        prefill rates have no runtime measurement path at all."""
        p = BreakEvenProvenance(
            3.2, PROV_MEASURED, 1681.0, PROV_FROZEN, 7245.5, PROV_FROZEN
        )
        self.assertEqual(p.frozen_inputs, ("tp_tok_s", "pp_tok_s"))
        self.assertFalse(p.is_fully_solved)

    def test_fully_solved_when_the_operator_supplied_both_rates(self):
        p = BreakEvenProvenance(3.0, PROV_MEASURED, 2000.0, PROV_ENV, 8000.0, PROV_ENV)
        self.assertTrue(p.is_fully_solved)
        self.assertIn("every input is rig-local", p.describe())

    def test_7004_is_reproduced_by_the_shipped_inputs(self):
        """The number the ticket calls a seed, derived from its three inputs."""
        from sglang.srt.managers.phase_policy import break_even_tokens

        self.assertEqual(break_even_tokens(3.2, 1681.0, 7245.5), 7004)
