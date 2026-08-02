"""#394 slice 3: the link-proportional expert COMPUTE placement solve.

Slice 2 moved cold-expert byte ownership and left per-rank H2D unchanged by
construction. This solve moves the assignment itself -- which rank executes
which expert -- by solving the "moe" family vector so that the STREAMED expert
mass follows the links while the GPU-RESIDENT mass stays exactly where the base
plan put it.

What is pinned here, and why each pin exists:

* **Defaults are byte-identical.** Uniform links, or an offload-free launch,
  must produce the plan the boot started from -- and where there is genuinely
  no lever the launch is REFUSED by name instead of silently serving the
  baseline under the treatment's name.
* **Proportionality on a synthetic foreign rig** (#434 suite pattern): the
  predicted cold-mass shares equal the link shares, the resident shares are
  untouched, and the per-rank transfer times equalise. No number in this file
  is read from the development rig's hardware; the reference-rig figures are
  used only as ONE synthetic profile among several, exactly as
  ``test_planner_generality_434`` does.
* **Rank-uniformity** via the #431 ``barlink_uniformity`` recorder: every rank
  must derive the SAME vector, because the ranges have to stay disjoint and
  cover the expert set. The can-fail arm perturbs one rank's inputs and asserts
  the recorder's comparator catches it.
* **Range integrity** across the hazard axes the ledger names: uneven expert
  ranges under the #82 expert-dim shard must stay >= 1 expert per rank, sum to
  the expert count exactly, and be disjoint and gapless (#109 MMQ-OOB family,
  #112 ``moe.cuh`` nrows binding, #80 combine correctness).

Hermetic: no GPU, no NVML, no model, no shared memory. Every hardware fact is
injected.
"""

import unittest

from sglang.srt.distributed.device_communicators import barlink_uniformity as uniformity
from sglang.srt.distributed.utils import partition_units
from sglang.srt.layers.moe.expert_compute_placement import (
    COMPUTE_PLACEMENT_LINK,
    COMPUTE_POLICY_ENV,
    ComputePlacement,
    NoComputeLever,
    cold_traffic_coefficients_from_measurement,
    solve_link_proportional_expert_vector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Synthetic rig profiles. Named after their SHAPE, not after any box: the point
# of #434 is that a solve reads the profile it is given.
# --------------------------------------------------------------------------

#: One fat slot, one starved slot, one fat slot; the fat cards also hold more
#: weight. Numerically this is the 2026-08-02 battery's measured profile, used
#: here as a synthetic input among others -- nothing branches on it.
PROFILE_ASYMMETRIC_SLOTS = dict(
    base_plan=[400, 256, 344],
    resident_fractions=[0.485, 0.42, 0.42],
    link_weights=[14.42, 6.45, 13.41],
)

#: Four ranks, two pairs of identical cards on differently wired slots.
PROFILE_TWO_PAIRS = dict(
    base_plan=[3, 3, 2, 2],
    resident_fractions=[0.5, 0.5, 0.5, 0.5],
    link_weights=[16.0, 16.0, 4.0, 4.0],
)

#: Homogeneous: same cards, same slots, same fractions.
PROFILE_HOMOGENEOUS = dict(
    base_plan=[1, 1, 1],
    resident_fractions=[0.6, 0.6, 0.6],
    link_weights=[8.0, 8.0, 8.0],
)


def _shares(values):
    total = float(sum(values))
    return [v / total for v in values]


class TestTheDefaultPathIsUnchanged(CustomTestCase):
    def test_uniform_links_solve_to_the_base_plan(self):
        """A homogeneous group has no lever, and the solve says so by identity.

        This is the byte-identity gate: on a rig where every rank sits behind
        the same link, the vector this returns must be the vector the boot
        would have used anyway, so enabling the flag there changes nothing a
        layer can observe.
        """
        placement = solve_link_proportional_expert_vector(**PROFILE_HOMOGENEOUS)
        self.assertTrue(placement.is_identity)
        self.assertEqual(len(set(placement.weights)), 1)

    def test_a_uniform_base_plan_with_uniform_links_stays_uniform(self):
        placement = solve_link_proportional_expert_vector(
            base_plan=[1, 1, 1, 1],
            resident_fractions=[0.25, 0.25, 0.25, 0.25],
            link_weights=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(len(set(placement.weights)), 1)
        self.assertTrue(placement.is_identity)

    def test_a_plan_that_already_matches_its_links_is_the_identity(self):
        """The real identity condition is ``l == b``, not "uniform links".

        A plan whose shares already equal its link shares is at the optimum,
        and the solve must return it unchanged -- otherwise every re-solve of a
        converged configuration would perturb it.
        """
        placement = solve_link_proportional_expert_vector(
            base_plan=[10, 6, 6],
            resident_fractions=[0.4, 0.4, 0.4],
            link_weights=[10.0, 6.0, 6.0],
        )
        self.assertTrue(placement.is_identity)
        for got, want in zip(_shares(placement.weights), _shares([10, 6, 6])):
            self.assertAlmostEqual(got, want, places=3)

    def test_uniform_links_over_an_uneven_plan_equalise_the_cold_mass(self):
        """NOT an identity, and deliberately so.

        Under uniform links the transfer optimum is an EQUAL cold mass per
        rank; the base plan is capacity-proportional and therefore is not it.
        Byte-identical defaults come from the flag being opt-in, not from the
        solve declining to move anything.
        """
        placement = solve_link_proportional_expert_vector(
            base_plan=[10, 6, 6],
            resident_fractions=[0.4, 0.4, 0.4],
            link_weights=[9.0, 9.0, 9.0],
        )
        self.assertFalse(placement.is_identity)
        cold = placement.cold_shares
        self.assertAlmostEqual(cold[0], cold[1], places=12)
        self.assertAlmostEqual(cold[1], cold[2], places=12)

    def test_offload_off_is_refused_by_name_not_resolved_to_a_no_op(self):
        """f == 1 everywhere: nothing streams, so no link can matter.

        Refusing is the point. Resolving this to the base plan would let an
        operator run the treatment arm of an A/B, get the baseline, and see
        nothing in the log that says so.
        """
        with self.assertRaises(NoComputeLever) as caught:
            solve_link_proportional_expert_vector(
                base_plan=[2, 1, 1],
                resident_fractions=[1.0, 1.0, 1.0],
                link_weights=[14.0, 6.0, 13.0],
            )
        message = str(caught.exception)
        self.assertIn("nothing to move", message)
        self.assertIn("--rank-moe-resident-fraction", message)


class TestTheSolveFollowsTheProfile(CustomTestCase):
    def test_cold_mass_lands_exactly_on_the_link_shares(self):
        """The objective, stated as an equality rather than a direction."""
        placement = solve_link_proportional_expert_vector(**PROFILE_ASYMMETRIC_SLOTS)
        want = _shares(PROFILE_ASYMMETRIC_SLOTS["link_weights"])
        got = _shares(placement.cold_shares)
        for r, (g, w) in enumerate(zip(got, want)):
            self.assertAlmostEqual(g, w, places=9, msg=f"rank {r}")

    def test_every_rank_ends_on_the_same_transfer_time(self):
        """The slowest-rank rule: at the optimum there is no slowest rank."""
        for name, profile in (
            ("asymmetric", PROFILE_ASYMMETRIC_SLOTS),
            ("two-pairs", PROFILE_TWO_PAIRS),
        ):
            with self.subTest(profile=name):
                placement = solve_link_proportional_expert_vector(**profile)
                shares = placement.predicted_transfer_shares()
                world = len(shares)
                for r, share in enumerate(shares):
                    self.assertAlmostEqual(
                        share, 1.0 / world, places=9, msg=f"{name} rank {r}"
                    )

    def test_the_resident_mass_does_not_move(self):
        """VRAM-neutrality, which is what makes this installable without a re-fit.

        Each rank's resident share must equal ``f_r * b_r`` of the plan it
        started from. If it drifted, the card whose VRAM ledger was solved for
        the old plan would be short by the difference.
        """
        profile = PROFILE_ASYMMETRIC_SLOTS
        placement = solve_link_proportional_expert_vector(**profile)
        base = _shares(profile["base_plan"])
        for r, fraction in enumerate(profile["resident_fractions"]):
            self.assertAlmostEqual(
                placement.resident_shares[r], fraction * base[r], places=12
            )

    def test_the_starved_slot_loses_experts_and_the_fat_slots_gain(self):
        """Direction check on the one profile the ledger has a measurement for.

        rank 1 sits on the narrow slot and was the clock at 1.40x the group
        mean; it must end up with strictly fewer experts than the base plan
        gave it, and the two wide-slot ranks with more.
        """
        profile = PROFILE_ASYMMETRIC_SLOTS
        placement = solve_link_proportional_expert_vector(**profile)
        base = _shares(profile["base_plan"])
        got = _shares(placement.weights)
        self.assertLess(got[1], base[1])
        self.assertGreater(got[0], base[0])
        self.assertGreater(got[2], base[2])

    def test_full_offload_degenerates_to_pure_link_proportional(self):
        """ANALYSE_393 Path A' in its pure form is the f -> 0 limit."""
        placement = solve_link_proportional_expert_vector(
            base_plan=[400, 256, 344],
            resident_fractions=[0.001, 0.001, 0.001],
            link_weights=[14.42, 6.45, 13.41],
        )
        want = _shares([14.42, 6.45, 13.41])
        for r, (g, w) in enumerate(zip(_shares(placement.weights), want)):
            self.assertAlmostEqual(g, w, places=3, msg=f"rank {r}")


class TestTheMeasuredCalibration(CustomTestCase):
    """The residual the first-order model leaves, turned into a number.

    ``b * (1 - f)`` predicts what share of the group's H2D each rank moves. On
    a real boot it does not match, because the ranks re-fetch at different
    rates. The coefficient carries that difference; nothing else in the solve
    is allowed to.
    """

    def test_a_model_perfect_boot_calibrates_to_all_ones(self):
        """The falsifiable direction: no residual means no correction."""
        base = [400, 256, 344]
        fractions = [0.485, 0.42, 0.42]
        modelled = [b * (1 - f) for b, f in zip(base, fractions)]
        coefficients = cold_traffic_coefficients_from_measurement(
            base, fractions, modelled
        )
        for r, c in enumerate(coefficients):
            self.assertAlmostEqual(c, 1.0, places=9, msg=f"rank {r}")

    def test_a_calibrated_solve_with_unit_coefficients_is_the_plain_solve(self):
        plain = solve_link_proportional_expert_vector(**PROFILE_ASYMMETRIC_SLOTS)
        calibrated = solve_link_proportional_expert_vector(
            traffic_coefficients=[1.0, 1.0, 1.0], **PROFILE_ASYMMETRIC_SLOTS
        )
        self.assertEqual(plain.weights, calibrated.weights)
        self.assertFalse(calibrated.is_calibrated)

    def test_a_rank_that_streams_more_per_expert_is_given_fewer(self):
        profile = dict(PROFILE_ASYMMETRIC_SLOTS)
        plain = solve_link_proportional_expert_vector(**profile)
        # rank 0 turns out to re-fetch twice as much per owned cold expert.
        calibrated = solve_link_proportional_expert_vector(
            traffic_coefficients=[2.0, 1.0, 1.0], **profile
        )
        self.assertTrue(calibrated.is_calibrated)
        self.assertLess(
            _shares(calibrated.cold_shares)[0], _shares(plain.cold_shares)[0]
        )

    def test_the_calibrated_solve_equalises_the_MEASURED_transfer_time(self):
        """The whole point: the optimum moves to where the measurement is."""
        profile = dict(PROFILE_ASYMMETRIC_SLOTS)
        coefficients = [1.25, 1.05, 0.70]
        placement = solve_link_proportional_expert_vector(
            traffic_coefficients=coefficients, **profile
        )
        shares = placement.predicted_transfer_shares()
        for r, share in enumerate(shares):
            self.assertAlmostEqual(share, 1.0 / 3.0, places=9, msg=f"rank {r}")

    def test_calibration_refuses_a_rank_that_streamed_nothing(self):
        with self.assertRaises(ValueError) as caught:
            cold_traffic_coefficients_from_measurement(
                [1, 1, 1], [1.0, 0.5, 0.5], [10.0, 20.0, 20.0]
            )
        self.assertIn("says nothing about", str(caught.exception))

    def test_a_non_positive_coefficient_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            solve_link_proportional_expert_vector(
                traffic_coefficients=[1.0, 0.0, 1.0], **PROFILE_ASYMMETRIC_SLOTS
            )
        self.assertIn("finite positive", str(caught.exception))

    def test_the_uncalibrated_description_says_it_is_uncalibrated(self):
        plain = solve_link_proportional_expert_vector(**PROFILE_ASYMMETRIC_SLOTS)
        self.assertIn("UNCALIBRATED", plain.describe())
        calibrated = solve_link_proportional_expert_vector(
            traffic_coefficients=[1.3, 1.0, 0.7], **PROFILE_ASYMMETRIC_SLOTS
        )
        self.assertNotIn("UNCALIBRATED", calibrated.describe())
        self.assertIn("traffic coefficients", calibrated.describe())

    def test_the_coefficient_env_is_length_checked_and_registered(self):
        import os

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.expert_compute_placement import (
            TRAFFIC_COEFFICIENT_ENV,
            _traffic_coefficients_from_env,
        )

        self.assertIn(TRAFFIC_COEFFICIENT_ENV, dir(envs))
        previous = os.environ.get(TRAFFIC_COEFFICIENT_ENV)
        try:
            os.environ.pop(TRAFFIC_COEFFICIENT_ENV, None)
            self.assertIsNone(_traffic_coefficients_from_env(3))
            os.environ[TRAFFIC_COEFFICIENT_ENV] = "1.25,1.05,0.7"
            self.assertEqual(_traffic_coefficients_from_env(3), (1.25, 1.05, 0.7))
            with self.assertRaises(ValueError):
                _traffic_coefficients_from_env(2)
            os.environ[TRAFFIC_COEFFICIENT_ENV] = "1.25,nonsense,0.7"
            with self.assertRaises(ValueError):
                _traffic_coefficients_from_env(3)
        finally:
            os.environ.pop(TRAFFIC_COEFFICIENT_ENV, None)
            if previous is not None:
                os.environ[TRAFFIC_COEFFICIENT_ENV] = previous


class TestInvarianceOnForeignRigs(CustomTestCase):
    """#434 pattern: the answer follows the profile, not the labels or units."""

    def test_scaling_every_link_by_a_constant_changes_nothing(self):
        base = solve_link_proportional_expert_vector(**PROFILE_ASYMMETRIC_SLOTS)
        scaled = solve_link_proportional_expert_vector(
            base_plan=PROFILE_ASYMMETRIC_SLOTS["base_plan"],
            resident_fractions=PROFILE_ASYMMETRIC_SLOTS["resident_fractions"],
            link_weights=[7.0 * w for w in PROFILE_ASYMMETRIC_SLOTS["link_weights"]],
        )
        self.assertEqual(base.weights, scaled.weights)

    def test_scaling_the_base_plan_by_a_constant_changes_nothing(self):
        base = solve_link_proportional_expert_vector(**PROFILE_ASYMMETRIC_SLOTS)
        scaled = solve_link_proportional_expert_vector(
            base_plan=[13 * b for b in PROFILE_ASYMMETRIC_SLOTS["base_plan"]],
            resident_fractions=PROFILE_ASYMMETRIC_SLOTS["resident_fractions"],
            link_weights=PROFILE_ASYMMETRIC_SLOTS["link_weights"],
        )
        self.assertEqual(base.weights, scaled.weights)

    def test_relabeling_the_ranks_permutes_the_answer(self):
        profile = PROFILE_ASYMMETRIC_SLOTS
        straight = solve_link_proportional_expert_vector(**profile)
        order = [2, 0, 1]
        permuted = solve_link_proportional_expert_vector(
            base_plan=[profile["base_plan"][i] for i in order],
            resident_fractions=[profile["resident_fractions"][i] for i in order],
            link_weights=[profile["link_weights"][i] for i in order],
        )
        self.assertEqual(permuted.weights, tuple(straight.weights[i] for i in order))

    def test_a_two_rank_group_is_solved_like_any_other(self):
        placement = solve_link_proportional_expert_vector(
            base_plan=[1, 1],
            resident_fractions=[0.5, 0.5],
            link_weights=[3.0, 1.0],
        )
        self.assertGreater(placement.weights[0], placement.weights[1])
        for share, want in zip(placement.predicted_transfer_shares(), (0.5, 0.5)):
            self.assertAlmostEqual(share, want, places=9)


class TestRangeIntegrityUnderTheExpertShard(CustomTestCase):
    """The #82 expert-dim shard's invariants, over a sweep of solved vectors.

    The layer turns this vector into ``[lo, hi)`` per rank through
    ``partition_units`` (``fused_moe_triton/layer.py``). A vector that produced
    an empty range, a gap or an overlap would be an out-of-bounds expert index
    in ``moe.cuh`` (#109/#112) or a dropped contribution in the all-reduce
    (#80), and neither fails loudly at the seam.
    """

    def test_solved_vectors_partition_the_expert_set_exactly(self):
        profiles = [
            PROFILE_ASYMMETRIC_SLOTS,
            PROFILE_TWO_PAIRS,
            PROFILE_HOMOGENEOUS,
            dict(
                base_plan=[5, 1, 1],
                resident_fractions=[0.9, 0.1, 0.1],
                link_weights=[1.0, 20.0, 20.0],
            ),
        ]
        for profile in profiles:
            placement = solve_link_proportional_expert_vector(**profile)
            world = len(placement.weights)
            for num_experts in (world, 8, 32, 64, 128, 256, 257):
                if num_experts < world:
                    continue
                with self.subTest(world=world, experts=num_experts):
                    sizes = partition_units(num_experts, list(placement.weights))
                    self.assertEqual(len(sizes), world)
                    self.assertEqual(sum(sizes), num_experts)
                    self.assertTrue(
                        all(s >= 1 for s in sizes),
                        f"empty expert range in {sizes} for {placement.weights}",
                    )
                    # Disjoint and gapless: the prefix sums must tile [0, E).
                    cursor = 0
                    for size in sizes:
                        cursor += size
                    self.assertEqual(cursor, num_experts)

    def test_no_rank_is_ever_solved_to_zero_weight(self):
        """Even a rank on a link 1000x narrower than its peers keeps a range."""
        placement = solve_link_proportional_expert_vector(
            base_plan=[1, 1, 1],
            resident_fractions=[0.01, 0.01, 0.01],
            link_weights=[1000.0, 1000.0, 1.0],
        )
        self.assertTrue(all(w >= 1 for w in placement.weights))
        sizes = partition_units(64, list(placement.weights))
        self.assertTrue(all(s >= 1 for s in sizes), sizes)


class TestRankUniformity(CustomTestCase):
    """The collective-family pin, through the #431 decision recorder.

    The expert ranges are derived independently on every rank from the vector
    each rank holds. If two ranks held different vectors the ranges would
    overlap or leave a hole, and the all-reduce would sum a wrong set -- a
    silent wrong answer, not a hang, which is why it is pinned rather than
    trusted. The recorder is the standing instrument for this family
    (#94/#194/#312/#431); using it here keeps one comparator for all of them.
    """

    def setUp(self):
        super().setUp()
        self._previous = uniformity.set_recording_for_test(True)
        uniformity.clear_all()

    def tearDown(self):
        uniformity.clear_all()
        uniformity.set_recording_for_test(self._previous)
        super().tearDown()

    @staticmethod
    def _record_vector_for(rank: int, **profile) -> None:
        """One rank solves and records the vector it would install."""
        placement = solve_link_proportional_expert_vector(**profile)
        uniformity.recorder_for(f"rank{rank}").record(
            op="moe_expert_range",
            nbytes=sum(placement.weights),
            path=",".join(str(w) for w in placement.weights),
            rounds=len(placement.weights),
        )

    def _sequences(self, world: int):
        return {
            rank: uniformity.recorder_for(f"rank{rank}").snapshot()
            for rank in range(world)
        }

    def test_every_rank_derives_the_same_vector_from_the_same_profile(self):
        for rank in range(3):
            self._record_vector_for(rank, **PROFILE_ASYMMETRIC_SLOTS)
        self.assertIsNone(uniformity.first_divergence(self._sequences(3)))
        uniformity.assert_sequences_uniform(self._sequences(3), context="#394 slice 3")

    def test_can_fail_a_rank_that_reads_a_different_link_table_is_caught(self):
        """The falsifier: without it the pin above proves only that it ran."""
        for rank in range(3):
            profile = dict(PROFILE_ASYMMETRIC_SLOTS)
            if rank == 1:
                # One rank sees its own card's nameplate where the others saw
                # the measured probe -- the exact shape of a partial provenance
                # resolution.
                profile["link_weights"] = [14.42, 15.75, 13.41]
            self._record_vector_for(rank, **profile)
        detail = uniformity.first_divergence(self._sequences(3))
        self.assertIsNotNone(detail)
        self.assertIn("moe_expert_range", detail)
        with self.assertRaises(uniformity.CollectiveSequenceDivergence):
            uniformity.assert_sequences_uniform(self._sequences(3))

    def test_the_solve_is_a_pure_function_of_its_three_vectors(self):
        """No environment read, no clock, no cache: same inputs, same answer."""
        first = solve_link_proportional_expert_vector(**PROFILE_TWO_PAIRS)
        second = solve_link_proportional_expert_vector(**PROFILE_TWO_PAIRS)
        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.shares, second.shares)


class TestInputValidation(CustomTestCase):
    def test_vector_lengths_must_agree(self):
        with self.assertRaises(ValueError) as caught:
            solve_link_proportional_expert_vector(
                base_plan=[1, 1, 1],
                resident_fractions=[0.5, 0.5],
                link_weights=[1.0, 2.0, 3.0],
            )
        self.assertIn("same TP group", str(caught.exception))

    def test_a_resident_fraction_outside_the_unit_interval_is_refused(self):
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(fraction=bad):
                with self.assertRaises(ValueError):
                    solve_link_proportional_expert_vector(
                        base_plan=[1, 1],
                        resident_fractions=[bad, 0.5],
                        link_weights=[1.0, 2.0],
                    )

    def test_a_non_positive_link_weight_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            solve_link_proportional_expert_vector(
                base_plan=[1, 1],
                resident_fractions=[0.5, 0.5],
                link_weights=[0.0, 2.0],
            )
        self.assertIn("finite positive", str(caught.exception))


class TestTheFlagContract(CustomTestCase):
    """The launch surface: one symbol, one resolution point, no silent path."""

    def test_server_args_and_the_solver_spell_the_symbol_the_same_way(self):
        from sglang.srt import server_args as sa

        self.assertEqual(sa._COMPUTE_PLACEMENT_LINK, COMPUTE_PLACEMENT_LINK)

    def test_the_flag_parser_accepts_the_symbol_and_a_vector(self):
        from sglang.srt.server_args import _parse_rank_moe_ratio

        self.assertEqual(_parse_rank_moe_ratio("link"), COMPUTE_PLACEMENT_LINK)
        self.assertEqual(_parse_rank_moe_ratio(" link "), COMPUTE_PLACEMENT_LINK)
        self.assertEqual(_parse_rank_moe_ratio("4,2,3"), [4, 2, 3])

    def test_the_launcher_resolves_the_symbol_before_spawning_workers(self):
        """Call-site pin: consumer counting would miss a removed call.

        The resolution must happen in the launcher, after the rank -> card
        vector is published (the link weights are keyed by those UUIDs) and
        before the spawn loop (the workers get a pickled ServerArgs).
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[5]
        source = (root / "python/sglang/srt/entrypoints/engine.py").read_text()
        tree = ast.parse(source)
        called = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "resolve_moe_compute_placement_flag"
        ]
        self.assertEqual(
            len(called),
            1,
            "engine.py must call resolve_moe_compute_placement_flag exactly once",
        )
        published = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "publish_rank_card_uuids"
        ]
        self.assertTrue(published)
        self.assertGreater(called[0], published[0])

    def test_an_unresolved_symbol_that_reaches_a_worker_is_a_hard_error(self):
        """Falsifier for the silent-baseline failure mode.

        The worker used to accept anything that was not a list by ignoring it,
        which for a symbolic value means "serve the base plan and say nothing".
        Exercised rather than grepped: the point is the behaviour.
        """
        from sglang.srt.managers.scheduler import uneven_family_plans

        class _Args:
            rank_mlp_ratio = None
            rank_moe_ratio = COMPUTE_PLACEMENT_LINK
            rank_vocab_ratio = None

        with self.assertRaises(ValueError) as caught:
            uneven_family_plans(_Args())
        message = str(caught.exception)
        self.assertIn("rank_moe_ratio", message)
        self.assertIn("resolve_moe_compute_placement_flag", message)

    def test_a_resolved_vector_installs_normally(self):
        """The other half of the guard: lists still reach the plan installer."""
        from sglang.srt.managers.scheduler import uneven_family_plans

        class _Args:
            rank_mlp_ratio = None
            rank_moe_ratio = [123, 61, 104]
            rank_vocab_ratio = None

        self.assertEqual(uneven_family_plans(_Args()), {"moe": [123, 61, 104]})

    def test_the_environment_channel_refuses_the_symbol_by_name(self):
        import os

        from sglang.srt.server_args import ServerArgs

        previous = os.environ.get("SGLANG_UNEVEN_MOE_VECTOR")
        os.environ["SGLANG_UNEVEN_MOE_VECTOR"] = "link"
        try:
            args = ServerArgs.__new__(ServerArgs)
            args.rank_tp_ratio = [1, 1]
            args.tp_size = 2
            args.rank_mlp_ratio = None
            args.rank_moe_ratio = None
            args.rank_vocab_ratio = None
            args.enable_dp_lm_head = False
            args.rank_gpu_id = None
            with self.assertRaises(ValueError) as caught:
                args._handle_uneven_mlp_ratio()
            self.assertIn("explicit vector only", str(caught.exception))
        finally:
            if previous is None:
                os.environ.pop("SGLANG_UNEVEN_MOE_VECTOR", None)
            else:
                os.environ["SGLANG_UNEVEN_MOE_VECTOR"] = previous

    def test_the_symbol_passes_family_validation_without_a_length_check(self):
        """A symbol is not a vector; validating its length would reject 'link'.

        The base-plan precondition still applies, and is asserted separately
        below -- this only pins that the generic vector checks skip the symbol.
        """
        args_ok = self._args_with(rank_tp_ratio=[1, 1, 1], tp_size=3)
        args_ok._handle_uneven_mlp_ratio()
        self.assertEqual(args_ok.rank_moe_ratio, COMPUTE_PLACEMENT_LINK)

    def test_the_symbol_still_requires_an_active_base_plan(self):
        args = self._args_with(rank_tp_ratio=None, tp_size=3)
        with self.assertRaises(ValueError) as caught:
            args._handle_uneven_mlp_ratio()
        self.assertIn("--rank-tp-ratio", str(caught.exception))

    @staticmethod
    def _args_with(rank_tp_ratio, tp_size):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.__new__(ServerArgs)
        args.rank_tp_ratio = rank_tp_ratio
        args.tp_size = tp_size
        args.rank_mlp_ratio = None
        args.rank_moe_ratio = COMPUTE_PLACEMENT_LINK
        args.rank_vocab_ratio = None
        args.enable_dp_lm_head = False
        args.rank_gpu_id = None
        return args


class TestTheResolverRunsEndToEnd(CustomTestCase):
    """An execution smoke of the launcher path, not just of the pure solve.

    Desk-written code that was never executed is unvalidated, and the pure
    solve above would pass every test in this file while the resolver around it
    raised on its first line. Every hardware fact is injected through the
    documented environment channels -- the published rank -> card vector and an
    explicit link ratio -- so this runs with no driver.
    """

    _ENVS = (
        "SGLANG_RANK_CARD_UUIDS",
        "SGLANG_MOE_HOST_SHARD_RATIO",
        "SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
        "SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS",
        COMPUTE_POLICY_ENV,
    )

    def setUp(self):
        super().setUp()
        import os

        self._saved = {k: os.environ.get(k) for k in self._ENVS}
        os.environ["SGLANG_RANK_CARD_UUIDS"] = "GPU-aaa,GPU-bbb,GPU-ccc"
        os.environ["SGLANG_MOE_HOST_SHARD_RATIO"] = "14.42,6.45,13.41"
        os.environ["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"] = "0.485,0.42,0.42"
        os.environ.pop("SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS", None)
        os.environ.pop(COMPUTE_POLICY_ENV, None)

    def tearDown(self):
        import os

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().tearDown()

    @staticmethod
    def _args(**overrides):
        class _Args:
            tp_size = 3
            ep_size = 1
            nnodes = 1
            rank_tp_ratio = [400, 256, 344]
            rank_moe_ratio = COMPUTE_PLACEMENT_LINK
            rank_moe_resident_fraction = None

            def override(self, source, **fields):
                self.override_source = source
                for key, value in fields.items():
                    setattr(self, key, value)

        args = _Args()
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_the_resolver_installs_an_explicit_vector_and_names_its_policy(self):
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        args = self._args()
        placement = resolve_moe_compute_placement_flag(args)
        self.assertIsNotNone(placement)
        self.assertEqual(args.rank_moe_ratio, list(placement.weights))
        self.assertEqual(len(args.rank_moe_ratio), 3)
        self.assertEqual(os.environ[COMPUTE_POLICY_ENV], "link-proportional")
        # The override entry point, not a bare assignment (mutation ratchet).
        self.assertIn("#394", args.override_source)
        # And the rank the narrow link starves loses weight.
        self.assertLess(_shares(args.rank_moe_ratio)[1], _shares([400, 256, 344])[1])

    def test_the_resolver_is_a_no_op_without_the_symbol(self):
        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        args = self._args(rank_moe_ratio=None)
        self.assertIsNone(resolve_moe_compute_placement_flag(args))
        self.assertIsNone(args.rank_moe_ratio)

        args = self._args(rank_moe_ratio=[4, 2, 3])
        self.assertIsNone(resolve_moe_compute_placement_flag(args))
        self.assertEqual(args.rank_moe_ratio, [4, 2, 3])

    def test_the_calibrated_run_labels_itself_differently(self):
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        os.environ["SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS"] = "1.1251,1.0726,0.8023"
        plain_args = self._args()
        os.environ.pop("SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS")
        plain = resolve_moe_compute_placement_flag(plain_args)

        os.environ["SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS"] = "1.1251,1.0726,0.8023"
        calibrated = resolve_moe_compute_placement_flag(self._args())
        self.assertNotEqual(plain.weights, calibrated.weights)
        self.assertEqual(os.environ[COMPUTE_POLICY_ENV], "link-proportional-calibrated")

    def test_expert_parallelism_is_refused_by_name(self):
        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        with self.assertRaises(NoComputeLever) as caught:
            resolve_moe_compute_placement_flag(self._args(ep_size=2))
        self.assertIn("ep_size=2", str(caught.exception))

    def test_a_missing_base_plan_is_refused_by_name(self):
        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        with self.assertRaises(NoComputeLever) as caught:
            resolve_moe_compute_placement_flag(self._args(rank_tp_ratio="auto"))
        self.assertIn("--rank-tp-ratio", str(caught.exception))

    def test_offload_off_is_refused_through_the_resolver_too(self):
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        os.environ["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"] = "1.0"
        with self.assertRaises(NoComputeLever) as caught:
            resolve_moe_compute_placement_flag(self._args())
        self.assertIn("nothing to move", str(caught.exception))

    def test_an_absent_card_vector_is_refused_not_guessed(self):
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        os.environ.pop("SGLANG_RANK_CARD_UUIDS")
        with self.assertRaises(NoComputeLever) as caught:
            resolve_moe_compute_placement_flag(self._args())
        self.assertIn("physical card", str(caught.exception))

    def test_an_explicitly_equal_ratio_env_warns_loudly(self):
        """The one configuration trap this flag has.

        ``SGLANG_MOE_HOST_SHARD_RATIO`` is both the strongest source of the
        link weights AND the way an arm holds cold-tier byte ownership at the
        baseline. Pinning it equal for the second reason hands this solve a
        uniform link profile, and the result is not the base plan either -- it
        is the cold-mass-equalising vector, a third thing nobody asked for.
        Not a refusal (a genuinely uniform rig is legal), but it has to be
        visible in the boot log.
        """
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        os.environ["SGLANG_MOE_HOST_SHARD_RATIO"] = "1,1,1"
        with self.assertLogs(
            "sglang.srt.layers.moe.expert_compute_placement", level="WARNING"
        ) as logs:
            placement = resolve_moe_compute_placement_flag(self._args())
        text = "\n".join(logs.output)
        self.assertIn("SGLANG_MOE_HOST_SHARD_RATIO", text)
        self.assertIn("SGLANG_MOE_COLD_TIER_SHM", text)
        # And the vector it produced really is the third thing: equal cold mass
        # rather than either the base plan or a link-weighted split.
        cold = placement.cold_shares
        self.assertAlmostEqual(cold[0], cold[1], places=12)
        self.assertAlmostEqual(cold[1], cold[2], places=12)
        self.assertFalse(placement.is_identity)

    def test_a_measured_ratio_does_not_trip_the_equal_ratio_warning(self):
        """Can-fail control: the warning must not fire on a real profile."""
        import logging

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        logger = logging.getLogger("sglang.srt.layers.moe.expert_compute_placement")
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            resolve_moe_compute_placement_flag(self._args())
        finally:
            logger.removeHandler(handler)
        self.assertEqual(
            [r.getMessage() for r in records if r.levelno >= logging.WARNING], []
        )

    def test_the_arm_script_does_not_pin_the_ratio_env_on_a_compute_arm(self):
        """Pin the fix in the proof-window script, since it is not importable.

        The compute arms must hold byte ownership at the baseline through
        ``SGLANG_MOE_COLD_TIER_SHM``, not through an equal ratio -- the latter
        would neuter their own treatment.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[5]
        script = (root / "scripts/dev/394_s2_proof/boot_ab.sh").read_text()
        arm = script.split("compute|compute-cal)", 1)[1].split(";;", 1)[0]
        self.assertIn("unset SGLANG_MOE_HOST_SHARD_RATIO", arm)
        self.assertNotIn("export SGLANG_MOE_HOST_SHARD_RATIO", arm)
        self.assertIn("--rank-moe-ratio link", arm)

    def test_a_card_vector_of_the_wrong_length_is_refused(self):
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        os.environ["SGLANG_RANK_CARD_UUIDS"] = "GPU-aaa,GPU-bbb"
        with self.assertRaises(NoComputeLever):
            resolve_moe_compute_placement_flag(self._args())


class TestTheArmCanIdentifyItself(CustomTestCase):
    """#390 dump fields, so a proof arm's null result is a tested null."""

    def test_the_policy_label_defaults_to_the_base_plan(self):
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            compute_policy_label,
        )

        previous = os.environ.pop(COMPUTE_POLICY_ENV, None)
        try:
            self.assertEqual(compute_policy_label(), "base-plan")
            os.environ[COMPUTE_POLICY_ENV] = "link-proportional"
            self.assertEqual(compute_policy_label(), "link-proportional")
        finally:
            os.environ.pop(COMPUTE_POLICY_ENV, None)
            if previous is not None:
                os.environ[COMPUTE_POLICY_ENV] = previous

    def test_the_dump_carries_the_compute_policy_and_vector(self):
        from sglang.srt.layers.moe import expert_stats

        collector = expert_stats.ExpertStatsCollector(path="/dev/null", rank_tag="tp0")
        payload = collector.snapshot(reason="test")
        self.assertIn("moe_compute_policy", payload["totals"])
        self.assertIn("moe_compute_vector", payload["totals"])

    def test_the_policy_env_is_registered(self):
        from sglang.srt.environ import envs

        self.assertIn(COMPUTE_POLICY_ENV, dir(envs))

    def test_the_placement_describes_itself_with_its_provenance(self):
        placement = solve_link_proportional_expert_vector(
            link_source="card-probe-h2d",
            link_provenance="measured",
            link_detail="synthetic profile",
            **PROFILE_ASYMMETRIC_SLOTS,
        )
        self.assertIsInstance(placement, ComputePlacement)
        text = placement.describe()
        self.assertIn("--rank-moe-ratio", text)
        self.assertIn("card-probe-h2d", text)
        self.assertIn("measured", text)


#: The 2026-08-02 ARM3 battery's own configuration, as one more synthetic
#: profile. Every number is read off that battery's record
#: (`/spinning/gpu-battery-results/2026-08-02_439_arm3/RESULTS.md`): the base
#: plan is the resolved `--rank-tp-ratio` from its boot log, the fractions are
#: its launch flag, and 256 is V4-Flash's routed expert count. Nothing branches
#: on it -- it is here because it is the configuration the three defects below
#: were found in, so the falsifiers reproduce them exactly.
BATTERY_BASE_PLAN = [30407, 19080, 19080]
BATTERY_FRACTIONS = [0.485, 0.42, 0.42]
BATTERY_LINKS = [14.42, 6.45, 13.41]
BATTERY_EXPERTS = 256


class TestTheResolverReadsTheLaunchFLAG(CustomTestCase):
    """Defect 1 of the ARM3 battery: the solve saw ``[1.0, 1.0, 1.0]``.

    ``resolve_moe_compute_placement_flag`` runs in the LAUNCHER, before the
    ServerArgs reach the runtime context. ``resident_fraction._from_flag()``
    read them through that context and swallowed the resulting ValueError, so
    the flag source returned None, the default 1.0 broadcast, and the arm was
    refused with "nothing to move" while
    ``--rank-moe-resident-fraction 0.485,0.42,0.42`` sat on the object in the
    resolver's hand.

    The suite above never caught it because ``TestTheResolverRunsEndToEnd``
    supplies the fraction through the ENVIRONMENT, which is the other, working
    source. These cases use the flag and nothing but the flag.
    """

    _ENVS = (
        "SGLANG_RANK_CARD_UUIDS",
        "SGLANG_MOE_HOST_SHARD_RATIO",
        "SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
        "SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS",
        COMPUTE_POLICY_ENV,
        "SGLANG_MOE_COMPUTE_BASE_PLAN",
    )

    def setUp(self):
        super().setUp()
        import os

        self._saved = {k: os.environ.get(k) for k in self._ENVS}
        os.environ["SGLANG_RANK_CARD_UUIDS"] = "GPU-aaa,GPU-bbb,GPU-ccc"
        os.environ["SGLANG_MOE_HOST_SHARD_RATIO"] = ",".join(
            str(w) for w in BATTERY_LINKS
        )
        # The defect's precondition: NOTHING in the environment. The launch
        # carries the fraction on the flag only, which is how the battery ran.
        for key in (
            "SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
            "SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS",
            COMPUTE_POLICY_ENV,
            "SGLANG_MOE_COMPUTE_BASE_PLAN",
        ):
            os.environ.pop(key, None)

    def tearDown(self):
        import os

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().tearDown()

    @staticmethod
    def _args(**overrides):
        class _Args:
            tp_size = 3
            ep_size = 1
            nnodes = 1
            rank_tp_ratio = list(BATTERY_BASE_PLAN)
            rank_moe_ratio = COMPUTE_PLACEMENT_LINK
            rank_moe_resident_fraction = list(BATTERY_FRACTIONS)

            def override(self, source, **fields):
                self.override_source = source
                for key, value in fields.items():
                    setattr(self, key, value)

        args = _Args()
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_the_context_really_is_unpublished_here(self):
        """The defect's precondition, asserted rather than assumed."""
        from sglang.srt.runtime_context import get_context

        with self.assertRaises(ValueError) as caught:
            get_context().server_args
        self.assertIn("not set yet", str(caught.exception))

    def test_can_fail_the_context_route_alone_still_reads_the_default(self):
        """The unfixed behaviour, pinned so the fix cannot be quietly undone.

        Without an explicit ServerArgs there is genuinely nothing to read in
        this process, and 1.0 is the honest answer -- the bug was asking the
        question that way from a caller that HELD the object.
        """
        from sglang.srt.layers.moe.resident_fraction import (
            _from_flag,
            resident_fraction_vector,
        )

        self.assertIsNone(_from_flag())
        self.assertEqual(resident_fraction_vector(tp_size=3), (1.0, 1.0, 1.0))

    def test_the_resolver_reads_the_per_rank_fractions_off_the_launch_args(self):
        from sglang.srt.layers.moe.expert_compute_placement import (
            _resident_fraction_vector,
        )

        seen = _resident_fraction_vector(self._args(), 3)
        self.assertEqual(seen, tuple(BATTERY_FRACTIONS))

    def test_the_battery_arm_now_solves_instead_of_being_refused(self):
        """End to end through the resolver, on the battery's own inputs.

        The vector is asserted exactly: it is the one the battery reached only
        after hand-setting the environment variable as a workaround, so a
        different answer here means the fix resolved something else.
        """
        from sglang.srt.layers.moe.expert_compute_placement import (
            resolve_moe_compute_placement_flag,
        )

        args = self._args()
        placement = resolve_moe_compute_placement_flag(args)
        self.assertIsNotNone(placement)
        self.assertEqual(placement.weights, (160, 79, 119))
        self.assertEqual(args.rank_moe_ratio, [160, 79, 119])

    def test_a_scalar_flag_broadcasts_to_every_rank(self):
        from sglang.srt.layers.moe.expert_compute_placement import (
            _resident_fraction_vector,
        )

        args = self._args(rank_moe_resident_fraction=[0.5])
        self.assertEqual(_resident_fraction_vector(args, 3), (0.5, 0.5, 0.5))

    def test_the_env_flag_cross_check_survives_the_fix(self):
        """The fix hands the module a source; it does not RANK the sources.

        A launch that says two different things is still refused -- dropping
        that check would have been the easy way to make the defect go away.
        """
        import os

        from sglang.srt.layers.moe.expert_compute_placement import (
            _resident_fraction_vector,
        )

        os.environ["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"] = "0.3,0.3,0.3"
        with self.assertRaises(ValueError) as caught:
            _resident_fraction_vector(self._args(), 3)
        self.assertIn("disagree", str(caught.exception))

if __name__ == "__main__":
    unittest.main()
