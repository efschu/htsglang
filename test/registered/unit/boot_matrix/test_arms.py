"""The boot-matrix arm list is well-formed data (#349).

Cheap invariants a typo in the matrix would break, checked before any GPU time
is spent driving it. Nothing here boots or checks; it reads the tuple.
"""

import unittest

from sglang.srt.boot_matrix.arms import (
    ARMS,
    BASE_EXPECT,
    DFLASH_DRAFT_MODEL,
    EVEN_RATIO_RANK_MIB,
    LANE_RATIO,
    Arm,
    arm_by_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestArmList(CustomTestCase):
    def test_names_are_unique(self):
        names = [a.name for a in ARMS]
        self.assertEqual(len(names), len(set(names)))

    def test_reject_arms_name_their_markers(self):
        for arm in ARMS:
            if arm.kind == "reject":
                self.assertTrue(
                    arm.reject_markers,
                    f"{arm.name}: a reject arm with no markers cannot tell a "
                    "clean refusal from an unrelated crash",
                )
                self.assertFalse(
                    arm.expect, f"{arm.name}: a reject arm never boots"
                )

    def test_boot_arms_declare_an_effective_config(self):
        for arm in ARMS:
            if arm.kind == "boot":
                self.assertTrue(
                    arm.expect,
                    f"{arm.name}: a boot arm with no expected config cannot "
                    "catch the #340 silent-env class",
                )

    def test_the_108_reject_surface_is_covered(self):
        """Every #108 reject reason must have an arm, or the guard it protects
        can rot unnoticed."""
        reject_axes = " ".join(a.axis for a in ARMS if a.kind == "reject")
        for needle in (
            "topk",
            "multi-layer",
            "off the weighted",
            "cross-algorithm",
            "kv-session-offload",
        ):
            self.assertIn(needle, reject_axes, f"no reject arm for {needle!r}")

    def test_the_combined_arm_exists(self):
        """Arm G -- 'all axes together' -- is the one that catches the
        #132 x weightless class, so its absence is a hole in the net."""
        g = arm_by_name("G_all_axes")
        self.assertEqual(g.kind, "boot")
        self.assertIn("132", g.catches)

    def test_bar1_arm_carries_the_capture_caveat(self):
        k = arm_by_name("K_bar1_graphs")
        self.assertTrue(k.capture_note)
        self.assertGreater(
            k.expected_seconds,
            600.0,
            "the bar1 graph arm must budget for the cold-cache capture #366 saw",
        )

    def test_base_expect_is_inherited(self):
        """A_default's declared config is the base one, unmodified."""
        a = arm_by_name("A_default")
        for key, val in BASE_EXPECT.items():
            self.assertEqual(a.expect[key], val)

    def test_arm_validation_rejects_a_markerless_reject(self):
        with self.assertRaises(ValueError):
            Arm(name="x", axis="", catches="", kind="reject")

    def test_arm_validation_rejects_a_bad_coherence_tier(self):
        with self.assertRaises(ValueError):
            Arm(name="x", axis="", catches="", kind="boot", coherence="sometimes")

    def test_arm_by_name_raises_on_unknown(self):
        with self.assertRaises(KeyError):
            arm_by_name("nope")


if __name__ == "__main__":
    unittest.main()


class TestSweep1ArmRepairs(CustomTestCase):
    """Every arm sweep 1 could not boot, pinned to the reason it could not.

    These are not style assertions. Each one names a flag or env the server
    REQUIRES for the crossing the arm claims to exercise; without it the arm
    dies at argument resolution in seconds and the matrix reports a defect that
    is its own.
    """

    def test_offload_arms_opt_into_the_spec_bring_up_gate(self):
        """--enable-kv-session-offload x spec is gated on KVSO_ALLOW_SPEC=1.

        Five arms crossed exactly that and none of them set it, so all five
        died on the gate. The gate is correct; the arms have to opt in the way
        an operator would.

        DERIVED, not enumerated. The first version of this test listed the five
        arms by name, which meant a NEW offload arm could be added and die on
        the same gate with the suite green -- the enumerate-the-known-cases
        shape that #549 cost an afternoon. The condition is now read off each
        arm: enables offload, does not drop the base spec flags, therefore
        needs the gate.
        """
        checked = []
        for arm in ARMS:
            if arm.kind != "boot":
                continue
            if "--enable-kv-session-offload" not in arm.flags:
                continue
            if "--speculative-algorithm" in arm.drop_flags:
                continue  # a deliberate no-spec control (see H)
            checked.append(arm.name)
            self.assertEqual(
                arm.env.get("KVSO_ALLOW_SPEC"),
                "1",
                f"{arm.name} crosses offload with spec and must opt into the "
                "gate, or it dies on the gate and reports whatever the gate "
                "said instead of testing its own axis",
            )
        # The derivation must actually select arms; a filter that matches
        # nothing would make this test vacuously green forever.
        self.assertGreaterEqual(len(checked), 5, checked)

    def test_the_hicache_pair_arm_opts_into_both_gates(self):
        """#550: kvso x HiCache is opt-in behind KVSO_ALLOW_HICACHE on top of
        the spec gate. An arm carrying only one of the two dies on the other
        and reports that refusal as its result."""
        for arm in ARMS:
            if "--enable-hierarchical-cache" not in arm.flags:
                continue
            if "--enable-kv-session-offload" not in arm.flags:
                continue
            self.assertEqual(arm.env.get("KVSO_ALLOW_HICACHE"), "1", arm.name)
            # The joint budget guard can only price HiCache at parse time from
            # an ABSOLUTE --hicache-size; --hicache-ratio has no honest
            # parse-time number, so a contention arm must pin the absolute one.
            self.assertIn("--hicache-size", arm.flags, arm.name)

    def test_h_is_the_no_spec_control_and_needs_no_gate(self):
        """If H ever needs KVSO_ALLOW_SPEC, the gate stopped meaning what it says."""
        arm = arm_by_name("H_ps2_prefill_spill")
        self.assertNotIn("KVSO_ALLOW_SPEC", dict(arm.env))
        self.assertIn("--speculative-algorithm", arm.drop_flags)

    def test_cross_algorithm_arms_carry_the_required_force(self):
        """--speculative-cross-algorithm-force is documented '(required)'."""
        for name in ("C_crossalgo", "D_offload_x_crossalgo", "G_all_axes",
                     "reject_dcp_crossalgo"):
            arm = arm_by_name(name)
            if "--speculative-cross-algorithm" not in arm.flags:
                continue
            self.assertIn(
                "--speculative-cross-algorithm-force",
                arm.flags,
                f"{name} enables cross-algo without the required force value",
            )

    def test_the_dual_lane_arm_carries_its_mandatory_budget(self):
        arm = arm_by_name("L_video_cotenancy")
        self.assertIn("--dual-group-lane-budget-mib", arm.flags)

    def test_the_offlane_reject_drops_the_incompatible_reserve(self):
        """--rank-auto-reserve-mib only applies with --rank-tp-ratio auto, and
        this arm runs off that lane entirely -- so both base flags go."""
        arm = arm_by_name("reject_dcp_offlane")
        self.assertIn("--rank-auto-reserve-mib", arm.drop_flags)
        self.assertIn("--rank-tp-ratio", arm.drop_flags)


class TestRejectArmsNameTheirOwnGuard(CustomTestCase):
    """Item 2, at the data end: a marker set that any refusal can satisfy is
    how two arms reported PASS without reaching their crossing."""

    def test_every_dcp_reject_arm_names_the_dcp_flag(self):
        for arm in ARMS:
            if arm.kind != "reject" or "dcp" not in arm.name:
                continue
            self.assertIn(
                "--draft-kv-layout dcp",
                arm.reject_markers,
                f"{arm.name} could be satisfied by a refusal that never "
                f"mentions the layout it exists to test",
            )

    def test_every_reject_arm_names_at_least_two_markers(self):
        """One marker is a substring away from matching an unrelated guard."""
        for arm in ARMS:
            if arm.kind != "reject":
                continue
            self.assertGreaterEqual(
                len(arm.reject_markers), 2, f"{arm.name} is too loosely pinned"
            )


class TestTheStaleRejectWasRetiredHonestly(CustomTestCase):
    """Item 4. #108 slice 2 REMOVED the draft-extend refusal, so an arm that
    still asserts it reports a defect every run and teaches everyone to ignore
    the matrix. It is replaced by an arm that pins the contract that exists."""

    def test_the_removed_refusal_is_no_longer_asserted(self):
        self.assertNotIn("reject_dcp_draftextend", [a.name for a in ARMS])

    def test_the_replacement_is_a_boot_arm_on_the_covered_lane(self):
        arm = arm_by_name("M_dcp_draftextend")
        self.assertEqual(arm.kind, "boot")
        self.assertIn("--draft-kv-layout", arm.flags)
        self.assertIn("dcp", arm.flags)

    def test_the_replacement_declares_the_layout_it_must_resolve_to(self):
        """A silent fallback to 'replicated' is the regression to catch."""
        arm = arm_by_name("M_dcp_draftextend")
        self.assertEqual(arm.expect.get("draft_kv_layout"), "dcp")


class TestSweep2ArmRepairs(CustomTestCase):
    """The gates sweep 2 uncovered behind the sweep-1 ones.

    Each repair revealed the next requirement, which is what a first pass
    through a never-executed arm looks like. These pin the second layer.
    """

    def test_cross_algo_and_dflash_arms_carry_a_draft_checkpoint(self):
        """Both rungs stay resident, so the DFLASH rung's weights are needed
        even when the NEXTN rung is the forced one."""
        for name in ("C_crossalgo", "D_offload_x_crossalgo", "G_all_axes",
                     "I_dflash_shards", "reject_dcp_crossalgo"):
            arm = arm_by_name(name)
            self.assertIn(
                "--speculative-draft-model-path",
                arm.flags,
                f"{name} refuses at boot without a draft checkpoint",
            )
            i = list(arm.flags).index("--speculative-draft-model-path")
            self.assertEqual(arm.flags[i + 1], DFLASH_DRAFT_MODEL)

    def test_the_canonical_draft_flag_spelling_is_used(self):
        """#382 note: --speculative-draft-model is an ALIAS of
        --speculative-draft-model-path, so both parse -- but the two refusal
        messages name different ones. The arms use the canonical name."""
        for arm in ARMS:
            self.assertNotIn("--speculative-draft-model", arm.flags)

    def test_arms_on_the_even_split_omit_the_ratio_and_supply_budgets(self):
        """The even split is expressed by OMITTING --rank-tp-ratio.

        ServerArgs hard-rejects an all-identical explicit vector -- "identical
        entries is the even split, omit the flag instead". Sweep 2's repair
        spelled out 1,1,1 and the validation that landed afterwards refuses
        it, which is the validation doing its job. Once the ratio is not
        `auto` the server also stops deriving budgets from NVML, so the
        per-rank budget has to be explicit.
        """
        for name in ("L_video_cotenancy", "reject_dcp_offlane"):
            arm = arm_by_name(name)
            self.assertIn("--rank-tp-ratio", arm.drop_flags)
            self.assertIn("--rank-gpu-memory-mib", arm.flags)
            self.assertIn(
                "--rank-auto-reserve-mib",
                arm.drop_flags,
                f"{name} must drop the auto reserve it can no longer use",
            )
        # offlane runs on the plain even split, so it restates nothing; the
        # lane arm must restate a ratio, because its own guard demands one.
        self.assertNotIn(
            "--rank-tp-ratio", arm_by_name("reject_dcp_offlane").flags
        )

    def test_the_lane_arm_carries_a_non_uniform_ratio_that_nests(self):
        """The lane sits between two guards: it REQUIRES an explicit integer
        vector, and an all-identical vector is refused. So it needs a
        non-uniform ratio that still nests -- verified card-lessly against
        this rig's model geometry with the real nesting functions, not
        guessed."""
        arm = arm_by_name("L_video_cotenancy")
        i = list(arm.flags).index("--rank-tp-ratio")
        parts = arm.flags[i + 1].split(",")
        self.assertGreater(len(set(parts)), 1, "an identical vector is refused")
        self.assertEqual(parts, ["2", "1", "1"])

    def test_the_lane_budget_is_per_rank_because_its_shards_differ(self):
        """A scalar applies uniformly; under 2,1,1 the shards do not."""
        arm = arm_by_name("L_video_cotenancy")
        i = list(arm.flags).index("--rank-gpu-memory-mib")
        self.assertEqual(len(arm.flags[i + 1].split(",")), 3)

    def test_the_declared_lane_ratio_actually_nests(self):
        """Pins the derivation itself, so a model or geometry change that
        breaks 2,1,1 fails here instead of in the window."""
        from sglang.srt.distributed.dual_group import (
            derive_nested_plan,
            nesting_failures,
            transformer_nesting_probes,
        )

        ratio = [int(x) for x in LANE_RATIO.split(",")]
        plan = derive_nested_plan(ratio)
        probes = transformer_nesting_probes(
            plan,
            num_attention_heads=24,
            num_kv_heads=4,
            intermediate_size=17408,
            linear_attn_units=16,
            vocab_units=248320,
            weight_block_size=[128, 128],
            quant_method="fp8",
        )
        self.assertEqual(nesting_failures(plan, probes), [])

    def test_the_budget_fits_the_smallest_card_on_this_rig(self):
        """20054 MiB 3080s; the value must leave room for the CUDA context."""
        self.assertLess(int(EVEN_RATIO_RANK_MIB), 20054)
        self.assertGreater(int(EVEN_RATIO_RANK_MIB), 8000)
