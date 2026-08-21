"""#784/#789: the corridor band grades USER-FREE VRAM, not the flip's working capital.

THE DEFECT THIS SUITE PINS. An armed phase-flip boot must hold an arming floor
on every rank so a flip can arm out of that column. With the shipped constants
that floor is at minimum

    corridor_band_floor_mib() + DEFAULT_SEAM_ENTRY_RESERVE_MIB + DEFAULT_ARMING_MARGIN_MIB
    = 819 + 512 + 192 = 1523 MiB

while the band ceiling the boot is graded against is

    corridor_band_ceiling_mib() = 1024 + 20 % = 1228 MiB

so **1523 > 1228 on every rank, on any rig, unconditionally**. The boot is
graded against a threshold its own arming rule forbids it to reach. Nothing
checked the pair: ``check_threshold_pair`` only ever refused an arming floor
BELOW the law, and ``corridor_band_ceiling_mib`` had no production consumer at
all.

THE RESOLUTION, AND WHY IT IS NOT A WIDENING OF THE BAND. Under the reserve
semantics a reserve is the USER's external free space, and internal demand is
booked exactly, in the ledger. The seam entry reserve and the arming margin are
internal demand: they are working capital the guard actively defends so that a
flip can arm. They are not idle VRAM buying no tokens -- they buy the flip. So
they belong in the ledger's charged column, not in the free column the band
grades. Crediting them is the Option-A formula the acceptance verdict has
documented since 2026-07-22 (``net = free_min - sum(registered posts)``) and
never implemented: ``corridor_verdict_774.sh`` hard-codes ``posts = {i: 0}``,
so ``net == raw`` on every boot that has ever been graded.

WHAT THE CREDIT MUST NOT DO, pinned here because it is the whole risk of this
change: it must not manufacture a pass. A rank resting far above its arming
floor is genuinely stranding VRAM, and crediting the arming reserve must leave
that rank over-band. The observed instance is boot instr9, where ranks 0 and 1
rest at 5229 / 6612 MiB against a 1523 MiB floor; after the credit they must
still read ~4525 / ~5908 MiB net and still FAIL. The credit makes the grading
honest; it does not make the boot pass.
"""

import unittest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.managers import phase_flip_seam_reserve as sr


class TheShippedPairIsUnsatisfiableWithoutTheCredit(unittest.TestCase):
    """Defect A, stated as arithmetic so it cannot be argued away."""

    def test_the_minimum_arming_floor_exceeds_the_band_ceiling(self):
        # The floor a rank must hold with a ZERO measured seam draw. A zero
        # measurement does NOT reach the guard as a zero: the runtime takes
        # max(DEFAULT_SEAM_ENTRY_RESERVE_MIB, measured_draw), so the shipped
        # 512 MiB allowance is the smallest reserve the pair can ever carry
        # and 819 + 512 + 192 = 1523 is the smallest floor any rig can ask
        # for.
        minimum = cg.arming_floor_mib() + sr.DEFAULT_ARMING_MARGIN_MIB
        self.assertEqual(minimum, 1523)
        self.assertGreater(
            minimum,
            cg.corridor_band_ceiling_mib(),
            "if this ever stops being true the defect is fixed elsewhere and "
            "this suite should be re-derived rather than relaxed",
        )

    def test_the_gap_is_the_committed_working_capital(self):
        arming = cg.arming_floor_mib() + sr.DEFAULT_ARMING_MARGIN_MIB
        self.assertEqual(
            cg.committed_arming_mib(arming),
            arming - cg.corridor_band_floor_mib(),
        )


class TheCreditIsTheArmingFloorAboveTheBandFloor(unittest.TestCase):
    def test_a_bare_band_floor_is_not_working_capital(self):
        # A rank asked to hold exactly the band floor holds nothing on the
        # flip's behalf: the whole column is the user's.
        self.assertEqual(cg.committed_arming_mib(cg.corridor_band_floor_mib()), 0)

    def test_the_credit_is_never_negative(self):
        self.assertEqual(cg.committed_arming_mib(0), 0)
        self.assertEqual(cg.committed_arming_mib(cg.corridor_band_floor_mib() - 300), 0)

    def test_a_measured_arena_tail_raises_the_credit_by_its_own_size(self):
        # Rank 2 of boot instr9: floor 3226 = 819 + 2215 measured draw + 192.
        base = cg.arming_floor_mib() + sr.DEFAULT_ARMING_MARGIN_MIB
        tail = 2215 - cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB
        raised = cg.arming_floor_mib(seam_entry_reserve_mib=2215) + (
            sr.DEFAULT_ARMING_MARGIN_MIB
        )
        self.assertEqual(
            cg.committed_arming_mib(raised) - cg.committed_arming_mib(base), tail
        )


class TheGradedQuantityIsFreeMinusTheCredit(unittest.TestCase):
    def test_a_rank_resting_exactly_at_its_arming_floor_lands_on_the_band_floor(self):
        arming = cg.arming_floor_mib() + sr.DEFAULT_ARMING_MARGIN_MIB
        self.assertEqual(
            cg.net_free_mib(arming, arming),
            cg.corridor_band_floor_mib(),
        )

    def test_that_rank_is_therefore_in_band(self):
        arming = cg.arming_floor_mib() + sr.DEFAULT_ARMING_MARGIN_MIB
        net = cg.net_free_mib(arming, arming)
        self.assertGreaterEqual(net, cg.corridor_band_floor_mib())
        self.assertLessEqual(net, cg.corridor_band_ceiling_mib())

    def test_the_same_holds_for_a_rank_with_a_large_measured_arena_tail(self):
        # The credit follows the rank's OWN floor, so rank 2's 3226 MiB floor
        # is not punished for being larger than rank 0's 1523.
        arming = cg.arming_floor_mib(seam_entry_reserve_mib=2215) + (
            sr.DEFAULT_ARMING_MARGIN_MIB
        )
        self.assertEqual(cg.net_free_mib(arming, arming), cg.corridor_band_floor_mib())


class TheCreditDoesNotManufactureAPass(unittest.TestCase):
    """The anti-papering pin. Stranded VRAM must stay visible."""

    ARMING = 1523

    def test_boot_instr9_rank0_is_still_over_band_after_the_credit(self):
        net = cg.net_free_mib(5229, self.ARMING)
        self.assertEqual(net, 5229 - (self.ARMING - cg.corridor_band_floor_mib()))
        self.assertGreater(net, cg.corridor_band_ceiling_mib())

    def test_boot_instr9_rank1_is_still_over_band_after_the_credit(self):
        net = cg.net_free_mib(6612, self.ARMING)
        self.assertGreater(net, cg.corridor_band_ceiling_mib())

    def test_boot_instr9_rank2_is_still_over_band_after_its_own_larger_credit(self):
        net = cg.net_free_mib(5475, 3226)
        self.assertGreater(net, cg.corridor_band_ceiling_mib())

    def test_a_breach_below_the_arming_floor_stays_a_breach(self):
        # Free BELOW the floor means the flip cannot arm; the credit must not
        # lift such a rank into the band.
        net = cg.net_free_mib(self.ARMING - 400, self.ARMING)
        self.assertLess(net, cg.corridor_band_floor_mib())

    def test_the_credit_is_monotone_in_free(self):
        a = cg.net_free_mib(2000, self.ARMING)
        b = cg.net_free_mib(3000, self.ARMING)
        self.assertEqual(b - a, 1000)


class TheThresholdPairGateNowHasACeilingSide(unittest.TestCase):
    def test_can_fail_a_band_whose_floor_exceeds_its_ceiling_is_refused(self):
        # A fraction above 1.0 inverts the band. Before the ceiling-side
        # check, such a pair was accepted in silence and every boot under it
        # was ungradeable.
        with self.assertRaises(ValueError) as ctx:
            cg.check_threshold_pair(2000, band_floor_mib=1900, band_ceiling_mib=1200)
        self.assertIn("1900", str(ctx.exception))
        self.assertIn("1200", str(ctx.exception))

    def test_the_shipped_pair_is_accepted_once_the_credit_exists(self):
        arming = cg.arming_floor_mib() + sr.DEFAULT_ARMING_MARGIN_MIB
        cg.check_threshold_pair(arming)  # must not raise

    def test_the_floor_side_refusal_is_unchanged(self):
        with self.assertRaises(ValueError):
            cg.check_threshold_pair(cg.corridor_band_floor_mib() - 1)


class TheRegisteredPostSurvivesTheRoundTrip(unittest.TestCase):
    """The credit is worthless unless the acceptance verdict can read it.

    ``corridor_verdict_774.sh`` documents ``net = free_min - sum(registered
    posts)`` and hard-codes the posts to zero, so it has always graded raw
    free. These pin the ONE line the boot emits and the ONE parser that reads
    it back, so the verdict never reimplements the format in awk.
    """

    def test_a_post_round_trips_to_the_committed_credit(self):
        arming = 1523
        line = cg.format_corridor_post(rank=0, gpu_id=2, arming_mib=arming)
        self.assertEqual(
            cg.parse_corridor_posts(line), {2: cg.committed_arming_mib(arming)}
        )

    def test_the_line_carries_its_own_prefix(self):
        line = cg.format_corridor_post(rank=1, gpu_id=0, arming_mib=1523)
        self.assertTrue(line.startswith(cg.CORRIDOR_POST_PREFIX))

    def test_unrelated_log_text_contributes_nothing(self):
        self.assertEqual(cg.parse_corridor_posts("no posts here\nnor here\n"), {})

    def test_co_resident_ranks_on_one_card_ADD(self):
        # Two ranks sharing a physical GPU each hold their own arming floor,
        # so the card's committed column is the sum. Not this rig's layout,
        # but the rule has to be general.
        text = "\n".join(
            (
                cg.format_corridor_post(rank=0, gpu_id=1, arming_mib=1523),
                cg.format_corridor_post(rank=1, gpu_id=1, arming_mib=1523),
            )
        )
        self.assertEqual(
            cg.parse_corridor_posts(text), {1: 2 * cg.committed_arming_mib(1523)}
        )

    def test_a_rank_that_re_emits_is_counted_ONCE_at_its_latest_value(self):
        # THE DOUBLE-COUNT TRAP. A phase-flip boot resolves its arming floor
        # in the PP phase and again when the TP stack is built, so the same
        # rank emits twice in one log. Summing the lines would charge the card
        # twice and hand the verdict a pass it did not earn.
        text = "\n".join(
            (
                cg.format_corridor_post(rank=0, gpu_id=2, arming_mib=1523),
                cg.format_corridor_post(rank=0, gpu_id=2, arming_mib=3226),
            )
        )
        self.assertEqual(
            cg.parse_corridor_posts(text), {2: cg.committed_arming_mib(3226)}
        )

    def test_a_rank_that_moved_card_is_not_counted_on_both(self):
        text = "\n".join(
            (
                cg.format_corridor_post(rank=0, gpu_id=0, arming_mib=1523),
                cg.format_corridor_post(rank=0, gpu_id=1, arming_mib=1523),
            )
        )
        self.assertEqual(
            cg.parse_corridor_posts(text), {1: cg.committed_arming_mib(1523)}
        )

    def test_a_floor_at_the_band_floor_registers_no_post(self):
        line = cg.format_corridor_post(
            rank=0, gpu_id=0, arming_mib=cg.corridor_band_floor_mib()
        )
        self.assertEqual(cg.parse_corridor_posts(line), {0: 0})

    def test_a_malformed_line_is_ignored_rather_than_killing_the_verdict(self):
        text = cg.CORRIDOR_POST_PREFIX + " rank=x gpu=y arming_floor_mib=z\n"
        self.assertEqual(cg.parse_corridor_posts(text), {})


class TheVerdictIsCutPerPhase(unittest.TestCase):
    """#784: the pool is solved in one phase and was graded in another.

    A phase-flip boot binds its KV pool in the PP prefill layout -- that is
    where the id space is min-reduced across ranks and where the binding rank's
    bracket runs down to its arming floor. The TP decode layout sizes nothing:
    it inherits the id space through the canonical KV page, and it has released
    its arena tail, so its free column reads gibibytes higher.

    The acceptance verdict had no phase detection at all. It sampled one window
    and called it the verdict. On 2026-08-20 that window fell entirely inside
    the TP phase -- rank 2 read 5475 MiB there while at rest in PP it sat at
    3232 MiB against a 3226 MiB floor, six MiB of slack. The boot was graded on
    a phase that sizes nothing, and the number it was graded on was 2243 MiB
    away from the number that binds.

    So a window that never observed the sizing phase is NOT a pass. It is an
    absent measurement, and saying so is the point.
    """

    def _nets(self, **by_phase):
        return by_phase

    def test_the_sizing_phase_is_the_prefill_layout(self):
        self.assertEqual(cg.SIZING_PHASE, "pp")

    def test_a_window_that_never_sampled_the_sizing_phase_is_not_a_pass(self):
        verdict, findings = cg.phase_corridor_verdict({"tp": {0: 900, 1: 900, 2: 900}})
        self.assertNotEqual(verdict, "PASS")
        self.assertTrue(any("pp" in f for f in findings))

    def test_in_band_in_the_sizing_phase_is_a_pass(self):
        verdict, _ = cg.phase_corridor_verdict({"pp": {0: 900, 1: 1000, 2: 825}})
        self.assertEqual(verdict, "PASS")

    def test_over_band_in_the_sizing_phase_fails(self):
        verdict, findings = cg.phase_corridor_verdict(
            {"pp": {0: 6323, 1: 5062, 2: 825}}
        )
        self.assertEqual(verdict, "FAIL")
        # The binder is correctly filled; the other two are the stranding.
        self.assertTrue(any("GPU0" in f for f in findings))
        self.assertTrue(any("GPU1" in f for f in findings))
        self.assertFalse(any("GPU2" in f for f in findings))

    def test_over_band_in_the_decode_phase_alone_does_not_fail_the_boot(self):
        # The TP window reads high because the arena tail is released there.
        # That is the layout doing what it should, not a sizing defect, so it
        # is reported and does not decide the verdict.
        verdict, findings = cg.phase_corridor_verdict(
            {"pp": {0: 900, 1: 900, 2: 825}, "tp": {0: 4525, 1: 5908, 2: 3068}}
        )
        self.assertEqual(verdict, "PASS")
        self.assertTrue(any("tp" in f for f in findings))

    def test_a_near_oom_reading_fails_in_any_phase(self):
        verdict, _ = cg.phase_corridor_verdict(
            {"pp": {0: 900, 1: 900, 2: 825}, "tp": {0: 120, 1: 900, 2: 900}}
        )
        self.assertEqual(verdict, "FAIL")

    def test_under_band_in_the_sizing_phase_warns_and_does_not_stop(self):
        # The corridor law is asymmetric by the user's own ruling: over-band
        # is a failed acceptance, under-band is a finding and a planner task.
        verdict, findings = cg.phase_corridor_verdict({"pp": {0: 400, 1: 900, 2: 900}})
        self.assertEqual(verdict, "WARN")
        self.assertTrue(any("GPU0" in f for f in findings))

    def test_no_samples_at_all_is_not_a_pass(self):
        verdict, _ = cg.phase_corridor_verdict({})
        self.assertNotEqual(verdict, "PASS")


class TheCodeAndItsAcceptanceGateAgreeOnTheCeiling(unittest.TestCase):
    def test_the_ceiling_is_the_rounded_value_the_verdict_script_uses(self):
        # corridor_verdict_774.sh computes int(round(law + law * fraction)) =
        # 1229 while the code used int(...) = 1228. One number, one place.
        law = cg.corridor_law_mib()
        self.assertEqual(
            cg.corridor_band_ceiling_mib(),
            int(round(law + law * cg.CORRIDOR_BAND_FRACTION)),
        )


if __name__ == "__main__":
    unittest.main()
