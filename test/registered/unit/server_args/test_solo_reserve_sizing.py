"""Task #332, posten 3: size the default placement path by a named reserve.

The measurement (2026-07-31, "NVFP4-Beleg" in
``docs/dev/INTEGRATION_R3_VALIDATION.md`` §5): the solo-5090 arm booted with
``--mem-fraction-static 0.90`` and left **2.44 GiB** of the card unused. At
that run's 31.9 KiB per KV token that is ~80,000 tokens outside the pool, next
to the 153,007 that were in it -- so the arm's real ceiling was ~233k, not the
number it reported.

Where the slack comes from: on the default (non-``--rank-gpu-id``) path
``_profile_available_bytes`` withholds
``pre_model_load_memory * (1 - mem_fraction_static)``. That is a blind
percentage of the card, and nothing in it is itemized -- neither the graph
capture nor the activation working set nor the CUDA context appears as a post.
The itemized alternative already existed (#68,
``derived_rank_auto_reserve_mib``: budget = NVML TOTAL minus a reserve that
names its components) but was unreachable without ``--rank-gpu-id``, i.e.
exactly not on a solo boot.

This pins the arithmetic on both sides: the reproduction of the reported gap,
and the conversion the new path performs -- total minus reserve, exactly, with
no margin, cap or rounding layered on top (the same rule
``--rank-gpu-memory-mib`` follows).

CPU only, NVML mocked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import sglang.srt.utils as sglang_utils
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# The measured arm (INTEGRATION_R3_VALIDATION.md, "NVFP4-Beleg")
# ---------------------------------------------------------------------------
#: NVML total of the RTX 5090 in that window, MiB (hygiene section: 30,423 of
#: 32,607 MiB occupied at the tightest point).
CARD_TOTAL_MIB = 32607
#: The fraction the arm was launched with.
ARM_FRACTION = 0.90
#: Free VRAM after the boot, as reported in §5.
ARM_LEFTOVER_GIB = 2.44
#: KV bytes per token: fp8 KV, K and V 2.33 GiB each for 153,007 tokens.
KIB_PER_TOKEN = 31.9
#: The pool the arm actually got.
ARM_TOKENS = 153_007
#: What §5 says a reserve-based boot should reach instead.
ARM_REACHABLE_TOKENS = 233_000

MIB = 1024
GIB_IN_MIB = 1024


def _tokens_for(mib: float) -> float:
    """MiB of KV budget -> tokens, at the arm's measured bytes per token."""
    return mib * MIB / KIB_PER_TOKEN


def make_args(**kwargs) -> ServerArgs:
    """``model_path='dummy'`` short-circuits ``__post_init__``."""
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args, total_mib: int = CARD_TOTAL_MIB) -> ServerArgs:
    with patch.object(
        sglang_utils, "get_device_memory_capacity", return_value=total_mib
    ):
        args._handle_uneven_tp()
    return args


class TestTheReportedGapReproduces(CustomTestCase):
    """The beleg's own numbers, recomputed from the formula that produced them."""

    def test_the_fraction_withholds_a_tenth_of_the_card(self):
        """slack = pre_model_load_memory * (1 - fraction), and on a fresh solo
        boot pre_model_load_memory is essentially the whole card."""
        slack_mib = CARD_TOTAL_MIB * (1 - ARM_FRACTION)
        self.assertAlmostEqual(slack_mib, 3260.7, places=1)
        # Bigger than what was still free after the boot: the difference is
        # what the graphs and activations really took out of it.
        self.assertGreater(slack_mib, ARM_LEFTOVER_GIB * GIB_IN_MIB)

    def test_the_leftover_is_the_reported_eighty_thousand_tokens(self):
        tokens = _tokens_for(ARM_LEFTOVER_GIB * GIB_IN_MIB)
        self.assertAlmostEqual(tokens, 80_000, delta=1_000)

    def test_the_arm_ceiling_is_the_reported_two_hundred_thirty_three_k(self):
        reachable = ARM_TOKENS + _tokens_for(ARM_LEFTOVER_GIB * GIB_IN_MIB)
        self.assertAlmostEqual(reachable, ARM_REACHABLE_TOKENS, delta=2_000)
        # And the reported pool is 66 % of it -- the gap is a third of the arm.
        self.assertAlmostEqual(ARM_TOKENS / reachable, 0.656, places=2)


class TestTheReserveConversion(CustomTestCase):
    def test_the_withheld_slack_equals_the_named_reserve(self):
        """The whole point: what is held back is the number the user wrote.

        On an idle card ``pre_model_load_memory`` is the card's total, so
        ``total * (1 - fraction)`` must come back out as exactly the reserve.
        """
        for reserve in (1024, 2000, 2500, 4096):
            args = run_handler(make_args(rank_auto_reserve_mib=reserve))
            withheld = CARD_TOTAL_MIB * (1 - args.mem_fraction_static)
            self.assertAlmostEqual(withheld, reserve, places=6)

    def test_it_is_total_minus_reserve_over_total(self):
        args = run_handler(make_args(rank_auto_reserve_mib=2000))
        self.assertEqual(
            args.mem_fraction_static, (CARD_TOTAL_MIB - 2000) / CARD_TOTAL_MIB
        )

    def test_nothing_is_rounded_away(self):
        """``round(x, 3)`` -- what the stock derivation does -- would discard
        up to 16 MiB on this card. This path exists to stop giving budget
        away, so it must not round."""
        args = run_handler(make_args(rank_auto_reserve_mib=2500))
        rounded = round(args.mem_fraction_static, 3)
        self.assertNotEqual(args.mem_fraction_static, rounded)
        lost_mib = abs(args.mem_fraction_static - rounded) * CARD_TOTAL_MIB
        self.assertGreater(lost_mib, 1.0)

    def test_no_implicit_margin_is_applied_on_top(self):
        """A reserve of R must not become R + anything."""
        reserve = 3000
        args = run_handler(make_args(rank_auto_reserve_mib=reserve))
        self.assertEqual(
            args.mem_fraction_static, (CARD_TOTAL_MIB - reserve) / CARD_TOTAL_MIB
        )
        self.assertGreater(
            args.mem_fraction_static, (CARD_TOTAL_MIB - reserve - 1) / CARD_TOTAL_MIB
        )

    def test_the_arm_recovers_the_reported_gap(self):
        """Sizing the measured arm by reserve instead of by 0.90.

        A 2 GiB reserve on this card is a smaller withholding than 0.90's
        3261 MiB by 1213 MiB, which at the arm's 31.9 KiB/token is ~39k more
        tokens -- and a reserve chosen at the level the leftover proved was
        spare recovers essentially all of the 80k.
        """
        args = run_handler(make_args(rank_auto_reserve_mib=2048))
        withheld = CARD_TOTAL_MIB * (1 - args.mem_fraction_static)
        recovered = _tokens_for(CARD_TOTAL_MIB * (1 - ARM_FRACTION) - withheld)
        self.assertAlmostEqual(recovered, 39_000, delta=2_000)

        # The leftover the arm measured is the honest upper bound on what a
        # reserve can reclaim; ask for it and the pool reaches the §5 ceiling.
        spare = ARM_LEFTOVER_GIB * GIB_IN_MIB
        tight = run_handler(
            make_args(
                rank_auto_reserve_mib=int(CARD_TOTAL_MIB * (1 - ARM_FRACTION) - spare)
            )
        )
        gained = _tokens_for(
            CARD_TOTAL_MIB * (tight.mem_fraction_static - ARM_FRACTION)
        )
        self.assertAlmostEqual(ARM_TOKENS + gained, ARM_REACHABLE_TOKENS, delta=2_000)


class TestTheDefaultPathIsUnchanged(CustomTestCase):
    def test_no_flag_leaves_the_fraction_underived(self):
        """``_handle_gpu_memory_settings`` still owns the stock heuristic."""
        args = run_handler(make_args())
        self.assertIsNone(args.mem_fraction_static)

    def test_the_auto_sentinel_is_not_an_opt_in(self):
        """'auto' is the flag's DEFAULT, so it cannot mean "size by reserve"
        without changing every boot that never passed the flag."""
        args = run_handler(
            make_args(rank_auto_reserve_mib=ServerArgs.AUTO_RANK_MEMORY_RESERVE_MIB)
        )
        self.assertIsNone(args.mem_fraction_static)

    def test_an_explicit_fraction_alone_still_works(self):
        args = run_handler(make_args(mem_fraction_static=0.9))
        self.assertEqual(args.mem_fraction_static, 0.9)


class TestTheRefusals(CustomTestCase):
    def test_a_fraction_and_a_reserve_together_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserve REPLACES the fraction"):
            run_handler(make_args(rank_auto_reserve_mib=2000, mem_fraction_static=0.9))

    def test_a_per_rank_list_needs_rank_gpu_id(self):
        with self.assertRaisesRegex(ValueError, "per-rank LIST"):
            run_handler(make_args(rank_auto_reserve_mib="3000,2700,2700"))

    def test_a_non_numeric_reserve_is_named(self):
        with self.assertRaisesRegex(ValueError, "must be a MiB integer"):
            run_handler(make_args(rank_auto_reserve_mib="plenty"))

    def test_a_reserve_that_leaves_no_budget_is_rejected(self):
        for reserve in (0, -512, CARD_TOTAL_MIB, CARD_TOTAL_MIB + 1):
            with self.assertRaisesRegex(ValueError, "usable budget"):
                run_handler(make_args(rank_auto_reserve_mib=reserve))

    def test_multi_node_is_out_of_scope(self):
        with self.assertRaisesRegex(ValueError, "single-node"):
            run_handler(make_args(rank_auto_reserve_mib=2000, nnodes=2))

    def test_an_unreadable_device_total_is_named(self):
        with self.assertRaisesRegex(ValueError, "total memory"):
            run_handler(make_args(rank_auto_reserve_mib=2000), total_mib=None)


class TestItStillDefersToTheRankPath(CustomTestCase):
    def test_rank_gpu_id_keeps_owning_the_conversion(self):
        """With ``--rank-gpu-id`` the per-rank budget path converts instead,
        and the reserve keeps its original per-GPU placement meaning."""
        import sglang.srt.server_args as server_args_module

        fake = {0: (32768, 30000), 1: (20480, 19000)}
        args = make_args(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib=2000,
        )
        with patch.object(
            server_args_module,
            "_query_rank_gpu_memory_mib",
            lambda ids: {i: fake[i] for i in sorted(set(ids))},
        ):
            with patch.object(
                sglang_utils, "get_device_memory_capacity", return_value=32768
            ):
                args._handle_uneven_tp()
        # Per-rank fractions, not one card-wide fraction.
        self.assertIsNone(args.mem_fraction_static)
        self.assertEqual(len(args._rank_mem_fraction_static), 2)
        self.assertAlmostEqual(
            args._rank_mem_fraction_static[0], (32768 - 2000) / 32768
        )
        self.assertAlmostEqual(
            args._rank_mem_fraction_static[1], (20480 - 2000) / 20480
        )


if __name__ == "__main__":
    unittest.main()
