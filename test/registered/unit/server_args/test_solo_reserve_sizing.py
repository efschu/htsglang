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

Task #336 then fixed WHICH card that total comes from. The first cut read
``get_device_memory_capacity``, i.e. ``nvidia-smi --query-gpu=memory.total``
minimised over every card the DRIVER lists. Measured 2026-07-31 on the same
arm: pinned by UUID to the 5090 and resident there (NVML index 1,
30,423/32,607 MiB, both 3080s at 0 MiB), the reserve line still said "device
total 20480 MiB" -- a 3080 -- and produced fraction 0.900000, by coincidence
exactly the anchor's ``--mem-fraction-static``. The pool came out byte-
identical (153,007 tokens) and the knob looked like a clean reproduction while
doing nothing. ``TestItReadsTheCardTheProcessRunsOn`` reproduces that
renumbering and pins the fix.

CPU only, NVML mocked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
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
#: NVML total of the RTX 3080s in the same rig -- and the number the broken
#: sizing line printed while pinned to the 5090.
OTHER_CARD_TOTAL_MIB = 20480
#: NVML index the 5090 sat at in that window; index 0 was a 3080.
FIVE_NVML_INDEX = 1

MIB = 1024
GIB_IN_MIB = 1024


def _tokens_for(mib: float) -> float:
    """MiB of KV budget -> tokens, at the arm's measured bytes per token."""
    return mib * MIB / KIB_PER_TOKEN


def make_args(**kwargs) -> ServerArgs:
    """``model_path='dummy'`` short-circuits ``__post_init__``."""
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args, total_mib: int = CARD_TOTAL_MIB) -> ServerArgs:
    """Run the handler with NVML answering ``total_mib`` for every visible id.

    ``total_mib=None`` stands for "NVML could not resolve the device", which
    the path must refuse loudly rather than paper over.
    """

    def fake_totals(gpu_ids, flag):
        if total_mib is None:
            raise ValueError(f"{flag} could not resolve the device total (#336).")
        return {gpu_id: total_mib for gpu_id in sorted(set(gpu_ids))}

    with patch.object(server_args_module, "_query_gpu_total_mib", fake_totals):
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
        with self.assertRaisesRegex(ValueError, "could not resolve"):
            run_handler(make_args(rank_auto_reserve_mib=2000), total_mib=None)


# ---------------------------------------------------------------------------
# #336: the total must come from the card this process actually runs on
# ---------------------------------------------------------------------------


class TestItReadsTheCardTheProcessRunsOn(CustomTestCase):
    """The measured no-op: right card, wrong total, plausible-looking result.

    The rig is 2x RTX 3080 (20,480 MiB) + 1x RTX 5090 (32,607 MiB) and NVML
    enumerates the 5090 at index 1. The arm pinned CUDA_VISIBLE_DEVICES to the
    5090's UUID, so inside the process the card is CUDA device 0 -- while NVML
    device 0 is a 3080 and ``nvidia-smi``'s driver-wide minimum is 20,480.
    """

    @staticmethod
    def _nvml_rig(mapping, nvml_totals):
        """Patch the CUDA -> NVML bridge and NVML's per-index totals.

        Exercises the real ``_query_gpu_total_mib``, so the renumbering is
        crossed by the code under test and not by the fixture.
        """
        import contextlib

        class _FakePynvml:
            NVMLError = Exception

            @staticmethod
            def nvmlInit():
                return None

            @staticmethod
            def nvmlShutdown():
                return None

            @staticmethod
            def nvmlDeviceGetCount():
                return len(nvml_totals)

            @staticmethod
            def nvmlDeviceGetHandleByIndex(index):
                return index

            @staticmethod
            def nvmlDeviceGetMemoryInfo(handle):
                class _Mem:
                    total = nvml_totals[handle] * 1024 * 1024

                return _Mem()

        @contextlib.contextmanager
        def _ctx():
            import sys

            with patch.dict(sys.modules, {"pynvml": _FakePynvml}):
                with patch.object(
                    server_args_module,
                    "_torch_to_nvml_gpu_index_mapping",
                    lambda: dict(mapping),
                ):
                    yield

        return _ctx()

    #: The rig as NVML saw it: index 0 and 2 are 3080s, index 1 is the 5090.
    RIG = [OTHER_CARD_TOTAL_MIB, CARD_TOTAL_MIB, OTHER_CARD_TOTAL_MIB]

    def test_a_uuid_pin_sizes_against_the_pinned_card_not_nvml_zero(self):
        """CUDA device 0 IS the 5090 here; NVML index 0 is not."""
        args = make_args(rank_auto_reserve_mib=2048)
        with self._nvml_rig({0: FIVE_NVML_INDEX}, self.RIG):
            args._handle_uneven_tp()
        self.assertEqual(
            args.mem_fraction_static, (CARD_TOTAL_MIB - 2048) / CARD_TOTAL_MIB
        )

    def test_the_old_behaviour_would_have_produced_the_measured_no_op(self):
        """Reading NVML index 0 gives 0.90 -- exactly the anchor's fraction.

        This is the falsifier for the whole posten: the wrong card does not
        produce an obviously wrong number, it produces the number the run was
        being compared against.
        """
        wrong = (OTHER_CARD_TOTAL_MIB - 2048) / OTHER_CARD_TOTAL_MIB
        self.assertEqual(wrong, ARM_FRACTION)
        right = (CARD_TOTAL_MIB - 2048) / CARD_TOTAL_MIB
        self.assertNotEqual(right, ARM_FRACTION)
        # And the difference is the ~39k tokens the beleg said were owed.
        gained = _tokens_for(CARD_TOTAL_MIB * (right - wrong))
        self.assertAlmostEqual(gained, 39_000, delta=2_000)

    def test_a_divergent_enumeration_order_without_a_pin_is_crossed(self):
        """No CVD pin, three visible cards, torch order != NVML order.

        torch's FASTEST_FIRST puts the 5090 at CUDA index 0 while NVML keeps
        it at index 1. A solo boot lands on CUDA 0, so the total must be the
        5090's.
        """
        args = make_args(rank_auto_reserve_mib=2048)
        with self._nvml_rig({0: 1, 1: 0, 2: 2}, self.RIG):
            args._handle_uneven_tp()
        self.assertEqual(
            args.mem_fraction_static, (CARD_TOTAL_MIB - 2048) / CARD_TOTAL_MIB
        )

    def test_base_gpu_id_moves_which_card_is_read(self):
        """``--base-gpu-id 1`` places rank 0 on CUDA 1, i.e. NVML 0 here."""
        args = make_args(rank_auto_reserve_mib=2048, base_gpu_id=1)
        with self._nvml_rig({0: 1, 1: 0, 2: 2}, self.RIG):
            args._handle_uneven_tp()
        self.assertEqual(
            args.mem_fraction_static,
            (OTHER_CARD_TOTAL_MIB - 2048) / OTHER_CARD_TOTAL_MIB,
        )

    def test_an_unresolvable_mapping_is_loud_not_a_fallback(self):
        """No bridge -> refuse. The silent fallback IS the defect."""
        args = make_args(rank_auto_reserve_mib=2048)
        with self._nvml_rig({}, self.RIG):
            with self.assertRaisesRegex(ValueError, "Refusing to guess"):
                args._handle_uneven_tp()
        self.assertIsNone(args.mem_fraction_static)

    def test_a_device_outside_nvmls_range_is_loud(self):
        args = make_args(rank_auto_reserve_mib=2048)
        with self._nvml_rig({0: 7}, self.RIG):
            with self.assertRaisesRegex(ValueError, "Refusing to guess"):
                args._handle_uneven_tp()

    def test_mixed_cards_under_one_scalar_reserve_are_refused(self):
        """One fraction cannot be exact on a 20 GiB and a 32 GiB card."""
        args = make_args(rank_auto_reserve_mib=2048, tp_size=2)
        with self._nvml_rig({0: 1, 1: 0}, self.RIG):
            with self.assertRaisesRegex(ValueError, "different totals"):
                args._handle_uneven_tp()

    def test_identical_cards_across_ranks_are_fine(self):
        args = make_args(rank_auto_reserve_mib=2048, tp_size=2)
        with self._nvml_rig({0: 0, 1: 2}, self.RIG):
            args._handle_uneven_tp()
        self.assertEqual(
            args.mem_fraction_static,
            (OTHER_CARD_TOTAL_MIB - 2048) / OTHER_CARD_TOTAL_MIB,
        )

    def test_the_placement_enumerates_the_local_rank_grid(self):
        self.assertEqual(make_args()._default_placement_gpu_ids(), [0])
        self.assertEqual(
            make_args(tp_size=4)._default_placement_gpu_ids(), [0, 1, 2, 3]
        )
        self.assertEqual(
            make_args(tp_size=2, gpu_id_step=2)._default_placement_gpu_ids(), [0, 2]
        )
        self.assertEqual(
            make_args(tp_size=2, base_gpu_id=4)._default_placement_gpu_ids(), [4, 5]
        )

    def test_a_non_cuda_device_keeps_the_capacity_helper(self):
        """NVML is CUDA-only; an XPU/HPU boot has no bridge to cross."""
        args = make_args(rank_auto_reserve_mib=2048, device="xpu")
        with patch.object(
            sglang_utils, "get_device_memory_capacity", return_value=16384
        ):
            with self._nvml_rig({0: 0}, self.RIG):
                args._handle_uneven_tp()
        self.assertEqual(args.mem_fraction_static, (16384 - 2048) / 16384)


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
