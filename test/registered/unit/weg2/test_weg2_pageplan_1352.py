# SPDX-License-Identifier: Apache-2.0
"""#1352 REMAP -- the page plan, at THIS RIG's measured geometries.

THE NUMBERS ARE NOT INVENTED.  Every per-card page count below is the
``WEG2-FLIP-TAG ... granules=`` census of boot ``weg2xsn20``
(``/spinning/evidence-665-f1/boot_weg2_weg2xsn20_3267f109fb_0911_154706.{P,D}.log``),
which is the DEVICE-side instrument -- ``tms_tag_bytes`` over the saver's own
metadata, summed per group and per card, with the granule count printed beside
the bytes.  That distinction is the whole reason this file quotes it rather
than the ring table: see :class:`InstrumentProvenanceTest`.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import xchg_pageplan as pp
from sglang.test.test_utils import CustomTestCase

MIB = 1024 * 1024
PAGE = pp.PAGE_BYTES_DEFAULT

#: boot weg2xsn20, WEG2-FLIP-TAG granules, per card, per GROUP.  nvml1 is the
#: 5090; nvml0 and nvml2 are the two 3080s.  Bytes are 2 MiB x pages exactly.
MEASURED_PAGES = {
    # card -> (group P pages, group D pages)
    1: (8210, 9304),
    0: (2855, 3568),
    2: (4676, 3568),
}

#: The PP cut of the standing form, read from the RING-CKPT lines of xsn21b's
#: front log (attn + linear layers per stage): 39 / 13 / 12, which is NOT the
#: 44/10/10 the briefing carried.  Kept here because the tensor counts below
#: are derived from it.
PP_STAGE_LAYERS = (39, 13, 12)

#: The TP cut of group D, from the B4n smoke's own printed widths
#: (``tp_widths=[2720,1120,1280]`` against an even split of 1706): 17/7/8 of 32.
TP_WIDTHS = (17, 7, 8)


def _layout(group, card, tag, sizes, *, base=0):
    """A SideLayout laid out contiguously from ``base``, pieces in order."""
    pieces = []
    offset = base
    for i, n in enumerate(sizes):
        pieces.append(pp.PieceExtent(param_name=f"t{i}", card=card, tag=tag,
                                     offset=offset, nbytes=n))
        offset += n
    return pp.SideLayout(group=group, card=card, tag=tag,
                         pieces=tuple(pieces), modelled=False)


def _leg_layouts(run_bytes=3 * MIB + 7):
    """The three cards of this rig, as PP-against-TP RUNS.

    A RUN is the unit both sides agree on -- the same bytes on both ends, which
    is what ``_feeders`` requires and what the transport's ``_blocks_of``
    already produces.  Under PP, card ``s`` holds every run of its own layers;
    under TP, card ``t`` holds run ``t`` of EVERY layer.  So a P card's runs
    scatter to all three D cards and a D card's runs gather from all three P
    cards, which is the actual cross-card shape -- and the diagonal (run ``t``
    of a layer held by stage ``t``) is the on-card minority the Cut-1
    simplification deliberately sends through the host anyway.

    ``run_bytes`` is deliberately NOT a multiple of the page size: every run
    then straddles page lines on both sides, which is the case rule 3 exists
    for and the one a page-aligned fixture would never exercise.
    """
    p_pieces = {0: [], 1: [], 2: []}
    d_pieces = {0: [], 1: [], 2: []}
    p_off = {0: 0, 1: 0, 2: 0}
    d_off = {0: 0, 1: 0, 2: 0}
    layer = 0
    for stage, n_layers in enumerate(PP_STAGE_LAYERS):
        card = (1, 0, 2)[stage]           # PP stage -> nvml card of this rig
        for _ in range(n_layers):
            for t, width in enumerate(TP_WIDTHS):
                nbytes = run_bytes * width
                name = f"L{layer}.r{t}"
                p_pieces[card].append(
                    pp.PieceExtent(name, card, "weights_p", p_off[card], nbytes))
                p_off[card] += nbytes
                d_card = (1, 0, 2)[t]
                d_pieces[d_card].append(
                    pp.PieceExtent(name, d_card, "weights_d", d_off[d_card], nbytes))
                d_off[d_card] += nbytes
            layer += 1
    srcs = tuple(pp.SideLayout("P", c, "weights_p", tuple(p_pieces[c]), False)
                 for c in (1, 0, 2))
    dsts = tuple(pp.SideLayout("D", c, "weights_d", tuple(d_pieces[c]), False)
                 for c in (1, 0, 2))
    return srcs, dsts


def _sized(group, card, tag, pages, n_runs=64):
    """A layout of ``n_runs`` equal runs summing to exactly ``pages`` pages."""
    total = pages * PAGE
    sizes = [total // n_runs] * n_runs
    sizes[-1] += total - sum(sizes)
    return _layout(group, card, tag, sizes)


class PageArithmeticTest(CustomTestCase):
    def test_a_piece_reports_every_page_it_straddles(self):
        """A piece that starts and ends mid-page touches BOTH neighbours'."""
        piece = pp.PieceExtent("t", 0, "w", offset=PAGE - 16, nbytes=32)
        self.assertEqual(list(piece.pages(PAGE)), [0, 1])

    def test_an_empty_piece_touches_no_page(self):
        piece = pp.PieceExtent("t", 0, "w", offset=0, nbytes=0)
        self.assertEqual(list(piece.pages(PAGE)), [])

    def test_boundary_pages_are_counted_not_estimated(self):
        """Rule 3's reserve is a COUNT over the layout, never a rule of thumb."""
        side = _layout("D", 0, "w", [PAGE - 16, 32, PAGE])
        # piece 0 ends inside page 0; piece 1 straddles pages 0 and 1; so page
        # 0 carries a boundary and page 1 carries the boundary with piece 2.
        self.assertEqual(side.boundary_pages(PAGE), 2)

    def test_a_layout_cut_on_page_lines_has_no_boundary_page(self):
        side = _layout("D", 0, "w", [PAGE, PAGE, PAGE])
        self.assertEqual(side.boundary_pages(PAGE), 0)


class MeasuredGeometryTest(CustomTestCase):
    """The count-check, at the three cards the rig actually has."""

    def test_the_fund_is_the_measured_page_difference_per_card(self):
        """Rule 4, per card, in the unit the driver works in."""
        # the pp_to_tp term is max(0, D - P): on nvml2 it is ZERO because P is
        # the bigger image there, and the 1108-page difference is owed on the
        # MIRROR leg instead -- which is why the fund is sized over both.
        expected = {1: 9304 - 8210, 0: 3568 - 2855, 2: 0}
        for card, (p_pages, d_pages) in MEASURED_PAGES.items():
            src = _sized("P", card, "weights_p", p_pages)
            dst = _sized("D", card, "weights_d", d_pages)
            # the size term alone, boundary reserve stripped, so the assertion
            # is about rule 4 and not about rule 3 riding along with it
            size_term = pp.fund_pages(src, dst, PAGE) - dst.boundary_pages(PAGE)
            self.assertEqual(size_term, max(0, expected[card]),
                             f"card {card}: P={p_pages} D={d_pages} pages")

    def test_one_fund_per_card_covers_both_legs(self):
        """The fund is allocated ONCE at boot, so it is the max over legs.

        On nvml2 the P image is the BIGGER one (4676 > 3568), so a fund sized
        from the pp_to_tp leg alone would be zero there and short by 1108 pages
        on the mirror -- exactly the #1275 shape.
        """
        src = _sized("P", 2, "weights_p", 4676)
        dst = _sized("D", 2, "weights_d", 3568)
        forward = pp.fund_pages(src, dst, PAGE) - dst.boundary_pages(PAGE)
        mirror = pp.fund_pages(dst, src, PAGE) - src.boundary_pages(PAGE)
        self.assertEqual(forward, 0)
        self.assertEqual(mirror, 4676 - 3568)
        self.assertGreaterEqual(pp.fund_for_card(src, dst, PAGE), mirror)

    def test_the_measured_page_totals_are_consistent_with_the_bytes(self):
        """The census is self-consistent: 2 MiB x pages == the printed bytes."""
        for card, (p_pages, d_pages) in MEASURED_PAGES.items():
            self.assertEqual(p_pages * 2, {1: 16420, 0: 5710, 2: 9352}[card])
            self.assertEqual(d_pages * 2, {1: 18608, 0: 7136, 2: 7136}[card])

    def test_both_legs_schedule_at_the_real_cross_card_shape(self):
        """The whole leg, both directions, on the PP-against-TP run shape."""
        slot = 128 * MIB                    # one 128 MiB host slot = 64 pages
        srcs, dsts = _leg_layouts()
        for direction, a, b in ((pp.PP_TO_TP, srcs, dsts),
                                (pp.TP_TO_PP, dsts, srcs)):
            funds = pp.minimum_fund(a, b, direction=direction, slot_bytes=slot,
                                    page_bytes=PAGE)
            plans = pp.plan_leg(a, b, direction=direction, funds=funds,
                                slot_bytes=slot, page_bytes=PAGE)
            self.assertEqual(sorted(plans), [0, 1, 2])
            pp.verify_leg(plans)
            for card, plan in plans.items():
                self.assertLessEqual(plan.host_slot_peak, slot // PAGE,
                                     f"card {card} {direction}: slot overrun")
                self.assertGreater(plan.pages_moved, 0,
                                   f"card {card} {direction}: nothing remapped, "
                                   "so the flip would still be allocating")
                self.assertIn("pages_moved=", plan.line())

    def test_a_source_page_read_by_two_cards_is_not_released_early(self):
        """Rule 1's cross-card half, on the shape that has it.

        Under PP->TP every P card feeds all three D cards, so the leg planner
        must hold a page past its OWN card's last read.  A per-card planner
        cannot see that, which is why :func:`plan_leg` exists; here the
        independent verifier is what confirms it.
        """
        slot = 64 * MIB
        srcs, dsts = _leg_layouts()
        funds = pp.minimum_fund(srcs, dsts, direction=pp.PP_TO_TP,
                                slot_bytes=slot, page_bytes=PAGE)
        plans = pp.plan_leg(srcs, dsts, direction=pp.PP_TO_TP, funds=funds,
                            slot_bytes=slot, page_bytes=PAGE)
        pp.verify_leg(plans)

    def test_the_minimum_fund_exceeds_the_net_size_difference(self):
        """The finding: the schedule needs more than the images differ by.

        The net difference is what the two images differ by at the END; the
        stream needs backed pages at EVERY point, and the transfers that fund
        them lag the collects that consume them.  Sizing the boot reservation
        on the net difference is short by that lead -- intermittently, which
        reads as a flaky flip rather than as a sizing error.
        """
        slot = 128 * MIB
        srcs, dsts = _leg_layouts()
        funds = pp.minimum_fund(srcs, dsts, direction=pp.PP_TO_TP,
                                slot_bytes=slot, page_bytes=PAGE)
        for dst in dsts:
            src = next(s for s in srcs if s.card == dst.card)
            net = max(0, dst.total_pages(PAGE) - src.total_pages(PAGE))
            self.assertGreaterEqual(funds[dst.card], net)


class DeadlockFreedomTest(CustomTestCase):
    def test_a_fund_that_is_too_small_is_a_named_refusal_not_a_hang(self):
        srcs, dsts = _leg_layouts()
        with self.assertRaises(pp.Weg2RemapPlanUnschedulable) as ctx:
            pp.plan_leg(srcs, dsts, direction=pp.PP_TO_TP,
                        funds={0: 0, 1: 0, 2: 0}, slot_bytes=128 * MIB,
                        page_bytes=PAGE)
        msg = str(ctx.exception)
        self.assertIn("W94", msg)
        self.assertIn("Short by", msg)
        self.assertIn("a fund of", msg, "the refusal must say what would fix it")
        self.assertIn("a bigger slot", msg, "it must say which knob is NOT short")

    def test_a_slot_below_one_page_is_refused_before_any_window(self):
        srcs, dsts = _leg_layouts()
        with self.assertRaises(pp.Weg2RemapPlanUnschedulable):
            pp.plan_leg(srcs, dsts, direction=pp.PP_TO_TP,
                        funds={0: 10_000, 1: 10_000, 2: 10_000},
                        slot_bytes=PAGE // 2, page_bytes=PAGE)

    def test_a_destination_page_fed_by_more_pages_than_the_slot_holds(self):
        """The one shape no window size rescues -- and it says SLOT, not plan."""
        # 40 eight-byte runs, each alone on its own SOURCE page, all packed
        # into a single DESTINATION page: one destination page, 40 feeders.
        src = pp.SideLayout(
            group="P", card=0, tag="p", modelled=False,
            pieces=tuple(pp.PieceExtent(f"t{i}", 0, "p", offset=i * PAGE, nbytes=8)
                         for i in range(40)),
        )
        dst = pp.SideLayout(
            group="D", card=0, tag="d", modelled=False,
            pieces=tuple(pp.PieceExtent(f"t{i}", 0, "d", offset=i * 8, nbytes=8)
                         for i in range(40)),
        )
        with self.assertRaises(pp.Weg2RemapPlanUnschedulable) as ctx:
            pp.plan_pages(src, dst, direction=pp.PP_TO_TP, fund=100,
                          slot_bytes=8 * PAGE, page_bytes=PAGE)
        self.assertIn("the SLOT is short", str(ctx.exception))


class MutantTest(CustomTestCase):
    """THE NAMED DANGER DIRECTION: a page moves before its last reader.

    The mutant is applied to the PLAN, not to the planner, so the independent
    verifier is what has to catch it -- which is the only way to know the
    verifier is load-bearing rather than a restatement of the scheduler.
    """

    def _real_leg(self):
        srcs, dsts = _leg_layouts()
        funds = pp.minimum_fund(srcs, dsts, direction=pp.PP_TO_TP,
                                slot_bytes=128 * MIB, page_bytes=PAGE)
        return pp.plan_leg(srcs, dsts, direction=pp.PP_TO_TP, funds=funds,
                           slot_bytes=128 * MIB, page_bytes=PAGE)

    def _real_plan(self):
        return self._real_leg()[1]          # the 5090

    def test_the_honest_plan_passes_the_verifier(self):
        pp.verify_leg(self._real_leg())

    def test_mutant_a_page_transferred_before_its_last_reader_goes_red(self):
        plan = self._real_plan()
        steps = list(plan.steps)
        last = [s for s in steps if s.kind == pp.STEP_TRANSFER][-1]
        mutant = [pp.Step(kind=pp.STEP_TRANSFER, window=0, src_page=last.src_page,
                          dst_page=last.dst_page, src_card=last.src_card)] + steps
        leg = dict(self._real_leg())
        leg[1] = pp.PagePlan(**{**plan.__dict__, "steps": tuple(mutant)})
        with self.assertRaises(pp.Weg2RemapPageRefused) as ctx:
            pp.verify_leg(leg)
        self.assertIn("before any card", str(ctx.exception))

    def test_mutant_a_collect_into_an_unbacked_page_goes_red(self):
        plan = self._real_plan()
        mutant = [pp.Step(kind=pp.STEP_COLLECT, window=0,
                          dst_page=plan.pages_moved + plan.fund_pages + 1)]
        bad = pp.PagePlan(**{**plan.__dict__, "steps": tuple(mutant)})
        with self.assertRaises(pp.Weg2RemapPageRefused) as ctx:
            pp.verify_collect_invariant(bad)
        self.assertIn("no map_fund or transfer has backed", str(ctx.exception))

    def test_mutant_a_deposit_after_the_transfer_goes_red(self):
        """Rule 1 from the other side: reading a page that is already gone."""
        plan = self._real_plan()
        first = next(s for s in plan.steps if s.kind == pp.STEP_TRANSFER)
        steps = list(plan.steps) + [
            pp.Step(kind=pp.STEP_DEPOSIT, window=first.window + 1,
                    src_page=first.src_page, src_card=first.src_card)
        ]
        leg = dict(self._real_leg())
        leg[1] = pp.PagePlan(**{**plan.__dict__, "steps": tuple(steps)})
        with self.assertRaises(pp.Weg2RemapPageRefused) as ctx:
            pp.verify_leg(leg)
        self.assertIn("already transferred away", str(ctx.exception))

    def test_mutant_an_unknown_step_kind_is_refused_not_ignored(self):
        plan = self._real_plan()
        bad = pp.PagePlan(**{**plan.__dict__,
                             "steps": (pp.Step(kind="prefetch", window=0),)})
        with self.assertRaises(pp.Weg2RemapPageRefused):
            pp.verify_collect_invariant(bad)


class ModelledOffsetsTest(CustomTestCase):
    """A plan on modelled placement is refused in production, by name."""

    class _Piece:
        def __init__(self, name, nbytes):
            self.param_name = name
            self.nbytes = nbytes

    def test_a_modelled_layout_is_marked_as_modelled(self):
        side = pp.extents_from_manifest_order(
            [self._Piece("a", PAGE), self._Piece("b", PAGE)],
            group="P", card=0, tag="w",
        )
        self.assertTrue(side.modelled)
        self.assertEqual(side.total_pages(PAGE), 2)

    def test_a_production_plan_refuses_modelled_offsets(self):
        a = pp.extents_from_manifest_order(
            [self._Piece("a", PAGE), self._Piece("b", PAGE)], group="P", card=0, tag="p")
        b = pp.extents_from_manifest_order(
            [self._Piece("a", PAGE), self._Piece("b", PAGE)], group="D", card=0, tag="d")
        with self.assertRaises(pp.Weg2RemapPageRefused) as ctx:
            pp.plan_pages(a, b, direction=pp.PP_TO_TP, fund=8, slot_bytes=8 * PAGE)
        self.assertIn("MODELLED byte offsets", str(ctx.exception))
        # and it goes through when the caller says it is desk arithmetic
        plan = pp.plan_pages(a, b, direction=pp.PP_TO_TP, fund=8,
                             slot_bytes=8 * PAGE, allow_modelled_offsets=True)
        self.assertTrue(plan.modelled_offsets)
        self.assertIn("offsets=MODELLED", plan.line())

    def test_a_piece_without_nbytes_is_refused_not_treated_as_zero(self):
        class Bare:
            param_name = "x"

        with self.assertRaises(pp.Weg2RemapPageRefused):
            pp.extents_from_manifest_order([Bare()], group="P", card=0, tag="w")


class SourcelessDestinationTest(CustomTestCase):
    def test_a_destination_piece_with_no_source_is_refused(self):
        src = _layout("P", 0, "p", [PAGE])
        dst = pp.SideLayout(group="D", card=0, tag="d", modelled=False,
                            pieces=(pp.PieceExtent("nobody", 0, "d", 0, PAGE),))
        with self.assertRaises(pp.Weg2RemapPageRefused) as ctx:
            pp.plan_pages(src, dst, direction=pp.PP_TO_TP, fund=4,
                          slot_bytes=8 * PAGE, page_bytes=PAGE)
        self.assertIn("Assembly does not create a source", str(ctx.exception))

    def test_an_unknown_direction_is_refused_rather_than_planned(self):
        src = _sized("P", 0, "p", 10, n_runs=4)
        dst = _sized("D", 0, "d", 10, n_runs=4)
        with self.assertRaises(pp.Weg2RemapPageRefused):
            pp.plan_pages(src, dst, direction="sideways", fund=4,
                          slot_bytes=8 * PAGE, page_bytes=PAGE)


class Cut1CostTest(CustomTestCase):
    def test_cut1_prices_the_oncard_bytes_at_the_measured_rate(self):
        """~10.3 GiB on-card, twice across PCIe, at the measured 13-14 GB/s."""
        oncard = int(10.28 * 1024) * MIB
        lo = pp.cut1_cost_ms(oncard, 14.0)
        hi = pp.cut1_cost_ms(oncard, 13.0)
        self.assertLess(lo, hi)
        # the band this rig's own WEG2-FLIP-TAG lines imply, in seconds
        self.assertGreater(lo / 1000.0, 1.4)
        self.assertLess(hi / 1000.0, 1.8)

    def test_a_zero_rate_is_refused_rather_than_dividing_by_zero(self):
        with self.assertRaises(pp.Weg2RemapPageRefused):
            pp.cut1_cost_ms(1 << 30, 0.0)


class InstrumentProvenanceTest(CustomTestCase):
    """WHY THIS FILE QUOTES THE FLIP-TAG CENSUS AND NOT THE RING TABLE.

    The ring table's ``image_D``/``image_P`` per card (21796/22782 on the 5090)
    is a HOST measurement -- the sleeping group's netted RssShmem, apportioned
    over the cards by their share of a census -- and its per-card ``tags`` term
    is DIRECTION-IDENTICAL (21712 on both sides of the 5090's line).  A
    quantity that is the same for both groups cannot be the per-group weight
    bytes under PP 39/13/12 against TP 17/7/8, so it cannot size a per-card,
    per-direction page fund at all.  The device-side census can, and does:
    16420 MiB of P against 18608 MiB of D on the same card.

    This test does not assert about the ring table -- it pins the DIFFERENCE,
    so that anyone who later swaps the instrument sees the sign flip.
    """

    RING_TABLE_MIB = {1: (22782, 21796), 0: (8661, 8286), 2: (13144, 12575)}

    def test_the_two_instruments_disagree_in_sign_on_the_5090(self):
        p_pages, d_pages = MEASURED_PAGES[1]
        device_diff_mib = (d_pages - p_pages) * 2
        ring_p, ring_d = self.RING_TABLE_MIB[1]
        ring_diff_mib = ring_d - ring_p
        self.assertGreater(device_diff_mib, 0, "device census: D is the bigger image")
        self.assertLess(ring_diff_mib, 0, "ring table: P is the bigger image")
        self.assertEqual(device_diff_mib, 2188)
        self.assertEqual(ring_diff_mib, -986)


if __name__ == "__main__":
    unittest.main()


class ThreeRoutesTest(CustomTestCase):
    """User ruling 2026-09-12: an unexchangeable class is NOT a refusal.

    It keeps today's path -- sleep writes it to the host ring, wake reads it
    back -- and the ring is sized from the manifest to exactly those bytes.
    The ring shrinks; it does not disappear.
    """

    def _routes(self, residual_names=()):
        def route_of(name):
            if name in residual_names:
                return pp.ROUTE_RESIDUAL_RING
            return pp.ROUTE_EXCHANGE
        return route_of

    def test_an_all_transferable_form_gets_a_ring_of_zero(self):
        """This checkpoint's own case: I8 + BF16, no repack class."""
        srcs, _ = _leg_layouts()
        rings = pp.residual_ring_bytes(srcs, self._routes())
        self.assertEqual(set(rings.values()), {0})

    def test_the_ring_is_sized_to_exactly_the_residual_class(self):
        srcs, _ = _leg_layouts()
        names = [srcs[0].pieces[0].param_name, srcs[0].pieces[1].param_name]
        want = sum(p.nbytes for p in srcs[0].pieces if p.param_name in names)
        rings = pp.residual_ring_bytes(srcs, self._routes(tuple(names)))
        self.assertEqual(rings[srcs[0].card], want)
        self.assertGreater(want, 0)

    def test_a_residual_run_is_dropped_from_the_plan_not_refused(self):
        """The whole point of the correction: no class-level refusal."""
        srcs, dsts = _leg_layouts()
        names = tuple(p.param_name for p in srcs[0].pieces[:4])
        route_of = self._routes(names)
        a = tuple(pp.exchangeable_only(s, route_of) for s in srcs)
        b = tuple(pp.exchangeable_only(d, route_of) for d in dsts)
        funds = pp.minimum_fund(a, b, direction=pp.PP_TO_TP,
                                slot_bytes=128 * MIB, page_bytes=PAGE)
        plans = pp.plan_leg(a, b, direction=pp.PP_TO_TP, funds=funds,
                            slot_bytes=128 * MIB, page_bytes=PAGE)
        pp.verify_leg(plans)
        for name in names:
            for side in a:
                self.assertNotIn(name, {p.param_name for p in side.pieces})

    def test_dropping_a_residual_run_does_not_compact_the_arena(self):
        """The pages a (c) run occupies stay occupied.

        A layout that closed the gap would hand the planner an arena that does
        not exist and every page index after the gap would name another page.
        """
        srcs, _ = _leg_layouts()
        side = srcs[0]
        route_of = self._routes((side.pieces[0].param_name,))
        thinned = pp.exchangeable_only(side, route_of)
        self.assertEqual(thinned.pieces[0].offset, side.pieces[1].offset)
        self.assertEqual(thinned.total_bytes(), side.total_bytes())

    def test_an_unrecognised_route_string_is_a_caller_defect(self):
        srcs, _ = _leg_layouts()
        with self.assertRaises(pp.Weg2RemapPageRefused) as ctx:
            pp.residual_ring_bytes(srcs, lambda name: "maybe")
        self.assertIn("unrecognised route string", str(ctx.exception))
