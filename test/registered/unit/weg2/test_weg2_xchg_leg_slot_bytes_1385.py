# SPDX-License-Identifier: Apache-2.0
"""#1385/xsn31-4 -- the ALLOCATOR must build the buffer the LEDGER priced.

ROOT CAUSE, boot weg2xsn31/4 (BOOT_weg2xsn31_0913_4.md), two independent
instruments deckungsgleich (a 0.2 s filesystem poller and the lane's own
WEG2-XCHG-HOST-SLOT lines): with `--xchg-lanes-concurrent 2` on a 5-lane
cut, TWO files existed simultaneously (the cap DOES work: 2, not 5), but
EACH was 18,151,770,624 B (16.905 GiB) against a priced 3,221,225,472 B
(3.00 GiB) -- 5.6x over, per lane. Cushion latched (0.19 GiB < 1.50 GiB
floor) 4 seconds after flip start.

THE ARITHMETIC THAT FINDS IT: 756,323,776 (this checkpoint's WIDEST LAYER,
quoted throughout this module's own history) x 24 (`BounceTerms.lane_slots`
for `max_tag_bytes=2907 MiB`, `slot_bytes=128 MiB`) = 18,151,770,624 --
EXACT, to the byte, against the measured file size. `weight_exchange_bounce.
run_bounce_leg` was reading `leg_geometry(terms)` (the PRE-#1374 AMENDMENT-2
pair: one band = one WHOLE LAYER) for the byte size of ONE band, while
SEPARATELY reading `terms.lane_slots` (Option 1's band COUNT, sized
assuming `terms.slot_bytes`-wide bands) for the band count -- two
mechanisms from the SAME #1374 "Option 1" day, never reconciled: the count
changed when Option 1 landed, the per-band SIZE it multiplies did not.

THE DIRECTION (coordinator order, verified twice against fresh metal --
xsn31/4 with concurrent=2, xsn31/5 with concurrent=1): "Allokator folgt der
Preisformel". xsn31/5 confirmed the cap itself is flawless (the 0.2 s
poller never saw more than ONE file all boot, at concurrent=1) but the
per-file size was UNCHANGED (still 16.905 GiB, because that boot ran on the
allocator this file fixes, not on the fix) -- so re-pricing the ledger on
the real, uneven-PP-stage tag size (the other direction) was never a live
option: it would make the ARM number honest and the boot impossible (P's
`--pp-stage-ratio 39,13,12` makes some tags carry many multiples of one
layer). This file is the allocator-follows-the-price fix and its guard.

WHOLE-TAG-IN-ONE-BUFFER IS NOT THE SAME CLAIM AS
WHOLE-TAG-IN-ONE-CONTIGUOUS-BAND, and confusing them is the whole defect.
Option 1's own promise -- "a deposit completes without its collector" -- is
about the BUFFER'S TOTAL CAPACITY (`terms.lane_slots` bands must cover the
whole tag so nothing wraps mid-tag), never about writing the tag as one
giant blob. `weight_exchange_transport.batch_descs` already slices any
FLAT/STRIDED2D unit across as many slot-sized bands as it takes (this is
what already lets `lm_head`, 4x a layer, band across the pre-#1374 slot
size without complaint) -- there was never a technical need for a
band the size of a whole layer, let alone a whole tag. The execution test
below moves real bytes end to end through 512-byte bands and gets every
row right, which is the answer to the question this file's docstring
promised to answer honestly: the layered form is not a compromise.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as bx
from sglang.srt.weg2 import xchg_bounce as xb
from sglang.test.test_utils import CustomTestCase

from .test_weg2_xchg_bounce_execution_smoke_1273 import (
    DEPTH,
    LAYER0_BYTES,
    N_LAYERS,
    PLAIN_LAYER_BYTES,
    _all_descs,
    _manager,
    _mismatched_rows,
    _seed_source,
)
from .test_weg2_xchg_transport_1273 import FakeDeviceOps, _fresh_boot

MIB = xb.MIB
GIB = xb.GIB

# ---------------------------------------------------------------------------
# THE MEASURED FIXTURE -- xsn31/4's own numbers, verbatim.
# ---------------------------------------------------------------------------
WIDEST_LAYER_BYTES = 756323776          # this checkpoint's widest layer (721.3 MiB)
ONCARD_SLOT_BYTES = 128 * MIB           # --weg2-xchg-oncard-slot-mib default
MAX_TAG_BYTES = 2907 * MIB              # measured: this checkpoint's biggest tag
MEASURED_FILE_BYTES = 18_151_770_624    # measured, 2 independent instruments
PRICED_LANE_BYTES = 3_221_225_472       # 128 MiB x 24, what the ARM charged


def _xsn31_terms(**kw):
    base = dict(
        bytes_per_direction=29119878266, n_layers=64,
        widest_layer_bytes=WIDEST_LAYER_BYTES, pairs=3, depth=2,
        slot_bytes=ONCARD_SLOT_BYTES, n_lanes=5, max_tag_bytes=MAX_TAG_BYTES,
    )
    base.update(kw)
    return xb.bounce_terms(**base)


class TheMeasuredFixturePinsTheIncident(CustomTestCase):
    """Permanent regression anchor: WHY the bug produced exactly this
    number, independent of whether the fix is present."""

    def test_widest_layer_times_lane_slots_is_the_measured_bug_value(self):
        terms = _xsn31_terms()
        self.assertEqual(terms.lane_slots, 24)
        self.assertEqual(WIDEST_LAYER_BYTES * terms.lane_slots,
                         MEASURED_FILE_BYTES)

    def test_slot_bytes_times_lane_slots_is_the_priced_value(self):
        terms = _xsn31_terms()
        self.assertEqual(int(terms.slot_bytes) * terms.lane_slots,
                         PRICED_LANE_BYTES)
        self.assertEqual(terms.lane_buffer_bytes, PRICED_LANE_BYTES)

    def test_the_two_numbers_are_5x_apart_not_a_rounding_difference(self):
        ratio = MEASURED_FILE_BYTES / PRICED_LANE_BYTES
        self.assertGreater(ratio, 5.0)
        self.assertLess(ratio, 6.0)


class TheFixMatchesThePriceNotTheBug(CustomTestCase):
    """`leg_slot_bytes` is the ONE producer `run_bounce_leg` reads for the
    per-band size under Option 1 -- THE fix."""

    def test_leg_slot_bytes_is_the_oncard_slot_not_the_widest_layer(self):
        terms = _xsn31_terms()
        self.assertEqual(bx.leg_slot_bytes(terms), ONCARD_SLOT_BYTES)
        self.assertNotEqual(bx.leg_slot_bytes(terms), WIDEST_LAYER_BYTES)

    def test_leg_slot_bytes_times_lane_slots_equals_the_priced_total(self):
        terms = _xsn31_terms()
        allocated = bx.leg_slot_bytes(terms) * terms.lane_slots
        self.assertEqual(allocated, PRICED_LANE_BYTES)
        self.assertEqual(allocated, terms.lane_buffer_bytes)
        self.assertNotEqual(allocated, MEASURED_FILE_BYTES,
                            "the fix must no longer reproduce the bug")

    def test_leg_geometry_itself_is_UNCHANGED_pre_option1_stays_byte_identical(self):
        """`leg_geometry` is the PRE-#1374 function and #1332/execution-smoke
        already pin its contract when `max_tag_bytes` is unset; this fix must
        not touch that path. Built WITHOUT `max_tag_bytes` on purpose."""
        terms = xb.bounce_terms(
            bytes_per_direction=PLAIN_LAYER_BYTES * N_LAYERS, n_layers=N_LAYERS,
            widest_layer_bytes=LAYER0_BYTES, pairs=6, depth=DEPTH,
            slot_bytes=xb.SLOT_BYTES_DEFAULT,
        )
        self.assertEqual(bx.leg_geometry(terms), (LAYER0_BYTES, DEPTH))
        self.assertEqual(bx.leg_slot_bytes(terms), LAYER0_BYTES,
                         "Option 1 absent (max_tag_bytes=0): byte-identical "
                         "to leg_geometry, exactly as before this fix")


class TheDangerDirectionMutantGuard(CustomTestCase):
    """M: price SMALLER than allocation -- exactly the state that cost
    boot weg2xsn31/4 its cushion. Manually verified (see the commit message)
    by reverting `leg_slot_bytes` to always return `leg_geometry(terms)[0]`
    and confirming this class fails; restored after, diff empty."""

    def test_the_old_widest_layer_path_would_price_less_than_it_allocates(self):
        """The OLD (buggy) computation, named explicitly rather than
        reproduced by calling the fixed function: if `run_bounce_leg` used
        `leg_geometry(terms)[0]` (the pre-#1374 value) as the per-band size
        while still using Option 1's `lane_slots` as the band COUNT, the
        real allocation would exceed the ledger's own priced total -- a
        ledger funding less host memory than the runtime actually pins,
        which is the #1358 defect reproduced in the other direction."""
        terms = _xsn31_terms()
        old_buggy_allocation = int(leg_geometry_old_slot(terms)) * terms.lane_slots
        priced = terms.lane_buffer_bytes
        self.assertGreater(
            old_buggy_allocation, priced,
            "the danger direction: allocation must never exceed what was "
            "priced, and the pre-fix code did, by construction")
        self.assertEqual(old_buggy_allocation, MEASURED_FILE_BYTES)

    def test_the_fixed_function_never_allocates_more_than_priced(self):
        for cap in (1, 2, 5, 10):
            with self.subTest(lanes_concurrent=cap):
                terms = _xsn31_terms(lanes_concurrent=cap)
                allocated_per_lane = bx.leg_slot_bytes(terms) * terms.lane_slots
                self.assertLessEqual(allocated_per_lane, terms.lane_buffer_bytes)
                self.assertEqual(allocated_per_lane, terms.lane_buffer_bytes)


def leg_geometry_old_slot(terms):
    """The PRE-FIX value, named as a function so the mutant test above reads
    as 'the old formula' rather than re-deriving `leg_geometry`'s own
    result inline -- kept local to this test file, never imported by
    product code, so a future reader cannot mistake it for a second
    producer."""
    return bx.leg_geometry(terms)[0]


# ---------------------------------------------------------------------------
# STRUCTURAL: run_bounce_leg must actually READ leg_slot_bytes, and the
# AMENDMENT-2 refusal must be skipped -- not removed -- under Option 1.
# ---------------------------------------------------------------------------


class RunBounceLegIsWiredToTheFix(CustomTestCase):
    def test_run_bounce_leg_derives_slot_bytes_from_leg_slot_bytes(self):
        src = inspect.getsource(bx.run_bounce_leg)
        self.assertIn("leg_slot_bytes(terms)", src)
        self.assertNotIn("derived_slot, derived_depth = leg_geometry(terms)",
                         src, "the old unconditional call must be gone")

    def test_the_option1_boolean_is_computed_once_and_reused(self):
        src = inspect.getsource(bx.run_bounce_leg)
        self.assertEqual(src.count("_option1_leg = ("), 1,
                         "one producer of the predicate, reused everywhere")
        # Reused at the refusal skip AND the slot-count selection -- three
        # readers of one boolean, never a second computation of it.
        self.assertIn("if not _option1_leg:", src)
        self.assertIn("if _option1_leg", src)
        self.assertEqual(
            src.count("getattr(terms, \"max_tag_bytes\", 0) or 0) > 0"), 1,
            "the raw predicate must be spelled out exactly once -- every "
            "other reader takes `_option1_leg`, or a second spelling could "
            "drift from the first the way the slot-size/count split did")

    def test_the_amendment_2_refusal_is_conditioned_on_option1(self):
        src = inspect.getsource(bx.run_bounce_leg)
        i = src.index("if not _option1_leg:")
        j = src.index("refuse_if_plan_exceeds_slot(slot_bytes, descs, terms)", i)
        self.assertLess(i, j)
        self.assertLess(j - i, 1500,
                        "the refusal must be the very next thing guarded "
                        "(a verbose comment block may sit between the `if` "
                        "and the call, but nothing else), not an unrelated "
                        "later call that happens to match")

    def test_refuse_if_slot_short_still_runs_unconditionally(self):
        """Only the AMENDMENT-2 (layer-in-one-slot) check is Option-1-
        specific; the widest-INDIVISIBLE-ROW check must still run always --
        that one is about `batch_descs`' own hard floor, unrelated to which
        geometry chose `slot_bytes`."""
        src = inspect.getsource(bx.run_bounce_leg)
        i = src.index("refuse_if_slot_short(slot_bytes, descs)")
        j = src.index("if not _option1_leg:")
        self.assertLess(i, j, "refuse_if_slot_short must precede the "
                        "Option-1-conditioned Amendment-2 check, and run "
                        "regardless of the branch")


class ExistingAmendment2CoverageIsUnaffected(CustomTestCase):
    """The pre-#1374 (Option-1-absent) path's own refusal tests
    (test_weg2_xchg_bounce_1332.py, test_weg2_xchg_bounce_execution_
    smoke_1273.py) must keep passing byte-identically -- checked here as an
    explicit smoke rather than trusted from the neighbour suites alone."""

    def test_a_layer_wider_than_the_slot_still_refuses_without_max_tag_bytes(self):
        """Direct call to `refuse_if_plan_exceeds_slot`, matching this
        module's own existing test style (test_weg2_xchg_bounce_1332.py) --
        no `max_tag_bytes`, so this is the Option-1-ABSENT path this fix
        must leave untouched."""
        terms = xb.bounce_terms(
            bytes_per_direction=1000, n_layers=2,
            widest_layer_bytes=1000, pairs=1, depth=1, slot_bytes=1,
        )
        descs = [wx.XchgDesc(
            tag="t", src_rank=0, dst_rank=0, param_name="model.layers.0.w",
            kind=wx.FLAT, nbytes=1000, rows=1, run_bytes=1000, spitch=0,
            dpitch=0,
        )]
        with self.assertRaises(xb.Weg2XchgBounceUnderCovered):
            bx.refuse_if_plan_exceeds_slot(1, descs, terms)


# ---------------------------------------------------------------------------
# EXECUTION SMOKE: real bytes, real bands, small scale (no gigabyte
# allocations in a hermetic test) -- proves the fix end to end through the
# PRODUCT call site, not only the pure function.
# ---------------------------------------------------------------------------

#: Small-scale Option-1 geometry: an oncard slot far smaller than one layer
#: (the same STRUCTURAL relationship as production: 128 MiB slot < 721 MiB
#: layer), and a tag spanning two layers, so more than one band is required
#: -- the shape that actually exercises banding, not a single-band fluke.
SMALL_SLOT_BYTES = 512
SMALL_MAX_TAG_BYTES = LAYER0_BYTES * 2


class TheFixMovesRealBytesThroughSmallBands(CustomTestCase):
    def _run(self, **term_kw):
        terms = xb.bounce_terms(
            bytes_per_direction=PLAIN_LAYER_BYTES * N_LAYERS, n_layers=N_LAYERS,
            widest_layer_bytes=LAYER0_BYTES, pairs=6, depth=DEPTH,
            slot_bytes=SMALL_SLOT_BYTES, max_tag_bytes=SMALL_MAX_TAG_BYTES,
            **term_kw,
        )
        d = tempfile.mkdtemp()
        boot = _fresh_boot()
        ops = FakeDeviceOps(d, 0)
        _seed_source(ops)
        result = _manager()._weg2_xchg_bounce_leg(
            descs=_all_descs(), ops=ops, boot_nonce=boot, terms=terms,
            shm_root=d, mode=wx.INJECT_AUTHORITATIVE,
        )
        return terms, result, ops

    def test_the_allocated_peak_matches_the_priced_lane_buffer(self):
        terms, result, ops = self._run()
        self.assertEqual(result.slot_bytes, SMALL_SLOT_BYTES)
        self.assertEqual(result.host_bytes_peak, terms.lane_buffer_bytes)
        old_buggy = LAYER0_BYTES * terms.lane_slots
        self.assertNotEqual(result.host_bytes_peak, old_buggy)
        self.assertLess(result.host_bytes_peak, old_buggy,
                        "the fix must allocate LESS than the pre-fix bug, "
                        "on the same terms")

    def test_every_destination_row_is_still_exactly_right(self):
        """Not just smaller -- still CORRECT: real bytes move through many
        512-byte bands (not one band the size of a layer) and land exactly
        where the plan says, proving the layered form loses nothing."""
        _terms, result, ops = self._run()
        self.assertEqual(result.verdict, "MATCH")
        self.assertEqual(_mismatched_rows(ops, _all_descs()), [])
        self.assertGreater(result.bands, 1,
                           "the fixture must actually exercise banding, or "
                           "this test cannot see a slot-addressing regression")


if __name__ == "__main__":
    unittest.main()
