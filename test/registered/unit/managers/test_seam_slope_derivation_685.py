"""#685: derive the per-rank seam slope instead of carrying a frozen triple.

WHAT THE SLOPE IS. `phase_flip_runtime._staging_bytes` (:4403) peaks at
``incoming + max(outgoing, local)``, and the incoming leg (:4487-4497) is
``dst.row_nbytes(layer) * rows`` summed over ``tr.recv_layers[peer]``. So a
rank's per-ROW seam slope is exactly the number of full-attention layers it
must RECEIVE at a pp->tp cutover, times the KV cell.

HOW MANY LAYERS THAT IS::

    received_r = max(0, tp_share_r * n_attention_total - attention_held_r)

On the reference rig (``--phase-flip-tp-vector 32,16,16``, cut ``28,20,16``,
attention ``[7,5,4]``) that is +1 / -1 / 0 layers, which is why the measured
slopes are 2360.1 / 424.1 / 547.6 B/token: rank 0 receives one layer's KV per
row and the other two receive none, so their whole slope is baseline. The 5.6x
is one received layer against a baseline, not a rank-0 pathology.

WHY DERIVING IT MATTERS. The slope is a FUNCTION OF THE CUT and moves in WHOLE
LAYER steps. A frozen triple is valid only for the cut and flip vector it was
measured at; carried across a cut that moves an attention layer it misprices
by whole KV cells, in either direction. ``TheFrozenTripleMispricesAMovedCut``
below is that failure, planted.

SCOPE, STATED SO THIS IS NOT MISREAD AS A LIVE-PATH CHANGE. This module is a
pure function. It does NOT touch ``SHIP_PIN.basis_per_row_bytes``: that field
is a frozen PROVENANCE record whose whole job is to stay fixed so
``test_arming_floor_funding_662`` can detect drift against it, and the live
sizing path does not read it. See the report accompanying this commit for
where the derived vector belongs in the live path and what it needs first.
"""

import unittest

from sglang.srt.managers import phase_flip_seam_reserve as sr
from sglang.srt.managers.seam_slope import derive_seam_slope_bytes_per_token
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

#: The live flagset's flip weight vector (--phase-flip-tp-vector 32,16,16).
FLIP_VECTOR = (32, 16, 16)
#: Attention layers per stage at the shipping cut 28,20,16 on a 64-layer,
#: period-4 checkpoint.
SHIPPING_ATTENTION = (7, 5, 4)
N_ATTENTION = 16
#: KV bytes per token per full-attention layer, from the recorder's kv posts.
KV_CELL_BYTES = 2326.7

#: The residual every rank pays regardless of what it receives: checksums, the
#: one-layer streaming window, allocator grain. Read off the two ranks that
#: receive nothing, plus rank 0's remainder.
MEASURED_BASELINE = (33.4, 424.1, 547.6)


class TheDerivationReproducesTheMeasuredSlopes(unittest.TestCase):
    """Behaviour-neutrality at the CURRENT cut and vector.

    Tolerance is the measurement error the decomposition was established with:
    rank 0 predicted 2326.7 B/token against 2360.1 measured, 1.4 %.
    """

    TOLERANCE = 0.02

    def test_rank0_matches_the_frozen_basis_within_the_measured_error(self):
        got = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR, SHIPPING_ATTENTION, KV_CELL_BYTES, N_ATTENTION
        )
        frozen = sr.SHIP_PIN.basis_per_row_bytes
        self.assertAlmostEqual(got[0] / frozen[0], 1.0, delta=self.TOLERANCE)

    def test_the_full_vector_matches_once_the_baselines_are_supplied(self):
        got = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR,
            SHIPPING_ATTENTION,
            KV_CELL_BYTES,
            N_ATTENTION,
            MEASURED_BASELINE,
        )
        for derived, frozen in zip(got, sr.SHIP_PIN.basis_per_row_bytes):
            self.assertAlmostEqual(derived / frozen, 1.0, delta=self.TOLERANCE)

    def test_the_two_receiving_nothing_are_pure_baseline(self):
        """The substance of the 5.6x: ranks 1 and 2 receive no layer at all,
        so a derivation that gave them a slope would be modelling bytes that
        never move -- the over-reservation this feeds back into."""
        got = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR, SHIPPING_ATTENTION, KV_CELL_BYTES, N_ATTENTION
        )
        self.assertEqual(got[1], 0.0)
        self.assertEqual(got[2], 0.0)


class TheFrozenTripleMispricesAMovedCut(unittest.TestCase):
    """The planted failure: move one attention layer and the frozen triple is
    wrong by whole KV cells, while the derived vector tracks."""

    #: 28,20,16 -> a cut whose attention split moved one layer from stage 2
    #: onto stage 1. Stage 2 now holds 3 of 16 while the flip still hands it
    #: 4/16 of the token axis, so it must RECEIVE one layer it did not before.
    MOVED_ATTENTION = (7, 6, 3)

    def test_the_moved_cut_makes_a_previously_free_rank_pay(self):
        got = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR, self.MOVED_ATTENTION, KV_CELL_BYTES, N_ATTENTION
        )
        self.assertAlmostEqual(got[2], KV_CELL_BYTES, delta=1.0)

    def test_the_frozen_triple_underprices_that_rank_by_a_whole_cell(self):
        frozen = sr.SHIP_PIN.basis_per_row_bytes
        got = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR,
            self.MOVED_ATTENTION,
            KV_CELL_BYTES,
            N_ATTENTION,
            MEASURED_BASELINE,
        )
        self.assertGreater(
            got[2] - frozen[2],
            KV_CELL_BYTES * 0.9,
            "the frozen triple no longer misprices the moved cut, so it has "
            "stopped being cut-dependent and this guard is watching nothing",
        )

    def test_moving_a_layer_onto_rank0_raises_its_slope_by_one_cell(self):
        base = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR, (7, 5, 4), KV_CELL_BYTES, N_ATTENTION
        )
        fewer = derive_seam_slope_bytes_per_token(
            FLIP_VECTOR, (6, 6, 4), KV_CELL_BYTES, N_ATTENTION
        )
        self.assertAlmostEqual(fewer[0] - base[0], KV_CELL_BYTES, delta=1.0)


class TheDerivationRefusesNonsense(unittest.TestCase):
    def test_a_mis_sized_flip_vector_is_refused(self):
        with self.assertRaises(ValueError):
            derive_seam_slope_bytes_per_token((32, 16), (7, 5, 4), 2326.7, 16)

    def test_a_zero_flip_vector_is_refused(self):
        with self.assertRaises(ValueError):
            derive_seam_slope_bytes_per_token((0, 0, 0), (7, 5, 4), 2326.7, 16)

    def test_a_negative_cell_is_refused(self):
        with self.assertRaises(ValueError):
            derive_seam_slope_bytes_per_token(FLIP_VECTOR, (7, 5, 4), -1.0, 16)


class TheFrozenBasisIsNotTouched(unittest.TestCase):
    """This change must not move the drift watchdog.

    ``SHIP_PIN.basis_per_row_bytes`` exists so that when the live values move,
    ``test_arming_floor_funding_662`` goes red and the pin is re-derived. A
    commit that overwrote it with a derived vector would delete that signal
    while changing no live behaviour, because the sizing path reads the
    per-boot measured record, not the pin.
    """

    def test_the_pin_still_carries_the_measured_triple(self):
        self.assertEqual(
            sr.SHIP_PIN.basis_per_row_bytes,
            (2360.3031340235552, 424.1172657292698, 550.6682501797533),
        )


if __name__ == "__main__":
    unittest.main()
