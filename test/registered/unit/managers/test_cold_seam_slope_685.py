"""#685 cold path: price the per-token seam on a first boot instead of zero.

THE DEFECT. ``read_seam_reserve`` returns ``SeamReserve(provenance=cold)`` when
no measured record exists, and every field defaults to zero -- including
``per_row_bytes``. ``solve_pool_tokens`` then evaluates
``staging(T) = A + max(F, a*T)`` with ``a = 0``, so the per-token term vanishes
and the pool is sized floor-only. A first boot therefore grants a pool whose
cutover it cannot fund. That is the cold-overshoot class: the arming floor was
already charged on a cold record (#662-F4/A0), the SLOPE never was.

WHY IT CAN BE DERIVED RATHER THAN GUESSED. #685 established that a rank's
per-row seam cost is the number of full-attention layers it must RECEIVE at the
cutover:

    received_r = max(0, tp_share_r * n_attention_total - attention_held_r)

Every term is rank-LOCAL and known at boot: the share from
``--phase-flip-tp-vector``, ``attention_held`` from the modules this runner
actually built, ``n_attention_total`` from the model config, and the per-layer
cell from the configurator's own cell divided by this rank's attention count.
Nothing is measured and nothing is transferred from another rig.

THE THREE RULES THIS FILE PINS.

* DERIVED IS A FALLBACK, NEVER AN OVERRIDE. A stored record wins whenever one
  exists: it is a measurement of this rig, and the derivation is a model.
* ABSTAIN RATHER THAN APPROXIMATE. If the cell, the flip vector, the attention
  count or the model total is missing, the slope stays zero and says so. A
  configurator with no single cell (hybrid SWA, MiniMax sparse) is the case the
  existing code already refuses to invent a cell for, and this follows it.
* THE POOL MUST ACTUALLY SHRINK. A derivation that changed no sizing decision
  would be decoration; the last case asserts the token count drops.
"""

import logging
import types
import unittest

import torch

torch.set_default_device("cpu")

from sglang.srt.managers import phase_flip_seam_reserve as seam  # noqa: E402
from sglang.srt.managers.seam_slope import (  # noqa: E402
    derive_seam_slope_for_rank,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (  # noqa: E402
    ModelRunnerKVCacheMixin,
)

MIXIN_LOGGER = "sglang.srt.model_executor.model_runner_kv_cache_mixin"

#: The live flagset.
FLIP_VECTOR_STR = "32,16,16"
N_ATTENTION_TOTAL = 16
#: Rank 0 holds 7 of 16 while the flip hands it 32/64 of the token axis, so it
#: must receive one layer. Ranks 1 and 2 shed or break even.
ATTENTION_HELD = (7, 5, 4)
#: Per-token bytes for one full-attention layer, and this rank's whole cell.
PER_LAYER_CELL = 2326.7


class TheSingleRankDerivationExists(unittest.TestCase):
    """The cold path is rank-local: it never needs its peers' layer counts."""

    def test_rank0_receives_one_layer(self):
        got = derive_seam_slope_for_rank(
            flip_tp_vector=(32, 16, 16),
            rank=0,
            attention_held=7,
            kv_bytes_per_token_per_attn_layer=PER_LAYER_CELL,
            n_attention_total=N_ATTENTION_TOTAL,
        )
        self.assertAlmostEqual(got, PER_LAYER_CELL, delta=1.0)

    def test_the_shedding_and_neutral_ranks_receive_nothing(self):
        for rank, held in ((1, 5), (2, 4)):
            got = derive_seam_slope_for_rank(
                flip_tp_vector=(32, 16, 16),
                rank=rank,
                attention_held=held,
                kv_bytes_per_token_per_attn_layer=PER_LAYER_CELL,
                n_attention_total=N_ATTENTION_TOTAL,
            )
            self.assertEqual(got, 0.0, f"rank {rank}")

    def test_it_agrees_with_the_whole_vector_form(self):
        from sglang.srt.managers.seam_slope import (
            derive_seam_slope_bytes_per_token,
        )

        whole = derive_seam_slope_bytes_per_token(
            (32, 16, 16), ATTENTION_HELD, PER_LAYER_CELL, N_ATTENTION_TOTAL
        )
        for rank, held in enumerate(ATTENTION_HELD):
            self.assertAlmostEqual(
                derive_seam_slope_for_rank(
                    flip_tp_vector=(32, 16, 16),
                    rank=rank,
                    attention_held=held,
                    kv_bytes_per_token_per_attn_layer=PER_LAYER_CELL,
                    n_attention_total=N_ATTENTION_TOTAL,
                ),
                whole[rank],
                places=6,
            )

    def test_an_out_of_range_rank_is_refused(self):
        with self.assertRaises(ValueError):
            derive_seam_slope_for_rank(
                flip_tp_vector=(32, 16, 16),
                rank=7,
                attention_held=7,
                kv_bytes_per_token_per_attn_layer=PER_LAYER_CELL,
                n_attention_total=N_ATTENTION_TOTAL,
            )


def _runner(
    *,
    flip_vector=FLIP_VECTOR_STR,
    attention_held=7,
    n_layers=64,
    interval=4,
    rank=0,
):
    holder = types.SimpleNamespace(
        server_args=types.SimpleNamespace(
            phase_flip_tp_vector=flip_vector,
            enable_phase_flip=True,
        ),
        model_config=types.SimpleNamespace(
            num_hidden_layers=n_layers,
            full_attention_interval=interval,
        ),
    )
    holder._lane_kv_bearing_layer_count = lambda: attention_held
    holder._seam_world_rank = lambda: rank
    holder._maybe_price_cold_seam = types.MethodType(
        ModelRunnerKVCacheMixin._maybe_price_cold_seam, holder
    )
    return holder


def _configurator(cell):
    return types.SimpleNamespace(_cell_size=cell)


class TheColdPathIsPricedFromTheDerivation(unittest.TestCase):
    def test_a_cold_reserve_gains_the_derived_slope(self):
        runner = _runner()
        cold = seam.SeamReserve(provenance=seam.PROVENANCE_COLD)
        self.assertEqual(cold.per_row_bytes, 0.0)
        cell = int(PER_LAYER_CELL * 7)
        with self.assertLogs(MIXIN_LOGGER, level=logging.INFO) as cm:
            priced = runner._maybe_price_cold_seam(cold, _configurator(cell))
        self.assertAlmostEqual(priced.per_row_bytes, PER_LAYER_CELL, delta=2.0)
        self.assertTrue(any("derived" in line.lower() for line in cm.output), cm.output)

    def test_the_derived_reserve_does_not_masquerade_as_measured(self):
        """THE CONSUMPTION BOUNDARY, pinned rather than papered over.

        ``SeamReserve.active`` is ``id_space > 0 and (any cost field > 0)``.
        ``id_space`` is the token count the record was MEASURED at, and the
        downstream solve anchors on it together with ``have_bytes``
        (``t_floor = t_m + (have_m - A - F) // cell``). A derived slope has no
        measurement point, so setting those two to make it ``active`` would be
        fabricating the anchor -- the approximation this work is under orders
        not to make.

        So the derived slope is CARRIED but the reserve stays inactive, and
        the budget path still returns floor-only. Closing that needs an
        anchor-free consumption branch; the anchor-free solver
        (``solve_pool_tokens``) exists but has no live caller to follow, so
        which budget it is solved against is a design decision on the boot
        path, not a wiring detail. Reported, not guessed.
        """
        runner = _runner()
        cold = seam.SeamReserve(provenance=seam.PROVENANCE_COLD)
        self.assertFalse(cold.active)
        priced = runner._maybe_price_cold_seam(
            cold, _configurator(int(PER_LAYER_CELL * 7))
        )
        self.assertGreater(priced.per_row_bytes, 0.0)
        self.assertEqual(priced.id_space, 0)
        self.assertFalse(
            priced.active,
            "a derived reserve became active, which means it acquired a "
            "measurement anchor it does not have",
        )


class AStoredRecordAlwaysWins(unittest.TestCase):
    """Derived is the cold FALLBACK, never an override."""

    def test_a_stored_reserve_is_returned_unchanged(self):
        runner = _runner()
        stored = seam.SeamReserve(
            per_row_bytes=424.1, provenance=seam.PROVENANCE_STORED
        )
        got = runner._maybe_price_cold_seam(
            stored, _configurator(int(PER_LAYER_CELL * 7))
        )
        self.assertIs(got, stored)
        self.assertEqual(got.per_row_bytes, 424.1)

    def test_an_override_reserve_is_returned_unchanged(self):
        runner = _runner()
        override = seam.SeamReserve(
            fixed_bytes=1 << 20, provenance=seam.PROVENANCE_OVERRIDE
        )
        self.assertIs(
            runner._maybe_price_cold_seam(
                override, _configurator(int(PER_LAYER_CELL * 7))
            ),
            override,
        )


class AbsentInputsAbstainRatherThanApproximate(unittest.TestCase):
    def _cold(self):
        return seam.SeamReserve(provenance=seam.PROVENANCE_COLD)

    def test_no_single_cell_abstains(self):
        """The case the existing code already refuses to invent a cell for."""
        runner = _runner()
        cold = self._cold()
        got = runner._maybe_price_cold_seam(cold, _configurator(0))
        self.assertIs(got, cold)

    def test_no_flip_vector_abstains(self):
        runner = _runner(flip_vector=None)
        cold = self._cold()
        self.assertIs(runner._maybe_price_cold_seam(cold, _configurator(16000)), cold)

    def test_no_attention_layers_on_this_runner_abstains(self):
        runner = _runner(attention_held=0)
        cold = self._cold()
        self.assertIs(runner._maybe_price_cold_seam(cold, _configurator(16000)), cold)

    def test_a_rank_that_receives_nothing_is_left_at_zero(self):
        """Rank 1 sheds KV. Charging it a per-token seam would reserve against
        a transfer that does not happen -- the over-reservation #685 named."""
        runner = _runner(attention_held=5, rank=1)
        cold = self._cold()
        got = runner._maybe_price_cold_seam(
            cold, _configurator(int(PER_LAYER_CELL * 5))
        )
        self.assertEqual(got.per_row_bytes, 0.0)


class TheSizerActuallyRefusesTheUnfundablePool(unittest.TestCase):
    """The consequence, not the intention: the token count must drop.

    A derivation that changed no sizing decision would be decoration.
    """

    CORRIDOR_RELAXED = 8 << 30
    CELL = int(PER_LAYER_CELL * 7)

    def test_zero_slope_grants_more_tokens_than_the_derived_one(self):
        floor_only = seam.solve_pool_tokens(
            corridor_relaxed_bytes=self.CORRIDOR_RELAXED,
            cell_bytes=self.CELL,
            fixed_bytes=227 << 20,
            per_row_bytes=0.0,
        )
        priced = seam.solve_pool_tokens(
            corridor_relaxed_bytes=self.CORRIDOR_RELAXED,
            cell_bytes=self.CELL,
            fixed_bytes=227 << 20,
            per_row_bytes=PER_LAYER_CELL,
        )
        self.assertGreater(
            floor_only,
            priced,
            "the derived slope changes no sizing decision, so pricing it is "
            "decoration rather than a fix",
        )

    def test_the_cold_boot_pool_shrinks_once_the_slope_is_priced(self):
        runner = _runner()
        cold = seam.SeamReserve(provenance=seam.PROVENANCE_COLD)
        priced = runner._maybe_price_cold_seam(cold, _configurator(self.CELL))
        before = seam.solve_pool_tokens(
            corridor_relaxed_bytes=self.CORRIDOR_RELAXED,
            cell_bytes=self.CELL,
            fixed_bytes=0,
            per_row_bytes=cold.per_row_bytes,
        )
        after = seam.solve_pool_tokens(
            corridor_relaxed_bytes=self.CORRIDOR_RELAXED,
            cell_bytes=self.CELL,
            fixed_bytes=0,
            per_row_bytes=priced.per_row_bytes,
        )
        self.assertGreater(before, after)


if __name__ == "__main__":
    unittest.main()
