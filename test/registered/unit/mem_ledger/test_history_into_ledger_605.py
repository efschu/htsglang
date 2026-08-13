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
"""The boot-history bands reaching the ledger, and refusing it (#605).

RED-FIRST. ``build_card_ledgers`` had no way to be told what this rig's own
boots measured, so both terms shipped as constants from a window that no
longer exists. The default -- no history -- must stay byte-identical, which is
the first test here and the one that protects every existing caller.
"""

import unittest

from sglang.srt.mem_ledger.boot_history import (
    POST_HARDWARE_RESIDUAL,
    POST_LOAD_TRANSIENT,
    BootHistory,
    HistoryBand,
)
from sglang.srt.mem_ledger.calibration import CalibrationProfile, CardResidual
from sglang.srt.mem_ledger.engine import (
    LOAD_TRANSIENT_REFERENCE_MIB,
    TERM_HARDWARE_RESIDUAL,
    TERM_LOAD_TRANSIENT,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
)

UUID = "GPU-31d7ef41"


def _cards():
    return [
        CardFacts(
            gpu_id=0,
            uuid=UUID,
            name="NVIDIA GeForce RTX 5090",
            total_mib=32607,
            reserved_mib=518,
        )
    ]


def _calibration():
    return CalibrationProfile(
        fingerprint="a191a0712717",
        driver="580",
        build="torch2.x",
        cards=(
            CardResidual(
                uuid=UUID,
                name="NVIDIA GeForce RTX 5090",
                cuda_context_bytes=600 * (1 << 20),
                allocator_granularity_bytes=64 * (1 << 20),
                lazy_workspace_bytes=0,
            ),
        ),
    )


def _inputs():
    return DemandInputs(
        weight_mib_per_rank=[14000],
        activation_mib_per_rank=[500.0],
        capture_tokens_per_rank=[0],
        capture_mib_per_rank=[600.0],
        phase_footprint_fingerprint="a191a0712717",
        phase_footprint_source_per_rank=("[measured] test",),
    )


def _build(history=None):
    return build_card_ledgers(
        _inputs(),
        cards=_cards(),
        rank_gpu_id=[0],
        user_reserve_mib={0: 1024},
        calibration=_calibration(),
        history=history,
    )[0]


def _term(ledger, name):
    return next((t for t in ledger.terms if t.name == name), None)


def _history(*bands):
    return BootHistory(bands={(b.uuid, b.post): b for b in bands}, n_boots=472)


def _band(post, charge, low, high, refused=False, reason="because"):
    return HistoryBand(
        uuid=UUID,
        post=post,
        charge_mib=charge,
        low_mib=low,
        high_mib=high,
        n_boots=462,
        refused=refused,
        reason=reason,
    )


class TestDefaultIsByteIdentical(unittest.TestCase):
    def test_no_history_leaves_both_terms_exactly_as_before(self):
        """Every existing caller passes no history and must be unaffected."""
        ledger = _build(history=None)
        self.assertEqual(_term(ledger, TERM_HARDWARE_RESIDUAL).mib, 664)
        self.assertEqual(
            _term(ledger, TERM_LOAD_TRANSIENT).mib, LOAD_TRANSIENT_REFERENCE_MIB
        )

    def test_history_without_a_band_for_this_card_changes_nothing(self):
        ledger = _build(history=_history())
        self.assertEqual(_term(ledger, TERM_HARDWARE_RESIDUAL).mib, 664)
        self.assertEqual(
            _term(ledger, TERM_LOAD_TRANSIENT).mib, LOAD_TRANSIENT_REFERENCE_MIB
        )


class TestMeasuredBeatsInherited(unittest.TestCase):
    def test_the_residual_band_replaces_the_old_window_constant(self):
        """664 MiB was calibrated on window-2026-08-06 and measures 25% low."""
        ledger = _build(history=_history(_band(POST_HARDWARE_RESIDUAL, 902, 802, 902)))
        term = _term(ledger, TERM_HARDWARE_RESIDUAL)
        self.assertEqual(term.mib, 902)
        self.assertNotEqual(term.mib, 664)

    def test_the_band_and_its_boot_count_are_written_into_the_derivation(self):
        """A calibrated number without its spread cannot be invalidated."""
        ledger = _build(history=_history(_band(POST_HARDWARE_RESIDUAL, 902, 802, 902)))
        derivation = _term(ledger, TERM_HARDWARE_RESIDUAL).derivation
        self.assertIn("802", derivation)
        self.assertIn("902", derivation)
        self.assertIn("462", derivation)

    def test_a_calibrated_transient_band_replaces_the_70_mib_constant(self):
        ledger = _build(history=_history(_band(POST_LOAD_TRANSIENT, 210, 190, 210)))
        self.assertEqual(_term(ledger, TERM_LOAD_TRANSIENT).mib, 210)


class TestAWidePostRefusesRatherThanAverages(unittest.TestCase):
    def test_a_refused_transient_band_makes_the_term_unbounded(self):
        """0-18486 MiB is not a constant; the ledger must refuse, not average."""
        ledger = _build(
            history=_history(
                _band(
                    POST_LOAD_TRANSIENT,
                    None,
                    0,
                    18486,
                    refused=True,
                    reason="spans 0-18486 MiB over 462 boots",
                )
            )
        )
        self.assertIsNone(_term(ledger, TERM_LOAD_TRANSIENT))
        joined = " ".join(ledger.unbounded)
        self.assertIn(TERM_LOAD_TRANSIENT, joined)
        self.assertIn("18486", joined)

    def test_a_refused_residual_band_falls_back_to_the_calibration(self):
        """A refusal on a term the probe CAN measure keeps the probe's number,
        rather than throwing away a real calibration for a wide history."""
        ledger = _build(
            history=_history(
                _band(POST_HARDWARE_RESIDUAL, None, 100, 900, refused=True)
            )
        )
        self.assertEqual(_term(ledger, TERM_HARDWARE_RESIDUAL).mib, 664)


class TestCoLocationStillMultiplies(unittest.TestCase):
    def test_two_ranks_on_one_card_pay_the_band_twice(self):
        inputs = DemandInputs(
            weight_mib_per_rank=[14000, 14000],
            activation_mib_per_rank=[500.0, 500.0],
            capture_tokens_per_rank=[0, 0],
            capture_mib_per_rank=[600.0, 600.0],
            phase_footprint_fingerprint="a191a0712717",
            phase_footprint_source_per_rank=("[measured] t", "[measured] t"),
        )
        ledger = build_card_ledgers(
            inputs,
            cards=_cards(),
            rank_gpu_id=[0, 0],
            user_reserve_mib={0: 1024},
            calibration=_calibration(),
            history=_history(_band(POST_HARDWARE_RESIDUAL, 902, 802, 902)),
        )[0]
        self.assertEqual(_term(ledger, TERM_HARDWARE_RESIDUAL).mib, 902 * 2)


if __name__ == "__main__":
    unittest.main()
