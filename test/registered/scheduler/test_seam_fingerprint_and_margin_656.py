# SPDX-License-Identifier: Apache-2.0
"""#656 R4: the seam record's identity, and the solver's margin.

Two defects, both found on metal in the kvuniverse-r4 window:

* The seam record hangs off ``measured_kv_budget_fingerprint_fields``, whose
  digest did NOT include the four settings that move the record's measured
  position by hundreds of MiB. Boot K1 consumed boot J's UNCAPPED record while
  running with ``--gdn-resident-state-slots 4``; that direction under-sizes
  and is harmless, but the reverse over-sizes and is the boot-E wedge.
* ``seam_allowed_tokens`` targets EQUALITY with the floor, so the pool it
  solves lands exactly on it. Boot K3 derived 610942 and re-measured rank2 at
  1430 MiB against a 1455 MiB floor -- 25 MiB the wrong side of its own gate.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.phase_flip_seam_reserve import (
    DEFAULT_MARGIN_MIB,
    ENV_MARGIN_MIB,
    SeamReserve,
    PROVENANCE_STORED,
    seam_allowed_tokens,
    seam_margin_bytes,
)
from sglang.srt.uneven_perf import measured_kv_budget_fingerprint_fields

MIB = 1 << 20


def _args(**over):
    """A minimally complete args object for the fingerprint function."""
    base = dict(
        model_path="/m",
        tp_size=1,
        rank_gpu_id=[0, 1, 2],
        rank_tp_ratio=None,
        rank_kv_ratio="coupled",
        rank_auto_reserve_mib="auto",
        rank_gpu_memory_mib=[31583, 15750, 18205],
        mem_fraction_static=0.74,
        kv_cache_dtype="fp8_e4m3",
        context_length=393216,
        page_size=1,
        quantization=None,
        max_running_requests=4,
        chunked_prefill_size=512,
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=None,
        speculative_adaptive=False,
        speculative_adaptive_config=None,
        speculative_num_draft_tokens=4,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=24)),
        pp_size=3,
        pp_layer_ratio=[28, 20, 16],
        # the four this test is about, at their defaults
        gdn_resident_state_slots=None,
        enable_kv_session_offload=False,
        pp_stage_ratio=None,
        phase_flip_tp_vector=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestSeamFingerprintIdentity(unittest.TestCase):
    """One assertion per setting that moves the measured position."""

    def test_defaults_keep_pre_existing_digests_valid(self):
        # The keys must be ABSENT at their defaults, or every registry written
        # before this change is orphaned on a rig that runs the defaults.
        fields = measured_kv_budget_fingerprint_fields(_args())
        for name in (
            "gdn_resident_state_slots",
            "enable_kv_session_offload",
            "pp_stage_ratio",
            "phase_flip_tp_vector",
        ):
            self.assertNotIn(name, fields, f"{name} must be omitted at default")

    def test_gdn_resident_state_slots_changes_identity(self):
        # Measured on this rig: the cap moved `have` by +562/+350/+308 MiB at
        # an identical pool (boot K1 against boot J).
        self.assertNotEqual(
            measured_kv_budget_fingerprint_fields(_args()),
            measured_kv_budget_fingerprint_fields(
                _args(gdn_resident_state_slots=4)
            ),
        )

    def test_kv_session_offload_changes_identity(self):
        self.assertNotEqual(
            measured_kv_budget_fingerprint_fields(_args()),
            measured_kv_budget_fingerprint_fields(
                _args(enable_kv_session_offload=True)
            ),
        )

    def test_pp_stage_ratio_changes_identity(self):
        # PP weight split -> PP_bytes -> the arena tail max(0, PP - TP), which
        # IS the seam floor on the binding rank (1455 MiB on rank2).
        self.assertNotEqual(
            measured_kv_budget_fingerprint_fields(_args(pp_stage_ratio=[14, 10, 8])),
            measured_kv_budget_fingerprint_fields(_args(pp_stage_ratio=[15, 10, 7])),
        )

    def test_phase_flip_tp_vector_changes_identity(self):
        # The other half of that same subtraction.
        self.assertNotEqual(
            measured_kv_budget_fingerprint_fields(
                _args(phase_flip_tp_vector="32,16,16")
            ),
            measured_kv_budget_fingerprint_fields(
                _args(phase_flip_tp_vector="30,16,18")
            ),
        )


class TestSeamMargin(unittest.TestCase):
    def setUp(self):
        self._clean = mock.patch.dict(os.environ, {}, clear=False)
        self._clean.start()
        os.environ.pop(ENV_MARGIN_MIB, None)
        self.addCleanup(self._clean.stop)

    def test_default_margin_is_the_documented_value(self):
        self.assertEqual(seam_margin_bytes(), DEFAULT_MARGIN_MIB * MIB)

    def test_override_is_honoured(self):
        os.environ[ENV_MARGIN_MIB] = "64"
        self.assertEqual(seam_margin_bytes(), 64 * MIB)

    def test_malformed_override_falls_back_to_default_not_to_zero(self):
        # The failure this exists to prevent is a zero-margin pool, so an
        # unparseable or negative override must not be read as "no margin".
        for bad in ("", "abc", "-1"):
            os.environ[ENV_MARGIN_MIB] = bad
            self.assertEqual(
                seam_margin_bytes(), DEFAULT_MARGIN_MIB * MIB, f"override {bad!r}"
            )

    def test_margin_reduces_the_solved_pool_in_the_floor_bound_regime(self):
        """CAN-FAIL PROOF: boot K3's rank2, with and without the margin.

        Floor-bound regime (a*t_floor <= F). Without a margin the solver
        returns the equality point it shipped on metal; with one it stands
        back by exactly margin/cell tokens.
        """
        rank2 = SeamReserve(
            fixed_bytes=1455 * MIB,
            per_row_bytes=544.2,
            have_bytes=1822 * MIB,
            id_space=563974,
            provenance=PROVENANCE_STORED,
        )
        cell = 8192

        os.environ[ENV_MARGIN_MIB] = "0"
        unmargined = seam_allowed_tokens(cell, rank2)
        # This is the number boot K3 actually derived (610942), reproduced from
        # the record it was derived from.
        self.assertEqual(unmargined, 563974 + (1822 - 1455) * MIB // cell)

        os.environ[ENV_MARGIN_MIB] = "192"
        margined = seam_allowed_tokens(cell, rank2)
        self.assertEqual(margined, unmargined - 192 * MIB // cell)
        self.assertLess(margined, unmargined)

    def test_margin_also_binds_in_the_slack_bound_regime(self):
        """rank0 is slack-bound, where F appears only in the regime test.

        Taking the margin off the floor instead of off the measured position
        would leave this branch completely unmargined -- the regression this
        asserts against.
        """
        rank0 = SeamReserve(
            fixed_bytes=455 * MIB,
            per_row_bytes=2360.6,
            have_bytes=4306 * MIB,
            id_space=563974,
            provenance=PROVENANCE_STORED,
        )
        cell = 14336

        os.environ[ENV_MARGIN_MIB] = "0"
        unmargined = seam_allowed_tokens(cell, rank0)
        os.environ[ENV_MARGIN_MIB] = "192"
        margined = seam_allowed_tokens(cell, rank0)

        # Slack-bound: allowed = (have + t_m*cell) / (cell + a), so the margin
        # must still move it.
        self.assertLess(margined, unmargined)
        self.assertAlmostEqual(
            unmargined - margined,
            (192 * MIB) / (cell + 2360.6),
            delta=2,
        )


if __name__ == "__main__":
    unittest.main()
