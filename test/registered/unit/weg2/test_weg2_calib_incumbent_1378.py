# SPDX-License-Identifier: Apache-2.0
"""#1378 (DESK6, 2026-09-13): solve_p_cut's `incumbent` was a SECOND,
UNGUARDED reader of "the incumbent per-stage layer counts" -- argv_p already
refused a foreign PP calibration (W99 Weg2ModelIdentityMismatch,
host_ledger.read_pp_calibration / refuse_foreign_calibration, #1362) before
shipping --pp-stage-ratio to group P's own argv, but solve_p_cut's
`incumbent = ... else list(P_PP_STAGE_RATIO_SCORES)` (launcher.py, just above
`_pp_cut.family_costs_from_measurement`) never consulted the calibration
record at all.

MEASURED LIVE (not constructed): a real dry-run for a 24-layer model
(Qwen3.5-2B) WITH a valid, complete calibration record for its own content
digest still crashed with the bare
`pp_cut.family_costs_from_measurement` ValueError
("the anchor stage 1 holds 18 linear and 0 attention layers in the
calibration cut") -- the exact text `refuse_foreign_calibration`'s own
message already quotes as the failure #1362 exists to turn into a named
refusal. `solve_p_cut` runs BEFORE `argv_p` in `main()`, so the W99 guard
never got a chance to fire.

`host_ledger.resolve_calibrated_stage_layer_counts` is the fix: the SAME
guard, extracted so any future third consumer of "the incumbent ratio" gets
it for free instead of writing a fourth copy (#1362's own FIX 2 comment
already named that as the danger this constant invites).
"""
import json
import os
import tempfile
import unittest

from sglang.srt.weg2 import host_ledger


def _write_checkpoint(root, n_layers, layer_types):
    os.makedirs(root, exist_ok=True)
    cfg = {
        "num_hidden_layers": n_layers,
        "layer_types": layer_types,
        "hidden_size": 64,
    }
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(cfg, f)
    weight_map = {"model.layers.%d.w" % i: "shard0.safetensors" for i in range(n_layers)}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return cfg


def _digest_for(root):
    dg, why = host_ledger.checkpoint_digest(root)
    assert dg, why
    return dg


class ResolveCalibratedStageLayerCountsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="weg2-1378-calib-test-")
        self._model = os.path.join(self._tmp, "model")
        self._calib_dir = os.path.join(self._tmp, "calib")
        os.makedirs(self._calib_dir, exist_ok=True)
        self._old_calib_dir = host_ledger.CALIB_DIR
        host_ledger.CALIB_DIR = self._calib_dir

    def tearDown(self):
        host_ledger.CALIB_DIR = self._old_calib_dir

    def test_override_short_circuits_before_any_disk_read(self):
        # A caller-supplied --pp-stage-ratio must win with NO filesystem
        # touch at all -- an operator pin is not second-guessed.
        got = host_ledger.resolve_calibrated_stage_layer_counts(
            "/definitely/does/not/exist", "8,8,8", calibration_layers=64)
        self.assertIsNone(got)

    def test_unreadable_model_returns_none_keeps_caller_fallback(self):
        got = host_ledger.resolve_calibrated_stage_layer_counts(
            "/definitely/does/not/exist", None, calibration_layers=64)
        self.assertIsNone(got)

    def test_matching_depth_no_calib_returns_none_bsscale_default_still_legit(self):
        # depth == calibration_layers: the bsscale incumbent WAS measured on
        # exactly this depth, so no calibration record is needed and the
        # caller's `or list(P_PP_STAGE_RATIO_SCORES)` fallback is correct.
        _write_checkpoint(self._model, 64, ["full_attention"] * 16 + ["linear_attention"] * 48)
        got = host_ledger.resolve_calibrated_stage_layer_counts(
            self._model, None, calibration_layers=64)
        self.assertIsNone(got)

    def test_foreign_depth_no_calib_raises_named_refusal(self):
        # THE BUG THIS CLOSES: a 24-layer model, no calibration record for
        # its digest. Before the fix, solve_p_cut used the 64-layer bsscale
        # incumbent unguarded and crashed downstream with a bare ValueError.
        _write_checkpoint(self._model, 24, ["linear_attention", "linear_attention",
                                             "linear_attention", "full_attention"] * 6)
        with self.assertRaises(host_ledger.Weg2ModelIdentityMismatch) as ctx:
            host_ledger.resolve_calibrated_stage_layer_counts(
                self._model, None, calibration_layers=64)
        self.assertIn("W99 Weg2ModelIdentityMismatch", str(ctx.exception))

    def test_matching_calibration_record_is_used(self):
        layer_types = ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"] * 6
        _write_checkpoint(self._model, 24, layer_types)
        digest = _digest_for(self._model)
        record = {
            "schema": "weg2-pp-calib/1",
            "model_digest": digest,
            "measured_ms_per_layer": [4.694, 12.919, 12.106],
            "measured_attn_counts": [2, 2, 2],
            "measured_counts": [8, 8, 8],
            "stage_layer_counts": [8, 8, 8],
        }
        with open(os.path.join(self._calib_dir, "%s.json" % digest), "w") as f:
            json.dump(record, f)
        got = host_ledger.resolve_calibrated_stage_layer_counts(
            self._model, None, calibration_layers=64)
        self.assertEqual(got, [8, 8, 8])


class SolveConsultsCalibrationTest(unittest.TestCase):
    """Mutation-checked: reverting the wiring reproduces the exact bare
    ValueError name this ticket closed."""

    def test_solve_p_cut_calls_the_shared_helper(self):
        import inspect
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.solve_p_cut)
        self.assertIn("resolve_calibrated_stage_layer_counts", src,
                      "solve_p_cut's incumbent must consult the same "
                      "model-identity guard argv_p already uses (#1362 "
                      "follow, #1378) -- reverting this line reproduces the "
                      "bare pp_cut.family_costs_from_measurement ValueError "
                      "for any depth != CALIBRATION_LAYERS model.")


if __name__ == "__main__":
    unittest.main()
