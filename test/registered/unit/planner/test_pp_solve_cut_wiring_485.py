"""--pp-solve-cut wiring: refusals, and the default path's inertness (#485).

The flag exists so the solved cut is REACHABLE, not so it becomes the
default. The most important test in this file is therefore
``test_unset_touches_nothing``: the shipping configuration must boot exactly
as it did before this flag existed.

Every other test here is a refusal. That is the point of the feature's
design, and it is a direct consequence of C38/C39: this gate spent three
shifts reading as calibrated while being ~3000 MiB wrong on any cut it had
not been fitted on, because a term nobody had priced defaulted to zero and
zero reads as free memory. So the wiring sources its terms from a measured
census and stops the boot, naming the number, when it cannot.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.planner import pp_cut_calibration as cal
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _census(pp_rank, n_attn, n_linear, **kw):
    blob = {
        "pp_rank": pp_rank,
        "n_attn_layers": n_attn,
        "n_linear_layers": n_linear,
        "params_mib": {
            "layers_attention": 364.88 * n_attn,
            "layers_linear": 366.29 * n_linear,
            "visual": 930.0,
        },
        "pools_mib": 1000.0,
        "graphs_mib": 500.0,
        "alloc_mib": 1.0,
        "reserved_mib": 1.0,
        "nvml_used_mib": 20000.0,
        "nvml_free_mib": 1000.0,
        "nvml_total_mib": 21000.0,
    }
    if pp_rank == 0:
        blob["params_mib"]["embed_tokens"] = 2425.0
    blob.update(kw)
    return blob


def _write(dirpath, blobs):
    for b in blobs:
        with open(os.path.join(dirpath, f"census_pp{b['pp_rank']}.json"), "w") as fh:
            json.dump(b, fh)


class TestCalibrationRefusals(CustomTestCase):
    def test_missing_directory_is_refused(self):
        with self.assertRaises(cal.PPCutCalibrationError) as cm:
            cal.load_census_calibration("/nonexistent/census/dir")
        self.assertIn("SGLANG_RESIDENCY_CENSUS", str(cm.exception))

    def test_empty_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(cal.PPCutCalibrationError):
                cal.load_census_calibration(d)

    def test_a_partial_census_is_refused_not_interpolated(self):
        # Ranks 0 and 2 present, rank 1 missing. A residual guessed for the
        # missing rank is exactly the half-measured number C39 is about.
        with tempfile.TemporaryDirectory() as d:
            _write(d, [_census(0, 10, 30), _census(2, 3, 9)])
            with self.assertRaises(cal.PPCutCalibrationError) as cm:
                cal.load_census_calibration(d)
            self.assertIn("complete", str(cm.exception))

    def test_a_census_without_lm_head_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            blobs = [_census(0, 10, 30), _census(1, 3, 9), _census(2, 3, 9)]
            _write(d, blobs)  # none carries lm_head
            with self.assertRaises(cal.PPCutCalibrationError) as cm:
                cal.load_census_calibration(d)
            self.assertIn("lm_head", str(cm.exception))

    def test_a_complete_census_loads_and_records_its_cut(self):
        with tempfile.TemporaryDirectory() as d:
            last = _census(2, 3, 9)
            last["params_mib"]["lm_head"] = 2425.0
            _write(d, [_census(0, 10, 30), _census(1, 3, 9), last])
            c = cal.load_census_calibration(d)
            self.assertEqual(c.calibrated_on_counts, (40, 12, 12))
            self.assertAlmostEqual(c.attn_layer_mib, 364.88, delta=0.01)
            self.assertAlmostEqual(c.linear_layer_mib, 366.29, delta=0.01)
            self.assertAlmostEqual(c.embedding_mib, 2425.0, delta=0.01)
            self.assertAlmostEqual(c.lm_head_mib, 2425.0, delta=0.01)
            # The residual is DERIVED, never defaulted: used - params - pools.
            self.assertEqual(len(c.residual_mib), 3)
            # And it is reported per rank, because it is not one number.
            self.assertNotEqual(c.residual_mib[0], c.residual_mib[1])

    def test_the_state_term_is_zero_until_the_arena_is_subtracted(self):
        # load_census_calibration cannot separate the recurrent-state pool
        # from the KV arena without the geometry, and it says so by leaving
        # the term at zero rather than guessing a share of `pools`.
        with tempfile.TemporaryDirectory() as d:
            last = _census(2, 3, 9)
            last["params_mib"]["lm_head"] = 2425.0
            _write(d, [_census(0, 10, 30), _census(1, 3, 9), last])
            c = cal.load_census_calibration(d)
            self.assertEqual(c.state_per_linear_mib, 0.0)
            # With the arena priced away, what remains is per linear layer.
            c2 = cal.with_arena_split_state(
                c,
                census_dir=d,
                kv_bytes_per_token_per_attn_layer=2048.0,
                pool_tokens=1000,
                tp_token_shares=None,
            )
            self.assertGreater(c2.state_per_linear_mib, 0.0)
            self.assertEqual(c2.calibrated_on_pool, 1000)


class TestDefaultPathUnchanged(CustomTestCase):
    def test_unset_touches_nothing(self):
        # THE REGRESSION THAT MATTERS. With the flag unset the handler must
        # return before reading a config, touching NVML, or setting a ratio.
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.__new__(ServerArgs)
        args.pp_solve_cut = None
        args.pp_layer_ratio = None
        args.pp_stage_ratio = None
        args.pp_size = 3
        # No other attribute is set: if the handler reads one, this raises
        # AttributeError and the test fails, which is the assertion.
        self.assertIsNone(ServerArgs._handle_pp_solve_cut(args))
        self.assertIsNone(args.pp_layer_ratio)

    def test_explicit_overrides_refuse_to_be_overruled(self):
        from sglang.srt.server_args import ServerArgs

        for field in ("pp_layer_ratio", "pp_stage_ratio"):
            args = ServerArgs.__new__(ServerArgs)
            args.pp_solve_cut = "/tmp/whatever"
            args.pp_size = 3
            args.pp_layer_ratio = None
            args.pp_stage_ratio = None
            setattr(args, field, [1, 2, 3])
            with self.assertRaises(ValueError) as cm:
                ServerArgs._handle_pp_solve_cut(args)
            self.assertIn(field.replace("_", "-"), str(cm.exception))

    def test_no_pipeline_is_refused(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.__new__(ServerArgs)
        args.pp_solve_cut = "/tmp/whatever"
        args.pp_size = 1
        args.pp_layer_ratio = None
        args.pp_stage_ratio = None
        with self.assertRaises(ValueError) as cm:
            ServerArgs._handle_pp_solve_cut(args)
        self.assertIn("no pipeline", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
