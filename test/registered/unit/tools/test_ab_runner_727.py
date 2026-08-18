"""#727 A/B runner: every gate can fail, the floor is the baseline's own,
and the decision rule is the ticket's -- pinned before the window runs it.

Desk-written-never-executed rule: the runner boots nothing at the desk, so
this suite is what proves the orchestration ALIVE -- every gate driven in
both directions, the A-vs-A floor arithmetic, the stop-at-baseline-failure
branch, the verdict table (including the #735 slot consequence), and the
artifact verifier against synthetic checkpoints built to be wrong in each
of the ways it must catch.
"""

import json
import os
import struct
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../tools"))

import ab_vocab_int8_727 as ab  # noqa: E402

SHAPE_W = [248320, 5120]
SHAPE_S = [248320, 1]


def _write_shard(path, tensors):
    header = {
        name: {"dtype": dt, "shape": shape, "data_offsets": [0, 0]}
        for name, (dt, shape) in tensors.items()
    }
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def _build_artifacts(root, lmh_dtype_in_both="I8", break_hardlinks=False):
    """Synthetic B and C artifact dirs with the real shard/table shape."""
    emb = os.path.join(root, ab.ARM_MODEL["B"])
    both = os.path.join(root, ab.ARM_MODEL["C"])
    os.makedirs(emb)
    os.makedirs(both)
    for i in range(1, 19):
        name = f"model-{i:05d}-of-00018.safetensors"
        if i == 3:
            t = {
                "model.language_model.embed_tokens.weight": ("I8", SHAPE_W),
                "model.language_model.embed_tokens.weight_scale": ("BF16", SHAPE_S),
            }
            _write_shard(os.path.join(emb, name), t)
        elif i == 18:
            _write_shard(
                os.path.join(emb, name), {"lm_head.weight": ("BF16", SHAPE_W)}
            )
            _write_shard(
                os.path.join(both, name),
                {
                    "lm_head.weight": (lmh_dtype_in_both, SHAPE_W),
                    "lm_head.weight_scale": ("BF16", SHAPE_S),
                },
            )
        else:
            _write_shard(os.path.join(emb, name), {"filler": ("BF16", [1])})
        if i != 18:
            if break_hardlinks and i == 5:
                _write_shard(os.path.join(both, name), {"filler": ("BF16", [1])})
            else:
                os.link(os.path.join(emb, name), os.path.join(both, name))
    json.dump(
        {"quantization_config": {"ignore": ["lm_head"]}},
        open(os.path.join(emb, "config.json"), "w"),
    )
    json.dump(
        {"quantization_config": {"ignore": []}},
        open(os.path.join(both, "config.json"), "w"),
    )
    return root


class TestArtifactVerifier(CustomTestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="ab727-")
        self.addCleanup(self._rm)

    def _rm(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_correct_artifacts_pass(self):
        _build_artifacts(self.tmp)
        lines = ab.verify_artifacts(self.tmp)
        self.assertEqual(len(lines), 3)

    def test_a_bf16_lm_head_in_C_is_caught(self):
        """The exact defect a window must never boot into: arm C without its
        one distinguishing tensor."""
        _build_artifacts(self.tmp, lmh_dtype_in_both="BF16")
        with self.assertRaises(ab.GateFailure) as ctx:
            ab.verify_artifacts(self.tmp)
        self.assertIn("lm_head.weight", str(ctx.exception))

    def test_broken_hardlink_economy_is_caught(self):
        """B and C must differ in exactly one tensor, proven by inode
        sharing -- a re-written shard 5 would make deltas unattributable."""
        _build_artifacts(self.tmp, break_hardlinks=True)
        with self.assertRaises(ab.GateFailure) as ctx:
            ab.verify_artifacts(self.tmp)
        self.assertIn("hardlink", str(ctx.exception))

    def test_the_real_artifacts_pass_when_present(self):
        root = "/spinning/llm_stuff/club-3090/models-cache"
        if not os.path.isdir(os.path.join(root, ab.ARM_MODEL["C"])):
            self.skipTest("rig models-cache not present")
        lines = ab.verify_artifacts(root)
        self.assertEqual(len(lines), 3)


class TestGateZero(CustomTestCase):
    def test_counts_per_arm(self):
        one = "x\nINT8-VOCAB ENGAGED: 248320 x 5120\ny\n"
        ab.gate0_engaged("no lines here", "A1")
        ab.gate0_engaged(one, "B")
        ab.gate0_engaged(one * 2, "C")

    def test_silent_dense_fallback_fails_B(self):
        with self.assertRaises(ab.GateFailure):
            ab.gate0_engaged("booted fine, nothing engaged", "B")

    def test_engagement_on_the_baseline_fails_A(self):
        """The other direction: int8 engaging on the BF16 baseline means the
        arms are not what they claim."""
        with self.assertRaises(ab.GateFailure):
            ab.gate0_engaged("INT8-VOCAB ENGAGED: oops", "A1")

    def test_C_needs_both_engagements(self):
        with self.assertRaises(ab.GateFailure):
            ab.gate0_engaged("INT8-VOCAB ENGAGED: only one", "C")


class TestGateA(CustomTestCase):
    BASE = {"pp0": 20000.0, "pp1": 15000.0, "pp2": 21000.0}

    def test_B_passes_with_pp0_saving_only(self):
        vram = {"pp0": 20000.0 - 1212, "pp1": 15000.0, "pp2": 21000.0}
        ab.gateA_vram(vram, self.BASE, "B", tol_mib=200)

    def test_B_fails_when_the_saving_did_not_materialize(self):
        ab_vram = dict(self.BASE)
        with self.assertRaises(ab.GateFailure):
            ab.gateA_vram(ab_vram, self.BASE, "B", tol_mib=200)

    def test_C_needs_both_stages(self):
        vram = {"pp0": 20000.0 - 1212, "pp1": 15000.0, "pp2": 21000.0}
        with self.assertRaises(ab.GateFailure) as ctx:
            ab.gateA_vram(vram, self.BASE, "C", tol_mib=200)
        self.assertIn("pp2", str(ctx.exception))

    def test_C_passes_with_both(self):
        vram = {"pp0": 20000.0 - 1212, "pp1": 15000.0, "pp2": 21000.0 - 1212}
        ab.gateA_vram(vram, self.BASE, "C", tol_mib=200)


def _metrics(score, det, ttft=800.0, tps=20.0):
    return {"score": score, "determined": det, "ttft_ms": ttft, "decode_tps": tps}


class TestFloorAndGateB(CustomTestCase):
    def test_delta_inside_the_a_floor_passes(self):
        a1, a2 = _metrics(0.80, 0.90), _metrics(0.78, 0.88)
        floor = ab.Floor.from_arms(a1, a2, eps_score=0.0, eps_perf_frac=0.0)
        ab.gateB_quality(_metrics(0.785, 0.885), a1, floor, "B")

    def test_delta_outside_the_floor_fails(self):
        a1, a2 = _metrics(0.80, 0.90), _metrics(0.80, 0.90)
        floor = ab.Floor.from_arms(a1, a2, eps_score=0.0, eps_perf_frac=0.0)
        with self.assertRaises(ab.GateFailure):
            ab.gateB_quality(_metrics(0.75, 0.90), a1, floor, "C")

    def test_the_floor_is_the_baselines_own_variance(self):
        """A noisy baseline WIDENS what B/C may show -- the ticket's reason
        for running A twice. The same 0.05 drop that fails against identical
        baselines passes when A itself moved 0.06 between boots."""
        a1, a2 = _metrics(0.80, 0.90), _metrics(0.74, 0.90)
        floor = ab.Floor.from_arms(a1, a2, eps_score=0.0, eps_perf_frac=0.0)
        ab.gateB_quality(_metrics(0.75, 0.90), a1, floor, "C")

    def test_better_than_baseline_never_fails(self):
        a1, a2 = _metrics(0.80, 0.90), _metrics(0.80, 0.90)
        floor = ab.Floor.from_arms(a1, a2, eps_score=0.0, eps_perf_frac=0.0)
        ab.gateB_quality(_metrics(0.85, 0.95), a1, floor, "B")


class TestGateC(CustomTestCase):
    def test_perf_within_floor_passes(self):
        a1, a2 = _metrics(0.8, 0.9, 800, 20.0), _metrics(0.8, 0.9, 850, 19.5)
        floor = ab.Floor.from_arms(a1, a2, eps_score=0.0, eps_perf_frac=0.0)
        ab.gateC_perf(_metrics(0.8, 0.9, 845, 19.6), a1, floor, "C")

    def test_a_decode_regression_beyond_floor_fails(self):
        a1, a2 = _metrics(0.8, 0.9, 800, 20.0), _metrics(0.8, 0.9, 800, 20.0)
        floor = ab.Floor.from_arms(a1, a2, eps_score=0.0, eps_perf_frac=0.0)
        with self.assertRaises(ab.GateFailure) as ctx:
            ab.gateC_perf(_metrics(0.8, 0.9, 800, 17.0), a1, floor, "C")
        self.assertIn("decode", str(ctx.exception))


class TestDecisionRule(CustomTestCase):
    """The ticket's table, verbatim shape, with the #735 consequence."""

    def test_both_pass_ships_both_and_funds_735(self):
        v = ab.decision(b_passed=True, c_passed=True)
        self.assertIn("SHIP BOTH", v)
        self.assertIn("REFUTED", v)
        self.assertIn("21-24", v)

    def test_c_fails_ships_embed_only_and_defunds_735(self):
        v = ab.decision(b_passed=True, c_passed=False)
        self.assertIn("SHIP EMBED-ONLY", v)
        self.assertIn("CONFIRMED", v)
        self.assertIn("loses its funder", v)

    def test_b_fails_stops_everything(self):
        v = ab.decision(b_passed=False, c_passed=False)
        self.assertIn("STOP", v)
        # C's result cannot rescue a failed B: the gather half is the
        # precondition of the whole scheme.
        self.assertEqual(v, ab.decision(b_passed=False, c_passed=True))


class TestMockedEndToEnd(CustomTestCase):
    """The CLI smoke the desk is allowed: fixtures instead of boots, real
    orchestration, real verdict, results JSON written."""

    def _fixture(self, tmp, arm, engaged, pp0, pp2, score):
        data = {
            "boot_log": "INT8-VOCAB ENGAGED\n" * engaged,
            "vram": {"pp0": pp0, "pp1": 15000.0, "pp2": pp2},
            "metrics": _metrics(score, 0.90),
            "perf": _metrics(score, 0.90),
        }
        json.dump(data, open(os.path.join(tmp, f"{arm}.json"), "w"))

    def test_full_mock_run_reaches_ship_both(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ab727-e2e-") as tmp:
            root = os.path.join(tmp, "models")
            os.makedirs(root)
            _build_artifacts(root)
            self._fixture(tmp, "A1", 0, 20000, 21000, 0.80)
            self._fixture(tmp, "A2", 0, 20000, 21000, 0.79)
            self._fixture(tmp, "B", 1, 20000 - 1212, 21000, 0.795)
            self._fixture(tmp, "C", 2, 20000 - 1212, 21000 - 1212, 0.795)
            results = os.path.join(tmp, "out.json")
            tool = os.path.join(
                os.path.dirname(ab.__file__), "ab_vocab_int8_727.py"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    tool,
                    "--models-root",
                    root,
                    "--mock",
                    tmp,
                    "--results",
                    results,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            out = json.load(open(results))
            self.assertIn("SHIP BOTH", out["verdict"])
            self.assertEqual(len(out["arms"]), 4)

    def test_baseline_failure_aborts_the_run(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ab727-abort-") as tmp:
            root = os.path.join(tmp, "models")
            os.makedirs(root)
            _build_artifacts(root)
            # A1's boot log shows an engagement -- the arms are mislabeled;
            # nothing below is readable and the window must stop.
            self._fixture(tmp, "A1", 1, 20000, 21000, 0.80)
            for arm, engaged in (("A2", 0), ("B", 1), ("C", 2)):
                self._fixture(tmp, arm, engaged, 20000, 21000, 0.80)
            results = os.path.join(tmp, "out.json")
            tool = os.path.join(
                os.path.dirname(ab.__file__), "ab_vocab_int8_727.py"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    tool,
                    "--models-root",
                    root,
                    "--mock",
                    tmp,
                    "--results",
                    results,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 1)
            out = json.load(open(results))
            self.assertIn("ABORT", out["verdict"])


if __name__ == "__main__":
    unittest.main()
