# SPDX-License-Identifier: Apache-2.0
"""#489 (c) / #726 -- the harness judged without a GPU.

Desk prep, per the desk-written-never-executed rule: everything that can be
checked without a card is checked here. What CANNOT be checked here is stated
in the window ticket rather than implied -- no kernel numerics are validated by
these tests, because that needs the silicon the ticket exists to book.

Two families:

* TOOLCHAIN -- the instruction the whole ticket rests on really does assemble
  for both of this rig's SM versions, on THIS toolchain. Slot-2's #726 note
  established that it assembles; this pins it so a CUDA upgrade that drops
  sm_120a, or a PTX-version regression, fails here rather than in the window.
* DECISION -- the rules that turn measurements into BUILD or DECLINE, driven
  with synthetic results including ones that MUST decline.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "bench", "489c")
)

import decision as D  # noqa: E402

PTXAS = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
CU = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "bench", "489c", "qk_arms.cu"
)

PTX_TEMPLATE = """.version {ver}
.target {arch}
.address_size 64
.visible .entry probe(.param .u64 p)
{{
  .reg .b32 a<4>, b<2>, c<4>, d<4>;
  mov.b32 a0, 0; mov.b32 a1, 0; mov.b32 a2, 0; mov.b32 a3, 0;
  mov.b32 b0, 0; mov.b32 b1, 0;
  mov.b32 c0, 0; mov.b32 c1, 0; mov.b32 c2, 0; mov.b32 c3, 0;
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
      {{d0,d1,d2,d3}}, {{a0,a1,a2,a3}}, {{b0,b1}}, {{c0,c1,c2,c3}};
  ret;
}}
"""


def _assemble(arch: str, ver: str):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "p.ptx")
        with open(src, "w") as fh:
            fh.write(PTX_TEMPLATE.format(arch=arch, ver=ver))
        return subprocess.run(
            [PTXAS, f"-arch={arch}", src, "-o", os.path.join(td, "p.cubin")],
            capture_output=True,
            text=True,
        )


@unittest.skipUnless(os.path.exists(PTXAS), "no ptxas on this host")
class TestTheInstructionAssembles(unittest.TestCase):
    """No GPU needed: assembling is a toolchain question, not a device one."""

    def test_imma_m16n8k32_s8_assembles_for_sm_86(self):
        """Both 3080s. IMMA is native here since sm_75 -- this is the arm that
        the published bf16-dequant lane never used."""
        self.assertEqual(_assemble("sm_86", "8.0").returncode, 0)

    def test_imma_m16n8k32_s8_assembles_for_sm_120a(self):
        """The 5090."""
        self.assertEqual(_assemble("sm_120a", "8.7").returncode, 0)

    def test_the_PTX_VERSION_FLOOR_DIFFERS_between_the_two_targets(self):
        """A finding from building this, pinned so the harness cannot regress
        into emitting one version for both.

        sm_86 assembles at PTX ISA 8.0; sm_120a REFUSES it and needs >= 8.7. A
        harness that emitted 8.0 for both would fail on the 5090 for a reason
        that has nothing to do with the instruction, and the window would be
        spent debugging the emitter.
        """
        low = _assemble("sm_120a", "8.0")
        self.assertNotEqual(low.returncode, 0)
        self.assertIn("does not support", (low.stderr or "") + (low.stdout or ""))
        self.assertEqual(_assemble("sm_120a", "8.7").returncode, 0)


@unittest.skipUnless(os.path.exists(NVCC), "no nvcc on this host")
class TestTheArmsCompile(unittest.TestCase):
    def _cubin(self, arch):
        with tempfile.TemporaryDirectory() as td:
            return subprocess.run(
                [NVCC, "-cubin", f"-arch={arch}", CU, "-o", os.path.join(td, "a.cubin")],
                capture_output=True,
                text=True,
            )

    def test_all_three_arms_compile_for_sm_86(self):
        r = self._cubin("sm_86")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])

    def test_all_three_arms_compile_for_sm_120a(self):
        r = self._cubin("sm_120a")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])


# --------------------------------------------------------------- decision ---


def _pt(card, sm, depth, arm, ms, shape="native", rms=None, batch=1, secs=12.0):
    return D.ArmResult(card, sm, depth, batch, arm, ms, secs, shape, rms)


def _sweep(card, sm, int8_ms, fp8_ms, depths=D.SPEC_DEPTHS, rms=0.005):
    out = []
    for d in depths:
        out.append(_pt(card, sm, d, "int8_imma", int8_ms, "imma_s32", rms))
        out.append(_pt(card, sm, d, "fp8_deployed", fp8_ms, "dequant_hmma"))
    return out


class TestTheBuildRule(unittest.TestCase):
    def test_two_winning_cards_out_of_three_BUILDS(self):
        res = (
            _sweep("3080a", "sm_86", 1.0, 2.0)
            + _sweep("3080b", "sm_86", 1.0, 2.0)
            + _sweep("5090", "sm_120", 2.0, 1.0)
        )
        rep = D.evaluate(res)
        self.assertTrue(rep["build"])
        self.assertEqual(rep["verdict"], "BUILD")

    def test_one_winning_card_DECLINES(self):
        """The count branch, isolated from the kill branch on purpose.

        The two losing cards lose only at 1K, not at 58K: an sm_86 loss AT 58K
        would fire the spec kill instead, and then this test would be passing
        for the wrong reason. Written the first way, it did exactly that.
        """
        res = _sweep("3080a", "sm_86", 1.0, 2.0)
        for card, sm in (("3080b", "sm_86"), ("5090", "sm_120")):
            sweep = _sweep(card, sm, 1.0, 2.0)
            sweep = [r for r in sweep if r.depth != 1_024]
            sweep += [
                _pt(card, sm, 1_024, "int8_imma", 2.0, "imma_s32", 0.005),
                _pt(card, sm, 1_024, "fp8_deployed", 1.0),
            ]
            res += sweep
        rep = D.evaluate(res)
        self.assertFalse(rep["kill_condition_fired"], "must isolate the count branch")
        self.assertFalse(rep["build"])
        self.assertIn("fewer than the two required", rep["verdict"])

    def test_a_single_losing_DEPTH_disqualifies_a_card(self):
        """'Wins at ALL depths' is the rule; the deep end is the whole point."""
        res = _sweep("3080a", "sm_86", 1.0, 2.0)
        res = [r for r in res if r.depth != 131_072]
        res += [
            _pt("3080a", "sm_86", 131_072, "int8_imma", 2.0, "imma_s32", 0.005),
            _pt("3080a", "sm_86", 131_072, "fp8_deployed", 1.0),
        ]
        cards = D.per_card(res)
        self.assertFalse(cards["3080a"]["wins_at_all_depths"])

    def test_a_gain_UNDER_THE_NOISE_FLOOR_is_not_a_win(self):
        res = _sweep("3080a", "sm_86", 1.0, 1.10)  # +10%, floor is 14.1%
        self.assertFalse(D.per_card(res)["3080a"]["wins_at_all_depths"])
        res2 = _sweep("3080a", "sm_86", 1.0, 1.20)  # +20%
        self.assertTrue(D.per_card(res2)["3080a"]["wins_at_all_depths"])


class TestTheSpecKillCondition(unittest.TestCase):
    def test_the_58K_inversion_on_sm86_FIRES_the_kill(self):
        res = _sweep("3080a", "sm_86", 1.0, 2.0)
        res = [r for r in res if r.depth != D.INVERSION_DEPTH]
        res += [
            _pt("3080a", "sm_86", D.INVERSION_DEPTH, "int8_imma", 3.0, "imma_s32", 0.005),
            _pt("3080a", "sm_86", D.INVERSION_DEPTH, "fp8_deployed", 1.0),
        ]
        self.assertTrue(D.kill_condition_fired(res))

    def test_the_same_inversion_on_sm120_does_NOT_fire_it(self):
        """The spec's kill is about sm_86 specifically -- two of three cards."""
        res = _sweep("5090", "sm_120", 1.0, 2.0)
        res = [r for r in res if r.depth != D.INVERSION_DEPTH]
        res += [
            _pt("5090", "sm_120", D.INVERSION_DEPTH, "int8_imma", 3.0, "imma_s32", 0.005),
            _pt("5090", "sm_120", D.INVERSION_DEPTH, "fp8_deployed", 1.0),
        ]
        self.assertFalse(D.kill_condition_fired(res))

    def test_a_WASH_at_58K_is_not_a_reproduction(self):
        res = _sweep("3080a", "sm_86", 1.0, 1.0)
        self.assertFalse(D.kill_condition_fired(res))

    def test_the_kill_OVERRIDES_a_passing_build_rule(self):
        """The disagreement the two rules can produce, made explicit: sm_120
        carrying an sm_86 inversion is exactly what #489 forbids reporting as
        a win."""
        res = (
            _sweep("3080a", "sm_86", 1.0, 2.0)
            + _sweep("5090", "sm_120", 1.0, 2.0)
        )
        res = [r for r in res if not (r.card == "3080a" and r.depth == D.INVERSION_DEPTH)]
        res += [
            _pt("3080a", "sm_86", D.INVERSION_DEPTH, "int8_imma", 3.0, "imma_s32", 0.005),
            _pt("3080a", "sm_86", D.INVERSION_DEPTH, "fp8_deployed", 1.0),
        ]
        rep = D.evaluate(res)
        self.assertTrue(rep["kill_condition_fired"])
        self.assertFalse(rep["build"])
        self.assertIn("kill condition", rep["verdict"])


class TestTheHarnessRefusesUnjudgeableRuns(unittest.TestCase):
    def test_runs_under_ten_seconds_are_refused(self):
        res = _sweep("3080a", "sm_86", 1.0, 2.0)
        res[0] = dataclasses_replace(res[0], seconds_measured=2.0)
        with self.assertRaises(D.BenchError) as e:
            D.validate(res)
        self.assertIn("ten seconds", str(e.exception))

    def test_an_AGGREGATED_card_entry_is_refused(self):
        res = _sweep("mean", "sm_86", 1.0, 2.0)
        with self.assertRaises(D.BenchError) as e:
            D.validate(res)
        self.assertIn("forbids averaging", str(e.exception))

    def test_an_arm_that_did_not_report_its_shape_is_refused(self):
        res = _sweep("3080a", "sm_86", 1.0, 2.0)
        res[0] = dataclasses_replace(res[0], selected_shape="")
        with self.assertRaises(D.BenchError) as e:
            D.validate(res)
        self.assertIn("inferred", str(e.exception))

    def test_accuracy_outside_the_codec_bound_DECLINES(self):
        res = _sweep("3080a", "sm_86", 1.0, 2.0, rms=0.05)
        res += _sweep("3080b", "sm_86", 1.0, 2.0, rms=0.05)
        rep = D.evaluate(res, heavy_tail=True)
        self.assertFalse(rep["build"])
        self.assertIn("accuracy", rep["verdict"])

    def test_the_heavy_tail_bound_is_the_looser_one(self):
        res = _sweep("3080a", "sm_86", 1.0, 2.0, rms=0.010)
        self.assertTrue(D.accuracy_ok(res, heavy_tail=True))
        self.assertFalse(D.accuracy_ok(res, heavy_tail=False))

    def test_the_spec_sweep_keeps_58K_and_the_deep_end(self):
        self.assertIn(58_000, D.SPEC_DEPTHS)
        self.assertIn(327_680, D.SPEC_DEPTHS)
        self.assertEqual(D.SPEC_BATCHES, (1, 4))


class TestTheReportNeverAverages(unittest.TestCase):
    def test_render_names_every_card_separately(self):
        res = _sweep("3080a", "sm_86", 1.0, 2.0) + _sweep("5090", "sm_120", 1.0, 2.0)
        text = D.render(D.evaluate(res))
        self.assertIn("3080a (sm_86)", text)
        self.assertIn("5090 (sm_120)", text)
        self.assertNotIn("average", text.lower())


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


if __name__ == "__main__":
    unittest.main()


class TestTheRunnerIsActuallyRunnable(unittest.TestCase):
    """Caught by running it: an edit removed the ``__main__`` guard, so the
    script defined main() and exited 0 in silence -- a turnkey ticket that
    turns nothing. Pinned, because a runner that no-ops looks like success."""

    RUNNER = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "bench", "489c",
        "run_489c.py",
    )

    def test_the_module_calls_main(self):
        with open(self.RUNNER) as fh:
            src = fh.read()
        self.assertIn('if __name__ == "__main__":', src)
        self.assertIn("raise SystemExit(main())", src)

    def test_the_required_geometry_has_no_silent_default(self):
        """--q-heads is the uneven-TP shard fact; defaulting it would change
        every arm's arithmetic intensity without saying so."""
        with open(self.RUNNER) as fh:
            src = fh.read()
        self.assertIn('"--q-heads", type=int, help', src)
        self.assertIn("refuses to invent it", src)

    def test_the_ptx_floor_table_covers_both_of_this_rigs_targets(self):
        sys.path.insert(0, os.path.dirname(self.RUNNER))
        import run_489c  # noqa: PLC0415

        self.assertEqual(run_489c.ARCH_PTX_FLOOR["sm_86"], "8.0")
        self.assertEqual(run_489c.ARCH_PTX_FLOOR["sm_120a"], "8.7")
