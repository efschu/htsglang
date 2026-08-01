"""#363 card-gate tooling -- the verdict trace and the two gate scripts.

Hermetic. The scripts carry their own ``--smoke``, and this file runs those
smokes in-process so they are part of the suite rather than something a human
has to remember to invoke: a gate tool that rots between the desk pass and the
window is worse than no tool, because it is discovered with the cards held.

What is pinned directly here is the trace format, because both gates read it
and one of its fields exists for a reason a reader would otherwise remove.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

from sglang.srt.managers.regime_runtime import MODE_OBSERVE, RegimeObserver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SCRIPTS = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "regime_gates"
    )
)
_PYTHON_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
)


def _load_bands():
    """``scripts/regime_gates/bands.py`` as a module.

    Not on the import path (it is a script, not a package), and the
    distributional statistic has to be pinned directly rather than only
    through the script's own exit code.
    """
    spec = importlib.util.spec_from_file_location(
        "_gate3_bands", os.path.join(_SCRIPTS, "bands.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drive(obs, rounds, *, prefill=True, held=40_000, ms=10.0):
    for _ in range(rounds):
        obs.on_round(
            phase="prefill" if prefill else "decode",
            held_tokens=held,
            capacity_tokens=453_632,
            running_bs=1,
            rank_forward_ms=ms,
        )


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestVerdictTrace(CustomTestCase):
    def test_the_trace_is_header_then_verdicts_then_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            obs = RegimeObserver(
                consensus_interval=2, tp_size=1, mode=MODE_OBSERVE, trace_path=path
            )
            _drive(obs, 8)
            obs.close_trace()
            rows = _read(path)
        self.assertEqual(rows[0]["kind"], "header")
        self.assertEqual(rows[0]["mode"], MODE_OBSERVE)
        self.assertEqual(rows[-1]["kind"], "summary")
        self.assertTrue(all(r["kind"] == "verdict" for r in rows[1:-1]))
        self.assertEqual(len(rows) - 2, 4)

    def test_the_summary_is_the_last_line_and_carries_the_gate_1_number(self):
        """Its presence is what separates 'zero desyncs' from 'zero so far'."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            obs = RegimeObserver(
                consensus_interval=2, tp_size=1, mode=MODE_OBSERVE, trace_path=path
            )
            _drive(obs, 4)
            obs.close_trace()
            rows = _read(path)
        self.assertIn("desyncs", rows[-1])
        self.assertIn("actuations", rows[-1])
        self.assertEqual(rows[-1]["actuations"], 0)

    def test_a_verdict_carries_the_numerator_not_only_the_ratio(self):
        """Gate 2's counterfactual varies the capacity DENOMINATOR. A trace
        holding only ``occupancy`` would hold occupancy constant under exactly
        the change it exists to expose, and every counterfactual would come
        back clean for the wrong reason. Found by the gate-2 tool's own smoke;
        this is the pin that keeps the field."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            obs = RegimeObserver(
                consensus_interval=2, tp_size=1, mode=MODE_OBSERVE, trace_path=path
            )
            _drive(obs, 2, held=90_000)
            obs.close_trace()
            verdict = _read(path)[1]
        self.assertEqual(verdict["held_tokens"], 90_000)
        self.assertEqual(verdict["capacity_tokens"], 453_632)
        self.assertAlmostEqual(verdict["occupancy"], 90_000 / 453_632)

    def test_no_trace_path_writes_nothing_and_costs_nothing(self):
        obs = RegimeObserver(consensus_interval=2, tp_size=1, mode=MODE_OBSERVE)
        _drive(obs, 8)
        obs.close_trace()  # idempotent, and a no-op here

    def test_an_unwritable_path_disables_the_trace_and_never_raises(self):
        """An observability feature must not be able to stop a server."""
        obs = RegimeObserver(
            consensus_interval=2,
            tp_size=1,
            mode=MODE_OBSERVE,
            trace_path="/nonexistent-dir-363/trace.jsonl",
        )
        _drive(obs, 8)
        self.assertEqual(obs.summary()["verdicts"], 4)

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            obs = RegimeObserver(
                consensus_interval=2, tp_size=1, mode=MODE_OBSERVE, trace_path=path
            )
            _drive(obs, 4)
            obs.close_trace()
            obs.close_trace()
            self.assertEqual(sum(1 for r in _read(path) if r["kind"] == "summary"), 1)


class TestGateToolSmokes(CustomTestCase):
    """The scripts' own card-less smokes, run as part of the suite."""

    def _run(self, script, *args):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = _PYTHON_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, os.path.join(_SCRIPTS, script), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        return proc

    def test_the_workload_dry_run_names_all_four_shapes(self):
        proc = self._run("workload.py", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for phase in ("prefill_burst", "decode_drain", "idle", "mixed"):
            self.assertIn(phase, proc.stdout)
        self.assertIn("no socket opened", proc.stdout)

    def test_the_gate_1_readout_smoke_passes(self):
        proc = self._run("readout.py", "--smoke")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SMOKE OK", proc.stdout)
        # The pass case round-trips through the gate that will read it.
        self.assertIn("desyncs_zero accepted=True", proc.stdout)

    def test_the_gate_2_replay_smoke_reproduces_the_f2_result(self):
        """The unguarded counterfactual manufactures kv_pressure and the
        guarded one does not -- the phase-2 finding, through the tool that
        will be pointed at a live trace."""
        proc = self._run("f2_replay.py", "--smoke")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SMOKE OK", proc.stdout)
        self.assertIn("interlock_load_bearing=True", proc.stdout)
        self.assertIn("kv_pressure", proc.stdout)

    def test_the_readout_refuses_a_trace_it_cannot_attribute(self):
        """End to end through the real script: a run with one regime is not
        evidence, and the refusal says why."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "flat.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"kind": "header", "mode": "observe"}) + "\n")
                f.write(
                    json.dumps({"kind": "verdict", "round": 8, "regime": "mixed"})
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "kind": "summary",
                            "verdicts": 1,
                            "desyncs": 0,
                            "uncoordinated": False,
                        }
                    )
                    + "\n"
                )
            proc = self._run("readout.py", "--trace", path)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("GATE 1 NOT PASSED", proc.stdout)

    def test_a_trace_this_observer_wrote_replays_through_the_gate_2_tool(self):
        """The two halves meet: the writer's real output is the reader's real
        input. A format drift between them would show up here and nowhere
        else until the window."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            obs = RegimeObserver(
                consensus_interval=2, tp_size=1, mode=MODE_OBSERVE, trace_path=path
            )
            _drive(obs, 20, prefill=True, held=90_000)
            _drive(obs, 20, prefill=False, held=90_000)
            obs.close_trace()
            proc = self._run("f2_replay.py", "--trace", path)
            self.assertIn("recorded", proc.stdout)
            self.assertNotIn("cannot be replayed", proc.stdout)

    def test_the_gate_3_band_smoke_passes(self):
        proc = self._run("bands.py", "--smoke")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SMOKE OK", proc.stdout)

    def test_the_gate_3_falsifier_passes_and_shows_the_old_false_alarm(self):
        """The three cases that justify the distributional statistic.

        The first one is the false alarm of record and it must still be
        visible: the retained pointwise path reports ``ARMS_DISSIMILAR`` on
        two arms with the SAME duty cycle whose bursts landed on different
        boundary indices, which is what two boots of one workload always do.
        """
        proc = self._run("bands.py", "--falsify")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("FALSIFIER OK", proc.stdout)
        self.assertIn("OLD pointwise : band=1  ARMS_DISSIMILAR", proc.stdout)


class TestGate3DistributionalStatistic(CustomTestCase):
    """#363 gate 3: the statistic is chosen per signal, and both guards live.

    The pointwise band assumes the value at position ``i`` describes the same
    thing in both runs. #388 made the shares near-binary at
    ``window_rounds = 64``, so bursts land on different boundary indices --
    they must, the re-record's arms had 41 and 56 active boundaries -- and the
    pointwise band went to 1 on every signal. These pin the replacement.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bands = _load_bands()

    def test_every_signal_is_classified(self):
        """A new classifier input must not silently inherit a statistic."""
        self.assertEqual(set(self.bands.SIGNALS), set(self.bands.STATISTIC))
        self.assertEqual(
            self.bands.STATISTIC["rank_ms_spread_pct"],
            self.bands.STATISTIC_POINTWISE,
            "the pointwise path is kept, not blanket-replaced",
        )
        for name in ("prefill_share", "decode_share", "occupancy"):
            self.assertEqual(
                self.bands.STATISTIC[name], self.bands.STATISTIC_DISTRIBUTIONAL
            )

    def test_the_duty_cycle_is_the_fraction_at_or_above_the_threshold(self):
        self.assertEqual(self.bands.duty_cycle([0, 0, 1, 1], 0.35), 0.5)
        self.assertEqual(self.bands.duty_cycle([0.35, 0.34], 0.35), 0.5)
        self.assertEqual(self.bands.duty_cycle([], 0.35), 0.0)

    def test_a_shifted_burst_with_the_same_duty_is_not_a_difference(self):
        """The exact false alarm of record, as an assertion.

        Ten prefill-heavy windows out of forty in both arms, early in one and
        late in the other. Pointwise this is the maximum possible band; the
        thing the gate actually asks -- how often does the signal cross 0.35 --
        is identical.
        """
        a = self.bands._bursts(40, range(0, 10))
        b = self.bands._bursts(40, range(20, 30))
        va = [r["prefill_share"] for r in a]
        vb = [r["prefill_share"] for r in b]
        self.assertEqual(
            self.bands.signal_band(va, vb), 1.0, "pointwise: maximally different"
        )
        self.assertEqual(self.bands.duty_cycle(va, 0.35), 0.25)
        self.assertEqual(self.bands.duty_cycle(vb, 0.35), 0.25)
        worst = self.bands.duty_disagreement(va, vb, [0.35, 0.15])
        self.assertEqual(worst["delta"], 0.0)
        self.assertLess(worst["z"], self.bands.DISSIMILAR_Z)

    def test_a_real_duty_difference_still_trips_arms_dissimilar(self):
        """The guard is re-stated for a distribution, not dropped."""
        va = [r["prefill_share"] for r in self.bands._bursts(40, range(0, 10))]
        vb = [r["prefill_share"] for r in self.bands._bursts(40, range(0, 30))]
        worst = self.bands.duty_disagreement(va, vb, [0.35, 0.15])
        self.assertAlmostEqual(worst["delta"], 0.5)
        self.assertGreater(worst["z"], self.bands.DISSIMILAR_Z)

    def test_the_guard_reads_the_constants_thresholds_and_not_every_value(self):
        """Why not a sup-over-all-thresholds (Kolmogorov-Smirnov) form.

        Two arms each holding steady at slightly different levels are totally
        separated at any threshold between them, so a KS-style guard calls
        them incomparable -- but that offset is a real, reproducible bias and
        it IS the band. At the constants' own thresholds they agree, and the
        0.03 offset survives as the measurement.
        """
        va, vb = [0.50] * 20, [0.53] * 20
        between = self.bands.duty_disagreement(va, vb, [0.515])
        self.assertEqual(between["delta"], 1.0)  # what a KS form would see
        at_constants = self.bands.duty_disagreement(va, vb, [0.90, 0.70])
        self.assertEqual(at_constants["delta"], 0.0)
        self.assertEqual(self.bands.thresholds_for("decode_share"), [0.90, 0.70])

    def test_a_barely_moving_signal_reports_the_same_band_as_before(self):
        """Regression pin against the traces the first version was built on.

        The distributional band is the peak disagreement, and for a flat
        signal that is the number the pointwise band reported.
        """
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.bands._pair(
                tmp,
                [{"decode_share": 0.50, "prefill_share": None} for _ in range(20)],
                [{"decode_share": 0.53, "prefill_share": None} for _ in range(20)],
            )
            new = self.bands.report(a, b)["bands"]["decode_share"]
            old = self.bands._old_pointwise(a, b, "decode_share")
        self.assertAlmostEqual(new["band"], 0.03)
        self.assertAlmostEqual(old["band"], 0.03)
        self.assertEqual(new["status"], old["status"])

    def test_underpowered_still_blocks_under_the_new_statistic(self):
        """A distribution from two windows is a number, not a measurement."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.bands._pair(
                tmp, self.bands._bursts(40, range(0, 10)), self.bands._bursts(2, [0])
            )
            rep = self.bands.report(a, b)
        self.assertEqual(rep["bands"]["prefill_share"]["status"], "UNDERPOWERED")
        self.assertFalse(rep["passed"])

    def test_a_crossing_rate_smaller_than_its_own_disagreement_is_inside_band(self):
        """The verdict the duty check exists to produce.

        One arm never crosses 0.35 and the other crosses in two windows out of
        forty. The threshold is reachable, and the disagreement is small enough
        in absolute terms that the comparability guard stays quiet -- but the
        crossing rate is smaller than twice its own run-to-run band, so the
        decision does not reproduce and the verdict is ``INSIDE_BAND``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.bands._pair(
                tmp, self.bands._bursts(40, []), self.bands._bursts(40, [0, 1])
            )
            rep = self.bands.report(a, b)
        self.assertEqual(rep["bands"]["prefill_share"]["status"], "OK")
        vp = next(v for v in rep["constants"] if v["constant"] == "enter_prefill")
        self.assertEqual(vp["verdict"], "INSIDE_BAND")

    def test_every_signal_is_read_on_the_active_boundaries_only(self):
        """Including ``rank_ms_spread_pct``, which goes stale rather than absent.

        An idle window carries no forward. The shares are ``None`` there and
        drop out by themselves; the other three report a real or stale value
        and used to drag the band with it.
        """
        rows = [
            {"prefill_share": 1.0, "decode_share": 0.0, "rank_ms_spread_pct": 3.0},
            {"prefill_share": None, "decode_share": None, "rank_ms_spread_pct": 99.0},
        ]
        self.assertEqual(self.bands.series(rows, "rank_ms_spread_pct"), [3.0])
        self.assertEqual(self.bands.series(rows, "prefill_share"), [1.0])


if __name__ == "__main__":
    unittest.main()
