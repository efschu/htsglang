"""#363 card-gate tooling -- the verdict trace and the two gate scripts.

Hermetic. The scripts carry their own ``--smoke``, and this file runs those
smokes in-process so they are part of the suite rather than something a human
has to remember to invoke: a gate tool that rots between the desk pass and the
window is worse than no tool, because it is discovered with the cards held.

What is pinned directly here is the trace format, because both gates read it
and one of its fields exists for a reason a reader would otherwise remove.
"""

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


def _drive(obs, rounds, *, prefill=True, held=40_000, ms=10.0):
    for _ in range(rounds):
        obs.on_round(
            prefill_active=prefill,
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


if __name__ == "__main__":
    unittest.main()
