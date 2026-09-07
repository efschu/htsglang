"""The microbench's correctness gate must read the tensor the call returns.

Why this file exists (boot ``weg2bl1``, 2026-09-07). ``scripts/weg2/
barlink_round_bench.py`` is the artifact that replaces #1234's hand constant,
and spec section 6 makes it a precondition of C2: no widened round bound ships
until one decomposition above 16 rounds has EXECUTED with its result checked.
The shipped script could never pass that gate. ``BarlinkCommunicator.all_reduce``
is out-of-place -- "Returns a new tensor", ``barlink.py:1114-1119`` -- and the
gate called it for effect, discarded the return and asserted on the untouched
input. Every rank therefore read back its own ``rank + 1``, and the bench
aborted with ``WRONG RESULT ... 25 rounds -- expected 6.0, got 3.0``. The
decisive control: the same check fails at window 40 / 10 rounds, the operating
point that boot served 30 100 tokens through correctly. A benchmark whose
verification is wrong reports a transport defect that is not there -- and that
is the one failure mode a throughput bench cannot see.

CPU-only. No CUDA, no process group, no gpuq window: the fakes stand in for
the communicator, which is the whole point -- the gate must be exercised
without the hardware it grades.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


BENCH = (
    Path(__file__).resolve().parents[4] / "scripts" / "weg2"
    / "barlink_round_bench.py"
)


def _bench():
    """Load the script by path -- ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("barlink_round_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _OutOfPlace:
    """barlink's real contract: a fresh tensor, the input untouched."""

    def __init__(self, world):
        self.world = world

    def all_reduce(self, t):
        return torch.full_like(t, float(sum(range(1, self.world + 1))))


class _InPlace:
    """torch.distributed's contract: mutates and returns None."""

    def __init__(self, world):
        self.world = world

    def all_reduce(self, t):
        t.fill_(float(sum(range(1, self.world + 1))))
        return None


class _Broken:
    """A transport that answers with the input. The gate must SEE this."""

    def all_reduce(self, t):
        return t


class TestTheGateReadsTheReturn(CustomTestCase):

    def setUp(self):
        self.mod = _bench()
        self.world = 3
        self.buf = torch.full((64,), 1.0, dtype=torch.float32)

    def test_out_of_place_result_is_the_one_that_gets_checked(self):
        out = self.mod.verify_all_reduce(
            _OutOfPlace(self.world), self.buf, self.world, "w16 96 MiB 25 rounds"
        )
        self.assertTrue(bool(torch.all(out == 6.0).item()))
        self.assertTrue(bool(torch.all(self.buf == 1.0).item()),
                        msg="the gate must not depend on mutating the input")

    def test_in_place_transports_still_pass(self):
        out = self.mod.verify_all_reduce(
            _InPlace(self.world), self.buf, self.world, "in-place"
        )
        self.assertTrue(bool(torch.all(out == 6.0).item()))

    def test_a_genuinely_wrong_answer_still_aborts(self):
        with self.assertRaises(SystemExit) as e:
            self.mod.verify_all_reduce(
                _Broken(), self.buf, self.world, "w16 128 MiB 33 rounds"
            )
        msg = str(e.exception)
        self.assertIn("WRONG RESULT", msg)
        self.assertIn("w16 128 MiB 33 rounds", msg)

    def test_the_measured_path_uses_the_gate_and_not_a_bare_call(self):
        src = BENCH.read_text()
        self.assertIn("verify_all_reduce(", src)
        self.assertNotIn("            comm.all_reduce(probe)\n", src)


if __name__ == "__main__":
    unittest.main()
