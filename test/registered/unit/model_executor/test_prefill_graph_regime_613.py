# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#613: the prefill graph pays per REGIME, so the gate is per batch.

WHAT WAS MEASURED, and it is the whole basis for this module. Window 4 on
2026-08-05 (`3b4526c4ac`) ran prefill-eager vs prefill-graphs under barlink --
production's transport -- interleaved, ms per fixed unit of work, each point
against its own A-vs-A floor:

| point | eager | graphs | paired delta | floor | verdict |
|---|---|---|---|---|---|
| 1900 single-stream | 1196.3 ms | 1320.6 ms | **+10.25% slower** | 0.12% | REPORTABLE |
| 256 x4 concurrent | 230.1 ms | 219.0 ms | **-4.62% faster** | 1.99% | REPORTABLE |

All three pairs agreed in sign at both points, and both deltas cleared their
own floor by 85x and 2.3x respectively. So the captured prefill pays where the
work is launch-train bound and loses where it is GEMM bound, exactly as
predicted -- and the conclusion recorded there is that "the answer is not on/off
but workload-dependent, and a blanket flag is the wrong shape."

WHAT WAS NOT MEASURED, and must therefore not be asserted: anything between
those two points. Two points are not a curve. A gate that interpolated a
crossover would be inventing the number that decides every borderline batch,
which is the fitted-cell mistake. So this gate permits the graph ONLY inside
the regime that was measured to win, and refuses everything else BY NAME --
including the unmeasured middle, which is refused as unmeasured rather than as
slow.

DEFAULT OFF. With the lever unset the verdict is always "permit" and
``can_run_graph`` behaves exactly as it does today; the gate cannot change a
boot that did not ask for it.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from sglang.srt.model_executor.runner.prefill_graph_regime import (
    MEASURED_LOSS_TOKENS_PER_REQ,
    MEASURED_WIN_TOKENS_PER_REQ,
    PREFILL_GRAPH_REGIME_ENV,
    regime_enabled,
    regime_permits_graph,
)


def _on():
    return mock.patch.dict(os.environ, {PREFILL_GRAPH_REGIME_ENV: "1"})


def _off():
    return mock.patch.dict(os.environ, {PREFILL_GRAPH_REGIME_ENV: "0"})


class TestDefaultOff(unittest.TestCase):
    def test_unset_is_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PREFILL_GRAPH_REGIME_ENV, None)
            self.assertFalse(regime_enabled())

    def test_disabled_permits_everything(self):
        # Byte-identical to today: the gate must not refuse anything it was
        # not switched on for -- including the batch it would otherwise refuse.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PREFILL_GRAPH_REGIME_ENV, None)
            v = regime_permits_graph(batch_size=1, num_tokens=1974)
            self.assertTrue(v.permits)
            self.assertIn("disabled", v.reason)


class TestTheMeasuredWinIsPermitted(unittest.TestCase):
    def test_the_256x4_point_is_permitted(self):
        # The exact shape that measured -4.62%.
        with _on():
            v = regime_permits_graph(batch_size=4, num_tokens=256 * 4)
            self.assertTrue(v.permits, v.reason)

    def test_a_smaller_shorter_concurrent_batch_is_permitted(self):
        with _on():
            self.assertTrue(regime_permits_graph(batch_size=2, num_tokens=128).permits)


class TestTheMeasuredLossIsRefused(unittest.TestCase):
    def test_the_1900_single_stream_point_is_refused(self):
        # The exact shape that measured +10.25%.
        with _on():
            v = regime_permits_graph(batch_size=1, num_tokens=1974)
            self.assertFalse(v.permits)
            self.assertIn("single-stream", v.reason)

    def test_any_single_request_prefill_is_refused(self):
        # batch_size 1 has no launch train to recover, which is the mechanism
        # the measurement attributes the loss to.
        with _on():
            self.assertFalse(regime_permits_graph(batch_size=1, num_tokens=64).permits)


class TestTheUnmeasuredMiddleIsRefusedAsUNMEASURED(unittest.TestCase):
    """The point of the whole module: do not invent the crossover."""

    def test_a_concurrent_but_long_batch_is_refused(self):
        with _on():
            v = regime_permits_graph(batch_size=4, num_tokens=800 * 4)
            self.assertFalse(v.permits)
            self.assertIn("unmeasured", v.reason.lower())

    def test_the_refusal_names_both_bracketing_points(self):
        with _on():
            v = regime_permits_graph(batch_size=4, num_tokens=800 * 4)
            self.assertIn(str(MEASURED_WIN_TOKENS_PER_REQ), v.reason)
            self.assertIn(str(MEASURED_LOSS_TOKENS_PER_REQ), v.reason)

    def test_the_boundary_itself_is_permitted(self):
        # <= the measured win point is inside what was measured.
        with _on():
            self.assertTrue(
                regime_permits_graph(
                    batch_size=2, num_tokens=2 * MEASURED_WIN_TOKENS_PER_REQ
                ).permits
            )

    def test_one_token_above_the_measured_point_is_refused(self):
        with _on():
            self.assertFalse(
                regime_permits_graph(
                    batch_size=2, num_tokens=2 * MEASURED_WIN_TOKENS_PER_REQ + 2
                ).permits
            )


class TestDegenerateInputs(unittest.TestCase):
    def test_zero_batch_size_does_not_divide_by_zero(self):
        with _on():
            v = regime_permits_graph(batch_size=0, num_tokens=0)
            self.assertFalse(v.permits)


class TestTheRunnerConsultsTheGate(unittest.TestCase):
    """Pin the WIRING. A gate nothing calls is the #349 defect again."""

    def test_can_run_graph_calls_the_regime_gate(self):
        import inspect

        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        src = inspect.getsource(PrefillCudaGraphRunner.can_run_graph)
        self.assertIn("regime_permits_graph", src)


if __name__ == "__main__":
    unittest.main()
