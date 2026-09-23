"""H3 (23.09.): the prefill timing instruments must flush on EVERY pipeline stage.

Bug (black box): with SGLANG_MOE_OFFLOAD_TIMING=1 a PP3 boot logged
MOE-OFFLOAD-TIMING-PREFILL and ATTN-TIMING-PREFILL on stage 0 only. fn7t
(19.09.) has 100 such lines, all PP0. The flush waited for "layer 0 again",
and only stage 0 runs layer 0 -- stage 1 of the Next-Flash P group starts at
layer 29, stage 2 at layer 40. Those stages never logged a line and kept
appending CUDA events forever, so the per-stage decomposition of a chunk
(expert stream / MoE GEMM / attention) did not exist for the stages whose
balance decides the pipeline.

The cases drive the real flush functions with a stage that starts at layer 29
and assert that the SECOND forward's first layer emits the FIRST forward's
line with all of that stage's layers in it. On the pre-fix code no line is
emitted at all.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase


class _Ev:
    """A CUDA event stand-in: elapsed_time is the difference of fake stamps."""

    def __init__(self, t_ms: float):
        self.t = t_ms

    def elapsed_time(self, other: "_Ev") -> float:
        return other.t - self.t


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class TestStageHead(CustomTestCase):
    def test_learned_head_is_the_first_layer_seen(self):
        from sglang.srt.layers.prefill_timing import StageHead

        h = StageHead()
        # stage 1 of a 29/11/8 split: layers 29..39, two forwards
        seen = [h.opens_forward(l) for l in list(range(29, 40)) * 2]
        self.assertEqual([i for i, s in enumerate(seen) if s], [0, 11])
        # layer 0 still opens forwards on stage 0 (the pre-fix behaviour)
        h0 = StageHead()
        self.assertEqual([h0.opens_forward(l) for l in (0, 1, 2, 0)], [True, False, False, True])
        # a caller without layer ids keeps flushing per call, as before
        hn = StageHead()
        self.assertEqual([hn.opens_forward(None) for _ in range(3)], [True, True, True])


class TestWaveTimingFlushesOffStageZero(CustomTestCase):
    def test_moe_prefill_line_on_a_stage_starting_at_layer_29(self):
        from sglang.srt.layers.moe import expert_offload as eo

        cap = _Capture()
        log = logging.getLogger(eo.__name__)
        log.addHandler(cap)
        old_level = log.level
        log.setLevel(logging.INFO)
        eo._WAVE_TP.update({"ev": [], "layers": 0, "waves": 0, "spill": 0, "tokens": 0, "forwards": 0})
        eo._WAVE_TP_HEAD.head = None
        try:
            with mock.patch("torch.cuda.synchronize"):
                for fwd in range(2):
                    for layer in range(29, 40):
                        # one wave per layer: fetch 3 ms, apply 1 ms
                        ev = (_Ev(0.0), _Ev(3.0), _Ev(4.0))
                        eo._wave_timing_note_prefill(layer, [ev], 5, 4096)
        finally:
            log.removeHandler(cap)
            log.setLevel(old_level)
            eo._WAVE_TP_HEAD.head = None
        lines = [l for l in cap.lines if l.startswith("MOE-OFFLOAD-TIMING-PREFILL")]
        self.assertEqual(len(lines), 1, cap.lines)
        self.assertIn("tokens=4096 layers=11 waves=11 spill_experts=55", lines[0])
        self.assertIn("fetch_ms=33.0 apply_ms=11.0", lines[0])
        # the second forward is still open, not lost and not doubled
        self.assertEqual(len(eo._WAVE_TP["ev"]), 11)


if __name__ == "__main__":
    unittest.main()
