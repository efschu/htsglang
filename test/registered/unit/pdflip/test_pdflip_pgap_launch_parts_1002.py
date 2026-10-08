# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-G (02.10.2026): where P's launch host time goes, and gpu_gap p50/p90.

N5a (b49f0282c2) P PP0, pdflip-22-59, 130k tokens from depth 77824, chunks of
2048: gpu_fwd_ms 475-540 per chunk, gpu_gap_ms 78-180 on EVERY chunk --
also on the passes without a publish (fwd=84 gap 81.0, deferred_publish=0) --
while host launch=426-640 ms per forward. The chunk publish (124-143 ms CPU)
already runs AFTER the next launch (P-HOST-OVERLAP); the card starves because
the LAUNCH's own host time is about the forward's GPU time. `launch` is one
number; this splits it (fb_init = ForwardBatch.init_new, fwd_raw = the model
forward's _forward_raw incl. attention metadata and the kernel launches, rest =
the scheduler's own run_batch work) and prints gpu_gap / launch p50/p90 every
16 forwards. Instrument only (FLLIPER_PDFLIP_P_HOSTGAP=1). Red on 9dff3f50ed.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

from flliper.srt.managers import pdflip_p_overlap as pov  # noqa: E402


def test_red_launch_parts_are_printed_beside_launch():
    line = pov.format_line(0, 84, 2048, 81.0, 503.9,
                           {"launch": 557.0, "fb_init": 21.0, "fwd_raw": 512.0})
    assert "launch=557" in line
    assert "launch_parts[fb_init=21 fwd_raw=512 rest=24]" in line


class _Ev:
    def __init__(self, t):
        self.t = t

    def query(self):
        return True

    def record(self):
        pass

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000.0


def test_red_summary_every_16_forwards_names_gap_and_launch_quantiles():
    clock = [0.0]

    def ev():
        return _Ev(clock[0])

    m = pov.GapMeter(0, event_factory=ev)
    lines = []
    for i in range(16):
        m.begin()
        clock[0] += 0.48                       # the forward
        m.end(i, 2048, {"launch": 500.0 + i})
        clock[0] += 0.08 if i % 2 else 0.18    # the gap before the next one
        lines += m.harvest()
    summ = [x for x in lines if x.startswith("#PGAP-SUM")]
    assert len(summ) == 1
    assert "n=16" in summ[0] and "gpu_gap_ms p50=" in summ[0] and "p90=" in summ[0]
    assert "launch_ms p50=" in summ[0]


def test_red_the_launch_is_split_at_its_two_host_sites():
    from flliper.srt.managers import tp_worker
    from flliper.srt.model_executor import model_runner

    assert '_pdflip_pov_span("fb_init")' in inspect.getsource(tp_worker)
    assert '_pdflip_pov_span("fwd_raw")' in inspect.getsource(model_runner)
