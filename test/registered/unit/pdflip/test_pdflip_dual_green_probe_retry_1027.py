# SPDX-License-Identifier: Apache-2.0
"""Desk E1 (27.10.): the boot probe of the green-context ladder must not drop a rung on ONE noisy reading.

Bug that happened (boot dkr27bnvfp4dualbar1drei10271155, PP2 = RTX 3080, 11:57:55): the probe timed the 75 % rung
(52 of 68 SM) once against one baseline, read x1.05 (the criterion needs >= 0.3 of the predicted x1.31 slowdown = x1.092)
and dropped the rung for the whole boot (``W-DUAL-SHARE-FALLBACK mech=green ... mask does not bite``). PP1, the
identical card in the same boot, built and used the same rung, and 121 of 122 3080 ladders in all archived P logs
passed it: the mask bites, one reading was noisy (power cap / a neighbour on the card). A mask that really does not
bite reads ~x1.0 EVERY time, so the retry costs nothing against the case the probe exists for.

No GPU: a scripted backend answers ``time_stream`` per stream.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import dual_green as G  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FRACTIONS = (1.0, 0.75)


class _Stream:
    name = "gc52"


class ScriptedBackend(G.GreenBackend):
    """RTX 3080 geometry (68 SM, granularity 2). ``rung_ms`` is consumed one entry per timing of the rung stream (the
    last one repeats), ``base_ms`` one per timing of the primary stream (the last one repeats)."""

    def __init__(self, rung_ms, base_ms=(10.0,)):
        self.rung_ms, self.base_ms = list(rung_ms), list(base_ms)
        self.rung_calls = 0
        self.base_calls = 0

    def info(self):
        return G.DevInfo(0, "NVIDIA GeForce RTX 3080", "sm86", "GPU-3080", 68, 2, "cuda13020")

    def split_sizes(self, want):
        size = -(-max(1, want) // 2) * 2
        return [size] * (68 // size)

    def create(self, want):
        return self.split_sizes(want)[0], _Stream()

    def time_stream(self, stream):
        if stream is None:
            v = self.base_ms[min(self.base_calls, len(self.base_ms) - 1)]
            self.base_calls += 1
            return v
        v = self.rung_ms[min(self.rung_calls, len(self.rung_ms) - 1)]
        self.rung_calls += 1
        return v


def _build(backend):
    warns, logs = [], []
    ld = G.GreenLadder(backend, FRACTIONS, log=logs.append, warn=warns.append).build()
    return ld, warns, logs


def test_one_noisy_reading_does_not_drop_a_rung_that_bites_on_the_retry():
    be = ScriptedBackend(rung_ms=[10.5, 13.0])        # x1.05 (the PP2 reading), then x1.30 (what the mask does)
    ld, warns, logs = _build(be)
    assert sorted(ld.rungs) == [1] and ld.failed == {} and warns == []
    assert ld.rungs[1].sm == 52 and ld.counters["probe_rejected"] == 0
    assert abs(ld.rungs[1].timed_ratio - 1.30) < 1e-6          # the decisive reading, not the noisy one
    probe_lines = [x for x in logs if x.startswith(G.MARK + " P probe ")]
    assert len(probe_lines) == 1 and "ratios=[1.050,1.300]" in probe_lines[0] and "attempts=2" in probe_lines[0]


def test_a_mask_that_never_bites_is_still_dropped_after_the_attempts_and_says_so():
    be = ScriptedBackend(rung_ms=[10.2])                       # x1.02 every time
    ld, warns, _ = _build(be)
    assert ld.rungs == {} and sorted(ld.failed) == [1] and ld.counters["probe_rejected"] == 1
    assert be.rung_calls == 3                                  # the default: three attempts, no more
    assert len(warns) == 1 and "mask does not bite" in warns[0] and "after 3 attempts" in warns[0]
