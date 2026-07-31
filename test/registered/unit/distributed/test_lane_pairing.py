# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#274 slice D: the pairing objective of the two-class scheduler.

The empirical basis is pinned in the module under test: pairing two
SM-saturating grains loses without gaining (C3: E 1.130 vs 1.440 for the
same lane budget), and the loss is SM compute competition, not preemption
granularity.  The policy is a work-conserving reorder of the lane's job
queue; these tests pin every decision rule, the classification that feeds
it, and the regression contract that a disabled policy is byte-identical
to the FIFO order the lane ran before the module existed.

Every test is hermetic: grains are classified from mock metrics (rows,
phase), the serving signal is driven with a hand-held clock, and no CUDA
is touched anywhere.
"""

from __future__ import annotations

import json
import random
import unittest

from sglang.srt.model_executor.lane_pairing import (
    DEFAULT_SAT_ROWS,
    IDLE_LABEL,
    PHASE_DECODE,
    PHASE_PREFILL,
    PairingPolicy,
    ServingGrainSignal,
    classify_rows,
    lane_job_grain_rows,
    serving_batch_grain,
)


def _job(prompt_len: int, *, prefill_done: bool = False, spec: bool = False):
    """A lane job as DualGroupLane.enqueue builds it, reduced to the fields
    the pairing policy reads."""
    return {
        "input_ids": list(range(prompt_len)),
        "prefill_ms": 1.0 if prefill_done else None,
        "spec": spec or None,
    }


class TestClassification(unittest.TestCase):
    """Deterministic labels from cheap metrics -- rows and phase only."""

    def test_pinned_label_table(self):
        # (phase, rows, sat_rows) -> saturating. The anchors that matter are
        # each an order of magnitude from the default threshold.
        table = [
            (PHASE_PREFILL, 2048, DEFAULT_SAT_ROWS, True),  # C3 prefill lane
            (PHASE_PREFILL, 96, DEFAULT_SAT_ROWS, True),  # short prompt >= 64
            (PHASE_PREFILL, 63, DEFAULT_SAT_ROWS, False),
            (PHASE_PREFILL, 64, DEFAULT_SAT_ROWS, True),  # boundary: >=
            (PHASE_DECODE, 1, DEFAULT_SAT_ROWS, False),  # plain decode
            (PHASE_DECODE, 16, DEFAULT_SAT_ROWS, False),  # 4 reqs x k+1 verify
            (PHASE_DECODE, 0, DEFAULT_SAT_ROWS, False),  # empty grain
            (PHASE_PREFILL, 30, 26, True),  # Q3_K calibration
            (PHASE_PREFILL, 100, 117, False),  # bf16 calibration
        ]
        for phase, rows, sat_rows, want in table:
            label = classify_rows(phase, rows, sat_rows)
            self.assertEqual(
                label.saturating,
                want,
                f"({phase}, rows={rows}, sat_rows={sat_rows}) -> {label}",
            )
            self.assertEqual(label.rows, rows)
            self.assertEqual(label.phase, phase)

    def test_deterministic(self):
        a = classify_rows(PHASE_PREFILL, 2048)
        b = classify_rows(PHASE_PREFILL, 2048)
        self.assertEqual(a, b)

    def test_lane_job_next_grain(self):
        # A queued job's next grain is its whole-prompt prefill.
        self.assertEqual(lane_job_grain_rows(_job(2048), 3), (PHASE_PREFILL, 2048))
        # An active non-spec job decodes one row at a time.
        self.assertEqual(
            lane_job_grain_rows(_job(2048, prefill_done=True), 3),
            (PHASE_DECODE, 1),
        )
        # An active speculative job verifies steps+1 rows per round.
        self.assertEqual(
            lane_job_grain_rows(_job(2048, prefill_done=True, spec=True), 3),
            (PHASE_DECODE, 4),
        )

    def test_serving_batch_grain(self):
        # Prefill chunk: rows are the extend token count.
        self.assertTrue(serving_batch_grain(True, 2048, 1, 1, 64).saturating)
        # Spec decode, 4 requests x 4 draft rows = 16: not saturating at 64.
        label = serving_batch_grain(False, None, 4, 4, 64)
        self.assertFalse(label.saturating)
        self.assertEqual(label.rows, 16)
        # The same shape saturates under a Q3_K-calibrated threshold of 16.
        self.assertTrue(serving_batch_grain(False, None, 4, 4, 16).saturating)


class TestServingGrainSignal(unittest.TestCase):
    def test_fresh_read_and_staleness(self):
        sig = ServingGrainSignal(stale_ms=100.0)
        label = classify_rows(PHASE_PREFILL, 2048)
        sig.publish(label, now=10.0)
        self.assertEqual(sig.read(now=10.05), label)
        # Aging out IS the idle signal: run_batch stops being called when
        # the serving group drains, no idle hook needed.
        self.assertEqual(sig.read(now=10.2), IDLE_LABEL)

    def test_unpublished_reads_idle(self):
        self.assertEqual(ServingGrainSignal().read(now=1.0), IDLE_LABEL)


def _policy(**kw):
    kw.setdefault("sat_rows", 64)
    kw.setdefault("max_defer_ms", 500.0)
    kw.setdefault("signal", ServingGrainSignal(stale_ms=100.0))
    return PairingPolicy(**kw)


def _publish(policy, rows, *, now, phase=PHASE_PREFILL):
    policy.signal.publish(classify_rows(phase, rows, policy.sat_rows), now=now)


class TestPairingPolicy(unittest.TestCase):
    """Every rule of pick(), pinned."""

    def test_serving_not_saturating_is_fifo(self):
        pol = _policy()
        _publish(pol, 4, now=1.0, phase=PHASE_DECODE)
        jobs = [_job(2048), _job(8)]
        self.assertEqual(pol.pick(jobs, now=1.01), 0)

    def test_serving_idle_is_fifo(self):
        pol = _policy()
        jobs = [_job(2048), _job(8)]
        self.assertEqual(pol.pick(jobs, now=1.0), 0)

    def test_sat_sat_avoided_picks_first_non_saturating(self):
        pol = _policy()
        _publish(pol, 2048, now=1.0)
        jobs = [_job(2048), _job(1024), _job(8), _job(4)]
        # First NON-saturating in queue order, not the smallest.
        self.assertEqual(pol.pick(jobs, now=1.01), 2)
        self.assertEqual(pol.reordered_total, 1)
        self.assertEqual(pol.last_decision.reason[:9], "saturatin")

    def test_head_non_saturating_runs(self):
        pol = _policy()
        _publish(pol, 2048, now=1.0)
        jobs = [_job(8), _job(2048)]
        self.assertEqual(pol.pick(jobs, now=1.01), 0)

    def test_all_saturating_is_work_conserving_fifo(self):
        # C3: saturating+saturating CONCURRENCY still beats taking turns
        # (E 1.130 vs 0.974) -- the policy must never idle or defer when the
        # queue offers no alternative.
        pol = _policy()
        _publish(pol, 2048, now=1.0)
        jobs = [_job(2048), _job(1024)]
        self.assertEqual(pol.pick(jobs, now=1.01), 0)

    def test_starvation_cap(self):
        pol = _policy(max_defer_ms=500.0)
        jobs = [_job(2048), _job(8)]
        _publish(pol, 2048, now=1.0)
        self.assertEqual(pol.pick(jobs, now=1.01), 1)  # head deferred
        jobs.pop(1)
        jobs.append(_job(8))
        # Still inside the cap: deferred again.
        _publish(pol, 2048, now=1.3)
        self.assertEqual(pol.pick(jobs, now=1.31), 1)
        jobs.pop(1)
        jobs.append(_job(8))
        # Past the cap (>= 500 ms since first deferral): the head runs even
        # though the pairing is bad.
        _publish(pol, 2048, now=1.6)
        self.assertEqual(pol.pick(jobs, now=1.61), 0)
        self.assertEqual(pol.starvation_overrides_total, 1)

    def test_deferral_record_cleared_when_head_runs(self):
        pol = _policy()
        jobs = [_job(2048), _job(8)]
        _publish(pol, 2048, now=1.0)
        self.assertEqual(pol.pick(jobs, now=1.01), 1)
        # Serving drains; the head runs on FIFO and its deferral record is
        # spent -- a LATER deferral of the same dict must start a new clock.
        self.assertEqual(pol.pick(jobs, now=2.0), 0)
        self.assertEqual(pol._deferred_since, {})

    def test_trivial_queue_short_circuits(self):
        pol = _policy()
        _publish(pol, 2048, now=1.0)
        self.assertEqual(pol.pick([_job(2048)], now=1.01), 0)
        # The short circuit must not even count a pick: it is the path every
        # near-empty queue takes.
        self.assertEqual(pol.picks_total, 0)

    def test_snapshot_is_json_serializable(self):
        pol = _policy()
        _publish(pol, 2048, now=1.0)
        pol.pick([_job(2048), _job(8)], now=1.01)
        json.dumps(pol.snapshot())


class TestDisabledIsByteIdenticalFifo(unittest.TestCase):
    """The regression contract: policy off = the FIFO order the lane ran
    before this module existed, on every input."""

    def test_disabled_always_picks_head(self):
        pol = _policy(enabled=False)
        rng = random.Random(274)
        for _ in range(200):
            jobs = [
                _job(
                    rng.choice([4, 8, 96, 1024, 2048]),
                    prefill_done=rng.random() < 0.2,
                )
                for _ in range(rng.randint(1, 6))
            ]
            _publish(pol, rng.choice([1, 16, 2048]), now=1.0)
            self.assertEqual(pol.pick(jobs, now=1.0), 0)
        # Disabled, the policy also keeps no state and counts nothing --
        # anything else would be observable work on the disabled path.
        self.assertEqual(pol.picks_total, 0)
        self.assertEqual(pol.reordered_total, 0)
        self.assertEqual(pol._deferred_since, {})

    def test_runtime_flip_round_trip(self):
        # The single-boot A/B: the same policy object, flipped at runtime.
        pol = _policy(enabled=False)
        jobs = [_job(2048), _job(8)]
        _publish(pol, 2048, now=1.0)
        self.assertEqual(pol.pick(jobs, now=1.01), 0)
        pol.enabled = True
        _publish(pol, 2048, now=1.02)
        self.assertEqual(pol.pick(jobs, now=1.03), 1)
        pol.enabled = False
        _publish(pol, 2048, now=1.04)
        self.assertEqual(pol.pick(jobs, now=1.05), 0)


class TestLanePickIntegration(unittest.TestCase):
    """The lane's pick site, exercised on the real code path shape: a list
    popped at the policy's index. DualGroupLane itself needs a model runner,
    so the contract is pinned on the exact expression the lane runs."""

    def _pick_like_lane(self, jobs, policy):
        # Mirrors DualGroupLane._step_locked_scope: FIFO unless a policy is
        # attached and the queue is non-trivial.
        idx = 0
        if policy is not None and len(jobs) > 1:
            idx = policy.pick(jobs, now=1.01)
        return jobs.pop(idx)

    def test_no_policy_is_fifo(self):
        jobs = [_job(2048), _job(8)]
        first = jobs[0]
        self.assertIs(self._pick_like_lane(jobs, None), first)

    def test_policy_reorders_and_queue_stays_consistent(self):
        pol = _policy()
        _publish(pol, 2048, now=1.0)
        j_sat, j_small = _job(2048), _job(8)
        jobs = [j_sat, j_small]
        self.assertIs(self._pick_like_lane(jobs, pol), j_small)
        self.assertEqual(jobs, [j_sat])


if __name__ == "__main__":
    unittest.main()
