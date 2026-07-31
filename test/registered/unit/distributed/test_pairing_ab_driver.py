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
"""#274 slice D: the pairing A/B driver, without a card.

The card window cannot afford a driver that (a) reads its counters after the
serving join and re-imports the drain tail #284 repaired, (b) runs a policy
arm without actually flipping the policy, or (c) reports an E whose
decomposition does not reproduce it.  All three are checkable without a GPU.
"""

import importlib.util
import pathlib
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DUAL_GROUP = _REPO_ROOT / "scripts" / "dual_group"


def _load(name: str, path: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _p in (
    str(_DUAL_GROUP),
    str(_DUAL_GROUP / "r8"),
    str(_DUAL_GROUP / "r9"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

probe = _load("lane_accept_probe", _DUAL_GROUP / "lane_accept_probe.py")
r8 = _load("lane_spec_window", _DUAL_GROUP / "r8" / "lane_spec_window.py")
axes = _load("lane_share_axes", _DUAL_GROUP / "r9" / "lane_share_axes.py")
pab = _load("pairing_ab", _DUAL_GROUP / "slice_d" / "pairing_ab.py")


def _win(
    *,
    serving_tok_s=None,
    lane_tok_s=None,
    lane_decode_tok_s=None,
    lane_prefill_tok_s=None,
    occ=None,
    cost=None,
    duty=None,
    prefill_fraction=None,
):
    return {
        "serving_tok_s": serving_tok_s,
        "lane_tok_s": lane_tok_s,
        "lane_decode_tok_s": lane_decode_tok_s,
        "lane_prefill_tok_s": lane_prefill_tok_s,
        "lane_occupancy": occ,
        "lane_cost_ms_per_token": cost,
        "lane_duty": duty,
        "lane_prefill_fraction": prefill_fraction,
    }


class TestArmRowMath(CustomTestCase):
    def test_e_and_decomposition_identity(self):
        solo_lane = _win(
            lane_tok_s=60.0,
            lane_decode_tok_s=40.0,
            lane_prefill_tok_s=20.0,
            occ=0.9,
            cost=15.0,
            duty=1.0,
            prefill_fraction=0.333,
        )
        solo_serving = _win(serving_tok_s=50.0)
        # Constructed to satisfy the identity share = occ_r / cost_r exactly:
        # occ_r 1.0, cost_r 2.0 -> share 0.5, i.e. the same graph ran twice
        # as long on the card it kept -- #284's sm_competition cell.
        shared = _win(
            serving_tok_s=45.0,
            lane_tok_s=30.0,
            lane_decode_tok_s=20.0,
            lane_prefill_tok_s=10.0,
            occ=0.9,
            cost=30.0,
            duty=1.0,
            prefill_fraction=0.333,
        )
        row = pab._arm_row(shared, solo_lane, solo_serving)
        self.assertAlmostEqual(row["share_serving"], 0.9, places=4)
        self.assertAlmostEqual(row["share_lane_total"], 0.5, places=4)
        self.assertAlmostEqual(row["E"], 1.4, places=4)
        # The r9 decomposition rides along and must reproduce the share it
        # decomposes: (occ_r / cost_r) == share within rounding.
        self.assertLess(row["identity_error"], 1e-3)
        self.assertEqual(row["carrier"], "sm_competition")
        self.assertAlmostEqual(row["prefill_fraction_drift"], 0.0, places=4)

    def test_composition_drift_is_named(self):
        solo_lane = _win(
            lane_tok_s=60.0,
            lane_decode_tok_s=40.0,
            lane_prefill_tok_s=20.0,
            prefill_fraction=0.4,
        )
        shared = _win(
            serving_tok_s=45.0,
            lane_tok_s=30.0,
            lane_decode_tok_s=25.0,
            lane_prefill_tok_s=5.0,
            prefill_fraction=0.2,
        )
        row = pab._arm_row(shared, solo_lane, _win(serving_tok_s=50.0))
        self.assertAlmostEqual(row["prefill_fraction_drift"], 0.5, places=4)


class TestMixedLaneLoadAlternates(CustomTestCase):
    def test_strict_alternation_and_shapes(self):
        posted = []
        orig = pab._post
        pab._post = lambda base, path, payload, timeout=None: posted.append(payload)
        try:
            load = pab.MixedLaneLoad(
                "http://x", [1, 2, 3], list(range(1600)), 128, prefill_new_tokens=8
            )
            for _ in range(4):
                load._post_one()
        finally:
            pab._post = orig
        jobs = [p["server_args"]["dual_group_lane_prefill"] for p in posted]
        # prefill-shaped first, then decode-shaped, strictly alternating.
        self.assertEqual([len(j["input_ids"]) for j in jobs], [1600, 3, 1600, 3])
        self.assertEqual([j["max_new_tokens"] for j in jobs], [8, 128, 8, 128])
        self.assertEqual(load.posted_prefill, 2)
        self.assertEqual(load.posted_decode, 2)


class _SeqRecorder:
    def __init__(self, events):
        self.events = events


class _FakeServing(_SeqRecorder):
    completed_tokens = 100
    completed_requests = 2

    def start(self):
        self.events.append("serving.start")

    def stop(self):
        self.events.append("serving.stop")


class _FakeLane(_SeqRecorder):
    posted = 3
    posted_prefill = 1
    posted_decode = 2
    DEPTH = 4
    errors = 0

    def start(self):
        self.events.append("lane.start")

    def stop(self):
        self.events.append("lane.stop")


class TestWindowReadsBeforeStop(CustomTestCase):
    """The #284 window-boundary fix, pinned on THIS driver's window."""

    def test_counters_read_before_loads_stop(self):
        events = []
        orig_probe, orig_idle = pab._lane_probe, pab.wait_lane_idle
        pab.wait_lane_idle = lambda base, budget_s=60.0: True

        def probe(base):
            events.append("probe")
            return {
                "work.decode_tokens": 100.0 * (events.count("probe")),
                "work.prefill_tokens": 10.0,
                "clock.device_ms": 50.0,
                "clock.busy_wall_ms": 60.0,
                "pairing.enabled": True,
            }

        pab._lane_probe = probe
        try:
            win = pab._window("http://x", 0.01, _FakeServing(events), _FakeLane(events))
        finally:
            pab._lane_probe, pab.wait_lane_idle = orig_probe, orig_idle
        # probe, start, start, probe, THEN the stops: the second probe must
        # precede both stops or the window re-imports the drain tail.
        self.assertEqual(
            events,
            [
                "probe",
                "serving.start",
                "lane.start",
                "probe",
                "serving.stop",
                "lane.stop",
            ],
        )
        self.assertEqual(win["lane_decode_tokens"], 100)
        self.assertEqual(win["pairing_enabled"], True)


class TestPolicyFlipSequence(CustomTestCase):
    """A policy arm that does not flip the policy measures nothing."""

    def test_run_pairing_ab_flips_interleaved(self):
        flips = []
        orig_flip, orig_window = pab._pairing_flip, pab._window
        pab._pairing_flip = lambda base, on: flips.append(bool(on))

        def fake_window(base, window_s, serving, lane):
            return {
                "serving_tok_s": 50.0 if serving else None,
                "lane_tok_s": 40.0 if lane else None,
                "lane_decode_tok_s": 30.0 if lane else None,
                "lane_prefill_tok_s": 10.0 if lane else None,
                "lane_occupancy": 0.5 if lane else None,
                "lane_cost_ms_per_token": 20.0 if lane else None,
                "lane_duty": 1.0 if lane else None,
                "lane_prefill_fraction": 0.25 if lane else None,
                "pairing_reordered": 0.0,
            }

        pab._window = fake_window
        try:
            out = pab.run_pairing_ab(
                "http://x",
                "tok",
                0.01,
                [1, 2],
                list(range(1600)),
                32,
                deadline=pab.time.time() + 60.0,
                repeats=2,
            )
        finally:
            pab._pairing_flip, pab._window = orig_flip, orig_window
        # Floors OFF, then OFF/ON interleaved twice, then OFF at the end so
        # the next phase starts from the regression posture.
        self.assertEqual(flips, [False, False, False, True, False, True, False])
        self.assertEqual(sorted(out["arms"].keys()), ["off_1", "off_2", "on_1", "on_2"])
        for row in out["arms"].values():
            self.assertIn("E", row)


if __name__ == "__main__":
    unittest.main()
