"""#284: the axis-isolation driver, without a card.

The driver decides three things that a card window cannot afford to get wrong
and that need no card to check: whether its decomposition actually reproduces
the share it decomposes, whether the eager arm really switches the lane's
graphs off (an arm that silently ran the captured path would answer the
question with the baseline), and whether a missing floor drops an arm instead
of dividing by a stale one.

The fake server carries the device clock the real lane publishes, so the
occupancy arithmetic is exercised on the shape it will meet on the rig.
"""

import importlib.util
import pathlib
import sys
import threading
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DUAL_GROUP = _REPO_ROOT / "scripts" / "dual_group"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _p in (str(_DUAL_GROUP), str(_DUAL_GROUP / "r8")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

probe = _load("lane_accept_probe", _DUAL_GROUP / "lane_accept_probe.py")
r8 = _load("lane_spec_window", _DUAL_GROUP / "r8" / "lane_spec_window.py")
axes = _load("lane_share_axes", _DUAL_GROUP / "r9" / "lane_share_axes.py")


class FakeLaneServer:
    """A lane that emits tokens and burns device time at chosen rates."""

    def __init__(self, tok_per_post=8, device_ms_per_token=18.0):
        self.decode_tokens = 0.0
        self.device_ms = 0.0
        self.busy_wall_ms = 0.0
        self.spans = 0.0
        self.posted = []
        self.tok_per_post = tok_per_post
        self.device_ms_per_token = device_ms_per_token
        self.queued = 0
        self.active = False
        self._lock = threading.Lock()

    def get(self, base, path, timeout=None):
        with self._lock:
            return {
                "internal_states": [
                    {
                        "dual_group_lanes": [
                            {
                                "lane_id": 0,
                                "queued": self.queued,
                                "active": self.active,
                                "work_total": {"decode_tokens": self.decode_tokens},
                                "device_clock": {
                                    "device_ms": self.device_ms,
                                    "busy_wall_ms": self.busy_wall_ms,
                                    "spans": self.spans,
                                    "forced_reads": 0,
                                },
                            }
                        ],
                        "lane_share": {"e_ema": 1.1, "gates": []},
                    }
                ],
                "server_args": {},
            }

    def post(self, base, path, payload, timeout=None):
        if path == "/set_internal_state":
            job = payload["server_args"]["dual_group_lane_prefill"]
            with self._lock:
                self.posted.append(job)
                self.decode_tokens += self.tok_per_post
                self.device_ms += self.tok_per_post * self.device_ms_per_token
                self.spans += self.tok_per_post
            return {}
        if path == "/generate":
            return {"meta_info": {"completion_tokens": 11}}
        raise AssertionError(path)


class _Patched:
    def __init__(self, server):
        self.server = server
        self._saved = {}

    def __enter__(self):
        for mod in (probe, r8, axes):
            for attr in ("_get", "_post"):
                if hasattr(mod, attr):
                    self._saved[(mod, attr)] = getattr(mod, attr)
            mod._get = self.server.get
            mod._post = self.server.post
        return self.server

    def __exit__(self, *exc):
        for (mod, attr), value in self._saved.items():
            setattr(mod, attr, value)
        return False


class TestDecomposition(CustomTestCase):
    """share = occupancy_ratio / cost_ratio, checked against itself."""

    @staticmethod
    def _win(tok_s, occ, cost, duty=1.0):
        return {
            "lane_tok_s": tok_s,
            "lane_occupancy": occ,
            "lane_cost_ms_per_token": cost,
            "lane_duty": duty,
        }

    def test_the_identity_holds_and_is_reported(self):
        solo = self._win(50.0, 0.90, 18.0)
        shared = self._win(15.0, 0.45, 30.0)
        got = axes._decompose(shared, solo)
        self.assertAlmostEqual(got["share_lane"], 0.3, places=6)
        self.assertLess(got["identity_error"], 1e-6)

    def test_more_device_time_per_token_is_sm_competition(self):
        solo = self._win(50.0, 0.90, 18.0)
        shared = self._win(25.0, 0.90, 36.0)
        got = axes._decompose(shared, solo)
        self.assertEqual(got["carrier"], "sm_competition")
        self.assertAlmostEqual(got["occupancy_ratio"], 1.0, places=6)

    def test_the_same_cost_on_less_card_time_is_a_submission_gap(self):
        solo = self._win(50.0, 0.90, 18.0)
        shared = self._win(25.0, 0.45, 18.0)
        got = axes._decompose(shared, solo)
        self.assertEqual(got["carrier"], "submission_gap")

    def test_a_lane_that_was_not_fed_is_starved(self):
        solo = self._win(50.0, 0.90, 18.0, duty=1.0)
        shared = self._win(25.0, 0.45, 18.0, duty=0.5)
        got = axes._decompose(shared, solo)
        self.assertEqual(got["carrier"], "starved")

    def test_an_undisturbed_lane_has_no_carrier(self):
        solo = self._win(50.0, 0.90, 18.0)
        got = axes._decompose(self._win(51.0, 0.92, 18.0), solo)
        self.assertIsNone(got["carrier"])

    def test_a_lane_without_a_clock_still_reports_its_share(self):
        solo = {"lane_tok_s": 50.0}
        got = axes._decompose({"lane_tok_s": 25.0}, solo)
        self.assertAlmostEqual(got["share_lane"], 0.5, places=6)
        self.assertNotIn("carrier", got)


class TestProbeAndWindow(CustomTestCase):
    def test_both_counter_families_come_from_one_call(self):
        server = FakeLaneServer()
        with _Patched(server):
            got = axes._lane_probe("http://x")
        self.assertIn("work.decode_tokens", got)
        self.assertIn("clock.device_ms", got)
        self.assertIn("clock.busy_wall_ms", got)

    def test_a_window_differences_work_and_device_time(self):
        server = FakeLaneServer(tok_per_post=8, device_ms_per_token=20.0)
        with _Patched(server):
            lane = axes.DepthLaneLoad(
                "http://x", [1, 2], 8, False, 1, "target_verify", depth=2
            )
            win = axes._window("http://x", 1.0, None, lane)
        self.assertGreater(win["lane_tokens"], 0)
        self.assertAlmostEqual(win["lane_cost_ms_per_token"], 20.0, places=3)
        self.assertGreater(win["lane_occupancy"], 0.0)


class TestWindowBoundary(CustomTestCase):
    """The fifth defect of this family: counters read after the loads stop.

    ``serving.stop()`` joins workers that are inside a /generate call, and the
    lane keeps working through that join. Counters read afterwards carry a
    drain tail the measured wall clock does not, and it lands on the SHARED
    windows only -- the numerator of share_lane.
    """

    def test_the_counters_are_read_before_the_loads_are_stopped(self):
        server = FakeLaneServer(tok_per_post=8, device_ms_per_token=20.0)
        seen = {}

        class SlowStopLane(axes.DepthLaneLoad):
            def stop(self):
                # Keep producing during the join, as the real lane does.
                for _ in range(50):
                    self._post_one()
                seen["stopped"] = True
                super().stop()

        with _Patched(server):
            lane = SlowStopLane(
                "http://x", [1, 2], 8, False, 1, "target_verify", depth=2
            )
            win = axes._window("http://x", 0.5, None, lane)
        self.assertTrue(seen.get("stopped"))
        # The 50 posts issued during the join must be outside the window.
        self.assertLess(win["lane_tokens"], 50 * 8)

    def test_an_impossible_duty_is_named_not_clamped(self):
        """A lane that reports more busy wall time than the window had did not
        measure a busy lane; it measured outside its own window. Clamping it to
        1.0 would turn the impossibility back into a plausible reading."""
        server = FakeLaneServer()

        class OverBusyLane(axes.DepthLaneLoad):
            def start(self):
                # Two seconds of busy wall time inside a 0.2 s window.
                server.busy_wall_ms += 2000.0
                super().start()

        with _Patched(server):
            lane = OverBusyLane(
                "http://x", [1, 2], 8, False, 1, "target_verify", depth=2
            )
            win = axes._window("http://x", 0.2, None, lane)
        self.assertGreater(win["lane_duty"], 1.0)
        self.assertIn("window_defect", win)
        self.assertIn("duty", win["window_defect"])

    def test_a_clean_window_carries_no_defect_note(self):
        server = FakeLaneServer()
        with _Patched(server):
            lane = axes.DepthLaneLoad(
                "http://x", [1, 2], 8, False, 1, "target_verify", depth=2
            )
            win = axes._window("http://x", 0.2, None, lane)
        self.assertNotIn("window_defect", win)


class TestEagerArm(CustomTestCase):
    """The arm that answers the question has to actually be a different arm."""

    def test_the_eager_arm_switches_both_graphs_off_in_the_posted_job(self):
        server = FakeLaneServer()
        with _Patched(server):
            lane = axes.DepthLaneLoad(
                "http://x",
                [1, 2],
                8,
                True,
                1,
                "target_verify",
                depth=2,
                overrides=axes.EAGER_OVERRIDES,
            )
            lane._post_one()
        job = server.posted[-1]
        self.assertIs(job["verify_graph"], False)
        self.assertIs(job["head_graph"], False)
        self.assertIs(job["spec"], True)

    def test_the_baseline_arm_posts_no_overrides(self):
        server = FakeLaneServer()
        with _Patched(server):
            lane = axes.DepthLaneLoad(
                "http://x", [1, 2], 8, False, 1, "target_verify", depth=2
            )
            lane._post_one()
        job = server.posted[-1]
        self.assertNotIn("verify_graph", job)
        self.assertNotIn("head_graph", job)

    def test_the_depth_is_the_axis_it_claims_to_be(self):
        a = axes.DepthLaneLoad("http://x", [1], 8, False, 1, "v", depth=1)
        b = axes.DepthLaneLoad("http://x", [1], 8, False, 1, "v", depth=2)
        self.assertEqual((a.DEPTH, b.DEPTH), (1, 2))
        # ... and not a class attribute shared by both instances.
        self.assertNotEqual(a.DEPTH, b.DEPTH)


class TestArmBookkeeping(CustomTestCase):
    def test_an_arm_whose_floor_is_missing_is_skipped_not_divided(self):
        server = FakeLaneServer()
        with _Patched(server):
            axes.tokenize = lambda base, text, tok: [1, 2, 3]
            got = axes.run_axes(
                "http://x",
                "tok",
                "squares",
                0.2,
                8,
                8,
                1,
                "target_verify",
                # C needs serving_c1, which is only measured when C is asked
                # for; asking for it after the deadline leaves the arm without
                # a floor, which must not silently reuse the c4 one.
                ["C_light_load"],
                deadline=0.0,
            )
        self.assertEqual(got["arms"]["C_light_load"], {"skipped": "missing floor"})

    def test_the_deadline_drops_the_tail_rather_than_truncating_a_window(self):
        server = FakeLaneServer()
        with _Patched(server):
            axes.tokenize = lambda base, text, tok: [1, 2, 3]
            got = axes.run_axes(
                "http://x",
                "tok",
                "squares",
                0.2,
                8,
                8,
                1,
                "target_verify",
                ["A_baseline"],
                deadline=0.0,
            )
        self.assertIn("skipped", got)
        self.assertEqual(got["floors"], {})

    def test_a_complete_arm_reports_share_carrier_and_e(self):
        server = FakeLaneServer()
        with _Patched(server):
            axes.tokenize = lambda base, text, tok: [1, 2, 3]
            got = axes.run_axes(
                "http://x",
                "tok",
                "squares",
                0.2,
                8,
                8,
                1,
                "target_verify",
                ["A_baseline"],
                deadline=1e18,
            )
        arm = got["arms"]["A_baseline"]
        self.assertIn("share_lane", arm)
        self.assertIn("share_serving", arm)
        self.assertIn("E", arm)
        self.assertEqual(arm["solo_lane_floor"], "lane_captured")


if __name__ == "__main__":
    unittest.main()
