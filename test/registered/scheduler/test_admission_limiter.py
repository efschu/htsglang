"""Hermetic tests for the floating admission limit (#287).

No torch, no GPU, no server: the limiter, its pure helpers and the source-
level ordering guarantee at the retraction call site.
"""

import contextvars
import re
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.managers.admission_limiter import (
    ADMISSION_RELIEF_FEATURE,
    REASON_API,
    REASON_KV_PRESSURE,
    REASON_PRE_RETRACT,
    REASON_RELEASE,
    AdmissionLimiter,
    AdmissionLimitError,
    admission_limiter_scope,
    current_admission_limiter,
    replicated_pool_usage,
    resolve_admission_start,
    set_admission_limiter,
    spill_session_cap,
    throttle_before_retract,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEDULER_PY = _REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler.py"


class TestAdmissionLimiterConstruction(unittest.TestCase):
    def test_start_defaults_to_ceiling(self):
        lim = AdmissionLimiter(64)
        self.assertEqual(lim.ceiling, 64)
        self.assertEqual(lim.current, 64)
        self.assertEqual(lim.start, 64)
        self.assertFalse(lim.auto)

    def test_start_below_ceiling(self):
        lim = AdmissionLimiter(64, 8)
        self.assertEqual(lim.current, 8)
        self.assertEqual(lim.ceiling, 64)

    def test_rejects_start_above_ceiling(self):
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(8, 16)

    def test_rejects_floor_above_ceiling(self):
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(4, floor=8)

    def test_rejects_non_positive_ceiling(self):
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(0)

    def test_rejects_release_at_or_above_throttle(self):
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(64, throttle_high=0.8, release_low=0.8)
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(64, throttle_high=0.8, release_low=0.9)

    def test_rejects_out_of_range_marks(self):
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(64, throttle_high=1.5)
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(64, release_low=0.0)

    def test_rejects_zero_hysteresis(self):
        with self.assertRaises(AdmissionLimitError):
            AdmissionLimiter(64, release_hysteresis=0)


class TestBackwardCompatibility(unittest.TestCase):
    """Without a ceiling the limiter must be a passive holder."""

    def test_passive_holder_never_moves_on_its_own(self):
        lim = AdmissionLimiter(32, 32, auto=False)
        for usage in (0.0, 0.5, 0.95, 1.0, 0.1):
            self.assertFalse(lim.observe(usage, running_bs=32))
        self.assertEqual(lim.current, 32)
        self.assertEqual(lim.throttle_count, 0)
        self.assertEqual(lim.release_count, 0)

    def test_pre_retract_throttle_is_inert_without_auto(self):
        lim = AdmissionLimiter(32, 32, auto=False)
        self.assertFalse(throttle_before_retract(lim, running_bs=32))
        self.assertEqual(lim.current, 32)

    def test_pre_retract_throttle_tolerates_no_limiter(self):
        self.assertFalse(throttle_before_retract(None, running_bs=8))

    def test_spill_cap_ignores_passive_limiter(self):
        lim = AdmissionLimiter(32, 4, auto=False)
        self.assertEqual(spill_session_cap(0, lim), 0)
        self.assertEqual(spill_session_cap(7, lim), 7)


class TestCeilingDimensionsAndFloat(unittest.TestCase):
    def test_limit_never_exceeds_ceiling(self):
        lim = AdmissionLimiter(16, 15, auto=True, release_hysteresis=1)
        for _ in range(20):
            lim.observe(0.0, running_bs=1)
        self.assertEqual(lim.current, 16)

    def test_limit_never_falls_below_floor(self):
        lim = AdmissionLimiter(64, 64, floor=4, auto=True)
        # Sustained pressure with the batch draining alongside the limit.
        for _ in range(200):
            lim.observe(0.99, running_bs=lim.current)
        self.assertEqual(lim.current, 4)


class TestThrottle(unittest.TestCase):
    def test_throttle_stops_inflow_and_demands_one_drain(self):
        lim = AdmissionLimiter(64, 32, auto=True)
        self.assertTrue(lim.throttle(running_bs=20))
        # Below the running batch: no admission until one request finishes.
        self.assertEqual(lim.current, 19)
        self.assertEqual(lim.last_reason, REASON_KV_PRESSURE)
        self.assertEqual(lim.throttle_count, 1)

    def test_throttle_uses_the_tighter_of_limit_and_batch(self):
        lim = AdmissionLimiter(64, 8, auto=True)
        lim.throttle(running_bs=40)
        self.assertEqual(lim.current, 7)

    def test_throttle_at_floor_is_a_no_op(self):
        lim = AdmissionLimiter(64, 1, floor=1, auto=True)
        self.assertFalse(lim.throttle(running_bs=1))
        self.assertEqual(lim.current, 1)
        self.assertEqual(lim.throttle_count, 0)

    def test_observe_throttles_at_the_high_mark(self):
        lim = AdmissionLimiter(64, 32, auto=True, throttle_high=0.9)
        self.assertFalse(lim.observe(0.89, running_bs=32))
        self.assertEqual(lim.current, 32)
        self.assertTrue(lim.observe(0.90, running_bs=32))
        self.assertEqual(lim.current, 31)


class TestRelease(unittest.TestCase):
    def test_release_needs_consecutive_low_samples(self):
        lim = AdmissionLimiter(64, 16, auto=True, release_low=0.5, release_hysteresis=3)
        self.assertFalse(lim.observe(0.4, running_bs=4))
        self.assertFalse(lim.observe(0.4, running_bs=4))
        self.assertEqual(lim.current, 16)
        self.assertTrue(lim.observe(0.4, running_bs=4))
        self.assertGreater(lim.current, 16)
        self.assertEqual(lim.last_reason, REASON_RELEASE)

    def test_band_sample_resets_the_streak(self):
        lim = AdmissionLimiter(64, 16, auto=True, release_low=0.5, release_hysteresis=3)
        lim.observe(0.4, running_bs=4)
        lim.observe(0.4, running_bs=4)
        # Inside the hysteresis band -- neither throttle nor release, and the
        # partial evidence is discarded.
        self.assertFalse(lim.observe(0.7, running_bs=4))
        self.assertFalse(lim.observe(0.4, running_bs=4))
        self.assertFalse(lim.observe(0.4, running_bs=4))
        self.assertEqual(lim.current, 16)
        self.assertTrue(lim.observe(0.4, running_bs=4))

    def test_release_is_geometric_and_saturates_at_the_ceiling(self):
        lim = AdmissionLimiter(
            256, 32, auto=True, release_low=0.5, release_hysteresis=1
        )
        steps = 0
        while lim.current < 256 and steps < 1000:
            lim.observe(0.1, running_bs=1)
            steps += 1
        self.assertEqual(lim.current, 256)
        # A +1 walk would need 224 steps; the geometric step must be far cheaper.
        self.assertLess(steps, 40)

    def test_throttle_resets_a_pending_release_streak(self):
        lim = AdmissionLimiter(64, 16, auto=True, release_hysteresis=2)
        lim.observe(0.1, running_bs=4)
        lim.observe(0.99, running_bs=4)
        self.assertEqual(lim.current, 3)
        self.assertFalse(lim.observe(0.1, running_bs=3))


class TestApiSetter(unittest.TestCase):
    def test_set_limit_within_range(self):
        lim = AdmissionLimiter(64, 64)
        lim.set_limit(9)
        self.assertEqual(lim.current, 9)
        self.assertEqual(lim.last_reason, REASON_API)
        lim.set_limit(64)
        self.assertEqual(lim.current, 64)

    def test_set_limit_above_ceiling_is_rejected(self):
        lim = AdmissionLimiter(64, 64)
        with self.assertRaises(AdmissionLimitError) as ctx:
            lim.set_limit(65)
        self.assertIn("ceiling", str(ctx.exception))
        self.assertEqual(lim.current, 64)

    def test_set_limit_below_floor_is_rejected(self):
        lim = AdmissionLimiter(64, 64, floor=4)
        with self.assertRaises(AdmissionLimitError):
            lim.set_limit(3)
        self.assertEqual(lim.current, 64)

    def test_set_limit_rejects_non_integer(self):
        lim = AdmissionLimiter(64, 64)
        with self.assertRaises(AdmissionLimitError):
            lim.set_limit("many")

    def test_manual_setter_works_without_the_auto_controller(self):
        # The API knob is available even with no ceiling configured; it is an
        # explicit operator action, not a behavior change of the default path.
        lim = AdmissionLimiter(32, 32, auto=False)
        lim.set_limit(4)
        self.assertEqual(lim.current, 4)


class TestResolveAdmissionStart(unittest.TestCase):
    def test_no_start_means_start_at_the_ceiling(self):
        self.assertEqual(resolve_admission_start(48, None), 48)

    def test_start_is_divided_by_dp_size(self):
        self.assertEqual(resolve_admission_start(64, 32, dp_size=4), 8)

    def test_start_is_clamped_to_the_resolved_ceiling(self):
        # Memory pressure cut the ceiling below the requested start.
        self.assertEqual(resolve_admission_start(10, 32), 10)

    def test_start_never_drops_below_one(self):
        self.assertEqual(resolve_admission_start(16, 2, dp_size=8), 1)

    def test_rejects_non_positive_ceiling(self):
        with self.assertRaises(AdmissionLimitError):
            resolve_admission_start(0, 4)


class TestReplicatedPoolUsage(unittest.TestCase):
    def test_fraction(self):
        self.assertAlmostEqual(replicated_pool_usage(500, 1000), 0.5)

    def test_zero_capacity_is_no_pressure(self):
        self.assertEqual(replicated_pool_usage(10, 0), 0.0)

    def test_clamped_to_unit_interval(self):
        self.assertEqual(replicated_pool_usage(2000, 1000), 1.0)
        self.assertEqual(replicated_pool_usage(-5, 1000), 0.0)

    def test_identical_inputs_give_identical_verdicts(self):
        # The whole point: two ranks feeding the same replicated numbers into
        # two independently constructed limiters must stay in lockstep.
        a = AdmissionLimiter(64, 32, auto=True)
        b = AdmissionLimiter(64, 32, auto=True)
        for held in (100, 900, 950, 200, 150, 100, 100, 100, 100, 100, 100, 100):
            usage = replicated_pool_usage(held, 1000)
            a.observe(usage, running_bs=16)
            b.observe(usage, running_bs=16)
            self.assertEqual(a.current, b.current)


class TestSpillSessionCap(unittest.TestCase):
    def test_no_limiter_returns_the_configured_regler(self):
        self.assertEqual(spill_session_cap(5, None), 5)
        self.assertEqual(spill_session_cap(0, None), 0)

    def test_armed_limiter_supplies_a_cap_when_the_regler_is_off(self):
        lim = AdmissionLimiter(64, 6, auto=True)
        self.assertEqual(spill_session_cap(0, lim), 6)

    def test_tighter_side_wins(self):
        lim = AdmissionLimiter(64, 6, auto=True)
        self.assertEqual(spill_session_cap(3, lim), 3)
        self.assertEqual(spill_session_cap(9, lim), 6)

    def test_cap_follows_the_float(self):
        lim = AdmissionLimiter(64, 20, auto=True)
        self.assertEqual(spill_session_cap(0, lim), 20)
        lim.throttle(running_bs=10)
        self.assertEqual(spill_session_cap(0, lim), 9)


class TestPerLaneIsolation(unittest.TestCase):
    """#274: the limit is per group/lane, never a module singleton."""

    def test_unset_context_resolves_to_none(self):
        def probe():
            return current_admission_limiter()

        self.assertIsNone(contextvars.copy_context().run(probe))

    def test_scope_installs_and_restores(self):
        outer = AdmissionLimiter(8)
        inner = AdmissionLimiter(4)

        def body():
            set_admission_limiter(outer)
            self.assertIs(current_admission_limiter(), outer)
            with admission_limiter_scope(inner):
                self.assertIs(current_admission_limiter(), inner)
            self.assertIs(current_admission_limiter(), outer)

        contextvars.copy_context().run(body)

    def test_two_lanes_do_not_see_each_other(self):
        serving = AdmissionLimiter(64, 64, auto=True)
        lane = AdmissionLimiter(4, 4, auto=True, lane_id=0)
        seen = {}
        barrier = threading.Barrier(2)

        def run(name, limiter):
            with admission_limiter_scope(limiter):
                barrier.wait(timeout=5)
                limiter.throttle(running_bs=limiter.current)
                seen[name] = current_admission_limiter()

        threads = [
            threading.Thread(target=run, args=("serving", serving)),
            threading.Thread(target=run, args=("lane", lane)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertIs(seen["serving"], serving)
        self.assertIs(seen["lane"], lane)
        self.assertEqual(serving.current, 63)
        self.assertEqual(lane.current, 3)
        # And the process-wide context was never touched by either thread.
        self.assertIsNone(current_admission_limiter())

    def test_snapshot_carries_the_lane_id(self):
        lane = AdmissionLimiter(4, 4, lane_id=2)
        self.assertEqual(lane.snapshot()["lane_id"], 2)
        self.assertEqual(lane.snapshot()["ceiling"], 4)

    def test_dual_group_lane_builds_and_installs_its_own(self):
        # The lane's limiter is lazy and keyed to its own
        # --dual-group-lane-max-requests, and the runtime scope installs it.
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        stub = SimpleNamespace(
            _admission_limiter=None,
            lane_id=3,
            runner=SimpleNamespace(
                server_args=SimpleNamespace(dual_group_lane_max_requests=2)
            ),
        )
        lim = DualGroupLane.admission_limiter.fget(stub)
        self.assertEqual(lim.ceiling, 2)
        self.assertEqual(lim.lane_id, 3)
        self.assertFalse(lim.auto)
        # Cached: the lane must not mint a fresh limiter per tick.
        self.assertIs(DualGroupLane.admission_limiter.fget(stub), lim)

    def test_lane_runtime_scope_installs_the_lane_limiter(self):
        src = (
            _REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "dual_group_lane.py"
        ).read_text()
        block = src[src.index("    def _lane_runtime_scope(self):") :]
        block = block[: block.index("\n    @property")]
        self.assertIn("admission_limiter_scope(self.admission_limiter)", block)


class TestThrottleBeforeRetractOrdering(unittest.TestCase):
    """The ordering is the feature: drain the inflow before discarding
    sessions that already hold state. Checked structurally against the real
    source so a later refactor that moves the call cannot pass silently."""

    def _update_running_batch_source(self) -> str:
        src = _SCHEDULER_PY.read_text()
        start = src.index("    def update_running_batch(self")
        # Next method definition at the same indentation ends the block.
        end = src.index("\n    def ", start + 1)
        return src[start:end]

    #: The retraction fallback inside ``update_running_batch``. #679
    #: (82ba7e2c10, 2026-08-16) moved the body behind
    #: ``_retract_decode_and_requeue``; the literal pinned here was
    #: ``batch.retract_decode(``, which stopped existing at that commit while
    #: this test was left as it was (#898, determined 2026-08-26).
    RETRACT_CALL = "self._retract_decode_and_requeue("
    THROTTLE_CALL = "throttle_before_retract("

    def _locate(self, block: str, needle: str) -> int:
        """``str.index`` raises a bare ``ValueError: substring not found``,
        which says neither which literal went missing nor where to look. A
        source pin that drifts must name its own drift -- that is the whole
        reason the pin exists."""
        at = block.find(needle)
        if at < 0:
            self.fail(
                f"{needle!r} no longer appears in Scheduler.update_running_batch "
                f"({_SCHEDULER_PY}). Either the ordering this test guards was "
                f"refactored away, or the call was renamed and this pin must "
                f"follow it. Block is {len(block)} chars."
            )
        return at

    def test_source_calls_throttle_before_retract_decode(self):
        block = self._update_running_batch_source()
        throttle_at = self._locate(block, self.THROTTLE_CALL)
        retract_at = self._locate(block, self.RETRACT_CALL)
        self.assertLess(
            throttle_at,
            retract_at,
            "the admission throttle must run before the retraction fallback",
        )

    def test_throttle_is_gated_on_the_pressure_flag(self):
        block = self._update_running_batch_source()
        self.assertRegex(
            block,
            re.compile(
                r"if kv_full_retract_flag:\s*\n(\s*#.*\n)*"
                r"\s*throttle_before_retract\("
            ),
            "the throttle must fire only on real KV pressure",
        )

    def test_pre_retract_reason_is_distinguishable(self):
        lim = AdmissionLimiter(64, 32, auto=True)
        throttle_before_retract(lim, running_bs=32)
        self.assertEqual(lim.last_reason, REASON_PRE_RETRACT)


class TestLadderRegistration(unittest.TestCase):
    def test_admission_cap_sits_between_kv_vector_flip_and_data_movers(self):
        """#287 user directive: the relief order is a SERVICE-cost order --
        KV-vector flip (service-neutral per #320) < admission lowering
        (turns sessions away) < any data movement (spill/offload)."""
        from sglang.srt.model_executor.kv_pressure_ladder import (
            RELIEF_FEATURES,
            RELIEF_ORDER,
        )

        self.assertIn(ADMISSION_RELIEF_FEATURE, RELIEF_FEATURES)
        self.assertEqual(RELIEF_ORDER[0], "dcp_ratio")
        self.assertEqual(RELIEF_ORDER[1], ADMISSION_RELIEF_FEATURE)
        self.assertLess(
            RELIEF_ORDER.index(ADMISSION_RELIEF_FEATURE),
            min(
                RELIEF_ORDER.index("kv_spill"),
                RELIEF_ORDER.index("session_offload"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
