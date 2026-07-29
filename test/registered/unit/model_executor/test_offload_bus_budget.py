"""#286 / Erg. 7c: the PCIe bus as a budgeted item -- hermetic tests.

Covers the byte-denominated debt-model bucket (port of the #236/#242
``SpillRateBucket`` pattern), the arbiter's weighted guaranteed shares,
priority-gated borrowing of idle surplus, starvation protection (a heavy
high-priority consumer can never permanently starve a low-priority one),
the open-budget default (rate 0 = every request granted, byte-identical),
injectable measured rates, and the stage-2 planner integration
(``OffloadRegister.set_bus_arbiter``: the planner asks the arbiter, not
only the injected overlap cost function). No GPU, no CUDA.
"""

import unittest

from sglang.srt.model_executor.offload_bus_budget import (
    BUS_CONSUMER_EXPERT_STREAMING,
    BUS_CONSUMER_KV_SPILL,
    BUS_CONSUMER_STAGE2_PHASE,
    BusBudgetArbiter,
    ByteRateBucket,
)
from sglang.srt.model_executor.offload_register import (
    OffloadRegister,
    resolve_class_policies,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _Clock:
    def __init__(self, start: float = 100.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_arbiter(rate=1000.0, **consumers):
    """Arbiter with a fake clock; consumers as name=(weight, priority)."""
    clock = _Clock()
    arb = BusBudgetArbiter(total_rate_bytes_per_s=rate, clock=clock)
    for name, (weight, priority) in consumers.items():
        arb.register_consumer(name, weight=weight, priority=priority)
    return arb, clock


class TestByteRateBucket(unittest.TestCase):
    def test_debt_model_matches_the_236_pattern(self):
        bucket = ByteRateBucket(100.0, burst_seconds=1.0)
        self.assertTrue(bucket.ready())  # starts at burst capacity
        # One oversized consumption pushes into debt instead of blocking
        # forever -- the average rate converges, nothing stalls permanently.
        bucket.consume(500.0)
        self.assertFalse(bucket.ready())
        bucket.advance(0.0)
        bucket.advance(4.1)  # 410 B refill pays the debt off
        self.assertTrue(bucket.ready())

    def test_surplus_never_negative(self):
        bucket = ByteRateBucket(100.0)
        bucket.consume(1000.0)
        self.assertEqual(bucket.surplus(), 0.0)


class TestArbiter(unittest.TestCase):
    def test_open_budget_default_grants_everything(self):
        arb = BusBudgetArbiter()  # rate 0 = open, byte-identical default
        arb.register_consumer(BUS_CONSUMER_STAGE2_PHASE)
        grant = arb.request(BUS_CONSUMER_STAGE2_PHASE, 10**12)
        self.assertTrue(grant.granted)
        self.assertIn("open budget", grant.reason)

    def test_unknown_consumer_is_a_hard_error(self):
        arb = BusBudgetArbiter(1000.0)
        with self.assertRaisesRegex(ValueError, "unknown bus consumer"):
            arb.request("nobody", 1)

    def test_weighted_guaranteed_shares(self):
        arb, _clock = make_arbiter(
            rate=1000.0,
            **{
                BUS_CONSUMER_EXPERT_STREAMING: (3.0, 0),
                BUS_CONSUMER_STAGE2_PHASE: (1.0, 2),
            },
        )
        stats = arb.as_dict()
        self.assertEqual(
            stats[BUS_CONSUMER_EXPERT_STREAMING]["rate_bytes_per_s"], 750.0
        )
        self.assertEqual(stats[BUS_CONSUMER_STAGE2_PHASE]["rate_bytes_per_s"], 250.0)

    def test_denial_recovers_on_its_own(self):
        arb, clock = make_arbiter(rate=1000.0, **{BUS_CONSUMER_STAGE2_PHASE: (1.0, 2)})
        self.assertTrue(arb.request(BUS_CONSUMER_STAGE2_PHASE, 900).granted)
        self.assertTrue(arb.request(BUS_CONSUMER_STAGE2_PHASE, 900).granted)
        denied = arb.request(BUS_CONSUMER_STAGE2_PHASE, 900)
        self.assertFalse(denied.granted)
        self.assertIn("recovers", denied.reason)
        clock.advance(2.0)  # refill pays the debt
        self.assertTrue(arb.request(BUS_CONSUMER_STAGE2_PHASE, 100).granted)

    def test_borrowing_takes_only_idle_surplus(self):
        arb, _clock = make_arbiter(
            rate=1000.0,
            **{
                BUS_CONSUMER_EXPERT_STREAMING: (1.0, 0),
                BUS_CONSUMER_STAGE2_PHASE: (1.0, 2),
            },
        )
        # stage2 wants more than its 500-B share; expert stream is idle with
        # 500 B surplus -> borrow succeeds.
        grant = arb.request(BUS_CONSUMER_STAGE2_PHASE, 800)
        self.assertTrue(grant.granted)
        if grant.reason != "guaranteed share":
            self.assertGreater(grant.borrowed_bytes, 0.0)
        # The victim's bucket never went below zero (its floor is protected).
        self.assertGreaterEqual(
            arb.as_dict()[BUS_CONSUMER_EXPERT_STREAMING]["bucket_level"], 0.0
        )

    def test_borrowing_blocked_by_pending_higher_priority_demand(self):
        arb, _clock = make_arbiter(
            rate=3000.0,
            **{
                BUS_CONSUMER_EXPERT_STREAMING: (1.0, 0),
                BUS_CONSUMER_KV_SPILL: (1.0, 1),
                BUS_CONSUMER_STAGE2_PHASE: (1.0, 2),
            },
        )
        # Exhaust expert streaming far into debt -> it has pending demand.
        arb.request(BUS_CONSUMER_EXPERT_STREAMING, 5000)
        denied = arb.request(BUS_CONSUMER_EXPERT_STREAMING, 5000)
        self.assertFalse(denied.granted)
        # Put stage2's own share into debt so only borrowing could serve it.
        self.assertTrue(arb.request(BUS_CONSUMER_STAGE2_PHASE, 1500).granted)
        # stage2 (lowest class) may not borrow while the more important
        # class is waiting...
        stage2 = arb.request(BUS_CONSUMER_STAGE2_PHASE, 600)
        self.assertFalse(stage2.granted)
        self.assertIn("higher-priority", stage2.reason)
        # ...unless that class withdraws its announcement; then the idle
        # kv_spill surplus covers the 600 B.
        arb.clear_pending(BUS_CONSUMER_EXPERT_STREAMING)
        stage2 = arb.request(BUS_CONSUMER_STAGE2_PHASE, 600)
        self.assertTrue(stage2.granted)
        self.assertEqual(stage2.reason, "borrowed idle surplus")

    def test_starvation_protection_low_priority_still_progresses(self):
        arb, clock = make_arbiter(
            rate=1000.0,
            **{
                BUS_CONSUMER_EXPERT_STREAMING: (4.0, 0),
                BUS_CONSUMER_STAGE2_PHASE: (1.0, 2),
            },
        )
        # The heavy high-priority consumer hammers the bus into deep debt.
        for _ in range(10):
            arb.request(BUS_CONSUMER_EXPERT_STREAMING, 5000)
            clock.advance(0.1)
        # The low-priority consumer's own weighted share refills regardless
        # (borrowing never dips another bucket below zero), so within its
        # own budget it is served -- no permanent starvation.
        clock.advance(1.0)  # its 200-B/s share refills to burst
        grant = arb.request(BUS_CONSUMER_STAGE2_PHASE, 100)
        self.assertTrue(grant.granted)
        self.assertEqual(grant.reason, "guaranteed share")

    def test_measured_rate_is_injectable_later(self):
        arb, _clock = make_arbiter(rate=0.0, **{BUS_CONSUMER_STAGE2_PHASE: (1.0, 2)})
        self.assertTrue(arb.request(BUS_CONSUMER_STAGE2_PHASE, 10**9).granted)
        arb.set_measured_rate(100.0)  # GPU measurement phase feeds this
        arb.request(BUS_CONSUMER_STAGE2_PHASE, 10**9)  # into debt
        self.assertFalse(arb.request(BUS_CONSUMER_STAGE2_PHASE, 1).granted)

    def test_reconfigure_consumer(self):
        arb, _clock = make_arbiter(rate=1000.0, **{BUS_CONSUMER_STAGE2_PHASE: (1.0, 2)})
        arb.register_consumer(BUS_CONSUMER_STAGE2_PHASE, weight=2.0, priority=1)
        stats = arb.as_dict()[BUS_CONSUMER_STAGE2_PHASE]
        self.assertEqual(stats["weight"], 2.0)
        self.assertEqual(stats["priority"], 1)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            BusBudgetArbiter(-1.0)
        arb = BusBudgetArbiter(100.0)
        with self.assertRaises(ValueError):
            arb.register_consumer("x", weight=0.0)
        with self.assertRaises(ValueError):
            arb.set_measured_rate(-5.0)


class TestPlannerIntegration(unittest.TestCase):
    """The stage-2 planner asks the arbiter IN ADDITION to the overlap cost
    function -- the bus is a budgeted item shared with expert streaming and
    KV spill."""

    def _register(self, arbiter):
        clock = _Clock()
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
            phase_hysteresis_window_s=0.0,
            clock=clock,
            overlap_budget_fn=lambda item, cur, nxt: True,
        )
        if arbiter is not None:
            reg.set_bus_arbiter(arbiter, BUS_CONSUMER_STAGE2_PHASE)
        reg.register(
            "drafter",
            "drafter_heads",
            600,
            1.0,
            phase_mask=("draft",),
            time_constant_tier="phase",
        )
        return reg, clock

    def test_without_arbiter_only_overlap_gates(self):
        reg, _clock = self._register(None)
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["drafter"])

    def test_arbiter_denial_names_the_bus_budget(self):
        arb, _aclock = make_arbiter(rate=100.0, **{BUS_CONSUMER_STAGE2_PHASE: (1.0, 2)})
        arb.request(BUS_CONSUMER_STAGE2_PHASE, 10_000)  # deep debt
        reg, _clock = self._register(arb)
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, [])
        self.assertIn("bus budget", plan.skipped["drafter"])

    def test_arbiter_grant_admits_the_candidate_and_meters_bytes(self):
        arb, _aclock = make_arbiter(
            rate=10_000.0, **{BUS_CONSUMER_STAGE2_PHASE: (1.0, 2)}
        )
        reg, _clock = self._register(arb)
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["drafter"])
        stats = arb.as_dict()[BUS_CONSUMER_STAGE2_PHASE]
        self.assertEqual(stats["granted_bytes"], 600.0)


if __name__ == "__main__":
    unittest.main()
