# SPDX-License-Identifier: Apache-2.0
"""#305 cut 2: the control tick the arbiter names as missing.

``arbiter.return_to_idle``'s own docstring says *"the control plane calls it on
its own tick"* and there was no tick. These tests pin the one that now exists,
and they pin it against ``ladder.py`` rather than against a hand-written table:
the whole claim is that the tick steps only along DECLARED reachable edges, so
a test that restated the edges would prove nothing.

Hermetic. Every card is a declared total, every engine a fake adapter that
allocates nothing.

    python -m pytest test/registered/unit/registry/test_control_tick_305.py -v
"""

import inspect
import tempfile
import unittest
from pathlib import Path

from sglang.srt.registry import ladder, tick as tick_mod
from sglang.srt.registry.adapter import Health, register_adapter
from sglang.srt.registry.arbiter import EngineRegistry
from sglang.srt.registry.ladder import COLD, HOT, TEIL_HOT, WARM
from sglang.srt.registry.ledger import MIB, ReservationStore
from sglang.srt.registry.spec import (
    EngineClass,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
)
from sglang.srt.registry.tick import HELD, REFUSED, STEPPED, ControlTick

GIB = 1024 * MIB
CARD = "GPU-tick0000-0000-0000-0000-00000000000t"
RIG = {CARD: 32 * GIB}

#: Three test classes, each shaped like one of the three shipped ones, plus one
#: that is deliberately never declared. Declared through the ladder's own
#: extension point so the tick reads them exactly as it reads the real table.
LADDER_SHAPED = "tick_full"  # HOT / TEIL_HOT / COLD, like class1_srt
ENDS_ONLY = "tick_ends"  # HOT / COLD, like class3_utility
WARM_ONLY = "tick_warm"  # HOT / WARM / COLD, like class2_diffusion
UNDECLARED = "tick_undeclared"  # on no rung table at all

ladder.declare_class(
    LADDER_SHAPED,
    {HOT, TEIL_HOT, COLD},
    absent_because={WARM: "test double shaped like class1_srt: no host image"},
    replace=True,
)
ladder.declare_class(
    ENDS_ONLY,
    {HOT, COLD},
    absent_because={
        TEIL_HOT: "test double shaped like class3_utility: HOT / COLD only",
        WARM: "test double shaped like class3_utility: HOT / COLD only",
    },
    replace=True,
)
ladder.declare_class(
    WARM_ONLY,
    {HOT, WARM, COLD},
    absent_because={TEIL_HOT: "test double shaped like class2_diffusion"},
    replace=True,
)


class TickFake:
    """Records residency, allocates nothing, refuses nothing."""

    klass = 1

    def __init__(self, spec, context):
        self.spec = spec
        self.context = context
        self._state = ResidencyState.COLD
        self._cards = ()
        self.history = []

    def estimate(self, spec, cards):
        per_card = int(spec.launch["mib_per_card"]) * MIB
        return ResourceProfile(
            posts={c: {"declared": per_card} for c in cards},
            peak_bytes={c: per_card for c in cards},
        )

    def bind(self, cards):
        self._cards = tuple(cards)

    def state(self):
        return self._state

    def pids(self):
        return (77,) if self._state != ResidencyState.COLD else ()

    def promote(self, target):
        self.history.append(("promote", target))
        self._state = target

    def demote(self, target):
        self.history.append(("demote", target))
        self._state = target

    def measured(self):
        return {} if self._state == ResidencyState.COLD else {c: 0 for c in self._cards}

    def health(self):
        return Health(ok=True, detail="tick fake")


for _name in (LADDER_SHAPED, ENDS_ONLY, WARM_ONLY, UNDECLARED):
    register_adapter(_name, TickFake)


class TickTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.now = 2_000_000.0
        self.store = ReservationStore(
            Path(self._tmp.name),
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
        )
        self.registry = EngineRegistry(
            store=self.store,
            card_totals=RIG,
            idle_after_s=100.0,
            clock=lambda: self.now,
        )
        self.addCleanup(self.registry.shutdown)

    def add(self, engine_id, adapter=LADDER_SHAPED, mib=1024, **kw):
        self.registry.register(
            EngineSpec(
                engine_id=engine_id,
                klass=EngineClass(1),
                adapter=adapter,
                placement=(CARD,),
                launch={"mib_per_card": mib},
                **kw,
            )
        )
        return engine_id

    def tick(self, **kw):
        return ControlTick(self.registry, clock=lambda: self.now, **kw)

    def advance(self, seconds):
        self.now += seconds


class TestItIsOffByDefault(TickTestCase):
    def test_a_tick_with_no_interval_is_disabled(self):
        self.assertFalse(self.tick().enabled)

    def test_start_on_a_disabled_tick_is_a_no_op(self):
        t = self.tick()
        self.assertFalse(t.start())
        self.assertIsNone(t._thread)

    def test_the_launch_flag_defaults_to_off(self):
        from sglang.srt.registry.launch import make_parser

        args = make_parser().parse_args([])
        self.assertIsNone(args.tick_interval_s)
        self.assertEqual(tick_mod.interval_from_env(), 0.0)

    def test_an_enabled_tick_starts_and_stops(self):
        t = self.tick(interval_s=0.01)
        self.assertTrue(t.enabled)
        self.assertTrue(t.start())
        self.addCleanup(t.stop)
        self.assertTrue(t._thread.is_alive())
        t.stop()
        self.assertIsNone(t._thread)


class TestItStepsOnlyAlongDeclaredEdges(TickTestCase):
    def _idle(self, engine_id, state=ResidencyState.HOT):
        self.registry.ensure_state(engine_id, state)
        self.advance(1_000.0)

    def test_a_full_ladder_class_steps_HOT_to_TEIL_HOT_not_straight_to_COLD(self):
        """The rung ``return_to_idle`` throws away is the one worth having: the
        process, the CUDA context and the graphs survive a TEIL-HOT park."""
        self.add("full")
        self._idle("full")
        report = self.tick().run_once()
        decision = report.of("full")
        self.assertEqual(decision.action, STEPPED)
        self.assertEqual((decision.src_rung, decision.dst_rung), (HOT, TEIL_HOT))
        self.assertEqual(self.registry.instance("full").state, ResidencyState.WARM_GPU)
        self.assertEqual(decision.skipped_rungs, ())

    def test_the_step_taken_is_one_the_ladder_declares(self):
        """Pinned against ladder.can, not against a copy of the table."""
        self.add("full")
        self._idle("full")
        decision = self.tick().evaluate().of("full")
        self.assertTrue(ladder.can(LADDER_SHAPED, decision.src_rung, decision.dst_rung))

    def test_TEIL_HOT_steps_to_COLD_and_names_the_rung_it_stepped_over(self):
        """Adjacency is not reachability: this class has no WARM, so the next
        rung down is COLD and the skip has to be visible, not silent."""
        self.add("full")
        self._idle("full", ResidencyState.WARM_GPU)
        decision = self.tick().run_once().of("full")
        self.assertEqual((decision.src_rung, decision.dst_rung), (TEIL_HOT, COLD))
        self.assertEqual(decision.skipped_rungs, (WARM,))
        self.assertEqual(self.registry.instance("full").state, ResidencyState.COLD)

    def test_an_ends_only_class_goes_HOT_to_COLD_over_both_middle_rungs(self):
        self.add("ends", adapter=ENDS_ONLY)
        self._idle("ends")
        decision = self.tick().run_once().of("ends")
        self.assertEqual((decision.src_rung, decision.dst_rung), (HOT, COLD))
        self.assertEqual(decision.skipped_rungs, (TEIL_HOT, WARM))

    def test_a_warm_only_class_steps_HOT_to_WARM(self):
        self.add("warm", adapter=WARM_ONLY)
        self._idle("warm")
        decision = self.tick().run_once().of("warm")
        self.assertEqual((decision.src_rung, decision.dst_rung), (HOT, WARM))
        self.assertEqual(decision.skipped_rungs, (TEIL_HOT,))

    def test_COLD_is_the_floor_and_the_tick_says_so_rather_than_looping(self):
        self.add("full")
        self.advance(1_000.0)
        decision = self.tick().run_once().of("full")
        self.assertEqual(decision.action, HELD)
        self.assertIn("lowest rung", decision.reason)

    def test_the_tick_NEVER_promotes(self):
        """Waking a model is the request path's job. A tick that promoted would
        be cut 4, whose #375 gate is recorded UNFULFILLED."""
        self.add("full")
        self.advance(1_000.0)
        for _ in range(3):
            self.tick().run_once()
        adapter = self.registry.adapter("full")
        self.assertEqual([h for h in adapter.history if h[0] == "promote"], [])
        self.assertEqual(self.registry.instance("full").state, ResidencyState.COLD)


class TestItRefusesUnbuiltEdgesLoudly(TickTestCase):
    def test_an_undeclared_class_is_refused_and_left_where_it_is(self):
        self.add("mystery", adapter=UNDECLARED)
        self.registry.ensure_state("mystery", ResidencyState.HOT)
        self.advance(1_000.0)
        with self.assertLogs("sglang.srt.registry.tick", level="WARNING") as logs:
            report = self.tick().run_once()
        decision = report.of("mystery")
        self.assertEqual(decision.action, REFUSED)
        self.assertIn("mystery", report.refused)
        self.assertIn(UNDECLARED, decision.reason)
        # Refused, not moved: the actuator was never driven.
        self.assertEqual(self.registry.instance("mystery").state, ResidencyState.HOT)
        self.assertTrue(any("refused" in line for line in logs.output))

    def test_the_refusal_quotes_the_ladder_not_a_generic_no(self):
        self.add("mystery", adapter=UNDECLARED)
        self.registry.ensure_state("mystery", ResidencyState.HOT)
        self.advance(1_000.0)
        with self.assertLogs("sglang.srt.registry.tick", level="WARNING"):
            decision = self.tick().run_once().of("mystery")
        self.assertIn("known:", decision.reason)

    def test_a_refusal_does_not_stop_the_rest_of_the_tick(self):
        self.add("mystery", adapter=UNDECLARED)
        self.add("full")
        self.registry.ensure_state("mystery", ResidencyState.HOT)
        self.registry.ensure_state("full", ResidencyState.HOT)
        self.advance(1_000.0)
        with self.assertLogs("sglang.srt.registry.tick", level="WARNING"):
            report = self.tick().run_once()
        self.assertEqual(report.of("mystery").action, REFUSED)
        self.assertEqual(report.of("full").action, STEPPED)


class TestWhatItHoldsAndWhy(TickTestCase):
    def _hot_and_idle(self, engine_id, **kw):
        self.add(engine_id, **kw)
        self.registry.ensure_state(engine_id, ResidencyState.HOT)
        self.advance(1_000.0)
        return engine_id

    def test_a_pinned_engine_is_held(self):
        self.add("pinned", pinned=True)
        self.registry.ensure_state("pinned", ResidencyState.HOT)
        self.advance(1_000.0)
        decision = self.tick().run_once().of("pinned")
        self.assertEqual((decision.action, decision.reason), (HELD, "pinned"))
        self.assertEqual(self.registry.instance("pinned").state, ResidencyState.HOT)

    def test_the_default_hot_set_is_held(self):
        self._hot_and_idle("resting")
        self.registry.set_default_hot(["resting"])
        decision = self.tick().run_once().of("resting")
        self.assertEqual(decision.action, HELD)
        self.assertIn("default_hot", decision.reason)

    def test_a_recently_used_engine_is_held_with_its_numbers(self):
        self._hot_and_idle("busy")
        self.registry.instance("busy").last_used_ts = self.now - 5.0
        decision = self.tick().run_once().of("busy")
        self.assertEqual(decision.action, HELD)
        self.assertIn("5s ago", decision.reason)
        self.assertIn("100s idle threshold", decision.reason)

    def test_an_engine_with_a_request_in_flight_is_held(self):
        """The cut-1 half and the cut-2 half meeting: a hold taken by the
        request path is what stops the tick demoting mid-generation."""
        self.add("serving")
        self.registry.acquire_for_request("serving")
        self.advance(1_000.0)
        # last_used_ts alone would say idle; the in-flight count says otherwise.
        self.registry.instance("serving").last_used_ts = self.now - 1_000.0
        decision = self.tick().run_once().of("serving")
        self.assertEqual(decision.action, HELD)
        self.assertIn("in flight", decision.reason)
        self.assertEqual(self.registry.instance("serving").state, ResidencyState.HOT)

    def test_once_released_the_same_engine_steps_down(self):
        self.add("serving")
        self.registry.acquire_for_request("serving")
        self.registry.release_after_request("serving")
        self.advance(1_000.0)
        decision = self.tick().run_once().of("serving")
        self.assertEqual(decision.action, STEPPED)


class TestEvaluateMovesNothing(TickTestCase):
    def test_evaluate_decides_without_touching_the_adapter(self):
        self.add("full")
        self.registry.ensure_state("full", ResidencyState.HOT)
        self.advance(1_000.0)
        adapter = self.registry.adapter("full")
        before = list(adapter.history)
        report = self.tick().evaluate()
        self.assertEqual(report.of("full").action, STEPPED)
        self.assertEqual(adapter.history, before)
        self.assertEqual(self.registry.instance("full").state, ResidencyState.HOT)


class TestTheConstraintsFromTheDetermination(TickTestCase):
    def test_the_tick_does_not_reach_for_the_286_mover(self):
        """#286's RealMovementBackend has zero production callers and this does
        not become its first: the tick drives ladder edges, not the mover."""
        # The module docstring says the mover's name in order to record that it
        # is deliberately not used, so the pin is on the CODE, not on the prose.
        body = inspect.getsource(tick_mod).replace(tick_mod.__doc__ or "", "", 1)
        for forbidden in ("RealMovementBackend", "offload_movement", "MovementBackend"):
            self.assertNotIn(forbidden, body)
        # And it imports nothing from the offload layer at all.
        for line in body.splitlines():
            if line.startswith(("import ", "from ")):
                self.assertNotIn("offload", line)

    def test_the_arbiters_rank_table_agrees_with_the_ladders_order(self):
        """``ensure_state`` tells a promotion from a demotion by rank; if that
        order ever disagreed with the ladder, a step down would promote."""
        from sglang.srt.registry.arbiter import _RESIDENCY_RANK

        by_rank = sorted(_RESIDENCY_RANK, key=lambda s: _RESIDENCY_RANK[s])
        self.assertEqual(
            [ladder.rung_of_state(s) for s in by_rank], list(ladder.RUNG_ORDER)
        )

    def test_HOT_to_WARM_GPU_is_a_demotion_not_a_promotion(self):
        """The defect that made TEIL-HOT unreachable downward: routing on the
        target's name alone sent HOT -> WARM_GPU into ``_promote``."""
        self.add("full")
        self.registry.ensure_state("full", ResidencyState.HOT)
        self.registry.ensure_state("full", ResidencyState.WARM_GPU)
        history = self.registry.adapter("full").history
        self.assertIn(("demote", ResidencyState.WARM_GPU), history)
        self.assertNotIn(("promote", ResidencyState.WARM_GPU), history)


if __name__ == "__main__":
    unittest.main()
