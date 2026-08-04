# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#546 idle park: the decision, the state machine, and the bytes.

Hermetic. CPU throughout, ``CUDA_VISIBLE_DEVICES=99``:

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \\
      python -m pytest test/registered/translator/test_idle_park.py -v

WHAT THESE TESTS ARE FOR, and what they deliberately cannot prove. The park
mechanism is the #286 audio ledger, exercised here for real -- tensors are
moved, modules become unusable, weights come back bit-identical. What a desk
cannot show is that ``nvidia-smi`` drops, because that needs a card and a
driver; the GPU step in NOTE_546 owns that claim and nothing here asserts it.

The arrival tests drive a FAKE CLOCK. Sleeping through a 120 s threshold is
not a test, it is a delay, and a controller whose only proof is a sleep can
never be tested at the timescales it actually runs at (nine hours of idle).
"""

import threading
import time
import unittest
from typing import List, Optional

import torch
from torch import nn

from sglang.srt.translator import residency
from sglang.srt.translator.idle_park import (
    IdleParkConfig,
    IdleParkController,
    ParkState,
    WakeTimeout,
    percentile,
)
from sglang.srt.translator.ledger import AudioAssetLedger


class FakeClock:
    """A monotonic clock that only moves when a test says so."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def tiny_module(in_features=64, out_features=128):
    torch.manual_seed(546)
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.Linear(out_features, in_features),
    )


class FakeCt2Handle:
    """A CTranslate2 ``Whisper`` handle, at the surface the route uses.

    Only three members matter and all three are real CTranslate2 4.x API:
    ``unload_model(to_cpu=)``, ``load_model(keep_cache=)`` and the
    ``model_is_loaded`` property. Faking exactly those is what makes this a
    contract test rather than a mock of our own wishes.
    """

    def __init__(self) -> None:
        self.model_is_loaded = True
        self.unload_calls: List[dict] = []
        self.load_calls: List[dict] = []

    def unload_model(self, to_cpu: bool = False) -> None:
        if not self.model_is_loaded:
            raise AssertionError("CTranslate2 was unloaded twice")
        self.unload_calls.append({"to_cpu": to_cpu})
        self.model_is_loaded = False

    def load_model(self, keep_cache: bool = False) -> None:
        if self.model_is_loaded:
            raise AssertionError("CTranslate2 was loaded while already loaded")
        self.load_calls.append({"keep_cache": keep_cache})
        self.model_is_loaded = True


class RecordingLedger(AudioAssetLedger):
    """A ledger that records movement order and can be made to block."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: List[str] = []
        self.park_gate: Optional[threading.Event] = None
        self.restore_gate: Optional[threading.Event] = None
        self.park_entered = threading.Event()
        self.restore_entered = threading.Event()
        self.fail_park = False

    def park_all(self, release_cache: bool = True) -> int:
        self.events.append("park:start")
        self.park_entered.set()
        if self.park_gate is not None:
            self.park_gate.wait(timeout=10.0)
        if self.fail_park:
            self.events.append("park:fail")
            raise RuntimeError("mover exploded")
        freed = super().park_all(release_cache=release_cache)
        self.events.append("park:done")
        return freed

    def ensure_resident(self, *names):
        self.events.append("restore:start")
        self.restore_entered.set()
        out = super().ensure_resident(*names)
        self.events.append("restore:done")
        return out

    def restore_rank(self, rank: int):
        self.events.append("restore:start")
        self.restore_entered.set()
        if self.restore_gate is not None:
            self.restore_gate.wait(timeout=10.0)
        out = super().restore_rank(rank)
        self.events.append("restore:done")
        return out


def build(config=None, clock=None, busy=None, assets=("trunk",)):
    clock = clock or FakeClock()
    ledger = RecordingLedger(clock=clock)
    for name in assets:
        ledger.register(name, tiny_module())
    controller = IdleParkController(
        ledger,
        config or IdleParkConfig(),
        busy_probe=busy,
        clock=clock,
    )
    return controller, ledger, clock


# ---------------------------------------------------------------------------
# A. the decision: arrival patterns
# ---------------------------------------------------------------------------


class TestPercentile(unittest.TestCase):
    def test_nearest_rank_on_small_samples(self):
        self.assertEqual(percentile([], 0.95), 0.0)
        self.assertEqual(percentile([5.0], 0.95), 5.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 100.0], 0.95), 100.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)


class TestArrivalPatterns(unittest.TestCase):
    """The three patterns the policy exists to tell apart."""

    def test_a_burst_conversation_never_parks(self):
        controller, _ledger, clock = build()
        # Two minutes of conversation: a turn every 3-9 s. Cumulatively far
        # past the 120 s floor, which is exactly why a bare timer fails here.
        for gap in [4, 7, 3, 9, 5, 6, 4, 8, 3, 7, 5, 9, 4, 6, 8, 3, 7, 5]:
            clock.advance(gap)
            controller.notify_activity()
            self.assertFalse(
                controller.tick().parked,
                f"parked inside a live conversation after a {gap}s gap",
            )
        self.assertEqual(controller.parks, 0)
        self.assertIs(controller.state, ParkState.RESIDENT)

    def test_a_long_silence_parks_exactly_once(self):
        controller, _ledger, clock = build()
        for gap in (5, 4, 6, 5, 7):
            clock.advance(gap)
            controller.notify_activity()
        # The conversation ends. Ticks every 15 s for ten minutes.
        parks = 0
        for _ in range(40):
            clock.advance(15)
            if controller.tick().parked:
                parks += 1
        self.assertEqual(parks, 1, "an idle tenant must park once, not repeatedly")
        self.assertIs(controller.state, ParkState.PARKED)

    def test_a_stray_request_wakes_once_and_the_dwell_holds_the_repark(self):
        controller, ledger, clock = build()
        clock.advance(600)
        self.assertTrue(controller.tick().parked)

        clock.advance(30)
        self.assertGreaterEqual(controller.ensure_awake(), 0.0)
        self.assertIs(controller.state, ParkState.RESIDENT)
        self.assertEqual(controller.wakes, 1)

        # Silence resumes. The threshold (120 s) is cleared long before the
        # dwell (180 s) expires, so the dwell is what holds -- and it must,
        # or a single stray request becomes a park/wake loop.
        clock.advance(130)
        decision = controller.tick()
        self.assertFalse(decision.parked)
        self.assertIn("dwell", decision.reason)

        clock.advance(60)  # dwell now expired, idle 190 s
        self.assertTrue(controller.tick().parked)
        self.assertEqual(ledger.events.count("park:done"), 2)

    def test_the_gap_that_ends_a_park_is_not_a_conversational_gap(self):
        """The poison the ring buffer must not swallow.

        Nine hours of parked silence, then one request. If that 9 h gap
        entered the percentile, the threshold would jump to the ceiling and
        the tenant would effectively stop parking after its first idle
        period -- the feature would disable itself the first time it worked.
        """
        controller, _ledger, clock = build()
        for gap in (5, 4, 6, 5):
            clock.advance(gap)
            controller.notify_activity()
        before, _ = controller.threshold()

        clock.advance(600)
        controller.tick()
        clock.advance(9 * 3600)
        # The production order: the server announces the arriving request
        # BEFORE it waits for the wake (server.ensure_awake / prefetch_wake
        # both call notify_activity first). So the nine-hour gap is genuinely
        # offered to the ring, and only the park guard keeps it out.
        controller.notify_activity()
        controller.ensure_awake()

        after, terms = controller.threshold()
        self.assertEqual(before, after)
        self.assertLess(terms["gap_p95_x_margin_s"], 60.0)


class TestThresholdTerms(unittest.TestCase):
    def test_the_floor_binds_before_anything_is_measured(self):
        controller, _ledger, _clock = build()
        threshold, terms = controller.threshold()
        self.assertEqual(threshold, 120.0)
        self.assertEqual(terms["gap_p95_x_margin_s"], 0.0)
        self.assertEqual(terms["break_even_s"], 0.0)

    def test_too_few_gaps_leaves_the_inter_arrival_term_undefined(self):
        controller, _ledger, clock = build(
            IdleParkConfig(floor_s=10.0, min_gap_samples=4)
        )
        for _ in range(3):
            clock.advance(30)
            controller.notify_activity()
        _threshold, terms = controller.threshold()
        self.assertEqual(terms["gap_samples"], 3.0)
        self.assertEqual(
            terms["gap_p95_x_margin_s"], 0.0,
            "a percentile guessed from three samples is a learned mistake",
        )

    def test_a_long_conversational_gap_raises_the_bar(self):
        controller, _ledger, clock = build(IdleParkConfig(floor_s=30.0))
        for gap in (5, 4, 200, 6, 5):
            clock.advance(gap)
            controller.notify_activity()
        threshold, terms = controller.threshold()
        self.assertEqual(terms["gap_p95_x_margin_s"], 800.0)
        self.assertEqual(threshold, 800.0)

    def test_the_ceiling_clamps_a_pathological_gap(self):
        controller, _ledger, clock = build(
            IdleParkConfig(floor_s=30.0, ceiling_s=600.0)
        )
        for gap in (5, 4, 5000, 6, 5):
            clock.advance(gap)
            controller.notify_activity()
        threshold, _terms = controller.threshold()
        self.assertEqual(threshold, 600.0)

    def test_a_slow_measured_restore_raises_the_break_even_bar(self):
        controller, ledger, clock = build(IdleParkConfig(floor_s=30.0))
        # Fake a measured 8 s restore, as a cold PCIe leg on a busy card.
        ledger.get("trunk").measured_restore_ms = 8000.0
        threshold, terms = controller.threshold()
        self.assertEqual(terms["break_even_s"], 160.0)
        self.assertEqual(threshold, 160.0)

    def test_a_fast_measured_restore_does_not_lower_the_floor(self):
        controller, ledger, _clock = build(IdleParkConfig(floor_s=120.0))
        ledger.get("trunk").measured_restore_ms = 200.0
        threshold, _terms = controller.threshold()
        self.assertEqual(threshold, 120.0)

    def test_the_config_refuses_an_inverted_clamp(self):
        with self.assertRaises(ValueError) as ctx:
            IdleParkConfig(floor_s=300.0, ceiling_s=100.0).validate()
        self.assertIn("below the floor", str(ctx.exception))


class TestOverrides(unittest.TestCase):
    def test_never_park_beats_enabled(self):
        controller, _ledger, clock = build(
            IdleParkConfig(enabled=True, never_park=True, floor_s=1.0)
        )
        clock.advance(10_000)
        decision = controller.tick()
        self.assertFalse(decision.parked)
        self.assertIn("never-park", decision.reason)
        self.assertFalse(IdleParkConfig(never_park=True).active())

    def test_disabled_never_parks(self):
        controller, _ledger, clock = build(IdleParkConfig(enabled=False, floor_s=1.0))
        clock.advance(10_000)
        self.assertFalse(controller.tick().parked)

    def test_a_turn_in_flight_blocks_the_park(self):
        busy = {"value": True}
        controller, _ledger, clock = build(
            IdleParkConfig(floor_s=1.0), busy=lambda: busy["value"]
        )
        clock.advance(10_000)
        decision = controller.tick()
        self.assertFalse(decision.parked)
        self.assertIn("in flight", decision.reason)
        busy["value"] = False
        self.assertTrue(controller.tick().parked)

    def test_a_ledger_with_no_assets_reports_why(self):
        clock = FakeClock()
        controller = IdleParkController(
            AudioAssetLedger(clock=clock), IdleParkConfig(floor_s=1.0), clock=clock
        )
        clock.advance(100)
        self.assertIn("no ledgered assets", controller.tick().reason)


# ---------------------------------------------------------------------------
# B. the state machine
# ---------------------------------------------------------------------------


class TestStateMachine(unittest.TestCase):
    def test_park_then_wake_returns_to_resident(self):
        controller, _ledger, clock = build()
        self.assertIs(controller.state, ParkState.RESIDENT)
        clock.advance(600)
        self.assertGreater(controller.tick().freed_bytes, 0)
        self.assertIs(controller.state, ParkState.PARKED)
        controller.ensure_awake()
        self.assertIs(controller.state, ParkState.RESIDENT)

    def test_double_park_is_idempotent(self):
        controller, ledger, clock = build()
        clock.advance(600)
        first = controller.park_now("test")
        second = controller.park_now("test")
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(controller.parks, 1)
        self.assertEqual(ledger.events.count("park:start"), 1)
        self.assertEqual(controller.park_refusals, 1)

    def test_waking_a_resident_tenant_costs_nothing(self):
        controller, ledger, _clock = build()
        self.assertEqual(controller.ensure_awake(), 0.0)
        self.assertEqual(ledger.events, [])
        self.assertEqual(controller.wakes, 0)

    def test_double_wake_restores_once(self):
        controller, ledger, clock = build()
        clock.advance(600)
        controller.park_now("test")
        controller.ensure_awake()
        controller.ensure_awake()
        self.assertEqual(ledger.events.count("restore:start"), 1)
        self.assertEqual(controller.wakes, 1)

    def test_a_request_during_a_park_queues_behind_it(self):
        """The ordering the whole state machine exists for.

        A request arriving mid-park must neither race the mover nor be
        refused: it waits for the park to finish, then restores. Asserted on
        the ledger's own event order, so a controller that let the two
        interleave would show restore:start before park:done.
        """
        controller, ledger, clock = build()
        clock.advance(600)
        ledger.park_gate = threading.Event()

        parked: List[int] = []
        woke: List[float] = []
        park_thread = threading.Thread(
            target=lambda: parked.append(controller.park_now("test")), daemon=True
        )
        park_thread.start()
        self.assertTrue(ledger.park_entered.wait(timeout=5.0))
        self.assertIs(controller.state, ParkState.PARKING)

        wake_thread = threading.Thread(
            target=lambda: woke.append(controller.ensure_awake()), daemon=True
        )
        wake_thread.start()
        # The waiter must be blocked, not spinning through to a no-op wake.
        time.sleep(0.2)
        self.assertFalse(ledger.restore_entered.is_set())

        ledger.park_gate.set()
        park_thread.join(timeout=10.0)
        wake_thread.join(timeout=10.0)

        self.assertEqual(
            ledger.events,
            ["park:start", "park:done", "restore:start", "restore:done"],
        )
        self.assertGreater(parked[0], 0)
        self.assertEqual(controller.parks, 1)
        self.assertEqual(controller.wakes, 1)
        self.assertIs(controller.state, ParkState.RESIDENT)

    def test_concurrent_wakes_restore_once_and_all_return(self):
        controller, ledger, clock = build()
        clock.advance(600)
        controller.park_now("test")
        results: List[float] = []
        threads = [
            threading.Thread(
                target=lambda: results.append(controller.ensure_awake()), daemon=True
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        self.assertEqual(len(results), 4)
        self.assertEqual(ledger.events.count("restore:start"), 1)
        self.assertEqual(controller.wakes, 1)

    def test_a_wake_that_waits_too_long_raises_rather_than_hanging(self):
        # Real clock here on purpose: the wake budget is a WALL deadline, so
        # a frozen fake clock would make the deadline unreachable and prove
        # nothing. The threshold logic is not exercised by this test.
        ledger = RecordingLedger()
        ledger.register("trunk", tiny_module())
        controller = IdleParkController(ledger, IdleParkConfig())
        ledger.park_gate = threading.Event()
        park_thread = threading.Thread(
            target=lambda: controller.park_now("test"), daemon=True
        )
        park_thread.start()
        self.assertTrue(ledger.park_entered.wait(timeout=5.0))
        with self.assertRaises(WakeTimeout):
            controller.ensure_awake(timeout_s=0.2)
        ledger.park_gate.set()
        park_thread.join(timeout=15.0)

    def test_a_caller_that_arrived_during_a_park_drives_the_wake_itself(self):
        """No thread may be stranded because it arrived in the wrong state.

        The one-loop shape: a caller that entered while PARKING re-decides
        after the park completes and becomes the wake's driver, rather than
        waiting for a driver that was never going to appear.
        """
        controller, ledger, clock = build()
        clock.advance(600)
        ledger.park_gate = threading.Event()
        park_thread = threading.Thread(
            target=lambda: controller.park_now("test"), daemon=True
        )
        park_thread.start()
        self.assertTrue(ledger.park_entered.wait(timeout=5.0))

        woke: List[float] = []
        wake_thread = threading.Thread(
            target=lambda: woke.append(controller.ensure_awake(timeout_s=10.0)),
            daemon=True,
        )
        wake_thread.start()
        time.sleep(0.1)
        ledger.park_gate.set()
        park_thread.join(timeout=10.0)
        wake_thread.join(timeout=10.0)

        self.assertEqual(len(woke), 1)
        self.assertIs(controller.state, ParkState.RESIDENT)
        self.assertEqual(controller.wakes, 1)

    def test_a_failed_park_leaves_the_tenant_repairable(self):
        """Over-claiming parked is a wasted restore; under-claiming is a crash."""
        controller, ledger, clock = build()
        clock.advance(600)
        ledger.fail_park = True
        with self.assertRaises(RuntimeError):
            controller.park_now("test")
        self.assertIs(controller.state, ParkState.PARKED)
        ledger.fail_park = False
        controller.ensure_awake()
        self.assertIs(controller.state, ParkState.RESIDENT)

    def test_a_failed_wake_reaches_every_waiter_of_that_attempt(self):
        """A failure belongs to ONE attempt, and to all of its waiters.

        Two ways to get this wrong, and both are silent. Clear the error when
        the first waiter raises and the other threads sail on believing the
        assets are back -- a turn then runs against meta tensors. Never clear
        it and every later call inherits a failure it had nothing to do with,
        so one bad wake poisons the tenant for the life of the process.
        """
        # REAL clock: this test's waiters have to be able to TIME OUT. Under
        # a frozen clock a wait that should end in a WakeTimeout instead sits
        # forever, so a broken controller would hang the suite rather than
        # fail it -- and a test that can only hang is not a can-fail proof.
        ledger = RecordingLedger()
        ledger.register("trunk", tiny_module())
        controller = IdleParkController(
            ledger, IdleParkConfig(wake_timeout_s=5.0)
        )
        controller.park_now("test")

        boom = RuntimeError("PCIe fell over")
        calls = {"n": 0}

        def failing_rank(_rank):
            calls["n"] += 1
            raise boom

        ledger.restore_rank = failing_rank

        errors: List[BaseException] = []

        def wait():
            try:
                controller.ensure_awake(timeout_s=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=wait, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        self.assertEqual(len(errors), 3, "a waiter was told the wake succeeded")
        self.assertTrue(all(e is boom for e in errors))
        self.assertIs(controller.state, ParkState.PARKED)

        # A FRESH call must retry, not inherit. Repair the ledger first.
        del ledger.restore_rank
        self.assertGreaterEqual(controller.ensure_awake(timeout_s=10.0), 0.0)
        self.assertIs(controller.state, ParkState.RESIDENT)

    def test_activity_resets_the_idle_clock(self):
        controller, _ledger, clock = build()
        clock.advance(119)
        self.assertFalse(controller.tick().parked)
        controller.notify_activity()
        clock.advance(119)
        self.assertFalse(controller.tick().parked)
        self.assertAlmostEqual(controller.idle_seconds(), 119.0, places=3)


# ---------------------------------------------------------------------------
# C. the bytes actually move
# ---------------------------------------------------------------------------


class TestBytesReallyMove(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.ledger = AudioAssetLedger(clock=self.clock)
        self.module = tiny_module()
        self.reference = {
            k: v.detach().clone() for k, v in self.module.state_dict().items()
        }
        self.probe = torch.randn(4, 64)
        with torch.inference_mode():
            self.expected = self.module(self.probe).clone()
        self.ledger.register("trunk", self.module)
        self.controller = IdleParkController(
            self.ledger, IdleParkConfig(floor_s=10.0), clock=self.clock
        )

    def test_the_park_leaves_no_live_tensor_behind(self):
        self.clock.advance(100)
        freed = self.controller.tick().freed_bytes
        self.assertGreater(freed, 0)
        for tensor in self.module.state_dict().values():
            self.assertEqual(tensor.device.type, "meta")

    def test_a_parked_module_cannot_serve_a_turn(self):
        self.clock.advance(100)
        self.controller.tick()
        with self.assertRaises(Exception):
            with torch.inference_mode():
                self.module(self.probe)

    def test_the_wake_restores_bit_identical_weights(self):
        self.clock.advance(100)
        self.controller.tick()
        self.controller.ensure_awake()
        with torch.inference_mode():
            after = self.module(self.probe)
        self.assertTrue(torch.equal(after, self.expected))
        for key, original in self.reference.items():
            self.assertTrue(torch.equal(self.module.state_dict()[key], original))

    def test_the_park_is_reported_per_device(self):
        self.clock.advance(100)
        self.controller.tick()
        by_device = self.ledger.parked_bytes_by_device()
        self.assertEqual(list(by_device), ["cpu"])
        self.assertGreater(by_device["cpu"], 0)
        # And it clears on the way back, so a stale figure cannot be read as
        # memory that is still available. Checked on the ASSET too, not only
        # on the grouping: the grouping filters on ``parked``, so a retained
        # per-asset figure hides there and surfaces in the health endpoint as
        # "5916 MiB parked" next to a fully resident tenant.
        self.controller.ensure_awake()
        self.assertEqual(self.ledger.parked_bytes_by_device(), {})
        for entry in self.ledger.to_json()["assets"]:
            self.assertFalse(entry["parked"])
            self.assertEqual(
                entry["parked_mib"], 0.0,
                "a resident asset still reports parked bytes",
            )

    def test_the_park_hands_the_pages_back_to_the_driver(self):
        """The step between "tensors freed" and "nvidia-smi drops".

        A desk cannot observe nvidia-smi, so what is pinned here is that the
        release is CALLED as part of every park. Whether ~5.9 GiB actually
        leaves the process is the GPU step's claim (NOTE_546 §6) and nothing
        in this file asserts it.
        """
        calls: List[int] = []
        self.ledger.release_device_cache = lambda: calls.append(1)
        self.clock.advance(100)
        self.controller.tick()
        self.assertEqual(len(calls), 1)

    def test_host_copies_are_page_locked_by_default(self):
        """Pinned staging is worth ~2x on the leg a user waits through.

        Only the decision is checkable here -- ``pin_memory=True`` needs CUDA,
        and on a CPU tensor the ledger correctly skips it. The measured
        restore latency is the GPU step's number.
        """
        self.assertTrue(self.ledger.pin_host_copies)
        self.assertTrue(self.ledger.to_json()["pin_host_copies"])
        self.assertFalse(AudioAssetLedger(pin_host_copies=False).pin_host_copies)

    def test_repeated_cycles_are_stable(self):
        for _ in range(3):
            self.clock.advance(1000)
            self.controller.tick()
            self.controller.ensure_awake()
        with torch.inference_mode():
            after = self.module(self.probe)
        self.assertTrue(torch.equal(after, self.expected))
        self.assertEqual(self.controller.parks, 3)
        self.assertEqual(self.controller.wakes, 3)


# ---------------------------------------------------------------------------
# C2. the staged wake: need order, early release, split latency
# ---------------------------------------------------------------------------


class TestStagedWake(unittest.TestCase):
    """The wake is a pipeline, not a barrier."""

    def _stack(self):
        clock = FakeClock()
        ledger = RecordingLedger(clock=clock)
        # The real deployment's four ledgered modules plus the recognizer.
        for name in ("codec", "talker_trunk", "speaker_encoder", "asr"):
            ledger.register(name, tiny_module())
        controller = IdleParkController(
            ledger, IdleParkConfig(floor_s=10.0), clock=clock
        )
        return controller, ledger, clock

    def test_the_need_order_is_the_pipeline_order_not_the_alphabet(self):
        from sglang.srt.translator.ledger import DEFAULT_WAKE_RANKS

        _controller, ledger, _clock = self._stack()
        order = ledger.wake_order()
        self.assertEqual(order[0], "asr", "the recognizer is what a turn needs first")
        self.assertEqual(order[-1], "codec", "the codec is what a turn needs last")
        self.assertLess(DEFAULT_WAKE_RANKS["asr"], DEFAULT_WAKE_RANKS["talker_trunk"])
        self.assertLess(
            DEFAULT_WAKE_RANKS["talker_trunk"], DEFAULT_WAKE_RANKS["codec"]
        )

    def test_the_recognizer_comes_back_first(self):
        controller, ledger, clock = self._stack()
        clock.advance(100)
        controller.tick()
        restored: List[str] = []
        original = ledger.restore

        def spy(name):
            restored.append(name)
            return original(name)

        ledger.restore = spy
        controller.ensure_awake()
        self.assertEqual(restored[0], "asr")
        self.assertEqual(restored[-1], "codec")

    def test_a_rank_0_caller_is_released_before_the_codec_is_back(self):
        """The whole point: an idle-night turn must not wait for the codec."""
        controller, ledger, clock = self._stack()
        clock.advance(100)
        controller.tick()

        seen: List[str] = []
        original = ledger.restore
        gate = threading.Event()

        def spy(name):
            if name == "codec":
                # Hold the last rank hostage. A barrier-shaped wake would now
                # block the rank-0 caller too.
                gate.wait(timeout=10.0)
            out = original(name)
            seen.append(name)
            return out

        ledger.restore = spy
        controller.ensure_awake(up_to_rank=0, timeout_s=10.0)
        self.assertIn("asr", seen)
        self.assertNotIn("codec", seen)
        self.assertIs(controller.state, ParkState.RESTORING)

        gate.set()
        controller.ensure_awake(timeout_s=10.0)
        self.assertIn("codec", seen)
        self.assertIs(controller.state, ParkState.RESIDENT)

    def test_the_split_latency_is_reported(self):
        controller, _ledger, clock = self._stack()
        clock.advance(100)
        controller.tick()
        controller.ensure_awake()
        report = controller.to_json()
        self.assertIsNotNone(report["last_wake_ms"])
        self.assertIsNotNone(report["last_first_serve_ms"])
        self.assertLessEqual(
            controller.last_first_serve_ms, controller.last_wake_ms,
            "time-to-first-serve cannot exceed time-to-full-restore",
        )

    def test_wake_start_carries_the_ranks_and_the_mib_needed(self):
        residency.clear_sinks()
        seen: List[residency.ResidencyEvent] = []
        residency.add_sink(seen.append)
        self.addCleanup(residency.clear_sinks)

        controller, _ledger, clock = self._stack()
        clock.advance(100)
        controller.tick()
        seen.clear()
        controller.ensure_awake()
        start = next(e for e in seen if e.event == residency.EVENT_WAKE_START)
        self.assertEqual(start.detail["ranks"], [0, 1, 2])

    def test_a_re_park_returns_everything_again(self):
        """Addendum 2: nothing may linger across a park/wake/park cycle."""
        controller, ledger, clock = self._stack()
        clock.advance(100)
        first = controller.tick().freed_bytes
        first_by_device = dict(ledger.parked_bytes_by_device())

        controller.ensure_awake()
        self.assertEqual(ledger.parked_bytes_by_device(), {})

        clock.advance(10_000)  # past the dwell and the threshold
        second = controller.tick().freed_bytes
        self.assertEqual(
            second, first,
            "the second park freed a different amount; something stayed "
            "behind across the cycle",
        )
        self.assertEqual(ledger.parked_bytes_by_device(), first_by_device)
        self.assertTrue(all(a["parked"] for a in ledger.to_json()["assets"]))
        self.assertEqual(ledger.to_json()["resident_mib"], 0.0)


# ---------------------------------------------------------------------------
# D. the park route (non-torch assets)
# ---------------------------------------------------------------------------


class TestParkRoute(unittest.TestCase):
    def _route(self):
        from sglang.srt.translator.asr_backends import CtranslateWhisperParkRoute

        handle = FakeCt2Handle()
        return handle, CtranslateWhisperParkRoute(handle, device="cuda")

    def test_the_route_spills_to_host_and_reloads(self):
        handle, route = self._route()
        route.park()
        self.assertEqual(handle.unload_calls, [{"to_cpu": True}])
        self.assertFalse(handle.model_is_loaded)
        route.restore()
        self.assertEqual(handle.load_calls, [{"keep_cache": True}])
        self.assertTrue(handle.model_is_loaded)

    def test_to_cpu_is_not_optional(self):
        """A bare unload would re-read 1.5 GiB from disk on every wake."""
        handle, route = self._route()
        route.park()
        self.assertTrue(handle.unload_calls[0]["to_cpu"])

    def test_the_route_is_idempotent_in_both_directions(self):
        handle, route = self._route()
        route.park()
        route.park()
        route.restore()
        route.restore()
        self.assertEqual(len(handle.unload_calls), 1)
        self.assertEqual(len(handle.load_calls), 1)

    def test_a_handle_without_the_api_is_refused_at_registration(self):
        from sglang.srt.translator.asr_backends import CtranslateWhisperParkRoute
        from sglang.srt.translator.backends import BackendError

        class NotAWhisper:
            pass

        with self.assertRaises(BackendError) as ctx:
            CtranslateWhisperParkRoute(NotAWhisper())
        self.assertIn("unload_model", str(ctx.exception))

    def test_an_unmeasurable_size_reads_as_unknown_not_as_free(self):
        _handle, route = self._route()
        # No CUDA here, so the free-memory delta is unknowable.
        self.assertEqual(route.size_bytes(), 0)

    def test_a_route_asset_parks_through_the_same_ledger_and_controller(self):
        clock = FakeClock()
        ledger = AudioAssetLedger(clock=clock)
        handle, route = self._route()
        ledger.register("tts_trunk", tiny_module())
        ledger.register_route("asr", route)
        controller = IdleParkController(
            ledger, IdleParkConfig(floor_s=10.0), clock=clock
        )

        clock.advance(100)
        self.assertTrue(controller.tick().parked)
        self.assertFalse(handle.model_is_loaded)
        self.assertTrue(ledger.get("asr").parked)
        self.assertTrue(ledger.get("tts_trunk").parked)

        controller.ensure_awake()
        self.assertTrue(handle.model_is_loaded)
        self.assertFalse(ledger.get("asr").parked)

    def test_a_route_asset_reports_itself_in_the_ledger_json(self):
        ledger = AudioAssetLedger()
        _handle, route = self._route()
        ledger.register_route("asr", route)
        entry = ledger.to_json()["assets"][0]
        self.assertTrue(entry["route"])
        self.assertEqual(entry["device"], "cuda:0")


# ---------------------------------------------------------------------------
# E. residency events (the #553 coupling surface)
# ---------------------------------------------------------------------------


class TestResidencyEvents(unittest.TestCase):
    def setUp(self):
        residency.clear_sinks()
        residency.reset_card_cache()
        self.seen: List[residency.ResidencyEvent] = []
        residency.add_sink(self.seen.append)
        self.addCleanup(residency.clear_sinks)

    def _await_events(self, count: int, timeout_s: float = 5.0) -> None:
        """wake_complete is a REPORT, emitted by the wake worker after the
        waiters are released -- deliberately, so a slow telemetry consumer
        cannot sit inside a user's latency. So it is polled for, not assumed
        to have landed by the time ensure_awake returns."""
        deadline = time.monotonic() + timeout_s
        while len(self.seen) < count and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_a_park_and_a_wake_emit_the_three_events_in_order(self):
        controller, _ledger, clock = build()
        clock.advance(600)
        controller.tick()
        controller.ensure_awake()
        self._await_events(3)
        self.assertEqual(
            [e.event for e in self.seen],
            [
                residency.EVENT_PARK_COMPLETE,
                residency.EVENT_WAKE_START,
                residency.EVENT_WAKE_COMPLETE,
            ],
        )

    def test_wake_start_fires_BEFORE_the_restore(self):
        """The asymmetry a consumer's correctness depends on.

        park_complete after (only released memory may be claimed);
        wake_start before (a consumer must be told while it can still act).
        """
        controller, ledger, clock = build()
        clock.advance(600)
        controller.tick()
        order: List[str] = []
        residency.clear_sinks()
        residency.add_sink(lambda e: order.append(f"event:{e.event}"))
        ledger.events.clear()
        controller.ensure_awake()
        self.assertEqual(order[0], f"event:{residency.EVENT_WAKE_START}")
        self.assertEqual(ledger.events[0], "restore:start")

    def test_the_event_carries_the_tenant_and_the_freed_mib(self):
        controller, _ledger, clock = build()
        clock.advance(600)
        controller.tick()
        event = self.seen[0]
        self.assertEqual(event.tenant_id, "translator")
        self.assertEqual(event.event, residency.EVENT_PARK_COMPLETE)
        self.assertGreater(event.detail["freed_mib"], 0.0)

    def test_card_identity_is_nvml_or_honestly_absent(self):
        cards = residency.cards_from_bytes({"cuda:0": 4 << 20})
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertAlmostEqual(card.mib, 4.0)
        # No NVML on this desk -> unresolved, and the event says so rather
        # than reporting the torch ordinal as a physical index.
        self.assertFalse(card.card_resolved)
        self.assertIsNone(card.nvml_index)
        self.assertEqual(card.to_json()["card_resolved"], False)

    def test_host_side_bytes_are_not_reported_as_card_residency(self):
        self.assertEqual(residency.cards_from_bytes({"cpu": 8 << 20}), ())

    def test_the_marker_line_is_machine_readable(self):
        import json

        event = residency.ResidencyEvent(
            tenant_id="translator",
            event=residency.EVENT_PARK_COMPLETE,
            cards=(
                residency.CardResidency(
                    card_uuid="abc", nvml_index=1, card_name="RTX 5090", mib=5916.0
                ),
            ),
        )
        line = event.marker_line()
        self.assertTrue(line.startswith(residency.MARKER + " "))
        payload = json.loads(line.split(residency.MARKER + " ", 1)[1])
        self.assertEqual(payload["cards"][0]["nvml_index"], 1)
        self.assertEqual(payload["cards"][0]["mib"], 5916.0)
        self.assertEqual(payload["total_mib"], 5916.0)

    def test_an_unknown_event_name_is_refused(self):
        with self.assertRaises(ValueError):
            residency.ResidencyEvent(tenant_id="t", event="parked_maybe", cards=())

    def test_a_broken_sink_cannot_take_the_tenant_down(self):
        def explode(_event):
            raise RuntimeError("consumer is down")

        residency.add_sink(explode)
        controller, _ledger, clock = build()
        clock.advance(600)
        self.assertTrue(controller.tick().parked)
        self.assertIs(controller.state, ParkState.PARKED)

    def test_emit_itself_swallows_a_broken_sink(self):
        """Pinned at the emitter, not only at the controller.

        The controller has its own guard around ``_emit``, so a controller-
        level test passes even if ``emit`` propagates -- defence in depth
        hides which layer is doing the defending. This asserts the emitter's
        own contract directly, and that a later sink still runs: a fan-out
        that stops at the first broken consumer would silently cut every
        consumer registered after it.
        """
        seen_after = []

        def explode(_event):
            raise RuntimeError("consumer is down")

        residency.add_sink(explode)
        residency.add_sink(seen_after.append)
        event = residency.ResidencyEvent(
            tenant_id="translator",
            event=residency.EVENT_PARK_COMPLETE,
            cards=(),
        )
        residency.emit(event)  # must not raise
        self.assertEqual(len(seen_after), 1)


# ---------------------------------------------------------------------------
# F. the #286 register keeps honest books
# ---------------------------------------------------------------------------


class TestRegisterAccounting(unittest.TestCase):
    def setUp(self):
        import os
        from unittest.mock import patch

        from sglang.srt.model_executor import offload_register

        self.module_under_test = offload_register
        env = patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.register = offload_register.configure_global_register("capacity")
        self.addCleanup(offload_register.reset_global_register)

    def test_a_parked_asset_is_parked_in_the_register_too(self):
        clock = FakeClock()
        ledger = AudioAssetLedger(clock=clock)
        ledger.register("trunk", tiny_module())
        self.assertTrue(ledger.registered_with_runtime)

        controller = IdleParkController(
            ledger, IdleParkConfig(floor_s=10.0), clock=clock
        )
        clock.advance(100)
        controller.tick()

        parked_bytes = self.register._parked_bytes_locked("audio_modules")
        self.assertGreater(
            parked_bytes, 0,
            "the register still reads an idle, host-resident tenant as "
            "device-resident; every planner pricing this class is wrong",
        )
        controller.ensure_awake()
        self.assertEqual(self.register._parked_bytes_locked("audio_modules"), 0)

    def test_mark_parked_ignores_an_unknown_item(self):
        self.module_under_test.maybe_mark_parked("translator:nope", True)  # no raise

    def test_mark_parked_is_idempotent_in_the_stats(self):
        self.register.register(
            item_id="t:x", offload_class="audio_modules",
            size_bytes=1024, restore_cost_ms=1.0,
        )
        before = self.register.stats.parks
        self.register.mark_parked("t:x", True)
        self.register.mark_parked("t:x", True)
        self.assertEqual(self.register.stats.parks, before + 1)


# ---------------------------------------------------------------------------
# G. the config surface
# ---------------------------------------------------------------------------


class TestConfigSurface(unittest.TestCase):
    def test_the_launcher_exposes_the_documented_flags_and_defaults(self):
        from sglang.srt.translator.launch import build_parser

        args = build_parser().parse_args([])
        self.assertTrue(args.idle_park, "the translator deployment defaults to ON")
        self.assertFalse(args.never_park)
        self.assertEqual(args.idle_park_floor_s, 120.0)
        self.assertEqual(args.idle_park_ceiling_s, 900.0)
        self.assertEqual(args.idle_park_gap_margin, 4.0)
        self.assertEqual(args.idle_park_break_even, 20.0)
        self.assertEqual(args.idle_park_dwell_s, 180.0)
        self.assertEqual(args.residency_event_url, "")

    def test_the_off_switches_parse(self):
        from sglang.srt.translator.launch import build_parser

        args = build_parser().parse_args(["--no-idle-park"])
        self.assertFalse(args.idle_park)
        args = build_parser().parse_args(["--never-park"])
        self.assertTrue(args.never_park)
        self.assertTrue(args.idle_park)  # never-park wins at the config layer

    def test_the_deployment_config_carries_the_park_settings(self):
        from sglang.srt.translator.config import TranslatorConfig

        config = TranslatorConfig()
        config.validate()
        self.assertTrue(config.idle_park.enabled)
        self.assertEqual(config.idle_park.floor_s, 120.0)

    def test_an_invalid_park_config_is_refused_by_the_deployment(self):
        from sglang.srt.translator.config import (
            TranslatorConfig,
            TranslatorConfigError,
        )

        config = TranslatorConfig(idle_park=IdleParkConfig(gap_margin=0.0))
        with self.assertRaises(TranslatorConfigError):
            config.validate()

    def test_an_audio_frame_is_not_an_arrival(self):
        """The signal would be destroyed by counting frames.

        Frames arrive every few tens of milliseconds while a microphone is
        open. If ``prefetch_wake`` counted them, the gap ring would fill with
        millisecond gaps, the p95 term would sit at ~0 forever, and the policy
        would silently collapse back into the fixed timer it replaces. So the
        service's prefetch path must not touch the controller's arrival
        bookkeeping -- only a turn or a real REST request may.
        """
        import ast
        import inspect
        import textwrap

        from sglang.srt.translator.server import TranslatorService

        def calls(fn) -> bool:
            # The CALL, not the word: the docstring names the rule it obeys.
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            return any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "notify_activity"
                for n in ast.walk(tree)
            )

        self.assertFalse(
            calls(TranslatorService.prefetch_wake),
            "prefetch_wake counts an audio frame as an arrival; the "
            "inter-arrival percentile is now measuring the frame rate",
        )
        self.assertTrue(
            calls(TranslatorService.ensure_awake),
            "no path records an arrival at all, so the gap ring stays empty",
        )

    def test_the_report_explains_the_current_threshold(self):
        controller, _ledger, clock = build()
        clock.advance(60)
        controller.tick()
        report = controller.to_json()
        self.assertEqual(report["state"], "resident")
        self.assertEqual(report["threshold_s"], 120.0)
        self.assertIn("floor_s", report["terms"])
        self.assertIn("has not reached", report["last_decision"]["reason"])


# ---------------------------------------------------------------------------
# H. the asset class survives #488
# ---------------------------------------------------------------------------


class TestAssetClassProvenance(unittest.TestCase):
    def test_the_park_is_defined_over_the_ledger_asset_class(self):
        """#488 replaces the BACKENDS, not the class or the state machine."""
        from sglang.srt.model_executor.short_term_offload_register import (
            ASSET_CLASSES,
            LadderRank,
        )
        from sglang.srt.translator.ledger import OFFLOAD_CLASS

        descriptor = ASSET_CLASSES[OFFLOAD_CLASS]
        self.assertEqual(descriptor.ladder_rank, LadderRank.COLD_SECOND_MODEL)
        self.assertTrue(descriptor.wired)

    def test_a_new_backend_joins_by_supplying_a_route_only(self):
        clock = FakeClock()
        ledger = AudioAssetLedger(clock=clock)

        class NativeLaneRoute:
            """What a #488 backend would register: three methods, no more."""

            def __init__(self):
                self.parked = False

            @property
            def device(self):
                return "cuda:0"

            def park(self) -> int:
                self.parked = True
                return 123 << 20

            def restore(self) -> None:
                self.parked = False

            def size_bytes(self) -> int:
                return 0 if self.parked else 123 << 20

        route = NativeLaneRoute()
        ledger.register_route("native_talker", route)
        controller = IdleParkController(
            ledger, IdleParkConfig(floor_s=1.0), clock=clock
        )
        clock.advance(100)
        self.assertEqual(controller.tick().freed_bytes, 123 << 20)
        self.assertTrue(route.parked)
        controller.ensure_awake()
        self.assertFalse(route.parked)


if __name__ == "__main__":
    unittest.main()
