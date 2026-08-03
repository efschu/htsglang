"""Universal client liveness: detection, grace, and not blocking anybody (#344b).

Three shapes of dead client are tested per stream class, because they fail
differently and only one of them was ever handled:

* **closes hard** -- the transport reports it, the generator's ``finally``
  runs, nothing to detect. This is the case that already worked and it is
  here so a regression in it is visible.
* **stops reading, never closes** -- the socket stays open, the generator is
  suspended at its ``yield``, its ``finally`` never runs, and back-pressure
  holds every resource behind it. Only a timeout distinguishes this from a
  slow viewer.
* **vanishes** -- the consumer task is gone and nobody will ever close the
  generator. Same suspension, no one left to notice.

And one gate that all of them share: **a healthy client on the same server
keeps working throughout**. A liveness mechanism that stalls the event loop
while it tears down a dead client has traded one outage for another, so every
timing test here runs a second, well-behaved stream alongside and asserts it
kept receiving frames while the first one was declared dead and released.

Timing tests use real (short) durations rather than a fake clock on purpose:
the property under test is concurrency, and a clock the test advances by hand
cannot show that two streams progressed independently. Policy and arithmetic
tests use the fake clock, where wall time would buy nothing.
"""

import asyncio
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from sglang.srt.liveness import (
    DEFAULT_TIMEOUT_RATIONALE,
    DEFAULT_TIMEOUTS_S,
    Attachment,
    AttachmentPhase,
    AttachmentRegistry,
    ClaimKind,
    ConsumerGone,
    ConsumerWatchdog,
    EndpointClass,
    LedgerGraceBridge,
    LivenessConfig,
    LivenessPolicy,
    ResourceClaim,
    attach_ledger_grace_bridge,
    await_with_liveness,
    guard_generate_stream,
    guarded_stream,
)
from sglang.srt.registry.ledger import (
    MIB,
    ReservationEntry,
    ReservationStore,
    TenantState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


# The timeout every wall-clock test in this file uses. Long enough that a
# loaded CI box does not declare a healthy stream dead, short enough that the
# whole file stays under half a minute.
FAST_TIMEOUT_S = 0.4
FAST_POLL_S = 0.02


def fast_policy(endpoint_class=EndpointClass.LLM_STREAM, **kwargs) -> LivenessPolicy:
    params = dict(
        endpoint_class=endpoint_class,
        timeout_s=FAST_TIMEOUT_S,
        poll_interval_s=FAST_POLL_S,
        teardown_timeout_s=1.0,
    )
    params.update(kwargs)
    return LivenessPolicy(**params)


class FakeClock:
    """A clock the test drives, so a 900 s timeout costs no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def frame_source(count: int = 10_000, delay: float = 0.005):
    """An endless-enough producer. One frame is one accepted write."""
    for index in range(count):
        await asyncio.sleep(delay)
        yield f"data: {index}\n\n".encode()


class Released:
    """Records that a release ran, and how long the whole thing took."""

    def __init__(self) -> None:
        self.calls = 0
        self.event = asyncio.Event()

    async def __call__(self) -> None:
        self.calls += 1
        self.event.set()


# ---------------------------------------------------------------------------
# The class table
# ---------------------------------------------------------------------------


class PolicyTableTest(CustomTestCase):
    def test_every_class_has_a_default_and_a_recorded_reason(self):
        """A number without an argument behind it is not a policy."""
        for endpoint_class in EndpointClass:
            self.assertIn(endpoint_class, DEFAULT_TIMEOUTS_S, endpoint_class.value)
            self.assertIn(
                endpoint_class, DEFAULT_TIMEOUT_RATIONALE, endpoint_class.value
            )
            self.assertGreater(len(DEFAULT_TIMEOUT_RATIONALE[endpoint_class]), 40)

    def test_the_most_contended_resource_gets_the_least_patience(self):
        """KV blocks are scarcer than a subscriber queue, so bound them harder."""
        self.assertLess(
            DEFAULT_TIMEOUTS_S[EndpointClass.LLM_STREAM],
            DEFAULT_TIMEOUTS_S[EndpointClass.VIDEO_STREAM],
        )
        self.assertLess(
            DEFAULT_TIMEOUTS_S[EndpointClass.PREVIEW_TAP],
            DEFAULT_TIMEOUTS_S[EndpointClass.LLM_STREAM],
        )

    def test_the_registry_lease_default_matches_the_ledger(self):
        """Two numbers for one lease is a drift waiting to happen."""
        from sglang.srt.registry.ledger import DEFAULT_LEASE_SECONDS

        self.assertEqual(
            DEFAULT_TIMEOUTS_S[EndpointClass.REGISTRY_LEASE], DEFAULT_LEASE_SECONDS
        )

    def test_grace_starts_before_death_and_never_after_it(self):
        policy = LivenessPolicy(
            endpoint_class=EndpointClass.LLM_STREAM, timeout_s=100.0
        )
        self.assertAlmostEqual(policy.resolved_grace_after(), 25.0)
        self.assertLess(policy.resolved_grace_after(), policy.resolved_timeout())
        # An operator who asks for a grace window longer than the timeout gets
        # the timeout, not an unreachable state.
        clamped = LivenessPolicy(timeout_s=10.0, grace_after_s=999.0)
        self.assertEqual(clamped.resolved_grace_after(), 10.0)
        # And a fraction of 1.0 is how the grace window is turned off.
        off = LivenessPolicy(timeout_s=10.0, grace_fraction=1.0)
        self.assertEqual(off.resolved_grace_after(), 10.0)

    def test_a_disabled_class_has_no_timeout_and_no_grace(self):
        policy = LivenessPolicy(endpoint_class=EndpointClass.VIDEO_STREAM, timeout_s=0)
        self.assertIsNone(policy.resolved_timeout())
        self.assertIsNone(policy.resolved_grace_after())


class ConfigTest(CustomTestCase):
    def test_a_spec_string_sets_only_the_classes_it_names(self):
        config = LivenessConfig.parse("llm_stream=12,video_stream=34")
        self.assertEqual(
            config.policy_for(EndpointClass.LLM_STREAM).resolved_timeout(), 12.0
        )
        self.assertEqual(
            config.policy_for(EndpointClass.VIDEO_STREAM).resolved_timeout(), 34.0
        )
        self.assertEqual(
            config.policy_for(EndpointClass.PREVIEW_TAP).resolved_timeout(),
            DEFAULT_TIMEOUTS_S[EndpointClass.PREVIEW_TAP],
        )

    def test_an_unknown_class_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            LivenessConfig.parse("teleportation=5")
        self.assertIn("teleportation", str(ctx.exception))

    def test_a_bare_word_is_refused_rather_than_ignored(self):
        """``--client-liveness-timeouts llm_stream`` is a typo, not a request."""
        with self.assertRaises(ValueError):
            LivenessConfig.parse("llm_stream")

    def test_describe_covers_every_class(self):
        described = LivenessConfig().describe()
        self.assertEqual(set(described), {c.value for c in EndpointClass})

    def test_the_server_flags_exist_under_the_documented_names(self):
        from sglang.srt.server_args import ServerArgs

        names = {f.name for f in fields(ServerArgs)}
        for flag in (
            "client_liveness_timeouts",
            "client_liveness_poll_interval_s",
            "client_liveness_teardown_timeout_s",
            "client_liveness_grace_fraction",
        ):
            self.assertIn(flag, names, flag)


# ---------------------------------------------------------------------------
# The watchdog, per class
# ---------------------------------------------------------------------------


class WatchdogTest(CustomTestCase):
    def test_silence_beyond_the_class_timeout_releases(self):
        clock = FakeClock()

        async def run():
            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="j1",
                policy=LivenessPolicy(
                    endpoint_class=EndpointClass.LLM_STREAM,
                    timeout_s=100.0,
                    poll_interval_s=0.01,
                ),
                release=released,
                clock=clock,
            )
            watchdog.start()
            await asyncio.sleep(0.05)
            self.assertEqual(released.calls, 0)
            clock.advance(101.0)
            await asyncio.wait_for(released.event.wait(), timeout=2.0)
            self.assertTrue(watchdog.released)
            self.assertTrue(watchdog.state.declared_dead)
            await watchdog.stop()

        asyncio.run(run())

    def test_progress_resets_the_clock_indefinitely(self):
        """A slow consumer that is still there is not a dead consumer."""
        clock = FakeClock()

        async def run():
            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="j2",
                policy=LivenessPolicy(timeout_s=100.0, poll_interval_s=0.01),
                release=released,
                clock=clock,
            )
            watchdog.start()
            for _ in range(5):
                clock.advance(90.0)
                await asyncio.sleep(0.03)
                watchdog.note_progress(1)
            self.assertEqual(released.calls, 0)
            await watchdog.stop()

        asyncio.run(run())

    def test_a_release_that_hangs_is_bounded_not_awaited_forever(self):
        async def run():
            entered = asyncio.Event()

            async def stuck_release():
                entered.set()
                await asyncio.sleep(3600)

            watchdog = ConsumerWatchdog(
                job_id="j3",
                policy=LivenessPolicy(
                    timeout_s=0.05, poll_interval_s=0.01, teardown_timeout_s=0.1
                ),
                release=stuck_release,
            )
            watchdog.start()
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            for _ in range(200):
                if watchdog.released:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(watchdog.released, "teardown timeout did not escalate")
            await watchdog.stop()

        asyncio.run(run())

    def test_every_endpoint_class_can_be_watched(self):
        """No class is wired to a policy the watchdog cannot construct."""
        clock = FakeClock()

        async def run():
            for endpoint_class in EndpointClass:
                released = Released()
                watchdog = ConsumerWatchdog(
                    job_id=f"j-{endpoint_class.value}",
                    policy=LivenessPolicy(
                        endpoint_class=endpoint_class, poll_interval_s=0.01
                    ),
                    release=released,
                    clock=clock,
                )
                watchdog.start()
                clock.advance(DEFAULT_TIMEOUTS_S[endpoint_class] + 1.0)
                await asyncio.wait_for(released.event.wait(), timeout=2.0)
                self.assertEqual(released.calls, 1, endpoint_class.value)
                await watchdog.stop()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# The three shapes of dead client, on a real stream, with a healthy neighbour
# ---------------------------------------------------------------------------


class DeadClientShapesTest(CustomTestCase):
    """Each test runs a doomed stream and a healthy one on the same loop."""

    def _healthy(self, counter):
        """A well-behaved consumer that keeps reading for the whole test."""
        released = Released()
        watchdog = ConsumerWatchdog(
            job_id="healthy", policy=fast_policy(), release=released
        )
        stream = guarded_stream(frame_source(), watchdog)

        async def drain():
            async for _ in stream:
                counter.append(1)

        return watchdog, released, drain, stream

    def _assert_healthy_kept_working(self, task, counter, released, before):
        self.assertFalse(released.event.is_set(), "the healthy stream was dropped")
        self.assertGreater(
            len(counter) - before,
            0,
            "the healthy stream received nothing while the dead one was torn down",
        )
        task.cancel()

    def test_a_consumer_that_stops_reading_without_closing_is_released(self):
        async def run():
            counter = []
            healthy_wd, healthy_rel, healthy_drain, _hs = self._healthy(counter)
            healthy_task = asyncio.create_task(healthy_drain())

            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="stalled", policy=fast_policy(), release=released
            )
            stream = guarded_stream(frame_source(), watchdog)

            async def stall():
                seen = 0
                async for _ in stream:
                    seen += 1
                    if seen >= 2:
                        # No break, no aclose: the socket is open, the window
                        # is full, and this generator will never resume.
                        await asyncio.sleep(3600)

            stalled_task = asyncio.create_task(stall())
            await asyncio.sleep(0.1)
            before = len(counter)
            await asyncio.wait_for(released.event.wait(), timeout=FAST_TIMEOUT_S * 6)
            self.assertEqual(released.calls, 1)
            self._assert_healthy_kept_working(
                healthy_task, counter, healthy_rel, before
            )
            stalled_task.cancel()
            await healthy_wd.stop()
            await watchdog.stop()

        asyncio.run(run())

    def test_a_consumer_that_closes_hard_is_torn_down_without_the_timeout(self):
        """The pre-existing path. It must still work, and must not fire late."""

        async def run():
            counter = []
            healthy_wd, healthy_rel, healthy_drain, _hs = self._healthy(counter)
            healthy_task = asyncio.create_task(healthy_drain())

            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="closer", policy=fast_policy(), release=released
            )
            stream = guarded_stream(frame_source(), watchdog)

            seen = 0
            async for _ in stream:
                seen += 1
                if seen >= 3:
                    break
            await stream.aclose()

            self.assertGreaterEqual(seen, 3)
            # The release belongs to the timeout path. A clean close runs the
            # generator's own finally instead, so it must not have fired.
            self.assertEqual(released.calls, 0)
            self.assertFalse(watchdog.state.declared_dead)

            before = len(counter)
            await asyncio.sleep(FAST_TIMEOUT_S * 2)
            self.assertEqual(released.calls, 0, "a closed stream was released twice")
            self._assert_healthy_kept_working(
                healthy_task, counter, healthy_rel, before
            )
            await healthy_wd.stop()

        asyncio.run(run())

    def test_a_consumer_that_vanishes_mid_stream_is_released(self):
        """The client process is gone. Nobody will ever close the generator."""

        async def run():
            counter = []
            healthy_wd, healthy_rel, healthy_drain, _hs = self._healthy(counter)
            healthy_task = asyncio.create_task(healthy_drain())

            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="vanished", policy=fast_policy(), release=released
            )
            # The reference is held for the whole test, exactly as the ASGI
            # server holds it: without it the generator would be finalized by
            # the interpreter and the case under test would not exist.
            stream = guarded_stream(frame_source(), watchdog)
            got_two = asyncio.Event()

            async def consume():
                seen = 0
                async for _ in stream:
                    seen += 1
                    if seen >= 2:
                        got_two.set()
                        await asyncio.sleep(3600)

            consumer = asyncio.create_task(consume())
            await asyncio.wait_for(got_two.wait(), timeout=2.0)
            # Killed while parked between frames, so the generator stays
            # suspended at its yield rather than being thrown into.
            consumer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await consumer

            before = len(counter)
            await asyncio.wait_for(released.event.wait(), timeout=FAST_TIMEOUT_S * 6)
            self.assertEqual(released.calls, 1)
            self._assert_healthy_kept_working(
                healthy_task, counter, healthy_rel, before
            )
            await healthy_wd.stop()
            await watchdog.stop()

        asyncio.run(run())

    def test_release_is_bounded_by_the_configured_timeout(self):
        """The number in the flag is the number that is honoured."""

        async def run():
            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="timed",
                policy=fast_policy(timeout_s=0.25),
                release=released,
            )
            stream = guarded_stream(frame_source(), watchdog)
            loop = asyncio.get_running_loop()

            async def stall():
                async for _ in stream:
                    await asyncio.sleep(3600)

            task = asyncio.create_task(stall())
            start = loop.time()
            await asyncio.wait_for(released.event.wait(), timeout=3.0)
            elapsed = loop.time() - start
            self.assertGreaterEqual(elapsed, 0.25)
            # Generous upper bound: the assertion is that the release tracks
            # the configured duration, not that the loop is real-time.
            self.assertLess(elapsed, 2.0)
            task.cancel()
            await watchdog.stop()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# The token-stream wiring
# ---------------------------------------------------------------------------


class FakeTokenizerManager:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    def abort_request(self, rid: str = "", abort_all: bool = False) -> None:
        self.aborted.append(rid)


class FakeStreamingResponse:
    def __init__(self, body_iterator) -> None:
        self.body_iterator = body_iterator


class FakeObj:
    def __init__(self, rid) -> None:
        self.rid = rid


class GenerateStreamGuardTest(CustomTestCase):
    def test_a_stalled_token_stream_aborts_its_request(self):
        """The abort is what frees KV. Nothing else in the chain does."""

        async def run():
            manager = FakeTokenizerManager()
            config = LivenessConfig(
                timeouts_s={EndpointClass.LLM_STREAM.value: FAST_TIMEOUT_S},
                poll_interval_s=FAST_POLL_S,
            )
            response = guard_generate_stream(
                FakeStreamingResponse(frame_source()),
                tokenizer_manager=manager,
                obj=FakeObj("rid-42"),
                config=config,
            )

            async def stall():
                async for _ in response.body_iterator:
                    await asyncio.sleep(3600)

            task = asyncio.create_task(stall())
            for _ in range(300):
                if manager.aborted:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(manager.aborted, ["rid-42"])
            task.cancel()

        asyncio.run(run())

    def test_a_batched_request_aborts_every_rid(self):
        async def run():
            manager = FakeTokenizerManager()
            config = LivenessConfig(
                timeouts_s={EndpointClass.LLM_STREAM.value: FAST_TIMEOUT_S},
                poll_interval_s=FAST_POLL_S,
            )
            response = guard_generate_stream(
                FakeStreamingResponse(frame_source()),
                tokenizer_manager=manager,
                obj=FakeObj(["a", "b", "c"]),
                config=config,
            )

            async def stall():
                async for _ in response.body_iterator:
                    await asyncio.sleep(3600)

            task = asyncio.create_task(stall())
            for _ in range(300):
                if len(manager.aborted) == 3:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(sorted(manager.aborted), ["a", "b", "c"])
            task.cancel()

        asyncio.run(run())

    def test_a_response_without_a_request_id_is_returned_untouched(self):
        manager = FakeTokenizerManager()
        source = frame_source()
        response = FakeStreamingResponse(source)
        guarded = guard_generate_stream(
            response, tokenizer_manager=manager, obj=FakeObj(None)
        )
        self.assertIs(guarded.body_iterator, source)

    def test_a_non_streaming_response_is_returned_untouched(self):
        manager = FakeTokenizerManager()
        sentinel = object()
        self.assertIs(
            guard_generate_stream(
                sentinel, tokenizer_manager=manager, obj=FakeObj("r")
            ),
            sentinel,
        )

    def test_a_revived_consumer_after_death_is_ended_not_resumed(self):
        """Its resources are gone; streaming on would stream from a corpse."""

        async def run():
            watchdog = ConsumerWatchdog(
                job_id="revived",
                policy=fast_policy(timeout_s=0.1),
                release=Released(),
            )
            stream = guarded_stream(frame_source(delay=0.0), watchdog)
            self.assertIsNotNone(await stream.__anext__())
            await asyncio.sleep(0.5)
            self.assertTrue(watchdog.released)
            with self.assertRaises(ConsumerGone):
                await stream.__anext__()
            await watchdog.stop()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Grace: held, in grace, reclaimable -- versus actively used
# ---------------------------------------------------------------------------


class GraceRegistryTest(CustomTestCase):
    def test_an_active_attachment_is_not_reclaimable(self):
        registry = AttachmentRegistry()
        attachment = registry.register(
            "a",
            endpoint_class=EndpointClass.VIDEO_STREAM.value,
            claims=[ResourceClaim(ClaimKind.VRAM_LEASE, "card-0", 8 * MIB, "t")],
        )
        self.assertFalse(attachment.reclaimable)
        self.assertEqual(registry.reclaimable_bytes(), 0)

    def test_grace_makes_the_claims_visible_to_the_ladder(self):
        registry = AttachmentRegistry()
        registry.register(
            "a",
            endpoint_class=EndpointClass.VIDEO_STREAM.value,
            claims=[ResourceClaim(ClaimKind.VRAM_LEASE, "card-0", 8 * MIB, "t")],
        )
        registry.set_phase("a", AttachmentPhase.GRACE, silent_for_s=9.0)
        self.assertEqual(registry.reclaimable_bytes(), 8 * MIB)
        self.assertEqual(
            registry.reclaimable_bytes(ClaimKind.VRAM_LEASE, "card-0"), 8 * MIB
        )
        self.assertEqual(registry.reclaimable_bytes(ClaimKind.VRAM_LEASE, "card-1"), 0)
        self.assertEqual(len(registry.reclaimable()), 1)

    def test_a_declared_dead_attachment_is_not_offered_to_a_second_reclaimer(self):
        """Its own release is already running; racing it tears down twice."""
        registry = AttachmentRegistry()
        registry.register(
            "a",
            endpoint_class=EndpointClass.VIDEO_STREAM.value,
            claims=[ResourceClaim(ClaimKind.VRAM_LEASE, "card-0", 8 * MIB, "t")],
        )
        registry.set_phase("a", AttachmentPhase.GRACE)
        registry.set_phase("a", AttachmentPhase.DEAD)
        self.assertEqual(registry.reclaimable_bytes(), 0)

    def test_observers_fire_on_change_only(self):
        registry = AttachmentRegistry()
        seen: list[Attachment] = []
        registry.add_observer(seen.append)
        registry.register("a", endpoint_class="llm_stream")
        registry.set_phase("a", AttachmentPhase.GRACE)
        registry.set_phase("a", AttachmentPhase.GRACE)
        registry.set_phase("a", AttachmentPhase.ACTIVE)
        self.assertEqual(
            [a.phase for a in seen],
            [AttachmentPhase.GRACE, AttachmentPhase.ACTIVE],
        )

    def test_unregister_tells_observers_the_claims_are_gone(self):
        registry = AttachmentRegistry()
        seen: list[Attachment] = []
        registry.add_observer(seen.append)
        registry.register("a", endpoint_class="llm_stream")
        registry.set_phase("a", AttachmentPhase.GRACE)
        registry.unregister("a")
        self.assertEqual(seen[-1].phase, AttachmentPhase.DEAD)
        self.assertEqual(registry.snapshot(), ())

    def test_a_broken_observer_does_not_break_the_stream(self):
        registry = AttachmentRegistry()
        good: list[Attachment] = []

        def explode(_attachment):
            raise RuntimeError("ledger on fire")

        registry.add_observer(explode)
        registry.add_observer(good.append)
        registry.register("a", endpoint_class="llm_stream")
        registry.set_phase("a", AttachmentPhase.GRACE)
        self.assertEqual(len(good), 1)

    def test_a_watchdog_walks_active_to_grace_to_dead(self):
        clock = FakeClock()

        async def run():
            registry = AttachmentRegistry(clock=clock)
            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="w",
                policy=LivenessPolicy(
                    endpoint_class=EndpointClass.VIDEO_STREAM,
                    timeout_s=100.0,
                    poll_interval_s=0.01,
                    grace_fraction=0.25,
                ),
                release=released,
                clock=clock,
                claims=[ResourceClaim(ClaimKind.VRAM_LEASE, "card-0", 4 * MIB, "t")],
                registry=registry,
            )
            watchdog.start()
            await asyncio.sleep(0.05)
            self.assertEqual(registry.get("w").phase, AttachmentPhase.ACTIVE)

            clock.advance(30.0)  # past 25 s of grace, well short of 100 s
            await asyncio.sleep(0.05)
            self.assertEqual(registry.get("w").phase, AttachmentPhase.GRACE)
            self.assertEqual(registry.reclaimable_bytes(), 4 * MIB)
            self.assertEqual(released.calls, 0, "grace must not release")

            # The consumer comes back. Off the ladder again, before a
            # reclaimer acts on a claim that is live.
            watchdog.note_progress(10)
            await asyncio.sleep(0.05)
            self.assertEqual(registry.get("w").phase, AttachmentPhase.ACTIVE)
            self.assertEqual(registry.reclaimable_bytes(), 0)

            clock.advance(101.0)
            await asyncio.wait_for(released.event.wait(), timeout=2.0)
            await watchdog.stop()
            self.assertIsNone(registry.get("w"))

        asyncio.run(run())

    def test_a_declared_dead_attachment_leaves_the_registry_by_itself(self):
        """``stop()`` is never reached on this path -- the stream is suspended.

        The same rule that puts teardown in the release callback applies to
        the registry entry: without an unregister inside the death path the
        registry would grow one entry per dropped client and keep reporting
        claims that were already handed back.
        """

        async def run():
            registry = AttachmentRegistry()
            released = Released()
            watchdog = ConsumerWatchdog(
                job_id="orphan",
                policy=fast_policy(timeout_s=0.1),
                release=released,
                claims=[ResourceClaim(ClaimKind.KV, "rid-1")],
                registry=registry,
            )
            stream = guarded_stream(frame_source(), watchdog)

            async def stall():
                async for _ in stream:
                    await asyncio.sleep(3600)

            task = asyncio.create_task(stall())
            await asyncio.wait_for(released.event.wait(), timeout=3.0)
            self.assertIsNone(registry.get("orphan"))
            self.assertEqual(registry.snapshot(), ())
            task.cancel()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# The one wired consumer: the VRAM ledger
# ---------------------------------------------------------------------------


class LedgerGraceBridgeTest(CustomTestCase):
    CARD = "GPU-aaaa"

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.now = [1000.0]
        self.store = ReservationStore(
            self.root,
            clock=lambda: self.now[0],
            total_bytes_resolver=lambda uuid: 32768 * MIB,
        )
        self.store.acquire(
            card_uuid=self.CARD,
            tenant_id="k3-video-enhance",
            klass=3,
            reserved_bytes=6000 * MIB,
            state=TenantState.HOT,
        )

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def test_a_fresh_entry_is_not_in_grace(self):
        entry = self.store.read(self.CARD).tenant("k3-video-enhance")
        self.assertFalse(entry.in_grace)
        self.assertEqual(self.store.read(self.CARD).grace_bytes, 0)

    def test_grace_reaches_the_ledger_and_clears_again(self):
        registry = AttachmentRegistry()
        attach_ledger_grace_bridge(registry, self.store)
        registry.register(
            "enhance-1",
            endpoint_class=EndpointClass.VIDEO_STREAM.value,
            claims=[
                ResourceClaim(
                    ClaimKind.VRAM_LEASE, self.CARD, 6000 * MIB, "k3-video-enhance"
                )
            ],
        )

        registry.set_phase("enhance-1", AttachmentPhase.GRACE, silent_for_s=80.0)
        card = self.store.read(self.CARD)
        self.assertTrue(card.tenant("k3-video-enhance").in_grace)
        self.assertEqual(card.grace_bytes, 6000 * MIB)
        self.assertIn("in grace", card.render())

        registry.set_phase("enhance-1", AttachmentPhase.ACTIVE)
        card = self.store.read(self.CARD)
        self.assertFalse(card.tenant("k3-video-enhance").in_grace)
        self.assertEqual(card.grace_bytes, 0)

    def test_the_lease_is_not_shortened_by_grace(self):
        """A tenant in grace is still running and still holds its bytes."""
        before = self.store.read(self.CARD).tenant("k3-video-enhance")
        self.store.set_grace(self.CARD, "k3-video-enhance", True)
        after = self.store.read(self.CARD).tenant("k3-video-enhance")
        self.assertEqual(after.lease_expiry_ts, before.lease_expiry_ts)
        self.assertEqual(after.reserved_bytes, before.reserved_bytes)

    def test_a_claim_without_a_card_or_tenant_is_skipped(self):
        bridge = LedgerGraceBridge(self.store)
        bridge(
            Attachment(
                attachment_id="x",
                endpoint_class="video_stream",
                phase=AttachmentPhase.GRACE,
                claims=(ResourceClaim(ClaimKind.PIPELINE, "job-1"),),
            )
        )
        self.assertFalse(self.store.read(self.CARD).tenant("k3-video-enhance").in_grace)

    def test_an_unknown_tenant_is_logged_not_raised(self):
        """A stream may outlive the ledger entry it was booked against."""
        bridge = LedgerGraceBridge(self.store)
        bridge(
            Attachment(
                attachment_id="x",
                endpoint_class="video_stream",
                phase=AttachmentPhase.GRACE,
                claims=(
                    ResourceClaim(ClaimKind.VRAM_LEASE, self.CARD, 1, "who-is-this"),
                ),
            )
        )

    def test_a_pre_344_ledger_file_still_reads(self):
        """An older server's file has no in_grace key at all."""
        legacy = {
            "tenant_id": "old",
            "klass": 1,
            "state": TenantState.HOT.value,
            "reserved_bytes": 100,
        }
        entry = ReservationEntry.from_json(legacy)
        self.assertFalse(entry.in_grace)
        self.assertEqual(entry.grace_since_ts, 0.0)
        # And round-trips with the new fields present.
        again = ReservationEntry.from_json(json.loads(json.dumps(entry.to_json())))
        self.assertEqual(again, entry)


# ---------------------------------------------------------------------------
# The one-long-await shape: image and speech lanes
# ---------------------------------------------------------------------------


class FakeRequest:
    def __init__(self, disconnected_after: int | None = None) -> None:
        self.polls = 0
        self._after = disconnected_after

    async def is_disconnected(self) -> bool:
        self.polls += 1
        return self._after is not None and self.polls >= self._after


class AwaitWithLivenessTest(CustomTestCase):
    def test_a_connected_client_gets_its_answer(self):
        async def run():
            async def work():
                await asyncio.sleep(0.05)
                return "image"

            result = await await_with_liveness(
                work(),
                raw_request=FakeRequest(),
                endpoint_class=EndpointClass.IMAGE_GENERATION,
                job_id="img-1",
                config=LivenessConfig(poll_interval_s=0.01),
            )
            self.assertEqual(result, "image")

        asyncio.run(run())

    def test_a_client_that_hangs_up_cancels_the_lane_job(self):
        async def run():
            started = asyncio.Event()
            cancelled = asyncio.Event()
            abandoned = Released()

            async def work():
                started.set()
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            with self.assertRaises(ConsumerGone):
                await await_with_liveness(
                    work(),
                    raw_request=FakeRequest(disconnected_after=2),
                    endpoint_class=EndpointClass.IMAGE_GENERATION,
                    job_id="img-2",
                    config=LivenessConfig(poll_interval_s=0.01),
                    on_abandoned=abandoned,
                )
            await asyncio.sleep(0.05)
            self.assertTrue(started.is_set())
            self.assertTrue(cancelled.is_set(), "the lane job kept running")
            self.assertEqual(abandoned.calls, 1)

        asyncio.run(run())

    def test_a_request_object_without_disconnect_support_still_works(self):
        async def run():
            async def work():
                return 7

            self.assertEqual(
                await await_with_liveness(
                    work(),
                    raw_request=object(),
                    endpoint_class=EndpointClass.AUDIO_SPEECH,
                    job_id="spk-1",
                ),
                7,
            )

        asyncio.run(run())


class TimeToFirstTokenIsOutsideTheBudgetTest(CustomTestCase):
    """#514 / #505-C-01: the LLM_STREAM budget does NOT have to cover TTFT.

    Audit #505 ranked the unmeasured 90 s ``LLM_STREAM`` default as its most
    damaging finding on the reasoning that "the clock starts at watchdog
    construction and advances only on bytes accepted by the transport, so a
    long first-token latency counts in full". The first half is right and the
    conclusion is wrong, because of an ordering the audit did not check:
    ``_handle_streaming_request`` AWAITS the first chunk out of the generator
    (``generator.__anext__()``) before it constructs the ``StreamingResponse``,
    and only the returned response is handed to ``guard_generate_stream``. So
    the watchdog does not exist yet while prefill runs, and the budget it then
    starts covers the gap BETWEEN chunks -- milliseconds at decode -- not the
    time to the first one.

    That makes the finding far less severe than reported, and it makes this
    ordering load-bearing: move the wrap in front of the pre-pull, or drop the
    pre-pull, and the desk-picked 90 s silently becomes an abort threshold on
    prefill. These tests pin the ordering so that change cannot pass unseen.
    """

    def test_the_watchdog_clock_starts_when_it_is_created_not_when_the_request_arrived(
        self,
    ):
        """The budget is spent from wrap time, so anything the server did
        before the wrap -- queueing, prefill -- costs the consumer nothing."""

        async def run():
            clock = FakeClock()
            clock.advance(1000.0)  # a long prefill happens here, pre-wrap
            watchdog = ConsumerWatchdog(
                job_id="ttft",
                policy=fast_policy(),
                release=Released(),
                clock=clock,
            )
            # Zero silence at wrap time: the 1000 s that elapsed before the
            # watchdog existed are not charged against the class budget.
            self.assertEqual(watchdog.state.silent_for(clock()), 0.0)
            self.assertEqual(watchdog.state.started_at, clock())

        asyncio.run(run())

    def test_the_pre_pull_happens_before_the_streaming_response_exists(self):
        """Structural pin, both OpenAI streaming endpoints.

        Read from source rather than executed: the property is an ORDERING
        inside a method whose real inputs are a tokenizer manager and a live
        request, and the ordering is exactly what a refactor would move.
        """
        import ast
        import inspect

        from sglang.srt.entrypoints.openai import serving_chat, serving_completions

        for module in (serving_chat, serving_completions):
            tree = ast.parse(inspect.getsource(module))
            handlers = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_handle_streaming_request"
            ]
            self.assertTrue(
                handlers, f"{module.__name__} has no _handle_streaming_request"
            )
            for handler in handlers:
                pre_pull = [
                    n.lineno
                    for n in ast.walk(handler)
                    if isinstance(n, ast.Await)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == "__anext__"
                ]
                responses = [
                    n.lineno
                    for n in ast.walk(handler)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "StreamingResponse"
                ]
                self.assertTrue(
                    pre_pull,
                    f"{module.__name__}._handle_streaming_request no longer awaits "
                    "the first chunk before returning: TTFT now falls inside the "
                    "liveness budget and the unmeasured LLM_STREAM default "
                    "becomes a prefill abort threshold (#505-C-01)",
                )
                self.assertTrue(responses, f"{module.__name__} builds no response")
                self.assertLess(
                    min(pre_pull),
                    min(responses),
                    f"{module.__name__}: the first chunk must be pulled BEFORE the "
                    "StreamingResponse is constructed",
                )

    def test_the_guard_is_applied_to_an_already_awaited_response(self):
        """The wrap must sit after the await, in ``serving_base``. If
        ``guard_generate_stream`` ever wrapped the coroutine instead of its
        result, the watchdog would start before the first chunk."""
        import ast
        import inspect

        from sglang.srt.entrypoints.openai import serving_base

        tree = ast.parse(inspect.getsource(serving_base))
        awaits = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Await)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "_handle_streaming_request"
        ]
        guards = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "guard_generate_stream"
        ]
        self.assertTrue(awaits, "serving_base no longer awaits the streaming handler")
        self.assertTrue(guards, "serving_base no longer guards the token stream")
        self.assertLess(
            min(awaits),
            min(guards),
            "guard_generate_stream must wrap the AWAITED response, not the "
            "coroutine -- otherwise the liveness budget starts covering TTFT "
            "(#505-C-01)",
        )

    def test_the_rationale_records_that_ttft_is_outside_the_budget(self):
        """The number is still unmeasured; what is now pinned is its SCOPE.
        A reader deciding whether 90 s is safe must be told what it covers."""
        rationale = DEFAULT_TIMEOUT_RATIONALE[EndpointClass.LLM_STREAM]
        self.assertIn("first chunk", rationale.lower())


if __name__ == "__main__":
    unittest.main()
