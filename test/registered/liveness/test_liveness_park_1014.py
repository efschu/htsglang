"""LIVEPARK (14.10.): a stream D parked on purpose is not a dead consumer.

NF boot ``boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10140732_f4afb10f28_1014_073255``:
D.log Z.403857 ``attachment stream-pdflip-134-476 (llm_stream): grace -> dead after 90.8s``,
front.log Z.27137 ``leg2 ... ClientPayloadError`` + 503. D had parked the request at
08:59:50 (PARK-RUNNING immediate-over-x), resumed it at 09:00:45, held it for P
(W50-REROUTE midstream 09:00:47) and parked it again at 09:01:02 -- no byte for 90 s,
all of it D's own doing. The watchdog stays on; the exception is the hold:

* held (scheduler's notice names the rid) -> no reaping, however long;
* the clock restarts at the resumption -> a resumed stream silent for > 90 s is reaped;
* a running stream that is never held -> reaped after 90 s exactly as before.
"""

import asyncio
import unittest

from flliper.srt.liveness import (
    AttachmentPhase,
    AttachmentRegistry,
    ClaimKind,
    ConsumerWatchdog,
    EndpointClass,
    HoldRegistry,
    LivenessConfig,
    LivenessPolicy,
    ResourceClaim,
    global_hold_registry,
    guard_generate_stream,
)
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Released:
    def __init__(self) -> None:
        self.calls = 0
        self.event = asyncio.Event()

    async def __call__(self) -> None:
        self.calls += 1
        self.event.set()


def llm_policy() -> LivenessPolicy:
    # the class default (90 s) -- the number the metal hit
    return LivenessPolicy(endpoint_class=EndpointClass.LLM_STREAM, poll_interval_s=0.01)


async def settle() -> None:
    await asyncio.sleep(0.05)


class HeldWatchdogTest(CustomTestCase):
    def test_default_is_still_90s(self):
        self.assertEqual(llm_policy().resolved_timeout(), 90.0)

    def test_parked_stream_beyond_90s_is_not_reaped(self):
        clock = FakeClock()
        holds = HoldRegistry()

        async def run():
            released = Released()
            wd = ConsumerWatchdog(job_id="stream-pdflip-134-476", policy=llm_policy(), release=released,
                                  clock=clock, held=lambda: holds.any_held(["pdflip-134-476"]))
            wd.start()
            holds.replace(["pdflip-134-476"])  # PARK-RUNNING
            await settle()
            for _ in range(6):  # 6 x 60 s = 360 s parked
                clock.advance(60.0)
                await settle()
            self.assertEqual(released.calls, 0)
            self.assertFalse(wd.state.declared_dead)
            self.assertTrue(wd.state.held)
            await wd.stop()

        asyncio.run(run())

    def test_clock_restarts_at_resumption_then_reaps_a_resumed_stall(self):
        clock = FakeClock()
        holds = HoldRegistry()

        async def run():
            released = Released()
            wd = ConsumerWatchdog(job_id="j", policy=llm_policy(), release=released, clock=clock,
                                  held=lambda: holds.any_held(["r1"]))
            wd.start()
            clock.advance(60.0)  # 60 s running silence before the park ...
            await settle()
            holds.replace(["r1"])
            await settle()
            clock.advance(300.0)
            await settle()
            holds.replace([])  # PARK-RESUME
            await settle()
            self.assertFalse(wd.state.held)
            clock.advance(80.0)  # ... does not add up with the 80 s after it
            await settle()
            self.assertEqual(released.calls, 0)
            clock.advance(11.0)  # 91 s after the resumption: a real stall
            await asyncio.wait_for(released.event.wait(), timeout=2.0)
            self.assertTrue(wd.state.declared_dead)

        asyncio.run(run())

    def test_running_stream_without_tokens_is_reaped_as_before(self):
        clock = FakeClock()
        holds = HoldRegistry()
        holds.replace(["someone-else"])  # another rid parked: not this stream's hold

        async def run():
            released = Released()
            wd = ConsumerWatchdog(job_id="j", policy=llm_policy(), release=released, clock=clock,
                                  held=lambda: holds.any_held(["r-running"]))
            wd.start()
            await settle()
            clock.advance(89.0)
            await settle()
            self.assertEqual(released.calls, 0)
            clock.advance(2.0)
            await asyncio.wait_for(released.event.wait(), timeout=2.0)
            self.assertTrue(wd.state.declared_dead)

        asyncio.run(run())

    def test_a_failing_hold_predicate_falls_back_to_reaping(self):
        clock = FakeClock()

        def broken() -> bool:
            raise RuntimeError("boom")

        async def run():
            released = Released()
            wd = ConsumerWatchdog(job_id="j", policy=llm_policy(), release=released, clock=clock,
                                  held=broken)
            wd.start()
            await settle()
            clock.advance(91.0)
            await asyncio.wait_for(released.event.wait(), timeout=2.0)

        asyncio.run(run())

    def test_a_hold_takes_the_attachment_out_of_grace(self):
        clock = FakeClock()
        holds = HoldRegistry()
        registry = AttachmentRegistry(clock=clock)

        async def run():
            released = Released()
            wd = ConsumerWatchdog(job_id="stream-g", policy=llm_policy(), release=released, clock=clock,
                                  claims=[ResourceClaim(kind=ClaimKind.KV, key="g")], registry=registry,
                                  held=lambda: holds.any_held(["g"]))
            wd.start()
            await settle()
            clock.advance(30.0)  # > 22.5 s: grace
            await settle()
            self.assertTrue(wd.in_grace)
            self.assertEqual(registry.get("stream-g").phase, AttachmentPhase.GRACE)
            holds.replace(["g"])
            await settle()
            self.assertFalse(wd.in_grace)
            self.assertEqual(registry.get("stream-g").phase, AttachmentPhase.ACTIVE)
            await wd.stop()

        asyncio.run(run())


class FakeTokenizerManager:
    def __init__(self) -> None:
        self.aborted = []

    def abort_request(self, rid: str = "", abort_all: bool = False) -> None:
        self.aborted.append(rid)


class FakeResponse:
    def __init__(self, body_iterator) -> None:
        self.body_iterator = body_iterator


class FakeObj:
    def __init__(self, rid) -> None:
        self.rid = rid


async def parked_producer(gate: asyncio.Event):
    """One token, then D parks: the producer is silent until ``gate``."""
    yield b"data: tok0\n\n"
    await gate.wait()
    yield b"data: tok1\n\n"


class GuardGenerateStreamHoldTest(CustomTestCase):
    """The guard wires the process registry; the consumer reads all the time (the front)."""

    TIMEOUT = 0.3

    def _run(self, rid: str, hold: bool):
        async def run():
            reg = global_hold_registry()
            reg.replace([rid] if hold else [])
            manager = FakeTokenizerManager()
            gate = asyncio.Event()
            response = guard_generate_stream(
                FakeResponse(parked_producer(gate)), tokenizer_manager=manager, obj=FakeObj(rid),
                config=LivenessConfig(timeouts_s={EndpointClass.LLM_STREAM.value: self.TIMEOUT},
                                      poll_interval_s=0.02))
            got = []

            async def consume():
                async for chunk in response.body_iterator:
                    got.append(chunk)

            task = asyncio.create_task(consume())
            await asyncio.sleep(self.TIMEOUT * 4)  # parked 4x the limit
            aborted = list(manager.aborted)
            reg.replace([])  # resume
            gate.set()
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001
                pass
            return aborted, got

        return asyncio.run(run())

    def test_held_rid_is_not_aborted(self):
        aborted, got = self._run("pdflip-park-1", hold=True)
        self.assertEqual(aborted, [])
        self.assertEqual(got, [b"data: tok0\n\n", b"data: tok1\n\n"])

    def test_unheld_rid_is_aborted_as_before(self):
        aborted, _ = self._run("pdflip-run-1", hold=False)
        self.assertEqual(aborted, ["pdflip-run-1"])


class _Sender:
    def __init__(self) -> None:
        self.sent = []

    def send_output(self, output, recv_obj=None) -> None:
        self.sent.append(output)


class _Req:
    def __init__(self, rid) -> None:
        self.rid = rid


class _Sched:
    def __init__(self) -> None:
        self.ipc_channels = type("IPC", (), {})()
        self.ipc_channels.send_to_tokenizer = _Sender()

        class _SA:
            tokenizer_worker_num = 1

        self.server_args = _SA()

    @property
    def sent(self):
        return self.ipc_channels.send_to_tokenizer.sent


class SchedulerPublishTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from flliper.srt.managers.scheduler import _pdflip_liveness_hold_publish

        cls.publish = staticmethod(_pdflip_liveness_hold_publish)

    def test_nothing_parked_never_sends(self):
        s = _Sched()
        for _ in range(3):
            self.assertFalse(self.publish(s))
        s.pdflip_d_parked = []
        self.assertFalse(self.publish(s))
        self.assertEqual(s.sent, [])

    def test_union_of_the_three_hold_lists_on_change_only(self):
        s = _Sched()
        s.pdflip_d_parked = [_Req("a")]  # flip / lane / RESUME-VIA-P park
        s.pdflip_dormant_hold = [_Req("b")]  # #1443 dormant hold
        s.pdflip_post_wake_settle = [_Req("c")]  # #1471 wake read
        self.assertTrue(self.publish(s))
        self.assertFalse(self.publish(s))  # unchanged: silent
        self.assertEqual(len(s.sent), 1)
        self.assertEqual(s.sent[0].held, ["a", "b", "c"])
        s.pdflip_d_parked, s.pdflip_dormant_hold, s.pdflip_post_wake_settle = [], [], []
        self.assertTrue(self.publish(s))  # resumption: the empty set goes out once
        self.assertEqual(s.sent[-1].held, [])
        self.assertFalse(self.publish(s))
        self.assertEqual(len(s.sent), 2)

    def test_multi_http_worker_publishes_nothing(self):
        s = _Sched()
        s.server_args.tokenizer_worker_num = 2
        s.pdflip_d_parked = [_Req("a")]
        self.assertFalse(self.publish(s))
        self.assertEqual(s.sent, [])


class TokenizerSideTest(CustomTestCase):
    def test_notice_roundtrips_and_installs(self):
        from flliper.srt.managers.io_struct import (
            PdFlipLivenessHoldOutput,
            msgpack_decode,
            msgpack_encode,
        )
        from flliper.srt.managers.tokenizer_manager import TokenizerManager

        n = msgpack_decode(msgpack_encode(PdFlipLivenessHoldOutput(held=["x", "y"])))
        self.assertIsInstance(n, PdFlipLivenessHoldOutput)
        TokenizerManager._handle_pdflip_liveness_hold(None, n)
        self.assertTrue(global_hold_registry().is_held("x"))
        self.assertTrue(global_hold_registry().any_held(["q", "y"]))
        TokenizerManager._handle_pdflip_liveness_hold(None, PdFlipLivenessHoldOutput(held=[]))
        self.assertFalse(global_hold_registry().is_held("x"))


if __name__ == "__main__":
    unittest.main()
