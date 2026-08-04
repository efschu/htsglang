# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#546 idle park, seen through the HTTP surface. Hermetic, no GPU.

`test_idle_park.py` pins the controller. This file pins the WIRING, which is
the half that is invisible to a controller test and has its own way of being
silently wrong: a park that nothing ever calls, a health block that reports a
different object than the one serving requests, gauges registered against a
controller that is not there, or a wake hook on a path no request takes.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \\
      python -m pytest test/registered/translator/test_idle_park_app.py -v
"""

import unittest

import torch
from fastapi.testclient import TestClient
from torch import nn

from sglang.srt.translator import metrics
from sglang.srt.translator.backends import FakeAsr, FakeEmbedder, FakeMt, FakeTts
from sglang.srt.translator.config import TranslatorConfig
from sglang.srt.translator.idle_park import IdleParkConfig, IdleParkController
from sglang.srt.translator.ledger import AudioAssetLedger
from sglang.srt.translator.server import Stack, TranslatorService, build_app

LANG_A, LANG_B = "de", "es"


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> "FakeClock":
        self.now += float(seconds)
        return self


def build(idle_park: bool = True):
    clock = FakeClock()
    config = TranslatorConfig(
        default_participants=(LANG_A, LANG_B),
        max_sessions=3,
        idle_park=IdleParkConfig(enabled=idle_park, floor_s=60.0),
    )
    stack = Stack(
        asr=FakeAsr(languages=(LANG_A, LANG_B), pitch_map=[(150.0, LANG_A)]),
        embedder=FakeEmbedder(min_seconds=0.5),
        mt=FakeMt(),
        tts=FakeTts(languages=(LANG_A, LANG_B), min_reference_seconds=1.0),
    )
    ledger = AudioAssetLedger(clock=clock)
    torch.manual_seed(546)
    ledger.register("asr", nn.Linear(32, 32))
    ledger.register("codec", nn.Linear(32, 32))
    controller = IdleParkController(
        ledger, config.idle_park, clock=clock,
        busy_probe=lambda: bool(getattr(stack.tts, "busy", False)),
    )
    service = TranslatorService(config, stack, idle_park=controller)
    return service, controller, clock


class TestHealthSurface(unittest.TestCase):
    def test_health_reports_the_controller_that_is_actually_serving(self):
        service, controller, clock = build()
        client = TestClient(build_app(service))

        block = client.get("/api/translator/health").json()["idle_park"]
        self.assertEqual(block["state"], "resident")
        self.assertEqual(block["threshold_s"], 60.0)
        self.assertTrue(block["enabled"])

        clock.advance(600)
        controller.tick()
        block = client.get("/api/translator/health").json()["idle_park"]
        self.assertEqual(block["state"], "parked")
        self.assertGreater(block["parked_mib"], 0.0)

    def test_a_deployment_without_a_controller_says_so(self):
        """The all-fake boot has no ledgered assets and must not pretend."""
        config = TranslatorConfig(default_participants=(LANG_A, LANG_B))
        stack = Stack(
            asr=FakeAsr(languages=(LANG_A, LANG_B), pitch_map=[(150.0, LANG_A)]),
            embedder=FakeEmbedder(),
            mt=FakeMt(),
            tts=FakeTts(languages=(LANG_A, LANG_B)),
        )
        service = TranslatorService(config, stack)
        client = TestClient(build_app(service))
        self.assertIsNone(client.get("/api/translator/health").json()["idle_park"])


class TestMetricsSurface(unittest.TestCase):
    def setUp(self):
        metrics.reset_for_test()
        metrics.enable()
        self.addCleanup(metrics.reset_for_test)

    def test_the_park_gauges_are_read_at_scrape_time(self):
        service, controller, clock = build()
        client = TestClient(build_app(service))

        body = client.get("/metrics").text
        self.assertIn("translator_assets_parked 0.000000", body)

        clock.advance(600)
        controller.tick()
        body = client.get("/metrics").text
        self.assertIn("translator_assets_parked 1.000000", body)
        self.assertIn("translator_parked_mib", body)
        self.assertIn("translator_park_threshold_seconds 60.000000", body)
        # A gauge captured at registration would still say 0 here, which is
        # the one moment the number matters.
        parked_mib = float(
            next(
                line for line in body.splitlines()
                if line.startswith("translator_parked_mib ")
            ).split()[1]
        )
        self.assertGreater(parked_mib, 0.0)

    def test_both_wake_latencies_are_exposed(self):
        service, controller, clock = build()
        client = TestClient(build_app(service))
        clock.advance(600)
        controller.tick()
        controller.ensure_awake()
        body = client.get("/metrics").text
        self.assertIn("translator_last_wake_ms", body)
        self.assertIn("translator_last_first_serve_ms", body)


class TestRequestPathsWake(unittest.TestCase):
    def test_opening_a_conversation_starts_the_wake(self):
        service, controller, clock = build()
        client = TestClient(build_app(service))
        clock.advance(600)
        controller.tick()
        self.assertTrue(controller.parked)

        response = client.post(
            "/api/translator/sessions", json={"participants": [LANG_A, LANG_B]}
        )
        self.assertEqual(response.status_code, 200)
        # Started, not awaited -- but by the time the handler has returned and
        # the tiny fake assets have moved, it is resident.
        self.assertGreaterEqual(controller.ensure_awake(), 0.0)
        self.assertFalse(controller.parked)

    def test_a_parked_tenant_still_answers_the_read_surfaces(self):
        """Parked is not down: languages and health must still work."""
        service, controller, clock = build()
        client = TestClient(build_app(service))
        clock.advance(600)
        controller.tick()
        self.assertEqual(client.get("/api/translator/languages").status_code, 200)
        self.assertEqual(client.get("/api/translator/health").status_code, 200)
        self.assertTrue(controller.parked, "a read surface woke the tenant")

    def test_a_REAL_TURN_over_the_socket_wakes_a_parked_tenant(self):
        """The end-to-end proof, and the one no unit test can stand in for.

        Everything else here calls the controller directly. This drives the
        actual path a phone drives -- websocket, audio frames, release, drain
        -- against a PARKED tenant, and requires a complete turn to come out.
        A wake hook wired to a method the request path never reaches, or an
        awaitable of the wrong kind, passes every controller test and fails
        exactly here. (One did: ``create_task`` over an executor Future raises
        ``TypeError`` on the first frame after a park, and only this test saw
        it.)
        """
        import json
        import math

        import numpy as np

        from sglang.srt.translator.audio import PIPELINE_SAMPLE_RATE, Pcm16Codec
        from sglang.srt.translator.backends import AudioChunk

        service, controller, clock = build()
        client = TestClient(build_app(service))
        clock.advance(600)
        controller.tick()
        self.assertTrue(controller.parked)
        parks_before = controller.parks

        rate = PIPELINE_SAMPLE_RATE
        t = np.arange(int(2.0 * rate), dtype=np.float32) / rate
        speech = (0.3 * np.sin(2.0 * math.pi * 150.0 * t)).astype(np.float32)
        codec = Pcm16Codec(sample_rate=rate, frame_ms=20)

        with client.websocket_connect("/api/translator/stream") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "kind": "hello",
                        "session_id": "park-1",
                        "participants": [LANG_A, LANG_B],
                        "codecs": ["pcm16"],
                    }
                )
            )
            ready = json.loads(ws.receive_text())
            self.assertEqual(ready["kind"], "ready")
            for frame in codec.encode(AudioChunk(speech, rate)):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))

            done = None
            seen = []
            for _ in range(400):
                message = ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    continue
                event = json.loads(text)
                seen.append(event.get("kind"))
                if event.get("kind") == "turn.done":
                    done = event
                    break

        self.assertIsNotNone(done, f"no turn completed after a park; saw {seen}")
        self.assertIn("turn.transcript", seen)
        self.assertIn("turn.translation", seen)
        self.assertFalse(controller.parked, "the turn ran against parked assets")
        self.assertEqual(controller.wakes, 1)
        self.assertEqual(controller.parks, parks_before)


if __name__ == "__main__":
    unittest.main()
