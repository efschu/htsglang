# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#347-M1: the idle workbench's HTTP surface, against a real server.

DESIGN #347 W8 says the queue is readable and controllable over the serving
API so the frontend is one client of it rather than the only way to see it.
The only way to demonstrate that is to talk to a real socket: these tests run
the real FastAPI app on a real port with the real routes and the real error
handlers, and drive them with plain ``requests`` -- no SDK, because there is
no standard protocol for "what is the rig doing while nobody is watching" and
a client for this surface is a curl.

What is mocked: the machine (a synthetic card), the inference engine the
harness always mocks, and the tenants (one mock, one real tuner tenant whose
script is a stand-in). What is real: the socket, the routes, the payload
shaping, the service, the scheduler and the priority order.

Requires no GPU. ``CUDA_VISIBLE_DEVICES=99`` is the intended way to run it.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import requests
from openai_sdk_harness import TOKENIZER_NAME, live_server

from sglang.srt.training.feasibility import GIB, CardResources, MachineResources
from sglang.srt.training.tenant import DemandSample, IdleMonitor
from sglang.srt.workbench.scheduler import WorkbenchConfig
from sglang.srt.workbench.service import WorkbenchService
from sglang.srt.workbench.tenant import (
    IdleWorkTenant,
    SegmentOutcome,
    SegmentStatus,
    WorkEstimate,
    WorkSegment,
)
from sglang.srt.workbench.tenants.fp8_tuner import Fp8BlockTunerTenant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


SYNTHETIC_MACHINE = MachineResources(
    cards=(
        CardResources(
            uuid="GPU-synthetic-0",
            index=0,
            name="Synthetic 80GB",
            total_bytes=80 * GIB,
            available_bytes=80 * GIB,
        ),
    ),
    ram_total_bytes=256 * GIB,
    ram_available_bytes=200 * GIB,
    disk_free_bytes=2000 * GIB,
    disk_path="/synthetic",
)

DEVICES = [
    {
        "uuid": "GPU-synthetic-0",
        "index": 0,
        "name": "Synthetic 80GB",
        "total_bytes": 80 * GIB,
    }
]


class SleeperSegment(WorkSegment):
    def __init__(self, tenant):
        self.tenant = tenant

    async def wait(self) -> SegmentOutcome:
        import asyncio

        while True:
            await asyncio.sleep(0.05)

    async def preempt(self, *, timeout_s: float = 60.0) -> SegmentOutcome:
        self.tenant.preempted += 1
        return SegmentOutcome(status=SegmentStatus.PREEMPTED, detail="stopped")

    async def cancel(self, *, timeout_s: float = 30.0) -> SegmentOutcome:
        self.tenant.cancelled += 1
        return SegmentOutcome(status=SegmentStatus.CANCELLED, detail="cancelled")


class SleeperTenant(IdleWorkTenant):
    name = "sleeper"
    priority = 20

    def __init__(self):
        super().__init__()
        self.items = 1
        self.started = 0
        self.preempted = 0
        self.cancelled = 0

    def pending(self) -> int:
        return self.items

    def estimate(self) -> WorkEstimate:
        return WorkEstimate(per_card_bytes=GIB, posts={"mock": GIB})

    async def start_segment(self, grant, sink) -> WorkSegment:
        self.started += 1
        sink_message = f"sleeper started on {list(grant.card_indices)}"
        from sglang.srt.workbench.tenant import WorkEvent

        sink(WorkEvent("info", sink_message))
        return SleeperSegment(self)

    def enqueue(self, item):
        self.items += 1
        return f"sleeper-{self.items}"


def build_workbench(root: Path, *, enabled: bool = True) -> WorkbenchService:
    script = root / "tuning_block_wise_kernel.py"
    script.write_text("# stand-in\n")
    tuner = Fp8BlockTunerTenant(
        artifact_root=root / "fp8_tuner",
        script_path=script,
        device_resolver=lambda: list(DEVICES),
    )
    return WorkbenchService(
        WorkbenchConfig(
            enabled=enabled,
            artifact_root=root / "workbench",
            poll_seconds=0.05,
            reject_backoff_s=0.05,
            preempt_timeout_s=2.0,
            cancel_timeout_s=2.0,
        ),
        tenants=[SleeperTenant(), tuner],
        monitor=IdleMonitor(
            [lambda: DemandSample(source="fake_serving", busy=False)],
            grace_seconds=0.0,
        ),
        machine_resolver=lambda: SYNTHETIC_MACHINE,
    )


def wait_until(predicate, timeout_s: float, what: str):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"{what} did not happen within {timeout_s}s")


class WorkbenchHttpSurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.workbench = build_workbench(cls.root)
        cls._server = live_server(
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME),
            workbench_service=cls.workbench,
        )
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls._tmp.cleanup()

    def get(self, path: str, **params):
        return requests.get(f"{self.base_url}{path}", params=params, timeout=20)

    def post(self, path: str, body: dict):
        return requests.post(f"{self.base_url}{path}", json=body, timeout=20)

    def test_the_snapshot_reports_the_priority_ordered_tenant_table(self):
        body = self.get("/x-htsglang/workbench").json()
        self.assertTrue(body["ok"])
        bench = body["workbench"]
        self.assertTrue(bench["enabled"])
        names = [t["name"] for t in bench["tenants"]]
        self.assertEqual(names, ["sleeper", "fp8_tuner"])
        self.assertEqual(bench["tenants"][0]["priority"], 20)
        self.assertIn("config", bench)
        self.assertIn("idle", bench)

    def test_the_scheduler_actually_runs_and_the_log_says_so(self):
        wait_until(
            lambda: self.get("/x-htsglang/workbench").json()["workbench"]["running"]
            == "sleeper",
            15.0,
            "the sleeper tenant starting",
        )
        events = self.get("/x-htsglang/workbench/events", limit=200).json()
        self.assertTrue(events["ok"])
        messages = [e["message"] for e in events["data"]]
        self.assertTrue(any("sleeper started" in m for m in messages), messages[-5:])
        self.assertTrue(any(e["tenant"] == "sleeper" for e in events["data"]))

    def test_events_paginate_by_sequence_number(self):
        first = self.get("/x-htsglang/workbench/events", limit=1).json()
        self.assertEqual(len(first["data"]), 1)
        second = self.get(
            "/x-htsglang/workbench/events", after=first["last_seq"], limit=100
        ).json()
        self.assertTrue(
            all(e["seq"] > first["last_seq"] for e in second["data"]), second["data"]
        )

    def test_pausing_the_bench_preempts_the_running_segment(self):
        wait_until(
            lambda: self.get("/x-htsglang/workbench").json()["workbench"]["running"]
            == "sleeper",
            15.0,
            "the sleeper tenant starting",
        )
        sleeper = self.workbench.workbench.tenant("sleeper")
        before = sleeper.preempted
        response = self.post("/x-htsglang/workbench/pause", {"paused": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["paused"])
        wait_until(
            lambda: sleeper.preempted > before, 15.0, "the segment being preempted"
        )
        self.assertTrue(
            self.post("/x-htsglang/workbench/pause", {"paused": False}).json()["ok"]
        )

    def test_pausing_one_tenant_by_name(self):
        response = self.post(
            "/x-htsglang/workbench/pause", {"paused": True, "tenant": "fp8_tuner"}
        )
        self.assertEqual(response.json()["tenant"], "fp8_tuner")
        self.assertTrue(response.json()["paused"])
        self.post(
            "/x-htsglang/workbench/pause", {"paused": False, "tenant": "fp8_tuner"}
        )

    def test_an_unknown_tenant_is_a_404_with_the_known_names(self):
        response = self.post(
            "/x-htsglang/workbench/pause", {"paused": True, "tenant": "nope"}
        )
        self.assertEqual(response.status_code, 404)
        body = response.json()["error"]
        self.assertEqual(body["code"], "unknown_tenant")
        self.assertIn("fp8_tuner", body["message"])

    def test_enqueue_adds_a_tuner_shape(self):
        before = self.get("/x-htsglang/workbench").json()["workbench"]
        pending = {t["name"]: t["pending"] for t in before["tenants"]}["fp8_tuner"]
        response = self.post(
            "/x-htsglang/workbench/enqueue",
            {"tenant": "fp8_tuner", "item": {"n": 7168, "k": 5120, "batch_size": 4}},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("N=7168,K=5120,M=4", body["item"])
        self.assertEqual(body["pending"], pending + 1)

    def test_a_bad_enqueue_is_a_400_that_says_what_was_wrong(self):
        response = self.post(
            "/x-htsglang/workbench/enqueue",
            {"tenant": "fp8_tuner", "item": {"n": "wide", "k": 5120}},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "workbench_error")

    def test_a_missing_tenant_field_is_a_400(self):
        response = self.post("/x-htsglang/workbench/enqueue", {"item": {}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("tenant", response.json()["error"]["message"])


class DisabledWorkbenchTest(unittest.TestCase):
    """A switched-off bench answers by name, never with a 404."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls._tmp = tempfile.TemporaryDirectory()
        cls._server = live_server(
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME),
            workbench_service=build_workbench(Path(cls._tmp.name), enabled=False),
        )
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls._tmp.cleanup()

    def test_the_snapshot_still_answers_with_the_tenant_table(self):
        body = requests.get(f"{self.base_url}/x-htsglang/workbench", timeout=20).json()
        self.assertFalse(body["workbench"]["enabled"])
        self.assertEqual(len(body["workbench"]["tenants"]), 2)

    def test_a_control_call_is_a_503_naming_the_flag(self):
        response = requests.post(
            f"{self.base_url}/x-htsglang/workbench/pause",
            json={"paused": True},
            timeout=20,
        )
        self.assertEqual(response.status_code, 503)
        body = response.json()["error"]
        self.assertEqual(body["code"], "workbench_disabled")
        self.assertIn("--enable-idle-workbench", body["message"])


if __name__ == "__main__":
    unittest.main()
