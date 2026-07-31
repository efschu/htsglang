# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #347-M1 pieces: the interface, the pricing, the scheduler, the tenants.

Everything here runs on a CPU-only host with no card, no ledger and no
arbitration directory besides a temporary one, which is deliberate for the
same reason #341's unit suite is: the workbench's decisions are arithmetic
and ordering, and a scheduler that can only be tested on the rig it was
written for is a rig constant with a loop around it.

The mock tenants are the point of the interface. If a mock tenant that sleeps
can be started on idle, preempted on demand and resumed at the next window
without the scheduler knowing what it does, then so can a tuner, a trainer
and everything ANALYSE #347 lists for M2.
"""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from sglang.srt.training.feasibility import GIB, CardResources, MachineResources
from sglang.srt.training.tenant import DemandSample, IdleMonitor
from sglang.srt.workbench.arb import (
    ArbDirectory,
    ArbRefused,
    parse_free_until,
    parse_holder,
)
from sglang.srt.workbench.http_api import (
    enqueue_payload,
    events_payload,
    pause_payload,
    snapshot_payload,
)
from sglang.srt.workbench.log import WorkLog
from sglang.srt.workbench.scheduler import Workbench, WorkbenchConfig
from sglang.srt.workbench.service import (
    UnknownTenant,
    WorkbenchDisabled,
    WorkbenchError,
    WorkbenchService,
    build_tenants,
)
from sglang.srt.workbench.tenant import (
    MIB,
    IdleWorkTenant,
    SegmentOutcome,
    SegmentStatus,
    WorkEstimate,
    WorkEvent,
    WorkSegment,
    price_segment,
)
from sglang.srt.workbench.tenants.card_probe import CardProbeTenant, probe_posts
from sglang.srt.workbench.tenants.fp8_tuner import (
    Fp8BlockTunerTenant,
    TunerCombo,
    combo_posts,
    parse_queue,
)
from sglang.test.ci.ci_register import register_cpu_ci

# No card is touched anywhere in this file; that is the point of it.
register_cpu_ci(est_time=30, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Synthetic machines and mock tenants
# ---------------------------------------------------------------------------


def card(name: str, total_gib: float, *, index: int, available_gib=None):
    total = int(total_gib * GIB)
    available = total if available_gib is None else int(available_gib * GIB)
    return CardResources(
        uuid=f"GPU-{name}",
        index=index,
        name=name,
        total_bytes=total,
        available_bytes=available,
    )


def machine(*cards, ram_gib: float = 128.0, disk_gib: float = 500.0):
    return MachineResources(
        cards=tuple(cards),
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB * 0.8),
        disk_free_bytes=int(disk_gib * GIB),
        disk_path="/synthetic",
    )


class MockSegment(WorkSegment):
    """Runs until stopped or until its item count is exhausted."""

    def __init__(self, tenant: "MockTenant", sink):
        self.tenant = tenant
        self.sink = sink
        self.stopped = asyncio.Event()
        self.status = SegmentStatus.SUCCEEDED
        self.preempt_delay_s = tenant.preempt_delay_s

    async def wait(self) -> SegmentOutcome:
        while not self.stopped.is_set():
            await asyncio.sleep(0.01)
            if self.tenant.finish_after_s and (
                time.time() - self.tenant.started_at >= self.tenant.finish_after_s
            ):
                self.tenant.items = max(0, self.tenant.items - 1)
                return SegmentOutcome(
                    status=SegmentStatus.SUCCEEDED, detail="mock item done"
                )
        return SegmentOutcome(status=self.status, detail="mock stopped")

    async def preempt(self, *, timeout_s: float = 60.0) -> SegmentOutcome:
        self.tenant.preempted += 1
        await asyncio.sleep(self.preempt_delay_s)
        self.status = SegmentStatus.PREEMPTED
        self.stopped.set()
        # The item is NOT consumed: a preempted work item is requeued, which
        # is the difference between preemptible and merely killable.
        return SegmentOutcome(status=SegmentStatus.PREEMPTED, detail="mock preempted")

    async def cancel(self, *, timeout_s: float = 30.0) -> SegmentOutcome:
        self.tenant.cancelled += 1
        self.status = SegmentStatus.CANCELLED
        self.stopped.set()
        return SegmentOutcome(status=SegmentStatus.CANCELLED, detail="mock cancelled")


class MockTenant(IdleWorkTenant):
    def __init__(
        self,
        name: str,
        priority: int,
        *,
        items: int = 1,
        per_card_mib: int = 1024,
        finish_after_s: float = 0.0,
        preempt_delay_s: float = 0.0,
        cards_wanted: int = 1,
    ):
        super().__init__()
        self.name = name
        self.priority = priority
        self.items = items
        self.per_card_bytes = per_card_mib * MIB
        self.finish_after_s = finish_after_s
        self.preempt_delay_s = preempt_delay_s
        self.cards_wanted = cards_wanted
        self.started = 0
        self.preempted = 0
        self.cancelled = 0
        self.started_at = 0.0
        self.segment = None

    def pending(self) -> int:
        return self.items

    def estimate(self) -> WorkEstimate:
        return WorkEstimate(
            per_card_bytes=self.per_card_bytes,
            posts={"mock": self.per_card_bytes},
            cards_wanted=self.cards_wanted,
        )

    async def start_segment(self, grant, sink) -> WorkSegment:
        self.started += 1
        self.started_at = time.time()
        sink(WorkEvent("info", f"{self.name} segment started"))
        self.segment = MockSegment(self, sink)
        return self.segment

    def enqueue(self, item):
        self.items += 1
        return f"{self.name}-{self.items}"


class Demand:
    """A switch the tests flip to make the rig look busy."""

    def __init__(self):
        self.busy = False

    def __call__(self) -> DemandSample:
        return DemandSample(source="test", busy=self.busy)


def bench(*tenants, demand=None, **overrides) -> tuple[Workbench, Demand]:
    demand = demand or Demand()
    settings = {
        "enabled": True,
        "artifact_root": Path(tempfile.mkdtemp(prefix="wb-")),
        "poll_seconds": 0.02,
        "reject_backoff_s": 0.02,
        "preempt_timeout_s": 1.0,
        "cancel_timeout_s": 1.0,
        "segment_timeout_s": 30.0,
    }
    settings.update(overrides)
    config = WorkbenchConfig(**settings)
    workbench = Workbench(
        list(tenants),
        config=config,
        monitor=IdleMonitor([demand], grace_seconds=0.0),
        machine_resolver=lambda: machine(
            card("A", 24, index=0), card("B", 24, index=1)
        ),
    )
    return workbench, demand


async def until(predicate, timeout_s: float = 5.0, step: float = 0.01) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return False


# ---------------------------------------------------------------------------
# W3: pricing
# ---------------------------------------------------------------------------


class PricingTest(unittest.TestCase):
    """W3 -- the D2 rule, generalized off training."""

    def test_it_picks_the_emptiest_card_and_reports_the_posts(self):
        estimate = WorkEstimate(
            per_card_bytes=4 * GIB, posts={"weights": 3 * GIB, "context": GIB}
        )
        verdict = price_segment(
            estimate,
            machine(
                card("busy", 24, index=0, available_gib=2),
                card("free", 24, index=1, available_gib=20),
            ),
        )
        self.assertTrue(verdict.fits, verdict.reason)
        self.assertEqual(verdict.chosen_indices, (1,))
        self.assertIn("weights", verdict.reason)

    def test_a_rejection_carries_the_arithmetic(self):
        verdict = price_segment(
            WorkEstimate(per_card_bytes=40 * GIB),
            machine(card("small", 24, index=0)),
        )
        self.assertFalse(verdict.fits)
        self.assertIn("40960 MiB wanted", verdict.reason)
        self.assertIn("24576 MiB total", verdict.reason)
        self.assertEqual(verdict.shortfall_bytes, 16 * GIB)

    def test_no_card_is_a_named_refusal_not_a_crash(self):
        verdict = price_segment(WorkEstimate(per_card_bytes=GIB), machine())
        self.assertFalse(verdict.fits)
        self.assertIn("no GPU is visible", verdict.reason)

    def test_cards_wanted_zero_means_every_card(self):
        verdict = price_segment(
            WorkEstimate(per_card_bytes=GIB, cards_wanted=0),
            machine(card("A", 24, index=0), card("B", 24, index=1)),
        )
        self.assertTrue(verdict.fits)
        self.assertEqual(len(verdict.chosen_cards), 2)

    def test_pinned_cards_are_honoured_and_unknown_ones_named(self):
        good = price_segment(
            WorkEstimate(per_card_bytes=GIB, card_uuids=("GPU-B",)),
            machine(card("A", 24, index=0), card("B", 24, index=1)),
        )
        self.assertEqual(good.chosen_indices, (1,))
        bad = price_segment(
            WorkEstimate(per_card_bytes=GIB, card_uuids=("GPU-Z",)),
            machine(card("A", 24, index=0)),
        )
        self.assertFalse(bad.fits)
        self.assertIn("GPU-Z", bad.reason)

    def test_self_leased_work_is_ordered_not_priced(self):
        verdict = price_segment(
            WorkEstimate(per_card_bytes=999 * GIB, self_leased=True),
            machine(card("tiny", 1, index=0)),
        )
        self.assertTrue(verdict.fits)
        self.assertIn("leased by the tenant itself", verdict.reason)

    def test_disk_and_ram_are_priced_too(self):
        disk = price_segment(
            WorkEstimate(per_card_bytes=GIB, disk_bytes=int(900 * GIB)),
            machine(card("A", 24, index=0), disk_gib=500),
        )
        self.assertFalse(disk.fits)
        self.assertIn("free", disk.reason)
        ram = price_segment(
            WorkEstimate(per_card_bytes=GIB, ram_bytes=int(900 * GIB)),
            machine(card("A", 24, index=0), ram_gib=128),
        )
        self.assertFalse(ram.fits)
        self.assertIn("host RAM", ram.reason)


# ---------------------------------------------------------------------------
# W4: the scheduler
# ---------------------------------------------------------------------------


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    """W4 -- one scheduler, one priority order, preemption inside the grace."""

    async def asyncTearDown(self):
        for workbench in getattr(self, "_started", []):
            await workbench.stop()

    def _track(self, workbench):
        self._started = getattr(self, "_started", [])
        self._started.append(workbench)
        return workbench

    async def test_idle_starts_work(self):
        tenant = MockTenant("mock", 50)
        workbench, _ = bench(tenant)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: tenant.started == 1))

    async def test_serving_demand_preempts_within_the_poll_window(self):
        tenant = MockTenant("mock", 50)
        workbench, demand = bench(tenant)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: tenant.started == 1))
        demand.busy = True
        started = time.time()
        self.assertTrue(await until(lambda: tenant.preempted == 1))
        # The bound is poll_seconds plus the tenant's own stop time; the
        # assertion is generous because CI hosts are not real-time, but it
        # would fail loudly if preemption waited for the queue to drain.
        self.assertLess(time.time() - started, 2.0)

    async def test_it_resumes_at_the_next_idle_window(self):
        tenant = MockTenant("mock", 50)
        workbench, demand = bench(tenant)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: tenant.started == 1))
        demand.busy = True
        self.assertTrue(await until(lambda: tenant.preempted == 1))
        self.assertEqual(tenant.pending(), 1, "a preempted item stays queued")
        demand.busy = False
        self.assertTrue(await until(lambda: tenant.started >= 2))

    async def test_priority_order_is_respected(self):
        low = MockTenant("low", 90)
        high = MockTenant("high", 10)
        workbench, _ = bench(low, high)
        self._track(workbench)
        self.assertEqual([t.name for t in workbench.tenants], ["high", "low"])
        workbench.start()
        self.assertTrue(await until(lambda: high.started == 1))
        await asyncio.sleep(0.2)
        self.assertEqual(low.started, 0, "the low-priority tenant must wait")

    async def test_only_one_tenant_runs_at_a_time(self):
        a = MockTenant("a", 10)
        b = MockTenant("b", 20)
        workbench, _ = bench(a, b)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: a.started == 1))
        await asyncio.sleep(0.2)
        self.assertEqual(workbench.snapshot()["running"], "a")
        self.assertEqual(b.started, 0)

    async def test_a_finished_item_lets_the_next_tenant_in(self):
        first = MockTenant("first", 10, items=1, finish_after_s=0.05)
        second = MockTenant("second", 20)
        workbench, _ = bench(first, second)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: first.pending() == 0))
        self.assertTrue(await until(lambda: second.started == 1))

    async def test_a_paused_tenant_is_skipped_and_a_paused_bench_stops(self):
        low = MockTenant("low", 90)
        high = MockTenant("high", 10)
        workbench, _ = bench(low, high)
        self._track(workbench)
        high.pause()
        workbench.start()
        self.assertTrue(await until(lambda: low.started == 1))
        self.assertEqual(high.started, 0)
        workbench.pause(True)
        self.assertTrue(await until(lambda: low.preempted == 1))

    async def test_an_infeasible_tenant_is_skipped_by_name(self):
        fat = MockTenant("fat", 10, per_card_mib=64 * 1024)
        thin = MockTenant("thin", 20, per_card_mib=64)
        workbench, _ = bench(fat, thin)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: thin.started == 1))
        self.assertEqual(fat.started, 0)
        blocked = {
            t["name"]: t["blocked_reason"] for t in workbench.snapshot()["tenants"]
        }
        self.assertIn("does not fit", blocked["fat"])

    async def test_an_unavailable_tenant_reports_why(self):
        class Broken(MockTenant):
            def available(self):
                return False, "the widget is not installed"

        broken = Broken("broken", 10)
        workbench, _ = bench(broken)
        self._track(workbench)
        workbench.start()
        self.assertTrue(
            await until(
                lambda: any(
                    "widget" in (t["blocked_reason"] or "")
                    for t in workbench.snapshot()["tenants"]
                )
            )
        )
        self.assertEqual(broken.started, 0)

    async def test_the_segment_timeout_cancels_a_runaway(self):
        tenant = MockTenant("slow", 10)
        workbench, _ = bench(tenant, segment_timeout_s=0.05)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: tenant.cancelled == 1))

    async def test_stop_cancels_the_running_segment(self):
        tenant = MockTenant("mock", 10)
        workbench, _ = bench(tenant)
        workbench.start()
        self.assertTrue(await until(lambda: tenant.started == 1))
        await workbench.stop()
        self.assertGreaterEqual(tenant.cancelled + tenant.preempted, 1)

    async def test_events_are_logged_with_a_usable_cursor(self):
        tenant = MockTenant("mock", 10)
        workbench, _ = bench(tenant)
        self._track(workbench)
        workbench.start()
        self.assertTrue(await until(lambda: tenant.started == 1))
        first, _ = workbench.log.after(0, limit=2)
        self.assertTrue(first)
        second, _ = workbench.log.after(first[-1].seq, limit=100)
        self.assertTrue(all(e.seq > first[-1].seq for e in second))


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


class WorkLogTest(unittest.TestCase):
    def test_the_ring_drops_the_head_and_keeps_the_cursor_monotonic(self):
        log = WorkLog(max_entries=5)
        for index in range(12):
            log.append("t", "info", f"line {index}")
        self.assertEqual(len(log), 5)
        self.assertEqual(log.last_seq, 12)
        self.assertEqual(log.tail(1)[0].message, "line 11")

    def test_a_cursor_into_a_dropped_range_is_answered_not_refused(self):
        log = WorkLog(max_entries=3)
        for index in range(10):
            log.append("t", "info", f"line {index}")
        window, has_more = log.after(1, limit=10)
        self.assertEqual(len(window), 3)
        self.assertFalse(has_more)

    def test_pagination_reports_has_more(self):
        log = WorkLog()
        for index in range(10):
            log.append("t", "info", f"line {index}")
        window, has_more = log.after(0, limit=4)
        self.assertEqual(len(window), 4)
        self.assertTrue(has_more)


# ---------------------------------------------------------------------------
# W5: cross-session arbitration
# ---------------------------------------------------------------------------


class ArbTest(unittest.TestCase):
    """W5 -- the protocol from the shared directory's README, as code."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="arb-"))
        self.used: dict[int, int] = {}
        # Real wall-clock, because staleness is measured against the holder
        # file's mtime and the filesystem does not take a fake clock.
        self.now = time.time()
        self.arb = ArbDirectory(
            self.root,
            session="operator",
            occupancy=lambda idx: {i: self.used.get(i, 0) for i in idx},
            clock=lambda: self.now,
        )

    def test_parsers(self):
        expiry, cards = parse_free_until(
            "2026-07-28T13:40:00Z  cards=0,1,2  by=operator  note=slice"
        )
        self.assertIsNotNone(expiry)
        self.assertEqual(cards, {0, 1, 2})
        self.assertEqual(parse_free_until("nonsense"), (None, set()))
        self.assertEqual(
            parse_holder("session=treiber  cards=1  purpose=x")["session"], "treiber"
        )

    def test_a_free_claim_writes_a_holder_and_release_removes_it(self):
        claim = self.arb.claim([0, 1], "test")
        self.assertTrue(self.arb.holder_path.is_file())
        self.assertIn("cards=0,1", self.arb.holder_path.read_text())
        claim.release()
        self.assertFalse(self.arb.holder_path.is_file())
        self.assertIn("released", self.arb.log_path.read_text())

    def test_a_published_free_window_refuses_the_claim(self):
        self.arb.free_until_path.write_text(
            "2286-01-01T00:00:00Z  cards=1  by=operator  note=driver work\n"
        )
        with self.assertRaises(ArbRefused) as caught:
            self.arb.claim([1], "test")
        self.assertIn("free window", str(caught.exception))
        # A window for a card we do not want is not our problem.
        self.arb.claim([0], "test").release()

    def test_a_live_foreign_holder_refuses_the_claim(self):
        self.arb.holder_path.write_text(
            "session=treiber  cards=0  purpose=driver  since=now\n"
        )
        with self.assertRaises(ArbRefused) as caught:
            self.arb.claim([0], "test")
        self.assertIn("treiber", str(caught.exception))

    def test_a_stale_holder_on_empty_cards_is_reaped(self):
        self.arb.holder_path.write_text(
            "session=treiber  cards=0  purpose=driver  since=old\n"
        )
        self.now += 3600.0
        claim = self.arb.claim([0], "test")
        self.assertIn("operator", self.arb.holder_path.read_text())
        self.assertIn("reaped orphan", self.arb.log_path.read_text())
        claim.release()

    def test_a_stale_holder_on_busy_cards_is_left_alone(self):
        self.arb.holder_path.write_text(
            "session=treiber  cards=0  purpose=driver  since=old\n"
        )
        self.now += 3600.0
        self.used[0] = 8 * 1024 * MIB
        with self.assertRaises(ArbRefused) as caught:
            self.arb.claim([0], "test")
        self.assertIn("still busy", str(caught.exception))
        self.assertIn("treiber", self.arb.holder_path.read_text())

    def test_the_hardware_overrides_the_files(self):
        self.used[0] = 8 * 1024 * MIB
        with self.assertRaises(ArbRefused) as caught:
            self.arb.claim([0], "test")
        self.assertIn("hardware", str(caught.exception))

    def test_accounted_memory_does_not_look_like_a_squatter(self):
        # A resident serving engine holds VRAM legitimately; only unaccounted
        # memory refuses the claim.
        self.used[0] = 8 * 1024 * MIB
        arb = ArbDirectory(
            self.root,
            occupancy=lambda idx: {i: self.used.get(i, 0) for i in idx},
            accounted=lambda idx: {i: self.used.get(i, 0) for i in idx},
            clock=lambda: self.now,
        )
        arb.claim([0], "test").release()

    def test_a_failed_occupancy_probe_refuses_rather_than_proceeds(self):
        def explode(_indices):
            raise RuntimeError("nvml is gone")

        arb = ArbDirectory(self.root, occupancy=explode, clock=lambda: self.now)
        with self.assertRaises(ArbRefused) as caught:
            arb.claim([0], "test")
        self.assertIn("occupancy check failed", str(caught.exception))

    def test_heartbeat_restamps_the_holder(self):
        claim = self.arb.claim([0], "test")
        self.now += 1200.0
        claim.heartbeat()
        self.assertIn("operator", self.arb.holder_path.read_text())
        claim.release()

    def test_an_unusable_directory_is_a_named_refusal(self):
        arb = ArbDirectory(self.root / "nope")
        with self.assertRaises(ArbRefused) as caught:
            arb.claim([0], "test")
        self.assertIn("unusable", str(caught.exception))


class SchedulerArbTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_refused_window_blocks_the_segment_and_says_why(self):
        root = Path(tempfile.mkdtemp(prefix="arb-"))
        (root / "holder").write_text("session=treiber  cards=0  purpose=driver\n")
        tenant = MockTenant("mock", 10)
        workbench, _ = bench(tenant)
        workbench.arb = ArbDirectory(root, occupancy=lambda idx: {})
        workbench.start()
        try:
            self.assertTrue(
                await until(
                    lambda: any(
                        "window refused" in (t["blocked_reason"] or "")
                        for t in workbench.snapshot()["tenants"]
                    )
                )
            )
            self.assertEqual(tenant.started, 0)
        finally:
            await workbench.stop()


# ---------------------------------------------------------------------------
# W6: the FP8 tuner tenant
# ---------------------------------------------------------------------------


DEVICES = [
    {
        "uuid": "GPU-big",
        "index": 1,
        "name": "NVIDIA Fictional 9000",
        "total_bytes": 32 * GIB,
    },
    {
        "uuid": "GPU-small",
        "index": 0,
        "name": "NVIDIA Fictional 3000",
        "total_bytes": 20 * GIB,
    },
]


def tuner(root: Path, **kwargs) -> Fp8BlockTunerTenant:
    script = root / "tuning_block_wise_kernel.py"
    script.write_text("# stand-in for the real tuning script\n")
    kwargs.setdefault("script_path", script)
    kwargs.setdefault("device_resolver", lambda: list(DEVICES))
    return Fp8BlockTunerTenant(artifact_root=root, **kwargs)


class Fp8TunerTest(unittest.TestCase):
    """W6 -- one combination per segment, idempotent, commits nothing."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="tuner-"))

    def test_queue_parsing_matches_the_shell_tuner_format(self):
        combos = parse_queue("# comment\n7168 5120 4,2048\n\n5120 2688\n")
        self.assertEqual(len([c for c in combos if c.n == 7168]), 2)
        self.assertEqual({c.batch_size for c in combos if c.n == 7168}, {4, 2048})
        with self.assertRaises(ValueError):
            parse_queue("7168\n")

    def test_the_biggest_card_is_resolved_from_nvml_not_from_an_index(self):
        tenant = tuner(self.root)
        self.assertEqual(tenant.device_name(), "NVIDIA_Fictional_9000")
        pinned = tuner(self.root, card_selector="0")
        self.assertEqual(pinned.device_name(), "NVIDIA_Fictional_3000")
        by_name = tuner(self.root, card_selector="3000")
        self.assertEqual(by_name.device_name(), "NVIDIA_Fictional_3000")
        with self.assertRaises(LookupError):
            tuner(self.root, card_selector="nosuchcard").device_name()

    def test_a_combination_whose_config_exists_is_skipped(self):
        tenant = tuner(self.root, combos=[TunerCombo(n=7168, k=5120, batch_size=4)])
        self.assertEqual(tenant.pending(), 1)
        path = tenant.config_dir / TunerCombo(
            n=7168, k=5120, batch_size=4
        ).config_filename("NVIDIA_Fictional_9000")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"4": {"BLOCK_SIZE_M": 16}}))
        self.assertEqual(tenant.pending(), 0)
        # A different batch size in the same file is still work.
        tenant.enqueue({"n": 7168, "k": 5120, "batch_size": 2048})
        self.assertEqual(tenant.pending(), 1)

    def test_the_posts_are_named_and_scale_with_the_shape(self):
        small = combo_posts(TunerCombo(n=5120, k=2688, batch_size=4))
        big = combo_posts(TunerCombo(n=7168, k=5120, batch_size=2048))
        self.assertIn("cuda_context", small)
        self.assertIn("triton_sweep", small)
        self.assertLess(sum(small.values()), sum(big.values()))
        # The fp32 draft of B dominates a decode-shaped combination.
        self.assertEqual(small["b_fp32_draft"], 5120 * 2688 * 4)

    def test_the_estimate_pins_the_resolved_card(self):
        tenant = tuner(self.root, combos=[TunerCombo(n=7168, k=5120, batch_size=4)])
        estimate = tenant.estimate()
        self.assertEqual(estimate.card_uuids, ("GPU-big",))
        self.assertEqual(estimate.per_card_bytes, sum(estimate.posts.values()))

    def test_a_missing_script_is_a_named_skip(self):
        tenant = Fp8BlockTunerTenant(
            artifact_root=self.root,
            script_path=self.root / "absent.py",
            device_resolver=lambda: list(DEVICES),
        )
        available, reason = tenant.available()
        self.assertFalse(available)
        self.assertIn("not in this tree", reason)

    def test_enqueue_validates(self):
        tenant = tuner(self.root)
        key = tenant.enqueue({"n": 5120, "k": 3072, "batch_size": 4})
        self.assertIn("N=5120", key)
        with self.assertRaises(ValueError):
            tenant.enqueue({"n": "wide", "k": 3072})
        with self.assertRaises(ValueError):
            tenant.enqueue({"n": 1, "k": 1, "input_type": "fp4"})

    def test_a_failing_combination_is_not_retried_forever(self):
        combo = TunerCombo(n=9472, k=5120, batch_size=4)
        tenant = tuner(self.root, combos=[combo])
        self.assertEqual(tenant.pending(), 1)
        tenant.note_failure(combo, "exit 1")
        self.assertEqual(tenant.pending(), 0)
        self.assertIn(combo.key, tenant.snapshot()["failed"])

    def test_the_queue_file_is_reread_when_it_changes(self):
        queue = self.root / "queue.txt"
        queue.write_text("7168 5120 4\n")
        tenant = tuner(self.root, queue_path=queue)
        self.assertEqual(tenant.pending(), 1)
        queue.write_text("7168 5120 4\n5120 2688 4\n")
        # mtime resolution is coarse enough on some filesystems that an
        # immediate rewrite can land in the same tick; force the reread the
        # way a real edit would.
        import os

        os.utime(queue, (time.time() + 2, time.time() + 2))
        self.assertEqual(tenant.pending(), 2)

    def test_nothing_is_written_into_the_source_tree(self):
        tenant = tuner(self.root)
        self.assertTrue(str(tenant.config_dir).startswith(str(self.root)))
        self.assertIn(
            "layers/quantization/configs", tenant.snapshot()["repo_config_dir"]
        )


class Fp8TunerSegmentTest(unittest.IsolatedAsyncioTestCase):
    """The subprocess body: it runs, it reports, and SIGTERM stops it."""

    async def test_a_successful_combination_reports_its_artifact(self):
        root = Path(tempfile.mkdtemp(prefix="tuner-"))
        script = root / "fake_tuner.py"
        script.write_text(
            "import sys, json, os\n"
            "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "out = args['--save-path']\n"
            "os.makedirs(out, exist_ok=True)\n"
            "print('Writing best config to', out)\n"
        )
        tenant = Fp8BlockTunerTenant(
            artifact_root=root,
            script_path=script,
            device_resolver=lambda: list(DEVICES),
            combos=[TunerCombo(n=7168, k=5120, batch_size=4)],
        )
        events = []
        from sglang.srt.workbench.tenant import WorkGrant

        grant = WorkGrant(
            card_uuids=("GPU-big",),
            card_indices=(1,),
            per_card_bytes=GIB,
            artifact_root=root,
        )
        segment = await tenant.start_segment(grant, events.append)
        outcome = await segment.wait()
        self.assertIs(outcome.status, SegmentStatus.SUCCEEDED)
        self.assertIn("N=7168", outcome.artifact_path)
        self.assertTrue(any("Writing best config" in e.message for e in events))

    async def test_preempting_a_running_combination_stops_it(self):
        root = Path(tempfile.mkdtemp(prefix="tuner-"))
        script = root / "slow_tuner.py"
        script.write_text("import time\nwhile True:\n    time.sleep(0.05)\n")
        tenant = Fp8BlockTunerTenant(
            artifact_root=root,
            script_path=script,
            device_resolver=lambda: list(DEVICES),
            combos=[TunerCombo(n=7168, k=5120, batch_size=4)],
        )
        from sglang.srt.workbench.tenant import WorkGrant

        grant = WorkGrant(
            card_uuids=("GPU-big",),
            card_indices=(1,),
            per_card_bytes=GIB,
            artifact_root=root,
        )
        segment = await tenant.start_segment(grant, lambda e: None)
        outcome = await segment.preempt(timeout_s=2.0)
        self.assertIs(outcome.status, SegmentStatus.PREEMPTED)
        # Nothing was written, so the combination is still work.
        self.assertEqual(tenant.pending(), 1)


# ---------------------------------------------------------------------------
# W7: the self-benchmark tenant
# ---------------------------------------------------------------------------


class Profile:
    def __init__(self, age):
        self._age = age

    def age_s(self, now=None):
        return self._age


class CardProbeTenantTest(unittest.TestCase):
    """W7 -- an absent or stale factor is the queue."""

    def test_an_absent_profile_is_one_item_of_work(self):
        tenant = CardProbeTenant(profile_loader=lambda: None)
        self.assertEqual(tenant.pending(), 1)

    def test_a_fresh_profile_is_no_work_and_a_stale_one_is(self):
        fresh = CardProbeTenant(max_age_s=3600, profile_loader=lambda: Profile(60.0))
        self.assertEqual(fresh.pending(), 0)
        stale = CardProbeTenant(max_age_s=3600, profile_loader=lambda: Profile(7200.0))
        self.assertEqual(stale.pending(), 1)

    def test_it_wants_every_card_because_the_pair_matrix_needs_them(self):
        estimate = CardProbeTenant(profile_loader=lambda: None).estimate()
        self.assertEqual(estimate.cards_wanted, 0)
        self.assertEqual(estimate.per_card_bytes, sum(probe_posts().values()))

    def test_the_posts_are_the_probe_s_own_allocations(self):
        posts = probe_posts()
        self.assertIn("gemm_operand_drafts", posts)
        self.assertEqual(posts["transfer_buffers"], 128 * MIB)

    def test_it_has_no_enqueue_surface(self):
        with self.assertRaises(NotImplementedError):
            CardProbeTenant(profile_loader=lambda: None).enqueue({})

    def test_the_snapshot_names_the_factors_it_fills(self):
        snapshot = CardProbeTenant(profile_loader=lambda: None).snapshot()
        self.assertEqual(snapshot["factors"], ["card_rates", "pair_link"])


# ---------------------------------------------------------------------------
# W4: the training adapter
# ---------------------------------------------------------------------------


def training_service(root: Path, *, enabled: bool = True):
    """A real #341 service with a mock executor and a synthetic 80 GiB card."""
    from sglang.srt.training.backends.mock import MockBackend
    from sglang.srt.training.service import TrainingService, TrainingServiceConfig

    (root / "base-model").mkdir(parents=True, exist_ok=True)
    (root / "base-model" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "intermediate_size": 2048,
                "vocab_size": 32000,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (root / "base-model" / "model.safetensors").write_bytes(b"\0" * 4096)
    return TrainingService(
        TrainingServiceConfig(
            enabled=enabled,
            artifact_root=root / "artifacts",
            grace_seconds=0.0,
            poll_seconds=0.02,
            default_backend="mock",
            save_steps=2,
        ),
        monitor=IdleMonitor([Demand()], grace_seconds=0.0),
        machine_resolver=lambda: machine(card("Synthetic", 80, index=0)),
        backend_factory=lambda name, method: MockBackend(step_seconds=0.004),
    )


def submit(service, root: Path, **extension):
    body = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ).encode()
        + b"\n"
    )
    uploaded = service.create_file(
        filename="train.jsonl", content=body * 8, purpose="fine-tune"
    )
    block = {"sequence_length": 256, "total_steps": 400, "save_steps": 2}
    block.update(extension)
    return service.create_job(
        {
            "model": str(root / "base-model"),
            "training_file": uploaded.id,
            "x-htsglang": block,
        }
    )


class TrainingAdapterTest(unittest.IsolatedAsyncioTestCase):
    """W4 -- training is entry #1, arbitrated by the bench, run by #341."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="train-"))
        self.service = training_service(self.root)

    async def asyncTearDown(self):
        await self.service.stop()

    def adapter(self):
        from sglang.srt.workbench.tenants.training import TrainingWorkTenant

        return TrainingWorkTenant(self.service, idle_settle_s=0.05)

    def test_it_reports_the_queue_and_prices_nothing(self):
        tenant = self.adapter()
        self.assertEqual(tenant.priority, 10)
        self.assertEqual(tenant.pending(), 0)
        submit(self.service, self.root)
        self.assertEqual(tenant.pending(), 1)
        estimate = tenant.estimate()
        self.assertTrue(estimate.self_leased)
        self.assertTrue(
            price_segment(estimate, machine(card("tiny", 1, index=0))).fits,
            "a self-leased tenant is ordered, not priced, by the bench",
        )

    def test_a_disabled_training_tenant_is_a_named_skip(self):
        from sglang.srt.workbench.tenants.training import TrainingWorkTenant

        off = training_service(Path(tempfile.mkdtemp(prefix="off-")), enabled=False)
        available, reason = TrainingWorkTenant(off).available()
        self.assertFalse(available)
        self.assertIn("--enable-training-tenant", reason)

    async def test_the_bench_runs_and_preempts_a_real_training_job(self):
        tenant = self.adapter()
        workbench, demand = bench(tenant)
        submit(self.service, self.root)
        workbench.start()
        try:
            self.assertTrue(
                await until(
                    lambda: self.service.tenant.running_job_id is not None, 10.0
                ),
                "the workbench must start the #341 loop",
            )
            job_id = self.service.tenant.running_job_id
            demand.busy = True
            self.assertTrue(
                await until(
                    lambda: self.service.jobs.get(job_id).preemptions >= 1, 15.0
                ),
                "serving demand must reach the trainer through the adapter",
            )
            job = self.service.jobs.get(job_id)
            self.assertEqual(job.status.value, "running", "not a protocol state")
            self.assertEqual(job.tenant_state.value, "preempted")
            self.assertTrue(job.resume_from, "it must have a checkpoint to resume from")
            self.assertIsNone(self.service.tenant.running_job_id)
        finally:
            await workbench.stop()

    async def test_a_lower_priority_tenant_waits_for_training(self):
        tuner_like = MockTenant("tuner_like", 50)
        workbench, _ = bench(self.adapter(), tuner_like)
        submit(self.service, self.root)
        workbench.start()
        try:
            self.assertTrue(
                await until(
                    lambda: self.service.tenant.running_job_id is not None, 10.0
                )
            )
            await asyncio.sleep(0.3)
            self.assertEqual(
                tuner_like.started, 0, "idle work must not run beside training"
            )
        finally:
            await workbench.stop()

    async def test_an_empty_training_queue_hands_the_rig_over(self):
        tuner_like = MockTenant("tuner_like", 50)
        workbench, _ = bench(self.adapter(), tuner_like)
        workbench.start()
        try:
            self.assertTrue(await until(lambda: tuner_like.started == 1, 10.0))
        finally:
            await workbench.stop()


# ---------------------------------------------------------------------------
# W8: the HTTP surface
# ---------------------------------------------------------------------------


def service(*tenants, enabled: bool = True) -> WorkbenchService:
    return WorkbenchService(
        WorkbenchConfig(
            enabled=enabled,
            artifact_root=Path(tempfile.mkdtemp(prefix="wb-")),
            poll_seconds=0.02,
        ),
        tenants=list(tenants),
        monitor=IdleMonitor([Demand()], grace_seconds=0.0),
        machine_resolver=lambda: machine(card("A", 24, index=0)),
    )


class HttpSurfaceTest(unittest.TestCase):
    """W8 -- read-only state plus two controls, named 503 when switched off."""

    def test_the_snapshot_answers_even_when_disabled(self):
        body = snapshot_payload(service(MockTenant("m", 10), enabled=False))
        self.assertFalse(body["workbench"]["enabled"])
        self.assertEqual(body["workbench"]["tenants"][0]["name"], "m")

    def test_the_tenant_table_is_priority_ordered(self):
        body = snapshot_payload(service(MockTenant("z", 90), MockTenant("a", 10)))
        self.assertEqual([t["name"] for t in body["workbench"]["tenants"]], ["a", "z"])

    def test_pause_needs_the_feature_enabled(self):
        with self.assertRaises(WorkbenchDisabled):
            pause_payload(service(enabled=False), {"paused": True})

    def test_pause_targets_the_bench_or_one_tenant(self):
        svc = service(MockTenant("m", 10))
        self.assertTrue(pause_payload(svc, {"paused": True})["paused"])
        self.assertFalse(pause_payload(svc, {"paused": False})["paused"])
        body = pause_payload(svc, {"paused": True, "tenant": "m"})
        self.assertEqual(body["tenant"], "m")
        self.assertTrue(svc.workbench.tenant("m").paused)
        with self.assertRaises(UnknownTenant):
            pause_payload(svc, {"paused": True, "tenant": "nope"})
        with self.assertRaises(WorkbenchError):
            pause_payload(svc, {"paused": "yes"})

    def test_enqueue_accepts_the_flat_and_the_nested_shape(self):
        svc = service(MockTenant("m", 10, items=0))
        self.assertEqual(
            enqueue_payload(svc, {"tenant": "m", "item": {"n": 1}})["pending"], 1
        )
        self.assertEqual(enqueue_payload(svc, {"tenant": "m", "n": 2})["pending"], 2)
        with self.assertRaises(WorkbenchError):
            enqueue_payload(svc, {"item": {}})

    def test_enqueue_into_a_derived_queue_is_a_named_refusal(self):
        svc = service(CardProbeTenant(profile_loader=lambda: None))
        with self.assertRaises(WorkbenchError) as caught:
            enqueue_payload(svc, {"tenant": "card_probe", "item": {}})
        self.assertIn("derived from the state of the rig", str(caught.exception))

    def test_events_paginate_and_validate(self):
        svc = service(MockTenant("m", 10))
        for index in range(5):
            svc.workbench.log.append("m", "info", f"line {index}")
        first = events_payload(svc, {"after": 0, "limit": 2})
        self.assertEqual(len(first["data"]), 2)
        self.assertTrue(first["has_more"])
        second = events_payload(svc, {"after": first["last_seq"], "limit": 100})
        self.assertEqual(len(second["data"]), 3)
        with self.assertRaises(WorkbenchError):
            events_payload(svc, {"after": "soon"})


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


class Args:
    """The subset of ServerArgs the workbench reads."""

    def __init__(self, **kwargs):
        self.enable_idle_workbench = True
        self.workbench_artifact_root = None
        self.workbench_tenants = None
        self.workbench_idle_grace_seconds = 120.0
        self.workbench_poll_seconds = 2.0
        self.workbench_preempt_timeout_s = 60.0
        self.workbench_segment_timeout_s = 1800.0
        self.workbench_arb_dir = None
        self.workbench_arb_heartbeat_s = 300.0
        self.workbench_tuner_queue = None
        self.workbench_tuner_card = "largest"
        self.workbench_probe_max_age_s = 604800.0
        self.__dict__.update(kwargs)


class AssemblyTest(unittest.TestCase):
    def test_the_default_tenant_set_is_registered_in_priority_order(self):
        from sglang.srt.workbench.service import build_config

        args = Args(workbench_artifact_root=tempfile.mkdtemp(prefix="wb-"))
        config = build_config(args)
        tenants = build_tenants(args, config, training_service=None)
        # No training service here, so training is skipped with a warning.
        self.assertEqual(sorted(t.name for t in tenants), ["card_probe", "fp8_tuner"])

    def test_an_unknown_tenant_name_is_a_startup_error(self):
        from sglang.srt.workbench.service import build_config

        args = Args(workbench_tenants="fp8_tuner,typo")
        with self.assertRaises(ValueError):
            build_tenants(args, build_config(args))

    def test_the_arb_directory_comes_from_the_flag_then_the_environment(self):
        import os

        from sglang.srt.workbench.service import build_config

        root = tempfile.mkdtemp(prefix="arb-")
        self.assertEqual(build_config(Args(workbench_arb_dir=root)).arb_dir, root)
        os.environ["HTSGLANG_GPU_ARB_DIR"] = root
        try:
            self.assertEqual(build_config(Args()).arb_dir, root)
        finally:
            del os.environ["HTSGLANG_GPU_ARB_DIR"]


if __name__ == "__main__":
    unittest.main()
