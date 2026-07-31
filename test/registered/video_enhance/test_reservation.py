"""The cross-process VRAM ledger: invariant, leases, locking, waste.

Hermetic. Every test runs against a ``tmp_path``-style root, supplies the
card total explicitly instead of asking NVML, and injects its own clock and
pid-liveness predicate. The two concurrency tests use real threads and real
subprocesses respectively, because a lock that is only exercised from one
flow of control is not tested at all.

The NVML identity helper the ledger keys on is covered here too, against an
injected fake binding, because a ledger keyed on the wrong card is the same
defect as no ledger at all.

    python -m pytest test/registered/video_enhance/test_reservation.py -v
"""

import builtins
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

from sglang.srt.video_enhance import reservation as reservation_module
from sglang.srt.video_enhance.reservation import (
    DEFAULT_CORRIDOR_BYTES,
    MIB,
    CardBusyError,
    ReservationRejected,
    ReservationStore,
    TenantState,
    UnknownTenantError,
    available_bytes,
    default_store_root,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Filesystem and process-level tests only; no device is touched.
register_cpu_ci(est_time=25, suite="base-a-test-cpu")

CARD = "GPU-11111111-2222-3333-4444-555555555555"
CARD_TOTAL = 10 * 1024 * MIB  # 10 GiB, as NVML would report it


class _Clock:
    """A hand-advanced clock, so lease expiry is a decision and not a wait."""

    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def dead_pid():
    """A pid that has certainly exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def child_script(body):
    """A standalone child that loads the ledger module straight from its file.

    Importing ``sglang.srt.video_enhance.reservation`` the normal way drags in
    the whole package for a child whose only job is to take a lock, so the
    child loads the single file instead. Registering it in ``sys.modules``
    before executing it is required: ``@dataclass`` resolves annotations
    through ``sys.modules[cls.__module__]``.
    """
    return textwrap.dedent(f"""
        import importlib.util, pathlib, sys, time
        spec = importlib.util.spec_from_file_location(
            "ve_reservation", {reservation_module.__file__!r}
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        """) + textwrap.dedent(body)


class LedgerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.clock = _Clock()

    def make_store(self, **kwargs):
        kwargs.setdefault("clock", self.clock)
        return ReservationStore(self.root, **kwargs)


class TestEntrySchemaAndRoundTrip(LedgerTestCase):
    def test_acquire_writes_the_documented_schema(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=4 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
            posts={"stage_sr": 2048 * MIB},
            pid=os.getpid(),
        )
        payload = json.loads(store.ledger_path(CARD).read_text())
        self.assertEqual(payload["card_uuid"], CARD)
        (entry,) = payload["entries"]
        self.assertEqual(
            set(entry),
            {
                "tenant_id",
                "klass",
                "state",
                "reserved_bytes",
                "measured_bytes",
                "posts",
                "pid",
                "heartbeat_ts",
                "lease_expiry_ts",
                # #344: whether this tenant's consumer is a dead suspect. Part
                # of the written schema because the reclamation ladder reading
                # the file runs in another process.
                "in_grace",
                "grace_since_ts",
            },
        )
        self.assertEqual(entry["state"], "HOT")
        self.assertEqual(entry["posts"], {"stage_sr": 2048 * MIB})

    def test_read_of_an_untouched_card_is_empty_not_an_error(self):
        self.assertEqual(self.make_store().read(CARD).entries, ())

    def test_reacquire_replaces_rather_than_stacks(self):
        store = self.make_store()
        for size in (2, 6):
            store.acquire(
                card_uuid=CARD,
                tenant_id="enhance-0",
                klass=3,
                reserved_bytes=size * 1024 * MIB,
                nvml_total_bytes=CARD_TOTAL,
            )
        ledger = store.read(CARD)
        self.assertEqual(len(ledger.entries), 1)
        self.assertEqual(ledger.reserved_bytes, 6 * 1024 * MIB)

    def test_release_removes_the_entry(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        self.assertTrue(store.release(CARD, "enhance-0"))
        self.assertFalse(store.release(CARD, "enhance-0"))
        self.assertEqual(store.read(CARD).entries, ())

    def test_default_root_honours_the_environment_override(self):
        override = str(self.root / "elsewhere")
        os.environ["HTSGLANG_VRAM_LEDGER_ROOT"] = override
        self.addCleanup(os.environ.pop, "HTSGLANG_VRAM_LEDGER_ROOT", None)
        self.assertEqual(default_store_root(), Path(override))


class TestInvariant(LedgerTestCase):
    def test_corridor_belongs_to_the_card(self):
        """The last 400 MiB is the card's, so no tenant may claim it."""
        store = self.make_store()
        exact = CARD_TOTAL - DEFAULT_CORRIDOR_BYTES
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=exact,
            nvml_total_bytes=CARD_TOTAL,
        )
        store.release(CARD, "enhance-0")
        with self.assertRaises(ReservationRejected):
            store.acquire(
                card_uuid=CARD,
                tenant_id="enhance-0",
                klass=3,
                reserved_bytes=exact + 1,
                nvml_total_bytes=CARD_TOTAL,
            )

    def test_rejection_names_card_holders_request_and_shortfall(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="llm-hot",
            klass=1,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        with self.assertRaises(ReservationRejected) as ctx:
            store.acquire(
                card_uuid=CARD,
                tenant_id="enhance-0",
                klass=3,
                reserved_bytes=4 * 1024 * MIB,
                nvml_total_bytes=CARD_TOTAL,
            )
        error = ctx.exception
        message = str(error)
        self.assertIn(CARD, message)
        self.assertIn("llm-hot", message)
        self.assertIn("4096 MiB", message)  # requested
        self.assertIn("8192 MiB", message)  # already held
        self.assertIn("400 MiB", message)  # corridor
        self.assertIn("10240 MiB", message)  # NVML total
        # 8192 + 4096 + 400 - 10240 = 2448
        self.assertIn("2448 MiB", message)
        self.assertEqual(error.shortfall_bytes, 2448 * MIB)
        self.assertEqual(error.holders, ("llm-hot",))
        # Nothing was written.
        self.assertEqual([e.tenant_id for e in store.read(CARD).entries], ["llm-hot"])

    def test_host_resident_tenants_do_not_block_an_admission(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="llm-parked",
            klass=1,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
            state=TenantState.WARM_HOST,
        )
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        self.assertEqual(store.read(CARD).reserved_bytes, 8 * 1024 * MIB)

    def test_promotion_back_to_the_gpu_is_checked_like_an_admission(self):
        store = self.make_store(total_bytes_resolver=lambda _uuid: CARD_TOTAL)
        store.acquire(
            card_uuid=CARD,
            tenant_id="llm-parked",
            klass=1,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
            state=TenantState.WARM_HOST,
        )
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        with self.assertRaises(ReservationRejected):
            store.set_state(CARD, "llm-parked", TenantState.HOT)
        # Demotion is always allowed: it only frees bytes.
        store.set_state(CARD, "enhance-0", TenantState.WARM_HOST)
        store.set_state(CARD, "llm-parked", TenantState.HOT)
        self.assertEqual(store.read(CARD).reserved_bytes, 8 * 1024 * MIB)

    def test_available_bytes_matches_what_an_admission_would_be_checked_against(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="llm-hot",
            klass=1,
            reserved_bytes=6 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        free = available_bytes(store, CARD, CARD_TOTAL)
        self.assertEqual(free, CARD_TOTAL - 6 * 1024 * MIB - DEFAULT_CORRIDOR_BYTES)
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=free,
            nvml_total_bytes=CARD_TOTAL,
        )
        self.assertEqual(available_bytes(store, CARD, CARD_TOTAL), 0)

    def test_unknown_tenant_is_named(self):
        store = self.make_store()
        with self.assertRaises(UnknownTenantError):
            store.heartbeat(CARD, "never-acquired")


class TestLeaseAndHeartbeat(LedgerTestCase):
    def _acquire(self, store, pid, lease_seconds=60.0):
        return store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
            pid=pid,
            lease_seconds=lease_seconds,
        )

    def test_expired_lease_with_a_dead_pid_is_reclaimed(self):
        store = self.make_store()
        self._acquire(store, pid=dead_pid())
        self.clock.advance(61.0)
        reclaimed = store.reap(CARD)
        self.assertEqual([e.tenant_id for e in reclaimed], ["enhance-0"])
        self.assertEqual(store.read(CARD).entries, ())

    def test_expired_lease_with_a_live_pid_is_never_reclaimed(self):
        """A starved heartbeat thread is not a dead tenant.

        The process is still holding device memory; handing the same bytes to
        a second tenant on the strength of a lapsed lease is exactly the
        double-allocation the ledger exists to prevent.
        """
        store = self.make_store()
        self._acquire(store, pid=os.getpid())
        self.clock.advance(10_000.0)
        self.assertEqual(store.reap(CARD), [])
        self.assertEqual(store.read(CARD).reserved_bytes, 8 * 1024 * MIB)

        with self.assertRaises(ReservationRejected):
            store.acquire(
                card_uuid=CARD,
                tenant_id="other",
                klass=3,
                reserved_bytes=8 * 1024 * MIB,
                nvml_total_bytes=CARD_TOTAL,
            )

    def test_live_lease_with_a_dead_pid_is_not_reclaimed_yet(self):
        store = self.make_store()
        self._acquire(store, pid=dead_pid())
        self.clock.advance(10.0)
        self.assertEqual(store.reap(CARD), [])

    def test_heartbeat_pushes_the_lease_out(self):
        store = self.make_store()
        entry = self._acquire(store, pid=dead_pid(), lease_seconds=60.0)
        self.assertEqual(entry.lease_expiry_ts, self.clock.now + 60.0)

        self.clock.advance(50.0)
        refreshed = store.heartbeat(CARD, "enhance-0", lease_seconds=60.0)
        self.assertEqual(refreshed.heartbeat_ts, self.clock.now)
        self.assertEqual(refreshed.lease_expiry_ts, self.clock.now + 60.0)

        self.clock.advance(50.0)
        self.assertEqual(store.reap(CARD), [])

    def test_acquire_reclaims_a_crashed_tenant_and_then_fits(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="crashed",
            klass=3,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
            pid=dead_pid(),
            lease_seconds=60.0,
        )
        self.clock.advance(61.0)
        store.acquire(
            card_uuid=CARD,
            tenant_id="fresh",
            klass=3,
            reserved_bytes=8 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        self.assertEqual([e.tenant_id for e in store.read(CARD).entries], ["fresh"])


class TestWasteAccounting(LedgerTestCase):
    def test_waste_is_reserved_minus_measured_and_is_only_reported(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=6 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        # Nothing measured yet: the whole reservation reads as waste, and it
        # must stay reserved regardless.
        self.assertEqual(store.waste(CARD), 6 * 1024 * MIB)

        store.update_measured(CARD, "enhance-0", 4 * 1024 * MIB)
        self.assertEqual(store.waste(CARD), 2 * 1024 * MIB)
        self.assertEqual(store.read(CARD).reserved_bytes, 6 * 1024 * MIB)

    def test_waste_goes_negative_when_a_tenant_overruns_its_declaration(self):
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=2 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        store.update_measured(CARD, "enhance-0", 3 * 1024 * MIB)
        self.assertEqual(store.waste(CARD), -1024 * MIB)

    def test_waste_ignores_tenants_that_gave_their_bytes_back(self):
        store = self.make_store(total_bytes_resolver=lambda _uuid: CARD_TOTAL)
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=4 * 1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        store.set_state(CARD, "enhance-0", TenantState.COLD)
        self.assertEqual(store.waste(CARD), 0)


class TestConcurrency(LedgerTestCase):
    def test_racing_threads_cannot_both_pass_the_invariant(self):
        """Four threads, three fitting slots, a widened critical section.

        The probe holds each acquire inside the lock long enough that every
        thread would read the same pre-state if the lock were absent, in
        which case all four would pass the check. Exactly three succeeding is
        what proves the section is serialised.
        """
        barrier = threading.Barrier(4)
        store = self.make_store(critical_section_probe=lambda _op: time.sleep(0.05))

        outcomes = {}
        lock = threading.Lock()

        def worker(index):
            barrier.wait()
            try:
                store.acquire(
                    card_uuid=CARD,
                    tenant_id=f"tenant-{index}",
                    klass=3,
                    reserved_bytes=3 * 1024 * MIB,
                    nvml_total_bytes=CARD_TOTAL,
                )
                result = "acquired"
            except ReservationRejected:
                result = "rejected"
            with lock:
                outcomes[index] = result

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(sorted(outcomes.values()).count("acquired"), 3)
        self.assertEqual(sorted(outcomes.values()).count("rejected"), 1)

        ledger = store.read(CARD)
        self.assertEqual(len(ledger.entries), 3)
        self.assertLessEqual(ledger.reserved_bytes + DEFAULT_CORRIDOR_BYTES, CARD_TOTAL)

    def test_racing_processes_cannot_both_pass_the_invariant(self):
        """The same proof across real processes, where flock is the only guard.

        Threads share an interpreter; processes share nothing but the files,
        so this is the case the deployment actually has -- co-located tenants
        are separate processes by design.
        """
        script = child_script(f"""
            store = module.ReservationStore(
                {str(self.root)!r},
                critical_section_probe=lambda op: time.sleep(0.15),
            )
            try:
                store.acquire(
                    card_uuid={CARD!r},
                    tenant_id=sys.argv[1],
                    klass=3,
                    reserved_bytes={3 * 1024 * MIB},
                    nvml_total_bytes={CARD_TOTAL},
                )
            except module.ReservationRejected:
                sys.exit(3)
            sys.exit(0)
            """)
        children = [
            subprocess.Popen([sys.executable, "-c", script, f"proc-{i}"])
            for i in range(6)
        ]
        codes = [child.wait(timeout=120) for child in children]
        self.assertEqual(codes.count(0), 3, f"exit codes: {codes}")
        self.assertEqual(codes.count(3), 3, f"exit codes: {codes}")

        ledger = self.make_store().read(CARD)
        self.assertEqual(len(ledger.entries), 3)
        self.assertLessEqual(ledger.reserved_bytes + DEFAULT_CORRIDOR_BYTES, CARD_TOTAL)

    def test_a_partial_write_is_never_observable(self):
        """One reader, many writers, temp-file plus rename underneath.

        Every read must parse and must see a whole number of entries; a
        torn JSON file would surface here as a decode failure.
        """
        store = self.make_store()
        stop = threading.Event()
        seen = []
        failures = []

        def reader():
            while not stop.is_set():
                try:
                    seen.append(len(store.read(CARD).entries))
                except Exception as exc:  # pragma: no cover - the failure path
                    failures.append(exc)
                    return

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for index in range(40):
                store.acquire(
                    card_uuid=CARD,
                    tenant_id=f"tenant-{index % 3}",
                    klass=3,
                    reserved_bytes=1024 * MIB,
                    nvml_total_bytes=CARD_TOTAL,
                )
        finally:
            stop.set()
            thread.join(timeout=30)

        self.assertEqual(failures, [])
        self.assertTrue(seen)
        self.assertTrue(all(0 <= count <= 3 for count in seen))


class TestCardExclusiveLock(LedgerTestCase):
    def test_the_lock_is_exclusive_across_processes(self):
        """Engine builds and memory probes run one at a time per card (§6.4)."""
        ready = self.root / "held"
        script = child_script(f"""
            store = module.ReservationStore({str(self.root)!r})
            with store.card_exclusive_lock({CARD!r}, purpose="engine-build"):
                pathlib.Path({str(ready)!r}).write_text("held")
                time.sleep(2.0)
            """)
        child = subprocess.Popen([sys.executable, "-c", script])
        try:
            deadline = time.time() + 30
            while not ready.exists() and time.time() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists(), "child never took the lock")

            store = ReservationStore(self.root)
            with self.assertRaises(CardBusyError) as ctx:
                with store.card_exclusive_lock(CARD, timeout=0.3, purpose="probe"):
                    pass
            self.assertIn(CARD, str(ctx.exception))
            self.assertIn("probe", str(ctx.exception))
        finally:
            child.wait(timeout=60)

        # Released with the child, and re-takeable.
        with ReservationStore(self.root).card_exclusive_lock(CARD, timeout=5.0):
            pass

    def test_the_card_lock_does_not_block_the_ledger(self):
        """A minutes-long build must not stall a co-tenant's heartbeat."""
        store = self.make_store()
        store.acquire(
            card_uuid=CARD,
            tenant_id="enhance-0",
            klass=3,
            reserved_bytes=1024 * MIB,
            nvml_total_bytes=CARD_TOTAL,
        )
        with store.card_exclusive_lock(CARD, purpose="engine-build"):
            store.heartbeat(CARD, "enhance-0")
            self.assertEqual(len(store.read(CARD).entries), 1)


class _FakeHandle:
    def __init__(self, index, uuid, name, total, bus):
        self.index = index
        self.uuid = uuid
        self.name = name
        self.total = total
        self.bus = bus


class _FakePynvml:
    """Just enough NVML surface to exercise the identity helper offline."""

    def __init__(self, handles):
        self._handles = handles
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return len(self._handles)

    def nvmlDeviceGetHandleByIndex(self, index):
        return self._handles[index]

    def nvmlDeviceGetUUID(self, handle):
        # NVML returns bytes on some bindings and str on others.
        return handle.uuid.encode()

    def nvmlDeviceGetName(self, handle):
        return handle.name

    def nvmlDeviceGetMemoryInfo(self, handle):
        return type("Mem", (), {"total": handle.total})()

    def nvmlDeviceGetPciInfo(self, handle):
        return type("Pci", (), {"busId": handle.bus})()


class TestNvmlIdentitySeam(unittest.TestCase):
    """UUID-keyed identity, resolved without a driver present."""

    HANDLES = [
        _FakeHandle(
            0,
            "GPU-aaaa",
            "NVIDIA GeForce RTX 3080",
            20 * 1024 * MIB,
            b"00000000:01:00.0",
        ),
        _FakeHandle(
            1,
            "GPU-bbbb",
            "NVIDIA GeForce RTX 5090",
            32 * 1024 * MIB,
            "00000000:2D:00.0",
        ),
        _FakeHandle(
            2,
            "GPU-cccc",
            "NVIDIA GeForce RTX 3080",
            20 * 1024 * MIB,
            "00000000:41:00.0",
        ),
    ]

    def setUp(self):
        self.fake = _FakePynvml(self.HANDLES)
        self._saved = sys.modules.get("pynvml")
        sys.modules["pynvml"] = self.fake
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("pynvml", None)
        else:
            sys.modules["pynvml"] = self._saved

    def test_devices_carry_uuid_name_total_and_bus(self):
        from sglang.srt.video_enhance import nvml

        devices = nvml.list_devices()
        self.assertEqual(
            [d.uuid for d in devices], ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]
        )
        self.assertEqual(devices[1].total_mib, 32 * 1024)
        self.assertEqual(devices[0].pci_bus_id, "00000000:01:00.0")
        self.assertEqual(self.fake.init_calls, self.fake.shutdown_calls)

    def test_name_fragment_resolves_a_unique_card_and_refuses_an_ambiguous_one(self):
        from sglang.srt.video_enhance import nvml

        self.assertEqual(nvml.resolve_index_by_name_fragment("5090"), 1)
        with self.assertRaises(nvml.DeviceNotFoundError) as ctx:
            nvml.resolve_index_by_name_fragment("3080")
        self.assertIn("matches 2 devices", str(ctx.exception))
        with self.assertRaises(nvml.DeviceNotFoundError):
            nvml.resolve_index_by_name_fragment("Radeon")

    def test_current_device_uuid_reads_the_pinning(self):
        from sglang.srt.video_enhance import nvml

        for value, expected in (("1", "GPU-bbbb"), ("GPU-cccc", "GPU-cccc")):
            with unittest.mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                self.assertEqual(nvml.current_device_uuid(), expected)

    def test_out_of_range_pinning_is_named(self):
        from sglang.srt.video_enhance import nvml

        with unittest.mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "9"}):
            with self.assertRaises(nvml.DeviceNotFoundError):
                nvml.current_device_uuid()

    def test_the_ledger_falls_through_to_nvml_for_the_card_total(self):
        """No explicit total and no resolver: the UUID is the only input."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ReservationStore(Path(tmp))
            store.acquire(
                card_uuid="GPU-bbbb",
                tenant_id="enhance-0",
                klass=3,
                reserved_bytes=31 * 1024 * MIB,
            )
            with self.assertRaises(ReservationRejected) as ctx:
                store.acquire(
                    card_uuid="GPU-bbbb",
                    tenant_id="enhance-1",
                    klass=3,
                    reserved_bytes=2 * 1024 * MIB,
                )
            self.assertIn("32768 MiB", str(ctx.exception))


class TestNvmlAbsent(unittest.TestCase):
    def test_the_module_imports_and_every_entry_point_says_why_it_cannot_answer(self):
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("pynvml is not installed")
            return real_import(name, *args, **kwargs)

        saved = sys.modules.pop("pynvml", None)
        try:
            with unittest.mock.patch.object(builtins, "__import__", blocked):
                from sglang.srt.video_enhance import nvml

                self.assertFalse(nvml.is_available())
                for call in (
                    nvml.list_devices,
                    nvml.current_device_uuid,
                    lambda: nvml.resolve_index_by_name_fragment("5090"),
                    lambda: nvml.total_bytes_for_uuid("GPU-aaaa"),
                ):
                    with self.assertRaises(nvml.NvmlUnavailableError) as ctx:
                        call()
                    self.assertIn("pynvml", str(ctx.exception))
        finally:
            if saved is not None:
                sys.modules["pynvml"] = saved


if __name__ == "__main__":
    unittest.main()
