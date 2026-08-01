"""#391: the load-time expert stager is fed by MANY loader threads.

`deepseek_v4.load_weights` runs the weight loaders on a
`concurrent.futures.ThreadPoolExecutor`, and `should_async_load` is True for
every GGUF tensor (they are CPU tensors), so `StreamingExpertStager.submit` is
a CONCURRENT entry point -- gate and up of one layer both route into the same
`w13_qweight` stager from two threads.

Boot 10 of #391 (`/spinning/gpu-battery-results/2026-08-01_391_dsv4flash10`)
died on that three times out of three, always on TP1, always at the tail of the
stream, on both branches of `_place`:

    expert_offload.py:1398   spill[index].copy_(row)         <- spill half None
    expert_offload.py:1395   resident_buf[index].copy_(row)  <- resident half None
    TypeError: 'NoneType' object is not subscriptable

`_ensure_tiers` published `self._row_shape` -- the guard that says "the tiers
exist" -- BEFORE allocating them, unlocked. A second thread arriving inside
that window saw the guard, skipped the build and got a pair whose halves were
still None. `spill.pin_memory()` is what makes the window wide enough to hit:
it page-locks the whole cold tier.

Both falsifiers below carry the PRE-FIX code as a subclass and reproduce the
failure with an injected window, so a regression cannot pass silently: the
tests fail if the shipped class is ever reverted to that shape.

No GPU, no CUDA context: `torch.cuda.is_available()` is False here, so the
spill tier is plain host memory and the pinning branch is skipped exactly as it
is in the pull loop.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

from sglang.srt.layers.moe.expert_offload import (
    StreamingExpertStager,
    _nbytes,
    plan_load_time_staging,
    reset_streaming_staging_ledger,
    streaming_staging_ledger,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_EXPERTS = 8
SHARD_KEYS = ("w1", "w3")
SHARD_SHAPE = (2, 4)
ROW_SHAPE = (2 * SHARD_SHAPE[0], SHARD_SHAPE[1])


def _shard(expert_id: int, shard_id: str) -> torch.Tensor:
    """A distinguishable constant block, so a misplaced row is visible."""
    value = float(expert_id * 10 + SHARD_KEYS.index(shard_id))
    return torch.full(SHARD_SHAPE, value, dtype=torch.float32)


def _expected_row(expert_id: int) -> torch.Tensor:
    return torch.cat([_shard(expert_id, key) for key in SHARD_KEYS], dim=0)


def _plan():
    plan = plan_load_time_staging(NUM_EXPERTS, fraction=0.5)
    assert plan is not None and plan.spill_ids, "the fixture needs both tiers"
    return plan


def _allocator(plan, gate=None, hold=0.0):
    """`allocate` for the stager, optionally holding the build open.

    `gate` is set at the START of the allocation, i.e. at exactly the point the
    pre-fix code had already published its guard, and `hold` keeps the build
    unfinished for that long. That is the injected window: the other threads
    wait for `gate` and submit while the tiers do not exist yet.
    """

    def allocate(row_shape, dtype):
        if gate is not None:
            gate.set()
        if hold:
            time.sleep(hold)
        return torch.zeros((plan.buffer_slots,) + tuple(row_shape), dtype=dtype)

    return allocate


class _PreFixTierStager(StreamingExpertStager):
    """`_ensure_tiers` exactly as boot 10 ran it: guard published FIRST."""

    def _ensure_tiers(self, row):
        row_shape = tuple(int(d) for d in row.shape)
        if self._row_shape is None:
            self._row_shape = row_shape  # <- published before the tiers exist
            self._dtype = row.dtype
            buf = self._allocate(row_shape, row.dtype)
            self.resident_buf = buf
            if self.plan.spill_ids:
                self.spill = torch.empty(
                    (len(self.plan.spill_ids),) + row_shape,
                    dtype=row.dtype,
                    device="cpu",
                )
        return self.resident_buf, self.spill


class _PreFixClaimStager(StreamingExpertStager):
    """`submit` as it was: the shard set is completed and claimed unlocked.

    The completion test is lifted into a named local and two rendezvous points
    are inserted around it -- a no-op restructure that lets the test hold both
    threads at the two states the unlocked version allows: both shards in, and
    both threads having decided they were the last one. The shipped `submit`
    does insert, test and claim inside one locked block, so the second state is
    not reachable there at all.
    """

    hook = staticmethod(lambda phase: None)

    def submit(self, expert_id, shard_id, tensor):
        expert_id = int(expert_id)
        parts = self._pending.setdefault(expert_id, {})
        parts[shard_id] = tensor
        self._inflight[expert_id] = self._inflight.get(expert_id, 0) + _nbytes(tensor)
        self.hook("inserted")
        complete = len(parts) == len(self.shard_keys)
        self.hook("decided")
        if complete:
            del self._pending[expert_id]
            self._place(expert_id, parts)


def _drive_one_expert_per_thread(stager, gate, experts=range(NUM_EXPERTS)):
    """Thread 0 opens the build; every other thread submits inside the window."""
    experts = list(experts)
    errors = []

    def work(expert_id):
        try:
            if expert_id != experts[0]:
                if not gate.wait(timeout=30):
                    raise AssertionError("the build window never opened")
            for shard_id in SHARD_KEYS:
                stager.submit(expert_id, shard_id, _shard(expert_id, shard_id))
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=len(experts)) as pool:
        list(pool.map(work, experts))
    return errors


class TestStreamingStagerTierRace(unittest.TestCase):
    def setUp(self):
        reset_streaming_staging_ledger()

    def test_prefix_ensure_tiers_reproduces_the_boot_10_crash(self):
        """CAN-FAIL: the old order hands out a half-None pair, as on boot 10."""
        plan = _plan()
        gate = threading.Event()
        stager = _PreFixTierStager(
            plan,
            SHARD_KEYS,
            _allocator(plan, gate=gate, hold=0.5),
            label="pre-fix",
        )

        errors = _drive_one_expert_per_thread(stager, gate)

        self.assertTrue(errors, "the injected window produced no failure at all")
        self.assertTrue(
            all(isinstance(e, TypeError) for e in errors),
            f"expected the boot-10 error class, got {errors}",
        )
        self.assertIn("NoneType", str(errors[0]))
        self.assertIn("not subscriptable", str(errors[0]))

    def test_ensure_tiers_serializes_under_the_same_window(self):
        """The shipped class: every thread waits for the build, none crashes."""
        plan = _plan()
        gate = threading.Event()
        allocations = []
        allocate = _allocator(plan, gate=gate, hold=0.5)

        def counting_allocate(row_shape, dtype):
            allocations.append(tuple(row_shape))
            return allocate(row_shape, dtype)

        stager = StreamingExpertStager(
            plan, SHARD_KEYS, counting_allocate, label="fixed"
        )

        errors = _drive_one_expert_per_thread(stager, gate)

        self.assertEqual(errors, [])
        self.assertEqual(
            allocations,
            [ROW_SHAPE],
            "the tiers must be built exactly once, by exactly one thread",
        )
        resident_buf, spill = stager.finalize()
        self._assert_tiers_exact(plan, resident_buf, spill)

    def test_prefix_submit_claim_reproduces_the_double_completion(self):
        """CAN-FAIL: two threads carrying one expert both claim to be last."""
        plan = _plan()
        barriers = {
            phase: threading.Barrier(len(SHARD_KEYS), timeout=30)
            for phase in ("inserted", "decided")
        }
        stager = _PreFixClaimStager(
            plan, SHARD_KEYS, _allocator(plan), label="pre-fix-claim"
        )
        stager.hook = lambda phase: barriers[phase].wait()
        errors = []

        def work(shard_id):
            try:
                stager.submit(0, shard_id, _shard(0, shard_id))
            except BaseException as exc:  # noqa: BLE001 - the point of the test
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=len(SHARD_KEYS)) as pool:
            list(pool.map(work, SHARD_KEYS))

        self.assertTrue(
            errors, "both threads completed the same expert without complaint"
        )

    def test_submit_claims_one_expert_once_under_contention(self):
        """The shipped class: the same two-thread submission, 200 times over."""
        for _ in range(200):
            reset_streaming_staging_ledger()
            plan = _plan()
            stager = StreamingExpertStager(
                plan, SHARD_KEYS, _allocator(plan), label="fixed-claim"
            )
            barrier = threading.Barrier(len(SHARD_KEYS), timeout=30)
            errors = []

            def work(shard_id, stager=stager, barrier=barrier, errors=errors):
                try:
                    barrier.wait()
                    stager.submit(0, shard_id, _shard(0, shard_id))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=len(SHARD_KEYS)) as pool:
                list(pool.map(work, SHARD_KEYS))

            self.assertEqual(errors, [])
            self.assertEqual(stager._placed, {0})
            self.assertTrue(
                torch.equal(stager.resident_buf[0], _expected_row(0)),
                "the expert was not placed, or placed twice into a cleared dict",
            )

    def test_threaded_result_is_the_single_thread_result(self):
        """No-contention pin: same tiers, same ledger, byte for byte."""
        plan = _plan()

        reset_streaming_staging_ledger()
        serial = StreamingExpertStager(
            plan, SHARD_KEYS, _allocator(plan), label="serial"
        )
        for expert_id in range(NUM_EXPERTS):
            for shard_id in SHARD_KEYS:
                serial.submit(expert_id, shard_id, _shard(expert_id, shard_id))
        serial_buf, serial_spill = serial.finalize()
        serial_ledger = streaming_staging_ledger()
        serial_figures = (
            serial_ledger.streamed_bytes,
            serial_ledger.resident_bytes,
            serial_ledger.pinned_bytes,
            serial_ledger.inflight_bytes,
            serial_ledger.tensors,
        )
        self.assertEqual(serial_ledger.inflight_bytes, 0)

        reset_streaming_staging_ledger()
        gate = threading.Event()
        threaded = StreamingExpertStager(
            plan, SHARD_KEYS, _allocator(plan, gate=gate, hold=0.2), label="threaded"
        )
        self.assertEqual(_drive_one_expert_per_thread(threaded, gate), [])
        threaded_buf, threaded_spill = threaded.finalize()
        threaded_ledger = streaming_staging_ledger()

        self.assertTrue(torch.equal(serial_buf, threaded_buf))
        self.assertTrue(torch.equal(serial_spill, threaded_spill))
        self.assertEqual(
            serial_figures,
            (
                threaded_ledger.streamed_bytes,
                threaded_ledger.resident_bytes,
                threaded_ledger.pinned_bytes,
                threaded_ledger.inflight_bytes,
                threaded_ledger.tensors,
            ),
        )
        self._assert_tiers_exact(plan, threaded_buf, threaded_spill)

    def test_ledger_deltas_survive_many_threads(self):
        """`x += n` on a shared counter is a read-modify-write; record() locks."""
        reset_streaming_staging_ledger()
        ledger = streaming_staging_ledger()
        rounds = 500

        def work(_):
            for _ in range(rounds):
                ledger.record(streamed=1, inflight=1)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(8)))

        self.assertEqual(ledger.streamed_bytes, 8 * rounds)
        self.assertEqual(ledger.inflight_bytes, 8 * rounds)
        self.assertEqual(ledger.peak_inflight_bytes, 8 * rounds)

    # -- helpers ------------------------------------------------------------

    def _assert_tiers_exact(self, plan, resident_buf, spill):
        for slot, expert_id in enumerate(plan.resident_ids):
            self.assertTrue(
                torch.equal(resident_buf[slot], _expected_row(expert_id)),
                f"resident slot {slot} does not hold expert {expert_id}",
            )
        for row, expert_id in enumerate(plan.spill_ids):
            self.assertTrue(
                torch.equal(spill[row], _expected_row(expert_id)),
                f"spill row {row} does not hold expert {expert_id}",
            )


if __name__ == "__main__":
    unittest.main()
