"""#396(a): on-demand cold-expert materialization -- hermetic falsifiers.

No GPU, no model, no checkpoint: the "checkpoint" is a temporary file whose
bytes are generated here, so every claim below is about the staging contract
itself rather than about one model's layout.

The four claims, and the observation that would break each:

1. With the gate ON and refs supplied, the load path reads ZERO cold-expert
   bytes. Observed through a ``source`` that records every call: a single call
   for a cold expert falsifies it.
2. First touch materializes, and under N concurrent first touches of the same
   expert the pool performs EXACTLY ONE read. Observed through the pool's own
   ``disk_reads`` counter plus a barrier that makes the threads collide.
3. With the gate OFF the staging order is the one it always was. Pinned by
   recording the eager order FIRST and asserting the gate-off run with refs
   supplied reproduces it call for call.
4. A vanished file is loud. Observed as ``LazyExpertUnavailable``, never a
   zero row.
"""

import os
import tempfile
import threading

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertStagingPlan,
    stage_experts_into_tiers,
)
from sglang.srt.layers.moe.lazy_expert_staging import (  # noqa: E402
    ExpertFileRef,
    LazyExpertUnavailable,
    LazySpillPool,
    expert_refs_from_expert_major_tensor,
    lazy_expert_staging_enabled,
)

E = 6
R = 2
ROW = (4, 3)
DTYPE = torch.float32
ROW_BYTES = 4 * 3 * 4


def _plan():
    return ExpertStagingPlan(
        num_experts=E,
        resident_count=R,
        buffer_slots=R + 1,
        resident_ids=tuple(range(R)),
        spill_ids=tuple(range(R, E)),
    )


def _expert_tensor(expert_id: int) -> torch.Tensor:
    """Deterministic, distinguishable content: expert i is all (i + 1)."""
    return torch.full(ROW, float(expert_id + 1), dtype=DTYPE)


def _checkpoint(tmpdir: str, num_experts: int = E, pad: int = 0) -> str:
    """One expert-major blob: ``pad`` header bytes then E rows back to back."""
    path = os.path.join(tmpdir, "experts.bin")
    with open(path, "wb") as fh:
        fh.write(b"\x00" * pad)
        for e in range(num_experts):
            fh.write(_expert_tensor(e).numpy().tobytes())
    return path


class _RecordingSource:
    """``source(expert_id)`` that remembers the order it was asked in."""

    def __init__(self):
        self.calls = []

    def __call__(self, expert_id):
        self.calls.append(int(expert_id))
        return _expert_tensor(expert_id)


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERT_LAZY_STAGING", "0")
    yield


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERT_LAZY_STAGING", "1")
    yield


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_EXPERT_LAZY_STAGING", raising=False)
    assert lazy_expert_staging_enabled() is False


def test_gate_reads_the_environment(gate_on):
    assert lazy_expert_staging_enabled() is True


# ---------------------------------------------------------------------------
# ref derivation
# ---------------------------------------------------------------------------


def test_refs_from_expert_major_tensor_are_contiguous_and_exact():
    refs = expert_refs_from_expert_major_tensor(
        "/nonexistent", 1024, ROW_BYTES * E, E, ROW, DTYPE
    )
    assert len(refs) == E
    assert [refs[e].offset for e in range(E)] == [
        1024 + e * ROW_BYTES for e in range(E)
    ]
    assert {refs[e].nbytes for e in range(E)} == {ROW_BYTES}


def test_refs_refuse_a_tensor_whose_expert_stride_is_not_exact():
    with pytest.raises(ValueError, match="expert stride is not exact"):
        expert_refs_from_expert_major_tensor(
            "/nonexistent", 0, ROW_BYTES * E + 1, E, ROW, DTYPE
        )


def test_ref_reads_back_the_bytes_it_describes():
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp, pad=17)
        refs = expert_refs_from_expert_major_tensor(
            path, 17, ROW_BYTES * E, E, ROW, DTYPE
        )
        for e in range(E):
            row = torch.zeros(ROW, dtype=DTYPE)
            refs[e].read_into(row)
            assert torch.equal(row, _expert_tensor(e))


# ---------------------------------------------------------------------------
# falsifier 1 + 3: what the LOAD path reads, with the gate on and off
# ---------------------------------------------------------------------------


def _stage(plan, source, lazy_refs=None):
    out = torch.zeros((plan.buffer_slots,) + ROW, dtype=DTYPE)
    spill = stage_experts_into_tiers(plan, source, out, lazy_refs=lazy_refs)
    return out, spill


def test_gate_on_reads_zero_cold_expert_bytes_at_load(gate_on):
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        refs = expert_refs_from_expert_major_tensor(
            path, 0, ROW_BYTES * E, E, ROW, DTYPE
        )
        plan = _plan()
        src = _RecordingSource()
        out, spill = _stage(plan, src, lazy_refs=lambda e: refs[e])

        # The resident half is staged as it always was; the cold half is not
        # asked for at all. That asymmetry IS the feature.
        assert src.calls == list(plan.resident_ids)
        assert isinstance(spill, LazySpillPool)
        assert spill.disk_reads == 0
        assert spill.materialized_rows() == ()
        for slot, e in enumerate(plan.resident_ids):
            assert torch.equal(out[slot], _expert_tensor(e))


def test_gate_off_reproduces_the_eager_staging_order_call_for_call(gate_off):
    """Pin today's behavior, then prove the flag-off run is that behavior."""
    plan = _plan()

    baseline = _RecordingSource()
    out_ref, spill_ref = _stage(plan, baseline)
    pinned_order = list(baseline.calls)
    # The order the pull loop has always used: residents in slot order, then
    # the cold pool in row order.
    assert pinned_order == list(plan.resident_ids) + list(plan.spill_ids)

    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        refs = expert_refs_from_expert_major_tensor(
            path, 0, ROW_BYTES * E, E, ROW, DTYPE
        )
        gated = _RecordingSource()
        out_gated, spill_gated = _stage(plan, gated, lazy_refs=lambda e: refs[e])

    # Supplying refs while the gate is off must change nothing: same call
    # order, same device buffer, same cold pool, and a real tensor rather than
    # the lazy proxy.
    assert gated.calls == pinned_order
    assert not isinstance(spill_gated, LazySpillPool)
    assert torch.equal(out_gated, out_ref)
    assert torch.equal(spill_gated, spill_ref)


def test_lazy_pool_holds_the_same_bytes_the_eager_pool_would(gate_on):
    plan = _plan()
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        refs = expert_refs_from_expert_major_tensor(
            path, 0, ROW_BYTES * E, E, ROW, DTYPE
        )
        _, lazy = _stage(plan, _RecordingSource(), lazy_refs=lambda e: refs[e])
        for row, e in enumerate(plan.spill_ids):
            assert torch.equal(lazy[row], _expert_tensor(e))
        assert lazy.disk_reads == len(plan.spill_ids)


# ---------------------------------------------------------------------------
# falsifier 2: exactly one read per expert under concurrent first touches
# ---------------------------------------------------------------------------


def _lazy_pool(path, spill_ids=(2, 3, 4, 5)):
    refs = expert_refs_from_expert_major_tensor(path, 0, ROW_BYTES * E, E, ROW, DTYPE)
    storage = torch.zeros((len(spill_ids),) + ROW, dtype=DTYPE)
    return LazySpillPool(storage, spill_ids, refs)


def test_concurrent_first_touches_read_each_expert_exactly_once():
    threads_per_row = 12
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        pool = _lazy_pool(path)
        rows = list(range(len(pool)))
        start = threading.Barrier(threads_per_row * len(rows))
        seen = {}
        errors = []
        lock = threading.Lock()

        def touch(row):
            try:
                start.wait(timeout=30)
                got = pool[row].clone()
                with lock:
                    seen.setdefault(row, []).append(got)
            except BaseException as exc:  # noqa: BLE001 - reported, not hidden
                with lock:
                    errors.append(exc)

        workers = [
            threading.Thread(target=touch, args=(row,))
            for row in rows
            for _ in range(threads_per_row)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=30)

        assert not errors, errors
        # One read per ROW, not one per TOUCH: 48 touches, 4 reads.
        assert pool.disk_reads == len(rows)
        for row in rows:
            expected = _expert_tensor(pool.spill_ids[row])
            assert len(seen[row]) == threads_per_row
            for got in seen[row]:
                # Every waiter observes the FINISHED row, never a torn one.
                assert torch.equal(got, expected)


def test_repeat_touches_do_not_re_read():
    with tempfile.TemporaryDirectory() as tmp:
        pool = _lazy_pool(_checkpoint(tmp))
        for _ in range(5):
            _ = pool[0]
        assert pool.disk_reads == 1
        assert pool.materialized_rows() == (0,)


# ---------------------------------------------------------------------------
# falsifier 4: loud on absence, never a zero row
# ---------------------------------------------------------------------------


def test_vanished_file_raises_instead_of_yielding_zeros():
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        pool = _lazy_pool(path)
        os.unlink(path)
        with pytest.raises(LazyExpertUnavailable, match="could not be opened"):
            _ = pool[0]
        # And the row is still not claimed as materialized.
        assert pool.materialized_rows() == ()


def test_truncated_file_raises_instead_of_a_short_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        pool = _lazy_pool(path)
        with open(path, "r+b") as fh:
            fh.truncate(ROW_BYTES * 3)  # experts 3..5 no longer backed
        with pytest.raises(LazyExpertUnavailable, match="short"):
            _ = pool[len(pool) - 1]


def test_failure_is_re_raised_to_every_waiter_not_only_the_first():
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        pool = _lazy_pool(path)
        os.unlink(path)
        with pytest.raises(LazyExpertUnavailable):
            _ = pool[0]
        with pytest.raises(LazyExpertUnavailable):
            _ = pool[0]


def test_shape_mismatch_is_named():
    ref = ExpertFileRef("/nonexistent", 0, ROW_BYTES, (2, 6), DTYPE)
    with pytest.raises(LazyExpertUnavailable, match="does not match"):
        ref.read_into(torch.zeros(ROW, dtype=DTYPE))


# ---------------------------------------------------------------------------
# transparency: the consumers' surface answers without materializing
# ---------------------------------------------------------------------------


def test_accounting_surface_answers_without_reading_anything():
    with tempfile.TemporaryDirectory() as tmp:
        pool = _lazy_pool(_checkpoint(tmp))
        # These are exactly the calls register_load_time_presplit and the
        # offload cache's freed-bytes tally make on the cold pool.
        assert pool.numel() * pool.element_size() == 4 * ROW_BYTES
        assert tuple(pool.shape) == (4,) + ROW
        assert pool.dtype is DTYPE
        assert pool.is_pinned() is False
        assert len(pool) == 4
        assert pool.disk_reads == 0


def test_the_cuda_graph_uva_view_refuses_a_lazy_pool_by_name():
    """The one consumer that would bypass the accessor is refused, not served.

    ``device_view_of_pinned`` exists to bake the pool's ADDRESS into a CUDA
    graph. A captured graph then reads that memory directly, so a lazy row
    nobody touched before capture would be read as allocation garbage at full
    speed with no error. Refusing the combination is the only safe answer, and
    this is the falsifier for it: if the refusal is ever removed, this test
    stops raising and the silent-garbage path is live again.
    """
    from sglang.srt.layers.moe.expert_offload import device_view_of_pinned

    with tempfile.TemporaryDirectory() as tmp:
        pool = _lazy_pool(_checkpoint(tmp))
        with pytest.raises(RuntimeError, match="bypass the materialize"):
            device_view_of_pinned(pool)


def test_the_pool_does_not_expose_an_address_taking_surface():
    """No ``data_ptr`` / ``is_contiguous``: a raw-address consumer must fail
    loudly at wiring time rather than read unmaterialized rows."""
    with tempfile.TemporaryDirectory() as tmp:
        pool = _lazy_pool(_checkpoint(tmp))
        assert not hasattr(pool, "data_ptr")
        assert not hasattr(pool, "is_contiguous")
        assert pool.disk_reads == 0


def test_missing_ref_is_a_load_time_error_not_a_first_token_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = _checkpoint(tmp)
        refs = expert_refs_from_expert_major_tensor(
            path, 0, ROW_BYTES * E, E, ROW, DTYPE
        )
        del refs[4]
        storage = torch.zeros((4,) + ROW, dtype=DTYPE)
        with pytest.raises(ValueError, match="no file ref for cold experts"):
            LazySpillPool(storage, (2, 3, 4, 5), refs)
