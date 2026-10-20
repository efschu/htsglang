"""F3: the lend and its return never hold the decode on a device-wide sync, a host read or the map.

THE METAL (NF rc26, 20.10.): a lend or return held the scheduler thread 0.3-1.2 s -- two device-wide
``torch.cuda.synchronize()`` in ``_rows_off_live`` (behind HiCache / PLE / L3 streams that never touch a
row), 4-6 ``.cpu()`` table reads per layer x 48, and the lend's 600-960 cuMem* calls on the scheduler thread.
User order 20.10.: no synchronous waiting; the decode is never held for nothing.

THE ORDER that replaces the syncs (``d_row_fence``):
  E0 -- the current stream waits on the tail of every ROW STREAM (device-side);
  W  -- the OFF copies and table writes on the current stream (device ops, no host read);
  E1 -- after W; every row stream's later kernels wait on it;
  UNMAP only after E1 completed (``query``), or the host waits on E1 ALONE when the bytes are needed now.

What must hold (desk rank of test_pdflip_d_seat_live_wipe_s1_0928, fake fence ops):
(1) a return turns the rows OFF at once and releases the cells only after E1 -- never before, whatever polls;
(2) ``drain_releases(block=True)`` waits on E1 alone (no ``torch.cuda.synchronize`` anywhere);
(3) the lend's map runs on the executor; the rows stay OFF until it finished and ``finish_lend`` ran;
(4) every bank move settles an in-flight lend first (no move races the map thread's spans);
(5) the gate: async ``lend`` starts the map, ``poll`` lands it, an extend joins it; ``quiesce_for_sleep``
    lands everything; the bytes an extend needs go before it (``_bytes_now``: E1 only);
(6) the warm refill: rows reserved at once, routed only after the copy's event (commit in ``poll_warm``);
(7) rank unity: three ranks with different fence / map timings take the same verdicts, keep rank-local
    amounts, never enter a collective, and none releases a cell before its own E1.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_row_fence as F  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_t_f3_rb", os.path.join(_HERE, "test_pdflip_d_stage_grow_release_first_rb.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
h = rb.h

ROW = 4096
STEP = 4 * ROW * 2


class Ops:
    """Fence ops on a fake device: events complete only when the test says so."""

    def __init__(self):
        self.log, self.n, self.done = [], 0, set()

    def current_stream(self):
        return "cur"

    def event(self):
        self.n += 1
        return "e%d" % self.n

    def record(self, ev, stream):
        self.log.append(("record", ev, stream))

    def wait(self, stream, ev):
        self.log.append(("wait", stream, ev))

    def query(self, ev):
        return ev in self.done

    def synchronize(self, ev):
        self.log.append(("host-wait", ev))
        self.done.add(ev)

    def new_stream(self):
        return "warm"

    def stream_ctx(self, s):
        import contextlib

        return contextlib.nullcontext()

    def complete_all(self):
        self.done.update("e%d" % i for i in range(1, self.n + 1))


class _Fwd:
    def __repr__(self):
        return "fwd"


class _StrictFuture(concurrent.futures.Future):
    """A future whose result is never waited for unless the test's map thread ran (a wait would hang)."""

    def result(self, timeout=None):
        if not self.done():
            raise AssertionError("F3: the scheduler thread waited on a map that has not run")
        return super().result(timeout)


class ManualExecutor:
    """The map thread, stepped by the test."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        fut = _StrictFuture()
        self.jobs.append((fut, fn, args))
        return fut

    def run_all(self):
        for fut, fn, args in self.jobs:
            if not fut.done():
                fut.set_result(fn(*args))


@pytest.fixture
def ops(monkeypatch):
    F.reset_for_tests()
    o = Ops()
    F.set_ops_for_tests(o)
    F.register_row_stream(FWD)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("F3: a device-wide synchronize")))
    yield o
    F.reset_for_tests()


FWD = _Fwd()


def _floor(v="0"):
    return envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override(v)


def _bank_mapped(tms, r):
    return sum(tms.mapped(b.data_ptr()) for b in r.banks)


def _lent(tms):
    r, ctl = rb._live_s0(tms)
    assert ctl.lend(10 * STEP) == 8 and ctl.rows_on == 40
    return r, ctl


# ---- (1)-(2) the deferred unmap -----------------------------------------------------------------------------

def test_a_return_turns_rows_off_now_and_releases_cells_only_after_e1(ops):
    tms = h.CoreTms()
    with h._armed(tms), _floor():
        r, ctl = _lent(tms)
        mapped_lent = _bank_mapped(tms, r)
        assert ctl.unlend() == 8
        assert ctl.rows_on == 32 and all(int(c._pool_tables.seat_on) == 32 for c in r.caches)  # OFF now
        assert len(ctl.pending_releases) == 1 and _bank_mapped(tms, r) == mapped_lent      # cells kept
        for _ in range(3):
            ctl.drain_releases(block=False)                                                # E1 not done
        assert _bank_mapped(tms, r) == mapped_lent
        e1 = ctl.pending_releases[0].fence.e1
        ops.done.add(e1)
        ctl.drain_releases(block=False)
        assert not ctl.pending_releases
        m = ctl.row_tensors[0]
        assert tms.mapped(m.ptr) == dsv.span_bytes(ctl.bank_spans(m, 32))
    # the order on the device: E0 behind the forward stream, then (OFF writes), E1, the forward waits on E1
    kinds = [x for x in ops.log if x[0] in ("record", "wait")]
    assert kinds[:2] == [("record", "e1", FWD), ("wait", "cur", "e1")]
    assert ("record", e1, "cur") in kinds and ("wait", FWD, e1) in kinds


def test_bytes_needed_now_wait_on_e1_alone(ops):
    tms = h.CoreTms()
    with h._armed(tms), _floor():
        r, ctl = _lent(tms)
        ctl.unlend_to(36)
        e1 = ctl.pending_releases[0].fence.e1
        ctl.drain_releases(block=True)
        assert ("host-wait", e1) in ops.log and not ctl.pending_releases
        assert [x for x in ops.log if x[0] == "host-wait"] == [("host-wait", e1)]


def test_a_stage_shrink_releases_behind_e1_before_the_kv_maps(ops):
    tms = rb.CardTms()
    with h._armed(tms), _floor():
        r, ctl = rb._live_s0(tms)
        tms.budget = tms.total()
        seen = []
        real = ctl._set_bank_plans

        def _spy(plans, census):
            seen.append([x for x in ops.log if x[0] == "host-wait"])
            return real(plans, census)

        with mock.patch.object(ctl, "_set_bank_plans", _spy):
            ctl.apply_stage(2, 1)  # 32 -> 16 rows (bank-first), the KV maps into the cells
        assert seen and seen[0], "the cells went before the host waited on E1"
        assert tms.total() <= tms.budget


# ---- (3)-(4) the lend's map on the executor -----------------------------------------------------------------

def test_the_rows_go_on_only_after_the_map_finished(ops):
    tms = h.CoreTms()
    ex = ManualExecutor()
    with h._armed(tms), _floor():
        r, ctl = rb._live_s0(tms)
        mapped0 = _bank_mapped(tms, r)
        assert ctl.start_lend(10 * STEP, executor=ex) == 8
        assert ctl.lend_in_flight() and ctl.rows_on == 32
        assert all(int(c._pool_tables.seat_on) == 32 for c in r.caches)
        assert ctl.finish_lend(block=False) is None and _bank_mapped(tms, r) == mapped0  # map not run
        ex.run_all()                                                                    # the map thread
        assert all(int(c._pool_tables.seat_on) == 32 for c in r.caches)                 # still OFF
        done = ctl.finish_lend(block=False)
        assert done["rows"] == 8 and ctl.rows_on == 40 and ctl.lent_from == 32
        assert all(int(c._pool_tables.seat_on) == 40 for c in r.caches)


@pytest.mark.parametrize("move", ["unlend", "apply_stage", "reseat_live", "grow_room", "lend"])
def test_every_bank_move_settles_an_in_flight_lend_first(ops, move):
    tms = rb.CardTms()
    ex = ManualExecutor()
    with h._armed(tms), _floor(), envs.FLLIPER_PDFLIP_D_SEAT_REWAKE.override(True):
        r, ctl = rb._live_s0(tms)
        ctl.start_lend(10 * STEP, executor=ex)

        def _finish(block):
            ex.run_all()  # the join: the thread's map completes
            return dsv.SeatVram.finish_lend(ctl, block)

        with mock.patch.object(ctl, "finish_lend", _finish):
            if move == "unlend":
                ctl.unlend()
            elif move == "apply_stage":
                ctl.apply_stage(2, 1)
            elif move == "reseat_live":
                ctl.reseat_live(1, 0)
            elif move == "grow_room":
                ctl.grow_room(2, 1)
            else:
                ctl.lend(STEP)
        assert not ctl.lend_in_flight()


# ---- (5) the gate --------------------------------------------------------------------------------------------

def _decode():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    return types.SimpleNamespace(forward_mode=ForwardMode.DECODE)


def _extend():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    return types.SimpleNamespace(forward_mode=ForwardMode.EXTEND, extend_num_tokens=10 ** 6)


def test_the_gate_starts_the_map_poll_lands_it_an_extend_joins(ops, monkeypatch):
    tms = h.CoreTms()
    ex = ManualExecutor()
    monkeypatch.setattr(dtl, "ASYNC_LEND", True)
    monkeypatch.setattr(dsv, "_lend_executor", lambda: ex)
    with h._armed(tms), _floor(), envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(1), \
            mock.patch.object(dtl, "_card_free_bytes", lambda: 10 * STEP):
        r, ctl = rb._live_s0(tms)
        r.sched.waiting_queue, r.sched.chunked_req = [], None
        assert dtl.on_batch(r.sched, _decode()) == "lend" and ctl.lend_in_flight() and ctl.rows_on == 32
        assert dtl.on_batch(r.sched, _decode()) is None             # one lend in flight
        dtl.round_start(r.sched)                                       # poll: the map is not done
        assert ctl.lend_in_flight() and ctl.rows_on == 32
        ex.run_all()
        dtl.round_start(r.sched)                                       # poll lands it
        assert not ctl.lend_in_flight() and ctl.rows_on == 40
        # a second lend in flight when an extend comes: the gate joins it, then returns behind E1
        ctl.unlend()
        ops.complete_all()
        ctl.drain_releases(block=False)
        assert dtl.on_batch(r.sched, _decode()) == "lend"
        ex.run_all()
        out = dtl.on_batch(r.sched, _extend())
        assert out == "return" and not ctl.lend_in_flight() and ctl.rows_on == 32
        assert not ctl.pending_releases                                # the extend's bytes went (E1)


def test_quiesce_for_sleep_lands_everything(ops, monkeypatch):
    tms = h.CoreTms()
    ex = ManualExecutor()
    with h._armed(tms), _floor():
        r, ctl = _lent(tms)
        ctl.unlend_to(36)
        ctl.start_lend(10 * STEP, executor=ex)

        def _finish(block):
            ex.run_all()
            return dsv.SeatVram.finish_lend(ctl, block)

        with mock.patch.object(ctl, "finish_lend", _finish):
            dtl.quiesce_for_sleep(r.sched)
        assert not ctl.lend_in_flight() and not ctl.pending_releases


# ---- (6) the warm refill -----------------------------------------------------------------------------------------

def test_warm_rows_are_routed_only_after_their_copy_landed(ops):
    from flliper.srt.layers.moe import expert_pool_device as epd

    tms = h.CoreTms()
    with h._armed(tms), _floor():
        r, ctl = rb._live_s0(tms)
        cache = r.caches[0]
        t = cache._pool_tables
        store = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) + 100.0
        bank = torch.zeros(int(t.row_key.numel()), 4)
        cache._pool_srcs, cache._pool_dsts = [store], [bank]
        rows = torch.arange(int(t.seat_base), int(t.seat_base) + 2)
        experts = torch.tensor([5, 6])
        epd.reserve_warm_rows(t, rows)
        job = F.submit_warm(cache, experts, rows, torch.tensor([True, True]), torch.tensor([0, 1]))
        assert job is not None and int(t.row_key[rows[0]]) == epd.SEAT_OFF_KEY   # reserved: never routed
        assert torch.equal(bank[int(rows[0])], store[0])                         # the bytes on the warm stream
        assert F.poll_warm() == 0 and int(t.row_key[rows[0]]) == epd.SEAT_OFF_KEY # copy not landed: reserved
        ops.done.add(job.ev)
        assert F.poll_warm() == 1
        assert int(t.row_key[rows[0]]) == 5 and int(t.hot_phys[5]) == int(rows[0])
        assert int(t.row_key[rows[1]]) == 6


def test_a_shrink_drops_the_pending_warm_rows_of_its_layer(ops):
    """A rows-OFF move inside its fence (E0 waited on the warm stream: the copy landed) releases the layer's
    pending warm rows -- a later commit never writes an expert into an OFF row."""
    from flliper.srt.layers.moe import expert_pool_device as epd

    tms = h.CoreTms()
    with h._armed(tms), _floor():
        r, ctl = rb._live_s0(tms)
        cache = r.caches[0]
        t = cache._pool_tables
        cache._pool_srcs = [torch.ones(8, 4)]
        cache._pool_dsts = [torch.zeros(int(t.row_key.numel()), 4)]
        top = int(t.seat_base) + int(t.seat_on)
        rows = torch.arange(top - 2, top)                     # the two highest ON rows
        epd.reserve_warm_rows(t, rows)
        job = F.submit_warm(cache, torch.tensor([5, 6]), rows, torch.tensor([True, True]), torch.tensor([0, 1]))
        assert job is not None
        k_new = int(t.seat_on) - 1                             # one row stays ON, one goes OFF
        cache.set_seat_rows_on(k_new, device_write=True)
        ops.done.add(job.ev)
        assert F.poll_warm() == 0                              # the job is gone
        assert int(t.row_key[top - 2]) == -1                   # kept row: free, not reserved
        assert int(t.row_key[top - 1]) == epd.SEAT_OFF_KEY     # OFF row: OFF, never an expert
        assert int(t.hot_phys[5]) < 0 and int(t.hot_phys[6]) < 0


# ---- (7) rank unity ------------------------------------------------------------------------------------------------

def test_three_ranks_same_verdicts_rank_local_timing_no_collective_no_early_unmap(monkeypatch):
    import torch.distributed as tdist

    def _boom(*a, **k):
        raise AssertionError("F3: a collective in the lend path")

    for name in ("all_reduce", "all_gather", "broadcast", "barrier"):
        monkeypatch.setattr(tdist, name, _boom)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("F3: a device-wide synchronize")))
    results = {}
    for rank, late in ((0, 0), (1, 2), (2, 5)):           # E1 of each return completes `late` polls later
        F.reset_for_tests()
        o = Ops()
        F.set_ops_for_tests(o)
        F.register_row_stream(FWD)
        tms = h.CoreTms()
        ex = ManualExecutor()
        early = []
        with h._armed(tms), _floor():
            r, ctl = rb._live_s0(tms)
            real = ctl._set_bank_plans

            def _guard(plans, census, _ctl=ctl, _o=o, _real=real):
                for pr in _ctl.pending_releases[:1]:
                    if pr.fence.e1 not in _o.done:
                        early.append(pr.fence.e1)
                return _real(plans, census)

            ctl._set_bank_plans = _guard
            verdicts = []
            budget = (STEP, 7 * 2 * ROW, 2 * STEP)[rank]     # 4 / 7 / 8 rows: the amount is the card's
            for step in range(3):
                verdicts.append(ctl.start_lend(budget, executor=ex) > 0)
                ex.run_all()
                ctl.finish_lend(block=False)
                ctl.unlend()
                for _ in range(late):
                    ctl.drain_releases(block=False)
                o.complete_all()
                ctl.drain_releases(block=False)
            results[rank] = (verdicts, ctl.rows_on, ctl.lent_rows)
        F.reset_for_tests()
        assert not early, early
    assert {tuple(v[0]) for v in results.values()} == {(True, True, True)}   # one verdict sequence
    assert {v[1] for v in results.values()} == {32}
    assert len({v[2] for v in results.values()}) == 3                       # amounts rank-local
