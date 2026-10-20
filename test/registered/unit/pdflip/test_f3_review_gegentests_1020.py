"""F3 review (20.10.): counter-tests to d623c89b86 (desk/nf-f3-lend-nosync-1020).

The builder's suite leaves several danger directions unguarded (measured with 15 mutants at the ordering
points: apply() / reseat_live() unmapping before E1, and every row-stream registration, survive it). The
tests below close those gaps (GREEN on d623c89b86) and pin the review's findings (RED on d623c89b86):

GREEN -- coverage the builder's tests lack:
  G1  ``apply()`` shrink: cells released only after the host waited on E1;
  G2  ``reseat_live()`` grow (bank gives rows back before the GDN maps): same;
  G3  every bank-row stream the code creates joins the ROW-FENCE's set (H31/rearm/resume-warm ops, the pool
      prefetch stream, the warm stream, the scheduler's forward stream via ``poll``);
  G4  device shrink == host shrink on the edge tables (empty drop range, all kept rows hot, k=0, ties,
      no mover).
RED -- findings (each names its Auflage in the report nf-f3-review-1020.md):
  R1  the DECOUPLE spill lane's ``spill_stream`` runs a full forward (it reads bank rows) and is not a row
      stream (Auflage A1);
  R2  a failed lend map that lands in ``poll`` is swallowed at DEBUG (Auflage A2);
  R3  a sleep whose ``settle`` raises leaves warm jobs that ``poll_warm`` commits after the wake's rearm
      (Auflage A3);
  R4  the device mover copies every one of the m lanes (no-op lanes copy a row onto itself) and books them
      as moved bytes -- the host path booked only the moves (Auflage A5).
Serial, CPU only (CUDA_VISIBLE_DEVICES=""), MemoryMax=2G.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import random
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("_t_f3_builder", os.path.join(_HERE, "test_f3_lend_no_sync_1020.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

F, dsv, dtl, rb, h = B.F, B.dsv, B.dtl, B.rb, B.h
ops = B.ops  # the builder's fixture (fake fence ops, device-wide synchronize raises)


# ---- G1 / G2: apply() and reseat_live() unmap only behind E1 -------------------------------------------------

def test_g1_apply_shrink_releases_cells_only_after_e1(ops):
    tms = rb.CardTms()
    with h._armed(tms), B._floor():
        r, ctl = rb._live_s0(tms)
        assert ctl.rows_on == 32
        released = []
        real_set = ctl.set_plan

        def _set_plan(ptr, name, spans, *, live, census=None):
            if any(int(ptr) == m.ptr for m in ctl.row_tensors) and live:
                prev = ctl.spans_by_ptr.get(int(ptr))
                if prev is not None and dsv.span_bytes(spans) < dsv.span_bytes(prev):
                    released.append([x for x in ops.log if x[0] == "host-wait"])
            return real_set(ptr, name, spans, live=live, census=census)

        with mock.patch.object(ctl, "set_plan", _set_plan):
            ctl.apply(2)  # the cap form: k = 0, the live bank shrinks 32 -> 0
        assert ctl.rows_on == 0
        assert released, "apply(2) released no bank cell -- the test does not exercise the shrink"
        assert all(w for w in released), "a bank cell went before the host waited on E1"


def test_g2_reseat_grow_releases_bank_cells_only_after_e1(ops):
    tms = rb.CardTms()
    with h._armed(tms), B._floor(), dsv_rewake_on():
        r, ctl = rb._live_s0(tms)
        ctl.reseat_live(1, 0)                 # 32 -> 39 rows (GDN shrinks)
        assert ctl.rows_on == 39
        ops.complete_all()
        n_waits = len([x for x in ops.log if x[0] == "host-wait"])
        released = []
        real_set = ctl.set_plan

        def _set_plan(ptr, name, spans, *, live, census=None):
            if any(int(ptr) == m.ptr for m in ctl.row_tensors) and live:
                prev = ctl.spans_by_ptr.get(int(ptr))
                if prev is not None and dsv.span_bytes(spans) < dsv.span_bytes(prev):
                    released.append(len([x for x in ops.log if x[0] == "host-wait"]))
            return real_set(ptr, name, spans, live=live, census=census)

        ops.done.clear()  # the next fence's E1 is NOT done: only a host wait completes it
        with mock.patch.object(ctl, "set_plan", _set_plan):
            ctl.reseat_live(2, 0)             # GROW of the GDN slots: the bank gives 39 -> 32 back first
        assert ctl.rows_on == 32
        assert released, "reseat_live(2) released no bank cell -- the test does not exercise the shrink"
        assert all(n > n_waits for n in released), "a bank cell went before the host waited on E1"


def dsv_rewake_on():
    from flliper.srt.environ import envs

    return envs.FLLIPER_PDFLIP_D_SEAT_REWAKE.override(True)


# ---- G3: every row stream is registered -----------------------------------------------------------------------

class _FakeStream:
    _n = 0

    def __init__(self, *a, **k):
        _FakeStream._n += 1
        self.id = _FakeStream._n

    def __eq__(self, o):
        return isinstance(o, _FakeStream) and o.id == self.id

    def __hash__(self):
        return hash(("fake", self.id))


def _registered(s):
    return any(x is s for x in F.row_streams())


def test_g3_h31_ops_streams_are_row_streams(monkeypatch):
    from flliper.srt.layers.moe import expert_offload as eo

    F.reset_for_tests()
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    s = eo._CudaStreamOps().new_stream()
    assert _registered(s)
    F.reset_for_tests()


def test_g3_pool_prefetch_stream_is_a_row_stream(monkeypatch):
    from flliper.srt.layers.moe import expert_offload as eo

    F.reset_for_tests()
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(eo, "_POOL_PREFETCH_STREAM", None)
    s = eo.pool_prefetch_stream()
    assert _registered(s)
    F.reset_for_tests()


def test_g3_warm_stream_is_a_row_stream():
    F.reset_for_tests()

    class _O(B.Ops):
        def new_stream(self):
            return _FakeStream()

    F.set_ops_for_tests(_O())
    s = F.warm_stream()
    assert _registered(s)
    F.reset_for_tests()


def test_g3_poll_registers_the_forward_stream():
    F.reset_for_tests()
    fwd = _FakeStream()
    sched = types.SimpleNamespace(forward_stream=fwd)
    with mock.patch.object(dtl, "_controller", lambda s: None):
        dtl.poll(sched)
    assert _registered(fwd)
    F.reset_for_tests()


def test_r1_the_decouple_spill_stream_is_a_row_stream():
    """RED (Auflage A1): FLLIPER_KVSO_DECOUPLE leases ``spill_stream`` for a SECOND forward per iteration
    (scheduler.py, _dispatch_concurrent_spill) -- it reads every bank row the MoE layers route to. It is not
    in the ROW-FENCE's set: an unmap after E1 could run under a spill forward. (Off in the NF profile today;
    the fence must not depend on that.)"""
    F.reset_for_tests()
    fwd, spill = _FakeStream(), _FakeStream()
    sched = types.SimpleNamespace(forward_stream=fwd, spill_stream=spill)
    with mock.patch.object(dtl, "_controller", lambda s: None):
        dtl.poll(sched)
    try:
        assert _registered(spill), "spill_stream (a forward stream) is not a row stream"
    finally:
        F.reset_for_tests()


# ---- G4: device shrink == host shrink on edge tables ----------------------------------------------------------

from flliper.srt.layers.moe import expert_pool_device as epd  # noqa: E402

E, R, S, C, X, W = 64, 4, 3, 20, 24, 3


def _base(k_old):
    t = epd.allocate_pool_tables("cpu", E, R + C + X, R + S, S, {e: e for e in range(R)},
                                 [-1] * R + list(range(E - R)), seat_rows=X)
    epd.set_seat_rows_on(t, k_old, device_write=True)
    return t


def _fill(t, rows, uses, experts):
    for r, u, e in zip(rows, uses, experts):
        t.row_key[r] = e
        t.hot_phys[e] = r
        t.row_use[r] = u


def _equal_paths(t, k, mover=True):
    import copy

    th = t
    td = copy.copy(t)
    for f in ("row_key", "row_use", "hot_phys", "pf_row", "clock"):
        setattr(td, f, getattr(t, f).clone())
    bank = torch.arange(int(t.row_key.numel()) * W, dtype=torch.float32).reshape(-1, W)
    bh, bd = bank.clone(), bank.clone()
    dep = []

    def hm(moves):
        for s, d in moves:
            bh[d] = bh[s]

    def dm(src, dst):
        bd.index_copy_(0, dst, bd.index_select(0, src))

    snaps = []
    epd.set_seat_rows_on(th, k, device_write=True, move_rows=hm if mover else None, departed_out=dep)
    with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")), \
            mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")), \
            mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")):
        epd.seat_rows_off_device(td, k, move_rows=dm if mover else None, recall=lambda *a: snaps.append(a))
    for f in ("row_key", "row_use", "hot_phys", "pf_row"):
        assert torch.equal(getattr(td, f), getattr(th, f)), f
    assert torch.equal(bd, bh)
    kb, ub, ha = snaps[0]
    assert epd.departed_experts(kb.tolist(), ub.tolist(), ha.tolist(), num_experts=E) == dep
    assert epd.bijection_breaks(td) == 0


def _rows(t, k_old):
    lru, base = int(t.lru_start), int(t.seat_base)
    return list(range(lru, base + k_old))


def test_g4_empty_drop_range():
    t = _base(10)
    rows = _rows(t, 10)[:-10]  # only the kept LRU rows hold experts; the seat rows are free
    _fill(t, rows, [5 + i for i in range(len(rows))], list(range(R, R + len(rows))))
    t.clock[0] = 1000
    _equal_paths(t, 2)


def test_g4_all_kept_rows_hot_no_move():
    t = _base(12)
    rows = _rows(t, 12)
    _fill(t, rows, [900 + (i % 7) for i in range(len(rows))], list(range(R, R + len(rows))))
    t.clock[0] = 500  # every kept row was selected in the running step (use >= clock): no destination
    _equal_paths(t, 4)


def test_g4_k_zero_all_rows_go():
    t = _base(X)
    rows = _rows(t, X)
    rng = random.Random(5)
    ex = list(range(R, E))
    rng.shuffle(ex)
    _fill(t, rows[::2], [rng.randrange(1, 50) for _ in rows[::2]], ex)
    t.clock[0] = 2000
    _equal_paths(t, 0)


def test_g4_ties_keep_the_host_order():
    t = _base(16)
    rows = _rows(t, 16)
    _fill(t, rows[1:], [7] * (len(rows) - 1), list(range(R, R + len(rows) - 1)))  # all equal uses
    t.clock[0] = 8
    _equal_paths(t, 6)


def test_g4_no_mover():
    t = _base(14)
    rows = _rows(t, 14)
    _fill(t, rows[3:], list(range(1, len(rows) - 2)), list(range(R, R + len(rows) - 3)))
    t.clock[0] = 3000
    _equal_paths(t, 5, mover=False)


# ---- R2: a failed map in poll is not silent -------------------------------------------------------------------

def test_r2_a_failed_map_landing_in_poll_is_not_swallowed_at_debug(ops, caplog):
    """RED (Auflage A2): before F3 a refused lend map (``PdFlipDSeatVramRefused``, rc != OOM) stopped the
    scheduler by name. F3 lands the map in ``poll`` inside ``except Exception: logger.debug(...)`` -- the
    refusal vanishes, the partially mapped cells stay mapped, nothing is named."""
    caplog.set_level(logging.DEBUG)
    tms = h.CoreTms()
    ex = B.ManualExecutor()
    with h._armed(tms), B._floor():
        r, ctl = rb._live_s0(tms)
        ctl.start_lend(10 * B.STEP, executor=ex)

        for fut, fn, args in ex.jobs:
            fut.set_exception(dsv.PdFlipDSeatVramRefused("D-SEAT-VRAM: tms_set_spans(bank, now=True) rc=-1"))
        raised = None
        try:
            dtl.poll(r.sched, ctl)
        except dsv.PdFlipDSeatVramRefused as exc:
            raised = exc
        loud = [rec for rec in caplog.records if rec.levelno >= logging.WARNING and "rc=-1" in rec.getMessage()]
        assert raised is not None or loud, "a refused lend map was swallowed at DEBUG"


# ---- R3: the sleep's quiesce must drop warm jobs even when settle raises ---------------------------------------

def test_r3_quiesce_drops_warm_jobs_even_when_settle_raises(ops):
    """RED (Auflage A3): ``quiesce_for_sleep`` runs ``ctl.settle()`` and only then ``quiesce_warm()``; a settle
    that raises (a refused map re-raised by finish_lend) skips it, the weight updater logs LEND-QUIESCE
    skipped and the sleep pauses the bank. The pending warm job survives the sleep; the first ``poll`` after
    the wake commits it INTO the rearmed tables (row_key[row] = e for a row whose bytes the pause dropped)."""
    tms = h.CoreTms()
    with h._armed(tms), B._floor():
        r, ctl = rb._live_s0(tms)
        cache = r.caches[0]
        t = cache._pool_tables
        cache._pool_srcs = [torch.ones(8, 4)]
        cache._pool_dsts = [torch.zeros(int(t.row_key.numel()), 4)]
        rows = torch.arange(int(t.seat_base), int(t.seat_base) + 1)
        epd.reserve_warm_rows(t, rows)
        job = F.submit_warm(cache, torch.tensor([5]), rows, torch.tensor([True]), torch.tensor([0]))
        assert job is not None and F._WARM_JOBS

        def _raise():
            raise dsv.PdFlipDSeatVramRefused("settle: refused map")

        with mock.patch.object(ctl, "settle", _raise):
            try:
                dtl.quiesce_for_sleep(r.sched)
            except dsv.PdFlipDSeatVramRefused:
                pass
        assert not F._WARM_JOBS, "a warm job outlived the sleep's quiesce"


# ---- R4: the device mover books (and copies) the no-op lanes ---------------------------------------------------

def test_r4_device_mover_books_only_the_moves():
    """RED (Auflage A5, instrument + device time): with no move possible (every kept row hot) the host path
    copies 0 rows; the device path copies all m lanes (a row onto itself) and books m rows' bytes into
    ``_pdflip_bank_bytes_copied`` -- the LIVE-SPANS / reseat copy figures then report moves that never
    happened, and every shrink costs m row copies on the stream the next forward waits on (E1)."""
    from flliper.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    t = _base(12)
    rows = _rows(t, 12)
    _fill(t, rows, [900 + (i % 7) for i in range(len(rows))], list(range(R, R + len(rows))))
    t.clock[0] = 500
    host = types.SimpleNamespace(_resident={"w": torch.zeros(int(t.row_key.numel()), 16)}, _pool_tables=t,
                                 _pdflip_bank_bytes_copied=0)
    dev = types.SimpleNamespace(_resident={"w": torch.zeros(int(t.row_key.numel()), 16)}, _pool_tables=t,
                                _pdflip_bank_bytes_copied=0)
    import copy

    td = copy.copy(t)
    for f in ("row_key", "row_use", "hot_phys", "pf_row", "clock"):
        setattr(td, f, getattr(t, f).clone())
    dev._pool_tables = td
    epd.set_seat_rows_on(t, 4, device_write=True,
                         move_rows=lambda mv: MoEExpertOffloadCache._move_bank_rows(host, mv))
    epd.seat_rows_off_device(td, 4, move_rows=lambda s, d: MoEExpertOffloadCache._move_bank_rows_device(dev, s, d))
    assert host._pdflip_bank_bytes_copied == 0
    assert dev._pdflip_bank_bytes_copied == host._pdflip_bank_bytes_copied, (
        "device mover booked %d bytes for 0 moves" % dev._pdflip_bank_bytes_copied)
