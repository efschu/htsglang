"""NF-RC31-R6REST (21.10., SOLL R6 + R7 + R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN: Einstufung uncached/S/R6-Rest/Sitz
nur aus D-Radix/L2/L3 zum Entscheidungszeitpunkt").

Live chain, NF boot ``boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10212025_82784e1138_1021_202608`` (rc31b, evidence tree):
- D: pdflip-6-401 (halb, u=6550) chunk 1 = 2240 tokens, forward 21:15:33,730 -> 36,826 (VRAM-PEAK phase=chunk rows=2240).
- front 36,910 ``R6-COMMIT-PENDING d_rest=6550`` (D's recv step of that iteration), 36,980 ``R6-NOT-STARTED running=4310``,
  36,982 ``SET-VERDICT ... S=8096 X=7370 ... REROUTE_P why=R6 ... D prefill of 4310 running``, 36,989 ``R6-REDIRECT ...
  in_flight=pdflip-6-401:0`` -- the beacon was exact at the read: no COMMITTED, nothing launched beyond chunk 1.
- 37,114 the abort leaves the front (the leg's ROS poll, 125 ms after the verdict); D's next recv step came first: chunk 2 = 2300
  (STUECKELUNG at :36, H107 EAGER-LRU T=2300 at :37, forward to 40,066), the abort applied after it.
- 40 ``PDFLIP-D-PARK RETAINED rid=pdflip-6-401 retained=155521 of 157590 (tail 2069)``; P computed 2070 + 3786: S real 5856 < X.

Fix: the R6 verdict stops the counted prefill FIRST (abort now), holds, and decides on D's halt record -- written by the rank when it
APPLIES the abort (``DExtendBeacon.note_halt``): the free at the halt. Danger directions: a halt record that reports the free of an
EARLIER launch (too big -> a flip under X again), a halt record overwritten by the in-flight chunk's done (too small -> R6 lost), a
halt of an earlier grant read for a re-grant, a D without halt records that blocks every R6 verdict."""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402

RID = "pdflip-6-401"
SID = 7
U, CHUNK1, CHUNK2 = 6550, 2240, 2300
FREE1, FREE2 = U - CHUNK1, U - CHUNK1 - CHUNK2          # 4310, 2010


class _Ev:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done


def _read(d):
    return dxb.read_group_full(str(d), "D", SID, session_of=lambda pid: SID)


def _ranks(tmp_path, n=3, t_ns=2 * 10**9):
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: t_ns) for r in range(n)]
    for w in ws:
        w.announce_halt()
        w.note_recv(rid=RID)
    return ws


def _chunk1_ended(ws):
    evs = []
    for w in ws:
        ev = _Ev()
        w.note_extend(rids=[RID], rests=[U], extend_lens=[CHUNK1])
        w.arm_done(key=1, rids=[RID], ready=ev.query)
        evs.append(ev)
    for w, ev in zip(ws, evs):
        ev.done = True
        assert w.poll_done() == 1


# ------------------------------------------------------------------------------------------------ writer (rank side)
def test_live_6401_chunk2_launched_before_the_abort_the_halt_reports_2010_not_4310(tmp_path):
    """The free read at the verdict (4310) was exact; D launched chunk 2 (2300) before the abort landed. The halt record of the
    applied abort says 2010 -- the tokens that really reach P (chunk 2 ends on D)."""
    ws = _ranks(tmp_path)
    _chunk1_ended(ws)
    assert dxb.free_of(_read(tmp_path), RID, 1.0) == FREE1                 # the front's read at 36,980
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) is None              # not halted: D may still launch
    for w in ws:
        w.commit_next(rids=[RID])                                          # D's next recv step (no abort in it yet)
        w.note_extend(rids=[RID], rests=[FREE1], extend_lens=[CHUNK2])     # chunk 2 launched
        w.arm_done(key=2, rids=[RID], ready=lambda: False)
    for w in ws:
        assert w.note_halt(rid=RID)                                        # the abort applied after chunk 2's launch
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) == FREE2
    assert dxb.rest_of(_read(tmp_path), RID, 1.0) == FREE1                 # D still computes chunk 2 (in flight)


def test_halt_before_the_next_chunk_reports_the_full_free_and_resolves_committed(tmp_path):
    """The abort applied in the same step that committed the next chunk (``process_pending_chunked_abort`` runs before the batch
    is built): nothing more launches -- halt free 4310, and the rid's own COMMITTED marker is resolved at once."""
    ws = _ranks(tmp_path)
    _chunk1_ended(ws)
    for w in ws:
        w.commit_next(rids=[RID])
    assert dxb.free_of(_read(tmp_path), RID, 1.0) == dxb.COMMITTED
    for w in ws:
        assert w.note_halt(rid=RID)
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) == FREE1
    assert dxb.free_of(_read(tmp_path), RID, 1.0) == FREE1                 # the marker never outlives the halt


def test_the_in_flight_chunks_done_does_not_overwrite_the_halt(tmp_path):
    """Danger direction "too small": chunk 2's forward ends after the halt -- its done record is the rid's own (rest 2010), the
    halt record (free at the halt) stays 2010 and is never replaced by a later write of the rid."""
    (w,) = _ranks(tmp_path, n=1)
    _chunk1_ended([w])
    ev2 = _Ev()
    w.note_extend(rids=[RID], rests=[FREE1], extend_lens=[CHUNK2])
    w.arm_done(key=2, rids=[RID], ready=ev2.query)
    assert w.note_halt(rid=RID)
    ev2.done = True
    assert w.poll_done() == 1
    w.note_extend_done(rids=[RID], key=2)
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) == FREE2
    assert dxb.rest_of(_read(tmp_path), RID, 1.0) == FREE2


def test_a_halt_of_an_earlier_grant_is_not_read_for_a_regrant(tmp_path):
    """R6 back on D (R7): the same rid is granted again -- the old halt record (t_first before the new grant) is no halt of it."""
    clock = {"t": 2 * 10**9}
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: clock["t"])
    w.announce_halt()
    w.note_recv(rid=RID)
    w.note_extend(rids=[RID], rests=[U], extend_lens=[CHUNK1])
    assert w.note_halt(rid=RID)
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) == FREE1
    clock["t"] = 9 * 10**9
    w.note_recv(rid=RID)
    w.note_extend(rids=[RID], rests=[FREE1], extend_lens=[FREE1])
    assert dxb.halted_free(_read(tmp_path), RID, 5.0) is None


def test_a_rid_the_leg_never_extended_writes_no_halt_and_a_rank_without_the_wiring_is_not_capable(tmp_path):
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"))
    assert not dxb.halt_capable(_read(tmp_path))                          # no announce_halt: the front keeps the old path
    w.note_recv(rid=RID)
    assert not w.note_halt(rid=RID)
    assert dxb.halted_free(_read(tmp_path), RID, 0.0) is None
    w.announce_halt()
    assert dxb.halt_capable(_read(tmp_path))


def test_the_capability_record_survives_the_ring_wrapping(tmp_path):
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), slots=16)
    w.announce_halt()
    for i in range(200):
        r = "r%d" % i
        w.note_recv(rid=r)
        w.note_extend(rids=[r], rests=[10], extend_lens=[10])
    assert dxb.halt_capable(_read(tmp_path))


def test_build_announces_and_the_scheduler_applies_the_halt_where_the_abort_is_applied(tmp_path):
    """Wiring: ``build`` announces; ``process_pending_chunked_abort`` writes the halt right where ``chunked_req`` is cleared (no
    chunk of it launches after); ``_abort_request_now`` writes it for an abort applied at receipt (not the chunked one)."""
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(dxb.build)
    assert "w.announce_halt()" in src
    pc = inspect.getsource(S.Scheduler.process_pending_chunked_abort)
    assert pc.index("self.chunked_req = None") < pc.index("Scheduler._pdflip_d_ext_halt(self, req.rid)")
    ab = inspect.getsource(S.Scheduler._abort_request_now)
    assert "Scheduler._pdflip_d_ext_halt(self, recv_req.rid)" in ab
    assert "str(_cr_halt.rid).startswith(recv_req.rid)" in ab                # the chunked one waits for its application
    S.Scheduler._pdflip_d_ext_halt(types.SimpleNamespace(), "x")            # a bare-namespace harness: no beacon, no error
    calls = []
    ns = types.SimpleNamespace(pdflip_d_ext_beacon=types.SimpleNamespace(note_halt=lambda rid: calls.append(rid)))
    S.Scheduler._pdflip_d_ext_halt(ns, RID)
    assert calls == [RID]


# ------------------------------------------------------------------------------------------------ front
def _front(monkeypatch, tmp_path, *, announce=True, member_u=3786, waited=0.1):
    from test_nf_r6_rest_exakt_1020 import _front_live
    from test_pdflip_set_scheduler_1017 import _on

    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=member_u, ended=True, rest_start=U, ext_len=CHUNK1, x=7370, klass="halb",
                    waited=waited, rid_run=RID, half_x=3685)
    ws, _ev = f._test_running[RID]
    if announce:
        for w in ws:
            w.announce_halt()
    return now, f, ws


def test_front_a_d_without_halt_records_keeps_the_redirect_on_the_read_named(monkeypatch, tmp_path):
    """A D group that announces no halt records: the R6 verdict stands on the free read at the verdict, as before (it would
    otherwise wait for the liveness guard on every R6 verdict) -- named ``r6_halt_d_not_capable``."""
    now, f, _ws = _front(monkeypatch, tmp_path, announce=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(now) == "P"
    assert f.counters["set_reroute_p"] == 1 and f.counters["r6_halt_d_not_capable"] == 1 and f.counters["r6_halt_sent"] == 0


def test_front_switch_off_restores_the_redirect_on_the_read(monkeypatch, tmp_path):
    now, f, _ws = _front(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), envs.FLLIPER_PDFLIP_R6_HALT_FIRST.override(False):
        assert f._decode_collect(now) == "P"
    assert f.counters["set_reroute_p"] == 1 and f.counters["r6_halt_sent"] == 0


def test_front_a_stop_d_never_confirms_is_named_and_counted_with_the_current_free(monkeypatch, caplog, tmp_path):
    """Liveness guard (never a decision clock): no halt record within ``R6_HALT_STALE_S`` -> R6-HALT-STALE, the beacon's
    current free (4310) is taken -> S = 8096 > X -> REROUTE_P."""
    now, f, _ws = _front(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
        monkeypatch.setattr(F.Front, "R6_HALT_STALE_S", 0.0)
        assert f._decode_collect(now + 0.1) == "P"
    assert f.counters["r6_halt_stale"] == 1 and f.counters["set_reroute_p"] == 1
    assert any("R6-HALT-STALE rid=%s" % RID in m for m in caplog.messages)


def test_front_a_flip_another_way_while_the_stop_is_pending_takes_it_to_p(monkeypatch, tmp_path):
    """The phase moves on (another flip, epoch +1) before the decision: the stopped prefill rides to P (P-BATCH-ALL)."""
    now, f, _ws = _front(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(now) == "hold"
        f.epoch += 1
        F.Front._r6_halt_tick(f)
    assert f._r6_halt[RID]["decision"] == "P" and f.counters["r6_halt_to_p"] == 1


def _requeue(f, rid):
    """Run ``_requeue_park_handback`` (the closed leg) until it waits for its request's turn; returns the task's loop."""
    async def _rpc(group, path, body, timeout):
        return 200, ""
    f.rpc = _rpc
    f.ftok = None
    f.x_exact = False
    req = types.SimpleNamespace(path="/v1/messages")
    loop = asyncio.new_event_loop()
    task = loop.create_task(f._requeue_park_handback(req, rid, {}, "t" * 30, True, None, None))
    for _ in range(5):
        loop.run_until_complete(asyncio.sleep(0))
    return loop, task


def _close(loop, task):
    task.cancel()
    try:
        loop.run_until_complete(task)
    except BaseException:  # noqa: BLE001
        pass
    loop.close()


def test_front_the_leg_closed_before_the_verdict_waits_then_goes_back_to_d_priced_at_the_halted_rest(monkeypatch, tmp_path):
    """R7 path end to end on the front: the leg closes (D's abort) while the verdict waits -> the request is held out of every
    queue; D halts at 2010 -> no flip, it joins the queue D-eligible at u = 2010 (kurz by its halted rest, SOLL R6k note) for
    D's admission (QUEUED-SHORT), never P's batch."""
    now, f, ws = _front(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(now) == "hold"
        f._kick_controller = lambda *a, **k: None
        f._dp_mark = lambda *a, **k: None
        f._lane_for = lambda rid: 0
        loop, task = _requeue(f, RID)
        try:
            assert not [q for q in f.queue if q.rid == RID] and f._r6_halt[RID]["p"] is not None
            for w in ws:
                w.note_extend(rids=[RID], rests=[FREE1], extend_lens=[CHUNK2])
                w.note_halt(rid=RID)
            assert f._decode_collect(now + 0.1) == "hold"
            # review A2 (SOLL R2): kurz by its halted rest -> D's admission line at once (QUEUED-SHORT), not after the set
            assert not [q for q in f.queue if q.rid == RID]
            q = [q for q in f._ready_for_d if q.rid == RID]
            assert len(q) == 1 and q[0].d_eligible and int(q[0].est_uncached) == FREE2 and q[0].d_direct
            assert f.counters["r6_halt_kurz_to_d_now"] == 1
            assert f._r6k_class[RID][0] == "kurz"
        finally:
            _close(loop, task)
