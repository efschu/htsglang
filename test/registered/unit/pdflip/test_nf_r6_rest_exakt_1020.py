"""NF-R6-REST-EXAKT (20.10., SOLL R6 + R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN: Einstufung uncached/S/R6-Rest/Sitz
nur aus D-Radix/L2/L3 zum Entscheidungszeitpunkt").

Live case rc28 boot ``..._297198745d_1020_174205`` (front + D log, /spinning/docker-acceptance/nf/evidence):
- 18:36:31,120 pdflip-12-275 kurz u=1394 (R2) -> D. D: #969 EXTENT chunk 1 (26496, 27264, 768) extend_ms ...391923-...391973.
- 18:36:32,918-,930 ``PDFLIP-VRAM-PEAK phase=chunk rows=769`` on TP0/TP1/TP2: chunk 1's forward has ENDED (768 of 1394 computed).
- 18:36:32,977-,983 front: ``SET-VERDICT ... REROUTE_P why=R6 ... D prefill of 1394 running``, ``r6_rest=1394``.
- 18:36:33,x D: #969 EXTENT chunk 2 (27264, 27890, 626) -- the next batch was launched only now.
Root: D-EXTEND-BEACON v3 wrote the after-rest (626) only from ``process_batch_result``, which under overlap scheduling (D runs
``overlap=True``) comes after the NEXT batch was built and launched. The front read the chunk-START record (1394). D's exact rest at
the verdict was 626. Fix: the rank arms a device event behind every extend forward and polls it (non-blocking) at its loop top and
before the next launch; the result path stays the backstop.

Danger directions (mutants): the OLD start value keeps counting after the forward ended (rest too big, a wrong flip); a rest
written TOO EARLY or TWICE (rest too small -> the R6 flip is missing)."""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _front as _fake_front  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402

RID_K, RID_G = "pdflip-12-275", "pdflip-12-274"
X_LIVE = 7016                 # 18:36:32,980 SET-VERDICT X=7016 halfX=3508
U_K, CHUNK1, REST_AFTER = 1394, 768, 626
U_G = 9875
SID = 7


class _Ev:
    """A device event stand-in: ``query`` is what the rank polls (never a sync)."""

    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done


def _read(d, rid=RID_K):
    got = dxb.read_group_full(str(d), "D", SID, session_of=lambda pid: SID)
    return None if got is None or dxb.rid_key(rid) not in got else got[dxb.rid_key(rid)][1]


def _writers(tmp_path, ranks=3, t_ns=10**9):
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: t_ns) for r in range(ranks)]
    for w in ws:
        w.note_recv(rid=RID_K)
    return ws


# ------------------------------------------------------------------------------------------------ writer (rank side)
def test_live_12_275_the_forward_of_chunk1_ended_the_poll_writes_626_before_the_next_launch(tmp_path):
    """rc28 18:36:32,918-,930: chunk 1 (768 of 1394) ended on all three ranks; the next launch came after the verdict. The poll
    (loop top / before the launch) writes D's exact rest 626 -- the front's reading at ,977 is 626, not the chunk-start 1394."""
    ws = _writers(tmp_path)
    evs = [_Ev() for _ in ws]
    for w, ev in zip(ws, evs):
        w.note_extend(rids=[RID_K], rests=[U_K], extend_lens=[CHUNK1])          # chunk 1 starts: 1394 left
        w.arm_done(key=37869, rids=[RID_K], ready=ev.query)
    assert _read(tmp_path) == U_K                                                # forward running: its tokens are not computed
    for w in ws:
        assert w.poll_done() == 0                                                # not ended: nothing written
    assert _read(tmp_path) == U_K
    for ev in evs:
        ev.done = True                                                           # 32,918-,930
    for w in ws:
        assert w.poll_done() == 1
    assert _read(tmp_path) == REST_AFTER                                         # D's exact rest at the verdict (,977)


def test_without_the_poll_the_reading_keeps_the_chunk_start_rest_base_behaviour(tmp_path):
    """The base (v3, result path only): until ``process_batch_result`` ran -- after the NEXT launch -- the record says 1394."""
    ws = _writers(tmp_path)
    for w in ws:
        w.note_extend(rids=[RID_K], rests=[U_K], extend_lens=[CHUNK1])
    assert _read(tmp_path) == U_K
    for w in ws:
        w.note_extend_done(rids=[RID_K], key=37869)                              # the backstop still writes it
    assert _read(tmp_path) == REST_AFTER


def test_the_result_path_after_the_poll_writes_nothing_twice_chunk2_rest_survives(tmp_path):
    """Overlap order on D: poll wrote chunk 1's done (626); chunk 2 launches (626 left, 626 computed); THEN chunk 1's result is
    processed. A second done of chunk 1 would pop chunk 2's FIFO entry and write 0 while chunk 2 still runs (rest too small ->
    the R6 flip missing). Must stay 626 until chunk 2's own forward has ended."""
    (w,) = _writers(tmp_path, ranks=1)
    ev1, ev2 = _Ev(), _Ev()
    w.note_extend(rids=[RID_K], rests=[U_K], extend_lens=[CHUNK1])
    w.arm_done(key=37869, rids=[RID_K], ready=ev1.query)
    ev1.done = True
    assert w.poll_done() == 1 and _read(tmp_path) == REST_AFTER
    w.note_extend(rids=[RID_K], rests=[REST_AFTER], extend_lens=[REST_AFTER])   # chunk 2 starts
    w.arm_done(key=37870, rids=[RID_K], ready=ev2.query)
    w.note_extend_done(rids=[RID_K], key=37869)                                  # chunk 1's result, late (overlap)
    assert _read(tmp_path) == REST_AFTER                                         # chunk 2 runs: 626 not computed yet
    assert w.poll_done() == 0
    ev2.done = True
    assert w.poll_done() == 1 and _read(tmp_path) == 0                           # chunk 2 ended: computed
    w.note_extend_done(rids=[RID_K], key=37870)                                  # its result: nothing twice
    assert _read(tmp_path) == 0


def test_the_poll_stops_at_the_first_forward_that_has_not_ended(tmp_path):
    """One stream: a later forward cannot have ended before an earlier one -- the poll writes oldest first and stops."""
    (w,) = _writers(tmp_path, ranks=1)
    ev1, ev2 = _Ev(False), _Ev(True)
    w.note_extend(rids=[RID_K], rests=[3000], extend_lens=[1000])
    w.arm_done(key=1, rids=[RID_K], ready=ev1.query)
    w.note_extend(rids=[RID_K], rests=[2000], extend_lens=[1000])
    w.arm_done(key=2, rids=[RID_K], ready=ev2.query)
    assert w.poll_done() == 0 and _read(tmp_path) == 3000                       # review A1: chunk 1 still runs -> its start rest
    ev1.done = True
    assert w.poll_done() == 2 and _read(tmp_path) == 1000


def test_a_result_flushes_older_armed_batches_first_fifo_stays_aligned(tmp_path):
    """A batch armed but never polled (event unreadable) and a later unarmed batch (a skip batch): the later result writes the
    older armed done first, then its own -- each rid's FIFO pops in order."""
    (w,) = _writers(tmp_path, ranks=1)

    def _boom():
        raise RuntimeError("no event")

    w.note_extend(rids=[RID_K], rests=[3000], extend_lens=[1000])
    w.arm_done(key=5, rids=[RID_K], ready=_boom)
    assert w.poll_done() == 0
    w.note_extend(rids=[RID_K], rests=[2000], extend_lens=[2000])                # key 6, not armed
    w.note_extend_done(rids=[RID_K], key=6)
    assert _read(tmp_path) == 0
    w.note_extend_done(rids=[RID_K], key=5)                                      # its own result: already written
    assert _read(tmp_path) == 0


def test_rank_agreement_max_keeps_the_rest_until_every_rank_polled_its_end(tmp_path):
    """R11: each rank polls its own event; the front's MAX over the ranks keeps the chunk-start rest until EVERY rank wrote its
    end -- deterministic, independent of the file order."""
    ws = _writers(tmp_path)
    evs = [_Ev() for _ in ws]
    for w, ev in zip(ws, evs):
        w.note_extend(rids=[RID_K], rests=[U_K], extend_lens=[CHUNK1])
        w.arm_done(key=37869, rids=[RID_K], ready=ev.query)
    for i in (0, 1):
        evs[i].done = True
        ws[i].poll_done()
    assert _read(tmp_path) == U_K                                                # TP2 not ended yet
    evs[2].done = True
    ws[2].poll_done()
    assert _read(tmp_path) == REST_AFTER


# ------------------------------------------------------------------------------------------------ scheduler wiring (rank side)
def test_scheduler_arms_every_extend_forward_and_polls_before_the_launch_without_a_sync():
    from flliper.srt.managers import scheduler as S

    fwd = inspect.getsource(S.Scheduler._run_batch_forward)
    assert "self._pdflip_d_ext_arm_done(batch)" in fwd
    i_arm, i_fwd = fwd.index("self._pdflip_d_ext_arm_done(batch)"), fwd.rindex("forward_batch_generation(")
    assert i_arm > i_fwd                                                         # behind the forward, not before it
    loop = inspect.getsource(S.Scheduler.event_loop_overlap)
    assert loop.count("self._pdflip_d_ext_poll()") == 2
    assert loop.index("self._pdflip_d_ext_poll()") < loop.index("self.get_next_batch_to_run(")
    assert loop.rindex("self._pdflip_d_ext_poll()") < loop.index("batch_result = self.run_batch(batch)")
    res = inspect.getsource(S.Scheduler.process_batch_result)
    assert 'key=getattr(batch, "forward_iter", None)' in res
    arm = inspect.getsource(S.Scheduler._pdflip_d_ext_arm_done) + inspect.getsource(S.Scheduler._pdflip_d_ext_poll)
    assert "synchronize" not in arm and ".wait(" not in arm                     # a query, never a sync


def test_arm_helper_records_on_the_forward_stream_and_hands_the_query(tmp_path):
    from flliper.srt.managers import scheduler as S

    rec = {}

    class _E:
        def record(self, stream=None):
            rec["stream"] = stream

        def query(self):
            return True

    (w,) = _writers(tmp_path, ranks=1)
    w.note_extend(rids=[RID_K], rests=[U_K], extend_lens=[CHUNK1])
    fake = type("Sch", (), {})()
    fake.device_module = type("DM", (), {"Event": _E})
    fake.enable_overlap = True
    fake.forward_stream = "FWD"
    fake.pdflip_d_ext_beacon = w
    batch = type("B", (), {})()
    batch.forward_iter = 37869
    batch.reqs = [type("R", (), {"rid": RID_K})()]
    S.Scheduler._pdflip_d_ext_arm_done(fake, batch)
    assert rec["stream"] == "FWD"
    S.Scheduler._pdflip_d_ext_poll(fake)
    assert _read(tmp_path) == REST_AFTER
    fake.pdflip_d_ext_beacon = None
    S.Scheduler._pdflip_d_ext_poll(fake)                                         # beacon off: nothing, no error


# ------------------------------------------------------------------------------------------------ the front's verdict (R6)
def _front_live(now, tmp_path, *, member_u, ended, rest_start=U_K, ext_len=CHUNK1, x=X_LIVE, ranks=3, klass=None, waited=3.9,
                rid_run=RID_K, half_x=3508):
    """The real front: D decodes, the running D prefill ``rid_run`` granted at now-1.86 and posted, its leg redirectable; its
    chunk began at now-1.0; ``ended`` = its forward's device end was polled. The set member waits ``waited`` s (live 3.9 <
    T_halb). ``klass`` = the class the running prefill went to D with (SOLL R6k; None = unknown, named)."""
    import asyncio as _a
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = x
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    add_running(f, now, tmp_path, rid_run, rest_start=rest_start, ext_len=ext_len, ended=ended, ranks=ranks, klass=klass,
                half_x=half_x)
    f._d_ext_reading = lambda: dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)
    if member_u >= x:                                    # gross: a queued P candidate (R4)
        g = F.Pending(RID_G, "/v1/messages", {}, "t", now - waited, _a.new_event_loop().create_future(),
                      est_prompt=32979, est_uncached=member_u)
        g.x_set = x
        f.queue.append(g)
    else:                                                # halb: a set member collecting (R3)
        f._set_st()["shorts"][RID_G] = (now - waited, member_u, x)
    return f


def add_running(f, now, tmp_path, rid, *, rest_start, ext_len, ended, ranks=3, klass=None, key=37869, half_x=3508):
    """A RUNNING D prefill ``rid`` (granted, posted, leg redirectable, its chunk begun at now-1.0 on ``ranks`` real beacon
    writers, ``ended`` polled) and the class it went to D with (SOLL R6k)."""
    f._d_pf_book().grant(rid=rid, tokens=rest_start, now=now - 1.86)
    f._d_pf_book().enter_leg2(rid=rid)
    f.groups["D"].outstanding[rid] = now - 1.86
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[rid] = now - 1.86
    f._leg2_lookahead.add(rid)
    clock = {"t": int((now - 1.0) * 1e9)}
    ev = {"done": bool(ended)}
    ws = []
    for r in range(ranks):
        w = dxb.DExtendBeacon(str(tmp_path / ("D-pid%d%d.bin" % (r + 1, sum(map(ord, rid)) % 1000))), clock_ns=lambda: clock["t"])
        w.note_recv(rid=rid)
        w.note_extend(rids=[rid], rests=[rest_start], extend_lens=[ext_len])
        w.arm_done(key=key, rids=[rid], ready=lambda: ev["done"])
        w.poll_done()
        ws.append(w)
    if klass is not None:
        F.Front._r6k_note_class(f, rid, klass, "arrival", half_x)
    f.__dict__.setdefault("_test_running", {})[rid] = (ws, ev)
    return ws, ev


def end_forward(f, rid):
    """The running prefill's forward ends on the device: every rank's poll writes its after-rest."""
    ws, ev = f._test_running[rid]
    ev["done"] = True
    for w in ws:
        w.poll_done()


def test_front_live_12_275_class_unknown_r6_counts_the_exact_626(monkeypatch, caplog, tmp_path):
    """Exactness of the number (class not known to the front: named ``r6k_class_unknown``, R6 applies as before): r6_rest=626, not
    1394 -> S=10501 > X=7016, REROUTE_P why=R6 with counted 626."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=U_G, ended=True)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == REST_AFTER
        assert f._decode_collect(now) == "P"
    sv = [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]
    assert any("verdict=REROUTE_P" in m and "S=%d" % (U_G + REST_AFTER) in m for m in sv), sv
    assert any("R6-REDIRECT" in m and "counted=%s:%d" % (RID_K, REST_AFTER) in m for m in caplog.messages)
    assert f.counters["r6k_class_unknown"] >= 1


def test_front_live_12_275_r6k_the_kurz_stays_on_d_the_gross_member_waits_its_t_half(monkeypatch, caplog, tmp_path):
    """SOLL R6k (user 20.10. "b"): 12-275 went to D as kurz (R2, u=1394 < halbX 3508). Its rest is not in S, it is not redirected,
    and it pulls no flip forward: the gross member (waited 3.9 s < T_halb 4.5 s, D decodes) keeps waiting (R4)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=U_G, ended=True, klass="kurz")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"
    assert not any("verdict=REROUTE_P" in m for m in caplog.messages if "PDFLIP SET-VERDICT" in m)
    assert not any("R6-REDIRECT" in m or "PARK-HANDBACK" in m for m in caplog.messages)
    assert f.counters["r6_redirected"] == 0 and f.counters["r6k_kurz_stays_on_d"] >= 1
    assert any("R6K-KURZ-STAYS-ON-D rid=%s d_rest=%d" % (RID_K, REST_AFTER) in m for m in caplog.messages)


def test_front_halb_member_exact_rest_keeps_s_under_x_no_flip(monkeypatch, caplog, tmp_path):
    """Where the exact number decides: halb member u=3600 (halbX 3508 <= u < X 7016), a running HALB D prefill of 4000 (R6 applies,
    SOLL R6k) whose first chunk (2048) has ended. Stale start rest: S=7600 > X -> REROUTE_P (R6). Exact rest 1952: S=5552 < X ->
    the set collects (R3), no flip under X (R7)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=3600, ended=True, rest_start=4000, ext_len=2048, klass="halb")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 1952
        assert f._decode_collect(now) == "hold"
    assert not any("verdict=REROUTE_P" in m for m in caplog.messages if "PDFLIP SET-VERDICT" in m)
    assert f.counters["r6_redirected"] == 0


def test_front_halb_member_while_the_chunk_still_runs_the_start_rest_counts_r6(monkeypatch, tmp_path):
    """Danger direction 'rest too small -> flip missing': the same halb chunk whose forward has NOT ended -- its 4000 are not
    computed, S=7600 > X -> REROUTE_P (SOLL R6, EZ-S2b review A2: a running forward counts its rest)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=3600, ended=False, rest_start=4000, ext_len=2048, klass="halb")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 4000
        assert f._decode_collect(now) == "P"
    assert f.counters["set_reroute_p"] == 1 and f.counters["r6_redirected"] == 1


def test_the_armed_event_is_held_strongly_never_a_weakref(tmp_path):
    """Death fix ce594d90ba (torch 2.11: a weakref on a torch.cuda Stream/Event dangles): the beacon keeps the armed event alive by
    a strong reference (its bound ``query``) until the done is written; the module takes no weakref at all."""
    import gc

    (w,) = _writers(tmp_path, ranks=1)
    w.note_extend(rids=[RID_K], rests=[U_K], extend_lens=[CHUNK1])
    ev = _Ev(True)
    w.arm_done(key=1, rids=[RID_K], ready=ev.query)
    del ev
    gc.collect()
    assert w.poll_done() == 1 and _read(tmp_path) == REST_AFTER
    assert "weakref" not in inspect.getsource(dxb)



# ------------------------------------------------------------------------------------------------ R6k: finished on D before the flip
def _hold_front(monkeypatch, tmp_path, *, klass="kurz", ended=False, rest_start=1394, ext_len=768):
    _on(monkeypatch)
    now = time.time()
    return now, _front_live(now, tmp_path, member_u=U_G, ended=ended, rest_start=rest_start, ext_len=ext_len, klass=klass)


def test_r6k_hold_measured_limit_then_done_when_the_kurz_rest_ended(monkeypatch, caplog, tmp_path):
    """12-275 after chunk 1 (768 of 1394 in 1.0 s, rest 626): the flip waits; limit = 2 x 626 x 1.0/768 s = 1630 ms
    (basis=measured, < T_halb). Chunk 2 ends -> rest 0 -> outcome=done, the flip may begin."""
    now, f = _hold_front(monkeypatch, tmp_path, ended=True)
    end_forward(f, RID_K)                                     # chunk 1 polled: rest 626
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert F.Front._r6k_finish_hold(f, now) is True
        assert F.Front._r6k_finish_hold(f, now + 0.5) is True
        ws, _ev = f._test_running[RID_K]
        for w in ws:                                          # chunk 2 (626) launched and ended on every rank
            w.note_extend(rids=[RID_K], rests=[REST_AFTER], extend_lens=[REST_AFTER])
            w.note_extend_done(rids=[RID_K])
        assert F.Front._r6k_finish_hold(f, now + 0.9) is False
    hold = [m for m in caplog.messages if "R6K-FINISH-BEFORE-FLIP outcome=hold" in m]
    done = [m for m in caplog.messages if "R6K-FINISH-BEFORE-FLIP outcome=done" in m]
    assert hold and "rests=626" in hold[0] and "basis=measured" in hold[0] and "limit_ms=1630" in hold[0], hold
    assert done and "wait_ms=900" in done[0] and RID_K in done[0], done
    assert f.counters["r6k_finish_before_flip_done"] == 1


def test_r6k_hold_first_chunk_in_flight_limit_is_t_half_and_the_timeout_is_named(monkeypatch, caplog, tmp_path):
    """No rate measured yet (chunk 1 in flight): limit = T_halb 4.5 s. Past it the flip goes as before, named -- never an endless
    block; in the same D phase it never holds again."""
    now, f = _hold_front(monkeypatch, tmp_path, ended=False)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert F.Front._r6k_finish_hold(f, now) is True
        assert F.Front._r6k_finish_hold(f, now + 4.4) is True
        assert F.Front._r6k_finish_hold(f, now + 4.6) is False
        assert F.Front._r6k_finish_hold(f, now + 5.0) is False
    assert any("outcome=hold" in m and "basis=t_half" in m and "limit_ms=4500" in m for m in caplog.messages)
    assert any("outcome=timeout" in m and "wait_ms=4600" in m for m in caplog.messages)
    assert f.counters["r6k_finish_before_flip_timeout"] == 1


def test_r6k_no_hold_for_a_halb_running_prefill_or_a_computed_kurz(monkeypatch, tmp_path):
    """Only a RUNNING kurz with a rest holds: a halb one is R6's (redirect), a kurz whose last forward ended holds nothing."""
    now, f = _hold_front(monkeypatch, tmp_path, klass="halb", rest_start=4000, ext_len=2048)
    assert F.Front._r6k_finish_hold(f, now) is False
    d2 = tmp_path / "done"
    d2.mkdir()
    now, f = _hold_front(monkeypatch, d2, ended=False, rest_start=600, ext_len=600)
    end_forward(f, RID_K)
    assert F.Front._r6k_finish_hold(f, now) is False


def _park_rpc(f, calls):
    async def _rpc(group, path, body, timeout):
        calls.append(path)
        return 404, ""
    f.rpc = _rpc
    f._park_attempt_epoch = -1
    f._park_unsupported = False


def test_r6k_the_park_waits_for_the_kurz_rest_and_its_attempt_is_not_spent(monkeypatch, caplog, tmp_path):
    """The D->P quiesce: ``_wait_bound_park`` answers R6K_HOLD (admission closed, NO park RPC, the once-per-phase attempt not
    spent) while the kurz rest runs; once D reported it ended, the park goes out. Mutant 'parks again at the chunk boundary'
    (no hold) sends the park at once."""
    now, f = _hold_front(monkeypatch, tmp_path, ended=False)
    calls = []
    _park_rpc(f, calls)
    loop = asyncio.new_event_loop()
    try:
        with caplog.at_level(logging.INFO, logger="pdflip.front"):
            got = loop.run_until_complete(f._wait_bound_park(None, immediate=f.queue[0], cause="before-flip"))
            assert got == F.R6K_HOLD and calls == [] and f.admit_d is False
            assert f._park_attempt_epoch != f.epoch
            ws, _ev = f._test_running[RID_K]
            for w in ws:
                w.note_extend(rids=[RID_K], rests=[REST_AFTER], extend_lens=[REST_AFTER])
            end_forward(f, RID_K)                             # chunk 1 ended (rest 626) ...
            assert loop.run_until_complete(f._wait_bound_park(None, immediate=f.queue[0], cause="before-flip")) == F.R6K_HOLD
            for w in ws:
                w.note_extend_done(rids=[RID_K])              # ... chunk 2 ended: computed
            got = loop.run_until_complete(f._wait_bound_park(None, immediate=f.queue[0], cause="before-flip"))
    finally:
        loop.close()
    assert got != F.R6K_HOLD and calls, (got, calls)          # the park went out after the kurz ended on D
    assert any("R6K-FINISH-BEFORE-FLIP outcome=done" in m for m in caplog.messages)


def test_r6k_the_manual_flip_does_not_wait(monkeypatch, tmp_path):
    now, f = _hold_front(monkeypatch, tmp_path, ended=False)
    calls = []
    _park_rpc(f, calls)
    loop = asyncio.new_event_loop()
    try:
        got = loop.run_until_complete(f._wait_bound_park(None, cause=F.MANUAL_FLIP_PARK_CAUSE))
    finally:
        loop.close()
    assert got != F.R6K_HOLD and calls


def test_r6k_controller_holds_the_flip_on_both_paths():
    src = inspect.getsource(F.Front.controller)
    assert src.count("== R6K_HOLD") == 2
    i = src.index("elif Front._r6k_finish_hold(self):")
    assert i < src.index('await self.flip("D", "P")', i)



# ------------------------------------------------------------------------------------------------ review auflagen K1 K2 N1 A1 (20.10.)
def test_a1_overlap_start_record_of_chunk2_keeps_chunk1_start_rest_until_chunk1_ended(tmp_path):
    """Review A1 (SOLL R6, danger direction 'rest too small'): chunk 2 launched on the host while chunk 1 (3000 left, 1000 in it)
    still runs on the device: the record stays 3000 (FIFO head), not chunk 2's 2000; chunk 1 ends -> 2000; chunk 2 ends -> 1000."""
    (w,) = _writers(tmp_path, ranks=1)
    ev1, ev2 = _Ev(), _Ev()
    w.note_extend(rids=[RID_K], rests=[3000], extend_lens=[1000])
    w.arm_done(key=1, rids=[RID_K], ready=ev1.query)
    w.note_extend(rids=[RID_K], rests=[2000], extend_lens=[1000])
    w.arm_done(key=2, rids=[RID_K], ready=ev2.query)
    assert _read(tmp_path) == 3000
    ev1.done = True
    assert w.poll_done() == 1 and _read(tmp_path) == 2000
    ev2.done = True
    assert w.poll_done() == 1 and _read(tmp_path) == 1000


def test_k1_a_kurz_class_contradicted_by_ds_exact_rest_counts_r6(monkeypatch, caplog, tmp_path):
    """Review K1 (user 19.10. exact numbers): noted kurz at halfX 3508, but D's exact rest is 5000 >= halfX -> u >= halfX proven:
    not kurz, R6 counts it (S = 3600 + 5000 > X -> REROUTE_P), named r6k_kurz_contradicted; no finish-hold for it."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=3600, ended=False, rest_start=5000, ext_len=2048, klass="kurz")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert F.Front._r6k_finish_hold(f, now) is False
        assert f._dc_carried_running() == 5000
        assert f._decode_collect(now) == "P"
    assert f.counters["r6k_kurz_contradicted"] >= 1 and f.counters["r6_redirected"] == 1
    assert any("R6K-KURZ-CONTRADICTED rid=%s d_rest=5000 halfX=3508" % RID_K in m for m in caplog.messages)


def test_k1_b_every_d_refusal_clears_the_kurz_note(monkeypatch, tmp_path):
    """Review K1: D refused (no extent; and the early return of an entry without measure_class) -> the kurz note is gone, the
    class is unknown, R6 applies as before (named)."""
    from types import SimpleNamespace

    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=U_G, ended=False, klass="kurz")
    assert RID_K in f._r6k_class
    F.Front._measure_return(f, RID_K, SimpleNamespace(measure_class=""), None, 27890)
    assert RID_K not in f._r6k_class
    F.Front._r6k_note_class(f, RID_K, "kurz", "arrival", 3508)
    F.Front._measure_return(f, RID_K, SimpleNamespace(measure_class="kurz", measured_u=1, measure_src="x"), None, 27890)
    assert RID_K not in f._r6k_class


def test_k2_the_class_is_noted_even_when_the_instrument_raises(monkeypatch, tmp_path):
    """Review K2: the note stands before the instrument's try -- an instrument that raises leaves the class known."""
    _on(monkeypatch)
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE

    def _boom(self):
        raise RuntimeError("instrument")

    monkeypatch.setattr(F.Front, "_dec_group", _boom)
    F.Front._dec_arrival(f, RID_K, None, remainder=U_K, x_set=X_LIVE, route="short", p_only=False,
                         t_arrive_m=time.monotonic(), n_prompt=27890)
    assert f._r6k_class[RID_K][0] == "kurz" and f._r6k_class[RID_K][2] == 3508


def test_n1_the_no_park_hold_closes_admission():
    """Review N1 (R1a): on the controller path without a park the R6K hold closes admission -- no new kurz goes to D meanwhile."""
    src = inspect.getsource(F.Front.controller)
    i = src.index("elif Front._r6k_finish_hold(self):")
    j = src.index("continue", i)
    assert "self.admit_d = False" in src[i:j]
