"""NF-RC30-R6STALE (21.10., SOLL R6 + R6k + R7 + R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN").

Live case NF boot ``boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210632_1d822bd150_1021_063252`` (front + D log,
/spinning/docker-acceptance/nf/evidence):
- 07:53:01,960 pdflip-48-687 CLASSIFY rule=R3 u=3280 (d_probe) X=6114 halfX=3057 -> halb (R6k does NOT apply).
- 07:53:10,549 D-ADMIT 48-687 (D_AFTER_TIMER, R3). D: extend_ms 07:53:11.128 chunk 1 = 2496 (GROUP-NARROWED), forward ends
  14.365; chunk 2 = 784 launched 14.365 (``PDFLIP-VRAM-PEAK phase=chunk t0=...194365``), ends 15.616.
- 07:53:11,134 SET-REPROBE begin verdict_on_old=REROUTE_P (pdflip-48-689 u=4584): chunk 1 had been launched 6 ms before.
- 07:53:14,400 SET-VERDICT S=7864 X=6114 verdict=REROUTE_P why=R6 ... D prefill of 3280 running; R6-REDIRECT counted
  pdflip-48-687:3280. 07:53:24,873 P served it: prompt 39376, cached 39360 -> P computed 16 tokens.
Same class: 07:51:27,477 pdflip-42-671 counted 4681 -> P 9 (chunk 2 launched 53 ms before the verdict); 07:52:37,790
pdflip-46-680 counted 5151 -> P 2911 (chunk 1 = 2304 launched 22 ms before; 5151 - 2304 = 2847 plus the store's page).

Root: the D-EXTEND-BEACON ``rest`` is the start rest of the OLDEST batch whose forward has not ended (review A1, FIFO head) --
it includes the chunks D has already handed to a forward. Those end on D (the abort lands at D's next loop top, after them) and
P reads them from the store. The R6 verdict lands right at D's pass boundary (the re-probe answers there), i.e. just after D
launched the next chunk while the previous chunk's done was not yet polled, so ``rest`` was the arrival u. Fix: beacon v4 carries
``free`` = the tokens NO forward was launched for, and R6 counts exactly that into S (and redirects only what has some).
"""
from __future__ import annotations

import logging
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from test_nf_r6_rest_exakt_1020 import _front_live, _Ev  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402

RID_RUN, RID_SET = "pdflip-48-687", "pdflip-48-689"
X48, HALF48 = 6114, 3057
U_RUN, CHUNK1, CHUNK2 = 3280, 2496, 784
U_SET = 4584
SID = 7


def _read_full(d, rid=RID_RUN):
    return dxb.read_group_full(str(d), "D", SID, session_of=lambda pid: SID)[dxb.rid_key(rid)]


# ------------------------------------------------------------------------------------------------ rank side (beacon v4)
def test_live_48_687_chunk2_launched_before_chunk1_done_rest_3280_free_0(tmp_path):
    """The record D's ranks held at 07:53:14,400: chunk 1 (3280 left, 2496 in it) launched and its done NOT polled, chunk 2 (784)
    launched. ``rest`` stays 3280 (D still computes it: review A1), ``free`` = 0 (no token is left that no forward was launched
    for); chunk 1 ends -> rest 784, free 0; chunk 2 ends -> rest 0."""
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: 10**9)
    w.note_recv(rid=RID_RUN)
    ev1, ev2 = _Ev(), _Ev()
    w.note_extend(rids=[RID_RUN], rests=[U_RUN], extend_lens=[CHUNK1])
    w.arm_done(key=1, rids=[RID_RUN], ready=ev1.query)
    assert _read_full(tmp_path)[1:] == (U_RUN, U_RUN - CHUNK1)          # 07:53:11,134: free 784
    w.poll_done()
    w.note_extend(rids=[RID_RUN], rests=[CHUNK2], extend_lens=[CHUNK2])  # chunk 2's prefix already covers chunk 1
    w.arm_done(key=2, rids=[RID_RUN], ready=ev2.query)
    assert _read_full(tmp_path)[1:] == (U_RUN, 0)                        # 07:53:14,400
    reading = dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)
    assert dxb.rest_of(reading, RID_RUN, 0.5) == U_RUN and dxb.free_of(reading, RID_RUN, 0.5) == 0
    ev1.done = True
    assert w.poll_done() == 1 and _read_full(tmp_path)[1:] == (CHUNK2, 0)
    ev2.done = True
    assert w.poll_done() == 1 and _read_full(tmp_path)[1:] == (0, 0)


def test_live_46_680_chunk1_in_flight_free_is_what_p_computed_minus_the_store_page(tmp_path):
    """pdflip-46-680: u=5151, chunk 1 = 2304 in flight at the verdict -> free 2847; P computed 2911 = 2847 + the 64-token page the
    store holds page-aligned."""
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: 10**9)
    w.note_recv(rid="pdflip-46-680")
    w.note_extend(rids=["pdflip-46-680"], rests=[5151], extend_lens=[2304])
    w.arm_done(key=1, rids=["pdflip-46-680"], ready=_Ev().query)
    got = _read_full(tmp_path, "pdflip-46-680")
    assert got[1:] == (5151, 2847) and 2911 - got[2] == 64


def test_ranks_free_is_the_min_rest_is_the_max(tmp_path):
    """R11: rank 1 launched chunk 2 already, rank 2 not yet (ms apart; chunk boundaries are group-agreed, #794): free = MIN
    (0), rest = MAX (3280)."""
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: 10**9) for r in range(2)]
    for w in ws:
        w.note_recv(rid=RID_RUN)
        w.note_extend(rids=[RID_RUN], rests=[U_RUN], extend_lens=[CHUNK1])
        w.arm_done(key=1, rids=[RID_RUN], ready=_Ev().query)
    ws[0].note_extend(rids=[RID_RUN], rests=[CHUNK2], extend_lens=[CHUNK2])
    assert _read_full(tmp_path)[1:] == (U_RUN, 0)


def test_a_v3_file_stays_readable_free_falls_back_to_the_rest(tmp_path):
    """A v3 rank file (no ``free``): the reader keeps it, ``free`` = -1 and ``free_of`` = the rest (the pre-v4 figure)."""
    p = tmp_path / "D-pid1.bin"
    raw = bytearray(dxb._HDR_SIZE + 4 * struct.calcsize(dxb._REC_FMT_V2))
    struct.pack_into(dxb._HDR_FMT, raw, 0, dxb._MAGIC, 3, 4, 1, 1)
    struct.pack_into(dxb._REC_FMT_V2, raw, dxb._HDR_SIZE, dxb.rid_key(RID_RUN), 10**9, 1234, 1)
    p.write_bytes(bytes(raw))
    reading = dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)
    assert reading[dxb.rid_key(RID_RUN)] == (10**9, 1234, -1)
    assert dxb.free_of(reading, RID_RUN, 0.5) == 1234
    assert dxb.free_of({dxb.rid_key(RID_RUN): (10**9, 900)}, RID_RUN, 0.5) == 900     # a caller's 2-tuple stub
    assert dxb.free_of({}, RID_RUN, 0.5) is None


# ------------------------------------------------------------------------------------------------ the front's verdict (R6/R7)
def _front_48(now, tmp_path, *, chunk2_launched, waited):
    """The real front at epoch 48: D decodes, the running halb D prefill 48-687 (u=3280, class halb from its CLASSIFY, R3) with
    chunk 1 (2496) in flight, chunk 2 (784) launched or not; the halb set member 48-689 (u=4584) has waited ``waited`` s."""
    f = _front_live(now, tmp_path, member_u=U_SET, ended=False, rest_start=U_RUN, ext_len=CHUNK1, x=X48, klass="halb",
                    waited=waited, rid_run=RID_RUN, half_x=HALF48)
    if chunk2_launched:
        ws, _ev = f._test_running[RID_RUN]
        for w in ws:
            w.note_extend(rids=[RID_RUN], rests=[CHUNK2], extend_lens=[CHUNK2])
            w.arm_done(key=37870, rids=[RID_RUN], ready=_Ev().query)
    return f


def _verdicts(caplog):
    return [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]


def test_live_48_687_at_14_400_nothing_reaches_p_no_r6_flip_the_set_goes_to_d(monkeypatch, caplog, tmp_path):
    """Repro 07:53:14,400 (red before the fix: r6_rest=3280, S=7864 > X=6114 -> REROUTE_P + R6-REDIRECT counted 3280, P then
    computed 16). Exact: the whole rest of 48-687 is in launched forwards (free 0) -> S=4584 < X; T_halb is over (6.8 s > 4.5 s)
    -> the set goes to D (R3), no flip under X (R7), 48-687 is not redirected and finishes on D."""
    _on(monkeypatch)
    now = time.time()
    f = _front_48(now, tmp_path, chunk2_launched=True, waited=6.8)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        got = f._decode_collect(now)
    assert got != "P", got
    sv = _verdicts(caplog)
    assert sv and not any("verdict=REROUTE_P" in m or "verdict=FLIP_P" in m for m in sv), sv
    assert any("verdict=D_AFTER_TIMER" in m and "S=%d" % U_SET in m for m in sv), sv
    assert not any("R6-REDIRECT" in m or "PARK-HANDBACK" in m for m in caplog.messages)
    assert f.counters["r6_redirected"] == 0 and f.counters["r6_rest_in_flight"] >= 1
    assert any("R6-REST-IN-FLIGHT rid=%s d_rest=%d free=0" % (RID_RUN, U_RUN) in m for m in caplog.messages)


def test_live_48_687_at_11_134_only_the_unlaunched_784_counts_the_set_collects(monkeypatch, caplog, tmp_path):
    """Repro 07:53:11,134 (verdict_on_old=REROUTE_P on 3280, which started the 3.3 s re-probe): chunk 1 (2496) was launched 6 ms
    before -> what can still move to P is 784; S=4584+784=5368 < X=6114 and T_halb runs (3.5 s < 4.5 s) -> COLLECT (R3)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_48(now, tmp_path, chunk2_launched=False, waited=3.5)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == U_RUN - CHUNK1
        assert f._decode_collect(now) == "hold"
    assert not any("verdict=REROUTE_P" in m for m in _verdicts(caplog))
    assert f.counters["r6_redirected"] == 0


def test_r6k_kurz_running_is_still_not_counted_whatever_free_says(monkeypatch, caplog, tmp_path):
    """SOLL R6k unchanged: a running KURZ D prefill (class from the arrival) is not in S and not redirected -- the v4 ``free`` does
    not bring it back."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=U_SET, ended=False, rest_start=2000, ext_len=1000, x=X48, klass="kurz",
                    waited=3.5, rid_run=RID_RUN, half_x=HALF48)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
    assert f.counters["r6k_kurz_stays_on_d"] >= 1 and f.counters["r6_rest_in_flight"] == 0


def test_r1u_uses_the_counted_free_piece(monkeypatch, tmp_path):
    """R1u: the counted run's uncached piece is the LAST ``free`` tokens (``len(ids) - free``), what P computes of it."""
    _on(monkeypatch)
    now = time.time()
    f = _front_48(now, tmp_path, chunk2_launched=False, waited=3.5)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        f._dc_carried_running()
    assert f._r6_counted == {RID_RUN: U_RUN - CHUNK1}
    assert f._r6_in_flight == {RID_RUN: CHUNK1}
    assert "len(ia) - int(rest)" in __import__("inspect").getsource(F.Front._r1u_running)
