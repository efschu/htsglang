"""DASHBOARD-AUS-IPC, FEHLT-Liste (Inventar /spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md,
30.09.): the dashboard reads only IPC since 8c1e2f977b; these leaves were missing.

1. front.d_seats / d_phase_n: 27B null (its wakes carry no seat count) -> ``d_seats`` on every front.
2. rankstats.sched.park_window_left_ms: null on every rank -> the ms still open, -1 = no window.
3. flip_first_work: 27B 11 flip_done / 3 flip_first_work, NF 32 / 30 -> one per flip_done
   (any D content after a P->D flip counts, a resumed stream too; else ``what: "none"``).
4. D cached tokens split: P->D hand-off vs D's own prefix hit (front.d_cached_tokens, from the
   front's served rows D / D_after_P -- no new request field through D's API).
5. rankstats.cache.store_incomplete_delivered/_deliverable: the #1324 site's two numbers, summed.
6. rankstats.decode.last_bs: the batch size of the last round.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_dashipc_felder", os.path.join(os.path.dirname(__file__), "test_pdflip_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)
GIB, Z30U, _boot, _front, _of, _rpc = _d.GIB, _d.Z30U, _d._boot, _d._front, _d._of, _d._rpc

from flliper.srt.pdflip import front_state_ipc as fsi  # noqa: E402
from flliper.srt.pdflip import host_ledger  # noqa: E402
from flliper.srt.pdflip import park_window_gate as pwg  # noqa: E402
from flliper.srt.pdflip import rankstats  # noqa: E402


# ---------------------------------------------------------------- 1

def test_1_d_seats_on_a_front_whose_wakes_carry_no_seat_count():
    f = _front()
    f.d_bs = 6
    fields = f._ipc_front_fields()
    assert fields["d_phase_n"] is None                          # 27B / before the first wake
    assert fields["d_seats"] == {"n": 6, "cap": 6, "parked_n": 0, "source": "d_bs"}
    f._d_phase_n = 3
    assert f._ipc_front_fields()["d_seats"] == {"n": 3, "cap": 6, "parked_n": 0,
                                                "source": "wake_phase_seats"}


# ---------------------------------------------------------------- 2

def test_2_park_window_left_ms_is_a_value_on_every_rank():
    s = SimpleNamespace()
    assert rankstats._park_window_left_ms(s) == -1               # no window in force
    pwg.note(s, SimpleNamespace(left_ms=900, epoch=4, a_ms=10.0, b_ms=0.1, c_ms=0.001))
    t0 = s._pdflip_park_window["t_set"]
    assert rankstats._park_window_left_ms(s, now=t0 + 0.25) == 650
    assert rankstats._park_window_left_ms(s, now=t0 + 5.0) == 0  # never below 0
    pwg.note(s, SimpleNamespace(left_ms=-1, epoch=4))
    assert rankstats._park_window_left_ms(s) == -1
    s2 = SimpleNamespace(forward_ct=0, metrics_reporter=None, waiting_queue=[],
                         running_batch=SimpleNamespace(reqs=[]))
    sched = rankstats.scheduler_counters(s2)["sched"]
    assert sched["park_window_left_ms"] == -1 and sched["park_window_open"] is False


# ---------------------------------------------------------------- 3

def test_3_the_clock_pairs_every_done_flip():
    c = fsi.FirstWorkClock()
    assert c.arm(1, "P", "D", 10.0) is None
    c.done(12.0)
    ev = c.arm(2, "D", "P", 20.0)                                # D never worked
    assert (ev["epoch"], ev["dir"], ev["what"], ev["reason"], ev["flip_time_ms"], ev["flip_total_ms"]) == \
        (1, "P>D", "none", "next_flip_before_work", None, 2000)
    assert c.seen("P", "p_leg1_dispatch", "r", 21.0)["dir"] == "D>P"
    c.done(22.0)
    assert c.flush("front_stop") is None                         # already fired
    c.arm(3, "P", "D", 30.0)                                     # never reached done: nothing to pair
    assert c.flush("front_stop") is None
    c.arm(4, "P", "D", 40.0)
    c.done(41.5)
    assert c.flush("front_stop")["reason"] == "front_stop"


def test_3_the_front_publishes_one_first_work_per_flip_done():
    """A flip whose woken P never works before the front stops: base = 1
    flip_done, 0 flip_first_work; now the stop names it (``what: "none"``,
    ``reason: front_stop``, the time to the flip's end). The next-flip case is
    the clock test above."""
    sd = _boot(tempfile.mkdtemp(prefix="dashfeld-"))
    f = _front()
    f.rpc = _rpc
    f.p_leg1_stall_s = 0.0

    async def body():
        await f.flip("D", "P")          # P woken, no leg 1 follows
        await asyncio.sleep(0.2)

    with mock.patch.dict(os.environ, {"PDFLIP_STATE_DIR": sd}), \
            mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=dict(Z30U)), \
            mock.patch.object(host_ledger, "read_cgroup", return_value={"max": 84 * GIB}):
        asyncio.run(body())
        deadline = time.time() + 3.0
        while time.time() < deadline and len(_of(sd, "flip_done")) < 1:
            time.sleep(0.02)
        assert _of(sd, "flip_first_work") == []
        f.do_stop("TEST", "stop")
        deadline = time.time() + 3.0
        while time.time() < deadline and len(_of(sd, "flip_first_work")) < 1:
            time.sleep(0.02)
        time.sleep(0.2)
    done, fw = _of(sd, "flip_done"), _of(sd, "flip_first_work")
    assert len(done) == 1 and len(fw) == 1
    w = fw[0]["data"]
    assert (w["dir"], w["what"], w["reason"]) == ("D>P", "none", "front_stop")
    assert w["flip_time_ms"] is None and w["flip_total_ms"] >= 0
    assert w["flip_begin_ts"] == done[0]["data"]["flip_begin_ts"]


def test_3_a_none_event_never_carries_flip_total_as_the_flip_time():
    """User 29.09. (flipzeit-ist-p-ende-bis-erstes-decode): the flip time is P end ->
    first decode token (and decode end -> first prefill), NEVER flip_total. A ``none``
    event has no first work, so no flip time: ``flip_time_ms`` null, the begin->done span
    only as ``flip_total_ms``. The dashboard's B1/B2 tile (rigdash ipcfields
    _flip_first_work) and the history marks take every non-null ``flip_time_ms`` of the
    direction -- a flip_total there would sit in the flip-time median."""
    c = fsi.FirstWorkClock()
    rows = []
    c.arm(1, "P", "D", 10.0)
    c.done(13.0)
    rows.append(c.seen("D", "decode_token", "r1", 21.0))            # real: 11 s
    c.arm(2, "D", "P", 30.0)
    c.done(32.0)
    rows.append(c.arm(3, "P", "D", 40.0))                          # D>P none (flip_total 2 s)
    c.done(43.0)
    rows.append(c.flush("front_stop"))                             # P>D none (flip_total 3 s)
    assert [r["what"] for r in rows] == ["decode_token", "none", "none"]
    p2d_ms = [r["flip_time_ms"] for r in rows if r["dir"] == "P>D" and r.get("flip_time_ms") is not None]
    d2p_ms = [r["flip_time_ms"] for r in rows if r["dir"] == "D>P" and r.get("flip_time_ms") is not None]
    assert p2d_ms == [11000] and d2p_ms == []
    assert [r.get("flip_total_ms") for r in rows[1:]] == [2000, 3000]


def test_3_any_d_content_after_the_flip_is_first_work():
    import inspect

    from flliper.srt.pdflip import front as front_mod

    src = inspect.getsource(front_mod.Front.leg2)
    i = src.index("async def _write_client(chunk: bytes)")
    # FW-PING (01.10.): the chunk goes along -- D's keepalive ping is no work
    assert 'self._ipc_first_work_seen("D", "decode_token", rid,' in src[i:i + 600]
    assert "chunk=chunk, path=request.path)" in src[i:i + 700]
    # 27B (flip time = P end -> first decode token): a 503 D streams (state
    # refusal, the stream branch forwards r.status) is no decode token -- the same
    # status-200 guard the leg's first-content site has
    assert "if chunk and r.status == 200:" in src[i:i + 200]


# ---------------------------------------------------------------- 4

def test_4_d_cached_tokens_split_handoff_from_prefix_hit():
    f = _front()
    f._ipc_note_served_d(SimpleNamespace(leg1_ran=True), 60000, 59000, 20, None)   # hand-off
    f._ipc_note_served_d(SimpleNamespace(leg1_ran=False), 3000, 2500, 20, None)    # D's own prefix
    f._ipc_note_served_d(None, 500, 0, 10, None)                                   # d-direct, cold
    assert f._ipc_front_fields()["d_cached_tokens"] == {"total": 61500, "handoff": 59000,
                                                        "d_prefix_hit": 2500}


# ---------------------------------------------------------------- 5

def test_5_store_incomplete_delivered_and_deliverable():
    s = SimpleNamespace(_pdflip_store_short_seen=2, _pdflip_store_short_delivered=12544 + 4095,
                        _pdflip_store_short_deliverable=25920 + 97870)
    c = rankstats._cache_block(s)
    assert (c["store_incomplete_n"], c["store_incomplete_delivered"], c["store_incomplete_deliverable"]) == \
        (2, 16639, 123790)
    assert rankstats._cache_block(SimpleNamespace())["store_incomplete_delivered"] == 0


def test_5_the_scheduler_site_sums_both():
    import inspect

    from flliper.srt.managers import scheduler

    src = inspect.getsource(scheduler.Scheduler)
    i = src.index('self._pdflip_store_short_seen = getattr(self, "_pdflip_store_short_seen", 0) + 1')
    blk = src[i:i + 600]
    assert "_pdflip_store_short_delivered" in blk and "_pdflip_store_short_deliverable" in blk


# ---------------------------------------------------------------- 6

def test_6_decode_last_bs():
    drl = SimpleNamespace(cum_rounds=5, cum_gpu_ms=100.0, cum_by_bs={2: [3, 60.0], 3: [2, 40.0]}, last_bs=3)
    mr = SimpleNamespace(decode_round_log=drl, gen_tokens_total=0, last_running_reqs=4)
    d = rankstats._decode_block(mr)
    assert d["last_bs"] == 3 and d["running"] == 4
    drl.last_bs = None
    assert rankstats._decode_block(mr)["last_bs"] is None


def test_6_the_round_log_records_the_last_bs():
    import inspect

    from flliper.srt.managers.scheduler_components import decode_round_log as D

    src = inspect.getsource(D.DecodeRoundLog)
    i = src.index("slot_bs = self.cum_by_bs.get(acc.bs)")
    assert "self.last_bs = acc.bs" in src[i - 80:i]
    assert "self.last_bs = None" in inspect.getsource(D.DecodeRoundLog.__init__)
