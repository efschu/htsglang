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
    "_t_dashipc_felder", os.path.join(os.path.dirname(__file__), "test_weg2_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)
GIB, Z30U, _boot, _front, _of, _rpc = _d.GIB, _d.Z30U, _d._boot, _d._front, _d._of, _d._rpc

from sglang.srt.weg2 import front_state_ipc as fsi  # noqa: E402
from sglang.srt.weg2 import host_ledger  # noqa: E402
from sglang.srt.weg2 import park_window_gate as pwg  # noqa: E402
from sglang.srt.weg2 import rankstats  # noqa: E402


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
    t0 = s._weg2_park_window["t_set"]
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
    assert (ev["epoch"], ev["dir"], ev["what"], ev["reason"], ev["flip_time_ms"]) == \
        (1, "P>D", "none", "next_flip_before_work", 2000)
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

    with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}), \
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
    assert w["flip_time_ms"] >= 0 and w["flip_begin_ts"] == done[0]["data"]["flip_begin_ts"]


def test_3_any_d_content_after_the_flip_is_first_work():
    import inspect

    from sglang.srt.weg2 import front as front_mod

    src = inspect.getsource(front_mod.Front.leg2)
    i = src.index("async def _write_client(chunk: bytes)")
    assert 'self._ipc_first_work_seen("D", "decode_token", rid)' in src[i:i + 500]


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
    s = SimpleNamespace(_weg2_store_short_seen=2, _weg2_store_short_delivered=12544 + 4095,
                        _weg2_store_short_deliverable=25920 + 97870)
    c = rankstats._cache_block(s)
    assert (c["store_incomplete_n"], c["store_incomplete_delivered"], c["store_incomplete_deliverable"]) == \
        (2, 16639, 123790)
    assert rankstats._cache_block(SimpleNamespace())["store_incomplete_delivered"] == 0


def test_5_the_scheduler_site_sums_both():
    import inspect

    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler.Scheduler)
    i = src.index('self._weg2_store_short_seen = getattr(self, "_weg2_store_short_seen", 0) + 1')
    blk = src[i:i + 600]
    assert "_weg2_store_short_delivered" in blk and "_weg2_store_short_deliverable" in blk


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

    from sglang.srt.managers.scheduler_components import decode_round_log as D

    src = inspect.getsource(D.DecodeRoundLog)
    i = src.index("slot_bs = self.cum_by_bs.get(acc.bs)")
    assert "self.last_bs = acc.bs" in src[i - 80:i]
    assert "self.last_bs = None" in inspect.getsource(D.DecodeRoundLog.__init__)
