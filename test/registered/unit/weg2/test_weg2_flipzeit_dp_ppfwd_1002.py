"""FLIPZEIT D>P END = the first forward on the FIRST P pipeline stage (user order 02.10.,
NF + 27B identical names).

FLIPZEIT runs from the last token of the outgoing phase to the first token of the incoming
one ("... zu erstes Token Decode oder PREFILL BATCH BEGINN"). For D>P the end is the BEGIN OF
THE FIRST FORWARD ON THE FIRST P PIPELINE STAGE (PP0) after the wake -- not the P leg-1
dispatch, and not the PP-last stage: PP0 -> PP1 -> PP2 of the first chunk is pipeline fill,
i.e. prefill (y7l: ~6.5 s). The front reads it off the progress beacon of the PP0 rank
(``<arena>/progress/P-pid<pid>.bin``, ``t_start_ns``) at the first ``forward_ct`` rise after
``flip_done``: ``flip_user_time.prefill_start_ts`` with ``prefill_start_source="pp_first_forward"``,
the same end in the D>P ``flip_first_work`` and in the front's flip phase (front.flip Nachlauf,
the dashboard's / VictoriaMetrics' weg2_flip_user_view_ms input). The PP-last stage's first
forward rides along as ``flip_user_time.pp_last_start_ts``. Without the beacon:
``prefill_start_source="missing"``, ``prefill_start_ts`` None, no flip time -- never the
leg-1 dispatch (before: ``leg1_dispatch`` / ``leg1_end_minus_p_prefill_s``).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import struct
import subprocess
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import front_requests as frq  # noqa: E402
from sglang.srt.weg2 import front_state_ipc as fsi  # noqa: E402
from sglang.srt.weg2 import host_ledger  # noqa: E402
from sglang.srt.weg2 import progress_beacon as fp  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "_t_dashipc_ppfwd", os.path.join(os.path.dirname(__file__), "test_weg2_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

FMT = "<qqqq"


def _beat(d, group, pid, ct, ts_ns, td_ns=0):
    with open(os.path.join(d, f"{group}-pid{pid}.bin"), "wb") as fh:
        fh.write(struct.pack(FMT, int(ct), int(ts_ns), int(td_ns), int(pid)))


def _ranks(titles):
    """Real processes whose title is a scheduler title (exec -a), in this session."""
    procs = [subprocess.Popen(["bash", "-c", f'exec -a "{t}" sleep 60']) for t in titles]
    deadline = time.time() + 5.0
    for p in procs:
        while time.time() < deadline:
            try:
                with open(f"/proc/{p.pid}/cmdline", "rb") as fh:
                    if fh.read().startswith(b"sglang::"):
                        break
            except OSError:
                pass
            time.sleep(0.01)
    return procs


def _kill(procs):
    for p in procs:
        p.kill()
        p.wait()


# ---------------------------------------------------------------- the beacon reading

def test_pp_rank_comes_from_the_scheduler_title():
    procs = _ranks(["sglang::scheduler_PP2_TP0", "sglang::scheduler_TP1", "sglang::scheduler_DP0_PP1_TP0"])
    try:
        assert [fp.pp_rank_of(p.pid) for p in procs] == [2, 0, 1]
    finally:
        _kill(procs)
    assert fp.pp_rank_of(os.getpid()) is None          # no scheduler
    assert fp.pp_rank_of(2 ** 22 + 12345) is None      # no process


def test_first_and_last_stage_by_pp_rank_and_unknown_stays_unknown():
    stages = {10: 0, 11: 0, 12: 1, 13: 2, 14: 2}
    st, why = fp.pp_stage_pids({p: (0, 0, 0) for p in stages}, stages.get)
    assert (st, why) == ({"first": {10: 0, 11: 0}, "last": {13: 2, 14: 2}}, None)
    st, why = fp.pp_stage_pids({7: (0, 0, 0)}, {7: 0}.get)       # pp_size 1: one stage is both
    assert st == {"first": {7: 0}, "last": {7: 0}}
    assert fp.pp_stage_pids({}, stages.get) == ({}, "no_beacon")
    assert fp.pp_stage_pids({11: (0, 0, 0), 99: (0, 0, 0)}, stages.get)[1] == "pp_rank_unknown pid=99"


def test_first_rise_is_the_first_forward_or_missing_never_a_later_one():
    base = {13: 40, 14: 40}
    assert fp.first_rise(base, {13: (40, 5, 6), 14: (40, 5, 6)}) is None
    r = fp.first_rise(base, {13: (41, 2_000_000_000, 0), 14: (41, 1_500_000_000, 0)})
    assert (r["ts"], r["pid"], r["ct"]) == (1.5, 14, 41)              # TP ranks: the earliest start
    # a reading after a SECOND forward: the first start is overwritten -> missing, not a value
    r = fp.first_rise({13: 40}, {13: (42, 9_000_000_000, 0)})
    assert r == {"missing": "late_read pid=13 forward_ct 40->42"}


def test_the_probe_thread_reports_pp0_as_the_end_and_pp_last_alongside_once_each():
    d = tempfile.mkdtemp(prefix="ppfwd-probe-")
    stages = {101: 0, 102: 1, 103: 2}
    for pid in stages:
        _beat(d, "P", pid, 7, 1_000_000_000, 1_100_000_000)
    got = []
    probe = fp.PpForwardProbe(d, "P", 77, session_of=lambda pid: 77, pp_of=stages.get, poll_s=0.001)
    assert probe.result is None and probe.baseline == {"first": {101: 7}, "last": {103: 7}}
    probe.start(lambda stage, res: got.append((stage, res)))
    _beat(d, "P", 102, 8, 4_000_000_000)                       # a middle stage is neither
    time.sleep(0.05)
    assert got == [] and probe.result is None
    _beat(d, "P", 101, 8, 5_000_000_000)                       # PP0 begins: the D>P end
    deadline = time.time() + 2.0
    while time.time() < deadline and not got:
        time.sleep(0.005)
    assert got == [("first", {"ts": 5.0, "pid": 101, "ct": 8, "pp_rank": 0})]
    assert probe.result["ts"] == 5.0 and not probe.done()
    _beat(d, "P", 103, 8, 6_250_000_000)                       # the last stage after the fill
    while time.time() < deadline and len(got) < 2:
        time.sleep(0.005)
    assert got[1] == ("last", {"ts": 6.25, "pid": 103, "ct": 8, "pp_rank": 2})
    probe.stop("x")
    assert probe.result["ts"] == 5.0 and len(got) == 2


def test_no_beacon_is_missing_at_once():
    p = fp.PpForwardProbe("", "P", 77)
    assert p.results == {"first": {"missing": "no_beacon_dir"}, "last": {"missing": "no_beacon_dir"}}
    p = fp.PpForwardProbe(tempfile.mkdtemp(), "P", 77, session_of=lambda pid: 77)
    assert p.result == {"missing": "no_beacon"} and p.done()
    with mock.patch.dict(os.environ, {fp.ENV: "0"}):
        assert fp.PpForwardProbe("/x", "P", 77).result["missing"].startswith("beacon_off")


# ---------------------------------------------------------------- the clocks

def _clock():
    c = fsi.DpFlipClock()
    c.note_park(5, 100.0, 660.0)
    c.begin(5, 100.7, oldest_waiter_ts=95.0)
    c.done(103.2)
    return c


def test_flip_user_time_ends_at_pp0s_first_forward_and_carries_pp_last():
    ev = _clock().first_prefill("weg2-6-1", 103.25, {"ts": 103.9, "pid": 101, "ct": 8, "pp_rank": 0},
                                {"ts": 110.4, "pid": 103, "ct": 8, "pp_rank": 2})
    assert (ev["prefill_start_source"], ev["prefill_start_ts"], ev["flip_user_ms"]) == \
        ("pp_first_forward", 103.9, 3900)
    assert ev["parts"]["first_chunk_ms"] == 700 and ev["parts"]["legs_ms"] == 2500
    assert (ev["prefill_start_pid"], ev["prefill_start_pp_rank"], ev["leg1_dispatch_ts"]) == (101, 0, 103.25)
    # the pipeline fill is prefill, outside the flip time
    assert (ev["pp_last_start_ts"], ev["pp_last_start_pp_rank"], ev["pp_fill_ms"]) == (110.4, 2, 6500)
    ev = _clock().first_prefill("r", 103.25, {"ts": 103.9}, {"missing": "late_read pid=3 forward_ct 1->3"})
    assert (ev["flip_user_ms"], ev["pp_last_start_ts"], ev["pp_fill_ms"]) == (3900, None, None)
    assert ev["pp_last_start_missing"].startswith("late_read")


def test_without_the_reading_there_is_no_flip_time_and_no_dispatch_fallback():
    for res, why in ((None, "no_probe"), ({"missing": "late_read pid=1 forward_ct 1->3"}, "late_read")):
        ev = _clock().first_prefill("r", 103.25, res)
        assert (ev["prefill_start_source"], ev["prefill_start_ts"], ev["flip_user_ms"]) == ("missing", None, None)
        assert ev["parts"]["first_chunk_ms"] is None and ev["prefill_start_missing"].startswith(why)
        assert ev["leg1_dispatch_ts"] == 103.25                 # named, never the end


def test_flip_first_work_carries_the_same_end():
    c = fsi.FirstWorkClock()
    c.arm(6, "D", "P", 100.7)
    c.done(103.2)
    ev = c.seen("P", "p_leg1_dispatch", "r", 103.25)
    out = fsi.FirstWorkClock.dp_end(ev, {"ts": 103.9, "pid": 101, "ct": 8, "pp_rank": 0})
    assert (out["prefill_start_source"], out["prefill_start_ts"], out["first_work_ts"], out["flip_time_ms"],
            out["leg1_dispatch_ts"]) == ("pp_first_forward", 103.9, 103.9, 3200, 103.25)
    out = fsi.FirstWorkClock.dp_end(ev, {"missing": "no_beacon"})
    assert (out["prefill_start_source"], out["first_work_ts"], out["flip_time_ms"]) == ("missing", None, None)


def test_the_flip_phase_nachlauf_ends_at_the_forward_not_at_the_call():
    f = frq.FlipPhase()
    f.layer("D>P", 100.7, "x")
    f.done(103.2)
    f.first_work(104.5, "p_leg1_dispatch", at=103.9)
    assert (f.snap["phase"], f.snap["last"]["nachlauf_ms"], f.snap["last"]["first_work_ts"]) == (None, 700, 103.9)
    f.layer("D>P", 200.0, "x")
    f.done(201.0)
    f.first_work(202.0, "p_leg1_dispatch", at=None)            # missing: closed, no value
    assert (f.snap["phase"], f.snap["last"]["nachlauf_ms"], f.snap["last"]["first_work_ts"]) == (None, None, None)


# ---------------------------------------------------------------- the front

class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _run_dp_flip(arena, on_post):
    sd = _d._boot(tempfile.mkdtemp(prefix="ppfwd-front-"))
    f = _d._front()
    f.rpc = _d._rpc
    f.p_leg1_stall_s = 0.0
    f.groups["P"].sid = os.getsid(0)
    body = json.dumps({"usage": {"prompt_tokens": 5000, "completion_tokens": 1,
                                 "prompt_tokens_details": {"cached_tokens": 0}},
                       "sglext": {"weg2_prefill_s": 0.25}}).encode()

    def post(url, json=None):
        on_post()
        return _Resp(200, body)

    f.session = SimpleNamespace(post=post)
    p = front_mod.Pending(rid="weg2-1-1", path="/v1/chat/completions", payload={"messages": []}, text="x",
                          t_arrive=time.time(), fut=None)

    async def run():
        f._ipc_dp_clock().note_park(f.epoch, time.time() - 0.4, 400.0)
        await f.flip("D", "P")
        await f.leg1(p)
        await asyncio.sleep(0.2)

    env = {"WEG2_STATE_DIR": sd, "SGLANG_HICACHE_ARENA_DIR": arena}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=dict(_d.Z30U)), \
            mock.patch.object(host_ledger, "read_cgroup", return_value={"max": 84 * _d.GIB}):
        asyncio.run(run())
        deadline = time.time() + 3.0
        while time.time() < deadline and not (_d._of(sd, "flip_user_time") and _d._of(sd, "flip_first_work")):
            time.sleep(0.02)
    return f, sd


def test_the_front_ends_dp_at_pp0s_first_forward_with_pp_last_alongside():
    arena = tempfile.mkdtemp(prefix="ppfwd-arena-")
    bd = os.path.join(arena, "progress")
    os.makedirs(bd)
    procs = _ranks(["sglang::scheduler_PP0", "sglang::scheduler_PP1", "sglang::scheduler_PP2"])
    pids = [p.pid for p in procs]
    for pid in pids:
        _beat(bd, "P", pid, 30, 1, 2)                             # forwards of the previous P phase
    seen = {}

    def on_post():                                                # P computes: stage 0 first, then the last
        seen["dispatch"] = time.time()
        seen["pp0_ns"] = time.time_ns() + 5_000_000                # PP0 begins 5 ms after the dispatch
        _beat(bd, "P", pids[0], 31, seen["pp0_ns"])
        time.sleep(0.02)                                           # the probe thread sees PP0 alone
        seen["pp_last_ns"] = seen["pp0_ns"] + 30_000_000           # +30 ms: the pipeline fill
        _beat(bd, "P", pids[1], 31, seen["pp0_ns"] + 15_000_000)
        _beat(bd, "P", pids[2], 31, seen["pp_last_ns"])

    try:
        f, sd = _run_dp_flip(arena, on_post)
    finally:
        _kill(procs)
    want = round(seen["pp0_ns"] / 1e9, 3)
    ut = [e["data"] for e in _d._of(sd, "flip_user_time")]
    assert len(ut) == 1
    assert (ut[0]["prefill_start_source"], ut[0]["prefill_start_ts"]) == ("pp_first_forward", want)
    assert (ut[0]["prefill_start_pid"], ut[0]["prefill_start_pp_rank"]) == (pids[0], 0)
    assert (ut[0]["pp_last_start_ts"], ut[0]["pp_last_start_pp_rank"]) == (round(seen["pp_last_ns"] / 1e9, 3), 2)
    assert abs(ut[0]["pp_fill_ms"] - 30) <= 1
    assert abs(ut[0]["flip_user_ms"] - (want - ut[0]["start_ts"]) * 1000.0) <= 2  # both ends rounded to ms
    assert ut[0]["leg1_dispatch_ts"] < want                    # the dispatch is not the end
    fw = [e["data"] for e in _d._of(sd, "flip_first_work") if e["data"]["dir"] == "D>P"]
    assert len(fw) == 1
    assert (fw[0]["prefill_start_source"], fw[0]["prefill_start_ts"], fw[0]["first_work_ts"]) == \
        ("pp_first_forward", want, want)
    last = front_mod.Front._flip_phase(f).snap["last"]
    assert last["first_work_ts"] == want and last["nachlauf_ms"] is not None


def test_the_front_without_a_beacon_reports_missing_not_the_dispatch():
    f, sd = _run_dp_flip(tempfile.mkdtemp(prefix="ppfwd-empty-"), lambda: None)  # no beacon files
    ut = [e["data"] for e in _d._of(sd, "flip_user_time")]
    assert len(ut) == 1
    assert (ut[0]["prefill_start_source"], ut[0]["prefill_start_ts"], ut[0]["flip_user_ms"]) == \
        ("missing", None, None)
    fw = [e["data"] for e in _d._of(sd, "flip_first_work") if e["data"]["dir"] == "D>P"]
    assert len(fw) == 1
    assert (fw[0]["prefill_start_source"], fw[0]["first_work_ts"], fw[0]["flip_time_ms"]) == ("missing", None, None)
    last = front_mod.Front._flip_phase(f).snap["last"]
    assert last["first_work_ts"] is None and last["nachlauf_ms"] is None
