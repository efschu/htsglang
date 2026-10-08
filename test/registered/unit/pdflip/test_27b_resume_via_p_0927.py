"""RESUME-VIA-P (27B rc12k27 b23, 27.09.): a streamed request D refuses with W50
after it has generated tokens is kept parked on D; P prefills its context; D
continues the same stream (pdflip/resume_via_p.py, scheduler
``_pdflip_answer_x_refusals``, front ``_rvp_*``).

Metal: pdflip-24-100 (745 s into its stream), pdflip-58-208, pdflip-46-165 -- each
parked, resumed with its prefix gone (state=cold host_hit=0), X gate
uncached=80508 > X=12288 -> W50 in-band after the first byte -> client dead.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_park_read as pr  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    # ROS (27.09. NF): this file pins the output>0 rule it was written for;
    # the open-stream rule has its own file (test_27b_resume_open_stream_0927).
    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
    return tmp_path


def _req(rid="pdflip-24-100", out=302, stream=True, prompt=115):
    r = types.SimpleNamespace(rid=rid, stream=stream, origin_input_ids=list(range(prompt)),
                              output_ids=list(range(1000, 1000 + out)), kv_arrival_seq=1,
                              is_fast_lane=False, spill_class=None)
    r.full_untruncated_fill_ids = r.origin_input_ids + r.output_ids
    return r


def test_eligible_is_a_streamed_request_with_output_on_group_d(d_env, monkeypatch):
    assert rvp.eligible(_req())
    assert not rvp.eligible(_req(out=0)), "fresh: the front re-routes it before its first byte"
    assert not rvp.eligible(_req(stream=False)), "non-stream: nothing reached the client -> X-REQUEUE"
    r = _req()
    setattr(r, rvp.ATTEMPTS_ATTR, rvp.MAX_ATTEMPTS)
    assert not rvp.eligible(r), "bounded: the n-th refusal ends by name"
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert not rvp.eligible(_req())
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv(rvp.ENV, "0")
    assert not rvp.eligible(_req())


def test_write_and_take_round_trip(d_env):
    d = rvp.needs_p_dir()
    assert d == os.path.join(str(d_env), "needs-p")
    assert rvp.needs_p_dir(env={}, tag="t1") == "/dev/shm/pdflip-arena-t1/needs-p"
    p = rvp.write_request("pdflip-24-100", [1, 2, 3], 80508, 12288, "x_refusal_midstream")
    assert p and os.path.exists(p)
    got = rvp.take_requests(d)
    assert got[0]["rid"] == "pdflip-24-100" and got[0]["input_ids"] == [1, 2, 3] and got[0]["d_extent"] == 80508
    assert rvp.take_requests(d) == [], "consumed"
    with open(os.path.join(d, "bad.json"), "w") as f:
        f.write("{")
    assert rvp.take_requests(d) == [] and not os.listdir(d), "unreadable: removed, skipped"


def _sched(rank=0):
    return types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=rank), pdflip_d_parked=[])


def test_keep_on_d_parks_clears_the_old_cycle_and_writes_on_rank0_only(d_env):
    r = _req()
    setattr(r, pr.CAP_ATTR, (256, 417))
    r._pdflip_store_delivered = 79103
    s0 = _sched(0)
    assert rvp.keep_on_d(s0, r, 80508, 12288)
    assert s0.pdflip_d_parked == [r] and ds.park_site(r) == ds.SITE_FLIP
    assert getattr(r, pr.CAP_ATTR) is None and r._pdflip_store_delivered is None
    assert getattr(r, rvp.ATTEMPTS_ATTR) == 1
    got = rvp.take_requests(rvp.needs_p_dir())
    assert got[0]["input_ids"] == r.full_untruncated_fill_ids and got[0]["d_extent"] == 80508
    r1 = _req()
    s1 = _sched(1)
    assert rvp.keep_on_d(s1, r1, 80508, 12288)
    assert s1.pdflip_d_parked == [r1] and rvp.take_requests(rvp.needs_p_dir()) == [], "rank 1 writes nothing"
    rvp.keep_on_d(s0, r, 80508, 12288)
    assert s0.pdflip_d_parked == [r], "parked once"


# -- the scheduler's W31 answer ------------------------------------------------


class _Chan:
    def __init__(self):
        self.sent = []

    def send_output(self, out, req):
        self.sent.append((out, req))


def _x_sched(queue):
    chan = _Chan()
    released = []
    ns = types.SimpleNamespace(
        server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288), waiting_queue=list(queue),
        tree_cache=types.SimpleNamespace(release_aborted_request=lambda rid: released.append(rid)),
        enable_hicache_storage=True, enable_hierarchical_cache=False,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=chan), ps=types.SimpleNamespace(tp_rank=0),
        pdflip_d_parked=[], pdflip_uncached_extent=lambda req, head=None: 80508,
    )
    return ns, chan, released


def test_scheduler_keeps_the_midstream_request_and_aborts_the_fresh_one(d_env, monkeypatch):
    from flliper.srt.managers import scheduler as S

    monkeypatch.setattr(S, "release_admission_acquired_mamba_slot", lambda req, tc, site: False)
    mid, fresh = _req("pdflip-24-100"), _req("pdflip-70-300", out=0)
    for r in (mid, fresh):
        r.time_stats = types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **kw: None))
    ns, chan, released = _x_sched([mid, fresh])
    S.Scheduler._pdflip_answer_x_refusals(ns, [mid, fresh])
    assert ns.waiting_queue == []
    assert [q.rid for _o, q in chan.sent] == ["pdflip-70-300"], "only the fresh one is answered W50"
    assert "W50 PdFlipTpPrefillExceeded" in chan.sent[0][0].finished_reason["message"]
    assert ns.pdflip_d_parked == [mid] and released == ["pdflip-24-100", "pdflip-70-300"]
    assert rvp.take_requests(rvp.needs_p_dir())[0]["rid"] == "pdflip-24-100"


def test_scheduler_switch_off_is_the_abort(d_env, monkeypatch):
    from flliper.srt.managers import scheduler as S

    monkeypatch.setenv(rvp.ENV, "0")
    monkeypatch.setattr(S, "release_admission_acquired_mamba_slot", lambda req, tc, site: False)
    mid = _req()
    mid.time_stats = types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **kw: None))
    ns, chan, _ = _x_sched([mid])
    S.Scheduler._pdflip_answer_x_refusals(ns, [mid])
    assert [q.rid for _o, q in chan.sent] == ["pdflip-24-100"] and ns.pdflip_d_parked == []


# -- the front ------------------------------------------------------------------


def _front(tmp_path):
    from flliper.srt.pdflip import front as F

    ns = types.SimpleNamespace(tag="dkrtest", queue=[], counters={"rvp_rerouted": 0, "rvp_resumed": 0},
                               kicks=[])
    ns._kick_controller = lambda why: ns.kicks.append(why)
    for name in ("_rvp_state", "_note_front_price", "_rvp_take", "_rvp_p_finished", "_rvp_resumed"):
        setattr(ns, name, getattr(F.Front, name).__get__(ns))
    return F, ns


def test_front_queues_a_p_only_leg1_and_closes_the_record(d_env, caplog):
    F, ns = _front(d_env)
    ns._note_front_price("pdflip-24-100", 34)
    rvp.write_request("pdflip-24-100", list(range(80510)), 80508, 12288, "x_refusal_midstream")

    async def body():
        with caplog.at_level("INFO"):
            assert ns._rvp_take() == 1
            p = ns.queue[0]
            assert p.resume_via_p and p.p_only and p.path == "/generate" and p.rid == "pdflip-24-100"
            assert p.payload["input_ids"][-1] == 80509 and p.payload["sampling_params"]["max_new_tokens"] == 1
            assert p.est_uncached == 80508 and p.est_prompt == 80510
            assert ns.kicks == ["arrival"]
            p.leg1_done = True  # ROS-1P: p-done only for a leg that succeeded
            ns._rvp_p_finished(p)
            assert p.fut.done() and "pdflip-24-100" in ns._rvp_p_done
            ns._rvp_resumed("pdflip-24-100")
            ns._rvp_resumed("pdflip-24-100")  # the second chunk logs nothing
        msgs = caplog.messages
        assert any("PDFLIP W50-REROUTE rid=pdflip-24-100 front_price=34 d_extent=80508 reason=x_refusal_midstream "
                   "path=midstream" in m for m in msgs)
        assert sum("RESUME-VIA-P done rid=pdflip-24-100 p_ms=" in m and "d_resume_ms=" in m for m in msgs) == 1

    asyncio.new_event_loop().run_until_complete(body())


def test_front_switch_off_takes_nothing(d_env, monkeypatch):
    monkeypatch.setenv(rvp.ENV, "0")
    F, ns = _front(d_env)
    rvp.write_request("pdflip-24-100", [1, 2], 80508, 12288, "x")
    assert ns._rvp_take() == 0 and ns.queue == []


def test_the_immediate_park_trigger_sees_the_p_only_leg1():
    from flliper.srt.pdflip import phase_policy as pp

    p = types.SimpleNamespace(rid="r", est_uncached=10, p_only=True, x_requeues=0, leg1_done=False,
                              skip_leg1=False, x_deferred=False)
    assert pp.immediate_park_trigger([p], 12288) is p


def test_wiring_front_hooks():
    src = open(os.path.join(os.path.dirname(rvp.__file__), "front.py"), encoding="utf-8").read()
    assert "self._rvp_take()\n                if self.awake == \"D\":" in src
    done = src.index("def _on_leg1_done(p: Pending) -> None:")
    assert src.index("if p.resume_via_p:", done) < src.index("self._ready_for_d.append(p)", done)
    assert 'if getattr(self, "_rvp_p_done", None) and rid in self._rvp_p_done:' in src
    assert "path=fresh" in src and "self._note_front_price(rid, remainder)" in src
