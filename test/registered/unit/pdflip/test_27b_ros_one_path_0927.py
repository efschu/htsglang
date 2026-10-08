"""ROS-1P (NF rc12p, dkrnfh91dprbar1dauer09271422): exactly ONE path per rid.

pdflip-9-39: held-uncommitted -> X-REQUEUE, D kept its park and streamed 154 tokens on the old
stream while the requeued leg 2 answered an empty 200 (W28). pdflip-10-44 / pdflip-10-42: P-only legs
kept starting after the stream ended (3x / 2x, 44.6 s of P), 13 of 19 p-done followed a failed
leg ("Duplicate request ID"), and WAIT-BOUND named pdflip-10-42 2 min after it was served.
"""
from __future__ import annotations

import asyncio
import collections
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.delenv(rvp.ENV_OPEN_STREAM, raising=False)
    return tmp_path


def _front():
    from flliper.srt.pdflip import front as F

    ns = types.SimpleNamespace(tag="dkrtest", queue=[], counters=collections.Counter(), kicks=[],
                               aborts=[])
    ns._kick_controller = lambda why: ns.kicks.append(why)

    async def rpc(g, path, body, timeout):
        ns.aborts.append((g, path, body))
        return 200, "ok"

    ns.rpc = rpc
    ns.groups = {"D": "D-group"}
    for name in ("_rvp_state", "_note_front_price", "_rvp_take", "_rvp_p_finished", "_rvp_resumed",
                 "_ros_lookahead_begin", "_ros_lookahead_end", "_rvp_stream_begin",
                 "_rvp_stream_end", "_rvp_abort_d"):
        setattr(ns, name, getattr(F.Front, name).__get__(ns))
    ns._rvp_state()
    return F, ns


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_pdflip_9_39_held_uncommitted_takes_the_one_p_path_no_requeue(d_env):
    _F, ns = _front()

    async def body():
        ns._ros_lookahead_begin("pdflip-9-39")
        rvp.write_request("pdflip-9-39", list(range(10)), 18000, 12288, "x_refusal_midstream")
        assert ns._rvp_take() == 1
        assert ns._rvp_requeue == {}, "no X-REQUEUE record -> no second path"
        assert [q.rid for q in ns.queue] == ["pdflip-9-39"]
        assert ns._ros_lookahead_end("pdflip-9-39") is None

    _run(body())


def test_pdflip_10_44_one_p_leg_per_rid_and_none_after_the_stream_ended(d_env):
    _F, ns = _front()

    async def body():
        for _ in range(3):  # D wrote needs-p three times (MAX_ATTEMPTS)
            rvp.write_request("pdflip-10-44", list(range(10)), 20000, 12288, "x_refusal_midstream")
            ns._rvp_take()
        assert [q.rid for q in ns.queue] == ["pdflip-10-44"], "one P-only leg, not three"
        assert ns.counters["rvp_duplicate_dropped"] == 2
        ns._rvp_stream_end("pdflip-10-44")  # the client stream ended
        assert ns.queue == [] and "pdflip-10-44" not in ns._rvp_inflight
        rvp.write_request("pdflip-10-44", list(range(10)), 20000, 12288, "x_refusal_midstream")
        assert ns._rvp_take() == 1 and ns.queue == [], "no P-only leg for an ended stream"
        assert ns.counters["rvp_after_end_dropped"] == 1
        ns._rvp_stream_begin("pdflip-10-44")  # a fresh leg 2 of the rid (X-REQUEUE)
        rvp.write_request("pdflip-10-44", list(range(10)), 20000, 12288, "x_refusal_midstream")
        ns._rvp_take()
        assert [q.rid for q in ns.queue] == ["pdflip-10-44"]

    _run(body())


def test_stream_end_drops_the_needs_p_file_and_records(d_env):
    _F, ns = _front()
    rvp.write_request("pdflip-10-42", [1, 2], 1, 1, "x")
    ns._rvp_p_done["pdflip-10-42"] = (0.0, 1.0)
    ns._rvp_stream_end("pdflip-10-42")
    assert not os.path.exists(os.path.join(rvp.needs_p_dir(), "pdflip-10-42.json"))
    assert "pdflip-10-42" not in ns._rvp_p_done


def test_p_done_only_on_success_else_p_failed_and_named_abort(d_env, caplog):
    F, ns = _front()

    async def body():
        loop = asyncio.get_event_loop()
        bad = F.Pending(rid="pdflip-10-44", path="/generate", payload={}, text="x", t_arrive=0.0,
                        fut=loop.create_future(), est_prompt=10, est_uncached=10, span_known=True,
                        p_only=True, resume_via_p=True)
        bad.fut.set_exception(RuntimeError("leg1 on P returned 400: Duplicate request ID"))
        ns._rvp_inflight.add(bad.rid)
        with caplog.at_level("INFO"):
            ns._rvp_p_finished(bad)
            await asyncio.sleep(0.01)
        assert "pdflip-10-44" not in ns._rvp_p_done, "no p-done for a failed leg"
        assert ns.aborts == [("D-group", "/abort_request", {"rid": "pdflip-10-44"})]
        assert "pdflip-10-44" not in ns._rvp_inflight
        good = F.Pending(rid="pdflip-10-43", path="/generate", payload={}, text="x", t_arrive=0.0,
                         fut=loop.create_future(), est_prompt=10, est_uncached=10, span_known=True,
                         p_only=True, resume_via_p=True)
        good.leg1_done = True
        ns._rvp_p_finished(good)
        assert "pdflip-10-43" in ns._rvp_p_done
        late = F.Pending(rid="pdflip-10-42", path="/generate", payload={}, text="x", t_arrive=0.0,
                         fut=loop.create_future(), est_prompt=10, est_uncached=10, span_known=True,
                         p_only=True, resume_via_p=True)
        late.leg1_done = True
        ns._rvp_stream_end("pdflip-10-42")
        ns._rvp_p_finished(late)
        assert "pdflip-10-42" not in ns._rvp_p_done, "p-done after the stream ended is dropped"

    _run(body())
    msgs = caplog.messages
    assert any("RESUME-VIA-P p-failed rid=pdflip-10-44" in m for m in msgs)
    assert not any("RESUME-VIA-P p-done rid=pdflip-10-44" in m for m in msgs)


def test_wait_bound_skips_done_futures_and_wiring():
    src = open(os.path.join(os.path.dirname(rvp.__file__), "front.py"), encoding="utf-8").read()
    assert "oldest = max((p for p in self.queue if not p.fut.done())," in src
    i = src.index("    async def leg2(self, request")
    blk = src[i:src.index("    async def _requeue_after_x_refusal", i)]
    assert "self._rvp_stream_begin(rid)" in blk and "self._rvp_stream_end(rid)" in blk
    j = src.index("    def _rvp_take(self)")
    assert "self._rvp_requeue[rid] = r" not in src[j:j + 3000], "held-uncommitted is not re-queued"
