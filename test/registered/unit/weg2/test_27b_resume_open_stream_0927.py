"""ROS (NF rc12m-dpr 12:16:12, weg2-16-59 / weg2-18-64): RESUME-VIA-P for a
stream the front already opened, output = 0.

Metal: both rids had their P leg 1, D-ADMIT (12:15:18 / 12:15:21), then waited
~51 s on D; D's Anthropic stream sends message_start and a ping every 5 s, so
the front's refusal lookahead (8 chunks) committed the stream to the client
with nothing generated. X-GATE uncached=18749 > X=12288 -> W50 in-band "after
the first byte -- re-route impossible", an empty 200. RESUME-VIA-P did not
apply: it required output_ids > 0.

The rule: D holds EVERY streamed refusal (the replicated ``stream`` flag, no
rank-local read of a front marker); the front, which knows whether it
committed, answers -- committed: P-only leg 1 (RESUME-VIA-P); still in the
lookahead: X-REQUEUE with the refusal D would have sent. Non-stream keeps the
abort. Switch SGLANG_WEG2_RESUME_OPEN_STREAM (default on, 0 = output>0 rule).
"""
from __future__ import annotations

import asyncio
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import resume_via_p as rvp  # noqa: E402


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.delenv(rvp.ENV_OPEN_STREAM, raising=False)
    return tmp_path


def _req(rid="weg2-16-59", out=0, stream=True, prompt=36285):
    r = types.SimpleNamespace(rid=rid, stream=stream, origin_input_ids=list(range(prompt)),
                              output_ids=list(range(out)), kv_arrival_seq=1,
                              is_fast_lane=False, spill_class=None)
    r.full_untruncated_fill_ids = r.origin_input_ids + r.output_ids
    return r


# -- D -------------------------------------------------------------------------


def test_eligible_covers_an_open_stream_without_output(d_env, monkeypatch):
    assert rvp.open_stream_enabled()
    assert rvp.eligible(_req(out=0)), "the metal case: streamed, output 0"
    assert rvp.eligible(_req(out=5))
    assert not rvp.eligible(_req(stream=False)), "non-stream keeps the abort (X-REQUEUE)"
    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
    assert not rvp.eligible(_req(out=0)), "switch off: the output>0 rule (the metal behaviour)"
    assert rvp.eligible(_req(out=5))


class _Chan:
    def __init__(self):
        self.sent = []

    def send_output(self, out, req):
        self.sent.append((out, req))


def _x_sched(queue):
    chan = _Chan()
    ns = types.SimpleNamespace(
        server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288), waiting_queue=list(queue),
        tree_cache=types.SimpleNamespace(release_aborted_request=lambda rid: None),
        enable_hicache_storage=True, enable_hierarchical_cache=False,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=chan), ps=types.SimpleNamespace(tp_rank=0),
        weg2_d_parked=[], weg2_uncached_extent=lambda req, head=None: 18749,
    )
    return ns, chan


@pytest.mark.parametrize("ros", ["1", "0"])
def test_scheduler_holds_the_open_stream_and_aborts_non_stream(d_env, monkeypatch, ros):
    from sglang.srt.managers import scheduler as S

    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, ros)
    monkeypatch.setattr(S, "release_admission_acquired_mamba_slot", lambda req, tc, site: False)
    opened, plain = _req("weg2-16-59"), _req("weg2-99-1", stream=False)
    for r in (opened, plain):
        r.time_stats = types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **kw: None))
    ns, chan = _x_sched([opened, plain])
    S.Scheduler._weg2_answer_x_refusals(ns, [opened, plain])
    answered = [q.rid for _o, q in chan.sent]
    if ros == "1":
        assert answered == ["weg2-99-1"] and ns.weg2_d_parked == [opened]
        got = rvp.take_requests(rvp.needs_p_dir())
        assert got[0]["rid"] == "weg2-16-59" and got[0]["input_ids"] == opened.origin_input_ids
        assert got[0]["d_extent"] == 18749
    else:  # the metal: the open stream is aborted in-band (W50 after the first byte)
        assert answered == ["weg2-16-59", "weg2-99-1"] and ns.weg2_d_parked == []


def test_held_refusal_body_parses_like_ds_w50():
    from sglang.srt.weg2 import front as F

    body = rvp.held_refusal_body({"d_extent": 18749, "x": 12288})
    assert F.x_refusal_marker_in(body.decode()) and F._d_refusal_extent(body) == 18749


# -- front ---------------------------------------------------------------------


def _front(tmp_path):
    from sglang.srt.weg2 import front as F

    ns = types.SimpleNamespace(tag="dkrtest", queue=[], counters={"rvp_rerouted": 0, "rvp_resumed": 0,
                                                                  "rvp_uncommitted_requeue": 0},
                               kicks=[])
    ns._kick_controller = lambda why: ns.kicks.append(why)
    for name in ("_rvp_state", "_note_front_price", "_rvp_take", "_rvp_p_finished", "_rvp_resumed",
                 "_ros_lookahead_begin", "_ros_lookahead_end"):
        setattr(ns, name, getattr(F.Front, name).__get__(ns))
    return F, ns


class _Content:
    """D's Anthropic stream: message_start, then a ping every `gap` s."""

    def __init__(self, chunks, gap):
        self.chunks, self.gap = list(chunks), gap

    async def readany(self):
        if not self.chunks:
            await asyncio.sleep(3600)
        await asyncio.sleep(self.gap)
        return self.chunks.pop(0)


def _r(chunks, gap=0.05):
    return types.SimpleNamespace(content=_Content(chunks, gap))


START = b'event: message_start\ndata: {"type":"message_start"}\n\n'
PING = b'event: ping\ndata: {"type": "ping"}\n\n'
DELTA = b'event: content_block_start\ndata: {}\n\n'


def test_held_while_in_the_lookahead_is_x_requeue(d_env, monkeypatch, caplog):
    monkeypatch.setattr("sglang.srt.weg2.front.ROS_POLL_S", 0.02)
    F, ns = _front(d_env)

    async def body():
        stop = ns._ros_lookahead_begin("weg2-16-59")
        assert stop is not None and "weg2-16-59" in ns._leg2_lookahead

        async def d_holds():
            await asyncio.sleep(0.12)
            rvp.write_request("weg2-16-59", list(range(10)), 18749, 12288, "x_refusal_midstream")

        asyncio.ensure_future(d_holds())
        with pytest.raises(F._RosHeld):
            await F._anthropic_refusal_lookahead(_r([START] + [PING] * 20), stop)
        rec = ns._ros_lookahead_end("weg2-16-59")
        assert rec is not None and rec["d_extent"] == 18749
        assert ns.queue == [], "no P-only leg: the X-REQUEUE answers it"
        assert "weg2-16-59" not in ns._leg2_lookahead and ns._rvp_requeue == {}

    with caplog.at_level("WARNING"):
        asyncio.new_event_loop().run_until_complete(body())
    assert any("path=held-uncommitted" in m for m in caplog.messages)


def test_held_after_the_commit_is_resume_via_p(d_env):
    F, ns = _front(d_env)

    async def body():
        stop = ns._ros_lookahead_begin("weg2-16-59")
        head, refused = await F._anthropic_refusal_lookahead(_r([START] + [PING] * 7), stop)
        assert not refused and head.count(b"event: ping") == 7  # 8 chunks: the lookahead gives up
        assert ns._ros_lookahead_end("weg2-16-59") is None  # -> the stream is committed
        rvp.write_request("weg2-16-59", list(range(10)), 18749, 12288, "x_refusal_midstream")
        assert ns._rvp_take() == 1
        p = ns.queue[0]
        assert p.resume_via_p and p.p_only and p.rid == "weg2-16-59"

    asyncio.new_event_loop().run_until_complete(body())


def test_a_live_stream_passes_the_watched_lookahead_unchanged(d_env):
    F, ns = _front(d_env)

    async def body():
        stop = ns._ros_lookahead_begin("weg2-1-1")
        head, refused = await F._anthropic_refusal_lookahead(_r([START, PING, DELTA]), stop)
        assert head == START + PING + DELTA and not refused
        assert ns._ros_lookahead_end("weg2-1-1") is None

    asyncio.new_event_loop().run_until_complete(body())


def test_an_in_band_refusal_still_wins_in_the_watched_read(d_env):
    F, ns = _front(d_env)

    async def body():
        stop = ns._ros_lookahead_begin("weg2-1-2")
        chunk = await F._read_chunk_watched(_r([b'data: {"error":"W50 Weg2TpPrefillExceeded ..."}\n\n']), stop)
        assert F.x_refusal_marker_in(chunk.decode())
        ns._ros_lookahead_end("weg2-1-2")

    asyncio.new_event_loop().run_until_complete(body())


def test_switch_off_is_the_old_lookahead(d_env, monkeypatch):
    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
    F, ns = _front(d_env)
    assert ns._ros_lookahead_begin("weg2-16-59") is None
    monkeypatch.delenv(rvp.ENV_OPEN_STREAM)
    monkeypatch.setenv(rvp.ENV, "0")
    F, ns = _front(d_env)
    assert ns._ros_lookahead_begin("weg2-16-59") is None


def test_wiring_in_the_leg2_stream_branch():
    src = open(os.path.join(os.path.dirname(rvp.__file__), "front.py"), encoding="utf-8").read()
    i = src.index("_ros_stop = self._ros_lookahead_begin(rid)")
    blk = src[i:i + 2600]
    assert "_anthropic_refusal_lookahead(r, _ros_stop)" in blk
    assert "_read_chunk_watched(r, _ros_stop)" in blk
    assert "_ros_rec = self._ros_lookahead_end(rid) if _ros_stop is not None else None" in blk
    assert "_rvp.held_refusal_body(_ros_rec)" in blk
    j = blk.index("_rvp.held_refusal_body(_ros_rec)")
    assert "_requeue_after_x_refusal(" in blk[j - 300:j]
    # the check sits before the stream is committed
    assert i < src.index("await resp.prepare(request)", i)
