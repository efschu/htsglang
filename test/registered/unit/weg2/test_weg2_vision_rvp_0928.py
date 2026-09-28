# SPDX-License-Identifier: Apache-2.0
"""VISION after the first byte: D cannot cover an image of a request that has
already streamed. Before this, W123 (the RESUME-VIA-P leg carries input_ids
alone; P could not encode the image from them). Now D parks the stream as a
RESUME-VIA-P hold with reason ``vision_not_in_prefix`` and the front's P leg is
the ORIGINAL request body (image included): P prefills the prompt with its
tower, D resumes from P's pages and prefills only the streamed output tail."""
from __future__ import annotations

import asyncio
import collections
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import resume_via_p as rvp  # noqa: E402
from sglang.srt.weg2 import vision_d_guard as g  # noqa: E402


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.delenv(rvp.ENV_OPEN_STREAM, raising=False)
    return tmp_path


def _mm_req(output=3):
    items = [types.SimpleNamespace(offsets=[(6290, 9528)])]
    return types.SimpleNamespace(
        rid="weg2-2-12", stream=True, output_ids=list(range(output)),
        full_untruncated_fill_ids=list(range(9537 + output)), origin_input_ids=list(range(9537)),
        multimodal_inputs=types.SimpleNamespace(mm_items=items),
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)))


def test_eligible_keeps_multimodal_out_of_the_ids_leg_and_lets_the_vision_hold_in(d_env):
    req = _mm_req()
    assert rvp.eligible(req) is False, "the input_ids leg can never carry an image"
    assert rvp.eligible(req, vision=True) is True


def test_d_parks_a_streamed_vision_refusal_as_a_vision_hold(d_env, monkeypatch):
    from sglang.srt.managers.scheduler import Scheduler

    kept, sent = [], []
    monkeypatch.setattr(rvp, "keep_on_d", lambda s, r, ext, x, reason="": kept.append((r.rid, ext, x, reason)) or True)
    req = _mm_req(output=3)
    h = types.SimpleNamespace(
        waiting_queue=[req], tree_cache=None, enable_hicache_storage=False,
        enable_hierarchical_cache=False, server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda out, r: sent.append(out))),
        _weg2_vision_d_covered=lambda r, hi: 0)
    Scheduler._weg2_answer_vision_d_refusals(h, [req], None)
    assert sent == [], "no W123 to a client that already holds text"
    assert kept == [("weg2-2-12", 9540, 12288, rvp.REASON_VISION)]
    assert h.waiting_queue == []


def test_d_without_rvp_keeps_the_named_w123(d_env, monkeypatch):
    from sglang.srt.managers.scheduler import Scheduler

    monkeypatch.setenv(rvp.ENV, "0")
    sent = []
    req = _mm_req(output=3)
    h = types.SimpleNamespace(
        waiting_queue=[req], tree_cache=None, enable_hicache_storage=False,
        enable_hierarchical_cache=False, server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda out, r: sent.append(out))),
        _weg2_vision_d_covered=lambda r, hi: 0)
    Scheduler._weg2_answer_vision_d_refusals(h, [req], None)
    assert sent[0].finished_reason["message"].startswith(g.W_NOT_IN_PREFIX)


def _front():
    from sglang.srt.weg2 import front as F

    ns = types.SimpleNamespace(tag="dkrtest", queue=[], counters=collections.Counter(), kicks=[],
                               aborts=[])
    ns._kick_controller = lambda why: ns.kicks.append(why)

    async def rpc(g_, path, body, timeout):
        ns.aborts.append((g_, path, body))
        return 200, "ok"

    ns.rpc = rpc
    ns.groups = {"D": "D-group"}
    for name in ("_rvp_state", "_note_front_price", "_rvp_take", "_rvp_p_finished", "_rvp_stream_begin",
                 "_rvp_stream_end", "_rvp_abort_d"):
        setattr(ns, name, getattr(F.Front, name).__get__(ns))
    ns._rvp_state()
    return F, ns


def test_front_p_leg_is_the_original_body_with_the_image(d_env):
    async def body():
        F, ns = _front()
        orig = {"model": "m", "stream": True, "rid": "weg2-2-12",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "what is this"}]}]}
        ns._rvp_bodies["weg2-2-12"] = ("/v1/chat/completions", orig)
        rvp.write_request("weg2-2-12", list(range(9540)), 9540, 12288, rvp.REASON_VISION)
        assert ns._rvp_take() == 1
        (p,) = ns.queue
        assert p.path == "/v1/chat/completions" and p.resume_via_p and p.p_only
        assert p.payload["messages"] == orig["messages"] and p.payload["rid"] == "weg2-2-12"
        assert "weg2-2-12" in ns._rvp_inflight, "the D->P drain must not wait for the hold"
        assert ns.counters["rvp_vision"] == 1

    asyncio.run(body())


def test_front_without_the_body_aborts_the_hold_by_name(d_env):
    async def body():
        F, ns = _front()
        rvp.write_request("weg2-2-12", list(range(10)), 10, 12288, rvp.REASON_VISION)
        ns._rvp_take()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ns.queue == [] and ns.counters["rvp_vision_no_body"] == 1
        assert ns.aborts and ns.aborts[0][1] == "/abort_request"

    asyncio.run(body())


def test_stream_end_drops_the_body():
    F, ns = _front()
    ns._rvp_bodies["r"] = ("/v1/chat/completions", {})
    ns._rvp_stream_end("r")
    assert "r" not in ns._rvp_bodies
