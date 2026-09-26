"""H125b: the native ``/generate`` multimodal fields get the front's vision verdict.

THE DEFECT (desk finding H125, 26.09.). The front counted images and videos
only as chat content parts (``messages[].content[].type``). A native
``/generate`` body carries them as ``image_data`` / ``video_data`` (and may
carry ``input_embeds`` / ``audio_data``), so every counter scored 0 and the
request ROUTED AS TEXT. Under ``--weg2-vision off`` (every NF boot) both groups
run ``--no-enable-multimodal``, the tokenizer has no mm_processor, and the
image was dropped SILENTLY: a plausible answer about a picture nobody looked
at -- the one failure shape #1356 exists to make loud.

Pinned here:
  * ``off``: ``image_data`` -> W101, ``video_data`` -> W103, both 501, the
    same log line as the chat parts, NOT routed, nothing queued;
  * ``transient``: ``image_data`` goes to the stage (W102, queued for P,
    ``p_only``) -- the native path exists: P's tokenizer builds mm_items from
    ``image_data`` and PP0's in-rank stage encodes them; ``video_data`` stays
    W103 (the stage encodes still images only);
  * ``input_embeds`` / ``audio_data``: W125 in EVERY mode -- the Weg-2 path
    routes, matches and hands off by token ids, and the model has no audio;
  * text bodies (``image_data`` absent, None, "" or []) route exactly as before;
  * the chat W101 line is byte-for-byte the line it was.

Red on the H125 stand (d460558): the counters ignore the native fields.
Hermetic: a real Front and its real ``handle_generate``; no server, no GPU.
"""

import asyncio
import logging

import pytest

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

X = 4096
PNG = "data:image/png;base64,iVBORw0KGgo="


class _Req:
    def __init__(self, payload, path="/generate"):
        self._payload = payload
        self.path = path

    async def json(self):
        return self._payload


def _front(vision):
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                 tp_prefill_max_tokens=X, vision=vision)


def _send(front, payload, path="/generate"):
    """Run the REAL handle_generate until it answers or queues. Returns
    (response or None, queued Pending or None)."""

    async def body():
        task = asyncio.ensure_future(front.handle_generate(_Req(payload, path)))
        for _ in range(500):
            await asyncio.sleep(0)
            if front.queue or task.done():
                break
        resp = None
        if task.done() and not task.cancelled() and task.exception() is None:
            resp = task.result()  # a text body may die later on the absent session: not ours
        queued = front.queue[-1] if front.queue else None
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - the waiter is ours to drop
                pass
        return resp, queued

    return asyncio.run(body())


def _gen(**kw):
    body = {"text": "<|vision_start|><|image_pad|><|vision_end|>What is in this picture?",
            "sampling_params": {"max_new_tokens": 8}}
    body.update(kw)
    return body


def _error(resp):
    import json

    return json.loads(resp.body.decode())["error"]


# ------------------------------------------------------------------ counters --
@pytest.mark.parametrize("value,n", [
    (None, 0), ("", 0), ([], 0), ([[]], 0), (PNG, 1), ([PNG, PNG], 2),
    ([[PNG], [PNG, PNG]], 3), ({"url": PNG}, 1), ([None, PNG], 1),
])
def test_native_image_data_is_counted(value, n):
    assert front_mod._image_parts({"text": "x", "image_data": value}) == n


def test_native_video_and_embeds_are_counted():
    assert front_mod._video_parts({"text": "x", "video_data": ["a.mp4"]}) == 1
    assert front_mod._embed_parts({"input_embeds": [[0.1, 0.2]]}) == 1
    assert front_mod._embed_parts({"text": "x", "audio_data": "a.wav"}) == 1
    assert front_mod._embed_parts({"text": "x"}) == 0


def test_chat_parts_are_counted_as_before():
    chat = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "an image of"},
        {"type": "image_url", "image_url": {"url": PNG}}]}]}
    assert front_mod._image_parts(chat) == 1
    assert front_mod._image_parts({"messages": [{"role": "user", "content": "the word image"}]}) == 0


@pytest.mark.parametrize("mode", ["off", "resident", "transient"])
def test_embeds_are_refused_in_every_mode(mode):
    assert front_mod.vision_verdict(0, 0, mode, 1)[0] == front_mod.VERDICT_REFUSE_EMBEDS
    assert front_mod.vision_verdict(0, 0, mode)[0] == front_mod.VERDICT_ROUTE  # default arg: as before


# -------------------------------------------------------------------- off ------
def test_off_refuses_native_image_data_by_name(caplog):
    f = _front("off")
    with caplog.at_level(logging.WARNING, logger="weg2.front"):
        resp, queued = _send(f, _gen(image_data=PNG))
    assert resp is not None and resp.status == 501, "the image must not route as text"
    assert _error(resp).startswith("W101 Weg2VisionRefused")
    assert queued is None and not f.queue
    lines = [r.getMessage() for r in caplog.records if "W101" in r.getMessage()]
    assert lines and "image_parts=1 video_parts=0 mode=off -- request refused" in lines[-1]


def test_off_refuses_native_video_data_by_name():
    resp, queued = _send(_front("off"), _gen(video_data=["clip.mp4"]))
    assert resp is not None and resp.status == 501
    assert _error(resp).startswith("W103 Weg2VideoRefused") and queued is None


@pytest.mark.parametrize("field,value", [("input_embeds", [[0.0] * 4]), ("audio_data", "a.wav")])
def test_off_refuses_embeds_and_audio_by_name(field, value, caplog):
    with caplog.at_level(logging.WARNING, logger="weg2.front"):
        resp, queued = _send(_front("off"), _gen(**{field: value}))
    assert resp is not None and resp.status == 501
    assert _error(resp).startswith("W125 Weg2InputEmbedsRefused") and queued is None
    assert any(r.getMessage().startswith("W125 Weg2InputEmbedsRefused") and "embed_parts=1" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("value", [None, "", []])
def test_text_generate_routes_as_before(value):
    f = _front("off")
    body = _gen() if value is None else _gen(image_data=value)
    resp, queued = _send(f, body)
    assert resp is None or resp.status != 501


def test_the_chat_refusal_line_is_unchanged(caplog):
    chat = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": PNG}}]}], "max_tokens": 4}
    with caplog.at_level(logging.WARNING, logger="weg2.front"):
        resp, _ = _send(_front("off"), chat, path="/v1/chat/completions")
    assert resp.status == 501
    line = [r.getMessage() for r in caplog.records if "W101" in r.getMessage()][-1]
    assert line.endswith("image_parts=1 video_parts=0 mode=off -- request refused with 501 and NOT routed")


# --------------------------------------------------------------- transient ------
def test_transient_hands_native_image_data_to_the_stage(caplog):
    f = _front("transient")
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        resp, queued = _send(f, _gen(image_data=PNG))
    assert resp is None or resp.status != 501
    assert queued is not None, "a transient image is queued for P"
    assert queued.p_only is True
    assert queued.payload.get("image_data") == PNG, "the image travels with the request"
    assert any("W102 Weg2VisionStage" in r.getMessage() and "image_parts=1" in r.getMessage()
               for r in caplog.records)


def test_transient_still_refuses_native_video():
    resp, queued = _send(_front("transient"), _gen(video_data="clip.mp4"))
    assert resp is not None and resp.status == 501
    assert _error(resp).startswith("W103") and queued is None
