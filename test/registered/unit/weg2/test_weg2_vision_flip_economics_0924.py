"""xsn438: an image request waited ~50 s on D for a text threshold it can never reach.

Boot weg2xsn438 (fe316bdf40, --weg2-vision transient), front log:
  21:19:52.386 W102 Weg2VisionStage rid=weg2-8-5 image_parts=1 -- routing to P
  21:19:52.387 W102 ... route short -> long (P)
  21:19:52.387 ROUTE-VERDICT rid=weg2-8-5 verdict=long uncached=35 ... est_prompt=36 chars=106
  21:19:52.387 LATE-BATCH rid=weg2-8-5 deferred_to_epoch=9
  21:19:52.570 FLIP-ECONOMICS queued_uncached=35 queued_tokens=36 threshold=4096 ... verdict=hold
  ... 224 x hold ...
  21:20:37.569 FLIP-ECONOMICS ... fairness=True ... verdict=flip   (the 45 s fairness switch)
  21:20:44.358 D-ADMIT rid=weg2-8-5 ... oldest_wait_s=52.0

WHY IT HELD. `_flip_economics_ok` is the D->P departure latch: flip when the
queued UNCACHED work reaches X* (`--flip-min-work-tokens`, default X), because
that amortises the round trip 2*flip_s against the alternative -- D prefilling
the work itself (law 4, X). Text queued on D's watch has that alternative: it
is either long (uncached > X, one request alone reaches the default X* = X) or
short work D takes itself once a seat frees (the trickle the latch holds, T10).
The image request is the one queued request whose route is `long` by RULE, not
by length (W102: its embeddings exist only on P's PP0, D carries no tower;
memory vision-tower-platzierung, "D prefillt keine Bild-Requests"), and its
`est_uncached` counts its 106 text chars and not its image. The latch compared
work D can never do against the price of D doing it.

THE FIX (switch SGLANG_WEG2_VISION_FLIP_URGENT, default off): a request only P
can serve counts as flip-worthy on its own, exactly as a long text request
does. Not a pre-emption: the latch is still only asked when D's work is
exhausted (or fairness closed D), the dwell latch still holds, P still drains
the whole backlog. Counting the image tokens instead (1024 per 1024x1024 image)
would still hold this very request (35 + 1024 < 4096) -- it answers a pricing
question, not the P-only one.

Hermetic: a real Front, a real handle_generate for the routing half, no
server, no GPU.
"""

import asyncio
import logging

import pytest

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, Pending
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

X = 4096  # xsn438: threshold=4096 (X* = X)
IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


class _Req:
    """The two things handle_generate reads from an aiohttp request."""

    def __init__(self, payload, path="/v1/chat/completions"):
        self._payload = payload
        self.path = path

    async def json(self):
        return self._payload


def _payload(text="Describe this picture in one short sentence, please.", images=1):
    content = [{"type": "text", "text": text}] + [dict(IMG) for _ in range(images)]
    return {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 16}


def _front(vision="transient", **kw):
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                 tp_prefill_max_tokens=X, vision=vision, **kw)


def _route_one(front, payload):
    """Run the REAL handle_generate until the request is queued (or answered),
    then cancel the waiter. Returns the queued Pending, or None."""

    async def body():
        task = asyncio.ensure_future(front.handle_generate(_Req(payload)))
        for _ in range(500):
            await asyncio.sleep(0)
            if front.queue or task.done():
                break
        queued = front.queue[-1] if front.queue else None
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - the waiter is ours to drop
            pass
        return queued

    return asyncio.run(body())


# ---------------------------------------------------------------- routing half
def test_the_xsn438_image_request_is_queued_as_p_only_work():
    f = _front()
    p = _route_one(f, _payload())
    assert p is not None, "a transient image request must be queued for P"
    assert p.est_uncached < X, "the text-only estimate is what the latch sees (xsn438: 35)"
    assert p.p_only is True, "an image under transient is served by P only (W102)"


def test_text_and_resident_images_are_not_p_only():
    f = _front()
    long_text = _route_one(f, {"prompt": "x" * (3 * X + 300)})  # > X: long by length
    assert long_text is not None and long_text.p_only is False
    g = _front(vision="resident")  # both groups carry the tower: D may serve it
    assert _route_one(g, _payload()) is None, "a short resident image goes SHORT to D, unqueued"


# ---------------------------------------------------------------- latch half
def test_switch_off_keeps_the_hold_the_boot_showed():
    f = _front()
    f.vision_flip_urgent = False
    _route_one(f, _payload())
    assert f._flip_economics_ok(fairness_fired=False) is False
    assert f._flip_economics_ok(fairness_fired=True) is True, "fairness still frees it"


def test_switch_on_flips_on_p_only_work_below_the_threshold():
    f = _front()
    f.vision_flip_urgent = True
    _route_one(f, _payload())
    assert f._flip_economics_ok(fairness_fired=False) is True


def test_switch_on_leaves_text_below_the_threshold_on_hold():
    f = _front()
    f.vision_flip_urgent = True
    loop = asyncio.new_event_loop()
    try:
        f.queue.append(Pending("r", "/generate", {}, "x", 0.0, loop.create_future(),
                               est_prompt=100, est_uncached=100))
        assert f._flip_economics_ok(fairness_fired=False) is False
    finally:
        loop.close()


def test_switch_on_is_not_a_preemption():
    """A1-1: fairness is the ONLY pre-emption. The switch may not close D."""
    f = _front()
    f.vision_flip_urgent = True
    _route_one(f, _payload())
    assert f.admit_d is True
    f._flip_economics_ok(fairness_fired=False)
    assert f.admit_d is True, "the latch must not stop D admitting"


def test_the_line_names_both_terms(caplog):
    f = _front()
    f.vision_flip_urgent = True
    _route_one(f, _payload())
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f._flip_economics_ok(fairness_fired=False)
    lines = [r.getMessage() for r in caplog.records if "FLIP-ECONOMICS" in r.getMessage()]
    assert lines, "no FLIP-ECONOMICS line"
    assert "p_only=1" in lines[-1] and "vision_flip_urgent=True" in lines[-1], lines[-1]


# ---------------------------------------------------------------- the switch
@pytest.mark.parametrize("raw,want", [(None, False), ("", False), ("0", False), ("off", False),
                                      ("1", True), ("true", True), ("on", True), ("YES", True)])
def test_switch_resolution(raw, want):
    env = {} if raw is None else {"SGLANG_WEG2_VISION_FLIP_URGENT": raw}
    assert front_mod.vision_flip_urgent(env) is want


def test_the_front_names_the_switch_at_start(caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_VISION_FLIP_URGENT", "1")
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f = _front()
    assert f.vision_flip_urgent is True
    assert any("VISION-FLIP-URGENT on" in r.getMessage() for r in caplog.records)
    assert f.state_dict()["vision_flip_urgent"] is True
    monkeypatch.delenv("SGLANG_WEG2_VISION_FLIP_URGENT")
    assert _front().vision_flip_urgent is False, "default off"
