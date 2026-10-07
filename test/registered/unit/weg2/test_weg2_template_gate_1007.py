"""TEMPLATE-GATE (NF seat 07.10.): a request the chat template itself refuses is answered 400 at
the arrival -- before any route, seat, flip or park.

Observed on NF boot dkrnfint4h6abl 07.10. (07:22Z): an agent sent a ~22k-token request with a message
role the model's chat template does not know (``Unexpected message role.``). The front's exact count
failed with that ValueError, fell back to chars/3, routed the request LONG, flipped D->P (3 s), P answered
400, the front flipped back P->D (3 s) and the agent retried at once: 140+ refusals, a flip every 3 s,
no inference. A single sglang server answers the same request with a 400 in milliseconds
(serving_base ``except ValueError`` -> BadRequest); the front must not turn it into a flip pair.

Pinned here:
  (1) only the ValueError sglang builds from a jinja2.TemplateError is a template refusal;
  (2) the exact count raises ``FrontTemplateRefused`` for it and keeps the chars/3 fallback for
      every other failure (plain ValueError, timeout, tokenizer state);
  (3) handle_generate answers 400 (OpenAI wire on /v1/chat/completions, Anthropic wire on
      /v1/messages) and never routes, seats or flips;
  (4) the same on the REAL chat template of the served model when it is on this box.
"""

import asyncio
import logging

import jinja2
import pytest

from sglang.srt.weg2 import front as F
from test_weg2_front_x_exact import X, _front, _FakeTokens, _Req


def _template_error(msg="Unexpected message role."):
    try:
        try:
            raise jinja2.TemplateError(msg)
        except jinja2.TemplateError as te:
            raise ValueError(str(te)) from te  # what serving_chat does (line ~999)
    except ValueError as ve:
        return ve


class _RefusingTokens(_FakeTokens):
    def __init__(self, exc):
        super().__init__(3 * X)
        self.exc = exc

    def count(self, path, payload):
        raise self.exc


def _developer_payload(chars=3 * X + 600):
    return {"model": "m", "max_tokens": 10,
            "messages": [{"role": "developer", "content": "x" * chars},
                         {"role": "user", "content": "hi"}]}


def _serve(f, payload, path):
    flips = []

    async def flip(src, dst):
        flips.append((src, dst))

    f.flip = flip

    async def go():
        return await asyncio.wait_for(f.handle_generate(_Req(payload, path)), 2.0)

    return asyncio.run(go()), flips


# (1) -----------------------------------------------------------------------------------------

def test_only_a_template_cause_is_a_template_refusal():
    assert F.is_template_refusal(_template_error())
    assert not F.is_template_refusal(ValueError("path /x not counted by the front"))
    assert not F.is_template_refusal(jinja2.TemplateError("bare"))  # not the serving layer's ValueError
    try:
        try:
            raise TypeError("template bug")
        except TypeError as te:
            raise ValueError("wrapped") from te
    except ValueError as ve:
        assert not F.is_template_refusal(ve)  # TypeError cause: not a request fault
    assert not F.is_template_refusal(RuntimeError("front tokenizer not ready"))
    import pydantic

    with pytest.raises(pydantic.ValidationError) as ei:
        pydantic.TypeAdapter(int).validate_python("x")
    assert F.is_template_refusal(ei.value)  # the request schema refused it: every group answers 400


# (2) -----------------------------------------------------------------------------------------

def test_exact_count_raises_for_a_template_refusal_and_falls_back_otherwise():
    f = _front(True)
    f.ftok = _RefusingTokens(_template_error())

    async def price(exc_tokens):
        f.ftok = exc_tokens
        return await f._x_exact_price("weg2-0-1", "/v1/chat/completions", _developer_payload(), "t", 5000, 5000)

    with pytest.raises(F.FrontTemplateRefused, match="Unexpected message role"):
        asyncio.run(price(_RefusingTokens(_template_error())))
    assert f.counters["x_exact_fallback"] == 0
    # a ValueError that is not the template's: the estimate stands, named (old behaviour)
    assert asyncio.run(price(_RefusingTokens(ValueError("/generate without a single text prompt")))) is None
    assert f.counters["x_exact_fallback"] == 1


# (3) -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,key", [("/v1/chat/completions", "BadRequestError"),
                                      ("/v1/messages", "invalid_request_error")])
def test_arrival_is_answered_400_with_no_route_seat_or_flip(path, key, caplog):
    f = _front(True)
    f.ftok = _RefusingTokens(_template_error())
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        resp, flips = _serve(f, _developer_payload(), path)
    assert resp.status == 400 and "Retry-After" not in resp.headers
    body = resp.body.decode()
    assert "Unexpected message role." in body and key in body
    assert flips == [] and f.routed == []
    assert f.counters["template_gate_refused"] == 1
    assert f.counters["route_long"] == 0 and f.counters["requests"] == 0
    assert not f.queue
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("WEG2 TEMPLATE-GATE REFUSED") for m in msgs)
    assert not any(m.startswith("WEG2 ROUTE-VERDICT") or "WEG2-ROUTE" in m for m in msgs)


def test_the_old_fallback_still_routes_other_count_failures():
    f = _front(True)
    f.ftok = _RefusingTokens(ValueError("not a template fault"))
    resp, flips = _serve(f, _developer_payload(chars=100), "/v1/messages")
    assert resp.status != 400
    assert f.counters["template_gate_refused"] == 0 and f.counters["x_exact_fallback"] == 1


# (4) -----------------------------------------------------------------------------------------

#: a served-family checkpoint with its real chat template that IS on this box (the TOKENIZERS of the
#: x-exact test are release mounts, empty on the dev box)
REAL_TEMPLATE_DIR = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-abl-INT8-W8A16-DFlash2-lued"


@pytest.fixture(scope="module")
def real_front_tokens():
    import os

    from sglang.srt.weg2 import front_tokens as FT
    from test_weg2_front_x_exact import _args

    if not os.path.exists(os.path.join(REAL_TEMPLATE_DIR, "chat_template.jinja")):
        pytest.skip("served-family chat template not on this box")
    ft = FT.FrontTokens()
    ft.load(_args(REAL_TEMPLATE_DIR), is_multimodal=False)
    assert ft.state == "ready", ft.why
    return ft


@pytest.mark.parametrize("role", ["developer", "wizard"])
def test_the_real_template_and_schema_refusals_are_recognised_through_the_front_count(real_front_tokens, role):
    """``developer`` passes the request schema and dies in the Qwen template (ValueError from a
    TemplateError, the 22k-token request of 07.10.); a role outside the schema dies in pydantic."""
    ok = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}
    assert real_front_tokens.count("/v1/chat/completions", ok).n > 0
    with pytest.raises(ValueError) as ei:
        real_front_tokens.count("/v1/chat/completions", {"model": "m", "max_tokens": 4, "messages": [
            {"role": "user", "content": "hi"}, {"role": role, "content": "?"}]})
    assert F.is_template_refusal(ei.value), repr(ei.value)
