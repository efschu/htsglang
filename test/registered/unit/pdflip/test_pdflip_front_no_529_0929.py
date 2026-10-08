"""W88/W50 529 (29.09., NF boot dkrnfh91dprsavisnoadoptstcutvsyncbar1dauer09290232, e7c0200ccf).

    front 03:21:06 PDFLIP LEG2-TERMINAL-NAMED rid=pdflip-122-144 reason=error:W88 path=/v1/messages
                   error_event=forwarded
    D     03:21:06 W88 PdFlipStoreLoadNotProgressing rid=pdflip-122-144 ... terminal, answered 503
    D     03:21:06 Forwarding upstream stream error (overloaded_error): W88 PdFlipStoreLoadNotProgressing ...
    front 03:22:09 PDFLIP LEG2-TERMINAL-NAMED rid=pdflip-116-141 reason=W50 path=/v1/messages
                   error_event=forwarded   (RESUME-VIA-P attempt 2/2 spent, re-route impossible)

D answers 503, its Anthropic adapter maps 503 to ``overloaded_error``, and the front forwarded that
event verbatim. On the Anthropic wire ``overloaded_error`` IS the 529: Claude Code reads it as "API
overloaded" and aborts the agent. The rule of the 413 fix (``refusal_status``) extended: a state or
read refusal (W88, W50, ...) is a 503 with Retry-After and Anthropic's 5xx type -- in the forwarded
stream, in the synthesized terminal event, in a non-stream pass-through and in ``refusal_response``.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import (HANDBACK_NAME, STATE_REFUSAL_RETRY_AFTER_S,  # noqa: E402
                                   refusal_response)

A = "/v1/messages"
O = "/v1/chat/completions"
#: D's frames as its Anthropic adapter writes them (``_flush_on_error`` -> json.dumps)
W88_MSG = ("W88 PdFlipStoreLoadNotProgressing: the HiCache store read for this request did not deliver. "
           "arm=store_prefix_short span=97025 site=drain")
W50_MSG = ("W50 PdFlipTpPrefillExceeded: this group may prefill at most 12288 uncached tokens itself "
           "(--tp-prefill-max-tokens); this request's extent after prefix matching is 39693. Refused by "
           "name so the caller re-routes it through the prefill group -- never prefilled here silently.")


def _event(msg: str, etype: str = "overloaded_error") -> bytes:
    return (b"event: error\ndata: "
            + json.dumps({"type": "error", "error": {"type": etype, "message": msg}}).encode() + b"\n\n")


def _types(chunk: bytes):
    out = []
    for line in chunk.split(b"\n"):
        if line.startswith(b"data: "):
            js = json.loads(line[6:])
            if isinstance(js, dict) and "error" in js:
                out.append(js["error"].get("type"))
    return out


def test_the_forwarded_named_refusal_leaves_as_the_503_type_not_529():
    for msg in (W88_MSG, W50_MSG):
        out = F.restate_inband_refusal(_event(msg), A)
        assert _types(out) == [F.STATE_REFUSAL_ANTHROPIC_TYPE], out
        assert b"overloaded_error" not in out
        # the name and the whole detail are untouched, and it is still the terminal event
        assert json.loads(out.split(b"data: ", 1)[1])["error"]["message"] == msg
        assert F.stream_error_event_in(out, A)
    assert F.leg2_terminal_reason(F.restate_inband_refusal(_event(W50_MSG), A), A, finished=False) == "W50"
    assert F.leg2_terminal_reason(F.restate_inband_refusal(_event(W88_MSG), A), A,
                                  finished=False) == "error:W88"


def test_content_and_unnamed_errors_and_other_wires_pass_byte_for_byte():
    prose = (b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta",'
             b'"text":"the \\"type\\":\\"overloaded_error\\" field of W88 PdFlipStoreLoadNotProgressing"}}\n\n')
    assert F.restate_inband_refusal(prose, A) is prose
    unnamed = _event("upstream overloaded")
    assert F.restate_inband_refusal(unnamed, A) is unnamed
    oai = b"data: " + json.dumps({"error": {"message": W88_MSG, "type": "overloaded_error"}}).encode() + b"\n\n"
    assert F.restate_inband_refusal(oai, O) is oai


def test_the_synthesized_terminal_event_is_the_503_type():
    a = F.named_error_chunk(A, "W164 PdFlipLeg2Truncated rid=r: ...")
    js = json.loads(a.split(b"data: ", 1)[1])
    assert js["error"]["type"] == F.STATE_REFUSAL_ANTHROPIC_TYPE
    assert b"overloaded_error" not in a


def test_refusal_response_on_the_anthropic_wire_is_503_retry_after_not_529():
    r = refusal_response(A, HANDBACK_NAME, "W53 PdFlipStoreHandbackFailed rid=r", 503)
    assert r.status == 503 and r.headers["Retry-After"] == str(STATE_REFUSAL_RETRY_AFTER_S)
    assert json.loads(r.text)["error"]["type"] == F.STATE_REFUSAL_ANTHROPIC_TYPE


def test_wiring_every_d_chunk_and_the_pass_through_are_restated():
    src = open(F.__file__).read()
    assert "await _push(restate_inband_refusal(chunk, request.path))" in src
    assert "await _push(restate_inband_refusal(first_chunk, request.path))" in src
    assert "await _push(restate_inband_refusal(early_body, request.path))" in src
    i = src.index("return web.Response(body=restate_inband_refusal(body, request.path),")
    assert '"Retry-After": str(STATE_REFUSAL_RETRY_AFTER_S)' in src[i:i + 300]
