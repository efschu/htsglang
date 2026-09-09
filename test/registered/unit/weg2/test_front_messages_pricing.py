"""Q0-B: the weg2 front forwards ``/v1/messages``, and prices the body it forwards.

THE DEFECT THIS PINS, measured on boot weg2sn5 (2026-09-09T20:3xZ, record
``/spinning/gpu-arb/weg2/BOOT_weg2sn5_0909.md``): group D answered
``POST :30032/v1/messages`` with 200 while the front on :30030 answered 404,
and the split router at :30099 passes the path through unchanged. Every
Claude-Code-shaped agent speaks the Anthropic Messages API, so the whole Qwen
fleet was unreachable while the servers behind the front were healthy.

PRIOR ART, checked before a line was written (the gate is part of the record):
``git log --all -S'/v1/messages' -- python/sglang/srt/weg2/`` = 0 commits, and
``FORWARD_PATHS`` has carried exactly three paths since it was introduced on
2026-09-07 in b614219e3e. The Anthropic Messages endpoint ITSELF is old and
works (``entrypoints/http_server.py`` ``@app.post("/v1/messages")``, tasks
#540/#557/#666/#764) -- what the user remembered as "already integrated" is
that server-side front, and it is exactly why nothing here reimplements the
protocol. The front only had to pass the path through, plus the two front-side
touches this file covers: PRICING and the STREAM ROLL-UP.

The three assertions, in the order they would have caught the bug:

1.  THE PATHS. ``/v1/messages`` is forwarded like ``/v1/chat/completions``;
    ``/v1/messages/count_tokens`` is a passthrough and NOT a forward, because
    it generates nothing and must not take a seat or provoke a flip.

2.  THE PRICE. The Anthropic body shape and the equivalent OpenAI body price
    IDENTICALLY, and both now include ``tools[]`` -- which NEITHER priced
    before. Under-pricing is not cosmetic here: ``price_remainder`` feeds the
    SHORT bound, so an under-priced long prompt routes SHORT to D, exceeds D's
    prefill cap and returns as W50. A Claude-Code agent carries 10-20k tokens
    of tool schemas, so dropping ``tools`` mis-routes the fleet's every turn.

3.  THE STREAM PRICE. Anthropic announces the prompt count ONCE, in
    ``message_start``, at the FRONT of the stream. ``leg2`` retains a bounded
    tail (256 KiB, trimmed to 128 KiB), so a tail scan -- the OpenAI-shaped
    instrument -- reads 0 on any answer longer than that window and the
    request lands in W28 'unpriced'. The accumulator is fed every chunk as it
    is forwarded, and the long-answer case below is the one a tail scan cannot
    pass.
"""

import json

import pytest

from sglang.srt.weg2 import front as F


# --------------------------------------------------------------------------
# 1. the paths
# --------------------------------------------------------------------------


def test_v1_messages_is_forwarded():
    assert "/v1/messages" in F.FORWARD_PATHS
    # the shape it must match, not a bare membership check
    assert "/v1/chat/completions" in F.FORWARD_PATHS


def test_count_tokens_is_a_passthrough_not_a_forward():
    """It decodes nothing: a seat, a Pending or a flip for it would be a bug."""
    assert "/v1/messages/count_tokens" in F.PASSTHROUGH_POST
    assert "/v1/messages/count_tokens" not in F.FORWARD_PATHS


def test_forward_paths_are_registered_as_post_routes():
    """The pin that the tuple is actually WIRED, not merely declared.

    ``desk-written-never-executed``: a path added to the tuple but never
    registered would pass every assertion above and still answer 404.
    """
    import inspect

    src = inspect.getsource(F)
    assert "for path in PASSTHROUGH_POST:" in src
    assert "app.router.add_post(path, front.handle_passthrough_post)" in src
    assert "for path in FORWARD_PATHS:" in src
    assert "app.router.add_post(path, front.handle_generate)" in src


# --------------------------------------------------------------------------
# 2. the price
# --------------------------------------------------------------------------

SYSTEM = "You are a coding agent. Follow the operating rules exactly." * 8
USER = "Explain what --user-reserve-mib does to the corridor floor." * 40
TOOLS = [
    {
        "name": f"tool_{i}",
        "description": "A tool that does something specific and is described at length. " * 6,
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
            "required": ["a"],
        },
    }
    for i in range(20)
]


def test_anthropic_and_openai_bodies_price_identically():
    """Same conversation, two wire shapes, one price.

    OpenAI carries the system turn as ``messages[0]``; Anthropic carries it in
    a top-level ``system`` field. If those priced differently, the same agent
    would be routed differently depending only on which endpoint it used.
    """
    openai_body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "tools": TOOLS,
    }
    anthropic_body = {
        "model": "m",
        "system": SYSTEM,
        "messages": [{"role": "user", "content": USER}],
        "tools": TOOLS,
    }
    assert F.request_text(openai_body) == F.request_text(anthropic_body)


def test_anthropic_system_as_block_list_prices_like_the_string():
    a = {"system": SYSTEM, "messages": [{"role": "user", "content": USER}]}
    b = {
        "system": [{"type": "text", "text": SYSTEM}],
        "messages": [{"role": "user", "content": USER}],
    }
    assert F.request_text(a) == F.request_text(b)


def test_tools_are_priced_at_all():
    """The 10-20k-token half of a Claude-Code turn.

    Before this ticket ``tools`` was priced on NEITHER wire shape.
    """
    without = {"messages": [{"role": "user", "content": USER}]}
    with_tools = {"messages": [{"role": "user", "content": USER}], "tools": TOOLS}
    n0 = len(F.request_text(without))
    n1 = len(F.request_text(with_tools))
    assert n1 > n0
    # the schemas are the bulk of the body, not a rounding term
    assert n1 - n0 > 3000


def test_tool_use_and_tool_result_blocks_are_priced():
    """Reading only ``text`` drops the entire tool half of the conversation."""
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me look"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "grep -rn FORWARD_PATHS " + "x" * 500},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "OUTPUT " + "y" * 2000,
                    }
                ],
            },
        ]
    }
    text = F.request_text(payload)
    assert "x" * 500 in text, "tool_use.input was dropped"
    assert "y" * 2000 in text, "tool_result.content was dropped"


def test_tool_result_with_block_list_content_is_priced():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "z" * 900}],
                    }
                ],
            }
        ]
    }
    assert "z" * 900 in F.request_text(payload)


def test_image_block_is_not_priced_as_its_base64_payload():
    """An over-price large enough to change the route is also a defect."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png",
                                   "data": "A" * 50000},
                    }
                ],
            }
        ]
    }
    assert len(F.request_text(payload)) < 200


def test_openai_shape_still_prices_as_before():
    """Backward compatibility: the shape that was already served."""
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "hi"},
        ]
    }
    assert F.request_text(payload) == "user:hello\nassistant:hi\n"


def test_generate_shape_still_prices_as_before():
    assert F.request_text({"text": "abc"}) == "abc"
    assert F.request_text({"prompt": ["a", "b"]}) == "a\nb"


# --------------------------------------------------------------------------
# 3. the leg-2 price: non-stream
# --------------------------------------------------------------------------


def test_usage_of_reads_the_anthropic_body():
    """Without this the front turns a healthy 200 into a W28 503."""
    body = {
        "type": "message",
        "usage": {
            "input_tokens": 18651,
            "output_tokens": 96,
            "cache_read_input_tokens": 18649,
        },
    }
    pt, ct, comp, priced = F.usage_of(body)
    assert priced is True
    assert (pt, ct, comp) == (18651, 18649, 96)


def test_usage_of_still_reads_the_openai_body():
    body = {"usage": {"prompt_tokens": 10, "completion_tokens": 3,
                      "prompt_tokens_details": {"cached_tokens": 7}}}
    assert F.usage_of(body) == (10, 7, 3, True)


def test_usage_of_unpriced_body_is_still_unpriced():
    """The fail-CLOSED contract must survive the new branch."""
    assert F.usage_of({"type": "message"})[3] is False
    assert F.usage_of({"usage": {}})[3] is False


# --------------------------------------------------------------------------
# 3b. the leg-2 price: stream
# --------------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _captured_stream(n_deltas: int = 3, input_tokens: int = 18651,
                     cache_read: int = 18649, output_tokens: int = 42) -> bytes:
    """A message_start / content_block_delta / message_delta / message_stop run."""
    out = bytearray()
    out += _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "Qwen3.8-27B", "content": [],
            "usage": {"input_tokens": input_tokens,
                      "cache_read_input_tokens": cache_read,
                      "output_tokens": 1},
        },
    })
    out += _sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    for _ in range(n_deltas):
        out += _sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "token "},
        })
    out += _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    out += _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    out += _sse("message_stop", {"type": "message_stop"})
    return bytes(out)


def test_stream_rollup_parses_a_captured_sequence():
    acc = F.AnthropicStreamUsage()
    acc.feed(_captured_stream())
    pt, ct, comp, priced = acc.result()
    assert priced is True
    assert (pt, ct, comp) == (18651, 18649, 42)
    assert acc.saw_start and acc.saw_stop


@pytest.mark.parametrize("size", [1, 7, 64, 999])
def test_stream_rollup_survives_arbitrary_chunk_boundaries(size):
    """Network chunks do not respect SSE line boundaries."""
    blob = _captured_stream()
    acc = F.AnthropicStreamUsage()
    for i in range(0, len(blob), size):
        acc.feed(blob[i:i + size])
    assert acc.result() == (18651, 18649, 42, True)


def test_stream_rollup_prices_an_answer_longer_than_the_retained_tail():
    """THE case a tail scan structurally cannot pass.

    ``leg2`` keeps at most 256 KiB and trims to the last 128 KiB, and the
    Anthropic prompt count is announced only in ``message_start`` at the very
    head. This reproduces that geometry: the head is pushed far out of any
    128 KiB window, and the price must still be right.
    """
    head = _captured_stream(n_deltas=0)
    # the deltas alone exceed the trim window several times over
    filler = b"".join(
        _sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "w" * 200},
        })
        for _ in range(4000)
    )
    assert len(filler) > 512 * 1024
    acc = F.AnthropicStreamUsage()
    acc.feed(head)
    acc.feed(filler)
    pt, ct, comp, priced = acc.result()
    assert priced is True and pt == 18651

    # and the instrument it replaces genuinely fails on this input, which is
    # why the accumulator exists (mutation-proof in situ, not an assertion
    # about our own code's shape)
    tail = bytearray(head + filler)
    if len(tail) > 262144:
        del tail[:-131072]
    assert F.usage_of_stream_tail(bytes(tail))[3] is False


def test_stream_rollup_is_unpriced_when_no_message_start_was_seen():
    """Fail-closed, same contract as ``usage_of``."""
    acc = F.AnthropicStreamUsage()
    acc.feed(_sse("message_stop", {"type": "message_stop"}))
    assert acc.result() == (0, 0, 0, False)


def test_stream_rollup_prices_a_stream_cut_short_after_message_start():
    """A truncated stream still carries a REAL input count; refusing to price
    it would re-introduce the W28 fail-closed this class exists to prevent."""
    acc = F.AnthropicStreamUsage()
    acc.feed(_captured_stream(n_deltas=1).split(b"event: message_delta")[0])
    pt, _, _, priced = acc.result()
    assert priced is True and pt == 18651


def test_openai_streams_are_untouched_by_the_new_path():
    """The tail scan still owns the OpenAI wire."""
    tail = b'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\ndata: [DONE]\n\n'
    assert F.usage_of_stream_tail(tail) == (5, 0, 2, True)


def test_stream_options_are_not_injected_into_a_messages_body():
    """``stream_options`` is an OpenAI field; sending it to /v1/messages risks
    a 400 from the very endpoint this ticket exists to reach."""
    import inspect

    src = inspect.getsource(F.Front.leg2)
    assert 'request.path != "/v1/messages"' in src
