"""P49 -- the 27B #49 agent-turn pricing on the NF line, behind ONE switch, on H100.

THE CHAIN. rc2.1l (d1c7094ba6) -> H100 (RC7 (4b) of the 27B line, 71460c8130,
unconditional: the /v1/messages stream's prompt and cached counts) -> #49
(196f6a8f57) -> this NF commit (the switch). rc9o (dkrnfbar1rc9o09260808, NF on
2eed285057, qwen agent load) routed 31 of 36 requests LONG and flipped 34
times; every streamed D leg 2 booked e.g. ``prompt_tokens=3 cached_tokens=0``
(weg2-14-11, a 22,931-token prompt P had prefilled) -- a MEASURED ZERO that
retracts every credit (#1324). H100 fixes that wire; #49 is built on the
presence H100 makes visible (its held credit and its measured-token prefix
price read the same response's prompt_tokens -- with the rc2.1l wire that
would be prompt_tokens=3).

THE SWITCH. ``SGLANG_WEG2_ENABLE_AGENT_SPAN`` (default off, until the user
decides -- memory kein-d-direct-prefill-ueber-x). Off must be rc2.1l+H100 byte
for byte: request_text, the span LRU's pricing and the ROUTE-VERDICT line are
rc2.1l's (H100 touches neither); the Messages usage counts are H100's. The
rc2.1l implementations are embedded below as the reference (copied from
d1c7094ba6) and compared on the measured replay.

SEPARATION (test_separation_rc21l_h100_h49): the replay of the measured 27B
agent sequence under three wires/pricings -- rc2.1l (streamed D legs book a
measured zero), H100 alone (switch off), H100 + #49 (switch on) -- at several
tool-block sizes.

FORM A. Nothing here is rank-local: the front prices from D's response
(prompt_tokens / cached_tokens of the group's answer, TP0's radix under H98),
and D's own X-GATE (group-MIN match, then W31 -> W50 -> re-route to P before
the first byte) bounds any over-credit exactly as on the 27B.
"""

import ast
import collections
import hashlib
import importlib.util
import inspect
import json
import os
import textwrap

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_27b_tests():
    spec = importlib.util.spec_from_file_location(
        "_p49_span_49", os.path.join(HERE, "test_weg2_front_span_agent_49.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T49 = _load_27b_tests()


def _switch(value):
    try:
        fld = envs.SGLANG_WEG2_ENABLE_AGENT_SPAN
    except AttributeError:  # the base (rc2.1l) has no switch: it IS "off"
        import contextlib

        return contextlib.nullcontext()
    return fld.override(value)


@pytest.fixture
def on():
    with _switch(True):
        yield


@pytest.fixture
def off():
    with _switch(False):
        yield


# --------------------------------------------------------------------------
# the rc2.1l reference (d1c7094ba6, python/sglang/srt/weg2/front.py)
# --------------------------------------------------------------------------


def ref_request_text(payload):
    if "messages" in payload and isinstance(payload["messages"], list):
        parts = []
        sys_field = payload.get("system")
        if sys_field:
            parts.append(f"system:{F._content_text(sys_field)}\n")
        for m in payload["messages"]:
            if not isinstance(m, dict):
                parts.append(f"{m}\n")
                continue
            parts.append(f"{m.get('role', '')}:{F._content_text(m.get('content', ''))}\n")
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            parts.append(
                "tools:" + json.dumps(tools, ensure_ascii=False, sort_keys=True) + "\n"
            )
        return "".join(parts)
    p = payload.get("prompt", payload.get("text", ""))
    if isinstance(p, list):
        return "\n".join(str(x) for x in p)
    return str(p)


class RefSpanLRU:
    def __init__(self, cap=F.SPAN_LRU):
        self.cap = cap
        self.entries = collections.OrderedDict()

    def record_presence(self, text, cached_tokens, **_ignored):
        if not text:
            return
        key = hashlib.sha1(text.encode()).hexdigest()
        self.entries.pop(key, None)
        if int(cached_tokens) <= 0:
            return
        self.entries[key] = (text, int(cached_tokens))
        while len(self.entries) > self.cap:
            self.entries.popitem(last=False)

    def span_tokens(self, text):
        best = 0
        known = False
        for etext, etok in self.entries.values():
            cp = F.common_prefix_len(etext, text)
            if cp <= 0:
                continue
            known = True
            est = int(etok * (cp / max(1, len(etext))))
            best = max(best, est)
        return best, known


def ref_price_remainder(text, spans, epoch=None):
    est_prompt = int(len(text) / F.CHARS_PER_TOKEN) + 1
    span, known = spans.span_tokens(text)
    return max(0, est_prompt - span), est_prompt, known


REF_ROUTE_VERDICT = (
    "WEG2 ROUTE-VERDICT rid=%s verdict=%s uncached=%d (base for X=%d, what D must PREFILL, "
    "at CHARS_PER_TOKEN=%.1f minus the MEASURED cached-on-D presence) presence_span=%d "
    "presence_src=%s (#1324: the credit's witness -- d_leg2_cached is a realised "
    "cached_tokens reading from group D, never a prefill on P) carrier_est=%d src=%s "
    "(THE COMPARED VALUE for carrier_max=%d: the WHOLE prompt's KV through the host staging "
    "pool, at CARRIER_CHARS_PER_TOKEN=%.1f) est_prompt=%d chars=%d (#1290)"
)


# --------------------------------------------------------------------------
# the replay, driven by either implementation
# --------------------------------------------------------------------------


def replay(bodies, rt, spans, price, zero_wire=False):
    """T49.replay with the implementation injected (same group model).

    ``zero_wire``: the rc2.1l /v1/messages stream -- every D leg 2 books
    cached_tokens=0 and prompt_tokens=uncached (rc9o: 3 for weg2-14-11).
    """
    X, CARRIER_MAX = T49.X, T49.CARRIER_MAX
    awake, epoch, flips = "P", 0, 0
    radix, rows = [], []
    for rid, kind, body, pt in bodies:
        text = rt(body)
        try:
            rem, est, known = price(text, spans, epoch if awake == "D" else None)
        except TypeError:  # rc2.1l's own price_remainder (run on the base)
            rem, est, known = price(text, spans)
        carrier_est = int(len(text) / F.CARRIER_CHARS_PER_TOKEN) + 1
        route = F.serviceable_route(rem, carrier_est, X, CARRIER_MAX)
        d_rest = pt - max([p for t, p in radix if text.startswith(t)] or [0]) \
            if awake == "D" else pt
        if awake == "D" and route == "short":
            ct, path = pt - d_rest, "D-direct"
            radix.append((text, pt))
        else:
            flips += 2 if awake == "D" else 1
            epoch += 2 if awake == "D" else 1
            awake, ct, path = "D", pt - 2, "P+flips"
            radix = [(text, pt)]
        if zero_wire:
            ct, pt = 0, max(1, pt - ct)
        try:
            spans.record_presence(text, ct, prompt_tokens=pt, held_epoch=epoch)
        except TypeError:  # rc2.1l's own SpanLRU (run on the base)
            spans.record_presence(text, ct)
        rows.append((rid, rem, est, known, route, path))
    return rows, flips


# --------------------------------------------------------------------------
# (1) the switch itself
# --------------------------------------------------------------------------


def test_switch_is_registered_and_off_by_default(monkeypatch):
    """Unified tree (operator 26.09.): the default is the profile's
    (weg2/form.py ModelProfile.agent_span, pinned in
    test_unify_agent_span_profile.py) -- off without a form and for
    nextflash."""
    fld = envs.SGLANG_WEG2_ENABLE_AGENT_SPAN
    monkeypatch.delenv("SGLANG_WEG2_FORM", raising=False)
    if not fld.is_set():
        assert fld.get() is False, "default off until the NF seat releases it"
        assert F.SpanLRU().agent_span is False


# --------------------------------------------------------------------------
# (2) off == rc2.1l
# --------------------------------------------------------------------------


def test_off_replay_is_rc21l_row_for_row(off):
    bodies = T49.build_bodies()
    got, flips = replay(bodies, F.request_text, F.SpanLRU(), F.price_remainder)
    ref, ref_flips = replay(bodies, ref_request_text, RefSpanLRU(), ref_price_remainder)
    assert got == ref
    assert flips == ref_flips
    # and rc2.1l is the defect: every warm turn over P
    assert all(r[5] == "P+flips" for r in got)


def test_off_request_text_is_rc21l(off):
    tools = [{"name": "Bash", "description": "run", "input_schema": {"x": 1}}]
    for body in (
        {"system": "S", "messages": [{"role": "user", "content": "hi"}], "tools": tools},
        {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
         "tools": tools},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]},
        {"prompt": "plain"}, {"text": ["a", "b"]},
    ):
        assert F.request_text(body) == ref_request_text(body)
    assert F.request_text({"messages": [], "tools": tools}).startswith("tools:")


def test_off_span_entry_is_the_presence_witness_only(off):
    s = F.SpanLRU()
    assert s.agent_span is False
    s.record_presence("t" * 900, 200, prompt_tokens=300, held_epoch=4)
    ((_, ct, pt, held),) = s.entries.values()
    assert (ct, pt, held) == (200, 0, None)
    # a held D serve with ct=0 is a retraction, as on rc2.1l
    s.record_presence("t" * 900, 0, prompt_tokens=300, held_epoch=4)
    assert not s.entries


def test_off_span_tokens_is_rc21l_uncapped(off):
    """H61's presence pins read span_tokens: off it is rc2.1l's number, not
    capped by the chars/3 estimate of a short text (13 vs 12668 on H61)."""
    s, r = F.SpanLRU(), RefSpanLRU()
    for t, ct in (("q" * 40, 12668), ("p" * 900, 250), ("p" * 1200, 0), ("z" * 70, 9)):
        s.record_presence(t, ct, prompt_tokens=ct + 3, held_epoch=2)
        r.record_presence(t, ct)
    for probe in ("q" * 40, "q" * 20, "p" * 1000, "p" * 900 + "x", "z" * 80, "none"):
        assert s.span_tokens(probe) == r.span_tokens(probe), probe


@pytest.mark.parametrize("value", [False, True])
def test_h100_usage_does_not_depend_on_the_switch(value):
    """H100 is unconditional: the switch moves #49 only."""
    with _switch(value):
        body = {"type": "message", "usage": {"input_tokens": 2, "cache_read_input_tokens": 18649,
                                             "output_tokens": 96}}
        assert F.usage_of(body) == (18651, 18649, 96, True)
        acc = F.AnthropicStreamUsage()
        acc.feed(_adapter_stream({"input_tokens": 3, "cache_read_input_tokens": 22928,
                                  "output_tokens": 1401}))
        # rc9o's line for weg2-14-11 read prompt_tokens=3 cached_tokens=0
        assert acc.result() == (22931, 22928, 1401, True)


def test_off_route_verdict_line_is_rc21l_byte_for_byte():
    src = inspect.getsource(F.Front.handle_generate)
    fmt, nargs = None, None
    for n in ast.walk(ast.parse(textwrap.dedent(src))):
        if (isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)
                and n.args[0].value.startswith("WEG2 ROUTE-VERDICT")):
            fmt, nargs = n.args[0].value, len(n.args) - 1
    assert fmt is not None
    vals = ("weg2-3-4", "short", 812, 4096, 3.0, 20100, "d_leg2_cached", 30000, "estimate",
            314553, 2.4, 21000, 63000)
    assert nargs == len(vals) + 1  # + the #49 suffix
    off_line = fmt % (vals[:7] + ("",) + vals[7:])
    assert off_line == REF_ROUTE_VERDICT % vals
    assert "getattr(self.spans, \"agent_span\", False)" in src


# --------------------------------------------------------------------------
# (3) on: the port acts on NF's wire and NF's template
# --------------------------------------------------------------------------


def test_on_replay_keeps_agent_turns_on_d(on):
    bodies = T49.build_bodies()
    rows, flips = replay(bodies, F.request_text, F.SpanLRU(), F.price_remainder)
    assert flips == 5, flips
    assert sum(r[5] == "D-direct" for r in rows) == 42


def _adapter_stream(final_usage):
    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    return b"".join([
        sse("message_start", {"type": "message_start", "message": {
            "id": "m", "type": "message", "role": "assistant", "model": "m", "content": [],
            "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                              "usage": final_usage}),
        sse("message_stop", {"type": "message_stop"}),
    ])


def test_on_rc9o_wire_teaches_the_span(on):
    """rc9o weg2-14-11 -> weg2-15-12 on the Messages stream, with the switch on.

    D answered the 22,931-token prompt (P prefilled it, 3 uncached on D); the
    next turn appends ~500 chars. rc2.1l booked (3, 0) = a retraction, so the
    next turn priced its whole 24.8k estimate and went LONG over P.
    """
    acc = F.AnthropicStreamUsage()
    acc.feed(_adapter_stream({"input_tokens": 3, "cache_read_input_tokens": 22928,
                              "output_tokens": 1401}))
    pt, ct, _, priced = acc.result()
    assert (pt, ct, priced) == (22931, 22928, True)
    spans = F.SpanLRU()
    t0 = "x" * 74051  # weg2-14-11 chars
    spans.record_presence(t0, ct, prompt_tokens=pt, held_epoch=16)
    rem, est, known = F.price_remainder(t0 + "y" * 509, spans, epoch=16)  # weg2-15-12
    assert known and rem <= 200, rem
    assert F.serviceable_route(rem, est, 4096, 314553) == "short"


def test_on_a_real_rest_above_x_still_routes_long(on):
    """kein-d-direct-prefill-ueber-x on NF: a 20k paste on a warm D goes to P."""
    spans = F.SpanLRU()
    t = "c" * 150000
    spans.record_presence(t, 50000, prompt_tokens=50000, held_epoch=4)
    rem, est, known = F.price_remainder(t + "z" * 60000, spans, epoch=4)
    assert known and rem >= 20000
    assert F.serviceable_route(rem, est, 4096, 314553) == "long"


_NF_TEMPLATE = ("/spinning/llm_stuff/club-3090/models-cache/"
                "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist/chat_template.jinja")


@pytest.mark.skipif(not os.path.exists(_NF_TEMPLATE), reason="NF model not on this host")
def test_nf_template_renders_tools_first():
    """(A)'s premise on NF: the Flash-Next template emits <tools> at the head
    of the system turn, before the system text and every message."""
    t = open(_NF_TEMPLATE).read()
    i_tools = t.index("<tools>")
    i_sys = t.index("messages[0].content", t.index("if tools and tools is iterable"))
    i_loop = t.index("for message in messages")
    assert i_tools < i_sys < i_loop


@pytest.mark.parametrize("tool_chars", [3000, 6000, T49.TOOL_BLOCK_CHARS])
def test_separation_rc21l_h100_h49(tool_chars, capsys):
    """What H100 alone brings, and what #49 adds on top (flips, D-direct turns)."""
    bodies = T49.build_bodies(tool_chars)
    out = {}
    with _switch(False):
        rows, fl = replay(bodies, F.request_text, F.SpanLRU(), F.price_remainder, zero_wire=True)
        out["rc2.1l"] = (fl, sum(r[5] == "D-direct" for r in rows))
        rows, fl = replay(bodies, F.request_text, F.SpanLRU(), F.price_remainder)
        out["H100"] = (fl, sum(r[5] == "D-direct" for r in rows))
    with _switch(True):
        rows, fl = replay(bodies, F.request_text, F.SpanLRU(), F.price_remainder)
        out["H100+#49"] = (fl, sum(r[5] == "D-direct" for r in rows))
    print(f"tool_chars={tool_chars} " + " ".join(
        f"{k}: flips={v[0]} d_direct={v[1]}" for k, v in out.items()))
    # rc2.1l's wire teaches nothing: every warm turn over P
    assert out["rc2.1l"][1] == 0
    # each step is never worse than the one before it
    assert out["rc2.1l"][0] >= out["H100"][0] >= out["H100+#49"][0]
    # with the switch on, only the two real rests above X flip (1 + 2x2)
    assert out["H100+#49"][0] == 5


def _rc9i_pair():
    """rc9i front.log:566 (R28): weg2-4-6 after weg2-1-4 on the NF agent wire.

    weg2-1-4 went over P; D's leg 2 held 48152 tokens of it. The next turn's
    ROUTE-VERDICT read presence_span=43846 = 48152 x 0.9106: the character
    match stopped in front of the 13,302-char tool block (8.94 % of the text,
    ~4.4k est tokens), rendered LAST by rc2.1l's request_text.
    """
    tools = T49._tools(T49.TOOL_BLOCK_CHARS)
    system = "You are Claude Code. " * 200
    base = {"system": system, "tools": tools, "stream": True}
    total = int(T49.TOOL_BLOCK_CHARS / (1 - 0.9106))  # the text length that ratio implies
    body_chars = total - T49.TOOL_BLOCK_CHARS - len(f"system:{system}\n")
    m0 = [{"role": "user", "content": "u" * (body_chars - len("user:\n"))}]
    b0 = {**base, "messages": m0}
    b1 = {**base, "messages": m0 + [{"role": "assistant", "content": "a" * 700},
                                    {"role": "user", "content": "r" * 800}]}
    return b0, b1


@pytest.mark.parametrize("value", [False, True])
def test_rc9i_weg2_4_6_h100_alone_stays_long_the_port_goes_short(value):
    """R28: H100 alone moves 0 of 93 waiting turns on rc9i; tools-first plus the
    measured-token prefix price is what makes this turn SHORT."""
    b0, b1 = _rc9i_pair()
    with _switch(value):
        spans = F.SpanLRU()
        t0, t1 = F.request_text(b0), F.request_text(b1)
        # weg2-1-4's D leg 2 after P (a 200 D served, epoch 5), as H100 reads it
        try:
            spans.record_presence(t0, 48152, prompt_tokens=48155, held_epoch=5)
            rem, est, known = F.price_remainder(t1, spans, epoch=5)
        except TypeError:  # the base's rc2.1l API
            spans.record_presence(t0, 48152)
            rem, est, known = F.price_remainder(t1, spans)
    route = F.serviceable_route(rem, est, 4096, 314553)
    if not value:
        # the rc9i reading, reproduced: the credit stops at the tool block
        span = est - rem
        assert abs(span / 48152 - 0.9106) < 0.002, span
        assert rem > 4096 and route == "long"
    else:
        assert rem <= 600, rem  # the 1.5k appended chars at chars/3, nothing else
        assert route == "short"
