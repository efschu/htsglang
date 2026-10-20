# SPDX-License-Identifier: Apache-2.0
"""PREFIX-DIVERGE-TEXT (NF y7n cad1bf38f9, 02.10.): the decoded text around a
long re-prefill's divergence point, the segment/message it falls into, and
the two client payloads compared message by message.

y7n: pdflip-2-7 (6588) -> pdflip-6-10 (82044, ``common=6449``) -> pdflip-10-13
(112508, ``common=7169``) -> pdflip-12-14 (155044, ``common=7569``); each
``common`` lies 5 tokens into an assistant segment
(``<|im_start|>assistant\\n<think>\\n``) -- the line must say whether the
client re-sent that assistant turn differently (``hint=client_changed``) or
our render of an identical turn changed (``hint=render_only``).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import re

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_diverge_text as DT  # noqa: E402
from flliper.srt.pdflip.front_tokens import Count, TokenSpans  # noqa: E402

X = 4000
SPECIALS = {"<|im_start|>": 1, "<|im_end|>": 2}
INV = {v: k for k, v in SPECIALS.items()}


class _CharTok:
    """One token per character, the two chat specials as single ids."""

    def encode(self, text, add_special_tokens=False):
        out = []
        for part in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            if part in SPECIALS:
                out.append(SPECIALS[part])
            else:
                out.extend(ord(ch) + 10 for ch in part)
        return out

    def decode(self, ids):
        return "".join(INV.get(int(i), None) or chr(int(i) - 10) for i in ids)


def _render(messages, keep_last_reasoning_only=False):
    """A Qwen-shaped template; ``keep_last_reasoning_only`` = a template that
    renders reasoning only on the LAST assistant turn (position-dependent)."""
    last_a = max([i for i, m in enumerate(messages) if m["role"] == "assistant"], default=-1)
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            r = m.get("reasoning_content") or ""
            if keep_last_reasoning_only and i != last_a:
                r = ""
            out.append(f"<|im_start|>assistant\n<think>\n{r}\n</think>\n\n{m['content']}<|im_end|>\n")
        else:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    out.append("<|im_start|>assistant\n<think>\n")
    return "".join(out)


class _FTok:
    state = "ready"
    why = ""
    executor = None

    def __init__(self, positional=False):
        self.m = {}
        self._tok = _CharTok()
        self.positional = positional

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        ids = np.asarray(self._tok.encode(_render(payload["messages"], self.positional)),
                         dtype=np.int32)
        return Count(n=int(ids.size), ids=ids, ms=1.0, reused=0, encoded=int(ids.size))


def _front(positional=False):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = "D"
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _FTok(positional)
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = None
    f._store_probe_t = 0.0
    f.STORE_PROBE_RETRY_S = 1e9
    return f


def _price(f, rid, messages, **knobs):
    payload = dict(model="m", messages=messages, stream=True, **knobs)
    return asyncio.run(f._x_exact_price(rid, "/v1/chat/completions", payload, rid, 1, 1))


def _text_lines(caplog, rid):
    return [m for m in caplog.messages if m.startswith(f"PDFLIP PREFIX-DIVERGE-TEXT rid={rid} ")]


SYS = {"role": "system", "content": "S" * 3000}
U1 = {"role": "user", "content": "hello there"}
A1 = {"role": "assistant", "content": "Hi! How can I help?",
      "reasoning_content": "The user greets me. Reply briefly."}
U2 = {"role": "user", "content": "x" * 120}
A2 = {"role": "assistant", "content": "ok", "reasoning_content": "fine"}
U3 = {"role": "user", "content": "PASTE " + "p" * 7000}


def test_client_resends_an_earlier_assistant_turn_without_its_reasoning(caplog):
    """The y7n shape: turn 3 re-sends turn 1's answer with different
    reasoning; the divergence is 5 tokens into that assistant segment."""
    caplog.set_level(logging.INFO)
    f = _front()
    _price(f, "pdflip-2-7", [SYS, U1, A1, U2])
    a1_stripped = {"role": "assistant", "content": A1["content"]}
    x = _price(f, "pdflip-6-10", [SYS, U1, a1_stripped, U2, A2, U3])
    assert x.pending > X
    (dv,) = [m for m in caplog.messages if m.startswith("PDFLIP PREFIX-DIVERGE rid=pdflip-6-10 ")]
    common = int(re.search(r" common=(\d+) ", dv).group(1))
    # [SYS, U1] + its generation prompt ends exactly where turn 1's answer
    # header "<|im_start|>assistant\n<think>\n" ends (19 ids here, 5 on the
    # real tokenizer: y7n's common=6449 = 6444 + 5)
    assert common == len(_CharTok().encode(_render([SYS, U1])))
    (line,) = _text_lines(caplog, "pdflip-6-10")
    assert f" best_prev=pdflip-2-7 common={common} " in line
    assert " seg=2 seg_role=assistant seg_char=30 msg_guess=2 " in line
    assert " msgs_same=2/4->6 " in line
    assert ("first_diff=msg=2 role=assistant/assistant "
            "prev={content:str19,reasoning_content:str34,role:str9} "
            "cur={content:str19,role:str9}") in line
    assert " hint=client_changed " in line
    assert "<|im_start|>assistant\\n<think>\\n'" in line
    assert "prev_after='The user greets me." in line
    assert "cur_after='\\n</think>\\n\\nHi! How can I help?" in line
    assert f.counters["prefix_diverge_text"] == 1


def test_identical_messages_rendered_differently_is_render_only(caplog):
    """A template that keeps reasoning only on the last assistant turn: the
    client re-sends turn 1 byte-identically, our render drops its reasoning
    once it is no longer last -- the line blames the render, not the client."""
    caplog.set_level(logging.INFO)
    f = _front(positional=True)
    _price(f, "pdflip-2-7", [SYS, U1, A1, U2])
    _price(f, "pdflip-6-10", [SYS, U1, A1, U2, A2, U3])
    (line,) = _text_lines(caplog, "pdflip-6-10")
    assert " seg=2 seg_role=assistant " in line and " msg_guess=2 " in line
    assert " msgs_same=4/4->6 first_diff=none (cur extends prev) " in line
    assert " hint=render_only " in line


def test_a_changed_render_knob_is_named(caplog):
    caplog.set_level(logging.INFO)
    f = _front()
    _price(f, "pdflip-0-1", [SYS, U1, A1, U2], tools=[{"type": "function", "name": "a"}])
    a1_stripped = {"role": "assistant", "content": A1["content"]}
    _price(f, "pdflip-2-3", [SYS, U1, a1_stripped, U2, A2, U3])
    (line,) = _text_lines(caplog, "pdflip-2-3")
    assert " first_diff=req prev={tools:list1} cur={} " in line
    assert " hint=client_changed " in line


def test_short_or_unshared_arrivals_print_no_text(caplog):
    caplog.set_level(logging.INFO)
    f = _front()
    _price(f, "pdflip-0-1", [SYS, U1])
    _price(f, "pdflip-0-2", [SYS, U1, A1, U2])          # pending small: no P leg, no line
    _price(f, "pdflip-0-3", [{"role": "user", "content": "q" * 9000}])  # shares < a page
    assert not [m for m in caplog.messages if m.startswith("PDFLIP PREFIX-DIVERGE-TEXT")]
    assert set(f._recent_prompts.meta) == {"pdflip-0-1", "pdflip-0-2", "pdflip-0-3"}


def test_helpers_segments_tools_caps_and_surrogates():
    tok = _CharTok()
    msgs = [{"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"x": 1}]},
            {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"},
            {"role": "assistant", "content": "done"}]
    roles = ["system", "user", "assistant", "user", "assistant", "assistant"]
    assert [DT.message_of_segment(msgs, s, roles) for s in range(6)] == \
        ["sys", "0", "1", "2-3", "4", "gen"]
    text = "<|im_start|>user\nabc<|im_end|>\n<|im_start|>assistant\nxyz"
    ids = np.asarray(tok.encode(text), dtype=np.int32)
    seg, role, off, rl = DT.segment_at(tok, ids, len(tok.encode(text)) - 1)
    assert (seg, role, rl) == (1, "assistant", ["user", "assistant"])
    assert off == len("<|im_start|>assistant\nxy")
    sur = np.asarray([ord("a") + 10, (1 << 30) + 5, (1 << 30) + 5, ord("b") + 10], dtype=np.int64)
    assert DT._decode(tok, sur) == "a<img*2>b"
    assert len(DT._cap("z" * 5000)) == DT.CAP_CHARS + 3
    p = DT.payload_digest(msgs, {})
    same, diff, idx = DT.first_message_diff(p, DT.payload_digest(msgs[:3], {}))
    assert (same, idx) == (3, 3) and diff.startswith("cur_shorter")
    assert DT.payload_digest(None, {}) is None
