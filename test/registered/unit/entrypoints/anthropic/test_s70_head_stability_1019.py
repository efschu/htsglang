"""S70 (NF rc16, 1019_121629 pdflip-94-158 / 96-161): the head system turn.

96-161 read 8384 of 89434 tokens from the store although 94-158 (same agent,
same images, same length) had published 89408 thirteen seconds earlier. The
store keys are a SHA-256 chain over the token ids; 96-161's chain left the
main chain inside page 131 -- the end of the head system turn. These tests
pin, over the REAL conversion (``AnthropicServing``), the REAL Qwen3.8-Flash-
Next chat template and tokenizer, and the REAL page-key chain:

* RED before S70 (strip off = the old path): two requests that differ only in
  Claude Code's ``x-anthropic-billing-header`` block leave the shared key chain
  in the head, and every page behind it is recomputed.
* GREEN with S70: the block is not rendered, the key chains are identical.
* DANGER DIRECTION: nothing but that one line is removed (a greedy mutant is
  caught), a user message is never touched, and two different images of the
  same size never share a pad value (no KV of a foreign image can be keyed as
  this image's).
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from flliper.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

import jinja2  # noqa: E402

from flliper.srt.entrypoints.anthropic import head_stability as hs  # noqa: E402
from flliper.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from flliper.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.utils import get_hash_str  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NF_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-AWQ-INT4-cyankiwi"
PAGE = 64

HEADER_A = "x-anthropic-billing-header: cc_version=2.1.288.7e3; cc_entrypoint=cli; cc_is_subagent=true;"
HEADER_B = "x-anthropic-billing-header: cc_version=2.1.288.7f3; cc_entrypoint=cli; cc_is_subagent=true;"
SYSTEM_TEXT = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n"
    + "You are an interactive agent that helps users with software engineering tasks. " * 40
)
TOOLS = [
    {
        "name": f"tool_{i}",
        "description": "Executes a command and returns its output. " * 12,
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run"}},
            "required": ["command"],
        },
    }
    for i in range(6)
]
MESSAGES = [
    {"role": "user", "content": "<system-reminder>\nAs you answer the user's questions...\n</system-reminder>\nFix the bug."},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Let me look.", "signature": "x"},
        {"type": "tool_use", "id": "toolu_1", "name": "tool_0", "input": {"command": "ls"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.py\nb.py\n" * 50},
    ]},
]


def _payload(header):
    system = [{"type": "text", "text": header}] if header else []
    system.append({"type": "text", "text": SYSTEM_TEXT})
    return {"model": "m", "max_tokens": 8, "system": system, "tools": TOOLS, "messages": MESSAGES}


class _FakeChat:
    def __init__(self, chat_template):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )

    def apply_reasoning_enabled(self, chat_request, enabled):
        pass

    def supports_native_reasoning_history(self):
        return True  # the Qwen3.8 template renders reasoning_content itself

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"


def _raise(msg):
    raise ValueError(msg)


class _Real:
    """The NF template and tokenizer, loaded once."""

    template = None
    tok = None

    @classmethod
    def load(cls):
        if cls.template is None:
            with open(os.path.join(NF_MODEL, "chat_template.jinja"), encoding="utf-8") as f:
                cls.template = f.read()
            from tokenizers import Tokenizer

            cls.tok = Tokenizer.from_file(os.path.join(NF_MODEL, "tokenizer.json"))
        return cls


def _ids(serving, payload):
    req = serving._convert_to_chat_completion_request(
        AnthropicMessagesRequest.model_validate(payload)
    )
    msgs = []
    for m in req.messages:
        d = m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else dict(m)
        if not isinstance(d.get("content"), str):
            d["content"] = "".join(
                p.get("text", "") for p in (d.get("content") or []) if isinstance(p, dict)
            )
        # as serving_chat hands them to the template: arguments as a mapping
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"] or "{}")
        msgs.append(d)
    tools = [t.model_dump(exclude_none=True) for t in (req.tools or [])]
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = _raise
    text = env.from_string(_Real.template).render(
        messages=msgs, tools=tools, add_generation_prompt=True
    )
    return text, _Real.tok.encode(text, add_special_tokens=False).ids


def _pages_shared(a, b):
    n = min(len(a), len(b)) // PAGE * PAGE
    ka = get_hash_str(a[:n], None, page_size=PAGE)
    kb = get_hash_str(b[:n], None, page_size=PAGE)
    shared = 0
    for x, y in zip(ka, kb):
        if x != y:
            break
        shared += 1
    return shared, n // PAGE


@unittest.skipUnless(os.path.isdir(NF_MODEL), "NF tokenizer/template not on this host")
class TestHeadKeyChain(unittest.TestCase):
    def setUp(self):
        _Real.load()
        with envs.FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(True):
            self.serving = AnthropicServing(_FakeChat(_Real.template))

    def test_red_old_path_breaks_the_chain_in_the_head(self):
        """Strip off (the pre-S70 path): the chain leaves at the header's page,
        right behind the tools block -- the 96-161 shape (page 131 of 1397)."""
        with mock.patch.dict(os.environ, {hs.ENV: "0"}):
            ta, a = _ids(self.serving, _payload(HEADER_A))
            tb, b = _ids(self.serving, _payload(HEADER_B))
        self.assertEqual(len(a), len(b))  # same length, as 94-158 / 96-161
        shared, full = _pages_shared(a, b)
        self.assertLess(shared, full)
        cut = tb.index("7f3")
        self.assertGreater(cut, tb.index("</IMPORTANT>"))  # behind the tools block
        self.assertEqual(shared, len(_Real.tok.encode(ta[:cut], add_special_tokens=False).ids) // PAGE)

    def test_green_strip_keeps_the_chain(self):
        ta, a = _ids(self.serving, _payload(HEADER_A))
        tb, b = _ids(self.serving, _payload(HEADER_B))
        self.assertNotIn("x-anthropic-billing-header", ta)
        self.assertEqual(a, b)
        shared, full = _pages_shared(a, b)
        self.assertEqual(shared, full)
        # and it equals the request that never carried the block
        _t0, c = _ids(self.serving, _payload(None))
        self.assertEqual(a, c)

    def test_front_text_moves_with_the_adapter(self):
        from flliper.srt.pdflip.front import request_text

        ra = request_text(_payload(HEADER_A), tools_first=True)
        rb = request_text(_payload(HEADER_B), tools_first=True)
        self.assertEqual(ra, rb)
        self.assertNotIn("x-anthropic-billing-header", ra)
        with mock.patch.dict(os.environ, {hs.ENV: "0"}):
            self.assertNotEqual(
                request_text(_payload(HEADER_A), tools_first=True),
                request_text(_payload(HEADER_B), tools_first=True),
            )


def _greedy_mutant(text):
    """A strip that drops the WHOLE block once it contains the tag."""
    if "x-anthropic-billing-header:" in text:
        return "", "x"
    return text, None


class TestDangerDirection(unittest.TestCase):
    CASES = [
        # (input, expected output)
        (HEADER_A, ""),
        ("  " + HEADER_A + "  ", ""),
        (HEADER_A + "\nYou are Claude Code.\nMore.", "You are Claude Code.\nMore."),
        ("Docs: the line x-anthropic-billing-header: cc_version=1; is analytics.", None),
        ("You are Claude Code.\n" + HEADER_A, None),  # not the first line
        ("x-anthropic-billing-header: free text without fields", None),
        ("x-anthropic-billing-header: cc_version=1; and prose after it", None),
    ]

    def _check(self, fn):
        for src, want in self.CASES:
            out, line = fn(src)
            if want is None:
                self.assertEqual(out, src, src)
                self.assertIsNone(line, src)
            else:
                self.assertEqual(out, want, src)

    def test_only_the_leading_line_goes(self):
        self._check(hs.strip_attribution_text)

    def test_greedy_mutant_is_caught(self):
        with self.assertRaises(AssertionError):
            self._check(_greedy_mutant)

    def test_parts_drop_only_the_block(self):
        out, stripped = hs.strip_attribution_parts([HEADER_A, SYSTEM_TEXT])
        self.assertEqual(out, [SYSTEM_TEXT])
        self.assertEqual(stripped, [HEADER_A])

    def test_user_text_is_never_touched(self):
        with envs.FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(True):
            s = AnthropicServing(_FakeChat("{{ messages }}"))
        p = {
            "model": "m", "max_tokens": 8, "system": "sys",
            "messages": [{"role": "user", "content": HEADER_A + "\nquestion"}],
        }
        req = s._convert_to_chat_completion_request(AnthropicMessagesRequest.model_validate(p))
        self.assertEqual(req.messages[-1].content, HEADER_A + "\nquestion")

    def test_switch_off_is_byte_identical(self):
        with mock.patch.dict(os.environ, {hs.ENV: "0"}):
            self.assertFalse(hs.env_on())
        self.assertTrue(hs.env_on({}))


class TestForeignImageNeverShares(unittest.TestCase):
    """The (a) danger direction: an image's pad value (its radix/store key) is
    its CONTENT hash -- two different images of identical size never share it,
    the same pixels always do."""

    def test_pad_value_follows_content(self):
        import torch

        from flliper.srt.managers.schedule_batch import Modality, MultimodalDataItem

        g = torch.Generator().manual_seed(7)
        px = torch.rand(64, 1536, generator=g)
        other = px.clone()
        other[17, 3] += 1e-3  # one value differs, same shape

        def pad(t):
            it = MultimodalDataItem(modality=Modality.IMAGE, feature=t.clone())
            it.set_pad_value()
            return it.pad_value

        self.assertEqual(pad(px), pad(px.clone()))
        self.assertNotEqual(pad(px), pad(other))


class TestHeadDiverge(unittest.TestCase):
    def test_names_the_moved_bytes(self):
        w = hs.HeadWatch(window=4)
        base = "tools...\n" + "S" * 400
        self.assertIsNone(w.observe(base + HEADER_A + "tail"))
        self.assertIsNone(w.observe(base + HEADER_A + "tail"))  # identical: silent
        rec = w.observe(base + HEADER_B + "tail")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["at_char"], len(base) + HEADER_A.index("7e3") + 1)  # "7e3" vs "7f3"
        self.assertIn("7e3", rec["prev"])
        self.assertIn("7f3", rec["new"])

    def test_adapter_names_a_reordered_tool_list(self):
        """Through the adapter: the same tools in another order move the head
        inside the TOOLS part -- the instrument sees the block the template
        renders first."""
        w = hs.HeadWatch(window=4)
        with envs.FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(True):
            s = AnthropicServing(_FakeChat("{{ messages }}"))
        p1 = dict(_payload(HEADER_A))
        p2 = dict(_payload(HEADER_A), tools=TOOLS[:2] + [TOOLS[3], TOOLS[2]] + TOOLS[4:])
        with mock.patch.object(hs, "_WATCH", w):
            s._convert_to_chat_completion_request(AnthropicMessagesRequest.model_validate(p1))
            s._convert_to_chat_completion_request(AnthropicMessagesRequest.model_validate(p2))
        self.assertEqual(w.n, 1)
        h1, h2 = list(w._heads)
        cut = hs._common_prefix_len(h1, h2)
        self.assertLess(cut, h1.index("SYSTEM:"))  # the move is in the tools part
        self.assertIn('"name": "tool_2"', h1[cut - 40:cut + 200])

    def test_unrelated_heads_are_silent(self):
        w = hs.HeadWatch(window=4)
        w.observe("A" * 1000)
        self.assertIsNone(w.observe("B" * 1000))
        self.assertIsNone(w.observe("A" * 100 + "C" * 900))  # shares < MIN_SHARED_CHARS


if __name__ == "__main__":
    unittest.main()
