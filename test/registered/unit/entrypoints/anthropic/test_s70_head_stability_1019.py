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

from flliper.srt.entrypoints import prompt_diverge as pd  # noqa: E402
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


class TestStripNote(unittest.TestCase):
    """Auflage 4: changed= is per head, and the same request converted again
    (front count, D probe, PLE hint, group) stays silent."""

    def setUp(self):
        hs._STRIP_SEEN.clear()
        self.rate = mock.patch.object(hs, "_STRIP_RATE", hs.RateLimit(per_s=1e9, burst=1e9))
        self.rate.start()

    def tearDown(self):
        self.rate.stop()

    def test_per_head_first_then_changed_then_silent(self):
        k1 = hs.head_key([SYSTEM_TEXT], TOOLS)
        k2 = hs.head_key(["another agent"], TOOLS)
        r = hs.note_stripped([HEADER_A], k1)
        self.assertTrue(r["first"])
        self.assertIsNone(hs.note_stripped([HEADER_A], k1))  # the same request again
        r = hs.note_stripped([HEADER_A], k2)  # another head: first, NOT changed
        self.assertTrue(r["first"])
        self.assertFalse(r["changed"])
        r = hs.note_stripped([HEADER_B], k1)  # same head, new value
        self.assertTrue(r["changed"])
        self.assertEqual(r["prev"], HEADER_A)

    def test_adapter_notes_once_per_request(self):
        with envs.FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(True):
            s = AnthropicServing(_FakeChat("{{ messages }}"))
        req = AnthropicMessagesRequest.model_validate(_payload(HEADER_A))
        with self.assertLogs(hs.logger, level="INFO") as cm:
            for _ in range(4):  # front count, D probe, PLE hint, group
                s._convert_to_chat_completion_request(req)
        lines = [m for m in cm.output if hs.TAG_STRIP in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("first=1 changed=0", lines[0])
        self.assertEqual(len(hs._STRIP_SEEN), 1)


class TestPromptDiverge(unittest.TestCase):
    """Auflage 2/3: the instrument on the rendered ids, behind every chat path."""

    def _w(self):
        return pd.PromptWatch(window=8, rate=pd.RateLimit(per_s=1e9, burst=1e9))

    def test_the_96_161_shape_is_named_with_token_and_page(self):
        w = self._w()
        base = list(range(1000, 1000 + 89434))
        self.assertIsNone(w.observe(ids=base, rid="pdflip-94-158"))  # new
        moved = list(base)
        moved[8400] = 7  # one token in page 131, same length
        rec = w.observe(ids=moved, rid="pdflip-96-161", decode=lambda s: ",".join(map(str, s)))
        self.assertIsNotNone(rec)
        self.assertEqual((rec["unit"], rec["at"], rec["page"]), ("token", 8400, 131))
        self.assertEqual(rec["prev_len"], 89434)
        self.assertIn("9400", rec["prev"])  # base[8400] = 9400
        self.assertIn(",7,", rec["new"])

    def test_silent_classes(self):
        w = self._w()
        a = list(range(5000))
        w.observe(ids=a)
        self.assertIsNone(w.observe(ids=a))  # repeat
        self.assertIsNone(w.observe(ids=a + [1, 2, 3]))  # extends (next turn)
        self.assertIsNone(w.observe(ids=a[:-100] + [9] * 100))  # tail (regenerate)
        self.assertIsNone(w.observe(ids=list(range(90000, 95000))))  # new head
        self.assertEqual(w.counts["moved"], 0)
        self.assertEqual(
            (w.counts["repeat"], w.counts["extends"], w.counts["tail"], w.counts["new"]),
            (1, 1, 1, 2),
        )

    def test_rate_limit_counts_what_it_holds_back(self):
        r = pd.RateLimit(per_s=1.0, burst=2)
        self.assertEqual(r.take(now=0.0), 0)
        self.assertEqual(r.take(now=0.0), 0)
        self.assertIsNone(r.take(now=0.1))
        self.assertIsNone(r.take(now=0.2))
        self.assertEqual(r.take(now=1.5), 2)  # the line after names the two held back

    def test_text_fallback(self):
        w = self._w()
        t = "x" * 9000
        w.observe("S" + t)
        rec = w.observe("T" + t)
        self.assertIsNone(rec)  # shares < MIN_SHARED: new
        w.observe("H" * 300 + "a" + t)
        rec = w.observe("H" * 300 + "b" + t)
        self.assertEqual((rec["unit"], rec["at"]), ("byte", 300))

    def test_hook_sits_in_process_messages(self):
        """The real _process_messages hands its result to the watch, on the
        input_ids path (no template needed) -- both chat paths pass here."""
        from flliper.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from flliper.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        fake = SimpleNamespace(
            _apply_default_chat_template_kwargs=lambda r: None,
            is_gpt_oss=False, is_gemma4=False,
            _patch_reasoning_skip_special_tokens=lambda r: None,
            _get_reasoning_from_request=lambda r: False,
            tokenizer_manager=SimpleNamespace(
                server_args=SimpleNamespace(reasoning_parser=None),
                tokenizer=SimpleNamespace(decode=lambda s: "|".join(map(str, s))),
            ),
            tool_call_parser=None,
        )
        req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}],
                                    input_ids=list(range(3000)))
        with mock.patch.object(pd, "observe", wraps=pd.observe) as spy:
            OpenAIServingChat._process_messages(fake, req, False)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(list(spy.call_args.kwargs["ids"]), list(range(3000)))

    @unittest.skipUnless(os.path.isdir(NF_MODEL), "NF tokenizer/template not on this host")
    def test_reordered_tools_are_named_inside_the_tools_block(self):
        _Real.load()
        with envs.FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(True):
            s = AnthropicServing(_FakeChat(_Real.template))
        p2 = dict(_payload(HEADER_A), tools=TOOLS[:2] + [TOOLS[3], TOOLS[2]] + TOOLS[4:])
        t1, a = _ids(s, _payload(HEADER_A))
        t2, b = _ids(s, p2)
        w = self._w()
        w.observe(ids=a)
        rec = w.observe(ids=b, decode=lambda x: _Real.tok.decode(x))
        self.assertIsNotNone(rec)
        sys_at = len(_Real.tok.encode(t1[: t1.index("</IMPORTANT>")], add_special_tokens=False).ids)
        self.assertLess(rec["at"], sys_at)
        self.assertIn("tool_", rec["prev"] + rec["new"])


if __name__ == "__main__":
    unittest.main()
