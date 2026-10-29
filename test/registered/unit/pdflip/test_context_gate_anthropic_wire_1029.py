"""The front's CONTEXT-GATE answers an over-context ``/v1/messages`` request in the Anthropic wire.

Bug (found by the API-robustness review, 29.10.): ``Front._context_gate`` always answered with the OpenAI-shaped body
``{"error": {"type": "BadRequestError", ...}}``, also for ``/v1/messages``. Claude Code shrinks (compacts) its history only on the
wording ``prompt is too long: N tokens > M maximum`` in the Anthropic error object, so a client behind the front got a plain 400
and no way out. The gate now knows the request path; every other path keeps the OpenAI body byte for byte.
"""
import collections
import json
import re
import types

from flliper.srt.pdflip import front as F

CTX = 131072


def _front():
    f = object.__new__(F.Front)           # same setup as test_context_gate_kv_cap_1023: the real Front, no network, no groups
    f.counters = collections.Counter()
    f.dual_layout = False
    f._store_probe_info = {"context_length": CTX}
    f._dec_gate = lambda *a, **k: None
    return f


def _xx(n):
    return types.SimpleNamespace(n=n, mm=False, unpriced=False, n_src="front_ids")


def _gate(path):
    f = _front()
    return F.Front._context_gate(f, "pdflip-1-1", "x", _xx(CTX + 288), CTX + 288, path=path)


def test_messages_path_gets_the_anthropic_error_claude_code_compacts_on():
    resp = _gate("/v1/messages")
    assert resp.status == 400
    body = json.loads(resp.body)
    assert body["type"] == "error" and body["error"]["type"] == "invalid_request_error"
    m = re.search(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", body["error"]["message"])   # Claude Code's own regex
    assert m is not None and (int(m.group(1)), int(m.group(2))) == (CTX + 288, CTX)


def test_other_paths_keep_the_openai_body_unchanged():
    for path in ("/v1/chat/completions", ""):
        body = json.loads(_gate(path).body)
        assert body["error"]["type"] == "BadRequestError" and body["error"]["code"] == 400
        assert f"The input ({CTX + 288} tokens) is longer than the model's context length ({CTX} tokens)." == body["error"]["message"]


def test_a_request_that_fits_passes_on_both_paths():
    for path in ("/v1/messages", "/v1/chat/completions"):
        f = _front()
        assert F.Front._context_gate(f, "r", "x", _xx(CTX), CTX, path=path) is None
