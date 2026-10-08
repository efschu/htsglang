"""DP-NACHLAUF 02.10.: leg 1 goes to P with the front's exact input_ids.

N5d (0c996cf05c 1002_124821, D->P epoch 25, pdflip-24-93, 132809 tokens): the
front dispatched leg 1 at the flip's done (13:00:43.565); PP0's scheduler saw
the request ~0.8 s later (P-CHUNK-POLICY at :44, the prefetch op's start
45.98 - 1.613 s) -- P's HTTP server rendered the chat template and tokenized
132k tokens again. The front had counted the same prompt exactly
(X-EXACT-PRICE; X-EXACT-TOKENS tokens_front=132814 tokens_group=132814 match=1).

Pinned (red before): the body for P's /generate (ids, rid, one token, no
stream); no ids / an unknown path / media -> the client's path; the switch;
leg 1 uses it and prints PDFLIP LEG1-INPUT-IDS.
"""
from __future__ import annotations

import inspect
import os

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402

CHAT = {"model": "m", "rid": "pdflip-24-93", "max_tokens": 1,
        "messages": [{"role": "user", "content": "hello"}]}


def test_switch_default_on():
    assert F.leg1_input_ids_on({})
    for off in ("0", "false", "no", "off"):
        assert not F.leg1_input_ids_on({F.LEG1_INPUT_IDS_ENV: off})


def test_chat_and_messages_become_generate_with_the_ids():
    ids = np.asarray([5, 6, 7], dtype=np.int32)
    for path in ("/v1/chat/completions", "/v1/messages", "/generate"):
        b = F.leg1_input_ids_payload(path, dict(CHAT), ids)
        assert b == {"input_ids": [5, 6, 7], "stream": False, "rid": "pdflip-24-93",
                     "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}, (path, b)


def test_no_ids_unknown_path_or_media_keep_the_client_path():
    ids = [1, 2]
    assert F.leg1_input_ids_payload("/v1/chat/completions", dict(CHAT), None) is None
    assert F.leg1_input_ids_payload("/v1/chat/completions", dict(CHAT), []) is None
    assert F.leg1_input_ids_payload("/v1/completions", dict(CHAT), ids) is None
    img = dict(CHAT, messages=[{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}])
    assert F.leg1_input_ids_payload("/v1/chat/completions", img, ids) is None
    anth = dict(CHAT, messages=[{"role": "user", "content": [
        {"type": "tool_result", "content": [{"type": "image", "source": {"type": "base64"}}]}]}])
    assert F.leg1_input_ids_payload("/v1/messages", anth, ids) is None
    assert F.leg1_input_ids_payload("/generate", {"rid": "r", "image_data": ["x"]}, ids) is None
    txt = dict(CHAT, messages=[{"role": "user", "content": [{"type": "text", "text": "a"}]}])
    assert F.leg1_input_ids_payload("/v1/chat/completions", txt, ids) is not None


def test_leg1_posts_the_ids_body_and_says_so():
    src = inspect.getsource(F.Front.leg1)
    assert "leg1_input_ids_payload(p.path, payload, _ids)" in src
    assert 'f"{g.url}{_post_path}", json=_post_body' in src
    assert "PDFLIP LEG1-INPUT-IDS rid=%s tokens=%d from=%s" in src
