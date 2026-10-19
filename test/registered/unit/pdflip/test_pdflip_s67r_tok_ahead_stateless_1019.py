"""S67r review A1 (reviewer test /tmp/rv67/test_rv67_tok_ahead_stateless.py, taken over verbatim in its body):
``/pdflip/tokenize_ahead`` (S63 F2 TOK-AHEAD, branches desk/flliper-nf-s63r-1019 / s66) tokenizes a ``~probe``-rid
payload with ``TokenizerManager._tokenize_one_request`` -- the same KeyError class as the rc16 D-MATCH-PROBE unless
it passes ``stateless=True`` (review S67, merge probe S67+S66: answer without it verbatim
``{'success': False, 'message': "KeyError: 'pdflip-7-1~probe'"}``).

SKIPPED while the endpoint is absent (this branch, rc16 base); it GRIPS at the rc17 merge with S63r/S66 and must be
green there -- the integration seat sets ``stateless=True`` in ``pdflip_tokenize_ahead``."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import test_pdflip_s67_dprobe_stateless_1019 as s67  # noqa: E402

from flliper.srt.entrypoints import http_server as hs  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="stage-a-pdflip-unit")

_tah = pytest.importorskip("flliper.srt.pdflip.tok_ahead",
                           reason="TOK-AHEAD (S63/S63r/S66) not in this tree -- the test grips at the merge")
if not hasattr(hs, "pdflip_tokenize_ahead"):
    pytest.skip("/pdflip/tokenize_ahead not in this tree -- the test grips at the merge", allow_module_level=True)


def test_tokenize_ahead_no_keyerror(monkeypatch):
    tm = s67._tm()
    tm.pdflip_tok_ahead = _tah.TokAhead(keep=8, ttl_s=60.0, max_bytes=1 << 30)
    monkeypatch.setattr(_tah, "gpu_side_reason", lambda: "")
    monkeypatch.setattr(hs, "_global_state", SimpleNamespace(tokenizer_manager=tm))
    body = {"path": "/generate", "payload": {"rid": "pdflip-7-1", "input_ids": [1, 2, 3, 4, 5],
                                             "image_data": ["data:image/png;base64,AAAA"]}}
    resp = asyncio.run(hs.pdflip_tokenize_ahead(s67._RawRequest(body)))
    out = json.loads(resp.body)
    assert tm.rid_to_state == {}
    assert out.get("success") is True, out
    assert tm.pdflip_tok_ahead.has("pdflip-7-1")
