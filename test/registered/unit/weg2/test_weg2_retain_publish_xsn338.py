"""xsn338: P publishes a finished request's nodes at its finish, not in a bubble."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import retain_publish as rp  # noqa: E402


def test_gate_is_group_p_and_env():
    assert rp.publish_at_retain_on({"SGLANG_WEG2_GROUP": "P"})
    assert not rp.publish_at_retain_on({"SGLANG_WEG2_GROUP": "D"})
    assert not rp.publish_at_retain_on({})
    assert not rp.publish_at_retain_on({"SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_PUBLISH_AT_RETAIN": "0"})
    assert rp.max_issue({}) == 64 and rp.max_issue({"SGLANG_WEG2_PUBLISH_AT_RETAIN_MAX": "8"}) == 8
    assert rp.max_issue({"SGLANG_WEG2_PUBLISH_AT_RETAIN_MAX": "x"}) == 64


def test_retain_site_calls_the_publish_after_the_handoff():
    from sglang.srt.mem_cache import unified_radix_cache as u
    src = open(u.__file__).read()
    i = src.index("self._weg2_handoff_write(req, radix_key)\n")
    assert "self._weg2_publish_at_retain(req)" in src[i:i + 200]
    j = src.index("def _weg2_publish_at_retain")
    assert "publish_unbacked_sweep(max_issue=_rp.max_issue())" in src[j:j + 1200]
