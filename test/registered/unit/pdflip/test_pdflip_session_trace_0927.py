# SPDX-License-Identifier: Apache-2.0
"""SESSION-TRACE (HS 27.09., P-HiCache-read order): the front logs a rid's agent session as a 10-hex hash
(X-Claude-Code-Session-Id, which Claude Code sends and the router on 30099 forwards; fallback the session part
of metadata.user_id) and, from the X-EXACT token ids, the common prefix with the same session's previous
prompt -- so "does P's hit end before pages the follow-up prompt contains?" is read off the log, not
guessed (NF rc12u: 4-7 cached 39232 = the prompt of 4-6; 8-9 cached 44800 = the prompt of 6-8)."""

import collections
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import session_trace as ST  # noqa: E402

SID = "3f1c2a9e-7b4d-4e21-9c0a-5d6e7f8a9b0c"


def test_header_first_then_metadata_then_none():
    assert ST.session_raw({ST.HEADER: SID}, {}) == (SID, "header")
    assert ST.session_raw({ST.HEADER.lower(): SID}, {}) == (SID, "header")
    uid = f"user_abc_account_def_session_{SID}"
    assert ST.session_raw({}, {"metadata": {"user_id": uid}}) == (SID, "metadata")
    assert ST.session_raw({}, {"metadata": {"user_id": "user_abc"}}) == ("", "none")
    assert ST.session_raw(None, None) == ("", "none")


def test_short_is_a_hash_never_the_id():
    s = ST.short(SID)
    assert len(s) == 10 and SID[:8] not in s and s == ST.short(SID)
    assert ST.short("") == ""


def test_common_prefix():
    assert ST.common_prefix([1, 2, 3, 4], [1, 2, 9]) == 2
    assert ST.common_prefix(np.arange(10), np.arange(12)) == 10
    assert ST.common_prefix([], [1]) == 0
    assert ST.common_prefix([5], [6]) == 0


def test_prefixes_per_session_bounded():
    sp = ST.SessionPrefixes(max_sessions=2)
    assert sp.note("a", "pdflip-4-6", np.arange(39394)) is None
    ids = np.concatenate([np.arange(39232), np.arange(5573) + 10**6])
    assert sp.note("a", "pdflip-4-7", ids) == ("pdflip-4-6", 39232, 39394)
    sp.note("b", "r1", [1])
    sp.note("c", "r2", [1])  # evicts "a"
    assert sp.note("a", "r3", [1]) is None


def _front():
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    return f


def test_front_lines(caplog):
    f = _front()
    req = types.SimpleNamespace(headers={ST.HEADER: SID})
    with caplog.at_level("INFO", logger=F.logger.name):
        f._sess_note("pdflip-4-6", req, {})
        f._sess_note("pdflip-4-7", req, {})
        f._sess_prefix("pdflip-4-6", np.arange(39394))
        f._sess_prefix("pdflip-4-7", np.concatenate([np.arange(39232), np.arange(9) + 10**6]))
    text = caplog.text
    h = ST.short(SID)
    assert f"PDFLIP SESSION rid=pdflip-4-6 sess={h} src=header" in text
    assert f"PDFLIP SESSION-PREFIX rid=pdflip-4-7 sess={h} prev_rid=pdflip-4-6 common=39232 prompt=39241" in text
    assert SID not in text
    assert f._sess_tag("pdflip-4-7") == f" sess={h}"
    assert f._sess_tag("pdflip-9-9") == ""


def test_no_session_logs_a_dash_and_no_prefix(caplog):
    f = _front()
    with caplog.at_level("INFO", logger=F.logger.name):
        f._sess_note("pdflip-1-1", types.SimpleNamespace(headers={}), {})
        f._sess_prefix("pdflip-1-1", [1, 2, 3])
    assert "PDFLIP SESSION rid=pdflip-1-1 sess=- src=none" in caplog.text
    assert "SESSION-PREFIX" not in caplog.text


def test_wiring():
    src = open(F.__file__).read()
    i = src.index('rid = f"pdflip-{self.epoch}-{self._rid}"')
    assert "self._sess_note(rid, request, payload)" in src[i:i + 300]
    i = src.index("ft.remember(text, c.ids)")
    assert "self._sess_prefix(rid, c.ids)" in src[i:i + 120]
    assert src.count("self._sess_tag(") == 3
    assert 'epoch=%d%s",\n                        p.rid, pt, ct, time.time() - t0, self.epoch, self._sess_tag(p.rid))' in src
