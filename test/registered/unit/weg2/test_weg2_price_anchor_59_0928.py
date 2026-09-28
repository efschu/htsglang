# SPDX-License-Identifier: Apache-2.0
"""#59 front price: never credit past D's deepest resumable (Mamba) anchor.

The front priced d_leg2_cached with D's MEASURED cached_tokens, but a hybrid
model on D resumes only from its deepest held Mamba anchor:
  rc12z weg2-12-61: weg2-8-53 served ct=55424 (prompt 57396), RETAIN value=False
    at its finish; weg2-12-61 (57792 tokens, common 57395) priced credit 55424 ->
    pending 2368 SHORT, D's extent 37952 -> X refusal midstream, 72 s P detour.
  rc12y weg2-16-42: weg2-13-41 served ct=46848 (prompt 46850), no recurrent state
    kept after RETAIN value=False; weg2-16-42 (46851, common 46726) priced 125.
D now sends ``weg2_resumable_depth`` (meta_info / sglext, NF's D side); the
front credits min(credit, depth). Absent field = the old credit.
"""

from __future__ import annotations

import collections
import inspect
import json
import logging
import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import TokenSpans  # noqa: E402


def _ids(n: int, common: int = None, salt: int = 7) -> np.ndarray:
    a = np.arange(n, dtype=np.int32)
    if common is not None and common < n:
        a[common:] = 10 ** 6 + salt + np.arange(n - common, dtype=np.int32)
    return a


# ---- the two metal cases, exact token pricing (X-EXACT, TokenSpans) ----------

def test_weg2_12_61_credit_capped_at_the_anchor_routes_long():
    ts = TokenSpans(agent_span=True)
    prev = _ids(57396)                         # weg2-8-53
    ts.record_presence(prev, 55424, prompt_tokens=57396, held_epoch=8,
                       resumable_depth=37952)
    cur = _ids(57792, common=57395)            # weg2-12-61, same session
    pending, credit, known, _src = ts.pending(cur, epoch=12)
    assert known and credit == 37952
    assert pending == 57792 - 37952 == 19840, "D's own extent on the metal"
    assert pending > 5074, "over X: LONG via P, not SHORT into an X refusal"


def test_weg2_12_61_without_the_field_is_the_old_price():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(57396), 55424, prompt_tokens=57396, held_epoch=8)
    pending, credit, _k, _s = ts.pending(_ids(57792, common=57395), epoch=12)
    assert (credit, pending) == (55424, 2368), "the rc12z log line, byte for byte"


def test_weg2_16_42_no_recurrent_state_retracts():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(46850), 46848, prompt_tokens=46850, held_epoch=16,
                       resumable_depth=0)
    pending, credit, known, _s = ts.pending(_ids(46851, common=46726), epoch=16)
    assert (pending, credit, known) == (46851, 0, False)


def test_weg2_16_42_without_the_field_is_the_old_price():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(46850), 46848, prompt_tokens=46850, held_epoch=16)
    pending, credit, _k, _s = ts.pending(_ids(46851, common=46726), epoch=16)
    assert (credit, pending) == (46726, 125), "the rc12y log line"


def test_the_held_epoch_credit_is_capped_too():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(1000), 200, prompt_tokens=1000, held_epoch=3,
                       resumable_depth=640)
    assert ts.pending(_ids(1000), epoch=3)[1] == 640, "held prompt 1000 capped"
    assert ts.pending(_ids(1000), epoch=4)[1] == 200, "outside the epoch: ct, below the cap"


def test_inflight_keeps_the_cap_until_the_finish_replaces_it():
    ts = TokenSpans(agent_span=True)
    ids = _ids(1000)
    ts.record_presence(ids, 900, prompt_tokens=1000, resumable_depth=512)
    ts.record_inflight(ids, 5)
    assert ts.pending(ids, epoch=5)[1] == 512
    ts.record_presence(ids, 990, prompt_tokens=1000, held_epoch=5, resumable_depth=960)
    assert ts.pending(ids, epoch=5)[1] == 960


def test_caps_are_dropped_with_their_entries():
    ts = TokenSpans(agent_span=True, cap=2)
    for k in range(4):
        ts.record_presence(_ids(100 + k, salt=k), 50, resumable_depth=10)
    assert set(ts.depth_caps) == set(ts.entries)


# ---- the chars/3 witness (SpanLRU) carries the same cap ----------------------

def test_spanlru_cap_and_retraction():
    s = F.SpanLRU(agent_span=True) if "agent_span" in inspect.signature(F.SpanLRU).parameters \
        else F.SpanLRU()
    text = "a" * 3000
    s.record_presence(text, 900, prompt_tokens=1000, resumable_depth=400)
    rem, known = s.uncached_tokens(text, 1000)
    assert known and rem == 600
    s.record_presence(text, 900, prompt_tokens=1000)          # field absent: old credit
    assert s.uncached_tokens(text, 1000)[0] == 100
    s.record_presence(text, 900, prompt_tokens=1000, resumable_depth=0)
    assert s.uncached_tokens(text, 1000) == (1000, False)
    assert not s.depth_caps


# ---- D's field on both wires -------------------------------------------------

def test_depth_parsed_from_meta_info_and_sglext():
    assert F.d_resumable_depth({"meta_info": {"weg2_resumable_depth": 37952}}) == 37952
    assert F.d_resumable_depth({"sglext": {"weg2_resumable_depth": "0"}}) == 0
    assert F.d_resumable_depth({"meta_info": {"prompt_tokens": 5}}) is None
    assert F.d_resumable_depth({"sglext": {"weg2_resumable_depth": "x"}}) is None
    assert F.d_resumable_depth(None) is None


def test_depth_from_the_stream_tail_finish_chunk():
    chunks = [
        {"choices": [{"delta": {"content": "hi"}}], "sglext": {"weg2_resumable_depth": 100}},
        {"choices": [], "usage": {"prompt_tokens": 57792}, "sglext": {"weg2_resumable_depth": 37952}},
    ]
    tail = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    assert F.d_resumable_depth_stream_tail(tail) == 37952
    assert F.d_resumable_depth_stream_tail(b"data: {\"usage\": {}}\n\ndata: [DONE]\n") is None


# ---- the log line and the wiring ---------------------------------------------

def _front():
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    return f


def test_cap_line_names_old_new_and_rid(caplog):
    f = _front()
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._note_resumable_depth("weg2-8-53", 57396, 55424, False, 37952)
        f._note_resumable_depth("weg2-9-1", 1000, 200, True, 640)
        f._note_resumable_depth("weg2-9-2", 1000, 200, False, 900)
        f._note_resumable_depth("weg2-9-3", 1000, 200, False, None)
    assert "WEG2 PRESENCE-DEPTH-CAP rid=weg2-8-53 credit 55424 -> 37952" in caplog.text
    assert "WEG2 PRESENCE-DEPTH-CAP rid=weg2-9-1 credit 1000 -> 640" in caplog.text
    assert "weg2-9-2" not in caplog.text and "weg2-9-3" not in caplog.text
    assert f.counters["presence_depth_capped"] == 2
    assert f.counters["presence_depth_uncapped"] == 1
    assert f.counters["presence_depth_absent"] == 1


def test_both_leg2_finish_sites_pass_the_depth():
    src = inspect.getsource(F.Front.leg2)
    assert src.count("resumable_depth=_depth") == 4, "spans + tspans, stream + body"
    assert "_depth = d_resumable_depth_stream_tail(bytes(tail))" in src
    assert "_depth = d_resumable_depth(js)" in src
