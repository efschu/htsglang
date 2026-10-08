# SPDX-License-Identifier: Apache-2.0
"""#59 front price: never credit past D's deepest resumable (Mamba) anchor.

The front priced d_leg2_cached with D's MEASURED cached_tokens, but a hybrid
model on D resumes only from its deepest held Mamba anchor:
  rc12z pdflip-12-61: pdflip-8-53 served ct=55424 (prompt 57396), RETAIN value=False
    at its finish; pdflip-12-61 (57792 tokens, common 57395) priced credit 55424 ->
    pending 2368 SHORT, D's extent 37952 -> X refusal midstream, 72 s P detour.
  rc12y pdflip-16-42: pdflip-13-41 served ct=46848 (prompt 46850), no recurrent state
    kept after RETAIN value=False; pdflip-16-42 (46851, common 46726) priced 125.
D now sends ``pdflip_resumable_depth`` (meta_info / sglext, NF's D side); the
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

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import TokenSpans  # noqa: E402


def _ids(n: int, common: int = None, salt: int = 7) -> np.ndarray:
    a = np.arange(n, dtype=np.int32)
    if common is not None and common < n:
        a[common:] = 10 ** 6 + salt + np.arange(n - common, dtype=np.int32)
    return a


# ---- the two metal cases, exact token pricing (X-EXACT, TokenSpans) ----------

def test_pdflip_12_61_credit_capped_at_the_anchor_routes_long():
    ts = TokenSpans(agent_span=True)
    prev = _ids(57396)                         # pdflip-8-53
    ts.record_presence(prev, 55424, prompt_tokens=57396, held_epoch=8,
                       resumable_depth=37952)
    cur = _ids(57792, common=57395)            # pdflip-12-61, same session
    pending, credit, known, _src = ts.pending(cur, epoch=12)
    assert known and credit == 37952
    assert pending == 57792 - 37952 == 19840, "D's own extent on the metal"
    assert pending > 5074, "over X: LONG via P, not SHORT into an X refusal"


def test_pdflip_12_61_without_the_field_is_the_old_price():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(57396), 55424, prompt_tokens=57396, held_epoch=8)
    pending, credit, _k, _s = ts.pending(_ids(57792, common=57395), epoch=12)
    assert (credit, pending) == (55424, 2368), "the rc12z log line, byte for byte"


def test_pdflip_16_42_no_recurrent_state_retracts():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(46850), 46848, prompt_tokens=46850, held_epoch=16,
                       resumable_depth=0)
    pending, credit, known, _s = ts.pending(_ids(46851, common=46726), epoch=16)
    assert (pending, credit, known) == (46851, 0, False)


def test_pdflip_16_42_without_the_field_is_priced_at_the_divergence_rule():
    # rc12y logged (credit 46726, pending 125) -- the over-credit this file's
    # docstring names. PX 28.09.: 16-42 leaves 13-41 at 46726, before both
    # the held prompt (46850) and the measured resume point (46848), so no
    # anchor of the witness lies on its path -- LONG via P even without the
    # #59 field.
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(46850), 46848, prompt_tokens=46850, held_epoch=16)
    pending, credit, _k, _s = ts.pending(_ids(46851, common=46726), epoch=16)
    assert (credit, pending) == (0, 46851)


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
    assert F.d_resumable_depth({"meta_info": {"pdflip_resumable_depth": 37952}}) == 37952
    assert F.d_resumable_depth({"sglext": {"pdflip_resumable_depth": "0"}}) == 0
    assert F.d_resumable_depth({"meta_info": {"prompt_tokens": 5}}) is None
    assert F.d_resumable_depth({"sglext": {"pdflip_resumable_depth": "x"}}) is None
    assert F.d_resumable_depth(None) is None


def test_depth_from_the_stream_tail_finish_chunk():
    chunks = [
        {"choices": [{"delta": {"content": "hi"}}], "sglext": {"pdflip_resumable_depth": 100}},
        {"choices": [], "usage": {"prompt_tokens": 57792}, "sglext": {"pdflip_resumable_depth": 37952}},
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
        f._note_resumable_depth("pdflip-8-53", 57396, 55424, False, 37952)
        f._note_resumable_depth("pdflip-9-1", 1000, 200, True, 640)
        f._note_resumable_depth("pdflip-9-2", 1000, 200, False, 900)
        f._note_resumable_depth("pdflip-9-3", 1000, 200, False, None)
    assert "PDFLIP PRESENCE-DEPTH-CAP rid=pdflip-8-53 credit 55424 -> 37952" in caplog.text
    assert "PDFLIP PRESENCE-DEPTH-CAP rid=pdflip-9-1 credit 1000 -> 640" in caplog.text
    assert "pdflip-9-2" not in caplog.text and "pdflip-9-3" not in caplog.text
    assert f.counters["presence_depth_capped"] == 2
    assert f.counters["presence_depth_uncapped"] == 1
    assert f.counters["presence_depth_absent"] == 1


def test_both_leg2_finish_sites_pass_the_depth():
    src = inspect.getsource(F.Front.leg2)
    assert src.count("resumable_depth=_depth") == 4, "spans + tspans, stream + body"
    assert "_depth = d_resumable_depth_stream_tail(bytes(tail))" in src
    assert "_depth = d_resumable_depth(js)" in src


# ---- #59 A (operator 28.09.): a NEW text in flight never credits past the depth known
# for its prefix before the leg; none known -> 0 until the finish sets D's real depth.
# Both prices: exact tokens (TokenSpans) and chars/3 (SpanLRU).

def _spanlru():
    return F.SpanLRU(agent_span=True) if "agent_span" in inspect.signature(F.SpanLRU).parameters \
        else F.SpanLRU()


def _tkey(ids):
    return TokenSpans._key(ids)


def _skey(text):
    import hashlib

    return hashlib.sha1(text.encode()).hexdigest()


def test_a_tokens_i_inflight_without_a_known_depth_credits_nothing():
    ts = TokenSpans(agent_span=True)
    cur = _ids(1000)
    ts.record_inflight(cur, 5)
    pending, credit, known, _src = ts.pending(cur, epoch=5)
    assert (pending, credit, known) == (1000, 0, False), "no depth known -> 0 until the finish"


def test_a_tokens_ii_inflight_is_capped_at_the_known_prefix_depth():
    ts = TokenSpans(agent_span=True)
    prev = _ids(600)                                   # the earlier turn, D's depth 384
    ts.record_presence(prev, 580, prompt_tokens=600, resumable_depth=384)
    cur = _ids(1000)                                   # the running turn on the same prefix
    ts.record_inflight(cur, 5)
    assert ts.depth_caps[_tkey(cur)] == 384
    pending, credit, known, _src = ts.pending(_ids(1100), epoch=5)   # the twin
    assert known and credit == 384 and pending == 1100 - 384, (pending, credit)


def test_a_tokens_iii_the_finish_sets_the_new_depth():
    ts = TokenSpans(agent_span=True)
    cur = _ids(1000)
    ts.record_inflight(cur, 5)
    assert ts.pending(cur, epoch=5)[1] == 0
    ts.record_presence(cur, 990, prompt_tokens=1000, held_epoch=5, resumable_depth=960)
    assert ts.pending(cur, epoch=5)[1] == 960
    assert ts.depth_caps[_tkey(cur)] == 960


def test_a_tokens_iv_an_old_entry_keeps_its_cap_in_flight():
    ts = TokenSpans(agent_span=True)
    old = _ids(1000)
    ts.record_presence(old, 900, prompt_tokens=1000, resumable_depth=512)
    ts.record_presence(_ids(2000), 1900, prompt_tokens=2000, resumable_depth=1800)  # a longer neighbour
    ts.record_inflight(old, 5)
    assert ts.depth_caps[_tkey(old)] == 512, "neither 0 nor the neighbour's depth"
    ts.record_inflight(_ids(1500), 5)                  # a new text beside them: inherits, bounded
    # PX 28.09.: the neighbour's anchor 1800 is past the 1500 the texts share
    # (was 1500 = min(1800, 1500): no anchor sits there); the deepest anchor
    # on the path is the old entry's 512
    assert ts.depth_caps[_tkey(_ids(1500))] == 512, "the deepest capped depth on the shared path"


def test_a_chars_i_inflight_without_a_known_depth_credits_nothing():
    s = _spanlru()
    cur = "a" * 3000
    s.record_inflight(cur, 1000, held_epoch=5)
    assert s.uncached_tokens(cur, 1000, epoch=5) == (1000, False)


def test_a_chars_ii_inflight_is_capped_at_the_known_prefix_depth():
    s = _spanlru()
    prev = "a" * 1800
    s.record_presence(prev, 580, prompt_tokens=600, resumable_depth=384)
    cur = prev + "b" * 1200
    s.record_inflight(cur, 1000, held_epoch=5)
    assert s.depth_caps[_skey(cur)] == 384
    rem, known = s.uncached_tokens(cur + "c" * 300, 1101, epoch=5)   # the twin
    assert known and rem == 100 + (1000 - 384), rem   # uncapped (the old in-flight credit) = 100


def test_a_chars_iii_the_finish_sets_the_new_depth():
    s = _spanlru()
    cur = "a" * 3000
    s.record_inflight(cur, 1000, held_epoch=5)
    assert s.uncached_tokens(cur, 1000, epoch=5) == (1000, False)
    s.record_presence(cur, 990, prompt_tokens=1000, held_epoch=5, resumable_depth=960)
    assert s.uncached_tokens(cur, 1000, epoch=5) == (40, True)
    assert s.depth_caps[_skey(cur)] == 960


def test_a_chars_iv_an_old_entry_keeps_its_cap_in_flight():
    s = _spanlru()
    old = "a" * 3000
    s.record_presence(old, 900, prompt_tokens=1000, resumable_depth=512)
    s.record_presence("a" * 6000, 1900, prompt_tokens=2000, resumable_depth=1800)
    s.record_inflight(old, 1000, held_epoch=5)
    assert s.depth_caps[_skey(old)] == 512, "neither 0 nor the neighbour's depth"
    s.record_inflight("a" * 4500, 1500, held_epoch=5)  # a new text beside them: inherits, bounded
    assert s.depth_caps[_skey("a" * 4500)] == 1500, "the longest capped prefix (1800), bounded by 4500/3"
