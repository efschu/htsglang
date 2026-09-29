# SPDX-License-Identifier: Apache-2.0
"""PREFILL-EINBRUCH-0929: follow-up turns of a few hundred new tokens on a
prefix the store already holds are priced LONG and prefilled on P.

NF z30u (4f23714e2d), front log ..._0929_065429.front.log: 77 turns routed
LONG prefilled < 2k new tokens on P (median 402 on a 49 792 hit); 69 of the
P hits sat exactly on an END-ANCHOR depth written before the verdict.

K2 -- a D leg 2 that was PARKED and resumed answers with the retained prefix
(prompt + tokens decoded before the park) as prompt_tokens == cached_tokens:
  weg2-2-7: 'X-EXACT-ERR ... tokens_front=21601 tokens_d=21760 match=0',
  D: 'WEG2-D-PARK RETAINED rid=weg2-2-7 retained=21761 of 21987'.
  All 73 token mismatches of that boot are D-park resumes. The entry then
  credits 0 to every follow-up (PX): weg2-6-13 (22851 tok, shares >= 21568
  with weg2-2-7) 'X-EXACT-PRICE ... credit=0 src=none' -> LONG, P prefilled
  1283 tokens on a 21568 store hit ('#988 LOADBACK ... 21568').

K1 -- the witness's leg 2 has not finished when the follow-up is priced: an
in-flight entry credits 0 (#59 A) and P's leg 1 feeds nothing (#1324).
  weg2-8-21: 'X-EXACT-PRICE pending=4624 credit=22848' -> LONG, P prefilled
  976 on a 26496 store hit. Behind SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE: at
  the first content of an after_p leg 2 (the publish is read), the prompt's
  END-ANCHOR is recorded as a store presence.
"""

from __future__ import annotations

import collections
import logging
import os
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import TokenSpans  # noqa: E402

X = 4096


def _ids(n, common=None, salt=7):
    a = np.arange(n, dtype=np.int32)
    if common is not None and common < n:
        a[common:] = 10 ** 6 + salt * 10 ** 5 + np.arange(n - common, dtype=np.int32)
    return a


class _Tok:
    def __init__(self):
        self.m = {}

    def ids_for(self, text):
        return self.m.get(text)


def _front(agent_span=True):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 7
    f.tspans = TokenSpans(agent_span=agent_span)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    f._x_exact_reprice_queue = lambda why: 0
    return f


def _follow_up():
    # weg2-6-13: 22851 tokens, leaves weg2-2-7's text at 21580 (past its
    # end anchor 21568, before its end 21601)
    cur = _ids(22851, common=21580, salt=1)
    cur[21580:] = 10 ** 6 + 9 * 10 ** 5 + np.arange(22851 - 21580, dtype=np.int32)
    return cur


# ---- K2: a reading never credits past its own text ---------------------------

def test_k2_park_resume_entry_credits_its_own_end_anchor(caplog):
    f = _front()
    f.ftok.m["prev"] = _ids(21601, salt=1)
    f._x_exact_rid["weg2-2-7"] = (21601, 21601, "none")
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        # the metal answer of the resumed leg 2
        f._x_exact_record("weg2-2-7", "prev", 21760, 21760, types.SimpleNamespace(d_direct=False),
                          None, resumable_depth=21760)
    pending, credit, known, _src = f.tspans.pending(_follow_up(), epoch=None)
    assert known
    assert credit == 21568, f"credit {credit}: the prompt-end anchor 21568 lies on the shared path"
    assert pending == 22851 - 21568 <= X, "SHORT: D reads 21568 back and prefills 1283"
    assert "WEG2 PRESENCE-OWN-TEXT-CLAMP rid=weg2-2-7 tokens_front=21601" in caplog.text
    assert f.counters["presence_own_text_clamped"] == 1
    # 27B's condition: count + clamped tokens per boot as a number, not a log
    assert f.counters["presence_own_text_clamped_tokens"] == 21760 - 21568


def test_presence_price_numbers_in_the_state():
    import inspect

    f = _front()
    f.ftok.m["prev"] = _ids(21601, salt=1)
    f.ftok.m["other"] = _ids(30000, salt=4)
    f._x_exact_record("a", "prev", 21760, 21760, types.SimpleNamespace(d_direct=False), None,
                      resumable_depth=21760)
    f._x_exact_record("b", "prev", 21824, 21824, types.SimpleNamespace(d_direct=False), None,
                      resumable_depth=21824)
    f._p_anchor_presence("c", "other", types.SimpleNamespace(leg1_prompt_tokens=30000))
    src = inspect.getsource(F.Front.state_dict)
    assert '"presence_price": {' in src
    # the block reads the same counters the state endpoint serves
    got = {
        "own_text_clamped": f.counters["presence_own_text_clamped"],
        "own_text_clamped_tokens": f.counters["presence_own_text_clamped_tokens"],
        "p_anchor_presence": f.counters["p_anchor_presence"],
        "p_anchor_presence_tokens": f.counters["p_anchor_presence_tokens"],
    }
    assert got == {"own_text_clamped": 2, "own_text_clamped_tokens": 192 + 256,
                   "p_anchor_presence": 1, "p_anchor_presence_tokens": 29952}


def test_k2_control_unparked_entry_is_unchanged(caplog):
    f = _front()
    f.ftok.m["prev"] = _ids(21601, salt=1)
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._x_exact_record("weg2-x", "prev", 21601, 21568, types.SimpleNamespace(d_direct=False),
                          None, resumable_depth=21568)
    _p, credit, _k, _s = f.tspans.pending(_follow_up(), epoch=None)
    assert credit == 21568
    assert "PRESENCE-OWN-TEXT-CLAMP" not in caplog.text


def test_k2_px_still_refuses_a_divergence_before_the_anchor():
    # PX (fb2986f740) unchanged: a text that leaves the entry before its end
    # anchor gets nothing from it, clamped or not
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(21601, salt=1), 21760, prompt_tokens=21760, resumable_depth=21760)
    _p, credit, _k, _s = ts.pending(_ids(22851, common=20470, salt=1))
    assert credit == 0


def test_k2_clamp_keeps_a_real_depth_inside_the_text():
    ts = TokenSpans(agent_span=True)
    # D resumed at 16384 inside the text but reported a prompt past it
    ts.record_presence(_ids(20000), 16384, prompt_tokens=20200, held_epoch=4,
                       resumable_depth=16384)
    _p, credit, _k, _s = ts.pending(_ids(21000), epoch=4)
    assert credit == 16384, "a measured depth within the text is not raised by the clamp"


# ---- K1: P's END-ANCHOR, once D has read it ----------------------------------

def test_k1_p_anchor_presence_credits_the_end_anchor_after_first_content(caplog):
    f = _front()
    prev = _ids(27472, salt=3)
    f.ftok.m["prev"] = prev
    # the leg 2 of the witness is in flight: #59 A gives it 0
    f.tspans.record_inflight(prev, 7)
    cur = _ids(28448, common=26500, salt=3)
    assert f.tspans.pending(cur, epoch=7)[1] == 0
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        anchor = f._p_anchor_presence("weg2-8-20", "prev",
                                      types.SimpleNamespace(leg1_prompt_tokens=27472))
    assert anchor == 27456
    # the shared path ends at 26500: the anchor 27456 lies past it -> PX gives 0
    assert f.tspans.pending(cur, epoch=7)[1] == 0
    # a follow-up that contains the whole prompt gets the anchor, not the
    # held whole prompt (never past the anchor D resumed from)
    full = _ids(28448, salt=3)
    full[:27472] = prev
    pending, credit, _k, _s = f.tspans.pending(full, epoch=7)
    assert credit == 27456 and pending == 28448 - 27456 <= X
    assert "WEG2 P-ANCHOR-PRESENCE rid=weg2-8-20 anchor=27456 p_prompt=27472" in caplog.text


def test_k1_p_anchor_presence_outlives_the_epoch():
    f = _front()
    prev = _ids(30000, salt=4)
    f.ftok.m["prev"] = prev
    f._p_anchor_presence("r", "prev", types.SimpleNamespace(leg1_prompt_tokens=30000))
    full = np.concatenate([prev, np.arange(10 ** 6, 10 ** 6 + 500, dtype=np.int32)])
    # a flip later (no epoch: D asleep or a new epoch) the store anchor stands
    _p, credit, _k, _s = f.tspans.pending(full, epoch=None)
    assert credit == 29952


def test_k1_finish_replaces_the_anchor_with_the_measurement():
    f = _front()
    prev = _ids(30000, salt=4)
    f.ftok.m["prev"] = prev
    f._p_anchor_presence("r", "prev", types.SimpleNamespace(leg1_prompt_tokens=30000))
    f._x_exact_record("r", "prev", 30000, 29952, types.SimpleNamespace(d_direct=False), None,
                      resumable_depth=29952)
    full = np.concatenate([prev, np.arange(10 ** 6, 10 ** 6 + 500, dtype=np.int32)])
    assert f.tspans.pending(full, epoch=None)[1] == 29952


def test_k1_switch_default_off_and_wired_at_first_content():
    import inspect

    assert F.envs.SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE.get() is False
    src = inspect.getsource(F.Front.leg2)
    assert "envs.SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE.get()" in src
    assert "self._p_anchor_presence(rid, text, pending)" in src
    # only on an after_p leg with content (pending set, not a single prefill)
    i = src.index("self._p_anchor_presence(rid, text, pending)")
    guard = src[src.rindex("if (_has_content", 0, i):i]
    assert "pending is not None" in guard and "not single_prefill" in guard
