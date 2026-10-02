# SPDX-License-Identifier: Apache-2.0
"""STORE-PRESENCE (user law 02.10.: L2/L3 are shared, the credit is the STORE
prefix) and X-EXACT-BACKFILL on NF.

27B port (02.10., desk/27b-nfport-route-1002): the NF cases below are kept as
token arithmetic (model-neutral); the 27B cases of boot b6a878f145 live in
test_weg2_x_credit_store_27b_1002.py.

y6y (0e1967fd36), front log ..._1002_074625.front.log: 8 LONG verdicts whose
P leg 1 computed <= X new tokens, all priced src=none / credit=0:
  * the six requests before 'X-EXACT READY' (weg2-0-1..0-5, 1-6, reason
    tokenizer_loading) never got token ids, so none of their legs recorded a
    presence: weg2-4-10 (P hit 67328 of 67585), 5-12 (67520 of 67740),
    5-11 (105216 of 105351), 10-17 (60544 of 60556) went LONG -- the 27B
    fix 74e51fb2c4 (X-EXACT-BACKFILL) never reached NF;
  * weg2-8-13 (07:51:06.956): P wrote weg2-5-11's END-ANCHOR 105344 at
    07:51:02, P's sleep flush published it at 07:51:04 ('#1470 FLUSH-PUBLISH
    ... unbacked_left=0'), D's first content came at 07:51:07.15 -- priced
    credit 0, LONG, P prefilled 162 on a 105344 hit.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import Count, TokenSpans  # noqa: E402


def _ids(n):
    return np.arange(n, dtype=np.int32)


class _Tok:
    state = "ready"

    def __init__(self, counts=None):
        self.m = {}
        self.counts = counts or {}
        self.executor = None

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        ids = self.counts[payload["text"]]
        return Count(n=int(ids.size), ids=ids, ms=1.0, reused=0, encoded=int(ids.size))


def _front(epoch=7, awake="P"):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = epoch
    f.awake = awake
    f.state = "serving"
    f.tp_prefill_max_tokens = 4323
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    return f


# ---- weg2-8-13: P's END-ANCHOR is a store prefix once P's flush ran -------------

def test_weg2_8_13_priced_short_on_ps_flushed_end_anchor(caplog):
    caplog.set_level(logging.INFO)
    f = _front()
    f.ftok.m["5-11"] = _ids(105351)
    f._p_leg1_store_note("weg2-5-11", "5-11", 105351)
    cur = _ids(105506)  # 8-13 shares 5-11's whole prompt
    assert f.tspans.pending(cur)[1] == 0, "before P's flush: no store witness"
    assert f._p_flush_store_presence([]) == 1
    pending, credit, known, src = f.tspans.pending(cur, epoch=None)
    assert (credit, src) == (105344, "store_anchor")
    assert pending == 162 <= 4323, "SHORT (P prefilled 162 on a 105344 hit)"
    assert any("WEG2 STORE-PRESENCE src=p_flush n=1" in m and "weg2-5-11:105344" in m
               for m in caplog.messages)
    assert f.counters["store_presence"] == 1


def test_an_anchor_ps_flush_lost_is_not_recorded():
    f = _front()
    f.ftok.m["5-11"] = _ids(105351)
    f._p_leg1_store_note("weg2-5-11", "5-11", 105351)
    assert f._p_flush_store_presence([105344]) == 0
    assert f.tspans.pending(_ids(105506))[1] == 0


def test_the_finish_reading_replaces_the_store_label():
    f = _front()
    f.ftok.m["5-11"] = _ids(105351)
    f._p_leg1_store_note("weg2-5-11", "5-11", 105351)
    f._p_flush_store_presence([])
    f._x_exact_record("weg2-5-11", "5-11", 105351, 105348, None, None, resumable_depth=105344)
    assert f.tspans.pending(_ids(105506))[3] == "d_leg2_cached"


# ---- weg2-4-10: the base prefix was priced while the tokenizer loaded -------------

def test_weg2_4_10_a_fallback_base_prefix_gets_ids_at_its_p_leg():
    async def go():
        f = _front()
        base = _ids(66967)  # weg2-0-5, priced by chars/3 at 07:49:22
        f.ftok.counts["0-5"] = base
        loop = asyncio.get_running_loop()
        f.ftok.executor = None  # run_in_executor(None, ...) = the default pool
        f._x_exact_note_fallback("weg2-0-5", "tokenizer_loading")
        await f._x_exact_backfill("weg2-0-5", "/generate", {"text": "0-5"}, "0-5")
        f._p_leg1_store_note("weg2-0-5", "0-5", 66967)
        f._p_flush_store_presence([])
        return f, loop

    f, _loop = asyncio.run(go())
    assert f.counters["x_exact_backfilled"] == 1
    pending, credit, _k, src = f.tspans.pending(_ids(67585), epoch=None)
    assert (credit, src) == (66944, "store_anchor")
    assert pending == 641 <= 3874, "weg2-4-10 SHORT (P prefilled 257 on a 67328 hit)"


def test_backfill_only_for_the_loading_and_timeout_reasons():
    async def go():
        f = _front()
        f.ftok.counts["t"] = _ids(10)
        f._x_exact_note_fallback("a", "multimodal")
        await f._x_exact_backfill("a", "/generate", {"text": "t"}, "t")
        return f

    f = asyncio.run(go())
    assert f.ftok.ids_for("t") is None and f.counters["x_exact_backfilled"] == 0


# ---- a re-price that crosses X downwards makes it a SHORT ---------------------------

def test_a_queued_long_repriced_below_x_becomes_d_eligible(caplog):
    caplog.set_level(logging.INFO)
    f = _front(awake="D")
    f.ftok.m["0-2"] = _ids(104893)
    f.ftok.m["1-7"] = _ids(105048)
    p = F.Pending("weg2-1-7", "/generate", {}, "1-7", time.time(),
                  asyncio.new_event_loop().create_future(), est_prompt=105048,
                  est_uncached=105048)  # 27B Pending has no x_routed
    f.queue.append(p)
    f._p_leg1_store_note("weg2-0-2", "0-2", 104893)
    f._p_flush_store_presence([])
    assert p.est_uncached == 105048 - 104832 == 216
    assert p.d_eligible
    assert f.counters["x_exact_reprice_short"] == 1


# ---- wiring ---------------------------------------------------------------------------

def test_wiring():
    src = inspect.getsource(F.Front)
    i = src.index('logger.info("WEG2-SERVED group=P leg=1 rid=%s')
    blk = src[i:i + 700]
    assert "await self._x_exact_backfill(p.rid, p.path, p.payload, p.text)" in blk
    assert "self._p_leg1_store_note(p.rid, p.text, pt)" in blk
    assert ('elif src == "P" and s_code == 200:\n'
            '            # STORE-PRESENCE: P\'s flush published its END-ANCHORs (L2/L3, shared)\n'
            '            self._p_flush_store_presence(_lost_all)') in src
    j = src.index("await self._x_exact_backfill(rid, request.path, payload, text)")
    assert j < src.index("if pending is not None and pending.skip_leg1:\n            single_prefill = True")
    k = src.index("if c is None:\n            self.counters[\"x_exact_fallback\"] += 1")
    assert "self._x_exact_note_fallback(rid, reason)" in src[k:k + 200]
