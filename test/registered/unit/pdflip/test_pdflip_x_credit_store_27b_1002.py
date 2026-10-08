# SPDX-License-Identifier: Apache-2.0
"""STORE-PRESENCE (NF ba76adffe2) on the 27B line -- replays of the FEHLPREIS
cases of 27B boot b6a878f145 (front log
boot_weg2_dkr27browauthoritycut43bar1fs10020728_b6a878f145_1002_072908.front.log,
P log ...P.log of the same boot).

Token ids are synthesised from the log's facts the way
test_pdflip_p_anchor_presence_replay_w1_0930 does it: a request whose X-EXACT
``reused`` prefix equals an earlier request's whole prompt shares exactly that
prompt with it, fresh tokens after it.

* pdflip-28-45 (07:39:11.692, D phase, X=4096): 'X-EXACT-PRICE pending=4971
  tokens=55761 credit=50790 src=d_leg2_cached ... reused=55607' -> LONG; P had
  served pdflip-26-43 (55607) at 07:39:06.964, P's sleep flush ran 07:39:07
  ('#1470 FLUSH-PUBLISH ... unbacked_left=0', 'PDFLIP-ANCHOR-LOST at=flush n=1
  depths=[52224]'), K1's witness for 26-43 came only at 07:39:12.105, the
  re-price then said 'est_uncached 4971 -> 209 X=8452 crossed=no' and the flip
  to P still ran for 159 real tokens (P: prompt 55761 cached 55602).
* pdflip-12-13 (07:35:16.618, D phase, X=9132): 'pending=46396 credit=0
  src=none ... reused=45284' -> LONG, P prefilled 1084 (cached 45312). Its
  prefix is pdflip-0-4's whole prompt (45284, priced by chars/3 before 'X-EXACT
  READY', ids backfilled, P leg 1 in epoch 1); 0-4's leg 2 ended unpriced
  (LEG2-TERMINAL-NAMED), so K1 never fired for it. P's flush at 07:33:18 lost
  only inner anchors ([36864, 40960, 45056, ...]).
* pdflip-6-10 (07:34:17.079) is NOT this class: reused=45244 < 0-4's END-ANCHOR
  45248 -- it leaves 0-4 before the anchor; P's real hit 45056 is an inner
  anchor. It must stay LONG (no over-credit).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import Count, TokenSpans  # noqa: E402

_fresh = [50_000_000]


def _fresh_ids(n):
    a = np.arange(_fresh[0], _fresh[0] + n, dtype=np.int32)
    _fresh[0] += n
    return a


def _extend(head, n):
    return np.concatenate([head, _fresh_ids(n)])


class _Tok:
    state = "ready"
    executor = None

    def __init__(self):
        self.m = {}
        self.counts = {}

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        ids = self.counts[payload["text"]]
        return Count(n=int(ids.size), ids=ids, ms=1.0, reused=0, encoded=int(ids.size))


def _front(epoch, awake, x):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = epoch
    f.awake = awake
    f.state = "serving"
    f.tp_prefill_max_tokens = x
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    return f


def _queued(f, rid, text, est):
    return F.Pending(rid, "/v1/messages", {}, text, time.time(),
                     asyncio.new_event_loop().create_future(), est_prompt=est,
                     est_uncached=est, route_long=True)


# ---- pdflip-28-45 ---------------------------------------------------------------------------

def _b6a_28_45():
    ids_24_40 = _fresh_ids(50795)
    ids_26_43 = _extend(ids_24_40, 55607 - 50795)        # reused=50795 (24-40's whole prompt)
    ids_28_45 = _extend(ids_26_43, 55761 - 55607)        # reused=55607 (26-43's whole prompt)
    f = _front(epoch=27, awake="P", x=4096)
    f.ftok.m.update({"24-40": ids_24_40, "26-43": ids_26_43, "28-45": ids_28_45})
    # 24-40's D finish (07:39:00.446): prompt 50795 cached 50790 -> credit 50790
    f.tspans.record_presence(ids_24_40, 50790, prompt_tokens=50795, held_epoch=None)
    return f, ids_28_45


def test_pdflip_28_45_base_price_is_the_logs():
    f, ids = _b6a_28_45()
    pending, credit, _k, src = f.tspans.pending(ids, epoch=None)
    assert (pending, credit, src) == (4971, 50790, "d_leg2_cached"), "the log's price, LONG at X=4096"


def test_pdflip_28_45_is_short_once_ps_flush_published_26_43(caplog):
    caplog.set_level(logging.INFO)
    f, ids = _b6a_28_45()
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)    # PDFLIP-SERVED group=P leg=1 07:39:06.964
    assert f.tspans.pending(ids)[0] == 4971, "noted, not yet flushed: no credit"
    assert f._p_flush_store_presence([52224]) == 1         # P sleep 200, ANCHOR-LOST [52224]
    pending, credit, _k, src = f.tspans.pending(ids, epoch=None)
    assert (credit, src) == (55552, "store_anchor")
    assert pending == 209 <= 4096, "SHORT at 07:39:11.692 (P: 55761 cached 55602, 159 real)"
    assert any("PDFLIP STORE-PRESENCE src=p_flush n=1" in m and "pdflip-26-43:55552" in m
               for m in caplog.messages)


def test_pdflip_28_45_queued_long_repriced_under_a_moved_x_becomes_d_eligible(caplog):
    # the log: priced LONG at X=4096, then 'X-EXACT-REPRICE ... 4971 -> 209 X=8452 crossed=no'
    # (X moved up meanwhile) -- the flip to P still ran. On the port it is a SHORT now.
    caplog.set_level(logging.INFO)
    f, ids = _b6a_28_45()
    f.awake = "D"
    f.tp_prefill_max_tokens = 8452
    p = _queued(f, "pdflip-28-45", "28-45", 4971)
    f.queue.append(p)
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)
    f._p_flush_store_presence([52224])
    assert p.est_uncached == 209
    assert p.d_eligible, "a LONG re-priced to <= X is a SHORT (D-SHORT-DRAIN may take it)"
    assert f.counters["x_exact_reprice_short"] == 1


def test_a_refused_short_repriced_lower_stays_out_of_the_d_short_drain():
    # RC2 review: a SHORT that D's #915 budget refused at arrival (route_long False,
    # d_eligible False, price already <= X) must not become d_eligible by a re-price
    f, ids = _b6a_28_45()
    f.awake = "D"
    f.tp_prefill_max_tokens = 8452
    p = _queued(f, "pdflip-28-45", "28-45", 4971)
    p.route_long = False
    f.queue.append(p)
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)
    f._p_flush_store_presence([])
    assert p.est_uncached == 209 and not p.d_eligible


def test_the_anchor_the_flush_lost_is_not_credited():
    f, ids = _b6a_28_45()
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)
    assert f._p_flush_store_presence([55552]) == 0
    assert f.tspans.pending(ids)[0] == 4971


def test_the_switch_off_records_nothing(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_P_ANCHOR_PRESENCE", "0")
    f, ids = _b6a_28_45()
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)
    assert f._p_flush_store_presence([]) == 0
    assert f.tspans.pending(ids)[0] == 4971


def test_the_dual_layout_notes_nothing():
    f, _ids = _b6a_28_45()
    f.dual_layout = True
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)
    assert not f.__dict__.get("_p_phase_served")


# ---- pdflip-12-13 / pdflip-6-10 -------------------------------------------------------------

def _b6a_epoch1():
    async def go():
        f = _front(epoch=1, awake="P", x=9132)
        ids_0_4 = _fresh_ids(45284)
        f.ftok.counts["0-4"] = ids_0_4
        # 07:31:30.866 'X-EXACT-FALLBACK rid=pdflip-0-4 reason=tokenizer_loading'
        f._x_exact_note_fallback("pdflip-0-4", "tokenizer_loading")
        # 07:32:05.243 'PDFLIP-SERVED group=P leg=1 rid=pdflip-0-4 prompt_tokens=45284 cached_tokens=45056'
        await f._x_exact_backfill("pdflip-0-4", "/v1/messages", {"text": "0-4"}, "0-4")
        f._p_leg1_store_note("pdflip-0-4", "0-4", 45284)
        # 07:33:18 P flush: 'PDFLIP-ANCHOR-LOST at=flush n=9 depths=[36864, 40960, 45056, 106496, ...]'
        f._p_flush_store_presence([36864, 40960, 45056, 106496, 110592, 114688, 118784, 122880, 126976])
        return f, ids_0_4

    return asyncio.run(go())


def test_pdflip_12_13_priced_short_on_0_4s_flushed_end_anchor():
    f, ids_0_4 = _b6a_epoch1()
    assert f.counters["x_exact_backfilled"] == 1, "ids at the P leg, before the flush"
    ids_12_13 = _extend(ids_0_4, 46396 - 45284)            # reused=45284 (0-4's whole prompt)
    pending, credit, _k, src = f.tspans.pending(ids_12_13, epoch=None)
    assert (credit, src) == (45248, "store_anchor")
    assert pending == 1148 <= 9132, "SHORT (the log: LONG, P prefilled 1084 on a 45312 hit)"


def test_pdflip_6_10_leaves_0_4_before_its_anchor_and_stays_long():
    f, ids_0_4 = _b6a_epoch1()
    ids_6_10 = _extend(ids_0_4[:45244], 45434 - 45244)      # reused=45244 < END-ANCHOR 45248
    pending, credit, _k, _src = f.tspans.pending(ids_6_10, epoch=None)
    assert credit == 0 and pending == 45434 > 3991, "no over-credit past the divergence (PX)"


def test_the_reprice_line_names_short_now(caplog):
    caplog.set_level(logging.INFO)
    f, _ids = _b6a_28_45()
    f.awake = "D"
    f.tp_prefill_max_tokens = 8452
    f.queue.append(_queued(f, "pdflip-28-45", "28-45", 4971))
    f._p_leg1_store_note("pdflip-26-43", "26-43", 55607)
    f._p_flush_store_presence([])
    line = [m for m in caplog.messages if "X-EXACT-REPRICE rid=pdflip-28-45" in m][0]
    assert "est_uncached 4971 -> 209 X=8452 crossed=no src=store_anchor" in line and "short_now=1" in line
