# SPDX-License-Identifier: Apache-2.0
"""PREFIX-DIVERGE (NF y7l 99d1977a63, 02.10.): "80k re-prefilled for no reason
instead of taken from the cache" -- client or store?

y7l (boot_weg2_dkrnfint4h6bar1dauer10021206_99d1977a63_1002_120637):
  * pdflip-4-18 (78349 tokens) priced credit=6400 src=l3_index, P loaded 6400;
    P's arena chain left pdflip-2-7's at page 51 (``#1439 ARENA-PRESENT
    leading_complete=51 break_slots=[(-1, 0)]``), the first page the disk
    read missed (42622c98, ``READ-TRACE why=no-file`` 12:13:35) was born at
    12:13:44 (store journal ``R``/``C``) by pdflip-4-18's OWN prefill;
  * pdflip-8-20 (165382 tokens) priced credit=78348, P loaded 78336 of
    pdflip-4-18's pages; pdflip-6-19 had left pdflip-4-18 at page 1222, so
    pdflip-8-20's page 1224 (a0e0a1a2, missed 12:17:10) was born at 12:17:20;
  * none of the 12850 pages written in that boot was evicted (the 3826 ``E``
    lines of D's EVICTION-BG are all older boots'), the KV arena never freed
    a page (complete 1505 -> 2813 -> 4046 -> 5468 of 6485 slots).

So the prompts changed, the cache did not lose them. The front prices every
arrival with exact token ids already; this line says, per long arrival, how
far it agrees with ANY earlier prompt against the credit it got.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import Count, TokenSpans  # noqa: E402

PAGE = 64
X = 4000


def _ids(n, seed):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


def _lcp(a, b):
    n = min(a.size, b.size)
    ne = np.flatnonzero(a[:n] != b[:n])
    return int(ne[0]) if ne.size else n


class _Store:
    """The L2/L3 store in small: stored token sequences with their Mamba
    anchors (token depths); ``depth`` = the deepest anchor on the shared path,
    which is what ``front_store.StorePresence.depth`` answers."""

    def __init__(self):
        self.seqs = []

    def put(self, ids, anchors):
        self.seqs.append((ids, sorted(anchors)))

    def depth(self, ids):
        best = 0
        for s, anchors in self.seqs:
            lcp = _lcp(s, ids)
            best = max([best] + [a for a in anchors if a <= lcp])
        return types.SimpleNamespace(tokens=best, tier="l3_index", pages=best // PAGE,
                                     l3_pages=best // PAGE, kv_pages=best // PAGE, ms=0.1)


class _Tok:
    state = "ready"
    why = ""
    executor = None

    def __init__(self):
        self.m = {}

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        ids = payload["ids"]
        return Count(n=int(ids.size), ids=ids, ms=1.0, reused=0, encoded=int(ids.size))


def _front(store):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = "D"
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = store
    return f


def _price(f, rid, ids):
    return asyncio.run(f._x_exact_price(rid, "/v1/chat/completions", {"ids": ids}, rid,
                                        int(ids.size), int(ids.size)))


def _lines(caplog, rid):
    return [m for m in caplog.messages if m.startswith(f"PDFLIP PREFIX-DIVERGE rid={rid} ")]


def test_y7l_pdflip_8_20_left_pdflip_6_19_and_the_store_had_all_it_shares(caplog):
    """y7l /10: A = pdflip-4-18, B = pdflip-6-19 (A's head, then 86k new),
    C = pdflip-8-20 (all of A, then B's new text again). C's credit is A's end
    anchor -- everything any earlier prompt shares with C."""
    caplog.set_level(logging.INFO)
    st = _Store()
    f = _front(st)
    a = _ids(7834, 1)
    _price(f, "pdflip-4-18", a)            # priced before A's prefill: nothing shared
    st.put(a, [6400, 7744, 7808])        # A prefilled on P: anchors incl. its END-ANCHOR
    b = np.concatenate([a[:7790], _ids(8600, 2)])
    xb = _price(f, "pdflip-6-19", b)
    assert xb.credit == 7744
    st.put(b, [b.size // PAGE * PAGE])
    c = np.concatenate([a, b[7790:], _ids(160, 3)])
    xc = _price(f, "pdflip-8-20", c)
    assert (xc.credit, xc.src) == (7808, "l3_index")
    assert xc.pending > X, "a P leg: the re-prefill the operator saw"
    (line,) = _lines(caplog, "pdflip-8-20")
    assert "verdict=client_diverged" in line
    assert "best_prev=pdflip-4-18 prev_prompt=7834" in line and "common=7834 " in line
    assert " lost=0 " in line
    assert f.counters["prefix_diverge_client_diverged"] == 3
    assert f.counters["prefix_diverge_store_short"] == 0


def test_y7l_pdflip_4_18_changed_its_head_inside_the_system_prompt(caplog):
    """pdflip-2-7 then pdflip-4-18: the same 72k of conversation, but the prompt
    leaves pdflip-2-7 at token 3291 (page 51); the store credits an older
    boot's 6400 -- more than any prompt of this boot shares."""
    caplog.set_level(logging.INFO)
    st = _Store()
    older = np.concatenate([_ids(3290, 4)[:3290], _ids(3200, 5)])  # an earlier boot's variant
    st.put(older, [6400])
    f = _front(st)
    p27 = np.concatenate([older[:3290], _ids(76000, 6)])
    _price(f, "pdflip-2-7", p27)
    st.put(p27, [p27.size // PAGE * PAGE])
    p418 = np.concatenate([older[:6414], p27[6414:72000]])
    x = _price(f, "pdflip-4-18", p418)
    assert x.credit == 6400 and x.pending > X
    (line,) = _lines(caplog, "pdflip-4-18")
    assert "verdict=client_diverged" in line and "best_prev=pdflip-2-7" in line
    assert "common=3291 common_page=3264 lost=0 " in line


def test_a_prefix_this_boot_prefilled_and_the_store_lost_is_store_short(caplog):
    """The class the operator suspected, reproduced: P prefilled E, the store
    no longer has it at the next turn -> WARNING, the lost tokens named."""
    caplog.set_level(logging.INFO)
    st = _Store()
    f = _front(st)
    e = _ids(8000, 7)
    _price(f, "pdflip-2-7", e)
    st.put(e, [7936])
    st.seqs.clear()                       # the store lost E's pages
    nxt = np.concatenate([e, _ids(2000, 8)])
    x = _price(f, "pdflip-4-18", nxt)
    assert x.credit == 0
    (line,) = _lines(caplog, "pdflip-4-18")
    assert "verdict=store_short" in line and "best_prev=pdflip-2-7" in line
    assert "common=8000 common_page=8000 lost=8000 " in line
    rec = [r for r in caplog.records if r.getMessage() == line][0]
    assert rec.levelno == logging.WARNING
    assert f.counters["prefix_diverge_store_short"] == 1
    assert f.counters["prefix_diverge_store_short_tokens"] == 8000


def test_a_short_arrival_is_noted_not_logged(caplog):
    caplog.set_level(logging.INFO)
    st = _Store()
    f = _front(st)
    a = _ids(9000, 9)
    _price(f, "pdflip-0-1", a)
    st.put(a, [8960])
    _price(f, "pdflip-0-2", np.concatenate([a, _ids(100, 10)]))
    assert _lines(caplog, "pdflip-0-2") == []
    assert list(f._recent_prompts.items) == ["pdflip-0-1", "pdflip-0-2"]


def test_the_verdict_and_the_ring():
    from flliper.srt.pdflip.front_tokens import RecentPrompts, reprefill_verdict

    assert reprefill_verdict(78336, 78349) == ("client_diverged", 0)
    assert reprefill_verdict(6400, 3290) == ("client_diverged", 0)
    assert reprefill_verdict(6208, 79484) == ("store_short", 79424 - 6208)
    rp = RecentPrompts(cap=2)
    for i in range(3):
        rp.note(f"r{i}", _ids(100 + i, i))
    assert list(rp.items) == ["r1", "r2"]
    q = _ids(101, 1)
    assert rp.best(q)[:3] == (101, "r1", 101)
    assert rp.best(q, exclude_rid="r1")[1] in (None, "r2")


def test_an_instrument_never_raises():
    f = _front(_Store())
    f._prefix_diverge("r", None, 10**6, 0, "none", 0)
    f._prefix_diverge("r", "not-an-array", 10**6, 0, "none", 0)
