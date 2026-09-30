"""X-EXACT-BACKFILL (30.09., freeze flip 3vsqkr, front log ...09301715_ead403b7ab): a request priced by the chars/3
fallback never got its token ids, so none of the presence records of its own legs could be written.

Measured: session s0 of k1_repeat_load (seed 930). T1 weg2-0-2 arrived at 17:18:39.309, 1.3 s before
``WEG2 X-EXACT READY`` (``X-EXACT-FALLBACK reason=tokenizer_loading``). It went LONG -> P, and D resumed P's end anchor
at its first content (LEG2-FIRST-CONTENT via=after_p, 17:18:52). But ``_p_anchor_presence`` reads
``ftok.ids_for(text)``, which is only a cache of the texts priced EXACTLY. So it got None and wrote no anchor (K1). D's
finish reading (``_x_exact_record``, 17:19:04, cached 20931) got the same None. So its twin T2 (weg2-2-4) and even
the follow-up T3 (weg2-8-7, the control) were priced credit=0 -> LONG -> P + 2 flips each; T4 (after T3's exact
reading) was credited. The other 4 sessions arrived after READY: all T2/T3 on D.

Fix: leg 2 of a request whose arrival fell back because the tokenizer was LOADING (or the count timed out) counts it
once, before D is asked, and remembers the ids. Every later reader of the text finds them. A fallback for a
reason that would give the wrong ids (multimodal, path, a payload the count raised on) is not backfilled.

DANGER DIRECTION = a wrong credit (W31/W50): the backfilled ids are exactly what the arrival pricing would have
counted (same ``ftok.count(path, payload)``); nothing else changes.
"""

import asyncio
import collections
import inspect
import types

import numpy as np

from sglang.srt.weg2 import front as F
from sglang.srt.weg2.front_tokens import Count, TokenSpans


class _Tok:
    """The front tokenizer as the readers see it: ids_for / remember / count / state / executor."""

    def __init__(self, ids, state="ready"):
        self.m, self.state, self.executor, self.calls = {}, state, None, 0
        self._ids = ids

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        self.calls += 1
        return Count(n=int(self._ids.size), ids=self._ids, ms=1.0, reused=0, encoded=int(self._ids.size))


def _front(ids, state="ready"):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 1
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok(ids, state)
    f._x_exact_rid = collections.OrderedDict()
    f._x_exact_reprice_queue = lambda why: 0
    return f


T1 = np.arange(20933, dtype=np.int32) + 7


def test_a_loading_fallback_is_counted_at_leg2_and_the_p_anchor_is_written():
    f = _front(T1)
    f._x_exact_note_fallback("weg2-0-2", "tokenizer_loading")
    asyncio.run(f._x_exact_backfill("weg2-0-2", "/v1/chat/completions", {"messages": []}, "T1"))
    assert f.ftok.calls == 1 and f.ftok.ids_for("T1") is not None
    assert f.counters["x_exact_backfilled"] == 1
    # the K1 witness of this leg now writes its anchor (it wrote 0 before: ids_for -> None)
    anchor = f._p_anchor_presence("weg2-0-2", "T1", types.SimpleNamespace(leg1_prompt_tokens=20933))
    assert anchor == 20928
    # the twin T2 = T1 + a short tail is credited on the shared path
    t2 = np.concatenate([T1, np.arange(43, dtype=np.int32) + 10 ** 6])
    pending, credit, _known, _src = f.tspans.pending(t2, epoch=None)
    assert credit == 20928 and pending == t2.size - 20928


def test_the_d_finish_reading_is_recorded_after_a_backfill():
    f = _front(T1)
    f._x_exact_note_fallback("r", "timeout")
    asyncio.run(f._x_exact_backfill("r", "/v1/chat/completions", {"messages": []}, "T1"))
    f._x_exact_record("r", "T1", 20933, 20931, types.SimpleNamespace(d_direct=False), None)
    t3 = np.concatenate([T1, np.arange(500, dtype=np.int32) + 10 ** 6])
    _pending, credit, _known, _src = f.tspans.pending(t3, epoch=None)
    assert credit > 0


def test_other_fallback_reasons_are_not_backfilled():
    for reason in ("multimodal", "path", "ValueError: /generate without a single text prompt", "tokenizer_failed"):
        f = _front(T1)
        f._x_exact_note_fallback("r", reason)
        asyncio.run(f._x_exact_backfill("r", "/v1/chat/completions", {"messages": []}, "T1"))
        assert f.ftok.calls == 0 and f.ftok.ids_for("T1") is None, reason


def test_nothing_happens_for_an_exactly_priced_or_unknown_rid():
    f = _front(T1)
    asyncio.run(f._x_exact_backfill("r", "/v1/chat/completions", {"messages": []}, "T1"))   # never fell back
    assert f.ftok.calls == 0
    f._x_exact_note_fallback("r", "tokenizer_loading")
    f.ftok.remember("T1", T1)                                                              # already known
    asyncio.run(f._x_exact_backfill("r", "/v1/chat/completions", {"messages": []}, "T1"))
    assert f.ftok.calls == 0


def test_a_tokenizer_still_loading_at_leg2_does_not_count():
    f = _front(T1, state="loading")
    f._x_exact_note_fallback("r", "tokenizer_loading")
    asyncio.run(f._x_exact_backfill("r", "/v1/chat/completions", {"messages": []}, "T1"))
    assert f.ftok.calls == 0 and f.ftok.ids_for("T1") is None


def test_the_wiring():
    src = inspect.getsource(F.Front._x_exact_price)
    assert "self._x_exact_note_fallback(rid, reason)" in src
    leg2 = inspect.getsource(F.Front.leg2)
    i = leg2.index("await self._x_exact_backfill(rid, request.path, payload, text)")
    # before D is asked: ahead of the first POST to group D in leg 2
    assert i < leg2.index("g.url")
