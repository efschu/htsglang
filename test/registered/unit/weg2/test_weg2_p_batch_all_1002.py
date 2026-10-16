"""X-GESETZ F-X (user 16.10.: "aktuell wird in P DEUTLICH unter der X-Schwelle
geprefillt") LIFTS P-BATCH-ALL below for SHORTs: a request under X is prefilled
on D; a SHORT arriving while P is awake is HELD for D (D's line, no leg 1 on P),
and the P drain's start sweep leaves every SHORT in D's line. P takes only over
X, p_only and the pure X-SUM release. The tests below were re-pinned to that
law; the y6x shape (D prefills 207 tokens after the flip) is the cost the
16.10. order accepts.

Historical (02.10.): P-BATCH-ALL (user 02.10. ~07:45Z, strict phase batch): while P is awake,
EVERY prefill is collected on P; D after the flip only decodes (E2 contract,
HANDBACK d_compute=0). The X/SHORT price only decides whether a D phase wakes P.

METAL (y6x, boot_weg2_dkrnfint4bar1dauer10020710_fceeda8493_1002_071105):
07:21:05.08 weg2-15-52 (prompt 36879, pending 207, credit 36672) arrived while
P prefilled weg2-14-51 -> 'SHORT-BEHIND-P ... queued and KEPT for D (no leg 1
on P)'; after the P->D flip D prefilled the 207 tokens ('Prefill rank batch,
#new-token: 207, #cached-token: 36672 ... gpu-ms: 1949.5', pool.host_fetch
1438 ms) with three decodes behind it. The prefix WAS in the shared store:
D's sleep flush at 07:21:00 published every un-backed node ('#1421
BACKUP-REFUSED ... depth=36672' then '#1470 FLUSH-PUBLISH issued=14
unbacked_left=0', 'Cache flushed successfully!').
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front import Front, Pending  # noqa: E402

X = 4096
PREFIX = "a" * 110016   # chars/3 -> 36673 tokens; D measured 36672 of it
TEXT = PREFIX + "b" * 621  # +207 tokens (the 15-52 tail)


class _Req:
    def __init__(self, path, payload):
        self.path = path
        self._payload = payload
        self._d = {}

    async def json(self):
        return self._payload

    def __setitem__(self, k, v):
        self._d[k] = v

    def get(self, k, default=None):
        return self._d.get(k, default)


def _front(awake, state="serving", flip_dst=None):
    f = Front("http://p", "http://d", awake, "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = state
    f.admit_d = True
    if flip_dst is not None:
        f._flip_dst = flip_dst
    # the measured presence of weg2-14-50's prefix (credit 36672)
    f.spans.record_presence(PREFIX, 36672)
    return f


def _arrive(f):
    async def go():
        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req("/generate", {"text": TEXT})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        out = (list(f.queue), list(f._ready_for_d))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return out

    return asyncio.run(go())


# ---- (1) the 15-52 shape, under the X law ----------------------------------------

def test_weg2_15_52_a_short_arriving_while_p_is_awake_is_held_for_d(caplog):
    caplog.set_level(logging.INFO)
    f = _front("P")
    queued, ready = _arrive(f)
    assert queued == [], "X-GESETZ 16.10.: no leg 1 on P for a SHORT"
    assert len(ready) == 1
    p = ready[0]
    assert p.est_uncached <= 208, "priced on the measured presence (chars/3 here; 207 exact)"
    assert p.skip_leg1 and p.short_kept and p.d_direct, "held in D's line, D prefills it after the flip"
    assert not any("WEG2 SHORT-TO-P rid=" in m for m in caplog.messages)
    assert any("WEG2 SHORT-HOLD-D rid=" in m and "why=x_law" in m for m in caplog.messages)


def test_a_short_arriving_while_d_flips_to_p_is_held_for_d():
    f = _front("D", state="flipping", flip_dst="P")
    queued, ready = _arrive(f)
    assert queued == [] and len(ready) == 1
    assert ready[0].skip_leg1 and ready[0].short_kept and ready[0].d_direct


def test_the_d_phase_keep_is_unchanged():
    # D awake, admission closed (PARK-IMMEDIATE fired): SK still keeps it for
    # D -- a D-phase decision; the next P drain leaves it with D (below)
    f = _front("D")
    f.admit_d = False
    queued, _ready = _arrive(f)
    assert len(queued) == 1 and queued[0].short_kept and queued[0].skip_leg1


# ---- (2) at the start of a P drain: only what the X law binds to P goes to P ------

def _pending(rid, age_s, est_uncached=200, **kw):
    loop = asyncio.new_event_loop()
    return Pending(rid=rid, path="/generate", payload={}, text="x", t_arrive=time.time() - age_s,
                   fut=loop.create_future(), est_prompt=1000, est_uncached=est_uncached, **kw)


def test_the_p_drain_start_leaves_shorts_with_d_and_takes_only_over_x(caplog):
    caplog.set_level(logging.INFO)
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    kept = _pending("kept", 9.0, skip_leg1=True, short_kept=True, d_direct=True)   # SHORT-KEPT
    drained = _pending("drained", 7.0, d_direct=True)                            # D-SHORT-DRAIN
    prefilled = _pending("handoff", 8.0, leg1_done=True)                         # P did leg 1
    carrier = _pending("carrier", 6.0, skip_leg1=True, leg1_done=True)           # P leg 1 done
    over = _pending("over", 6.5, est_uncached=X + 1, d_direct=True)              # re-priced over X
    queued_kept = _pending("qkept", 5.0, skip_leg1=True, short_kept=True)        # D-phase fall-through
    later = _pending("later", 1.0)                                               # a queued SHORT
    big = _pending("big", 2.0, est_uncached=X + 500)                             # LONG
    img = _pending("img", 3.0, p_only=True)                                      # W102
    xsum = _pending("xsum", 4.0, x_sum_p=True)                                   # pure X-SUM release
    f._ready_for_d.extend([kept, prefilled, drained, over, carrier])
    f.queue.extend([queued_kept, big, img, xsum, later])
    n = f._p_batch_takes_d_prefill()
    assert n == 1, "only the D-direct entry over X goes to P"
    assert [p.rid for p in f._ready_for_d] == ["kept", "handoff", "drained", "carrier", "qkept", "later"]
    assert [p.rid for p in f.queue] == ["over", "xsum", "img", "big"], "arrival order"
    for p in (kept, queued_kept, later):
        assert p.skip_leg1 and p.short_kept and p.d_direct and not p.leg1_done
    assert drained.d_direct and not drained.leg1_done, "a D-SHORT-DRAIN hand-over stays as it is"
    assert not over.d_direct and not over.skip_leg1
    assert f.counters["p_batch_from_d"] == 1
    assert f.counters["x_law_hold_d"] == 2, "the two queued SHORTs moved to D's line"
    assert sum("WEG2 X-LAW-HOLD-D rid=" in m and "where=p_drain_start" in m for m in caplog.messages) == 2


def test_the_p_drain_sweeps_before_it_dispatches_and_holds_a_late_short_for_d():
    src = inspect.getsource(F.Front)
    i = src.index("# awake == P: prefill the backlog until empty")
    blk = src[i:i + 22000]
    assert blk.index("self._p_batch_takes_d_prefill()") < blk.index("_p_drain_pool(")
    h = blk.index("if Front._x_law_d_short(self, p):")
    a = blk.index('if getattr(p, "short_kept", False) or p.skip_leg1:')
    assert h < a < blk.index("async with sem:")
    assert 'Front._x_hold_for_d(self, p, "p_drain", append=False)' in blk[h:a]
    assert 'self._to_p_batch(p, "drain")' in blk[a:a + 300]


# ---- (3) the cut-off ----------------------------------------------------------------

def test_the_cut_off_is_the_begin_of_the_flip_to_d():
    assert F.p_phase_arrival("P", "serving", None) is True
    assert F.p_phase_arrival("D", "flipping", "P") is True
    assert F.p_phase_arrival("P", "flipping", "D") is False, "the flip to D began: D's (no reverse flip)"
    assert F.p_phase_arrival("D", "serving", None) is False
    assert F.p_phase_arrival("D", "serving", "P") is False, "a stale flip_dst outside a flip"
