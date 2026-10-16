"""X-GESETZ F-X (user 16.10. 12:35Z: "aktuell wird in P DEUTLICH unter der X-Schwelle geprefillt ...
HALLOOO"; root analysis deskq/done/nf-xgesetz-short-auf-p-1016.md, classes A, B, C).

The 16.10. order LIFTS P-BATCH-ALL (02.10., fc98fce9e2: "while P is awake every prefill is collected on
P") for SHORT requests: a request under X is prefilled on D; P takes only over X, p_only (image, W102)
and the pure X-SUM release of DECODE-COLLECT (user 12.10.). NF h16s @ 1f4b3e73d4, boot 1016_124240
(window 12:46-13:16Z): 39 of 49 P leg-1 rids were SHORT -- 48 one-request P batches, 68.8 s of PP0 GPU.

  A  weg2-0-2   12:45:59,377 SHORT-TO-P uncached=46 presence=117312 src=l3_index (P woken by the image
                weg2-0-1) -> BATCH queued -> 12:46:12,534 SERVED group=P leg=1, PP0 gpu-ms 1449.7.
  B  weg2-4-37  12:48:56,560 DECODE-COLLECT release n=6 tokens=77570 X=7043 route=P p_bound=True: the
                77570 hold weg2-4-33 (74218, p_only after W50); the SHORTs sum ~3300 < X ->
                COLLECT-P-QUEUED "(sum over X)" (false text) -> SERVED P leg1.
  C  weg2-6-59  12:55:07 SHORT-KEPT "no leg 1 on P" -> 12:55:15,769 P-BATCH-TAKES was=short_kept
                where=ready_for_d -> 12:55:35 SERVED P leg1 (PP0 gpu-ms 1333.4).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front as _asr_front  # noqa: E402

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front import Front, Pending  # noqa: E402

X_0_2 = 6599           # the live X around 12:46 (an order of magnitude; the case is 46 << X)
PREFIX = "a" * 351936  # chars/3 = 117312 tokens: weg2-0-2's l3_index presence
TEXT = PREFIX + "b" * 138  # +46 tokens


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


def _arrive(f, text=TEXT):
    async def go():
        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req("/generate", {"text": text})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        out = (list(f.queue), list(f._ready_for_d))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return out

    return asyncio.run(go())


def _pending(rid, age_s, est_uncached, **kw):
    loop = asyncio.new_event_loop()
    return Pending(rid=rid, path="/generate", payload={}, text="x", t_arrive=time.time() - age_s,
                   fut=loop.create_future(), est_prompt=est_uncached + 1000, est_uncached=est_uncached, **kw)


# ---- A: a SHORT arriving while P is awake is held for D ----------------------------------------

def test_a_weg2_0_2_arrives_while_p_is_awake_and_is_held_for_d(caplog):
    caplog.set_level(logging.INFO)
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X_0_2)
    f.state = "serving"
    f.admit_d = True
    f.spans.record_presence(PREFIX, 117312)
    queued, ready = _arrive(f)
    assert queued == [], "class A: no leg 1 on P for a 46-token SHORT"
    assert len(ready) == 1
    p = ready[0]
    assert p.est_uncached <= 47 and p.d_direct and p.skip_leg1 and p.short_kept
    assert not any("WEG2 SHORT-TO-P rid=" in m for m in caplog.messages)
    assert any("WEG2 SHORT-HOLD-D rid=" in m and "why=x_law" in m for m in caplog.messages)


def test_a_a_long_arriving_while_p_is_awake_still_goes_to_p():
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X_0_2)
    f.state = "serving"
    f.admit_d = True
    queued, ready = _arrive(f, text="c" * 30000)  # 10000 tokens > X, nothing cached
    assert ready == [] and len(queued) == 1 and not queued[0].skip_leg1 and not queued[0].d_direct


# ---- C: the P drain's start leaves a SHORT-KEPT in D's line --------------------------------------

def test_c_weg2_6_59_short_kept_stays_with_d_at_the_p_drain_start(caplog):
    caplog.set_level(logging.INFO)
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=7043)
    f.state = "serving"
    kept = _pending("weg2-6-59", 30.0, 214, skip_leg1=True, short_kept=True, d_direct=True)
    pbound = _pending("weg2-6-64", 20.0, 9000)
    f._ready_for_d.append(kept)
    f.queue.append(pbound)
    assert f._p_batch_takes_d_prefill() == 0
    assert list(f._ready_for_d) == [kept] and kept.d_direct and kept.skip_leg1
    assert list(f.queue) == [pbound]
    assert not any("WEG2 P-BATCH-TAKES rid=weg2-6-59" in m for m in caplog.messages)


# ---- B: DECODE-COLLECT route P splits the set ------------------------------------------------------

def _collect_front(monkeypatch, queue, x=7043):
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", "0")
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)
    f = _asr_front(running=["r1"], n=6)
    f.tp_prefill_max_tokens = x
    f.t_awake = time.time() - 60.0
    f._dc_d_busy = lambda: True
    f._dc_decoding = lambda: False   # stage d-idle: the sum decides now
    f._dc_carried = lambda: 0
    f.queue = list(queue)
    return f


def test_b_weg2_4_37_the_short_members_of_a_p_bound_release_stay_d(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    big = _pending("weg2-4-33", 10.0, 74218, p_only=True)
    shorts = [_pending("weg2-4-37", 9.0, 52), _pending("weg2-4-38", 8.0, 1600), _pending("weg2-4-39", 7.0, 1648)]
    f = _collect_front(monkeypatch, [big] + shorts)
    route = f._decode_collect(time.time())
    assert route == "P", "the p_only member flips D to P"
    assert not big.x_sum_p and all(not q.x_sum_p for q in shorts)
    assert all(Front._x_law_d_short(f, q) for q in shorts) and not Front._x_law_d_short(f, big)
    assert any("DECODE-COLLECT split route=P cause=p_bound" in m for m in caplog.messages)
    assert any("DECODE-COLLECT release" in m and "cause=p_bound" in m for m in caplog.messages)
    # the P drain's sweep: the SHORTs wait in D's line, P gets the p_only one
    f._sync_batch_gate = lambda: None
    f._to_p_batch = lambda p, where: None
    assert Front._p_batch_takes_d_prefill(f) == 0
    assert [q.rid for q in f.queue] == ["weg2-4-33"]
    assert [q.rid for q in f._ready_for_d] == ["weg2-4-37", "weg2-4-38", "weg2-4-39"]


def test_b_the_pure_x_sum_release_stays_p_for_the_whole_set(monkeypatch):
    shorts = [_pending("s1", 9.0, 4000), _pending("s2", 8.0, 3900)]  # 7900 > X, nothing P-bound
    f = _collect_front(monkeypatch, shorts)
    assert f._decode_collect(time.time()) == "P"
    assert all(q.x_sum_p for q in shorts), "user 12.10.: the sum over X goes to P"
    f._sync_batch_gate = lambda: None
    assert Front._p_batch_takes_d_prefill(f) == 0
    assert [q.rid for q in f.queue] == ["s1", "s2"] and f._ready_for_d == []


def test_b_a_short_the_sum_leaves_over_goes_to_p(monkeypatch):
    big = _pending("big", 10.0, 9000)
    a, b = _pending("a", 9.0, 5000), _pending("b", 8.0, 3000)  # oldest first: a fits, a+b > X
    f = _collect_front(monkeypatch, [big, a, b])
    assert f._decode_collect(time.time()) == "P"
    assert not a.x_sum_p and b.x_sum_p


def test_b_a_seat_path_short_of_a_split_is_no_x_sum_verdict(monkeypatch):
    """A SHORT waiting on the seat path in the split window falls through WITHOUT collect_p: the arrival
    then holds it for D (no COLLECT-P-QUEUED, no leg 1 on P)."""
    big = _pending("big", 10.0, 9000)
    f = _collect_front(monkeypatch, [big])
    st = f._dc_st()
    st["shorts"]["seat-s"] = (time.time() - 5.0, 300)
    st["t_open"] = time.time() - 5.0
    assert f._decode_collect(time.time()) == "P"
    assert "seat-s" in (f._dc_st().get("x_law_d") or {})
    src = inspect.getsource(F.Front._acquire_short_seat)
    i = src.index('refused.append("collect_p")')
    assert 'rid not in (self._dc_st().get("x_law_d") or {})' in src[i - 800:i]


# ---- the X-law predicate -----------------------------------------------------------------------------

def test_the_x_law_predicate_names_what_p_takes():
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=4096)
    assert Front._x_law_d_short(f, _pending("s", 1, 4096))           # at X: D's (law 4: never ABOVE X)
    assert not Front._x_law_d_short(f, _pending("o", 1, 4097))       # over X
    assert not Front._x_law_d_short(f, _pending("i", 1, 10, p_only=True))
    assert not Front._x_law_d_short(f, _pending("w", 1, 10, x_requeues=1))
    assert not Front._x_law_d_short(f, _pending("r", 1, 10, x_routed=8))
    assert not Front._x_law_d_short(f, _pending("x", 1, 10, x_sum_p=True))
    assert not Front._x_law_d_short(f, _pending("l", 1, 10, leg1_done=True))
    assert not Front._x_law_d_short(f, _pending("v", 1, 10, resume_via_p=True))
    assert not Front._x_law_d_short(f, _pending("k", 1, 10, skip_leg1=True, sk_void_seq=3))
    assert Front._x_law_d_short(f, _pending("kept", 1, 10, skip_leg1=True, short_kept=True))
