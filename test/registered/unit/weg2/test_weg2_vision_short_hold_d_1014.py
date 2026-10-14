"""VISION-SHORT-1014 (NF h13, boot ..._84e787ec96_1014_135753, weg2-20-129).

14:25:49 a 5-image request (134240 tokens, mm_cached=5/5) was priced SHORT on
credit=133696 src=d_served_anchor (D's own presence) while the D->P flip ran;
the store probe had answered depth=0 ('L3-INDEX-PRESENCE ... tier=none
depth=0'). 'SHORT-TO-P ... joins P's batch' sent it into P's batch: P ran the
tower, found 88 leading KV pages and no mamba anchor ('#1035c ZERO-ANSWER
PARTITION ... claimed=0 by=mamba'), prefilled all 134240 tokens in 39.6 s, TTFT
50.5 s, every lane-0 request parked meanwhile. D held the prefix on device
across every flip of that session (cached_device=134208 at weg2-22-130).

A SHORT whose credit only D holds stays for D; a store-backed one (y6x,
test_weg2_p_batch_all_1002) still joins P's batch.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front import Front, Pending, d_presence_only_short  # noqa: E402

X = 7733
N = 134240
CREDIT = 133696


# ---- the rule ---------------------------------------------------------------------

def test_d_served_anchor_with_empty_store_is_ds():
    assert d_presence_only_short("d_served_anchor", N, 0, X)


def test_every_d_source_counts_as_ds_own():
    for src in ("d_leg2_cached", "d_served_epoch", "d_served_anchor", "d_seq_anchor", "d_inflight"):
        assert d_presence_only_short(src, N, 0, X), src


def test_store_sources_stay_ps():
    for src in ("l3_index", "l2_arena", "store_anchor", "p_anchor", "mm_persist", "none", "", None):
        assert not d_presence_only_short(src, N, 0, X), src


def test_a_store_backed_d_credit_stays_ps():
    # y6x: D's sleep flush had published the prefix -- the probe sees it
    assert not d_presence_only_short("d_leg2_cached", 36879, 36672, 4096)
    # the boundary: P's remainder exactly X is still P's
    assert not d_presence_only_short("d_leg2_cached", 10000, 10000 - X, X)
    assert d_presence_only_short("d_leg2_cached", 10000, 10000 - X - 1, X)


def test_an_image_requests_probe_stops_at_its_first_image():
    # weg2-20-129: the front probes c.ids[:first_image] (12604 tokens) -- even a
    # full pre-image hit leaves the 5 images and the turns after them to P
    assert d_presence_only_short("d_served_anchor", N, 12544, X)


def test_no_x_no_hold():
    assert not d_presence_only_short("d_served_anchor", N, 0, 0)


# ---- the arrival ------------------------------------------------------------------

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


def _front(src, store_depth, awake="D", state="flipping", flip_dst="P"):
    f = Front("http://p", "http://d", awake, "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = state
    f.admit_d = False
    f._flip_dst = flip_dst
    f.x_exact = True

    async def fake_price(rid, path, payload, text, remainder, est_prompt, multimodal=False):
        return SimpleNamespace(pending=N - CREDIT, n=N, known=True, credit=CREDIT, src=src,
                               store_depth=store_depth, mm=True, mm_cached=True,
                               image_end=21089, d_reach=CREDIT, d_floor=N - 32)

    f._x_exact_price = fake_price
    f._pb_register = lambda *a, **k: None
    f._early_flip_maybe = lambda *a, **k: None
    return f


def _arrive(f):
    async def go():
        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req("/generate", {"text": "a" * 3000})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        out = (list(f.queue), list(f._ready_for_d))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return out

    return asyncio.run(go())


def test_weg2_20_129_is_kept_for_d_not_ps_batch(caplog):
    caplog.set_level(logging.INFO)
    f = _front("d_served_anchor", 0)
    queued, ready = _arrive(f)
    assert queued == [], "never P's batch"
    assert len(ready) == 1
    p = ready[0]
    assert p.d_presence_only and p.d_direct and p.short_kept and p.skip_leg1
    assert not p.p_only, "image-cached: priced as text"
    assert f.counters["short_hold_d"] == 1 and f.counters["short_behind_p"] == 0
    assert any("WEG2 SHORT-HOLD-D rid=" in m and "src=d_served_anchor" in m for m in caplog.messages)
    assert not any("WEG2 SHORT-TO-P rid=" in m for m in caplog.messages)


def test_a_store_credited_short_still_joins_ps_batch(caplog):
    caplog.set_level(logging.INFO)
    f = _front("l3_index", CREDIT)
    queued, ready = _arrive(f)
    assert ready == [] and len(queued) == 1
    assert not queued[0].d_presence_only and not queued[0].skip_leg1
    assert any("WEG2 SHORT-TO-P rid=" in m for m in caplog.messages)


def test_a_d_credit_the_store_backs_still_joins_ps_batch():
    f = _front("d_leg2_cached", N - 500)
    queued, ready = _arrive(f)
    assert ready == [] and len(queued) == 1 and not queued[0].d_presence_only


# ---- the P drain leaves it ----------------------------------------------------------

def _pending(rid, age_s, **kw):
    loop = asyncio.new_event_loop()
    return Pending(rid=rid, path="/generate", payload={}, text="x", t_arrive=time.time() - age_s,
                   fut=loop.create_future(), est_prompt=1000, est_uncached=200, **kw)


def test_the_p_drain_does_not_take_a_d_presence_only_short():
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    held = _pending("held", 9.0, skip_leg1=True, short_kept=True, d_direct=True,
                    d_presence_only=True)
    kept = _pending("kept", 8.0, skip_leg1=True, short_kept=True, d_direct=True)
    f._ready_for_d.extend([held, kept])
    n = f._p_batch_takes_d_prefill()
    assert n == 1
    assert [p.rid for p in f._ready_for_d] == ["held"]
    assert [p.rid for p in f.queue] == ["kept"]
    assert held.d_presence_only and held.short_kept and held.d_direct


def test_a_d_presence_only_short_d_lost_goes_to_p_with_the_flag_cleared():
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    p = _pending("held", 1.0, skip_leg1=True, short_kept=True, d_direct=True, d_presence_only=True)
    f._to_p_batch(p, "test")
    assert not p.d_presence_only and not p.short_kept and not p.d_direct
