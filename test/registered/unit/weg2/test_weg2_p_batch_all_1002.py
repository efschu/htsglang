"""P-BATCH-ALL (user 02.10. ~07:45Z, strict phase batch): while P is awake,
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


# ---- (1) the 15-52 shape --------------------------------------------------------

def test_weg2_15_52_a_short_arriving_while_p_is_awake_gets_a_p_leg1(caplog):
    caplog.set_level(logging.INFO)
    f = _front("P")
    queued, ready = _arrive(f)
    assert ready == []
    assert len(queued) == 1
    p = queued[0]
    assert p.est_uncached <= 208, "priced on the measured presence (chars/3 here; 207 exact)"
    assert not p.skip_leg1 and not p.short_kept and not p.d_direct, \
        "P is awake: a normal leg 1 on P, never kept for a D prefill"
    assert not any("SHORT-BEHIND-P" in m for m in caplog.messages)
    assert any("WEG2 SHORT-TO-P rid=" in m and "joins P's batch" in m for m in caplog.messages)


def test_a_short_arriving_while_d_flips_to_p_joins_ps_batch():
    f = _front("D", state="flipping", flip_dst="P")
    queued, ready = _arrive(f)
    assert ready == [] and len(queued) == 1
    assert not queued[0].skip_leg1 and not queued[0].short_kept


def test_the_d_phase_keep_is_unchanged():
    # D awake, admission closed (PARK-IMMEDIATE fired): SK still keeps it for
    # D -- a D-phase decision; the next P drain hands it to P (below)
    f = _front("D")
    f.admit_d = False
    queued, _ready = _arrive(f)
    assert len(queued) == 1 and queued[0].short_kept and queued[0].skip_leg1


# ---- (2) queued for D at the D->P flip -> P's batch -----------------------------

def _pending(rid, age_s, **kw):
    loop = asyncio.new_event_loop()
    return Pending(rid=rid, path="/generate", payload={}, text="x", t_arrive=time.time() - age_s,
                   fut=loop.create_future(), est_prompt=1000, est_uncached=200, **kw)


def test_prefill_queued_for_d_at_the_d_to_p_flip_goes_to_p(caplog):
    caplog.set_level(logging.INFO)
    f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    kept = _pending("kept", 9.0, skip_leg1=True, short_kept=True, d_direct=True)   # SHORT-KEPT
    drained = _pending("drained", 7.0, d_direct=True)                            # D-SHORT-DRAIN
    prefilled = _pending("handoff", 8.0, leg1_done=True)                         # P did leg 1
    carrier = _pending("carrier", 6.0, skip_leg1=True, leg1_done=True)           # CARRIER-EXCEEDS
    queued_kept = _pending("qkept", 5.0, skip_leg1=True, short_kept=True)        # D-phase fall-through
    later = _pending("later", 1.0)
    f._ready_for_d.extend([kept, prefilled, drained, carrier])
    f.queue.extend([queued_kept, later])
    n = f._p_batch_takes_d_prefill()
    assert n == 3
    assert [p.rid for p in f._ready_for_d] == ["handoff", "carrier"]
    assert [p.rid for p in f.queue] == ["kept", "drained", "qkept", "later"], "arrival order"
    for p in (kept, drained, queued_kept):
        assert not p.skip_leg1 and not p.short_kept and not p.d_direct and not p.leg1_done
    assert f.counters["p_batch_from_d"] == 3
    assert sum("WEG2 P-BATCH-TAKES rid=" in m for m in caplog.messages) == 3


def test_the_p_drain_sweeps_before_it_dispatches_and_gives_a_kept_short_its_leg1():
    # 27B port: the drain body is the nested `_p_drain_pass` (shared with the
    # DUAL-TP3PP3 pump). The sweep sits on the FLIP form's call only; the dual
    # pump (`self._dual_pump(_p_drain_pass)` + continue) never reaches it.
    src = inspect.getsource(F.Front)
    call = src.index("self._p_batch_takes_d_prefill()\n                await _p_drain_pass()")
    assert src.index("self._dual_pump(_p_drain_pass)") < call
    i = src.index("async def _p_drain_pass() -> int:")
    blk = src[i:i + 20000]
    a = blk.index('if getattr(p, "short_kept", False) and not self.dual_layout:')
    assert a < blk.index("if p.skip_leg1:  # route CARRIER-EXCEEDS")
    assert 'self._to_p_batch(p, "drain")' in blk[a:a + 300]


def test_27b_flip_names_its_destination_for_arrivals():
    # 27B port: `awake` names the source until the flip is done, so the flip
    # records its destination at its begin (p_phase_arrival reads it)
    src = inspect.getsource(F.Front.flip)
    a = src.index('if not self._enter_state("flipping"):')
    assert src.index("self._flip_dst = dst", a) < src.index("self._flip_cushion_open(src, dst)", a)


def test_27b_dual_layout_arrival_is_not_a_p_phase_arrival(caplog):
    # DUAL-TP3PP3 (P and D awake together, no flip): its routing is unchanged
    caplog.set_level(logging.INFO)
    f = _front("P")
    f.dual_layout = True
    _arrive(f)
    assert not any("WEG2 SHORT-TO-P rid=" in m for m in caplog.messages)


# ---- (3) the cut-off ----------------------------------------------------------------

def test_the_cut_off_is_the_begin_of_the_flip_to_d():
    assert F.p_phase_arrival("P", "serving", None) is True
    assert F.p_phase_arrival("D", "flipping", "P") is True
    assert F.p_phase_arrival("P", "flipping", "D") is False, "the flip to D began: D's (no reverse flip)"
    assert F.p_phase_arrival("D", "serving", None) is False
    assert F.p_phase_arrival("D", "serving", "P") is False, "a stale flip_dst outside a flip"
