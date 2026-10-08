# SPDX-License-Identifier: Apache-2.0
"""Q-692 DUAL D-WEDGE (27B NVFP4 dual, P PP3 + D TP3 at once, no flip).

METAL ``boot_weg2_dkr27bnvfp4dual1mpsleepbar1fs10031727_bc2bd121c0_1003_172802``,
D stand-still 17:59:40-18:01:06:

(a) 17:59:29 ``SEAT-AGE DISPLACE rid_out=pdflip-0-373 older_waiting=pdflip-0-372``
    parks the running decode 373 (pressure park, span retained). The retain
    keeps the KV of its 25 decoded tokens but no GDN state at the span's end:
    the resume depth is P's anchor 94918, ``uncached=26`` > X=1.
(b) ``PDFLIP X-GATE rid=pdflip-0-373 uncached=26 X=1 verdict=W31`` ->
    ``D-HANDBACK-DEFER ... state=begin``: taken for a P hand-back, but P never
    writes this D-owned end; the vote held it in X-DEFER for 94 s, then
    RESUME-VIA-P every 30 s for ever.
(c) the D-park barrier gave every younger arrival ``pdflip_d_park_first`` while
    373 waited parked -> ADMISSION-WEDGE 4-6 queued, 0 running, 83 s.
(d) front, pdflip-0-321: P-INTAKE-STALL (17:56:08) then DUAL P-PAUSE (17:56:50);
    the pause's ``_on_leg1_done`` returned on the stale ``intake_stalled`` and
    the NEXT successful leg 1 consumed ``dual_requeued`` -- never handed to D.

One test per point, red on 36b5a4d3e9, green after; plus "Flip unverändert"
(FLLIPER_PDFLIP_DUAL_LAYOUT unset): no D-OWN-TAIL, no defer bypass, no gate and
no front change.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import scheduler as SC
from flliper.srt.managers import tp_head_congruence as thc
from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.pdflip import d_seats as DS
from flliper.srt.pdflip import dual_handback_defer as HB
from flliper.srt.pdflip import front as F
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

DUAL_D = {"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}
FLIP_D = {"FLLIPER_PDFLIP_GROUP": "D"}
SWITCHES = ("FLLIPER_PDFLIP_GROUP", "FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN",
            "FLLIPER_PDFLIP_P_TRIM_END_ANCHOR", "FLLIPER_PDFLIP_SEAT_ROTATE")

#: pdflip-0-373: prompt 94919, 25 decoded, P's anchor 94918
PROMPT, OUT, ANCHOR = 94919, 25, 94918


def _env(monkeypatch, env):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# -- (1) X gate: D's own tail ------------------------------------------------------

def _gate(x: int = 1, tp_size: int = 3):
    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        ps=SimpleNamespace(tp_size=tp_size),
        tree_cache=SimpleNamespace(cache_controller=SimpleNamespace(
            mem_pool_host=SimpleNamespace(size=10 ** 9))),
    )
    stub.pdflip_uncached_extent = lambda req, head=None: Scheduler.pdflip_uncached_extent(stub, req, head)
    stub._pdflip_host_carry_tokens = lambda: Scheduler._pdflip_host_carry_tokens(stub)
    return stub


def _d_req(rid="pdflip-0-373", out=OUT, local_prefix=ANCHOR, site=None):
    ids = [(i % 50000) + 11 for i in range(PROMPT)]
    output = [7] * out
    r = SimpleNamespace(rid=rid, origin_input_ids=list(ids), full_untruncated_fill_ids=ids + output,
                        output_ids=output, prefix_indices=list(range(local_prefix)),
                        host_hit_length=0, return_logprob=False, input_embeds=None,
                        session_id=None, multimodal_inputs=None, kv_arrival_seq=373)
    if site is not None:
        DS.mark_parked(r, site, now=1.0)
    return r


def _head(rid, group_match: int):
    canonical = thc.canonical_head_rids([rid])
    return thc.build_uniform_head_inputs(
        canonical, thc.build_head_order_payload(canonical, {rid: group_match}), None, True)


def test_1_the_displaced_decode_resumes_its_own_tail_on_dual_d(monkeypatch, caplog):
    """Metal (a)/(b): 373, 25 decoded, group match = P's anchor 94918 -> uncached 26.
    Parent: W31 (-> D-HANDBACK-DEFER 94 s -> RESUME-VIA-P loop). Fixed: admitted."""
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO, logger=SC.logger.name)
    req = _d_req(site=DS.SITE_PRESSURE)
    assert Scheduler._pdflip_x_refuses(_gate(), req, _head(req.rid, ANCHOR)) is False, (
        "D refused its own 26-token tail (25 decoded + the decode input) against X=1 -- "
        "the 94-s X-DEFER and the RESUME-VIA-P loop of pdflip-0-373")
    assert "PDFLIP X-GATE D-OWN-TAIL rid=pdflip-0-373 uncached=26 out=25 X=1 verdict=admit" in caplog.text
    assert "PDFLIP X-GATE rid=pdflip-0-373 uncached=26 X=1 replicated_term=group verdict=admit" in caplog.text


def test_1_danger_directions_of_the_own_tail(monkeypatch):
    _env(monkeypatch, DUAL_D)
    req = _d_req()
    # the allowance is the own tail and nothing more: one token short of the anchor prices
    assert Scheduler._pdflip_x_refuses(_gate(), req, _head(req.rid, ANCHOR - 1)) is True
    # a fresh hand-back (no output) is no own tail: P's tail, W31 as before
    fresh = _d_req(rid="pdflip-0-374", out=0)
    fresh.full_untruncated_fill_ids = list(fresh.origin_input_ids)
    assert Scheduler._pdflip_x_refuses(_gate(), fresh, _head(fresh.rid, PROMPT - 26)) is True
    # the verdict is the group's: two ranks with different local trees, one verdict
    head = _head(req.rid, ANCHOR)
    a = Scheduler._pdflip_x_refuses(_gate(), _d_req(local_prefix=0), head)
    b = Scheduler._pdflip_x_refuses(_gate(), _d_req(local_prefix=PROMPT + OUT - 1), head)
    assert a is b is False


# -- (2) no hand-back defer for D's own end ------------------------------------------

def test_2_no_handback_defer_for_a_d_owned_end(monkeypatch, caplog):
    """Metal (b): ``D-HANDBACK-DEFER n=58 ... tail=26 rid=pdflip-0-373 state=begin``
    right after SEAT-AGE DISPLACE. P never writes that end; no defer starts."""
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.WARNING, logger=HB.logger.name)
    for req in (_d_req(site=DS.SITE_PRESSURE), _d_req(), _d_req(out=0, site=DS.SITE_PRESSURE)):
        assert HB.begin(req, 26) is False, "a D-owned end was deferred as a P hand-back"
        assert getattr(req, HB.MARK_ATTR, None) is None
    assert "state=begin" not in caplog.text
    assert "D-HANDBACK-DEFER tail=26 rid=pdflip-0-373 state=d_own out=25" in caplog.text
    # a genuine P hand-back still defers -- also #248h's capacity park of a fresh one (FLIP site)
    fresh = _d_req(rid="pdflip-0-19", out=0)
    assert HB.begin(fresh, 2444) is True
    cap = _d_req(rid="pdflip-0-20", out=0, site=DS.SITE_FLIP)
    assert HB.begin(cap, 2444) is True


def test_2_an_admitted_mark_votes_no_pending_after_a_later_park(monkeypatch):
    """A hand-back deferred once, admitted, decoded, then displaced: its old mark
    must neither vote pending nor be re-read by the retry."""
    _env(monkeypatch, DUAL_D)
    req = _d_req(rid="pdflip-0-19", out=0)
    assert HB.begin(req, 2444) is True
    HB.note_admit(req)
    req.output_ids = [7] * OUT                       # D decoded, then SEAT-AGE DISPLACE
    assert HB.pending(req, 1e9) is False
    sched = SimpleNamespace(waiting_queue=[req], _prefetch_kvcache=lambda r: pytest.fail("re-read"))
    assert HB.retry(sched) == 0


# -- (3) the D-park barrier: a parked request waiting for a read holds no one back ---

def _gate_for(waiting):
    return DS.admission_gate(waiting, running=[])


@pytest.mark.parametrize("why", ["hb_mark", "x_defer"])
def test_3_a_deferred_park_holds_no_younger_newcomer_back(monkeypatch, caplog, why):
    """Metal (c): 373 parked in D-HANDBACK-DEFER / X-DEFER, pdflip-0-374..379 got
    ``pdflip_d_park_first`` -> ADMISSION-WEDGE 4-6 queued, 0 running, 83 s."""
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO, logger=DS.logger.name)
    parked = _d_req(site=DS.SITE_PRESSURE)
    if why == "hb_mark":
        setattr(parked, HB.MARK_ATTR, {"n": 58, "t0": 0.0, "passes": 0, "tail": 26, "spent": False,
                                       "issued": False, "retry_seen": 0})
    else:
        HB.note_x_defer(parked, True)
    newcomer = SimpleNamespace(rid="pdflip-0-376", kv_arrival_seq=376)
    gate = _gate_for([parked, newcomer])
    assert gate.skip(newcomer, admitted=[]) is None, (
        "a younger arrival waits behind a parked request that waits for a store read "
        "(ADMISSION-WEDGE)")
    assert gate.skip(parked, admitted=[]) is None, "the parked one itself reaches its X gate"
    assert "PDFLIP-D-PARK DEFER-EXEMPT rid=pdflip-0-373" in caplog.text
    # a parked request NOT waiting for a read keeps the barrier (the seat is its)
    plain = _d_req(rid="pdflip-0-375", site=DS.SITE_PRESSURE)
    assert _gate_for([parked, plain, newcomer]).skip(newcomer, admitted=[]) == "pdflip_d_park_first"


def test_3_the_x_defer_verdict_is_noted_for_the_barrier(monkeypatch):
    """The scheduler's X-completion arm writes the GROUP verdict onto the request."""
    _env(monkeypatch, DUAL_D)
    req = _d_req(site=DS.SITE_PRESSURE)
    stub = SimpleNamespace(_pdflip_local_store_read_pending_ms=lambda r: 5,
                           _pdflip_x_store_read_bound_s=lambda r: float("inf"))
    monkeypatch.setattr(thc, "group_store_read_pending_ms", lambda h, rid: 5)
    assert Scheduler._pdflip_x_defers(stub, req, None) is True
    assert HB.defer_exempt(req) is True
    monkeypatch.setattr(thc, "group_store_read_pending_ms", lambda h, rid: None)
    assert Scheduler._pdflip_x_defers(stub, req, None) is False
    assert HB.defer_exempt(req) is False, "a stale defer verdict must not free the barrier"


# -- (4) front: paused after an intake stall, the next leg 1 goes to D ----------------

def _front(dual=True):
    f = F.Front(prefill="http://p", decode="http://d", awake="D" if dual else "P", tag="q692",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                weight_chunks=2, flip_min_work_tokens=1, dual_layout=dual)

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.rpc = rpc
    return f


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _replay_321(dual=True):
    async def run():
        f = _front(dual)
        fut = asyncio.get_running_loop().create_future()
        p = F.Pending("pdflip-0-321", "/generate", {}, "x", time.time(), fut, est_prompt=40000,
                      est_uncached=40000)
        to_d = []

        def on_done(q):        # the real _on_leg1_done's first exits, in the real order
            if q.intake_stalled:
                return
            if f._dual_skip_after_requeue(q):
                return
            to_d.append(q.rid)

        # 17:56:08 leg 1 answered PDFLIP-INTAKE-STALL
        await f._requeue_intake_stalled(p, RuntimeError("PDFLIP-INTAKE-STALL"))
        on_done(p)
        f.queue.popleft()
        # 17:56:50 the next leg 1 is paused (DUAL P-PAUSE)
        p.dual_pause = True
        f._dual_requeue_paused(p)
        on_done(p)
        f.queue.popleft()
        # the leg 1 after it succeeds (the real one(): leg1_done, intake_stalled cleared)
        p.leg1_done = True
        p.intake_stalled = False
        on_done(p)
        return p, to_d

    return _run(run())


def test_4_a_paused_formerly_stalled_request_reaches_d():
    p, to_d = _replay_321()
    assert to_d == ["pdflip-0-321"], "pdflip-0-321 was prefilled and never handed to D"
    assert p.dual_requeued is False


def test_4_the_replay_mirrors_the_real_closure():
    src = inspect.getsource(F.Front.controller)
    a = src.index("def _on_leg1_done(p: Pending) -> None:")
    b = src.index("if p.intake_stalled:", a)
    c = src.index("if self._dual_skip_after_requeue(p):", a)
    assert b < c, "the replay's exit order is the real one"
    assert "p.intake_stalled = False" in src[:a], "the successful leg 1 clears the stall"
    assert "self._dual_requeue_paused(p)" in src[:a]


# -- Flip unverändert (FLLIPER_PDFLIP_DUAL_LAYOUT unset) -------------------------------

def test_flip_unchanged_x_gate_defer_gate(monkeypatch, caplog):
    _env(monkeypatch, FLIP_D)
    caplog.set_level(logging.INFO)
    req = _d_req(site=DS.SITE_PRESSURE)
    assert HB.d_own_tail(req) == 0 and HB.d_owned(req) is False
    assert Scheduler._pdflip_x_refuses(_gate(), req, _head(req.rid, ANCHOR)) is True, "W31 as before"
    assert "D-OWN-TAIL" not in caplog.text
    assert HB.begin(req, 26) is False and getattr(req, HB.MARK_ATTR, None) is None
    assert "state=d_own" not in caplog.text
    HB.note_x_defer(req, True)
    assert not hasattr(req, HB.X_DEFER_ATTR), "no barrier vote off the dual layout"
    setattr(req, HB.X_DEFER_ATTR, True)                 # even a stray mark changes nothing
    newcomer = SimpleNamespace(rid="pdflip-0-376", kv_arrival_seq=376)
    gate = _gate_for([req, newcomer])
    assert gate.skip(newcomer, admitted=[]) == "pdflip_d_park_first"
    assert gate.defer_exempt == frozenset() and gate.newcomers_free is False
    assert "DEFER-EXEMPT" not in caplog.text


def test_flip_unchanged_front():
    async def run():
        f = _front(dual=False)
        fut = asyncio.get_running_loop().create_future()
        p = F.Pending("pdflip-31-31", "/generate", {}, "x", time.time(), fut, est_prompt=6275,
                      est_uncached=6275)
        p.intake_stalled = True
        f._dual_requeue_paused(p)
        return f, p

    f, p = _run(run())
    assert p.intake_stalled is True, "the flip form's intake stall is untouched"
    assert f._dual_skip_after_requeue(p) is False
