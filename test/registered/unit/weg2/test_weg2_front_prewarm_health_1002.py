# SPDX-License-Identifier: Apache-2.0
"""y8f CHECK (desk item 010-y8f, boot-start halt): /health must not announce a
serving front while the boot's tokenizer prewarm still holds the route decision.

NF boot y8c (front log boot_weg2_dkrnfint4h6felbar1dauer10021929_71da6e387c_
1002_192959.front.log, evidence dir /spinning/docker-acceptance/nf/evidence/):

  19:33:06,135  WEG2-FRONT up tag=... awake=D  (line 449 -- front accepts)
  19:33:06,958  WEG2 SESSION rid=weg2-0-1     (line 459 -- first arrival)
  19:33:08Z     WEG2-LAUNCH R1 READY group=front port=30030 after 6.0 s (line 473)
  19:33:19,777  WEG2 FRONT-PREWARM done after 13.8 s state=ready load_s=13.5 (line 506)
  19:33:19,777  WEG2 X-EXACT-HOLD rid=weg2-0-1 waited_ms=12818 state=ready (line 507)

The launcher's readiness gate is ``wait_ready`` (launcher.py:5967): it polls
``GET /health`` and calls READY on the first 200 (launcher.py:5981). The front's
``handle_health`` judged ONLY the groups' health and ``state != "STOP"``
(front.py:8389 facts path, front.py:8403 old path) -- it never consulted
``_reported_state()`` (front.py:5865), the very function the 02.10. y7y fix made
/weg2/state use to say ``warming`` during this load (state_dict, front.py:8428).
So the boot declared serving at 19:33:08 while the prewarm ran to 19:33:19.777,
and the first request stalled 12818 ms behind the BOOT-START HOLD.

The invariant this check enforces (the item's two allowed fixes are the two
branches): the front must NEVER simultaneously
  (a) announce readiness on /health, AND
  (b) hold a route decision behind the still-running tokenizer load.
A fix that gates readiness reports NOT-200 while ``_reported_state()`` says
"warming" (truthful readiness); a fix that unblocks the prewarm path lets the
route decision go without waiting for the load. The guards below pin what the
fix must NOT change: readiness after the load, readiness with the prewarm
switch off, and readiness during a flip (the gate is "warming", not "load
unfinished", or every flip would read unhealthy).
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import front_health as FH

P_SID, D_SID = 4711, 4712


def _front(state="serving", loading=True):
    """A front with healthy groups (fresh poller facts, so /health judges
    nothing but the front itself) mid-boot-prewarm when ``loading``."""
    f = object.__new__(F.Front)
    f.groups = {"P": F.Group("P", "http://p", P_SID), "D": F.Group("D", "http://d", D_SID)}
    now = time.time()
    for g in f.groups.values():
        g.health_facts = FH.GroupFacts(True, True, 0, None, now)
    f.state, f.awake, f.epoch, f.stop = state, "D", 0, None
    f.t0 = now - 60
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f._batch_gate = None
    # boot-prewarm facts, exactly what _reported_state() reads (front.py:5871):
    f.x_exact = True
    f.__dict__["_x_exact_t0"] = time.monotonic()
    if loading:
        f._x_exact_ready_event()          # created, NOT set: the load runs
    else:
        f._x_exact_ready_event().set()    # the load ended
    # a tokenizer stack that is still loading: what _x_exact_await_tokenizer
    # (front.py:5877) holds a route decision behind.
    f.ftok = collections.namedtuple("Tok", "state")("loading")
    return f


class _Resp200:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session200:
    """Old live-probe path: both groups answer 200 (healthy groups)."""

    def get(self, url, timeout=None):
        return _Resp200()


def _body(resp):
    return json.loads(resp.body)


# --------------------------------------------------------------------------
# THE check: red on the base, green with the fix (either allowed branch).
# --------------------------------------------------------------------------

def _prewarm_ready_or_unheld(f):
    """(status, resp). A front that is truthful about warming returns NOT-200
    here; a front whose prewarm cannot block requests returns 200 with a route
    decision that does not wait for the load."""
    async def go():
        resp = await f.handle_health(None)
        if resp.status != 200:
            return resp.status, resp
        # The fix chose "prewarm cannot block requests": the route decision
        # must not wait for the ready event. A removed hold passes too.
        await_fn = getattr(f, "_x_exact_await_tokenizer", None)
        if await_fn is not None:
            # 2 s bound vs. the 30 s hold bound (X_EXACT_HOLD_MAX_S): on the
            # base this raises TimeoutError = the stall y8c priced at 12818 ms.
            await asyncio.wait_for(await_fn("y8f-check-rid"), 2.0)
        return resp.status, resp

    return asyncio.run(go())


def test_health_does_not_announce_serving_during_the_boot_prewarm_facts_path():
    """y8c facts path (SGLANG_WEG2_FRONT_HEALTH_FACTS default on, as in y8c):
    healthy groups + state serving + the load not ended -> NOT ready."""
    f = _front()
    status, resp = _prewarm_ready_or_unheld(f)
    if status != 200:
        # truthful readiness: the body must name the state /weg2/state calls it
        assert status == 503, f"expected 503 while warming, got {status}"
        assert _body(resp)["state"] == "warming", _body(resp)["state"]


def test_health_does_not_announce_serving_during_the_boot_prewarm_old_path():
    """The same invariant on the SGLANG_WEG2_FRONT_HEALTH_FACTS=0 path
    (front.py:8395): handle_health dispatches to it; a fix at the dispatch
    covers both, a facts-only fix leaves this red -- same invariant, both
    doors the launcher's wait_ready can knock."""
    f = _front()
    f.session = _Session200()
    with mock.patch.dict(os.environ, {FH.ENV: "0"}):
        status, resp = _prewarm_ready_or_unheld(f)
    if status != 200:
        assert status == 503, f"expected 503 while warming, got {status}"
        assert _body(resp)["state"] == "warming", _body(resp)["state"]


# --------------------------------------------------------------------------
# Guards: green on the base AND green with the fix -- what the fix must keep.
# --------------------------------------------------------------------------

def test_health_is_ready_once_the_prewarm_ended():
    """After the load (ready event set) healthy groups -> 200. A readiness
    gate that never opens is not a fix."""
    f = _front(loading=False)

    async def go():
        return await f.handle_health(None)

    resp = asyncio.run(go())
    assert resp.status == 200
    assert _body(resp)["state"] == "serving"


def test_health_is_ready_with_the_prewarm_switch_off():
    """SGLANG_WEG2_FRONT_TOKENIZER_PREWARM=0: /weg2/state says serving at once
    (test_weg2_front_tokenizer_prewarm_1002.py), so /health must too -- the
    readiness gate rides the same switch, it does not add a new hold."""
    f = _front()
    with envs.SGLANG_WEG2_FRONT_TOKENIZER_PREWARM.override(False):
        resp = asyncio.run(f.handle_health(None))
    assert resp.status == 200


def test_a_flip_while_the_load_runs_still_reads_healthy():
    """The gate is "warming", not "load unfinished": during a flip /health
    kept answering (the health poller and monitors ride it); flipping +
    healthy groups -> 200 even with the load unfinished."""
    f = _front(state="flipping")

    async def go():
        return await f.handle_health(None)

    resp = asyncio.run(go())
    assert resp.status == 200
    assert _body(resp)["state"] == "flipping"


def test_stop_state_still_refuses_health_during_the_load():
    f = _front(state="STOP")

    async def go():
        return await f.handle_health(None)

    resp = asyncio.run(go())
    assert resp.status == 503
