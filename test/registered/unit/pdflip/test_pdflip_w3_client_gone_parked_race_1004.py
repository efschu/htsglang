"""W3 CLIENT-GONE PARK RACE (NF y9nf6 191228abc5, boot ...10040638_191228abc5_1004_063851,
front log Z. 1608-1651, 06:45:19-21Z).

D parked five at epoch 8 (``PARK-RUNNING epoch=8 ... rids=['pdflip-0-1', 'pdflip-0-2',
'pdflip-0-5', 'pdflip-0-6', 'pdflip-8-16']``, stamp taken before the RPC at ~06:45:19.80). The
D->P flip's quiesce sent ``/flush_cache`` at 21.321; while it was in flight the client of
pdflip-0-2 hung up (``PDFLIP-CLIENT-GONE rid=pdflip-0-2 state=parked action=abort-d-park`` at
21.545). The H102 parked branch dropped the rid from ``_d_parked`` BEFORE its abort RPC --
but the rid leaves ``D.outstanding`` only when its leg 2 ends (``D-REFILL rid=pdflip-0-2
freed_by=leg2_finished`` at 21.651). The flush answered idle at 21.628, inside that window:
the flip ledger counted pdflip-0-2 as un-parked D work against an idle D ->
``PDFLIP STOP W3 PdFlipDrainWitnessDisagreement -- rank idle, front still holds requests``.
``park_lapse_race`` (Q-699b) could not absorb it: pdflip-0-2 was no longer in ``parked``.
The stop line printed ``sorted(S.outstanding)`` -- all four parked rids -- not the ledger
it had judged, which made the three healthy parks look lost.

Replayed here on the real ``Front._on_client_gone`` / ``Front._flip_ledger`` and the real
W3 helpers, in the metal's order: client gone (abort RPC held in flight) -> the quiesce's
witness -> leg 2 ends. RED on fe2fa47764, GREEN with the fix.
"""

import asyncio
import collections
import inspect
import time
from types import SimpleNamespace

from flliper.srt.pdflip import front as F
from flliper.srt.pdflip.front import Front

PARKED = ["pdflip-0-1", "pdflip-0-2", "pdflip-0-5", "pdflip-0-6"]
GONE = "pdflip-0-2"
# metal: 06:45:19.80, 1.8 s before the witness -- stamped "now" here so the
# part-B lapse clock (PARK_REQUEUE_S) cannot be what the test measures



def _front():
    f = object.__new__(Front)
    f.groups = {
        "D": SimpleNamespace(name="D", outstanding={r: object() for r in PARKED}),
        "P": SimpleNamespace(name="P", outstanding={}),
    }
    f.queue = collections.deque()
    f._ready_for_d = []
    f.counters = collections.Counter()
    f._d_parked = {r: time.time() for r in PARKED}
    f._rvp_inflight = set()
    f.abort_gate = asyncio.Event()
    f.aborts = []

    async def rpc(group, path, body, timeout):
        f.aborts.append((group.name, path, body.get("rid")))
        await f.abort_gate.wait()  # D's /abort_request in flight
        return 200, ""

    f.rpc = rpc
    return f


def _witness(f, idle=True):
    """The quiesce's double witness exactly as the flip path takes it."""
    S = f.groups["D"]
    wl = list(Front._flip_ledger(f, S))
    wv = F.witness_verdict(len(wl), idle)
    race = F.park_lapse_race(S.name, wl, f._d_parked, idle) if wv is not None else []
    return wl, (None if race else wv)


def _leg2_ends(f, rid):
    """The leg-2 handler's ``finally`` (H91 part C): the rid leaves D's ledger
    and the park ledger together."""
    f.groups["D"].outstanding.pop(rid, None)
    f._d_parked.pop(rid, None)


async def _replay():
    f = _front()
    # 21.545: the client of pdflip-0-2 hangs up while its request is parked
    gone = asyncio.ensure_future(Front._on_client_gone(f, GONE, None))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert f.aborts == [("D", "/abort_request", GONE)]  # the abort is in flight
    # 21.628: /flush_cache answers idle -- the quiesce takes its witness now
    during = _witness(f)
    # the abort returns, then 21.651: leg 2 of pdflip-0-2 ends
    f.abort_gate.set()
    await gone
    _leg2_ends(f, GONE)
    after = _witness(f)
    return f, during, after


def test_the_063851_specimen_is_no_w3():
    f, (wl, wv), (wl_after, wv_after) = asyncio.run(_replay())
    assert wv is None, (
        f"W3 inside the client-gone window: ledger={wl} parked={sorted(f._d_parked)} "
        "(the 06:45:21.628 stop)")
    assert wl == []
    # after its leg ended the rid is gone from both ledgers; the other parks ride on
    assert wv_after is None and wl_after == []
    assert sorted(f._d_parked) == ["pdflip-0-1", "pdflip-0-5", "pdflip-0-6"]
    assert GONE not in f.groups["D"].outstanding
    assert f.counters["client_gone_parked"] == 1


def test_a_real_disagreement_still_stops():
    """A rid D is NOT holding parked, with D idle, is still W3 -- the fix only
    keeps a parked rid parked until its leg ends."""
    f = _front()
    f.groups["D"].outstanding["pdflip-9-99"] = object()  # running on D for the front
    wl, wv = _witness(f, idle=True)
    assert wl == ["pdflip-9-99"]
    assert wv == "rank idle, front still holds requests"


def test_the_stop_line_names_the_judged_ledger():
    """The W3 line prints the ledger the witness judged, not every rid the
    front holds (the 063851 line listed four parks as if lost)."""
    src = inspect.getsource(F.Front)
    i = src.index('self.do_stop("W3 PdFlipDrainWitnessDisagreement"')
    stop = src[i:i + 400]
    assert "sorted(_wl)" in stop, stop
