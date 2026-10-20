"""LEG1-EARLY posts up to ``p_concurrency`` legs at the D->P flip (NF int24g 12.10., revised 15.10.).

int24g (boot ..._09235ed28a_1012_185900): 9 of 69 P->D flips drained 0.8-9.4 s because the D->P flip had posted
more LEG1-EARLY legs than the DECODE-COLLECT SEAT GATE's cap let the P phase dispatch. The cap is gone
(SEAT-AT-HAND-OFF 15.10., test_pdflip_seat_at_handoff_1015.py: route P needs no D seat, the P phase carries no seat
cap), so the limit is ``p_concurrency`` again and there is no cap to stay under; an early leg the P phase does not
reach (H91 phase cap, an intake stall) is settled by K9 P-DRAIN-EARLY-SETTLE. Pinned: both post sites use the
queue head up to ``p_concurrency``; the P->D flip names its outstanding rids. Hermetic, no CUDA.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_dashipc_earlycap", os.path.join(os.path.dirname(__file__), "test_pdflip_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def _pend(rid):
    return F.Pending(rid=rid, path="/v1/chat/completions", payload={"messages": []}, text="x",
                     t_arrive=time.time(), fut=None)


def _dp_flip(n_queue, p_concurrency=6):
    """The real Front.flip("D", "P") with ``n_queue`` waiting requests. Returns
    the front and the rids whose leg 1 was posted early."""
    f = _d._front()
    f.rpc = _d._rpc
    f.p_leg1_stall_s = 0.0
    f.p_concurrency = p_concurrency
    posted = []

    async def leg1(p):
        posted.append(p.rid)
        await asyncio.sleep(0)

    f.leg1 = leg1
    f.queue.extend(_pend("pdflip-1-%d" % i) for i in range(n_queue))

    async def run():
        await f.flip("D", "P")
        await asyncio.sleep(0.05)

    asyncio.run(run())
    return f, posted


def test_the_limit_is_p_concurrency():
    # no seat cap exists any more: the flip posts the queue head up to p_concurrency
    f, posted = _dp_flip(3)
    assert posted == ["pdflip-1-0", "pdflip-1-1", "pdflip-1-2"], posted
    assert f.counters["leg1_early"] == 3
    f, posted = _dp_flip(8)
    assert len(posted) == 6 and f.counters["leg1_early"] == 6


def test_both_post_sites_use_the_queue_head_up_to_p_concurrency():
    flip = inspect.getsource(F.Front.flip)
    gate = inspect.getsource(F.Front._early_flip_gate)
    for src in (flip, gate):
        assert "leg1_early_candidates(self.queue, self.p_concurrency)" in src
        assert "_leg1_early_limit" not in src and "_dc_p_cap" not in src


def test_p_to_d_flip_names_its_outstanding_rids(caplog):
    f = _d._front()
    f.rpc = _d._rpc
    f.awake = "P"
    f.groups["P"].outstanding["pdflip-88-655"] = object()
    with caplog.at_level(logging.INFO):
        try:
            asyncio.run(asyncio.wait_for(f.flip("P", "D"), 0.5))
        except Exception:  # noqa: BLE001 -- the drain waits for the rid; only the begin lines matter
            pass
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PDFLIP-FLIP begin" in m and "sleep=P wake=D outstanding=1" in m for m in msgs), msgs
    assert any("PDFLIP-FLIP outstanding" in m and "sleep=P rids=['pdflip-88-655']" in m for m in msgs), msgs
