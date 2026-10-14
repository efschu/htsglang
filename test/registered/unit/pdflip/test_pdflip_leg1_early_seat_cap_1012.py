"""LEG1-EARLY stays under the SEAT GATE cap (NF int24g, 12.10.).

int24g (boot ..._09235ed28a_1012_185900, front log): 9 of 69 P->D flips drained
0.8-9.4 s (Ep. 90 8320 ms, 110 9411 ms, 130 4619 ms). In all 9 the D->P flip
before had posted more LEG1-EARLY legs (2-5) than the seat gate's cap (1-3,
``_dc_p_cap`` = the free D seats, front 27809/27810: two LEG1-EARLY at cap 1);
the P phase dispatched only ``cap`` of them (``P-PHASE dispatched=1 ... end=leaving``),
the over-cap leg kept prefilling on P (bs=1, store read 1.8-2.8 s) and the next
P->D drain waited for it while 5-6 decodes stood parked.

Pinned (red on c59d8bda8d): with a cap armed the flip posts at most
``cap`` early legs (oldest first, early legs already in flight counted), reads
the cap without popping it (the pop stays in the P phase); without a cap
(PBOUND-FLIP-NOW, fairness, gate off: ``_dc_p_cap`` 0) it posts up to
``p_concurrency`` exactly as before. Both post sites (the D->P begin and the
EARLY-FLIP go) use the one limit. Hermetic, no CUDA.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_dashipc_earlycap", os.path.join(os.path.dirname(__file__), "test_pdflip_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def _pend(rid):
    return F.Pending(rid=rid, path="/v1/chat/completions", payload={"messages": []}, text="x",
                     t_arrive=time.time(), fut=None)


def _dp_flip(n_queue, cap, p_concurrency=6):
    """The real Front.flip("D", "P") with ``n_queue`` waiting requests and the
    seat gate's cap armed to ``cap`` (None = never armed). Returns the front
    and the rids whose leg 1 was posted early."""
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
    if cap is not None:
        f._dc_p_cap = cap

    async def run():
        await f.flip("D", "P")
        await asyncio.sleep(0.05)

    asyncio.run(run())
    return f, posted


def test_cap_one_queue_three_posts_exactly_one_early_leg():
    f, posted = _dp_flip(3, cap=1)
    assert posted == ["pdflip-1-0"], posted                 # the oldest, the one the P phase dispatches
    assert f.counters["leg1_early"] == 1
    assert f.__dict__.get("_dc_p_cap") == 1               # read, not popped: the P phase pops it


def test_cap_below_queue_posts_cap_legs_oldest_first():
    f, posted = _dp_flip(5, cap=3)
    assert posted == ["pdflip-1-0", "pdflip-1-1", "pdflip-1-2"], posted
    assert f.counters["leg1_early"] == 3


def test_cap_above_p_concurrency_is_still_bounded_by_p_concurrency():
    f, posted = _dp_flip(4, cap=5, p_concurrency=2)
    assert len(posted) == 2 and f.counters["leg1_early"] == 2


def test_no_cap_keeps_the_old_limit():
    # PBOUND-FLIP-NOW / fairness / gate off: no cap armed -> p_concurrency, as before
    f, posted = _dp_flip(3, cap=None)
    assert posted == ["pdflip-1-0", "pdflip-1-1", "pdflip-1-2"], posted
    assert f.counters["leg1_early"] == 3
    f, posted = _dp_flip(8, cap=0)                         # a 0 cap is "no cap" (pop(..., 0) or 0)
    assert len(posted) == 6 and f.counters["leg1_early"] == 6


def test_limit_function_counts_the_early_legs_already_in_flight():
    q = [SimpleNamespace(rid="a", _leg1_early=object()), SimpleNamespace(rid="b", _leg1_early=None),
         SimpleNamespace(rid="c", _leg1_early=None)]
    assert F.leg1_early_limit(q, 6, 0) == 6               # no cap: untouched
    assert F.leg1_early_limit(q, 6, None) == 6
    assert F.leg1_early_limit(q, 6, 1) == 0               # the one in flight already fills the cap
    assert F.leg1_early_limit(q, 6, 2) == 1
    assert F.leg1_early_limit(q, 2, 9) == 2               # p_concurrency still bounds
    assert F.leg1_early_limit([], 6, 1) == 1


def test_a_second_flip_does_not_exceed_the_cap_with_a_leftover_early_leg():
    # an early leg of an earlier phase still queued and in flight counts against the cap
    f = _d._front()
    f.rpc = _d._rpc
    f.p_leg1_stall_s = 0.0
    f.p_concurrency = 6
    posted = []

    async def leg1(p):
        posted.append(p.rid)

    f.leg1 = leg1

    async def run():
        old = _pend("pdflip-0-0")
        old._leg1_early = asyncio.ensure_future(asyncio.sleep(0))   # in flight on P since the last phase
        f.queue.extend([old, _pend("pdflip-0-1"), _pend("pdflip-0-2")])
        f._dc_p_cap = 1
        await f.flip("D", "P")
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert posted == [] and f.counters["leg1_early"] == 0, posted


def test_the_limit_is_logged_when_a_cap_is_armed_and_only_then(caplog):
    with caplog.at_level(logging.INFO):
        _dp_flip(3, cap=1)
    line = [r.getMessage() for r in caplog.records if "LEG1-EARLY seat-cap=" in r.getMessage()]
    assert len(line) == 1 and "seat-cap=1 limit=1 p_concurrency=6 queue=3" in line[0], line
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _dp_flip(3, cap=None)
    assert not [r for r in caplog.records if "LEG1-EARLY seat-cap=" in r.getMessage()]


def test_both_post_sites_use_the_one_limit_and_the_p_phase_still_pops_the_cap():
    flip = inspect.getsource(F.Front.flip)
    gate = inspect.getsource(F.Front._early_flip_gate)
    for src in (flip, gate):
        assert "leg1_early_candidates(self.queue, _lim)" in src
        assert "Front._leg1_early_limit(self)" in src
        assert "leg1_early_candidates(self.queue, self.p_concurrency)" not in src
    # the read must not pop: the pop belongs to the P phase's dispatch cap
    lim = inspect.getsource(F.Front._leg1_early_limit)
    assert ".pop(" not in lim and '__dict__.get("_dc_p_cap"' in lim
    assert 'self.__dict__.pop("_dc_p_cap", 0)' in inspect.getsource(F.Front)


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
