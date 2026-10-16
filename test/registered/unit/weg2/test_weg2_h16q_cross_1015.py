# SPDX-License-Identifier: Apache-2.0
"""NF-H16Q-INT 1015: the interactions no single block of the h16q integration could see.

Blocks: K9 (P-DRAIN-EARLY-SETTLE, INTAKE-BACKUP-INFLIGHT), B5 (ANCHOR-ACKS), K11 (QSA chunk grain), K10 (SEAT-RELIEF
a/b/c, PW-F, W3-FORMA), presence (bigram probe), on h16p 97e49033c2 (arenafull + A1/A2/E1/E2).

Pinned here (hermetic, the tree's own functions, no CUDA):

  X1  K9 x K10 (a)/(b): the relief epoch's P pool dispatches only RESUME-VIA-P legs; K9's settle afterwards only
      AWAITS legs already posted (it never dispatches a fresh leg 1), so a non-RVP request that was not posted stays
      queued -- and a posted one is left neither on P nor in the queue: the P->D flip begins with nothing outstanding.
  X2  K9 x K10 (c): the relief's cap 1 bounds LEG1-EARLY to 1 (leg1_early_limit reads _dc_p_cap); whatever early leg
      is posted (this flip's, or an older one still in the queue) is settled in the phase -> 0 posted legs left for the
      P->D drain (the 11:52 death shape: phase ended by a cap with early legs running).
  X3  K11 x B5: a B5 floor is a common tree-node depth; with the h16p numbers (87296, 67584, 107584 -- all on the 64
      page grid) every floor is on the QSA grain and the next extend passes K11's host check; an off-grain floor (not
      producible by a page-aligned tree) is stopped BY NAME (QSA-PREFIX-GRAIN with the rid), never by the text-less
      device assert.
  X4  B5 x arenafull A2: the anchor walk reads only the tree's match validators -- none of them (and not the B5 walk)
      reads ``l3_present``, the one flag A2 changed; A2 therefore moves no anchor (tripwire on the source).
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402


class _P:
    """The fields the pool, the K10 pick and K9's settle read."""

    def __init__(self, rid, rvp=False, posted=None):
        self.rid = rid
        self.resume_via_p = rvp
        self.fut = None
        self._leg1_early = posted

    def __repr__(self):
        return f"_P({self.rid})"


def _run(coro):
    # H16Q review A5: bounded -- a mutant that waits forever goes RED, never hangs the lock
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, 5.0))
    finally:
        loop.close()


async def _drain_then_settle(queue, *, rvp_only, max_dispatch=0, settle=True):
    """The P phase's two steps in the order the merged front runs them (_p_drain_pool, then K9's settle)."""
    log = []
    on_p = set()

    async def one(p):
        early = getattr(p, "_leg1_early", None)
        if early is not None:
            p._leg1_early = None
            await early
            log.append(("awaited", p.rid))
        else:
            on_p.add(p.rid)
            await asyncio.sleep(0)
            log.append(("fresh", p.rid))
        on_p.discard(p.rid)
        return p

    done = []
    await front_mod._p_drain_pool(
        queue, 4, one, done.append,
        lambda: (not rvp_only or front_mod.k10_rvp_queued(queue)),
        max_dispatch=max_dispatch,
        pick=((lambda q, pred: front_mod.k10_rvp_pick(q, pred)) if rvp_only else None))
    if settle:
        await front_mod._settle_posted_early_legs(queue, one, done.append)
    return log, done


def _posted(loop_result=None):
    f = asyncio.get_event_loop().create_future()
    f.set_result(loop_result)
    return f


def test_x1_k9_settle_never_dispatches_a_fresh_leg_in_a_k10_rvp_only_phase():
    async def go():
        posted_old = _P("weg2-old-posted", rvp=False, posted=_posted())   # an early leg P is computing anyway
        unposted = _P("weg2-shortx", rvp=False)                         # nothing on P for it
        rvp = _P("weg2-10-51", rvp=True)                                # its D seat is held
        q = collections.deque([unposted, posted_old, rvp])
        log, done = await _drain_then_settle(q, rvp_only=True)
        return log, done, q
    log, done, q = _run(go())
    assert ("fresh", "weg2-10-51") in log
    assert ("fresh", "weg2-shortx") not in log, "K9 must never start a leg K10's rvp-only phase withheld"
    assert ("awaited", "weg2-old-posted") in log, "a posted early leg is settled in the phase"
    assert [p.rid for p in q] == ["weg2-shortx"]
    assert not any(getattr(p, "_leg1_early", None) is not None for p in q), "nothing posted is left for the flip"


def test_x1_without_k9_the_posted_leg_would_be_left_for_the_p_to_d_drain():
    """The counter-case (K9 off): the posted leg stays queued with its early leg -- the 11:52 shape."""
    async def go():
        posted_old = _P("weg2-old-posted", rvp=False, posted=_posted())
        rvp = _P("weg2-10-51", rvp=True)
        q = collections.deque([posted_old, rvp])
        log, _ = await _drain_then_settle(q, rvp_only=True, settle=False)
        return log, q
    log, q = _run(go())
    assert [p.rid for p in q] == ["weg2-old-posted"] and q[0]._leg1_early is not None


def test_x2_k10_c_cap_one_bounds_leg1_early_and_k9_settles_every_posted_leg():
    assert front_mod.leg1_early_limit(collections.deque(), 4, 1) == 1      # the relief's cap 1 (AGE PLAN)
    async def go():
        old = _P("weg2-old", posted=_posted())                           # posted in an earlier flip, still queued
        head = _P("weg2-30-76", posted=_posted())                        # this flip's one early leg
        rest = _P("weg2-30-77")
        q = collections.deque([head, old, rest])
        assert front_mod.leg1_early_limit(q, 4, 1) == 0, "the cap counts the legs already in flight"
        log, _ = await _drain_then_settle(q, rvp_only=False, max_dispatch=1)
        return log, q
    log, q = _run(go())
    assert ("awaited", "weg2-30-76") in log and ("awaited", "weg2-old") in log
    assert ("fresh", "weg2-30-77") not in log, "the cap holds: no fresh dispatch beyond 1"
    assert [p.rid for p in q] == ["weg2-30-77"]
    assert not any(getattr(p, "_leg1_early", None) is not None for p in q)


# ------------------------------------------------------------------------------------------------- X3 K11 x B5

def _open(told, acks, anchors):
    from sglang.srt.managers import weg2_told_fallback as fb
    return fb._Open(told=told, deadline=0.0, acks=dict(acks), anchors={k: tuple(v) for k, v in anchors.items()})


def _fb_batch(prefix, extend, rid):
    return SimpleNamespace(seq_lens_cpu=[prefix + extend], extend_seq_lens_cpu=[extend], rids=[rid])


@pytest.mark.parametrize("told,acks,anchors", [
    (91712, {1: 91712, 2: 87296}, {1: (87296, 65536), 2: (65536,)}),     # mismatch 12:04:24 weg2-3-27
    (117184, {1: 107584, 2: 107584}, {1: (67584,), 2: (67584,)}),         # told_fidelity 12:16:46 weg2-22-70
])
def test_x3_b5_floors_are_on_the_qsa_grain_and_pass_k11(monkeypatch, told, acks, anchors):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa
    from sglang.srt.managers import weg2_told_fallback as fb
    monkeypatch.delenv(fb.ENV_ANCHOR_ACKS, raising=False)  # default on
    depths = fb.common_depths(_open(told, acks, anchors))
    assert depths, "B5 finds a common depth"
    for d in depths:
        assert d % 64 == 0 and d % 4 == 0
        assert qsa.qsa_prefix_grain_violations(_fb_batch(d, 16384, "weg2-x"), 4, 1) == []
        qsa.qsa_check_prefix_grain(_fb_batch(d, 16384, "weg2-x"), 4, 1)


def test_x3_an_off_grain_floor_is_stopped_by_name_not_by_the_device_assert(monkeypatch):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa
    from sglang.srt.managers import weg2_told_fallback as fb
    monkeypatch.delenv(fb.ENV_ANCHOR_ACKS, raising=False)
    d = fb.common_depths(_open(91712, {1: 91712, 2: 87298}, {1: (87298,), 2: ()}))[0]  # not producible by a paged tree
    assert d == 87298
    with pytest.raises(RuntimeError, match=r"QSA-PREFIX-GRAIN rid=weg2-3-27 prefix=87298"):
        qsa.qsa_check_prefix_grain(_fb_batch(d, 4096, "weg2-3-27"), 4, 1)


# ------------------------------------------------------------------------------------------------- X4 B5 x A2

def test_x4_the_b5_anchor_walk_and_every_match_validator_ignore_l3_present():
    from sglang.srt.managers import weg2_told_fallback as fb
    from sglang.srt.mem_cache.unified_cache_components import full_component, mamba_component, swa_component
    srcs = [inspect.getsource(fb._path_anchor_depths), inspect.getsource(fb.ack_anchors),
            inspect.getsource(fb.common_depths)]
    for mod in (full_component, mamba_component, swa_component):
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__ and "create_match_validator" in cls.__dict__:
                srcs.append(inspect.getsource(cls.create_match_validator))
    assert len(srcs) >= 6
    assert not any("l3_present" in s for s in srcs), (
        "an anchor rule now reads l3_present: re-check B5 against arenafull A2 (SGLANG_WEG2_L3_PRESENT_STRICT)")
