"""Review of KVD-1024 (d198e88c31): two findings, red on the author's head by design.

R1 (MUSS): ``Front._asr_older_blocked`` (the AGE PLAN's eligibility test for an OLDER request that needs P) still prices
the younger decodes by their SEAT (``{s.rid: s.tokens}``), not by ``_asr.victim_release`` -- the third victim-price
site of the front. On the 18-188/18-189 specimen the older P candidate looks "displaceable into D" by the seat 54133,
so a younger D waiter is held back, while the guaranteed release (36053 + 401) does not cover the deficit 38313.

R2 (SOLL, starvation direction): two victims sharing a prefix ONLY between themselves -- each counts the shared rows as
held by another, so the sum of the individual bounds never reaches what displacing both frees; D's verdict says
"nobody leaves" although the two youngest together cover the older one.
"""
from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402

from .test_pdflip_kvd_1024 import (  # noqa: E402
    FRONT_DEFICIT, FRONT_SEAT, HEAD, RESERVE, SEAT, SHARED, VICTIM,
    _age_front, _basis, _env, _lock, _req, _root, _tree, _unode, _verdict_sched,
)

YOUNG = "pdflip-18-190"


def test_r1_older_p_candidate_is_priced_by_the_guaranteed_release_not_the_seat():
    now = time.time()
    f = _age_front(now)
    st = f._asr_st()
    st["waiters"] = {YOUNG: now - 1.0}
    q = SimpleNamespace(rid=HEAD, t_arrive=now - 30.0, est_prompt=100000, est_uncached=100000, payload=None)
    f._asr_live_p_cands = lambda: [q]

    async def _kv(est, mx, rid, unc):
        st["last_kv_basis"] = _basis({VICTIM: SEAT - SHARED})
        return False, "kv", FRONT_DEFICIT

    f._arrival_seat_kv = _kv
    # the guaranteed release 36053 + 401 = 36454 < 38313: the older P candidate is BLOCKED -> the younger may act
    assert asyncio.run(Front._asr_older_blocked(f, YOUNG, now - 1.0, True)) is True


def test_r2_two_victims_sharing_a_prefix_only_between_them_cover_jointly():
    root = _root()
    shared = _unode(root, 10000)
    a = _unode(shared, 2000)
    b = _unode(shared, 2000)
    _lock(a, root)
    _lock(b, root)                       # shared: lock 2 -- held by the two victims ONLY
    v1 = _req("pdflip-18-189", 12000, a)
    v2 = _req("pdflip-18-190", 12000, b)
    sched = _verdict_sched([], [_req(HEAD, 10000)], _tree(root), pool=0)
    view = {"rid": HEAD, "price": 10000, "budget": 0, "reserve": {}, "pool": 0, "pass": None}
    # displacing both frees 10000 + 2000 + 2000 = 14000 >= 10001: the two youngest are enough
    assert dpr.kv_displace_would_fit(sched, HEAD, [v1, v2], view=view) is True
