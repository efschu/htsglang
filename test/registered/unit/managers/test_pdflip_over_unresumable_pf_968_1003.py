"""OVER-UNRESUMABLE on the PF ring (27B 673cc89f6a boot 1003_170754, PP1
#968 group stop 17:23:06Z, rid pdflip-24-204): the whole paced-told + PF group
fallback round trip with the follower early read (the 27B production form).

On the metal PP1's early read reached 59392 > told 57041 ("over at=ack"),
PP1 acked told, PP0 admitted at 57041 on every rank -- and PP1's tree held no
recurrent state at 57041 (one node 0..59392, state at its end): 45 s, #968,
W17. Pinned here: a follower whose over-read cannot resume AT told acks its
own read, PP0 answers told=0 for EVERY rank (one uniform plan, cap 0) -- and
a follower that can resume at told keeps the old "over" (no false fallback).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from flliper.srt.managers import pdflip_store_told as m  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402

RID = "aaaa-told"
TOLD = 100_000
OVER = 120_000


def _ring(monkeypatch, resumable_at_told: bool):
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_TOLD") or k in ("FLLIPER_PDFLIP_P_TWIN_DEFER", m.ENV_FOLLOWER_EARLY_READ):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    ring = R.Ring(m, monkeypatch, {RID: TOLD}, {r: {RID: (0.8, 0.6, 0.6)[r]} for r in range(3)})
    over = ring.stages[2]
    over.prompts = {RID: OVER}   # this rank's early read reaches past told

    def probe(scheduler, req, depth):
        # the #928 rule on PP2's tree: KV 0..OVER, recurrent state at OVER only
        if scheduler is not over:
            return None              # no probe on the other ranks (LedgerTree)
        if resumable_at_told or int(depth) >= OVER:
            return min(int(depth), OVER)
        return 0

    monkeypatch.setattr(m._tf, "pp0_admissible", probe)
    return ring


def test_an_unresumable_over_read_falls_back_to_told_zero_on_every_rank(monkeypatch):
    """RED on 673cc89f6a: PP2 settled "over", acked told, PP0 admitted at
    told on every rank -- the plan the metal could not materialise (#968)."""
    ring = _ring(monkeypatch, resumable_at_told=False)
    ring.arrive(RID)
    ring.run(100)
    plans = ring.plans(RID)
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2], plans
    assert plans[0][0][2] == 0, plans          # told 0 on every rank: P recomputes the prefix
    assert ring.stages[0]._pf_fallback_n == 1
    assert ring.stages[2]._pdflip_over_unresumable_n == 1


def test_a_resumable_over_read_is_still_satisfied_at_told(monkeypatch):
    """No false fallback: a rank that holds a state at told keeps "over"."""
    ring = _ring(monkeypatch, resumable_at_told=True)
    ring.arrive(RID)
    ring.run(100)
    plans = ring.plans(RID)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1, plans
    assert plans[0][0][2] == TOLD
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert getattr(ring.stages[2], "_pdflip_over_unresumable_n", 0) == 0
