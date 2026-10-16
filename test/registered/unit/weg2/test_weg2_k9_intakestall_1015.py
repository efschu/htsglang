# SPDX-License-Identifier: Apache-2.0
"""K9 (NF h16r 39fa0976ce, 15.10.2026): the P-INTAKE-STALL / phase-end chain.

(a) DEATH 11:52:21-11:54:11Z (ausfaelle/20261015T115411Z_tot, DEADMAN_FLIP_STALL), precursor 11:26:01-11:26:23Z (FLIP
    STALL 14.7 s, self-resolved): the D->P flip posted leg 1 EARLY for the queued requests; the drain ended before it
    reached them (a P-INTAKE-STALL of weg2-86-234 read by the early-leg callback; at 11:26 the phase cap) while P was
    prefilling them anyway. The P->D flip began with them outstanding (``WEG2-FLIP outstanding ... rids=['weg2-86-231',
    'weg2-86-233', 'weg2-86-237']``) and its drain waited for P's prefills (47.6 s, 51.9 s) under the flip-stall bound
    (24.5 s) -> WEG2-FLIP STALL -> deadman. Fix: the drain settles every queued request whose early leg is posted
    before the phase ends (``front._settle_posted_early_legs``).

(b) DOUBLE FLIP 11:47:11-11:47:25Z (7 in 45 min): weg2-74-213 (163712 tokens) was refused 503 the moment weg2-74-212
    (157632 rows) finished on P, while P's own store backups of that hand-off were still open (``hicache_backup(3)``,
    STALL-CENSUS tracked_evictable=157632 pool_avail=104512). The stall's premise "nothing frees" was false: an open
    backup of the tree now counts as in flight for the intake verdict (WAIT, not stall; ``p_intake.pool_terms``).
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
import types

import pytest

from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import p_intake as PI
from sglang.srt.weg2.intake_stall import INTAKE_FITS, INTAKE_IMPOSSIBLE, INTAKE_WAITS, intake_verdict


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (F.ENV_P_DRAIN_EARLY_SETTLE, PI.ENV_INTAKE_BACKUP_INFLIGHT):
        monkeypatch.delenv(k, raising=False)


class _P:
    """A Pending stand-in: identity matters (the real Pending compares fields)."""

    def __init__(self, rid, early=None):
        self.rid = rid
        self._leg1_early = early
        self.intake_stalled = False
        self.leg1_done = False

    def __eq__(self, other):  # the real Pending.__eq__ compares fields -- equal rids must not collide
        return isinstance(other, _P) and other.rid == self.rid

    __hash__ = object.__hash__


def _drain_11_52(loop):
    """The queue at the end of epoch 87's drain: weg2-86-234 requeued at the head by its stall (no early leg any
    more), 231/233/237 with their early legs posted at the D->P begin and still running on P."""
    futs = {r: loop.create_future() for r in ("weg2-86-231", "weg2-86-233", "weg2-86-237")}
    q = collections.deque([_P("weg2-86-234")] + [_P(r, futs[r]) for r in futs])
    return q, futs


def _one_reader(stall_rids=()):
    async def one(p):
        early = p._leg1_early
        p._leg1_early = None
        res = await early
        if p.rid in stall_rids:
            p.intake_stalled = True
            return p
        p.leg1_done = bool(res)
        return p
    return one


def test_k9_death_shape_old_exit_leaves_legs_outstanding_for_the_flip():
    """What the P->D flip met at 11:52:25Z: three posted early legs still queued -- outstanding for its drain."""
    loop = asyncio.new_event_loop()
    try:
        q, _ = _drain_11_52(loop)
        outstanding = [p.rid for p in q if p._leg1_early is not None]
        assert outstanding == ["weg2-86-231", "weg2-86-233", "weg2-86-237"]
    finally:
        loop.close()


def test_k9_the_drain_settles_the_posted_early_legs_before_the_phase_ends():
    loop = asyncio.new_event_loop()
    try:
        q, futs = _drain_11_52(loop)
        done = []

        async def main():
            async def finish():
                await asyncio.sleep(0.01)
                for f in futs.values():
                    f.set_result(True)
            asyncio.ensure_future(finish())
            return await F._settle_posted_early_legs(q, _one_reader(), done.append)

        n = loop.run_until_complete(main())
        assert n == 3
        assert [p.rid for p in done] == ["weg2-86-231", "weg2-86-233", "weg2-86-237"], "queue order kept"
        assert all(p.leg1_done for p in done)
        assert [p.rid for p in q] == ["weg2-86-234"], "the stalled head stays; nothing of this phase is left on P"
        assert not [p for p in q if p._leg1_early is not None], "the flip begins with no outstanding early leg"
    finally:
        loop.close()


def test_k9_a_settled_leg_that_stalls_is_read_like_the_drain_reads_it():
    loop = asyncio.new_event_loop()
    try:
        q, futs = _drain_11_52(loop)
        for f in futs.values():
            f.set_result(True)
        done = []
        loop.run_until_complete(F._settle_posted_early_legs(q, _one_reader({"weg2-86-233"}), done.append))
        assert [p.rid for p in done if p.intake_stalled] == ["weg2-86-233"]  # the real on_done skips it (requeued)
    finally:
        loop.close()


def test_k9_skip_leaves_a_lane_held_leg_and_identity_not_equality_decides():
    loop = asyncio.new_event_loop()
    try:
        f1, f2 = loop.create_future(), loop.create_future()
        f1.set_result(True)
        f2.set_result(True)
        a, b, c = _P("x", f1), _P("x", f2), _P("held", loop.create_future())
        q = collections.deque([a, c, b])
        done = []
        n = loop.run_until_complete(F._settle_posted_early_legs(q, _one_reader(), done.append,
                                                                 skip=lambda p: p.rid == "held"))
        assert n == 2 and done[0] is a and done[1] is b
        assert list(q) == [c]
    finally:
        loop.close()


def test_k9_nothing_posted_is_a_no_op():
    loop = asyncio.new_event_loop()
    try:
        q = collections.deque([_P("a"), _P("b")])
        assert loop.run_until_complete(F._settle_posted_early_legs(q, _one_reader(), lambda p: None)) == 0
        assert [p.rid for p in q] == ["a", "b"]
    finally:
        loop.close()


def test_k9_switch_default_on_and_off():
    assert F.p_drain_early_settle_enabled()
    assert not F.p_drain_early_settle_enabled({F.ENV_P_DRAIN_EARLY_SETTLE: "0"})


def test_k9_wiring_the_drain_settles_after_the_pool_and_before_the_flip_decision():
    src = inspect.getsource(F.Front)
    i_pool = src.index("passes = await _p_drain_pool(")
    i_settle = src.index("await _settle_posted_early_legs(")
    i_hold = src.index("if _phase_stats.get(\"short_rides\"):", i_settle)
    assert i_pool < i_settle < i_hold
    seg = src[i_pool:i_hold]
    assert "p_drain_early_settle_enabled()" in seg and 'self.state == "serving"' in seg
    assert "self.queue, one, _on_leg1_done" in seg


# ---------------------------------------------------------------------------
# (b) the intake verdict: an open store backup is in flight
# ---------------------------------------------------------------------------


class _Alloc:
    def __init__(self, n):
        self.n = n

    def available_size(self):
        return self.n


def _tree_11_47(backups):
    """PP0 at 11:47:19Z: pool_avail=104512, the 157632 rows of weg2-74-212 not payable (room cap = the free rows),
    ``hicache_backup(3)`` open."""
    t = types.SimpleNamespace(ongoing_write_through={}, ongoing_load_back={},
                              ongoing_backup={i: object() for i in range(backups)},
                              weg2_pp_room_cap=104512)
    t.full_evictable_size = lambda: 157632
    return t


def _verdict(tree):
    free, ev, inflight = PI.pool_terms(tree, _Alloc(104512))
    return intake_verdict(need_tokens=163712, pool_tokens=262144, free_tokens=free, evictable_tokens=ev,
                          inflight=inflight)


def test_k9b_open_backups_make_the_refusal_a_wait_not_the_stall():
    assert _verdict(_tree_11_47(3)) == INTAKE_WAITS


def test_k9b_red_without_the_term_the_stall_of_11_47(monkeypatch):
    monkeypatch.setenv(PI.ENV_INTAKE_BACKUP_INFLIGHT, "0")
    assert _verdict(_tree_11_47(3)) == INTAKE_IMPOSSIBLE


def test_k9b_no_backup_open_still_the_stall():
    """Bounded: once the backups are done, the ordinary verdict (and its hold) decides as before."""
    assert _verdict(_tree_11_47(0)) == INTAKE_IMPOSSIBLE


def test_k9b_fits_and_too_large_are_untouched():
    t = _tree_11_47(3)
    t.weg2_pp_room_cap = 262144
    assert _verdict(t) == INTAKE_FITS
    free, ev, inflight = PI.pool_terms(_tree_11_47(3), _Alloc(104512))
    assert intake_verdict(need_tokens=300000, pool_tokens=262144, free_tokens=free, evictable_tokens=ev,
                          inflight=inflight) == INTAKE_IMPOSSIBLE


def test_k9b_mutant_the_term_dropped_is_caught(monkeypatch):
    orig = PI.pool_terms

    def no_backup_term(tree, alloc):
        f, e, i = orig(tree, alloc)
        return f, e, bool(tree.ongoing_write_through) or bool(tree.ongoing_load_back)

    monkeypatch.setattr(PI, "pool_terms", no_backup_term)
    assert _verdict(_tree_11_47(3)) == INTAKE_IMPOSSIBLE, "the mutant (backups ignored) answers the 11:47 stall"
