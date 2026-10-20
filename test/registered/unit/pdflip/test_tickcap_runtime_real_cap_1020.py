"""TICKCAP (20.10.): the D-MEM-SCHED runtime tick with the REAL KV stage cap.

Eight test files replace ``_engage_kv_cap`` with a recorder (``tick_env`` and lambdas:
d_mem_sched_0929, cohort, gate_price, reserve_agree, unmerged_extend, floor_recheck,
boot_phase_q701, cand3_1536). No runtime-tick test drove the real ``KvRowCap`` with its
free/clear listener -- which is how the S77-A defect (a compaction refused by the cap's
own listener on every NF boot, and a destination above the cap on rc19) went through.

Here: the stage cap is engaged at the boot by ``kv_stage_boot_cap`` (as on NF), the real
paged allocator hands out pages, the real ``max_live_page`` reads the floor, and every
stage transition of ``runtime_tick`` runs on them:

* grow (S1 -> higher), the cap follows the mapped stage;
* GROW-ROOM refused (S77-B group vote 0): no rank moves, the cap stays;
* pending shrink (a live page above the wanted stage): the cap goes to the PENDING stage;
* cap lift (the queue head does not fit below the pending cap): the cap goes back to the
  mapped stage, and down again when the demand is gone;
* KV-COMPACT (S77-A/S84): the retained turn moves below the boundary, the cap follows.

After every tick one invariant over the allocator and the cap (``_check``): the cap is
where the machine says (pending unless lifted, else the stage), no free id above it,
free / withheld / live partition the id space with no duplicate, and an allocation of
everything available never hands out an id above the cap.

Only what needs a GPU is faked: no SeatVram controller (``apply_stage`` maps nothing),
one CPU byte buffer stands in for the born KV tensors (the compaction copies it).
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER, _req  # noqa: E402
from test_s77_kv_compact_s9_hold_1019 import FULL, PG, _node  # noqa: E402
from test_s84_kv_compact_listener_1019 import (  # noqa: E402,F401
    _armed_metal, floor_env, tick_env,
)


# ---- the invariant ---------------------------------------------------------------------------

def _ids(t):
    return [] if t is None else [int(x) for x in t.reshape(-1).tolist()]


def _free(alloc):
    out = []
    for name in ("free_pages", "release_pages"):
        out += _ids(getattr(alloc, name, None))
    return out


def _expected_cap_pages(dsv_, sched):
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    lifted = bool(getattr(ms, "_cap_lifted", False))
    want = ms.pending if (ms.pending is not None and not lifted) else ms.stage
    return ms.stage_tokens[want] // PG, want


def _check(dsv_, sched, alloc, where):
    """The cap where the machine says, no free id above it, a clean partition, and an
    allocation of everything available stays below the cap (then given back)."""
    cap = alloc._pdflip_kv_stage_cap
    n = int(alloc.num_pages)
    want_pages, want = _expected_cap_pages(dsv_, sched)
    free = _free(alloc)
    held = _ids(cap._withheld) if cap._withheld is not None else []
    if want_pages >= n:
        assert not cap.engaged, "%s: the top stage keeps a cap (%s)" % (where, cap.cap)
        assert held == [], "%s: the top stage withholds %d ids" % (where, len(held))
    else:
        assert cap.engaged and int(cap.cap) == want_pages, \
            "%s: cap %s, the machine says S%d = %d pages" % (where, cap.cap, want, want_pages)
        assert not free or max(free) <= want_pages, \
            "%s: free id %d above the cap %d" % (where, max(free), want_pages)
        assert not held or min(held) > want_pages, "%s: a withheld id below the cap" % where
    assert len(free) == len(set(free)), "%s: an id twice in the free lists" % where
    assert len(held) == len(set(held)), "%s: an id withheld twice" % where
    assert not (set(free) & set(held)), "%s: an id both free and withheld" % where
    assert all(1 <= p <= n for p in free + held), "%s: an id outside the pool" % where
    live = n - len(free) - len(held)
    assert live >= 0, "%s: free + withheld exceed the pool" % where
    # everything the allocator would hand out now stays at or below the cap
    avail = int(alloc.available_size())
    if avail:
        got = alloc.alloc(avail)
        assert got is not None, "%s: available_size %d not allocatable" % (where, avail)
        if want_pages < n:
            assert int((got // PG).max()) <= want_pages, "%s: an id above the cap handed out" % where
        alloc.free(got)
    return live


def _running(alloc, rid, tokens, out=0):
    """A running request of ``tokens`` (+``out``) whose pages the real allocator hands out."""
    need = -(-(tokens + out) // PG) * PG
    got = alloc.alloc(need)
    assert got is not None, "the allocator cannot place %s (%d tokens)" % (rid, need)
    return _req(rid, tokens, out), got


def _grow_to(dsv_, sched, alloc, stage_target, reqs):
    """Admit requests until the machine stands at ``stage_target`` (checked every tick)."""
    for _ in range(20):
        st = dsv_.runtime_tick(sched)
        _check(dsv_, sched, alloc, "grow tick")
        ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
        if ms.stage >= stage_target:
            return ms
    raise AssertionError("never reached S%d (at S%d)" % (stage_target, ms.stage))


def _metal(floor_env, monkeypatch):
    """The NF D: the stage cap engaged at the boot (S0, ``kv_stage_boot_cap``), then the wake
    that took the phase's stage engages the cap there (``on_wake`` -> ``_engage_kv_cap(kv_alloc,
    form.tokens[stage])``; floor_env's phase stands at S1 after its wake)."""
    dsv_, sched, caps, alloc, tree, born = _armed_metal(floor_env, monkeypatch)
    cap = alloc._pdflip_kv_stage_cap
    assert cap.engaged and int(cap.cap) == FLOOR_LADDER[0] // PG          # S0 at the boot
    phase = getattr(sched, dsv_.PHASE_ATTR)
    dsv_._engage_kv_cap(alloc, FLOOR_LADDER[int(phase.stage)], PG)        # the wake's stage cap
    assert int(cap.cap) == FLOOR_LADDER[int(phase.stage)] // PG
    return dsv_, sched, caps, alloc, tree, born


# ---- (1) boot + grow --------------------------------------------------------------------------

def test_boot_cap_then_every_grow_moves_the_cap_with_the_stage(floor_env, monkeypatch):
    dsv_, sched, caps, alloc, tree, born = _metal(floor_env, monkeypatch)
    cap = alloc._pdflip_kv_stage_cap
    got = None
    stage_seen = []
    for k, tokens in enumerate((60_000, 120_000, 200_000, 300_000, 450_000)):
        # one request that keeps growing (its decode / a longer turn): the stage follows
        sched.running_batch.reqs = [_req("g", tokens)]
        dsv_.runtime_tick(sched)
        _check(dsv_, sched, alloc, "grow %d" % k)
        ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
        stage_seen.append(ms.stage)
        if got is not None:
            alloc.free(got)
        # the demand is placeable under the cap the tick set, and never above it
        got = alloc.alloc(-(-tokens // PG) * PG)
        assert got is not None, "S%d cannot place a %d-token request" % (ms.stage, tokens)
        if cap.engaged:
            assert int((got // PG).max()) <= int(cap.cap)
    assert stage_seen == sorted(stage_seen) and stage_seen[-1] == 9, stage_seen
    assert len(set(stage_seen)) >= 4, stage_seen                       # several grows, not one jump
    assert not cap.engaged, "the top stage still caps"
    alloc.free(got)
    _check(dsv_, sched, alloc, "after the free")


# ---- (2) GROW-ROOM refused ----------------------------------------------------------------------

def test_a_group_refused_grow_keeps_the_cap_and_the_next_room_grows(floor_env, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    dsv_, sched, caps, alloc, tree, born = _metal(floor_env, monkeypatch)
    sched.running_batch.reqs = [_req("a", 30_000)]
    dsv_.runtime_tick(sched)
    _check(dsv_, sched, alloc, "start")
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    before = ms.stage
    real = dsv_._grow_room_refused
    peer = {"short": True}

    def vote(sched_, ctl, n, stage):
        gmin = sched_._pdflip_group_min_ints
        if peer["short"]:
            sched_._pdflip_group_min_ints = lambda vals: [0] * len(vals)   # a peer has no room
        try:
            return real(sched_, ctl, n, stage)
        finally:
            sched_._pdflip_group_min_ints = gmin
    monkeypatch.setattr(dsv_, "_grow_room_refused", vote)
    sched.running_batch.reqs = [_req("a", 30_000), _req("b", 200_000)]
    for i in range(2):
        st = dsv_.runtime_tick(sched)
        assert not st.changed and ms.stage == before, (i, st.reason)
        _check(dsv_, sched, alloc, "refused %d" % i)
    assert ms.counters["stage_up_refused_room"] == 2
    assert any(dsv_.GROW_ROOM_MARK in m for m in caplog.messages)
    peer["short"] = False
    st = dsv_.runtime_tick(sched)
    assert st.changed and ms.stage > before
    _check(dsv_, sched, alloc, "grown")
    got = alloc.alloc(-(-200_000 // PG) * PG)
    assert got is not None, "the grown stage cannot place the request it grew for"


# ---- (3) pending shrink -----------------------------------------------------------------------

def _pending_state(dsv_, sched, alloc):
    """S5 with a running request; it finishes but its pages stay live (a holder outside the
    tree: no compaction), so the end-event shrink stays PENDING."""
    sched.running_batch.reqs = [_req("big", 150_000)]
    dsv_.runtime_tick(sched)
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    assert ms.stage >= 4, ms.stage
    low = alloc.alloc(64 * PG)                                        # 64 low pages, given back below:
    hi = alloc.alloc(-(-150_000 // PG) * PG)                          # big's pages, up to the S-cap
    assert low is not None and hi is not None
    alloc.free(low)                                                   # room below the pending cap
    _check(dsv_, sched, alloc, "big admitted")
    stage_hi = ms.stage
    sched.running_batch.reqs = [_req("small", 20_000)]                 # big finished, pages held
    st = dsv_.runtime_tick(sched)
    assert not st.changed and ms.stage == stage_hi and ms.pending is not None, st.reason
    return ms, hi, stage_hi


def test_a_pending_shrink_caps_at_the_pending_stage_and_drops_when_the_floor_drains(
        floor_env, monkeypatch):
    dsv_, sched, caps, alloc, tree, born = _metal(floor_env, monkeypatch)
    ms, hi, stage_hi = _pending_state(dsv_, sched, alloc)
    live = _check(dsv_, sched, alloc, "pending")
    assert int(alloc._pdflip_kv_stage_cap.cap) == ms.stage_tokens[ms.pending] // PG
    above = sorted({int(p) for p in (hi // PG).tolist() if int(p) > alloc._pdflip_kv_stage_cap.cap})
    assert above, "the held pages do not reach above the pending cap"
    assert live >= len(above)
    # the holder lets go: its ids above the pending cap are withheld at once (the free hook)
    alloc.free(hi)
    held = set(_ids(alloc._pdflip_kv_stage_cap._withheld))
    assert set(above) <= held, "a freed id above the pending cap reached a free list"
    _check(dsv_, sched, alloc, "holder freed")
    st = dsv_.runtime_tick(sched)                                      # the floor drained
    assert st.changed and ms.stage < stage_hi and ms.pending is None, st.reason
    _check(dsv_, sched, alloc, "pending taken")


# ---- (4) cap lift -----------------------------------------------------------------------------

def test_the_cap_lifts_to_the_mapped_stage_for_the_queue_head_and_comes_down_again(
        floor_env, monkeypatch):
    dsv_, sched, caps, alloc, tree, born = _metal(floor_env, monkeypatch)
    ms, hi, stage_hi = _pending_state(dsv_, sched, alloc)
    _check(dsv_, sched, alloc, "pending")
    assert not getattr(ms, "_cap_lifted", False)
    pend_pages = ms.stage_tokens[ms.pending] // PG
    # fill the room below the pending cap, then a queue head that does not fit below it
    below = [p for p in _free(alloc) if p <= pend_pages]
    fill = torch.cat([torch.arange(p * PG, p * PG + PG) for p in below])
    keep = torch.tensor(below, dtype=torch.int64)
    alloc.free_pages = alloc.free_pages[~torch.isin(alloc.free_pages, keep)]   # no room below it now
    sched.waiting_queue = [_req("head", 8_000)]
    st = dsv_.runtime_tick(sched)
    assert getattr(ms, "_cap_lifted", False), st.reason
    _check(dsv_, sched, alloc, "lifted")
    assert int(alloc._pdflip_kv_stage_cap.cap) == ms.stage_tokens[ms.stage] // PG > pend_pages
    head = alloc.alloc(-(-8_000 // PG) * PG)
    assert head is not None, "the lifted cap does not place the queue head"
    assert int((head // PG).max()) <= ms.stage_tokens[ms.stage] // PG
    # the head is admitted (running now), nothing queued: the lift ends, the pending cap is back
    sched.waiting_queue = []
    sched.running_batch.reqs = [_req("small", 20_000)]
    alloc.free(head)
    alloc.free(fill)
    st = dsv_.runtime_tick(sched)
    assert not getattr(ms, "_cap_lifted", False)
    _check(dsv_, sched, alloc, "lift ended")


# ---- (5) KV-COMPACT, then a grow that hands the withheld sources back --------------------------

def test_compaction_under_the_real_cap_then_a_grow_hands_the_sources_back_once(
        floor_env, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, born = _metal(floor_env, monkeypatch)
    sched.running_batch.reqs = [_req("pdflip-2-66", 98_221, 61), _req("pdflip-2-68", 86_604, 51),
                                _req("pdflip-2-70", 150_000), _req("pdflip-2-71", 120_000)]
    dsv_.runtime_tick(sched)
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    assert ms.stage == 9
    _check(dsv_, sched, alloc, "S9")
    run = alloc.alloc(4999 * PG)                                       # the running decodes, low ids
    assert run is not None
    s8 = FLOOR_LADDER[8] // PG
    free_hi = sorted(p for p in _free(alloc) if p > s8)[:120]
    keep = torch.tensor(free_hi, dtype=torch.int64)
    hit = torch.isin(alloc.free_pages, keep)
    alloc.free_pages = alloc.free_pages[~hit]                          # the retained turn's pages
    turn = _node(tree, alloc, [], PG, last=7)
    turn.component_data[FULL].value = torch.cat([torch.arange(p * PG, p * PG + PG) for p in free_hi])
    before = born.t[turn.component_data[FULL].value].clone()
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000), _req("pdflip-2-71", 120_000)]
    st = dsv_.runtime_tick(sched)
    assert st.changed and ms.stage == 8, st.reason
    assert any(dsv_.KV_COMPACT_MARK in m and "verdict=moved" in m for m in caplog.messages)
    v = turn.component_data[FULL].value
    assert int((v // PG).max()) <= s8 and torch.equal(born.t[v], before)
    _check(dsv_, sched, alloc, "compacted")
    held = set(_ids(alloc._pdflip_kv_stage_cap._withheld))
    assert set(free_hi) <= held, "a moved source above S8 is not withheld"
    # the stage rises again (a long request): the cap releases, every withheld id back ONCE
    sched.running_batch.reqs = sched.running_batch.reqs + [_req("long", 260_000)]
    st = dsv_.runtime_tick(sched)
    assert ms.stage == 9 and not alloc._pdflip_kv_stage_cap.engaged
    _check(dsv_, sched, alloc, "regrown")
    assert set(free_hi) <= set(_free(alloc)), "a source withheld under S8 did not come back at S9"


# ---- (6) GROW-ROOM refused out of a PENDING shrink -------------------------------------------

def test_a_grow_refused_out_of_a_pending_shrink_puts_the_cap_back_on_the_stage(
        floor_env, monkeypatch):
    """The grow clears ``pending`` in the machine; the group refuses it (S77-B), the stage
    stays -- so the cap must leave the PENDING stage for the mapped one (the refusal's own
    re-engage at ``tokens[before]``), else the allocator sits at the old pending cap with no
    pending shrink left to lift it (the rc12z30y2 starvation class)."""
    dsv_, sched, caps, alloc, tree, born = _metal(floor_env, monkeypatch)
    ms, hi, stage_hi = _pending_state(dsv_, sched, alloc)
    _check(dsv_, sched, alloc, "pending")
    assert not getattr(ms, "_cap_lifted", False)
    pend_pages = int(alloc._pdflip_kv_stage_cap.cap)
    assert pend_pages == ms.stage_tokens[ms.pending] // PG
    real = dsv_._grow_room_refused

    def vote(sched_, ctl, n, stage):
        gmin = sched_._pdflip_group_min_ints
        sched_._pdflip_group_min_ints = lambda vals: [0] * len(vals)       # a peer has no room
        try:
            return real(sched_, ctl, n, stage)
        finally:
            sched_._pdflip_group_min_ints = gmin
    monkeypatch.setattr(dsv_, "_grow_room_refused", vote)
    sched.running_batch.reqs = [_req("small", 20_000), _req("next", 300_000)]   # needs a higher stage
    st = dsv_.runtime_tick(sched)
    assert not st.changed and ms.stage == stage_hi and ms.pending is None, st.reason
    assert ms.counters["stage_up_refused_room"] == 1
    _check(dsv_, sched, alloc, "refused out of pending")
    assert int(alloc._pdflip_kv_stage_cap.cap) == ms.stage_tokens[stage_hi] // PG > pend_pages
