"""S84: KV-COMPACT refused on NF every time -- "allocator._free_listeners armed".

THE METAL (NF rc19 10e9b2b492, boot 1019_230333, D.log): from 23:07:41 every end
event read ``PDFLIP D-MEM-SCHED KV-COMPACT n=1 verdict=refused want=S4 stage=S5
floor=164480 end=yes local=0 -- allocator._free_listeners armed (a slot-lifetime
mirror, e.g. DFlash DraftKVSlotMapper)`` -- 17 logged lines per rank TP0/TP1/TP2,
``verdict=moved`` 0. The listener was NOT DFlash (speculative_algorithm='EAGLE', the
MTP draft; every "dflash" in the D.log is the refusal text itself) and NOT the
backing-relief rung (``KV-BACKING ... Backing relief is NOT registered`` on all three
ranks, 23:06:24): it was the KV STAGE CAP itself. ``kv_stage_boot_cap`` ->
``_engage_kv_cap`` -> ``KvRowCap.engage`` registered its free/clear hooks at the boot
(``#251c KV-STAGE ... cap_pages=512/8192``, 23:06:11, every rank). So wherever a
stage form exists, S77-A could never move a page. The S77-A tests stub
``_engage_kv_cap`` (``tick_env``), which is why none of them saw it.

WHAT MUST HOLD.
(1) A listener declares its role (``allocator.base.mark_listener``); the guard reads
    the role, not the existence: the KvRowCap CAP FILTER passes, a DFlash SLOT MIRROR
    refuses, a listener without a role refuses (fail-closed), alias listeners refuse.
(2) Under a cap filter the move respects the cap: no destination above the lowest
    engaged cap (even with an id above it standing in a free list), the filter's
    withheld ids are neither live nor free, a freed source above the cap is withheld
    again (through ``free()``), and a ``clear()`` after the move books nothing twice
    (#485).
(3) The rc19 sequence with the REAL stage cap armed at the boot: RED on 10e9b2b492
    (refused), GREEN: S8, the turn's pages moved below the boundary.
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers.kv_backing_relief import KvRowCap  # noqa: E402
from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

from test_s77_kv_compact_s9_hold_1019 import (  # noqa: E402,F401
    FULL, PG, S8_PAGE, Tree, _Copier, _alloc, _draft_pool, _guard_sched, _metal_env, _names,
    _node, _take, _to_s9, floor_env, tick_env,
)
from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER, _req  # noqa: E402

try:  # the S84 marker API (absent on the base revision: these tests are red there)
    from flliper.srt.mem_cache.allocator.base import (  # noqa: E402
        LISTENER_ROLE_CAP_FILTER, LISTENER_ROLE_SLOT_MIRROR, mark_listener,
    )
except ImportError:  # pragma: no cover - base revision
    LISTENER_ROLE_CAP_FILTER = LISTENER_ROLE_SLOT_MIRROR = None

    def mark_listener(fn, role, owner=None):
        raise AssertionError("allocator.base.mark_listener missing (pre-S84)")

_REAL_ENGAGE = dsv._engage_kv_cap          # tick_env stubs it; the metal arms the real cap


def _real_cap(monkeypatch, dsv_, caps):
    def engage(alloc, tokens, page):
        caps.append(int(tokens))
        return _REAL_ENGAGE(alloc, tokens, page)
    monkeypatch.setattr(dsv_, "_engage_kv_cap", engage)


def _armed_metal(floor_env, monkeypatch):
    """The rc19 D: the stage cap engaged at the boot (S0), its hooks registered."""
    dsv_, sched, caps, alloc, tree, born = _metal_env(floor_env, monkeypatch)
    _real_cap(monkeypatch, dsv_, caps)
    pages = dsv_.kv_stage_boot_cap(alloc, PG)
    assert pages == FLOOR_LADDER[0] // PG
    assert alloc._free_listeners, "the stage cap registered no listener"
    return dsv_, sched, caps, alloc, tree, born


def _end_event(dsv_, sched, alloc, tree):
    ms = _to_s9(dsv_, sched)
    _take(alloc, list(range(1, 5000)))                               # the running decodes, below
    turn = _node(tree, alloc, list(range(5800, 6000)) + list(range(6145, 6330)), PG, last=7)
    v_old = turn.component_data[FULL].value.clone()
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000), _req("pdflip-2-71", 120_000)]
    st = dsv_.runtime_tick(sched)
    return ms, st, turn, v_old


# ---- (3) the rc19 metal ------------------------------------------------------------------

def test_rc19_nf_the_stage_caps_own_listener_no_longer_refuses_the_compaction(
        floor_env, monkeypatch, caplog):
    """RED on 10e9b2b492: ``verdict=refused ... allocator._free_listeners armed``.
    GREEN: S8, the retained turn below the boundary with its bytes, the sources above
    the new cap withheld by the cap (not in a free list), nothing booked twice."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, born = _armed_metal(floor_env, monkeypatch)
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    before = born.t[v_old].clone()
    assert st.changed and ms.stage == 8, st.reason
    assert any(dsv_.KV_COMPACT_MARK in m and "verdict=moved" in m for m in caplog.messages), \
        [m for m in caplog.messages if dsv_.KV_COMPACT_MARK in m]
    v = turn.component_data[FULL].value
    assert int((v // PG).max()) <= S8_PAGE
    assert torch.equal(born.t[v], before), "the moved prefix lost its bytes"
    cap = alloc._pdflip_kv_stage_cap
    assert cap.engaged and int(cap.cap) == S8_PAGE and caps[-1] == FLOOR_LADDER[8]
    held = cap._withheld.tolist()
    assert len(held) == len(set(held)), "an id is withheld twice"
    assert set(range(6145, 6330)) <= set(held), "a freed source above the cap is not withheld"
    assert int(alloc.free_pages.max()) <= S8_PAGE


def test_a_dflash_slot_mirror_still_refuses_and_says_so(floor_env, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, born = _armed_metal(floor_env, monkeypatch)
    mapper = types.SimpleNamespace(on_global_free=lambda idx: None, on_global_clear=lambda: None)
    alloc.register_free_listener(
        mark_listener(lambda idx: mapper.on_global_free(idx), LISTENER_ROLE_SLOT_MIRROR, mapper),
        mark_listener(lambda: mapper.on_global_clear(), LISTENER_ROLE_SLOT_MIRROR, mapper))
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert not st.changed and ms.stage == 9
    assert torch.equal(turn.component_data[FULL].value, v_old), "a page moved under a mirror"
    refused = [m for m in caplog.messages if dsv_.KV_COMPACT_MARK in m and "verdict=refused" in m]
    assert refused and "slot-lifetime mirror is armed" in refused[0], refused


@pytest.mark.parametrize("relief_cap", [8192, 8100])
def test_a_backing_relief_cap_filter_moves_or_names_its_withheld_pages(
        floor_env, monkeypatch, caplog, relief_cap):
    """The relief rung's own KvRowCap (a second cap filter, not the stage cap) armed:
    with nothing withheld above the boundary the group moves (destinations below both
    caps); with ids withheld above it (8101.. > S8) the floor counts them -- the plan
    refuses BY NAME and moves nothing, instead of moving for a floor that cannot drop."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, born = _armed_metal(floor_env, monkeypatch)
    ms = _to_s9(dsv_, sched)
    _take(alloc, list(range(1, 5000)))
    turn = _node(tree, alloc, list(range(6145, 6200)), PG, last=7)
    v0 = turn.component_data[FULL].value.clone()
    relief = KvRowCap(alloc)
    relief.engage(relief_cap)
    sched.running_batch.reqs = [_req("pdflip-2-70", 150_000), _req("pdflip-2-71", 120_000)]
    st = dsv_.runtime_tick(sched)
    lines = [m for m in caplog.messages if dsv_.KV_COMPACT_MARK in m and "verdict=" in m]
    if relief_cap >= alloc.num_pages:
        assert relief.withheld == 0 and len(alloc._free_listeners) == 2
        assert st.changed and ms.stage == 8, st.reason
        assert int((turn.component_data[FULL].value // PG).max()) <= S8_PAGE
        assert int(alloc.free_pages.max()) <= S8_PAGE
    else:
        assert relief.withheld > 0
        assert not st.changed and ms.stage == 9
        assert torch.equal(turn.component_data[FULL].value, v0), "pages moved for a floor that cannot drop"
        assert lines and "verdict=refused" in lines[-1] and "backing relief" in lines[-1], lines


# ---- (1) the guard -------------------------------------------------------------------------

def test_guard_passes_the_stage_caps_filter_and_refuses_the_rest():
    s, alloc = _guard_sched(_draft_pool())
    KvRowCap(alloc).engage(8)
    assert _names(s)[1] is None, "the stage cap's id-set filter refused the move"

    s, alloc = _guard_sched(_draft_pool())
    alloc.register_free_listener(mark_listener(lambda idx: None, LISTENER_ROLE_SLOT_MIRROR),
                                 mark_listener(lambda: None, LISTENER_ROLE_SLOT_MIRROR))
    why = _names(s)[1]
    assert why and "_free_listeners" in why and "mirror" in why

    s, alloc = _guard_sched(_draft_pool())
    alloc.register_free_listener(lambda idx: None)                    # no role: unknown
    why = _names(s)[1]
    assert why and "declares no role" in why

    s, alloc = _guard_sched(_draft_pool())                            # half-declared
    alloc.register_free_listener(mark_listener(lambda idx: None, LISTENER_ROLE_CAP_FILTER),
                                 lambda: None)
    assert "declares no role" in (_names(s)[1] or "")

    s, alloc = _guard_sched(_draft_pool())
    KvRowCap(alloc).engage(8)
    alloc.register_alias_listener(lambda src, dst: None)
    assert "_alias_listeners" in (_names(s)[1] or "")


def test_the_dflash_worker_registers_its_mapper_as_a_slot_mirror():
    import inspect

    from flliper.srt.speculative import dflash_worker_v2 as W

    src = inspect.getsource(W)
    i = src.index("register_free_listener(")
    assert "LISTENER_ROLE_SLOT_MIRROR" in src[i:i + 400], "the DFlash mapper registers without a role"


# ---- (2) the cap under a move --------------------------------------------------------------

P, NP, B = 4, 32, 16


def _cap_world(cap_page):
    alloc = _alloc(NP, P)
    tree = Tree(alloc)
    cop = _Copier(NP, P)
    _take(alloc, list(range(1, 9)))                                   # running requests below
    return alloc, tree, cop


def test_no_destination_above_the_cap_even_with_an_id_above_it_in_a_free_list():
    """Danger direction 'an id above the cap is handed out': page 15 (> cap 12) comes
    back into the free list past the cap's hook; the plan must not take it."""
    alloc, tree, cop = _cap_world(12)
    _take(alloc, [9, 10, 11])                                         # below the cap: only 12 free
    _node(tree, alloc, [20, 21], P, backed=False)                     # un-backed, above B
    _take(alloc, [15])                                                # live at the engage
    cap = alloc._pdflip_kv_stage_cap = KvRowCap(alloc)                # the stage cap
    cap.engage(12)
    alloc.free_pages = torch.cat([alloc.free_pages, torch.tensor([15])])   # past the hook
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None or all(d <= 12 for _m, _s, dd in plan.moves for d in dd), plan.moves
    assert plan is None and "no free page below" in why


def test_freed_sources_above_the_cap_are_withheld_and_a_clear_books_nothing_twice():
    """The sources go back through free(): the filter withholds those above the cap
    (none reaches a free list); a clear() afterwards recomputes, it does not add (#485)."""
    alloc, tree, cop = _cap_world(B)
    hot = _node(tree, alloc, [20, 21, 22], P, backed=False)
    cap = alloc._pdflip_kv_stage_cap = KvRowCap(alloc)                # the stage cap
    cap.engage(B)
    before = cop.t[hot.component_data[FULL].value].clone()
    sv = KC.survey(tree, alloc, P, B)
    assert 20 in sv.live_above and all(f <= B for f in sv.free)
    plan, why = KC.plan_for(sv, B)
    assert plan is not None and not plan.evicts, why
    KC.execute(tree, alloc, plan, cop)
    v = hot.component_data[FULL].value
    assert int((v // P).max()) <= B and torch.equal(cop.t[v], before)
    held = cap._withheld.tolist()
    assert {20, 21, 22} <= set(held) and len(held) == len(set(held))
    assert int(alloc.free_pages.max()) <= B
    got = alloc.alloc(alloc.available_size())
    assert got is not None and int((got // P).max()) <= B, "an id above the cap was handed out"
    alloc.clear()
    held = cap._withheld.tolist()
    assert len(held) == len(set(held)) == NP - B, "the clear booked withheld ids twice (#485)"
    assert int(alloc.free_pages.max()) <= B
    assert alloc.available_size() // P + cap.withheld == NP


@pytest.mark.parametrize("role", ["cap", "mirror"])
def test_the_rank_local_answer_enters_the_group_min(floor_env, monkeypatch, role):
    """The guard's answer is this rank's VOTE: with a mirror the rank still enters the
    S77-A candidate MIN (its flags all 0, so no rank moves); with only the cap filter
    its flags carry the clearable stages and the group moves."""
    dsv_, sched, caps, alloc, tree, born = _armed_metal(floor_env, monkeypatch)
    if role == "mirror":
        alloc.register_free_listener(mark_listener(lambda idx: None, LISTENER_ROLE_SLOT_MIRROR),
                                     mark_listener(lambda: None, LISTENER_ROLE_SLOT_MIRROR))
    votes = []

    def gmin(vals):
        votes.append(list(vals))
        return list(vals)
    sched._pdflip_group_min_ints = gmin
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    flags = [v for v in votes if len(v) >= 2 and len(v) % 2 == 0 and set(v) <= {0, 1}]
    assert flags, "the compaction never entered the group MIN"
    if role == "mirror":
        assert not st.changed and ms.stage == 9 and not any(flags[-1])
        assert torch.equal(turn.component_data[FULL].value, v_old)
    else:
        assert st.changed and ms.stage == 8, st.reason
        assert any(any(f) for f in flags)


# ---- review A1 (nf-review-s84-1019): the two cap edges the first test set left open ----------

def test_review_a1_the_id_right_above_the_cap_is_no_destination():
    """X1 (off-by-one): cap 12 -- page 13 (cap+1) back in a free list past the hook is
    above the cap as much as 15 is; the plan must not take it."""
    alloc, tree, cop = _cap_world(12)
    _take(alloc, [9, 10, 11])                                   # below the cap only 12 is free
    _node(tree, alloc, [20, 21], P, backed=False)               # un-backed, above B, 2 pages
    _take(alloc, [13])                                          # live at the engage
    cap = alloc._pdflip_kv_stage_cap = KvRowCap(alloc)
    cap.engage(12)
    alloc.free_pages = torch.cat([alloc.free_pages, torch.tensor([13])])   # past the hook
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None or all(d <= 12 for _m, _s, dd in plan.moves for d in dd), plan.moves
    assert plan is None and "no free page below" in why


def test_review_a1_a_lower_cap_filter_bounds_the_destinations_after_the_stage_caps_release():
    """X4 / the release() bypass: a cap filter BELOW the stage cap (relief at 12) and the
    stage cap's release() (a stage that rises) hands its withheld ids straight into the free
    list -- past the relief hook. They are free, withheld by nobody, above 12: no
    destination may come from them (the lowest engaged cap bounds the plan)."""
    alloc, tree, cop = _cap_world(12)
    _take(alloc, [9, 10, 11])                                   # below 12 only 12 is free
    _node(tree, alloc, [20, 21], P, backed=False)
    stage = alloc._pdflip_kv_stage_cap = KvRowCap(alloc)
    stage.engage(14)                                            # withholds every free id above 14
    relief = KvRowCap(alloc)
    relief.engage(12)                                           # a listener-registered cap filter: 13, 14
    assert {15, 16} <= set(stage._withheld.tolist()) and set(relief._withheld.tolist()) == {13, 14}
    stage.release()                                             # 15.. back, the relief hook bypassed
    assert {15, 16} <= set(alloc.free_pages.tolist())
    sv = KC.survey(tree, alloc, P, B)
    assert sv.held_above == [] and all(f <= 12 for f in sv.free), sv.free
    plan, why = KC.plan_for(sv, B)
    assert plan is None or all(d <= 12 for _m, _s, dd in plan.moves for d in dd), plan.moves
    assert plan is None and "no free page below" in why
