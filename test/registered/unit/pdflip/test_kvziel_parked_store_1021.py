"""KVZIEL-1021: the KV stage target counts the PARKED requests; store-backed content is room.

THE METAL (NF rc29 a773b4ebc4, boot ..._1021_022209, D.log).
(1) 02:27:54 the D->P flip park retracted the five running requests (``PDFLIP-PARK-RETRACT-SPLIT
    epoch=0 n=5``, ``PDFLIP-D-PARK RETAINED retained=192129 of 192169`` ...; one tick before
    D-MEM-SHARE read used=656861). The census read used=0 incoming=0, an END event,
    ``pending S-->S0`` (cap 32768) and a KV-COMPACT toward S0 (``verdict=refused want=S0 ...
    STAGES S0..S8``) against the spans the five resumed after the P phase (pdflip-0-28 / -30
    finished 02:28:34 at cache_len 192256). The parked requests are the stage's demand.
(2) Every candidate S0..S8 failed "after 9 evictions here": only L2-backed idle content was
    evictable. On this D the L2 arena is full (``#1427 ARENA-DROP need=2031 freed=0``), the
    finished turns' device nodes are STORE-backed (``l3_present``, host copy spilled) -- the S39
    drop takes such leaves (``S39 L3-DROP FORMA n=4 node=59 tokens=130112`` 7 s before) but
    KV-COMPACT counted them as un-backed. Store-backed content is secured: it is room, handed to
    the S39 drop (TP0 decides at its broadcast, every rank follows), never ``_evict_to_host``.

Danger direction "Ränge uneins": every rank must reach the same stage / the same plan -- the
parked demand from replicated request token counts (mutant: a rank-local span length), the
leave-path of a segment from the AGREED maps (mutant: this rank's own backing).
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import forma_l3_subtree as S39  # noqa: E402

from test_s77_kv_compact_s9_hold_1019 import FULL, Tree, _alloc, _node, _take  # noqa: E402

S0 = 32768
GRID = [S0 * (i + 1) for i in range(8)]   # 32768 .. 262144
P, NP = 4, 32


# ---- (1) the parked demand ------------------------------------------------------------------

def _req(rid, n_in, n_out=0, finished=False, span=None):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out,
                                 finished=lambda: finished,
                                 prefix_indices=torch.zeros(n_in if span is None else span))


def _sched(dsv_, caps, floor):
    sched = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                          speculative_num_draft_tokens=4),
        running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[], chunked_req=None,
        last_batch=None, page_size=64, token_to_kv_pool_allocator=object(),
        pdflip_d_parked=[], _pdflip_group_min_ints=lambda vals: vals)
    setattr(sched, dsv_.CTL_ATTR, False)
    setattr(sched, dsv_.PHASE_ATTR, dsv_.PhaseState(epoch="e1", n=6, cap=6, done=True, stage=7,
                                                    stage_tokens=GRID[7]))
    return sched


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND", "1")
    caps, floor = [], {"page": 0}
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: floor["page"])
    return _sched(dsv, caps, floor), caps, floor


def _five():
    # the park's five (tokens scaled to the desk grid: 5 x ~45k = 228k, S7 holds them)
    return [_req("pdflip-0-28", 46000), _req("pdflip-0-30", 46000), _req("pdflip-0-26", 45000),
            _req("pdflip-0-15", 45000), _req("pdflip-0-10", 46000)]


def _park(sched, reqs):
    """The flip park as ``d_park_runtime.park_running`` leaves it: running empty, the
    retracted requests in ``pdflip_d_parked``."""
    sched.running_batch.reqs = []
    sched.pdflip_d_parked = list(reqs)


def test_metal_02_27_54_a_flip_park_is_no_end_event_to_S0(env):
    """RED on a773b4ebc4: ``pending S-->S0`` (cap 32768) with the parked spans live up to S7."""
    sched, caps, floor = env
    five = _five()
    sched.running_batch.reqs = list(five)
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert ms.stage == 7
    _park(sched, five)
    floor["page"] = 228000 // 64                       # the retained spans hold their pages
    caps.clear()
    dsv.runtime_tick(sched)
    assert ms.pending is None, "pending S%s at a flip park (the parked spans are the demand)" % ms.pending
    assert ms.stage == 7 and ms.counters["stage_down"] == 0
    assert all(c >= GRID[7] for c in caps), caps       # base: cap 32768 below 228000 live


def test_the_target_is_the_smallest_stage_holding_parked_plus_incoming(env):
    sched, caps, floor = env
    sched.running_batch.reqs = [_req("a", 50000)]
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    _park(sched, [_req("a", 50000)])
    floor["page"] = 0                                   # nothing live above any stage
    dsv.runtime_tick(sched)
    air = dsv._air(sched)
    assert ms.stage == ms._smallest_holding(50000 + air) == 1   # S1 holds 50000 + air, not S0
    assert ms.pending is None


def test_parked_counted_once_and_a_finished_one_not_at_all(env):
    sched, _caps, _floor = env
    a, b, c = _req("a", 1000), _req("b", 2000), _req("c", 4000, finished=True)
    q = _req("q", 8000)
    sched.pdflip_d_parked = [a, b, c, q]
    got = dsv._parked_demand(sched, [a], [q])
    assert [r.rid for r in got] == ["b"]


def test_the_parked_marker_names_the_parked_demand(env, caplog):
    sched, _caps, _floor = env
    _park(sched, _five())
    with caplog.at_level(logging.INFO):
        dsv.runtime_tick(sched)
    assert any(dsv.PARKED_MARK in r.getMessage() and "parked=5" in r.getMessage()
               for r in caplog.records)


def _three_rank_run(monkeypatch, mutant=None):
    """Three D ranks: the SAME request lists (replicated), rank-local span lengths differ
    (Form A cuts: TP0 90880 vs workers 90624). Run decode -> park -> P phase -> wake and
    return each rank's decision sequence (stage, pending, last cap)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND", "1")
    if mutant is not None:
        monkeypatch.setattr(dsv, "_req_tokens", mutant)
    out = []
    for rank in range(3):
        caps, floor = [], {"page": 0}
        monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p, caps=caps: caps.append(int(t)) or t)
        monkeypatch.setattr(dsv, "max_live_page", lambda alloc, floor=floor: floor["page"])
        sched = _sched(dsv, caps, floor)
        run = [_req("r%d" % i, 30000, span=30000) for i in range(2)]
        parked = [_req("p%d" % i, 40000, span=40000 - (256 if rank else 0) * (i + 1)) for i in range(4)]
        seq = []
        sched.running_batch.reqs = run + parked
        dsv.runtime_tick(sched)
        ms = getattr(sched, dsv.MEM_SCHED_ATTR)
        seq.append((ms.stage, ms.pending, caps[-1] if caps else None))
        # the park: two keep running (a lane), four park
        sched.running_batch.reqs = list(run)
        sched.pdflip_d_parked = list(parked)
        floor["page"] = 220000 // 64
        dsv.runtime_tick(sched)
        seq.append((ms.stage, ms.pending, caps[-1] if caps else None))
        # the two finish: only the parked hold
        sched.running_batch.reqs = []
        dsv.runtime_tick(sched)
        seq.append((ms.stage, ms.pending, caps[-1] if caps else None))
        out.append(seq)
    return out


def _agree(runs) -> bool:
    return all(r == runs[0] for r in runs[1:])


def test_ranks_agree_on_the_stage_with_the_parked_demand(monkeypatch):
    runs = _three_rank_run(monkeypatch)
    assert _agree(runs), runs


def test_mutant_rank_local_span_length_parts_the_ranks(monkeypatch):
    """The danger direction: the parked demand read from a rank-local quantity (the span a
    rank retained -- its prefix_indices) instead of the request's own tokens. The ranks
    then step apart (one enters the floor collective, another does not: a hang on the metal)
    -- the agreement probe above must see it."""
    def rank_local(r):
        pi = getattr(r, "prefix_indices", None)
        return int(pi.numel()) if pi is not None else 0
    runs = _three_rank_run(monkeypatch, mutant=rank_local)
    assert not _agree(runs), runs


# ---- (2) store-backed content is room (S39 hand-off) ------------------------------------

def _store(n):
    n.component_data[FULL].host_value = None
    n.l3_present = True
    return n


def _world():
    alloc = _alloc(NP, P)
    return alloc, Tree(alloc)


def test_a_store_backed_leaf_is_room_through_the_s39_hand_off():
    alloc, tree = _world()
    _take(alloc, list(range(1, 17)))                       # no free page at or below 16
    x = _store(_node(tree, alloc, [20, 21], P))            # idle, KV in the store only
    sv0 = KC.survey(tree, alloc, P, 16)                    # without the S39 drop: un-backed
    plan0, why0 = KC.plan_for(sv0, 16)
    assert plan0 is None and "un-backed" in why0, why0
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan, why = KC.plan_for(sv, 16)
    assert plan is not None, why
    assert plan.staged and [m.node for m in plan.handoff] == [x]
    assert not plan.evicts and not plan.pairs            # nothing demoted, nothing moved yet


def test_split_a_mixed_chain_is_no_room_pure_chains_split_now_and_handoff():
    """KVPLAN-1021: a single-child chain of L2 and store nodes is ONE segment of mixed kind -- never room
    (its kind boundary is rank-local); a pure L2 branch goes now, a pure store branch is handed to S39."""
    alloc, tree = _world()
    _take(alloc, list(range(1, 17)))
    a = _node(tree, alloc, [17, 18], P)                    # L2-backed parent
    _store(_node(tree, alloc, [19, 20], P, parent=a))      # store-backed child: one mixed segment
    _node(tree, alloc, [21, 22], P)                        # a pure L2 branch
    _store(_node(tree, alloc, [23, 24], P))                # a pure store branch
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    kinds = {tuple(g.pages): g.kind for g in sv.segments}
    assert kinds[(17, 18, 19, 20)] == 3
    ag = KC.Agreement(None)
    segs, _w = KC.choose_evictions(sv, ag, 4)
    st = {tuple(g.pages): s for g, s in KC.split_evictions(segs, ag)}
    assert (17, 18, 19, 20) not in st
    assert st[(21, 22)] == 1 and st[(23, 24)] == 2


def _vote_world(store_here):
    alloc, tree = _world()
    _take(alloc, list(range(1, 17)))
    n = _node(tree, alloc, [20, 21], P)
    if store_here:
        _store(n)
    return alloc, tree


def _plans(worlds, floor=16):
    vecs = []
    for alloc, tree in worlds:
        sv = KC.survey(tree, alloc, P, floor, store_ok=True)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, floor, NP, [floor])))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    return [dsv._kv_compact_plans_from_vote(sv, group, floor, NP, [floor])[0][0] for sv, _v in vecs]


def test_a_content_whose_backing_differs_between_ranks_is_no_room_on_any_rank():
    """KVPLAN-1021: TP0 in the store only, TP1/TP2 L2-backed -- the ranks would take it by different paths;
    every rank answers 'no plan' alike (a uniform refusal, never a plan difference). All-store on every rank:
    one staged plan, one digest."""
    assert _plans([_vote_world(True), _vote_world(False), _vote_world(False)]) == [None, None, None]
    plans = _plans([_vote_world(True), _vote_world(True), _vote_world(True)])
    assert all(p is not None and p.staged and len(p.handoff) == 1 for p in plans), plans
    assert len({KC.plan_digest(5, p) for p in plans}) == 1


def _rc30_rank(tp0):
    """rc30 06:37:26 shape: two store branches A, B and an L2 branch C1 -> C2 whose tail TP0 has not backed
    yet (S108 TURN-END-L2: TP0 unbacked=3, TP1/TP2 1 for the same rid). 10 pages above boundary 16, 8 free below: the gap is 2."""
    alloc, tree = _world()
    _take(alloc, list(range(1, 9)))                        # pages 9..16 free: 8 below the boundary
    _store(_node(tree, alloc, [17, 18, 19], P))            # A (3)
    _store(_node(tree, alloc, [20, 21, 22, 23], P))        # B (4)
    c1 = _node(tree, alloc, [24, 25], P)                   # C1 L2
    _node(tree, alloc, [26], P, parent=c1, backed=not tp0)  # C2: L2 on the workers, un-backed on TP0
    return alloc, tree


def test_rc30_ranks_with_a_rank_local_unbacked_tail_agree_on_the_plan():
    plans = _plans([_rc30_rank(True), _rc30_rank(False), _rc30_rank(False)])
    assert all(p is not None for p in plans), plans
    ds = {KC.plan_digest(4, p) for p in plans}
    assert len(ds) == 1, "refused-plan-diff: the ranks evict other content (rc30 06:37:26)"


def test_mutant_rank_local_structure_in_the_plan_reproduces_the_plan_diff(monkeypatch):
    """Danger direction: the vote ignores the rank-local unsecured subtree (the S112b/KVZIEL form) -- the
    workers best-fit C (agreed size 2), TP0 cannot reach C (its tail is unbacked there) and takes A: the digests
    differ, TP1 = TP2 (the rc30 metal: digest here 1612949070171436656 on TP0, 3713009316671245308 on TP1/TP2)."""
    real = KC.segments

    def no_sub_ok(*a, **k):
        out = real(*a, **k)
        for g in out:
            g.sub_ok = g.evictable
        return out
    monkeypatch.setattr(KC, "segments", no_sub_ok)
    plans = _plans([_rc30_rank(True), _rc30_rank(False), _rc30_rank(False)])
    ds = [KC.plan_digest(4, p) if p is not None else None for p in plans]
    assert ds[0] != ds[1] == ds[2], ds


def _size_rank(tp0):
    """S112c B4 under Form A: TP0's leaf one page deeper (an extra store node below the boundary)."""
    alloc, tree = _world()
    _take(alloc, list(range(1, 14)))                       # 14, 15, 16 free below the boundary
    _store(_node(tree, alloc, [17, 18, 19], P))            # A (3)
    b = _store(_node(tree, alloc, [20, 21, 22, 23], P))    # B (4; 5 on TP0)
    if tp0:
        _store(_node(tree, alloc, [14], P, parent=b))      # gaps: workers 4, TP0 5 -> the group's 5
    return alloc, tree


def test_ranks_agree_with_a_rank_local_segment_size():
    plans = _plans([_size_rank(True), _size_rank(False), _size_rank(False)])
    assert all(p is not None for p in plans), plans
    assert len({KC.plan_digest(4, p) for p in plans}) == 1


def test_mutant_rank_local_size_in_the_plan_parts_the_ranks(monkeypatch):
    """Danger direction 'rang-lokale Größe im Plan': the choice weighs this rank's own segment size instead
    of the group MIN -- TP0's B (5) closes the gap alone, the workers' B (4) needs A too: digests differ."""
    monkeypatch.setattr(KC.Agreement, "size", lambda self, seg: seg.size)
    plans = _plans([_size_rank(True), _size_rank(False), _size_rank(False)])
    ds = [KC.plan_digest(4, p) if p is not None else None for p in plans]
    assert ds[0] != ds[1] == ds[2], ds


def test_queue_for_room_queues_on_tp0_only(monkeypatch):
    S39._reset_for_tests()
    alloc, tree = _world()
    x = _store(_node(tree, alloc, [20], P))
    monkeypatch.setattr(S39, "_r12", lambda: types.SimpleNamespace(role=lambda: "worker"))
    assert S39.queue_for_room(tree, [x]) == 0 and not S39._ST.queue
    monkeypatch.setattr(S39, "_r12", lambda: types.SimpleNamespace(role=lambda: "host"))
    assert S39.queue_for_room(tree, [x]) == 1 and id(x) in S39._ST.queue
    assert S39.queue_for_room(tree, [x]) == 1 and len(S39._ST.queue) == 1   # idempotent
    S39._reset_for_tests()


def _tick_world(monkeypatch, store=True):
    alloc, tree = _world()
    _take(alloc, list(range(1, 17)))
    n = _node(tree, alloc, [20, 21], P)
    if store:
        _store(n)
    else:
        n.component_data[FULL].host_value = None           # un-backed (no L2, no store)
    from test_s77_kv_compact_s9_hold_1019 import _Copier

    cop = _Copier(NP, P)
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=P, tree_cache=tree)
    monkeypatch.setattr(dsv, "kv_page_buffers", lambda s_: ([], None))
    monkeypatch.setattr(dsv.KvPageCopier, "from_buffers", classmethod(lambda cls, b, p: cop))
    toks = (16 * P, 24 * P, 28 * P, 32 * P)
    ms = types.SimpleNamespace(stage=3, stage_tokens=toks,
                               _smallest_holding=lambda f: next(j for j, t in enumerate(toks) if t >= f))
    return alloc, tree, n, sched, ms


def test_the_tick_stages_a_store_backed_room_and_does_not_memo_it(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    alloc, tree, n, sched, ms = _tick_world(monkeypatch)
    got = []
    monkeypatch.setattr(S39, "queue_for_room", lambda t, nodes, why="": got.extend(nodes) or len(nodes))
    fr = {}
    for _ in range(3):                                       # the snapshot lands in a later pass
        if dsv._kv_compact_for_shrink(sched, ms, fr, 0, True, 22 * P) is not None:
            break
        if fr.get("compact_staged"):
            break
    assert got == [n], got
    assert fr.get("compact_staged") == 1 and "compact_refused" not in fr
    assert n.component_data[FULL].value is not None         # TP0's broadcast drops it, not this pass
    assert any("verdict=staged" in m and "handoff_nodes=1" in m for m in caplog.messages), caplog.messages


def test_a_refusal_names_what_holds_the_pages(monkeypatch, caplog):
    """The instrument: an un-backed node (no L2, no store) above the boundary, no room -- the
    refusal prints HOLDERS with its class (rc29 02:27:54 printed no page and no witness)."""
    caplog.set_level(logging.INFO)
    alloc, tree, n, sched, ms = _tick_world(monkeypatch, store=False)
    fr = {}
    for _ in range(3):
        dsv._kv_compact_for_shrink(sched, ms, fr, 0, True, 22 * P)
        if fr.get("compact_refused"):
            break
    hold = [m for m in caplog.messages if " HOLDERS " in m]
    assert hold and "unbacked=2" in hold[0] and "free_below=0" in hold[0], hold
