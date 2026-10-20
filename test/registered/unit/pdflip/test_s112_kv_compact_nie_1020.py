"""S112: KV-COMPACT on NF rc27 never moved -- the three refusals of 20.10. and their roots.

THE METAL (NF rc27 893d09fa3f, boot ..._1020_145331, D.log). KV-COMPACT ``verdict=moved``
0 times; every logged refusal is one of three kinds:

(b) ``page 2561 (tokens from 163904) blocks on a rank`` (n=4..5, 10..16; page 2875 n=1..2,
    6..9). WITNESS on ALL THREE ranks: ``tree(node=11 locks=[1, 0, 0] wt=False lb=False
    backed=False pinned=True), request(rid=pdflip-0-14 running)`` -- the FULL path lock of
    the RUNNING request's matched prefix (``_req_inc_lock_ref(req.last_node)``); n=1:
    ``request(rid=pdflip-0-2 running)`` with no tree node -- the running request's own
    decode tail. pdflip-0-14 held it 15:00:15 .. 15:02:31 (136 s), and the next turn of
    the session locks it again at once. Lock, not write-through, not load-back.
(a) ``no free page below 1536 for un-backed node 17 (9 pages)`` (n=3, 14:59:16, used=0):
    the tail of the just finished turn is un-backed and must stay on the device, so its
    whole backed chain could not be evicted either ("leaf first"), and nothing evicted
    the backed idle nodes BELOW the boundary that are the only room.
(c) ``page 3996 (tokens from 255744) is live on some ranks and free on others`` (n=64,
    15:16:55): WITNESS TP0 ``vote=4, free_pages``, TP1/TP2 ``tree(node=189 ...
    backed=False)`` -- the finish of pdflip-4-73 inserted at ``anchor accepted at
    depth=90880`` on TP0 and ``90624`` on the workers (``MAMBA-HOST-RESUME`` 719 on TP0,
    420 per worker in the boot): the Form A allocators are NO replicas. Real, not the
    sampling time: the page stayed live on the workers until ``#1469 EVICT node=189``
    at 15:18:02.

THE FIX (S112): moves are RANK-LOCAL (each rank its own pages onto its own room: a move
changes no tree, only page ids), the group decides the stage and the evictions (R11);
a lock the running decode requests and the in-flight write-throughs explain is movable
(their ``req_to_token`` rows and ``prefix_indices`` follow), never evictable; evictions
are cut-independent UNITS of agreed backed idle device leaves, chosen to close the
group's gap (MAX of the local gaps) and taken on either side of the boundary.
"""
from __future__ import annotations

import logging
import os
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

from test_s77_kv_compact_s9_hold_1019 import (  # noqa: E402,F401
    CD, FULL, PG, Node, Tree, _alloc, _Copier, _node, _take, _tokens, floor_env, tick_env,
)
from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER  # noqa: E402
from test_s98_kv_compact_form_a_1020 import (  # noqa: E402
    CUT, S, TOP, _ThreeRankGroup, _tp0, _tp1, _tp2,
)

P, NP, B = 4, 32, 16


def _world():
    alloc = _alloc(NP, P)
    return alloc, Tree(alloc), _Copier(NP, P)


def _req(rid, pool_idx, last_node, tokens, prefix=None):
    return types.SimpleNamespace(rid=rid, req_pool_idx=pool_idx, last_node=last_node,
                                 kv_allocated_len=int(tokens.numel()), origin_input_ids=[0] * int(tokens.numel()),
                                 output_ids=[], prefix_indices=prefix)


def _rtt(rows, width):
    t = torch.zeros(len(rows), width, dtype=torch.int32)
    for i, r in enumerate(rows):
        t[i, :r.numel()] = r.to(torch.int32)
    return t


def _lock_path(node, k=1):
    x = node
    while x is not None and x.parent is not None:
        x.component_data[FULL].lock_ref += k
        x = x.parent


# ---- (b) a running request's lock is explained: its pages move with its row ------------

def _running_world(lock_extra=0):
    alloc, tree, cop = _world()
    x = _node(tree, alloc, [20, 21], P, backed=False)            # node 11 of the metal: unbacked
    _take(alloc, [25])                                            # the decode tail, no tree node
    _lock_path(x, 1 + lock_extra)
    row = torch.cat([x.component_data[FULL].value, _tokens([25], P)])
    req = _req("pdflip-0-14", 0, x, row, prefix=x.component_data[FULL].value.clone())
    rtt = _rtt([row], 64)
    pool = types.SimpleNamespace(req_to_token=rtt)
    sched = types.SimpleNamespace(req_to_token_pool=pool, running_batch=types.SimpleNamespace(reqs=[req]),
                                  waiting_queue=[], chunked_req=None, last_batch=None)
    return alloc, tree, cop, x, req, rtt, sched


def test_rc27_b_the_running_requests_locked_prefix_and_tail_move_with_its_row():
    """RED on 893d09fa3f: ``page 20 pinned`` / ``outside the tree``. GREEN: the plan moves
    the locked node and the tail; node value, request row and prefix tensor name the new
    pages, the bytes came along, the sources are free; nothing evicted."""
    alloc, tree, cop, x, req, rtt, sched = _running_world()
    sv = KC.survey(tree, alloc, P, B, running=[req], req_to_token=rtt)
    assert sv.owner[20].move_only and sv.owner[25].node is None and sv.run_pages == 3
    plan, why = KC.plan_for(sv, B)
    assert plan is not None and not plan.evicts, why
    before = cop.t[rtt[0, :3 * P].long()].clone()
    res = KC.execute(tree, alloc, plan, cop, remap_refs=dsv._kv_compact_remap_requests(sched, P, sv))
    assert res.moved_pages == 3 and res.remapped_rows == 2          # the row and prefix_indices
    row = rtt[0, :3 * P].long()
    assert int((row // P).max()) <= B
    assert torch.equal(cop.t[row], before), "the running request's KV lost its bytes"
    assert torch.equal(row[:2 * P], x.component_data[FULL].value.long())
    assert torch.equal(req.prefix_indices.long(), x.component_data[FULL].value.long())
    sv2 = KC.survey(tree, alloc, P, B, running=[req], req_to_token=rtt)
    assert sv2.live_above == [] and {20, 21, 25} <= set(sv2.free)


@pytest.mark.parametrize("who", ["second_lock", "no_running_request"])
def test_an_unexplained_lock_stays_a_pin(who):
    """A lock the running decode requests do not explain (a chunked request, an unmerged
    extend, a session, an admission pin) keeps the node a pin -- nothing moves."""
    alloc, tree, cop, x, req, rtt, sched = _running_world(lock_extra=1 if who == "second_lock" else 0)
    running = [req] if who == "second_lock" else []
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B, running=running, req_to_token=rtt), B)
    assert plan is None and ("pinned" in why or "outside the tree" in why), why


def test_a_running_requests_page_is_never_evicted():
    """No room below: the running request's (backed!) prefix is movable only -- the plan
    refuses instead of evicting what a running row names."""
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, B + 1)))
    x = _node(tree, alloc, [20, 21], P, backed=True)
    _lock_path(x)
    req = _req("r", 0, x, x.component_data[FULL].value)
    rtt = _rtt([x.component_data[FULL].value], 64)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B, running=[req], req_to_token=rtt), B)
    assert plan is None and "no evictable unit" in why, why


# ---- write-through in flight (S108 puts one there at every turn end) -----------------------

def _wt_world(queue=(), copier_inflight=None):
    alloc, tree, cop = _world()
    x = _node(tree, alloc, [20, 21], P, backed=True)
    x.write_through_pending_id = x.id
    _lock_path(x)
    tree.ongoing_write_through[x.id] = (x, object(), [x])
    cc = types.SimpleNamespace(write_queue=list(queue))
    if copier_inflight is not None:
        cc._s108_copier = types.SimpleNamespace(inflight=copier_inflight)
    tree.cache_controller = cc
    return alloc, tree, cop, x


def test_a_write_through_in_flight_with_its_read_enqueued_moves_and_is_never_evicted():
    alloc, tree, cop, x = _wt_world()
    sv = KC.survey(tree, alloc, P, B)
    assert sv.owner[20].move_only and not sv.owner[20].backed
    plan, why = KC.plan_for(sv, B)
    assert plan is not None and not plan.evicts and plan.moved_pages == 2, why
    _take(alloc, list(range(1, B + 1)))                            # no room: never evicted
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "no evictable unit" in why


@pytest.mark.parametrize("kind", ["host_queue", "s108_copier"])
def test_a_write_whose_read_is_not_yet_enqueued_blocks(kind):
    """The op captured the source indices and reads them LATER: the node must not move."""
    alloc, tree, cop, x = (_wt_world(queue=[object()]) if kind == "host_queue"
                           else _wt_world(copier_inflight=1))
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "pinned" in why


# ---- (a) rc27 n=3: an un-backed tail pins its chain; once backed, the chain is a unit ---------

def _n3_world(tails_backed):
    """P [1..3] -> Q [4..9] -> T1 [26], T2 [27]; P -> R1 [10..16] -> R2 [17..23] -> Rt [28, 29].
    Nothing free at or below B=16, nothing running (rc27 14:59:16: used=0)."""
    alloc, tree, cop = _world()
    p_ = _node(tree, alloc, [1, 2, 3], P)
    q = _node(tree, alloc, [4, 5, 6, 7, 8, 9], P, parent=p_)
    t1 = _node(tree, alloc, [26], P, parent=q, backed=tails_backed)
    t2 = _node(tree, alloc, [27], P, parent=q, backed=tails_backed)
    r1 = _node(tree, alloc, list(range(10, 17)), P, parent=p_)
    r2 = _node(tree, alloc, list(range(17, 24)), P, parent=r1)
    rt = _node(tree, alloc, [28, 29], P, parent=r2, backed=tails_backed)
    return alloc, tree, cop, dict(p=p_, q=q, t1=t1, t2=t2, r1=r1, r2=r2, rt=rt)


def test_rc27_a_an_unbacked_tail_pins_its_chain_named():
    alloc, tree, cop, n = _n3_world(tails_backed=False)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is None and "no free page below 16 for un-backed node" in why
    assert "no evictable unit" in why, why


def test_rc27_a_once_the_tails_are_backed_the_chain_goes_leaf_first_and_the_rest_moves():
    """RED on 893d09fa3f (``no free page below 16 for un-backed node`` even with every
    tail backed: the room below the boundary was never made). GREEN: the R chain is
    ONE unit (best fit for the gap of 11), evicted leaf first BEFORE the moves (it is the
    room), T1/T2 move into it, nothing above the boundary stays."""
    alloc, tree, cop, n = _n3_world(tails_backed=True)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    assert plan is not None, why
    assert [m.node for m in plan.evicts] == [n["rt"], n["r2"], n["r1"]]
    assert plan.evict_first and sorted(s for s, _d in plan.pairs) == [26, 27]
    KC.execute(tree, alloc, plan, cop)
    assert tree.evicted == [n["rt"].id, n["r2"].id, n["r1"].id]
    live = [int(x) for nd in tree._collect_all_nodes() if nd is not tree.root_node
            and nd.component_data[FULL].value is not None
            for x in (nd.component_data[FULL].value // P).unique().tolist()]
    assert max(live) <= B


def test_best_fit_takes_the_smallest_unit_that_closes_the_gap_and_the_older_on_a_tie():
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, B + 1)))
    a = _node(tree, alloc, [17], P)                                # older (lower page)
    b = _node(tree, alloc, [18], P)
    big = _node(tree, alloc, [19, 20, 21], P)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, 20), 20)     # gap 1: page 21
    ev = [m.node for m in plan.evicts] if plan is not None else None
    assert ev in ([a], [b]), why                                   # the smallest that closes;
    assert big not in ev                                           # the tie: by content key


# ---- (c) the allocators are no replicas: a diverged page is never part of an eviction --------

def test_a_backed_unit_whose_page_is_free_on_another_rank_is_not_evictable():
    """Rank 1 holds X (1 backed page, free on rank 0 -- the n=64 shape) and Y (2 pages,
    backed on both). Gap 1. Best fit would take X -- but its replicas differ, so both
    ranks evict Y: one digest. (Mutant ``hi`` ignored -> rank 1 takes X: two digests.)"""
    plans = []
    svs = []
    for rank in (0, 1):
        alloc, tree, cop = _world()
        _take(alloc, list(range(1, B + 1)))
        _node(tree, alloc, [20, 21], P)                            # Y
        if rank == 1:
            _node(tree, alloc, [25], P)                            # X: live here, free there
        svs.append((alloc, tree))
    vecs = []
    for alloc, tree in svs:
        sv = KC.survey(tree, alloc, P, 18)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, 18, NP, [18])))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    for sv, _v in vecs:
        plans.append(dsv._kv_compact_plans_from_vote(sv, group, 18, NP, [18])[0][0])
    assert all(p is not None for p in plans)
    assert plans[0].evict_pages == plans[1].evict_pages == [20, 21]
    assert KC.plan_digest(5, plans[0]) == KC.plan_digest(5, plans[1])


# ---- three ranks, real threads, real collective: TP0 nodes=32, TP1/TP2 nodes=35 ---------------
#
# The metal of the whole boot in one tick: a running request whose locked prefix and decode
# tail sit above S5 (b), a backed idle conversation BELOW S5 that is the only room (a), an
# un-backed finished tail above S5, a backed idle leaf above S5, and one page live on the
# workers but free on TP0 (c). TP0 cuts the tree in 32 nodes, the workers in 35 (rc24
# _113244: "32 nodes on TP0, 35 on TP1/TP2").

S5 = FLOOR_LADDER[5] // PG                                         # 3072


def _build_rank(name, alloc, tree):
    worker = name != "TP0"
    taken = [p for p in range(1, S5 + 1)
             if not (100 <= p <= 999 or 1001 <= p <= 1400 or 1401 <= p <= 1450)]
    _take(alloc, taken)                                            # rank-local holders below S5
    if worker:
        a1a = _node(tree, alloc, list(range(1001, 1200)), PG, backed=False)
        a1 = _node(tree, alloc, list(range(1200, 1401)), PG, backed=False, parent=a1a)
    else:
        a1 = _node(tree, alloc, list(range(1001, 1401)), PG, backed=False)
    a2 = _node(tree, alloc, list(range(3500, 3600)), PG, backed=False, parent=a1)
    _lock_path(a2)                                                 # the running request's lock
    _take(alloc, list(range(3600, 3610)))                          # its decode tail
    c_pages = list(range(100, 1000))
    parts = [c_pages[i * 32:(i + 1) * 32] for i in range(27)] + [c_pages[27 * 32:]]
    par = None
    for i, part in enumerate(parts):                               # C1..C28, one backed chain
        if worker and i == 4:
            par = _node(tree, alloc, part[:16], PG, parent=par)
            par = _node(tree, alloc, part[16:], PG, parent=par)
        else:
            par = _node(tree, alloc, part, PG, parent=par)
    _node(tree, alloc, list(range(3700, 3710)), PG, backed=False)  # U: un-backed tail
    _node(tree, alloc, list(range(4000, 4050)), PG)                # W: backed idle leaf
    if worker:
        _node(tree, alloc, [4100], PG, backed=False)               # X: the n=64 page
    row = torch.cat([a1a.component_data[FULL].value if worker else torch.empty(0, dtype=torch.int64),
                     a1.component_data[FULL].value, a2.component_data[FULL].value,
                     torch.tensor([p * PG + o for p in range(3600, 3610) for o in range(PG)])])
    return a2, row


def test_three_ranks_other_cuts_running_lock_room_eviction_and_a_diverged_page(
        floor_env, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    owners = {"TP0": (0, 0), "TP1": (S, CUT[1]), "TP2": (S, CUT[2])}
    group = _ThreeRankGroup()
    ranks = []
    for name, form in (("TP0", _tp0), ("TP1", _tp1), ("TP2", _tp2)):
        target, dw = form(monkeypatch)
        alloc = _alloc(TOP // PG, PG)
        alloc._kvcache = target
        tree = Tree(alloc)
        a2, row = _build_rank(name, alloc, tree)
        req = _req("pdflip-0-14", 0, a2, row, prefix=row[:-10 * PG].clone())
        rtt = _rtt([row], 1100 * PG)
        sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=PG,
                                      draft_worker=dw, tree_cache=tree,
                                      req_to_token_pool=types.SimpleNamespace(req_to_token=rtt),
                                      running_batch=types.SimpleNamespace(reqs=[req]),
                                      waiting_queue=[], chunked_req=None, last_batch=None,
                                      _pdflip_group_min_ints=group.min_ints(name))
        ranks.append((name, sched, req, rtt))
    counts = {n: sum(1 for x in s.tree_cache._collect_all_nodes() if x is not s.tree_cache.root_node)
              for n, s, _r, _t in ranks}
    assert counts == {"TP0": 32, "TP1": 35, "TP2": 35}             # the premise: other cuts
    monkeypatch.setattr(dsv, "kv_owner_block", lambda: owners[threading.current_thread().name])
    from test_s77_kv_compact_s9_hold_1019 import _REAL_MAX_LIVE_PAGE
    monkeypatch.setattr(dsv, "max_live_page", _REAL_MAX_LIVE_PAGE)
    floor = 4100 * PG
    out = {}

    def run(name, sched):
        # the tick as MemSched drives it: S8 -> the deepest stage the moves alone reach
        # (S6: "compact, do not evict" first), then the pending S5 -- the agreed eviction
        first = {}
        try:
            for stage, fl in ((8, floor), (6, None)):
                ms = types.SimpleNamespace(stage=stage, stage_tokens=list(FLOOR_LADDER),
                                           _smallest_holding=lambda f: next(
                                               j for j, t in enumerate(FLOOR_LADDER) if t >= f))
                got = dsv._kv_compact_for_shrink(sched, ms, {}, 5, True, fl if fl else first[name])
                first.setdefault(name, got)
            out[name] = got
        except Exception as exc:  # noqa: BLE001 -- the outcome is the assertion
            out[name] = exc

    threads = [threading.Thread(target=run, args=(n, s), name=n) for n, s, _r, _t in ranks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert not any(t.is_alive() for t in threads), "a rank hangs"
    for name, sched, req, rtt in ranks:
        assert isinstance(out[name], int), (name, out[name])
        assert out[name] <= FLOOR_LADDER[5], (name, out[name])         # the floor is below S5
        tree = sched.tree_cache
        live = sorted({int(x) for n in tree._collect_all_nodes() if n is not tree.root_node
                       and n.component_data[FULL].value is not None
                       for x in (n.component_data[FULL].value // PG).unique().tolist()})
        assert max(live) <= S5, name
        row = rtt[0, :req.kv_allocated_len].long()
        assert int((row // PG).max()) <= S5, (name, "the running row still names a page above S5")
        assert torch.equal(row[:-10 * PG], req.prefix_indices.long()), name
        assert len(tree.evicted) == (28 if name == "TP0" else 29), (name, len(tree.evicted))
    moved = [m for m in caplog.messages if "verdict=moved" in m]
    assert len(moved) == 6, moved
    assert sum("target=S6" in m and "evicted_nodes=0" in m for m in moved) == 3
    s5 = [m for m in moved if "target=S5" in m]
    assert len(s5) == 3 and len({m.split("digest=")[1].split()[0] for m in s5}) == 1
    assert sorted(int(m.split("evicted_nodes=")[1].split()[0]) for m in s5) == [28, 29, 29]
    # S6: vote, plan, moves, floor; S5: vote, plan, evictions, moves, floor
    assert group.calls["TP0"] == group.calls["TP1"] == group.calls["TP2"] == 9


# ---- the retry and the census --------------------------------------------------------------

def test_a_refusal_is_looked_at_again_when_the_groups_write_through_fingerprint_changed(monkeypatch):
    """S98e gap: a refusal waited for the next END event, though a landed write-through
    (no end event) was what it waited for. The floor collective carries the group's
    write-through/load-back fingerprint; a changed one re-surveys, an unchanged one not."""
    alloc, tree, cop = _world()
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=P, tree_cache=tree)
    surveys = []
    monkeypatch.setattr(dsv, "_kv_compact_rank_survey",
                        lambda s_, fp, snap=None: surveys.append(fp) or (None, "desk", None))
    ms = types.SimpleNamespace(stage=3, stage_tokens=(16, 32, 64, 128),
                               _smallest_holding=lambda f: 3)
    fr = {}
    setattr(sched, dsv.KV_BLOCKERS_ATTR, (1, 1, 0, 0))
    dsv._kv_compact_for_shrink(sched, ms, fr, 1, True, 100)           # end event: surveyed
    dsv._kv_compact_for_shrink(sched, ms, fr, 1, False, 100)          # same fingerprint: skipped
    assert len(surveys) == 1 and fr["compact_skipped"] == 1
    setattr(sched, dsv.KV_BLOCKERS_ATTR, (0, 0, 0, 0))                # the write landed everywhere
    dsv._kv_compact_for_shrink(sched, ms, fr, 1, False, 100)
    assert len(surveys) == 2


def test_the_floor_collective_carries_the_fingerprint():
    alloc, tree, cop = _world()
    tree.ongoing_write_through = {1: object(), 2: object()}
    seen = []
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=P, tree_cache=tree,
                                  _pdflip_group_min_ints=lambda v: seen.append(list(v)) or [
                                      v[0], 0, -3, 0, 0])             # peers: 0..3 writes in flight
    dsv._group_floor_tokens(sched)
    assert len(seen) == 1 and seen[0][1:] == [2, -2, 0, 0]
    assert getattr(sched, dsv.KV_BLOCKERS_ATTR) == (0, 3, 0, 0)


def test_a_new_refusal_class_is_logged_beyond_n16_and_counted(monkeypatch, caplog):
    """rc27: n=17..31 and 33..63 were never logged (n <= 16 and powers of two) while
    the log showed 6 blocked shrinks, one at used=0 (15:06:12). Every NEW class is
    logged once, every line carries the class census and the skipped count."""
    caplog.set_level(logging.INFO)
    alloc, tree, cop = _world()
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=P, tree_cache=tree)
    monkeypatch.setattr(dsv, "_KV_COMPACT_N", [40])
    monkeypatch.setattr(dsv, "_KV_COMPACT_REASONS", {})
    whys = iter(["page 9 pinned (node 3: lock)", "page 10 pinned (node 4: lock)",
                 "no free page below 8 for un-backed node 5 (2 pages)"])
    monkeypatch.setattr(dsv, "_kv_compact_rank_survey", lambda s_, fp, snap=None: (None, next(whys), None))
    ms = types.SimpleNamespace(stage=3, stage_tokens=(16, 32, 64, 128), _smallest_holding=lambda f: 3)
    for _ in range(3):
        dsv._kv_compact_for_shrink(sched, ms, {}, 1, True, 100)
    lines = [m for m in caplog.messages if "verdict=refused" in m]
    assert [m.split(" -- ")[1].split(" [")[0] for m in lines] == [
        "page 9 pinned (node 3: lock)", "no free page below 8 for un-backed node 5 (2 pages)"]
    assert "[classes " in lines[-1] and "skipped=0" in lines[-1]


# ==== S112b: review nf-review-s112-1020.md (JA MIT AUFLAGE) ===================================

# ---- A1: evictions agreed by CONTENT, never by page id (the reviewer's probe) ----------------

def _plan2(ranks, floor, bounds):
    vecs = []
    for alloc, tree, _c, _l in ranks:
        sv = KC.survey(tree, alloc, P, floor)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, floor, NP, bounds)))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    return [dsv._kv_compact_plans_from_vote(sv, group, floor, NP, bounds)[0][0] for sv, _v in vecs]


def test_rv_s112_an_agreed_eviction_after_rank_local_moves_is_the_same_content():
    """The reviewer's probe (/tmp/rv-s112-own/test_rv_s112_content.py), RED on df587bf229:
    tick 1 moves rank-locally (rank 1 holds the extra backed X, the n=64 shape), so page 14
    is Y on rank 0 and X on rank 1; tick 2 needs room -- by page id rank 0 evicted Y+Z and
    rank 1 X+Y under ONE digest. By content: both evict Y and Z (X is no candidate: rank 0
    does not hold it), and the digests agree."""
    ranks = []
    for rank in (0, 1):
        alloc, tree, cop = _world()
        _take(alloc, list(range(1, 14)))                      # free <= 16: 14, 15, 16
        lab = {}
        if rank == 1:
            lab[id(_node(tree, alloc, [17], P))] = "X"
        lab[id(_node(tree, alloc, [18], P))] = "Y"
        lab[id(_node(tree, alloc, [19], P))] = "Z"
        ranks.append((alloc, tree, cop, lab))
    for (alloc, tree, cop, _l), plan in zip(ranks, _plan2(ranks, 16, [16])):
        assert plan is not None
        KC.execute(tree, alloc, plan, cop)
    plans = _plan2(ranks, 14, [14])
    ev = [sorted(r[3][id(m.node)] for m in (p.evicts if p else [])) for r, p in zip(ranks, plans)]
    assert ev[0] == ev[1] == ["Y", "Z"], ev
    assert KC.plan_digest(5, plans[0]) == KC.plan_digest(5, plans[1])


def test_the_content_key_of_a_segment_does_not_depend_on_the_node_cut():
    keys = []
    for split in (False, True):
        alloc, tree, cop = _world()
        if split:
            a = _node(tree, alloc, [20, 21], P)
            _node(tree, alloc, [22, 23], P, parent=a)
        else:
            _node(tree, alloc, [20, 21, 22, 23], P)
        sv = KC.survey(tree, alloc, P, B)
        assert len(sv.segments) == 1
        keys.append(sv.segments[0].key)
    assert keys[0] == keys[1]


# ---- A2: no host wait -- the survey reads a snapshot, the move is ordered by events -----------

class _RecOps:
    """CUDA-like stream handles in F3's ``d_row_fence`` interface that only RECORD (desk)."""

    def __init__(self):
        self.log = []

    def current_stream(self):
        return "cur"

    def event(self):
        return object()

    def record(self, ev, stream):
        self.log.append(("record", stream))

    def wait(self, stream, ev):
        self.log.append(("wait", stream))

    def query(self, ev):
        return True

    def synchronize(self, ev):
        self.log.append(("synchronize",))


def test_the_move_is_fenced_by_events_and_never_waits(monkeypatch):
    """E0 (an event at the tail of every KV stream, the current stream waits on it) ->
    claim/copy/re-point -> E1 (the KV streams wait on the current stream): F3's RowFence
    with an explicit KV stream set; no synchronize anywhere."""
    from flliper.srt.pdflip import d_row_fence as RF

    alloc, tree, cop, x, req, rtt, sched = _running_world()
    sv = KC.survey(tree, alloc, P, B, running=[req], req_to_token=rtt)
    plan, why = KC.plan_for(sv, B)
    ops = _RecOps()
    monkeypatch.setattr(RF, "_OPS", ops)
    real_copy = cop.copy
    cop.copy = lambda s, d: ops.log.append(("copy",)) or real_copy(s, d)
    KC.execute_moves(tree, alloc, plan, cop, streams=["fwd", "write", "load"],
                     remap_refs=dsv._kv_compact_remap_requests(sched, P, sv))
    assert ops.log == ([("record", "fwd"), ("wait", "cur"), ("record", "write"), ("wait", "cur"),
                        ("record", "load"), ("wait", "cur"), ("copy",), ("record", "cur")]
                       + [("wait", s) for s in ("fwd", "write", "load")]), ops.log


def test_no_host_read_in_the_move_path():
    """Pinned by source: the move, the fence, the claim, the give-back, the re-point of
    the rows and prefix tensors carry no synchronize / .cpu() / .item() / .tolist() /
    .any() (the reviewer's d_seat_vram Z. 3971 and d_kv_compact Z. 409 are gone)."""
    import inspect

    import ast
    import textwrap

    def code(fn):            # the source without docstrings and comments
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
        return ast.unparse(tree)

    for fn in (KC.execute_moves, KC.kv_fence_begin, KC._give_back, KC._remap_tensor, KC.lut_for,
               KC._take_pages, dsv._kv_compact_remap_requests):
        src = code(fn)
        for banned in ("synchronize", ".cpu(", ".item(", ".tolist(", ".any(", "_sync_device",
                       "[~", ".nonzero(", "masked_select"):
            assert banned not in src, (fn.__name__, banned)
    assert "DeferredRead" in inspect.getsource(KC.Snapshot.__init__)


def _tick_world(monkeypatch):
    alloc, tree, cop = _world()
    n = _node(tree, alloc, [20, 21], P)
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=P, tree_cache=tree)
    monkeypatch.setattr(dsv, "kv_page_buffers", lambda s_: ([], None))
    monkeypatch.setattr(dsv.KvPageCopier, "from_buffers", classmethod(lambda cls, b, p: cop))
    ms = types.SimpleNamespace(stage=3, stage_tokens=(16 * P, 24 * P, 28 * P, 32 * P),
                               _smallest_holding=lambda f: next(
                                   j for j, t in enumerate((16 * P, 24 * P, 28 * P, 32 * P)) if t >= f))
    return alloc, tree, cop, n, sched, ms


def test_a_snapshot_in_flight_defers_by_name_and_moves_in_a_later_pass(monkeypatch, caplog):
    """Pass 1: the snapshot's DeferredRead has not landed -> the vote's ready=0 -> deferred,
    named (B5), no memo, nothing surveyed or moved, no wait. Pass 2: landed, sources the
    same objects -> moved."""
    import logging

    from flliper.srt.pdflip import d_row_fence as RF

    caplog.set_level(logging.INFO)
    alloc, tree, cop, n, sched, ms = _tick_world(monkeypatch)
    landed = {"v": False}
    monkeypatch.setattr(RF.DeferredRead, "ready", lambda self: landed["v"])
    fr = {}
    v0 = n.component_data[FULL].value
    assert dsv._kv_compact_for_shrink(sched, ms, fr, 0, True, 22 * P) is None
    assert fr["compact_deferred"] == 1 and "compact_refused" not in fr
    assert n.component_data[FULL].value is v0
    lines = [m for m in caplog.messages if "verdict=deferred" in m]
    assert lines and "in_a_row=1" in lines[0] and "snapshot_reissued_for=new" in lines[0], lines
    landed["v"] = True
    got = dsv._kv_compact_for_shrink(sched, ms, fr, 0, False, 22 * P)
    assert got is not None and got <= 16 * P, got
    assert int((n.component_data[FULL].value // P).max()) <= 16
    moved = [m for m in caplog.messages if "verdict=moved" in m]
    assert moved and "host_waits_move=0" in moved[0] and "host_waits=0 " not in moved[0]


def test_a_snapshot_whose_keys_changed_is_issued_anew(monkeypatch):
    alloc, tree, cop, n, sched, ms = _tick_world(monkeypatch)
    snap = KC.Snapshot(tree, alloc, P, (), None)
    assert snap.valid(tree, alloc, ())
    assert snap.free.data_ptr() != alloc.free_pages.data_ptr()   # B3: a real copy on the desk
    _take(alloc, [3])                                         # an allocation since
    assert snap.changed(tree, alloc, ()) == "free"
    snap2 = KC.Snapshot(tree, alloc, P, (), None)
    alloc.free_pages = alloc.free_pages[:]                    # B2: a NEW tensor, same block
    assert snap2.changed(tree, alloc, ()) == "free"


# ---- A3: every holder of a request is re-pointed; overlap + spec v2 result path ---------------

def test_holders_outside_the_four_containers_are_re_pointed(monkeypatch):
    """A D-parked request (``pdflip_d_parked`` list), a vision-parked one in a dict and a
    lane-held one in a tuple: their prefix_indices and rows name a moved idle tree page;
    after the move they name the destination (no re-match is relied on)."""
    alloc, tree, cop = _world()
    x = _node(tree, alloc, [20, 21], P)                        # idle, not locked
    v = x.component_data[FULL].value.clone()
    parked = _req("parked", 1, None, v, prefix=v.clone())
    vision = _req("vision", 2, None, v, prefix=v.clone())
    lane = _req("lane", 3, None, v, prefix=v.clone())
    rtt = _rtt([torch.zeros(0), v, v, v], 64)
    sched = types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=rtt),
                                  running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[],
                                  chunked_req=None, last_batch=None, pdflip_d_parked=[parked],
                                  _pdflip_vision_parked={"k": vision}, lane_hold=[("x", lane)])
    holders = dsv._kv_compact_req_holders(sched)
    assert {h.rid for h in holders} == {"parked", "vision", "lane"}
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    KC.execute(tree, alloc, plan, cop, remap_refs=dsv._kv_compact_remap_requests(sched, P))
    new = x.component_data[FULL].value.long()
    for h in holders:
        assert torch.equal(h.prefix_indices.long(), new), h.rid
        assert torch.equal(rtt[h.req_pool_idx, :new.numel()].long(), new), h.rid


def test_overlap_spec_v2_a_move_between_launch_and_result_frees_through_the_rows():
    """The last batch (spec v2, overlap) was launched before the move and its result is
    processed after it. Its ``out_cache_loc`` is stale -- and nothing on the result path
    reads it: the finish frees through ``req_to_token`` (``release_kv_cache``:
    ``pop_overallocated_kv_cache`` + the row), i.e. through the re-pointed row. Desk form
    of that release: the over-allocated tail is freed from the row, the committed head
    goes into the tree -- only destination pages, never a source twice."""
    import inspect

    from flliper.srt.managers.scheduler_components import batch_result_processor as brp
    from flliper.srt.mem_cache import common

    assert "out_cache_loc" not in inspect.getsource(brp.SchedulerBatchResultProcessor.process_batch_result_decode)
    assert "out_cache_loc" not in inspect.getsource(common.release_kv_cache)
    alloc, tree, cop, x, req, rtt, sched = _running_world()
    req.kv_committed_len = 2 * P + 5                           # the tail page: spec reservation
    sched.last_batch = types.SimpleNamespace(reqs=[req], out_cache_loc=rtt[0, 2 * P:2 * P + 4].clone())
    sv = KC.survey(tree, alloc, P, B, running=[req], req_to_token=rtt)
    plan, why = KC.plan_for(sv, B)
    KC.execute(tree, alloc, plan, cop, remap_refs=dsv._kv_compact_remap_requests(sched, P, sv))
    stale = sorted(set((sched.last_batch.out_cache_loc // P).tolist()))
    assert stale == [25]                                       # the result path would free 25 twice
    row = rtt[0, :req.kv_allocated_len].long()
    tail = sorted(set((row[2 * P:] // P).tolist()))
    assert tail != stale and max(tail) <= B
    free_before = alloc.free_pages.clone()
    alloc.free(row[2 * P:])                                    # the over-allocated tail
    pages = alloc.free_pages.tolist()
    assert len(pages) == len(set(pages)), "a page was freed twice"
    assert set(pages) - set(free_before.tolist()) == set(tail)


# ==== S112c: re-review nf-rereview-s112b-1020 (B7, B8) ==========================================

def test_b7_a_row_copy_that_fails_midway_rolls_the_rows_back(monkeypatch):
    """Two holders; the second row's copy raises: the first row is restored, the nodes and the free
    lists are rolled back, the move is refused by name."""
    alloc, tree, cop = _world()
    x = _node(tree, alloc, [20, 21], P)
    v = x.component_data[FULL].value.clone()
    r1 = _req("a", 1, None, v, prefix=None)
    r2 = _req("b", 2, None, v, prefix=None)
    rtt = _rtt([torch.zeros(0), v, v], 64)
    before = rtt.clone()
    sched = types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=rtt),
                                  running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[r1, r2],
                                  chunked_req=None, last_batch=None)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    real = KC._remap_tensor
    calls = {"n": 0}

    def bad(t, P_, lut):
        calls["n"] += 1
        out = real(t, P_, lut)
        return out[:-1] if calls["n"] == 3 else out          # node, row a, row b (bad shape)
    monkeypatch.setattr(KC, "_remap_tensor", bad)
    free0 = alloc.free_pages.clone()
    with pytest.raises(KC.CompactRefused):
        KC.execute(tree, alloc, plan, cop, remap_refs=dsv._kv_compact_remap_requests(sched, P))
    assert torch.equal(rtt, before), "a row kept the new pages after the move was refused"
    assert torch.equal(x.component_data[FULL].value, v) and torch.equal(alloc.free_pages, free0)


def test_b8_no_move_inside_a_free_group():
    alloc, tree, cop = _world()
    _node(tree, alloc, [20, 21], P)
    plan, why = KC.plan_for(KC.survey(tree, alloc, P, B), B)
    alloc.is_not_in_free_group = False
    with pytest.raises(KC.CompactRefused, match="free group"):
        KC.execute_moves(tree, alloc, plan, cop)


def test_b4_the_choice_counts_the_groups_min_size_so_both_ranks_pick_alike():
    """Same head S on both ranks, TP0's one page longer (Form A leaf depth); T backed idle below the
    boundary on both. Gap: TP0 4, worker 3 (group 4). Counted with the local size TP0 would take S
    alone (4 fits) and the worker S and T -- two digests. With the group MIN (3) both take S and T."""
    from test_s77_kv_compact_s9_hold_1019 import _key

    head = list(range(7000, 7000 + 4 * P))
    ranks = []
    for rank in (0, 1):
        alloc, tree, cop = _world()
        _take(alloc, [p for p in range(1, 17) if p not in (5, 6, 7)])
        t = _node(tree, alloc, [5, 6, 7], P)
        t.key = _key(list(range(9000, 9000 + 3 * P)))
        pages = [17, 18, 19, 20] if rank == 0 else [17, 18, 19]
        s_ = _node(tree, alloc, pages, P)
        s_.key = _key(head[:len(pages) * P])
        ranks.append((alloc, tree))
    vecs = []
    for alloc, tree in ranks:
        sv = KC.survey(tree, alloc, P, 16)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, 16, NP, [16])))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    plans = [dsv._kv_compact_plans_from_vote(sv, group, 16, NP, [16])[0][0] for sv, _v in vecs]
    assert all(p is not None for p in plans)
    assert [len(p.evict_keys) for p in plans] == [2, 2]
    assert KC.plan_digest(5, plans[0]) == KC.plan_digest(5, plans[1])
