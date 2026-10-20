"""Re-Review S112b (20.10., nf-rereview-s112b-1020.md): eigene Proben.

xfail(strict) = Befund (rot auf e3a7261a3f). S112c: alle fuenf Befunde behoben, die
Marker sind gefallen (B1 x2, B2, B3, B4); dazu die Bestaetigung (Kollision)."""
from __future__ import annotations

import ast
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from test_s77_kv_compact_s9_hold_1019 import FULL, _key, _node, _take  # noqa: E402
from test_s112_kv_compact_nie_1020 import NP, P, _world  # noqa: E402


# ---- A2: the snapshot key misses a reused block (ABA) -------------------------------------

def test_A2_snapshot_key_detects_changed_free_list_in_a_reused_block(monkeypatch):
    """The caching allocator hands a freed block back to the next tensor of the same size:
    free list content changed (page 14 handed out, page 13 freed), pointer / numel / offset
    equal. The snapshot must not be 'valid' then -- else the plan claims page 14 (live)."""
    alloc, tree, cop = _world()
    _take(alloc, list(range(1, 14)))                  # free <= 16: 14, 15, 16
    _node(tree, alloc, [20, 21], P)
    blk = alloc.free_pages.clone()
    alloc.free_pages = blk[:]
    snap = KC.Snapshot(tree, alloc, P, (), None)
    new = blk.clone()
    new[new == 14] = 13                               # 14 allocated, 13 freed, same length
    blk.copy_(new)                                    # the reused block
    alloc.free_pages = blk[:]                         # a new tensor in the old block
    sv = KC.survey(tree, alloc, P, 16, snap=snap)
    plan, _w = KC.plan_for(sv, 16)
    dsts = sorted(d for _s, d in plan.pairs) if plan is not None else []
    live_now = 14 not in set(alloc.free_pages.tolist())
    assert not snap.valid(tree, alloc, ()), (
        "snapshot 'valid' after the free list changed in a reused block; plan dsts=%s, "
        "page 14 live now=%s" % (dsts, live_now))


def test_A2_desk_snapshot_aliases_the_live_free_list():
    """On the CPU desk KC._to_host returns the live tensor itself (int64 .to() is a no-op):
    a desk snapshot can never be stale, so no desk test can see a stale snapshot."""
    alloc, tree, cop = _world()
    snap = KC.Snapshot(tree, alloc, P, (), None)
    assert snap.free.data_ptr() != alloc.free_pages.data_ptr(), "desk snapshot aliases free_pages"


# ---- A2: host waits left in the move path --------------------------------------------------

def _code(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return ast.unparse(tree)


def test_A2_take_pages_has_no_boolean_mask_index():
    """``t[~hit]`` on a CUDA tensor is a boolean-mask index = nonzero = the host waits for
    the current stream (which after _apply_war_barrier waits on the forward in flight)."""
    src = _code(KC._take_pages)
    assert "[~hit]" not in src, "_take_pages: boolean-mask index on the device free list"


def test_A2_host_waits_is_measured_not_a_literal():
    src = inspect.getsource(dsv._kv_compact_for_shrink)
    assert "host_waits=0 " not in src, "the moved line prints host_waits=0 as a literal"


# ---- A1 safety: a bitmap collision ends in a uniform refusal --------------------------------

def _key_of(tokens):
    # S112c (B4): segments are keyed at their HEAD (path state + first page + start depth)
    return KC.content_key_of_tokens(tokens, P)


def _colliding_contents():
    base = list(range(1000, 1000 + 2 * P))
    k0 = _key_of(base)
    for i in range(1, 200000):
        cand = list(range(5000 + i * 8, 5000 + i * 8 + 2 * P))   # differs in the first page
        if _key_of(cand) % KC.BITMAP == k0 % KC.BITMAP and _key_of(cand) != k0:
            return base, cand
    raise AssertionError("no collision found")


def test_A1_a_bitmap_collision_refuses_on_every_rank():
    """Rank 0: K0 evictable, K1 present but locked. Rank 1: K1 evictable, no K0. K0 and K1
    share a bitmap slot: the slot MIN says 'agreed' on both. Rank 0 picks K0, rank 1 K1 --
    the digests must differ (uniform refusal), never one digest."""
    c0, c1 = _colliding_contents()
    plans, ranks = [], []
    for rank in (0, 1):
        alloc, tree, cop = _world()
        if rank == 0:
            _take(alloc, list(range(1, 17)) + [19, 20])   # no room below 20
            b = _node(tree, alloc, [17, 18], P, lock=1)   # K1 here: locked (a running turn)
            b.key = _key(c1)
            a = _node(tree, alloc, [21, 22], P)           # K0: backed idle
            a.key = _key(c0)
        else:
            _take(alloc, list(range(1, 21)))
            b = _node(tree, alloc, [21, 22], P)           # K1: backed idle here
            b.key = _key(c1)
        ranks.append((alloc, tree))
    vecs = []
    for alloc, tree in ranks:
        sv = KC.survey(tree, alloc, P, 20)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, 20, NP, [20])))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    k = 1
    slot = _key_of(c0) % KC.BITMAP
    assert group[2 + k + slot] == 1                    # the collision: the slot reads 'agreed'
    for sv, _v in vecs:
        plans.append(dsv._kv_compact_plans_from_vote(sv, group, 20, NP, [20])[0][0])
    assert all(p is not None for p in plans)           # each rank found 'agreed' content ...
    assert plans[0].evict_keys == [_key_of(c0)] and plans[1].evict_keys == [_key_of(c1)]
    ds = [KC.plan_digest(5, p) for p in plans]
    assert ds[0] != ds[1], ds                          # ... and the digest refuses uniformly


# ---- A1 liveness: Form A leaf depths differ (TP0 90880 vs workers 90624) ---------------------

def test_A1_liveness_a_branch_whose_leaf_depth_differs_between_ranks_can_be_evicted():
    """Same session branch on both ranks, backed and idle; TP0's leaf is one page longer
    (the n=64 Form A insert-depth difference). No free page below the boundary: the room
    must come from evicting the branch. Content-agreed eviction of the branch is R11-safe
    (afterwards no rank holds it on the device) -- S112b keys the segment by its BOTTOM,
    so the keys differ and nothing is agreed."""
    content = list(range(7000, 7000 + 4 * P))
    plans = []
    ranks = []
    for rank in (0, 1):
        alloc, tree, cop = _world()
        _take(alloc, list(range(1, 17)))
        a = _node(tree, alloc, [17, 18], P)
        a.key = _key(content[:2 * P])
        leaf_pages = [19, 20] if rank == 0 else [19]
        l_ = _node(tree, alloc, leaf_pages, P, parent=a)
        l_.key = _key(content[2 * P:2 * P + len(leaf_pages) * P])
        ranks.append((alloc, tree))
    vecs = []
    for alloc, tree in ranks:
        sv = KC.survey(tree, alloc, P, 16)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, 16, NP, [16])))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    whys = []
    for sv, _v in vecs:
        pl, w = dsv._kv_compact_plans_from_vote(sv, group, 16, NP, [16])
        plans.append(pl[0])
        whys.append(w)
    assert all(p is not None for p in plans), whys
