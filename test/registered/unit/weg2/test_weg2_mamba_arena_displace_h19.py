"""fnFL2 H19: the mamba host arena keeps a request's DEEPEST anchors and the
end anchor always gets a slot.

Bug (boot fnFL2x130, chunk 512, 97841 tokens = 191 chunk anchors, 32 arena
slots): every landed anchor write keeps one reader reference per P rank for as
long as the node keeps its host value; nothing released them during the
prefill, so the arena held the FIRST 30 anchors (pages 0..239), refused every
later one (`stopped: mamba_full`) and the end anchor at 97792 never reached the
arena -- D capped its claim at page 240 and prefilled a second time.

Hermetic: one real shared arena (arena.c, gcc), three rank pools with their own
extents of the blob (PP0/PP1/PP2 join the same key and each completes its
part), a chain of real UnifiedTreeNode objects per rank, the real
`UnifiedRadixCache._weg2_mamba_claim` and its helpers; the eviction funnel is
reduced to "free this rank's host value" (what the mamba component's HOST
eviction does)."""
from __future__ import annotations

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from sglang.srt.weg2 import mamba_arena_displace as mad  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

RANKS = 3
EXT = 32
SB = RANKS * EXT           # one blob = three rank extents
TC = (ComponentType.FULL, ComponentType.MAMBA)
M = ComponentType.MAMBA


class _Backend:
    def _get_suffixed_key(self, k):
        return f"{k}.sfx"

    def _log_key(self, pool, k):
        return f"{k}.mamba"


def _pool(arena, rank):
    p = object.__new__(ArenaMambaPoolHost)
    p.size = 2
    p._arena_init_fields()
    p.arena = arena
    p.arena_slots = arena.slots
    p._backend = _Backend()
    p._page_bytes = SB
    p._own_extents = [(rank * EXT, EXT)]
    return p


def _stem(h):
    return f"{h}.mamba.sfx"


class _Rank:
    """One P rank: its tree, its pool on the shared arena, its unbacked queue
    (the chunk/retain/flush sweeps: parents first, stop at the first refusal)."""

    def __init__(self, arena, rank, cfg):
        self.mp = _pool(arena, rank)
        c = object.__new__(UnifiedRadixCache)
        c.root_node = UnifiedTreeNode(TC)
        c._weg2_rid_anchor_cfg = cfg
        c._weg2_anchor_ledger = mad.RidAnchorLedger()
        c._weg2_direct_mamba_rows = {}
        c.components = {M: None}
        c._evict_component_and_detach_lru = self._evict_host
        c._update_evictable_leaf_sets = lambda n: None
        self.cache = c
        self.tips = {}
        self.unbacked = []

    def _evict_host(self, node, comp, target=None, tracker=None):
        cd = node.component_data[M]
        self.mp.free(cd.host_value)
        cd.host_value = None
        return 0, 1

    def add(self, rid, h, tokens=512, end_anchor=False, parent=None):
        parent = parent or self.tips.get(rid, self.cache.root_node)
        n = UnifiedTreeNode(TC)
        n.key = [0] * tokens
        n.parent = parent
        parent.children[h] = n
        n._weg2_end_anchor = end_anchor
        self.cache._weg2_tag_anchor(n, rid)
        self.tips[rid] = n
        self.unbacked.append((n, h))
        return n

    def sweep(self):
        while self.unbacked:
            n, h = self.unbacked[0]
            rows = self.cache._weg2_mamba_claim(n, self.mp, h)
            if rows is None:
                return False
            n.component_data[M].host_value = rows
            self.mp.complete_write(rows)
            self.unbacked.pop(0)
        return True

    def anchors(self, rid):
        path, _ = mad.ancestor_path(target=self.tips[rid], root=self.cache.root_node)
        path.append((self.tips[rid], None))
        return [n for n, _ in path if n.weg2_anchor_rid == rid and n.component_data[M].host_value is not None]


def _arena(tmp_path, slots):
    return ShmArena(str(tmp_path / f"arena-mamba-{slots}.bin"), SB, slots)


def _state(arena, h):
    return arena.find_slots([_stem(h)])[0][1]


def _prefill(ranks, rid, chunks, tokens=512):
    """PP pipeline order: PP r works on chunk k-r; each chunk ends in that
    rank's chunk sweep; the last node is the END-ANCHOR node, published by
    the retain sweep; then the flush sweeps (#1470, up to 64 rounds)."""
    hs = [f"{rid}-h{k}" for k in range(chunks)]
    for step in range(chunks + RANKS - 1):
        for r, rk in enumerate(ranks):
            k = step - r
            if 0 <= k < chunks:
                rk.add(rid, hs[k], tokens=tokens, end_anchor=(k == chunks - 1))
                rk.sweep()
    for _ in range(64):
        if all([rk.sweep() for rk in ranks]):   # every rank sweeps each round
            break
    return hs


def _finished_short(ranks, rid):
    """A one-node request of the same phase (ctrl/mid probes of x130)."""
    for rk in ranks:
        rk.add(rid, f"{rid}-end", end_anchor=True)
        assert rk.sweep()


def test_x130_first_come_loses_the_end_anchor_when_off(tmp_path):
    """Regression proof: displacement off == the pre-H19 claim. 191 anchors,
    32 slots, two slots held by earlier requests -> the first 30 stay, the end
    anchor never gets a slot."""
    arena = _arena(tmp_path, 32)
    ranks = [_Rank(arena, r, cfg=0) for r in range(RANKS)]
    _finished_short(ranks, "ctrl")
    _finished_short(ranks, "mid")
    hs = _prefill(ranks, "weg2-0-4", 191)
    assert [_state(arena, h) for h in hs[:30]] == [2] * 30
    assert _state(arena, hs[30]) != 2
    assert _state(arena, hs[-1]) != 2           # the end anchor: never in the arena
    assert all(rk.unbacked for rk in ranks)      # the sweep stays stuck on anchor 31


def test_x130_deeper_anchors_displace_shallow_and_the_end_anchor_lands(tmp_path):
    arena = _arena(tmp_path, 32)
    ranks = [_Rank(arena, r, cfg=-1) for r in range(RANKS)]
    _finished_short(ranks, "ctrl")
    _finished_short(ranks, "mid")
    hs = _prefill(ranks, "weg2-0-4", 191)
    cap = mad.rid_anchor_cap(configured=-1, arena_slots=32)
    assert cap == 8
    assert not any(rk.unbacked for rk in ranks)
    assert _state(arena, hs[-1]) == 2            # END ANCHOR complete on all three ranks
    # every rank keeps exactly the `cap` deepest anchors of the request
    for rk in ranks:
        kept = rk.anchors("weg2-0-4")
        assert len(kept) == cap
        assert kept[-1] is rk.tips["weg2-0-4"]
    assert [_state(arena, h) for h in hs[-cap:]] == [2] * cap
    assert all(_state(arena, h) != 2 for h in hs[:-cap])   # displaced and dropped, no disk write
    # the other requests' anchors are untouched
    assert _state(arena, "ctrl-end") == 2 and _state(arena, "mid-end") == 2
    st = ranks[0].cache._weg2_anchor_ledger.peek("weg2-0-4")
    assert st.end_anchor == "slot" and st.written == 191 and st.deepest == 191 * 512
    assert st.displaced_share == 191 - cap
    assert arena.stats()["complete"] == cap + 2


def test_chunk_16384_is_unchanged(tmp_path):
    """6 chunk anchors + the end anchor of a 98k request stay below the share:
    no displacement, the same anchors as with displacement off."""
    kept = {}
    for cfg in (0, -1):
        (tmp_path / f"c{cfg}").mkdir()
        arena = _arena(tmp_path / f"c{cfg}", 32)
        ranks = [_Rank(arena, r, cfg=cfg) for r in range(RANKS)]
        hs = _prefill(ranks, "r16k", 7, tokens=16384)
        kept[cfg] = [_state(arena, h) for h in hs]
        if cfg == -1:
            st = ranks[0].cache._weg2_anchor_ledger.peek("r16k")
            assert st.displaced_share == 0 and st.displaced_full == 0 and st.refused == 0
    assert kept[0] == kept[-1] == [2] * 7


def test_foreign_requests_are_never_thinned_by_the_share(tmp_path):
    """A finished request A with several anchors, then a long request B: B
    thins only its own chain."""
    arena = _arena(tmp_path, 24)
    ranks = [_Rank(arena, r, cfg=4) for r in range(RANKS)]
    ha = _prefill(ranks, "A", 4)
    hb = _prefill(ranks, "B", 40)
    assert [_state(arena, h) for h in ha] == [2] * 4
    assert _state(arena, hb[-1]) == 2
    assert [len(rk.anchors("A")) for rk in ranks] == [4] * RANKS
    assert [len(rk.anchors("B")) for rk in ranks] == [4] * RANKS


def test_end_anchor_takes_a_foreign_intermediate_when_the_arena_is_full(tmp_path):
    """Arena full of A's anchors; B's end anchor has no anchor of its own to
    give -> A's shallowest INTERMEDIATE anchor goes; A's deepest (its end
    anchor) stays."""
    arena = _arena(tmp_path, 4)
    ranks = [_Rank(arena, r, cfg=8) for r in range(RANKS)]
    ha = _prefill(ranks, "A", 4)
    assert [_state(arena, h) for h in ha] == [2] * 4
    for rk in ranks:
        rk.add("B", "B-end", end_anchor=True, parent=rk.cache.root_node)
    for _ in range(4):
        if all([rk.sweep() for rk in ranks]):   # every rank sweeps each round
            break
    assert _state(arena, "B-end") == 2
    assert _state(arena, ha[0]) != 2
    assert [_state(arena, h) for h in ha[1:]] == [2] * 3
    assert all(rk.cache._weg2_anchor_ledger.peek("B").end_anchor == "slot" for rk in ranks)


def test_an_intermediate_node_never_takes_foreign_anchors(tmp_path):
    """A node that is not an end anchor, whose request holds anchors but none
    releasable right now (host lock), waits -- it never thins another request
    -- and takes its own shallowest anchor once that is free again."""
    arena = _arena(tmp_path, 3)
    rk = _Rank(arena, 0, cfg=8)
    rk.mp._own_extents = [(0, SB)]
    rk.add("A", "A0")
    rk.add("A", "A1", end_anchor=True)
    b0 = rk.add("B", "B0", parent=rk.cache.root_node)
    assert rk.sweep()
    b0.component_data[M].host_lock_ref = 1
    rk.add("B", "B1")
    assert not rk.sweep()
    assert [_state(arena, h) for h in ("A0", "A1", "B0")] == [2, 2, 2]
    b0.component_data[M].host_lock_ref = 0
    assert rk.sweep()
    assert [_state(arena, h) for h in ("A0", "A1", "B1")] == [2, 2, 2]
    assert _state(arena, "B0") != 2


def test_retain_line_sees_the_end_anchor_claimed_at_its_chunk(tmp_path, caplog):
    """The N-1 node's anchor is claimed by its chunk publish; the finish marks
    it END-ANCHOR afterwards. The retain line reads the slot off the tree."""
    arena = _arena(tmp_path, 32)
    rk = _Rank(arena, 0, cfg=-1)
    rk.mp._own_extents = [(0, SB)]
    for k in range(12):
        rk.add("R", f"R{k}")
    assert rk.sweep()
    tip = rk.tips["R"]
    tip._weg2_end_anchor = True
    rk.cache._weg2_mamba_pool = lambda: rk.mp
    chain = [n for n, _ in mad.ancestor_path(target=tip, root=rk.cache.root_node)[0]] + [tip]
    with caplog.at_level("INFO"):
        rk.cache._weg2_log_rid_anchors("R", chain)
    st = rk.cache._weg2_anchor_ledger.peek("R")
    assert st.end_anchor == "slot" and st.written == 12 and st.displaced_share == 4
    assert "WEG2 MAMBA-ARENA rid=R written=12 displaced=4(share=4,full=0)" in caplog.text
    assert "end_anchor=slot cap=8 slots=32 at=retain" in caplog.text


def test_drop_unreferenced_frees_only_complete_unreferenced_slots(tmp_path):
    arena = _arena(tmp_path, 4)
    got = arena.claim_slots(["a", "b", "c"], [SB] * 3)
    slots = [g[0] for g in got]
    arena.complete_slots(slots[:2], [g[2] for g in got[:2]], [(0, SB)])
    arena.ref_slots([slots[1]], +1)
    assert arena.drop_unreferenced(slots) == 1          # a: complete, no reference
    assert arena.find_slots(["a"])[0][0] == -1
    assert arena.find_slots(["b"])[0][1] == 2           # referenced: stays
    assert arena.find_slots(["c"])[0][1] == 1           # claimed: stays
    assert arena.stats()["complete"] == 1
    again = arena.claim_slots(["d"], [SB])
    assert again[0][1] == 0


def test_cap_parsing():
    assert mad.rid_anchor_cap(configured=0, arena_slots=32) == 0
    assert mad.rid_anchor_cap(configured=-1, arena_slots=32) == 8
    assert mad.rid_anchor_cap(configured=-1, arena_slots=140) == 35
    assert mad.rid_anchor_cap(configured=-1, arena_slots=4) == 2
    assert mad.rid_anchor_cap(configured=1, arena_slots=32) == 2
    assert mad.rid_anchor_cap(configured=12, arena_slots=32) == 12
