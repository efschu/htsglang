"""Q-610 (dual y8t 10031114, 5d12a95e92, 60-min acceptance 11:24:25Z): group P
of the dual layout never resets, so its tree kept a reader reference on every
mamba anchor it ever wrote until the 112-slot arena was pinned -- and every
request after that ended as W35 (long) or W53 (short).

THE METAL, weg2-0-99 (7695 tokens, front credit 6156 from d_leg2_cached):
  P PP0 11:24:25  #1427 ARENA-DROP n=19 need=1 freed=0 stages=i:0,ii:0,iii:0
  P PP0 11:24:26  #1427 ARENA-CLAIM REFUSED statuses=[4] (4 = no free slot)
                  #1421 BACKUP-REFUSED why=mamba_claim node=155 depth=7694 rid=weg2-0-99
                  P-FUND EVICT KV-ONLY node=155 end_anchor=True
  D TP0 11:24:27  #1028B FETCH CAP kv=1538 claimed=0 lost=1538 ... MAMBA: (0, -1)
                  #1035c ZERO-ANSWER PARTITION cause=CAPPED ... by=mamba
                  X-GATE uncached=1539 X=1 verdict=W31 -> D-HANDBACK-DEFER refused -> W50
  front           X-REQUEUE n=1 -> P leg 1 again (BACKUP-REFUSED why=anchor_only_claim)
                  -> X-REQUEUE n=2 verdict=W35 (weg2-0-100 N=25: W53)
  P PP0 census    ARENA-REF-HOLDERS (arena-78446592.bin) tree=92 arena_pinned=112

Hermetic, CPU: ONE real C arena (arena.c, gcc), three P rank pools (real
ArenaMambaPoolHost, each rank its own extent of the blob, joining the same
key), real UnifiedTreeNode chains, the real claim (`_weg2_mamba_claim` ->
`alloc_write` -> `_evict_for_claim` with the real #243 keep list), the real
hand-off marks/tombstones (handoff_pending on a temp arena dir). The HOST
eviction funnel is reduced to "free this rank's host value" -- what the mamba
component's HOST eviction does.
"""
from __future__ import annotations

import logging
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from sglang.srt.weg2 import handoff as ho  # noqa: E402
from sglang.srt.weg2 import handoff_pending as hp  # noqa: E402
from sglang.srt.weg2 import mamba_arena_displace as mad  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

RANKS = 3
EXT = 32
SB = RANKS * EXT            # one mamba blob = the three PP ranks' extents
TC = (ComponentType.FULL, ComponentType.MAMBA)
M = ComponentType.MAMBA
SLOTS = 8                   # the metal arena has 112; the class is the ratio


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
    """One group-P rank (PP r): its tree, its pool on the shared arena, its
    unbacked queue (the retain/flush sweeps: stop at the first refusal). The
    27B form: cap 0 (grid4096 -- no H19 displacement)."""

    def __init__(self, arena, rank):
        self.mp = _pool(arena, rank)
        c = object.__new__(UnifiedRadixCache)
        c.root_node = UnifiedTreeNode(TC)
        c._weg2_rid_anchor_cfg = 0
        c._weg2_anchor_ledger = mad.RidAnchorLedger()
        c._weg2_direct_mamba_rows = {}
        c.components = {M: None}
        c._evict_component_and_detach_lru = self._evict_host
        c._update_evictable_leaf_sets = lambda n: None
        c._weg2_mamba_pool = lambda: self.mp
        self.cache = c
        self.unbacked = []
        self.refused = 0

    def _evict_host(self, node, comp, target=None, tracker=None):
        cd = node.component_data[M]
        self.mp.free(cd.host_value)
        cd.host_value = None
        return 0, 1

    def add(self, rid, h, end_anchor=True, parent=None):
        parent = parent or self.cache.root_node
        n = UnifiedTreeNode(TC)
        n.key = [0] * 512
        n.parent = parent
        parent.children[h] = n
        n._weg2_end_anchor = end_anchor
        self.cache._weg2_tag_anchor(n, rid)
        if end_anchor:
            # production: _weg2_note_end_anchor marks the node and registers it
            getattr(self.cache, "_weg2_dual_note_end_anchor", lambda *a: None)(rid, n)
        self.unbacked.append((n, h))
        return n

    def retain(self):
        """production: _weg2_publish_at_retain starts with the release"""
        getattr(self.cache, "_weg2_dual_retain_release", lambda: None)()

    def sweep(self):
        while self.unbacked:
            n, h = self.unbacked[0]
            rows = self.cache._weg2_mamba_claim(n, self.mp, h)
            if rows is None:
                self.refused += 1   # '#1421 BACKUP-REFUSED why=mamba_claim'
                return False
            n.component_data[M].host_value = rows
            self.mp.complete_write(rows)
            self.unbacked.pop(0)
        return True

    def held(self):
        """the ARENA-REF-HOLDERS tree term: arena anchors this rank's tree holds"""
        out, stack = 0, list(self.cache.root_node.children.values())
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n.component_data[M].host_value is not None:
                out += 1
        return out


@pytest.fixture
def dual_p(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path / "arena-dir"))
    os.makedirs(tmp_path / "arena-dir", exist_ok=True)
    return tmp_path


def _arena(tmp_path, slots=SLOTS):
    return ShmArena(str(tmp_path / f"arena-mamba-{slots}.bin"), SB, slots)


def _state(arena, h):
    return arena.find_slots([_stem(h)])[0][1]


def _rounds(ranks, n=4):
    """every rank sweeps each round, PP0 first (the PP pipeline order)"""
    for _ in range(n):
        if all([rk.sweep() for rk in ranks]):
            return True
    return False


def _handoff(rid, h):
    """P's finish: #1442 hand-off file + #243 pending mark (group P)"""
    assert ho.write(rid, [1, 2, 3], [h])
    hp.mark(rid, 1, 64)   # group P only (False on D)


def _finish(ranks, rid, *, consumed=True, ended=True):
    """One request through P: END-ANCHOR node on every rank, hand-off, retain
    (release + sweep); then D takes it (consume at fetch) and the front ends
    the rid (its terminal rid-end drop)."""
    h = f"{rid}-end"
    for rk in ranks:
        rk.add(rid, h)
    _handoff(rid, h)
    for rk in ranks:
        rk.retain()
    ok = _rounds(ranks)
    if consumed:
        hp.consume(rid, "fetch")
    if ended:
        hp.drop(rid, "status_200")
    return ok, h


def test_replay_10031114_weg2_0_99_end_anchor_lands_in_a_p_pinned_arena(dual_p):
    """The metal state: every slot COMPLETE and referenced by P's tree on all
    three ranks (anchors of rids D read and the front ended long ago), no
    retain release in between (the claim path alone). weg2-0-99's END anchor
    must get a slot -- else D's FETCH CAP by=mamba, W50, requeue, W35."""
    arena = _arena(dual_p)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    old = []
    for k in range(SLOTS):
        rid = f"weg2-0-{90 + k}"
        h = f"{rid}-end"
        for rk in ranks:
            rk.add(rid, h)
        _handoff(rid, h)
        assert _rounds(ranks)
        hp.consume(rid, "fetch")
        hp.drop(rid, "status_200")
        old.append(h)
    # registry emptied as if those rids had registered before this process
    # learned of the release (the claim walk finds them by tag)
    for rk in ranks:
        getattr(rk.cache, "_weg2_dual_end_reg", None) and rk.cache._weg2_dual_end_reg.entries.clear()
    assert arena.stats()["complete"] == SLOTS
    assert [rk.held() for rk in ranks] == [SLOTS] * RANKS          # tree=92 of 112 on the metal
    for rk in ranks:
        rk.add("weg2-0-99", "weg2-0-99-end")
    _handoff("weg2-0-99", "weg2-0-99-end")
    _rounds(ranks)
    # what D's #1028B FETCH CAP asks: the tail's anchor in range, COMPLETE
    assert _state(arena, "weg2-0-99-end") == 2, "END anchor refused: FETCH CAP by=mamba -> W50 -> W35"
    assert not any(rk.unbacked for rk in ranks)
    # one old anchor made the room; the rest stay COMPLETE (findable by stem)
    assert sum(_state(arena, h) == 2 for h in old) == SLOTS - 1
    # P keeps only the live hand-off's reference
    assert [rk.held() for rk in ranks] == [1] * RANKS


def test_retain_release_keeps_the_arena_from_filling(dual_p, caplog):
    """Abnahme shape: 4 x the arena in finished requests, each read by D and
    ended at the front before the next. With the retain release no claim is
    ever refused (and the claim path is never needed)."""
    arena = _arena(dual_p)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    with caplog.at_level(logging.INFO):
        for k in range(4 * SLOTS):
            ok, h = _finish(ranks, f"weg2-0-{k}")
            assert ok, f"request {k}: END anchor refused (arena pinned by P's tree)"
            assert _state(arena, h) == 2
    assert sum(rk.refused for rk in ranks) == 0
    assert "Q-610 DUAL-ANCHOR-RELEASE" in caplog.text and "at=retain" in caplog.text
    assert "at=claim" not in caplog.text
    assert max(rk.held() for rk in ranks) <= 1


def test_a_pending_hand_off_is_never_released(dual_p, caplog):
    """End anchors D has not taken yet (pending marks) keep P's reference --
    a full arena of live hand-offs refuses the next claim, as before, and the
    walk is not repeated before the next retain."""
    arena = _arena(dual_p)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    hs = []
    for k in range(SLOTS):
        ok, h = _finish(ranks, f"weg2-1-{k}", consumed=False, ended=False)
        assert ok
        hs.append(h)
    for rk in ranks:
        rk.add("weg2-1-99", "weg2-1-99-end")
    with caplog.at_level(logging.INFO):
        assert not _rounds(ranks, n=3)
    assert [_state(arena, h) for h in hs] == [2] * SLOTS
    assert [rk.held() for rk in ranks] == [SLOTS] * RANKS
    # one walk per rank (kept_pending named), the later refusals do not walk
    assert caplog.text.count("at=claim released=0") == RANKS
    assert f"kept_pending={SLOTS}" in caplog.text


def test_consumed_but_not_ended_is_kept_until_the_front_ends_it(dual_p):
    """D took the rid (consume at fetch) but it still decodes: P keeps the
    reference; the front's rid-end drop releases it at the next retain."""
    arena = _arena(dual_p)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    _finish(ranks, "weg2-2-0", consumed=True, ended=False)
    for rk in ranks:
        rk.retain()
    assert [rk.held() for rk in ranks] == [1] * RANKS
    hp.drop("weg2-2-0", "status_200")
    for rk in ranks:
        rk.retain()
    assert [rk.held() for rk in ranks] == [0] * RANKS
    assert _state(arena, "weg2-2-0-end") == 2      # still COMPLETE in the arena


def test_a_reroute_is_not_an_end(dual_p):
    """W50-REROUTE drops the mark with 'reroute_fresh' (not terminal): the rid
    lives on, its end anchor keeps P's reference."""
    arena = _arena(dual_p)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    _finish(ranks, "weg2-3-0", consumed=False, ended=False)
    hp.drop("weg2-3-0", "reroute_fresh")
    for rk in ranks:
        rk.retain()
    assert [rk.held() for rk in ranks] == [1] * RANKS


def test_prefix_anchors_go_at_a_refused_claim_but_never_under_a_running_request(dual_p):
    """The claim walk also gives back P's prefix-cache anchors (forks) -- but
    never one on the path of a running request (device lock)."""
    arena = _arena(dual_p, slots=4)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    for rk in ranks:
        a = rk.add("weg2-4-0", "fork-a", end_anchor=False)
        rk.add("weg2-4-0", "fork-b", end_anchor=False)
        rk.add("weg2-4-1", "live-c", end_anchor=False)
        rk.add("weg2-4-1", "live-d", end_anchor=False, parent=a)
    assert _rounds(ranks)
    for rk in ranks:   # a running request holds fork-a and live-d (device lock)
        for h in ("fork-a", "live-d"):
            node = rk.cache.root_node.children.get(h) or rk.cache.root_node.children["fork-a"].children[h]
            node.component_data[ComponentType.FULL].lock_ref = 1
        rk.add("weg2-4-9", "new-end")
    _handoff("weg2-4-9", "new-end")
    assert _rounds(ranks)
    assert _state(arena, "new-end") == 2
    assert _state(arena, "fork-a") == 2 and _state(arena, "live-d") == 2
    for rk in ranks:
        assert rk.cache.root_node.children["fork-a"].component_data[M].host_value is not None
        assert rk.cache.root_node.children["fork-b"].component_data[M].host_value is None


@pytest.mark.parametrize("env", [
    {"SGLANG_WEG2_DUAL_LAYOUT": "0"},                  # flip layout: the reset releases
    {"SGLANG_WEG2_GROUP": "D"},                        # D: no P tree to release
    {"SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE": "0"},   # the switch
])
def test_inert_off_dual_p(dual_p, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    arena = _arena(dual_p)
    ranks = [_Rank(arena, r) for r in range(RANKS)]
    for k in range(SLOTS):
        ok, _ = _finish(ranks, f"weg2-5-{k}")
        assert ok
    ok, _ = _finish(ranks, "weg2-5-99")
    assert not ok                                       # the pre-fix claim, byte for byte
    assert [rk.held() for rk in ranks] == [SLOTS] * RANKS


def test_ended_reads_the_front_tombstone(dual_p):
    assert not hp.ended("weg2-6-0")
    hp.drop("weg2-6-0", "reroute_fresh")
    assert not hp.ended("weg2-6-0")
    hp.drop("weg2-6-0", "status_200")
    assert hp.ended("weg2-6-0")
    with envs.SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE.override(False):
        from sglang.srt.weg2 import dual_anchor_release as dar
        assert not dar.armed()
