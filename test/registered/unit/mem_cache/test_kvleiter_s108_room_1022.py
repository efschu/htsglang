"""KVLEITER-1022 (NF rc32 ded00bd5a7, boot ..._1022_030836 D): the turn end makes its own stage-(i) arena room.

THE METAL. From 03:17:29 to 03:33:47 (between two flips, D under decode load) every S108 turn end deferred its chain:
``PDFLIP PUBLISH-SWEEP ... refused_by=anchor_only_claim:265`` / ``arena_room:679,parent_unbacked:4641``, up to 798
un-backed device nodes. The KV arena had 6485 slots, 5973 COMPLETE, 1274 referenced (03:23:31 ARENA-REF-CENSUS);
the mamba arena 84 / 84 / 60. No FREE slot, and the turn end claims only FREE room (S108 review A2: the claim's
room-making wrote to disk then). S109d (later the same day) made that room-making write- and stat-free. KV-COMPACT
never evicts what has no copy, so the cached content above S5 was child-blocked behind the un-backed tails
(HOLDERS S5 03:19:21: store_child_blocked=1176 l2_child_blocked=56 unbacked=132 of 1382 idle pages, 18 could go)
and the KV ladder held S8 (r3080 >= 26 expert rows only from S5 down).

THE RULE. Once per turn end and pool kind the turn end frees up to its own demand through the clock's stage (i) only:
COMPLETE, referenced by no rank, kept by nobody (no P hand-off, no D park), with its L3 copy. Nothing is written,
nothing waits, no page without an L3 copy leaves L2, no tree node changes. CPU only.
"""
from __future__ import annotations

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402

PAGE = 64
L3_ASYNC_REAL = True   # the S109 conftest seam must keep the scheduler forms: this module tests what they free
_needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (real C arena)")


class _Pool:
    """A KV arena pool as the S108 claim sees it: free slots, a room-maker, a claim."""

    def __init__(self, free=0, room=0):
        self.free, self.room = free, room
        self.made, self.claimed = [], []

    def claim_needs_room(self, hashes):
        return len(hashes) > self.free

    def make_secured_room(self, need):
        self.made.append(int(need))
        got = min(int(need), self.room)
        self.room -= got
        self.free += got
        return got

    def alloc_write(self, hashes):
        assert len(hashes) <= self.free, "claimed beyond the free room: the claim would make room itself"
        self.free -= len(hashes)
        self.claimed.append(list(hashes))
        return torch.arange(len(hashes))


def _node(nid, pages):
    return types.SimpleNamespace(id=nid, hash_value=["h%d_%d" % (nid, i) for i in range(pages)], component_data={
        urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(value=torch.arange(pages * PAGE))})


def _turn_end_stub(pool, need=None, tried=True):
    why = []
    stub = types.SimpleNamespace(_s108_turn_end_active=True, page_size=PAGE, cache_controller=None,
                                 _pdflip_direct_pool=lambda: pool, _1421_refused=lambda w, n: why.append(w))
    if tried:
        stub._s108_room_tried = set()
    if need is not None:
        stub._s108_room_need = need
    return stub, why


def test_rc32_the_turn_end_makes_its_stage_i_room_then_claims():
    """rc32 03:19:21: no FREE slot, COMPLETE unreferenced secured slots beside them. The turn end frees its own
    demand (the chain's 5 un-backed pages) once and claims -- no arena_room deferral."""
    pool = _Pool(free=0, room=100)
    stub, why = _turn_end_stub(pool, need={"kv": 5, "mamba": 0})
    rows = urc.UnifiedRadixCache._pdflip_direct_claim(stub, _node(1, 3))
    assert rows is not False and rows is not None, why
    assert pool.made == [5] and why == []
    # the second node of the same turn end claims from the room made for the whole chain
    rows2 = urc.UnifiedRadixCache._pdflip_direct_claim(stub, _node(2, 2))
    assert rows2 is not False and pool.made == [5] and why == []


def test_room_is_made_once_per_turn_end_then_the_node_is_deferred_by_name():
    """The census (S109 review R10: ~5-7 ms CPU on a 6485-slot arena) runs at most once per turn end and kind; what
    the room did not pay is deferred by name, as before."""
    pool = _Pool(free=0, room=3)
    stub, why = _turn_end_stub(pool, need={"kv": 3})
    assert urc.UnifiedRadixCache._pdflip_direct_claim(stub, _node(1, 3)) is not False
    assert urc.UnifiedRadixCache._pdflip_direct_claim(stub, _node(2, 2)) is False
    assert pool.made == [3] and why == ["arena_room"] and stub._pdflip_sweep_last_refusal == "arena_claim"


def test_room_that_does_not_pay_the_node_defers_without_a_claim():
    pool = _Pool(free=0, room=1)
    stub, why = _turn_end_stub(pool, need={"kv": 1})
    assert urc.UnifiedRadixCache._pdflip_direct_claim(stub, _node(1, 2)) is False
    assert pool.made == [2] and pool.claimed == [] and why == ["arena_room"]


@pytest.mark.parametrize("case", ["outside_the_sweep", "no_room_maker"])
def test_no_room_outside_a_turn_end_sweep_or_without_a_room_maker(case):
    """Fail closed: a stub without the turn end's room state (``_s108_room_tried``) or a pool without
    ``make_secured_room`` claims FREE room only -- the S108 behaviour byte for byte."""
    pool = _Pool(free=0, room=100)
    if case == "no_room_maker":
        pool.make_secured_room = None
    stub, why = _turn_end_stub(pool, need={"kv": 5}, tried=(case != "outside_the_sweep"))
    assert urc.UnifiedRadixCache._pdflip_direct_claim(stub, _node(1, 3)) is False
    assert why == ["arena_room"] and pool.claimed == []


def test_the_turn_end_anchor_claim_takes_stage_i_room_once_and_never_h19():
    stub = types.SimpleNamespace(_s108_turn_end_active=True, _s108_room_tried=set(), _s108_room_need={"mamba": 2})
    stub._pdflip_mamba_claim_h19 = lambda *a: (_ for _ in ()).throw(AssertionError("H19 at a turn end"))
    stub._s108_mamba_free_claim = lambda mp, h: urc.UnifiedRadixCache._s108_mamba_free_claim(stub, mp, h)
    mp = _Pool(free=0, room=2)
    a = urc.UnifiedRadixCache._pdflip_mamba_claim(stub, object(), mp, "h1")
    b = urc.UnifiedRadixCache._pdflip_mamba_claim(stub, object(), mp, "h2")
    c = urc.UnifiedRadixCache._pdflip_mamba_claim(stub, object(), mp, "h3")
    assert a is not None and b is not None and c is None
    assert mp.made == [2] and stub._s108_mamba_room_n == 1


def test_the_turn_end_clears_its_room_state():
    """The room state lives only inside the turn end's sweep: a later claim (chunk publish, eviction) never sees
    it -- those make their room the way they always did."""
    import inspect

    src = inspect.getsource(urc.UnifiedRadixCache._s108_turn_end_l2)
    i_set = src.index("self._s108_room_tried = set()")
    i_on = src.index("self._s108_turn_end_active = True")
    i_fin = src.index("finally:", i_on)
    i_clear = src.index("self._s108_room_tried = None", i_fin)
    assert i_set < i_on < i_fin < i_clear


def test_room_demand_counts_unbacked_device_pages_and_device_anchors():
    M = urc.ComponentType.MAMBA

    def n(pages, backed=False, store=False, on_dev=True, anchor=None):
        cds = {urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(value=torch.arange(4) if on_dev else None)}
        cds[M] = types.SimpleNamespace(value=None if anchor is None else torch.arange(1),
                                       host_value=torch.arange(1) if anchor == "host" else None)
        cdl = [None] * (int(max(int(M), int(urc.BASE_COMPONENT_TYPE))) + 1)
        for k, v in cds.items():
            cdl[int(k)] = v
        return types.SimpleNamespace(component_data=cdl, backuped=backed, l3_present=store,
                                     hash_value=["x"] * pages)

    stub = types.SimpleNamespace(tree_components=(urc.BASE_COMPONENT_TYPE, M))
    nodes = [n(3, anchor="dev"), n(2, backed=True, anchor="dev"), n(4, store=True), n(5, on_dev=False),
             n(1, anchor="host"), n(6)]
    assert urc.UnifiedRadixCache._s108_room_demand(stub, nodes) == {"kv": 3 + 1 + 6, "mamba": 1}


# ------------------------------------------------------------------------- the real C arena: what stage (i) frees
def _arena(tmp_path, n=6):
    from test_s109_backup_wall_aux_leaves_1020 import _unreferenced_full_arena

    return _unreferenced_full_arena(tmp_path, n)


@_needs_gcc
def test_make_secured_room_frees_only_unreferenced_pages_with_their_l3_copy(tmp_path):
    """The danger direction (0 avoidable tokens): a page without its L3 copy or one any rank still references never
    leaves L2. 6 COMPLETE pages, 3 with a copy, one of those referenced: exactly 2 are freed."""
    from test_s109_backup_wall_aux_leaves_1020 import _write_behind

    p, arena, slots = _arena(tmp_path)
    _write_behind(arena, p._backend._l3idx, slots[:3])
    assert arena.ref_slots([slots[0]], +1) == 1               # a reader holds the first copied page
    assert p.make_secured_room(6) == 2
    st = arena.slot_states(slots)
    assert st[0] == 2, "a referenced page stays"
    assert st[1] != 2 and st[2] != 2, "the unreferenced copied pages left L2 (their copy is in L3)"
    assert st[3:] == [2, 2, 2], "pages without an L3 copy stay in L2"


@_needs_gcc
def test_make_secured_room_never_takes_a_kept_page_and_never_enters_stage_ii_or_iii(tmp_path, monkeypatch):
    """A P hand-off or D park page (``handoff_pending`` keep) is no turn-end room, copy or not; the full claim
    (stage_max 3) would reach stage (ii) for it."""
    from test_s109_backup_wall_aux_leaves_1020 import _write_behind
    from flliper.srt.pdflip import handoff_pending as hp

    p, arena, slots = _arena(tmp_path)
    _write_behind(arena, p._backend._l3idx, slots)
    sl, _g, klo, _khi = arena.complete_census()
    lo = {int(s): int(k) for s, k in zip(np.asarray(sl).tolist(), np.asarray(klo, dtype=np.uint64).tolist())}
    kept = np.asarray(sorted([lo[slots[0]], lo[slots[1]]]), dtype=np.uint64)
    keep = hp.Keep(kept, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), ["rid-k"], [[]])
    monkeypatch.setattr(hp, "keep_for", lambda pool: keep)
    entered = []
    monkeypatch.setattr(hp, "copied_mask", lambda *a, **k: entered.append("ii") or np.ones(2, dtype=bool))
    monkeypatch.setattr(hp, "evict_ordered", lambda *a, **k: entered.append("ii/iii") or [])
    assert p.make_secured_room(6) == 4
    assert entered == []
    assert arena.slot_states(slots[:2]) == [2, 2], "kept pages stay"
    p._evict_for_claim(arena, 1)                               # the full claim, for contrast
    assert entered, "the full claim reaches stage (ii) when stage (i) is short"
