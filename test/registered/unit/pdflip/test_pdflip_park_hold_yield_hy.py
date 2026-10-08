"""HY: a D park backup the full L2 refuses takes the L3-copied pages of a held request.

Metal (NF y3w e033a931db, D 01:39:13): ``park_running`` retracted pdflip-1-4
(121920 tokens) with a forced host write-through; the arena (6485 slots) was
full of the span pdflip-14-25's completed read had pinned (a 246k-token request
held on D, not running) -- the park's claim came back 18 slots short
(``ARENA-DROP need=18 freed=6``, ``#1421 BACKUP-REFUSED why=arena_claim``).
The sleep dropped the park's device KV, the wake could not resume it (W50
x_refusal_midstream), P re-prefilled 125145 tokens with cached=0: 30.6 s.
y3u 5bedac26f1 00:36:46 the same with need=469 (P: 130764 tokens, 32.0 s).

Here a real C arena stands for L2: the held span's pages are COMPLETE and
referenced (its tree host nodes), a refused park node lacks slots. With every
held page on disk the vote gives the held span back, the backup goes through
and the held request's read records are gone (the wake reads it from L3); with
fewer on-disk pages than the need nothing moves and the marker names it.
RED on a332187f28 (the module does not exist: nothing yields), GREEN with HY.
"""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

SLOT = 1024
SLOTS = 12
HELD = 10          # the held request's pinned span (pdflip-14-25's pages)
PARK = 4           # the park node's pages (pdflip-1-4); free = 2 -> need = 2


class _Node:
    def __init__(self, nid, slots, stems, parent=None):
        self.id = nid
        self.parent = parent
        self.children = {}
        self.backuped = True
        self.evicted = True
        self.hash_value = [s for s in stems]
        cd = types.SimpleNamespace(host_value=torch.tensor(slots, dtype=torch.int64), host_lock_ref=1)
        self.component_data = [cd, types.SimpleNamespace(host_value=None, host_lock_ref=0),
                               types.SimpleNamespace(host_value=None, host_lock_ref=0)]


class _Backend:
    def __init__(self, on_disk):
        self.on_disk = set(on_disk)

    def _stat_stems(self, stems):
        return {s: SLOT for s in stems if s in self.on_disk}

    def _suffix_for_key(self, key):
        return ("",)


class _Pool:
    """An arena host pool with page size 1 and no staging rows: host row = slot."""

    staging_rows = 0
    page_size = 1
    _pending_mask = None
    _pending = None

    def __init__(self, arena):
        self.arena = arena
        self.arena_slots = SLOTS
        self.arena_tokens = SLOTS


class _Tree:
    """The UnifiedRadixCache surface HY uses, over a real arena."""

    _pdflip_park_track = None

    def __init__(self, arena, on_disk, peer_vote=None):
        self.arena = arena
        self.pool = _Pool(arena)
        self.cache_controller = types.SimpleNamespace(storage_backend=_Backend(on_disk))
        self._prefetch_span_pins = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self._prefetch_completed_tokens = {}
        self._pdflip_dormant_done = {}
        self.peer_vote = peer_vote
        self.backed = []
        self.votes = 0

    def _pdflip_arena_pools(self):
        return {ComponentType.FULL: self.pool}

    def _all_reduce_attn_groups(self, t, op, label=""):
        self.votes += 1
        if self.peer_vote is not None:
            t.copy_(torch.minimum(t, torch.tensor(self.peer_vote(t.tolist()), dtype=t.dtype)))

    def _barrier_attn_groups(self, label=""):
        pass

    def pdflip_node_depth(self, node):
        return 0

    def _unpin_prefetched_span(self, rid):
        for node, _p in self._prefetch_span_pins.pop(rid, ()):
            node.component_data[0].host_lock_ref = 0

    def _is_host_leaf(self, node):
        return node.evicted and node.backuped and not node.children and \
            all(cd.host_lock_ref == 0 for cd in node.component_data)

    def _evict_host_leaf(self, node, tracker):
        slots = node.component_data[0].host_value.tolist()
        self.arena.ref_slots(slots, -1)  # the tree's reader reference goes
        node.component_data[0].host_value = None
        if node.parent is not None:
            node.parent.children.pop(node.id, None)

    def write_backup(self, node):
        """The claim with its room-making (#1427 stage i: unreferenced
        COMPLETE slots, here all on disk) -- refused when it does not fit."""
        got = self.arena.claim_slots(node.hash_value, [SLOT] * len(node.hash_value))
        short = [i for i, (_s, st, _g) in enumerate(got) if st == 4]
        if short:
            cands = self.arena.evict_candidates(len(short))
            self.arena.free_slots([c[0] for c in cands], reason="claim_room")
            again = self.arena.claim_slots([node.hash_value[i] for i in short], [SLOT] * len(short))
            if any(st == 4 for _s, st, _g in again):
                fresh = [s for s, st, g in got + again if st == 0]
                self.arena.free_slots(fresh, reason="claim_refused")
                self._1421(node)
                return 0
        node.backuped = True
        self.backed.append(node.id)
        return len(node.hash_value)

    def _1421(self, node):
        if self._pdflip_park_track is not None:
            self._pdflip_park_track[node.id] = node


def _setup(tmp_path, on_disk_n, peer_vote=None):
    arena = ShmArena(str(tmp_path / "kv.bin"), SLOT, SLOTS)
    held_stems = ["h%02d" % i for i in range(HELD)]
    buf = ctypes.create_string_buffer(SLOT)
    for st in held_stems:
        assert arena.write([st], [SLOT], [((0, SLOT),)], [ctypes.addressof(buf)]) == [1]
    slots = [s for s, _st in arena.find_slots(held_stems)]
    assert arena.ref_slots(slots, +1) == HELD  # the tree's references
    tree = _Tree(arena, on_disk=held_stems[:on_disk_n], peer_vote=peer_vote)
    root = _Node(0, [], [])
    a = _Node(1, slots[:5], held_stems[:5], parent=root)
    b = _Node(2, slots[5:], held_stems[5:], parent=a)
    a.children = {2: b}
    tree._prefetch_span_pins["pdflip-14-25"] = [(b, None), (a, None)]  # deepest first
    tree.prefetch_loaded_tokens_by_reqid["pdflip-14-25"] = HELD
    park = _Node(9, [], ["p%d" % i for i in range(PARK)])
    park.backuped = False
    return tree, park


def _park(tree, park):
    """park_running's order: record, retract (the forced backup), vote."""
    from flliper.srt.pdflip import park_hold_yield as hy

    hy.begin(tree)
    assert tree.write_backup(park) == 0, "the metal's refusal: the arena is full"
    parked = [types.SimpleNamespace(rid="pdflip-1-4"), types.SimpleNamespace(rid="pdflip-14-25")]
    retracted = [parked[0]]
    sched = types.SimpleNamespace(tree_cache=tree)
    return hy.settle(sched, retracted=retracted, parked=parked), parked


def test_y3w_the_refused_park_backup_takes_the_held_span_with_an_l3_copy(tmp_path, caplog):
    """RED on a332187f28 (no yield: the park stays unbacked, P recomputes it).
    GREEN: the held span goes back, the park is backed, the held request's
    read records are gone -- its wake reads it from L3."""
    caplog.set_level(logging.WARNING)
    tree, park = _setup(tmp_path, on_disk_n=HELD)
    verdict, parked = _park(tree, park)
    assert verdict == "yield"
    assert park.backuped and tree.backed == [9]
    assert "pdflip-14-25" not in tree.prefetch_loaded_tokens_by_reqid
    assert getattr(parked[1], "_pdflip_hold_yielded", False)
    line = [m for m in caplog.messages if "PARK-BACKUP REFUSED" in m][-1]
    assert "need=2 " in line and "held_by_hold=10 " in line and "held_on_disk=10 " in line
    assert "verdict=yield" in line


def test_need_above_the_on_disk_pages_changes_nothing(tmp_path, caplog):
    """A held span without a whole L3 copy is never given back: the refusal
    stands as today, the held read and its pages stay."""
    caplog.set_level(logging.WARNING)
    tree, park = _setup(tmp_path, on_disk_n=HELD - 1)
    verdict, _parked = _park(tree, park)
    assert verdict == "stands"
    assert not park.backuped
    assert tree.prefetch_loaded_tokens_by_reqid["pdflip-14-25"] == HELD
    line = [m for m in caplog.messages if "PARK-BACKUP REFUSED" in m][-1]
    assert "held_on_disk=9 " in line and "verdict=stands" in line


def test_a_peer_rank_without_the_copy_vetoes_the_group(tmp_path):
    """Rank-uniform: this rank sees every page on disk, a peer does not --
    the MIN vote says no on every rank, nothing moves here either."""
    from flliper.srt.pdflip import park_hold_yield as hy

    def peer(vec):
        refused, need, on_disk, whole = hy.unpack_vote(vec)
        return hy.pack_vote(refused=refused, need=need, on_disk=on_disk, whole=[False] * len(whole))

    tree, park = _setup(tmp_path, on_disk_n=HELD, peer_vote=peer)
    verdict, _ = _park(tree, park)
    assert verdict == "stands" and not park.backuped


def test_a_clean_park_votes_once_and_moves_nothing(tmp_path):
    tree, _park_node = _setup(tmp_path, on_disk_n=HELD)
    from flliper.srt.pdflip import park_hold_yield as hy

    hy.begin(tree)
    parked = [types.SimpleNamespace(rid="pdflip-14-25")]
    assert hy.settle(types.SimpleNamespace(tree_cache=tree), retracted=[], parked=parked) == "clean"
    assert tree.votes == 1 and tree.prefetch_loaded_tokens_by_reqid["pdflip-14-25"] == HELD


def test_the_switch_off_keeps_todays_refusal(tmp_path):
    tree, park = _setup(tmp_path, on_disk_n=HELD)
    with envs.FLLIPER_PDFLIP_ENABLE_PARK_HOLD_YIELD.override(False):
        verdict, _ = _park(tree, park)
    assert verdict is None and not park.backuped


def test_the_tree_records_a_park_backup_the_arena_refused():
    """The wiring: UnifiedRadixCache._1421_refused (the #1421 arena_claim
    site) files the node while a park records -- and only then, and only
    for arena_claim (a mamba_pin refusal is not a room problem)."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree = object.__new__(UnifiedRadixCache)
    node = types.SimpleNamespace(id=7)
    tree._1421_refused("arena_claim", node)  # no park recording: nothing filed
    assert tree._pdflip_park_track is None
    tree._pdflip_park_track = {}
    tree._1421_refused("mamba_pin", node)
    tree._1421_refused("arena_claim", node)
    assert tree._pdflip_park_track == {7: node}



def test_flip_edge_a_clean_park_never_stats_the_held_spans(tmp_path, monkeypatch):
    """FLIP-EDGE (N5d epoch 8, 12:52:29): nothing refused -> one vote, NO held_facts (it stat'ed
    169224 L3 stems of the held pdflip-6-10: rest_ms=1099 on every rank, verdict clean)."""
    tree, _park_node = _setup(tmp_path, on_disk_n=HELD)
    from flliper.srt.pdflip import park_hold_yield as hy

    calls = []
    monkeypatch.setattr(hy, "held_facts", lambda t, rid: calls.append(rid) or (0, 0))
    hy.begin(tree)
    parked = [types.SimpleNamespace(rid="pdflip-14-25")]
    assert hy.settle(types.SimpleNamespace(tree_cache=tree), retracted=[], parked=parked) == "clean"
    assert calls == [] and tree.votes == 1
