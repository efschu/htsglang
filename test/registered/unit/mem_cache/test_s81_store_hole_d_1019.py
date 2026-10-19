"""S81 orphan store hole on D (rc17 boot ..._7d93470b5e_1019_181004, pdflip-8-209, 19.10.).

Metal: D TP0 acked pages 85568..88192 (41) as share-0 direct writes at 18:26:31-35 (``#1469 EVICT ... backuped=True``
on TP0); the page was incomplete, S74 sent ONE PUBLISH; TP1/TP2 refused it (``#1421 BACKUP-REFUSED why=mamba_pin
node=246 pins=5/5``, every child ``why=parent_unbacked``) -- a transient, node 246 was backed at 18:26:36. Nobody asked
again: the S74 entry left the wait list when TP0's node went another way (0 ``CUT-L3-RESET`` in the boot), the claims
stayed CLAIMED/open0 for 8 min (``ARENA-REF-CENSUS-CLAIM claimed=41 ... open0``), the store chain ended at page 1337,
P recomputed 16482 tokens.

Now: an incomplete share-0 page stays watched by its stems; a worker still holding it on its device gets PUBLISH again
(pass-count backoff); with no owner copy left TP0 gives its host layer back and frees its orphan claim. REAL C arena.
RED on rc17 7d93470b5e: ``test_metal_*`` (the entry leaves, no second PUBLISH, the slot stays open0).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

SLOT = 4096
HALF = SLOT // 2
PAGE = 64
S81 = "FLLIPER_PDFLIP_S81_HOLE_RESOLVE"


class _Backend:
    @staticmethod
    def _get_suffixed_key(h):
        return f"{h}.kv"


class _Tree:
    """TP0's tree: the real verdict and completeness probe over a real arena."""
    _k12_direct_l3_verdict = U._k12_direct_l3_verdict
    _s74_cut_page_complete = U._s74_cut_page_complete

    def __init__(self, arena, republish=None):
        self.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(arena=arena), storage_backend=_Backend())
        self._arena = arena
        self.republished = []
        if republish is not None:
            self._k12_republish_orphaned = lambda node, n: (self.republished.append(node.id), republish(node))[1]

    def _pdflip_direct_pool(self):
        return SimpleNamespace(arena=self._arena)


def _node(hashes=("p1337", "p1338"), nid=243):
    root = SimpleNamespace(parent=None, children={})
    cd = SimpleNamespace(host_value=torch.arange(len(hashes) * PAGE), value=None, host_lock_ref=0)
    n = SimpleNamespace(id=nid, parent=root, key=list(range(len(hashes) * PAGE)), children={},
                        component_data=[cd, cd], hash_value=list(hashes), l3_present=False)
    root.children[0] = n
    return n


@pytest.fixture
def arena(tmp_path):
    a = ShmArena(str(tmp_path / "arena-786432.bin"), SLOT, 16)
    yield a
    a.close()


@pytest.fixture
def host_cut(monkeypatch):
    R._reset_state_for_tests()
    monkeypatch.setattr(R, "role", lambda: "host")
    monkeypatch.setattr(R, "cut_role", lambda: "host")
    monkeypatch.setattr(R, "l3_group_on", lambda env=None: True)
    monkeypatch.setattr(R, "note_k21_ack", lambda node, orphaned: None)
    for env in ("FLLIPER_PDFLIP_CUT_L3_COMPLETE", "FLLIPER_PDFLIP_L3_PRESENT_STRICT", "FLLIPER_PDFLIP_CUT_L3_PUBLISH",
                S81):
        monkeypatch.delenv(env, raising=False)
    yield
    R._reset_state_for_tests()


def _tp0_share0(arena, hashes):
    """TP0 (share 0): claims fresh and merges NO extent -> CLAIMED, open 0, covered 0 (the metal 'open0')."""
    for h in hashes:
        (s, st, g), = arena.claim_slots([f"{h}.kv"], [SLOT])
        assert st == 0
        assert arena.complete_slots([s], [g], []) == [0]


def _owners_write(arena, hashes):
    for h in hashes:
        for off in (0, HALF):
            (s, st, g), = arena.claim_slots([f"{h}.kv"], [SLOT])
            assert st in (0, 1)
            arena.complete_slots([s], [g], [(off, HALF)])


def _worker_set(arena, keys, pid=None):
    """A worker's S22 key set next to the arena (``dv`` = device-resident keys)."""
    path = str(arena.path) + R.DEV_SUFFIX + "." + str(pid or os.getpid())
    with open(path, "w") as f:
        json.dump({"ev": [], "nd": [], "dv": [[h, d] for h, d in keys]}, f)


def _events(verdicts, kind):
    return [e for v in verdicts if isinstance(v, R.FormAHostVerdict) for e in v.events if e[0] == kind]


def _key(n):
    return (n.hash_value[-1], len(n.key))


def _metal(arena):
    """18:26:31-35: TP0 acks the incomplete share-0 page, S74 waits and PUBLISHes once; the workers refuse."""
    t, n = _Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    assert len(_events(R.attach([]), R.PUBLISH)) == 1   # S74's one PUBLISH (refused on the workers)
    n.component_data[0].host_value = None              # TP0's node goes another way (host rows gone)
    return t, n


def test_metal_worker_still_holds_the_page_gets_publish_again_and_the_page_completes(arena, host_cut):
    t, n = _metal(arena)
    _worker_set(arena, [_key(n)])                       # the workers still hold it on their device
    again = _events(R.attach([]), R.PUBLISH)
    assert again == [(R.PUBLISH, n.hash_value[-1], len(n.key), 1, 1, 0)], "the owners are asked again"
    _owners_write(arena, n.hash_value)                  # pins drained: the owners join TP0's open slot
    R.attach([])
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [2, 2], "the hole closed"
    assert not R._S.l3_wait


def test_metal_no_owner_copy_left_revokes_tp0s_orphan_claim(arena, host_cut, caplog):
    t, n = _metal(arena)
    _worker_set(arena, [])                              # no worker holds it any more
    with caplog.at_level("WARNING"):
        R.attach([])
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [0, 0], "no orphan CLAIMED/open0 slot is left"
    assert not R._S.l3_wait
    assert "S81 HOLE-REVOKE" in caplog.text
    (s, st, g), = arena.claim_slots(["p1337.kv"], [SLOT])
    assert st == 0, "the next write is a fresh claim, not a join on an orphan"


def test_unknown_worker_state_never_revokes(arena, host_cut):
    """No readable worker key set: TP0 asks again, it never frees on a guess."""
    t, n = _metal(arena)
    assert len(_events(R.attach([]), R.PUBLISH)) == 1
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [1, 1]
    assert R._S.l3_wait


def test_publish_backoff_is_a_pass_count(arena, host_cut):
    t, n = _metal(arena)
    _worker_set(arena, [_key(n)])
    sent = [len(_events(R.attach([]), R.PUBLISH)) for _ in range(40)]
    at = [i + 1 for i, x in enumerate(sent) if x]
    gaps = [b - a for a, b in zip(at, at[1:])]
    assert at[0] == 1 and gaps == sorted(gaps) and max(gaps) <= R.S81_PUBLISH_MAX_GAP
    assert sum(sent) < 10, "bounded: not one PUBLISH per broadcast"
    assert all(x in (0, 1) for x in sent)


def test_attached_node_gives_its_host_layer_back_before_the_claim_goes(arena, host_cut):
    t = _Tree(arena, republish=lambda node: setattr(node.component_data[0], "host_value", None) or True)
    n = _node()
    _tp0_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    R.attach([])
    _worker_set(arena, [])
    R.attach([])
    assert t.republished == [n.id] and n.component_data[0].host_value is None
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [0, 0] and not R._S.l3_wait


def test_host_rows_tp0_cannot_give_back_are_held_and_the_slot_stays(arena, host_cut, caplog):
    t = _Tree(arena, republish=lambda node: False)       # host lock / device copy gone
    n = _node()
    _tp0_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    R.attach([])
    _worker_set(arena, [])
    with caplog.at_level("INFO"):
        R.attach([])
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [1, 1] and R._S.l3_wait
    assert "held=1" in caplog.text


def test_a_joined_or_open_slot_is_never_freed(arena, host_cut):
    t, n = _metal(arena)
    (s, st, g), = arena.claim_slots(["p1337.kv"], [SLOT])   # a worker joined, its write still open
    assert st == 1
    _worker_set(arena, [])
    R.attach([])
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [1, 1], "kept: a writer is on it"
    assert R._S.l3_wait


def test_reset_names_what_really_stayed_open(arena, host_cut, caplog):
    t, n = _metal(arena)
    R.attach([])
    with caplog.at_level("WARNING"):
        R.on_tree_reset(None)
    assert "S74 CUT-L3-RESET open=1" in caplog.text and not R._S.l3_wait


def test_switch_off_is_the_s74_leave(arena, host_cut, monkeypatch):
    monkeypatch.setenv(S81, "0")
    t, n = _metal(arena)
    _worker_set(arena, [_key(n)])
    assert _events(R.attach([]), R.PUBLISH) == [] and not R._S.l3_wait
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [1, 1], "the S74 form: the orphan stays"


def test_parents_are_republished_before_children(arena, host_cut):
    """The metal refusal was parent_unbacked: in one broadcast the worker applies the parent's PUBLISH first."""
    t = _Tree(arena)
    parent = _node(("p1337",), nid=246)
    child = SimpleNamespace(id=254, parent=parent, key=list(range(PAGE)), children={},
                            component_data=[SimpleNamespace(host_value=torch.arange(PAGE), value=None,
                                                            host_lock_ref=0)] * 2,
                            hash_value=["p1338"], l3_present=False)
    parent.children[1] = child
    for nd in (child, parent):                           # TP0 acked the child first
        _tp0_share0(arena, nd.hash_value)
        t._k12_direct_l3_verdict(nd, 0)
    R.attach([])
    _worker_set(arena, [R.node_key(parent), R.node_key(child)])
    pubs = _events(R.attach([]), R.PUBLISH)
    assert [e[2] for e in pubs] == [PAGE, 2 * PAGE]
