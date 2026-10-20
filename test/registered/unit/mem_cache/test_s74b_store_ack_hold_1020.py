"""S74b STORE-ACK-HOLD (rc28 boot ..._297198745d_1020_174205, pdflip-4-223 ANKER-VERFEHLT-P, 20.10.).

Metal (D.log): TP0 direct-wrote share 0 of the session prefix pages 1332..1368 ([85248, 87616); TP0 nodes 447/437,
the workers' node 457) at 18:10:19. The page was incomplete -- the workers' claim was refused (``#1421 BACKUP-REFUSED
why=arena_claim node=457``, KV arena full). The direct ack's S74 CUT-L3-WAIT sent its PUBLISH; in the SAME ack drain the
node's plain-sidecar (QSA index) store write acked and ran the generic store-ack verdict: ``_pdflip_rebind_host_to_arena``
answered True for rows that ARE arena rows ("already arena rows", no COMPLETE check) -> REBIND + L3 + ``l3_present``.
TP1/TP2 in that one broadcast: ``S74 CUT-L3-PUBLISH out=refused node=457``, ``R12 CUT-REBIND-MISS``, and the L3 event
(``l3_present`` -> no sweep wrote node 457 again: 0 BACKUP-REFUSED for it after 18:10:19). TP0's S81 entry left
silently (0 ``S81 HOLE-*`` lines in the boot). The claims stayed CLAIMED/open0 (``claimed=42 oldest=950/55575ms ...
open0``) to the sleep 18:13:12 (``L3-COMPLETE-ONLY incomplete=30 of 64`` / ``7 of 7``, ``PARK-DEMOTE-INCOMPLETE
absent=37 busy=37``), the KV chain ended at page 1332, P recomputed 17690 tokens of pdflip-4-223.

The REAL ``_drain_storage_control_queues_impl`` drains the store ack here, over a REAL C arena.
RED on rc28 297198745d: ``test_store_ack_of_an_incomplete_direct_page_holds`` and ``test_metal_*``.
"""
from __future__ import annotations

import json
import os
from queue import Queue
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
STAGING = 1000          # host rows >= STAGING are arena rows (a direct write)
S74B = "FLLIPER_PDFLIP_S74B_STORE_ACK_HOLD"


class _Backend:
    @staticmethod
    def _get_suffixed_key(h):
        return f"{h}.kv"


class _Tp0Tree:
    """TP0's tree: the REAL ack drain, rebind, S74b hold, S74 probe and direct verdict; the rest is bookkeeping."""
    _drain_storage_control_queues_impl = U._drain_storage_control_queues_impl
    _pdflip_rebind_host_to_arena = U._pdflip_rebind_host_to_arena
    _k12_direct_l3_verdict = U._k12_direct_l3_verdict
    _s74_cut_page_complete = U._s74_cut_page_complete
    enable_storage_metrics = False
    storage_metrics_collector = None
    staging_write_ring = None
    page_size = PAGE

    def __init__(self, arena):
        self._arena = arena
        pool = SimpleNamespace(arena=arena, arena_read=True, staging_rows=STAGING, page_size=PAGE,
                               slot_used=None)
        self.released = []
        self.cache_controller = SimpleNamespace(
            mem_pool_host=pool, storage_backend=_Backend(), ack_backup_queue=Queue(),
            prefetch_revoke_queue=Queue(), host_mem_release_queue=Queue(),
            append_host_mem_release=lambda *a, **k: self.released.append(k))
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self.locks = []

    def dec_host_lock_ref(self, node, lock_params):
        self.locks.append(lock_params)

    def _pdflip_direct_pool(self):
        return SimpleNamespace(arena=self._arena)

    def _pdflip_host_is_transit(self):
        return False

    def store_ack(self, node, op_id=7, completed=None):
        """The node's store operation (the plain-sidecar write of a direct-written node) acks and is drained."""
        pages = len(node.hash_value)
        op = SimpleNamespace(id=op_id, hash_value=list(node.hash_value),
                             completed_tokens=pages * PAGE if completed is None else completed)
        self.ongoing_backup[op_id] = (node, ("lp", op_id))
        self.cache_controller.ack_backup_queue.put(op)
        self._drain_storage_control_queues_impl(0, None, 0, None, False)


def _node(hashes=("p1361", "p1362"), nid=437, arena_rows=True):
    root = SimpleNamespace(parent=None, children={})
    base = STAGING if arena_rows else 0
    cd = SimpleNamespace(host_value=torch.arange(base, base + len(hashes) * PAGE), value=object(), host_lock_ref=0)
    n = SimpleNamespace(id=nid, parent=root, key=list(range(len(hashes) * PAGE)), children={},
                        component_data=[cd, cd], hash_value=list(hashes), l3_present=False, backuped=True,
                        _pdflip_chain_piece=False)
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
                "FLLIPER_PDFLIP_S81_HOLE_RESOLVE", S74B):
        monkeypatch.delenv(env, raising=False)
    yield
    R._reset_state_for_tests()


def _tp0_share0(arena, hashes):
    """TP0 (share 0): claims fresh, merges NO extent -> CLAIMED, open 0, covered 0 (the metal 'open0')."""
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
    path = str(arena.path) + R.DEV_SUFFIX + "." + str(pid or os.getpid())
    with open(path, "w") as f:
        json.dump({"ev": [], "nd": [], "dv": [[h, d] for h, d in keys]}, f)


def _events(verdicts, kind):
    return [e for v in verdicts if isinstance(v, R.FormAHostVerdict) for e in v.events if e[0] == kind]


def _key(n):
    return (n.hash_value[-1], len(n.key))


def _metal(arena, monkeypatch=None):
    """18:10:19: TP0's direct ack (S74 CUT-L3-WAIT + its PUBLISH), then the sidecar store ack in the same drain."""
    t, n = _Tp0Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)            # the direct-write ack: incomplete -> await_l3 (+ PUBLISH)
    t.store_ack(n)                            # the plain-sidecar (QSA index) write of the same node acks
    return t, n


# --------------------------------------------------------------------------------------------------- the RED form
def test_rc28_form_switch_off_marks_l3_on_an_incomplete_page(arena, host_cut, monkeypatch):
    """Switch 0 = rc28: the store ack rebinds ("already arena rows") and sends REBIND + L3 on an incomplete page --
    every worker sets l3_present and never writes its share; TP0's S74/S81 entry then leaves silently."""
    monkeypatch.setenv(S74B, "0")
    t, n = _metal(arena)
    assert n.l3_present is True
    v = R.attach([])
    assert len(_events(v, R.PUBLISH)) == 1 and len(_events(v, R.REBIND)) == 1 and len(_events(v, R.L3)) == 1
    assert arena.find_states(["p1361.kv", "p1362.kv"]) == [1, 1], "the page stays CLAIMED (open0)"
    R.attach([])
    assert not R._S.l3_wait, "the watch left: 'another path marked it' -- nobody asks the owners again"


# ------------------------------------------------------------------------------------------------------ the fix
def test_store_ack_of_an_incomplete_direct_page_holds(arena, host_cut, caplog):
    with caplog.at_level("INFO"):
        t, n = _metal(arena)
    assert n.l3_present is False, "no l3_present on an incomplete page"
    v = R.attach([])
    assert _events(v, R.L3) == [] and _events(v, R.REBIND) == [], "no worker is told the page is in the store"
    assert len(_events(v, R.PUBLISH)) == 1, "the direct ack's one PUBLISH, not a second one"
    assert _key(n) in R._S.l3_wait and R._S.l3_wait[_key(n)][0] is n
    assert t.locks == [("lp", 7)], "the store op's host lock is given back as before"
    assert "S74b STORE-ACK-HOLD" in caplog.text


def test_held_page_gets_l3_once_every_owner_wrote(arena, host_cut):
    t, n = _metal(arena)
    R.attach([])
    _owners_write(arena, n.hash_value)
    assert arena.find_states(["p1361.kv", "p1362.kv"]) == [2, 2]
    ev = _events(R.attach([]), R.L3)
    assert ev == [(R.L3, "p1362", 2 * PAGE, 1, 1, 0)] and n.l3_present is True and not R._S.l3_wait


def test_metal_owners_are_asked_again_while_they_hold_the_page(arena, host_cut):
    """The rc28 recovery that never came: the worker still holds node 457 on its device -> S81 PUBLISHes again
    (backoff in passes) and the page completes once the W3 spill gave the claim room."""
    t, n = _metal(arena)
    _worker_set(arena, [_key(n)])
    sent = [len(_events(R.attach([]), R.PUBLISH)) for _ in range(6)]
    assert sent[0] == 1 and sum(sent[1:]) >= 1, f"the owners are asked again: {sent}"
    _owners_write(arena, n.hash_value)
    ev = _events(R.attach([]), R.L3)
    assert len(ev) == 1 and n.l3_present is True
    assert arena.find_states(["p1361.kv", "p1362.kv"]) == [2, 2], "the hole closed"


def test_store_ack_of_a_complete_direct_page_rebinds_as_before(arena, host_cut):
    t, n = _Tp0Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    _owners_write(arena, n.hash_value)
    t.store_ack(n)
    assert n.l3_present is True
    v = R.attach([])
    assert len(_events(v, R.REBIND)) == 1 and len(_events(v, R.L3)) == 1


def test_hold_without_an_earlier_watch_adds_one_and_publishes(arena, host_cut):
    """A store ack that comes without the direct ack's CUT-L3-WAIT (any order): the hold watches it itself."""
    t, n = _Tp0Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    t.store_ack(n)
    assert n.l3_present is False and _key(n) in R._S.l3_wait
    assert len(_events(R.attach([]), R.PUBLISH)) == 1


def test_staging_rows_keep_the_rebind_slot_check(arena, host_cut):
    """Rows in the staging ring: the rebind itself checks every slot (state 2) -- S74b stays out, the rc28 path
    (await_complete) decides, no l3_present on the incomplete page either way."""
    t, n = _Tp0Tree(arena), _node(arena_rows=False)
    _tp0_share0(arena, n.hash_value)
    assert U._s74b_store_ack_hold(t, n) is False
    t.store_ack(n)
    assert n.l3_present is False and not R._S.l3_wait and R._S.rebind_wait


def test_no_cut_and_worker_roles_are_untouched(arena, host_cut, monkeypatch):
    t, n = _Tp0Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    monkeypatch.setattr(R, "cut_role", lambda: None)
    t.store_ack(n)
    assert n.l3_present is True, "no token cut: the complete-ack verdict as before"
    R._reset_state_for_tests()
    n2 = _node(("q1", "q2"), nid=9)
    _tp0_share0(arena, n2.hash_value)
    monkeypatch.setattr(R, "cut_role", lambda: "worker")
    monkeypatch.setattr(R, "role", lambda: "worker")
    kept = []
    monkeypatch.setattr(R, "worker_keeps", lambda tree, why, node: kept.append((why, node.id)))
    t.store_ack(n2, op_id=8)
    assert kept == [("store-ack-cut", 9)] and n2.l3_present is False and not R._S.l3_wait


def test_s74_switch_off_disables_the_hold_too(arena, host_cut, monkeypatch):
    """FLLIPER_PDFLIP_CUT_L3_COMPLETE=0 is the pre-S74 world on purpose; S74b does not resurrect it half-way."""
    monkeypatch.setenv("FLLIPER_PDFLIP_CUT_L3_COMPLETE", "0")
    t, n = _Tp0Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    assert U._s74b_store_ack_hold(t, n) is False


def test_unreadable_page_state_keeps_the_rc28_verdict(arena, host_cut, monkeypatch):
    t, n = _Tp0Tree(arena), _node()
    _tp0_share0(arena, n.hash_value)
    monkeypatch.setattr(arena, "find_states", lambda stems: (_ for _ in ()).throw(OSError("unreadable")))
    assert U._s74b_store_ack_hold(t, n) is False


def test_metal_piece_no_worker_holds_under_its_key_is_revoked_not_frozen(arena, host_cut):
    """TP0 node 447 ([83072, 87168), a PUBLISH-SPLIT head piece) has no worker node under its key (the workers split at
    85248, not 87168): with the hold, S81 sees it and gives TP0's host layer back + frees its orphan share-0 claims --
    the page is cleanly absent and the next write is fresh (rc28: l3_present froze the orphan claims for 3 min)."""
    t, n = _Tp0Tree(arena), _node(("p1360", "p1361"), nid=447)
    n._pdflip_chain_piece = True

    def _give_back(node, npages):
        node.component_data[0].host_value = None
        return True

    t._k12_republish_orphaned = _give_back
    _tp0_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    t.store_ack(n)
    assert n.l3_present is False
    _worker_set(arena, [("p9999", 87616)])               # the workers hold the pages under another key
    R.attach([])
    R.attach([])
    assert arena.find_states(["p1360.kv", "p1361.kv"]) == [0, 0], "no orphan CLAIMED/open0 slot is left"
    assert not R._S.l3_wait
