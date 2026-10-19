"""S74 CUT-L3-WAIT (rc16 D, 19.10. 12:56:11-12:59:11, pdflip-114-244 / 114-246).

Metal: group D runs Form A with the token cut -- TP0 claims every KV page with
share 0 (no extents), the owners TP1/TP2 merge theirs. The K15 anchor save
(``pdflip_save_anchor_before_tombstone``) runs on TP0 ALONE and backs a node
whose KV is not backed with a full ``write_backup``: TP0 claims the pages
fresh, acks at once (nothing to copy) and ``_k12_direct_l3_verdict`` set
``l3_present`` and sent the R12 L3 event. Every worker set ``l3_present`` from
it, so its publish sweep never wrote its share (``l3_present`` is a skip
reason): the claims stayed CLAIMED with no writer (``ARENA-REF-CENSUS-CLAIM
claimed=22 -> 104 -> 153 oldest=5193/.../pid12775/host-write/g9/open0``, the
oldest exactly the first K15 save 12:56:11), the store's KV prefix ended at
the first such page for the rest of the phase (front ``kv_pages=1432`` /
``1204`` from 12:57:12 on), the reset dropped the claims, and the wake read
and P read stopped there: 22817 tokens recomputed for 114-244.
``K21 RESET-DROPS open_tp0=22`` = 22 TP0 acks no worker ack ever paired.

Now: under the cut TP0 marks ``l3_present`` / sends L3 only for a COMPLETE
page (the shared arena header, no collective); an incomplete page waits and
is re-checked at every broadcast (``attach``), a worker keeps the node
un-backed and writes its share. Switch ``FLLIPER_PDFLIP_CUT_L3_COMPLETE``
(default ON; ``0`` = the pre-S74 verdict -- the RED form below).
"""
from __future__ import annotations

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


class _Backend:
    @staticmethod
    def _get_suffixed_key(h):
        return f"{h}.kv"


class _Tree:
    """TP0's tree: the real verdict and the real completeness probe over a real arena."""
    _k12_direct_l3_verdict = U._k12_direct_l3_verdict
    _s74_cut_page_complete = U._s74_cut_page_complete

    def __init__(self, arena):
        self.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(arena=arena), storage_backend=_Backend())


def _node(hashes=("p1432", "p1433")):
    root = SimpleNamespace(parent=None, children={})
    cd = SimpleNamespace(host_value=torch.arange(len(hashes) * PAGE), value=None, host_lock_ref=0)
    n = SimpleNamespace(id=1250, parent=root, key=list(range(len(hashes) * PAGE)), children={},
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
    monkeypatch.delenv("FLLIPER_PDFLIP_CUT_L3_COMPLETE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_L3_PRESENT_STRICT", raising=False)
    yield
    R._reset_state_for_tests()


def _tp0_claims_share0(arena, hashes):
    """TP0 (share 0) claims every page fresh and merges no extent -- the K15 full backup."""
    out = []
    for h in hashes:
        (s, st, g), = arena.claim_slots([f"{h}.kv"], [SLOT])
        assert st == 0
        out.append((s, g))
    return out


def _owner_writes(arena, hashes, off):
    for h in hashes:
        (s, st, g), = arena.claim_slots([f"{h}.kv"], [SLOT])
        assert st in (0, 1, 2)
        arena.complete_slots([s], [g], [(off, HALF)])


def _l3_events(verdicts):
    return [e for v in verdicts if isinstance(v, R.FormAHostVerdict) for e in v.events if e[0] == R.L3]


def test_red_form_switch_off_marks_l3_on_tp0s_share0_ack(arena, host_cut, monkeypatch):
    """The pre-S74 verdict (switch 0) = the metal: l3_present + L3 event on an incomplete page."""
    monkeypatch.setenv("FLLIPER_PDFLIP_CUT_L3_COMPLETE", "0")
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    assert n.l3_present is True
    assert len(_l3_events(R.attach([]))) == 1, "every worker would skip its share"
    assert arena.find_states(["p1432.kv", "p1433.kv"]) == [1, 1], "the page stays CLAIMED (open0)"


def test_incomplete_page_gets_no_l3_until_every_owner_wrote(arena, host_cut):
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    assert n.l3_present is False
    assert _l3_events(R.attach([])) == [], "no worker is told the page is in L3"
    assert len(R._S.l3_wait) == 1
    # the owners' shares arrive (their sweep writes the un-backed node; the claim JOINS TP0's slot)
    _owner_writes(arena, n.hash_value, 0)
    assert _l3_events(R.attach([])) == [], "one owner is not the page"
    _owner_writes(arena, n.hash_value, HALF)
    assert arena.find_states(["p1432.kv", "p1433.kv"]) == [2, 2]
    ev = _l3_events(R.attach([]))
    assert len(ev) == 1 and ev[0][1:3] == ("p1433", 2 * PAGE)
    assert n.l3_present is True and not R._S.l3_wait


def test_complete_page_at_the_ack_marks_at_once(arena, host_cut):
    """The ordinary sweep: the owners' acks came first -- the verdict is immediate, as before."""
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    _owner_writes(arena, n.hash_value, 0)
    _owner_writes(arena, n.hash_value, HALF)
    t._k12_direct_l3_verdict(n, 0)
    assert n.l3_present is True and len(_l3_events(R.attach([]))) == 1 and not R._S.l3_wait


def test_no_cut_keeps_the_ack_verdict(arena, host_cut, monkeypatch):
    """Off the token cut TP0's own ack carries the bytes: the verdict on its ack stands."""
    monkeypatch.setattr(R, "cut_role", lambda: None)
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    assert n.l3_present is True and len(_l3_events(R.attach([]))) == 1


def test_waiting_node_that_went_another_way_leaves(arena, host_cut):
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    n.component_data[0].host_value = None    # host rows gone (republish / transit)
    assert _l3_events(R.attach([])) == [] and not R._S.l3_wait


def test_reset_names_and_drops_the_open_waits(arena, host_cut, caplog):
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    with caplog.at_level("WARNING"):
        R.on_tree_reset(None)
    assert "S74 CUT-L3-RESET open=1 oldest_wait_ms=" in caplog.text and not R._S.l3_wait


def test_unreadable_arena_keeps_the_old_verdict(host_cut):
    """No arena / a probe error: True -> the pre-S74 verdict, never a new stop."""
    t = _Tree(None)
    n = _node()
    t._k12_direct_l3_verdict(n, 0)
    assert n.l3_present is True


class _WorkerTree:
    """A Form A worker: the real targeted publish over a stub write_backup (an owner's share into the real arena)."""
    _s74_worker_publish = U._s74_worker_publish

    def __init__(self, arena, wn, off, backed=False):
        self.root_node, self._components_tuple = wn.parent, ()
        self.arena, self.off, self.calls = arena, off, []
        wn.evicted, wn.backuped, wn.write_through_pending_id = False, backed, None
        wn.component_data[0].value = torch.arange(len(wn.hash_value) * PAGE)

    def write_backup(self, node):
        self.calls.append(node.id)
        _owner_writes(self.arena, node.hash_value, self.off)
        return len(node.hash_value)


def _worker_consume(monkeypatch, sent, wtree):
    monkeypatch.setattr(R, "role", lambda: "worker")
    monkeypatch.setattr(R, "cut_role", lambda: "worker")
    R.consume(list(sent), wtree)
    monkeypatch.setattr(R, "role", lambda: "host")
    monkeypatch.setattr(R, "cut_role", lambda: "host")


def test_worker_gets_publish_not_l3_and_writes_its_share_at_the_broadcast(arena, host_cut, monkeypatch):
    """Review A1: the CUT-L3-WAIT sends PUBLISH (never L3); each worker writes its share at the SAME broadcast,
    its claim joins TP0's open share-0 slot (status 1), the page completes, the next attach sends L3."""
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    sent = R.attach([])
    kinds = [e[0] for v in sent for e in v.events]
    assert kinds == [R.PUBLISH] and _l3_events(sent) == []
    w1, w2 = _node(), _node()
    t1, t2 = _WorkerTree(arena, w1, 0), _WorkerTree(arena, w2, HALF)
    _worker_consume(monkeypatch, sent, t1)
    R._S.pending = {}
    _worker_consume(monkeypatch, sent, t2)
    assert t1.calls == [1250] and t2.calls == [1250] and w1.l3_present is False
    assert arena.find_states(["p1432.kv", "p1433.kv"]) == [2, 2]
    R._S.pending = {}
    ev = _l3_events(R.attach([]))
    assert len(ev) == 1 and n.l3_present is True


def test_publish_and_transit_of_the_same_key_in_one_broadcast_both_apply(arena, host_cut, monkeypatch, caplog):
    """S74r review A1 (mutant M2): PUBLISH has its OWN pending slot -- a TRANSIT (or STATE/REBIND) of the same
    (hash, depth) in the same broadcast must not displace it, nor it them. Both apply on the worker, and the
    SENT line / the worker's per-consume sum name the PUBLISH."""
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    k = R.node_key(n)
    R._S.ledger.append((R.TRANSIT, k[0], k[1], 1, 0, 0))   # TP0's deferred transit of the same node, same pass
    with caplog.at_level("INFO"):
        sent = R.attach([])
    assert sorted(e[0] for v in sent for e in v.events) == sorted([R.PUBLISH, R.TRANSIT])
    assert "publish=1" in caplog.text
    w = _node()
    wt = _WorkerTree(arena, w, 0)
    released = []
    wt._pdflip_release_chain_piece_host = lambda node: released.append(node.id)
    wt.ongoing_write_through = {}
    caplog.clear()
    with caplog.at_level("INFO"):
        _worker_consume(monkeypatch, sent, wt)
    assert released == [1250], "the TRANSIT was applied"
    assert wt.calls == [1250], "the PUBLISH was applied"
    assert "S74 CUT-L3-PUBLISH-SUM" in caplog.text and "n=1" in caplog.text
    # the reverse order in the ledger: still both
    R._S.pending = {}
    R._S.ledger = [(R.TRANSIT, k[0], k[1], 1, 0, 0), (R.PUBLISH, k[0], k[1], 1, 1, 0)]
    sent2 = R.attach([])
    w2 = _node()
    wt2 = _WorkerTree(arena, w2, HALF)
    rel2 = []
    wt2._pdflip_release_chain_piece_host = lambda node: rel2.append(node.id)
    wt2.ongoing_write_through = {}
    _worker_consume(monkeypatch, sent2, wt2)
    assert rel2 == [1250] and wt2.calls == [1250]


def test_worker_publish_skips_a_backed_node(arena, host_cut, monkeypatch):
    """The ordinary sweep: the worker's own write is already issued/backed -- PUBLISH writes nothing twice."""
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    sent = R.attach([])
    w = _node()
    wt = _WorkerTree(arena, w, 0, backed=True)
    _worker_consume(monkeypatch, sent, wt)
    assert wt.calls == []


def test_publish_switch_off_sends_no_event(arena, host_cut, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_CUT_L3_PUBLISH", "0")
    t, n = _Tree(arena), _node()
    _tp0_claims_share0(arena, n.hash_value)
    t._k12_direct_l3_verdict(n, 0)
    assert R.attach([]) == []


def test_partial_reap_frees_tp0s_share0_claim_after_the_window(arena):
    """Review A1, measured on the real C arena: TP0's share-0 claim after its merge has no open writer and no
    reference -- inside the window it stays (an owner JOINS without room), past it #231 reaps it and the owner's
    claim is FRESH (needs a free slot). This is why the PUBLISH event exists."""
    (s, st, g), = arena.claim_slots(["p2000.kv"], [SLOT])
    assert st == 0
    assert arena.complete_slots([s], [g], []) == [0]          # TP0 merges share 0
    assert arena.reap_partial(30.0) == [], "inside FLLIPER_PDFLIP_ARENA_PARTIAL_REAP_S: kept"
    (s1, st1, _), = arena.claim_slots(["p2000.kv"], [SLOT])
    assert (s1, st1) == (s, 1), "an owner inside the window joins TP0's slot"
    arena.complete_slots([s1], [g], [(0, HALF)])
    (s3, st3, g3), = arena.claim_slots(["p2001.kv"], [SLOT])
    arena.complete_slots([s3], [g3], [])
    # past the window (age 0 here): the share-0 claim is reaped -- and so is the page ONE owner merged (the other
    # never came): both are the rc16 212-1185 'missing' pages (#231 ARENA-REAP-PARTIAL total=2132 on TP0)
    assert sorted(arena.reap_partial(0.0)) == sorted([s, s3])
    assert arena.find_states(["p2001.kv"]) == [0]
    (s4, st4, _), = arena.claim_slots(["p2001.kv"], [SLOT])
    assert st4 == 0, "the late owner's claim is fresh: it needs room"


def test_wait_lines_carry_ms_and_retry_is_page_capped(arena, host_cut, monkeypatch, caplog):
    """Review A1/A2: WAIT/LATE/RESET name the wait in ms; the re-check probes at most the page cap per broadcast
    (round robin) and names its cost."""
    monkeypatch.setenv("FLLIPER_PDFLIP_CUT_L3_RETRY_PAGES", "2")
    t = _Tree(arena)
    nodes = [_node((f"a{i}", f"b{i}")) for i in range(3)]
    for i, n in enumerate(nodes):
        n.id = 100 + i
        _tp0_claims_share0(arena, n.hash_value)
    with caplog.at_level("INFO"):
        for n in nodes:
            t._k12_direct_l3_verdict(n, 0)
        R.attach([])
    assert "oldest_wait_ms=" in caplog.text and "CUT-L3-RETRY waiting=3 probed=1 pages=2 cap=2" in caplog.text
    order = list(R._S.l3_wait)
    assert order[-1] == ("b0", 2 * PAGE), "the probed entry went behind the unprobed"
    for n in nodes:
        _owner_writes(arena, n.hash_value, 0)
        _owner_writes(arena, n.hash_value, HALF)
    caplog.clear()
    with caplog.at_level("INFO"):
        for _ in range(3):
            R.attach([])
    assert not R._S.l3_wait and all(n.l3_present for n in nodes)
    assert "CUT-L3-LATE" in caplog.text and "waited_ms=" in caplog.text


def test_switch_default_on():
    assert R.publish_on({}) is True and R._retry_pages({}) == 8192
    assert R.cut_l3_complete_on({}) is True
    assert R.cut_l3_complete_on({"FLLIPER_PDFLIP_CUT_L3_COMPLETE": "0"}) is False
