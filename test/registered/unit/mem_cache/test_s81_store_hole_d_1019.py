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


# ----------------------------------------------------------------- review nf-review-s81-1019: A1 / A2 / A3
def _census(arena, stem):
    (s, st), = arena.find_slots([stem])
    row = dict(arena.claim_census([s])[0]) if s >= 0 else {}
    row.pop("age_ms", None)
    return s, st, row


def test_a1_join_between_census_and_revoke_keeps_the_slot_and_the_joiners_writer(arena, host_cut, monkeypatch):
    """The review repro: the census passes, a worker joins (claims=2 open=1, no bytes yet), THEN the free runs.
    Before A1 (arena_release_claims) the joiner's writer was taken off and the slot freed -> its completion 3."""
    t, n = _metal(arena)
    _worker_set(arena, [])
    orig = R._s81_orphan_slots
    seen = {}

    def racing(a, stems):
        out = orig(a, stems)
        assert out, "the census passed"
        (s, st, g), = a.claim_slots(["p1337.kv"], [SLOT])   # the worker joins inside the window
        assert st == 1
        seen.update(s=s, g=g, row=_census(a, "p1337.kv")[2])
        return out

    monkeypatch.setattr(R, "_s81_orphan_slots", racing)
    R.attach([])
    s, st, row = _census(arena, "p1337.kv")
    assert (s, st) == (seen["s"], 1), "the joined slot stays CLAIMED"
    assert row == seen["row"] and row["claims"] == 2 and row["open"] == 1, "the joiner's writer is untouched"
    assert R._S.l3_wait, "release said kept: the entry stays watched"
    assert arena.complete_slots([seen["s"]], [seen["g"]], [(0, HALF)]) == [0], "merged -- not 3 'recycled'"


def test_a1_c_revoke_status_per_condition(arena):
    """arena_revoke_orphan_claims: 0 only for the orphan; joined / open / bytes / referenced -> 1 untouched;
    moved generation -> 2."""
    def share0(stem):
        (s, st, g), = arena.claim_slots([stem], [SLOT])
        arena.complete_slots([s], [g], [])
        return s, g
    s0, g0 = share0("o.kv")
    s1, g1 = share0("joined.kv")
    arena.claim_slots(["joined.kv"], [SLOT])                    # a worker joined, open, no bytes
    (s2, _, g2), = arena.claim_slots(["open.kv"], [SLOT])        # TP0's own writer still open
    (s3, _, g3), = arena.claim_slots(["bytes.kv"], [SLOT])
    arena.complete_slots([s3], [g3], [(0, HALF)])                # bytes merged
    s4, g4 = share0("ref.kv")
    arena.ref_slots([s4], +1)                                    # referenced
    before = {k: _census(arena, k)[2] for k in ("joined.kv", "open.kv", "bytes.kv", "ref.kv")}
    st = arena.revoke_orphan_claims([s0, s1, s2, s3, s4, s0], [g0, g1, g2, g3, g4, g0])
    assert st == [0, 1, 1, 1, 1, 2]
    assert arena.find_states(["o.kv"]) == [0]
    for k, row in before.items():
        assert _census(arena, k)[1] == 1 and _census(arena, k)[2] == row, k


@pytest.mark.parametrize("case", ["pid", "open", "covered", "claims", "ref"])
def test_a2_precheck_refuses_each_condition_and_nothing_is_touched(arena, host_cut, monkeypatch, case):
    """Each release condition alone, in the dangerous form 'joined / open / no bytes yet': the Python census says
    None, the full pass leaves the slot CLAIMED and every writer count as it was."""
    t, n = _metal(arena)
    s = arena.find_slots(["p1337.kv"])[0][0]
    if case == "pid":
        monkeypatch.setattr(R.os, "getpid", lambda: 1)            # the claim is another process's
    elif case == "open":
        arena.claim_slots(["p1337.kv"], [SLOT])                  # a joiner, open, no bytes
    elif case == "covered":
        (s2, st2, g2), = arena.claim_slots(["p1337.kv"], [SLOT])
        arena.complete_slots([s2], [g2], [(0, HALF)])            # bytes, writer closed (claims 2)
    elif case == "claims":
        (s2, st2, g2), = arena.claim_slots(["p1337.kv"], [SLOT])
        arena.complete_slots([s2], [g2], [])                     # joined and closed, NO bytes: only claims=2 says it
    elif case == "ref":
        arena.ref_slots([s], +1)
    stems = ["p1337.kv", "p1338.kv"]
    before = _census(arena, "p1337.kv")
    assert R._s81_orphan_slots(arena, stems) is None
    _worker_set(arena, [])
    R.attach([])
    assert _census(arena, "p1337.kv") == before, "slot and writer counts untouched"
    assert R._S.l3_wait


def test_a2_publish_gap_is_capped_at_64_over_ten_republishes(arena, host_cut):
    t, n = _metal(arena)
    _worker_set(arena, [_key(n)])
    at = []
    for i in range(1, 700):
        if _events(R.attach([]), R.PUBLISH):
            at.append(i)
        if len(at) >= 12:
            break
    gaps = [b - a for a, b in zip(at, at[1:])]
    assert len(at) >= 12 and max(gaps) == R.S81_PUBLISH_MAX_GAP == 64 and gaps[-3:] == [64, 64, 64]


def test_a3_real_k12_release_drops_tp0s_reference_and_the_orphan_is_revoked(arena, host_cut):
    """The real K12 path: UnifiedRadixCache._k12_republish_orphaned -> _pdflip_release_chain_piece_host -> the REAL
    ArenaMHAHostPool.free of TP0's host rows gives the slot reference back (#1427: a writer's node references its
    CLAIMED slot from the claim on), then the atomic revoke frees the orphan. Only the tree's component funnel is a
    stand-in (LRU bookkeeping); the release, the R12 STATE and the pool free are the production code."""
    from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    pool = SimpleNamespace(arena=arena, row_slot=None, staging_rows=0, _arena_page_tokens=PAGE, arena_slots=16,
                           _pending_mask=None, _pending={})

    class _K12Tree(_Tree):
        _k12_republish_orphaned = U._k12_republish_orphaned
        _pdflip_release_chain_piece_host = U._pdflip_release_chain_piece_host

        def __init__(self, a):
            super().__init__(a)
            self._components_tuple = (SimpleNamespace(component_type=ComponentType.FULL),)
            self.evictable_host_leaves = set()
            self.pp_size = 1

        def _evict_component_and_detach_lru(self, node, comp, target=None, tracker=None):
            cd = node.component_data[int(comp.component_type)]
            freed = ArenaMHAHostPool.free(pool, cd.host_value)
            cd.host_value = None
            return 0, freed

    t = _K12Tree(arena)
    n = _node()
    for cd in n.component_data:
        cd.value = torch.arange(2 * PAGE)                       # TP0's device copy is still there
    slots = []
    for h in n.hash_value:
        (s, st, g), = arena.claim_slots([f"{h}.kv"], [SLOT])
        arena.ref_slots([s], +1)                                # the writer's node references its claim
        arena.complete_slots([s], [g], [])                      # share 0: open0
        slots.append(s)
    hv = torch.cat([torch.arange(s * PAGE, (s + 1) * PAGE) for s in slots])
    n.component_data[0] = SimpleNamespace(host_value=hv, value=torch.arange(2 * PAGE), host_lock_ref=0)
    n.component_data[1] = SimpleNamespace(host_value=None, value=None, host_lock_ref=0)
    t._k12_direct_l3_verdict(n, 0)
    R.attach([])
    assert arena.slot_refs(slots) == [1, 1]
    _worker_set(arena, [])
    at_revoke = []
    real_revoke = arena.revoke_orphan_claims
    arena.revoke_orphan_claims = lambda sl, gs: (at_revoke.append(arena.slot_refs(sl)), real_revoke(sl, gs))[1]
    v = R.attach([])
    assert at_revoke == [[0, 0]], "the real host release gave TP0's references back before the free"
    assert n.component_data[0].host_value is None, "TP0's host layer given back by the real release"
    assert arena.find_states(["p1337.kv", "p1338.kv"]) == [0, 0], "revoked, not kept"
    assert not R._S.l3_wait
    assert any(e[0] == R.STATE for e in _events(v, R.STATE)), "R12 STATE for the workers"


@pytest.mark.parametrize("case", ["open_alone", "covered_alone"])
def test_a2_single_condition_open_or_covered_alone(arena, case):
    """Exactly ONE condition violated, claims stays 1 (the cases above also carry claims=2): an open writer of
    the claimant itself, or bytes it merged itself. The census refuses, the atomic free keeps, nothing touched."""
    (s, st, g), = arena.claim_slots(["q.kv"], [SLOT])
    if case == "covered_alone":
        arena.complete_slots([s], [g], [(0, HALF)])
    before = _census(arena, "q.kv")
    assert before[2]["claims"] == 1
    assert (before[2]["open"], before[2]["covered"] > 0) == ((1, False) if case == "open_alone" else (0, True))
    assert R._s81_orphan_slots(arena, ["q.kv"]) is None
    assert arena.revoke_orphan_claims([s], [g]) == [1]
    assert _census(arena, "q.kv") == before
