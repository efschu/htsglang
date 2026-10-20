"""Review S108 x S109 (20.10.): counter-tests of the adversarial review (nf-s108-s109-review-1020).

Each test states what the user orders require; a RED test is a finding of the review, not a flake:

  "D schreibt fertige Turns sofort nach L2", "WEHE DAS SCHREIBEN IN DEN CACHE BLOCKIERT DURCH IRGENDEINE
  SYNCHRON ODER WARTESCHEISSE", "nichts verlaesst L2 ohne L3-Kopie", "0 vermeidbare Token".

S108 tests (R1, R2, R3) run against the S108 tree (this branch, 601cdfbc94).
S109 tests (R4, R5) need ``flliper.srt.mem_cache.l3_secured`` (S109b, 81037f0c40): run them with
PYTHONPATH pointing at an S109 checkout first; without it they skip.
CPU only, no GPU.
"""
from __future__ import annotations

import inspect
import os
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402

try:  # S108 fixtures (fake card, controller, sync detector) -- the builder's own scaffolding
    import test_s108_d_turn_end_l2_1020 as s108t  # noqa: E402
    from test_s108_d_turn_end_l2_1020 import card  # noqa: E402,F401 -- the fixture
    from flliper.srt.pdflip import d_turn_end_l2 as te  # noqa: E402
except Exception:  # noqa: BLE001 -- an S109-only checkout
    s108t = te = None

try:
    from flliper.srt.mem_cache import l3_secured as l3s  # noqa: E402
except Exception:  # noqa: BLE001 -- an S108-only checkout
    l3s = None

need_s108 = pytest.mark.skipif(s108t is None, reason="S108 tree not on the path")
need_s109 = pytest.mark.skipif(l3s is None, reason="S109b l3_secured not on the path")
L3_ASYNC_REAL = True   # the S109 conftest seam must not swap the scheduler forms R8 inspects


# ------------------------------------------------------------------------------------------------- S108
@need_s108
def test_r1_write_back_drain_does_not_wait_for_a_turn_end_copy(card):
    """R1 (S108, severity high): D runs write_policy=write_back (rc27 D.log: hicache_write_policy='write_back');
    ``evict()`` ends with ``writing_check(write_back=True)``, which synchronizes EVERY ack in the queue -- the
    S108 DoneEvents included. A turn-end copy still held by the copier then blocks the scheduler thread.
    Required: the drain returns while the copier holds the op (it may wait only for what it issued itself)."""
    assert "self.writing_check(write_back=True)" in inspect.getsource(urc.UnifiedRadixCache.evict)
    gate = threading.Event()
    c = s108t._controller(s108t.CUT[1], gate)
    with te.deferred_issue(c, "rid-r1"):
        assert c.write(s108t.dev(s108t._slots(2, seed=11)), node_id=41) is not None
    tree = types.SimpleNamespace(cache_controller=c, ongoing_write_through={41: None})
    tree._finish_write_through_ack = lambda ack_id: tree.ongoing_write_through.pop(ack_id, None)
    done = threading.Event()

    def evict_tail():
        urc.UnifiedRadixCache.writing_check(tree, write_back=True)
        done.set()

    th = threading.Thread(target=evict_tail, name="scheduler", daemon=True)
    th.start()
    returned = done.wait(1.0)
    gate.set()
    th.join(10)
    s108t._wait_copier(c)
    assert returned, ("evict()'s write_back drain waited for the S108 copier: the scheduler thread blocked "
                      "on a turn-end copy it did not need")


@need_s108
def test_r2_turn_end_mamba_claim_reaches_no_disk_secure():
    """R2 (S108 x S109d, severity high): the turn end guards only the KV claim (``claim_needs_room``); the
    mamba-anchor claim of the same node runs unguarded through ``_pdflip_mamba_claim`` -> H19 SHARE/FULL
    displacement -> ``_pdflip_release_anchor`` -> ``ArenaMambaPoolHost.drop_unreferenced`` ->
    ``_secure_before_drop`` -> ``HiCacheFile.arena_secure_to_disk`` (stat, and a 56 MiB write + fsync for an
    anchor without a copy) -- in the scheduler thread, on TP0, at every turn end that displaces.
    Required: no path from the turn end to a disk secure."""
    from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost

    claim = inspect.getsource(urc.UnifiedRadixCache._pdflip_direct_claim)
    i_guard = claim.index("_s108_turn_end_active")
    i_mamba = claim.index("self._pdflip_mamba_claim(")
    guarded_after = "_s108_turn_end_active" in claim[i_mamba - 400:i_mamba]
    h19 = inspect.getsource(urc.UnifiedRadixCache._pdflip_mamba_claim_h19)
    rel = inspect.getsource(urc.UnifiedRadixCache._pdflip_release_anchor)
    drop = inspect.getsource(ArenaMambaPoolHost.drop_unreferenced)
    sec = inspect.getsource(ArenaMambaPoolHost._secure_before_drop)
    chain = ("_pdflip_release_anchor(" in h19 and "drop_unreferenced(" in rel
             and "_secure_before_drop(" in drop and "arena_secure_to_disk" in sec)
    assert i_guard < i_mamba
    assert not (chain and not guarded_after), (
        "turn end -> mamba claim -> H19 displacement -> drop_unreferenced -> _secure_before_drop -> "
        "arena_secure_to_disk: file I/O in the scheduler thread, unguarded by _s108_turn_end_active")


@need_s108
def test_r3_a_923_hit_inline_is_a_refusal_on_the_copier_a_rank_death(card):
    """R3 (S108, characterisation, severity medium): the inline write REFUSES a #923 row (returns None, nothing
    committed, the scheduler goes on); the turn-end form commits first and raises TurnEndCopyError at the next
    ack poll -- uncaught in ``_count_ready_acks`` -> the scheduler of THIS rank dies alone (the cut makes the
    condition rank-local). Green = the escalation exists as described."""
    if not hasattr(te.DoneEvent, "refuse"):
        pass
    else:
        pytest.skip("4218abac79: the copier refuses by name (review A4 done) -- R3 described the old form")
    c = s108t._controller(s108t.CUT[1], compact_rows=8)
    assert c.write(s108t.dev(s108t._slots(1, seed=12)), node_id=51) is None          # inline: refused
    with te.deferred_issue(c, "rid-r3"):
        assert c.write(s108t.dev(s108t._slots(1, seed=12)), node_id=52) is not None   # committed
    s108t._wait_copier(c)
    tree = types.SimpleNamespace(_drain_depth_every=0, pp_rank=0)
    with pytest.raises(te.TurnEndCopyError):
        urc.UnifiedRadixCache._count_ready_acks(tree, c.ack_write_queue, "writing_check")


# ------------------------------------------------------------------------------------------------- S109
class _Arena:
    """The slot header the record is judged against: generation and state per slot."""

    def __init__(self, path: str, n: int = 8):
        self.path = path
        self.slots = n
        self.gens = np.full(n, 5, dtype=np.int64)
        self.states = np.full(n, 2, dtype=np.int64)   # COMPLETE

    def slot_gens_np(self, s):
        return self.gens[np.asarray(s, dtype=np.int64)]

    def slot_states(self, lst):
        return [int(self.states[int(i)]) for i in lst]

    def slot_stem(self, i):
        return "stem-%d" % int(i)


class _Store:
    """The #1459 L3 stem index after the store's LRU evictor unlinked the files (cap 139.70 GiB, 129.29 GiB
    reused at the rc27 launch): nothing listed."""

    def __init__(self):
        self.files = set()

    def l3_listed(self, stems):
        return [s in self.files for s in stems]


@need_s109
def test_r4_a_page_whose_l3_file_the_lru_evicted_is_not_secured(tmp_path):
    """R4 (S109b, severity high): the write-behind publishes (slot, generation) once; the store's LRU evictor
    (``LRUFileEvictor``, on_evict = metadata_cache.remove only) never clears it. ``secured`` reads the record
    FIRST and asks the index only for the rest -- so a page whose L3 file is gone still reads as secured, and
    every S109 spill (K23 host-leaf, S109 aux leaf, S109c, clock, claim) may drop its last L2 copy.
    Required: no copy on L3 -> not secured."""
    arena = _Arena(str(tmp_path / "arena-786432.bin"))
    l3s.publish(arena, [0, 1], [5, 5])              # the write-behind copied both pages
    store = _Store()                                 # ... and the LRU evictor unlinked both files since
    # 81037f0c40 (S109c): secured(arena, slots) -- the record alone; S109d adds the index as a FALLBACK only
    if "backend" in inspect.signature(l3s.secured).parameters:
        got = l3s.secured(arena, [0, 1], store)
    else:
        got = l3s.secured(arena, [0, 1])
    assert not got.any(), "a page with no file on L3 reads as secured: %s" % got.tolist()


@need_s109
def test_r5_the_l3_lru_eviction_invalidates_the_secured_record():
    """R5 (S109b, severity high, source): the evictor's eviction hook must reach the shared record."""
    from flliper.srt.mem_cache import hicache_storage as hs

    src = inspect.getsource(hs.HiCacheFile.__init__)
    i = src.index("on_evict=(")
    hook = src[i:i + 200]
    assert ("l3_secured" in hook or "l3sec" in hook or "_l3s" in hook), (
        "LRUFileEvictor on_evict is %r -- the S109b record is never invalidated" % hook.split(")")[0])


# ------------------------------------------------------------------------------- Nachprüfung 4218abac79
def _ack_stub(c, ids):
    t = types.SimpleNamespace(cache_controller=c, ongoing_write_through={i: None for i in ids},
                              _drain_depth_every=0, pp_rank=0, finished=[])
    t._finish_write_through_ack = lambda a: (t.ongoing_write_through.pop(a, None), t.finished.append(a))
    t._s108_ack = lambda a, ev: urc.UnifiedRadixCache._s108_ack(t, a, ev)
    t.writing_check = lambda write_back=False, only=None: urc.UnifiedRadixCache.writing_check(t, write_back, only)
    t._count_ready_acks = lambda q, label: urc.UnifiedRadixCache._count_ready_acks(t, q, label)
    return t


class _Ready:
    def query(self):
        return True

    def synchronize(self):
        pass


class _Open:
    def __init__(self):
        self.synced = 0

    def query(self):
        return False

    def synchronize(self):
        self.synced += 1


@need_s108
def test_r6_head_of_line_the_poll_does_not_reach_a_ready_inline_ack_behind_an_open_copy(card):
    """R6 (4218abac79, characterisation, delay -- not a wait): the non-blocking ack poll
    (``_count_ready_acks``) stops at the first open entry. A READY inline ack queued behind an open S108
    copy is not processed until the copy lands: its write-through lock, staging-ring share and COMPLETE
    are held for the copy's duration. Green = the delay exists as described (metal: #1472 WT-ACK)."""
    if not hasattr(urc.UnifiedRadixCache, "_s108_ack"):
        pytest.skip("pre-4218abac79 tree")
    gate = threading.Event()
    c = s108t._controller(s108t.CUT[1], gate)
    with te.deferred_issue(c, "rid-r6"):
        c.write(s108t.dev(s108t._slots(1, seed=21)), node_id=71)
    c.ack_write_queue.append(s108t.hcc.HiCacheAck(None, _Ready(), [72]))
    t = _ack_stub(c, [71, 72])
    urc.UnifiedRadixCache.writing_check(t)
    delayed = t.finished == []
    gate.set()
    s108t._wait_copier(c)
    urc.UnifiedRadixCache.writing_check(t)
    assert delayed and sorted(t.finished) == [71, 72]


@need_s108
def test_r7_drain_own_returns_unwaited_when_the_own_ack_is_a_turn_end_copy(card):
    """R7 (4218abac79, characterisation, defensive): ``_s108_drain_own(node.id)`` never waits for a
    DoneEvent, even when it is the node's OWN ack -- the caller (K21/K15) then reads ``node.backuped``
    (True since the issue) as landed. Not reachable today (a node with an S108 op in flight is write-
    through locked: no device leaf, no tombstone); a guard would keep it so."""
    if not hasattr(urc.UnifiedRadixCache, "_s108_drain_own"):
        pytest.skip("pre-4218abac79 tree")
    gate = threading.Event()
    c = s108t._controller(s108t.CUT[1], gate)
    with te.deferred_issue(c, "rid-r7"):
        c.write(s108t.dev(s108t._slots(1, seed=22)), node_id=81)
    t = _ack_stub(c, [81])
    done = threading.Event()
    th = threading.Thread(target=lambda: (urc.UnifiedRadixCache._s108_drain_own(t, 81), done.set()),
                          name="scheduler", daemon=True)
    th.start()
    returned_unwaited = done.wait(1.0) and t.finished == []
    gate.set()
    th.join(10)
    s108t._wait_copier(c)
    assert returned_unwaited


# ------------------------------------------------------------------------------- Re-Review S109 @ 1f77ec3b10
@need_s109
def test_r8_a_page_leaving_l2_on_its_l3_copy_is_promoted_in_the_l3_lru():
    """R8 (S109b/d, severity medium-high): ZERO-ANCHOR 1015 (``HiCacheFile._l3_refresh_kept``, measured: a parked
    tail lost from BOTH tiers -- the arena freed kept slots against old L3 files and the L3 LRU took exactly those)
    made "a copy that becomes the only copy ... the youngest the store has": ``arena_secure_to_disk`` touches every
    stem it relies on (in-memory form, no stat). The S109 scheduler forms decide from the record AND index and never
    call ``arena_secure_to_disk`` -- so the copy a spill relies on keeps its WRITE-time LRU age and is the next victim.
    Required: the scheduler forms that release L2 on an L3 copy promote it (or queue the promotion)."""
    import flliper.srt.mem_cache.pool_host.arena_pool as ap
    import flliper.srt.mem_cache.pool_host.arena_mamba_pool as amp

    srcs = {
        "secure_rows_to_l3": inspect.getsource(ap.ArenaMHAHostPool.secure_rows_to_l3),
        "_secure_before_drop": inspect.getsource(amp.ArenaMambaPoolHost._secure_before_drop),
        "l3_secured": inspect.getsource(l3s),
    }
    promoted = {k: ("_l3_refresh_kept" in v or ".touch(" in v) for k, v in srcs.items()}
    assert any(promoted.values()), "no L3 LRU promotion on the async release paths: %s" % promoted


@need_s109
def test_r10_unsecured_census_cost_per_claim_on_a_full_arena(tmp_path):
    """R10 (S109d, measurement): ``unsecured_keys_lo`` runs on the scheduler thread at every claim that has to make
    room (``_evict_for_claim``): one C census plus one ``slot_stem`` (ctypes) per record-secured COMPLETE slot and an
    index lookup. Real C arena, D's KV arena size (6485 slots), every page copied. Measured 4.8-6.8 ms with a
    Python-set index (the shm #1459 index is not faster); the bound here is 20 ms -- a characterisation, the number
    goes into the report as a metal check (claim ms), not a verdict."""
    import time

    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    SB, N = 4096, 6485
    a = ShmArena(str(tmp_path / "arena-kv.bin"), SB, N)
    stems = ["k%060x" % i for i in range(N)]
    for i in range(0, N, 512):
        part = stems[i:i + 512]
        got = a.claim_slots(part, [SB] * len(part))
        a.complete_slots([g[0] for g in got], [g[2] for g in got], [(0, SB)])
    sl, gens, _klo, _khi = a.complete_census()
    assert len(sl) == N
    l3s.publish(a, sl, gens)

    class _Idx:
        def __init__(self, f):
            self.f = set(f)

    class _Be:
        _l3idx = _Idx(stems)

        def l3_listed(self, st):
            return [x in self._l3idx.f for x in st]

    be = _Be()
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        un = l3s.unsecured_keys_lo(a, be)
        best = min(best, (time.perf_counter() - t) * 1e3)
    print("R10 unsecured_keys_lo ms=%.2f slots=%d unsecured=%d" % (best, N, un.shape[0]))
    assert un.shape[0] == 0
    assert best < 20.0, "unsecured_keys_lo costs %.1f ms CPU per claim on the scheduler thread" % best


@need_s109
def test_r11_an_l3_lru_unlink_makes_the_write_behind_copy_the_page_again(tmp_path):
    """R11 (S109-A1 follow-on, severity high): the L3 LRU unlink voids the SHARED record (``invalidate_stems``) but not
    the write-behind's own visit baseline ``arena._l3wb_sec`` -- and the pass picks ``new`` (and urgent) slots ONLY
    from that baseline. A page still COMPLETE in L2 whose L3 file the cap took is then never copied again: it reads
    unsecured forever, no spill/clock/claim may free it (keep list), and the arena loses that slot for the rest of the
    boot (S108 fills L2 with exactly such long-lived pages). Required: after the unlink the pass sees the slot as new."""
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    SB, N = 4096, 8
    a = ShmArena(str(tmp_path / "arena-kv.bin"), SB, N)
    stem = "k%060x" % 1
    (s, st, g), = a.claim_slots([stem], [SB])
    assert a.complete_slots([s], [g], [(0, SB)]) == [1]
    sl, gens, klo, _khi = a.complete_census()
    # the write-behind secured the page: its baseline and the shared record
    a._l3wb_sec = (np.full(N, -1, dtype=np.int64), np.zeros(N, dtype=np.uint64))
    a._l3wb_sec[0][sl] = gens
    a._l3wb_sec[1][sl] = klo
    l3s.arrays(a, create=True)
    l3s.publish(a, sl, gens)
    # the L3 LRU took the file: the evict hook
    assert l3s.invalidate_stems(a, [stem]) == 1
    assert not l3s.secured(a, [s]).any()                       # the spills keep it (good) ...
    base_gen, base_lo = a._l3wb_sec
    new = np.nonzero((base_gen[sl] != gens) | (base_lo[sl] != klo))[0]
    assert new.shape[0] == 1, ("... but the write-behind's baseline still says copied: the pass never visits the slot "
                               "again (urgent requests are filtered by the same baseline)")
