"""PPHASE-GAPS-1024: the claim census of the uncopied pages asks the #1459 index only for slots it has not confirmed.

THE METAL (27B rc39a b9cddabda1, P = PP3, boot dkr27bbar1drei10240803_b9cddabda1_1024_080355; duo 1024_070103).
User order 24.10. ~09:00Z: "die prefill rate auf P ist nicht immer so hoch wie sie sein koennte". The P phases ran at
61 % (int8) / 39 % (duo) card occupancy on PP0. Of the card idle between forwards, the largest single term on both
forms is the store publish on PP0's scheduler thread (``#PGAP deferred_publish`` + the retain publish inside
``process``): 29.8 s of 276 s (int8) and 41.3 s of 247 s (duo). Split by what the publish's claim did right before
(``#1427 ARENA-DROP`` lines between two ``PDFLIP-PUBLISH-*`` lines of PP0):

    int8  chunk publish, KV drop: n=30 mean 430-646 ms   retain, KV drop: n=24 mean 867-1148 ms
          mamba-only drop (112 slots): 20-66 ms            no drop: 15-23 ms
    duo   chunk publish, KV drop: n=12 mean 1326-1687 ms  retain, KV drop: n=22 mean 1570-1863 ms

The arena-size term is ``l3_secured.unsecured_keys_lo`` (S109d), which ``ArenaMHAHostPool._evict_for_claim`` runs
for EVERY claim that must make room: it asks the index again for every COMPLETE slot the record calls copied --
~0.5 M lookups on the 27B KV arena (720896 slots of one token each).

THE FIX (l3_secured ``LISTED_MEMO``): per arena and PROCESS, the generation at which a slot's stem was last FOUND
listed; the SHARED record is still read for every slot on every census, only the index question is skipped for a
record-positive slot confirmed at its current generation. The danger direction is a page reported COPIED that is not
(the keep list would let the clock take it). Safety rests on two cross-process things, both held here:
(a) the record gate -- the memo answers only for record-positive slots, so a record voided by ANY process (another
rank's L3 LRU) reads uncopied through a warm memo (R1, second ShmArena object on the same path = another process);
(b) the claim's free decision (``verify_candidates`` -> ``secured``) asks the index WITHOUT the memo.
census(memo) == census(no memo) is held in-process through the orderly transitions; where record and index disagree
across processes (S109-A1 window; a sister's void + same-generation re-publish under index lag between two censuses
of this process) the residue is named and the free decision refuses (R2b). ``HiCacheFile.clear`` now voids the record (R3;
review SOLL 1); the ``_stat_stems`` strike does not (residue, R3b: the free decision refuses). Review tests R1-R4 (review-pphase-gaps-1024.md,
desk/27b-pphase-gaps-review-1024 @ 74f392b61f) are taken over below.
"""
from __future__ import annotations

import os
import shutil
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flliper.srt.mem_cache import l3_secured as l3s  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

#: keep the scheduler forms (no instant write-behind seam from conftest)
L3_ASYNC_REAL = True
_REAL_UNSEC = l3s.unsecured_keys_lo   # bound at import, before any seam could replace it


def _stem(i):
    return "k%058x_%d" % (i * 7919, i)


def _arena(tmp_path, n, missing=(), sb=256, cap=1 << 16, name="a"):
    """A real C arena of ``n`` COMPLETE slots, the record published for all, a real #1459 index listing every stem
    except the slots in ``missing``; and a backend counting the index lookups of the array form."""
    from flliper.srt.mem_cache import hicache_storage as hs
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena
    from flliper.srt.mem_cache.storage.file.l3_index import L3Index

    a = ShmArena(str(tmp_path / f"arena-{name}-{sb}.bin"), sb, n)
    stems = [_stem(i) for i in range(n)]
    for i in range(0, n, 1024):
        part = stems[i:i + 1024]
        got = a.claim_slots(part, [sb] * len(part))
        a.complete_slots([g[0] for g in got], [g[2] for g in got], [(0, sb)])
    sl, gens, _klo, _khi = a.complete_census()
    assert len(sl) == n
    l3s.publish(a, sl, gens)
    idx = L3Index(str(tmp_path / f"l3idx-{name}.bin"), cap=cap)
    miss = {a.slot_stem(int(s)) for s in missing}
    idx.add([st for st in stems if st not in miss])
    be = types.SimpleNamespace(_l3idx=idx, asked=[])
    be.l3_listed = types.MethodType(hs.HiCacheFile.l3_listed, be)
    _ptrs = types.MethodType(hs.HiCacheFile.l3_listed_ptrs, be)

    def _counting(ptrs, n_):
        be.asked.append(int(n_))
        return _ptrs(ptrs, n_)

    be.l3_listed_ptrs = _counting
    return a, be, idx


@pytest.fixture(autouse=True)
def _quiet_request(monkeypatch):
    """``request`` only queues for a write-behind this test has not got; record what was asked."""
    seen = []
    monkeypatch.setattr(l3s, "request", lambda arena, slots: seen.append(np.asarray(slots).copy()) or len(slots))
    monkeypatch.setenv(l3s.STEM_PTRS_ENV, "1")
    monkeypatch.setattr(l3s, "LISTED_MEMO", True)
    return seen


def _both(a, be, monkeypatch):
    """(census with memo, census without memo) of the same arena state; the memo survives between calls."""
    monkeypatch.setattr(l3s, "LISTED_MEMO", False)
    off = _REAL_UNSEC(a, be)
    monkeypatch.setattr(l3s, "LISTED_MEMO", True)
    on = _REAL_UNSEC(a, be)
    return on, off


def _key_lo(a, slot):
    sl, _g, klo, _khi = a.complete_census()
    return int(np.asarray(klo)[np.nonzero(np.asarray(sl) == slot)[0][0]])


def _gen(a, slot):
    return int(np.asarray(a.slot_gens_np(np.array([slot], dtype=np.int64)))[0])


def test_memo_census_equals_full_census_through_every_orderly_transition(tmp_path, monkeypatch):
    n = 64
    missing = (3, 9, 40)
    a, be, idx = _arena(tmp_path, n, missing=missing)
    # 0. start: the uncopied are exactly the unlisted
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off) and on.shape[0] == len(missing)
    on2, _ = _both(a, be, monkeypatch)            # memo filled now: same answer from the memo
    assert np.array_equal(on2, off)
    # 1. a missing copy lands (index lists it): both read it secured
    idx.add([a.slot_stem(9)])
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off) and on.shape[0] == len(missing) - 1
    # 2. the L3 LRU takes slot 20's file: index entry gone AND the record voided (on_evict)
    st20 = a.slot_stem(20)
    idx.remove([st20])
    l3s.invalidate_stems(a, [st20])
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off) and _key_lo(a, 20) in set(on.tolist()), "a file the LRU took is no copy"
    # 3. index lag: the record is re-published for that generation but the index does not list the file yet --
    #    the memo must not answer from its pre-eviction entry (invalidate_stems cleared it)
    l3s.publish(a, np.array([20]), np.array([_gen(a, 20)]))
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off) and _key_lo(a, 20) in set(on.tolist())
    # 4. slot 30 re-completed under a NEW generation with another stem; the record names the new generation, the
    #    index lists only the OLD stem -- the memo (keyed by generation) must ask again
    old_gen = _gen(a, 30)
    a.free_slots([30], reason="test")
    got = a.claim_slots(["new-stem-for-slot-30"], [256])
    assert int(got[0][0]) == 30, "the only free slot"
    a.complete_slots([got[0][0]], [got[0][2]], [(0, 256)])
    assert _gen(a, 30) != old_gen
    l3s.publish(a, np.array([30]), np.array([_gen(a, 30)]))
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off) and _key_lo(a, 30) in set(on.tolist()), "a new generation is a new page"


def test_index_gone_record_not_yet_voided_is_refused_at_the_free_decision(tmp_path, monkeypatch):
    """S109-A1's own window (index entry removed, on_evict not yet run): the memo may still keep the page off the keep
    list -- the free decision (``verify_candidates`` -> ``secured``) asks the index WITHOUT the memo and refuses."""
    a, be, idx = _arena(tmp_path, 32)
    _REAL_UNSEC(a, be)                             # memo: every slot confirmed
    st = a.slot_stem(7)
    idx.remove([st])                               # ... and the LRU's on_evict has not run yet
    on = _REAL_UNSEC(a, be)
    assert _key_lo(a, 7) not in set(on.tolist())   # the documented window: the memo still answers "copied"
    sec = l3s.verify_candidates(a, [(7, _key_lo(a, 7), 0, 256)], be)
    assert sec["mask"] == [False] and sec["on_disk"] == 0, "the claim must not free a page without its copy"
    # and once on_evict ran, the census itself says so again
    l3s.invalidate_stems(a, [st])
    assert _key_lo(a, 7) in set(_REAL_UNSEC(a, be).tolist())


def test_second_census_asks_the_index_only_for_new_copies(tmp_path, monkeypatch):
    n = 4096
    stale = list(range(5, 4096, 97))
    a, be, _idx = _arena(tmp_path, n)
    sl, gens, _k, _h = a.complete_census()
    sl = np.asarray(sl, dtype=np.int64)
    gens = np.asarray(gens, dtype=np.int64)
    # these slots' record says "another generation": not copied yet (the write-behind has not visited them)
    bad = gens.copy()
    bad[np.isin(sl, stale)] += 1000
    l3s.publish(a, sl, bad)
    be.asked.clear()
    first = _REAL_UNSEC(a, be)
    assert first.shape[0] == len(stale)
    assert sum(be.asked) == n - len(stale)
    be.asked.clear()
    m0 = dict(l3s.MEMO_STATS)
    again = _REAL_UNSEC(a, be)
    assert np.array_equal(again, first)
    assert sum(be.asked) == 0, "nothing changed: no index lookup at all"
    assert l3s.MEMO_STATS["memo"] - m0["memo"] == n - len(stale)
    # the write-behind secures the stale ones (record = current generation): only those are asked
    l3s.publish(a, sl, gens)
    be.asked.clear()
    third = _REAL_UNSEC(a, be)
    assert third.shape[0] == 0
    assert sum(be.asked) == len(stale)


def test_memo_off_asks_every_slot_every_time(tmp_path, monkeypatch):
    a, be, _idx = _arena(tmp_path, 1024)
    monkeypatch.setattr(l3s, "LISTED_MEMO", False)
    _REAL_UNSEC(a, be)
    be.asked.clear()
    _REAL_UNSEC(a, be)
    assert sum(be.asked) == 1024


def test_a_new_backend_starts_a_fresh_memo(tmp_path, monkeypatch):
    """The memo belongs to one index: another backend (another #1459 table) is asked from scratch."""
    a, be, _idx = _arena(tmp_path, 512)
    _REAL_UNSEC(a, be)
    from flliper.srt.mem_cache import hicache_storage as hs
    from flliper.srt.mem_cache.storage.file.l3_index import L3Index

    empty = L3Index(str(tmp_path / "l3idx-empty.bin"), cap=1 << 12)
    be2 = types.SimpleNamespace(_l3idx=empty)
    be2.l3_listed = types.MethodType(hs.HiCacheFile.l3_listed, be2)
    be2.l3_listed_ptrs = types.MethodType(hs.HiCacheFile.l3_listed_ptrs, be2)
    assert _REAL_UNSEC(a, be2).shape[0] == 512, "an index that lists nothing: every page uncopied"


def test_census_cost_characterisation(tmp_path, monkeypatch):
    """Characterisation (not a verdict on the metal): 262144 COMPLETE slots, all copied -- the census of a claim
    before and after the memo is warm. Bound here: the warm census at most a third of the cold one."""
    a, be, _idx = _arena(tmp_path, 262144, cap=1 << 19, name="big")

    def best(k):
        b = 1e9
        for _ in range(k):
            t = time.perf_counter()
            _REAL_UNSEC(a, be)
            b = min(b, time.perf_counter() - t)
        return b * 1e3

    monkeypatch.setattr(l3s, "LISTED_MEMO", False)
    cold = best(3)
    monkeypatch.setattr(l3s, "LISTED_MEMO", True)
    _REAL_UNSEC(a, be)            # fill the memo
    warm = best(3)
    print("PPHASE-GAPS-1024 census ms no-memo=%.1f memo-warm=%.1f slots=262144" % (cold, warm))
    assert warm < cold / 3.0


# --- review R1-R4 (review-pphase-gaps-1024.md), adapted -----------------------------------------------------------


def _sister(path, sb, n):
    """Another process's view of the same arena: its own ShmArena object (own memo), the same record file."""
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    b = ShmArena(path, sb, n)
    assert l3s.arrays(b, create=True) is not None
    return b


def _memo_of(a):
    cur = getattr(a, l3s._MEMO_ATTR, None)
    return None if cur is None else cur[1]


def _path_of(tmp_path, name="a", sb=256):
    return str(tmp_path / f"arena-{name}-{sb}.bin")


def test_r1_sister_void_is_seen_through_a_warm_memo(tmp_path, monkeypatch):
    """MUSS 1 (record gate): the evicting process voided the SHARED record; this process's memo still holds the old
    generation (``_forget_listed`` ran on the sister's object only). The record gate, not the memo, decides."""
    n = 64
    a, be, idx = _arena(tmp_path, n)
    _REAL_UNSEC(a, be)
    b = _sister(_path_of(tmp_path), 256, n)
    st = a.slot_stem(11)
    idx.remove([st])
    assert l3s.invalidate_stems(b, [st]) == 1
    m = _memo_of(a)
    assert m is not None and int(m[11]) == _gen(a, 11), "precondition: this process's memo is untouched"
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off)
    assert _key_lo(a, 11) in set(on.tolist()), "a record voided by the sister process must read uncopied here"


def test_r2n_census_that_saw_the_void_forgets_the_memo(tmp_path, monkeypatch):
    """SOLL 2 (narrowed R2): this process ran a census while the sister's void stood -- the memo entry is forgotten,
    so the sister's later same-generation re-publish under index lag is asked again (census(memo)==census(no memo))."""
    n = 64
    a, be, idx = _arena(tmp_path, n)
    _REAL_UNSEC(a, be)
    b = _sister(_path_of(tmp_path), 256, n)
    st = a.slot_stem(20)
    idx.remove([st])
    l3s.invalidate_stems(b, [st])
    _REAL_UNSEC(a, be)                                       # this process sees the record negative once
    assert int(_memo_of(a)[20]) == -1
    l3s.publish(b, np.array([20]), np.array([_gen(b, 20)]))  # sister's write-behind: record back, index not yet
    on, off = _both(a, be, monkeypatch)
    assert _key_lo(a, 20) in set(off.tolist())
    assert np.array_equal(on, off)


def test_r2_residue_void_and_republish_between_two_censuses_is_refused_at_the_free_decision(tmp_path, monkeypatch):
    """The named residue: void AND same-generation re-publish by the sister between two censuses of this process --
    the memo still answers "copied" (documented), and the memo-free free decision refuses the page (R2b)."""
    n = 32
    a, be, idx = _arena(tmp_path, n)
    _REAL_UNSEC(a, be)
    b = _sister(_path_of(tmp_path), 256, n)
    st = a.slot_stem(5)
    idx.remove([st])
    l3s.invalidate_stems(b, [st])
    l3s.publish(b, np.array([5]), np.array([_gen(b, 5)]))
    on, off = _both(a, be, monkeypatch)
    assert _key_lo(a, 5) in set(off.tolist())
    assert _key_lo(a, 5) not in set(on.tolist()), "the residue as documented (if this flips, update the module note)"
    sec = l3s.verify_candidates(a, [(5, _key_lo(a, 5), 0, 256)], be)
    assert sec["mask"] == [False] and sec["on_disk"] == 0


def _clear_backend(tmp_path, a, idx):
    """A minimal HiCacheFile for the real ``clear`` (its own store dir, this arena opened)."""
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    (store / "somefile").write_bytes(b"x")
    ev = types.SimpleNamespace(cleared=0, forgot=[])
    ev.clear = lambda: setattr(ev, "cleared", ev.cleared + 1)
    ev.forget = lambda stems: ev.forgot.extend(stems)
    hb = types.SimpleNamespace(file_path=str(store), _evictor=ev, metadata_cache=None, _arenas={"kv": a},
                               _l3idx=idx)
    hb._l3_index = lambda: idx
    return hb


def test_r3_store_clear_voids_the_record_and_the_memo(tmp_path, monkeypatch):
    """SOLL 1: ``HiCacheFile.clear`` empties the index and unlinks every file -- with a warm memo the census read the
    whole arena as copied before. Now the record is voided: census(memo) == census(no memo) == every page."""
    from flliper.srt.mem_cache import hicache_storage as hs

    n = 64
    a, be, idx = _arena(tmp_path, n)
    _REAL_UNSEC(a, be)
    hb = _clear_backend(tmp_path, a, idx)
    assert hs.HiCacheFile.clear(hb, force=True) is True
    on, off = _both(a, be, monkeypatch)
    assert off.shape[0] == n and np.array_equal(on, off)
    assert int((np.asarray(l3s.arrays(a)[0]) >= 0).sum()) == 0, "the shared record is void"


def test_r3b_free_decision_refuses_after_index_clear(tmp_path, monkeypatch):
    """Even a raw index clear (no record void at all) is refused at the free decision."""
    n = 32
    a, be, idx = _arena(tmp_path, n)
    _REAL_UNSEC(a, be)
    idx.clear()
    cands = [(s, _key_lo(a, s), 0, 256) for s in range(n)]
    sec = l3s.verify_candidates(a, cands, be)
    assert sec["on_disk"] == 0 and not any(sec["mask"])


def test_r4_per_slot_form_with_memo_on_equals_memo_off(tmp_path, monkeypatch):
    """SOLL 4: the per-slot fallback (backend without the pointer twin) with the memo ON."""
    n = 128
    a, _be, idx = _arena(tmp_path, n, name="p")
    be = types.SimpleNamespace(_l3idx=idx)
    be.l3_listed = lambda stems: [bool(x) for x in idx.has(list(stems))] if stems else []
    for s_ in (3, 50, 90):
        idx.remove([a.slot_stem(s_)])
    f0 = l3s.FAST_STATS["fallback"]
    on, off = _both(a, be, monkeypatch)
    assert l3s.FAST_STATS["fallback"] > f0, "precondition: the per-slot form ran"
    assert np.array_equal(on, off) and on.shape[0] == 3
    m = _memo_of(a)
    assert m is not None and int((m >= 0).sum()) == n - 3, "only the listed slots are remembered"
    idx.remove([a.slot_stem(7)])
    l3s.invalidate_stems(a, [a.slot_stem(7)])
    on, off = _both(a, be, monkeypatch)
    assert np.array_equal(on, off) and on.shape[0] == 4
