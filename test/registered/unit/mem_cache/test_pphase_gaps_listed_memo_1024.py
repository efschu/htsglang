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

THE FIX (l3_secured ``LISTED_MEMO``): per arena, the generation at which a slot's stem was last FOUND listed; the
record is still read for every slot on every census, only the index question is skipped for a slot confirmed at its
current generation. The danger direction is a page reported COPIED that is not (the keep list would let the clock
take it): these tests hold census(memo) == census(no memo) through every orderly transition (new copies, the L3 LRU
taking a file, a record re-published under index lag, a slot re-completed under a new generation), and they hold the
one documented window -- index entry gone, record not yet voided -- at the claim's free decision
(``verify_candidates`` asks the index without the memo and refuses the page).
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
