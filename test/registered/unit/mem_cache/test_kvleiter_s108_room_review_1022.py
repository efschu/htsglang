"""Review KVLEITER-1022 (11c1dc2b35): danger directions of the turn end's stage-(i) room the author's tests leave open.

Real C arena, CPU only. Each test names the danger direction it holds:

* R1 a page a running read is walking (S24 ``fill_protected_stems``) never leaves L2 at a turn end;
* R2 a page whose L3 file the L3 LRU took (record yes, #1459 index no) is no copy: it stays in L2;
* R3 a page still being written (CLAIMED, an in-flight L1->L2 write of any rank) is never turn-end room;
* R4 the room is bounded by the demand: ``make_secured_room(k)`` frees at most k slots, never the whole arena;
* R5 (xfail strict, pre-existing, Auflage A1): a claim that found a page COMPLETE (status 2) takes its reader
  reference AFTER the C claim returned; another process's clock (from 11c1dc2b35 also every D rank's turn end)
  can evict the slot in between -- the refused reference is not checked and the claim hands out a freed slot;
* R6 (xfail strict, Auflage A2): with the hand-off/park keep list unreadable, stage (i) must free nothing -- today
  it falls back to the keep-less clock and frees kept pages (with copy: read back from L3, no token loss).
"""
from __future__ import annotations

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

L3_ASYNC_REAL = True   # keep the scheduler forms of the S109 seam: this module tests what they free
_needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (real C arena)")


def _arena(tmp_path, n=6):
    from test_s109_backup_wall_aux_leaves_1020 import _unreferenced_full_arena

    return _unreferenced_full_arena(tmp_path, n)


def _secure(p, arena, slots):
    from test_s109_backup_wall_aux_leaves_1020 import _write_behind

    _write_behind(arena, p._backend._l3idx, slots)


@_needs_gcc
def test_r1_a_page_a_running_read_walks_is_never_turn_end_room(tmp_path, monkeypatch):
    from flliper.srt.mem_cache import hicache_storage as hs

    p, arena, slots = _arena(tmp_path)
    _secure(p, arena, slots)
    prot = [arena.slot_stem(int(s)) for s in slots[:2]]
    monkeypatch.setattr(hs, "fill_protected_stems", lambda: list(prot))
    assert p.make_secured_room(6) == 4
    assert arena.slot_states(slots[:2]) == [2, 2], "the read's not yet walked pages stay in L2"


@_needs_gcc
def test_r2_a_page_whose_l3_file_the_lru_took_is_no_copy(tmp_path):
    p, arena, slots = _arena(tmp_path)
    _secure(p, arena, slots)
    for s in slots[:3]:                                   # the L3 LRU unlinked these files (index), record stays
        p._backend._l3idx.files.discard(arena.slot_stem(int(s)))
    assert p.make_secured_room(6) == 3
    assert arena.slot_states(slots[:3]) == [2, 2, 2], "record without index is no copy: the page stays"


@_needs_gcc
def test_r3_a_page_in_flight_is_never_turn_end_room(tmp_path):
    p, arena, slots = _arena(tmp_path)
    _secure(p, arena, slots)
    stems = p._stems(["inflight0", "inflight1"])
    got = arena.claim_slots(stems, [p._page_bytes] * 2)
    fresh = [int(s) for s, st, _g in got if int(st) == 0]
    assert len(fresh) == 2 or p.make_secured_room(2) >= 0     # a full arena: room first, then claim
    if len(fresh) < 2:
        got = arena.claim_slots(stems, [p._page_bytes] * 2)
        fresh = [int(s) for s, st, _g in got if int(st) == 0]
    assert len(fresh) == 2
    _secure(p, arena, fresh)                              # even a (wrong) record does not make a CLAIMED page room
    p.make_secured_room(len(slots) + 2)
    assert arena.slot_states(fresh) == [1, 1], "a CLAIMED page (write in flight) stays claimed"


@_needs_gcc
def test_r4_the_room_is_bounded_by_the_demand(tmp_path):
    p, arena, slots = _arena(tmp_path)
    _secure(p, arena, slots)
    assert p.make_secured_room(2) == 2
    assert sum(1 for s in arena.slot_states(slots) if s == 2) == len(slots) - 2, "no reserve beyond the demand"


@_needs_gcc
@pytest.mark.xfail(strict=True, reason="pre-existing (#1424f): the status-2 reader reference is not checked; "
                                       "Auflage A1 of review KVLEITER-1022")
def test_r5_a_found_complete_page_evicted_before_its_reference_is_not_handed_out(tmp_path):
    p, arena, slots = _arena(tmp_path)
    _secure(p, arena, slots)
    real_ref = arena.ref_slots
    fired = []

    def _ref_after_other_rank_clock(sl, delta):
        if delta > 0 and not fired:
            fired.append(1)
            p.make_secured_room(len(slots))              # another D rank's turn end, between claim and reference
        return real_ref(sl, delta)

    arena.ref_slots = _ref_after_other_rank_clock
    rows = p.alloc_write(["u0"])                          # u0 is COMPLETE, unreferenced, copied: status 2
    arena.ref_slots = real_ref
    assert fired
    if rows is None:
        return                                            # a refused claim is the correct outcome
    from flliper.srt.mem_cache.pool_host.arena_pool import _psz

    slot = (int(rows.tolist()[0]) - int(p.staging_rows)) // int(_psz(p))
    assert slot == slots[0], "status 2: the claim names u0's own slot"
    assert arena.slot_states([slot]) == [2], "the claim handed out a slot the clock freed under it"


@_needs_gcc
@pytest.mark.xfail(strict=True, reason="Auflage A2 of review KVLEITER-1022: stage (i) without a keep list is not "
                                       "fail-closed")
def test_r6_without_a_readable_keep_list_the_turn_end_frees_nothing(tmp_path, monkeypatch):
    from flliper.srt.pdflip import handoff_pending as hp

    p, arena, slots = _arena(tmp_path)
    _secure(p, arena, slots)

    def _boom(pool):
        raise OSError("handoff dir unreadable")

    monkeypatch.setattr(hp, "keep_for", _boom)
    assert p.make_secured_room(6) == 0
