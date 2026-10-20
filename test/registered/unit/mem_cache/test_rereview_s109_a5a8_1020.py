"""Re-review S109-A5..A8 @ 8e8e4272a7 (nf-s109-rereview-a5a8-1020): counter-tests.

Q1 (A5 race): the write-behind's stat -> publish window against the LRU unlink. The pass stats a ``new`` page
(on disk), the L3 LRU unlinks that file (index entry gone, ``on_evict`` voids record + baseline), then the pass
publishes the generation it saw -- baseline and shared record say "copied" again while the file is gone. The page
is never ``new`` again: the R11 leak, through a race.

Q2 (A5 bound): ping-pong at the L3 cap, with the REAL ``LRUFileEvictor`` (cap, LRU, on_evict hook) and the real
pass. (a) cap above the L2-resident bytes: one trigger, the rewrites end within a pass; (b) cap below them (the
necessary condition named): the pass rewrites forever -- characterisation of where the bound comes from.

Q3 (A8 TOCTOU): the aged reaper frees the hand-over slot between ``_hand_drain``'s state/generation check and the
copy -- the copy runs on a slot that is no longer the candidate.
"""
from __future__ import annotations

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


def _boot(tmp_path, monkeypatch, cap=None):
    import test_l3_reuse_after_restart_0928 as RR
    from flliper.srt.mem_cache import l3_write_behind as gate

    gate._reset_for_tests()
    monkeypatch.setenv("FLLIPER_HICACHE_FILE_BACKEND_EVICT_OFFPATH", "0")
    be = RR._boot(tmp_path, monkeypatch, "boot1")
    if cap is not None:
        from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        idx = be._l3idx
        be._evictor = LRUFileEvictor(
            be.file_path, RR.SFX, tp_rank=0, writes_shared_keys=False,
            path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
            extra_config={"max_size": int(cap), "eviction_ratio": 0.9},
            on_evict=be._l3sec_on_evict,
        )
        be._evictor.l3_index = idx
        assert be._evictor.enabled
    from flliper.srt.mem_cache import l3_secured as L3S

    for a in be._arenas.values():
        L3S.register_backend(a, be)
        L3S.arrays(a, create=True)
    return RR, be


# ------------------------------------------------------------------------------------------------------- Q1 (A5)
def test_q1_a5_an_lru_unlink_between_the_pass_stat_and_its_publish_must_not_leave_the_page_copied(
        tmp_path, monkeypatch):
    from flliper.srt.mem_cache import l3_secured as L3S

    RR, be = _boot(tmp_path, monkeypatch)
    hs = RR._hashes("s")
    RR._prefill(be, hs)
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES
    kv = be._arenas[RR.KV_TOTAL]
    victim = be._get_suffixed_key(hs[0])
    (slot, _st), = kv.find_slots([victim])
    # the page is ``new`` for this process while its file is on disk (the sibling group wrote it, or the first
    # visit after a restart): baseline and shared record do not hold this generation
    kv._l3wb_sec[0][slot] = -1
    L3S.arrays(kv)[0][slot] = -1
    orig = be._stat_stems

    def _stat_then_lru_unlink(stems):
        out = orig(stems)
        if victim in stems:
            # the L3 LRU (its own thread, evictor lock) takes the file right after the stat: unlink, #1459 index,
            # on_evict -> invalidate_stems (record AND baseline)
            os.remove(be._existing_path(victim))
            be._l3idx.remove([victim])
            be._l3sec_on_evict(victim)
        return out

    be._stat_stems = _stat_then_lru_unlink
    be.l3_write_behind_pass()
    be._stat_stems = orig
    assert not os.path.exists(be._existing_path(victim))
    assert L3S.secured(kv, [slot], be).tolist() == [False], "the index says: no copy"
    r = be.l3_write_behind_pass()
    assert r["written"] == 1 and os.path.exists(be._existing_path(victim)), (
        "the page is COMPLETE in L2 without an L3 file, but the pass published the stale stat over the "
        "invalidation (baseline=%d record=%d gen=%d): it is never new again -- R11 through the race; %r"
        % (int(kv._l3wb_sec[0][slot]), int(L3S.arrays(kv)[0][slot]),
           int(np.asarray(kv.slot_gens_np([slot]))[0]), r))


# ------------------------------------------------------------------------------------------------------- Q2 (A5)
def _set_cap(be, files_worth):
    """The cap in ALLOCATED bytes (commit reconciles to the filesystem's block size): ``files_worth`` page files."""
    ev = be._evictor
    with ev._lock:
        n = len(ev._lru)
        unit = ev._total_bytes // max(1, n)
        ev.max_size_bytes = int(files_worth * unit)
    return unit


def test_q2a_a5_ping_pong_ends_when_the_cap_holds_more_than_l2(tmp_path, monkeypatch):
    """L2-resident set B is the OLDEST in the LRU; a younger L3-only set A sits behind it; new data C at the cap
    evicts B's files first (the worst order for A5). The rewrites of B evict A (L3-only) and end: the passes
    converge to written == 0 and every B page has its copy again."""
    from flliper.srt.mem_cache import l3_secured as L3S

    RR, be = _boot(tmp_path, monkeypatch, cap=1 << 40)
    hb, ha, hc = RR._hashes("b"), RR._hashes("a"), RR._hashes("c")
    RR._prefill(be, hb)
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES       # B: oldest files, stays in L2
    RR._prefill(be, ha)
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES       # A: then leaves L2 (L3-only)
    for h in ha:
        for tot, key in ((RR.KV_TOTAL, be._get_suffixed_key(h)),
                         (RR.Q_TOTAL, be._get_suffixed_key(f"{h}.{RR.PoolName.QSA_INDEXER}"))):
            (s_, _), = be._arenas[tot].find_slots([key])
            assert s_ >= 0
            be._arenas[tot].free_slots([s_])
    _set_cap(be, 40)                               # 40 files < B 16 + A 16 + C 16: C's writes evict (B first)
    RR._prefill(be, hc)                            # new data
    passes = [be.l3_write_behind_pass()["written"] for _ in range(8)]
    print("Q2a passes", passes, "evictor", {k: v for k, v in be._evictor.stats().items() if "evict" in k or "entries" in k})
    assert sum(passes) > 2 * RR.PAGES, ("the scenario must make A5 rewrite B pages", passes)
    assert passes[-1] == 0 and passes[-2] == 0, passes
    assert sum(passes) <= 2 * RR.PAGES + 2 * RR.PAGES, ("C once plus each B page at most once", passes)
    kv = be._arenas[RR.KV_TOTAL]
    bslots = [s for s, _ in kv.find_slots([be._get_suffixed_key(h) for h in hb])]
    assert all(L3S.secured(kv, bslots, be).tolist()), "every L2-resident page has its copy again"


def test_q2b_a5_the_bound_needs_the_cap_above_l2_resident_bytes(tmp_path, monkeypatch):
    """Characterisation of the necessary condition: cap BELOW the L2-resident bytes -> each pass evicts a resident
    page's file and the next pass rewrites it, forever (bounded per pass only by the pass budget)."""
    RR, be = _boot(tmp_path, monkeypatch, cap=1 << 40)
    RR._prefill(be, RR._hashes("b"))
    assert be.l3_write_behind_pass()["written"] == 2 * RR.PAGES
    _set_cap(be, RR.PAGES)                         # half of the 16 resident files
    RR._prefill(be, RR._hashes("c"))               # one trigger: new data
    ws = [be.l3_write_behind_pass()["written"] for _ in range(6)]
    print("Q2b passes", ws)
    assert all(w > 0 for w in ws[1:]), ("cap < L2-resident: perpetual rewrite", ws)


# ------------------------------------------------------------------------------------------------------- Q3 (A8)
def test_q3_a8_the_reaper_between_the_check_and_the_copy_must_not_get_a_copy(tmp_path):
    """``_hand_drain`` reads state and generation, THEN copies. The aged reaper (any process, 60 s) can free the
    slot in between -- a new claim may fill it; ``arena_secure_to_disk`` names the page by the CANDIDATE's key
    (``arena._stems[(lo, hi)]``) and would write the slot's new bytes under the old stem. Required: no copy of a
    slot that is not the candidate any more at copy time (or the copy is voided and counted dropped)."""
    from flliper.srt.mem_cache import l3_secured as L3S

    from test_s109_backup_wall_aux_leaves_1020 import _unreferenced_full_arena

    p, arena, slots = _unreferenced_full_arena(tmp_path)
    be = p._backend
    seen = []

    def _copy(ar, cands, writer="x"):
        s0 = int(cands[0][0])
        seen.append((int(ar.slot_states([s0])[0]), int(np.asarray(ar.slot_gens_np([s0]))[0])))
        return {"written": 1, "lost": 0}

    be.arena_secure_to_disk = _copy
    cands = arena.evict_candidates(1)
    s0 = cands[0][0]
    import flliper.srt.mem_cache.storage.file.hicache_arena as HA

    gen = HA._evict_gens_for(arena.path).get(s0)
    orig = arena.slot_gens_np
    hit = [0]

    def _gens_then_reap(sl):
        out = orig(sl)
        if not hit[0]:
            hit[0] = 1
            arena.free_slots([s0])          # the aged reaper fires right after the check read the generation
        return out

    arena.slot_gens_np = _gens_then_reap
    d0 = L3S.HAND_STATS["dropped_without_l3"]
    with L3S._HAND_LOCK:
        L3S._HAND.append((arena, be, tuple(cands[0]), "test", gen))
    L3S._hand_drain()
    arena.slot_gens_np = orig
    assert seen == [] or L3S.HAND_STATS["dropped_without_l3"] == d0 + 1, (
        "copied a slot that was reaped after the check: state/gen at copy %r, candidate gen %r" % (seen, gen))
