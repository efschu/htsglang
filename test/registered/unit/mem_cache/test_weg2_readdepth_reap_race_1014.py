"""READ-DEPTH 1014 (NF h16h 853534a9ff, boot _211215, P): the three P stages
fill ONE shared KV arena from L3, and each of them calls ``arena.reap_stale()``
at the end of its own clock eviction (``HiCacheFile._arena_evict_to_disk``).
The old reap freed EVERY EVICTING slot with refcount 0 -- including the
candidates another LIVE stage was still copying to disk. The next fill claimed
such a slot fresh (same generation), the live evictor's own free then freed the
fill's claim (generation++), and the fill's completion came back 3:

* PP0 22:30:02 ``L3-FILL LATE-COMPLETE REFUSED n=1 ... statuses=[3]`` ->
  weg2-51-579 ``landed=300 of 1556`` (PP1 1556, PP2 694, told 16384);
* PP1 22:47:16 ``n=5 ... statuses=[3] total=770`` -> weg2-68-683 PP0/PP1
  landed 0 (PP0 waited 5 s on the vanished stem), cached 0;
* PP0+PP2 22:59:50 ``n=2 ... total=271`` (263 slots on PP0 in ONE fill) ->
  weg2-78-784 ``landed=761 of 1148`` on both, told 48384.

Port of 27B dae863c7c8 (1): only a slot EVICTING for >= 60 s is stale, taken
privately (S_REAPING), its generation moved on; ``arena_free_slots`` never
clobbers a slot the reaper holds.

Each test drives the REAL C arena (gcc) and, where it is the fill that
loses, the REAL ``HiCacheFile.arena_fill_from_disk``. Red on b76e5fce79 (and
on desk/nf-h18port-1014 e3c9a228ec: none of the port fixes touches the reap),
green with the fix; the MUTANT (age gate 0 = the old instant reap) is red."""

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
SLOTS = 8


def _full_arena(tmp_path):
    """An arena with every slot COMPLETE (the boot's 6485-slot P arena ran at
    complete=6284 and above), keys old0..old7."""
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, SLOTS)
    stems = [f"old{i}" for i in range(SLOTS)]
    claims = arena.claim_slots(stems, [TOTAL] * SLOTS)
    assert [c[1] for c in claims] == [0] * SLOTS
    assert arena.complete_slots([c[0] for c in claims], [c[2] for c in claims],
                                [(0, TOTAL)]) == [1] * SLOTS
    return arena


def _old_reap(arena):
    """MUTANT / the base behaviour: the age gate removed (every EVICTING slot
    with refcount 0 is reaped at once)."""
    lib = arena._lib
    if hasattr(lib, "arena_reap_stale_aged"):
        return int(lib.arena_reap_stale_aged(arena._base, 0))
    return int(lib.arena_reap_stale(arena._base))


def _interleave(arena, reap):
    """Stage A (evictor, live) takes 4 candidates and is copying them to disk;
    stage B's eviction tail reaps; stage C (fill) claims 4 fresh stems; A
    frees its candidates; C completes. Returns C's completion statuses
    (None = C got no slot while A was live)."""
    cands = arena.evict_candidates(4)
    assert len(cands) == 4
    reap(arena)                                         # B: '... arena.reap_stale()'
    claims = arena.claim_slots([f"new{i}" for i in range(4)], [TOTAL] * 4)
    mine = [(s, g) for s, st, g in claims if st == 0]
    arena.free_slots([c[0] for c in cands], reason="evict_to_disk")   # A finishes
    if not mine:
        return None
    return arena.complete_slots([s for s, _ in mine], [g for _, g in mine], [(0, TOTAL)])


def test_a_live_evictors_candidates_are_not_reaped(tmp_path):
    arena = _full_arena(tmp_path)
    statuses = _interleave(arena, lambda a: a.reap_stale())
    # base: [3, 3, 3, 3] -- the fill's claims were freed under it ('recycled')
    assert statuses is None or 3 not in statuses
    # A's free went through: four FREE slots, every one claimable fresh now
    claims = arena.claim_slots([f"new{i}" for i in range(4)], [TOTAL] * 4)
    assert [c[1] for c in claims] == [0] * 4
    assert arena.complete_slots([c[0] for c in claims], [c[2] for c in claims],
                                [(0, TOTAL)]) == [1] * 4


def test_mutant_age_gate_off_recycles_the_fills_claims(tmp_path):
    """MUTANT: the reap without the age gate is the metal's status 3."""
    arena = _full_arena(tmp_path)
    assert _interleave(arena, _old_reap) == [3, 3, 3, 3]


def test_a_dead_evictors_slots_are_still_reaped_after_the_age(tmp_path):
    """The reap's purpose stays: an evictor that died between candidates and
    free leaves EVICTING slots; past the age they are freed, generation moved
    on (a late completion on them is refused)."""
    arena = _full_arena(tmp_path)
    cands = arena.evict_candidates(2)
    assert arena.reap_stale() == 0                      # young: a live evictor's
    import time
    time.sleep(0.02)
    assert arena.reap_stale(min_age_s=0.01) == 2        # aged: its evictor died
    claims = arena.claim_slots(["x0", "x1"], [TOTAL] * 2)
    assert sorted(c[0] for c in claims) == sorted(c[0] for c in cands)
    assert all(c[1] == 0 for c in claims)


def _backend(root, arena, evictor):
    """A HiCacheFile whose clock evict is the live evictor finishing: stage A
    frees its candidates (the room the fill then claims)."""
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}

    def _evict(_arena, want, need=None):
        return evictor()

    be._arena_evict_to_disk = _evict
    return be


def _fill_case(tmp_path, monkeypatch, reap):
    """weg2-78-784 in small: the fill of PP0 needs 4 L3 pages, the arena is
    full, PP2's clock holds 4 candidates mid copy; PP1's eviction tail reaps."""
    root = tmp_path / "store"
    root.mkdir()
    stems = [f"p{i}" for i in range(4)]
    for i, s in enumerate(stems):
        (root / (s + ".bin")).write_bytes(bytes([0x50 + i]) * TOTAL)
    arena = _full_arena(tmp_path)
    cands = arena.evict_candidates(4)                   # PP2: EVICTING, copying to disk
    state = {"freed": False}

    def a_finishes():
        if not state["freed"]:
            state["freed"] = True
            arena.free_slots([c[0] for c in cands], reason="evict_to_disk")
            return len(cands)
        return 0

    reap(arena)                                         # PP1: '... arena.reap_stale()'
    real_read = hs.l3_read_pages_parallel

    def read_while_a_finishes(pio, paths, total, ptrs):
        rc = real_read(pio, paths, total, ptrs)
        a_finishes()                                    # PP2's free lands mid-fill
        return rc

    monkeypatch.setattr(hs, "l3_read_pages_parallel", read_while_a_finishes)
    be = _backend(str(root), arena, a_finishes)
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    return arena, out


def test_case_78_784_the_fill_lands_every_page(tmp_path, monkeypatch):
    arena, out = _fill_case(tmp_path, monkeypatch, lambda a: a.reap_stale())
    assert all(s is not None for s in out), out         # base: [None]*4 -- landed 0 of 4
    for i, s in enumerate(out):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([0x50 + i]) * 4


def test_mutant_case_78_784_old_reap_cuts_the_fill(tmp_path, monkeypatch):
    _arena, out = _fill_case(tmp_path, monkeypatch, _old_reap)
    assert out[0] is None                               # LATE-COMPLETE 3: the prefix ends at page 0


def _death_0109(tmp_path, reap):
    """The h16h death 01:09:10 (PP0, weg2-208-1751): ``ArenaChainMismatch #1424
    CHAIN MISMATCH node=1958 page=201 of 1505 bad=47: key slot -1 (state 0)``.
    PP0's fill landed and REFERENCED all 1552 pages (READ-STAGES pages=1552);
    47 of them sat in slots another stage's live evictor still held as
    candidates (reaped early by a third stage's ``reap_stale``); the evictor's
    own free then freed PP0's COMPLETE, referenced page and unlinked its key
    (``arena_free_slots`` checks no reference). PP1/PP2, joined on those stems,
    polled them vanish (``JOIN-WAIT still_claimed=47 waited_ms=5004``); PP0's
    tree kept naming the slots, the load-back's chain proof found no key."""
    arena = _full_arena(tmp_path)
    cands = arena.evict_candidates(4)                    # stage A, live, copying to disk
    reap(arena)                                          # stage B's eviction tail
    stems = [f"page{i}" for i in range(4)]
    claims = arena.claim_slots(stems, [TOTAL] * 4)       # PP0's fill
    mine = [(s, g) for s, st, g in claims if st == 0]
    if mine:
        assert arena.complete_slots([s for s, _ in mine], [g for _, g in mine], [(0, TOTAL)]) == [1] * len(mine)
        assert arena.ref_slots([s for s, _ in mine], +1) == len(mine)   # the walk's reference: the tree names them
    arena.free_slots([c[0] for c in cands], reason="evict_to_disk")      # A finishes (no check)
    return mine, stems, arena


def test_death_0109_a_referenced_landed_page_keeps_its_key(tmp_path):
    mine, stems, arena = _death_0109(tmp_path, lambda a: a.reap_stale())
    assert mine == []        # the age gate: no slot of a LIVE evictor was free for PP0's fill
    # the room comes when A has freed its candidates: PP0 claims, lands, references
    claims = arena.claim_slots(stems, [TOTAL] * 4)
    assert [c[1] for c in claims] == [0] * 4
    assert arena.complete_slots([c[0] for c in claims], [c[2] for c in claims], [(0, TOTAL)]) == [1] * 4
    assert arena.ref_slots([c[0] for c in claims], +1) == 4
    for (slot, _st, _g), stem in zip(claims, stems):
        assert arena.find_slots([stem]) == [(slot, 2)]      # the load-back's chain proof holds


def test_mutant_death_0109_old_reap_unlinks_a_referenced_page(tmp_path):
    """MUTANT: the instant reap -- the chain proof of the load-back fails
    ('key slot -1 (state 0)') for every page of PP0 in A's slots."""
    mine, stems, arena = _death_0109(tmp_path, _old_reap)
    assert len(mine) == 4
    assert [arena.find_slots([st])[0] for st in stems] == [(-1, 0)] * 4
