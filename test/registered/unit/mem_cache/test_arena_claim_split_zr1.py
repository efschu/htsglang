"""ZR-1 (NF y6h D, pdflip-4-14): three pages of a 1091-page prefix stayed
CLAIMED for good -- ``#1439 ARENA-PRESENT keys=1091 leading_complete=1067
census=[(1, 3), (2, 1088)] claimed=6``, ``READ-TRACE asked=24 readable=21``
-- and the wake re-computed what D had published itself.

Under the #239 token cut a D page has three direct writers: TP0 (the attention
host, share 0: it claims with no extents), TP1 (its token rows) and TP2 (the
others). They publish the same node at the same moment (the idle publish is
group-synchronous). arena.c claim_slot published a key in two steps -- the key
by CAS, the slot id by a store after it -- and a claimer that lost the CAS
moved on to the NEXT index cell. A second claimer of the same key therefore
either read the key with the previous occupant's slot id (the 'stale cell'
branch took the cell over) or put the key into the next empty cell: the key
got TWO claimed slots, the writers of one page split between them, and
neither slot ever covered the page (claimed=6 = 3 pages x 2 slots).

Hermetic: the real C arena mapped by real forked processes (one per rank),
the same stems claimed at the same moment."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache.storage.file import hicache_arena  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

SLOT = 4096
HALF = SLOT // 2
ROUNDS = 8
KEYS = 2048


def _rank(path, nslots, stems, ext, bar, q):
    """One D rank: claim the node's pages, write its extents, merge them."""
    a = ShmArena(path, SLOT, nslots)
    tag = getattr(hicache_arena, "ensure_writer_tag", None)   # the base has no census
    if tag is not None:
        tag(a._lib, tag=8 + ext[0])
    bar.wait()
    got = a.claim_slots(stems, [SLOT] * len(stems))
    pend = [(s, g) for s, st, g in got if st in (0, 1)]
    if pend:
        a.complete_slots([s for s, _ in pend], [g for _, g in pend], ext[1])
    q.put(sum(1 for _, st, _ in got if st == 0))
    a.close()


def _publish_node(tmp_path, r, extents_by_rank):
    path = str(tmp_path / f"arena-{r}.bin")
    nslots = KEYS * 4
    owner = ShmArena(path, SLOT, nslots)
    stems = [f"zr1-r{r}-p{i}" for i in range(KEYS)]
    ctx = mp.get_context("fork")
    bar = ctx.Barrier(len(extents_by_rank))
    q = ctx.Queue()
    procs = [ctx.Process(target=_rank, args=(path, nslots, stems, ext, bar, q))
             for ext in extents_by_rank]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    fresh = sum(q.get() for _ in procs)
    return owner, stems, fresh


def test_three_d_ranks_publishing_one_node_leave_every_page_complete(tmp_path):
    """Form A token cut: TP0 merges nothing, TP1 and TP2 one half each. Every
    page of every round must be COMPLETE in ONE slot -- none CLAIMED, none
    held twice."""
    ranks = [(0, []), (1, [(0, HALF)]), (2, [(HALF, HALF)])]
    for r in range(ROUNDS):
        arena, stems, fresh = _publish_node(tmp_path, r, ranks)
        st = arena.stats()
        states = [s for _, s in arena.find_slots(stems)]
        assert fresh == KEYS, f"round {r}: {fresh - KEYS} extra fresh claims (a key claimed twice)"
        assert st["claimed"] == 0, f"round {r}: {st['claimed']} slots left CLAIMED"
        assert st["complete"] == KEYS and states == [2] * KEYS
        arena.close()


def test_concurrent_claims_give_each_key_exactly_one_slot(tmp_path):
    """The claim alone (no completion): six processes claim the same keys at
    once -- every key has one claimed slot, and every claimer got that one."""
    ranks = [(t, []) for t in range(6)]
    for r in range(ROUNDS // 2):
        arena, stems, fresh = _publish_node(tmp_path, 100 + r, ranks)
        assert arena.stats()["claimed"] == KEYS
        assert fresh == KEYS
        rows = arena.claim_census([s for s, _ in arena.find_slots(stems[:64])])
        assert all(row["same_key_slots"] == 1 for row in rows)
        assert all(row["claims"] == 6 for row in rows)
        arena.close()


@pytest.fixture
def arena(tmp_path):
    a = ShmArena(str(tmp_path / "arena-4096.bin"), SLOT, 16)
    yield a
    a.close()


def test_census_names_who_claimed_joined_and_merged(arena):
    lib = arena._lib
    hicache_arena.ensure_writer_tag(lib, tag=8)          # D TP0: claims, no extents
    (s, st, g), = arena.claim_slots(["page"], [SLOT])
    assert st == 0
    hicache_arena.ensure_writer_tag(lib, tag=9)          # D TP1 joins and merges its half
    (s1, st1, g1), = arena.claim_slots(["page"], [SLOT])
    assert (s1, st1) == (s, 1)
    assert arena.complete_slots([s], [g], [(0, HALF)]) == [0]
    hicache_arena.ensure_writer_tag(lib, tag=10)         # D TP2 joins, never merges
    arena.claim_slots(["page"], [SLOT])
    hicache_arena.ensure_writer_tag(lib, tag=8)
    assert arena.complete_slots([s], [g], []) == [0]     # TP0 resolves with no bytes
    row, = arena.claim_census([s])
    assert row["state"] == 1 and row["gen"] == g
    assert hicache_arena.writer_tag_text(row["owner_tag"]) == "D0"
    assert hicache_arena.writer_tags_text(row["joined"]) == "D1+D2"
    assert hicache_arena.writer_tags_text(row["merged"]) == "D0+D1"
    assert (row["claims"], row["open"]) == (3, 1)        # TP2 is still open
    assert (row["covered"], row["total"]) == (HALF, SLOT)
    assert row["same_key_slots"] == 1
    text = hicache_arena.claim_open_text("page_stem", row)
    assert "claimed_by=D0 joined=D1+D2 merged=D0+D1 open=1 claims=3 covered=2048/4096" in text


def test_a_dead_holder_of_the_index_lock_is_named_and_taken_over(arena, tmp_path):
    """The guard: a process that died inside the index critical section leaves
    its pid in the lock word -- the next claim takes it over (counted), it
    never spins for good."""
    import ctypes
    import subprocess

    p = subprocess.Popen(["sleep", "0"])
    p.wait()                                             # a pid that no longer exists
    lock = ctypes.c_uint32.from_address(arena._base.value + 96)
    lock.value = p.pid
    assert arena.index_lock_steals() == 0
    (s, st, _), = arena.claim_slots(["after-death"], [SLOT])
    assert st == 0 and s >= 0
    assert arena.index_lock_steals() == 1
    assert lock.value == 0


def test_short_read_names_the_open_claims(tmp_path, caplog):
    """READ-TRACE short -> one ARENA-CLAIM-OPEN line per CLAIMED page it could
    not read, with the census of its writers."""
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    a = ShmArena(str(tmp_path / "arena-4096.bin"), SLOT, 16)
    hicache_arena.ensure_writer_tag(a._lib, tag=9)
    (s, _, g), = a.claim_slots(["k3_sfx"], [SLOT])
    a.complete_slots([s], [g], [(0, HALF)])

    class _Store:
        CLAIM_OPEN_SCAN = HiCacheFile.CLAIM_OPEN_SCAN
        CLAIM_OPEN_LINES = HiCacheFile.CLAIM_OPEN_LINES
        _log_claim_open = HiCacheFile._log_claim_open

        def _arena_dir(self):
            return str(tmp_path)

        def _canonical_total_for_stem(self, stem):
            return SLOT

        def _arena_for(self, total):
            return a

    with caplog.at_level(logging.INFO):
        _Store()._log_claim_open(7, ["k1_sfx", "k3_sfx"])
    lines = [r.getMessage() for r in caplog.records if "ARENA-CLAIM-OPEN" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].startswith("ARENA-CLAIM-OPEN n=7 slot=%d stem=k3_sfx gen=%d claimed_by=D1" % (s, g))
    assert "merged=D1" in lines[0] and "covered=2048/4096" in lines[0]
    a.close()
