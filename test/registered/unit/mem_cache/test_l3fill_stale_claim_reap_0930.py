"""L3FILL-JOINED (3) (30.09., NF y4a ep36): 29 stems stayed JOINED for >= 6 s
on D and on P -- a claim whose writer is alive but delivered no byte is never
reaped by #231 (only claims with NO open writer are), and every fill of the
stem is a miss for good. Now such a claim is QUARANTINED from its key after
FLLIPER_PDFLIP_L3FILL_STALE_CLAIM_S (default 5 s) without byte progress and the
fill reads the stem from disk into a fresh slot. Generation-safe: the old
writer's late completion is refused by name (status 6) and can never land in
the stem's new slot nor in the old slot's next owner."""

import logging
import os
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.environ import envs
from flliper.srt.mem_cache import hicache_storage as hs
from flliper.srt.mem_cache.hicache_storage import HiCacheFile
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _setup(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    for s in ("p0", "p1", "p2"):
        (root / (s + ".bin")).write_bytes(bytes([0x40 + int(s[1])]) * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    [(old, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # alive, never delivers a byte
    assert status == 0
    return _backend(str(root)), arena, old, gen


def test_a_stale_live_claim_is_quarantined_and_the_stem_read_from_disk(tmp_path, caplog):
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    with envs.FLLIPER_PDFLIP_L3FILL_STALE_CLAIM_S.override(0.05), \
            caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
    new = out[1]
    assert new is not None and new != old            # base: None -- the prefix ended at p1
    assert out[2] is not None                        # base: p2 never read
    assert arena.find_slots(["p1"]) == [(new, 2)]
    assert bytes(arena.slot_view(new, TOTAL)[:4]) == b"\x41" * 4
    [(_, g, _age, _pid, _role, opened, state)] = arena.claim_info([old])
    assert (g, opened, state) == (gen, 1, 4)         # quarantined, generation kept, writer still open
    assert any("L3-FILL STALE-CLAIM-REAP" in r.getMessage() for r in caplog.records)


def test_the_old_writers_late_completion_is_refused_by_generation(tmp_path):
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    with envs.FLLIPER_PDFLIP_L3FILL_STALE_CLAIM_S.override(0.05):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True)
    new = out[0]
    # the old writer comes back and writes its (other) bytes, then completes
    arena.slot_view(old, TOTAL)[:4] = b"\xee" * 4
    assert arena.complete_slots([old], [gen], [(0, TOTAL)]) == [6]    # named: reaped under it
    assert arena.find_slots(["p1"]) == [(new, 2)]
    assert bytes(arena.slot_view(new, TOTAL)[:4]) == b"\x41" * 4      # the new home is untouched
    # the last one out freed the quarantined slot; its generation moved on
    [(_, g2, _age, _pid, _role, opened, state)] = arena.claim_info([old])
    assert state == 0 and g2 == gen + 1 and opened == 0
    # a second late call of the old writer can never hit the slot's next owner
    [(nxt, st, ngen)] = arena.claim_slots(["other"], [TOTAL])
    if nxt == old:
        assert arena.complete_slots([old], [gen], [(0, TOTAL)]) == [3]
        assert arena.free_if_gen([old], [gen]) == [0]
        assert arena.claim_info([old])[0][6] == 1               # still the new owner's claim


def test_a_young_claim_is_not_reaped(tmp_path):
    be, arena, old, gen = _setup(tmp_path)
    with envs.FLLIPER_PDFLIP_L3FILL_STALE_CLAIM_S.override(30.0):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
    assert out[1] is None and out[2] is None
    assert arena.claim_info([old])[0][6] == 1                   # still CLAIMED by its writer


def test_a_join_does_not_count_as_progress(tmp_path):
    """Every reader's JOIN refreshed touched_ms; the stale clock is byte progress."""
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    for _ in range(3):   # readers joining over and over (y4a: 5 cycles x 3 ranks + P)
        arena.claim_slots(["p1"], [TOTAL])
        arena.unclaim([old], [gen])
    assert arena.quarantine_stale([old], [gen], 50) == [1]


def test_the_sweep_frees_a_quarantine_whose_writer_never_returns(tmp_path):
    _be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    assert arena.quarantine_stale([old], [gen], 50) == [1]
    assert arena.quarantine_sweep(10_000) == 0                 # writer open, backstop not reached
    time.sleep(0.06)
    assert arena.quarantine_sweep(50) == 1
    assert arena.claim_info([old])[0][6] == 0
    assert arena.stats()["claimed"] == 0
