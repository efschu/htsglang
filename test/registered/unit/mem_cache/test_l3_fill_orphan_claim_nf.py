"""NF: a page on disk behind an ORPHAN claim in the L2 arena is read, not recomputed.

Metal (NF y3w e033a931db):

* D 01:39:13 / 01:40:20: the whole TP group gave up a claim of ~1900 pages
  together -- TP1 ``#1427r CLAIM-RELEASE site=claim_refused fresh=753 freed=0
  kept=753`` (then ``kept=512``, ``kept_total=1265``), TP0/TP2 ``#1427
  ARENA-CLAIM REFUSED statuses=[1, 2, 4]`` (their joins unclaimed). The slots
  stayed CLAIMED with no open writer; the KV arena never reaps (#231's reap ran
  for the mamba arena only).
* P PP0 01:43:12, weg2-26-39: the probe held 1066 COMPLETE pages and counted
  page 612 from its L3 copy (``PROBE-HOLD pages=1067 held=1066``); the read's
  L3 -> L2 fill claimed page 612, JOINED the orphan (status 1), skipped it as
  "another writer is filling it" and ended there: ``READ-STAGES pages=612
  l3fill_pages=1 l3fill_ms=0`` -> 38528 tokens re-prefilled although on disk
  (``#1157 ... hit_pages=1067 ... completed_local=39168``).
* The same fill left its join open: the orphan could never be reaped
  afterwards, whatever the age.

Hermetic: the real C arena on a temp file, the real
``HiCacheFile.arena_fill_from_disk``; a 64-byte page on disk. RED on
a332187f28, GREEN with the fix.
"""

import os
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

PAGE = 64
STEM = "p612_sfx"


class _Evictor:
    def touch(self, *a, **k):
        pass

    def reserve(self, *a, **k):
        return True

    def commit(self, stem):
        pass

    def abort(self, stem):
        pass


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._sharded_path = _path
    be._ensure_shard_dir = lambda path: None
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._evictor = _Evictor()
    be._key_geom = {"is_mla_model": False}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _on_disk(root, stem, fill):
    with open(os.path.join(root, stem + ".bin"), "wb") as f:
        f.write(bytes([fill]) * PAGE)


def _orphan(arena, stem):
    """D's group refusal: TP1 claims fresh, TP0 joins, both give up."""
    (s0, st0, g0), = arena.claim_slots([stem], [PAGE])
    (s1, st1, g1), = arena.claim_slots([stem], [PAGE])
    assert (st0, st1) == (0, 1) and s0 == s1
    arena.release_claims([s0], [g0], reason="claim_refused")  # kept: joined by TP0
    arena.unclaim([s1], [g1])  # TP0's join given up
    return s0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PARTIAL_REAP_S", "0.05")
    root = tmp_path / "store"
    root.mkdir()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)
    return str(root), arena, _backend(str(root))


def test_weg2_26_39_a_page_behind_an_orphan_claim_is_read_from_l3(setup):
    """RED on a332187f28: [None] -- the fill joins the orphan and skips it,
    the read ends there. GREEN: the orphan is reaped, the page filled from
    disk and COMPLETE with its bytes."""
    root, arena, be = setup
    _on_disk(root, STEM, 0x5A)
    _orphan(arena, STEM)
    time.sleep(0.1)  # older than the reap age: no writer came back
    assert arena.find_states([STEM]) == [1]
    back = HiCacheFile.arena_fill_from_disk(be, arena, [STEM], PAGE, prefix=True)
    assert back[0] is not None
    assert arena.find_states([STEM]) == [2]
    assert bytes(arena.slot_view(back[0], PAGE)[:4]) == b"\x5a" * 4


def test_a_live_writers_claim_is_never_taken_and_the_fill_leaves_no_open_mark(setup):
    """A writer still filling the slot keeps it (a miss for this read). RED
    on a332187f28: the fill's join stays open, so after that writer gives up
    the orphan can never be reaped. GREEN: the join is unclaimed at once."""
    root, arena, be = setup
    _on_disk(root, STEM, 0x5A)
    (slot, st, gen), = arena.claim_slots([STEM], [PAGE])  # the live writer
    assert st == 0
    time.sleep(0.1)
    back = HiCacheFile.arena_fill_from_disk(be, arena, [STEM], PAGE, prefix=True)
    assert back == [None]
    assert arena.find_states([STEM]) == [1]  # still the writer's
    arena.release_claims([slot], [gen], reason="claim_refused")  # the writer gives up
    time.sleep(0.1)
    assert arena.reap_partial(0.05) == [slot], "the fill left an open writer on the slot"


def test_a_fresh_claim_is_unchanged(setup):
    root, arena, be = setup
    _on_disk(root, STEM, 0x21)
    back = HiCacheFile.arena_fill_from_disk(be, arena, [STEM], PAGE, prefix=True)
    assert back[0] is not None and arena.find_states([STEM]) == [2]
    assert bytes(arena.slot_view(back[0], PAGE)[:2]) == b"\x21\x21"
