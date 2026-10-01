"""ARENA-COPY-GEN (N2, 01.10.): an L2 -> L3 copy is bound to the slot it copies.

Needle-MISS analysis (27B boot 10010740, image 32fc2683b1): the persisted
needle entry was born in the arena at 07:43:29 and copied to disk at 07:44:11.
The log shows the slot untouched in that window (mamba arena complete=1, no
claim, no free, no LOST), so the copy did not corrupt THAT entry -- but the
code had three holes through which a copy CAN publish another page's bytes
under the old stem:

(1) EVICT PATH x reap_stale: ``arena_evict_candidates`` sets the candidates
    EVICTING and the evicting process copies them to disk
    (``arena_secure_to_disk``) before freeing them. ``arena_reap_stale`` --
    called by EVERY process after its own eviction -- freed every EVICTING
    slot with refcount 0 at once ("its evictor died"), including the
    candidates another live process was still copying. The next claim takes
    the slot, writes its page, and the copy renames the new bytes into the
    OLD stem's file.
(2) WRITE-BEHIND x free + reclaim: the write-behind pins by KEY only, and a
    free of the slot (``arena_free_slots`` ignores pins) plus a re-claim moves
    the generation under the copy; the published file then holds the new
    page. No check after the copy.
(3) UNPIN x re-claim: the write-behind's unpin was a raw ``refcount - 1`` --
    after a re-claim it took a reference of the NEW generation's reader.

Hermetic: the real C arena on a temp file, the real HiCacheFile copy paths
(``arena_secure_to_disk``, ``l3_write_behind_pass``) and the real pageio
writer; the race is injected at the one seam between "decided to copy" and
"bytes on disk"."""
from __future__ import annotations

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.canonical_page_store import CanonicalExtentWindow  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file import pageio as _pageio  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 64
SFX = "_27B_a32fcecf07"


class _KVPage:
    pass


def _backend(tmp_path, monkeypatch, slots=1):
    from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    be = object.__new__(HiCacheFile)
    be.file_path = str(root)
    be._known_shards = set()
    be._legacy_flat = False
    be._key_geom = {"is_mla_model": False}
    be.metadata_cache = None
    be.dcp_owner_mode = False
    be.canonical_kv_page = _KVPage()
    be._canonical_kv_extents = CanonicalExtentWindow(TOTAL, ((0, TOTAL),))
    be.canonical_qsa_page = None
    be.canonical_mamba_blob = None
    be.canonical_draft_page = None
    be.kv_config_suffix = SFX
    be._kv_config_suffix_is_group_wide = True
    be.config_suffix = SFX + "_0_1"
    be._config_suffix_is_group_wide = False
    be._canonical_probe_mismatch = lambda: None
    be._l3idx = None          # no #1459 stem index: the disk answers (a new boot's
    be._l3idx_tried = True    # index is seeded from disk; here the stat is the seed)
    be._evictor = LRUFileEvictor(
        str(root), SFX, tp_rank=0, writes_shared_keys=False,
        path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
    )
    arena_dir = tmp_path / "shm"
    arena_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(arena_dir))
    arena = ShmArena(str(arena_dir / f"arena-{TOTAL}.bin"), TOTAL, slots)
    be._arenas = {TOTAL: arena}
    return be, arena


def _stem(be, tag):
    return be._get_suffixed_key(tag + "ab" * 30)


def _put(arena, stem, fill):
    pay = torch.full((TOTAL,), fill, dtype=torch.uint8)
    return arena.write([stem], [TOTAL], [((0, TOTAL),)], [pay.data_ptr()])


def _on_disk(be, stem):
    p = be._existing_path(stem)
    if not p or not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return f.read()


class _RacingPio:
    """The real pageio writer, with ``before`` run once between the copy
    decision and the bytes -- the window of the race."""

    def __init__(self, before):
        self._real = _pageio.load()
        self._before = before
        self.fired = 0

    def write_pages(self, *a, **kw):
        if not self.fired:
            self.fired += 1
            self._before()
        return self._real.write_pages(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _patch_pio(monkeypatch, racing):
    monkeypatch.setattr(_pageio, "load", lambda: racing)


# --------------------------------------------------------------------------- (1)
def test_reap_stale_leaves_a_live_evictors_candidates_alone(tmp_path, monkeypatch):
    """RED on 0fba4f83e7: another process's reap_stale frees the candidate
    while it is being secured; the new page's bytes land in the OLD stem's
    file. GREEN: a young EVICTING slot is not stale, the re-claim finds no
    room (status 4) and the old stem's file holds the old bytes."""
    be, arena = _backend(tmp_path, monkeypatch)
    sa, sb = _stem(be, "aa"), _stem(be, "bb")
    assert _put(arena, sa, 0x11) == [1]
    cands = arena.evict_candidates(1)
    assert [c[0] for c in cands] == [0]

    def other_process():
        arena.reap_stale()           # a peer's eviction epilogue
        _put(arena, sb, 0x22)        # a peer's next page claims what is free

    _patch_pio(monkeypatch, _RacingPio(other_process))
    be.arena_secure_to_disk(arena, cands)
    got = _on_disk(be, sa)
    assert got is not None, "the old page must reach L3"
    assert got == bytes([0x11]) * TOTAL, f"old stem holds foreign bytes {got[:4]!r}"


def test_reap_stale_still_reaps_a_dead_evictors_slots(tmp_path, monkeypatch):
    """The reaper keeps its job: an EVICTING slot nobody freed for longer
    than the stale age goes back to FREE."""
    from sglang.srt.mem_cache.storage.file import hicache_arena as _ha

    be, arena = _backend(tmp_path, monkeypatch)
    assert _put(arena, _stem(be, "aa"), 0x11) == [1]
    assert len(arena.evict_candidates(1)) == 1
    assert arena.reap_stale(min_age_s=0.0) == 1
    assert _put(arena, _stem(be, "bb"), 0x22) == [1]
    assert _ha.REAP_STALE_MIN_AGE_S >= 10.0


def test_secure_to_disk_unpublishes_a_copy_whose_slot_moved(tmp_path, monkeypatch):
    """Belt behind (1): whatever frees and re-claims an EVICTING candidate
    during its copy, the copy is checked after the write and an invalid one
    is unpublished (a clean miss) -- never the foreign bytes."""
    be, arena = _backend(tmp_path, monkeypatch)
    sa, sb = _stem(be, "aa"), _stem(be, "bb")
    assert _put(arena, sa, 0x11) == [1]
    cands = arena.evict_candidates(1)

    def foreign_free():
        arena.free_slots([0], reason="test_foreign")
        assert _put(arena, sb, 0x22) == [1]

    _patch_pio(monkeypatch, _RacingPio(foreign_free))
    out = be.arena_secure_to_disk(arena, cands)
    assert _on_disk(be, sa) is None
    assert out.get("torn") == 1
    assert out["written"] == 0


# --------------------------------------------------------------------------- (2)(3)
def _wb(be):
    return be.l3_write_behind_pass(budget_bytes=1 << 20, quiet=lambda: None, slice_s=0.0)


def test_write_behind_never_publishes_a_recycled_slot(tmp_path, monkeypatch):
    """RED on 0fba4f83e7: free + re-claim during the write-behind copy -> the
    OLD stem's file holds the NEW page. GREEN: the copy is checked against
    the census generation after the write and unpublished; the page is not
    marked secured, so a later pass (if it is still in L2) writes it."""
    be, arena = _backend(tmp_path, monkeypatch)
    sa, sb = _stem(be, "aa"), _stem(be, "bb")
    assert _put(arena, sa, 0x11) == [1]

    def recycle():
        arena.free_slots([0], reason="test_recycle")
        assert _put(arena, sb, 0x22) == [1]

    _patch_pio(monkeypatch, _RacingPio(recycle))
    tot = _wb(be)
    got = _on_disk(be, sa)
    assert got != bytes([0x22]) * TOTAL, "old stem published with the recycled slot's bytes"
    assert got is None
    assert tot.get("torn") == 1
    assert tot["written"] == 0


def test_write_behind_unpin_never_takes_the_next_generations_reference(tmp_path, monkeypatch):
    """RED on 0fba4f83e7: the raw unpin after a re-claim drops the NEW
    page's reader reference (refcount 1 -> 0, evictable while read)."""
    be, arena = _backend(tmp_path, monkeypatch)
    sa, sb = _stem(be, "aa"), _stem(be, "bb")
    assert _put(arena, sa, 0x11) == [1]

    def recycle_and_read():
        arena.free_slots([0], reason="test_recycle")
        assert _put(arena, sb, 0x22) == [1]
        assert arena.ref_slots([0], +1) == 1     # a reader of the new page

    _patch_pio(monkeypatch, _RacingPio(recycle_and_read))
    _wb(be)
    assert arena.slot_refs([0]) == [1]


def test_write_behind_plain_copy_unchanged(tmp_path, monkeypatch):
    """No race: the page is written once, secured, and the pin is returned."""
    be, arena = _backend(tmp_path, monkeypatch)
    sa = _stem(be, "aa")
    assert _put(arena, sa, 0x11) == [1]
    tot = _wb(be)
    assert tot["written"] == 1 and tot.get("torn", 0) == 0
    assert _on_disk(be, sa) == bytes([0x11]) * TOTAL
    assert arena.slot_refs([0]) == [0]
    assert _wb(be)["written"] == 0              # secured: not written again


# --------------------------------------------------------------------------- L3-CSUM
BLOB = 1 << 20


def _blob_backend(tmp_path, monkeypatch, slots=2):
    be, _ = _backend(tmp_path, monkeypatch, slots=1)
    arena = ShmArena(str(tmp_path / "shm" / f"arena-{BLOB}.bin"), BLOB, slots)
    be._arenas = {BLOB: arena}
    return be, arena


def _put_blob(arena, stem, seed):
    g = torch.Generator().manual_seed(seed)
    pay = torch.randint(0, 256, (BLOB,), dtype=torch.uint8, generator=g)
    assert arena.write([stem], [BLOB], [((0, BLOB),)], [pay.data_ptr()]) == [1]
    return bytes(pay.numpy())


def test_blob_copy_writes_a_crc_sidecar_and_the_fill_verifies_it(tmp_path, monkeypatch):
    from sglang.srt.mem_cache import hicache_storage as hs

    be, arena = _blob_backend(tmp_path, monkeypatch)
    sa = be._get_suffixed_key("cc" + "ab" * 30 + ".mamba")
    want = _put_blob(arena, sa, 7)
    assert be.l3_write_behind_pass(budget_bytes=1 << 24, quiet=lambda: None, slice_s=0.0)["written"] == 1
    path = be._existing_path(sa)
    side = hs.l3_csum_read(path)
    import zlib
    assert side == (zlib.crc32(want) & 0xFFFFFFFF, BLOB)
    # a fresh boot's arena: the verified blob fills
    be2, arena2 = _blob_backend(tmp_path / "b2", monkeypatch)
    be2.file_path = be.file_path
    out = be2.arena_fill_from_disk(arena2, [sa], BLOB)
    assert out[0] is not None
    assert bytes(arena2.slot_view(out[0], BLOB)) == want


def test_a_blob_whose_bytes_changed_on_disk_is_a_named_miss_and_moved_aside(tmp_path, monkeypatch):
    from sglang.srt.mem_cache import hicache_storage as hs

    be, arena = _blob_backend(tmp_path, monkeypatch)
    sa = be._get_suffixed_key("dd" + "ab" * 30 + ".mamba")
    _put_blob(arena, sa, 9)
    be.l3_write_behind_pass(budget_bytes=1 << 24, quiet=lambda: None, slice_s=0.0)
    path = be._existing_path(sa)
    with open(path, "r+b") as f:          # one flipped byte in the persisted blob
        f.seek(12345)
        b = f.read(1)
        f.seek(12345)
        f.write(bytes([b[0] ^ 0x40]))
    be2, arena2 = _blob_backend(tmp_path / "b2", monkeypatch)
    be2.file_path = be.file_path
    out = be2.arena_fill_from_disk(arena2, [sa], BLOB)
    assert out[0] is None                                  # a MISS, never the bytes
    assert not os.path.exists(path)
    assert os.path.exists(path + hs.L3_CSUM_BAD_SUFFIX)    # kept, never deleted
    assert arena2.find_slots([sa])[0][0] == -1             # the claim went back


def test_a_blob_without_a_sidecar_is_read_as_before(tmp_path, monkeypatch):
    from sglang.srt.mem_cache import hicache_storage as hs

    be, arena = _blob_backend(tmp_path, monkeypatch)
    sa = be._get_suffixed_key("ee" + "ab" * 30 + ".mamba")
    want = _put_blob(arena, sa, 11)
    be.l3_write_behind_pass(budget_bytes=1 << 24, quiet=lambda: None, slice_s=0.0)
    path = be._existing_path(sa)
    os.unlink(path + hs.L3_CSUM_SUFFIX)                   # a pre-N2 blob
    be2, arena2 = _blob_backend(tmp_path / "b2", monkeypatch)
    be2.file_path = be.file_path
    out = be2.arena_fill_from_disk(arena2, [sa], BLOB)
    assert out[0] is not None and bytes(arena2.slot_view(out[0], BLOB)) == want
