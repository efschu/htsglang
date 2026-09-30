"""#257 caching (i), vision boot 0928 (P PP0 06:20:28, pdflip-4-27): the store
probe reported 813 of 830 pages and took no reference; the read, batch by
batch, found page 219 neither COMPLETE in the arena nor on disk and ended there
(completed=14016, 0.04 s into a 52.8 s budget) -- the prompt was prefilled
from 0. Between probe and read a claim on the full shared arena (5461 slots,
92-99 % complete) freed unreferenced COMPLETE slots WITHOUT a disk copy
(#1427 ``_evict_for_claim`` stage i; PP0's drop counter went 8 -> 58 with no
line).

(a) the probe holds what it reports until the read; (d) a claim never frees a
page that has no L3 copy -- it writes it first; (e) a hold never blocks a
claim and is bounded in time.

Hermetic: the real C arena on a temp file, the real ``_arena_page_get`` /
``_evict_for_claim`` / ``HiCacheFile.arena_fill_from_disk``; pages of 64 B,
page_size 1 (one token per page)."""

import os
import shutil
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers import cache_controller as _cc  # noqa: E402
from flliper.srt.managers.cache_controller import HiCacheController  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 2, 4
CELL = H * D
PAGE = 64
S = 5
REPORTED = 813          # the probe's answer (of 830 requested)
FIRST_READ = 219        # where the read ended on the metal


class _Win:
    total_bytes = PAGE
    extents = ((1 * CELL, L * CELL), (PAGE // 2 + 1 * CELL, L * CELL))


class _Backend:
    def __init__(self, arena):
        self._canonical_kv_extents = _Win()
        self.canonical_draft_page = _Win()
        self._a = arena

    def _arena_for(self, total_bytes):
        return self._a

    def _get_suffixed_key(self, key):
        return key + "_sfx"

    def _suffix_for_key(self, key):
        return ("_sfx",)


class _Op:
    request_id = "pdflip-4-27"

    def __init__(self):
        self.completed_tokens = 0
        self._t = False

    def increment(self, n):
        self.completed_tokens += n
        return True

    def is_terminated(self):
        return self._t


def _pool(tmp_path, slots):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, slots)
    p.bind(arena, _Win(), role="kv", pin=False)
    return p, arena


def _write(arena, stem, fill):
    pay = torch.full((PAGE,), fill & 0xFF, dtype=torch.uint8)
    assert arena.write([stem], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]


def _controller(p, be):
    c = object.__new__(HiCacheController)
    c.mem_pool_host = p; c.storage_backend = be; c.page_size = 1
    return c


def _probe(c, op, hashes):
    """What prefetch_thread_func does after the hit query: hold what it reports
    (absent before #257 -- then nothing is held)."""
    pin = getattr(_cc, "probe_hold_pin", None)
    return pin(c, op, hashes) if callable(pin) else 0


def _read(c, op, hashes, host):
    """The aux thread's batched read, until a short batch (_page_transfer)."""
    return c._arena_page_get(op, hashes, host)


def _release_rest(c, op):
    try:
        from flliper.srt.mem_cache import probe_hold
    except ImportError:
        return
    probe_hold.release(op, c.mem_pool_host, 0, reason="read-end")


def test_pdflip_4_27_a_claim_between_probe_and_read_no_longer_cuts_the_read(tmp_path):
    """RED on a9e3a842ae: after the first 219 pages are read, a claim under
    pressure frees the 594 unreferenced pages the probe had reported (no free
    slot, no copy); the read ends at 219. GREEN: the probe holds them, the
    claim finds no candidate (and returns at once -- it never waits), the read
    takes all 813."""
    # the arena is full: the request's pages first, other requests' behind
    p, arena = _pool(tmp_path, slots=2 * REPORTED)
    p.arena = None  # bound lazily through the backend, as on the metal
    be = _Backend(arena)
    hashes = [f"h{i}" for i in range(REPORTED)]
    for i, h in enumerate(hashes):
        _write(arena, h + "_sfx", i)
    for i in range(REPORTED):
        _write(arena, f"other{i}_sfx", i)
    c = _controller(p, be)
    op = _Op()
    host = p.alloc_read(REPORTED)
    _probe(c, op, hashes)
    assert _read(c, op, hashes[:FIRST_READ], host[:FIRST_READ]) == FIRST_READ
    # a foreign claim: the arena is full, it needs the rest of it
    t0 = time.monotonic()
    p._evict_for_claim(arena, REPORTED - FIRST_READ, claim_stem="foreign")
    assert time.monotonic() - t0 < 1.0, "a claim never waits on a hold"
    n = _read(c, op, hashes[FIRST_READ:], host[FIRST_READ:])
    _release_rest(c, op)
    assert FIRST_READ + n == REPORTED, f"the read ended at {FIRST_READ + n} of {REPORTED}"
    assert op.completed_tokens == REPORTED


def test_holds_above_the_group_min_and_after_the_read_go_back(tmp_path):
    """A hold is a reference like any read's: what the read did not take is
    released (group MIN trim, read end); the arena ends with only the read's
    own references."""
    from flliper.srt.mem_cache import probe_hold

    p, arena = _pool(tmp_path, slots=32)
    p.arena = None
    be = _Backend(arena)
    hashes = [f"h{i}" for i in range(12)]
    for i, h in enumerate(hashes):
        _write(arena, h + "_sfx", i)
    c = _controller(p, be)
    op = _Op()
    assert _cc.probe_hold_pin(c, op, hashes) == 12
    probe_hold.release(op, p, 8, reason="group-min")        # the group agreed on 8
    host = p.alloc_read(8)
    assert c._arena_page_get(op, hashes[:8], host) == 8
    probe_hold.release(op, p, 0, reason="read-end")
    held, refs, complete = arena.ref_census()
    assert (held, refs, complete) == (8, 8, 12), "the read's 8 references only"
    assert probe_hold.held_total() == 0


def test_a_stale_hold_is_released_by_name(tmp_path):
    from flliper.srt.mem_cache import probe_hold

    p, arena = _pool(tmp_path, slots=8)
    p.arena = None
    be = _Backend(arena)
    hashes = [f"h{i}" for i in range(4)]
    for i, h in enumerate(hashes):
        _write(arena, h + "_sfx", i)
    c = _controller(p, be)
    op = _Op()
    assert _cc.probe_hold_pin(c, op, hashes) == 4
    assert not probe_hold.expire_if_stale(op, p)
    assert probe_hold.expire_if_stale(op, p, now=op.probe_pins_t0 + probe_hold.PROBE_HOLD_MAX_S + 1)
    assert arena.ref_census()[1] == 0
    assert probe_hold.held_total() == 0


class _Evictor:
    def reserve(self, stem, size, key=None, owner_writes_whole_file=False):
        return True

    def commit(self, stem):
        pass

    def abort(self, stem):
        pass


def _file_backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._sharded_path = _path
    be._ensure_shard_dir = lambda path: None
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._evictor = _Evictor()
    be._key_geom = {"is_mla_model": False}
    be._arena_evict_to_disk = lambda arena, want: 0
    return be


def test_a_pressure_claim_loses_no_page_without_an_l3_copy(tmp_path):
    """(d) RED on a9e3a842ae: the claim-time room frees 5 COMPLETE pages that
    exist nowhere else -- a later read finds neither slot nor file. GREEN: each
    is written to L3 before its slot is freed (dropped_without_l3 = 0), and the
    read takes it back from disk with its bytes."""
    root = tmp_path / "store"; root.mkdir()
    p, arena = _pool(tmp_path, slots=8)
    be = _file_backend(str(root))
    p._backend = be
    stems = [f"k{i}" for i in range(8)]
    for i, st in enumerate(stems):
        _write(arena, st, 0x40 + i)
    before = getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0)
    freed = p._evict_for_claim(arena, 5, claim_stem="foreign")
    assert freed == 5
    gone = [st for st, (slot, state) in zip(stems, arena.find_slots(stems)) if slot < 0]
    assert len(gone) == 5
    on_disk = [st for st in gone if os.path.exists(os.path.join(str(root), st + ".bin"))]
    assert on_disk == gone, f"pages lost from L2 and L3 at once: {sorted(set(gone) - set(on_disk))}"
    assert getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0) == before
    back = HiCacheFile.arena_fill_from_disk(be, arena, gone, PAGE)
    assert all(s is not None for s in back)
    for st, slot in zip(gone, back):
        assert bytes(arena.slot_view(slot, PAGE)[:4]) == bytes([0x40 + stems.index(st)]) * 4


def test_a_page_already_on_disk_is_not_written_again(tmp_path):
    root = tmp_path / "store"; root.mkdir()
    p, arena = _pool(tmp_path, slots=4)
    be = _file_backend(str(root))
    p._backend = be
    stems = [f"k{i}" for i in range(4)]
    for i, st in enumerate(stems):
        _write(arena, st, i)
        (root / (st + ".bin")).write_bytes(bytes([0x7E]) * PAGE)  # an older L3 copy
    p._evict_for_claim(arena, 4, claim_stem="foreign")
    for st in stems:
        assert (root / (st + ".bin")).read_bytes() == bytes([0x7E]) * PAGE


def _l3p_backend(tmp_path, monkeypatch):
    """The L3P store as 9b2a060c2a builds it: a PERSISTENT per-identity
    directory (``L3_IDENTITY.json``), the real sharded layout, the real
    ``LRUFileEvictor`` as the one bookkeeper, and the real #1459 stem index,
    seeded from the directory (``_l3p_seed_index``) -- a stem the index does
    not name is NOT on disk."""
    from flliper.srt.mem_cache.storage.file.l3_index import L3Index
    from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PERSIST", "1")
    root = tmp_path / "store" / "nf-identity"
    root.mkdir(parents=True)
    (root / "L3_IDENTITY.json").write_text("{}")
    be = object.__new__(HiCacheFile)
    be.file_path = str(root)
    be._known_shards = set()
    be._legacy_flat = False
    be._key_geom = {"is_mla_model": False}
    be._evictor = LRUFileEvictor(
        str(root), "_sfx", tp_rank=0, writes_shared_keys=False,
        path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
    )
    idx = L3Index(str(tmp_path / "l3idx.bin"), cap=1 << 12)
    be._l3idx = idx
    be._evictor.l3_index = idx
    return be, idx


def _previous_boot_left(be, stem, fill):
    path = be._sharded_path(stem)
    be._ensure_shard_dir(path)
    with open(path, "wb") as f:
        f.write(bytes([fill]) * PAGE)


def test_l3p_a_pressure_claim_leaves_every_page_on_disk_and_in_the_index(tmp_path, monkeypatch):
    """(d) on the L3P path: an inherited page (seeded into the index) is not
    rewritten; every page the claim writes lands in the identity directory
    AND in the stem index, so the next probe (``_stat_stems``, index first)
    and the read (``arena_fill_from_disk``) find it. A page on disk whose
    stem the index does not know yet (the seed runs off-thread) is written
    again, never dropped."""
    be, idx = _l3p_backend(tmp_path, monkeypatch)
    _previous_boot_left(be, "k0", 0x7E)
    _previous_boot_left(be, "k1", 0x7E)
    assert be._l3p_seed_index(idx) == 2
    _previous_boot_left(be, "k2", 0x7D)  # on disk, not seeded yet
    p, arena = _pool(tmp_path, slots=8)
    p._backend = be
    stems = [f"k{i}" for i in range(8)]
    for i, st in enumerate(stems):
        _write(arena, st, 0x40 + i)
    before = getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0)
    assert p._evict_for_claim(arena, 6, claim_stem="foreign") == 6
    gone = [st for st, (slot, state) in zip(stems, arena.find_slots(stems)) if slot < 0]
    assert len(gone) == 6
    unindexed = [st for st, h in zip(gone, idx.has(gone)) if not h]
    assert not unindexed, f"pages on disk the next probe cannot see (not indexed): {unindexed}"
    assert sorted(be._stat_stems(gone)) == sorted(gone), "the next probe misses a page the claim moved"
    assert getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0) == before
    for st in ("k0", "k1"):
        if st in gone:
            with open(be._sharded_path(st), "rb") as f:
                assert f.read() == bytes([0x7E]) * PAGE, "an inherited page was rewritten"
    back = HiCacheFile.arena_fill_from_disk(be, arena, gone, PAGE)
    assert all(s is not None for s in back)
