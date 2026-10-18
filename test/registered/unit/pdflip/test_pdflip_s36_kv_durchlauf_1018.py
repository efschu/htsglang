"""S36 (18.10.) Teil D: L2 voll -> L3 lesen, IMMER (KV-DURCHLAUF) + L2->L3 schreibt nie still + #810 Buchung.

User 18.10.: "wenn im L2 zu wenig platz ist, warum wird dann nicht L3 genommen?" / "wozu ein puffer? die daten
fliessen doch schon" / "das sollte schon immer so sein, das von L2 auf L3 geschrieben wird ohne zu blockieren".

Ground (fLLiper rc9 restart, D TP0 10:09:06-10:10:14Z, 68 s without a D round): arena ``slots=6485 complete=6485
pinned=6485``, no page with an L3 copy (``ARENA-DROP ... written:0 freed=0``), the read in ``L3-FILL ROOM-WAIT``,
the write-behind in ``LOAD-YIELD`` behind that read.

Hermetic, REAL objects: the C arena (ShmArena on a temp file), the real ``HiCacheFile`` fill and store files
written by the real pageio writer, the real ``HiCacheController._arena_page_get``, the real ``ArenaMHAHostPool``
(paged, P=4) resolve / load / free, the real write-behind pass. No CUDA: the device pool is CPU tensors.

* repro (red before S36): arena FULL (every slot COMPLETE and referenced, no L3 copy), a store read of pages
  that ARE on disk comes on a prefetch io thread -> no ROOM-WAIT, every page delivered, as disk ids;
* the load puts exactly the L3 bytes into the device rows, per layer, K and V, whole pages and owner lanes,
  byte-identical to the arena path for the same page;
* group-uniform: three ranks on one shared arena deliver the same count (mutant: durchlauf on one rank only);
* the disk pages go back with the rows (free / release queue / HostPoolGroup), and the load chain proof
  accepts a disk page only for its own key;
* the QSA index sidecar of a disk page: held at the read, loaded from the held file at the load;
* SOLL 2: the write-behind writes during legs, sleep, dormancy and pending loads (mutant: the old gate),
  reads overtake per chunk (bounded), the bytes go O_DIRECT where aligned;
* #810: under the arena host pool the parse-time ledger books the staging range, not --hicache-size.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.managers.cache_controller import HiCacheController  # noqa: E402
from flliper.srt.mem_cache import hicache_storage as HS  # noqa: E402
from flliper.srt.mem_cache.canonical_page_store import CanonicalExtentWindow  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D, P = 2, 2, 4, 4          # layers, heads, head_dim, tokens per page (uint8 cells: cell = 8 B)
CELL = H * D
BLOCK = P * CELL                 # one layer's K (or V) of one page
TOTAL = 2 * L * BLOCK            # 128 B canonical page: K [L x BLOCK] | V [L x BLOCK]
SLOTS = 4
S = 8                            # staging rows (a multiple of P)
SFX = "_NF_s36"
ROOM_MS = 1500                   # the old ROOM-WAIT bound in the repro (default 5000)


def _kd():
    from flliper.srt.pdflip import kv_durchlauf as kd

    return kd


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from flliper.srt.mem_cache import l3_write_behind as gate

    gate._reset_for_tests()
    try:
        _kd()._reset_for_tests()
    except ImportError:          # the pre-S36 tree (the repro's red run)
        pass
    monkeypatch.setenv("FLLIPER_PDFLIP_L3FILL_ROOM_WAIT_MS", str(ROOM_MS))
    yield
    gate._reset_for_tests()
    try:
        _kd()._reset_for_tests()
    except ImportError:
        pass


class _KVPage:
    pass


def _backend(tmp_path, arena):
    from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    root = tmp_path / "l3"
    root.mkdir(exist_ok=True)
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
    be._l3idx = None
    be._l3idx_tried = True
    be._evictor = LRUFileEvictor(
        str(root), SFX, tp_rank=0, writes_shared_keys=False,
        path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
    )
    be._arenas = {TOTAL: arena}
    be._arena_for = lambda total_bytes: arena
    return be


def _pool(arena, owner=None):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    p.bind(arena, CanonicalExtentWindow(TOTAL, ((0, TOTAL),)), role="kv", pin=False)
    if owner is not None:
        p._owner_tok = owner
    return p


def _page_bytes(i: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + i)
    return torch.randint(0, 255, (TOTAL,), dtype=torch.uint8, generator=g)


def _hashes(tag, n):
    return [f"{tag}{i:02d}" + "cd" * 30 for i in range(n)]


def _put_l3(be, h, data):
    from flliper.srt.mem_cache.storage.file.pageio import load

    stem = be._get_suffixed_key(h)
    path = be._sharded_path(stem)
    be._ensure_shard_dir(path)
    assert load().write_pages([path], [TOTAL], [((0, TOTAL),)], [data.data_ptr()], False) == [0]
    return stem


def _fill_arena_full(be, arena):
    """Every slot COMPLETE and referenced (a live tree holds it), no L3 copy: nothing can be given back."""
    for j in range(SLOTS):
        h = f"full{j:02d}" + "ef" * 30
        stem = be._get_suffixed_key(h)
        pay = torch.full((TOTAL,), 0xA0 + j, dtype=torch.uint8)
        assert arena.write([stem], [TOTAL], [((0, TOTAL),)], [pay.data_ptr()]) == [1]
        slot = arena.find_slots([stem])[0][0]
        assert arena.ref_slots([slot], +1) == 1
    st = arena.stats()
    assert int(st["complete"]) == SLOTS


class _Op:
    def __init__(self, rid="s36-1"):
        self.request_id = rid
        self.completed_tokens = 0
        self.probe_pins = None

    def increment(self, n):
        self.completed_tokens += n
        return True


def _controller(pool, be):
    c = object.__new__(HiCacheController)
    c.mem_pool_host = pool
    c.storage_backend = be
    c.page_size = P
    return c


def _read_on_prefetch_thread(c, op, hashes, host):
    """The store read runs where the room wait would run: a prefetch io thread."""
    out = {}

    def run():
        t0 = time.perf_counter()
        out["n"] = c._arena_page_get(op, hashes, host)
        out["s"] = time.perf_counter() - t0

    t = threading.Thread(target=run, name=HS.PREFETCH_IO_THREAD_PREFIX + "s36")
    t.start()
    t.join(30)
    assert not t.is_alive()
    return out


def _setup(tmp_path, n=6, name="kv"):
    arena = ShmArena(str(tmp_path / f"{name}.bin"), TOTAL, SLOTS)
    be = _backend(tmp_path, arena)
    _fill_arena_full(be, arena)
    hs = _hashes("h", n)
    data = [_page_bytes(i) for i in range(n)]
    for h, d in zip(hs, data):
        _put_l3(be, h, d)
    return arena, be, hs, data


def _dev_pool(rows=64):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros((rows, H, D), dtype=torch.uint8) for _ in range(L)],
        v_buffer=[torch.zeros((rows, H, D), dtype=torch.uint8) for _ in range(L)],
    )


def _expect_cells(data, page, lane, layer, kv):
    off = (0 if kv == "k" else L * BLOCK) + layer * BLOCK + lane * CELL
    return data[page][off:off + CELL]


# ---------------------------------------------------------------------------------------------------------------
def test_repro_full_arena_a_read_runs_through_without_room_wait(tmp_path):
    """RED before S36: the read sat in L3-FILL ROOM-WAIT for the whole bound and delivered nothing (the prefix
    ended at the first page the full arena could not take). GREEN: every on-disk page delivered at once."""
    arena, be, hs, data = _setup(tmp_path)
    p = _pool(arena)
    c = _controller(p, be)
    host = p.alloc_read(len(hs) * P)
    r = _read_on_prefetch_thread(c, _Op(), hs, host)
    assert r["n"] == len(hs), f"delivered {r['n']} of {len(hs)} pages"
    assert r["s"] < ROOM_MS / 1000.0 / 3, f"the read waited {r['s']:.2f}s (ROOM-WAIT bound {ROOM_MS} ms)"
    kd = _kd()
    assert bool(kd.disk_mask(host).all()), "every page is a disk page (the arena had no slot)"
    assert kd.REGISTRY.live() == len(hs)
    assert int(arena.stats()["complete"]) == SLOTS and int(arena.stats()["claimed"]) == 0, \
        "the durchlauf takes no arena slot and leaves no claim"


def test_the_load_puts_the_l3_bytes_into_the_device_rows(tmp_path):
    arena, be, hs, data = _setup(tmp_path)
    p = _pool(arena)
    c = _controller(p, be)
    host = p.alloc_read(len(hs) * P)
    assert _read_on_prefetch_thread(c, _Op(), hs, host)["n"] == len(hs)
    dev = _dev_pool()
    dst = torch.arange(40, 40 + len(hs) * P, dtype=torch.int64)
    for layer in range(L):
        p.load_to_device_per_layer(dev, host, dst, layer, "direct")
    for i in range(len(hs)):
        for t in range(P):
            row = 40 + i * P + t
            for layer in range(L):
                assert torch.equal(dev.k_buffer[layer][row].reshape(-1), _expect_cells(data, i, t, layer, "k"))
                assert torch.equal(dev.v_buffer[layer][row].reshape(-1), _expect_cells(data, i, t, layer, "v"))
    assert int(dev.k_buffer[0][0].sum()) == 0, "rows outside the load stay untouched"


def test_disk_load_is_byte_identical_to_the_arena_load_and_takes_any_row_order(tmp_path):
    """Same page, two paths: arena slot vs held L3 file -- identical device bytes. Rows sorted by the merge
    (move_indices), duplicated, and a partial page (owner lanes) included."""
    arena, be, hs, data = _setup(tmp_path, n=3)
    p = _pool(arena)
    c = _controller(p, be)
    host = p.alloc_read(3 * P)
    assert _read_on_prefetch_thread(c, _Op(), hs, host)["n"] == 3
    # the same pages through a fresh arena (the L2 path)
    arena2 = ShmArena(str(tmp_path / "kv2.bin"), TOTAL, SLOTS)
    p2 = _pool(arena2)
    slots = []
    for h, d in zip(hs, data):
        stem = be._get_suffixed_key(h)
        assert arena2.write([stem], [TOTAL], [((0, TOTAL),)], [d.data_ptr()]) == [1]
        slots.append(arena2.find_slots([stem])[0][0])
    host2 = p2.alloc_read(3 * P)
    p2.resolve_rows(host2, slots)
    p2._pinned[:] = True          # no CUDA here: the registration the guard asks for is a no-op
    pick = torch.tensor([0, 1, 2, 3, 5, 6, 9, 9, 11, 4], dtype=torch.int64)   # lanes, a duplicate, page 1 partial
    dst = torch.arange(10, 10 + int(pick.numel()), dtype=torch.int64)
    a, b = _dev_pool(), _dev_pool()
    os.environ["FLLIPER_PDFLIP_ARENA_PAGE_LOAD"] = "0"    # the arena's per-layer gather (any row set)
    try:
        for layer in range(L):
            p.load_to_device_per_layer(a, host[pick], dst, layer, "direct")
            p2.load_to_device_per_layer(b, host2[pick], dst, layer, "direct")
    finally:
        os.environ.pop("FLLIPER_PDFLIP_ARENA_PAGE_LOAD", None)
    for layer in range(L):
        assert torch.equal(a.k_buffer[layer], b.k_buffer[layer])
        assert torch.equal(a.v_buffer[layer], b.v_buffer[layer])
    assert int(a.k_buffer[0][10:20].sum()) > 0


def test_mixed_read_arena_pages_and_disk_pages_in_one_prefix(tmp_path):
    """Pages already COMPLETE in the arena keep their slot; the rest run through -- one prefix, one load."""
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, SLOTS)
    be = _backend(tmp_path, arena)
    hs = _hashes("m", 5)
    data = [_page_bytes(50 + i) for i in range(5)]
    for h, d in zip(hs, data):
        _put_l3(be, h, d)
    # page 0 and 1 in the arena (COMPLETE, referenced by a tree), the other two slots full of foreign pages
    for h, d in zip(hs[:2], data[:2]):
        stem = be._get_suffixed_key(h)
        assert arena.write([stem], [TOTAL], [((0, TOTAL),)], [d.data_ptr()]) == [1]
        assert arena.ref_slots([arena.find_slots([stem])[0][0]], +1) == 1
    for j in range(2):
        stem = be._get_suffixed_key(f"x{j}" + "ab" * 30)
        pay = torch.full((TOTAL,), 7, dtype=torch.uint8)
        assert arena.write([stem], [TOTAL], [((0, TOTAL),)], [pay.data_ptr()]) == [1]
        assert arena.ref_slots([arena.find_slots([stem])[0][0]], +1) == 1
    p = _pool(arena)
    p._pinned[:] = True          # no CUDA here: the registration the arena guard asks for is a no-op
    host = p.alloc_read(5 * P)
    r = _read_on_prefetch_thread(_controller(p, be), _Op(), hs, host)
    assert r["n"] == 5 and r["s"] < ROOM_MS / 3000.0
    kd = _kd()
    m = kd.disk_mask(host).view(5, P)[:, 0].tolist()
    assert m == [False, False, True, True, True]
    dev = _dev_pool()
    dst = torch.arange(0, 5 * P, dtype=torch.int64)
    for layer in range(L):
        p.load_to_device_per_layer(dev, host, dst, layer, "direct")
    for i in range(5):
        for t in range(P):
            assert torch.equal(dev.k_buffer[1][i * P + t].reshape(-1), _expect_cells(data, i, t, 1, "k"))
            assert torch.equal(dev.v_buffer[0][i * P + t].reshape(-1), _expect_cells(data, i, t, 0, "v"))


def test_group_uniform_every_rank_delivers_the_same_count(tmp_path, monkeypatch):
    """Three ranks, one shared arena (the ranks' processes map the same shm file), the same read: every rank
    delivers every on-disk page, none waits. Mutant direction: the durchlauf on one rank only (the others wait
    out the room bound and deliver short -> the group MIN cuts everyone)."""
    from flliper.srt import rank_role

    arena, be, hs, data = _setup(tmp_path)
    got = []
    for rank in range(3):
        monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda r=rank: r != 0)
        a_r = ShmArena(str(tmp_path / "kv.bin"), TOTAL, SLOTS)   # attach: the same file, this rank's mapping
        be_r = _backend(tmp_path, a_r)
        p = _pool(a_r)
        host = p.alloc_read(len(hs) * P)
        r = _read_on_prefetch_thread(_controller(p, be_r), _Op(f"s36-r{rank}"), hs, host)
        got.append((r["n"], r["s"]))
    counts = [n for n, _ in got]
    assert counts == [len(hs)] * 3, f"per-rank delivered pages {counts}"
    assert max(s for _, s in got) < ROOM_MS / 3000.0, f"a rank waited: {[round(s, 2) for _, s in got]}"
    assert min(counts) == len(hs), "the check_prefetch_progress MIN keeps the whole read"


def test_a_file_gone_since_the_stat_ends_the_prefix_without_waiting(tmp_path):
    arena, be, hs, data = _setup(tmp_path)
    kd = _kd()
    real = kd.REGISTRY._open
    gone = be._sharded_path(be._get_suffixed_key(hs[3]))

    def _open(path):
        if path == gone:
            raise FileNotFoundError(2, "gone")
        return real(path)

    kd.REGISTRY._open = _open
    p = _pool(arena)
    host = p.alloc_read(len(hs) * P)
    r = _read_on_prefetch_thread(_controller(p, be), _Op(), hs, host)
    assert r["n"] == 3 and r["s"] < ROOM_MS / 3000.0
    assert kd.REGISTRY.live() == 3, "the disk pages past the cut were released"


def test_disk_pages_go_back_with_their_rows(tmp_path):
    from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup

    arena, be, hs, data = _setup(tmp_path)
    p = _pool(arena)
    host = p.alloc_read(len(hs) * P)
    assert _read_on_prefetch_thread(_controller(p, be), _Op(), hs, host)["n"] == len(hs)
    kd = _kd()
    assert kd.REGISTRY.live() == 6
    p.free(host[: 2 * P])                       # evict_host of a node: two pages
    assert kd.REGISTRY.live() == 4
    p.release_queued_rows(host[2 * P: 3 * P])  # the release queue (a span the tree did not adopt)
    assert kd.REGISTRY.live() == 3
    g = object.__new__(HostPoolGroup)
    g.anchor_entry = types.SimpleNamespace(host_pool=p)
    g.free(host[3 * P:])                       # the group's free never drops them as #718 strays
    assert kd.REGISTRY.live() == 0
    assert getattr(p, "_stale_index_refusals", 0) == 0
    assert int(arena.stats()["complete"]) == SLOTS


def test_a_released_disk_page_is_never_loaded(tmp_path):
    arena, be, hs, data = _setup(tmp_path, n=2)
    p = _pool(arena)
    host = p.alloc_read(2 * P)
    assert _read_on_prefetch_thread(_controller(p, be), _Op(), hs, host)["n"] == 2
    p.free(host)
    with pytest.raises(RuntimeError, match="KV-DURCHLAUF LOAD REFUSED"):
        p.load_to_device_per_layer(_dev_pool(), host, torch.arange(2 * P), 0, "direct")


def test_the_chain_proof_accepts_a_disk_page_only_for_its_own_key(tmp_path):
    from flliper.srt.mem_cache.pool_host import arena_pool as ap

    arena, be, hs, data = _setup(tmp_path, n=3)
    p = _pool(arena)
    host = p.alloc_read(3 * P)
    assert _read_on_prefetch_thread(_controller(p, be), _Op(), hs, host)["n"] == 3
    node = types.SimpleNamespace(hash_value=list(hs), host_value=host.clone(), id=1)
    stems_of = lambda keys: [be._get_suffixed_key(k) for k in keys]  # noqa: E731
    rows_of = lambda n: n.host_value  # noqa: E731
    assert ap.provable_pages(p, [node], rows_of, stems_of) == (3, 3)
    assert torch.equal(ap.verify_load_chain(p, [node], node.host_value.clone(), rows_of, stems_of), host)
    # a node whose second key is not the held file's: not proven there (no COMPLETE slot either)
    bad = types.SimpleNamespace(hash_value=[hs[0], hs[2], hs[1]], host_value=host.clone(), id=2)
    assert ap.provable_pages(p, [bad], rows_of, stems_of)[0] == 1
    with pytest.raises(Exception, match="CHAIN MISMATCH"):
        ap.verify_load_chain(p, [bad], bad.host_value.clone(), rows_of, stems_of)


def test_the_qsa_index_of_a_disk_page_is_held_and_loaded_from_l3(tmp_path, monkeypatch):
    from flliper.srt import rank_role
    from flliper.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from flliper.srt.mem_cache.qsa_pool_host import QSAPagedHostPool
    from flliper.srt.mem_cache.storage.file.pageio import load

    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    arena, be, hs, data = _setup(tmp_path, n=3)
    Q = 2 * 16                                    # 2 layers x 16 B index block per KV page
    be.canonical_qsa_page = CanonicalExtentWindow(Q, ((0, Q),), label="qsa")
    qdata = []
    for i, h in enumerate(hs):
        qs = be._get_suffixed_key(be._log_key(PoolName.QSA_INDEXER, h))
        path = be._sharded_path(qs)
        be._ensure_shard_dir(path)
        qd = torch.randint(0, 255, (Q,), dtype=torch.uint8, generator=torch.Generator().manual_seed(7 + i))
        assert load().write_pages([path], [Q], [((0, Q),)], [qd.data_ptr()], False) == [0]
        qdata.append(qd)
    p = _pool(arena)
    host = p.alloc_read(3 * P)
    assert _read_on_prefetch_thread(_controller(p, be), _Op(), hs, host)["n"] == 3
    q = object.__new__(QSAPagedHostPool)
    q.pool_name = "qsa_indexer"; q.layer_num = 2; q.item_bytes = 16; q.slot_page_size = P; q.page_size = P
    q.dtype = torch.uint8; q.num_host_pages = 4; q.layout = "layer_first"
    q.kv_buffer = [torch.zeros((4, 16), dtype=torch.uint8) for _ in range(2)]
    q.device_buffers = [torch.zeros((16, 16), dtype=torch.uint8) for _ in range(2)]
    be.registered_pools = {PoolName.QSA_INDEXER: q}
    res = be.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, host_indices=host, keys=list(hs))])
    assert res[PoolName.QSA_INDEXER] == [True, True, True]
    assert all(int(b.sum()) == 0 for b in q.kv_buffer), "no mirror row is written for a disk page"
    kd = _kd()
    assert kd.REGISTRY.stats["qsa_fd"] == 3
    dst = torch.arange(4 * P, 7 * P, dtype=torch.int64)           # device pages 4..6
    for layer in range(2):
        q.load_to_device_per_layer(None, host, dst, layer, "direct")
    for i in range(3):
        for layer in range(2):
            assert torch.equal(q.device_buffers[layer][4 + i], qdata[i][layer * 16:(layer + 1) * 16])
    # a disk page whose QSA index is nowhere: a QSA miss at the read (the read's QSA hit pages end there)
    os.unlink(be._sharded_path(be._get_suffixed_key(be._log_key(PoolName.QSA_INDEXER, hs[0]))))
    kd.REGISTRY.entry(int(kd.pages_of_rows(host[:1], P)[0][0])).qsa_mode = None
    res = be.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, host_indices=host[:P], keys=[hs[0]])])
    assert res[PoolName.QSA_INDEXER] == [False]


# ---- SOLL 2: L2 -> L3 never holds still -------------------------------------------------------------------------
def _wb_backend(tmp_path, n=6):
    arena = ShmArena(str(tmp_path / "wb.bin"), TOTAL, 16)
    be = _backend(tmp_path, arena)
    be._l3p_persistent_dir = lambda: str(tmp_path / "l3")
    for i in range(n):
        stem = be._get_suffixed_key(f"w{i:02d}" + "aa" * 30)
        pay = _page_bytes(200 + i)
        assert arena.write([stem], [TOTAL], [((0, TOTAL),)], [pay.data_ptr()]) == [1]
    return be, arena


def test_the_write_behind_writes_during_legs_sleep_dormancy_and_loads(tmp_path):
    """rc9 restart, D TP0: LOAD-YIELD 430 s in 25 min, the tree pages written:0. Now every gate is open: a leg
    runs, the group sleeps, the scheduler is dormant AND a load is pending -- the pass writes every page."""
    from flliper.srt.mem_cache import l3_write_behind as gate

    be, arena = _wb_backend(tmp_path)
    gate.leg_enter("release_memory_occupation", types.SimpleNamespace(pdflip_dormant=True))
    gate.register_load_probe(lambda: 3)
    assert gate.former_quiet_reason() == "leg", "the old gate would have held still here"
    tot = be.l3_write_behind_pass(slice_s=0.0)
    assert tot["paused"] is None and tot["written"] == 6, tot
    gate.leg_exit("release_memory_occupation", True)
    assert gate.former_quiet_reason() == "asleep"
    assert gate.quiet_reason() is None
    on = be._stat_stems([be._get_suffixed_key(f"w{i:02d}" + "aa" * 30) for i in range(6)])
    assert len(on) == 6, "every arena page has its L3 copy -- a full arena stays discardable"


def test_reads_overtake_writes_per_chunk_and_writes_still_move(tmp_path, monkeypatch):
    from flliper.srt.mem_cache import l3_write_behind as gate

    monkeypatch.setenv("FLLIPER_HICACHE_L3_WB_READ_YIELD_MS", "200")
    gate.read_enter()
    ended = []

    def finish():
        time.sleep(0.05)
        ended.append(time.perf_counter())
        gate.read_exit()

    t = threading.Thread(target=finish)
    t0 = time.perf_counter()
    t.start()
    waited = gate.write_turn()
    t.join()
    assert 0.03 <= waited < 0.19, f"the chunk waited {waited:.3f}s for the read in flight"
    assert ended and ended[0] - t0 <= waited + 0.02
    # a read that never ends: the chunk is written after the bound anyway (writes never stall)
    gate.read_enter()
    try:
        w2 = gate.write_turn(0.05)
    finally:
        gate.read_exit()
    assert 0.04 <= w2 < 0.15 and gate.read_priority_stats()["forced"] >= 1


def test_the_pass_takes_a_read_turn_before_every_small_chunk(tmp_path, monkeypatch):
    from flliper.srt.mem_cache import l3_write_behind as gate

    monkeypatch.setenv("FLLIPER_PDFLIP_L3_WRITE_BEHIND_CHUNK_PAGES", "2")
    be, arena = _wb_backend(tmp_path)
    turns = []
    real = gate.write_turn
    monkeypatch.setattr(gate, "write_turn", lambda *a, **k: (turns.append(1), real(*a, **k))[1])
    tot = be.l3_write_behind_pass(slice_s=0.0)
    assert tot["written"] == 6 and len(turns) == 3, (tot, turns)


def test_the_write_behind_bytes_go_o_direct_where_aligned(tmp_path):
    from flliper.srt.mem_cache.storage.file.pageio import load

    pio = load()
    buf = np.zeros(3 * 4096, dtype=np.uint8)
    off = (-buf.ctypes.data) % 4096
    pay = buf[off:off + 4096]
    pay[:] = np.arange(4096) % 251
    dc: list = []
    st = pio.write_pages([str(tmp_path / "a")], [4096], [((0, 4096),)], [pay.ctypes.data], False,
                         direct=True, direct_counts=dc)
    assert st == [0]
    assert (tmp_path / "a").read_bytes() == pay.tobytes()
    assert dc[0] in (0, 1)        # 1 on XFS/ZFS-with-O_DIRECT, 0 on a filesystem that refuses it (tmpfs)
    dc2: list = []
    st = pio.write_pages([str(tmp_path / "b")], [100], [((0, 100),)], [pay.ctypes.data + 1], False,
                         direct=True, direct_counts=dc2)
    assert st == [0] and dc2 == [0], "an unaligned extent goes buffered, same bytes"
    assert (tmp_path / "b").read_bytes() == pay[1:101].tobytes()


# ---- #810 ledger -------------------------------------------------------------------------------------------------
def test_810_books_the_staging_range_under_the_arena_host_pool(monkeypatch, caplog):
    import logging

    from flliper.srt import server_args as sa

    posts = []

    def fake_err(p, total, avail):
        posts.extend(p)
        return None

    monkeypatch.setattr("flliper.srt.mem_cache.pinned_host_budget.joint_pinned_host_error", fake_err)
    ns = types.SimpleNamespace(hicache_size=4, tp_size=3, pp_size=1, kv_session_offload_host_ram_gib=0)
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_HOST", "1")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_STAGING_GB", "0.05")
    with caplog.at_level(logging.INFO):
        sa.ServerArgs._post_hicache_staging_host_ledger(ns)
    assert len(posts) == 1 and posts[0].nbytes == int(0.05e9) * 3, posts
    assert "is not allocated (#1430)" in caplog.text
    posts.clear()
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_HOST", "0")
    sa.ServerArgs._post_hicache_staging_host_ledger(ns)
    assert posts[0].nbytes == int(4e9) * 3, "without the arena host pool --hicache-size is the tier"
