"""y6o (01.10.): the Form-A worker loadback copies only the lanes it owns.

MEASURED (boot y6o, NF P->D, 21:36-21:59Z): after the D wake TP1/TP2 load
their owner rows with ``_owner_page_prefix_load`` -> ``_load_pages_all_layers``
(mode=dma) as WHOLE pages: 3601 pages = 2.83 GB per worker, although TP1 owns
40 and TP2 24 of the 64 lanes of a page (uneven DCP-KV). TP1 on the x4 link:
~440 ms of DMA that TP0 waits for in the first collectives after the wake.

The fix (``arena_lane_dma``): one 2D copy per run of consecutive slots --
src pitch = owner split x cell, width = owned run x cell -- into a compact
stage, then the per-layer lane scatter. These tests pin

* the bytes: the marker's ``bytes=`` is ``pages x 2L x m x cell`` (RED on
  42147ea269: ``pages x page_bytes``), and the y6o arithmetic;
* the plan touches EXACTLY the owned cells of each page (no more, no less);
* byte equality of the loaded device KV against the whole-page path
  (switch off), on bf16 buffers, multi-run and single-run owner splits, many
  blocks, out-of-order device rows;
* the named fallbacks: a refused copy mid-load, a lane set without the 2D form;
* the cudaMemcpy2DAsync argument order (fake runtime) and that the desk
  resolves torch's own libcudart.

Hermetic: a tmp arena file, no CUDA.
"""

import hashlib
import logging
import os
import re
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from flliper.srt.mem_cache import pool_host as _ph  # noqa: F401
from flliper.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from flliper.srt.mem_cache.canonical_page_store import (
    CanonicalPageWindow,
    owner_row_window,
    owner_token_runs,
)
from flliper.srt.mem_cache.pool_host import arena_lane_dma as ld
from flliper.srt.mem_cache.pool_host import arena_pool as ap
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool, owner_page_tokens
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 3, 2, 4          # attention layers, kv heads, head dim
DT = torch.bfloat16
CELL = H * D * DT.itemsize  # 16 B
P = 16                     # tokens per page
BLOCK = P * CELL
PAGE = 2 * L * BLOCK
STAGING = 5
NSLOTS = 40
SPEC = CanonicalPageSpec(num_attn_layers=L, kv_bytes_per_token_per_attn_layer=2 * BLOCK)
WHOLE = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=L)

# the y6o shape on the desk page: S = P, uneven 10/16 + 6/16; and a multi-run split
TP1 = (P, P, 0, 10)
TP2 = (P, P, 10, 16)
SPLIT4 = (P, 4, 1, 3)      # lanes 1,2,5,6,9,10,13,14


def _pool(arena, owner):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = DT; p.device = "cpu"; p.pin_memory = False; p.size = STAGING
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(STAGING, dtype=torch.int64)
    p.slot_used = torch.zeros(STAGING, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, STAGING, H, D, dtype=DT)
    p._arena_init_fields()
    p.bind(arena, owner_row_window(WHOLE, P, owner_token_runs(*owner)), role="kv", pin=False,
           owner_rows=owner)
    p._all_pinned = True   # production: pre-pinned at bind (#1436) -> "dma"
    return p


def _fill_random(pool, seed=7):
    g = torch.Generator().manual_seed(seed)
    pool._page_view.copy_(torch.randint(0, 256, tuple(pool._page_view.shape), generator=g,
                                        dtype=torch.int64).to(torch.uint8))


def _dev(rows):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(rows, H, D, dtype=DT) for _ in range(L)],
        v_buffer=[torch.zeros(rows, H, D, dtype=DT) for _ in range(L)])


def _owner_rows(pool, slots):
    return [STAGING + s * P + t for s in slots for t in pool._owner_tok.tolist()]


def _load(pool, host, dev_rows):
    dev = _dev(max(dev_rows) + 1)
    hi, di = torch.tensor(host, dtype=torch.int64), torch.tensor(dev_rows, dtype=torch.int64)
    gathered = []
    orig = ArenaMHAHostPool._transfer_paged
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ArenaMHAHostPool, "_transfer_paged",
                   lambda self, *a: gathered.append(1) or orig(self, *a))
        for l in range(L):
            pool.load_to_device_per_layer(dev, hi, di, l, "direct")
    assert gathered == []      # whole owner groups never take the per-layer gather
    return dev


def _digest(dev):
    h = hashlib.sha256()
    for bufs in (dev.k_buffer, dev.v_buffer):
        for t in bufs:
            h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _expect_from_arena(pool, dev, host, dev_rows):
    """Every device row holds exactly its arena cell, every other row is 0."""
    pv = pool._page_view
    hit = set(dev_rows)
    for l in range(L):
        for kv, bufs, offs in ((0, dev.k_buffer, pool._k_offs_b), (1, dev.v_buffer, pool._v_offs_b)):
            got = bufs[l].view(torch.uint8).reshape(bufs[l].shape[0], -1)
            for h, r in zip(host, dev_rows):
                s, t = divmod(h - STAGING, P)
                o = offs[l] + t * CELL
                assert torch.equal(got[r], pv[s, o:o + CELL]), (h, r, l, kv)
            for r in range(got.shape[0]):
                if r not in hit:
                    assert int(got[r].abs().sum()) == 0, (r, l, kv)


def _marker_bytes(caplog):
    lines = [r.getMessage() for r in caplog.records if "PDFLIP-ARENA-PAGE-LOAD" in r.getMessage()]
    assert lines, "no PDFLIP-ARENA-PAGE-LOAD marker"
    return [int(re.search(r" bytes=(\d+)", x).group(1)) for x in lines], lines


@pytest.fixture
def arena(tmp_path):
    return ShmArena(str(tmp_path / "kv.bin"), PAGE, NSLOTS)


@pytest.fixture(autouse=True)
def _dma(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ARENA_PAGE_LOAD_MODE", "dma")
    monkeypatch.delenv("FLLIPER_PDFLIP_ARENA_OWNER_LANE_DMA", raising=False)
    monkeypatch.setattr(ap, "_PAGE_LOAD_N", 0)


# ----------------------------------------------------------------- the bytes
@pytest.mark.parametrize("owner,m", [(TP1, 10), (TP2, 6), (SPLIT4, 8)])
def test_the_loadback_moves_only_the_owned_lanes(arena, caplog, owner, m):
    """RED on 42147ea269: bytes = pages x PAGE (whole pages)."""
    w = _pool(arena, owner)
    _fill_random(w)
    slots = [3, 4, 5, 6, 9, 10, 20]
    host = _owner_rows(w, slots)
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        _load(w, host, list(range(len(host))))
    got, lines = _marker_bytes(caplog)
    assert got == [len(slots) * 2 * L * m * CELL], lines
    assert "lane_dma=2d" in lines[0] and f"owner_lanes={m}/{P}" in lines[0]
    assert f"whole_bytes={len(slots) * PAGE}" in lines[0]


def test_y6o_arithmetic():
    """NF: P=64, 12 attention layers, 512-B cell, owner split 64; y6o loaded
    3601 pages per worker."""
    pb = 2 * 12 * 64 * 512
    assert pb == 786432
    whole_stage = max(16, ((256 << 20) // pb) // 4)   # the worker's whole-page stage (85 pages)
    for (lo, hi), want, blocks in (((0, 40), 1769963520, 27), ((40, 64), 1061978112, 16)):
        lanes = owner_page_tokens((64, 64, lo, hi))
        g = ld.owner_lane_geometry(lanes=lanes, page_tokens=64, cell=512,
                                   k_offs_b=[l * 64 * 512 for l in range(12)],
                                   v_offs_b=[(12 + l) * 64 * 512 for l in range(12)], page_bytes=pb)
        assert (g.width, g.spitch, g.rows_per_page) == ((hi - lo) * 512, 64 * 512, 24)
        assert 3601 * g.compact_page_bytes == want          # vs 3601 x 786432 = 2831831040
        sp = ld.lane_stage_pages(whole_stage_pages=whole_stage, geom=g)
        assert sp * g.compact_page_bytes <= whole_stage * pb  # no stage byte above the old one
        assert -(-3601 // sp) == blocks                      # vs 43 whole-page blocks


@pytest.mark.parametrize("owner", [TP1, TP2, SPLIT4, (P, 8, 0, 8), (P, 2, 1, 2), (P, P, 0, P)])
def test_the_plan_touches_exactly_the_owned_cells(owner):
    lanes = owner_page_tokens(owner).tolist()
    k_offs = [l * BLOCK for l in range(L)]
    v_offs = [(L + l) * BLOCK for l in range(L)]
    g = ld.owner_lane_geometry(lanes=lanes, page_tokens=P, cell=CELL, k_offs_b=k_offs,
                               v_offs_b=v_offs, page_bytes=PAGE)
    assert g is not None
    slots = [2, 3, 4, 7, 8, 12]
    runs = ap.page_dma_runs(slots, 4)          # registration pieces of 4 pages split runs too
    plan = ld.owner_lane_dma_plan(geom=g, runs=runs)
    assert len(plan) == len(runs)
    touched = torch.zeros(NSLOTS * PAGE, dtype=torch.int32)
    stage_rows = torch.full((len(slots) * g.compact_page_bytes,), -1, dtype=torch.int64)
    for dst_off, src_off, height in plan:
        for r in range(height):
            a = src_off + r * g.spitch
            touched[a:a + g.width] += 1
            d = dst_off + r * g.width
            stage_rows[d:d + g.width] = torch.arange(a, a + g.width)
    want = torch.zeros(NSLOTS * PAGE, dtype=torch.int32)
    for s in slots:
        for o in k_offs + v_offs:
            for t in lanes:
                a = s * PAGE + o + t * CELL
                want[a:a + CELL] = 1
    assert torch.equal(touched, want)          # every owned byte once, nothing else
    assert bool((stage_rows >= 0).all())       # the compact stage is filled densely
    # the compact page is (block, lane) ordered: the scatter's view
    for i, s in enumerate(slots):
        for j, o in enumerate(k_offs + v_offs):
            for q, t in enumerate(lanes):
                d = i * g.compact_page_bytes + j * len(lanes) * CELL + q * CELL
                assert int(stage_rows[d]) == s * PAGE + o + t * CELL


def test_lanes_without_the_row_form_have_no_geometry():
    kw = dict(page_tokens=P, cell=CELL, k_offs_b=[l * BLOCK for l in range(L)],
              v_offs_b=[(L + l) * BLOCK for l in range(L)], page_bytes=PAGE)
    assert ld.owner_lane_geometry(lanes=[0, 1, 3], **kw) is None          # unequal runs
    assert ld.owner_lane_geometry(lanes=[0, 4, 12], **kw) is None         # unequal spacing
    assert ld.owner_lane_geometry(lanes=[], **kw) is None
    bad = dict(kw, v_offs_b=[(L + l) * BLOCK + CELL for l in range(L)])   # blocks do not tile
    assert ld.owner_lane_geometry(lanes=[0, 1], **bad) is None


# ------------------------------------------------------- byte equality vs old
@pytest.mark.parametrize("owner", [TP1, TP2, SPLIT4])
@pytest.mark.parametrize("stage_pages", [None, 1, 3])
def test_loaded_kv_is_byte_equal_to_the_whole_page_path(tmp_path, monkeypatch, owner, stage_pages):
    slots = [0, 1, 2, 3, 11, 12, 13, 30, 31, 39]
    digests = []
    for lane_dma in ("1", "0"):
        arena = ShmArena(str(tmp_path / f"kv{lane_dma}.bin"), PAGE, NSLOTS)
        w = _pool(arena, owner)
        _fill_random(w, seed=11)
        host = _owner_rows(w, slots)
        g = torch.Generator().manual_seed(3)
        dev_rows = (torch.randperm(len(host) + 9, generator=g)[:len(host)] + 2).tolist()
        monkeypatch.setenv("FLLIPER_PDFLIP_ARENA_OWNER_LANE_DMA", lane_dma)
        if stage_pages is not None:
            monkeypatch.setattr(ap, "lane_stage_pages", lambda **k: stage_pages)
        dev = _load(w, host, dev_rows)
        _expect_from_arena(w, dev, host, dev_rows)
        digests.append(_digest(dev))
    assert digests[0] == digests[1]


def test_switch_off_restores_the_whole_page_bytes(arena, caplog, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ARENA_OWNER_LANE_DMA", "0")
    w = _pool(arena, TP1)
    _fill_random(w)
    slots = [5, 6, 7]
    host = _owner_rows(w, slots)
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        dev = _load(w, host, list(range(len(host))))
    got, _ = _marker_bytes(caplog)
    assert got == [len(slots) * PAGE]
    _expect_from_arena(w, dev, host, list(range(len(host))))


def test_unregistered_slots_keep_the_cpu_stage(arena, caplog):
    """The owner path forces "cpu" for unregistered slots -- no 2D copy from
    pageable memory."""
    w = _pool(arena, TP2)
    w._all_pinned = False
    w._pinned[:] = False
    w._pin = False
    _fill_random(w)
    host = _owner_rows(w, [1, 2])
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        dev = _load(w, host, list(range(len(host))))
    _, lines = _marker_bytes(caplog)
    assert "mode=cpu" in lines[0] and "lane_dma" not in lines[0]
    _expect_from_arena(w, dev, host, list(range(len(host))))


# ------------------------------------------------------------- the fallbacks
def test_a_refused_copy_falls_back_to_whole_pages_for_the_rest(arena, monkeypatch, caplog):
    w = _pool(arena, TP1)
    _fill_random(w)
    slots = [0, 1, 2, 3, 4, 5, 6]
    host = _owner_rows(w, slots)
    dev_rows = list(range(len(host)))[::-1]
    monkeypatch.setattr(ap, "lane_stage_pages", lambda **k: 2)
    calls = []
    orig = ld.copy_plan

    def _refuse_third(**kw):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("cudaMemcpy2DAsync rc=1 invalid argument")
        return orig(**kw)

    monkeypatch.setattr(ld, "copy_plan", _refuse_third)
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        dev = _load(w, host, dev_rows)
    _expect_from_arena(w, dev, host, dev_rows)
    assert w._lane_dma_off is True
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PDFLIP-ARENA-LANE-DMA failed after 4 of 7 pages" in x for x in msgs), msgs
    got, _ = _marker_bytes(caplog)
    assert got == [3 * PAGE]                   # the rest, as whole pages
    # the next load stays on whole pages
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        _load(w, _owner_rows(w, [9]), list(range(10)))
    got, _ = _marker_bytes(caplog)
    assert got == [PAGE]


def test_a_lane_set_without_the_row_form_loads_whole_pages(arena, caplog):
    w = _pool(arena, TP1)
    w._owner_tok = torch.tensor([0, 1, 3], dtype=torch.int64)
    _fill_random(w)
    slots = [4, 8]
    host = [STAGING + s * P + t for s in slots for t in (0, 1, 3)]
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        dev = _load(w, host, list(range(len(host))))
    got, _ = _marker_bytes(caplog)
    assert got == [len(slots) * PAGE]
    assert any("PDFLIP-ARENA-LANE-DMA off" in r.getMessage() for r in caplog.records)
    _expect_from_arena(w, dev, host, list(range(len(host))))


# --------------------------------------------------------------- the runtime
class _FakeCudart:
    def __init__(self, rc=0):
        self.rc, self.calls, self.cleared = rc, [], 0

    def cudaMemcpy2DAsync(self, dst, dpitch, src, spitch, width, height, kind, stream):
        self.calls.append((dst.value, dpitch, src.value, spitch, width, height, kind, stream.value))
        return self.rc

    def cudaGetErrorString(self, rc):
        return b"invalid argument"

    def cudaGetLastError(self):
        self.cleared += 1
        return 0


def test_issue_2d_argument_order():
    g = ld.owner_lane_geometry(lanes=owner_page_tokens(TP2), page_tokens=P, cell=CELL,
                               k_offs_b=[l * BLOCK for l in range(L)],
                               v_offs_b=[(L + l) * BLOCK for l in range(L)], page_bytes=PAGE)
    plan = ld.owner_lane_dma_plan(geom=g, runs=[(0, 4, 2), (2, 9, 1)])
    lib = _FakeCudart()
    n = ld.issue_2d(lib=lib, path="fake", plan=plan, geom=g, src_base=1 << 40, dst_base=1 << 32,
                    stream=0x77)
    w = 6 * CELL
    assert lib.calls == [
        ((1 << 32) + 0, w, (1 << 40) + 4 * PAGE + 10 * CELL, P * CELL, w, 2 * 2 * L, 1, 0x77),
        ((1 << 32) + 2 * g.compact_page_bytes, w, (1 << 40) + 9 * PAGE + 10 * CELL, P * CELL, w,
         2 * L, 1, 0x77),
    ]
    assert n == 3 * 2 * L * w
    bad = _FakeCudart(rc=1)
    with pytest.raises(RuntimeError, match="cudaMemcpy2DAsync rc=1 invalid argument"):
        ld.issue_2d(lib=bad, path="fake", plan=plan, geom=g, src_base=0, dst_base=0, stream=0)
    assert bad.cleared == 1


@pytest.mark.skipif(not torch.version.cuda, reason="a CPU-only torch has no libcudart")
def test_the_runtime_is_torchs_own_libcudart():
    path = ld._torch_cudart_path()
    assert os.path.basename(path) == "libcudart.so." + torch.version.cuda.split(".")[0]
    lib, _ = ld._cudart()
    assert lib.cudaMemcpy2DAsync is not None
