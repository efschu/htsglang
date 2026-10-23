"""#179 (23.10.2026): the arena write's run stage is capped at one extend chunk.

Boot 10231552 (27B int4ar, D = TP2 on 2x RTX 3080), 16:41:26Z, TP0:
'PDFLIP-ARENA-WRITE run mode failed (OutOfMemoryError ... Tried to allocate
882.00 MiB ... 765.75 MiB is free) ... cell mode from now on'. The stage was one
(pages, 2 * run) tensor for the whole turn-end write (~28.2k pages x 32 KiB).

The repro drives the REAL ``ArenaMHAHostPool._backup_arena`` on a CPU-only desk:
a card-memory dummy refuses every ``torch.empty`` above the free bytes the boot
had (scaled to the fixture), the hicache kernels are emulated byte-exactly on
host pointers. Old behaviour (one fill) -> refused -> cell; new behaviour
(chunked fills) -> run mode, bytes identical, and an OOM no longer turns the run
mode off for the rest of the boot."""
from __future__ import annotations

import ctypes
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from flliper.srt.pdflip import arena_write as aw  # noqa: E402

MIB = 1 << 20

# --- the boot's own numbers ---------------------------------------------------

BOOT_RUN = 16384            # runs=2x16384B on the D ranks (16 layers x 1 KiB cell, page = 1 token)
BOOT_ASK_MIB = 882.0        # 'Tried to allocate 882.00 MiB' (TP0)
BOOT_FREE_MIB = 765.75      # '765.75 MiB is free' (TP0)
BOOT_CHUNK_TOKENS = 4096    # chunked_prefill_size of the D group (launcher argv)


def test_boot_numbers_one_fill_refused_chunked_fits():
    pages = int(BOOT_ASK_MIB * MIB) // (2 * BOOT_RUN)          # 28224 pages in the failing call
    assert aw.run_stage_bytes(pages, BOOT_RUN) / MIB == pytest.approx(BOOT_ASK_MIB)
    assert aw.run_stage_bytes(pages, BOOT_RUN) / MIB > BOOT_FREE_MIB   # the old single fill: refused
    cp = aw.stage_chunk_pages(BOOT_CHUNK_TOKENS, 1, pages)
    assert cp == 4096
    assert aw.run_stage_bytes(cp, BOOT_RUN) / MIB == 128.0           # the named transient
    assert aw.run_stage_bytes(cp, BOOT_RUN) / MIB < BOOT_FREE_MIB
    assert len(aw.stage_chunks(pages, cp)) == 7


def test_stage_chunk_pages_rules():
    assert aw.stage_chunk_pages(4096, 1, 137) == 137          # a short write keeps its exact stage
    assert aw.stage_chunk_pages(4096, 1, 28224) == 4096
    assert aw.stage_chunk_pages(4096, 64, 1000) == 64          # paged pool: chunk tokens / page tokens
    assert aw.stage_chunk_pages(4000, 64, 1000) == 63          # ceil
    assert aw.stage_chunk_pages(None, 1, 500) == 500           # no chunked prefill: one fill (named by the caller)
    assert aw.stage_chunk_pages(-1, 1, 500) == 500
    assert aw.stage_chunk_pages(4096, 1, 0) == 0
    assert aw.stage_chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert aw.stage_chunks(0, 4) == []
    with pytest.raises(ValueError):
        aw.stage_chunks(3, 0)


def test_is_oom_separates_card_refusal_from_kernel_failure():
    assert aw.is_oom(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 882.00 MiB"))
    assert aw.is_oom(RuntimeError("hicache.cuh:518: CUDA error: out of memory"))
    assert not aw.is_oom(RuntimeError("JIT build failed"))
    assert not aw.is_oom(RuntimeError("nvcc: out of memory while compiling hicache.cuh"))   # not the card
    assert not aw.is_oom(ValueError("run 1536 B / page 4096 B not a whole number"))


class _FakeEvent:
    clock = [0.0]

    def __init__(self):
        self.t = None
        self.done = True

    def record(self):
        _FakeEvent.clock[0] += 2.5
        self.t = _FakeEvent.clock[0]

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.t - self.t


def test_write_clock_folds_completed_pairs_per_mode():
    c = aw.WriteClock(event_factory=_FakeEvent)
    t = c.mark()
    c.stop(t, "run", 4 * MIB)
    t = c.mark()
    c.stop(t, "cell", 2 * MIB)
    assert c.totals["run"][0] == 1 and c.totals["run"][2] == 4 * MIB
    assert c.totals["cell"][0] == 1
    s = c.summary()
    assert s.startswith("cell=1w/") and " run=1w/" in s
    # a pair still in flight is kept, not read
    t = c.mark()
    c._open.append((t, types.SimpleNamespace(query=lambda: False), "run", 1))
    c.drain()
    assert c.totals["run"][0] == 1 and len(c._open) == 1
    # an instrument never fails the write
    bad = aw.WriteClock(event_factory=lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))
    assert bad.mark() is None
    bad.stop(None, "run", 1)
    assert bad.totals == {}


# --- the real _backup_arena on a card-memory dummy ----------------------------

L, CELL = 2, 1024
RUN = L * CELL                       # 2048 B per side
PAGE = 2 * RUN                       # arena page: K run then V run
K_OFFS = [0, CELL]
V_OFFS = [RUN, RUN + CELL]
N_TOK, N_SLOTS, B = 1500, 1200, 1000
OLD_ASK = B * 2 * RUN                # the one-fill stage of this write
FREE = int(OLD_ASK * BOOT_FREE_MIB / BOOT_ASK_MIB)   # the boot's free/ask ratio


class _Dev:
    type = "cuda"


class _OnCard:
    """A host tensor that says it lives on the card (``.device.type == 'cuda'``)."""

    def __init__(self, t):
        self._t = t

    device = _Dev()

    def to(self, device=None, dtype=None, **kw):
        return self._t.to(dtype=dtype) if dtype is not None else self._t

    def __getattr__(self, name):
        return getattr(self._t, name)


class _CardDummy:
    """torch with a card that has ``free`` bytes: ``empty(..., device=card)``
    above it raises the OutOfMemoryError the boot saw."""

    def __init__(self, free):
        self.free = free
        self.asks = []

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, shape, dtype=None, device=None, **kw):
        t = torch.empty(shape, dtype=dtype)
        if isinstance(device, _Dev):
            nbytes = t.numel() * t.element_size()
            self.asks.append(nbytes)
            if nbytes > self.free:
                raise torch.OutOfMemoryError(
                    f"CUDA out of memory. Tried to allocate {nbytes / MIB:.2f} MiB. GPU 0 has a total "
                    f"capacity of 19.58 GiB of which {self.free / MIB:.2f} MiB is free.")
        return t


def _copy(dst, src, n):
    ctypes.memmove(int(dst), int(src), int(n))


class _Kernels:
    def __init__(self, fail_run=None):
        self.calls = []
        self.fail_run = fail_run

    def run(self, *, ptr_dst, indices_dst, ptr_src, indices_src, cache_src_stride_bytes,
            cache_dst_stride_bytes, element_size, block_quota):
        """hicache_transfer_all_layer<kIsMLA=true> on host pointers."""
        if self.fail_run is not None:
            raise self.fail_run
        self.calls.append(("run", int(indices_dst.numel())))
        pd, ps = [int(x) for x in ptr_dst.tolist()], [int(x) for x in ptr_src.tolist()]
        for i_d, i_s in zip(indices_dst.tolist(), indices_src.tolist()):
            for l in range(len(pd)):
                _copy(pd[l] + i_d * cache_dst_stride_bytes, ps[l] + i_s * cache_src_stride_bytes, element_size)

    def cell(self, *, k_ptr_dst, v_ptr_dst, indices_dst, k_ptr_src, v_ptr_src, indices_src,
             kv_cache_src_stride_bytes, kv_cache_dst_stride_bytes, element_size, block_quota):
        """hicache_transfer_all_layer<kIsMLA=false> on host pointers."""
        self.calls.append(("cell", int(indices_dst.numel())))
        kd, vd = [int(x) for x in k_ptr_dst.tolist()], [int(x) for x in v_ptr_dst.tolist()]
        ks, vs = [int(x) for x in k_ptr_src.tolist()], [int(x) for x in v_ptr_src.tolist()]
        for i_d, i_s in zip(indices_dst.tolist(), indices_src.tolist()):
            for l in range(len(kd)):
                _copy(kd[l] + i_d * kv_cache_dst_stride_bytes, ks[l] + i_s * kv_cache_src_stride_bytes, element_size)
                _copy(vd[l] + i_d * kv_cache_dst_stride_bytes, vs[l] + i_s * kv_cache_src_stride_bytes, element_size)


@pytest.fixture
def desk(monkeypatch):
    g = torch.Generator().manual_seed(179)
    kb = [torch.randint(0, 256, (N_TOK, CELL), dtype=torch.uint8, generator=g) for _ in range(L)]
    vb = [torch.randint(0, 256, (N_TOK, CELL), dtype=torch.uint8, generator=g) for _ in range(L)]
    arena = torch.zeros(N_SLOTS * PAGE, dtype=torch.uint8)
    base = arena.data_ptr()
    pool = types.SimpleNamespace(
        element_dim=CELL, dtype=torch.uint8, _k_offs=list(K_OFFS), _v_offs=list(V_OFFS),
        _page_bytes=PAGE, _data_base=base, token_stride_size=CELL, layer_num=L,
        pin_slots=lambda s: None,
        _arena_ptrs=lambda dev: (torch.tensor([base + o for o in K_OFFS], dtype=torch.uint64),
                                 torch.tensor([base + o for o in V_OFFS], dtype=torch.uint64)),
    )
    dpool = types.SimpleNamespace(
        k_buffer=[_OnCard(kb[0])] + kb[1:], v_buffer=vb,
        k_data_ptrs=torch.tensor([t.data_ptr() for t in kb], dtype=torch.uint64),
        v_data_ptrs=torch.tensor([t.data_ptr() for t in vb], dtype=torch.uint64),
    )
    kern = _Kernels()
    card = _CardDummy(FREE)
    monkeypatch.setattr(ap, "torch", card)
    monkeypatch.setattr(ap, "_pinned_to", lambda t, dev: t)
    monkeypatch.setattr(ap, "jit_transfer_hicache_all_layer_mla", lambda **kw: kern.run(**kw))
    monkeypatch.setattr(ap, "jit_transfer_hicache_all_layer", lambda **kw: kern.cell(**kw))
    monkeypatch.setattr(ap, "_ARENA_WRITE_CLOCK", aw.WriteClock(event_factory=_FakeEvent))
    monkeypatch.setattr(ap, "_ARENA_WRITE_OOM", [0])
    monkeypatch.setattr(ap, "_STAGE_CHUNK_TOKENS", [128])
    monkeypatch.delenv(aw.MODE_ENV, raising=False)
    slots = torch.randperm(N_SLOTS, generator=g)[:B].to(torch.int64)
    toks = torch.randint(0, N_TOK, (B,), generator=g, dtype=torch.int64)
    ns = types.SimpleNamespace(pool=pool, dpool=dpool, arena=arena, kb=kb, vb=vb, kern=kern, card=card,
                               slots=slots, toks=toks)

    def write(slots_=None, toks_=None):
        s = slots if slots_ is None else slots_
        t = toks if toks_ is None else toks_
        ap.ArenaMHAHostPool._backup_arena(pool, dpool, s, _OnCard(t))

    ns.write = write
    return ns


def _expected(ns, slots, toks):
    want = torch.zeros_like(ns.arena)
    for s, t in zip(slots.tolist(), toks.tolist()):
        for l in range(L):
            want[s * PAGE + K_OFFS[l]:s * PAGE + K_OFFS[l] + CELL] = ns.kb[l][t]
            want[s * PAGE + V_OFFS[l]:s * PAGE + V_OFFS[l] + CELL] = ns.vb[l][t]
    return want


def test_repro_one_fill_is_refused_by_the_boots_free_bytes(desk, monkeypatch):
    """The old shape (one fill = the whole write) on the boot's free/ask ratio:
    the card refuses the stage and the write goes the cell way."""
    monkeypatch.setattr(ap, "_STAGE_CHUNK_TOKENS", [None])
    desk.write()
    assert desk.card.asks == [OLD_ASK] and OLD_ASK > FREE
    assert [c[0] for c in desk.kern.calls] == ["cell"]
    assert ap._ARENA_WRITE_OOM[0] == 1
    assert torch.equal(desk.arena, _expected(desk, desk.slots, desk.toks))


def test_chunked_run_write_fits_the_free_bytes_and_writes_the_same_bytes(desk):
    desk.write()
    assert max(desk.card.asks) == 128 * 2 * RUN <= FREE
    calls = desk.kern.calls
    assert [c[0] for c in calls] == ["run"] * 8                     # ceil(1000 / 128) fills, no cell
    assert sum(c[1] for c in calls) == B * (RUN // aw.RUN_ELEMENT_BYTES)
    assert ap._ARENA_WRITE_OOM[0] == 0
    assert getattr(desk.pool, "_write_mode", None) is None
    assert torch.equal(desk.arena, _expected(desk, desk.slots, desk.toks))
    assert ap._ARENA_WRITE_CLOCK.totals["run"][0] == 1


def test_an_oom_does_not_turn_the_run_mode_off_for_the_boot(desk, monkeypatch):
    """Old: 'cell mode from now on' after ONE refused stage. New: that write goes
    the cell way, the next write runs again."""
    desk.card.free = 1                       # the card is full for this one write
    desk.write()
    assert [c[0] for c in desk.kern.calls] == ["cell"]
    assert getattr(desk.pool, "_write_mode", None) is None
    desk.card.free = FREE
    desk.kern.calls.clear()
    s2, t2 = desk.slots.flip(0), desk.toks.flip(0)
    desk.write(s2, t2)
    assert [c[0] for c in desk.kern.calls] == ["run"] * 8
    assert torch.equal(desk.arena, _expected(desk, s2, t2))


def test_a_kernel_failure_still_falls_back_one_way(desk):
    desk.kern.fail_run = RuntimeError("JIT build failed")
    desk.write()
    assert desk.pool._write_mode == "cell"
    assert [c[0] for c in desk.kern.calls] == ["cell"]
    assert torch.equal(desk.arena, _expected(desk, desk.slots, desk.toks))


def test_a_short_write_keeps_its_exact_stage(desk):
    desk.write(desk.slots[:37], desk.toks[:37])
    assert desk.card.asks == [37 * 2 * RUN]
    assert [c[0] for c in desk.kern.calls] == ["run"]


def test_an_oom_in_the_middle_of_the_fills_rewrites_the_whole_write_by_cell(desk, monkeypatch):
    """Fills 1-2 went by run, the card refuses at fill 3: the cell kernel writes
    every page again (same bytes, same slots, same stream) -- nothing half-written."""
    n = [0]

    def pinned(t, dev):
        n[0] += 1
        if n[0] == 7:              # 2 pointer tensors, then 2 index tensors per fill: fill 3's first
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB.")
        return t

    monkeypatch.setattr(ap, "_pinned_to", pinned)
    desk.write()
    assert [c[0] for c in desk.kern.calls] == ["run", "run", "cell"]
    assert ap._ARENA_WRITE_OOM[0] == 1 and getattr(desk.pool, "_write_mode", None) is None
    # the begun run is timed as aborted, the cell fallback on its own
    assert ap._ARENA_WRITE_CLOCK.totals["run_aborted"][0] == 1
    assert ap._ARENA_WRITE_CLOCK.totals["cell"][0] == 1 and "run" not in ap._ARENA_WRITE_CLOCK.totals
    assert torch.equal(desk.arena, _expected(desk, desk.slots, desk.toks))


def test_the_previous_stage_is_released_before_the_next_is_asked(desk):
    desk.write()
    first = desk.pool._write_stage_keep
    assert first is not None and first.shape[0] == 128
    desk.card.free = 128 * 2 * RUN          # room for exactly ONE stage
    desk.write(desk.slots.flip(0), desk.toks.flip(0))
    assert [c[0] for c in desk.kern.calls].count("cell") == 0


def test_paged_pool_fills_carry_whole_pages(desk, monkeypatch):
    """P = 2 tokens per arena page: fill k gathers didx[c0*P:c1*P] and plans slots[c0:c1]."""
    P = 2
    block = P * CELL
    run = L * block
    page = 2 * run
    k_offs = [0, block]
    v_offs = [run, run + block]
    pool = desk.pool
    pool._arena_page_tokens = P
    pool._k_offs, pool._v_offs, pool._page_bytes = list(k_offs), list(v_offs), page
    n_slots, b = 300, 250
    arena = torch.zeros(n_slots * page, dtype=torch.uint8)
    pool._data_base = arena.data_ptr()
    monkeypatch.setattr(ap, "_STAGE_CHUNK_TOKENS", [64])        # 32 pages per fill -> 8 fills
    g = torch.Generator().manual_seed(2)
    slots = torch.randperm(n_slots, generator=g)[:b].to(torch.int64)
    toks = torch.randint(0, N_TOK, (b * P,), generator=g, dtype=torch.int64)
    desk.write(slots, toks)
    assert [c[0] for c in desk.kern.calls] == ["run"] * 8
    assert max(desk.card.asks) == 32 * 2 * run
    want = torch.zeros_like(arena)
    for p, s in enumerate(slots.tolist()):
        for j in range(P):
            t = int(toks[p * P + j])
            for l in range(L):
                o = s * page + k_offs[l] + j * CELL
                want[o:o + CELL] = desk.kb[l][t]
                o = s * page + v_offs[l] + j * CELL
                want[o:o + CELL] = desk.vb[l][t]
    assert torch.equal(arena, want)


def test_chunk_tokens_are_read_again_when_not_yet_published(monkeypatch):
    import flliper.srt.runtime_context as rc

    monkeypatch.setattr(ap, "_STAGE_CHUNK_TOKENS", [])
    monkeypatch.setattr(rc, "get_server_args", lambda: (_ for _ in ()).throw(RuntimeError("not set")))
    assert ap._stage_chunk_tokens() is None and ap._STAGE_CHUNK_TOKENS == []     # not frozen
    monkeypatch.setattr(rc, "get_server_args", lambda: types.SimpleNamespace(chunked_prefill_size=4096))
    assert ap._stage_chunk_tokens() == 4096 and ap._STAGE_CHUNK_TOKENS == [4096]
    monkeypatch.setattr(ap, "_STAGE_CHUNK_TOKENS", [])
    monkeypatch.setattr(rc, "get_server_args", lambda: types.SimpleNamespace(chunked_prefill_size=-1))
    assert ap._stage_chunk_tokens() is None and ap._STAGE_CHUNK_TOKENS == [None]   # stated: no chunked prefill
