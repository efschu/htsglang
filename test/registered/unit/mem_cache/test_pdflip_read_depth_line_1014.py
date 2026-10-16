"""READ-DEPTH 1014 (C5): ``PDFLIP-READ-DEPTH rid= pp= landed= of= anchor=
cut_at= cause=`` -- one line per short store read on every stage, naming the
page the walk stopped at and why (the Prefill-Cache monitor counts causes).

The causes of the five h16h cases (boot _211215, P) are driven through the
REAL ``HiCacheController._arena_page_get`` + the REAL
``HiCacheFile.arena_fill_from_disk`` on the REAL C arena:

* pdflip-51-579 / 68-683 / 78-784: the fill's own claim recycled under it
  (``LATE-COMPLETE REFUSED statuses=[3]``) -> ``cause=late_complete``;
* pdflip-72-690: a page the fill landed is evicted by a peer clock before the
  walk's reference (silent on the metal) -> ``cause=fill_evicted_before_ref``;
* a page the store does not hold -> ``cause=not_on_disk``;
* a full read emits nothing.

An instrument: it decides nothing, the landed count is the walk's as before
(asserted against the same walk without the line)."""

import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import cache_controller as cc
from flliper.srt.mem_cache import hicache_storage as hs
from flliper.srt.mem_cache.hicache_storage import HiCacheFile
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena
from flliper.srt.pdflip import read_depth as rd

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096


def _store(tmp_path, stems, missing=()):
    root = tmp_path / "store"
    root.mkdir()
    for i, s in enumerate(stems):
        if s not in missing:
            (root / (s + ".bin")).write_bytes(bytes([0x20 + i]) * TOTAL)
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(str(root), stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda ss: {s: os.path.getsize(_path(s)) for s in ss if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _walk(be, arena, stems, rid="pdflip-x-1"):
    """The P stage's read of ``stems`` through the real page get; returns
    (landed pages, the operation)."""
    pool = types.SimpleNamespace(arena_read=True, arena=arena, _page_bytes=TOTAL,
                                 ensure_bound=lambda b, role: True,
                                 resolve_rows=lambda hi, slots: None)
    ctl = types.SimpleNamespace(mem_pool_host=pool, storage_backend=be, page_size=1)
    op = types.SimpleNamespace(probe_pins=None, completed_tokens=0, request_id=rid,
                               hash_value=list(stems))
    op.increment = lambda k: setattr(op, "completed_tokens", op.completed_tokens + k) or True
    orig = cc.pdflip_suffixed_stems
    cc.pdflip_suffixed_stems = lambda b, hv: list(hv)
    try:
        got = cc.HiCacheController._arena_page_get(ctl, op, list(stems), None)
    finally:
        cc.pdflip_suffixed_stems = orig
    return got, op


def _line(op, caplog):
    with caplog.at_level(logging.WARNING, logger=rd.__name__):
        cause = rd.line(op, 1, 0, 12.0)
    lines = [r.getMessage() for r in caplog.records if "PDFLIP-READ-DEPTH" in r.getMessage()]
    return cause, lines


def _full_arena(tmp_path, n):
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, n)
    old = [f"old{i}" for i in range(n)]
    cl = arena.claim_slots(old, [TOTAL] * n)
    assert arena.complete_slots([c[0] for c in cl], [c[2] for c in cl], [(0, TOTAL)]) == [1] * n
    return arena


def test_late_complete_names_the_reap_race(tmp_path, monkeypatch, caplog):
    """pdflip-78-784 in small, the age gate off (the base's instant reap): the
    walk ends at page 0, cause late_complete."""
    stems = [f"p{i}" for i in range(4)]
    be = _store(tmp_path, stems)
    arena = _full_arena(tmp_path, 8)
    cands = arena.evict_candidates(4)
    arena._lib.arena_reap_stale_aged(arena._base, 0)        # the old reap (another stage)
    real = hs.l3_read_pages_parallel
    done = {"x": False}

    def read(pio, paths, total, ptrs):
        rc = real(pio, paths, total, ptrs)
        if not done["x"]:
            done["x"] = True
            arena.free_slots([c[0] for c in cands], reason="evict_to_disk")   # the live evictor
        return rc

    monkeypatch.setattr(hs, "l3_read_pages_parallel", read)
    got, op = _walk(be, arena, stems, rid="pdflip-78-784")
    assert got == 0
    cause, lines = _line(op, caplog)
    assert cause == rd.LATE_COMPLETE
    assert len(lines) == 1 and "rid=pdflip-78-784" in lines[0] and "landed=0 of=4" in lines[0]
    assert "cut_at=0 cause=late_complete" in lines[0]


def test_a_landed_page_evicted_before_the_reference(tmp_path, monkeypatch, caplog):
    """pdflip-72-690: PP0 filled 589 pages and landed 557 -- no L3-FILL line;
    a peer stage's clock (COMPLETE, unreferenced -> EVICTING -> FREE) took
    pages between the fill and the walk's reference. Without REF-AT-LAND the
    line names it ``fill_evicted_before_ref``; with REF-AT-LAND (300384f533,
    integration line) the clock cannot take a landed page: the read is full."""
    stems = [f"p{i}" for i in range(4)]
    be = _store(tmp_path, stems)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    real_fill = HiCacheFile.arena_fill_from_disk

    def fill_then_peer_clock(self, ar, ss, total, prefix=False, **kw):
        out = real_fill(self, ar, ss, total, prefix=prefix, **kw)
        got = ar.evict_candidates(16)                        # PP1's clock, the real one
        ar.free_slots([c[0] for c in got], reason="test_peer_clock")
        return out

    be.arena_fill_from_disk = types.MethodType(fill_then_peer_clock, be)
    got, op = _walk(be, arena, stems, rid="pdflip-72-690")
    cause, lines = _line(op, caplog)
    if "take_ref" in HiCacheFile.arena_fill_from_disk.__code__.co_varnames:
        assert got == 4 and cause is None                    # REF-AT-LAND: nothing to take
    else:
        assert got == 0
        assert cause == rd.FILL_EVICTED_BEFORE_REF
        assert "landed=0 of=4" in lines[0] and "cut_at=0" in lines[0]


def test_a_page_the_store_lacks(tmp_path, caplog):
    stems = [f"p{i}" for i in range(4)]
    be = _store(tmp_path, stems, missing={"p1"})
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    got, op = _walk(be, arena, stems)
    assert got == 1
    cause, lines = _line(op, caplog)
    assert cause == rd.NOT_ON_DISK and "cut_at=1" in lines[0]


def test_a_full_read_emits_nothing(tmp_path, caplog):
    stems = [f"p{i}" for i in range(4)]
    be = _store(tmp_path, stems)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    got, op = _walk(be, arena, stems)
    assert got == 4
    cause, lines = _line(op, caplog)
    assert cause is None and lines == []


def test_the_census_counts_per_cause(tmp_path, caplog):
    before = rd.census().get(rd.NOT_ON_DISK, 0)
    stems = [f"p{i}" for i in range(3)]
    be = _store(tmp_path, stems, missing={"p2"})
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    _got, op = _walk(be, arena, stems)
    _cause, lines = _line(op, caplog)
    assert rd.census()[rd.NOT_ON_DISK] == before + 1
    assert f"not_on_disk:{before + 1}" in lines[0]


def test_the_line_never_changes_the_walk(tmp_path, monkeypatch):
    """The instrument decides nothing: the same walk with the reason hooks
    neutralised lands the same pages."""
    stems = [f"p{i}" for i in range(4)]
    be = _store(tmp_path, stems, missing={"p3"})
    a1 = ShmArena(str(tmp_path / "kv1.bin"), TOTAL, 16)
    got1, _ = _walk(be, a1, stems)
    monkeypatch.setattr(rd, "note_cut", lambda *a, **k: None)
    monkeypatch.setattr(rd, "fill_reasons_take", lambda: {})
    monkeypatch.setattr(hs, "_read_depth_fill_reasons", lambda *a, **k: None)
    a2 = ShmArena(str(tmp_path / "kv2.bin"), TOTAL, 16)
    got2, _ = _walk(be, a2, stems)
    assert got1 == got2 == 3
