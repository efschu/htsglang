# SPDX-License-Identifier: Apache-2.0
"""PROBE-FAST (02.10.): the front's store presence probe off the per-stem Python path.

N5q (...10021513_9126170083_1002_151409.front.log) epoch 4, pdflip-3-5:
  12.988  PDFLIP SESSION (arrival)
  13.120  X-EXACT count done (count_ms=132.0, reused=47 encoded=98751)
  13.540  'L3-INDEX-PRESENCE tier=l2_arena depth=73728 pages=73728 l3_pages=0
          kv_pages=74064 tokens=98798 probe_ms=413.3' -- SERIAL after the count
          (the probe needs the exact ids), in the same tokenizer worker
  13.541  ROUTE-VERDICT long / BATCH queued
Same shape in N5q pdflip-6-8 / 8-10 (438 / 435 ms) and N5t 6-6 / 8-8 / 10-10.

The 27B keys pages of ONE token, so the probe asks ~74k stems. Desk profile of
that shape: the per-stem Python (encode per tier, a ctypes array per call and
tier, list/bool conversions, a memo dict, an any() per stem) dominates the C
lookups. PROBE-FAST (``FLLIPER_PDFLIP_FRONT_PROBE_FAST``, default on) writes the
stems once into a NUL-terminated byte buffer, hands one ``char **`` to every
tier, asks the arenas first and the L3 index / further arenas only for what the
union still lacks, and reads numpy verdicts. Marker ``probe=fast asked=N`` on
the L3-INDEX-PRESENCE line.

DANGER DIRECTION: a different Depth than the list form (an over- or
under-credit changes the route). Pinned: equality on a grid of store shapes,
the non-uniform-hash and non-ASCII fallbacks, the switch. Hermetic, CPU.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import time
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=40, suite="stage-a-test-cpu")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.storage.file.l3_index import L3Index  # noqa: E402
from flliper.srt.pdflip import front_store as FS  # noqa: E402

SFX = "_Qwen3.8-27B-INT8-gdncov-vocabembed_probefasttest"
FIELDS = ("tokens", "kv_pages", "pages", "tier", "l3_pages")


def _ids(n, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


class _Store:
    def __init__(self, tmp, comps=()):
        self.dir = os.path.join(tmp, "l3-store")
        os.makedirs(self.dir)
        self.arena = os.path.join(tmp, "pdflip-arena-probefast")
        for g in ("P", "D"):
            with open(os.path.join(self.dir, f"L3_SUFFIXES.{g}.json"), "w") as f:
                json.dump({"group": g, "suffixes": [SFX]}, f)
        for c in comps:  # a sampled shard name makes the component known to the probe
            d = os.path.join(self.dir, "ab")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f"ab00.{c}{SFX}.bin"), "wb").close()
        os.makedirs(FS.l3_index_path(self.arena).rsplit("/", 1)[0])
        self.index = L3Index(FS.l3_index_path(self.arena), cap=1 << 19)
        self.arenas = {}

    def l3(self, stems):
        for off in range(0, len(stems), 4096):
            self.index.add(stems[off:off + 4096])

    def l2(self, stems, width, slots):
        from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

        os.makedirs(self.arena, exist_ok=True)
        a = self.arenas.get(width)
        if a is None:
            a = self.arenas[width] = ShmArena(os.path.join(self.arena, f"arena-{width}.bin"),
                                              width, slots)
        buf = (ctypes.c_char * width)()
        for off in range(0, len(stems), 4096):
            part = stems[off:off + 4096]
            st = a.write(part, [width] * len(part), [((0, width),)] * len(part),
                         [ctypes.addressof(buf)] * len(part))
            assert set(st) <= {1, 2}, st

    def probe(self, hasher=None):
        probe, why = FS.open_store_presence(self.dir, self.arena, 1, True, True)
        assert probe is not None, why
        if hasher is not None:
            probe.hasher = hasher
        return probe


def kv(h, a, b):
    return [f"{x}{SFX}" for x in h[a:b]]


def mamba(h, idx):
    return [f"{h[i]}.mamba{SFX}" for i in idx]


def comp(h, c, a, b):
    return [f"{x}.{c}{SFX}" for x in h[a:b]]


def _same(probe, ids):
    fast = probe.depth(ids, fast=True)
    slow = probe.depth(ids, fast=False)
    assert fast.form == "fast" and slow.form == "list", (fast, slow)
    assert tuple(getattr(fast, k) for k in FIELDS) == tuple(getattr(slow, k) for k in FIELDS), (fast, slow)
    return fast, slow


# ---------------------------------------------------------------------------
# 1. the same Depth on a grid of store shapes
# ---------------------------------------------------------------------------

def _shape(store, name, h):
    n = len(h)
    if name == "n5q_l2_only":            # N5q pdflip-3-5, scaled: KV in L2 past the anchor
        store.l2(kv(h, 0, n - 300), 64, n + 64)
        store.l2(mamba(h, [4095, n - 300 - 280]), 128, 64)
    elif name == "l3_only":
        store.l3(kv(h, 0, n - 50) + mamba(h, [1000, n - 900]))
    elif name == "union_split":          # L3 head, L2 tail, anchors one per tier
        store.l3(kv(h, 0, n // 2) + mamba(h, [n // 4]))
        store.l2(kv(h, n // 2, n), 64, n + 64)
        store.l2(mamba(h, [n - 10]), 128, 64)
    elif name == "hole_in_both":         # a page missing in both tiers ends the run
        store.l3(kv(h, 0, n // 3))
        store.l2(kv(h, n // 3 + 1, n), 64, n + 64)
        store.l2(mamba(h, [n // 3 - 7, n - 3]), 128, 64)
    elif name == "anchor_only_in_l3_past_l2_run":
        store.l2(kv(h, 0, n - 1000), 64, n + 64)
        store.l3(mamba(h, [n - 1200, n - 500]))
    elif name == "no_anchor":
        store.l2(kv(h, 0, n), 64, n + 64)
    elif name == "early_miss":
        store.l2(kv(h, 1, n), 64, n + 64)
        store.l3(mamba(h, [n - 5]))
    elif name == "nothing":
        pass
    else:
        raise AssertionError(name)


SHAPES = ["n5q_l2_only", "l3_only", "union_split", "hole_in_both",
          "anchor_only_in_l3_past_l2_run", "no_anchor", "early_miss", "nothing"]


@pytest.mark.parametrize("shape", SHAPES)
def test_fast_equals_list_on_every_store_shape(tmp_path, shape):
    ids = _ids(20000, seed=hash(shape) % 1000)
    h = FS.bigram_page_hasher(ids, 1, True)
    store = _Store(str(tmp_path))
    _shape(store, shape, h)
    fast, _ = _same(store.probe(), ids)
    if shape == "n5q_l2_only":
        assert (fast.tier, fast.l3_pages, fast.tokens) == ("l2_arena", 0, len(h) - 300 - 280 + 1)


def test_an_all_pages_component_cuts_the_run_alike(tmp_path):
    ids = _ids(12000, seed=5)
    h = FS.bigram_page_hasher(ids, 1, True)
    store = _Store(str(tmp_path), comps=("qsa",))
    store.l2(kv(h, 0, 11000), 64, 16000)
    store.l3(comp(h, "qsa", 0, 7000))
    store.l2(comp(h, "qsa", 7000, 9000), 64, 12100)
    store.l2(mamba(h, [3000, 8500, 10000]), 128, 64)
    p = store.probe()
    assert p.all_pages == ("qsa",), p.describe()
    fast, _ = _same(p, ids)
    assert fast.tokens == 8501


def test_a_prompt_shorter_than_a_chunk_and_empty_ids(tmp_path):
    store = _Store(str(tmp_path))
    ids = _ids(300, seed=9)
    h = FS.bigram_page_hasher(ids, 1, True)
    store.l3(kv(h, 0, 299) + mamba(h, [200]))
    p = store.probe()
    fast, _ = _same(p, ids)
    assert fast.tokens == 201
    assert p.depth(np.zeros(0, dtype=np.int32), fast=True).tokens == 0


def test_non_uniform_hash_widths_take_the_str_buffer_alike(tmp_path):
    store = _Store(str(tmp_path))
    ids = _ids(3000, seed=11)

    def hasher(ids_, page, bigram):
        return [("%x" % (int(t) * 2654435761 + i)) for i, t in enumerate(ids_[1:])]

    h = hasher(ids, 1, True)
    assert len(set(map(len, h))) > 1
    store.l2(kv(h, 0, 2500), 64, 3100)
    store.l3(mamba(h, [1999]))
    fast, _ = _same(store.probe(hasher=hasher), ids)
    assert fast.tokens == 2000


def test_a_non_ascii_stem_falls_back_to_the_list_form(tmp_path):
    store = _Store(str(tmp_path))
    ids = _ids(1200, seed=12)

    def hasher(ids_, page, bigram):
        return ["é%07d" % i for i in range(int(ids_.size) - 1)]

    h = hasher(ids, 1, True)
    store.l3(kv(h, 0, 1000) + mamba(h, [700]))
    p = store.probe(hasher=hasher)
    d = p.depth(ids, fast=True)
    assert d.form == "list" and d.tokens == 701


def test_the_switch_defaults_on_and_off_is_the_list_form(tmp_path):
    store = _Store(str(tmp_path))
    ids = _ids(2000, seed=13)
    p = store.probe()
    assert envs.FLLIPER_PDFLIP_FRONT_PROBE_FAST.get() is True
    assert p.depth(ids).form == "fast"
    with envs.FLLIPER_PDFLIP_FRONT_PROBE_FAST.override(False):
        assert p.depth(ids).form == "list"


def test_the_pointer_lookups_answer_what_the_list_lookups_answer(tmp_path):
    store = _Store(str(tmp_path))
    ids = _ids(5000, seed=14)
    h = FS.bigram_page_hasher(ids, 1, True)
    store.l3(kv(h, 0, 2000))
    store.l2(kv(h, 1500, 4000), 64, 5100)
    p = store.probe()
    st = FS.StorePresence._Asked(len(h), len(h[0]))
    todo = np.arange(len(h))
    stems, ptrs, n, keep = p._stems_fast(h, None, todo, st)
    assert stems() == kv(h, 0, len(h))
    assert list(p.index.has_ptrs(ptrs, n) != 0) == p.index.has(stems())
    for a in p.arenas.values():
        assert list(a.find_states_ptrs(ptrs, n)) == a.find_states(stems())
    del keep


# ---------------------------------------------------------------------------
# 2. the cost: the N5q shape (74064 KV pages in L2, the anchor at 73728)
# ---------------------------------------------------------------------------

def test_red_the_n5q_shape_costs_well_under_half_of_the_list_form(tmp_path):
    ids = _ids(98798, seed=35)
    h = FS.bigram_page_hasher(ids, 1, True)
    store = _Store(str(tmp_path))
    store.l2(kv(h, 0, 74064), 64, 80000)
    store.l2(mamba(h, [4095, 73727]), 128, 64)
    p = store.probe()
    fast, slow = _same(p, ids)
    assert (fast.tokens, fast.kv_pages, fast.tier) == (73728, 74064, "l2_arena")
    t_fast = min(p.depth(ids, fast=True).ms for _ in range(3))
    t_slow = min(p.depth(ids, fast=False).ms for _ in range(3))
    assert t_fast < 0.5 * t_slow, f"fast {t_fast:.1f} ms vs list {t_slow:.1f} ms"


# ---------------------------------------------------------------------------
# 3. the marker on the front's line
# ---------------------------------------------------------------------------

def test_the_presence_line_names_the_form(tmp_path, caplog):
    from flliper.srt.pdflip import front as F

    ids = _ids(3000, seed=16)
    h = FS.bigram_page_hasher(ids, 1, True)
    store = _Store(str(tmp_path))
    store.l3(kv(h, 0, 2999) + mamba(h, [2047]))
    f = F.Front.__new__(F.Front)
    f.ftok = types.SimpleNamespace(executor=None)
    f.store_probe = store.probe()
    f.counters = __import__("collections").Counter()
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        got = asyncio.run(f._store_probe_depth("pdflip-3-5", ids, 5.0))
    assert got == (2048, "l3_index")
    line = [r.getMessage() for r in caplog.records if "L3-INDEX-PRESENCE" in r.getMessage()]
    assert line and "probe=fast asked=" in line[0], line
