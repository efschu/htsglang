"""DP-NACHLAUF 02.10.: the store probe answers L3 presence / readability per
chunk of pages on first ask, not for every KV page up front.

N5m (5576ce0f16, D->P, pdflip-8-9, 72786 tokens): PP0's prefetch queued 1270 ms
before a 92-ms read; #969G counted 100k -> 200k key derivations inside the
probe -- the trailing-pages rule (mamba anchors) asks a few dozen pages, the
bulk derived ~61k keys per pool. Pinned (red before): chunked and bulk give
the same answer on a randomised store (arena states, L3 index, readable set,
an all-pages and a trailing-pages pool); the chunked form derives far fewer
keys; the #969G probe past its cap no longer reads the env.
"""
from __future__ import annotations

import os
import random
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache import hicache_storage as hs  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolTransfer  # noqa: E402


def _fake(seed, n=3000, kv_fast=2500):
    rnd = random.Random(seed)
    keys = [f"k{i:05d}" for i in range(n)]
    calls = {"derive": 0}
    arena = {}
    for k in keys:
        arena[k] = 2
        arena[f"{k}.mamba"] = 2 if rnd.random() < 0.002 else 0
        arena[f"{k}.draft"] = 2 if rnd.random() < 0.7 else 0
    l3 = {f"{k}.mamba" for k in keys if rnd.random() < 0.01} | {f"{k}.draft" for k in keys if rnd.random() < 0.5}
    readable = {s for s in l3 if rnd.random() < 0.8}
    disk_kv = set(keys[:kv_fast + rnd.randint(0, 400)])
    f = hs.HiCacheFile.__new__(hs.HiCacheFile)

    def gck(k, name=None):
        calls["derive"] += 1
        return k if name in (None, "__default__", hs.PoolName.KV) else f"{k}.{name}"

    f._canonical_probe_mismatch = lambda: None
    f._arena_kv_present_prefix = lambda ks: kv_fast
    f._collect_existing_component_keys = lambda chunk, pt: {f"{k}.bin" for k in chunk if k in disk_kv}
    f._get_component_key = gck
    f._arena_dir = lambda: "/arena"
    f._canonical_window = lambda k0: SimpleNamespace(total_bytes=1)
    f._arena_for = lambda nb: SimpleNamespace(find_states=lambda stems: [arena.get(s, 0) for s in stems])
    f._suffix_for_key = lambda k: ("",)
    f._l3_index = lambda: SimpleNamespace(has=lambda stems: [s in l3 for s in stems])
    f._readable_stems = lambda stems: [s for s in stems if s in readable]
    pts = [PoolTransfer(name="mamba", keys=["a"], hit_policy=PoolHitPolicy.TRAILING_PAGES),
           PoolTransfer(name="draft", keys=None, hit_policy=PoolHitPolicy.ALL_PAGES)]
    return f, keys, pts, calls


@pytest.mark.parametrize("seed", range(6))
def test_chunked_equals_bulk(monkeypatch, seed):
    out = {}
    for mode in ("0", "1"):
        monkeypatch.setenv(hs.PROBE_CHUNKED_ENV, mode)
        f, keys, pts, calls = _fake(seed)
        r = f.batch_exists_v2(keys, pts, None)
        out[mode] = (r.kv_hit_pages, dict(r.extra_pool_hit_pages), calls["derive"])
    assert out["0"][:2] == out["1"][:2], out
    assert out["1"][2] < out["0"][2], out


def test_switch_default_on():
    assert hs.probe_chunked_on({})
    assert not hs.probe_chunked_on({hs.PROBE_CHUNKED_ENV: "0"})


def test_trace_past_the_cap_reads_no_env(monkeypatch):
    hs._969g_trace("lookup", "x", "s0")   # sets the cached cap
    assert hs._969G_CAP is not None
    hs.HiCacheFile._969g_n = hs._969G_CAP + 5

    def boom():
        raise AssertionError("env read past the cap")

    monkeypatch.setattr(hs.envs.FLLIPER_HICACHE_KEY_TRACE_CAP, "get", boom, raising=False)
    before = getattr(hs.HiCacheFile, "_969g_suppressed", 0)
    hs._969g_trace("lookup", "x", "s1")
    assert getattr(hs.HiCacheFile, "_969g_suppressed", 0) == before + 1


def _fetch_cap_lines(caplog):
    return [r.getMessage() for r in caplog.records if "#1028B FETCH CAP" in r.getMessage()]


def test_anchor_probe_skipped_when_claim_nonzero(monkeypatch, caplog):
    """N5m n=3: claimed=61343 lost=27 still scanned every key per pool."""
    monkeypatch.setenv(hs.PROBE_CHUNKED_ENV, "1")
    f, keys, pts, calls = _fake(5)
    pts = pts[:1]   # the mamba trailing rule alone: claim > 0, < kv
    f._1028b_n = 0
    with caplog.at_level("WARNING"):
        r = f.batch_exists_v2(keys, pts, None)
    assert 0 < r.kv_hit_pages < r.kv_uncapped, r
    lines = _fetch_cap_lines(caplog)
    assert lines and "skipped(claimed>0)" in lines[0], lines
    # far fewer derivations than one per key per pool
    assert calls["derive"] < 2 * len(keys), calls


def test_anchor_probe_still_answers_claimed_zero(monkeypatch, caplog):
    monkeypatch.setenv(hs.PROBE_CHUNKED_ENV, "1")
    f, keys, pts, calls = _fake(1)
    # no mamba anywhere -> trailing boundary 0 -> claimed=0
    f._arena_for = lambda nb: SimpleNamespace(find_states=lambda stems: [0 if ".mamba" in s else 2 for s in stems])
    f._l3_index = lambda: SimpleNamespace(has=lambda stems: [False if ".mamba" in s else True for s in stems])
    f._1028b_n = 0
    with caplog.at_level("WARNING"):
        r = f.batch_exists_v2(keys, pts, None)
    assert r.kv_hit_pages == 0
    lines = _fetch_cap_lines(caplog)
    assert lines and "skipped" not in lines[0] and "'mamba': (0, -1)" in lines[0], lines


def test_hybrid_prefetch_operation_carries_stage_times():
    import inspect
    from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

    init = inspect.getsource(hcc.PrefetchOperation.__init__)
    assert "self.stage_times" in init
    src = inspect.getsource(hcc.HybridCacheController._storage_hit_query)
    assert '_st["hashed"] = time.monotonic()' in src
