"""DP-NACHLAUF 02.10.: PP0's told clamp reuses the prefetch probe's anchored
answer when the read completed exactly that span.

N5x warm D->P: all reads finished in the flip, yet TOLD-ACKED came 0.19-0.23 s
after the wake -- the followers waited 214 ms (CHAIN-RECV blocked) for PP0's
first pass, whose #1416d TOLD-PROBE re-hashed 73728 pages and ran a full
batch_exists_v2 (the prefetch thread's 'exists' was 207-232 ms for the same
question). Pinned (red before): told == hit*page -> no store call, told
returned; told short of the hit / no recorded hit / switch off -> the probe
runs as before (and a clamp it computes is applied); the hit is consumed once;
the hybrid controller records the hit.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_store_told as m  # noqa: E402


class _Backend:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def batch_exists_v2(self, keys, transfers, extra):
        self.calls += 1
        return SimpleNamespace(kv_hit_pages=self.answer)


def _setup(monkeypatch, answer):
    be = _Backend(answer)
    cc = SimpleNamespace(page_size=1, storage_backend=be,
                         get_hash_str=lambda ids, last, page_size=1: [f"k{i}" for i in range(len(ids))],
                         _presence_pool_transfers=lambda: ["mamba"])
    sched = SimpleNamespace(cache_controller=cc, tree_cache=SimpleNamespace(cache_controller=cc))
    req = SimpleNamespace(rid="weg2-3-4", origin_input_ids=list(range(120)))
    monkeypatch.setattr(m, "_tree_key_probe_armed", lambda: False)
    monkeypatch.delenv(m.ENV_CLAMP_REUSE, raising=False)
    return sched, cc, be, req


def test_exact_span_reuses_the_prefetch_answer(monkeypatch):
    sched, cc, be, req = _setup(monkeypatch, answer=50)
    m.note_probe_hit(cc, "weg2-3-4", 100)
    assert m._anchor_clamp(sched, req, 100) == 100
    assert be.calls == 0
    assert "weg2-3-4" not in cc._weg2_probe_hit_pages          # consumed once


def test_short_read_or_no_hit_or_switch_off_probes(monkeypatch):
    sched, cc, be, req = _setup(monkeypatch, answer=80)
    m.note_probe_hit(cc, "weg2-3-4", 100)
    assert m._anchor_clamp(sched, req, 90) == 80               # short read: probe + clamp
    assert be.calls == 1
    assert m._anchor_clamp(sched, req, 100) == 80              # no hit recorded any more
    assert be.calls == 2
    monkeypatch.setenv(m.ENV_CLAMP_REUSE, "0")
    m.note_probe_hit(cc, "weg2-3-4", 100)
    m._anchor_clamp(sched, req, 100)
    assert be.calls == 3


def test_hybrid_controller_records_the_hit():
    from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

    src = inspect.getsource(hcc.HybridCacheController._storage_hit_query)
    assert "_wst.note_probe_hit(self, getattr(operation, 'request_id', None), kv_hit_pages)".replace("'", '"') in src
