# SPDX-License-Identifier: Apache-2.0
"""L15-10 S4n-e: a P stage adopts a hot follow-up's prefix at admission
(orchestration on the REAL tree; the copies are covered by the take tests)."""

from __future__ import annotations

import importlib.util
import os

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker as IC,
)
from sglang.srt.weg2 import l15_share_admit as sa
from sglang.srt.weg2 import l15_share_take

_HERE = os.path.dirname(__file__)


def _fx():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_tree_rewrite_1001",
        os.path.join(_HERE, "test_weg2_l15_tree_rewrite_1001.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m._fixture()


def _admit(fx, monkeypatch, *, take_raises=None, fetch_raises=False):
    calls = {}

    def fake_kv(shares, *, rid, n, stage_layers, p_buffers, p_rows, map_extent):
        if take_raises:
            raise l15_share_take.L15TakeError(take_raises)
        calls["rows"] = list(p_rows)
        return n * len(stage_layers)

    def fake_anchor(shares, *, rid, spec, ratios, stage, p_temporal, p_conv,
                    p_slot, map_extent):
        calls["slot"] = p_slot
        return 64

    monkeypatch.setattr(l15_share_take, "take_kv", fake_kv)
    monkeypatch.setattr(l15_share_take, "take_anchor", fake_anchor)

    def fetch(r):
        if fetch_raises:
            from sglang.srt.weg2.l15_hold_share import L15ShareError
            raise L15ShareError("no hold share published for D rank %d" % r)
        return ({"prefix": [0, 2, 3], "spans": []}, [])

    logs = []
    why = sa.admit(rid="r2", token_ids=list(range(700, 740)),
                   hint={"prev_rid": "r1", "n": 30}, fetch=fetch, n_d_ranks=2,
                   kv_alloc=fx.allocator, mamba_alloc=fx.pool.mamba_allocator,
                   tree_cache=fx.cache, stage_att_layers=[1, 3], p_buffers={},
                   spec=None, stage_linear=(0, 2), p_temporal=None, p_conv=None,
                   map_extent=None, log=logs.append)
    return why, calls, logs


def test_success_adopts_the_prefix_as_device_nodes(monkeypatch):
    m, fx = _fx()
    why, calls, logs = _admit(fx, monkeypatch)
    assert why is None, why
    hit = m._match(fx, list(range(700, 730)))
    assert hit.device_indices.tolist() == calls["rows"][: len(hit.device_indices)]
    assert len(hit.device_indices) >= 29
    dup, _i, shared = IC._mamba_double_claimed(fx.pool.mamba_allocator, fx.cache)
    assert dup == 0 and shared == 0
    assert logs and logs[0].startswith("HOT-HANDOVER rid=r2 from=r1 n=30")


def test_take_refusal_leaves_tree_and_allocators_untouched(monkeypatch):
    m, fx = _fx()
    free_kv = fx.allocator.available_size()
    free_mb = fx.pool.mamba_allocator.available_size()
    why, _c, _l = _admit(fx, monkeypatch, take_raises="rid r1 is not held by D")
    assert why.startswith("take: rid r1 is not held")
    assert fx.allocator.available_size() == free_kv
    assert fx.pool.mamba_allocator.available_size() == free_mb
    assert len(m._match(fx, list(range(700, 730))).device_indices) == 0


def test_missing_share_is_named(monkeypatch):
    m, fx = _fx()
    why, _c, _l = _admit(fx, monkeypatch, fetch_raises=True)
    assert why.startswith("share: no hold share published")


def test_hint_file_roundtrip(tmp_path):
    sa.write_hot_hint(str(tmp_path), "r2", "r1", 30)
    assert sa.hot_hint(str(tmp_path), "r2") == {"prev_rid": "r1", "n": 30}
    sa.reap_hot_hint(str(tmp_path), "r2")
    assert sa.hot_hint(str(tmp_path), "r2") is None
