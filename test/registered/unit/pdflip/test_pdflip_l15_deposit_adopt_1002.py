# SPDX-License-Identifier: Apache-2.0
"""L15-14e: D's view of the deposit records (pure parts; wiring later)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from flliper.srt.pdflip import l15_deposit_adopt as A


def _rec(d, rid, stage, **kw):
    r = {"epoch": 4, "rid": rid, "e_start": 30, "n": 10, "anchor_row": 3,
         "upto": 10, "cells": 1, "anchor_bytes": 64, "att_layers": [],
         "linear": [0, 2], "skip_ranks": [0], "failed": None}
    r.update(kw)
    (d / ("depdone.%s.%s.json" % (rid, stage))).write_text(json.dumps(r))


def test_complete_only_when_every_stage_tiles_and_matches(tmp_path):
    _rec(tmp_path, "a", "L0-2", att_layers=[3], linear=[0, 2])
    _rec(tmp_path, "a", "L2-4", att_layers=[7], linear=[2, 4])
    _rec(tmp_path, "b", "L0-2", att_layers=[3], linear=[0, 2])          # missing stage
    _rec(tmp_path, "c", "L0-2", att_layers=[3], linear=[0, 2], failed="aborted")
    _rec(tmp_path, "c", "L2-4", att_layers=[7], linear=[2, 4])
    _rec(tmp_path, "d", "L0-2", att_layers=[3], linear=[0, 2], upto=9)
    _rec(tmp_path, "d", "L2-4", att_layers=[7], linear=[2, 4])
    _rec(tmp_path, "e", "L0-2", att_layers=[3], linear=[0, 2], epoch=3)
    _rec(tmp_path, "e", "L2-4", att_layers=[7], linear=[2, 4], epoch=3)
    _rec(tmp_path, "f", "L0-2", att_layers=[3], linear=[0, 2], anchor_bytes=0)
    _rec(tmp_path, "f", "L2-4", att_layers=[7], linear=[2, 4])
    ok, bad = A.load_deposits(str(tmp_path), 4, [3, 7], 4)
    assert [d.rid for d in ok] == ["a"]
    assert ok[0] == A.Deposit("a", 4, 30, 10, 3, (0,))
    assert "attention layers" in bad["b"] and "aborted" in bad["c"]
    assert "fewer than 10" in bad["d"] and "epoch 3" in bad["e"]
    assert "no END anchor" in bad["f"]


def test_tokens_roundtrip_and_clear(tmp_path):
    A.write_tokens(str(tmp_path), "a", [5, 6, 7])
    assert A.read_tokens(str(tmp_path), "a") == [5, 6, 7]
    assert A.read_tokens(str(tmp_path), "zz") is None
    _rec(tmp_path, "a", "L0-2")
    (tmp_path / "dep.a.json").write_text("{}")
    assert A.clear_epoch_files(str(tmp_path)) == 3


def test_host_rows_to_l2_and_the_span():
    pool = SimpleNamespace(staging_rows=4, _arena_page_tokens=1,
                           slot_gens=lambda s: [7] * len(s))
    assert A.host_rows_to_l2(pool, [2, 4, 9], print) == ((-1, 0, 5), (-1, 7, 7), (-1, 0, 0))
    dep = A.Deposit("a", 4, 30, 3, 2, (0,))
    sp = A.span_for(dep, [1, 2, 3], lambda ids: (None, 0), pool, None, print)
    assert "covers 0 of 3" in sp
    assert "token ids 2 != n 3" in A.span_for(dep, [1, 2], lambda ids: (None, 0), pool, None, print)


def test_rows_free_and_agree():
    import torch

    dep = A.Deposit("a", 4, 5, 3, 2, ())
    kv = SimpleNamespace(free_pages=torch.tensor([5, 6, 7, 9]))
    mb = SimpleNamespace(free_slots=torch.tensor([1, 2]))
    assert A.rows_free(dep, kv, mb) is None
    assert "not free" in A.rows_free(A.Deposit("b", 4, 8, 2, 2, ()), kv, mb)
    assert "anchor row 3" in A.rows_free(A.Deposit("c", 4, 5, 1, 3, ()), kv, mb)
    assert A.agree(["a", "b"], lambda v: [v, ["b", "c"], ["b"]]) == ["b"]


def test_adopt_deposits_on_the_real_tree_agreed_only(tmp_path):
    """Two complete deposits; this rank can take both, a peer only 'a'
    -> exactly 'a' becomes device nodes (slots + anchor reserved)."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_tree_rewrite_1001",
        os.path.join(os.path.dirname(__file__), "test_pdflip_l15_tree_rewrite_1001.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    fx = m._fixture()
    for rid, e0, a in (("a", 40, 3), ("b", 50, 4)):
        _rec(tmp_path, rid, "L0-2", att_layers=[3], linear=[0, 2], e_start=e0, n=6,
             upto=6, anchor_row=a, skip_ranks=[])
        A.write_tokens(str(tmp_path), rid, list(range(800 + e0, 806 + e0)))

    class _Sp:
        pass

    def fake_span(dep, tokens, match, hp, hm, log):
        sp = _Sp()
        sp.rid = dep.rid
        return sp

    import pytest
    mp = pytest.MonkeyPatch()
    mp.setattr(A, "span_for", fake_span)
    try:
        logs = []
        got = A.adopt_deposits(
            directory=str(tmp_path), epoch=4, rank=1, prefix=[0, 1, 2],
            att_layers=[3], n_linear=2, match=None, tree_cache=fx.cache,
            kv_alloc=fx.allocator, mamba_alloc=fx.pool.mamba_allocator,
            host_pool=None, device_pool=None, host_mamba=None, dev_mamba=None,
            gather=lambda v: [v, ["a"]], log=logs.append)
    finally:
        mp.undo()
    assert got == ["a"]
    hit = m._match(fx, list(range(840, 846)))
    assert len(hit.device_indices) >= 5
    assert any("adopted=1" in x for x in logs)
