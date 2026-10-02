# SPDX-License-Identifier: Apache-2.0
"""L15-14e: D's view of the deposit records (pure parts; wiring later)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from sglang.srt.weg2 import l15_deposit_adopt as A


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
