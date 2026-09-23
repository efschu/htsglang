"""fnFL2x24 (23.09.): a Form A expert worker's placement manifest for the
draft region must be WRITTEN EMPTY (#103), not fail.

Since the 27B merge (xsn389) ``card_inventory`` skips meta parameters
itself ("meta-no-bytes") and returns ``None`` for an empty inventory; the
shadow count in ``_write_placement_manifest`` then saw no meta tensor and
iterated over ``None``:

    WEG2-XCHG-MANIFEST-WRITE group=? rank=1 pieces=0 reason=write-failed:
    TypeError: 'NoneType' object is not iterable

on D TP1/TP2 -> no empty weights_draft manifest -> the join saw D on ONE
card -> W68 on all three D ranks -> no draft plan -> P PP0 parked in the
5090 credit wait (129.8 s STALL, W17).
"""
from __future__ import annotations

import os

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

TAG = "weights_draft"


def _arm(monkeypatch, tmp_path, inventory_result):
    monkeypatch.setattr(sh, "card_inventory", lambda **kw: inventory_result)
    monkeypatch.setattr(ms, "weg2_group_name", lambda: "D")
    monkeypatch.setattr(xm, "manifest_dir", lambda default="": str(tmp_path))
    monkeypatch.setattr(xm, "boot_token", lambda: "b1")


def _write(rank: int):
    logs = []
    wx._write_placement_manifest(
        torch.nn.Module(), rank=rank, region_tag=TAG, log=logs.append,
        tp_rank=rank, pp_rank=0, tp_size=3)
    return logs


def test_x24_a_meta_only_inventory_writes_the_empty_shadow_manifest(
        monkeypatch, tmp_path):
    """RED ON dfdb417326: TypeError, 'write-failed', no file."""
    _arm(monkeypatch, tmp_path, (
        None, [("model.embed.weight", "meta-no-bytes")], 1,
        "no-carried-tags:family=()"))
    logs = _write(1)
    assert not any("write-failed" in l for l in logs), logs
    assert any("#103" in l and "shadow_pieces=1" in l for l in logs), logs
    got = [m for m in xm.load_manifests(str(tmp_path), boot_token="b1")
           if m.region_tag == TAG]
    assert [(m.rank, len(m.pieces)) for m in got] == [(1, 0)]


def test_x24_a_rank_with_nothing_to_say_still_writes_no_manifest(
        monkeypatch, tmp_path):
    """The #103 carve-out is for SHADOWS only: an empty inventory without a
    meta skip publishes no rank line the join could count as a holder."""
    _arm(monkeypatch, tmp_path, (None, [], 0, "no-carried-tags:family=()"))
    logs = _write(2)
    assert not any("write-failed" in l for l in logs), logs
    assert any("no manifest written" in l for l in logs), logs
    assert xm.load_manifests(str(tmp_path), boot_token="b1") == ()
