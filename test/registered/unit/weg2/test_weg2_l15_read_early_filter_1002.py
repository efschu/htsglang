# SPDX-License-Identifier: Apache-2.0
"""L15-READ-EARLY-FILTER: the early #248 read skips the rids the whole D
group keeps on the card -- only when every rank predicts the same hold."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.weg2 import l15_manifest, park_l3
from sglang.srt.weg2.l15_manifest import HoldSpan, Manifest


def _sched(rank):
    cell = torch.zeros(4, 2, 2)
    return SimpleNamespace(
        ps=SimpleNamespace(tp_rank=rank), tp_size=3,
        server_args=SimpleNamespace(tp_size=3, rank_gpu_id=None),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(k_buffer=[cell], v_buffer=[cell]))))


def _write(rank, rids, anchor=7):
    m = Manifest(epoch=1, pid=1, spans=tuple(
        HoldSpan(rid=r, depth=2, slots=(1, 2), anchor_slot=1, l2_slots=(3, 4),
                 l2_gens=(1, 1), anchor_l2_slot=anchor, anchor_l2_gen=1) for r in rids),
        rows_by_rank=(1, 1, 1), anchor_slots=2)
    l15_manifest.write(l15_manifest.manifest_path("D", rank, __import__("os").environ), m)


def _env(monkeypatch, tmp_path, refill):
    monkeypatch.setenv("SGLANG_WEG2_L15", "1")
    monkeypatch.setenv("SGLANG_WEG2_L15_MIB", "c1=64,c2=64")
    monkeypatch.setenv("SGLANG_WEG2_L15_MANIFEST", str(tmp_path) + "/")
    if refill:
        monkeypatch.setenv("SGLANG_WEG2_L15_REFILL", "1")
    else:
        monkeypatch.delenv("SGLANG_WEG2_L15_REFILL", raising=False)



def test_every_rank_predicts_the_same_hold_then_filter(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, refill=True)
    for r in range(3):
        _write(r, ["a", "b"])
    votes = [None] * 3
    for r in range(3):
        votes[r] = None
    # each rank's own vote, then the gathered list (identical on every rank)
    own = []
    for r in range(3):
        park_l3.l15_agreed_held_rids(_sched(r), gather=lambda v: own.append(v) or [v])
    assert own[0] is not None and own[0] == own[1] == own[2]
    assert park_l3.l15_agreed_held_rids(_sched(1), gather=lambda v: [v, v, v]) == {"a", "b"}


def test_cap0_without_refill_or_a_missing_manifest_means_no_filter(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, refill=False)
    for r in range(3):
        _write(r, ["a"])
    own = []
    park_l3.l15_agreed_held_rids(_sched(0), gather=lambda v: own.append(v) or [v])
    assert own == [None]                        # cap-0 rank will not vote hold
    v1 = []
    park_l3.l15_agreed_held_rids(_sched(1), gather=lambda v: v1.append(v) or [v])
    assert park_l3.l15_agreed_held_rids(_sched(1), gather=lambda v: [None, v1[0], v1[0]]) == set()


def test_master_off_no_collective(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L15", raising=False)
    called = []
    assert park_l3.l15_agreed_held_rids(_sched(0), gather=lambda v: called.append(v)) == set()
    assert called == []


def test_wake_begin_skips_the_agreed_rids(monkeypatch):
    reqs = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b"), SimpleNamespace(rid="c")]
    sched = SimpleNamespace(weg2_dormant_hold=reqs)
    seen = []
    monkeypatch.setattr(park_l3, "early_enabled", lambda *a: True)
    monkeypatch.setattr(park_l3, "enabled", lambda: True)
    monkeypatch.setattr(park_l3, "_group_d", lambda: True)
    monkeypatch.setattr(park_l3, "l15_agreed_held_rids", lambda s: {"a", "c"})
    monkeypatch.setattr(park_l3, "issue_deferred_reads",
                        lambda s, hold, max_n=None: seen.extend(r.rid for r in hold) or list(hold))
    park_l3.issue_reads_at_wake_begin(sched)
    assert seen == ["b"]


def test_wake_begin_asks_the_group_once_per_wake_under_the_spread(monkeypatch):
    """RELEASE-INTEG 1002: PDFLIP-S calls the wake begin once per weight tag (max_n=1); the
    filter's host gather runs ONCE per wake (cached on the wake sequence), every rank alike."""
    reqs = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b"), SimpleNamespace(rid="c")]
    sched = SimpleNamespace(weg2_dormant_hold=reqs, _weg2_wake_seq=4, weg2_dormant=True)
    asks = []
    monkeypatch.setattr(park_l3, "early_enabled", lambda *a: True)
    monkeypatch.setattr(park_l3, "enabled", lambda: True)
    monkeypatch.setattr(park_l3, "_group_d", lambda: True)
    monkeypatch.setattr(park_l3, "note_hold_order", lambda hold: None)
    monkeypatch.setattr(park_l3, "l15_agreed_held_rids", lambda s: asks.append(1) or {"a"})
    monkeypatch.setattr(park_l3, "issue_deferred_reads", lambda s, hold, max_n=None: list(hold)[:max_n])
    for _ in range(3):
        park_l3.issue_reads_at_wake_begin(sched, max_n=1)
    assert len(asks) == 1
    sched._weg2_wake_seq = 5  # the next wake asks again
    park_l3.issue_reads_at_wake_begin(sched, max_n=1)
    assert len(asks) == 2
