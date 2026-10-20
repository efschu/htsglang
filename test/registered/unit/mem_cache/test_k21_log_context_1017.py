"""h16v S13 A3 (log only, K21): stop messages name FIFO depth (own, tp0), both findings and handover_n;
on_tree_reset logs every discarded open stop / open pair in ONE line with counters."""

from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import test_form_a_host_shadow_r12 as T  # noqa: E402
from test_k21_ack_pairing_1017 import _run  # noqa: E402,F401
from test_k21_both_directions_1017 import _wire  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

m = T.m


def _stop_text(orphans, steps, release_fails=False):
    ranks = T._ranks()
    leaves = [_wire(r, o, release_fails=(release_fails and i == 0)) for i, (r, o) in enumerate(zip(ranks, orphans))]
    if release_fails:
        leaves[0].component_data[T.FULL].host_value = T.torch.arange(0, T.NODE)
    with T._switch(True):
        for st in steps:
            try:
                if st[0] == "ack":
                    r = ranks[st[1]]
                    with T._as_rank(r):
                        U._k12_direct_l3_verdict(r, leaves[st[1]], 1 if orphans[st[1]] else 0)
                else:
                    T._broadcast(ranks)
            except m.K21RankDivergence as e:
                return str(e)
    return None


def test_worker_orphan_stop_names_fifo_findings_and_handover_n():
    msg = _stop_text([False, True, False], [("ack", 1), ("bc",), ("bc",), ("ack", 0), ("bc",)])
    assert msg is not None
    assert "fifo own=" in msg and "tp0=" in msg
    assert "own_finding=1" in msg and "tp0_finding=0" in msg
    assert "handover_n=" in msg


def test_tp0_orphan_no_state_stop_names_both_findings():
    msg = _stop_text([True, False, False], [("ack", 0), ("bc",), ("bc",), ("ack", 1), ("ack", 2), ("bc",)],
                     release_fails=True)
    assert msg is not None
    assert "own_finding=0" in msg and "tp0_finding=1" in msg and "fifo own=" in msg and "handover_n=" in msg


def test_reset_logs_one_line_with_open_stop_and_open_pairs(caplog):
    ranks = T._ranks()
    leaves = [_wire(r, True) for r in ranks]
    with T._switch(True):
        with T._as_rank(ranks[1]):
            U._k12_direct_l3_verdict(ranks[1], leaves[1], 1)  # an open own finding
            m._S.k21_stop_msg = "K21 test stop"  # an open deferred stop
            with caplog.at_level(logging.WARNING):
                m.on_tree_reset(ranks[1])
    lines = [r.getMessage() for r in caplog.records if "K21 RESET-DROPS" in r.getMessage()]
    assert len(lines) == 1
    assert "open_stop=1" in lines[0] and "open_own=1" in lines[0] and "open_tp0=0" in lines[0]
    assert "handover_n=" in lines[0] and "K21 test stop" in lines[0]


def test_reset_without_open_k21_state_is_silent(caplog):
    ranks = T._ranks()
    with T._switch(True):
        with T._as_rank(ranks[1]):
            with caplog.at_level(logging.WARNING):
                m.on_tree_reset(ranks[1])
    assert not [r for r in caplog.records if "K21 RESET-DROPS" in r.getMessage()]
