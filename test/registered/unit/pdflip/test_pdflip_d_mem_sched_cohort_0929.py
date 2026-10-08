"""Flipzeit-Regression 29.09. (2/3), y3j 09291933 (bs4 x 32k after the wake):
D-MEM-SCHED sized the stage for the running requests plus the queue HEAD only;
the second admission got NO_TOKEN and its batch-full gate held until a whole
decode had ended, so the four first tokens came 6 s apart (19,3 / 19,8 / 22,1 /
25,7 s after P-Ende)."""

import asyncio
import collections
import inspect
import json
import logging
import types
import unittest
from array import array
from unittest import mock

import pytest
import torch

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ------------------------------------------------------ 2. D-MEM-SCHED cohort
S0 = 32768
GRID = [S0 * (i + 1) for i in range(8)]


def _req(rid, n_in, n_out=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out,
                                 finished=lambda: False)


@pytest.fixture
def env(monkeypatch):
    from flliper.srt.pdflip import d_seat_vram as dsv

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND", "1")
    caps = []
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: 0)
    sched = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                          speculative_num_draft_tokens=4),
        running_batch=types.SimpleNamespace(reqs=[], batch_is_full=False), waiting_queue=[],
        chunked_req=None, last_batch=None, page_size=64, token_to_kv_pool_allocator=object(),
        _pdflip_group_min_ints=lambda vals: vals)
    setattr(sched, dsv.CTL_ATTR, False)
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e30", n=6, cap=6, done=True, stage=1,
                                                  stage_tokens=GRID[1]))
    return dsv, sched, caps


def test_y3j_the_demand_counts_every_queued_request_a_free_seat_can_take(env):
    dsv, sched, _caps = env
    sched.running_batch.reqs = [_req("pdflip-27-76", 7270)]
    sched.waiting_queue = [_req("pdflip-28-%d" % i, 32835) for i in (85, 86, 87, 88)]
    used, incoming, _rids = dsv.global_demand(sched)
    assert used == 7270
    assert incoming == 4 * 32835                         # base: 32835 (the head only)


def test_the_demand_takes_at_least_the_head_and_at_most_the_free_seats(env):
    dsv, sched, _caps = env
    sched.running_batch.reqs = [_req("r%d" % i, 1000) for i in range(6)]
    sched.waiting_queue = [_req("q1", 5000), _req("q2", 7000)]
    assert dsv.global_demand(sched)[1] == 5000            # seats full: the head, as before
    sched.running_batch.reqs = sched.running_batch.reqs[:5]
    assert dsv.global_demand(sched)[1] == 5000            # one seat: one request
    sched.waiting_queue = []
    assert dsv.global_demand(sched)[1] == 0


def test_y3j_19_52_24_the_cohort_grows_the_stage_and_reopens_the_admission(env, caplog):
    """After the wake at S1 with 27-76 running, 28-85..88 queued with their
    prefixes loaded and the last admission's NO_TOKEN standing: base held S1
    (need 44k) and batch_is_full, so 28-86 waited 6 s for 28-85's decode."""
    dsv, sched, caps = env
    sched.running_batch.reqs = [_req("pdflip-27-76", 7270)]
    sched.running_batch.batch_is_full = True
    sched.waiting_queue = [_req("pdflip-28-%d" % i, 32835) for i in (85, 86, 87, 88)]
    with caplog.at_level(logging.INFO):
        dsv.runtime_tick(sched)
    st = getattr(sched, dsv.PHASE_ATTR)
    assert st.stage_tokens >= 7270 + 4 * 32835            # base: 65536
    assert sched.running_batch.batch_is_full is False     # base: True until a decode ends
    assert any(dsv.REOPEN_MARK in r.getMessage() for r in caplog.records)


def test_a_stage_that_holds_leaves_the_gate_alone(env):
    dsv, sched, _caps = env
    sched.running_batch.reqs = [_req("a", 1000)]
    sched.running_batch.batch_is_full = True
    sched.waiting_queue = [_req("q", 1000)]
    dsv.runtime_tick(sched)
    assert sched.running_batch.batch_is_full is True      # no grow: no new room, no re-ask
