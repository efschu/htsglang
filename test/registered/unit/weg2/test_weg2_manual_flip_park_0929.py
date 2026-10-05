"""Flipzeit-Regression 29.09. (3/3), y3k-korr 20:42 / 20:52: POST /weg2/flip
from D closed admission and waited in the drain for a decode it never parked
(FLIP STALL 13,4 s, then 17,9 s -> DEADMAN_FLIP_STALL)."""

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

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------- 3. manual flip parks D
def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_y3k_korr_a_manual_flip_from_d_parks_before_the_flip():
    from sglang.srt.weg2 import front as fr

    calls = []

    async def _park(wait_s, immediate=None, cause="over-x"):
        calls.append(("park", cause))
        return "parked"

    async def _flip(src, dst):
        calls.append(("flip", src, dst))
        fake.awake = dst

    fake = types.SimpleNamespace(state="serving", awake="D", admit_d=True,
                                 _manual_flip_refusal=lambda: None, _wait_bound_park=_park,
                                 flip=_flip, state_dict=lambda: {})
    _run(fr.Front.handle_manual_flip(fake, None))
    assert calls[0] == ("park", fr.MANUAL_FLIP_PARK_CAUSE)   # base: the flip came first
    assert calls[1] == ("flip", "D", "P") and calls[2] == ("flip", "P", "D")


def test_a_manual_flip_from_p_parks_nothing():
    from sglang.srt.weg2 import front as fr

    calls = []

    async def _park(*a, **k):
        calls.append("park")

    async def _flip(src, dst):
        calls.append((src, dst))
        fake.awake = dst

    fake = types.SimpleNamespace(state="serving", awake="P", admit_d=True,
                                 _manual_flip_refusal=lambda: None, _wait_bound_park=_park,
                                 flip=_flip, state_dict=lambda: {})
    _run(fr.Front.handle_manual_flip(fake, None))
    assert "park" not in calls


def test_the_manual_park_names_itself_and_parks_the_running_decode(caplog):
    from sglang.srt.weg2 import front as fr

    sent = {}

    async def _rpc(group, path, body, timeout):
        sent.update(body)
        return 200, json.dumps({"parked": ["weg2-2-4"]})

    D = types.SimpleNamespace(outstanding={"weg2-2-4": object()})
    fake = types.SimpleNamespace(
        _park_attempt_epoch=None, epoch=38, groups={"D": D}, admit_d=True,
        counters=collections.Counter(), _ready_for_d=[], queue=[], _park_unsupported=False,
        d_wait_bound_s=60.0, rpc=_rpc, _d_parked={},
        # Later park bookkeeping the verdict path calls on self (no-ops here).
        _d_inflight_park=lambda depths: None, _seq_park=lambda marks: None,
        _park_handback=lambda rids: None,
        _ipc_dp_clock=lambda: types.SimpleNamespace(note_park=lambda *args: None))
    fake._flip_ledger = lambda g: [r for r in g.outstanding if r not in fake._d_parked]
    with caplog.at_level(logging.WARNING):
        verdict = _run(fr.Front._wait_bound_park(fake, None, cause=fr.MANUAL_FLIP_PARK_CAUSE))
    assert verdict == "parked"
    assert "weg2-2-4" in fake._d_parked
    assert sent["reason"] == fr.MANUAL_FLIP_PARK_CAUSE
    assert fake.counters["park_manual_flip"] == 1 and fake.counters["wait_bound_fired"] == 0
    assert any("WEG2 MANUAL-FLIP PARK" in r.getMessage() for r in caplog.records)
