# SPDX-License-Identifier: Apache-2.0
"""#1720 SEATS-STALL-OFF (deskq/done/1720 F1): SGLANG_WEG2_DUAL_SEATS_STALL_OFF.

On dual group P (--max-running-requests 1) the seat gate declines with ``running=empty`` while the one
request slot flies in the PP ring; the ``gate=seats`` IntakeStallWatch observation is then a false positive.
With the switch on AND dual P the observation is skipped; otherwise (switch off, or the flip path without
the dual layout) it is called exactly as before.

The gate statement is cut out of the REAL ``Scheduler._get_new_batch_prefill_raw`` (AST) and executed
against a stub ``self``, so the test drives the production code, not a copy.
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.weg2 import dual_card_stall as DCS  # noqa: E402

SW = "SGLANG_WEG2_DUAL_SEATS_STALL_OFF"
DUAL_ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}


def _seat_gate_stmt():
    """The innermost ``if`` of the raw prefill builder that issues the gate=seats observation."""
    src = inspect.cleandoc(inspect.getsource(Scheduler._get_new_batch_prefill_raw))
    tree = ast.parse(src)
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            seg = ast.get_source_segment(src, node) or ""
            if "gate=seats" in seg and "_weg2_intake_stall_observe" in seg:
                if best is None or len(seg) < len(ast.get_source_segment(src, best)):
                    best = node
    assert best is not None, "gate=seats observation not found in _get_new_batch_prefill_raw"
    return ast.Module(body=[best], type_ignores=[])


def _run_gate(monkeypatch, *, switch: bool, dual: bool):
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", SW):
        monkeypatch.delenv(k, raising=False)
    if switch:
        monkeypatch.setenv(SW, "1")
    if dual:
        for k, v in DUAL_ENV.items():
            monkeypatch.setenv(k, v)
    calls = []
    stub = SimpleNamespace(
        waiting_queue=["head", "x"],
        req_to_token_pool=SimpleNamespace(available_size=lambda: 0),
        get_num_allocatable_reqs=lambda n: 0,
        _weg2_intake_stall_observe=lambda *a, **k: calls.append((a, k)),
    )
    # the production method when it exists; a False stub before the fix (so the red run fails on the
    # assertion, not on an AttributeError)
    real = getattr(Scheduler, "_weg2_seats_stall_off", None)
    stub._weg2_seats_stall_off = MethodType(real, stub) if real else (lambda: False)
    running_batch = SimpleNamespace(is_empty=lambda: True)
    exec(compile(_seat_gate_stmt(), "<seat-gate>", "exec"), {"self": stub, "running_batch": running_batch})
    return calls


def test_a_switch_off_observe_called(monkeypatch):
    # unchanged behaviour, with and without the dual layout
    assert len(_run_gate(monkeypatch, switch=False, dual=True)) == 1
    calls = _run_gate(monkeypatch, switch=False, dual=False)
    assert len(calls) == 1
    assert "gate=seats" in calls[0][1]["note"]


def test_b_switch_on_dual_p_observe_skipped(monkeypatch):
    assert _run_gate(monkeypatch, switch=True, dual=True) == []


def test_c_switch_on_not_dual_p_observe_still_called(monkeypatch):
    # flip path (no dual layout) unchanged
    assert len(_run_gate(monkeypatch, switch=True, dual=False)) == 1
    # dual layout on the D group is not dual P either
    for k, v in (("SGLANG_WEG2_DUAL_LAYOUT", "1"), ("SGLANG_WEG2_GROUP", "D")):
        monkeypatch.setenv(k, v)
    assert DCS.seats_stall_off() is False


def test_helper_reads_env_only(monkeypatch):
    monkeypatch.setenv(SW, "1")
    assert DCS.seats_stall_off(DUAL_ENV) is True
    assert DCS.seats_stall_off({}) is False
    monkeypatch.delenv(SW)
    assert DCS.seats_stall_off(DUAL_ENV) is False


def test_marker_logged_once_only_when_active(monkeypatch, caplog):
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", SW):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(DCS, "_SEATS_STALL_OFF_LOGGED", False)
    log = logging.getLogger("t1720")
    with caplog.at_level(logging.INFO, logger="t1720"):
        assert DCS.log_seats_stall_off_once(log) is False  # switch off: silent
        monkeypatch.setenv(SW, "1")
        assert DCS.log_seats_stall_off_once(log) is False  # on but not dual P: silent
        for k, v in DUAL_ENV.items():
            monkeypatch.setenv(k, v)
        assert DCS.log_seats_stall_off_once(log) is True
        assert DCS.log_seats_stall_off_once(log) is False  # once
    assert sum("#1720 SEATS-STALL-OFF" in r.getMessage() for r in caplog.records) == 1
