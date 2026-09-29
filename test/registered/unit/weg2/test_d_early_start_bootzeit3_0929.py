"""BOOTZEIT 3 (29.09.): early D start -- the gate before D's weight load.

z30r3: D's init (spawn, imports, dist init) is 30 s and starts only after P's
READY + sleep + D plan. The gate lets D init run alongside P's load and holds
the LOAD until the launcher, after sleep(P), has compared D's planned budget
with the measured one: go iff planned <= measured on every card.
"""

import inspect
import json
import os
import tempfile
import threading
import time

import pytest

from sglang.srt.weg2 import d_early_start as des


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    des.reset_for_tests()
    monkeypatch.delenv(des.GATE_ENV, raising=False)
    yield
    des.reset_for_tests()


def test_no_env_is_a_noop():
    assert des.wait_gate_from_env() is None


def test_go_passes_once_per_process(monkeypatch, tmp_path):
    gate = str(tmp_path / "gate.json")
    monkeypatch.setenv(des.GATE_ENV, gate)
    threading.Timer(0.3, des.write_gate, args=(gate, des.VERDICT_GO, "fits")).start()
    waited = des.wait_gate_from_env()
    assert waited is not None and waited >= 0.2
    # the draft model's second load does not wait again
    assert des.wait_gate_from_env() is None


def test_refuse_raises_named(monkeypatch, tmp_path):
    gate = str(tmp_path / "gate.json")
    des.write_gate(gate, des.VERDICT_REFUSE, "card nvml1 short 420 MiB")
    monkeypatch.setenv(des.GATE_ENV, gate)
    with pytest.raises(des.DEarlyGateRefused, match="short 420"):
        des.wait_gate_from_env()


def test_timeout_is_named(tmp_path):
    t = [0.0]
    with pytest.raises(des.DEarlyGateTimeout):
        des.wait_gate(str(tmp_path / "never.json"), 1.0, poll_s=0.5,
                      clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))


def test_write_is_atomic_and_whole(tmp_path):
    gate = str(tmp_path / "gate.json")
    des.write_gate(gate, des.VERDICT_GO, "fits", {"a": 1}, {"a": 2})
    doc = json.load(open(gate))
    assert doc["verdict"] == "go" and doc["planned"] == {"a": 1} and doc["measured"] == {"a": 2}
    assert not [p for p in os.listdir(tmp_path) if ".tmp." in p]
    with pytest.raises(ValueError):
        des.write_gate(gate, "maybe", "")


def test_budget_verdict_z30r3_numbers():
    # expectation pass vs measured after sleep(P), z30r3 front.log 05:48:27 / 05:50:26
    planned = {"u1": 25600, "u0": 17184, "u2": 16976}
    measured = {"u1": 26400, "u0": 17728, "u2": 17936}
    ok, lines = des.budget_verdict(planned, measured, {"u1": "nvml1", "u0": "nvml0", "u2": "nvml2"})
    assert ok and any("slack +800" in l for l in lines)
    # one card over by one MiB refuses; an unmeasured card refuses
    ok2, lines2 = des.budget_verdict({**planned, "u1": 26401}, measured)
    assert not ok2 and any("short 1" in l for l in lines2)
    ok3, _ = des.budget_verdict(planned, {"u1": 26400, "u0": 17728})
    assert not ok3


def test_gate_sits_before_the_load_and_the_avail_reading():
    """The gate must hold before 'Load weight begin' and before_avail_memory:
    otherwise the load timer and D's mem accounting include P's awake bytes."""
    from sglang.srt.model_executor import model_runner as mr

    src = inspect.getsource(mr.ModelRunner.load_model)
    i_gate = src.index("wait_gate_from_env()")
    assert i_gate < src.index("before_avail_memory =")
    assert i_gate < src.index("Load weight begin")
