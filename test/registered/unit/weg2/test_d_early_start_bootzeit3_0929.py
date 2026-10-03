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


# --- stage 0 (27B review of cb98c3d94a): no D context before P is sized ---


def test_stage0_noop_without_env(monkeypatch):
    monkeypatch.delenv(des.STAGE0_ENV, raising=False)
    assert des.wait_stage0_from_env() is None


def test_stage0_refuse_raises_named(monkeypatch, tmp_path):
    gate = str(tmp_path / "s0.json")
    des.write_gate(gate, des.VERDICT_REFUSE, "nvml1: D held 482 MiB before P was sized")
    monkeypatch.setenv(des.STAGE0_ENV, gate)
    with pytest.raises(des.DEarlyGateRefused, match="482 MiB"):
        des.wait_stage0_from_env()


def test_p_sized_record_roundtrip(monkeypatch, tmp_path):
    monkeypatch.delenv(des.P_SIZED_DIR_ENV, raising=False)
    assert des.note_p_memory_sized(0, 0, 16.68) is None  # off without the env
    monkeypatch.setenv(des.P_SIZED_DIR_ENV, str(tmp_path))
    des.note_p_memory_sized(2, 0, 9.06)
    des.note_p_memory_sized(0, 0, 16.68)
    recs = des.read_p_sized(str(tmp_path))
    assert [(r["pp_rank"], r["used_by_me_mib"]) for r in recs] == [(0, 17080), (2, 9277)]


def test_stage0_verdict_waits_then_goes_and_refuses_on_taint():
    recs = [{"pp_rank": k, "tp_rank": 0, "used_by_me_mib": 1000 + k} for k in range(3)]
    v, _ = des.stage0_verdict(recs[:2], 3, {"u1": 0})
    assert v is None  # PP2 not sized yet
    v, lines = des.stage0_verdict(recs, 3, {"u1": 0, "u0": 0})
    assert v is True and "D held 0 MiB" in lines[-1]
    # a D context seen before P was sized taints P's used_by_me: refuse, even if P is done
    v, lines = des.stage0_verdict(recs, 3, {"u1": 888}, {"u1": "nvml1"})
    assert v is False and any("nvml1: D held 888 MiB" in l for l in lines)


# --- free-read journal (27B follow-up): P's reads from stage 0 to first wake ---


def _p_reader_under_test():
    import torch

    return torch.cuda.mem_get_info(0)


def test_free_read_journal_records_the_call_site_and_restores(monkeypatch, tmp_path):
    import torch

    fake = lambda *a, **k: (512 << 20, 20480 << 20)  # noqa: E731
    monkeypatch.setattr(torch.cuda, "mem_get_info", fake)
    monkeypatch.delenv(des.FREE_READ_JOURNAL_ENV, raising=False)
    assert des.start_free_read_journal(0, 0) is None  # off without the env
    monkeypatch.setenv(des.FREE_READ_JOURNAL_ENV, str(tmp_path))
    path = des.start_free_read_journal(1, 0)
    assert _p_reader_under_test() == (512 << 20, 20480 << 20)  # value unchanged
    des.stop_free_read_journal()
    assert torch.cuda.mem_get_info is fake  # restored
    rows = des.read_free_journal(path)
    assert len(rows) == 1 and rows[0]["free_mib"] == 512 and rows[0]["total_mib"] == 20480
    assert rows[0]["site"].endswith(":_p_reader_under_test")
    _p_reader_under_test()  # after stop: not journaled
    assert len(des.read_free_journal(path)) == 1


def test_free_read_diff_flags_the_site_that_sees_d():
    early = [{"site": "g.py:9:spendable", "via": "", "free_mib": 900},
             {"site": "c.py:1:census", "via": "", "free_mib": 4000}]
    serial = [{"site": "g.py:9:spendable", "via": "", "free_mib": 1382},
              {"site": "c.py:1:census", "via": "", "free_mib": 4010}]
    lines = des.free_read_diff(early, serial)
    assert any(l.startswith("DEVIATES g.py:9:spendable") and "delta -482" in l for l in lines)
    assert any(l.startswith("same c.py:1:census") for l in lines)
    only = des.free_read_diff(early[:1], [])
    assert only[0].startswith("DEVIATES") and "one boot only" in only[0]


def test_journal_window_is_stage0_to_first_wake():
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    init = inspect.getsource(sch.Scheduler.init_model_worker)
    assert init.index("note_p_memory_sized(") < init.index("start_free_read_journal(")
    resume = inspect.getsource(wu)
    i = resume.index("def resume_memory_occupation(")
    assert "stop_free_read_journal()" in resume[i:i + 400]


def test_stage0_sits_before_the_first_cuda_call_and_p_reports_after_sizing():
    from sglang.srt.managers import scheduler as sch

    run = inspect.getsource(sch.run_scheduler_process)
    i = run.index("wait_stage0_from_env()")
    assert i < run.index("install_triton_loader_window()") < run.index("Scheduler(")
    init = inspect.getsource(sch.Scheduler.init_model_worker)
    assert init.index("note_post_capture_leftover(") < init.index("note_p_memory_sized(")
