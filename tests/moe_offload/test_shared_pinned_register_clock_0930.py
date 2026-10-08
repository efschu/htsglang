"""PIN-REGISTER (30.09., NF y3z/y4a): a slow D load was a handful of store files
whose registration took 4-17 s; the per-file clock names each one. Desk test:
a fake cudart stands in for the driver, no GPU."""
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from flliper.srt.layers.moe import shared_pinned as sp


class _FakeCudart:
    def __init__(self, delay_s=0.0, calls=None):
        self.delay_s, self.calls = delay_s, calls if calls is not None else []

    def cudaHostRegister(self, ptr, nbytes, flags):
        self.calls.append(("register", int(nbytes)))
        time.sleep(self.delay_s)
        return 0

    def cudaHostUnregister(self, ptr):
        self.calls.append(("unregister", 0))
        return 0


@pytest.fixture
def fake_cuda(monkeypatch):
    calls = []
    fake = _FakeCudart(calls=calls)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    monkeypatch.setattr(sp, "_REG_N", {"n": 0})
    return fake, calls


def test_every_registration_logs_its_file_and_clock(tmp_path, fake_cuda, caplog):
    with caplog.at_level(logging.INFO, logger=sp.__name__):
        sp.shared_pinned_empty(str(tmp_path / "L36-w13"), (8, 16), torch.float32, register=True)
    lines = [r.getMessage() for r in caplog.records if sp.REG_MARK in r.getMessage()]
    assert len(lines) == 1
    assert "file=L36-w13" in lines[0] and "bytes=512" in lines[0] and "register_ms=" in lines[0]
    assert "SLOW" not in lines[0]


def test_no_register_no_line(tmp_path, fake_cuda, caplog):
    with caplog.at_level(logging.INFO, logger=sp.__name__):
        sp.shared_pinned_empty(str(tmp_path / "L0-w2"), (2, 2), torch.float32, register=False)
    assert not [r for r in caplog.records if sp.REG_MARK in r.getMessage()]


def test_a_slow_registration_is_logged_past_the_cap(tmp_path, fake_cuda, caplog, monkeypatch):
    fake, _calls = fake_cuda
    monkeypatch.setattr(sp, "_REG_N", {"n": sp._REG_LOG_CAP})   # the cap is spent
    monkeypatch.setattr(sp, "_REG_SLOW_MS", 20.0)
    with caplog.at_level(logging.INFO, logger=sp.__name__):
        sp.shared_pinned_empty(str(tmp_path / "fast"), (2, 2), torch.float32, register=True)
        fake.delay_s = 0.05
        sp.shared_pinned_empty(str(tmp_path / "slow"), (2, 2), torch.float32, register=True)
    lines = [r.getMessage() for r in caplog.records if sp.REG_MARK in r.getMessage()]
    assert len(lines) == 1 and "file=slow" in lines[0] and lines[0].endswith("SLOW")
