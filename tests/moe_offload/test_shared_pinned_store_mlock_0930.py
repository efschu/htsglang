"""STORE-MLOCK (30.09., NF y3z/y4a): with SGLANG_WEG2_STORE_MLOCK=1 every store
mapping is mlock'ed BEFORE its cudaHostRegister; a failed mlock is refused by
name, never swallowed; the switch is off by default. Desk test: fake cudart and
a fake mlock seam, no GPU (one test runs the real mlock on a 64 KiB mapping)."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import shared_pinned as sp


class _FakeCudart:
    def __init__(self, calls):
        self.calls = calls

    def cudaHostRegister(self, ptr, nbytes, flags):
        self.calls.append(("register", int(ptr), int(nbytes)))
        return 0

    def cudaHostUnregister(self, ptr):
        return 0


@pytest.fixture
def seams(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FakeCudart(calls))

    def fake_mlock(ptr, nbytes):
        calls.append(("mlock", int(ptr), int(nbytes)))
        return 0

    monkeypatch.setattr(sp, "_libc_mlock", fake_mlock)
    return calls


def test_the_switch_is_off_by_default(tmp_path, seams):
    with envs.SGLANG_WEG2_STORE_MLOCK.override(None):
        envs.SGLANG_WEG2_STORE_MLOCK.clear()
        assert envs.SGLANG_WEG2_STORE_MLOCK.get() is False
        sp.shared_pinned_empty(str(tmp_path / "off"), (4, 4), torch.float32, register=True)
    assert [c[0] for c in seams] == ["register"]


def test_mlock_runs_before_the_registration_on_the_same_bytes(tmp_path, seams):
    with envs.SGLANG_WEG2_STORE_MLOCK.override(True):
        t, _ = sp.shared_pinned_empty(str(tmp_path / "L36-w13"), (8, 16), torch.float32,
                                      register=True)
    assert [c[0] for c in seams] == ["mlock", "register"]
    (_, p_ml, n_ml), (_, p_reg, n_reg) = seams
    assert p_ml == p_reg == t.data_ptr() and n_ml == n_reg == 8 * 16 * 4


def test_a_failed_mlock_is_refused_by_name(tmp_path, seams, monkeypatch):
    monkeypatch.setattr(sp, "_libc_mlock", lambda ptr, nbytes: 12)   # ENOMEM
    with envs.SGLANG_WEG2_STORE_MLOCK.override(True):
        with pytest.raises(sp.Weg2StoreMlockRefused) as ei:
            sp.shared_pinned_empty(str(tmp_path / "L0-w2"), (4, 4), torch.float32,
                                   register=True)
    assert "WEG2-STORE-MLOCK REFUSED" in str(ei.value) and "errno 12" in str(ei.value)
    assert not [c for c in seams if c[0] == "register"]   # no pin after a refused lock


def test_the_real_mlock_locks_the_mapping(tmp_path, monkeypatch):
    """The real syscall on a 64 KiB store: VmLck of this process grows by it."""
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FakeCudart([]))

    def vmlck_kb():
        for line in open("/proc/self/status"):
            if line.startswith("VmLck:"):
                return int(line.split()[1])
        return 0

    before = vmlck_kb()
    with envs.SGLANG_WEG2_STORE_MLOCK.override(True):
        try:
            t, _ = sp.shared_pinned_empty(str(tmp_path / "real"), (16384,), torch.float32,
                                          register=False)
        except sp.Weg2StoreMlockRefused as exc:   # a desk without memlock budget
            pytest.skip(str(exc))
    assert vmlck_kb() - before >= 64
    del t
