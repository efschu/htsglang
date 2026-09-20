"""#49 (20.09., fn8c12): PDL stays off on capability 12.x unless opted in."""

import types

from sglang.jit_kernel import utils as ju


def _arch(major, minor=0):
    return types.SimpleNamespace(major=major, minor=minor, target_name=f"sm_{major}{minor}")


def _pdl():
    # the memo keys by target_name; read the undecorated decision so each case is its own
    return ju.is_arch_support_pdl.__wrapped__()


def test_pdl_is_off_on_sm12_by_default(monkeypatch):
    monkeypatch.setattr(ju, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(ju, "is_musa_runtime", lambda: False)
    monkeypatch.delenv("SGLANG_PDL_ON_SM12", raising=False)
    ju._PDL_SM12_LOGGED["done"] = False
    monkeypatch.setattr(ju, "get_jit_cuda_arch", lambda: _arch(12, 0))
    assert _pdl() is False
    monkeypatch.setattr(ju, "get_jit_cuda_arch", lambda: _arch(12, 1))
    assert _pdl() is False


def test_pdl_stays_on_for_hopper_and_datacenter_blackwell(monkeypatch):
    monkeypatch.setattr(ju, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(ju, "is_musa_runtime", lambda: False)
    for major in (9, 10):
        monkeypatch.setattr(ju, "get_jit_cuda_arch", lambda m=major: _arch(m, 0))
        assert _pdl() is True
    monkeypatch.setattr(ju, "get_jit_cuda_arch", lambda: _arch(8, 6))
    assert _pdl() is False


def test_the_opt_in_restores_pdl_on_sm12(monkeypatch):
    monkeypatch.setattr(ju, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(ju, "is_musa_runtime", lambda: False)
    monkeypatch.setattr(ju, "get_jit_cuda_arch", lambda: _arch(12, 0))
    monkeypatch.setenv("SGLANG_PDL_ON_SM12", "1")
    assert _pdl() is True
    assert ju.pdl_on_sm12_opted_in({}) is False
