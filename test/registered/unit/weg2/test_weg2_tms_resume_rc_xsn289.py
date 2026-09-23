"""weg2xsn269/289: a refused VMM remap RAISES instead of exit(1).

Hermetic (CUDA_VISIBLE_DEVICES=""): the rc function is injected; the hook's
C++ side is pinned by a source ratchet (symbol + rollback + return 0).
"""
import os
import pathlib

import pytest

from sglang.srt.managers import weg2_memory_saver as wms

CSRC = pathlib.Path(wms.__file__).resolve().parents[1] / "weg2" / "tms_csrc"


class _Adapter:
    def __init__(self):
        self.resumed = []

    def resume(self, tag):
        self.resumed.append(tag)


def test_rc_zero_returns_and_skips_adapter():
    a = _Adapter()
    calls = []
    rc = wms.weg2_tms_resume(a, "weights_k", rc_fn=lambda t: calls.append(t) or 0)
    assert rc == 0 and calls == [b"weights_k"] and a.resumed == []


def test_rc_nonzero_raises_with_tag_and_code():
    a = _Adapter()
    with pytest.raises(RuntimeError) as ei:
        wms.weg2_tms_resume(a, "weights_k", rc_fn=lambda t: 2)
    msg = str(ei.value)
    assert "WEG2-TMS-RESUME REFUSED" in msg and "tag=weights_k" in msg and "rc=2" in msg
    assert a.resumed == []


def test_without_symbol_falls_back_to_adapter(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_TMS_PRELOAD_SO", raising=False)
    monkeypatch.setattr(wms, "_TMS_RC_HANDLE", None)
    monkeypatch.setattr(wms, "_TMS_RC_MISSING", False)
    a = _Adapter()
    assert wms.weg2_tms_resume(a, "weights_k") == 0
    assert a.resumed == ["weights_k"]
    assert wms._TMS_RC_MISSING is True


def test_missing_so_path_is_remembered_not_retried(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_TMS_PRELOAD_SO", "/nonexistent/hook.so")
    monkeypatch.setattr(wms, "_TMS_RC_HANDLE", None)
    monkeypatch.setattr(wms, "_TMS_RC_MISSING", False)
    assert wms._tms_resume_rc_symbol() is None
    assert wms._TMS_RC_MISSING is True


def test_csrc_ratchet_rc_symbol_rollback_and_return():
    core = (CSRC / "core.cpp").read_text()
    entry = (CSRC / "entrypoint.cpp").read_text()
    utils = (CSRC / "utils.h").read_text()
    assert "int TorchMemorySaver::resume(const std::string& tag)" in core
    assert "cu_mem_create_rc(&newAllocHandle" in core
    assert "WEG2-TMS-RESUME REFUSED" in core and "md.state = AllocationState::PAUSED" in core
    assert "int tms_resume_rc(const char* tag)" in entry
    assert "static CUresult cu_mem_create_rc(" in utils
    # the old exit(1) path in pass 1 is gone: cu_mem_create (void) no longer used there
    pass1 = core.split("--- pass 1", 1)[1].split("weg2_map_t1 =", 1)[0]
    assert "CUDAUtils::cu_mem_create(&newAllocHandle" not in pass1
