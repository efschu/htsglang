# L15-13b (AP 1001): the D WAKE fit check must count only the bytes the
# resume will actually have to MAP, not the full plan.
#
# With TMS keep spans (L15-13a, b172a12a6a) a paused kv_cache allocation keeps
# its held rows MAPPED, so tms_tag_bytes(tag) -- the FULL plan, the bytes the
# next resume covers INCLUDING the kept ones -- double-counts the held rows
# against the card's free budget at wake. The kept bytes are already resident,
# so the old check demanded them twice and produced a false "cannot fit" at
# every D wake with a hold. The resume only has to map need = plan - mapped_now.
#
# This test pins:
#   * kv_resume_need_bytes (the pure subtraction),
#   * the adapter's tag_mapped_bytes symbol read (real + no-op + base contract),
#   * the fit check now running on need (not plan),
#   * the wiring of both wake sites (source contains kv_resume_need_bytes).
#
# Plain pytest functions only: CustomTestCase.retry() would hide a failure.
#
# Run:
#   cd /spinning/wt-l15-0930 && CUDA_VISIBLE_DEVICES="" \
#   PYTHONPATH=/spinning/wt-l15-0930/python \
#   nice -n 19 ionice -c3 \
#   /spinning/htsglang-gpu/.venv/bin/python -m pytest \
#   test/registered/unit/weg2/test_weg2_l15_wake_need_1001.py -q -p no:cacheprovider

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.weg2.wake_kv import (
    kv_resume_fit_refusal,
    kv_resume_need_bytes,
)


# ---------------------------------------------------------------------------
# kv_resume_need_bytes: the pure subtraction
# ---------------------------------------------------------------------------
def test_need_absent_plan_is_absent_need():
    # No plan, no verdict: a None plan passes straight through as None.
    assert kv_resume_need_bytes(None, 5) is None


def test_need_absent_mapped_is_full_plan():
    # The probe is absent (None) -> the whole plan must be mapped (old behavior).
    assert kv_resume_need_bytes(100, None) == 100


def test_need_zero_mapped_is_full_plan():
    # Hold off / nothing kept -> mapped 0 -> need == plan (byte-identical).
    assert kv_resume_need_bytes(100, 0) == 100


def test_need_subtracts_resident_mapped_bytes():
    assert kv_resume_need_bytes(100, 40) == 60


def test_need_clamped_at_zero_never_unmaps():
    # mapped > plan (e.g. a stale higher reading): the resume never unmaps.
    assert kv_resume_need_bytes(100, 140) == 0


# ---------------------------------------------------------------------------
# The fit check runs on need, not plan -- this removes the over-demand
# ---------------------------------------------------------------------------
def test_fit_refusal_gone_once_need_excludes_kept_spans():
    # need = plan - mapped = 100 - 60 = 40; free 50 >= 40 -> physically possible.
    assert kv_resume_fit_refusal(50, kv_resume_need_bytes(100, 60), 0) is None
    # The OLD check against the full plan: free 50 < 100 -> a (false) refusal.
    assert kv_resume_fit_refusal(50, 100, 0) is not None


# ---------------------------------------------------------------------------
# The adapter's tag_mapped_bytes: bytes physically mapped NOW for a tag
# ---------------------------------------------------------------------------
class _FakeSymbol:
    # A stand-in for the C entry point: settable restype/argtypes, callable.
    restype = None
    argtypes = None

    def __init__(self, value):
        self._value = value

    def __call__(self, tag_bytes):
        return self._value


def test_real_adapter_tag_mapped_bytes_none_when_symbol_absent(monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tmsa

    adapter = tmsa._TorchMemorySaverAdapterReal()
    monkeypatch.setattr(tmsa, "_weg2_ring_symbol", lambda name: None)
    assert adapter.tag_mapped_bytes("kv_cache") is None


def test_real_adapter_tag_mapped_bytes_int_when_symbol_present(monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tmsa

    adapter = tmsa._TorchMemorySaverAdapterReal()
    monkeypatch.setattr(
        tmsa, "_weg2_ring_symbol", lambda name: _FakeSymbol(40)
    )
    assert adapter.tag_mapped_bytes("kv_cache") == 40


def test_noop_adapter_tag_mapped_bytes_none():
    from sglang.srt.utils import torch_memory_saver_adapter as tmsa

    adapter = tmsa._TorchMemorySaverAdapterNoop()
    assert adapter.tag_mapped_bytes("kv_cache") is None


def test_base_adapter_tag_mapped_bytes_contract():
    # The base class declares the method and raises NotImplementedError, like
    # tag_bytes; the real / no-op subclasses override it.
    from sglang.srt.utils import torch_memory_saver_adapter as tmsa

    with pytest.raises(NotImplementedError):
        tmsa.TorchMemorySaverAdapter.tag_mapped_bytes(object(), "kv_cache")


# ---------------------------------------------------------------------------
# Source wiring: both wake sites feed the fit check with need, not plan
# ---------------------------------------------------------------------------
def test_wake_sites_reference_kv_resume_need_bytes():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[4]
    for rel in (
        "python/sglang/srt/managers/scheduler_components/weight_updater.py",
        "python/sglang/srt/flip_form_a_sleep_hook.py",
    ):
        path = root / rel
        assert path.is_file(), f"missing source file: {path}"
        text = path.read_text()
        assert "kv_resume_need_bytes" in text, (
            f"{rel} does not wire kv_resume_need_bytes into the wake fit check"
        )
