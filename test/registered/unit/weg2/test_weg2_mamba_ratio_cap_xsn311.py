"""Nutzer 18.09.: Mamba state-pool ratio capped by SGLANG_WEG2_MAMBA_RATIO,
never below the per-request hard floor."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx  # noqa: E402
from sglang.srt.mem_cache import mamba_pool_floor as mpf  # noqa: E402


def _runner():
    sa = types.SimpleNamespace(
        disable_radix_cache=False, disable_overlap_schedule=False,
        enable_mamba_extra_buffer=lambda: True, enable_mamba_extra_buffer_lazy=lambda: False,
    )
    return types.SimpleNamespace(server_args=sa)


def test_cap_applies_above_the_floor_and_never_below(monkeypatch):
    monkeypatch.setattr(mpf, "mamba_slots_per_running_req", lambda sa: 5)
    monkeypatch.setattr(mpf, "mamba_hard_floor", lambda sa, n: 2 * n)
    r = _runner()
    monkeypatch.delenv("SGLANG_WEG2_MAMBA_RATIO", raising=False)
    assert mx.ModelRunnerKVCacheMixin._calculate_mamba_ratio(r) == 5          # 3 + overlap 2, floor 5
    monkeypatch.setenv("SGLANG_WEG2_MAMBA_RATIO", "3")
    assert mx.ModelRunnerKVCacheMixin._calculate_mamba_ratio(r) == 3
    monkeypatch.setenv("SGLANG_WEG2_MAMBA_RATIO", "1")               # below the floor (2): floor wins
    assert mx.ModelRunnerKVCacheMixin._calculate_mamba_ratio(r) == 2
    monkeypatch.setenv("SGLANG_WEG2_MAMBA_RATIO", "9")               # above: no change
    assert mx.ModelRunnerKVCacheMixin._calculate_mamba_ratio(r) == 5
