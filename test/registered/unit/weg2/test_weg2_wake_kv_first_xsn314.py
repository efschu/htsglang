"""Wake-Parallel stage 1 (user 18.09.): the kv_cache pool is resumed BEFORE
the weight legs when the card can fund it now, so the held requests' arena
loads overlap the legs; otherwise the old order."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

MIB = 1 << 20


def _stand_in(free, need, floor):
    return types.SimpleNamespace(
        _weg2_tag_bytes=lambda tag: need,
        _weg2_free_bytes=lambda: free,
        _weg2_corridor_floor_bytes=lambda: floor,
    )


def test_early_only_when_free_minus_floor_covers_kv_plus_margin(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_WAKE_KV_FIRST", raising=False)
    f = wu.SchedulerWeightUpdaterManager._weg2_wake_kv_first_ok
    assert f(_stand_in(12000 * MIB, 9400 * MIB, 767 * MIB), ["kv_cache"])        # 12000-767-256 >= 9400
    assert not f(_stand_in(10000 * MIB, 9400 * MIB, 767 * MIB), ["kv_cache"])    # too tight: old order
    assert not f(_stand_in(None, 9400 * MIB, 0), ["kv_cache"])                   # no probe: old order
    assert not f(_stand_in(12000 * MIB, 0, 0), ["kv_cache"])                     # no kv bytes known
    monkeypatch.setenv("SGLANG_WEG2_WAKE_KV_FIRST", "0")
    assert not f(_stand_in(12000 * MIB, 9400 * MIB, 767 * MIB), ["kv_cache"])


def test_the_kv_block_is_one_closure_called_early_or_late():
    src = open(wu.__file__).read()
    d = src.index("        def _weg2_kv_block():")
    e = src.index("        if GPU_MEMORY_TYPE_KV_CACHE in tags and self._weg2_wake_kv_first_ok(tags):")
    l = src.index("        if GPU_MEMORY_TYPE_KV_CACHE in tags and not _weg2_kv_done:")
    assert d < e < l
    body = src[d:e]
    assert "self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)" in body
    assert "WEG2-DORMANT cleared" in body and "_weg2_release_dormant_hold" in body
    # the early call precedes the weight legs' collect worker
    assert e < src.index('thread_name_prefix="weg2-wake-collect"', d)  # the collect worker of THIS handler
    assert src.count("            _weg2_kv_block()") == 2  # early + late call, plus the def
