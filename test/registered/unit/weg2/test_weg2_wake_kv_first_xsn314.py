"""Wake-Parallel stage 1 (user 18.09.): the kv_cache pool is resumed BEFORE
the weight legs when the card can fund it now, so the held requests' arena
loads overlap the legs; otherwise the old order."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

MIB = 1 << 20


def _stand_in(free, need, floor, ref=20000):
    # ref = the legs' tightest card-free (MiB) at this rank's last wake (xsn323)
    return types.SimpleNamespace(
        _weg2_tag_bytes=lambda tag: need,
        _weg2_free_bytes=lambda: free,
        _weg2_corridor_floor_bytes=lambda: floor,
        _weg2_leg_min_free_mib=ref,
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


def test_the_legs_reserve_gates_the_early_pool_xsn323(monkeypatch):
    """xsn323 (5090): free 14804 funded the 6668 MiB pool before the legs, but
    the legs' tightest point of the old order was 9028 MiB at weights_3 --
    with the pool up 2360, below floor 1055 + staging 1309 + tag 2918: credit
    wait, the sleeper's tail never paused, W35 after 120 s."""
    monkeypatch.delenv("SGLANG_WEG2_WAKE_KV_FIRST", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_WAKE_KV_LEG_RESERVE_MIB", raising=False)
    f = wu.SchedulerWeightUpdaterManager._weg2_wake_kv_first_ok
    assert not f(_stand_in(14804 * MIB, 6668 * MIB, 1055 * MIB, ref=9028), ["kv_cache"])
    assert not f(_stand_in(14804 * MIB, 6668 * MIB, 1055 * MIB, ref=None), ["kv_cache"])   # first wake: old order
    # a 3080 P rank: free 7971, pool 1928, legs' tightest 7600 -> 5672 >= 1095+256+4352 = 5703? no -> LATE
    assert not f(_stand_in(7971 * MIB, 1928 * MIB, 1095 * MIB, ref=7600), ["kv_cache"])
    assert f(_stand_in(7971 * MIB, 1928 * MIB, 1095 * MIB, ref=7700), ["kv_cache"])
    monkeypatch.setenv("SGLANG_WEG2_WAKE_KV_LEG_RESERVE_MIB", "1000")
    assert f(_stand_in(14804 * MIB, 6668 * MIB, 1055 * MIB, ref=9028), ["kv_cache"])


def test_the_kv_block_is_one_closure_called_early_or_late():
    src = open(wu.__file__).read()
    d = src.index("        def _weg2_kv_resume_part():")  # stage 3: two halves + wrapper
    e = src.index("        _plan = _wk_plan(")  # stage 2: the plan replaces the bare gate
    l = src.index("        if (GPU_MEMORY_TYPE_KV_CACHE in tags or self._weg2_kv_deferred) and not _weg2_kv_done:")  # stage 2
    assert d < e < l
    body = src[d:e]
    assert "self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)" in body
    assert "WEG2-DORMANT cleared" in body and "_weg2_release_dormant_hold" in body
    # the early call precedes the weight legs' collect worker
    assert e < src.index('thread_name_prefix="weg2-wake-collect"', d)  # the collect worker of THIS handler
    assert src.count("            _weg2_kv_block()") == 1  # only the late site runs both halves
    # the EARLY path resumes the pool only -- never clears DORMANT before the weights (xsn315)
    k = src.index('if _plan == "early":')
    early = src[k:k + 500]
    assert "_weg2_kv_resume_part()" in early and "weg2_dormant = False" not in early
    rp = src.index("        def _weg2_kv_resume_part():"); cp = src.index("        def _weg2_kv_clear_part():")
    assert "weg2_dormant = False" not in src[rp:cp] and "weg2_dormant = False" in src[cp:cp + 2000]


def test_the_early_kv_only_call_defers_the_cuda_graph_resume():
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = open(wu.__file__).read()
    assert "_weg2_graph_deferred: bool = False" in src
    k = src.index('if _plan == "early":')
    assert "self._weg2_graph_deferred = True" in src[k:k + 900]
    g = src.index("        def _weg2_graph_block():")
    assert "self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)" in src[g:g + 600]
    assert "if GPU_MEMORY_TYPE_CUDA_GRAPH in tags and not self._weg2_graph_deferred:" in src
    l = src.index("if self._weg2_graph_deferred:\n                _weg2_graph_block()")
    assert l > src.index("thread_name_prefix=\"weg2-wake-collect\"", g)  # late: after the legs

