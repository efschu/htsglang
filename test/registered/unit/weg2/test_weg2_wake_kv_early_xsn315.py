"""Wake-Parallel stage 2 (user 18.09.): the kv resume RPC rides with the
legs; the handler plans early / defer / late / done per call."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import wake_kv as wk  # noqa: E402


def test_plan():
    P = wk.wake_kv_plan
    assert P(kv_in_tags=True, weights_in_tags=False, fundable=True, deferred=False, epoch=7, epoch_done=None) == "early"
    assert P(kv_in_tags=True, weights_in_tags=False, fundable=False, deferred=False, epoch=7, epoch_done=None) == "defer"
    assert P(kv_in_tags=False, weights_in_tags=True, fundable=False, deferred=True, epoch=7, epoch_done=None) == "late"
    assert P(kv_in_tags=True, weights_in_tags=True, fundable=False, deferred=False, epoch=7, epoch_done=None) == "late"
    assert P(kv_in_tags=True, weights_in_tags=False, fundable=True, deferred=False, epoch=7, epoch_done=7) == "done"
    assert P(kv_in_tags=False, weights_in_tags=True, fundable=False, deferred=False, epoch=7, epoch_done=None) == "none"
    # xsn319: the old order -- kv-only AFTER the legs of this epoch is the whole block now
    assert P(kv_in_tags=True, weights_in_tags=False, fundable=True, deferred=False, epoch=7, epoch_done=None, weights_done=True) == "late"
    assert not wk.early_send_on({}) and wk.early_send_on({wk.EARLY_ENV: "1"})  # default OFF since xsn318


def test_front_sends_the_kv_resume_with_the_legs_first():
    from sglang.srt.weg2 import front as fr
    src = open(fr.__file__).read()
    i = src.index("_legs.insert(0, self.timed_rpc(D, \"/resume_memory_occupation\"")
    assert '{"tags": [KV_TAG], "epoch": flip_epoch}' in src[i:i + 200]
    assert "WEG2-WAKE-KV-EARLY rpc" in src
    # the post-legs kv call stays (idempotent per epoch in the handler)
    assert src.index('self._flip_stage = "wake-kv"') > i


def test_handler_plans_and_the_post_wake_pass_timer_is_armed():
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    from sglang.srt.managers import scheduler as sch
    src = open(wu.__file__).read()
    assert "_weg2_kv_deferred: bool = False" in src and "_weg2_kv_epoch_done: object = None" in src
    i = src.index("_plan = _wk_plan(")
    blk = src[i:i + 1600]
    for w in ('_plan == "early"', '_plan == "defer"', '_plan == "done"', "self._weg2_kv_deferred = True"):
        assert w in blk, w
    assert "(GPU_MEMORY_TYPE_KV_CACHE in tags or self._weg2_kv_deferred) and not _weg2_kv_done" in src
    assert "scheduler._weg2_post_wake_pass_n = 0" in src
    s2 = open(sch.__file__).read()
    assert "WEG2-POST-WAKE-PASS n=%d" in s2 and "self._weg2_post_wake_pass_log(batch)" in s2
