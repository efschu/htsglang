# SPDX-License-Identifier: Apache-2.0
"""DUAL P SLEEP under --dual-share (user rule: P sleeps and parks its weights in host RAM).

gmps9 (dkr27bnvfp4dual1mbar1fs10012051): with the weights not resident, the shared part
loaded INSIDE the private weights tag pools; the union bind freed its copies there and a
private pool never returns them ('card free 3.35 -> 3.35 GiB'; PP0 tag pools after load
'inactive_gib' 0.13 + 0.56 + 4 x 0.55 + 0.36 = 3.25 GiB), P-PP0's KV budget was refused.

DANGER DIRECTIONS guarded here, MUTANT per guard (asserted in-suite):
* step 1: the shared part loads and binds inside transient_load_scope (outside the tag
  pool, saver tracking off); a nested back_into_tag_pool stays transient unless forced;
  the bound part's buffers are re-homed under the weights tag; a bind that keeps own
  tensors in the scope is a named stop (W-DUAL-SHARED-KEPT);
* hard riegel: live blocks left in a load pool after load = untagged live bytes ->
  W-DUAL-P-UNTAGGED on a sleeping dual P rank (no-op elsewhere);
* step 2: the summed empty segments of the weights tag pools after load are capped
  (W-DUAL-P-POOL-SLACK).
"""
from __future__ import annotations

import contextlib
import inspect
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.managers import weg2_memory_saver as S
from sglang.srt.model_executor import dual_stage_hull as H
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _exec_mutant(mod, fn, fixed, back):
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant: " + fixed
    ns = dict(vars(mod))
    exec(compile(src.replace(fixed, back), mod.__file__, "exec"), ns)
    return ns[fn.__name__]


# -- step 1: the transient scope ------------------------------------------------------


def test_transient_scope_is_a_noop_without_a_tag_pool(monkeypatch):
    monkeypatch.setattr(S, "_ACTIVE_TAG_POOL", None)
    with S.transient_load_scope("x") as t:
        assert t is False and not S._TRANSIENT_ONLY


def test_inside_the_transient_scope_back_into_tag_pool_stays_out_unless_forced(monkeypatch):
    @contextlib.contextmanager
    def fake_outside(reason="", into="load"):
        yield True

    monkeypatch.setattr(S, "outside_tag_pool", fake_outside)
    monkeypatch.setattr(S, "_STEPPED_OUT_POOLS", [types.SimpleNamespace(id=7)])
    calls = []
    import torch.cuda.memory as tcm

    monkeypatch.setattr(tcm, "_cuda_beginAllocateCurrentThreadToPool", lambda d, p: calls.append(("in", p)),
                        raising=False)
    monkeypatch.setattr(tcm, "_cuda_endAllocateToPool", lambda d, p: calls.append(("out", p)), raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with S.transient_load_scope("dual-share-shared-part") as t:
        assert t is True and S._TRANSIENT_ONLY == ["dual-share-shared-part"]
        with S.back_into_tag_pool() as back:
            assert back is False, "a repack survivor of the shared part stays transient"
        assert calls == []
        with S.back_into_tag_pool(force=True) as back:
            assert back is True
        assert calls == [("in", 7), ("out", 7)]
    assert not S._TRANSIENT_ONLY


def test_the_ignore_transient_mutant_turns_the_scope_test_red(monkeypatch):
    m = _exec_mutant(S, S.back_into_tag_pool, "if not _STEPPED_OUT_POOLS or (_TRANSIENT_ONLY and not force):",
                     "if not _STEPPED_OUT_POOLS:")
    monkeypatch.setattr(S, "back_into_tag_pool", contextlib.contextmanager(m.__wrapped__)
                        if hasattr(m, "__wrapped__") else m)
    with pytest.raises(AssertionError):
        test_inside_the_transient_scope_back_into_tag_pool_stays_out_unless_forced(monkeypatch)


def test_the_shared_part_loads_and_binds_inside_the_transient_scope():
    src = inspect.getsource(H.build_dual_stage_model)
    i = src.index('with transient_load_scope("dual-share-shared-part") as transient:')
    blk = src[i:i + 300]
    assert "parts[local] = _load_lane_part(" in blk and "_bind_shared_part(runner, parts[local], bind, transient=transient)" in blk


class _Part(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.register_buffer("cos", torch.ones(3))


def _bind(monkeypatch, kept):
    import sglang.srt.weg2.union_arena_bind as UB

    monkeypatch.setattr(UB, "bind_image", lambda *a, **kw: (100, kept))
    monkeypatch.setattr(S, "back_into_tag_pool", lambda force=False: contextlib.nullcontext(True))
    runner = types.SimpleNamespace()
    part = _Part()
    old = part.cos
    t = H._BindTarget("/u", "card", 0)
    H._bind_shared_part(runner, part, t, transient=True)
    return runner, part, old


def test_a_bind_that_keeps_own_tensors_in_the_scope_is_a_named_stop(monkeypatch):
    with pytest.raises(H.DualSharedPartKept, match="W-DUAL-SHARED-KEPT"):
        _bind(monkeypatch, kept=4096)


def test_the_bound_parts_buffers_are_rehomed(monkeypatch):
    runner, part, old = _bind(monkeypatch, kept=0)
    assert part.cos is not old and torch.equal(part.cos, old)
    assert runner.dual_share_rehomed_bytes == 3 * 4 and runner.dual_share_bound


def test_the_no_kept_check_mutant_turns_the_kept_test_red(monkeypatch):
    m = _exec_mutant(H, H._bind_shared_part, "if transient and kept > 0:", "if False:")
    monkeypatch.setattr(H, "_bind_shared_part", m)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        test_a_bind_that_keeps_own_tensors_in_the_scope_is_a_named_stop(monkeypatch)


# -- hard riegel: untagged live bytes ---------------------------------------------------


def test_untagged_live_bytes_are_a_hard_stop():
    S.assert_no_untagged_live({})
    S.assert_no_untagged_live({"load": 0.0})
    with pytest.raises(S.Weg2DualPUntagged, match="W-DUAL-P-UNTAGGED: 12.0 MiB"):
        S.assert_no_untagged_live({"load": 12.0})


def test_the_release_records_what_a_kept_pool_still_holds(monkeypatch):
    pool = types.SimpleNamespace(id=1)
    monkeypatch.setattr(S, "_LOAD_TRANSIENT_POOL", pool)
    monkeypatch.setattr(S, "_ACTIVE_TAG_POOL", None)
    monkeypatch.setattr(S, "_STEPPED_OUT_POOLS", [])
    monkeypatch.setattr(S, "_tms_cdll_in_region", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(S, "_pool_has_live_blocks", lambda p, r: True)
    monkeypatch.setattr(S, "_pool_live_mib", lambda p: 37.5)
    monkeypatch.setattr(S, "_LAST_KEPT_LIVE_MIB", {})
    monkeypatch.setattr(S, "_KEPT_TRANSIENT_POOLS", [])
    assert S._release_one_load_pool("load", "after-load") == 0.0
    assert S._LAST_KEPT_LIVE_MIB == {"load": 37.5}
    with pytest.raises(S.Weg2DualPUntagged):
        S.assert_no_untagged_live()


def _tracked(monkeypatch, env, kept):
    for k in ("SGLANG_WEG2_DUAL_SHARE", "SGLANG_WEG2_GROUP", S.WEIGHTS_RESIDENT_ENV):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(S, "_LAST_KEPT_LIVE_MIB", dict(kept))
    monkeypatch.setattr(S, "weights_pool_slack_mib", lambda *a, **kw: {"weights": 10.0})
    return H.assert_dual_p_sleep_tracked(types.SimpleNamespace(dual_share_rehomed_bytes=0))


SLEEPING_P = {"SGLANG_WEG2_DUAL_SHARE": "1", "SGLANG_WEG2_GROUP": "P", S.WEIGHTS_RESIDENT_ENV: "0"}


def test_the_riegel_runs_only_on_a_sleeping_dual_share_p_rank(monkeypatch):
    assert _tracked(monkeypatch, SLEEPING_P, {}) is True
    with pytest.raises(S.Weg2DualPUntagged):
        _tracked(monkeypatch, SLEEPING_P, {"load": 5.0})
    for env in ({**SLEEPING_P, S.WEIGHTS_RESIDENT_ENV: "1"}, {"SGLANG_WEG2_GROUP": "P"},
                {**SLEEPING_P, "SGLANG_WEG2_GROUP": "D"}):
        assert _tracked(monkeypatch, env, {"load": 5.0}) is False


def test_the_riegel_is_wired_after_the_load_pool_release():
    from sglang.srt.model_executor import model_runner as MR

    src = inspect.getsource(MR)
    i = src.index('release_load_transient_pool(reason="after-load")')
    assert "assert_dual_p_sleep_tracked(self)" in src[i:i + 600]


def test_the_no_riegel_mutant_turns_the_untagged_test_red(monkeypatch):
    m = _exec_mutant(S, S.assert_no_untagged_live, "if live > tolerance_mib:", "if False:")
    monkeypatch.setattr(S, "assert_no_untagged_live", m)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        test_untagged_live_bytes_are_a_hard_stop()


# -- step 2: the weights tag pools' empty segments ---------------------------------------


def test_the_pool_slack_of_the_weights_tags_is_capped(monkeypatch):
    occ = {"weights": 0.13, "weights_0": 0.56, "weights_1": 0.55, "kv_cache": 9.0}
    got = S.weights_pool_slack_mib(lambda t: {"inactive_gib": occ[t]}, tags=list(occ))
    assert set(got) == {"weights", "weights_0", "weights_1"}, "kv is not a weights tag"
    assert sum(got.values()) == pytest.approx((0.13 + 0.56 + 0.55) * 1024)
    gmps9_pp0 = {"weights": 0.13 * 1024, **{f"weights_{i}": 0.55 * 1024 for i in range(4)},
                 "weights_4": 0.56 * 1024, "weights_5": 0.36 * 1024}
    with pytest.raises(S.Weg2DualPPoolSlack, match="W-DUAL-P-POOL-SLACK"):
        S.assert_weights_pool_slack(gmps9_pp0, cap_mib=1024.0)
    assert S.assert_weights_pool_slack({"weights": 160.0, "weights_7": 51.0}, cap_mib=1024.0) == pytest.approx(211.0)


def test_the_uncapped_slack_mutant_turns_the_slack_test_red(monkeypatch):
    m = _exec_mutant(S, S.assert_weights_pool_slack, "if total > cap:", "if False:")
    monkeypatch.setattr(S, "assert_weights_pool_slack", m)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        test_the_pool_slack_of_the_weights_tags_is_capped(monkeypatch)
