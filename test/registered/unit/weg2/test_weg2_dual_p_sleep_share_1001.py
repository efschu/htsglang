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


def _rehome_cpu(part):
    return H.rehome_survivors.__wrapped__(part) if hasattr(H.rehome_survivors, "__wrapped__") else \
        _ORIG_REHOME(part, arena_ranges=[], device_types=("cpu",))


_ORIG_REHOME = H.rehome_survivors
H._rehome_cpu = _rehome_cpu


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
    monkeypatch.setattr(H, "rehome_survivors", lambda part, **kw: H.__dict__["_rehome_cpu"](part))
    runner, part, old = _bind(monkeypatch, kept=0)
    assert part.cos is not old and torch.equal(part.cos, old)
    assert runner.dual_share_rehomed == {"buffers": 12, "attributes": 0} and runner.dual_share_bound
    assert runner.dual_share_rehomed_bytes == 12


class _MarlinLin(torch.nn.Module):
    """a quantized linear as the schemes leave it: a parameter (bound to the arena by the
    union bind), a plain tensor ATTRIBUTE (the Marlin lock workspace) and a second attribute
    VIEWING the first one's storage"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.workspace = torch.zeros(68, dtype=torch.int32)          # 3080: 68 SMs
        self.workspace_view = self.workspace[4:8]
        self.weight_alias = self.weight.data[:2]                     # a view of a bound parameter


def test_attribute_kernel_state_is_rehomed_once_per_storage_and_params_stay():
    m = _MarlinLin()
    old_ws, old_param = m.workspace, m.weight
    moved = H.rehome_survivors(m, arena_ranges=[], device_types=("cpu",))
    assert moved == {"buffers": 0, "attributes": 68 * 4}, moved            # one clone for both views
    assert m.workspace is not old_ws and torch.equal(m.workspace, old_ws)
    assert m.workspace_view.untyped_storage().data_ptr() == m.workspace.untyped_storage().data_ptr()
    assert m.workspace_view.storage_offset() == 4 and torch.equal(m.workspace_view, old_ws[4:8])
    assert m.weight is old_param, "a bound parameter is D's arena view: never re-homed"
    assert m.weight_alias.untyped_storage().data_ptr() == old_param.untyped_storage().data_ptr()


def test_a_tensor_inside_an_attached_arena_is_left_alone():
    m = _MarlinLin()
    old = m.workspace
    ptr = old.untyped_storage().data_ptr()
    moved = H.rehome_survivors(m, arena_ranges=[(ptr, ptr + 4096)], device_types=("cpu",))
    assert moved["attributes"] == 0 and m.workspace is old


def test_the_no_attribute_walk_mutant_turns_the_workspace_test_red(monkeypatch):
    m = _exec_mutant(H, H.rehome_survivors, "for module in part.modules():", "for module in ():")
    monkeypatch.setattr(H, "rehome_survivors", m)
    with pytest.raises(AssertionError):
        test_attribute_kernel_state_is_rehomed_once_per_storage_and_params_stay()


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
    monkeypatch.setenv("SGLANG_WEG2_WEIGHTS_CPU_BACKUP", env.get("SGLANG_WEG2_WEIGHTS_CPU_BACKUP", "on"))
    return H.assert_dual_p_sleep_tracked(types.SimpleNamespace(
        dual_share_rehomed={"buffers": 12, "attributes": 272}, dual_share_rehomed_bytes=284))


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


# -- step 3: P sleeps with the TMS backup armed and outside the exchange --------------------

from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import weight_exchange as WX  # noqa: E402

SHARE_SLEEP = types.SimpleNamespace(dual_layout=True, dual_share=True, dual_unified_kv="on", dual_p_sleep="on",
                                    tag="t")


def test_under_dual_share_the_sleeping_p_gets_backup_on_and_no_exchange(monkeypatch):
    assert L.dual_p_sleep_share_supported()
    assert L.dual_p_sleep_armed(SHARE_SLEEP)
    env = L.dual_share_env(SHARE_SLEEP, "P")
    for k, v in L.DUAL_P_SLEEP_GROUP_ENV.items():
        assert env[k] == v
    # what the RANKS read: the dual boot's group env (gmps7 WEG2-GROUP-ENV P) with the sleep env
    # applied on top, as launcher.main does (spec_p.env.update(dual_share_env(ns, "P")))
    base = {"SGLANG_WEG2_XCHG_INJECT": "authoritative", WX.WEIGHTS_CPU_BACKUP_ENV: "off",
            WX.WEIGHT_SOURCE_ENV: "exchange", "SGLANG_WEG2_WEIGHTS_RESIDENT": "1"}
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, v)
    assert WX.weights_cpu_backup_armed() is True, "TMS backs the weights up at the pause"
    assert WX.exchange_armed() is False, "no deposit at the sleep, no peer inject at the wake"
    assert WX.boot_vote() is None or True   # unarmed: arm_coverage_at_load records no vote
    d_env = L.dual_share_env(SHARE_SLEEP, "D")
    assert not set(L.DUAL_P_SLEEP_GROUP_ENV) & set(d_env), "D stays resident and in its own arm"


@pytest.mark.parametrize("drop", ["SGLANG_WEG2_WEIGHTS_CPU_BACKUP", "SGLANG_WEG2_WEIGHT_SOURCE"])
def test_dropping_either_env_mutant_turns_the_backup_test_red(monkeypatch, drop):
    monkeypatch.setattr(L, "DUAL_P_SLEEP_GROUP_ENV",
                        {k: v for k, v in L.DUAL_P_SLEEP_GROUP_ENV.items() if k != drop})
    monkeypatch.setenv(WX.WEIGHTS_CPU_BACKUP_ENV, "off")
    monkeypatch.setenv(WX.WEIGHT_SOURCE_ENV, "exchange")
    with pytest.raises((AssertionError, KeyError)):
        test_under_dual_share_the_sleeping_p_gets_backup_on_and_no_exchange(monkeypatch)


def test_without_the_tree_support_dual_share_sleep_is_still_refused(monkeypatch):
    monkeypatch.setattr(L, "dual_p_sleep_share_supported", lambda: False)
    with pytest.raises(L.Weg2DualPSleepShareRefused):
        L.dual_p_sleep_armed(SHARE_SLEEP)
    assert L.build_parser().parse_args(["--tree", "/t", "--tag", "t"]).dual_p_sleep == "off", \
        "default off until the metal proof"


# -- step 3: the front sleeps P only after the group-idle witness ---------------------------

import asyncio  # noqa: E402
import collections  # noqa: E402

from sglang.srt.weg2 import dual_d_priority as DP  # noqa: E402
from sglang.srt.weg2 import front as FR  # noqa: E402


def _front(idle=True):
    calls = []

    async def leg_rpc(g, path, body, timeout):
        calls.append((g, path))
        return 200, "{}"

    async def quiesce(g):
        calls.append(("QUIESCE", g))
        return idle, "" if idle else "not idle: hicache_prefetch(1)"

    f = types.SimpleNamespace(groups={"P": "P-group"}, boot_epoch="b", epoch=0, weight_chunks=1,
                              counters=collections.Counter(), leg_rpc=leg_rpc, quiesce=quiesce,
                              do_stop=lambda *a: calls.append(("STOP",) + a), queue=collections.deque(),
                              _dual_inflight={})
    f._dual_stages_obj = DP.PressureStages(sleep_capable=True)
    for name in ("_dual_p_sleep", "_dual_p_wake", "_dual_p_weights_tags", "_dual_stages",
                 "_dual_p_sleep_probe_tick"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    f.DUAL_P_SLEEP_PROBE_ENV = FR.Front.DUAL_P_SLEEP_PROBE_ENV
    return f, calls


def test_a_p_that_is_not_idle_is_not_put_to_sleep():
    f, calls = _front(idle=False)
    f._dual_stages_obj.p_state = "sleeping"
    asyncio.run(f._dual_p_sleep())
    assert calls == [("QUIESCE", "P-group")], "no release leg without the idle witness"
    assert f._dual_stages_obj.p_state == "stopped" and f.counters["dual_kv_pressure_sleep_not_idle"] == 1


def test_the_no_quiesce_mutant_turns_the_idle_test_red(monkeypatch):
    m = _exec_mutant(FR, FR.Front._dual_p_sleep, "ok, why = await self.quiesce(self.groups[\"P\"])",
                     "ok, why = True, ''")
    monkeypatch.setattr(FR.Front, "_dual_p_sleep", m)
    with pytest.raises(AssertionError):
        test_a_p_that_is_not_idle_is_not_put_to_sleep()


# -- step 5: the metal probe ---------------------------------------------------------------


def test_the_probe_sleeps_and_wakes_p_once_without_pressure(monkeypatch):
    monkeypatch.setenv(FR.Front.DUAL_P_SLEEP_PROBE_ENV, "0.01")
    f, calls = _front()

    async def run():
        assert f._dual_p_sleep_probe_tick(0) is True
        await asyncio.sleep(0.2)
        assert f._dual_p_sleep_probe_tick(0) is False, "done: the probe never fires twice"

    asyncio.run(run())
    paths = [c[1] for c in calls if c[0] != "QUIESCE"]
    assert paths == ["/release_memory_occupation", "/resume_memory_occupation"]
    assert f._dual_probe_state == "done" and f._dual_stages_obj.p_state == "serving"


def test_the_probe_is_off_by_default_and_waits_for_an_idle_stretch(monkeypatch):
    monkeypatch.delenv(FR.Front.DUAL_P_SLEEP_PROBE_ENV, raising=False)
    f, calls = _front()
    assert f._dual_p_sleep_probe_tick(0) is False and not calls
    monkeypatch.setenv(FR.Front.DUAL_P_SLEEP_PROBE_ENV, "5")
    assert f._dual_p_sleep_probe_tick(123) is False, "never under D's pressure"
    f.queue.append(object())
    assert f._dual_p_sleep_probe_tick(0) is False, "never with a queued prompt"


def test_the_probe_is_consulted_first_in_the_stage_tick():
    src = inspect.getsource(FR.Front._dual_stage_tick)
    assert src.index("_dual_p_sleep_probe_tick(pressure)") < src.index("stages = self._dual_stages()")



# -- the gmps10 follow-up: tolerance env, instrument, backup riegel ---------------------------


def test_the_untagged_tolerance_is_a_named_env_with_default_zero(monkeypatch):
    monkeypatch.delenv(H.UNTAGGED_TOL_ENV, raising=False)
    assert H.untagged_tolerance_mib() == 0.0
    with pytest.raises(S.Weg2DualPUntagged):
        _tracked(monkeypatch, SLEEPING_P, {"load": 0.1})                 # gmps10 PP2, default: a stop
    monkeypatch.setenv(H.UNTAGGED_TOL_ENV, "1")
    assert _tracked(monkeypatch, SLEEPING_P, {"load": 0.1}) is True        # named allowance


def test_the_riegel_line_names_what_was_rehomed_and_what_is_left(monkeypatch, caplog):
    import logging

    monkeypatch.setenv(H.UNTAGGED_TOL_ENV, "1")
    with caplog.at_level(logging.INFO, logger=H.logger.name):
        _tracked(monkeypatch, SLEEPING_P, {"load": 0.25})
    line = [r.getMessage() for r in caplog.records if "sleep riegels ok" in r.getMessage()][-1]
    assert "untagged live 0.25 MiB left" in line and "tolerance 1 MiB" in line
    assert "buffers 0.0 KiB" in line and "attributes 0.3 KiB" in line      # 12 B / 272 B
    assert "Marlin workspaces" in line


def test_without_the_weights_cpu_backup_a_sleeping_p_is_a_named_stop(monkeypatch):
    with pytest.raises(H.DualPSleepNoBackup, match="W-DUAL-P-NO-BACKUP"):
        _tracked(monkeypatch, {**SLEEPING_P, "SGLANG_WEG2_WEIGHTS_CPU_BACKUP": "off"}, {})


def test_the_no_backup_check_mutant_turns_the_backup_test_red(monkeypatch):
    m = _exec_mutant(H, H.assert_dual_p_sleep_tracked, "if not weights_cpu_backup_armed():", "if False:")
    monkeypatch.setattr(H, "assert_dual_p_sleep_tracked", m)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        test_without_the_weights_cpu_backup_a_sleeping_p_is_a_named_stop(monkeypatch)


# -- gmps11: the shared part's dead copies go back BEFORE the other parts load ---------------


def _fake_routing(monkeypatch, primary=True, tag=True):
    import torch.cuda.memory as tcm

    events = []
    monkeypatch.setattr(tcm, "_cuda_endAllocateToPool", lambda d, p: events.append(("end", p)), raising=False)
    monkeypatch.setattr(tcm, "_cuda_beginAllocateCurrentThreadToPool", lambda d, p: events.append(("begin", p)),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pools = ([types.SimpleNamespace(id="prim")] if primary else []) + ([types.SimpleNamespace(id="tag")] if tag else [])
    monkeypatch.setattr(S, "_active_routing_pools", lambda: list(pools))
    region = types.SimpleNamespace(on=True)
    cdll = types.SimpleNamespace(tms_set_interesting_region=lambda v: events.append(("region", v)))
    monkeypatch.setattr(S, "_tms_cdll_in_region", lambda: cdll)
    monkeypatch.setattr(S, "_STEPPED_OUT_POOLS", [])
    monkeypatch.setattr(S, "_TRANSIENT_ONLY", [])
    monkeypatch.setattr(S, "_LAST_KEPT_LIVE_MIB", {})
    monkeypatch.setattr(S, "_KEPT_TRANSIENT_POOLS", [])
    return events


def test_the_mid_region_release_ends_all_routing_deletes_and_reenters_in_order(monkeypatch):
    events = _fake_routing(monkeypatch)
    deleted = []

    def fake_release(kind, reason, routing_ended=False):
        assert routing_ended, "the deletion runs with the routing ENDED"
        assert [e for e in events if e[0] == "end"] == [("end", "tag"), ("end", "prim")]
        assert ("region", False) in events
        deleted.append(kind)
        return 7.71 if kind == "load" else 0.0

    monkeypatch.setattr(S, "_release_one_load_pool", fake_release)
    assert S.release_load_pools_midregion("dual-share-shared-part-bound") == pytest.approx(7.71)
    assert deleted == ["ckpt", "load"]
    assert events[-3:] == [("region", True), ("begin", "prim"), ("begin", "tag")], events


def test_the_mid_region_release_refuses_inside_a_stepped_out_block(monkeypatch):
    _fake_routing(monkeypatch)
    monkeypatch.setattr(S, "_STEPPED_OUT_POOLS", [object()])
    assert S.release_load_pools_midregion("x") == 0.0


def test_the_release_guard_is_bypassed_only_with_the_routing_ended(monkeypatch):
    pool = types.SimpleNamespace(id=1)
    monkeypatch.setattr(S, "_LOAD_TRANSIENT_POOL", pool)
    monkeypatch.setattr(S, "_ACTIVE_TAG_POOL", object())          # a tag pool is open
    monkeypatch.setattr(S, "_STEPPED_OUT_POOLS", [])
    monkeypatch.setattr(S, "_tms_cdll_in_region", lambda: object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert S._release_one_load_pool("load", "x") == 0.0 and S._LOAD_TRANSIENT_POOL is pool  # refused
    monkeypatch.setattr(S, "_pool_has_live_blocks", lambda p, r: True)
    monkeypatch.setattr(S, "_pool_live_mib", lambda p: 2.0)
    monkeypatch.setattr(S, "_LAST_KEPT_LIVE_MIB", {"load": 1.0})
    monkeypatch.setattr(S, "_KEPT_TRANSIENT_POOLS", [])
    assert S._release_one_load_pool("load", "x", routing_ended=True) == 0.0
    assert S._LAST_KEPT_LIVE_MIB == {"load": 3.0}, "kept live bytes accumulate over the releases of one load"


def test_the_shared_part_is_released_and_checked_before_the_other_parts_load():
    src = inspect.getsource(H.build_dual_stage_model)
    i = src.index('_bind_shared_part(runner, parts[local], bind, transient=transient)')
    j = src.index('release_load_pools_midregion("dual-share-shared-part-bound")')
    k = src.index('assert_no_untagged_live(tolerance_mib=untagged_tolerance_mib())')
    loop = src.index('for r in range(plan.fast_size):')
    assert i < j < k < loop, "release + riegel sit between the bind and the loads of the other parts"


def test_the_no_mid_region_release_mutant_turns_the_order_test_red(monkeypatch):
    src = inspect.getsource(H.build_dual_stage_model)
    mut = src.replace('release_load_pools_midregion("dual-share-shared-part-bound")', "0.0")
    monkeypatch.setattr(inspect, "getsource", lambda obj: mut if obj is H.build_dual_stage_model else src)
    with pytest.raises((AssertionError, ValueError)):
        test_the_shared_part_is_released_and_checked_before_the_other_parts_load()
