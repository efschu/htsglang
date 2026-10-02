"""DP-NACHLAUF 02.10.: WAKE-PRELOAD on L15 wakes runs AFTER the L15 act.

l15-lead (3e0b2cd3a3, first real holds): once _l15_wake_act has run (hold:
held rows reserved, pools fixed; fallback: drop done) no L15 path touches the
pools any more -- before the act it is unsafe (a fallback resets). The old
gate skipped the preload on a per-rank term (this rank's stashed manifest,
which a cap-0 rank does not have) right before the preload's group vote.
Pinned (red before): the defer decision uses group-uniform terms only (group
D, kv in the call's tags, the L15 master env); in resume_memory_occupation
the kv-resume preload defers on it, the act comes next, then the post-act
preload (site=after_l15_act), behind the group kv verdict; wake_preload.run
logs its site.
"""
from __future__ import annotations

import inspect
import logging
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import wake_preload as wp  # noqa: E402

M = wu.SchedulerWeightUpdaterManager


def _self(group="D"):
    return SimpleNamespace(_weg2_group_name=lambda: group, _l15_wake_rpc=M._l15_wake_rpc,
                           _l15_wake_manifest=None)


def test_defer_gate_is_group_uniform(monkeypatch):
    from sglang.srt.weg2 import l15_plan

    monkeypatch.setattr(l15_plan, "master_on", lambda env: True)
    assert M._l15_preload_after_act(_self("D"), ["weights", "kv_cache"])
    assert not M._l15_preload_after_act(_self("D"), ["weights"])          # weights-only call
    assert not M._l15_preload_after_act(_self("P"), ["kv_cache"])         # not group D
    monkeypatch.setattr(l15_plan, "master_on", lambda env: False)
    assert not M._l15_preload_after_act(_self("D"), ["kv_cache"])         # master off: preload as before
    # a rank's own manifest plays no part (a cap-0 rank stashes none)
    s = _self("D")
    s._l15_wake_manifest = object()
    assert not M._l15_preload_after_act(s, ["kv_cache"])
    src = inspect.getsource(M._l15_preload_after_act)
    assert "_l15_wake_manifest" not in src


def test_order_defer_then_act_then_preload():
    src = inspect.getsource(M.resume_memory_occupation)
    i_defer = src.index("_wpl.run(self, l15_hold_aware=self._l15_preload_after_act(tags))")
    i_act = src.index("self._l15_wake_act(")
    i_post = src.index('_wpl.run(self, l15_hold_aware=False, site="after_l15_act")')
    assert i_defer < i_act < i_post
    gate = src[i_act:i_post]
    assert "if self._l15_preload_after_act(tags) and not _weg2_kv_refusal:" in gate


def test_run_logs_its_site(monkeypatch, caplog):
    monkeypatch.delenv(wp.ENV, raising=False)
    sched = SimpleNamespace(weg2_dormant_hold=[SimpleNamespace(rid="r")], enable_hierarchical_cache=True,
                            ps=SimpleNamespace(pp_size=1, tp_size=1), tree_cache=SimpleNamespace())
    from sglang.srt.managers import weg2_store_told as st
    monkeypatch.setattr(st, "armed", lambda s: False)
    with caplog.at_level(logging.INFO):
        assert wp.run(SimpleNamespace(scheduler=sched), l15_hold_aware=True) == 0
    assert any("off reason=l15_hold_aware" in r.getMessage() and "site=kv_resume" in r.getMessage()
               for r in caplog.records)
