"""fnFL2x82 (23.09.): the lane PLAN is derived at boot, once, into the cache
the first flip's lane threads read.

Bug regression. x80 (D TP0, first wake): the manifest join was warm
(fnFL2x40's prewarm, memo hit 11) but the plan derived from it was not --
the three lane threads of the first tag missed the ``("join", hook, group,
rank)`` key at once and each derived the 20-GiB plan (WEG2-LANE-DERIVE
400/402/420 ms); the first collect started 0,33 s after the first resume and
PP0's first tag waited 632 ms for it (x77 535, x78 469). Hermetic: the
manifest module is stubbed, no files, no process group.
"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402


class _Bare(wu.SchedulerWeightUpdaterManager):
    def __init__(self, group, rank, agreed="AGREED"):
        self._g, self._r = group, rank
        self._agreed = agreed
        self.hook_plans = []

    def _weg2_group_name(self):
        return self._g

    def _weg2_rank(self):
        return self._r

    def _weg2_shadow_manifest(self, group, peer, rank, *, leg, epoch):
        return self._agreed, ("agreed" if self._agreed is not None else "peer-missing")

    def _weg2_shadow_plan(self, hook, group, rank, *, agreed, require_agreement):
        # fnFL2x84: the hook's plan, keyed by the agreement (a boot constant)
        self.hook_plans.append((hook, group, rank, agreed, require_agreement))
        return ("PLAN", "")


def _stub_manifests(monkeypatch, calls):
    monkeypatch.setattr(xm, "manifests_for_boot", lambda **kw: (["m"], ""))
    monkeypatch.setattr(xm, "join_manifests", lambda mans, **kw: "JOIN")

    def plan_from_join(join, *, direction):
        calls.append(direction)
        return f"PLAN:{direction}"

    monkeypatch.setattr(xm, "plan_from_join", plan_from_join)


def test_the_boot_warm_up_fills_both_hooks_of_this_ranks_lane_cache(monkeypatch):
    calls = []
    _stub_manifests(monkeypatch, calls)
    m = _Bare("D", 0)
    ms = m._weg2_warm_leg_cache()
    assert sorted(ms) == ["hook:destination", "hook:source", "leg:authoritative", "leg:source"]
    # fnFL2x84: the hook's plan for both hooks, narrowed to the pair's agreement
    assert m.hook_plans == [("source", "D", 0, "AGREED", True),
                            ("destination", "D", 0, "AGREED", True)]
    lc = m._weg2_xchg_leg_cache
    # D imports on the authoritative hook (pp_to_tp) and exports on source (tp_to_pp)
    assert lc[("join", "authoritative", "D", 0)] == ("JOIN", "PLAN:pp_to_tp")
    assert lc[("join", sh.HOOK_SOURCE, "D", 0)] == ("JOIN", "PLAN:tp_to_pp")
    assert sorted(calls) == ["pp_to_tp", "tp_to_pp"]


def test_x84_without_an_agreement_the_hook_plan_is_not_warmed_but_the_lane_plan_is(monkeypatch):
    """The agreement needs the co-located peer's manifest; when it is not
    there yet only the hook half is skipped (-1), the lane plans still land."""
    calls = []
    _stub_manifests(monkeypatch, calls)
    m = _Bare("P", 2, agreed=None)
    ms = m._weg2_warm_leg_cache()
    assert ms["hook:source"] == -1.0 and ms["hook:destination"] == -1.0
    assert m.hook_plans == []
    assert ("join", "authoritative", "P", 2) in m._weg2_xchg_leg_cache


def test_without_a_group_or_manifests_nothing_is_warmed(monkeypatch):
    calls = []
    _stub_manifests(monkeypatch, calls)
    assert _Bare("", 0)._weg2_warm_leg_cache() == {}
    monkeypatch.setattr(xm, "manifests_for_boot", lambda **kw: (None, "not yet"))
    assert _Bare("P", 1)._weg2_warm_leg_cache() == {}
    assert calls == []


def test_the_join_prewarm_round_warms_the_lane_cache_and_the_derive_is_single_flight():
    """Bookkeeping: the boot thread must call the warm-up after a successful
    join round, and the lane derivation must hold its lock around the miss
    (three threads deriving the same key at once was x80's 0,33 s)."""
    src = open(wu.__file__).read()
    i = src.index("ms = xm.prewarm_joins()")
    assert "ms.update(self._weg2_warm_leg_cache())" in src[i:i + 400]
    j = src.index('_jk = ("join", str(hook), str(group), int(rank))')
    blk = src[j:j + 2400]
    assert "with _dl:" in blk and blk.index("with _dl:") < blk.index("plan = xm.plan_from_join(")
    assert isinstance(threading.Lock(), type(threading.Lock()))
