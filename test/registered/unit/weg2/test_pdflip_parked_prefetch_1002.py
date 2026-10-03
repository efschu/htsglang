# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-A (02.10.2026): a running store read of a request in D's park list
never holds the D->P flip back.

N5d 1002_124821 epoch 6 (D->P begin 12:51:58.621, 14.3 s, layer 12.6 s):
weg2-6-10 (SHORT on D, 171166 tokens, 169224 in L3) had its store read running
(READ-STAGES l3fill_pages=104379 total_ms=15886) when the front parked D
(``park_running ... queued-behind=['weg2-6-10']``); every /flush_cache quiesce
poll answered 400 ``hicache_prefetch(1: weg2-6-1)`` (547x), FLIP STALL
stage=quiesce at 9.8 s. The request went into the #1443 hold anyway once the
read ended and was re-read at the wake.

Now: the open reads of the park list are not a quiesce / sleep-drain /
release-idle term (helper ``weg2_sleep_drain.parked_owned_prefetch``, marker
``WEG2-QUIESCE-PARKED-PREFETCH``, switch
``SGLANG_WEG2_PARKED_PREFETCH_NOT_A_SLEEP_TERM``). Hermetic, CPU. Each test
named red_* is red on e03d95c2b4.
"""

from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def _req(rid):
    return types.SimpleNamespace(rid=rid)


def test_red_the_park_lists_reads_are_exempt_and_nothing_else():
    from sglang.srt.managers.weg2_sleep_drain import parked_owned_prefetch

    got = parked_owned_prefetch(parked=[_req("weg2-6-8"), _req("weg2-6-10")],
                                ongoing_prefetch={"weg2-6-10": 1, "weg2-6-11": 2})
    assert got == frozenset({"weg2-6-10"})
    assert parked_owned_prefetch(parked=[], ongoing_prefetch={"weg2-6-10": 1}) == frozenset()
    assert parked_owned_prefetch(parked=[_req("weg2-6-10")], ongoing_prefetch={}) == frozenset()
    off = {"SGLANG_WEG2_PARKED_PREFETCH_NOT_A_SLEEP_TERM": "0"}
    assert parked_owned_prefetch(parked=[_req("weg2-6-10")], ongoing_prefetch={"weg2-6-10": 1},
                                 env=off) == frozenset()


def _sched(parked, ongoing):
    from sglang.srt.managers.scheduler import Scheduler

    s = types.SimpleNamespace(
        enable_hierarchical_cache=True, weg2_d_parked=parked,
        tree_cache=types.SimpleNamespace(ongoing_prefetch=ongoing),
        ps=types.SimpleNamespace(pp_size=1, pp_rank=0))
    seen = []

    def is_fully_idle(for_health_check=False, exempt_prefetch=()):
        seen.append(frozenset(exempt_prefetch))
        return all(str(r) in exempt_prefetch for r in ongoing)

    s.is_fully_idle = is_fully_idle
    s.idle_blockers = lambda exempt_prefetch=(): [
        "hicache_prefetch(%s)" % r for r in ongoing if r not in exempt_prefetch]
    s._weg2_parked_owned_prefetch = lambda: Scheduler._weg2_parked_owned_prefetch(s)
    return Scheduler, s, seen


def test_red_the_quiesce_verdict_is_idle_while_only_a_parked_read_runs():
    """D (pp_size=1) single-rank verdict: idle with the park list's read open."""
    Scheduler, s, seen = _sched([_req("weg2-6-10")], {"weg2-6-10": object()})
    idle, detail = Scheduler.group_idle_verdict(s)
    assert idle is True and frozenset({"weg2-6-10"}) in seen
    assert "hicache_prefetch" not in detail


def test_a_read_of_a_request_not_in_the_park_still_blocks():
    Scheduler, s, _ = _sched([_req("weg2-6-8")], {"weg2-6-10": object()})
    idle, detail = Scheduler.group_idle_verdict(s)
    assert idle is False and "hicache_prefetch(weg2-6-10)" in detail


def test_red_the_sleep_drain_and_the_release_idle_assert_exempt_it_too():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu.WeightUpdater._weg2_hold_owned_prefetch) if hasattr(wu, "WeightUpdater") \
        else inspect.getsource(wu)
    assert "parked_owned_prefetch(" in src
