# SPDX-License-Identifier: Apache-2.0
"""DUAL D PRIORITY UNDER KV PRESSURE (user decision 01.10.): decode is never
interrupted in the dual layout.

Metal gmps7 (dkr27bnvfp4dual1mbar1fs10011748, D 17:53:15): "KV cache pool is full.
Retract requests. #retracted_reqs: 4" -> four W50 re-routes -> PP0's grant for four
contexts -> cuMemCreate OOM, PP0 dead.

DANGER DIRECTIONS guarded here, MUTANT per guard (asserted in-suite):
* (i) group D of the dual layout never retracts: _retract_decode_and_requeue
  raises W-DUAL-D-RETRACT BEFORE the batch is touched; off the dual layout (and on
  group P) the retract runs as before; the two callers that swallow exceptions
  re-raise the named stop.
"""
from __future__ import annotations

import inspect
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import scheduler as SC
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}


class _Touched(Exception):
    """the batch's retract_decode ran (the decode was interrupted)"""


def _retract(monkeypatch, env, fn=None):
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    def _retract_decode(_args):
        raise _Touched()

    batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="weg2-0-1")], retract_decode=_retract_decode)
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 0),
        new_token_ratio_tracker=types.SimpleNamespace(current=0.3),
        tree_cache=types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace()),
        _weg2_d_park_draft_snapshot=lambda b: None,
        server_args=None,
    )
    meth = types.MethodType(fn or SC.Scheduler._retract_decode_and_requeue, sched)
    return meth(batch, kv_full_retract_flag=True)


def test_dual_d_never_retracts_a_decode(monkeypatch):
    with pytest.raises(DP.Weg2DualDRetract, match="W-DUAL-D-RETRACT"):
        _retract(monkeypatch, DUAL_D)


@pytest.mark.parametrize("env", [{}, {"SGLANG_WEG2_GROUP": "D"},
                                 {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}],
                         ids=["no-weg2", "flip-D", "dual-P"])
def test_off_the_dual_d_the_retract_runs_as_before(monkeypatch, env):
    with pytest.raises(_Touched):
        _retract(monkeypatch, env)


def test_the_swallowing_callers_reraise_the_named_stop():
    for name in ("_weg2_d_retract_guard_sites",):
        assert not hasattr(SC.Scheduler, name)  # no second implementation of the guard
    src = inspect.getsource(SC.Scheduler)
    for marker in ('logger.warning("#888b carrier yield failed: %s", e)',
                   'logger.warning("%s rung 3 (retract) failed: %s", self._LADDER_PREFIX, e)'):
        i = src.index(marker)
        head = src[max(0, i - 400):i]
        assert "isinstance(e, Weg2DualDRetract)" in head and "raise" in head, marker


def test_the_guard_removed_mutant_turns_the_retract_test_red(monkeypatch):
    fn = inspect.unwrap(SC.Scheduler._retract_decode_and_requeue)
    src = textwrap.dedent(inspect.getsource(fn))
    guard = "refuse_d_retract(batch, kv_full=kv_full_retract_flag, reason=reason)"
    assert src.count(guard) == 1, "the guard moved -- re-aim the mutant"
    ns = dict(vars(SC))
    exec(compile(src.replace(guard, "pass"), SC.__file__, "exec"), ns)
    with pytest.raises(_Touched):                      # the decode WAS interrupted
        _retract(monkeypatch, DUAL_D, fn=ns["_retract_decode_and_requeue"])


# -- (ii) D's level counts its locked rows (a hold the bookkeeping misses) -------

from sglang.srt.weg2 import dual_d_kv_stage as DK  # noqa: E402


def _d(mapped, free, evictable):
    actor = types.SimpleNamespace(mapped_tokens=mapped,
                                  allocator=types.SimpleNamespace(available_size=lambda: free))
    sched = types.SimpleNamespace(tree_cache=types.SimpleNamespace(evictable_size=lambda: evictable))
    return sched, actor


def test_d_locked_rows_are_mapped_minus_free_minus_evictable():
    sched, actor = _d(mapped=221184, free=4000, evictable=10000)
    assert DK.d_locked_rows(sched, actor) == 221184 - 4000 - 10000


def test_a_hold_the_bookkeeping_misses_keeps_d_from_shrinking_under_it():
    # metal gmps7 17:53:05: 3 running (~156k incl. the queue) + weg2-0-13's #243 hold 61625
    demand, held, air, step = 156232, 61625, 1600 + 6 * 4, 4096
    sched, actor = _d(mapped=221184, free=221184 - demand - held, evictable=0)
    want = DK.want_local_tokens(demand, DK.d_locked_rows(sched, actor), air, step)
    assert want >= demand + held, "the level covers the held rows"
    verdict, _level = DK.decide(221184, want, p_waiting=True, below_rounds=0, step=step)
    assert verdict != "shrink", "D must not give back rows a hold still occupies"


def test_the_bookkeeping_only_mutant_turns_the_hold_test_red(monkeypatch):
    monkeypatch.setattr(DK, "want_local_tokens", lambda demand, locked, air, step:
                        DK.want_tokens(demand, 0, air, step))
    with pytest.raises(AssertionError):
        test_a_hold_the_bookkeeping_misses_keeps_d_from_shrinking_under_it()


def test_the_tick_uses_the_locked_level():
    assert "want_local_tokens(demand_local, d_locked_rows(sched, actor)" in inspect.getsource(DK.tick)
