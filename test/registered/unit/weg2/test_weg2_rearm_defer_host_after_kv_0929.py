"""#284b: the DEFER_HOST rows go on the side stream only after the wake's last host wait.

MEASURED (NF y3j 09291933, #284 on, REARM_DEFER_HOST_GROUPS=P; 22 D->P wakes):
the rearm issued the rows at once (``WEG2-RESUME expert-rearm serial=0
deferred=9460 ms=15``, PP1 ``reload`` 1262 -> 16 ms), and the next host wait of
the same wake paid the whole copy again -- per rank ``WEG2-WAKE-TAIL-SUB
static_import`` equals the serial rearm it replaced (PP1 1146 vs 1262 ms,
PP2 685 vs 716, PP0 134 vs 238). ``_import_static_state`` copies the host stash
(pageable) into the model's buffers and queues behind the side stream's copy on
the card's H2D engine. The kv RPC of the same wake holds a second host wait,
the lmem restore's ``cuCtxSetLimit`` (a device-idle wait). The P wake stayed
at 2.6 s (y3p 09292250, #284 off: PP1 leg_collects 1344 + reload 1300 ms,
median of 34 D->P flips; the legs are 89 % of the D->P flip).

Fix: the rearm leaves the rows pending; the resume RPC's last statement issues
them once the group is no longer dormant (kv resumed, admission open), so no
host wait of the wake remains behind them. They land while PP0 computes its
first chunk (y3p: kv-RPC end -> PP1's first forward 1.51..4.60 s, median 2.90,
against PP1's ~1.3 s of rows); the next forward still waits the events, and
the first tick starts the fill itself if the wake never did.

Pinned on CPU (fake stream handles):
  * only early (DEFER_HOST) layers are started by the wake; H31b's decode
    layers keep their first-decode start;
  * a dormant group issues nothing (the weights RPC of a late-kv wake), the
    admission-open one issues once;
  * the tick backstop starts and lands the rows when the wake never did;
  * wiring: nothing is issued between the rearm and the static import; the
    start is the resume RPC's last statement, behind the late kv clear and
    the fence.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe import expert_offload as eo

from test_weg2_rearm_defer_host_p_0929 import (  # noqa: E402  (sibling test module)
    HOST,
    _cache,
    _extend,
    _Ops,
    _rows_digest,
    _serial,
)


@pytest.fixture
def ops(monkeypatch):
    o = _Ops()
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=o))
    yield o


def _manager(dormant):
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    h = SimpleNamespace(scheduler=SimpleNamespace(weg2_dormant=dormant))
    return lambda: wu.SchedulerWeightUpdaterManager._weg2_defer_host_fill_start(h)


def test_the_wake_starts_only_host_planned_layers(ops):
    """H31b (D): the rows wait for the first decode tick -- the wake leaves them."""
    d = _cache(pool=True)  # H31b defers pool layers only
    assert d.rearm_after_wake(defer=True) == 0
    fill = eo.deferred_rows_fill()
    assert fill.pending == [d] and not d._deferred_rows.early
    assert fill.start_host_planned(why="test") is False
    assert ops.events == [] and not fill.started and fill.pending == [d]


def test_dormant_group_issues_nothing_the_open_one_issues_once(ops):
    c = _cache()
    c.rearm_after_wake(defer=HOST)
    fill = eo.deferred_rows_fill()
    # the weights RPC of a late-kv wake: still dormant -> nothing on the copy engine
    assert _manager(dormant=True)() is False
    assert ops.events == [] and not fill.started
    # the kv RPC cleared the dormancy: the rows go now, nobody waits them
    assert _manager(dormant=False)() is True
    assert len(ops.events) == 1 and ops.waited == [] and fill.started
    assert _manager(dormant=False)() is False  # idempotent
    assert len(ops.events) == 1
    eo.deferred_rows_tick(_extend())  # P's first forward waits the event
    assert ops.waited == ops.events and not fill.pending
    assert _rows_digest(c) == _rows_digest(_serial())


def test_no_scheduler_issues_nothing(ops):
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    c = _cache()
    c.rearm_after_wake(defer=HOST)
    assert wu.SchedulerWeightUpdaterManager._weg2_defer_host_fill_start(
        SimpleNamespace(scheduler=None)) is False
    assert ops.events == []


def test_tick_backstop_starts_and_lands_when_the_wake_never_did(ops):
    c = _cache(pool=True)
    c.rearm_after_wake(defer=HOST)
    eo.deferred_rows_tick(_extend())
    assert len(ops.events) == 1 and ops.waited == ops.events
    assert c._deferred_rows is None and not eo.deferred_rows_fill().pending
    assert _rows_digest(c) == _rows_digest(_serial(pool=True))


def test_wiring_nothing_is_issued_before_the_wakes_host_waits():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    res = inspect.getsource(wu.SchedulerWeightUpdaterManager.resume_memory_occupation)
    # RED on 6f38e405b0: the rearm issued the rows before the static import
    assert "deferred_rows_fill().start(" not in res
    marks = [
        "_m, prefetch=_rearm_pf, defer=_defer, sync=False)",  # the late rearm
        "_import_static_state(",  # host wait 1 (pageable H2D)
        "_weg2_kv_clear_part()\n                _weg2_kv_ok = True",  # late kv clear: dormancy ends
        '"WEG2-WAKE-TAIL ms "',  # the fence is behind
        "self._weg2_defer_host_fill_start()",
        '_weg2_leg_commit("resume"',
    ]
    pos = [res.find(m) for m in marks]
    assert all(p >= 0 for p in pos), dict(zip(marks, pos))
    assert pos == sorted(pos), dict(zip(marks, pos))
    assert res.count("self._weg2_defer_host_fill_start()") == 1
