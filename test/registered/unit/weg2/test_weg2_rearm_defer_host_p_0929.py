"""#284 (SGLANG_WEG2_REARM_DEFER_HOST_GROUPS): the P wake does not wait for its Platztausch rows.

MEASURED (x178 b89592806a, z30w-park 50ae2014b0, z30x2-kvdemand 424346f693;
all 63 D->P wakes): the P wake RPC ends with PP1, never with PP0. PP1's
``WEG2-WAKE-TAIL leg_collects=1643 reload=895`` -- the reload is the serial
Platztausch rearm, ``WEG2-RESUME expert-rearm rows_from_store=7040 serial=7040
deferred=0 ... ms=893`` (z30w median 6952 rows / 919 ms, kvdemand 7656 / 1068);
PP2 6656 rows / 476 ms. PP0 (232 rows, 17 ms) waits 474-538 ms in the
``GROUP-FENCE resume`` for them. The rows are copied from the pinned store over
the 3080's link and the rearm host-waits the copy before it answers.

The fix reuses H31b's DeferredRowsFill in an EARLY form for a group whose MoE
layers plan on the host (P's prefill, ``run_waves``): the rearm zeroes the pad,
issues the extra rows on a side stream right behind itself and returns; the
rank's NEXT forward -- of any mode -- makes its stream wait each layer's event
(no host wait) and promotes the layer. PP1's first forward comes only after
PP0's first chunk, so the copy lands during the fence, the RPC answer and
PP0's compute.

Pinned on CPU (fake stream handles; the rows and the order are the contract):
  * metal shape: a host-planned layer's rearm loads nothing serially under
    DEFER_HOST (RED on 895559fed2: 8 rows serial);
  * the rows are issued at start, the first EXTEND tick waits the events and
    lands every layer (RED: nothing is pending on the base);
  * run_waves' land waits the layer's event; the sleep settles early rows;
  * a pool layer under DEFER_HOST lands at the first extend too;
  * D keeps H31b, P is armed only by the env; 27B (no offload layer) is neutral;
  * wiring: the wake starts the fill behind the late rearm.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_pool_device as ep

HOST = getattr(eo, "DEFER_HOST", "host")
E = 12
PREFIX = [1, 2, 3, 4, 5]
PAD_ROW = 5
EXTRA = [(6, 9, 3), (7, 10, 4)]  # (row, expert, store slot)
R = 8
C, S = 6, 2
ATTRS = ("w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale")
SLOTS = 16


class _Ev:
    def __init__(self, done):
        self.done, self.synced = done, False

    def query(self):
        return self.done

    def synchronize(self):
        self.synced = True
        self.done = True


class _Ops:
    """Fake stream handles: the copies run synchronously, the events say when
    they would be done; ``waited`` = the events the forward stream waited."""

    def __init__(self, done=False):
        self.done = done
        self.events, self.waited = [], []

    def new_stream(self):
        return object()

    def stream_ctx(self, _s):
        return contextlib.nullcontext()

    def after_current(self, _s):
        pass

    def record(self, _s):
        ev = _Ev(self.done)
        self.events.append(ev)
        return ev

    def current_waits(self, ev):
        self.waited.append(ev)


@pytest.fixture
def ops(monkeypatch):
    o = _Ops()
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=o))
    yield o


def _layout():
    order = PREFIX + [0] + [e for _r, e, _p in EXTRA]
    slot = {e: i for i, e in enumerate(order)}
    cold = [e for e in range(E) if e not in slot]
    pool_index = {e: 8 + i for i, e in enumerate(cold)}
    return order, slot, pool_index


def _cache(pool=False, layer_id=3, seed=0):
    order, slot, pool_index = _layout()
    g = torch.Generator().manual_seed(seed)
    c = eo.MoEExpertOffloadCache.__new__(eo.MoEExpertOffloadCache)
    runs = tuple(eo._refill_runs([(PAD_ROW, -1)] + [(r, p) for r, _e, p in EXTRA]))
    c.layer = SimpleNamespace(layer_id=layer_id, _moe_offload_refill_runs=runs)
    c.resident_count, c.num_local_experts = R, E
    c.planner = SimpleNamespace(resident_ids=set(order), resident_slot=dict(slot))
    c._spill_pool_index = [pool_index.get(e, -1) for e in range(E)]
    c._resident, c._pinned = {}, {}
    for attr in ATTRS:
        spill = torch.randint(-2**30, 2**30, (SLOTS, 4), generator=g, dtype=torch.int32)
        buf = torch.full((R + C, 4), 7, dtype=torch.int32)  # D's residue
        buf[: len(PREFIX)] = torch.randint(0, 99, (len(PREFIX), 4), generator=g, dtype=torch.int32)
        c._resident[attr], c._pinned[attr] = buf, spill
    c._scratch_holds = {}
    c._pool_ready = pool
    c._pool_pf_buffers = None
    if pool:
        hot, host = c._pool_layout()
        c._pool_tables = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot, host)
    return c


def _rows_digest(c):
    h = hashlib.sha256()
    for attr in ATTRS:
        h.update(c._resident[attr].numpy().tobytes())
    return h.hexdigest()


def _serial(**kw):
    c = _cache(**kw)
    c.rearm_after_wake()
    return c


def _extend():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))


def test_metal_shape_the_p_rearm_loads_no_row_serially(ops):
    """x178 PP1: rows_from_store=7040 serial=7040 reload=895 ms before the fence."""
    c = _cache()
    assert c.rearm_after_wake(defer=HOST) == 0
    fill = eo.deferred_rows_fill()
    assert fill.pending == [c] and fill.rows_pending() == 2 * len(ATTRS)
    for attr in ATTRS:
        assert torch.all(c._resident[attr][PAD_ROW] == 0)  # pad at once
        assert torch.all(c._resident[attr][6:8] == 7)  # extras not yet


def test_rows_issue_at_start_and_the_first_extend_waits_them(ops):
    c = _cache()
    c.rearm_after_wake(defer=HOST)
    fill = eo.deferred_rows_fill()
    fill.start(why="behind the rearm, before the next forward")
    assert len(ops.events) == 1 and ops.waited == []  # issued, nobody waited
    assert c._deferred_rows is not None
    eo.deferred_rows_tick(_extend())  # P's first forward: an extend
    assert ops.waited == ops.events  # stream wait, no host wait
    assert c._deferred_rows is None and not fill.pending
    assert _rows_digest(c) == _rows_digest(_serial())


def test_run_waves_land_waits_the_layer_event(ops):
    c = _cache()
    c.rearm_after_wake(defer=HOST)
    eo.deferred_rows_fill().start()
    assert c.land_deferred_rows() is True
    assert ops.waited == ops.events
    assert _rows_digest(c) == _rows_digest(_serial())


def test_sleep_settles_early_rows(ops):
    c = _cache()
    c.rearm_after_wake(defer=HOST)
    eo.deferred_rows_fill().start()
    eo.deferred_rows_fill().settle()
    assert ops.events[0].synced
    assert c._deferred_rows is None and not eo.deferred_rows_fill().pending


def test_a_pool_layer_under_host_mode_lands_at_the_first_extend(ops):
    c = _cache(pool=True)
    c.rearm_after_wake(defer=HOST)
    eo.deferred_rows_fill().start()
    eo.deferred_rows_tick(_extend())
    assert c._deferred_rows is None and not eo.deferred_rows_fill().pending
    ref = _serial(pool=True)
    assert _rows_digest(c) == _rows_digest(ref)
    for name in ("hot_phys", "host_row"):
        assert torch.equal(getattr(c._pool_tables, name), getattr(ref._pool_tables, name))


def test_d_keeps_h31b_and_p_is_armed_only_by_the_env():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    armed = wu.SchedulerWeightUpdaterManager._weg2_rearm_defer_armed
    h = SimpleNamespace()
    h._weg2_group_name = lambda: "D"
    assert armed(h) is True
    h._weg2_group_name = lambda: "P"
    with envs.SGLANG_WEG2_REARM_DEFER_HOST_GROUPS.override(""):
        assert armed(h) is False
    with envs.SGLANG_WEG2_REARM_DEFER_HOST_GROUPS.override("P"):
        assert armed(h) == eo.DEFER_HOST
        with envs.SGLANG_WEG2_REARM_DEFER.override(False):
            assert armed(h) is False
    assert envs.SGLANG_WEG2_REARM_DEFER_HOST_GROUPS.default == ""


def test_27b_model_without_offload_layers_is_neutral(ops):
    """27B: no Platztausch buffer, nothing to rearm -- the host mode is a no-op."""
    root = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    assert eo.rearm_expert_offload_after_wake(root, defer=HOST) == (0, 0)
    fill = eo.deferred_rows_fill()
    fill.start()
    eo.deferred_rows_tick(_extend())
    assert not fill.pending and ops.events == [] and ops.waited == []


def test_the_wake_starts_the_fill_behind_the_late_rearm():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    res = inspect.getsource(wu.SchedulerWeightUpdaterManager.resume_memory_occupation)
    marks = [
        "_m, prefetch=_rearm_pf, defer=_defer, sync=False)",
        "if _defer == DEFER_HOST:",
        "deferred_rows_fill().start(",
        "deferred=%d",
    ]
    pos = [res.find(m) for m in marks]
    assert all(p >= 0 for p in pos), dict(zip(marks, pos))
    assert pos == sorted(pos), dict(zip(marks, pos))
