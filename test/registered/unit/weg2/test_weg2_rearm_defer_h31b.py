"""fnFL2 H31b (SGLANG_WEG2_REARM_DEFER): the Platztausch extra rows load AFTER the first token.

x147 (1d95257feb, H31 on, Kante K): the prefetch DURING the legs was mechanically
green (prefetched=1972/1740/232, wait_ms 0.1) and the P->D flip grew from 2.57 s
(x146, serial) to 4.10 s: the worker cards' input link is the pacer of the legs
(TP1 x4 6.5 GB/s receives ~14 GB of exchange plus the arena's KV there),
leg_collects +286 ms, and the arena's KV read was still short at the drain
(94208 of 97792, held +0.9 s). Serial after the legs (x146) the same bytes cost
TP1 ~165 ms before the fence. Both are before the first token.

H31b: on the waking decode group a POOL layer's extra experts go into the
device tables as COLD (host_row = their store slot -- a decode miss fetches the
same bytes), the pad is zeroed at once, the first DECODE forward starts the row
load on a side stream (one event per layer) and later forwards promote the
finished layers to the full layout; an eager forward lands its layer first
(``run_waves`` plans with full residency on the host); the sleep settles a
running fill before the first pause. Also: the target is rearmed before the
draft unpark and the rearm syncs its own stream, never the device.

Pinned here on CPU pool tables (the kernels are not needed: the tables and the
rows are the whole contract):
  * deferred rearm: pad zeroed, extras untouched, extras cold in the tables;
  * no start on a non-decode forward, start on the first decode forward,
    promotion only of layers whose event is done, in stream order;
  * after the cycle, rows AND tables equal the serial rearm's (digest);
  * an eager forward lands its layer (rows on the current stream);
  * sleep settles (host wait on the events) and forgets;
  * a map the tables cannot express falls back to the serial load;
  * mutant "fill discarded" changes the digest;
  * wiring: scheduler tick on the forward's stream in both run_batch branches,
    weight_updater order target-rearm < unpark < draft-rearm, settle before the
    park, defer only on group D.
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

E = 12  # local experts; 0 is the zero pad expert
PREFIX = [1, 2, 3, 4, 5]  # rows 0..4 come over the exchange
PAD_ROW = 5  # expert 0
EXTRA = [(6, 9, 3), (7, 10, 4)]  # (row, expert, store slot)
R = 8
C, S = 6, 2  # scratch rows (LRU 4 + staging 2)
ATTRS = ("w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale")
SLOTS = 16


@pytest.fixture(autouse=True)
def _fresh_fill(monkeypatch):
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=None))
    yield


def _layout():
    order = PREFIX + [0] + [e for _r, e, _p in EXTRA]
    slot = {e: i for i, e in enumerate(order)}
    cold = [e for e in range(E) if e not in slot]
    pool_index = {e: 8 + i for i, e in enumerate(cold)}  # store slots of the cold ones
    return order, slot, pool_index


def _cache(layer_id=3, runs=None, seed=0):
    order, slot, pool_index = _layout()
    g = torch.Generator().manual_seed(seed)
    c = eo.MoEExpertOffloadCache.__new__(eo.MoEExpertOffloadCache)
    if runs is None:
        runs = tuple(eo._refill_runs([(PAD_ROW, -1)] + [(r, p) for r, _e, p in EXTRA]))
    c.layer = SimpleNamespace(layer_id=layer_id, _moe_offload_refill_runs=runs)
    c.resident_count, c.num_local_experts = R, E
    c.planner = SimpleNamespace(resident_ids=set(order), resident_slot=dict(slot))
    c._spill_pool_index = [pool_index.get(e, -1) for e in range(E)]
    c._resident, c._pinned = {}, {}
    for attr in ATTRS:
        spill = torch.randint(-2**30, 2**30, (SLOTS, 4), generator=g, dtype=torch.int32)
        buf = torch.full((R + C, 4), 7, dtype=torch.int32)  # P's residue
        buf[: len(PREFIX)] = torch.randint(0, 99, (len(PREFIX), 4), generator=g, dtype=torch.int32)
        c._resident[attr], c._pinned[attr] = buf, spill
    c._scratch_holds = {}
    c._pool_ready = True
    hot, host = c._pool_layout()
    c._pool_tables = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot, host)
    c._pool_pf_buffers = None
    return c


def _digest(c):
    h = hashlib.sha256()
    for attr in ATTRS:
        h.update(c._resident[attr].numpy().tobytes())
    t = c._pool_tables
    for name in ("hot_phys", "host_row", "row_key", "row_use", "staging_rows", "pf_row"):
        h.update(getattr(t, name).numpy().tobytes())
    return h.hexdigest()


def _serial_digest(**kw):
    c = _cache(**kw)
    assert c.rearm_after_wake() == 2 * len(ATTRS)
    return _digest(c)


def _decode():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))


def _extend():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))


def test_deferred_rearm_zeroes_the_pad_and_puts_the_extras_cold():
    c = _cache()
    assert c.rearm_after_wake(defer=True) == 0
    fill = eo.deferred_rows_fill()
    assert fill.pending == [c] and fill.rows_pending() == 2 * len(ATTRS)
    for attr in ATTRS:
        buf = c._resident[attr]
        assert torch.all(buf[PAD_ROW] == 0)
        assert torch.all(buf[6:8] == 7)  # not loaded yet
    t = c._pool_tables
    for row, e, p in EXTRA:
        assert int(t.hot_phys[e]) == -1 and int(t.host_row[e]) == p
        assert int(t.row_key[row]) == -1
        assert int(t.row_use[row]) == ep.ROW_USE_NEVER  # never an LRU victim
    for e in PREFIX:
        assert int(t.hot_phys[e]) >= 0 and int(t.host_row[e]) == -1


def test_cycle_ends_in_the_serial_rows_and_tables():
    c = _cache()
    c.rearm_after_wake(defer=True)
    eo.deferred_rows_tick(_extend())  # first pass after the wake: skip-extend
    assert not eo.deferred_rows_fill().started
    eo.deferred_rows_tick(_decode())  # first decode forward: start
    assert eo.deferred_rows_fill().started and c._deferred_rows is not None
    eo.deferred_rows_tick(_decode())  # next forward: promote (CPU: already done)
    assert c._deferred_rows is None and not eo.deferred_rows_fill().pending
    assert _digest(c) == _serial_digest()


def test_rearm_function_defers_through_the_module_walk():
    c = _cache()
    mod = torch.nn.Module()
    mod._expert_offload = c
    mod._moe_offload_refill_runs = c.layer._moe_offload_refill_runs
    root = torch.nn.Module()
    root.experts = mod
    layers, rows = eo.rearm_expert_offload_after_wake(root, defer=True)
    assert (layers, rows) == (1, 0)
    assert eo.deferred_rows_fill().rows_pending() == 8
    layers, rows = eo.rearm_expert_offload_after_wake(root, defer=False)
    assert (layers, rows) == (1, 8) and not eo.deferred_rows_fill().pending
    assert _digest(c) == _serial_digest()


def test_eager_forward_lands_its_layer_first():
    c = _cache()
    c.rearm_after_wake(defer=True)
    assert c.land_deferred_rows() is True
    assert c._deferred_rows is None and not eo.deferred_rows_fill().pending
    assert _digest(c) == _serial_digest()
    assert c.land_deferred_rows() is False


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
    they would be done."""

    def __init__(self, done_flags):
        self.done_flags = list(done_flags)
        self.events, self.waited = [], []

    def new_stream(self):
        return object()

    def stream_ctx(self, _s):
        return contextlib.nullcontext()

    def after_current(self, _s):
        pass

    def record(self, _s):
        ev = _Ev(self.done_flags.pop(0) if self.done_flags else True)
        self.events.append(ev)
        return ev

    def current_waits(self, ev):
        self.waited.append(ev)


def test_tick_promotes_only_finished_layers_in_stream_order(monkeypatch):
    ops = _Ops([True, False])
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=ops))
    a, b = _cache(layer_id=1, seed=1), _cache(layer_id=2, seed=2)
    a.rearm_after_wake(defer=True)
    b.rearm_after_wake(defer=True)
    eo.deferred_rows_tick(_decode())
    assert len(ops.events) == 2
    eo.deferred_rows_tick(_decode())
    assert a._deferred_rows is None and b._deferred_rows is not None
    assert ops.waited == [ops.events[0]]  # the forward stream waits the layer's event
    ops.events[1].done = True
    eo.deferred_rows_tick(_decode())
    assert b._deferred_rows is None and not eo.deferred_rows_fill().pending
    assert _digest(a) == _serial_digest(layer_id=1, seed=1)
    assert _digest(b) == _serial_digest(layer_id=2, seed=2)


def test_eager_land_after_start_waits_the_layer_event_not_a_copy(monkeypatch):
    ops = _Ops([False])
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=ops))
    c = _cache()
    c.rearm_after_wake(defer=True)
    eo.deferred_rows_tick(_decode())
    c.land_deferred_rows()
    assert ops.waited == [ops.events[0]]
    assert _digest(c) == _serial_digest()


def test_sleep_settles_a_running_fill_and_forgets(monkeypatch):
    ops = _Ops([False])
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=ops))
    c = _cache()
    c.rearm_after_wake(defer=True)
    eo.deferred_rows_tick(_decode())
    eo.deferred_rows_fill().settle()
    assert ops.events[0].synced
    assert c._deferred_rows is None and not eo.deferred_rows_fill().pending
    # the next wake starts from scratch
    c.rearm_after_wake(defer=True)
    assert eo.deferred_rows_fill().pending == [c]


def test_a_map_the_tables_cannot_express_loads_serially():
    # an extra run on a row that holds no resident expert (row 13 is scratch)
    runs = ((PAD_ROW, -1, 1), (13, 3, 1))
    c = _cache(runs=runs)
    assert c.rearm_after_wake(defer=True) == len(ATTRS)
    assert not eo.deferred_rows_fill().pending


def test_non_pool_layer_is_never_deferred():
    c = _cache()
    c._pool_ready = False
    assert c.rearm_after_wake(defer=True) == 2 * len(ATTRS)
    assert not eo.deferred_rows_fill().pending


def test_mutant_fill_discarded_changes_the_digest(monkeypatch):
    def _discarding_issue(self, caches):
        for cache in caches:
            self.events[id(cache)] = None  # "issued", bytes never copied

    monkeypatch.setattr(eo.DeferredRowsFill, "_issue", _discarding_issue)
    c = _cache()
    c.rearm_after_wake(defer=True)
    eo.deferred_rows_tick(_decode())
    eo.deferred_rows_tick(_decode())
    assert not eo.deferred_rows_fill().pending
    assert _digest(c) != _serial_digest()


# --------------------------------------------------------------------------
# wiring


def test_scheduler_ticks_on_the_forward_stream_in_both_branches():
    from sglang.srt.managers import scheduler as sched_mod

    # the module text, cut at run_batch: inspect's block finder stops early
    # inside this method's body
    mod = inspect.getsource(sched_mod)
    i0 = mod.index("        # Run forward\n        if self.is_generation:\n")
    src = mod[i0:i0 + 6000]
    tick = "self._weg2_rearm_defer_tick(batch)"
    i_non = src.index("if not self.enable_overlap:")
    assert src.index(tick) == src.index(tick, i_non) < src.index("if self.enable_overlap:")
    i_wait = src.index("self.forward_stream.wait_stream(self.schedule_stream)")
    assert "with self.forward_stream_ctx:" in src[i_wait - 80:i_wait]
    i_tick = src.index(tick, i_wait)
    assert i_tick < src.index("self.model_worker.forward_batch_generation(", i_wait)
    assert src.count(tick) == 2


def test_wake_orders_target_rearm_unpark_draft_rearm():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    cls = wu.SchedulerWeightUpdaterManager
    res = inspect.getsource(cls.resume_memory_occupation)
    marks = [
        '_weg2_ph("leg_collects")',
        "_defer = self._weg2_rearm_defer_armed()",
        "_scratch = self._weg2_zero_local_scratch(_early)",
        "_m, prefetch=_rearm_pf, defer=_defer, sync=True)",
        "self._weg2_unpark_draft_start(",
        "_scratch += self._weg2_zero_local_scratch(_late)",
        "_m, prefetch=_rearm_pf, defer=_defer, sync=False)",
        "deferred=%d",
    ]
    pos = [res.find(m) for m in marks]
    assert all(p >= 0 for p in pos), dict(zip(marks, pos))
    assert pos == sorted(pos), dict(zip(marks, pos))
    rel = inspect.getsource(cls.release_memory_occupation)
    assert rel.index("self._weg2_rearm_defer_settle()") < rel.index(
        "self._weg2_park_draft_at_sleep(credit)") < rel.index("self.memory_saver_adapter.pause(tag)")


def test_defer_is_armed_only_on_the_decode_group():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    h = SimpleNamespace()
    armed = wu.SchedulerWeightUpdaterManager._weg2_rearm_defer_armed
    for group, env, want in (("D", True, True), ("P", True, False), ("D", False, False)):
        h._weg2_group_name = lambda g=group: g
        with envs.SGLANG_WEG2_REARM_DEFER.override(env):
            assert armed(h) is want


def test_rearm_syncs_its_stream_never_the_device():
    src = inspect.getsource(eo.rearm_expert_offload_after_wake)
    assert "torch.cuda.current_stream().synchronize()" in src
    assert "torch.cuda.synchronize()" not in src


def test_in_leg_prefetch_is_off_by_default():
    assert envs.SGLANG_WEG2_REARM_PREFETCH.default is False
    assert envs.SGLANG_WEG2_REARM_DEFER.default is True
