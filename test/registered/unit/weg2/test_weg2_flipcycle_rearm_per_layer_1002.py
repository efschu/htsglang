"""FLIPCYCLE H2 (02.10.): P's first extend after the wake waits only for EACH
layer's own deferred extra rows, not for all of them before the forward starts.

y6z (081355Z), D->P epoch 9: P-PP0 `WEG2-REARM-DEFER landed ... fill_ms=973`,
`PLE-PREFETCH ... gather_ms=743.8` -- the first PP0 forward started ~0.96 s
after the wake (chunk window 2392 ms for a 1435 ms forward); PP1's rows landed
360 ms after PP0 had finished. The tick at the forward start made the forward
stream wait EVERY early layer's event. Now an extend promotes only the layers
whose rows already landed; every other layer lands in run_waves /
run_eager_pool (land_deferred_rows), the stream waiting THAT layer's event.
A decode (graph replay, no land call) keeps the full wait.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo

_HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "_rearm_defer_host_p_0929", _HERE / "test_weg2_rearm_defer_host_p_0929.py")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


@pytest.fixture
def ops(monkeypatch):
    o = base._Ops()
    monkeypatch.setattr(eo, "_DEFERRED_ROWS_FILL", eo.DeferredRowsFill(stream_ops=o))
    yield o


def _decode():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))


def test_switch_is_on_by_default():
    assert envs.SGLANG_WEG2_ENABLE_REARM_DEFER_PER_LAYER.get() is True


def test_the_first_extend_does_not_wait_rows_still_in_flight(ops):
    a, b = base._cache(layer_id=0, seed=1), base._cache(layer_id=1, seed=2)
    a.rearm_after_wake(defer=base.HOST)
    b.rearm_after_wake(defer=base.HOST)
    fill = eo.deferred_rows_fill()
    fill.start()
    assert len(ops.events) == 2
    eo.deferred_rows_tick(base._extend())  # P's first forward
    assert ops.waited == []                 # the forward stream waits nothing yet
    assert fill.pending == [a, b]
    # layer 0's MoE: lands its own rows, waits only its own event
    assert a.land_deferred_rows() is True
    assert ops.waited == [ops.events[0]]
    assert b._deferred_rows is not None
    assert b.land_deferred_rows() is True
    assert ops.waited == ops.events and not fill.pending
    ref_a, ref_b = base._serial(layer_id=0, seed=1), base._serial(layer_id=1, seed=2)
    assert base._rows_digest(a) == base._rows_digest(ref_a)
    assert base._rows_digest(b) == base._rows_digest(ref_b)


def test_landed_rows_are_promoted_at_the_tick(ops):
    ops.done = True  # the copies already finished
    a = base._cache(layer_id=0)
    a.rearm_after_wake(defer=base.HOST)
    eo.deferred_rows_fill().start()
    eo.deferred_rows_tick(base._extend())
    assert a._deferred_rows is None and not eo.deferred_rows_fill().pending


def test_a_decode_keeps_the_full_wait(ops):
    a = base._cache(layer_id=0)
    a.rearm_after_wake(defer=base.HOST)
    eo.deferred_rows_fill().start()
    eo.deferred_rows_tick(_decode())
    assert ops.waited == ops.events and a._deferred_rows is None


def test_pool_layer_lands_in_run_eager_pool_path(ops):
    c = base._cache(pool=True)
    c.rearm_after_wake(defer=base.HOST)
    eo.deferred_rows_fill().start()
    eo.deferred_rows_tick(base._extend())
    assert c._deferred_rows is not None  # still open: the layer lands it
    assert c.land_deferred_rows() is True
    ref = base._serial(pool=True)
    assert base._rows_digest(c) == base._rows_digest(ref)
    for name in ("hot_phys", "host_row"):
        assert __import__("torch").equal(getattr(c._pool_tables, name), getattr(ref._pool_tables, name))
