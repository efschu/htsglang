"""upstream #36738: the HiCache load-back H2D waits for the in-flight forward.

Hermetic, CPU (fake streams). Under the overlap scheduler (27B D group) a device
page reclaimed for a load-back can still be written by the forward in flight on
the forward stream; the producer start event only orders the load after the
schedule stream. Pins:

* both consume points -- ``HiCacheController.start_loading`` and the hybrid
  override ``HybridCacheController.start_loading`` (the path the 27B/NF hybrid
  UnifiedRadixCache runs) -- issue ``load_stream.wait_stream(<fence>)`` BEFORE
  the first per-layer H2D, for every bound fence stream, and nothing when no
  fence is bound;
* ``Scheduler._bind_hicache_load_fence`` binds the scheduler's
  ``forward_stream`` (plus the spill lane's ``spill_stream`` when leased), and
  is wired after tree-cache init and again in ``init_overlap``.
"""

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.managers.cache_controller as cc
import sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller as hcc
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)


class _Stream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, other):
        self.log.append(("wait_stream", self.name, other.name))


class _Event:
    def __init__(self, log):
        self.log = log

    def record(self, *a, **k):
        self.log.append(("record",))

    def wait(self, stream):
        self.log.append(("wait_event", stream.name))


class _Producer:
    def __init__(self, log):
        self.start_event = _Event(log)
        self.finish_event = _Event(log)

    def complete(self, i):
        pass


class _Counter:
    def __init__(self, log):
        self.events = {0: _Producer(log)}

    def update_producer(self):
        return 0


class _HostPool:
    transfer_layer_domain = 2

    def __init__(self, log):
        self.log = log

    def load_to_device_per_layer(self, *a, **k):
        self.log.append(("h2d",))


@pytest.fixture
def fakes(monkeypatch):
    log = []

    @contextlib.contextmanager
    def _stream_ctx(s):
        yield s

    fake_dm = SimpleNamespace(stream=_stream_ctx)
    op = SimpleNamespace(
        host_indices=torch.arange(4),
        device_indices=torch.arange(4),
        node_ids=[1],
        pool_transfers=None,
    )
    fake_op_cls = SimpleNamespace(merge_ops=lambda q: op)
    for mod in (cc, hcc):
        monkeypatch.setattr(mod, "device_module", fake_dm)
        monkeypatch.setattr(mod, "consume_gate", lambda *a, **k: True)
        monkeypatch.setattr(mod, "CacheOperation", fake_op_cls)
    return log, op


def _common(ctl, log, op):
    ctl.load_queue = [op]
    ctl.ack_load_queue = []
    ctl.layer_done_counter = _Counter(log)
    ctl.load_stream = _Stream("load", log)
    ctl.mem_pool_host = _HostPool(log)
    ctl.mem_pool_device = SimpleNamespace()
    ctl.io_backend = "direct"
    ctl.layer_num = 2
    ctl.draft_tier_armed = lambda direction: False
    ctl._dcp_kv_transfer_pairs = lambda h, d: (h, d)
    return ctl


def _hi(log, op):
    ctl = _common(HiCacheController.__new__(HiCacheController), log, op)
    ctl.move_indices = lambda h, d: (h, d)
    return ctl


def _hybrid(log, op):
    ctl = _common(HybridCacheController.__new__(HybridCacheController), log, op)
    ctl.move_hybrid_indices = lambda o: (o.host_indices, o.device_indices, None)
    ctl._record_transfer_indices_on_stream = lambda *a, **k: None
    return ctl


@pytest.mark.parametrize("make", [_hi, _hybrid], ids=["HiCache", "Hybrid"])
def test_load_waits_for_forward_before_first_h2d(fakes, make):
    log, op = fakes
    ctl = make(log, op)
    ctl.load_fence_stream = _Stream("forward", log)
    assert ctl.start_loading() == 0
    assert ("wait_stream", "load", "forward") in log
    first_h2d = log.index(("h2d",))
    assert log.index(("wait_stream", "load", "forward")) < first_h2d
    assert len(ctl.ack_load_queue) == 1


@pytest.mark.parametrize("make", [_hi, _hybrid], ids=["HiCache", "Hybrid"])
def test_every_bound_stream_is_waited(fakes, make):
    log, op = fakes
    ctl = make(log, op)
    ctl.load_fence_stream = (_Stream("forward", log), _Stream("spill", log))
    ctl.start_loading()
    first_h2d = log.index(("h2d",))
    for name in ("forward", "spill"):
        assert log.index(("wait_stream", "load", name)) < first_h2d


@pytest.mark.parametrize("make", [_hi, _hybrid], ids=["HiCache", "Hybrid"])
def test_no_fence_bound_means_no_wait(fakes, make):
    log, op = fakes
    ctl = make(log, op)
    ctl.load_fence_stream = None
    ctl.start_loading()
    assert not [e for e in log if e[0] == "wait_stream"]
    assert ("h2d",) in log


def _sched(**kw):
    from sglang.srt.managers.scheduler import Scheduler

    ns = SimpleNamespace(**kw)
    Scheduler._bind_hicache_load_fence(ns)
    return ns


def test_scheduler_binds_forward_stream():
    ctl = SimpleNamespace(load_fence_stream=None)
    fs = object()
    _sched(
        enable_hierarchical_cache=True,
        tree_cache=SimpleNamespace(cache_controller=ctl),
        forward_stream=fs,
    )
    assert ctl.load_fence_stream is fs


def test_scheduler_binds_spill_lane_too():
    ctl = SimpleNamespace(load_fence_stream=None)
    fs, ss = object(), object()
    _sched(
        enable_hierarchical_cache=True,
        tree_cache=SimpleNamespace(cache_controller=ctl),
        forward_stream=fs,
        spill_stream=ss,
    )
    assert ctl.load_fence_stream == (fs, ss)


def test_scheduler_bind_is_inert_without_hicache():
    ctl = SimpleNamespace(load_fence_stream=None)
    _sched(
        enable_hierarchical_cache=False,
        tree_cache=SimpleNamespace(cache_controller=ctl),
        forward_stream=object(),
    )
    assert ctl.load_fence_stream is None
    # no controller (plain radix cache) -> no error
    _sched(
        enable_hierarchical_cache=True,
        tree_cache=SimpleNamespace(),
        forward_stream=object(),
    )


def test_scheduler_wires_the_bind():
    from sglang.srt.managers import scheduler as sched_mod

    src = inspect.getsource(sched_mod)
    anchor = src.index("self.tree_cache = result.tree_cache")
    assert "self._bind_hicache_load_fence()" in src[anchor : anchor + 400]
    init_overlap = inspect.getsource(sched_mod.Scheduler.init_overlap)
    spill = init_overlap.index("self.spill_stream = self.device_module.Stream()")
    assert init_overlap.index("self._bind_hicache_load_fence()") > spill
