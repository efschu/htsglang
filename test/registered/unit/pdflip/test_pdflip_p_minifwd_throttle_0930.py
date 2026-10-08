"""P-MINIFWD-THROTTLE (30.09.): the 16k chunk right behind a lone rest forward
on PP0, and the hold that was meant to prevent the rest, both at the metal.

(1) THE HOLD NEVER RAN. y3v (5327bdfa17, ...0930_005810) printed
    ``P-MINIFWD-HOLD skipped (RuntimeError: Boolean value of Tensor with more
    than one value is ambiguous)`` 15x (once "with no values"), y3w 4x, and
    ``PDFLIP P-MINIFWD-HOLD`` 0x: ``_minifwd_rest`` read the carried request's
    ``prefix_indices`` -- a device TENSOR on the metal, a list in the
    d468ab19c7 tests -- through ``x or ()``, and the adapter's except
    swallowed the TypeError-shaped RuntimeError on every pass that got there.

(2) THE STALL. PP0 16k chunks launched straight behind a lone rest forward
    while PP1 still ran the previous 16k chunk fetched their experts at
    312-359 us instead of 192-193 us (y3u 7x, y3v 3x, y3w 1x; +0.91-1.26 s
    each, 11 of 11 such forwards, 0 of ~250 lockstep chunks). Every other
    segment of the forward is unchanged; the extra time equals the time the
    rest's hidden states waited for PP1 minus a FIXED offset into the forward
    (bs=1: 0.24-0.26 s; bs=2: 0.45-0.48 s, the difference = the bs=2 PLE
    gather). That is a stall, not a slower link: the joined fetch ran on the
    layer's side stream (29 per-layer streams on PP0) and every one of them
    shares one of the device's 8 hardware queues (CUDA_DEVICE_MAX_CONNECTIONS
    = 8, engine.py) with the PP send stream, whose isend of the rest sits
    there until PP1 posts its receive. A joined fetch gains nothing from the
    side stream (the side stream waits for the forward, the forward for the
    side stream), so it now runs on the forward's own stream.

Hermetic, CPU (CUDA_VISIBLE_DEVICES=''), stream calls recorded by fakes.
Red on c378ba4002, green with the fix.
"""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.pdflip import p_minifwd_hold as mh  # noqa: E402
from test_pdflip_p_minifwd_0930 import (  # noqa: E402,F401  (fixtures by name)
    _p_form,
    _pass_scheduler,
    _queue,
)


@pytest.fixture(autouse=True)
def _fresh_hold():
    mh.reset()
    yield
    mh.reset()


# ------------------------------------------------------------ (1) the hold
@pytest.mark.parametrize(
    "prefix",
    [torch.arange(16384, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)],
    ids=["tensor-many (y3v 14x, y3w 4x)", "tensor-empty (y3v 1x)"],
)
def test_the_rest_reads_a_tensor_prefix(_p_form, prefix):
    """The metal form: ``prefix_indices`` is a tensor. Base: RuntimeError."""
    s = _pass_scheduler(polls=3)
    s.chunked_req.prefix_indices = prefix
    assert told._minifwd_rest(s) == 16450 - 16384


def test_the_hold_fires_on_the_metal_form(_p_form, caplog):
    """y3r pdflip-62-92 with the metal's tensor prefix: the waiter's told rides
    the rest's pass. Base: 'P-MINIFWD-HOLD skipped (RuntimeError ...)', no
    told on the wire, the rest runs alone."""
    s = _pass_scheduler(polls=3)
    s.chunked_req.prefix_indices = torch.arange(16384, dtype=torch.int64)
    _queue(s)
    mh.note_forward(67, 1, 988.3)  # y3v 01:11:29, the lone 67-token rest
    with caplog.at_level("WARNING"):
        wire = told.pp0_publish(s, ["x"])
    assert "skipped" not in caplog.text
    assert [type(w).__name__ for w in wire] == ["str", "PdFlipStoreTold"]
    assert mh._STATE["told"] == 1


# ------------------------------------------------------------ (2) the stall
class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))

    def __repr__(self):
        return self.name


@pytest.fixture
def streams(monkeypatch):
    """The forward's stream and a layer side stream, every call recorded."""
    log = []
    fwd = _Stream("forward", log)
    cur = [fwd]

    @contextmanager
    def _ctx(stream):
        log.append(("enter", stream.name))
        prev, cur[0] = cur[0], stream
        try:
            yield
        finally:
            cur[0] = prev

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: cur[0])
    monkeypatch.setattr(torch.cuda, "stream", _ctx)
    monkeypatch.setenv("FLLIPER_MOE_OFFLOAD_FETCH", "memcpy")  # CPU copies
    return SimpleNamespace(log=log, fwd=fwd)


E, R, C, W = 12, 2, 4, 3


def _cache(monkeypatch, log):
    monkeypatch.setenv("FLLIPER_MOE_SCRATCH_SLOTS", str(C))
    layer = SimpleNamespace(num_local_experts=E, layer_id=2,
                            moe_runner_config=SimpleNamespace(routed_scaling_factor=1.0))
    cache = eo.MoEExpertOffloadCache(layer, R / E)
    spill = torch.arange(E - R, dtype=torch.float32).unsqueeze(1).repeat(1, W) + R
    cache._pinned = {"w13": spill}
    cache._resident = {"w13": torch.full((R + C, W), -1.0)}
    cache._installed = True
    cache._stream = _Stream("side", log)  # a CUDA rank has one per layer
    return cache


def _side_touched(log):
    return [e for e in log if "side" in e]


def test_a_joined_fetch_never_enters_the_side_stream(monkeypatch, streams):
    """The expert-major wave's fetch (join=True): no command on the layer's
    side stream, so nothing parked on its hardware queue can hold it. Base:
    wait(side<-forward), enter(side), wait(forward<-side)."""
    cache = _cache(monkeypatch, streams.log)
    cache._fetch([(5, R), (7, R + 1)])
    assert _side_touched(streams.log) == []
    bank = cache._resident["w13"]
    assert bank[R].tolist() == [5.0] * W and bank[R + 1].tolist() == [7.0] * W


def test_a_lookahead_still_overlaps_on_the_side_stream(monkeypatch, streams):
    """WP8 (join=False) keeps its overlap, and the next joined fetch of the
    same cache still waits for those copies before compute reads them."""
    cache = _cache(monkeypatch, streams.log)
    cache._fetch([(9, R + 2)], join=False)
    assert ("enter", "side") in streams.log
    assert ("wait", "forward", "side") not in streams.log  # left in flight
    del streams.log[:]
    cache._fetch([(5, R)])
    assert ("wait", "forward", "side") in streams.log  # made visible
    del streams.log[:]
    cache._fetch([(6, R + 1)])  # nothing in flight any more
    assert _side_touched(streams.log) == []
    assert cache._resident["w13"][R + 2].tolist() == [9.0] * W


def test_the_marker_counts_forward_stream_fetches(monkeypatch, streams):
    n0 = eo._FORWARD_STREAM_FETCH["n"]
    cache = _cache(monkeypatch, streams.log)
    cache._fetch([(5, R)])
    cache._fetch([(6, R)])
    assert eo._FORWARD_STREAM_FETCH["n"] == n0 + 2
