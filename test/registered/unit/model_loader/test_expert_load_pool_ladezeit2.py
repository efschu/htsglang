"""Ladezeit 2 (23.09., fnFL2x26): the expert-shard consumer pool.

Measured with SGLANG_LOAD_PROFILE on fnFL2x26: 50 % of a 107 s rank load was
the per-shard strided host copy in FusedMoE._load_w13/_load_w2, serial on the
loader thread while the eight file workers waited. The pool takes those
calls off that thread. These cases pin what would load SILENTLY WRONG or
hang if a rewrite got it wrong: the bytes, the error path, the throttle,
and the serial form.
"""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.model_loader import load_consumer as lc


def _stacked_param(E=6, rows=64, cols=32):
    return torch.zeros(E, rows, cols, dtype=torch.int32)


def _shards(E=6, rows=64, cols=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    # checkpoint tensors come TRANSPOSED (llm-compressor packed layout): a
    # strided source is exactly what the serial path copied
    return {(e, s): torch.randint(-2**31, 2**31 - 1, (cols, rows // 2), generator=g,
                                   dtype=torch.int32)
            for e in range(E) for s in ("w1", "w3")}


def _load_shard(param, expert_id, shard_id, src):
    """The shape of FusedMoE._load_w13 for a w13 param: w1 -> first half of
    the rows, w3 -> second half, from a transposed source."""
    half = param.shape[1] // 2
    start = 0 if shard_id == "w1" else half
    param.data[expert_id].narrow(0, start, half).copy_(src.t())


def _fill(param, shards, pool):
    for (e, s), src in shards.items():
        pool.submit(_load_shard, param, e, s, src)
    pool.drain()
    pool.close()


def test_parallel_and_serial_form_write_identical_bytes():
    shards = _shards()
    serial = _stacked_param()
    with envs.SGLANG_LOAD_CONSUMER_THREADS.override(0):
        _fill(serial, shards, lc.ExpertLoadPool(lc.consumer_threads()))
    parallel = _stacked_param()
    _fill(parallel, shards, lc.ExpertLoadPool(4))
    assert torch.equal(serial, parallel)
    # and the serial form really wrote every shard (not a zero param)
    assert serial.abs().sum() > 0


def test_a_failing_shard_surfaces_at_drain_and_stops_further_submits():
    pool = lc.ExpertLoadPool(2)
    seen = []

    def bad(*_a):
        raise ValueError("shape 320 vs 20")

    def good(i):
        seen.append(i)

    pool.submit(bad)
    # give the failure a chance to land before the next submit
    for _ in range(200):
        if pool.completed >= 1:
            break
        time.sleep(0.005)
    with pytest.raises(RuntimeError, match="shape 320 vs 20"):
        pool.submit(good, 1)
    with pytest.raises(RuntimeError, match="refusing further loads"):
        pool.drain()
    pool.close()
    assert seen == []


def test_submit_blocks_when_two_times_threads_are_in_flight():
    """The throttle bounds what the deferred calls keep alive (mmaps, host
    RAM): a submit beyond 2 x threads must WAIT for a consumer."""
    gate = threading.Event()
    pool = lc.ExpertLoadPool(2)  # in_flight = 4

    def wait_for_gate():
        gate.wait(10)

    for _ in range(4):
        pool.submit(wait_for_gate)
    blocked = threading.Event()
    done = threading.Event()

    def fifth():
        blocked.set()
        pool.submit(wait_for_gate)
        done.set()

    t = threading.Thread(target=fifth, daemon=True)
    t.start()
    assert blocked.wait(2)
    assert not done.wait(0.3), "the fifth submit went through with 4 in flight"
    gate.set()
    assert done.wait(5)
    pool.drain()
    pool.close()
    assert pool.submitted == 5 and pool.completed == 5


def test_serial_form_runs_inline_on_the_calling_thread():
    pool = lc.ExpertLoadPool(0)
    tids = []
    pool.submit(lambda: tids.append(threading.get_ident()))
    assert tids == [threading.get_ident()]
    assert not pool.parallel
    pool.drain()


def test_the_env_default_is_four_consumers():
    """Bookkeeping: the default is the production form (Ladezeit 2 ON);
    a stray os.environ read or a flipped default would load serially again
    without any boot noticing."""
    with envs.SGLANG_LOAD_CONSUMER_THREADS.override(None):
        pass
    envs.SGLANG_LOAD_CONSUMER_THREADS.clear()
    assert lc.consumer_threads() == 4
    with envs.SGLANG_LOAD_CONSUMER_THREADS.override(-3):
        assert lc.consumer_threads() == 0
