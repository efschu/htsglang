"""Falsifiers for #259 -- a HiCache control collective must not wedge.

Bug (reproduced): one TP rank died of OOM; the survivor then sat permanently in

    all_reduce            torch/distributed/distributed_c10d.py
    _all_reduce_attn_groups   unified_radix_cache.py
    drain_storage_control_queues
    check_hicache_events
    _get_new_batch_prefill_raw    scheduler.py

Two independent defects on that one path, both from the known
"rank-local condition before a group collective" family:

1. Dead peer -> infinite wait. The groups these collectives run on are the
   gloo ``cpu_group``s, created with a 2h default timeout that nothing on this
   path shortens, so a dead-but-not-closed peer parks the survivor for hours.
   The survivor must abort with a NAMED error instead.

2. Rank-dependent vector length. ``drain_storage_control_queues`` sized its
   reduced tensor from ``list(cc.extra_host_mem_release_queues)`` -- a dict
   built from THIS rank's host-pool entries. Ranks with different sidecar-pool
   sets enter the same all_reduce with different ``numel``; gloo does not
   reject that, it wedges.
"""

import threading
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_radix_cache import (
    HiCacheCollectiveError,
    HiCacheCollectiveTimeoutError,
    UnifiedRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeGroup:
    """Stand-in for a gloo ProcessGroup; only identity is ever used."""


class _FakeWork:
    """torch.distributed.Work stand-in with controllable completion."""

    def __init__(self, completes: bool = True):
        self._completes = completes
        self.wait_calls = 0

    def is_completed(self) -> bool:
        return self._completes

    def wait(self, *args, **kwargs):
        self.wait_calls += 1
        if self._completes:
            return True
        timeout = kwargs.get("timeout", args[0] if args else None)
        if timeout is None:
            # The real UNTIMED wait against a dead peer: never returns.
            threading.Event().wait()
            return True
        # A TIMED wait against a dead peer: gloo raises when the deadline
        # expires (measured 2026-08-17). The stub modelled only the untimed
        # case because `bounded_wait` used to poll `is_completed()` and never
        # handed a deadline to `wait()` -- the polling that turned out to be
        # the #630 livelock, since `is_completed()` reports while `wait()`
        # drives. See test_pp_sync_rendezvous_630.py.
        # #734: a timeout raises AT its deadline; raising instantly reads as
        # a dead peer to the discriminator, which is a different defect.
        import time as _t

        _t.sleep(float(timeout.total_seconds()) if hasattr(timeout, "total_seconds") else float(timeout))
        raise RuntimeError("gloo: wait timeout (stub)")


def _holder(*, timeout_s=0.05, tp_world_size=2):
    """Minimal carrier exposing only what the collective helpers touch."""
    h = types.SimpleNamespace(
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=_FakeGroup(),
        tp_world_size=tp_world_size,
        collective_timeout_s=timeout_s,
    )
    h._wait_bounded = types.MethodType(UnifiedRadixCache._wait_bounded, h)
    h._all_reduce_attn_groups = types.MethodType(
        UnifiedRadixCache._all_reduce_attn_groups, h
    )
    h._barrier_attn_groups = types.MethodType(UnifiedRadixCache._barrier_attn_groups, h)
    return h


class TestBoundedCollective(unittest.TestCase):
    """RED before the fix: these calls never return."""

    def test_dead_peer_all_reduce_raises_named_error(self):
        h = _holder()
        work = _FakeWork(completes=False)
        with mock.patch.object(
            torch.distributed, "all_reduce", return_value=work
        ), mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            with self.assertRaises(HiCacheCollectiveTimeoutError) as ctx:
                h._all_reduce_attn_groups(
                    torch.tensor([1], dtype=torch.int),
                    torch.distributed.ReduceOp.MIN,
                    label="drain_storage_control_queues",
                )
        msg = str(ctx.exception)
        # The error must name the call site so the wedge is diagnosable from
        # the log line alone, without a py-spy stack.
        self.assertIn("drain_storage_control_queues", msg)
        self.assertIn("SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S", msg)

    def test_dead_peer_barrier_raises_named_error(self):
        h = _holder()
        work = _FakeWork(completes=False)
        with mock.patch.object(
            torch.distributed, "barrier", return_value=work
        ), mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            with self.assertRaises(HiCacheCollectiveTimeoutError):
                h._barrier_attn_groups(label="release_aborted_request")

    def test_healthy_collective_completes_and_is_waited(self):
        h = _holder()
        work = _FakeWork(completes=True)
        with mock.patch.object(
            torch.distributed, "all_reduce", return_value=work
        ) as ar, mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            h._all_reduce_attn_groups(
                torch.tensor([1], dtype=torch.int), torch.distributed.ReduceOp.MIN
            )
        self.assertTrue(ar.call_args.kwargs["async_op"])
        self.assertEqual(work.wait_calls, 1)

    def test_single_rank_group_issues_no_collective(self):
        h = _holder(tp_world_size=1)
        with mock.patch.object(torch.distributed, "all_reduce") as ar:
            h._all_reduce_attn_groups(
                torch.tensor([1], dtype=torch.int), torch.distributed.ReduceOp.MIN
            )
        ar.assert_not_called()

    def test_timeout_disabled_falls_back_to_blocking_wait(self):
        h = _holder(timeout_s=0.0)
        work = _FakeWork(completes=True)
        with mock.patch.object(
            torch.distributed, "all_reduce", return_value=work
        ), mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            h._all_reduce_attn_groups(
                torch.tensor([1], dtype=torch.int), torch.distributed.ReduceOp.MIN
            )
        self.assertEqual(work.wait_calls, 1)


class _FakeQueue:
    def __init__(self, n: int):
        self._n = n

    def qsize(self) -> int:
        return self._n


class _FakeController:
    def __init__(self, extra: dict):
        self.prefetch_revoke_queue = _FakeQueue(1)
        self.ack_backup_queue = _FakeQueue(2)
        self.host_mem_release_queue = _FakeQueue(3)
        self.extra_host_mem_release_queues = extra


class TestRankUniformCollectiveShape(unittest.TestCase):
    """RED before the fix: the two ranks reduce tensors of different numel."""

    def _drain_vector(self, extra: dict):
        captured = {}

        def _capture(tensor, op, label="hicache"):
            captured["numel"] = tensor.numel()
            captured["values"] = list(map(int, tensor.tolist()))

        h = types.SimpleNamespace(
            cache_controller=_FakeController(extra),
            _all_reduce_attn_groups=_capture,
            _drain_storage_control_queues_impl=lambda **kw: captured.update(impl=kw),
        )
        UnifiedRadixCache.drain_storage_control_queues(h)
        return captured

    def test_drain_vector_length_is_rank_invariant(self):
        # Rank 0 owns two sidecar host pools, rank 1 owns one -- the asymmetry
        # uneven DCP produces. The reduced vector must not notice.
        rank0 = self._drain_vector(
            {PoolName.MAMBA: _FakeQueue(4), PoolName.SWA: _FakeQueue(5)}
        )
        rank1 = self._drain_vector({PoolName.MAMBA: _FakeQueue(6)})
        self.assertEqual(rank0["numel"], rank1["numel"])

    def test_drain_vector_keeps_head_triple_and_local_pool_values(self):
        got = self._drain_vector({PoolName.MAMBA: _FakeQueue(4)})
        self.assertEqual(got["values"][:3], [1, 2, 3])
        self.assertEqual(got["impl"]["n_revoke"], 1)
        self.assertEqual(got["impl"]["n_backup"], 2)
        self.assertEqual(got["impl"]["n_release"], 3)
        self.assertEqual(got["impl"]["extra_release_counts"], {PoolName.MAMBA: 4})

    def test_pool_outside_the_fixed_universe_raises(self):
        with self.assertRaises(HiCacheCollectiveError):
            self._drain_vector({"not_a_known_pool": _FakeQueue(1)})


if __name__ == "__main__":
    unittest.main()
