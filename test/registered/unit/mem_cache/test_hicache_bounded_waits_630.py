"""Falsifiers for #630 -- PP + disk HiCache must not wedge at warmup.

Bug (reproduced): with PP=3 and a disk HiCache backend, warmup never finished.
The health endpoint stayed 503 forever, no error, no crash. Two families of
unbounded blocking call on the HiCache control path:

1. ``_drain_async_work`` waits with a raw ``work.wait()`` loop over the isends
   posted by ``_pp_sync`` on the PP gloo ``cpu_group``. Its sibling
   ``_wait_bounded`` polls against a deadline; the asymmetry is the bug. The
   send side alone was thought to be self-unblocking because the LAST PP rank
   has an empty ``work_list`` -- but that reasoning never covered the receive
   side.

2. ``_pp_sync`` calls ``torch.distributed.recv`` with no bound. Every PP rank
   ABOVE the first blocks there indefinitely.

And the worse half: ``hiradix_cache`` -- which ``registry.py`` selects by
default whenever hierarchical cache is on and the model is not hybrid
SSM/SWA -- received NONE of the #259 bounded-wait work. Its
``_all_reduce_attn_groups``, ``_barrier_attn_groups``, ``_drain_async_work``
and ``_pp_sync`` were all raw blocking calls.

Nothing here is infinite in production: the gloo group timeout is
``timedelta(seconds=120 * 60)`` (parallel_state.py:619), so the wedge ends
after two hours with a generic RuntimeError from inside gloo naming no call
site. That is what "silent" meant.

Every test below runs the call under test in a daemon thread and joins with a
timeout, so on UNFIXED code the case FAILS with "never returned" instead of
hanging the runner.
"""

import threading
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_collective import (
    COLLECTIVE_POLL_MIN_S,
    COLLECTIVE_POLL_SPINS,
    HiCacheCollectiveTimeoutError,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# Deadline used by the holders: long enough that the spin window is crossed
# deliberately, short enough that the suite stays fast.
_TIMEOUT_S = 0.05
# How long a call gets to come back before we call it unbounded. Three orders
# of magnitude above _TIMEOUT_S, so a slow machine cannot produce a false
# "unbounded" verdict.
_JOIN_S = 10.0


class _FakeGroup:
    """Stand-in for a gloo ProcessGroup; only identity is ever used."""


class _WedgedWork:
    """A ``Work`` that never completes -- the dead/lagging peer.

    UNTIMED ``wait()`` blocks forever, exactly as the real one does against a
    peer that never posts the matching operation. Any test that reaches that
    branch has proven the call site is unbounded.

    TIMED ``wait(timeout=...)`` raises ``RuntimeError``, which is what gloo
    actually does when the deadline expires -- measured 2026-08-17 against a
    real gloo Work. The stub carried only the untimed behaviour because
    ``bounded_wait`` used to poll ``is_completed()`` and never handed a deadline
    to ``wait()`` at all. That polling was the #630 livelock (see
    test_pp_sync_rendezvous_630.py): ``is_completed()`` reports, ``wait()``
    drives, so two polling peers never advanced the exchange. Now that the
    bound is handed to the wait itself, a stub that ignored the timeout would
    hang this suite instead of exercising it.
    """

    def __init__(self):
        self.wait_calls = 0

    def is_completed(self) -> bool:
        return False

    def wait(self, *args, **kwargs):
        self.wait_calls += 1
        timeout = kwargs.get("timeout", args[0] if args else None)
        if timeout is None:
            threading.Event().wait()
            return True
        # #734: A TIMEOUT RAISES **AT** ITS DEADLINE. Raising instantly made
        # this stub indistinguishable from a dead peer, and the dead-peer
        # discriminator (elapsed well inside the bound => transport failure,
        # not expiry) correctly classified it as one. Sleeping the bound is
        # what a real expiring wait does, and it keeps this test exercising
        # the path it was written for.
        import time as _t

        _t.sleep(float(timeout.total_seconds()) if hasattr(timeout, "total_seconds") else float(timeout))
        raise RuntimeError("gloo: wait timeout (stub)")


class _CompletedWork:
    """A ``Work`` that is already done -- the healthy case."""

    def __init__(self):
        self.wait_calls = 0

    def is_completed(self) -> bool:
        return True

    def wait(self, *args, **kwargs):
        self.wait_calls += 1
        return True


def _run_bounded(fn):
    """Run ``fn`` in a daemon thread; return its raised exception.

    Returns ``None`` if it returned normally. Raises AssertionError if it did
    not come back at all -- that is the unfixed-code verdict, reported as a
    failure rather than a hung runner.
    """
    box = {}

    def _target():
        try:
            fn()
            box["returned"] = True
        except BaseException as exc:  # noqa: BLE001 - the exception IS the result
            box["exc"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(_JOIN_S)
    if t.is_alive():
        raise AssertionError(
            f"call did not return within {_JOIN_S}s against a peer that never "
            "completes: the wait is UNBOUNDED. In production this parks the "
            "rank until the gloo group's 2h timeout expires."
        )
    return box.get("exc")


def _unified_holder(timeout_s=_TIMEOUT_S, pp_rank=1, pp_size=3):
    h = types.SimpleNamespace(
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=_FakeGroup(),
        tp_world_size=2,
        pp_group=_FakeGroup(),
        pp_rank=pp_rank,
        pp_size=pp_size,
        work_list=[],
        collective_timeout_s=timeout_s,
    )
    for name in (
        "_wait_bounded",
        "_drain_async_work",
        "_pp_sync",
        "_all_reduce_attn_groups",
        "_barrier_attn_groups",
    ):
        setattr(h, name, types.MethodType(getattr(UnifiedRadixCache, name), h))
    return h


def _hiradix_holder(timeout_s=_TIMEOUT_S, pp_rank=1, pp_size=3):
    h = types.SimpleNamespace(
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=_FakeGroup(),
        tp_world_size=2,
        pp_group=_FakeGroup(),
        pp_rank=pp_rank,
        pp_size=pp_size,
        work_list=[],
        collective_timeout_s=timeout_s,
    )
    for name in (
        "_wait_bounded",
        "_drain_async_work",
        "_pp_sync",
        "_all_reduce_attn_groups",
        "_barrier_attn_groups",
    ):
        setattr(h, name, types.MethodType(getattr(HiRadixCache, name), h))
    return h


_HOLDERS = (("unified", _unified_holder), ("hiradix", _hiradix_holder))


class TestWedgedWorkStubIsFaithful(unittest.TestCase):
    """The stub must genuinely model the wedge, or every test below is vacuous."""

    def test_raw_wait_on_wedged_work_never_returns(self):
        work = _WedgedWork()
        with self.assertRaises(AssertionError) as ctx:
            _run_bounded(work.wait)
        self.assertIn("UNBOUNDED", str(ctx.exception))


class TestDrainAsyncWorkBounded(unittest.TestCase):
    """RED before the fix: _drain_async_work never returns, in BOTH classes."""

    def test_drain_async_work_raises_named_error(self):
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make()
                h.work_list = [_CompletedWork(), _WedgedWork()]
                exc = _run_bounded(h._drain_async_work)
                self.assertIsInstance(exc, HiCacheCollectiveTimeoutError)
                msg = str(exc)
                # Must name the collective, the specific work, the rank, and
                # the wait -- the whole point is diagnosability from one line.
                self.assertIn("pp_sync/isend", msg)
                self.assertIn("[1]", msg)
                self.assertIn("pp_rank=1/3", msg)
                self.assertIn("waited", msg)
                self.assertIn("SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S", msg)

    def test_healthy_drain_waits_each_work_once_and_clears(self):
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make()
                works = [_CompletedWork(), _CompletedWork()]
                h.work_list = list(works)
                self.assertIsNone(_run_bounded(h._drain_async_work))
                # Unchanged healthy behaviour: every work waited exactly once,
                # list drained.
                self.assertEqual([w.wait_calls for w in works], [1, 1])
                self.assertEqual(h.work_list, [])

    def test_timeout_disabled_restores_raw_blocking_wait(self):
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make(timeout_s=0.0)
                work = _CompletedWork()
                h.work_list = [work]
                self.assertIsNone(_run_bounded(h._drain_async_work))
                self.assertEqual(work.wait_calls, 1)


class TestPPSyncRecvBounded(unittest.TestCase):
    """RED before the fix: the recv side of _pp_sync has no bound at all.

    The earlier reasoning -- "the last PP rank has an empty work_list and
    unblocks the chain" -- covers only the SEND side, which is why this was
    missed.
    """

    def test_pp_sync_recv_raises_named_error(self):
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make(pp_rank=1, pp_size=3)
                data = torch.tensor([0], dtype=torch.int)
                with mock.patch.object(
                    torch.distributed, "irecv", return_value=_WedgedWork()
                ), mock.patch.object(
                    torch.distributed, "recv", side_effect=_never_returns
                ), mock.patch.object(
                    torch.distributed, "isend", return_value=_CompletedWork()
                ):
                    exc = _run_bounded(lambda: h._pp_sync(data))
                self.assertIsInstance(exc, HiCacheCollectiveTimeoutError)
                msg = str(exc)
                self.assertIn("pp_sync/recv<-pp0", msg)
                self.assertIn("pp_rank=1/3", msg)

    def test_pp_sync_recv_posts_the_same_operation_recv_would(self):
        """The irecv substitution must not change source, tag or group."""
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make(pp_rank=2, pp_size=3)
                data = torch.tensor([7], dtype=torch.int)
                with mock.patch.object(
                    torch.distributed, "irecv", return_value=_CompletedWork()
                ) as irecv, mock.patch.object(
                    torch.distributed, "isend", return_value=_CompletedWork()
                ):
                    self.assertIsNone(_run_bounded(lambda: h._pp_sync(data)))
                kwargs = irecv.call_args.kwargs
                self.assertIs(irecv.call_args.args[0], data)
                self.assertEqual(kwargs["group_src"], 1)
                self.assertIs(kwargs["group"], h.pp_group)
                from sglang.srt.distributed.communication_tags import P2PTag

                self.assertEqual(kwargs["tag"], P2PTag.HIRADIX_PP_SYNC)

    def test_first_pp_rank_posts_no_receive(self):
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make(pp_rank=0, pp_size=3)
                data = torch.tensor([1], dtype=torch.int)
                with mock.patch.object(
                    torch.distributed, "irecv"
                ) as irecv, mock.patch.object(
                    torch.distributed, "recv"
                ) as recv, mock.patch.object(
                    torch.distributed, "isend", return_value=_CompletedWork()
                ):
                    self.assertIsNone(_run_bounded(lambda: h._pp_sync(data)))
                irecv.assert_not_called()
                recv.assert_not_called()
                self.assertEqual(len(h.work_list), 1)

    def test_single_pp_group_is_a_noop(self):
        for label, make in _HOLDERS:
            with self.subTest(cls=label):
                h = make(pp_rank=0, pp_size=1)
                with mock.patch.object(torch.distributed, "irecv") as irecv:
                    h._pp_sync(torch.tensor([1], dtype=torch.int))
                irecv.assert_not_called()


def _never_returns(*args, **kwargs):
    """torch.distributed.recv stand-in: the unbounded blocking call."""
    threading.Event().wait()


class TestHiRadixCollectivesBounded(unittest.TestCase):
    """#630: hiradix_cache never received the #259 bounded-wait work."""

    def test_all_reduce_raises_named_error(self):
        h = _hiradix_holder()
        with mock.patch.object(
            torch.distributed, "all_reduce", return_value=_WedgedWork()
        ) as ar, mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            exc = _run_bounded(
                lambda: h._all_reduce_attn_groups(
                    torch.tensor([1], dtype=torch.int),
                    torch.distributed.ReduceOp.MIN,
                    label="drain_storage_control_queues",
                )
            )
        self.assertIsInstance(exc, HiCacheCollectiveTimeoutError)
        self.assertIn("drain_storage_control_queues", str(exc))
        # Bounded polling requires the async form; the blocking one cannot be
        # polled at all.
        self.assertTrue(ar.call_args.kwargs["async_op"])

    def test_barrier_raises_named_error(self):
        h = _hiradix_holder()
        with mock.patch.object(
            torch.distributed, "barrier", return_value=_WedgedWork()
        ) as ba, mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            exc = _run_bounded(
                lambda: h._barrier_attn_groups(label="release_aborted_request")
            )
        self.assertIsInstance(exc, HiCacheCollectiveTimeoutError)
        self.assertIn("release_aborted_request", str(exc))
        self.assertTrue(ba.call_args.kwargs["async_op"])

    def test_healthy_all_reduce_unchanged(self):
        h = _hiradix_holder()
        work = _CompletedWork()
        with mock.patch.object(
            torch.distributed, "all_reduce", return_value=work
        ), mock.patch.object(torch.distributed, "get_world_size", return_value=2):
            self.assertIsNone(
                _run_bounded(
                    lambda: h._all_reduce_attn_groups(
                        torch.tensor([1], dtype=torch.int),
                        torch.distributed.ReduceOp.MIN,
                    )
                )
            )
        self.assertEqual(work.wait_calls, 1)

    def test_single_rank_group_issues_no_collective(self):
        h = _hiradix_holder()
        h.tp_world_size = 1
        with mock.patch.object(torch.distributed, "all_reduce") as ar:
            h._all_reduce_attn_groups(
                torch.tensor([1], dtype=torch.int), torch.distributed.ReduceOp.MIN
            )
        ar.assert_not_called()


class TestHealthyPathCostsNoSyscall(unittest.TestCase):
    """The bound must not put a sleep in front of a completed collective."""

    def test_poll_spins_before_it_sleeps(self):
        # A completed Work is observed on the FIRST is_completed() call, so
        # neither branch below is reached. The spin window exists for the work
        # that completes a few microseconds late: COLLECTIVE_POLL_SPINS bare
        # loop iterations run before the first time.sleep, which covers a CPU
        # collective between local ranks by a wide margin.
        self.assertGreaterEqual(COLLECTIVE_POLL_SPINS, 512)
        self.assertLessEqual(COLLECTIVE_POLL_MIN_S, 0.001)

    def test_completed_work_never_sleeps(self):
        h = _unified_holder()
        h.work_list = [_CompletedWork() for _ in range(4)]
        with mock.patch("time.sleep") as slept:
            self.assertIsNone(_run_bounded(h._drain_async_work))
        slept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
