"""#829: an expired HiCache deadline must not kill collectives on healthy peers.

THE DEFECT, and it is #630's second-order cost. ``bounded_wait`` bounded its
wait by handing the deadline to the ``Work``::

    completed = work.wait(timeout=datetime.timedelta(seconds=timeout_s))

That was the ONLY timed ``Work.wait`` on a gloo ``Work`` anywhere in
``sglang/srt`` -- every other ``.wait(timeout=`` in the tree is a
``threading.Event`` or a ``Popen`` -- and #630 (e4f1ae2556, 2026-08-17 13:53)
introduced it. An expired ``wait(timeout=...)`` CLOSES THE GLOO PAIR, measured
hermetically by #824 W4 and written up on ``ParkedWait``: the waiter then gets
"Application timeout caused pair closure" and the PEER gets "Connection closed
by peer" from its next send.

WHY THAT IS A CONTRACT VIOLATION AND NOT A TRADE-OFF. ``bounded_recv``'s
docstring calls the raise terminal -- "the process is on its way down" -- which
is a RANK-LOCAL promise. The pair is shared, so it is not containable: this
rank's deadline kills collectives on peers that are healthy and are waiting on
nothing of ours. 34 of 262 boot logs carry the peer's half, always at a BARE
``work.wait()`` in the PP tensor-dict transport, and those victims run on the
SAME gloo context this function does -- ``kv_cache_builder.py:230`` passes
``pp_cache_group=pp_group.cpu_group``, which is the group
``parallel_state.recv_object`` (:2130-2132) posts its ``irecv`` on.

THE TEST THAT MATTERS IS THE TWO-PROCESS ONE. A stub cannot fail the way this
failed: the pair closure happens inside gloo, below anything a mock observes.
``TheGlooPairMustSurviveAnExpiredDeadline`` therefore drives REAL gloo in two
real processes and asserts that a collective issued AFTER a bounded_wait
expiry still completes. It is red on the shipped implementation and green on
ParkedWait, which is the whole ticket in one assertion.

The unit tests around it pin the three contracts the conversion must NOT
change: the terminal raise, the ``_timeout_message`` wording, and the #734
numeric dead-peer-vs-timeout discriminator.

Hermetic: CPU gloo only, CUDA_VISIBLE_DEVICES="". No serving, no boot.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

from sglang.srt.mem_cache.hicache_collective import (
    HiCacheCollectiveError,
    HiCacheCollectiveTimeoutError,
    bounded_wait,
)

# --- the two-process gloo child -------------------------------------------
#
# rank 0 posts a receive rank 1 never satisfies, lets the bound expire, and
# then issues an all_reduce that rank 1 DOES join. If the expiry poisoned the
# pair, that all_reduce raises and the child prints PAIR_DEAD.
_CHILD = textwrap.dedent(
    '''
    import os, sys, time, torch, torch.distributed as dist

    rank = int(sys.argv[1])
    init_file = sys.argv[2]
    dist.init_process_group(
        backend="gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=2,
    )
    group = dist.new_group([0, 1], backend="gloo")

    from sglang.srt.mem_cache.hicache_collective import (
        bounded_wait,
        HiCacheCollectiveTimeoutError,
    )

    if rank == 0:
        buf = torch.zeros(1, dtype=torch.long)
        # Rank 1 never sends this tag, so the bound must expire.
        work = dist.irecv(buf, src=1, group=group, tag=4242)
        expired = False
        try:
            bounded_wait(work, "probe/recv<-1", 1.0, "rank=0/2")
        except HiCacheCollectiveTimeoutError:
            expired = True
        except BaseException as exc:
            print("UNEXPECTED_ERROR:%r" % (exc,), flush=True)
            os._exit(3)
        if not expired:
            print("NO_EXPIRY", flush=True)
            os._exit(4)

        # THE ASSERTION. A collective issued after the expiry, on the same
        # group and therefore the same gloo pair, with a peer that is healthy.
        try:
            t = torch.ones(1, dtype=torch.long)
            dist.all_reduce(t, group=group)
            print("PAIR_SURVIVED:%d" % int(t.item()), flush=True)
        except BaseException as exc:
            print("PAIR_DEAD:%r" % (exc,), flush=True)
            os._exit(5)
    else:
        # Healthy peer: waits out rank 0's deadline, then joins the collective.
        time.sleep(3.0)
        try:
            t = torch.ones(1, dtype=torch.long)
            dist.all_reduce(t, group=group)
            print("PEER_OK", flush=True)
        except BaseException as exc:
            print("PEER_DEAD:%r" % (exc,), flush=True)
            os._exit(6)
    os._exit(0)
    '''
)


class _StubWork:
    """Records HOW it was waited on. The distinction is the whole ticket."""

    def __init__(self, *, unblock_after: float = None, raise_after: float = None):
        self.timed_wait_calls = 0
        self.unbounded_wait_calls = 0
        self._unblock_after = unblock_after
        self._raise_after = raise_after
        self._released = threading.Event()

    def wait(self, *args, **kwargs):
        timeout = kwargs.get("timeout", args[0] if args else None)
        if timeout is not None:
            # THE PAIR-DESTROYING CALL. Counting it is how the red-first
            # assertion works: after #829 nothing may reach this branch.
            self.timed_wait_calls += 1
            raise RuntimeError("gloo: wait timeout (stub) -- pair closed")
        self.unbounded_wait_calls += 1
        if self._raise_after is not None:
            time.sleep(self._raise_after)
            raise RuntimeError("gloo: Connection closed by peer (stub)")
        if self._unblock_after is not None:
            time.sleep(self._unblock_after)
            return True
        # Real gloo blocks here forever when the peer never sends.
        self._released.wait()
        return True


class TheDeadlineMustNotBeHandedToTheWork(unittest.TestCase):
    def test_an_expiry_never_calls_the_timed_wait(self):
        work = _StubWork()
        with self.assertRaises(HiCacheCollectiveTimeoutError):
            bounded_wait(work, "probe/expiry", 0.3, "rank=0/2")
        self.assertEqual(
            work.timed_wait_calls,
            0,
            "the deadline was handed to the Work, which is the call that "
            "closes the gloo pair and kills healthy peers",
        )
        self.assertGreaterEqual(work.unbounded_wait_calls, 1)

    def test_the_expiry_message_is_unchanged(self):
        work = _StubWork()
        with self.assertRaises(HiCacheCollectiveTimeoutError) as ctx:
            bounded_wait(work, "pp_sync/isend[0]->pp1", 0.3, "pp_rank=1/3")
        msg = str(ctx.exception)
        self.assertIn("pp_sync/isend[0]->pp1", msg)
        self.assertIn("pp_rank=1/3", msg)
        self.assertIn("waited", msg)
        self.assertIn("SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S", msg)

    def test_a_healthy_wait_returns_and_is_unbounded(self):
        work = _StubWork(unblock_after=0.05)
        self.assertIsNone(bounded_wait(work, "probe/ok", 5.0, "rank=0/2"))
        self.assertEqual(work.timed_wait_calls, 0)
        self.assertEqual(work.unbounded_wait_calls, 1)

    def test_the_734_discriminator_still_names_a_dead_peer(self):
        """A transport failure well inside the bound is NOT a timeout."""
        work = _StubWork(raise_after=0.05)
        with self.assertRaises(HiCacheCollectiveError) as ctx:
            bounded_wait(work, "pp_sync/isend[2]->pp2", 10.0, "pp_rank=2/3")
        msg = str(ctx.exception)
        self.assertNotIsInstance(ctx.exception, HiCacheCollectiveTimeoutError)
        self.assertIn("NOT a timeout", msg)
        self.assertIn("dead rank, not a slow one", msg)

    def test_the_documented_escape_hatch_still_blocks_raw(self):
        work = _StubWork(unblock_after=0.01)
        self.assertIsNone(bounded_wait(work, "probe/raw", 0, "rank=0/2"))
        self.assertEqual(work.timed_wait_calls, 0)
        self.assertEqual(work.unbounded_wait_calls, 1)

    def test_a_none_work_is_a_completed_noop(self):
        self.assertIsNone(bounded_wait(None, "probe/none", 1.0, "rank=0/1"))


class TheGlooPairMustSurviveAnExpiredDeadline(unittest.TestCase):
    """REAL two-process gloo. A stub cannot fail the way this failed."""

    def test_a_collective_after_an_expiry_still_completes(self):
        import torch.distributed as dist

        if not dist.is_gloo_available():  # pragma: no cover - env guard
            self.skipTest("gloo backend unavailable")

        with tempfile.TemporaryDirectory() as td:
            script = os.path.join(td, "child.py")
            with open(script, "w") as fh:
                fh.write(_CHILD)
            init_file = os.path.join(td, "rendezvous")

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["GLOO_SOCKET_IFNAME"] = env.get("GLOO_SOCKET_IFNAME", "lo")

            procs = [
                subprocess.Popen(
                    [sys.executable, script, str(r), init_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                )
                for r in (0, 1)
            ]
            outs = []
            try:
                for p in procs:
                    out, err = p.communicate(timeout=180)
                    outs.append((p.returncode, out, err))
            finally:
                for p in procs:
                    if p.poll() is None:
                        p.kill()

            rc0, out0, err0 = outs[0]
            self.assertNotIn(
                "PAIR_DEAD",
                out0,
                "the expired deadline closed the gloo pair: a healthy peer's "
                "collective failed afterwards. Tail: " + (err0 or "")[-800:],
            )
            self.assertIn(
                "PAIR_SURVIVED",
                out0,
                f"rank 0 rc={rc0} out={out0!r} err={(err0 or '')[-800:]}",
            )
            self.assertEqual(rc0, 0, f"rank 0 err: {(err0 or '')[-800:]}")


if __name__ == "__main__":
    unittest.main()
