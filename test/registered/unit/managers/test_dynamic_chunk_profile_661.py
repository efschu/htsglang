# SPDX-License-Identifier: Apache-2.0
"""#661: PP0's profiling failure must not strand its peers in the broadcast.

THE BUG THIS PINS, MEASURED ON METAL 2026-08-11
------------------------------------------------
``profile_and_init_predictor`` profiles on PP0 only and ends in a
``pp_group.broadcast_object_list(..., src=0)`` that EVERY rank enters
unconditionally. Its caller, ``Scheduler.init_chunked_prefill``, wraps the
whole call in a rank-local ``try/except`` that disables dynamic chunking and
carries on.

Those two facts compose into a boot deadlock. On this rig, with
``--enable-dynamic-chunking`` on the PP=3 Route-A recipe, PP0 raised

    alloc_req_slots runs out of memory ... available_size()=4 (request slots)

under ``--max-running-requests 4``. PP0 caught it, set
``enable_dynamic_chunking = False`` FOR ITSELF, and walked into the event
loop; PP1 and PP2 blocked in ``broadcast_object_list`` waiting for a src that
had already left. py-spy confirmed all three stacks. The HTTP port never
opened and the boot had to be killed.

    A rank-local ``except`` around a collective is a deadlock waiting for the
    one rank that took a different branch.

THE FIX, AND WHAT THESE TESTS HOLD IT TO
-----------------------------------------
The failure is caught on the rank that can have it and published as DATA (an
empty sample set). Every rank still enters the broadcast, every rank receives
the same empty lists, and every rank raises the same error AFTER the
collective -- so the caller's ``except`` disables dynamic chunking on all
ranks, which is what it always meant to do.

The load-bearing test is :meth:`test_every_rank_enters_the_broadcast`, and it
is a real deadlock falsifier rather than an assertion about return values: the
fake ``broadcast_object_list`` is a ``threading.Barrier`` with a timeout, so a
rank that never arrives fails the test by TIMING OUT, which is exactly how the
bug presents in production.
"""

import contextlib
import threading
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_N = 3


class _FakePPGroup:
    """A pp_group whose broadcast is a BARRIER, so absence is observable."""

    def __init__(self, rank: int, barrier: threading.Barrier):
        self.rank = rank
        self._barrier = barrier
        self.is_first_rank = rank == 0
        self.entered = False

    def broadcast_object_list(self, data, src=0):
        self.entered = True
        # src publishes BEFORE the rendezvous, the others read after it, so
        # the payload cannot be observed half-written.
        if self.is_first_rank:
            _SHARED["data"] = [list(data[0]), list(data[1])]
        # A rank that never gets here leaves the others waiting -- which is
        # the production symptom, reproduced as a test failure.
        self._barrier.wait(timeout=5)
        if not self.is_first_rank:
            data[0], data[1] = _SHARED["data"]


_SHARED: dict = {}


class _StubScheduler:
    """Just enough surface for the function under test.

    ``tp_worker`` is deliberately ABSENT on PP0, so the profiling body raises
    the moment it reaches for the model runner. That is the point: this test
    does not care WHICH exception the profile hits, only that an exception
    there cannot strand the peers.
    """

    def __init__(self, rank: int, barrier: threading.Barrier):
        self.pp_group = _FakePPGroup(rank, barrier)
        self.chunked_prefill_size = 512
        self.length_predictor = None

        class _PS:
            pp_rank = rank
            attn_tp_size = 1
            attn_cp_size = 1

        self.ps = _PS()


def _run(rank: int, barrier: threading.Barrier, out: dict):
    from sglang.srt.managers.scheduler_pp_mixin import (
        SchedulerPPMixin,
    )

    stub = _StubScheduler(rank, barrier)
    # #815: `torch.distributed.is_available` / `is_initialized` are PROCESS
    # globals, so the save/restore for them belongs on the main thread (see
    # `_patched_distributed` in the driver) and must not happen per rank.
    # Racing it here made every rank save whatever the previous rank had
    # already installed: one rank restored the real function, the next
    # restored the lambda, and `is_initialized` stayed pinned to True for the
    # rest of the process. That leak later fell on
    # `test_session_branch_rewind_unit.py`, seven tests in a different
    # directory, with nothing in either file pointing at the other.
    try:
        SchedulerPPMixin.profile_and_init_predictor(stub)
        out[rank] = ("no-raise", None)
    except BaseException as e:  # noqa: BLE001 - the verdict is the payload
        out[rank] = (type(e).__name__, str(e)[:120])
    finally:
        out[f"entered{rank}"] = stub.pp_group.entered


class TestProfileFailureIsGroupUniform(CustomTestCase):
    @staticmethod
    @contextlib.contextmanager
    def _patched_distributed():
        """`is_available`/`is_initialized` forced True for all ranks at once.

        One save and one restore, on the thread that owns the workers, so the
        process global is handed back exactly as it was found.
        """
        import torch

        real_avail, real_init = (
            torch.distributed.is_available,
            torch.distributed.is_initialized,
        )
        torch.distributed.is_available = lambda: True
        torch.distributed.is_initialized = lambda: True
        try:
            yield
        finally:
            torch.distributed.is_available = real_avail
            torch.distributed.is_initialized = real_init

    def _drive(self):
        _SHARED.clear()
        barrier = threading.Barrier(_N)
        out: dict = {}
        threads = [
            threading.Thread(target=_run, args=(r, barrier, out)) for r in range(_N)
        ]
        with self._patched_distributed():
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        for t in threads:
            self.assertFalse(t.is_alive(), "a rank never returned -- deadlock")
        return out

    def test_every_rank_enters_the_broadcast(self):
        """The deadlock falsifier. PP0 fails; nobody is left in the barrier."""
        out = self._drive()
        for r in range(_N):
            self.assertTrue(
                out.get(f"entered{r}"),
                f"rank {r} never entered the broadcast -- this is the "
                f"deadlock the fix exists to prevent",
            )

    def test_every_rank_declines_together(self):
        """One verdict, the same on every rank, raised AFTER the collective."""
        out = self._drive()
        kinds = {out[r][0] for r in range(_N)}
        self.assertEqual(
            kinds,
            {"RuntimeError"},
            f"ranks disagreed about the outcome: {[out[r] for r in range(_N)]}",
        )
        for r in range(_N):
            self.assertIn("declines dynamic chunking together", out[r][1])

    def test_the_predictor_is_not_fitted_on_an_empty_sample_set(self):
        """Fitting nothing would dress a guess as a measurement."""
        self._drive()
        # The raise happens before ChunkSizePredictor is constructed; if a
        # successor moves the check below the fit, this catches it.
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        import inspect

        src = inspect.getsource(SchedulerPPMixin.profile_and_init_predictor)
        self.assertLess(
            src.index("declines dynamic chunking together"),
            src.index("ChunkSizePredictor("),
            "the empty-sample verdict must precede the fit",
        )


if __name__ == "__main__":
    unittest.main()
