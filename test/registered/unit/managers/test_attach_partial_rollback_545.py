"""#545: a partial attach must not leave the group half-attached.

THE STATE THIS PREVENTS. ``attach_hicache_storage`` fans out to every rank and
merges the results into ONE verdict. If rank 1 of 3 fails, the caller is told
"failed" -- and ranks 0 and 2 are still running storage threads with a backend
bound. Half-attached storage is exactly the state the detach contract exists to
prevent, and a clean-looking failure over it tells the operator the opposite of
what is true. The site carried ``# TODO: partial rollback if failed``.

THE RULE PINNED HERE: the terminal state is all-attached or all-detached, never
mixed. On a mixed result the coordinator detaches the group; if that rollback
itself fails anywhere, the response NAMES the stranded ranks rather than
reporting a clean failure.

WHY A GROUP DETACH RATHER THAN A TARGETED ONE. Detach is idempotent by
construction on both cache classes -- it asks the controller to clean up even
when ``enable_storage`` is already False, precisely so leftovers from a partial
attach are swept -- and the communicator offers no rank-addressed send. It is
also collective-free (local drain limits), so a rank whose peers have already
left a collective cannot hang on it.

WHY THE RANK FIELD WAS NEEDED. ``FanOutCommunicator.handle_recv`` appends
results in ARRIVAL order, so list position does not identify a rank. Without
``rank`` on the output, "ranks 0 and 2 are stranded" is unsayable.

Hermetic: the communicators are stubs; no scheduler, no server, no CUDA.
"""

import asyncio
import unittest

from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqOutput,
    DetachHiCacheStorageReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin


def _ok(rank):
    return AttachHiCacheStorageReqOutput(success=True, message="ok", rank=rank)


def _fail(rank, msg="boom"):
    return AttachHiCacheStorageReqOutput(success=False, message=msg, rank=rank)


class _Mgr:
    """Only what ``attach_hicache_storage`` touches."""

    def __init__(self, attach_results, detach_results=None):
        self._attach_results = attach_results
        self._detach_results = detach_results
        self.detach_calls = 0
        self.server_args = _ServerArgs()

    def auto_create_handle_loop(self):
        pass

    async def attach_hicache_storage_communicator(self, obj):
        return self._attach_results

    async def detach_hicache_storage_communicator(self, obj):
        self.detach_calls += 1
        if isinstance(self._detach_results, Exception):
            raise self._detach_results
        return self._detach_results

    attach_hicache_storage = TokenizerControlMixin.attach_hicache_storage


class _ServerArgs:
    def __init__(self):
        self.overrides = []

    def override(self, key, **kw):
        self.overrides.append((key, kw))


def _run(mgr):
    return asyncio.run(mgr.attach_hicache_storage("file"))


class TestMixedResultRollsBack(unittest.TestCase):
    def test_rank_1_of_3_failing_triggers_a_group_detach(self):
        """THE PIN. Ranks 0 and 2 attached; the group must not stay that way."""
        mgr = _Mgr(
            [_ok(0), _fail(1), _ok(2)],
            [
                DetachHiCacheStorageReqOutput(success=True, message="d", rank=r)
                for r in (0, 1, 2)
            ],
        )
        out = _run(mgr)
        self.assertEqual(mgr.detach_calls, 1, "the group was left half-attached")
        self.assertFalse(out.success, "rollback must never flip the verdict")

    def test_the_message_says_the_group_is_now_detached(self):
        mgr = _Mgr(
            [_ok(0), _fail(1), _ok(2)],
            [
                DetachHiCacheStorageReqOutput(success=True, message="d", rank=r)
                for r in (0, 1, 2)
            ],
        )
        out = _run(mgr)
        self.assertIn("fully detached", out.message)
        self.assertIn("[0, 2]", out.message, "the rolled-back ranks are named")

    def test_the_original_failure_is_still_reported(self):
        mgr = _Mgr(
            [_ok(0), _fail(1, "disk missing"), _ok(2)],
            [
                DetachHiCacheStorageReqOutput(success=True, message="d", rank=r)
                for r in (0, 1, 2)
            ],
        )
        out = _run(mgr)
        self.assertIn("disk missing", out.message, "rollback must not hide the cause")


class TestStrandedRanksAreNamed(unittest.TestCase):
    """A rollback that half-works is worse than one that fails loudly, unless
    it says exactly which processes are still holding a backend."""

    def test_a_failed_rollback_names_the_stranded_ranks(self):
        mgr = _Mgr(
            [_ok(0), _fail(1), _ok(2)],
            [
                DetachHiCacheStorageReqOutput(success=True, message="d", rank=0),
                DetachHiCacheStorageReqOutput(success=True, message="d", rank=1),
                DetachHiCacheStorageReqOutput(success=False, message="stuck", rank=2),
            ],
        )
        out = _run(mgr)
        self.assertIn("ROLLBACK INCOMPLETE", out.message)
        self.assertIn("[2]", out.message)
        self.assertIn("manually", out.message)
        self.assertFalse(out.success)

    def test_a_raising_rollback_is_reported_not_propagated(self):
        mgr = _Mgr([_ok(0), _fail(1)], RuntimeError("transport down"))
        out = _run(mgr)
        self.assertFalse(out.success)
        self.assertIn("ROLLBACK FAILED", out.message)
        self.assertIn("transport down", out.message)


class TestNoRollbackWhenThereIsNothingToUndo(unittest.TestCase):
    def test_all_ranks_failing_does_not_detach(self):
        """Nothing attached, so nothing to roll back -- a detach here would be
        an unnecessary group operation on a failed admin call."""
        mgr = _Mgr([_fail(0), _fail(1), _fail(2)])
        out = _run(mgr)
        self.assertEqual(mgr.detach_calls, 0)
        self.assertFalse(out.success)
        self.assertNotIn("ROLLBACK", out.message)

    def test_full_success_does_not_detach(self):
        mgr = _Mgr([_ok(0), _ok(1), _ok(2)])
        out = _run(mgr)
        self.assertEqual(mgr.detach_calls, 0)
        self.assertTrue(out.success)
        self.assertTrue(mgr.server_args.overrides, "success still records config")


class TestTheRankFieldExists(unittest.TestCase):
    """Without it, naming stranded ranks is unsayable: the fan-out collects
    results in arrival order, so list position is not a rank."""

    def test_attach_output_carries_a_rank_defaulting_to_unstamped(self):
        self.assertEqual(AttachHiCacheStorageReqOutput(success=True).rank, -1)

    def test_detach_output_carries_a_rank(self):
        self.assertEqual(DetachHiCacheStorageReqOutput(success=True).rank, -1)


if __name__ == "__main__":
    unittest.main()
