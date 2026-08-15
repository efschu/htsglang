# SPDX-License-Identifier: Apache-2.0
"""#631: the arena commit reclaims torch's cache before it gives up.

THE CRASH THIS PINS, measured on metal 2026-08-09 12:47:45 (rank 1, a
3080, mixed acceptance load, POLICY=auto):

    RuntimeError: cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY
      ... _swap -> restore_backing -> finalize -> _back_spans
          -> commit_range -> cuMemCreate

followed by SIGQUIT and the loss of the whole instance.

The phase flip releases the source pool's physical pages and immediately
commits the destination's, and the swap site asserted in a comment that
this "cannot fail for want of memory" because boot sized the budget for
max(PP, TP). That reasoning silently assumes nothing else on the card
takes physical pages between boot and the flip. torch's caching allocator
does precisely that and by design never hands freed blocks back to the
driver, while the arena needs RAW driver pages that torch's cache cannot
serve. Under long prefills the reserve grows and the flip starves.

These tests are hermetic: they drive a fake driver, so they run on the
desk with no GPU and still pin the ORDER of operations that matters --
reclaim happens between the two attempts, not before the first one.
"""

import sys
import types
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase


class _Result:
    """Stand-in for the cuda-bindings CUresult enum members."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


SUCCESS = _Result("CUDA_SUCCESS")
OOM = _Result("CUDA_ERROR_OUT_OF_MEMORY")
INVALID = _Result("CUDA_ERROR_INVALID_VALUE")


class _FakeCUresult:
    CUDA_SUCCESS = SUCCESS
    CUDA_ERROR_OUT_OF_MEMORY = OOM
    CUDA_ERROR_INVALID_VALUE = INVALID


class _FakeDriver:
    CUresult = _FakeCUresult

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def cuMemCreate(self, step, prop, flags):  # noqa: N802 - driver name
        self.calls += 1
        return self._results.pop(0)


class TestArenaCommitReclaimsTorchCache(CustomTestCase):
    def setUp(self):
        # The module imports torch at top level; the reclaim path calls
        # torch.cuda.empty_cache / memory_reserved / memory_allocated only.
        self.events = []
        import sglang.srt.mem_cache.kv_vmm_backing as backing

        self.backing = backing

    def _patched(self, driver):
        """Patch the module's driver accessor and torch reclaim hooks."""
        events = self.events

        def _empty_cache():
            events.append(("empty_cache", driver.calls))

        return mock.patch.multiple(
            self.backing,
            _driver=lambda: driver,
            torch=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    empty_cache=_empty_cache,
                    memory_reserved=lambda *a: 7 << 30,
                    memory_allocated=lambda *a: 3 << 30,
                )
            ),
        )

    def test_happy_path_never_reclaims(self):
        """Zero cost when the driver can serve the request."""
        driver = _FakeDriver([(SUCCESS, 4242)])
        with self._patched(driver):
            handle = self.backing._mem_create_reclaiming(1 << 21, object())
        self.assertEqual(handle, 4242)
        self.assertEqual(driver.calls, 1)
        self.assertEqual(self.events, [], "empty_cache must not run on the happy path")

    def test_oom_reclaims_then_retries_and_succeeds(self):
        """The measured crash, now survived."""
        driver = _FakeDriver([(OOM, None), (SUCCESS, 99)])
        with self._patched(driver):
            handle = self.backing._mem_create_reclaiming(1 << 21, object())
        self.assertEqual(handle, 99)
        self.assertEqual(driver.calls, 2)
        # ORDER IS THE POINT: the reclaim sits BETWEEN the two attempts.
        # Reclaiming before the first attempt would pay the re-warm cost on
        # every commit; reclaiming after the second would not help it.
        self.assertEqual(self.events, [("empty_cache", 1)])

    def test_can_fail_a_genuine_full_card_still_raises(self):
        """The retry must not turn a real exhaustion into silence."""
        driver = _FakeDriver([(OOM, None), (OOM, None)])
        with self._patched(driver):
            with self.assertRaisesRegex(RuntimeError, "cuMemCreate failed"):
                self.backing._mem_create_reclaiming(1 << 21, object())
        self.assertEqual(driver.calls, 2)
        self.assertEqual(self.events, [("empty_cache", 1)])

    def test_a_non_oom_failure_is_not_retried(self):
        """Only OUT_OF_MEMORY is a reclaim candidate.

        Retrying an INVALID_VALUE would hide a programming error behind a
        cache flush and double every such failure's cost.
        """
        driver = _FakeDriver([(INVALID, None)])
        with self._patched(driver):
            with self.assertRaisesRegex(RuntimeError, "INVALID_VALUE"):
                self.backing._mem_create_reclaiming(1 << 21, object())
        self.assertEqual(driver.calls, 1)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
