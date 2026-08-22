# SPDX-License-Identifier: Apache-2.0
"""#631: the cutover re-maps parked handles instead of allocating.

WHAT THIS IS FOR. The phase flip's seam releases the source layout's
physical KV pages and immediately commits the destination's. Both halves
are real driver traffic: ``decommit_range`` does ``cuMemUnmap`` +
``cuMemRelease``, and ``restore_backing`` does ``cuMemCreate`` for the
whole destination span. That create sits INSIDE the no-return region --
after it, the source pages are already gone -- so a driver that refuses
it takes the instance down, which is exactly what happened on metal
(2026-08-09 12:47:45, rank 1) and what the reclaim-and-retry in
``_mem_create_reclaiming`` was bolted on to survive.

Retention removes the question instead of answering it: the released
handles are UNMAPPED but kept, and the destination's commit re-maps them.
After one round trip the seam performs ZERO driver allocations, so there
is no allocation left to refuse.

THE COUPLING THAT MAKES IT WORK, pinned below because it is the part a
reader will not guess: a CUDA physical handle has a FIXED size and cannot
be split or merged, so a parked handle can only serve a request of the
same size. With one monolithic handle per buffer the PP and TP spans
produce different sizes and nothing is ever reusable -- retention would
park memory and still allocate. A commit CHUNK makes every handle one
granule, and only then are pages fungible between the layouts. That is
why the two are exposed as one knob, and why asking for retention without
a chunk is refused rather than honoured.

Hermetic: a fake driver, no GPU, no stub build.
"""

import types
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase


class _Result:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


SUCCESS = _Result("CUDA_SUCCESS")
OOM = _Result("CUDA_ERROR_OUT_OF_MEMORY")


class _FakeCUresult:
    CUDA_SUCCESS = SUCCESS
    CUDA_ERROR_OUT_OF_MEMORY = OOM


class _FakeDriver:
    """Records every driver call so ORDER and COUNT can both be pinned."""

    CUresult = _FakeCUresult

    def __init__(self, create_results=None):
        self._create_results = list(create_results or [])
        self._next_handle = 1000
        self.creates = []
        self.releases = []
        self.maps = []
        self.unmaps = []

    def cuMemCreate(self, step, prop, flags):  # noqa: N802
        self.creates.append(step)
        if self._create_results:
            return self._create_results.pop(0)
        self._next_handle += 1
        return (SUCCESS, self._next_handle)

    def cuMemMap(self, addr, size, offset, handle, flags):  # noqa: N802
        self.maps.append((addr, size, handle))
        return SUCCESS

    def cuMemSetAccess(self, addr, size, desc, count):  # noqa: N802
        return SUCCESS

    def cuMemUnmap(self, addr, size):  # noqa: N802
        self.unmaps.append((addr, size))
        return SUCCESS

    def cuMemRelease(self, handle):  # noqa: N802
        self.releases.append(handle)
        return SUCCESS


CHUNK = 1 << 20  # 1 MiB granule, so the arithmetic in the tests is readable


def _bare_arena(backing, retain_handles, chunk=CHUNK):
    """A KvVmmArena with only the fields commit/decommit touch.

    Built with ``__new__`` deliberately: the real constructor calls
    ``cuInit``, reserves virtual address space and compiles a C stub, none
    of which is under test here and all of which needs a GPU.
    """
    arena = object.__new__(backing.KvVmmArena)
    arena.device_id = 0
    arena.granularity = CHUNK
    arena.base = 0x100000000
    arena.reserved = 1 << 40
    arena._prop = object()
    arena._access = object()
    arena._extents_by_offset = {}
    arena._committed_by_offset = {}
    arena._range_backed = 0
    arena._closed = False
    arena._chunk = chunk
    arena._retain_handles = retain_handles
    arena._retained = {}
    arena._retained_bytes = 0
    # #464 (dce55c1430 / 17e7c8e36a): __init__ resolves this via
    # ``resolve_coalesce_resume(explicit)`` (kv_vmm_backing.py:469), consulted
    # by ``commit_range`` at kv_vmm_backing.py:667. Bind the real resolver
    # rather than stub a bool: it is already the defensive, DEFAULT-OFF
    # function (explicit None falls through to the unset
    # SGLANG_VMM_COALESCE_RESUME env var -> False in this process), which is
    # exactly the state a real arena has when nothing asked for coalescing --
    # and this file is about handle RETENTION (``_retain_handles`` above), not
    # about #464's coalescer, so it must not silently start exercising it.
    arena._coalesce_resume = backing.resolve_coalesce_resume(None)
    return arena


class TestHandleRetention(CustomTestCase):
    def setUp(self):
        import sglang.srt.mem_cache.kv_vmm_backing as backing

        self.backing = backing

    def _patched(self, driver):
        return mock.patch.multiple(
            self.backing,
            _driver=lambda: driver,
            torch=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    device=lambda *a, **k: _NullCtx(),
                    synchronize=lambda *a, **k: None,
                    empty_cache=lambda *a, **k: None,
                    memory_reserved=lambda *a: 0,
                    memory_allocated=lambda *a: 0,
                )
            ),
        )

    # -- the property the whole feature exists for -------------------------

    def test_a_second_commit_of_the_same_span_allocates_NOTHING(self):
        """THE pin: after one release/restore round trip the seam is
        zero-allocation. If this ever goes red, the cutover is asking the
        driver for memory inside the no-return region again."""
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=True)
        with self._patched(driver):
            arena.commit_range(0, 4 * CHUNK)
            self.assertEqual(len(driver.creates), 4)
            arena.decommit_range(0, 0)
            creates_after_release = len(driver.creates)
            arena.commit_range(0, 4 * CHUNK)

        self.assertEqual(
            len(driver.creates),
            creates_after_release,
            "the re-commit asked the driver for pages; retention is not working",
        )
        self.assertEqual(len(driver.maps), 8)  # mapped twice, allocated once

    def test_release_does_not_hand_pages_back_to_the_driver(self):
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=True)
        with self._patched(driver):
            arena.commit_range(0, 3 * CHUNK)
            released = arena.decommit_range(0, 0)

        self.assertEqual(driver.releases, [], "a parked handle was released")
        self.assertEqual(len(driver.unmaps), 3, "the pages must still be UNMAPPED")
        self.assertEqual(arena.retained_bytes, 3 * CHUNK)
        # decommit_range keeps reporting bytes UNMAPPED, which is no longer
        # the same thing as bytes made free to the driver. Pinned so the
        # meaning shift is caught by a test rather than by a wrong number
        # in a capacity table.
        self.assertEqual(released, 3 * CHUNK)

    def test_default_behaviour_is_unchanged_without_retention(self):
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=False)
        with self._patched(driver):
            arena.commit_range(0, 2 * CHUNK)
            arena.decommit_range(0, 0)
            arena.commit_range(0, 2 * CHUNK)

        self.assertEqual(len(driver.releases), 2)
        self.assertEqual(len(driver.creates), 4)
        self.assertEqual(arena.retained_bytes, 0)

    # -- the honest limits -------------------------------------------------

    def test_a_differently_sized_request_cannot_reuse_a_parked_handle(self):
        """Handles are fixed-size objects. This is the limit that makes the
        commit chunk mandatory rather than merely helpful."""
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=True, chunk=None)
        with self._patched(driver):
            arena.commit_range(0, 4 * CHUNK)  # one monolith of 4 MiB
            arena.decommit_range(0, 0)
            arena.commit_range(0, 6 * CHUNK)  # a 6 MiB span: no match

        self.assertEqual(len(driver.creates), 2)

    def test_a_partial_release_parks_only_the_released_extents(self):
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=True)
        with self._patched(driver):
            arena.commit_range(0, 4 * CHUNK)
            arena.decommit_range(0, 2 * CHUNK)

        self.assertEqual(arena.retained_bytes, 2 * CHUNK)
        self.assertEqual(arena.committed_bytes(0), 2 * CHUNK)

    # -- not leaking, and the escape hatch ---------------------------------

    def test_drop_retained_gives_the_pages_back(self):
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=True)
        with self._patched(driver):
            arena.commit_range(0, 3 * CHUNK)
            arena.decommit_range(0, 0)
            freed = arena.drop_retained()

        self.assertEqual(freed, 3 * CHUNK)
        self.assertEqual(len(driver.releases), 3)
        self.assertEqual(arena.retained_bytes, 0)

    def test_close_does_not_leak_the_park(self):
        driver = _FakeDriver()
        arena = _bare_arena(self.backing, retain_handles=True)
        with self._patched(driver):
            arena.commit_range(0, 2 * CHUNK)
            arena.decommit_range(0, 0)
            self.assertEqual(arena.retained_bytes, 2 * CHUNK)
            driver.cuMemAddressFree = lambda base, size: SUCCESS
            arena.close()

        self.assertEqual(len(driver.releases), 2)
        self.assertEqual(arena.retained_bytes, 0)

    def test_an_oom_create_reclaims_the_park_before_giving_up(self):
        """The park can be the thing holding the pages the create needs."""
        driver = _FakeDriver(create_results=[OOM, (SUCCESS, 77)])
        dropped = []

        with self._patched(driver):
            handle = self.backing._mem_create_reclaiming(
                CHUNK,
                object(),
                reclaim=lambda: (dropped.append(True), 3 * CHUNK)[1],
            )

        self.assertEqual(handle, 77)
        self.assertEqual(dropped, [True])


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    unittest.main()
