# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656 spec item 6, rung 1: the spill ladder's allocator-cache release.

WHAT THESE TESTS PIN, and why each pin exists rather than being obvious:

* The rung runs BETWEEN the source release and the destination restore.
  Ordering is the whole content of the rung. Running it before the source
  release would reclaim while the outgoing layout's pages are still mapped, so
  there would be less to give back; running it after the destination restore
  would reclaim after the allocation that needed the pages had already had to
  succeed without them. A test that only checked "empty_cache was called"
  would pass for both wrong orders.

* Depth 0 must be BYTE-IDENTICAL to the pre-#656 seam. The default path has to
  stay untouched, and the cheapest way for that to rot is for the reclaim to
  run unconditionally and be "harmless".

* A depth above the implemented rung is REFUSED, not clamped. A clamp would
  make a depth sweep report that rung 3 is worth exactly what rung 1 is worth,
  which reads as a measurement and is an artefact.
"""

from __future__ import annotations

import unittest
from typing import List

from sglang.srt.managers import phase_flip_spill as spill


class _Pool:
    """A KV pool that records the order of the calls made against it."""

    def __init__(self, name: str, log: List[str]):
        self.name = name
        self._log = log

    def release_backing(self) -> None:
        self._log.append(f"release:{self.name}")

    def restore_backing(self) -> None:
        self._log.append(f"restore:{self.name}")


class _Runner:
    def __init__(self, pool):
        self.token_to_kv_pool = pool


class _Worker:
    def __init__(self, pool):
        self.model_runner = _Runner(pool)


class _Scheduler:
    def __init__(self, pool, server_args):
        self.tp_worker = _Worker(pool)
        self.server_args = server_args


class _Stacks:
    def __init__(self, pool):
        self.tp_worker = _Worker(pool)


class _Args:
    def __init__(self, depth):
        self.phase_flip_spill_depth = depth


class ResolveSpillDepthTest(unittest.TestCase):
    def test_names_and_integers_agree(self):
        self.assertEqual(spill.resolve_spill_depth(_Args("none")), 0)
        self.assertEqual(spill.resolve_spill_depth(_Args("cache")), 1)
        self.assertEqual(spill.resolve_spill_depth(_Args(0)), 0)
        self.assertEqual(spill.resolve_spill_depth(_Args(1)), 1)
        self.assertEqual(spill.resolve_spill_depth(_Args("CACHE")), 1)

    def test_unset_is_none_depth(self):
        self.assertEqual(spill.resolve_spill_depth(None), 0)
        self.assertEqual(spill.resolve_spill_depth(_Args(None)), 0)

    def test_garbage_is_refused_with_the_vocabulary(self):
        with self.assertRaises(spill.PhaseFlipSpillError) as cm:
            spill.resolve_spill_depth(_Args("cahce"))
        self.assertIn("cache", str(cm.exception))

    def test_rung_2_is_accepted_now_that_the_carrier_exists(self):
        # Rung 2 was refused for four shifts because the restore allocated a
        # FRESH arena and moved the addresses the TP decode graphs had baked.
        # #656 successor 29 put the weights on a VA-stable KvVmmArena span, so
        # the refusal is lifted -- but only for rung 2.
        self.assertEqual(spill.resolve_spill_depth(_Args("draft")), 2)
        self.assertEqual(spill.resolve_spill_depth(_Args(2)), 2)

    def test_unimplemented_rung_is_refused_not_clamped(self):
        # The refusal is the feature: a clamp would make a sweep of rungs 2
        # and 3 report a difference of zero and look like a measurement.
        # Rung 3 is the ARENA TAIL now and it IS wired (#656 successor 30);
        # the refused rung is 4, the draft CUDA graphs, which cannot be
        # refilled from a host image and must be re-captured.
        for depth in ("draft+graphs", 4):
            with self.assertRaises(spill.PhaseFlipSpillError) as cm:
                spill.resolve_spill_depth(_Args(depth))
            self.assertIn("not wired", str(cm.exception))

    def test_out_of_range_is_refused(self):
        with self.assertRaises(spill.PhaseFlipSpillError):
            spill.resolve_spill_depth(_Args(99))
        with self.assertRaises(spill.PhaseFlipSpillError):
            spill.resolve_spill_depth(_Args(-1))

    def test_implemented_depth_does_not_silently_exceed_the_ladder(self):
        self.assertLessEqual(spill.IMPLEMENTED_DEPTH, spill.MAX_DEPTH)
        self.assertEqual(spill.IMPLEMENTED_DEPTH, spill.DEPTH_ARENA_TAIL)
        # "draft" must keep meaning 2: recorded evidence uses the integers.
        self.assertEqual(spill.DEPTH_DRAFT_WEIGHTS, 2)


class ReleaseAllocatorCacheTest(unittest.TestCase):
    def test_depth_zero_does_nothing_at_all(self):
        calls = []
        with _patched_torch(calls):
            self.assertEqual(spill.release_allocator_cache("pp_to_tp", depth=0), 0)
        self.assertEqual(calls, [])

    def test_depth_one_reclaims_and_reports_delta_for_the_named_device(self):
        calls = []
        with _patched_torch(calls, free_seq=[1000, 3500]):
            got = spill.release_allocator_cache(
                "pp_to_tp", depth=1, device_index=2
            )
        # Both reads name device 2 EXPLICITLY. Every worker in this deployment
        # sees all three cards, so a current-device read can report a card the
        # rank does not own -- which is exactly the bug this pins.
        self.assertEqual(
            calls, ["mem_get_info:2", "empty_cache", "mem_get_info:2"]
        )
        self.assertEqual(got, 2500)
        self.assertNotIn("current_device", calls)

    def test_without_a_device_it_falls_back_to_the_current_one(self):
        calls = []
        with _patched_torch(calls, free_seq=[10, 20]):
            spill.release_allocator_cache("pp_to_tp", depth=1)
        self.assertIn("current_device", calls)
        self.assertEqual(
            [c for c in calls if c.startswith("mem_get_info")],
            ["mem_get_info:0", "mem_get_info:0"],
        )

    def test_a_failing_reclaim_never_takes_the_flip_down(self):
        # The flip is CORRECT without the rung, only tighter. An exception
        # here would convert a memory optimisation into an availability bug.
        calls = []
        with _patched_torch(calls, raise_on_empty=True):
            self.assertEqual(spill.release_allocator_cache("tp_to_pp", depth=1), 0)


class SeamOrderingTest(unittest.TestCase):
    """The rung must sit strictly between release and restore."""

    def _run(self, depth: str, direction: str) -> List[str]:
        from sglang.srt.managers import phase_flip_runtime as rt

        log: List[str] = []
        pp_pool = _Pool("pp", log)
        tp_pool = _Pool("tp", log)
        sched = _Scheduler(pp_pool, _Args(depth))
        stacks = _Stacks(tp_pool)

        real_release = spill.release_allocator_cache
        real_mark = rt.seam_census.mark

        def fake_release(direction_, *, depth_arg=None, **kw):
            d = kw.get("depth", depth_arg)
            if d and d >= spill.DEPTH_ALLOCATOR_CACHE:
                log.append("reclaim")
                return 1
            return 0

        rt.seam_census.mark = lambda *a, **k: log.append(f"mark:{a[0]}")
        spill.release_allocator_cache = fake_release
        try:
            # The stage's own layer ordinals: the waved seam needs them to
            # map global ordinals onto the PP pool's local indices. The
            # whole-pool __call__ path under test here does not use them.
            swap = rt._build_kv_backing_swap(sched, stacks, (0, 1))
            swap(direction)
        finally:
            spill.release_allocator_cache = real_release
            rt.seam_census.mark = real_mark
        return log

    def test_depth_zero_seam_is_byte_identical_to_the_old_one(self):
        log = self._run("none", "pp_to_tp")
        self.assertEqual(
            log,
            ["release:pp", "mark:backing_release", "restore:tp",
             "mark:backing_restore"],
        )
        self.assertNotIn("reclaim", log)

    def test_reclaim_runs_between_release_and_restore(self):
        log = self._run("cache", "pp_to_tp")
        self.assertEqual(
            log,
            ["release:pp", "mark:backing_release", "reclaim",
             "mark:allocator_cache_release", "restore:tp",
             "mark:backing_restore"],
        )
        # Stated as an ordering invariant too, so a future refactor that keeps
        # every element but permutes them fails here with a readable message.
        self.assertLess(log.index("release:pp"), log.index("reclaim"))
        self.assertLess(log.index("reclaim"), log.index("restore:tp"))

    def test_both_directions_reclaim_with_the_source_released_first(self):
        log = self._run("cache", "tp_to_pp")
        self.assertEqual(
            log,
            ["release:tp", "mark:backing_release", "reclaim",
             "mark:allocator_cache_release", "restore:pp",
             "mark:backing_restore"],
        )

    def test_a_bad_depth_is_refused_at_wiring_time_not_mid_cutover(self):
        # Resolution happens when the swap is BUILT. If it happened inside the
        # swap, the exception would land after release_backing() had already
        # handed the source pool's pages back, leaving neither layout backed.
        from sglang.srt.managers import phase_flip_runtime as rt

        # "draft+graphs" rather than "draft": rung 2 became implemented in
        # #656 successor 29, so the still-unwired rung is now 3. The property
        # under test is WHEN the refusal happens, not which rung triggers it.
        sched = _Scheduler(_Pool("pp", []), _Args("draft+graphs"))
        with self.assertRaises(spill.PhaseFlipSpillError):
            rt._build_kv_backing_swap(sched, _Stacks(_Pool("tp", [])), (0, 1))


class _patched_torch:
    """Stand in for torch.cuda without needing a device.

    A real-CUDA test could not assert the call ORDER, which is the property
    that matters, and would make the suite unrunnable on a busy rig.
    """

    def __init__(self, calls, free_seq=(1000, 1000), raise_on_empty=False):
        self.calls = calls
        self.free_seq = list(free_seq)
        self.raise_on_empty = raise_on_empty
        self._saved = None

    def __enter__(self):
        import sys
        import types

        calls = self.calls
        seq = self.free_seq
        raise_on_empty = self.raise_on_empty

        mod = types.ModuleType("torch")
        cuda = types.SimpleNamespace()

        def mem_get_info(device=None):
            calls.append(f"mem_get_info:{device}")
            return (seq.pop(0) if seq else 0, 10**9)

        def empty_cache():
            calls.append("empty_cache")
            if raise_on_empty:
                raise RuntimeError("driver said no")

        def current_device():
            calls.append("current_device")
            return 0

        cuda.is_available = lambda: True
        cuda.current_device = current_device
        cuda.mem_get_info = mem_get_info
        cuda.empty_cache = empty_cache
        mod.cuda = cuda
        self._saved = sys.modules.get("torch")
        sys.modules["torch"] = mod
        return self

    def __exit__(self, *exc):
        import sys

        if self._saved is not None:
            sys.modules["torch"] = self._saved
        else:  # pragma: no cover - torch is always importable in this tree
            sys.modules.pop("torch", None)
        return False


if __name__ == "__main__":
    unittest.main()
