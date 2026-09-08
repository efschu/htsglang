# Copyright 2023-2024 SGLang Team
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
"""#1264 train2 fix 2 -- the three defects boot weg2t2b actually carried.

WHAT THE BOOT RECORD PROPOSED AND WHAT THE LOGS SAY.  BOOT_weg2t2b_0908.md read
the wall as "group D spins 177,175 rounds in a HiCache drain with nothing to
drain ... D never reaches sleep-kv".  D's own log refutes the second half:

    12:43:04 TP0/TP1/TP2  WEG2-DORMANT set: kv_cache paused, admission seams
                          refuse with W25 Weg2DormantRefused ...
    12:43:04              "POST /release_memory_occupation HTTP/1.1" 200 OK

D DID reach sleep-kv and completed it in under a second.  What did not survive
is the FRONT, one line later::

    [12:43:04,131] ERROR weg2.front: controller error: cannot unpack
                                     non-iterable CardFree object

``front.py`` line 2330 unpacked ``_nvml_free()``'s frozen ``CardFree``
dataclasses as 3-tuples.  The controller caught the ``TypeError``, logged it and
``continue``-d -- into a loop whose first statement is ``if self.state !=
"serving": continue``, with ``state`` left at "flipping" by the flip that had
just died.  The stall line's ``stage=sleep-kv`` is simply the last stage the
front assigned before raising; ``_flip_stage`` is only advanced at the gathered
legs.  So the HiCache rounds are what an awake, idle, never-again-addressed D
does forever -- a consequence, not the cause -- and the fix is here.

Three parts, three fixtures, all executable without a GPU or a boot:

* (A) the ``CardFree`` reader, and the CLASS behind it: an exception that
  escapes an open flip must STOP by name, never log-and-continue.
* (B) the ring table's source: the dormant sample joined to the chosen stem's
  own boot, so the ring stops ratcheting itself upward.
* (C) the idle tree-cache sanity walk, bounded on #1262's mechanism.
"""

import os
import pathlib
import re
import unittest
from types import SimpleNamespace

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import host_ledger, ring_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


# ======================================================================
# (A) the killer, and the class it belongs to
# ======================================================================
class TestCardFreeReader(CustomTestCase):
    """front.py:2330 -- the weg2t2b boot killer, reproduced and closed."""

    @staticmethod
    def _cards():
        return [
            front_mod.CardFree(
                nvml_index=1, uuid="GPU-a", free_mib=4272, reserved_mib=518
            ),
            front_mod.CardFree(
                nvml_index=0, uuid="GPU-b", free_mib=1847, reserved_mib=425
            ),
        ]

    def test_cardfree_is_not_iterable_red_first(self):
        """THE SPECIMEN.  The pre-fix expression raises exactly the boot's error.

        This is the red half: without it the fix below is a change with no
        demonstrated fault.  The message is asserted verbatim because it is the
        string the front log carries.
        """
        with self.assertRaises(TypeError) as ctx:
            {idx: free for idx, _uuid, free in self._cards()}  # noqa: B018
        self.assertIn("cannot unpack non-iterable CardFree object", str(ctx.exception))

    def test_attribute_read_gives_the_free_map(self):
        """The shipped expression, on the same objects."""
        free_mib = {c.nvml_index: c.free_mib for c in self._cards()}
        self.assertEqual(free_mib, {1: 4272, 0: 1847})

    def test_no_tuple_unpack_of_nvml_free_survives_anywhere(self):
        """SIBLING SWEEP, as a test rather than as a one-off scan.

        ``py_compile`` is structurally blind to this defect (it is a runtime
        unpack), so the guard has to be an AST scan, and it has to live where it
        runs again: a future reader written against the old 3-tuple shape fails
        here instead of on metal.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(front_mod))

        def is_nvml(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_nvml_free"
            )

        offenders = []
        for node in ast.walk(tree):
            for gen in getattr(node, "generators", []) or []:
                if is_nvml(gen.iter) and isinstance(gen.target, (ast.Tuple, ast.List)):
                    offenders.append(gen.target.lineno)
            if (
                isinstance(node, (ast.For, ast.AsyncFor))
                and is_nvml(node.iter)
                and isinstance(node.target, (ast.Tuple, ast.List))
            ):
                offenders.append(node.target.lineno)
        self.assertEqual(offenders, [], f"tuple-unpack of _nvml_free() at {offenders}")


class TestFlipEscapeVerdict(CustomTestCase):
    """The CLASS: an exception mid-flip is W4, not a logged shrug."""

    def test_open_flip_becomes_a_named_stop(self):
        v = front_mod.flip_escape_verdict(
            "flipping", "sleep-kv", 0, TypeError("cannot unpack non-iterable CardFree object")
        )
        self.assertIsNotNone(v)
        code, detail = v
        self.assertEqual(code, "W4 Weg2WakeRefused")
        # The detail must carry the three facts a reader needs: what, where in
        # the flip, and that no retry is legal.
        self.assertIn("TypeError", detail)
        self.assertIn("'sleep-kv'", detail)
        self.assertIn("epoch 0", detail)
        self.assertIn("No retry", detail)

    def test_error_outside_a_flip_is_not_a_stop(self):
        """The narrowing must be a narrowing: serving-state errors still pass.

        Without this the fix would turn every transient controller error into a
        group STOP, which is a worse boot killer than the one it closes.
        """
        for state in ("serving", "STOP"):
            self.assertIsNone(
                front_mod.flip_escape_verdict(state, "none", 3, RuntimeError("x")),
                f"state={state} must not stop the group",
            )

    def test_the_controller_actually_calls_it(self):
        """desk-written-never-executed: the helper must be WIRED, not merely
        present.  Structural, because driving the real controller needs an event
        loop, a session and six ranks."""
        import inspect

        src = inspect.getsource(front_mod.Front.controller)
        self.assertIn("flip_escape_verdict", src)
        self.assertIn("self.do_stop(*verdict)", src)


# ======================================================================
# (B) the ring table's source
# ======================================================================
#: The two REAL samples from /spinning/evidence-665-f1/weg2_measured_record.json
#: as of boot weg2t2b.  Same group, same weight tags, 47 minutes apart -- the
#: ratchet in two rows.  Inlined so the test does not read the evidence tree
#: (which does not exist on the remote desk).
_SAMPLE_T2A = {
    "group": "P",
    "at": "2026-09-08T11:54:49Z",
    "boot_tag": "weg2t2a",
    "commit": "d7e7df4ed1",
    "rss_shmem_gib": 36.30790328979492,
    "weight_tags_gib": 28.833746910095215,
    "extra_gib": 7.474156379699707,
}
_SAMPLE_T2B = {
    "group": "P",
    "at": "2026-09-08T12:41:15Z",
    "boot_tag": "weg2t2b",
    "commit": "d32e0bb316",
    "rss_shmem_gib": 37.82841110229492,
    "weight_tags_gib": 28.833746910095215,
    "extra_gib": 8.994664192199707,
}


class TestBootTagOfStem(CustomTestCase):
    def test_parses_the_launchers_own_stem_shape(self):
        self.assertEqual(
            ring_table.boot_tag_of_stem(
                "boot_weg2_weg2rg6_7f88b1c75d_0908_070324"
            ),
            "weg2rg6",
        )
        self.assertEqual(
            ring_table.boot_tag_of_stem(
                "boot_weg2_weg2t2b_d32e0bb316_0908_124003"
            ),
            "weg2t2b",
        )

    def test_a_stem_of_another_shape_is_none_not_a_guess(self):
        """None makes the caller treat the sidecar as ABSENT.  Returning a
        best-effort substring would re-open exactly the wrong-boot join."""
        for bad in ("", "boot_weg2_nope", "weg2t2b", "boot_weg2_x_zz_0908_070324"):
            self.assertIsNone(ring_table.boot_tag_of_stem(bad), bad)


class TestMeasuredRecordIsBoundToItsBoot(CustomTestCase):
    def _sidecar(self, samples):
        import json
        import os
        import tempfile

        d = tempfile.mkdtemp()
        p = os.path.join(d, host_ledger.MEASURED_RECORD_NAME)
        with open(p, "w") as f:
            json.dump({"samples": samples}, f)
        return p

    def test_unfiltered_read_takes_the_newest_boots_sample_red_first(self):
        """THE DRIFT CHANNEL, shown before it is closed.

        Unfiltered, the sidecar answers with whatever booted LAST -- while the
        correction that is subtracted from it comes from the chosen STEM's front
        log.  Two boots, one subtraction.
        """
        path = self._sidecar([_SAMPLE_T2A, _SAMPLE_T2B])
        got = host_ledger.read_measured_record(path)
        self.assertEqual(got["P"]["boot_tag"], "weg2t2b")

    def test_filtered_read_returns_that_boots_sample(self):
        path = self._sidecar([_SAMPLE_T2A, _SAMPLE_T2B])
        self.assertEqual(
            host_ledger.read_measured_record(path, boot_tag="weg2t2a")["P"][
                "boot_tag"
            ],
            "weg2t2a",
        )

    def test_a_boot_with_no_sample_is_an_absence_not_a_stranger(self):
        """The load-bearing case: boot weg2rg6 is the chosen stem and has NO
        sample.  The answer must be empty -- the caller then charges the census
        and says the cross-check was unavailable -- never t2b's number."""
        path = self._sidecar([_SAMPLE_T2A, _SAMPLE_T2B])
        self.assertEqual(host_ledger.read_measured_record(path, boot_tag="weg2rg6"), {})

    def test_the_ratchet_is_what_the_two_samples_measure(self):
        """DENOMINATOR: both samples carry the SAME weight_tags_gib, so the
        1.52 GiB that grew is the non-weight half -- the host ring's own
        MAP_SHARED pages, i.e. the very quantity being sized."""
        self.assertAlmostEqual(
            _SAMPLE_T2A["weight_tags_gib"], _SAMPLE_T2B["weight_tags_gib"], places=6
        )
        self.assertGreater(
            _SAMPLE_T2B["extra_gib"] - _SAMPLE_T2A["extra_gib"], 1.5
        )

    def test_dormant_images_propagates_the_filter(self):
        path = self._sidecar([_SAMPLE_T2A, _SAMPLE_T2B])
        self.assertEqual(ring_table.dormant_images(path, boot_tag="weg2rg6"), {})
        self.assertIn("P", ring_table.dormant_images(path, boot_tag="weg2t2a"))


class TestSolveNamesAndOrdersItsSource(CustomTestCase):
    def test_same_form_ordering_is_wired(self):
        import inspect

        src = inspect.getsource(ring_table.solve)
        self.assertIn("same_form + other_form", src)
        # and the sample is joined on the STEM, not on "newest"
        self.assertIn("boot_tag=stem_boot_tag", src)

    def test_the_chosen_source_line_is_emitted_by_the_launcher(self):
        """FIX 3 round 3 kept every SKIPPED line and dropped line 0 -- the one
        naming the boot actually used.  The launcher must print it."""
        import inspect

        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher)
        self.assertIn('plan.lines.append("WEG2-HOST-RING SOURCE "', src)
        self.assertIn("_reason_lines[1:]", src)


# ======================================================================
# (C) the idle tree-cache walk
# ======================================================================
class _FakeTreeCache:
    def __init__(self):
        self.component_evictable_size_ = {0: 10}
        self.component_protected_size_ = {0: 0}
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}


class _FakeScheduler:
    """Only the surface `_check_tree_cache_bounded` touches."""

    def __init__(self, clock):
        from sglang.srt.managers.scheduler import Scheduler

        self.tree_cache = _FakeTreeCache()
        self.walks = 0
        self._clock = clock
        self.invariant_checker = SimpleNamespace(_check_tree_cache=self._walk)
        self._tree_cache_fingerprint = Scheduler._tree_cache_fingerprint.__get__(self)
        self._check_tree_cache_bounded = Scheduler._check_tree_cache_bounded.__get__(self)

    def _walk(self):
        self.walks += 1

    def slow_walk(self):
        """A walk that costs REAL time, so the duty cycle has a real price to
        derive its refusal window from."""
        import time

        self.walks += 1
        time.sleep(0.050)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestIdleTreeSanityIsBounded(CustomTestCase):
    def test_fingerprint_moves_when_a_node_is_created(self):
        """The O(1) stage.  ``UnifiedTreeNode.counter`` is a class-level
        monotonic id source, so any insert or split moves the fingerprint."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

        clock = _Clock()
        s = _FakeScheduler(clock)
        before = s._tree_cache_fingerprint()
        self.assertIsNotNone(before)
        self.assertEqual(before, s._tree_cache_fingerprint(), "fingerprint must be stable")
        UnifiedTreeNode.counter += 1  # what a node construction does
        self.assertNotEqual(before, s._tree_cache_fingerprint())

    def test_a_static_tree_is_walked_once_not_every_idle_pass(self):
        """THE DEFECT, in one assertion.  Pre-fix this walked on every idle
        iteration; PP2 was caught in it on metal (weg2t2b)."""
        clock = _Clock()
        s = _FakeScheduler(clock)
        for _ in range(200):
            clock.t += 0.001  # 1 ms of idle loop between passes
            s._check_tree_cache_bounded()
        self.assertEqual(
            s.walks, 1, "an unchanged tree must be walked once, not 200 times"
        )
        self.assertEqual(s._idle_tree_cadence.agreed, 199)

    def test_a_changed_tree_is_walked_but_duty_bounded(self):
        """When the fingerprint DOES move every pass, the DUTY CYCLE -- not the
        fingerprint -- is what bounds the cost.

        Real time on purpose: the cadence reads ``time.monotonic`` and the
        method times the walk with ``time.perf_counter``, so a fake clock would
        prove nothing about the shipped code.  One 50 ms walk at
        ``IDLE_CENSUS_MAX_DUTY`` = 5 % buys ~950 ms of refusal, which is the
        whole property.
        """
        import time

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

        s = _FakeScheduler(_Clock())
        s.invariant_checker = SimpleNamespace(_check_tree_cache=s.slow_walk)

        s._check_tree_cache_bounded()  # first walk establishes the price
        self.assertEqual(s.walks, 1)
        self.assertGreater(s._idle_tree_cadence.last_cost_ms, 25.0)

        for _ in range(100):
            UnifiedTreeNode.counter += 1  # fingerprint moves on every pass
            s._check_tree_cache_bounded()
        self.assertEqual(s.walks, 1, "duty cycle must hold the walk back")
        self.assertGreater(s._idle_tree_cadence.deferred_cadence, 90)
        self.assertEqual(s._idle_tree_cadence.disagreed, 101)

        # ... and it must not hold it back FOREVER: a bound that never lets the
        # diagnostic run again is not a bound, it is a deletion.
        time.sleep(s._idle_tree_cadence.seconds_until_allowed() + 0.05)
        UnifiedTreeNode.counter += 1
        s._check_tree_cache_bounded()
        self.assertEqual(s.walks, 2)

    def test_fingerprint_is_not_recorded_when_the_walk_raises(self):
        """A tree that FAILED sanity must not be skipped on the next pass."""
        clock = _Clock()
        s = _FakeScheduler(clock)

        def boom():
            raise AssertionError("Sanity check FAILED (1 violations across 3 nodes)")

        s.invariant_checker = SimpleNamespace(_check_tree_cache=boom)
        with self.assertRaises(AssertionError):
            s._check_tree_cache_bounded()
        self.assertIsNone(getattr(s, "_idle_tree_fingerprint", None))

    def test_on_idle_calls_the_bounded_form(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.on_idle)
        self.assertIn("self._check_tree_cache_bounded()", src)
        self.assertNotIn("self.invariant_checker._check_tree_cache()", src)


class TestIsFullyIdleIsNotAnUnboundedPath(CustomTestCase):
    """#1264 (C), the half of the boot record that does NOT reproduce.

    BOOT_weg2t2b_0908.md listed PP1's ``is_fully_idle (scheduler.py:14303)``
    beside PP2's ``sanity_check`` as a second unbounded path.  It is not one:
    every term of ``is_fully_idle`` is a ``len()``, an ``is_empty()`` or an enum
    compare, and line 14303 was ``if self.disaggregation_mode ==
    DisaggregationMode.PREFILL:``.  A sampling profiler lands there because the
    function runs on EVERY idle pass, not because it is slow -- which is the
    indicator law: a frame is a finding only once you have checked that it
    measures what it claims.

    Binding it would have cost coverage for nothing, so nothing was bound, and
    this test records the reading so the next reader does not re-litigate it.
    """

    def test_no_whole_structure_walk_in_is_fully_idle(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.is_fully_idle)
        # The O(n)-shaped constructs a walk would need. `all(...)` over
        # `running_mbs`/`mbs` is O(pp_size) and is allowed by name.
        for banned in ("_collect_all_nodes", "sanity_check", "read_free_rows"):
            self.assertNotIn(banned, src)
        self.assertFalse(
            re.search(r"\bfor\s+\w+\s+in\s+self\.(tree_cache|token_to_kv_pool)", src),
            "is_fully_idle must not iterate a pool or the tree",
        )


# ======================================================================
# fix 2b -- the two INSTRUMENT defects that sent the triage to the wrong end
# ======================================================================
_DEADMAN = "/spinning/gpu-arb/devtools/boot_deadman.sh"


class TestStallLineNamesTheLastKnownStage(CustomTestCase):
    """fix 2b (1).  `stage=sleep-kv` was reported as if current, two minutes
    after that stage had completed and the controller had died."""

    def test_report_carries_the_age_and_never_a_bare_stage(self):
        line = front_mod.flip_stage_report("sleep-kv", 123.06)
        self.assertEqual(line, "stage_last_known=sleep-kv age_s=123.1")
        # RED-FIRST, as a property of the OUTPUT: the old spelling asserted
        # currency, and the whole defect is that word. `stage=` must not appear
        # as a field of its own anywhere in the report.
        self.assertNotRegex(line, r"(^|\s)stage=")

    def test_an_unstamped_stage_prints_unknown_not_zero(self):
        """A zero here would read as 'just now', which is the same lie in a
        different font."""
        self.assertEqual(
            front_mod.flip_stage_report("none", None), "stage_last_known=none age_s=unknown"
        )

    def test_the_stall_line_source_no_longer_spells_a_bare_stage(self):
        import inspect

        src = inspect.getsource(front_mod.Front.flip_stall_check)
        self.assertIn("flip_stage_report", src)
        self.assertNotIn("stage={self._flip_stage}", src)

    def test_every_assignment_is_stamped_because_it_is_a_property(self):
        """STRUCTURAL, and that is the point: there are seven assignment sites
        and the eighth would otherwise ship unstamped.  Written through the
        setter, value and timestamp cannot disagree."""
        f = front_mod.Front.__new__(front_mod.Front)
        f._flip_stage_value = "none"
        f._flip_stage_t = 0.0
        f._flip_stage = "drain"
        self.assertEqual(f._flip_stage, "drain")
        first = f._flip_stage_t
        self.assertGreater(first, 0.0, "the setter must stamp a monotonic time")
        f._flip_stage = "sleep-kv"
        self.assertGreaterEqual(f._flip_stage_t, first)
        # and the age is derived from THAT write, with an injectable now
        self.assertAlmostEqual(
            f.flip_stage_age_s(now=f._flip_stage_t + 42.0), 42.0, places=3
        )


class TestControllerDeathIsItsOwnVerdict(CustomTestCase):
    """fix 2b (2).  The death produced `controller error: ...` -- a generic
    handler message with no marker -- and only the 4x stall timer spoke."""

    _EXC = TypeError("cannot unpack non-iterable CardFree object")

    def test_the_line_carries_marker_epoch_stage_age_and_exception(self):
        line = front_mod.controller_dead_line(0, "sleep-kv", 123.06, self._EXC)
        self.assertIn("WEG2-FLIP CONTROLLER-DEAD epoch=0", line)
        self.assertIn("stage_last_known=sleep-kv age_s=123.1", line)
        self.assertIn("exc=TypeError", line)
        self.assertIn("No retry", line)

    def test_the_controller_emits_it_before_the_verdict(self):
        """desk-written-never-executed: ORDER matters -- the death line must be
        on the log before do_stop's own line, so a watcher reading forward sees
        the cause first."""
        import inspect

        src = inspect.getsource(front_mod.Front.controller)
        self.assertIn("controller_dead_line", src)
        self.assertLess(
            src.index("controller_dead_line"),
            src.index("self.do_stop(*verdict)"),
            "the CONTROLLER-DEAD line must precede the W4 verdict",
        )

    @unittest.skipUnless(os.path.exists(_DEADMAN), "boot_deadman.sh not on this box")
    def test_the_deadman_pattern_matches_what_the_front_emits(self):
        """THE WIRE, checked from both ends in one assertion.

        The tier's value is entirely in the two strings agreeing, and they live
        in different repositories -- front.py here, boot_deadman.sh under
        /spinning/gpu-arb.  Nothing else would notice them drifting apart until
        a boot died unwatched, which is the failure this whole fix is about.
        """
        pattern = None
        for raw in pathlib.Path(_DEADMAN).read_text().splitlines():
            if "CONTROLLER-DEAD epoch=" in raw and "grep -cE" in raw:
                pattern = raw.split("'")[1]
                break
        self.assertIsNotNone(pattern, "the deadman carries no CONTROLLER-DEAD pattern")
        emitted = front_mod.controller_dead_line(0, "sleep-kv", 123.06, self._EXC)
        self.assertRegex(emitted, pattern)
        # ... and the W4 half, which do_stop writes.
        code, detail = front_mod.flip_escape_verdict("flipping", "sleep-kv", 0, self._EXC)
        self.assertRegex(f"WEG2 STOP {code} -- {detail}", pattern)

    @unittest.skipUnless(os.path.exists(_DEADMAN), "boot_deadman.sh not on this box")
    def test_the_deadman_prose_does_not_arm_its_own_tier(self):
        """#995: both tokens appear in the deadman's own comments.  An
        unanchored pattern would arm the tier off documentation."""
        import re as _re

        pattern = None
        for raw in pathlib.Path(_DEADMAN).read_text().splitlines():
            if "CONTROLLER-DEAD epoch=" in raw and "grep -cE" in raw:
                pattern = raw.split("'")[1]
                break
        prose = [
            "this deadman treats a `WEG2-FLIP CONTROLLER-DEAD` line as a kill signal",
            "the flip refuses with W4 Weg2WakeRefused when a leg fails",
        ]
        for line in prose:
            self.assertIsNone(
                _re.search(pattern, line), f"prose must not arm the tier: {line!r}"
            )


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
