"""``override()`` must be the same write, plus provenance -- pinned hermetically.

Two post-resolution mutations were routed through ``ServerArgs.override()``
instead of bare assignment:

* ``model_executor/dual_group_lane.py:5495`` -- ``dual_group_lane_eager``, set
  inside ``Scheduler.__init__`` on the live server_args.
* ``mem_cache/gdn_slot_ladder.py`` -- ``gdn_profiled_state_slots``, set during
  ModelRunner KV profiling. It was ``_gdn_profiled_state_slots``, and the
  underscore was deliberate: ``ServerArgs.__setattr__`` exempts private names
  from the strict guard, so the mutation was invisible to the mechanism built
  to see it.

WHY THIS FILE EXISTS. The risk in that change is not the value, it is the
SEMANTICS: these run on a live object the engine is already serving on, and
only a boot exercises that for real. What a boot would check is that the post
state is the same as the assignment produced and that nothing trips the guard.
That part is decidable at a desk, so it is pinned here rather than deferred to
a window -- the boot validation is still owed and is on the F4-r4 window list.

The equivalence is asserted against a REAL ServerArgs with declarations
materialized and ``SGLANG_STRICT_CONFIG_MUTATION=1``, i.e. the exact
configuration under which a bare assignment is supposed to raise. A test that
used a stub here would prove nothing about the guard.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.gdn_slot_ladder import PROFILED_SLOTS_ATTR
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

# The two fields that moved onto the sanctioned path. The first is a declared
# dataclass field, the second an ad-hoc attribute -- override() has a different
# internal route for each (declaration stash vs runtime-mutation log), so both
# are exercised.
DECLARED_FIELD = "dual_group_lane_eager"
ADHOC_FIELD = PROFILED_SLOTS_ATTR


def _resolved_args():
    args = ServerArgs(model_path="dummy")
    object.__setattr__(args, "_declarations_materialized", True)
    return args


class _Strict:
    """Arm the strict mutation guard for the duration of a block."""

    def __enter__(self):
        self.saved = envs.SGLANG_STRICT_CONFIG_MUTATION.get()
        envs.SGLANG_STRICT_CONFIG_MUTATION.set(True)
        return self

    def __exit__(self, *exc):
        envs.SGLANG_STRICT_CONFIG_MUTATION.set(self.saved)
        return False


class TestTheGuardCanNowSeeBothWrites(CustomTestCase):
    """The evasion is closed: under the strict guard a BARE assignment of
    either field raises. For the gdn field this is the whole point -- with the
    old underscore name it would have sailed through."""

    def test_a_bare_write_of_the_declared_field_raises(self):
        args = _resolved_args()
        with _Strict():
            with self.assertRaises(AttributeError):
                setattr(args, DECLARED_FIELD, True)

    def test_a_bare_write_of_the_gdn_field_raises(self):
        args = _resolved_args()
        with _Strict():
            with self.assertRaises(AttributeError):
                setattr(args, ADHOC_FIELD, 20)

    def test_the_old_private_name_would_still_have_slipped_through(self):
        """Kept as the falsifier for the rename: this is what the code used to
        do, and it is exactly as invisible now as it was then. The rename is
        what closed the route -- not a change to the guard."""
        args = _resolved_args()
        with _Strict():
            setattr(args, "_gdn_profiled_state_slots", 20)  # no raise
        self.assertEqual(args._gdn_profiled_state_slots, 20)


class TestOverrideProducesTheSamePostState(CustomTestCase):
    """The invariant a boot would check: same resulting value, no guard trip."""

    def test_the_declared_field_matches_a_raw_assignment(self):
        raw = _resolved_args()
        object.__setattr__(raw, DECLARED_FIELD, True)  # what the old code did

        via = _resolved_args()
        with _Strict():
            via.override("dual_group_lane.spans_cards", **{DECLARED_FIELD: True})

        self.assertEqual(getattr(via, DECLARED_FIELD), getattr(raw, DECLARED_FIELD))
        self.assertIs(getattr(via, DECLARED_FIELD), True)

    def test_the_gdn_field_matches_a_raw_assignment(self):
        raw = _resolved_args()
        object.__setattr__(raw, ADHOC_FIELD, 20)

        via = _resolved_args()
        with _Strict():
            via.override(
                "gdn_slot_ladder.remember_profiled_state_slots", **{ADHOC_FIELD: 20}
            )

        self.assertEqual(getattr(via, ADHOC_FIELD), getattr(raw, ADHOC_FIELD))
        self.assertEqual(getattr(via, ADHOC_FIELD), 20)

    def test_override_does_not_trip_the_guard_it_satisfies(self):
        """override() sets _in_override around the write; if that ever stopped
        working these call sites would raise at boot, not in a test."""
        args = _resolved_args()
        with _Strict():
            args.override("test.both", **{DECLARED_FIELD: True, ADHOC_FIELD: 7})
        self.assertEqual(
            (args.dual_group_lane_eager, getattr(args, ADHOC_FIELD)), (True, 7)
        )

    def test_the_guard_flag_is_left_down_afterwards(self):
        """A leaked _in_override would disarm the guard for everything that
        follows on this object -- a worse failure than the one being fixed."""
        args = _resolved_args()
        with _Strict():
            args.override("test.leak", **{DECLARED_FIELD: True})
            with self.assertRaises(AttributeError):
                setattr(args, DECLARED_FIELD, False)


class TestProvenanceIsWhatOverrideAdds(CustomTestCase):
    """The reason to prefer override() over assignment is that it records who
    did it. If the record were empty the change would be pure ceremony."""

    def test_the_adhoc_field_is_recorded_with_its_source(self):
        args = _resolved_args()
        with _Strict():
            args.override(
                "gdn_slot_ladder.remember_profiled_state_slots", **{ADHOC_FIELD: 20}
            )
        log = getattr(args, "_runtime_mutations", [])
        self.assertIn(
            ("gdn_slot_ladder.remember_profiled_state_slots", {ADHOC_FIELD: 20}), log
        )

    def test_the_declared_field_is_recorded_somewhere(self):
        args = _resolved_args()
        with _Strict():
            args.override("dual_group_lane.spans_cards", **{DECLARED_FIELD: True})
        recorded = getattr(args, "_resolved_overrides", []) + getattr(
            args, "_runtime_mutations", []
        )
        self.assertTrue(
            any(src == "dual_group_lane.spans_cards" for src, _ in recorded),
            f"the override left no provenance: {recorded}",
        )


if __name__ == "__main__":
    unittest.main()
