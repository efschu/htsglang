"""#829 -- the STRUCTURAL premises that make the W12 root unreachable.

WHAT THIS FILE IS FOR, and why it is not a duplicate of
``test_pp_arm_slot_outlives_ring_829.py``.

That file pins the BEHAVIOUR: a committed cutover does not carry the arm slot,
an abandon in the same epoch still restores it, the leftover drain still runs.
All nine of its tests pass on integ/round5, and the W12 root is closed.

The reason it is closed, however, rests on two facts that NOTHING pins:

  P1  the microbatch slot ring is rebuilt in exactly ONE place
      (``Scheduler.init_pp_loop_state``), and that place clears the
      ring-scoped slot carriers via ``pp_flip_forget_ring_scoped_slots``;
  P2  the pass loop can be JUMPED to a recorded slot from exactly ONE place
      (the ``elif would_restore:`` arm at ``scheduler_pp_mixin.py:2776``),
      which is the arm the epoch gate guards.

Both are true today and both are invisible to every behavioural test. A second
rebuild site, or a second writer of ``_pp_flip_resume_slot``, would bypass the
guards without failing anything -- and would return the tree to the state that
killed ``boot_window2_0823_1554``:

    #631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id=2 seq=19 rows=4096
    epoch=6 arrived while this rank is on mb_id=0 in flip epoch 6

    PHASE-FLIP PASS-CLOCK: rank 0 ran 0 slot iteration(s) (armed at mb_id=2,
    disarmed at mb_id=0); group passes [0, 0, 0], SPREAD 0;
    group RESUME SLOTS [2, 1, 1] -- DIVERGED

So these are UNIQUENESS PINS, not behaviour tests. They assert that the
premises of the unreachability argument still hold, and they fail loudly the
moment someone adds a site that the argument did not consider. A failure here
is not necessarily a bug -- it is a demand that the new site be checked against
the epoch gate and this docstring updated.

WHY THE TWO GUARDS ARE COMPLEMENTARY RATHER THAN REDUNDANT, since a reader
will ask whether one can be dropped:

  * the CLEAR (P1) covers a rebuild that does NOT advance the flip epoch --
    ``init_pp_loop_state`` has three callers (boot, the cutover's topology
    swap, and ``event_loop_pp``'s own entry) and only the cutover advances the
    epoch. The epoch gate cannot see those rebuilds at all.
  * the EPOCH GATE covers an arm recorded before a cutover that the clear
    somehow did not reach.

Dropping either one leaves a real hole, which is why both are pinned.

DELIBERATELY NOT PINNED: the ORDER of the clear against the array
assignments. Both orders are correct -- there is no reentrancy between them --
and pinning an arbitrary order would make a harmless refactor red.
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~2s: parses one source file with `ast`. No torch, no accelerator, no group.
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


MIXIN = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler_pp_mixin.py"
)

#: The arrays that ARE the microbatch slot ring. Rebuilding any of them
#: retires every slot number recorded against the old one.
RING_ARRAYS = ("mbs", "last_mbs", "running_mbs")

#: The rebuild's single legitimate home.
REBUILD_FN = "init_pp_loop_state"

#: The helper that retires ring-scoped slot numbers.
CLEAR_FN = "pp_flip_forget_ring_scoped_slots"


def _tree():
    return ast.parse(MIXIN.read_text(), filename=str(MIXIN))


def _enclosing_functions(tree, predicate):
    """Names of functions containing a statement matching *predicate*.

    Returns a list of (function_name, lineno) so a failure message can name
    the offending site instead of only its count.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if inner is node:
                continue
            # Do not attribute a nested function's statements to its parent.
            if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if predicate(inner):
                found.append((node.name, getattr(inner, "lineno", -1)))
    return found


def _assigns_self_attr(names):
    """Statement assigns to ``self.<name>`` for one of *names*."""

    def predicate(node):
        if not isinstance(node, ast.Assign):
            return False
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in names
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return True
        return False

    return predicate


def _assigns_attr_non_none(attr):
    """Statement assigns a NON-None value to ``<anything>.<attr>``.

    The `None` writes are the clears (``pp_flip_forget_ring_scoped_slots``
    nulls all three, and the falling edge consumes the epoch), and they are
    exactly what must NOT be counted as a way to set a slot.
    """

    def predicate(node):
        if not isinstance(node, ast.Assign):
            return False
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            return False
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == attr:
                return True
        return False

    return predicate


class TestTheRingIsRebuiltInOnePlace(CustomTestCase):
    """P1 -- the rebuild choke point."""

    def test_only_init_pp_loop_state_rebuilds_the_ring(self):
        sites = _enclosing_functions(_tree(), _assigns_self_attr(RING_ARRAYS))
        self.assertTrue(sites, "found no ring rebuild at all -- the pin is blind")
        offenders = sorted({fn for fn, _ in sites if fn != REBUILD_FN})
        self.assertEqual(
            offenders,
            [],
            "the microbatch slot ring is rebuilt outside "
            f"{REBUILD_FN}(), in: {offenders}. Every rebuild retires the slot "
            "numbers recorded against the old ring, so a rebuild that does not "
            f"reach {CLEAR_FN}() re-opens the #829 root (boot_window2_0823_1554: "
            "PP0 re-entered epoch 6 on slot 2 of a fresh ring). Either move the "
            "rebuild back, or clear the carriers at the new site and update "
            "this file's docstring.",
        )

    def test_the_rebuild_clears_the_ring_scoped_carriers(self):
        tree = _tree()
        rebuild = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == REBUILD_FN
            ),
            None,
        )
        self.assertIsNotNone(rebuild, f"{REBUILD_FN}() has vanished")
        calls = [
            n
            for n in ast.walk(rebuild)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == CLEAR_FN
        ]
        self.assertEqual(
            len(calls),
            1,
            f"{REBUILD_FN}() must call {CLEAR_FN}() exactly once; found "
            f"{len(calls)}. Without it a committed cutover carries "
            "_pp_flip_arm_mb_id into a ring that never issued it.",
        )


class TestTheRingIsJumpedFromOnePlace(CustomTestCase):
    """P2 -- the jump choke point."""

    def test_only_one_site_sets_a_resume_slot(self):
        sites = _enclosing_functions(
            _tree(), _assigns_attr_non_none("_pp_flip_resume_slot")
        )
        self.assertEqual(
            len(sites),
            1,
            "_pp_flip_resume_slot is what makes the pass loop JUMP to a "
            f"recorded slot, and it must have exactly one writer; found: "
            f"{sites}. The single writer is the `elif would_restore:` arm, "
            "which is reachable only when the epoch gate has already ruled the "
            "ring NOT rebuilt. A second writer would bypass that gate.",
        )

    def test_only_one_site_records_an_arm_slot(self):
        sites = _enclosing_functions(
            _tree(), _assigns_attr_non_none("_pp_flip_arm_mb_id")
        )
        self.assertEqual(
            len(sites),
            1,
            "_pp_flip_arm_mb_id must be recorded on the rising edge only; "
            f"found: {sites}. Each extra writer is another slot number that "
            "can outlive the ring that issued it.",
        )

    def test_the_arm_slot_and_its_epoch_are_recorded_together(self):
        """A slot without its generation is the #829 bug by construction.

        Pinned as a pair because the epoch is the only thing that makes the
        slot number interpretable, and #795 made the same argument for the
        proxy stamp.
        """
        tree = _tree()
        slot_sites = _enclosing_functions(
            tree, _assigns_attr_non_none("_pp_flip_arm_mb_id")
        )
        epoch_sites = _enclosing_functions(
            tree, _assigns_attr_non_none("_pp_flip_arm_epoch")
        )
        self.assertEqual(
            sorted({fn for fn, _ in slot_sites}),
            sorted({fn for fn, _ in epoch_sites}),
            "the arm slot and the arm epoch must be recorded in the same "
            f"function; slot in {slot_sites}, epoch in {epoch_sites}.",
        )


if __name__ == "__main__":
    unittest.main()
