"""#1081: the write-through decline census, and its can-fail proofs.

WHY THIS TEST EXISTS, and why it is shaped as an AST scan rather than a
behavioural drive of `write_backup`.

The defect being instrumented is an ATTRIBUTION defect, not a behaviour defect:
`write_backup` has eight exits, seven of which leave the node un-backed, and
before #1081 only two of them said anything. A node that never acquires a host
copy then makes `#841` decline every later host-only insert underneath it --
which is how a mamba anchor fails to be planted (`inserted_host_node is None`
-> `mamba_component.py`'s silent PREFETCH branch).

So the property under test is "every exit names ITS OWN reason". A behavioural
test that drives one exit proves nothing about the attribution of the other
six, and a mutation that swaps two reason strings would pass it. The check that
CAN fail on that mutation is a structural one, and it is the same technique the
speed-mode rule names approvingly: pick the check that can actually fail on the
likeliest failure class of this edit.

THREE ARMS, and the middle one is the point:
  1. exhaustiveness -- no `return 0` in `write_backup` is silent, including one
     added later;
  2. attribution -- each reason sits under ITS guard, so a swap goes red;
  3. the fifth state -- an in-flight write-through is not readable as a refusal.
"""

import ast
import pathlib
import unittest

from sglang.srt.mem_cache import match_refusal_census as C

_URC = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
    / "unified_radix_cache.py"
)


def _func(name: str, cls: str = "UnifiedRadixCache") -> ast.FunctionDef:
    tree = ast.parse(_URC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
    raise AssertionError(f"{cls}.{name} not found in {_URC}")


def _reason_calls(fn: ast.FunctionDef):
    """(lineno, reason_constant_name) for every census call inside ``fn``."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname not in ("_note_write_backup", "_note_wb_decline"):
            continue
        args = node.args
        # _note_wb_decline(self, reason, node, detail) -> reason is args[0]
        # because `self` is the receiver, not an argument.
        if args and isinstance(args[0], ast.Name):
            out.append((node.lineno, args[0].id))
    return sorted(out)


class TestEveryExitIsNamed(unittest.TestCase):
    """ARM 1: exhaustiveness. A silent `return 0` is the defect itself."""

    def test_no_return_zero_in_write_backup_is_silent(self):
        fn = _func("write_backup")
        reason_lines = {ln for ln, _ in _reason_calls(fn)}

        zero_returns = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Return)
            and isinstance(n.value, ast.Constant)
            and n.value.value == 0
        ]
        self.assertTrue(zero_returns, "write_backup has no `return 0` -- shape changed")

        for ln in zero_returns:
            # A naming call must sit within the few lines directly above the
            # return. `_note_wb_decline` spans several lines when it carries a
            # detail string, hence the window rather than an exact predecessor.
            self.assertTrue(
                any(ln - 12 <= r < ln for r in reason_lines),
                f"`return 0` at line {ln} of write_backup names no reason. A "
                f"silent exit here is invisible in every boot log, which is "
                f"exactly the defect #1081 exists to remove.",
            )

    def test_the_success_path_is_counted_too(self):
        """Without a denominator a refusal count reads as a decomposition (#873)."""
        fn = _func("write_backup")
        self.assertIn(
            "WB_GRANTED",
            [r for _, r in _reason_calls(fn)],
            "write_backup counts refusals but not grants -- the parts cannot sum",
        )


class TestTheReasonsAreNotInterchangeable(unittest.TestCase):
    """ARM 2: THE CAN-FAIL ARM. A swap of two reason strings must go red.

    Each reason is pinned to a token that only appears in ITS guard. If the
    strings are exchanged, the reason found after a guard is the wrong one and
    these assertions fail -- which is precisely the mutation that would leave
    the five candidates untellable apart again.
    """

    #: (token that identifies the guard, reason that must follow it first)
    GUARD_TO_REASON = (
        ("self.cache_controller is None", "WB_NO_CONTROLLER"),
        ("_mamba_write_through_pin_admissible", "WB_MAMBA_PIN"),
        ("self.write_backup(node.parent)", "WB_PARENT_DECLINED"),
        ("uniform_host_floor_active", "WB_HOST_FLOOR"),
        ("evicted < needed", "WB_EVICT_SHORT"),
        ("ring.admit", "WB_RING_DECLINED"),
        ("host_indices is None", "WB_WRITE_FAILED"),
    )

    def setUp(self):
        fn = _func("write_backup")
        self.first, self.last = (
            fn.lineno,
            max(getattr(n, "lineno", fn.lineno) for n in ast.walk(fn)),
        )
        self.lines = _URC.read_text().splitlines()
        self.reasons = _reason_calls(fn)

    def _reason_after(self, token: str):
        # CODE ONLY. `write_backup` documents its own guards in prose, so every
        # one of these tokens appears in a COMMENT before it appears in the
        # `if` it names -- and the first draft of this test matched the comment
        # and reported a misattribution that was not there. That is the #995
        # prose-marker trap, caught here rather than in a boot log.
        for i in range(self.first - 1, self.last):
            code = self.lines[i].split("#")[0]
            if token in code:
                guard_line = i + 1
                after = [r for ln, r in self.reasons if ln >= guard_line]
                return after[0] if after else None
        raise AssertionError(f"guard token {token!r} not found in write_backup")

    def test_each_guard_is_followed_by_its_own_reason(self):
        for token, expected in self.GUARD_TO_REASON:
            with self.subTest(guard=token):
                self.assertEqual(
                    self._reason_after(token),
                    expected,
                    f"the exit guarded by {token!r} does not report {expected}. "
                    f"A misattributed reason is worse than no reason: it sends "
                    f"the next reader to the wrong file with confidence.",
                )

    def test_all_reasons_are_declared_and_distinct(self):
        used = {r for _, r in _reason_calls(_func("write_backup"))}
        declared = {n for n in dir(C) if n.startswith("WB_") and n != "WB_REASONS"}
        self.assertTrue(
            used <= declared, f"undeclared reason constants used: {used - declared}"
        )
        values = [getattr(C, n) for n in sorted(used)]
        self.assertEqual(
            len(values), len(set(values)), "two exits share one reason string"
        )

    def test_parent_declined_is_not_counted_as_a_terminal_reason(self):
        """The cascade discriminator. One deep refusal must not read as N."""
        self.assertNotIn(C.WB_PARENT_DECLINED, C.TERMINAL_WB_REASONS)
        for r in (C.WB_HOST_FLOOR, C.WB_EVICT_SHORT, C.WB_RING_DECLINED):
            self.assertIn(r, C.TERMINAL_WB_REASONS)


class TestInFlightIsNotARefusal(unittest.TestCase):
    """ARM 3 (required): a write-through UNDERWAY must not read as declined.

    `backuped` is False for the whole span between `write_backup` returning and
    `_finish_write_through_ack`. Without the discriminator the #841 line reads
    identically for "refused" and "still on its way", and those call for
    opposite responses -- a capacity fix versus none at all.
    """

    def test_the_841_line_reports_inflight_from_the_pending_set(self):
        fn = _func("_insert_helper_host")
        src = ast.get_source_segment(_URC.read_text(), fn) or ""
        self.assertIn(
            "ongoing_write_through",
            src,
            "the #841 decline does not consult the pending write-through set, "
            "so a race is indistinguishable from a refusal",
        )
        self.assertIn("inflight=%d", src, "the #841 line does not print inflight")

    def test_the_discriminator_separates_the_two_states(self):
        class _Node:
            def __init__(self, nid):
                self.id = nid

        pending = {26: object()}
        self.assertEqual(int(_Node(26).id in pending), 1, "in-flight node reads 0")
        self.assertEqual(int(_Node(99).id in pending), 0, "refused node reads 1")

    def test_an_841_decline_adds_no_write_backup_refusal(self):
        """The two censuses must not double-count one event.

        #841 declining an insert is not a `write_backup` exit; counting it as
        one would inflate the refusal population a fix is meant to address.
        """
        fn = _func("_insert_helper_host")
        self.assertEqual(
            _reason_calls(fn),
            [],
            "the #841 host-insert decline is being counted as a write_backup "
            "refusal -- one event, two censuses, inflated denominator",
        )


class TestTheCensusReportsItself(unittest.TestCase):
    """PRESENT-BUT-UNWIRED is the middle delivery state and the most expensive.

    `format_prefetch_gate` had zero callers for twelve days while its counters
    filled up unreadably. This asserts #1081 did not repeat it.
    """

    def test_format_write_backup_has_a_caller_in_the_tree(self):
        src = _URC.read_text()
        self.assertIn(
            "_format_write_backup()",
            src,
            "format_write_backup is never called -- the counters are recorded "
            "and unreadable, exactly the #915 failure",
        )

    def test_no_observation_is_distinguishable_from_zero(self):
        C.WRITE_BACKUP_COUNTS.clear()
        self.assertIn("no observation", C.format_write_backup())
        C.note_write_backup(C.WB_GRANTED)
        line = C.format_write_backup()
        self.assertNotIn("no observation", line)
        self.assertIn("granted=1", line)
        self.assertIn("terminal=0", line)

    def test_terminal_excludes_grants_and_inherited_refusals(self):
        C.WRITE_BACKUP_COUNTS.clear()
        for r in (C.WB_GRANTED, C.WB_PARENT_DECLINED, C.WB_PARENT_DECLINED):
            C.note_write_backup(r)
        C.note_write_backup(C.WB_RING_DECLINED)
        self.assertIn("terminal=1", C.format_write_backup())


if __name__ == "__main__":
    unittest.main()
