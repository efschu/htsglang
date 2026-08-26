"""#872: a node the flip fence TRIED and FAILED to persist must not count as
a node the fence never had to touch.

WHY THIS SUITE EXISTS, from the measurement rather than from an argument.

The ticket was framed as "retention fires at the seam but nothing reaches the
store". That framing is refuted for the shipped configuration -- R7
(``boot_accept0826r7fix_0826_1817.log``, 2026-08-26) wrote 24277 complete
canonical pages during its window and served 126 of 264 storage prefetches
back at ``completed_local=4096`` -- and the module docstring carries the proof.
What survived the refutation is one line of the fence's own accounting.

``flip_writeback`` ends its staging loop with ``if written: staged += 1``.
``UnifiedRadixCache.write_backup`` has seven paths that return a bare ``0``
without raising and without logging (the mamba pin budget, the rank-uniform
host floor, a short ``evict_host``, the staging ring, a ``write`` that returned
None, a missing controller, the write-through parent recursion). So a refused
node fell out of every counter the fence publishes, and the fence's own log
line spelled it exactly like a fence with nothing to do.

The R7 log shows the population and shows how invisible it was: three fences
read ``eligible=9 staged=0 already_staged=8``, and the one refused node in each
of them is legible ONLY by subtracting two printed fields. Every other fence
shape in that boot -- ``eligible=5 staged=5 already_staged=0`` and
``eligible=4 staged=0 already_staged=4`` -- has the same subtraction come out
at zero, so nothing in the instrument said which kind of line was which.

RED-FIRST: each test below is written against the R7 shape it names, and each
one passes only because the refusal is counted, attributed and named. The fix
counts; it deliberately does not repair. The store write reads ``host_value``,
which is precisely what the refusal withheld, so there is no fallback that
would not mean writing bytes whose host state does not exist.
"""

import logging
import unittest

from sglang.srt.mem_cache.hicache_flip_writeback import (
    FlipWritebackReport,
    _mamba_pin_skipped,
    flip_writeback,
)
from sglang.test.test_utils import CustomTestCase


class _Node:
    """The parts of a radix TreeNode the fence reads."""

    def __init__(self, hash_value, *, backuped=False):
        self.hash_value = [hash_value]
        self.backuped = backuped
        self.children = {}
        self.parent = None

    def add(self, child):
        child.parent = self
        self.children[len(self.children)] = child
        return child


class _Tree:
    """A cache whose ``write_backup`` refuses a named set of nodes with 0.

    That is the live shape, not a caricature: every one of the seven refusal
    paths in ``UnifiedRadixCache.write_backup`` returns a bare ``0``. The fence
    cannot tell them apart from the return value and must not pretend to.
    """

    def __init__(self, *, refuse=(), mamba_refuse=(), mamba_counter=True):
        self.cache_controller = type(
            "_CC", (), {"storage_backend": type("_B", (), {"canonical_kv_page": 1})()}
        )()
        self.enable_storage = True
        self.root_node = _Node(None)
        self.root_node.hash_value = None
        self.ongoing_backup = {}
        self._refuse = set(refuse) | set(mamba_refuse)
        self._mamba_refuse = set(mamba_refuse)
        if mamba_counter:
            self._mamba_pin_skipped = 0

    def add_chain(self, specs):
        """specs: list of (hash_value, backuped). Root children, flat."""
        for hash_value, backuped in specs:
            self.root_node.add(_Node(hash_value, backuped=backuped))

    def write_backup(self, node, write_back=False):
        key = node.hash_value[0]
        if key in self._refuse:
            if key in self._mamba_refuse and hasattr(self, "_mamba_pin_skipped"):
                self._mamba_pin_skipped += 1
            return 0
        node.backuped = True
        return 1

    def writing_check(self, write_back=False):
        pass

    def _drain_storage_control_queues_local(self):
        self.ongoing_backup.clear()


def _r7_nine_eight() -> _Tree:
    """``eligible=9 staged=0 already_staged=8``, three times in R7.

    Eight nodes the ordinary write-through policy had already carried to host
    (and therefore to the store), and one the fence tried and could not.
    """
    tree = _Tree(refuse={"n9"})
    tree.add_chain([(f"n{i}", True) for i in range(1, 9)] + [("n9", False)])
    return tree


def _r7_five_five() -> _Tree:
    """``eligible=5 staged=5 already_staged=0``, 39 times in R7. Healthy."""
    tree = _Tree()
    tree.add_chain([(f"h{i}", False) for i in range(5)])
    return tree


def _r7_four_four() -> _Tree:
    """``eligible=4 staged=0 already_staged=4``, 33 times in R7.

    The shape the refuted framing read as the defect. Nothing was staged
    because nothing NEEDED staging.
    """
    tree = _Tree()
    tree.add_chain([(f"a{i}", True) for i in range(4)])
    return tree


class TestTheSilentZeroIsCounted(CustomTestCase):
    def test_the_r7_nine_eight_line_names_its_one_refused_node(self):
        """THE RED. Before the fix this fence published eligible=9 staged=0
        already_staged=8 and nothing else: the refused node was reachable only
        by subtracting two fields, and the subtraction is zero on every healthy
        shape too, so it distinguished nothing."""
        report = flip_writeback(_r7_nine_eight(), deadline_s=1.0)
        self.assertEqual(report.eligible, 9)
        self.assertEqual(report.staged, 0)
        self.assertEqual(report.already_staged, 8)
        self.assertEqual(report.refused_silently, 1)

    def test_the_healthy_r7_shape_refuses_nothing(self):
        """39 of 96 R7 fences. A counter that fired here would be the
        crying-wolf gate this instrument exists to avoid becoming."""
        report = flip_writeback(_r7_five_five(), deadline_s=1.0)
        self.assertEqual(report.staged, 5)
        self.assertEqual(report.already_staged, 0)
        self.assertEqual(report.refused_silently, 0)

    def test_an_all_already_staged_fence_refuses_nothing(self):
        """33 of 96 R7 fences, and the shape the refuted framing pointed at.
        staged=0 here is an idempotent skip, and the new counter must say so
        rather than lend the old framing a number."""
        report = flip_writeback(_r7_four_four(), deadline_s=1.0)
        self.assertEqual(report.eligible, 4)
        self.assertEqual(report.staged, 0)
        self.assertEqual(report.already_staged, 4)
        self.assertEqual(report.refused_silently, 0)

    def test_every_refused_node_is_counted_not_just_the_first(self):
        tree = _Tree(refuse={"n2", "n3", "n5"})
        tree.add_chain([(f"n{i}", False) for i in range(1, 7)])
        report = flip_writeback(tree, deadline_s=1.0)
        self.assertEqual(report.eligible, 6)
        self.assertEqual(report.staged, 3)
        self.assertEqual(report.refused_silently, 3)

    def test_the_counts_partition_the_eligible_set(self):
        """staged + already_staged + refused_silently must equal `eligible` on
        every shape, or the report is once again readable only by subtraction.

        Two further dispositions exist and are NOT in this sum on purpose,
        because each already draws its own named log line: the #841
        unbacked-parent skip and a `write_backup` that RAISED. Both were zero
        across all 96 R7 fences, and neither can be confused with "nothing to
        do" the way a bare 0 return could. The shapes below carry neither.
        """
        for name, tree in (
            ("nine_eight", _r7_nine_eight()),
            ("five_five", _r7_five_five()),
            ("four_four", _r7_four_four()),
        ):
            with self.subTest(shape=name):
                r = flip_writeback(tree, deadline_s=1.0)
                self.assertEqual(
                    r.staged + r.already_staged + r.refused_silently,
                    r.eligible,
                    msg=f"{name}: dispositions do not partition eligible",
                )


class TestTheMambaPinShareIsAttributedOrDeclared(CustomTestCase):
    def test_a_pin_budget_refusal_is_attributed_to_the_pin_budget(self):
        tree = _Tree(mamba_refuse={"m1", "m2"})
        tree.add_chain([("m1", False), ("m2", False), ("ok", False)])
        report = flip_writeback(tree, deadline_s=1.0)
        self.assertEqual(report.refused_silently, 2)
        self.assertEqual(report.refused_mamba_pin, 2)

    def test_a_refusal_from_another_path_is_not_charged_to_the_pin_budget(self):
        report = flip_writeback(_r7_nine_eight(), deadline_s=1.0)
        self.assertEqual(report.refused_silently, 1)
        self.assertEqual(report.refused_mamba_pin, 0)

    def test_a_tree_without_the_counter_reports_unmeasured_not_zero(self):
        """The #872 defect one level down was a probe whose miss path returned
        what a healthy zero returns. -1 and `?` keep the two apart."""
        tree = _Tree(refuse={"n1"}, mamba_counter=False)
        tree.add_chain([("n1", False), ("n2", False)])
        report = flip_writeback(tree, deadline_s=1.0)
        self.assertEqual(report.refused_silently, 1)
        self.assertEqual(report.refused_mamba_pin, -1)
        self.assertIn("refused_mamba_pin=?", report.as_log())

    def test_the_probe_rejects_a_bool_counter(self):
        """`True` is an int in Python and would read as the count 1."""
        self.assertEqual(_mamba_pin_skipped(type("_T", (), {})()), -1)
        self.assertEqual(
            _mamba_pin_skipped(type("_T", (), {"_mamba_pin_skipped": True})()), -1
        )
        self.assertEqual(
            _mamba_pin_skipped(type("_T", (), {"_mamba_pin_skipped": 7})()), 7
        )


class TestTheFenceSaysSoOutLoud(CustomTestCase):
    def test_the_log_line_carries_both_new_fields(self):
        report = flip_writeback(_r7_nine_eight(), deadline_s=1.0)
        line = report.as_log()
        self.assertIn("refused_silently=1", line)
        self.assertIn("refused_mamba_pin=0", line)

    def test_a_refusal_draws_a_warning_that_names_the_other_paths(self):
        with self.assertLogs(
            "sglang.srt.mem_cache.hicache_flip_writeback", level=logging.WARNING
        ) as caught:
            flip_writeback(_r7_nine_eight(), deadline_s=1.0)
        text = "\n".join(caught.output)
        self.assertIn("refused without raising", text)
        self.assertIn("unified_radix_cache.py:2210", text)

    def test_a_clean_fence_draws_no_refusal_warning(self):
        logger = logging.getLogger("sglang.srt.mem_cache.hicache_flip_writeback")
        with self.assertLogs(logger, level=logging.INFO) as caught:
            flip_writeback(_r7_five_five(), deadline_s=1.0)
        self.assertNotIn(
            "refused without raising",
            "\n".join(m for m in caught.output if "WARNING" in m),
        )


class TestPersistedNothingIsUnchangedByThis(CustomTestCase):
    """#783's predicate must keep its meaning. `refused_silently` is a real
    loss but a fence can carry one and still have persisted everything else;
    folding it in would be the same conflation in the other direction."""

    def _report(self, **kw):
        base = dict(
            eligible=1,
            staged=0,
            already_staged=0,
            acknowledged=0,
            outstanding=0,
            elapsed_s=0.0,
            deadline_s=2.0,
        )
        base.update(kw)
        return FlipWritebackReport(**base)

    def test_a_refusal_alone_does_not_flip_the_predicate(self):
        self.assertTrue(self._report(refused_silently=1).persisted_nothing)
        self.assertFalse(
            self._report(
                eligible=9, already_staged=8, acknowledged=3, refused_silently=1
            ).persisted_nothing
        )

    def test_the_default_keeps_every_existing_constructor_working(self):
        r = self._report()
        self.assertEqual(r.refused_silently, 0)
        self.assertEqual(r.refused_mamba_pin, -1)


if __name__ == "__main__":
    unittest.main()
