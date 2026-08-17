"""#584: the --rank-gpu-memory-mib contract, and two housekeeping resolutions.

THE CONTRACT, verbatim from ``ServerArgs.apply_rank_memory_budget``:

    "It is applied as-is: the MiB value is the rank's ENTIRE budget, so no
     further utilization ceiling or safety factor is added here."

``_handle_attention_backend_compatibility`` then did exactly that, in one place
and only for one combination: aiter + ``context_len > 8192`` multiplied
``mem_fraction_static`` by 0.85 regardless of how the fraction had been set. On
the rank-budget path the documented guarantee was therefore false -- silently,
and only for a backend/context pair nobody would think to check.

THE FIX IS NARROW ON PURPOSE. The 0.85 is untouched on the non-rank path: its
provenance is an upstream CI-fixing commit (``685c06451f``, "[ci] Try fixing
broken CIs (#12317)"), i.e. an unmeasured number of the #505c class. Removing
it there would replace one unmeasured decision with another. It is noted, not
touched.

The operator asked for a specific number of MiB. If that number is too large
for aiter at long context, the operator lowers it -- this code does not lower
it for them behind their back.

HOW THIS IS TESTED. The enclosing method needs a loaded model config, so it
cannot be driven whole in a hermetic test. Rather than restate the branch here
-- a copy passes happily while the shipped code rots -- the test EXTRACTS the
branch's own source text from server_args.py and executes it against stubs. The
code under test is therefore the shipped code; if someone edits or deletes the
branch, these tests see the edit.
"""

import inspect
import textwrap
import types
import unittest

_START = 'if resolved_view(self).attention_backend == "aiter":'
_END = "self.mem_fraction_static *= 0.85"


def _extract_shipped_branch() -> str:
    """Return the source text of the aiter/long-context branch as shipped."""
    from sglang.srt import server_args

    lines = inspect.getsource(server_args).splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == _START]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == _END]
    if len(starts) != 1 or len(ends) != 1:
        raise AssertionError(
            f"expected exactly one aiter branch, found {len(starts)} starts and "
            f"{len(ends)} ends -- the branch moved; fix this extractor rather "
            f"than the assertion"
        )
    if ends[0] < starts[0]:
        raise AssertionError("branch markers are out of order")
    return textwrap.dedent("\n".join(lines[starts[0] : ends[0] + 1]))


class _Recorder:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def _run_shipped_branch(*, rank_budget, context_len, backend="aiter", fraction=0.90):
    """Execute the shipped branch text with everything else stubbed out."""
    args = types.SimpleNamespace(mem_fraction_static=fraction)
    if rank_budget:
        args._rank_mem_fraction_static = fraction
    log = _Recorder()
    scope = {
        "self": args,
        "logger": log,
        "model_config": types.SimpleNamespace(context_len=context_len),
        "resolved_view": lambda _self: types.SimpleNamespace(attention_backend=backend),
        "getattr": getattr,
    }
    exec(_extract_shipped_branch(), scope)  # noqa: S102 - shipped source, by design
    return args.mem_fraction_static, log.warnings


class TestTheRankBudgetPathIsNotScaled(unittest.TestCase):
    """Red-first: every one of these failed before the guard, on the exact
    combination the contract promised was safe."""

    def test_a_rank_budget_survives_aiter_at_long_context(self):
        fraction, _ = _run_shipped_branch(rank_budget=True, context_len=131072)
        self.assertEqual(
            fraction,
            0.90,
            "the MiB value is the rank's ENTIRE budget; scaling it makes the "
            "documented guarantee false",
        )

    def test_the_refusal_is_announced_not_silent(self):
        """A budget that quietly survives teaches the operator nothing about
        the backend it survived."""
        _, warnings = _run_shipped_branch(rank_budget=True, context_len=131072)
        self.assertEqual(len(warnings), 1)
        self.assertIn("NOT scaling", warnings[0])

    def test_the_warning_tells_the_operator_what_to_do_instead(self):
        _, warnings = _run_shipped_branch(rank_budget=True, context_len=131072)
        self.assertIn("--rank-gpu-memory-mib yourself", warnings[0])

    def test_the_warning_names_the_context_length_that_triggered_it(self):
        _, warnings = _run_shipped_branch(rank_budget=True, context_len=131072)
        self.assertIn("131072", warnings[0])


class TestTheNonRankPathIsByteIdentical(unittest.TestCase):
    """The change must not reach anything it was not aimed at."""

    def test_the_scaling_still_applies_without_a_rank_budget(self):
        fraction, warnings = _run_shipped_branch(rank_budget=False, context_len=131072)
        self.assertAlmostEqual(fraction, 0.90 * 0.85)
        self.assertEqual(warnings, [], "the untouched path gained no new noise")

    def test_short_context_is_unaffected_on_both_paths(self):
        for rank_budget in (True, False):
            with self.subTest(rank_budget=rank_budget):
                fraction, warnings = _run_shipped_branch(
                    rank_budget=rank_budget, context_len=4096
                )
                self.assertEqual(fraction, 0.90)
                self.assertEqual(warnings, [])

    def test_the_boundary_is_strictly_greater_than_8192(self):
        """8192 exactly is below the threshold; 8193 is above it. Pinned
        because an off-by-one here silently changes who gets scaled."""
        at, _ = _run_shipped_branch(rank_budget=False, context_len=8192)
        over, _ = _run_shipped_branch(rank_budget=False, context_len=8193)
        self.assertEqual(at, 0.90)
        self.assertAlmostEqual(over, 0.90 * 0.85)

    def test_a_different_backend_is_unaffected(self):
        fraction, _ = _run_shipped_branch(
            rank_budget=False, context_len=131072, backend="fa3"
        )
        self.assertEqual(fraction, 0.90)


class TestTheContractStillExistsToBeHonoured(unittest.TestCase):
    def test_the_promise_sentence_is_still_in_the_tree(self):
        """If this sentence is ever deleted, the guard above loses its reason
        and someone should have to notice."""
        from sglang.srt import server_args

        self.assertIn(
            "no further utilization ceiling or safety",
            inspect.getsource(server_args),
        )


class TestTheDeadAccessorIsResolved(unittest.TestCase):
    """#584: ``kv_pool_mib_per_rank`` was exported as canonical with zero
    callers -- the module documented one contract while the tree used another.

    Resolved by DEMOTION, not wiring, and on evidence: the used path feeds the
    ledger into the SAME ``budget = (total - non_kv) // colocated`` arithmetic
    the reserve path uses, deliberately, so "the ledger changes where the
    number comes from, not how a budget is formed". Wiring this function would
    add a second budget-formation rule -- precisely what that design avoids.
    """

    def test_it_is_no_longer_exported_as_canonical(self):
        from sglang.srt.mem_ledger import contract

        self.assertNotIn("kv_pool_mib_per_rank", contract.__all__)

    def test_the_real_entry_point_is_still_exported(self):
        from sglang.srt.mem_ledger import contract

        self.assertIn("enforce_boot_contract", contract.__all__)

    def test_it_still_exists_rather_than_being_deleted(self):
        """Kept because its surplus rule documents a ledger invariant."""
        from sglang.srt.mem_ledger import contract

        self.assertTrue(callable(contract.kv_pool_mib_per_rank))

    def test_its_docstring_names_the_path_that_actually_runs(self):
        """Demotion without a forwarding address is just a third contract."""
        from sglang.srt.mem_ledger import contract

        doc = contract.kv_pool_mib_per_rank.__doc__ or ""
        self.assertIn("_vram_ledger_non_kv_per_gpu", doc)


class TestTheKnownWrongTermsAreFiledNotFixed(unittest.TestCase):
    """#584 item 3: the two terms go into the recorder's calibration queue.

    Desk-recalibrating them would repeat the exact class the recorder exists
    to end. The honest state of a wrong number is "wrong, known to be, with
    the measurement that would fix it named".
    """

    def test_the_queue_exists(self):
        from sglang.srt.mem_ledger.measured import CALIBRATION_QUEUE

        self.assertTrue(CALIBRATION_QUEUE)

    def test_both_known_wrong_terms_are_filed(self):
        from sglang.srt.mem_ledger.measured import CALIBRATION_QUEUE

        for term in ("LOAD_TRANSIENT_REFERENCE_MIB", "GRAPH_MIB_PER_CAPTURED_TOKEN"):
            with self.subTest(term=term):
                self.assertIn(term, CALIBRATION_QUEUE)

    def test_every_entry_names_a_route_out(self):
        """A queue whose entries have no exit condition is a comment."""
        from sglang.srt.mem_ledger.measured import CALIBRATION_QUEUE

        for term, why in CALIBRATION_QUEUE.items():
            with self.subTest(term=term):
                self.assertIn("Needs:", why, f"{term} is filed without a route out")

    def test_every_filed_term_is_a_real_ledger_constant(self):
        """Guards against the queue outliving what it describes."""
        from sglang.srt.mem_ledger import engine
        from sglang.srt.mem_ledger.measured import CALIBRATION_QUEUE

        for term in CALIBRATION_QUEUE:
            with self.subTest(term=term):
                self.assertTrue(
                    hasattr(engine, term),
                    f"{term} is queued for calibration but no longer exists",
                )

    def test_the_values_are_unchanged_by_this_cut(self):
        """Filing is not fixing. If these ever move, it must be a measurement
        that moved them, not a desk."""
        from sglang.srt.mem_ledger.engine import (
            GRAPH_MIB_PER_CAPTURED_TOKEN,
            LOAD_TRANSIENT_REFERENCE_MIB,
        )

        self.assertEqual(LOAD_TRANSIENT_REFERENCE_MIB, 70)
        self.assertEqual(GRAPH_MIB_PER_CAPTURED_TOKEN, 2)

    def test_each_constant_points_back_at_the_queue(self):
        """The person who finds the constant must find the filing, not just
        the person who finds the queue.

        This asserted ``count == 2`` when the only two mentions were the two
        pointers. That counted the evidence instead of stating the invariant,
        and it fired the moment a THIRD, legitimate mention appeared -- the
        runtime warning that announces the inherited load-transient fallback.
        The invariant was never "exactly two mentions exist"; it is "each
        queued constant has a pointer next to IT", so that is what is checked
        now, per constant, by locality.
        """
        from sglang.srt.mem_ledger import engine
        from sglang.srt.mem_ledger.measured import CALIBRATION_QUEUE

        lines = inspect.getsource(engine).splitlines()
        for term in CALIBRATION_QUEUE:
            with self.subTest(term=term):
                defs = [
                    i for i, ln in enumerate(lines) if ln.startswith(f"{term} = ")
                ]
                self.assertEqual(len(defs), 1, f"{term}: expected one definition")
                window = lines[max(0, defs[0] - 12) : defs[0]]
                self.assertTrue(
                    any("CALIBRATION_QUEUE" in ln for ln in window),
                    f"{term} is queued but its definition carries no pointer to "
                    f"the queue within the 12 lines above it",
                )

    def test_leaving_the_queue_requires_a_mapping_not_a_better_guess(self):
        from sglang.srt.mem_ledger import measured

        self.assertIn("Never by picking a better number.", inspect.getsource(measured))


if __name__ == "__main__":
    unittest.main()
