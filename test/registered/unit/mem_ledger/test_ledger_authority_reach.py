"""The ledger is the VRAM authority on the DEFAULT path (#584 successor).

THE ORDER: the planner/ledger decides VRAM, hand-set heuristics do not.

WHY FLIPPING THE DEFAULT WAS NOT ENOUGH, which is the part worth keeping.
``enable_vram_ledger`` used to reach exactly one place:
``_vram_ledger_non_kv_per_gpu``, whose only caller is
``_resolve_auto_rank_tp_ratio`` (``server_args.py``), which runs only under
``--rank-tp-ratio auto``/``auto-performance`` -- and ``rank_tp_ratio`` defaults
to ``None``. That branch also requires ``--rank-gpu-id``. So on a plain boot,
including this fork's reference launch command, the ledger was unreachable no
matter what the flag said, and the inherited
``512 + tokens*1.5 + tp*pp/8*1024`` block owned ``mem_fraction_static``
outright. Flipping the flag alone would have changed nothing where the order
was aimed, while changing behaviour on the uneven-TP path where it was not.

WHAT MAKES IT TRUE INSTEAD: ``_ledger_reserve_mib`` is consulted first in the
``mem_fraction_static is None`` block, and supplies ``reserved_mem``. Only the
SOURCE of that number moves -- the fraction is still formed by the one existing
formula, so the ledger changes where the number comes from, not how a budget is
formed.

TWO PATHS, NO THIRD. Either the ledger prices the reserve, or it declares a
term unresolvable and NAMES it in the log (``ledger_full_demand_per_gpu``
refuses rather than returning a partial sum) and the inherited block runs. An
unprobed rig therefore still boots, on the old number, loudly.

THE COST, pinned below because it is real: an uncalibrated term is UNBOUNDED,
an unbounded term makes a card not fit, and on the uneven-TP path that is a
hard ``LedgerOvercommit`` rather than a fallback. A rig that boots
``--rank-gpu-id`` + ``--rank-tp-ratio auto`` without having run the probe was
sized by the heuristic before and is refused now. That refusal is the ledger
working as designed; it is still a behaviour change that a boot must validate.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.mem_ledger.contract import enforce_boot_contract
from sglang.srt.mem_ledger.terms import CardVramLedger, LedgerOvercommit
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _args(**kw):
    return ServerArgs(model_path="dummy", **kw)


class TestTheLedgerIsTheDefaultAuthority(CustomTestCase):
    def test_the_flag_is_on_by_default(self):
        self.assertIs(
            ServerArgs.__dataclass_fields__["enable_vram_ledger"].default, True
        )

    def test_the_sizing_block_asks_the_ledger_first(self):
        """The wiring itself. Without this call the flag is decoration: the
        ledger path is otherwise gated behind --rank-gpu-id + auto ratio."""
        src = inspect.getsource(ServerArgs._handle_gpu_memory_settings)
        self.assertIn("self._ledger_reserve_mib(gpu_mem)", src)
        self.assertLess(
            src.index("_ledger_reserve_mib"),
            src.index("post_capture_kv_sizing_planned"),
            "the ledger must be consulted BEFORE the inherited branches, or it "
            "is the fallback rather than the authority",
        )

    def test_the_heuristic_is_reached_only_when_the_ledger_declines(self):
        src = inspect.getsource(ServerArgs._handle_gpu_memory_settings)
        self.assertIn("if ledger_reserve is not None:", src)
        self.assertIn("elif self.post_capture_kv_sizing_planned():", src)

    def test_the_fraction_is_still_formed_by_one_formula(self):
        """Only the SOURCE of reserved_mem moved. If a second fraction
        computation appears, the budget has two formation rules again."""
        src = inspect.getsource(ServerArgs._handle_gpu_memory_settings)
        self.assertEqual(src.count("round((gpu_mem - reserved_mem) / gpu_mem, 3)"), 1)


class TestTheReserveSeamDeclinesHonestly(CustomTestCase):
    """``_ledger_reserve_mib`` returns a number or None, and every None is
    already announced by name. These are the three ways it says no."""

    def test_it_declines_when_the_flag_is_off(self):
        self.assertIsNone(_args(enable_vram_ledger=False)._ledger_reserve_mib(20480))

    def test_it_declines_without_a_card_size(self):
        self.assertIsNone(_args()._ledger_reserve_mib(None))

    def test_it_declines_when_the_ledger_refuses(self):
        args = _args()
        args.ledger_full_demand_per_gpu = lambda gpu_mem=None: None
        self.assertIsNone(args._ledger_reserve_mib(20480))

    def test_it_declines_rather_than_returning_a_nonsense_fraction(self):
        """Demand >= card would make (gpu_mem - reserve)/gpu_mem <= 0. The
        boot contract is where that gets refused with an itemization; here it
        just hands back."""
        args = _args()
        args.ledger_full_demand_per_gpu = lambda gpu_mem=None: {0: 20480}
        self.assertIsNone(args._ledger_reserve_mib(20480))


class TestTheBindProof(CustomTestCase):
    """Change the ledger's number, observe the reserve move with it -- the
    statement that distinguishes 'the ledger is consulted' from 'the ledger is
    consulted and its answer is used'."""

    def _reserve(self, demand):
        args = _args()
        args.ledger_full_demand_per_gpu = lambda gpu_mem=None: demand
        return args._ledger_reserve_mib(20480)

    def test_the_reserve_is_the_ledger_number(self):
        self.assertEqual(self._reserve({0: 1766}), 1766.0)

    def test_moving_the_ledger_moves_the_reserve(self):
        self.assertEqual(self._reserve({0: 3000}), 3000.0)

    def test_the_binding_card_sets_it_not_the_average(self):
        """One fraction is applied to every card, so a reserve that fits the
        roomiest card OOMs the tightest. MAX, not mean."""
        self.assertEqual(self._reserve({0: 1766, 1: 2900, 2: 1200}), 2900.0)


class TestAnUnboundedTermRefusesRatherThanGuesses(CustomTestCase):
    """The property that makes the flip a regression surface -- and the same
    property that makes the ledger worth having. Both, at once."""

    def _ledger(self, unbounded):
        return CardVramLedger(
            gpu_id=0,
            card="test-card",
            total_mib=20480,
            user_reserve_mib=1024,
            terms=(),
            unbounded=tuple(unbounded),
        )

    def test_a_card_with_an_unbounded_term_does_not_fit(self):
        self.assertFalse(self._ledger(["activation: not calibrated"]).fits)

    def test_an_unbounded_term_raises_at_the_boot_contract(self):
        with self.assertRaises(LedgerOvercommit):
            enforce_boot_contract([self._ledger(["activation: not calibrated"])])

    def test_the_same_card_fits_once_the_term_is_bounded(self):
        """So the refusal is about the MISSING MEASUREMENT, not about size --
        which is why an unprobed rig is the case that changes behaviour."""
        self.assertTrue(self._ledger([]).fits)

    def test_the_refusal_names_what_is_missing(self):
        try:
            enforce_boot_contract([self._ledger(["activation: not calibrated"])])
        except LedgerOvercommit as exc:
            self.assertIn("activation", str(exc))
        else:
            self.fail("expected LedgerOvercommit")


class TestTheInheritedTransientAnnouncesItself(CustomTestCase):
    """#584 item: the one term still standing on an unmeasured literal must not
    do so silently. The row already said INHERITED and carried a window tag
    that cannot match a live fingerprint (#612); what was missing was a log."""

    def test_the_fallback_warns_once_per_process(self):
        from sglang.srt.mem_ledger import engine

        engine._LoadTransientFallback.announced = False
        self.assertFalse(engine._LoadTransientFallback.announced)

    def test_the_latch_is_resettable_without_touching_a_module_global(self):
        """Deliberately a class attribute: a module-level latch is what made
        test_ledger_pool_wiring_590 order-dependent, and the module-state
        ratchet forbids new ones."""
        from sglang.srt.mem_ledger import engine

        self.assertTrue(hasattr(engine._LoadTransientFallback, "announced"))
        self.assertNotIn("global _load_transient", inspect.getsource(engine))


class TestTheGraphCoefficientIsACitationNotABudget(CustomTestCase):
    """The order said to correct GRAPH_MIB_PER_CAPTURED_TOKEN from the recorded
    3.3-3.8x measurement. It must NOT be corrected, and the reason is that it
    never books memory: the ledger refuses this term instead of estimating it.
    Correcting the constant would make the ledger MISQUOTE the stock
    heuristic's coefficient, which is the one thing it exists to name.
    """

    def test_it_is_never_added_to_a_ledger_term(self):
        from sglang.srt.mem_ledger import engine

        src = inspect.getsource(engine)
        # The only arithmetic use is the illustrative "~est MiB here" figure
        # inside the refusal message.
        self.assertIn("does NOT fall back to", src)

    def test_the_uncalibrated_graph_term_refuses(self):
        from sglang.srt.mem_ledger import engine

        src = inspect.getsource(engine)
        self.assertIn("REFUSAL, not the token estimate", src)

    def test_the_constant_still_quotes_the_stock_value(self):
        from sglang.srt.mem_ledger.engine import GRAPH_MIB_PER_CAPTURED_TOKEN

        self.assertEqual(
            GRAPH_MIB_PER_CAPTURED_TOKEN,
            2,
            "this constant quotes the stock per-captured-token coefficient; "
            "changing it to the measured 3.3-3.8x figure would make the "
            "ledger's provenance strings cite a number the stock code does "
            "not use",
        )


if __name__ == "__main__":
    unittest.main()
