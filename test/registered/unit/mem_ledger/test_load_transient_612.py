"""#612: the load transient becomes a priced term.

THE GAP. The 2026-08-06 #602 corridor acceptance runs sampled per-card NVML
free at 10 Hz, first idle and then under serving load, and found the load
MINIMUM up to ~70 MiB below the idle reading on every card -- with every other
ledger term priced and the corridor solver spending exactly the budget it was
given. So the demand model was short by an allocator peak that no row
mentioned, and both places that consume the ledger inherited the shortfall: the
#593 full-demand reserve (which sums everything outside the rank budget) and
the #602 corridor solve (which sums the terms that materialize after the free
reading it anchors on).

THE FALSIFIER these tests run is the one the gap deserves: build the same card
twice, once with the term and once with it removed, and show that the second
under-states the need by exactly the transient. That is a statement about the
SUM the reserve is formed from, not about a row appearing in a printout.

HONEST CONFOUND, repeated here because it is easy to lose: the 70 MiB is
INHERITED from that window's free-memory sampling, not measured by anything in
this tree, and the window ran one rank per card so it cannot say whether the
quantity is per card or per rank. The term charges per rank, which is the
reading that cannot under-charge a co-located card, and it says so in its own
derivation string.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    BUDGET_FUNDED_TERMS,
    LOAD_TRANSIENT_REFERENCE_MIB,
    LOAD_TRANSIENT_REFERENCE_TAG,
    TERM_LOAD_TRANSIENT,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
    demand_outside_budget_mib,
)
from sglang.srt.mem_ledger.reconcile import TERM_TO_POST
from sglang.srt.mem_ledger.terms import Provenance
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CARD = CardFacts(
    gpu_id=1,
    uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
    name="NVIDIA GeForce RTX 3080",
    total_mib=20480,
)


def inputs(ranks: int = 1, **over):
    base = dict(
        weight_mib_per_rank=[0] * ranks,
        activation_mib_per_rank=[1766.0] * ranks,
        capture_tokens_per_rank=[96] * ranks,
        capture_mib_per_rank=[640.0] * ranks,
        phase_footprint_source_per_rank=["[upper_bound] reference window"] * ranks,
        phase_footprint_fingerprint="a191a0712717",
        mamba_pool_mib_per_rank=[512.0] * ranks,
        flashinfer_workspace_mib=200,
        chunked_prefill_size=2048,
        max_running_requests=4,
        nccl_buffer_mib_per_gpu={1: 128.0},
        nccl_signature="tp3.dcp3.pp1",
    )
    base.update(over)
    return DemandInputs(**base)


def ledger(ranks: int = 1, **over):
    return build_card_ledgers(
        inputs(ranks, **over),
        cards=[CARD],
        rank_gpu_id=[1] * ranks,
        user_reserve_mib={1: 1024},
    )[0]


class TestTheTermExistsAndIsCharged(unittest.TestCase):
    def test_a_default_card_carries_the_inherited_transient(self):
        term = ledger().term(TERM_LOAD_TRANSIENT)
        self.assertIsNotNone(term, "no ledger row prices the load transient")
        self.assertEqual(term.mib, LOAD_TRANSIENT_REFERENCE_MIB)
        self.assertFalse(term.not_applicable)

    def test_the_inherited_number_says_so_in_its_own_row(self):
        term = ledger().term(TERM_LOAD_TRANSIENT)
        self.assertIn("INHERITED", term.derivation)
        self.assertIn("2026-08-06", term.derivation)
        self.assertEqual(term.fingerprint, LOAD_TRANSIENT_REFERENCE_TAG)
        self.assertIn("window-2026-", term.mark)

    def test_the_fingerprint_can_never_pass_for_a_rig_measurement(self):
        """The window tag is not a hardware fingerprint and must not look like
        one: a reader comparing it against the live fingerprint has to get a
        MISS, which is what keeps the number honest until a probe replaces
        it."""
        self.assertNotEqual(
            LOAD_TRANSIENT_REFERENCE_TAG,
            inputs().phase_footprint_fingerprint,
        )
        self.assertTrue(LOAD_TRANSIENT_REFERENCE_TAG.startswith("window-"))


class TestTheTermMovesTheNeedModel(unittest.TestCase):
    def _demand_without_the_term(self, lg) -> int:
        return sum(
            t.mib
            for t in lg.terms
            if t.name not in BUDGET_FUNDED_TERMS and t.name != TERM_LOAD_TRANSIENT
        )

    def test_without_the_term_the_reserve_under_states_by_the_transient(self):
        """THE HERMETIC FALSIFIER.

        ``demand_outside_budget_mib`` is the number #593 installs as the
        per-card reserve. Removing this term from that sum reproduces the
        pre-#612 model exactly, and the difference is the transient.
        """
        lg = ledger()
        with_term = int(demand_outside_budget_mib(lg))
        without_term = self._demand_without_the_term(lg)
        self.assertEqual(with_term - without_term, LOAD_TRANSIENT_REFERENCE_MIB)

    def test_the_term_is_outside_the_rank_budget(self):
        """A transient the allocator raises and gives back is not funded by the
        rank budget: the budget's residual is the KV pool, and the pool is
        resident while this peak happens on top of it."""
        self.assertNotIn(TERM_LOAD_TRANSIENT, BUDGET_FUNDED_TERMS)

    def test_co_located_ranks_each_pay_it(self):
        one = int(demand_outside_budget_mib(ledger(ranks=1)))
        two = int(demand_outside_budget_mib(ledger(ranks=2)))
        self.assertEqual(
            ledger(ranks=2).term(TERM_LOAD_TRANSIENT).mib,
            2 * LOAD_TRANSIENT_REFERENCE_MIB,
        )
        self.assertGreaterEqual(two - one, LOAD_TRANSIENT_REFERENCE_MIB)

    def test_the_corridor_solve_charges_it(self):
        """#602's documented gap: the corridor anchors on a free reading taken
        after weight load and before pool sizing, so the load peak is already
        given back at the anchor while the serving phase raises one again.
        Leaving it out is exactly why the solve could land below its own
        target under load."""
        from sglang.srt.server_args import ServerArgs

        self.assertIn(TERM_LOAD_TRANSIENT, ServerArgs.corridor_late_term_names())


class TestAMeasurementReplacesTheConstant(unittest.TestCase):
    def test_a_supplied_measurement_is_used_and_marked_calibrated(self):
        term = ledger(load_transient_mib_per_rank=[123.0]).term(TERM_LOAD_TRANSIENT)
        self.assertEqual(term.mib, 123)
        self.assertIs(term.provenance, Provenance.CALIBRATED)
        self.assertIn("measured", term.derivation)
        self.assertEqual(term.fingerprint, "a191a0712717")

    def test_a_measurement_is_summed_over_co_located_ranks(self):
        term = ledger(ranks=2, load_transient_mib_per_rank=[10.0, 20.0]).term(
            TERM_LOAD_TRANSIENT
        )
        self.assertEqual(term.mib, 30)

    def test_the_term_has_a_measured_counterpart_in_the_reconciliation(self):
        """The path off the inherited constant. Without an entry here the term
        would be reported UNMEASURED forever and the constant could never be
        falsified by a boot."""
        self.assertIn(TERM_LOAD_TRANSIENT, TERM_TO_POST)
        key, basis = TERM_TO_POST[TERM_LOAD_TRANSIENT]
        self.assertEqual(key, ("field", "allocator_transient_bytes", "weights_loaded"))
        self.assertIn("LOWER BOUND", basis)


class TestTheRecorderMeasuresWhatTheTermClaims(unittest.TestCase):
    def test_a_mark_carries_the_allocator_transient_field(self):
        """The recorder computes peak-minus-current at write time, so a reader
        of one line does not have to know which two counters to subtract."""
        import inspect

        from sglang.srt.mem_ledger import flight_recorder

        src = inspect.getsource(flight_recorder.mark)
        self.assertIn("allocator_transient_bytes", src)
        self.assertIn("reserved_peak_bytes", src)

    def test_the_field_is_the_reserved_pair_and_not_the_allocated_one(self):
        """NVML sees the allocator's RESERVATION. Measuring the allocated peak
        instead would produce a number the free-memory floor never feels."""
        import inspect

        from sglang.srt.mem_ledger import flight_recorder

        src = inspect.getsource(flight_recorder.mark)
        line = [
            x
            for x in src.splitlines()
            if "allocator_transient_bytes" in x and "record[" in x
        ]
        self.assertTrue(line)
        body = src.split('record["allocator_transient_bytes"]', 1)[1]
        head = body[:200]
        self.assertIn("reserved_peak_bytes", head)
        self.assertNotIn("allocated_peak_bytes", head)


if __name__ == "__main__":
    unittest.main()
