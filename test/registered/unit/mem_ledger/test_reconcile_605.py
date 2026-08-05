"""#605: the modeled ledger against the measured boot, term by term.

The #602 window established that internal demand overpredicts by 4664 / 993 /
2701 MiB. That total funds no fix: it says the model is wrong without saying
WHICH term is wrong. This module puts each modeled term beside the measurement
that brackets it, and these tests pin the two ways such a table could lie.

1. By ABSORBING. If a measured byte that no term claims is quietly folded into
   the nearest term, the table balances and the unmodeled post disappears. The
   residuum is therefore computed as an independent quantity and printed as its
   own row, whether it is 4000 MiB or 0.
2. By PRETENDING TO MEASURE. A term with no phase boundary bracketing its
   allocation is UNMEASURED, which is a statement about the instrument's reach.
   Reporting it as 0 measured -- and therefore as a large overprediction --
   would manufacture a defect in a term that may be perfectly correct.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    TERM_ACTIVATION,
    TERM_ATTN_WORKSPACE,
    TERM_GRAPH_CAPTURE,
    TERM_HARDWARE_RESIDUAL,
    TERM_MAMBA_POOL,
    TERM_NVML_CARVE_OUT,
    TERM_PARENT_CONTEXT,
    TERM_WEIGHTS,
)
from sglang.srt.mem_ledger.reconcile import TERM_TO_POST, reconcile, reconcile_card
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20


def _mark(
    phase, *, reserved=0, self_bytes=0, non_torch=0, carve=0, procs=None, pid=100
):
    return {
        "phase": phase,
        "rank": 0,
        "pid": pid,
        "boot_id": "boot1",
        "reserved_bytes": reserved * MIB,
        "nvml_self_bytes": self_bytes * MIB,
        "non_torch_bytes": non_torch * MIB,
        "nvml_carve_out_bytes": carve * MIB,
        "nvml_processes": procs or {},
        "monotonic": 0.0,
    }


def _boot_marks():
    """A boot whose posts are exactly known, so the reconciliation is checkable.

    weights 7000, state pool 600 (in a gap that also holds a 12000 KV pool),
    attention workspaces 300, graph capture 900, CUDA context etc. 500.
    """
    return [
        _mark("pre_weight_load", reserved=0, self_bytes=500, non_torch=500),
        _mark("weights_loaded", reserved=7000, self_bytes=7500, non_torch=500),
        _mark("kv_pool_sized", reserved=19600, self_bytes=20100, non_torch=500),
        _mark("capture_begin", reserved=19900, self_bytes=20400, non_torch=500),
        _mark("capture_end", reserved=20800, self_bytes=21300, non_torch=500),
        _mark(
            "boot_complete",
            reserved=20800,
            self_bytes=21300,
            non_torch=500,
            carve=425,
            procs={"100": 21300 * MIB},
        ),
        _mark("first_forward", reserved=21200, self_bytes=21700, non_torch=500),
    ]


def _ledger(terms, *, kv_pool=12000, demand=None):
    rows = [{"name": n, "mib": m, "provenance": "modeled"} for n, m in terms]
    return {
        "gpu_id": 1,
        "card": "NVIDIA GeForce RTX 3080",
        "ranks": [0],
        "kv_pool_mib": kv_pool,
        "demand_mib": demand if demand is not None else sum(m for _n, m in terms),
        "terms": rows,
    }


class TestTermByTerm(unittest.TestCase):
    def test_a_correct_model_shows_zero_error_on_every_term(self):
        ledger = _ledger(
            [
                (TERM_WEIGHTS, 7000),
                (TERM_MAMBA_POOL, 600),
                (TERM_GRAPH_CAPTURE, 900),
                (TERM_HARDWARE_RESIDUAL, 500),
                (TERM_NVML_CARVE_OUT, 425),
            ]
        )
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        errors = {c.term: c.error_mib for c in result.comparisons}
        self.assertEqual(errors[TERM_WEIGHTS], 0)
        self.assertEqual(errors[TERM_MAMBA_POOL], 0)
        self.assertEqual(errors[TERM_GRAPH_CAPTURE], 0)
        self.assertEqual(errors[TERM_HARDWARE_RESIDUAL], 0)
        self.assertEqual(errors[TERM_NVML_CARVE_OUT], 0)

    def test_an_overpredicting_term_is_named_and_quantified(self):
        """The deliverable: not 'demand is 2701 MiB too high' but 'the graph
        capture term is 2701 MiB too high'."""
        ledger = _ledger(
            [
                (TERM_WEIGHTS, 7000),
                (TERM_GRAPH_CAPTURE, 3601),
                (TERM_HARDWARE_RESIDUAL, 500),
            ]
        )
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        by_term = {c.term: c for c in result.comparisons}
        self.assertEqual(by_term[TERM_GRAPH_CAPTURE].measured_mib, 900)
        self.assertEqual(by_term[TERM_GRAPH_CAPTURE].error_mib, 2701)
        self.assertEqual(by_term[TERM_WEIGHTS].error_mib, 0)

    def test_the_kv_pool_is_subtracted_from_the_pool_sizing_gap(self):
        """That gap holds the KV pool AND the state pool; charging the state
        pool with the KV pool's bytes would show a 12000 MiB underprediction
        on a term that is right."""
        ledger = _ledger([(TERM_MAMBA_POOL, 600)])
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        self.assertEqual(result.comparisons[0].measured_mib, 600)

    def test_a_gap_too_small_for_its_kv_pool_refuses_to_be_split(self):
        """If the pool did not get what the ledger budgeted, the gap cannot be
        divided -- and a negative state pool would be nonsense reported as
        fact."""
        ledger = _ledger([(TERM_MAMBA_POOL, 600)], kv_pool=19000)
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        self.assertIsNone(result.comparisons[0].measured_mib)
        self.assertIn("cannot be split", result.comparisons[0].note)


class TestResiduum(unittest.TestCase):
    def test_an_unmodeled_post_lands_in_the_residuum_and_is_not_absorbed(self):
        """This ledger omits the state pool, the attention workspaces and the
        carve-out. Measured demand is 21300 + 425 - 12000 = 9725; the terms
        claim 7000 + 900 + 500 = 8400. The 1325 MiB difference is exactly the
        three omitted posts (600 + 300 + 425) and must be visible rather than
        smeared across the terms that are present."""
        ledger = _ledger(
            [
                (TERM_WEIGHTS, 7000),
                (TERM_GRAPH_CAPTURE, 900),
                (TERM_HARDWARE_RESIDUAL, 500),
            ]
        )
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        self.assertEqual(result.measured_demand_mib, 9725)
        self.assertEqual(result.residuum_mib, 600 + 300 + 425)
        self.assertIn("no term claims", result.residuum_note)

    def test_a_complete_model_leaves_a_zero_residuum_that_still_prints(self):
        """Every post of the synthetic boot modeled, including the attention
        workspaces: 7000 + 600 + 300 + 900 + 500 + 425 = 9725 = the measured
        demand. A zero residuum is printed like any other, so a reader can see
        that it was computed rather than omitted."""
        ledger = _ledger(
            [
                (TERM_WEIGHTS, 7000),
                (TERM_MAMBA_POOL, 600),
                (TERM_ATTN_WORKSPACE, 300),
                (TERM_GRAPH_CAPTURE, 900),
                (TERM_HARDWARE_RESIDUAL, 500),
                (TERM_NVML_CARVE_OUT, 425),
            ]
        )
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        self.assertEqual(result.residuum_mib, 0)
        self.assertIn("RESIDUUM", result.render())
        self.assertIn("every measured MiB is claimed", result.residuum_note)


class TestHonestAbsences(unittest.TestCase):
    def test_a_term_with_no_bracketing_boundary_reads_unmeasured_not_zero(self):
        """Reporting 0 measured would manufacture a full-size overprediction
        on a term this instrument simply cannot see."""
        ledger = _ledger([("some unbracketed term", 1234)])
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        comparison = result.comparisons[0]
        self.assertIsNone(comparison.measured_mib)
        self.assertIsNone(comparison.error_mib)
        self.assertIn("UNMEASURED", comparison.row())
        self.assertIn("UNMEASURED terms are not evidence", result.render())

    def test_a_truncated_mark_log_names_the_phases_it_lacks(self):
        ledger = _ledger([(TERM_WEIGHTS, 7000)])
        marks = _boot_marks()[:2]  # boot died after weights_loaded
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        self.assertIn("kv_pool_sized", result.missing_phases)
        self.assertIn("INCOMPLETE", result.render())
        # The one term whose gap DID close is still measured.
        self.assertEqual(result.comparisons[0].measured_mib, 7000)

    def test_activation_is_labelled_a_lower_bound(self):
        """The first forward is not necessarily the deepest the rank sees, so
        this measurement bounds the term from below and must say so."""
        self.assertIn("LOWER BOUND", TERM_TO_POST[TERM_ACTIVATION][1])


class TestParentContext(unittest.TestCase):
    """TERM_PARENT_CONTEXT is dead on the production path (engine.py hardcodes
    parent_binds_cuda_context=False). Whether that is an under-charge is a
    question NVML answers directly, not one to infer from one card sitting
    higher than its siblings."""

    def test_a_foreign_process_on_the_card_is_measured(self):
        marks = _boot_marks()
        marks[5]["nvml_processes"] = {"100": 21300 * MIB, "42": 500 * MIB}
        ledger = _ledger([(TERM_PARENT_CONTEXT, 0)])
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        self.assertEqual(result.comparisons[0].measured_mib, 500)
        self.assertEqual(result.comparisons[0].error_mib, -500)

    def test_a_rank_only_card_measures_a_real_zero(self):
        ledger = _ledger([(TERM_PARENT_CONTEXT, 0)])
        result = reconcile_card(ledger, _boot_marks(), rank=0, rank_pids=[100])
        self.assertEqual(result.comparisons[0].measured_mib, 0)
        self.assertIsNotNone(
            result.comparisons[0].measured_mib, "a measured zero, not an absence"
        )


class TestWholeBoot(unittest.TestCase):
    def test_cards_are_matched_to_ranks_through_the_ledger_not_by_index(self):
        payload = {
            "boot_id": "boot1",
            "cards": [
                _ledger([(TERM_WEIGHTS, 7000)]) | {"gpu_id": 2, "ranks": [1]},
                _ledger([(TERM_WEIGHTS, 7000)]) | {"gpu_id": 0, "ranks": [0]},
            ],
        }
        marks = {0: _boot_marks(), 1: _boot_marks()}
        results = reconcile(payload, marks)
        self.assertEqual([(r.gpu_id, r.rank) for r in results], [(2, 1), (0, 0)])

    def test_a_rank_that_left_no_marks_is_skipped_not_invented(self):
        payload = {"cards": [_ledger([(TERM_WEIGHTS, 7000)]) | {"ranks": [0, 1]}]}
        results = reconcile(payload, {0: _boot_marks()})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 0)


if __name__ == "__main__":
    unittest.main()
