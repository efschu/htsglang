"""#605 Section 3: per-runner gap arithmetic in reconcile._delta_bytes.

The global rule (first frm -> last to) mis-books per-runner allocations when
a speculative process runs ``initialize()`` twice (target + NEXTN draft).
Draft weights (e.g. 2052 MiB) were flowing into the state-pool gap because
the global span crossed the draft's weight load.

THE FIX: partition marks by ``extra.draft_worker``, compute a delta per runner,
sum across runners. ``first_forward`` fires once per runner with identical
values -- dedupe by taking the first, never sum, or the activation term doubles.

Old boots without runner tags: compute a single global delta and report that
runner attribution is unavailable, rather than defaulting missing tags to
False and silently mis-attributing.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    TERM_GRAPH_CAPTURE,
    TERM_MAMBA_POOL,
    TERM_WEIGHTS,
)
from sglang.srt.mem_ledger.reconcile import (
    reconcile_card,
    _delta_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20


def _mark(
    phase,
    *,
    reserved=0,
    self_bytes=0,
    non_torch=0,
    carve=0,
    procs=None,
    pid=100,
    draft_worker=None,
):
    """Build a single phase mark.

    ``draft_worker``: True for the draft runner, False for the target,
    None when the tag is absent (old boot behaviour).
    """
    record: dict = {
        "phase": phase,
        "rank": 0,
        "pid": pid,
        "boot_id": "boot-spec",
        "reserved_bytes": reserved * MIB,
        "nvml_self_bytes": self_bytes * MIB,
        "non_torch_bytes": non_torch * MIB,
        "nvml_carve_out_bytes": carve * MIB,
        "nvml_processes": procs or {},
        "monotonic": 0.0,
    }
    if draft_worker is not None:
        record["extra"] = {"draft_worker": draft_worker}
    return record


def _dual_runner_marks():
    """Build a synthetic marks sequence for a speculative boot with two runners.

    Layout (reserved_bytes in MiB), MODELLED after the real 21:11 boot:

    process_start          0        {no tag -- process-level}
    pre_weight_load(F)    0        {draft_worker: False}  target
    weights_loaded(F)   10000      {draft_worker: False}  target +10000
    pre_weight_load(T)   8500       {draft_worker: True}   draft start (1500 freed)
    weights_loaded(T)   18000       {draft_worker: True}   draft +9500
    kv_pool_sized(F)    17000       {draft_worker: False}  target state-pool gap end
    kv_pool_sized(T)     28000      {draft_worker: True}   draft state-pool gap end
    capture_begin(F)     28000      {draft_worker: False}
    capture_begin(T)     28000      {draft_worker: True}
    capture_end(F)       31000      {draft_worker: False}  +3000 target graphs
    capture_end(T)       31000      {draft_worker: True}   cumulative same (draft +1000)
    boot_complete        31000      {no tag -- process-level}
    first_forward        31400      {no tag -- process-level}
    first_forward        31400      {no tag -- identical duplicate}

    Key arithmetic (per-runner deltas for weights_loaded -> kv_pool_sized):
    Target (F): 17000 - 10000 = 7000
    Draft  (T): 28000 - 18000 = 10000
    Sum:    17000 MiB.  Minus KV pool (12000) = 5000 MiB state pool.

    OLD global rule: first(weights_loaded)=10000, last(kv_pool_sized)=28000
    Global delta = 18000. Minus KV pool (12000) = 6000 MiB state pool.

    The old rule OVER-measures by 1000 MiB because it spans the 1500-MiB
    reserved drop (which the per-runner sum recovers as independent deltas
    of each runner).
    """
    return [
        _mark("process_start", reserved=0, self_bytes=0),
        _mark("pre_weight_load", reserved=0, self_bytes=0, draft_worker=False),
        _mark("weights_loaded", reserved=10000, self_bytes=10000, draft_worker=False),
        # Reserved drops 1500 MiB between target weights and draft pre_weight_load
        _mark("pre_weight_load", reserved=8500, self_bytes=8500, draft_worker=True),
        _mark("weights_loaded", reserved=18000, self_bytes=18000, draft_worker=True),
        _mark("kv_pool_sized", reserved=17000, self_bytes=17000, draft_worker=False),
        _mark("kv_pool_sized", reserved=28000, self_bytes=28000, draft_worker=True),
        _mark("capture_begin", reserved=28000, self_bytes=28000, draft_worker=False),
        _mark("capture_begin", reserved=28000, self_bytes=28000, draft_worker=True),
        _mark("capture_end", reserved=31000, self_bytes=31000, draft_worker=False),
        _mark("capture_end", reserved=31000, self_bytes=31000, draft_worker=True),
        _mark(
            "boot_complete",
            reserved=31000,
            self_bytes=31000,
            carve=425,
            procs={"100": 31000 * MIB},
        ),
        _mark("first_forward", reserved=31400, self_bytes=31400),
        _mark("first_forward", reserved=31400, self_bytes=31400),
    ]


def _ledger(terms, *, kv_pool=12000, demand=None):
    rows = [{"name": n, "mib": m, "provenance": "modeled"} for n, m in terms]
    return {
        "gpu_id": 1,
        "card": "NVIDIA GeForce RTX 5090",
        "ranks": [0],
        "kv_pool_mib": kv_pool,
        "demand_mib": demand if demand is not None else sum(m for _n, m in terms),
        "terms": rows,
    }


def _old_boot_marks():
    """Marks WITHOUT runner tags (old boot predating draft_worker tagging).

    A single-runner boot so the global delta is correct but attribution
    is unknown.
    """
    return [
        _mark("pre_weight_load", reserved=0, self_bytes=500, non_torch=500),
        _mark("weights_loaded", reserved=10000, self_bytes=10500, non_torch=500),
        _mark("kv_pool_sized", reserved=22000, self_bytes=22500, non_torch=500),
        _mark("capture_begin", reserved=22300, self_bytes=22800, non_torch=500),
        _mark("capture_end", reserved=23200, self_bytes=23700, non_torch=500),
        _mark(
            "boot_complete",
            reserved=23200,
            self_bytes=23700,
            non_torch=500,
            carve=425,
            procs={"100": 23700 * MIB},
        ),
        _mark("first_forward", reserved=23600, self_bytes=24100, non_torch=500),
    ]


class TestDeltaBytesPartitioning(unittest.TestCase):
    """Low-level tests for _delta_bytes per-runner logic."""

    def test_two_runners_partition_and_sum_weights(self):
        """Draft weights must NOT appear in the target runner's gap."""
        marks = _dual_runner_marks()
        delta, ambiguous = _delta_bytes(
            marks, "pre_weight_load", "weights_loaded", "reserved_bytes"
        )
        # Target (F): 10000 - 0 = 10000
        # Draft  (T): 18000 - 8500 = 9500
        # Sum = 19500
        self.assertEqual(delta, 19500 * MIB)
        self.assertFalse(ambiguous)

    def test_state_pool_delta_excludes_draft_weight_overlap(self):
        """Per-runner state-pool gap does not absorb the intermediate reserved
        drop that the global rule would span over.

        Target (F): 17000 - 10000 = 7000
        Draft  (T): 28000 - 18000 = 10000
        Sum = 17000 MiB. Minus KV pool (12000) = 5000 MiB state pool.

        OLD global rule: 28000 - 10000 = 18000 MiB.
        Minus KV pool (12000) = 6000 MiB state pool.
        Over-measures by 1000 MiB.
        """
        marks = _dual_runner_marks()
        delta, ambiguous = _delta_bytes(
            marks, "weights_loaded", "kv_pool_sized", "reserved_bytes"
        )
        self.assertEqual(delta, 17000 * MIB)
        self.assertFalse(ambiguous)

    def test_capture_sum_across_runners(self):
        """Both runners' capture costs add."""
        marks = _dual_runner_marks()
        delta, ambiguous = _delta_bytes(
            marks, "capture_begin", "capture_end", "reserved_bytes"
        )
        # Target (F): 31000 - 28000 = 3000
        # Draft  (T): 34000 - 28000 = 3000
        # Sum = 6000
        self.assertEqual(delta, 6000 * MIB)
        self.assertFalse(ambiguous)

    def test_missing_runner_tag_returns_ambiguous_true(self):
        """Old boots without draft_worker tags: global delta is computed,
        but ``had_ambiguous`` is True so the caller knows."""
        marks = _old_boot_marks()
        delta, ambiguous = _delta_bytes(
            marks, "pre_weight_load", "weights_loaded", "reserved_bytes"
        )
        self.assertEqual(delta, 10000 * MIB)
        self.assertTrue(ambiguous)

    def test_first_forward_deduped_in_untagged_set(self):
        """Process-level phases that fire per-runner with identical values
        must not double-count. ``first_forward`` fires twice with identical
        reserved_bytes; after dedup the gap is still correct."""
        marks = _dual_runner_marks()
        delta, ambiguous = _delta_bytes(
            marks, "boot_complete", "first_forward", "reserved_bytes"
        )
        # boot_complete and first_forward have no runner tags.
        # They go into the "other" partition.
        # No tagged marks exist for these phases, so tagged lists are empty.
        # Falls through to global delta path.
        # boot_complete: 31000; first_forward (first): 31400
        # delta = 400 * MIB. ambiguous=True.
        self.assertEqual(delta, 400 * MIB)
        self.assertTrue(ambiguous)


class TestReconcileCardPerRunner(unittest.TestCase):
    """Integration: reconcile_card with dual-runner marks."""

    def test_state_pool_term_sees_per_runner_measurement(self):
        """With per-runner arithmetic, TERM_MAMBA_POOL measures exactly what
        the ledger models, not what the ledger models PLUS draft weights.

        Per-runner state-pool delta: 17000 MiB.
        Minus KV pool (12000) = 5000 MiB state pool.
        """
        ledger = _ledger(
            [
                (TERM_WEIGHTS, 19500),
                (TERM_MAMBA_POOL, 5000),
            ],
            kv_pool=12000,
        )
        marks = _dual_runner_marks()
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        by_term = {c.term: c for c in result.comparisons}
        self.assertEqual(by_term[TERM_MAMBA_POOL].measured_mib, 5000)
        self.assertEqual(by_term[TERM_MAMBA_POOL].error_mib, 0)

    def test_weights_term_takes_the_LARGEST_episode_never_the_sum(self):
        """FALSIFIED PREMISE, corrected here (#605 second reconcile run).

        This test previously asserted that the weights term measures the SUM
        of the target and draft weight loads, and it was wrong for a reason
        the marks state plainly: between the two loads the process FREES the
        first. On the ship boot 1464299 the 5090's reserved bytes fell from
        21724 to 8758 MiB between the target's ``kv_pool_sized`` and the next
        runner's ``pre_weight_load``. Summing across that free produced a
        weights row of 27800 MiB on a card whose entire process footprint at
        ``boot_complete`` was 26364 MiB -- an impossible post that nonetheless
        looked like a measurement.

        The card must fund the LARGEST episode, because that is the most
        weight it holds at any one instant. Here that is the target's 10000
        MiB, not 10000 + 9500.
        """
        ledger = _ledger(
            [(TERM_WEIGHTS, 10000)],
            kv_pool=12000,
        )
        marks = _dual_runner_marks()
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        by_term = {c.term: c for c in result.comparisons}
        self.assertEqual(by_term[TERM_WEIGHTS].measured_mib, 10000)
        self.assertEqual(by_term[TERM_WEIGHTS].error_mib, 0)
        self.assertNotEqual(by_term[TERM_WEIGHTS].measured_mib, 19500)
        # Every episode is still named, so nothing is absorbed silently.
        self.assertIn("9500", by_term[TERM_WEIGHTS].note)

    def test_capture_term_sums_both_runners(self):
        """Graph capture costs from both runners add."""
        ledger = _ledger(
            [(TERM_GRAPH_CAPTURE, 6000)],  # modeled: 3000 + 3000
            kv_pool=12000,
        )
        marks = _dual_runner_marks()
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        by_term = {c.term: c for c in result.comparisons}
        self.assertEqual(by_term[TERM_GRAPH_CAPTURE].measured_mib, 6000)
        self.assertEqual(by_term[TERM_GRAPH_CAPTURE].error_mib, 0)

    def test_old_boot_flags_ambiguous_runner_terms(self):
        """A boot without runner tags: measurement is still computed but
        the CardReconciliation reports which terms are ambiguous."""
        ledger = _ledger(
            [
                (TERM_WEIGHTS, 10000),
                (TERM_MAMBA_POOL, 5000),
            ],
            kv_pool=12000,
        )
        marks = _old_boot_marks()
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        by_term = {c.term: c for c in result.comparisons}
        self.assertEqual(by_term[TERM_WEIGHTS].measured_mib, 10000)
        # The weights term no longer needs runner tags at all: episodes are
        # delimited by the marks' own ORDER, so a boot without tags is
        # measured exactly, not ambiguously. The ambiguity flag survives for
        # the terms that are still computed with a runner-partitioned delta
        # (the state pool below), which is what it was built for.
        self.assertNotIn(TERM_WEIGHTS, result.ambiguous_runner_terms)
        self.assertIn(TERM_MAMBA_POOL, result.ambiguous_runner_terms)
        self.assertIn("AMBIGUOUS RUNNER", result.render())


class TestCanFailRevertProof(unittest.TestCase):
    """Can-fail proof: the old global rule mis-books draft weights into the
    state-pool term.

    WHY THIS MATTERS: without this, a test that passes only under the new
    code could be a false positive. The can-fail proves the NEW assertion
    actually depends on the code change.
    """

    def test_old_global_rule_over_measures_state_pool(self):
        """Compute the old global-rule delta manually and prove it over-
        measures the state pool by 1000 MiB (the intermediate reserved drop)."""
        marks = _dual_runner_marks()

        # Old global rule: first(weights_loaded) -> last(kv_pool_sized)
        first_ww = next(m for m in marks if m.get("phase") == "weights_loaded")
        last_kp = list(
            reversed([m for m in marks if m.get("phase") == "kv_pool_sized"])
        )[0]
        old_global_raw = int(last_kp["reserved_bytes"]) - int(
            first_ww["reserved_bytes"]
        )
        old_state_pool_mib = old_global_raw // MIB - 12000

        # New per-runner rule
        new_delta, _ = _delta_bytes(
            marks, "weights_loaded", "kv_pool_sized", "reserved_bytes"
        )
        new_state_pool_mib = new_delta // MIB - 12000

        # Old global rule over-measures by 1000 MiB
        self.assertEqual(old_state_pool_mib, 6000)
        self.assertEqual(new_state_pool_mib, 5000)
        self.assertEqual(old_state_pool_mib - new_state_pool_mib, 1000)

    def test_new_rule_matches_modeled_state_pool(self):
        """The per-runner measurement matches what the ledger models."""
        ledger = _ledger(
            [(TERM_MAMBA_POOL, 5000)],
            kv_pool=12000,
        )
        marks = _dual_runner_marks()
        result = reconcile_card(ledger, marks, rank=0, rank_pids=[100])
        by_term = {c.term: c for c in result.comparisons}
        # Error 0 proves the new rule gives the correct measurement.
        self.assertEqual(by_term[TERM_MAMBA_POOL].error_mib, 0)


if __name__ == "__main__":
    unittest.main()
