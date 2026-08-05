"""#602: the part of a card that is never allocatable becomes a term.

Every rank of the production boot sat 1.9-2.5 GiB above its ledger budget and
the binding 3080 ran at 295 MiB free, under a user limit of 1024. None of it
was an untracked allocation. The budget was derived as ``nvml_total - demand``,
and ``nvml_total`` is the NOMINAL board size: the driver holds a slice of it
back and never hands it to anyone. Measured on this rig with every process
gone -- card idle, ``used=0`` -- a 20480 MiB RTX 3080 reports 20055 MiB free
and 425 MiB reserved, and a 32607 MiB RTX 5090 reports 518 MiB reserved.

The failure is quiet in the obvious check, which is why it survived: NVML's v2
memory struct subtracts the carve-out from BOTH ``used`` and ``free``, so
``total - used - free`` reads ~0 and nothing looks wrong. It only shows up when
an allocation the budget promised would fit does not.

The number is READ, not modelled and not probed -- see Provenance.REPORTED.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    TERM_NVML_CARVE_OUT,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
    demand_outside_budget_mib,
)
from sglang.srt.mem_ledger.terms import Provenance
from sglang.srt.registry.nvml import DeviceInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20
# Measured on this rig, both cards idle, via nvmlDeviceGetMemoryInfo v2.
RTX_3080_TOTAL_MIB = 20480
RTX_3080_RESERVED_MIB = 425
RTX_5090_TOTAL_MIB = 32607
RTX_5090_RESERVED_MIB = 518


def card(reserved_mib: int = RTX_3080_RESERVED_MIB) -> CardFacts:
    return CardFacts(
        gpu_id=1,
        uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
        name="NVIDIA GeForce RTX 3080",
        total_mib=RTX_3080_TOTAL_MIB,
        reserved_mib=reserved_mib,
    )


def inputs(ranks: int = 1) -> DemandInputs:
    return DemandInputs(
        weight_mib_per_rank=[0] * ranks,
        activation_mib_per_rank=[1766.0] * ranks,
        capture_tokens_per_rank=[96] * ranks,
        capture_mib_per_rank=[640.0] * ranks,
        phase_footprint_source_per_rank=["[upper_bound] reference window"] * ranks,
        phase_footprint_fingerprint="a191a0712717",
        mamba_pool_mib_per_rank=[512.0] * ranks,
        gdn_scratch_mib_per_rank=[300.0] * ranks,
        indexer_scratch_mib_per_rank=[120.0] * ranks,
        indexer_chunk_cap_mib=256,
        flashinfer_workspace_mib=200,
        chunked_prefill_size=2048,
        max_running_requests=4,
        nccl_buffer_mib_per_gpu={1: 128.0},
        nccl_signature="tp3.dcp3.pp1",
    )


def ledger(reserved_mib: int = RTX_3080_RESERVED_MIB, ranks: int = 1):
    return build_card_ledgers(
        inputs(ranks),
        cards=[card(reserved_mib)],
        rank_gpu_id=[1] * ranks,
        user_reserve_mib={1: 1024},
    )[0]


class TestTheCarveOutIsReadFromTheDriver(unittest.TestCase):
    def test_nvml_reports_reserved_and_allocatable_is_total_minus_it(self):
        info = DeviceInfo(
            index=0,
            uuid="GPU-5c648f96",
            name="NVIDIA GeForce RTX 3080",
            total_bytes=RTX_3080_TOTAL_MIB * MIB,
            reserved_bytes=RTX_3080_RESERVED_MIB * MIB,
            pci_bus_id="00000000:05:00.0",
        )
        self.assertEqual(info.total_mib, RTX_3080_TOTAL_MIB)
        self.assertEqual(info.reserved_mib, RTX_3080_RESERVED_MIB)
        self.assertEqual(
            info.allocatable_mib,
            RTX_3080_TOTAL_MIB - RTX_3080_RESERVED_MIB,
            "allocatable is what a budget may spend; spending total is #602",
        )

    def test_a_driver_without_the_v2_field_reports_zero_not_a_guess(self):
        """No fallback constant. A card whose driver does not report the
        carve-out keeps the old behaviour visibly, rather than inheriting a
        number measured on a different card."""
        info = DeviceInfo(
            index=0,
            uuid="GPU-x",
            name="card",
            total_bytes=RTX_3080_TOTAL_MIB * MIB,
            pci_bus_id="00000000:05:00.0",
        )
        self.assertEqual(info.reserved_mib, 0)
        self.assertEqual(info.allocatable_mib, RTX_3080_TOTAL_MIB)


class TestTheTermBinds(unittest.TestCase):
    def test_the_term_carries_the_reported_mib_and_says_so(self):
        terms = {t.name: t for t in ledger().terms}
        self.assertIn(TERM_NVML_CARVE_OUT, terms)
        term = terms[TERM_NVML_CARVE_OUT]
        self.assertEqual(term.mib, RTX_3080_RESERVED_MIB)
        self.assertEqual(
            term.provenance,
            Provenance.REPORTED,
            "the driver states this number; calling it CALIBRATED would "
            "imply a probe and a fingerprint that do not exist for it",
        )

    def test_the_sum_moves_by_the_carve_outs_own_delta(self):
        """The bind proof: a different reserved figure has to move the demand
        by exactly that difference, or the term is decorative."""
        before = int(demand_outside_budget_mib(ledger(RTX_3080_RESERVED_MIB)))
        after = int(demand_outside_budget_mib(ledger(RTX_5090_RESERVED_MIB)))
        self.assertEqual(after - before, RTX_5090_RESERVED_MIB - RTX_3080_RESERVED_MIB)

    def test_a_card_reporting_no_carve_out_gets_no_row(self):
        names = {t.name for t in ledger(0).terms}
        self.assertNotIn(
            TERM_NVML_CARVE_OUT,
            names,
            "a zero row would read as 'measured none' where the truth is "
            "'the driver did not say'",
        )

    def test_it_is_charged_once_per_card_not_once_per_rank(self):
        """It is one reservation the driver makes against the board. Two
        co-located ranks do not double it, and charging it per process would
        overstate a shared-GPU boot by a whole carve-out."""
        one = int(demand_outside_budget_mib(ledger(ranks=1)))
        two = int(demand_outside_budget_mib(ledger(ranks=2)))
        one_terms = {t.name: t.mib for t in ledger(ranks=1).terms}
        two_terms = {t.name: t.mib for t in ledger(ranks=2).terms}
        self.assertEqual(one_terms[TERM_NVML_CARVE_OUT], RTX_3080_RESERVED_MIB)
        self.assertEqual(two_terms[TERM_NVML_CARVE_OUT], RTX_3080_RESERVED_MIB)
        # The rest of the ledger does scale with rank count; this asserts the
        # carve-out specifically is NOT what grew.
        self.assertGreater(two, one)


class TestThePlannerCorridorIsNoLongerAConstant(unittest.TestCase):
    """#602: the planner's release criterion was a hardcoded 400 MiB while the
    user's rule is 1024, and the two were never the same number.

    The fix is not to restate 1024 here. It is that the demand now carries the
    carve-out and the user reserve, so the planner's existing binding check
    (``_unfundable_reason``: residual < demand is UNBOOTABLE) already means
    "the user keeps their MiB", and anything added on top would be a pad.
    """

    def test_the_default_extra_corridor_is_zero(self):
        from sglang.srt.uneven_perf import planner_corridor_mib

        self.assertEqual(
            planner_corridor_mib(),
            0,
            "a non-zero default is a margin on top of the user's own "
            "reserve, invented by the code",
        )

    def test_it_no_longer_reads_the_daemons_emergency_floor(self):
        """The reservation daemon's 400 MiB floor answers a different
        question and must not double as the planner's release criterion."""
        from sglang.srt.registry.ledger import DEFAULT_CORRIDOR_BYTES
        from sglang.srt.uneven_perf import planner_corridor_mib

        self.assertEqual(DEFAULT_CORRIDOR_BYTES // MIB, 400)
        self.assertNotEqual(planner_corridor_mib(), DEFAULT_CORRIDOR_BYTES // MIB)

    def test_the_experiment_seam_still_works(self):
        import os

        from sglang.srt.uneven_perf import planner_corridor_mib

        os.environ["SGLANG_PLANNER_CORRIDOR_MIB"] = "256"
        try:
            self.assertEqual(planner_corridor_mib(), 256)
        finally:
            del os.environ["SGLANG_PLANNER_CORRIDOR_MIB"]


if __name__ == "__main__":
    unittest.main()
