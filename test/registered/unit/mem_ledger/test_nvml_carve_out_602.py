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

import types
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
from sglang.srt.server_args import ServerArgs
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


class _ReserveStub:
    """A ServerArgs at the moment the reserve is decided, wired to the REAL
    ledger and the REAL reserve_demand_per_gpu.

    Nothing here supplies the two quantities under test: the carve-out comes
    out of ``build_card_ledgers`` from the card's own reserved_mib, and the
    user reserve comes out of ``ServerArgs.user_reserve_mib_per_gpu``. So if
    either carrier is removed from production code, this stub cannot paper
    over it -- which is the point of pinning the invariant here.
    """

    def __init__(self, user_reserve_mib=1024, reserved_mib=RTX_3080_RESERVED_MIB):
        self.chunked_prefill_size = 2048
        self.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24, bs=None)
        )
        self.tp_size = 1
        self.rank_gpu_id = [1]
        self.rank_user_reserve_mib = user_reserve_mib
        self._reserved_mib = reserved_mib

    def _apply_gpu_mem_capacity_defaults(self, gpu_mem):
        pass

    def _widen_decode_capture_to_session_ceiling(self, decode_cfg):
        pass

    def ledger_full_demand_per_gpu(self, gpu_mem=None):
        """The ledger's own demand for this card, exactly as the real wrapper
        would return it: ``{gpu_id: demand_outside_budget_mib(ledger)}``.

        Stubbed at this seam rather than one layer lower because the real
        wrapper refuses whenever any term is unbounded, and the hardware
        residual is unbounded without a calibration matching this rig's
        fingerprint -- which a hermetic test cannot supply and should not
        fake. The carve-out still reaches the assertions THROUGH this value
        (it is a term inside that ledger), so both carriers remain delta-
        provable against the real ``reserve_demand_per_gpu``.
        """
        return {
            1: int(demand_outside_budget_mib(ledger(self._reserved_mib)))
        }

    def reserve_demand_per_gpu(self, gpu_mem, counts):
        return ServerArgs.reserve_demand_per_gpu(self, gpu_mem, counts)

    def user_reserve_mib_per_gpu(self, rank_gpu_id):
        return ServerArgs.user_reserve_mib_per_gpu(self, rank_gpu_id)


class TestTheCorridorZeroIsCoupledToTheDemand(unittest.TestCase):
    """THE PIN for planner_corridor_mib() == 0.

    The zero is correct ONLY while the reserve demand carries both the NVML
    carve-out and the user reserve. If a future change takes either back out,
    this class goes red -- so the defect surfaces in a test run rather than in
    a boot that quietly serves under the user's floor again.

    The repair when it does go red is to put the term back, NOT to make the
    corridor non-zero: a constant there cannot know a card's carve-out or the
    user's number, and would double-count on any card still carrying them.
    """

    def setUp(self):
        ServerArgs._full_demand_refusal_named = False

    def test_carrier_one_the_demand_moves_with_the_user_reserve(self):
        at_1024 = _ReserveStub(user_reserve_mib=1024).reserve_demand_per_gpu(
            RTX_3080_TOTAL_MIB, {1: 1}
        )
        at_2048 = _ReserveStub(user_reserve_mib=2048).reserve_demand_per_gpu(
            RTX_3080_TOTAL_MIB, {1: 1}
        )
        self.assertEqual(
            at_2048[1] - at_1024[1],
            1024,
            "the reserve demand does not move with --rank-user-reserve-mib, "
            "so the user's headroom is being spent on KV and a corridor of 0 "
            "no longer protects it (#602)",
        )

    def test_carrier_two_the_demand_moves_with_the_carve_out(self):
        small = _ReserveStub(reserved_mib=RTX_3080_RESERVED_MIB).reserve_demand_per_gpu(
            RTX_3080_TOTAL_MIB, {1: 1}
        )
        large = _ReserveStub(reserved_mib=RTX_5090_RESERVED_MIB).reserve_demand_per_gpu(
            RTX_3080_TOTAL_MIB, {1: 1}
        )
        self.assertEqual(
            large[1] - small[1],
            RTX_5090_RESERVED_MIB - RTX_3080_RESERVED_MIB,
            "the reserve demand does not move with the NVML carve-out, so the "
            "budget is spending memory no allocation can obtain and a "
            "corridor of 0 no longer protects it (#602)",
        )

    def test_and_therefore_the_corridor_adds_nothing(self):
        """Stated as one assertion so the coupling is not separable: the zero
        is a CONSEQUENCE of the two carriers above, not an independent
        policy choice."""
        from sglang.srt.uneven_perf import planner_corridor_mib

        stub = _ReserveStub()
        demand = stub.reserve_demand_per_gpu(RTX_3080_TOTAL_MIB, {1: 1})[1]
        bare = int(demand_outside_budget_mib(ledger(RTX_3080_RESERVED_MIB)))
        self.assertEqual(
            demand - bare,
            1024,
            "the user reserve is the only thing reserve_demand_per_gpu should "
            "add on top of the ledger demand",
        )
        self.assertEqual(
            planner_corridor_mib(),
            0,
            "with both quantities inside the demand, any extra corridor "
            "double-counts them -- the forbidden pad",
        )


if __name__ == "__main__":
    unittest.main()


class TestTheCarveOutSurvivesTheProductionCardPath(unittest.TestCase):
    """The gap the first cut of #602 shipped.

    Every test above builds CardFacts by hand with reserved_mib set, so they
    all passed while the BOOT priced the term at zero: the card object the
    boot actually resolves (_RankGpuCard, via memory_info_for_uuid) had no
    such field, and _build_card_ledgers read it with getattr(..., 0). The
    demand moved by exactly the user reserve and not one MiB more, on a log
    that otherwise looked correct.

    These pin the CHAIN rather than the endpoint: NVML -> MemoryInfo ->
    _RankGpuCard -> CardFacts. A hand-built fixture cannot substitute for it.
    """

    def test_memory_info_carries_reserved_and_allocatable(self):
        from sglang.srt.registry.nvml import MemoryInfo

        mem = MemoryInfo(
            total_bytes=RTX_3080_TOTAL_MIB * MIB,
            free_bytes=1000 * MIB,
            used_bytes=19000 * MIB,
            reserved_bytes=RTX_3080_RESERVED_MIB * MIB,
        )
        self.assertEqual(mem.reserved_mib, RTX_3080_RESERVED_MIB)
        self.assertEqual(
            mem.allocatable_mib, RTX_3080_TOTAL_MIB - RTX_3080_RESERVED_MIB
        )

    def test_the_rank_card_requires_the_field_instead_of_defaulting(self):
        """_RankGpuCard must REFUSE to be built without it. If this ever
        becomes optional again, the carve-out can silently price at 0 in a
        boot while every hand-built fixture still passes."""
        from sglang.srt.server_args import _RankGpuCard

        with self.assertRaises(TypeError):
            _RankGpuCard(
                cuda_ordinal=0,
                nvml_index=1,
                uuid="GPU-x",
                pci_bus_id="00000000:05:00.0",
                name="NVIDIA GeForce RTX 3080",
                total_mib=RTX_3080_TOTAL_MIB,
                free_mib=1000,
            )

    def test_card_facts_reads_the_attribute_not_a_getattr_default(self):
        """Reads the source, because the defect was invisible at runtime: a
        getattr default turns a missing field into a priced-zero term."""
        import inspect

        from sglang.srt.server_args import ServerArgs

        src = inspect.getsource(ServerArgs._build_card_ledgers)
        self.assertIn("reserved_mib=card.reserved_mib", src)
        self.assertNotIn('getattr(card, "reserved_mib"', src)
