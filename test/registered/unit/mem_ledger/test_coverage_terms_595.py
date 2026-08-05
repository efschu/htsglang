"""#595: the two coverage gaps become first-class terms.

Window 7 OOMed with a reserve that summed only the terms the ledger happened
to have. Two real allocations had no post at all:

  * NCCL communicator buffers -- no post whatsoever, on a configuration that
    demonstrably uses NCCL (custom all-reduce is disabled for --rank-gpu-id on
    heterogeneous cards, so TP collectives fall back to it);
  * every attention backend's private workspace except flashinfer's.

Both now exist as terms. NCCL starts UNBOUNDED by design -- it cannot be
derived, only measured -- so the full-demand reserve refuses until a window
supplies the number. That refusal is the point: it is the difference between
"we do not know" and a zero that reads like "there is none".
"""

import types
import unittest

from sglang.srt.mem_ledger.engine import (
    TERM_ATTN_WORKSPACE,
    TERM_NCCL_BUFFERS,
    TRTLLM_MHA_WORKSPACE_MIB,
    TRTLLM_MLA_WORKSPACE_MIB,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
    demand_outside_budget_mib,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CARD = CardFacts(
    gpu_id=1,
    uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
    name="NVIDIA GeForce RTX 3080",
    total_mib=20480,
)


def inputs(**over):
    base = dict(
        weight_mib_per_rank=[0],
        activation_mib_per_rank=[1766.0],
        capture_tokens_per_rank=[96],
        capture_mib_per_rank=[640.0],
        phase_footprint_source_per_rank=["[upper_bound] reference window"],
        phase_footprint_fingerprint="a191a0712717",
        mamba_pool_mib_per_rank=[512.0],
        gdn_scratch_mib_per_rank=[300.0],
        indexer_scratch_mib_per_rank=[120.0],
        indexer_chunk_cap_mib=256,
        flashinfer_workspace_mib=200,
        chunked_prefill_size=2048,
        max_running_requests=4,
        # Priced so the NCCL term is not the thing under test unless a case
        # says so.
        nccl_buffer_mib_per_gpu={1: 128.0},
        nccl_signature="tp3.dcp3.pp1",
    )
    base.update(over)
    return DemandInputs(**base)


def ledger(**over):
    return build_card_ledgers(
        inputs(**over), cards=[CARD], rank_gpu_id=[1], user_reserve_mib={1: 1024}
    )[0]


def demand(**over) -> int:
    return int(demand_outside_budget_mib(ledger(**over)))


class TestNcclTermBinds(unittest.TestCase):
    def test_the_nccl_term_moves_the_sum_by_its_own_delta(self):
        before = demand()
        after = demand(nccl_buffer_mib_per_gpu={1: 128.0 + 96})
        self.assertEqual(after - before, 96)

    def test_it_is_a_named_term_on_the_ledger(self):
        names = [t.name for t in ledger().terms]
        self.assertIn(TERM_NCCL_BUFFERS, names)

    def test_unmeasured_nccl_is_unbounded_not_zero(self):
        """The whole reason the term exists. A zero here would read as 'NCCL
        allocates nothing', which is false on every boot that uses it."""
        lg = ledger(nccl_buffer_mib_per_gpu=None)
        self.assertTrue(any(TERM_NCCL_BUFFERS in u for u in lg.unbounded), lg.unbounded)
        self.assertNotIn(TERM_NCCL_BUFFERS, [t.name for t in lg.terms])

    def test_the_refusal_says_how_to_measure_it(self):
        lg = ledger(nccl_buffer_mib_per_gpu=None)
        msg = next(u for u in lg.unbounded if TERM_NCCL_BUFFERS in u)
        self.assertIn("communicator init", msg)
        self.assertIn("NCCL_DEBUG", msg)

    def test_a_measurement_carries_the_communicator_set_it_is_valid_for(self):
        """Keyed on the signature, not on the hardware fingerprint alone: the
        buffers scale with peer and channel counts, which come from
        tp/dcp/pp sizes -- configuration, not hardware."""
        term = next(t for t in ledger().terms if t.name == TERM_NCCL_BUFFERS)
        self.assertEqual(term.fingerprint, "tp3.dcp3.pp1")
        self.assertIn("tp3.dcp3.pp1", term.derivation)

    def test_colocated_ranks_each_pay_their_own_communicator(self):
        two = build_card_ledgers(
            inputs(
                weight_mib_per_rank=[0, 0],
                activation_mib_per_rank=[1766.0, 1766.0],
                capture_tokens_per_rank=[96, 96],
                capture_mib_per_rank=[640.0, 640.0],
                phase_footprint_source_per_rank=["s", "s"],
                mamba_pool_mib_per_rank=[512.0, 512.0],
                gdn_scratch_mib_per_rank=[300.0, 300.0],
                indexer_scratch_mib_per_rank=[120.0, 120.0],
            ),
            cards=[CARD],
            rank_gpu_id=[1, 1],
            user_reserve_mib={1: 1024},
        )[0]
        term = next(t for t in two.terms if t.name == TERM_NCCL_BUFFERS)
        self.assertEqual(term.mib, 256)


class TestAttentionBackendDispatch(unittest.TestCase):
    def test_unstated_backend_is_byte_identical_to_before(self):
        """This rig runs flashinfer and every existing caller omits the field;
        neither may change."""
        self.assertEqual(demand(attention_backend=""), demand())
        self.assertEqual(demand(attention_backend="flashinfer"), demand())

    def test_trtllm_mla_is_priced_from_its_own_constant(self):
        d = demand(attention_backend="trtllm_mla", flashinfer_workspace_mib=200)
        base_without_ws = demand(flashinfer_workspace_mib=0)
        self.assertEqual(d - base_without_ws, TRTLLM_MLA_WORKSPACE_MIB)

    def test_trtllm_mha_is_priced_from_its_own_constant(self):
        d = demand(attention_backend="trtllm_mha", flashinfer_workspace_mib=200)
        base_without_ws = demand(flashinfer_workspace_mib=0)
        self.assertEqual(d - base_without_ws, TRTLLM_MHA_WORKSPACE_MIB)

    def test_the_two_fixed_backends_are_not_the_same_number(self):
        """150 vs 512: a single constant for 'trtllm' would be wrong for one
        of them by 362 MiB."""
        self.assertNotEqual(TRTLLM_MLA_WORKSPACE_MIB, TRTLLM_MHA_WORKSPACE_MIB)

    def test_an_unpriced_backend_is_unbounded_not_zero(self):
        lg = ledger(attention_backend="aiter")
        self.assertTrue(
            any(TERM_ATTN_WORKSPACE in u for u in lg.unbounded), lg.unbounded
        )

    def test_the_unpriced_backend_refusal_names_the_backend(self):
        lg = ledger(attention_backend="cutlass_mla")
        msg = next(u for u in lg.unbounded if TERM_ATTN_WORKSPACE in u)
        self.assertIn("cutlass_mla", msg)


class TestTheIntendedInteraction(unittest.TestCase):
    """With NCCL unbounded by default the reserve refuses on every boot until
    a window measures it. That is intended; what must NOT change is the
    fallback the refusal lands on."""

    def setUp(self):
        from sglang.srt.server_args import ServerArgs

        ServerArgs._full_demand_refusal_named = False
        self.SA = ServerArgs

    def test_an_unmeasured_nccl_term_refuses_the_full_demand_path(self):
        stub = types.SimpleNamespace()
        stub._build_card_ledgers = lambda: [ledger(nccl_buffer_mib_per_gpu=None)]
        stub._apply_gpu_mem_capacity_defaults = lambda gpu_mem: None
        stub._widen_decode_capture_to_session_ceiling = lambda cfg: None
        stub.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24, bs=None)
        )
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            got = self.SA.ledger_full_demand_per_gpu(stub, 20480)
        self.assertIsNone(got)
        self.assertIn("NCCL", "\n".join(cm.output))

    def test_a_measured_nccl_term_lets_the_path_answer(self):
        """A fully priced ledger. Built explicitly rather than from the
        fixture above, because that one has no VRAM calibration and would
        refuse on the hardware-residual term instead -- which would make this
        test pass or fail for the wrong reason."""
        priced = types.SimpleNamespace(
            gpu_id=1,
            card="NVIDIA GeForce RTX 3080",
            unbounded=(),
            terms=(
                types.SimpleNamespace(name="runtime activation + metadata", mib=1766),
                types.SimpleNamespace(name=TERM_NCCL_BUFFERS, mib=128),
            ),
        )
        stub = types.SimpleNamespace()
        stub._build_card_ledgers = lambda: [priced]
        stub._apply_gpu_mem_capacity_defaults = lambda gpu_mem: None
        stub._widen_decode_capture_to_session_ceiling = lambda cfg: None
        stub.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24, bs=None)
        )
        got = self.SA.ledger_full_demand_per_gpu(stub, 20480)
        self.assertEqual(got, {1: 1894})


if __name__ == "__main__":
    unittest.main()
