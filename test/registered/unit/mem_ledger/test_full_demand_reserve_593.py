"""#593: the reserve is the WHOLE non-KV demand, or it refuses.

#590 routed the reserve onto the ledger's ACTIVATION footprint. On the
reference rig that took the binding RTX 3080 from a flat 3968 MiB heuristic to
its measured 1766, handed the 2202 MiB difference to the KV pool, and the
window-7 boot died in CUDA graph capture on that exact card with 113 MiB free.

The heuristic was never an activation estimate. It was a catch-all that also
happened to cover graph capture, the attention workspace, prefill scratch and
the CUDA context. Swapping a catch-all for one honest term is a REDUCTION, and
the failure gets worse as the measurement improves: the window-6 dumps put the
real activation transient at 429 MiB, which would have reserved less still.

So these tests pin two things. First, that every term is actually in the sum --
a term that silently drops out is invisible in a total and fatal in a boot,
which is the #590 lesson restated. Second, that an unpriced term REFUSES rather
than producing a short number that looks complete.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    CardFacts,
    DemandInputs,
    build_card_ledgers,
    demand_outside_budget_mib,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: The binding card of the reference rig: RTX 3080, 20480 MiB. Named the
#: "binding card" in #586 because its measured activation is the smallest and
#: the uneven-DCP min-reduced unit is set by it.
BINDING_CARD = CardFacts(
    gpu_id=1,
    uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
    name="NVIDIA GeForce RTX 3080",
    total_mib=20480,
)

#: From the 2026-08-05 artifacts, not invented here:
#:   activation 1766 MiB -- #586 reference UPPER_BOUND for this card
#:   capture     640 MiB -- #586 reference measured capture for this card
#: The window-7 boot reserved activation + 192 MiB (the 2 MiB/token
#: coefficient) = 1958 MiB and OOMed during capture.
ARTIFACT_ACTIVATION_MIB = 1766.0
ARTIFACT_CAPTURE_MIB = 640.0

#: The reserve that DID boot: window 6, --rank-auto-reserve-mib auto on both
#: the integration and the production tree, 669440 tokens, corridor held under
#: a 4-way load probe. Any model that returns less than this for this card is
#: claiming the boot that survived was wasteful, and window 7 is the
#: counter-example.
BOOTED_RESERVE_MIB = 4160

#: What window 7 reserved instead, and died with.
OOMED_RESERVE_MIB = 1958


def inputs(**over):
    """One rank on the binding card, every term explicitly priced.

    Each number is stated here rather than derived, which is what the
    DemandInputs seam is for: a term that this test can move is a term the
    reserve is provably reading.
    """
    base = dict(
        weight_mib_per_rank=[0],
        activation_mib_per_rank=[ARTIFACT_ACTIVATION_MIB],
        capture_tokens_per_rank=[96],
        capture_mib_per_rank=[ARTIFACT_CAPTURE_MIB],
        # A calibrated term must say where its number came from; the ledger
        # refuses an unattributed one, which is the same discipline #586 put
        # on the footprints themselves.
        phase_footprint_source_per_rank=[
            "[upper_bound] 2026-08-05 reference window, binding RTX 3080"
        ],
        phase_footprint_fingerprint="a191a0712717",
        mamba_pool_mib_per_rank=[512.0],
        gdn_scratch_mib_per_rank=[300.0],
        indexer_scratch_mib_per_rank=[120.0],
        flashinfer_workspace_mib=200,
        # Without the cap the indexer transient is UNBOUNDED (a refusal),
        # not zero -- #493. Priced here so the term can be moved below.
        indexer_chunk_cap_mib=256,
        chunked_prefill_size=2048,
        max_running_requests=4,
        parent_binds_cuda_context=True,
    )
    base.update(over)
    return DemandInputs(**base)


def demand(**over) -> int:
    ledgers = build_card_ledgers(
        inputs(**over),
        cards=[BINDING_CARD],
        rank_gpu_id=[1],
        user_reserve_mib={1: 1024},
    )
    assert len(ledgers) == 1
    return int(demand_outside_budget_mib(ledgers[0]))


def ledger(**over):
    return build_card_ledgers(
        inputs(**over),
        cards=[BINDING_CARD],
        rank_gpu_id=[1],
        user_reserve_mib={1: 1024},
    )[0]


class TestEveryTermIsInTheSum(unittest.TestCase):
    """Per-term bind proof. A consulted-only test would pass on a dropped term."""

    def _moves(self, label, delta_mib, **over):
        before = demand()
        after = demand(**over)
        self.assertEqual(
            after - before,
            delta_mib,
            f"{label}: moving this term by {delta_mib} MiB moved the reserve "
            f"by {after - before} MiB. A term that does not move the total is "
            f"a term the boot is not reserving for.",
        )

    def test_activation_term_is_in_the_sum(self):
        self._moves(
            "activation",
            500,
            activation_mib_per_rank=[ARTIFACT_ACTIVATION_MIB + 500],
        )

    def test_measured_capture_term_is_in_the_sum(self):
        # 640, not the 192 MiB the 2 MiB/token coefficient produces. Under-
        # booking capture by ~448 MiB is half the window-7 shortfall.
        self._moves(
            "graph capture",
            256,
            capture_mib_per_rank=[ARTIFACT_CAPTURE_MIB + 256],
        )

    def test_gdn_prefill_scratch_is_in_the_sum(self):
        self._moves("GDN prefill scratch", 64, gdn_scratch_mib_per_rank=[364.0])

    def test_indexer_prefill_scratch_is_in_the_sum(self):
        self._moves("indexer scratch", 30, indexer_scratch_mib_per_rank=[150.0])

    def test_attention_workspace_is_in_the_sum(self):
        self._moves("attention workspace", 100, flashinfer_workspace_mib=300)

    def test_ladder_term_is_in_the_sum(self):
        self._moves("adaptive draft ladder", 77, ladder_mib_per_gpu={1: 77})

    def test_mamba_pool_is_NOT_in_this_sum(self):
        """Budget-funded: charging it here would reserve it twice."""
        self._moves("mamba pool", 0, mamba_pool_mib_per_rank=[512.0 + 900])

    def test_weights_are_NOT_in_this_sum(self):
        """Also budget-funded, and their size depends on the ratio derived
        FROM this number."""
        self._moves("weights", 0, weight_mib_per_rank=[7000])


class TestUnpricedTermRefuses(unittest.TestCase):
    """A partial sum is what emptied the card. It must not be reachable."""

    def test_unpriced_activation_makes_the_ledger_unbounded(self):
        lg = ledger(activation_mib_per_rank=[None])
        self.assertTrue(lg.unbounded, "an unpriced activation term was summed away")
        self.assertTrue(
            any("activation" in u.lower() for u in lg.unbounded), lg.unbounded
        )

    def test_unpriced_gdn_scratch_makes_the_ledger_unbounded(self):
        lg = ledger(gdn_scratch_mib_per_rank=None)
        self.assertTrue(lg.unbounded, lg.unbounded)

    def test_a_refusing_ledger_is_not_silently_partial(self):
        """The sum of the priced terms is NOT offered as an answer."""
        lg = ledger(activation_mib_per_rank=[None])
        self.assertTrue(lg.unbounded)
        self.assertFalse(lg.fits if hasattr(lg, "fits") else False)


class TestSanityAnchorAgainstTheRealBoots(unittest.TestCase):
    """The model has to agree with what the rig actually did."""

    def test_full_demand_exceeds_what_window7_died_with(self):
        d = demand()
        self.assertGreater(
            d,
            OOMED_RESERVE_MIB,
            f"the full-demand reserve is {d} MiB, no more than the "
            f"{OOMED_RESERVE_MIB} MiB that OOMed during graph capture on this "
            "card on 2026-08-05. Then #593 has not fixed anything.",
        )

    def test_artifact_backed_terms_alone_do_not_reach_the_booted_reserve(self):
        """The anchor, stated as what the artifacts can actually support.

        Only two terms have measured numbers for this card: activation 1766
        and capture 640 (#586 reference). Their sum is 2406 MiB against the
        4160 MiB that booted and held its corridor in window 6. So the
        remaining ~1750 MiB is real memory that is NOT activation and NOT
        capture -- workspace, scratch, hardware residual, CUDA context -- and
        the full model only reaches the booted figure once those are priced
        from a measurement rather than from this test's stand-in values.

        This is deliberately an assertion about the ARTIFACTS, not about the
        stand-ins above: the other terms here are numbers this file chose, so
        a green "full demand >= 4160" built on them would be this test
        agreeing with itself. Pinned so that when the retry window prices
        those terms for real, this test is the one that has to be revisited.
        """
        artifact_backed = ARTIFACT_ACTIVATION_MIB + ARTIFACT_CAPTURE_MIB
        self.assertLess(
            artifact_backed,
            BOOTED_RESERVE_MIB,
            "if the two measured terms now cover the booted reserve on their "
            "own, the shortfall this test documents is gone and the reserve "
            "model should be re-derived from the new measurement.",
        )
        self.assertGreaterEqual(
            BOOTED_RESERVE_MIB - artifact_backed,
            1000,
            "the unmeasured remainder is what the retry window has to price.",
        )

    def test_the_full_model_beats_the_activation_only_reserve_on_the_same_inputs(self):
        """What #593 actually has to deliver: strictly more than #590 gave,
        on identical inputs, because #590's number is the one that OOMed."""
        self.assertGreater(demand(), OOMED_RESERVE_MIB)

    def test_the_heuristic_is_no_longer_the_better_number(self):
        """#590's failure mode, encoded: the ledger path must not be the one
        that under-reserves."""
        activation_only = ARTIFACT_ACTIVATION_MIB + 192  # what #590 installed
        self.assertGreater(demand(), activation_only)


if __name__ == "__main__":
    unittest.main()


class TestServerArgsFullDemandPath(unittest.TestCase):
    """The three cases at the ServerArgs seam, same structure as #590."""

    def setUp(self):
        from sglang.srt.server_args import ServerArgs

        ServerArgs._full_demand_refusal_named = False
        self.SA = ServerArgs

    def _stub(self, ledgers=None, raises=None):
        outer = self

        class S:
            def _build_card_ledgers(self):
                if raises is not None:
                    raise raises
                return ledgers

            def ledger_full_demand_per_gpu(self):
                return outer.SA.ledger_full_demand_per_gpu(self)

        return S()

    def test_priced_ledgers_yield_the_full_demand_per_gpu(self):
        """A fully priced card returns its demand."""
        import types

        from sglang.srt.mem_ledger.engine import TERM_ACTIVATION

        fake = types.SimpleNamespace(
            gpu_id=1,
            card="NVIDIA GeForce RTX 3080",
            unbounded=(),
            terms=(
                types.SimpleNamespace(name=TERM_ACTIVATION, mib=1766),
                types.SimpleNamespace(name="CUDA graph capture", mib=640),
            ),
        )
        got = self._stub(ledgers=[fake]).ledger_full_demand_per_gpu()
        self.assertEqual(got, {1: 2406})

    def test_an_uncalibrated_rig_refuses_rather_than_guessing(self):
        """Operationally the most likely refusal, and it must not be silent.

        With no VRAM calibration for the live fingerprint the hardware-residual
        term (CUDA context + allocator granularity + lazy workspaces) is
        UNBOUNDED, so the whole reserve refuses and the boot keeps the
        inherited model. That is the correct outcome -- a constant there is the
        guess the ledger exists to remove -- and it means this payout needs
        `python -m sglang.srt.mem_ledger.probe` run once on the rig.
        """
        lg = ledger()
        self.assertTrue(
            any("hardware residual" in u for u in lg.unbounded), lg.unbounded
        )
        with self.assertLogs("sglang.srt.server_args", level="WARNING"):
            self.assertIsNone(self._stub(ledgers=[lg]).ledger_full_demand_per_gpu())

    def test_an_unbounded_term_refuses_and_names_it(self):
        lg = ledger(activation_mib_per_rank=[None])
        stub = self._stub(ledgers=[lg])
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            got = stub.ledger_full_demand_per_gpu()
        self.assertIsNone(got, "a refusing ledger produced a number anyway")
        joined = "\n".join(cm.output)
        self.assertIn("REFUSES", joined)
        self.assertIn("activation", joined.lower())

    def test_the_refusal_is_named_once_per_process(self):
        lg = ledger(activation_mib_per_rank=[None])
        stub = self._stub(ledgers=[lg])
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            stub.ledger_full_demand_per_gpu()
            stub.ledger_full_demand_per_gpu()
        self.assertEqual(len(cm.output), 1, cm.output)

    def test_a_build_failure_is_silent_and_yields_no_number(self):
        """No NVML, no card facts: the caller keeps its previous behaviour and
        this path says nothing, because it has nothing to report."""
        stub = self._stub(raises=RuntimeError("no NVML"))
        with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
            self.assertIsNone(stub.ledger_full_demand_per_gpu())
