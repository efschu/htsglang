"""#598: the NCCL term gets a third state -- NOT_APPLICABLE.

#595 gave the NCCL communicator buffers a term that starts UNBOUNDED, on the
reasoning that this rig "always pays it": the window-8 boot line said custom
all-reduce was disabled and the boot "falls back to NCCL for TP collectives".
That inference was wrong for THIS transport. barlink takes ownership of the TP
group AFTER that decision, and a group barlink owns never constructs a PyNccl
communicator at all -- the boot logs "barlink is active for group 'tp:0':
skipping PyNccl communicator construction", and window 9's
NCCL_DEBUG=INFO/NCCL_DEBUG_SUBSYS=INIT,ALLOC run produced zero allocation
lines because there is nothing to allocate.

So the term needs to distinguish three things that a single number cannot:

    UNBOUNDED       a communicator is built and nobody measured it (refusal)
    priced          measured for a named communicator set (0 is a valid value)
    NOT_APPLICABLE  no communicator is built, so there is nothing to measure

The tests below bind both directions, keep the NCCL case exactly as it was,
and pin that the verdict comes from the SAME predicates the construction site
branches on rather than from a copy that can drift.
"""

import os
import sys
import types
import unittest
from unittest import mock

from sglang.srt.mem_ledger.engine import (
    TERM_NCCL_BUFFERS,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
    communicator_groups_from_server_args,
    demand_outside_budget_mib,
)
from sglang.srt.mem_ledger.nccl_transport import (
    CommunicatorGroup,
    classify_communicator_groups,
)
from sglang.srt.mem_ledger.terms import LedgerError, LedgerTerm, Provenance
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

CARD = CardFacts(
    gpu_id=1,
    uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
    name="NVIDIA GeForce RTX 3080",
    total_mib=20480,
)

#: A launch whose groups are all multi-rank. With SGLANG_BARLINK set, barlink
#: owns every one of them; with it unset, they all build NCCL.
TP_GROUPS = (
    CommunicatorGroup(name="world", world_size=2),
    CommunicatorGroup(name="tp", world_size=2),
)


def barlink_on():
    return mock.patch.dict(os.environ, {"SGLANG_BARLINK": "1"})


def barlink_off():
    return mock.patch.dict(os.environ, {"SGLANG_BARLINK": "0"})


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
        nccl_buffer_mib_per_gpu=None,
        nccl_signature="tp2.pp1",
    )
    base.update(over)
    return DemandInputs(**base)


#: A calibration profile stub, so a ledger built here can be COMPLETE. Without
#: it the hardware-residual term refuses and every "does the full-demand path
#: resolve?" assertion would pass or fail for the wrong reason.
def calibration_stub():
    residual = types.SimpleNamespace(
        total_mib=900,
        cuda_context_bytes=800 << 20,
        allocator_granularity_bytes=50 << 20,
        lazy_workspace_bytes=50 << 20,
    )
    return types.SimpleNamespace(
        fingerprint="a191a0712717",
        by_uuid=lambda: {CARD.uuid: residual},
    )


def ledger(*, calibration=None, **over):
    return build_card_ledgers(
        inputs(**over),
        cards=[CARD],
        rank_gpu_id=[1],
        user_reserve_mib={1: 1024},
        calibration=calibration,
    )[0]


def nccl_term(lg):
    return next((t for t in lg.terms if t.name == TERM_NCCL_BUFFERS), None)


def nccl_refusals(lg):
    return [u for u in lg.unbounded if TERM_NCCL_BUFFERS in u]


class TestBarlinkOwnedGroupsPriceZero(unittest.TestCase):
    """Direction one: barlink owns the groups -> the term resolves at 0."""

    def test_a_barlink_owned_launch_is_not_a_refusal(self):
        with barlink_on():
            lg = ledger(communicator_groups=TP_GROUPS)
        self.assertEqual(nccl_refusals(lg), [])

    def test_it_contributes_zero_mib(self):
        with barlink_on():
            with_groups = demand_outside_budget_mib(
                ledger(communicator_groups=TP_GROUPS)
            )
            measured_zero = demand_outside_budget_mib(
                ledger(communicator_groups=TP_GROUPS, nccl_buffer_mib_per_gpu={1: 0.0})
            )
        self.assertEqual(with_groups, measured_zero)

    def test_the_row_carries_the_not_applicable_marker(self):
        with barlink_on():
            term = nccl_term(ledger(communicator_groups=TP_GROUPS))
        self.assertIsNotNone(term)
        self.assertTrue(term.not_applicable)
        self.assertEqual(term.mib, 0)

    def test_the_derivation_names_the_skip_condition(self):
        """The justification has to be the CONSTRUCTION condition, not a
        restatement of the outcome."""
        with barlink_on():
            term = nccl_term(ledger(communicator_groups=TP_GROUPS))
        self.assertIn("skipping PyNccl", term.derivation)
        self.assertIn("barlink owns", term.derivation)
        self.assertIn("should_build_barlink", term.derivation)

    def test_a_single_rank_launch_also_resolves(self):
        """No barlink, no NCCL either: one rank builds no device
        communicator."""
        with barlink_off():
            lg = ledger(
                communicator_groups=(CommunicatorGroup(name="world", world_size=1),)
            )
        self.assertEqual(nccl_refusals(lg), [])
        self.assertTrue(nccl_term(lg).not_applicable)

    def test_a_stated_empty_group_set_resolves(self):
        with barlink_off():
            lg = ledger(communicator_groups=())
        self.assertTrue(nccl_term(lg).not_applicable)


class TestNcclOwnedGroupsKeepTodaysSemantics(unittest.TestCase):
    """Direction two: nothing about the NCCL case may move."""

    def test_nccl_groups_are_still_unbounded_until_ingest(self):
        with barlink_off():
            lg = ledger(communicator_groups=TP_GROUPS)
        self.assertTrue(nccl_refusals(lg), lg.unbounded)
        self.assertIsNone(nccl_term(lg))

    def test_the_refusal_still_says_how_to_measure_it(self):
        with barlink_off():
            msg = nccl_refusals(ledger(communicator_groups=TP_GROUPS))[0]
        self.assertIn("communicator init", msg)
        self.assertIn("NCCL_DEBUG", msg)

    def test_an_ingested_measurement_prices_it_as_before(self):
        with barlink_off():
            lg = ledger(
                communicator_groups=TP_GROUPS, nccl_buffer_mib_per_gpu={1: 128.0}
            )
        term = nccl_term(lg)
        self.assertEqual(term.mib, 128)
        self.assertFalse(term.not_applicable)
        self.assertIs(term.provenance, Provenance.CALIBRATED)
        self.assertEqual(term.fingerprint, "tp2.pp1")

    def test_an_unstated_group_set_is_byte_identical_to_before_598(self):
        """Every pre-#598 caller passes nothing here, and for those the term
        must keep exactly two states."""
        lg = ledger()
        self.assertTrue(nccl_refusals(lg), lg.unbounded)
        self.assertIsNone(nccl_term(lg))
        priced = ledger(nccl_buffer_mib_per_gpu={1: 128.0})
        self.assertEqual(nccl_term(priced).mib, 128)
        self.assertFalse(nccl_term(priced).not_applicable)

    def test_unstated_is_not_silently_resolved_by_barlink_being_on(self):
        """A caller that does not describe its groups gets the conservative
        answer even on a barlink boot: the ledger prices what it was told, not
        what it guessed."""
        with barlink_on():
            lg = ledger()
        self.assertTrue(nccl_refusals(lg), lg.unbounded)


class TestNotApplicableIsNotAMeasuredZero(unittest.TestCase):
    """The can-fail piece for the third state: collapsing NOT_APPLICABLE into
    ZERO must break something. Both charge 0 MiB, so the only thing that can
    tell them apart is the claim on the row."""

    def _na_and_zero(self):
        with barlink_on():
            na = nccl_term(ledger(communicator_groups=TP_GROUPS))
        with barlink_off():
            zero = nccl_term(
                ledger(communicator_groups=TP_GROUPS, nccl_buffer_mib_per_gpu={1: 0.0})
            )
        return na, zero

    def test_both_charge_zero_so_the_number_cannot_distinguish_them(self):
        na, zero = self._na_and_zero()
        self.assertEqual((na.mib, zero.mib), (0, 0))

    def test_the_marker_does_distinguish_them(self):
        na, zero = self._na_and_zero()
        self.assertTrue(na.not_applicable)
        self.assertFalse(zero.not_applicable)

    def test_their_provenance_differs_because_their_source_differs(self):
        """NOT_APPLICABLE follows from configuration (MODELED); a measured zero
        came off a card (CALIBRATED). They are invalidated by different
        events."""
        na, zero = self._na_and_zero()
        self.assertIs(na.provenance, Provenance.MODELED)
        self.assertIs(zero.provenance, Provenance.CALIBRATED)

    def test_the_rendered_ledger_shows_which_of_the_two_a_zero_row_is(self):
        with barlink_on():
            na_render = ledger(communicator_groups=TP_GROUPS).render()
        with barlink_off():
            zero_render = ledger(
                communicator_groups=TP_GROUPS, nccl_buffer_mib_per_gpu={1: 0.0}
            ).render()
        self.assertIn("modeled/n-a", na_render)
        self.assertNotIn("/n-a", zero_render)

    def test_a_not_applicable_term_may_not_charge_memory(self):
        with self.assertRaises(LedgerError):
            LedgerTerm(
                name="x",
                mib=1,
                provenance=Provenance.MODELED,
                derivation="d",
                inputs=("a",),
                not_applicable=True,
            )

    def test_the_marker_survives_serialization(self):
        with barlink_on():
            term = nccl_term(ledger(communicator_groups=TP_GROUPS))
        self.assertTrue(term.to_json()["not_applicable"])


class TestPerGroupNotPerBoot(unittest.TestCase):
    """A launch does not have one transport; it has one per group."""

    MIXED = (
        CommunicatorGroup(name="world", world_size=2, use_pynccl=True),
        CommunicatorGroup(name="cpu_only", world_size=2, use_pynccl=False),
    )

    def test_one_nccl_group_beside_a_non_nccl_group_still_refuses(self):
        with barlink_off():
            lg = ledger(communicator_groups=self.MIXED)
        self.assertTrue(nccl_refusals(lg), lg.unbounded)

    def test_the_refusal_names_the_group_that_builds_nccl(self):
        with barlink_off():
            msg = nccl_refusals(ledger(communicator_groups=self.MIXED))[0]
        self.assertIn("world", msg)
        self.assertNotIn("cpu_only (", msg)

    def test_the_priced_row_names_the_groups_it_covers(self):
        with barlink_off():
            term = nccl_term(
                ledger(
                    communicator_groups=self.MIXED,
                    nccl_buffer_mib_per_gpu={1: 64.0},
                )
            )
        self.assertIn("world", term.derivation)

    def test_a_group_set_where_none_builds_nccl_resolves(self):
        with barlink_off():
            lg = ledger(
                communicator_groups=(
                    CommunicatorGroup(name="cpu_only", world_size=2, use_pynccl=False),
                    CommunicatorGroup(name="solo", world_size=1),
                )
            )
        self.assertTrue(nccl_term(lg).not_applicable)

    def test_an_unresolvable_group_is_unbounded_and_named(self):
        """'Could not tell' must not read like 'there is none'."""
        with mock.patch.dict(
            sys.modules, {"sglang.srt.distributed.parallel_state": None}
        ):
            lg = ledger(communicator_groups=TP_GROUPS)
        msgs = nccl_refusals(lg)
        self.assertTrue(msgs, lg.unbounded)
        self.assertIn("could not be resolved", msgs[0])
        self.assertIn("tp", msgs[0])
        self.assertIsNone(nccl_term(lg))


class TestTheVerdictComesFromTheConstructionPredicate(unittest.TestCase):
    """The shared-predicate requirement. A private copy of
    ``use_pynccl and world_size > 1 and not barlink_active`` would pass every
    test above and then go stale the moment the real condition changes -- an
    under-charge, i.e. an OOM at the far end."""

    def test_it_calls_through_to_should_build_pynccl(self):
        with (
            barlink_on(),
            mock.patch(
                "sglang.srt.distributed.parallel_state.should_build_pynccl",
                return_value=True,
            ) as spy,
        ):
            verdicts = classify_communicator_groups(TP_GROUPS)
        self.assertTrue(spy.called)
        # The stub says "builds NCCL" even though barlink is on; a
        # re-implementation would ignore it and answer False.
        self.assertTrue(all(v.builds_nccl for v in verdicts))

    def test_it_calls_through_to_should_build_barlink(self):
        with (
            barlink_off(),
            mock.patch(
                "sglang.srt.distributed.parallel_state.should_build_barlink",
                return_value=True,
            ) as spy,
        ):
            verdicts = classify_communicator_groups(TP_GROUPS)
        self.assertTrue(spy.called)
        self.assertTrue(all(not v.builds_nccl for v in verdicts))

    def test_the_construction_site_uses_the_same_predicate(self):
        """The other half of "cannot drift": GroupCoordinator must branch on
        should_build_barlink too, not on an inline copy of its body."""
        import inspect

        from sglang.srt.distributed.parallel_state import GroupCoordinator

        src = inspect.getsource(GroupCoordinator.__init__)
        self.assertIn("should_build_barlink(self.world_size)", src)
        self.assertNotIn("envs.SGLANG_BARLINK.get() and self.world_size", src)

    def test_the_ledger_does_not_read_the_barlink_switch_itself(self):
        """Reading the switch here -- through sglang.environ or through
        os.environ -- would be the parallel check the import exists to
        prevent."""
        import inspect

        from sglang.srt.mem_ledger import engine, nccl_transport

        src = inspect.getsource(nccl_transport)
        for forbidden in ("os.environ", "getenv", "envs."):
            self.assertNotIn(forbidden, src, forbidden)
        self.assertNotIn("envs.SGLANG_BARLINK", inspect.getsource(engine))

    def test_the_predicates_are_the_ones_the_skip_log_line_belongs_to(self):
        from sglang.srt.distributed.parallel_state import (
            should_build_barlink,
            should_build_pynccl,
        )

        with barlink_on():
            self.assertTrue(should_build_barlink(2))
            self.assertFalse(should_build_pynccl(True, 2, should_build_barlink(2)))
        with barlink_off():
            self.assertFalse(should_build_barlink(2))
            self.assertTrue(should_build_pynccl(True, 2, should_build_barlink(2)))


class TestGroupsAreStatedFromTheSameServerArgs(unittest.TestCase):
    def test_the_world_group_spans_every_placed_rank(self):
        sa = types.SimpleNamespace(tp_size=2, pp_size=1, dcp_size=1)
        groups = {g.name: g for g in communicator_groups_from_server_args(sa, [0, 1])}
        self.assertEqual(groups["world"].world_size, 2)
        self.assertEqual(groups["tp"].world_size, 2)

    def test_a_dcp_group_appears_only_when_it_has_more_than_one_rank(self):
        sa = types.SimpleNamespace(tp_size=3, pp_size=1, dcp_size=3)
        names = [g.name for g in communicator_groups_from_server_args(sa, [0, 1, 2])]
        self.assertIn("dcp", names)
        sa = types.SimpleNamespace(tp_size=3, pp_size=1, dcp_size=1)
        names = [g.name for g in communicator_groups_from_server_args(sa, [0, 1, 2])]
        self.assertNotIn("dcp", names)

    def test_a_single_rank_launch_states_only_single_rank_groups(self):
        sa = types.SimpleNamespace(tp_size=1, pp_size=1, dcp_size=1)
        groups = communicator_groups_from_server_args(sa, [0])
        self.assertTrue(all(g.world_size == 1 for g in groups), groups)


class TestTheWindowNineOutcomeIsInverted(unittest.TestCase):
    """The point of the task. Window 9 ended with the full-demand reserve
    refusing on exactly three terms -- the NCCL post, one per card -- on a
    boot that allocates no NCCL buffers at all."""

    def setUp(self):
        from sglang.srt.server_args import ServerArgs

        ServerArgs._full_demand_refusal_named = False
        self.SA = ServerArgs

    def _stub(self, lg):
        stub = types.SimpleNamespace()
        stub._build_card_ledgers = lambda: [lg]
        stub._apply_gpu_mem_capacity_defaults = lambda gpu_mem: None
        stub._widen_decode_capture_to_session_ceiling = lambda cfg: None
        stub.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24, bs=None)
        )
        return stub

    def test_a_barlink_shaped_config_now_resolves(self):
        with barlink_on():
            lg = ledger(communicator_groups=TP_GROUPS, calibration=calibration_stub())
            self.assertEqual(lg.unbounded, ())
            got = self.SA.ledger_full_demand_per_gpu(self._stub(lg), 20480)
        self.assertIsNotNone(got)
        self.assertEqual(set(got), {1})
        self.assertEqual(got[1], demand_outside_budget_mib(lg))

    def test_the_same_config_on_an_nccl_transport_still_refuses(self):
        """The inversion is caused by the TRANSPORT and by nothing else: same
        ledger, same measurements, barlink off -> the window-9 outcome."""
        with barlink_off():
            lg = ledger(communicator_groups=TP_GROUPS, calibration=calibration_stub())
            self.assertTrue(nccl_refusals(lg), lg.unbounded)
            with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
                got = self.SA.ledger_full_demand_per_gpu(self._stub(lg), 20480)
        self.assertIsNone(got)
        self.assertIn("NCCL", "\n".join(cm.output))

    def test_the_resolved_reserve_does_not_charge_anything_for_nccl(self):
        with barlink_on():
            na = ledger(communicator_groups=TP_GROUPS, calibration=calibration_stub())
        with barlink_off():
            priced = ledger(
                communicator_groups=TP_GROUPS,
                calibration=calibration_stub(),
                nccl_buffer_mib_per_gpu={1: 128.0},
            )
        self.assertEqual(
            demand_outside_budget_mib(priced) - demand_outside_budget_mib(na), 128
        )


if __name__ == "__main__":
    unittest.main()
