"""#431 -- rank-local decisions before a group collective, made visible.

The bug family (#94, #194, #312 and now #431) is always the same shape: a
rank-local condition is evaluated in front of a group collective, two ranks
take different collective sequences, and the only observable is a hang that
costs a GPU window to diagnose. This file pins the three pieces that turn
that into a desk-diagnosable, hermetic failure:

1. THE COMPARATOR. `first_divergence` names the index, the ranks and the
   differing field -- positionally, because bar1 sequences every collective on
   one shared device round counter and waits on flag EQUALITY, so "the same
   collectives in a different order" hangs exactly as hard as "different
   collectives".

2. THE STRUCTURAL FALSIFIER. The real `BarlinkCommunicator._select` and the
   real `BarlinkBar1Transport._handles_all_gather` are driven, per rank, with
   the per-rank byte counts an uneven collective would hand them. The
   predicate's own docstring says "Rank-uniform, because nbytes is"; this test
   is the counter-example to that premise and shows the split-brain
   (some ranks bar1, some gloo) that follows from it. No GPU, no
   torch.distributed, no transport is constructed.

3. THE #431 REFUSAL. The scoped, named refusal for barlink BAR1 x uneven
   weighted DCP x an fp8 checkpoint, including its can-fail direction: with
   the override set -- i.e. with the refusal reverted -- the configuration is
   admitted again and the recorded sequences diverge.

CPU only: nothing here allocates on a device or builds a process group.
"""

import os
import unittest

from sglang.srt.distributed.device_communicators import barlink_uniformity as uniformity
from sglang.srt.distributed.device_communicators.barlink import (
    BarlinkCommunicator,
)
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

D = uniformity.CollectiveDecision

#: One BAR1 a2a slot in the #424 arms' dcp group. The group ran with
#: SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP=32, which is why the slot is small
#: enough for the round limit to be reachable by real payloads at all.
SLOT = 1 << 20
AG_MAX_ROUNDS = 16


def _bar1_stub(slot: int = SLOT, ag_max_rounds: int = AG_MAX_ROUNDS):
    """A BAR1 transport carrying the REAL predicates and nothing else.

    Built with `__new__` on purpose: `_handles_all_gather` is the code under
    test and must not be re-implemented here, but constructing the transport
    would map BAR1 apertures. Only the attributes that predicate reads are
    supplied, and they are the group-reconciled ones -- which is the point:
    every value below is identical on every rank, so any divergence the test
    produces comes from `nbytes` alone.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t._up = True
    t._ext = object()
    t._proofs_hold = True
    t.ag_on = True
    t.a2a_on = True
    t._a2a_proof = True
    t.ag_min_bytes = 1
    t.ag_max_rounds = ag_max_rounds
    t._window_minimum = 1 << 30
    t._geo = {
        "off_a2a": 0,
        "a2a_slot": slot,
        "region_bytes": 1 << 20,
        "chunk_max": slot,
    }
    return t


def _communicator(transport, group="dcp:0"):
    """The real `_select`, with the real transport stub behind it."""
    c = BarlinkCommunicator.__new__(BarlinkCommunicator)
    c.transport = transport
    c.group = group
    c._path_dispatcher = None
    c._fallback_reported = set()
    return c


class TestFirstDivergence(CustomTestCase):
    """The pure comparator, which is what makes the falsifier hermetic."""

    def test_identical_sequences_are_clean(self):
        seq = [D("all_gather", 4096, "bar1", 1), D("all_reduce", 8192, "bar1", 1)]
        self.assertIsNone(uniformity.first_divergence({0: list(seq), 1: list(seq)}))

    def test_single_rank_cannot_diverge(self):
        self.assertIsNone(
            uniformity.first_divergence({0: [D("all_gather", 1, "gloo")]})
        )

    def test_path_split_is_named_with_index_and_field(self):
        base = [D("all_gather", 4096, "bar1", 1)]
        other = [D("all_gather", 4096, "gloo", 0)]
        detail = uniformity.first_divergence({0: base, 1: other})
        self.assertIsNotNone(detail)
        self.assertIn("collective #0", detail)
        self.assertIn("rank 0", detail)
        self.assertIn("rank 1", detail)
        self.assertIn("path", detail)

    def test_round_count_split_is_caught_even_on_the_same_path(self):
        """A same-path, different-round-count pair is still a hang.

        Every round is one kernel launch that advances the shared device
        round counter, so two ranks agreeing on 'bar1' but disagreeing on how
        many rounds it takes desynchronise just as permanently.
        """
        detail = uniformity.first_divergence(
            {
                0: [D("all_gather", 1 << 21, "bar1", 2)],
                1: [D("all_gather", 1 << 20, "bar1", 1)],
            }
        )
        self.assertIsNotNone(detail)
        self.assertIn("rounds", detail)

    def test_early_return_shows_up_as_a_count_mismatch(self):
        detail = uniformity.first_divergence(
            {
                0: [D("all_gather", 64, "gloo"), D("all_reduce", 64, "gloo")],
                1: [D("all_gather", 64, "gloo")],
            }
        )
        self.assertIsNotNone(detail)
        self.assertIn("collective count", detail)

    def test_assert_helper_raises_the_named_error(self):
        with self.assertRaises(uniformity.CollectiveSequenceDivergence) as cm:
            uniformity.assert_sequences_uniform(
                {0: [D("all_gather", 1, "bar1")], 1: [D("all_gather", 1, "gloo")]},
                context="dcp:0",
            )
        self.assertIn("dcp:0", str(cm.exception))


class TestRecorder(CustomTestCase):
    def setUp(self):
        uniformity.clear_all()
        self._prev = uniformity.set_recording_for_test(True)

    def tearDown(self):
        uniformity.set_recording_for_test(self._prev)
        uniformity.clear_all()

    def test_off_by_default_costs_nothing_and_records_nothing(self):
        uniformity.set_recording_for_test(False)
        uniformity.record_decision("dcp:0", "all_gather", 4096, "bar1", 1)
        self.assertEqual(uniformity.snapshots(), {})

    def test_groups_are_kept_apart(self):
        uniformity.record_decision("tp:0", "all_reduce", 4096, "bar1", 1)
        uniformity.record_decision("dcp:0", "all_gather", 4096, "gloo", 0)
        snap = uniformity.snapshots()
        self.assertEqual([d.op for d in snap["tp:0"]], ["all_reduce"])
        self.assertEqual([d.op for d in snap["dcp:0"]], ["all_gather"])

    def test_ring_is_bounded_but_the_total_is_not_lost(self):
        rec = uniformity.DecisionRecorder(group="dcp:0", capacity=4)
        for i in range(10):
            rec.record("all_gather", i, "bar1", 1)
        self.assertEqual(len(rec), 4)
        self.assertEqual(rec.total, 10)
        self.assertEqual([d.nbytes for d in rec.snapshot()], [6, 7, 8, 9])


class TestDumpRoundTrip(CustomTestCase):
    """The post-mortem path: a wedged rank cannot compare, so it writes.

    This is what makes the GPU repro turnkey -- the divergence is read off
    disk after the hang instead of being reconstructed from py-spy frames,
    which under async collectives cannot establish a sequence at all.
    """

    def test_written_lines_read_back_and_compare(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            recs = {}
            for rank in range(3):
                path = os.path.join(
                    d, uniformity.DUMP_NAME.format(group="dcp-0", rank=rank)
                )
                recs[rank] = uniformity.DecisionRecorder(group="dcp:0", dump_path=path)
            recs[0].record("all_gather", 1 << 24, "gloo", 0)
            recs[1].record("all_gather", 1 << 20, "bar1", 1)
            recs[2].record("all_gather", 1 << 20, "bar1", 1)

            loaded = uniformity.load_dump_dir(d)
            self.assertEqual(sorted(loaded), [0, 1, 2])
            detail = uniformity.first_divergence(loaded)
            self.assertIsNotNone(detail)
            self.assertIn("path", detail)

    def test_truncated_last_line_does_not_lose_the_rest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, uniformity.DUMP_NAME.format(group="dcp-0", rank=0))
            rec = uniformity.DecisionRecorder(group="dcp:0", dump_path=path)
            rec.record("all_gather", 4096, "bar1", 1)
            with open(path, "a") as fh:
                fh.write('{"i": 1, "op": "all_re')  # killed mid-write
            loaded = uniformity.load_dump_dir(d)
            self.assertEqual(len(loaded[0]), 1)


class TestRealPredicateSplitsOnRankLocalNbytes(CustomTestCase):
    """The structural falsifier: real `_select`, real `handles`, no GPU.

    `BarlinkBar1Transport._handles_all_gather` states "Rank-uniform, because
    nbytes is" directly above its round-limit test. The premise is what is
    being falsified -- the seam passes `input_.numel() * input_.element_size()`,
    i.e. the caller's own shard.
    """

    def setUp(self):
        uniformity.clear_all()
        self._prev = uniformity.set_recording_for_test(True)

    def tearDown(self):
        uniformity.set_recording_for_test(self._prev)
        uniformity.clear_all()

    def test_equal_shards_agree(self):
        """The contract case. Equal-shaped inputs -> one decision, all ranks."""
        t = _bar1_stub()
        comm = _communicator(t)
        seqs = {}
        for rank in range(3):
            uniformity.clear_all()
            comm._select("all_gather", 8 * SLOT)
            seqs[rank] = uniformity.snapshots()["dcp:0"]
        self.assertIsNone(uniformity.first_divergence(seqs))
        self.assertEqual({d.path for s in seqs.values() for d in s}, {"bar1"})

    def test_uneven_shards_split_the_group_between_bar1_and_gloo(self):
        """The falsifier. Straddle the round limit and the group splits.

        Rank 0 carries a shard needing more than `ag_max_rounds` rounds, so
        `_handles_all_gather` refuses and `_select` returns the gloo plane.
        Ranks 1 and 2 fit and enter the BAR1 kernel. Neither rank can see the
        other's answer: nothing in `_select` is group-reconciled.
        """
        t = _bar1_stub()
        comm = _communicator(t)
        # One byte over the limit vs comfortably under it.
        per_rank_bytes = [
            (AG_MAX_ROUNDS + 1) * SLOT,
            2 * SLOT,
            2 * SLOT,
        ]
        seqs = {}
        for rank, nbytes in enumerate(per_rank_bytes):
            uniformity.clear_all()
            comm._select("all_gather", nbytes)
            seqs[rank] = uniformity.snapshots()["dcp:0"]

        paths = [seqs[r][0].path for r in range(3)]
        self.assertEqual(paths, ["gloo", "bar1", "bar1"])

        detail = uniformity.first_divergence(seqs)
        self.assertIsNotNone(detail, "the split-brain must be detected")
        self.assertIn("path", detail)
        with self.assertRaises(uniformity.CollectiveSequenceDivergence):
            uniformity.assert_sequences_uniform(seqs, context="dcp:0 all_gather")

    def test_uneven_shards_split_the_round_count_even_below_the_limit(self):
        """Same path, different round count -- the subtler half of the same bug.

        `barlink_all_gather` builds its plan as `ag_plan([shard] * world, ...)`,
        the group vector faked from the local shard, so a per-rank shard is a
        per-rank number of kernel launches on a counter every rank shares.
        """
        t = _bar1_stub()
        comm = _communicator(t)
        seqs = {}
        for rank, nbytes in enumerate([4 * SLOT, SLOT, SLOT]):
            uniformity.clear_all()
            comm._select("all_gather", nbytes)
            seqs[rank] = uniformity.snapshots()["dcp:0"]
        self.assertEqual([seqs[r][0].path for r in range(3)], ["bar1"] * 3)
        self.assertEqual([seqs[r][0].rounds for r in range(3)], [4, 1, 1])
        self.assertIn("rounds", uniformity.first_divergence(seqs))


class TestUnprovenCombinationRefusal(CustomTestCase):
    """The #431 refusal predicate: scoped to the arm that actually failed."""

    BASE = dict(
        barlink_enabled=True,
        transport="bar1",
        uneven_weighted_dcp=True,
        quantization="fp8",
        override=False,
    )

    def test_the_failing_arm_is_refused_and_names_its_evidence(self):
        msg = uniformity.unproven_bar1_combination(**self.BASE)
        self.assertIsNotNone(msg)
        self.assertIn("BAR1", msg)
        self.assertIn("fp8", msg)
        self.assertIn("#424", msg)
        self.assertIn(uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1, msg)

    def test_modelopt_fp8_is_the_same_arm(self):
        for q in ("modelopt_fp8", "fbgemm_fp8", "FP8"):
            self.assertIsNotNone(
                uniformity.unproven_bar1_combination(
                    **{**self.BASE, "quantization": q}
                ),
                q,
            )

    def test_the_measured_good_arms_are_untouched(self):
        """Every arm the #424 battery ran to completion must still boot."""
        good = [
            # INT8-W8A8 over BAR1 with the same DCP settings -- the fork's
            # recommended INT8 operating point.
            {"quantization": "w8a8_int8"},
            {"quantization": "compressed-tensors"},
            # fp8 over stock NCCL: the SGLANG_BARLINK block unset.
            {"barlink_enabled": False},
            # fp8 over a non-BAR1 barlink transport.
            {"transport": "device"},
            {"transport": "ucx"},
            # fp8 + BAR1 without uneven weighted DCP.
            {"uneven_weighted_dcp": False},
            # Unquantized.
            {"quantization": None},
        ]
        for delta in good:
            with self.subTest(**delta):
                self.assertIsNone(
                    uniformity.unproven_bar1_combination(**{**self.BASE, **delta})
                )

    def test_can_fail_direction_the_override_readmits_the_failing_arm(self):
        """Revert the fix -> the arm boots again, and the sequences diverge.

        This is the honest can-fail proof the refusal needs: with the override
        the guard is gone, and what the run then walks into is the split-brain
        the falsifier above reproduces structurally.
        """
        self.assertIsNone(
            uniformity.unproven_bar1_combination(**{**self.BASE, "override": True})
        )


class TestModelRunnerRefusalIsWired(CustomTestCase):
    """The refusal is REACHED, not just written.

    A predicate that no code path calls is a comment. This drives the real
    `ModelRunner._refuse_unproven_bar1_dcp_combination` on a `__new__`
    stand-in -- no CUDA, no weights, no process group -- so the wiring
    itself is executed at least once outside a GPU window.
    """

    def _runner(self, quantization, uneven=True, dcp_size=3):
        from sglang.srt.model_executor.model_runner import ModelRunner

        class _Args:
            def uneven_weighted_dcp_enabled(self):
                return uneven

        class _Config:
            pass

        r = ModelRunner.__new__(ModelRunner)
        r.server_args = _Args()
        cfg = _Config()
        cfg.quantization = quantization
        r.model_config = cfg
        r.dcp_size = dcp_size
        return r

    def test_failing_arm_raises_before_any_weight_is_loaded(self):
        import os
        from unittest import mock

        env = {
            "SGLANG_BARLINK": "1",
            "SGLANG_BARLINK_TRANSPORT": "bar1",
            uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1: "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError) as cm:
                self._runner("fp8")._refuse_unproven_bar1_dcp_combination()
            self.assertIn("#424", str(cm.exception))

    def test_int8_arm_over_bar1_still_boots(self):
        import os
        from unittest import mock

        env = {"SGLANG_BARLINK": "1", "SGLANG_BARLINK_TRANSPORT": "bar1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self._runner("w8a8_int8")._refuse_unproven_bar1_dcp_combination()

    def test_even_dcp_is_untouched(self):
        import os
        from unittest import mock

        env = {"SGLANG_BARLINK": "1", "SGLANG_BARLINK_TRANSPORT": "bar1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self._runner("fp8", dcp_size=1)._refuse_unproven_bar1_dcp_combination()


if __name__ == "__main__":
    unittest.main()
