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

3. THE #431 NOTICE. The scoped, named notice for barlink BAR1 x uneven
   weighted DCP x an fp8 checkpoint. It was a hard refusal until the
   2026-08-02 re-check measured that arm completing; it is a loud slow-boot
   WARNING now, with the refusal still reachable on request. Both directions
   are pinned here, including the can-fail ones: a test that goes red if the
   warning stops being emitted, and a test that goes red if the force-off
   stops raising.

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


class TestBar1Fp8UnevenDcpNotice(CustomTestCase):
    """The #431 notice predicate: warns by default, refuses on request.

    The direction of this predicate flipped once (#424 refusal -> #431
    re-check warning), so every test here states which direction it pins and
    why, and the two override variables are pinned by VALUE, not only by
    presence -- the whole compatibility argument is that
    `ALLOW=0` still refuses while `ALLOW=1` no longer decides anything.
    """

    BASE = dict(
        barlink_enabled=True,
        transport="bar1",
        uneven_weighted_dcp=True,
        quantization="fp8",
        force_refusal=False,
    )

    def _clean_env(self, **extra):
        """The two override variables removed unless a test sets them."""
        from unittest import mock

        env = dict(os.environ)
        env.pop(uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1, None)
        env.pop(uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1, None)
        env.update(extra)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_the_measured_arm_warns_and_names_all_three_axes(self):
        """Default direction: a notice, not a refusal, and a specific one.

        A warning that does not name the combination is a warning the
        operator cannot act on, so the three axes are asserted individually.
        """
        notice = uniformity.bar1_fp8_uneven_dcp_notice(**self.BASE)
        self.assertIsNotNone(notice)
        self.assertFalse(notice.refuse)
        self.assertIn("BAR1", notice.message)
        self.assertIn("uneven WEIGHTED DCP", notice.message)
        self.assertIn("fp8", notice.message)

    def test_the_warning_states_the_timing_and_where_the_evidence_is(self):
        """CAN-FAIL #1 for the text: the numbers an operator waits on.

        Goes red if the slow-boot text loses the per-rank window length, the
        'not a hang' statement, the warm-boot statement, or the pointer to
        the artifacts -- i.e. if the warning degrades back into a bare label.
        """
        notice = uniformity.bar1_fp8_uneven_dcp_notice(**self.BASE)
        message = notice.message
        self.assertIn("190 s", message)
        self.assertIn("not a hang", message)
        self.assertIn("Warm boots", message)
        self.assertIn(uniformity.RECHECK_EVIDENCE, message)
        self.assertIn("ANALYSE_431_fp8_bar1_dcp_deadlock.md", message)
        self.assertIn(uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1, message)

    def test_modelopt_fp8_is_the_same_arm(self):
        for q in ("modelopt_fp8", "fbgemm_fp8", "FP8"):
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "quantization": q}
            )
            self.assertIsNotNone(notice, q)
            self.assertFalse(notice.refuse, q)

    def test_the_other_arms_get_no_notice_at_all(self):
        """Every arm outside the combination stays completely silent.

        Unchanged from the refusal era: the scope was never BAR1 in general
        or uneven DCP in general, and a warning on the recommended INT8 +
        BAR1 operating point would be noise on a measured-good path.
        """
        quiet = [
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
        for delta in quiet:
            with self.subTest(**delta):
                self.assertIsNone(
                    uniformity.bar1_fp8_uneven_dcp_notice(**{**self.BASE, **delta})
                )

    def test_the_other_arms_stay_quiet_even_with_the_force_off_set(self):
        """The force-off widens nothing. It only changes what happens to the
        one combination that is already in scope."""
        with self._clean_env(**{uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1: "1"}):
            for delta in ({"quantization": "w8a8_int8"}, {"transport": "device"}):
                with self.subTest(**delta):
                    self.assertIsNone(
                        uniformity.bar1_fp8_uneven_dcp_notice(
                            **{**self.BASE, "force_refusal": None, **delta}
                        )
                    )

    def test_force_refusal_argument_restores_the_hard_error_text(self):
        notice = uniformity.bar1_fp8_uneven_dcp_notice(
            **{**self.BASE, "force_refusal": True}
        )
        self.assertTrue(notice.refuse)
        self.assertIn("is refused", notice.message)
        self.assertIn("force_refusal argument", notice.message)

    def test_the_explicit_force_off_env_refuses(self):
        with self._clean_env(**{uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1: "1"}):
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "force_refusal": None}
            )
        self.assertTrue(notice.refuse)
        self.assertIn(uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1, notice.source)

    def test_the_force_off_env_set_to_zero_states_the_default(self):
        with self._clean_env(**{uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1: "0"}):
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "force_refusal": None}
            )
        self.assertFalse(notice.refuse)

    def test_legacy_allow_zero_still_means_what_it_always_meant(self):
        """Backward compatibility, the direction that matters.

        A launch script pinning ALLOW=0 asked for 'do not admit this arm'.
        That must still be a hard refusal, or the flip would have silently
        turned an operator's opt-out into an opt-in.
        """
        with self._clean_env(**{uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1: "0"}):
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "force_refusal": None}
            )
        self.assertTrue(notice.refuse)
        self.assertIn(uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1, notice.source)

    def test_legacy_allow_one_is_honoured_and_says_it_is_now_redundant(self):
        """The other direction must not become a silent no-op.

        ALLOW=1 still means 'admit this arm' and is still obeyed -- it just
        no longer decides anything, and the notice says so rather than
        letting the operator believe the variable is load-bearing.
        """
        with self._clean_env(**{uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1: "1"}):
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "force_refusal": None}
            )
        self.assertFalse(notice.refuse)
        self.assertIn(uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1, notice.message)
        self.assertIn("no longer changes anything", notice.message)

    def test_the_explicit_name_wins_over_the_legacy_one(self):
        with self._clean_env(
            **{
                uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1: "0",
                uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1: "0",
            }
        ):
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "force_refusal": None}
            )
        self.assertFalse(notice.refuse)
        self.assertIn(uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1, notice.source)

    def test_neither_variable_set_is_the_warning(self):
        with self._clean_env():
            notice = uniformity.bar1_fp8_uneven_dcp_notice(
                **{**self.BASE, "force_refusal": None}
            )
        self.assertFalse(notice.refuse)
        self.assertEqual(notice.source, "default")


class TestModelRunnerNoticeIsWired(CustomTestCase):
    """The notice is REACHED, not just written.

    A predicate that no code path calls is a comment. This drives the real
    `ModelRunner._check_bar1_fp8_uneven_dcp_combination` on a `__new__`
    stand-in -- no CUDA, no weights, no process group -- so the wiring
    itself is executed at least once outside a GPU window.
    """

    RUNNER_LOGGER = "sglang.srt.model_executor.model_runner"

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

    def _env(self, **extra):
        from unittest import mock

        env = dict(os.environ)
        env.pop(uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1, None)
        env.pop(uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1, None)
        env["SGLANG_BARLINK"] = "1"
        env["SGLANG_BARLINK_TRANSPORT"] = "bar1"
        env.update(extra)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_can_fail_the_arm_warns_on_the_real_logger_and_does_not_raise(self):
        """CAN-FAIL #2: red if the warning stops being emitted.

        `assertLogs` fails the test when no record arrives on that logger, so
        deleting the `logger.warning(...)` call, downgrading it to INFO, or
        making the predicate return `None` for this arm all turn this red.
        The `assertNotIn` half is the other direction: booting must not be
        blocked any more.
        """
        with self._env():
            with self.assertLogs(self.RUNNER_LOGGER, level="WARNING") as caught:
                self._runner("fp8")._check_bar1_fp8_uneven_dcp_combination()
        joined = "\n".join(caught.output)
        self.assertIn("SLOW FIRST BOOT", joined)
        self.assertIn("190 s", joined)
        self.assertNotIn("is refused", joined)

    def test_can_fail_the_force_off_still_raises_before_any_weight_is_loaded(self):
        """CAN-FAIL #3: red if the force-off stops refusing.

        The operator escape hatch back to pre-#431-recheck behaviour. If the
        override is ever dropped or inverted, no ValueError is raised here
        and this goes red.
        """
        with self._env(**{uniformity.ENV_REFUSE_FP8_UNEVEN_DCP_BAR1: "1"}):
            with self.assertRaises(ValueError) as cm:
                self._runner("fp8")._check_bar1_fp8_uneven_dcp_combination()
        self.assertIn("is refused", str(cm.exception))

    def test_legacy_allow_zero_raises_exactly_as_it_used_to(self):
        with self._env(**{uniformity.ENV_ALLOW_FP8_UNEVEN_DCP_BAR1: "0"}):
            with self.assertRaises(ValueError):
                self._runner("fp8")._check_bar1_fp8_uneven_dcp_combination()

    def test_int8_arm_over_bar1_boots_without_a_word(self):
        with self._env():
            with self.assertNoLogs(self.RUNNER_LOGGER, level="WARNING"):
                self._runner("w8a8_int8")._check_bar1_fp8_uneven_dcp_combination()

    def test_even_dcp_is_untouched(self):
        with self._env():
            with self.assertNoLogs(self.RUNNER_LOGGER, level="WARNING"):
                self._runner("fp8", dcp_size=1)._check_bar1_fp8_uneven_dcp_combination()


if __name__ == "__main__":
    unittest.main()
