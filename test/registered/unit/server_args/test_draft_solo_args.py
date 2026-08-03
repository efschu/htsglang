"""Unit tests for the draft-solo placement server args
(--speculative-draft-placement / --speculative-draft-gpu) — CPU only.

Split (the default) must be byte-identical to the pre-feature behavior:
the flags default to 'split'/None and every solo branch in the engine is
guarded on placement == 'solo'. Solo hard-rejects the out-of-scope modes
(topk > 1, rejection sampling, DP/PP/EP, non-EAGLE-family algorithms,
multi-node, PD disaggregation) with clear messages.
"""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def make_args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__, so
    the placement handler can be exercised in isolation (it normally runs
    after handle_speculative_decoding, i.e. on RESOLVED algorithm names —
    tests therefore pass 'EAGLE', never the 'NEXTN' alias)."""
    return ServerArgs(model_path="dummy", **kwargs)


def solo_args(**kwargs):
    defaults = dict(
        speculative_draft_placement="solo",
        speculative_algorithm="EAGLE",
        tp_size=2,
    )
    defaults.update(kwargs)
    return make_args(**defaults)


class TestCliParsing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def parse(self, *extra):
        return self.parser.parse_args(["--model-path", "m", *extra])

    def test_defaults_are_split(self):
        parsed = self.parse()
        self.assertEqual(parsed.speculative_draft_placement, "split")
        self.assertIsNone(parsed.speculative_draft_gpu)

    def test_solo_parses(self):
        parsed = self.parse(
            "--speculative-draft-placement", "solo", "--speculative-draft-gpu", "2"
        )
        self.assertEqual(parsed.speculative_draft_placement, "solo")
        self.assertEqual(parsed.speculative_draft_gpu, 2)

    def test_invalid_placement_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            self.parse("--speculative-draft-placement", "sideways")


class TestSplitDefault(CustomTestCase):
    def test_split_is_default_and_passes(self):
        args = make_args()
        self.assertEqual(args.speculative_draft_placement, "split")
        self.assertFalse(args.speculative_draft_solo_active())
        args._handle_speculative_draft_placement()  # no raise

    def test_split_with_gpu_rejected(self):
        args = make_args(speculative_draft_gpu=1)
        with self.assertRaisesRegex(ValueError, "speculative-draft-gpu"):
            args._handle_speculative_draft_placement()


class TestSoloValidation(CustomTestCase):
    def test_solo_eagle_tp2_ok(self):
        args = solo_args()
        args._handle_speculative_draft_placement()
        self.assertTrue(args.speculative_draft_solo_active())
        self.assertEqual(args.speculative_draft_solo_rank(), 0)

    def test_solo_topk_gt1_rejected(self):
        with self.assertRaisesRegex(ValueError, "topk"):
            solo_args(speculative_eagle_topk=4)._handle_speculative_draft_placement()

    def test_solo_topk1_ok(self):
        solo_args(speculative_eagle_topk=1)._handle_speculative_draft_placement()

    def test_solo_rejection_sampling_rejected(self):
        with self.assertRaisesRegex(ValueError, "rejection"):
            solo_args(
                speculative_use_rejection_sampling=True
            )._handle_speculative_draft_placement()

    def test_solo_dp_rejected(self):
        with self.assertRaisesRegex(ValueError, "data parallel"):
            solo_args(dp_size=2)._handle_speculative_draft_placement()

    def test_solo_dp_attention_rejected(self):
        with self.assertRaisesRegex(ValueError, "data parallel"):
            solo_args(enable_dp_attention=True)._handle_speculative_draft_placement()

    def test_solo_pp_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            solo_args(pp_size=2)._handle_speculative_draft_placement()

    def test_solo_ep_rejected(self):
        with self.assertRaisesRegex(ValueError, "expert parallelism"):
            solo_args(ep_size=2)._handle_speculative_draft_placement()

    def test_solo_multinode_with_draft_gpu_rejected(self):
        """The node-bound half: a per-node device index cannot name a rank
        across hosts, so this must still fail -- and say why."""
        with self.assertRaisesRegex(ValueError, "PER-NODE CUDA device index"):
            solo_args(
                nnodes=2, speculative_draft_gpu=0
            )._handle_speculative_draft_placement()

    def test_solo_multinode_without_draft_gpu_allowed(self):
        """The other direction, and the reason the gate was narrowed.

        Verified against the wiring, not the docstring (PLAN 61): solo's whole
        cross-rank contract is one get_tp_group().broadcast() -- the ordinary
        TP group, which a TP=5 group spanning two hosts demonstrably serves.
        Without --speculative-draft-gpu the solo host is rank 0 and no device
        lookup happens, so nothing here is node-bound.

        RELAXED, NOT PROVEN: no multi-node solo run has booted yet. This test
        pins the ARGUMENT contract only.
        """
        args = solo_args(nnodes=2)
        args._handle_speculative_draft_placement()  # must not raise
        self.assertEqual(args.speculative_draft_solo_rank(), 0)

    def test_solo_multinode_still_rejects_the_structural_modes(self):
        """Narrowing nnodes must not have loosened its neighbours."""
        for kwargs, needle in (
            (dict(dp_size=2), "data parallel"),
            (dict(pp_size=2), "pipeline"),
            (dict(ep_size=2), "expert parallelism"),
        ):
            with self.assertRaisesRegex(ValueError, needle):
                solo_args(nnodes=2, **kwargs)._handle_speculative_draft_placement()

    def test_solo_pd_disagg_rejected(self):
        with self.assertRaisesRegex(ValueError, "disaggregation"):
            solo_args(
                disaggregation_mode="decode"
            )._handle_speculative_draft_placement()

    def test_solo_tp1_rejected(self):
        with self.assertRaisesRegex(ValueError, "tp-size"):
            solo_args(tp_size=1)._handle_speculative_draft_placement()

    def test_solo_without_algorithm_rejected(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            make_args(
                speculative_draft_placement="solo", tp_size=2
            )._handle_speculative_draft_placement()

    def test_solo_frozen_kv_mtp_rejected(self):
        # FROZEN_KV_MTP is is_eagle() but reads the target KV in place — no
        # single rank holds the full target KV, so solo must reject it with
        # its own message (not the generic non-EAGLE one).
        with self.assertRaisesRegex(ValueError, "FROZEN_KV_MTP"):
            solo_args(
                speculative_algorithm="FROZEN_KV_MTP"
            )._handle_speculative_draft_placement()

    def test_solo_other_algorithms_rejected(self):
        # The DFLASH FAMILY (DFLASH + DSPARK) is supported: self-drafting block
        # models on a weight-TP=1 host. The remaining non-EAGLE algorithms stay
        # rejected.
        for algo in ("STANDALONE", "NGRAM"):
            with self.assertRaises(ValueError, msg=algo):
                solo_args(
                    speculative_algorithm=algo
                )._handle_speculative_draft_placement()

    def test_solo_dflash_accepted(self):
        # DFLASH goes solo; it must pass placement validation (block model is
        # non-adaptive, so no topk/rejection-sampling guards apply).
        solo_args(speculative_algorithm="DFLASH")._handle_speculative_draft_placement()

    def test_solo_dspark_accepted(self):
        # #470: DSPARK has the same shape as DFLASH (self-drafting block model,
        # post-all-reduce hidden-state input, token-id round output). Its one
        # delta -- the confidence head's variable block length -- rides the
        # same per-round broadcast as an extra integer per request; see
        # speculative/dspark_components/dspark_solo.py.
        solo_args(speculative_algorithm="DSPARK")._handle_speculative_draft_placement()

    def test_solo_dspark_rejects_adaptive(self):
        with self.assertRaisesRegex(ValueError, "adaptive"):
            solo_args(
                speculative_algorithm="DSPARK", speculative_adaptive=True
            )._handle_speculative_draft_placement()

    def test_solo_dspark_still_rejects_the_structural_modes(self):
        # Lifting the ALGORITHM refusal must not lift the placement invariants.
        for kwargs in (
            dict(dp_size=2),
            dict(pp_size=2),
            dict(ep_size=2),
            dict(disaggregation_mode="prefill"),
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                solo_args(
                    speculative_algorithm="DSPARK", **kwargs
                )._handle_speculative_draft_placement()

    def test_solo_dflash_rejects_adaptive(self):
        with self.assertRaisesRegex(ValueError, "adaptive"):
            solo_args(
                speculative_algorithm="DFLASH", speculative_adaptive=True
            )._handle_speculative_draft_placement()

    def test_solo_multi_layer_eagle_rejected(self):
        with self.assertRaisesRegex(ValueError, "multi-layer"):
            solo_args(
                enable_multi_layer_eagle=True
            )._handle_speculative_draft_placement()


class TestSoloRankResolution(CustomTestCase):
    def test_default_rank_is_zero(self):
        args = solo_args()
        self.assertEqual(args.speculative_draft_solo_rank(), 0)

    def test_gpu_maps_via_default_formula(self):
        args = solo_args(tp_size=3, speculative_draft_gpu=2)
        args._handle_speculative_draft_placement()
        self.assertEqual(args.speculative_draft_solo_rank(), 2)

    def test_gpu_maps_via_rank_gpu_id(self):
        # rank 0 lives on cuda:2 -> --speculative-draft-gpu 2 picks rank 0.
        args = solo_args(tp_size=3, rank_gpu_id=[2, 0, 1], speculative_draft_gpu=2)
        args._handle_speculative_draft_placement()
        self.assertEqual(args.speculative_draft_solo_rank(), 0)

    def test_gpu_with_base_gpu_id_offset(self):
        args = solo_args(tp_size=2, base_gpu_id=1, speculative_draft_gpu=2)
        args._handle_speculative_draft_placement()
        self.assertEqual(args.speculative_draft_solo_rank(), 1)

    def test_unmapped_gpu_rejected_with_mapping(self):
        args = solo_args(tp_size=2, speculative_draft_gpu=7)
        with self.assertRaisesRegex(ValueError, "rank 0 -> cuda:0"):
            args._handle_speculative_draft_placement()


class TestSoloDraftGpuResolution(CustomTestCase):
    """--speculative-draft-gpu is the one node-bound element of solo placement.

    Two properties are pinned: it resolves at STARTUP (not after the model has
    loaded), and its --rank-gpu-id co-location tie-break is announced rather
    than silent.
    """

    def test_unmatched_draft_gpu_fails_at_startup(self):
        """Fail fast, with the map, instead of at worker construction."""
        with self.assertRaisesRegex(ValueError, "does not match any TP rank"):
            solo_args(
                tp_size=2, speculative_draft_gpu=7
            )._handle_speculative_draft_placement()

    def test_draft_gpu_resolves_to_the_rank_on_that_device(self):
        args = solo_args(tp_size=3, rank_gpu_id=[0, 1, 2], speculative_draft_gpu=2)
        args._handle_speculative_draft_placement()
        self.assertEqual(args.speculative_draft_solo_rank(), 2)

    def test_colocation_tiebreak_is_deterministic_and_announced(self):
        """Several ranks on the named device: lowest wins, and it is logged.

        Deterministic and rank-uniform, so it cannot desync the group -- but
        the user named a DEVICE while two ranks live on it, so the choice is
        stated instead of being discovered later.
        """
        args = solo_args(tp_size=3, rank_gpu_id=[0, 1, 1], speculative_draft_gpu=1)
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            args._handle_speculative_draft_placement()
        joined = "\n".join(cm.output)
        self.assertIn("co-located ranks", joined)
        self.assertIn("[1, 2]", joined)
        self.assertEqual(args.speculative_draft_solo_rank(), 1)

    def test_no_warning_without_colocation(self):
        args = solo_args(tp_size=2, rank_gpu_id=[0, 1], speculative_draft_gpu=1)
        with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
            args._handle_speculative_draft_placement()


if __name__ == "__main__":
    unittest.main()
