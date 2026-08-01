"""``--draft-kv-layout dcp`` is admitted only on the path it was written for
(#108), and every refusal fires at argument resolution.

WHAT THE FLAG DOES
------------------
``dcp`` token-shards the speculative DRAFT KV pool with the SAME weighted
owner rule, replicated kv-heads and cross-rank LSE merge the TARGET pool uses.
That is a reuse, not a new mechanism -- which is exactly why the admitted set
is narrow: the machinery covers ONE shape, a linear draft chain (MTP/NEXTN,
topk == 1) of ONE draft KV layer running on the uneven-weighted-DCP lane.

WHY THE REFUSALS ARE THE FEATURE
--------------------------------
Every rejected combination below fails SILENTLY if it is let through, not
loudly:

* topk > 1 writes a branching draft KV chain the owner rule does not describe
  (and #76 already measured tree verify under uneven DCP producing non-greedy,
  run-to-run-varying output at temperature 0).
* multi-layer EAGLE holds one draft ModelRunner per chain position; sharding N
  pools by one owner rule needs per-layer kernels that do not exist.
* draft-solo puts the draft on a single rank -- there is no peer group to
  shard across, so the owner rule would hand the sole owner a fraction of the
  rows it owns all of.
* kv-session-offload's spec-in-tick draft surgery writes RAW GLOBAL allocator
  slot ids straight into the draft k/v buffers, bypassing the owner rule; a
  compact pool reads those as other tokens' rows (#60's L3 zero-page class).
* off the weighted lane there is no token weight vector at all, so ``dcp`` has
  nothing to shard BY, and the flag would be an expensive no-op.

Nothing here touches a model, a GPU or a worker: a raise out of
``_reject_unsupported_draft_kv_dcp`` on a bare ``ServerArgs`` IS the proof
that the reject lands before any weight load (confirmed on the rig: the boot
dies with zero "Load weight begin" lines). The one condition that is NOT
decidable here -- how many attention layers the resolved draft CHECKPOINT
carries -- is gated in the draft ModelRunner and covered by
``TestMultiLayerDraftCheckpointGate`` below.

SLICE 2 REMOVED THE BLANKET REFUSAL
-----------------------------------
Slice 1 additionally turned away even the otherwise-admitted shape, because
the draft-EXTEND uneven-DCP metadata split did not exist. Slice 2 built it
(``flashinfer_backend.call_begin_forward``, the ``EAGLE_DRAFT_EXTEND``
branch), so that catch-all is gone and the covered shape passes. The
per-condition surface below is unchanged and still as narrow as it was.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.layers.dcp.owner import (
    draft_kv_layout_is_dcp,
    draft_pool_is_replicated,
    reject_multi_layer_draft_kv_dcp,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


WEIGHTED_ENV = {
    "SGLANG_UNEVEN_DCP": "1",
    "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
}


def make_args(**kwargs):
    """``model_path='dummy'`` short-circuits ``__post_init__`` (no resolution,
    no strict mutation guard), so the DCP handler can be driven in isolation
    with exactly the fields under test."""
    return ServerArgs(model_path="dummy", **kwargs)


def admitted_args(**overrides):
    """The one configuration the feature covers: weighted uneven DCP, EAGLE
    (the alias NEXTN resolves to), chain topk == 1."""
    kw = dict(
        draft_kv_layout="dcp",
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        rank_tp_ratio=[3, 1],
        dcp_size=2,
        tp_size=2,
    )
    kw.update(overrides)
    return make_args(**kw)


class TestDraftKvLayoutDefaultIsInert(unittest.TestCase):
    """The default path must not even reach the new code."""

    def test_default_is_replicated(self):
        self.assertEqual(make_args().draft_kv_layout, "replicated")

    def test_replicated_never_raises_whatever_else_is_set(self):
        # The hostile matrix for the 'dcp' gate, with the flag at its default:
        # not one of these may become a new boot failure.
        for algo in (None, "EAGLE", "EAGLE3", "NGRAM", "DFLASH", "STANDALONE"):
            for topk in (1, 4):
                for solo in (False, True):
                    with self.subTest(algo=algo, topk=topk, solo=solo):
                        args = make_args(
                            speculative_algorithm=algo,
                            speculative_eagle_topk=topk,
                            enable_multi_layer_eagle=solo,
                            enable_kv_session_offload=solo,
                        )
                        # the #108 gate specifically; the rest of
                        # _handle_dcp_validation has its own tests
                        args._reject_unsupported_draft_kv_dcp()

    def test_the_predicates_read_the_default_as_replicated(self):
        args = make_args()
        self.assertFalse(draft_kv_layout_is_dcp(args))
        self.assertTrue(draft_pool_is_replicated(True, args))
        self.assertFalse(draft_pool_is_replicated(False, args))

    def test_predicates_tolerate_a_server_args_stand_in(self):
        """CUDA-graph runners and test doubles hand the backend partial
        server-args objects (sometimes None). An absent field must read as the
        unchanged default, not explode."""
        for stub in (None, object()):
            with self.subTest(stub=type(stub).__name__):
                self.assertFalse(draft_kv_layout_is_dcp(stub))
                self.assertTrue(draft_pool_is_replicated(True, stub))


class TestDraftKvLayoutSliceTwoRemovedTheBlanketRefusal(unittest.TestCase):
    """Slice 1 refused the whole layout at boot because the draft-EXTEND
    uneven-DCP metadata split did not exist. Slice 2 built it, so that blanket
    refusal is gone and the covered shape must now pass.

    This test is the inverse of the one it replaces: it pins that the
    catch-all is NOT reinstated, while the per-condition surface below stays
    exactly as narrow as it was.
    """

    def test_no_blanket_not_usable_refusal_remains(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            # would have raised in slice 1
            admitted_args()._reject_unsupported_draft_kv_dcp()

    def test_the_specific_diagnostics_still_fire(self):
        """Removing the catch-all must not have removed the real guards."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            with self.assertRaises(ValueError) as cm:
                admitted_args(
                    speculative_eagle_topk=4
                )._reject_unsupported_draft_kv_dcp()
            self.assertIn("--speculative-eagle-topk", str(cm.exception))


class TestDraftKvLayoutDcpAdmitted(unittest.TestCase):
    def test_the_covered_shape_passes(self):
        """The one configuration the feature covers, admitted since slice 2."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            admitted_args()._reject_unsupported_draft_kv_dcp()

    def test_the_predicates_flip_for_an_opted_in_draft(self):
        args = admitted_args()
        self.assertTrue(draft_kv_layout_is_dcp(args))
        # THE point of the predicate: an opted-in draft worker is no longer
        # "replicated", so both the pool sizing and the attention backend take
        # the target's path for it.
        self.assertFalse(draft_pool_is_replicated(True, args))
        self.assertFalse(draft_pool_is_replicated(False, args))


class TestDraftKvLayoutDcpRefusals(unittest.TestCase):
    def _assert_rejects(self, args, *needles):
        with self.assertRaises(ValueError) as cm:
            args._reject_unsupported_draft_kv_dcp()
        msg = str(cm.exception)
        for needle in needles:
            self.assertIn(needle, msg)
        # every message must name the way out, not just the problem
        self.assertIn("replicated", msg)
        return msg

    def test_off_the_weighted_lane_is_rejected(self):
        # no env pair at all
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_UNEVEN_DCP", None)
            os.environ.pop("SGLANG_UNEVEN_DCP_WEIGHTED", None)
            self._assert_rejects(admitted_args(), "SGLANG_UNEVEN_DCP")

    def test_uniform_ratio_is_rejected(self):
        """A uniform vector is not a weighted plan: there is no non-trivial
        share to shard by."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(admitted_args(rank_tp_ratio=[1, 1]), "rank_tp_ratio")

    def test_no_ratio_is_rejected(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(admitted_args(rank_tp_ratio=None), "rank_tp_ratio")

    def test_dcp_must_span_the_tp_group(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(
                admitted_args(dcp_size=1, tp_size=2), "dcp_size == tp_size"
            )
            self._assert_rejects(
                admitted_args(dcp_size=2, tp_size=4), "dcp_size == tp_size"
            )

    def test_no_speculation_is_rejected(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(
                admitted_args(speculative_algorithm=None), "speculative"
            )

    def test_tree_topk_is_rejected(self):
        """THE correctness falsifier: topk > 1 is the shape #76 measured
        producing non-greedy, run-to-run-varying output under uneven DCP."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            for topk in (2, 4, 8):
                with self.subTest(topk=topk):
                    msg = self._assert_rejects(
                        admitted_args(speculative_eagle_topk=topk),
                        "--speculative-eagle-topk",
                    )
                    self.assertIn(str(topk), msg)

    def test_non_chain_algorithms_are_rejected_by_name(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            for algo in ("EAGLE3", "STANDALONE", "DFLASH", "DSPARK"):
                with self.subTest(algo=algo):
                    self._assert_rejects(
                        admitted_args(speculative_algorithm=algo), algo
                    )

    def test_multi_layer_eagle_is_rejected(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(
                admitted_args(enable_multi_layer_eagle=True),
                "--enable-multi-layer-eagle",
            )

    def test_cross_algorithm_serving_is_rejected(self):
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(
                admitted_args(speculative_cross_algorithm=True),
                "cross-algorithm",
            )

    def test_draft_solo_placement_is_rejected(self):
        """One rank holds the whole draft there, so there is no DCP peer group
        to shard across -- the owner rule would hand the sole owner ratio_r/S
        of rows it owns all of."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            args = admitted_args()
            with patch.object(
                type(args), "speculative_draft_solo_active", lambda self: True
            ):
                self._assert_rejects(args, "draft-solo")

    def test_kv_session_offload_is_rejected(self):
        """The unbounded-writer interaction: spec_in_tick_draft_pre writes raw
        global slot ids into the draft buffers, bypassing the owner rule."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            self._assert_rejects(
                admitted_args(enable_kv_session_offload=True),
                "--enable-kv-session-offload",
            )


class TestGateRunsAfterArgResolution(unittest.TestCase):
    """WHERE the gate is called is itself load-bearing, and got this wrong once.

    The gate's inputs are all products of earlier resolution passes:

      --speculative-algorithm  NEXTN -> EAGLE (handle_speculative_decoding)
      --speculative-eagle-topk defaulted, not necessarily user-supplied
      --rank-tp-ratio          'auto-performance' -> a concrete vector
      dcp_size                 auto-set to tp_size under SGLANG_UNEVEN_DCP

    Placed in ``_handle_dcp_validation`` (where the sibling DCP gates live) it
    reads every one of them RAW. Measured on the rig: a correct TP=3
    ``--speculative-algorithm NEXTN --rank-tp-ratio auto-performance
    --draft-kv-layout dcp`` boot was refused with
    ``rank_tp_ratio=auto-performance, dcp_size=1`` -- a false rejection of the
    exact configuration the feature exists for, and one that no unit test
    driving ``_reject_unsupported_draft_kv_dcp`` directly can see.

    So the placement is pinned here, from both ends.
    """

    def test_the_gate_is_not_in_the_pre_resolution_handler(self):
        import inspect

        src = inspect.getsource(ServerArgs._handle_dcp_validation)
        self.assertNotIn(
            "self._reject_unsupported_draft_kv_dcp()",
            src,
            "the #108 gate reads resolved values; _handle_dcp_validation runs "
            "before --rank-tp-ratio, dcp_size and the algorithm alias are "
            "resolved",
        )

    def test_the_gate_runs_after_the_speculative_resolution(self):
        import inspect

        src = inspect.getsource(ServerArgs.__post_init__)
        gate = src.index("self._reject_unsupported_draft_kv_dcp()")
        spec = src.index("self._handle_speculative_draft_placement()")
        uneven = src.index("self._handle_uneven_tp()")
        self.assertLess(spec, gate, "algorithm alias / topk must be resolved first")
        self.assertLess(
            uneven, gate, "--rank-tp-ratio and dcp_size must be resolved first"
        )

    def test_an_unresolved_ratio_string_would_have_been_refused(self):
        """The exact false rejection, as a falsifier: if anyone moves the gate
        back before ``_handle_uneven_tp``, this is the shape it sees."""
        with patch.dict(os.environ, WEIGHTED_ENV):
            with self.assertRaises(ValueError) as cm:
                admitted_args(
                    rank_tp_ratio="auto-performance", dcp_size=1
                )._reject_unsupported_draft_kv_dcp()
            # the LANE message, not the not-implemented catch-all
            self.assertIn("rank_tp_ratio=auto-performance", str(cm.exception))


class TestMultiLayerDraftCheckpointGate(unittest.TestCase):
    """Tier 2: the draft CHECKPOINT's layer count, which ServerArgs cannot see.

    A checkpoint resolving through the NEXTN -> EAGLE alias can still carry
    several attention layers. With L > 1 the draft's own decode reads
    positions an earlier layer of the same forward wrote, and the cross-rank
    LSE merge would have to run per layer inside the draft step.
    """

    def test_one_layer_is_admitted(self):
        for layers in (0, 1):
            with self.subTest(layers=layers):
                reject_multi_layer_draft_kv_dcp(layers, "Qwen3_5ForCausalLMMTP")

    def test_multi_layer_is_rejected_and_names_the_numbers(self):
        for layers in (2, 4, 8):
            with self.subTest(layers=layers):
                with self.assertRaises(ValueError) as cm:
                    reject_multi_layer_draft_kv_dcp(layers, "SomeEagleDraft")
                msg = str(cm.exception)
                self.assertIn(str(layers), msg)
                self.assertIn("SomeEagleDraft", msg)
                self.assertIn("replicated", msg)


if __name__ == "__main__":
    unittest.main()
