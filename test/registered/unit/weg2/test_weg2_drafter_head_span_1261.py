# SPDX-License-Identifier: Apache-2.0
"""Train 2 reconcile (#1261 x #1259 b): the ring is sized for the image that
EXISTS, so a head that is never built is never parked for.

THE TWO HALVES THAT MET ON TRAIN 2.

* Train fix 5 (#1261) derives each PP stage's weight bytes from the CHECKPOINT
  and puts a DRAFTER term on the last stage -- ``mtp`` + a second embedding +
  a second ``lm_head`` -- because a NEXTN head is loaded as its own model
  runner. That term is what boot weg2tr2 died of not having: rg6's group P
  carried no MTP head, tr2's carries one, and PP2 asked for 2426 MiB with 204
  free.
* The mtp-head slice (#1259 b) makes the draft-KV PRODUCER build the head
  inside ``qwen3_5_mtp.lm_head_from_target()``, so the head's own
  ``[vocab, hidden]`` table is NEVER ALLOCATED -- it shares the co-located
  target's, which is resident on that same last stage.

Composed unreconciled, fix 5 parks 2425 MiB of ballast per boot: the weg2tr2
defect with the sign flipped, and one that presents as a REFUSAL of a form that
fits rather than as a death, which is the harder failure to attribute.

RED-FIRST, MEASURED against the merge tip 623ee1b041 through the real solver on
this rig's own rg6 table -- not asserted:

    nvml2 (PP2, the drafter's stage)   span1   H       max_tag_P  need_d2p  slack
    unreconciled (head priced)         12548   12548   6960       14375     -1827  W32 REFUSED
    reconciled   (head shared)         10123   10123   4535        9525      +598  FUNDS

WHAT IS PINNED HERE

1. ONE PREDICATE, not two tables: the launcher-side derivation and the
   producer-side allocation both branch on
   ``qwen3_5_mtp.mtp_builds_own_lm_head``.
2. The A/B on the SAME checkpoint: producer scope on vs off, PP2's span differs
   by EXACTLY the checkpoint's ``lm_head`` bytes, and PP0/PP1 do not move.
3. A boot where the head IS built is still priced for it -- group D's NEXTN
   drafter carries ``--speculative-algorithm`` WITHOUT
   ``--speculative-draft-kv-only`` and must keep the full 4043 MiB term.
4. ``tie_word_embeddings`` is not deferrable, in both callers.

Hermetic apart from the checkpoint: ``CUDA_VISIBLE_DEVICES=""``, no server, no
NVML, no GPU. The checkpoint-bound cases SKIP where it is absent (a remote
desk) rather than inventing weight headers.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table
from sglang.test.test_utils import CustomTestCase

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"

#: The rg6-proven incumbent cut this train ships.
LAYERS = [32, 18, 14]
ATTN = [8, 4, 4]


def _p_argv(*, draft_kv_only: bool, spec: bool = True):
    """A group-P argv shaped like the one the launcher builds."""
    argv = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--pp-stage-ratio", ",".join(str(n) for n in LAYERS),
        "--pp-attn-stage-ratio", ",".join(str(a) for a in ATTN),
        "--pp-size", "3", "--tp-size", "1",
    ]
    if spec:
        argv += [
            "--speculative-algorithm", "NEXTN",
            "--speculative-num-steps", "2",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "3",
        ]
    if draft_kv_only:
        argv += ["--speculative-draft-kv-only"]
    return argv


class TestTheOnePredicate(CustomTestCase):
    """No checkpoint needed: this is the decision itself."""

    def test_the_producer_and_the_launcher_branch_on_the_same_function(self):
        # The point of the reconcile. Two callers, two PROCESSES, one function.
        import ast
        import inspect

        from sglang.srt.models import qwen3_5_mtp

        for fn in (qwen3_5_mtp.build_mtp_lm_head, ring_table.checkpoint_stage_weights):
            src = inspect.getsource(fn)
            self.assertIn(
                "mtp_builds_own_lm_head", src,
                f"{fn.__name__} must ask the shared predicate, not restate it",
            )
        # ... and the producer-side branch must not have grown a second copy of
        # the condition beside the call.
        tree = ast.parse(inspect.getsource(qwen3_5_mtp.build_mtp_lm_head))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "mtp_builds_own_lm_head"
        ]
        self.assertEqual(len(calls), 1)

    def test_the_truth_table(self):
        from sglang.srt.models.qwen3_5_mtp import mtp_builds_own_lm_head

        # scope OFF -> the head builds its own table, whatever the tie is.
        self.assertTrue(mtp_builds_own_lm_head(False, False))
        self.assertTrue(mtp_builds_own_lm_head(False, True))
        # scope ON -> deferred ... except under tie_word_embeddings, where the
        # head IS the module's own resident embedding and no second table
        # exists to defer.
        self.assertFalse(mtp_builds_own_lm_head(True, False))
        self.assertTrue(mtp_builds_own_lm_head(True, True))

    def test_the_argv_predicate_is_the_flag_the_scheduler_gates_the_producer_on(self):
        # scheduler.py returns before constructing DraftKvProducer unless
        # server_args.speculative_draft_kv_only; DraftKvProducer.__init__ then
        # enters lm_head_from_target() unconditionally. So the flag's presence
        # in group P's argv IS the scope being active for that boot.
        import inspect

        from sglang.srt.speculative import draft_kv_producer

        src = inspect.getsource(draft_kv_producer.DraftKvProducer.__init__)
        self.assertIn("lm_head_from_target()", src)
        self.assertIn(
            "--speculative-draft-kv-only",
            inspect.getsource(ring_table.stage_weights_from_argv),
        )


@unittest.skipUnless(os.path.isdir(MODEL), f"checkpoint absent: {MODEL}")
class TestTheSpanFollowsTheImageThatExists(CustomTestCase):

    @staticmethod
    def _head_mib():
        from sglang.srt.planner import pp_cut

        return pp_cut.checkpoint_weight_terms(MODEL).lm_head_weight_bytes / ring_table.MIB

    def test_pp2_span_differs_by_exactly_the_heads_own_lm_head(self):
        """THE A/B. Same checkpoint, same cut, producer scope on vs off."""
        off = ring_table.checkpoint_stage_weights(
            MODEL, LAYERS, ATTN, True, drafter_head_from_target=False)
        on = ring_table.checkpoint_stage_weights(
            MODEL, LAYERS, ATTN, True, drafter_head_from_target=True)
        head = self._head_mib()
        self.assertGreater(head, 0.0)
        self.assertAlmostEqual(
            off[2].total_mib - on[2].total_mib, head, places=2,
            msg="PP2's span must fall by exactly the head's own lm_head bytes",
        )
        # AND NOWHERE ELSE. The drafter lands on the LAST stage only; a
        # derivation that moved any other stage would be re-pricing the target.
        for s in (0, 1):
            self.assertAlmostEqual(off[s].total_mib, on[s].total_mib, places=6)

    def test_the_target_lm_head_is_untouched_it_is_the_drafters_copy_that_goes(self):
        # The stage still carries the TARGET's output table -- that is the one
        # the shared head IS. Only the second copy disappears.
        on = ring_table.checkpoint_stage_weights(
            MODEL, LAYERS, ATTN, True, drafter_head_from_target=True)
        head = self._head_mib()
        self.assertAlmostEqual(on[2].lm_head_mib, head, places=2)
        self.assertGreater(on[2].drafter_mib, 0.0, "mtp + embedding still cost")
        self.assertLess(on[2].drafter_mib, head)

    def test_the_terms_line_says_the_table_is_not_built(self):
        # An absent term must be reported as absent WITH ITS REASON, never as a
        # silently smaller number -- the L6 row is meant to be redoable by hand.
        on = ring_table.checkpoint_stage_weights(
            MODEL, LAYERS, ATTN, True, drafter_head_from_target=True)
        self.assertIn("NOT BUILT", on[2].terms())
        self.assertIn("lm_head_from_target", on[2].terms())

    def test_a_drafter_that_does_build_its_head_is_still_priced_for_it(self):
        """Group D's NEXTN drafter: --speculative-algorithm WITHOUT
        --speculative-draft-kv-only. Nothing enters the scope there, so the
        full 4043 MiB term must survive."""
        d_like, why = ring_table.stage_weights_from_argv(
            _p_argv(draft_kv_only=False))
        self.assertIsNotNone(d_like, why)
        producer, why = ring_table.stage_weights_from_argv(
            _p_argv(draft_kv_only=True))
        self.assertIsNotNone(producer, why)
        self.assertAlmostEqual(
            d_like[2].total_mib - producer[2].total_mib, self._head_mib(), places=2)

    def test_no_drafter_at_all_prices_no_drafter_term(self):
        none_, why = ring_table.stage_weights_from_argv(
            _p_argv(draft_kv_only=False, spec=False))
        self.assertIsNotNone(none_, why)
        self.assertEqual(none_[2].drafter_mib, 0.0)
        self.assertEqual(none_[2].drafter_terms, "")

    def test_the_tie_is_read_off_the_checkpoints_own_config(self):
        # This checkpoint is untied, so the deferral applies. Asserted rather
        # than assumed, because a TIED checkpoint must keep the head priced and
        # this is the input that decides it.
        self.assertFalse(ring_table._tie_word_embeddings(MODEL))
        self.assertFalse(ring_table._tie_word_embeddings("/nonexistent/model"))


if __name__ == "__main__":
    unittest.main()
