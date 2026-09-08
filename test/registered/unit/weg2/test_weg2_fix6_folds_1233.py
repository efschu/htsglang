# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1233, fix 6): the four cheap seams the fix-5 review named beside the
blocking denominator.

1. ONE AUTHORITY FOR THE LAYER KINDS.  ``launcher.model_layer_kinds`` read only
   ``layer_types`` and refused otherwise, while the server's own
   ``ServerArgs.declared_layer_kinds`` -- the authority it claims to mirror --
   also accepts ``layers_block_type`` and ``full_attention_interval``.  On such
   a checkpoint the server derives a real hybrid split while the launcher
   publishes NO chunk->card map, and no map means ``interleave_pause_order``
   returns the identity order: the order that killed boot weg2dk4 with a device
   OOM.  The two now share ONE derivation
   (``server_args.declared_layer_kinds_from_config``), so the probe order
   cannot drift either.
2. AN EMPTY CARD LIST IS A REFUSAL.  ``interleave_pause_order`` refused an
   INCOMPLETE map but would have raised ``ValueError`` inside ``front.flip`` on
   a tag whose card list is empty -- a crash at the flip instead of a named
   degradation.
3. W11 GETS ITS SECOND INSTRUMENT.  After fix 3, ``resident_mib`` is a model-
   GRAPH quantity: a table ``_drop_parameters`` unbinds while another holder
   keeps it alive vanishes from ``model.parameters()`` while still occupying
   VRAM -- precisely the fix-2 failure the W11 gate was built to catch, and now
   invisible to it.  The BUILD ACCOUNTING identity closes that:
   ``nvml_delta_mib`` (the build, under the memory saver) must be explained by
   ``resident_mib + head_released_mib`` within a measured tolerance.
4. THE DEAD CLAUSE GOES.  ``_draft_kv_producer_wants`` carried
   ``not batch.forward_mode.is_idle()`` after ``is_extend()``, and
   ``ForwardMode.IDLE.is_extend()`` is already False -- an unfalsifiable guard
   reads as protection that is not there.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no GPU.
"""

import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front, launcher
from sglang.test.test_utils import CustomTestCase


def _body(fn):
    """The function's source with its docstring removed."""
    doc = inspect.getdoc(fn) or ""
    src = inspect.getsource(fn)
    for line in doc.splitlines():
        src = src.replace(line, "")
    return src


def _model_dir(d, cfg):
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


class TestLayerKindsHaveOneAuthority(CustomTestCase):
    def test_layer_types_still_wins_and_still_marks_full_attention(self):
        with tempfile.TemporaryDirectory() as d:
            _model_dir(d, {
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention", "full_attention",
                    "linear_attention", "full_attention",
                ],
            })
            self.assertEqual(
                launcher.model_layer_kinds(d), [False, True, False, True]
            )

    def test_layers_block_type_is_accepted_like_the_server_accepts_it(self):
        with tempfile.TemporaryDirectory() as d:
            _model_dir(d, {
                "num_hidden_layers": 3,
                "layers_block_type": [
                    "linear_attention", "linear_attention", "full_attention",
                ],
            })
            self.assertEqual(launcher.model_layer_kinds(d), [False, False, True])

    def test_full_attention_interval_derives_the_hybrid_the_server_derives(self):
        # Qwen3.5/3.6 GDN hybrids: every interval-th layer, 1-based.  The old
        # launcher REFUSED here, published no map, and the flip fell back to the
        # identity pause order.
        with tempfile.TemporaryDirectory() as d:
            _model_dir(d, {"num_hidden_layers": 6, "full_attention_interval": 3})
            self.assertEqual(
                launcher.model_layer_kinds(d),
                [False, False, True, False, False, True],
            )

    def test_a_homogeneous_checkpoint_is_all_attention_not_a_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            _model_dir(d, {"num_hidden_layers": 3})
            self.assertEqual(launcher.model_layer_kinds(d), [True, True, True])

    def test_the_probe_order_is_the_servers_top_level_first(self):
        # The asymmetry the review named: server_args probes the top level then
        # text_config; the launcher probed text_config first.  Only a config
        # carrying the key at BOTH levels tells them apart -- and then the two
        # must not disagree about the checkpoint.
        cfg = {
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "text_config": {
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"],
            },
        }
        with tempfile.TemporaryDirectory() as d:
            _model_dir(d, cfg)
            kinds = launcher.model_layer_kinds(d)
            depth = launcher.model_num_layers(d)
        from sglang.srt.server_args import (
            declared_layer_kinds_from_config,
            declared_num_hidden_layers_from_config,
        )
        self.assertEqual(kinds, declared_layer_kinds_from_config(cfg, depth))
        self.assertEqual(depth, declared_num_hidden_layers_from_config(cfg))
        self.assertEqual(kinds, [True, True])

    def test_the_launcher_calls_the_servers_derivation_not_a_copy(self):
        # The CODE, docstring stripped: prose may name the keys it defers on,
        # the body must not re-derive them.
        code = _body(launcher.model_layer_kinds)
        self.assertIn("declared_layer_kinds_from_config", code)
        self.assertNotIn("full_attention_interval", code)
        self.assertNotIn("layers_block_type", code)

    def test_an_unreadable_depth_is_still_a_named_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            _model_dir(d, {"hidden_size": 5120})
            with self.assertRaises(launcher.Weg2LaunchRefused):
                launcher.model_layer_kinds(d)


class TestEmptyCardListIsARefusal(CustomTestCase):
    def _order(self, tag_cards):
        return front.interleave_pause_order(
            ["weights_0", "weights_1", "weights"],
            tag_cards,
            {0: 4000, 1: 9000},
        )

    def test_a_tag_with_no_cards_refuses_to_reorder_by_name(self):
        order, why = self._order({"weights_0": (0,), "weights_1": ()})
        self.assertEqual(order, ["weights_0", "weights_1", "weights"])
        self.assertIn("REFUSED", why)
        self.assertIn("weights_1", why)

    def test_a_complete_map_still_pauses_the_tightest_card_first(self):
        order, why = self._order({"weights_0": (1,), "weights_1": (0,)})
        self.assertEqual(order, ["weights_1", "weights_0", "weights"])
        self.assertEqual(why, "tightest-card-first")


class TestW11GetsItsSecondInstrument(CustomTestCase):
    #: boot weg2dk5, P log :260 -- the three instruments on one L2 line.
    DK5_LINE = (
        "WEG2 DRAFT-KV-PRODUCER armed stage=2/3 drafter=a30db4b7c362c786 layout=v1 "
        "heads=4 head_dim=256 page_bytes=2048 embed=resident mtp_mib=405.2 "
        "embed_mib=1213.0 resident_mib=1682.9 head_released_mib=2425.0 "
        "nvml_delta_mib=3998.0 embed_dtype=torch.int8 build_s=41.0"
    )

    def _log(self, line):
        fd, path = tempfile.mkstemp(suffix=".log")
        with os.fdopen(fd, "w") as f:
            f.write(line + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_the_boot_that_passed_is_accounted_for(self):
        out = launcher.check_draft_resident(self._log(self.DK5_LINE))
        self.assertTrue(out["ok"])
        self.assertEqual(out["head_released_mib"], 2425.0)
        self.assertEqual(out["nvml_delta_mib"], 3998.0)
        # build 3998.0 vs residue 1682.9 + released 2425.0 = 4107.9 -> the
        # unaccounted residual is 109.9 MiB, inside the named tolerance.
        self.assertAlmostEqual(out["unaccounted_mib"], -109.9, delta=0.05)
        self.assertTrue(out["accounted"])

    def test_a_release_that_frees_nothing_is_refused_by_the_second_instrument(self):
        # The fix-2 failure, replayed: the module was swapped, so nothing was
        # released, and the table stays on the card.  resident_mib alone still
        # passes -- it is a model-graph quantity and the table is no longer in
        # the graph.  The accounting identity is what refuses.
        line = self.DK5_LINE.replace("head_released_mib=2425.0", "head_released_mib=0.0")
        out = launcher.check_draft_resident(self._log(line))
        self.assertTrue(out["resident_ok"])
        self.assertFalse(out["accounted"])
        self.assertFalse(out["ok"])
        self.assertAlmostEqual(out["unaccounted_mib"], 2315.1, delta=0.05)

    def test_an_unmeasured_build_is_a_named_refusal_not_a_pass(self):
        line = self.DK5_LINE.replace("nvml_delta_mib=3998.0", "nvml_delta_mib=-1.0")
        out = launcher.check_draft_resident(self._log(line))
        self.assertFalse(out["ok"])
        self.assertIsNone(out["unaccounted_mib"])
        self.assertFalse(out["accounted"])

    def test_an_over_budget_residue_still_refuses_on_the_first_instrument(self):
        line = self.DK5_LINE.replace("resident_mib=1682.9", "resident_mib=3998.0")
        out = launcher.check_draft_resident(self._log(line))
        self.assertFalse(out["resident_ok"])
        self.assertFalse(out["ok"])

    def test_an_absent_line_is_still_not_ok(self):
        out = launcher.check_draft_resident(self._log("nothing to see here"))
        self.assertFalse(out["ok"])
        self.assertIsNone(out["resident_mib"])


class TestTheDeadClauseIsGone(CustomTestCase):
    def test_the_wants_predicate_no_longer_carries_an_unfalsifiable_guard(self):
        from sglang.srt.managers.scheduler import Scheduler

        code = _body(Scheduler._draft_kv_producer_wants)
        self.assertNotIn("is_idle", code)
        self.assertIn("is_extend", code)

    def test_idle_is_already_not_extend_which_is_why_it_could_go(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.assertFalse(ForwardMode.IDLE.is_extend())


if __name__ == "__main__":
    unittest.main()
