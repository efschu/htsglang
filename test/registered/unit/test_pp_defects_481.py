# SPDX-License-Identifier: Apache-2.0
"""#481: three pipeline-parallel defects found in the #445 window.

Each class is a falsifier first: it fails against the unfixed tree and passes
against the fixed one, and every class also pins the NON-PP behaviour it must
leave alone.

a) ``--pp-stage-ratio`` could not read a GGUF checkpoint's sibling
   ``config.json``. ``--model-path`` names the ``.gguf`` FILE for every GGUF
   launch in this fork (rig-runbook §4.5.4b), while the depth probe joined
   ``config.json`` onto that path, so it looked for
   ``<...>.gguf/config.json``. The flag then refused with "hidden layer count
   is not readable" on every GGUF model. The canon for this already exists at
   ``server_args.py:9276-9280`` (#402/#414): when the path ends in ``.gguf``,
   the sibling config lives in its directory.

b) ``--rank-moe-resident-fraction`` validated its length against ``tp_size``
   alone. Every other rank vector in this family takes a world-length vector
   under PP in world-rank order ``pp_rank * tp_size + tp_rank``
   (``--rank-gpu-id``: ``server_args.py:9029-9036``; ``--rank-gpu-memory-mib``:
   ``:9262``), so a pipeline over heterogeneous cards could not give stage 1 a
   different fraction from stage 0 -- and a per-stage vector that WAS accepted
   was silently applied to every stage.

c) ``expert_stats`` tagged its dump ``tp{moe_tp_rank}ep{moe_ep_rank}``
   (``expert_offload.py:2625-2628``), which is not unique under PP: stage 0
   rank 0 and stage 1 rank 0 both write ``<path>.tp0ep0.json`` and the second
   dump overwrites the first.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PARTITION_ENV = "SGLANG_PP_LAYER_PARTITION"


def _write_config(directory: str, num_layers: int) -> str:
    path = os.path.join(directory, "config.json")
    with open(path, "w") as fh:
        json.dump({"num_hidden_layers": num_layers}, fh)
    return path


class TestGgufSiblingConfigDepth(CustomTestCase):
    """(a) the depth probe must find the config next to a ``.gguf`` file."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = self._dir.name
        _write_config(self.root, 48)
        self.gguf = os.path.join(self.root, "model-00001-of-00004.gguf")
        with open(self.gguf, "wb") as fh:
            fh.write(b"GGUF")
        self._env = os.environ.pop(PARTITION_ENV, None)

    def tearDown(self):
        os.environ.pop(PARTITION_ENV, None)
        if self._env is not None:
            os.environ[PARTITION_ENV] = self._env

    @staticmethod
    def _args_for(model_path: str, **kwargs) -> ServerArgs:
        """ServerArgs pointing at ``model_path`` without booting anything.

        The full ``__post_init__`` demands an accelerator, so it is entered
        with the placeholder path it short-circuits on
        (``server_args.py:5730-5731``) and the real path is set afterwards.
        Every method under test reads ``model_path`` when it is CALLED.
        """
        args = ServerArgs(model_path="dummy", **kwargs)
        args.model_path = model_path
        return args

    def test_depth_is_read_from_the_sibling_config_of_a_gguf_file(self):
        args = self._args_for(self.gguf)
        self.assertEqual(args.declared_num_hidden_layers(), 48)
        self.assertEqual(
            args.declared_config_path(), os.path.join(self.root, "config.json")
        )

    def test_layer_kinds_come_from_the_same_file(self):
        args = self._args_for(self.gguf)
        kinds = args.declared_layer_kinds()
        self.assertIsNotNone(kinds)
        self.assertEqual(len(kinds), 48)
        self.assertTrue(all(kinds))

    def test_pp_stage_ratio_derives_a_split_for_a_gguf_checkpoint(self):
        args = self._args_for(self.gguf, pp_size=2, pp_stage_ratio=[3, 1])
        args._handle_pp_stage_ratio()
        self.assertEqual(args.pp_layer_ratio, [36, 12])

    def test_a_directory_model_path_still_reads_its_own_config(self):
        """Neutrality: the non-GGUF path is untouched."""
        args = self._args_for(self.root)
        self.assertEqual(args.declared_num_hidden_layers(), 48)
        self.assertEqual(
            args.declared_config_path(), os.path.join(self.root, "config.json")
        )

    def test_a_gguf_file_without_a_sibling_config_still_reports_none(self):
        """The refusal must survive where there really is no config."""
        with tempfile.TemporaryDirectory() as bare:
            lonely = os.path.join(bare, "solo.gguf")
            with open(lonely, "wb") as fh:
                fh.write(b"GGUF")
            args = self._args_for(lonely)
            self.assertIsNone(args.declared_config_path())
            self.assertIsNone(args.declared_num_hidden_layers())
            with self.assertRaisesRegex(ValueError, "Pass --pp-layer-ratio"):
                args.pp_size = 2
                args.pp_stage_ratio = [3, 1]
                args._handle_pp_stage_ratio()


class TestResidentFractionWorldLength(CustomTestCase):
    """(b) the vector's length must be judged against the world under PP."""

    def _validate(self, **kwargs):
        """Run the flag's own validator.

        ``ServerArgs.__post_init__`` returns immediately for the placeholder
        model path (``server_args.py:5730-5731``), so the handler is called
        directly -- the same shape ``test_pp_stage_ratio.py`` uses.
        """
        args = ServerArgs(model_path="dummy", **kwargs)
        args._handle_uneven_tp()
        return args

    def test_world_length_vector_is_accepted_under_pp(self):
        args = self._validate(
            tp_size=2,
            pp_size=2,
            rank_moe_resident_fraction=[0.5, 0.5, 0.4, 0.4],
        )
        self.assertEqual(args.rank_moe_resident_fraction, [0.5, 0.5, 0.4, 0.4])

    def test_tp_length_vector_is_still_accepted_under_pp(self):
        """A per-stage vector keeps meaning "the same on every stage"."""
        args = self._validate(
            tp_size=2, pp_size=2, rank_moe_resident_fraction=[0.5, 0.4]
        )
        self.assertEqual(args.rank_moe_resident_fraction, [0.5, 0.4])

    def test_a_length_that_is_neither_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            self._validate(
                tp_size=2, pp_size=2, rank_moe_resident_fraction=[0.5, 0.4, 0.3]
            )
        self.assertIn("--rank-moe-resident-fraction", str(ctx.exception))
        self.assertIn("world", str(ctx.exception).lower())

    def test_without_pp_the_rule_is_exactly_the_old_one(self):
        """Neutrality: tp_size or 1, nothing else, and the same message."""
        self._validate(tp_size=3, rank_moe_resident_fraction=[0.5, 0.4, 0.3])
        self._validate(tp_size=3, rank_moe_resident_fraction=[0.5])
        with self.assertRaises(ValueError) as ctx:
            self._validate(tp_size=3, rank_moe_resident_fraction=[0.5, 0.4])
        self.assertIn("must equal --tp-size (3)", str(ctx.exception))
        self.assertNotIn("world", str(ctx.exception).lower())

    def test_the_consumer_indexes_a_world_vector_by_world_rank(self):
        from sglang.srt.layers.moe import resident_fraction as rf

        vec = (0.5, 0.5, 0.4, 0.4)
        with patch.object(rf, "_from_flag", return_value=vec), patch.object(
            rf, "_from_env", return_value=None
        ), patch.object(rf, "_moe_tp_size", return_value=2), patch.object(
            rf, "_tp_size", return_value=2
        ), patch.object(
            rf, "_pp_size", return_value=2
        ):
            for pp_rank, tp_rank, expected in (
                (0, 0, 0.5),
                (0, 1, 0.5),
                (1, 0, 0.4),
                (1, 1, 0.4),
            ):
                with patch.object(rf, "_pp_rank", return_value=pp_rank), patch.object(
                    rf, "_tp_rank", return_value=tp_rank
                ):
                    self.assertEqual(rf.resident_fraction_for_rank(), expected)

    def test_a_stage_length_vector_still_broadcasts_across_stages(self):
        """Neutrality for the shape that already worked."""
        from sglang.srt.layers.moe import resident_fraction as rf

        vec = (0.5, 0.4)
        with patch.object(rf, "_from_flag", return_value=vec), patch.object(
            rf, "_from_env", return_value=None
        ), patch.object(rf, "_moe_tp_size", return_value=2), patch.object(
            rf, "_tp_size", return_value=2
        ), patch.object(
            rf, "_pp_size", return_value=2
        ):
            for pp_rank, tp_rank, expected in ((0, 1, 0.4), (1, 1, 0.4)):
                with patch.object(rf, "_pp_rank", return_value=pp_rank), patch.object(
                    rf, "_tp_rank", return_value=tp_rank
                ):
                    self.assertEqual(rf.resident_fraction_for_rank(), expected)

    def test_without_pp_the_vector_is_indexed_exactly_as_before(self):
        from sglang.srt.layers.moe import resident_fraction as rf

        vec = (0.5, 0.4, 0.3)
        with patch.object(rf, "_from_flag", return_value=vec), patch.object(
            rf, "_from_env", return_value=None
        ), patch.object(rf, "_moe_tp_size", return_value=3), patch.object(
            rf, "_tp_size", return_value=3
        ), patch.object(
            rf, "_pp_size", return_value=1
        ), patch.object(
            rf, "_pp_rank", return_value=0
        ):
            for tp_rank, expected in ((0, 0.5), (1, 0.4), (2, 0.3)):
                with patch.object(rf, "_tp_rank", return_value=tp_rank):
                    self.assertEqual(rf.resident_fraction_for_rank(), expected)


class TestExpertStatsRankTag(CustomTestCase):
    """(c) the dump tag must separate pipeline stages."""

    def test_stages_get_distinct_tags(self):
        from sglang.srt.layers.moe.expert_stats import moe_rank_tag

        self.assertEqual(
            moe_rank_tag(moe_tp_rank=0, moe_ep_rank=0, pp_rank=0, pp_size=2),
            "pp0tp0ep0",
        )
        self.assertEqual(
            moe_rank_tag(moe_tp_rank=0, moe_ep_rank=0, pp_rank=1, pp_size=2),
            "pp1tp0ep0",
        )
        self.assertNotEqual(
            moe_rank_tag(moe_tp_rank=0, moe_ep_rank=0, pp_rank=0, pp_size=2),
            moe_rank_tag(moe_tp_rank=0, moe_ep_rank=0, pp_rank=1, pp_size=2),
        )

    def test_without_pp_the_tag_is_byte_identical_to_the_old_one(self):
        """Neutrality: existing dump filenames must not move."""
        from sglang.srt.layers.moe.expert_stats import moe_rank_tag

        for tp_rank in range(3):
            for ep_rank in range(2):
                self.assertEqual(
                    moe_rank_tag(
                        moe_tp_rank=tp_rank, moe_ep_rank=ep_rank, pp_rank=0, pp_size=1
                    ),
                    f"tp{tp_rank}ep{ep_rank}",
                )

    def test_the_tag_reaches_the_output_path(self):
        from sglang.srt.layers.moe.expert_stats import ExpertStatsCollector

        first = ExpertStatsCollector(
            path="/tmp/stats", rank_tag="pp0tp0ep0"
        ).output_path()
        second = ExpertStatsCollector(
            path="/tmp/stats", rank_tag="pp1tp0ep0"
        ).output_path()
        self.assertNotEqual(first, second)

    def test_the_offload_call_site_resolves_the_pipeline_rank(self):
        """The tag builder must be what the production call site uses.

        A helper nobody calls would pass every assertion above and change
        nothing, so this pins the wiring rather than the helper.
        """
        import inspect

        from sglang.srt.layers.moe import expert_offload

        source = inspect.getsource(expert_offload.MoEExpertOffloadCache.__init__)
        self.assertTrue(
            "moe_rank_tag(" in source,
            "the offload call site does not build its tag with moe_rank_tag",
        )
        self.assertTrue(
            'f"tp{getattr(layer' not in source,
            "the offload call site still builds the old pp-blind tag inline",
        )


if __name__ == "__main__":
    unittest.main()
