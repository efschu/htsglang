# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 5 (29.09.): the expert-params mapping of Qwen4-Exp's load_weights
as an index instead of a linear scan.

Metal (z30w-park, NF P, PP0): weight_loading 61.8 s for 133632 expert tensors;
per tensor the loader thread walked ``make_expert_params_mapping`` (1536
entries for 512 experts) testing ``weight_name in name`` until the first hit.
The identical loop over PP0's names costs 10.8 s on the rig's CPU (81 us per
tensor), all of it under the GIL the four expert consumers need.

What the index must never change: the entries the loop body sees and their
order. ``candidates(name)`` is the list the scan would visit without hitting
``continue``, i.e. ``[m for m in mapping if m[1] in name]``.
"""

import ast
import inspect
import re
import textwrap
import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.model_loader.expert_mapping_index import ExpertMappingIndex
from sglang.srt.models import qwen4_exp

NUM_EXPERTS = 512  # Qwen3.8-Flash-Next


def _nf_mapping():
    return FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=NUM_EXPERTS,
    )


def _scan(mapping, name):
    return [m for m in mapping if m[1] in name]


def _nf_names(layers):
    for prefix in ("model.layers", "model.language_model.layers"):
        for layer in layers:
            for e in range(NUM_EXPERTS + 1):  # +1: the fused shared expert id
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for kind in ("weight_packed", "weight_scale", "weight_shape"):
                        yield f"{prefix}.{layer}.mlp.experts.{e}.{proj}.{kind}"


class TestExpertMappingIndexEquivalence(unittest.TestCase):
    def test_every_nf_expert_name_sees_the_scans_entries_in_order(self):
        # Derived property: the index returns what the scan would stop at.
        # Layers 0/9/47 cover one- and two-digit layer ids; experts 0..512
        # cover every id width the checkpoint has plus the unmapped 512.
        mapping = _nf_mapping()
        index = ExpertMappingIndex(mapping)
        n = 0
        for name in _nf_names((0, 9, 47)):
            self.assertEqual(index.candidates(name), _scan(mapping, name), name)
            n += 1
        self.assertEqual(n, 2 * 3 * (NUM_EXPERTS + 1) * 9)
        self.assertEqual(index.lookups, n)

    def test_names_the_substring_rule_treats_specially(self):
        # Overlapping and repeated occurrences, prefixes of longer ids, and
        # names without any expert segment -- the cases a regex can get wrong.
        mapping = _nf_mapping()
        index = ExpertMappingIndex(mapping)
        for name in (
            "model.layers.3.mlp.experts.3.experts.4.gate_proj.weight_packed",
            "model.layers.3.mlp.experts.12.gate_proj.experts.1.up_proj.x",
            "model.layers.3.mlp.experts.012.gate_proj.weight_packed",
            "model.layers.3.mlp.experts.1.gate_proj",  # no trailing dot
            "model.layers.3.mlp.experts.gate_up_proj",
            "model.layers.3.mlp.shared_expert.gate_proj.weight_packed",
            "model.layers.3.mlp.shared_experts.7.gate_proj.weight",
            "visual.blocks.0.mlp.experts.7.down_proj.weight",
            "model.layers.3.mlp.experts.7.down_proj..weight",
            "",
        ):
            self.assertEqual(index.candidates(name), _scan(mapping, name), name)

    def test_entries_of_another_form_are_still_scanned_in_list_order(self):
        # A mapping entry the index cannot key (not "experts.<id>.<seg>.")
        # must keep its substring test AND its list position.
        mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w13_", "experts.5.gate_proj.", 5, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
            ("experts.w2_", "experts.5.down_proj.", 5, "w2"),
            ("experts.w13_", "experts.5.gate_proj.", 5, "w3"),  # duplicate key
        ]
        index = ExpertMappingIndex(mapping)
        self.assertEqual(index.unindexed, 2)
        for name in (
            "model.layers.1.mlp.experts.gate_up_proj.experts.5.gate_proj.w",
            "model.layers.1.mlp.experts.5.down_proj.experts.down_proj",
            "model.layers.1.mlp.experts.5.gate_proj.weight",
            "model.layers.1.mlp.experts.down_proj",
        ):
            self.assertEqual(index.candidates(name), _scan(mapping, name), name)

    def test_the_nf_mapping_is_fully_indexed(self):
        # The speed is the whole point: if make_expert_params_mapping ever
        # changes the weight_name form, every entry falls back to the scan and
        # the index silently costs what the scan cost.
        self.assertEqual(ExpertMappingIndex(_nf_mapping()).unindexed, 0)


class TestLoadWeightsUsesTheIndex(unittest.TestCase):
    def _loop_src(self):
        return textwrap.dedent(
            inspect.getsource(
                qwen4_exp.Qwen4ExpForConditionalGeneration._load_weights_with_pool
            )
        )

    def test_switch_is_on_by_default_and_off_keeps_the_scan(self):
        self.assertTrue(envs.SGLANG_OPT_LOAD_EXPERT_MAPPING_INDEX.get())
        mapping = _nf_mapping()
        self.assertIsInstance(qwen4_exp._expert_mapping_index(mapping), ExpertMappingIndex)
        with envs.SGLANG_OPT_LOAD_EXPERT_MAPPING_INDEX.override(False):
            self.assertIsNone(qwen4_exp._expert_mapping_index(mapping))

    def test_non_fused_branch_iterates_the_candidates(self):
        # Call edge: the loop builds the index once and the non-fused branch
        # asks it per name; the fused branch keeps its own two-entry list.
        fn = ast.parse(self._loop_src()).body[0]
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        built = [
            c for c in calls
            if isinstance(c.func, ast.Name) and c.func.id == "_expert_mapping_index"
        ]
        self.assertEqual(len(built), 1)
        asked = [
            c for c in calls
            if isinstance(c.func, ast.Attribute) and c.func.attr == "candidates"
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "expert_mapping_index"
        ]
        self.assertEqual(len(asked), 1)
        self.assertEqual([a.id for a in asked[0].args if isinstance(a, ast.Name)], ["name"])

    def test_census_line_formats_every_argument(self):
        fn = ast.parse(self._loop_src()).body[0]
        hits = []
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "info" and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and str(n.args[0].value).startswith("BOOTZEIT5 EXPERT-MAPPING")):
                hits.append((n.args[0].value, len(n.args) - 1))
        self.assertEqual(len(hits), 1)
        fmt, nargs = hits[0]
        self.assertEqual(len(re.findall(r"%[-+0-9.]*[sdf]", fmt)), nargs)


if __name__ == "__main__":
    unittest.main()
