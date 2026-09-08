# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1254, train fix 3): group P SHIPS the incumbent cut, and the argv
and the flip-order map must state the SAME split.

THE DEFECT, measured on the train tip a917eb404c the moment the W21 store
refusal cleared and the dry-run reached these lines for the first time
(`--dry-run --idle-layout tp`, 2026-09-08 10:11Z)::

    WEG2-PP-SPLIT group=P scores --pp-stage-ratio [32, 18, 14]
        --pp-attn-stage-ratio [8, 4, 4] -> DERIVED layer split [32, 18, 14]
    ...
    PP-CUT solver: layers=44,10,10 attn=11,2,3 makespan_ms=360.8
        pool_tokens=499967 (alternatives: kv-floor cut 31,17,16 attn 7,5,4
        pool 988900 makespan 561.7)
    group P argv: ... --pp-stage-ratio 44,10,10 --pp-attn-stage-ratio 11,2,3

Two facts in one boot: ``main`` builds the WEG2-FLIP-ORDER MAP from the
INCUMBENT score constants via :func:`launcher.p_stage_layers`, while ``argv_p``
ships whatever :func:`launcher.solve_p_cut` returned.  While those differ the
map is COMPLETE BUT WRONG PER CARD (a weights tag charged to one card while
its bytes straddle two) -- the accounting class that killed boot weg2dk4 --
and ``interleave_pause_order`` refuses only an INCOMPLETE map, so the boot gets
a confident wrong pause order rather than a refusal.

Two things are pinned here, both of which that dry-run violated:

* the SHIPPED cut is chosen by objective and defaults to the INCUMBENT
  32,18,14 / attention 8,4,4 -- boot-proven on weg2rg6 (pool 714,788) and, by
  identity, the vector the map is derived from.  The maxkv law (user
  2026-09-08) allows the solver's speed cut (pool -47.7 %) only behind an
  explicit objective flag; both alternatives stay PRICED and PRINTED either
  way so the trade is visible without being paid by accident;
* the argv split and the map split are COMPARED before either group starts
  (W46), so a future divergence is a named refusal and never a silent one.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no NVML, no GPU.  The two
tests that need the checkpoint's real layer_types read its ``config.json`` and
SKIP where the checkpoint is absent (a remote desk), rather than inventing a
layer pattern the snap would treat differently.
"""

import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner.pp_cut_launch import CutCandidate, CutDecision
from sglang.srt.weg2 import launcher
from sglang.test.test_utils import CustomTestCase

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"

#: The three cuts of that dry-run, verbatim.
INCUMBENT = CutCandidate(layers=(32, 18, 14), attn=(8, 4, 4), makespan_ms=1362.3,
                         pool_tokens=714788.0)
SPEED = CutCandidate(layers=(44, 10, 10), attn=(11, 2, 3), makespan_ms=360.8,
                     pool_tokens=499967.0)
KV_FLOOR = CutCandidate(layers=(31, 17, 16), attn=(7, 5, 4), makespan_ms=561.7,
                        pool_tokens=988900.0)
GAPPED = CutCandidate(layers=(62, 1, 1), attn=(14, 1, 1), makespan_ms=537.7,
                      pool_tokens=153611.0, kind="gapped", layer_set="0-61,62,63")


def _decision(chosen=SPEED, ranked=(SPEED, KV_FLOOR, INCUMBENT), pinned=False):
    return CutDecision(
        chosen=chosen,
        kv_floor=KV_FLOOR,
        cap_tokens=262144,
        pinned=pinned,
        cost_provenance="test",
        ranked=tuple(ranked),
    )


def _kinds():
    with open(os.path.join(MODEL, "config.json")) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config") or cfg
    return [str(k) == "full_attention" for k in text["layer_types"]]


class TestShippedCutObjective(CustomTestCase):
    def test_default_objective_ships_the_incumbent_not_the_ranking_winner(self):
        cand, why = launcher.pick_shipped_cut(
            _decision(), launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "incumbent",
        )
        self.assertEqual(cand.layers, (32, 18, 14))
        self.assertEqual(cand.attn, (8, 4, 4))
        self.assertIn("INCUMBENT", why)
        # ... and it is the SOLVER'S OWN priced row, not one built here: the
        # pool figure comes back with it.
        self.assertEqual(cand.pool_tokens, INCUMBENT.pool_tokens)

    def test_the_parser_default_is_the_incumbent(self):
        act = next(
            a for a in launcher.build_parser()._actions
            if a.dest == "p_cut_objective"
        )
        self.assertEqual(act.default, "incumbent")
        self.assertEqual(tuple(act.choices), ("incumbent", "maxkv", "speed"))

    def test_speed_and_maxkv_ship_only_when_asked(self):
        speed, _ = launcher.pick_shipped_cut(
            _decision(), launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "speed")
        self.assertEqual(speed.layers, SPEED.layers)
        maxkv, _ = launcher.pick_shipped_cut(
            _decision(), launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "maxkv")
        self.assertEqual(maxkv.layers, KV_FLOOR.layers)

    def test_an_unranked_incumbent_refuses_instead_of_substituting(self):
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.pick_shipped_cut(
                _decision(ranked=(SPEED, KV_FLOOR)),
                launcher.P_PP_STAGE_RATIO_SCORES,
                launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "incumbent")
        self.assertIn("W40", str(cm.exception))
        self.assertIn("32,18,14", str(cm.exception))

    def test_the_incumbent_objective_never_reaches_the_gapped_branch(self):
        """A gapped winner cannot ship under the default: its layout is a SET
        the count form cannot state, and the map is contiguous."""
        cand, _ = launcher.pick_shipped_cut(
            _decision(chosen=GAPPED, ranked=(GAPPED, SPEED, KV_FLOOR, INCUMBENT)),
            launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "incumbent")
        self.assertEqual(cand.kind, "contiguous")
        self.assertEqual(cand.layers, (32, 18, 14))


@unittest.skipUnless(os.path.isdir(MODEL), f"checkpoint absent: {MODEL}")
class TestArgvSplitAgreesWithTheMap(CustomTestCase):
    """The W46 premise, against the checkpoint's own layer_types."""

    def test_the_incumbent_argv_and_the_map_are_the_same_split(self):
        kinds = _kinds()
        argv_split = launcher.shipped_layer_split(
            ",".join(str(n) for n in launcher.P_PP_STAGE_RATIO_SCORES),
            ",".join(str(a) for a in launcher.P_PP_ATTN_STAGE_RATIO_SCORES),
            "", kinds, 3,
        )
        self.assertEqual(list(argv_split), list(launcher.p_stage_layers(kinds)))
        self.assertEqual(list(argv_split), [32, 18, 14])

    def test_the_solved_speed_cut_would_have_diverged_from_the_map(self):
        """THE DEFECT, as arithmetic: the argv of the 10:11Z dry-run and the
        map of the same run describe different layouts."""
        kinds = _kinds()
        argv_split = launcher.shipped_layer_split("44,10,10", "11,2,3", "", kinds, 3)
        self.assertEqual(list(argv_split), [44, 10, 10])
        self.assertNotEqual(list(argv_split), list(launcher.p_stage_layers(kinds)))

    def test_the_kv_floor_solve_would_also_have_diverged(self):
        """Named so the record cannot read as 'any kv-floor cut is safe': this
        box's maxkv solve is 31,17,16, which is NOT the rg6-proven pin."""
        kinds = _kinds()
        argv_split = launcher.shipped_layer_split("31,17,16", "7,5,4", "", kinds, 3)
        self.assertNotEqual(list(argv_split), list(launcher.p_stage_layers(kinds)))

    def test_a_layer_set_argv_is_resolved_by_the_runtimes_own_parser(self):
        kinds = _kinds()
        self.assertEqual(
            list(launcher.shipped_layer_split("", "", "0-31;32-49;50-63", kinds, 3)),
            [32, 18, 14],
        )

    def test_no_cut_stated_is_an_empty_answer_never_a_false_agreement(self):
        self.assertEqual(launcher.shipped_layer_split("", "", "", _kinds(), 3), [])


if __name__ == "__main__":
    unittest.main()
