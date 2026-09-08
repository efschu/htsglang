# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1254): WHICH priced cut group P ships, and the argv/map identity.

THE DEFECT, measured on train tip a917eb404c the moment the W21 store refusal
cleared and the dry-run reached these lines for the first time
(`--dry-run --idle-layout tp`, 2026-09-08 10:11Z)::

    WEG2-PP-SPLIT group=P scores --pp-stage-ratio [32, 18, 14]
        --pp-attn-stage-ratio [8, 4, 4] -> DERIVED layer split [32, 18, 14]
    ...
    PP-CUT solver: layers=44,10,10 attn=11,2,3 makespan_ms=360.8
        pool_tokens=499967 (alternatives: kv-floor cut 31,17,16 attn 7,5,4
        pool 988900 makespan 561.7)
    group P argv: ... --pp-stage-ratio 44,10,10 --pp-attn-stage-ratio 11,2,3

Two facts in one boot: ``main`` built the WEG2-FLIP-ORDER MAP from the INCUMBENT
score constants via :func:`launcher.p_stage_layers`, while ``argv_p`` shipped
whatever :func:`launcher.solve_p_cut` returned.  While those differ the map is
COMPLETE BUT WRONG PER CARD (a weights tag charged to one card while its bytes
straddle two) -- the accounting class that killed boot weg2dk4 -- and
``interleave_pause_order`` refuses only an INCOMPLETE map, so the boot gets a
confident wrong pause order rather than a refusal.

TRAIN 2 RECONCILED TWO MECHANISMS INTO ONE, and this file follows it.  The
train answered the defect by pinning the SHIPPED cut to the incumbent
(``--p-cut-objective``); the argv slice answered it by deriving the MAP from
the shipped cut (``flip_order_split``) and making the objective the solver's
own argument (``--pp-solve-objective``, default the kv-floor per the maxkv
law).  Two flags answering "which cut ships" is the second-bookkeeping shape,
so:

* there is ONE flag, ``--pp-solve-objective``, whose default is ``maxkv`` (the
  standing law) and whose third arm ``incumbent`` selects the rg6-proven pin
  from the solver's OWN ranked field -- an unranked incumbent is a W40
  REFUSAL, never a silent substitution;
* the map is derived from the SHIPPED cut, so the identity the train enforced
  by pinning now holds by construction for every arm;
* W46 stays as a REFUSAL: its two sides are still computed along independent
  paths (the map off ``cut.layer_counts``, the argv split by re-parsing the
  argv strings through the runtime's own parser), so it is the one place that
  reads P's argv back and compares it to the map.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no NVML, no GPU.  The tests
that need the checkpoint's real layer_types read its ``config.json`` and SKIP
where the checkpoint is absent (a remote desk), rather than inventing a layer
pattern the snap would treat differently.
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


def _decision(chosen=KV_FLOOR, ranked=(SPEED, KV_FLOOR, INCUMBENT), pinned=False):
    return CutDecision(
        chosen=chosen,
        kv_floor=KV_FLOOR,
        cap_tokens=262144,
        pinned=pinned,
        cost_provenance="test",
        ranked=tuple(ranked),
        objective="maxkv",
        makespan=SPEED,
    )


def _kinds():
    with open(os.path.join(MODEL, "config.json")) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config") or cfg
    return [str(k) == "full_attention" for k in text["layer_types"]]


class TestShippedCutObjective(CustomTestCase):
    def test_the_default_arm_ships_the_kv_floor_the_maxkv_law_names(self):
        cand, why = launcher.pick_shipped_cut(
            _decision(), launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "maxkv",
        )
        self.assertEqual(cand.layers, KV_FLOOR.layers)
        self.assertEqual(cand.attn, KV_FLOOR.attn)
        self.assertIn("maxkv", why)

    def test_the_incumbent_arm_ships_the_rg6_pin_from_the_ranked_field(self):
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

    def test_the_parser_carries_one_objective_flag_defaulting_to_makespan(self):
        # TRAIN 2: the train's --p-cut-objective is GONE and its third arm
        # lives on the argv slice's flag. Two flags for "which cut ships" is
        # the defect this asserts against, so both halves are pinned: the
        # surviving flag's default AND the absence of the other.
        #
        # THE DEFAULT IS `makespan` SINCE 2026-09-08, BY USER ORDER --
        # verbatim: "nimm als default ab jetzt makespan". It is the solver's
        # SPEED cut (42,11,11 / attn 10,3,3 in the perf boots).
        #
        # WHAT THAT DEFAULT COSTS, kept here so the trade stays selected rather
        # than inherited: makespan BUYS TTFT WITH KV -- P pool 587k
        # solver-priced against the incumbent's 714k measured, for +33.5 %
        # prefill measured in pp1/pp2 only. #1254 originally took makespan AWAY
        # from the default for exactly this reason (it was being paid without
        # anyone selecting it); it returns now because it was selected, and
        # because all three arms stay priced on the PP-CUT SHIPPED: line of
        # every boot, so the cost is visible per boot.
        #
        # The incumbent's own reason survives on its arm and is unchanged: PP2
        # carries per-stage fixed posts -- lm_head, the MTP/draft head, the
        # draft pools -- that PhasePoolModel does not price (966,544 read
        # against a PROFILED 155,164), which is why `maxkv` putting SIXTEEN
        # layers there is still wrong.
        dests = {a.dest for a in launcher.build_parser()._actions}
        self.assertNotIn("p_cut_objective", dests)
        act = next(
            a for a in launcher.build_parser()._actions
            if a.dest == "pp_solve_objective"
        )
        self.assertEqual(act.default, "makespan")
        self.assertEqual(set(act.choices), {"maxkv", "makespan", "incumbent"})

    def test_incumbent_is_not_a_solver_objective_and_is_never_passed_as_one(self):
        # ``incumbent`` names a CANDIDATE, not a ranking. Passing it through to
        # solve_launch_cut would be a silent third objective the solver does
        # not implement; it is answered by lookup instead, in a maxkv ranking.
        self.assertEqual(launcher.P_SOLVER_OBJECTIVE_OF["incumbent"], "maxkv")
        self.assertEqual(launcher.P_SOLVER_OBJECTIVE_OF["maxkv"], "maxkv")
        self.assertEqual(launcher.P_SOLVER_OBJECTIVE_OF["makespan"], "makespan")
        act = next(
            a for a in launcher.build_parser()._actions
            if a.dest == "pp_solve_objective"
        )
        self.assertEqual(
            set(act.choices), set(launcher.P_SOLVER_OBJECTIVE_OF),
            "every arm of the flag must have a solver objective and vice versa",
        )

    def test_makespan_and_incumbent_ship_only_when_asked(self):
        speed, _ = launcher.pick_shipped_cut(
            _decision(), launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "makespan")
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

    def test_an_unranked_incumbent_is_priced_as_absent_not_as_a_refusal(self):
        # The SHIPPED line prices all three arms on every boot, including the
        # boots that do not ship the incumbent. Absence there must be sayable
        # ("not ranked") -- only ASKING for it and not finding it refuses.
        self.assertIsNone(
            launcher.incumbent_candidate(
                _decision(ranked=(SPEED, KV_FLOOR)),
                launcher.P_PP_STAGE_RATIO_SCORES,
                launcher.P_PP_ATTN_STAGE_RATIO_SCORES,
            )
        )
        self.assertIsNotNone(
            launcher.incumbent_candidate(
                _decision(), launcher.P_PP_STAGE_RATIO_SCORES,
                launcher.P_PP_ATTN_STAGE_RATIO_SCORES,
            )
        )

    def test_the_incumbent_objective_never_reaches_the_gapped_branch(self):
        """A gapped winner cannot ship under the incumbent arm: its layout is a
        SET the count form cannot state, and the map is contiguous."""
        cand, _ = launcher.pick_shipped_cut(
            _decision(chosen=GAPPED, ranked=(GAPPED, SPEED, KV_FLOOR, INCUMBENT)),
            launcher.P_PP_STAGE_RATIO_SCORES,
            launcher.P_PP_ATTN_STAGE_RATIO_SCORES, "incumbent")
        self.assertEqual(cand.kind, "contiguous")
        self.assertEqual(cand.layers, (32, 18, 14))

    def test_the_shipped_cut_and_not_the_ranking_winner_builds_the_argv(self):
        """The reconciliation's load-bearing edit, matched to its error class.

        ``solve_p_cut`` selects a SHIPPED candidate and then builds
        ``stage_ratio`` / ``attn_ratio`` / ``PCutFacts`` from it.  Leaving any
        of those reading ``decision.chosen`` is invisible to every behavioural
        test -- both are the same row whenever the arm is the solver's own
        objective, which is the default -- and would silently re-open the
        argv-vs-map divergence on the ``incumbent`` arm.  Asserted on the AST
        because that is where it can actually fail.
        """
        import ast
        import inspect
        import textwrap

        fn = ast.parse(textwrap.dedent(inspect.getsource(launcher.solve_p_cut))).body[0]
        # The PRICING of the alternatives is ABOUT the ranking and legitimately
        # names decision.chosen; the argv and the returned facts may not.
        watched = []
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                names = {t.id for t in n.targets if isinstance(t, ast.Name)}
                if names & {"stage_ratio", "attn_ratio", "realized"}:
                    watched.append((sorted(names)[0], ast.dump(n.value)))
            if (
                isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "PCutFacts"
            ):
                for kw in n.keywords:
                    watched.append(("PCutFacts.%s" % kw.arg, ast.dump(kw.value)))
        self.assertTrue(watched, "solve_p_cut must build the argv and PCutFacts")
        offenders = [w for w, dump in watched if "id='decision'" in dump]
        self.assertEqual(
            offenders, [],
            "these read decision.chosen (the RANKING's winner) instead of the "
            "SHIPPED candidate; on the incumbent arm they are different rows, "
            "and that divergence is exactly the weg2dk4 accounting class",
        )


@unittest.skipUnless(os.path.isdir(MODEL), f"checkpoint absent: {MODEL}")
class TestArgvSplitAgreesWithTheMap(CustomTestCase):
    """The W46 premise, against the checkpoint's own layer_types."""

    def test_the_incumbent_argv_and_the_incumbent_map_are_the_same_split(self):
        kinds = _kinds()
        argv_split = launcher.shipped_layer_split(
            ",".join(str(n) for n in launcher.P_PP_STAGE_RATIO_SCORES),
            ",".join(str(a) for a in launcher.P_PP_ATTN_STAGE_RATIO_SCORES),
            "", kinds, 3,
        )
        self.assertEqual(list(argv_split), list(launcher.p_stage_layers(kinds)))
        self.assertEqual(list(argv_split), [32, 18, 14])

    def test_w46_agrees_for_every_arm_because_the_map_comes_off_the_cut(self):
        """TRAIN 2: what the two 'would have diverged' cases became.

        Before the reconciliation the map was the INCUMBENT's split and the
        argv was the SOLVED cut's, so 44,10,10 and 31,17,16 both diverged from
        it -- that is the defect, and it is still stated below against the
        incumbent so the record cannot read as 'any kv-floor cut is safe'.
        The map is derived from the shipped cut now, so W46's two sides agree
        for EVERY arm, and this walks all three.
        """
        kinds = _kinds()
        for stage, attn in (("32,18,14", "8,4,4"), ("31,17,16", "7,5,4"),
                            ("44,10,10", "11,2,3")):
            argv_split = launcher.shipped_layer_split(stage, attn, "", kinds, 3)
            cut = launcher.PCutFacts(
                stage_ratio=stage, attn_stage_ratio=attn, pool_tokens=0.0,
                attn_counts=tuple(int(a) for a in attn.split(",")),
                kv_mib_per_token_per_attn_layer=0.0, hidden_size=0,
                cap_tokens=0, layer_counts=tuple(argv_split),
            )
            map_split, note = launcher.flip_order_split(cut, len(kinds))
            self.assertEqual(list(map_split), list(argv_split), f"{stage}: {note}")

    def test_the_solved_cuts_still_differ_from_the_incumbent_split(self):
        """The defect's arithmetic, kept: these are the layouts that were
        shipped while the map answered for 32,18,14."""
        kinds = _kinds()
        for stage, attn, expect in (("44,10,10", "11,2,3", [44, 10, 10]),
                                    ("31,17,16", "7,5,4", None)):
            argv_split = launcher.shipped_layer_split(stage, attn, "", kinds, 3)
            if expect is not None:
                self.assertEqual(list(argv_split), expect)
            self.assertNotEqual(
                list(argv_split), list(launcher.p_stage_layers(kinds))
            )

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
