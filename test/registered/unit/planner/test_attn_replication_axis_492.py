"""The attention/KV family has TWO distribution axes, not one (#492).

The correction to #485 slice 1. That slice searched the attention family over
HEAD partitions, found that a 4-kv-head grid over 3 ranks with a >= 1-unit
floor admits exactly one of them, and recorded the family as "grid-pinned".
The conclusion does not follow from the search: kv heads are CLONEABLE. Under
this fork's own #62/#116 machinery every rank holds the full replicated
kv-heads and only its own token shard, and runs the attention core over the
all-gathered head set -- so the core's per-rank mass follows the DCP TOKEN
vector, which has no grid at all.

The falsifier is executed rather than asserted in prose: the whole #485
candidate space is enumerated and its realized attention partitions counted.
One partition means a solver restricted to that axis cannot move the family
whatever it does, which is what makes the "pinned" reading a category error
about the space rather than a fact about the family.

What must not move: everything #485 shipped. With no core plan in play the
arithmetic executes the identical float operations, so the four measured #475
arms and the whole joint-pair backtest are byte-identical.
"""

import math
import os
import unittest

from sglang.srt.distributed.utils import (
    attn_kv_replicated,
    partition_units,
    uneven_dcp_kv_replicated,
    set_tp_partition_ratios,
)
from sglang.srt.planner import rejected
from sglang.srt.uneven_perf import (
    AttnCorePlan,
    PerfCostModel,
    PlanInputs,
    _attn_candidates,
    _attn_head_axis_is_pinned,
    _attn_token_candidates,
    _cand_label,
    _cand_token_vector,
    _cand_vectors,
    _mlp_candidates,
    _replication_axis_lines,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_INT8 = os.path.join(_CACHE, "Qwen3.6-27B-INT8-W8A8") if _CACHE else ""
_FP8 = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: The #475/#485 fixtures, unchanged.
_BUDGETS = [28447, 16320, 16320]
_GEMM_INT8 = [681.4, 187.6, 183.8]
_GEMM_FP8 = [563.1, 57.6, 60.8]
_MIN_LINK = 5.1
#: Budgets under which the base plan actually FUNDS a KV pool. The #475
#: fixture above does not (it is a compute-only fixture: every rank is
#: negative after weights + overhead), so a capacity assertion made on it
#: could never fail. Kept separate rather than replacing the fixture, because
#: the compute numbers must stay comparable with #475/#485.
_FUNDED_BUDGETS = (49152, 32768, 32768)


def _model(model_path, base_plan=None, budgets=None):
    plan = list(base_plan or _BUDGETS)
    budget = list(budgets or _BUDGETS)
    pi = PlanInputs(
        tp_size=3,
        model_path=model_path,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=budget,
        rank_tp_ratio=plan,
    )
    return PerfCostModel(pi, plan, budget)


def _base_token_vector(plan):
    g = math.gcd(*plan)
    return tuple(v // g for v in plan)


class TestWhatTheRuntimeCanActuallyDoToday(CustomTestCase):
    """The runtime verdict, asserted against the shipped predicates.

    Two DIFFERENT replication mechanisms live in ``distributed/utils``, they
    are gated differently, and #485 conflated them into one "grid" story.
    """

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_kv_pool_replication_is_not_gated_on_kv_heads_versus_ranks(self):
        """The mechanism that makes the token axis real TODAY.

        ``uneven_dcp_kv_replicated`` keys on (a DCP group wider than one) AND
        (a base plan installed). It never reads the kv-head count, so on a
        4-kv-head checkpoint at tp=3 -- where ``attn_kv_replicated`` is False
        -- the KV pool is STILL replicated-heads + token-sharded. That is why
        replicating the attention compute role costs no extra KV bytes here:
        the bytes are token-proportional already.
        """
        set_tp_partition_ratios(_BUDGETS)
        self.assertTrue(uneven_dcp_kv_replicated(3))
        # ... at a kv-head count that comfortably covers the ranks.
        self.assertFalse(attn_kv_replicated(3, 4))
        # ... and with no plan installed nothing engages, unchanged.
        set_tp_partition_ratios(None)
        self.assertFalse(uneven_dcp_kv_replicated(3))

    def test_projection_replication_stays_an_unbuilt_posten(self):
        """The mechanism that is NOT available, named rather than assumed.

        ``attn_kv_replicated`` is strictly ``kv < tp``; the ``<=`` flip is
        measured-rejected in its own docstring. And on this geometry the
        REPLICATED-KV q split is structurally unrepresentable anyway: the
        #116 alignment repair needs ``units % groups == 0`` and
        ``groups < n``, and 24 q / 4 kv over 3 ranks gives units=6,
        groups=4 -- neither holds, so the raw split is returned and the #105
        ragged kernel rejects it at the first forward.

        Generalizing THIS is a runtime rebuild (the #169 head-gather family),
        not a threshold flip, and this slice does not silently build it.
        """
        set_tp_partition_ratios(_BUDGETS)
        self.assertFalse(attn_kv_replicated(3, 4))  # this checkpoint
        self.assertFalse(attn_kv_replicated(3, 3))  # the rejected <= flip
        self.assertTrue(attn_kv_replicated(3, 2))  # the built kv < tp path
        units, groups, n = 6, 4, 3
        self.assertNotEqual(units % groups, 0)
        self.assertGreaterEqual(groups, n)
        # The repair is a no-op under those conditions: raw split returned.
        self.assertEqual(
            partition_units(units, [4, 1, 1], groups),
            partition_units(units, [4, 1, 1]),
        )

    def test_the_head_grid_has_a_floor_the_token_axis_does_not(self):
        """Why one axis is discrete and the other is not, in one comparison.

        ``partition_units`` forces every rank >= 1 whole unit, so 4 kv heads
        over 3 ranks can only ever be ``[2,1,1]`` no matter what weights it is
        asked for. The token owner rule (``cp_token_prefix``) has no such
        floor beyond "a positive integer per rank", so the same weights map
        straight through.
        """
        for weights in ([16, 1, 1], [8, 1, 1], [3, 1, 1], [1, 1, 16]):
            with self.subTest(weights=weights):
                self.assertEqual(
                    sorted(partition_units(4, weights), reverse=True), [2, 1, 1]
                )
        # ... and the token axis reproduces every one of them exactly.
        self.assertEqual(
            PerfCostModel.token_shard_fractions(None, [16, 1, 1]),
            [16 / 18, 1 / 18, 1 / 18],
        )


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8) and _FP8 and os.path.isdir(_FP8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-{INT8-W8A8,FP8} not present",
)
class TestTheHeadOnlyCandidateSpaceIsEmpty(CustomTestCase):
    """THE FALSIFIER, executed.

    If the #485 candidate space could move the attention family, this suite is
    pointless and the "grid-pinned" reading was merely a local result. It
    cannot: the space realizes ONE attention partition.
    """

    def test_every_485_attention_candidate_realizes_the_same_partition(self):
        for path, gemm in ((_INT8, _GEMM_INT8), (_FP8, _GEMM_FP8)):
            m = _model(path)
            cands = _attn_candidates(m, gemm, _BUDGETS)
            with self.subTest(path=os.path.basename(path)):
                self.assertTrue(cands, "the ladder must not be empty")
                realized = {
                    tuple(m._shard_fractions("attn", _BUDGETS, list(v)))
                    for v in [list(_BUDGETS)] + [list(c) for c in cands]
                }
                self.assertEqual(
                    len(realized),
                    1,
                    "the head axis is expected to be EMPTY on a 4-kv-head "
                    f"checkpoint at tp=3, realized: {realized}",
                )
                self.assertEqual(realized, {(0.5, 0.25, 0.25)})
                self.assertTrue(_attn_head_axis_is_pinned(m, _BUDGETS, cands))

    def test_the_pinned_predicate_reads_the_attention_shard_alone(self):
        """The conflation that produced the wrong verdict, guarded.

        ``_attn_partition_key`` pairs the attention partition with the GDN
        one, and the 16-unit GDN grid resolves the whole ladder. Keying
        "is the head axis empty" on that pair reports five distinct
        candidates for a space in which the attention partition never moves.
        """
        from sglang.srt.uneven_perf import _attn_partition_key

        m = _model(_INT8)
        cands = _attn_candidates(m, _GEMM_INT8, _BUDGETS)
        pair_keys = {_attn_partition_key(m, list(c)) for c in cands}
        self.assertGreater(len(pair_keys), 1)
        self.assertTrue(_attn_head_axis_is_pinned(m, _BUDGETS, cands))

    def test_the_token_axis_is_not_empty_on_the_same_rig(self):
        for path, gemm in ((_INT8, _GEMM_INT8), (_FP8, _GEMM_FP8)):
            m = _model(path)
            toks = _attn_token_candidates(m, gemm, _base_token_vector(_BUDGETS))
            with self.subTest(path=os.path.basename(path)):
                self.assertTrue(toks)
                fracs = {
                    tuple(m.token_shard_fractions(t))
                    for t in [list(_base_token_vector(_BUDGETS))]
                    + [list(t) for t in toks]
                }
                self.assertGreater(
                    len(fracs),
                    1,
                    "the replication axis must realize distinct splits where "
                    "the head axis realizes exactly one",
                )


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-INT8-W8A8 not present",
)
class TestTheCoreShareBracket(CustomTestCase):
    def test_no_core_plan_is_the_pre_492_arithmetic(self):
        """Byte identity of the default path. Not "close": identical floats.

        The pre-#492 call must execute the same operations in the same order,
        which is why ``_with_core`` passes the argument only when it exists
        rather than passing ``None`` through.
        """
        m = _model(_INT8)
        base_tok = _base_token_vector(_BUDGETS)
        for vec, attn in (([8, 1, 1], None), ([4, 1, 1], [3, 1, 1])):
            with self.subTest(vec=vec, attn=attn):
                a = m.prefill_time_model(vec, _GEMM_INT8, _MIN_LINK, None, attn)
                b = m.prefill_time_model(
                    vec, _GEMM_INT8, _MIN_LINK, None, attn, None, None
                )
                self.assertEqual(a, b)
                # share 0 is the same physics, and must also be the same
                # number: a zero-weight redistribution is the identity.
                c = m.prefill_time_model(
                    vec,
                    _GEMM_INT8,
                    _MIN_LINK,
                    None,
                    attn,
                    AttnCorePlan(base_tok, 0.0),
                    AttnCorePlan(base_tok, 0.0),
                )
                self.assertEqual(a, c)

    def test_the_core_term_conserves_the_family_mass(self):
        """The bracket varies the inter-rank RATIO only, never the mass --
        the same discipline ``_attn_lane_bracket`` applies to the rates. If it
        invented mass, a "gain" could come from the model rather than the
        layout."""
        m = _model(_INT8)
        ref = m.per_family_prefill_compute_times(_BUDGETS, _GEMM_INT8)

        # time x rate == that rank's share of the family mass, so the sum over
        # ranks is the family mass and must not move with the share.
        def _mass(table):
            return sum(table[r] * _GEMM_INT8[r] for r in range(3))

        for share in (0.0, 0.25, 0.5, 1.0):
            for tok in ((16, 1, 1), (1, 1, 16), (5, 4, 3)):
                with self.subTest(share=share, tok=tok):
                    fam = m.per_family_prefill_compute_times(
                        _BUDGETS,
                        _GEMM_INT8,
                        None,
                        None,
                        AttnCorePlan(tok, share),
                    )
                    self.assertAlmostEqual(
                        _mass(fam["attn"]) / _mass(ref["attn"]),
                        1.0,
                        places=12,
                    )
                    # Every OTHER family is untouched at every share.
                    for name in ref:
                        if name == "attn":
                            continue
                        self.assertEqual(fam[name], ref[name], name)

    def test_the_core_term_follows_the_token_vector(self):
        """At share 1 the attention family's per-rank split IS the token
        split, and at share 0 it is the head split. Both asserted against the
        arithmetic they claim, not against a magic number."""
        m = _model(_INT8)
        tok = (16, 1, 1)
        paced = m.per_family_prefill_compute_times(
            _BUDGETS, _GEMM_INT8, None, None, AttnCorePlan(tok, 1.0)
        )["attn"]
        share = m.token_shard_fractions(tok)
        ratios = [paced[r] * _GEMM_INT8[r] for r in range(3)]  # time x rate == mass
        total = sum(ratios)
        for r in range(3):
            self.assertAlmostEqual(ratios[r] / total, share[r], places=9)
        free = m.per_family_prefill_compute_times(
            _BUDGETS, _GEMM_INT8, None, None, AttnCorePlan(tok, 0.0)
        )["attn"]
        head = m._shard_fractions("attn", _BUDGETS)
        ratios = [free[r] * _GEMM_INT8[r] for r in range(3)]
        total = sum(ratios)
        for r in range(3):
            self.assertAlmostEqual(ratios[r] / total, head[r], places=9)

    def test_the_draft_attention_family_is_not_dragged_onto_the_token_axis(
        self,
    ):
        """The #108 spec cross-charge, enforced in code and not only in prose.

        ``draft_attn`` shares the ``"attn"`` SHARD, so a shard-keyed core term
        would silently move the draft onto the token axis -- the layout this
        fork refuses below the kv threshold (10-16 % acceptance).
        """
        m = _model(_INT8)
        self.assertEqual(m.families["draft_attn"].shard, "attn")
        base = m.per_family_prefill_compute_times(_BUDGETS, _GEMM_INT8, None, None)
        paced = m.per_family_prefill_compute_times(
            _BUDGETS, _GEMM_INT8, None, None, AttnCorePlan((16, 1, 1), 1.0)
        )
        self.assertNotEqual(paced["attn"], base["attn"])
        self.assertEqual(paced["draft_attn"], base["draft_attn"])
        self.assertEqual(paced["gdn"], base["gdn"])

    def test_a_detuned_token_vector_prices_worse(self):
        """The second falsifier: reverse the aligned token vector, same grid,
        same everything else. An objective that is not reading the token axis
        returns the identical number."""
        m = _model(_INT8)
        base_tok = _base_token_vector(_BUDGETS)
        base_core = AttnCorePlan(base_tok, 1.0)
        ref = m.prefill_time_model(
            _BUDGETS, _GEMM_INT8, _MIN_LINK, None, None, base_core, base_core
        )

        def gain(tok):
            return (
                ref
                / m.prefill_time_model(
                    [4, 1, 1],
                    _GEMM_INT8,
                    _MIN_LINK,
                    None,
                    [3, 1, 1],
                    AttnCorePlan(tuple(tok), 1.0),
                    base_core,
                )
                - 1.0
            )

        aligned = gain([4, 1, 1])
        detuned = gain([1, 1, 4])
        self.assertGreater(aligned, detuned + 1e-9)

    def test_the_crossover_is_pure_geometry(self):
        """No fitted constant, no probe, no rig. Re-derived here from the
        config numbers so a refactor that quietly folds a calibration scalar
        into it fails."""
        m = _model(_INT8)
        expect = m.attn_proj_params_per_layer / (2.0 * m.q_heads * m.head_dim)
        self.assertAlmostEqual(m.attn_core_crossover_tokens(), expect, places=6)
        self.assertGreater(m.attn_core_crossover_tokens(), 0.0)

    def test_a_pinned_token_vector_never_beats_the_matched_one_on_context(
        self,
    ):
        """The axis's real price. A token vector solved for the barrier is not
        the one that maximises capacity, and the fundability gate must see the
        weaker number rather than the optimistic one."""
        # Deliberately NOT the #475 budget fixture: those budgets fund no pool
        # at all (every rank negative after weights), so the gate would be
        # vacuous and the assertion could not fail.
        m = _model(
            _INT8,
            base_plan=list(_FUNDED_BUDGETS),
            budgets=list(_FUNDED_BUDGETS),
        )
        matched = m.predict_capacity(list(_FUNDED_BUDGETS))
        self.assertTrue(matched["feasible"])
        self.assertGreater(matched["ctx"], 0)
        for tok in ([16, 1, 1], [1, 1, 16], [4, 1, 1], [10, 11, 11]):
            with self.subTest(tok=tok):
                pinned = m.predict_capacity(list(_FUNDED_BUDGETS), None, tok)
                self.assertLessEqual(pinned["ctx"], matched["ctx"] + 1e-6)
                self.assertEqual(pinned["token_vector"], tok)
        # ... and the matched vector itself is the maximiser, so the
        # inequality above is tight rather than trivially satisfied. Compared
        # against the analytic optimum with a rounding allowance: ``ctx`` for
        # the matched case is ``min(sum(P), 64*min(P))``, while a PINNED
        # vector is priced through the integer owner rule, and the 64-unit
        # integerization costs a couple of percent.
        best = m.predict_capacity(list(_FUNDED_BUDGETS), None, matched["token_vector"])[
            "ctx"
        ]
        self.assertGreater(best, matched["ctx"] * 0.95)
        for tok in ([16, 1, 1], [1, 1, 16], [4, 1, 1]):
            self.assertLess(
                m.predict_capacity(list(_FUNDED_BUDGETS), None, tok)["ctx"],
                best,
                msg=f"{tok} must fund less than the matched vector",
            )


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-INT8-W8A8 not present",
)
class TestTheReportedAxis(CustomTestCase):
    def test_the_report_names_both_axes_and_the_refutation(self):
        m = _model(_INT8)
        mlp = [list(c) for c in _mlp_candidates(m, _GEMM_INT8, _BUDGETS)]
        attn = [list(c) for c in _attn_candidates(m, _GEMM_INT8, _BUDGETS)]
        text = "\n".join(
            _replication_axis_lines(
                m,
                _BUDGETS,
                mlp + [tuple(_BUDGETS)],
                attn,
                _GEMM_INT8,
                _GEMM_INT8,
                None,
                _MIN_LINK,
            )
        )
        for needle in (
            "replication axis (#492)",
            "HEAD axis",
            "REPLICATION+TOKEN axis",
            "uneven_dcp_kv_replicated",
            "core crossover (#492)",
            "CORE-FREE",
            "CORE-PACED",
            "spec cross-charge (#108",
            "draft_kv_dcp_below_kv_threshold",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        self.assertTrue("CORE-INVARIANT" in text or "CORE-SENSITIVE" in text)

    def test_the_context_floor_binds_the_axis_and_is_reported(self):
        """The axis must never advertise a layout the boot would refuse.

        The token vector's capacity term is not a rounding detail: the
        weighted owner rule funds ``min_r(P_r / v_r)`` blocks, so pushing the
        context onto the fast rank throws the slow ranks' pools away. Without
        this gate the report names a +0.3-point compute win that funds an
        order of magnitude less context.
        """
        m = _model(
            _INT8, base_plan=list(_FUNDED_BUDGETS), budgets=list(_FUNDED_BUDGETS)
        )
        mlp = [list(c) for c in _mlp_candidates(m, _GEMM_INT8, _FUNDED_BUDGETS)]
        attn = [list(c) for c in _attn_candidates(m, _GEMM_INT8, _FUNDED_BUDGETS)]
        base_ctx = m.predict_capacity(list(_FUNDED_BUDGETS))["ctx"]

        def _run(floor):
            return "\n".join(
                _replication_axis_lines(
                    m,
                    list(_FUNDED_BUDGETS),
                    mlp + [tuple(_FUNDED_BUDGETS)],
                    attn,
                    _GEMM_INT8,
                    _GEMM_INT8,
                    None,
                    _MIN_LINK,
                    floor,
                    0.0,
                )
            )

        tight = _run(base_ctx)
        self.assertIn("capacity price (#492)", tight)
        self.assertIn("REJECTED by the context floor", tight)
        # With no floor at all the axis DOES propose a token vector, which is
        # what makes the gate above a real filter rather than dead code.
        loose = _run(None)
        self.assertIn("KV tokens", loose)
        self.assertNotIn("KV tokens", tight.split("capacity price")[0])

    def test_a_symmetric_lane_proposes_no_token_vector(self):
        """Generality: the axis is a property of LANE SPREAD, not of the
        machinery. Equal cards must produce no candidate at all rather than a
        spurious one."""
        m = _model(_INT8)
        self.assertEqual(
            _attn_token_candidates(m, [100.0, 100.0, 100.0], (1, 1, 1)), []
        )

    def test_the_candidate_label_carries_the_token_vector(self):
        self.assertEqual(_cand_vectors((8, 1, 1)), ([8, 1, 1], None))
        self.assertIsNone(_cand_token_vector((8, 1, 1)))
        self.assertIsNone(_cand_token_vector(((8, 1, 1), (3, 1, 1))))
        entry = ((8, 1, 1), (3, 1, 1), (4, 1, 1))
        self.assertEqual(_cand_vectors(entry), ([8, 1, 1], [3, 1, 1]))
        self.assertEqual(_cand_token_vector(entry), [4, 1, 1])
        self.assertEqual(_cand_label(entry), "8,1,1 + attn/GDN 3,1,1 + KV tokens 4,1,1")
        # The #485 shapes are unchanged.
        self.assertEqual(_cand_label((8, 1, 1)), "8,1,1")
        self.assertEqual(_cand_label(((8, 1, 1), (3, 1, 1))), "8,1,1 + attn/GDN 3,1,1")


class TestTheSpecCrossChargeIsRegistered(CustomTestCase):
    def test_the_draft_layout_flip_is_in_the_rejected_register(self):
        """The replication axis has a measured cost in the speculative path.
        It is carried as a named refusal with its own register row, not
        ignored and not fabricated into a time-shaped malus."""
        entry = rejected.by_key("draft_kv_dcp_below_kv_threshold")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.level, rejected.NOT_DEFAULT)
        self.assertIn("10-16", entry.verdict)
        self.assertIn("1.05", entry.verdict)
        self.assertIn("draft-kv-layout", entry.unlock)
        self.assertIn("TASK_108", entry.evidence)
        self.assertIn("uneven-dcp", entry.tags)


if __name__ == "__main__":
    unittest.main()
