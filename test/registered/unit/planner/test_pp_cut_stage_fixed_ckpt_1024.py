# SPDX-License-Identifier: Apache-2.0
"""PPCUT-INT4AR 24.10.: the stage-fixed record moved onto the booted checkpoint.

WHAT WAS WRONG.  ``P_PP_STAGE_FIXED_MIB`` (qwen27b, PP-POSTEN 29.09.) is a
measured role vector -- "weights + runtime state" minus the per-layer half --
fitted on boots that ran ``Qwen3.8-27B-INT8-gdncov-vocabembed``, whose input
embedding is I8 (1213.0 MiB).  The int4ar checkpoint carries a BF16 embedding
(2425.0 MiB).  The per-layer half is re-priced from every boot's own headers;
the non-layer half was not, so the pool model under-charged the first stage by
the embedding delta.

MEASURED, boot dkr27bint4arrc4bar1int4ar10241254 (rc42 c07b574410, cut 36,28):
the launcher's PP-POOL-JOIN priced ``priced_per_rank=301650,148747``; group P
realised ``KV pool sizing ... max_total_num_tokens=226647`` on PP0 and 146025 on
PP1.  PP0 is over-priced by 75003 tokens x 18432 B = 1318 MiB -- the OOM
direction -- and 1212 MiB of it is the embedding delta.  PP1 (64 MiB) is not.

WHAT IT DOES NOT CHANGE.  The shipped int4ar cut: PP1 binds on 36,28, so the
world pool stays 148747 and the makespan arm still ships 36,28 under the
cap+chunk floor 135168.  What it changes is every cut on which PP0 binds --
the kv-floor/maxkv row (41,23 priced 212917, really ~153.7k) and any floor
above ~150k -- which were admitted on a pool the card does not hold.

The fixture is the int4ar boot's own launcher inputs (launcher.log lines 89-91:
free 14488/14488, 187.7 MiB/layer, stage_fixed 1418.4/4960.6, mamba 1.5588 x 24,
draft cell 10240 B on the last stage) and is self-verifying against the
PP-POOL-JOIN figure the launcher printed.

Hermetic: no GPU, no NVML, no checkpoint, no server, no launcher run.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.planner.pp_cut import (
    LAYER_FAMILY_ATTENTION,
    LAYER_FAMILY_LINEAR,
    MIB,
    PhasePoolModel,
    attention_counts,
    pp_phase_pool,
    stage_pp_capacities,
)
from flliper.srt.planner.pp_cut_launch import (
    PPCutRefused,
    solve_launch_cut,
    stage_fixed_checkpoint_delta,
)
from flliper.test.test_utils import CustomTestCase

# Qwen3.8-27B geometry: 64 layers, full attention every 4th (index 3, 7, ...).
FAMILIES = tuple(
    LAYER_FAMILY_ATTENTION if i % 4 == 3 else LAYER_FAMILY_LINEAR for i in range(64)
)
RECORD_STAGE_FIXED = (1418.4, 4960.6)  # HW-DERIVE 3 -> 2 (role) of the record
REF_EMBED_MIB = 1212.97  # vocabembed: I8 1212.5 + scale 0.47
REF_HEAD_MIB = 2425.0
INT4AR_EMBED_MIB = 2425.0  # BF16 [248320, 5120]
INT4AR_HEAD_MIB = 2425.0
NVFP4_HEAD_MIB = 606.2  # lm_head.weight U8 [248320, 2560] as checkpoint_weight_terms counts it (scales not)

# Launcher-printed and runtime-realised numbers of the boot (verbatim).
PRICED_36_28 = (301650, 148747)
REALISED_PP0_36_28 = 226647
FLOOR = 135168  # cap+chunk = 131072 + 4096
CELL_PP0_36_28 = 9 * 2048


def _model(stage_fixed, free=(14488.0, 14488.0)):
    return PhasePoolModel(
        free_mib=tuple(free),
        weight_mib_per_layer=187.7,
        kv_mib_per_token_per_attn_layer=2048.0 / MIB,
        arming_floor_mib=(1229.0, 1229.0),
        stage_fixed_mib=tuple(stage_fixed),
        corridor_holdback_mib=0.0,
        mamba_mib_per_linear_layer_per_slot=1.5588,
        mamba_slots=24,
        extra_cell_bytes_by_stage=(0, 10240),
        page_size=1,
        zero_posts_acknowledged=(
            "mamba pre-capture reserve",
            "speculative intermediate state",
            "GGUF dequant scratch",
            "prefill activation reserve",
        ),
    )


def _corrected():
    out, _line = stage_fixed_checkpoint_delta(
        RECORD_STAGE_FIXED,
        ref_label="vocabembed",
        ref_embedding_mib=REF_EMBED_MIB,
        ref_lm_head_mib=REF_HEAD_MIB,
        embedding_mib=INT4AR_EMBED_MIB,
        lm_head_mib=INT4AR_HEAD_MIB,
    )
    return out


def _caps(cut, stage_fixed):
    return stage_pp_capacities(cut, attention_counts(FAMILIES, cut), _model(stage_fixed))


def _solve(stage_fixed, *, objective, pool_floor, cap_tokens=131072):
    return solve_launch_cut(
        layer_families=FAMILIES,
        incumbent_layers=(32, 32),
        measured_ms_per_layer=(35.16, 35.16),
        measured_provenance="test: launcher MEASURED_MS_PER_LAYER class-max",
        # Unknown names: the card-rate library cannot answer, so the measured
        # list is used -- hermetic whatever library this box carries.
        card_names=("TEST-CARD-A", "TEST-CARD-B"),
        pool_model=_model(stage_fixed),
        cap_tokens=cap_tokens,
        objective=objective,
        pool_floor=pool_floor,
        enumerate_gapped=False,
    )


class TheDeltaFunction(CustomTestCase):
    def test_larger_embedding_is_added_to_the_first_stage_only(self):
        out = _corrected()
        self.assertAlmostEqual(out[0], 1418.4 + (2425.0 - 1212.97), places=6)
        self.assertEqual(out[1], 4960.6)

    def test_smaller_head_is_not_credited_and_is_named(self):
        out, line = stage_fixed_checkpoint_delta(
            RECORD_STAGE_FIXED,
            ref_label="vocabembed",
            ref_embedding_mib=REF_EMBED_MIB,
            ref_lm_head_mib=REF_HEAD_MIB,
            embedding_mib=INT4AR_EMBED_MIB,
            lm_head_mib=NVFP4_HEAD_MIB,
        )
        self.assertEqual(out[1], 4960.6, "a smaller lm_head must never lower the post")
        self.assertIn("NOT credited", line)
        self.assertIn("lm_head -1818.8 MiB", line)

    def test_absent_head_is_named_as_absent(self):
        _, line = stage_fixed_checkpoint_delta(
            RECORD_STAGE_FIXED,
            ref_label="vocabembed",
            ref_embedding_mib=REF_EMBED_MIB,
            ref_lm_head_mib=REF_HEAD_MIB,
            embedding_mib=INT4AR_EMBED_MIB,
            lm_head_mib=0.0,
        )
        self.assertIn("lm_head absent from the headers", line)

    def test_reference_checkpoint_is_a_no_op(self):
        out, line = stage_fixed_checkpoint_delta(
            RECORD_STAGE_FIXED,
            ref_label="vocabembed",
            ref_embedding_mib=REF_EMBED_MIB,
            ref_lm_head_mib=REF_HEAD_MIB,
            embedding_mib=REF_EMBED_MIB,
            lm_head_mib=REF_HEAD_MIB,
        )
        self.assertEqual(out, RECORD_STAGE_FIXED)
        self.assertNotIn("NOT credited", line)

    def test_single_stage_carries_both_payloads(self):
        out, _ = stage_fixed_checkpoint_delta(
            (1000.0,),
            ref_label="r",
            ref_embedding_mib=100.0,
            ref_lm_head_mib=100.0,
            embedding_mib=150.0,
            lm_head_mib=130.0,
        )
        self.assertAlmostEqual(out[0], 1080.0)

    def test_empty_vector_is_returned_as_is(self):
        out, line = stage_fixed_checkpoint_delta(
            (),
            ref_label="r",
            ref_embedding_mib=1.0,
            ref_lm_head_mib=1.0,
            embedding_mib=9.0,
            lm_head_mib=9.0,
        )
        self.assertEqual(out, ())
        self.assertIn("nothing to move", line)


class TheInt4arBootReplay(CustomTestCase):
    def test_fixture_reproduces_the_launcher_pricing(self):
        caps = _caps((36, 28), RECORD_STAGE_FIXED)
        # 187.7 is the launcher's printed (rounded) mean; 15 tokens of slack.
        for got, want in zip(caps, PRICED_36_28):
            self.assertLess(abs(got - want), 15, (caps, PRICED_36_28))

    def test_record_overprices_pp0_by_more_than_a_gib(self):
        over_mib = (_caps((36, 28), RECORD_STAGE_FIXED)[0] - REALISED_PP0_36_28) * CELL_PP0_36_28 / MIB
        self.assertGreater(over_mib, 1200.0)

    def test_corrected_pp0_is_within_150_mib_of_metal(self):
        over_mib = (_caps((36, 28), _corrected())[0] - REALISED_PP0_36_28) * CELL_PP0_36_28 / MIB
        self.assertLess(abs(over_mib), 150.0, over_mib)

    def test_world_pool_of_the_shipped_cut_is_unchanged(self):
        before = pp_phase_pool((36, 28), (9, 7), _model(RECORD_STAGE_FIXED))
        after = pp_phase_pool((36, 28), (9, 7), _model(_corrected()))
        self.assertEqual(before, after)


class TheOomDirection(CustomTestCase):
    """MUTANT, OOM direction: a cut that only clears on the record's PP0 price."""

    def test_floor_200k_was_admitted_and_is_now_refused(self):
        before = _solve(RECORD_STAGE_FIXED, objective="makespan", pool_floor=200000)
        self.assertGreaterEqual(before.chosen.pool_tokens, 200000)
        self.assertGreaterEqual(before.chosen.layers[0], 40)
        with self.assertRaises(PPCutRefused) as ctx:
            _solve(_corrected(), objective="makespan", pool_floor=200000)
        self.assertIn("W40", str(ctx.exception))

    def test_maxkv_no_longer_ships_the_pp0_overpriced_cut(self):
        before = _solve(RECORD_STAGE_FIXED, objective="maxkv", pool_floor=FLOOR)
        after = _solve(_corrected(), objective="maxkv", pool_floor=FLOOR)
        self.assertEqual(tuple(before.chosen.layers), (41, 23))
        self.assertNotEqual(tuple(after.chosen.layers), (41, 23))
        # whatever ships now, its PP0 is priced on the corrected post
        caps = _caps(tuple(after.chosen.layers), _corrected())
        self.assertEqual(min(caps), after.chosen.pool_tokens)
        self.assertLess(after.chosen.pool_tokens, before.chosen.pool_tokens)


class TheMakespanDirection(CustomTestCase):
    """MUTANT, makespan direction: the floor narrows, the fastest survivor ships."""

    def test_cap_chunk_floor_ships_36_28_before_and_after(self):
        for sf in (RECORD_STAGE_FIXED, _corrected()):
            d = _solve(sf, objective="makespan", pool_floor=FLOOR)
            self.assertEqual(tuple(d.chosen.layers), (36, 28))
            self.assertEqual(tuple(d.chosen.attn), (9, 7))

    def test_no_faster_cut_clears_the_floor(self):
        d = _solve(_corrected(), objective="makespan", pool_floor=FLOOR)
        faster = [
            c for c in d.ranked
            if c.total_ms < d.chosen.total_ms and c.pool_tokens >= FLOOR
        ]
        self.assertEqual(faster, [], [c.fmt() for c in faster])

    def test_without_the_floor_the_even_cut_is_fastest(self):
        # a 96k request cap (floor 98304 + 4096) is what 32,32 would need
        d = _solve(_corrected(), objective="makespan", pool_floor=102400, cap_tokens=98304)
        self.assertEqual(tuple(d.chosen.layers), (32, 32))
        slow = _solve(_corrected(), objective="makespan", pool_floor=FLOOR)
        # the cut-time ratio the solver prices: 36 vs 32 layers on equal cards
        self.assertAlmostEqual(slow.chosen.makespan_ms / d.chosen.makespan_ms, 36.0 / 32.0, places=2)


class The32x32Fit(CustomTestCase):
    """Task 3: 32/32 against the floor, at the launcher's budget and at a measured one."""

    def test_32_32_is_below_the_floor_at_the_launcher_budget(self):
        self.assertLess(min(_caps((32, 32), _corrected())), FLOOR)

    def test_32_32_clears_with_the_measured_d_dormant_residue(self):
        # dormant_other priced 2652 (CONSTANT, xsn14 exchange form) against a
        # measured D sleep residue of 1258 MiB (+64 slack) on both cards.
        free = 14488.0 + (2652.0 - (1258.0 + 64.0))
        caps = stage_pp_capacities((32, 32), (8, 8), _model(_corrected(), free=(free, free)))
        self.assertGreaterEqual(min(caps), FLOOR)


if __name__ == "__main__":
    unittest.main()
