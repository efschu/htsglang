# SPDX-License-Identifier: Apache-2.0
"""The weight-byte exchange PLAN builder (#1273, spec §6/S1).

WHAT THIS SLICE HAS TO GET RIGHT, and why a test is the only place it can be
checked before metal: the plan says which bytes of which rank's storage become
which bytes of which other rank's storage. Every failure in this family is
SILENT -- a wrong device offset writes ``k`` over ``q`` on ranks 1 and 2 and
produces plausible garbage, no error, no crash (spec §8 R5).

THE TWO READINGS THAT ARE WRONG AND LOOK RIGHT:

* **checkpoint offsets for a fused parameter.** SECTION 1af's IDENTICAL-SUBBLOCK
  probe is CHECKPOINT-space: it lists ``in_proj_qkv`` as three groups at
  ``+0 / +10485760 / +20971520``. On the DEVICE the parameter is one fused
  ``MergedColumnParallelLinear(output_sizes=[k, k, v, v])``
  (``models/qwen3_5.py:541-553``) whose per-rank sub-block offsets are the
  prefix sum of THIS RANK'S sizes (``layers/linear.py:1088-1101``), not of the
  full ones. What coincides on rank 0 is the GLOBAL start, not the device row --
  which is why a single-rank check does not see the defect.
* **logical shape instead of storage.** ``.t()`` in the int8 quant path is a
  view: the logical shape is ``[K, N_local]`` over storage ``[N_local, K]``. A
  plan built from ``param.shape`` is transposed-wrong for every column-parallel
  class.

Both are pinned by a test that FAILS on the wrong reading, not by a test that
passes on the right one.

CPU only, real modules on the meta device: no GPU, no distributed group, no
CUDA call anywhere in this file.
"""

import os
import random
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    partition_sizes,
    set_tp_partition_ratios,
)
from sglang.srt.managers.weg2_memory_saver import chunk_tag_cards, weights_family_tags
from sglang.srt.weg2.weight_exchange import (
    COALESCE_FLOOR_BYTES,
    COLS,
    FLAT,
    REPLICATED,
    ROWS,
    STRIDED2D,
    ZEROFILL,
    GroupLayout,
    ParamGeom,
    StorageGeom,
    Weg2XchgPlanDisagree,
    Weg2XchgSourceMissing,
    XchgDesc,
    build_plan,
    coalesce,
    derive_waves,
    device_block_offsets,
    emit_plan_line,
    piece_histogram,
    shard_offsets,
    unmergeable_below_floor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Fixtures.  Every number carries its provenance; none is invented here.
# --------------------------------------------------------------------------

#: Qwen3.5-27B hidden size, spec §2.2 (``o_proj``: 5120 runs).
HIDDEN = 5120
#: The attention input dimension of ``o_proj``, spec §2.2 (spitch 6144).
ATTN_IN_FULL = 6144
#: The dense-MLP intermediate, spec §2.2 (``down_proj`` spitch 17408).
MLP_IN_FULL = 17408
#: The MLP quant-block unit family: 136 units of 128 elements = 17408
#: (``layers/linear.py:282 _quant_block_aligned_units``; spec §6/S1 cites it as
#: ``qwen2_moe.py:222``, which is where it used to live -- DRIFT, see the
#: record).
MLP_UNITS = 136
#: The three-rank vector that produces the spec's own per-rank MLP widths
#: [7808, 4864, 4736] -- 61 / 38 / 37 of the 136 units.
MLP_RATIO = [61, 38, 37]
MLP_WIDTHS_OK = [7808, 4864, 4736]
#: THE SEED (spec §6/S1).  The pre-correction MLP vector.  It sums to 17408 and
#: therefore survives every eyeball check, but 7792 / 128 = 60.875 is not a
#: whole quant-block unit, so no rank's storage can ever be this wide.
MLP_WIDTHS_WRONG = [7792, 4816, 4800]

#: GDN ``in_proj_qkvz``: ``output_sizes=[2048, 2048, 6144, 6144]``
#: (``models/qwen3_5.py:543``, spec §2.2).
QKVZ_OUTPUT_SIZES = [2048, 2048, 6144, 6144]
#: The GDN head-unit family (``gdn_tp_units``): key_dim 2048 in 128-element
#: k-head units = 16 units, split 7 / 5 / 4 by this rig's vector.
QKVZ_UNITS = 16
QKVZ_RATIO = [7, 5, 4]
#: SECTION 1af's CHECKPOINT sub-block offsets for ``in_proj_qkv`` -- the wrong
#: reading this file exists to refuse (spec §2.4 rule 1).
CKPT_QKV_OFFSETS = [0, 10485760, 20971520]

#: Vocabulary, spec §2.2: 248320 real rows; D pads to 64*tp = 192 -> 248448, so
#: D rank 2 owns 128 rows that exist on no card and in no checkpoint.
VOCAB_REAL = 248320
VOCAB_PADDED_D = 248448

#: Boot weg2sb4's P cut: ``--pp-stage-ratio 32,18,14``, 8 layers per chunk,
#: 8 chunks (spec §1.2).
SB4_STAGE_LAYERS = [32, 18, 14]
SB4_LAYERS_PER_CHUNK = 8
SB4_CHUNKS = 8
#: The wave table the spec derives from that cut (spec §1.2).
SB4_WAVES = [
    ["weights_0", "weights_4", "weights_6"],
    ["weights_1", "weights_5", "weights_7"],
    ["weights_2", "weights_3", "weights"],
]

CARDS = (0, 1, 2)


def _p_layout():
    """Group P: ``--tp-size 1 --pp-size 3``, so a rank holds WHOLE tensors --
    the ones of its own layer band.  Global ranks 0..2."""
    return GroupLayout(name="P", cards=CARDS, tp_size=1, ratios=None, base=0)


def _d_layout(ratios):
    """Group D: TP=3 over the same three cards, rank n on cards[n].  Global
    ranks 3..5, so the 6x6 matrix is indexed the same way in both directions."""
    return GroupLayout(
        name="D",
        cards=CARDS,
        tp_size=3,
        ratios=list(ratios) if ratios else None,
        base=3,
    )


def _mlp_down_geom(**kw):
    """``down_proj``: row-parallel, so the SHARDED axis is the storage COLUMN
    axis and the primitive is 2-D (spec §2.2)."""
    return ParamGeom(
        name="model.layers.0.mlp.down_proj.weight",
        tag="weights_0",
        shard_axis=COLS,
        rows_full=HIDDEN,
        cols_full=MLP_IN_FULL,
        itemsize=1,
        units=MLP_UNITS,
        stage=0,
        **kw,
    )


def _qkvz_geom(**kw):
    """``in_proj_qkvz``: fused column-parallel, FOUR device sub-blocks."""
    return ParamGeom(
        name="model.layers.0.linear_attn.in_proj_qkvz.weight",
        tag="weights_0",
        shard_axis=ROWS,
        rows_full=sum(QKVZ_OUTPUT_SIZES),
        cols_full=HIDDEN,
        itemsize=1,
        blocks=tuple(QKVZ_OUTPUT_SIZES),
        units=QKVZ_UNITS,
        stage=0,
        **kw,
    )


# --------------------------------------------------------------------------


class TestDeviceSubBlockLaw(CustomTestCase):
    """The fused classes, on the DEVICE and not in the checkpoint."""

    def test_in_proj_qkvz_has_four_device_blocks_not_three(self):
        per_rank = device_block_offsets(
            QKVZ_OUTPUT_SIZES, QKVZ_RATIO, 3, units=QKVZ_UNITS
        )
        self.assertEqual([len(b) for b in per_rank], [4, 4, 4])

        widths = [
            [partition_sizes(sz, QKVZ_RATIO, QKVZ_UNITS)[r] for sz in QKVZ_OUTPUT_SIZES]
            for r in range(3)
        ]
        for r in range(3):
            with self.subTest(rank=r):
                g_k, g_v = widths[r][0], widths[r][2]
                # Spec §2.2: device rows 0 / g_k / 2*g_k / 2*g_k + g_v, with
                # THIS rank's g_k and g_v.
                self.assertEqual(
                    [blk.dev_row for blk in per_rank[r]],
                    [0, g_k, 2 * g_k, 2 * g_k + g_v],
                )
                self.assertEqual([blk.size for blk in per_rank[r]], widths[r])
                self.assertEqual(
                    sum(widths[r]), per_rank[r][-1].dev_row + widths[r][-1]
                )

        # THE CAN-FAIL.  The checkpoint reading puts every rank's block b at the
        # SAME offset, so block 1 (`k`) would land where the device keeps the
        # tail of block 0 (`q`) -- plausible garbage, no error.
        ckpt_rows = [off // HIDDEN for off in CKPT_QKV_OFFSETS]
        self.assertEqual(ckpt_rows, [0, 2048, 4096])
        for r in range(3):
            with self.subTest(rank=r, reading="checkpoint"):
                self.assertNotEqual([blk.dev_row for blk in per_rank[r]][:3], ckpt_rows)
        # CORRECTED BY THIS TEST, 2026-09-08: the two readings do NOT coincide
        # on rank 0.  The device rows are the prefix of that rank's OWN sizes
        # ([0, 896, 1792] here), so they differ from the checkpoint's on EVERY
        # rank beyond block 0.  What coincides on rank 0 is the GLOBAL start --
        # and that is why reading the probe as device-space evidence survives a
        # single-rank check.
        self.assertEqual([blk.global_start for blk in per_rank[0]][:3], ckpt_rows)

    def test_global_starts_are_the_checkpoint_prefix_the_device_rows_are_not(self):
        """The two coordinate systems a descriptor has to hold apart."""
        per_rank = device_block_offsets(
            QKVZ_OUTPUT_SIZES, QKVZ_RATIO, 3, units=QKVZ_UNITS
        )
        full_prefix = [0, 2048, 4096, 10240]
        for b, base in enumerate(full_prefix):
            with self.subTest(block=b):
                self.assertEqual(per_rank[0][b].global_start, base)
                self.assertEqual(
                    [per_rank[r][b].global_start for r in range(3)],
                    [
                        base
                        + sum(
                            partition_sizes(
                                QKVZ_OUTPUT_SIZES[b], QKVZ_RATIO, QKVZ_UNITS
                            )[:r]
                        )
                        for r in range(3)
                    ],
                )


class TestStorageSpace(CustomTestCase):
    """Offsets come from ``stride()``/``element_size()``, never from shape."""

    def test_offsets_are_storage_space_not_logical(self):
        # An int8 column-parallel weight as the quant path leaves it: storage
        # [N_local, K], logical shape [K, N_local] through a `.t()` view.
        storage = torch.empty((4864, HIDDEN), dtype=torch.int8, device="meta")
        view = storage.t()
        self.assertEqual(tuple(view.shape), (HIDDEN, 4864))

        geom = StorageGeom.of(view)
        self.assertEqual((geom.rows, geom.cols, geom.pitch), (4864, HIDDEN, HIDDEN))
        self.assertEqual(geom.itemsize, 1)
        self.assertEqual(geom.nbytes, 4864 * HIDDEN)
        # It is the SAME storage as the untransposed tensor -- the property
        # that makes the derivation `.t()`-blind.
        self.assertEqual(geom, StorageGeom.of(storage))

        # THE CAN-FAIL: the logical reading swaps rows and pitch, so every row
        # offset it produces is wrong by a factor of 5120/4864.
        logical = (view.shape[0], view.shape[1], view.shape[1])
        self.assertNotEqual((geom.rows, geom.cols, geom.pitch), logical)

        # bf16 must not be read as int8: the multiply is by element_size().
        wide = StorageGeom.of(
            torch.empty((128, HIDDEN), dtype=torch.bfloat16, device="meta")
        )
        self.assertEqual(wide.itemsize, 2)
        self.assertEqual(wide.nbytes, 128 * HIDDEN * 2)

    def test_storage_geom_refuses_a_layout_it_cannot_express(self):
        """A non-unit innermost stride is not a pitch, and guessing one is how
        a plan becomes silently wrong."""
        base = torch.empty((64, 64), dtype=torch.int8, device="meta")
        with self.assertRaises(Weg2XchgPlanDisagree) as cm:
            StorageGeom.of(base[:, ::2])
        self.assertIn("W52 Weg2XchgPlanDisagree", str(cm.exception))


class TestTiling(CustomTestCase):
    """Every destination byte has exactly one source."""

    def test_every_destination_byte_has_exactly_one_source(self):
        src, dst = _p_layout(), _d_layout(MLP_RATIO)

        # The correct vector: the plan tiles every destination tensor exactly.
        plan = build_plan([_mlp_down_geom()], src, dst, waves=[["weights_0"]])
        covered = {}
        for d in plan.descs:
            covered[d.dst_rank] = covered.get(d.dst_rank, 0) + d.nbytes
        self.assertEqual(
            [covered[r] for r in range(3)], [HIDDEN * w for w in MLP_WIDTHS_OK]
        )
        self.assertEqual(sum(covered.values()), HIDDEN * MLP_IN_FULL)

        # THE SEED (spec §6/S1).  The pre-correction vector sums to 17408, so it
        # looks like a partition; it is not one the 136-unit family can produce.
        self.assertEqual(sum(MLP_WIDTHS_WRONG), MLP_IN_FULL)
        self.assertNotEqual(
            MLP_WIDTHS_WRONG, partition_sizes(MLP_IN_FULL, MLP_RATIO, MLP_UNITS)
        )
        self.assertNotEqual(MLP_WIDTHS_WRONG[0] % 128, 0)

        # Seeded as the PLAN's widths against the hardware's real extents: rank
        # 0's last 16 columns per row have no source at all.
        with self.assertRaises(Weg2XchgSourceMissing) as cm:
            build_plan(
                [
                    _mlp_down_geom(
                        dst_widths=MLP_WIDTHS_WRONG, dst_extents=MLP_WIDTHS_OK
                    )
                ],
                src,
                dst,
                waves=[["weights_0"]],
            )
        message = str(cm.exception)
        self.assertIn("W58 Weg2XchgSourceMissing", message)
        self.assertIn("down_proj", message)
        self.assertIn("dst_rank=0", message)
        self.assertIn("[7792, 7808)", message)

        # And the mirror: a plan built from the CORRECT vector against storage
        # that is the wrong width writes past the destination's own arena.
        with self.assertRaises(Weg2XchgPlanDisagree) as cm:
            build_plan(
                [_mlp_down_geom(dst_extents=MLP_WIDTHS_WRONG)],
                src,
                dst,
                waves=[["weights_0"]],
            )
        message = str(cm.exception)
        self.assertIn("W52 Weg2XchgPlanDisagree", message)
        self.assertIn("writes 7808 units", message)
        self.assertIn("holds 7792", message)

    def test_a_gap_in_the_destination_is_w58_not_a_silent_short_write(self):
        """A destination range with no source and no DECLARED pad is the
        draft/MTP class leaking back into the weights family (spec §7 W58)."""
        src, dst = _p_layout(), _d_layout(MLP_RATIO)
        holed = _mlp_down_geom(src_extent=MLP_IN_FULL - 128)
        with self.assertRaises(Weg2XchgSourceMissing) as cm:
            build_plan([holed], src, dst, waves=[["weights_0"]])
        self.assertIn("W58 Weg2XchgSourceMissing", str(cm.exception))
        self.assertIn("down_proj", str(cm.exception))

    def test_a_source_picked_twice_would_be_an_overlap_and_is_prevented(self):
        """The other half of 'exactly one'.  Overlap cannot be REACHED through
        this API -- both sides' blocks are prefix sums, and a destination unit
        with several candidate sources is resolved to ONE, co-located first --
        so what is pinned here is the resolution, and the overlap branch of the
        tiling check stays defensive for S3/S6 (see the record: UNPROVEN)."""
        src = GroupLayout(name="P", cards=CARDS, tp_size=1, ratios=None, base=0)
        dst = _d_layout(None)
        # stage=None: EVERY P rank holds the whole tensor (the vision tower in
        # D-wake, spec §1.4).  Three candidates per destination unit.
        tower = ParamGeom(
            name="visual.blocks.0.proj.weight",
            tag="weights",
            shard_axis=ROWS,
            rows_full=3072,
            cols_full=HIDDEN,
            itemsize=1,
            stage=None,
        )
        plan = build_plan([tower], src, dst, waves=[["weights"]])
        self.assertEqual(len(plan.descs), 3)
        # Every piece stayed on its own card: no link is crossed at all.
        self.assertEqual([d.src_rank for d in plan.descs], [0, 1, 2])
        self.assertEqual(plan.cross_bytes, 0)
        self.assertEqual(plan.oncard_bytes, 3072 * HIDDEN)


class TestVocabPad(CustomTestCase):
    def test_vocab_pad_rows_are_zerofill_on_rank2_only(self):
        """D pads the vocab to 64*tp = 192; P at tp=1 pads to 64 and 248320 is
        already a multiple of 64, so the 128 pad rows have no VRAM source and
        no checkpoint source (spec §2.2)."""
        self.assertEqual(VOCAB_PADDED_D % (64 * 3), 0)
        self.assertEqual(VOCAB_REAL % 64, 0)
        self.assertEqual(VOCAB_PADDED_D - VOCAB_REAL, 128)

        def geom(name, itemsize):
            return ParamGeom(
                name=name,
                tag="weights",
                shard_axis=ROWS,
                rows_full=VOCAB_PADDED_D,
                cols_full=HIDDEN,
                itemsize=itemsize,
                pad_units=VOCAB_PADDED_D - VOCAB_REAL,
                stage=0,
            )

        plan = build_plan(
            [geom("lm_head.weight", 1), geom("model.embed_tokens.weight", 2)],
            _p_layout(),
            _d_layout(None),
            waves=[["weights"]],
        )

        zf = [d for d in plan.descs if d.kind == ZEROFILL]
        self.assertEqual({d.dst_rank for d in zf}, {2})
        self.assertEqual({d.src_rank for d in zf}, {-1})
        self.assertEqual(len(zf), 2)  # one per tensor, not one per row
        self.assertEqual(
            sorted(d.nbytes for d in zf),
            sorted([128 * HIDDEN * 1, 128 * HIDDEN * 2]),
        )
        # Spec §2.2's own arithmetic: 1.88 MiB.
        self.assertEqual(sum(d.nbytes for d in zf), 1966080)
        self.assertAlmostEqual(plan.zerofill_bytes / (1 << 20), 1.875, places=3)

        # Even split of 248448 over three ranks; ranks 0 and 1 are entirely
        # sourced, so a ZEROFILL there would be a defect, not a pad.
        self.assertEqual(VOCAB_PADDED_D // 3, 82816)
        for r in (0, 1):
            with self.subTest(rank=r):
                self.assertEqual(
                    sum(
                        d.nbytes
                        for d in plan.descs
                        if d.dst_rank == r and d.kind == ZEROFILL
                    ),
                    0,
                )
        # The pad is not counted as moved bytes -- it crosses no link.
        self.assertEqual(
            plan.oncard_bytes + plan.cross_bytes,
            (VOCAB_REAL * HIDDEN * 1) + (VOCAB_REAL * HIDDEN * 2),
        )


class TestStridedPitches(CustomTestCase):
    def test_strided_classes_carry_pitches(self):
        """The 2-D classes: the FULL side's pitch is the full input dimension,
        the SHARDED side's pitch is its own run (spec §2.2, §1.4 rule 3)."""
        src, dst = _p_layout(), _d_layout(MLP_RATIO)

        # D wakes: P (full) -> D (sharded).  spitch == in_full, dpitch == run.
        d_wake = build_plan([_mlp_down_geom()], src, dst, waves=[["weights_0"]])
        self.assertEqual({d.kind for d in d_wake.descs}, {STRIDED2D})
        for d in d_wake.descs:
            with self.subTest(dst_rank=d.dst_rank):
                self.assertEqual(d.rows, HIDDEN)
                self.assertEqual(d.spitch, MLP_IN_FULL)
                self.assertEqual(d.dpitch, d.run_bytes)
                self.assertEqual(d.run_bytes, MLP_WIDTHS_OK[d.dst_rank])
                self.assertEqual(d.nbytes, d.rows * d.run_bytes)
        self.assertEqual([d.src_off for d in d_wake.descs], [0, 7808, 12672])
        self.assertEqual([d.dst_off for d in d_wake.descs], [0, 0, 0])

        # P wakes: the direction reverses and so do the pitches.
        p_wake = build_plan([_mlp_down_geom()], dst, src, waves=[["weights_0"]])
        self.assertEqual(len(p_wake.descs), 3)
        for d in p_wake.descs:
            with self.subTest(src_rank=d.src_rank):
                self.assertEqual(d.spitch, d.run_bytes)
                self.assertEqual(d.dpitch, MLP_IN_FULL)
                self.assertEqual(d.run_bytes, MLP_WIDTHS_OK[d.src_rank])
        self.assertEqual([d.dst_off for d in p_wake.descs], [0, 7808, 12672])

        # The attention 2-D class of spec §2.2: same law, different numbers.
        attn = ParamGeom(
            name="model.layers.0.o_proj.weight",
            tag="weights_0",
            shard_axis=COLS,
            rows_full=HIDDEN,
            cols_full=ATTN_IN_FULL,
            itemsize=1,
            stage=0,
            dst_widths=[3072, 1536, 1536],
        )
        plan = build_plan([attn], src, dst, waves=[["weights_0"]])
        self.assertEqual([d.run_bytes for d in plan.descs], [3072, 1536, 1536])
        self.assertEqual([d.spitch for d in plan.descs], [ATTN_IN_FULL] * 3)
        # Spec §2.2's "start col 0 / 3072 / 4608" is the SOURCE column.
        self.assertEqual([d.src_off for d in plan.descs], [0, 3072, 4608])

        # THE CAN-FAIL for the byte/element confusion: on a bf16 tensor every
        # pitch and run must DOUBLE.  int8 hides a missing element_size()
        # multiply because its itemsize is 1.
        wide = ParamGeom(
            name="model.layers.0.o_proj.weight_bf16",
            tag="weights_0",
            shard_axis=COLS,
            rows_full=HIDDEN,
            cols_full=ATTN_IN_FULL,
            itemsize=2,
            stage=0,
            dst_widths=[3072, 1536, 1536],
        )
        wide_plan = build_plan([wide], src, dst, waves=[["weights_0"]])
        self.assertEqual([d.spitch for d in wide_plan.descs], [ATTN_IN_FULL * 2] * 3)
        self.assertEqual([d.run_bytes for d in wide_plan.descs], [6144, 3072, 3072])
        self.assertEqual([d.src_off for d in wide_plan.descs], [0, 6144, 9216])


class TestDeterminism(CustomTestCase):
    def _inventory(self):
        return [
            _qkvz_geom(),
            _mlp_down_geom(),
            ParamGeom(
                name="model.layers.0.input_layernorm.weight",
                tag="weights_0",
                shard_axis=REPLICATED,
                rows_full=1,
                cols_full=HIDDEN,
                itemsize=2,
                stage=0,
            ),
        ]

    def test_plan_is_identical_under_shuffled_parameter_iteration(self):
        """``core.cpp:206-222`` iterates an ``unordered_map``; its order differs
        between the two processes.  Sorting by NAME makes the hazard
        structurally impossible instead of argued (judges' graft 9)."""
        src, dst = _p_layout(), _d_layout(MLP_RATIO)
        reference = build_plan(self._inventory(), src, dst, waves=[["weights_0"]])
        self.assertGreater(len(reference.descs), 3)  # denominator, not a tautology
        rng = random.Random(1273)
        for trial in range(8):
            shuffled = self._inventory()
            rng.shuffle(shuffled)
            plan = build_plan(shuffled, src, dst, waves=[["weights_0"]])
            with self.subTest(trial=trial):
                self.assertEqual(plan.descs, reference.descs)
                self.assertEqual(plan.plan_id, reference.plan_id)

        # And the id is POINTER-FREE: two processes hold the same tensors at
        # different addresses, so an id that moved with the address could never
        # be compared across ranks (spec §7 W52, "plan hash != the front's").
        with_ptrs = build_plan(
            self._inventory(),
            src,
            dst,
            waves=[["weights_0"]],
            ptr_of=lambda group, rank, name: 0x7F0000000000 + 4096 * rank,
        )
        self.assertEqual(with_ptrs.plan_id, reference.plan_id)
        self.assertNotEqual(with_ptrs.descs, reference.descs)

    def test_plan_id_moves_when_the_geometry_moves(self):
        """A hash that cannot change is not a hash."""
        src = _p_layout()
        a = build_plan(
            self._inventory(), src, _d_layout(MLP_RATIO), waves=[["weights_0"]]
        )
        b = build_plan(
            self._inventory(), src, _d_layout([60, 39, 37]), waves=[["weights_0"]]
        )
        self.assertNotEqual(a.plan_id, b.plan_id)
        self.assertEqual(len(a.plan_id), 12)


class TestCoalescing(CustomTestCase):
    """E4: pieces >= 2 MiB issued async cost <= 1 %; 256 KiB costs 4.6-6.4 %."""

    def _pieces(self, sizes, gap_before=()):
        """FLAT descriptors laid out back to back in both address spaces,
        except where ``gap_before`` names a piece to push away."""
        descs, src, dst = [], 0x1000, 0x9000
        for i, n in enumerate(sizes):
            if i in gap_before:
                src += 4096
                dst += 4096
            descs.append(
                XchgDesc(
                    tag="weights_0",
                    src_rank=0,
                    dst_rank=1,
                    param_name=f"p{i}",
                    kind=FLAT,
                    nbytes=n,
                    rows=1,
                    run_bytes=n,
                    spitch=0,
                    dpitch=0,
                    src_ptr=src,
                    dst_ptr=dst,
                    src_off=0,
                    dst_off=0,
                )
            )
            src += n
            dst += n
        return descs

    def test_no_piece_below_two_mib_survives_coalescing(self):
        """The floor is an invariant about MERGEABLE neighbours, not about
        absolute size: ``A_log`` is 84 bytes and has no neighbour, so a literal
        'no piece < 2 MiB' is unreachable by construction and would be a test
        that can only pass by lying."""
        self.assertEqual(COALESCE_FLOOR_BYTES, 2 * 1024 * 1024)

        small = [256 * 1024] * 12  # 3 MiB in 12 adjacent pieces
        merged = coalesce(self._pieces(small), COALESCE_FLOOR_BYTES)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].nbytes, sum(small))
        self.assertEqual(merged[0].pieces, 12)
        self.assertEqual(unmergeable_below_floor(merged), [])
        self.assertEqual(
            [d.nbytes for d in merged if d.nbytes < COALESCE_FLOOR_BYTES], []
        )

        # THE CAN-FAIL: one piece pushed out of adjacency stays below the floor
        # and is REPORTED, so the histogram cannot hide it.
        holed = coalesce(self._pieces(small, gap_before={6}), COALESCE_FLOOR_BYTES)
        self.assertEqual([d.nbytes for d in holed], [6 * 256 * 1024] * 2)
        self.assertEqual(len(unmergeable_below_floor(holed)), 2)

        # A 2-D piece is never merged with anything: its pitches are its shape.
        two_d = XchgDesc(
            tag="weights_0",
            src_rank=0,
            dst_rank=1,
            param_name="o",
            kind=STRIDED2D,
            nbytes=HIDDEN * 1536,
            rows=HIDDEN,
            run_bytes=1536,
            spitch=ATTN_IN_FULL,
            dpitch=1536,
            src_ptr=0x1000,
            dst_ptr=0x9000,
        )
        self.assertEqual(coalesce([two_d, two_d], COALESCE_FLOOR_BYTES), [two_d, two_d])

    def test_coalescing_never_crosses_a_pair_or_a_tag(self):
        a = self._pieces([1 << 20, 1 << 20])
        for changed in (
            a[1].replace(dst_rank=2),
            a[1].replace(tag="weights_1"),
            a[1].replace(src_rank=1),
        ):
            with self.subTest(changed=changed.tag + str(changed.dst_rank)):
                self.assertEqual(
                    len(coalesce([a[0], changed], COALESCE_FLOOR_BYTES)), 2
                )

    def test_histogram_counts_every_piece(self):
        descs = self._pieces([84, 256, 4096, 1 << 20, 8 << 20], gap_before={1, 2, 3, 4})
        hist = piece_histogram(descs)
        self.assertEqual(sum(hist.values()), len(descs))
        self.assertEqual(hist.get("<4K"), 2)
        self.assertEqual(hist.get(">=32M"), None)


class TestWaves(CustomTestCase):
    def _sb4_cards(self):
        return chunk_tag_cards(
            SB4_STAGE_LAYERS, SB4_LAYERS_PER_CHUNK, SB4_CHUNKS, CARDS
        )

    def test_waves_free_bytes_on_every_card(self):
        """The rule (spec §1.2), reproduced from boot weg2sb4's own cut."""
        cards = self._sb4_cards()
        self.assertEqual(cards["weights_6"], (1, 2))  # the straddling chunk
        waves = derive_waves(weights_family_tags(SB4_CHUNKS), cards, CARDS)
        self.assertEqual(waves, SB4_WAVES)
        for w, wave in enumerate(waves):
            with self.subTest(wave=w):
                touched = set()
                for tag in wave:
                    touched |= set(cards.get(tag, CARDS))
                self.assertEqual(touched, set(CARDS))
        # Every tag exactly once, base last -- the front's permutation guard
        # (``front.py:2656-2663``) is checked against this flattening.
        flat = [t for wave in waves for t in wave]
        self.assertEqual(sorted(flat), sorted(weights_family_tags(SB4_CHUNKS)))
        self.assertEqual(flat[-1], "weights")

    def test_wave_count_is_bounded_by_the_thinnest_card(self):
        """Spec §1.2 says 'the fewest waves such that every wave frees bytes on
        every card'.  Taken literally that is ONE wave (today's shape), which
        contradicts its own table and its own refusal of nine.  The derivation
        that reproduces the table is the MAXIMUM wave count under the every-card
        constraint, bounded by the card the fewest tags touch."""
        cards = self._sb4_cards()
        per_card = {c: 0 for c in CARDS}
        for tag in weights_family_tags(SB4_CHUNKS):
            for c in cards.get(tag, CARDS):
                per_card[c] += 1
        self.assertEqual(per_card, {0: 5, 1: 4, 2: 3})
        self.assertEqual(
            len(derive_waves(weights_family_tags(SB4_CHUNKS), cards, CARDS)),
            min(per_card.values()),
        )

    def test_a_degenerate_cut_collapses_to_one_wave(self):
        """Spec §1.2: 'A degenerate cut collapses to one wave, i.e. today's
        shape.'  Every chunk on one stage: no wave but the last can free
        anywhere else, so they all fold into it."""
        cards = chunk_tag_cards([64, 0, 0], SB4_LAYERS_PER_CHUNK, SB4_CHUNKS, CARDS)
        self.assertEqual(set(cards.values()), {(0,)})
        waves = derive_waves(weights_family_tags(SB4_CHUNKS), cards, CARDS)
        self.assertEqual(len(waves), 1)
        self.assertEqual(len(waves[0]), SB4_CHUNKS + 1)


class TestByteMatrixAndLogLine(CustomTestCase):
    def _plan(self):
        return build_plan(
            [_qkvz_geom(), _mlp_down_geom()],
            _p_layout(),
            _d_layout(MLP_RATIO),
            waves=[["weights_0"]],
        )

    def test_matrix_is_symmetric_and_names_six_ranks(self):
        plan = self._plan()
        m = plan.byte_matrix
        self.assertEqual(len(m), 6)
        self.assertEqual({len(row) for row in m}, {6})
        # P is global ranks 0..2, D is 3..5; in D-wake only P sends.
        self.assertEqual(sum(sum(row) for row in m[3:]), 0)
        self.assertEqual(
            sum(sum(row) for row in m[:3]),
            sum(d.nbytes for d in plan.descs if d.kind != ZEROFILL),
        )
        # The whole tag lives on P stage 0, so every byte leaves global rank 0.
        self.assertEqual(sum(m[1]) + sum(m[2]), 0)

    def test_acceptance_line_is_greppable_and_carries_its_denominators(self):
        plan = self._plan()
        with self.assertLogs("sglang.srt.weg2.weight_exchange", level="INFO") as cm:
            line = emit_plan_line(plan)
        self.assertEqual(cm.output, [f"INFO:sglang.srt.weg2.weight_exchange:{line}"])
        print("ACCEPTANCE " + line)
        self.assertTrue(line.startswith("WEG2-XCHG-PLAN "))
        for key in (
            "dir=P2D",
            "waves=",
            "descs=",
            "coalesced=",
            "min_piece_mib=",
            "bytes_gib=",
            "oncard_gib=",
            "cross_gib=",
            "zerofill_mib=",
            "hist=",
            "plan_id=",
        ):
            with self.subTest(key=key):
                self.assertIn(key, line)
        self.assertIn(f"plan_id={plan.plan_id}", line)
        # On-card is src_rank == dst_rank, which on this rig IS the same card:
        # both groups get the same CUDA_VISIBLE_DEVICES uuid string, so rank n
        # of either group runs on cards[n] (spec §1.3).
        self.assertEqual(
            plan.oncard_bytes,
            sum(
                d.nbytes
                for d in plan.descs
                if d.kind != ZEROFILL and d.src_rank == d.dst_rank
            ),
        )
        self.assertEqual(
            plan.oncard_bytes + plan.cross_bytes,
            sum(d.nbytes for d in plan.descs if d.kind != ZEROFILL),
        )
        self.assertEqual(plan.log_line("D2P").split()[1], "dir=D2P")


class TestAgainstRealModules(CustomTestCase):
    """The hermetic double: real layer classes on the meta device, so the
    geometry is READ from the fork's own constructors rather than restated."""

    def setUp(self):
        from sglang.srt.runtime_context import get_context

        self._saved = get_tp_partition_ratios()
        set_tp_partition_ratios(None)
        self._server_args = get_context().override_server_args()
        self._server_args.install()

    def tearDown(self):
        self._server_args.restore()
        set_tp_partition_ratios(self._saved)

    def test_the_real_fused_layer_agrees_with_the_plan_builders_blocks(self):
        from sglang.srt.layers.linear import MergedColumnParallelLinear

        set_tp_partition_ratios(QKVZ_RATIO)
        mods = []
        for r in range(3):
            with torch.device("meta"):
                mods.append(
                    MergedColumnParallelLinear(
                        input_size=HIDDEN,
                        output_sizes=list(QKVZ_OUTPUT_SIZES),
                        bias=False,
                        prefix="in_proj_qkvz",
                        tp_rank=r,
                        tp_size=3,
                        tp_units=QKVZ_UNITS,
                    )
                )
        per_rank = device_block_offsets(
            QKVZ_OUTPUT_SIZES, QKVZ_RATIO, 3, units=QKVZ_UNITS
        )
        for r, m in enumerate(mods):
            with self.subTest(rank=r):
                self.assertEqual(
                    list(m.output_partition_sizes), [blk.size for blk in per_rank[r]]
                )
                geom = StorageGeom.of(m.weight)
                self.assertEqual(geom.rows, m.output_size_per_partition)
                self.assertEqual(geom.cols, HIDDEN)
                self.assertEqual(
                    geom.rows, per_rank[r][-1].dev_row + per_rank[r][-1].size
                )
        self.assertEqual(
            sum(m.output_size_per_partition for m in mods), sum(QKVZ_OUTPUT_SIZES)
        )

    def test_the_real_row_parallel_layer_agrees_with_the_two_d_pitches(self):
        from sglang.srt.layers.linear import RowParallelLinear

        set_tp_partition_ratios(MLP_RATIO)
        mods = []
        for r in range(3):
            with torch.device("meta"):
                mods.append(
                    RowParallelLinear(
                        MLP_IN_FULL,
                        HIDDEN,
                        bias=False,
                        prefix="down_proj",
                        tp_rank=r,
                        tp_size=3,
                        tp_units=MLP_UNITS,
                    )
                )
        widths = [m.input_size_per_partition for m in mods]
        self.assertEqual(widths, MLP_WIDTHS_OK)
        self.assertEqual(sum(widths), MLP_IN_FULL)

        plan = build_plan(
            [_mlp_down_geom()], _p_layout(), _d_layout(MLP_RATIO), waves=[["weights_0"]]
        )
        self.assertEqual([d.run_bytes for d in plan.descs], widths)
        for r, m in enumerate(mods):
            geom = StorageGeom.of(m.weight)
            with self.subTest(rank=r):
                # The plan's ParamGeom is int8 (itemsize 1), so its BYTE pitch
                # equals the module's ELEMENT pitch; the equality is a
                # geometry check, not a dtype claim.
                self.assertEqual(geom.pitch, widths[r])
                self.assertEqual(plan.descs[r].dpitch, geom.pitch)
                self.assertEqual(plan.descs[r].rows, geom.rows)

    def test_shard_offsets_is_the_prefix_sum_tp_loaded_shard_start_computes(self):
        """The plan's geometry helper against the loader's own, on the same
        vectors (``distributed/utils.py:1808``)."""
        from sglang.srt.distributed.utils import tp_loaded_shard_start

        set_tp_partition_ratios(MLP_RATIO)
        ranges = shard_offsets(MLP_IN_FULL, MLP_RATIO, 3, units=MLP_UNITS)
        self.assertEqual([size for _, size in ranges], MLP_WIDTHS_OK)
        for r, (start, size) in enumerate(ranges):
            with self.subTest(rank=r):
                self.assertEqual(
                    start,
                    tp_loaded_shard_start(MLP_IN_FULL, 3, r, size, units=MLP_UNITS),
                )
        set_tp_partition_ratios(None)
        even = shard_offsets(MLP_IN_FULL, None, 4)
        self.assertEqual([s for s, _ in even], [0, 4352, 8704, 13056])


if __name__ == "__main__":
    unittest.main()
