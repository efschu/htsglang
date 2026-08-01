"""Marlin tile alignment for uneven-TP shards — the sixth sibling (#383).

THE FAMILY. Five configs already expose a synthetic ``weight_block_size`` so
the #82/#65 coarsening fires and their per-rank shards land on a marlin-valid
boundary: ``awq_uneven_tp_block`` (#289), ``gptq_uneven_tp_block`` (#300),
``AutoRoundConfig`` (#86), ``CompressedTensorsConfig._group_size_block``,
``int8_w8a8_uneven_tp_block`` (#353). Every one of them derives the block from
a GROUP SIZE.

THE SIXTH. A checkpoint with no group size exposes no block, so nothing fires —
and it can still resolve to marlin. compressed-tensors **FP8-dynamic**
(``CompressedTensorsW8A16Fp8``) sets ``use_marlin = _marlin_available()`` and
repacks through ``prepare_fp8_layer_for_marlin``; on sm86 there is no native
FP8 GEMM, so marlin is the path. Measured on Mistral-Small-24B FP8 at the rig
vector ``[29607, 17780, 17780]``: gate_up 65536 in 4096 (16-element) units
partitions to ``[29776, 17888, 17872]`` and the load dies with
``size_n = 17888 is not divisible by tile_n_size = 64`` (#377 gap 2).

THE RULE. When the layer's resolved quant method is marlin, coarsen its unit
family to marlin's per-axis minimum — 64 on the output dim, **128** on the
input dim (``GPTQ_MARLIN_MIN_THREAD_N`` / ``_K``, the constants
``verify_marlin_supported`` itself checks). Those are stricter than the repack
tiles (64/16), so aligning to the tile alone would still fail the GEMM check.

PER LAYER, NOT GLOBAL. A mixed checkpoint has layers on different schemes, and
an ignored layer resolves to ``UnquantizedLinearMethod``; coarsening those too
would inflate shards that carry no such constraint.

Everything here drives the REAL partitioner (``tp_partition_sizes``) on the
real shapes, not just the helper — a unit count that looks right and a
partition that is still mis-aligned is exactly the failure mode.
"""

import unittest

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import (
    _marlin_min_thread_by_block_idx,
    _marlin_packable_family,
    _quant_block_aligned_units,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: The rig vector every historic sibling was measured at.
RIG = [29607, 17780, 17780]
TP = 3


class _QC:
    """A non-marlin-family quant config (a block, or nothing)."""

    def __init__(self, block=None):
        self.weight_block_size = block


class Fp8Config(_QC):  # noqa: N801 - the CLASS NAME is the signal under test
    pass


class CompressedTensorsConfig(_QC):  # noqa: N801 - ditto
    pass


def _marlin_cfg(block=None):
    return CompressedTensorsConfig(block)


def _plain_cfg(block=None):
    return _QC(block)


class _Base(CustomTestCase):
    def setUp(self):
        self._saved = get_tp_partition_ratios()
        set_tp_partition_ratios(RIG)

    def tearDown(self):
        set_tp_partition_ratios(self._saved)

    def partition(self, total, units):
        return tp_partition_sizes(total, TP, units=units)

    def coarsen(self, total, units, block, marlin, idx=0):
        cfg = _marlin_cfg(block) if marlin else _plain_cfg(block)
        return _quant_block_aligned_units(total, units, cfg, idx)


class TestTheConstantsAreImportedNotRestated(_Base):
    def test_min_thread_pair(self):
        from sglang.srt.layers.quantization.marlin_utils import (
            GPTQ_MARLIN_MIN_THREAD_K,
            GPTQ_MARLIN_MIN_THREAD_N,
        )

        self.assertEqual(
            _marlin_min_thread_by_block_idx(),
            (GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MIN_THREAD_K),
        )

    def test_k_is_stricter_than_the_repack_tile(self):
        """The repack tile on k is 16; the GEMM wants 128. Aligning to the
        tile alone would pass the repack and fail verify_marlin_supported."""
        n, k = _marlin_min_thread_by_block_idx()
        self.assertEqual((n, k), (64, 128))
        self.assertGreater(k, 16)


class TestTheMeasuredFailure(_Base):
    """#377 gap 2, reproduced through the real partitioner and then fixed."""

    TOTAL = 65536  # Mistral-Small-24B gate_up (2 x intermediate 32768)
    UNITS = 4096  # the #82 16-element MLP unit family

    def test_the_historic_partition_is_reproduced(self):
        before = self.partition(self.TOTAL, self.UNITS)
        self.assertEqual(before, [29776, 17888, 17872])
        self.assertIn(17888, before, "the number from the boot log")
        self.assertTrue([x for x in before if x % 64], "must be mis-aligned")

    def test_the_coarsening_fixes_it(self):
        units = self.coarsen(self.TOTAL, self.UNITS, None, True)
        after = self.partition(self.TOTAL, units)
        self.assertEqual([x % 64 for x in after], [0, 0, 0])
        self.assertEqual(sum(after), self.TOTAL, "no elements lost")

    def test_without_the_fix_it_stays_broken(self):
        """The counterfactual, so the test cannot pass for another reason."""
        units = self.coarsen(self.TOTAL, self.UNITS, None, False)
        self.assertEqual(self.partition(self.TOTAL, units), [29776, 17888, 17872])


class TestSiblingCorpus(_Base):
    """Every historic shape in the alignment family yields a marlin-valid
    partition under the new rule, on BOTH axes."""

    # (label, total, element-granular unit family)
    SHAPES = (
        ("#289/#300 Qwen3.6-27B gate_up", 34816, 34816 // 16),
        ("#289 Qwen3.6-27B intermediate", 17408, 17408 // 16),
        ("#377 Mistral gate_up", 65536, 65536 // 16),
        ("#377 Mistral intermediate", 32768, 32768 // 16),
        ("hidden 5120", 5120, 5120 // 16),
        ("q 4096", 4096, 4096 // 16),
        ("vocab 131072", 131072, 131072 // 16),
    )

    def test_output_axis_is_64_aligned(self):
        for label, total, units in self.SHAPES:
            with self.subTest(shape=label):
                u = self.coarsen(total, units, None, True, idx=0)
                sizes = self.partition(total, u)
                self.assertEqual([x % 64 for x in sizes], [0] * TP, f"{label}: {sizes}")
                self.assertEqual(sum(sizes), total)

    def test_input_axis_is_128_aligned(self):
        for label, total, units in self.SHAPES:
            with self.subTest(shape=label):
                u = self.coarsen(total, units, None, True, idx=1)
                sizes = self.partition(total, u)
                self.assertEqual(
                    [x % 128 for x in sizes], [0] * TP, f"{label}: {sizes}"
                )

    def test_every_rank_still_gets_a_share(self):
        """Coarsening must not starve a rank -- the vision-o_proj lesson in
        _quant_block_aligned_units' own comment."""
        for label, total, units in self.SHAPES:
            with self.subTest(shape=label):
                u = self.coarsen(total, units, None, True, idx=0)
                self.assertTrue(all(s > 0 for s in self.partition(total, u)), label)


class TestExistingSiblingsAreUnchanged(_Base):
    """PLAN-EQUALITY PIN. The five group-size siblings already expose a block
    that dominates marlin's 64, so #383 must be a no-op for them. If this ever
    goes red the change has become a measured-recipe change for shipped
    vehicles and must be called out, not shipped."""

    def test_gptq_and_awq_vehicles_keep_their_partition(self):
        from sglang.srt.layers.quantization.awq.awq import awq_uneven_tp_block
        from sglang.srt.layers.quantization.gptq.gptq import gptq_uneven_tp_block

        for label, blk in (
            ("awq gs=128", awq_uneven_tp_block(128)),
            ("gptq gs=128", gptq_uneven_tp_block(128)),
        ):
            for total in (34816, 17408):
                with self.subTest(cfg=label, total=total):
                    u = total // 16
                    before = self.partition(total, self.coarsen(total, u, blk, False))
                    after = self.partition(total, self.coarsen(total, u, blk, True))
                    self.assertEqual(before, after, f"{label} {total} changed")
                    self.assertEqual([x % 64 for x in after], [0] * TP)

    def test_an_fp8_block_128_config_is_unchanged(self):
        """The FP8-block-128 coarsening this rule is modelled on."""
        total, u = 32768, 32768 // 16
        blk = [128, 128]
        self.assertEqual(
            self.partition(total, self.coarsen(total, u, blk, False)),
            self.partition(total, self.coarsen(total, u, blk, True)),
        )


class TestPerLayerNotGlobal(_Base):
    """A mixed checkpoint must not have its non-marlin layers coarsened."""

    def test_a_non_marlin_layer_keeps_its_units(self):
        total, u = 65536, 4096
        self.assertEqual(self.coarsen(total, u, None, False), u)
        self.assertEqual(self.coarsen(total, u, None, False), u)
        self.assertEqual(self.coarsen(total, u, None, False), u)

    def test_the_marlin_layer_beside_it_is_coarsened(self):
        total, u = 65536, 4096
        self.assertLess(self.coarsen(total, u, None, True), u)

    def test_no_quant_config_is_still_a_passthrough(self):
        """UnquantizedLinearMethod layers arrive with quant_config None."""
        self.assertEqual(_quant_block_aligned_units(65536, 4096, None, 0), 4096)


class TestTheVerdictIsRankUniform(_Base):
    """The design point, and the reason this is NOT keyed on ``use_marlin``.

    ``CompressedTensorsW8A8Fp8.get_min_capability()`` is 89, so the same
    checkpoint resolves differently per rank on a mixed rig: measured in #377,
    TP0 on the 5090 (sm120) took the native FP8 scheme and TP1/TP2 on the
    3080s (sm86) fell back to marlin -- and only TP1/TP2 raised. A unit count
    derived from that verdict would DIFFER BETWEEN RANKS, which is silently
    mismatched shapes rather than a loud abort.
    """

    def test_the_predicate_takes_no_device_input(self):
        import inspect

        # CODE only: the docstring explains at length why use_marlin is the
        # wrong signal, so scanning the whole source would trip on the
        # explanation rather than on a use.
        full = inspect.getsource(_marlin_packable_family)
        doc = _marlin_packable_family.__doc__ or ""
        src = full.replace(doc, "")
        for device_ish in (
            "use_marlin",
            "get_device_capability",
            "cuda",
            "_marlin_available",
            "capability",
        ):
            self.assertNotIn(
                device_ish, src, f"{device_ish!r} makes the verdict rank-local"
            )

    def test_the_real_marlin_capable_configs_match(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig as RealCT,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8Config as RealFp8

        self.assertTrue(_marlin_packable_family(RealFp8.__new__(RealFp8)))
        self.assertTrue(_marlin_packable_family(RealCT.__new__(RealCT)))

    def test_unrelated_and_none(self):
        self.assertFalse(_marlin_packable_family(_plain_cfg()))
        self.assertFalse(_marlin_packable_family(None))

    def test_the_same_config_gives_the_same_units_every_call(self):
        """Rank-uniformity as an assertion: nothing about the process can
        change the answer."""
        cfg = _marlin_cfg(None)
        vals = {_quant_block_aligned_units(65536, 4096, cfg, 0) for _ in range(5)}
        self.assertEqual(len(vals), 1)


class TestEvenSplitUntouched(_Base):
    """Without an installed ratio plan the unit family is never consulted, so
    the even path must be byte-identical whatever this rule decides."""

    def test_no_plan_means_no_change(self):
        set_tp_partition_ratios(None)
        # divisible by tp so the even path is defined at all
        total = 49152
        even = tp_partition_sizes(total, TP)
        for marlin in (True, False):
            u = self.coarsen(total, total // 16, None, marlin)
            self.assertEqual(tp_partition_sizes(total, TP, units=u), even)


if __name__ == "__main__":
    unittest.main()
