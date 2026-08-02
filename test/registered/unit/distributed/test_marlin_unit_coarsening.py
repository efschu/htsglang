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
    _marlin_min_thread_pair,
    _marlin_packable_family,
    _marlin_uneven_tp_block,
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
            _marlin_min_thread_pair(),
            (GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MIN_THREAD_K),
        )

    def test_the_block_is_ONE_value_for_both_axes(self):
        """#385: the per-axis reading (64 out / 128 in) is what broke the
        coupled dimension. One value, the lcm, on both."""
        n, k = _marlin_min_thread_pair()
        self.assertEqual(_marlin_uneven_tp_block(), 128)
        self.assertEqual(_marlin_uneven_tp_block() % n, 0)
        self.assertEqual(_marlin_uneven_tp_block() % k, 0)

    def test_k_is_stricter_than_the_repack_tile(self):
        """The repack tile on k is 16; the GEMM wants 128. Aligning to the
        tile alone would pass the repack and fail verify_marlin_supported."""
        n, k = _marlin_min_thread_pair()
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


class TestCoupledDimensionConsistency(_Base):
    """THE INVARIANT #383 BROKE, and the test that would have caught it.

    gate_up is COLUMN-parallel and splits its OUTPUT; down_proj is
    ROW-parallel and splits its INPUT; both are the SAME intermediate
    dimension. Whatever coarsening applies, the per-rank intermediate implied
    by gate_up must equal the one down_proj is built for -- otherwise the
    weight is repacked for one k and handed an activation of another.

    #383 shipped a per-axis rule (64 on output, 128 on input) and broke it:
    gate_up implied [14880, 8960, 8928] while down_proj expected
    [14848, 8960, 8960]. It reached hardware and surfaced as #377 gap 3,
    ``Tensor match failed for Tensor<568, 20480>`` at gptq_marlin.cuh:836 --
    which is not an alignment check at all but
    ``b_q_weight.size(0) == size_k / 16`` against the ACTIVATION's k.

    Every existing sibling states the rule in its own docstring
    (``awq_uneven_tp_block``: "Both dims carry the same value ... they
    partition the same intermediate dimension and must coarsen identically").
    The corpus tests checked each axis on its own and could not see a
    disagreement BETWEEN them; this class checks the pair.
    """

    #: (label, intermediate) -- gate_up is 2x this (gate and up fused).
    COUPLED = (
        ("Mistral-Small-24B", 32768),
        ("Qwen3.6-27B", 17408),
        ("small even", 8192),
    )

    def _pair(self, intermediate, marlin):
        """(gate_up-implied per-rank intermediate, down_proj-expected).

        gate_up is a MERGED column-parallel layer whose parts (gate, up) are
        each ``intermediate`` wide and are partitioned on the intermediate
        basis -- not one fused ``2*intermediate`` split, which rounds
        differently and is not what the layer does. Modelling it the fused way
        makes even the no-coarsening case look inconsistent, i.e. it encodes a
        false invariant; that mistake is why this helper says so.

        So both sides partition the SAME total on the SAME basis, and the only
        thing that can make them disagree is the coarsening choosing different
        blocks for idx 0 and idx 1 -- which is exactly what #383 did.
        """
        u = intermediate // 16
        gu_u = self.coarsen(intermediate, u, None, marlin, idx=0)  # column: output
        dp_u = self.coarsen(intermediate, u, None, marlin, idx=1)  # row: input
        return self.partition(intermediate, gu_u), self.partition(intermediate, dp_u)

    def test_gate_up_implied_equals_down_proj_expected(self):
        for label, inter in self.COUPLED:
            with self.subTest(model=label):
                implied, expected = self._pair(inter, True)
                self.assertEqual(
                    implied,
                    expected,
                    f"{label}: gate_up implies {implied} but down_proj is "
                    f"built for {expected} -- the weight would be repacked "
                    f"for one k and handed an activation of another",
                )

    def test_the_packed_row_counts_agree(self):
        """The quantity gptq_marlin.cuh:836 actually compares: k // 16."""
        for label, inter in self.COUPLED:
            with self.subTest(model=label):
                implied, expected = self._pair(inter, True)
                self.assertEqual(
                    [x // 16 for x in implied], [x // 16 for x in expected], label
                )

    def test_the_measured_377_shapes_are_in_the_corpus(self):
        """8960 / 9088 -- the per-rank intermediates behind the observed
        Tensor<560,...> and Tensor<568,...>."""
        implied, expected = self._pair(32768, True)
        self.assertEqual(implied, expected)
        self.assertIn(8960, expected)
        for v in expected:
            self.assertEqual(v % 128, 0, f"{v} is not marlin-aligned")
            self.assertEqual(v % 16, 0)

    def test_non_marlin_pairs_also_agree(self):
        """The invariant is not marlin-specific; it must hold whatever the
        coarsening decides, including when it decides nothing."""
        for label, inter in self.COUPLED:
            with self.subTest(model=label):
                implied, expected = self._pair(inter, False)
                self.assertEqual(implied, expected, label)


class TestEighthSiblingMxfp8(_Base):
    """#444b: MXFP8's ``[1, 32]`` is a QUANTIZATION fact, not an alignment one.

    Every sibling before this one either exposed no block (marlin, #383) or
    exposed a symmetric block it had registered on purpose. ``Fp8Config``
    pins ``weight_block_size = [1, 32]`` for ``use_mxfp8`` because the OCP
    spec fixes the scale layout to one row by 32 columns -- a statement about
    the checkpoint, not about how a shard may be cut. Read per-axis by
    ``_quant_block_aligned_units`` it said "output: element granularity,
    input: 32", which is the coupled-dimension disagreement #385 exists to
    prevent: gate_up's OUTPUT and down_proj's INPUT are the same intermediate
    and were coarsened by 1 and by 32.

    NOT REACHABLE ON THIS RIG TODAY. ``Fp8Config.get_min_capability`` returns
    100 for mxfp8, so no card here resolves the config at all; this is a
    latent registration gap, closed on the same terms as its seven siblings.
    """

    #: What ``Fp8Config.from_config`` pins for ``use_mxfp8``.
    MXFP8_BLOCK = [1, 32]

    def _cfg(self):
        # Fp8Config by CLASS NAME -- that is what _marlin_packable_family reads.
        return Fp8Config(list(self.MXFP8_BLOCK))

    def test_the_block_this_corpus_models_is_the_one_fp8_py_pins(self):
        """The corpus must not drift from the config it claims to model."""
        import inspect

        from sglang.srt.layers.quantization.fp8 import Fp8Config as RealFp8

        src = inspect.getsource(RealFp8.from_config)
        self.assertIn("weight_block_size = [1, 32]", src)

    def test_the_exposed_block_is_asymmetric(self):
        """The premise. If MXFP8 ever exposes a symmetric block upstream this
        sibling is obsolete and should be retired rather than kept green."""
        self.assertNotEqual(self.MXFP8_BLOCK[0], self.MXFP8_BLOCK[1])

    def test_both_axes_coarsen_by_the_same_block(self):
        for total in (34816, 32768, 17408, 65536):
            with self.subTest(total=total):
                u = total // 16
                out = _quant_block_aligned_units(total, u, self._cfg(), 0)
                inp = _quant_block_aligned_units(total, u, self._cfg(), 1)
                self.assertEqual(out, inp, f"{total}: {out} vs {inp}")

    def test_the_coupled_dimension_partitions_identically(self):
        """The invariant that matters: gate_up's OUTPUT split of the
        intermediate equals down_proj's INPUT split of the same number."""
        for label, inter in (("Mistral-Small-24B", 32768), ("Qwen3.6-27B", 17408)):
            with self.subTest(model=label):
                u = inter // 16
                implied = self.partition(
                    inter, _quant_block_aligned_units(inter, u, self._cfg(), 0)
                )
                expected = self.partition(
                    inter, _quant_block_aligned_units(inter, u, self._cfg(), 1)
                )
                self.assertEqual(
                    implied,
                    expected,
                    f"{label}: gate_up implies {implied}, down_proj expects {expected}",
                )

    def test_the_resulting_block_is_marlin_valid(self):
        """``Fp8Config`` is in ``_MARLIN_PACKABLE_CONFIGS``, so the shard must
        also satisfy marlin's minimum thread shape on both axes."""
        inter = 32768
        u = _quant_block_aligned_units(inter, inter // 16, self._cfg(), 0)
        sizes = self.partition(inter, u)
        self.assertEqual([x % _marlin_uneven_tp_block() for x in sizes], [0] * TP)
        self.assertEqual(sum(sizes), inter)
        self.assertTrue(all(s > 0 for s in sizes))

    def test_no_partition_tax_over_the_awq_group_32_vehicle(self):
        """ANALYSE_442's argument, executed: ``lcm(32, 128) == 128`` is the
        block AWQ already imposes for group size 32, so registering MXFP8
        costs nothing a shipped vehicle does not already pay."""
        from sglang.srt.layers.quantization.awq.awq import awq_uneven_tp_block

        awq_block = awq_uneven_tp_block(32)
        for total in (34816, 32768, 17408):
            with self.subTest(total=total):
                u = total // 16
                mx = self.partition(
                    total, _quant_block_aligned_units(total, u, self._cfg(), 0)
                )
                awq = self.partition(
                    total,
                    _quant_block_aligned_units(total, u, _plain_cfg(awq_block), 0),
                )
                self.assertEqual(mx, awq)

    def test_symmetric_exposures_are_untouched(self):
        """PLAN-EQUALITY PIN for the seven siblings: the new branch keys on
        ``raw[0] != raw[1]``, so every symmetric block must be unaffected."""
        from sglang.srt.layers.quantization.awq.awq import awq_uneven_tp_block
        from sglang.srt.layers.quantization.gptq.gptq import gptq_uneven_tp_block

        for label, blk in (
            ("fp8 block", [128, 128]),
            ("gguf", [256, 256]),
            ("awq gs=128", awq_uneven_tp_block(128)),
            ("gptq gs=128", gptq_uneven_tp_block(128)),
            ("int8 16", [16, 16]),
        ):
            for total in (34816, 32768, 17408):
                for idx in (0, 1):
                    with self.subTest(cfg=label, total=total, idx=idx):
                        u = total // 16
                        self.assertEqual(
                            _quant_block_aligned_units(total, u, _plain_cfg(blk), idx),
                            _quant_block_aligned_units(
                                total, u, _plain_cfg(list(blk)), idx
                            ),
                        )
                        # and the value is the one the pre-#444b rule produced
                        self.assertEqual(
                            _quant_block_aligned_units(total, u, _plain_cfg(blk), idx),
                            _legacy_block_aligned_units(total, u, blk[idx]),
                        )


def _legacy_block_aligned_units(total, units, block):
    """The pre-#444b arithmetic for a config that exposes a block, restated so
    the plan-equality pin compares against a value and not against itself."""
    from sglang.srt.distributed.utils import block_aligned_units

    if not block:
        return units
    return block_aligned_units(total, units, block)


if __name__ == "__main__":
    unittest.main()
