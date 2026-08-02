"""#426 -- the MoE source-side shard offset belongs to the checkpoint tensor.

INVARIANT under test: the start of THIS rank's slice of a full checkpoint
tensor is a function of that tensor's real geometry. It must never be derived
from the DESTINATION ``shard_size``, because the destination can be padded and
the checkpoint is not.

Upstream sgl-project/sglang#32781 is the invariant broken. DeepSeek-V4-Pro at
TP16 with Marlin: the real shard is 3072 / 16 = 192, Marlin rounds the
destination up to 256, and ``_load_w13`` / ``_load_w2`` used the padded 256 to
index the 3072-wide checkpoint tensor. Rank 13 asked for a narrow starting at
256 * 13 = 3328 and got::

    IndexError: start out of range (expected to be in range of [-3072, 3072],
                but got 3328)

Our fork had already factored the expression into one named helper
(``_moe_src_start``) and the uneven-TP branch already derived it from
``loaded_total``. The even-TP branch still returned ``shard_size * tp_rank``,
so we reproduced their bug on every padded destination.

Two things are pinned:

* the padded case is in range and lands on the right 192-wide slice;
* the UNPADDED case is bit-identical to the old expression, for every rank of
  every TP size exercised here. That is the whole compatibility claim -- when
  nothing pads, ``loaded_total == shard_size * tp_size`` and the two formulas
  are the same number.

GPU-free: ``_moe_src_start`` reads three ints and two layer attributes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: The reporter's geometry: DeepSeek-V4-Pro intermediate 3072, TP16, Marlin
#: pads the per-rank destination from 192 to 256.
V4_PRO_INTERMEDIATE = 3072
V4_PRO_TP = 16
V4_PRO_REAL_SHARD = V4_PRO_INTERMEDIATE // V4_PRO_TP  # 192
V4_PRO_PADDED_SHARD = 256


def _src_start(loaded_total, shard_size, tp_rank, *, tp_size, presharded=False):
    """Call the real helper against a minimal stand-in layer.

    No FusedMoE is constructed: the helper reads ``moe_tp_size``,
    ``moe_tp_family``, ``moe_tp_units`` and ``use_presharded_weights``, and
    nothing else. Building a layer would drag in a device and a quant method
    and would test those instead.
    """
    stub = SimpleNamespace(
        moe_tp_size=tp_size,
        moe_tp_family="moe",
        moe_tp_units=None,
        use_presharded_weights=presharded,
    )
    return FusedMoE._moe_src_start(stub, loaded_total, shard_size, tp_rank)


class TestPaddedDestinationStaysInsideTheCheckpoint(CustomTestCase):
    """The falsifier: unfixed, ranks 12..15 index past the end of the tensor."""

    def test_every_rank_of_the_reported_configuration_is_in_range(self):
        for tp_rank in range(V4_PRO_TP):
            with self.subTest(tp_rank=tp_rank):
                start = _src_start(
                    V4_PRO_INTERMEDIATE,
                    V4_PRO_PADDED_SHARD,
                    tp_rank,
                    tp_size=V4_PRO_TP,
                )
                self.assertLessEqual(
                    start + V4_PRO_REAL_SHARD,
                    V4_PRO_INTERMEDIATE,
                    f"rank {tp_rank} starts at {start}, past the "
                    f"{V4_PRO_INTERMEDIATE}-wide checkpoint tensor",
                )

    def test_the_starts_are_the_real_prefix_sums(self):
        for tp_rank in range(V4_PRO_TP):
            with self.subTest(tp_rank=tp_rank):
                self.assertEqual(
                    _src_start(
                        V4_PRO_INTERMEDIATE,
                        V4_PRO_PADDED_SHARD,
                        tp_rank,
                        tp_size=V4_PRO_TP,
                    ),
                    V4_PRO_REAL_SHARD * tp_rank,
                )

    def test_the_narrow_that_raised_upstream_now_succeeds(self):
        """Reproduce their traceback's last frame with a real tensor.

        Can-fail arm in both directions: the legacy expression is asserted to
        raise exactly the reported IndexError, so a run where nothing raises
        would mean the reproduction stopped reproducing.
        """
        checkpoint = torch.zeros(V4_PRO_INTERMEDIATE)
        for tp_rank in (13, 15):
            with self.subTest(tp_rank=tp_rank):
                legacy_start = V4_PRO_PADDED_SHARD * tp_rank
                with self.assertRaises(IndexError):
                    checkpoint.narrow(0, legacy_start, V4_PRO_REAL_SHARD)

                start = _src_start(
                    V4_PRO_INTERMEDIATE,
                    V4_PRO_PADDED_SHARD,
                    tp_rank,
                    tp_size=V4_PRO_TP,
                )
                slice_ = checkpoint.narrow(0, start, V4_PRO_REAL_SHARD)
                self.assertEqual(slice_.shape[0], V4_PRO_REAL_SHARD)


class TestUnpaddedIsBitIdenticalToTheOldExpression(CustomTestCase):
    """Control: the compatibility claim, over the whole grid we can enumerate."""

    def test_the_two_formulas_agree_whenever_nothing_pads(self):
        for tp_size in (1, 2, 3, 4, 6, 8, 16):
            for shard_size in (64, 128, 192, 512, 1408):
                loaded_total = shard_size * tp_size
                for tp_rank in range(tp_size):
                    with self.subTest(
                        tp_size=tp_size, shard_size=shard_size, tp_rank=tp_rank
                    ):
                        self.assertEqual(
                            _src_start(
                                loaded_total, shard_size, tp_rank, tp_size=tp_size
                            ),
                            shard_size * tp_rank,
                        )

    def test_tp1_is_always_zero(self):
        for shard_size in (192, 256, 3072):
            with self.subTest(shard_size=shard_size):
                self.assertEqual(
                    _src_start(V4_PRO_INTERMEDIATE, shard_size, 0, tp_size=1), 0
                )


class TestTheCasesWithNoSourceGeometryKeepTheLegacyValue(CustomTestCase):
    """Deliberate non-changes: no offset is invented where none can be read."""

    def test_presharded_checkpoints_are_untouched(self):
        """Each rank already holds its own tensor, so there is no full-tensor
        geometry to derive from; changing this would silently move a path this
        fix has no evidence about."""
        for tp_rank in range(4):
            with self.subTest(tp_rank=tp_rank):
                self.assertEqual(
                    _src_start(512, 512, tp_rank, tp_size=4, presharded=True),
                    512 * tp_rank,
                )

    def test_an_indivisible_source_length_is_untouched(self):
        self.assertEqual(_src_start(3070, 256, 2, tp_size=16), 256 * 2)


if __name__ == "__main__":
    unittest.main()
