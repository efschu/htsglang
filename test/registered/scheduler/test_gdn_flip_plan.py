# SPDX-License-Identifier: Apache-2.0
"""#631 GDN state flip arithmetic: hermetic contract tests (CPU-only).

Gates:

* spec validation -- shard vectors that do not partition their sub-block
  or head totals are refused loudly (mis-sliced conv state is the PD
  mamba transfer bug class);
* segment coverage -- all ranks' conv segments tile the full conv dim
  exactly once, per sub-block, never as one flat slice;
* slice/scatter byte identity -- full -> per-rank shards -> reassembled
  full is bit-exact for conv and temporal state;
* pair-payload roundtrip -- PP->TP payloads built per destination rank,
  consumed shard-side, then sent back TP->PP and scattered, reproduce
  the original full state bit-exactly per request;
* falsifier -- a receiver holding a DIVERGENT shard spec goes red
  (truncation/trailing-bytes loudness proven);
* real-config ledger pin -- the Qwen3.6-27B GDN constants reproduce the
  measured #625 per-slot state bytes and the DESIGN_631 section 3.4a
  both-layouts ledger term (the M2 lesson: mamba sizing is never left
  derived-only).
"""

import unittest

import torch

from sglang.srt.layers.dcp.gdn_flip_plan import (
    GdnShardSpec,
    conv_scatter,
    conv_slice,
    pack_gdn_pair_payload,
    temporal_scatter,
    temporal_slice,
    unpack_gdn_pair_payload,
)
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Small-world spec: sub-blocks (6, 6, 12) channels, 8 heads, 3 ranks.
SPEC = GdnShardSpec(
    sub_block_sizes=(6, 6, 12),
    sub_block_shards=((3, 2, 1), (3, 2, 1), (6, 4, 2)),
    head_shards=(4, 2, 2),
    num_heads=8,
)
HEAD_DIM, STATE = 5, 7
CONV_W = 3


def _full_conv(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(SPEC.conv_dim, CONV_W, generator=g, dtype=torch.bfloat16)


def _full_temporal(seed):
    g = torch.Generator().manual_seed(seed + 1000)
    return torch.randn(
        SPEC.num_heads, HEAD_DIM, STATE, generator=g, dtype=torch.bfloat16
    )


class TestSpecValidation(CustomTestCase):
    def test_bad_subblock_partition_refused(self):
        with self.assertRaisesRegex(KvReshardError, "do not partition"):
            GdnShardSpec(
                sub_block_sizes=(6, 6, 12),
                sub_block_shards=((3, 2, 2), (3, 2, 1), (6, 4, 2)),
                head_shards=(4, 2, 2),
                num_heads=8,
            )

    def test_bad_head_partition_refused(self):
        with self.assertRaisesRegex(KvReshardError, "partition 8 heads"):
            GdnShardSpec(
                sub_block_sizes=(6,),
                sub_block_shards=((3, 2, 1),),
                head_shards=(4, 2, 3),
                num_heads=8,
            )

    def test_rank_count_mismatch_refused(self):
        with self.assertRaisesRegex(KvReshardError, "shard entries"):
            GdnShardSpec(
                sub_block_sizes=(6,),
                sub_block_shards=((3, 3),),
                head_shards=(4, 2, 2),
                num_heads=8,
            )


class TestSegmentsAndSliceScatter(CustomTestCase):
    def test_conv_segments_tile_full_dim_exactly_once(self):
        covered = torch.zeros(SPEC.conv_dim, dtype=torch.int32)
        for r in range(SPEC.n_ranks):
            for off, n in SPEC.conv_segments(r):
                covered[off : off + n] += 1
        self.assertTrue(bool((covered == 1).all()), covered.tolist())
        # and per sub-block: rank 1's middle segment starts inside block 1
        self.assertEqual(SPEC.conv_segments(1), ((3, 2), (6 + 3, 2), (12 + 6, 4)))

    def test_slice_scatter_roundtrip_conv_and_temporal(self):
        conv, temp = _full_conv(3), _full_temporal(3)
        conv_out = torch.zeros_like(conv)
        temp_out = torch.zeros_like(temp)
        for r in range(SPEC.n_ranks):
            conv_scatter(conv_out, conv_slice(conv, SPEC, r).clone(), SPEC, r)
            temporal_scatter(
                temp_out, temporal_slice(temp, SPEC, r).clone(), SPEC, r
            )
        self.assertTrue(torch.equal(conv_out, conv))
        self.assertTrue(torch.equal(temp_out, temp))

    def test_wrong_shapes_refused(self):
        with self.assertRaisesRegex(KvReshardError, "channels"):
            conv_slice(torch.zeros(10, CONV_W), SPEC, 0)
        with self.assertRaisesRegex(KvReshardError, "heads"):
            temporal_slice(torch.zeros(5, HEAD_DIM, STATE), SPEC, 0)
        with self.assertRaisesRegex(KvReshardError, "owns"):
            conv_scatter(
                torch.zeros(SPEC.conv_dim, CONV_W),
                torch.zeros(1, CONV_W),
                SPEC,
                0,
            )


class TestPairPayloadRoundtrip(CustomTestCase):
    def test_pp_tp_pp_roundtrip_bit_exact(self):
        # Stage layers: 2 linear layers on this stage; one live request.
        conv = [_full_conv(s) for s in (10, 11)]
        temp = [_full_temporal(s) for s in (10, 11)]
        # PP->TP: build one payload per destination rank; the TP side
        # holds shards (simulated: just keep the payloads). TP->PP: each
        # rank sends its shard back; the receiving stage scatters.
        conv_re = [torch.zeros_like(c) for c in conv]
        temp_re = [torch.zeros_like(t) for t in temp]
        for r in range(SPEC.n_ranks):
            payload = pack_gdn_pair_payload(conv, temp, SPEC, r)
            unpack_gdn_pair_payload(payload, conv_re, temp_re, SPEC, r)
        for a, b in zip(conv_re, conv):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(temp_re, temp):
            self.assertTrue(torch.equal(a, b))

    def test_divergent_spec_on_receiver_goes_red(self):
        # Same totals, different shard boundaries: still a VALID spec, but
        # not the sender's -- the byte accounting must fail loudly, never
        # scatter silently-wrong channels. Can-fail proof.
        other = GdnShardSpec(
            sub_block_sizes=(6, 6, 12),
            sub_block_shards=((2, 2, 2), (2, 2, 2), (4, 4, 4)),
            head_shards=(3, 3, 2),
            num_heads=8,
        )
        conv = [_full_conv(20)]
        temp = [_full_temporal(20)]
        conv_re = [torch.zeros_like(conv[0])]
        temp_re = [torch.zeros_like(temp[0])]
        payload = pack_gdn_pair_payload(conv, temp, SPEC, 0)
        with self.assertRaises(KvReshardError):
            unpack_gdn_pair_payload(payload, conv_re, temp_re, other, 0)

    def test_layer_count_mismatch_refused(self):
        with self.assertRaisesRegex(KvReshardError, "conv layers"):
            pack_gdn_pair_payload(
                [_full_conv(1)], [_full_temporal(1), _full_temporal(2)], SPEC, 0
            )

    def test_trailing_bytes_loud(self):
        conv = [_full_conv(30)]
        temp = [_full_temporal(30)]
        payload = pack_gdn_pair_payload(conv, temp, SPEC, 1)
        padded = torch.cat([payload, torch.zeros(4, dtype=torch.uint8)])
        with self.assertRaisesRegex(KvReshardError, "trailing"):
            unpack_gdn_pair_payload(
                padded,
                [torch.zeros_like(conv[0])],
                [torch.zeros_like(temp[0])],
                SPEC,
                1,
            )


class TestRealConfigLedgerPin(CustomTestCase):
    """DESIGN_631 section 3.4b item 3: the GDN-both-layouts ledger term is
    ARITHMETIC from the model config, pinned against the measured #625
    boots -- never left derived-only (the M2 mamba-sizing lesson).

    Constants from Qwen3.6-27B-INT8-W8A8 config.json (text_config):
    linear_num_key_heads 16 x linear_key_head_dim 128 -> key_dim 2048;
    linear_num_value_heads 48 x linear_value_head_dim 128 -> value_dim
    6144; conv_dim = 2*2048 + 6144 = 10240; linear_conv_kernel_dim 4 ->
    3 state columns; temporal state [48, 128, 128]; bf16 (2 bytes);
    48 linear layers, PP split 24/12/12."""

    KEY_DIM, VALUE_DIM = 2048, 6144
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM
    CONV_COLS = 3  # kernel 4 - 1
    HEADS, HD, SS = 48, 128, 128
    ELT = 2  # bf16
    LINEAR_LAYERS = 48
    PP_LINEAR = (24, 12, 12)

    @property
    def per_layer_bytes(self):
        conv = self.CONV_DIM * self.CONV_COLS * self.ELT
        temporal = self.HEADS * self.HD * self.SS * self.ELT
        return conv, temporal

    def test_per_layer_bytes_match_625_boot(self):
        conv, temporal = self.per_layer_bytes
        # #625 PP0 boot: ssm 5.98 GiB / 169 slots / 24 layers; conv
        # 0.23 GiB / 169 / 24.
        measured_temporal = 5.98 * 2**30 / 169 / 24
        measured_conv = 0.23 * 2**30 / 169 / 24
        self.assertAlmostEqual(
            temporal / measured_temporal, 1.0, delta=0.05
        )
        self.assertAlmostEqual(conv / measured_conv, 1.0, delta=0.05)

    def test_both_layouts_16_slot_term_matches_ledger(self):
        conv, temporal = self.per_layer_bytes
        per_layer = conv + temporal
        # TP head shards proportional to the [30,17,17] vector on 48
        # heads: (22, 13, 13) is the integer split the plan derives.
        tp_heads = (22, 13, 13)
        slots = 16
        for r in range(3):
            pp_term = self.PP_LINEAR[r] * per_layer * slots
            tp_term = (
                self.LINEAR_LAYERS
                * per_layer
                * tp_heads[r]
                / self.HEADS
                * slots
            )
            both_mib = (pp_term + tp_term) / 2**20
            ledger_mib = (1146, 690, 690)[r]
            # The ledger row must stay within 15% of the config-derived
            # arithmetic; drift beyond that means the ledger is stale.
            self.assertAlmostEqual(
                both_mib / ledger_mib, 1.0, delta=0.15, msg=f"rank {r}"
            )


if __name__ == "__main__":
    unittest.main()
