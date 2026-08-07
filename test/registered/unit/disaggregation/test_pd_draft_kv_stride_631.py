# SPDX-License-Identifier: Apache-2.0
"""The draft KV pool's treatment at the PD transfer boundary is NOT identical
to the target pool's, and the difference is silent (#631).

WHY A FALSIFIER AND NOT AN ARGUMENT
-----------------------------------
DESIGN_631b §0a owed item (1) asks whether the draft pool's DCP treatment is
"genuinely identical to the target's at the transfer boundary, not merely
similar", and settles it BY CONSTRUCTION: under ``--draft-kv-layout dcp`` the
draft pool carries ``get_total_num_kv_heads()`` exactly as the target does, so
per-rank item lengths agree. That reasoning is correct about HEAD COUNTS and
says nothing about the two places the boundary actually reads:

1. ``mooncake/conn.py:2114`` -- the decode arm advertises ONE scalar item
   length for its whole registration, ``kv_args.kv_item_lens[0]``, i.e. the
   first TARGET layer. The appended draft layers' item lengths are never
   transmitted.
2. ``mooncake/conn.py:1517-1528`` -- the only stride guard compares that
   scalar against the prefill arm's ``kv_item_lens[0]``. Also index 0. A
   divergence confined to the appended draft layers passes it.

and one place the boundary computes addresses:

3. ``common/conn.py:736 get_mha_kv_ptrs_with_pp`` -- splits the FLAT
   registration list in half to recover K and V pointers. The target pool
   registers ``k_buffer + v_buffer`` (``memory_pool.py:2483``), so the halves
   are exact. ``prefill.py:186-195`` / ``decode.py:439-448`` then APPEND the
   draft pool's buffers to that same flat list, and the half-split has no
   notion that they are there.

This file exercises (3) with the REAL function on synthetic registrations, and
(1)+(2) with the real arithmetic restated once. Each pointer is a labelled
token, so the oracle is not "does it crash" but "is every source buffer paired
with the destination buffer holding the SAME layer and the SAME role".

THE CONTROL ARM IS THE POINT
----------------------------
``test_pp_prefill_without_draft_pairs_correctly`` must PASS. It is the same
PP geometry as the failing case with the draft pool removed, and it is what
makes a failure elsewhere evidence rather than an artefact of the harness: an
oracle that reports a mismatch for every input proves nothing.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Labelled registrations: a "pointer" is an int, and LABELS maps it back to a
# human-readable role so a mispairing names what got crossed with what.
# --------------------------------------------------------------------------


class Registration:
    """One arm's flat (ptrs, item_lens) exactly as the disagg code builds it."""

    def __init__(self):
        self.ptrs = []
        self.item_lens = []
        self.labels = {}

    def _add(self, label, item_len):
        ptr = 0x1000 + len(self.ptrs) * 0x100
        # Pointers must be unique across an arm for the oracle to be able to
        # tell buffers apart; the offset above guarantees it.
        self.ptrs.append(ptr)
        self.item_lens.append(item_len)
        self.labels[ptr] = label
        return ptr

    @classmethod
    def build(cls, layer_ids, target_item_len, draft_item_len=None):
        """``k_buffer + v_buffer`` (memory_pool.py:2483), then the draft pool
        appended as extra layers (prefill.py:186-195 / decode.py:439-448)."""
        r = cls()
        for lid in layer_ids:
            r._add(f"K{lid}", target_item_len)
        for lid in layer_ids:
            r._add(f"V{lid}", target_item_len)
        if draft_item_len is not None:
            r._add("Kd", draft_item_len)
            r._add("Vd", draft_item_len)
        return r


def _manager_for(prefill_start_layer):
    """``get_mha_kv_ptrs_with_pp`` reads exactly one field off ``self``."""
    return SimpleNamespace(
        kv_args=SimpleNamespace(prefill_start_layer=prefill_start_layer)
    )


def pair_roles(src_reg, dst_reg, prefill_start_layer):
    """Run the REAL split and return the (src_label, dst_label) pairs it forms.

    Mirrors ``_send_kvcache_generic``'s ``layers_params`` construction: K pairs
    take ``item_lens[i]``, V pairs take ``item_lens[stage + i]``.
    """
    mgr = _manager_for(prefill_start_layer)
    src_k, src_v, dst_k, dst_v, stage = CommonKVManager.get_mha_kv_ptrs_with_pp(
        mgr, src_reg.ptrs, dst_reg.ptrs
    )
    pairs = []
    for i in range(stage):
        pairs.append(
            (
                src_reg.labels.get(src_k[i], f"<oob {i}>"),
                dst_reg.labels.get(dst_k[i], f"<oob {i}>"),
                src_reg.item_lens[i],
            )
        )
    for i in range(stage):
        pairs.append(
            (
                src_reg.labels.get(src_v[i], f"<oob {i}>"),
                dst_reg.labels.get(dst_v[i], f"<oob {i}>"),
                src_reg.item_lens[stage + i],
            )
        )
    return pairs


def mispairings(pairs):
    return [(s, d) for s, d, _ in pairs if s != d]


TARGET_ITEM = 4096
DRAFT_ITEM = 4096

# A 48-layer target model over a 3-stage pipeline.
ALL_LAYERS = list(range(48))
STAGES = [(0, list(range(0, 16))), (16, list(range(16, 32))), (32, list(range(32, 48)))]


class ControlArm(CustomTestCase):
    """The oracle must be able to say PASS, or its failures mean nothing."""

    def test_monolithic_no_draft_pairs_correctly(self):
        src = Registration.build(ALL_LAYERS, TARGET_ITEM)
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM)
        self.assertEqual(mispairings(pair_roles(src, dst, 0)), [])

    def test_pp_prefill_without_draft_pairs_correctly(self):
        """CONTROL: the exact PP geometry of the failing case, draft removed.

        This is arm 1 of the #631 boot recipe -- PP prefill group, speculation
        still refused. It must pair perfectly, which is what licenses reading
        the draft-pool arm below as a real defect.
        """
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM)
        for start, layers in STAGES:
            with self.subTest(stage_start=start):
                src = Registration.build(layers, TARGET_ITEM)
                self.assertEqual(mispairings(pair_roles(src, dst, start)), [])

    def test_monolithic_with_draft_on_both_arms_pairs_correctly(self):
        """Both arms carry a draft pool and the flat lists are equal length.

        The half-split mislabels V0 as a K pointer on BOTH sides identically,
        so the pairing survives. Recorded because it is the reason the defect
        below is invisible on a non-PP pair: the mislabel is symmetric exactly
        when the two lists have the same length.
        """
        src = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        self.assertEqual(mispairings(pair_roles(src, dst, 0)), [])


class PipelinedDraftPoolIsMispaired(CustomTestCase):
    """The defect: a PP prefill arm that also registers a draft pool.

    W1 of DESIGN_631b REQUIRES this combination -- "the prefill arm must LOAD
    the draft layer's weights even though it never drafts", and
    ``prefill.py:164-165`` appends the draft pool on every stage whenever
    layer-sharding is off (which PP does not turn on). So the #631 topology
    walks straight into it.
    """

    def test_pp_prefill_with_draft_mispairs_silently(self):
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        found_any = False
        for start, layers in STAGES:
            src = Registration.build(layers, TARGET_ITEM, DRAFT_ITEM)
            bad = mispairings(pair_roles(src, dst, start))
            if bad:
                found_any = True
        self.assertTrue(
            found_any,
            "expected the PP-plus-draft registration to mispair; if this "
            "assertion fails the defect has been fixed and this test should "
            "be inverted into a regression guard",
        )

    def test_no_exception_is_raised_on_the_mispaired_path(self):
        """Silence is the hazard: the split returns, it does not refuse."""
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        src = Registration.build(STAGES[0][1], TARGET_ITEM, DRAFT_ITEM)
        # No assertRaises: the point is that this simply returns.
        pairs = pair_roles(src, dst, STAGES[0][0])
        self.assertTrue(len(pairs) > 0)


class StrideGuardCoversOnlyLayerZero(CustomTestCase):
    """(1) and (2): the item-length channel is one scalar wide.

    ``conn.py:2114`` sends ``kv_item_lens[0]``; ``conn.py:1517-1528`` compares
    ``kv_item_lens[0]``. Both indices are literal. A draft pool whose stride
    differs between the arms is therefore invisible to the only guard there
    is.
    """

    @staticmethod
    def guard_would_refuse(prefill_reg, decode_reg):
        """The guard as written at mooncake/conn.py:1517-1528."""
        advertised = decode_reg.item_lens[0]  # conn.py:2114
        return bool(prefill_reg.item_lens) and advertised != prefill_reg.item_lens[0]

    def test_guard_catches_a_target_stride_divergence(self):
        """CAN-FAIL ARM: the guard does fire when layer 0 diverges."""
        pre = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        dec = Registration.build(ALL_LAYERS, TARGET_ITEM * 2, DRAFT_ITEM)
        self.assertTrue(self.guard_would_refuse(pre, dec))

    def test_guard_misses_a_draft_only_stride_divergence(self):
        """THE DEFECT: identical target strides, divergent DRAFT strides.

        This is not a hypothetical skew. Step 1 established that on the #631
        topology the two arms are FORCED onto different --draft-kv-layout
        values (the PP prefill arm cannot take 'dcp'; the token-sharded decode
        arm cannot take 'replicated'), and the layout is what decides the
        draft pool's geometry.
        """
        pre = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        dec = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM * 3)
        self.assertFalse(
            self.guard_would_refuse(pre, dec),
            "if this now refuses, the stride guard has been widened past "
            "index 0 and this test should become a regression guard",
        )


if __name__ == "__main__":
    unittest.main()
