# SPDX-License-Identifier: Apache-2.0
"""#1273 B4i -- THE UNCOVERED POPULATION BOOT weg2xsn15 NAMED, COVERED.

RED-FIRST ON ``dfceb7004e``, and the population is the BOOT's, not an invented
one.  Boot weg2xsn15 (``BOOT_weg2xsn15_0911.md``) is the first exchange-arm boot
to reach ``state=serving`` -- 8 flips, 6/6 prompts -- and it ran NO shadow leg
at all, because ``W84 Weg2XchgCoverageRefused`` fired genuine **34 times on P**
with ``19 finding(s)`` before the first flip and suppressed the leg.  Its
``WEG2-XCHG-UNCOVERED`` table (291 lines on D, 99 on P) named the population for
the first time:

==============================================  ======  ==========================
tensor                                          lines   what it is
==============================================  ======  ==========================
``model.layers.N.linear_attn.conv1d.weight``       144  GDN causal conv, ``[C,1,K]``
``model.layers.N.linear_attn.attn.conv_weights``   144  a ``.view`` of the SAME storage
``visual.patch_embed.proj.weight``                   3  ``[1152,3,2,16,16]``, 5-D
==============================================  ======  ==========================

**RE-STAMP 3's premise is REFUTED BY THAT BOOT.**  The fix shipped in B4g
assumed these were all-zerofill padded vocabulary rows that had vanished from
the plan map, so it declared such names with 0 bytes and exempted them BY NAME
(``is_zerofill_by_design``, ``EXEMPT_REASON``).  The boot read ``exempt=0`` on
every one of the 40 COVER lines and ``reason=zerofill`` **0** times: these are
real, non-zero conv weights plus a 5-D vision projection, and no zerofill test
can exempt them.  So this slice does NOT widen the exemption -- it COVERS the
population, which is the only fix that makes ``uncovered=0`` a real zero rather
than a relabel.  (The exemption stays exactly where it was, for genuine
zerofill; a test below pins that it still works.)

**THE ROOT, and it is ONE condition.**  ``ParamGeom.of`` refuses every tensor
with more than two dimensions unless the caller names a shard axis a
``(rows, cols, pitch)`` triple can express.  The product inventory
(``weight_exchange_shadow.derive_leg_plan``, and ``derive_card_manifest``
beside it) builds every geometry with ``shard_axis=REPLICATED`` -- the PP form:
whole tensors, one holder each, no shard arithmetic anywhere -- so there IS no
shard axis to name, and the refusal was firing on a question it was not asked.
Those parameters were then dropped from the inventory (counted as
``undescribed=``), never reached ``build_plan``, never reached the plan map, and
came back out of ``build_coverage`` as ``uncovered``.  ``StorageGeom.of``
already flattens a genuinely contiguous block to storage rows and already
refuses a non-contiguous one BY NAME, so the replicated case needs nothing
else -- and the sharded case is untouched, which the GONE tests below pin.

The 144 aliasing ``attn.conv_weights`` lines need no separate fix and must not
get one: ``models/qwen3_5.py:380`` builds it as
``self.conv1d.weight.view(size(0), size(2))``, so it shares the parameter's
storage, and ``build_coverage`` judges an ATTRIBUTE as covered iff its storage
key is one an accounted-for parameter or buffer already claimed.  Covering the
parameter covers the view.  A test below asserts exactly that, so a future
change that gives the attribute its own descriptor is visible as a
double-charge rather than silent.

Hermetic: real CPU tensors, ``CUDA_VISIBLE_DEVICES=""``, no torch_memory_saver,
no checkpoint, no region.  The shapes are the boot's own families, scaled down;
every byte assertion is against ``untyped_storage().nbytes()``, never against a
literal.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

MIB = 1024 * 1024

#: The boot's own GDN conv shape, scaled: ``[C, 1, K]`` after
#: ``models/qwen3_5.py:314``'s ``unsqueeze(1)``.  K=4 is the real kernel width.
CONV_C, CONV_K = 768, 4
#: The boot's own 5-D vision patch embedding, scaled on the output channel
#: only: the real one is ``[1152, 3, 2, 16, 16]``, which is
#: ``Conv3dLayer``'s ``[out, in, kT, kH, kW]`` (``layers/conv.py``).
PATCH_SHAPE = (16, 3, 2, 16, 16)


# ---------------------------------------------------------------------------
# The double: the boot's three families in one module tree.
# ---------------------------------------------------------------------------


class _LinearAttnInner(nn.Module):
    """Where ``attn.conv_weights`` lives, and it is a plain ATTRIBUTE.

    ``layers/radix_linear_attention.py:75`` assigns it with ``self.conv_weights
    = conv_weights``, so it is neither a Parameter nor a registered buffer and
    appears in NO standard iterator -- only in ``walk_live_tensors``' third
    population (every module's ``__dict__``).
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        # models/qwen3_5.py:380 -- a VIEW of the parameter's storage.
        self.conv_weights = weight.view(weight.size(0), weight.size(2))


class _LinearAttn(nn.Module):
    def __init__(self):
        super().__init__()
        # models/qwen3_5.py:301-314: a ColumnParallelLinear whose weight is
        # then unsqueezed to [C, 1, K] in place.
        self.conv1d = nn.Conv1d(CONV_C, CONV_C, CONV_K, groups=CONV_C, bias=False)
        self.conv1d.weight = nn.Parameter(
            torch.zeros(CONV_C, 1, CONV_K, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.in_proj_qkv = nn.Parameter(
            torch.zeros(256, 512, dtype=torch.bfloat16), requires_grad=False
        )
        self.attn = _LinearAttnInner(self.conv1d.weight)


class _Layer(nn.Module):
    def __init__(self, gdn: bool):
        super().__init__()
        if gdn:
            self.linear_attn = _LinearAttn()
        else:
            # The full-attention slot of the GDN interleave -- layer 3 of the
            # boot's {0,1,2,4,5,6} per tag, which carries no conv at all.
            self.self_attn = nn.Module()
            self.self_attn.qkv_proj = nn.Parameter(
                torch.zeros(384, 512, dtype=torch.bfloat16), requires_grad=False
            )


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Module()
        self.proj.weight = nn.Parameter(
            torch.zeros(*PATCH_SHAPE, dtype=torch.bfloat16), requires_grad=False
        )


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()


class _Model(nn.Module):
    """The boot's population in miniature: a GDN interleave with one
    full-attention slot, a base-tag embedding, and the vision tower."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Layer(gdn=(i != 3)) for i in range(n_layers)]
        )
        self.model.embed_tokens = nn.Parameter(
            torch.zeros(1024, 512, dtype=torch.bfloat16), requires_grad=False
        )
        self.visual = _Visual()


def _live_bytes(model) -> dict:
    return {
        name: int(p.untyped_storage().nbytes())
        for name, p in model.named_parameters()
    }


def _tag_bytes_stub(_tag: str) -> int:
    """A ``tms_tag_bytes``-shaped callable that answers 0 -- ``slack_mib`` is
    printed, never compared, so the census does not need a saver."""
    return 0


class _ChunkedCase(unittest.TestCase):
    """The launcher's chunk geometry as the Weg-2 form publishes it."""

    def setUp(self):
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"

    def tearDown(self):
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)


# ===========================================================================
# THE ROOT: a REPLICATED geometry has no shard axis to name.
# ===========================================================================


class ReplicatedNdimGeometry(unittest.TestCase):
    def test_a_replicated_three_d_conv_weight_is_nameable(self):
        """``[C, 1, K]``, the boot's 288 conv lines, through the PRODUCT call.

        The argument list is the inventory loop's own
        (``weight_exchange_shadow.py:3166``): ``shard_axis=REPLICATED``,
        ``shard_total=0``, ``stage=rank``, NO ``shard_dim`` -- because a
        replicated parameter has no shard axis for the caller to name.
        """
        t = torch.zeros(CONV_C, 1, CONV_K, dtype=torch.bfloat16)
        geom = wx.ParamGeom.of(
            t,
            name="model.layers.0.linear_attn.conv1d.weight",
            tag="weights_0",
            shard_axis=wx.REPLICATED,
            shard_total=0,
            stage=0,
        )
        # The flattening is STORAGE's, not shape's: C*1 rows of K elements.
        self.assertEqual((geom.rows_full, geom.cols_full), (CONV_C, CONV_K))
        self.assertEqual(
            geom.rows_full * geom.cols_full * geom.itemsize,
            int(t.untyped_storage().nbytes()),
        )

    def test_a_replicated_five_d_patch_embedding_is_nameable(self):
        """``[1152, 3, 2, 16, 16]`` -- the boot's ``uncovered=1`` on the base
        tag, and the one tensor whose byte count a wrong flattening would get
        wrong by a factor rather than by a refusal."""
        t = torch.zeros(*PATCH_SHAPE, dtype=torch.bfloat16)
        geom = wx.ParamGeom.of(
            t,
            name="visual.patch_embed.proj.weight",
            tag=GPU_MEMORY_TYPE_WEIGHTS,
            shard_axis=wx.REPLICATED,
            shard_total=0,
            stage=0,
        )
        rows = PATCH_SHAPE[0] * PATCH_SHAPE[1] * PATCH_SHAPE[2] * PATCH_SHAPE[3]
        self.assertEqual((geom.rows_full, geom.cols_full), (rows, PATCH_SHAPE[4]))
        self.assertEqual(
            geom.rows_full * geom.cols_full * geom.itemsize,
            int(t.untyped_storage().nbytes()),
        )
        # THE CAN-FAIL, and it is the mutant this test exists for: a 5-D tensor
        # flattened on the leading axis alone claims 1/(3*2*16) of its bytes.
        self.assertNotEqual(geom.rows_full, PATCH_SHAPE[0])

    def test_a_replicated_ndim_parameter_emits_descriptors_for_every_byte(self):
        """The geometry is not the point; the DESCRIPTORS are.

        A geometry the plan accepts and then emits nothing for would still be
        uncovered, and ``build_plan`` would refuse it by name (W74). This is
        the byte identity the plan map is summed from.
        """
        t = torch.zeros(CONV_C, 1, CONV_K, dtype=torch.bfloat16)
        geom = wx.ParamGeom.of(
            t,
            name="model.layers.0.linear_attn.conv1d.weight",
            tag="weights_0",
            shard_axis=wx.REPLICATED,
            shard_total=0,
            stage=0,
        )
        cards = tuple(range(3))
        src = wx.GroupLayout(name="P", cards=cards, tp_size=1, base=0)
        dst = wx.GroupLayout(name="D", cards=cards, tp_size=1, base=len(cards))
        plan = wx.build_plan([geom], src, dst, waves=[["weights_0"]])
        got = wx.plan_bytes_from_descs(plan.raw_descs)
        self.assertEqual(
            got["weights_0"]["model.layers.0.linear_attn.conv1d.weight"],
            int(t.untyped_storage().nbytes()),
        )


# ===========================================================================
# GONE=0: what the refusal was FOR is still refused.
# ===========================================================================


class TheShardedRefusalStands(unittest.TestCase):
    def test_an_expert_major_moe_weight_is_still_refused(self):
        """``[E, N_local, K]`` sharded on N is E strided bands and no
        (rows, cols, pitch) triple names it. The refusal's own example, and it
        is a SHARDED geometry, so this slice must not touch it."""
        moe = torch.empty((4, 128, 64), dtype=torch.int8, device="meta")
        with self.assertRaises(wx.Weg2XchgPlanDisagree) as cm:
            wx.ParamGeom.of(
                moe,
                name="model.layers.0.mlp.experts.w13_weight",
                tag="weights_0",
                shard_axis=wx.ROWS,
                shard_total=512,
            )
        self.assertIn("W68 Weg2XchgPlanDisagree", str(cm.exception))

    def test_a_named_leading_axis_shard_is_still_accepted(self):
        """The other half of the original refusal: ``shard_dim=0`` names the
        axis and the sharded conv is planned as before."""
        conv = torch.empty((768, 1, 4), dtype=torch.int8, device="meta")
        geom = wx.ParamGeom.of(
            conv,
            name="model.layers.0.linear_attn.conv1d.weight",
            tag="weights_0",
            shard_axis=wx.ROWS,
            shard_total=768 * 3,
            shard_dim=0,
        )
        self.assertEqual((geom.rows_full, geom.cols_full), (768 * 3, 4))

    def test_a_non_contiguous_replicated_ndim_tensor_is_STILL_refused(self):
        """THE DANGER DIRECTION of this slice, and the reason it is safe.

        Admitting the replicated case must not admit a tensor whose storage has
        no rows a copy primitive can name -- that would flatten a permuted view
        to the WRONG extent and plan a copy of the wrong bytes with no refusal
        anywhere. ``StorageGeom.of`` is the one that answers, and it answers
        with W68.
        """
        t = torch.zeros(8, 4, 16, dtype=torch.bfloat16).permute(2, 0, 1)
        self.assertFalse(t.is_contiguous())
        with self.assertRaises(wx.Weg2XchgPlanDisagree) as cm:
            wx.ParamGeom.of(
                t,
                name="model.layers.0.linear_attn.conv1d.weight",
                tag="weights_0",
                shard_axis=wx.REPLICATED,
                shard_total=0,
                stage=0,
            )
        msg = str(cm.exception)
        self.assertIn("W68 Weg2XchgPlanDisagree", msg)
        self.assertIn("not a contiguous block", msg)


# ===========================================================================
# END TO END through the PRODUCT producer: the plan map, then the census.
# ===========================================================================


class TheBootsPopulationIsCovered(_ChunkedCase):
    """Driven through ``derive_leg_plan`` -- the producer the plan provider
    consumes (``default_plan_provider``) and the one both ends of the shadow
    use -- so what this asserts is the plan the boot will derive, not a second
    derivation written for the test."""

    def _plan_map(self, model, *, rank=0):
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        plan, reason = sh.derive_leg_plan(
            hook=sh.HOOK_SOURCE, group="P", peer="D", rank=rank, model=model
        )
        self.assertIsNotNone(plan, f"derivation refused: {reason}")
        return plan, wx.plan_bytes_from_descs(plan.descs)

    def test_every_parameter_is_in_the_plan_map_with_its_real_bytes(self):
        """``uncovered`` is downstream of THIS: a parameter absent from the
        plan map is an uncovered page by construction."""
        model = _Model()
        plan, got = self._plan_map(model)
        planned = {name: n for tag in got for name, n in got[tag].items()}
        live = _live_bytes(model)
        self.assertEqual(sorted(planned), sorted(live))
        for name, nbytes in live.items():
            self.assertEqual(planned[name], nbytes, name)
        # The acceptance line's own counter: nothing was dropped for being
        # unnameable.
        self.assertEqual(int(plan.undescribed), 0)

    def test_the_conv_weight_is_planned_under_its_LAYER_tag(self):
        """A descriptor under the wrong tag covers nothing: ``build_coverage``
        asks per tag, so a conv weight charged to the base tag leaves the layer
        tag uncovered AND makes the base tag's claim exceed its storage."""
        model = _Model()
        _plan, got = self._plan_map(model)
        name = "model.layers.0.linear_attn.conv1d.weight"
        self.assertIn(name, got["weights_0"])
        self.assertNotIn(name, got.get(GPU_MEMORY_TYPE_WEIGHTS, {}))
        self.assertIn("visual.patch_embed.proj.weight",
                      got[GPU_MEMORY_TYPE_WEIGHTS])

    def test_coverage_is_complete_and_nothing_is_exempted(self):
        """THE BOOT'S OWN ACCEPTANCE LINE: ``uncovered=0`` on every rank x tag
        row, with ``exempt=0`` -- a zero that is a real zero and not a relabel
        of the same tensors under a zerofill reason."""
        model = _Model()
        _plan, got = self._plan_map(model)
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=got, tag_bytes=_tag_bytes_stub
        )
        self.assertTrue(rows)
        for tag, row in rows.items():
            self.assertEqual(
                [t.name for t in row.uncovered], [], f"{tag}: uncovered"
            )
            self.assertEqual([s.name for s in row.short], [], f"{tag}: short")
            self.assertEqual(list(row.missing), [], f"{tag}: missing")
            self.assertEqual(list(row.exempt), [], f"{tag}: exempt")
            self.assertTrue(row.ok, tag)

    def test_the_aliasing_attribute_is_covered_BY_ITS_PARAMETERS_STORAGE(self):
        """``attn.conv_weights`` is 144 of the boot's 291 D lines and needs no
        descriptor of its own.

        It is a ``.view`` of ``conv1d.weight`` (``models/qwen3_5.py:380``), so
        covering the parameter covers it -- and the plan must NOT name it
        separately, which would charge the same storage twice.
        """
        model = _Model()
        _plan, got = self._plan_map(model)
        alias = "model.layers.0.linear_attn.attn.conv_weights"
        param = "model.layers.0.linear_attn.conv1d.weight"
        self.assertNotIn(alias, got["weights_0"])
        live = wx.walk_live_tensors(model)
        by_name = {t.name: t for t in live}
        self.assertEqual(by_name[alias].kind, wx.ATTRIBUTE)
        self.assertEqual(by_name[alias].storage_key, by_name[param].storage_key)
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=got, tag_bytes=_tag_bytes_stub
        )
        self.assertEqual([t.name for t in rows["weights_0"].uncovered], [])
        # And the claim is the storage ONCE, not twice.
        self.assertEqual(
            rows["weights_0"].planned_bytes,
            sum(
                int(p.untyped_storage().nbytes())
                for n, p in model.named_parameters()
                if wx.tag_of_parameter_name(n) == "weights_0"
            ),
        )

    def test_the_zerofill_exemption_still_works_for_genuine_zerofill(self):
        """The exemption is NOT removed, only stopped from being the answer to
        this population: a parameter the plan really does declare with zero
        planned bytes is still accounted for, named and given its reason."""
        model = _Model()
        _plan, got = self._plan_map(model)
        name = "model.layers.0.linear_attn.conv1d.weight"
        got["weights_0"][name] = 0
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=got, tag_bytes=_tag_bytes_stub
        )
        row = rows["weights_0"]
        self.assertEqual(list(row.exempt), [name])
        self.assertEqual([t.name for t in row.uncovered], [])
        self.assertIn("reason=", row.cover_line())


if __name__ == "__main__":
    unittest.main()
