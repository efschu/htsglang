"""H88-C (PLAN-H88-W4A8-1007 D5/D6): the expert weight LAYOUT as a boot-wide fact.

Units under test (new with H88-C, all CPU, no CUDA call):
  moe/moe_w4a8_layout.py    layout selection, one-layout-per-boot, the #323b offload guard, byte census
  moe/expert_store.py       compute_identity's layout tag; the ``*.factor.json`` scale-factor sidecar
                            (first-writer-wins agreement, adoption, foreign-identity replacement)
  moe/expert_offload.py     the conditional admission of CompressedTensorsWNA16A8MoE behind its marker
  weg2/launcher.py          moe_expert_layout_of_launch (named refusal P != D), publish_store_identity wiring

Why (H88 facts file 1007, 4): the W4A16 and W4A8 layouts have the SAME tensor names and shapes but
different bytes -- mixing them is silent. Hence: the layout is hashed into the store identity (D5), one
layout per boot is a launch-time refusal, the offload cache must name every expert-major tensor of the
new layout (#323b), and where processes share rows the per-tensor int16 scale FACTOR is agreed through
the store, never derived per process.

D6 / W71 census: the byte table comes from the repack shapes -- both layouts' packed tensors are
int32 [E, K/16, 2N] (gptq_marlin_repack: N*(num_bits//2) nibbles = N*16/8 nibbles), the scales are
[E, G, N] elements in both (permutation + int16 bit pattern, same element size); the W4A8 layout adds
exactly two float32 0-d factors per MoE layer (+8 B/layer). A census measured on the A16 layout prices
W4A8 within 8 bytes per layer; :func:`census_compare` produces that table (test pins the NF geometry).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import hashlib
import json
import os
import shutil
import struct
import tempfile
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_store as es
from sglang.srt.layers.moe import moe_w4a8_layout as MWL
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16a8_moe as S,
)

SW = MWL.SWITCH_ENV


def _clean_env(**extra):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (SW, "SGLANG_MOE_ACT_INT8", "SGLANG_WEG2_GROUP", "SGLANG_MOE_EXPERT_STORE_DIR",
                "SGLANG_MOE_EXPERT_STORE_IDENTITY")
    }
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# layout selection (the switch's two spellings, OR semantics)
# ---------------------------------------------------------------------------


class TestGroupLayout(unittest.TestCase):
    def test_default_is_a16(self):
        self.assertEqual(MWL.group_layout("", "", {}), MWL.LAYOUT_W4A16)

    def test_env_spec_switches_on_and_off(self):
        self.assertEqual(MWL.group_layout(f"x=1;{SW}=1", "", {}), MWL.LAYOUT_W4A8)
        self.assertEqual(MWL.group_layout(f"{SW}=true", "", {}), MWL.LAYOUT_W4A8)
        self.assertEqual(MWL.group_layout(f"{SW}=0", "", {}), MWL.LAYOUT_W4A16)
        # the group env spec wins over the launcher env (it is what the rank inherits)
        self.assertEqual(MWL.group_layout(f"{SW}=0", "", {SW: "1"}), MWL.LAYOUT_W4A16)
        self.assertEqual(MWL.group_layout("", "", {SW: "1"}), MWL.LAYOUT_W4A8)

    def test_flag_spellings(self):
        self.assertEqual(MWL.group_layout("", "--moe-act-int8 on", {}), MWL.LAYOUT_W4A8)
        self.assertEqual(MWL.group_layout("", "--moe-act-int8=on", {}), MWL.LAYOUT_W4A8)
        self.assertEqual(MWL.group_layout("", "--moe-act-int8 off", {}), MWL.LAYOUT_W4A16)
        self.assertEqual(MWL.group_layout("", "--moe-act-int8 on --moe-act-int8 off", {}), MWL.LAYOUT_W4A16)

    def test_or_semantics_flag_off_does_not_undo_env_on(self):
        # moe_act_int8_requested: either spelling switches it ON; a flag "off" does not undo an env "1"
        self.assertEqual(MWL.group_layout(f"{SW}=1", "--moe-act-int8 off", {}), MWL.LAYOUT_W4A8)

    def test_bad_value_refuses_by_name(self):
        with self.assertRaisesRegex(MWL.MoeLayoutMismatch, "not a boolean"):
            MWL.group_layout(f"{SW}=manchmal", "", {})


class TestOneLayoutPerBoot(unittest.TestCase):
    def _ns(self, **kw):
        base = dict(env_p="", extra_p="", env_d="", extra_d="")
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_equal_layouts_pass(self):
        self.assertIsNone(MWL.check_one_layout_per_boot(MWL.LAYOUT_W4A8, MWL.LAYOUT_W4A8))
        self.assertIsNone(MWL.check_one_layout_per_boot(MWL.LAYOUT_W4A16, MWL.LAYOUT_W4A16))

    def test_mixed_boot_refuses_naming_both(self):
        with self.assertRaises(MWL.MoeLayoutMismatch) as cm:
            MWL.check_one_layout_per_boot(MWL.LAYOUT_W4A8, MWL.LAYOUT_W4A16)
        msg = str(cm.exception)
        self.assertIn(MWL.REFUSAL, msg)
        self.assertIn(MWL.LAYOUT_W4A8, msg)
        self.assertIn(MWL.LAYOUT_W4A16, msg)

    def test_launch_layout(self):
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            self.assertEqual(MWL.launch_layout(self._ns()), (MWL.LAYOUT_W4A16,) * 3)
            self.assertEqual(
                MWL.launch_layout(self._ns(env_p=f"{SW}=1", env_d=f"{SW}=1")),
                (MWL.LAYOUT_W4A8,) * 3,
            )
            with self.assertRaises(MWL.MoeLayoutMismatch):
                MWL.launch_layout(self._ns(extra_p="--moe-act-int8 on"))
            # launcher env alone switches BOTH groups (ranks inherit it): one layout, no refusal
            self.assertEqual(
                MWL.launch_layout(self._ns(), base_env={SW: "1"}), (MWL.LAYOUT_W4A8,) * 3
            )


# ---------------------------------------------------------------------------
# the #323b offload guard
# ---------------------------------------------------------------------------


class TestOffloadCoversW4A8(unittest.TestCase):
    def test_real_offload_tuple_covers_both_sym_and_asym(self):
        attrs = list(eo.MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS)  # fmt: skip
        MWL.assert_offload_covers_w4a8(attrs, sym=True)
        MWL.assert_offload_covers_w4a8(attrs, sym=False)

    def test_missing_expert_major_tensor_names_it(self):
        attrs = [n for n in MWL.expert_major_names(True) if n != "w13_weight_scale"]
        with self.assertRaisesRegex(RuntimeError, "w13_weight_scale"):
            MWL.assert_offload_covers_w4a8(attrs, sym=True)

    def test_asym_zero_points_are_required(self):
        attrs = list(MWL.expert_major_names(True))  # no zero points
        MWL.assert_offload_covers_w4a8(attrs, sym=True)
        with self.assertRaisesRegex(RuntimeError, "w13_weight_zero_point"):
            MWL.assert_offload_covers_w4a8(attrs, sym=False)

    def test_layer_global_factor_in_tuple_refuses(self):
        attrs = list(MWL.expert_major_names(True)) + ["w13_act_scale_factor"]
        with self.assertRaisesRegex(RuntimeError, "layer-global"):
            MWL.assert_offload_covers_w4a8(attrs, sym=True)


class TestOffloadAdmissionMarker(unittest.TestCase):
    """The guard admits CompressedTensorsWNA16A8MoE only on a layer whose scheme set the covered mark."""

    def _a8_scheme(self):
        cls = type("CompressedTensorsWNA16A8MoE", (), {})
        return cls()

    def test_unmarked_layer_refused(self):
        layer = types.SimpleNamespace(layer_id=0)
        with self.assertRaisesRegex(RuntimeError, "w4a8_covered"):
            eo.assert_expert_offload_quant_supported(
                None, layer_id=0, scheme=self._a8_scheme(), layer=layer
            )

    def test_marked_layer_admitted(self):
        layer = types.SimpleNamespace(layer_id=0, **{MWL.OFFLOAD_COVERED_ATTR: True})
        eo.assert_expert_offload_quant_supported(
            None, layer_id=0, scheme=self._a8_scheme(), layer=layer
        )


# ---------------------------------------------------------------------------
# row cut: w4a8_repack_moe_weights(rows=...)
# ---------------------------------------------------------------------------


class TestRepackRowsUnit(unittest.TestCase):
    @staticmethod
    def _fake(p, k, n, b):
        # stand-in with the W4A8 kernel's shape contract: int32 [rows, k/16, 2n]
        return torch.zeros((p.shape[0], k // 16, 2 * n), dtype=torch.int32)

    def test_rows_none_equals_explicit_full_rows(self):
        packed = torch.zeros(3, 16, 128, dtype=torch.int32)
        full = S.w4a8_repack_moe_weights(packed, 8, 4, repack_fn=self._fake)
        explicit = S.w4a8_repack_moe_weights(packed, 8, 4, repack_fn=self._fake, rows=[0, 1, 2])
        self.assertEqual(full.shape, explicit.shape)
        torch.testing.assert_close(full, explicit, rtol=0, atol=0)

    def test_subset_fills_kept_rows_and_keeps_the_shape(self):
        packed = torch.arange(3 * 16 * 128, dtype=torch.int32).reshape(3, 16, 128)
        out = S.w4a8_repack_moe_weights(packed, 8, 4, repack_fn=self._fake, rows=[2])
        self.assertEqual(tuple(out.shape), (3, 8, 256))
        ref = self._fake(packed[2:3], 128, 128, 4)
        torch.testing.assert_close(out[2], ref[0], rtol=0, atol=0)

    def test_empty_rows_shape_contract(self):
        packed = torch.zeros(3, 16, 128, dtype=torch.int32)
        out = S.w4a8_repack_moe_weights(packed, 8, 4, repack_fn=self._fake, rows=[])
        self.assertEqual(tuple(out.shape), (3, 8, 256))  # nothing repacked, shape still right


# ---------------------------------------------------------------------------
# uneven expert shard: the whole-expert shard unit and the layout (D5 item 3)
# ---------------------------------------------------------------------------


class TestUnevenExpertShardContract(unittest.TestCase):
    """SGLANG_UNEVEN_MOE_EXPERT_SHARD (fused_moe_triton/layer.py:578-632) shards WHOLE experts per rank.

    For the W4A8 layout that is tile-compatible by construction: every Marlin tile lives inside one expert's
    [K/16, 2N] plane, so dim-0 slicing never cuts a tile -- and the per-tensor scale factor, which does NOT
    live inside one expert, is agreed through the store sidecar instead of per rank (the tests below pin the
    eligibility predicate, the transposing-method registration, and the sidecar key's rank-independence).
    """

    def _qcfg(self, name):
        m = mock.Mock()
        m.get_name.return_value = name
        return m

    def test_generic_expert_shard_eligible_for_compressed_tensors(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            expert_shard_generic_eligible,
        )

        ct = self._qcfg("compressed-tensors")
        self.assertTrue(expert_shard_generic_eligible(ct, True, 1, True))
        self.assertFalse(expert_shard_generic_eligible(self._qcfg("gguf"), True, 1, True))
        self.assertFalse(expert_shard_generic_eligible(ct, False, 1, True))  # no uneven plan
        self.assertFalse(expert_shard_generic_eligible(ct, True, 2, True))  # EP: refused
        self.assertFalse(expert_shard_generic_eligible(ct, True, 1, False))  # not opted in

    def test_a8_scheme_is_a_transposing_method(self):
        # the worker/consumer transpose predicate (the #68a single source) must name the W4A8 scheme;
        # without it the shard worker would transpose rows the scheme never asked about
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _CT_TRANSPOSING_METHODS,
        )

        self.assertIn("CompressedTensorsWNA16A8MoE", _CT_TRANSPOSING_METHODS)

    def test_factor_key_prefers_the_checkpoint_prefix_over_layer_id(self):
        # a target layer and an MTP draft layer CAN share layer_id; the sidecar of one must not be read as
        # the sidecar of the other. The key is the rank-independent module prefix where one exists.
        scheme = S.CompressedTensorsWNA16A8MoE.__new__(S.CompressedTensorsWNA16A8MoE)
        draft = types.SimpleNamespace(_sglang_prefix="mtp.layers.0.mlp.experts", layer_id=3)
        self.assertEqual(scheme._factor_key(draft), "mtp.layers.0.mlp.experts")
        no_prefix = types.SimpleNamespace(_sglang_prefix="", layer_id=3)
        self.assertEqual(scheme._factor_key(no_prefix), "L3")


# ---------------------------------------------------------------------------
# the factor sidecar (expert_store.claim_scale_factor)
# ---------------------------------------------------------------------------


class TestFactorSidecar(unittest.TestCase):
    def setUp(self):
        self.store = tempfile.mkdtemp(prefix="h88c_factor_")
        self.addCleanup(shutil.rmtree, self.store, True)

    def _claim(self, proposal, layer_key="L0", attr="w13_weight_scale", **kw):
        kw.setdefault("group_size", 128)
        kw.setdefault("layout", MWL.LAYOUT_W4A8)
        with mock.patch.dict(os.environ, _clean_env(SGLANG_MOE_EXPERT_STORE_IDENTITY="")):
            return es.claim_scale_factor(self.store, layer_key, attr, proposal, **kw)

    def test_first_publishes_second_adopts(self):
        f1, s1 = self._claim(0.001)
        self.assertEqual(s1, "published")
        f2, s2 = self._claim(0.009)
        self.assertEqual(s2, "adopted")
        self.assertEqual(f2, f1)  # float32 round trip, not approximate
        self.assertEqual(f1, struct.unpack(">f", struct.pack(">f", 0.001))[0])

    def test_no_proposal_without_sidecar_refuses_by_name(self):
        with self.assertRaisesRegex(es.W4A8FactorUnavailable, "no scale factor published"):
            self._claim(None)

    def test_no_proposal_with_sidecar_adopts(self):
        self._claim(0.002)
        f, s = self._claim(None)
        self.assertEqual((f, s), (struct.unpack(">f", struct.pack(">f", 0.002))[0], "adopted"))

    def test_layout_disagreement_is_an_error(self):
        self._claim(0.003)
        with self.assertRaisesRegex(RuntimeError, "was published for layout"):
            self._claim(0.003, layout=MWL.LAYOUT_W4A16)

    def test_group_size_disagreement_is_an_error(self):
        self._claim(0.004)
        with self.assertRaisesRegex(RuntimeError, "was published for layout|group_size"):
            self._claim(0.004, group_size=32)

    def test_foreign_identity_vouches_for_nothing_and_is_replaced(self):
        with mock.patch.dict(os.environ, _clean_env(SGLANG_MOE_EXPERT_STORE_IDENTITY="boot-A")):
            es.claim_scale_factor(self.store, "L0", "w13_weight_scale", 0.004, group_size=128, layout=MWL.LAYOUT_W4A8)
        # boot B (same directory, other identity): the sidecar of A vouches for nothing, B publishes its own
        with mock.patch.dict(os.environ, _clean_env(SGLANG_MOE_EXPERT_STORE_IDENTITY="boot-B")):
            f, s = es.claim_scale_factor(self.store, "L0", "w13_weight_scale", 0.008, group_size=128, layout=MWL.LAYOUT_W4A8)
            self.assertEqual(s, "published")
            self.assertEqual(es.read_factor(self.store, "L0", "w13_weight_scale")["identity"], "boot-B")
        with mock.patch.dict(os.environ, _clean_env(SGLANG_MOE_EXPERT_STORE_IDENTITY="boot-A")):
            self.assertIsNone(es.read_factor(self.store, "L0", "w13_weight_scale"))

    def test_sidecar_path_carries_the_sanitized_layer_key(self):
        path = es.factor_path(self.store, "model.layers.0.mlp.experts", "w13_weight_scale")
        self.assertTrue(path.endswith("model_layers_0_mlp_experts-w13_weight_scale.bin.factor.json"), path)

    def test_record_shape(self):
        self._claim(0.005, layer_key="L7", attr="w2_weight_scale")
        with mock.patch.dict(os.environ, _clean_env(SGLANG_MOE_EXPERT_STORE_IDENTITY="")):
            rec = es.read_factor(self.store, "L7", "w2_weight_scale")
        self.assertEqual(rec["layout"], MWL.LAYOUT_W4A8)
        self.assertEqual(rec["group_size"], 128)
        self.assertEqual(rec["tensor"], "w2_weight_scale")
        self.assertEqual(rec["identity"], "")
        self.assertIn("factor_f32", rec)
        # the sidecar sits next to the store file of the same (layer, tensor): same directory, same name + suffix
        self.assertEqual(os.path.dirname(es.factor_path(self.store, "L7", "w2_weight_scale")), self.store)
        self.assertEqual(
            es.factor_path(self.store, "L7", "w2_weight_scale"),
            es.store_path(self.store, "L7", "w2_weight_scale") + es.FACTOR_SUFFIX,
        )


# ---------------------------------------------------------------------------
# publish gate and adopted-factor band (fix round 1, review finding 1)
# ---------------------------------------------------------------------------


def _f32(x):
    return struct.unpack(">f", struct.pack(">f", float(x)))[0]


class TestPublishGate(unittest.TestCase):
    """Only a rank holding the WHOLE layer tensor publishes the factor.

    A subset view -- vetoed / cut rows, placeholder weights, an expert-dim shard window of
    SGLANG_UNEVEN_MOE_EXPERT_SHARD (layer._expert_shard_generic, fused_moe_triton/layer.py:593) -- has its own
    maximum, smaller (or differently placed) relative to the whole tensor. First-writer-wins would let such a view
    pin the sidecar on the tmpfs store; every later boot (P included, same identity) adopts a factor that maps the
    full view's largest scale past the x4096 reference -- the kernel accumulates int32 and wraps silently
    (marlin_template.h:1480-1503). Hence subset views ADOPT only; full views compute the SAME value from the same
    tensor (vLLM semantics max/4096 over the whole tensor), so publishing carries no order dependence.
    Pinned against the real store directory, no mocks."""

    def setUp(self):
        self.store = tempfile.mkdtemp(prefix="h88c_gate_")
        self.addCleanup(shutil.rmtree, self.store, True)
        self.scale = torch.rand(3, 4, 64, generator=torch.Generator().manual_seed(7)) + 0.25
        self.scheme = S.CompressedTensorsWNA16A8MoE.__new__(S.CompressedTensorsWNA16A8MoE)
        self.scheme.w4a8_group_size = 32

    def _layer(self, **extra):
        attrs = {"_sglang_prefix": "model.layers.0.mlp.experts", "moe_tp_rank": 0}
        attrs.update(extra)
        return types.SimpleNamespace(**attrs)

    def _env(self):
        return mock.patch.dict(os.environ, _clean_env(
            SGLANG_MOE_EXPERT_STORE_DIR=self.store, SGLANG_MOE_EXPERT_STORE_IDENTITY=""))

    def test_full_view_publishes(self):
        with self._env():
            f, src = self.scheme._resolve_factor(self._layer(), "w13_weight_scale", self.scale, None, False)
        self.assertEqual(src, "published")
        self.assertAlmostEqual(f, S.w4a8_scale_proposal(self.scale), delta=abs(f) * 1e-6)

    def test_subset_views_on_an_empty_store_refuse_by_name(self):
        # vetoed/cut rows, expert-shard window, placeholder weights: nothing published, refusal names the reason
        cases = (
            ("row cut", self._layer(), [0, 2], False),
            ("expert window", self._layer(_expert_shard_generic=True), None, False),
            ("placeholder", self._layer(), None, True),
        )
        with self._env():
            for name, layer, rows, ph in cases:
                with self.subTest(case=name):
                    with self.assertRaisesRegex(
                        es.W4A8FactorUnavailable, "only a rank with the whole tensor publishes"
                    ):
                        self.scheme._resolve_factor(layer, f"attr_{name.replace(' ', '_')}", self.scale, rows, ph)
        self.assertEqual([n for n in os.listdir(self.store) if n.endswith(es.FACTOR_SUFFIX)], [])

    def test_full_view_publishes_then_every_subset_adopts_identically(self):
        with self._env():
            f0, src0 = self.scheme._resolve_factor(self._layer(), "w13_weight_scale", self.scale, None, False)
            self.assertEqual(src0, "published")
            # a second FULL-view rank adopts the identical float32 (publishing has no order dependence:
            # every full view computes the same max/4096 from the same tensor)
            f1, src1 = self.scheme._resolve_factor(
                self._layer(moe_tp_rank=1), "w13_weight_scale", self.scale, None, False
            )
            self.assertEqual(src1, "adopted")
            self.assertEqual(f1, f0)
            for name, kwargs, rows, ph in (
                ("expert window", {"_expert_shard_generic": True}, None, False),
                ("row cut", {}, [0, 2], False),
                ("placeholder", {}, None, True),
            ):
                with self.subTest(case=name):
                    f, src = self.scheme._resolve_factor(
                        self._layer(**kwargs), "w13_weight_scale", self.scale, rows, ph
                    )
                    self.assertEqual(src, "adopted")
                    self.assertEqual(f, f0)
            rec = es.read_factor(self.store, "model.layers.0.mlp.experts", "w13_weight_scale")
            self.assertEqual(rec["factor"], f0)  # no subset overwrote the sidecar

    def test_subset_cannot_overwrite_even_with_a_larger_own_max(self):
        with self._env():
            f0, _ = self.scheme._resolve_factor(self._layer(), "w13_weight_scale", self.scale, None, False)
            bigger = self.scale * 100.0  # this rank's subset view has the larger maximum
            f, src = self.scheme._resolve_factor(
                self._layer(_expert_shard_generic=True), "w13_weight_scale", bigger, None, False
            )
            self.assertEqual(src, "adopted")
            self.assertEqual(f, f0)

    def test_no_store_keeps_the_h88b_local_factor_for_every_view(self):
        # no store = no sharing channel and no shared bytes: the rank's own factor applies, subset included
        # (this is the H88-B arithmetic the veto path without a store relies on)
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            for rows in (None, [0, 2]):
                with self.subTest(rows=rows):
                    f, src = self.scheme._resolve_factor(self._layer(), "w13_weight_scale", self.scale, rows, False)
                    self.assertEqual(src, "local")
                    self.assertAlmostEqual(f, S.w4a8_scale_proposal(self.scale, rows), delta=abs(f) * 1e-6)


class TestAdoptedFactorBand(unittest.TestCase):
    """The band's upper end for an ADOPTED factor is the x4096 reference, not the int16 limit 32767.

    The kernel sums ``frag_c(int32) += frag_c_tmp(int32 group dot) * scale(int)`` over every group of K
    (jit_kernel/csrc/gemm/marlin_a8_moe/marlin_template.h:1480-1503); a scale mapped above 4096 multiplies
    every group dot beyond the reference and the int32 accumulator wraps with no error. The H88-B reference and
    the GPU numerics test D7 orders both use round(s / max * 4096) (marlin_w4a8_utils.py:45-46,267-269)."""

    def setUp(self):
        self.s = (torch.rand(2, 4, 64, generator=torch.Generator().manual_seed(3)) + 0.25).to(torch.bfloat16)
        self.smax = float(self.s.float().abs().max())

    def test_upper_end_is_4096(self):
        self.assertEqual(S.W4A8_MAX_INT_BAND, (64, 4096))

    def test_self_consistent_factor_passes_including_the_float32_sidecar_round_trip(self):
        ref = _f32(self.smax / 4096)  # what claim_scale_factor stores and every rank adopts
        out, fac = S.w4a8_process_moe_scales(self.s, factor=ref)
        self.assertEqual(float(fac), ref)
        self.assertEqual(int(out.view(torch.int16).max()), 4096)  # largest scale lands exactly on the reference

    def test_factors_in_the_old_int16_headroom_refuse_by_name(self):
        # max_int in (4096, 32767] was accepted before the fix (int16 storage fits) but leaves the x4096
        # reference -> the int32 accumulator can overflow silently. Now: refused, named, with the sidecar hint.
        for target in (5000.0, 8000.0, 32000.0):
            with self.subTest(max_int=target):
                with self.assertRaisesRegex(
                    S.W4A8FactorOutOfBand, "FACTOR OUT OF BAND.*above the band.*int32 accumulator"
                ):
                    S.w4a8_process_moe_scales(self.s, factor=self.smax / target)

    def test_lower_end_unchanged(self):
        with self.assertRaisesRegex(S.W4A8FactorOutOfBand, "below the band"):
            S.w4a8_process_moe_scales(self.s, factor=self.smax / 40.0)

    def test_band_covers_rows_only(self):
        # only the rows this rank keeps are priced: a kept-row maximum inside the band passes even when a
        # never-read row would exceed it (that row is another process's, garbage here)
        s = torch.zeros(3, 4, 64, dtype=torch.bfloat16)
        s[0] = 0.5   # kept row: max 0.5
        s[2] = 9.0   # row this rank never reads
        factor = _f32(0.5 / 4096)  # maps the KEPT maximum to exactly 4096
        out, fac = S.w4a8_process_moe_scales(s, factor=factor, rows=[0, 1])
        self.assertEqual(float(fac), factor)
        self.assertEqual(int(out.view(torch.int16)[0].max()), 4096)


# ---------------------------------------------------------------------------
# compute_identity: the default layout stays byte-identical, others carry the tag (D5)
# ---------------------------------------------------------------------------


def _pre_h88c_identity(model, map_path):
    """The formula of compute_identity BEFORE H88-C, byte for byte (h2c-v1 without any layout field)."""
    h = hashlib.sha256()
    h.update(b"h2c-v1\0")
    if os.path.isdir(model):
        for name in sorted(os.listdir(model)):
            if name == "config.json" or name.endswith(".safetensors.index.json"):
                with open(os.path.join(model, name), "rb") as fh:
                    h.update(name.encode() + b"\0" + fh.read() + b"\0")
            elif name.endswith(".safetensors"):
                size = os.path.getsize(os.path.join(model, name))
                h.update(f"{name}\0{size}\0".encode())
    else:
        h.update(b"model-id\0" + str(model).encode() + b"\0")
    with open(map_path, "rb") as fh:
        h.update(b"map\0" + fh.read())
    return h.hexdigest()[:24]


class TestIdentityLayoutTag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="h88c_ident_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.model = os.path.join(self.tmp, "model")
        os.makedirs(self.model)
        with open(os.path.join(self.model, "config.json"), "w") as fh:
            json.dump({"num_experts": 512}, fh)
        with open(os.path.join(self.model, "model.safetensors.index.json"), "w") as fh:
            fh.write('{"weight_map": {}}')
        with open(os.path.join(self.model, "model-00001.safetensors"), "wb") as fh:
            fh.write(b"\0" * 11)
        self.map = os.path.join(self.tmp, "karte.json")
        with open(self.map, "w") as fh:
            json.dump({"version": 2, "slots": 324}, fh)

    def test_default_identity_is_byte_identical_to_pre_h88c(self):
        self.assertEqual(es.compute_identity(self.model, self.map), _pre_h88c_identity(self.model, self.map))
        self.assertEqual(
            es.compute_identity(self.model, self.map, MWL.LAYOUT_W4A16),
            _pre_h88c_identity(self.model, self.map),
        )

    def test_w4a8_identity_differs_and_is_stable(self):
        a8 = es.compute_identity(self.model, self.map, MWL.LAYOUT_W4A8)
        self.assertNotEqual(a8, es.compute_identity(self.model, self.map))
        self.assertEqual(a8, es.compute_identity(self.model, self.map, MWL.LAYOUT_W4A8))

    def test_missing_map_means_no_identity_in_every_layout(self):
        self.assertEqual(es.compute_identity(self.model, "", MWL.LAYOUT_W4A8), "")


# ---------------------------------------------------------------------------
# the launcher side (named refusal before any rank loads a byte)
# ---------------------------------------------------------------------------


class TestLauncherLayoutGate(unittest.TestCase):
    def setUp(self):
        try:
            from sglang.srt.weg2 import launcher as L
        except ImportError:  # pragma: no cover
            self.skipTest("launcher not importable in this tree")
        self.L = L

    def _ns(self, **kw):
        base = dict(env_p="", extra_p="", env_d="", extra_d="")
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_default_launch_reports_no_layout(self):
        log = []
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            self.assertEqual(self.L.moe_expert_layout_of_launch(self._ns(), log.append), "")
        self.assertEqual(log, [])  # default path: nothing logged, nothing changed

    def test_switched_launch_logs_the_layout(self):
        log = []
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            got = self.L.moe_expert_layout_of_launch(
                self._ns(env_p=f"{SW}=1", env_d=f"{SW}=1"), log.append
            )
        self.assertEqual(got, MWL.LAYOUT_W4A8)
        self.assertTrue(any("H88C MOE-LAYOUT layout=marlin_w4a8" in s for s in log), log)

    def test_mixed_boot_refuses_as_launch_refusal(self):
        log = []
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            with self.assertRaises(self.L.Weg2LaunchRefused) as cm:
                self.L.moe_expert_layout_of_launch(self._ns(env_d=f"{SW}=1"), log.append)
        self.assertIn(MWL.REFUSAL, str(cm.exception))

    def test_publish_store_identity_default_is_layout_free(self):
        tmp = tempfile.mkdtemp(prefix="h88c_pub_")
        self.addCleanup(shutil.rmtree, tmp, True)
        model = os.path.join(tmp, "model")
        os.makedirs(model)
        open(os.path.join(model, "config.json"), "w").write("{}")
        map_path = os.path.join(tmp, "karte.json")
        open(map_path, "w").write('{"version": 2}')
        log = []
        got = self.L.publish_store_identity(model, map_path, log.append)
        want = es.compute_identity(model, map_path)
        self.assertEqual(got, want)  # "" layout = exactly the identity of before H88-C
        self.assertFalse(any("layout=" in s for s in log), log)
        log2 = []
        got2 = self.L.publish_store_identity(model, map_path, log2.append, layout=MWL.LAYOUT_W4A8)
        self.assertEqual(got2, es.compute_identity(model, map_path, MWL.LAYOUT_W4A8))
        self.assertTrue(any("layout=marlin_w4a8" in s for s in log2), log2)


# ---------------------------------------------------------------------------
# D6 / W71 census: bytes per layout from the repack shapes
# ---------------------------------------------------------------------------


class TestCensusBytes(unittest.TestCase):
    # the real NF geometry: nf-abliteration-windowsxp/config.json text_config
    # hidden_size 2560, moe_intermediate_size 640, num_experts 512; main checkpoint g128, draft g32
    NF = dict(num_experts=512, hidden=2560, intermediate=640, group_size=128)

    def test_nf_geometry_bytes_exact(self):
        b16 = MWL.layer_tensor_bytes(MWL.LAYOUT_W4A16, **self.NF)
        b8 = MWL.layer_tensor_bytes(MWL.LAYOUT_W4A8, **self.NF)
        self.assertEqual(b16["w13_weight_packed"], 512 * 160 * 2560 * 4)  # 800 MiB
        self.assertEqual(b16["w2_weight_packed"], 512 * 40 * 5120 * 4)  # 400 MiB
        self.assertEqual(b16["w13_weight_scale"], 512 * 20 * 1280 * 2)  # 25 MiB
        self.assertEqual(b16["w2_weight_scale"], 512 * 5 * 2560 * 2)  # 12.5 MiB
        self.assertNotIn("w13_act_scale_factor", b16)
        self.assertEqual(b8["w13_act_scale_factor"], 4)
        self.assertEqual(b8["w2_act_scale_factor"], 4)
        for name in ("w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"):
            self.assertEqual(b16[name], b8[name], name)  # identical bytes: the W71 question answered

    def test_census_compare_rows(self):
        rows = MWL.census_compare(**self.NF)
        self.assertEqual(sum(r["delta"] for r in rows), 8)  # +8 B per MoE layer, nothing else
        for r in rows:
            if r["tensor"].endswith("_act_scale_factor"):
                self.assertEqual((r["a16_bytes"], r["a8_bytes"], r["delta"]), (0, 4, 4))
            else:
                self.assertEqual(r["delta"], 0, r)

    def test_draft_geometry_g32(self):
        rows = MWL.census_compare(num_experts=512, hidden=2560, intermediate=640, group_size=32)
        self.assertEqual(sum(r["delta"] for r in rows), 8)

    def test_channelwise_has_no_factor_rows(self):
        rows = MWL.census_compare(num_experts=4, hidden=128, intermediate=64, group_size=-1)
        self.assertEqual(sum(r["delta"] for r in rows), 0)
        self.assertFalse([r for r in rows if r["tensor"].endswith("_act_scale_factor")])

    def test_asym_zero_points_exist_in_both(self):
        rows = {r["tensor"]: r for r in MWL.census_compare(**self.NF, sym=False)}
        self.assertGreater(rows["w13_weight_zero_point"]["a16_bytes"], 0)
        self.assertEqual(rows["w13_weight_zero_point"]["delta"], 0)

    def test_unknown_layout_refuses(self):
        with self.assertRaises(ValueError):
            MWL.layer_tensor_bytes("marlin_w99", **self.NF)

    def test_a_stale_a16_census_does_not_change_w71_behaviour(self):
        # W71 prices from the census file + live NVML and never looks at the model or the format
        # (xchg_residency.resolve_census docstring); the doc must still say so.
        import inspect

        from sglang.srt.weg2 import xchg_residency as xr

        src = inspect.getsource(xr.resolve_census)
        self.assertIn("not looked at", src.lower().replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
