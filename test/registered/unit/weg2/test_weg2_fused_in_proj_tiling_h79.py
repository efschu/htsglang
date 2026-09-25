# SPDX-License-Identifier: Apache-2.0
"""H79 (fnNV4f1, 25.09.) -- the fused GDN in_proj against W84.

THE METAL.  The first NVFP4 flip boot (fnNV4f1, boot tree 77a89563c7) died in
its first P->D release with ``W29 ... W84 Weg2XchgCoverageRefused: 44
finding(s)`` on PP0 (16 on PP1, 12 on PP2), every one of them::

    SHORT tag=weights_N name=model.layers.L.linear_attn.in_proj_qkvz.weight
          planned_mib=80.000 live_mib=80.469
    SHORT tag=weights_N name=model.layers.L.linear_attn.in_proj_ba.weight
          planned_mib=0.469 live_mib=80.469

The NVIDIA NVFP4 checkpoint leaves ``in_proj_qkvz`` in BF16 (``#1483 ...
qkvz=UnquantizedLinearMethod``), so the upstream fusion
``Qwen3_5GatedDeltaNet.finalize_fused_in_proj`` (port #37500) stacks both
weights into ONE block and rebinds them as its row views.  The plan addresses
each Parameter by ``data_ptr()`` and its VIEW's geometry -- its claims were
right -- but the coverage walk charged every view the whole storage.  And the
block itself was allocated outside any layer band, in the BASE ``weights``
tag, while plan and walk file ``in_proj_*`` under the layer's band.

WHAT THIS PINS
* ``FusedInProjAgainstW84`` -- the metal's own shapes through the REAL fusion
  and the REAL plan producer (``derive_leg_plan``): RED on 77a89563c7 (SHORT x2),
  GREEN with H79 (one TILED storage, the fused block a view holder).
* ``FinalizeAllocatesInTheBand`` -- the ``torch.cat`` runs under the layer's
  band tag: RED on 77a89563c7 (it ran under the base tag), GREEN with H79.
* ``Int4PathIsUnchanged`` -- the INT4 shape (compressed-tensors qkvz, no
  fusion) is byte-identical: nothing allocated, nothing scoped, and the
  coverage lines equal the ones 77a89563c7 prints.  GREEN on BOTH trees.
* ``TilingStaysStrict`` -- W84 still refuses a gap, an overlap, a plan without
  ``ba`` and a claim past the view, and the holder of an untiled storage stays
  UNCOVERED.
* ``FlipFormDoesNotFuse`` (H79 part 2, operator decision 25.09.: variant (A)
  on top of (B)) -- a Weg-2 flip rank (group P or D) does NOT fuse: the post-
  load cat cannot reuse the 80 MiB storage it replaces, a tag pool never
  returns a freed block, and every fused layer left an 80 MiB hole in its band
  (fnNV4f1: band ``inactive_gib`` 0.18-0.26 against 0.00-0.01 unfused) for a
  0.15-0.6 % decode gain.  qkvz/ba keep their own storages, nothing is
  allocated, W84 passes without a tiled storage.  Without Weg-2 upstream still
  fuses.  RED on the (B)-only commit, GREEN with (A).

Hermetic: CPU tensors, ``CUDA_VISIBLE_DEVICES=""``, a fake saver cdll, no
checkpoint, no region.
"""

from __future__ import annotations

import contextlib
import os
import unittest
import unittest.mock
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="stage-a-weg2-unit")

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS  # noqa: E402
from sglang.srt.layers.quantization.unquant import (  # noqa: E402
    UnquantizedLinearMethod,
)
from sglang.srt.managers import weg2_memory_saver as wms  # noqa: E402
from sglang.srt.models import qwen3_5 as q35  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

MIB = 1024 * 1024

#: fnNV4f1 PP0's own GDN in_proj (hidden 2560): qkvz [16384, 2560] BF16 =
#: 80.000 MiB, ba [96, 2560] = 0.469 MiB, fused block 80.469 MiB.
NQ, NB, K = 16384, 96, 2560

QKVZ = "model.layers.{}.linear_attn.in_proj_qkvz.weight"
BA = "model.layers.{}.linear_attn.in_proj_ba.weight"
HOLDER = "model.layers.{}.linear_attn._fused_in_proj_weight"


class CompressedTensorsLinearMethod:
    """The INT4 qkvz's method as ``finalize_fused_in_proj`` sees it: anything
    that is not ``UnquantizedLinearMethod`` (fnFL2x176: ``#1483 GDN-PROJ
    layer0 qkvz=CompressedTensorsLinearMethod ba=UnquantizedLinearMethod``)."""


class _Proj(nn.Module):
    def __init__(self, rows: int, cols: int, method, *, packed: bool = False):
        super().__init__()
        self.quant_method = method
        self.bias = None
        if packed:
            # compressed-tensors pack-quantized: the checkpoint's own names.
            self.weight_packed = nn.Parameter(
                torch.arange(rows * cols // 8, dtype=torch.int32).view(rows, cols // 8),
                requires_grad=False)
            self.weight_scale = nn.Parameter(
                torch.ones(rows, cols // 32, dtype=torch.bfloat16),
                requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.zeros(rows, cols, dtype=torch.bfloat16), requires_grad=False)


class _GDN(nn.Module):
    """Exactly the state ``finalize_fused_in_proj`` reads and writes."""

    def __init__(self, layer_id: int, nq: int, nb: int, k: int, *, int4=False):
        super().__init__()
        self.layer_id = layer_id
        self.in_proj_qkvz = _Proj(
            nq, k,
            CompressedTensorsLinearMethod() if int4 else UnquantizedLinearMethod(),
            packed=int4)
        self.in_proj_ba = _Proj(nb, k, UnquantizedLinearMethod())
        self._fused_in_proj_weight = None
        self._fused_in_proj_qkvz_width = 0

    def finalize(self):
        """The PRODUCT method, unbound, on this double."""
        return q35.Qwen3_5GatedDeltaNet.finalize_fused_in_proj(self)


class _Layer(nn.Module):
    def __init__(self, gdn: _GDN):
        super().__init__()
        self.linear_attn = gdn


class _Model(nn.Module):
    """``model.layers.<id>.linear_attn.*`` -- the names the metal refused."""

    def __init__(self, gdns):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict(
            {str(g.layer_id): _Layer(g) for g in gdns})

    def gdn(self, layer_id: int) -> _GDN:
        return self.model.layers[str(layer_id)].linear_attn


class _FakeCdll:
    """The saver's C hook as ``weight_chunk_scope`` drives it."""

    def __init__(self):
        self.tag = GPU_MEMORY_TYPE_WEIGHTS
        self.sets = []

    def tms_get_interesting_region(self):
        return True

    def tms_set_current_tag(self, raw):
        self.tag = raw.decode()
        self.sets.append(self.tag)


@contextlib.contextmanager
def _cuda_no_lora(group: str = ""):
    """``finalize_fused_in_proj`` returns at once off CUDA and under LoRA.

    ``group`` is this rank's Weg-2 group as ``weg2_group_name`` answers it --
    ``""`` for an engine that is none, ``"P"``/``"D"`` for a flip rank.  The
    READER is substituted, never the env: it caches for the process lifetime,
    so an env set here would pass or fail depending on test order."""
    with unittest.mock.patch.multiple(
        q35,
        _is_cuda=True,
        get_lora=lambda: SimpleNamespace(enable_lora=False, lora_paths=None),
    ), unittest.mock.patch.object(wms, "weg2_group_name", lambda: group):
        yield


def _fuse_all(model: _Model, group: str = "") -> list:
    with _cuda_no_lora(group):
        return [g.linear_attn.finalize() for g in model.model.layers.values()]


def _plan_bytes(model):
    """The PRODUCT plan -- the producer ``default_plan_provider`` consumes."""
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    plan, reason = sh.derive_leg_plan(
        hook=sh.HOOK_SOURCE, group="P", peer="D", rank=0, model=model)
    assert plan is not None, f"derivation refused: {reason}"
    return wx.plan_bytes_from_descs(plan.descs)


def _zero(_tag: str) -> int:
    return 0


class _ChunkedCase(unittest.TestCase):
    """The launcher's chunk geometry: 8 layers per band, 8 bands."""

    def setUp(self):
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"

    def tearDown(self):
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)


# ===========================================================================
# THE METAL SHAPE -- RED on 77a89563c7, GREEN with H79.
# ===========================================================================


class FusedInProjAgainstW84(_ChunkedCase):
    def _fused_model(self):
        model = _Model([_GDN(0, NQ, NB, K)])
        _fuse_all(model)
        return model

    def test_the_metal_shape_passes_w84_as_one_tiled_storage(self):
        model = self._fused_model()
        g = model.gdn(0)
        qkvz, ba = g.in_proj_qkvz.weight, g.in_proj_ba.weight
        # The metal's precondition: two plan Parameters over ONE storage.
        self.assertEqual(qkvz.untyped_storage().data_ptr(),
                         ba.untyped_storage().data_ptr())
        self.assertEqual(qkvz.untyped_storage().nbytes(), (NQ + NB) * K * 2)
        got = _plan_bytes(model)
        # The plan's claims are the VIEWS -- 80.000 / 0.469 MiB, as on metal.
        self.assertEqual(got["weights_0"][QKVZ.format(0)], NQ * K * 2)
        self.assertEqual(got["weights_0"][BA.format(0)], NB * K * 2)
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=got, tag_bytes=_zero)
        row = rows["weights_0"]
        # RED on 77a89563c7: SHORT qkvz planned 80.000 live 80.469 and
        # SHORT ba planned 0.469 live 80.469 -- the 44/16/12 of fnNV4f1.
        self.assertEqual(
            [(s.name, round(s.planned_bytes / MIB, 3), round(s.live_bytes / MIB, 3))
             for s in row.short], [])
        self.assertEqual([t.name for t in row.uncovered], [])
        self.assertEqual(list(row.missing), [])
        self.assertTrue(row.ok, wx.coverage_refusal_message(rows))
        # The plan's bytes are the storage, once: 80.469 MiB.
        self.assertEqual(row.planned_bytes, (NQ + NB) * K * 2)

    def test_the_fused_block_is_a_view_holder_and_the_line_says_so(self):
        model = self._fused_model()
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=_plan_bytes(model),
            tag_bytes=_zero)
        row = rows["weights_0"]
        self.assertEqual(len(row.tiled), 1)
        self.assertEqual(row.view_holders, (HOLDER.format(0),))
        self.assertEqual(row.tiling, ())
        s = row.tiled[0]
        self.assertEqual(s.members, (
            (QKVZ.format(0), 0, NQ * K * 2),
            (BA.format(0), NQ * K * 2, NB * K * 2),
        ))
        self.assertEqual(s.holders, (HOLDER.format(0),))
        line = row.cover_line()
        self.assertTrue(line.endswith(" tiled=1 view_holders=1 tiling=0"), line)
        self.assertIn("short=0 missing=0", line)
        (tl,) = row.tiled_lines()
        self.assertEqual(
            tl,
            "WEG2-XCHG-TILED rank=0 tag=weights_0 verdict=tiled "
            "storage_mib=80.469 members=2 spans="
            f"{QKVZ.format(0)}[0.000:80.000]|{BA.format(0)}[80.000:80.469] "
            f"holders={HOLDER.format(0)}")

    def test_arm_coverage_emits_the_tiled_line_and_the_view_beside_the_storage(self):
        model = self._fused_model()
        lines = []
        vote = wx.arm_coverage(
            model, rank=0, planned_bytes_by_tag=_plan_bytes(model),
            tag_bytes=_zero, log=lines.append)
        self.assertTrue(vote.ok, vote.reason)
        self.assertEqual(
            sum(1 for ln in lines if ln.startswith("WEG2-XCHG-TILED ")), 1)
        pp = [ln for ln in lines if ln.startswith("WEG2-XCHG-PLAN-PARAM ")
              and "in_proj_qkvz" in ln]
        self.assertEqual(len(pp), 1)
        self.assertIn("planned_mib=80.000 live_tag=weights_0 live_mib=80.469", pp[0])
        self.assertTrue(pp[0].endswith(
            "verdict=here view_mib=80.000 view_offset_mib=0.000"), pp[0])

    def test_two_layers_two_bands_each_tiled_on_its_own(self):
        """PP0 holds 22 of these across ten bands; each storage is judged in
        the band its names are filed under, never across bands."""
        model = _Model([_GDN(1, 64, 8, 32), _GDN(9, 64, 8, 32)])
        _fuse_all(model)
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=_plan_bytes(model),
            tag_bytes=_zero)
        for tag, lid in (("weights_0", 1), ("weights_1", 9)):
            self.assertTrue(rows[tag].ok, wx.coverage_refusal_message(rows))
            self.assertEqual(rows[tag].view_holders, (HOLDER.format(lid),))


# ===========================================================================
# THE BAND -- RED on 77a89563c7 (base tag), GREEN with H79.
# ===========================================================================


class FinalizeAllocatesInTheBand(unittest.TestCase):
    def setUp(self):
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"

    def tearDown(self):
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)

    def _run(self, gdn):
        fake = _FakeCdll()
        at_cat = []
        real_cat = torch.cat

        def _cat(*a, **kw):
            at_cat.append(fake.tag)
            return real_cat(*a, **kw)

        with unittest.mock.patch.object(wms, "_tms_cdll_in_region", lambda: fake), \
                unittest.mock.patch.object(torch, "cat", _cat), _cuda_no_lora():
            with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS):
                status = gdn.finalize()
        return fake, at_cat, status

    def test_the_cat_runs_under_the_layers_band_tag(self):
        """Layer 9 with 8 layers per band is ``weights_1``.  On 77a89563c7 the
        cat ran under the base tag -- fnNV4f1's base pool held 2.91/0.63/1.67
        GiB on PP0/1/2 against 0.60/0.00/0.61 unfused."""
        fake, at_cat, status = self._run(_GDN(9, 64, 8, 32))
        self.assertEqual(at_cat, ["weights_1"])
        # ... and the region's base tag is back for whatever loads next.
        self.assertEqual(fake.tag, GPU_MEMORY_TYPE_WEIGHTS)
        self.assertEqual(status, "fused@weights_1")

    def test_the_views_are_still_the_upstream_views(self):
        gdn = _GDN(9, 64, 8, 32)
        q0 = torch.arange(64 * 32, dtype=torch.float32).view(64, 32)
        gdn.in_proj_qkvz.weight.data.copy_(q0.to(torch.bfloat16))
        gdn.in_proj_ba.weight.data.fill_(3)
        self._run(gdn)
        fused = gdn._fused_in_proj_weight
        self.assertEqual(tuple(fused.shape), (72, 32))
        self.assertEqual(gdn._fused_in_proj_qkvz_width, 64)
        self.assertEqual(gdn.in_proj_qkvz.weight.data_ptr(), fused.data_ptr())
        self.assertEqual(gdn.in_proj_ba.weight.data_ptr(),
                         fused.data_ptr() + 64 * 32 * 2)
        self.assertTrue(torch.equal(fused[:64], q0.to(torch.bfloat16)))
        self.assertTrue(bool((fused[64:] == 3).all()))


# ===========================================================================
# INT4 -- byte-identical. GREEN on 77a89563c7 AND with H79.
# ===========================================================================

#: Printed by 77a89563c7 for ``_Model([_GDN(1, 64, 8, 64, int4=True)])`` --
#: a boot without a fused block must keep every line byte for byte.
INT4_GOLDEN = [
    "WEG2-XCHG-COVER rank=0 tag=weights_0 planned_mib=0.0 buffers_mib=0.0 "
    "walk_mib=0.0 tms_mib=0.0 attribution=saver:tms_tag_bytes|walk:region+name "
    "attribution_delta_mib=n/a attribution_verdict=NO-SAVER-ANSWER "
    "slack_mib=n/a uncovered=0 short=0 missing=0 params=3 buffers=0 attrs=0 "
    "exempt=0 empty=0 local_scratch=0 not_source=0 tms_answered=no mode=ring",
    "WEG2-XCHG-PLAN-PARAM rank=0 tag=weights_0 "
    "name=model.layers.1.linear_attn.in_proj_ba.weight planned_mib=0.001 "
    "live_tag=weights_0 live_mib=0.001 dtype=torch.bfloat16 shape=[8, 64] "
    "verdict=here",
    "WEG2-XCHG-PLAN-PARAM rank=0 tag=weights_0 "
    "name=model.layers.1.linear_attn.in_proj_qkvz.weight_packed planned_mib=0.002 "
    "live_tag=weights_0 live_mib=0.002 dtype=torch.int32 shape=[64, 8] "
    "verdict=here",
    "WEG2-XCHG-PLAN-PARAM rank=0 tag=weights_0 "
    "name=model.layers.1.linear_attn.in_proj_qkvz.weight_scale planned_mib=0.000 "
    "live_tag=weights_0 live_mib=0.000 dtype=torch.bfloat16 shape=[64, 2] "
    "verdict=here",
]


class Int4PathIsUnchanged(_ChunkedCase):
    def test_an_unfused_layer_is_not_touched(self):
        # Outside Weg-2 AND on a flip rank: the INT4 qkvz declines before the
        # flip veto is even asked, so both read the same.
        for group in ("", "P", "D"):
            gdn = _GDN(1, 64, 8, 64, int4=True)
            before = {n: (p.data_ptr(), p.detach().clone())
                      for n, p in gdn.named_parameters()}
            fake = _FakeCdll()
            cats = []
            real_cat = torch.cat

            def _cat(*a, **kw):
                cats.append(1)
                return real_cat(*a, **kw)

            with unittest.mock.patch.object(wms, "_tms_cdll_in_region", lambda: fake), \
                    unittest.mock.patch.object(torch, "cat", _cat), \
                    _cuda_no_lora(group):
                with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS):
                    gdn.finalize()
            self.assertEqual(cats, [], group)
            self.assertEqual(fake.sets, [], "an unfused layer entered a band scope")
            self.assertIsNone(gdn._fused_in_proj_weight)
            for n, p in gdn.named_parameters():
                ptr, data = before[n]
                self.assertEqual(p.data_ptr(), ptr, n)
                self.assertTrue(torch.equal(p.detach(), data), n)

    def test_the_int4_coverage_lines_are_byte_identical(self):
        model = _Model([_GDN(1, 64, 8, 64, int4=True)])
        _fuse_all(model)  # the INT4 qkvz declines; nothing fuses
        got = _plan_bytes(model)
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_RING):
            rows = wx.build_coverage(
                model, rank=0, planned_bytes_by_tag=got, tag_bytes=_zero)
        self.assertEqual(sorted(rows), ["weights_0"])
        row = rows["weights_0"]
        self.assertTrue(row.ok)
        lines = [row.cover_line()] + wx.plan_param_lines(
            model, rank=0, tag="weights_0", planned_bytes_by_tag=got)
        self.assertEqual(lines, INT4_GOLDEN)


# ===========================================================================
# STRICT -- W84 refuses what the exchange could not source.
# ===========================================================================


class TilingStaysStrict(_ChunkedCase):
    NQ, NB, K = 64, 8, 32

    def _model_with_views(self, fused_rows, qkvz_rows, ba_rows):
        """A GDN whose weights are row views of ONE block, placed by hand."""
        g = _GDN(0, self.NQ, self.NB, self.K)
        fused = torch.zeros(fused_rows, self.K, dtype=torch.bfloat16)
        g.in_proj_qkvz.weight.data = fused[qkvz_rows[0]:qkvz_rows[1]]
        g.in_proj_ba.weight.data = fused[ba_rows[0]:ba_rows[1]]
        g._fused_in_proj_weight = fused
        return _Model([g])

    def _rows(self, model, planned=None):
        return wx.build_coverage(
            model, rank=0,
            planned_bytes_by_tag=planned if planned is not None else _plan_bytes(model),
            tag_bytes=_zero)

    def test_a_gap_between_the_views_is_refused(self):
        nq, nb, rb = self.NQ, self.NB, self.K * 2
        model = self._model_with_views(nq + nb + 1, (0, nq), (nq + 1, nq + 1 + nb))
        rows = self._rows(model)
        row = rows["weights_0"]
        self.assertFalse(row.ok)
        self.assertEqual([(f.kind, f.start, f.end) for f in row.tiling],
                         [("GAP", nq * rb, (nq + 1) * rb)])
        # The holder of an UNTILED storage is no view holder: uncovered.
        self.assertEqual([t.name for t in row.uncovered], [HOLDER.format(0)])
        self.assertEqual(row.view_holders, ())
        msg = wx.coverage_refusal_message(rows)
        self.assertIn("W84 Weg2XchgCoverageRefused", msg)
        self.assertIn("TILING tag=weights_0 kind=GAP", msg)
        self.assertTrue(row.cover_line().endswith(
            " tiled=0 view_holders=0 tiling=1"), row.cover_line())

    def test_an_overlap_is_refused(self):
        nq, nb, rb = self.NQ, self.NB, self.K * 2
        model = self._model_with_views(nq + nb - 1, (0, nq), (nq - 1, nq - 1 + nb))
        row = self._rows(model)["weights_0"]
        self.assertFalse(row.ok)
        self.assertEqual([(f.kind, f.start, f.end) for f in row.tiling],
                         [("OVERLAP", (nq - 1) * rb, nq * rb)])

    def test_a_non_contiguous_view_is_refused(self):
        g = _GDN(0, self.NQ, self.NB, self.K)
        fused = torch.zeros(self.NQ + self.NB, 2 * self.K, dtype=torch.bfloat16)
        g.in_proj_qkvz.weight.data = fused[: self.NQ, : self.K]
        g.in_proj_ba.weight.data = fused[self.NQ:, : self.K]
        g._fused_in_proj_weight = fused
        row = self._rows(_Model([g]))["weights_0"]
        self.assertFalse(row.ok)
        self.assertIn("NONCONTIGUOUS", [f.kind for f in row.tiling])

    def test_a_plan_without_ba_is_still_refused(self):
        model = _Model([_GDN(0, self.NQ, self.NB, self.K)])
        _fuse_all(model)
        planned = _plan_bytes(model)
        del planned["weights_0"][BA.format(0)]
        row = self._rows(model, planned)["weights_0"]
        self.assertFalse(row.ok)
        # One plan member left: the pre-H79 judgement, unchanged -- qkvz is
        # SHORT against the whole block and ba has no source at all.
        self.assertEqual([s.name for s in row.short], [QKVZ.format(0)])
        self.assertIn(BA.format(0), [t.name for t in row.uncovered])

    def test_a_claim_past_the_view_is_W68(self):
        model = _Model([_GDN(0, self.NQ, self.NB, self.K)])
        _fuse_all(model)
        planned = _plan_bytes(model)
        planned["weights_0"][QKVZ.format(0)] += 2
        with self.assertRaises(wx.Weg2XchgPlanDisagree) as cm:
            self._rows(model, planned)
        self.assertIn("W68", str(cm.exception))

    def test_an_alias_of_the_same_rows_is_not_an_overlap(self):
        """Two names over the SAME range were accepted before H79 and are."""
        model = _Model([_GDN(0, self.NQ, self.NB, self.K)])
        _fuse_all(model)
        g = model.gdn(0)
        g.in_proj_qkvz.weight_alias = nn.Parameter(
            g.in_proj_qkvz.weight.data, requires_grad=False)
        row = self._rows(model)["weights_0"]
        self.assertTrue(row.ok, row.tiling)
        self.assertEqual(len(row.tiled), 1)


# ===========================================================================
# (A): THE WEG-2 FLIP FORM DOES NOT FUSE -- RED on the (B)-only commit.
# ===========================================================================


class FlipFormDoesNotFuse(_ChunkedCase):
    def _finalize_recorded(self, gdn, group):
        fake = _FakeCdll()
        at_cat = []
        real_cat = torch.cat

        def _cat(*a, **kw):
            at_cat.append(fake.tag)
            return real_cat(*a, **kw)

        with unittest.mock.patch.object(wms, "_tms_cdll_in_region", lambda: fake), \
                unittest.mock.patch.object(torch, "cat", _cat), _cuda_no_lora(group):
            with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS):
                status = gdn.finalize()
        return status, fake, at_cat

    def test_a_flip_rank_keeps_qkvz_and_ba_in_their_own_storages(self):
        for group in ("P", "D"):
            gdn = _GDN(9, 64, 8, 32)
            gdn.in_proj_qkvz.weight.data.fill_(1)
            gdn.in_proj_ba.weight.data.fill_(2)
            before = {n: (p.data_ptr(), p.untyped_storage().data_ptr(),
                          p.untyped_storage().nbytes())
                      for n, p in gdn.named_parameters()}
            status, fake, at_cat = self._finalize_recorded(gdn, group)
            self.assertEqual(status, "skip:weg2-flip", group)
            # Nothing allocated -- the cat is the fusion's ONLY allocation --
            # and no band scope entered, so no pool grew.
            self.assertEqual(at_cat, [], group)
            self.assertEqual(fake.sets, [], group)
            self.assertIsNone(gdn._fused_in_proj_weight)
            self.assertEqual(gdn._fused_in_proj_qkvz_width, 0)
            for n, p in gdn.named_parameters():
                self.assertEqual(
                    (p.data_ptr(), p.untyped_storage().data_ptr(),
                     p.untyped_storage().nbytes()), before[n], (group, n))
                # its OWN storage, exactly its own bytes
                self.assertEqual(p.untyped_storage().nbytes(),
                                 p.numel() * p.element_size(), (group, n))
            self.assertNotEqual(
                gdn.in_proj_qkvz.weight.untyped_storage().data_ptr(),
                gdn.in_proj_ba.weight.untyped_storage().data_ptr())
            self.assertTrue(bool((gdn.in_proj_qkvz.weight == 1).all()))
            self.assertTrue(bool((gdn.in_proj_ba.weight == 2).all()))

    def test_the_metal_shape_on_a_flip_rank_needs_no_tiling(self):
        """fnNV4f1's own layer on group P: W84 passes on the one-Parameter-
        one-storage path, the plan claims each storage whole."""
        model = _Model([_GDN(0, NQ, NB, K)])
        self.assertEqual(_fuse_all(model, group="P"), ["skip:weg2-flip"])
        got = _plan_bytes(model)
        self.assertEqual(got["weights_0"][QKVZ.format(0)], NQ * K * 2)
        self.assertEqual(got["weights_0"][BA.format(0)], NB * K * 2)
        rows = wx.build_coverage(
            model, rank=0, planned_bytes_by_tag=got, tag_bytes=_zero)
        row = rows["weights_0"]
        self.assertTrue(row.ok, wx.coverage_refusal_message(rows))
        self.assertEqual((row.tiled, row.view_holders, row.tiling), ((), (), ()))
        self.assertNotIn(" tiled=", row.cover_line())
        self.assertEqual(row.planned_bytes, (NQ + NB) * K * 2)

    def test_without_weg2_upstream_still_fuses(self):
        gdn = _GDN(9, 64, 8, 32)
        status, fake, at_cat = self._finalize_recorded(gdn, "")
        self.assertEqual(status, "fused@weights_1")
        self.assertEqual(at_cat, ["weights_1"])
        self.assertEqual(gdn.in_proj_qkvz.weight.data_ptr(),
                         gdn._fused_in_proj_weight.data_ptr())

    def test_the_census_line_says_what_the_flip_form_did(self):
        census = q35.Qwen3_5GatedDeltaNet.fused_in_proj_census_line
        self.assertEqual(
            census(["skip:weg2-flip"] * 22),
            "#H79 GDN-FUSED-IN-PROJ gdn=22 fused=0 skipped=22 already=0 "
            "unbanded=0 reason=weg2-flip:22 tags=-")
        self.assertEqual(
            census(["fused@weights_0"] * 3 + ["fused@weights_1"] * 2
                   + ["fused@-"]),
            "#H79 GDN-FUSED-IN-PROJ gdn=6 fused=6 skipped=0 already=0 "
            "unbanded=1 reason=- tags=-:1,weights_0:3,weights_1:2")
        int4 = "skip:qkvz=CompressedTensorsLinearMethod/ba=UnquantizedLinearMethod"
        self.assertEqual(
            census([int4] * 22 + ["already"]),
            "#H79 GDN-FUSED-IN-PROJ gdn=23 fused=0 skipped=22 already=1 "
            "unbanded=0 reason=qkvz=CompressedTensorsLinearMethod/"
            "ba=UnquantizedLinearMethod:22 tags=-")


if __name__ == "__main__":
    unittest.main()
