# SPDX-License-Identifier: Apache-2.0
"""#1273 slice S2 -- coverage arming (W51) and the draft tag.

WEG2_REUSE_SPEC_0908.md section 6/S2 and section 4.1.  Two properties, and both
of them are about a SILENT wrongness, which is why they are tests and not a
boot observation:

1. **Coverage.**  The exchange fills the destination's weight pages from the
   source's VRAM.  A live tensor under an exchanged tag that the plan does not
   carry and that ``_export_static_state`` does not carry either is a page the
   destination never receives -- plausible garbage, no error, no crash.  The
   arming check walks ``named_parameters()`` + ``named_buffers()`` + a sweep of
   every module's ``__dict__`` for stray ``torch.Tensor`` attributes and
   refuses by name (**W51 Weg2XchgCoverageRefused**) rather than ship a tag it
   cannot account for.  The SLACK (allocator overhang, measured +0.08 to
   +0.58 GiB/rank) is PRINTED, never compared for equality: an equality assert
   would refuse every boot.

2. **The draft tag.**  Spec section 4.1: D's NEXTN/MTP draft runner
   (1382/1311/1311 MiB measured on boot weg2sb4) has no VRAM source on P, which
   carries no ``--speculative-*`` in this form.  It gets its own tag,
   ``weights_draft``, and that tag must be OUTSIDE the weights family so it is
   never in a leg, never in a census and never in a wave.

THE SPEC IS WRONG ABOUT (2) AT 3ea18deb95, and this file pins the correction.
``is_weights_family_tag`` (managers/weg2_memory_saver.py:1875) matches
``tag.startswith(WEIGHT_CHUNK_PREFIX)`` and ``WEIGHT_CHUNK_PREFIX`` is the
literal ``"weights_"``, so ``is_weights_family_tag("weights_draft")`` is
**True** at HEAD -- the opposite of what section 4.1 asserts.  A tag name is not
a predicate; the family is ``weights`` plus ``weights_<integer>``, and the
predicate now says so.  Same defect one layer down for S8's planned
``weights_vision``.

Hermetic: CPU tensors, no CUDA, no torch_memory_saver, no model checkpoint.
The chunk-scope case drives a FAKE cdll double, so the tag bookkeeping is
exercised without the C hook.
"""

import os
import unittest
import unittest.mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.managers import weg2_memory_saver as wms
from sglang.srt.weg2 import weight_exchange as wx

MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# The hermetic double: a module tree shaped like the seams that matter --
# a layer band (so the tag is a chunk tag), a registered non-persistent buffer
# (the rope cache), and a place to inject a stray tensor attribute.
# ---------------------------------------------------------------------------


class _Rope(nn.Module):
    """``cos_sin_cache`` is registered with ``persistent=False`` exactly as
    ``layers/rotary_embedding/base.py:173`` registers it -- a buffer that is
    absent from ``state_dict()`` and present in ``named_buffers()``, which is
    the whole reason it is carried by ``_export_static_state`` and not by the
    plan."""

    def __init__(self, mib: int):
        super().__init__()
        self.register_buffer(
            "cos_sin_cache",
            torch.zeros(mib * MIB // 4, dtype=torch.float32),
            persistent=False,
        )


class _Layer(nn.Module):
    def __init__(self, rope_mib: int = 0):
        super().__init__()
        self.qkv_proj = nn.Parameter(torch.zeros(4 * MIB, dtype=torch.uint8))
        self.o_proj = nn.Parameter(torch.zeros(2 * MIB, dtype=torch.uint8))
        if rope_mib:
            self.rotary_emb = _Rope(rope_mib)


class _Model(nn.Module):
    def __init__(self, n_layers: int = 2, rope_mib: int = 0):
        super().__init__()
        self.layers = nn.ModuleList(
            [_Layer(rope_mib if i == 0 else 0) for i in range(n_layers)]
        )
        self.embed_tokens = nn.Parameter(torch.zeros(8 * MIB, dtype=torch.uint8))


def _planned_names(model, *, drop=()):
    """The S1 interface, built here the way S1's plan builder will build it:
    ``{tag: {parameter name, ...}}`` over ``named_parameters()``."""
    out = {}
    for name, _p in model.named_parameters():
        tag = wx.tag_of_parameter_name(name)
        if name in drop:
            continue
        out.setdefault(tag, set()).add(name)
    return out


def _tag_bytes_stub(planned_extra_mib=0.0):
    """A ``tms_tag_bytes``-shaped callable: the saver's own per-tag byte sum."""

    def _fn(tag: str) -> int:
        return int(planned_extra_mib * MIB)

    return _fn


class _CaptureLog:
    """Collect the module's emitted lines without touching root logging."""

    def __init__(self):
        self.lines = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)


class DraftTagOutOfFamilyTest(unittest.TestCase):
    """Spec section 4.1 / S2: ``weights_draft`` is not a weights-family tag."""

    def test_draft_tag_is_not_in_the_weights_family(self):
        self.assertEqual(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, "weights_draft")
        # RED AT 3ea18deb95: startswith("weights_") is True for this name.
        self.assertFalse(wms.is_weights_family_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT))
        self.assertFalse(wms.is_weights_chunk_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT))
        # The family itself is unchanged: base tag plus weights_<integer>.
        self.assertTrue(wms.is_weights_family_tag(GPU_MEMORY_TYPE_WEIGHTS))
        for k in (0, 3, 7, 11):
            self.assertTrue(wms.is_weights_family_tag(f"weights_{k}"))
            self.assertTrue(wms.is_weights_chunk_tag(f"weights_{k}"))
        self.assertNotIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, wms.weights_family_tags(8))
        self.assertEqual(len(wms.weights_family_tags(8)), 9)

    def test_the_same_predicate_holds_for_any_non_numeric_suffix(self):
        """The class, not the instance: S8's planned ``weights_vision`` is the
        next tag with this shape and it must be out of the family too."""
        for tag in ("weights_vision", "weights_draft", "weights_", "weights_0b"):
            self.assertFalse(wms.is_weights_chunk_tag(tag), tag)
            self.assertFalse(wms.is_weights_family_tag(tag), tag)

    def test_resident_line_names_the_draft_tag_out_of_family(self):
        line = wx.resident_line(
            rank=1, tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT, mib=1311.0
        )
        self.assertTrue(line.startswith("WEG2-XCHG-RESIDENT "))
        self.assertIn("tag=weights_draft", line)
        self.assertIn("rank=1", line)
        self.assertIn("mib=1311.0", line)
        self.assertIn("in_family=no", line)


class DraftRegionTagTest(unittest.TestCase):
    """The region tag the draft runner opens, and the chunk scope inside it."""

    class _FakeCdll:
        def __init__(self):
            self.tag = None
            self.sets = []

        def tms_get_interesting_region(self):
            return True

        def tms_set_current_tag(self, raw):
            self.tag = raw.decode()
            self.sets.append(self.tag)

    def test_weights_region_tag_is_the_draft_tag_only_under_exchange(self):
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_RING):
            self.assertEqual(
                wx.weights_region_tag_for(is_draft_model_runner=True),
                GPU_MEMORY_TYPE_WEIGHTS,
            )
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertEqual(
                wx.weights_region_tag_for(is_draft_model_runner=True),
                GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )
            self.assertEqual(
                wx.weights_region_tag_for(is_draft_model_runner=False),
                GPU_MEMORY_TYPE_WEIGHTS,
            )

    def test_draft_region_tag_survives_a_chunk_scope(self):
        """``Qwen3_5ForCausalLMMTP.__init__`` builds a one-layer
        ``Qwen3_5ForCausalLM`` (qwen3_5_mtp.py:277-283), so the draft's block
        goes through ``make_layers`` -> ``weight_chunk_scope(0)``
        (utils/common.py:2017).  Under the base region that is right; under the
        draft region it would tag the draft's own layer ``weights_0`` -- back
        inside the family, and the scope's exit would then restore the BASE tag
        for everything built after it."""
        fake = self.__class__._FakeCdll()
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"
        try:
            with unittest.mock.patch.object(
                wms, "_tms_cdll_in_region", lambda: fake
            ):
                # Base region: unchanged behaviour, chunk tag then base tag.
                with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS):
                    with wms.weight_chunk_scope(0) as tag:
                        self.assertEqual(tag, "weights_0")
                        self.assertEqual(fake.tag, "weights_0")
                    self.assertEqual(fake.tag, GPU_MEMORY_TYPE_WEIGHTS)
                # Draft region: the scope must not steal the layer, and must
                # not restore the base tag over the draft tag.
                fake.sets.clear()
                with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT):
                    with wms.weight_chunk_scope(0) as tag:
                        self.assertIsNone(tag)
                    self.assertNotIn(
                        "weights_0",
                        fake.sets,
                        "the draft's layer was tagged into the family",
                    )
                    self.assertNotIn(
                        GPU_MEMORY_TYPE_WEIGHTS,
                        fake.sets,
                        "the chunk scope restored the BASE tag inside the draft region",
                    )
        finally:
            os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
            os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)

    def test_region_tag_is_restored_after_the_scope(self):
        self.assertEqual(
            wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS
        )
        with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT):
            self.assertEqual(
                wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS_DRAFT
            )
        self.assertEqual(
            wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS
        )


class CoverageTest(unittest.TestCase):
    def test_uncovered_tensor_refuses(self):
        """A stray ``torch.Tensor`` attribute inside a layer's module is a page
        with no source: W51, by name, with the module path in the message."""
        model = _Model()
        model.layers[1].scratch = torch.zeros(3 * MIB, dtype=torch.uint8)
        with self.assertRaises(wms.Weg2XchgCoverageRefused) as ctx:
            wx.arm_coverage(
                model,
                rank=0,
                planned_names_by_tag=_planned_names(model),
                tag_bytes=_tag_bytes_stub(),
                log=_CaptureLog(),
            )
        msg = str(ctx.exception)
        self.assertIn("W51", msg)
        self.assertIn("Weg2XchgCoverageRefused", msg)
        self.assertIn("layers.1.scratch", msg)

    def test_a_parameter_the_plan_does_not_carry_refuses(self):
        model = _Model()
        planned = _planned_names(model, drop=("layers.1.o_proj",))
        with self.assertRaises(wms.Weg2XchgCoverageRefused) as ctx:
            wx.arm_coverage(
                model,
                rank=0,
                planned_names_by_tag=planned,
                tag_bytes=_tag_bytes_stub(),
                log=_CaptureLog(),
            )
        self.assertIn("layers.1.o_proj", str(ctx.exception))

    def test_rope_cache_counts_as_a_buffer_not_slack(self):
        """The >=256 MiB rope cache is a REGISTERED BUFFER, carried across the
        flip by ``_export_static_state``; it belongs in ``buffers_mib`` and
        must not be charged to ``slack_mib``, which is the allocator overhang."""
        model = _Model(rope_mib=256)
        rows = wx.arm_coverage(
            model,
            rank=0,
            planned_names_by_tag=_planned_names(model),
            # 256 MiB of buffer + 12 MiB of parameters under weights_0, plus
            # 7 MiB of allocator slack.
            tag_bytes=lambda tag: int((256 + 12 + 7) * MIB) if tag == "weights_0" else 0,
            log=_CaptureLog(),
        )
        row = rows["weights_0"]
        self.assertAlmostEqual(row.buffers_bytes / MIB, 256.0, places=3)
        self.assertAlmostEqual(row.slack_bytes / MIB, 7.0, places=3)
        self.assertEqual(row.uncovered, ())

    def test_slack_is_printed_never_asserted(self):
        """+0.08 to +0.58 GiB/rank is measured and normal; an equality assert
        on the census would refuse every boot.  A NEGATIVE slack (the census
        answering 0 because the saver has no such symbol) is likewise printed,
        never raised -- the absence is the caller's to read."""
        model = _Model()
        log = _CaptureLog()
        rows = wx.arm_coverage(
            model,
            rank=0,
            planned_names_by_tag=_planned_names(model),
            tag_bytes=lambda tag: int(600 * MIB),
            log=log,
        )
        self.assertGreater(rows["weights_0"].slack_bytes, 0)
        rows = wx.arm_coverage(
            model,
            rank=0,
            planned_names_by_tag=_planned_names(model),
            tag_bytes=lambda tag: 0,
            log=log,
        )
        self.assertLess(rows["weights_0"].slack_bytes, 0)

    def test_alias_view_of_a_covered_tensor_is_not_uncovered(self):
        """``self.lm_head = self.model.embed_tokens`` is the shape in the tree
        (qwen3_5_mtp.py:288).  A stray attribute that is a VIEW of bytes the
        plan already carries is covered, and must not be counted twice."""
        model = _Model()
        model.layers[0].qkv_view = model.layers[0].qkv_proj.data[: 1 * MIB]
        rows = wx.arm_coverage(
            model,
            rank=0,
            planned_names_by_tag=_planned_names(model),
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertEqual(rows["weights_0"].uncovered, ())
        self.assertAlmostEqual(rows["weights_0"].planned_bytes / MIB, 12.0, places=3)

    def test_cover_line_is_emitted_verbatim_per_tag(self):
        model = _Model(rope_mib=8)
        log = _CaptureLog()
        wx.arm_coverage(
            model,
            rank=2,
            planned_names_by_tag=_planned_names(model),
            tag_bytes=lambda tag: int(30 * MIB),
            log=log,
        )
        cover = [l for l in log.lines if l.startswith("WEG2-XCHG-COVER ")]
        self.assertTrue(cover)
        line = [l for l in cover if "tag=weights_0" in l][0]
        # The spec's field order, verbatim.  Extra fields are APPENDED after
        # uncovered= (the denominator law: a population count names its own
        # population); the greppable prefix is unchanged.
        for field in (
            "rank=2",
            "tag=weights_0",
            "planned_mib=",
            "buffers_mib=",
            "tms_mib=",
            "slack_mib=",
            "uncovered=0",
        ):
            self.assertIn(field, line)
        head = line.split("uncovered=")[0]
        self.assertLess(head.index("planned_mib="), head.index("buffers_mib="))
        self.assertLess(head.index("buffers_mib="), head.index("tms_mib="))
        self.assertLess(head.index("tms_mib="), head.index("slack_mib="))
        # Every exchanged tag of this model gets a line, on this rank.
        self.assertEqual(
            sorted(l.split("tag=")[1].split(" ")[0] for l in cover),
            ["weights", "weights_0"],
        )

    def test_the_draft_tag_is_never_a_covered_population(self):
        """Coverage is asked about EXCHANGED tags only.  ``weights_draft`` is
        not one, so it never appears in the rows and its bytes are never
        charged to a family tag's slack."""
        model = _Model()
        rows = wx.arm_coverage(
            model,
            rank=0,
            planned_names_by_tag=_planned_names(model),
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertNotIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, rows)
        for tag in rows:
            self.assertTrue(wms.is_weights_family_tag(tag), tag)


class PlanInterfaceTest(unittest.TestCase):
    """The minimal interface S1 owns.  Pinned here so the two slices meet.

    TODO(S1, branch weg2/xchg-s1-0908): ``weight_exchange.build_plan()`` must
    expose exactly this shape -- ``{tag: {parameter name, ...}}`` over
    ``named_parameters()`` names, one entry per exchanged tag.  S2 consumes it
    and nothing else of the plan.
    """

    def test_plan_interface_is_tag_to_parameter_names(self):
        model = _Model()
        planned = _planned_names(model)
        self.assertEqual(sorted(planned), ["weights", "weights_0"])
        self.assertIn("layers.0.qkv_proj", planned["weights_0"])
        self.assertIn("embed_tokens", planned["weights"])
        # A plain dict of plain sets is enough: no plan object is imported.
        wx.arm_coverage(
            model,
            rank=0,
            planned_names_by_tag={k: set(v) for k, v in planned.items()},
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )

    def test_tag_of_parameter_name_uses_the_tree_helpers(self):
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"
        try:
            self.assertEqual(wx.tag_of_parameter_name("layers.0.qkv_proj"), "weights_0")
            self.assertEqual(wx.tag_of_parameter_name("layers.9.qkv_proj"), "weights_1")
            # The clamp is the tree's (weight_chunk_tag), not a reimplementation.
            self.assertEqual(wx.tag_of_parameter_name("layers.99.q"), "weights_7")
            self.assertEqual(wx.tag_of_parameter_name("embed_tokens"), "weights")
        finally:
            os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
            os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)


class WCodeTest(unittest.TestCase):
    def test_w51_is_the_coverage_refusal_and_says_so_once(self):
        self.assertIn("W51", wms.Weg2XchgCoverageRefused.__doc__ or "")
        self.assertEqual(wx.COVERAGE_REFUSAL_MARKER, "W51 Weg2XchgCoverageRefused")


if __name__ == "__main__":
    unittest.main()
