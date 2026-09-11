# SPDX-License-Identifier: Apache-2.0
"""#1273 slice S2 -- coverage arming (W84) and the draft tag.

WEG2_REUSE_SPEC_0908.md section 6/S2 and section 4.1.  Two properties, and both
of them are about a SILENT wrongness, which is why they are tests and not a
boot observation:

1. **Coverage.**  The exchange fills the destination's weight pages from the
   source's VRAM.  A live tensor under an exchanged tag that the plan does not
   carry and that ``_export_static_state`` does not carry either is a page the
   destination never receives -- plausible garbage, no error, no crash.  The
   arming check walks ``named_parameters()`` + ``named_buffers()`` + a sweep of
   every module's ``__dict__`` for stray ``torch.Tensor`` attributes and
   refuses by name (**W84 Weg2XchgCoverageRefused**) rather than ship a tag it
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

import contextlib
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
        self.qkv_proj = nn.Parameter(
            torch.zeros(4 * MIB, dtype=torch.uint8), requires_grad=False
        )
        self.o_proj = nn.Parameter(
            torch.zeros(2 * MIB, dtype=torch.uint8), requires_grad=False
        )
        if rope_mib:
            self.rotary_emb = _Rope(rope_mib)


class _Model(nn.Module):
    def __init__(self, n_layers: int = 2, rope_mib: int = 0):
        super().__init__()
        self.layers = nn.ModuleList(
            [_Layer(rope_mib if i == 0 else 0) for i in range(n_layers)]
        )
        self.embed_tokens = nn.Parameter(
            torch.zeros(8 * MIB, dtype=torch.uint8), requires_grad=False
        )


def _planned_bytes(model, *, drop=(), short=None, region_tag=GPU_MEMORY_TYPE_WEIGHTS):
    """The S1 interface, built here the way S1's plan builder will build it:
    ``{tag: {parameter name: PLANNED BYTES}}`` over ``named_parameters()``,
    the bytes being the sum of that parameter's ``XchgDesc.nbytes``.

    ``short`` seeds the risk-R5 shape: a parameter the plan covers by name and
    only partly by bytes.
    """
    short = dict(short or {})
    out = {}
    for name, p in model.named_parameters():
        if name in drop:
            continue
        tag = wx.tag_of_parameter_name(name, region_tag=region_tag)
        out.setdefault(tag, {})[name] = int(
            short.get(name, p.untyped_storage().nbytes())
        )
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


class _ChunkedCase(unittest.TestCase):
    """The launcher's chunk geometry, as the Weg-2 form publishes it: 8 layers
    per chunk, 8 chunks (WEG2_REUSE_SPEC_0908.md section 1.2, the sb4 form).
    Both of this double's layers therefore land in ``weights_0`` and the
    non-layer parameters in the base tag."""

    def setUp(self):
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"

    def tearDown(self):
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)



class DraftTagOutOfFamilyTest(unittest.TestCase):
    """The RING arm's family, and it is unchanged.

    NARROWED BY #1273 B4k / spec AMENDMENT 6: what this class pins is now the
    ring arm only.  Under ``exchange`` ``weights_draft`` IS a family tag (both
    groups were measured holding the MTP head), and that half lives in
    ``test_weg2_xchg_draft_family_1273.py``.  The assertions below hold here
    because no test in this file arms the exchange around them -- the predicate
    is form-gated through ``weg2_memory_saver.draft_tag_in_family``.

    What has NOT changed, and is the reason this class stays: the family is
    ``weights`` plus ``weights_<integer>`` plus AT MOST that one measured tag.
    A tag name is still not a predicate, and S8's ``weights_vision`` is still
    out on both arms.
    """

    def test_draft_tag_is_not_in_the_weights_family_under_the_RING_arm(self):
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


class ResidentLineTest(unittest.TestCase):
    """The RESIDENT line is an OBSERVATION, not a restatement of a constant.

    Refuter F1: the first form hardcoded ``tag=weights_draft`` at the emit site
    and then computed ``in_family`` from that literal, so the line printed
    ``in_family=no`` whether or not the draft tag had ever been applied, and
    printed the same under ``ring``, where the drafter IS in the family.
    """

    def test_resident_line_reports_the_tag_it_is_given(self):
        line = wx.resident_line(
            tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            mib=1311.0,
            rank=1,
            mode=wx.WEIGHT_SOURCE_EXCHANGE,
        )
        self.assertTrue(line.startswith("WEG2-XCHG-RESIDENT "))
        # The spec's own adjacency, so its grep matches: tag= then mib=.
        self.assertIn("tag=weights_draft mib=1311.0", line)
        self.assertIn("in_family=no", line)
        self.assertIn("rank=1", line)
        self.assertIn("mode=exchange", line)

    def test_the_same_line_says_in_family_yes_for_a_family_tag(self):
        """THE DANGER DIRECTION: if ``in_family`` were a property of a constant
        this module owns, this assertion could not exist."""
        line = wx.resident_line(
            tag=GPU_MEMORY_TYPE_WEIGHTS,
            mib=1382.0,
            rank=0,
            mode=wx.WEIGHT_SOURCE_RING,
        )
        self.assertIn("tag=weights mib=1382.0", line)
        self.assertIn("in_family=yes", line)
        self.assertIn("mode=ring", line)


class RunnerShapeTest(unittest.TestCase):
    """WHICH runner gets the draft tag.  ``is_draft_worker`` is a construction
    gate with several producers and only one of them holds draft weights
    (model_runner.py:514-521); classifying is the fix, subtracting one producer
    was the defect (refuter F2)."""

    DRAFT = wx.RunnerShape(is_draft_worker=True, speculative_configured=True)
    PRIMARY = wx.RunnerShape(is_draft_worker=False, speculative_configured=True)
    LANE = wx.RunnerShape(
        is_draft_worker=True, is_dual_group_lane=True, speculative_configured=True
    )
    FLIP_TP = wx.RunnerShape(
        is_draft_worker=True,
        is_phase_flip_tp_stack=True,
        speculative_configured=True,
    )
    UNKNOWN = wx.RunnerShape(is_draft_worker=True, speculative_configured=False)

    def test_classification_names_every_producer_of_the_construction_gate(self):
        self.assertEqual(wx.classify_runner(self.PRIMARY), wx.SHAPE_PRIMARY)
        self.assertEqual(wx.classify_runner(self.DRAFT), wx.SHAPE_DRAFT)
        self.assertEqual(wx.classify_runner(self.LANE), wx.SHAPE_DUAL_GROUP_LANE)
        self.assertEqual(
            wx.classify_runner(self.FLIP_TP), wx.SHAPE_PHASE_FLIP_TP_STACK
        )
        self.assertIsNone(wx.classify_runner(self.UNKNOWN))

    def test_the_draft_tag_is_the_drafters_only_under_exchange(self):
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_RING):
            self.assertEqual(
                wx.weights_region_tag_for(self.DRAFT), GPU_MEMORY_TYPE_WEIGHTS
            )
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertEqual(
                wx.weights_region_tag_for(self.DRAFT), GPU_MEMORY_TYPE_WEIGHTS_DRAFT
            )
            self.assertEqual(
                wx.weights_region_tag_for(self.PRIMARY), GPU_MEMORY_TYPE_WEIGHTS
            )

    def test_a_dual_group_lane_hull_never_gets_the_draft_tag(self):
        """#274's lane is constructed with ``is_draft_worker=True`` and
        ``is_phase_flip_tp_stack=False`` (model_runner.py:486-489) and its model
        is the assembled FULL-WIDTH TARGET hull (``build_lane_model``,
        :2496-2513).  Under the first form of this predicate it was a
        'drafter': the whole target out of the weights family, never paused,
        never exchanged, permanently resident, and no refusal naming it."""
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertEqual(
                wx.weights_region_tag_for(self.LANE), GPU_MEMORY_TYPE_WEIGHTS
            )

    def test_a_phase_flip_tp_stack_never_gets_the_draft_tag(self):
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertEqual(
                wx.weights_region_tag_for(self.FLIP_TP), GPU_MEMORY_TYPE_WEIGHTS
            )

    def test_an_unclassifiable_secondary_runner_refuses_under_exchange(self):
        """A fourth producer of the construction gate: sets no known exclusion
        and the boot carries no speculative config.  Both guesses are wrong in
        a way that only shows at the next flip, so neither is made."""
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            with self.assertRaises(wx.Weg2XchgRunnerShapeUnknown) as ctx:
                wx.weights_region_tag_for(self.UNKNOWN)
            msg = str(ctx.exception)
            self.assertIn("W76", msg)
            self.assertIn("Weg2XchgRunnerShapeUnknown", msg)
            self.assertIn("is_draft_worker=True", msg)

    def test_the_same_unclassifiable_runner_is_untouched_under_ring(self):
        """``ring`` is today, byte for byte -- including for a shape this
        predicate cannot name."""
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_RING):
            self.assertEqual(
                wx.weights_region_tag_for(self.UNKNOWN), GPU_MEMORY_TYPE_WEIGHTS
            )

    def test_runner_shape_is_read_off_the_runner_by_name(self):
        class _FakeArgs:
            speculative_algorithm = "NEXTN"

        class _FakeRunner:
            is_draft_worker = True
            is_phase_flip_tp_stack = False
            is_dual_group_lane = True
            server_args = _FakeArgs()

        shape = wx.RunnerShape.of(_FakeRunner())
        self.assertTrue(shape.is_draft_worker)
        self.assertTrue(shape.is_dual_group_lane)
        self.assertTrue(shape.speculative_configured)
        self.assertEqual(wx.classify_runner(shape), wx.SHAPE_DUAL_GROUP_LANE)


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

    class _FakeAdapter:
        """A ``memory_saver_adapter``-shaped double: it records the tag the
        region was opened with, and the region tag published while it was
        open."""

        def __init__(self):
            self.opened = []
            self.published_while_open = []

        @contextlib.contextmanager
        def region(self, tag, enable_cpu_backup=False):
            self.opened.append((tag, enable_cpu_backup))
            self.published_while_open.append(wms.current_weights_region_tag())
            yield

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
            with unittest.mock.patch.object(wms, "_tms_cdll_in_region", lambda: fake):
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
        self.assertEqual(wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS)
        with wms.weights_region_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT):
            self.assertEqual(
                wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS_DRAFT
            )
        self.assertEqual(wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS)

    def test_one_opener_publishes_exactly_the_tag_it_opens(self):
        """Refuter F6: publishing the tag and opening the region were two
        statements at two call sites, and one of them kept a hardcoded base
        tag.  One opener makes them impossible to disagree."""
        adapter = self.__class__._FakeAdapter()
        for tag in (GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_WEIGHTS_DRAFT):
            with wms.weights_region(adapter, tag, enable_cpu_backup=False) as got:
                self.assertEqual(got, tag)
                self.assertEqual(wms.current_weights_region_tag(), tag)
            self.assertEqual(wms.current_weights_region_tag(), GPU_MEMORY_TYPE_WEIGHTS)
        self.assertEqual(
            adapter.opened,
            [(GPU_MEMORY_TYPE_WEIGHTS, False), (GPU_MEMORY_TYPE_WEIGHTS_DRAFT, False)],
        )
        # PUBLISHED BEFORE THE REGION OPENS: an allocation made inside must
        # already see the tag.
        self.assertEqual(
            adapter.published_while_open,
            [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_WEIGHTS_DRAFT],
        )

    def test_the_opener_refuses_a_tag_that_is_not_a_weights_region_tag(self):
        adapter = self.__class__._FakeAdapter()
        for bad in ("weights_0", "kv_cache", "weights_vision"):
            with self.assertRaises(ValueError, msg=bad):
                with wms.weights_region(adapter, bad, enable_cpu_backup=False):
                    pass
        self.assertEqual(adapter.opened, [])


class RegionAwareTagTest(_ChunkedCase):
    """Review F1 / refuter F4: the tag a tensor lives under is decided by the
    REGION, and only inside the base region is it also decided by the name."""

    def test_tag_of_parameter_name_follows_the_region_not_the_name(self):
        # Base region: name-derived, as before.
        self.assertEqual(wx.tag_of_parameter_name("layers.0.qkv_proj"), "weights_0")
        self.assertEqual(wx.tag_of_parameter_name("embed_tokens"), "weights")
        # Draft region: `weight_chunk_scope` is a no-op there, so the drafter's
        # `layers.0.*` block is ALLOCATED under the draft tag while its NAME
        # still says layers.0.  A region-blind reading answers weights_0 and
        # censuses a family tag that does not exist in that process.
        for name in ("layers.0.qkv_proj", "embed_tokens", "mtp.model.layers.0.o_proj"):
            self.assertEqual(
                wx.tag_of_parameter_name(
                    name, region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT
                ),
                GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                name,
            )

    def test_the_walk_charges_a_draft_region_to_the_draft_tag(self):
        model = _Model()
        live = wx.walk_live_tensors(model, region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
        self.assertTrue(live)
        self.assertEqual(
            {t.tag for t in live}, {GPU_MEMORY_TYPE_WEIGHTS_DRAFT}
        )

    def test_coverage_under_a_draft_region_has_no_family_rows(self):
        """THE DANGER DIRECTION.  With a region-blind tag the drafter's
        parameters build family rows, get compared against
        ``tms_tag_bytes('weights_0')`` on a process that has no such tag, and
        W84 can fire over a population that is out of family by construction --
        the exact opposite of section 4.1's purpose."""
        model = _Model()
        rows = wx.build_coverage(
            model,
            rank=0,
            planned_bytes_by_tag={},
            tag_bytes=_tag_bytes_stub(),
            region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
        )
        self.assertEqual(rows, {})


class CoverageTest(_ChunkedCase):
    def test_uncovered_tensor_refuses(self):
        """A stray ``torch.Tensor`` attribute inside a layer's module is a page
        with no source: W84, by name, with the module path in the message."""
        model = _Model()
        model.layers[1].scratch = torch.zeros(3 * MIB, dtype=torch.uint8)
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=_planned_bytes(model),
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertFalse(vote.ok)
        with self.assertRaises(wms.Weg2XchgCoverageRefused) as ctx:
            wx.refuse_if_not_ok(vote)
        msg = str(ctx.exception)
        self.assertIn("W84", msg)
        self.assertIn("Weg2XchgCoverageRefused", msg)
        self.assertIn("layers.1.scratch", msg)

    def test_a_parameter_the_plan_does_not_carry_refuses(self):
        model = _Model()
        planned = _planned_bytes(model, drop=("layers.1.o_proj",))
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=planned,
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertFalse(vote.ok)
        self.assertIn("layers.1.o_proj", vote.reason)

    def test_a_partially_tiled_parameter_refuses(self):
        """RISK R5, and the reason the plan interface is BYTES and not names
        (refuter F3): a plan whose ``in_proj_qkvz`` carries three device
        sub-blocks instead of four covers the NAME completely and the BYTES
        partly.  Under a name-only check it passed with ``uncovered=0`` and a
        plausible slack; the destination's fourth block then held whatever the
        arena held."""
        model = _Model()
        planned = _planned_bytes(model, short={"layers.0.qkv_proj": 3 * MIB})
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=planned,
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertFalse(vote.ok)
        self.assertIn("SHORT", vote.reason)
        self.assertIn("layers.0.qkv_proj", vote.reason)
        row = vote.rows["weights_0"]
        self.assertEqual(len(row.short), 1)
        self.assertEqual(row.short[0].planned_bytes, 3 * MIB)
        self.assertEqual(row.short[0].live_bytes, 4 * MIB)
        self.assertEqual(row.uncovered, ())

    def test_planned_mib_is_the_plans_claim_not_the_models_storage(self):
        """``planned_mib`` names the PLAN.  A short claim must show up as
        slack, not be silently replaced by the storage size."""
        model = _Model()
        planned = _planned_bytes(model, short={"layers.0.qkv_proj": 3 * MIB})
        rows = wx.build_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=planned,
            tag_bytes=lambda tag: 0,
        )
        # weights_0 holds 2 layers: qkv 4 + o 2 + qkv 4 + o 2 = 12 MiB live,
        # of which the plan claims one MiB less than it should.
        self.assertAlmostEqual(rows["weights_0"].planned_bytes / MIB, 11.0, places=3)

    def test_a_plan_name_with_no_live_tensor_refuses(self):
        """The other half of the coverage relation: a plan built against a
        different shard geometry names parameters this rank does not have."""
        model = _Model()
        planned = _planned_bytes(model)
        planned["weights_0"] = dict(planned["weights_0"])
        planned["weights_0"]["layers.0.in_proj_qkvz"] = 7 * MIB
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=planned,
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertFalse(vote.ok)
        self.assertIn("MISSING", vote.reason)
        self.assertIn("layers.0.in_proj_qkvz", vote.reason)

    def test_rope_cache_counts_as_a_buffer_not_slack(self):
        """The >=256 MiB rope cache is a REGISTERED BUFFER, carried across the
        flip by ``_export_static_state``; it belongs in ``buffers_mib`` and
        must not be charged to ``slack_mib``, which is the allocator overhang."""
        model = _Model(rope_mib=256)
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=_planned_bytes(model),
            # 256 MiB of buffer + 12 MiB of parameters under weights_0, plus
            # 7 MiB of allocator slack.
            tag_bytes=lambda tag: (
                int((256 + 12 + 7) * MIB) if tag == "weights_0" else 0
            ),
            log=_CaptureLog(),
        )
        row = vote.rows["weights_0"]
        self.assertAlmostEqual(row.buffers_bytes / MIB, 256.0, places=3)
        self.assertAlmostEqual(row.slack_bytes / MIB, 7.0, places=3)
        self.assertEqual(row.uncovered, ())
        self.assertTrue(vote.ok)

    def test_slack_is_printed_never_asserted(self):
        """+0.08 to +0.58 GiB/rank is measured and normal; an equality assert
        on the census would refuse every boot.  A NEGATIVE slack (the census
        answering 0 because the saver has no such symbol) is likewise printed,
        never raised -- the absence is the caller's to read."""
        model = _Model()
        log = _CaptureLog()
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=_planned_bytes(model),
            tag_bytes=lambda tag: int(600 * MIB),
            log=log,
        )
        self.assertGreater(vote.rows["weights_0"].slack_bytes, 0)
        self.assertTrue(vote.ok)
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=_planned_bytes(model),
            tag_bytes=lambda tag: 0,
            log=log,
        )
        self.assertLess(vote.rows["weights_0"].slack_bytes, 0)
        self.assertTrue(vote.ok)

    def test_alias_view_of_a_covered_tensor_is_not_uncovered(self):
        """``self.lm_head = self.model.embed_tokens`` is the shape in the tree
        (qwen3_5_mtp.py:288).  A stray attribute that is a VIEW of bytes the
        plan already carries is covered, and must not be counted twice."""
        model = _Model()
        model.layers[0].qkv_view = model.layers[0].qkv_proj.data[: 1 * MIB]
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=_planned_bytes(model),
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertEqual(vote.rows["weights_0"].uncovered, ())
        self.assertAlmostEqual(
            vote.rows["weights_0"].planned_bytes / MIB, 12.0, places=3
        )

    def test_cover_line_is_emitted_verbatim_per_tag(self):
        model = _Model(rope_mib=8)
        log = _CaptureLog()
        wx.arm_coverage(
            model,
            rank=2,
            planned_bytes_by_tag=_planned_bytes(model),
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
            "short=0",
            "missing=0",
            "mode=",
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
        charged to a family tag's slack -- not even when a (wrong) plan tries
        to put it there."""
        model = _Model()
        planned = _planned_bytes(model)
        planned[GPU_MEMORY_TYPE_WEIGHTS_DRAFT] = {"mtp.model.layers.0.o_proj": MIB}
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=planned,
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertNotIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, vote.rows)
        for tag in vote.rows:
            self.assertTrue(wms.is_weights_family_tag(tag), tag)


class VoteTest(_ChunkedCase):
    """Refuter F5: derivation is rank-local, the DECISION belongs to a fence."""

    def test_arm_coverage_votes_and_never_raises(self):
        model = _Model()
        model.layers[1].scratch = torch.zeros(3 * MIB, dtype=torch.uint8)
        vote = wx.arm_coverage(
            model,
            rank=4,
            planned_bytes_by_tag=_planned_bytes(model),
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertIsInstance(vote, wx.CoverageVote)
        self.assertFalse(vote.ok)
        self.assertEqual(vote.rank, 4)
        self.assertIn("W84", vote.reason)

    def test_refuse_if_not_ok_raises_only_for_a_failing_vote(self):
        model = _Model()
        good = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag=_planned_bytes(model),
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertIs(wx.refuse_if_not_ok(good), good)
        bad = wx.CoverageVote(
            rank=0,
            mode=wx.WEIGHT_SOURCE_EXCHANGE,
            region_tag=GPU_MEMORY_TYPE_WEIGHTS,
            rows={},
            ok=False,
            reason=wx.NO_PLAN_REASON,
        )
        with self.assertRaises(wms.Weg2XchgCoverageRefused):
            wx.refuse_if_not_ok(bad)


class ArmAtLoadTest(_ChunkedCase):
    """Review F2: the wired call site.  An instrument nobody calls measures
    nothing, and the standing order is that building a tool means wiring it."""

    def tearDown(self):
        super().tearDown()
        wx.register_plan_provider(None)

    def test_arm_at_load_is_a_no_op_under_ring(self):
        model = _Model()
        log = _CaptureLog()
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_RING):
            self.assertIsNone(
                wx.arm_coverage_at_load(
                    model,
                    rank=0,
                    tag_bytes=_tag_bytes_stub(),
                    region_tag=GPU_MEMORY_TYPE_WEIGHTS,
                    log=log,
                )
            )
        self.assertEqual(log.lines, [])
        self.assertIsNone(wx.boot_vote())

    def test_arm_at_load_emits_resident_and_cover_and_records_the_vote(self):
        model = _Model()
        log = _CaptureLog()
        wx.register_plan_provider(lambda m: _planned_bytes(m))
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            vote = wx.arm_coverage_at_load(
                model,
                rank=1,
                tag_bytes=lambda tag: int(30 * MIB),
                region_tag=GPU_MEMORY_TYPE_WEIGHTS,
                log=log,
            )
        self.assertTrue(vote.ok)
        self.assertIs(wx.boot_vote(), vote)
        resident = [l for l in log.lines if l.startswith("WEG2-XCHG-RESIDENT ")]
        self.assertEqual(len(resident), 1)
        # MEASURED, not passed in: the mib is the saver's census for the tag
        # the region was opened with.
        self.assertIn("tag=weights mib=30.0", resident[0])
        self.assertIn("in_family=yes", resident[0])
        self.assertIn("rank=1", resident[0])
        self.assertIn("mode=exchange", resident[0])
        self.assertEqual(
            len([l for l in log.lines if l.startswith("WEG2-XCHG-COVER ")]), 2
        )

    def test_arm_at_load_under_the_draft_region_now_CENSUSES_it(self):
        """SUPERSEDED BY #1273 B4k / spec AMENDMENT 6, and the reversal is the
        point rather than a regression.

        This test used to assert ``resident only``: one RESIDENT line,
        ``in_family=no``, ``rows == {}`` -- the draft tag waved off because
        section 4.1 said group P had no VRAM source for those bytes.  Boot
        weg2xsn15 measured the opposite on both groups (``WEG2-XCHG-RESIDENT
        tag=weights_draft`` 1440/1280/1280 MiB on D and 1572 MiB on P's last
        stage), the user overruled 4.1, and the draft head now travels the
        family like every other layer.  So the arm CENSUSES this runner: a
        COVER line beside the RESIDENT line, ``in_family=yes``, and real rows.
        The ring arm is unchanged and is asserted separately
        (``test_weg2_xchg_draft_family_1273.py``).
        """
        model = _Model()
        log = _CaptureLog()
        planned = {
            GPU_MEMORY_TYPE_WEIGHTS_DRAFT: {
                name: int(p.untyped_storage().nbytes())
                for name, p in model.named_parameters()
            }
        }
        wx.register_plan_provider(lambda _m: planned)
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            vote = wx.arm_coverage_at_load(
                model,
                rank=2,
                tag_bytes=lambda tag: int(1311 * MIB),
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                log=log,
            )
        self.assertTrue(vote.ok, vote.reason)
        self.assertIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, vote.rows)
        resident = [ln for ln in log.lines if wx.RESIDENT_LINE_PREFIX in ln]
        cover = [ln for ln in log.lines if wx.COVER_LINE_PREFIX in ln]
        self.assertEqual(len(resident), 1, log.lines)
        self.assertIn("tag=weights_draft mib=1311.0", resident[0])
        self.assertIn("in_family=yes", resident[0])
        self.assertEqual(len(cover), 1, log.lines)
        self.assertIn("tag=weights_draft", cover[0])
        self.assertIn("uncovered=0", cover[0])

    def test_arm_at_load_installs_the_plan_provider(self):
        """S6 step 6b: the arm SELF-ARMS, so `exchange` is no longer
        fail-closed BY OMISSION.

        THIS TEST USED TO ASSERT THE OPPOSITE, and the change is the whole
        point of step 6b rather than a regression: before it,
        `register_plan_provider` had no registrant anywhere in the tree, so
        every rank under `exchange` voted not-ok with NO_PLAN_REASON -- a
        deliberate refusal, but not a working arm, and S1's own TODO said so.
        `arm_coverage_at_load` now installs `default_plan_provider` (which
        derives through the shadow's producer) when nothing is registered.

        The not-ok vote has NOT disappeared; it moved to the two places that
        can still honestly produce it, and both are tested in
        `test_weg2_xchg_plan_provider_1273`: a derivation that refuses, and a
        rank with no Weg-2 group identity. What is gone is "nobody ever
        registered anything".
        """
        model = _Model()
        log = _CaptureLog()
        wx.register_plan_provider(None)
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            wx.arm_coverage_at_load(
                model,
                rank=0,
                tag_bytes=_tag_bytes_stub(),
                region_tag=GPU_MEMORY_TYPE_WEIGHTS,
                log=log,
            )
        # And the provider's own refusal (no group identity in this process)
        # arrives as a NOT-OK VOTE, never as a raise: there is no group fence
        # here, so a rank-local raise would strand the other five.
        self.assertIsNotNone(
            wx.plan_provider(),
            "the arm left this rank without a plan provider, which is the "
            "state S1's TODO described and step 6b closed")
        wx.register_plan_provider(None)

    def test_the_ring_arm_still_installs_no_provider(self):
        """The default path pays nothing, as every other S2 line does."""
        wx.register_plan_provider(None)
        wx.arm_coverage_at_load(
            _Model(),
            rank=0,
            tag_bytes=_tag_bytes_stub(),
            region_tag=GPU_MEMORY_TYPE_WEIGHTS,
            log=_CaptureLog(),
        )
        self.assertIsNone(wx.plan_provider())

    def test_arm_at_load_does_not_raise_where_there_is_no_fence(self):
        model = _Model()
        model.layers[1].scratch = torch.zeros(3 * MIB, dtype=torch.uint8)
        wx.register_plan_provider(lambda m: _planned_bytes(m))
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            vote = wx.arm_coverage_at_load(
                model,
                rank=0,
                tag_bytes=_tag_bytes_stub(),
                region_tag=GPU_MEMORY_TYPE_WEIGHTS,
                log=_CaptureLog(),
            )
        self.assertFalse(vote.ok)


class RollForwardTagTest(unittest.TestCase):
    """Refuter F6: W73's roll-forward opens ONE region for BOTH shards."""

    def test_roll_forward_is_unchanged_under_ring(self):
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_RING):
            for has_draft in (True, False):
                self.assertEqual(
                    wx.roll_forward_weights_tag(has_draft_shard=has_draft),
                    GPU_MEMORY_TYPE_WEIGHTS,
                )

    def test_roll_forward_refuses_under_exchange_with_a_draft_shard(self):
        with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertIsNone(wx.roll_forward_weights_tag(has_draft_shard=True))
            self.assertEqual(
                wx.roll_forward_weights_tag(has_draft_shard=False),
                GPU_MEMORY_TYPE_WEIGHTS,
            )
        msg = wx.roll_forward_refusal_message()
        self.assertIn("weights_draft", msg)
        self.assertIn("W74", msg)
        self.assertIn("OWNER: S6", msg)


class WiringTest(unittest.TestCase):
    """The two production call sites, pinned by source so a later edit that
    quietly drops them is a RED test rather than an unwired instrument."""

    @staticmethod
    def _source(relative: str) -> str:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        with open(os.path.join(root, "python", "sglang", "srt", relative)) as fh:
            return fh.read()

    def test_model_runner_opens_the_region_through_the_one_opener_and_arms(self):
        src = self._source("model_executor/model_runner.py")
        self.assertIn("weights_region_tag_for(RunnerShape.of(self))", src)
        self.assertIn("with weights_region(", src)
        self.assertIn("arm_coverage_at_load(", src)

    def test_the_roll_forward_derives_its_region_tag(self):
        src = self._source("managers/scheduler_components/weight_updater.py")
        self.assertIn("roll_forward_weights_tag(", src)
        self.assertIn("with weights_region(", src)
        # The hardcoded base-tag region this replaced must be gone from the
        # reload path.
        self.assertNotIn(
            "with self.memory_saver_adapter.region(\n            GPU_MEMORY_TYPE_WEIGHTS,\n            enable_cpu_backup=False,\n        ):",
            src,
        )


class PlanInterfaceTest(_ChunkedCase):
    """The minimal interface S1 owns.  Pinned here so the two slices meet.

    TODO(S1, branch weg2/xchg-s1-0908): ``weight_exchange.build_plan()`` must
    expose exactly this shape -- ``{tag: {parameter name: planned bytes}}``,
    the bytes being the sum of that parameter's ``XchgDesc.nbytes``.  S2
    consumes it and nothing else of the plan.
    """

    def test_plan_interface_is_tag_to_parameter_bytes(self):
        model = _Model()
        planned = _planned_bytes(model)
        self.assertEqual(sorted(planned), ["weights", "weights_0"])
        self.assertEqual(planned["weights_0"]["layers.0.qkv_proj"], 4 * MIB)
        self.assertEqual(planned["weights"]["embed_tokens"], 8 * MIB)
        # A plain dict of plain dicts is enough: no plan object is imported.
        vote = wx.arm_coverage(
            model,
            rank=0,
            planned_bytes_by_tag={k: dict(v) for k, v in planned.items()},
            tag_bytes=_tag_bytes_stub(),
            log=_CaptureLog(),
        )
        self.assertTrue(vote.ok)

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
    def test_w84_is_the_coverage_refusal_and_says_so_once(self):
        self.assertIn("W84", wms.Weg2XchgCoverageRefused.__doc__ or "")
        self.assertEqual(wx.COVERAGE_REFUSAL_MARKER, "W84 Weg2XchgCoverageRefused")

    def test_w76_is_the_runner_shape_refusal_and_says_so_once(self):
        self.assertIn("W76", wx.Weg2XchgRunnerShapeUnknown.__doc__ or "")
        self.assertEqual(
            wx.RUNNER_SHAPE_REFUSAL_MARKER, "W76 Weg2XchgRunnerShapeUnknown"
        )


if __name__ == "__main__":
    unittest.main()
