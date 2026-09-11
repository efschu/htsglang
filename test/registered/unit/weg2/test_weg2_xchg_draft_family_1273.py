# SPDX-License-Identifier: Apache-2.0
"""#1273 B4k -- the DRAFT HEAD joins the weights family (spec AMENDMENT 6).

USER RULING 2026-09-11, verbatim: *"warum sollte der mtp kopf nur auf d liegen?
auch in p werden draft token erstellt, damit in d auch gedraftet werden kann --
die draft layer bytes, also die draft gewichte, muessen genauso aus dem vram
geflippt werden"* and *"ist prinzipiell aber nichts anderes als vom normalen
modell die gewichte zu 'verschieben'. das muss also nix neues werden, nur
'reused' werden."*  So spec section 4.1 is OVERRULED and this file pins the
overruling.

**SECTION 4.1's PREMISE IS STALE ON THE SHIPPED DEFAULT, and that is a fact
about this tree rather than a judgement.**  4.1 reads: *"Group P carries no
``--speculative-*`` in this form, so NONE of those bytes has a VRAM source on
the other side."*  On the authoritative tip ``DRAFT_KV_ON_P_DEFAULT = "on"``
(launcher.py:552) and ``P_DRAFT_KV_FLAGS`` (launcher.py:541) give group P the
four speculative flags, and ``model_runner.py:1548-1551`` says what that group
then does: *"its last pipeline stage runs the checkpoint's MTP head after every
target chunk so the draft KV rows exist"*.  Both groups hold the head.

MEASURED ON METAL, boot weg2xsn15 (``WEG2-XCHG-RESIDENT``, both logs)::

    D  TP0 tag=weights_draft mib=1440.0 in_family=no      (the 5090)
    D  TP1 tag=weights_draft mib=1280.0 in_family=no
    D  TP2 tag=weights_draft mib=1280.0 in_family=no
    P  PP2 tag=weights_draft mib=1572.0 in_family=no      (the LAST stage only)

That is 1440/1280/1280 MiB of group D residency the exchange never moves and no
flip ever pauses -- and it is the WHOLE of the excess B4g mis-reported as an
unattributed residual (see ``test_weg2_xchg_reserve_1273.py``).

**THE SHAPE, and why nothing new is built.**  P holds the head WHOLE on one
card; D holds it on three.  So it is a NON-IDENTICAL piece set and it travels
path (b) -- assembled in the depth-2 host bounce like any other non-identical
layer.  The embed/head shards the draft runner re-materialises are base-model
tensors already in the plan.  The machinery is reused; only the FAMILY
MEMBERSHIP of the tag changes.

**ONE GATE, AND THE RING ARM IS BYTE-IDENTICAL.**  Under ``ring`` the draft
runner never gets its own tag at all (``weights_region_tag_for`` returns the
base tag unless ``exchange_armed()``), and the pause order, the wave list and
the family predicate are exactly what they were: the drafter is paused as part
of the base tag, as today.  The membership is therefore gated on the same arm
the tag site already reads, in ONE place (``draft_tag_in_family``), so no second
reader of the mode exists.

**WHAT DOES NOT CHANGE, deliberately.**  The tag SITE
(``weights_region_tag_for`` / model_runner.py:2440-2470) is untouched: the draft
keeps its own tag, because the draft head is not a layer band of the target and
must not be charged to one.  Giving it the BASE tag instead was considered and
REFUTED: under the base region ``weight_chunk_scope`` is live, so the drafter's
``layers.0.*`` names would be allocated to ``weights_0`` -- and group P's PP2,
which holds target layers 52-63, would then carry ``weights_0`` bytes.  That
breaks ``chunk_tag_cards`` and is the #1233 boot weg2dk4 geometry-mismatch class.
``weights_vision`` (S8) stays OUT of the family under both arms: only the tag
whose bytes have a source on both sides joins, and that is a measurement, not a
naming rule.

RED ON ``2c9592fd19``: ``is_weights_family_tag("weights_draft")`` is False under
every arm, ``weights_family_tags`` never contains it, ``arm_coverage_at_load``
short-circuits on a literal tag compare so no COVER row is built for it, and
``interleave_pause_order`` would mis-sort it as a chunk.
"""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sglang.srt.constants import (  # noqa: E402
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.managers import weg2_memory_saver as wms  # noqa: E402
from sglang.srt.weg2 import front as fr  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

MIB = 1024 * 1024


class _Exchange:
    """Arm the exchange the way the launcher arms it: the env var the ONE
    reader (``weight_exchange.weight_source``) reads."""

    def __init__(self, value=wx.WEIGHT_SOURCE_EXCHANGE):
        self.value = value
        self.saved = None

    def __enter__(self):
        self.saved = os.environ.get(wx.WEIGHT_SOURCE_ENV)
        os.environ[wx.WEIGHT_SOURCE_ENV] = self.value
        return self

    def __exit__(self, *_a):
        if self.saved is None:
            os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[wx.WEIGHT_SOURCE_ENV] = self.saved
        return False


class _DraftModel(nn.Module):
    """The drafter's one-layer block, as ``qwen3_5_mtp.py`` builds it: names
    that still say ``layers.0.`` while the ALLOCATION is the draft region's."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.qkv_proj = nn.Parameter(
            torch.zeros(256, 512, dtype=torch.bfloat16), requires_grad=False
        )
        self.mtp = nn.Module()
        self.mtp.eh_proj = nn.Parameter(
            torch.zeros(128, 512, dtype=torch.bfloat16), requires_grad=False
        )


def _planned(model, region_tag):
    out = {}
    for name, p in model.named_parameters():
        tag = wx.tag_of_parameter_name(name, region_tag=region_tag)
        out.setdefault(tag, {})[name] = int(p.untyped_storage().nbytes())
    return out


# ===========================================================================
# THE FAMILY PREDICATE, form-gated in ONE place.
# ===========================================================================


class TheDraftTagIsInTheFamilyUnderExchange(unittest.TestCase):
    def test_the_predicate_matches_the_draft_tag_under_exchange(self):
        with _Exchange():
            self.assertTrue(wms.is_weights_family_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT))

    def test_the_ring_arm_is_byte_identical(self):
        """Spec 1.1: ``ring`` stays today, byte for byte.  Under ring the draft
        runner is paused as part of the BASE tag (the tag site never hands it
        its own tag), so admitting the tag to the family there would change a
        path nobody asked to change."""
        with _Exchange(wx.WEIGHT_SOURCE_RING):
            self.assertFalse(wms.is_weights_family_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT))
            self.assertEqual(len(wms.weights_family_tags(8)), 9)
            self.assertNotIn(
                GPU_MEMORY_TYPE_WEIGHTS_DRAFT, wms.weights_family_tags(8)
            )

    def test_it_is_never_a_CHUNK_tag(self):
        """A chunk tag is a LAYER BAND of the target and is named
        ``weights_<integer>``.  The draft head is not a band of anything, and
        every consumer that sorts or maps by band must keep seeing that."""
        for arm in (wx.WEIGHT_SOURCE_EXCHANGE, wx.WEIGHT_SOURCE_RING):
            with _Exchange(arm):
                self.assertFalse(
                    wms.is_weights_chunk_tag(GPU_MEMORY_TYPE_WEIGHTS_DRAFT), arm
                )

    def test_weights_vision_stays_OUT_under_both_arms(self):
        """The class, not the instance.  The draft joins because BOTH groups
        were measured holding its bytes; S8's vision tag has had no such
        measurement, and a naming rule is not a source."""
        for arm in (wx.WEIGHT_SOURCE_EXCHANGE, wx.WEIGHT_SOURCE_RING):
            with _Exchange(arm):
                for tag in ("weights_vision", "weights_", "weights_0b"):
                    self.assertFalse(wms.is_weights_family_tag(tag), f"{arm}:{tag}")

    def test_the_family_list_carries_it_BEFORE_the_base_tag(self):
        """``derive_waves`` takes ``tags[-1]`` as the base and
        ``interleave_pause_order``'s contract is that the base tag CLOSES the
        sleep.  A draft tag appended last would silently become the base."""
        with _Exchange():
            tags = wms.weights_family_tags(8)
            self.assertEqual(len(tags), 10)
            self.assertEqual(tags[-1], GPU_MEMORY_TYPE_WEIGHTS)
            self.assertEqual(tags[-2], GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
            self.assertEqual(tags[:8], [f"weights_{k}" for k in range(8)])

    def test_the_gate_has_ONE_reader_of_the_mode(self):
        """``draft_tag_in_family`` reads through
        ``weight_exchange.exchange_armed`` and not through a second copy of the
        env parse -- substituted, so the assertion is on the CALL."""
        import unittest.mock as m

        with m.patch.object(wx, "exchange_armed", return_value=True) as armed:
            self.assertTrue(wms.draft_tag_in_family())
        self.assertTrue(armed.called)
        with m.patch.object(wx, "exchange_armed", return_value=False):
            self.assertFalse(wms.draft_tag_in_family())


# ===========================================================================
# THE WAVE LIST AND THE PAUSE ORDER.
# ===========================================================================


class TheDraftTagTravelsTheExistingSchedule(unittest.TestCase):
    def test_derive_waves_keeps_the_base_tag_as_the_closer(self):
        with _Exchange():
            tags = wms.weights_family_tags(4)
            waves = wx.derive_waves(tags, {}, [0, 1, 2])
            flat = [t for w in waves for t in w]
            self.assertEqual(sorted(flat), sorted(tags))
            self.assertEqual(waves[-1][-1], GPU_MEMORY_TYPE_WEIGHTS)
            self.assertIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, flat)

    def test_the_pause_order_does_not_mis_sort_it_as_a_chunk(self):
        """THE REGRESSION THIS SLICE MUST NOT CAUSE.
        ``interleave_pause_order`` split its input on the RAW
        ``startswith("weights_")``, which is True for ``weights_draft``.  With
        the draft tag in the list it would be treated as a chunk, the
        chunk->card map has no entry for it, and the function would fall back to
        the IDENTITY refusal -- losing the tightest-card-first order that boot
        weg2dk4 paid for.  The split is the integer predicate now.
        """
        with _Exchange():
            tags = wms.weights_family_tags(3)
            tag_cards = {"weights_0": (0,), "weights_1": (1,), "weights_2": (2,)}
            free = {0: 5000, 1: 1000, 2: 3000}
            order, why = fr.interleave_pause_order(tags, tag_cards, free)
            self.assertEqual(why, "tightest-card-first", why)
            self.assertEqual(sorted(order), sorted(tags))
            # tightest card first among the CHUNKS, then the non-chunk tail
            self.assertEqual(order[:3], ["weights_1", "weights_2", "weights_0"])
            self.assertEqual(order[-1], GPU_MEMORY_TYPE_WEIGHTS)
            self.assertEqual(order[-2], GPU_MEMORY_TYPE_WEIGHTS_DRAFT)


# ===========================================================================
# THE COVERAGE ARM: the draft runner is censused instead of waved off.
# ===========================================================================


class TheDraftRunnerIsCovered(unittest.TestCase):
    def test_build_coverage_returns_a_row_for_the_draft_tag(self):
        with _Exchange():
            model = _DraftModel()
            planned = _planned(model, GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
            rows = wx.build_coverage(
                model,
                rank=0,
                planned_bytes_by_tag=planned,
                tag_bytes=lambda _t: 0,
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )
            self.assertIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, rows)
            row = rows[GPU_MEMORY_TYPE_WEIGHTS_DRAFT]
            self.assertEqual([t.name for t in row.uncovered], [])
            self.assertTrue(row.ok)
            self.assertEqual(
                row.planned_bytes,
                sum(
                    int(p.untyped_storage().nbytes())
                    for _n, p in model.named_parameters()
                ),
            )

    def test_an_uncovered_draft_parameter_still_refuses(self):
        """The membership must buy the REFUSAL too, not just the row: a draft
        tensor the plan does not carry is a page the destination never
        receives, exactly as for any other family tag."""
        with _Exchange():
            model = _DraftModel()
            planned = _planned(model, GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
            planned[GPU_MEMORY_TYPE_WEIGHTS_DRAFT].pop("mtp.eh_proj")
            rows = wx.build_coverage(
                model,
                rank=0,
                planned_bytes_by_tag=planned,
                tag_bytes=lambda _t: 0,
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )
            row = rows[GPU_MEMORY_TYPE_WEIGHTS_DRAFT]
            self.assertEqual([t.name for t in row.uncovered], ["mtp.eh_proj"])
            self.assertFalse(row.ok)

    def test_the_arm_builds_coverage_for_the_draft_runner(self):
        """``arm_coverage_at_load`` short-circuited on a LITERAL tag compare
        (``region_tag != GPU_MEMORY_TYPE_WEIGHTS``), which is a second answer to
        "is this exchanged" beside the family predicate.  One authority: the
        predicate.  Acceptance line of AMENDMENT 6 = a COVER line for this tag.
        """
        with _Exchange():
            model = _DraftModel()
            planned = _planned(model, GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
            wx.register_plan_provider(lambda _m: planned)
            try:
                lines = []
                vote = wx.arm_coverage_at_load(
                    model,
                    rank=0,
                    tag_bytes=lambda _t: 1280 * MIB,
                    region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                    log=lines.append,
                )
            finally:
                wx.register_plan_provider(None)
            resident = [ln for ln in lines if wx.RESIDENT_LINE_PREFIX in ln]
            cover = [ln for ln in lines if wx.COVER_LINE_PREFIX in ln]
            self.assertEqual(len(resident), 1, lines)
            self.assertIn("in_family=yes", resident[0])
            self.assertIn(f"tag={GPU_MEMORY_TYPE_WEIGHTS_DRAFT}", resident[0])
            self.assertEqual(len(cover), 1, lines)
            self.assertIn(f"tag={GPU_MEMORY_TYPE_WEIGHTS_DRAFT}", cover[0])
            self.assertIn("uncovered=0", cover[0])
            self.assertIsNotNone(vote)
            self.assertTrue(vote.ok)
            self.assertIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, vote.rows)

    def test_the_arm_is_still_a_no_op_under_ring(self):
        with _Exchange(wx.WEIGHT_SOURCE_RING):
            lines = []
            vote = wx.arm_coverage_at_load(
                _DraftModel(),
                rank=0,
                tag_bytes=lambda _t: 0,
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                log=lines.append,
            )
            self.assertIsNone(vote)
            self.assertEqual(lines, [])

    def test_the_tag_site_is_UNCHANGED_and_still_hands_out_the_draft_tag(self):
        """The draft keeps its OWN tag: it is not a layer band of the target
        and charging it to one is the #1233 weg2dk4 class.  One tag site, and
        this slice does not move it."""
        with _Exchange():
            shape = wx.RunnerShape(
                is_draft_worker=True,
                is_phase_flip_tp_stack=False,
                is_dual_group_lane=False,
                speculative_configured=True,
            )
            self.assertEqual(
                wx.weights_region_tag_for(shape), GPU_MEMORY_TYPE_WEIGHTS_DRAFT
            )


# ===========================================================================
# THE ROTATION: the draft runner can derive its own plan.
# ===========================================================================


class TheDraftRunnerCanPlanItsOwnWeights(unittest.TestCase):
    """``derive_leg_plan`` derived the rotation's ``classes`` from CHUNK tags
    only, so the draft runner -- whose inventory carries ``weights_draft`` and
    nothing else -- would have refused its own plan by name
    (``no-chunk-classes``) the moment the membership landed.  A rank that cannot
    plan its own weights while every other consumer plans them is the
    disagreement W29 exists to catch."""

    def setUp(self):
        os.environ["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = "8"
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "8"

    def tearDown(self):
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNK_LAYERS", None)
        os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS", None)

    def test_the_draft_runners_plan_is_derived_and_covers_every_byte(self):
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        with _Exchange():
            model = _DraftModel()
            plan, reason = sh.derive_leg_plan(
                hook=sh.HOOK_SOURCE, group="P", peer="D", rank=0, model=model,
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )
            self.assertIsNotNone(plan, f"derivation refused: {reason}")
            got = wx.plan_bytes_from_descs(plan.descs)
            self.assertIn(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, got)
            live = {
                n: int(p.untyped_storage().nbytes())
                for n, p in model.named_parameters()
            }
            self.assertEqual(
                sorted(got[GPU_MEMORY_TYPE_WEIGHTS_DRAFT]), sorted(live)
            )
            for name, nbytes in live.items():
                self.assertEqual(got[GPU_MEMORY_TYPE_WEIGHTS_DRAFT][name], nbytes)
            self.assertEqual(int(plan.undescribed), 0)
            rows = wx.build_coverage(
                model, rank=0, planned_bytes_by_tag=got,
                tag_bytes=lambda _t: 0,
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )
            self.assertEqual(
                [t.name for t in rows[GPU_MEMORY_TYPE_WEIGHTS_DRAFT].uncovered], []
            )

    def test_the_BASE_tag_is_still_OUT_of_the_rotation(self):
        """THE OPERATOR'S CONDITION, as a can-fail rather than a comment.

        The base tag's bytes (embeddings on the first stage, head on the last,
        buffers everywhere) are not a layer band, so a rotation over them has
        CONTENT that depends on which card asks and the six ranks cannot agree
        on ``classes_hash``.  A predicate widened from "family and not base" to
        plain "family" passes every other test in this file and fails here.
        """
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        src = inspect.getsource(sh.derive_leg_plan)
        head = src[src.index("classes = tuple(sorted("):]
        head = head[: head.index("if not classes")]
        self.assertIn("is_weights_family_tag", head)
        self.assertIn("GPU_MEMORY_TYPE_WEIGHTS", head)
        # The behaviour, not only the text: a BASE-tagged inventory must still
        # produce no rotation classes of its own.
        with _Exchange():
            model = _DraftModel()
            plan, reason = sh.derive_leg_plan(
                hook=sh.HOOK_SOURCE, group="P", peer="D", rank=0, model=model,
                region_tag=GPU_MEMORY_TYPE_WEIGHTS,
                tag_of=lambda _n, **_k: GPU_MEMORY_TYPE_WEIGHTS,
            )
            self.assertIsNone(plan, "the base tag alone must not be a rotation")
            self.assertIn("no-chunk-classes", reason)


if __name__ == "__main__":
    unittest.main()
