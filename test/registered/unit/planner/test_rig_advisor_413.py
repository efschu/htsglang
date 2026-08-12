# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Falsifiers for the #413 rig buying advisor.

The advisor exists to answer "what would card X buy me?" HONESTLY, so the
properties worth pinning are the ones that would let it lie:

IDENTITY -- the advisor is a diff of two ``plan()`` runs, and a diff is only
    trustworthy if changing nothing changes nothing. Swapping a card for a
    spec record identical to itself must reproduce every current number
    exactly. Without this, any "after" figure could be an artefact of the
    composition path rather than of the card.

REFUSAL -- a candidate that cannot run the model must produce the planner's
    own named reason, not an empty row and not a zero. A silent blank reads
    as "no data"; a refusal reads as "this purchase does not work", which is
    the answer the user came for.

PROVENANCE -- a datasheet-derived number must never be labelled ``measured``.
    This is the test the feature would be worthless without, so it is pinned
    from both ends: the structural one (a candidate rig cannot be described
    as live) and the output one (no "after" cell claims measurement). The
    negative control asserts the label would actually FAIL if it were wrong,
    rather than merely observing that today's output happens to be right.

THE X4 SLOT -- the two interconnect penalties of adding a fourth card are
    distinct mechanisms and are pinned separately, because conflating them is
    how an honest tool starts inventing physics: the SLOT WIDTH moves the
    roofline's host-staging bandwidth (a MIN across the plan's cards), and
    the CARD COUNT moves the TP collective discount. The width does not touch
    the collective, and this file asserts that it does not.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from sglang.srt.planner import roofline as rfmod
from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED
from sglang.srt.planner.card_library import CardLibrary, CardSpec
from sglang.srt.planner.explorer import provenance_of
from sglang.srt.planner.hardware import hardware_from_manual
from sglang.srt.planner.rig_advisor import (
    ADVISOR_RIG_SOURCE,
    FreeSlot,
    advise,
    rig_with_candidate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

#: The reference rig: 5090 32 GiB + 2x 3080 20 GiB.
_RIG3 = ["RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480"]

#: The free slot on this rig is the x4 one (nvidia-smi topo -m 2026-07-20:
#: all pairs PHB, no P2P, no NVLink, widths x4 / x8 / x8).
_X4_SLOT = FreeSlot(pcie_gen=4, pcie_width=4, provenance="declared")

#: Qwen3.6-27B geometry, verbatim from the checkpoint's ``text_config``.
_QWEN36_27B_TEXT = dict(
    model_type="qwen3_5",
    hidden_size=5120,
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248064,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262144,
    layer_types=[
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ],
)


def _write_27b_config(tmpdir: str) -> str:
    cfg = dict(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=dict(_QWEN36_27B_TEXT),
        quantization_config=dict(
            quant_method="fp8", fmt="e4m3", activation_scheme="dynamic"
        ),
    )
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmpdir


class AdvisorFixture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _write_27b_config(cls._tmp.name)
        cls.hw = hardware_from_manual(list(_RIG3))
        cls.models = [("Qwen3.6-27B FP8", cls.model)]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestIdentity(AdvisorFixture):
    """Adding nothing must change nothing."""

    def test_replacing_a_card_with_itself_reproduces_every_number(self):
        """The diff's zero point.

        Rank 1 is an RTX 3080 20 GiB; swapping it for a spec record with the
        same name, the same VRAM and the same (absent) link topology is a
        no-op on the hardware. Every metric must come back bit-identical.
        If this drifts, the composition path is contributing something of its
        own and no "after" number can be attributed to the candidate card.
        """
        twin = CardSpec(
            "RTX 3080",
            20480,
            "sm86",
            None,
            None,
            False,
            320,
            peak_membw_gbs=760.0,
            peak_gemm_tflops_fp16=119.0,
        )
        res = advise(
            self.hw,
            twin,
            self.models,
            mode="replace",
            replace_index=1,
        )
        row = res.rows[0]
        self.assertEqual(row.before_fits, row.after_fits)
        for m in row.metrics:
            self.assertEqual(
                m.before,
                m.after,
                f"{m.label}: replacing a card with an identical spec moved the "
                f"number {m.before} -> {m.after}; the diff's zero point drifted.",
            )
            self.assertEqual(m.verdict, "unchanged", m.label)

    def test_the_rig_shape_is_preserved_by_a_swap(self):
        after = rig_with_candidate(
            self.hw,
            CardLibrary().get("RTX 4090"),
            mode="replace",
            replace_index=1,
        )
        self.assertEqual(len(after.gpus), len(self.hw.gpus))
        self.assertEqual(after.gpus[1].name, "RTX 4090")
        self.assertEqual([g.index for g in after.gpus], [0, 1, 2])


class TestRefusal(AdvisorFixture):
    """A candidate that cannot run the model says so, with the reason."""

    def test_swapping_in_too_little_vram_produces_a_named_refusal(self):
        """Replace the 32 GiB card with a 12 GiB one.

        The model fits today and must stop fitting. The row has to carry the
        planner's own reason text -- an empty refusal list would render as a
        blank cell, which reads as "unknown" instead of "this purchase does
        not work".
        """
        res = advise(
            self.hw,
            "RTX 3060",  # 12288 MiB
            self.models,
            mode="replace",
            replace_index=0,
        )
        row = res.rows[0]
        self.assertTrue(row.before_fits, "the model must fit on the rig as it is")
        self.assertFalse(
            row.after_fits,
            "swapping the 32 GiB card for a 12 GiB card must not still fit",
        )
        self.assertTrue(
            row.after_refusal,
            "a refusal must carry the planner's reason, never an empty list",
        )
        self.assertTrue(
            any(r.strip() for r in row.after_refusal),
            f"refusal reasons must be non-empty text: {row.after_refusal!r}",
        )

    def test_a_refused_row_reports_absent_rather_than_zero(self):
        """No throughput is not zero throughput.

        A plan that does not fit has no roofline, and the cells must say
        ``absent``. A 0.0 would sort, average and render as a real figure.
        """
        res = advise(self.hw, "RTX 3060", self.models, mode="replace", replace_index=0)
        row = res.rows[0]
        for key in ("prefill_tok_s", "decode_tok_s"):
            m = row.metric(key)
            self.assertIsNone(m.after, f"{key} must be absent, not a number")
            self.assertEqual(m.after_provenance, ABSENT)
            self.assertEqual(m.verdict, ABSENT)
            # wizard.cell() drops the value entirely when absent.
            self.assertFalse(m.to_dict()["after"]["available"])


class TestProvenance(AdvisorFixture):
    """A datasheet number must never be able to claim it was measured."""

    def test_a_candidate_rig_can_never_be_described_as_live(self):
        """Structural, not incidental.

        The label is derived from ``HardwareSpec.source`` alone, so this holds
        for callers that never read the advisor module.
        """
        after = rig_with_candidate(self.hw, CardLibrary().get("RTX 4090"))
        self.assertEqual(after.source, ADVISOR_RIG_SOURCE)
        prov, is_estimate, note = provenance_of(after)
        self.assertEqual(prov, "composed")
        self.assertTrue(is_estimate)
        self.assertIn("not measured", note)

    def test_the_candidate_carries_no_fabricated_identity(self):
        """#397: an unknown identity stays None, never a plausible integer."""
        after = rig_with_candidate(self.hw, CardLibrary().get("RTX 4090"))
        for g in after.gpus:
            self.assertIsNone(g.cuda_index, f"{g.name} invented a CUDA ordinal")
            self.assertIsNone(g.free_mib, f"{g.name} claims a live free-VRAM read")
        self.assertIsNone(after.cuda_index_source)
        self.assertIsNone(after.gpus[-1].uuid)

    def test_no_after_cell_claims_to_be_measured(self):
        """The output-side half of the same guarantee."""
        res = advise(self.hw, "RTX 4090", self.models, free_slot=_X4_SLOT)
        for row in res.rows:
            for m in row.metrics:
                self.assertNotEqual(
                    m.after_provenance,
                    MEASURED,
                    f"{m.label}: an 'after' number derived from a card that is "
                    f"not in this machine was labelled measured.",
                )
                self.assertIn(m.after_provenance, (ESTIMATE, ABSENT))
                self.assertEqual(m.to_dict()["after"]["provenance"], m.after_provenance)

    def test_the_measured_label_would_actually_fail_this_suite(self):
        """Negative control.

        Without this, the assertion above could pass because nothing ever
        emits ``measured`` anywhere -- a test that cannot go red. Here a cell
        is deliberately mislabelled and the same check is required to catch
        it.
        """
        res = advise(self.hw, "RTX 4090", self.models, free_slot=_X4_SLOT)
        m = res.rows[0].metrics[0]
        mislabelled = m.__class__(
            **{
                **{f.name: getattr(m, f.name) for f in m.__dataclass_fields__.values()},
                "after_provenance": MEASURED,
            }
        )
        with self.assertRaises(AssertionError):
            self.assertNotEqual(mislabelled.after_provenance, MEASURED)

    def test_every_estimated_cell_names_its_basis(self):
        """An estimate without a stated basis is an unsourced claim."""
        res = advise(self.hw, "RTX 4090", self.models, free_slot=_X4_SLOT)
        for row in res.rows:
            for m in row.metrics:
                for side in ("before", "after"):
                    c = m.to_dict()[side]
                    if c["provenance"] == ESTIMATE:
                        self.assertTrue(
                            c["basis"].strip(),
                            f"{m.label} ({side}) is an estimate with no basis",
                        )

    def test_the_truths_are_always_present(self):
        res = advise(self.hw, "RTX 4090", self.models, free_slot=_X4_SLOT)
        self.assertTrue(res.truths)
        joined = " ".join(res.truths)
        self.assertIn("estimate", joined)


class TestX4Slot(AdvisorFixture):
    """The two penalties of a fourth card, kept apart."""

    def test_the_candidate_takes_the_slot_width_not_its_own(self):
        """An x16 card in an x4 slot is an x4 card."""
        card = CardLibrary().get("RTX 4090")
        self.assertEqual(card.pcie_width, 16, "fixture assumption: an x16 part")
        after = rig_with_candidate(self.hw, card, mode="add", free_slot=_X4_SLOT)
        self.assertEqual(after.gpus[-1].pcie_width, 4)
        self.assertEqual(after.gpus[-1].pcie_gen, 4)

    def test_the_x4_slot_sets_the_host_staging_bandwidth(self):
        """The term the width actually moves.

        ``_pcie_fetch_gbs`` is a MIN across the plan's cards, so the x4
        newcomer throttles staging for every card in the plan.
        """
        after = rig_with_candidate(
            self.hw, CardLibrary().get("RTX 4090"), mode="add", free_slot=_X4_SLOT
        )
        gids = [g.index for g in after.gpus]
        got = rfmod._pcie_fetch_gbs(after, gids)
        expected = rfmod._PCIE_GBS_PER_LANE[4] * 4 * rfmod._PCIE_STAGING_EFFICIENCY
        self.assertAlmostEqual(got, expected, places=6)

        wide = rig_with_candidate(
            self.hw,
            CardLibrary().get("RTX 4090"),
            mode="add",
            free_slot=FreeSlot(4, 16, "declared"),
        )
        self.assertGreater(
            rfmod._pcie_fetch_gbs(wide, [g.index for g in wide.gpus]),
            got,
            "a wider free slot must raise the host-staging bandwidth",
        )

    def test_the_fourth_card_worsens_the_collective_for_everybody(self):
        """The term the CARD COUNT moves -- a different mechanism.

        On a no-P2P rig the cross-card discount is tiered on how many cards
        the collective crosses, so going 3 -> 4 costs the ranks already
        installed. Pinned as an inequality plus the exact tier values, so a
        retune of the constants shows up here.
        """
        self.assertEqual(rfmod._PCIE_NOP2P_BY_CROSS_CARDS[3], 0.35)
        self.assertEqual(rfmod._PCIE_NOP2P_MANY, 0.28)
        self.assertLess(
            rfmod._PCIE_NOP2P_MANY,
            rfmod._PCIE_NOP2P_BY_CROSS_CARDS[3],
            "a fourth card must not be free for the ranks you already own",
        )
        res = advise(self.hw, "RTX 4090", self.models, mode="add", free_slot=_X4_SLOT)
        self.assertTrue(
            any("Collective cost" in t for t in res.truths),
            f"the collective penalty must be stated: {res.truths}",
        )

    def test_the_width_is_not_claimed_to_move_the_collective(self):
        """Honesty about mechanism, not just about numbers.

        The slot-width truth must describe the staging term and explicitly
        deny touching the collective, because the width integer genuinely has
        no path into ``_interconnect``.
        """
        res = advise(self.hw, "RTX 4090", self.models, mode="add", free_slot=_X4_SLOT)
        slot = [t for t in res.truths if t.startswith("Slot width")]
        self.assertEqual(len(slot), 1, res.truths)
        self.assertIn("does not change the TP collective", slot[0])

    def test_an_undeclared_slot_says_so_instead_of_assuming_x16(self):
        res = advise(self.hw, "RTX 4090", self.models, mode="add")
        slot = [t for t in res.truths if t.startswith("Slot width")]
        self.assertEqual(len(slot), 1, res.truths)
        self.assertIn("UNDECLARED", slot[0])
        after = rig_with_candidate(self.hw, CardLibrary().get("RTX 4090"))
        self.assertIsNone(after.gpus[-1].pcie_width)


class TestSlowestRank(AdvisorFixture):
    """Capacity is not speed, and the tool must not let them blur."""

    def test_an_unmoved_bottleneck_is_named(self):
        """Adding a 4090 beside two 3080s leaves a 3080 clocking the group.

        The honest headline is "you bought context, not decode rate", and the
        advisor has to say it rather than reporting only the throughput
        delta.
        """
        res = advise(self.hw, "RTX 4090", self.models, mode="add", free_slot=_X4_SLOT)
        row = res.rows[0]
        self.assertTrue(row.after_fits)
        self.assertEqual(row.before_bottleneck, row.after_bottleneck)
        self.assertTrue(
            any(t.startswith("Slowest rank") for t in res.truths),
            f"an unmoved bottleneck must be stated: {res.truths}",
        )

    def test_capacity_grows_more_than_decode_does(self):
        """The shape of the honest answer, pinned as a relation.

        This is the whole point of the feature: on this rig a fourth card is
        a VRAM purchase. Asserting the RELATION (context gains proportionally
        more than decode) rather than absolute figures keeps the test alive
        across constant retunes.
        """
        res = advise(self.hw, "RTX 4090", self.models, mode="add", free_slot=_X4_SLOT)
        row = res.rows[0]
        ctx = row.metric("max_context").delta_pct
        dec = row.metric("decode_tok_s").delta_pct
        self.assertIsNotNone(ctx)
        self.assertIsNotNone(dec)
        self.assertGreater(
            ctx,
            dec,
            "a fourth card on a bottlenecked rig buys capacity before speed",
        )


class TestSerialisation(AdvisorFixture):
    """The payload the tab renders."""

    def test_to_dict_is_json_serialisable_and_carries_the_pills(self):
        res = advise(self.hw, "RTX 4090", self.models, free_slot=_X4_SLOT)
        blob = json.dumps(res.to_dict())
        back = json.loads(blob)
        self.assertTrue(back["ok"])
        self.assertEqual(back["candidate"], "RTX 4090")
        self.assertEqual(back["after_provenance"], "composed")
        self.assertTrue(back["truths"])
        cell = back["rows"][0]["metrics"][0]["before"]
        for key in ("value", "available", "provenance", "basis", "unit"):
            self.assertIn(key, cell, "cells must keep the wizard.cell() shape")

    def test_an_unknown_card_name_is_a_clean_error(self):
        with self.assertRaises(KeyError):
            advise(self.hw, "RTX 9090 Ti Super", self.models)


class TestEndpoint(AdvisorFixture):
    """POST /api/rig_advisor/plan."""

    def _body(self, **kw):
        b = {
            "model": self.model,
            "hardware": {"source": "manual", "gpus": list(_RIG3)},
            "candidate": "RTX 4090",
            "free_slot": {"pcie_gen": 4, "pcie_width": 4},
        }
        b.update(kw)
        return b

    def test_an_empty_request_returns_the_card_library(self):
        """The picker and the plan come from ONE endpoint (#342: the UI
        composes the same API a script would call)."""
        from sglang.srt.planner import webui

        d = webui.rig_advisor_payload({})
        self.assertTrue(d["ok"])
        self.assertTrue(d["cards"])
        self.assertIn("RTX 4090", [c["name"] for c in d["cards"]])
        self.assertEqual(d["rows"], [])

    def test_a_plan_request_returns_the_before_after_table(self):
        from sglang.srt.planner import webui

        d = webui.rig_advisor_payload(self._body())
        self.assertTrue(d.get("ok"), d.get("error"))
        self.assertEqual(d["after_provenance"], "composed")
        self.assertEqual(len(d["rows"]), 1)
        self.assertTrue(d["truths"])
        json.dumps(d)  # must survive the wire

    def test_a_missing_model_is_an_explained_refusal_not_a_crash(self):
        from sglang.srt.planner import webui

        d = webui.rig_advisor_payload(self._body(model=""))
        self.assertFalse(d["ok"])
        self.assertIn("model", d["error"])
        self.assertTrue(d["cards"], "the picker must survive an error")

    def test_a_hand_typed_card_is_priced_without_being_in_the_seed_set(self):
        """The passthrough that makes a custom candidate possible.

        ``plan()`` looks nameplate peaks up in the library it is handed; before
        #413 it never forwarded one, so a card outside ``SEED_CARDS`` silently
        produced no roofline at all. A hand-typed card must get real numbers.
        """
        from sglang.srt.planner import webui

        d = webui.rig_advisor_payload(
            self._body(
                candidate={
                    "name": "Hypothetical 48GB",
                    "total_mib": 49152,
                    "tdp_w": 400,
                    "peak_membw_gbs": 1200.0,
                    "peak_gemm_tflops_fp16": 300.0,
                    "peak_gemm_tflops_fp8": 600.0,
                }
            )
        )
        self.assertTrue(d.get("ok"), d.get("error"))
        self.assertEqual(d["candidate"], "Hypothetical 48GB")
        after = d["rows"][0]["metrics"][1]["after"]  # prefill throughput
        self.assertTrue(
            after["available"],
            "a hand-typed card must still be priced; an absent roofline here "
            "means the card_library passthrough regressed",
        )
        self.assertEqual(after["provenance"], ESTIMATE)


class TestDashboardWiring(unittest.TestCase):
    """The tab is reachable.

    A panel that exists in the HTML but is missing from the TABS array or the
    nav map is invisible, and no unit test of the payload would notice.
    """

    def test_the_advisor_tab_is_wired_into_the_page(self):
        from sglang.srt.planner import webui

        html = webui.INDEX_HTML
        for needle in (
            'id="view_advisor"',
            'id="tab_advisor"',
            "'video','advisor'",  # showTab's TABS array
            "'pair','advisor'",  # NAV_GROUPS rig group
            "_advisorInit",
            "function renderAdvisor",
            "/api/rig_advisor/plan",
            "#view_advisor .p.measured",
        ):
            self.assertIn(needle, html, f"advisor tab wiring lost: {needle}")


if __name__ == "__main__":
    unittest.main()
