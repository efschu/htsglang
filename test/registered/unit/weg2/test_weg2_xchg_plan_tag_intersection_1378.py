# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn53: THE INTERSECTION NARROWING the 52d96f3769 guard orders.

The plan-content guard refuses any per-card divergence between the two
groups' tag sets -- and the measured weg2xsn53 boot died on exactly that
refusal (``only_src=['weights_5','weights_6','weights_7'] only_dst=[]`` on
card 0, W4 Weg2WakeRefused, epoch never left 0).  The guard's own sentence
orders the fix: ``Narrow the exchange to the INTERSECTION before deriving
bands.``  These tests pin that narrowing:

* a divergent tag set produces a PLAN (not the refusal),
* the tags outside the intersection are ABSENT from the plan's descs,
* the log line names what was dropped and how many bytes moved to the
  disk-reload fallback (#1394),
* and the narrowing is LOAD BEARING: a tag that only ONE group carries
  cannot reach the join, so the W68-family divergence cannot come back.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from test_weg2_xchg_manifest_join_1330 import (  # noqa: E402
    CARDS, D_RANKS, TAG, _manifests, _names, _piece, _split, _stage_of_layer)


def _add_source_only_tags(manifests, extra_tags):
    """Give group P's rank-0 manifest EXTRA TAGS the D side does not carry.

    The measured shape (weg2xsn53, card 0): only_src=['weights_5',
    'weights_6','weights_7'], only_dst=[] -- whole tags that exist on one
    side only, under names the other side never describes.  Each extra tag
    gets its own synthetic pieces, so the divergence is a TAG-SET
    divergence, not a retag of shared names."""
    out = []
    for man in manifests:
        if man.group != "P" or man.rank not in extra_tags:
            out.append(man)
            continue
        pieces = list(man.pieces)
        for k, tag in enumerate(sorted(extra_tags[man.rank])):
            name = f"model.layers.{man.rank}.source_only_{k}.proj.weight"
            pieces.append(xm.ManifestPiece(
                param_name=name, tensor_class="rows",
                rows_full=1024, cols_full=512, itemsize=1,
                tag=tag, nbytes=1024 * 512))
        out.append(xm.RankManifest(
            group=man.group, rank=man.rank, card=man.card,
            region_tag=man.region_tag, boot_token=man.boot_token,
            tp_rank=man.tp_rank, pp_rank=man.pp_rank,
            pieces=tuple(pieces)))
    return out


class TheIntersectionNarrowing(unittest.TestCase):
    def test_divergent_tag_sets_still_produce_a_plan(self):
        """A tag only group P carries is dropped BEFORE the join, so the
        guard's per-card comparison passes and a plan comes out.

        Red on the measured weg2xsn53 boot: the guard refused and the wake
        never ran (W4, epoch stayed 0 for the whole 200 s budget)."""
        # P rank 0 gets two extra tags its D counterpart does not carry
        manifests = _add_source_only_tags(
            _manifests(), {0: [TAG, "weights_extra_a", "weights_extra_b"]})
        lines = []
        plan, reason = xm.leg_plan_from_join(
            hook="source", group="P", rank=0, manifests=manifests,
            log=lines.append)
        self.assertIsNotNone(
            plan, f"the intersection must be narrowed, not refused: {reason}")
        self.assertNotIn("plan-tag-divergence", reason or "")
        # every desc the plan carries is inside the intersection
        dropped = {"weights_extra_a", "weights_extra_b"}
        self.assertFalse(
            [d for d in plan.descs if d.tag in dropped],
            "a tag outside the intersection must not reach the bands")

    def test_the_dropped_tags_are_named_with_their_bytes(self):
        manifests = _add_source_only_tags(
            _manifests(), {0: [TAG, "weights_extra_a"]})
        lines = []
        xm.leg_plan_from_join(hook="source", group="P", rank=0,
                              manifests=manifests, 
                              log=lines.append)
        scope = [ln for ln in lines if "exchange-scope=intersection" in ln]
        self.assertTrue(scope, f"the narrowing must be auditable: {lines}")
        self.assertIn("weights_extra_a", scope[0])
        self.assertIn("dropped_bytes=", scope[0])
        self.assertIn("disk-reload fallback", scope[0],
                      "the line must say WHERE the dropped bytes come back")

    def test_the_narrowing_is_load_bearing(self):
        """THE DYING MUTANT: the fixture DOES diverge per card, so a guard
        that still sees the divergence refuses -- which is exactly what the
        narrowing prevents.  Remove the narrowing pass and the first test
        above turns into the plan-tag-divergence refusal the weg2xsn53 boot
        measured on the metal."""
        manifests = _add_source_only_tags(
            _manifests(), {0: [TAG, "weights_extra_a"]})
        per_card = {}
        for man in manifests:
            for pc in man.pieces:
                per_card.setdefault(man.card, {}).setdefault(
                    man.group, set()).add(str(pc.tag))
        divergent = [
            card for card, groups in per_card.items()
            if len(groups) >= 2
            and [t for _g, t in sorted(groups.items())][0]
                != [t for _g, t in sorted(groups.items())][1]]
        self.assertTrue(
            divergent,
            "the fixture has no per-card divergence left, so this test can "
            "no longer kill the mutant (the narrowing would be untested)")
        for card in divergent:
            self.assertIn(card, [0], "the divergence is on card 0, the "
                          "measured shape")


if __name__ == "__main__":
    unittest.main()
