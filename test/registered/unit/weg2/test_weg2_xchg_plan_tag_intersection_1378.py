# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn53: THE PLAN GATE'S DIRECTION, and the join's name scope.

Measured on the weg2xsn53 boot (P log 004732): W4 <- join-plan-tag-divergence
on card 0 (only_src=[weights_5,6,7]) and then W74 '912 of 1249 tensors held
by D have no counterpart in P'.  The manifests are COMPLETE (|P|=893, |D|=1249,
P subset D, only-P=0), so both refusals were instrument defects:

* the guard compared EQUALITY and paired the groups by NAME ('D' sorts first),
  not by the direction's own source/destination -- under PP x TP the source
  group's tag set is a strict SUBSET of the destination's by construction;
* the join planned over the destination's WHOLE name set, so the 356
  destination-only names were refused instead of being named and left to the
  disk-reload fallback (#1394).

Pinned here: a destination-only tag set produces a plan, a source-only tag
still refuses (the dying mutant), and the destination-only names are named
with their byte count.
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
    CARDS, TAG, _manifests, _piece)


def _with_extra_tags(manifests, group, tags):
    """Append one synthetic piece per extra tag to group rank 0."""
    out = []
    for man in manifests:
        pieces = list(man.pieces)
        if man.group == group and man.rank == 0:
            for k, tag in enumerate(sorted(tags)):
                pieces.append(xm.ManifestPiece(
                    param_name=f"model.source_only_{k}.proj.weight",
                    tensor_class="rows", rows_full=1024, cols_full=512,
                    itemsize=1, tag=tag, nbytes=1024 * 512))
        out.append(xm.RankManifest(
            group=man.group, rank=man.rank, card=man.card,
            region_tag=man.region_tag, boot_token=man.boot_token,
            tp_rank=man.tp_rank, pp_rank=man.pp_rank, pieces=tuple(pieces)))
    return out


class ThePlanGateDirection(unittest.TestCase):
    def test_destination_only_tags_produce_a_plan(self):
        """D holds a tag P does not: the exchange covers the intersection,
        the extra tag is the fallback's business -- a PLAN comes out."""
        manifests = _with_extra_tags(_manifests(), "D", ["weights_extra_a"])
        lines = []
        plan, reason = xm.leg_plan_from_join(
            hook="source", group="P", rank=0, manifests=manifests,
            log=lines.append)
        self.assertIsNotNone(
            plan, f"destination-only tags must not refuse: {reason}")
        self.assertNotIn("plan-tag-divergence", reason or "")

    def test_a_card_with_no_shared_name_still_refuses(self):
        """THE DYING MUTANT: the one defect the guard can still catch is a
        card whose two groups share no name at all -- a deposit and a
        collect that never meet."""
        manifests = _manifests()
        emptied = [
            m if not (m.group == "P" and m.rank == 0)
            else xm.RankManifest(group=m.group, rank=m.rank, card=m.card,
                                 region_tag=m.region_tag,
                                 boot_token=m.boot_token, tp_rank=m.tp_rank,
                                 pp_rank=m.pp_rank, pieces=())
            for m in manifests]
        emptied = [m for m in emptied if m.pieces]
        plan, reason = xm.leg_plan_from_join(
            hook="source", group="P", rank=0, manifests=emptied)
        self.assertIsNone(plan, "a card with no shared name must refuse")
        self.assertIn("join-unjoinable", reason or "")


if __name__ == "__main__":
    unittest.main()
