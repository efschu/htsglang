# SPDX-License-Identifier: Apache-2.0
"""B4c item 1: the ring table RETAINS its per-rank per-tag peaks (#1273).

``parse_group_log`` builds ``tag_peak[tag][rank]`` and immediately sums it away
into ``tag_totals`` (``ring_table.py`` PEAK pass).  ``tag_totals`` is the
corridor's MAX STEP TOTAL -- one number per tag over the whole group -- and it
is the only thing that survived, so the per-CARD per-tag bytes the S6 exchange
census needs (``xchg_residency.CardCensus.tags``: ``group -> tag -> MiB``) had
no producer at all and ``--weg2-weight-source exchange`` refused at launch
(W71 Weg2XchgResidencyUnarmable, no census).

The retention is ADDITIVE and this file is its pin:

* ``tag_totals`` and its consumers are BYTE-IDENTICAL -- asserted as an
  identity against the per-rank sums, not as a golden number.
* ``tag_by_rank`` additionally carries the FAMILY BASE tag, which ``tag_peak``
  deliberately drops (``_family_roots``).  That drop is right for a corridor
  STEP SIZE and wrong for a census, and the reason is MEASURED rather than
  argued: on boot weg2sn5b the sum of every single-tag peak equals that rank's
  image EXACTLY on all six ranks, so the base tag carries its own remainder
  (embeddings, head, draft, buffers) and is not a second spelling of the
  family.  A census built from ``tag_peak`` alone would omit 910-2606 MiB per
  card and then be refused by ``check_partition`` for a wave tag it has no
  bytes for.

Hermetic: temp files only, no GPU, no evidence tree.
"""

import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CARD_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
CARD_B = "GPU-bbbbbbbb-0000-0000-0000-000000000002"


def _tag_line(group, rank, card, tag, mib, population=ring_table.TAG_POPULATION_ALL):
    return (
        f"[2026-09-09T15:11:45Z] INFO weg2.front: WEG2-FLIP-TAG group={group} "
        f"rank={rank} card={card} dir=d2h tag={tag} bytes={mib} MiB "
        f"population={population} (source: tms_tag_bytes) ms=10 GB/s=1.0\n"
    )


def _group_log(*lines):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "g.P.log")
    with open(p, "w") as fh:
        fh.writelines(lines)
    return p


#: Two cards, two chunk tags and the family BASE tag, all under the gathered
#: leg's ``rank=-1`` form (which is how the ring-era instrument writes it).
#: Every number is distinct so no assertion below can pass by coincidence, and
#: the two cards differ per tag so a card-collapsing bug cannot hide.
TWO_CARDS = (
    _tag_line("P", -1, CARD_A, "weights_0", 1001),
    _tag_line("P", -1, CARD_A, "weights_1", 1102),
    _tag_line("P", -1, CARD_A, "weights", 203),
    _tag_line("P", -1, CARD_B, "weights_0", 3004),
    _tag_line("P", -1, CARD_B, "weights_1", 3105),
    _tag_line("P", -1, CARD_B, "weights", 406),
)


class TheRingTableRetainsItsPerRankTagPeaks(CustomTestCase):
    def test_tag_by_rank_exists_and_is_per_rank_per_tag(self):
        g = ring_table.parse_group_log(_group_log(*TWO_CARDS))
        ranks = sorted(g.uuid_by_rank)
        self.assertEqual(len(ranks), 2, g.uuid_by_rank)
        a = next(r for r in ranks if g.uuid_by_rank[r] == CARD_A)
        b = next(r for r in ranks if g.uuid_by_rank[r] == CARD_B)
        self.assertEqual(g.tag_by_rank["weights_0"], {a: 1001, b: 3004})
        self.assertEqual(g.tag_by_rank["weights_1"], {a: 1102, b: 3105})

    def test_the_family_base_tag_is_retained_though_tag_totals_drops_it(self):
        """The one term the census needs and ``tag_peak`` provably lacks."""
        g = ring_table.parse_group_log(_group_log(*TWO_CARDS))
        self.assertNotIn("weights", g.tag_totals)
        a = next(r for r in g.uuid_by_rank if g.uuid_by_rank[r] == CARD_A)
        b = next(r for r in g.uuid_by_rank if g.uuid_by_rank[r] == CARD_B)
        self.assertEqual(g.tag_by_rank["weights"], {a: 203, b: 406})

    def test_tag_totals_is_unchanged_and_equals_the_per_rank_sums(self):
        """The byte-identity pin: the summation at the PEAK pass is untouched.

        Stated as an IDENTITY between the two fields rather than as a golden
        number, so it keeps discriminating when the fixture changes -- and it
        is asserted over ``tag_totals``' OWN key set, because ``tag_by_rank``
        is a strict superset by design (the base tag above).
        """
        g = ring_table.parse_group_log(_group_log(*TWO_CARDS))
        self.assertEqual(sorted(g.tag_totals), ["weights_0", "weights_1"])
        for tag, total in g.tag_totals.items():
            self.assertEqual(total, sum(g.tag_by_rank[tag].values()), tag)
        self.assertEqual(g.tag_totals, {"weights_0": 4005, "weights_1": 4207})

    def test_dropping_the_base_tag_loses_exactly_the_base_tags_bytes(self):
        """Why the retention is taken one filter earlier than ``tag_peak``.

        ``image`` is the PEAK pass over the records, so with one pass per tag
        it is that pass's own maximum and cannot be asserted against a sum
        here.  What IS asserted is the property that makes the base tag a real
        tag: a census restricted to ``tag_totals``' keys is short by exactly
        the base tag's measured bytes, never by zero.
        """
        g = ring_table.parse_group_log(_group_log(*TWO_CARDS))
        with_base = sum(sum(p.values()) for p in g.tag_by_rank.values())
        without_base = sum(
            sum(p.values()) for t, p in g.tag_by_rank.items() if t in g.tag_totals
        )
        self.assertEqual(with_base - without_base, 203 + 406)

    def test_a_second_pass_takes_the_peak_not_the_last_write(self):
        """Same rule ``tag_peak`` already follows, now visible per rank."""
        lines = TWO_CARDS + (
            _tag_line("P", -1, CARD_A, "weights_0", 7),
            _tag_line("P", -1, CARD_A, "weights_1", 9999),
        )
        g = ring_table.parse_group_log(_group_log(*lines))
        a = next(r for r in g.uuid_by_rank if g.uuid_by_rank[r] == CARD_A)
        self.assertEqual(g.tag_by_rank["weights_0"][a], 1001)
        self.assertEqual(g.tag_by_rank["weights_1"][a], 9999)
        self.assertEqual(g.tag_totals["weights_1"], 9999 + 3105)


if __name__ == "__main__":
    unittest.main()
