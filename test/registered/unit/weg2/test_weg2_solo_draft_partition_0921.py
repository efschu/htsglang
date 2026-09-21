"""fnFL2 v37 (21.09.): a wave tag the SOLO-placed drafter puts on exactly one
card must not refuse the exchange on the other two.

``--speculative-draft-placement solo`` puts the MTP head on one board.  Group
P carries the same head (``--draft-kv-on-p on``, so the draft KV crosses the
flip through HiCache instead of being re-prefilled), so the tag is symmetric
per card: present in BOTH groups on the card that holds it, absent from BOTH
on the cards that do not.  ``check_partition`` refused the second case and the
boot died with W71 on five (card x direction) cases whose printed arithmetic
all fit.

The ASYMMETRIC gap stays a refusal: one group holding bytes the other does not
means the exchange frees what it never takes back.
"""

import unittest

from sglang.srt.weg2 import xchg_residency

BIG = "GPU-aaaa"
SM = "GPU-bbbb"
WAVES = (("weights_0",), ("weights_draft",))


def _census(sm_tags):
    return xchg_residency.XchgCensus(
        cards={
            BIG: xchg_residency.CardCensus(
                uuid=BIG,
                tags={"P": {"weights_0": 100, "weights_draft": 33},
                      "D": {"weights_0": 90, "weights_draft": 33}},
                dormant_proc_used_mib=10,
            ),
            SM: xchg_residency.CardCensus(
                uuid=SM, tags=sm_tags, dormant_proc_used_mib=10
            ),
        },
        waves=WAVES,
    )


class SoloDraftPartition(unittest.TestCase):
    def test_a_tag_absent_from_both_groups_on_one_card_is_not_a_refusal(self):
        res = xchg_residency.check_partition(
            _census({"P": {"weights_0": 50}, "D": {"weights_0": 50}})
        )
        self.assertEqual(res, [])

    def test_an_asymmetric_gap_on_one_card_still_refuses_and_names_the_tag(self):
        res = xchg_residency.check_partition(
            _census({"P": {"weights_0": 50, "weights_draft": 7},
                     "D": {"weights_0": 50}})
        )
        self.assertEqual(len(res), 1, res)
        self.assertIn("weights_draft", res[0])
        self.assertIn("group D", res[0])

    def test_the_card_that_holds_the_solo_draft_prices_it_in_both_directions(self):
        """Not a partition question: the tag must still reach the arithmetic."""
        card = type("C", (), {})()
        card.uuid, card.nvml_index, card.name, card.total_mib = BIG, 1, "big", 4096
        small = type("C", (), {})()
        small.uuid, small.nvml_index, small.name, small.total_mib = SM, 0, "sm", 4096
        res = xchg_residency.solve(
            [card, small],
            _census({"P": {"weights_0": 50}, "D": {"weights_0": 50}}),
            floor_mib=10,
        )
        self.assertEqual(res.refusals, [])
        big_rows = [r for r in res.rows if r.uuid == BIG and r.direction == "d2p"]
        # wave 2 takes P's draft bytes on top of P's weights_0
        self.assertEqual(big_rows[-1].taken_w_mib, 133)
        sm_rows = [r for r in res.rows if r.uuid == SM and r.direction == "d2p"]
        self.assertEqual(sm_rows[-1].taken_w_mib, 50)


if __name__ == "__main__":
    unittest.main()
