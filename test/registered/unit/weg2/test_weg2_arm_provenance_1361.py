# SPDX-License-Identifier: Apache-2.0
"""#1361 [23a]/[23b] -- two numbers on the ARM line that had more than one reading.

(23a) `xchg_bounce=2.16` had THREE causes on the code of 2026-09-13, measured
      by the exchange seat: (depth=1, comparing=True), (depth=2,
      comparing=False), and (depth=2 under the pre-#1335 expression that
      under-priced by one widest layer). The figure alone cannot tell a
      correctly priced shallow arm from the underpricing -- and this seat
      proved that the hard way: it read 2.16 off its own dry run, asserted the
      cause was `depth=1`, and was wrong. The tree simply did not carry the fix
      yet. Same number, third cause, wrong one picked, by the seat that had
      just been warned about exactly this.

(23b) `S_D=2` was read by the boot seat as `S_D=1` from its own sizing table,
      because `_derive_d_l2_budget` lets a measured ownership vector in the
      sizing record OVERRIDE `--d-cap-rank-share` silently. Effective and
      unprinted: the #896 class.

Both are print-only. Neither moves an arm; both make an arm's number
attributable to the input that produced it.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

MT = 126 * 1024 ** 3
MA = 110 * 1024 ** 3


def _lines(**kw):
    _, _, lines = hl.choose(MT, MA, ring_bytes=1, ring_span1_bytes=1,
                            ring_provenance="t", **kw)
    return [x for x in lines if "WEG2-HOST-LEDGER ARM " in x]


class TheBounceStateIsPrintedBesideTheBounceTerm(CustomTestCase):
    PROV = {"depth": 2, "slots": 3, "inject_mode": "shadow"}

    def test_the_three_fields_ride_with_the_term(self):
        for ln in _lines(xchg_bounce_prov=dict(self.PROV)):
            self.assertIn("bounce_depth=2", ln)
            self.assertIn("bounce_slots=3", ln)
            self.assertIn("bounce_inject=shadow", ln)

    def test_two_states_with_the_SAME_price_are_distinguishable(self):
        """The whole point: equal numbers, different lines.

        A shallow arm that is correctly priced and a deep arm that is
        under-priced can cost the same. If the line does not separate them,
        neither can a reader.
        """
        shallow = _lines(xchg_bounce_prov={"depth": 1, "slots": 2,
                                           "inject_mode": "shadow"})[0]
        deep = _lines(xchg_bounce_prov={"depth": 2, "slots": 2,
                                        "inject_mode": "authoritative"})[0]
        self.assertNotEqual(
            shallow.split("host_weights")[0], deep.split("host_weights")[0],
            "two different priced states produced an identical line",
        )

    def test_an_unpriced_bounce_says_absent_not_a_default(self):
        """An unpriced state and a state priced at defaults must not share a
        spelling -- the rule the ratchet fields already obey."""
        for ln in _lines(xchg_bounce_prov=None):
            self.assertIn("bounce_state=absent", ln)
            self.assertNotIn("bounce_depth=", ln)

    def test_it_is_print_only_and_never_reaches_price(self):
        """Reachability in the OTHER direction: a print term that moved an arm
        would be a pricing change wearing a comment."""
        import inspect

        self.assertNotIn("xchg_bounce_prov", inspect.signature(hl.price).parameters)


class TheShareSourceIsNamed(CustomTestCase):
    def test_record_and_flag_are_distinguishable(self):
        rec = _lines(s_gb_d=4, d_cap_terms={"max_rank_share": 0.375,
                                            "max_rank_share_source": "record"})
        flg = _lines(s_gb_d=4, d_cap_terms={"max_rank_share": 0.4,
                                            "max_rank_share_source": "flag"})
        self.assertTrue(any("source=record" in x for x in rec))
        self.assertTrue(any("source=flag" in x for x in flg))
        self.assertTrue(any("sd_share=0.3750" in x for x in rec))

    def test_an_unstamped_terms_dict_says_unknown_not_flag(self):
        """Absence is not the flag. A terms dict written before this commit
        cannot testify, and guessing `flag` would be the same error as reading
        an absent pids field as an empty one ([22-fix5b])."""
        out = _lines(s_gb_d=4, d_cap_terms={"max_rank_share": 0.4})
        self.assertTrue(any("source=unknown" in x for x in out))

    def test_no_terms_dict_adds_nothing_to_the_line(self):
        for ln in _lines():
            self.assertNotIn("sd_share=", ln)


if __name__ == "__main__":
    unittest.main()
