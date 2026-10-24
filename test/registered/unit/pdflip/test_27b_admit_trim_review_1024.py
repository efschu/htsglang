# SPDX-License-Identifier: Apache-2.0
"""Review 24.10. of desk/27b-admit-trim-1024 @ 17edfeebc7 (PDFLIP-ADMIT-BOOK): the census misses every extend pass
that follows decode rounds.

``vram_peak_window.on_forward_end`` closes a pending ``phase=round`` window at the END of the extend forward, right
before the extend's own ``phase=chunk`` line (same instant: round ``t_unix_ms`` == chunk ``t0_unix_ms``). The
extend's vote trim and forward trim are logged at the forward's START, i.e. BEFORE that round line.
``budget_rest.admission_passes`` treats every round line as the end of a pass and drops the events before it, so the
chunk line that follows sees an empty pass (post None, fwd_before None). On the six int4ar boots that is ~1980 of
the 7604 chunk windows -- exactly the serving case (an admission while D decodes others).

Raw lines (int4ar, budget 17480/17264, rest 2079):
- ``_1023_223241`` 23:19:34: a handback is admitted after 568 refusals (174 s); the forward trim finds TP1 at
  card_free_before=32 MiB (TP0 222), no vote trim before it (the vote's ``armed`` survived the refused passes).
- ``_1024_003732`` 00:56:21: TP1 post 648 (vote trim), forward trim card_free_before=210 -- below the 216 the
  author's census reports as the minimum.

These tests are RED on 17edfeebc7 by design: they pin the instants the record must hold.
"""

import os

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402

FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "admit_trim_1024")
# the SHIPPED record, booked up to the 8-MiB grain (17edfeebc7 booked 320 and went red here; the author's rework reads
# the record so the test follows it -- it must stay >= 368 for the 32 MiB instant)
BOOKED = -(-int(F.PROFILES[F.PROFILE_QWEN27B].constants[BR.ADMIT_RECORD].value[0]) // 8) * 8


def _read(name):
    with open(os.path.join(FIX_DIR, name)) as fh:
        return fh.read()


class TheRoundLineClosedAtTheExtendEndIsNotAPassBoundary(CustomTestCase):
    def test_the_32_mib_instant_is_attributed_to_its_chunk(self):
        got = BR.admission_passes(_read("excerpt_review_223241_round_D.txt"))
        tp1 = {p.rows: p for p in got[1]}
        tp0 = {p.rows: p for p in got[0]}
        # 247-row chunk: one trim (the forward trim, no vote trim before the admission)
        self.assertEqual((tp1[247].fwd_before, tp1[247].post, tp1[247].vote_released), (32.0, None, None))
        self.assertEqual(tp0[247].fwd_before, 222.0)
        # 207-row chunk: vote trim 620 -> 680, forward trim at 610 (admission 70)
        self.assertEqual((tp1[207].post, tp1[207].fwd_before), (680.0, 610.0))

    def test_the_210_mib_admission_after_decode_rounds(self):
        got = BR.admission_passes(_read("excerpt_review_003732_round_D.txt"))
        p = {q.rows: q for q in got[1]}[70]
        self.assertEqual((p.post, p.post_src, p.fwd_before), (648.0, "trim", 210.0))
        self.assertEqual(p.admission, 438.0)


class TheBookedPostHoldsTheMissedInstants(CustomTestCase):
    def test_every_raw_forward_trim_plus_the_booked_post_reaches_the_edge(self):
        """Independent of the parser: every forward trim's card_free_before in the excerpts + the booked post."""
        lows = []
        for name in ("excerpt_review_223241_round_D.txt", "excerpt_review_003732_round_D.txt"):
            for ln in _read(name).splitlines():
                if BR._RX_RANK.match(ln) and "PDFLIP-EXTEND-CACHE-TRIM" in ln:
                    lows.append(float(dict(BR._RX_KV.findall(ln))["card_free_before"]))
        self.assertEqual(min(lows), 32.0)
        self.assertGreaterEqual(min(lows) + BOOKED, CG.NEAR_OOM_MIB)
