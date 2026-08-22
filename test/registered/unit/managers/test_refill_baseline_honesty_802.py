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
"""The REFILL instrument must not invite an invalid comparison.

MEASURED HARM, 2026-08-22. The line read:

    REFILL <dir> took 13.47 s for 12103.7 MiB (899 MiB/s) -- file-backed
    images. Compare against the ~3.1 s pinned baseline: ...

and a briefing built on it concluded the flip economy was "broken" with an
8x regression to hunt, attributing it to a suspected silent host-RAM
fallback. Every step of that reading is invited by the line, and the
comparison it offers is invalid three times over:

1. SCOPE. The ~3.1 s figure is a WHOLE FLIP, not a refill leg.
   NOTE_677_floor_components.md:135-143 uses it as "Against a ~3.1 s flip".
   The line places it beside a single leg's duration.
2. PATH. #690 measured the PINNED image path
   (NOTE_690_gdn_state_spread.md:58-85). That measurement predates the
   file-backed arm entirely, so it is not a baseline this path ever had.
3. BYTES. #690 moved 9614.9 MiB per rank, pp_to_tp. The refills being
   compared move 8574-16363 MiB. Comparing seconds across different byte
   counts hides that time here tracks bytes moved (r ~ 0.80).

The file-backed arm is not a regression and not a fallback: it is an
explicit opt-in (--phase-flip-image-file-backed), and its help text names
what it buys -- without it the images are ~68.7 GiB of unreclaimable host
RAM on a swapless box and the boot is OOM-killed during init. A reader who
takes the line at face value goes hunting for a defect where a deliberate
purchase is recorded.

So the instrument must report a RATE against a rate, and state the
baseline's conditions rather than a bare second-count. It may still say
the arm is slower -- it is -- but it must not let that read as a
regression against a baseline this path never had.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import phase_flip_boot as pfb

MIB = 1048576


def _line(elapsed=13.47, mib=12103.7, file_backed=True):
    return pfb.refill_report("pp_to_tp", elapsed, int(mib * MIB), file_backed)


class TestRefillBaselineHonesty(unittest.TestCase):
    def test_reports_an_achieved_rate(self):
        line = _line()
        self.assertIn("MiB/s", line, "a rate is the only comparable quantity")

    def test_does_not_offer_a_bare_second_count_as_the_baseline(self):
        """The measured harm, stated as a test."""
        line = _line()
        self.assertNotIn(
            "Compare against the ~3.1 s",
            line,
            "a whole-flip figure may not stand beside a single leg's duration",
        )

    def test_names_the_baseline_conditions(self):
        """A baseline without its conditions is not transferable."""
        line = _line().lower()
        for term in ("pinned", "whole flip"):
            self.assertIn(term, line, f"the baseline must name: {term}")

    def test_states_the_arm_is_a_deliberate_purchase(self):
        line = _line()
        self.assertIn(
            "reclaimable",
            line,
            "the reader must see what the slower path buys, not just its cost",
        )

    def test_pinned_path_does_not_claim_to_be_paying_for_reclaimability(self):
        line = _line(file_backed=False)
        self.assertIn("pinned", line)
        self.assertNotIn(
            "68.7",
            line,
            "the pinned arm buys nothing here; do not print the trade it declined",
        )

    def test_instrument_never_raises(self):
        """An instrument may never break a flip -- including on absurd input."""
        for elapsed in (0.0, -1.0):
            pfb.refill_report("pp_to_tp", elapsed, 0, True)


if __name__ == "__main__":
    unittest.main()
