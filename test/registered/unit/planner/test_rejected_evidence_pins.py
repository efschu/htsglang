# SPDX-License-Identifier: Apache-2.0
"""The rejected register's ``evidence`` must point at code that still exists (#625).

The register's whole premise is that a rejection carries the number or the
code site that produced it -- "a row with no evidence would be an opinion, and
an opinion cannot bind a later attempt" (``planner/rejected.py`` docstring).
An evidence string that names a FILE:LINE which has since moved is the same
failure in slow motion: the row still reads as authoritative while pointing at
unrelated code.

That is not hypothetical. #625 re-checked ``pp_with_spec``, whose evidence
read ``server_args.py:11214``; the assert it describes had drifted to
``:16240-16245``, and line 11214 by then sat in the middle of an unrelated
reserve-migration warning. The verdict was still correct -- the assert is
real and still hard -- but the pointer was not, and the register's own rule
is that verdicts are re-checked before every feature order.

This pins the CODE-SITE rows, not every row: a row whose evidence is a
measurement ("four boots x 16 points") has nothing in the tree to point at.
The pin is deliberately on the assert's TEXT rather than on its line number,
so ordinary edits above it do not turn this red -- only the assert actually
changing or disappearing does.
"""

import re
import unittest
from pathlib import Path

from sglang.srt.planner.rejected import BLOCKED, by_key

_SRT = Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
_SERVER_ARGS = _SRT / "server_args.py"

_LINE_REF = re.compile(r"([A-Za-z0-9_/]+\.py):(\d+)")


class PpWithSpecEvidenceTest(unittest.TestCase):
    """``pp_with_spec`` names the assert that actually blocks the combination."""

    def test_entry_is_still_blocked(self):
        entry = by_key("pp_with_spec")
        self.assertEqual(entry.level, BLOCKED)

    def test_evidence_names_server_args(self):
        entry = by_key("pp_with_spec")
        match = _LINE_REF.search(entry.evidence)
        self.assertIsNotNone(
            match, f"evidence carries no file:line reference: {entry.evidence!r}"
        )
        self.assertEqual(match.group(1), "server_args.py")

    def test_evidence_cites_land_on_BOTH_halves_of_the_guard(self):
        """The guard is two asserts now, so the row cites two lines.

        It used to be one conjunction. #704b found it split: the overlap half
        and the speculation half are separate statements 15 lines apart, so a
        single cite cannot land on both and a window around one no longer
        mentions the other.
        """
        entry = by_key("pp_with_spec")
        lines = _SERVER_ARGS.read_text().splitlines()
        cites = [
            int(n) for f, n in _LINE_REF.findall(entry.evidence) if f == "server_args.py"
        ]
        self.assertEqual(len(cites), 2, f"expected two cites, got {cites}")
        for cited in cites:
            self.assertLessEqual(cited, len(lines), "cite past end of server_args.py")
        head, spec = cites
        advice = self._where_the_guard_actually_is(lines)
        self.assertIn("pp_size > 1", "\n".join(lines[head - 1 : head + 2]), advice)
        window = "\n".join(lines[max(0, spec - 3) : spec + 8])
        self.assertIn("speculative_algorithm is None", window, advice)
        self.assertIn("enable_phase_flip", window, advice)

    @staticmethod
    def _where_the_guard_actually_is(lines) -> str:
        """Say what the cite SHOULD read, not merely that it is wrong.

        #898, 2026-08-26: this cite has now drifted and been re-pinned FOUR
        times -- #625 (:11214 -> :16240-16245), #815 78d27da51d
        (-> :18958/:18973), #810 a09e71f4a4, #837 ddf009c43f (-> :19269/:19284)
        -- and drifted a fifth time to :19436/:19451. Each re-pin was a hunt
        through server_args.py by hand.

        The CLASS is an absolute line number kept in prose about a file that
        grows above it; the class fix is a landmark-based citation format,
        which changes `_LINE_REF`, every code-site row and every reader, and is
        NOT free. What IS free is refusing to make the next reader hunt: the
        test that notices the drift already knows how to find the guard, so it
        prints the correct pair. See DETERMINATION_898 §4.4.
        """
        head = spec = None
        for i, line in enumerate(lines, start=1):
            if head is None and line.strip() == "if self.pp_size > 1:":
                candidate = i
                for j in range(i, min(i + 40, len(lines))):
                    if (
                        "speculative_algorithm is None or self.enable_phase_flip"
                        in lines[j]
                    ):
                        head, spec = candidate, j + 1
                        break
        if head is None:
            return (
                "the pp_size>1 / speculation guard was not found in "
                "server_args.py at all -- the row may describe a guard that "
                "no longer exists"
            )
        return (
            f"evidence cite has drifted; the guard now sits at "
            f"server_args.py:{head} (if pp_size > 1) and server_args.py:{spec} "
            f"(spec assert). Update the pp_with_spec row in planner/rejected.py."
        )

    def test_the_spec_half_is_still_a_hard_assert(self):
        """The verdict says 'hard assert, not an auto-disable'. Pin that word.

        If this is ever softened to a warning plus auto-disable, a PP boot
        would silently lose speculation instead of refusing, and this test is
        the thing that notices.
        """
        self.assertRegex(
            _SERVER_ARGS.read_text(),
            r"if self\.pp_size > 1:(?:.|\n)*?assert self\.speculative_algorithm is None"
            r" or self\.enable_phase_flip",
            "the pp_size>1 / speculation guard is no longer the hard assert the "
            "pp_with_spec register row describes",
        )

    def test_the_register_records_the_PHASE_FLIP_EXEMPTION(self):
        """The row must not claim PP is unconditionally no-spec.

        This is the check the old pin could not make, and the reason the row
        was stale rather than merely mis-pointed: the assert grew an
        `or self.enable_phase_flip` escape, so a phase-flip instance MAY run
        PP-prefill and speculation in one engine. A register row that still
        read 'every PP number is a no-spec number' would forbid on paper the
        exact configuration #704 is built on.
        """
        entry = by_key("pp_with_spec")
        text = (entry.verdict + " " + entry.why).lower()
        self.assertIn("phase-flip", text)
        self.assertIn("plain", text)

    def test_the_overlap_half_is_now_an_AUTO_DISABLE(self):
        """The sibling half went the other way, and the row says so.

        `_pipeline_parallel_overlap_disable` sets disable_overlap_schedule to
        True and warns, running BEFORE the assert, so that assert can no
        longer fire. Recording only the surviving hard assert would leave the
        register implying both halves still refuse.
        """
        overrides = (_SRT / "arg_groups" / "overrides.py").read_text()
        self.assertRegex(
            overrides,
            r"def _pipeline_parallel_overlap_disable\(view: Any\) -> dict:\s*\n"
            r"\s*if view\.pp_size > 1:(?:.|\n)*?"
            r'return \{"disable_overlap_schedule": True\}',
            "the overlap half is no longer the warning + auto-disable the "
            "pp_with_spec row describes",
        )
        # Naming the field is what makes this a real check: the PRE-#704b row
        # also contained the string "auto-disable", inside the phrase "not a
        # quiet auto-disable" -- the opposite claim. A substring test alone
        # would have passed against the stale text it is meant to catch.
        entry = by_key("pp_with_spec")
        why = entry.why.lower()
        self.assertIn("auto-disable", why)
        self.assertIn("disable_overlap_schedule", why)


if __name__ == "__main__":
    unittest.main()
