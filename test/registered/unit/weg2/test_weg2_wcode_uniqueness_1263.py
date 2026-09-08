# SPDX-License-Identifier: Apache-2.0
"""One W-code, one exception name -- enumerated, not remembered.

THE CLASS, three instances in one day and every one found by a READER:

* W31 named both this rig's serving-path re-route (``Weg2TpPrefillExceeded``)
  and ``Weg2HostRingExhausted``, the host-ring exhaustion that killed boot
  weg2tr2. A census that greps ``W31`` reports one number for a fatal
  exhaustion and a routing event -- how a postmortem merges a killer into
  noise. Train fix 5 renumbered it.
* It renumbered it to W47, which the #1235 argv slice had ALREADY assigned to
  ``Weg2TpObjectiveRefused``. One collision traded for another, because the
  new number was picked rather than enumerated.
* Renumbered again here to W50, this time by enumerating the used set.

A third hand-picked number would repeat it, so the enumeration is the guard
rather than the fix: this file recomputes the census on every run.

WHY A CENSUS AND NOT A REGISTRY. There is no table of W-codes to keep in sync
-- the code IS the message text -- so a registry would be second bookkeeping
beside the refusals themselves, and would drift exactly the way the front's
docstring drifted from its own constant. Scanning the messages is the one
reading that cannot go stale.

WHAT IS ASSERTED, and what is deliberately NOT:

* NEW collisions fail. That is the whole point.
* The FOUR pre-existing collisions are pinned as a named, dated set rather
  than silently allowed. Pinning them exactly means fixing one also fails
  here, which forces the list to shrink deliberately instead of rotting into
  a permanent allow-list. They are NOT fixed on this branch: it is the
  serving-boot base, and renumbering four more codes across five modules and
  their tests is a bigger diff than a label defect justifies before a boot.
  Each carries the two names and where they live so the next pass has the
  work already done.
"""

import os
import re
import unittest
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase

#: Where a Weg-2 refusal can be raised, logged or re-routed from. Enumerated
#: from the modules that actually carry ``W<nn> Weg2<Name>`` text today; a new
#: module is picked up by the directory walks, not by an edit here.
ROOTS = (
    "python/sglang/srt/weg2",
    "python/sglang/srt/managers",
    "python/sglang/srt/mem_cache",
    "scripts/weg2",
)
FILES = ("python/sglang/srt/server_args.py",)

#: ``W12``, ``W11b``, ``W40b`` -- the letter suffix is part of the code.
ASSIGNMENT = re.compile(r"\b(W\d{1,2}[a-z]?)\s+(Weg2[A-Za-z0-9_]+)")

#: THE KNOWN COLLISIONS, 2026-09-08, with both holders. Deferred, not
#: accepted: this branch renumbers W47 only (see the module docstring).
KNOWN_COLLISIONS = {
    "W10": {"Weg2CanonicalPageMissing", "Weg2DrafterIdentityMismatch"},
    "W35": {"Weg2VramCreditRefused", "Weg2XReQueueLoop"},
    "W36": {"Weg2AdmitterBarrierExpired", "Weg2DuplexDecisionRefused"},
    "W46": {"Weg2PPSplitMapMismatch", "Weg2TokenVectorRefused"},
}


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "python", "sglang", "srt", "weg2")):
            return here
    raise AssertionError("could not locate the repo root from this test file")


def census():
    """``{code: {exception_name: [file:line, ...]}}`` over the whole surface.

    Reads the SOURCE, never the imported module: a refusal inside an ``f``
    string is only text at runtime, and half of these are logged rather than
    raised, so there is no object to enumerate.
    """
    root = _repo_root()
    paths = [os.path.join(root, f) for f in FILES]
    for rel in ROOTS:
        for dirpath, _dirs, names in os.walk(os.path.join(root, rel)):
            paths += [
                os.path.join(dirpath, n) for n in names
                if n.endswith((".py", ".sh"))
            ]
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted(set(paths)):
        try:
            with open(p, encoding="utf-8") as fh:
                lines = fh.read().split("\n")
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for code, name in ASSIGNMENT.findall(line):
                out[code][name].append(f"{os.path.relpath(p, root)}:{i}")
    return out


class TestOneWCodePerException(CustomTestCase):

    def test_the_census_actually_finds_the_codes(self):
        """DENOMINATOR FIRST. A scan that silently matched nothing would pass
        every assertion below and prove nothing -- the failure mode this whole
        family of guards keeps producing."""
        c = census()
        self.assertGreater(len(c), 20, "the W-code scan found almost nothing")
        self.assertIn("W50", c, "the renumbered prefill refusal must be found")
        self.assertIn("W47", c, "the objective refusal must still be found")

    def test_no_new_collision(self):
        """RED-FIRST: this failed at the parent 64a2829afd with W47 in the
        list (two names, Weg2TpObjectiveRefused + Weg2TpPrefillExceeded)."""
        c = census()
        collisions = {code: set(names) for code, names in c.items() if len(names) > 1}
        new = {k: v for k, v in collisions.items() if k not in KNOWN_COLLISIONS}
        self.assertEqual(
            new, {},
            "a W-code names more than one exception. Enumerate the used set "
            "(this file's census) and pick a FREE number -- picking the next "
            "one by hand is how W31 became W47 and W47 became a second "
            "collision.",
        )

    def test_the_known_collisions_are_exactly_the_recorded_ones(self):
        """Pinned EXACTLY, so fixing one fails here too and the list shrinks
        deliberately rather than becoming a permanent allow-list."""
        c = census()
        collisions = {code: set(names) for code, names in c.items() if len(names) > 1}
        self.assertEqual(
            collisions, KNOWN_COLLISIONS,
            "the known-collision list no longer matches reality: a collision "
            "was fixed (shrink the list, in the same commit) or the census "
            "changed shape.",
        )

    def test_w47_is_the_objective_refusal_alone(self):
        c = census()
        self.assertEqual(set(c["W47"]), {"Weg2TpObjectiveRefused"})

    def test_w50_is_the_prefill_refusal_alone_and_the_wire_is_name_keyed(self):
        c = census()
        self.assertEqual(set(c["W50"]), {"Weg2TpPrefillExceeded"})
        # The durable half: the leg-2 detector matches the NAME, so the label
        # can be renumbered without breaking the wire. Asserted because that
        # property is the reason this renumber is safe at all.
        from sglang.srt.weg2 import front

        self.assertEqual(front.X_REFUSAL_MARKER, "Weg2TpPrefillExceeded")
        self.assertIsNone(
            re.match(r"W\d", front.X_REFUSAL_MARKER),
            "the wire marker must carry no W-CODE at all -- a protocol "
            "keyed on a renumberable label breaks at the renumber that "
            "fixes the collision",
        )
        self.assertTrue(front.X_REFUSAL_NAME.startswith("W50 "))
        self.assertTrue(
            front.x_refusal_marker_in("... W50 Weg2TpPrefillExceeded ...")
        )
        # ... and it still matches a body carrying the OLD label, because the
        # match never depended on the number.
        self.assertTrue(
            front.x_refusal_marker_in("... W47 Weg2TpPrefillExceeded ...")
        )

    def test_the_chosen_number_was_free_and_the_free_ones_are_named(self):
        """W50 is not 'the next one': it is the first free number above the
        highest assigned code, and the census can say which others are free."""
        c = census()
        used = {int(m.group(1)) for code in c
                for m in [re.match(r"W(\d{1,2})", code)] if m}
        self.assertNotIn(50, used - {50}, "W50 must have been free before this")
        for n in (15, 18, 23, 39):
            self.assertNotIn(n, used, f"W{n} was named free and is not")


if __name__ == "__main__":
    unittest.main()
