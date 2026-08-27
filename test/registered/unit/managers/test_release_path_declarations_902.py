"""#902 -- every held resource declares a door, and the check says which
layouts can open it.

THE CLASS, from NOTE_888b §5 verbatim: "a resource held by a resident that the
current layout forbids to make progress, with no release path reachable from
that layout." The seat fix proved the shape is real. The note's own sweep then
found the siblings -- the mamba anchor pin with no release at all, the kvso
host region whose only doors are finish/abort while the seam RETRACTS rather
than finishes -- and left them as prose in a document. Prose is what a boot
finds again, one resource at a time, one window each.

WHY IT GOES HERE AND NOT IN A NEW MECHANISM. `cutover_participants.py` already
declares WHO moves a participant (`hook`), PROOF the mover ran (`probe`), and
WHEN its state is honest (`ReadWindow`). NOTE_888b named the missing fourth
declaration before it existed: "the READ WINDOW's twin -- which paths a
resource's release site lives on, and whether every layout can reach one of
them." This is that field and the check that reads it. No new machinery, and
the file's own promise -- "what this list forgets, a boot finds" -- now covers
release as well as movement.

NOTHING HERE CHANGES A RELEASE, and the boundary is deliberate. The check
warns by name and returns findings; it never blocks a flip. A conformance
check that also rewired the releases it judges would be two changes wearing
one ticket, and the second one's failure mode is a wedged cutover mid-window
-- the outcome that ended two of this fork's windows.

WHAT EACH TEST HOLDS DOWN
  1. every row names a real symbol                  -- a registry of phantoms
     is worse than none, and three of this file's first drafts pointed at
     modules that did not hold the function;
  2. a row with no door is reported UNDECLARED      -- the #773 §8 pin;
  3. a finish/abort-only door is UNREACHABLE at the seam -- the kvso host
     region, NOTE_888b's named and still-unfixed sibling;
  4. a seam door is reachable from both layouts     -- mutant guard: a check
     that flags everything names nothing;
  5. strict=False lifts the prefill bar only        -- the rule tracks the
     purity law rather than restating it;
  6. the declaration is actually CONSULTED          -- the mutant the ticket
     names: a field nobody reads is a field that drifts.
"""

import importlib
import unittest

from sglang.srt.managers.cutover_participants import (
    HELD_RESOURCES,
    LAYOUT_CANNOT_REACH,
    ReleasePath,
    release_path_conformance,
)


def _resolve(dotted: str):
    """Import a dotted path that may end in Class.method."""
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        obj = mod
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj
    return None


class TestReleasePathDeclarations902(unittest.TestCase):
    def test_every_declared_release_symbol_exists(self):
        """A registry of phantoms is worse than no registry.

        Rows carrying ``pending_branch`` are skipped BY NAME -- their symbol
        lands with that branch, and leaving the row out until the merge is how
        a population thins.
        """
        missing = []
        for res in HELD_RESOURCES:
            if res.released_by is None or res.pending_branch:
                continue
            if _resolve(res.released_by) is None:
                missing.append(f"{res.name} -> {res.released_by}")
        self.assertEqual(missing, [], f"declared but absent: {missing}")

    def test_a_row_with_no_door_is_reported_undeclared(self):
        """#773 §8's pin: held, with no release site at all."""
        findings = release_path_conformance("pp")
        self.assertTrue(
            any(f.startswith("UNDECLARED mamba_anchor_pin") for f in findings),
            f"the pin has no door and must say so; got {findings}",
        )

    def test_a_finish_only_door_is_unreachable_at_the_seam(self):
        """The kvso host region -- NOTE_888b's named, still-unfixed sibling.

        Its only doors are finish/abort, and the seam retracts rather than
        finishes, so a parked session holds its host region for its whole
        life. That must be a finding in BOTH target layouts: the defect is
        that no layout reaches it, not that one does not.
        """
        for layout in ("pp", "tp"):
            findings = release_path_conformance(layout)
            self.assertTrue(
                any(f.startswith("UNREACHABLE kvso_host_region") for f in findings),
                f"{layout}: expected the host region to be unreachable; "
                f"got {findings}",
            )

    def test_a_seam_door_is_reachable_from_both_layouts(self):
        """MUTANT GUARD. A check that flags everything names nothing."""
        for layout in ("pp", "tp"):
            findings = release_path_conformance(layout)
            for name in ("draft_weights", "batch_is_full_latch", "request_seat"):
                self.assertFalse(
                    any(name in f for f in findings),
                    f"{layout}: {name} has a seam door and must not be "
                    f"flagged; got {findings}",
                )

    def test_strict_false_lifts_the_prefill_bar_only(self):
        """The rule tracks the purity law instead of restating it."""
        self.assertEqual(LAYOUT_CANNOT_REACH["tp"], (ReleasePath.PREFILL,))
        self.assertEqual(LAYOUT_CANNOT_REACH["pp"], (ReleasePath.DECODE,))

        strict = release_path_conformance("tp", strict=True)
        relaxed = release_path_conformance("tp", strict=False)
        self.assertLessEqual(
            len(relaxed),
            len(strict),
            "a non-strict boot may prefill in TP, so lifting the bar can only "
            "remove findings",
        )
        # Decode is never permitted in the PP window under either setting.
        self.assertEqual(
            release_path_conformance("pp", strict=True),
            release_path_conformance("pp", strict=False),
        )

    def test_the_declaration_is_actually_consulted(self):
        """THE MUTANT THE TICKET NAMES: a field nobody reads is a field that
        drifts. Adding a door to a flagged resource must clear its finding --
        which is only true if the check reads `paths` rather than a hardcoded
        list of known-bad names.
        """
        import dataclasses

        from sglang.srt.managers import cutover_participants as cp

        before = release_path_conformance("pp")
        self.assertTrue(any("kvso_host_region" in f for f in before))

        patched = tuple(
            dataclasses.replace(r, paths=(ReleasePath.SEAM,))
            if r.name == "kvso_host_region"
            else r
            for r in cp.HELD_RESOURCES
        )
        original = cp.HELD_RESOURCES
        cp.HELD_RESOURCES = patched
        try:
            after = release_path_conformance("pp")
        finally:
            cp.HELD_RESOURCES = original

        self.assertFalse(
            any("kvso_host_region" in f for f in after),
            "giving the resource a seam door must clear its finding; if it "
            "does not, the check is not reading the declaration",
        )


class TestTheArmSaysIt902(unittest.TestCase):
    """The check must run where a target layout is CHOSEN, or it is a desk
    artefact. #902's whole claim is that the class stops being discovered by
    windows, and that only holds if the arm consults the declaration."""

    def test_the_arm_names_undeclared_and_unreachable_holders(self):

        from sglang.srt.managers import phase_policy

        phase_policy._RELEASE_CONFORMANCE_SAID.discard("pp_to_tp")
        with self.assertLogs("sglang.srt.managers.phase_policy", level="WARNING") as cm:
            phase_policy.note_release_path_conformance("pp_to_tp", strict=True)
        blob = "\n".join(cm.output)
        self.assertIn("[#902] RELEASE-PATH CONFORMANCE", blob)
        self.assertIn("mamba_anchor_pin", blob)
        self.assertIn("kvso_host_region", blob)

    def test_it_says_it_once_per_direction(self):
        """The population is static; repeating it every arm trains readers to
        skip the line."""
        from sglang.srt.managers import phase_policy

        phase_policy._RELEASE_CONFORMANCE_SAID.discard("pp_to_tp")
        with self.assertLogs("sglang.srt.managers.phase_policy", level="WARNING"):
            phase_policy.note_release_path_conformance("pp_to_tp", strict=True)
        with self.assertNoLogs("sglang.srt.managers.phase_policy", level="WARNING"):
            phase_policy.note_release_path_conformance("pp_to_tp", strict=True)


if __name__ == "__main__":
    unittest.main()
