# SPDX-License-Identifier: Apache-2.0
"""#1395 -- the boot belongs in the artefact's IDENTITY, not only its content.

THE DEFECT, measured the same night it was found. #1348's own dump carries
``boot_token`` inside the JSON (precisely so a READER can detect a stale
dump -- see ``test_weg2_lane_coverage_1348.py``'s MF2 class) -- but the
WRITE path never consulted it: ``dump_filename(rank, group)`` derives
``phase_coverage_{group}_rank{N}.json`` with no boot identity at all, so
two boots of the SAME group/rank sharing one directory (#1292's shared,
reused evidence directory) write the identical filename and the later one
silently wins.

That is not a hypothetical: BOOT_weg2xsn31_0913_4.md VERSUCH 8 (2026-09-14,
window 2thy75) wrote phase_coverage_{P,D}_rank{0,1,2}.json (its own P.log
proves it: "lane_coverage: final dump .../phase_coverage_P_rank0.json
legs=0" at 03:21:36Z) -- and by the time the #1389 execution ratchet went
looking for them, they were gone, silently overwritten by a later,
unrelated boot on the same shared box using the same directory. #1292's own
collision guard (``W18 Weg2PhaseFootprintCollision``, ``mem_ledger/
activation_probe.py``) is precisely this defect's precedent: it refuses on
a DIFFERING profile digest, but two boots of the SAME form have the SAME
digest, so W18 never fires for them -- which is exactly the case that cost
tonight's evidence.

THE FIX IS AT THE IDENTITY, NEVER AT CLEANUP (the standing
SHM-RESIDUE-NUR-PER-HALTER rule's sibling for this instrument): two boots
of the same form must stop competing for one name, not have their leftovers
tidied after the fact. Every dump this module writes -- the rank dumps
(:func:`lane_coverage.arm`) and the expectation manifest
(:func:`lane_coverage.write_expect_manifest`) -- now lives under a
per-boot subdirectory named from ``boot_token`` itself
(:func:`lane_coverage._boot_subdir`), so a reader sees which boot an
artefact belongs to FROM ITS PATH, without opening the file. A defense in
depth (``W106 Weg2LaneCoverageCollision``, named text in a logged refusal,
matching #1292's own W18 shape and this module's OWN
"never raise at the caller" discipline) additionally refuses a write that
would still clobber an existing file carrying a DIFFERENT boot_token --
keyed on the boot, which is exactly the axis #1292's own digest-keyed W18
cannot see.
"""

import json
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase


class TheBootSubdirSanitisesTheToken(CustomTestCase):
    def test_colons_become_underscores(self):
        from sglang.srt.weg2 import lane_coverage as lc

        self.assertEqual(
            lc._boot_subdir("weg2xsn31:1789355854:1898898"),
            "weg2xsn31_1789355854_1898898",
        )

    def test_an_empty_token_gets_a_named_fallback_not_the_bare_directory(self):
        """An empty token collapsing to the bare directory would reintroduce
        the exact collision this function exists to remove, just for
        whichever caller forgot to pass one."""
        from sglang.srt.weg2 import lane_coverage as lc

        got = lc._boot_subdir("")
        self.assertTrue(got)
        self.assertNotEqual(got, "")

    def test_path_separators_in_a_token_cannot_escape_the_subdirectory(self):
        from sglang.srt.weg2 import lane_coverage as lc

        got = lc._boot_subdir("a/b" + os.sep + "c:d")
        self.assertNotIn("/", got)
        self.assertNotIn(os.sep, got)


class TwoBootsOfTheSameFormNeverCollideOnDisk(CustomTestCase):
    """THE REQUIRED MUTANT'S OWN BASELINE. Two ``arm()`` calls of the
    IDENTICAL group/rank/directory, only ``boot_token`` differing, must
    leave BOTH dumps on disk afterward -- "nebeneinander liegen", the
    order's own acceptable outcome, and the one this test pins.

    Manually verified (not re-run automatically) against the PRE-fix
    shape, by temporarily monkeypatching ``_boot_subdir`` to always return
    ``""`` (collapsing to the bare directory, i.e. #1348's original
    behaviour): this test goes RED -- the second ``arm()`` overwrites the
    first dump, so only ONE boot_token survives on disk and
    ``test_both_dumps_survive_with_their_own_boot_token`` fails on the
    missing first file. Restored, diff empty.
    """

    def test_both_dumps_survive_with_their_own_boot_token(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            self.assertTrue(
                lc.arm(directory=d, group="P", rank=0, boot_token="weg2xsn31:1:1"))
            lc.note_leg_end("leg1")
            lc._reset_for_test()

            self.assertTrue(
                lc.arm(directory=d, group="P", rank=0, boot_token="weg2xsn31:2:2"))
            lc.note_leg_end("leg1")
            lc._reset_for_test()

            first = os.path.join(d, "weg2xsn31_1_1", "phase_coverage_P_rank0.json")
            second = os.path.join(d, "weg2xsn31_2_2", "phase_coverage_P_rank0.json")
            self.assertTrue(os.path.isfile(first),
                            "the FIRST boot's dump is gone -- the second boot "
                            "silently overwrote it, exactly the #1395 defect")
            self.assertTrue(os.path.isfile(second), "the second boot's dump "
                            "is missing too")
            self.assertEqual(
                json.loads(open(first).read())["boot_token"], "weg2xsn31:1:1")
            self.assertEqual(
                json.loads(open(second).read())["boot_token"], "weg2xsn31:2:2")

    def test_the_expect_manifest_survives_the_same_way(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            p1 = lc.write_expect_manifest(d, "weg2xsn31:1:1", {"P": 3})
            p2 = lc.write_expect_manifest(d, "weg2xsn31:2:2", {"P": 3})
            self.assertNotEqual(p1, p2)
            self.assertTrue(os.path.isfile(p1))
            self.assertTrue(os.path.isfile(p2))
            self.assertEqual(json.loads(open(p1).read())["boot_token"], "weg2xsn31:1:1")
            self.assertEqual(json.loads(open(p2).read())["boot_token"], "weg2xsn31:2:2")


class TheDefenseInDepthRefusesARealCollision(CustomTestCase):
    """Even with per-boot subdirectories, a write that would STILL clobber
    an existing file carrying a DIFFERENT boot_token must refuse -- keyed
    on the boot (W106), the axis #1292's own digest-keyed W18 cannot see
    (two boots of the SAME form share one digest)."""

    def test_arm_refuses_rather_than_overwrites_a_foreign_token(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            # Plant a dump at the EXACT path a boot_token of "T" would
            # resolve to, carrying a DIFFERENT token -- the shape that can
            # only arise if two distinct tokens sanitise to the same
            # subdirectory, or a caller passes a bare, un-namespaced
            # directory a second time.
            subdir = os.path.join(d, lc._boot_subdir("T"))
            os.makedirs(subdir, exist_ok=True)
            path = os.path.join(subdir, "phase_coverage_P_rank0.json")
            with open(path, "w") as fh:
                json.dump({"boot_token": "OTHER-BOOT", "modules": {}}, fh)

            armed = lc.arm(directory=d, group="P", rank=0, boot_token="T")
            self.assertFalse(armed, "arm() must refuse rather than clobber "
                             "a foreign boot's dump")
            self.assertFalse(lc.enabled())
            # THE FOREIGN FILE IS UNTOUCHED -- refuse means refuse, not
            # "refuse and then write anyway".
            self.assertEqual(json.loads(open(path).read())["boot_token"], "OTHER-BOOT")
            lc._reset_for_test()

    def test_write_expect_manifest_refuses_rather_than_overwrites_a_foreign_token(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            subdir = os.path.join(d, lc._boot_subdir("T"))
            os.makedirs(subdir, exist_ok=True)
            path = os.path.join(subdir, lc.EXPECT_FILENAME)
            with open(path, "w") as fh:
                json.dump({"boot_token": "OTHER-BOOT", "expect": {"P": 1}}, fh)

            got = lc.write_expect_manifest(d, "T", {"P": 3})
            self.assertEqual(got, "", "a collision must return the "
                             "empty-string refusal shape, not a path")
            self.assertEqual(
                json.loads(open(path).read())["boot_token"], "OTHER-BOOT",
                "the foreign manifest must be untouched")

    def test_the_same_token_rewriting_is_not_a_collision(self):
        """A rank re-arming for a SECOND time under the identical
        boot_token (a restart with the exact same identity, not a
        different boot) must not refuse -- only a DIFFERING token is a
        collision."""
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            self.assertTrue(lc.arm(directory=d, group="P", rank=0, boot_token="T"))
            lc.note_leg_end("leg1")
            lc._reset_for_test()
            self.assertTrue(
                lc.arm(directory=d, group="P", rank=0, boot_token="T"),
                "re-arming under the SAME token must not read as a "
                "collision with itself",
            )
            lc._reset_for_test()


if __name__ == "__main__":
    import unittest
    unittest.main()
