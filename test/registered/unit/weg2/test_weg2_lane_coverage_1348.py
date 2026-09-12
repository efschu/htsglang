# SPDX-License-Identifier: Apache-2.0
"""#1348 -- the line-coverage diff of the exchange lane, and its refusals.

THE QUESTION THIS INSTRUMENT ANSWERS is not "did the boot work" but **"what
did the boot NOT execute"**.  Every wall XSN6..XSN15 sat in a seam that had
never executed before the boot that hit it; the per-module list of UNEXECUTED
lines is therefore the list of the walls not yet found.  A boot that produces
no such list has to find them one window at a time, which is what the last ten
boots did.

WHY THE REFUSAL IS HALF THE FEATURE (indicator law).  A coverage instrument
that prints ``0 unexecuted`` when its trace file is missing is worse than no
instrument: the reader takes an ABSENCE OF OBSERVATION for a FULL SWEEP and
closes exactly the seam the instrument was built to open.  So the three ways
this can silently degrade -- the dump file is not there, the dump file is
empty, the module never appears in the dump -- each print a NAMED refusal
(``W90 Weg2CoverageNoObservation``) and never a number.  Three of the tests
below are that refusal, driven one way each.

THE TALLY IS THE SECOND GUARD.  ``executed + unexecuted == executable`` is an
identity, not a hope: if it does not hold, either the source drifted between
the boot and this ingest (the dump is keyed to a sha256 per module for exactly
that reason) or the line analysis disagrees with the tracer.  Either way the
numbers are not comparable and the ingest REFUSES (``W91
Weg2CoverageTallyRefused``) instead of reporting a difference of two
incompatible denominators.

THE OFF STATE MUST BE BYTE-IDENTICAL.  ``coverage.py``'s own numbers put line
tracing at 2x-10x, which is fine for a diagnostic boot and never acceptable as
a standing cost on the tree that also serves.  ``test_off_is_byte_identical``
holds the line: with the launcher flag absent, ``coverage`` is not imported,
no tracer is installed, and every entry point returns before it touches
anything.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "python", "sglang", "srt", "weg2")):
            return here
    raise AssertionError("could not locate the repo root from this test file")


ROOT = _repo_root()
INGEST = os.path.join(ROOT, "scripts", "weg2", "lane_coverage_diff.py")


class TheAllowlistNamesFilesThatExist(CustomTestCase):
    """Every allowlisted path resolves in THIS tree, or the instrument lies.

    MUTANT DIRECTION (danger): a module is renamed or deleted and the
    allowlist keeps the stale path.  Coverage then records nothing for it, the
    ingest prints a refusal for a file that is not missing but GONE, and a
    reader reads "not executed" for code that no longer exists.  The allowlist
    is data, so it can rot silently; this test is the only thing that makes
    the rot loud.
    """

    def test_every_allowlisted_module_exists(self):
        from sglang.srt.weg2 import lane_coverage as lc

        missing = [p for p in lc.ALLOWLIST if not os.path.isfile(os.path.join(ROOT, p))]
        self.assertEqual(missing, [], f"allowlisted but absent from the tree: {missing}")

    def test_the_allowlist_covers_the_exchange_lane(self):
        """The lane's own modules, named -- not "whatever weg2 contains".

        Enumerated rather than globbed: a glob would silently widen the
        instrument's cost (and its output) every time the package grows a
        module, and the whole point of an allowlist is that the cost is a
        decision somebody made.
        """
        from sglang.srt.weg2 import lane_coverage as lc

        for must in (
            "python/sglang/srt/weg2/weight_exchange.py",
            "python/sglang/srt/weg2/weight_exchange_shadow.py",
            "python/sglang/srt/weg2/weight_exchange_transport.py",
            "python/sglang/srt/weg2/weight_exchange_bounce.py",
            "python/sglang/srt/managers/weg2_memory_saver.py",
            "python/sglang/srt/weg2/launcher.py",
            "python/sglang/srt/weg2/host_ledger.py",
            "python/sglang/srt/weg2/xchg_bounce.py",
            "python/sglang/srt/weg2/ring_guard.py",
            "python/sglang/srt/managers/scheduler_components/weight_updater.py",
        ):
            self.assertIn(must, lc.ALLOWLIST)


class OffIsByteIdentical(CustomTestCase):
    """Unarmed, the instrument imports no coverage and installs no tracer.

    THIS TEST DRIVES ``arm()`` AND NOT ONLY THE INERT ENTRY POINTS, and that
    is the whole point of it rather than a detail.  The first version called
    ``enabled``/``note_leg_end``/``note_teardown`` only -- all three of which
    return on the ``_armed`` flag before they reach any decision -- so mutant
    M4 (``arm`` falling back to a directory of its own instead of returning
    False) SURVIVED it: nothing in the test ever entered the mutated function.
    The product path calls ``wlc.arm(group=..., rank=...)`` with no directory
    and relies on the variable being absent, so that is the call the off-state
    proof has to make.
    """

    def test_off_is_byte_identical(self):
        code = (
            "import sys, json\n"
            "from sglang.srt.weg2 import lane_coverage as lc\n"
            "assert not lc.enabled(), 'armed with no directory published'\n"
            "armed = lc.arm(group='P', rank=0)  # the product's own call\n"
            "assert armed is False, 'arm() armed itself with no directory'\n"
            "assert not lc.enabled(), 'enabled() after a refused arm'\n"
            "lc.note_leg_end('leg1')\n"
            "lc.note_teardown()\n"
            "print(json.dumps({'coverage': 'coverage' in sys.modules,\n"
            "                  'trace': sys.gettrace() is not None}))\n"
        )
        env = dict(os.environ)
        env.pop(
            "SGLANG_WEG2_LANE_COVERAGE_DIR", None
        )  # the launcher pops it too; never inherited
        env["PYTHONPATH"] = os.path.join(ROOT, "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertFalse(got["coverage"], "coverage.py was imported with the arm OFF")
        self.assertFalse(got["trace"], "a tracer was installed with the arm OFF")


class TheDumpNameMatches1292(CustomTestCase):
    """One dump directory, two process groups -- the #1292 collision again.

    P and D are independently launched, share the dump directory, and rank is
    only unique WITHIN a group.  #1292 already paid for this once with the
    footprint dumps (P booted second and silently overwrote D's file).  This
    instrument takes the SAME filename derivation rather than a second one,
    because a second derivation is the thing that drifts.
    """

    def test_filename_carries_group_and_rank(self):
        from sglang.srt.weg2 import lane_coverage as lc

        self.assertEqual(lc.dump_filename(0, "P"), "phase_coverage_P_rank0.json")
        self.assertEqual(lc.dump_filename(2, "D"), "phase_coverage_D_rank2.json")

    def test_no_group_degrades_to_the_ungrouped_shape(self):
        from sglang.srt.weg2 import lane_coverage as lc

        self.assertEqual(lc.dump_filename(1, ""), "phase_coverage_rank1.json")


class ArmingWritesAtEveryLegEnd(CustomTestCase):
    """Armed, a leg end leaves a readable per-rank dump behind.

    WRITTEN PER LEG, not once at exit, for the reason ``seam_coverage`` gives
    for its own checkpoint: a rank that dies between two legs -- deadman, OOM,
    a wall -- must still leave the legs it COMPLETED on disk.  A dump that only
    exists after a clean shutdown is absent exactly on the boots this
    instrument was built for.
    """

    def test_leg_end_writes_a_dump_with_executed_lines(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=1)
            self.assertTrue(lc.enabled())
            # drive one allowlisted module so there is something to record
            from sglang.srt.weg2 import xchg_bounce as xb

            xb.staging_bytes_per_card(4096)
            lc.note_leg_end("leg1")
            path = os.path.join(d, "phase_coverage_P_rank1.json")
            self.assertTrue(os.path.isfile(path), f"no dump at {path}")
            blob = json.loads(open(path).read())
            self.assertEqual(blob["group"], "P")
            self.assertEqual(blob["rank"], 1)
            self.assertEqual(blob["legs"], 1)
            mod = "python/sglang/srt/weg2/xchg_bounce.py"
            self.assertIn(mod, blob["modules"])
            self.assertTrue(
                blob["modules"][mod]["executed"],
                "the driven module recorded no executed lines",
            )
            self.assertTrue(blob["modules"][mod]["sha256"])
            lc.note_teardown()
            lc._reset_for_test()

    def test_a_second_leg_merges_rather_than_replaces(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="D", rank=0)
            from sglang.srt.weg2 import xchg_bounce as xb

            xb.staging_bytes_per_card(4096)
            lc.note_leg_end("leg1")
            first = json.loads(
                open(os.path.join(d, "phase_coverage_D_rank0.json")).read()
            )
            xb.bounce_terms(
                bytes_per_direction=1 << 30,
                n_layers=48,
                widest_layer_bytes=1 << 20,
                pairs=3,
            )
            lc.note_leg_end("leg2")
            second = json.loads(
                open(os.path.join(d, "phase_coverage_D_rank0.json")).read()
            )
            self.assertEqual(second["legs"], 2)
            mod = "python/sglang/srt/weg2/xchg_bounce.py"
            self.assertTrue(
                set(first["modules"][mod]["executed"])
                <= set(second["modules"][mod]["executed"]),
                "leg 2 lost lines leg 1 had recorded -- this is a merge, not a replace",
            )
            lc.note_teardown()
            lc._reset_for_test()

    def test_the_overhead_is_measured_and_carried_in_the_dump(self):
        """Measured, never estimated -- the dump carries its own price.

        A diagnostic that cannot say what it cost cannot be argued about
        before the next boot, and "coverage.py is 2x-10x" is somebody else's
        measurement of somebody else's workload.
        """
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0)
            lc.note_leg_end("leg1")
            blob = json.loads(
                open(os.path.join(d, "phase_coverage_P_rank0.json")).read()
            )
            self.assertIn("overhead_ms", blob)
            self.assertGreaterEqual(blob["overhead_ms"]["arm"], 0.0)
            self.assertGreaterEqual(blob["overhead_ms"]["save_total"], 0.0)
            lc.note_teardown()
            lc._reset_for_test()


class TheInstrumentNeverKillsTheFlip(CustomTestCase):
    """First internal failure: log once, go dead, never raise at the caller.

    Mirrors ``seam_coverage``'s guard discipline verbatim, for the same
    reason: a half-working collector that keeps going produces a dump that
    LOOKS complete and is not, and this instrument's whole output is a claim
    about completeness.
    """

    def test_a_broken_save_disables_rather_than_raises(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0)
            lc._data_path = os.path.join(d, "nonexistent-subdir", "x", "y.json")
            lc.note_leg_end("leg1")  # must not raise
            self.assertTrue(lc._dead)
            lc.note_leg_end("leg2")  # still inert, still silent
            lc._reset_for_test()


class TheIngestPrintsUnexecutedLines(CustomTestCase):
    """The product line: per module, per rank, the lines that never ran."""

    def _dump(self, d, group, rank, modules):
        from sglang.srt.weg2 import lane_coverage as lc

        path = os.path.join(d, lc.dump_filename(rank, group))
        with open(path, "w") as fh:
            json.dump(
                {
                    "schema": lc.SCHEMA,
                    "group": group,
                    "rank": rank,
                    "legs": 1,
                    "pid": 1,
                    "instrument": "test",
                    "overhead_ms": {"arm": 0.0, "save_total": 0.0},
                    "modules": modules,
                },
                fh,
            )
        return path

    def _mod_entry(self, rel, executed):
        import hashlib

        raw = open(os.path.join(ROOT, rel), "rb").read()
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "imported": True,
            "imported_before_arm": False,
            "executed": sorted(executed),
        }

    def _run_ingest(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [sys.executable, INGEST, *args], env=env, capture_output=True, text=True
        )

    def test_unexecuted_line_shape_and_percentage(self):
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        from sglang.srt.weg2 import lane_coverage as lc

        executable = lc.executable_lines(os.path.join(ROOT, rel))
        half = sorted(executable)[: len(executable) // 2]
        with tempfile.TemporaryDirectory() as d:
            self._dump(d, "P", 0, {rel: self._mod_entry(rel, half)})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            line = [
                ln
                for ln in out.stdout.splitlines()
                if ln.startswith("WEG2-COVERAGE UNEXECUTED") and rel in ln
            ]
            self.assertEqual(len(line), 1, out.stdout)
            m = re.match(
                r"^WEG2-COVERAGE UNEXECUTED module=(\S+) rank=(\d+) "
                r"lines=\[(.*)\] executed_pct=([0-9.]+)",
                line[0],
            )
            self.assertIsNotNone(m, line[0])
            self.assertEqual(m.group(1), rel)
            self.assertEqual(m.group(2), "0")
            self.assertAlmostEqual(float(m.group(4)), 100.0 * len(half) / len(executable), places=1)

    def test_the_group_union_is_printed_per_module(self):
        """Three ranks, one union: a line unexecuted on ALL of them is the find.

        A line missed by one rank and hit by another is a rank-local path
        (which is information, and stays in the per-rank lines); a line missed
        by every rank in the group is a seam nothing in this boot reached, and
        that is what the union names.
        """
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        from sglang.srt.weg2 import lane_coverage as lc

        ex = sorted(lc.executable_lines(os.path.join(ROOT, rel)))
        with tempfile.TemporaryDirectory() as d:
            self._dump(d, "P", 0, {rel: self._mod_entry(rel, ex[:10])})
            self._dump(d, "P", 1, {rel: self._mod_entry(rel, ex[5:15])})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT)
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            union = [
                ln for ln in out.stdout.splitlines()
                if ln.startswith("WEG2-COVERAGE UNION")
            ]
            self.assertTrue(union, out.stdout)
            self.assertTrue(any(rel in ln and "group=P" in ln for ln in union))

    def test_the_tally_holds_or_the_ingest_refuses(self):
        """executed + unexecuted == executable, or W91 and a non-zero exit."""
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        from sglang.srt.weg2 import lane_coverage as lc

        ex = sorted(lc.executable_lines(os.path.join(ROOT, rel)))
        with tempfile.TemporaryDirectory() as d:
            entry = self._mod_entry(rel, ex[:5])
            # a line that is NOT executable -- the tracer and the analysis
            # disagree, so the two denominators are not comparable
            entry["executed"] = sorted(set(entry["executed"]) | {max(ex) + 10_000})
            self._dump(d, "P", 0, {rel: entry})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT)
            self.assertNotEqual(out.returncode, 0, out.stdout)
            self.assertIn("W91 Weg2CoverageTallyRefused", out.stdout + out.stderr)

    def test_source_drift_between_boot_and_ingest_refuses(self):
        """The sha256 in the dump is the boot's source, not this checkout's."""
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        with tempfile.TemporaryDirectory() as d:
            entry = self._mod_entry(rel, [])
            entry["sha256"] = "0" * 64
            entry["executed"] = [1]
            self._dump(d, "P", 0, {rel: entry})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT)
            self.assertIn("SOURCE-DRIFT", out.stdout + out.stderr)


class NoObservationIsNeverZero(CustomTestCase):
    """The indicator law, driven three ways.

    Each of these is a way the instrument can produce NOTHING while looking as
    if it produced a clean sweep.  The only acceptable output is a named
    refusal; ``0 unexecuted`` on any of them would close a seam that was never
    opened.
    """

    def _run_ingest(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [sys.executable, INGEST, *args], env=env, capture_output=True, text=True
        )

    def test_absent_dump_directory_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run_ingest("--dump-dir", os.path.join(d, "nope"), "--root", ROOT)
            self.assertIn("WEG2-COVERAGE NO-OBSERVATION", out.stdout)
            self.assertIn("W90 Weg2CoverageNoObservation", out.stdout)
            self.assertNotIn("unexecuted=0", out.stdout)

    def test_empty_dump_file_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "phase_coverage_P_rank0.json"), "w") as fh:
                fh.write("")
            out = self._run_ingest("--dump-dir", d, "--root", ROOT)
            self.assertIn("WEG2-COVERAGE NO-OBSERVATION", out.stdout)
            self.assertIn("reason=", out.stdout)

    def test_a_module_absent_from_the_dump_refuses_per_module(self):
        """The dump is real; ONE allowlisted module is missing from it.

        This is the shape that matters most, because the rest of the report
        looks perfectly healthy beside it -- the reader has to be told, per
        module, that this one was not observed rather than not executed.
        """
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        from sglang.srt.weg2 import lane_coverage as lc
        import hashlib

        raw = open(os.path.join(ROOT, rel), "rb").read()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "phase_coverage_P_rank0.json"), "w") as fh:
                json.dump(
                    {
                        "schema": lc.SCHEMA,
                        "group": "P",
                        "rank": 0,
                        "legs": 1,
                        "pid": 1,
                        "instrument": "test",
                        "overhead_ms": {"arm": 0.0, "save_total": 0.0},
                        "modules": {
                            rel: {
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "imported": True,
                                "imported_before_arm": False,
                                "executed": [1],
                            }
                        },
                    },
                    fh,
                )
            out = self._run_ingest("--dump-dir", d, "--root", ROOT)
            other = "python/sglang/srt/weg2/ring_guard.py"
            noobs = [
                ln
                for ln in out.stdout.splitlines()
                if ln.startswith("WEG2-COVERAGE NO-OBSERVATION") and other in ln
            ]
            self.assertEqual(len(noobs), 1, out.stdout)
            self.assertIn("reason=module-absent-from-dump", noobs[0])


class TheLauncherFlagIsOffByDefault(CustomTestCase):
    """Armed by a launcher FLAG, never by an ambient environment variable.

    The env variable this instrument reads is LAUNCHER OUTPUT and is POPPED
    when the flag is absent -- the same discipline ``build_env`` already
    applies to ``SGLANG_WEG2_GROUP`` and the host-ring family, and for the
    same reason: a value inherited from the operator's own shell must never be
    able to arm a 2x-10x tracer on an acceptance boot.
    """

    @staticmethod
    def _flag_block() -> str:
        """The ``add_argument`` call for the flag, as source text.

        Sliced by position rather than matched with a balanced-paren regex:
        the help text itself contains parentheses, and a regex that stops at
        the first ``)`` reads a truncated block and then reports a missing
        ``action=`` that is right there. That near-miss is the reason this is
        a helper and not an inline pattern.
        """
        src = open(os.path.join(ROOT, "python/sglang/srt/weg2/launcher.py")).read()
        i = src.index('add_argument(\n        "--xchg-coverage-diff"')
        return src[i : i + 1600]

    def test_the_flag_exists_and_defaults_off(self):
        block = self._flag_block()
        self.assertIn('action="store_true"', block)
        self.assertNotIn("default=True", block)

    def test_the_help_text_names_the_overhead(self):
        block = self._flag_block().lower()
        self.assertTrue(
            "overhead" in block or "slowdown" in block,
            "the help text must price the flag: a reader arming it deserves "
            "the number",
        )

    def test_build_env_pops_the_variable_when_the_flag_is_off(self):
        src = open(os.path.join(ROOT, "python/sglang/srt/weg2/launcher.py")).read()
        self.assertIn("SGLANG_WEG2_LANE_COVERAGE_DIR", src)
        self.assertIn('env.pop("SGLANG_WEG2_LANE_COVERAGE_DIR", None)', src)


class TheHooksAreWiredIntoTheProductPath(CustomTestCase):
    """Present-but-unwired is the expensive middle state (#859).

    A module that exists, imports and tests green but is called from nothing
    is indistinguishable from a delivered feature in every subject-line or
    ancestry search.  These two greps are the wiring half of the proof; the
    execution smoke is the other half.
    """

    def test_the_leg_hook_calls_the_instrument(self):
        src = open(
            os.path.join(
                ROOT,
                "python/sglang/srt/managers/scheduler_components/weight_updater.py",
            )
        ).read()
        self.assertIn("lane_coverage", src)
        self.assertIn("note_leg_end", src)

    def test_the_teardown_path_calls_the_instrument(self):
        src = open(
            os.path.join(ROOT, "python/sglang/srt/weg2/weight_exchange_region.py")
        ).read()
        self.assertIn("note_teardown", src)


if __name__ == "__main__":
    unittest.main()
