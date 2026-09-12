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
(``W92 Weg2CoverageNoObservation``) and never a number.  Three of the tests
below are that refusal, driven one way each.

THE TALLY IS THE SECOND GUARD.  ``executed + unexecuted == executable`` is an
identity, not a hope: if it does not hold, either the source drifted between
the boot and this ingest (the dump is keyed to a sha256 per module for exactly
that reason) or the line analysis disagrees with the tracer.  Either way the
numbers are not comparable and the ingest REFUSES (``W93
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
            lc.arm(directory=d, group="P", rank=1, boot_token="T")
            self.assertTrue(lc.enabled())
            # drive one allowlisted module INSIDE the leg bracket -- outside it
            # the tracer is deliberately off (review MF-4).
            from sglang.srt.weg2 import xchg_bounce as xb

            lc.begin_leg("leg1")
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
            lc.arm(directory=d, group="D", rank=0, boot_token="T")
            from sglang.srt.weg2 import xchg_bounce as xb

            lc.begin_leg("leg1")
            xb.staging_bytes_per_card(4096)
            lc.note_leg_end("leg1")
            first = json.loads(
                open(os.path.join(d, "phase_coverage_D_rank0.json")).read()
            )
            lc.begin_leg("leg2")
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
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
            lc.begin_leg("leg1")
            lc.note_leg_end("leg1")
            blob = json.loads(
                open(os.path.join(d, "phase_coverage_P_rank0.json")).read()
            )
            self.assertIn("overhead_ms", blob)
            self.assertGreaterEqual(blob["overhead_ms"]["traced"], 0.0)
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
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
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
                    "boot_token": "T",
                    "group": group,
                    "rank": rank,
                    "legs": 1,
                    "pid": 1,
                    "dead": False,
                    "instrument": "test",
                    "overhead_ms": {"arm": 0.0, "save_total": 0.0, "saves": 1},
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
            [sys.executable, INGEST, *args, "--boot-token", "T"],
            env=env, capture_output=True, text=True,
        )

    def test_unexecuted_line_shape_and_percentage(self):
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        from sglang.srt.weg2 import lane_coverage as lc

        executable = lc.executable_lines(os.path.join(ROOT, rel))
        half = sorted(executable)[: len(executable) // 2]
        with tempfile.TemporaryDirectory() as d:
            self._dump(d, "P", 0, {rel: self._mod_entry(rel, half)})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT, "--expect", "P=1")
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
            out = self._run_ingest("--dump-dir", d, "--root", ROOT, "--expect", "P=2")
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            union = [
                ln for ln in out.stdout.splitlines()
                if ln.startswith("WEG2-COVERAGE UNION")
            ]
            self.assertTrue(union, out.stdout)
            self.assertTrue(any(rel in ln and "group=P" in ln for ln in union))

    def test_the_tally_holds_or_the_ingest_refuses(self):
        """executed + unexecuted == executable, or W93 and a non-zero exit."""
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        from sglang.srt.weg2 import lane_coverage as lc

        ex = sorted(lc.executable_lines(os.path.join(ROOT, rel)))
        with tempfile.TemporaryDirectory() as d:
            entry = self._mod_entry(rel, ex[:5])
            # a line that is NOT executable -- the tracer and the analysis
            # disagree, so the two denominators are not comparable
            entry["executed"] = sorted(set(entry["executed"]) | {max(ex) + 10_000})
            self._dump(d, "P", 0, {rel: entry})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT, "--expect", "P=1")
            self.assertNotEqual(out.returncode, 0, out.stdout)
            self.assertIn("W93 Weg2CoverageTallyRefused", out.stdout + out.stderr)

    def test_source_drift_between_boot_and_ingest_refuses(self):
        """The sha256 in the dump is the boot's source, not this checkout's."""
        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        with tempfile.TemporaryDirectory() as d:
            entry = self._mod_entry(rel, [])
            entry["sha256"] = "0" * 64
            entry["executed"] = [1]
            self._dump(d, "P", 0, {rel: entry})
            out = self._run_ingest("--dump-dir", d, "--root", ROOT, "--expect", "P=1")
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
            self.assertIn("W92 Weg2CoverageNoObservation", out.stdout)
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


class MF1_AnExpectedRankThatWroteNoDumpIsNamed(CustomTestCase):
    """A rank or a whole GROUP that never wrote is the #1329 form itself.

    The first version iterated the dumps the glob FOUND. There was no
    expectation list, so a group whose shadow hook returned early -- exactly
    what `slots=True` did for three boots -- produced no dump, and the report
    named it with no word at all. The reader got P's reading list and took it
    for the boot's.

    The expectation comes from the BOOT (the launcher writes the manifest, and
    `--expect` overrides it), never from the dumps, because a list derived
    from what was found can never notice what is missing.
    """

    def _run(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [sys.executable, INGEST, *args], env=env, capture_output=True, text=True
        )

    def _dump(self, d, group, rank, token="boot-T"):
        from sglang.srt.weg2 import lane_coverage as lc
        import hashlib

        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        raw = open(os.path.join(ROOT, rel), "rb").read()
        ex = sorted(lc.executable_lines(os.path.join(ROOT, rel)))
        with open(os.path.join(d, lc.dump_filename(rank, group)), "w") as fh:
            json.dump(
                {
                    "schema": lc.SCHEMA,
                    "boot_token": token,
                    "group": group,
                    "rank": rank,
                    "legs": 1,
                    "pid": 1,
                    "dead": False,
                    "instrument": "test",
                    "overhead_ms": {"arm": 0.0, "save_total": 0.0, "saves": 1},
                    "modules": {
                        rel: {
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "imported": True,
                            "imported_before_arm": False,
                            "executed": ex[:10],
                        }
                    },
                },
                fh,
            )

    def test_a_group_that_wrote_no_dump_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            self._dump(d, "P", 0)
            out = self._run(
                "--dump-dir", d, "--root", ROOT,
                "--expect", "P=1,D=1", "--boot-token", "boot-T",
            )
            missing = [
                ln
                for ln in out.stdout.splitlines()
                if "rank-wrote-no-dump" in ln and "group=D" in ln
            ]
            self.assertEqual(len(missing), 1, out.stdout)
            self.assertIn("W92 Weg2CoverageNoObservation", missing[0])

    def test_the_union_refuses_when_ranks_are_missing(self):
        """An intersection over 2 of 3 ranks is a SUPERSET, printed as a find.

        BOOT_QUEUE calls the union line "the work list for the next wall". Over
        an incomplete group it contains lines the absent rank executed, so it
        is not a subset of anything -- it has to refuse, not shrink quietly to
        `ranks=2`.
        """
        with tempfile.TemporaryDirectory() as d:
            self._dump(d, "P", 0)
            self._dump(d, "P", 1)
            out = self._run(
                "--dump-dir", d, "--root", ROOT,
                "--expect", "P=3", "--boot-token", "boot-T",
            )
            union = [ln for ln in out.stdout.splitlines() if "WEG2-COVERAGE UNION" in ln]
            self.assertTrue(union, out.stdout)
            self.assertTrue(
                all("REFUSED" in ln for ln in union),
                f"a union over an incomplete group must refuse: {union}",
            )

    def test_no_expectation_list_at_all_is_a_refusal(self):
        """MR1: an EMPTY dump directory printed nothing at all.

        The guarded case was the ABSENT directory; an empty one walked the
        `for path in dumps` loop zero times and exited 0 with no output --
        which is what a reader sees when no rank ever armed.
        """
        with tempfile.TemporaryDirectory() as d:
            out = self._run("--dump-dir", d, "--root", ROOT)
            self.assertTrue(out.stdout.strip(), "an empty dump dir printed NOTHING")
            self.assertIn("WEG2-COVERAGE NO-OBSERVATION", out.stdout)

    def test_the_launcher_manifest_supplies_the_expectation(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc.write_expect_manifest(d, "boot-T", {"P": 2})
            self._dump(d, "P", 0)
            out = self._run("--dump-dir", d, "--root", ROOT)
            self.assertIn("rank-wrote-no-dump", out.stdout)
            self.assertIn("rank=1", out.stdout)


class MF2_AStaleDumpIsNeverReadAsThisBoot(CustomTestCase):
    """The dump directory is #1292's, and #1292's is reused across boots.

    With MF-1 fixed, the silence about a missing rank becomes a refusal. With
    MF-2 unfixed, that silence is instead FILLED: yesterday's
    `phase_coverage_D_rank0.json`, on an unchanged source tree, passes the
    sha256 gate untouched and prints a full reading list for a group that
    never ran today. The two defects compose into the worst reading the tool
    can produce, which is why the boot token is not optional.
    """

    def _run(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "python")
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [sys.executable, INGEST, *args], env=env, capture_output=True, text=True
        )

    def test_the_blob_carries_a_boot_identity(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0, boot_token="boot-XYZ")
            lc.note_leg_end("leg1")
            blob = json.loads(open(os.path.join(d, "phase_coverage_P_rank0.json")).read())
            self.assertEqual(blob["boot_token"], "boot-XYZ")
            self.assertGreater(blob["armed_at_epoch"], 0)
            lc._reset_for_test()

    def test_a_dump_from_another_boot_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            MF1_AnExpectedRankThatWroteNoDumpIsNamed()._dump(d, "D", 0, token="boot-YESTERDAY")
            out = self._run(
                "--dump-dir", d, "--root", ROOT,
                "--expect", "D=1", "--boot-token", "boot-TODAY",
            )
            self.assertIn("STALE-DUMP", out.stdout)
            self.assertIn("boot-YESTERDAY", out.stdout)
            # and it must NOT be counted as the expected rank having reported
            self.assertIn("rank-wrote-no-dump", out.stdout)


class MF3_AnEmptyShaIsARefusalNotASkip(CustomTestCase):
    """`if dump_sha and dump_sha != tree_sha` -- an empty hash is falsy.

    The instrument wrote `""` on any OSError at arm time, so the one guard
    that ties a dump to the source it was taken on had an OFF state nobody
    could see. On THIS strand the tree moves between boot and ingest
    routinely; that guard is the only thing standing between a reader and
    line numbers from another checkout.
    """

    def test_an_empty_sha_refuses(self):
        from sglang.srt.weg2 import lane_coverage as lc

        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "phase_coverage_P_rank0.json"), "w") as fh:
                json.dump(
                    {
                        "schema": lc.SCHEMA, "boot_token": "T", "group": "P", "rank": 0,
                        "legs": 1, "pid": 1, "dead": False, "instrument": "test",
                        "overhead_ms": {"arm": 0.0, "save_total": 0.0, "saves": 1},
                        "modules": {
                            rel: {"sha256": "", "imported": True,
                                  "imported_before_arm": False, "executed": [1]}
                        },
                    },
                    fh,
                )
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "T"],
                env=env, capture_output=True, text=True,
            )
            self.assertIn("no-source-hash", out.stdout)
            unexec = [ln for ln in out.stdout.splitlines()
                      if ln.startswith("WEG2-COVERAGE UNEXECUTED") and rel in ln]
            self.assertEqual(unexec, [], "printed a reading list against an unverified source")


class MF4_TheTracerRunsOnlyInsideTheLegBracket(CustomTestCase):
    """The tracer cost is NOT bounded by the allowlist -- measured 4.04x.

    The build's own docstring said 2.60x "because the allowlist is eleven
    files and not the tree". That is the instrument-text-lies class: the
    allowlist bounds what is RECORDED, not what is PAID -- the trace callback
    fires for every Python call in the process. The reviewer measured
    non-allowlisted code at 4.04x and fully traced code at 9.89x.

    Since the arm sat in the first flip leg and `note_teardown` deliberately
    did NOT stop, the whole serving process ran under `sys.settrace` for the
    rest of the boot -- so every ms/round, prefill and decode figure of an
    armed boot was confounded, and the wall-clock ring deadlines
    (`ring_guard.py` 2.0 s / 20.0 s) were being eaten by up to 4x.
    """

    def test_the_tracer_is_off_between_legs(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
            self.assertIsNone(sys.gettrace(), "arm() installed a tracer outside a leg")
            lc.begin_leg("leg1")
            self.assertIsNotNone(sys.gettrace(), "begin_leg installed no tracer")
            lc.note_leg_end("leg1")
            self.assertIsNone(sys.gettrace(), "the tracer survived the leg end")
            lc.note_teardown()
            self.assertIsNone(sys.gettrace(), "the tracer survived teardown")
            lc._reset_for_test()

    def test_the_dump_carries_the_traced_window(self):
        """How long the tracer was ON, so the reader can price the confound."""
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
            lc.begin_leg("leg1")
            lc.note_leg_end("leg1")
            blob = json.loads(open(os.path.join(d, "phase_coverage_P_rank0.json")).read())
            self.assertIn("traced", blob["overhead_ms"])
            lc._reset_for_test()

    def test_arming_writes_a_dump_before_any_leg_runs(self):
        """MF-1's other half: the dump EXISTS the moment the rank has identity.

        A rank whose hook returns before the first leg still has to appear in
        the report, otherwise the expectation list has to carry the whole
        burden of noticing it.
        """
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="D", rank=2, boot_token="T")
            path = os.path.join(d, "phase_coverage_D_rank2.json")
            self.assertTrue(os.path.isfile(path), "arm() left no dump behind")
            self.assertEqual(json.loads(open(path).read())["legs"], 0)
            lc._reset_for_test()


class TheArtefactLinesAreNamedNotMixedIn(CustomTestCase):
    """MR2 / S-1: 2104 of 8892 lines (23.7 %) are guaranteed artefacts.

    The tracer goes in at the first leg, so a module already imported had its
    module-scope lines executed unobserved. The first version flagged that per
    MODULE (`imported_before_arm=1`) and told the reader to subtract "exactly
    those lines" -- which the reader cannot see. Measured on a real boot
    shape, 10 of 11 modules carry the flag, so the advice covered almost the
    whole report.

    Mutant MR2 (report those modules as 100 % executed) survived all 21 of the
    first round's tests, because not one of them ever set the flag.
    """

    def test_module_scope_lines_are_printed_separately(self):
        from sglang.srt.weg2 import lane_coverage as lc
        import hashlib

        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        raw = open(os.path.join(ROOT, rel), "rb").read()
        ex = sorted(lc.executable_lines(os.path.join(ROOT, rel)))
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "phase_coverage_P_rank0.json"), "w") as fh:
                json.dump(
                    {
                        "schema": lc.SCHEMA, "boot_token": "T", "group": "P", "rank": 0,
                        "legs": 1, "pid": 1, "dead": False, "instrument": "test",
                        "overhead_ms": {"arm": 0.0, "save_total": 0.0, "saves": 1},
                        "modules": {
                            rel: {"sha256": hashlib.sha256(raw).hexdigest(),
                                  "imported": True, "imported_before_arm": True,
                                  "executed": ex[:10]},
                        },
                    },
                    fh,
                )
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "T"],
                env=env, capture_output=True, text=True,
            )
            line = [ln for ln in out.stdout.splitlines()
                    if ln.startswith("WEG2-COVERAGE UNEXECUTED") and rel in ln]
            self.assertEqual(len(line), 1, out.stdout)
            self.assertIn("artefact_lines=[", line[0])
            m = re.search(r"artefact_n=(\d+)", line[0])
            self.assertIsNotNone(m, line[0])
            self.assertGreater(int(m.group(1)), 0,
                               "a module imported before the arm has module-scope "
                               "artefacts by construction; naming zero of them is MR2")

    def test_module_scope_lines_are_a_real_subset(self):
        from sglang.srt.weg2 import lane_coverage as lc

        src = os.path.join(ROOT, "python/sglang/srt/weg2/xchg_bounce.py")
        scope = lc.module_scope_lines(src)
        ex = lc.executable_lines(src)
        self.assertTrue(scope <= ex)
        self.assertGreater(len(scope), 0)


class AModuleWithNoRecordedLinesRefuses(CustomTestCase):
    """MR3: `executed: []` on a PRESENT module must refuse, not fall through."""

    def test_zero_recorded_lines_is_no_observation(self):
        from sglang.srt.weg2 import lane_coverage as lc
        import hashlib

        rel = "python/sglang/srt/weg2/xchg_bounce.py"
        raw = open(os.path.join(ROOT, rel), "rb").read()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "phase_coverage_P_rank0.json"), "w") as fh:
                json.dump(
                    {
                        "schema": lc.SCHEMA, "boot_token": "T", "group": "P", "rank": 0,
                        "legs": 1, "pid": 1, "dead": False, "instrument": "test",
                        "overhead_ms": {"arm": 0.0, "save_total": 0.0, "saves": 1},
                        "modules": {
                            rel: {"sha256": hashlib.sha256(raw).hexdigest(),
                                  "imported": True, "imported_before_arm": False,
                                  "executed": []},
                        },
                    },
                    fh,
                )
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "T"],
                env=env, capture_output=True, text=True,
            )
            noobs = [ln for ln in out.stdout.splitlines()
                     if "NO-OBSERVATION" in ln and rel in ln]
            self.assertEqual(len(noobs), 1, out.stdout)


class ADeadCollectorSaysSo(CustomTestCase):
    """S-4: the promise was "the ingest will say so"; the blob had no field.

    A collector that dies at leg 3 of 24 leaves leg 3's dump on disk. It
    parses, the modules are present, the tally holds -- and every line legs
    4-24 executed is printed as a wall.
    """

    def test_the_dump_records_the_death_and_the_ingest_prints_it(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
            lc.begin_leg("leg1")
            # the real path: the first internal failure marks dead AND flushes
            lc._mark_dead("test-injected", RuntimeError("boom"))
            blob = json.loads(open(os.path.join(d, "phase_coverage_P_rank0.json")).read())
            self.assertTrue(blob["dead"])
            self.assertIn("test-injected", blob["dead_where"])
            lc._reset_for_test()


class TheCollisionGuardIsTwoSided(CustomTestCase):
    """S-3: measured -- a second `Coverage.start()` BLINDS the first silently.

    coverage 7.15.2 raises nothing; the first object records nothing for the
    duration of the second and resumes after. A blind window reads as
    UNEXECUTED, i.e. as a wall that is not one. The first version guarded only
    "seam was here first", `seam_coverage.py` did not know `lane_coverage`
    existed, and not one of the 21 tests drove either direction.
    """

    def test_the_lane_arm_refuses_when_seam_coverage_is_armed(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            os.environ["SGLANG_SEAM_COVERAGE_DIR"] = d
            try:
                self.assertFalse(lc.arm(directory=d, group="P", rank=0, boot_token="T"))
                self.assertFalse(lc.enabled())
            finally:
                os.environ.pop("SGLANG_SEAM_COVERAGE_DIR", None)
                lc._reset_for_test()

    def test_seam_coverage_refuses_when_the_lane_is_armed(self):
        from sglang.srt.managers import seam_coverage as sc
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
            try:
                self.assertTrue(lc.enabled())
                self.assertTrue(
                    sc.lane_coverage_armed(),
                    "seam_coverage cannot see the lane instrument at all",
                )
            finally:
                lc._reset_for_test()

    def test_the_launcher_refuses_both_switches_before_spawning(self):
        """Pre-spawn, because post-spawn the boot is already paid for.

        The launcher pops its OWN variable but never `SGLANG_SEAM_COVERAGE_DIR`.
        A leftover in the operator's shell makes all six ranks refuse by name
        AFTER the spawn -- the window is spent, and the refusal is in the rank
        logs where the ingest report never looks.
        """
        from sglang.srt.weg2 import launcher

        with self.assertRaises(SystemExit):
            launcher.refuse_double_coverage_arm(
                {"SGLANG_SEAM_COVERAGE_DIR": "/tmp/x"}, xchg_coverage_diff=True
            )
        launcher.refuse_double_coverage_arm({}, xchg_coverage_diff=True)


class TheWiringPinsAreAstNotSubstring(CustomTestCase):
    """MR4 (#1341 class): a COMMENTED-OUT call still contains the token.

    Three of the first round's tests read source text and asked for a
    substring. `# wlc.note_leg_end(hook)` satisfies every one of them, so the
    mutant that unwires the instrument entirely survived. An AST carries no
    comments, so the same question asked of the tree is answered honestly.
    """

    @staticmethod
    def _calls_in(path, func_names):
        import ast

        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = getattr(f, "attr", None) or getattr(f, "id", None)
                if name in func_names:
                    found.add(name)
        return found

    def test_the_leg_hook_really_calls_the_instrument(self):
        p = os.path.join(
            ROOT, "python/sglang/srt/managers/scheduler_components/weight_updater.py"
        )
        self.assertEqual(
            self._calls_in(p, {"note_leg_end", "begin_leg", "arm"}),
            {"note_leg_end", "begin_leg", "arm"},
        )

    def test_the_teardown_really_calls_the_instrument(self):
        p = os.path.join(ROOT, "python/sglang/srt/weg2/weight_exchange_region.py")
        self.assertIn("note_teardown", self._calls_in(p, {"note_teardown"}))

    def test_build_env_pops_the_variable_when_off(self):
        """Driven, not grepped: call build_env and read the dict it returns."""
        from sglang.srt.weg2 import launcher

        os.environ["SGLANG_WEG2_LANE_COVERAGE_DIR"] = "/tmp/inherited"
        os.environ["SGLANG_WEG2_LANE_COVERAGE_TOKEN"] = "inherited"
        try:
            env = launcher.build_env(
                tree="/t", venv="/v", cvd="", store_dir="/s", debug_hold=False,
                tag="t", group="P", lane_coverage_dir="", lane_coverage_token="",
            )
            self.assertNotIn("SGLANG_WEG2_LANE_COVERAGE_DIR", env)
            self.assertNotIn("SGLANG_WEG2_LANE_COVERAGE_TOKEN", env)
            env_on = launcher.build_env(
                tree="/t", venv="/v", cvd="", store_dir="/s", debug_hold=False,
                tag="t", group="P", lane_coverage_dir="/d", lane_coverage_token="tok",
            )
            self.assertEqual(env_on["SGLANG_WEG2_LANE_COVERAGE_DIR"], "/d")
            self.assertEqual(env_on["SGLANG_WEG2_LANE_COVERAGE_TOKEN"], "tok")
        finally:
            os.environ.pop("SGLANG_WEG2_LANE_COVERAGE_DIR", None)
            os.environ.pop("SGLANG_WEG2_LANE_COVERAGE_TOKEN", None)


class TheAllowlistIsSplitByProcessRole(CustomTestCase):
    """S-2: eleven modules applied to every dump = ~16 meaningless W92s a boot.

    `launcher.py` is never imported in a rank, so it refused on all six rank
    dumps; the ten rank modules are not imported in the launcher, so they
    refused on its one. Sixteen refusals per boot that mean nothing by
    construction -- while the one refusal that WOULD mean something (a rank
    that never wrote) was not printed at all. That is the indicator law
    inverted into noise.
    """

    def test_the_two_roles_are_disjoint_and_cover_the_allowlist(self):
        from sglang.srt.weg2 import lane_coverage as lc

        self.assertEqual(set(lc.RANK_MODULES) | set(lc.LAUNCHER_MODULES),
                         set(lc.ALLOWLIST))
        self.assertEqual(set(lc.RANK_MODULES) & set(lc.LAUNCHER_MODULES), set())
        self.assertIn("python/sglang/srt/weg2/launcher.py", lc.LAUNCHER_MODULES)
        self.assertNotIn("python/sglang/srt/weg2/launcher.py", lc.RANK_MODULES)

    def test_a_rank_dump_is_not_asked_about_launcher_modules(self):
        with tempfile.TemporaryDirectory() as d:
            MF1_AnExpectedRankThatWroteNoDumpIsNamed()._dump(d, "P", 0)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "boot-T"],
                env=env, capture_output=True, text=True,
            )
            self.assertNotIn("module=python/sglang/srt/weg2/launcher.py", out.stdout)


class TheDumpLineCarriesWhatItsDocstringPromises(CustomTestCase):
    """S-6, instrument-text-lies class B: the tracer factor was never on it.

    The docstring said all three numbers are re-printed per boot. Two are;
    the tracer factor cannot be, because nothing at boot measures the untraced
    counterpart. And `save_total_ms` is cumulative, so "56 ms per leg" was a
    division the reader had to do.
    """

    def test_the_line_carries_a_per_leg_figure(self):
        from sglang.srt.weg2 import lane_coverage as lc

        with tempfile.TemporaryDirectory() as d:
            lc._reset_for_test()
            lc.arm(directory=d, group="P", rank=0, boot_token="T")
            lc.begin_leg("l1")
            lc.note_leg_end("l1")
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "T"],
                env=env, capture_output=True, text=True,
            )
            dump = [ln for ln in out.stdout.splitlines() if ln.startswith("WEG2-COVERAGE DUMP")]
            self.assertTrue(dump, out.stdout)
            self.assertIn("save_per_leg_ms=", dump[0])
            self.assertIn("traced_ms=", dump[0])
            self.assertIn("timings=CONFOUNDED", dump[0])
            lc._reset_for_test()

    def test_the_docstring_no_longer_claims_the_tracer_factor_is_printed(self):
        src = open(os.path.join(ROOT, "python/sglang/srt/weg2/lane_coverage.py")).read()
        self.assertNotIn("every one of these three numbers is re-printed", src)


class AMissingExpectedRankEscalatesTheExitCode(CustomTestCase):
    """N-1: a caller that reads only ``$?`` must not read green over silence.

    Every other NO-OBSERVATION is a finding about a rank that DID report and
    stays rc=0 unless ``--strict``. A rank the boot promised and did not
    deliver is different in kind: it is the acceptance criterion failing, and
    the acceptance line in BOOT_QUEUE is written to be read by a person AND by
    a script.
    """

    def test_a_missing_expected_rank_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "T"],
                env=env, capture_output=True, text=True,
            )
            self.assertIn("rank-wrote-no-dump", out.stdout)
            self.assertNotEqual(out.returncode, 0, out.stdout)

    def test_a_complete_boot_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            MF1_AnExpectedRankThatWroteNoDumpIsNamed()._dump(d, "P", 0)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "python")
            env["CUDA_VISIBLE_DEVICES"] = ""
            out = subprocess.run(
                [sys.executable, INGEST, "--dump-dir", d, "--root", ROOT,
                 "--expect", "P=1", "--boot-token", "boot-T"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(out.returncode, 0, out.stdout)
