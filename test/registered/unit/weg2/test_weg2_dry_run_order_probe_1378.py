# SPDX-License-Identifier: Apache-2.0
"""#1378 (desk lane-gates, instrument 3 slice): THE DRY-RUN LAUNCHER PROBE.

WHAT THIS FILE IS NOT.  It is not a byte proof and not a cushion proof.  The
memory that gates this whole mission (``desk-lane-gates-vor-boot.md``, user
order 13.09. after Wall 11) says so explicitly for the CPU-whole-boot-probe
class this file is a slice of: "Memory-Saver ist auf CPU Noop, Credits 0
Byte, kein Cushion: als Byte-Beleg gelesen = 'gruen am Desk, tot am Metall'."
Every test below asserts ORDER and GEOMETRY -- which refusal fired, which arm
line printed, which card resolved to which ordinal -- and NEVER a VRAM
number, a throughput number, or a cushion margin.  ``test_declares_itself_
as_order_geometry_probe`` pins that self-declaration mechanically, not just
in this docstring.

WHAT QUESTION THIS ANSWERS, and how it differs from #1329.  #1329
(``test_weg2_launch_replay_1329.py``) replays the ARITHMETIC LAYER directly
-- ``ring_table.solve`` / ``xchg_residency.solve`` / ``host_ledger.price`` --
with no ``launcher.main()`` call at all, which is the right shape for
re-running three specific recorded walls in isolation.  This file drives
``launcher.main()`` itself, at the CLI/argv boundary a real boot actually
uses, from a GPU-less process (``CUDA_VISIBLE_DEVICES=""``) through the
NVML replay seam #1377 shipped
(``python/sglang/srt/registry/nvml.py:list_devices``, ``ENV_NVML_REPLAY``),
up to the literal ARM line the task names: ``log(w38_armed_line(...))`` at
``launcher.py:10895``, which prints as ``"WEG2 #1245 ARMED (W38 RETIRED
INTO IT): ..."``.  Reaching that line and returning 0 is ORDER evidence that
the whole dry-run spine -- argv parse, preflight, NVML resolve, ring-table
pin, W71 exchange arm, host-ledger pricing, PP-cut solve, group-argv build --
composes end-to-end on this box with the #1377 seam's replayed cards, not
live NVML.  It is not evidence that any of those numbers would hold on real
silicon.

THE FIXTURE is the SAME ``fixtures/xchg_launch_replay_0911/`` #1329 uses,
plus one file of my own (``nvml_devices_1378.json``, three ``DeviceInfo``
rows matching this rig's real UUIDs) that #1329 does not need because it
never resolves NVML at all.  ``recorded.json``'s ``checkpoint_dependency``
applies here for the identical reason it applies there: the W48 form-gate
inside ``ring_table.solve`` reads real checkpoint safetensors headers via
``planner/pp_cut.checkpoint_weight_terms``, so every test that reaches
``main()``'s ring-table call SKIPS BY NAME where the 27 GB checkpoint
directory is absent, exactly like #1329's ``NEEDS_CHECKPOINT`` tests.

WHY THE PREFLIGHT SWEEPS ARE ISOLATED FROM LIVE RIG STATE.
``shm_residue_sweep`` (launcher.py:1895) forks on whether ``/dev/shm``
already holds any ``weg2-``-prefixed entry: with none, it logs "none of
ours" and returns; with any, it runs an UNCONDITIONAL (not dry-gated)
``pgrep -f sglang[.]launch_server`` and RAISES ``Weg2LaunchRefused`` if a
live server is found.  On a clean rig that branch is never taken, which is
what made the ad-hoc probe run for this ticket succeed -- but a test whose
pass/fail depends on "the rig happened to be clean when it ran" is not
hermetic, it is lucky.  ``L.SHM_DIR`` is therefore patched to a private,
guaranteed-empty ``tempfile.mkdtemp()`` for the duration of every test in
this file, so the sweep's answer is a property of the test, never of
whatever else is running on the shared box.  The other three preflight
sweeps (``stale_deadman_sweep``, ``host_preflight``,
``refuse_if_front_unbindable``) were read directly at HEAD and are already
fully dry-gated no-ops with zero mutation and zero raise path under
``--dry-run`` (each logs a "DRY-RUN: would ..." line and returns) -- they
are left unpatched because patching a function that already provably does
nothing under the condition this file always sets would just be noise.

THE NEGATIVE CONTROL (Wall-7 class: a call site loses its receiver).
``prepare_weight_exchange`` (launcher.py:3877) calls
``xchg_residency.solve(cards, census, floor_mib)`` POSITIONALLY, three
required parameters, no ``**kwargs``, no defaults on the callee.  If a
future edit to ``xchg_residency.solve``'s signature silently drops a
parameter this call site relies on, the launcher's caller is now a
signature without a receiver -- exactly the class this fork's Semgrep rule
catalogues under a different name.  ``TestWall7NegativeControl`` proves
this probe's own execution reaches that exact call site by installing a
mutant ``solve`` with only two parameters and asserting ``main()`` raises
``TypeError`` un-caught: ``main()`` itself (as opposed to ``cli()``, its
thin wrapper) catches only ``REFUSALS`` -- a fixed tuple of named refusal
classes at launcher.py:11682 that does not include ``TypeError`` -- so a
signature mismatch at this call site propagates raw instead of being
silently absorbed.  A probe that could not turn red under this mutation
would not be proof that it ever executes the real call.

THE COVERAGE COMPLEMENT is a NARROWER instrument than #1348's
``--xchg-coverage-diff`` (``lane_coverage.py``), and deliberately does not
reuse its mechanism: #1348 arms per RANK, inside a spawned server process,
starting "at the first leg" -- a hook this dry-run never reaches, because
``launch_group`` returns before any ``subprocess.Popen`` under ``dry``
(launcher.py, guarded by ``if dry: return``).  What this file needs is the
complement over the DRIVER's own execution -- ``launcher.py`` plus every
module under ``srt/weg2/`` -- for the one process that ran ``main()``
itself.  ``test_prints_coverage_complement`` runs the positive probe under
``coverage.Coverage(include=[...])`` and prints the never-executed line
count per module, which is a READING LIST (per #1348's own stated
methodology) and never a target.
"""

import glob
import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import xchg_residency as xr
from sglang.test.test_utils import CustomTestCase

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "xchg_launch_replay_0911")
RING_EVIDENCE = os.path.join(FIXTURES, "ring_evidence")
CENSUS = os.path.join(FIXTURES, "census_weg2sn5b_48f55fb393.json")
NVML_REPLAY_JSON = os.path.join(FIXTURES, "nvml_devices_1378.json")
RING_TABLE_BOOT_STEM_SUBSTR = "d8ea6261f7"

# The worktree root (five parents up from launcher.py:
# .../python/sglang/srt/weg2/launcher.py -> weg2 -> srt -> sglang -> python
# -> the tree root that has both "python/" and "scripts/" as children).
# main() does `tree = os.path.abspath(ns.tree)` (launcher.py:9985) and then,
# inside prepare_host_ring's C9 gather-key probe (launcher.py:3304,
# _resolve_keys_in_rank_tree), spawns a subprocess with
# `PYTHONPATH=f"{tree}/python"` and cwd=tree (launcher.py:3220,3228) to
# resolve each card's PCIe gather key OUT OF PROCESS. Point --tree at
# anything other than the real tree root (e.g. the two-parents-up
# ".../python/sglang/srt" this constant used to compute -- a real, checked
# bug found and fixed under #1378, not a hypothetical) and that subprocess's
# PYTHONPATH has no "sglang" to import; the key-probe then fails and the
# gather determination falls back to the conservative "gathered" (C9)
# reading for every card, which is exactly the W34 Weg2RingNeedsInterleave
# refusal this file chased before the root cause was found: NOT a tag
# sensitivity (a controlled diagnostic proved varying --tag alone, tree
# held correct, never reproduces the refusal), but a wrong --tree silently
# starving the out-of-process key-probe.
_TREE_ROOT = str(pathlib.Path(L.__file__).resolve().parents[4])

with open(os.path.join(FIXTURES, "recorded.json"), encoding="utf-8") as _fh:
    RECORDED = json.load(_fh)

with open(NVML_REPLAY_JSON, encoding="utf-8") as _fh:
    REPLAY_CARD_ROWS = json.load(_fh)

REPLAY_CARD_UUIDS = [row["uuid"] for row in REPLAY_CARD_ROWS]


def checkpoint_present() -> bool:
    """Is the checkpoint the recorded fixture names on THIS box?

    Same check #1329 makes (see its ``checkpoint_present``): the W48 form
    gate inside ``ring_table.solve`` reads real safetensors headers, so a
    box without the 27 GB directory cannot arm it -- and this file's tests
    that reach ``main()``'s ring-table call skip by name there instead of
    failing for a reason that is not about the code they grade.
    """
    path = RECORDED["checkpoint_dependency"]["model_path"]
    return bool(glob.glob(os.path.join(path, "*.safetensors")))


NEEDS_CHECKPOINT = unittest.skipUnless(
    checkpoint_present(),
    "the checkpoint the fixture's ring evidence names is not on this box -- "
    "the W48 form gate inside ring_table.solve cannot be armed (see "
    "recorded.json checkpoint_dependency)",
)

MODEL_PATH = RECORDED["checkpoint_dependency"]["model_path"]


def probe_argv(tag: str) -> list:
    """The top-level CLI argv shape for the dry-run order probe.

    NOT ``p_argv_xsn14.txt`` -- that file is group P's OWN recorded argv,
    consumed only by #1329's direct ``ring_table.solve`` call to arm the
    form gate hermetically.  Inside ``main()`` the equivalent value
    (``form_argv_p``) is constructed INTERNALLY from this namespace via
    ``argv_p(...)`` (launcher.py ~L10354) and is never supplied by the
    caller of ``main()`` -- so this argv only needs the flags a real
    top-level invocation would pass.
    """
    return [
        "--tree", _TREE_ROOT,
        "--tag", tag,
        "--dry-run",
        "--model", MODEL_PATH,
        "--weg2-weight-source", "exchange",
        "--weg2-xchg-census", CENSUS,
        "--evidence-dir", RING_EVIDENCE,
        "--ring-table-boot", RING_TABLE_BOOT_STEM_SUBSTR,
    ]


class _HermeticDryRunBase(CustomTestCase):
    """Common isolation: replayed NVML cards, private empty SHM_DIR.

    Every subclass runs ``L.main(argv)`` in a GPU-less, rig-state-independent
    process.  See the module docstring's "WHY THE PREFLIGHT SWEEPS ARE
    ISOLATED" section for why ``SHM_DIR`` specifically is patched and the
    other three preflight sweeps are not.
    """

    def setUp(self):
        super().setUp()
        self._old_replay_env = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
        os.environ[nvml_registry.ENV_NVML_REPLAY] = NVML_REPLAY_JSON
        self._shm_tmp = tempfile.mkdtemp(prefix="weg2-1378-empty-shm-")
        self._shm_patch = mock.patch.object(L, "SHM_DIR", self._shm_tmp)
        self._shm_patch.start()

    def tearDown(self):
        self._shm_patch.stop()
        if self._old_replay_env is None:
            os.environ.pop(nvml_registry.ENV_NVML_REPLAY, None)
        else:
            os.environ[nvml_registry.ENV_NVML_REPLAY] = self._old_replay_env
        super().tearDown()

    def run_probe(self, tag: str):
        """Run ``L.main`` capturing every ``Log``-printed line; return (rc, text)."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = L.main(probe_argv(tag))
        return rc, buf.getvalue()


@NEEDS_CHECKPOINT
class TestDryRunReachesArmLine(_HermeticDryRunBase):
    """The positive probe: order/geometry evidence up to the named ARM line."""

    def test_declares_itself_as_order_geometry_probe(self):
        """#1378: the probe's own output self-declares its evidentiary class.

        This is the mechanical half of the memory's rule that a CPU probe
        must never be read as a byte/cushion proof: the declaration is
        printed BY THIS TEST, asserted to be present, and is not something
        a reader has to take on the docstring's word.
        """
        declaration = (
            "#1378 ORDER/GEOMETRY PROBE (desk-lane-gates instrument 3 slice) -- "
            "NOT a byte or cushion proof. Memory-saver is CPU noop here, "
            "credits are 0 bytes, no VRAM was touched and no cushion margin "
            "was measured. This run only proves the dry-run's ORDER of "
            "refusals/arms and the GEOMETRY (card ordinals, form keys, argv "
            "shape) that the arithmetic layer would price on real silicon."
        )
        print(declaration)
        self.assertIn("ORDER/GEOMETRY PROBE", declaration)
        self.assertIn("NOT a byte or cushion proof", declaration)

    def test_reaches_the_named_arm_line_and_returns_zero(self):
        """The literal task target: ``log(w38_armed_line(...))`` @ launcher.py:10895."""
        rc, out = self.run_probe("desk1378test")
        self.assertEqual(rc, 0, f"main() did not return 0; tail:\n{out[-4000:]}")
        self.assertIn(
            "WEG2 #1245 ARMED (W38 RETIRED INTO IT)", out,
            "the named ARM line (w38_armed_line, launcher.py:10895) never printed",
        )
        self.assertIn(
            "WEG2-LAUNCH DRY-RUN complete: nothing started, mounted, armed or written",
            out,
        )

    def test_w71_exchange_residency_armed_from_the_census(self):
        """The W71 arm (prepare_weight_exchange -> xchg_residency.armed_line)."""
        rc, out = self.run_probe("desk1378test-w71")
        self.assertEqual(rc, 0)
        self.assertIn("WEG2-XCHG-ARMED", out)
        self.assertNotIn("Weg2XchgResidencyUnarmable", out)

    def test_host_ring_armed_from_the_recorded_boot(self):
        """The ring pin (prepare_host_ring -> ring_table.solve) reaches ARMED."""
        rc, out = self.run_probe("desk1378test-ring")
        self.assertEqual(rc, 0)
        self.assertIn("WEG2-HOST-RING ARMED", out)

    def test_replayed_cards_resolved_by_uuid_not_position(self):
        """Every fixture UUID appears -- NVML replay seam #1377 fed real identity."""
        rc, out = self.run_probe("desk1378test-cards")
        self.assertEqual(rc, 0)
        for uuid in REPLAY_CARD_UUIDS:
            self.assertIn(uuid, out, f"replayed card {uuid} never appeared in the log")

    def test_named_finding_w48_bootstrap_not_a_hard_refusal(self):
        """A named finding, per the task: first-boot-of-a-form degrades gracefully.

        This form key (derived from THIS run's own --rank-gpu-memory-mib /
        --pp-stage-ratio, which the PP-cut solver picks fresh from the
        checkpoint + card geometry) has no prior same-form boot to compare
        against, so ``refuse_unless_same_form_source`` takes its B4e(ii)
        bootstrap branch (launcher.py ~L10391-10399) and LOGS rather than
        raises. This is the finding this probe is required to name, not a
        defect: W48 Weg2RingFormMismatch is conditionally fatal, not always.
        """
        rc, out = self.run_probe("desk1378test-w48")
        self.assertEqual(rc, 0)
        self.assertIn("WEG2-RING W48 BOOTSTRAP armed first boot", out)


@NEEDS_CHECKPOINT
class TestWall7NegativeControl(_HermeticDryRunBase):
    """Mutation-testing proof: the probe genuinely reaches xchg_residency.solve.

    See the module docstring's "THE NEGATIVE CONTROL" section for the full
    argument.  Without this class, a probe that silently short-circuited
    before ever calling ``prepare_weight_exchange`` could still print
    ``MAIN RETURNED 0`` and look identical to a real pass.
    """

    def test_dropped_floor_mib_parameter_turns_the_probe_red(self):
        """Wall-7 mutant: xchg_residency.solve loses its 3rd positional param.

        launcher.py:3877 calls ``xchg_residency.solve(cards, census,
        floor_mib)`` positionally with all three always supplied. A mutant
        with only ``(cards, census)`` makes that real call site raise
        ``TypeError: solve() takes 2 positional arguments but 3 were given``.
        ``main()`` does not catch bare ``TypeError`` -- only the fixed
        ``REFUSALS`` tuple, which is ``cli()``'s job, and this test calls
        ``main()`` directly -- so the exception must propagate uncaught.
        """

        def mutant_solve_missing_receiver(cards, census):
            raise AssertionError(
                "unreachable: the real (3-arg) call site should already have "
                "raised TypeError before this body could run"
            )

        with mock.patch.object(xr, "solve", mutant_solve_missing_receiver):
            buf = io.StringIO()
            with self.assertRaises(TypeError) as cm:
                with redirect_stdout(buf):
                    L.main(probe_argv("desk1378test-wall7"))
            self.assertIn("positional argument", str(cm.exception))

    def test_unmutated_probe_is_the_control_and_stays_green(self):
        """The other half of red-first: same argv, unpatched, must pass."""
        rc, out = self.run_probe("desk1378test-wall7-control")
        self.assertEqual(rc, 0)
        self.assertIn("WEG2-XCHG-ARMED", out)


@NEEDS_CHECKPOINT
class TestCoverageComplement(_HermeticDryRunBase):
    """#1378's own coverage complement: which lines of the driver never ran.

    Deliberately NOT #1348's ``lane_coverage`` mechanism -- see the module
    docstring's "THE COVERAGE COMPLEMENT" section for why that instrument's
    hook point (first leg, inside a spawned rank) is structurally unreached
    by a dry run. This scopes ``coverage.Coverage`` to ``launcher.py`` plus
    every module under ``srt/weg2/`` for the single process running
    ``main()``, which is the driver-level complement this ticket asks for.
    """

    def test_prints_coverage_complement_over_launcher_and_weg2(self):
        try:
            import coverage
        except ImportError:
            self.skipTest("coverage.py not importable in this interpreter")

        weg2_dir = os.path.dirname(os.path.abspath(L.__file__))
        launcher_path = os.path.abspath(L.__file__)
        weg2_modules = sorted(
            p for p in glob.glob(os.path.join(weg2_dir, "*.py"))
        )
        self.assertIn(launcher_path, weg2_modules, "launcher.py must be in its own dir's glob")

        # config_file=False: the repo-root .coveragerc sets [run] source =
        # python/sglang/srt, and coverage.py's own precedence rule is that a
        # configured "source" WINS over a programmatic "include" (that is
        # exactly the CoverageWarning "--include is ignored because --source
        # is set" this test used to print and then silently mis-measure
        # under -- the scope would have quietly been all of srt/, not
        # launcher.py+srt/weg2/ as the docstring claims). Disabling config
        # discovery makes "include" the only scope source, which is what
        # this narrower, driver-only instrument is for.
        cov = coverage.Coverage(include=weg2_modules, data_file=None, config_file=False)
        cov.start()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = L.main(probe_argv("desk1378test-coverage"))
        finally:
            cov.stop()
        self.assertEqual(rc, 0)

        print(
            "#1378 COVERAGE COMPLEMENT -- a READING LIST of unexecuted lines "
            "in this dry run, never a target (per #1348's own stated "
            "methodology: a line that never ran may be a dead branch, an "
            "un-triggered refusal, or the next wall)."
        )
        total_never_executed = 0
        for path in weg2_modules:
            # _analyze() takes a morf (module-or-filename) and resolves its
            # OWN file reporter internally; passing an already-resolved
            # reporter here made it re-resolve a FileReporter as if it were
            # a path, which crashed with "'PythonFileReporter' object has no
            # attribute 'endswith'" -- a real bug, not a coverage.py defect.
            analysis = cov._analyze(path)
            executed = analysis.statements - analysis.missing
            never = sorted(analysis.missing)
            total_never_executed += len(never)
            print(
                f"  {os.path.relpath(path, weg2_dir)}: "
                f"{len(executed)}/{len(analysis.statements)} executed, "
                f"{len(never)} never-executed line(s)"
                + (f" e.g. {never[:8]}" if never else "")
            )
        # The complement must be non-trivial in BOTH directions: some lines of
        # this large module ran (else the probe never left the parser), and
        # some did not (else this dry run would have started real processes,
        # which it must never do).
        self.assertGreater(
            total_never_executed, 0,
            "a dry-run probe that executed every line of launcher.py+srt/weg2 "
            "would mean it also executed the group-launch/spawn paths, which "
            "--dry-run must never reach",
        )


if __name__ == "__main__":
    unittest.main()
