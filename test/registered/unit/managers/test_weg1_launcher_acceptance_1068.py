"""#1068 WEG 1 slice 5 (+fix): the launcher sizes the host pools from the
in-tree demand formula, refuses instead of falling back, proves its export,
refuses a tree without the contract; the acceptance can fail per re-admitted
rid; every line it greps exists.

THE LAW. WEG1_BUILD_SPEC_0901.md section 4.6 (launcher: the HICACHE_RATIO
fallback -- 'nobody chose this number' -- is REPLACED by
`--hicache-host-role staging --hicache-size S --hicache-mamba-host-mib M`
from the section-5 arithmetic, refusal instead of fallback when the formula
is not solvable), section 11 A11.1/A11.2 (population terms from the
CONFIGURED concurrency, every term with its provenance on ONE line, the
ledger cap as a NAMED degradation), A11.4 (WEG1_PROMPT_MAX_TOKENS exported
for L8), section 10 A5 / A11.5 (follower materialisation per re-admitted rid)
and the operator decisions of 2026-09-02 on the slice-5 review: (1) the host
floor is the gate's 16 GiB (host_ledger_preflight.sh FLOOR_G), the spec's
'16 GB' is a unit slip; (2)+(6) the arithmetic is the VERSIONED module
sglang.srt.planner.weg1_host_sizing, the launcher runs it under
PYTHONPATH=$TREE/python and probes the module plus the slice-1 L13 refusal
symbol on TREE before composing anything, naming what is missing; (3) the
ADMISSION-WEDGE genuine form is the invariant_checker.py:835 verdict, the
http_server.py:1029 description is bare only; (7) the refusal branch and the
export are pinned on the dry-run; (8) A5/A11.5 count from the #969C
population and the first PP0 ADMIT per rid.

RED-FIRST, HONESTLY. Measured on the parent 285e3685b6 (scratch worktree,
PYTHONPATH=<scratch>/python, 2026-09-02): 8 red / 5 green of 13. RED: the
five sizing tests (the module sglang.srt.planner.weg1_host_sizing does not
exist there, ImportError named) and the three dry-run tests that expect the
sized argv or a SIZING refusal (the launcher's TREE-contract probe refuses
that tree first, exit 2 before any argv). GREEN ON THE PARENT BY
CONSTRUCTION, named: three pins of the launcher alone, which build fake
trees or read the launcher text (test_a_sizing_without_machine_lines_
stops_the_launcher, test_a_tree_without_the_contract_is_refused_by_name,
test_the_export_is_read_back_through_env_not_echoed), and two
CHARACTERISATION pins of operator artefacts outside the tree
(/spinning/gpu-arb, overridable through WEG1_GPU_ARB_DIR):
test_selftest_can_fail_per_rid_and_per_wedge_form (accept_weg1_1068.py
--selftest) and test_every_grepped_line_exists_verbatim_in_the_tree (the
anchors predate slice 5). On a box without the operator directory every
test SKIPS (foreign checkout).

SAFETY. The launcher tests never execute a launcher that lacks the dry-run
guard: they check the guard text statically first and fail by name. Running
the pre-slice launcher would start a serving boot on the cards. The fake
trees built here carry no model, no venv and no GPU access; the launcher
refuses them before it composes an argv.

SINGLE TARGETED COLD TESTS BY SUBPROCESS (reason stated per speed-mode
rules): the launcher and the acceptance are shell/operator files that cannot
be exercised on metal without a boot; the dry-run and the selftest are the
only instruments that can fail on them at the desk.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

import importlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import sglang
from sglang.test.test_utils import CustomTestCase

TREE = pathlib.Path(sglang.__file__).resolve().parents[2]
GPU_ARB = pathlib.Path(os.environ.get("WEG1_GPU_ARB_DIR", "/spinning/gpu-arb"))
ACCEPT = GPU_ARB / "accept_weg1_1068.py"
LAUNCHER = GPU_ARB / "boot_855_train0901.sh"
PREFLIGHT = GPU_ARB / "devtools" / "host_ledger_preflight.sh"
PY = sys.executable
MODULE = "sglang.srt.planner.weg1_host_sizing"
L13_SYMBOL = "_refuse_incomplete_phase_flip_hicache_sizing_1068"

# Boot-2 configuration (spec section 3, log 175) and the rig terms of
# section 5 / A11.1, each with the provenance the launcher must print.
BOOT2 = dict(
    max_running_requests=8,
    chunked_prefill_size=4096,
    ranks=3,
    memavail_bytes=int(119e9),  # spec section 3: MemAvailable 119.0 GB
    cell_pp0_bytes=16384,  # spec section 5, log 1329/1350
    prompt_max_tokens=39365,  # spec section 5, log 66938/154436
    n_queue=8,  # A11.1 default = max_running_requests
    mamba_host_mib=2400,  # spec section 5 acceptance value
    per_slot_rank0_mib=37.41,  # log 1351
    device_slots=20,  # log 1351 device_slots=20
)


def _sizing():
    """The in-tree module, or a failure that names it (red-first on 285e3685b6)."""
    try:
        return importlib.import_module(MODULE)
    except ImportError as e:  # pragma: no cover - the red branch
        raise AssertionError(
            f"in-tree module {MODULE} absent (red-first on 285e3685b6): {e}"
        )


def _load_file(path: pathlib.Path, name: str):
    if not path.is_file():
        raise AssertionError(f"operator artefact missing: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the artefact defines @dataclass classes, and
    # dataclasses._is_type resolves sys.modules[cls.__module__] (None ->
    # AttributeError on an unregistered module; measured on this rig).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _launcher_env(**over):
    env = dict(
        os.environ,
        WEG1_MEMAVAIL_GB="119",
        WEG1_CELL_PP0_BYTES="16384",
        TREE_ARG=str(TREE),
        TAG_ARG="weg1dryrun",
    )
    env.update(over)
    return env


def _dry_run(env):
    src = LAUNCHER.read_text()
    # NEVER execute a launcher without the guard: that starts a boot.
    if '"${1:-}" = "--dry-run"' not in src:
        raise AssertionError("launcher has no --dry-run guard; not executing it")
    p = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        env=env, capture_output=True, text=True, timeout=180,
    )
    return p.returncode, p.stdout + p.stderr


def _fake_tree(td: str, with_module: bool, with_symbol: bool) -> pathlib.Path:
    """A tree the launcher can probe: regular packages (so an installed
    sglang cannot shadow them), a server_args.py with or without the L13
    refusal, a sizing module that prints NO machine line, or none."""
    root = pathlib.Path(td) / "tree"
    pkg = root / "python" / "sglang" / "srt" / "planner"
    pkg.mkdir(parents=True)
    for d in (root / "python" / "sglang", root / "python" / "sglang" / "srt", pkg):
        (d / "__init__.py").write_text("")
    sa = root / "python" / "sglang" / "srt" / "server_args.py"
    if with_symbol:
        sa.write_text(f"def {L13_SYMBOL}(self):\n    pass\n")
    else:
        sa.write_text("# a tree without the slice-1 L13 refusal\n")
    if with_module:
        (pkg / "weg1_host_sizing.py").write_text(
            "print('stub sizing: prints no WEG1_ machine line and exits 0')\n"
        )
    subprocess.run(["git", "init", "-q", str(root)], capture_output=True, check=False)
    return root


@unittest.skipUnless(GPU_ARB.is_dir(), "operator dir absent: foreign checkout")
class TestTheLauncherSizesFromTheDemandFormula(CustomTestCase):
    """Section 5 / A11.1 / A11.2 arithmetic, hermetic (pure python), on the
    IN-TREE module."""

    def test_boot2_config_sizes_to_5_gb_and_2400_mib_under_the_16_gib_floor(self):
        m = _sizing()
        r = m.size_host_pools(**BOOT2)
        # Demand from the CONFIGURED concurrency (A11.1): (8 + 8 + 1) x 39365.
        self.assertEqual(r.demand_rows, 669205)
        self.assertEqual(r.s_demand_gb, 11)
        # Ledger cap (decision 1: the gate's 16 GiB, not the spec's 16 GB):
        # 119e9 - 16 GiB - 10 GiB - 27 GiB = 62.09 GB, minus anchors
        # 2 x 3 x 2400 MiB = 15.10 GB -> 46.99 GB over 8 x S -> S = 5.
        # (With the spec's 16e9 literal the same box gave S = 6: the 1.18 GB
        # the GiB floor costs sits exactly on that boundary; spec R1 names
        # S = 5 as its own fallback value.)
        self.assertEqual(r.cap_bytes, int(119e9) - 53 * 2**30)
        self.assertEqual(r.s_ledger_gb, 5)
        self.assertEqual(r.s_gb, 5)
        self.assertEqual(r.ledger_rows, 305176)  # 5e9 // 16384 + 1 (base.py:140-147)
        self.assertEqual(r.pool_rows, 305176)
        self.assertEqual(r.m_mib, 2400)
        self.assertEqual(r.anchor_slots_rank0, 64)  # 2400 // 37.41
        self.assertEqual(f"{r.spans_at_prompt_max:.2f}", "7.75")
        # A11.2: the cap is a NAMED degradation, printed verbatim.
        text = "\n".join(r.lines)
        self.assertIn(
            "#1068 HOST POOL DEMAND EXCEEDS LEDGER demand_rows=669205 "
            "(n_resident=8 n_queue=8 chain_lag=1 prompt_max=39365) "
            "ledger_rows=305176 spans_at_prompt_max=7.75 -> pool sized to the "
            "ledger; requests beyond the pool land sequentially via evict_host "
            "or are truncated-named (L2)",
            text,
        )
        # A11.1: ONE terms line, every term with its provenance; the floor
        # prints in GiB AND GB (decision 1), the ledger line prints the cap.
        terms = [ln for ln in r.lines if ln.startswith("#1068 WEG1 SIZING TERMS ")]
        self.assertEqual(len(terms), 1, text)
        for term in (
            "n_resident=8",
            "n_queue=8",
            "chain_lag=1",
            "prompt_max=39365",
            "cell_pp0=16384",
            "memavail=119.00 GB",
            "floor=16 GiB (17.18 GB)",
            "ranks=3",
            "m_mib=2400",
            "per_slot_rank0=37.41 MiB",
            "device_slots=20",
        ):
            self.assertIn(term, terms[0])
        self.assertIn("provenance", terms[0].lower())
        ledger = [ln for ln in r.lines if ln.startswith("#1068 WEG1 LEDGER cap=")]
        self.assertEqual(len(ledger), 1, text)
        self.assertIn("cap=62.09 GB", ledger[0])
        self.assertIn("floor 16 GiB (17.18 GB)", ledger[0])
        self.assertIn("S_ledger=5 GB", ledger[0])
        self.assertIsNone(r.refusal)

    def test_the_floor_is_the_gates_16_gib_not_the_specs_16_gb(self):
        m = _sizing()
        self.assertEqual(m.FLOOR_BYTES, 16 * 2**30)
        # The gate the launcher runs first (#721) refuses under FLOOR_G GiB.
        self.assertTrue(PREFLIGHT.is_file(), f"missing {PREFLIGHT}")
        gate = PREFLIGHT.read_text()
        self.assertIn("FLOOR_G=${FLOOR_G:-16}", gate)
        self.assertIn("GiB floor", gate)

    def test_demand_within_the_ledger_takes_the_demand_without_a_degradation_line(self):
        m = _sizing()
        big = dict(BOOT2, memavail_bytes=int(200e9))  # ledger 15 GB > demand 11
        r = m.size_host_pools(**big)
        self.assertEqual(r.s_ledger_gb, 15)
        self.assertEqual(r.s_gb, 11)
        self.assertNotIn("HOST POOL DEMAND EXCEEDS LEDGER", "\n".join(r.lines))
        small = dict(big, n_queue=0)  # (8 + 0 + 1) x 39365 = 354285 rows -> 6 GB
        r2 = m.size_host_pools(**small)
        self.assertEqual(r2.demand_rows, 354285)
        self.assertEqual(r2.s_demand_gb, 6)
        self.assertEqual(r2.s_gb, 6)
        self.assertNotIn("HOST POOL DEMAND EXCEEDS LEDGER", "\n".join(r2.lines))

    def test_an_unsolvable_ledger_is_a_refusal_naming_its_terms(self):
        m = _sizing()
        with self.assertRaises(m.SizingRefused) as cm:
            m.size_host_pools(**dict(BOOT2, memavail_bytes=int(60e9)))
        msg = str(cm.exception)
        for term in ("memavail", "floor 16 GiB", "reserve", "load transient", "anchors"):
            self.assertIn(term, msg)
        # The CLI form, run the way the launcher runs it, exits 2 (refusal,
        # never a fallback).
        env = dict(
            os.environ, WEG1_MEMAVAIL_GB="60", WEG1_CELL_PP0_BYTES="16384",
            PYTHONPATH=str(TREE / "python"),
        )
        p = subprocess.run(
            [PY, "-m", MODULE, "--max-running-requests", "8",
             "--chunked-prefill-size", "4096", "--ranks", "3"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("#1068 WEG1 SIZING REFUSED", p.stdout + p.stderr)

    def test_the_anchor_floor_is_a_refusal(self):
        m = _sizing()
        with self.assertRaises(m.SizingRefused) as cm:
            m.size_host_pools(**dict(BOOT2, mamba_host_mib=800))  # 21 slots < 29
        msg = str(cm.exception)
        self.assertIn("slots", msg)
        self.assertIn("device_slots", msg)
        self.assertIn("max_running_requests", msg)


@unittest.skipUnless(GPU_ARB.is_dir(), "operator dir absent: foreign checkout")
class TestTheLauncherDryRun(CustomTestCase):
    """Lesson (C) plus the round-1 mutants MC (refusal) and MD (export)."""

    def test_dry_run_prints_the_hybrid_command_line_without_starting_anything(self):
        self.assertTrue(LAUNCHER.is_file(), f"missing {LAUNCHER}")
        rc, out = _dry_run(_launcher_env())
        self.assertEqual(rc, 0, out)
        self.assertIn("NOT started", out)
        self.assertIn("#1068 WEG1 TREE CONTRACT PRESENT on TREE=", out)
        self.assertIn("floor 16 GiB (17.18 GB)", out)
        argv_lines = [ln for ln in out.splitlines() if "sglang.launch_server" in ln]
        self.assertEqual(len(argv_lines), 1, out)
        argv = argv_lines[0]
        for tok in (
            "--hicache-host-role staging",
            "--hicache-size 5",
            "--hicache-mamba-host-mib 2400",
            "--tp-size 1 --pp-size 3",
            "--enable-hierarchical-cache",
            "--enable-phase-flip",
            "--phase-flip-rebind-hicache",
            "--hicache-storage-backend file",
            "--max-running-requests 8",
        ):
            self.assertIn(tok, argv, argv)
        self.assertNotIn("--hicache-ratio", argv)
        # A11.4 (mutant MD): the export is proven from env(1) in a child
        # process, i.e. from the EXPORTED variable, never from an echo of
        # the shell variable. An assignment without `export` prints an empty
        # value here, and the launcher then refuses (exit 2), so this
        # assertion and the rc above both go red.
        self.assertIn("A11.4 export proof from env(1): WEG1_PROMPT_MAX_TOKENS=39365", out)
        self.assertNotIn("exported for L8", out)
        # The path those flags select, named with its tree sites.
        self.assertIn("UnifiedRadixCache", out)
        self.assertIn("HybridCacheController", out)
        # Deadman: the dry-run names what it WOULD arm and arms nothing.
        self.assertIn("boot_deadman.sh", out)
        self.assertIn("GRACE_S=600", out)
        self.assertNotIn("deadman armed: pid", out)

    def test_a_sizing_refusal_stops_the_launcher_with_exit_2_and_no_argv(self):
        """Mutant MC: a launcher that replaced the refusal by a silent
        fallback would print an EFFECTIVE line and an argv here."""
        rc, out = _dry_run(_launcher_env(WEG1_MEMAVAIL_GB="60"))
        self.assertEqual(rc, 2, out)
        self.assertIn("#1068 WEG1 SIZING REFUSED: ledger cap leaves no KV budget", out)
        self.assertIn(
            "#1068 WEG1 SIZING REFUSED (exit 2) -- not starting serving; "
            "there is no ratio fallback on this form.",
            out,
        )
        self.assertNotIn("sglang.launch_server", out)
        self.assertNotIn("hicache flags EFFECTIVE", out)
        self.assertNotIn("composed command line", out)

    def test_a_bad_knob_is_a_refusal_not_a_default(self):
        rc, out = _dry_run(_launcher_env(WEG1_QUEUE_DEPTH="abc"))
        self.assertEqual(rc, 2, out)
        self.assertIn("#1068 WEG1 SIZING REFUSED: WEG1_QUEUE_DEPTH='abc' is not an integer", out)
        self.assertIn("#1068 WEG1 SIZING REFUSED (exit 2)", out)
        self.assertNotIn("sglang.launch_server", out)
        self.assertNotIn("hicache flags EFFECTIVE", out)

    def test_a_sizing_without_machine_lines_stops_the_launcher(self):
        """The second refusal branch: a sizing that exits 0 but chooses no
        number is the exact defect the block exists for."""
        with tempfile.TemporaryDirectory(prefix="weg1_fake_tree_") as td:
            root = _fake_tree(td, with_module=True, with_symbol=True)
            rc, out = _dry_run(_launcher_env(TREE_ARG=str(root)))
        self.assertEqual(rc, 2, out)
        self.assertIn("#1068 WEG1 TREE CONTRACT PRESENT on TREE=", out)
        self.assertIn(
            "#1068 WEG1 SIZING emitted no S_GB / M_MIB / PROMPT_MAX machine line -- refusing to start",
            out,
        )
        self.assertNotIn("sglang.launch_server", out)
        self.assertNotIn("hicache flags EFFECTIVE", out)

    def test_a_tree_without_the_contract_is_refused_by_name(self):
        """Decision 2/6: the WEG1 boot path never silently boots a tree
        without the sizing module and the slice-1 L13 refusal (the
        d2b78d38d8 trap). Both missing parts are named."""
        with tempfile.TemporaryDirectory(prefix="weg1_fake_tree_") as td:
            root = _fake_tree(td, with_module=False, with_symbol=False)
            rc, out = _dry_run(_launcher_env(TREE_ARG=str(root)))
        self.assertEqual(rc, 2, out)
        line = [ln for ln in out.splitlines() if "#1068 WEG1 TREE CONTRACT MISSING on TREE=" in ln]
        self.assertEqual(len(line), 1, out)
        self.assertIn(MODULE, line[0])
        self.assertIn(L13_SYMBOL, line[0])
        self.assertIn("not starting serving", line[0])
        self.assertNotIn("sglang.launch_server", out)
        self.assertNotIn("hicache flags EFFECTIVE", out)
        self.assertNotIn("#1068 WEG1 SIZING TERMS", out)

    def test_the_export_is_read_back_through_env_not_echoed(self):
        src = LAUNCHER.read_text()
        self.assertIn("env | sed -n 's/^WEG1_PROMPT_MAX_TOKENS=//p'", src)
        self.assertIn("#1068 WEG1 A11.4 EXPORT FAILED", src)
        self.assertNotIn("exported for L8", src)
        for zombie in ("HICACHE_RATIO", "_HR_FALLBACK", "--hicache-ratio", "devtools/weg1_host_sizing"):
            self.assertNotIn(zombie, src, zombie)


@unittest.skipUnless(GPU_ARB.is_dir(), "operator dir absent: foreign checkout")
class TestTheAcceptanceCanFail(CustomTestCase):
    """Lesson (A): a grep that cannot fail is not an acceptance. These two
    are CHARACTERISATION pins of an operator artefact and of the tree's
    anchors: green on the parent by construction (stated in the docstring)."""

    def test_selftest_can_fail_per_rid_and_per_wedge_form(self):
        self.assertTrue(ACCEPT.is_file(), f"missing {ACCEPT}")
        p = subprocess.run(
            [PY, str(ACCEPT), "--selftest"],
            capture_output=True, text=True, timeout=300,
        )
        out = p.stdout + p.stderr
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("SELFTEST empty-log exit=1", out)
        self.assertIn("missing line", out)
        self.assertIn("SELFTEST full-log exit=0", out)
        # Decision 8: the follower gap is caught per rid, naming rid,
        # follower and P; A11.5 stays existential and NAMES the gap cutover.
        self.assertIn("SELFTEST follower-gap-one exit=1 (want 1); A5 FAIL names rid/follower/P: True; A11.5 verdict as expected (named, still PASS): True", out)
        self.assertIn("SELFTEST follower-gap-all exit=1 (want 1); A5 FAIL names rid/follower/P: True; A11.5 verdict as expected (FAIL): True", out)
        # Decision 3: the two renderings of the wedge token.
        self.assertIn(
            "SELFTEST wedge-forms: description (http_server.py:1029) genuine=0 (want 0); "
            "verdict (invariant_checker.py:835) genuine=1 (want 1)",
            out,
        )
        self.assertIn("SELFTEST PASS", out)

    def test_every_grepped_line_exists_verbatim_in_the_tree(self):
        """Lesson (B): grep each anchor in python/ at HEAD; cite file:line.
        `git grep` when TREE is a repository, `grep -r` otherwise (a tree
        extracted with git archive has no .git)."""
        m = _load_file(ACCEPT, "accept_weg1_1068")
        is_repo = subprocess.run(
            ["git", "-C", str(TREE), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        ).returncode == 0
        missing = []
        for marker in m.MARKERS:
            if marker.tree_anchor is None:
                continue  # launcher-origin line, lives in the operator dir
            if is_repo:
                cmd = ["git", "-C", str(TREE), "grep", "-n", "-F", "--", marker.tree_anchor, "--", "python/"]
            else:
                cmd = ["grep", "-rnF", "--", marker.tree_anchor, str(TREE / "python")]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            hits = [ln for ln in p.stdout.splitlines() if ln.strip()]
            if not hits:
                missing.append(f"{marker.name}: {marker.tree_anchor!r}")
        self.assertEqual(missing, [], "acceptance anchors absent from the tree")
        # Decision 3, pinned on the tree: the wedge anchor is the VERDICT
        # emitter, and the description form exists too (bare only).
        wedge = m.BY_NAME["WEDGE"]
        self.assertIn("NO first token for", wedge.tree_anchor)
        self.assertIn("invariant_checker.py:835", wedge.site)
        desc = "`ADMISSION-WEDGE: N queued`"
        if is_repo:
            cmd = ["git", "-C", str(TREE), "grep", "-n", "-F", "--", desc, "--", "python/sglang/srt/entrypoints/http_server.py"]
        else:
            cmd = ["grep", "-nF", "--", desc, str(TREE / "python/sglang/srt/entrypoints/http_server.py")]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertTrue(p.stdout.strip(), "http_server.py description form absent")


if __name__ == "__main__":
    unittest.main()
