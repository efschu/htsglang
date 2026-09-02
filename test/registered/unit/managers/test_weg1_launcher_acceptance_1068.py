"""#1068 WEG 1 slice 5: the launcher sizes the host pools from the demand
formula, the acceptance script can fail, and every line it greps exists.

THE LAW. WEG1_BUILD_SPEC_0901.md section 4.6 (launcher: the HICACHE_RATIO
fallback -- 'nobody chose this number' -- is REPLACED by
`--hicache-host-role staging --hicache-size S --hicache-mamba-host-mib M`
from the section-5 arithmetic, refusal instead of fallback when the formula
is not solvable), section 11 A11.1/A11.2 (population terms from the
CONFIGURED concurrency, every term with its provenance on ONE line, the
ledger cap as a NAMED degradation), section 10 (acceptance lines A1-A10 with
trapsafe_count.py, deadman armed with a pgrep proof) and the slice-2 lessons
carried into this slice's briefing: (A) an acceptance that cannot fail is not
an acceptance, so its can-fail proof is a test; (B) every line the
acceptance greps must exist VERBATIM in the tree at HEAD; (C) the dry-run
prints the command line of the HYBRID serving path the boot actually runs.

WHERE THE ARTEFACTS LIVE. The launcher and the acceptance are operator
files outside the python tree (/spinning/gpu-arb, overridable through
WEG1_GPU_ARB_DIR). On a box without that directory every test here SKIPS
(foreign checkout); on a box that HAS it a missing artefact is a FAIL, which
is what makes the file red-first on 285e3685b6: the sizing helper and the
acceptance script do not exist there, and the launcher has no `--dry-run`
guard.

SAFETY. The launcher test never executes a launcher that lacks the dry-run
guard: it checks the guard text statically first and fails by name. Running
the pre-slice launcher would start a serving boot on the cards.

RED on 285e3685b6: weg1_host_sizing.py absent (ImportError -> fail by name),
accept_weg1_1068.py absent, boot_855_train0901.sh carries no `--dry-run`.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest

import sglang
from sglang.test.test_utils import CustomTestCase

TREE = pathlib.Path(sglang.__file__).resolve().parents[2]
GPU_ARB = pathlib.Path(os.environ.get("WEG1_GPU_ARB_DIR", "/spinning/gpu-arb"))
SIZING = GPU_ARB / "devtools" / "weg1_host_sizing.py"
ACCEPT = GPU_ARB / "accept_weg1_1068.py"
LAUNCHER = GPU_ARB / "boot_855_train0901.sh"
PY = sys.executable

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


def _load(path: pathlib.Path, name: str):
    if not path.is_file():
        raise AssertionError(
            f"slice-5 artefact missing: {path} (red-first on 285e3685b6)"
        )
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the artefacts define @dataclass classes, and
    # dataclasses._is_type resolves sys.modules[cls.__module__] (None ->
    # AttributeError on an unregistered module; measured on this rig).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(GPU_ARB.is_dir(), "operator dir absent: foreign checkout")
class TestTheLauncherSizesFromTheDemandFormula(CustomTestCase):
    """Section 5 / A11.1 / A11.2 arithmetic, hermetic (pure python)."""

    def test_boot2_config_sizes_to_6_gb_and_2400_mib(self):
        m = _load(SIZING, "weg1_host_sizing")
        r = m.size_host_pools(**BOOT2)
        # Demand from the CONFIGURED concurrency (A11.1): (8 + 8 + 1) x 39365.
        self.assertEqual(r.demand_rows, 669205)
        self.assertEqual(r.s_demand_gb, 11)
        # Ledger cap (section 5): 119 - 16 GB - 10 GiB - 27 GiB = 63.27 GB,
        # minus anchors 2 x 3 x 2400 MiB = 15.10 GB, over 8 x S -> S = 6.
        self.assertEqual(r.s_ledger_gb, 6)
        self.assertEqual(r.s_gb, 6)
        self.assertEqual(r.ledger_rows, 366211)  # 6e9 // 16384 + 1 (base.py:140-147)
        self.assertEqual(r.pool_rows, 366211)
        self.assertEqual(r.m_mib, 2400)
        self.assertEqual(r.anchor_slots_rank0, 64)  # 2400 // 37.41
        self.assertEqual(f"{r.spans_at_prompt_max:.2f}", "9.30")
        # A11.2: the cap is a NAMED degradation, printed verbatim.
        text = "\n".join(r.lines)
        self.assertIn(
            "#1068 HOST POOL DEMAND EXCEEDS LEDGER demand_rows=669205 "
            "(n_resident=8 n_queue=8 chain_lag=1 prompt_max=39365) "
            "ledger_rows=366211 spans_at_prompt_max=9.30 -> pool sized to the "
            "ledger; requests beyond the pool land sequentially via evict_host "
            "or are truncated-named (L2)",
            text,
        )
        # A11.1: ONE terms line, every term with its provenance.
        terms = [ln for ln in r.lines if ln.startswith("#1068 WEG1 SIZING TERMS ")]
        self.assertEqual(len(terms), 1, text)
        for term in (
            "n_resident=8",
            "n_queue=8",
            "chain_lag=1",
            "prompt_max=39365",
            "cell_pp0=16384",
            "memavail=119.00 GB",
            "ranks=3",
            "m_mib=2400",
            "per_slot_rank0=37.41 MiB",
            "device_slots=20",
        ):
            self.assertIn(term, terms[0])
        self.assertIn("provenance", terms[0].lower())
        self.assertIsNone(r.refusal)

    def test_demand_within_the_ledger_takes_the_demand_without_a_degradation_line(self):
        m = _load(SIZING, "weg1_host_sizing")
        cfg = dict(BOOT2, n_queue=0)  # (8 + 0 + 1) x 39365 = 354285 rows -> 6 GB
        r = m.size_host_pools(**cfg)
        self.assertEqual(r.demand_rows, 354285)
        self.assertEqual(r.s_demand_gb, 6)
        self.assertEqual(r.s_gb, 6)
        self.assertNotIn("HOST POOL DEMAND EXCEEDS LEDGER", "\n".join(r.lines))
        big = dict(BOOT2, memavail_bytes=int(200e9))  # ledger 16 GB > demand 11
        r2 = m.size_host_pools(**big)
        self.assertEqual(r2.s_ledger_gb, 16)
        self.assertEqual(r2.s_gb, 11)
        self.assertNotIn("HOST POOL DEMAND EXCEEDS LEDGER", "\n".join(r2.lines))

    def test_an_unsolvable_ledger_is_a_refusal_naming_its_terms(self):
        m = _load(SIZING, "weg1_host_sizing")
        with self.assertRaises(m.SizingRefused) as cm:
            m.size_host_pools(**dict(BOOT2, memavail_bytes=int(60e9)))
        msg = str(cm.exception)
        for term in ("memavail", "floor", "reserve", "load transient", "anchors"):
            self.assertIn(term, msg)
        # The CLI form exits 2 (refusal, never a fallback).
        env = dict(os.environ, WEG1_MEMAVAIL_GB="60", WEG1_CELL_PP0_BYTES="16384")
        p = subprocess.run(
            [PY, str(SIZING), "--max-running-requests", "8",
             "--chunked-prefill-size", "4096", "--ranks", "3"],
            env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("#1068 WEG1 SIZING REFUSED", p.stdout + p.stderr)

    def test_the_anchor_floor_is_a_refusal(self):
        m = _load(SIZING, "weg1_host_sizing")
        with self.assertRaises(m.SizingRefused) as cm:
            m.size_host_pools(**dict(BOOT2, mamba_host_mib=800))  # 21 slots < 29
        msg = str(cm.exception)
        self.assertIn("slots", msg)
        self.assertIn("device_slots", msg)
        self.assertIn("max_running_requests", msg)


@unittest.skipUnless(GPU_ARB.is_dir(), "operator dir absent: foreign checkout")
class TestTheLauncherDryRun(CustomTestCase):
    """Lesson (C): the composed command line of the hybrid serving path."""

    def test_dry_run_prints_the_hybrid_command_line_without_starting_anything(self):
        self.assertTrue(LAUNCHER.is_file(), f"missing {LAUNCHER}")
        src = LAUNCHER.read_text()
        # NEVER execute a launcher without the guard: that starts a boot.
        self.assertTrue(
            '"${1:-}" = "--dry-run"' in src,
            "launcher has no --dry-run guard (red-first on 285e3685b6); "
            "not executing it",
        )
        env = dict(
            os.environ,
            WEG1_MEMAVAIL_GB="119",
            WEG1_CELL_PP0_BYTES="16384",
            TREE_ARG=str(TREE),
            TAG_ARG="weg1dryrun",
        )
        p = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            env=env, capture_output=True, text=True, timeout=180,
        )
        out = p.stdout + p.stderr
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("NOT started", out)
        argv_lines = [ln for ln in out.splitlines() if "sglang.launch_server" in ln]
        self.assertEqual(len(argv_lines), 1, out)
        argv = argv_lines[0]
        for tok in (
            "--hicache-host-role staging",
            "--hicache-size 6",
            "--hicache-mamba-host-mib 2400",
            "--enable-hierarchical-cache",
            "--enable-phase-flip",
            "--phase-flip-rebind-hicache",
            "--hicache-storage-backend file",
            "--max-running-requests 8",
        ):
            self.assertIn(tok, argv, argv)
        self.assertNotIn("--hicache-ratio", argv)
        # The path those flags select, named with its tree sites.
        self.assertIn("UnifiedRadixCache", out)
        self.assertIn("HybridCacheController", out)
        # Deadman: the dry-run names what it WOULD arm and arms nothing.
        self.assertIn("boot_deadman.sh", out)
        self.assertIn("GRACE_S=600", out)
        self.assertNotIn("deadman armed: pid", out)


@unittest.skipUnless(GPU_ARB.is_dir(), "operator dir absent: foreign checkout")
class TestTheAcceptanceCanFail(CustomTestCase):
    """Lesson (A): a grep that cannot fail is not an acceptance."""

    def test_selftest_fails_on_an_empty_log_and_passes_on_a_full_one(self):
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

    def test_every_grepped_line_exists_verbatim_in_the_tree(self):
        """Lesson (B): grep each anchor in python/ at HEAD; cite file:line."""
        m = _load(ACCEPT, "accept_weg1_1068")
        missing = []
        for marker in m.MARKERS:
            if marker.tree_anchor is None:
                continue  # launcher-origin line, lives in the operator dir
            p = subprocess.run(
                ["git", "-C", str(TREE), "grep", "-n", "-F", "--",
                 marker.tree_anchor, "--", "python/"],
                capture_output=True, text=True, timeout=120,
            )
            hits = [ln for ln in p.stdout.splitlines() if ln.strip()]
            if not hits:
                missing.append(f"{marker.name}: {marker.tree_anchor!r}")
        self.assertEqual(missing, [], "acceptance anchors absent from the tree")


if __name__ == "__main__":
    unittest.main()
