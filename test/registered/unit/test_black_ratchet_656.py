"""Ratchet guard: the phase-flip surface's `black` debt may only shrink.

WHY A TEST AND NOT A HOOK
-------------------------
`.pre-commit-config.yaml` has pinned `black` 26.1.0 since before any of this,
and MERGE-R9 section 1 is the second consecutive shift to record that it is
not running. The mechanical reason, checked rather than assumed: the repo's
hook directory contains only `reference-transaction`, i.e. `pre-commit
install` was never run here. A hook nobody installed cannot enforce anything,
and neither can a GitHub Actions check -- Actions are disabled on this fork.

What DOES run is the registered suite. So the gate lives here, where the same
command that proves the code proves the formatting, and a shift that forgets
`pre-commit install` still gets told.

WHAT IT GUARDS, AND WHAT IT DOES NOT
------------------------------------
SCOPE is the phase-flip surface this fork authors -- the `phase_flip_*`
modules plus the handful of files the seam reaches into, and the registered
`*_631.py` / `*_656.py` tests. NOT the whole tree: 762 of 6211 tracked `.py`
files are dirty under the pinned black, nearly all of it upstream, and a
762-entry allowlist is a list nobody maintains. Stating the limit is the
point; a gate that claimed the tree and quietly covered a tenth of it would
be worse than this one.

WITHIN that scope the rule is a RATCHET, not a clean-tree assertion. The
files already dirty are listed by name in `_KNOWN_DIRTY` and tolerated. A
file NOT on the list must be clean, so:

  * a NEW file that lands unformatted fails here;
  * an existing CLEAN file that gets dirtied fails here;
  * formatting one of the listed files and not removing its entry fails here
    too -- the list may only shrink, so the debt cannot be re-accrued under
    cover of an entry that is no longer earned.

MERGE-R9's recorded fourteen -- the eleven pre-existing and three new files
its table counted -- were formatted in the commit that added this file and
are therefore absent from the list below. Everything still on it is older
debt, named so the next shift can pick it off rather than rediscover it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import pathlib
import subprocess
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[3]

#: Globs, relative to the repo root, that this ratchet governs.
_SCOPE = (
    "python/sglang/srt/managers/phase_flip_*.py",
    "python/sglang/srt/managers/phase_purity.py",
    "python/sglang/srt/managers/corridor_guard.py",
    "python/sglang/srt/managers/kv_backing_relief.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/mem_cache/kv_vmm_backing.py",
    "python/sglang/srt/mem_ledger/corridor_trace.py",
    "python/sglang/srt/model_executor/weights_arena.py",
    "test/registered/**/*_631.py",
    "test/registered/**/*_656.py",
)

#: Files inside the scope that are ALREADY dirty. Shrink-only: remove an entry
#: when you format the file, never add one. Adding a name here to make a red
#: run green is the exact move this ratchet exists to make visible.
_KNOWN_DIRTY = frozenset(
    {
        "python/sglang/srt/managers/phase_flip_boot.py",
        "python/sglang/srt/managers/phase_flip_draft_bootstrap.py",
        "python/sglang/srt/managers/phase_flip_output_trace.py",
        "python/sglang/srt/managers/phase_flip_presence.py",
        "python/sglang/srt/managers/phase_flip_resident_carry.py",
        "test/registered/scheduler/test_flip_live_slot_agreement_656.py",
        "test/registered/scheduler/test_seam_fingerprint_and_margin_656.py",
        "test/registered/unit/distributed/test_census_wire_domain_631.py",
        "test/registered/unit/entrypoints/test_launch_phase_sigterm_656.py",
        "test/registered/unit/managers/test_corridor_admission_631.py",
        "test/registered/unit/managers/test_corridor_even_fill_631.py",
        "test/registered/unit/managers/test_corridor_guard_631.py",
        "test/registered/unit/managers/test_kv_admission_floor_631.py",
        "test/registered/unit/managers/test_kv_arena_reclaim_631.py",
        "test/registered/unit/managers/test_kv_arena_span_ops_631.py",
        "test/registered/unit/managers/test_kv_backing_collective_631.py",
        "test/registered/unit/managers/test_kvso_flip_contract_631.py",
        "test/registered/unit/managers/test_phase_flip_corridor_gate_631.py",
        "test/registered/unit/managers/test_phase_flip_draft_bootstrap_631.py",
        "test/registered/unit/managers/test_phase_flip_draft_carrier_631.py",
        "test/registered/unit/managers/test_phase_flip_live_slots_no_pool_idx_631.py",
        "test/registered/unit/managers/test_phase_flip_mover_streaming_631.py",
        "test/registered/unit/managers/test_phase_flip_seam_census_631.py",
        "test/registered/unit/managers/test_phase_flip_spec_seam_631.py",
        "test/registered/unit/managers/test_phase_flip_spill_depth_631.py",
        "test/registered/unit/managers/test_phase_flip_staging_reserve_631.py",
        "test/registered/unit/managers/test_phase_purity_631.py",
        "test/registered/unit/managers/test_pp_flip_slot_hold_631.py",
        "test/registered/unit/managers/test_pp_proxy_stamp_631.py",
        "test/registered/unit/managers/test_spec_counter_wire_631.py",
        "test/registered/unit/managers/test_spec_mamba_commit_width_631.py",
        "test/registered/unit/managers/test_spec_verify_width_631.py",
        "test/registered/unit/managers/test_truncation_align_admission_656.py",
    }
)

#: The version `.pre-commit-config.yaml` pins. Formatting is only reproducible
#: against one version, so a mismatch SKIPS rather than reports a verdict it
#: cannot stand behind -- a red run caused by the checker's own version is a
#: red run nobody can act on.
_PINNED_BLACK = "26.1.0"


def _scoped_files():
    out = set()
    for pattern in _SCOPE:
        for path in _REPO.glob(pattern):
            if path.is_file():
                out.add(path.relative_to(_REPO).as_posix())
    return out


def _black_version():
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "black", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:  # pragma: no cover - depends on the host
        return None
    for token in proc.stdout.split():
        if token[0].isdigit():
            return token
    return None


def _dirty(files):
    """Paths `black --check` would reformat. Reads black's own lines, never a
    pipeline's exit code -- MERGE-R9 section 1 recorded a `0 dirty of 20`
    table produced by `... | tail -5; echo rc=$?`, which captures TAIL's
    status and never consults the tool."""
    proc = subprocess.run(
        [sys.executable, "-m", "black", "--check", "--quiet", "--diff", *sorted(files)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=900,
    )
    out = set()
    for line in proc.stdout.splitlines():
        # `--diff` headers name the file: "--- path  2026-... +0000"
        if line.startswith("--- "):
            out.add(line[4:].split("\t")[0].strip())
    for line in proc.stderr.splitlines():
        if line.startswith("would reformat "):
            out.add(line[len("would reformat ") :].strip())
    return out


class TheFormattingDebtMayOnlyShrinkTest(unittest.TestCase):
    def setUp(self):
        version = _black_version()
        if version is None:
            self.skipTest("black is not importable in this interpreter")
        if version != _PINNED_BLACK:
            self.skipTest(
                f"black {version} is not the pinned {_PINNED_BLACK}; a "
                "formatting verdict from another version is not actionable"
            )
        self.files = _scoped_files()
        self.assertTrue(self.files, "the scope matched no files -- glob rot")

    def test_no_unlisted_file_in_the_scope_is_dirty(self):
        new = sorted(_dirty(self.files) - _KNOWN_DIRTY)
        self.assertEqual(
            [],
            new,
            "these files are not black-clean and are not on the ratchet's "
            "known-dirty list:\n  "
            + "\n  ".join(new)
            + "\n\nRun the pinned formatter over them:\n"
            f"  python -m black {' '.join(new)}\n"
            "and, if the hook is what let this through, install it:\n"
            "  pre-commit install\n"
            "(the repo's hook directory carries no pre-commit hook, which is "
            "why MERGE-R9 found the pinned black had stopped running).",
        )

    def test_the_known_dirty_list_has_no_stale_entries(self):
        """Shrink-only, enforced in the other direction.

        A file that has since been formatted must leave the list. Otherwise
        the list becomes a permanent licence and the next regression on that
        file is invisible.
        """
        dirty = _dirty(self.files)
        stale = sorted(name for name in _KNOWN_DIRTY if name not in dirty)
        self.assertEqual(
            [],
            stale,
            "these are listed as known-dirty but are clean now; remove them "
            "from _KNOWN_DIRTY so the file stays guarded:\n  " + "\n  ".join(stale),
        )

    def test_every_listed_name_still_exists(self):
        """A rename must not silently retire a guard."""
        missing = sorted(name for name in _KNOWN_DIRTY if not (_REPO / name).is_file())
        self.assertEqual([], missing, missing)


if __name__ == "__main__":
    unittest.main()
