"""#800: run the suite against one deliberately broken build at a time.

Each entry names a call edge or a guard, the exact source mutation that
disables it, and the tests that MUST turn red. A guard whose removal changes
nothing is not a guard; a test that stays green under it proves nothing.
"""

import atexit
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

#: The repository root, derived from this file's own location so the harness
#: runs in any checkout or worktree rather than one hard-coded path.
ROOT = Path(__file__).resolve().parents[4]
DISP = ROOT / "python/sglang/srt/managers/pp_stash_disposition.py"
MIXIN = ROOT / "python/sglang/srt/managers/scheduler_pp_mixin.py"
RUNTIME = ROOT / "python/sglang/srt/managers/phase_flip_runtime.py"
SUITE = "test/registered/unit/managers/test_pp_presence_withholding_deadlock_800.py"

MUTATIONS = [
    (
        "M1 the exemption itself: admission_decision blocks again (the wedge)",
        DISP,
        '    "admission_decision": PP_LOOP_ONLY,',
        '    "admission_decision": BLOCKS_FLIP,',
        [
            "test_an_admission_decision_no_longer_withholds_presence",
            "test_the_gate_announces_with_an_admission_decision_stashed",
        ],
    ),
    (
        "M2 the exemption made over-broad: an owed output stops blocking",
        DISP,
        '    "output": BLOCKS_FLIP,',
        '    "output": PP_LOOP_ONLY,',
        [
            "test_can_fail_an_output_still_withholds_presence",
            "test_can_fail_the_gate_still_withholds_with_an_output_stashed",
        ],
    ),
    (
        "M3 the undeclared state removed: an unknown kind is waved through",
        DISP,
        "    return _DISPOSITIONS.get(str(kind), UNDECLARED)",
        "    return _DISPOSITIONS.get(str(kind), PP_LOOP_ONLY)",
        [
            "test_can_fail_an_undeclared_kind_still_withholds_and_says_so_by_name",
        ],
    ),
    (
        "M4 the escape's call edge cut out of the shipped service turn",
        MIXIN,
        "            self.pp_flip_retire_undeclared_stash()",
        "            pass  # mutation: the escape is no longer called",
        ["test_the_shipped_service_turn_runs_the_escape"],
    ),
    (
        "M5 the escape fires unconditionally, ignoring its deadline",
        MIXIN,
        "            if deadline <= 0 or age < deadline:",
        "            if False:",
        [
            "test_can_fail_the_escape_does_nothing_when_switched_off",
            "test_an_undeclared_kind_is_retired_once_the_escape_deadline_expires",
            "test_the_escape_clock_restarts_when_the_key_empties",
        ],
    ),
    (
        "M6 the escape retires declared kinds too (corpse S with a clock)",
        MIXIN,
        "        keys = stash_keys_with_disposition(inbox, (UNDECLARED,))",
        "        keys = stash_keys_with_disposition(inbox, (UNDECLARED, PP_LOOP_ONLY))",
        ["test_can_fail_the_escape_never_retires_a_declared_kind"],
    ),
    (
        "M7 a raising channel probe reads as clean again",
        RUNTIME,
        '                logger.warning("%s channel probe failed: %s", LOG_PREFIX, exc)\n                unclean = (',
        '                logger.warning("%s channel probe failed: %s", LOG_PREFIX, exc)\n                unclean = None\n                _mutated = (',
        [
            "test_a_raising_channel_probe_withholds_instead_of_announcing",
            "test_a_raising_channel_probe_still_abandons_on_the_deadline",
        ],
    ),
    (
        "M8 the cutover's call edge removed",
        RUNTIME,
        '        _retire_fn = getattr(scheduler, "pp_flip_retire_pp_loop_stash", None)',
        "        _retire_fn = None  # mutation: the cutover no longer retires",
        ["test_the_cutover_call_site_runs_before_the_ring_is_rebuilt"],
    ),
    (
        "M9 the abandonment goes back to naming one of two causes",
        RUNTIME,
        '                f"THIS rank withheld its own presence for "',
        '                f"upstream, always upstream, for "',
        ["test_the_abandonment_names_this_rank_s_own_withhold"],
    ),
]


def run_suite():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            SUITE,
            "-q",
            "-p",
            "no:randomly",
            "--no-header",
            "--tb=no",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(ROOT / "python"),
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": "/usr/bin:/bin",
            "HOME": "/root",
        },
        timeout=600,
    )
    return proc.stdout


def failed_names(out):
    names = set()
    for line in out.splitlines():
        if line.startswith("FAILED "):
            # Strip pytest's parametrise suffix: a red `test_x[admission_
            # decision]` is a red `test_x`, and reading it otherwise let a live
            # mutant look killed-nowhere on the first run of this harness.
            name = line.split("::")[-1].split(" ")[0]
            names.add(name.split("[")[0])
    return names


def check_tree_is_pristine():
    """REFUSE TO START on a tree that still carries a mutation.

    This harness was itself the source of a defect it exists to catch. Two of
    its runs were killed mid-mutation; `finally` does not run under SIGKILL, so
    `"output": PP_LOOP_ONLY` and `if False:` were left in the shipped files and
    every later "green" run measured a build nobody had written. A restore that
    only happens on the clean path is not a restore.
    """
    missing = [
        (path.name, old)
        for _label, path, old, _new, _expect in MUTATIONS
        if old not in path.read_text()
    ]
    if missing:
        print("REFUSING TO RUN: the tree is not pristine. Missing anchors:")
        for name, old in missing:
            print(f"  {name}: {old.splitlines()[0][:70]}")
        print(
            "A previous run was killed before it could restore. Restore these "
            "by hand (or from git) before mutating anything further."
        )
        return False
    return True


def main():
    if not check_tree_is_pristine():
        return 2
    ok = True
    only = sys.argv[1:]
    # Restore on ANY exit, including a signal: the `finally` below covers the
    # clean path and an exception, and these cover the rest.
    live = {}

    def restore_all(*_args):
        for path, backup in list(live.items()):
            shutil.copy2(backup, path)
            live.pop(path, None)

    atexit.register(restore_all)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, f: (restore_all(), sys.exit(128 + s)))

    for label, path, old, new, expect_red in MUTATIONS:
        if only and not any(label.startswith(o) for o in only):
            continue
        src = path.read_text()
        if old not in src:
            print(f"[SETUP FAIL] {label}: anchor not found in {path.name}")
            ok = False
            continue
        backup = tempfile.mktemp()
        shutil.copy2(path, backup)
        live[path] = backup
        try:
            path.write_text(src.replace(old, new, 1))
            out = run_suite()
            red = failed_names(out)
        finally:
            shutil.copy2(backup, path)
            live.pop(path, None)
        missing = [t for t in expect_red if t not in red]
        extra_ok = red - set(expect_red)
        verdict = "OK  " if not missing else "FAIL"
        if missing:
            ok = False
        print(f"[{verdict}] {label}")
        print(f"        red: {sorted(red) if red else 'NOTHING WENT RED'}")
        if missing:
            print(f"        MISSING (stayed green): {missing}")
        if extra_ok:
            print(f"        also red (acceptable): {sorted(extra_ok)}")
    print("\nALL MUTATIONS KILLED" if ok else "\nSOME MUTATIONS SURVIVED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
