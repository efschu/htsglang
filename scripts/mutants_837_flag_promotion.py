#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#837: apply each mutation, run the suite, report DEAD or ALIVE, revert.

WHY THIS EXISTS. Every test in this ticket passed the first time it was run,
which is exactly the condition under which a suite proves nothing. A test that
has never been observed to fail is a test whose failure mode is unknown, and
this tree has a standing rule about it: a gate needs a can-fail proof, not a
green run. It matters more than usual for a promotion: the whole change is
plumbing, and plumbing that silently does nothing looks identical to plumbing
that works, from every angle except a mutant.

EACH MUTANT IS A REAL MISTAKE SOMEBODY COULD MAKE, and the interesting ones are
the TEMPTING version of the change rather than an obviously broken one. M7 and
M8 are the two per-half overrides published through ``_b()`` -- the helper every
neighbouring clause in the same method already uses, which is precisely why
someone would reach for it, and which collapses a tri-state onto a boolean. M9
and M10 are the ``is not None`` guard dropped in favour of the shorter truthy
test, the mutation that ends byte-identity for every deployment that has not
moved. M13 keeps the refusal and only drops the CLI spelling from its message,
which is the version that still looks like a working validator.

THE REPORT NAMES WHICH TESTS DIED, not just how many. A mutation that turns the
whole suite red is weak evidence: it says the suite notices damage, not that the
specific arm aimed at that specific defect is the one that fires.

The mutated files are the SHIPPING sources, edited in place and restored in a
``finally``. Do not run this concurrently with an edit to server_args.py or
environ.py.

Run with the measurement environment, hermetically:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
      <venv>/bin/python scripts/mutants_837_flag_promotion.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY = re.compile(
    r"^(?:SUB)?FAILED(?P<params>\([^)]*\))?\s+\S+::(?P<test>\S+)"
)

ROOT = Path(__file__).resolve().parent.parent
SERVER_ARGS = ROOT / "python/sglang/srt/server_args.py"
ENVIRON = ROOT / "python/sglang/srt/environ.py"

TESTS = ROOT / "test/registered/unit/server_args"

# The #837 suite plus the two #781 suites it extends. The neighbours are in the
# run because a clause added to a shared publisher must not damage the clauses
# that were already there, and because a mutant that kills EVERYTHING is a
# weaker result than one that kills the arm it was aimed at.
SUITE = [
    "test_seam_flag_promotion_837.py",
    "test_promoted_flag_publication_781.py",
    "test_phase_policy_flag_promotion_781.py",
]


@dataclass
class Mutant:
    name: str
    why: str
    path: Path
    old: str
    new: str
    killed_by: list[str] = field(default_factory=list)


def _drop_publish(name: str, field_name: str, env_key: str, value_expr: str,
                  why: str) -> Mutant:
    """The clause removed entirely: the flag is accepted and goes nowhere."""
    return Mutant(
        name=name,
        why=why,
        path=SERVER_ARGS,
        old=(
            f"        if self.{field_name} is not None:\n"
            f'            os.environ["{env_key}"] = {value_expr}'
        ),
        new=(
            f"        if False:\n"
            f'            os.environ["{env_key}"] = {value_expr}'
        ),
    )


def _drop_guard(name: str, field_name: str, why: str) -> Mutant:
    """The ``is not None`` guard dropped: an unset flag publishes anyway."""
    return Mutant(
        name=name,
        why=why,
        path=SERVER_ARGS,
        old=f"        if self.{field_name} is not None:",
        new="        if True:",
    )


MUTANTS: list[Mutant] = [
    _drop_publish(
        "M1 --seam-shrink is accepted and published nowhere",
        "seam_shrink",
        "SGLANG_SEAM_SHRINK",
        "_b(self.seam_shrink)",
        why=(
            "the master gate reaches argparse and never reaches the runtime; "
            "the flag would show up in `ps` and in --help while the flip kept "
            "reading the boot env, which is the exact state #837 exists to end"
        ),
    ),
    _drop_publish(
        "M2 --seam-shrink-prearm-quiesce is published nowhere",
        "seam_shrink_prearm_quiesce",
        "SGLANG_SEAM_SHRINK_PREARM_QUIESCE",
        "str(\n                self.seam_shrink_prearm_quiesce\n            )",
        why="half A of the attribution pair cannot be cut apart from half B",
    ),
    _drop_publish(
        "M3 --seam-shrink-defer-grow is published nowhere",
        "seam_shrink_defer_grow",
        "SGLANG_SEAM_SHRINK_DEFER_GROW",
        "str(\n                self.seam_shrink_defer_grow\n            )",
        why="half B of the attribution pair, the same defect from the other end",
    ),
    _drop_publish(
        "M4 --seam-shrink-grow-debt-rounds is published nowhere",
        "seam_shrink_grow_debt_rounds",
        "SGLANG_SEAM_SHRINK_GROW_DEBT_ROUNDS",
        "str(\n                self.seam_shrink_grow_debt_rounds\n            )",
        why=(
            "the #834 ratchet patience silently stays at 32 rounds; a window "
            "that shortened it to catch an unpaid grow would wait the default "
            "and conclude the debt never happened"
        ),
    ),
    _drop_publish(
        "M5 --flip-seam-drain-budget-ms is published nowhere",
        "flip_seam_drain_budget_ms",
        "SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS",
        "str(\n                self.flip_seam_drain_budget_ms\n            )",
        why=(
            "the #830 seam guard keeps the derived 1094 ms no matter what the "
            "operator asked for, including a deliberate 0 to stand it down"
        ),
    ),
    _drop_publish(
        "M6 --hicache-read-buffers is published nowhere",
        "hicache_read_buffers",
        "SGLANG_HICACHE_READ_BUFFERS",
        "str(self.hicache_read_buffers)",
        why=(
            "the #720 read-buffer ring stays disabled; the knob DESIGN_706 "
            "still documents as a boot-env line would remain unreachable by "
            "flag, which is the pattern #781 exists to end"
        ),
    ),
    Mutant(
        name="M7 the prearm tri-state is published through _b()",
        why=(
            "THE TEMPTING MUTANT, and the reason the tri-state has its own "
            "arm. `_b()` is what every neighbouring clause in this method "
            "uses, so reaching for it is the natural edit. It maps -1 and 1 "
            "onto the same '1': a W13b run asking for 'follow the master' "
            "would instead FORCE that half on, measure both halves, and "
            "attribute the result to the one it believed it had held down"
        ),
        path=SERVER_ARGS,
        old=(
            '            os.environ["SGLANG_SEAM_SHRINK_PREARM_QUIESCE"] = str(\n'
            "                self.seam_shrink_prearm_quiesce\n"
            "            )"
        ),
        new=(
            '            os.environ["SGLANG_SEAM_SHRINK_PREARM_QUIESCE"] = _b(\n'
            "                self.seam_shrink_prearm_quiesce\n"
            "            )"
        ),
    ),
    Mutant(
        name="M8 the defer-grow tri-state is published through _b()",
        why="the same collapse on the other half; each half needs its own arm",
        path=SERVER_ARGS,
        old=(
            '            os.environ["SGLANG_SEAM_SHRINK_DEFER_GROW"] = str(\n'
            "                self.seam_shrink_defer_grow\n"
            "            )"
        ),
        new=(
            '            os.environ["SGLANG_SEAM_SHRINK_DEFER_GROW"] = _b(\n'
            "                self.seam_shrink_defer_grow\n"
            "            )"
        ),
    ),
    _drop_guard(
        "M9 the drain-budget guard is dropped (unset publishes 'None')",
        "flip_seam_drain_budget_ms",
        why=(
            "END OF BYTE-IDENTITY. Every boot that never asked for this knob "
            "would start writing SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS=None, which "
            "EnvInt cannot parse -- it warns and falls back to the default, so "
            "the damage is a log line and a habit, not a crash"
        ),
    ),
    _drop_guard(
        "M10 the seam-shrink guard is dropped (unset publishes '0')",
        "seam_shrink",
        why=(
            "the same end of byte-identity in its most plausible disguise: "
            "'0' looks harmless because off IS the default, but the key is now "
            "written by every boot, so an environment that legitimately set it "
            "is overwritten by a flag nobody passed"
        ),
    ),
    Mutant(
        name="M11 the tri-state validation accepts anything",
        why=(
            "a typo'd 2 reaches _seam_shrink_half, whose try/except contract "
            "is that a gate lookup must never break a flip -- so it is read as "
            "follow-the-master and the window silently measures both halves"
        ),
        path=SERVER_ARGS,
        old="            if value is not None and value not in (-1, 0, 1):",
        new="            if False:",
    ),
    Mutant(
        name="M12 the negative-value validation accepts anything",
        why=(
            "a negative budget or count is clamped to 0 by max(0, ...) deep in "
            "the runtime, so --flip-seam-drain-budget-ms -1 would stand the "
            "#830 guard down while reading like a request to tighten it"
        ),
        path=SERVER_ARGS,
        old="            if value is not None and value < 0:",
        new="            if False:",
    ),
    Mutant(
        name="M13 the refusal no longer names the flag in CLI spelling",
        why=(
            "the validator still refuses, so most of the suite stays green. "
            "An operator greps the message for the thing they typed, and "
            "'seam_shrink_grow_debt_rounds must be >= 0' does not match "
            "--seam-shrink-grow-debt-rounds; it sends them into the source"
        ),
        path=SERVER_ARGS,
        old="                    f\"--{field.replace('_', '-')} must be >= 0, got {value}.\"",
        new='                    f"{field} must be >= 0, got {value}."',
    ),
    Mutant(
        name="M14 the validator is never called from __post_init__",
        why=(
            "a validator nothing calls validates nothing, and this one is the "
            "only place a bad tri-state can be refused at all -- every "
            "downstream reader swallows it by design"
        ),
        path=SERVER_ARGS,
        old="        self._handle_seam_shrink_flags_837()",
        new="        pass",
    ),
    Mutant(
        name="M15 SGLANG_SEAM_SHRINK loses its deprecation notice",
        why=(
            "the bridge stops announcing itself. A boot script that still sets "
            "the env keeps working -- which is the point of the bridge -- but "
            "nothing tells its author that the flag now exists, so the #539 "
            "boot gate goes on refusing a key nobody knows to stop setting"
        ),
        path=ENVIRON,
        old=(
            "_warn_deprecated_env_to_cli_flag(\n"
            '    "SGLANG_SEAM_SHRINK",\n'
            "    \"Please use '--seam-shrink' instead.\",\n"
            ")"
        ),
        new="pass",
    ),
]


def _run() -> tuple[bool, list[str]]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(ROOT / "python")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *SUITE, "-q", "-p", "no:cacheprovider"],
        cwd=TESTS,
        env=env,
        capture_output=True,
        text=True,
    )
    # The summary is parsed, not just the exit code, because "the suite went
    # red" is a much weaker result than "the arm aimed at this defect fired".
    #
    # TWO THINGS THE OBVIOUS PARSE GETS WRONG, both found by running it. pytest
    # emits ANSI colour here even with stdout captured, so a startswith(
    # "FAILED") test never matches and every mutant reports "killed by:
    # nothing" while still counting DEAD -- a silently empty attribution.
    # And pytest-subtests labels a failing subtest SUBFAILED(<params>), so the
    # per-value arms -- which is most of this suite -- are invisible to a
    # FAILED-only match.
    out = _ANSI.sub("", proc.stdout)
    failed: list[str] = []
    for line in out.splitlines():
        hit = _SUMMARY.match(line)
        if hit is None:
            continue
        name = hit.group("test") + (hit.group("params") or "")
        if name not in failed:
            failed.append(name)
    return proc.returncode == 0, failed


def main() -> int:
    print("#837 mutant run. Baseline first: a mutant report against a red")
    print("baseline is unreadable, so the baseline is proved before anything")
    print("is mutated.\n")
    ok, failed = _run()
    if not ok:
        print(f"BASELINE RED: {failed}")
        return 2
    print("  baseline: green\n")

    alive: list[str] = []
    for m in MUTANTS:
        src = m.path.read_text()
        if src.count(m.old) != 1:
            print(
                f"SKIP {m.name}: anchor found {src.count(m.old)} times in "
                f"{m.path.name}, expected exactly 1"
            )
            alive.append(m.name + " (ANCHOR LOST)")
            continue
        m.path.write_text(src.replace(m.old, m.new))
        try:
            ok, failed = _run()
        finally:
            m.path.write_text(src)
        m.killed_by = failed
        verdict = "DEAD" if not ok else "ALIVE"
        if ok:
            alive.append(m.name)
        print(f"{verdict:5s} {m.name}")
        print(f"      why: {m.why}")
        for f in failed[:6]:
            print(f"      killed by: {f}")
        if len(failed) > 6:
            print(f"      ... and {len(failed) - 6} more")
        print()

    ok, failed = _run()
    print(f"restored green: {ok}")
    if alive:
        print(f"\nALIVE MUTANTS ({len(alive)}) -- each is an untested claim:")
        for a in alive:
            print(f"  {a}")
        return 1
    print(f"\nall {len(MUTANTS)} mutants DEAD")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
