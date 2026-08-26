#!/usr/bin/env python3
"""#887 CAN-FAIL PROOF: deliberately break the fix, prove the suite goes red.

A test that passes in both worlds measures nothing. Each mutant below is a
plausible wrong implementation of the one-chunk exception -- most of them are
the reading somebody would reach for FIRST -- and the run is a pass only when
every one of them is KILLED by a named test.

Not a pytest module on purpose: it rewrites source files, so it must never be
collected into a normal run. Invoke it directly:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/tmp/wt887-stub:<worktree>/python \\
        python3 test/registered/unit/managers/mutation_proof_887.py

THE EXTRACTION IS COUNTED AGAINST THE SUMMARY, deliberately: a killed-count
derived from parsing that disagrees with the number of mutants attempted means
the HARNESS is broken, and a 0 for the wrong reason is indistinguishable from
green. The verdict below is computed from pytest's exit status, not from
scraping ANSI-coloured output.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
SRC = ROOT / "python" / "sglang" / "srt" / "managers"
TESTS = ROOT / "test" / "registered" / "unit" / "managers"

PURITY = SRC / "phase_purity.py"
CONFORM = SRC / "layout_conformance.py"
FLIP = SRC / "phase_flip_runtime.py"
SUITE = TESTS / "test_one_chunk_tp_prefill_887.py"

#: (name, file, find, replace, the test that must die)
MUTANTS = [
    (
        "the <= reading the permission invites (W37-D's 258 batches go quiet)",
        CONFORM,
        "if not (0 < computed < cap):",
        "if not (0 < computed <= cap):",
        "test_a_batch_AT_the_chunk_size_is_still_a_violation",
    ),
    (
        "the grant is never booked -- 'up to one chunk' becomes unbounded",
        PURITY,
        "            _spend_tp_compute_chunk(scheduler)\n",
        "            pass\n",
        "test_one_chunk_is_admitted_and_then_the_rule_returns",
    ),
    (
        "the ledger ignores the epoch -- one chunk per PROCESS, not per phase",
        PURITY,
        "    if epoch != _tp_phase_epoch(scheduler):\n        return 0\n",
        "    if False:\n        return 0\n",
        "test_the_budget_is_per_tp_phase_not_per_process",
    ),
    (
        "the mode question answers the round question (sizing/threshold widen)",
        PURITY,
        (
            "        return self.mode in (MODE_OFF, MODE_PREFILL_IN_TP)\n\n"
            + "    def prefill_allowed_in_tp_now"
        ),
        (
            "        return self.mode in (MODE_OFF, MODE_PREFILL_IN_TP) or (\n"
            + "            self.tp_compute_chunk_budget > 0\n"
            + "        )\n\n    def prefill_allowed_in_tp_now"
        ),
        "test_the_mode_question_and_the_round_question_stay_separate",
    ),
    (
        "the budget outranks the seam exemption -- a RESTORE spends a chunk",
        PURITY,
        "    if seam_transport_exempt(scheduler) and seam_transport_premise_holds(scheduler):\n        return False\n",
        "    if False:\n        return False\n",
        "test_a_restore_never_spends_the_compute_budget",
    ),
    (
        "an unresolved chunk size excuses any size ('no cap' read as 'no limit')",
        CONFORM,
        "    if cap <= 0:\n        return None\n",
        "    if cap <= 0:\n        cap = 1 << 60\n",
        "test_an_unknown_chunk_size_never_excuses_anything",
    ),
    (
        "the excuse is not restricted to prefill-in-tp -- decode in PP rides it",
        CONFORM,
        '    if batch_class != "prefill" or phase != "tp":\n        return None\n',
        "    if False:\n        return None\n",
        "test_decode_in_pp_is_never_excused_by_the_prefill_budget",
    ),
    (
        "every permitted chunk is reported as budgeted -- the #857 shape hides",
        CONFORM,
        '    budgeted = "yes" if int(budget_configured or 0) > 0 else "no"',
        '    budgeted = "yes"',
        "test_a_sub_chunk_batch_with_NO_valve_configured_is_named_as_such",
    ),
    (
        "a negative budget is accepted instead of refused",
        PURITY,
        "        if chunks < 0:\n            raise PhasePurityError(",
        "        if False:\n            raise PhasePurityError(",
        "test_a_malformed_budget_is_loud",
    ),
    (
        "the permitted chunk is silenced rather than named (#870's own gap)",
        CONFORM,
        "    _COUNTERS.tp_compute_exceptions += 1\n",
        "    pass\n",
        "test_the_permitted_sub_chunk_batch_is_not_a_violation_but_IS_NAMED",
    ),
    (
        "the gate grants a chunk its own detector will flag (two rules, not one)",
        PURITY,
        "        if purity.prefill_allowed_in_tp_now(spent) and tp_compute_fits_in_one_chunk(\n"
        + "            scheduler\n        ):",
        "        if purity.prefill_allowed_in_tp_now(spent):",
        "test_the_gate_refuses_when_MORE_than_a_chunk_is_pending",
    ),
    (
        "an unreadable pending count PERMITS instead of refusing",
        PURITY,
        "    return 0 < pending < chunk",
        "    return True",
        "test_an_unreadable_pending_count_refuses_rather_than_permits",
    ),
    (
        "a genuine RESTORE is counted as a compute exception (tpc inflated)",
        CONFORM,
        "    if transport_verified and int(cached_tokens) > 0:\n        return None\n"
        + "    cap = 0 if chunk_tokens is None else int(chunk_tokens)",
        "    cap = 0 if chunk_tokens is None else int(chunk_tokens)",
        "test_a_genuine_RESTORE_is_not_counted_as_a_compute_exception",
    ),
    (
        "the #858b runnability term ignores the budget -- valve unreachable",
        FLIP,
        "        if int(budget_remaining or 0) > 0:\n            return True\n",
        "        if False:\n            return True\n",
        "test_the_858b_runnability_term_sees_a_remaining_budget",
    ),
    (
        "describe() hides the budget -- the exception is invisible at boot",
        PURITY,
        'if self.mode == MODE_STRICT and self.tp_compute_chunk_budget > 0:\n            return f"{MODE_STRICT}:{self.tp_compute_chunk_budget}"',
        "if False:\n            return MODE_STRICT",
        "test_the_describe_string_names_the_budget",
    ),
]


def run_suite() -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(SUITE),
            "-q",
            "--tb=no",
            "-rf",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        # A NON-ZERO EXIT IS THE MEASUREMENT, not an error: a mutant is killed
        # precisely when this run fails. `check=True` would abort the harness
        # on its first success.
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    rc, out = run_suite()
    if rc != 0:
        print("REFUSED: the suite is not green BEFORE mutation. Fix that first.")
        print(out[-3000:])
        return 2

    killed, survived = [], []
    for name, path, find, repl, expect in MUTANTS:
        original = path.read_text()
        if find not in original:
            print(f"HARNESS BROKEN: anchor not found for {name!r} in {path.name}")
            return 2
        try:
            path.write_text(original.replace(find, repl, 1))
            rc, out = run_suite()
            if rc == 0:
                survived.append((name, expect, "the suite stayed GREEN"))
            elif expect not in out:
                # Killed, but not by the test that claims to own it: the
                # coverage claim is wrong even though the mutant died.
                survived.append((name, expect, "died, but NOT by its named test"))
            else:
                killed.append((name, expect))
        finally:
            path.write_text(original)

    print("=" * 72)
    for name, expect in killed:
        print(f"KILLED   {name}\n         by {expect}")
    for name, expect, why in survived:
        print(f"SURVIVED {name}\n         expected {expect} -- {why}")
    print("=" * 72)
    # THE COUNTING PROBE: the two halves must add up to the mutants attempted,
    # or the extraction is broken and neither number means anything.
    total = len(killed) + len(survived)
    assert total == len(MUTANTS), f"extraction lost mutants: {total} != {len(MUTANTS)}"
    print(f"{len(killed)}/{len(MUTANTS)} mutants killed, {len(survived)} survived")

    rc_after, _ = run_suite()
    print(f"suite restored and green: {rc_after == 0}")
    return 0 if (not survived and rc_after == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
