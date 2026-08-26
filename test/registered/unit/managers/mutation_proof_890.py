#!/usr/bin/env python3
"""#890 CAN-FAIL PROOF: deliberately break the revocation, prove the suite reds.

A test that passes in both worlds measures nothing. Each mutant below is a
plausible wrong implementation of "the seam-transport exemption is revoked when
the restore it was granted for is refused" -- several of them are the reading
somebody would reach for FIRST, and two of them ARE the pre-fix code -- and the
run is a pass only when every one is KILLED by a named test.

BOTH DIRECTIONS ARE MUTATED, on purpose. A revocation that is too WIDE is not a
safer error than one that is too narrow: it disarms the W30 exemption and puts
back the livelock that measured 150 flips in 17 minutes with zero decode
batches. So the blanket-disarm mutants (5, 6) matter as much as the
does-nothing mutants (1, 2, 3).

Not a pytest module on purpose: it rewrites source files, so it must never be
collected into a normal run. Invoke it directly:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
        python3 test/registered/unit/managers/mutation_proof_890.py

THE EXTRACTION IS COUNTED AGAINST THE SUMMARY, deliberately: a killed-count
derived from parsing that disagrees with the number of mutants attempted means
the HARNESS is broken, and a 0 for the wrong reason is indistinguishable from
green. The verdict is computed from pytest's exit status, not from scraping
ANSI-coloured output.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
SRC = ROOT / "python" / "sglang" / "srt" / "managers"
TESTS = ROOT / "test" / "registered" / "unit" / "managers"

PURITY = SRC / "phase_purity.py"
BATCH = SRC / "schedule_batch.py"
SCHED = SRC / "scheduler.py"
SUITE = TESTS / "test_seam_restore_revocation_890.py"

#: (name, file, find, replace, the test that must die)
MUTANTS = [
    (
        "THE PRE-FIX CODE: the premise never asks what the restore did",
        PURITY,
        "            if getattr(req, SEAM_RESTORE_REFUSED_ATTR, False):\n"
        "                revoked += 1\n                continue\n",
        "            if False:\n                revoked += 1\n                continue\n",
        "test_the_premise_no_longer_holds_for_a_refused_request",
    ),
    (
        "the LAYOUT refusal drops the copy but leaves the permission standing",
        BATCH,
        "        # the pool CLASS, which is likewise the same on every rank.\n"
        "        req.seam_restore_refused = True\n",
        "        # the pool CLASS, which is likewise the same on every rank.\n",
        "test_a_layout_refusal_marks_the_request",
    ),
    (
        "the EXTENT refusal is left unrevoked (only the W38 axis was fixed)",
        BATCH,
        "        # replicated across the group.\n"
        "        req.seam_restore_refused = True\n",
        "        # replicated across the group.\n",
        "test_an_extent_refusal_marks_the_request",
    ),
    (
        "the revocation is a LIFE SENTENCE -- a working restore never clears it",
        BATCH,
        "    req.seam_restore_refused = False\n"
        '    _SEAM_STATE_COUNTS["restored"] += 1',
        '    _SEAM_STATE_COUNTS["restored"] += 1',
        "test_a_successful_restore_clears_an_earlier_refusal",
    ),
    (
        "BLANKET DISARM: one refused request revokes the whole population",
        PURITY,
        "                revoked += 1\n                continue\n",
        "                return False\n",
        "test_one_refused_request_does_not_revoke_the_others",
    ),
    (
        "BLANKET DISARM 2: the no-copy path is marked, so every request loses it",
        BATCH,
        '    saved = getattr(req, "kv_cache_cpu", None)\n    if saved is None:\n        return False',
        '    saved = getattr(req, "kv_cache_cpu", None)\n    if saved is None:\n'
        "        req.seam_restore_refused = True\n        return False",
        "test_a_request_that_never_carried_a_copy_is_not_marked",
    ),
    (
        "the gate takes the exemption without the premise (order restored, W37-D)",
        PURITY,
        "    if seam_transport_exempt(scheduler) and seam_transport_premise_holds(scheduler):\n        return False\n",
        "    if seam_transport_exempt(scheduler):\n        return False\n",
        "test_the_gate_closes_on_the_next_round_after_a_refused_restore",
    ),
    (
        "the refusal announcement latches -- the revocation is visible once ever",
        PURITY,
        '        if getattr(scheduler, "_seam_premise_refused_announced", False):\n'
        "            scheduler._seam_premise_refused_announced = False\n"
        "        return True\n",
        "        return True\n",
        "test_the_refusal_flag_re_arms_once_the_premise_holds_again",
    ),
    (
        "SIBLING: the hypothetical probe drops back to asking only the stamp",
        SCHED,
        "                if seam_readmit_candidates(self) and seam_transport_premise_holds(self):\n"
        "                    return True",
        "                if seam_readmit_candidates(self):\n                    return True",
        "test_a_stamped_population_whose_restore_was_REFUSED_is_not_admissible",
    ),
    # NOT A MUTANT, recorded so nobody adds it back: dropping
    # `seam_readmit_candidates(self)` and keeping only the premise is
    # SEMANTICALLY IDENTICAL -- `seam_transport_premise_holds` opens with
    # `reqs = seam_readmit_candidates(scheduler); if not reqs: return False`.
    # It was attempted here, survived, and survived correctly. A harness that
    # counted it as an escape would be reporting on its own bad mutant.
    (
        "SIBLING 2: the probe INVERTS the premise (a revoked population reads "
        "admissible and a live one does not)",
        SCHED,
        "                if seam_readmit_candidates(self) and seam_transport_premise_holds(self):\n"
        "                    return True",
        "                if seam_readmit_candidates(self) and not seam_transport_premise_holds(\n"
        "                    self\n                ):\n                    return True",
        "test_a_stamped_population_with_a_LIVE_premise_still_reads_admissible",
    ),
    (
        "the field is not declared on Req -- a dynamic attribute nobody can find",
        BATCH,
        "        self.seam_restore_refused = False\n",
        "        pass\n",
        "test_the_field_is_declared_on_req",
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
