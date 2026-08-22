# Integration note: 808 + 739 + 810 + 773-docs (#559 class)

Branch `integ/808-739-810`, built from `fix/808-flip-staging-funding @ 587e4c2e39`
in a fresh worktree. The serving tree `/spinning/wt-808` was never touched.

## What was merged, and why this base

`fix/808-flip-staging-funding` already carries the whole
`integ/800-802-presence` chain and the seven-commit #773 mamba-floor chain.
Verified with `git merge-base --is-ancestor`, not by branch name:

    98f8f790eb (integ/800-802-presence)      in 808: YES
    689161de77 (#802 floor fix)              in 808: YES
    703b05c723 / 1b323d6bcc / 12f42498d2     in 808: YES   (#773 floor chain)
    292eceee70 (feat/739-wedge-class)        in 808: NO    -> merged
    bfabd2d933 + 66cf0f68d8 (feat/810)       in 808: NO    -> merged
    d128a5fe2d (#773 docs)                   in 808: NO    -> merged

Merges, in order, all clean (no conflicts, no manual resolution):

    8f997516a9  Merge feat/739-wedge-class          (4 files, +472)
    7e1e189cf4  Merge feat/810-hicache-staging-ring (5 files, +629)
    4cd9990b70  Merge fix/773-mamba-slot-vacate     (1 file docs, +217)

Total diff against 587e4c2e39 is exactly 10 files. The only file shared
between a merged line and the 808 side is `invariant_checker.py`, which 808
does not touch since the merge base — hence the clean merge.

`feat/810` is taken at today's head (bfabd2d933 + 66cf0f68d8). The
backpressure ring still in progress on that branch is a follow-up merge, not
part of this one.

## Test results

All runs hermetic: `CUDA_VISIBLE_DEVICES=""`,
`PYTHONPATH=<worktree>/python`, project venv.

Battery: the four registered unit directories that contain every suite the
merged lines were measured green on — `test/registered/unit/{managers,
planner,server_args,mem_cache}` (548 files). A superset was chosen over a
hand-picked file list so that no suite is silently dropped.

| Stage                      | Result                                   |
|----------------------------|------------------------------------------|
| Base 587e4c2e39 (clean)    | 61 failed, 8245 passed, 1852 skipped, 847s |
| After merge feat/739       | 61 failed, 8255 passed, 1852 skipped, 895s |
| After merge feat/810       | 61 failed, 8283 passed, 1852 skipped, 909s |
| Final (all three merged)   | 61 failed, 8283 passed, 1852 skipped, 850s |

The set of failing test ids is **byte-identical to the base run** at every
stage (`diff` of the sorted FAILED lists is empty). The 61 failures are
pre-existing on a clean 587e4c2e39 checkout under `CUDA_VISIBLE_DEVICES=""`
and are not claimed to be fixed or caused here; they concentrate in
`test_session_branch_rewind_unit.py` (7), `test_pp_flip_slot_hold_631.py` (7),
`test_kv_arena_handle_retention_631.py` (7) and `test_vacuous_decode_exit_730.py` (6).
The full base list is the comparator, not an estimate.

Passed count rises by exactly the tests the merges bring: +10 from
`test_wedge_class_739.py`, +28 from the two #810 suites.

Pre-merge branch-green run of the #810 suites on their own head 66cf0f68d8
(they were a foreign claim, not measured by the merger's predecessor):
28 passed.

Mutation proofs on the merged, clean tree:

    test/registered/unit/managers/mutation_proof_739.py   ALL MUTANTS KILLED (M1-M5)
    test/registered/unit/managers/mutation_proof_800.py   ALL MUTATIONS KILLED (through M9)

Lint, using the gate the repo actually enforces
(`.pre-commit-config.yaml`: `ruff --select=F401,F821,UP037`):

    ruff       All checks passed
    codespell  no findings

## Scope

No GPU claim, no boot, no serving change. This branch is intended as the base
for follow-up work that was branched from 587e4c2e39 in parallel (#814).
