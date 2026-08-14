# HANDOFF MERGE-R12 — the ratchet fires, one round after it started running

Shift `485-metal-r12` (phase 1). Worktree `/spinning/wt-merge-r12`, branch
`merge/r12-batch`, based on `origin/feat/route-a-631` at `95e2e0eb0e` (the tip
both lines carried). Frozen pre-merge baseline `/spinning/wt-merge-r12-base`,
detached at `95e2e0eb0e`, clean tree. Evidence:
`/spinning/evidence-631/merge-r12/`.

Merged: `feat/desk-485-excursion` at `f59103a370` — the planner gate
`_pp_cut_seam_staging` fed from the transient census, the refusal-without-term
in `server_args`, the cutover's per-direction `note_free()` seam load state, the
census `alloc=`/`res=` columns, `blocking_guards` earned/unanimous/bounded
retirement (driver deliberately unwired), and the reconciled 485 documents.

**Both lines are at the same SHA**, `ls-remote` verified against
`git rev-parse HEAD` after every push. As in R7–R11, this handoff cannot name
the commit that contains it: the actual tip is the docs-only commit sitting
directly on top of the SHA named in §7.

ERRORS FIRST.

---

## 1. THE BLACK RATCHET CAUGHT SOMETHING, AND THAT IS THE WHOLE POINT

`test_black_ratchet_656.py` went **red on the merge tip**:
`python/sglang/srt/managers/phase_flip_runtime.py` was not clean under the
pinned `black` 26.1.0. The merged branch grew that file by 233 lines (the
`blocking_guards` retirement) and left two expressions outside the formatter.

**This is the first time that ratchet has ever caught anything**, and the reason
it could is MERGE-R11. R11 §2 found the file ran in **no arm, in any sweep,
ever**, and fixed it in-batch rather than deferring. Had it deferred, this dirt
would have merged silently — and MERGE-R9 had already found once that the
pinned `black` had stopped running at all. A guard that has never fired is
indistinguishable from a guard that cannot fire; this one is now neither.

Fixed red-first on `feat/485-fixes` at `9ecb2f6458`:

| evidence | result |
|---|---|
| family arm at the merge tip `f8509c8c62` | **1 failed**, 1344 passed, 7 skipped |
| the same test standalone at `f8509c8c62` | **1 failed**, 2 passed |
| `black --version` in the run interpreter | **26.1.0**, verified to BE the pin the ratchet requires |
| after `9ecb2f6458` | **3 passed** |
| family arm at the post-fix tip `6d89418854` | **1345 passed, 7 skipped, rc=0** |

The change is 5 insertions / 5 deletions of line breaks in two expressions. No
token other than whitespace moves, which is why the M1 metal boot that had
already started against the unformatted tree measures the same code.

## 2. THE BRANCH ADDED TESTS WITHOUT ADDING COVERAGE FOR THEM

`feat/desk-485-excursion` lands two regression tests:

* `test/registered/unit/managers/test_seam_cap_retire_485.py` — inside the
  `managers` arm. Covered.
* `test/registered/unit/planner/test_seam_staging_producer_485.py` — inside
  `test/registered/unit/planner/`, which is named by **no arm** in the canonical
  set and by **no line** of `run_631_flip_family.sh`. `scripts/run_631_flip_family.sh`
  is byte-for-byte the same *length* on the branch as on the base, so the branch
  shipped tests and shipped no sweep for them.

That second file is the only thing in the tree that pins the fed gate: it proves
the gate **ADMITS** `40,12,12` at seam 0.0 and **REFUSES** it at the measured
5800. If the gate ever stopped reading the census, that is the test that goes
red, and nothing would have run it.

This is the R11 shape one directory over — the **eighth consecutive round** with
a short family list. Fixed in this batch, not deferred:

* `fdf02b0acc`'s successor `f8509c8c62` adds **both** files to the family script.
* `unit/planner` is added as a **ninth arm** on **both** sides, so the directory
  stops being invisible as a whole rather than one file at a time.

**Attribution, measured not asserted.** Family 1309P 7S → **1345P 7S**, and the
+36 is exactly **16 + 20** — the two files' own collected counts, run
individually. The 20 reproduces the branch's own desk validation for
`test_seam_staging_producer_485.py` exactly.

The `#363` verdict addendum at `95e2e0eb0e` had already recorded `unit/planner`
as a non-arm and run it explicitly at **2366P 123S with 2 failures**. This
shift's base column reproduces that number independently — see §4.

## 3. AN INSTRUMENT CAVEAT: THE FAMILY LOG IS ANSI-COLOURED

`run_631_flip_family.sh` does not pass `--color=no` by default, so its log
contains `\x1b[31mFAILED\x1b[0m`. A `^(FAILED|ERROR|SUBFAILED)` grep — the one
the arm runner uses for every *other* arm — matches **zero lines** in it. The
family red in §1 was caught by the script's **return code**, not by the name
grep, and a shift that trusted the name grep alone would have recorded the
family as having no failures.

This is R11 §1's ruff-ANSI trap in a second location, and it is recorded rather
than silently worked around: the step-2 re-run in §4 passes `--color=no`
explicitly. Whoever next edits the arm runner should decide whether the family
arm's names are worth collecting at all, or whether rc is the honest key for it.

## 4. SUITE — ten arms, both sides, failure sets diffed by NAME

Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation. Runner `/spinning/evidence-631/merge-r12/arms.sh`,
which differs from R11's in exactly one documented way: `planner` is an arm, on
both sides.

| suite | BASE `95e2e0eb0e` | step 1 (merge+sweep) `f8509c8c62` | step 2 (fix) `6d89418854` |
|---|---|---|---|
| #631 flip family | **1309P 7S** | **1F 1344P 7S** (§1) | **1345P 7S** |
| `unit/managers` | 9F 1451P 18S | 9F **1467P** 18S | — |
| `unit/mem_cache` | 940F 748P 707S | 940F 748P 707S | — |
| `unit/mem_ledger` | 444P | 444P | — |
| `unit/model_executor` | 15F 594P | 15F 594P | — |
| `unit/planner` (**new arm**) | **2F 2366P 123S** | 2F **2386P** 123S | — |
| `unit/server_args` | 624P | 624P | — |
| `unit/turnkey` | 116P | 116P | — |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | — |
| `unit/docker` | 4P | 4P | — |

**Continuity.** The BASE column reproduces R11's post-merge column **exactly**
on all nine of R11's arms — family 1309P 7S, `managers` 9F 1451P 18S,
`mem_cache` 940F 748P 707S, `mem_ledger` 444P, `model_executor` 15F 594P,
`server_args` 624P, `turnkey` 116P, `utils` 46F 348P 4S, `docker` 4P. The tenth
arm's base, `planner` 2F 2366P 123S, reproduces the number the `#363` addendum
recorded from an independent explicit run. The chain back through R11, R10, R9,
R8, R7 and R6's frozen bases is unbroken.

**Failure SETS diffed by name — all ten identical, zero new failures:**

| arm | names at base | at step 1 | diff |
|---|---|---|---|
| `managers` | 10 | 10 | **identical** |
| `mem_cache` | 940 | 940 | **identical** |
| `model_executor` | 15 | 15 | **identical** |
| `planner` | 2 | 2 | **identical** |
| `utils` | 46 | 46 | **identical** |
| `mem_ledger` / `server_args` / `turnkey` / `docker` | 0 | 0 | identical |

**The SUBFAILED second pass re-measured, not assumed.** Prefix census on this
base: `managers` 10 = 8 `FAILED` + 1 `ERROR` + 1 `SUBFAILED`; `utils` 46 = 44
`FAILED` + 2 `SUBFAILED`; the new `planner` arm's 2 are both plain `FAILED`. A
`^FAILED`-only grep would find 8 of 10 and 44 of 46 — the same two arms and the
same two counts R10 and R11 recorded.

**Predicted vs delivered deltas.** The shift brief predicted `server_args`
+tests, `managers` +tests and new *scheduler* tests. Delivered: `managers`
**+16**, `planner` **+20**, family **+36**, and `server_args` **unchanged at
624P**. The branch's 68 new lines in `server_args.py` are the fed gate and its
refusal, and they ship with **no test in the `server_args` arm** — their
coverage lives in `unit/planner`, which is exactly why §2 mattered. Recorded
because a prediction that missed is worth more than one quietly dropped.

**+36 tests, 0 new failures, 1 red found and fixed.**

## 5. REGISTER — untouched by this merge, and verified so

| check | result |
|---|---|
| `git diff --numstat` on `CONTRADICTIONS_REGISTER.md`, base→tip | **empty** — the file is not in the merge at all |
| deleted lines | **0** |
| file length | **2856**, unchanged from R11's tip |
| `C605-1`…`C605-17` occurrence counts | **identical at base and tip**, byte for byte |
| row count | **39** rows, `## N.` form |
| duplicate row labels | **none** |

The union-merge check is trivially satisfied here because the branch adds no
register rows. It was still run rather than skipped, and the duplicate-row grep
was **proven able to fail** first, per R11 §1: appending a second `## 84.` to a
copy makes it report `## 84.`, so the clean result on the real file means
something.

## 6. STATE AT HANDOVER

- **Both lines at the same SHA** — see §7's push record; `ls-remote`-verified
  against local `HEAD` after every push.
- Working branch `merge/r12-batch` in `/spinning/wt-merge-r12` — same SHA, kept.
- Fix branch `feat/485-fixes` at `9ecb2f6458`, merged, kept.
- Feature branch `feat/desk-485-excursion` at `f59103a370`, kept, unmodified.
- New frozen baseline `/spinning/wt-merge-r12-base` at `95e2e0eb0e`, detached,
  clean — kept, so R13 can diff against the tree R12 measured against. R11's,
  R10's, R9's, R8's, R7's and R6's frozen bases are also still there.
- Phase 1 touched no GPU: every arm ran `CUDA_VISIBLE_DEVICES=99`, in transient
  systemd scopes (`r12-base-arms`, `r12-tip-arms`, `r12-tip2-family`) in
  `claudework.slice`, outside `claude.service`. The GPU work of this shift is
  the #485 metal ticket and is documented separately.
- Port 30099 never touched as a process. No `pkill`. `git stash` never invoked.
  Pushed to `origin` = the efschu fork only.

## 7. WHAT THE NEXT SHIFT SHOULD PICK UP

1. **`unit/mem_cache` still needs a device arm.** 940 of its 1688 collected
   tests are accelerator-gated and cannot be judged on a CPU desk. Unchanged
   from R11 §8.1, and now two rounds old.
2. **`test/registered/unit/` root is still swept by nothing** except the two
   files the family script names by hand. ~35 more `.py` files there run in no
   arm. R11 §8.2, still open.
3. **The family arm's ANSI log** (§3) — decide whether it collects names or
   only an rc.
4. **`server_args`'s fed gate has no test in its own arm** (§4). The coverage is
   real but it lives one directory away, and the arm that carries the flag
   parsing would not notice its removal.
5. R10 §11 remains open: the union repair has never run on metal, and the ~280k
   band (register 81) remains unreproduced.
