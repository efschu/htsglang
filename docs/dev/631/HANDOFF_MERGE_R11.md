# HANDOFF MERGE-R11 — a file that ran in no arm, and an arm that ran no file

Shift `363-stage-clock` (phase 1). Worktree `/spinning/wt-merge-r11`, branch
`merge/r11-batch`, based on `origin/feat/route-a-631` at `2b71b5b242` (the
MERGE-R10 tip both lines carried). Frozen pre-merge baseline
`/spinning/wt-merge-r11-base`, detached at `2b71b5b242`, clean tree. Evidence:
`/spinning/evidence-631/merge-r11/`.

Merged: `feat/desk-hardening-656` at `660eb05bc6` — arena-tail additive fix,
NIXL collection unblock, contract-612, kvso spec note, mamba stub closure,
black ratchet + the fourteen formatted files.

**Both lines are at the same SHA**, `ls-remote` verified against
`git rev-parse HEAD` after every push. As in R7–R10, this handoff cannot name
the commit that contains it: the actual tip is the docs-only commit sitting
directly on top of the SHA named in §7.

ERRORS FIRST.

---

## 1. TWO INSTRUMENTS OF MINE REPORTED GREEN WHILE MATCHING NOTHING

Both were caught inside this shift, and both had already printed a reassuring
line before they were caught.

**The duplicate-row check on the register.** The canon requires "no duplicate
row labels anywhere in the file". The grep used was
`^### (Row |C)?[0-9]+`. The register's rows are `## 84. ...` — two hashes, and
a period. The pattern matched **zero lines in a 2856-line file** and the check
printed its success branch. It was rewritten to `^## [0-9]+\.` and then
**proven able to fail** by re-running it over the same file with a duplicate
`## 84.` appended, which it caught. Rows 52–90 are contiguous with no
duplicates — that statement is now worth something.

**The ruff rule census.** Ruff's `concise` output is ANSI-coloured even into a
pipe, so `grep -oE ': [A-Z]+[0-9]+ '` matched nothing on **both** sides and the
diff of two empty files printed `NO RUFF DELTA`. Two empty sets are always
equal. Fixed by stripping the escapes; the census is non-empty in §6 and the
comparison means something.

**The shape, and it is the register's own recurring one:** a check that cannot
fail has certified nothing, and *an empty result compared against an empty
result is the most convincing way to certify nothing*. Both of these printed
PASS. The rule is not "write careful greps" — it is **make the instrument fail
once, deliberately, before believing it**, which is exactly what the corpus
already demands of the test arms and had not been demanding of the merge
shift's own shell checks.

## 2. A TEST FILE THAT RAN IN NO ARM, IN ANY SWEEP, EVER

`test/registered/unit/test_black_ratchet_656.py` landed with the merged branch
at `d7cfdb4dd1`. **Every arm in the canonical set names a SUBDIRECTORY** of
`test/registered/unit/` — `managers`, `mem_ledger`, `model_executor`,
`server_args`, `turnkey`, `utils`, `docker`. Nothing collects the directory
ROOT. `scripts/run_631_flip_family.sh` did not list it either. There is no
repo-side suite definition that covers unit root: checked, and it does not
exist.

So the file ran **nowhere** from the moment it landed. Run by hand for the
first time in this shift: **3 passed**. It happened to be green. Nothing in
the process would have said so had it been red — and its whole purpose is to
be a ratchet that "fails loudly" on new formatting dirt.

This is the **seventh consecutive round** with a short family list (R10 §2 the
sixth, R9 §2 the fifth, R8 §§1–2 before that), and the first where the file
had no coverage at all rather than merely no canonical coverage. Fixed in this
batch, not deferred, per the standing rule: `fdf02b0acc` adds it plus the two
other flip-surface files the branch shipped. **Family 1268P 7S → 1309P 7S, and
the +41 is exactly 18 + 20 + 3** — the three files' own collected counts, so
the number is attributed rather than asserted.

**Note for whoever next edits the arm set:** `test/registered/unit/` root holds
~35 more `.py` files. They are outside this shift's merge and were not audited;
whether any of *them* run in some other sweep is an open question, and it is
the same question that has now been answered "no" once.

## 3. AN ARM THAT COLLECTED NOTHING, AND WHAT IT HID

`unit/mem_cache` was **uncollectable** on the base: `ERROR
test/registered/unit/mem_cache/test_hicache_nixl_storage.py`,
`Interrupted: 1 error during collection`. One optional backend import at module
scope aborted the whole directory. The merged branch's `5a4da87aa5` fixes it,
and the arm collects for the first time.

**It was never in the canonical arm list**, which is why the interruption was
invisible for as long as it lasted: R10's runner ran seven directories and this
was not one of them. Added as an arm here — on **both** sides, so the
transition is a measured delta and not an arm that simply appeared.

What it hides is the part that matters. Collected on a CPU desk the arm is
**940F 748P 707S**, and the failures were attributed rather than accepted:

| attribution | count |
|---|---|
| `RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform plugin is available.` | 939 |
| `RuntimeError: No CUDA GPUs are available` | 1 |
| **anything else** | **0** |

940 of 940 are accelerator-gated under the canon's `CUDA_VISIBLE_DEVICES=99`.
So this arm is green-by-construction nowhere and red-by-construction on every
CPU desk: **as a CPU arm it can never distinguish a regression from its own
device gate.** Until it is run inside a GPU window with a visible device, a
change to any of those 940 is unobserved by this suite. That is a strictly
better position than an uncollectable directory — 748 tests do now run — but
it is not coverage, and the next GPU window is where it should be settled.

## 4. SUITE — nine arms, both sides, failure sets diffed by NAME

Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation. Runner `/spinning/evidence-631/merge-r11/arms.sh`,
which differs from R10's in exactly two documented ways: the failure-name grep
matches `FAILED|ERROR|SUBFAILED`, and `mem_cache` is an arm.

| suite | BASE `2b71b5b242` | step 1 (merge) `53cc6bd6ed` | step 2 (sweep) `fdf02b0acc` |
|---|---|---|---|
| #631 flip family (canonical script) | **1268P 7S** | **1268P 7S** | **1309P 7S** |
| `unit/managers` | 9F 1433P 18S | 9F **1451P** 18S | — |
| `unit/mem_cache` | **uncollectable (1 error)** | **940F 748P 707S** | — |
| `unit/mem_ledger` | **1F** 437P | **0F 444P** | — |
| `unit/model_executor` | 15F 594P | 15F 594P | — |
| `unit/server_args` | 615P | **624P** | — |
| `unit/turnkey` | 116P | 116P | — |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | — |
| `unit/docker` | 4P | 4P | — |

Every predicted delta landed: `mem_ledger` 1F→0F, `managers` +18P, `server_args`
+9P, `mem_cache` uncollectable→collected.

**Continuity.** This shift's BASE column reproduces R10's post-step-3 column
**exactly** on all eight of R10's arms — flip family 1268P 7S, `managers`
9F 1433P 18S, `mem_ledger` 1F 437P, `model_executor` 15F 594P, `server_args`
615P, `turnkey` 116P, `utils` 46F 348P 4S, `docker` 4P. The chain back through
R10's, R9's, R8's, R7's and R6's frozen bases is unbroken.

**Failure SETS diffed by name:**

| arm | names at base | at step 1 | diff |
|---|---|---|---|
| `managers` | 10 | 10 | **identical** |
| `model_executor` | 15 | 15 | **identical** |
| `utils` | 46 | 46 | **identical** |
| `mem_ledger` | 1 | **0** | **one gone, none new** — `test_communicator_group_contract_612.py::TestTheDeclarationNamesEveryGroupTheRuntimeBuilds::test_no_runtime_group_is_missing_from_the_declaration` |
| `server_args` / `turnkey` / `docker` | 0 | 0 | identical |
| `mem_cache` | 1 (collection ERROR) | 940 | §3 |

**The SUBFAILED second pass is load-bearing and was re-measured, not assumed.**
Prefix census of the name files on this base:

* `managers` 10 = 8 `FAILED` + 1 `ERROR` + 1 `SUBFAILED(loop='event_loop_pp')`
* `utils` 46 = 44 `FAILED` + 2 `SUBFAILED(module=…gdn_cutedsl / …kda_cutedsl)`

A `^FAILED`-only grep finds **8 of 10 and 44 of 46**. R10 §6 recorded this for
these two arms; it still holds here, on the same two arms, with the same two
counts.

**+59 tests, 0 new failures**, across both steps: +18 `managers`, +9
`server_args`, +7 `mem_ledger` (the closed red plus its file's siblings), +41
family (§2 attribution), and 748 newly-*running* `mem_cache` passes on top.

## 5. REGISTER — union-merge verified, 0 deletions, `C605-*` intact

| check | result |
|---|---|
| deleted lines in `CONTRADICTIONS_REGISTER.md` | **0** (`git diff --numstat` reports `223  0`) |
| added lines | **223** |
| file length | 2633 → **2856** |
| `C605-1`…`C605-17` occurrence counts | **identical at base and at tip**, byte for byte |
| duplicate row labels | **none**, by a grep proven able to catch a planted one (§1) |

Rows added by the merged branch: **85** (arena tail additive), **86** (the
ledger declared no phase-flip communicator), **87** (the kvso-under-PP refusal),
**88** (a missing optional backend hid 2331 tests), **89** (the pinned `black`
was never installed), **90** (the four reds the collection fix exposed). Rows
52–90 contiguous, no duplicates. The merge itself conflicted in no file.

## 6. LINT — no new kind of debt, and the comparison is now split honestly

Comparing all 24 touched `.py` files across sides is invalid: 4 of them exist
only on the tip. Split:

| gate | base (20 common) | tip (20 common) | tip (4 new) |
|---|---|---|---|
| `ruff` | 102 E402, 3 E731, 2 E741, 356 F722, 3 F841 | **101 E402**, rest identical | 8 E402 |
| `codespell` | 8 | **8** | **0** |

One E402 fewer on the common files; the 8 on the new files are the same class
that already dominates the census, and **no new rule appears on either side**.
`black` is no longer a hand-gate on the flip surface — the ratchet is a test
now, and as of §2 it is a test that actually runs.

## 7. STATE AT HANDOVER

- **Both lines at the same SHA** — see §8's push record; `ls-remote`-verified
  against local `HEAD` after every push.
- Working branch `merge/r11-batch` in `/spinning/wt-merge-r11` — same SHA, kept.
- Feature branch `feat/desk-hardening-656` at `660eb05bc6`, kept, unmodified.
- New frozen baseline `/spinning/wt-merge-r11-base` at `2b71b5b242`, detached,
  clean — kept, so R12 can diff against the tree R11 measured against. R10's,
  R9's, R8's, R7's and R6's frozen bases are also still there.
- No GPU was touched by phase 1. Every arm ran `CUDA_VISIBLE_DEVICES=99`; both
  suites ran in their own transient systemd units (`r11-base-arms`,
  `r11-tip-arms`), cgroup-verified outside `claude.service`, per MERGE-R10 §9.
- Port 30099 never touched as a process. No `pkill`. `git stash` never invoked.
  Pushed to `origin` = the efschu fork only.

## 8. WHAT THE NEXT SHIFT SHOULD PICK UP

1. **`unit/mem_cache` needs a device arm** (§3). 940 of its tests cannot be
   judged on a CPU desk, and it is now in the canonical arm list where that
   limitation is at least visible.
2. **`test/registered/unit/` root is swept by nothing** (§2). One file is fixed;
   the directory still has ~35 more that no arm names.
3. Everything R10 §11 left open is still open: the union repair has never run
   on metal, `blocking_guards` is append-only at `DEFAULT_SEAM_ABANDON_CAP = 8`,
   and the ~280k band (register 81) remains unreproduced.
4. **Merge-shift shell checks deserve the can-fail discipline the test arms
   already have** (§1). Two of this shift's printed PASSes were vacuous.
