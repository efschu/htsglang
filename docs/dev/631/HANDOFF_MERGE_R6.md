# HANDOFF MERGE-R6 — three merges, zero conflicts, and a stale local ref that could have eaten R5

Shift `656-merge-r6`. Worktree `/spinning/wt-merge-r6`, branch `merge/r6-batch`,
based on `origin/feat/route-a-631` at `598f570ba4` (the MERGE-R5 tip both lines
carried). Frozen baseline worktree: `/spinning/wt-merge-r6-base`, detached at
`598f570ba4`, clean tree. Evidence and logs: `/spinning/evidence-631/merge-r6/`.

Last **merge** tip on both lines: **`fea62d371a`**, `ls-remote` verified after
every one of the three pushes. The commit carrying this document sits directly
on top of it and is the actual branch tip; it touches nothing but this file, so
the suite numbers below are the numbers for the shipped tree.

ERRORS FIRST.

---

## 1. The local `feat/route-a-631` ref is TWO merge rounds stale, and it is checked out

This is the one thing on this shift that could have silently destroyed work.

At shift start:

| ref | SHA |
|---|---|
| `origin/feat/route-a-631` | `598f570ba4` (MERGE-R5 tip) |
| `origin/integration/r2` | `598f570ba4` |
| **local** `feat/route-a-631` | **`0ae49fafb4`** — the MERGE-**R4** base |
| **local** `integration/r2` | **`0ae49fafb4`** |

`0ae49fafb4` is an ancestor of `598f570ba4`, so nothing is corrupt — but a merge
run as `git checkout feat/route-a-631 && git merge <branch>` would have built the
R6 batch **on the R4 base**, and the result would have been a tip missing the
entire R5 batch (the #647 completion fix, #695, and the three ops-hardening
commits) while looking perfectly healthy: it would merge cleanly, the suite would
pass, and the push would be a fast-forward-refused or, worse, look sane on a
force. The R5 handoff does not warn about this because R5 left the refs behind
itself.

What I did instead, and what R7 should keep doing:

* every merge source was named as `origin/<branch>`, never the local ref;
* every push was `git push origin HEAD:refs/heads/<line>`, so the stale local
  refs were never an input to anything;
* `git ls-remote` after each push, compared against `git rev-parse HEAD`.

`local integration/r2` is **now fast-forwarded** to `fea62d371a` (it was checked
out in no worktree, so this was free).

**`local feat/route-a-631` is deliberately NOT updated**: it is checked out at
`0ae49fafb4` in `/spinning/wt-631-routea`, which belongs to another strand. Git
refuses to move a ref that is checked out elsewhere, and forcing it would move
another session's working tree out from under it. **So the trap is still armed
for the next shift.** Either that worktree's owner updates it, or R7 keeps using
`origin/` refs exclusively. Do not "fix" it by force.

## 2. R5's own flip-family log was NOT run with `--color=no`, despite R5 naming that trap

`/spinning/evidence-631/merge-r5/flipfamily_final.log` contains raw ANSI escapes
(`\e[32m1116 passed\e[0m`). MERGE-R5's handoff explicitly warns about extracting
counts from coloured output, and its `run_arm.sh` does pass `--color=no` — but
the flip-family arm went through `scripts/run_631_flip_family.sh` without it.
The count was read correctly anyway (it was read by eye), so nothing downstream
is wrong. It is recorded because the guard was present in one runner and absent
in its sibling, which is the same shape as the R5 §1 defect (`hostmem_sample.sh`
knew the truncated `comm`, the #695 script did not).

R6's runner passes `--color=no` on **both** paths —
`/spinning/evidence-631/merge-r6/run_flip.sh` forwards it into the canonical
script, and `run_arm.sh` carries it per directory.

## 3. A pass count that did not move, which is the kind that gets misread

`unit/turnkey` is **116 passed on the baseline and 116 passed after step 3**,
even though `chore/release-chain-prep` adds a `REFUSE_WHEEL_DIST_SHADOW` case to
that directory's test file. That reads exactly like an under-collection.

It is not. The new case is a row in the `cases` list inside
`TestPreflight::test_every_failure_mode_is_reachable_and_named`, which iterates
with a plain `for` loop and `assertIn(want, names)` per row — no `subTest`, so
the row asserts without contributing a test id. Verified positively rather than
assumed: that single test was run alone (`1 passed`), and its loop would fail on
the new row if the refusal did not fire.

Recorded because "the count did not move" is the signature this tree has been
burned by three times (the three under-collections named in
`run_631_flip_family.sh`'s own header), and here it is the benign version.

## 4. Nothing on GPU, on purpose

Every suite run used `CUDA_VISIBLE_DEVICES=99`. RES-R5 §2 establishes that the
ship config tolerates **under ~300 MiB** of foreign residency on the 5090, and
that a pytest co-tenant taking ~500 MiB is a *boot blocker*, not a nuisance —
and it names `/spinning/wt-363-stages` (one of this shift's source branches) as
the observed offender. A merge shift that ran its suites on real cards would be
the thing that blocks the next boot. Serving on 30030 was not touched, no GPU
arbitration window was claimed, port 30099 was not touched, no `pkill` was used,
and `git stash` was never invoked.

---

## 5. WHAT MERGED

Order as briefed: residuals, then #363, then release prep. **Each step's suite
was green and pushed to both lines before the next merge was started**; nothing
was batched.

A path-overlap check across all three source branches ran **before** the first
merge and returned **empty for all three pairs** — 8 + 11 + 14 = 33 distinct
files, no file touched by two branches. That prediction held: **all three merges
were conflict-free.**

| step | source | at | resulting tip | conflicts |
|---|---|---|---|---|
| 1 | `val/r5-residuals` | `21f07330b2` | `21f07330b2` (fast-forward) | **0** |
| 2 | `feat/regime-stage-actuator-363` | `b2f0a749ac` | `c3919fe1cb` (`--no-ff`) | **0** |
| 3 | `chore/release-chain-prep` | `ca15df759e` | **`fea62d371a`** (`--no-ff`) | **0** |

Step 1 was a genuine fast-forward: `val/r5-residuals` was cut from `598f570ba4`
and nothing had landed on the line since, so no merge commit exists for it and
its own commit message carries the record. Steps 2 and 3 are `--no-ff` merge
commits whose messages carry the suite tables.

Total delta against `598f570ba4`: **33 files, 6528 insertions, 10 deletions**
(the deletions are the `test_regime_observe.py` extension and the `preflight.py`
signature change, nothing removed wholesale). Author on every
commit `efschu <efschu@users.noreply.github.com>`, no trailers, no
`Co-Authored-By`.

### What each step carries

**Step 1 — `val/r5-residuals`** (residual shift close-out). The #695 exact-size
pin verdict (C25: same-harness A/B, one md5-frozen tree, arms differing by
`SGLANG_PHASE_FLIP_EXACT_PIN` alone — the pin is **57.4 ms faster at p50**, not
slower, which retires MERGE-R5 §6's open question), the #644 allocator verdict,
the 40,12,12 spot check, `mem_ledger/host_anon_644.py`, and the
`test_exact_pin_opt_out_695.py` opt-out pin.

**Step 2 — `feat/regime-stage-actuator-363`** (#363 intra-phase stage actuator).
`managers/regime_ms_clock.py` (ms/round decision loop, group-reduced
compute/wait split) and `managers/regime_admission.py` (stage-flip pricing +
corridor admission). **`--regime-stage-clock` defaults `False`** — verified at
`server_args.py:5340` — so the default path is unchanged. The branch's own
handoff labels all of it **desk code: nothing here has run on a GPU**, and its
measurement ticket `docs/dev/363/TICKET_363_STAGE_CLOCK.md` is written and NOT
run. It merges as a gated-off addition, not as a validated feature.

**Step 3 — `chore/release-chain-prep`** (2 commits). #384 build-time and
preflight guards against the `sgl_kernel` wheel shadow
(`utils/kernel_dist_guard.py`, `REFUSE_WHEEL_DIST_SHADOW`),
`docs/dev/RELEASE_CHECKLIST.md`, `deploy/release/nccl-tuning.env`, the
`docker/kernel-wheel` staging directory, and four AUDIT-251 flags closed. Its
handoff's §1.1 stands unaltered by this merge: **the Docker build gate has never
run inside a real `docker build`** — no image was built by that branch and none
by this shift.

---

## 6. SUITE — every failure count identical to baseline, at every step

Baseline is the frozen worktree `/spinning/wt-merge-r6-base` at `598f570ba4`,
same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`), same
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`.
One directory per pytest invocation so a truncation is isolated to its directory.

| suite | BASE `598f570ba4` | after step 1 | after step 2 | after step 3 |
|---|---|---|---|---|
| #631 flip family (canonical script) | **1116 passed** | 1116 passed | 1116 passed | **1116 passed** |
| `unit/managers` | 9F 1286P 18S | 9F 1286P 18S | 9F **1357P** 18S | 9F 1357P 18S |
| `unit/mem_ledger` | 1F 343P | 1F **359P** | 1F 359P | 1F 359P |
| `unit/model_executor` | 15F 588P | 15F **594P** | 15F 594P | 15F 594P |
| `unit/server_args` | 615P | 615P | 615P | 615P |
| `unit/turnkey` | 116P | 116P | 116P | 116P (see §3) |
| `unit/utils` | 46F 331P 4S | 46F 331P 4S | 46F 331P 4S | 46F **348P** 4S |
| `unit/docker` | *(directory absent)* | — | — | **4 passed** |

**Every failure count is identical on both sides at every step.** The pass
deltas are exactly the new tests and land in exactly the step that introduces
them: +16 `mem_ledger` and +6 `model_executor` at step 1 (22 = the two new
residual test files), +71 `managers` at step 2 (the three new #363 test files
plus the `test_regime_observe.py` extension), +17 `utils` and +4 `docker` at
step 3 (the #384 guard and entrypoint pins). **Total +114 tests, 0 new
failures.**

The pre-existing failure sets are inherited, not introduced: 46 in `unit/utils`,
15 in `unit/model_executor`, 9 in `unit/managers`, 1 in `unit/mem_ledger`
(`test_communicator_group_contract_612.py`). They are identical file-for-file on
the frozen baseline and are out of scope for a merge shift.

### Lint

`ruff` over all 21 touched `.py` files: **456 errors before, 456 after**, and
the per-file breakdown is identical (`server_args.py` 357, `scheduler.py` 94,
`model_runner.py` 5). **Zero new** — and every new module (`kernel_dist_guard.py`,
`regime_ms_clock.py`, `regime_admission.py`, `host_anon_644.py`, all new test
files) is ruff-clean, contributing nothing to the count.

`codespell` over touched `.py`/`.md`/`.sh`/`.env`: 4 hits, **all four proven
pre-existing** by running the same files on the frozen baseline and getting the
identical set at the same lines. One of them (`schedul` in
`CONTRADICTIONS_REGISTER.md:1771`) is the deliberate 15-character
`TASK_COMM_LEN` truncation from MERGE-R5 §1 and **must not be "fixed"**.

---

## 7. REGISTER UNION — verified, nothing lost

`docs/dev/631/CONTRADICTIONS_REGISTER.md` is touched by exactly **one** of the
three branches (`val/r5-residuals`); the other two do not touch it, so there was
no union to reconcile and no opportunity for a last-writer-wins loss.

Verified rather than assumed, across the whole R6 range `598f570ba4..fea62d371a`:

| check | result |
|---|---|
| deleted lines in the register | **0** (`git diff --numstat` reports `63  0`) |
| added lines | **63** (48 of them non-blank) |
| file length | 1803 → **1866** |
| `### C<n>` entry headings | 7 → **9** |

Append-only, monotone, every pre-existing entry still present — the seven
pre-existing `C` headings are byte-identical and in the same order. The two new
entries are **C25** (the #695 exact-size pin verdict, which supersedes
MERGE-R5 §6) and **C26** (#644's residual ~16 GB is untrimmed allocator arena,
not retention — which closes MERGE-R5 §8 item 3).

---

## 8. STATE AT HANDOVER

- **Both lines at the same SHA** — `fea62d371a` after the third merge, plus the
  handoff commit carrying this file on top — `ls-remote`-verified against local
  `HEAD` after every push. Pushed to **`origin` = the efschu fork only**;
  `upstream` was never a push target.
- Working branch `merge/r6-batch` in `/spinning/wt-merge-r6` — same SHA, kept.
- Frozen baseline `/spinning/wt-merge-r6-base` at `598f570ba4`, detached, clean —
  **kept deliberately** so R7 can diff against the same tree R6 measured against.
- **Serving, GPUs, arbitration, port 30099: untouched.** No boot, no window, no
  `pkill`, no `git stash`. Nothing under `/etc` modified.
- `local integration/r2` fast-forwarded to the tip. `local feat/route-a-631`
  still at `0ae49fafb4` — see §1, this is on purpose.

## 9. REMAINING UNMERGED BRANCHES

78 local branches are not merged into the tip. The ones that are recent enough
to be live work, newest first:

| date | SHA | branch |
|---|---|---|
| 2026-08-12 | `d38bb6df32` | `trial/cumulative` |
| 2026-08-09 | `982b6434ce` | `feat/route-a-631-resume-gate` |
| 2026-08-09 | `00a1c50fcb` | `feat/gguf-q4-bringup-651` |
| 2026-08-08 | `27f3bf7996` | `fix/collective-stream-622` |
| 2026-08-08 | `18370879e3` | `integration/r3-probe-next2` |
| 2026-08-07 | `b851df7626` | `feat/dual-group-631` |

(`backup/pre-email-fix-s13` and `backup/pre-deps-strip-s13`, both 2026-08-09, are
backups and are not merge candidates.) The three standing Claude strands own
`feat/gguf-q4-bringup-651` (#651), `fix/collective-stream-622` (#622/#649) and
the Route-A line; none of them was asked to be merged this shift, and none was.

## 10. NEXT, IN ORDER

1. **#363 is desk code on the line now.** It is gated off, so it costs nothing
   at runtime, but the axis is unvalidated until
   `docs/dev/363/TICKET_363_STAGE_CLOCK.md` is run — specifically pre-step P2,
   the A-vs-A band measurement that decides whether the 5 % enter watermark sits
   above this rig's noise floor. If P2 comes back above 5 %, the axis simply
   never flips, which is the safe direction, but "never flips" and "works" look
   identical from outside.
2. **First real `docker build` is the #384 gate's first test** (release-prep
   handoff §1.1). If layer 3a fails, check ARG scope and the `purelib` path
   before suspecting the detector.
3. **Resolve the stale `feat/route-a-631` local ref** (§1) — needs the owner of
   `/spinning/wt-631-routea`, not a force from a merge shift.
4. Carried unchanged from MERGE-R5 §8, none of it addressed by this shift:
   label the #695 census lines with a PP-unique rank identity (MERGE-R5 §4);
   retire `route_a_631_prod_boot.sh` in favour of the turnkey unit path (its
   argv still diverges from the ship capture in seven flags); VAL-R4's ticket 4
   (`--pp-solve-cut` recommendable arm); the `--deterministic-hetero` /
   `--chunked-prefill-size` ergonomics refusal.
5. RES-R5 §2's boot-blocker threshold: `s33_boot_from_capture.sh` waits on a flat
   2000 MiB while the ship budget leaves ~289 MiB of slack on the 5090. Booked
   by RES-R5, still unfixed, and it will keep costing three-minute boots.
