# HANDOFF MERGE-R4 — the approved seven-branch merge into the #631 line

Shift: `656-merge-r4`. Tree `/spinning/wt-merge-r4`, working branch
`merge/r4-batch`, pushed to `feat/route-a-631` and `integration/r2` in
lockstep after every step. Evidence: `/spinning/evidence-631/merge-r4/`.

All seven branches are merged, each with its own suite run, each pushed
before the next was started. Final tip **`75c86bf255`** on both branches.
Serving is UP on 30030 from the merged tree at that tip.

ERRORS FIRST.

## 1. The line's planner directory was 109-red, and nobody said so

The briefing named **two** pre-existing red tests
(`test_rejected_evidence_pins`) and asked me to confirm they stay exactly
two. They do. But the number that matters is the other one: at the
pre-merge baseline `0ae49fafb4`, `test/registered/unit/planner/` was

    109 failed, 2230 passed, 123 skipped, 3 errors

Not two. One hundred and nine, plus three collection errors. I found this
only because I took a baseline before merging step 7, and I could prove none
of it was mine only because I re-ran the identical suite in a **separate
worktree checked out at `0ae49fafb4`** and diffed the failure sets: 109 vs
109, `diff` empty, identical set. My five preceding merges are exactly
neutral on that directory.

**Step 7 then repaired 107 of them.** After merging
`feat/rig-advisor-413`, the same directory reads

    2 failed, 2416 passed, 123 skipped, 0 errors

and the two survivors are precisely the two named
`test_rejected_evidence_pins` cases. That is the briefing's "two
wizard-blocking planner fixes" (`runtime_reserve_mib` now binds, `library=`
is passed through) doing far more work than the phrase suggests: they were
also the cause of a hundred-odd unrelated-looking planner failures.

Two things follow, and the second is the uncomfortable one:

1. The advisor branch was not merely a dashboard feature. It carried the fix
   for a broad, standing red in the planner suite.
2. **A 109-red directory sat on the line across at least four shifts without
   appearing in any handoff.** The flip family is green (1116) and is what
   every shift runs, so the curated suite kept reporting health while an
   uncurated neighbour directory was a third failing. If a suite is the only
   thing looked at, everything outside it is unobserved by construction.
   Whoever inherits this should decide whether the planner directory joins
   the routine sweep — I did not add it, because changing the canonical
   family list mid-merge would have made my own step-to-step comparisons
   incomparable.

## 2. Two corrections I am carrying, both mine

**A search of mine returned "not found" and was wrong.** Asked to confirm
the two red pins, I first searched for the name in file *contents* and
reported that it did not exist in the tree. It does:
`test/registered/unit/planner/test_rejected_evidence_pins.py` — a file whose
*name* matches but whose body never spells the string I grepped for. A
name-shaped search must look at filenames as well as contents before "it is
not there" is reported. Had I not caught it, a known-red allowance would
have been quietly retired on my say-so.

**`feat/route-a-631` is checked out in the serving worktree, so its local
ref lags origin.** `/spinning/wt-631-routea` holds the branch; git will not
let a second worktree check it out, and force-updating a branch under a live
serving tree is not something to do casually. So every push in this shift
was `git push origin merge/r4-batch:feat/route-a-631` (and the same for
`integration/r2`) — a fast-forward on the remote, verified by SHA each time.
**Consequence for the next shift: `/spinning/wt-631-routea` will report
"behind"; the truth is on origin, and `merge/r4-batch` is the ref that
matches it.** Fetch before believing a local ref.

## 3. No conflict was resolved, because none occurred

The briefing predicted a conflict zone at steps 1-2 (triton/flashinfer
backends against the line's #623 census wiring) and a second at step 7 (the
advisor's planner fixes against N50/N51's planner work). **Neither
materialised** — not in the seven individual trial merges, not in the
cumulative trial in the approved order, and not in the real thing. Zero
conflicted files at every step. That is worth stating precisely, because a
clean merge is the easiest place to hide an untested assumption:

- Steps 1-2 and the line touch **disjoint file sets** since merge base
  `c2ceac7f31`. The line's #623 work landed in `planner/`, `managers/`,
  `utils/`; the branches' mirror wiring landed in `layers/attention/` and
  `layers/dcp/`. The predicted collision was a collision of *topics*, not of
  files.
- Step 7 touches `planner/plan.py` and `planner/feasibility.py`. N51's three
  commits touch `planner/pp_cut.py`, `planner/pp_cut_calibration.py`,
  `planner/transient_census.py`. Also disjoint. "The line's residency census
  is newer, keep it" never had to be applied, because nothing proposed to
  overwrite it.

The only file touched by both the line and any branch is
`python/sglang/srt/server_args.py` (step 5), and the reason its auto-merge is
safe was checked rather than assumed: the line adds `pp_attn_stage_ratio`
and `pp_solve_cut` near line 1472; step 5 adds `deterministic_hetero` near
5650 with its handler near 15443. Different regions of a 15k-line file, no
shared symbol, no shared call site. `test/registered/unit/server_args/` is
green after the merge.

## 4. The seven steps

Every step: merge, run `scripts/run_631_flip_family.sh` plus the accumulated
touched directories, both green **before** the next merge, then push both
branches and verify the remote SHA. Never batch-merged, never tested once.

Baseline at `0ae49fafb4`: flip family **1116 passed**; touched dirs **209
passed, 10 skipped, 480 subtests**.

| # | Branch | Tip | Merge commit | Conflicts | Flip family | Step suite |
|---|---|---|---|---|---|---|
| 1 | `fix/replay-mirror-dequant-629` | `0a55facc74` | `d39afdafb7` | none | 1116 | 237 p / 10 s / 1121 sub |
| 2 | `fix/swa-target-verify-3287` | `dc168c1dac` | `28b5b04aaf` | none | 1116 | 255 p / 10 s / 1127 sub |
| 3 | `fix/loader-pd-bundle-643` | `a4a521d981` | `adc39a8fdf` | none | 1116 | 835 p / 25 s / 1389 sub |
| 4 | `audit/env-workaround-251-sweep` | `1ff60dcb7c` | `3b3290d780` | none | 1116 | 847 p / 25 s / 1389 sub |
| 5 | `feat/deterministic-hetero-412` | `934114d1de` | `1e09388cb1` | none | 1116 | 1568 p / 21 s / 1480 sub |
| 6 | `feat/turnkey-autoboot-539` | `80b12e280d` | `85c052ccdb` | none | 1116 | turnkey 52 p |
| 7 | `feat/rig-advisor-413` | `5e425f650b` | `75c86bf255` | none | 1116 | planner+turnkey 2416 p / 2 f / 0 err; `test_wizard` 54 p |

The flip family holds at **exactly 1116** across all seven merges — the
number N49, N50 and N51 also report — so the line's own behaviour is unmoved
by everything merged into it. The touched-dir counts grow only by the tests
each branch brings.

Cumulative test surface: `test/registered/unit/layers/attention/`,
`test/srt/distributed/`, `test/registered/unit/disaggregation/`,
`test/registered/unit/model_loader/`,
`test/registered/unit/test_env_workaround_defaults_251.py`,
`test/registered/unit/server_args/`, `tests/determinism/`,
`test/registered/unit/turnkey/`, `test/registered/unit/planner/`.

### Cross-checks performed rather than trusted

**Step 4, the barlink sampler default.** Verified in the *merged* tree, not
in the branch: `sampler_enabled()` reads
`os.environ.get(ENV_ENABLE, "1").strip() != "0"` — the sampler runs unless
someone sets `SGLANG_BARLINK_LAUNCH_DUMP=0` explicitly, and an unset
destination resolves to the previous `SAMPLE_DIR`. Default path
byte-identical, override opt-out only.

**Step 6, "disabled" is a claim about this host too.** The units ship
disabled and `systemctl list-unit-files` shows **zero** htsglang units
installed here, so the merge cannot have changed what boots.

**Ruff: 392 findings vs 391 at baseline, and the delta is a false
positive.** `server_args.py` carries 355 pre-existing `F722` "syntax error in
forward annotation" reports — one per flag declared with the
`A[type, Arg(help=...)]` idiom, which ruff cannot parse. Step 5 declares one
new flag, so the count becomes 356. Same rule, same idiom, no new class of
finding. The thirteen files the merges add are ruff-clean. Counted over an
identical 25-file set, not eyeballed.

## 5. The confirmation window

Ship config, booted from the **merged** tree at `1e09388cb1` (the last
serving-relevant code merge, i.e. after step 5), 22 minutes of real mixed
load, judged on the same three instruments and the same axes N51 used, via
the same `s50/extract_window.sh` extractor — so this is comparable to N51's
window rather than merely similar. Boot to ready: 143 s, confirmed with two
real generations, not health alone.

**Verdict: 0 breaches on both instruments.**

| | merge-r4 (this shift) | s51 ship (N51) |
|---|---|---|
| NVML samples | 10383 | 10331 |
| gpu0 min free | 1523 | 1585 |
| gpu1 min free | 2610 | 2009 |
| gpu2 min free | 1945 | 1949 |
| NVML breaches | **0** | 0 |
| seam-census troughs | 558 | 540 |
| deepest trough | 1522 | 1584 |
| seam breaches (<1024) | **0** | 0 |
| census CORRIDOR LAW BROKEN | 0 | 0 |
| phase flips (DONE/3 ranks) | 186 | 180 |
| FLIP ABANDONED | 0 | 0 |
| tracebacks / CUDA errors | 0 | 0 |
| soak | ok=276 err=0 | ok=270 err=0 |
| decode / prefill tokens | 94144 / 1309559 | 92160 / 1279920 |

The two instruments agree to **1 MiB** on the binding rank (NVML min 1523,
seam deepest 1522), the same agreement N51 observed (1585 / 1584). That
agreement is the point of running both: NVML samples at 100 ms and the seam
trough is shorter than that, so neither instrument alone is sufficient.

Host ledger over the window: cgroup `memory.current` peak 103836 MiB, shmem
peak 76872 MiB, `MemAvailable` min 30827 MiB, rank processes 3/3 throughout,
`oom_kill` delta **0**. This is the instrument C40 turned out to hinge on,
and it is clean.

A third, independent corroboration: a breach alarm tailing the corridor CSV
for sub-1024 values across the whole window fired **zero** events.

Do not over-read one window. N51's own caveat stands and applies here
unchanged — one clean window is not a certification, and the boot-to-boot
spread on this rig has been larger than the margin. What this window
supports is the narrower claim it was run for: **the five serving-relevant
merges are inert on the shipped configuration**, which is what each of those
branches claimed and what a unit test cannot show.

## 6. State at handover

- **origin `feat/route-a-631` = `integration/r2` = `75c86bf255`**, verified
  by SHA after every one of the seven pushes. Fork only; no upstream push.
- **All seven source branches remain on the fork, undeleted**, at their
  original tips (deletion is the user's call).
- **Serving is UP on 30030**, ship config, from `/spinning/wt-merge-r4/python`
  at the final tip `75c86bf255`. Ready in 152 s; verified with **two real
  generations** ("restored", "merge-r4-final"), not health alone. Corridor
  free at handover 1847/3454/3209 MiB, all above the 1024 law. **Nobody owes
  a restore.**
  - I deliberately rebooted after step 7 rather than leaving the window's
    process running: its code was `1e09388cb1` while the worktree had moved
    on, and a lazily imported `planner/plan.py` would have been read from a
    tree the process never loaded. The reboot also proves the final tip
    boots — a stale healthy process proves only that an older one did.
- **Router 30099 never touched** — 401 on every check, including at handover.
- Both stops in this shift were by **PID after a py-spy dump**, never
  `pkill`. Both released all three cards fully (20053/32086/20052 MiB) and
  returned host shmem to 0.6 GB.
- Worktrees left on disk: `/spinning/wt-merge-r4` (the merge tree, serving
  runs from it) and `/spinning/wt-merge-r4-base` (detached at `0ae49fafb4`,
  kept because it is the evidence for the 109-red A/B in section 1; remove it
  once that is no longer interesting).

## 7. For the next shift, in order

1. **Decide what to do about the 109-red planner directory** — now 2-red
   after step 7, but the question section 1 raises is unanswered: which other
   directories are outside every curated sweep and therefore unobserved?
2. **The two remaining `test_rejected_evidence_pins` are still red** and are
   now the *only* red in that directory. They are no longer lost in noise, so
   they are cheap to actually fix.
3. **N51's list is untouched by this shift and still stands**: three more
   windows on 40,12,12; give the cut gate a host-memory term; test the
   transient table across a cut boundary; itemise the 1250 MiB cut-shaped
   residual. This merge changed none of it.
4. `--deterministic-hetero` (step 5) and the advisor tab (step 7) have never
   been exercised on metal in this shift — the window ran the **ship** config,
   which enables neither. Both are default-off, which is why that was
   acceptable here, but neither has a metal boot behind it yet.
