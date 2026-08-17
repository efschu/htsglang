# MERGE TRAIN 2026-08-17 — preparation

**Status: PREPARATION ONLY. Nothing here was merged and no branch pointer was
moved.** Every conflict below was found by trial merges in a throwaway worktree
(`throwaway/merge-train-probe`, deleted after measurement); the execution is
F4-r4's post-window job.

Merge target: **`integration/r2`** (`a73a0d8da8`, 2026-08-14) — the newest
integration line. `main` (`82e7cdcff9`, 2026-07-12) is the upstream mirror and
is not a merge target for fork work.

Serving line: the deploy tree is detached at **`5fed8a62ed`**, which is exactly
`feat/677-park-wiring`. Nothing on the serving line is orphaned on a nameless
head — checked, and it matters, because a detached serving head with unbranched
commits is how a train loses a fix.

## 1. Inventory

All train branches are **independent siblings**: a containment check across the
set found no branch containing another, so ordering is a conflict question, not
a dependency chain. Their common ancestor is `7936bc4850` (2026-08-16).

Delta columns are measured against the SERVING LINE (`5fed8a62ed`), i.e. "what
this branch adds beyond what is already serving".

| branch | head | +commits | files | pushed |
|---|---|---|---|---|
| `feat/677-park-wiring` (serving) | `5fed8a62ed` | — | — | yes |
| `fix/713-admission-intake` | `3c1aa6f29b` | **0** | 0 | yes |
| `fix/699-progress-clock-wiring` | `2753c764ba` | 1 | 4 | yes |
| `fix/717-rebuild` | `a8b068b718` | 7 | 20 | yes |
| `fix/673-kvso-thread-stop` | `4334c642d4` | 12 | 33 | yes |
| `fix/673-lockstep-sentinel-stop` | `85c5d62c86` | 13 | 36 | yes |
| `fix/673-dual-group-lane-stop` | `91adb7c156` | 13 | 35 | yes |
| `fix/673-barlink-watchdog-stop` | `b543749713` | 15 | 40 | yes |
| `fix/621-collective-invariant-pins` | `9247189e57` | 16 | 43 | yes |
| `feat/706-phase-uniform-hicache-keys` | `79216e6052` | 17 | 46 | **1 unpushed** |
| `fix/602-fill-side` | `1da6dbca9d` | 62 | 78 | yes |
| `fix/701-ledger-wiring` | `90e792bd46` | 78 | 100 | yes |
| `feat/704-prefill-ladder` | `16384ca3c7` | 83 | 107 | yes |

Two findings the provided list did not carry, both from git:

* **`fix/713-admission-intake` is already in the serving line** (+0 commits, 0
  files). It is not a train item; merging it is a no-op. Do not schedule it.
* **`fix/728-max-bytes-uniform` — IDENTITY RESOLVED 2026-08-17 (Slot-3): it was
  the second case, a lane that had branched and not yet committed.** It now
  points at **`887a6d477f`** and **is pushed** to `origin`. It is NOT an alias
  and must NOT be deleted — deleting by that name would destroy the #728 fix.
  It is a real train item, for the FOLLOW-UP train only (see §5). The original
  observation below was correct at 08:08 and is now stale; kept for the record.
  It read: points at `79216e6052`, byte-identical to
  `feat/706-phase-uniform-hicache-keys`, and **has no remote**. Either it is a
  local alias of that head or a lane branched and has not committed yet. It
  must not be merged as a separate item, and someone should say which it is
  before the train runs; a local-only branch is the one kind that vanishes with
  its worktree.

Test evidence per branch is in the commit messages (this lane's convention:
regression tallies and a can-fail matrix per commit). The train operator should
not re-derive it; the gating suites are in §4.

## 2. Divergence audit

The `42229763af` class — "in the deploy tree, absent from main" — was checked
and is **not the risk it looked like**:

| ref | carries `42229763af` |
|---|---|
| `main` | ABSENT |
| `integration/r2` | **PRESENT** |
| serving line `5fed8a62ed` | PRESENT |

It is absent only from upstream `main`, which is true of every fork commit, so
"deploy vs main" is not the useful axis. The useful axis is the merge target:

* **167 commits** are on the serving line and not in `integration/r2`.
* Of those, **6 are on NO other train branch** — the genuinely deploy-only set,
  and the only commits the train can actually lose:

```
5fed8a62ed [#713] Count the resident-but-unprefilled request the policy ...
c4bc982d64 [#677] Advance the hold counter, so the bound is real ...
332cb3b345 [#677] Wire the hold, mirror it on the TP side, ...
59592d666b [#677] Demand decides the layout, not the timer -- in both ...
fc6f97b69b [#708] Make the kv-availability probe stand-in-safe, and re-k...
761d0d75bd Merge commit '3c1aa6f29b' into feat/677-park-wiring
```

They are all reachable from `feat/677-park-wiring`, so merging that branch
carries them. **If any step drops or rewrites that branch, these six are the
loss.**

## 3. Order and conflict map

Proposed order, dependency-aware, serving range last. Every "clean"/"CONFLICTS"
below is measured, not predicted.

| # | step | result | conflicting files | resolution owner |
|---|---|---|---|---|
| 1 | `fix/621-collective-invariant-pins` | **clean** | — | — |
| 2 | `fix/699-progress-clock-wiring` | **clean** | — | — |
| 3 | `fix/673-lockstep-sentinel-stop` | **clean** | — | — |
| 4 | `fix/673-kvso-thread-stop` | **CONFLICT** | `managers/scheduler_teardown.py` | #673 thread-stop lane |
| 5 | `fix/673-dual-group-lane-stop` | **CONFLICT** | `managers/scheduler_teardown.py` | #673 thread-stop lane |
| 6 | `fix/673-barlink-watchdog-stop` | **CONFLICT** | `managers/scheduler.py`, `managers/scheduler_teardown.py` | #673 lane + barlink owner (#722) |
| 7 | `feat/706-phase-uniform-hicache-keys` | **clean** | — | — |
| 8 | `fix/602-fill-side` | **CONFLICT** | `managers/seam_slope.py`, `mem_cache/common.py`, `planner/pp_cut.py`, `test_device_evict_floor_694.py`, `test_pp_cut_prefill_speed_702.py` | planner/seam lane |
| 9 | `fix/701-ledger-wiring` | **CONFLICT** | `managers/schedule_policy.py`, `managers/scheduler.py`, `managers/seam_slope.py`, `planner/pp_cut.py`, `test_pp_cut_prefill_speed_702.py` | planner/seam lane |
| 10 | `feat/704-prefill-ladder` | **CONFLICT** | `managers/schedule_policy.py`, `managers/seam_slope.py`, `planner/pp_cut.py`, `test_pp_cut_prefill_speed_702.py` | planner/seam lane |
| 11 | `fix/717-rebuild` | **clean** | — | (but see §5) |
| 12 | `feat/677-park-wiring` (serving) | **clean** | — | — |

### Two conflict clusters, and what they mean

**Cluster A — `scheduler_teardown.py` (steps 4-6).** All four #673 thread-stop
branches add their stop logic to the same file, which this lane created in the
#673 teardown fix. They do not conflict with the trunk; they conflict with EACH
OTHER, so whichever lands first wins the file's shape and the other three must
rebase onto it. Cheapest resolution: land them as ONE branch, or nominate an
order inside the #673 lane and rebase the other three before the train starts.
The barlink one additionally touches `scheduler.py` and belongs to #722, which
is live — it should be last within the cluster and reviewed by that owner.

**Cluster B — planner/seam (steps 8-10).** `fix/602-fill-side`,
`fix/701-ledger-wiring` and `feat/704-prefill-ladder` each rewrote
`seam_slope.py`, `planner/pp_cut.py` and `test_pp_cut_prefill_speed_702.py`.
These are three lanes editing one model of the same thing; the conflict is
SEMANTIC, not textual, and resolving it by taking hunks would produce a seam
model nobody designed. This cluster needs its owners in the room before the
train, not during it.

**The clean set is genuinely clean**: steps 1-3, 7, 11, 12 merged in sequence
with no conflicts at all, which is the train that can run unattended if the two
clusters are deferred.

## 4. Test matrix and baselines

Baselines measured ON THE MERGED PROBE STATE (`integration/r2` + steps 1, 2, 3,
7, 11, 12), hermetically, `CUDA_VISIBLE_DEVICES=""`:

| suite | result | note |
|---|---|---|
| `test/registered/unit/mem_cache` | **940 failed / 973 passed** | 940 matches the standing baseline: GPU-required tests under CVD="" |
| `test/registered/unit/distributed` | **21 failed / 2716 passed** | matches the standing 21 |
| `test/registered/unit/managers` | **12 failed / 2274 passed** | see below |

**The managers count is train-composition-dependent, and that is the trap.**
This lane's branch alone shows **4** failures in that suite; the merged probe
shows **12**. The extra 8 arrive with the #677/#713/#631 lanes, not from the
merge. Failing classes observed on the probe:

```
test_phase_purity_631 (2 tests), FirstChunkIsSizedDynamically,
PrefillAdmissionBudgetTest, TestThePpPhaseIsGovernedByDrainNotAStopwatch,
TheBackstopsAreUntouched, TheFullCycle, TheOuterBackstopStillExists,
TheWedgeIsLeft
```

`FirstChunkIsSizedDynamically` and `PrefillAdmissionBudgetTest` are in the
long-standing 4. **Each branch owner must record their own managers baseline
before the train**, or the post-merge number cannot be attributed — quoting a
single standing figure for this suite is what would turn "the branch shipped
it" into "the merge broke it".

Gating per step: steps 1-3 and 7 gate on `unit/mem_cache` + `unit/managers`;
steps 8-10 additionally gate on `unit/planner`; step 6 gates on
`unit/distributed` (barlink); step 12 gates on all three plus
`test/registered/scheduler/test_phase_flip_runtime.py` (67 passed on this
lane's measurements).

## 5. Held out

**`fix/717-rebuild` — `2ce1ed7ba6` is held for its own review boot (standing
decision). Are the later #441 commits separable? YES, by file.**

Contents of the branch, oldest first:

| commit | ticket | files |
|---|---|---|
| `675793cdc8` | #717 | `managers/kv_backing_relief.py` + 3 tests |
| `be1fcec6d6` | #677 | `docs/dev/NOTE_677_floor_components.md` |
| `2ce1ed7ba6` | #677 | that NOTE + `TICKET_718_*.md` + **`managers/phase_flip_runtime.py`** + 2 tests |
| `2a73f12804` | #725/#724 | `planner/activation_quant_crossover.py` + doc + test |
| `67572ceac3` | #441 | doc + test + `tools/441/falsify_lf_ph_441.py` |
| `4512136e0e` | #441 | doc + `sgl-kernel/csrc/kvcacheio/transfer.cu` + kernel test + 2 tests + `scripts/handover/` |
| `a8b068b718` | #536 | doc only |

* `2ce1ed7ba6` is **not** docs-only — it changes `phase_flip_runtime.py`, which
  is exactly why it earns a review boot.
* The two #441 commits touch **no file** that `2ce1ed7ba6` touches, so they are
  cherry-pickable without it. The only textual entanglement on the branch is
  `be1fcec6d6` -> `2ce1ed7ba6` (both edit the same NOTE).
* **But `4512136e0e` touches `sgl-kernel/csrc/kvcacheio/transfer.cu`** — a CUDA
  kernel. Riding it on this train means a kernel rebuild, which is its own
  boot-gated risk and should not be smuggled in behind a docs-and-tests
  framing.

Also held, and named so nobody schedules them:

* **`fix/713-admission-intake`** — already in the serving line, +0 commits.
* **`fix/728-max-bytes-uniform`** — RESOLVED: real branch at `887a6d477f`,
  pushed. Held from the speed boot ON PURPOSE: it changes barlink transport
  bring-up (`pipe_on` is now AND-reduced across the group before `max_payload`
  and `geometry`), so it needs its own soak rather than riding a speed boot.
  **FOLLOW-UP TRAIN.**
* **Cluster A — SUPERSEDED by `fix/673-teardown-stack`** (this branch). All
  four #673 thread-stop branches are composed there in one ordered sequence,
  with the current integration base merged in so it applies post-train. Steps
  4-6 of §3 should be struck and replaced by that single item; step 3
  (`fix/673-lockstep-sentinel-stop`) is contained in it and must not be
  scheduled separately, or the same functions land twice.
* Cluster B (§3) — still deferred until its three owners have agreed a
  resolution. The clean six-step train can run without it.

## 6. Open items before the train runs

1. `feat/706-phase-uniform-hicache-keys` had **1 unpushed commit** at the time
   of writing (pushed with this document). Every other train branch was pushed.
2. Cluster A needs one owner and one order; Cluster B needs its three owners.
3. Per-branch `unit/managers` baselines (see §4).
4. ~~`fix/728-max-bytes-uniform` identity.~~ RESOLVED: real branch, pushed at
   `887a6d477f`, follow-up train.
5. Cluster A: RESOLVED as `fix/673-teardown-stack`; steps 3-6 collapse to it.
