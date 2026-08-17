# MERGE_TRAIN_2: the follow-up train

Companion to `MERGE_TRAIN_2026-08-17.md` (train 1). Same discipline: every
claim here is a MEASURED trial merge or a measured test run, not a reading of
the branch names. Trust git, not this list -- the commands are given so each
line can be re-derived.

Prep only. No branch pointer on a real line moves in this document; execution
is F4-r4's.

## 1. What train 2 is

Train 1 lands the serving-line fixes. Train 2 lands the cluster that train 1
deliberately held out, plus the two branches that were not ready when train 1
was assembled.

Ordered content, as measured:

| step | content | head | result |
| --- | --- | --- | --- |
| 1 | `reconcile/cluster-b-seam-model` | `6e627f4018` | **clean** |
| 2 | `fix/673-teardown-stack` (Slot-3's composed Cluster A) | `a66f5e263b` | **1 conflict, doc only** |
| 3 | #728 barlink (`fix/728-max-bytes-uniform`) | `887a6d477f` | **clean** |
| 4 | ~~`4512136e0e` + the #441 kernel commits~~ | -- | **NO-OP -- see 3** |

Assembled train-2 head: **`c5a4c1bca8`**.

## 2. The base it was measured against

Not `integration/r2`. Train 2 must be trialled against the head train 1
PROJECTS, or the conflict counts are fiction. That base was reconstructed as:

```
  integration/r2
  + fix/621-collective-invariant-pins
  + fix/699-progress-clock-wiring
  + fix/673-lockstep-sentinel-stop
  + 4c846373ac   (feat/706-phase-uniform-hicache-keys)
  + 67572ceac3
  + feat/677-park-wiring
  = 6bab764c33
```

All conflict and test numbers below are against `6bab764c33`, in throwaway
worktrees (`/spinning/wt-train2`, `/spinning/wt-t2probe`), hermetic
(`CUDA_VISIBLE_DEVICES=""`).

## 3. Ancestry: where the sgl-kernel rebuild is actually triggered

This was an open item carried out of train 1, and the answer moves the rebuild
step. Measured with `git merge-base --is-ancestor`:

| commit | in `6bab764c33` (post-train-1)? | in `887a6d477f`? | in `a66f5e263b`? | in `6e627f4018`? |
| --- | --- | --- | --- | --- |
| `4512136e0e` | NO | NO | NO | **YES** |
| `a8b068b718` | NO | NO | NO | **YES** |

So both kernel commits ride in as ancestors of the **reconciliation branch**,
i.e. **train-2 step 1**. Consequences:

1. **The separately-listed step 4 is a no-op.** Merging `4512136e0e` after step
   1 does not advance the head. It should be struck from the train, not
   executed and reported as "already included" -- the latter reads like a
   mistake in the log.
2. **The sgl-kernel rebuild belongs to step 1**, and must be triggered as soon
   as the reconciliation branch is in -- not at the end of the train.
3. **Train 1 is unaffected.** It merges `67572ceac3`, which carries neither
   commit. The earlier correction sent to F4-r4 about train 1's exclusions
   stands; nothing about train 1 changes because of this finding.

### 3.1 The rebuild step, named

Any train that lands `4512136e0e` / `a8b068b718` changes `sgl-kernel/` and the
INSTALLED wheel no longer matches the tree. Per the rig runbook's wheel-pin
discipline, the wheel is a pinned artifact with a recorded sha256 -- it is not
rebuilt implicitly and it is not picked up by an editable install.

Rebuild after step 1, before any boot-gated acceptance:

```
  SGL_KERNEL_LIMIT_CUDA_ARCHS=86;120     # rig cards only
  SGL_KERNEL_SKIP_SM90_VARIANT=ON
  SGL_KERNEL_ENABLE_FA3=OFF
  MAX_JOBS=4                             # -j4: swapless box, do not raise
```

with `nvcc` from the venv's pip toolkit (`site-packages/nvidia/cu13/…`), NOT
`/usr/local/cuda` (which is 12.9) -- point `CMAKE_CUDA_COMPILER`,
`CUDAToolkit_ROOT` and `CUDA_HOME` at it. Budget ~24 min cold.

Then update the pin: new wheel path, new sha256, new size, and the source
commit it was built from, in the runbook's kernel-wheel table. **A rebuilt
wheel whose pin was not updated is the failure mode this discipline exists to
prevent** -- the next reader cannot tell which tree the installed binary came
from.

Acceptance for the rebuild is BOOT-GATED, not desk-gated: the registered-op
count must be unchanged and the instance must reach ready. That gate belongs to
a GPU window; this document does not run it.

## 4. The one conflict, and how to resolve it

Step 2 (`fix/673-teardown-stack`) conflicts in exactly one path:

```
  docs/dev/MERGE_TRAIN_2026-08-17.md
```

Documentation only. No source conflict anywhere in the train.

Both lanes edited the train-1 prep doc. In the probe it was resolved by taking
the composed-stack version, which is fine for a throwaway but WRONG for the
real train: that discards this lane's §5b/§5c corrections. **The real
resolution is a UNION of both lanes' edits**, per-hunk.

Per-hunk, not `git checkout --theirs`. Whole-file resolution on this train
already cost one regression (train 1's `default_pp_micro_batch_size`, dropped
by a `--theirs` that took the whole file and broke
`test_pp_micro_batch_cap_701.py` at import). The resolver used here keeps the
merged file and decides only the conflicted regions.

## 5. Test matrix, composition-aware

Hermetic, `CUDA_VISIBLE_DEVICES=""`, on the assembled head `c5a4c1bca8`:

| suite | train-2 head | train 1 | note |
| --- | --- | --- | --- |
| `unit/mem_cache` | **1086 passed / 0 failed** (1651 skipped) | 940 failed | the 940 became SKIPS on reconcile+; this is the composition effect, not a fix count |
| `unit/managers` | 19 failed / 2367 passed | 4 -> 12 -> 19 by composition | see 5.1 |
| `unit/planner` | 8 failed / 2842 passed | 8 | all `test_webui`/chess, missing optional dep -- not this train |
| `unit/distributed` | **27 failed** / 2764 passed | 21 | +6, attributed in 5.1 |
| `test_scheduler_teardown_673.py` | 10 passed | -- | step 2's own suite |

### 5.1 The +6 in `unit/distributed`, attributed

Do not accept "the train added 6 failures" -- it did not. The failures
concentrate in three files
(`test_uneven_dcp_pool_geometry.py`, `test_stock_dcp_allocator_reach_487.py`,
`test_prefetch_progress_symmetry_580.py`). Measured on those three files alone,
each branch checked out detached:

| branch / head | failed | passed |
| --- | --- | --- |
| `887a6d477f` (#728, step 3) | 9 | 40 |
| `a66f5e263b` (teardown, step 2) | 9 | 40 |
| `feat/704-prefill-ladder` (`16384ca3c7`) | 9 | 40 |
| `fix/602-fill-side` (`1da6dbca9d`) | 9 | 40 |
<!-- branch names verified with `git branch --points-at`; the last two tips
     carry #723 / #715 subjects, which is the branch's latest work, not a
     mislabel of the branch. -->

| **`fix/701-ledger-wiring` (`90e792bd46`)** | **15** | **34** |
| `reconcile/cluster-b-seam-model` | **15** | 34 |
| train-2 head `c5a4c1bca8` | **15** | 34 |

The chain is exact and it terminates at a single branch: the 6 extra failures
are **pre-existing on `fix/701-ledger-wiring`**, inherited unchanged by the
reconciliation (step 1), and carried unchanged into the train-2 head. Neither
the reconciliation merge nor step 2 nor step 3 creates any of them.

So: **#701 carries this debt in, and #701 owns it.** It is not a merge defect,
and resolving it is not a precondition for running the train -- but the train
must not be reported as green on `unit/distributed` either. State it as "27
failed, 21 pre-existing at train 1 + 6 inherited from #701".

## 6. Do-not-drop list, carried forward

From train 1, still binding:

* `default_pp_micro_batch_size` on the serving line -- the whole-file-resolution
  casualty. `test_pp_micro_batch_cap_701.py` fails at IMPORT if it goes, which
  makes it a cheap post-merge check.
* the six serving-only commits identified in train 1 §2 -- they are on the
  serving line and not in any cluster branch; a reset to a cluster head loses
  them silently.
* `MERGE_TRAIN_2026-08-17.md` §5b and §5c (the Cluster B correction and the
  `fix/717-rebuild` tip correction) -- these are precisely what step 2's
  conflict threatens (4).

New to train 2:

* the sgl-kernel wheel pin (3.1). Rebuilding without updating it is the drop.
* `#690` (`c92e78a288`) needs no step in either train: its three files are
  already byte-identical in the post-train-1 head, and merging the commit would
  drag Cluster B in behind it (5-file conflict). Cherry-picking it is a no-op.
  Recorded here so it is not "rediscovered" as a gap.

## 7. Open items

1. The union resolution of the doc conflict (4) has not been performed on a
   real line -- only proven to be a single doc-only conflict.
2. The kernel rebuild's boot gate (3.1) needs a GPU window.
3. `fix/701-ledger-wiring`'s 6 failures (5.1) are attributed but not diagnosed.
