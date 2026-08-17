# #411 reconciliation, step 2: B's pinning ported onto lineage A

Date: 2026-08-17. Hermetic, no boots. Branch based on `03adbf8137` -- the
`feat/411-portable-sessions` tip -- so #411 can resume on the same branch with
pinning available rather than waiting for a separate merge.

## Step 1: lineage A adopted

Per `VERDICT_411_two_410_formats.md`. Nothing in A was replaced; this slice is
purely additive to it.

## Step 2: the port, and a correction to my own estimate

**My "~60 lines of call sites" was wrong.** That number assumed A already had
the store-level pin stack and needed only the ledger wired into its
`CheckpointLedger`. It did not: on `03adbf8137` there is **no** `pin_ledger.py`,
**no** `pin_checkpoint` on `HiCacheFile`, and **no** `is_pinned` anywhere in the
evictor. The entire protection had to come across.

What actually moved:

| piece | size | note |
|---|---|---|
| `mem_cache/pin_ledger.py` | 330 lines | taken whole; depends on nothing lineage-specific |
| `envs.SGLANG_HICACHE_PIN_BUDGET_BYTES` | 1 | absent on A |
| evictor: `pins=` param, skip-and-repin | ~12 | the actual protection |
| evictor: `_allocated_size` | ~20 | **required** -- `stems_with_sizes` calls it |
| evictor: `stats()` | ~25 | A had none; without it pinned bytes are unobservable |
| store: ledger construction + `pins=` | ~10 | built BEFORE the evictor, deliberately |
| store: `pin_checkpoint` / `unpin` / `pin_stats` / `capacity_stats` | ~45 | |

**It was a port, not a copy, and the difference mattered twice.**

1. B's `pin_checkpoint` resolves paths through `_existing_path` / `_sharded_path`
   -- a read-through sharded layout that **does not exist on A**. Copying it
   would have pinned paths this store never writes. A `_pin_path` helper using
   A's flat `file_path/{stem}.bin` join replaces it, matching what A's evictor
   actually unlinks.
2. The two lineages' `hicache_storage.py` differ by 618 inserted lines, of which
   only **18** are pin-related. Taking B's file wholesale would have dragged the
   canonical page store, the metadata-cache layer and the sharding migration
   across under the banner of "porting pinning".

## The blocker this port found, and did not paper over

`reclaimable = used_bytes - pinned_bytes` is **not a coherent subtraction on A**.

* A's evictor accounts **APPARENT** bytes: `st.st_size` in
  `_scan_existing_files`, `value_bytes` in `reserve`.
* The ported ledger charges **ALLOCATED** bytes via `_allocated_size`
  (`max(st_blocks * 512, st_size)`).

On any filesystem that allocates more than it stores -- including this one --
`pinned_bytes` can exceed `used_bytes` for the same files. The ported suite
caught it immediately: `0 != -384`.

This is exactly the #715 shape the ledger's own docstring names, arriving from
the direction nobody was watching: not a wrong number, but two right numbers in
different units. My first instinct was `max(0, ...)`, which produces a clean
plausible zero and hides an incoherent ledger behind it. That clamp is gone.
`stats()` now reports **`accounting_overshoot_bytes`**, non-zero exactly when
the two accountings disagree, so the incoherence is visible rather than
absorbed.

`test_pinned_bytes_are_not_reported_as_reclaimable` is skipped **with the
blocker named**, not deleted and not made to pass. It re-activates the day both
sides charge one unit.

**Resolving it is the next slice, and the direction is not symmetric.** B's
allocated accounting is the correct one -- the ZFS incident behind it undercounted
real usage 17x across 5.8M files -- so the fix is to move A's evictor onto
allocated bytes, not to move the ledger onto apparent ones. That changes *when*
A evicts, so it needs its own red-first slice and its own validation; slipping it
in at the end of a port would be a behaviour change smuggled inside a refactor.

## State

* pins protect pages on A: the evict-skip works and is covered.
* 10 passed / 3 skipped / 0 failed in the ported ledger suite. The three skips
  are named: two need `canonical_page_store` (a B-lineage module A does not
  carry, guarded so they self-activate if it is ported), one is the unit blocker.
* A's own `test_session_checkpoint.py`: **48 passed**, unchanged by the port.
* ruff clean on every file touched.

## Not done, and not started

* **Step 2 remainder:** wiring `PinCoverageIncomplete` into A's
  `CheckpointLedger`. Deliberately deferred behind the unit blocker -- coverage
  refusal is an accounting promise, and installing it on top of an incoherent
  subtraction would build the honest layer on the dishonest one.
* **Step 3:** the `DESIGN_410` de-duplication and the disposal of B's
  `mem_cache` modules. **Importer check IS run, and it clears the path:** on
  B's tip, `mem_cache/session_manifest.py` and `mem_cache/session_checkpoint.py`
  are imported by exactly two production files -- `session_bundle.py`, which
  #121 already established is superseded WIP, and B's own
  `session_checkpoint.py` referencing the manifest beside it -- plus their three
  test files. **Nothing on a live path imports either.** So step 3 can be a
  clean removal rather than a deprecation shim; a deprecation docstring is only
  worth its cost when something still calls the module, and nothing does. The
  three test files go with them, except the coverage semantics, which are
  already re-expressed against A in this port's suite.
* **Step 4:** #411 itself -- compat gate, pin-through-import, TP>1 verdict.

For the #706 Lane: this branch supersedes `train/0818-desk-410-pinning` as the
merge candidate, and lineage B's `mem_cache/session_manifest.py` +
`mem_cache/session_checkpoint.py` should not be merged as a second format.

---

# Step 2: A's evictor moves to ALLOCATED bytes

The blocker above is cleared. `_scan_existing_files`, `commit` and `touch` now
all charge `_allocated_size`, so the evictor and the pin ledger share one unit.

**`reserve` deliberately still estimates.** A reservation is taken before the
file exists, so the payload length is the only number available; `commit`
reconciles it against the filesystem's own answer once the write has landed.
Statting a file that is not there yet would be the alternative, and it is worse.

**Measured on this rig, not assumed.** ZFS charges a 64-byte page **512** bytes
-- an 8x divergence -- while a 64 KiB file reports `st_blocks == 1` because of
delayed allocation. Both halves of `max(st_blocks * 512, st_size)` are therefore
load-bearing here, and both are covered.

The red test is an **eviction decision**, not an accounting field: six tiny
pages into a store sized for three ALLOCATED pages. Under apparent accounting
the store believes it is at 320 of 1536 bytes and evicts nothing; under
allocated accounting it is over cap and must. A test that only asserted
`used_bytes` would have passed against an evictor that computed the right
number and still evicted on the wrong one.

The previously skipped `test_pinned_bytes_are_not_reported_as_reclaimable` is
**un-skipped and green**, and `accounting_overshoot_bytes == 0` is asserted
directly.

## The regression this slice caused, and the prior art that fixed it

The first broad sweep after the accounting change came back **599 failed**
against the port-only run's **590** -- nine new failures, all in
`test_hicache_file_lru_unit.py`, the evictor's own suite. Runtime also went from
12 s to 235 s, which was the louder signal: the store was over cap from the
first write and churning.

The cause was not the change. That suite sizes its fixtures in **apparent**
bytes -- `max_size="300"` with three 100-byte tensors, asserting
`_total_bytes == 300` -- so on ZFS, where a 100-byte file occupies 512, its
premise collapses the moment the accounting is correct.

**B had already solved this**, and the amended prior-art gate is what found it:
grepping the FILE name rather than the ticket number showed the same suite
existing in both lineages, and B's copy sizes everything in a measured
`_UNIT = 512` with the reasoning inline ("the evictor accounts what the
filesystem charges ... so sizes in these tests are whole units to keep the
arithmetic exact"). Adopted verbatim: 34 passed in 8.6 s, and no B-only imports
came with it.

**The other two files in the failure list were pre-existing, and I checked
rather than assumed.** Run at the port-only commit and at this one:
`test_swa_eviction_boundary.py` 9 failed on both, `test_mamba_checkpoint_interval.py`
19 failed / 15 passed on both. Identical. They belong to the standing
no-accelerator family, not to this change. Lint is unchanged too -- A's original
copy of the LRU suite and B's adapted one both carry the same 14 pre-existing
ruff findings, so adopting B's version imports no new debt.

## What this slice also settled: A pins a DIFFERENT TIER

Worth recording before step 3, because it changes what step 3 is.

`SessionCheckpointRuntime._pin` already exists on A and does
`tree.inc_lock_ref(node)` -- it locks the **radix chain**, in memory, for the
life of the process. B's pin ledger protects the **file tier**: on-disk stems,
against LRU eviction, across a restart.

These are complementary, not duplicate -- exactly as slice 2's own message
predicted ("lock_ref/host_ref_counter are the radix-tier authority and are a
different layer"). So the port did not add a second copy of something A had; it
added the tier A had no protection for. An A checkpoint today survives radix
eviction and does **not** survive file-tier eviction.

## Correction to the step-3 instruction

The brief says to wire `PinCoverageIncomplete` into A's `CheckpointLedger`.
That class documents itself as "Pure, hermetic" and holds only in-memory
records -- it never touches the store, so a store-coverage refusal cannot live
there without giving it an I/O dependency it was designed not to have.

The refusal belongs in `SessionCheckpointRuntime._checkpoint`, which is what
already talks to the tiers, and which after this port must additionally take
the FILE-tier pins it has never taken. That is step 3.
