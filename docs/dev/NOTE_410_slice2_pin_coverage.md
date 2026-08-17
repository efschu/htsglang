# #410 slice 2 remainder: the pin that was dropped without saying so

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots. Built on
`22ef67cef0` (slice 2 wiring), which is on `feat/410-checkpoint-pinning` and
not yet in the shipping lineage -- this branch is based on it so the work
composes rather than duplicating. That merge is a train item.

## The per-node marker question, answered

The brief asked whether pinning needs a per-node marker the evictor respects,
which #703 documented as absent. **It does not: that marker already exists for
the tier that matters.** `HiCacheFile` hands `is_pinned=self.pins.is_pinned`
to `LRUFileEvictor` (`hicache_storage.py:685`), the evictor skips pinned stems,
and `reclaimable_bytes` already excludes pinned bytes.

`hicache_demotion.py` is genuinely pin-blind -- grepped it, and the only two
"pin" hits are the `typing` import and a log string. But that is **not** a hole
for #410, and the distinction is worth stating rather than fixing something
that is not broken: demotion moves a node DOWN a tier by writing it to the file
store. It preserves data; it is how a page reaches the tier where pins protect
it at all. A pin-blind demoter cannot lose a pinned checkpoint's page. Making
it pin-aware would be work with no failure behind it.

So the marker was not the build. The build is below.

## What was actually wrong

Slice 2 already refuses a manifest whose references the store does not hold:
`create_checkpoint` calls `verify_against_store` and raises `ManifestIncomplete`
before pinning, with the docstring explaining that pinning an absent key
"protects nothing while still looking like success".

But verification and pinning ask **two different authorities**:

* `verify_against_store` -> `HiCacheFile.exists`, which returns True from the
  **metadata cache** when it holds an entry (`hicache_storage.py:1161`).
* `pin_checkpoint` -> `_existing_path` + `os.stat` inside `stems_with_sizes`,
  i.e. the **filesystem**, which documents that "missing files are dropped
  rather than pinned at size 0".

Dropping is correct for the budget -- charging for protection nobody gets is
the #715 error. What was wrong is that the drop was **silent**: nothing
compared what was requested against what was pinned. A reference whose
metadata-cache entry outlived its file, or that vanished between the two
checks, produced a checkpoint reporting success while protecting less than it
claimed. The shortfall then surfaced at BRANCH as "a reference was evicted,
refusing to branch" -- which is exactly the outcome slice 2 exists to
eliminate, arriving later and further from its cause.

This is the third instance of the same family in this ticket's own history:
#698 was a trigger that never fired, slice 2's ledger was a protection never
taken, and this was a protection **partially** taken and reported as whole.

## The build

* `PinResult.unpinned` (`pin_ledger.py`) -- the references the caller asked for
  that were not pinned. Default empty, so nothing else changes.
* `HiCacheFile.pin_checkpoint` records them at the point where the CONTENT key
  is still known, so a refusal can name what the caller asked for rather than a
  suffixed stem.
* `PinCoverageIncomplete` (`session_checkpoint.py`) -- raised at CREATE, after
  rolling the pins back, naming every reference it could not pin.
* `CheckpointRecord.references_requested` / `.references_pinned` -- so a
  successful create states its coverage instead of implying it.

Nothing here tries to make a deleted page survive; nothing can. The change
moves the refusal to where the caller can still act on it.

**`pin=False` is untouched.** An unpinned checkpoint promises nothing, so it
checks nothing and reports zero coverage rather than a shortfall. That
behaviour is pinned by a test.

## Tests

6 new, hermetic, red first, driving the REAL store and the REAL metadata cache
-- never the ledger's own bookkeeping, since such a test would have passed
against the unfixed tree.

The first test is a guard on the guard: it asserts the setup really does make
`exists` lie. If the metadata cache did not retain the entry the remaining
cases would be vacuous, so that premise is checked rather than assumed.

Two mutations, each red on exactly the two refusal tests and green elsewhere:
neutering the coverage branch in `create_checkpoint`, and neutering the
missing-file detection in `pin_checkpoint`. Both halves of the chain are
therefore load-bearing.

Regression: 37 passed across the three #410 suites (manifest, slice-2 wiring,
this one), ruff clean.

## Still boot-gated, per slice 1

The byte-identity gate stays a window item. Nothing here changes what a branch
produces; it changes when an unkeepable promise is reported.
