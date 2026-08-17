# #411 portable sessions: STOP -- there are two #410 formats

Date: 2026-08-17. Hermetic, no boots. **No #411 code was written, and that is
the deliverable.**

## The finding

#411's job is to make "the #410 format" portable. There is no *the* format.
Two independent implementations of #410 exist, neither aware of the other, and
the #411 WIP is built on one of them while the #410 work delivered TODAY --
including my own, earlier in this session -- extends the other.

Building #411 further, on either base, would entrench a fork **in the very
artifact #411 exists to make portable**. A portable format that ships in two
incompatible versions is the worst available outcome for this ticket, so the
build stops here and the reconciliation goes first.

## The two lineages

| | **A** | **B** |
|---|---|---|
| root | `682578f5e7` (2026-08-02) | `31bd063587` + `22ef67cef0` + `35a68b27a4` (2026-08-17) |
| module | `managers/session_checkpoint.py` (923 lines) | `mem_cache/session_manifest.py` + `mem_cache/session_checkpoint.py` |
| design doc | `docs/dev/DESIGN_410_session_checkpoints.md` | *cites the same filename, which does not exist in its tree* |
| versioned manifest | `CHECKPOINT_ENVELOPE_VERSION = 1`, `build_checkpoint_manifest:138` | `session_manifest.py`, own version convention |
| compat gate | `geometry_of:175`, `verify_geometry:180` | -- |
| restore verification | `verify_restore:218` | `verify_against_store` |
| branching | `account_branch:280`, `BranchAccounting:246` | `branch_plan` |
| lifecycle | `CheckpointLedger:337`, `SessionCheckpointRuntime:436` | `create/load/delete_checkpoint` |
| HTTP surface | `GET|POST /session/{id}/checkpoints` | none |
| scheduler / io_struct / server_args wiring | yes | none |
| tests | 938 lines | 601 lines + 189 (slice 2 remainder) |
| **pinning** | **none** | `pin_ledger.py` (`3550e20acf`), slice-2 wiring, coverage refusal |

`feat/411-portable-sessions` carries three cuts (~4832 insertions across 21
files) and is built on **A**: `682578f5e7` is an ancestor of it, `31bd063587`
and `22ef67cef0` are not. A's own commit message calls its manifest "the
versioned manifest ... which the ledger also names as #411's portable format".

## How the duplication happened, stated plainly

Slice 1's message opens "DESIGN FIRST, and the design's first move is to notice
what is already built", then surveys #706, #212 and #703 and concludes "what
was missing is the SESSION level: which pages, which blob, at which position,
under which sampling and template state."

**That layer was not missing.** It had existed since 2026-08-02 in
`managers/session_checkpoint.py`, with the versioned envelope, the GDN/mamba
anchor (`mamba_key`, `hybrid_gdn`), geometry verification, branch accounting,
HTTP endpoints and a 938-line suite. Slice 1 also cites
`DESIGN_410_session_checkpoints.md` as authoritative -- a document that exists
only in lineage A and is absent from slice 1's own tree, which is the visible
seam where the survey passed over the implementation beside it.

I am not exempt from this. Earlier in this session I built the slice-2 coverage
remainder (`35a68b27a4`) on lineage B and did not notice lineage A either; my
prior-art gate went to `DESIGN_410` and the slice commits and stopped there. The
gate catches an absent feature well and a DUPLICATE feature badly, because
grepping the ticket number finds work under that number and says nothing about a
second implementation living under a different module path. Both files are
called `session_checkpoint.py`; they differ only by package.

## The reconciliation, and why it is cheap

The two are not rivals across the board. They are complementary in exactly one
direction, which makes the merge obvious rather than political:

* **A is the format.** It is wired end to end -- HTTP, scheduler, io_struct,
  server_args -- has the geometry gate #411's import needs, and #411's three
  cuts already sit on it. Re-doing that on B would discard working wiring for
  no gain.
* **B's unique contribution is the protection A has none of.** Lineage A
  contains **zero** pin machinery: grepped `pin_checkpoint|pin_ledger|is_pinned`
  across its `session_checkpoint.py` -> 0 hits, and `pin_ledger.py` does not
  exist on `feat/411-portable-sessions`. So an A checkpoint's references are
  ordinary LRU entries and can be evicted out from under it.

That second point is not a nicety for #411, it is load-bearing: **a bundle
exported from A today carries references that nothing pins**, so the export can
go stale between write and import for exactly the reason slice 2 was built to
prevent. Porting B's pin ledger + slice-2 lifecycle + the coverage refusal
(`35a68b27a4`) onto A is what makes an exported bundle mean anything on the
target.

**Recommended order:**

1. Adopt **A** as the single #410 format. Retire B's `mem_cache/session_manifest.py`
   and `mem_cache/session_checkpoint.py` rather than leaving a second format in
   the tree -- a retired duplicate that still imports is a future accident.
2. Port B's `pin_ledger.py`, the slice-2 create/delete ordering, and the
   coverage refusal onto A's `CheckpointLedger` / `SessionCheckpointRuntime`.
   The pin ledger itself is store-level and largely lineage-agnostic; what
   needs rewriting is the ~60 lines that call it.
3. Resolve the `DESIGN_410_session_checkpoints.md` collision to one document.
4. **Then** continue #411 on `feat/411-portable-sessions`, which needs no rebase
   because it is already on A.

Only step 2 is real work. Steps 1, 3 and 4 are bookkeeping.

## What #411 still needs after the merge, from its own brief

Recorded now so the reconciliation does not have to be re-derived later. Not
built, because building any of it today would pick a format:

* export = A's manifest + payload bundle in ONE versioned file, refuse-unknown;
* import = the #241 key as a NAMED refusal, never silent conversion -- A already
  has `verify_geometry:180` to build it from;
* geometry moves only through the declared-compatible umsharder (#297/#261);
* **pin-through-import**: slice 2's coverage refusal applies at import too -- a
  bundle whose pages cannot all be pinned on the target must refuse WITH NAMES.
  This is unbuildable until step 2 lands, because the target has no pin ledger;
* TP>1 export needs per-rank manifest merging (#261 residue); if that is not
  desk-fundable it gets a named refusal and a filed residue.

Cross-rig transport stays out of scope: file-based export/import, the user moves
the file. The byte-identity gate stays boot-gated per #410 convention.

## Method note

The prior-art gate as briefed -- grep the catalog, the docs, and
`git log --all --grep=<ticket>` -- found lineage A's commits immediately. What
it did not do is make the DUPLICATION visible, because every hit was correctly
filed under #410/#411 and nothing in a ticket-number grep says "there are two of
these". The cheap addition, on any ticket that builds a named artifact: grep for
the FILENAME as well as the number. `session_checkpoint.py` would have returned
two paths on the first try.
