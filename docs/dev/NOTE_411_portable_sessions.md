# #411 — portable sessions: the file boundary and its gate

Desk only, 2026-08-17. No boot, no GPU, no second machine.

## 0 — Branch premise: #410 is on unmerged branches, and I had to re-base

The brief lists #410 as an existing piece. It is — but **not on the branch I
was working from**. All three #410 commits live on `feat/session-checkpoints-410`
and none is in the ancestry of the #553 branch I had been on:
`session_checkpoint.py` simply did not exist in the worktree.

So `feat/411-portable-sessions` is cut from `feat/session-checkpoints-410`,
not from the earlier line. This is the "absent in this branch ≠ does not
exist" distinction, applied before building rather than after.

**And Slice 2 (pinning) is on a THIRD branch**, `feat/410-checkpoint-pinning`,
which is not in this base either. That matters concretely: the export refusal
below advises "re-pin the checkpoint (#410 slice 2)", and on this branch that
advice names a capability the reader does not have. It is the right advice for
the merged world and a dangling reference today; recorded rather than
softened.

## 1 — Geometry is BINDING, not advisory. The design says otherwise; the code is right.

The brief asked whether #706's geometry-neutral format makes the geometry
field advisory. **It does not**, and the two sources disagree:

- `DESIGN_410` §3 says the manifest is "Content only, no placement … a
  manifest written by a PP-prefill instance is readable by a TP-decode
  instance, by another rig with a different parallel geometry".
- The implementation carries `GEOMETRY_FIELDS = ("tp_size", "page_size",
  "dcp_owner_mode")` and `verify_geometry` **refuses** on mismatch.

The implementation is right, and its own comment says why: pages written under
a different tp_size / page_size / owner mode "are laid out differently and
nothing in the read path would notice". It refuses and names the offline
converter (`hicache_migrate --manifest`) rather than reshaping in-process —
"A checkpoint is never silently converted."

That is the same conclusion #545 reached from the other side: the umsharder is
load-bearing, not a harness convenience, so a cross-geometry move without it
is not a hard path but a meaningless one.

**Consequence for #411:** cross-rig import across *identical* geometry is a
file copy plus a gate. Across *different* geometry it is a file copy plus an
offline umsharder run plus a gate. There is no in-process conversion, by
design, and this module does not add one.

## 2 — What was built

`managers/session_portable.py` + 17 pins.

**Container: a tar**, and the reasons are the choice:
`tarfile` is stdlib so a portability format acquires no dependency — which
matters most on the machine that has to READ it; page payloads are opaque
blobs of varying size, so tar streams them without anyone inventing a
length-prefix format to get wrong; and it opens with tools every machine
already has, which a format people debug on a second rig should not need this
repo for.

**The manifest is written FIRST**, and that ordering is load-bearing:
`read_manifest` reads one member, so the gate can refuse a 40 GiB bundle
without extracting a payload. Refusing after unpacking is a worse refusal.

**The gate, in order, each a named refusal and never a conversion:**

1. **envelope version** — unknown versions refused, never best-effort parsed.
   Checked FIRST because a bundle from an unknown version may not even carry
   the fields the later checks read, so an identity complaint about it would
   mislead.
2. **model identity** — weights, dtype, quantization, KV byte format. Note
   this is why the gate has three axes and not five: kv-cache dtype lives
   *inside* the identity hash, so a differing dtype changes the hash and lands
   here.
3. **geometry** — delegated to `verify_geometry`, which already owns the rule
   and already names the converter. Re-deriving it here would have been a
   second authority for a correctness rule.

**No holes.** `export_bundle` refuses when any referenced page is
unretrievable, and removes its partial file: an export that omitted a page
would produce a bundle that imports cleanly and **decodes wrong** on a machine
that cannot tell. #410's evicted-page-refused-at-branch rule applies at export
too. Import re-checks completeness before returning, mirroring #410's
completeness-before-seeding rule.

Mutation: dropping the import gate fails 5 of 17 — a wrong-identity bundle
imports silently.

## 3 — The gap this gate does NOT close, stated in the refusal itself

The identity hash covers the kv-cache dtype **name**, not the byte layout
within it. Two builds that both call themselves `int8` but differ in group
size or scale dtype produce the same identity and pass this gate.

That is my #726 finding carried forward. It predates this module and is not
closed here — but `IDENTITY_LAYOUT_GAP` is quoted into every identity refusal,
so a reader learns where the gate stops rather than inferring coverage it does
not have.

## 4 — Honest limits

- **This is the file layer, not the seeding path.** `import_bundle` returns
  `(manifest, pages)` after gating; wiring those into #410's
  completeness-before-seeding path is the next cut and is deliberately not
  done here, because that path lives on the checkpoint runtime and wants its
  own red-first pass rather than being tacked on.
- **Cross-RIG import is unproven.** Everything here is one machine, tmpdirs
  and an injected page reader. The second-machine proof is a window/laptop
  item: export on this rig, import on `efeu-TP14`, continue byte-identically.
  Nothing in this note claims it.
- **Slice 2 dependency**, see §0: the export refusal's re-pin advice needs
  `feat/410-checkpoint-pinning` merged to be actionable. Exports of long-idle
  sessions are exactly the case where a page has been evicted, so this is the
  dependency that decides whether the refusal is rare or routine.

## 5 — Window acceptance (filed, not run)

1. Export a checkpoint on this rig; import on a second machine with the SAME
   model identity and geometry; continuation byte-identical to the source's,
   greedy. That is #261's proof extended across the file and the machine.
2. Import with a deliberately different `--kv-cache-dtype`: refused by name,
   identity mismatch, before any payload is read.
3. Import across a different `tp_size`: refused by name, and the message
   names `hicache_migrate --manifest`. Then run that converter offline and
   import successfully — which is the #411-as-converter claim, end to end.
4. Export a long-idle session whose pages have been evicted: refused, not
   holed. With Slice 2 merged, re-pin and export successfully.

---

# CUT 2 — the wire, and three corrections to my own Cut 1

The prior-art gate found more in Cut 1 than the wire it was meant to add.

## C1 — Cut 1 exported the WRONG BLOB SET, and it was a #212 failure

Cut 1 read `page_hashes` — a field name taken from DESIGN_410's **prose**. The
manifest `session_handover.build_manifest` actually writes carries
`kv_keys`, `mamba_key`, `hybrid_gdn` and `draft_keys`.

Not cosmetic. A hybrid-GDN session exported by Cut 1 would have carried its KV
pages and **not its recurrent state**, producing a bundle that imports cleanly
and replays a **wrong session** — because a missing mamba blob truncates the
prefix match at the destination and silently re-prefills. That is exactly what
`validate_manifest_completeness` calls "a wrong session, not a slow one".

Fixed: `referenced_blobs()` enumerates kv + mamba + draft, de-duplicated, with
`page_hashes` still honoured so a prose-shaped manifest is not silently
dropped.

## C2 — Cut 1's gate duplicated `verify_import`

Version and identity checks are already in `session_handover.verify_import`,
together with blob presence **and** the #212 hybrid-GDN clause. I had
re-implemented two of the four.

## C3 — my first Cut 2 pass duplicated `verify_restore`

Having replaced C2 with a composition of `verify_import` + `verify_geometry`,
I then found that composition is itself a function:
`session_checkpoint.verify_restore`, same order, same stated reasoning
("Identity failing is the more fundamental problem, so it is the message the
caller sees"), and specified as such in DESIGN_410 §7.

So `check_compatibility` now calls `verify_restore` and adds exactly one thing
of its own: the #726 layout-gap note on an identity refusal. Two rounds of
removing my own duplication, one level apart.

## C4 — the design/code contradiction I reported does NOT exist in the repo

Cut 1's note said DESIGN_410 claims "content only, no placement" while the code
binds geometry, and proposed fixing the design text.

**That was wrong.** The repo document (`docs/dev/DESIGN_410_session_checkpoints.md`
§7) already states the shipped behaviour: `verify_restore` = `verify_import`
**then** `verify_geometry`, with cross-geometry "refused **by name**, naming
the offline `hicache_migrate --manifest` route, and … never silently
converted."

The "content only" text is in a **different, older draft** in the evidence
directory. I compared code against a stale copy and reported a contradiction
that the shipped doc does not contain. No design text needed fixing; the brief's
item 2 is moot, and saying so is better than editing a correct document to
match a claim I got wrong.

## C5 — Slice 2 is still absent, so the dangling reference stays

Re-checked on this base: `3550e20acf` is not an ancestor and
`mem_cache/pin_ledger.py` does not exist. The export refusal's "re-pin the
checkpoint (#410 slice 2)" advice therefore still names a capability this
branch lacks. Kept, honestly, per the standing instruction.

## What Cut 2 actually added

`import_bundle` now gates on the tar's member **names** — which the directory
yields without extracting a byte — so an incomplete or incompatible bundle is
refused before any payload is read, and nothing is returned for a caller to
have begun seeding from. **Partial-seed-then-fail is unreachable**, which is
the failure direction the brief named: a gate after extraction is recoverable,
a gate after seeding is not, because a half-seeded session decodes wrong
rather than failing.

22 pins. Mutation: dropping the gate fails 6, including every partial-seed pin.

Unchanged: this is still the file layer. Handing `(manifest, pages)` to the
live seeding path is the next cut, and cross-rig proof remains the filed
window item.

---

# CUT 3 — the seeding wire, and what the paused WIP was worth

## The predecessor: harvested in one place, superseded in another

`0dc48c92d8` ("#411 WIP: session bundle container, PAUSED mid-slice") is my own
earlier attempt, on a branch this base does not carry. It built
`mem_cache/session_bundle.py` — the same feature I rebuilt as
`managers/session_portable.py` without knowing it existed. Checking it was the
prior-art gate working on my own output.

**HARVESTED: per-payload digests.** The WIP had `_digest`, `verify_bundle` and
typed `BundleTruncated` / `BundleIntegrityError`. My version had no integrity
check at all. Content-addressing does NOT give this for free: the key names
what the bytes *should* be and nothing re-checks that they are, so a truncated
or corrupted transfer would have seeded a wrong session rather than failing.
`export_bundle` now writes a `digests.json` member and `import_bundle` verifies
it after extraction and before returning. Absent digests are tolerated (an
older bundle is readable, it simply carries no integrity claim) and that
tolerance is stated rather than silent.

**SUPERSEDED: `CompatKey`.** The WIP carried its own compatibility key
(`model_identity`, `num_attn_layers`, `kv_cell_bytes`, `page_size`, `mamba`)
documented "No geometry". That is a THIRD authority for a rule
`verify_restore` already owns — and it encodes the same "no geometry" reading
I retracted in C4. Explicitly superseded, not merged.

One thing in it is worth remembering rather than adopting: `num_attn_layers`
and `kv_cell_bytes` are exactly the axes the identity hash does NOT cover
(#726's layout gap). Adding them is a change to the compatibility key itself,
which is a design decision for the key's owner, not something a container
should smuggle in.

## The wire

`SessionCheckpointRuntime.import_bundle_and_seed` composes the lifecycle
branch/rewind already use — `_preconditions` → gate → `controller.branch_from`
→ ledger — rather than a parallel path. A second seeding path would be a
second place for the mid-generation rule, the TP=1 limit and the page_head
refusal to be got wrong.

**ORDER IS THE GUARANTEE.** Every refusal precedes `branch_from`, the only
step that creates a session, so a failure anywhere leaves nothing to roll back
— not "a rollback that works", but no state to roll back. The M7-style
ordering mutation (seed before the store verification) turns that pin red.

**Two verifications, and the second is not redundant.** The bundle's own gate
runs against its member names; `verify_restore` then runs against the STORE.
Between those moments the blobs move. Passing the first proves the bundle is
complete; passing the second proves this rank can actually read what it was
handed. A store that silently took nothing fails the second and seeds nothing.

**Store writes before the gate are not session state.** They are
content-addressed cache entries — harmless if the import then refuses, and
what makes the store-backed verification possible at all. The mid-generation
check runs *before* them, because a refusal that has already spent store IO is
a worse refusal.

## The C1 direction is now a standing falsifier

A hybrid bundle whose mamba blob is missing refuses at the gate and never
seeds. That is the failure Cut 1 would have shipped — KV pages travelling
without the recurrent state, the destination silently re-prefilling, a wrong
session rather than a slow one. It is pinned as a falsifier that stays, not as
a one-off regression test.

## Unchanged

Slice 2 (pinning) is still absent on this base — re-checked, `pin_ledger.py`
does not exist — so an imported bundle's checkpoint is not pinned on seed and
the dangling-reference note stands. Cross-rig proof remains the filed window
item; nothing here has crossed a machine.

---

# CUT 4 — pin-through-import, and C5 closes

## C5 is closed, by the dependency arriving rather than by an argument

Cut 2 recorded that the export refusal's advice -- "re-pin the checkpoint (#410
slice 2)" -- named a capability the branch lacked, and filed the Slice-2
dependency as the one "that decides whether the refusal is rare or routine".

On this base that dependency is satisfied. `mem_cache/pin_ledger.py` exists,
`LRUFileEvictor` honours it, and `take_file_tier_pins` is wired into the
checkpoint path. The advice is now actionable, so the dangling reference stops
dangling. See `NOTE_411_reconciliation_port.md` for how the ledger got here and
`VERDICT_411_two_410_formats.md` for why there were two #410 lineages to
reconcile first.

## Why import needs its own pin, when it already verifies

`import_bundle_and_seed` materialises the payloads and then runs
`verify_restore` against the store. **Verification proves the pages are there
NOW; it does not stop the evictor reclaiming them a moment later.**

An imported bundle is the worst case for that. Its pages are freshly written --
so they are the youngest entries in the LRU, which sounds safe -- but they are
referenced by no running request until the seeded session is used. A store
under pressure can evict a just-imported session before it is ever read, and the
user sees a session that will not continue.

## The ordering, which is the same rule as the checkpoint path

    materialise -> verify_restore -> PIN -> branch_from

AFTER verification, because there is nothing to pin until the pages are in the
store. BEFORE seeding, because a session that cannot be protected must not be
created: the alternative is a live session whose prefix is already reclaimable,
which fails later, further from the cause, and to a user rather than an
operator.

Note the asymmetry with the radix pin: `_pin` runs AFTER `branch_from` because
it needs the seeded node to lock. The file-tier pin does not, so it goes first.
Two tiers, two positions, each chosen by what it needs rather than by symmetry.

## The GDN blob is part of the pin set, not an extra

#212: a KV-only prefix is worth zero usable tokens on a hybrid model. A session
whose mamba blob was evicted is not partially usable, it is unusable, so
`pin_imported_pages` appends `mamba_key` to the KV keys. Mutating that line
away reds `test_the_gdn_blob_is_pinned_too` and nothing else, which is the
proof that the clause is load-bearing rather than decorative.

## A target with no file tier is reported, not refused

#407 may hold the imported session on vram or host, where there are no stems to
pin. That is logged as unprotected -- "survives radix eviction only, not a
restart" -- exactly as the checkpoint path does. Refusing a placement the tier
policy chose would be this layer overruling the one whose decision it is.

## A test defect worth recording

The first version of the ordering test asserted
`src.index("pin_imported_pages") < src.index("branch_from")` and failed against
correct code: the method's own docstring names `branch_from` while describing
the order, so the bare-name index found the prose, not the statement. Fixed by
matching the CALL forms. Grepping a name inside source you are asserting about
finds documentation as readily as code.

## Unchanged

Cross-rig import remains unproven and stays the filed window item: everything
here is one machine, tmpdirs and an injected page reader. `IDENTITY_LAYOUT_GAP`
still stands -- the identity hash covers the kv-cache dtype NAME, not its byte
layout -- and is still quoted into every identity refusal.

---

# TP>1 VERDICT: the refusal stands, and per-rank merge is not desk-fundable

Asked to implement per-rank manifest merging if it was desk-fundable, and to
leave a named refusal plus a filed residue if it was not. It is not, and the
reason is structural rather than a missing function.

## The refusal already exists and is correct

`session_checkpoint.py:654` and `session_handover.py:608` both refuse
`tp_size != 1 or pp_size != 1` by name, and the checkpoint one already points at
the cause: "the manifest is rank-local and a TP>1 checkpoint needs the per-rank
manifest merging named as #261's follow-up". Nothing here needed adding; the
gate a reader would hit already tells them what is missing.

## Why merging is not a desk addition

A TP>1 export must collect every rank's manifest into one, because each rank
holds a different shard of the KV under its own suffixed stems. Collecting them
means a group collective.

`SessionCheckpointRuntime` is built to issue none. Its class docstring
(`session_checkpoint.py:548-552`) states the property and the reason:

> Runs entirely on the scheduler thread (control requests are processed between
> scheduling iterations), so the radix tree cannot mutate under a snapshot, a
> lock reference or a rewind ... **No group collective is issued anywhere:
> every operation is rank-local.**

That is not incidental. It is the whole correctness argument for why none of
this needs a lock: the runtime never blocks the scheduler thread waiting on a
peer. Adding a manifest-merge collective would put a rank-to-rank wait inside a
control request handled between scheduling iterations, which is how a control
path becomes a scheduler stall on the slowest rank -- and, if a peer is mid-
drain, a deadlock.

So per-rank merge is not "the merge function nobody wrote". It is a change to
the concurrency contract that makes the lock-free argument hold, and it belongs
with #261's drain coordination, where that contract is already being negotiated,
rather than bolted onto a runtime whose safety depends on the opposite property.

## Filed residue, with its dependency named

**TP>1 portable export.** Needs, in this order:

1. per-rank drain coordination (#261's follow-up), which is where the collective
   and its ordering already have to be designed;
2. a merged manifest whose page set is the union of the ranks' shards, keyed so
   an import can tell which shard a page belongs to;
3. import-side placement: the geometry gate refuses a differing `tp_size`
   outright today, so a TP>1 bundle either lands on an identical geometry or
   goes through the manifest-scoped umsharder -- which itself inherits the
   `page_size == 1` limit (`session_checkpoint.py:662`).

Item 1 is the blocker and it is not this module's to solve.

## What step 5 does deliver

* compat gate: built in Cuts 1-2 and verified here as a COMPOSITION of
  `verify_import` + `verify_geometry` rather than a second implementation --
  Cut 2 removed Cut 1's hand-rolled duplicate, and `IDENTITY_LAYOUT_GAP` is
  quoted into every identity refusal so a reader learns where the gate stops;
* pin-through-import: Cut 4 above, closing C5;
* TP>1: this verdict.

Cross-rig import remains the filed window item. Nothing in this note claims it.
