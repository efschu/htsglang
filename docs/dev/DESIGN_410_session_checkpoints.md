# DESIGN 410 — server-side conversation checkpoints, branching and rewind

Status: **desk-complete, BOOT-PENDING.** Every gate and every accounting rule
below is proved hermetically (83 tests, `CUDA_VISIBLE_DEVICES=99`). Nothing in
this document has run on a GPU. The byte gate that would license the
performance and correctness claims is specified in §8 and has NOT been run.

## 1. What it is

A checkpoint freezes one server-side session — its KV pages **and** its
GDN/Mamba state — so the conversation can later be:

* **branched**: a new session continues from the checkpoint, the original
  continues independently;
* **rewound**: the same session's continuation point moves back to the
  checkpoint;

in both cases **without re-prefilling** the shared prefix.

## 2. Reuse map

The feature is a re-aim of machinery that already exists and is already
byte-proven. Nothing here is a new serialization, a new gate family or a new
sharing mechanism.

| What | Where it comes from | File:line |
| --- | --- | --- |
| Snapshot (flush KV + GDN to the store, emit manifest) | #261 live handover, SNAPSHOT phase, extracted verbatim to module scope | `python/sglang/srt/managers/session_handover.py:271` `export_session_snapshot` |
| Manifest format (versioned) | #261 `build_manifest`, unchanged; #410 adds an ADDITIVE `checkpoint` envelope | `session_handover.py:176`, `session_checkpoint.py:131` `build_checkpoint_manifest` |
| #212 GDN gate (recurrent state must travel explicitly) | #261 `validate_manifest_completeness`, called by the shared snapshot | `session_handover.py:209` |
| Hybrid-model detection | `BasePrefixCache.supports_mamba()` — the one source, per the #261 gate fix | `session_handover.py:405`, `base_prefix_cache.py:370` |
| RadixKey sequence-type discipline (`array("q")`) | #261 gate fix, factored so every control-plane caller shares it | `session_handover.py:184` `match_prefix_result` |
| #241 identity gate (model / dtype / quantization / kv-cache dtype) | #261 `verify_import` | `session_handover.py:253`, `hicache_storage.py:27` |
| Storage preconditions + bounded drain | #261, extracted to module scope | `session_handover.py:181,215` |
| Tier target | #407 registry via the consumer shim | `memtier/consumers.py:175` `checkpoint_tier_targets` |
| Page sharing | `UnifiedRadixCache` prefix sharing + `inc_lock_ref` — nothing new | `unified_radix_cache.py:819,1207` |
| Token splice (branch/rewind) | `SessionParams.offset`, the controller's existing history rewrite | `session/session_controller.py:203` |

The #261 refactor is behaviour-preserving: its 23 existing tests pass
unchanged against the extracted functions.

## 3. API surface

| Route | Body | Returns |
| --- | --- | --- |
| `POST /session/{id}/checkpoint` | `{durable?, label?, token_ids?, deadline_s?}` | `checkpoint_id`, full manifest |
| `POST /session/{id}/branch` | `{checkpoint_id, new_session_id?}` | new `session_id`, page-sharing accounting |
| `POST /session/{id}/rewind` | `{checkpoint_id}` | accounting, confirmation |
| `GET/POST /session/{id}/checkpoints` | — | the session's checkpoints, oldest first |

A body `session_id` that disagrees with the URL is refused rather than
silently preferring one of them. Internally all four are one
`SessionCheckpointReqInput` with an `action`; `drop` exists on the control
struct and has no route yet.

Flags: `--enable-session-checkpoints` (default **off** — the routes then
refuse and nothing else in the server changes),
`--session-checkpoint-vram-max-age-s` (60),
`--session-checkpoint-host-max-age-s` (900).

## 4. Checkpoint identity

`checkpoint_id = handover_id_for(token_ids, model_identity_hash)` — the #261
derivation, unchanged. Content-addressed, so:

* checkpointing an unchanged prefix twice returns the SAME id and is
  idempotent (the blobs in a content-addressed store are identical anyway);
* distinct turns diverge in their token ids and get distinct ids with no
  counter to keep;
* the same prefix under a different model or kv-cache dtype is a different
  id, because the #241 hash is part of the input.

## 5. Tier selection (#407's first production consumer)

`checkpoint_tier_targets(registry, bytes_needed, age_s, durable)` returns an
ordered candidate list or an itemised refusal. Three payload facts drive the
query:

* **payload class** is `EXPENSIVE_RECONSTRUCTABLE` normally — a checkpoint
  can be rebuilt, but only by redoing user-visible work — and
  `PERSISTENCE_REQUIRED` when `durable` is set or the checkpoint is older
  than `host_max_age_s`. Only a `PERSISTENT` tier admits the latter, so host
  RAM is refused **by name** for a durable checkpoint;
* **no `object_class`**. `OFFLOAD_CLASSES` has no member for KV pages, and
  `TierQuery.object_class` documents that case ("hibernate images and HiCache
  pages do not [have one], and are gated by volatility alone"). Inventing a
  member would make every existing tier record refuse checkpoints until its
  `admits` set was edited;
* **the GDN blob does NOT make this `DEVICE_BOUND`.** That class exists
  because a *lossy or reordered* round trip of recurrent state is a
  correctness failure (DESIGN_407 X2), and it would pin every checkpoint to
  its origin card — the opposite of a checkpoint. The #261 route is neither
  lossy nor reordered: the state travels as its own explicitly named `.mamba`
  blob through the content-addressed store, and `validate_manifest_
  completeness` refuses a hybrid-GDN manifest without it. **This is a
  deliberate classification decision, not an oversight**; it is the one place
  where #410 reads the #407 taxonomy rather than following it literally.

Age narrows the KINDS considered (VRAM → RAM → Disk) rather than re-ranking
them; the registry still does all the picking, with its own provenance. The
two age thresholds are POLICY and carry no provenance, because there is
nothing to measure about them.

## 6. Branching shares pages; it does not copy them

Catalog-first: the radix tree already gives the whole mechanism. Two sessions
with a common token prefix walk the same `UnifiedTreeNode` chain, and
`_split_node` splits at the divergence point so the common ancestor keeps the
**same device page indices**. Copy-on-restore is what the tree does anyway.

#410's only contribution is the **pin**: `inc_lock_ref` on the matched node,
so the shared chain cannot be evicted between the branch call and the
branch's first turn. Two branches of one checkpoint take one pin.

Accounting (`BranchAccounting`), reported on every branch and rewind:

* `copied` is **structurally zero** and asserted so;
* `shared` is what the tree still holds;
* `prefetch` is the honest remainder — pages the tree has evicted, which the
  next turn reads back from the tier. That is a storage read, not a
  re-prefill, and is reported separately from both.

## 7. Restore gates and the refusal matrix

`verify_restore` = #261's `verify_import` (manifest version, #241 identity
hash, every named blob present including the GDN blob) **then**
`verify_geometry` (tp_size, page_size, dcp_owner_mode).

Same-geometry rewind and branch need no umsharder — they land on the same
running group. Cross-geometry is refused **by name**, naming the offline
`hicache_migrate --manifest` route, and is never silently converted. The case
this exists for is real: a checkpoint read back from a PERSISTENT tier that a
previous boot wrote under a different geometry.

v1 limits, each a named refusal rather than a workaround:

* TP=1 / PP=1 source (inherited from #261: the manifest is rank-local, and a
  TP>1 checkpoint needs the per-rank manifest merging #261 named as its
  follow-up);
* `page_size == 1` (inherited from `dcp_owner_mode`);
* HiCache `file` storage backend only;
* `--hicache-mem-layout page_head` refused. Its host write-back dispatches to
  `transfer_kv_all_layer_lf_ph` (`pool_host/mha.py:401`), which segfaults
  (#441a). The remaining layouts reach `transfer_kv_all_layer_direct_lf_pf`
  (#436, direct IO) or the layer-first route. A checkpoint is a host-tier
  write of a WHOLE session, so it would take the broken route for every page
  rather than occasionally — same refusal family as
  `draft_migrate.REFUSED_LAYOUTS`. Refused at argument time AND on the
  control route, because a server can be reconfigured after boot;
* one request node per session — a non-streaming session *tree* must say
  which branch point it means via explicit `token_ids`.

## 8. GPU byte gate (specified, NOT run — BOOT-PENDING)

Two arms, both on the reference TP=1 hybrid-GDN recipe with hierarchical
cache + `file` storage, `page_size 1`, `--enable-session-checkpoints`.
Determined answers only (greedy, fixed seed), and an **A-vs-A floor measured
in the same boot before any delta is reported**.

**Gate A — resume trajectory is identical to a never-paused reference.**
Reference: one session, N turns, uninterrupted. Arm: the same session,
checkpointed between turn k and turn k+1, then continued. The token ids of
turns k+1..N must be **byte-identical**. On a hybrid-GDN model this is the
#212 falsifier at runtime: a truncated recurrent state produces a plausible
but different continuation, which only a byte comparison catches.
Negative control (must FAIL the comparison): the same run with the `.mamba`
blob removed from the store before the resume — if that arm also matches, the
gate is not testing what it claims to.

**Gate B — branch-then-continue equals fresh-prefill-then-continue.**
Reference: a fresh session prefilled with the checkpoint's exact token prefix,
then continued with the branch's turn. Arm: branch from the checkpoint, then
the same turn. Token ids must be byte-identical. Report alongside it: the
branch's `shared_pages` (expected = the whole prefix on a warm tree) and the
measured prefill token count of each arm — the point of the feature is that
the branch's is zero.

**Gate C — the parent is untouched.** After Gate B's branch, continue the
PARENT session and compare against a reference where no branch was ever
taken. Byte-identical, or the branch mutated shared state.

Also to be captured in the same boot: the checkpoint's wall time and the
`tier_id` + `tier_provenance` the registry actually chose on the rig
(the hermetic tests use fixture tiers; nothing here has seen a real
`for_machine()` result).

## 9. Honest state

* **Proved hermetically**: manifest round-trip and #261 compatibility, the
  #212 GDN gate in both directions, the geometry/identity/blob/tier refusal
  matrix, branch page-sharing accounting, the ledger's many-derivations
  semantics, the session splice (rewind consumed once, branch never mutates
  the parent), and the flag/TP/page-size/host-layout preconditions.
* **BOOT-PENDING**: everything in §8. No GPU has run this code. In
  particular the `TierRegistry.for_machine()` path, the real
  `inc_lock_ref`/`dec_lock_ref` on `UnifiedRadixCache`, and the storage
  prefetch that turns a `prefetch_pages` count into an actual read have only
  been exercised against fakes.
* **Not attempted**: TP>1 checkpoints, cross-geometry restore, checkpoints of
  a non-streaming session tree without explicit `token_ids`, and a `drop`
  HTTP route.
* **Caught by re-running the suite, not by writing it**: the argument-time
  validation was first written as a block inside `_handle_kv_pressure_ladder`,
  where `model_path="dummy"` — the repo-wide convention for argument tests —
  returns from `__post_init__` before it, so four gates asserted a `ValueError`
  that never fired. It is now its own `_handle_session_checkpoints`, invoked
  explicitly by the tests like every sibling handler, plus a source-level pin
  (`test_the_handler_is_wired_into_post_init`) that fails if the dispatcher
  ever stops calling it. A handler that exists and is not called is the
  desk-written-never-executed failure in miniature, and no constructor test
  can see it.
* **Incidental fix**: `Session._concat_token_arrays` used `+` to join token
  sequences, which is a `TypeError` for the `array("q")` histories the
  streaming path builds. #410's rewind is the first caller to take the copy
  path on a streaming session, which is where it surfaced; the concatenation
  is now type-preserving (`session_controller.py:36`).

---

# Addendum, 2026-08-17: pinning tiers, accounting, and the reconciliation

This section is the surviving record of a second #410 implementation that was
developed in parallel and is NOT being merged. See
`VERDICT_411_two_410_formats.md` for how the duplication happened and
`NOTE_411_reconciliation_port.md` for the port. The findings below came from
that lineage and are kept because they are true of this one.

## The pin tiers are two, and only one existed here

`SessionCheckpointRuntime._pin` takes `tree.inc_lock_ref(node)`: the radix
chain, in memory, for the life of the process.

**An A checkpoint survives radix eviction and does not survive file-tier
eviction.** The pages it references live in the HiCache file store as ordinary
LRU entries; nothing stopped the evictor reclaiming them and nothing survived a
restart. `take_file_tier_pins` now takes that second tier through the pin ledger
(`mem_cache/pin_ledger.py`), which `LRUFileEvictor` honours by skipping pinned
stems.

## Coverage is a refusal, not a statistic

`stems_with_sizes` drops a stem whose file is absent. That is right for the
budget -- charging for protection nobody gets is the #715 error -- and silent to
the caller. So a checkpoint could pin four of six pages and report success,
with the other two evicted and the failure surfacing at the BRANCH.

`PinCoverageIncomplete` moves that refusal to checkpoint time and names the
references. Pins are taken BEFORE the ledger record exists: a checkpoint that
cannot be protected must not become a record promising a branch whose pages are
already reclaimable.

A checkpoint the #407 registry places on vram or host has no file tier at all.
That is logged as unprotected, never raised -- refusing a placement the tier
policy chose would be this layer overruling the one whose decision it is.

## The evictor charges ALLOCATED bytes

A filesystem charges the blocks it allocated, not the length the file reports:
on this rig's ZFS a 64-byte page occupies 512. The pin ledger charges allocated
bytes, so while the evictor charged apparent ones `reclaimable = used - pinned`
subtracted two different units and could go negative. Both now charge allocated;
`stats()` exposes `accounting_overshoot_bytes` so a future divergence is visible
rather than clamped away.

`reserve` still estimates from the payload length -- the file does not exist
yet -- and `commit` reconciles against the filesystem's own answer.

## Interface note for the per-session KV export/import consumer

Slot-2 is building a per-session KV export/import slice against this module's
INTERFACE without modifying it. The seam it should wire to is
`world_roundtrip`'s `write_snapshot` / `read_snapshot` pair, mapped per session
onto `export_session_snapshot` -- the same #261 serialization this module
already uses, so an exported session and a checkpoint are the same bytes.

If #411 needs to change that interface, the change belongs to this lineage and
will be recorded here first, so that consumer is never chasing a moving seam.

## Merge-train note

Do not merge the parallel lineage's `mem_cache/session_manifest.py` or
`mem_cache/session_checkpoint.py`: they are a second format for the same
feature. Its unique contribution -- the pin ledger and the coverage refusal --
is already ported here. Its modules had no live-path importers (only the
superseded `session_bundle.py`, a self-reference, and their own tests), so
nothing is stranded by leaving them behind.
