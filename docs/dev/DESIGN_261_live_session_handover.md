# #261 (second half) — live session handover without server stop, and the draft re-sharder as its own spec type

Status: design + desk implementation record. The first half of #261 (offline
round-trip, both directions, byte-identical at real blob sizes) is merged;
this document covers the remaining two items and persists the decisions
(feature-analysis duty).

Companion code:
- `python/sglang/srt/managers/session_handover.py` — session-scoped five-phase
  runtime (source and destination sides).
- `python/sglang/srt/mem_cache/draft_migrate.py` — `DraftBlobSpec` (the second
  blob-spec type after `MambaBlobSpec`) + the per-algorithm capability
  registry.
- `python/sglang/srt/mem_cache/hicache_migrate.py` — gains manifest scoping and
  the draft pool branch.
- `scripts/handover/live_handover_probe.py` — the card-window byte gate driver.

## 1. What "live" means here, and what it reuses

The merged flow requires stopping the source server for exactly one reason:
`hicache_migrate` reads raw files out of the store directory with no
coordination against a live writer. Nothing about the migration itself needs
the process dead — the store files are content-addressed and immutable once
written (atomic tmp+rename in `HiCacheFile.set`).

The live path therefore does NOT invent a new transport. It scopes the
existing store-based flow to ONE session and adds the coordination the offline
flow got for free from the process being dead:

- **park-now**: a control command that quiesces one session, force-flushes its
  KV pages and GDN state to the store, and returns a MANIFEST naming every
  byte that constitutes the session.
- **manifest-scoped umsharder**: `hicache_migrate --manifest` reads exactly
  the manifest-listed files (complete and immutable, because the session is
  parked) and ignores everything else — so it is safe against a source store
  that other sessions keep writing to.
- **request-less import**: a destination control command that verifies the
  migrated keys are present in ITS store and that the model identity matches,
  without needing a generate request.

Both servers keep serving all other sessions the whole time; nothing global
stops, no collective is issued anywhere in the path (see §5).

## 2. The #329 five-phase vocabulary at SESSION scope

| Phase | Server scope (#329, design-only) | Session scope (this feature, built) |
|---|---|---|
| QUIESCE | drain all forwards, park drafter | refuse if the session has an in-flight request; park the prefix (new requests extending it are refused with a named error); eviction-lock the radix chain |
| SNAPSHOT | serialize KV + GDN + session table | force write-through + storage backup of the session's chain; bounded drain; manifest with exists-verification (the GDN gate lives here) |
| RE-FORM | rebuild communicators | run the manifest-scoped umsharder into the destination geometry — outside the source process, source state untouched |
| RESTORE | reshard into new geometry | destination verify-import: every manifest key present in its store, model identity hash equal |
| RESUME | recapture graphs, reopen admission | destination serves the next turn (storage prefetch hit); source commit releases the lock and the parked prefix |

**Rollback rule (mirrors #329 §5):** rollback is legal at every point BEFORE
the source receives `commit`. Abort = unlock + unpark; the session continues
on the source as if nothing happened, because nothing on the source was
mutated — the snapshot only ever ADDED files to the store. After `commit`
there is deliberately no rollback: the destination has confirmed possession,
and two servers both believing they own a session is the failure mode the
state machine exists to prevent.

## 3. The manifest

JSON, produced by the source, consumed by the umsharder and the destination:

- `model_identity_hash` — `compute_model_identity_hash(server_args)`; the
  #241 gate crossed VISIBLY: destination refuses on mismatch, geometry fields
  are the only thing allowed to differ.
- `geometry` — source tp_size/tp_rank, dcp_owner_mode, page_size.
- `token_ids` — the exact prefix handed over (the destination replays these).
- `kv_keys` — every page hash of the chain, root to leaf, in order.
- `mamba_key` — `{leaf_last_hash}.mamba`, or null for non-hybrid models.
- `hybrid_gdn` — whether the model REQUIRES the mamba blob. This is the #212
  falsifier hook: for a hybrid model a missing mamba blob fails the export
  LOUDLY (the store route would otherwise truncate the match via
  `MambaRadixCache._match_post_processor` and silently re-prefill).
- `draft_keys` — `.draft` component keys present for the chain (informational;
  disposition is decided at migrate time, see §6).

## 4. Declared v1 limits (named, not hidden)

- **Source export: TP=1 only** (attn_tp_size == 1, no PP). The primary
  direction of the merged flow (fast card → group). A TP>1 live export needs
  per-rank manifest merging and per-rank drain coordination; refused with a
  named error, the stop-based flow covers it.
- `file` storage backend, `page_size == 1`, hierarchical cache enabled — the
  same limits `dcp_owner_mode` and the offline flow already carry.
- The whole prefix must be device- or host-resident at export time (match
  length == prefix length); partial residency is refused rather than guessed.
- Destination verify-import checks presence + identity, not byte sizes; size
  gates live in the umsharder (hard errors naming both numbers) and in the
  read path (short read = error).
- Branched sessions (the prefix has forked children in active use) are legal —
  the chain is walked root→leaf along the matched path; sharing with other
  sessions is safe because the snapshot only reads and the store is
  content-addressed (a shared page is simply already present).

## 5. Collective-family discipline

The rank-local-condition-before-collective audit applies vacuously by
construction: the live handover path issues NO group collective anywhere.

- Source side is a single scheduler process (TP=1 limit).
- Destination verify-import fans out via the existing ZMQ
  `FanOutCommunicator`; each rank answers from rank-local `exists()` checks
  only. The gather is bounded by the tokenizer manager's own timeout
  machinery, not by a device collective.
- All waits are bounded: the snapshot drain reuses the
  `_flush_pending_storage_backups_before_reset` pattern with an explicit
  deadline (#259/#312 discipline) and fails loudly with queue depths on
  timeout — and the failure path unlocks and unparks (rollback), never leaves
  the session wedged.

## 6. Draft re-sharder — the second spec type

The settled verdict stands: draft KV is the exact MIRROR of owner-mode target
KV (head-sharded, token-complete), so the skip-instead-of-rename in the
offline flow is correct and stays the default for the whole-store CLI path.
What was missing is the real umsharder for the non-mirroring case, and an END
to silent skipping on the handover path.

**`DraftBlobSpec`** (`draft_migrate.py`) mirrors `MambaBlobSpec` exactly as
the #261 module docstring sketched:

- Declared fields: draft `num_layers` (independent of the target model),
  `num_kv_heads` (total), `head_dim`, `itemsize`, `mem_layout`, `units`.
- With `page_size == 1` (already forced), `layer_first` / `page_first` /
  `page_first_direct` all flatten to `[2][layer][head][dim]`: a head shard is
  ONE contiguous range per (kv-half, layer) — the same extent family as
  `temporal_extents`. `page_head` puts heads outermost and is REFUSED with a
  named reason until someone builds its case.
- Head-shard widths come from `partition_sizes(...)` imported from the
  runtime — same anti-drift rule as the GDN split.
- The two rank-independent configurations (`attn_kv_replicated`, MLA draft)
  are a declared KEY-REWRITE mode (`--draft-kv-replicated`) — visible only in
  the caller's config, never inferred from the store.

**Capability registry** (`DRAFT_RESHARD_CAPABILITIES`), keyed by the canonical
`SpeculativeAlgorithm` names — ONE source (#379): the name is resolved through
`SpeculativeAlgorithm.from_string()` / `SPECULATIVE_ALGORITHM_ALIASES`, so an
unknown name is refused at parse time with the full `known_names()` list, and
this module can never grow a second name list that drifts.

| Algorithm | Capability | Reason |
|---|---|---|
| EAGLE (incl. NEXTN alias) | RESHARD | plain-MHA linear-append draft pool; host page layout is `MHATokenToKVPoolHost.get_data_page` |
| NGRAM, NONE | NO_DRAFT_KV | no draft model KV exists; "nothing to re-shard" is a verdict, not a skip |
| EAGLE3, STANDALONE, DFLASH, DSPARK, FROZEN_KV_MTP | REFUSE | store-layout / pool-relationship not modeled; refusal names the algorithm and the reason (never silent conversion — the #411 contract) |
| plugin-registered names | REFUSE | valid algorithm, but no declared re-shard capability |

Capabilities are declarations we can back with the code we read, nothing
more. Widening a REFUSE row to RESHARD requires modeling that algorithm's
store behavior first — that is the point of the registry.

**No more silent skipping on the handover path:** when `--manifest` is given
and the manifest lists draft keys, the CLI REQUIRES an explicit disposition —
either `--draft-spec-algorithm NAME` (+ geometry flags; re-shards, or refuses
per the registry) or `--draft-cold-start` (explicit, printed acknowledgment
that the draft pool starts cold). Neither present = hard error naming the
choice. The legacy whole-store path keeps its default skip (already printed,
never silent) for backward compatibility.

Flag spellings are canonical from birth (#382): `--draft-spec-algorithm`
matches `--speculative-algorithm`; no aliases are introduced.

## 7. Falsifiers (all hermetic, CUDA_VISIBLE_DEVICES=99)

1. **Planted GDN omission**: hybrid manifest whose mamba blob is absent from
   the store → export gate MUST fail loudly (and unlock/unpark). Also run
   with the blob present → passes (the gate can fail AND can pass).
2. **Planted non-mirroring draft layout, undeclared**: manifest with draft
   keys, no disposition flag → hard refusal naming both options.
3. **Declared but incompatible algorithm** (e.g. DFLASH) → refusal naming the
   algorithm and reason.
4. **Unknown algorithm name** → parse-time refusal listing known names.
5. **With the re-sharder declared (EAGLE)**: 1→N→1 draft round trip is
   byte-identical under `verify_plan` (every byte consumed exactly once);
   wrong ratios / wrong blob size are hard errors naming both numbers.
6. **State machine**: abort after commit refused; export with in-flight
   request refused; new request against a parked prefix refused with a named
   error; double export refused; commit/abort of an unknown handover refused.

## 8. Card-window byte gate (the oracle, #124 discipline)

`scripts/handover/live_handover_probe.py`: source A (TP=1) and destination B
run SIMULTANEOUSLY. Drive a determined-answer conversation on A; between
turns: export → migrate → verify-import → continue on B; reference arm = the
same conversation continued on A (never migrated). A-vs-A floor first (A
continued twice must be identical), probes short enough to stay under the
known GDN-prefill non-determinism onset. Both servers serve unrelated traffic
throughout the window to prove "live". This is a GPU-window task and is
recorded as such; the desk half proves everything that does not need a card.

## 9. Open items (named)

- Live export from a TP>1 source (per-rank manifest merge + drain
  coordination).
- Destination-side byte-size probe in verify-import (today: presence +
  identity; sizes are gated in the umsharder and at read time).
- The card-window trajectory gate itself (§8) — scripts ready, window
  arbitrated separately.
- `page_head` draft layout case.
- #411 is cited from the task briefing as the never-silent-conversion
  contract; the number has no anchor in this repository's history — the
  contract is enforced here regardless.
