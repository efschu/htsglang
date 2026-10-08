
## Stage 4 (#1427, 16.09. evening): the write side is mapped

Stage 3 had only brought the read side in place; Writing continued
Card -> 1-GB-Staging-Ring -> Store-Thread -> Arena -> Rebinding, and the
Mamba states through a 13-slot host anchor pool. The park test (6 x 100k)
died at exactly this transit (xsn184-187).

* KV/Draft (`pool_host/arena_pool.py`): `write_backup` claims per page-
  hash one arena slot (`arena_claim`: fresh / join a previous shard
  join / already COMPLETE), the backup thread DMAs the K and V extents
  of this rank with the All-Layer-Kernel directly into the slot (arena data region
  as host buffer, page stride), the ack merges the extents into the
  coverage (`arena_complete`, COMPLETE as soon as each layer shard is inside),
  takes the reader reference and marks the node store-present. No
  staging, no ring, no store-write, no rebinding.
* Mamba (`pool_host/arena_mamba_pool.py`): views per local layer
  (temporal) and per Conv segment (q|k|v) onto the Mamba arena, exactly the slice
  from `temporal_extents`/`conv_extents`; Backup = index copy card -> view,
  Load = return path, Prefetch resolves COMPLETE blobs in place
  (`arena_resolve_reads`, hook in `_batch_io_v2`). Window parts come from
  `_canonical_mamba_window`.
* Refs to CLAIMED slots are allowed (the writer's node holds one from
  the claim); `free_slots` bumps the generation, a late completion
  is rejected.
* Still in transit: nothing for KV/Draft/Mamba; Sidecar pools (SWA/Indexer) do not exist
  on this model.

### Stage 4, metal lessons (xsn188-192, 16.09. evening)

* Binding BEFORE the check (#1427e): `pool.arena` is only set after `ensure_bound`
  set; whoever checks first and then binds lets every rank without Prefetch
  (PP1/PP2 never read the store) on the staging ring. Same for the
  Draft pool (#1427g, otherwise only binds at the first draft read, which P never does)
  and the Mamba pool of the newly built D group (#1427c, window parts after the
  Rebind to the CURRENT Pool).
* An unbound pool emits no placeholders and writes no foreign
  Arena-Ids into its staging buffer (#1427c/#1427g).
* Not in the Arena = Miss, never disk-read into a placeholder (#1427d);
  a Raise in the Storage-Thread left the request waiting forever on
  `pool_transfers_done` (#1033e: Flag on every exit).
* `op_fn is self._read_page` compares two fresh Bound-Method-Objects and
  is always False (#1427e) -- compare by name.
* KV-Legs of the Front need the same Epoch-Retry as the Weight-Legs
  (#1428); a dropped Keep-alive would otherwise kill the Boot on the first Wake
  even though P answered 200.

## Addendum 16.09. late: Layout-Ladder per size class, YaRN per class (Design, user order)

**Is (after #1447):** The P-Cut-Solver prices every Frontier-Candidate with its
Host-Bounce (`PP-CUT HOST-PRICE`), the Cap-Floor (262144 + Chunk) is on the
Exchange-Arm standard, and it ships the fastest fundable cut. Fundable
is today only 39,13,12, because only its lane set is surveyed (5 Lanes, 15,75 GiB);
each unsurveyed cut is priced at the worst-case (9 Lanes, 27,75 GiB) and
falls on the ledger (Excess −1,3 GiB). The `PDFLIP-XCHG-HOST-SLOT` instrument, from which the
Lanes are learned, has delivered no rows since Ring-off (0 in xsn203–208) —
that is the single wall before the Ladder: **first re-wire Lane-Measurement, then
can a second cut be measured at all** (Boot with
`FLLIPER_PDFLIP_PCUT_BOUNCE_SLACK_GIB` and Ledger-headroom, after that it is in
`XCHG_LANES_BY_CUT` bzw. dem Record).

**Ladder (Design):** Size classes by estimated Prompt-Length of the Frontend
(`est_prompt`, already present): ≤32k, ≤64k, ≤128k, ≤262k. Per class the fastest
cut of the frontier, whose pool holds `Klasse + Chunk` (bs1: the pool only needs the
single Prefill). The Solver already delivers the whole Frontier per Boot; the
Ladder is a table `Klasse → (Schnitt, Pool, ms/Chunk)` from the same computation.

**Re-Cut at runtime:** a P-internal Layout-Flip (PP-Stages swap Layers over
the same Host-Bounce-Lanes as the P↔D-Exchange, no D-party involved), triggered by the
Frontend, if the next Batch-class demands a different cut than the
installed; cost ~1 Flip (1,6–2,0 s). Only if the class backlog the Flip
(same Break-even-Rule as `FLIP-ECONOMICS`).

**YaRN per class:** RoPE-Scaling is load-time configuration of the Model; a
factor change at runtime would mean re-loading. Therefore: factor fixed per Boot
(`--json-model-override-args rope_scaling`), classes >262k (factor 2/3/4 → 524k/786k/
1M) only with one boot of this factor; D-Pool 697856 holds ~2,6×262k, P needs for that
the Pool-heavy cuts (33,18,13 ff.). Automation = Boot-Choice, not runtime-Switch.
