# ANALYSE #329 cut 2 — the five phases against existing code

Question: how much of the quiesce/snapshot/restore/resume machinery already
exists **callable in-process**, and which phase lacks an implementation?

Answer up front: **RESTORE is the phase that does not exist in-process.**
QUIESCE and RESUME have every part (one of them unwired), SNAPSHOT has all but
a general per-session KV export, and nothing composes the four.

## The map

| phase | sub-capability | status | file:line |
| --- | --- | --- | --- |
| QUIESCE | stop admitting | **EXISTS**, in-process, wired | gate `tokenizer_manager.py:662-663`; toggle `:1772-1791`; scheduler `scheduler.py:7915`, loop gates `:2293`, `:2356` |
| QUIESCE | drain to a boundary (rank-local) | **EXISTS**, two variants | strict idle `scheduler.py:6603`; loose "parked" `phase_flip_runtime.py:528` |
| QUIESCE | group-wide quiescent consensus | **EXISTS but embedded** | `phase_flip_runtime.py:2461-2650`; coupled to the PP↔TP flip, needs lifting, not calling |
| QUIESCE | park the drafter | **EXISTS, UNWIRED** | `speculative/runtime_draft.py:185-315`; zero production callers, pinned by `test_unwired_features_421.py:144` |
| SNAPSHOT | weights → disk | **EXISTS**, in-process | `park_weights_to_disk` `model_loader/hibernate.py:474-605`, writer `sparse_write.py:273` (#456) |
| SNAPSHOT | GDN/mamba state | **EXISTS**, in-process, tested | `MambaPool.export_state_blob` `mem_cache/memory_pool.py:970-1023` |
| SNAPSHOT | buffers incl. non-persistent | **EXISTS**, general rule, twice | `weight_updater.py:361-366`; `translator/ledger.py:436-446` (post-#568) |
| SNAPSHOT | KV per session, general export | **EXISTS**, other lineage | `session_handover.export_session_snapshot:415` on `train/0818-desk-410-reconcile`; see `ANALYSE_329_per_session_kv_determination.md` |
| RESTORE | **weights, in-process** | **MISSING** | `HibernateModelLoader.load_model` `model_loader/loader.py:2468-2545` builds a NEW skeleton via `_initialize_model` `:2529` — cold-process shaped, selected at parse time `server_args.py:16097-16109` |
| RESTORE | GDN/mamba state | **EXISTS**, in-process | `import_state_blob` `memory_pool.py:1024-1055` |
| RESTORE | buffers | **EXISTS**, in-place into the live model | `_import_static_state` `weight_updater.py:369-374` |
| RESTORE | KV per session, general import | **EXISTS**, other lineage | `session_handover.verify_import:253` (present on both lineages) |
| RESUME | bring graphs back | **EXISTS**, two mechanisms | recapture `model_runner.py:2822-2830` → `:3643`; **VMM remap** `weight_updater.py:228-245` + `torch_memory_saver_adapter.py:50-92` |
| RESUME | unpark drafter | **EXISTS, UNWIRED** | as above |
| RESUME | re-open admission | **EXISTS**, wired | `tokenizer_manager.py:1787-1791`; `scheduler.py:7998-8018` |
| — | an orchestrator composing the four | **MISSING** | nearest: the weight-update round trip, and `release/resume_memory_occupation` `weight_updater.py:184-273` |

## Three findings worth keeping

**1. RESTORE is cold-process shaped, and that is the real gap.** #89 can park a
live process and restore a *new* one. Cut 2 needs the same process to take its
own state back. Everything else on the RESTORE row already works in-process;
only the weights leg assumes a skeleton it is about to build.

**2. RESUME should prefer remap over recapture in THIS cut.** Because
membership does not change, the captured shapes are provably identical, so the
VMM pause/resume path restores the graph pages without capturing anything.
True recapture is the fallback for a caller whose shapes moved — which in cut 2
would itself be a bug. The seam is therefore named `restore_graphs`, for the
outcome rather than one mechanism.

**3. A correction to an earlier reading of mine.** I first reported that the
#568 `inv_freq` fix does **not** generalise, citing
`NON_CHECKPOINT_NAME_PATTERNS = ("workspace",)` in `phase_flip_boot.py:67`.
That is the wrong subsystem: it is the weights-arena exclusion, not the buffer
fix. The actual #568 fix (`translator/ledger.py:436-446`) iterates
`module.named_buffers()` and carries **every** buffer `state_dict()` omits —
a rule over the persistence property, not a list of names — and
`weight_updater._export_static_state` (`:361-366`) never had the bug at all,
capturing `named_buffers()` directly. **The fix generalises.** The separate
"19 buffers" figure is not corroborated by static inspection
(`ANALYSE_spill_matrix_20260804.md:263` finds 2 statically and calls the count
runtime-logged).

What remains true and is why the class is still in the manifest: a *future*
implementation that reaches for `state_dict()` would silently drop these
buffers again, and `to_empty()` would hand back allocator garbage — NaN
`cos`/`sin`, NaN logits. The asset class names the rule so a third
implementation cannot regress quietly.

## Not conflated

`OFFLOAD_CLASSES` / `AssetClassDescriptor` (`offload_register.py:113-144`,
`short_term_offload_register.py:220-491`) is the **VRAM offload-priority**
registry — what may be evicted under pressure and in what order.
`FEATURE_CATALOG.md:3288-3304` warns explicitly that the "#286 offload
register" is two modules and that conflating them is a hazard. It is not
#329's asset inventory, and `world_roundtrip.ASSET_CLASSES` is deliberately its
own list with its own reasons.

## What cut 2 built here, and what it did not

Built (desk-fundable): the phase machine, the asset ledger, the completeness
gate, the membership guard, the rollback guarantee and the no-reflex rule —
`managers/world_roundtrip.py`, with every card-touching step an injected seam
so both arms of the falsifier run hermetically.

Not built, and named rather than smuggled:

* the in-process weight RESTORE (the missing phase above) — it needs a live
  model to take bytes back without rebuilding a skeleton, and that is a
  serving-path change requiring review and a boot;
* wiring the two `world_roundtrip` snapshot seams to the EXISTING per-session
  mover -- blocked on Slot-3's #410 A+B reconciliation, see
  `ANALYSE_329_per_session_kv_determination.md`;
* wiring the drafter lifecycle, which has been unwired since #309 and is pinned
  as such;
* the byte-identity window falsifier: round-trip identity of KV and GDN state
  against the pre-park state, on a card, in the #124 harness shape. Note the
  standing limit before designing it — GDN prefill is non-reproducible above
  ~109 tokens (`FEATURE_CATALOG.md` §18.6), so the identity assertion belongs
  on the STATE BLOB, not on generated output.
