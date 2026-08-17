# NOTE 745: the checkpoint writer and the resume path ALREADY EXIST — twice

**Determination: do not build steps (2) and (3) as specified.** Periodic GDN
snapshots at chunk boundaries, and "resume from the nearest checkpoint at-or-
below the hit position, then re-prefill the tail", are both implemented and in
the tree. They exist in TWO lineages, and the actual gap is that the two cannot
be combined and neither is enabled on the live boot.

The user's mechanism is right. It is also mostly already built, which the gate
is exactly there to catch.

## 1. What exists, with file:line

**Resume-from-nearest-checkpoint (#745 step 3) — BUILT.**
`_match_prefix_helper` walks the text-matched path and tracks the deepest node
that still carries a state, gated on the checkpoint grid:

```
mamba_radix_cache.py:1513-1517
    if node.mamba_value is not None and is_on_interval(
        cum_tokens, self.mamba_checkpoint_interval
    ):
        best_value_len = len(value); best_last_node = node
```

`_match_post_processor` then truncates the returned KV prefix to that point
(`:1685`, `value = value[:best_value_len]`), COWs the state into a fresh active
slot (`:1611-1654`), and the scheduler re-prefills the tail
(`managers/schedule_batch.py:2644-2660`, and `:2719-2733` for the interval
variant).

**Snapshot writer at chunk boundaries (#745 step 2) — BUILT.**
Retention happens at `cache_unfinished_req` (`mamba_radix_cache.py:771`, the
chunked-prefill boundary) and `cache_finished_req` (`:603`), attaching state to
the node at that depth (`_insert_helper`, `:1794`). Branch nodes deliberately
carry none: `_split_node` sets `new_node.mamba_value = None  # mamba cache can
not be split` (`:1699-1704`).

**Cadence flag — BUILT.** `--mamba-checkpoint-interval` (`server_args.py:4106`,
validated `:14048`) pins checkpoint positions to absolute multiples so resume
points are deterministic. `is_on_interval(pos, None) -> True`
(`mamba_ckpt_utils.py:34-38`), so an unset interval means every position is
eligible, not that checkpointing is off.

**Host tier for mamba state — BUILT, in the OTHER lineage.**
`HiMambaRadixCache` carries its own device<->host path for recurrent state:
`write_backup` (`hi_mamba_radix_cache.py:373`), `init_load_back` (`:553`), into
`self.mamba_pool_host`, with an H->storage archive path
(`mamba_archive_transfers`, `:2388`).

**Transport for a blob — BUILT.** Canonical self-checking GDN blob
(`gdn_slot_executor.py:150-213`, #555), `TieredGdnBlobStore` (`:214`, #711),
file-capable `DestinationTier` (`managers/kv_session_spill_destination.py:406`,
#224).

## 2. The actual gap: an explicit refusal, and two switches that are off

The two halves are MUTUALLY EXCLUSIVE by an explicit gate — this is the
file:line the prior-art rule asks for, and it refuses precisely the combination
#745 wants:

```
server_args.py:14066-14071
    if self.enable_hierarchical_cache:
        raise ValueError(
            "--mamba-checkpoint-interval does not support "
            "--enable-hierarchical-cache yet (HiMambaRadixCache has its "
            "own match/evict paths)."
        )
```

So today you may have deterministic grid-pinned checkpoints (VRAM-only), or a
host tier for mamba state (its own match/evict paths), but not both.

And on the LIVE boot both are off:

```
mamba_checkpoint_interval  = None      # no grid pinning (all positions eligible)
enable_hierarchical_cache  = False     # no host tier for mamba state
gdn_resident_state_slots   = None      # #364 ladder not active
max_mamba_cache_size       = 12
max_running_requests       = 4
chunked_prefill_size       = 512
```

## 3. Why the misses happen (corrects NOTE_740 §4)

Not "no state at that position" — that was my claim and it is retracted in
NOTE_740 §4a. Resume-from-nearest exists, so a full re-prefill needs NO state
anywhere on the matched path. The cause is **eviction**: 12 slots are shared
between the running requests and every retained checkpoint, under LRU
(`evict_mamba`, `mamba_radix_cache.py:1090`). Checkpoints do not survive to the
session's next turn.

That is the same conclusion #745 reached from the other end — the states need
somewhere to live beyond 12 VRAM slots — but it means the missing piece is
TIERING AND SURVIVAL, not a writer.

## 4. #212 is NOT obsoleted by #555, and #745 does not violate it

#212's verdict is encoded as a settled constraint:

```
gdn_slot_ladder.py:26-30
   THE BLOB IS OPAQUE AND NEVER TAKES THE RADIX ROUTE (#212). A recurrent
   state is positional and not prefix-shareable; a MambaRadixCache
   truncation of it is a silent full re-prefill.
```

The blocker was never the FORMAT, so a canonical blob (#555) does not obsolete
it: the objection is that a recurrent state cannot be TRUNCATED to a shorter
prefix. #745 never truncates — it anchors at exact positions and replays the
tail from the nearest anchor at-or-below. That is the operation a positional
state permits, and it is what `_match_post_processor` already does inside the
tree. **#745 is the composition #212 leaves open, not a re-litigation of it.**
Recorded here so a later reader does not mistake one for the other.

## 5. Step (5): the VRAM payoff is a BOOT knob, not runtime vacating

Two in-tree claims look contradictory and are both true, in different domains:

* `FEATURE_CATALOG.md:514-521` (DISPROVEN, do not rebuild): idle mamba/GDN slot
  states "free an INDEX, not memory" — `MambaPool` is plain `torch.zeros`,
  `MambaSlotAllocator.free` pushes to a free list; zero bytes NVML can see.
* `gdn_slot_ladder.py:10-15` (#364): the pool is BOOT-SIZED to the cap, so the
  surplus slots' bytes "are simply never taken from the hybrid memory budget
  and land in the KV pool through the existing route".

Reconciliation: **lowering the cap at boot returns real bytes; vacating a slot
at runtime returns an index.** The route is single-ledger —
`handle_max_mamba_cache` (`model_executor/model_runner_kv_cache_mixin.py:1999`)
spends from `total_rest_memory`, and what it does not spend the KV sizing gets
(`:2050-2068`). So step (5) must be delivered as a lowered boot cap
(`--gdn-resident-state-slots` or `max_mamba_cache_size`), and the freed MiB
**raise the binding stage's token ceiling**. Runtime idle-vacate is a
concurrency lever over fewer slots, not a source of MiB.

## 6. Idle-vacate x resume: already safe

`vacate_plan` takes `resumed_ids` and guarantees correctness before savings —
"restores are unconditional, vacates fill whatever room the restores need"
(`gdn_slot_ladder.py:169-215`), with active sessions untouchable. A request
resuming from a checkpoint cannot be vacated mid-resume PROVIDED it is passed
in `resumed_ids`/`active_ids`. That is a wiring obligation on the resume path,
not missing machinery.

## 7. Recommended order, revised

1. **Measure before building.** Boot an arm with
   `--enable-hierarchical-cache` (which already tiers mamba state to host) and
   re-run the #740 shape. If the hit rate lifts, #745 is a configuration and
   validation task and the writer never needed rebuilding.
2. Only then, if deterministic resume points are also wanted, lift the
   `server_args.py:14066` refusal by teaching `HiMambaRadixCache` the grid — a
   narrow change inside existing machinery, not a new subsystem.
3. Step (5) as a boot-cap change, with the token-ceiling acceptance from one
   log.
4. #743 (slot-count probe) remains the cheap comparison arm and is now directly
   interpretable: it varies exactly the quantity this note names as binding.

**Nothing here is boot-verified.** Every statement is source-level plus one
read-only `/get_server_info` probe.
