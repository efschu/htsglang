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

## 6a. #609 RECONCILIATION — the marking HOLDS, and my step-7 aim was WRONG

**Verdict: the 2026-08-06 UNREACHABLE marking on `HiMambaRadixCache` stands,
and it is STRONGER than "the switches are off".** It is not that the flag fails
to reach the class; it is that a hybrid-SSM model under
`--enable-hierarchical-cache` is routed somewhere else entirely:

```
mem_cache/registry.py:107-112
    # Hybrid SSM/SWA under hierarchical cache ALWAYS takes UnifiedRadixCache.
    # HiMambaRadixCache has no construction site anywhere (see its module docstring).
    if ctx.enable_hierarchical_cache:
        if ctx.is_hybrid_ssm or ctx.is_hybrid_swa:
            return _create_unified_radix_cache(ctx, server_args, params)
```

`HiMambaRadixCache` is referenced only by its own module, by comments, and by a
type annotation in `hybrid_cache/hybrid_pool_assembler.py:1537`. It has no
construction site. The single live construction is
`registry.py:133 return MambaRadixCache(params)`.

**This retracts section 7 item 1 as written.** I recommended booting
`--enable-hierarchical-cache` "because it already tiers mamba state to host",
citing `hi_mamba_radix_cache.py:373/:553`. Those lines are real but they are in
a class that never gets built — exactly the #581 aiming error the #609 marking
exists to prevent. The recommendation was right by accident and wrong by
citation.

**The mechanism survives, in the class that IS reachable.** The unified tree's
mamba component does the same job:

* host-backed nodes are VALID MATCHES:
  `unified_cache_components/mamba_component.py:71-74` — "HiCache: evicted +
  backuped (host_value present) is also a valid match";
* and they trigger load-back: `:139-144` — "if mamba was evicted from device
  but has host backup, ensure mamba_host_hit_length >= 1 so load_back is
  triggered";
* `_mamba_pool_host` is "set to host mamba pool when HiCache enabled" (`:62`).

So the boot arm is re-aimed, not abandoned: `--enable-hierarchical-cache` on
this hybrid-SSM model yields **UnifiedRadixCache + its mamba component**, which
is where the survival-beyond-device-slots actually lives.

## 6b. DISK REACH — the file backend is mamba-aware, and already knows #555

Mamba state is NOT host-RAM-terminal. `PoolName.MAMBA` is a first-class pool
across the storage layer (`mooncake_store.py:730`, `storage_hf3fs.py:596`), and
the plain FILE backend carries it by key:

```
hicache_storage.py:808-828
    def _is_shared_mamba_key(self, key: str) -> bool:
        """True when the GDN/mamba blob for this key is the canonical one.
        Gated on the window existing, because only then are the blob's bytes
        full-width in both axes (every layer, every head) instead of this
        rank's shard. Without it the blob stays per-rank and per-stage, exactly
        as it is today."""
```

`_get_component_key` appends `.{component_name}` generically and
`_sharded_path` writes `{stem}.bin` under the configured file path. So disk
writes for GDN state need no new transport.

What #555 ADDS here is already wired as `canonical_mamba_blob`: with the
canonical window present the blob is full-width across layers and heads and is
therefore shareable across ranks/stages; without it, each rank writes its own
shard — which is still correct, just not shared. On THIS boot (pp_size=3,
tp_size=1) each stage owns whole layers, so per-stage blobs are the natural
unit and the canonical mode is an optimisation, not a prerequisite.

**Scope, not built:** nothing in #745 needs a new disk path. If cross-stage
sharing is later wanted, it is the canonical-window gate above, not a writer.

## 6c. HOST-LEDGER for the boot arm

GDN state per layer per slot, from the live checkpoint config
(`linear_num_value_heads=48`, `linear_key/value_head_dim=128`,
`linear_num_key_heads=16`, `linear_conv_kernel_dim=4`, `mamba_ssm_dtype=float32`):

```
temporal = 48 * 128 * 128                    = 786 432
conv     = (48*128 + 2*16*128) * (4-1)       =  30 720
                                    total    = 817 152 elements
fp32                                         = 3.1172 MiB / layer / slot
```

Live topology is **pp_size=3, tp_size=1** — PP, so each stage holds whole GDN
layers at full width (no head sharding). 64 layers, `full_attention_interval=4`
-> 48 GDN layers, split 17 / 16 / 15 across the three stages.

| stage | GDN layers | device pool (12 slots) | host post @ ratio 2.0 (24 slots) |
| --- | --- | --- | --- |
| 0 | 17 | 636 MiB | **1272 MiB** |
| 1 | 16 | 598 MiB | **1197 MiB** |
| 2 | 15 | 561 MiB | **1122 MiB** |
| **total** | 48 | 1796 MiB | **3591 MiB (3.51 GiB)** |

That 3.51 GiB is the HOST-LEDGER post the boot arm must be gated on
(`hicache_ratio = 2.0` is assumed to be the `host_to_device_ratio`
`MambaPoolHost.__init__` takes, `memory_pool_host.py:78-84`; if the arm sets an
explicit mamba host size instead, re-derive from the same per-slot figure).

**Step (5) sizing, for later:** dropping the resident floor 12 -> 6 returns
318 / 299 / 281 MiB per stage (~898 MiB total), which independently confirms
the ~0.9-1.2 GB estimate. Those MiB **raise the binding stage's token ceiling**
via `handle_max_mamba_cache` -> `total_rest_memory` (section 5).

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
