# NOTE 747: the refusal is not a safety guard — it prevents a LYING FLAG

**Determination (step 1): do not lift `server_args.py`'s
`--mamba-checkpoint-interval` x `--enable-hierarchical-cache` refusal yet.**
It is neither a #718-class corruption guard nor a merely-never-composed
leftover. It stops a combination in which one of the two flags would silently
do NOTHING. Deleting it ships a flag that lies; the lift is a build.

**Outcome (step 2): the build is done and the refusal is LIFTED** — all five
seams below are mirrored into the unified mamba component through shared
rules in `mamba_ckpt_utils.py`, and the lift commit removes both refusals
(hierarchical cache AND the unified-radix-tree env, which guarded the same
component). Per-seam status in §7.

## 1. What the refusal says, and where it came from

```
server_args.py:13884-13889  (_handle_mamba_checkpoint_interval)
    if self.enable_hierarchical_cache:
        raise ValueError(
            "--mamba-checkpoint-interval does not support "
            "--enable-hierarchical-cache yet (HiMambaRadixCache has its "
            "own match/evict paths)."
        )
```

`git blame` -> `05933d03694` (2026-07-13), the flag's OWN introducing commit.
Its body lists the guard among ordinary validations — "multiple of page size
and of the mamba/FLA chunk size, not above the chunked prefill size, radix
cache required, hierarchical/unified radix tree rejected" — with no measured
failure behind it. It was written defensively at introduction, not in response
to a bug.

## 2. The stated reason is aimed at a class that never runs

The message blames `HiMambaRadixCache`. Per NOTE_745 §6a that class has **no
construction site**: `registry.py:107-112` routes a hybrid-SSM model under
`--enable-hierarchical-cache` to `_create_unified_radix_cache`, and the comment
there says so outright. So this arm's rationale describes machinery that is
never built — the same aiming error #609 exists to catch, and the second time
it has bitten in this area.

The class that DOES run is `UnifiedRadixCache` plus its mamba component.

## 3. The real conflict: the unified component does not know the grid

```
grep -n 'checkpoint_interval|is_on_interval|mamba_track_interval' \
    unified_cache_components/mamba_component.py unified_radix_cache.py
-> no matches
```

`MambaRadixCache` consults the grid in six places
(`mamba_radix_cache.py:626-633, 654, 796-802, 1101, 1513-1514, 1534`). The
unified component consults it nowhere. So composing the two flags today would
not corrupt anything — it would leave `--mamba-checkpoint-interval` **silently
ignored**, while its help promises deterministic absolute-multiple checkpoint
positions.

That is the #742 defect class exactly: a flag that does nothing while claiming
to. Lifting the refusal without implementing the grid would REPLACE an honest
refusal with a dishonest acceptance, which is strictly worse.

Precedent for the method: #547 -> #550, where a refusal that merely described
two features was read against the tree to decide impossibility vs unbuilt. This
one is unbuilt.

## 4. What the lift actually costs

Mirror the grid at the unified component's five seams:

| seam | file:line | mirrors |
| --- | --- | --- |
| match gating | `mamba_component.py:64` `create_match_validator`, `:77` `finalize_match_result` | `mamba_radix_cache.py:1513-1534` |
| retention gating | `:149` `commit_insert_component_data` | `:654`, `:796-802` |
| cache_len choice | `:394` `prepare_for_caching_req` | `:626-633` |
| split behaviour | `:180` `redistribute_on_node_split` | `_split_node` (`:1699-1704`) |
| eviction protection | `:190` `evict_component`, `:227` `drive_eviction` | `:1092-1105` |

### 4.1 The eviction seam is the subtle one, and it gets EASIER here

With the interval set, `MambaRadixCache` evicts in two passes and spares the
deepest anchors of every path:

> "losing the deepest one silently moves the resume point of identical requests
> and re-introduces run-to-run drift" (`mamba_radix_cache.py:1092-1097`)

So the interval carries a DETERMINISM contract, not just a position grid. That
protection exists because, on a device-only pool, a spilled anchor is a DEAD
anchor.

Under the unified tree that premise weakens in the user's favour: an evicted
anchor with a host backup is still a valid match and triggers `load_back`
(`mamba_component.py:71-74`, `:139-144`). An anchor that survives on host/disk
does not move the resume point. The protection window can therefore be weaker
here than in `MambaRadixCache`, not stronger — which is precisely the outcome
the directive is after (8k deterministic anchors that survive on disk).

## 5. Recommended shape, when it is built

* Cadence **8192** tokens per the user's <=8k tail-re-prefill budget.
  CORRECTION to this note's first draft (which called 8192 "a multiple of
  chunked_prefill_size=512"): the validation runs the OTHER way —
  `server_args.py` refuses an interval EXCEEDING `--chunked-prefill-size`,
  because prefill steps are clipped to checkpoint boundaries and a chunk
  budget below one grid unit would never reach one. Cadence 8192 therefore
  requires `--chunked-prefill-size >= 8192`. Unset stays byte-identical.
* Interval anchors become host-tier-eligible like any other retained node — no
  separate pin class, so the existing writeback/eviction accounting stays one
  ledger.
* Red-first: a hermetic construction smoke of the combination, plus an
  interval-anchor-reaches-host-tier test, can-fail both ways.
* The refusal is deleted only in the same commit that makes the grid effective;
  until then it stays as the honest answer.

## 6. Status of step 1

#742, the contract-honesty half of this task, was built and shipped first
(`da555ba0e7`). §§1-5 recorded the determination; §7 records the build.

## 7. The build (step 2): per-seam status

All decisions live in `mem_cache/mamba_ckpt_utils.py` and both lineages call
them; `test_mamba_anchor_seams_747.py` pins every rule in both directions
plus source-level parity (the modulo exists only in the shared helper; the
walks CALL the rules).

| seam | shared rule | status |
| --- | --- | --- |
| match gating | `is_resume_candidate(depth, interval, has_device_value, has_host_value, device_only)` — state present AND on-grid; called by `MambaRadixCache._match_prefix_helper` and by `MambaComponent.create_match_validator` (both variants). The unified walk now accumulates `cum_tokens` (absolute matched key depth, evicted-but-backuped nodes included) and hands it to every component validator. Host-backed anchors are gated identically: storage written by a non-interval run cannot re-introduce off-grid resume points. | BUILT (seams 3-5 commit) |
| retention gating | `is_on_interval` in `prepare_for_caching_req` (no_buffer arms, finished after the ReplaySSM cursor reset + unfinished) — off-grid end caches NOTHING, never floors. Backstop in `commit_insert_component_data`: an off-grid leaf commit (`len(params.key)` off the grid) is refused with `mamba_exist=True`, covering InsertParams producers that bypass prepare (session restore) and peer components shrinking `effective_cache_len` below mamba's on-grid choice. | BUILT (seams 3-5 commit) |
| cache_len choice | `is_on_interval` in `prepare_for_caching_req` (extra_buffer arm) — an off-grid tracked position warns and caches nothing, mirroring `:626-640`. | BUILT (seams 3-5 commit) |
| split behaviour | Already-parity: `redistribute_on_node_split` tombstones the new parent's mamba entry exactly as `_split_node` does (`new_node.mamba_value = None`); a split node is never an anchor, so no grid arithmetic arises. The branching-grid half (`branch_grid = interval or chunk_size`) is mirrored in `finalize_match_result`. | BUILT (seams 1-2 commit) |
| eviction protection | `protect_deepest_anchors(interval, host_tier_present)` — the explicit branch: device-only pool protects the deepest anchors (an evicted anchor is a lost anchor, today's behaviour, byte-identical); host tier present relaxes it (an evicted anchor stays a valid match and loads back). Both directions pinned. | BUILT (seams 1-2 commit) |

Additionally mirrored so no knob goes silently inert (the #742 class):
`SGLANG_MAMBA_CKPT_STRICT_RESUME` in `finalize_match_result`, where the chunk
sums are exact token depths (device-only unified tree). Under a host tier the
flag's premise does not arise: device-LRU churn cannot move the resume point
because evicted anchors stay matchable.

The lift commit removes both `server_args.py` refusals (hierarchical cache +
unified-radix-tree env — both guarded this same component), extends the flag
help with the composition and the 8192 cadence (>= 8192 chunked prefill), and
flips `test_hierarchical_cache_rejected` into two composition pins (accepts,
and still validates the grid rules).

## 8. Not desk-provable (boot-gated acceptances, for the window list)

Everything above is hermetic-unit-level (CPU, `CUDA_VISIBLE_DEVICES=""`).
The following need a GPU boot and ride a later WINDOW_TICKET_745 arm (ARM-1
deliberately boots hierarchical WITHOUT the interval):

1. A serving boot with `--enable-hierarchical-cache
   --mamba-checkpoint-interval` (chunked prefill >= interval) reaches ready
   and serves.
2. An interval anchor demonstrably survives to the host tier/disk and a
   subsequent identical request resumes from it (load_back observed at an
   interval-multiple position, `mamba-ckpt` debug attribution on).
3. Determinism across churn: identical requests resume at the same anchor
   after device eviction of that anchor (the weaker host-tier eviction
   branch behaving as designed under real pressure).
4. The co-location of the int8 checkpoint pool with the host tier stays
   refused (`enable_int8_mamba_checkpoint` x hierarchical is a separate,
   still-standing refusal — untouched by #747).
