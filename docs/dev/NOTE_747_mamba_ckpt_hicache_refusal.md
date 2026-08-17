# NOTE 747: the refusal is not a safety guard — it prevents a LYING FLAG

**Determination: do not lift `server_args.py`'s
`--mamba-checkpoint-interval` x `--enable-hierarchical-cache` refusal yet.**
It is neither a #718-class corruption guard nor a merely-never-composed
leftover. It stops a combination in which one of the two flags would silently
do NOTHING. Deleting it ships a flag that lies; the lift is a build.

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

* Default cadence **8192** tokens, a multiple of `chunked_prefill_size=512`
  (16 chunks), per the user's <=8k tail-re-prefill budget. Unset stays byte-
  identical.
* Interval anchors become host-tier-eligible like any other retained node — no
  separate pin class, so the existing writeback/eviction accounting stays one
  ledger.
* Red-first: a hermetic construction smoke of the combination, plus an
  interval-anchor-reaches-host-tier test, can-fail both ways.
* The refusal is deleted only in the same commit that makes the grid effective;
  until then it stays as the honest answer.

## 6. Status

Not built. #742, the contract-honesty half of this task, IS built and shipped
on this branch. Everything above is source-level; nothing boot-verified.
