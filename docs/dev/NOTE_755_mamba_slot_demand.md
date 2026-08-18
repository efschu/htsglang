# NOTE 755: the 3-slots-per-request floor — what is demand, what is pin, what could share

**Question (user goal "no useless mamba states in VRAM"): under the #747
world (host-backed anchors), can the donation slot and the pinned-checkpoint
slot share or drop?**

**Determination: the floor is exact today, and the reduction 3 -> 2 per
request EXISTS as a mechanism — but it is a lock-lifetime reordering across
the cache-insert flow, gated on guaranteed host backup, and needs boot
proof. It is named below for the build that does it; it is not a desk-size
edit of the floor formula. Editing only the formula would under-floor the
pool and resurrect the #581 late assert the floor exists to prevent.**

## 1. The three terms, mapped (live no_buffer config, ping-pong = 0)

Formula: `mamba_pool_floor.py` (single source of truth; consumed by
`server_args._validate_max_mamba_cache_size` — the 05:55 ValueError — and by
the hicache write-through pin budget).

| term | site | lifecycle |
| --- | --- | --- |
| 1 active | `HybridReqToTokenPool.alloc` (`req.mamba_pool_idx`) | the state being computed; held for the whole run |
| 1 donation | `mamba_radix_cache.cache_unfinished_req` no_buffer arm: `_alloc_mamba_slot()` then `mamba_pool.copy_from(active, donated)` | transient at each cache boundary: the tree receives a COPY so the request keeps computing on its own slot; the fresh slot then BECOMES the new pinned checkpoint via `insert` |
| 1 pinned checkpoint | same flow, tail: `dec_lock_ref(req.last_node)` … `inc_lock_ref(new_last_node)` | the node the request resumes from stays lock-protected (unevictable) for the whole run |

The worst instant is the donation: the OLD pin is still held while the
donated slot is allocated and inserted — deliberately, per the
alloc-before-mutate invariant stated at the site ("Every allocation that can
fail is done BEFORE any request-visible mutation"). Releasing the old pin
first and then failing the alloc would leave the request with no resume
anchor on a device-only pool. Hence 1 + 1 + 1 concurrent.

## 2. Reducibility, term by term

* **active — irreducible.** It is the running state itself.
* **pinned checkpoint — reducible IN PRINCIPLE under a host tier.** The pin
  exists because on a device-only pool an evicted anchor is a DEAD anchor
  (the #747 eviction-seam premise, `protect_deepest_anchors`). With
  `--enable-hierarchical-cache` a host-backed anchor stays a valid match and
  loads back (`mamba_component.py` match validators + `load_back`), so
  device eviction of the resume point is a slowdown, not a loss. The pin
  then guards only the COW read window, not the whole run.
* **donation — the transient is real, but it OVERLAPS the pin.** The donated
  slot is the next pinned checkpoint; the double-count exists only because
  the old pin is held across the alloc. Reorder to
  `dec_lock_ref(old) -> alloc -> copy -> insert -> inc_lock_ref(new)` and
  the concurrent demand at the boundary is active + donated = 2: **the
  donation slot and the pinned-checkpoint slot share.** Floor per request
  3 -> 2; the specimen's 12 -> 8 for 4 running — exactly "4 reqs on 8
  slots".

## 3. Why this is not built here

The reorder is safe ONLY when the old anchor is host-backed at release time
(write-through completed, so a failed alloc or an eviction inside the
window degrades to load_back / re-prefill, never to a dead anchor). That
requires:

1. gating on hierarchical cache AND a confirmed backup (`host_value`
   present or write-through-pending accounting) — per node, not per boot;
2. the unified lineage's insert flow releases locks for ALL components at
   one point (`UnifiedRadixCache.cache_unfinished_req`:
   `dec_lock_ref(req.last_node)` after insert+match); an early mamba-only
   release needs component-asymmetric lock lifetimes — a lock-protocol
   change, not a formula change;
3. the retraction path exercised on metal: a request that retracts inside
   the unpinned window must demonstrably resume via host load_back
   (boot-gated; the desk cannot prove a race window empty).

A floor-formula-only edit (dropping the pinned term when hierarchical is
on) would claim slots the runtime still locks — the pool would admit a
concurrency it cannot serve, which is precisely the late-assert class
(#581) the floor was built to kill. Refused.

## 4. What the operator can do today, and what decides the rest

At `--max-mamba-cache-size 12 --max-running-requests 4`, every slot is
demand — the config carries ZERO evictable cache and therefore already
holds no "useless" states by the floor's own model. The states that FEEL
useless are the 4 pinned checkpoints; their uselessness is exactly the
host-backup question above. The measurable trade today is concurrency vs
KV: `WINDOW_TICKET_755_mamba_slot_ab.md` prices 12/4 against 6/2 on the
composite boot.
