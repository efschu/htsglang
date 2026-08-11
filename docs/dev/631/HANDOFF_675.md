# HANDOFF 675 — #656 / #631 Route A, successor 31

Read `HANDOFF_674` §0 first (the C7 law and the two open register items), then
this. `CONTRADICTIONS_REGISTER.md` still governs every number quoted here.

This shift built the two rungs the order asked for and then spent most of its
metal time being taught, twice, that they were wrong. Both lessons are worth
more than the code.

---

## 0. THE ONE-LINE STATE

The KV rung's DEVICE HALF IS PROVEN ON METAL and is currently OFF by default.
It works (`free 4 -> 1844 MiB, reclaimed 1840 MiB from [kv-backing]`) and it
wedged the group, for a reason that is a design fact rather than a bug in the
bytes: **a rank-local cap cannot drive a collective admission decision.** Fix
that (§1a) and the rung ships.

The `allocator-cache` rung is ON, unconditional, and proven paying.

---

## 1. ERRORS FIRST

### 1a. THE CAP DESYNCED THE PP GROUP — read this before touching the rung

With an 8 MiB chunk the device half worked, three flips completed with it
live, and then the scheduler stopped heartbeating; `/health` reported
"couldn't get a response from detokenizer" while every rank was alive and
logging normally.

The ranks had capped to **449039 / 451037 / 175225 / 145734 rows**. The cap
changes `available_size()`, which feeds ADMISSION, and each rank sized its own
shrink from its own free memory and its own live set. So the group no longer
agreed on how much work it could take, and a PP group that disagrees about
admission desyncs.

**This is the same law the corridor guard already states for its fleet probe**
— a rank-local decision inside a collective waits forever on the rank that
branched differently. The guard is RIGHT to read NVML per-rank for a REFUSAL,
which is unanimous by construction. It is WRONG for a CAPACITY, which every
rank must agree on.

    A refusal may be decided locally. A capacity may not.

THE FIX, and it is not a retreat: the shrink target must be a **collective
minimum**, the way the seam's abandon already rides `_collective_min`, so
every rank caps to the same row count. The gate runs at the same point in the
flip protocol on every rank, which is exactly where that collective is safe.
Until then the rung is behind `SGLANG_KV_BACKING_RELIEF=1`.

### 1b. THE RELIEF PROVIDER CONSUMED 2.5 GiB INSTEAD OF FREEING IT

Booted at a raised arming floor (`SGLANG_CORRIDOR_FLOOR_MIB=4000`) as the
can-fail instrument. It fired inside one boot, and the gate's own detail line
is the whole story:

    CORRIDOR-GUARD REFUSED on device 0: want 488 MiB, free 3040 -> 460 MiB,
    reclaimed 428 MiB from [allocator-cache, draft-weights]

**Free fell by 2.5 GiB during a reclaim.** It ended in
`cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`.

Root cause: **the ship config's KV arena has NO COMMIT CHUNK.** Without one it
holds one extent per buffer, and `decommit_range` releases only extents lying
WHOLLY above the keep point — so a shrink to any watermark inside that extent
releases exactly ZERO while still lowering `pool.size`. The measured driver
delta correctly reported 0. What was wrong was the RESPONSE to it: the failure
path called `recover()`, and recovery GROWS the pool (`finalize` ->
`cuMemCreate`). **The cleanup allocated, inside a gate that had armed because
memory was short, on every single arm.**

Three fixes, each pinned by a test:

1. a chunkless arena is a **registration disqualifier**, not a warning;
2. the failure path **never re-commits** — the cap stays on instead, because
   it is free and it is the invariant that nothing is handed out above the
   watermark. Recovery happens on the `tp->pp` leg, at an idle boundary;
3. a zero-byte shrink **exhausts** the provider. One failed attempt is
   evidence about the ARENA, not about this moment; repeating it is what
   turned a one-off into 2.5 GiB.

This is the FOURTH "frees nothing the driver can see" catch in this chain
(drafter estimate, idle mamba slots, kvso, now this) and **the first that
actively consumed memory**. The generalisation is worth carrying:

    A payload that cannot free is survivable. A payload that cannot free AND
    tries to undo its own attempt is not — undo is allocation, and it runs at
    the exact moment memory is scarcest.

### 1c. THE ALLOCATOR CAP READ AS A POOL LEAK AND KILLED ALL THREE RANKS

With `SGLANG_FLIP_SEAM_CHUNK_MIB=256` set, the rung shrank the pool and the
cap withheld 80165 slots — i.e. **it worked**. The first idle check then
killed every rank:

    ValueError: pool memory leak detected! [full] total=500000,
    available=419745, evictable=90, protected=0, session_held=0, uncached=0

`419745 + 90 + 80165 = 500000`. Nothing leaked. The invariant is
`available + evictable + protected + session_held + uncached == total`, and
capacity deliberately held out of circulation is in none of those buckets.
Fixed by a `withheld` term sourced from `allocator.residency_withheld_slots`,
which `KvRowCap` publishes on every mutation, in TOKENS (the paged allocator
multiplies its free list by `page_size`, so an id count is wrong by that
factor on every paged lane). A test pins that the term is not a licence: the
same shortfall with no cap engaged is still a leak.

**The lesson is not "add a term".** A new residency state has to be declared
to every ledger that sums the pool, and the #486 named-posten law already said
so. The next rung that removes slots — cached-evict, host-spill — inherits
this obligation.

### 1d. THE SHRINK STILL FREED NOTHING, BECAUSE RELEASE IS PER-BUFFER

With the chunk set and the invariant fixed, the rung ran clean and still
returned zero on every attempt:

    KV-BACKING shrink to 421738 rows reported 0 MiB but the driver's free
    column did not move

Release is extent-granular **per buffer**. The arena holds each of the
`2*layer_num` buffers at its own offset and `decommit_range` frees only
extents lying WHOLLY above the keep point, so an ask for N bytes moves only
`N/n_buffers` in each one. A 78262-row shrink asked about 40 MiB of each of
~28 buffers, cleared no extent in any of them, and returned nothing.

The provider now computes its own granularity — one commit chunk in EVERY
buffer, expressed in rows — and rounds the ask UP to it rather than attempting
a guaranteed no-op.

**THE NUMBER THAT DECIDES WHETHER THIS RUNG EXISTS AT ALL:**

    min_release_rows = commit_chunk_bytes * n_buffers / bytes_per_row

At the 256 MiB chunk that is `256 MiB * 28 / 15 KiB = 489132 rows` — **more
than the entire 500000-row pool**, so the rung could never pay a partial
release no matter how much slack the pool had. At 8 MiB it is ~15285 rows
(~230 MiB/rank), which is the right size against a gate asking ~500 MiB.
**Pick the chunk from this formula, not from intuition about page sizes.**

### 1e. A LATENT FAULT THIS RUNG MADE REACHABLE (fixed before it fired)

`zero_kv_data_buffers` zeroed the WHOLE VA-sized tensor. A KV buffer spans the
reservation; its backing does not. So `/flush_cache` against a pool whose
backing had shrunk writes into unmapped address space —
`cudaErrorIllegalAddress`, which kills every rank rather than raising. It was
unreachable while the only shrink path (#330 dial) destroyed the live set and
never ran under load. **The corridor procedure itself calls `/flush_cache`
before every idle reading**, so the two would have met on the acceptance run.
Zeroing now stops at `safe_zero_rows`.

---

## 2. WHAT SHIPPED

* **`RELIEF_LOCAL`, a tier below rebalance** (`corridor_guard.py`), for relief
  that moves no payload anywhere. Two providers live in it.
* **`allocator-cache`**: torch's unused cached blocks handed back to the
  driver. The seam already did this, but inside `_staging_affordable`, i.e.
  AFTER the gate had formed its verdict — so the gate judged against a free
  column understated by 1028-1426 MiB/card and could refuse a `pp->tp` flip
  while a gibibyte of nobody's memory sat on the card. **Metal-proven paying**:
  `reclaimed 166 MiB from [allocator-cache]`, `186 MiB` on the peer rank.
* **`kv-backing`** (`managers/kv_backing_relief.py`, spec item 12 device
  half): `KvRowCap` + `KvBackingRelief`. The cap is the piece that did not
  exist — `shrink()`'s precondition is "rows above the new span must be dead",
  and the only prior shrink path satisfied it by DESTROYING the live set. The
  cap withholds high ids from the FREE LIST instead: live allocations are
  never touched, `available_size()` falls out correct, and the scheduler
  simply admits less work. Three leaks are pinned as tests (eviction does not
  compact; `clear()` rebuilds `arange(1, size+1)`; a cap that bought nothing
  is not carried).
* **`safe_zero_rows`** on `MHATokenToKVPool` + the flush fix (§1c).
* **`withheld`** in the scheduler's pool invariant (§1b).
* `s25_acceptance_evidence.py` now books the **item-16 spread time series**
  and a **relief-ladder section** (per-provider spend counts, host-forced
  count, kv-backing shrink/recover counts).
* `s30_reboot_corridor_guard.sh` took its own session name as a hardcoded
  heartbeat exemption, so the next shift's first reboot refused against its
  own liveness proof. Now `$SELF`.

---

## 3. THE CONFIG FACT THE NEXT SHIFT MUST NOT LOSE

**The KV rung requires `SGLANG_FLIP_SEAM_CHUNK_MIB` to be set.** The ship
config does not set it, so its arena is chunkless and the rung refuses to
register (correctly — see §1a). Boots in this shift used `256`.

And the chunk must be SMALL — see §1c for the formula. 256 MiB makes the rung
structurally unable to pay; boots in this shift settled on 8 MiB.

This is not a tuning knob, it is a precondition: a chunkless arena cannot
release anything partially, which makes every row-range primitive in
`kv_vmm_backing.py` inert. It also gates the seam's own waved span release
(`phase_flip_runtime.py:1616` probes `supports_backing_spans` before using
`release_backing_span`), so setting it changes seam behaviour too — a
deliberate change, sanctioned by spec item 6, but one to state when quoting
any number across boots with and without it.

---

## 4. NEXT, IN ORDER

1. **Make the shrink target a COLLECTIVE MINIMUM and turn the rung back on**
   (§1a). Everything else about it is proven: it registers, it caps safely, it
   returns real driver bytes, it survives the idle invariant, and it rescued a
   card that was at 4 MiB free. The only missing property is agreement across
   ranks. Re-enable with `SGLANG_KV_BACKING_RELIEF=1` and
   `SGLANG_FLIP_SEAM_CHUNK_MIB=8`, and confirm RECOVERY on the `tp->pp` leg
   (the pool must return to its boot rows) — recovery is the property that
   keeps this from being "a smaller pool as the fix".
2. **`kv_reshard` as the `RELIEF_REBALANCE` provider** (item 16 levelling).
   Still the tier the ladder reaches BEFORE host, still already built
   (`KvReshardRuntime.arm(target)` -> `on_round` -> `_execute`), and still the
   only thing that touches the structural finding that the cards never fill
   evenly (spread 2895-4013 MiB across 5 windows / 3 boots).
3. **The HOST half of item 12**: evict cached prefix entries to lower
   `max_live` (data discarded, recomputable), then spill live sessions to
   kvso's pinned host pool (data moved, restorable). Both lower the watermark
   and then reuse the exact code path in §2 — they are separate providers at
   higher cost, not changes to `kv-backing`. NOTE the §1b obligation: each one
   must declare its slots to the pool invariant.
4. **CHUNK A/B** (user pushed it three times), then the **GDN-cut A/B** with
   the mandatory per-arm arena-tail re-measure (`net = prefill gain + tail
   delta`, register C1/C7).
5. **Final all-axes acceptance** under real router traffic.

---

## 5. PROCESS NOTES THAT EARNED THEIR PLACE

* **The can-fail instrument paid for itself twice in one shift.** Raising the
  arming floor until the gate fires on real metal found both §1a and §1b
  inside two boots. Neither would have appeared in a normal-floor run for
  hours, and §1a would have looked like a corridor mystery rather than a
  provider bug.
* **Measure the driver delta, never the payload's own number.** It is what
  made §1a legible: the provider reported 0 honestly, so the bug was visible
  as "reported 0 and then burned memory" rather than as a phantom success.
* A new test file is not run until it is added to
  `scripts/run_631_flip_family.sh`. Mine sat uncollected through one full
  suite run that reported the same 857 as before it existed.
