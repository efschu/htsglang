# HANDOFF 674 — #656 / #631 Route A, successor 30 (second half)

Read HANDOFF_673 first (same shift, first half): the corridor gate's wiring,
its metal proof, and the closure of two ordered rungs that were worth nothing.
This file covers what happened after that and what the next shift should do.

---

## 1. ERRORS FIRST

### 1a. kvso cannot fund the guard — the THIRD "frees nothing" catch

Spec item 12 names kvso as the KV-to-host spill class, and the obvious reading
is that it becomes the guard's `RELIEF_HOST` provider. **It cannot.**
`managers/kv_session_offload.py` frees device slots into the ALLOCATOR FREE
LIST and never touches physical backing:

    :3754  self.allocator.free(over.to(torch.int64))
    :3856  self.allocator.free(seg.to(torch.int64))
    grep "empty_cache|release_backing|decommit|cuMem" -> NO MATCHES

NVML free does not move, the guard re-probes and counts 0, and the provider
would be **inert while looking exactly like a working one**.

That is the third time this chain has priced a payload that frees nothing the
driver can see: the 1925 MiB drafter estimate (HANDOFF_672), idle mamba/GDN
slots (HANDOFF_673 §1b), and now kvso. **The pattern is always an internal
bookkeeping free mistaken for a physical release.** Before registering ANY
provider, answer one question: does this call reach `cuMemUnmap`/`cuMemRelease`,
or does it push an index onto a free list? The existing pin is
`test_corridor_guard_631.py:183
test_a_provider_that_frees_to_torchs_cache_does_not_count`.

### 1b. My fill load starved the real agents and made serving look crashed

`s26_fill_load.py` takes exactly 4 streams and `max_running_requests=4`, so my
own synthetic top-up saturated the cap and every real qwen request queued
behind it. From outside this is indistinguishable from a hung server, and it
was reported as a crash. It was not: health was 200 throughout and no real
request errored.

**Fill load may only top occupancy up BELOW the cap; it must never saturate
it.** When real workers run, drop the synthetic streams.

The single 503 in the log is unrelated and benign: a `/health` probe at
22:56:18 landing mid-cutover during the first boot flip — my own readiness
poll, not an agent request.

### 1c. I mis-diagnosed the real traffic as absent

I grepped `/v1/chat/completions` and found one hit, and concluded the qwen
agents were not reaching the rig. They were. The router drives
**`/v1/messages?beta=true`** and **`/v1/completions`**. Count those, not the
OpenAI chat path.

---

## 2. WHAT SHIPPED THIS HALF

* **The `pp->tp` deadlock escape.** `ensure_headroom(..., refusal_is_fatal=)`,
  passed by the `pp_to_tp` leg only. Item 16 withholds host RAM while any card
  has headroom; item 15c says the host tier's price is tempo, never a corridor
  breach. They collide exactly in this rig's normal state (spread 2895-4013
  MiB). Resolved by what a refusal COSTS per leg: refusing `tp->pp` is
  survivable, refusing `pp->tp` starves decode outright under strict purity.
  The escape OPENS the host tier; it does not reorder the ladder — rebalance
  and park are still spent first. `host_forced_count` counts every use and the
  warning prints the whole free column, so each one reads as the price of the
  missing rebalance tier rather than a licence.
* **Rung 3, the weights-arena tail** (HANDOFF_673 §4 built): 1180/300/210 MiB
  released at boot, 57 release/restore cycles under load, 0 abandons, corridor
  held, min free +316/+88/+352 MiB at matched occupancy.
* Suite 845 passed / 1 failed (the inherited `_staging_bytes` red).

---

## 3. THE REAL KV LEVER, AND EXACTLY WHY IT IS NOT WIRED YET

The production KV pool **is** a `KvVmmArena` under #631 —
`model_runner_kv_cache_mixin.py:3925` passes `swappable_backing=True` when
`enable_phase_flip`, reaching `memory_pool.py:2439 KvVmmBufferOwner`, whose
`decommit_range` is `cuMemUnmap` + `cuMemRelease`. Driver-returning primitives:

    memory_pool.py:2588  release_backing_span(layers, lo_row, hi_row) -> bytes
    memory_pool.py:3705  runtime_set_backing_rows(rows) -> bytes released

**The blocker is the watermark, not the release.** `shrink()` requires "rows
above the new span must be dead", and nothing in the tree computes a safe
shrink point from the live set. The #330 vram dial does not compute one — it
DESTROYS the live set instead (`vram_dial.py:957`: `tree_cache.reset()`,
`req_to_token_pool.clear()`, then `allocator.resize`). And `vram_dial.py:1053`
refuses `--enable-vram-dial` outright when kvso is on.

A correct KV provider must, in the scheduler thread:

1. verify `pool.supports_backing_spans` and that handle retention is OFF
   (`SGLANG_FLIP_SEAM_RETAIN_HANDLES`) — retained handles are memory the
   driver still owns, so the honest number collapses if it is set;
2. pick a watermark ABOVE the highest LIVE row (`kv_reshard.py:879
   _live_slots()` via `tree_cache.all_values_flatten()`; the flip already has
   `build_flip_live_slots_fn`);
3. evict only the CACHED entries above it;
4. **cap the allocator so nothing is handed out above the watermark** — skip
   this and the next allocation touches unbacked memory, which is a FAULT, not
   an error;
5. then release, and return the bytes unmodified.

Step 4 is the one that makes this a real build rather than an adapter.

**The REBALANCE actuator already exists and is the cheaper win.**
`managers/kv_reshard.py` (#297) moves KV rows between DCP ranks at runtime —
`KvReshardRuntime.arm(target)` -> `on_round` -> `_execute`, NCCL
`batch_isend_irecv`, reads-before-writes, checksum-verified, proven on
`7,3,3 -> 2,11,10`. `distributed/corridor_vector.py` only SOLVES a vector
(`solve_corridor_vector`); this is the mover. Wiring reshard as the guard's
`RELIEF_REBALANCE` provider is the item-16 levelling path and needs no new
transport — and levelling is what the measurements have been screaming for all
shift (5090 never below ~3.8 GiB free while both 3080s bind at 1.7-2.3 GiB).

**Do this before the host-tier provider.** It is less code, it is the tier the
ladder reaches first, and every `host_forced_count` it prevents is a real spill
avoided.

---

## 4. REAL-LOAD ACCEPTANCE: THE STRUCTURAL PROBLEM

Spec item 14 requires real agent traffic as the acceptance carrier. Measured
this shift: two qwen audit agents produced **37938 live slots, 7.6% occupancy**
against a 500000 pool.

`max_running_requests=4` caps concurrency at 4. So ">=80% occupancy on REAL
traffic" requires roughly **4 concurrent agents each holding a very long
context** — not four short ones, and not two. An agent doing a repo audit
holds tens of thousands of tokens, not hundreds of thousands.

The next shift should size the acceptance load deliberately: 4 concurrent
long-context agent tasks (large-file reading, long transcripts), and book
queue-wait/TTFT per worker so "slow because queued" is never again mistaken
for "crashed". Synthetic fill may top up only BELOW the cap.

---

## 5. NEXT, IN ORDER

1. **Wire `kv_reshard` as the `RELIEF_REBALANCE` provider** (§3). Item 16's
   levelling path, already built, biggest measured gap.
2. **The KV backing-release provider** (§3 steps 1-5), `RELIEF_HOST`, which is
   what finally funds the `pp->tp` seam. The escape from §2 is already waiting
   for it.
3. **PP-levelling A/B, GDN-only cut** — and note the correction: HANDOFF_673
   §1c said "do not build". That verdict was right about the *grid* and wrong
   about the *GDN-only* lever. The ~308 MiB/layer corridor cost applies to
   moving a FULL-ATTENTION layer, which carries KV. With 2-layer granularity a
   cut can land mid-group and move 2 GDN layers — compute, ~no KV — onto the
   5090's idle prefill capacity with full-attn placement unchanged. A hand-set
   `--pp-layer-ratio` is acceptable for the A/B if the runtime is honest about
   the bypass. Book prefill tok/s, per-rank compute/wait ms, corridor.
4. Resident-working-set capacity table per (phase, load): bs1/bs4 x PP/TP, one
   corridor CSV each at real occupancy, conservation check against 669440.
5. **CHUNK A/B -- the FIRST item-8 arm** (user pushed it ahead of DFLASH,
   PP-prefill-graphs and the 5090-imbalance arm; asked about dynamic prefill
   batching three times). VERIFIED PREMISE: the ship config runs
   `chunked_prefill_size=2048` -- the bare default, not even present in argv
   (`/tmp/s30_argv_new.txt` has no `--chunked-prefill-size`) -- and
   `--enable-dynamic-chunking` is absent. The flag IS built
   (`server_args.py:1168`) and is first-chunk-fixed, but has never been
   activated in the ship config.

   Protocol: same-boot floor first; static ladder 2048 / 4096 / 8192 / 16384;
   then the dynamic arm WITH ENGAGEMENT PROOF -- log evidence that the chunk
   size actually moved at runtime, including on the FIRST chunk (the fixed
   path), not merely that the flag parsed. Book prefill tok/s, TTFT and
   corridor per arm, under real qwen task load (fill only below the cap).

   Note `server_args.py:14243`: `if self.enable_dynamic_chunking and
   self.pp_size > 1 and chunked:` -- there is a PP-specific interaction, and
   this instance IS pp_size>1 in the prefill phase. Read that branch before
   trusting an arm.

   DECISION RULE, fixed in advance: if the dynamic arm wins or TIES the best
   static rung with no corridor cost, ACTIVATE it in the ship config and book
   the flip in the bench and the start recipe. If it loses, book the numbers
   and set the best static rung explicitly. Either way the ship config stops
   running an unmeasured 2048 default.

6. Remaining item-8 arms (DFLASH x graphs, PP-prefill-graphs,
   5090 stage-imbalance), threshold-purity arm, final all-axes acceptance
   under real router traffic.

## 6. PROCESS

* **Cap every waiting Bash call at <=60 s.** A long poll loop blocks
  coordinator message delivery for its whole duration; chain short calls.
* Label the LOAD MIX in every corridor CSV (real vs synthetic, and why).
* The can-fail instrument (`SGLANG_CORRIDOR_FLOOR_MIB`) found the wedge in
  HANDOFF_673 §1a. Every new provider should get the same treatment: raise the
  arming floor until the gate fires on real metal, then read what it did.
