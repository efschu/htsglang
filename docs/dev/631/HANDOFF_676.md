# HANDOFF 676 — #656 / #631 Route A, successor 32

Read `HANDOFF_675` §1a first (the desync this shift closed), then
`CONTRADICTIONS_REGISTER.md`. Every number here is valid for its geometry,
its pool and its residency state, and each one below says which.

---

## 0. THE ONE-LINE STATE

**The KV rung is ungated, on by default, and proven uniform on metal.** The
shrink target is now agreed by one MIN all-reduce instead of computed per
rank, and the pp->tp leg — the deadlock class — has a funder for the first
time: at the same arming floor that produced three corridor refusals before
the change, it produced **zero**.

The HOST half is NOT built, and the reason is structural rather than
unfinished work. See §2, which is the most important section for the next
shift.

---

## 1. ERRORS FIRST

### 1a. THE FIX FOR THE DESYNC, AND WHY THE OBVIOUS PLACE IS STILL WRONG

HANDOFF_675 §1a diagnosed it exactly: a rank-local cap cannot drive a
collective admission decision. The implemented shape is its second option,
sharpened by what the first boot taught.

The target is an element-wise MIN all-reduce over a four-field proposal every
rank makes:

    [ desire, -floor, current, -current ]

* **the AMBITION is a minimum** — the rank closest to the corridor floor sets
  it, because that is the rank the flip would otherwise be refused on;
* **the LIMIT is a maximum** — the target must clear the highest live row on
  EVERY rank, and it is an absolute row id, so a value below a peer's live
  set is `cudaErrorIllegalAddress`: a fault that kills every rank rather than
  raising;
* **the limit wins.**

WHERE it is reduced is the entire safety argument, and it is NOT inside the
guard's ladder. The guard's providers run behind its ARM CONDITION, which is
rank-local by construction (`free - want < floor` against this rank's own
NVML reading), so a reduction there hangs the first time one rank arms and a
peer does not. The reduction sits in `_corridor_gate`, **outside every early
return in that function including `guard is None`**, on the same
unconditional path as the seam's existing fit reduction, and BEFORE the guard
so the bytes it frees are money the guard's probe then sees.

**The failure mode is always "nobody caps", never "some cap".** A rank with
no relief object, no guard, no scheduler, or an unreadable live set proposes
`ABSTAIN`, whose `current=0` makes the group decline. Exhaustion is
deliberately NOT contagious: a rank whose arena cannot release partially
stops ASKING but still obeys the group target, because leaving it uncapped is
the exact admission disagreement this replaces.

The wiring tests assert this by COUNTING reductions on each early-return
path, not by reading the source. A collective reached by only some ranks is
the bug; a test that cannot see the count cannot see the bug.

### 1b. THE PROVIDER PRICED ITSELF AGAINST THE RESERVATION, NOT THE BACKING

First metal boot of the ungated rung, and it is the more instructive of the
two faults because every unit test passed through it.

    KV-BACKING runtime_set_backing_rows(379067) failed:
    cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY

A **shrink** called `cuMemCreate`. `pool.size` is the logical row count and it
does not move when the backing does — `initial_backing_rows` says in as many
words that it "does NOT touch self.size" — while the committed span lives in
`full_pool_backed_rows`. After the first shrink to 347161 rows, `size` still
read 500000, so the next target came out at 379067: ABOVE the committed span.
`runtime_set_backing_rows` converges the backing in BOTH directions, so that
was a grow — from inside relief, on the card being relieved.

    A row count is only a shrink relative to the number you measured it
    against. Measure it against the backing.

The fake pool in the tests now models the two-directional converge and
charges a card for a grow, so a caller that confuses the two fails there
instead of on metal.

### 1c. RECOVERY BREACHED THE CORRIDOR IT WAS SERVING

Same boot. `recover()` grew straight back to the boot rows with no reference
to free memory and drove rank 1 to **6 MiB free**, with a `cuMemCreate` OOM on
the way.

HANDOFF_675 placed recovery on the `tp->pp` leg because that is "an idle
boundary, where an allocation is affordable". Measured, it is not — and the
reason generalises: **the pool being re-committed is exactly as large as the
relief that was taken**, so the bigger the rung's success, the less
affordable its undo. Recovery is now bounded by this card's distance from the
corridor LAW (never the gate's proof-time arming floor, which would cripple
recovery for an instrument), and what it cannot re-commit stays capped with
the boot rows remembered for a later leg. An admission-capacity loss is
recoverable; a breach is not.

This is the FIFTH member of the "undo is allocation" family in this chain,
and the second that ran at the worst possible moment.

### 1d. THE FILL LEVER IS NOT THE ONE I PULLED

Booted the acceptance run with `--max-total-tokens 620000` to fill the cards
toward the corridor floor (spec item 15). The server clamped the pool to
**512552** at `_profile_available_bytes`, so the realised step over the
previous 500000 boot was **+12552 tokens** — nothing.

`--rank-gpu-memory-mib 31800,14000,15600` is what binds, not
`--max-total-tokens`. A successor who wants the cards filled must move the
per-rank budget and re-sample the corridor under load; raising the pool alone
is a no-op above the profile ceiling. **Do not read the acceptance run's fill
figures as evidence that the cards cannot be filled further** — they are
evidence that this lever does not reach.

---

## 2. THE HOST HALF IS BLOCKED, AND THE BLOCKER IS ONE LINE

The order asked for spilled KV pages to land in kvso's pinned host pool.
**kvso and the phase flip are mutually exclusive in this tree:**

    phase_flip_runtime.py:691, in flip_blocking_guards()
        if getattr(scheduler, "kv_session_offload", None) is not None:
            guards.append("kv-session-offload")

A feature in `flip_blocking_guards` REFUSES FLIP ARMING. So enabling kvso
turns the flip off, and the acceptance the user is waiting for needs the flip
on. HANDOFF_675 §4c mapped kvso's internals and concluded "the two halves
already compose correctly" — that audit is right about the DATA PATH and did
not reach the arming guard.

Three ways out, in the order a successor should consider them:

1. **The pinned pool without kvso's controller — CONFIRMED VIABLE, with the
   entry points.** The guard tests `scheduler.kv_session_offload`, not the
   pool class, and an audit this shift established that the pool is a pure
   data structure:

   * `MHATokenToKVPoolHost` (`mem_cache/pool_host/mha.py:88`, base
     `pool_host/base.py:105`), constructor `mha.py:91-103`, takes
     `(device_pool, host_to_device_ratio, host_size, page_size, layout,
     pin_memory=True, ...)`. **It takes no scheduler reference** — only the
     device-side `MHATokenToKVPool` (for dtype, dims, layer_num, head_num,
     head_dim) plus sizing.
   * D2H: `backup_from_device_all_layer(device_pool, host_indices,
     device_indices, io_backend)` — `mha.py:352`.
   * H2D, per layer: `load_to_device_per_layer(device_pool, host_indices,
     device_indices, layer_id, io_backend)` — `mha.py:235`.
   * Precedent for building it outside the manager already exists:
     `_kv_sess_attach_host_pool()`
     (`model_runner_kv_cache_mixin.py:2655-2808`) constructs it and hangs it
     on `model_runner.kv_sess_host_pool`; the kvso manager merely READS it
     later via the backend's `_sess_host_pool`. **The pool and the manager
     are independent.**

   So this route is open, it matches the brief exactly (kvso as destination
   only), and it needs no change to the flip guard. It is the one to take.
2. **Cached-prefix eviction first, which needs no host RAM at all.**
   HANDOFF_675 §4.3 lists it as the cheaper of the two host-half rungs: evict
   cached prefix entries (data discarded, recomputable) to lower `max_live`,
   which is precisely the term that limits the collective target's LIMIT
   field. It composes with the device half through exactly one number and
   costs no pinned memory on a host that is already tight.
3. **Lift the guard.** Largest and least attractive: a session's host image is
   layout-specific (PP KV is layer-bound per rank, TP KV is token-sharded), so
   a session spilled in one phase cannot simply be restored in the other.

**Whichever route: host RAM is already the tighter resource.** HANDOFF_675
measured peak 112.1 GiB of 120 with 9 cumulative cgroup `oom_kill`s. Size
against what is AVAILABLE, and book host RAM in the same ledger as VRAM from
the first rung rather than after the first oom.

---

## 3. WHAT SHIPPED

* **`collective_kv_target`** + `KvBackingRelief.propose()` /
  `apply_target()` (`kv_backing_relief.py`), and
  **`collective_kv_backing_relief`** (`phase_flip_spill.py`) as the driver.
* **`kv-backing` is no longer a corridor-guard provider.** The object is
  still cached on the scheduler, because recovery reaches it there. The
  absence of the registration is asserted by a test, so it cannot quietly
  come back.
* **One leg only.** The rung shrinks on `pp_to_tp` and abstains on `tp_to_pp`.
  The scheduler's pool IS the PP layout's pool: capping it entering TP gives
  up backing that will hold nothing for a whole phase, and `pp_to_tp` is the
  leg whose refusal is fatal. Entering PP the same pool goes live again and
  `recover_kv_backing` runs at that leg's post-cutover hook, so a shrink there
  is churn whose undo is a `cuMemCreate`. The abstention still ENTERS the
  reduction: the direction is agreed upstream, and "safe because an upstream
  value is agreed" is exactly the reasoning that produces hangs.
* **The default flipped to ON**, escape hatch retained facing the other way.
* **`SGLANG_FLIP_SEAM_CHUNK_MIB=8` exported by `route_a_631_prod_boot.sh`**,
  with the formula that picks it rather than an intuition about page sizes.
* **`ARGV_SET`** in `s30_reboot_corridor_guard.sh`, so a run can move one
  server argument and replay the rest of the live process verbatim.
* 21 new tests across `test_kv_backing_collective_631.py` (registered in
  `scripts/run_631_flip_family.sh`) and `test_phase_flip_corridor_gate_631.py`.

---

## 4. THE METAL EVIDENCE FOR THE RUNG

Boot 2 with both fixes, arming floor 4000 MiB as the can-fail instrument
(`/spinning/evidence-631/s32/serving-kvrung2.log`):

| property | before (HANDOFF_675) | this shift |
|---|---|---|
| shrink target across ranks | 449039 / 451037 / 175225 / 145734 | **347161 on all three** |
| driver bytes returned | 1840 MiB on one card | **2016 / 1440 / 1152 MiB** |
| corridor refusals at the same floor | 3 | **0** |
| `cuMemCreate` failures | 1 (from inside relief) | **0** |
| group state | wedged, `/health` detokenizer error | **health 200, 18 flip commits** |

The zero-refusal row is the second ticket this closes. Under strict purity a
refused `pp->tp` starves decode outright, and the only REBALANCE provider —
the drafter — is already spilled for the whole PP phase, so the fatal leg had
no funder at all.

---

## 5. THE ACCEPTANCE RUN

Extract: `/spinning/evidence-631/s32/accept/EXTRACT.txt`. Config, including
the exact argv and env deltas and the code commit the instance was built
from, is `CONFIG.txt` in the same directory. Log
`/spinning/evidence-631/s32/serving-accept.log`.

**WHICH LOG CARRIES WHICH AXIS, stated before the numbers**, because this
shift has two and they are not interchangeable:

* `serving-kvrung2.log` (boot 2, raised arming floor 4000 MiB) carries
  **spec item 12** — the KV rung shrinking, uniformly, for real driver bytes.
  It is the can-fail instrument and its corridor figures may NOT be read as
  acceptance.
* `serving-accept.log` (this run, the real 1024 MiB law) carries every other
  axis, plus the fact that the relief ladder is REACHED at the law and that
  its cheapest tier suffices at this fill level.

Quoting the second log for item 12 would be the "green on the axes it checks"
error HANDOFF_675 §4b warned about; quoting the first for the corridor would
be worse.

### The numbers

**66 minutes unmanned**, 08:54:35Z to 10:00:06Z, one instance, one log,
28692 corridor samples at 100 ms.

| axis | result |
|---|---|
| corridor | **0 breaches on all three cards.** MIN free 1139 / 2404 / 1623 MiB, p50 2111 / 4852 / 2531 |
| corridor, second half | headroom above the law **+115 / +1380 / +599 MiB** (s31: +948 / +2992 / +1160) |
| flips | **279 `pp_to_tp` + 276 `tp_to_pp`**, both layouts, **0 abandons, 0 tracebacks** |
| strict purity | **True** — 43995 prefill batches, **ZERO** carrying a graph |
| decode graphs | **99.0%** (909 of 918) |
| MTP | accept length **2.695** (n=918) |
| occupancy | live slots max **238607 = 46.6%** of a 512552-row pool |
| real traffic | 101 `/v1/completions`, 58 `/v1/messages`, 58 `count_tokens`, 1 `/v1/chat/completions` |
| relief ladder | gate **armed 89 times, 0 refused, 0 host-forced**; `allocator-cache` paid **178 times** |
| KV rung | **0 shrinks** — see below |
| item 16 spread | mean **2640 MiB**, median 2759, best 247, worst 3237 |
| host RAM | peak 112.1 GiB, `oom_kill` 9 **cumulative and unchanged from s31's reading, so zero new kills this run** |
| script verdict | `ACCEPTANCE: GREEN` |

### WHAT THE GREEN DOES AND DOES NOT COVER

It covers more than s31's did, and the difference is worth naming precisely.

**The gate now ARMS at the real law floor.** s31's run never armed once, so
its ladder was unreached — a state that proves the ladder does not break, not
that it works. Here it armed 89 times under real load and cleared every one:
**0 refusals in 555 flips**, with 178 provider spends. Item 15a is exercised
at the law, not only at a proof floor.

**gpu0 held within 115 MiB of the corridor law for 66 minutes.** That is the
"filled to the limit" regime spec item 15 asks for, and it never breached.

Three things it still does not cover, stated as plainly as HANDOFF_675 stated
its own gaps:

1. **Item 12's KV rung did not fire in this run.** Zero shrinks. Every arm
   was satisfied by the tier below it — `allocator-cache` returning 604-824
   MiB — which is the tier law working correctly and is also why the rung was
   never needed. Item 12 is proven in `serving-kvrung2.log`, at a raised
   arming floor, and **must be quoted from there**. See §6 next-step 0.
2. **Item 16 is still NOT satisfied.** Spread mean 2640 MiB, essentially
   unchanged from s31's 2917. Nothing this shift touched levelling, and §6
   item 2 now explains why the obvious actuator cannot be used at the seam.
3. **YaRN long-context (spec item 4) is not in this run at all.** The
   checkpoint is the `yarn1.5` build, but no leg drove context past 262144,
   so the "bs=1 above the standard context" acceptance component remains
   unmeasured. No successor should read this GREEN as covering it.

---

## 6. NEXT, IN ORDER

0. **Decide whether the KV rung should ever outrank the allocator cache.**
   Measured in the acceptance run at the real 1024 MiB floor: the gate armed
   repeatedly and `allocator-cache` covered the deficit EVERY time, returning
   604-824 MiB per arm. So the KV rung proposed nothing and item 12 went
   unexercised — correct tier behaviour (free money before KV capacity), and
   also the reason the rung may almost never fire in production.

   The mechanism is the `cheap_relief_bytes` discount in
   `KvBackingRelief.propose`: the rung subtracts what the cheaper tier could
   return before sizing its own ask. That estimate is
   `memory_reserved - memory_allocated`, which OVERSTATES what `empty_cache`
   actually hands back, and the overstatement is deliberate (under-shrinking
   is retried, over-shrinking costs capacity for nothing).

   The failure mode to watch: the gate can refuse AFTER the rung declined on
   the strength of a cache that then under-delivers. The rung runs before the
   gate, so there is no second chance inside one gate call. It is
   self-correcting across ROUNDS — the abandon is free, the cache is spent by
   then, and the next proposal sees a smaller discount — but nobody has
   watched that sequence happen. Watch for a refusal immediately following an
   `allocator-cache` reclaim that fell short of its hoard.

1. **The host half by route 1 or 2 of §2** — and read §2 before planning it,
   because the obvious route is closed by a line no previous handoff names.
2. **`kv_reshard` as the `RELIEF_REBALANCE` provider** (item 16 levelling) —
   **but not as a guard provider, and that is a design fact, not a detail.**
   An audit this shift (`kv_reshard.py`) found `arm(vec, source)` may be
   called at any time, while `_execute` is reached from `on_round` only when
   `ready_fn()` holds — wired at `:906` to `scheduler.is_fully_idle()`.

   A corridor-guard provider must free bytes SYNCHRONOUSLY and return what it
   freed, at a gate that runs inside a flip. An actuator that acts later, at
   the next fully-idle round, returns 0 there. Registering it would fund the
   ladder on paper and pay nothing — the exact failure mode this chain has
   shipped seven times and now asserts against.

   **And the seam is not idle**, which a second audit settled rather than
   assumed. `is_fully_idle()` (`scheduler.py:6211`) requires an empty running
   batch, an empty waiting queue and drained PP microbatches; the flip's
   quiescence predicate deliberately requires none of those — the source says
   so in as many words at `phase_flip_runtime.py:303` ("NOT #297 fully-idle")
   and `:307` ("deliberately does NOT require an empty waiting queue or an
   empty running batch"), and `:392` rejects `_pp_microbatches_drained`
   precisely because it would demand the resident decode set be empty.

   So `_execute` cannot run at the seam, and under continuous load a fully
   idle round may not arrive at all. The rebalance tier wants ONE of: an
   `arm()` at the gate whose effect the NEXT gate sees (honest, delayed, and
   booked as such), or a synchronous levelling primitive that does not go
   through the reshard runtime. Decide which before writing code — and note
   that the delayed option cannot report bytes to the guard, so it must not
   be registered as a provider either.

   Other requirements it carries: `tree_cache.all_values_flatten()`,
   `tp_cpu_group`, `tp_group.device_group`, a `HybridLinearKVPool` from
   `token_to_kv_pool_allocator.get_kvcache()`, and an empty
   `blocking_guards` — the same list that closes the kvso route in §2.
3. **Fill the cards for real.** §1d explains why the pool figure is not the
   lever; an audit of `_profile_available_bytes`
   (`model_runner_kv_cache_mixin.py:608`, clamped at `:4406` via
   `_apply_token_constraints`) names the ones that are:

   * **`--mem-fraction-static`** — the fraction of free memory held back as
     non-static slack. Lowering it raises the KV budget directly, and it is
     the lever nobody in this chain has touched.
   * **`--rank-gpu-memory-mib`** — the absolute per-rank budget, currently
     `31800,14000,15600` against 32607/20480/20480 MiB cards.

   And the confirmation of C15 in the source: **`--max-total-tokens` can only
   LOWER.** Raising it above the profiled value has no effect and emits a
   warning (`:4423-4427`) — which is exactly what happened when 620000 came
   back as 512552, and it means a successor can stop looking there.

   Until the cards sit near the floor the corridor's second half stays unmet,
   the spread stays around 2600 MiB, and the relief ladder is rarely reached
   past its cheapest tier, whatever the deeper rungs are capable of.
4. **The DYNAMIC chunking arm**, still unmeasured — audited this shift, so
   the next attempt can start from facts instead of `server_args.py:14243`.

   * **It would activate here.** The gate is
     `enable_dynamic_chunking and pp_size > 1 and chunked`
     (`server_args.py:14243`), re-asserted at `scheduler.py:1531-1532`. This
     instance is `pp_size=3`, so the feature is live the moment the flag is.
   * **It moves in BOTH directions.** Down to `base_chunk_size // 4` — 128
     from the shipped 512 (`scheduler_pp_mixin.py:2717`) — and up to
     `ceil(chunked * 1.25)` via the raised prefill ceiling
     (`server_args.py:14244-14246`). Since 512 is an interior optimum and
     16384 was a fatal OOM, the DOWNWARD half is the value hypothesis and the
     upward half is the risk.
   * **Recomputed once per scheduling iteration**: `scheduler.py:5108` calls
     `dynamic_chunked_prefill_size()` (`:4891`), which delegates to
     `predict_next_chunk_size` (`scheduler_pp_mixin.py:1743` -> `:2656`),
     quadratic solver at `:2688-2739`, smoothed at `:2713`.

   **THE TRAP, and it decides whether the A/B is falsifiable at all:** the
   ONLY line that reports a predicted chunk size is **DEBUG-level**
   (`scheduler_pp_mixin.py:1770-1773`). Nothing at INFO or above fires when
   the chunk deviates from `--chunked-prefill-size`. So the order's
   "engagement proof — chunk size provably moved at runtime from chunk 1" is
   **unobtainable at the default log level**. Raise the logger for that
   module before the run, or add an INFO line; otherwise the arm produces a
   throughput number with no evidence the mechanism ever engaged, which is
   the shape of finding this chain has retracted seven times.
5. Then the **GDN-cut A/B**, with the mandatory per-arm arena-tail re-measure
   (C1: the tail is a function of the split).

---

## 6b. THE LEDGER OBLIGATION, NOW WITH THE EXACT LIST

HANDOFF_675 §1c added a `withheld` term to the pool invariant after a cap
read as a leak and killed all three ranks, and said the next rung that
removes slots inherits the obligation. An audit this shift produced the
precise inheritance, in `mem_cache/invariant_checker.py`:

    available + evictable + protected + session_held + uncached + withheld
        == total                                   (:112, shared function)

| pool | terms it actually passes |
|---|---|
| full attention (`:173`) | all six — the ONLY one carrying `withheld` (`:184`, from `allocator.residency_withheld_slots`) |
| SWA (`:199`) | five — no `withheld` |
| mamba (`:213`) | four — neither `uncached` nor `withheld` |
| mamba-int8 (`:262`, `:270`) | four/five, with different terms zeroed |
| `req_to_token_pool` (`:467`) | a separate two-term invariant that does not call the shared function at all |

**Today's cap is safe because it touches only the full-attention allocator.**
A rung that withholds slots on the SWA or mamba lanes — and spec item 11
names idle mamba states explicitly as spill class — will read as a leak and
kill every rank at the first idle check, exactly as the first one did. Add
the term to that pool's call site FIRST, and add the test that the same
shortfall with no cap engaged is still a leak.

---

## 7. PROCESS NOTES THAT EARNED THEIR PLACE

* **The can-fail instrument paid for itself a third time.** Two boots at a
  raised arming floor found both faults in §1b and §1c inside twelve minutes.
  Neither would have appeared in a normal-floor run, because at the real
  floor the rung barely engages at this fill level.
* **A green unit suite said nothing about either fault.** Both were about the
  difference between two numbers the fakes had modelled as one. When a test
  double collapses a distinction, it does not merely fail to catch the bug —
  it certifies it.
* **Grep the guard list before designing against a subsystem.**
  `flip_blocking_guards` is eleven lines long and would have redirected a
  whole design a shift earlier.
