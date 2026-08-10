# #656 HANDOFF v25 — successor 22

Written 2026-08-10, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this before HANDOFF_664. 664's diagnosis was right and is now built and
measured; its section 12a model is superseded by a simpler one; and two of
its open questions (the `pp-stage-ratio` remap rule, the "two distinct
bounds" framing) are closed.

---

## 1. MY ERRORS, ranked — read these before my results

**0. My first measurement of the thing I was fixing counted the wrong
bytes, and it looked plausible.** The live-set probe reported 60.1 MiB
against a 19.2 MiB floor. Twenty of those MiB were the KV POOLS: an
in-place op returns the tensor it mutated, so `write_rows`' `k[idx] = ...`
handed the probe the destination pool itself, 32 buffers of 0.64 MiB. I
only caught it because I printed the *breakdown at the high-water* rather
than the peak alone, saw thirty-four identical 0.64 MiB entries, and went
looking for a tensor of exactly that size. **A peak number with no
itemisation cannot be checked against anything.** Every instrument in this
chain that has misled someone was reporting a single scalar. The probe now
takes an explicit exclusion set and prints the resident list on failure.

**1. I asked for a one-layer geometry move and got a four-layer one, and I
had been warned.** 664 §6 left the `15,10,7 -> 16/9/7` remap unexplained
and said to verify the achieved split before spending a measurement on it.
I did verify (see §5) — but I had already sized the *expected gain* from
the un-quantised request, so my desk estimate for rank1 (+838 MiB) was
against a move that does not exist. The real move gave +1558 and cost rank0
2792. The estimate being wrong did not cost a boot only because I checked
the split before reading the result.

**2. I sized a pool from a single-request transient and it breached in
eight minutes.** The mover's live set is the union of ALL resident
requests' slots; the scratch ladder issues one request at a time; the
deployment runs bs=4. Measured on the binding card: 1120 MiB alone, 1370
under a bs=4 soak, **1790** with a 111405-token prefill on top — and still
deepening when I stopped the load. I had sized pool 550000 against 1120.
**Every transient figure in this corpus, successor 21's and mine, is a
single-request measurement and therefore a LOWER bound**, and every
capacity row that names a pool without naming both the concurrency and the
longest prefill is not reproducible. What this does NOT touch is the mover
A/B in §2: same ladder, same request, same trigger, only the code moved.

**3. My first green-run watchdog cried wolf in its second minute.** A 5 s
`curl` timeout against a scheduler doing a bs=4 round with a long prefill
in flight is not a health signal. Three probes at 15 s all returned 200.
**A liveness check has to be slower than the slowest legitimate round**, and
the fix is consecutive failures, not a longer single timeout — a real wedge
must still be caught.

---

## 2. THE HEADLINE: the mover was the transient, and fixing it removed the breach

664 §13 traced the length-scaling VRAM transient to the phase-flip KV mover
and named the fix without building it. Built here.

**Hermetic** (three-rank threaded flip, production `row_nbytes = 2048 B`,
`scripts/s22_mover_live_set.py`):

| direction | peak before | peak after | plan floor | ratio |
|---|---|---|---|---|
| pp_to_tp | 39.4 MiB | **19.2 MiB** | 19.2 MiB | 2.05x -> **1.00x** |
| tp_to_pp | 36.6 MiB | **20.2 MiB** | 19.2 MiB | 1.91x -> **1.05x** |

**Metal**, pool 400000, the exact 111405-token request that breached the
corridor for successor 21, same geometry, only the code moved:

| axis | s21 old mover (664 §12) | s22 streamed mover |
|---|---|---|
| corridor min, rank1 (binding) | **719 MiB — 305 UNDER floor** | **2765 MiB — 1741 ABOVE** |
| corridor min, rank0 / rank2 | 3106 / 1151 | 5852 / 2915 |
| request wall time | 68.5 s | **22.6 s** |
| `FLIP ABANDONED` | 0 | 0 |
| `#cached-token` | — | 0 (a genuine full prefill) |

Three changes; the third had not been itemised by anyone:

1. `_pack_outgoing` fills one exact-size buffer per peer in place instead of
   one tensor per layer + `torch.cat` + a checksum-appended copy;
2. `read_rows_into` is the pool-view primitive that allows it, and
   `read_rows` is now built on top of it so the two cannot drift into
   producing different bytes for the same rows;
3. **the outgoing buffers are released the moment `_exchange` returns.**
   They used to stay referenced through the local read, the backing swap and
   every write — the widest part of the move.

The 3x wall-time gain is the same fix, not a separate one: the mover
allocated and freed hundreds of MiB per flip, dozens of times per request.

## 3. THE ORDERING HAZARD IS DISCHARGED, NOT MERELY RESPECTED

`_staging_bytes` was `2 x outgoing + incoming` **with the local retained leg
missing entirely**. It is now
`incoming + max(outgoing, local) + one_layer_window` — a MAX, because the
send buffers die at the exchange and the local leg is not read until after.

664 §13c warned that shipping this alone makes the livelock strictly more
reachable. Streaming landed first, so the honest budget (20.2 MiB predicted
vs 19.2/20.2 measured) is *smaller* than the dishonest one it replaced
(30.2 MiB). On metal: a flip that reserves 457 MiB under the new formula
would have reserved 650 MiB under the old one while needing less than
either.

`test_phase_flip_staging_reserve_631` pinned the superseded formula BY NAME
(`test_outgoing_is_counted_twice_and_incoming_once`), which is how a missing
term looked deliberate for five successors. Rewritten with a fixture in
which the retained leg dominates.

**No livelock at pool 600000**, where the old mover wedged at 500000.

## 4. 664 §12a's MODEL IS SUPERSEDED: the two terms separate

Same 111405-token trigger at two pools 200000 apart:

| | rank1 | rank0 | rank2 |
|---|---|---|---|
| transient at pool 400000 | 1120 | 1346 | 982 |
| transient at pool 600000 | 1120 | 1366 | 926 |

Identical. The transient is a property of the REQUEST and the geometry; the
idle floor is a property of the POOL. With the old mover the staging peak
scaled with the resident live set which scaled with the pool, which is
exactly why 664 needed a per-pool fit and had to call the corridor and
livelock bounds "two distinct bounds". **One additive bound:**

```
  holds  iff  idle_free(pool) - transient(longest prefill) >= 1024 MiB
```

| card | idle slope, MiB per 1000 pool tokens | max pool holding 1024 |
|---|---|---|
| rank1 (3080) | 10.30 | **567,000** <- binding |
| rank0 (5090) | 14.58 | 729,800 |
| rank2 (3080) | 8.36 | 617,100 |

Validated forward, not just fitted: the model predicted rank1's idle free at
pool 550000 as 2324 MiB; measured 2329.

### ALL PER-CARD TRIPLES IN THIS CORPUS ARE IN nvidia-smi INDEX ORDER

`(rank1, rank0, rank2)`, NOT rank order. `SGLANG_RANK_CARD_UUIDS` on the
live schedulers puts rank0 on the 5090 (nvidia-smi index 1) and ranks 1/2 on
the 3080s (indices 0 and 2), and 664 §12a labels its own slopes "24 (rank1)
/ 32 (rank0) / 20 (rank2)" in that order. Read the wrong way round,
`719/3106/1151` says the 5090 was the binding card. It was not.

## 5. THE `pp-stage-ratio` REMAP RULE, closed (664 §6)

Stage layer counts quantise to **multiples of 4** — the full-attention block
period (64 layers, one full-attention layer in four). `pp-stage-ratio` is in
32nds; the raw request is `ratio/32*64` layers, rounded to a multiple of 4,
sum held at 64.

| stage ratio | raw | achieved `pp_layer_ratio` | full-attn per stage |
|---|---|---|---|
| 14,10,8 | 28,20,16 | **28,20,16** (exact) | 7,5,4 |
| 15,9,8 | 30,18,16 | **32,16,16** | 8,4,4 |

The boot log prints `pp_layer_ratio=[...]` and `pp_stage_ratio=[...]`
adjacent — read the split there, not from an arena size.

## 6. THE 600000 TARGET IS NOT MET, AND THE REASON IS STRUCTURAL

| pool 600000 | rank1 | rank0 | rank2 |
|---|---|---|---|
| floor under load, 14,10,8 | **689** (335 under) | 2916 | 1223 |
| floor under load, 15,9,8 | 2131 | **64** (960 under) | 1167 |

The levelling step the user's §10 note asks for is a FOUR-layer move, and it
overshoots into a near-OOM: rank0 goes from 1892 MiB of margin to 64 MiB
free, because it gains four layers AND a full-attention layer's worth of
transient at once.

**So the surplus is not an untaken tuning nicety — at this layer count it is
unreachable with the layer knob.** The token vector cannot substitute: after
alignment every rank's KV is `max(KV_pp, KV_tp)` with both equal, so
lowering a rank's TP share leaves it PP-bound and buys it nothing while
costing whoever receives the share.

Two named routes remain, both real work:

1. **Spill rung 2 (draft weights)** — 1925 MiB on the binding card,
   resident in both phases, and the PP phase has no drafter at all. At this
   aligned geometry PP binds, so it pays its full size where it is needed,
   against a 335 MiB shortfall. Needs the VA-stable carrier (`KvVmmArena`)
   because the dead module's `restore()` moves draft addresses that the TP
   decode graphs bake. **This is now the highest-value unbuilt item.**
2. **Rebuilding the PP prefill KV layout**, which user spec item 2
   explicitly permits — the only lever that can break the
   `max(KV_pp, KV_tp)` symmetry that makes the token vector inert.

## 7. A TRAP IN THE ONE TOOL MANDATED FOR CAPACITY STEPS

`scripts/seam_scaling_reboot.py` read `/tmp/boot_cmdline.txt` +
`/tmp/boot_env.txt`, written by whoever last ran the boot script. Against
the running server they were 10 hours stale:

| | capture file | live server |
|---|---|---|
| `pp-stage-ratio` | 2,1,1 | 14,10,8 |
| `rank-gpu-memory-mib` | 22700,11920,11970 | 31800,17400,17450 |
| `SGLANG_UNEVEN_TOKEN_VECTOR` | 28,26,20 | 14,10,8 |
| `PHASE_FLIP_PURITY` | **off** | **strict** |

The tool prescribed for single-variable discipline would have booted a
different geometry with strict purity DISABLED and reported it as a
one-variable step. **That is worse than a hand-built environment, because it
looks disciplined.** Rewritten: reads the live process, stops it by PID in
the same invocation (capture and kill cannot be separated without losing the
baseline), takes explicit `--set-arg/--add-flag/--del-flag/--set-env`
substitutions, warns when more than one moved, and writes a
baseline-vs-replay record per boot to
`/spinning/evidence-631/boot-captures/`.

## 8. WHAT I DID NOT REACH

* **Decode decomposition (program item 3).** Not done. One observation that
  narrows it for a successor: with `max_new_tokens` small, no `Decode batch`
  or `accept len` line appears at all — decode logging is gated on
  `decode_log_interval` (default 40 iterations), so a short-generation
  workload produces ZERO decode evidence and the bs=1-vs-4 question cannot
  be answered from such a log. Drive long generations, or lower the
  interval, before concluding anything about decode batch size.
* **Prefill chunk A/B and the coordinator's DYNAMIC arm (item 4/5).** Not
  measured. Static-chunk negative control is recorded: with the flag off
  there are zero `[PP Dynamic Chunk]` lines and every prefill chunk is
  `#new-token: 2048`. Two things a successor should know before booting the
  dynamic arm:
  - `enable_dynamic_chunking` passes its `pp_size > 1` gate in this config,
    then runs `profile_and_init_predictor`, which executes **128 real
    prefill requests at boot on PP0** and self-disables with
    `[PP Dynamic Chunk] Failed to profile prefill latency` on any exception.
    That log line, or `Predictor ready (quadratic)`, is the engagement
    evidence — flag presence is not.
  - `server_args.max_prefill_buffer_tokens` grows to
    `max(chunked, max_prefill_tokens, ceil(chunked*1.25))` when dynamic
    chunking is on with `pp_size > 1`: **2048 -> 16384, an 8x prefill buffer
    ceiling**, which lands directly on the 1024 MiB corridor. Cost it before
    crediting any speed win.
  - the runtime `Predicted chunk size` line is `logger.debug` and will not
    appear at the default level; the observable that does is the spread of
    `#new-token:` across a request's prefill chunks.
* **Spill rung 2**, §6's highest-value item.
* **The host-RAM lever** (664 §6): `RssShmem` 58.3 GiB of pinned weight
  images, unswappable on a 117 GiB box, still the mechanism of the
  `oom_kill 9` precedent. Untouched.

## 9. THE CORRIDOR AND THE STAGING BOUND HAVE COME APART

At pool 550000 the corridor broke by 485 MiB — rank1 to 539 MiB free — and
the instance stayed at `/health` 200 with **0 `FLIP ABANDONED` and 0
tracebacks** throughout. Successor 21's pool 500000 livelocked at a rank
that was **13 MiB** short of its staging reserve.

That is the practical payoff of §2 and §3 together: the corridor is now a
BUDGETING question, where for the whole of this chain it was an
AVAILABILITY one. A successor who overshoots the pool now gets a card
sitting under 1024 MiB, not an instance that stops answering.

## 10. STATE LEFT BEHIND

Serving geometry `14,10,8`, verified as `pp_layer_ratio=[28,20,16]`,
`SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8`, `MAMBA_SLOTS=12`,
`RANK_MIB=31800,17400,17450`, CTX 393216, purity strict, policy auto,
spill depth cache.

**Pool: use 470000.** Derived, not guessed — `idle_free(P) - 1790 >= 1024`
on rank1 with a measured slope of 10.30 MiB per 1000 pool tokens gives a
ceiling of ~503000, and 470000 leaves 339 MiB for further high-water growth
because the transient had not finished deepening when the load stopped.

Ladder of what was measured, so nobody re-runs it:

| pool | geometry | outcome |
|---|---|---|
| 400000 | 14,10,8 | holds, 1741 MiB margin, single-request load |
| 470000 | 14,10,8 | **the green-run target** |
| 550000 | 14,10,8 | breaks by 485 MiB under bs=4 + long prefill |
| 600000 | 14,10,8 | breaks by 335 MiB, single-request load |
| 600000 | 15,9,8 | breaks by 960 MiB (rank0 to 64 MiB free) |

Against the 190000 successor 21 shipped, 470000 is **+147%**, and unlike
their 500000 row it is measured under the concurrency the deployment
actually runs at.

## 11. THE GREEN RUN, NOT DONE — and the exact recipe, because it is now cheap

I did not complete a >=60 minute green run. The one I started was at pool
550000 and disqualified itself in eight minutes for the reason in §1.2; I
converted it into the transient measurement rather than let it burn an hour
proving a number I already knew was wrong.

Everything needed is built. Run it exactly like this:

```bash
cd /spinning/wt-631-routea
# 1. Boot. ONLY through the replay tool -- it reads the live process, so a
#    server must be up first. Verify the achieved split from its output.
/spinning/htsglang-gpu/.venv/bin/python scripts/seam_scaling_reboot.py 470000
#    then confirm: pp_layer_ratio=[28,20,16], pp_stage_ratio=[14,10,8]

# 2. Corridor time series at 100 ms, for the whole window plus slack.
setsid bash scripts/corridor_sample.sh 4500 /spinning/evidence-631/s22/green2/corridor.csv &

# 3. Load, all three kinds at once -- the acceptance claim is about the
#    COMBINATION, and each alone hides something:
setsid bash scripts/s20_soak.sh 3900 /spinning/evidence-631/s22/green2/soak 4 &
setsid bash /spinning/evidence-631/s22/green/longprefill.sh &   # 4 x 111405 tokens
#    plus REAL agent traffic: spawn qwen subagents on genuine read-only
#    repo work. They must NOT carry a model override, or they bypass the
#    instance under test.

# 4. Verdict, one command, every axis:
/spinning/htsglang-gpu/.venv/bin/python scripts/s22_green_verdict.py \
    /spinning/evidence-631/s22/green2/corridor.csv

# 5. Decode decomposition -- run it INSIDE the window, not after:
/spinning/htsglang-gpu/.venv/bin/python scripts/s22_decode_probe.py \
    --concurrency 4 --max-new 600 --rounds 3
```

Predicted at 470000: rank1 floor ~1363 (margin +339), rank0 and rank2 far
above. If rank1 lands materially below that, the transient deepened past
1790 and the pool must come down by `shortfall / 10.30` thousand tokens.

**Do not reboot while agent traffic is live** — I lost time to exactly that
constraint at the end of my session, and it is the right constraint.

## 12. WHAT I WOULD DO NEXT, in order

1. **The green run above.** Everything is built; it is an hour of waiting.
2. **Spill rung 2 (draft weights)** on the VA-stable `KvVmmArena` carrier.
   1925 MiB on the binding card against a 335 MiB shortfall at pool 600000,
   and §6 shows it is the only lever that can reach the user's target now
   that the layer knob is proven quantised. Highest value by a distance.
3. **Decode decomposition** with `scripts/s22_decode_probe.py`. The bs=1
   suspicion is already falsified (histogram `{1:15, 2:42, 3:90}`); the open
   question is in-phase tok/s versus plain TP3, which is what the probe
   separates.
4. **Chunk A/B including the dynamic arm**, with the two traps in §8:
   the discriminator is a chunk EXCEEDING `chunked_prefill_size` (a static
   run already shows ~19 distinct sizes), and enabling the flag grows
   `max_prefill_buffer_tokens` 2048 -> 16384, which lands on the corridor.

## 13. THE POOL LEVER IS EXHAUSTED — found at the very end, and it redirects the remedy

Rebooting 550000 -> 470000 as a clean single-variable step, the KV pool
really did shrink: `max_total_num_tokens=470000` applied, and per-rank
`K size` came back 3.14 / 2.24 / 1.79 GB in exactly the 14:10:8 ratio,
about 750 MiB less on rank1 than at 550000.

**Idle free on rank1 moved 2329 -> 2303. Twenty-six MiB.** The model
predicted 824.

So below some point, lowering `--max-total-tokens` does not buy corridor
headroom at all: the freed KV bytes are re-absorbed inside the SAME
per-rank static budget. This is HANDOFF_664 §6's audited finding arriving
as a measurement — `--rank-gpu-memory-mib` is ADVISORY (it becomes
`mem_fraction_static`, consumed once in `_profile_available_bytes`),
`torch.cuda.set_per_process_memory_fraction` is called nowhere in the tree,
and nothing stops the allocator expanding into the corridor.

### What this changes

* **§4's capacity curve is valid as a fit over 400000-600000 and must not
  be extrapolated below it.** Three points sat on a line and the fourth,
  outside the fitted range, is 850 MiB off. A two-point slope through a
  regime boundary is not a model.
* **The corridor lever is `--rank-gpu-memory-mib` on the binding rank, not
  the pool.** To give rank1 ~850 MiB of corridor the step is
  `--set-arg rank-gpu-memory-mib 31800,16550,17450` — a single-variable
  step through the replay tool. The pool may then re-size itself downward,
  and THAT is the honest trade between pool and corridor. This chain has
  never made that trade explicitly; it has only ever moved the pool and
  hoped.
* It retro-explains why successor 21's four-boot geometry search moved the
  bottleneck between cards without ever lifting the minimum much: the
  budgets stayed put, so the floor did too.
* It promotes 664 §6's "single highest-value unbuilt item" — a per-rank
  `set_per_process_memory_fraction` derived from `RANK_MIB` — from a
  tidiness argument to the actual mechanism. With it the corridor becomes
  an enforced invariant and the allocator reclaims instead of expanding.

### The falsifier, left running

The green run at 470000 was left running deliberately as the test: if it
breaches at about the same depth as 550000 did (rank1 ~539), the pool lever
is confirmed dead over this range and the next successor should go straight
to the budget lever. Read it with:

```
/spinning/htsglang-gpu/.venv/bin/python scripts/s22_green_verdict.py \
    /spinning/evidence-631/s22/green2/corridor.csv
```

**Revised order for §12, in light of this:** the budget lever
(`--rank-gpu-memory-mib` on rank1, then the enforced
`set_per_process_memory_fraction`) comes BEFORE spill rung 2. It is a flag
change plus a small patch, against rung 2's VA-stable carrier work, and it
addresses the term that is actually binding.
