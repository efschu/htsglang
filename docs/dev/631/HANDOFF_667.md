# HANDOFF 667 — successor 24, task #656 / #631 Route A

**Errors and open risks are ranked first, as they must be. The fixes are
below them.**

Branch `feat/route-a-631`. Commits added by this successor:

| commit | what |
|---|---|
| `427db8f279` | wave the flip's seam over layers — the 270k livelock, removed |
| `510fb632a0` | price the seam's two constants honestly, block the gather |
| `74fef893a1` | bench section 2e |

---

## 1. WHAT IS STILL WRONG, ranked

### 1.1 An abandon under strict purity is STILL a wedge, not a degradation

This is the top residual and it is structural, not incidental. The waved
seam makes the refusal far harder to reach — staging at a FULL POOL is
now 550 MiB against 3102 spendable in the fixture, so no request that
fits the pool should provoke one — but *if* a flip is ever abandoned
while a long request is resident under strict purity, the same livelock
returns: the request cannot decode in PP, so it stays resident, so the
live set stays large, so the next attempt refuses identically.

The three candidate remedies, unchanged from HANDOFF_666 and still
unbuilt:

* **(a)** make the abandon path recoverable from inside the scheduler —
  retract or preempt the resident request, or drop its prefix cache, then
  retry;
* **(b)** bound staging further (row-chunking *within* a wave, which needs
  the destination wave restored BEFORE the source wave is released — see
  1.3 for why that ordering is not free);
* **(c)** fix `is_fully_idle()` so `/flush_cache` works in the state that
  needs it. **That alone turns the wedge into a recoverable event** and is
  the cheapest of the three.

I did NOT build (c), deliberately: diagnosing it needs the wedged state,
and inventing a scheduler change from a guess is how this file fills up
with plausible wrong answers. What I built instead is the diagnostic that
answers it in one line next time — `flush_cache` now names WHICH clause of
`is_fully_idle()` is false (`Scheduler.idle_blockers()`). The next
occurrence produces the answer directly; do not re-derive it from theory.

My unverified suspicion, recorded as a suspicion: `last_batch` is the
likely clause, because a parked request leaves the last prefill batch
referenced while no new batch can run. **Check the log line, do not trust
this sentence.**

### 1.2 The wave count is capped by the SMALLEST STAGE, and here that is 4

`default_wave_count` is `min(len(stage) for stage in layer_map)` — because
a wave's releases pay for its commits only while every rank contributes at
least one of its own layers. On this rig the flip layer map is
**[7, 5, 4] over 16 full-attention layers** (NOT [28,20,16] — that is the
*pp_stage_ratio* over all 64 layers; the flip only moves full-attention
KV). So the seam gets **4 waves, not 16**, and staging falls by ~4x rather
than ~16x.

It was enough. But anyone who changes `--pp-stage-ratio` to something with
a smaller minimum stage gets fewer waves and more staging, silently. If a
future geometry has a stage with 1 or 2 full-attention layers, the seam is
effectively unwaved again. **This coupling is not currently guarded by
anything that would fail loudly.** A boot-time warning when the wave count
falls below (say) 4 would be cheap insurance.

### 1.3 Aliased pools and a waved seam are mutually exclusive

Found by a falsifier that failed rather than by reading code, which is the
only reason it is known. `TestSharedArenaReadsPrecedeWrites` models the two
layouts sharing one arena — the named capacity follow-up "one arena per
rank sized max(PP, TP)". Waving interleaves reads of one layout with writes
of the other, so under aliasing a wave's writes can land on rows a later
wave has not read yet.

The runtime now detects overlap from real pointers (`KvPoolView.overlaps`)
and collapses the seam to ONE wave with a loud log. That is correct and
safe, but it means **building the aliased arena silently gives back the
livelock's precondition.** If that design is ever picked up, the seam must
be re-thought at the same time — the two cannot simply be composed.

### 1.4 The corridor is still LOOSE — the other half of the law is unmet

The law has two halves: never breach 1024 MiB free per card, AND be best
filled (free NEAR 1024, not far above it). Measured minima on the binding
card were 2203 MiB during the 270k probe and higher still early in the
green run. That is roughly 1200+ MiB per card of headroom the user is
paying for and not receiving.

Staging is no longer the binding term, so the capacity lever is open again
for the first time in several successors. See section 4 for the ledger and
the step I did not get to.

### 1.5 Traps that cost me time, recorded so they cost the next reader none

* **The calibration fixture drifts when the formula changes.** Blocking the
  gather cut the window term, so the fixture that "reproduces the measured
  3855 MiB" needed its row width recalibrated (536 → 543 bytes). That is
  legitimate, but it means the fixture pins the SHAPE of the demand, not a
  bit-exact replay. Do not treat it as a replay.
* **A waiter that greps for `error` matches `indexing errors`** in a
  transformers warning and fires immediately. My first probe waiter
  reported "finished" while the probe was still building its prompt.
* **The log holds several boots.** Every count must be taken after the LAST
  `PHASE-FLIP armed at boot`, or it silently mixes configurations.
* **`_staging_bytes` is called with 4 args by older tests**, where the
  runtime is built by `__new__` and has no `_n_layers` and no
  `_pre_write_fns`. `waves=None` is therefore a NULL FILTER, not "one wave
  over range(n_layers)", and the swap lookup uses `getattr(..., ())`.
* The replay tool still refuses a no-op relaunch; reboot the same config
  with `--set-arg max-total-tokens <same value>`.

---

## 2. WHAT WAS FIXED, and how it is known

### The defect

The seam swapped both layouts' physical KV backing exactly ONCE, so every
byte crossing it had to be resident at that instant. Staging was
`sum(row_nbytes * n_rows)` over the whole plan — proportional to the
resident live set, unbounded in request length. The gate was honest; the
move was wrong.

### The fix

Wave the seam over layer groups: release the source layout's backing and
restore the destination's one wave at a time. Each wave takes a
proportional slice of every rank's own layer block, so a wave's releases
pay for its commits and residency never rises above the resting layout
while the staged bytes fall by the wave count.

New API: `KvVmmBufferOwner.{finalize,shrink}(buffer_indices=...)`,
pool-level `release_backing(layers)` / `restore_backing(layers)`,
`phase_flip_plan.{layer_waves, default_wave_count}`, and
`WavedBackingSwap` replacing the one-shot swap closure.

Then two constants that dominated once the legs were divided: the backing
slack (a flat one-layer charge, 3x too big — now walked from the actual
wave plan) and the gather window (`k[rows]` and a strided `.contiguous()`
materialised the whole row list — now blocked over rows).

### Evidence

**CPU**, flip family suite: **680 passed, 0 failed**.

* the rig fixture reproduces the measured 3853 MiB unwaved, needs **418
  MiB waved**, and **550 MiB at a FULL POOL** — the closure, since the
  live set cannot exceed the pool;
* the unwaved path is pinned as still unaffordable, so the class **can
  fail**;
* **byte identity**: the same hermetic three-rank flip at 1 wave and at 4
  produces identical destination pools, and the waved run lands the
  reference rows (equality between two runs proves nothing if both are
  wrong, so both are checked).

**CUDA**, real driver VMM: 4 passed, including a per-buffer subset round
trip — releasing buffers 0 and 2 leaves 1 and 3 mapped, readable and
writable, driver free memory rises, addresses survive, and an
out-of-range subset raises before anything is unmapped.

**Metal**, the reproducer that wedged (270031 tokens, bs=1, purity strict):

| axis | successor 23 | successor 24 |
|---|---|---|
| FLIP ABANDONED | ~1/s, forever | **0** |
| flip committed | never | **yes**, at 270003 / 270012 live slots |
| health | 503 | **200 throughout** |
| staging reserved | 3855 needed vs 3102 spendable | **2258 / 2192 / 1896 MiB** |
| corridor min free | — | **2203 / 5154 / 2419**, 0 breaches |
| recovery | reboot only | not needed |

**>262144 YaRN leg (spec item 4): GREEN.** All three needles verified
including the deep one at ~95% depth, at `prompt_tokens` 270031, plus a
deep-only re-ask from the same session. It DECODED — under strict purity
that can only have happened in TP, so this is not a prefill-only pass.

---

## 3. THE GREEN RUN — 64.5 min, ZERO ABANDONS, corridor held

Commit `510fb632a0`, pool 380000, `RANK_MIB 31800,14000,15600`, purity
strict, spill depth cache, 4 seam waves. Recipe deliberately identical to
the 68-min row (bench 2d) so the two compare: bs=4 soak + a repeating
111405-token prefill ladder + a repeating decode probe + live agent
lanes, all concurrently.

| axis | 68-min run (unwaved) | this run (waved) |
|---|---|---|
| duration / corridor samples | 68.1 min / 29740 | **64.5 min / 28114** |
| **FLIP ABANDONED** | **51 lines = 17 events** | **0** |
| flip DONE lines | 834 (= 278 flips) | **813 (= 271 flips)** |
| tracebacks | 0 | **0** |
| corridor min (idx 0/1/2) | 1215 / 3542 / 1349 | **2699 / 5732 / 2831** |
| breaches of 1024 | 0 | **0** |
| worst margin | +191 | **+1675** |
| prefill batches / with a CUDA graph | 10989 / **0** | 8049 / **0** |
| purity refusals | 138 | 136 |
| decode batches / accept len | 1011 / **2.54** | 1647 / **2.734** |
| decode `#running-req` | {1:453, 2:270, 3:234, 4:54} | **{1:624, 2:516, 3:489, 4:18}** |
| host `memory.peak` / `oom_kill` | 112.1 GiB / 9 | **112.1 GiB / 9 (unchanged)** |

**Agent traffic is evidenced, not asserted.** The serving log carries 142
`/v1/messages` and their 142 `/v1/messages/count` companions — the
agent-SDK request shape arriving through the router — plus 124
`/v1/chat/completions`, on top of 119 `/v1/completions` from the soak and
ladder legs. Two qwen analysis lanes were live for the whole window doing
real work, launched with no model override.

Evidence: `/spinning/evidence-631/s24/green/` (corridor.csv, soak.log,
ladder.log, decode_probe.log).

### Reading it

* **The abandons are gone.** 17 events under the old code, 0 here, at a
  comparable flip count. That is the whole point of the waved seam and it
  is the number to re-check after any change to the seam.
* **Strict purity held**: 8049 prefill batches, not one with a CUDA graph.
* **Accept length recovered to 2.734** from 2.54, against a plain-TP
  reference of about 2.9. Not explained — see 5.3 — but moving the right
  way, and this run has 1647 decode batches behind it rather than 1011.
* **Decode reaches bs=4**, so the bs=1 suspicion inherited from earlier
  successors is settled for a second time.
* **The corridor is now VERY loose**: worst margin +1675 MiB where the old
  run had +191. Part of that is the smaller staging peak and part is a
  lighter prefill count (8049 vs 10989), so do not read the whole 1484 MiB
  as recovered headroom. It is, however, unambiguous that the pool lever is
  open again.

---

## 4. CAPACITY: the ledger, and the step not taken

Staging is off the critical path, so the pool lever is live again.

Geometry, established rather than assumed:

* flip layer map **[7, 5, 4] over 16 full-attention layers**;
* one KV row (K+V, one layer, one token slot) = **2048 bytes**, derived
  from `sent 1063153 cells / 2076.47 MiB`;
* resident KV per pool token: rank0 `7*2048 = 14336 B`, rank1
  `5*2048 = 10240 B`, rank2 `4*2048 = 8192 B` — and each equals that
  rank's TP-layout cost (`16 * 2048 * share`), as it must;
* physical mapping resolved **empirically** (not assumed — NVML order and
  rank order diverge here): nvidia-smi **index 1 is the 5090 and hosts
  rank 0**; index 0 hosts rank 1; index 2 hosts rank 2. Both 3080s are
  20480 MiB.

**The binding card is nvidia-smi index 0 (rank 1)** — smallest measured
free at 2203 MiB during the 270k probe.

### The staging curve, and why it is trustworthy

Pricing the live geometry through the real code path
(`build_phase_flip_transition` + `_staging_bytes`, rank 1, 4 waves) gives

    S(L) = 4.517 MiB per 1000 live slots * L  +  357 MiB constant

and at L = 176873 it predicts **1156 MiB against the 1153 MiB the green
run actually logged**. The model is therefore not an assumption; it
reproduces the measurement to 0.3%.

### The ceiling, and the bad news

The binding condition is no longer the corridor as such — it is the
ANTI-WEDGE condition, which is stricter and is the same inequality: a flip
must stay affordable when the live set fills the pool, because an
unaffordable flip at high occupancy under strict purity is the livelock
again.

With baseline free on the binding card inferred at 3852 MiB
(2699 measured minimum + the 1153 MiB staging peak added back):

    3852 - (P - 380000)*9.766/1000 - S(P) >= 1024      ->  **P <= 432,861**

**So the ledgered ceiling is about 432000 tokens, +14% over today — and
the >= 600000 target in my brief is NOT reachable by stepping the pool.**
Reaching 600000 needs the staging SLOPE to fall from 4.517 to about 0.537
MiB per 1000 slots, an 8.4x reduction. No pool arithmetic gets there.

This corrects the natural reading of the loose corridor. The +1675 MiB of
apparent margin is not spendable on pool, because roughly two thirds of
every pool token's cost now comes back as staging at full occupancy:
resident 9.766 MiB/1000 tokens, staging 4.517 on top of it.

### What WOULD get there, quantified

The slope exists because a wave's payload is held across that wave's
backing swap. Invert the intra-wave order and it stops being a slope:

* today: **release** the source wave, then **restore** the destination
  wave, holding the wave's LIVE payload across the swap -> cost scales
  with the live set;
* instead: **restore** the destination wave FIRST, then row-chunk the
  exchange inside the wave (read a row block, exchange it, write it
  straight into the already-backed destination), then **release** the
  source wave last. The peak becomes one wave's DESTINATION POOL SPAN plus
  one row block -> a constant in the live set.

That span is `(n_layers/W) * share * row_bytes` per pool token, i.e.
2.441 MiB/1000 at W=4, 0.610 at W=16. And it lifts the wave-count cap as a
side effect: `default_wave_count` is limited by the smallest stage only
because a wave's releases must pay for its commits, and restore-first
budgets that overlap explicitly instead.

    W = 16, restore-first:  P <= ~630,000     (clears the 600000 target)

**This is derived, not measured.** It assumes the corridor baseline holds
and that the row-chunked exchange costs what the arithmetic says. It is
the single highest-value unbuilt item and it is a contained change to
`_execute` plus `WavedBackingSwap`.

### Do not carry any of this into a boot without measuring

The 432000 ceiling was NOT tested — I did not step the pool. The baseline
free term is INFERRED (measured minimum plus the staging peak added back),
so it is the one number in the chain that a real boot could contradict. A
first trial at 410000 with a >= 12 min mixed load would validate or break
the whole ledger cheaply, and that is the right next action.

---

## 5. WHAT I DID NOT GET TO

In the order I would take them:

1. **Validate the capacity ledger** with one reboot at 410000 and a >= 12
   min mixed load. It either confirms section 4 or breaks it, cheaply.
2. **Restore-first + intra-wave row chunking** (section 4). It is what
   turns the staging slope into a constant and the only route to 600000
   that the arithmetic supports.
3. **Decode decomposition — ANSWERED, see section 7.** What remains is
   the bs4/bs8 A/B against plain TP3, which I did not run.
4. **Accept length 2.54 vs the ~2.9 measured under plain TP** — cause
   unknown.
5. **Prefill chunk A/B.** Note HANDOFF_666's correction: the dynamic arm is
   NOT free — `chunked_prefill_size` has ten memory-sizing consumers and
   nine read it RAW without the 1.25x inflation.
6. **The wave-count floor guard** (1.2) and the **recoverable abandon**
   (1.1c).

---

## 6. STATE AT HANDOFF

* Branch `feat/route-a-631`, pushed to the fork. Tree clean.
* **Serving is UP and healthy** on `510fb632a0`, port 30030, pool 380000,
  geometry `pp_stage_ratio 14,10,8`, flip layer map [7,5,4] over 16
  full-attention layers, purity strict, policy auto, 4 seam waves.
* Two qwen analysis lanes may still be running (capacity byte ledger;
  decode decomposition). **Stop them before rebooting serving.** Their
  findings were not folded into this document — re-ask if needed.
* GPU arbitration holder is mine; heartbeat stopped before release.
* Harness nit worth one minute of someone's time:
  `scripts/s24_green_run.sh` passes `--out '$OUT/ladder_$i.json'` inside a
  nested quote where `$i` does not expand, so every ladder pass overwrites
  one file. The per-rung numbers still land in `ladder.log`; only the JSON
  is affected.

### The one-line summary

The 270k one-request livelock is fixed at its root — the seam is waved,
staging no longer tracks the live set, and a full pool now prices at 550
MiB where a 270k request used to price at 3855. It is proven by the
reproducer that used to wedge, by a 64.5-minute traffic run with zero
abandons, and by the >262144 YaRN leg decoding rather than merely
prefilling. What remains is capacity (the corridor is loose), the decode
decomposition, and the recoverable-abandon safety net.


---

## 7. DECODE DECOMPOSITION — answered, with the part I had to reject

Open since successor 17. The phase intervals come from the server log's own
flip markers (`PHASE-FLIP DONE pp_to_tp` opens a TP interval, the next
`tp_to_pp` closes it), taken on the **PP2 prefix only** so the three-rank
triplet counts once. Window is the green run, 13:20:00–14:25:00.

| quantity | value | source |
|---|---|---|
| TP intervals | 137 | flip markers |
| TP time | 1174 s of 3900 s | flip markers |
| **TP duty cycle** | **30.1 %** | flip markers |
| mean TP interval | 8.6 s | flip markers |
| decode tok/s **while a batch runs** | **93.7** | scheduler's own `gen throughput`, 2541 batches |
| decode tok/s **in-phase** (TP window, incl. TP idle) | **>= 34.0** | 39936 soak tokens / 1174 s |
| decode tok/s **wall-clock** | **>= 10.2** | 39936 / 3900 s |
| **cost of the alternation** | **3.32x** | = 1 / duty cycle |

Three levels, and the ordering is the result: 93.7 while decoding, ~34
across the TP phase, ~10 across the wall clock. The gap from 93.7 to 34 is
**idle time inside the TP phase** — the instance is in the decode layout
with nothing to decode. The gap from 34 to 10 is the PP phase, where decode
cannot run at all.

**The in-phase and wall-clock figures are LOWER BOUNDS.** 39936 is the
soak leg's client-side count; the decode probe and the agent lanes also
decoded and are not in it. The duty cycle and the 93.7 are not bounds.

Note the "ratio" carries no information beyond the duty cycle — same
numerator over two denominators, so 3.32x IS 1/0.301. The honest headline
is the duty cycle, and the lever on it is TP interval length (8.6 s mean)
against PP interval length, not decode speed.

### The part I rejected, and why it is recorded

A qwen analysis lane produced this decomposition and reported **4.24 tok/s
in-phase / 1.35 wall-clock**, from a count of 7040 accepted tokens summed
out of `OUTTRACE round kind=decode ... accept_lens=[...]` lines.

**I could not reproduce that count** — the pattern sums to zero against the
same log (there are 23049 OUTTRACE lines, but in the form
`[#631 OUTTRACE] pre[-2] site=pp_end ...`). Two independent sources
contradict 7040: the soak alone reports 39936 decode tokens client-side,
and the scheduler's own mean `gen throughput` is 93.7 tok/s, which cannot
coexist with 4.24 tok/s in-phase. So the throughput half of that report is
**not adopted**.

Its duty-cycle half DID reproduce (31.9 % over a wider window that included
post-run idle, against my 30.1 % restricted to the load window), and its
accept length (2.73) matches my independent 2.734. Methodology sound,
arithmetic input unverified — which is exactly why agent reports are not
evidence.
