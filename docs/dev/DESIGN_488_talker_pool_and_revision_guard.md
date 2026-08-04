# DESIGN #488 — TRT lane, talker pool, and the revision guard

## 0. RESUME HERE — state at the 2026-08-04 session checkpoint

Written at a forced session end. Everything below this section is the design;
this section is where the work stands and what to pick up.

**Landed and pushed** (branch `feat/talker-lane-488`):

| slice | state | evidence |
|---|---|---|
| slice 1, raw predictor loop | committed | worth 1.04x alone; its job was enabling capture |
| slice 2, CUDA graphs | committed, **RTF 1.706 → 0.176, 9.71x** | `/spinning/gpu-battery-results/2026-08-04_488_slice2_graphs/` |
| pool-gate mechanism | committed | §2.2d — the reserve worked, the baseline was the false premise |
| Step A install seam | **built, smoke-green, committed** | §4b; `smoke_install_seam.py`, six properties |
| graph-overhead decomposition | **measured, committed** | §2.2f |

**The number that changed the plan, and where it stands.** `ANALYSE_488 §7.4`
budgeted the graphs cut at ~242 MiB. Measured on real geometry: **1184 MiB**
(predictor 183.5, trunk 1000.5 at 1024 slots) — 4.9x the estimate, identical on
sm86 and sm120.

The discriminator that explains it **has been run** and is complete: trunk cost
is **linear in cache slots**, measured at 1024 / 256 / 64 → 1000.5 / 262.3 /
77.8 MiB, fitting `0.96 MiB per slot + 16 MiB` against a static cache of only
0.109 MiB per slot. Capture retains every per-layer expanded-KV intermediate
instead of freeing it, so the cut costs ~8.8x the cache it captures. Slot count
is therefore the dominant VRAM knob of the whole feature, and 1024 is the wrong
default — see §2.2f. Nothing about this is still open to measurement; what is
open is the *choice*, below.

**Open, in the agreed order:**

1. **Trunk slot count.** Blocked on one unmeasured quantity: the real prompt
   length from `generate_icl_prompt` (reference-audio codes + text). Needed
   value is `max prompt tokens + max frames per clause`. Measure it, then the
   law in §2.2f prices the choice directly. Worth 738 MiB between 1024 and 256.
2. **Step A tenant wiring** — artifact is ready (§4b: sampling arguments,
   refusal/fallback table, 183.5 MiB measured, in-process gate
   `gate_loaded_model`). Tenant is **not** to be touched by this strand; the
   wiring waits on the translator strand's latency diagnosis so before/after is
   measured on their instrument.
3. **Pool decision** — price table and three unblock paths are in §2.2g;
   recommendation is path 2 (quantise the instance) plus a right-sized trunk
   cache, *not* raising the serving reserve. Needs a go/no-go before building.
4. **Dispatcher** with the turn-scoped cancel acceptance gate (§2.3), which is
   a hard gate and is co-owned with the translator strand.
5. **TRT lane promotion** — `video_enhance/engine_cache.py` → `srt/trt_lane/`,
   byte-identical video acceptance as the regression proof, talker engine as
   first consumer. Not started. §1a records what graphs left for it: the frame
   is 94.6 % kernel, so fusion is the sole remaining lever.

**Two traps recorded so a successor does not re-spring them.**

* **Read the tree that actually ran.** The serving venv's editable install
  points at `/spinning/htsglang-gpu`, which is a **detached HEAD from
  2026-07-13** that cannot even parse the running cmdline. The live server runs
  `/spinning/wt-530-serving/python` via `PYTHONPATH` in `/proc/<pid>/environ`.
  Reading the editable-install path would have produced a confident wrong
  answer about the reserve. Always resolve the tree from `/proc/<pid>/environ`
  plus `cwd`, never from the venv.
* **The predictor's sampling is not the tenant's.** The tenant passes
  `temperature=0.9, top_p=0.9`; those drive the **trunk**. The code predictor
  keeps `subtalker_*` defaults `top_k=50, top_p=1.0, temperature=0.9`. §4b.


Written 2026-08-04, after the precursor measurement
(`/spinning/gpu-battery-results/2026-08-04_488_precursor/RESULTS.md`:
OVERHEAD-BOUND confirmed, 84.7 % launch gap, recoverable factor 6.53x).

Server-side specification. The client halves named in §4 belong to agent 8.

## 1. TRT is the set lane; graphs are the control arm

User directive: build it with TensorRT. Accepted as the target, and the
precursor supports it with numbers rather than deference — kernel time is
**21.85 ms per frame against a 2.2 ms bandwidth floor**. That 10x is kernel
*shape*, not traffic: hundreds of tiny launches each far below peak. CUDA
graphs remove the 84.7 % launch gap and nothing else; only fusion touches the
10x. Expected ladder, to be measured not asserted:

| stage | frame ms | RTF | what it removes |
|---|---|---|---|
| today | 142.8 | 1.71 | — |
| raw loop, no `generate()` | ~110 est. | ~1.3 | 602 syncs/frame |
| + CUDA graphs | ~22 | **~0.26** | the launch gap |
| + TRT fused engines | ~4-9 est. | **~0.05-0.10** | tiny-kernel shape |

**Both arms get built and the crossover is reported either way.** If TRT loses,
that is a result with numbers, not a reason to have skipped it.

### 1a. MEASURED 2026-08-04 — the graphs rung, and what it leaves for TRT

`/spinning/gpu-battery-results/2026-08-04_488_slice2_graphs/`. The table above
was the hypothesis; this is the measurement, all rungs in one window on the
5090 (400 W limit against a 575 W default — a lowered power target, so these
numbers do not compare across power states):

| stage | predicted | **measured** frame ms | **measured** RTF |
|---|---|---|---|
| today | 1.71 | 142.18 | **1.706** |
| raw loop, no `generate()` | ~1.3 | 136.84 | **1.642** |
| + CUDA graphs | ~0.26 | **14.64** | **0.176** |
| + TRT fused engines | 0.05-0.10 | — | not yet built |

Two corrections fall out, and both move the plan:

* **The raw loop alone bought 1.04x, not the estimated ~1.3 RTF.** The 602
  syncs per frame were not the cost. Slice 1's value was *enabling* capture — a
  region that syncs cannot be captured at all — not speed. The estimate is
  corrected on the record rather than dropped.
* **The graphs rung beat its prediction (0.176 against ~0.26)**, because the
  static cache also removes device work the eager path was doing (`DynamicCache`
  concatenations and per-step mask construction): kernel time per frame fell
  21.26 → 13.85 ms, not just the gap.

**What this leaves for the TRT half, stated precisely.** The graphed frame is
**94.6 % kernel** and the launch gap is 4.6-5.9 %. The 6.53x the precursor
called recoverable has been collected in full; there is no overhead left to
attack. What remains is exactly the residual §1 named: **13.85 ms of kernel
against the 2.2 ms bandwidth floor**, ~6.3x of kernel *shape*. So the
0.05-0.10 band stays arithmetically reachable, and fusion is now the **sole**
remaining lever rather than one of two — which raises the value of the TRT lane
relative to when both arms were open.

Engine discipline, carried over from the video chain (#484 family):

* **Two engines**, matching the two graphs already specified in
  `ANALYSE_488_talker_lane_layout.md` §7.3: the **predictor step** (5 layers,
  17-slot static scratch, one head) and the **vocoder chunk**
  (`chunked_decode`, fixed chunk + 25-frame left context). The 28-layer trunk
  step is a third candidate but is captured by graphs first — it is one engine
  per frame against fifteen for the predictor, so the predictor is where fusion
  pays.
* **I/O signature pinned** per engine: name, dtype, shape, and layout, recorded
  next to the engine file. A silent I/O rename is the failure mode that makes a
  wrong-but-running engine.
* **Dynamic-shape profiles**: predictor is fully static (batch 1, seq 1, scratch
  17) — no profile needed, which is why it is the good first engine. The vocoder
  needs a profile over chunk length (min = 1 frame for the tail, opt = the
  chosen streaming chunk, max = 300 to keep `chunked_decode`'s own default
  reachable).
* **sha256 pins + a consumer matrix**: engine file hash, the checkpoint hash it
  was built from, the TRT version, and the arch. An engine is loaded only when
  all four match; a mismatch refuses by name rather than falling back silently.
* **Separate engines per arch**: `sm120` for the 5090, `sm86` for any 3080 pool
  member. TRT engines are not portable across compute capability, and §2 shows
  the 3080s are exactly where the pool wants to go.

## 1b. CORRECTION 2026-08-04 — TRT is built as a fork lane (#337), not a talker side path

§1's engine discipline stands; its *placement* was wrong. The standing user
order is **#337** (static TRT rungs: precompiled engines as a registry in the
fork, granularly selectable, offloadable through the **#407** tier registry,
hot/cold ± compression, graphs-vs-TRT crossover per regime, sm86 honesty) plus
**#469** (RIFE-TRT as a second backend track). The talker engines are the
**first production consumer** of that lane, not a private path beside it.

**The foundation already exists and must be promoted, not duplicated.**
`video_enhance/engine_cache.py` (261 lines) already implements exactly the
discipline §1 listed, and implements it well:

* `EngineKey` = `nvml_uuid | device_name | driver | trt_version | onnx_sha256 |
  precision | ShapeTriplet(min/opt/max) | builder flags` (`:80-111`) — the
  arch-separation §1 asked for is already structural, not a convention;
* `Provenance` manifests with source-artifact url/sha256/bytes/fetched_at
  (`:123-132`), and **an engine file with no manifest is treated as absent,
  not as usable** (`:19-23`);
* atomic writes (`_atomic_write_bytes`, `:247`).

Adjusted plan for the TRT half:

1. **Promote** the engine cache to a fork-level module (`srt/trt_lane/`), with
   `video_enhance` becoming its first *existing* consumer rather than its
   owner. Byte-identical behaviour for the video chain is the acceptance
   condition — this is a move plus a seam, not a rewrite.
2. **Add what #337 needs on top**: a selectable rung registry (granular
   per-consumer engine choice), **#286 ledger registration** so a loaded engine
   is a parkable asset class like every other VRAM tenant, and **#407 tier
   registry** integration for hot/cold placement ± compression.
3. **Crossover as a first-class output**, per regime: graphs vs TRT measured
   and reported, not asserted. §1's ladder is the hypothesis under test.
4. **sm86 honesty**: separate engines per compute capability, and where an
   engine cannot be built or does not pay on sm86, that is reported by name
   rather than silently falling back.
5. The talker's two engines (predictor step, vocoder chunk) are then declared
   as consumers of this lane; RIFE/SR (#469) attach to the same machinery
   afterwards.

Ordering is unchanged: **graphs first** — the 6.53x is on the table and needs
none of this. The TRT half is cut as lane infrastructure from its first line.

## 2. Talker pool — replication across cards, with an honest per-card verdict

The user's proposal composes exactly with the refutation already on record:
chunks are **independent**, so replication is the right axis where tensor
parallelism was the wrong one. Same three cards, opposite decomposition.

### 2.1 Measured instance footprint — not estimated

The precursor run gives this directly. Free VRAM on the 5090 went
**3605 → 903 MiB** while one standalone talker was resident, of which 24 MiB was
the calibration transient:

> **one bf16 talker instance = 2678 MiB** (checkpoint 1745 + CUDA context +
> cuBLAS workspaces + the 12 Hz codec + the speaker encoder).

### 2.2 Fixed-cost table against the 400 MiB corridor

| card | free now | usable (free − 400) | bf16 (2678) | fp8 (~1806) | verdict |
|---|---|---|---|---|---|
| 5090 (sm120), rank 0 + tenant resident | 3605 | 3205 | **fits** (527 spare) | fits | 1 extra instance, bf16 OK |
| 3080 #0 (sm86), INT8 TP rank | 2951 | 2551 | **DOES NOT FIT** (−127) | fits (745 spare) | fp8/TRT only |
| 3080 #2 (sm86), INT8 TP rank | 2951 | 2551 | **DOES NOT FIT** (−127) | fits (745 spare) | fp8/TRT only |

fp8 figure: the checkpoint's 1745 MiB halves to ~873, everything else unchanged
→ ~1806 MiB. It is an estimate; the 2678 is measured.

**The honest refusal, per the brief:** a **bf16** talker does not fit on either
3080 — it misses by 127 MiB, and taking it would breach the corridor on a card
carrying an INT8 serving rank. So the pool is **1 instance today** (the tenant's
own, on the 5090), or 2 if a second 5090 instance is wanted.

**This re-ranks fp8.** It was lever (3), a speed lever whose gain is unreadable
until the gap closes. It is now a **prerequisite for the pool**: without fp8 (or
an equivalently sized TRT engine) the 3080s are unreachable and the pool cannot
exist. That is a stronger reason to build it than the speed argument was, and it
should be scheduled as such.

### 2.2b CORRECTION 2026-08-04 — the 127 MiB is a rebooking, and fp8 is NOT the gate

The user is right that the miss is not a wall: the card is filled by *our own*
server, so the question is which knob hands back 300-400 MiB. §2.2's "fp8 is
the gate" is withdrawn. What replaces it is narrower than the brief hoped,
because two of the named levers do not reach this boot — read at source rather
than assumed.

**#330 runtime VRAM dial: NOT reachable on the running server.** Two
independent blockers:

1. `--enable-vram-dial` is a **boot flag**, and the deployed launch line does
   not carry it (`/proc/3953200/cmdline`, read 2026-08-04). Without it
   `build_kv_capacity_runtime` is never called and `POST /vram_budget` has no
   runtime behind it. So the "no restart" property does not apply *to this
   instance* — enabling it is itself a restart.
2. Even with the flag, the dial requires **weighted uneven DCP**:
   `if not uneven_dcp_active(dcp_size): raise KvCapacityError`
   (`managers/vram_dial.py:1110`), i.e. `--rank-kv-ratio` non-`coupled` or the
   `SGLANG_UNEVEN_DCP` + `_WEIGHTED` env pair. The boot has
   `--rank-tp-ratio auto-performance --rank-gpu-id 0,1,2` and **no
   `--rank-kv-ratio`**, so the predicate is false.

It also refuses under 11 named combinations (`vram_dial.py:1045-1086`), of
which **dual-group lane** is one — worth noting because §4 of ANALYSE_488
already found the lane is not the pool host either.

**The knob that does reach it is already in the launch line.**
`--rank-auto-reserve-mib 13000,3800,3800` is the #332 per-card reserve, and the
two 3080 entries are the two cards in question. Raising them
**3800 → 4200** hands back ~400 MiB per 3080 at the next serving restart, with
no new flag, no new code and no dependence on the dial's DCP precondition.
That is the cheapest correct form of the rebooking.

**The honest price, in tokens.** Served model geometry: 64 layers of which
**16 are full attention** (`full_attention_interval 4`), 4 kv heads, head_dim
256, `--kv-cache-dtype fp8_e4m3` = 1 byte:

> 16 x 4 x 256 x 2 (K+V) x 1 B = **32 KiB per token** on a rank's own shard
> (heads are replicated under the DCP token-shard geometry; the 48 GDN layers
> carry per-sequence state, not per-token KV, and are unaffected).

| rebooked | tokens lost on that rank |
|---|---|
| 300 MiB | **9,600** |
| 400 MiB | **12,800** |

For scale: 12,800 tokens is ~4.9 % of one 262144-token context, on a boot with
`--max-running-requests 4`. The global effect depends on the token vector, so
the per-rank figure is the honest one to quote.

**Consequence for the plan.** bf16 talker instances (2678 MiB) fit on the 3080s
(2551 MiB usable → 2951 after a 400 MiB rebooking) **as soon as the reserve
change ships with the next serving restart**. The pool does not wait for fp8.
fp8 and TRT engine size return to being what they were before this section
overstated them: **performance and quality levers, not existence conditions**.

### 2.2c STATE CHECK 2026-08-04 (post-reboot boot) — the 400 MiB did NOT show up as free VRAM

§2.2b predicted that raising `--rank-auto-reserve-mib` 3800 → 4200 on the two
3080s would take their free VRAM 2951 → 3351 MiB (usable 2551 → 2951) and
thereby admit a 2678 MiB bf16 talker. The reserve change **did** ship: the
running boot carries `--rank-auto-reserve-mib 13000,4200,4200` (read from
`/proc/1236/cmdline`). The free VRAM did not follow.

Measured on the running boot, 2026-08-04:

| card | free | usable (free − 400 corridor) | bf16 talker 2678 | verdict |
|---|---|---|---|---|
| 3080 #0 | **2957 MiB** | 2557 | −121 | **still does not fit** |
| 3080 #2 | **2959 MiB** | 2559 | −119 | **still does not fit** |

That is 6-8 MiB above the pre-change baseline, not 400. The shortfall is
essentially unchanged from §2.2's original −127 MiB, so **the pool gate is not
open** and the claim that it is should not be carried forward.

### 2.2d MECHANISM RESOLVED 2026-08-04 — the reserve worked; the BASELINE was the false premise

Traced through the code that actually ran (`/spinning/wt-530-serving/python`,
reached via `PYTHONPATH` in `/proc/1236/environ` — *not* the `htsglang-gpu`
tree, which is a detached HEAD from 2026-07-13 that cannot even parse this
cmdline) and confirmed against the boot log.

**The reserve did enter the arithmetic, exactly as designed.** The boot logged:

> `derived memory budgets [19607, 16280, 16280] MiB from NVML totals`
> `(reserve per GPU: {0: 13000, 1: 4200, 2: 4200} MiB).`

`budget = NVML_total − reserve`, split across ranks on that GPU
(`server_args.py:9113-9134`), converted once to a per-rank fraction
(`:9915-9934`), and consumed as the KV-pool ceiling
(`model_runner_kv_cache_mixin.py:558-589`). So 20480 − 4200 = 16280 for each
3080. Nothing is wrong with the knob.

**What the knob does NOT do is cap allocation, and every rank overshoots it:**

| rank | total | reserve | budget | actually resident | overshoot |
|---|---|---|---|---|---|
| 3080 rank 1 | 20480 | 4200 | 16280 | 16966 | **+686** |
| 3080 rank 2 | 20480 | 4200 | 16280 | 16964 | **+684** |
| 5090 rank 0 | 32607 | 13000 | 19607 | 22570 | **+2963** |

This is documented behaviour, not a defect, and it was on record in this repo
before §2.2b was written — `server_args.py:10758-10763`: *"this value shapes the
rank BUDGET ... **It does NOT cap any runtime allocation.**"*, and
`docs/dev/NOTE_493_indexer_prefill_transient.md` §3, which measured the same
class of surprise on a different model. The unbudgeted terms it names are the
CUDA context, NCCL buffers, the flashinfer workspace, graph capture, and
per-layer prefill scratch.

**But the overshoot is a CONSTANT, not a function of the reserve** — which is
what settles the puzzle. Free VRAM is
`total − (budget + overshoot) − driver`, so free moves **1:1** with the
reserve. Reconstructing the counterfactual with the same 686 MiB overshoot:

* reserve 3800 → budget 16680 → free **2557 MiB**
* reserve 4200 → budget 16280 → free **2957 MiB** ✓ (observed)

So the 400 MiB *did* materialise. The false premise is §2.2b's **baseline**:
the "2951 MiB free at reserve 3800" it reasoned from is within 6 MiB of
today's post-change reading, which is what a number measured on a boot that
*already* carried 4200 looks like.

**Honest limit on that conclusion.** The pre-reboot boot's log did not survive
the container restart (`/tmp` holds only this boot's `w537_*` files), so the
alternative reading — the old boot really did have 2951 free at 3800, and the
new boot's overshoot happens to be ~400 MiB larger — cannot be excluded from
surviving evidence. It does not matter for any decision, because both readings
give the same present state and the same lever.

**The lever, priced.** Free moves 1:1 with the reserve, and KV budget moves
1:1 against it. At the model's 32 KiB per token on a rank's own shard:

| 3080 reserve | free | usable (−400) | bf16 talker 2678 | KV cost |
|---|---|---|---|---|
| 4200 (today) | 2957 | 2557 | **−121** | — |
| 4400 | 3157 | 2757 | +79 (tight) | ≈ −6,400 tok |
| 4600 | 3357 | 2957 | +279 | ≈ −12,800 tok |

Against a 333254-token pool that is 1.9 % / 3.8 % — *if* the 3080s are the
min-reducing ranks, which is not established (`_apply_token_constraints`
min-reduces token capacity across ranks, and the 5090 rank has both a larger
budget and a larger weight shard). Worth measuring before spending.

### 2.2e RETRACTED 2026-08-04 — the "slice 2 devalued the pool" verdict was cut on one axis

**The verdict below is withdrawn. It is kept, not deleted, because the way it
was wrong is more instructive than the conclusion was.**

It priced the pool on a single axis — the latency of *one* synthesis — found
that graphs had already collected most of that win, and concluded the pool was
now worth ~10x less. The axis it never checked: a turn is **not one
synthesis**. `session.py:1852-1871` already splits a turn into clauses and
enqueues them, so a pool parallelises *within* a turn. On that axis graphs and
the pool **compose multiplicatively** — graphs make each clause ~10x cheaper,
the pool overlaps N clauses — and a pool that is worthless for one clause is
worth close to Nx on a multi-clause turn. Turn latency is what the user
experiences; single-clause latency is not.

**The rule this cost, stated so it is reusable:** a building block is devalued
only after checking **every** axis it is used on, not the first one. A speedup
on axis A does not retire a mechanism that also serves axis B. Concretely here:
"faster per clause" (graphs) and "more clauses at once" (pool) are orthogonal,
and orthogonal levers multiply. Same failure mode as reading a conclusion off a
sample narrower than the question.

The correct pricing, with the graph overhead now measured rather than
estimated, is §2.2f. The paragraph below is the retracted one.

#### (retracted) CONSEQUENCE — slice 2 devalued this pool by ~10x, and it should be deprioritised

The pool's justification was head-of-line blocking: `inprocess_tts.py:124-134`,
a turn arriving mid-synthesis *"pays the whole of that synthesis, a median
4.8 s"*. Slice 2 measured the token loop at **9.71x faster** (RTF 1.706 →
0.176), which turns that median into roughly **half a second**, and lets a
single instance sustain ~5.7x real-time audio.

So the pool's remaining value is **multi-user concurrency, not single-user
latency** — and the live use case driving this work (the DE↔ES translator) is
one user. Spending 6,400-12,800 tokens of a shared serving resource to place a
second talker on a 3080 now buys about a tenth of what it would have bought
before slice 2.

**Recommendation: do not spend the KV tokens.** Leave the reserve at 4200,
leave the pool unbuilt, and revisit only if concurrent speakers become a real
requirement. The 121 MiB is no longer a blocker to route around; it is a price
signal on something that stopped being worth buying.

*(end of retracted paragraph)*

### 2.2f THE REAL PRICE — graph overhead MEASURED, and it is 5x the estimate

`ANALYSE_488 §7.4` budgeted the whole graphs cut at ~242 MiB, of which ~64 MiB
for the trunk graph pool. Measured on the real geometry
(`scripts/dev/488_talker_profile/measure_graphed_footprint.py`, artefacts in
`/spinning/gpu-battery-results/2026-08-04_488_slice2_graphs/`):

| term | measured | estimate | ratio |
|---|---|---|---|
| 15 predictor graphs + 16-slot scratch + uniform pool | **183.5 MiB** | ~60 | 3.1x |
| trunk graph + 1024-slot static cache | **1000.5 MiB** | ~117 + 64 | 5.5x |
| **total** | **1184.0 MiB** | ~242 | **4.9x** |

Identical on sm86 and sm120 to the decimal, so the overhead is allocator- and
shape-driven, not architecture-driven — one number prices both cards.

**The trunk term is linear in cache slots**, measured at three points
(1024 → 1000.5, 256 → 262.3, 64 → 77.8 MiB), fitting

> **trunk overhead ≈ 0.96 MiB per slot + 16 MiB**

against a static cache that is only **0.109 MiB per slot**. So capture costs
~8.8x the cache it is capturing, because a captured region retains every
intermediate — including the per-layer expanded-KV copies GQA makes — instead
of freeing them. That is the mechanism, and it makes **slot count the dominant
VRAM knob of the whole cut**.

**Consequence: 1024 slots is the wrong default.** It buys 85 s of audio at
12 Hz for a workload whose clause is ~3.2 s, and costs 738 MiB more than 256
slots. The right value is `max prompt tokens + max frames per clause`, and the
prompt length has NOT been measured yet (`generate_icl_prompt` carries
reference-audio codes plus text), so the default is not fixed here — the law
above lets any choice be priced once that number exists. Until then 1024 stands
as the safe upper bound, with its price named.

**Instance footprint, composed** (2678 MiB eager base measured in the
precursor, overhead measured here), against a 3080's usable 2557 MiB
(2957 free − 400 corridor):

| instance form | MiB | vs 2557 |
|---|---|---|
| eager, no graphs | 2678 | −121 |
| + predictor graphs (Step A) | 2862 | −305 |
| + trunk graph @ 64 slots | 2939 | −382 |
| + trunk graph @ 256 slots | 3124 | −567 |
| + trunk graph @ 1024 slots | 3862 | −1305 |

So the honest correction to §2.2c: the gap is not 121 MiB. **A fully graphed
instance misses a 3080 by 567 MiB at 256 slots** — the graphs cut, which is
what makes a pool member fast enough to be worth pooling, is also what put it
out of reach. Note also that an *eager* pool member is not a workaround: at
RTF 1.7 it is a straggler, and §2.3's in-order assembly makes one straggler
stall everything behind it.

### 2.2g THREE WAYS TO UNBLOCK, priced against the measured numbers

| # | path | closes | cost | verdict |
|---|---|---|---|---|
| 1 | raise the 3080 reserve | 1:1 in MiB | 567 MiB ⇒ reserve 4200→4770, ≈18,100 KV tokens (~5.4 % of the 333254 pool) | works, but spends a **shared serving** resource on a **per-tenant** feature |
| 2 | quantise the talker (fp8/int8 weights) | ~872 MiB (checkpoint 1745→~873) | instance @256 slots → **2252 MiB, fits with 305 MiB spare**; needs a prosody quality gate | **cheapest correct route**: it shrinks the thing that does not fit instead of taxing the thing that works |
| 3 | second instance on the 5090 | — | graphed @256 = 3124 MiB against 3463 free ⇒ leaves 339 MiB, **below the 400 corridor** | does not fit *today*; depends on the tenant's own size, which moved 4524 → 5916 MiB this session |

**Recommendation.** If the pool is wanted — and per §2.2e it is, on the
turn-latency axis — the route is **shrink the instance, not the serving pool**:
path 2, combined with a right-sized trunk cache. Two supporting facts make this
the clear ordering:

* right-sizing the trunk cache alone (1024 → 256) is worth **738 MiB**, which
  is **six times the entire original 121 MiB gap**. The cheapest MiB available
  is the one this cut spent by accident, and it is recovered by choosing a
  number rather than by building anything.
* path 1 is the easiest to *implement* (one flag) and the worst to *own*: it
  permanently converts serving KV capacity into talker headroom, and §2.2d
  showed how badly that knob's effects get misremembered between sessions.

fp8 therefore returns to the plan not as a speed lever (`ANALYSE_488 §7.6`
correctly ranked that third) but as **the pool's enabling condition** — which
is what §2.2 originally claimed before §2.2b withdrew it. The difference is
that it is now backed by measured numbers on both sides.

Still to check honestly before any fp8 quantisation effort, per the brief: an
sm86 TRT engine at bf16/fp16 may already land under 2551 MiB on its own, in
which case the 3080s are reachable with neither fp8 nor a reserve change. That
is a measurement on the built engine, not a prediction.

Secondary and complementary levers, unchanged from the brief and NOT priced
here because the reserve route is cheaper: `--rank-kv-ratio` shifting KV mass
toward the 5090 at the next restart, #364 GDN idle vacate, #286 parking.

### 2.3 Pool mechanics

* **Chunk queue**, not turn queue: the unit of work is the clause that
  `session.py:1858` already enqueues. No new segmentation.
* **N instances, one claim each.** An instance takes the head of the queue.
* **IN-ORDER ASSEMBLY is mandatory and is the whole risk.** Chunk *n+1* must
  never reach the client before *n*. The queue hands out a monotonic
  `unit_index` (it exists — `session.py:1856`), and a reorder buffer releases
  chunk *n+1* only once *n* has been fully emitted. Out-of-order audio is worse
  than slow audio: it is unintelligible.
* **Reference-voice sharing**: the speaker embedding is computed once per turn
  and is small; it is passed with the work item. Each instance holds its own
  speaker encoder anyway (part of the 2678 MiB), so nothing is shared by
  pointer across processes.
* **TURN-SCOPED CANCEL — a HARD acceptance gate, not a nice-to-have.** The
  revision guard (§3.3) can invalidate a turn *while the pool is mid-flight on
  its clauses*, and the pool is precisely what makes that likely: N instances
  running ahead means more speculative work exists at the moment a correction
  lands. Cancel must therefore be **turn-scoped and complete**, in all three
  places a clause of that turn can be:
  * **running** — the in-flight unit on each instance is aborted;
  * **queued** — every queued unit carrying that `turn_id` is invalidated
    before it is claimed;
  * **already delivered** — `turn.speech.abort {turn_id}` (§4) makes the client
    drop that turn's audio, and only that turn's.

  Then the clause pipeline **refills with the corrected clauses**, and in-order
  assembly continues across the seam: the reorder buffer must not emit a
  corrected chunk before the correction prefix, and must never emit a chunk of
  the abandoned generation at all.

  **Acceptance criterion, stated so it can fail:** take a multi-clause turn,
  inject a material correction mid-synthesis while at least two instances hold
  clauses of it, and require (a) **no audio whatsoever** reaches the client
  from any discarded clause, (b) the corrected turn arrives **complete and in
  order**, prefix first, (c) **no orphan chunk** survives into the next turn,
  and (d) no `unit_index` gap or repeat in the emitted sequence. All four are
  observable on the existing turn log plus the client's received sequence, so
  this is a test, not an inspection.

  **Why drop-and-refill rather than salvage:** at RTF 0.176 a discarded clause
  costs ~0.5 s of GPU time that was going to be idle anyway. Salvaging partial
  audio across a meaning change is complex, risks emitting a fragment of a
  sentence the speaker did not mean, and buys back something nearly free.
  Correct semantics beat recovered work here, and the arithmetic says so rather
  than taste.

  Ownership: the trigger side (deciding a correction is material, stamping
  `turn_id`) belongs to the translator strand; the queue/instance/reorder side
  is this one. The gate above is written to be runnable by whichever side holds
  the harness, and neither half is testable alone.

* **Straggler rule**: a 3080 instance is ~2.4x slower than the 5090 on weight
  bandwidth alone. In-order assembly means a slow instance holding chunk *n*
  stalls everything behind it. So the dispatcher must be **head-biased**: the
  fastest free instance takes the lowest outstanding index, never a later one.
  Without that rule the pool can be slower than one instance.

## 3. Chunk policy — always eager, with a revision guard

Final user design: always synthesize clause-wise; a guard detects a
meaning-break against the final MT and repairs it. The self-stabilising
argument is accepted and is implemented **explicitly**, not hoped for.

### 3.1 Context-adaptive clause choice (the mechanism, made explicit)

The insight: when a repair is injected, playback falls further behind the
speaker, and that lag is itself an analysis buffer — later clauses can be
synthesized against more settled text.

Implemented as a rule, not a side effect:

```
lag = (audio queued but not yet played) in seconds
if lag >= LAG_SETTLED_S:      synthesize against the FINAL MT text
elif lag >= LAG_PARTIAL_S:    synthesize against the stream, but only clauses
                              already followed by a further clause
else:                          synthesize the newest clause eagerly
```

The lag regulates aggressiveness by itself: a repair raises the lag, which moves
the next clauses onto the settled path, which makes a second break unlikely.
Both thresholds are server parameters with defaults, exposed for tuning.

### 3.2 The guard — two stages, asynchronous, never in front of playback

Hooked on the **MT-final event** (`session.py:1944`, where `accumulator.flush()`
produces the tail and `mt_total_ms` is stamped), running **beside** playback:

1. **Deterministic first, no LLM call.** If the concatenation of already-spoken
   clauses is a prefix of the final translation (after normalisation:
   whitespace, casing, terminal punctuation), there is nothing to repair. This
   is the overwhelmingly common case and it must cost nothing.
2. **Only on divergence, ask Qwen on the fast lane** whether the difference is
   *material*. Word reordering, a synonym, a moved adverb — not a break. A
   changed subject, a negation, a reversed clause relation — a break. The prompt
   returns a strict verdict token plus the corrected text; anything unparsable
   is treated as "no break" (a false repair is worse than a missed one: it
   interrupts audio the user is understanding).

### 3.3 Repair path

On a material break:

1. **Abort playback for that turn** — cut the server-side chunk queue for the
   turn and signal the client. `stopAll` exists client-side; what is missing is
   a **per-turn abort channel** (§4).
2. **Play the correction prefix from RAM, immediately** — see §3.4.
3. **Re-synthesize the corrected sentences** in parallel with the prefix
   playing, so the repair is audible in ~0 ms and the real content follows
   without a second gap.

### 3.4 Pre-synthesized correction prefix cache

The prefix ("Entschuldige, ich meinte:" / "Perdona, quería decir:") is fixed
text, so it is synthesized **once per (target language × active voice)** and
held as PCM in RAM:

* built as soon as a cloned voice is established, and at boot for each preset
  voice;
* keyed `(voice_id, target_lang)`;
* **invalidated on roster events** — voice switch, voice merge, voice delete.
  Those events already exist (`speakers.py` / the roster path); the cache
  subscribes rather than polling.
* cost: ~1.5 s of 24 kHz float32 mono per entry ≈ 144 KiB. Negligible in RAM,
  and it is RAM, not VRAM, so it does not touch the corridor.

### 3.5 Measurement — observability, not a mode selector

Per the correction: revision rate and repair latency are **measured and
reported**, but they do not drive mode selection. Sentence-final remains a plain
user option with **no recommendation label**. Logged per turn on the existing
decision/turn log: `clauses_streamed`, `prefix_match` (stage-1 outcome),
`guard_invoked`, `guard_verdict`, `repair_latency_ms`, `lag_at_clause_s`.

## 4. Server interfaces for agent 8 (client halves are his)

| interface | server side (mine) | client side (agent 8) |
|---|---|---|
| chunk policy | `chunk_policy: eager \| sentence_final` parameter on the synthesis path, plus `lag_settled_s` / `lag_partial_s` | the selector UI, no recommendation label |
| per-turn abort | new event `turn.speech.abort {turn_id}`; server cuts the turn's chunk queue and stops the in-flight unit | handle the event by dropping queued audio for that `turn_id` only — **not** a global `stopAll` |
| correction turn | `turn.speech` unit carrying `kind: "correction"` and `prefix_cached: true` | render the turn as repaired (visual marker), keep the original text visible struck-through or greyed |
| guard telemetry | the six fields in §3.5 on the turn log | nothing required |

The abort channel is the one genuinely new protocol element; everything else
extends fields on events that already exist.

## 4b. STEP A HANDOFF — the install seam, ready for the tenant

Built and executed 2026-08-04. **The tenant is not touched by this work**; what
follows is the artifact the wiring consumes, so the wiring is a two-line change
made by whoever owns the tenant, after their latency diagnosis has established
the before-picture.

**What it is.** `GraphedPredictorFrame.install()` swaps
`talker.code_predictor.generate` for fifteen captured graphs. Nothing else
changes: `generate_voice_clone`, the prompt builder, the trunk loop and the
codec are untouched, because the reference reads `.sequences` off the result
and gets the same shape it always did.

**Measured effect** (`2026-08-04_488_slice2_graphs/RESULTS.md`): frame
142.18 → 47.64 ms, **RTF 1.706 → 0.572, 2.98x**; a 5511 ms clause becomes
**~1850 ms**. The remaining 3.3x needs Step B (the trunk graph + decode
driver), which is a bigger cut and unlocks intra-clause streaming.

**Sampling arguments — required, and the one way to get this wrong.**

```python
driver = GraphedPredictorFrame.install(
    talker, model=tts._model,          # model= is what the defaults are read from
    do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
)
```

Those four are **required keyword arguments with no defaults**, because a graph
bakes its warpers: they are part of what was compiled, not configuration. And
they are **not** the tenant's `temperature=0.9, top_p=0.9`
(`inprocess_tts.py:358-360`) — those are the **trunk's** knobs. The code
predictor is driven by `subtalker_*`, which the tenant does not pass and which
therefore keep the reference's own defaults (`modeling_qwen3_tts.py:2036-2039`):
`top_k=50, top_p=1.0, temperature=0.9`. Installing the tenant's `top_p=0.9`
would apply a warper the reference never applies — permanent, silent, audible
only as timbre. `install()` reads the reference's signature and **refuses at
load time** on a mismatch, naming both tuples; the refusal is pinned by a test.

**Refusal and fallback behaviour, stated exactly:**

| condition | behaviour | why |
|---|---|---|
| install-time sampling ≠ reference `subtalker_*` defaults | **raises** `GraphCaptureRefusal` | load-time, before any audio; a wrong warper is silent forever |
| a call arrives with sampling the graph was not captured for | **falls back** to the reference `generate`, warns once, increments `driver.sampling_fallbacks` | the reference output is *correct*, just ~10x slower — raising would break a live turn to prevent something the fallback already prevents |
| `generate()` before `capture()` | raises | otherwise the eager path's timings get reported as the graphed arm's |
| replay out of capture order (shared pool) | raises | returns another graph's live intermediates as plausible numbers |

`driver.sampling_fallbacks > 0` means the tenant is silently running eager and
**not** getting the speedup — expose it in telemetry rather than wondering why
the latency table did not move. `driver.uninstall()` restores the reference
without a restart; that is the operator's way out.

**VRAM cost — measured, not estimated: 183.5 MiB.** Identical on sm86 and
sm120. Covers the fifteen graphs (one shared pool), the 16-slot scratch cache
and a 4096-frame uniform pool (0.24 MiB). Step B would add the trunk term,
which is `0.96 MiB × slots + 16 MiB` — see §2.2f, and note that its default
slot count is an open, priced decision rather than a settled one.

**Why a uniform pool exists at all.** A replayed graph cannot draw randomness —
whatever entropy was consumed at capture is baked, so every frame would sample
identically. The pool is device-resident and its cursor advances **inside** the
last step's graph, so per frame the host does nothing. Wrap-around after 4096
frames reuses uniform *values* against different logits (not a repeated
output); `reseed()` is available for a caller that wants to cut even that.

**Gate script, runnable in the live tenant.**

```python
from validate_graphs import gate_loaded_model
report = gate_loaded_model(tts._model)      # keep_installed=False by default
```

In-process against the weights already loaded: no second copy, no restart, no
conversation state touched, no audio synthesised. It reads the sampling from
the reference itself (so it cannot be handed the wrong values), installs,
compares the graphed tokens against the eager path over several frames, reports
VRAM delta and the rig-wide corridor before and after, and **uninstalls before
returning** unless asked not to. Green means bit-identical on that instance's
own weights.

**Execution evidence, because desk-written code is unvalidated.**
`scripts/dev/488_talker_profile/smoke_install_seam.py` runs the whole seam on a
randomly-initialised model of the same class for ~50 MiB, and proves six
properties: capture works, the pool advances inside the graph (identical input
⇒ *different* tokens, which a baked draw would break), explicit uniforms
reproduce the eager path **bit-exactly**, the seam is live, `uninstall()`
restores it, and the sampling-trap refusal fires. **Green on 2026-08-04.** The
full-weights in-process gate is `gate_loaded_model`, and it runs where the
weights already are — the tenant — rather than paying 2.7 GiB to load a second
copy on a contended card.

## 5. Ordering, with the dependency that is easy to miss

1. **Raw 15-step predictor loop** — DONE (slice 1). Worth 1.04x on its own; its
   real job was making the region capturable.
2. **CUDA graphs** — DONE (slice 2), **RTF 0.176, 9.71x**, gates green. Both
   the control arm and, at 94.6 % kernel, the thing that leaves TRT nothing to
   attack except kernel shape.
3. **TRT engines** (predictor, then vocoder) — the set lane.
4. **fp8** — no longer optional: §2.2 makes it the gate on the 3080s.
5. **Talker pool** — needs 4 to reach more than one card.
6. **Revision guard + repair** — independent of 1-5 and can proceed in parallel;
   it is latency *perception*, not throughput.

Streaming synthesis is not a separate item: it falls out of 1, and it must not
ship before RTF < 1 or an intra-clause stream underruns the player.
