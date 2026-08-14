# #584 — CARD-RATE MEASUREMENT PASS, and the P2 re-solve it unblocked

Shift `m584`, window 2026-08-14 10:57Z–11:2xZ. Tree: the merged line
`4d5419609a` → `feat/584-cardrates`. Evidence `/spinning/evidence-631/m584/`.
No serving boot ran in this window; the only device work was the card probe.

**Headline.** The measurement pass exists, ran on all three cards, and put
measured rates on disk keyed by card UUID. With them, `--pp-solve-cut` executed
**end to end through the wired handler for the first time on this rig**. Under
the governing rule now pinned in `RUNSHEET_485_CERTIFICATION.md` §6b the answer
is a **REFUSAL**, and per §5.1b outcome (3) that refusal *is* the result: **no
W2 certification window is owed**. The budget lever is exhausted by physics; the
only recovering lever is a **72 % seam reduction**, and it selects a different
cut again.

---

## 1. WHAT WAS ACTUALLY MISSING — not a probe

`WINDOW_VERDICT_485_R12.md` §2 defect 3 named it as "a measurement pass". The
measurement already existed. `rigmon/card_probe.py` (#213) measures
`gemm_bf16_tflops` and `membw_gbs` per card with `uneven_perf`'s own kernels,
caches them UUID-keyed under `~/.cache/sglang`, and feeds eighteen planner
sites. Three things were missing, and only the third is interesting:

1. `_pp_cut_card_rates` built a seed-only `CardLibrary()`. Every one of the 16
   `SEED_CARDS` carries `gemm_tflops=None` / `membw_gbs=None` by design — they
   are curated **nameplate** entries.
2. `CardLibrary.save`/`load` existed for exactly this purpose and **nothing on
   any branch ever called either**.
3. **The reason nobody called them: neither takes a default, and no code
   anywhere computed a path.** The class had persistence with no *location*. A
   store nobody can name is a store nobody can fill — so the gate was not merely
   uncalibrated, it was *uncalibratable*, and its refusal had no reachable
   remedy.

So the deliverable is a **bridge plus a location**, not a new benchmark. A
second opinion on GEMM rates would have been the wrong thing to build.

## 2. THE PASS

`python/sglang/srt/planner/card_rate_pass.py`, CLI
`python -m sglang.srt.planner.card_rate_pass --run | --show`.

| | |
|---|---|
| **measures** | nothing new — runs the #213 probe and projects it |
| **quantities** | `gemm_tflops`, `membw_gbs`: exactly the two `_pp_cut_card_rates` reads |
| **key** | **card UUID**, with PCI BDF attached from `registry/nvml.py`'s `IdentityMap` |
| **persists to** | `~/.cache/sglang/card_library.json` (env `SGLANG_CARD_LIBRARY`), beside the probe it projects from |
| **plus** | `card_library.json.by-uuid.json` — the UUID-keyed sidecar |

**Why a sidecar.** `CardLibrary`'s own format is **name-keyed**, so saving it
alone would drop per-card identity. This rig carries two RTX 3080s that
`props.name` cannot tell apart, and they do not measure identically. The
sidecar keeps the spread auditable; the library is the projection the solver
consumes. The residency census now records `gpu_uuid` beside `gpu_name` for the
same reason — its own comment already said *"the IdentityMap belongs in the
artifact, not in the reader's assumption"*, while it wrote only the name.

**Where two cards share a name, the slowest wins.** A pipeline's makespan is set
by its pacer, so pricing a name by its faster instance under-predicts the
makespan of whichever stage lands on the slower one, and an under-predicted
makespan admits cuts that should have been refused. The reverse can only
over-predict. Same direction as every other refusal in this gate.

## 3. MEASURED ON METAL

```
GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d  0A:00.0  RTX 5090  203.57 TFLOPS  1661.7 GB/s
GPU-5c648f96-be1d-42d5-0221-34d11ab137f7  05:00.0  RTX 3080   51.14 TFLOPS   716.6 GB/s
GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4  0B:00.0  RTX 3080   50.81 TFLOPS   717.4 GB/s
```

All three flagged `THROTTLED`, and that flag is **correct and expected**: this
rig runs deliberately reduced power targets (3080s 200 W of 320 W, 5090 400 W
of 600 W, `m584/nvml_identity.csv`). These therefore ARE its operating rates,
which is what the gate should price. The caveat is recorded rather than
suppressed, per #149.

### The borrowed rates were stale, and that matters for the retired number

R12 fed the gate rates a previous shift measured (`s50/gate_check.py`: 5090
231.97 / 1533.8, 3080 65.57 / 717.4). Against measurement:

| card | quantity | borrowed | measured | delta |
|---|---|---:|---:|---:|
| 5090 | gemm | 231.97 | **203.57** | **−12.2 %** |
| 5090 | membw | 1533.8 | **1661.7** | +8.3 % |
| 3080 | gemm | 65.57 | **50.81** | **−22.5 %** |
| 3080 | membw | 717.4 | **717.4** | 0.0 % |

Bandwidth reproduces to the decimal; **GEMM does not**, in the direction and
roughly the magnitude the 2026-08-05 power-target reduction would produce. So
the `37,14,13` admission R12 recorded rested on **pre-power-cap** compute
rates. Recorded because §6a already retired the +25.5 % on a change of cut, and
this is a second, independent reason the old numbers do not describe this rig.

### A capacity collision the unit tests predicted and metal confirmed

The seed set holds `RTX 3080` (10240 MiB) and `RTX 3080 20GB` as deliberately
distinct entries. The driver calls **both** `NVIDIA GeForce RTX 3080`, and
`_canonical` strips only vendor words — so this rig's 20 GB cards resolve onto
the **10 GB** profile. Left alone the library would have described a 20480 MiB
card as 10240 MiB while carrying that card's measured rates. The pass corrects
capacity from the measurement and records the correction as a caveat. It fired
on metal exactly as `test_t11b` said it would.

## 4. P2 — THE WIRED GATE RAN, AND THE GOVERNING ANSWER IS A REFUSAL

First execution of `--pp-solve-cut` end-to-end on measured rates. Pool 280000,
budget `31400,19300,19300`:

| census | outcome |
|---|---|
| **M0** (ship cut, the most demanding) | **REFUSED** — no feasible layer split |
| M1 (`40,12,12`) | ADMITS `37,14,13` |
| W1 (`37,14,13`) | ADMITS `37,14,13` |

That the admitted cut is **unchanged at `37,14,13`** despite a −12 % / −22 %
move in the GEMM rates is worth stating: on this shape the choice is driven by
memory feasibility, not by the compute balance the rates set.

**§6b takes the most demanding census. The governing outcome is REFUSAL** —
runsheet §5.1b outcome (3). **No W2 window is owed**, and none was run.

A second, independent route reaches the same place: under §6b's pooled-worst
transient (7055 MiB → 8079 MiB of at-rest free required), the cut M1/W1 admit
fails C2′ by **−3643.3 MiB** (W1's at-rest was 4435.7). The refusal does not
depend on which of the two routes a reader prefers.

## 5. LEVER ARITHMETIC — the refusal arrives with its remedy

Swept against the **real M0 census**, not a fixture
(`m584/p2_levers_m0.txt`, `p2_levers_ceiling.txt`, `p2_seam_lever_m0.txt`).

### L1 — `--rank-gpu-memory-mib`: EXHAUSTED BY PHYSICS

| binding-rank (5090) budget | outcome |
|---|---|
| 31400 (base) … 32500 | refused — no feasible split |
| **32607 = the card's ENTIRE NVML total** | **refused** |
| 32900 (+1500) | refused — *physical impossibility*, exceeds the card |

And at the rig's absolute ceiling, every card at full nameplate
(`32607,20480,20480` — a configuration that leaves zero corridor and zero
context overhead, i.e. not deployable):

```
still REFUSED: needs 31218 MiB weights + 8750 MiB KV = 39968 MiB
               against 48684 MiB usable
```

**The aggregate fits by 8716 MiB and the solve still fails.** The refusal is a
**contiguity/packing** result — R12 suspected this; it is now measured. No
budget lever can reach it, because there is no budget left to give.

### L3 — `--max-total-tokens`: CONFIRMED NOT A LEVER, now on metal

280000 → 240000 → 200000 → 160000 → 120000 → 80000 → 40000 → 20000: **every
one refused.** §5.1a asserted this from a fixture proxy; it is now measured on
the real census. The shortfall sits in the fixed terms and the pool is not one
of them. The seam-cap guard's operator advice still lists it first, and on this
shape it is the one knob that cannot help.

### L2 — the seam: THE ONLY RECOVERING LEVER, and it is a large ask

Scaling only the `SEAM_*` terms of the census the gate reads:

| seam (max across ranks) | outcome |
|---:|---|
| 1838.0 MiB — **as measured** | refused |
| 624.9 / 588.2 / 551.4 | refused |
| **514.6** | **ADMITS `29,19,16`** |
| 477.9 … 0.0 | admits `29,19,16` |

**Threshold: the seam must fall to ≈515 MiB — a reduction of ≈1323 MiB, 72 % of
the measured draw.** Independently close to §5.1a's fixture-derived "~500 MiB",
which is a useful corroboration of that note.

**And it selects a THIRD cut.** Not `40,12,12`, not `37,14,13`, but
`29,19,16`. Per §6a any throughput number would again be retired. A shift that
achieves the seam reduction does not inherit a certified cut; it inherits a new
one to certify.

## 6. #363 — flip targets, counted (bonus, not the protocol)

| question | answer |
|---|---|
| Is the card probe now visible to the planner's own lookup? | **YES** — `cached_card_probe()` returns the matched 3-card probe with rates |
| `build_stage_table` flip targets from planner candidates | **0** |
| …with per-stage measurements present | **2 stages, 1 flip target**, `reach=reshard` |

**#584's card-rate half CLEARS #363's blocker-one precondition** — the
`PlannerFeedUnavailable('no card probe on disk')` that every rank logged is
gone. **It does not produce flip targets**, because `build_stage_table` refuses
a solved-but-unmeasured candidate:

```
RegimeError: stage table refused (#578): the planner solved 1 stage(s) --
solved-enc -- but they carry no measurement. Each needs measured_gain_pct,
measured_band_pct and flip_cost_s ... The solver cannot predict any of the three.
```

The #363 verdict predicted this hermetically; it is now confirmed with a real
probe on disk. **The remaining gap is the OTHER half of #584's measurement
pass** — per-stage A-vs-A gain, band, and instrumented flip cost — and the Q2b
row proves the machinery works the moment those exist. That is the next slice,
and it is now the only thing between #363 and a flip target.

## 7. WHAT THIS SHIFT DID NOT DO

* **No W2 certification window.** Outcome (3) says a refusal is the result; the
  window would have certified a cut the governing gate does not admit.
* **No A/B throughput probe.** §6a and §7a both forbid it before a cut is
  admitted, and none is.
* **No corridor certification statistics**, because no certification boot ran.
  The only corridor numbers this shift can offer are the restored ship's, which
  are a restore check and not a window.
* **The 1258 MiB excess is still unexplained.** §6b is written so that
  explaining it is what narrows the reference class back to the cut's own
  class. That remains the highest-value open item on #485.
