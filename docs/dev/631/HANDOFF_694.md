# HANDOFF 694 — the gate is honest now, and it still says no

Successor 50. Branch `feat/route-a-631`. Evidence `/spinning/evidence-631/s50/`
(`README.txt` indexes it, `RESULTS.md` is the verdict, `gate_check.py` and
`ledger.py` are runnable and reproduce every number below).

**Both of the brief's blockers are closed with numbers. The verdict is still
GATE, on two blockers that are new, measured, and specific.**

---

## 1. ERRORS FIRST

### 1a. I repeated law 27, and the refuting measurement was already mine

I aimed both planner boots with a load draw of **956 MiB** (N49's
`at-rest 1932 − min 976`). That figure came from a **deep-prefill A/B** boot.
I applied it to a **22-minute mixed soak**. Measured, same rank, same law:

| load state | rank0 draw below at-rest |
|---|---:|
| deep-prefill A/B | 956 MiB |
| mixed soak, planner cut | **1989 MiB** |
| mixed soak, ship config | **3148 MiB** |

Law 27 says two minima from differently-loaded boots are not a derivative.
The corollary I had to learn separately: **one measured draw is not
transferable to another load state either.** Worse, I had measured the 3148
MYSELF four hours earlier in this shift's own ship window and did not check
my estimate against it before spending two boots. Recorded as law 31.

### 1b. My mid-shift per-linear-layer term was a fit, and the census refuted it

I fitted 158 MiB/linear-layer from two boots. It predicted a third boot to
88 MiB — and it was still the wrong mechanism. The census then MEASURED the
term directly at **51.2 MiB/linear-layer**, on two cuts and both card types,
which is N49's original figure. Predictive success over a narrow range is not
mechanism (law 29).

### 1c. My own wiring priced the 5090 as a 3080

`_pp_cut_card_rates` looked each stage's card up by using `--rank-gpu-id` as
an NVML index. On this rig `--rank-gpu-id 0,1,2` puts stage 0 on NVML index
**1**, because `CUDA_VISIBLE_DEVICES` is set by UUID. Found by EXECUTING the
handler against a real census, not by reading it. Fixed by having each rank
record its own card in the census, so the IdentityMap travels in the artifact
(law 32).

### 1d. The unit-test "red" for the gate fix is weak; the numeric red is in a script

The new unit tests fail on the pre-fix module with `AttributeError` (the
fields do not exist), which proves they exercise new API and not that the old
model was wrong by 3000 MiB. **The numeric red is `s50/ledger.py`**, which
prices the same measured boots through both models from identical inputs.
Believe that one. `s50/canfail_prefix.txt` records the weak one honestly.

### 1e. What I did not do

* **No same-boot ship control** (fourth shift). Both A/B arms are compared to
  N48/N49's 98.276 s.
* **`seam_staging_mib` still uncalibrated.** It did not bind — the residency
  verdict refused first — but it is still a zero in a shipping gate.
* **Arm B not run** (fourth shift).
* **The planner-cut window never completed**: rank0 died at 17 of 22 minutes
  (§3b). The corridor verdict below rests on the 17 minutes that ran.

---

## 2. THE TWO BLOCKERS FROM THE BRIEF — BOTH CLOSED

### 2a. The residency gate: was ~3000 MiB off on rank0, now ~19 MiB

New instrument: **`planner/residency_census.py`**, env-gated on
`SGLANG_RESIDENCY_CENSUS`, read-only, byte-identical when off, so it rides
along on corridor-measuring boots. Per rank it reports parameter bytes by
owner, the allocator's pool/graph posts, fragmentation and the driver's
free/total, and writes JSON for reuse.

Both of C38's defects were real and are fixed — role-scoped non-layer weights
(embedding on stage 0, `lm_head` on the last, vision replicated) and
`tp_token_shares` fed the resolved TOKEN vector via the new
`token_shares_from_vector`. Two things C38 got wrong:

* **its numbers cancelling is arithmetically false** (C39). Under-charge 3358
  MiB against an over-charge of at most 1212. What actually hid the error is
  that `fixed_overhead_mib` is **a residual fitted on the ship boot**, and a
  fitted residual absorbs anything constant on the cut it was fitted on. The
  pre-fix gate predicted a held-out boot of the SAME family to 109 MiB while
  being 3040 MiB wrong across the cut boundary. That is the real failure mode
  and it is more general than a cancelling pair (law 28).
* **one more unpriced term:** the attention layer is 364.88 MiB resident, not
  the 325.0 the config formula gives — `attn_output_gate` adds a second
  q-sized projection. 482 MiB over 16 layers, cut-shaped.

Corrected gate against metal, calibrated on one planner boot, predicting the
other (different cut AND different pool):

| boot | predicted rank0 free at rest | measured | error |
|---|---:|---:|---:|
| planner 42,11,11 @271000 | 2017.0 | 1991.7 | **+25.3** |
| gate cut 40,12,12 @280000 | 2676.2 | 2657.7 | **+18.5** |

**Still open, quantified:** the residual is NOT cut-invariant — rank0
measures 3174 MiB at 28 layers, 4419 at 40, 4440 at 42. Flat between
neighbours, 1250 MiB apart across the ship/planner boundary. So the gate must
be calibrated near the cut it judges, and `--pp-solve-cut` warns when the
census is more than 4 layers away.

### 2b. The 48 MiB corridor miss: option (a), it fell out of the residency fix

No margin term was added and **the pool was not reduced**. With residency
honest, the gate's existing transient term already refuses N49's cut:

```
stage 0-42 on rank rank0-5090: needs 31593 MiB (15370 layer weights + 3358
non-layer weights + 1638 recurrent state + 5469 KV + 5758 transient) ...
only 31064 MiB usable after the 1024 MiB corridor -- over by 529 MiB.
```

and instead returns **`40,12,12` / attn `10,3,3` at the full pool 280000**,
306 MiB of runnable headroom: same attention split, two linear layers moved
off the binding card.

---

## 3. METAL

### 3a. The gain survives and grows; the gate's cut is the fastest measured

| arm | cut / attn | pool | flip | depth 179200 | vs control |
|---|---|---:|---|---:|---:|
| A control (ship) | 14,10,8 / 7,5,4 | 280000 | ON | 98.276 s (N48) | — |
| C planner (N49) | 42,11,11 / 10,3,3 | 280000 | ON | 66.072 s | +48.7 % |
| C planner (mine) | 42,11,11 / 10,3,3 | 271000 | ON | 65.854 s | +49.2 % |
| **gate's own cut** | **40,12,12 / 10,3,3** | **280000** | **ON** | **65.207 s** | **+50.7 %** |

5 samples each, spreads 2.11 % and 0.91 %, 0 cache-hit rejects. Note the
model expected `40,12,12` to be 3.6 % SLOWER and metal says 1.0 % faster: the
makespan model does not resolve near-neighbour cuts. It resolved the thing
that mattered — which of the two is fundable.

### 3b. The window: ship CLEAN, planner cut BREACHED then DIED

**Ship config, 22 min: 0 breaches on BOTH instruments** (9654 NVML samples,
minima 1555/2614/1945; 546 seam troughs). 182 flips, 0 abandoned, soak
ok=269 err=0. N49's minima were 1523/2608/1945 — reproduced within 32 MiB on
two consecutive shifts.

**Gate cut, 17 of 22 min. TWO SEPARATE FAILURES:**

1. **Corridor broke.** rank0 MIN **669 MiB**, 2 of 9979 NVML samples; seam
   census agrees independently (`CORRIDOR LAW BROKEN: 2`, trough 668). Both
   at 11:44:06. Cause is §1a: the soak draws ~1989 MiB and the cut leaves
   2658 at rest.
2. **rank0 died silently at 11:48:41** — 4.5 min AFTER the breaches, with
   **1926 MiB free** over the preceding 20 s, mid-decode-batch, no traceback,
   no OOM line anywhere, no kernel OOM record, host RAM 107 GB free. Ranks 1
   and 2 then took barlink's peer-liveness abort
   (`Bar1CollectiveAborted ... flip_tp:0 ... peer rank gone: rank 0`).
   **The barlink abort is the consequence, not the cause, and this is not the
   corridor event** — the rank survived that by four minutes. Recorded as
   C40, #622/#649 family.

159 flips both directions, 0 abandoned, seam machinery inert throughout.

---

## 4. WIRE-OR-GATE: **GATE**

Bootable yes, gain yes (+50.7 %), corridor **no**.

`solve_pp_cut` is now REACHABLE (`--pp-solve-cut <census-dir>`, §5) as the
brief asked, and the default is byte-identical — but the ship config remains
what boots, and **N49's manual `42,11,11` recommendation is WITHDRAWN**: it
breaches at pool 280000, and so does the cut the corrected gate prefers.

### The blockers, in the order to attack them

1. **`transient_mib` is calibrated on the wrong load state.** 1346 MiB
   modelled, ~1989 measured under the shipping soak on this cut, ~3148 on the
   ship config. Every verdict is optimistic by that gap. One window per cut
   measures it. With an honest 2000 MiB the gate refuses `40,12,12` at 280000
   by ~348 MiB and would look one linear layer lighter on rank0
   (`38,13,13`) — that is the next boot, and DO NOT shrink the pool to get
   there.
2. **C40, the silent rank death.** Until it is understood, no planner-family
   cut can be certified on a 20-minute window, because the window is the
   instrument it kills. Cross-reference the #622/#649 strand: this is a fresh
   instance with a complete log and a named pid.
3. **Itemize the residual** (§2a): 1250 MiB of it is cut-shaped and sits in
   the census's graphs/workspaces post. Closing it makes cross-family
   verdicts trustworthy and is pure desk work against the census JSONs.
4. Calibrate `seam_staging_mib`; same-boot ship control; arm B.

---

## 5. WHAT LANDED

* `planner/residency_census.py` — the instrument (§2a).
* `planner/pp_cut.py` — role-scoped non-layer weights, cut-shaped recurrent
  state, `token_shares_from_vector`, the token-vector contract. All new
  fields default to 0.0, so every existing caller and verdict is unchanged.
* `planner/pp_cut_calibration.py` — census -> solver inputs, refusing on a
  partial census instead of defaulting a term to zero.
* `server_args.py` — `--pp-solve-cut`, materializing `--pp-layer-ratio` so
  the existing validation and the hybrid family-census gate still run on the
  solved cut. Explicit `--pp-layer-ratio`/`--pp-stage-ratio` refuse to be
  combined rather than being overruled.
* Register: C39, C40, laws 28-32.
* 26 new tests. Planner directory has **115 pre-existing failures at HEAD**
  (unrelated: `_ServerArgsView` attribute errors); my changes move it to 115,
  i.e. zero new failures — measured by running the directory against a
  pristine `server_args.py`.

## 6. STATE ON EXIT

Serving UP on 30030, SHIP config, from `/spinning/wt-631-routea/python`.
Router 30099 never touched (401 on every check). All three cards were at
3 MiB between every boot. I stopped serving twice and brought it back; nobody
owes a restore.
