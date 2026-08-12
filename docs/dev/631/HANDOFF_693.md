# HANDOFF 693 — the #485 wall was two bugs, and the cut runs under the flip

Successor 49. Branch `feat/route-a-631` at `10213cd58e`; `integration/r2` and
`feat/pp-family-cut-485-rebased` pushed at the same commit. Evidence:
`/spinning/evidence-631/s49/` (`RESULTS.md` is the verdict, `CONFOUND.md` the
experiment that reframed C34, `PREDICTION_confound.md` the desk call recorded
before the boot that decided it).

---

## 1. ERRORS FIRST

### 1a. My recorded prediction was wrong, and it was wrong in an instructive way

Before the confound boot I wrote down, in `PREDICTION_confound.md`, that the
seam wedge would follow the CUT. I derived it from `_staging_bytes`
(`incoming + max(outgoing, local)` over the widest wave), computed 10 layer-row
units for arm C's rank0 against 8 for arm A, and noted that the same formula
predicts the refusing rank of BOTH of N48's wedges — rank0 for arm C, rank2 for
arm D. It does. Metal says the cause is the TOKEN VECTOR: same cut, ship
vector, 185 abandons became 0.

The derivation was not wrong about the mechanism. It was wrong about which
input drives it — the row counts in that formula come from the ARENA, and the
arena is set by the token vector. **A model that predicts the right rank for
the right mechanism can still be naming the wrong cause when the causes were
never separated.** Recording the prediction first is what made this legible
instead of a quiet retrofit; do the same.

### 1b. I used `pkill -f` and killed my own shell

Standing law says kill by PID after py-spy, never `pkill -f`. I ran
`pkill -f "corridor_sample.sh"` to stop a sampler, and the pattern matched the
bash process running my own command (exit 144). No damage — router still 401,
heartbeat alive, all three cards at 3 MiB — but the law exists precisely
because the blast radius is not visible in the pattern you type. Samplers
expire on their own timeout; there was no reason to kill it at all.

### 1c. I dumped a 300-line log grep into my own context

Counting abandon lines per rank, I piped `grep -o "^\[[^]]*\]" | sort | uniq -c`
without collapsing the timestamps, so 338 near-identical lines came back. The
two facts I needed (which rank refuses; the retry cadence is a fixed 3 s) fit
in two lines. Aggregate before printing.

### 1d. My first regression tests could not fail

The `KvRowCap` fix was written as two changes: the semantic one and a
`torch.unique` "belt". With the belt in place the three new tests passed
whether or not the semantic fix existed — I only found that because I ran the
can-fail probe against the semantic half alone. The belt is REMOVED (C37).
**Probe each candidate fix separately; a patch containing a masking change and
a real change passes as a unit and teaches nothing.**

### 1e. A near-miss worth pinning: `7,5,4` and `14,10,8` are the same vector

The captured ship env sets `SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8`, while N48's
arm scripts and this shift's confound boot both pass `7,5,4` and call it "the
ship vector". They are the SAME ratio: `resolve_cp_token_split`
(`distributed/utils.py`) gcd-reduces the vector, and `gcd(14,10,8) = 2`, so
`14,10,8 -> 7,5,4` exactly. I checked this against the live process's
`/proc/<pid>/environ` rather than assuming it, because if they had differed
the confound boot would have had two variables in it and §2's conclusion would
be void. They do not differ. Do not "fix" one to match the other.

### 1f. What I did not do

* **No same-boot control.** The headline compares my arm-C boot against N48's
  arm-A boot. Admissible only because N48 measured that control's cross-boot
  reproducibility at 0.09 %, but it is the weakest link in the table and one
  more boot would remove the caveat entirely.
* **Arm B was not run** (again). Third shift running.
* **`seam_staging_mib` is not calibrated.** The term is in the model; the
  number is not measured. See §4.

---

## 2. THE RESULT

**The planner cut runs under `--enable-phase-flip` and keeps its win.**

| arm | cut / attn | flip | depth 179200 | vs control |
|---|---|---|---:|---:|
| A control (ship) | 14,10,8 / 7,5,4 | **ON** | 98.276 s (N48) | — |
| **C planner** | **42,11,11 / 10,3,3** | **ON** | **66.072 s** | **+48.7 %** |
| A control | ship | off | 95.436 s (N48) | — |
| C planner | planner | off | 63.246 s (N48) | +50.9 % |

Pool 280000 throughout, spec+flip on for the top pair. 5 samples, spread
0.59 %, 0 cache-hit rejects, 42 flips, 0 abandoned, five deep prefills
survived — the operation that killed both earlier arm-C boots.

**+48.7 % is the shipping number.** The flip costs the cut ~2 points, about
what it costs the control (3.0 %).

### C34's wall was two independent bugs, both now fixed (`6c7e8a1411`)

**(a) The seam wedge follows the token vector.** N48 set
`SGLANG_UNEVEN_TOKEN_VECTOR` per arm to the arm's attention split, so cut and
arena moved together in every arm. Holding it at ship `7,5,4`: 185 abandons ->
**0**, health 200. The cut was never locked out of the flip.

**(b) `KvRowCap` double-booked withheld ids on `clear()`.** `withheld ==
2 x (total - available)` EXACTLY, on two pools and two token vectors
(12783/25566 and 81640/163280). `_apply` accumulates and was wired as the
on-clear hook as well as the on-free hook; `clear()` rebuilds
`arange(1, size+1)` so the ids are taken twice. Only a config that ENGAGES the
cap sees it, which needs a corridor deficit — the ship cut never does, and
doubling zero is invisible. Worse half: `release()` returns the doubled tensor
to the free list, handing one KV row to two requests.

### And the retry that killed the instance is bounded now

Backoff (0,1,3,7,... arm requests declined, clamped) plus a cap
(`SGLANG_SEAM_ABANDON_CAP`, default 8) ending in a terminal verdict that
installs a blocking guard, so the instance stays in its phase and keeps
serving instead of dying silently. The clock is ARM REQUESTS — one broadcast
per arm, so every rank counts the same sequence; a wall clock is rank-local
and a round count is not uniform within one arm. Decided in `arm()`, before
anything is armed, so no collective can observe a disagreement. Margin delays
are exempt (their own C20 budget terminates them), so it is inert on the ship
path.

---

## 3. WHY IT IS STILL GATED

The brief's condition was bootable + gain holds + window clean. Two of three:

* bootable — **yes**, pool 280000, 42 clean flips
* gain — **yes**, +48.7 %
* corridor — **NO**

| boot | pool | gpu1 min (rank0/5090) | breaches |
|---|---:|---:|---:|
| N48 arm A control | 280000 | 7212 | 0 |
| planner cut | 340000 | **608** | 135 (then `cuMemCreate` OOM) |
| planner cut | 280000 | **976** | **6 of 3568** |

The cut puts 10 of 16 attention layers on rank0 against the ship's 7, ~43 %
more KV on that card. At 280000 it misses the 1024 law by **48 MiB for 6
samples**. Close is not held.

`solve_pp_cut` stays unwired for two now-quantified reasons: `seam_staging_mib`
has no calibrated value (default 0 = the gate that certified arm C), and the
RESIDENCY model itself mispredicts rank0 — it called arm C feasible with
2617 MiB spare against a measured 606.

**What is shippable today** is the cut as a MANUAL configuration:
`--pp-stage-ratio 42,11,11 --pp-attn-stage-ratio 10,3,3` at
`--max-total-tokens 280000`, ship token vector — once it holds the corridor.

**How much pool to give back, and do NOT use the theoretical slope.** rank0
holds 10 of 16 attention layers, so KV there "should" cost
`10 x 2048 B/token = 0.0195 MiB/token`, which says 48 MiB is bought by ~2500
tokens. The two measured boots say otherwise:

| pool | rank0 min free |
|---:|---:|
| 340000 | 608 MiB |
| 280000 | 976 MiB |

That is **0.0061 MiB/token**, 3.2x SHALLOWER than the layer-count model — the
same direction of error as the residency misprediction in §3, and probably the
same cause. Taking the measured slope, reaching ~1100 MiB of margin needs
about **20000 tokens off, i.e. pool ~260000**, not 2500.

Two points, two separate boots, and minima read a load state (C7), so treat
this as a RANKER that aims the next boot rather than a calibration. But aim it
at the measurement: the theoretical slope would have under-shot by 8x and cost
a boot.

---

## 4. NEXT SHIFT, IN ORDER

1. **One boot at pool ~260000 to close the corridor gap** (see §3 for why
   260000 and not 277000 — use the MEASURED slope, not the layer-count one),
   then re-measure 179200. If it holds the law with the gain intact, the
   manual configuration is shippable and the wire-or-gate question becomes
   only about the solver. Cheapest remaining item by a wide margin. Fold the
   §3 slope check into it: a third (pool, min-free) point turns a two-point
   ranker into a usable line.
2. **The same-boot control** (§1e): arm A, flip on, pool 280000, in one boot
   with arm C. Removes the only caveat on the +48.7 %.
3. **Calibrate `seam_staging_mib`.** Two demands are measured (4881 MiB at
   attn 10/16 pool 340000; 4343 MiB at attn 6/16 pool 280000) and they do not
   separate the shape — and the confound says the demand follows the arena,
   not the attention split. Get a third and fourth point by varying the pool
   at a FIXED cut, which isolates the row-count axis. Do not fit two points.
4. **Why the residency model mispredicts rank0** (§3). It is a residency
   error, not a transient one, and it is unexplained.
5. **Exercise the seam cap on metal.** It is unit-tested (19 tests) and has
   never fired on a real boot, because the confound removed the configuration
   that would trigger it. Boot arm C with the 10,3,3 token vector deliberately
   and confirm the verdict fires, the instance stays up, and /health keeps
   answering — that is the wedge that started this and it should now end in a
   log line instead of a corpse.
6. **Arm B**, still not run.
7. **`grow_size` re-admits ids above the cap without notifying the listener**
   (`mem_cache/allocator/base.py`) — same class of hole as the clear bug, no
   symptom observed yet.
