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
RESIDENCY model itself mispredicts rank0 — **and that second one is now fully
diagnosed, see §3a.**

### 3a. The residency misprediction is SOLVED, from logs already on disk

The model over-predicts rank0's free VRAM by **~3900 MiB** (not the ~2000 I
first estimated; N48's "2617 MiB spare" is the optimistic end). Four terms,
ranked, all reconstructed to within 8 MiB:

1. **Missing non-layer weights on rank0, ~3760 MiB, CONSTANT — the answer.**
   `_price_stage` prices a per-layer census of the transformer layers only.
   The checkpoint is `Qwen3_5ForConditionalGeneration`, a VL model quantized
   INT8-W8A8 with `lm_head`, the whole visual tower and the embeddings in the
   quantizer's `ignore` list — i.e. **bf16 and completely unpriced**.
   `embed_tokens` = vocab 248320 x hidden 5120 x 2 B = **2425 MiB** exactly
   (untied: `tie_word_embeddings: false`), on rank0 only; `lm_head` the same
   2425 MiB on rank2 only; plus ~1096 MiB of replicated bf16 vision tower and
   loader constant on every rank. Fitting the six measured `Load weight end`
   values across two cuts x three ranks closes to the megabyte.
2. **`tp_token_shares` is fed the WRONG VECTOR.** `pp_cut` was given
   `--phase-flip-tp-vector 32,16,16` = 0.5/0.25/0.25, but the arena is sized
   by `parse_flip_token_vector` reading `SGLANG_UNEVEN_TOKEN_VECTOR` =
   `14,10,8` = 0.4375/0.3125/0.25. Worth ∓547-664 MiB, and the sign is what
   makes this vicious: on the SHIP cut `max(7, 8)=8` overcharges KV by exactly
   as much as the weights undercharge, so **the two errors cancel and the ship
   cut looks fine**. On the planner cut `n_attn=10 > 8`, the cancellation
   disappears and the full shortfall shows naked. That is the whole reason
   this went unnoticed.
3. **The PP-stage mamba/GDN pool is inside `fixed_overhead` and IS
   cut-shaped** — `51.2 MiB x n_linear`, so +563 MiB going ship -> planner on
   rank0. **This partly refutes C35.** CALIBRATION.md's "mamba/GDN pool
   1229/614/614 = exactly the TP vector" was the **TP-stack pool only**; there
   are TWO pools, and the PP-stage one was never itemized, so it vanished into
   the residual and made the residual look cut-invariant. C35's conclusion
   about the CARD (the two 3080s are interchangeable) stands; its conclusion
   that `fixed_overhead_mib` is cut-invariant does not.
4. **Draft KV scales with pool** inside `fixed_overhead` (266 MiB @280k ->
   328 @340k), so that term is not pool-invariant either.

The ledger reproduces NVML free to **within 64 MiB on all three boots** with a
single 577 MiB CUDA-context constant. That matters for §3's headline: the
planner cut at pool 340000 was **already 672 MiB free AT REST, below the
corridor before a single token was served** — it was never a load problem.

**A corrected model gets all three verdicts right**, from terms that are
already measured: planner @340k infeasible by ~900 MiB (metal: OOM); planner
@280k 321 MiB headroom, refuse or shave (metal: 6 samples 48 MiB under); ship
@280k 7633 MiB headroom (metal: 7212 min free). **This is the wiring
blocker's fix and it needs no new measurement.**

**What is shippable today** is the cut as a MANUAL configuration:
`--pp-stage-ratio 42,11,11 --pp-attn-stage-ratio 10,3,3` at
`--max-total-tokens 280000`, ship token vector — once it holds the corridor.

**How much pool to give back. I got this wrong once mid-shift; read the
retraction.** I first compared rank0's LOAD MINIMUM at pool 340000 (608 MiB)
against its LOAD MINIMUM at 280000 (976 MiB), got 0.0061 MiB/token, called the
layer-count slope "3.2x too steep" and told the next shift to aim at pool
260000. **That is wrong and it is a textbook C7 error**: those two minima read
two different load states, so their difference is not a pool derivative at
all. Corrected against the at-rest ledger (§3a):

| pool | rank0 free AT REST | rank0 min under load |
|---:|---:|---:|
| 280000 | 1932 | 976 |
| 340000 | 672 | 608 |

At rest the slope is `1260 MiB / 60000 tok = 0.021 MiB/token`, which is the
layer-count figure `10 x 2048 B = 0.0195` to within the ledger's own noise.
**The theoretical slope was right and my "measurement" was two states
subtracted from each other.**

So: the load draws rank0 about **956 MiB below its at-rest level** at pool
280000 (1932 -> 976). To land the minimum near 1100 MiB the at-rest level
needs ~124 MiB more, i.e. **~6400 tokens off: pool ~273500**. Not 260000, and
not the 277000 I first guessed either.

---

## 4. NEXT SHIFT, IN ORDER

1. **Fix the residency model from §3a — no GPU needed, all four terms are
   already measured.** Price the non-layer weights (embed on stage 0, lm_head
   on the last stage, the replicated bf16 vision tower), feed
   `tp_token_shares` from `SGLANG_UNEVEN_TOKEN_VECTOR` and not from the flip
   WEIGHT vector, and move the PP-stage mamba pool out of `fixed_overhead`
   into a `51.2 MiB x n_linear` term. The corrected model already gets all
   three measured boots right. This is the highest-value item on the ticket
   and it is desk work.
2. **One boot at pool ~273500 to close the corridor gap** (§3: at-rest slope
   0.021 MiB/token, load draws ~956 MiB below at-rest; do NOT use the
   load-minimum slope I first published — see the retraction). Then
   re-measure 179200. If it holds the law with the gain intact, the manual
   configuration is shippable. Fold in the §3a confirming probe: one
   `named_parameters()` walk on rank0 after `Load weight end`, grouped by
   top-level owner, costs microseconds and separates "per-rank vision
   constant" from "stage-0 embedding" — which decides whether the fix is one
   rank0 constant or three role-scoped ones.
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
