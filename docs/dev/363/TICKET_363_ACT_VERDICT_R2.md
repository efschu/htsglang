# TICKET #363 — ACT WINDOW VERDICT R2 (R14)

Window `363-act-r2`, 2026-08-14, held from 15:34Z. Worktree
`/spinning/wt-merge-r14` (branch `merge/r14-batch`, frozen base `4acdaff936` =
`origin/feat/route-a-631` = `origin/integration/r2`),
`PYTHONPATH=/spinning/wt-merge-r14/python`. Evidence
`/spinning/evidence-363-act-r2/`. Runsheet executed and repaired in flight:
`docs/dev/363/RUNSHEET_363_ACT_WINDOW.md`.

**The headline.** The R13 window failed honestly with a decision rule that
never computed a signal, and named two defects behind it. This window fixed
those two, and in doing so found that **`0 flip targets` had four causes, not
two** — each one invisible until the one before it moved. Four are now fixed
and the stage table's refusal has changed from *"the planner could not solve"*
to *"the planner solved 1 stage and it carries no measurement"*, which is the
refusal the measurement pass exists to clear.

---

## 1. THE DEFECT CHAIN, AND WHY IT LOOKED LIKE ONE DEFECT

Every one of these was found by executing a step, and every one was hidden by
the one before it. That is the structural finding of this window: `0 flip
targets` was never a single fault, it was a queue of them behind one symptom.

| # | defect | how it hid | state |
|---|---|---|---|
| 1 | rate fingerprint could not date a capacity-disambiguated profile | — | fixed in R13 (`42484da800`) |
| 2 | reshard cutover allocates its transient buffers unguarded — OOM + corridor breach | crashed the server before anything downstream ran | **FIXED** `4886e6b963` |
| 3 | card probe unreachable from a narrowed `CUDA_VISIBLE_DEVICES` | feed died at "no card probe on disk" | **FIXED** `837a38c0bc` |
| 4 | solver never surfaced its own per-rank KV split | invisible while (3) failed earlier | **FIXED** `f3682cbafa` |
| 5 | booted stage's weight vector read from the FLAG, not resolved state | invisible while (4) failed earlier | **FOUND, not fixed** — §5 |
| 6 | hardware profile unreachable from a narrowed view | silent: ranks priced fp8 on the dense bf16 lane | **FIXED** `be8a7458b8` |

---

## 2. DEFECT 2 — THE ACTUATOR NO LONGER TAKES THE SERVER DOWN

R13 reproduced this twice, deterministically: `torch.empty` in `_exchange`
asked for 616 MiB with 550 MiB free under load, and 758 MiB with 256 MiB free
DRAINED — worse when idle, not better — and in both runs the corridor law
broke **seven seconds before** the traceback. That direction is the whole
finding: the allocation drove free memory under the floor; free memory did not
drift into the allocation.

`fit_check` existed and passed both times. It answers whether the TARGET
VECTOR fits the backed pool. Nothing priced the TRANSIENT.

**The guard is in the group-wide MIN reduction, not at the allocation site.**
`headroom` joins `armed` and `ready` in `on_round`'s reduction, so every rank
prices the move and either all move or **none allocates a byte**. A guard at
`torch.empty` would let the affording ranks enter `_exchange` and post sends
while the short one aborted — "the #94/#194/#259 hang, or worse, a silent
mixed-ownership pool", in the module's own words. A rank that is not ready
reports the neutral `1` rather than a number it never computed; safe because
the refusal is only consulted once `min(ready) == 1`, i.e. once every rank did
evaluate.

Arithmetic: `free − corridor_floor − peak >= 0`, floor subtracted **first**,
because the corridor is the user's reserve and was never memory the reshard
could spend. `peak` is read off the PACK and EXCHANGE phases in their order
and named term by term — `staged`, `packed`, `largest_peer_pack`, `recv` — so
a refusal shows its work. The plan is built once at the boundary and handed to
`_execute`: a guard that prices a different move than the one it admits is not
a guard. Free memory is NVML FREE (not total-minus-used — the ~424/518 MiB
carve-out is invisible to the subtraction), and the card is resolved by UUID,
never index.

Refusal is clean and terminal: disarm, name it, stay on the incumbent vector.
Disarm rather than hold because the drained run needed MORE memory than the
loaded one — not a transient a retry loop grows out of. Fail-closed if free
memory cannot be read: refusing costs a flip, guessing cost the server.

**Cross-tree falsifier**, one script, both columns
(`evidence-363-act-r2/falsifier_defect2.py`, scenario "rank 1 free == corridor
floor"):

| | BASE `4acdaff936` | TIP R14 |
|---|---|---|
| exchange entered by ranks | `[0, 1, 2]` | `[]` |
| exchange entry count | 3 | **0** |
| ranks that cut over | `[0, 1, 2]` | `[]` |
| `refused_headroom` per rank | `[null, null, null]` | **`[1, 1, 1]`** |
| exceptions / hung threads | none / 0 | none / 0 |
| verdict | **DEFECT REPRODUCED** | **FIXED** |

22 red-first tests. The desync falsifier is the arm that distinguishes this
design from the obvious one: one rank short → all refuse, `_exchange` never
entered, no cutover, pool byte-identical, no hang, and it holds whichever rank
is the short one. The can-admit arms are there so no refusal above is vacuous.

---

## 3. DEFECT 3 — NO RANK COULD FIND THE CARD PROBE

`#513` replaced a newest-by-mtime lookup with a keyed one so a probe of another
card set could not become the rig's profile. The key is right; the COMPARISON
was not. It matched card sets for EQUALITY — correct for the process that
WRITES a probe, wrong for every process that reads one, because a scheduler
rank runs under a narrowed device view.

Replaced with **containment** (`visible ⊆ probe`, same driver), matched on UUID
throughout and never on an index — a UUID survives narrowing unchanged, an
index is precisely what narrowing renumbers. Containment is directional, so
#513's protection still points where #513 aimed it: a two-card probe still
cannot serve a three-card view. The WHOLE probe is returned, not the caller's
slice, because `key_solver.rates_from_probe` indexes cards by `cuda_index` in
the full `--rank-gpu-id` space.

**On metal, before and after** (`falsifier_defect3.{base,tip}.txt`):

| view | BASE | TIP R14 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=0,1,2` | FOUND (3 cards) | FOUND (3 cards) |
| `CUDA_VISIBLE_DEVICES=1` | **NONE** | **FOUND (3 cards)** |
| `CUDA_VISIBLE_DEVICES=0` | **NONE** | **FOUND (3 cards)** |
| `CUDA_VISIBLE_DEVICES=2` | **NONE** | **FOUND (3 cards)** |

And in a real three-rank boot, all three ranks log it:

```
card probe: this process sees 1 card(s); the matching probe describes 3,
including all of them.
```

12 red-first tests, 7 red before the change; #513's own 23 still pass.

---

## 4. DEFECT 4 — THE SOLVER NEVER HANDED OVER ITS OWN SPLIT

Fixing defect 3 let the feed get further, and it hit the next wall on the
first boot after — logged by all three ranks:

```
PlannerFeedUnavailable("the solver API returns no per-rank KV vector
(Stage.kv_token_vector). key_solver.capacity() computes it as cap['p'] but
key_solver_payload does not surface it; exposing it is the remaining wiring
step. A split derived here would be a fabricated reshard target")
```

`capacity()` computed the per-rank split and threw it away; every caller read
only the SUM. It is now surfaced unmodified as `per_rank_kv_tokens` —
absolute tokens, the solver's own numbers.

**Resolved, never derived.** `Stage.kv_token_vector` lives in the #297 reshard
RATIO space, so the absolute split has to be resolved into it. The feed matches
the solver's PROPORTIONS against the vectors the OPERATOR declared, and the
answer is always one of those or nothing. Rounding an absolute split into a
small ratio would produce exactly what the old refusal warned about: a vector
nothing has backed with pool rows that nonetheless reads as the solver's
answer. Shares, not raw counts, because two vectors differing by a common
factor describe the same layout. Tolerance 2 % of pool share, carrying its own
argument: one unit of a 23-unit vector is 4.3 % of the pool, so a tighter
value would refuse every vector an operator could declare.

When nothing matches, the feed refuses and NAMES what it wanted — which is how
this window found its way to a working boot:

```
the solver wants the per-rank KV split [12533, 147633, 190697] tokens =
3.6%, 42.1%, 54.4%, and no declared vector is within 2% of it
([2, 11, 10] off by 10.9%; [3, 10, 10] off by 10.9%; [30, 17, 17] off by 43.3%)
```

16 red-first tests, including the refusal-not-snap arm.

---

## 5. DEFECT 5 — THE BOOTED STAGE REPORTS WEIGHTS IT IS NOT RUNNING

**Found, deliberately not fixed in this window.** `reachability()` checks the
weight vector FIRST, and `_booted_stage` reads it from
`server_args.rank_mlp_ratio` — the FLAG. Under `--rank-tp-ratio
auto-performance` that flag is `None`, so the booted stage reports
`weights=auto` while the server is in fact running a specific resolved
partition. Every planner candidate carries a concrete `rank_mlp_ratio`, so on
**any auto-ratio boot every candidate is `REACH_NO_WEIGHT_MOVER`** regardless
of its KV vector.

It is the same class as defects 3 and 6 — reading a declared INPUT where the
resolved runtime STATE is what matters — and the resolved value is already
available as `distributed.utils.get_tp_partition_ratios("mlp")`.

Not fixed here because the comparison needs care rather than a one-line swap:
the candidate reports `_ratio_of(units)` (a gcd-reduced ratio of MLP UNITS)
while the installed plan is a RATIO from which units are derived by a
quantizing partition, so the two must be canonicalised into the same space
before they are compared — otherwise the fix trades a false mismatch for a
false match, which is worse. That is a slice with its own tests.

**It does not block, because the intended procedure is to boot the matching
arm** — `REACH_NO_WEIGHT_MOVER`'s own text says "Boot the other arm to use
it", and §12 puts the weight-cut axis out of scope (#354/#357). This window
booted with an explicit `--rank-mlp-ratio`, which makes the flag concrete and
the comparison correct.

---

## 6. DEFECT 6 — THE HARDWARE PROFILE, AND THE QUIET FAILURE

Defect 3's twin, in `uneven_perf._load_profile`, with an exact-equality check
on the same inventory key:

```
CUDA_VISIBLE_DEVICES=0,1,2 -> hardware profile PRESENT
CUDA_VISIBLE_DEVICES=1     -> hardware profile ABSENT
```

This one is quieter than the card probe's miss and worse for it. With the
profile absent, `rates_from_probe` loses the v3 `gemm_lanes` map and the rank
prices an fp8 / int8 / W4A16 checkpoint on the **dense bf16 probe**
(#324/#359). Nothing fails. The rank simply solves a different problem than
the dashboard does, from the same disk, and neither says so. Measured before
the fix, same payload, same rig: a full view answered `enc` with mlp
`[135, 0, 1]`; a rank answered `[1, 0, 0]`.

Same containment fix, **read path only** — `_load_profile`,
`profile_cache_path` and the probing/write path keep exact rig identity,
because a profile is WRITTEN for the set it measured. After the fix all
narrowed views report PRESENT with 3 gpus described, and full-view and
narrowed-view solves return identical candidates for `enc`, `dec` and `maxkv`.

### 6.1 A measurement-discipline note that cost half an hour

Reproducing a rank's solve at the desk disagreed with the rank until the
desk's ENVIRONMENT matched the boot's: `SGLANG_UNEVEN_DCP=1`,
`SGLANG_UNEVEN_DCP_WEIGHTED=1`, `SGLANG_MAMBA_SSM_DTYPE=bfloat16`. Without
them the mamba state pool is sized differently and **every** candidate comes
back infeasible. With them the desk reproduces the ranks' numbers exactly.
A desk reproduction that omits the serving env is not a reproduction, and it
fails in the direction that looks like a real finding.

---

## 7. THE SOLVER'S KEYS ON THIS RIG

Reproduced identically from a full view and from a narrowed rank view after
defect 6, budgets `[27107, 16680, 16680]`, `Qwen3.6-27B-FP8`, fp8_e4m3 KV,
NEXTN spec, `max_running_requests 16`:

| goal | regime | `rank_mlp_ratio` | per-rank KV tokens | shares |
|---|---|---|---|---|
| `enc` | prefill_heavy | `[135, 0, 1]` | `[12533, 147633, 190697]` | 3.6 / 42.1 / 54.4 |
| `dec` | decode_heavy | `[30, 17, 21]` | `[207139, 60849, 82875]` | 59.0 / 17.3 / 23.6 |
| `maxkv` | kv_pressure | `[94, 13, 29]` | `[117726, 116075, 117062]` | 33.6 / 33.1 / 33.4 |

All three sum to 350863 tokens, which is the cheap check that they are
redistributions of one pool rather than three different pools.

---

## 8. THE MILESTONE: THE FEED IS BOUND AND THE PLANNER SOLVES

Boot `B0c`, `--rank-mlp-ratio 94,13,29 --rank-kv-ratio 30,17,17
--kv-reshard-vectors '30,17,17;1,1,1'` — the `maxkv` arm, so the candidate's
weight vector equals the booted one and only the KV vector differs:

```
REGIME-OBSERVE planner feed: solver split [117726, 116075, 117062] tokens =
33.6%, 33.1%, 33.4% resolved to declared vector [1, 1, 1]
(max share deviation 0.25%)
```

and the stage table's refusal **changed**:

```
stage table refused (#578): the planner solved 1 stage(s) -- planner:maxkv --
but they carry no measurement. ... This is NOT the old 'unfed' state: the feed
is bound and working; what is missing is the measurement pass.
```

Before this window, on every boot for three windows running, that line read
`1 stage(s), 1 reachable at runtime, 0 flip target(s)` with
`PlannerFeedUnavailable` beside it. **The planner now solves a stage on
metal**, and the only thing between it and a flip target is the measurement
the runsheet's B1/B2/B3 produce.

---

## 9. THE MEASUREMENT RUN

Boots B1, B2, B3, B5 all carried `--tp-size 3 --rank-gpu-id 0,1,2
--rank-mlp-ratio 94,13,29 --rank-kv-ratio 30,17,17 --rank-auto-reserve-mib
5500,3800,3800 --kv-cache-dtype fp8_e4m3 --context-length 32768
--max-running-requests 16 --speculative-algorithm NEXTN ... --kv-reshard-vectors
'30,17,17;1,1,1'`, port 30041. Driver identical on every measurement arm:
`--repeats 6 --burst 16 --burst-tokens 6000 --drain 12 --drain-tokens 900
--mixed 8 --idle-s 25`. 216 requests per arm, **0 failed**.

### 9.1 The A-vs-A floor is 0.03 %

```
A-vs-A    floor = 0.03 %  (1243.317 vs 1242.893 ms)
```

n=58 and n=56 `prefill_heavy` boundaries after `--warmup 20`, both clearing
`--min-samples 30` (B1 produced 76, B2 78). R13's floor on the same driver was
**1.48 %**. The difference is not luck: R13's arms booted
`--rank-tp-ratio auto-performance`, so the resolved plan was free to differ
between them; these arms pin `--rank-mlp-ratio` and `--rank-kv-ratio` and are
therefore identical by construction, down to an identical 8-token probe
request sent to each before its driver. Well below the 5 % trigger, so
`enter_margin_pct` stays at 5.0 and nothing was re-tuned.

### 9.2 The guard fired under load, and the server stayed up

B3 armed `(30,17,17) -> (1,1,1)` at 16:45:54 under the driver. At round
**175576** all three ranks refused, together:

| rank free | corridor floor | transient needed | margin |
|---|---|---|---|
| 3626.4 MiB | 1024.0 | 2308.8 (staged 594.8 + packed 594.8 + pack-peak 297.6 + recv 821.6) | **+293.5** |
| 4470.4 MiB | 1024.0 | 2305.1 (592.9 + 592.9 + 296.6 + 822.7) | **+1141.3** |
| 2293.7 MiB | 1024.0 | 3220.3 (1050.5 + 1050.5 + 525.3 + 593.9) | **−1950.6** |

Group MIN **−1950.6 MiB → REFUSED**, and this is the sentence the design was
built for: **two of the three ranks could afford the move and would have
allocated under a per-rank guard.** That is the desync the reduction prevents,
observed on metal rather than argued from the module docstring. The server
stayed up, the corridor never moved, and the request stayed on the incumbent
layout.

The transient here is 2.3–3.2 GiB because `30,17,17 -> 1,1,1` redistributes
roughly half the pool. R13's 616 / 758 MiB crash was the small end of the
scale, which is worth stating plainly: the unguarded actuator had far more
room to be wrong than the two reproductions showed.

### 9.3 Drained is WORSE, and now there is a reason

Retried on a drained pool per RUNSHEET §5 rule 4 — round **251296**, margins
**+517.7 / −329.0 / −2815.7 MiB**, needing 2928 / 2931 / 4085 MiB against the
loaded run's 2308 / 2305 / 3220. R13 saw the same inversion (758 drained vs
616 loaded) and could only record it. The mechanism is the radix cache: with
no traffic nothing evicts it, so MORE rows are live and MORE rows must move.
"Drain it first" is the wrong instinct for this actuator.

### 9.4 The guard admits — the can-admit arm, on metal

After `POST /flush_cache` the live set is empty and the move commits:

```
KV-RESHARD DONE (30, 17, 17) -> (1, 1, 1) (epoch 1) in 1.7 ms: 0 live slots,
0 local row moves, sent 0 rows / 0.00 MiB, received 0 rows / 0.00 MiB
```

Nine `DONE` lines over three flips (three ranks each), 1.7–3.2 ms. **Every one
is an EMPTY-POOL LOWER BOUND** — `0 live slots` — and RUNSHEET §4.4's warning
that an empty-pool cost is not the cost the controller pays stands in full.
The value of these samples is not the number; it is that the guard ADMITS when
the transient is affordable, so §9.2's refusal is a verdict and not a constant.

### 9.5 The measurement pass, and the deadlock it exposed

The pass refused:

```
refused: no boundary carries an ms/round split for regime 'prefill_heavy'
in rounds [None, 174408]. Boot with --regime-stage-clock ...
```

B3 **was** booted with `--regime-stage-clock` and carried 152 `prefill_heavy`
verdict rows. Every one had `ms_decision: None`. That is defect 7 (§1), fixed
this window — and the metal smoke B5 then showed the fix is **necessary but
not sufficient**: `ms_decision` is still absent, now for defect 8's reason.

### 9.6 Defect 8 — where the chain actually ends

`scheduler.py:4969` supplies the split, and its own comment states the
condition: `rank_compute_ms` / `rank_wait_ms` are "**None on a graph-covered
forward**". On this rig every forward is graph-covered, so `_ms_split_n` never
leaves 0, `pack_ms_sample` packs the sentinel, and `_intra_phase_decide`
abstains before any table question is asked.

**This is why `signal_pct` is still not computed on metal, and it is the last
link.** It is also a genuine design tension rather than an oversight: the
project's own standing rule is to validate WITH graphs and speculation, and
this axis needs a non-graph forward to see its two terms. The fix is a
decision — instrument inside the replay, take the split from the device timer,
or accept a graph-free measurement arm — and it belongs to whoever owns the
ms-axis slice.

### 9.7 The six criteria

| ID | criterion | result |
|---|---|---|
| A1 | `stage_clock_proposals > 0` and `actuations > 0` | **FAIL** — 0 and 0 |
| A2 | flips over the window <= 4 | PASS, vacuously |
| A3 | ms/round in SHIFT beats the control, in the wait term | **UNANSWERABLE** — no split exists to compare (§9.6) |
| A4 | zero corridor samples below 1024 MiB, every boot | **PASS — all six boots, B3 included** |
| A5 | `desyncs == 0`, summary present | **PASS** — summaries on all 3 ranks of every arm |
| A6 | every `ms_decision` carries the rule | **VACUOUS** — no `ms_decision` rows (§9.6) |

**The honest verdict: the decision rule was again NOT put on metal, and again
this is not "act correctly refused to flip".** Nothing was evaluated. What
changed is that the reason is now known to the end of the chain instead of
being one unexplained symptom: four of the seven links were unknown when this
window opened, and A4 moved from FAIL to PASS through the exact operation that
used to break it.

---

## 10. CORRIDOR

Instrument's own can-fail arm first — `corridor_report.py --smoke` **3/3**:
clean series PASSES, a planted 900 MiB sample FAILS, an all-999 series FAILS
(the string-compare trap that produced a false reassurance in an earlier
window).

100 ms sampling, NVML FREE, every boot:

| boot | samples/card | gpu0 min | gpu1 min | gpu2 min | verdict |
|---|---|---|---|---|---|
| B0 | 2373 | 1883 | 7786 | 3271 | PASS |
| B0b | 4480 | 1883 | 7786 | 3271 | PASS |
| B1 | 10806 | 4449 | 2264 | 3609 | PASS |
| B2 | 10488 | 4471 | 2294 | 3627 | PASS |
| **B3** | **22615** | **4399** | **2200** | **3569** | **PASS** |
| B5 | 2080 | 4541 | 2392 | 3683 | PASS |

**Zero samples below 1024 MiB on any card of any boot.** B3 is the one that
matters: it contains three refused reshards and three admitted ones, and it is
the boot whose R13 equivalent breached twice.

### 10.1 Gate 3 — `kv_ascend_mark`, and a caveat R13 could not have seen

Peak occupancy this window: **B1 0.651823, B2 0.651592, B3 0.651925** against
the 0.85 mark — **UNREACHED**, and reproduced to three decimals across arms.

R13 measured **0.8418** and concluded gate 3 was "1.0 % from reachable". Under
the **identical driver**, this window reads 0.65. The difference is the boot
arm: pinning `--rank-mlp-ratio 94,13,29` funds a larger KV pool, so the same
load occupies a smaller fraction of it. **Gate 3's reachability is a property
of the ARM, not of the rig**, and a shift that chases the last 1 % with a
heavier burst should first check which arm it is standing on.

---

## 11. TEARDOWN

Corridor sampler stopped and each `corridor.csv` closed before its boot.
Trace summary line present on all three ranks of B1, B2 and B3 — verified by
count, because the pass refuses without it and the summary is written last by
construction.

Heartbeat stopped **before** the holder was released (`.hb-stop-363-act-r2`
17:16:47Z, unit confirmed inactive, last beat 17:16:35Z). Aged to 152 s before
the restore, so the restore script's own guard passed on the first attempt
rather than refusing once as it did in R13.

Serving on 30030 restored via the sanctioned `res-r5 restore_ship.sh` in scope
`ship-restore-r14.scope`, and verified with a **real generation** — text
exactly `MERIDIAN42`, `finish_reason=stop`, `completion_tokens=6` — never a
health 200, because the #622 wedge signature answers health and emits no
tokens. Raw: `/spinning/evidence-363-act-r2/restore_verify.json`.

Port 30099 never touched. No broad `pkill`; every process stopped by its own
unit name.

---

## 12. WHAT THE NEXT SHIFT OWES, IN ORDER

1. **Defect 8 (§9.6).** The ms compute/wait split is `None` on a graph-covered
   forward, and that is now the ONLY thing between this ticket and a computed
   `signal_pct`. It needs a decision, not a bug fix.
2. **Defect 5 (§5).** The booted stage's weight vector, read from resolved
   state instead of the flag, with both sides canonicalised into the same
   ratio space before comparison. Until then `--rank-tp-ratio auto-performance`
   boots can never have a flip target, whatever else is fixed.
3. **Gate 3's arm dependence (§10.1)** — record which arm every
   `kv_ascend_mark` reading was taken on, and stop comparing across arms.
4. **The flip cost is still a lower bound (§9.4).** Every instrumented flip
   this window ran on an empty pool, because the guard correctly refuses a
   loaded one at this operating point. A real flip cost needs either more
   headroom or a smaller redistribution than `30,17,17 -> 1,1,1`.
