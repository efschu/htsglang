# HANDOFF 687 — #656 / #631 Route A, successor 43

The shift that root-caused the corridor breach and fixed it red-first. The
verdict is not the one the brief expected, and the difference matters: **the
breach is a LATENT DEFECT that has been in every window of this corpus, not a
regression introduced between successor 38 and successor 42.** Errors first.

---

## 0. THE ONE-LINE STATE

**The corridor breach is root-caused to a named mechanism with in-process
evidence, fixed in `18ff17ec6e`, and the fix is on metal. Every candidate the
brief named for the "regression" was FALSIFIED — the code that landed between
the two windows is inert, and the earlier GREEN window contains the identical
event, green by 59 MiB.** #659 is still NOT closed; C26 remains the blocker
and this shift did not get it onto metal (§4).

---

## 1. ERRORS FIRST

### 1a. THE BREACH IS NOT A REGRESSION, AND TREATING IT AS ONE WOULD HAVE SENT THE FIX TO THE WRONG MODULE

The brief framed this as "a regression against a triple-green baseline". It is
not. The same excursion is in the green baseline:

| window | samples < 1100 on gpu0 | duration | when | trough | vs the 1024 law |
|---|---|---|---|---|---|
| s38 (GREEN) | **12**, one contiguous run | **1.54 s** | t+1034.9 s | 1083 | **+59** |
| s34 (GREEN) | — | — | — | 1043 | **+19** |
| s42 (BREACH) | **12**, one contiguous run | **1.53 s** | t+982.1 s | **940** | **−84** |

One excursion per 30-minute window, twelve samples, the same duration to
10 ms, at the same point of the acceptance script — the completion of the
271k-token YaRN leg. s38's own register entry (C20) already recorded its
margin as **+59 MiB**. A baseline that is green by 59 MiB and by 19 MiB is not
a baseline a 142 MiB draw regressed away from; it is the same distribution
being sampled three times, with the floor inside its left tail.

**Everything the brief listed as a candidate was falsified, each with a
can-find control rather than a silent grep:**

* **N41's merged changes — INERT.** The registry observer is a clause appended
  to an existing line and reads only host-side structures; the P2 gate fix
  only extends the gate's reach to `pp_to_tp` and allocates nothing; the
  help-string fix has no runtime footprint. The words `registry` and
  `observer` are logged **zero** times in either window.
* **The #657 allocation-steering series — INERT, and it was the better
  a-priori suspect** (four commits touching `mem_cache/allocator/base.py`).
  It ships OFF (`SGLANG_CORRIDOR_STEERING=1` to arm), the variable is absent
  from the byte-identical environment, and `STEER`/`steer`/`free list` appear
  **0** times in both logs.
* **Environment drift — NONE.** The env blocks of the two boots are
  byte-identical, verified variable by variable.
* **Graph capture — IDENTICAL.** Both capture 3 graph sets, `backend=full`,
  `bs=[1,2,3,4]`, same tokens per bs. 0.56/0.30/0.28 GB vs 0.57/0.32/0.32 GB.
* **The pool difference (512552 vs 503950) is A WARM PAGE CACHE, not code.**
  `Load weight begin. avail mem` is byte-identical on all three ranks; the gap
  opens *during* the load, which took **11.11 s cold and 2.35 s warm**. A 4-5x
  faster load keeps more shard buffers in flight and leaves ~0.1 GB more
  allocator high-water on every rank. The smaller pool gives the memory back:
  the final decode-layout pools are identical in both runs.

### 1b. THE MECHANISM, NAMED — A `pp_to_tp` CUTOVER'S BACKING-RESTORE WALK ON RANK 1

Not a prefill and not a graph capture: prefill was *deferred* by policy
(`47441 tok prefill deferred to the next pp window`), there is no `Prefill
batch` line in the interval, and there are zero capture lines. The instance
was quiescent by design and doing nothing but the seam.

The breaching rank's OWN instrument recorded the trough to within 1 MiB of the
external sampler:

```
[2026-08-12 01:18:29 PP1] [#631 seam-census] pp_to_tp rank 1: transient 2066 MiB
  (baseline free 3006 MiB, trough 940 MiB at 'backing_restore_span')
```

The descent is the arena's page-commit walk, in ~24 MiB steps, verbatim:

```
backing_restore_span free=1028 step-20 slack=1054   <- last legal sample
backing_restore_span free=1004 step-24 slack=1054   <- CROSSES THE 1024 LAW
backing_restore_span free=980  step-24 slack=1054
backing_restore_span free=940  step-16 slack=1054   <- trough
backing_restore      free=940  step+0  slack=1054
kv_write             free=940  step+0  slack=1054   <- the 1.5 s plateau
gdn_state            free=940  step+0  slack=1054
weights_refill       free=1406 step+466
```

**IDENTITY, because this corpus has lost a shift to getting it wrong
(register law 9).** Corridor `gpu0` is **nvidia-smi index 0, a 3080, and it is
rank 1 — not rank 0 and not the 5090.** Confirmed three ways: census
baselines/troughs match the sampler per card (PP1 3006/940 = gpu0 3007/941),
`CORRIDOR-GUARD cleared on device 0 ... free 2578 -> 3788` matches gpu1's
trough exactly, and it agrees with law 9's stated permutation. **PP0, on the
5090, never came near the floor** (trough 2375) and needed the gate on 11 of
86 flips; the two 3080s needed it on essentially every one.

### 1c. THREE FAILURES HAD TO LINE UP, AND THE SHARPEST ONE IS NOW A LAW

**(i) The remedy that could have funded the walk was waiting for a hardware
failure, not for the policy floor.** `_mem_create_reclaiming` already knew the
exact remedy — torch sits on reserved-but-unused blocks and `empty_cache`
returns them to the driver — but its trigger is `CUDA_ERROR_OUT_OF_MEMORY`,
i.e. free memory reaching **zero**. The law floor is 1024 MiB above zero, so
the walk crossed it long before anything was refused and the remedy never ran.

Read the `slack=1054` column above again: **torch was holding 1054 MiB of
cached blocks through the entire trough.** The bytes to stay legal were there
the whole time and nothing asked for them, because nothing had failed yet.
Booked as **register law 16**.

**(ii) The recogniser held the evidence and never read it.**
`phase_flip_seam_census` samples the exact NVML observable at every stage,
names the 1024 MiB floor in its own docstring, and contained no comparison
against it — and emits its line only AFTER the flip has completed. The process
measured 940 and said nothing; the breach was found hours later in an external
CSV. Worth carrying separately: **the 100 ms external sampler UNDERSTATES
depth** — it read PP0's floor as 2578 where PP0's census recorded 2375 — so
the census is the tighter instrument and is the one a successor should judge
against.

**(iii) The seam-entry law check was priced on the wrong term.** It computes

```python
law_ok = margin_bytes > 0 and (verdict.free_after - staging_bytes >= law_floor)
```

`staging_bytes` is what the seam RESERVES. The census measures the DRAW at
2066 MiB against 1625 MiB staged, and `free_after` (3154) itself overstates
the 3006 the cutover actually entered with — 589 MiB of optimism against an
84 MiB breach. **s38's own yield was 18 MiB sub-law by that same arithmetic**
(free 2190, staged 1184 → 1006) and survived only because its estimate
overshot the real draw by ~388 MiB. Passing on an estimator's conservatism is
not a margin; it is the C20 pattern one level up.

### 1d. WHY THE FIX IS NOT "MAKE THE GATE REFUSE", WHICH IS THE OBVIOUS ANSWER

A `law_ok = False` routes into the abort path, and for `pp_to_tp` that path is
the **decode wedge**: under strict purity decode runs only in TP, so a
persistently refused `pp_to_tp` starves decode outright and nothing the PP
phase holds can ever end the refusal. Measured 2026-08-10: **411 abandons, 0
requests completed in 6 minutes, /health 503 with every rank alive and
logging.** Trading a 1.5 s corridor dip for a total outage is not a fix, and
register law 13 says the same from the other side.

So the actuator went where the bytes are taken, and the predicate only
predicts. That division is the whole design.

---

## 2. WHAT SHIPPED — `18ff17ec6e`

**1. `kv_vmm_backing._corridor_preempt`** — the same reclaim, with its trigger
moved from "the driver refused" to "the next commit would cross the law".
Rank-local (no collective — laws 14/15), does not shrink the pool (law 13),
one driver read on the common path, and it preempts rather than recovers. It
declines to act when there is no slack worth spending, deliberately: an
unfundable walk is a REAL finding about the budget and must not be hidden
behind cache churn.

**2. `phase_flip_seam_census` compares.** The law check lives where the sample
already is, announced ONCE at the stage that crosses (60 identical ERROR lines
on the no-return path would be their own outage), counted thereafter, verdict
in the HEAD of the summary line where every consumer of this corpus greps.

**3. The gate prices a PREDICTION on the measured draw.** The per-flip
in-cutover transient the census already computed was written to a stats dict
and read by nobody; it now feeds a running per-direction MAX which the seam
entry consults to predict a sub-law trough and say so. It does not move
`law_ok` — see §1d.

## 2b. C26 IS ROOT-CAUSED AND FIXED — `bdb2c3db53`

**The mechanism, and it is not a race or a corner case: PS2 is admitted onto
a backend that has no hook to divert its sentinels.**

`spill_extend_alloc` returns `make_sentinels(...)` *as* `out_cache_loc`, and
exactly one thing in the tree diverts that tensor away from the KV write:
`_dcp_write_scatter`'s `_sess_prefill_owner_write` branch — which is reachable
**only** from the token-sharded DCP lane, because `forward_extend` enters that
lane under `if self.uneven_dcp`. On plain TP the backend still BUILDS the
prefill-spill state (`_sess_mode="plain"`, `_sess_prefill_spill` computed, the
staging carve reserved) and nothing complains; `forward_extend` then falls
through to the stock `set_kv_buffer` and the sentinels go into `store_kvcache`.

The arithmetic closes exactly, which is why this is a mechanism and not a
theory: `host_base=4097` against a 4096-row allocator, request
`boundary=2620 L=3012`, so the 392 written indices are **6717..7108 against a
`size_limit` of ~4097** — every row out of bounds, layer 0, both ranks.

Two premises from HANDOFF_686 are corrected by this:

* **The crash is NOT delayed.** The ~60 s gap between the parks and the assert
  is a coincidence of the log. The parks at 01:37:28 completed cleanly and are
  unrelated; `fast-3` was admitted at 01:38:26 and the very next line is the
  exception. **The assert is raised by that same extend forward**, so there is
  no stale-sentinel decode path to hunt and `adopt_born_spilled_prefills` is
  never reached.
* **The `#501` flag-after-last-decline hypothesis is refuted for this crash.**
  `req.born_spilled_deep = False` at the end of `spill_extend_alloc` has no
  later reader. (The flag family does have a real instance, but on the
  DECLINE side — see §4.)

**The fix is at the admission gate**, where the information was missing rather
than wrong: `prefill_spill_deep_gate` now takes `backend_write_hook`, derived
from `_sess_mode != "plain"`. That is replicated boot config, fixed before the
first forward and identical on every rank, so the verdict needs no collective,
and a decline is a NON-ADMISSION rather than a rank-local skip around one
(law 14). Declining is bit-for-bit the pre-PS2 behaviour: the request stays
queued and the fast lane retries it, which is exactly what the log shows it
already doing one line earlier.

Belt and braces: `spill_extend_alloc` raises in Python if it is ever reached
on a plain backend. The two failures are not equivalent — a `RuntimeError`
names the cause and unwinds; the device-side assert it replaces poisons the
CUDA context and points the traceback at an `all_reduce` three frames from the
mistake.

**The pressure band for the next shift is now known, and one knob removes the
crash without perturbing the pressure.** `probe_boot_v5.sh` and
`probe_boot_v6.sh` are byte-identical except the park directory; all the
difference lives in the driver env (`SAT=6 FAST_SHOTS=4` vs `SAT=3
FAST_SHOTS=2`). Recommended order:

1. **`--chunked-prefill-size 256`.** The uncached tail is 392 tokens, so
   392 > 256 forces chunking and `prefill_spill_deep_ok` returns False on its
   "ONE CHUNK ONLY" condition — PS2 cannot be admitted at all, while PS1, the
   fast-lane spill and the entire park path are untouched. This is the knob
   for the first park/unpark completion proof.
2. `FAST_NEW=1200` (driver, default 600) — raises `need` so both spill demands
   fire at LOWER occupancy, reaching the park before the budget drains.
3. `SAT=4`, `FAST_SHOTS=3` — the intermediate between the two known boots.
4. `MTT=5120` — keeps `rem_total_tokens` around 1500 instead of 102, so every
   prompt lands in PS1's window and PS2 never applies.

Keep `HOSTGIB=1.5` / `MAXSPILLS=3` exactly as they are: a larger host budget
yields more free regions and makes the park HARDER.

---

## 3. THE EVIDENCE

| axis | result |
|---|---|
| **#631 flip family** | **1088 passed / 0 failed** (1076 inherited and unmoved + 12 new) |
| can-fail proof | with the floor disabled, the 4 "it acts" arms go RED and the sibling arms stay green |
| ruff / codespell | clean on the changed files |
| root cause | in-process seam census, matching the external sampler to 1 MiB |
| C26 pins (new) | 9 passed, both arms; kvso + tier-selection + park-tier suites unmoved (167 total) |

### 3a. THE CONFIRMATION WINDOW — GREEN, AND THE FIX IS WHAT MADE IT GREEN

Ship config, rebooted from `18ff17ec6e` (the fix), healthy 02:26:38Z, real
generate verified. `s34_acceptance_run.sh 31`.

**The fix fired in the identical event, and its own log line states the
counterfactual:**

```
[2026-08-12 02:43:27 PP1] cuMemCreate: committing 8388608 bytes would leave
  1016 MiB free, below the 1024 MiB corridor law floor. Releasing torch's
  cached blocks FIRST (1032 MiB of slack held, reserved 22.88 GiB /
  allocated 21.88 GiB). This is the same reclaim the OUT_OF_MEMORY path
  takes, moved ahead of the crossing instead of after the refusal.
```

Same rank (PP1), same direction (`pp_to_tp`), same trigger (the completion of
the 271k YaRN leg). It fired **once** — after the release, torch's slack drops
below the 64 MiB threshold and the remaining commits of that walk return at
the slack check, so the expensive path is self-limiting to about one call per
walk. The 1032 MiB of slack it spent is the same resource s42's trough
recorded as `slack=1054` and never asked for.

| window | entry free | draw | trough | vs the law |
|---|---|---|---|---|
| s38 (no fix) | 3469 | 2386 | 1083 | +59, by luck |
| **s42 (no fix)** | 3006 | 2066 | **940** | **−84, BREACH** |
| **s43 (fix)** | 2946 | 1922 | **1024** | **+0, HELD BY THE MECHANISM** |

The seam census for that flip:
`pp_to_tp rank 1: transient 1922 MiB (baseline free 2946 MiB, trough 1024 MiB
at 'backing_restore_span')`.

**This is why the close trigger was written in two parts.** A green window
alone would have been indistinguishable from s38's luck. Here the mechanism is
demonstrably reachable, demonstrably decisive, and the number it prevented
(1016 MiB) is in the record next to the number it achieved (1024 MiB).
`CORRIDOR LAW BROKEN` appears **0** times, which is the recogniser agreeing.

**THE WINDOW, JUDGED AGAINST N38 (the brief's reference):**

| axis | N38 (GREEN) | N42 (BREACH) | **N43 (this, fixed)** |
|---|---|---|---|
| corridor samples | 15235 | 14352 | **15671** |
| per-card MIN free MiB | 1083 / 1580 / 1581 | **941** / 1870 / 1243 | **1141 / 1624 / 1241** |
| **breaches vs 1024** | 0 / 0 / 0 | **12** / 0 / 0 | **0 / 0 / 0** |
| gpu0 quiescent p50 | 2369 | **2313** | **2313** |
| requests | — | 103 ok, 0 err | **104 ok, 0 err** |
| occupancy legs | — | 2 legs 3/3, peak 389324 | 3/3, peak **389324** |
| YaRN prompt tokens | 271237 | 271237 | **271237** (>262144) |
| phase flips | — | 86 | **90** |
| **seam entry YIELDS** | 1 | **3** | **3** |
| tracebacks / CUDA errors | 0 | 0 | **0** |
| deepest seam-census trough | — | **940** | **1024** |

**Read the last three rows together, because that is the whole argument.**
This window ran at the SAME quiescent baseline as the breaching one (p50 2313
on gpu0, against N38's 2369 — the warm-cache level, §1a) and took the SAME
number of seam-entry yields (3, against N38's 1). By the pre-fix mechanics
those are the conditions that produced −84 MiB. It held at exactly the floor
instead, and the log says which line did it.

**THE PATCH LEVEL THE WINDOW RAN, STATED EXACTLY** (Patchstand vor Last). The
serving instance reports `SGLANG_BOOT_COMMIT=18ff17ec6e`. That hash no longer
exists: the first push was rejected by GitHub's email-privacy rule because
three commits carried the wrong author address, and rewriting them to
`efschu@users.noreply.github.com` renumbered them. **The content is
byte-identical** — only the author header changed — and `18ff17ec6e` is now
`7964f3c525`, whose tree is `c80521c6ef`. So the window ran exactly the code
that is on the fork; the hash in the process environment is simply the
pre-rewrite name for it. A successor comparing the two will find no diff, and
should not spend a boot looking for one.

---

## 4. WHAT IS NOT DONE, STATED SO NOBODY READS IT AS DONE

* **#659 IS NOT CLOSED. The blocker has MOVED but not gone.** C26's crash is
  root-caused and fixed (§2b) with 9 pins and a can-fail proof, so the thing
  that killed the instance will not kill it again. What is still missing is
  the same single claim HANDOFF_686 named: **one parked session COMPLETING**,
  byte-identically, with `parked_count > 0` on the rebuilt instrument.
  That needs a probe boot, and a probe boot needs the two 3080s, which the
  confirmation window held for its whole duration.

  **This was a deliberate decision, not an overrun.** Running it would have
  meant stopping the ship config with limited context left, and the failure
  mode of running out mid-probe is serving DOWN — which is strictly worse than
  an unclosed issue. The brief ranks the corridor first and says to protect
  the ship config; both point the same way. The next shift inherits a fixed
  crash, a known pressure band, and a ~20-minute job:

  1. stop the ship config (capture `/proc/<pid>/cmdline`+`environ` first —
     `/spinning/evidence-631/s43/boot_ship_30030.sh` is that capture, current);
  2. boot `evidence-631/s42/probe_boot_v5.sh` **plus
     `--chunked-prefill-size 256`** — that single knob makes the 392-token
     tail chunk, which fails `prefill_spill_deep_ok`'s one-chunk condition, so
     PS2 cannot be admitted while PS1, the fast-lane spill and the whole park
     path are untouched (§2b);
  3. drive `force_spill.py` with `SAT=4 FAST_SHOTS=3`, keep `HOSTGIB=1.5
     MAXSPILLS=3` exactly as they are;
  4. let the load subside so `unpark_decision` fires, and assert the round
     trip through `proof_driver2.py` — whose verdict already hard-fails on
     `parked_count == 0` (exit 3), so it cannot pass vacuously;
  5. reboot the ship config and confirm health 200.
* **The C27 fix is proven in the pins and on ONE window.** The close trigger
  written into the register is deliberately two-part: 0 breaches **and** the
  preemption demonstrably reachable. A green window in which the mechanism
  never armed proves nothing, and this corpus has shipped that mistake before.
* **The `_seam_draw_max` term is per-rank and cold on the first flip of each
  direction.** Until a rank has completed one cutover in a direction the
  measured term is 0 and the behaviour is exactly the old one. That is
  deliberate — an unmeasured bucket is never a licence to invent a number —
  but it means the very first cutover after a boot is unpredicted.
