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

---

## 3. THE EVIDENCE

| axis | result |
|---|---|
| **#631 flip family** | **1088 passed / 0 failed** (1076 inherited and unmoved + 12 new) |
| can-fail proof | with the floor disabled, the 4 "it acts" arms go RED and the sibling arms stay green |
| ruff / codespell | clean on the changed files |
| root cause | in-process seam census, matching the external sampler to 1 MiB |

### The confirmation window — SEE §3a

---

## 4. WHAT IS NOT DONE, STATED SO NOBODY READS IT AS DONE

* **#659 IS NOT CLOSED.** The blocker is unchanged from HANDOFF_686: no parked
  session has completed its generation, because the instance dies on #224's
  PS2 born-spilled-deep path (C26). This shift prioritised the corridor breach
  as the brief directed, and the window occupies all three cards, so the C26
  probe boot could not run beside it.
* **The C27 fix is proven in the pins and on ONE window.** The close trigger
  written into the register is deliberately two-part: 0 breaches **and** the
  preemption demonstrably reachable. A green window in which the mechanism
  never armed proves nothing, and this corpus has shipped that mistake before.
* **The `_seam_draw_max` term is per-rank and cold on the first flip of each
  direction.** Until a rank has completed one cutover in a direction the
  measured term is 0 and the behaviour is exactly the old one. That is
  deliberate — an unmeasured bucket is never a licence to invent a number —
  but it means the very first cutover after a boot is unpredicted.
