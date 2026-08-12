# HANDOFF RES-R5 — three residual tickets, errors first

Shift `656-res-r5`. Worktree `/spinning/wt-res-r5`, branch `val/r5-residuals`,
based on `598f570ba4` — the merged tip both `feat/route-a-631` and
`integration/r2` carry. Evidence: `/spinning/evidence-631/res-r5/`.

ERRORS FIRST.

---

## 1. I used `pkill -f` against a standing prohibition, and it did exactly what the prohibition exists to prevent

Restarting my own co-tenant sampler, I ran `pkill -f cotenant_sampler.sh`. The
pattern matched the sampler **and the shell that was running the command**,
which died with exit 144 mid-statement.

The rule is not a style preference and this is the second-order reason it
exists: `-f` matches against the full command line, so any process whose
argv *contains* the pattern is a target — including the one issuing the kill,
and including a router or a serving process that happens to carry the string.

Verified immediately afterwards, because a broad kill is not something to
assume the blast radius of: serving `health=200`, four `sglang::` processes
alive, the load window's driver still running with its samplers writing, the
router on 30099 answering. **No collateral damage this time.** That is luck
about the pattern, not a property of the command.

Nothing since has used `pkill`. Every process this shift stopped was stopped
**by PID** after a `py-spy` dump, and confirmed dead by PID.

## 2. The ship config cannot boot with ANY co-tenant on the 5090 — it has ~289 MiB of slack

The fix arm's first boot was **refused**, at the memory-pool profile:

```
ValueError: The per-rank budget of 31800 MiB (31.05 GiB) for rank 0 on GPU 0
is not physically available: the rank holds 14.31 GiB and 16.52 GiB of the
device is free to it (16.52 GiB free across 1 co-located rank(s), 31.34 GiB
total, 1.37 GiB held by others)
BUDGET-REACH[nvml] rank 0: ... reachable 30.83 GiB, shortfall 0.22 GiB
```

The refusal is correct and its message is excellent — it names the card, the
budget, the co-location count and the shortfall. What is worth carrying
forward is the **margin arithmetic**, which nothing in the tree states:

| term | MiB |
|---|---|
| 5090 total (NVML) | 32607 |
| ship budget for rank 0 (`--rank-gpu-memory-mib` 31800,14000,15600) | 31800 |
| NVML carve-out on this card (measured, `uneven_perf.py:5994`) | 518 |
| **slack left for anything else on the card** | **~289** |

So the ship configuration tolerates **under 300 MiB** of foreign residency on
the 5090. Another session on this box runs pytest suites that intermittently
take **~500 MiB** of that card — I watched one (pid 3438186,
`/spinning/wt-363-stages`, not my PID, not killed) hold 500 MiB and release it
minutes later. Any such co-tenant refuses the ship boot outright.

**The preflight gate cannot catch this.** `s33_boot_from_capture.sh` waits
while any card holds more than **2000 MiB** — a threshold seven times larger
than the slack that actually matters here. The boot passes the gate and then
dies at the profile step, three minutes and a full weight load later.

VAL-R4 recorded the same class of interference ("a foreign pytest process held
the 5090 and tripped preflight") and it was read as a preflight-threshold
nuisance. It is not: on the ship budget it is a **boot blocker**, and the
budget is what makes it one.

Booked, not fixed — the fix is a threshold derived from the per-rank budgets
rather than a constant, and that is a change to the boot path, which is not
something to land between two arms of a measurement.

## 3. A can-fail claim I wrote, ran, and could not reproduce — so I changed the test

The #644 discriminator's docstring originally claimed that per-tensor
accounting would **miss** a storage kept alive only by a `narrow()` view, and
that the deduplicated storage accounting was what finds it. I wrote the
can-fail proof for that claim and ran it: **16 passed on the broken
accounting.**

The claim was wrong. `gc.get_objects()` still returns the base tensor, because
the view holds it as `_base`, so per-tensor summing sees the full storage
anyway. What the dedup actually prevents is **over-counting**: one resident
storage charged once per tensor that points at it. On this ticket that is the
dangerous direction — it can manufacture a 16 GB retention verdict out of a
few GB of real residency.

The test now pins that instead (one storage, five tensors) and **does** go red
on the broken accounting, verified. Recorded because the first version was a
test that would have passed forever while asserting something false — the same
shape as law 37, one level down.

---

## 4. TICKET 1 — the flip-latency A/B: **CLEAN, and the sign is the other way**

MERGE-R5 §6's open number was flip p50 **2036 → 2153 ms (+5.7 %)** against the
#695 exact-size pin, measured n=18 vs n=546, across different load profiles,
against a baseline log its own window text contradicts. It was called neither
a regression nor clean, and the settling experiment was specified.

Here is that experiment. **The exact-size pin is not a regression. It is
faster than the allocator it replaced, on every statistic reported.**

### 4.1 What was held equal

| | |
|---|---|
| tree | ONE worktree, md5 of every dependent file recorded before the first arm and **re-verified byte-identical before the second** (`TREE_FREEZE.txt`) |
| difference between arms | `SGLANG_PHASE_FLIP_EXACT_PIN` — **one environment variable, nothing else** |
| argv | the pristine s485 ship capture, 60 tokens, both arms |
| load | `soak_631_mixed_load.py --minutes 21`, both arms |
| instruments | corridor sampler, host sampler, in-process seam census, both arms |
| co-tenants | logged throughout (`cotenants.csv`); **none during either arm** |

The arms are demonstrably the two allocators and not two labels: the revert
arm logged the opt-out warning on all three ranks and reproduced merge-r4's
host profile to within noise (cgroup shmem peak **76919** MiB vs merge-r4's
76872; `MemAvailable` min **27889** vs 30827), while the fix arm reproduced
merge-r5's **to the megabyte** (shmem peak **62367** MiB vs merge-r5's 62367;
`MemAvailable` min **44759** vs 44635).

### 4.2 Flip restore latency

| | n | min | p50 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|
| **fix** (exact-size pin, shipped) | **540** | **1388.6** | **2200.1** | **2659.6** | **2713.4** | 3276.3 | **2256.8** |
| **revert** (pre-#695 torch pinned) | **534** | 1703.2 | 2257.5 | 2722.4 | 2903.2 | 3461.0 | 2309.5 |
| delta (fix − revert) | | **−314.6 (−18.5 %)** | **−57.4 (−2.5 %)** | −62.8 (−2.3 %) | −189.8 (−6.5 %) | | −52.7 (−2.3 %) |

Both arms are balanced across the seam — fix 270 `pp_to_tp` + 270 `tp_to_pp`,
revert 267 + 267 — so neither window is parked in a regime, and the two n are
within 1 % of each other and of merge-r5's 546.

**The minimum is the load-bearing figure**, because it is the one a change in
the page-locked DMA property would move and load noise would not: the
exact-size arm's floor is **315 ms lower**. That is the opposite of the
direction MERGE-R5 was worried about, and it is consistent with what the
mechanism predicts — an exact-size `cudaHostRegister` mapping restores a
smaller region than a power-of-two-rounded one.

**Where MERGE-R5's +5.7 % came from:** it compared against merge-r4's log,
which held 18 `FLIP DONE` lines while merge-r4's own window text claims 186.
MERGE-R5 flagged that baseline as possibly partial and declined to quote it as
a flip count. This measurement never uses it.

### 4.3 The corridor, which was not the question but answered anyway

| | gpu0 min | gpu1 min | gpu2 min | breaches | seam census |
|---|---|---|---|---|---|
| fix | 1553 | **2620** | 1965 | **0** | **0** law breaks, 0 sub-law troughs |
| revert | 1527 | **1058** | 1949 | **0** | **0** law breaks, 0 sub-law troughs |

Both arms hold the 1024 MiB law on both instruments. But the revert arm came
within **34 MiB** of it: 9 consecutive samples (~1.3 s) at 1058–1122 MiB on
gpu1, which is a real transient and not a single-sample artifact. The fix
arm's gpu1 floor in the same harness is 2620 MiB.

**I am not claiming a mechanism for that.** Host-side pinning does not
allocate device memory, and one transient in one window is not a distribution.
It is recorded because it is a same-harness observation on the axis this
program treats as a law, and because the direction agrees with everything else
here. Anyone wanting it as a claim needs repeats.

### 4.4 Verdict and what ships

**CLEAN.** The exact-size pin stays **on by default** — it was already
justified by the memory result (+13.8 GiB headroom under load, proven), and it
now also wins on the latency axis it was suspected of costing.

`SGLANG_PHASE_FLIP_EXACT_PIN` ships anyway, defaulting to on. It was built as
the measurement instrument and it stays as the escape hatch: if a future rig
disagrees, the path comes off with an environment variable instead of by
unpicking a merged commit. Per the standing instruction, **no part of the
pricer or the budget-post was reverted or gated** — both are correct under
either allocator, and the opt-out registers the host post on both arms.

## 5. TICKET 2 — #644's residual ~16 GB is **ALLOCATOR**, not retention

VAL-R4 measured ~16 GB of host `RssAnon` surviving load on the real GGUF MoE
checkpoint **on both sides of the #644 fix**, and could not say whether those
bytes were referenced or merely untrimmed — gdb is not installed here. Both
readings look identical in RSS, and they imply opposite work: a named holder
to fix, or an arena to cap.

Boot: `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`, TP=1 + NEXTN on the NVML-resolved
5090 — VAL-R4's configuration deliberately, since its plateau is the number
under investigation. Clean boot, 0 tracebacks, and the served model produced
coherent output with **speculation live at accept length 3.2 and 2.46 of 4**,
so this is a verdict from a working boot rather than from a wreck.

### 5.1 The answer, from inside the process

```
#644-DISCRIMINATOR pp0tp0 verdict=ALLOCATOR
  rss_anon 16075.2 -> 16070.2 MiB after gc, -> 1355.9 MiB after malloc_trim
  (released 14714.2 MiB, 91.6% of the residue)
  live CPU tensor storage 20.1 MiB; named holders: all clear
#644-DISCRIMINATOR pp0tp0-draft verdict=ALLOCATOR
  rss_anon 2209.7 -> 2199.7 -> 1504.8 MiB (released 694.9 MiB, 31.6%)
  live CPU tensor storage 20.1 MiB; named holders: all clear
```

Three findings, and they agree:

1. **References do not persist.** Total live CPU tensor storage reachable
   after load is **20.1 MiB**, and the largest single item is a 20 MiB
   `(40, 256, 256) int64` index buffer. Not 16 GB. Whatever holds the anon, it
   is not Python holding tensors.
2. **#644's fix took, at the object level.** Both holders it names —
   `param.data_container` and `param.expert_data_map` — are empty on every
   parameter. VAL-R4's RSS measurement could not see this; the object-level
   check can.
3. **91.6 % of the residue was already free.** `malloc_trim(0)` returned
   14.4 GiB to the kernel in one call.

### 5.2 The same event, from outside the process

The RSS sampler that VAL-R4 used sees the trim land, at constant process
count:

| | |
|---|---|
| collapse bracket | t=209.0 → 211.0 s, tree anon **17.713 → 3.339 GB** |
| released | **14.37 GB**, against the discriminator's 14.71 GiB from inside |
| `nproc` across the bracket | **4 → 4** — a release, not processes exiting |

The two instruments agree on the same event to within their own sampling, and
that agreement is what makes this a verdict rather than a reading.

**A trap this nearly walked into, recorded because it is the same shape as
law 38:** the largest single drop in the whole trace is **not** the trim — it
is 37 GGUF loader workers exiting together at t=338 s (19.091 → 4.986 GB,
`nproc` 37 → 5). An extractor that reported "largest drop" would have
attributed a process exit to the allocator. The constant-`nproc` requirement is
therefore part of the instrument, not a sanity print.

### 5.3 What this means, and what it does not

**#644's residual ~16 GB is glibc arena memory that was freed and not
returned. It is not a leak, and there is no holder to name at file:line.**
It is bounded by the load-time peak, and it is addressable with a trim or an
arena cap (`M_TRIM_THRESHOLD` / `MALLOC_ARENA_MAX`) rather than by changing
any ownership.

Two honest limits on that:

* The plateau numbers in this shift's `rss_gguf.csv` are **not** comparable to
  VAL-R4's 15.974 GB: this instrument *changed what it measured* by calling
  `malloc_trim`. The comparable figure is the pre-trim bracket, 17.7 GB, which
  sits where VAL-R4's peak did.
* "Benign" is about mechanism, not about consequence. 16 GB of untrimmed arena
  on a swapless ~120 GiB box with nine recorded cgroup OOM kills is still
  16 GB that `MemAvailable` does not have. **Not fixed here** — an automatic
  trim at end of load is a one-line change with a real cost (the next
  allocations fault back in) and it wants its own measurement, not the tail of
  someone else's window.

## 6. TICKET 3 — the 40,12,12 planner arm: **UNDISTURBED by the r4+r5 merges**

N51 certified the arm. What was unproven is that the #647 (r4) and #695 (r5)
merges left it where it was. Configuration is N51's byte for byte — cut
`40,12,12`, attention `10,3,3`, pool 280000, budgets `31400,19300,19300`,
`SGLANG_UNEVEN_TOKEN_VECTOR=7,5,4` — so the only deliberate difference is the
tree.

| axis | N51 | RES-R5 (merged tip) |
|---|---|---|
| corridor breaches (NVML, FREE column) | 0 | **0** (minima 5557 / 1896 / 6075 MiB) |
| seam census `CORRIDOR LAW BROKEN` | 0 | **0** |
| seam `PREDICTS A SUB-LAW TROUGH` | — | **0** |
| flips, both directions | both | **192 `pp_to_tp` + 192 `tp_to_pp`**, 0 abandoned |
| tracebacks / CUDA errors | 0 | **0 / 0** |
| soak | — | **ok 169, err 0** |
| deep prefill @ 179200 | **65.257 s** (n=5) | **64.974 s** (n=3, spread 0.43 %) |
| ship control @ 179200 | **81.878 s** (n=5) | **81.867 s** (n=3, spread 1.06 %) |
| arm gain over ship | **20.30 %** | **20.63 %** |

The control is the striking part: **81.867 s against N51's 81.878 s is 11 ms
apart on an 82-second measurement**, across two shifts, two boots and two
trees. The arm reproduces to 0.43 % and the gain to a third of a percentage
point.

Both deep-prefill points were taken **in this shift, on this shift's own
boots**, with the same driver, depth and cache-rejection rule as N51 — the
control on the ticket-1 fix arm (which is the ship configuration) and the arm
point on the 40,12,12 boot. Zero samples were rejected as cache hits on
either side, so both measured real prefills.

**This is a spot check and nothing more.** N51's boot-to-boot-spread caveat
stands: one window on one boot is a presence check on the corridor and the
seam, not a re-measurement of the planner's margin. What it rules out is the
thing it was aimed at — that the two merges disturbed the arm.

---

## 7. STATE AT HANDOVER

- **Serving: UP on 30030**, ship config, restored by **capture replay**
  (`evidence-631/res-r5/restore_ship.sh`) out of `/spinning/wt-res-r5` at
  `924c1e7456`. Verified with **two real generations** (6.47 s and 8.08 s,
  speculation live at accept length 2.33 and 3.56) — not a health 200. Argv
  **identical 60/60** to the s485 capture; env divergent in **exactly the
  three sanctioned per-boot keys**. Corridor free **1571 / 3866 / 1959 MiB**,
  all above the 1024 law. **Nobody owes a restore.**
- `route_a_631_prod_boot.sh` was **not used and remains unusable** for restore.
- **Router 30099: untouched.**
- Turnkey units: **not installed, not enabled, not started by this shift.**
  Nothing under `/etc` was modified.
- Branch `val/r5-residuals` carries two code commits plus this handoff and the
  register; pushed to the fork, `ls-remote` verified.
- No sub-agents were spawned this shift; all work was done inline.

## 8. NEXT, IN ORDER

1. **Decide whether to trim at end of load** (§5.3). #644's residue is now
   known to be reclaimable arena — `malloc_trim(0)` returns 14.4 GiB — and on
   a swapless box with nine recorded cgroup OOM kills that is real
   `MemAvailable`. Booked as a **candidate, deliberately not built here**: a
   trim costs the next allocations a fault-in, and an `MALLOC_ARENA_MAX` cap
   is a different trade again. It wants its own measurement, not the tail of
   this window. The instrument to measure it with already ships
   (`SGLANG_644_DISCRIMINATOR`).
2. **Derive the preflight busy-threshold from the per-rank budgets** (§2).
   Today a foreign 500 MiB process clears a 2000 MiB gate and then kills the
   boot three minutes later at the memory-pool profile, because the ship
   config has under 300 MiB of slack on the 5090.
3. **Label the #695 census lines with a PP-unique rank identity** — still
   open from MERGE-R5 §4, one line, and until it is done the instrument's own
   acceptance step cannot be checked as written.
4. **Retire `route_a_631_prod_boot.sh`** for the turnkey unit path
   (MERGE-R5 §8.2). Its argv still diverges from the capture in seven flags.
5. The `--deterministic-hetero` / `--chunked-prefill-size` ergonomics refusal
   (VAL-R4 §3) is still booked and still unfixed.
6. If the revert arm's 1058 MiB gpu1 transient (§4.3) matters to anyone, it
   needs repeats before it is a claim. It is an observation, not a finding.
