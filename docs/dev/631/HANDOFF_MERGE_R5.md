# HANDOFF MERGE-R5 — two merges, four ops defects, and #695 proven on metal

Shift `656-merge-r5`. Worktree `/spinning/wt-merge-r5`, branch
`merge/r5-batch`, based on `origin/feat/route-a-631` at `281bbfb739`.
Evidence: `/spinning/evidence-631/merge-r5/`.

ERRORS FIRST.

---

## 1. The #695 metal recipe could never have worked — it cannot find a rank

**Found on first contact with metal, and it would have consumed the first GPU
window anyone spent on it.**

`host_shmem_695.py` discovered ranks with
`comm.startswith("sglang::scheduler")`. The kernel stores `comm` in
`TASK_COMM_LEN` = 16 bytes, so `/proc/<pid>/comm` returns the 15-character
prefix `sglang::schedul`. `"sglang::scheduler"` is 17 characters. **The test
was False for every process that has ever existed**, and against a fully
healthy three-rank boot the script printed

```
no sglang::scheduler process found. Boot the server first, or name the pids with --pid.
```

Measured on the live instance:

```
$ cat /proc/2641744/comm
sglang::schedul          # 15 chars, not 17
```

Two things make this worth reading twice:

1. **The tree already knew.** `scripts/hostmem_sample.sh` matches
   `RANK_COMM="sglang::schedul"` — the truncated form — and has done all
   along. The #695 script is the outlier, so this was not an unknown fact
   about Linux, it was an inconsistency between two sibling instruments.
2. The #695 branch's own handoff called its desk work "hermetic". It was, and
   that is exactly why this survived: the hermetic tests never called
   `discover_scheduler_pids` with a real `comm`.

Fixed red-first, commit `1798cbfb6b`: matches the truncated prefix, and adds
an injectable `read_comm`/`pids` seam so the matching is testable without a
server. 5 of the 6 new tests fail on the pre-fix script (verified by running
them against a clean worktree of the parent commit); the sixth pins the
kernel-truncation premise and passes on both sides.

---

## 2. SIGTERM to the launcher: the orphan window is real, but I saw the mild end of it

VAL-R4 recorded SIGTERM orphaning three ranks that kept **~55 GB of VRAM**
with the parent already gone. The mechanism is confirmed at file:line:
`engine.py` installed a launch-phase handler for **SIGQUIT only**, so SIGTERM
kept Python's default disposition — immediate termination, no `finally`, no
`atexit` — for the whole of `_launch_subprocesses` → weight load → warmup.
`launch_server.py`'s `finally: kill_process_tree(...)` therefore never ran,
`Engine.shutdown`'s `atexit` never ran, and the ranks are non-daemonic
`mp.Process` children in the parent's own process group, so nothing else
reaped them. The running-phase SIGTERM handler in `tokenizer_manager` does not
close this: it is installed by `auto_create_handle_loop()`, **which runs on the
first request** — it covers a serving instance and specifically not a booting
one.

**What I actually observed, recorded because it is weaker than the report it
confirms:** stopping serving this window, the three ranks were alive in the
first sample after the launcher exited and had **exited on their own a few
seconds later**, with all three cards back to 3 MiB used. That is a
*transient* orphan window that self-resolved, **not** a reproduction of the
persistent ~55 GB hold. The mechanism is the same; the severity I can prove
from this run is the mild end of it. The fix should not be sold on this run's
evidence — VAL-R4's is the load-bearing observation.

Fixed red-first, commit `be700dfa4a`: the signal installation is extracted
into `_install_launch_phase_signal_handlers` so the disposition is testable
without signalling the test runner, and SIGTERM now reaps the tree and exits
rather than returning into a boot that has been asked to stop. All 6 tests
fail before the fix.

Note the asymmetry that remains: under systemd the gap was always masked,
because `htsglang-serving@.service` sets `KillMode=control-group`. **Only the
shell path was exposed**, and that is the path VAL-R4 and this window were on.

---

## 3. Preflight refused an idle machine, and the fix is a subtraction not a bigger constant

VAL-R4 hit `REFUSE_CARD_BUSY subject=RTX 5090 observed=521 MiB in use (no
compute pids) expected=<= 512 MiB` on a rig with nothing running, and had to
hand-raise the threshold to 600 to proceed.

The interesting part is not the constant. It is that **the carve-out is
measured, named, tested and budgeted everywhere else in this tree** —
`registry/nvml.py` reports it as `reserved_bytes`, `mem_ledger` books it as
`TERM_NVML_CARVE_OUT`, `reconcile.py` subtracts it, `uneven_perf.py:5994`
records the measured magnitudes (**425 MiB on a 3080, 518 on a 5090**), and
`corridor_sample.sh`'s own header states them. `turnkey/preflight.py` was the
single consumer that received `MemoryInfo` and **dropped `reserved_bytes` on
the floor**, folding the carve-out into `used` and comparing the sum against a
flat constant.

Fixed red-first, commit `0d3d973aed`: `CardObs` carries `reserved_bytes` and
the check reads `foreign_bytes() = total - free - reserved`, so the quantity
under test is derived from the measured carve-out instead of guessed above it.
`card_busy_mib` goes back to being what it was documented as — an allowance for
genuine foreign bytes. The refusal now names the foreign figure and the
discounted carve-out separately, because the old text reported one number that
silently contained both. An unknown carve-out (`reserved_bytes = 0`, which is
also NVML's own fallback) degrades to the old conservative answer rather than
widening the threshold; that case is pinned by its own test.

---

## 4. The #695 census cannot tell you which rank it is talking about

All three census lines label themselves **`rank0`**:

```
HOST-SHMEM rank0 host=27.28GiB ...
HOST-SHMEM rank0 host=16.95GiB ...
HOST-SHMEM rank0 host=16.00GiB ...
```

`scheduler.py:1358` passes `rank=getattr(model_runner, "tp_rank", None)`, and
`tp_rank` is 0 on **every** rank of a PP=3 / TP=1 deployment. So the lines are
not attributable to a rank, and the recipe's own acceptance step ("one
`HOST-SHMEM rank<N>` line per rank") cannot be checked as written — you get
three lines and three identical labels.

**Not fixed inside this window, deliberately:** the tree under test was frozen
while the confirmation window ran. It is a labelling defect in an instrument,
not a memory defect, and every number the instrument reports is correct. Next
shift: label with a composite that is unique under PP (pp_rank + tp_rank), not
with `tp_rank` alone.

---

## 5. WHAT IS PROVEN

### 5.1 The merges — both shipped, each tested before the next

Order was correctness first, per the program. **Each merge was tested and
pushed to both lines before the next was started**; nothing was batched.

| step | commit | lines |
|---|---|---|
| (a) `val/r4-metal` — the #647 shared-expert-gate completion fix | `8cdc81555d` | pushed to `feat/route-a-631` + `integration/r2`, `ls-remote` verified |
| (b) `fix/scheduler-shmem-residency` — #695 | `794b83bece` | same, verified |
| ops hardening (3 commits) | `0d3d973aed` | same, verified |

Both merges were **clean, zero conflicts** — confirmed in advance by a
path-overlap check that returned empty.

Suites, each against a **same-environment baseline worktree**, because a
failure count means nothing without the number beside it:

| suite | baseline | after |
|---|---|---|
| flip family (the canonical 3-directory list) | 1116 passed | **1116 passed** — unchanged across all three pushes |
| `unit/model_loader` + `unit/quantization` (#647) | — | **470 passed, 29 skipped, 94 subtests** — exactly VAL-R4's number |
| `unit/{mem_ledger,memtier,model_executor,model_loader}` | 16 failed, 1426 passed | 16 failed, **1463 passed** (+37 = exactly the new tests) |
| `unit/model_executor` alone | 15 failed, 577 passed | 15 failed, **588 passed** (+11) |
| `unit/managers` | 9 failed, 1286 passed, 18 skipped | **identical** |
| `unit/turnkey` | — | **56 passed** |

Every failure count is identical on both sides. The entrypoints CORS failures
(4 failed + 3 errors) were **proven pre-existing** by running that directory on
a clean worktree of the merge-(b) commit — identical set.

**A measurement trap worth naming**, because I fell into it for two rounds:
extracting a pytest summary with `tail -c 300` reported *nothing* for the
`model_executor` and 4-directory runs, and I twice read that as the "test box
kills long pytest runs at random" behaviour from the shmem handoff §1.6. **It
was not.** Both runs had completed normally; the summary line was simply
buried under a flood of trailing `forward-peak dump failed` stderr. Nothing
was ever truncated. Grep the whole file, not its tail.

### 5.2 #695 on metal — ALL FIVE GATES PASS

The BEFORE arm did not need a boot: **the instance that was already running
was the parent commit** (`281bbfb739`, no #695), so it was captured read-only
before being stopped. Same ship argv, 60/60.

| gate | result |
|---|---|
| 1. no large `anon-shared` mapping is a power of two *(load-bearing)* | **PASS** |
| 2. total anon-shared drops ~13.65 GiB | **PASS — 74.40 → 60.24 GiB, drop 14.16 GiB** |
| 3. generation diff empty | **PASS — text AND `output_ids` byte-identical**, accept_len 3.556 on both arms |
| 4. one census line per rank, small residual | **PASS on count** (3 lines, residual +0.30 GiB each); labelling defect in §4 |
| 5. no `exact-size pinned host image unavailable` fallback | **PASS — 0 hits** |

Before, every large mapping was a power of two; after, none is:

| rank | before | after |
|---|---|---|
| A | 16384, 16384, 512 | 13692.297, 13482.184, 455.051 |
| B | 16384, 8192, 512 | 9114.957, 7659.523, 277.551 |
| C | 8192, 8192, 512 | 8144.004, 7659.523, 277.551 |

**The after-sizes match the payload figures this repo already recorded at
`phase_flip_spill.py:851-854` to two decimal places** — 9114.957 observed vs
9114.95 recorded, 13482.184 vs 13482.18, 8144.004 vs 8144.00. That is what
makes the result a statement about *the allocation* rather than a free-memory
reading, which is the distinction the recipe was built around.

**Risk 1, `cudaHostRegister` on a ~13 GiB region: CLOSED, it works.** The
largest single registration was 13692.297 MiB and no fallback fired on this
driver.

**The census boot line is emitted into the log** — the one thing the shmem
handoff listed as proven-only-in-halves. Confirmed by `grep HOST-SHMEM` on
this boot.

### 5.3 The confirmation window

See §6 for the numbers.

---

## 6. CONFIRMATION WINDOW — clean on both instruments, and the saving holds under load

21 minutes of real mixed load (`soak_631_mixed_load.py`, decode streams plus
periodic 12k-token prefills) on the ship config out of the merged tree, against
the merge-r4 window as the comparison arm. Full page:
`evidence-631/merge-r5/WINDOW_SHIP_R5.txt`.

| | merge-r4 (comparison) | merge-r5 (this window) |
|---|---|---|
| corridor samples | 10383 | 9512 |
| `gpu0_free` MIN / breaches | 1523 / **0** | **1527 / 0** |
| `gpu1_free` MIN / breaches | 2610 / **0** | **2620 / 0** |
| `gpu2_free` MIN / breaches | 1945 / **0** | **1965 / 0** |
| seam census `CORRIDOR LAW BROKEN` | 0 | **0** |
| seam `PREDICTS A SUB-LAW TROUGH` | 0 | **0** |
| soak | ok 261, err 0 | **ok 261, err 0** |
| flips | — | **546 DONE: 273 `pp_to_tp` + 273 `tp_to_pp`** |
| `FLIP ABANDONED` | 0 | **0** |
| tracebacks / CUDA errors | 0 | **0 / 0** |

**Zero breaches on both instruments, and the flip ran in both directions in
exactly equal numbers** — 273 each way, which is what a window that actually
exercised the seam looks like rather than one parked in a regime.

### The #695 saving under load, which is the number that matters

| host instrument | merge-r4 | merge-r5 |
|---|---|---|
| cgroup `shmem` peak | 76872 MiB | **62367 MiB** (−14505 MiB, −14.2 GiB) |
| `MemAvailable` min | 30827 MiB | **44635 MiB** (+13.8 GiB) |
| cgroup `oom_kill` delta | 0 | **0** |
| rank processes | 3/3 survived | **3/3 survived** |

The at-rest 14.16 GiB is the mechanism; **this** is the claim. Against a defect
whose recorded consequence was nine cumulative cgroup OOM kills, host headroom
under load went up by 13.8 GiB.

### Risk 2, flip restore latency: a number, and an honest confound

| arm | n | min | p50 | p90 | max | mean |
|---|---|---|---|---|---|---|
| merge-r4 (rounded images) | **18** | 1372.1 | 2036.4 | 2542.1 | 2543.3 | 1961.8 |
| merge-r5 (exact-size, `cudaHostRegister`) | **546** | 1360.6 | 2152.8 | 2652.1 | 2808.5 | 2227.7 |

p50 +116 ms (+5.7 %), mean +266 ms (+13.6 %).

**I am not calling this a regression, and I am not calling it clean.** The
comparison is confounded three ways and every one of them favours caution:

1. **n=18 against n=546.** Eighteen flips is not a distribution.
2. The two windows ran **different load profiles**, and flip cost tracks live
   slot count and seam traffic, both of which the soak drives.
3. That n=18 is itself suspect: merge-r4's `WINDOW_SHIP.txt` reports **186**
   flips for its window while its `ship_boot_merge_r4.log` contains only 18
   `FLIP DONE` lines, so the baseline may be a partial log. **Do not quote 18
   as merge-r4's flip count.**

The minima are effectively identical (1372.1 vs 1360.6), which is what you
would expect if the page-locked DMA property is unchanged and the difference
lives in load-dependent tail behaviour rather than in the registration.

**Settling experiment, cheap and specified:** boot the parent commit and the
merged tip through the *same* harness (`window_merge_r5.sh`, same soak
arguments) and compare p50 over comparable n. Until then the fix ships **on
by default** — the memory result is large, proven under load, and addresses a
mechanism with nine recorded OOM kills, while the latency delta is small,
unattributed and measured against a baseline that may be a partial log.
Recorded here so nobody has to rediscover the question.

---

## 7. STATE AT HANDOVER

- **Serving: UP on 30030, from the MERGED tree** `/spinning/wt-merge-r5`,
  booted from the pristine s485 capture and verified with **real generations**
  (3.4 s and 6.3 s after the load window, speculation live at accept length
  3.0) — not a health 200. Corridor free **1741 / 3426 / 3183 MiB**, all above
  the 1024 law. **Nobody owes a restore.**
- The instance that was running at shift start was the parent commit, and it
  served as the #695 BEFORE arm before being stopped. Its argv was verified
  **identical 60/60** to the s485 capture, and its env diverged from the
  capture in **exactly the three sanctioned per-boot keys** and nothing else —
  so the capture is proven faithful, which is what made the replay safe.
- **Router 30099: untouched.**
- Turnkey units: **not installed, not enabled, not started by this shift.**
  Nothing under `/etc` was modified. The unit *rendering* change is tested
  against temp dirs only.
- Branch `merge/r5-batch`; both `feat/route-a-631` and `integration/r2` carry
  the same tip, `ls-remote` verified after every push.

## 8. NEXT, IN ORDER

0. **Label the #695 census lines with a rank identity that is unique under
   PP** (§4). One line, and until it is done the instrument's own acceptance
   step cannot be checked.
1. **Settle the flip-latency question** with the same-harness A/B in §6. It is
   the only open number attached to a shipped change.
2. **Retire `route_a_631_prod_boot.sh` in favour of the turnkey unit path.**
   The env drift class is now closed by a refusal gate, but the script's
   **argv still diverges from the ship capture in seven flags** (`--model-path`
   by the `yarn1.5` suffix, `--pp-stage-ratio` 14,10,8 vs 2,1,1,
   `--context-length` 393216 vs 262144, `--rank-gpu-memory-mib`,
   `--phase-flip-tp-vector`, `--max-total-tokens`, `--max-mamba-cache-size`).
   One capture must govern env **and** argv as a unit; the #539 orchestrator
   already does exactly that.
3. **#644's residual ~16 GB is still unsettled** (VAL-R4 §2). It needs the GGUF
   MoE checkpoint, not this ship config, so it could not ride this window's
   boot — that is why it is untouched, not an oversight.
4. VAL-R4's ticket 4 (`--pp-solve-cut` recommendable arm) remains unrun.
5. The `--deterministic-hetero` / `--chunked-prefill-size` ergonomics refusal
   (VAL-R4 §3) is still booked and still unfixed.
