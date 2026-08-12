# HANDOFF 695 — the silent death was a logged SIGKILL, and the ledger was the host's

Successor 51. Branch `feat/route-a-631`. Evidence `/spinning/evidence-631/s51/`
(`RESULTS.md` is the verdict; `transient_red.py`, `gate_from_census.py` and
`hostmem_sample.sh` are runnable and reproduce the numbers below).

---

## 1. ERRORS FIRST

### 1a. C40 was never a silent death, and I found the line by reading the log

s50 reported rank0 dying with "no traceback, no OOM line anywhere, no kernel
OOM record". Line **49494** of that same boot log:

```
Subprocess scheduler_0 (pid=986533) crashed with exit code -9.
```

`-9` is SIGKILL. The launcher said how the child died and the shift's whole
verdict rested on it not having. Two of the three "absent" searches could
never have returned anything on this rig: a SIGKILLed process cannot write a
traceback, and the kernel's OOM report goes to a ring buffer an LXC container
cannot read (`dmesg` denied, no `/dev/kmsg`, `journalctl -k` empty). Law 33:
**a search that comes back empty is evidence only if it could have come back
full.**

### 1b. My own baseline change was wrong and I reverted it inside the shift

I moved the transient census's reference from the post-capture level to a
running maximum of the free column, reasoning that the flip's boot-time
backing swap frees 3776 MiB after capture. The arm boot refuted it: rank0's
free oscillates 1355-7820 MiB with the flip phase, so the maximum is not a
resting state — and `fixed_overhead_mib` is calibrated at the post-capture
point, so the higher reference charges the released layout twice on the one
rank where the constraint binds. Reverted, with the maximum recorded
alongside and the raw minima written to the artifact. Law 35, written as the
retraction it is.

### 1c. I ran CPU test suites during the ship confirmation window

The window is clean on both instruments, but rank0's NVML minimum is 2009 MiB
against N50's 2614 and N49's 2608. I cannot separate host busyness from
window-to-window variance with one window, so **the reproduction claim is
limited to "0 breaches on both instruments, three shifts running"** — the
minima themselves are not claimed reproduced.

### 1d. What I did not do

* No second arm on the cut the honest gate prefers.
* `seam_staging_mib` still uncalibrated (fifth shift).
* Arm B not run (fifth shift).
* The A/B control is same-SHIFT and same-code, not same-boot (§4).

---

## 2. BLOCKER 1 — CLOSED

Root cause is a HOST memory kill. Measured this shift, serving up:

| | |
|---|---:|
| container ceiling | ~120 GiB |
| shmem held by the 3 schedulers | **75.1 GiB** (unlinked `/dev/shm/sglang_loads_*.shm`) |
| swap | 0 |
| cgroup `oom_kill` to date | 9 |
| MemAvailable min, ship config soak | 23.5 GiB |
| MemAvailable min, 40,12,12 arm soak | **15.9 GiB** |

Not a cross-boot leak: stopping serving returned shmem 75.1 -> 0.0 GiB. The
planner cut simply costs ~8 GiB more host memory than the ship config, and
the container's real working margin is ~45 GiB, not 120.

Landed: signal-naming + cgroup-OOM attribution in `utils/watchdog.py` (and it
exonerates the OOM killer when the counter did not move, proven on a manual
SIGKILL); mode-independent liveness on `/health` (the #604 check was
unreachable on a default boot); `scripts/hostmem_sample.sh`.

**Core dumps cannot be routed to a path from inside this container** —
`core_pattern` is host-global and read-only here. The achievable version
(`ulimit -c` + a narrowed `coredump_filter`, core lands in the process CWD) is
in the boot script. A SIGKILL produces no core regardless.

## 3. BLOCKER 2 — CLOSED

The brief said the transient was calibrated on the wrong load state. On the
**wired** path it was not calibrated at all: `--pp-solve-cut` never set
`transient_mib`, so it took its `0.0` default. Executed against s50's census:

| transient charged | admits @ 280000 |
|---|---|
| wired path, `(0,0,0)` | **42,11,11** — the cut metal measured breaching |
| s50 desk, `(1346,…)` | 40,12,12 — also breached on metal |
| measured worst | 36,15,13 |

Landed: `planner/transient_census.py` (per-load-state minima, env-gated,
default off, ARMED is a module global so the default path is one boolean per
batch); the gate charges the WORST state and names it in the refusal;
`--pp-solve-cut` REFUSES a census with no measured transient instead of
pricing it at zero. Law 34.

---

## 4. METAL

### The arm window is CLEAN — the first planner-family window that is

`40,12,12 / attn 10,3,3 @ pool 280000`, flip ON, 22 minutes:

| instrument | result |
|---|---|
| NVML, 10317 samples | minima 5591 / **1355** / 6125, **0 breaches** |
| seam census, 600 troughs | deepest **1354 MiB**, **0 breaches** |
| flips | 200 both directions, 0 abandoned, seam machinery inert |
| soak | ok=296 err=0, 0 tracebacks |
| host | MemAvailable min 15.9 GiB, oom_kill delta 0, 3 ranks alive throughout |

s50's identical configuration breached twice (669 MiB) and lost rank0 at
minute 17. The two corridor instruments agree to **1 MiB** (1355 vs 1354).

### The honest gate agrees with metal to 44 MiB — but this is consistency, not prediction

Run through the WIRED construction against this boot's census and its
measured per-load-state transients (`gate_from_census.py`): it REFUSES
`42,11,11` (over by 461 MiB, naming load state `EXTEND`), ADMITS `40,12,12`
with 374.9 MiB of runnable headroom, and picks `40,12,12` itself. Measured
margin above the corridor was 331 MiB. **The transient came from this same
boot**, so what is demonstrated is that the arithmetic closes, not that it
forecasts. Applying one boot's table across a cut boundary is the next
shift's test.

### THE GAIN IS +25.5 %, NOT +50 %

Five shifts carried "no same-boot ship control" and quoted N48's 98.276 s. I
measured a control this shift, same code, same depth, on the configuration
that actually ships:

| arm | pool | median @ 179200 | spread |
|---|---:|---:|---:|
| ship control, measured this shift | 620000 | **81.878 s** | 0.31 % |
| arm 40,12,12 | 280000 | **65.257 s** | 1.67 % |
| ship control, N48, four shifts old | **280000** | 98.276 s | — |

The arm reproduces s50 to 0.08 % (65.257 vs 65.207), so the whole discrepancy
is in the DENOMINATOR. N48's control differs from mine in the KV pool AND in
four shifts of code and I cannot decompose it. Both answer real questions,
but **the number that belongs in a recommendation is the one whose
denominator is what boots**: +25.5 %.

### Minor: /flush_cache 400s after a soak on a PP instance

`not-idle because: chunked_req, last_batch, pp_microbatches` with 0 queued
and 0 running. One completed request clears it. Any cache-controlled
benchmark run straight after traffic will fail its first flush.

---

## 5. WIRE-OR-GATE: RECOMMENDABLE, GAIN RESTATED, NOT THE BOOT DEFAULT

All four of the brief's conditions pass: corridor 0 breaches on both
instruments over 22 minutes, no rank death (and C40's mechanism is now
understood), flips both directions, gain confirmed — restated to +25.5 %.

So `--pp-solve-cut` against a census carrying measured per-load-state
transients is **recommendable**. The operator flips defaults. Three caveats
belong in that decision:

1. **One clean window is not a certification.** s50's identical config
   breached at 669 MiB; mine held at 1355. That is ~690 MiB of boot-to-boot
   spread on the binding rank against a 331 MiB margin — the margin is inside
   the spread.
2. **The host budget is now the tighter constraint** and the gate has no term
   for it: 15.9 GiB MemAvailable on the arm against 23.5 on ship, in a
   container with a ~45 GiB real working margin.
3. **The gain is +25.5 %.**

### NEXT SHIFT, IN ORDER

1. **Hold the cut across boots.** Three more windows on `40,12,12`. The
   question is no longer "does it hold" but "how often" — the margin is
   inside the observed boot-to-boot spread, and one window cannot answer it.
2. **Give the gate a HOST term.** It prices VRAM only. The planner cut costs
   ~8 GiB more host memory than the ship config, in a container whose margin
   is ~45 GiB and whose OOM killer has fired 9 times. This is the mechanism
   that killed s50's window.
3. **Test the transient table ACROSS a cut boundary** — apply this boot's
   table to a different cut and report the held-out error, as s50 did for
   the at-rest term. Until then the gate is shown consistent, not predictive.
4. Itemize the 1250 MiB cut-shaped residual (s50's item 3, still open);
   calibrate `seam_staging_mib` (fifth shift); arm B (fifth shift).

## 6. STATE ON EXIT

Serving UP on 30030, SHIP config, from `/spinning/wt-631-routea/python` at
this shift's tip, verified with a real generation and not health alone.
I stopped serving twice (12:36 for the arm, 13:15 for the restore) and
brought it back at 13:19. **Nobody owes a restore.** Router 30099 never
touched. All three cards were released between boots (20053 / 32086 / 20052
MiB free). Heartbeat stopped before the holder was released.
