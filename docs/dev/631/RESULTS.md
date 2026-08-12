# #485 under the flip: the gain survives, and the wall was two bugs

Successor 49, 2026-08-12. Tree `/spinning/wt-631-routea`, branch
`feat/route-a-631`, fixes at `6c7e8a1411`. All boots replay the same captured
ship argv/env N48 used (`../s485/ship_argv.txt`, `ship_env.txt`).

## 1. The headline

**The planner cut runs under `--enable-phase-flip` and keeps its win.**

| arm | cut / attn | flip | pool | depth 179200 median | vs its control |
|---|---|---|---:|---:|---:|
| A control (ship) | 14,10,8 / 7,5,4 | **ON** | 280000 | 98.276 s (N48) | — |
| **C planner** | **42,11,11 / 10,3,3** | **ON** | 280000 | **66.072 s** | **+48.7 %** |
| A control (ship) | 14,10,8 / 7,5,4 | off | 280000 | 95.436 s (N48) | — |
| C planner | 42,11,11 / 10,3,3 | off | 280000 | 63.246 s (N48) | +50.9 % |

**+48.7 % with the flip on, against +50.9 % with it off.** The honest pair is
reported; the flip-on number is the shipping number and it is the smaller one.
The flip costs the cut about two points of its advantage, which is the same
direction and roughly the same size as what the flip costs the control
(98.276 vs 95.436, i.e. 3.0 %).

Instrument: 5 scored samples, spread **0.59 %**, 0 rejected for cache hits,
every prompt a unique random prefix with `/flush_cache` before each. That is a
same-boot floor for this arm. The CONTROL is N48's boot, not mine — a
cross-boot comparison, admissible here only because N48 measured cross-boot
reproducibility on that exact control at **0.09 %** (98.369 vs 98.276, two
boots, two pools). A same-boot control was not run; the window went to the
ship-config confirmation instead, and that is the weakest link in this table.

The boot completed **42 flips, 0 abandoned, 0 tracebacks**, health 200
throughout, and survived five 179200-token prefills — the exact operation that
killed both earlier arm-C boots.

## 2. What the wall actually was — two bugs, neither of them the seam

C34 recorded one wall ("the flip seam locks the cut out"). It was two
independent defects that the arm design had bundled. See `CONFOUND.md` for
the experiment; the fixes are in `6c7e8a1411`.

**(a) The seam wedge follows the TOKEN VECTOR, not the cut.** Same cut, same
pool 340000, same budgets, flip on — only `SGLANG_UNEVEN_TOKEN_VECTOR`
changed from `10,3,3` to the ship `7,5,4`: 185 group abandons became **0**,
and the instance reached `/health` 200. The planner cut was never locked out
of the flip.

**(b) `KvRowCap` double-booked its withheld ids on `clear()`.** Measured
twice, and the arithmetic is exact:

| boot | pool | vector | total−available | withheld | ratio |
|---|---:|---|---:|---:|---:|
| N48 arm C | 280000 | 10,3,3 | 12783 | 25566 | **2.000** |
| mine | 340000 | 7,5,4 | 81640 | 163280 | **2.000** |

An exact factor of two across two pools and two token vectors is a duplicate
booking. `_apply` accumulates and was wired as the on-CLEAR hook as well as
the on-free hook; `clear()` rebuilds `arange(1, size+1)`, so the ids above the
cap are taken a second time. Only a configuration that ENGAGES the cap can
see it — which needs a corridor deficit — so the ship cut never does, and
doubling zero is invisible. Fixed; the crash signature changed from the pool
invariant to a genuine OOM, which is how the fix was confirmed on metal.

## 3. Why it is pool 280000 and not 340000

At pool 340000 the planner cut boots, flips 6x clean, serves — and then dies
on the first deep prefill with `cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`.
That is not an accounting artifact: rank0 (the 5090) sits at **606 MiB free**
against the 1024 MiB corridor law.

| boot | pool | gpu0 min | **gpu1 min (rank0/5090)** | gpu2 min | breaches |
|---|---:|---:|---:|---:|---:|
| N48 arm A control | 280000 | 4361 | **7212** | 2623 | 0 |
| mine, planner cut | 340000 | 9369 | **608** | 6999 | 135 |
| mine, planner cut | 280000 | 5335 | **976** | 5861 | **6 of 3568** |

The cut puts 10 of 16 attention layers on rank0 against the ship cut's 7, so
the same pool costs ~43 % more VRAM there. At 280000 it very nearly holds:
**976 MiB minimum, 48 MiB under the law, 6 samples of 3568 (0.17 %)**. Close
is not held. This is a real corridor breach and it is the reason for the
verdict below.

## 4. Wire-or-gate: GATE — and the blockers are now specific

The brief's condition was "wire if bootable + gain holds + window clean". Two
of three are met and the third is not:

* bootable — **yes**, at pool 280000, with 42 clean flips;
* gain holds — **yes**, +48.7 % under the flip;
* corridor clean — **no**, 6 samples 48 MiB under the law.

`solve_pp_cut` stays unwired, for reasons that are now numbers rather than a
category:

1. **`seam_staging_mib` exists but has no calibrated value.** The term was
   added this shift (law 23 consumed: the solver now maximizes the tightest
   RUNNABLE headroom, residency minus peak, and `validate_pp_cut` says "it
   FITS AT REST and still cannot run"). Its default is 0, and a gate that
   models transient demand as zero is the same gate that certified arm C.
   Two measured demands exist (4881 MiB at attn 10/16 pool 340000; 4343 MiB
   at attn 6/16 pool 280000) and they do not separate the shape — and this
   shift's confound showed the demand follows the ARENA, not the attention
   split, which is the opposite of what the layer-count reading predicted.
   Calibrate before wiring; do not fit two points to a formula.
2. **The residency verdict itself mispredicts rank0.** It called arm C
   feasible with 2617 MiB spare; metal at pool 340000 gives 606 MiB free and
   an OOM. That is a residency error, not a transient one, and it is
   unexplained.

**What IS shippable today:** the cut as a MANUAL, validated configuration.
`--pp-stage-ratio 42,11,11 --pp-attn-stage-ratio 10,3,3` at
`--max-total-tokens 280000` with the ship token vector delivers +48.7 % deep
prefill under the flip. It needs a little more corridor room than it has.

**Size the reduction from the measurement, not the layer count.** rank0 holds
10 of 16 attention layers, so KV there "should" cost 0.0195 MiB/token, which
says 48 MiB costs ~2500 tokens. The two boots measured 608 MiB free at pool
340000 and 976 MiB at 280000 -- **0.0061 MiB/token, 3.2x shallower**, the same
direction of error as the residency misprediction above and probably the same
cause. On the measured slope, ~1100 MiB of margin needs roughly 20000 tokens
off, i.e. **pool ~260000**. Two points across two boots, and minima read a
load state (C7), so this aims the next boot rather than calibrating anything
-- but the theoretical slope would have under-shot by 8x.
