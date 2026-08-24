# NOTE 856 — the seam cost ledger, measured

Everything here is measured from `/spinning/evidence-665-f1/boot_w25_0824_1125.log`
(W25, pin `63c3c0dd00`, 11 flip epochs x 3 ranks, 85 requests all HTTP 200).
No number in this file is estimated, and where a question could not be answered
from the capture that is said instead of filled in.

## The question this answers

> "und der tp to pp und pp to tp dauer warum genau so lang? wenn beide vor dem
> flip gestoppt werden kann doch alles auf einmal parallel runter/hochgeladen
> werden. immer? warum dauert das 16 waves / 116k live slots? 11,6s?"

**The 16 waves and the 116502 live slots are not what costs the time.**

## 1. The five-term decomposition, from the seam's own DONE line

`PHASE-FLIP DONE` already carries the split. Epoch 10, binding rank PP0,
`tp_to_pp`, the 11.6 s flip the question is about:

    total 11646.8 ms = read 10.3 + exchange 502.0 + write 325.2
                     + movers 10534.2 + cutover 325.2-ish

    movers = 90.4 %   read+exchange+write = 7.2 %   cutover = 1.6 %

The per-segment census (`SeamCensus.format_timing_line`, emitted at INFO) is
sharper still, and it is the single most important line in the whole capture:

    seam-census] timing tp_to_pp rank 0: 10466.8 ms across 448 segment(s),
      worst 'refill_highwater->weights_refill' 9516.2 ms (91% of the walk)
      kv_write->gdn_state    174.3
      cutover->done          152.4
      flip_writeback->hicache_quiesce  74.8
      kv_pack->kv_local_read  57.4 | 28.6 | ...

**91 % of a `tp_to_pp` flip is ONE segment: the weights-arena refill read.**
Three consecutive `tp_to_pp` flips measured 9516.2 / 9496.5 / 12108.2 ms in
that one segment, against totals of 10466.8 / 10568.0 / 13181.2 ms.

## 2. The live slots are not the driver — 947x the volume, 1.26x the time

Across the window, `live slots` grew from 123 to 116502 while the flip barely
moved:

    epoch  dir        live slots   total ms
      1    pp_to_tp         123     5077.9
      3    pp_to_tp       22642     5211.0
      5    pp_to_tp       45832     6537.0
      7    pp_to_tp       70540     6523.8
      9    pp_to_tp       91788     6545.4
     11    pp_to_tp      116502     6416.0

947x the rows, 1.26x the wall time. At the LARGEST reshard (epoch 11, PP0) the
entire KV movement was:

    sent 509600 cells / 995.31 MiB, received 422416 cells / 825.03 MiB
    read 19.9 ms + exchange 486.0 ms + write 394.8 ms  =  901 ms

901 ms, i.e. **14 % of a `pp_to_tp` flip and ~7 % of a `tp_to_pp` one**. The
GDN/mamba leg is smaller again: `PHASE-FLIP-GDN moved 1 slot(s)`, 11.69-18.70
MiB, 174-201 ms.

## 3. The waves are not a serialization defect

`16 seam wave(s)` is derived, not tuned: `restore_first_wave_count(layer_map)`
(`layers/dcp/phase_flip_plan.py:157`) returns `max(1, n_layers)` — one wave per
attention layer. An override exists (`SGLANG_FLIP_SEAM_WAVES`, read raw at
`managers/phase_flip_runtime.py:3707`, NOT registered in `environ.py`) and the
code's own comment says the payload leg stops shrinking around W=8. Given §2
that lever is worth at most a fraction of 901 ms.

The three ranks are already parallel. Totals agree to ~40 ms (10466.4 / 10470.5
/ 10509.7) while the movers legs differ by ~600 ms (9691.8 / 9097.5 / 9162.5),
and the peers absorb PP0's overrun as **cutover wait** (1313.3 / 1220.8 ms
against PP0's 225.3 ms). That is a rendezvous, exactly as #690 recorded ("the
gdn_state spread is a rendezvous WAIT, not serialization"). **The critical path
is PP0's refill alone.**

## 4. So the seam is a weights-refill RATE problem

`PHASE-FLIP-BOOT REFILL` reports the leg as bytes and a rate. Same rank, same
bytes, opposite directions:

    direction   rank PP0 bytes    rate              time
    pp_to_tp    15925.8 MiB       3214-3915 MiB/s   4.07-4.96 s
    tp_to_pp    16362.7 MiB       1351-1723 MiB/s   9.50-12.11 s

**2.7 % more bytes, 2.2x slower, 2.3x longer.** The two 3080s show the same
shape (8573.8 MiB at 2251-2903 MiB/s outbound vs 8961.3/9481.6 MiB at 874-1081
MiB/s inbound). Aggregate over three ranks: `tp_to_pp` moves 34.8 GiB at
~3.4 GiB/s; `pp_to_tp` moves 33.1 GiB at ~7.3 GiB/s.

### What the asymmetry is NOT

`weights_arena.py`'s own docstring records the three rate regimes measured on
this rig: **O_DIRECT 8304 MiB/s**, **buffered 2595 MiB/s** (the ARC copy is a
second pass and the ARC is capped at 5 GiB against a 16 GiB image), **page-fault
path 1108 MiB/s**. The observed rates straddle the lower two. But the obvious
readings were checked and all three fail:

* **Not a path fallback.** `SGLANG_PHASE_FLIP_REFILL_STAGED` defaults `True`
  (`environ.py:348`) and is not overridden in the boot; `arena_refill`
  (`weights_arena.py:1093-1095`) dispatches on the image being file-backed, not
  on direction. Both directions take `_staged_file_refill`.
* **Not a missing fd.** Both `#802` warning paths (`could not open a read fd`,
  `O_DIRECT unavailable`) appear **zero times** in the 3.45 MB capture, against
  9 `flip host image FILE-BACKED` registrations.
* **Not the O_DIRECT alignment cliff.** #809 flagged that an unaligned offset
  silently regresses 8304 -> 2595 MiB/s, but the loop only ever issues aligned
  offsets: chunks are 32 MiB multiples of `_DIRECT_ALIGN=4096` and `want` is
  rounded down to the alignment (`weights_arena.py:600-607`); only the trailing
  checksum tail is buffered by design.
* **Not the pre-#802 fault path either**, on #802's own discriminator: on the
  fault path rank rates CONVERGE (821 / 775 MiB/s on links differing 1.80x). In
  W25 both directions DIVERGE with the link (`tp_to_pp` 1.59x, `pp_to_tp`
  1.40x), which says a real transfer took place in both.

### The instrument is what is actually missing

The refill leg reports **one aggregate MiB/s**. That single number cannot
separate storage-read-bound from PCIe-H2D-bound from ARC-copy-bound, and the
read (`os.preadv`) and the H2D (`copy_` on a stream, depth 2) are pipelined so
the aggregate is `min(read_rate, h2d_rate)` with no way to see which bound.

**That is the #851 class defect — one number with several meanings — sitting
inside the term that is 91 % of the seam.** Attributing the 2.5x direction gap
is therefore not a desk question; the next proof window needs the leg split
into its two rates before any fix is chosen. Recorded as an OPEN root, not
guessed at: two independent readers (the strand and an explorer sweep) failed to
attribute it from code, and the search sets are named above.

### CLOSED on W26 metal (2026-08-24) — STORAGE-BOUND

The split ran. Pin `9effad7f0d`, log
`/spinning/evidence-665-f1/boot_w26_0824_1354.log`, 11 flip epochs, 39 paired
`REFILL` + bound samples. Full write-up in `NOTE_856_refill_storage_bound.md`
and `/spinning/gpu-arb/W26-RESULT-856-refill-root.md`.

    dir       rank    n      MiB   tot_s  read_s   h2d_s  drain   MiB/s  read%
    pp_to_tp  PP0     7  15925.8   4.452   4.230  0.0073  0.002    3612  99.8%
    pp_to_tp  PP1     7   8573.8   3.292   3.014  0.0051  0.006    2626  99.8%
    pp_to_tp  PP2     7   8573.8   3.328   3.052  0.0049  0.003    2596  99.8%
    tp_to_pp  PP0     6  16362.7   7.601   7.384  0.0043  0.002    2156  99.9%
    tp_to_pp  PP1     6   8961.3   6.743   6.460  0.0030  0.007    1330 100.0%
    tp_to_pp  PP2     6   9481.6   6.895   6.604  0.0037  0.002    1376  99.9%

`verdicts observed: {'STORAGE-BOUND'}` — every sample, both directions, all
three ranks. The read is 99.8-100.0 % of the accounted leg and the H2D wait is
**3-7 ms**. The link never waits; it idles at ~21-35 % of the rate the same
card demonstrably reaches in the same window (`nvidia-smi dmon` peak `rxpci`
6893 / 10599 / 4683 MB/s).

**The 2.5x direction gap is a property of the FILE, not of the code.** A
standalone O_DIRECT probe (no CUDA, no server, same 32 MiB chunk as the refill
loop) read both PP0 images back to back in one process:
PP image **3186.4 MiB/s**, TP image **9985.1 MiB/s** — 3.1x, same flags,
seconds apart. Compression is ruled out (on-disk ratios 1.09x vs 1.11x); the
surviving explanation is ARC residency, `c_max` being 5.0 GiB against ~29 GiB
of weight images. So §4's three eliminations above were all correct — the term
they were hunting was never in this code path.

**And the read side does not scale**, which forecloses the obvious fix: the
same probe at 1 / 2 / 4 / 8 concurrent readers returned 3186 / 3659 / 3688 /
3655 MiB/s on the PP image — **1.15x and flat**. The pool is saturated by a
single stream, not latency-bound, so parallelising `preadv` buys ~15 %.
Deeper pipelining buys nothing either (`h2d_wait_s` is already ~0.005 s).
The only lever that reaches the target is not reading from disk at all, i.e.
#809 §8's pinned share.

**Its sizing A/B (#809 §9) still has NOT run.** W26 attempted it twice and both
attempts were OOM-killed in the LAUNCH phase, before any flip (`grep -c REFILL`
= 0 in both `boot_w26pin_*.log`; cgroup `oom_kill` 3 -> 4). That failure is
itself informative: whole-image pinning (~68.7 GiB) plus the weights-load
page-cache spike does not fit this container, so the fix must be a PARTIAL
share, and a pre-launch `free -g` gate does not protect the box because the
spike builds during the load. Details in `NOTE_856_refill_storage_bound.md`.

## 5. Consequence for the chosen design

The user's decision — the flip carries NO KV, it is loaded from HiCache —
is validated by §2 and §3 rather than contradicted: the KV+GDN movement is
~901 ms of an 11.6 s seam, and removing it entirely leaves ~10.5 s. So:

* the seam-TIME prize of the HiCache route is ~0.9 s;
* its real prize is FUNDING — the `tp_to_pp` staging reserve is 2339.11 MiB on
  PP0, and W25 refused 33 flip arms (25 on the staging rate limit) with 17
  FLIP ABANDONED. Removing seam-sized staging from the flip path is what those
  refusals are about;
* the remaining seam is the weights refill, and it is where §4's open root
  lives.

One supporting argument was checked and does NOT hold in this window, recorded
so nobody re-derives it: pending prefill across 181 samples was median 1911,
p90 21339, **max 23228 tok**, against resident live slots reaching 116502. In
W25 the resident set was ~5x the instantaneous backlog, not a minority of it.
The no-carry decision does not need that argument — carrying the resident set
costs seam-blocking time and seam-sized staging to pre-warm rows that
read-through serves anyway.

## 6. The price half (#856 b) — a separate, PROVEN defect

`C` is defined by this module's own model (`phase_policy.py:82`) as
`round-trip flip cost, seconds`, and `break_even_tokens` refuses a bad premise
by saying flipping "never repays the {flip_cost_s}s round trip". But
`observe_flip_leg` fed ONE LEG per sample, and both directions went into ONE
EMA.

Reproduced exactly. Replaying PP0's eleven `PHASE-FLIP DONE` totals through a
single estimator at `ALPHA=0.3`:

    5.0779  6.6944  6.2494  7.5450  7.2426  9.0241
    8.2740  9.2457  8.4356  9.3990  8.5041

and the boot's own decision lines printed `N=15853 / 18110 / 18464 / 18614` at
exactly the samples pricing to 7.2426 / 8.2740 / 8.4356 / 8.5041. The blend
converged to 8.50 s — **below every `tp_to_pp` leg and above every `pp_to_tp`
leg**, i.e. to neither — while the true round trip was 11.6 + 6.4 = 18.06 s.
The bar also oscillated ~2000 tok with flip-direction parity (8.50 after a
`pp_to_tp`, 9.40 after a `tp_to_pp`), which is an artifact of the blend and
says nothing about cost.

#819's own closing sentence is the rule it broke one level up: *"a component
and its container are different quantities and an EMA fed both alternately
converges to neither."* Two directions are different quantities too.

**The correction RAISES the bar** (C 8.50 -> 18.06 s, N 18614 -> ~39500), so it
makes TP-stickiness on 16-20k prompts more correct, not less. That is stated
plainly rather than softened: the remedy for a bar that is too high is a
cheaper seam, not a permanently under-priced one. Once §4's root is fixed the
same arithmetic lowers the bar honestly, and `dN/dC = 1/(1/1681.0 - 1/7245.5)
= 2188.8 tok/s` says by how much.

### What shipped

`RoundTripFlipCost` holds ONE `FlipCostEstimator` PER LEG and sums them.
The leg estimator is reused, not rebuilt, so every property #677 pinned on it
holds per leg — including that it tracks DOWN as readily as up, which is what
makes a future seam fix actually lower the bar instead of latching high.
The seed is split in half so an uncalibrated instance values exactly the
round-trip seed and the pre-#856 path is unchanged.

Provenance gained a third word. `flip_cost_measured()` is a boolean over a
quantity with three states, and it printed the middle one as "measured";
`flip_cost_provenance()` now returns `seed` / `half-measured (<leg> only)` /
`measured`. Same class of fix as #853(i) on the exposure gate and #854 on the
economy detector, one layer further in.

### Also found, NOT fixed here

`observe_flip_leg` is called only from the flip-COMPLETION branch
(`scheduler.py:6185-6216`). A boot whose flips are all refused or abandoned
never calls it, so it prices off the seed for the whole session — and the #777
staleness WARNING is gated on the same event, so such a boot gets neither a
reprice nor a warning. W25 did not manifest it (33 completed cutovers), but the
shape is the #851 silent-zero again. Recorded as an open follow-up.

Also recorded: `C`, `X` and `P` are env-var-only
(`SGLANG_PHASE_POLICY_FLIP_COST_S` etc.), never promoted to `ServerArgs`, while
#781 promoted every one of their siblings (`min_dwell_s`, `pp_window_s`,
`tp_decode_floor_s`, `decode_stall_slo_s`, `decode_contention`, `drain_mode`).
