# NOTE 702: the prefill-speed PP cut, solved on the completed enumeration

The user's open question (2026-08-16): more layers on the 5090, pool price
stated. Solved 2026-08-18 on the harvest lineage with the #723-fixed
enumeration (per-depth Pareto sets — the pool-dominant tails the old
speed-argmin silently discarded are in the table). Runner:
`tools/solve_prefill_frontier_702.py` (pure CPU, anchors injected).

## Anchors and their provenance (honest limits FIRST)

* `ms_per_layer = (1.7571, 7.740, 7.275)` — measured, instrument boot
  `c3e94878ff`. `fixed_ms = 0`: every speed ratio below is the OPTIMISTIC
  end of the family until the two-point timing calibration lands
  (slice 1a-i). Treat ratios as upper bounds.
* Pool rows: #707 seam closed form anchored at the instrument boot's
  `(28,20,16)` — the ONLY boot-measured pool row. Every other row
  extrapolates across a layout shift of `free_at_measure` with a holdback
  KNOWN to vary by rank (45.1/44.1/60.3 %). Exact arithmetic, extrapolated
  anchor.
* Links x8 = 13 GB/s (5090, PP0/PP1), x4 = 6.4 GB/s; gather 24.09
  MiB/attn-layer — measured.
* Noise floor 0.141 (A-vs-A). Speed gains under it are labeled: arithmetic
  may prefer them, no measurement could confirm them.
* NOT modeled: decode impact (this is the prefill objective), quality
  (cut-invariant), the flip's TP phase (its own vector, separate flag).

## The incumbent, verified from the boot's own argv

`argv_735_composite.txt:13` → `--pp-stage-ratio 31,17,16`, attn `7,5,4`
(matches the natural every-4th pattern). NOT the instrument boot's
`(28,20,16)` and NOT rev5's `[30,18,16]`. Its closed-form pool: 436,275
tokens (the instrument observation 436,278 reproduced to rounding).
Pipelined stage time 131.58 ms; its own gather overhead 7.4 % → NET
baseline 0.931 in the table (relative NETs below divide by it).

## The frontier (speed = stage-time ratio vs live incumbent; NET includes gather; PIPE = unreachable without the §4.2g pipelining lever; noise = under the floor)

```
           cut       attn      ms  speed    NET      pool  poolx  ovh% flags
  (44, 10, 10) (11, 2, 3)   77.40  1.700  1.333   172,791  0.396  27.5 PIPE
   (45, 9, 10) (11, 2, 3)   79.07  1.664  1.310   148,243  0.340  27.0 PIPE
  (43, 10, 11) (10, 3, 3)   80.03  1.644  1.385   165,601  0.380  18.7 PIPE
   (46, 7, 11) (11, 2, 3)   80.83  1.628  1.287   123,695  0.284  26.4 PIPE
   (47, 6, 11) (11, 2, 3)   82.58  1.593  1.266    99,146  0.227  25.9 PIPE
   (48, 5, 11) (12, 1, 3)   84.34  1.560  1.222   111,275  0.255  27.6 PIPE
  (42, 11, 11) (10, 3, 3)   85.14  1.545  1.411   192,604  0.441   9.5
   (49, 4, 11) (12, 1, 3)   86.10  1.528  1.203    88,772  0.203  27.1 PIPE
  (41, 11, 12) (10, 3, 3)   87.30  1.507  1.327   219,607  0.503  13.6
   (50, 2, 12) (12, 1, 3)   87.86  1.498  1.184    66,270  0.152  26.5 PIPE
   (51, 1, 12) (12, 1, 3)   89.61  1.468  1.165    43,767  0.100  26.0 PIPE
    (52, 4, 8) (13, 1, 2)   91.37  1.440  1.128    59,223  0.136  27.6 PIPE
  (40, 12, 12) (10, 3, 3)   92.88  1.417  1.327   246,610  0.565   6.7
    (53, 3, 8) (13, 1, 2)   93.13  1.413  1.111    38,451  0.088  27.1 PIPE
  (39, 12, 13)  (9, 3, 4)   94.58  1.391  1.192   246,824  0.566  16.7
    (54, 2, 8) (13, 1, 2)   94.88  1.387  1.095    17,680  0.041  26.6 PIPE
    (56, 4, 4) (14, 1, 1)   98.40  1.337  1.048    14,607  0.033  27.6 PIPE
  (38, 13, 13)  (9, 3, 4)  100.62  1.308  1.192   276,827  0.635   9.7
  (37, 13, 14)  (9, 3, 4)  101.85  1.292  1.119   306,830  0.703  15.5
  (36, 14, 14)  (9, 3, 4)  108.36  1.214  1.119   336,833  0.772   8.6
  (35, 14, 15)  (8, 4, 4)  109.12  1.206  1.053   348,352  0.798  14.5
  (34, 15, 15)  (8, 4, 4)  116.10  1.133  1.053   382,106  0.876   7.6 noise
  (33, 15, 16)  (8, 4, 4)  116.40  1.130  0.995   415,859  0.953  13.6 noise
  (32, 15, 17)  (8, 3, 5)  123.68  1.064  0.918   397,957  0.912  16.0 noise
  (31, 16, 17)  (7, 4, 5)  123.84  1.062  0.918   397,957  0.912  15.8 noise
  (32, 16, 16)  (8, 4, 4)  123.84  1.062  0.995   436,275  1.000   6.7 noise
  (30, 16, 18)  (7, 4, 5)  130.95  1.005  0.873   343,951  0.788  15.1 noise
  (29, 17, 18)  (7, 4, 5)  131.58  1.000  0.873   343,951  0.788  14.5
  (30, 17, 17)  (7, 4, 5)  131.58  1.000  0.918   397,957  0.912   9.0
  (31, 17, 16)  (7, 5, 4)  131.58  1.000  0.931   436,275  1.000   7.4 LIVE
  (32, 17, 15)  (8, 4, 4)  131.58  1.000  0.944   449,613  1.031   5.9
  (28, 17, 19)  (7, 4, 5)  138.22  0.952  0.833   289,946  0.665  14.3
  (28, 18, 18)  (7, 4, 5)  139.32  0.944  0.873   343,951  0.788   8.2
  (29, 18, 17)  (7, 4, 5)  139.32  0.944  0.895   397,957  0.912   5.6
  (30, 18, 16)  (7, 5, 4)  139.32  0.944  0.883   436,275  1.000   7.0
  (31, 18, 15)  (7, 5, 4)  139.32  0.944  0.883   478,888  1.098   7.0
  (28, 19, 17)  (7, 4, 5)  147.06  0.895  0.850   397,957  0.912   5.3
  (29, 19, 16)  (7, 5, 4)  147.06  0.895  0.839   436,275  1.000   6.6
  (30, 19, 15)  (7, 5, 4)  147.06  0.895  0.839   503,782  1.155   6.6
  (28, 20, 16)  (7, 5, 4)  154.80  0.850  0.800   436,275  1.000   6.3 MEAS
  (29, 20, 15)  (7, 5, 4)  154.80  0.850  0.800   462,920  1.061   6.3
```

Decoupled pool (#704b token-sharded KV): **513,875 tokens, exactly
cut-independent** — under that regime the pool column stops constraining
the cut entirely, and only the PIPE flag limits depth.

## Bootability (arming floor, #676/#707 — priced, not hidden)

29 tails across the enumeration are REFUSED by the seam closed form
(rank's free column no longer clears the arming floor → the layout cannot
arm a flip): leads 28–31 lose 2–3 extreme tails each, leads 55/57/58/59
lose 4–8. Every row IN the table is arming-feasible by construction — the
provider refuses, the solver skips, and the refusal counts are stated here
so the pruning is visible.

## Recommendations (NET vs the incumbent AS SERVED, i.e. ÷0.931)

1. **Prefill-speed cut: `(42,11,11)`**, attn (10,3,3). Predicted NET
   prefill **1.52x** vs the live `(31,17,16)` (stage-time 1.545x, gather
   overhead 9.5 %), arming-feasible, above the noise floor, needs NO
   pipelining lever. **Pool price, stated: coupled pool 192,604 tokens =
   0.44x of the incumbent's 436,275** — a max-context 262,144-token
   request no longer fits in the coupled regime; this cut wants #704b
   decoupled KV (513,875, cut-free) or an accepted context cap.
2. **Balanced cut: `(40,12,12)`**, attn (10,3,3). Predicted NET **1.43x**
   at pool 246,610 = 0.57x, overhead 6.7 % (LOWER than the incumbent's
   7.4 %). Same caveats, milder price.
3. **Zero-risk correction (not a speed rec): `(32,17,15)`** — identical
   stage time to the incumbent, +3.1 % pool (449,613), lowest overhead in
   its speed class (5.9 %). The speed delta is below measurement
   resolution; the pool gain is closed-form arithmetic. A free config flip
   whenever a boot is happening anyway.
4. `(32,16,16)` strictly dominates the incumbent on paper (1.062x at equal
   pool) but sits UNDER the noise floor — running it as an A/B would
   measure noise; it is listed, not recommended.

## Boot-ticket lines (window list)

```
702-speed  : composite argv unchanged except --pp-stage-ratio 42,11,11
             acceptance: 72k-prefill probe >= ~1.4x incumbent tok/s
             (A-vs-A floor first); pool >= ~190k reported at boot;
             no seam-arming refusal in the window; flips still commit.
702-balance: same with --pp-stage-ratio 40,12,12; >= ~1.3x; pool >= ~240k.
Both after the harvest boot's acceptances; one cut per boot, never two
changes in one window.
```
