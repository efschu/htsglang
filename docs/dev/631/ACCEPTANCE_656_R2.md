# ACCEPTANCE 656 — RE-RUN R2

Branch `feat/acceptance-remediation-656` @ `d644c266ec`, rebased on
`integration/r2` @ `6d169c04ab` (MERGE-R8). Boot `boot_v3`, **one boot, one
continuous log**, argv identical to the #656 acceptance and to R1's remediation
boot — same TP vector `30,16,18`, same budget vector `31583,15750,18205`, same
`SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB=384` — so the three runs are an A/B/C on the
CODE and on nothing else.

Evidence: `/spinning/evidence-631/remediation-656-r2/`. Every number below comes
from `extract.sh` reading the serving log and the corridor CSV, not from hand
counting. Raw: `EXTRACT_V3.txt`.

Window: ready 16:15:58Z, load 16:16:06Z → 17:11:32Z = **55.4 minutes
continuous**, plus the deep-probe run inside the same log. Corridor sampled
continuously at 100 ms for **56.2 minutes / 33648 samples per card**.

---

## 1. AXIS-BY-AXIS VERDICT

| # | axis | verdict | evidence |
|---|---|---|---|
| 1 | auto phase-flip controller, no manual flips | **PASS** | `--phase-flip-policy auto`; **364 cutover events, 182 pp→tp / 182 tp→pp**, 0 abandons, 0 CANNOT FUND, 0 manual flips |
| 2 | CUDA graphs ON + NEXTN spec ON, verified | **PASS** | `Capture draft verify CUDA graph begin` on all 3 ranks; **600 of 609 decode batches `cuda graph: True`** (the 9 are the pre-capture batches of the first TP window); accept length n=163, min 2.09 / median 2.53 / max 3.43 / mean 2.593 |
| 3 | MAX-KV derived pool + corridor ≥ 1024 continuous | **PASS** | pool **585730** derived and uncapped, seam-warm on all 3 ranks; corridor minima **1070 / 2139 / 1466 MiB**, **0 of 33648 samples below the law on every card** |
| 4 | bs1 MAX-KV, idle-vacate fires for slots 2-4 | **PASS (amended criterion) + SPEC NOTE** | **0 vacate lines**, as on every flip boot: structurally unreachable under `pp_size>1`. The pressure ladder does fire — **2181 spill-rung lines**. The spec note below is carried in full |
| 5 | YaRN bs1 decode past 262144 | **PASS** | **300026 tokens, exact planted answer** past `max_position_embeddings` 262144. (280026 returned empty — defect B, §3) |
| 6 | real agent load through router 30099 + scripted bs≤4 | **PASS** | `ROUTER-LEG-OK` via `/v1/messages`; **84 of 84 router requests HTTP 200, 0 failures**, concurrent with the direct load for the whole window; **345 direct generations, 25 soak cycles, 0 NON-JSON failures** |
| 7 | zero tolerance: no traceback, no breach, no wedge | **PASS** | **0 tracebacks, 0 scheduler exceptions, 0 SIGQUIT, 0 KvReshardError, 0 "NOT A CHECKSUM", 0 wire frame divergences, 0 corridor breaches, 0 wedges, 0 CANNOT FUND** |

**7 of 7.** The three axes the #656 acceptance failed (3, 4, 7) are the three
this shift worked on, and axis 4 passes on the amended criterion the re-run was
given, with its spec note carried honestly rather than substituted away.

### The three runs side by side

| | acceptance | R1 remediation | **R2 (this)** |
|---|---|---|---|
| cutovers | 320, then died | 218 + 12 frame abandons | **364, 0 abandons** |
| terminal state | SIGQUIT, instance dead | wedge, `/health` 503, no tokens | **serving, load driver ran to completion** |
| corridor minimum | 886 / 1941 / 1304 | 1012 / 2057 / 1370 | **1070 / 2139 / 1466** |
| samples below the law | 25, in 5 episodes | 2, in 1 episode | **0** |
| tracebacks / SIGQUIT | 2 / 1 | 0 / 0 | **0 / 0** |
| pool | 597106 | 578390 | 585730 |

---

## 2. WHAT THIS RUN DOES **NOT** PROVE

Stated first, because the axis table above is the kind of clean sheet that
invites over-reading.

**The two mechanisms this shift built did not fire on this boot.**

```
cap agreement moves:     0
purity stand-downs:      0
yields WITHHELD:         0
yields taken:            0
corridor-bounded recov.: 1
```

One corridor-bounded recovery did occur (PP1, 16:25:13Z, 558489 of 585730 rows
— a 27241-row shortfall, the exact divergence class R1 measured). It did not
open a divergence, because the next `pp_to_tp` shrink caps every rank to one
agreed absolute row target and therefore **levels the group by construction**.
That is why the window is narrow, why R1 saw only two episodes in 33 minutes,
and why a clean boot is cheap to get.

So **boot_v3's 0 divergences is a no-regression result, not a positive proof of
the agreement.** The positive proofs are elsewhere and are stated as such:

* **the levelling works** — boot_v2, 16:00:30Z: three ranks at 579870 / 579722 /
  578606 exposed rows converged to 578606 in ONE round, and every field of
  every POOL CENSUS line was identical from the next round on;
* **the wedge is closed** — boot_v2, 16:01:24Z: a persistent divergence blocked
  `tp_to_pp`, all three ranks relaxed purity for PREFILL simultaneously, and the
  instance kept serving (`/health` 200, tokens flowing) where R1's boot answered
  503 with none;
* **the desk red arms fail when disarmed**, including the reproduction of the
  metal 40404-row divergence on a CPU fixture.

Likewise the corridor result is attributable to the pool sizing and to the gate
never having to yield (0 yields taken), **not** to the yield-withholding this
shift added. That mechanism is desk-proven and has not fired on metal.

**Power, on the C22 half.** At the acceptance's observed 1-in-320 rate, 364
clean cutovers happen about 32% of the time on an instance where nothing was
fixed. A 95% claim needs ~957. This run is better evidence than the
acceptance's 320 and it is not a 95% claim.

---

## 3. DEFECT B — most of its evidence base was not the defect

Counted across all prior boots, every load-driver prompt above 21 tokens
returned a one-token completion — 4001, 16001, 32001, 64001, at 100%, on both
boots. Too uniform for an intermittent defect, and the difference is one line:
those rungs are pure filler and **ask nothing**.

The control (boot_v2, 15:56Z), same filler and count with one question added:

```
WITHOUT a question:  in=4001  completion=1
WITH    a question:  in=4026  completion=12  cached=0  text='...BANANA47'
```

**They are the model emitting EOS on a prompt that asks nothing**, not a serving
defect. Both the acceptance and R1 counted them as content — right in outcome,
wrong in reason. Register row `C-B1`.

What survives is only the probes that DO ask, and there the signal is sharp and
reproducible: pooled with the probe run below, **~280k has returned empty on 5
of 7 COLD attempts** across four boots, while 240016, 250026, 300016/300026,
332532 and 338916 are 12 of 12 exact. A ceiling cannot explain that — a ceiling
cannot be passed by going deeper. The mamba-floor fix from MERGE-R8 is on the
line in every one of those runs and did not change it.

The 2-factor probe (FILLER × DEPTH) is §4.

---

## 4. THE DEEP-PROBE RUN — the confound is resolved, and it is DEPTH

12 probes, FILLER (unique | repeated) x DEPTH (240016 | 280012 | 300020), 2
each, every one planting a secret and asking for it, sentence counts computed
with the checkpoint's own tokenizer and the depth MEASURED per probe. Raw:
`deep_probe_2factor.txt`.

**Read it by CACHE, not by filler, and the run says so itself.** Its own guard
fired: `probes served a CACHED prefix: 4`. Repeated filler is the same text in
every probe, so after the first one the radix cache answers most of the prompt
(`cached=239616` at 280016, `cached=279552` at 300016, wall 50 s against 200 s).
I designed that arm as a CONTENT control and it is a CACHE control — the exact
confound R1 identified in the acceptance's bracket, one layer down. Said plainly
rather than presented as a 2-factor result.

So the COLD probes (`cached=0`) are the ones that speak, and there were 8:

| depth | cold probes | empty | rate |
|---|---|---|---|
| 240016 | 4 | 0 | 0% |
| **280016** | **2** | **1** | **50%** |
| 300016 | 2 | 0 | 0% |

**And the content hypothesis is dead.** `p02` is UNIQUE filler, `cached=0`, at
280016 tokens, and it returned `completion=1`. Every previous empty on record
used the repeated filler, which left "280026 is a bad depth" and "repetitive
content at depth is fragile" both alive. A unique-filler failure at the same
depth kills the second one.

**Pooled with every cold deep probe on record**, across four boots:

| depth | cold attempts | empty |
|---|---|---|
| 240016 | 4 | 0 |
| 250026 | 1 | 0 |
| **~280016** | **7** | **5** |
| 300016 / 300026 | 5 | 0 |
| 332532 | 1 | 0 |
| 338916 | 1 | 0 |

**~71% at ~280k, 0% at every other depth measured, including 58900 tokens
deeper.** That is a narrow failing BAND above `max_position_embeddings`
(262144), not a ceiling — a ceiling cannot be passed by going deeper, and 300k,
332k and 338k all pass. It is intermittent within the band (`p08` at the same
depth, same filler, same boot, returned the answer), so it is a RATE and not a
threshold.

**Cached prefills in the band are clean**: 4 of 4 exact at 280016/300016 when
the cache answered. The defect needs the cold prefill to actually run.

**Did the mamba-floor fix change it?** No. MERGE-R8's validator is on the line
in every run in both tables above, including all seven ~280k cold attempts, and
the rate is unchanged.

**Not explained.** Named, bounded, reproduced with a clean instrument, and
handed on: the next step is a bisect of the band (260k / 270k / 280k / 290k,
unique filler, cold) to find its edges, which is ~5 probes and 20 minutes.

---

## 5. THE SPEC NOTE FOR AXIS 4, CARRIED IN FULL

The spec requires that bs2-4 reserves — **including unused mamba states** — are
SPILLED during bs1 time. On a phase-flip boot that requirement is
**structurally unreachable**, and this run does not meet it:

`--enable-kv-session-offload` is refused outright under `pp_size > 1`
(`server_args.py:7405`, proved by executing against the real class), and the GDN
slot ladder's park population is exactly kv-session-offload's spilled set. So
there is no argv that produces vacate lines on a flip boot — adding the flag
makes the server refuse at parse time.

What DOES fire is the phase-flip pressure ladder (2181 rung lines this run:
cache, draft weights, weights-arena tail). That is real spilling and it is
heavily exercised, but it is **flip-seam spilling, not the idle-session mamba
vacate the spec asks for**, and reporting the rung count as if it satisfied the
spec item would be exactly the substitution the register exists to prevent.

The dependency is a SOURCING one, not a storage one (`GdnStateStore` is an
interface with three methods and no lifecycle), so the cheapest honest route is
a **PP-safe source of idle sessions** for the GDN ladder rather than lifting the
PP refusal, whose stated reason (host pool rows sized from the boot vector) is
real. Until that exists the spec item should be stated as a TP-phase-only
capability rather than carried as a failing axis.

---

## 6. OPEN

* **The seam cost MODEL is still ~3.8× low on the binding rank** (484 MiB
  modelled against ~1830 MiB drawn). Unchanged by this shift. What changed is
  that the gate can now act on a MEASURED draw instead of the model, so the
  model's error costs delays rather than breaches — when it fires, which it did
  not here.
* **The arena tail is priced with `max()` and the measurement says `+`.** The
  stage walk shows `weights_refill` committing with ~1214 MiB of seam state
  still outstanding, so on the `tp_to_pp` leg the arena tail and the wave peak
  DO coexist. Deliberately not changed: it widens the entry requirement on
  every seam, and the measured-draw withholding already protects the corridor
  from the real number rather than from a model.
* **The first cutover of a boot has no measured draw**, so the withholding
  cannot protect it. Early cutovers run at low occupancy, which is why this has
  never been the breaching regime, but it is a real gap.
* **A second frame-divergence source may still exist.** boot_v2 showed frames
  diverging with an identical pool census; that was traced to free-list ORDER
  and fixed, but the fix has not been exercised by a metal divergence since
  (boot_v3 had none to catch). The ballot now digests in three parts and names
  the diverging term, so the next occurrence will say which one it is.
* **Defect B is characterised but not explained.** §4.
