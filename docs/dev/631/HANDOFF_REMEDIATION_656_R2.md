# REMEDIATION 656 R2 — the 40404 rows have a cause, and the wedge has a valve

Shift `656-remediation-r2`, 2026-08-13. Branch `feat/acceptance-remediation-656`,
worktree `/spinning/wt-remediation-656`, continuing R1 @ `d76c68d529` (itself
rebased on `integration/r2` @ `6d169c04ab`, MERGE-R8). Evidence:
`/spinning/evidence-631/remediation-656-r2/`.

R1 left four things: a named divergence with no mechanism, a wedge, a 12 MiB
corridor residual, and an unrun probe. This shift closes the first two, corrects
the attribution of the third, and reframes the fourth.

---

## 1. C22 — the mechanism, and it was already in R1's log

**The shrink is collective. The recovery was not.**

R1's ballot named the SIZE (PP1 holds 40404 rows its peers do not enumerate)
and left the cause open. Parsing all 1980 POOL CENSUS lines of `boot_m1.log`
and comparing the three ranks field by field gives **7 divergent events in two
episodes**, and both open at a `tp_to_pp` **post-cutover** — the moment the KV
pool's physical backing is recovered:

| time | when | direction | unaccounted PP0/PP1/PP2 |
|---|---|---|---|
| 14:04:22 | post-cutover | tp_to_pp | 0 / **34313** / 0 |
| 14:42:25 | post-cutover | tp_to_pp | 0 / **40404** / 0 |

Two lines in the entire boot say what ran there, both on PP1, both at exactly
those instants:

```
14:04:23 PP1  KV-BACKING recovered to 544077 of 578390 rows (corridor-bounded)
14:42:25 PP1  KV-BACKING recovered to 537986 of 578390 rows (corridor-bounded)
```

`578390 − 544077 = 34313`. `578390 − 537986 = 40404`. **Both census deltas, to
the row.** No other rank logged a corridor-bounded recovery in 33 minutes.

`KvBackingRelief.shrink` is decided once for the group by a MIN reduction —
which is why every `KV-BACKING released` line shows an identical row count on
all three ranks. `recover()` is bounded by `self._free_bytes() − law`, **this
rank's own** distance from the corridor law. Rank 1 is the binding card, so
rank 1 is the rank that gets bounded, and it stays capped where its peers do
not. A capped rank enumerates a different id space; the seam's payload length
is derived from that enumeration; the frames diverge; the ballot abandons —
persistently, because a cap does not clear by itself.

Same law as the collective shrink, reached from the other side: **a refusal may
be decided locally, a CAPACITY may not.**

**Ruled out, with evidence:** #616B prefix-cache evolution and radix eviction
(all three ranks report `cached=268816`, identical, at every divergent census);
kvso spill sentinels (refused under `pp_size>1`, manager is `None`); the
pressure ladder (its rungs return *driver* pages and touch no allocator row).

### The fix — agreement, not detection

`cap_proposal` / `collective_cap_target` / `reconcile_to` level every rank's
exposed row count to what the poorest rank has BACKED. It rides the KV rung's
**existing** reduction, widened 4 → 8 fields — no second collective, because
the collective COUNT is itself a desync detector here and the gate's own tests
pin one reduction per gate call — and it lands strictly before `_frame_digest`
in the same round.

Levelling costs the healthy ranks nothing real: under pure PP every rank holds
the same token rows, so rows above the group minimum could never have been
admitted against. It is not a ratchet — the level rises again as soon as the
poorest rank recovers.

---

## 2. The wedge — a controller that cannot flip must still serve

R1 proved the ballot converts a crash into a **wedge**: `/health` 503, every
rank alive, KV intact, no tokens. The chain is short and entirely by design —
strict purity forbids decode in PP, decode waits for a `pp_to_tp` that is
refused every round, nothing reaches the detokenizer, the heartbeat expires.

**Zero tolerance forbids a wedge as much as a crash.** So when the layout a
work class needs is unreachable, the purity prohibition on *that class* is
lifted, loudly, until a flip commits.

Safe because purity is a THROUGHPUT rule, never a correctness one: its own
`threshold:<n>` and `off` modes run decode in the PP layout as supported
configurations, and the documented cost is latency and throughput. Group-uniform
because both inputs are already reduced quantities. Self-clearing because a
committed cutover resets the streaks.

This also unblocked axis 3 — see §3.

---

## 3. Axis 3 — R1's residual breach is a SEAM, not a prefill

R1 reported the remaining 12 MiB dip as a bs=1 deep-prefill transient ("138 MiB
at a seam became 12 MiB at a prefill"). **The 100 ms trace says otherwise.**
Both sub-law samples sit at `14:42:25.634` and `.734`, and the seam census for
that instant reads:

```
14:42:22 PP1  seam entry margin YIELDED (tp_to_pp) after 2 consecutive abandoned attempts
14:42:25 PP1  CORRIDOR LAW BROKEN during tp_to_pp rank 1 at stage 'weights_refill':
              free 1012 MiB is below the 1024 MiB floor
stage walk:   ... backing_restore free=1250 | kv_write 1250 | gdn_state 1250
              | weights_refill free=1012 step-238 | cutover free=1290
```

It is the acceptance's own mechanism — entry on the corridor law alone after a
C20 yield, then the in-cutover draw — 12 MiB deep instead of 138. The
deep-prefill phase is *when*, not *why*.

**The closer was already half-built.** The gate has long measured this rank's
worst in-cutover draw (`_seam_draw_max`) and predicted the trough from it,
while refusing to act, with its own reason stated: *"refusing pp->tp starves
decode outright [...] trading a 1.5 s corridor dip for a total outage is not a
fix."* With the valve in §2 that premise is false. So the YIELD — the one path
that deliberately enters at the law — is now **withheld** when the measured
draw predicts a sub-law trough. It objects with the margin-delay tag, which is
exempt from the seam-abandon cap, so a self-clearing condition cannot stand the
flip down for good.

---

## 4. TWO DEFECTS THIS SHIFT PUT IN AND METAL TOOK BACK OUT

Recorded because both were invisible from the desk and the reasons generalise.

**(a) The agreement tried to GROW.** `cap_proposal` first offered
`backed + (free − law)/bytes_per_row` — what `recover` would be *allowed* to
commit. On the `pp_to_tp` leg that hands back exactly the rows the collective
shrink had just taken to fund the seam. 2026-08-13 15:40:23Z, all three ranks:
`cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`, rank 0 driven to **3 MiB free**
(1021 below the law), the seam then unfundable, and three minutes into the boot
the instance sat in TP with a 9-token prefill it could not run.

The agreement is now **strictly non-allocating**: a rank proposes what it has
BACKED, and reconcile only engages or releases the cap. Growing keeps its one
owner. The level still rises without any commit, because a rank levelled down
kept its pages and follows a rising target by *releasing* a cap.

*Why the desk missed it:* a fake pool grows whenever the model says the bytes
are there, and the model was the thing that was wrong.

**(b) The valve was keyed on a counter the damping layer freezes.** Three group
abandons of `tp_to_pp` armed the seam BACKOFF, which then declined the next arm
requests **without entering the seam** — so `_seam_abandons_in_a_row` froze at 3
while the policy logged `tp_to_pp arm refused (7 in a row)` on all three ranks
and the prefill sat unrunnable for four minutes. The valve now also reads the
policy's own `arm_refusals` streak.

---

## 5. DEFECT B — the noise around it, removed

Counted across both previous boots, the load driver's own lines:

| prompt | R1 boot | acceptance boot | result |
|---|---|---|---|
| in=21 | 90x | 168x | out=48, finish=length |
| in=4001 | 55x | 107x | **out=1, finish=stop** |
| in=16001 | 14x | 27x | **out=1, finish=stop** |
| in=32001 | 15x | 28x | **out=1, finish=stop** |
| in=64001 | 15x | 28x | **out=1, finish=stop** |

Every prompt above 21 tokens returned a one-token completion, on both boots, at
every depth — 100%, not intermittent. Too uniform to be defect B, and the
difference is one line: the load rungs are pure filler and **ask nothing**,
while the yarn probes append a secret and a question.

**The control, boot_v2, 15:56Z** — same filler, same count, one question added:

```
WITHOUT a question:  in=4001  completion=1   finish=stop
WITH    a question:  in=4026  completion=12  finish=stop  cached=0
                     text='Answer:\n\n<think>\n\n</think>\n\nBANANA47'
```

Correct answer, needle 4000 tokens back, cold prefill. **The load driver's
one-token completions are the model emitting EOS on a prompt that asks
nothing.** They are not a serving defect. Both the acceptance and R1 read them
as content — right in outcome, wrong in reason — and it inflated the apparent
reach of defect B.

What remains: R1's yarn probes DO carry a question, and 280026 returned empty on
2 of 3 cold attempts. That is what the calibrated probe run measures. This
control removes the noise around it; it does not remove it.

---

## 6. METAL

Three boots this shift, and two of them are results.

**boot_v1 (15:37Z) — the agreement tried to grow.** Killed after 4 minutes.
Evidence `boot_v1_capfail.log`; diagnosis in §4(a).

**boot_v2 (15:49Z) — the two positive proofs.** Both mechanisms fired:

* 16:00:30Z, the levelling: three ranks at 579870 / 579722 / 578606 exposed
  rows converged to 578606 in ONE round, and every field of every POOL CENSUS
  line was identical from the next round on;
* 16:01:24Z, the valve: a persistent divergence blocked `tp_to_pp`, all three
  ranks relaxed purity for PREFILL simultaneously and identically, and the
  instance kept serving — `/health` 200, tokens flowing, the load driver
  advancing — where R1's boot answered 503 with none.

It also produced this shift's second defect (the free-list order, §4(b)).

**boot_v3 (16:12Z) — the acceptance re-run.** One boot, one continuous log,
argv identical to the acceptance and to R1's boot, 55.4 minutes of load plus
the deep-probe run in the same log, corridor sampled at 100 ms throughout.

| | acceptance | R1 | **R2** |
|---|---|---|---|
| cutovers | 320, then died | 218 + 12 frame abandons | **364, 0 abandons** |
| terminal state | SIGQUIT | wedge, 503, no tokens | **serving** |
| corridor minima | 886 / 1941 / 1304 | 1012 / 2057 / 1370 | **1070 / 2139 / 1466** |
| samples below the law | 25 (5 episodes) | 2 (1 episode) | **0 of 33648** |
| tracebacks / SIGQUIT | 2 / 1 | 0 / 0 | **0 / 0** |

**7 of 7 axes.** Full table and caveats in
`evidence-631/remediation-656-r2/ACCEPTANCE_656_R2.md`.

### WHAT boot_v3 DOES NOT PROVE, and it matters

Neither mechanism this shift built fired on it: 0 levelling events, 0 purity
stand-downs, 0 yields withheld, 0 yields taken. One corridor-bounded recovery
DID occur (PP1, 16:25:13Z, 558489 of 585730 rows) and opened no divergence,
because the next `pp_to_tp` shrink caps every rank to one agreed absolute row
target and therefore **levels the group by construction**. That is why the
divergence window is narrow, why R1 saw only two episodes in 33 minutes, and
why a clean boot is cheap to get.

So boot_v3 is a NO-REGRESSION result on C22, not a positive proof; the positive
proofs are boot_v2's two events and the desk red arms. The corridor result is
likewise attributable to the pool sizing and to the gate never having to yield
— not to the yield-withholding, which has still never fired on metal.

At the acceptance's 1-in-320 rate, 364 clean cutovers happen about 32% of the
time on an instance where nothing was fixed. ~957 would be a 95% claim.

---

## 7. DEFECT B — the rate, and the confound resolved

12 probes on boot_v3, FILLER x DEPTH, 2 each, every one planting a secret and
asking for it. **Read it by CACHE, not by filler**: the run's own guard fired
(`probes served a CACHED prefix: 4`) because repeated filler is the same text
in every probe, so the radix cache answers most of it after the first. That arm
was designed as a CONTENT control and is a CACHE control — R1's confound, one
layer down.

The 8 COLD probes (`cached=0`) are the ones that speak:

| depth | cold probes | empty |
|---|---|---|
| 240016 | 4 | 0 |
| **280016** | **2** | **1** |
| 300016 | 2 | 0 |

**`p02` is UNIQUE filler, `cached=0`, 280016 tokens, `completion=1`.** Every
previous empty on record used the repeated filler, which left "bad depth" and
"repetitive content is fragile" both alive. A unique-filler failure at the same
depth kills the second.

Pooled with every cold deep probe on record across four boots: **~5 of 7 empty
at ~280k, 0 of 12 at 240016 / 250026 / 300016 / 300026 / 332532 / 338916.** A
narrow failing BAND above `max_position_embeddings` (262144) — not a ceiling,
since 300k, 332k and 338k all pass — and intermittent within it, so a RATE and
not a threshold. Cached prefills in the band are clean (4 of 4). The
mamba-floor fix is on the line in all of it and changed nothing.

Next step, cheap and specified: bisect the band (260k / 270k / 280k / 290k,
unique filler, cold) — about 5 probes and 20 minutes.

---

## 8. OPEN, AND HONEST

* **The arena tail is priced with `max()`, and the measurement says `+`.**
  `_staging_bytes` returns `max(wave_peak, draft_restore, arena_tail)` on the
  reasoning that the peaks belong to different legs. The stage walk shows the
  `weights_refill` commit happening with ~1214 MiB of seam state still
  outstanding (entry 2464, refill at 1250), so on the `tp_to_pp` leg the arena
  tail and the wave peak DO coexist. Deliberately not changed here: it widens
  the entry requirement on every seam, and the measured-draw withholding in §3
  already protects the corridor from the real number rather than from a model.
  It is the next model correction to buy, and it is measured, not guessed.
* **The first cutover of a boot has no measured draw**, so the yield-withholding
  cannot protect it — `_seam_draw_max` is 0 until this rank has seen a cutover,
  and an unmeasured bucket is never a licence to invent a number. Early cutovers
  run at low occupancy, which is why this has never been the breaching regime,
  but it is a real gap and the seam record's persisted shortfall is the obvious
  seed for it (different quantity from the draw, so not a one-liner).
* **The seam cost MODEL is still ~3.8x low on the binding rank** (R1 §7,
  unchanged by this shift). What changed is that the gate now acts on a MEASURED
  draw instead of the model, so the model's error costs delays rather than
  breaches.
* **Idle-vacate** is unchanged from R1 §3: structurally unreachable under
  `pp_size>1`, booked with its spec-compliance note, needs a PP-safe
  idle-session source.
