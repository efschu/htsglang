# REMEDIATION 656 — the two instance-level failures, and two diagnoses

Shift `656-remediation`, 2026-08-13. Branch `feat/acceptance-remediation-656`,
worktree `/spinning/wt-remediation-656`, rebased onto `origin/integration/r2`
@ `6d169c04ab` (MERGE-R8) before the metal leg. Evidence:
`/spinning/evidence-631/remediation-656/`.

The #656 acceptance failed 3 of 7 axes. This shift owns the two
instance-level failures (C22, the corridor) and two diagnoses (idle-vacate,
the empty completion).

---

## 1. C22 — ROOT CAUSE: the flip's wire frame is never agreed

**The message was a misreport, and the log carried its own falsifier.**

The acceptance died at 13:03:16Z after 320 clean cutovers:

```
KvReshardError: PHASE-FLIP payload checksum mismatch from peer 1:
  sender 4626949667419791296, receiver 30942312421   (PP0)
  sender -4450328002521349435, receiver 17682061978  (PP2)
```

A `uint8` sum over N bytes lies in `[0, 255N]`. The second sender field is
**negative**; the first would need an 18-petabyte payload. **Neither was ever
a checksum**, so the data was never the thing that differed. Both receiver
values are ordinary sums. This is arithmetic, not inference, and it is pinned
in `test_the_acceptance_sender_fields_were_never_checksums`.

### What actually breaks

The per-peer payload length is a product of **rank-local** terms:

* the live slot set — `build_flip_live_slots_fn` documents it as
  "Replicated: the tree and the batch state are rank-replicated between
  rounds" and **nothing verifies it**;
* the wave partition — `_flip_waves`' own docstring names two rank-local
  inputs (`_pools_alias`, `SGLANG_FLIP_SEAM_WAVES`) and says ranks would then
  "call `_exchange` a different number of times… it is not checked".

Nothing on the wire carries a length. And the receiver's size check is
**vacuous by construction**:

```python
if payload is None or payload.numel() != incoming_nbytes[peer]:
```

`payload` is the buffer this rank allocated, at exactly `incoming_nbytes[peer]`
bytes. The check compares the receiver's own allocation against the receiver's
own expectation and can never fire on a sender-side divergence.

So when the frames diverge, NCCL matches the send against the recv, delivers
the shorter count, **completes**, and the trailer is read out of the unwritten
tail of a `torch.empty` buffer. The guard then reports a data corruption that
did not happen — and raising at the seam takes the instance down. One cutover
in 320: an unattended auto-flip MTTF of about an hour.

### Why the hermetic suite ran straight through it

`_MailboxExchange` hands the receiver the **sender's** tensor, so a frame
divergence surfaces there as a clean size error. NCCL hands the receiver the
buffer the receiver allocated. `_NcclLikeExchange` models the real delivery
semantics (receiver-allocated, poison-filled, `min(sent, expected)` bytes),
and that difference is the whole distance between a caught bug and an
hour-long MTTF.

### This corrects the register, and it UNIFIES the two incidents

MERGE-R7 attributes C22 to #657's corridor steering re-sorting on a rank-local
1 s clock. **That path was not enabled on the acceptance boot**
(`corridor_post_sizing_mib=None`, `SGLANG_CORRIDOR_REBALANCE=0`) and it failed
anyway. But the attribution is not merely wrong — it is a *trigger* mistaken
for a *cause*: the steering re-sorts the free list on a private clock, which
changes which rows a rank holds, which changes its frame. Disabling the
steering removed that trigger and left the defect. Register row `C22-b`.

### The fix

* **A pre-move frame ballot.** `_frame_digest` fingerprints the live slot set,
  the wave partition, the vector and the direction, and rides the `[x, -x]`
  MIN pair on the collective the round **already runs** — no extra round trip.
  Ranks that disagree **abandon unanimously before a byte moves** instead of
  raising at the seam. This is #639's remedy for the prefix-length vector,
  applied to the premise the seam actually depends on.
* **`checksum_is_representable()`** splits the two diagnoses the guard
  conflated. Out of range → a FRAMING failure, named as such. In range → the
  data differs, which is what the guard is for.
* **`frame_aborts`** is counted apart from fit/staging/corridor aborts: a
  broken replication premise and a pool that is too small want opposite
  responses from an operator.

No new communicator is introduced — relevant to contract-612, since the ballot
widens an existing `_collective_min` payload from 3 to 5 elements on a group
the round already reduces on, so nothing new lands inside the `nccl_init`
boundary.

### THE BALLOT FIRED ON METAL, AND IT NAMED THE DIVERGENCE

**2026-08-13 14:46:39Z**, during a cold 280012-token deep prefill on the
remediation boot. Evidence: `BALLOT_FIRED_ON_METAL.txt`.

```
PP0  framed digest 1545804850
PP2  framed digest 1545804850
PP1  framed digest 1237458399     <- the group spans [1545804850, 1237458399]
```

All three ranks abandoned the flip unanimously. **No KvReshardError. No
SIGQUIT. The process lived.** Under the code this replaces, that same round
would have been the acceptance's crash.

**PP1 is the same rank both peers blamed in the acceptance** ("from peer 1").

And the pool census at that instant says exactly what diverged:

| rank | free rows | unaccounted |
|---|---|---|
| PP0 | 309574 | 0 |
| PP2 | 309574 | 0 |
| **PP1** | **269170** | **40404** |

`309574 − 269170 = 40404`, exactly. All three agree on the resident request
(`rid fa034de4…`, seqlen 268817, `kv_allocated_len` 268816) and on
`cached=268816`, so **the difference is not the request**: rank 1's allocator
holds 40404 rows that neither the radix tree nor any resident request
accounts for, and its live-slot enumeration therefore differs from its peers'
by that many rows. **40404 rows × row_bytes IS the payload-length mismatch**
that, before the ballot, produced the garbage trailer.

So the open question §7 of the first draft listed — *which* rank-local term
diverged — is closed within the same window the ballot shipped: it is the
live slot set, the divergence is an allocator-owned residue on one rank, it
is persistent rather than a race, and it appears under a deep resident
prefill. The acceptance's fatal cutover had a 312221-token resident request;
this one had 268816. Same regime.

### AND IT IS NOT BENIGN: a PERSISTENT divergence is a WEDGE

The honest other half, measured in the same window. The divergence did not
clear. The ballot refused the flip eight times, the seam cap guard declared
`pp_to_tp` unfundable, and the policy held the instance in PP — which is the
leg **decode** needs. The result:

* process alive on all three ranks, still logging;
* every request's KV intact, nothing scattered;
* `/health` **503**, "couldn't get a response from detokenizer";
* **no tokens**.

That is the 411-abandon deadlock class the corridor guard's arming/law
separation was introduced to prevent, reached by a different route. **The
ballot converts an instance-fatal crash into an instance wedge.** That is
strictly better — recoverable instead of dead, diagnosed instead of
mysterious, with the two digests and the 40404-row delta printed — but the
first draft of the log line said "serving continues", and on the pp→tp leg
that is false. The message now says so, and names the census comparison that
identifies the odd rank.

**The real fix is AGREEMENT, not detection.** The ballot proves the
replicated-live-set premise is false on this rig; a successor should replace
the premise rather than keep testing it — broadcast the live slot set from a
designated rank, or reduce it to the intersection, so the ranks cannot frame
different payloads in the first place. Detection was the right first move
because it is cheap, unanimous and non-fatal, and because it is what produced
the 40404 number that a fix can be built against.

### Two pre-existing tests changed, and they got STRONGER

`test_can_fail_abort_applied_on_one_rank_mid_flip` and
`test_can_fail_batch_membership_disagreement` pinned the old mechanism (a loud
raise). Their binding property is "detected, refused, never silently commits".
They now assert the ballot's outcome: **no rank raises, every rank books
`frame_aborts`, and not one byte moved on any rank** — which the old version
did not require, since it tolerated ranks whose payloads happened to be
consistent scattering theirs.

---

## 2. CORRIDOR — the breach is a YIELD, not a sizing shortfall

**Read the card attribution first (register C21).** `--rank-gpu-id` is in CUDA
order and this rig is FASTEST_FIRST, so **rank0 is the 5090** (nvidia-smi index
1) and ranks 1/2 are the 3080s (smi 0 and 2). Confirmed three ways on the
acceptance boot: the budget vector `31583,15750,18205` gives rank0 the 32 GB
card; the seam records' at-rest headroom (3444 / 848 / 1164 MiB above the law)
matches the smi columns 1/0/2; and PP1 raises 21 of the 24 guard refusals.
**The log's PP1 is the smi `gpu0` column.**

That single fact reverses the acceptance's own remedy.

### The mechanism, measured

All five GPU0 breaches begin **2-4 s after a C20 seam-entry-margin YIELD, on
that same rank**. 5 of 5.

| episode | entry level | trough | drawdown | law+drawdown | shortfall |
|---|---|---|---|---|---|
| 12:10:31Z | 2738 | **886** | 1852 | 2876 | **138** |
| 12:57:51Z | 2792 | 978 | 1814 | 2838 | 46 |
| 12:58:09Z | 2804 | 990 | 1814 | 2838 | 34 |
| 12:58:27Z | 2800 | 970 | 1830 | 2854 | 54 |
| 12:59:06Z | 2774 | 948 | 1826 | 2850 | 76 |

This is not the gate being bypassed — it is the gate's own escape hatch. PP1
refused the entry margin twice, and the yield policy entered on the corridor
LAW alone. The yield's own log line says what that costs: *"it enters at the
law and is therefore exposed to the in-cutover draw register C20 measures."*
It was.

**Honest scope on the drawdown**: 1814-1852 MiB is measured over a 9 s window
around each episode, so it is an **upper bound** on the seam's own price —
concurrent prefill is inside it. Register C21 measured seam-*scoped* drawdowns
of 504 MiB by a purpose-built joiner on a different config; the two are not
the same quantity and this one is not offered as a correction of it. The
load-bearing numbers — entry level, trough, shortfall — are read straight off
the column and do not depend on the split.

### What the acceptance got wrong

> "Give rank0 roughly 200-300 MiB back in the budget vector."

Wrong rank (rank0 is the 5090; minimum free 1941 MiB, it was never near the
law) and wrong quantity (gpu0's median free is 1788 MiB and its p1 is 1220 —
there is no steady-state shortfall to refund). The worst shortfall is **138
MiB**, at a transient, and it equals the breach depth exactly. Register row
`C21-b`.

### The fix

* **The threshold pair is declared together.** `CORRIDOR_LAW_MIB` is the one
  declaration; `arming_floor_mib(seam_entry_reserve_mib)` derives the gate's
  watermark from it; `check_threshold_pair()` refuses an arming floor below
  the law (such a gate would return "no reclaim needed" for an allocation that
  ends under the corridor — laundering a breach as a passed check, which the
  guard's own refusal message forbids). `corridor_trace.summary()` defaulted to
  a private literal `1024` and now reads the guard's constant, so an instrument
  cannot report a different verdict from the gate it audits. The default pair
  is unchanged: 1024 + 512 = 1536, exactly what the acceptance ran.
* **The runtime now AUDITS the trace it arms.** R8 gave `corridor_trace` its
  production call site; arming alone left the process able to measure its own
  corridor and unable to say anything about it. `_corridor_trace_tick` reads
  the verdict every 10 s (sampling stays at 100 ms) and logs loudly **only
  when the floor gets worse**, so each line is a new deepest instant rather
  than a repeat.
* **The sizing term, derived and never a constant.** Seam solver margin
  192 → 384 MiB: 192 (the existing measurement error bar) + 138 (measured
  shortfall) + 38 (measured spread of the drawdown) = 368, rounded up for
  allocator granularity. **Pool re-derived: 597106 → 578390 tokens.** 18716
  tokens is the price of the law on this configuration, and it is a real
  price, reported as one.

---

## 3. IDLE-VACATE — diagnosis: BOTH (a) and (b), and (b) is decisive

The acceptance logged **0 vacate lines**. Three candidate explanations were on
the table: the flag was missing from the recipe, the mechanism is unreachable
under the flip policy, or a merge regressed it. **It is not a regression.**
It is a recipe fact standing in front of a structural wall, and the wall is
what matters.

### The gate chain, walked outward

1. **`scheduler.py:4842`** —
   `if self.server_args.gdn_resident_state_slots is not None:` is the only
   thing that builds `gdn_slot_executor`. The acceptance boot did not pass the
   flag (`extract.sh`: *"gdn resident cap: 0 (0 = flag not set this boot)"*),
   so the executor was never constructed and no vacate line could exist. That
   is explanation (a) — and it was a deliberate choice, not an oversight: the
   flag had just been shown to be **instance-fatal** at `4` (defect A, the
   mamba floor), so the acceptance boot dropped it.

2. **`gdn_slot_runtime.py`, the idle-session source** — the park population is
   `scheduler.kv_session_offload.live_offload_reqs()`, and the getter returns
   `[]` when the manager is `None`. The executor then builds, arms, and its
   plan has nothing to vacate.

3. **`server_args.py:7405-7409`** — the wall:

   ```
   if self.pp_size > 1 or self.dp_size > 1:
       raise ValueError(
           "--enable-kv-session-offload (S1) supports single-node pure "
           f"TP/DCP only (pp_size={self.pp_size}, dp_size={self.dp_size})."
       )
   ```

   A phase-flip boot **is** `pp_size=3`. So `--enable-kv-session-offload` is
   not merely inert on a flip boot — **the boot refuses to start with both
   flags**. `scheduler.kv_session_offload` is `None` on every phase-flip
   instance, by construction.

   **Proved by execution, not by reading**, because a lane sent to check this
   reported the opposite — it searched `kv_session_offload.py` (which indeed
   contains no `pp_size`) and concluded no refusal existed. The refusal is in
   `server_args.py`. Against the real class:

   ```
   ServerArgs._handle_kv_session_offload   called from __post_init__ (:6255)
     early return "if not self.enable_kv_session_offload"   at char  9326
     the pp_size > 1 raise                                  at char 25487
   ```

   The raise sits AFTER the early return, so it is reachable exactly when the
   flag is on. An agent report is not evidence; the source of the real class,
   read at runtime, is.

### Therefore the recipe cannot be fixed

There is no argv that produces vacate lines on a phase-flip boot. Adding
`--enable-kv-session-offload` to the flip recipe makes the server refuse at
parse time. The flag's own help text has said so all along: *"its standing
population is kv-session-offload's spilled set -- which is refused under
pp_size>1, so on a PHASE-FLIP (PP) boot the ladder arms and never fires."*

### Honest spec-compliance note

The spec requires that bs2-4 reserves — **including unused mamba states** —
are SPILLED during bs1 time. On a phase-flip boot with the merged mechanisms
that requirement is **structurally unreachable**, and axis 4 cannot pass. What
does fire on this path is the phase-flip pressure ladder (rungs 1-3: cache,
draft weights, weights-arena tail), which is live and load-bearing and heavily
exercised. That is real spilling, but it is **flip-seam spilling, not the
idle-session mamba vacate the spec asks for**, and reporting the rung count as
if it satisfied axis 4 would be exactly the substitution this register exists
to prevent.

### Booked as its own ticket

**The dependency is a SOURCING dependency, not a storage one.** `GdnStateStore`
is an interface — *"Where a vacated GDN state waits. Three methods, no
lifecycle."* — so the vacate does not need kv-session-offload's host pool to
put a blob somewhere. It needs kvso only to enumerate which sessions are idle.
So the cheapest honest route is **not** to lift the PP refusal (whose stated
reason, host pool rows sized from the boot vector, is real) but to give the
GDN slot ladder a **PP-safe source of idle sessions** of its own. Until that
exists, the spec item should be stated as a TP-phase-only capability rather
than carried as a failing axis.

---

## 4. THE EMPTY COMPLETION — the confound, and what it hides

**The acceptance's own bracket cannot speak to this defect, and its author
said so.** `yarn_bracket.txt` shows 280026 EXACT three times running, at
127.2 s → 46.8 s → 17.9 s. That collapse is the radix cache filling in: the
probes shared a filler prefix, so repeats 2 and 3 **never re-ran the deep
prefill**. Only the first probe was a cold one.

Re-read with that in mind, the acceptance's evidence points the other way
from its own summary:

| probe | prefill | result |
|---|---|---|
| 280026, first attempt (acceptance run 1) | **cold** | **EMPTY** |
| 280026 x3 (bracket) | 1 cold, 2 cached | EXACT (the cold one at 127.2 s) |
| 300026 (acceptance) | cold | EXACT |

and the remediation boot reproduced it on its **first cold deep probe**:
`prompt_tokens=280026 completion=1 finish=stop wall=193.1s` — an empty
completion on a cold 280k prefill, on the fixed tree, with the mamba-floor
validator from R8 on the line.

So the shape to test is **cold prefill**, not depth: `yarn_cold20.sh` gives
every probe a UNIQUE filler (the tag is woven into every sentence), so no
prefix has ever been seen and the cache cannot answer any part of it, and it
alternates a depth above `max_position_embeddings` with a control below it.
Numbers in §6.

What is already settled and does not need re-litigating: **there is no
ceiling** (300026 decodes exactly, 37882 positions past
`max_position_embeddings`), and **the lazy RoPE cache is exonerated** — the
acceptance ran its probes at the shipped eager default and saw the same
signature, so MERGE-R7 follow-up 1's attribution came from an uncontrolled
lazy-only pair.

### Every COLD deep probe on record, across both boots

| tokens | boot | filler | result |
|---|---|---|---|
| 250026 | acceptance | repeated | EXACT |
| **280026** | acceptance run 1 | repeated | **EMPTY** |
| **280026** | acceptance bracket (127.2 s) | repeated | EXACT |
| 300026 | acceptance | repeated | EXACT |
| **280026** | remediation, load phase 2 (193.1 s) | repeated | **EMPTY** |
| 300026 | remediation, load phase 2 (226.9 s) | repeated | EXACT |
| 338916 | remediation, probe attempt 1 (244.0 s) | **unique** | EXACT |

**280026 has failed 2 of 3 cold attempts. Every other depth is 4 of 4
exact, including one 58890 tokens deeper.** That is a sharper shape than the
acceptance's "intermittent, not depth-determined", and it is not a ceiling
either — a ceiling cannot be passed by going deeper.

**Two confounds are live and the probe run separates only one of them.** All
the failures used the REPEATED filler (`"The quick brown fox jumps over the
lazy dog. " * n`); the one unique-filler probe passed. So "280026 is a bad
depth" and "highly repetitive content at depth is fragile" both fit the
table. The run below uses unique filler at **280012** — 14 tokens off the
failing depth and different content — so a failure there implicates depth
and a clean sweep implicates content, but neither is isolated by this run
alone. Said plainly rather than presented as a rate for a single cause.

### THE RATE IS STILL NOT CHARACTERISED, and here is exactly why

**The probe run did not finish.** It was aborted by the instance wedging
(§1) during its first recalibrated probe. What exists is:

| probe | tokens | filler | result |
|---|---|---|---|
| attempt 1, probe 2 | 338916 | unique | **EXACT**, 244.0 s, `cached=0` |
| attempt 1, probes 1 and 3 | 414320 | unique | refused, clean 400 naming the 393216 limit |
| attempt 2 | 280012 | unique | **never returned** — the instance wedged mid-prefill |

So the deliverable this shift owed on defect B is **not delivered**: there is
no rate. Saying so plainly rather than presenting `1 of 1 exact` as a
characterisation.

**But the abort is itself a result.** The wedge fired *during a cold
280012-token prefill*, and the divergence it caught (§1) had a 268816-token
request resident. The acceptance's fatal cutover had 312221. **Deep resident
prefill is the regime in which the live-set divergence appears**, which
makes the deep-prefill probe and the C22 hunt the same experiment rather
than two. A successor running the rate probes should expect to trip the
ballot, and should run them with the flip policy pinned (no auto-flip) if
the intent is to measure the completion and not the seam.

**What to run, unchanged:** `yarn_cold20.sh` with `N=6`, depths 11196 →
280012 and 9644 → 240016, both unique-filler, `cached` asserted 0 in the
summary. It is written, calibrated against the checkpoint's own tokenizer,
and unrun.

### The mis-calibration, recorded because it cost a probe

The first attempt reused the acceptance's sentence COUNTS. The unique tag
makes each sentence longer, so 17016 sentences came to 414320 tokens and the
server refused with a clean 400 naming the 393216 limit — the same incidental
confirmation the acceptance got. The counts are now computed exactly with the
checkpoint's own tokenizer (11196 → 280012, 9644 → 240016) instead of scaled
from someone else's probe.

---

## 5. METAL — one boot, and what it is honestly worth

Boot `boot_m1`, branch `feat/acceptance-remediation-656`, argv **identical to
the acceptance** (same TP vector `30,16,18`, same budget vector
`31583,15750,18205`) so the seam records stayed WARM and the mandatory
two-boot protocol was satisfied. One deliberate difference:
`SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB=384`, derived in §2.

* **Pool: 578390 tokens**, derived and uncapped, against the acceptance's
  597106. **18716 tokens is what the corridor law costs on this
  configuration**, and it is a real price, not a rounding.
* `corridor trace armed at 100 ms` on all three ranks — R8's call site, now
  proven ON on a flip boot.
* cgroup `/system.slice/htsglang-serving-20260813T134830Z-2227367.scope` —
  not `claude.service`.

### EXACTLY WHAT CODE THIS BOOT RAN, because it is not the whole branch

The boot started 13:48:30Z, when HEAD was `522eae8c9d` (R8 + the C22 frame
ballot). The corridor commits landed afterwards. So:

| change | in the booted tree? |
|---|---|
| C22 frame ballot + representability check | **YES** — running, 152+ cutovers |
| `corridor_trace` armed at 100 ms (R8's call site) | **YES** — armed on all 3 ranks |
| the seam margin at 384 MiB | **YES**, via the env, which is version-independent |
| the declared THRESHOLD PAIR + `check_threshold_pair` | **NO** — desk only |
| the corridor AUDIT (report + record on worsening) | **NO** — desk only |
| the per-rank self-calibrating margin | **NO** — desk only |

The corridor RESULT is therefore attributable to the pool re-derivation, which
is the mechanism §2 claims and the one the env exercises. The declaration and
audit work is desk-proven and **has not run on metal**; the acceptance re-run
will be its first. Stating this rather than letting one green boot be read as
covering the whole branch.

### The corridor result

Continuous 100 ms sampling across the whole load window:

**Segment it by regime, because the two halves give different verdicts.**
63.5 minutes, 38020 samples per card:

| regime | window | gpu0 min | gpu1 min | gpu2 min | below the law |
|---|---|---|---|---|---|
| **mixed flip soak** | 46.1 min | **1050** | 2097 | 1370 | **0 / 0 / 0** |
| **deep-prefill probes (bs=1)** | 17.3 min | **1012** | 2057 | 1376 | **2 / 0 / 0** |
| full window | 63.5 min | 1012 | 2057 | 1370 | 2 / 0 / 0 |

vs the acceptance's 886 / 1941 / 1304 with 25 samples in 5 episodes.

**So the corridor holds under the flip soak and BREACHES — 12 MiB, 2 samples,
one episode at 14:42:25Z — under a maximal-depth single-session prefill**
(the 338916-token probe). This is a different regime from the acceptance's
five breaches, which were all at seam entries after a yield; there is no
yield here and no flip involved.

It has to be reported as a breach, not rounded away: the law is a continuous
minimum and the spec puts bs=1 YaRN past `max_position_embeddings` squarely
in scope, so a deep-prefill trough is inside the acceptance's own axis-3
window. What changed is the magnitude and the cause: **138 MiB at a flip seam
became 12 MiB at a deep prefill.** The seam mechanism is closed; a prefill-
side trough is a different, much smaller, and so far uncharacterised item.

And the causal chain for the seam half is visible in
the log rather than inferred:

| | acceptance | remediation |
|---|---|---|
| `CORRIDOR-GUARD REFUSED` | 24 (21 of them on rank 1) | **0** |
| `FLIP DELAYED` (entry margin short) | many | **0** |
| `seam entry margin YIELDED` | 8 | **0** |
| corridor breach episodes | 5 | **0** |

The gate never fired at all. A pool 18716 tokens smaller leaves the C20 entry
margin affordable on every attempt, so the rank never has to enter on the law
alone — and every breach in the acceptance followed an entry that did.

**One caveat stated rather than buried: the flip rate is 18% lower.** 4.25
cutovers/min here against the acceptance's 5.20, on the same load script.
Some of the improvement could therefore be a slightly gentler flip cadence
rather than the pool change alone. What the rate cannot explain is the gate
counters going from 24 refusals and 8 yields to exactly zero — a lower
cadence thins the sample, it does not make a margin affordable — but the
corridor minimum should be re-read on the acceptance re-run at a matched
cadence before the +26 MiB figure is treated as the settled headroom.

**Fill quality, honestly.** gpu0 at +26 MiB is exactly the user's *frei nahe
1024, nicht mehr*. gpu1 is **under-filled by about a gibibyte**. That is the
cost of raising the margin with the GLOBAL env knob, which charges every rank
for a shortfall only rank 1 measured. The mechanism merged on this branch is
**per rank** (the shortfall lives in that rank's own seam record), so a boot
on the merged mechanism should leave gpu1 better filled. That is the next
capacity win and it is already paid for.

**The seam cost MODEL is still ~3.8x low** and this shift did not fix it: the
sizer prices rank 1's seam at 484 MiB while the column draws ~1830. What
changed is the resting level, not the model. See
`seam_model_vs_measured.txt`.

### The C22 result, and its power

Across the 218-cutover soak: **0 KvReshardError, 0 "NOT A CHECKSUM", 0
tracebacks, 0 scheduler exceptions, 0 SIGQUIT, 0 CANNOT FUND, 0 abandons.**

**And then, in the deep-prefill phase, the ballot FIRED** — 12 frame-
divergence abandons, still 0 KvReshardError and 0 SIGQUIT (§1). That is the
result this shift actually bought: not an absence, an **occurrence** that was
caught. The defect reproduced on metal, on the same rank the acceptance
blamed, and the instance did not die.

It also means the power argument below applies only to the soak half. **The
ballot's positive metal proof does not depend on it at all**: a divergence
happened, the digests differed by a measurable amount, and the guard did what
the desk arm said it would.

**What the soak DOES prove, and it is not nothing: the ballot does not
false-positive on metal.** Every completed cutover is a round in which the
frame digest was computed independently on three ranks and reduced to the
same value. A digest built from an accidentally rank-local term — free
memory, a clock, a pointer, anything the file's own docstrings warn about —
would have abandoned *every* flip, loudly and immediately. Instead the group
agreed on every one, so the inputs the digest is built from really are
replicated on real hardware, which is the half of the ballot's correctness a
desk fixture cannot establish.

**On the other half it is a no-regression result, not a proof.** At the
acceptance's
observed 1-in-320 rate, a clean run of 180-220 cutovers happens **50-57% of
the time on an instance where nothing was fixed**. A 95%-level metal claim
needs **~957 cutovers** (~3 h at this rate); 99% needs ~1471. Full table in
`c22_soak_power.txt`.

So the proof of the C22 fix is the **desk red arm**, which can fail and does:
stub `_frame_digest` to a constant and a live-set divergence reaches the wire
and raises with a trailer field of `-6510615555426900571` — the acceptance's
own signature, reproduced on a CPU desk; restore the ballot and the same
divergence abandons unanimously with not one byte moved.

This also puts the acceptance in perspective: **at 320 cutovers it had a 37%
chance of seeing nothing**. C22 was found as much by luck as by design, which
is the argument for folding a ~1000-cutover soak into the acceptance re-run
rather than treating 320 as a standard.

---

## 6. IS THE ACCEPTANCE RE-RUN UNBLOCKED?

**No — and the reason is a better one than the shift started with.** Every
failure is smaller, named and mechanised, but a clean pass is not yet
available on any of the three failing axes.

| axis | acceptance | after this shift |
|---|---|---|
| 3 MAX-KV + corridor >= 1024 continuous | FAIL — 886 MiB, 25 samples, 5 episodes, at flip seams | **improved, still FAIL** — the seam mechanism is closed (0 breaches in 46 min of mixed flip soak), but a bs=1 maximal-depth prefill still dips to **1012 MiB, 2 samples, one episode**. 138 MiB at a seam became 12 MiB at a prefill |
| 7 zero tolerance | FAIL — 2 tracebacks, 1 SIGQUIT, 1 breach | **improved, still FAIL** — no traceback, no SIGQUIT, and C22 no longer kills the instance; but a PERSISTENT frame divergence holds the instance in PP and it **wedges** (`/health` 503, no tokens). Zero tolerance forbids a wedge as much as a crash |
| 4 idle-vacate | FAIL — 0 vacate lines | **still fails, and cannot be fixed by a recipe** — `--enable-kv-session-offload` is refused outright under `pp_size>1` (`server_args.py:7405`), so no argv produces vacate lines on a flip boot. Needs a PP-safe idle-session source or a spec amendment |

**What the shift did buy**, stated without inflation:

* C22 went from **kills the instance** to **wedges it, loudly, with the two
  digests and a 40404-row delta printed**. Recoverable instead of dead,
  diagnosed instead of mysterious.
* The root cause moved from a wrong attribution (#657 steering) to a measured
  one (rank 1's allocator holds 40404 rows its peers do not enumerate).
* The corridor's seam breach is closed by a derived, measured term rather
  than a guessed one, and the residual is an order of magnitude smaller and
  in a different mechanism.
* Idle-vacate stopped being an open question at all.

**The one change that would unblock axis 7 is not detection, it is
agreement** — broadcast or intersect the live slot set so the ranks cannot
frame different payloads. That is a bounded piece of work and it is now
specified by a number rather than a hypothesis.

**Two things the re-run must carry, or it will draw a wrong conclusion:**

1. **Run the soak long.** ~957 cutovers for a 95% claim on C22. The
   acceptance's 320 had 37% power. The counter is already in the log
   (`PHASE-FLIP cutover complete`, once per rank — divide by 3).
2. **Re-derive the pool on the merged mechanism, not the env.** The metal
   boot pinned the margin globally at 384 MiB; the merged mechanism applies
   the measured shortfall **per rank**, which should recover most of the
   gibibyte gpu1 is currently leaving idle.

---

## 7. OPEN, AND HONEST

* **The seam cost model is ~3.8x low on the binding rank** (484 MiB modelled,
  ~1830 MiB drawn) and this shift compensated for it rather than fixing it.
  The next measurement anyone should buy is
  `scripts/s38_seam_price_vs_draw.py` against a boot of THIS configuration,
  which gives the seam-SCOPED price instead of the 9 s window upper bound
  used here. Register C21 holds the hermetic half.
* **The corridor audit's can-fail is desk-only.** It logs and records only
  when the floor gets worse; on this boot the floor never went below the law,
  so the reporting path did not execute on metal. Its unit-level pieces are
  tested (`test_seam_margin_selfcal_656.py`), the report-only-on-worsening
  rule is not.
* **The self-calibrating margin has never completed a full cycle on metal.**
  Boot A must breach for boot B to consume the record, and boot A did not
  breach — which is the desired outcome and also why the loop is unproven
  end-to-end.
* **CLOSED IN THIS WINDOW: the diverging term is the live slot set**, and it
  is an allocator-owned residue of 40404 rows on rank 1 that no tree entry
  and no resident request accounts for (section 1). What is NOT known is
  why rank 1 accumulates it and the other two do not. That is the next
  question, and `unaccounted` in the POOL CENSUS line is the instrument
  for it.
* **The wedge is the top open item.** A persistent divergence starves the
  pp->tp leg and the instance stops emitting tokens. The fix is agreement
  (broadcast or intersect the live slot set), not a longer backoff: a
  backoff only decides how quickly a wedge is reached.
* **The wire still carries no length.** The receiver's size check remains
  vacuous by construction; the ballot prevents a divergence from reaching the
  wire, and the representability check names it if one ever does. A 16-byte
  framed header (magic + length) would make the payload self-describing and
  is the obvious follow-up — deliberately not taken here, because it changes
  a wire format pinned by
  `test_streamed_pack_equals_the_concatenation_reference`.
* **Idle-vacate needs a PP-safe idle-session source** (§3), not a lifted PP
  refusal — the refusal's stated reason (host pool rows sized from the boot
  vector) is real, and `GdnStateStore` shows the dependency is sourcing, not
  storage.
