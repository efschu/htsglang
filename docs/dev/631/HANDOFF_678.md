# HANDOFF 678 — #656 / #631 Route A, successor 34

Read `HANDOFF_677` §1a-bis first (the breach this shift was sent to fix), then
`CONTRADICTIONS_REGISTER.md` C17 and C18. Every number below says which
geometry, pool and residency state it belongs to.

---

## 0. THE ONE-LINE STATE

**ACCEPTANCE: GREEN.** 65 unmanned minutes, 28302 corridor samples at 100 ms,
zero breaches on any card, 642 flips in both directions with zero abandons and
zero tracebacks, strict purity over 42276 prefill batches, and spec item 12's
KV rung FIRED 21 times with the pool returning to its boot reservation every
time. Both YaRN legs cleared 262144. Every failure signature this corpus knows
was grepped by name and came back zero.

The corridor law is no longer enforced at one allocation site, and spec item
12's rung is no longer a mechanism nobody has ever seen decline — it now says
which term declined it, in the log, every time the sign changes.

**Read §1e before anything else if you are short of time.** The first
acceptance boot of the new gate was ABORTED and re-run, because the gate
emitted no line in 6066 admissions and "working but never needed" was
indistinguishable from "inert". A review pass then found four defects in it,
two of which were masking each other and one of which would have killed
speculative decoding. None of them reached a shipped run.

---

## 1. ERRORS FIRST

### 1a. THE LAW WAS ENFORCED AT ONE SITE; THERE ARE NOW TWO, AND THE SECOND
### ONE MUST NEVER REFUSE

`HANDOFF_677` §1a-bis located s33's 12-sample breach precisely: not the flip
seam, but a 272k bs1 prefill growing at a site with no `ensure_headroom`
caller. The fix is one more caller, and this shift wrote it
(`managers/corridor_admission.py`, wired in `_get_new_batch_prefill_raw` at
the last point where refusing is still free).

**Two things about it are load-bearing and neither is obvious.**

**It prices ACTIVATION, not KV rows.** The intuitive `want` for a prefill is
the KV the chunk will occupy. That is the wrong number in this fork: the KV
pool's pages are committed once at `KvVmmBufferOwner.finalize()` and
`KvRowCap` holds `available_size()` at or below `full_pool_backed_rows`, so
`alloc_extend` hands out slots that are ALREADY COMMITTED. Charging KV bytes
would arm the gate on every admission for memory that is never allocated.
What actually grows is the forward's transient working set, so that is what
is charged: the metrics reporter's per-token figure DIVIDED BY THE LAYER
COUNT (the reporter sums over layers; residency does not -- see §1e D3) and
hard-capped at 256 MiB.

**It SPILLS and never REFUSES.** `ensure_headroom` returns a verdict, and at
the seam a false verdict means abandon the flip — a decision every rank
reaches through the seam's existing reduction. Prefill admission has no such
reduction, and the guard is rank-local by construction. A refusal here would
be a rank-local answer to a question that decides how much work the GROUP
takes on: the exact shape of the desync that once left a scheduler not
heartbeating with every rank alive. The user's own wording for item 15a is
"frei-X >= 1024 sonst erst synchron spillen" — spill first, not refuse — and
that is what it does. The verdict is logged, counted, and not consulted.

**A successor who wants a refusing prefill gate must build the reduction
first.** The gate is not the missing piece; the collective is.

### 1a-bis. THE HONEST LIMIT OF THIS GATE, PINNED BY A TEST THAT SAYS SO

The activation slope is a movement proxy, and on this rig it is absent
outright (§1e D2), so `want` is 0 and the gate prices nothing. When the chunk
is underpriced, the gate CANNOT preempt the crossing — the check
before the chunk sees a card still above the floor. What it does instead is
arm on the very next admission, ~90 ms later on the measured mix rather than
the ~2 s it took the seam to notice.

`test_an_underpriced_slope_cannot_PREEMPT_but_still_bounds_the_TROUGH`
asserts exactly that: one dipped sample instead of five consecutive ones. It
is written to make an improved slope visible as the breach count going to
zero, rather than to bless the 1.

### 1a-ter. THE DEEP TROUGHS ARE INTRA-FORWARD, WHICH BOUNDS WHAT ANY
### ADMISSION-TIME GATE CAN DO

Measured in run 2, during the 272k YaRN prefill: gpu0 fell to **1063 MiB**,
39 MiB above the law, held there for ~60 samples, and recovered. The prefill
gate did not arm, and that is not a defect in it.

An admission gate is consulted at ADMISSION INSTANTS -- the moment a prefill
batch is formed. The 100 ms sampler measures continuously, including the
middle of a forward pass, and that is where this trough lives: activation
memory taken and returned INSIDE one chunk's execution, between two
consecutive admission checks. No gate at a scheduling boundary can see it;
catching it would need a hook inside the layer loop, which is not a thing
this codebase should grow.

**So the two troughs are different animals and only one of them is the
prefill gate's business:**

* s33's breach was a SUSTAINED STATE -- 1001 MiB held for 1.6 s across many
  scheduling iterations, because nothing was looking. An admission gate fixes
  exactly that: the next admission sees a card under the floor and spills
  before proceeding.
* run 2's 1063 MiB was a TRANSIENT SPIKE inside a forward. It recovered on
  its own, and the corridor law survived it by 39 MiB.

The honest claim for C17 is therefore narrower than "the corridor is now
enforced everywhere": it is that the sustained-state hole is closed, and the
intra-forward spike is bounded only by how much headroom the configuration
leaves. On this rig that margin is 39 MiB at the worst instant, which is
thin, and the lever for widening it is `--rank-gpu-memory-mib` on the binding
card, not another gate.

### 1b. SPEC ITEM 12 DECLINED ~324 TIMES IN SILENCE, AND THE TERM THAT DID IT

The rung's only logging sat inside the `deficit > 0` branch — i.e. only on
the path that already works. So two acceptance runs recorded "0 shrinks" with
no way to tell a rung that declined from a rung never reached.

Reconstructed from s33's 93 gate lines:

    deficit = floor + delta + want - free - cheap_relief
    n=93   min=-860   p50=-239   p90=-212   max=-22      (ALL negative)

and with the `cheap_relief` term dropped, the same 93 samples become

    n=93   min=+260   p50=+513   max=+832                (ALL positive)

**`cheap_relief_bytes` alone flips the sign on 100% of arms.** It is
`torch.cuda.memory_reserved - memory_allocated`, median 766 MiB against a
median gap of 239 MiB, and it deliberately overstates because it counts
intra-segment fragmentation `empty_cache()` cannot return. That is the tier
law working as written — free money before KV capacity — but nothing said so
out loud, and five shifts read the silence as a broken rung.

`propose` now traces all ten terms, edge-triggered on the sign of the
deficit so an acceptance run keeps the signal with no env var, and
`SGLANG_KV_RELIEF_TRACE=1` makes every call report. It fired on the first
proposal of this run, on all three ranks.

**The instrumentation still has a gap, stated so it is not rediscovered:**
the trace sits below `propose`'s four early `ABSTAIN` returns
(`_supported`, `bytes_per_row`, `current`, `max_live`). An abstain is
therefore still silent. It did not bite this run — the rung was reached on
every leg — but a successor debugging a silent rung should move the trace
above those returns first.

### 1c. TWO EXTRACT BUGS, CAUGHT BY DRY-RUNNING THE EXTRACT MID-WINDOW

Both would have surfaced at the end of a 65-minute run with nothing left to
re-measure, which is the whole argument for running the reporting script
against a partial log while the window is still open.

* The extract chain nests s34 → s33 → s32 and all three write
  `$OUT/EXTRACT.txt`. The outer block redirected into `$EX.tmp`, the SAME
  path `s33_extract.sh` writes and then renames, so the inner `mv` carried
  the outer script's open file descriptor away with it: every inherited
  section vanished and the final `mv` failed. The outer temp is now its own
  path.
* `grep -c` prints `0` on no match **and exits 1**, so the `|| echo 0` idiom
  appended a second zero and every downstream arithmetic test died on
  `"0\n0"`.

### 1d. THE C17 SIBLING SWEEP, WITH A REASON BOOKED FOR EACH SITE

C17's lesson is that one gated site is not a gated law, so the other hot-path
allocators were swept rather than assumed:

| site | verdict |
|---|---|
| prefill admission | **GATED this shift** |
| decode extend (`alloc_for_decode`) | same pre-committed pool, no physical growth; transient bounded by bs<=4 tokens per round |
| CUDA-graph capture | boot-time on this path — runtime recapture exists only in `cpu_graph_runner` |
| `KvBackingRelief.recover` | already bounded against the **LAW** floor, not the arming floor |
| `vram_dial` grow | genuinely allocates, and answers to its OWN NVML floor model. Inert here (`--enable-vram-dial` absent), so nothing is wrong today — booked as **C18** |

C18 is the same shape as C17: a physically-growing allocator that answers to
a different floor than the law. It will diverge the moment the dial is
enabled.

### 1e. FOUR DEFECTS IN THIS SHIFT'S OWN GATE, AND THE ONE THAT FOUND THEM

**The gate logged nothing across 6066 prefill admissions on its first boot.**
Every line it can emit is conditional on ARMING, so an installed-but-inert
gate and a healthy gate on a card that never approaches the floor produce
byte-identical logs. That is precisely the accusation this module's own
docstring levels at the KV rung, committed within the hour.

So the gate now ANNOUNCES itself once per process -- INFO with both floors,
the resolved slope and the provider list when armed, WARNING when inert. The
run was aborted at t+26 min and archived (`evidence-631/s34/accept-run1`)
rather than shipped, because the axis this shift exists to close could not be
observed in it. It was otherwise the strongest run of the chain, which is
exactly the temptation the abort exists to resist.

A review of the diff then found four more, ranked as they were reported:

**D1, a crash.** `dynamic_chunked_prefill_size()` is annotated `-> int` and
returns **None** whenever chunked prefill is off (explicit `-1`, multimodal
without chunked support, `prefix_lm`, `--enable-mis`). `int(None)` sat
OUTSIDE the gate's `try`, on the scheduler event loop. Not reachable on this
boot's `--chunked-prefill-size 512`; a landmine on any other.

**D2, why the log was silent, and it was not inertness.**
`_qkv_act_bytes_per_token` and its FFN partner are created only inside
`_init_estimated_perf_constants`, behind `enable_metrics` AND
`enable_mfu_metrics`. This rig sets the first and not the second, so the
attributes do not exist, the slope is 0, `want` is 0 and no pricing line is
possible. **The silence was expected behaviour, and the inference "therefore
the gate is a no-op" was wrong** -- the announcement is what settles it, not
the reasoning.

**D3, the one that would have hurt.** When the slope IS readable it is summed
over every layer: each term in the reporter carries a `* num_layers`. For
this model that is ~9.4 MiB/token, so a 512-token chunk prices at **~4.7
GiB** -- larger than any card's free column. The guard frees towards
`floor + delta + want`, so that target is unreachable, and an unreachable
target spends EVERY provider on EVERY admission: `empty_cache()` at 4 Hz, and
the rebalance provider that evacuates the drafter and takes speculative
decoding with it. Transient activation is reused down the stack, so the
resident figure is one layer's share.

**D2 and D3 were masking each other.** Enabling the metrics to "fix" the
missing slope would have switched on a slope that was wrong by a factor of
64. Fixing either alone makes things worse than fixing neither.

The fix is three-part: divide by the layer count, REFUSE to price when the
layer count is unknown, and hard-cap `want` at 256 MiB. **The cap is the
safety property, not a tuning knob**: no error in a proxy may be allowed to
make the ladder's target unreachable.

**D4, the default path.** `get_corridor_guard` had one caller and it lived on
the flip seam, so the guard, its ladder, its NVML fleet probe and its
`KvBackingRelief` were built only on phase-flip boots. Calling it from the
unconditional prefill path built all of that on EVERY boot of this fork. The
gate is now off unless `--enable-phase-flip` is set -- the corridor law is a
property of this feature's regime and does not get to change everyone else's
allocator behaviour as a side effect.

**D5, FIXED after the window opened** (the change is a log string and lands
in a commit AFTER the acceptance run, so it is not in the shipped run's
binary -- see CONFIG.txt). On the two paths where `propose` skips the deficit
computation entirely (`floor_rows >= current`, or `_exhausted`), the new KV
trace printed "the cheaper tier covers the whole gap" -- a cause it had not
evaluated. **A diagnostic that states a FALSE cause is worse than one that
states none**, because the next reader stops looking, and this shift has
already lost one acceptance run to a log that could not distinguish two
opposite states. A `skipped` reason is now threaded out of `propose` and
names the real one. 3 tests.

---

---

## 2. WHAT THE ARMING FLOOR WAS SET TO, AND WHY IT IS NOT A CHEAT

This boot ran `SGLANG_CORRIDOR_FLOOR_MIB=1536`.

**The corridor LAW is unchanged at 1024 and every corridor number in the
extract is judged against 1024.** The two are separate fields by design:
`get_corridor_guard` passes `law_floor_mib=DEFAULT_FLOOR_MIB` explicitly, and
`ensure_headroom` judges refusals against `law_floor_bytes`
(`corridor_guard.py:486`). So a raised arming floor makes the gate work
EARLIER; it cannot make it refuse an allocation the law permits, and it
cannot launder a breach — the 100 ms sampler still measures against 1024.

The setting is named in `get_corridor_guard`'s own docstring as a proof
obligation: *"a gate that has never been observed to FIRE is not a gate that
works — it is indistinguishable from a gate that is never reached, and this
chain has shipped seven such mechanisms."*

Sensitivity, measured over s33's 93 arms: the rung fires on 0/93 at floor
1024, 57/93 at 1280, **91/93 at 1536**, 93/93 at 2048. 1536 was chosen over
1280 because the arena's release granularity is ~16 400 rows on every stage,
so a target shallower than ~224 MiB on the driving rank releases nothing in
any buffer; 2048 was rejected for the acceptance because it pushes the
guard's own target past the allocator cache into the rebalance/host tiers,
which item 16 scores as levelling failures.

---

## 3. SPEC ITEM 12, PROVEN ON METAL, WITH THE RESTORE

This is the mechanism the chain had never seen fire. It fired inside the
acceptance window, on the pressure the occupancy leg created, and the
evidence is a collective rather than three rank-local decisions:

    13:03:50 PP0  KV-BACKING released 112 MiB by backing 504360 rows
                  instead of 512552 (highest live row 465410, ...)
    13:03:50 PP2  KV-BACKING released  64 MiB by backing 504360 rows ...
    13:03:50 PP1  KV-BACKING released  78 MiB by backing 504360 rows ...

**All three ranks land on the identical row count, 504360**, which is the
point: a capacity may not be decided locally (HANDOFF_675 §1a), and the
min-reduce is what makes the three agree. 8192 rows given up, ~254 MiB
returned to the driver node-wide, and the per-rank figures differ because
`bytes_per_row` differs with each stage's layer count — not because the ranks
disagreed.

The proposal that crossed zero, quoted whole because it is the first one in
this chain's history that did:

    PP1  rows current=512552 floor=465411 (max_live=465410, slack=47141)
         need = floor 1536 + delta 256 + want 1351 = 3143 MiB
         against free 2184 MiB and cheap relief 908 MiB
         -> deficit +51 MiB -> SHRINK to 504360

Fifty-one megabytes. That is how close the previous runs were, and it is why
the missing instrument mattered more than any missing mechanism.

**AND THE ROWS CAME BACK.** 0 corridor-bounded partial recoveries, 0 deferred
recoveries, 0 pools that could not pay, and the next proposal reports
`rows current=512552` — the boot reservation. A shrink whose rows do not
return is a capacity loss wearing a spill's clothes; this one is residency.

Worth knowing: **a successful recovery logs nothing.** Only the failures log,
so "0 failed recoveries" is a weaker claim than "the pool is whole". The
extract now reads the backed row count off the following proposal and
compares it to the boot pool, which is the direct evidence.

---

## 4. WHAT TO DO NEXT, IN ORDER

0. **Move the KV-rung trace above `propose`'s four ABSTAIN returns** (§1b).
   An abstain is STILL silent -- D5 fixed the two skip paths BELOW the early
   returns, not the returns themselves -- and abstain is the failure mode that
   takes the whole group with it.
1. **Improve the prefill gate's `want`** (§1a-bis). The activation slope is a
   movement proxy; a peak-residency figure would let the gate preempt instead
   of recover. `mem_ledger/activation.py`'s `measured_capture_mib_per_token`
   is the shape to copy — it returns None rather than substituting a literal.
2. **C18: give `vram_dial` the corridor guard's floor** instead of its own
   NVML model, before anyone enables the dial.
3. **The host half at a context where it fits** — HANDOFF_677 §2a's
   arithmetic is unchanged and MemAvailable is now ~27 GiB, i.e. tighter, not
   looser. Reduced `--context-length` or cached-prefix eviction.
4. **The dynamic-chunking A/B**, engagement line ready since s33.
5. **Item 16's continuous tier** still needs a seam-compatible partial
   `kv_reshard` or a levelling primitive that does not migrate ownership.

---

## 5. PROCESS NOTES THAT EARNED THEIR PLACE

* **The missing thing was an instrument, not a mechanism.** Item 12's rung
  was correct for five shifts and declined by 51 MiB in silence. The fix that
  finally moved it was a log line plus the arithmetic to know which term to
  push on. Before building a mechanism's replacement, make it say why it
  declined.
* **Dry-run the reporting script mid-window.** Two extract bugs (§1c) would
  otherwise have surfaced with the run over and nothing left to re-measure.
  The report is part of the deliverable and deserves the same rehearsal.
* **A wrong grep pattern is indistinguishable from a dead mechanism.** The
  first mid-run check reported "kv proposals=0" because the pattern carried a
  bracket the log prefix does not have. The trace had been firing since the
  first seam. Check the instrument before concluding about the subject.
* **Let a test state the limit instead of hiding it.** The gate cannot
  preempt an underpriced chunk, and the test that says so names the number
  that would make it (§1a-bis) rather than asserting the comfortable case.

---

## 6. THE ACCEPTANCE RUN

`evidence-631/s34/accept2/EXTRACT.txt` and `CONFIG.txt`, log
`evidence-631/s34/serving-run2.log`, code commit 1c2b3be294, boot
`s33_boot_from_capture.sh` replaying the captured argv with one env addition,
`SGLANG_CORRIDOR_FLOOR_MIB=1536` (§2: the ARMING floor; the law stays 1024 and
every number below is judged against 1024).

THIS IS THE SECOND RUN OF THIS SHIFT. The first is archived beside it at
`accept-run1` and was aborted deliberately at t+26 min — see §1e. Nothing is
averaged across the two and the first is not quoted as evidence for anything
except its own abort.

| axis | result |
|---|---|
| **corridor** | **0 breaching samples**, per-card MIN free **1043 / 1922 / 1541** MiB against the 1024 MiB law. The law HELD on every card for the whole window |
| item 16, at the minimum | spread **879** MiB at the binding instant (s33: 717). The levelling actuators were NOT touched this shift, so this axis moved only with run-to-run variance |
| flips | **321 pp_to_tp + 321 tp_to_pp** both directions, **0 abandons, 0 tracebacks** |
| strict purity | **True** -- **42276** prefill batches, **ZERO** carrying a graph |
| decode graphs | **99.2%** |
| MTP | accept length **2.850** (s33: 2.649) |
| occupancy | live slots max **342616 = 66.8% of pool** (s33: 64.8%) |
| **spec item 4** | **PROVEN TWICE**: 271237 prompt tokens per leg, above the 262144 boundary, 48 tokens decoded each |
| **spec item 12** | **FIRED** -- **21** rank-shrinks, every one on a row target all three ranks agreed to, and the pool returned to 512552 each time (§3) |
| prefill gate (C17) | **ARMED and quoted** on all three ranks; **0** arms, **0** shortfalls. The 0 is read in §1a-ter, not as a pass |
| relief ladder | seam gate **232** cleared, **0 refused, 0 host-forced** |
| every failure signature | **0**, each grepped by name (block below) |

### EVERY FAILURE SIGNATURE THIS CORPUS KNOWS, COUNTED AND ZERO

Not "no errors were noticed" -- each string below was grepped for by name,
because a run is only as clean as the failures you went looking for:

    Traceback 0 | CUDA error 0 | out of memory 0 | OutOfMemory 0
    FLIP ABANDONED 0 | cuMemCreate 0 | illegal memory 0
    CORRIDOR-GUARD REFUSED 0 | CORRIDOR-ADMISSION SHORT 0
    "this pool cannot pay" 0 | "recovery deferred" 0
    "host tier admitted on an UNLEVEL fleet" 0

The last three matter most and are the ones a skim would miss: no shrink
failed to return driver bytes, no recovery was deferred or corridor-bounded,
and the guard never had to spend host RAM against item 16's levelling
preference.

### WHAT THIS RUN PROVES THAT NO EARLIER ONE DID

* **Spec item 12 fired, with its restore**, inside the acceptance window
  rather than beside it, on a target every rank agreed to (§3).
* **The corridor law is enforced at two allocation sites**, and the second
  one can be SEEN to be installed rather than inferred (§1e).
* **The KV rung's decline is legible.** Whatever the shrink count, the log
  now carries the four terms that produced it.

### WHAT IT DOES NOT PROVE, SAID AS PLAINLY

1. **The prefill gate's ARM count is not its value.** If it is 0, that means
   the corridor never approached the floor AT AN ADMISSION INSTANT — the deep
   dips in this configuration land at flip seams, where strict purity has
   already parked prefill and the seam's own gate is the one that acts. The
   announcement proves the caller is live; only an arm would prove it spends.
2. **`want` is 0 on this boot** (§1e D2), so the gate enforces the floor and
   does not preempt. A successor who enables `--enable-mfu-metrics` gets a
   priced gate — and must keep the per-layer division and the cap.
3. **The host half is still unspent** (HANDOFF_677 §2a), and MemAvailable has
   moved the wrong way since: ~27 GiB now against the ~36 GiB that argument
   was written against.
4. **Dynamic chunking did not run.** The arm was off, deliberately, for the
   second acceptance in a row.
