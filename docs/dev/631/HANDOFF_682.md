# HANDOFF 682 — #656 / #631 Route A, successor 38

A hardening shift against HANDOFF_681's two named residuals. Both are closed
as far as evidence allows, and one of them closed by being **falsified in the
form it was handed over**, which is the more useful half of this shift.

---

## 0. THE ONE-LINE STATE

**The KV rung can no longer be starved into killing the instance, the seam
gate can no longer demand from it what it cannot fund, and the suite is
FULLY GREEN for the first time in this chain: 1052 passed, 0 failed.** The
inherited `_staging_bytes` red was not a production defect — it priced ONE
seam wave and measured a SIXTEEN-wave run.

Read §1a before quoting anything about an unpriced seam tail: the metal half
of that claim came from joining a rank id to a card index, and on this rig
those are different permutations.

---

## 1. ERRORS FIRST

### 1a. THE "UNPRICED TAIL" WAS A CARD MIX-UP, AND THE PERMUTATION IS A TRAP FOR EVERYONE

HANDOFF_681 §2b reported that a cutover entering HIGH draws ~1040 MiB on the
binding card "that `_staging_bytes` does not price". The draw is real. The
attribution is not.

`--rank-gpu-id 0,1,2` is read in **CUDA** order, and `CUDA_DEVICE_ORDER`
defaults to FASTEST_FIRST here, so **CUDA device 0 is the 5090** while
**nvidia-smi index 0 is a 3080**. Every corridor CSV in this corpus is in
nvidia-smi order; every log line is prefixed with a RANK. The mapping on this
rig is:

    rank 0 (PP0)  ->  nvidia-smi gpu1   (5090, 31800 MiB budget)
    rank 1 (PP1)  ->  nvidia-smi gpu0   (3080, 14000 MiB) <- the BINDING card
    rank 2 (PP2)  ->  nvidia-smi gpu2   (3080, 15600 MiB)

Confirmed three independent ways: the per-rank MiB vector against the card
sizes; the guard's own free readings (PP1 clears at free 1886/2726 MiB, which
match the gpu0 column's p50 of 2349, while PP2's 2968/3236 match gpu2's 2715);
and the boot's own profile (rank 0 capacity 619990 tokens against rank 2's
308808). PP0 armed its guard **10** times in 65 minutes while PP1 armed 232 —
the rank that barely arms cannot be the card with the tightest corridor.

Joined correctly, with each seam's own price taken from its `staging reserved`
field (`scripts/s38_seam_price_vs_draw.py`, over s37's own accept2 window):

| rank / card | pp->tp priced (MiB) | deepest NVML drawdown (MiB) |
|---|---|---|
| rank 1 / gpu0 (binding) | p50 **1177**, p90 1225, max 1478 | max **504** |
| rank 2 / gpu2 | p50 976, max 1305 | max 774 |
| rank 0 / gpu1 (5090) | p50 2555, max 2802 | max 1504 |

**On every rank the gate reserved more than the card gave up.** There is no
unpriced tail visible at the corridor, and the 1040 MiB figure is a 1177 MiB
price on a card the price did not belong to.

> The generalisable half: *a rank id and a device index are two different
> permutations until a boot line says otherwise.* Registered as law 9.

### 1b. THE INHERITED RED WAS A UNIT MISMATCH IN THE TEST, NOT A HOLE IN THE GATE

`test_staging_bytes_predicts_the_measured_peak` has been red across several
shifts with "`_staging_bytes` 20.2 MiB over-reserves against the measured live
set 3.8 MiB" — a 5.4x over-reservation that would be a serious defect.

It called `_staging_bytes(tr, dir, src, dst)` — **one wave over every layer** —
and then measured a run that the seam executes in **16 waves**. Production
never spends the single-wave number: `_execute` passes `_flip_waves(direction)`
into the same call. Priced the way production prices it, the same fixture
gives 2.398 MiB.

The test now prices with the run's own wave plan and keeps both of its
original claims, each against the quantity it is actually about. **Suite:
1052 passed, 0 failed.**

### 1c. WHAT IS LEFT OF RESIDUAL 2, MEASURED AND BOOKED AS C21 RATHER THAN FIXED

With the wave plan matched, the price is **0.64x the measured live set**
(2.398 vs 3.769 MiB on the three-rank CPU fixture). The formula models the
seam's peak as `incoming + max(outgoing, local)`; the probe's high-water holds
an outgoing leg (0.689 + 0.684 MiB) beside a 1.025 MiB local read and its
1.025 MiB gather window.

**Not widened, deliberately.** It is not reproduced on metal — §1a's table is
the evidence, and torch's allocator cache is what absorbs it, which is also
why 348 flips produced 0 breaches. The gate's only action is to refuse, and a
refusal does not drain the resident set it refused on, so reserving for a
transient the driver is never asked for buys an earlier wedge (HANDOFF_664
§13c). Pinned as a ONE-SIDED ratchet: closing the gap passes, widening it
fails.

An experiment that did NOT work, recorded so it is not repeated: releasing the
wave's `jobs` list at the end of each wave (its entries are views into the
incoming payload, and it stays bound across the loop edge) changed the
measured peak by **zero bytes**. Reverted rather than shipped as a plausible
no-op.

### 1d. TWO PROBE BOOTS DIED FOR REASONS THAT WERE NOT THE FEATURE

Recorded because the next shift will hit them and should not spend the time
twice:

* **An external SIGTERM.** The first probe was decoding healthily at 154
  tok/s and took `SIGTERM received. Draining requests and shutting down` at
  20:14:05 — a graceful stop from OUTSIDE. My arbitration heartbeat had been
  killed with the harness task that launched it, so no peer could see a
  holder. **Launch the heartbeat with `setsid ... & disown` and verify the
  file's mtime advances**; a `nohup` inside a tool call is not enough.
* **A host RAM OOM at boot.** The second attempt died with the cgroup's
  `oom_kill` counter going 9 -> 10 and `memory.peak` at 120.3 GB of 120 GB.
  Rank 1's traceback is a gloo `Connection closed by peer`, which is the
  SYMPTOM of a sibling being killed, not a cause. This rig boots close to its
  host RAM ceiling; s37's own green window already recorded 9 such kills.

---

## 2. THE MECHANISM (register C20, residual 1)

Two terms. The first is the safety property; the second is what keeps it
cheap.

### 2a. THE KV RUNG KEEPS AN ADMISSION RESERVE

`KvBackingRelief._floor_rows` was `max_live + 1 + margin_rows`: it protects
the rows that EXIST and reserves nothing to admit new work with. It now adds
an **admission reserve** above the live high-water mark.

| | |
|---|---|
| where | `kv_backing_relief.KvBackingRelief._floor_rows` |
| how much | the scheduler's own `chunked_prefill_size` (512 on the ship config — the exact number the failure quoted back), via `_admission_reserve_rows` |
| override | `SGLANG_KV_ADMISSION_RESERVE_ROWS`; **0 restores the previous floor exactly** |
| why ABOVE `max_live` | every id above the high-water mark is unallocated by definition; ids below it may all be in use. Only that range's freeness is guaranteed |
| group uniformity | free: `collective_kv_target` already reduces to the MAX floor across ranks, so a target clearing one rank's reserve clears every rank's |

### 2b. THE GATE MAY NOT DEMAND WHAT THE RUNG CANNOT FUND

`collective_kv_backing_relief` takes `discretionary_bytes` — the part of the
ask the caller can do without — and bounds it by `KvBackingRelief.
fundable_bytes()`, the bytes this rank can return without crossing that floor.
The seam gate declares its C20 entry margin as the discretionary half.

**The mandatory half is never bounded.** If the rung cannot fund the seam's
staging, the guard must see the full ask and refuse the seam: a free,
unanimous abandon with every request intact. Laundering that into a smaller
ask would produce a seam that "fits" and an allocator that disagrees.

Why the bound is needed at all once the floor exists: the ask sets the SIZE of
the shrink, not only its limit. `deficit = floor + delta + want - free -
cheap`, so an ask nothing can fund drives the rung to its floor on EVERY seam,
spending all of its slack every time for bytes it does not have.

---

## 3. THE EVIDENCE

### 3a. THE FORCED-MARGIN PROBE: THE SAME INPUT, AND THE INSTANCE LIVES

`scripts/s38_admission_floor_probe.sh 9`, `SGLANG_SEAM_ENTRY_MARGIN_MIB=8192`
— the configuration that killed the instance for s37 — on the tree that
carries the floor. Evidence: `/spinning/evidence-631/s38/floor-probe3/` and
`floor-probe3.out`.

| | s37 (no admission floor) | **s38** |
|---|---|---|
| margin armings | 452 | **939** |
| seam entry DELAYED | 72 | **156** |
| seam entry margin YIELDED | 35 | **78** |
| bounded rung asks (`KV rung asked for`) | — (no bound existed) | **117** |
| pp->tp cutovers | 42 before the death | **78** |
| `Available full tokens: 0` | **the death, at 19:25:15** | **0** |
| tracebacks | 3 ranks | **0** |
| gate refusals | 0 | **0** |
| corridor breaches | 0 | **0** (3926 samples) |
| corridor MIN per card | — | **1687 / 2482 / 2087 MiB** |
| soak | — | **30 ok / 0 err** |
| health at the end | **000** | **200** |

**The branches still execute — 156 delays and 78 yields, more than double
s37's counts — so this is not the survival of a mechanism that failed to
arm.** The instance served the whole 9-minute window under an ask no ladder
on this rig can fund.

Both new terms print their own account, which is how a successor can tell
they are wired without inferring it:

    KV-BACKING proposal on device 0: rows current=512552 floor=593
      (max_live=80 + admission reserve 512, slack=511959)

    PHASE-FLIP-SPILL KV rung asked for 3949 of 8192 MiB discretionary
      (pp_to_tp): it can return 4999 MiB above its admission floor, and
      asking for more would drive it to that floor for bytes it does not have

### 3b. THE CONFIRMATION WINDOW ON THE SHIP CONFIG

30 minutes on the ship config (margin 512, budget 2, lender off), booted
20:51:22Z on `4aa1bf93a1`, with two agent lanes as the traffic carrier.
Evidence: `/spinning/evidence-631/s38/ship-window/EXTRACT.txt`. **This is a
REGRESSION check against s37's standing green, not a new stamp** — the window
is half as long and its arms are not matched to s34's.

| | s34 | s37 (65 min) | **s38 (30 min)** |
|---|---|---|---|
| corridor breaches | 0 | 0 | **0** (13118 samples) |
| IN-CUTOVER min, binding card | 1043 (+19) | 1123 (+99) | **1083 (+59)** |
| flips each way | 321 | 348 | **150** |
| FLIP ABANDONED / tracebacks | 0 / 0 | 0 / 0 | **0 / 0** |
| seams REFUSED below the law | 0 | 0 | **0** |
| strict purity / decode graph | True / 99.2% | True / 99.7% | **True / 99.3%** |
| MTP accept length | 2.850 | 2.797 | **2.869** |
| YaRN leg | 271237 | 271237 | **271237** |
| pool at the last KV proposal | 512552 | 512552 | **512552** |
| live slots, max | — | 295620 (57.7%) | **327699 (63.9%)** |
| prefill / decode batches, t+25 | 14830 / 387 | 14764 / 345 | **15475 / 363** |
| median PP dwell | 17.0 s | 17.0 s | **17.0 s** |
| soak t+24 | 59 | 41 | **48** |
| soak, full window | 218 ok / 0 err | 196 ok / 0 err | **71 ok / 0 err** |

**C20 VERDICT: HELD (exit 0).** The run sits in the s34/s37 family on every
axis the margin does not touch, and is ahead of s37 on decode batches, MTP
accept length and soak completions.

**The one number that moved down, and why it is not a regression.** The
binding card's in-cutover minimum is 1083 against s37's 1123. This window ran
**hotter**: live slots peaked at 63.9% of the pool against s37's 57.7%, and
the seam's own reservation rose with it (staging mean 1317.6 vs 1300.1 MiB).
That is also why the margin's delay branch fired on the SHIPPED
configuration for the first time — 2 delays, 1 yield, where s37 recorded
none — which is the graded verdict doing exactly its job under load rather
than a new failure. Neither of this shift's terms was active: the admission
floor sat at 593 rows against a deepest cut of 448782, and the discretionary
bound never triggered.

**Read the gate's two refusal counters separately.** The extract reports
`gate: 202 cleared, 3 refused`, and the C20 block reports `seams REFUSED
(below the law): 0` with 0 abandoned flips. Both are right: the guard is
shared, and those 3 refusals are the PREFILL-ADMISSION caller
(`corridor_admission.py`), which spills and never refuses. A refusal counted
there is not a refused seam, and a successor comparing 3 against s37's 0 would
otherwise book a regression that did not happen.

On the shipped margin both terms are near-inert by construction — a 512-row
reserve against 512552 backed rows, and a margin that funds on every seam —
so the window's job is to show that the ordinary path did not move.

**Both inertness claims are visible in the window's own log rather than
argued:**

    KV-BACKING proposal on device 0: rows current=512552 floor=593
      (max_live=80 + admission reserve 512, slack=511959)

    "KV rung asked for" ............................ 0 occurrences

Zero bounded asks means the discretionary path never triggered: the 512 MiB
margin is fundable on every seam, so the rung receives the SAME
`staging + margin` it received for s37. The floor is a bound on the
pathological case, not a tax on the ordinary one.

---

## 4. WHAT TO DO NEXT, IN ORDER

0. **C21: price the seam against what it HOLDS, not against what the driver is
   asked for** (§1c). The hermetic gap is 0.64x and the instrument already
   exists; the fix is a formula change and it needs its own window because it
   moves the gate's refusal point on every seam.
1. **The margin's own sizing is still measured on PRE-margin data**
   (HANDOFF_681 §4.2, untouched). `C20_SIZING` comes from s34.
2. **A one-boot A/B of the margin's price** (HANDOFF_681 §4.1, untouched): the
   KV rung fires 348 times against s34's 21, and admission throughput is the
   axis that prices it.
3. **An abandoned pp->tp leaves the KV pool capped** (HANDOFF_681 §4.3).
4. **The abandon path does not drain the abort deferral window**
   (HANDOFF_681 §4.4).
5. **`SGLANG_FORWARD_PEAK_PATH` on the next acceptance boot** (HANDOFF_679
   §2.1, HANDOFF_680 §4.2, HANDOFF_681 §4.5).
6. **C18**: give `vram_dial` the corridor guard's floor before the dial is on.
7. **The corridor counters are still write-only** (HANDOFF_680 §4.5).

### 4b. THE CORRIDOR LAW IS NOT CONFIGURABLE, AND ABOUT FORTY PLACES ASSUME SO

Audited during this window (the second agent-traffic lane).
`SGLANG_CORRIDOR_FLOOR_MIB` moves the **arming** floor only; the LAW is
`corridor_guard.DEFAULT_FLOOR_MIB = 1024`, wired explicitly at
`phase_flip_spill.py:1422`, and roughly forty scripts plus every extract
judge against a literal 1024. That is CONSISTENT today and the split is
deliberate — a proof run must make the gate work earlier, never make the pool
permanently smaller. It is only a trap if the user's corridor ever changes
value: at that moment the arming floor would follow an env var and every
verdict in the corpus would still score against 1024. If that day comes, make
`DEFAULT_FLOOR_MIB` the single source the scripts read, rather than editing
forty constants.

### 4a. THREE ITEMS FELL OUT OF THE LIST WITHOUT BEING RESOLVED

An audit of the numbered lists across HANDOFF_680/681/682 (run as one of this
window's agent-traffic lanes) found that the 680 -> 681 transition dropped six
items, and three of them were neither done nor superseded — they simply stopped
being carried:

* **`0c`, loosen the drafter's phase precondition.** The provider refuses
  outside PP; 680 wanted the condition relaxed rather than asserted.
* **`4`, the host half / dynamic-chunking A/B.**
* **`6`, the draft-weights provider reports an ARENA COUNT, not an NVML
  delta** — which is the ledger law (register law 2) applied to a rung this
  corpus still prices from.

The other three were legitimately closed (the lender A/B ran, the green stamp
was restored, and 680's "seam staging demand" became C20). Re-entered here so
the next list inherits them; a shift that decides one is not worth doing
should say so in the Verworfenes register rather than let it fall out again.

---

## 5. PROCESS NOTES

* **State the permutation before joining two instruments** (§1a). A rank id
  and a card index agreed by accident in every previous handoff because the
  claims happened not to depend on it.
* **A test can be red for a reason that is not the code's.** Before treating
  an inherited red as a defect, check that both sides of its comparison
  describe the same run (§1b).
* **A heartbeat launched inside a tool call dies with it** (§1d). Arbitration
  that nobody can see is arbitration that does not exist.
