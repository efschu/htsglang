# HANDOFF 605 R2 — the demand is deterministic, and R1's headline was wrong

Shift: flightrec-r2, 2026-08-13. Branch `feat/vram-flightrecorder-605`,
commit `3bea8f66c1`. Evidence: `/spinning/evidence-631/flightrec-r2/`.

**Read §1 before quoting anything from HANDOFF_605_FLIGHTRECORDER (R1).** R1's
central claim is retracted here, by measurement.

---

## 1. RETRACTION — R1 said the pool is sized differently every boot. It is not.

R1 reported a per-post spread over "14 boots of one unchanged config"
(`kv_pool_target` 1364–2408 MiB) and concluded the corridor overshoot was made
of boot-to-boot nondeterminism in the KV pool's sizing.

**The 14 boots were not one config.** The peer shift (#656) was changing flags
between them — `--phase-flip-tp-vector` among others — and R1 pooled ≥11
distinct configurations into one distribution and called the resulting variance
nondeterminism. That is the sample-breadth error in its textbook form: the
conclusion was drawn as wide as the QUESTION rather than as wide as the SAMPLE.

Grouping the same 14 boots by their full post signature gives 11 signatures, one
of which has four members. Those four, spread over 2.5 hours and interleaved
with the others, are **byte-identical on every post and every card**. A fifth
boot taken by this shift with the same captured argv — but running a DIFFERENT
code tree, 38 files ahead — reproduced it exactly:

| boot | 5090 resident | 3080-5c64 | 3080-62db |
|---|---:|---:|---:|
| 1006903-1786601360 | 28436 | 18002 | 16674 |
| 1132987-1786604465 | 28436 | 18002 | 16674 |
| 1263882-1786607702 | 28436 | 18002 | 16674 |
| 1388228-1786610712 | 28436 | 18002 | 16674 |
| **1464299-1786612548 (R2, different code)** | **28436** | **18002** | **16674** |

**Given its configuration, the boot's resident demand is exact and repeatable to
the byte.** The only quantity that moves at all across those five boots is the
NVML-unattributed driver line on one card (164 → 166 → 276 MiB), which is §4.

So the corrected answer to "why is this not computed exactly beforehand": it
CAN be. Nothing in the demand is random. What was missing is an instrument that
could see it, and a per-boot record of the configuration so two boots could be
told apart — the second of which is §3.

## 2. `weights_target`'s 702 MiB — hypothesis falsified

R1 flagged a 702 MiB spread in `weights_target` and guessed
`--phase-flip-policy auto` was picking layouts. **Falsified.** Across the 14
boots `weights_target` takes exactly three values per card, and the increments
are IDENTICAL on all three cards: +28 and +674 MiB
(5090 14002/14030/14704; 3080-5c64 8658/8686/9360; 3080-62db 9626/9654/10328).

A layout re-partition moves weight between cards and conserves the total — one
card up means another down. All three moving up together by the same amounts is
not a re-partition; it is a per-rank constant added by a changed flag. The
values track the peer's `--phase-flip-tp-vector` edits, i.e. this is
between-config variation (§1), not a layout lottery.

## 3. The modelled ledger now exists on ship boots

The dump had never fired on the shipped configuration. The reason is structural:
the config pins `--rank-gpu-memory-mib`, which is the pin path, and the pin path
skips the planner, so no ledger was ever constructed. Fourteen boots of measured
marks sat beside zero modelled counterparts, and `reconcile.py` — the whole
payout of #605 — had nothing to compare against.

`ServerArgs.__post_init__` now builds one **for the record** once the
configuration has resolved, gated on the recorder already being armed. The
ledger is discarded: never returned, never consulted for a size, so a config
that does not use the ledger keeps not using it and merely stops being
unobservable. `ledger_1464299-1786612548.json` is the first such dump in this
chain.

## 4. The residual, itemized — it is not a hidden tenant

R1 left 166–276 MiB per card named but not itemized. NVML's own per-process list
settles it by direct observation: **exactly one PID per card, no others**, on
every boot examined.

Excluded by measurement, not by argument:
- tokenizer/parent second context (#237) — no second PID on any card
- lane pools outside the rank budget (#400) — same
- multimodal second context (#403) — same

What remains is card-level `used` that NVML attributes to **no process**:
driver-side per-context bookkeeping that `nvmlDeviceGetComputeRunningProcesses`
does not bill to anyone. It is **irreducible with NVML** — naming it further
needs a non-NVML instrument. It is bounded and now has a measured band:
**164–276 MiB per card**, and it is the ONLY term that varies between
byte-identical boots.

## 5. The commit watermark is now a recorded post

`KvVmmArena` carries a read-only `arena_census()` (a `WeakValueDictionary`, not
a `WeakSet` — observing an arena must not keep it alive, a lesson this tree
already learned once in the attention workspace registry), and `mark()` records
`kv_arena_reserved / backed / retained` at every phase. R1's chain is now
measured rather than argued.

At `first_forward` on the R2 boot:

| card | arena reserved | arena backed | uncommitted | free | claimable to the law |
|---|---:|---:|---:|---:|---:|
| 5090 | 31182 | 22690 | 8492 | 3463 | 2439 |
| 3080-5c64 | 20920 | 14058 | 6862 | 1776 | 752 |
| 3080-62db | 19622 | 12838 | 6784 | 3214 | 2190 |

`retained` is 0 on this boot and is reported separately from `backed` on
purpose: parked physical handles are memory NVML charges the process for while
`backed` does not, and folding them together would recreate the false-zero of
R1 §1b one level down.

**The number for the KV-universe shift:** the arena holds 6.8–8.5 GiB of
reserved-but-uncommitted address space on every card, so the commit watermark is
**not capacity-limited**. Using the under-load worst instant (§6) rather than
the boot snapshot, the safely claimable amounts are **+1777 / +518 / +1038 MiB**
(5090 / 5c64 / 62db). That is policy headroom, available today, with the arena
already reserved to absorb it.

## 6. Corridor under load — R1's snapshot caveat is lifted

75 s, 100 ms cadence, 750 samples per card, under five real generations
(`corridor_under_load_cheap.json`):

| card | free min | free max | margin to 1024 | breach |
|---|---:|---:|---:|---|
| 3080-5c64 | **1542** | 3626 | +518 | no |
| 5090 | **2801** | 5507 | +1777 | no |
| 3080-62db | **2062** | 4546 | +1038 | no |

Both samplers reduce on the MINIMUM, never the mean: a floor is broken by the
worst instant. This is corridor-law-grade evidence — a continuous under-load
minimum — and no longer a boot snapshot. Even at the worst instant, 518 MiB sits
unclaimed on the binding card.

### ERRORS FIRST — the instrument's own cost

The first sampler ran at **25250.9 us per card per sample, a 75.7% duty cycle**,
because the registry's `memory_info_for_uuid` opens an
`nvmlInit`/`nvmlShutdown` pair and scans every device by UUID on each call —
thirty init/shutdown cycles per second at this cadence. An instrument spending
three quarters of the wall clock measuring is competing with what it measures.

Handles are now resolved once per run: **28.9 us per card, 0.0867% duty, 0
overruns** — 874x cheaper. The expensive and cheap runs agree within ~150 MiB,
which is what makes the cheap reading trustworthy rather than merely convenient.
Every summary publishes `sample_cost_us_*` and `duty_pct` beside its readings so
a reader can reject a reading taken with too costly an instrument.

Also errors-first: this shift's own restore monitor false-positived on the
string `sigquit` inside the `server_args=` config dump. The filter was widened
for coverage and caught a field name; tightened to
`Received sigquit|Traceback \(most recent|CUDA_ERROR_OUT_OF_MEMORY|KvReshardError`.

## 7. Open, in order

1. **`corridor_trace.py` has no production call site.** The in-process sampler
   is tested and ready but the natural home is the scheduler, which belongs to
   the #656 shift. The out-of-process sampler covers the law's own quantity
   today; the in-process one adds per-rank torch and arena columns when a
   non-conflicting call site exists.
2. **`reconcile.py` has still never RUN** against a ship boot. The counterpart
   now exists (§3) and the measured posts exist; nobody has yet put them in one
   table on the same boot. That is the next payout and it needs no GPU.
3. **The driver-unattributed band (§4) is unexplained at 164–276 MiB** and is
   the only non-deterministic term. Non-NVML instrument required.
4. §5's claimable numbers are the boot's and the leg's. A different traffic mix
   moves the trough; the sampler is the way to re-measure it, cheaply.

## 8. Register

C27–C30 appended to `CONTRADICTIONS_REGISTER.md`: the R1 spread retraction, the
weights-layout falsification, the pin-path reason the ledger never dumped, and
the residual's excluded candidates.
