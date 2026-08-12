# HANDOFF 686 — #656 / #631 Route A, successor 42 (queue item #659)

The shift that built #659's second spill rung and put it on metal — and that
had two of its own premises falsified BY that metal, one of them the
group-uniformity argument the previous cut shipped. #659 is **not** closed, and
the reason is precise and reproducible rather than "ran out of time". Errors
first.

---

## 0. THE ONE-LINE STATE

**The file park tier exists, registers with MEASURED metrics on a live boot,
and the ladder now picks instead of hardcoding `tier_index=0`. What is NOT
proven is a park/unpark round trip on metal: no load shape I could construct
made kv-session-offload spill at all (2.9x pool oversubscription, a 10-way
burst and the occupancy leg all finished with `kv_session_host_ram` used = 0),
and the byte-identity METHOD I planned is independently invalid on this rig.**

Serving is UP on 30030, ship config, restored from my own commit, health 200,
confirmation window run and clean. Nobody owes a restore.

---

## 1. ERRORS FIRST

### 1a. kv-session-offload REFUSES `pp_size > 1`, SO THE BRIEF'S STEP 3 CANNOT BE RUN AS WRITTEN

The brief says "boot the Route-A recipe PLUS `--enable-kv-session-offload` on a
probe port". That boot does not exist. Measured at arg-parse, not inferred:

```
ValueError: --enable-kv-session-offload (S1) supports single-node pure TP/DCP
only (pp_size=3, dp_size=1).
```

Route A **is** PP=3 prefill → flip → TP=3 decode. So the ship recipe and the
feature under test are mutually exclusive at the argument level. Every future
"prove kvso on the Route-A recipe" ask is asking for something the boot gate
refuses, and it should be reframed as one of: prove it on the DECODE layout
(pure TP), or lift S1.

I ran it on the decode layout. That is a real deviation and it is why no
phase-flip interaction was exercised.

### 1b. THE GROUP-UNIFORMITY PREMISE OF THE ORDERING WAS FALSIFIED BY THE FIRST BOOT

The first cut of this shift quantized bandwidth onto a 0.25 GB/s grid and
argued in its own docstring that "the quantum is coarser than the spread
between ranks". The first live two-rank boot printed both ranks' registration
lines:

| rank | measured bandwidth, SAME directory, SAME instant |
|---|---|
| TP0 | **2.41 GB/s** |
| TP1 | **7.00 GB/s** |

A **2.9x** spread, because the two probes contend with each other. No absolute
grid survives that: any fixed boundary between 2.41 and 7.00 splits one medium
across two buckets, and two ranks then order a two-tier ladder differently,
park one session to two places, and the completion min-reduce turns that into a
**hang** rather than an error (register laws 8 and 14).

Replaced by a **ratio** test (`PROMOTION_RATIO = 4.0`): the configured order
stands unless a tier measures 4x faster. A ratio survives contention because
contention scales every tier on a rank *together* — two tiers on one medium tie
on every rank; the refused remote path (0.075 GB/s vs ~3 GB/s local, 40x)
clears the bar on every rank.

> **The general lesson, and it is not specific to this tier:** a per-rank
> benchmark is not a rank-uniform quantity, and rounding does not make it one.
> If a group decision must consume a measured value, consume a RATIO of two
> values measured on the same rank, or reduce it through a collective. Booked
> below as a proposed register law.

### 1c. THE BYTE-IDENTITY METHOD I PLANNED IS INVALID ON THIS RIG, AND IT FAILED OPEN

Plan: run the same greedy continuation quiescent and under pressure, compare
byte-for-byte, attribute any difference to the park round trip. Measured
result, with **zero parks and zero spills** in that run:

```
REFERENCE  digest=e214a846623f4f72 chars=78
UNDER LOAD digest=102fc89d0b247b32 chars=177
BYTE-IDENTICAL: False
```

The two differ with nothing parked, so the method measures batching
nondeterminism, not the tier. This is the known GDN prefill nondeterminism
(memory: `GDN-Prefill-Nichtdeterminismus`, upstream, ~109 tokens) arriving
where it was not expected.

**It failed OPEN, which is the part worth carrying**: an earlier run of the
same driver printed `BYTE-IDENTICAL: True` — also with zero parks. Had I
stopped there I would have shipped a green byte-identity claim for a mechanism
that never ran. The claim was only exposed as vacuous because the driver also
records the LEDGER, and the ledger said `used:park:file = 0`.

> **Any future byte-identity claim for this tier must compare the park ROUND
> TRIP** (bytes in vs bytes out of the tier, or the #224 meta/fingerprint
> identity check), never two model generations.

**The instrument was rebuilt so that it cannot pass vacuously, and the rebuild
was itself proven able to fail.** `proof_driver2.py`'s verdict is a conjunction
whose FIRST term is `parked_count > 0`, carrying its own exit code (3), so "the
mechanism did not run" and "the mechanism ran correctly" can no longer produce
the same verdict. Exercised offline against synthetic ledgers
(`evidence-631/s42/INSTRUMENT_CANFAIL.txt`):

| case | exit |
|---|---|
| **v1's exact failure: identical text, nothing parked** | **3 (hard fail)** |
| nothing parked and text differs | 3 |
| parked via disk files, text identical | 0 |
| parked via metrics bytes, text identical | 0 |
| parked via counters, text differs | 1 |
| counters report zero parks | 3 |

The text comparison survives only as a recorded OBSERVATION about model
determinism; it is no longer the pass criterion.

### 1d. NO LOAD SHAPE REACHED A SPILL, SO PARK/UNPARK ON METAL IS UNPROVEN

Three attempts against the probe, all with `max_total_tokens=16384` and a host
tier deliberately capped to exactly ONE region (`effective max_spills reduced
3 -> 1`, the server's own words):

| attempt | load | result |
|---|---|---|
| 4 sessions x ~6000 tok | 24k demanded vs 16k pool | no spill |
| 10-way burst, 400 new tokens each | all 10 completed in 25.7 s | no spill |
| `s33_occupancy_leg.py --sessions 4 --tokens 12000` | **peak 47951 concurrent prompt tokens, 2.9x the pool** | no spill |

`kv_session_host_ram` used stayed **0.0** throughout and the park directory
stayed **empty**.

**The cause is now READ OUT OF THE CODE, not guessed.** The spill trigger is in
the DECODE path, `scheduler.py:5578`:

```python
num_tokens_next = batch.new_tokens_required_next_decode()
evict_from_tree_cache(self.tree_cache, num_tokens_next)
kv_full_retract_flag = self.kv_session_offload.dcp_min_avail() < num_tokens_next
if kv_full_retract_flag and self.kv_session_offload.try_spill(batch):
```

So a spill fires when the pool cannot fund **the next decode step** — not when a
prefill is large. **Every load I drove was prefill-heavy with short
generations**, which frees on finish and never reaches that predicate. Pushing
*more prompt tokens* was the wrong axis, and 2.9x oversubscription on the wrong
axis is still zero.

Two further eligibility rules that shape any forcing attempt
(`kv_session_offload.py:950-985`, `spill_victim_candidates`):

* **the oldest normal session is TABU** under plain decode-OOM pressure, so a
  single running request yields an EMPTY candidate set — at least two
  concurrent decoders are required before a spill is even possible;
* **under EAGLE/MTP a request may only leave the batch from the BACK**
  (`spec_back_only_victim`), so speculation narrows the victim set further.

The forcing recipe that follows: a small `--max-total-tokens`, several
CONCURRENT requests with LONG `max_new_tokens`, and modest prompts — decode
pressure, not prefill pressure. Prepared as
`evidence-631/s42/probe_boot_v4.sh` (ctx 8192 / MTT 4096, so a 1 GiB host
budget holds exactly one region and the second spill demand must park) plus
`proof_driver2.py` retargeted to decode-dominated load.

### 1e. TWO PROBE MEASUREMENT TRAPS, BOTH HIT BEFORE THEY WERE AVOIDED

* **Compression.** First probe wrote zeros and measured **6.2 GB/s** fsync'd.
  That is the ZFS compressor, not the disk. Incompressible payload on the same
  path: **3.13 GB/s**.
* **Probe size.** A short write is absorbed rather than sustained:

      probe size   64 MiB  256 MiB  512 MiB  1 GiB  2 GiB  8 GiB
      write GB/s     8.69     6.65     6.80   4.27   4.92   3.13

  A 64 MiB probe overstates this volume by **2.8x**. Shipped at 256 MiB
  (127 ms/rank at boot) with the table in the constant's docstring and the
  entry labelled an upper bound.
* **Read-back is ARC, not disk**, so the tier ranks on the WRITE rate and the
  read-back is recorded in `properties` as cache-warm.

### 1f. THE FIRST PROBE BOOT DIED ON A DEVICE-SIDE ASSERT — NOT #659, BUT WORTH THE CRASH LANE

Uneven TP=3 (`--rank-tp-ratio 32,16,16`) + `--max-total-tokens 8192` + fp8 KV +
NEXTN, under 5 concurrent ~3000-token sessions:

```
jit_kernel/csrc/elementwise/kvcache.cuh:112 store_kvcache(...)
  Assertion `index >= 0 && index < size_limit` failed        (kElementBytes=256)
```

**Zero park and zero spill activity beforehand**, so it is not a #659 fault. A
KV slot index escaped the pool bound with the pool pathologically small (8192
tokens against ~15k of concurrent demand). Handing this to the #622/#649 crash
lane as a datapoint with a reproducer recipe
(`/spinning/evidence-631/s42/probe_boot_30040.sh`).

---

## 2. WHAT SHIPPED

`08585169fe` and `4bc9e0093a`, pushed to `efschu/htsglang` (remote is
**`origin`**), `feat/route-a-631`.

### 2a. THE SECOND RUNG: A FILE PARK TIER, MEASURED AT REGISTRATION

`managers/kv_spill_park_tier.py` (new). The brief's preferred second tier does
not exist on this rig — s41 refused Rig-2 RAM by measurement and peer VRAM by
the standing REBALANCE verdict — so the cheapest REAL second tier is a local
filesystem, registered as what it is: **a tier, not a remote tier**.

Nothing is read from `memtier/profiles`. `probe_park_filesystem` runs a real
incompressible fsync'd write plus a `fdatasync` latency sample against the
directory the blobs actually take, at registration, and an unprobeable path
yields `Rate.absent(reason)` rather than a borrowed number — C24's lesson as
code.

The capacity is `min(configured budget, free - df headroom)` **in that order**,
so a generous budget on a nearly-full volume yields a small tier instead of a
disk-full outage (#558 family). Three new flags:
`--kv-session-offload-park-dir`, `--kv-session-offload-park-budget-gib`,
`--kv-session-offload-park-df-headroom-gib`; a park knob with no park tier is a
hard error, and a negative budget is rejected rather than read as unbounded.

### 2b. THE SELECTION POLICY IS LIVE — THE REGISTRY IS NO LONGER AN OBSERVER

`_start_park`'s hardcoded `tier_index=0` is gone. The ladder ranks the
configured park tiers and refuses the rest by name; the cut-1 falsifier still
passes because with one healthy tier the ranking is degenerate and returns 0.

**Every input to that decision is rank-uniform, and that is the load-bearing
part**: the ask is a boot constant (`region_tokens x size_per_token`, quantized
up), capacities are measured at registration and never re-read, the ranking is
computed once and cached, and ordering is by ratio (§1b). There is deliberately
**no per-park `try/except`** around the choice — a rank that fell into one and
parked, beside a rank that succeeded and declined, would diverge into the
completion min-reduce and wedge the instance rather than error (law 14).

### 2c. THE #224 PARK COUNTERS ARE READ, AFTER THREE SHIFTS OF BEING WRITE-ONLY

Every reference in the tree was a write, so a park that failed and a park that
never happened produced the same silence. Now:

1. **Selection policy** — faults are charged to the TIER that took them
   (`park_faults:<tier>`, written from the same min-reduced verdict as the flat
   counters, so still rank-uniform); `park_health` turns the tally into a #407
   health verdict and a tier that has eaten `PARK_FAULT_BLOCK` parks is refused
   by name. Smoked: 3 faults → `BLOCKED` → park declined, no silent fallback.
2. **Ledger** — one `kv-spill ledger:` line per shortfall episode, reporting
   traffic and saying it is traffic (occupancy has a different source).
3. **Dashboard** — `park_capacity_by_tier` gives park occupancy a measured
   DENOMINATOR for the first time; before, the gauge drew occupancy against
   nothing.

---

## 3. THE EVIDENCE

| axis | result |
|---|---|
| **#631 flip family** | **1076 passed / 0 failed**, before AND after — inherited baseline unmoved |
| `test_kv_spill_park_tier_659.py` (new) | 18 passed; every load-bearing assertion has a sibling arm giving the opposite verdict from the same function |
| adjacent (`test_kv_spill_destination_unit`, `test_kv_spill_tier_selection_659`, `test_spill_tiers`, `test_tier_occupancy`, `test_hicache_file_bounds_558`) | 112 passed |
| ruff / codespell on new+edited files | clean |
| **park tier REGISTERED on metal** | `fs:CT999:/spinning/evidence-631/s42/park (filesystem) capacity=21.47 GB [measured] bandwidth=2.41 / 7.00 GB/s [measured] latency=1.8 us volatility=persistent stages_through=host:CT999` — both ranks |
| **measured capacity in the live ledger** | `sglang:spill_tier_total_bytes{spill_tier="park:file"} = 2.147483648e10` on 30040 — a denominator that did not exist before this shift |
| **corridor, probe boot** | 3017 samples at 100 ms, per-card MINIMUM free **1309 / 32086 / 1309 MiB**, **0 breaches** of the 1024 floor |
| park/unpark round trip on metal | **NOT PROVEN** — no spill was reachable (§1d) |
| byte-identical restore from the park tier | **NOT PROVEN**, and the planned method is invalid (§1c) |
| restored session back on CUDA graphs (item 13) | **NOT REACHED** — depends on a park happening |

### The confirmation window (restored ship config)

Ship config restored from my own commit and healthy at **01:01:49Z**;
`s34_acceptance_run.sh 30` ran on it. Numbers in
`/spinning/evidence-631/s42/window/`.

**Inertness of the new code on the ship config, proven with a can-find
control** — the ship config carries no `--kv-session-offload-destinations`
(`kv_session_offload_destinations=None` in its own `server_args` line), so
every new path must be absent:

| grep | ship window log | probe log (control, same grep) |
|---|---|---|
| `park tier REGISTERED` | **0** | 2 |
| `destinations ARMED` | **0** | 2 |
| `kv-spill selection` | **0** | — |
| `kv-spill ladder` / `kv-spill ledger` / `park_faults` | **0** | — |

The control matters: a grep that finds nothing proves nothing until you show it
finds the thing where the thing exists.

Inertness holds on a SECOND, independent axis: the ship's `/metrics` carries
**zero** `spill_tier` series and **zero** `park:` keys, while the probe's
carried `park:file` with both a used and a (new) total. Two different
observation channels, same verdict — which is what separates "inert" from
"the one place I looked was quiet".

**The patch level the window actually ran, stated exactly** (Patchstand vor
Last). The restored instance reports `SGLANG_BOOT_COMMIT=08585169fe` and was
booted from that commit **plus the then-uncommitted registration-log line**,
which is now part of `4bc9e0093a`. So the running instance does NOT equal HEAD
(`00a5bb23a4`): the ratio fix of §1b landed after the boot and is not loaded in
it. This costs the window nothing — both are inert on the ship config, which
the table above proves rather than assumes — but the next shift should know
that the serving process is two commits behind the tree and re-boot before
attributing anything to HEAD.

---

## 4. WHAT IS NOT DONE, STATED SO NOBODY READS IT AS DONE

* **#659 IS NOT CLOSED.** Its (c) tier and (b)/(d) ledger halves landed and are
  live; its metal proof did not. Marking it closed on the strength of a
  registration line would be exactly the "green claim for a mechanism that
  never ran" that §1c nearly produced.
* **No park/unpark ever happened on metal.** See §1d for the three load shapes
  that failed to trigger one and the hypothesis for the next attempt.
* **The phase-flip x kvso crossing is untested and, as written, unbootable**
  (§1a). Nothing in this shift says anything about that pair.
* **The selection policy's multi-tier behaviour is desk-proven only.** Every
  configuration on this rig has exactly ONE park tier, so the promotion path
  (§1b) has hermetic coverage and no metal.
* **`quantize_bytes` is still an absolute grid** for capacity. That is
  defensible — a df reading is not a contended benchmark — but it is the same
  shape §1b falsified for bandwidth, and it has not been checked across ranks.
* **The park tier's `reserved` is frozen at registration, so its headroom does
  not shrink as it fills.** `_build_park_descriptors` passes `parked_bytes=0`
  once and the descriptors are cached, which means the capacity refusal is
  reachable in the hermetic tests (which construct a full tier directly) but
  effectively unreachable at runtime until the tier is 20 GiB over budget in
  reality. This is a KNOWN TENSION, not an oversight: a live `reserved` is
  exactly the per-park-varying input that §1b/law 14 says must not enter a
  group decision, so wiring `park_bytes_by_tier` straight into it would trade
  a stale denominator for a divergence risk. The honest fix is the same one
  law 15 names — reduce the occupancy through the collective the transfer
  already enters, then feed the reduced value — and it is the first thing the
  next shift on this rung should build, because without it the tier's byte
  budget is advisory.

---

## 5. PROPOSED REGISTER ENTRIES

* **New law (15): a per-rank benchmark is not a rank-uniform quantity, and
  rounding does not make it one.** Two ranks measured one directory 2.9x apart
  at the same instant. If a group decision must consume a measured value,
  consume a RATIO of two values measured on the same rank, or reduce it through
  a collective every rank enters. Sibling of laws 8 and 14. (§1b)
* **New contradiction (C25, open): kv-session-offload refuses `pp_size > 1`,
  so it cannot run on the Route-A ship recipe.** Close/reopen trigger: a boot
  in which S1 admits PP, or a decision to test kvso only on the decode layout.
  (§1a)
* **Amend C-family note on byte-identity method:** on this rig, two model
  generations are not comparable byte-for-byte across different batch
  compositions (GDN nondeterminism), so a byte-identity claim must be made
  about the transported BYTES, not about generated text. (§1c)

---

## 6. PROCESS NOTES

* The fork remote is **`origin`**; `upstream` is sgl-project. PAT in
  `/root/GITHUB_PAT`.
* `--rank-gpu-memory-mib` takes a **scalar** under even TP; a list requires
  `--rank-tp-ratio`. Cost one boot.
* The kvso host budget guard is exact and worth reading: it refused
  `--kv-session-offload-host-ram-gib 1` with "the allocated pool holds 32769
  tokens < 32770 tokens per region", i.e. short by ONE token. 1.5 GiB gives
  exactly one region, which is the lever that makes an overflow test possible.
* `KVSO_ALLOW_SPEC=1` is required for kvso x NEXTN; it is a bring-up gate, not
  a refusal.
* A `pgrep -f "port 30040"` matches transient shell pipelines of your own
  session. Resolve the pid to a cmdline before killing anything.
* Reusable artefacts, all under `/spinning/evidence-631/s42/`:
  `restore_ship_30030.sh` (reconstructed from the LIVE process before stopping
  it — do this before every window), `probe_boot_v2.sh`, `proof_driver.py`
  (records the ledger per phase, which is what caught the vacuous green).
