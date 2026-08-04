# TICKET 470 RESULT — Boot A measured; Boot B blocked by a chain of first-boot defects

Window 2026-08-04, `/spinning/gpu-battery-results/2026-08-04_dsv4f_window/`.
Desk prediction: `NOTE_470_residency_cut_price.md`.

**Power state (all measurements):** 3080s 200 W (default 320), 5090 400 W
(default 575). Not comparable with any full-power anchor from an earlier day.

---

## 1. Boot A — the residency cut costs ~1.4 % of decode ms/round (see the revision below)

The ticket's R1 gate is whether the residency cut costs more than the draft arm
returns. Boot A prices the cut. Both sub-arms ran to completion with full probe
sets.

| | a_base | a_cut |
|---|---|---|
| `--rank-moe-resident-fraction` | 0.485,0.42,0.42 | **0.23,0.42,0.42** |
| ready | 351 s | 352 s |
| **decode ms/round** (bs=1, ctx 940) | **131.353** | **133.206** |
| own A-vs-A floor, ms/round | **6.443 %** | 0.33 % |
| determined | 5/8 | 5/8, identical answers |
| chatprobe | template applied | template applied |

**Delta = +1.41 %, against a governing floor of 6.443 %** (the larger of the two
arms' own floors, which is the rule the boot script itself states). By that rule
alone the cut is not resolvable above the noise — but `a_base`'s floor was later
shown to be an outlier, so see the revision in this section before quoting that.

### The cut landed exactly as designed

Offload ledger, per rank:

| rank | card | a_base resident | a_cut resident | pinned(host) a_base → a_cut |
|---|---|---|---|---|
| 0 | 5090 | 19.37 GiB | **9.16 GiB** | 20.78 → 31.00 GiB |
| 1 | 3080 | 10.57 GiB | 10.57 GiB | 14.44 → 14.44 |
| 2 | 3080 | 10.57 GiB | 10.57 GiB | 14.44 → 14.44 |

Rank 0 gave up **10.21 GiB** of resident experts — the DSpark head needs 10.12
GiB. Ranks 1 and 2 are untouched, so the cut is asymmetric exactly as predicted.

### The desk prediction, scored

`NOTE_470_residency_cut_price.md` committed a falsifiable prediction before the
window. Scored honestly:

* **Delta predicted 10.12 GiB, measured 10.21 GiB — held to 1 %.** The quantity
  the prediction was actually about was right.
* **Absolute residency was wrong**: predicted 24.29 → 14.17 GiB, measured
  19.37 → 9.16 GiB, i.e. ~5 GiB low on both sides. Same root cause as
  ANALYSE_478: the desk VRAM model assumed a 4.4 GiB non-weight term against a
  measured 21.4 GiB.
* **The cut fraction was wrong in the direction that mattered for planning.**
  The desk note said 42 % of rank 0's resident experts; the value derived from
  measurement was 52 % (0.485 → 0.23). The ticket's own default,
  `RESIDENT_FRACTION_CUT=0.383`, would have freed only ~4.1 GiB against a
  10.12 GiB head and OOM'd rank 0 partway through the draft build. That default
  is arithmetic from ANALYSE_463 §4.4 and TICKET_470 §7.6 flags it as
  unmeasured; **it is now measured and should be corrected to ~0.23.**
* **The predicted perf consequence did NOT appear.** The note argued rank 0
  would become the pacesetter because it holds the biggest shard, the lowest hit
  rate (0.7794), and loses 42 % of its resident set over a PCIe link with no
  P2P. Measured: +1.41 %, inside the floor. Either the router is peaked enough
  that the cold tail costs little at bs=1, or bs=1 decode is not where this cut
  bites. **Recorded as a prediction that failed**, not quietly dropped.

### What this means for R1

The cost side of the gate is **small — ~1.4 %, see the revision below**. R1 is
therefore NOT refuted on cost — any positive return from the draft arm clears it. The gate now
hangs entirely on Boot B, which did not run.

### REVISED after the #462 eager control (same window)

The two floors differ 20x (6.443 % vs 0.33 %), and the later arms settled the
question. `462_eager` ran the SAME configuration as `a_base` in this window and
measured 131.475 ms/round against a_base's 131.353 — **0.09 % apart across two
independent boots** — with floors of 0.33 %, 0.40 % and 0.72 % on three other
arms.

So 6.443 % is an outlier (a_base was also the arm that ran three 1000-token
accept generations), and the real measurement band here is well under half a
percent.

**Revised reading: the cut most likely costs ~1.3-1.4 % of decode ms/round —
small, but REAL rather than zero.** The "+1.41 % inside a 6.443 % floor, not
resolvable" statement above is what the gating rule produces mechanically from
a_base's own floor, and it is retained as the formally-defensible bound. The
better-supported estimate is a small real cost.

This does not change the R1 conclusion: ~1.4 % is small enough that any positive
return from a working draft arm clears the gate.

---

## 2. Boot B — blocked, after three real defects were found and fixed

TICKET_470 states the solo slice is **"DESK-WRITTEN AND NEVER EXECUTED ... no
DSpark arm has booted on this rig"**. This window is the first execution, and it
found a chain of four defects. Three are fixed and committed; the fourth stopped
the arm.

### Fixed 1 — the solo-shadow marlin exemption existed only in the docstring

`_refuse_unsupported_speculative_moe_backend` documents itself as per-rank and
says *"only the ranks that actually build draft weights are affected -- a solo
SHADOW builds on the `meta` device and never reaches a kernel"*. The predicate
underneath tested `cuda_available`, `is_marlin()` and SM support and nothing
else; `build_draft_tp_worker` calls it unconditionally on every rank. So the
configuration the guard's own error message recommends
(`--speculative-draft-placement solo --speculative-draft-gpu 0`) was unreachable
on this rig: the 5090 host built fine, both SM86 shadows raised during init.

Falsifier: `test/srt/test_dspark_solo_shadow_marlin_guard.py`, four arms, two of
them can-fail arms that keep the guard biting on a solo host with an incapable
card and on non-solo placement.

**Sub-lesson, recorded because it nearly shipped:** the first version of this
fix compared `gpu_id` against `--speculative-draft-gpu`. It passed its hermetic
test and failed identically on hardware, because `gpu_id` inside a worker
process is not the global CUDA ordinal that flag names. The workers' own
predicate is `tp_rank == speculative_draft_solo_rank()`
(`eagle_worker_v2.py:328`). **A hermetic test passing is not evidence that a
predicate selects the right ranks when the test supplies the same wrong axis the
code reads.**

### Fixed 2 — the draft inherited the target's GGUF load format

`--speculative-draft-load-format` is documented to inherit `--load-format` when
unset (`server_args.py:3268-3273`). The target is GGUF and the DSpark head is a
safetensors directory, so the draft loader was handed a directory while
expecting a single `.gguf` file: `ValueError: ... is not a file.` Fixed in the
boot recipe by naming the draft's own format (`auto`).

### Fixed 3 — (see Fixed 1 sub-lesson; the axis correction is its own commit)

### NOT FIXED — the draft build inherits the target's per-rank MoE vectors

```
ValueError: the resident-fraction vector has 3 entries (0.23,0.42,0.42)
but tensor parallelism is 1.
```

Under solo placement the draft module graph is built inside a **weight-TP=1**
override (`model_runner.py:2088-2112`: `get_parallel().override(tp_size=1, ...)`
for `is_draft_solo_host or is_draft_solo_shadow`). The resident-fraction
validator reads the *current parallel state's* tp_size — 1 inside that context —
and compares it against the vector length carried over from the target's
`ServerArgs` by `build_draft_tp_worker`'s `deepcopy` (`resident_fraction.py:147-160`).

The draft is unsharded and its experts are not placed by the target's offload
tier, so it should not consult the target's per-rank MoE vectors at all. The fix
is to neutralise them in the draft `ServerArgs` copy — alongside the other
per-rank vectors that have the same problem by construction
(`rank_tp_ratio`, `rank_auto_reserve_mib`, `rank_moe_ratio`, `rank_kv_ratio`).

That neutralisation WAS implemented and committed — and it is **necessary but
not sufficient**, which the second boot proved. `resident_fraction._from_flag()`
falls back to the RUNTIME CONTEXT (`get_server_args()`) when no ServerArgs is
handed to it, and the context still holds the TARGET's arguments during the
draft build: `draft_server_args` is passed to `TpModelWorker` but never
published. So the validator kept reading the target's 3-entry vector against a
weight-TP=1 parallel state and refused identically.

Closing it means publishing the draft arguments into the runtime context around
the draft build, which changes the draft-copy contract for DFLASH and EAGLE too.
Deliberately not done against a restore deadline. The copy-level fix is kept
because a draft copy carrying target-shaped per-rank vectors is wrong regardless
of which reader notices first.

**Explicitly rejected workaround:** passing a single scalar resident fraction.
The error message suggests it and it would boot, but 0.23 everywhere also cuts
ranks 1 and 2, which are supposed to be untouched — it would silently invalidate
the comparison against `a_cut`. A config that boots is not the same as a config
that measures the intended thing.

---

## 3. State of the gate

* Boot A: **done**, cut price ~0 within measurement.
* Boot B: **not run.** Per TICKET_470 §5, *"an unattributed multiplier is not a
  result"* — no accept length, no ms/verify, and no verdict on R1 is claimed.
* The ANALYSE_447 §2.4 compressor-idempotence question is **unanswered**; it
  requires Boot B. The `a_cut` greedy reference it would be compared against
  exists (`idem_reference_470_a_cut.json`).

Next step is the draft-copy fix above, then Boot B unchanged. Everything else in
the arm is now known to work: solo placement resolves, the draft GPU resolves to
the 5090 by UUID through the CUDA ordinal, and the marlin backend is reachable.

---

## 4. FIXED AFTER THE WINDOW (#535/B1) — the draft args are now published

`build_draft_tp_worker` publishes `draft_server_args` into the runtime context
for the duration of the draft build. The restore already existed in the
function's `finally`; only the publication was missing, which is why the
neutralised copy changed nothing.

REACH, stated rather than implied. `draft_server_args` is a `deepcopy` that
differs from the target's in EXACTLY the fields overridden in that function:
`skip_tokenizer_init`, the three attention-backend fields, `context_length`,
and — under solo placement only — the five per-rank vectors. Every reader that
resolves through `get_server_args()` inside the build now sees those values,
which is what the copy was made for. **DFLASH is covered** (same builder).
**EAGLE is not**: `eagle_worker_v2.py:350`,
`multi_layer_eagle_worker_v2.py:187` and `standalone_worker_v2.py:87` pass the
TARGET's `ServerArgs` object straight into `TpModelWorker` with no draft copy,
so there is nothing to publish there — the same defect class is OPEN for them
and giving them a copy is a separate change with its own boot evidence.

Falsifier: `test/registered/unit/spec/test_draft_args_context_publication.py`,
8 hermetic arms with no GPU. It pins the CONTRACT — what the context resolves
to inside the build, that it is the same object `TpModelWorker` was handed,
that the neutralised fields are the ones it resolves, that the target's args
come back on both the normal and the exception path — plus the load-bearing
arm: `resident_fraction_vector(tp_size=1)`, the exact call that refused Boot B,
executed from inside the build. Its can-fail sibling runs the same call against
the target's published args and asserts the `"3 entries"` ValueError, so a
green result is not an artefact of `tp_size=1`. Executed can-fail: removing the
publication turns 5 of the 8 red.

**Not a Boot B result.** Nothing here has run on a card. The next window runs
Boot B unchanged; the acceptance signal is a boot that reaches ready with
`--rank-moe-resident-fraction 0.23,0.42,0.42` intact (NOT collapsed to a
scalar, which would silently cut ranks 1 and 2 and invalidate the comparison
against `a_cut`), and then an accept length and ms/verify. Only that answers
R1.
