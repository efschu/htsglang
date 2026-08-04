# TICKET 470 RESULT — Boot A measured; Boot B blocked by a chain of first-boot defects

Window 2026-08-04, `/spinning/gpu-battery-results/2026-08-04_dsv4f_window/`.
Desk prediction: `NOTE_470_residency_cut_price.md`.

**Power state (all measurements):** 3080s 200 W (default 320), 5090 400 W
(default 575). Not comparable with any full-power anchor from an earlier day.

---

## 1. Boot A — the residency cut is FREE within measurement

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
arms' own floors, which is the rule the boot script itself states). The cut is
**not resolvable above the noise**.

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

The cost side of the gate is **~0 within measurement**. R1 is therefore NOT
refuted on cost — any positive return from the draft arm clears it. The gate now
hangs entirely on Boot B, which did not run.

Caveat worth naming: the two floors differ 20x (6.443 % vs 0.33 %). A floor that
loose on `a_base` deserves suspicion — it was the arm that also ran three
1000-token accept generations — so the +1.41 % is best read as "no effect large
enough to see", not as a precise zero.

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
(`rank_gpu_id`, `rank_tp_ratio`, `rank_auto_reserve_mib`, `rank_moe_ratio`,
`rank_kv_ratio`) — but that is an architectural change to the draft-copy
contract and was not made under window time pressure.

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
