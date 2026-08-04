# DSV4F window 2026-08-04 — fixed-cost sheet and run order

**Desk phase output. No GPU number in this document.** Feasibility is computed
before the window so no arm burns a ~6 minute weight load to discover it does
not fit.

## Power state (mandatory tag on every artifact)

The user lowered all power targets today. **Every measurement in this window
carries this tag, every delta needs a fresh same-boot A-vs-A floor taken at this
power state, and no number here may be compared against a full-power baseline
from a previous day** — those anchors are dead for comparison and quotable only
as an order of magnitude, with a warning label.

| NVML idx | CUDA idx | card | power.limit | default | max |
|---|---|---|---|---|---|
| 0 | 1 | RTX 3080 `05:00.0` | 200 W | 320 W | 320 W |
| 1 | **0** | RTX 5090 `0A:00.0` | 400 W | 575 W | 600 W |
| 2 | 2 | RTX 3080 `0B:00.0` | 200 W | 320 W | 320 W |

**NVML index ≠ CUDA index on this rig** (PCI order vs FASTEST_FIRST). All
`--rank-*` vectors and `--speculative-draft-gpu` are CUDA-indexed. See
ANALYSE_478 §1 — this is the single most likely source of a silent defect in
the window and every arm writes both orderings to `$RUN/device_order.json`.

## Rig constants

| | |
|---|---|
| `MemTotal` | 104.0 GiB |
| `SwapTotal` | **0** — cgroup reclaim can only take page cache, never the pinned pool |
| `memory.max` | `max` (no cgroup limit; `MemTotal` is the wall) |
| `/dev/shm` | 96 GiB |
| VRAM | 31.84 + 20.00 + 20.00 = 71.84 GiB |
| interconnect | no P2P, no NVLink, all PHB — every expert miss is a PCIe H2D |
| free-VRAM corridor | ≥400 MiB on **all** cards (#493), measured at forward peak, not idle |

Reference load cost: the 97 GiB IQ3_XXS stream took **239–240 s** of pure weight
load and **~5.5–6 min** launch→ready in the #417 window. Q3_K_XL is 22 GiB
larger on disk and 36 GiB larger in memory, so budget **8–10 min** to ready.

## Fixed-cost per arm

All rows: TP=3, context 8192, rank0 = 5090 = cuda:0, MXFP4 restated as the Q5_0
the loader repacks it to (22/17 = 1.294, `gguf_mxfp4_repack.py:39-41`).

| arm | in-memory weights | VRAM resident | host cold pool | + runtime ≈14 GiB | verdict |
|---|---|---|---|---|---|
| **1a** #478 IQ3_XXS (control) | 97.67 GiB | 61.4 GiB | 36.28 GiB | ~50 GiB / 104 | **fits, comfortable** |
| **1b** #478 Q3_K_XL | 133.46 GiB | 61.4 GiB | 72.07 GiB | ~86 GiB / 104 | **marginal — see below** |
| **2** #470 DSpark (on IQ3_XXS) | 97.67 + 10.12 head | 51.3 GiB | 46.40 GiB | ~60 GiB / 104 | **fits**; VRAM cut is the cost |
| **3** #462 breakable (on IQ3_XXS) | 97.67 GiB | 61.4 GiB − graph pool | 36.28 + pool | fits | **VRAM-gated, see below** |
| **4** #390/#394 expert_stats | — | — | — | — | free, armed in every arm |

### Arm 1b is the one that can fail, and one unknown decides it

VRAM is saturated in both tiers, so resident experts barely move (54.5 → 54.1
GiB) and **the entire +35.8 GiB of the quant swap lands in host RAM.** That puts
the unreclaimable floor at ~86 GiB against 104 GiB — feasible with ~18 GiB of
slack.

But the model's resident-fraction vector for IQ3_XXS (0.611/0.592/0.592) is
higher than the vector the proven recipe actually used (0.485/0.42/0.42), so the
real KV+activation VRAM term is larger than assumed. Carrying that discount to
Q3_K_XL gives ~0.35/0.30/0.30, a cold pool near **86 GiB** and a floor near
**100 GiB of 104** — i.e. **it does not fit.**

**Therefore: arm 1a runs FIRST, not second.** It is known-bootable, it is
required anyway as the same-power-state comparison arm, and it yields the one
measurement (true per-rank KV+activation VRAM, true cold-pool size) that decides
whether 1b launches at 8192 context, at 4096, or not at all. Feed the measured
term back through `gguf_footprint.py` before booting 1b.

Two things must change for 1b if it runs:
* **Stream-trim marks move UP, not down.** The recipe's `SOFT_GIB=88` /
  `TARGET_GIB=78` set a reclaim target *below* 1b's ~86 GiB unreclaimable floor.
  `maybe_trim()` would then ask for a reclaim it can never satisfy on every
  advice batch for the whole load, reclaiming page cache as fast as the loader
  creates it — a thrashing load, not a fast one. Use SOFT ≈ 96 / TARGET ≈ 90.
* **Pre-flight refusal** on the projected cold pool, so a bad config fails in
  seconds instead of after 8–10 minutes.

Lever if it refuses: `--context-length 8192 → 4096`. Every GiB of VRAM freed
moves ~1 GiB out of the host cold pool, roughly 1:1.

### Arm 2's cost is priced exactly (NOTE_470)

The 10.12 GiB DSpark head can only sit on the 5090 (MXFP4 Marlin is SM90/SM120
only), which is cuda:0 = rank 0. It cuts **rank0 resident experts by 42 %**
(24.29 → 14.17 GiB, fraction 0.611 → 0.356) and pushes exactly that into the
host pool. Ranks 1 and 2 are untouched, so the cut is **asymmetric** — rank 0
becoming the pacesetter is the expected failure mode, and Boot A must report
per-rank ms/round, not an aggregate. Arms must be work-matched (#482/#523): a
changed miss rate changes how far a run gets in fixed wall time.

### Arm 3 is VRAM-gated and needs headroom bought before it boots

The base recipe carries `--disable-cuda-graph`; #462 is exactly the mechanism
that lifts it, so this arm must fund a CUDA graph pool that no other arm pays
for. The #417 window measured free VRAM at forward peak of **420 MiB (5090) and
84/86 MiB (3080s)** — i.e. the #493 corridor is already at its floor with no
graph pool at all. **The pool cannot be funded out of thin air; resident experts
must be sold to buy it.**

Exchange rate, so the trade is a decision and not a guess:

| rank | card | expert shard | VRAM freed per 0.01 of resident fraction |
|---|---|---|---|
| 0 | 5090 | 39.77 GiB | **0.40 GiB** |
| 1 | 3080 | 25.51 GiB | **0.26 GiB** |
| 2 | 3080 | 25.51 GiB | **0.26 GiB** |

The arm's first action is therefore to measure the pool the capture actually
wants, then lower `--rank-moe-resident-fraction` by the corresponding amount —
and to record that it did, because the graph-vs-eager comparison is otherwise
confounded by a residency change on the graph side.

## Run order and gates

Priority when time runs short: **1 > 2 > 3 > 4.** Fewer arms with clean floors
beats more arms in a hurry.

1. **1a — IQ3_XXS control.** Boots. Yields: A-vs-A floor at this power state,
   prefill/decode/ms-round baseline, determined-answer baseline, the true
   KV+activation term, the cold-pool bucket question (ANALYSE_478 §3 open item),
   and expert_stats.
   → *Gate for 1b: recompute the footprint with the measured term.*
2. **1b — Q3_K_XL**, only if the recomputed footprint fits. Same window, same
   power state, same probes.
3. **2 — #470 Boot A** (residency-cut price), then **Boot B** (DSpark). The
   ticket is explicit: if Boot A cannot run, Boot B does not run — *"an
   unattributed multiplier is not a result"*. Greedy only (solo refuses
   non-greedy by name). Acceptance from `meta_info.spec_accept_length`, never
   `spec_ema_accept_len`. Assert the log line `Preparing MXFP4 experts for
   Marlin backend`, and assert the resolved draft GPU is a 5090.
4. **3 — #462 F2.** Order is mandatory: F2 break-cost first with the probe ON
   and a clean eager control, then the replay-without-recapture gate, then
   ms/verify A/B with the probe OFF. Expect 43 crossings/round; assert it against
   the DEBUG count of `Break graph due to function: _moe_offload_fetch_step`.
   Verdict is `43 × (break + rendezvous + planning + publish)` vs the launch
   overhead saved. No kill threshold — effort/return decides.
5. **4 — expert_stats** is harvested from arms 1–3, not booted separately.

## Window protocol

* Take `/spinning/gpu-arb/holder` with a heartbeat only after the user's go.
  Current holder is the **SERVING** line (`agent-530-serving`, pgid 3850257,
  port 30030, healthy). Copy it to `$RUN/holder_serving_original.txt` first.
* Safety net before every access regardless of what the files say
  (`gpu-arb/README.md` rule 2): any card >500 MiB used = abort.
* Stop serving + translator tenant group-scoped on their own pgids only. Never
  `pkill -f`. `py-spy dump` before killing anything wedged.
* `cachetrim.sh`, if started, **must be stopped at server-ready** — leaving it
  running during serving inflated the A-vs-A floor from 2.55 % to 39.91 %.
* **End of window is mandatory, not best-effort:** restore INT8 serving via
  `/tmp/w530_boot.sh`, smoke it (`/health` + one MT probe), stop the heartbeat
  **before** releasing, restore the holder to the SERVING line verbatim. The
  user must find a serving system.
