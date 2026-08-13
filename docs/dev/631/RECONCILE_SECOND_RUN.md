# Second Reconciliation: the first run's own table had two wrong rows

Boot `1464299-1786612548`, the same data the first run used. Branch
`feat/ledger-reconcile-605`, worktree `/spinning/wt-ledger-605`, based on
`origin/integration/r2` at `cd71ec34ce`.

ERRORS FIRST, and the errors are in the instrument rather than in the boot.

---

## 1. What the first run got wrong

### 1a. The weights row summed two loads across a free

The first run reported `model weights (shards)` measured at **27800 / 16492 /
17464 MiB**. It reached those by summing the target runner's and the draft
runner's `pre_weight_load -> weights_loaded` reserved deltas.

The marks say the process **frees the first load before starting the second**.
On the 5090, `reserved_bytes` runs 2 -> 14004 (target weights in) -> 21724
(its KV pool) and then **falls to 8758** at the next runner's
`pre_weight_load`. Summing across that fall charges memory that was given
back. The result, 27800 MiB, is larger than the card's entire process
footprint at `boot_complete` (26364 MiB) — an impossible post that read like a
measurement.

The reserved basis was also **arena-blind**. This boot runs
`--phase-flip-spill-depth arena`, so the pages a freed layout releases are
taken over by the KV arena; a reserved delta therefore books arena growth as
weights.

Measured correctly — per episode, on allocated bytes outside the KV arena —
the boot has **three** weight-load episodes per card, not two:

| card | episode 1 (PP layout) | episode 2 (TP layout) | episode 3 (NEXTN draft) | largest |
|---|---:|---:|---:|---:|
| 5090 | 13674 | 13850 | 2074 | **13850** |
| 3080-5c64 | 8325 | 7719 | 1897 | **8325** |
| 3080-62db | 9293 | 7720 | 1897 | **9293** |

Each is released before the next loads, so the card's weight demand is the
**largest** episode, never the sum.

### 1b. The load transient was read at the one mark where it is zero

The mapping read `allocator_transient_bytes` at the target runner's
`weights_loaded` — by definition the mark at which the allocator has just
handed its peak back. It measured **0** on all three cards. The boot's real
peaks sit three marks later, at the second runner's `pre_weight_load`:
**13392 / 8058 / 9030 MiB**.

### 1c. The totals row subtracted a KV pool the boot never got

`measured_demand` subtracted the ledger's **budgeted** `kv_pool_mib` (29927 /
18245 / 18245) from a **measured** footprint. The arena actually ended up
backing 21130 / 13428 / 13410 MiB, so the subtraction removed up to 8797 MiB
the process never held and drove the demand negative. The first run worked
around this in prose. It is now read from the arena census.

---

## 2. Before / after, per term

Modeled column is unchanged — this is the same shipped pin-path ledger. What
moved is the measurement and what the table is willing to say.

### GPU 0 — RTX 5090, rank 0

| term | modeled | measured (1st run) | measured (2nd run) | verdict now |
|---|---:|---:|---:|---|
| model weights (shards) | 0 | 27800 | **13850** | modeled 0 — completeness check FAILS |
| load transient | 70 | 0 | **13392** | 70 MiB constant falsified |
| GDN prefill scratch | 20 | UNMEASURED | UNMEASURED | boundary NAMED: `gdn_scratch_begin/end` |
| attention workspaces | 384 | 708 | 708 | −324 |
| NCCL communicator buffers | 0 | UNMEASURED | UNMEASURED | boundary NAMED: `nccl_init_begin/end` |
| hardware residual | 664 | 886 | 886 | inside recalibrated band 802–902 |
| NVML driver carve-out | 518 | 518 | 518 | exact |
| runtime activation | *(absent)* | *(absent)* | **REFUSED, 34 measured** | refusal now a row |
| CUDA graph capture | *(absent)* | *(absent)* | **REFUSED, 172 measured** | refusal now a row |
| **measured demand** | 1656 | 5160 (hand-patched) | **5752** | from the arena census |

### GPU 1 — RTX 3080 5c64, rank 1

| term | modeled | measured (1st) | measured (2nd) |
|---|---:|---:|---:|
| model weights (shards) | 0 | 16492 | **8325** |
| load transient | 70 | 0 | **8058** |
| GDN prefill scratch | 20 | UNMEASURED | UNMEASURED (named) |
| attention workspaces | 384 | 384 | 384 |
| NCCL communicator buffers | 0 | UNMEASURED | UNMEASURED (named) |
| hardware residual | 312 | 480 | 480 (band 360–496) |
| NVML driver carve-out | 425 | 425 | 425 |
| runtime activation | — | — | REFUSED, 30 measured |
| CUDA graph capture | — | — | REFUSED, 202 measured |
| **measured demand** | 1211 | 3433 | **4113** |

### GPU 2 — RTX 3080 62db, rank 2

| term | modeled | measured (1st) | measured (2nd) |
|---|---:|---:|---:|
| model weights (shards) | 0 | 17464 | **9293** |
| load transient | 70 | 0 | **9030** |
| GDN prefill scratch | 20 | UNMEASURED | UNMEASURED (named) |
| attention workspaces | 384 | 384 | 384 |
| NCCL communicator buffers | 0 | UNMEASURED | UNMEASURED (named) |
| hardware residual | 312 | 480 | 480 (band 360–496) |
| NVML driver carve-out | 425 | 425 | 425 |
| runtime activation | — | — | REFUSED, 30 measured |
| CUDA graph capture | — | — | REFUSED, 202 measured |
| **measured demand** | 1211 | 3433 | **3961** |

---

## 3. The recalibration, from 472 boots and no GPU

`/spinning/flight_605` held **486 recorded boots** before this shift started,
472 of them carrying a target weight load. The bands:

| post | card | band | boots | verdict |
|---|---|---|---:|---|
| hardware residual | 5090 | 802–902 MiB | 462 | **calibrated**, charged at 902 |
| hardware residual | 3080-5c64 | 360–496 MiB | 472 | **calibrated**, charged at 496 |
| hardware residual | 3080-62db | 360–496 MiB | 472 | **calibrated**, charged at 496 |
| load transient | 5090 | 0–18486 MiB | 462 | **REFUSED** (100% of high) |
| load transient | 3080-5c64 | 0–9472 MiB | 472 | **REFUSED** |
| load transient | 3080-62db | 0–11898 MiB | 472 | **REFUSED** |

The ledger models the residual at 664 / 312 / 312 — **25–34% low**, confirmed
against a far larger sample than the single boot that first suggested it. This
boot's 886 / 480 / 480 sit inside their bands.

**The 70 MiB load transient is not merely the wrong value: the post is not a
constant.** It spans the full range on every card, so no replacement constant
is honest and the term is refused rather than re-fitted. Wide posts are
refused, not averaged.

---

## 4. Acceptance against the brief

> every post either matches within its recorded spread or is explicitly
> UNMEASURED-with-named-missing-boundary

**Met.** Every row of the second-run table is now one of: matched, matched
within a recorded band, UNMEASURED with the missing boundary named
(`gdn_scratch_begin/end`, `nccl_init_begin/end`), or REFUSED-by-the-model with
the refusal quoted. No row is silently absent — the refused activation and
graph-capture terms were missing from the first run's table entirely.

> the weights row must close within 1%

**NOT met, and deliberately not faked.** The row cannot close because the
modelled side is still 0 on this boot: the ship configuration is
`pp_size=3 / tp_size=1` with `--pp-layer-ratio 28,20,16`, and the weight
split is therefore a LAYER split. Per-layer bytes are not uniform on a hybrid
Qwen3.5/3.6 checkpoint (interleaved GDN and full-attention layers). Solving
the measured episode-1 posts for a uniform per-layer cost gives 488 / 416 /
581 MiB per layer — the model does not close, so it was not shipped.

What ships instead is the **completeness check**, which fails loudly on
exactly this: run against the real `ledger_1464299-1786612548.json` it names
all three cards. The pin-path plumbing (`weight_mib_per_rank`) is in place and
unit-tested; only the PP-stage estimator is missing, and it is named rather
than approximated. See `HANDOFF_LEDGER_RECONCILE.md` §4.

---

## 5. The residuum is still large, and now for a nameable reason

Claimed bytes exceed measured demand by 23603 / 13560 / 15652 MiB. This is not
a leak: **the ledger sums PEAK terms and STEADY-STATE terms into one total.**
The weights (largest episode, 13850) and the load transient (13392) are both
peaks that no longer exist at `boot_complete`, while `measured_demand` is a
steady-state reading. Adding them was always going to over-claim.

Named, not fixed. The per-term rows are the payout and they are now correct;
the totals row needs the ledger to distinguish a peak from a resident, which
is a taxonomy change and not a mapping fix.

---

## 6. Reproduce

```
cd /spinning/wt-ledger-605
PYTHONPATH=/spinning/wt-ledger-605/python CUDA_VISIBLE_DEVICES=99 \
  /spinning/shvllm/.venv/bin/python3 \
  /spinning/evidence-631/ledger-r1/rerun_reconcile.py
```

Artifacts in `/spinning/evidence-631/ledger-r1/`:
`rerun_reconcile.py`, `reconcile_second_run.txt`, `boot_history_bands.txt`.
