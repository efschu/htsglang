# ANALYSE 478 RESULT — UD-Q3_K_XL is REFUSED on this rig

Window 2026-08-04, `/spinning/gpu-battery-results/2026-08-04_dsv4f_window/`.
Desk prediction and method: `ANALYSE_478_quant_arm.md`.

**Power state (all measurements):** 3080s 200 W (default 320), 5090 400 W
(default 575). The user lowered every target on 2026-08-03. Nothing here may be
compared against a full-power anchor from an earlier day.

## Verdict

**UD-Q3_K_XL cannot be loaded on this rig on the #77/#123 offload path.** Two
attempts, both killed by the **host OOM killer** (`exit code: -9`) during
`Load weight begin` — never reaching VRAM allocation, never reaching ready.

| | attempt 1 | attempt 2 |
|---|---|---|
| resident fraction | 0.40,0.35,0.35 | 0.45,0.40,0.40 |
| context / max-total-tokens | 8192 / 16384 | 4096 / 8192 |
| stream-trim SOFT/TARGET | 96 / 90 | 84 / 74 |
| RAM peak before kill | **103.9 GiB** | **103.9 GiB** |
| `file` at peak | 87.8 GiB | 86.5 GiB |
| `anon` at peak | 14.2 GiB | 15.6 GiB |
| outcome | OOM-killed | OOM-killed |

`MemTotal` is 104.0 GiB and `SwapTotal` is 0. Both attempts died at the wall.

## Why the second attempt did not help, and what that proves

The two attempts moved the resident fraction in the direction that *reduces*
the host pool (up, not down — a lower fraction spills MORE to host, which is
the opposite of what intuition suggests). The projected steady-state pool fell
from ~79 GiB to ~73 GiB. **The measured RAM peak did not move at all**
(103.9 GiB both times, `file` 87.8 → 86.5 GiB).

That is the finding: **the kill is a load-time phenomenon, not a steady-state
one.** The resident/spill split only governs where bytes live once the model is
up. During loading, the process simultaneously holds

* the page cache of a 119.4 GiB GGUF stream being read, and
* the pinned host pool being built (unreclaimable), and
* the runtime's own ~15 GiB of anonymous memory,

and the sum of those transients exceeds 104 GiB regardless of how the
steady-state split is configured. Tuning residency cannot reach the failure.

## Why the stream-trim could not save it — a mechanism correction

`ANALYSE_478_quant_arm.md` §3 predicted thrashing, and thrashing is exactly
what the trace shows (attempt 1, GiB): 86.2 → 76.8 → 95.5 → 82.7 → 101.5 →
103.5 → 103.9 → killed. The sawtooth is the trim firing and losing.

But the predicted *reason* was wrong, and the real one is worse. The desk note
said the trim would fail because its target (78 GiB) sat below the
unreclaimable floor (~86 GiB). Raising the marks was therefore expected to fix
it. It did not — attempt 1 ran with SOFT 96 / TARGET 90, above the floor, and
still died.

The actual mechanism: **CUDA pinned host memory is accounted in the cgroup's
`file` bucket, not `anon`.** Measured on the 1a control arm, whose offload
ledger reports a pinned pool of 20.78 + 14.44 + 14.44 = **49.66 GiB** while
`rammon` showed `anon` steady at 14.6 GiB. So:

* `maybe_trim()` (`gguf_shards.py:529-540`) computes `ask = current - target`
  from `memory.current`, which **includes the pinned pool**;
* `memory.reclaim` can only take clean page cache, never pinned pages;
* so `current` can never fall to `target`, the trim never stops asking, and it
  evicts the loader's own read-ahead as fast as the loader creates it.

The module's own safety argument (`gguf_shards.py:479-486`) says the pinned
pool is "structurally out of reach" of reclaim — which is true — but it assumes
that pool is *anonymous*, and therefore that `memory.current` minus the
reclaimable part is a meaningful target. On this path it is not. **A trim
target below `anon + pinned` is unsatisfiable no matter how it is set, and the
code cannot currently distinguish the pinned bytes from page cache because the
kernel files them in the same bucket.**

This is a real defect in the trim's model of its own budget, independent of
#478, and it is the reason the mechanism protects nothing for a
large-pinned-pool boot. Worth a follow-up ticket: the trim needs the pinned
figure from the offload ledger (which it has, `pinned_bytes`) to compute
`target = pinned + anon + headroom` rather than treating `current` as if it
were mostly reclaimable.

### FIXED AFTER THE WINDOW (#537), still unproven on a card

`ProgressCoupledTrim` now computes an unreclaimable floor and raises its target
to it whenever the floor sits above the configured target:

* `anon` from `memory.stat` — the half of the floor the kernel does report;
* the pinned pool summed over every LIVE rank, published through
  `layers/moe/pinned_host_ledger.py`. Per-process would not have been enough:
  the trim compares against the CGROUP's `memory.current`, which spans all
  three rank processes, so rank 1 reading only its own 14.44 GiB would have
  corrected the floor by less than a third of the measured 49.66 GiB;
* plus `SGLANG_GGUF_STREAM_TRIM_HEADROOM_GIB`, the only policy term, shipped at
  **0.0** because calibrating it needs a load-time page-cache measurement this
  window could not take. Pinned as inert.

The comment at `gguf_shards.py:479-486` is corrected in place rather than
deleted: the safety argument it makes is right, the anonymous-accounting
assumption under it is the documented error, and both are now stated.

Falsifiers (hermetic, `CUDA_VISIBLE_DEVICES=99`):
`test/registered/unit/model_executor/test_forward_peak_and_stream_trim.py`
(`TestStreamTrimBudgetModel`, 7 arms) drives the trim against a cgroup model
whose `file` bucket holds page cache AND the pinned pool. Fixed, it drains the
reclaimable part once and goes quiet (`trims == 1` over ten calls); with the
pinned pool made invisible again — the pre-#537 world model, same code path —
it asks on every call forever and is still asking for 10 GiB that no longer
exist. `test/registered/unit/layers/moe/test_pinned_host_ledger.py` (8 arms)
pins the cross-rank sum and the dead-rank skip. Executed can-fail: mutating the
floor override out turns the suite red.

**What this does NOT establish:** that the Q3_K_XL boot now survives. The
mechanism correction removes a defect that made the trim harmful; §"What would
actually be required" above is unchanged, and the arm is still expected to be
capacity-refused. The GPU-window falsifier is the same command that produced
the sawtooth: re-run the UD-Q3_K_XL attempt and read the `memory.current`
trace. Fixed, it must NOT sawtooth — the trim goes quiet at the floor and the
OOM (if it still comes) arrives from the load transient, not from the trim
fighting the loader.

## What would actually be required

Not tuning. The checkpoint needs either

1. a cold tier that is **not** resident host RAM — the #389 NVMe tier, which is
   exactly the case ANALYSE_389 reserves for ">150 GB" but which this 133.5 GiB
   in-memory tier reaches for a different reason (RAM ceiling, not model size);
   the measured NVMe bandwidth of 1.8 GB/s makes this a serving-viability
   question, not a loading one; or
2. more host RAM; or
3. a loader path that streams experts into the pinned pool **without** leaving
   the source pages in cache — the transient, not the steady state, is what
   kills it.

## Consequence for #478

**The active driver stays on UD-IQ3_XXS.** The question "should the driver move
to UD-Q3_K_XL" is answered NO on capacity grounds alone, before any quality or
throughput comparison was possible. No quality delta between the tiers was
measured and none is claimed.

The desk analysis's headline number stands and was the right thing to compute
first: the tier is **133.5 GiB in memory, not 119.4 GiB on disk**, because 47.8
GiB of MXFP4 has no kernel here and is repacked to Q5_0 at 22/17. Had the window
planned against the disk figure it would have spent two ~10-minute loads
discovering the same wall with no model of why.

## Cross-check that the instrument was right

The 1a control arm's offload ledger reports expert totals of 40.15 + 25.01 +
25.01 = **90.17 GiB**. The desk tensor-table scan predicted **90.172 GiB**. The
footprint parser is validated against the runtime to three decimals, which is
why its 133.5 GiB figure for Q3_K_XL is trusted here rather than re-derived.

Where the desk model *was* wrong: it predicted 54.51 GiB of resident experts for
IQ3_XXS against a measured **40.51 GiB**, because it assumed a non-weight VRAM
term of 4.4 GiB against a measured **21.4 GiB** (`max_total_num_tokens=102912`
is ~12.5x over-provisioned for context 8192 at one request). That error was
flagged in the desk note as the one unknown that would decide the arm, and it
decided it.
