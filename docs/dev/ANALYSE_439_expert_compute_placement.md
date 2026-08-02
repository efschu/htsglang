# ANALYSE 439 — does `--rank-moe-ratio` already move COMPUTE?

Catalog-first question for #394 slice 3: `docs/dev/FEATURE_CATALOG.md` §1 lists
`--rank-moe-ratio` as "experts BETWEEN ranks". Before building anything, decide
with file:line evidence whether that flag moves the COMPUTE assignment (router
dispatch → which rank runs expert `e`) or only weight placement, and whether it
composes with the #77/#123 offload path.

Tree: `integration/r3-probe-next2` @ `b3d8a6d041`.

## Verdict

**It depends on the MoE path, and the catalog sentence is only true for one of
them.**

| path | what `--rank-moe-ratio` splits | moves compute? |
|---|---|---|
| **#82 GGUF expert-dim shard** (`_gguf_expert_shard`) | whole experts on dim 0 | **YES** |
| every other MoE (intermediate-dim TP) | the expert INTERMEDIATE dim | no — every rank computes every routed expert on a narrower slice |

And on the path where it moves compute, **it composes with the offload path
already**: the whole #77/#123/#391c staging plan is built over the OWNED expert
range, so changing the range changes what each rank streams.

**Therefore slice 3 collapses to a planner item + wiring**, exactly as the task
framing anticipated. No bridge, no new mechanism, no new dispatch.

## Evidence

### 1. The flag reaches the expert range

`server_args.py` → scheduler → the process-global family plan:

* `python/sglang/srt/server_args.py:9811` —
  `("rank_moe_ratio", "--rank-moe-ratio", "SGLANG_UNEVEN_MOE_VECTOR")` in
  `UNEVEN_FAMILY_RATIO_SPECS`.
* `python/sglang/srt/managers/scheduler.py` (`uneven_family_plans` /
  `configure_scheduler_process`) — installs it as the `"moe"` family through
  `set_tp_partition_ratios(base_plan, families=...)`.
* `python/sglang/srt/distributed/utils.py:840-912` —
  `tp_partition_sizes/size/offset(..., family="moe")` read that plan and fall
  back to the base `--rank-tp-ratio` vector when the family has no vector of
  its own.

### 2. On the GGUF expert-dim shard, the range IS the compute assignment

`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`:

* `:440-444` — `_gguf_expert_shard` is true for a GGUF MoE under an active
  `"moe"` family plan.
* `:446-460` — the range comes straight from the family plan:
  `tp_partition_offset(self.num_experts, ..., self.moe_tp_family)` and
  `tp_partition_size(...)`, stored as `self._gguf_expert_range = (lo, lo + n)`.
* `:1245-1252` (weight load) — an expert outside `[lo, hi)` is **dropped**:
  "foreign experts are dropped here and served by their owner rank."
* `:2640-2662` — the remap table: owned global ids map to their local slot,
  **every foreign id maps to the trailing zero padding expert** (`n_local`).
* `:2013-2024` (`forward_local`) — the topk ids are remapped through that table
  before dispatch, so a rank's GEMM runs only over the experts it owns and
  contributes exactly 0 for the rest; `:1995-1997` all-reduces the disjoint
  partials.

So: the router is replicated and rank-uniform, and the OWNER is the rank that
executes the expert. Moving the range moves the compute.

The contrast is stated in the tree itself, at
`python/sglang/srt/layers/moe/resident_fraction.py:1-6`: the resident fraction
"sets the GPU-resident / host-pinned split **within one rank's own expert
shard**. It does not move experts between ranks -- that is
``--rank-moe-ratio``, a different axis."

### 3. On every other MoE path it does NOT move compute

`layer.py:477-503` — without the GGUF expert shard, the same family vector
partitions `intermediate_size` (`assert_activation_aligned_shards` +
`tp_partition_size`), i.e. every rank holds a slice of EVERY expert. Confirmed
independently at
`python/sglang/srt/layers/moe/expert_offload.py:3253-3263`
(`repack_door_shards_experts_on_dim0`): "Anything else is an intermediate-dim TP
MoE, which holds an essential slice of EVERY expert and can therefore delegate
none of them."

### 4. It composes with the offload path

`layer.py:1322-1332` (`_gguf_owned_expert_count`) — the staging plan's expert
count is `hi - lo + 1` (owned range plus the pinned pad expert) under the shard,
and the parameter's declared expert dim otherwise.
`layer.py:1482-1528` (`_new_gguf_stream_stager`) — that count feeds
`plan_load_time_staging(count, fraction=..., pinned_experts=..., cold_shard=...)`,
which is the #77/#123 resident/spill split. `layer.py:1334-1344`
(`_gguf_local_expert_index`) addresses the tiers by `expert_id - lo`.

So resident count, host pinned pool and cold set are all functions of the owned
range. Changing the range changes each rank's streamed mass — which is the whole
lever, and it needed no new code.

### 5. Why slice 2 could not produce the gain

`expert_offload.py:1018-1038` (`partition_cold_experts`) splits **this rank's
own** cold experts across the group by link weight, and `cold_tier_fetch.py`
lets a rank DMA a delegated row out of a peer's segment. The rank that COMPUTES
expert `e` is unchanged, and it still pulls `e` across its own link. Hence
`FEATURE_CATALOG.md` §3's honest scope note, and hence the measured null.

## What was therefore built

A solve, not a mechanism: `python/sglang/srt/layers/moe/expert_compute_placement.py`.

    resident_r = f_r * b_r                        # held fixed -> VRAM-neutral
    share_r    = resident_r + (1 - sum resident) * normalise(l_r / c_r)

`b` = base plan, `f` = `--rank-moe-resident-fraction`, `l` = the #394 link
provenance chain (`resolve_host_shard_ratio`, env > card-probe > NVML >
refusal), `c` = optional per-rank cold-traffic coefficients measured from a
prior boot. Surface: `--rank-moe-ratio link`, resolved ONCE in the launcher
(`entrypoints/engine.py`, right after `publish_rank_card_uuids`), never in a
worker.

Holding the resident mass fixed is what makes it installable without re-fitting
the VRAM ledger: only the streamed remainder moves.

## Residual the first-order model leaves

`b_r (1 - f_r)` predicts H2D shares of 37.2 / 26.8 / 36.0 % on the reference
recipe; `BENCH_394_v4flash_club3090.md` measured 42.1 / 28.9 / 29.0 % (per-rank
hit rates 0.772 / 0.843 / 0.841). The uncalibrated solve therefore predicts
1.358x on the transfer term and the coefficient-calibrated one 1.584x, against
BENCH_394's 1.536x ideal-placement reference. All three are predictions from
measured inputs; the arm that turns them into a measurement is specified in
`scripts/dev/394_s2_proof/ARM3_COMPUTE.md`.

## Hazard axes checked

* **#109 MMQ-OOB / uneven ranges × GGUF shard boundaries** — dim 0 carries no
  quant constraint (`layer.py:430-439`) and the shard unit is one expert
  (`layer.py:466`). The solved vector is swept against `partition_units` for
  >= 1 expert per rank, exact sum and gapless tiling.
* **Rank-uniform router/dispatch (collective family)** — a single resolution
  point in the launcher; pinned with the #431 `barlink_uniformity` recorder plus
  a can-fail arm that perturbs one rank's link table.
* **#80 combine correctness** — unchanged code path; the ranges stay a
  partition of `[0, E)`, so the all-reduce keeps summing disjoint partials.
* **#112 `moe.cuh` nrows binding** — per-rank expert counts come from the same
  `partition_units` the layer calls.
* **VRAM ledger** — resident mass held fixed by construction; corridor and
  reserve are arm 1's.
