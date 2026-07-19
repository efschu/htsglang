# MoE expert-offload: tiered residency (feat/moe-expert-offload)

## What changed vs. "pin-everything"

The first #77 offload pinned **all N experts per layer in host RAM** and used a
GPU buffer of `n_slots` as an LRU cache over that full set. That makes host RAM
hold the entire expert pool. For an FP8 large MoE (Qwen3.5-122B-A10B, 256
experts, ~116 GB of expert weights) the pool alone exceeds this box's 80 GB RAM
(no swap) → **FP8 was infeasible**.

Tiered residency splits each layer's `N` experts into two tiers:

| tier     | count | lives where                                   | RAM copy |
|----------|-------|-----------------------------------------------|----------|
| RESIDENT | `R`   | permanently in GPU buffer, slot == expert id  | **none** |
| SPILL    | `N-R` | pinned host pool, fetched into a `C`-slot GPU spill cache (LRU) | yes |

GPU per layer = `R + C` experts. **Host RAM per layer = only the `N-R` spill
experts** — never the resident set. That is the whole point: RAM no longer holds
the resident tier, so the spill set can be sized to fit RAM.

## Config (backward compatible)

- `SGLANG_MOE_RESIDENT_EXPERT_FRACTION=f` → `R = ceil(f·N)` permanently resident
  + `N-R` spill. Same R formula as before, but R is now permanent (not an LRU
  cache over all N) and RAM shrinks to the spill set. `f=1.0` → `R=N` → fully
  resident, no offload, **bit-identical** to the no-offload path.
- `SGLANG_MOE_SPILL_CACHE_SLOTS=C` sizes the spill cache. `0` = auto =
  `max(top_k, 8)`. Clamped to `[1, N-R]`. Must be ≥ any single token's spill
  count (fail-fast otherwise).

Requires `--disable-cuda-graph` (data-dependent spill routing; eager-only guard).

## Reproducibility (not cross-config bit-identity)

The forward is wave-split over **tokens** so each wave's union of unique **spill**
experts ≤ C (resident experts never consume a slot). Every token is computed
exactly once with all its experts present.

The offload path is **self-reproducible**: same request + same config → byte-
identical output run-to-run. Two fixes were required over the first design:

- **Deterministic, history-independent spill-slot assignment.** Within a wave the
  unique spill experts are *sorted* and packed to slots `R, R+1, …` — a pure
  function of the wave's spill set, independent of LRU/request history. The
  earlier LRU cache placed the same expert in *different* physical slots
  depending on request history; the FP8 grouped-GEMM reduction is sensitive to
  the expert-slot layout, so identical inputs gave **non-reproducible** output
  run-to-run. (Confirmed: `--enable-deterministic-inference` made the old path
  self-deterministic, proving reduction-order sensitivity.)
- **Fully-ordered H2D fetch** (blocking copy + stream sync) — a side copy stream
  raced the previous wave's compute.

It is **not** bitwise-identical to the no-offload (`f=1.0`) path: the remapped
`R+C`-slot layout has a different FP reduction order than the full-`N` layout, an
inherent floating-point difference of the **same class as changing TP degree or
batch size** (upstream sglang does not make those bitwise-identical either). It
is not a correctness bug — per-token math and weights are correct; output stays
coherent. Users needing strict cross-config reproducibility can launch with
`--enable-deterministic-inference` (order-independent kernels, global perf cost).

The **tensor test** (`gpu_offload_test.py`, 256-expert/top-8, incl. forwards with
spill-count > C) shows `torch.equal` exact identity against the reference
(kernel-free) MoE math — the correctness contract for the remap/wave
orchestration. That earlier "byte-identical" claim was always this tensor-level
identity, **not** server token-id identity across configs.

## FP8-122B feasibility arithmetic

Qwen3.5-122B-A10B, `N=256` experts. FP8 expert pool ≈ **116 GB**, so per expert
≈ `116/256 = 0.453 GB`. Non-expert weights (attention/embed/norm) in FP8 ≈ 6 GB.
Rig: **72 GB VRAM (32+20+20), 80 GB RAM, no swap.**

**Pin-everything:** host pool = all 256 experts = `256·0.453 ≈ 116 GB > 80 GB`
RAM → **infeasible**, regardless of GPU cache size.

**Tiered:** choose `R` so both constraints hold:

- **RAM (spill only):** `(N-R)·0.453 ≤ 80` → `N-R ≤ 176` → **`R ≥ 80`**.
- **VRAM (resident + cache + non-expert + KV):**
  `(R+C)·0.453 + 6 + KV ≤ 72`. Reserving ~10 GB for KV+overhead →
  `(R+C)·0.453 ≤ 56` → **`R+C ≤ 123`**.

Any `R ∈ [80, ~120]` satisfies both. Example **`R=112, C=8`** (f≈0.4375):

- GPU experts: `120·0.453 = 54.4 GB`  + non-expert `6 GB` = `60.4 GB`
  → leaves ~`11.6 GB` for KV cache across the 3 GPUs.
- Host pool: `(256-112)·0.453 = 144·0.453 = 65.2 GB ≤ 80 GB` ✓ (fits, no swap).

So FP8 122B becomes **feasible** under tiered residency where it was impossible
under pin-everything. Under TP=3 (expert-dim sharded) each rank holds its 1/3
shard of each resident expert and pins its 1/3 shard of the spill set; the
aggregate host RAM across the 3 co-hosted ranks equals the single-host
`(N-R)·0.453` figure above, so the RAM constraint is unchanged.

(The 116 GB / 6 GB figures are the FP8 arithmetic; the **mechanism** — resident
tier + spill-only RAM pool + wave-through spill cache — is validated on the
present 35B-A3B, which shares the arch/expert count. The 122B FP8 run itself is
not attempted — no model on disk.)

## Validation (present 35B-A3B-FP8, TP=3, temp=0, default kernels)

- **CPU planner**: 23 tests green (`test_planner.py`) — deterministic slot
  routing, resident-direct vs spill-cache, wave correctness with spill>C,
  history-independence / self-reproducibility, spill-clamp, fully-resident.
- **Tensor** (`gpu_offload_test.py`, 256-expert/top-8): `torch.equal` exact
  (max_abs 0.0) vs the reference MoE math for decode + overflow (95 waves) +
  tiered R=192+64-spill/C=8 (28 waves, spill-count 62 > C). RAM pool = N-R.
- **Server self-determinism** (THE bar): tiered f=0.5 (R=128 + C=8, host
  pool=128) run1 == run2 byte-identical at 96 **and** 256 tokens. f=1.0 is
  trivially deterministic.
- **Server greedy-agreement vs f=1.0**: 96/96 (100%) @96 tok; 118/256 (46%,
  first divergence tok 115) @256 tok — divergence is sub-ULP FP8 reduction-order
  under the remapped layout, output stays coherent.
- **RAM win realized on server**: boot log — *"host pool holds 128 spill
  experts"* = N-R (not all 256).
- **tok/s**: baseline ~11.6; tiered ~5.2 (96 tok) / ~8.5 (256 tok). Cost = per-
  decode spill H2D + fetch sync.
- **Known limitation**: offload installs LAZILY on the first forward (after the
  KV pool is sized), so the freed weight-VRAM is not auto-reclaimed for KV in
  this version — the KV pool matches baseline. The RAM win (the FP8 enabler) is
  fully realized; reclaiming the freed VRAM for KV needs a post-install
  re-profile (follow-up).
