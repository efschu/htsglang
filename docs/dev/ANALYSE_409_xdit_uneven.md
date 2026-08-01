# ANALYSE 409 — Uneven (pressure-proportional) parallelism for diffusion models, with xDiT as candidate lane engine

Desk survey, no card time. Reads the current
`github.com/xdit-project/xDiT` source at commit `175e0bfe` (2026-07-14,
"feat: flux2 pipefusion (#736)"), its docs, its license, and the papers behind
its mechanisms, against this fork on `integration/r3-probe-next2`.

Builds on and does not repeat: the Class-2 architecture in
`docs/dev/DESIGN_333_multimodal_classes.md`, the already-built uneven
sequence-parallel delta in `docs/dev/DESIGN_333_M3_diffusion_lane.md`, the
uneven-DCP token-split lineage in `docs/advanced_features/uneven_kv_token_ratio.md`,
the cost model in `docs/dev/DESIGN_348b_cost_model.md`, and the card-identity
contract in `docs/dev/AUDIT_331_card_identity.md`.

---

## 1. Verdict

**xDiT is parallel-but-equal in four of its five mechanisms — and already
uneven in the fifth. The one axis that is ratio-capable today, PipeFusion's
per-stage layer count (`--attn_layer_num_for_pp`), is also the only mechanism
whose communication volume this rig can afford. That coincidence, not the
sequence-parallel work, is the finding.**

Six load-bearing findings:

1. **`--attn_layer_num_for_pp` already exists upstream and is a genuine
   ratio axis.** It is a user-facing list of per-stage transformer-block
   counts (`xfuser/config/args.py:302-308`,
   `xfuser/model_executor/models/transformers/base_transformer.py:102-108`).
   In PipeFusion the stage owns the layers and the patches are the pipeline
   micro-batch — every stage processes every patch — so per-rank work is
   exactly proportional to that rank's block count. Uneven PipeFusion is
   therefore a configuration question upstream, not a code question. This
   **corrects** `DESIGN_333_multimodal_classes.md` §0.3, which records
   "xDiT upstream supports equal splits only" as medium-confidence and
   unverified. It is verified now, and it is wrong for PipeFusion.

2. **The task brief's expectation that PipeFusion's ratio axis is the patch
   assignment is not how PipeFusion is built.** Patches are micro-batches
   ("It partitions an input image into $M$ non-overlapping patches. The DiT
   network is partitioned into $N$ stages … $M$ and $N$ can be unequal",
   `docs/methods/pipefusion.md`). Uneven *patch heights* change pipeline
   granularity and fill/drain latency; they do not move work between ranks.
   The patch geometry is nevertheless already expressed as a size *list*
   (`latents.split(get_runtime_state().pp_patches_height, dim=2)`,
   `xfuser/model_executor/pipelines/base_pipeline.py:515`) and the pipeline
   P2P negotiates per-segment shapes at runtime
   (`_communicate_shapes`, `xfuser/core/distributed/group_coordinator.py:831`),
   so it is parametric — just not the axis that pays.

3. **Sequence parallelism is where the equality is structural, and it is
   structural twice.** Ulysses does an equal-split all-to-all with the split
   arguments explicitly nulled and the output reshaped back to the input
   shape (`_sdpa_all_to_all_single`, `xfuser/model_executor/layers/usp.py:87-93`);
   ragged shares need *both* a ragged sequence axis and a ragged head axis,
   and the head axis quantizes the achievable ratio at 24 heads. Ring is
   worse: `yunchang.ring.utils.RingComm.send_recv` allocates the receive
   buffer as `torch.empty_like(to_send)`, so a ragged neighbour is a shape
   mismatch on the first hop — and the fix lives in a *different repository*
   (`feifeibear/long-context-attention`). Ring's P-step block schedule is
   additionally *optimal exactly at equal shares*; making it ragged makes it
   worse before it makes it better.

4. **On this rig, equal-split three-way parallelism is a regression against
   simply using the 5090 alone.** Equal N-way splitting beats the fastest
   card alone only if the fast/slow capability ratio `r < N`. With N=3 and a
   measured `r` between 2.5 and 5.19 (§3), the equal split lands between
   +20% and −42% against solo-5090. Proportional shares turn the same
   hardware into a 1.4x–1.8x win. **The uneven share is not an optimisation
   on top of diffusion parallelism here; it is the precondition for
   diffusion parallelism being worth booting at all.**

5. **Neither "fork xDiT" nor "port xDiT" is the right route, because the
   fork already vendors an xDiT-derived diffusion runtime and already built
   the uneven sequence split.** `python/sglang/multimodal_gen/runtime/distributed/`
   carries `# Copyright 2024 xDiT team` headers, and #333-M3 already shipped
   `_apportion` / `SGLANG_SP_CAPACITY_WEIGHTS` in
   `sp_shard_utils.py::build_shard_plan()`. The honest question is narrower:
   **PipeFusion is the one xDiT mechanism our vendored lane does not have**
   (only the `PipelineGroupCoordinator` scaffolding was copied,
   `multimodal_gen/runtime/distributed/group_coordinator.py:907`), and it is
   the mechanism this rig needs. Route: **selective port of PipeFusion into
   our lane, after an upstream-xDiT measurement establishes the yield.**

6. **We are not first, and the PR-Check duty says so plainly.** STADI
   (arXiv 2509.04719, Sept 2025) already allocates variably sized image
   patches to GPUs by measured compute capability, and additionally cuts
   denoising steps on slower cards. Every production engine checked —
   xDiT, ComfyUI, diffusers, TensorRT, Ray Serve — is verified negative,
   and xDiT's issue tracker returns zero hits for every heterogeneity term.
   Our remaining delta is the *axis* (layer split, weights sharded, ~50x
   less traffic), *losslessness* (STADI's temporal trick changes the
   image), and the planner/ledger/identity integration nobody publishes
   (§5).

Ranked cut list (yield / effort, per the standing "no kill threshold — take
cheap wins" rule):

| # | Cut | Effort | Yield on this rig | Status |
|---|-----|--------|-------------------|--------|
| 1 | Measure upstream xDiT PipeFusion pp=3 on FLUX.1-dev, NVML-ordered ranks, `--attn_layer_num_for_pp` sweep | ~0 code, 1 card window | Establishes the whole yield model against real silicon; the default `[19,19,19]` block split is *already* a 2.4:1:1 FLOP split (§3.4) | proposed |
| 2 | FLOP-weighted planner solver for the block list (double vs. single block cost), fed by `compute_rates_for_cards` (#348b) | ~80 LOC | Turns cut 1 from a hand-tuned list into a planned one; reuses the existing cost model | proposed |
| 3 | Port PipeFusion into `multimodal_gen` with the layer list as a first-class ratio parameter | ~1200 LOC, real | Lifts the replicated-weights VRAM cap that blocks every model >18 GB on the 3080s, and cuts collective volume ~50x vs. USP | proposed |
| 4 | #333-M4 as already planned: `all_gather_v`, uneven weight-TP, varlen joint attention | already scoped | Completes the uneven-SP half we built in M3 | scoped, not started |
| — | Uneven Ring attention | ~500 LOC across two repos | Negative before positive (§2.3) | **excluded**, hard reason |
| — | Uneven CFG-parallel | n/a | Work unit is indivisible; and FLUX/Qwen-Image are guidance-distilled, no CFG at all | **excluded**, hard reason |

---

## 2. Where the equal-split assumption lives, per mechanism

### 2.0 Summary table

| Mechanism | Where the equality lives | Ragged verdict | Effort to make ratio-capable | Value here |
|---|---|---|---|---|
| **PipeFusion — layer/stage axis** | *no equality* — `attn_layer_num_for_pp` list, `base_transformer.py:102-108`; default fallback is `ceil(L/N)` at `:111-118` | **parametric — already ratio-capable** | **0** (config), ~80 LOC for a planner-driven list | **highest** |
| **PipeFusion — patch axis** | `_calc_patches_metadata`, `runtime_state.py:656-687`: one uniform `pipeline_patches_height` plus a remainder tail | parametric (sizes are already a list; P2P negotiates shapes) | ~80 LOC + relax 2 raises | low — not a rank-load axis |
| **Ulysses** | `usp.py:87-93` a2a with `input_split_sizes=None, output_split_sizes=None` and `reshape(x_shape)`; asserts `h % ws == 0` (`:103`) and `s % ws == 0` (`:145`); uniform head slice in `_preprocess_joint_tensors` (`:159-169`); `runtime_state.py:649` raises on `H % sp != 0` | **needs rework** (primitive is parametric, callers are not) | ~300-400 LOC; ratio quantized by head count (24) | medium, but see §3.3 comm |
| **Ring** | `yunchang.ring.utils.RingComm.send_recv` → `torch.empty_like(to_send)` (upstream repo); P-step schedule in `ring_flash_attn.py:83-143` | **structurally wired equal** | ~500+ LOC across 2 repos, and the schedule must be redesigned | **negative** |
| **CFG-parallel** | degree is a constant 2; `_process_cfg_split_batch`, `base_pipeline.py:534-560` — rank 0 takes the negative branch, rank 1 the positive | **not applicable** — the work unit is one whole forward pass | n/a | none |
| **Tensor parallel** | `feedforward.py:37-62`, `chunk(tp_degree, dim=0)[tp_rank]` / `chunk(tp_degree, dim=1)`; `all_reduce` at `:77` is size-invariant | **parametric** — swap `chunk` for `split(sizes)`, no collective change | ~20 LOC | negligible — FFN only, one model (StepVideo) |
| **Fully-shard (FSDP2)** | `fs_degree`, `base_model.py:731-770`; DTensor equal shards | memory axis, compute is replicated | n/a | not a ratio axis |

### 2.1 PipeFusion — the axis that already exists

`_split_transformer_blocks` (`base_transformer.py:76-118`) concatenates every
named block list of the transformer into one index space and hands each
pipeline stage a contiguous range:

- with `attn_layer_num_for_pp` set: `start = sum(list[:rank])`,
  `end = sum(list[:rank+1])`, guarded by
  `assert sum(attn_layer_num_for_pp) == sum(num_blocks_list)`;
- without it: `num_blocks_per_stage = ceil(total / pp_world_size)`, the
  classic equal split.

Blocks not owned by this stage are replaced with `nn.ModuleList([])` — the
weights are genuinely sharded, not merely skipped. The main loop
(`pipeline_flux.py:578-681`) is `for timestep: for patch_idx in
range(num_pipeline_patch): recv → backbone_forward(my blocks) → isend`, so a
rank's per-timestep work is `num_pipeline_patch × (its block cost)`. Uneven
block counts are therefore uneven rank load, exactly.

Two caveats that a planner must handle and that upstream does not:

- **Blocks are not equal-cost.** FLUX.1 is 19 double-stream blocks followed
  by 38 single-stream blocks (`transformer_flux.py:235-236`). A double block
  is ~36·d² parameters (two streams × [4d² attention + 8d² MLP] + 2 × 6d²
  modulation), a single block ~15·d² (7d² in, 5d² out, 3d² modulation) —
  a 2.4:1 cost ratio. Counting blocks uniformly is not counting FLOPs.
- **Stages must be contiguous ranges.** That is fine for a monotone ratio
  vector but forbids interleaving.

The pipeline transport is already shape-agnostic:
`PipelineGroupCoordinator.set_recv_buffer` takes a *list* of per-patch shapes
(`group_coordinator.py:745-765`), and `_check_shape_and_buffer` /
`_communicate_shapes` (`:779-860`) exchange shapes per `segment_idx` and
allocate accordingly. Nothing in the PipeFusion transport assumes equal
patches.

### 2.2 Ulysses — parametric primitive, non-parametric callers

The all-to-all itself is the one torch primitive that *does* support ragged
shares, and xDiT explicitly declines it:

```python
def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None,
                               group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)          # <- only valid when in-size == out-size
    return x
```

(`xfuser/model_executor/layers/usp.py:87-93`.) Passing real
`input_split_sizes` / `output_split_sizes` is the whole mechanical change;
the `reshape(x_shape)` and the surrounding permute/view chains
(`:96-150`) are what actually break.

The deeper point is that **Ulysses needs two ragged axes, not one.** Before
the a2a each rank holds `(b, h, s_i, d)`; after it, `(b, h_i, S, d)`. Linear
and modulation layers cost `∝ s_i`; the attention core costs `∝ h_i · S²`.
Making only the sequence ragged leaves the attention core equal-split, which
on FLUX at 1024px is roughly a third of the block FLOPs and rises
quadratically with resolution. So a ratio-capable Ulysses is a ragged
`(s_i, h_i)` pair with `Σ s_i = S`, `Σ h_i = H`. With H = 24 (FLUX and
Qwen-Image both, `transformer_flux.py:238`, `transformer_qwen.py:115`) and a
target ratio of 3.5:1:1, the ideal head split 15.5/4.25/4.25 quantizes to
15/5/4 or 16/4/4 — a 6-18% share error on the small cards, which caps how
well Ulysses can ever track a measured ratio.

Three further hard gates in the same lineage:

- `runtime_state.py:649` — `latents_height % num_sp_patches != 0` raises. At
  1024px, `latents_height = 128`, and `128 % 3 = 2`: **Ulysses-3 on a
  1024px FLUX image is not expressible at all today.** Uneven shares would
  remove this constraint as a side effect, which matters on any rig with a
  prime or awkward device count.
- `runtime_state.py:680-686` — the last pipeline patch must be a multiple of
  `patch_size × num_sp_patches`.
- `runtime_state.py:568-580` — `num_heads % ulysses_degree != 0` raises.

One incidental observation, recorded because it is evidence rather than a
bug report: `flatten_patches_height` (`runtime_state.py:691-695`) builds its
list grouped by SP rank and then slices it grouped by pipeline patch
(`:700-705`). The two groupings agree only when all patch heights are equal.
When the tail patch differs, the resulting partition is still a disjoint,
contiguous cover of the height — so it is not a correctness bug — but the
patch index no longer means the same span on every rank. Code that survives
only because the shares are uniform is the signature of a structural
assumption, not a parametric one.

### 2.3 Ring — the honest "do not"

`xdit_ring_flash_attn_forward` (`ring_flash_attn.py:83-143`) is the textbook
loop: issue `comm.send_recv(k)` / `send_recv(v)` and `commit()` at the top of
step *t*, compute the block, `comm.wait()` at the bottom, swap buffers.
`RingComm.send_recv` in `feifeibear/long-context-attention` allocates
`torch.empty_like(to_send)` when no receive tensor is supplied, so rank *i*
sizes its receive buffer from *its own* shard. Ragged shares mismatch on hop
one. That fix is upstream of xDiT, in a second repository — a two-repo
maintenance surface for a mechanism we do not otherwise want.

The schedule is the deeper objection. Per-step work is
`W[i][t] = s_i · s_{(i-t) mod P}`. Under equal shares this matrix is
constant — ring is *perfectly* balanced, which is precisely why the
xDiT docs can say "Since DiT does not use Causal Attention, there is no need
for load balancing operations on Ring-Attention" (`docs/methods/usp.md:11`).
Under proportional shares the row sums stay equal (total work is
proportional, which is what we want) but the individual steps do not, and
the ring's chained dependency — rank *i* cannot enter step *t+1* until rank
*i-1* has entered step *t* — converts that per-step skew into propagation
delay. Ring is the one mechanism where making the shares ragged first
degrades a property the equal split was getting for free. Excluded on that
basis, not on effort.

### 2.4 CFG-parallel and TP

CFG-parallel has a constant degree of 2 (README, "CFG Parallel … with a
constant parallelism of 2") and splits by *branch*: rank 0 evaluates the
negative prompt, rank 1 the positive (`base_pipeline.py:534-560`), rejoined
by `get_cfg_group().all_gather`. There is no fractional unit — you cannot
give a rank 60% of a forward pass. It is also inapplicable to the vehicles
that matter: `docs/performance/flux.md` states "Since Flux.1 does not
utilize Classifier-Free Guidance (CFG), it is not compatible with cfg
parallel", and the same holds for guidance-distilled Qwen-Image and the
Lightning/Turbo variants. Not an axis.

xDiT's TP is the easiest mechanical change and the least valuable one. It is
column-parallel then row-parallel over the FFN only
(`feedforward.py:31-77`), and the closing `get_tp_group().all_reduce` is
size-invariant, so ragged shards need no collective change at all — replace
`chunk(tp_degree, dim)[tp_rank]` with `split(sizes, dim)[tp_rank]` on both
dims and it works. But the supported-model table marks TP for exactly one
model (StepVideo), attention weights stay replicated, and attention is
recomputed on every rank. Ratio-capable in ~20 LOC, worth ~0.

### 2.5 Device identity

`get_device(local_rank)` returns `torch.device("cuda", local_rank)`
(`xfuser/envs.py:84-92`) — plain torch enumeration order, no NVML, no UUID.
For any experiment on mixed cards the rank→card assignment must be pinned
from outside via `CUDA_VISIBLE_DEVICES` with NVML-resolved indices, per
`AUDIT_331_card_identity.md`. This is the known torch≠NVML ordering trap and
it bites immediately here, because *which* card gets stage 0 is the entire
experiment.

---

## 3. The yield model

### 3.1 Assumptions, stated

- Three cards: one RTX 5090 at relative compute rate `r`, two RTX 3080-20GB
  at rate 1 each.
- `W` = total DiT work for one denoising step, measured in 3080-seconds.
  Solo-5090 baseline is `W/r`.
- DiT inference is compute-bound, not bandwidth-bound: every layer is a
  large GEMM over thousands of tokens, structurally an LLM *prefill*, never
  a decode. The relevant capability ratio is therefore the compute one.
- **`r` is not yet measured for this workload.** The scores we hold are for
  quantized LLM formats: #252 measured 5.19:1 on a 2048-token prefill chunk;
  #324's phi0 microbench gives ~3.7-4.0:1 on Marlin lanes; datasheet bf16
  dense with fp32 accumulate is ~209 vs. ~60 TFLOPS, i.e. ~3.5:1; the
  bandwidth ratio (decode-relevant, not relevant here) is 2.35:1, and
  today's `auto-performance` compromise ships 1.74:1:1. A **bf16 dense GEMM
  score at DiT shapes does not exist in the registry** and is a prerequisite
  for a planned split. Central estimate below: `r = 3.5`, with the band
  carried explicitly.

### 3.2 The share arithmetic

Both barrier-synchronised mechanisms (USP: one all-to-all pair per attention
op) and the pipeline (PipeFusion: steady-state throughput set by the slowest
stage) obey the same max-law over per-rank share ÷ capability:

```
T(s) = W · max_i (s_i / c_i)
T_equal        = W/3
T_proportional = W/(r + 2)
gain           = (r + 2)/3
```

| `r` | solo-5090 | equal 3-way | proportional 3-way | prop vs. equal | equal vs. solo | prop vs. solo |
|---|---|---|---|---|---|---|
| 2.5 | 0.400 W | 0.333 W | 0.222 W | **1.50x** | 1.20x | 1.80x |
| 3.5 | 0.286 W | 0.333 W | 0.182 W | **1.83x** | **0.86x** | 1.57x |
| 5.19 | 0.193 W | 0.333 W | 0.139 W | **2.41x** | **0.58x** | 1.39x |

The general statement: **equal N-way splitting beats the fastest card alone
only when `r < N`.** At our central estimate `r = 3.5` with N = 3, booting
all three cards with equal shares is a 14% *regression* against ignoring two
of them. That is the slowest-rank-sets-the-clock lesson in its sharpest form — the two
3080s pace every single one of FLUX's 57 blocks × 28 steps = 1596
synchronisation points, and the 5090 idles 71% of each one.

### 3.3 Communication, and why it decides the mechanism before the ratio does

The ratio argument above is identical for USP and PipeFusion. The
communication argument is not, and on this rig it is 50x larger than the
ratio effect. FLUX.1-dev, 1024px, 28 steps, 3 ranks, bf16;
sequence = 4096 image + 512 text = 4608 tokens, d = 3072:

| Mechanism | Per-rank volume | Derivation |
|---|---|---|
| **USP / Ulysses** | **~40 GB** | per attention op: qkv a2a on the local `(b,h,4608/3,d)` shard, ~19 MB moved, plus the output a2a ~6 MB → ~25 MB; × 57 blocks × 28 steps = 1596 ops |
| **PipeFusion** | **~0.8 GB** | one activation hop per patch per stage boundary: `4608 × 3072 × 2 B = 28 MB` per timestep; × 28 steps |
| DistriFusion | > USP | async all-gather of every layer's activations, weights replicated |

This rig has no GPUDirect P2P (chipset), no NVLink, all PHB, and GPU0 on a
×4 link (`docs/dev/` rig-interconnect notes). Cross-card traffic is
host-staged. At a generous 6 GB/s effective, USP's 40 GB is ~2 hours of
collective per image against ~5 seconds of compute; PipeFusion's 0.8 GB is
~0.13 s, overlapped with compute by the async P2P. This is not a tuning
gap — **sequence parallelism is not deployable on this interconnect at all**,
at any share vector, and xDiT says as much in its own words: PipeFusion
"demonstrates significant advantages in weakly interconnected network
hardware such as PCIe/Ethernet" (`docs/methods/pipefusion.md`), while its own
4×H100 measurements put USP ahead only because of NVLink.

Sanity anchor from upstream's own table (`docs/performance/flux.md`,
FLUX.1-dev 1024px, 28 steps, torch.compile): 1×H100 4.30 s → 4×H100 Ulysses
1.63 s, a 2.63x speedup at 4 GPUs, i.e. 66% efficiency *with* NVLink and
*with* equal cards. That is the ceiling SP has under ideal conditions.

### 3.4 The near-free coincidence

FLUX's block cost profile and this rig's ratio nearly match by accident.
Cost units (d² multiples): 19 double blocks × 36 = 684, 38 single blocks ×
15 = 570; total 1254. xDiT's *default* equal-block-count split at pp=3 is
`[19, 19, 19]`, which is:

- stage 0 = all 19 double blocks = **684 units**
- stage 1 = 19 single blocks = **285 units**
- stage 2 = 19 single blocks = **285 units**

— a **2.4 : 1 : 1 FLOP split, out of the box**, against a hardware ratio in
the 2.5-5.19 band. Put the 5090 on rank 0 and the default configuration is
already most of the way to proportional. Tuning to `r = 3.5` wants
`[20, 19, 18]` → 699/285/270 vs. the ideal 700.6/276.7/276.7, a ≤3% share
error at single-block granularity.

This makes cut 1 a placement-and-flag experiment with zero xDiT code change,
which is the cheapest possible way to falsify the whole yield model.

### 3.5 Why PipeFusion's uneven axis is doubly right here

The layer split sets *both* compute share *and* VRAM share, in the same
direction. Under USP the weights are replicated on every rank, so the model
size is capped by the smallest card — 20 GB — which excludes FLUX.1-dev in
bf16 (23.8 GB backbone) and every 14B-class video model. Under PipeFusion
the weights are sharded by the same list that sets the compute share, so
giving the 5090 more layers gives it more work *and* more weights *and* more
stale-patch KV cache, all proportionally. The pressure-proportional planner
gets one knob that moves three pressures coherently.

---

## 4. Fork vs. port — and the third answer

The task framed two routes. Reading our own tree makes a third one correct.

**What we already have.** `python/sglang/multimodal_gen/` is a complete
vendored diffusion runtime — scheduler, ZMQ transport, workers, FastAPI app,
distributed parallel state — and its `runtime/distributed/parallel_state.py`
and `group_coordinator.py` carry `# Copyright 2024 xDiT team` headers. #333-M3
already built the uneven sequence split: `SpShard` with `offsets`/`lens`,
`_apportion(total, weights)` doing the same 64-unit largest-remainder
apportionment as `partition_units` on the AR side, and
`SGLANG_SP_CAPACITY_WEIGHTS` fed from the shared cost model
(`sp_shard_utils.py`). **We are not choosing whether to have an xDiT-derived
diffusion lane. We have one.**

**What we do not have.** PipeFusion. The vendored copy took the
`PipelineGroupCoordinator` class (`num_pipefusion_patches` at
`multimodal_gen/runtime/distributed/group_coordinator.py:907`) but no patch
pipeline, no `_split_transformer_blocks`, no async displaced-patch loop. Our
lane's parallelism is Ulysses + ring + CFG + DP + VAE-parallel — precisely
the set that §3.3 rules out on this interconnect.

| Criterion | (a) Fork xDiT as a tenant lane | (b) Port USP/PipeFusion beside our DCP machinery | (c) **Selective PipeFusion port into the lane we already run** |
|---|---|---|---|
| Maintenance surface | Whole second engine: two model paths (`pipelines/` legacy + `models/runner_models/`), two USP implementations (`usp.py` + `usp_legacy.py`), a hard diffusers-version coupling the README itself warns about ("you may need to try several recent diffusers versions") | Duplicates code the fork already carries — DESIGN_333's stated reason for "xDiT — do not port" | One mechanism, into a file tree we already patch |
| Planner reuse (#348b `compute_rates_for_cards`) | none — xDiT has no planner, no cost model, no capability notion | full | full, and shares the rate source with #333-M3's `_apportion` |
| Identity (`IdentityMap`, NVML UUID) | none — `torch.device("cuda", local_rank)`, `envs.py:84`; would need an external `CUDA_VISIBLE_DEVICES` shim per rank | full | full |
| #400 ledger / VRAM korridor | xDiT declares no VRAM budget; a tenant that cannot `estimate()` cannot be admitted pre-boot, which is exactly the failure #400 exists to prevent | full | full — and PipeFusion's layer list makes `estimate()` *analytically* exact per card, which the replicated-weights SP path cannot offer |
| #274 tenancy | possible as a process-isolated tenant (one CUDA context per card as a ledger line item), but every arbiter contract — capture lock, tick broker, `PromotionRejected` — would need a shim | native | native |
| What #333 assumed | "xDiT — do not port" (`DESIGN_333_multimodal_classes.md:1380-1385`) | same | consistent with it; PipeFusion was simply not in scope when that was written |

**Recommendation: (c), gated on (1).** Fork upstream xDiT only as a
*measurement harness* — a throwaway checkout, not a maintained fork — to
establish the PipeFusion yield curve on real mixed silicon before spending
~1200 LOC porting it. If cut 1 shows PipeFusion pp=3 with a proportional
layer list beating solo-5090 by the predicted 1.4-1.8x, port it. If it does
not, the honest outcome is that this rig runs diffusion solo on the 5090 and
#333-M4 remains a feature for rigs with better interconnect — which is the
rig-is-a-floor rule applied, not a verdict on the feature.

---

## 5. Prior art

**We are not first. One paper does exactly the thing, and a reviewer would
cite it immediately.**

### 5.1 The direct hit: STADI

**STADI: Fine-Grained Step-Patch Diffusion Parallelism for Heterogeneous
GPUs** — Han Liang, Jiahui Zhou, Zicheng Zhou, Xiaoxi Zhang, Xu Chen;
arXiv 2509.04719, v1 2025-09-05, v2 2025-09-15. Verified by reading the
abstract page directly.

Two mechanisms, both compute-capability-proportional:

- **spatial** — "elastic patch parallelism … allocates variably sized image
  patches to GPUs according to their computational capability";
- **temporal** — a step allocator that *reduces the number of denoising
  steps on slower GPUs*, with an LCM optimisation to align synchronisation
  points.

Claimed up to 45% end-to-end latency reduction against patch parallelism on
"load-imbalanced and heterogeneous multi-GPU clusters". No code release
found.

Three deltas remain, and they are not rhetorical:

1. **Axis.** STADI's spatial mechanism is *patch* parallelism —
   DistriFusion-lineage, weights replicated on every GPU, activations
   exchanged per layer. That is the axis §3.3 shows is unaffordable on a
   PCIe/PHB rig, and it does not lift the smallest-card VRAM cap. Our
   candidate axis is PipeFusion's *layer* split, which shards the weights
   and moves ~50x less data. Different axis, different rig class.
2. **Losslessness.** The temporal mechanism changes the output — a GPU that
   runs fewer denoising steps produces a different image. Under our
   Quality-Last rule that is a lossy feature and would be gated behind every
   byte-identical win, not shipped alongside them. Our cut 1 is explicitly
   byte-gateable (§8), theirs is not.
3. **Integration.** A capability-proportional split is only as good as the
   capability numbers. STADI measures capability; we already have a
   measured, per-family, per-card cost model (#324/#348b), a ledger that
   refuses an infeasible plan pre-boot (#400), and an identity contract that
   survives enumeration drift (#331). That is the part nobody publishes.

**Consequence for the PR-Check duty:** any future write-up or upstream PR
frames its delta against STADI explicitly. Novelty claims about
"proportional shares for diffusion" are not available.

### 5.2 The adjacent hit: HexiSeq

**HexiSeq** — arXiv 2605.07569 (2026-05-11). Fully asymmetric
context-parallel *and* head-parallel partitioning of sequence and attention
heads by hardware speed, for long-context LLM *training* on heterogeneous
clusters. Not DiT, not inference — but it is the same two-axis insight §2.2
derives for Ulysses (ragged sequence *and* ragged heads, or the attention
core stays equal-split). Independent corroboration that the two-axis
requirement is real, and evidence that the idea is in the air across
adjacent fields in 2026.

### 5.3 Verified negatives

| Where | Query | Result |
|---|---|---|
| xDiT issues + PRs | `heterogeneous`, `uneven`, `unbalanced`, `imbalance`, `unequal` | **0 results each**, verified via GitHub search, not inferred |
| xDiT issues + PRs | `load balance` | **1 hit**: PR #727 "Add Head Balancing for AITER Sparge Backends" (merged 2026-06-23) — permutes attention heads across Ulysses ranks by per-head *sparsity* cost. Visible in-tree at `usp.py:271` and `args.py:799`. Balances a workload skew across **identical** GPUs. Not capability heterogeneity. |
| USP paper (2405.07719) | load balancing | Only causal-mask imbalance, solved by chunk reordering across equal-degree devices. "Heterogeneous" in that paper means *networks*, matching `docs/methods/hybrid.md:5`. |
| PipeFusion (2405.14430), xDiT (2411.01738), DistriFusion (2402.19481), AsyncDiff | heterogeneous devices | none; all benchmark on uniform L40/A100/H100 nodes |
| ComfyUI built-in MultiGPU | mixed cards | documented as unsupported: the installed GPUs "must be the same"; `MultiGPU_WorkUnits` "assumes two GPUs with identical speeds" |
| ComfyUI-Distributed | — | data-parallel (same workflow, different seeds per card), not model-parallel; no intra-model split at all |
| `diffusers` `device_map="balanced"` | — | **confirmed serial**: places *components* (text encoder / transformer / VAE) on different GPUs for memory fitting, executed one stage at a time. Diffusers' actual parallel primitives (Ring, Ulysses, USP) all assume equal, evenly divisible degrees. |
| TensorRT multi-device (`IDistCollectiveLayer`), vLLM-Omni, SGLang-Diffusion, Ray Serve | — | scale-out across identical GPUs, or whole-request replica placement; no intra-forward capability-proportional split |
| Nunchaku, DiffSynth-Studio | — | **not verified either way** — nothing in docs/search, full source trees not read. Recorded as unverified, not as confirmed absent. |

Named explicitly so they are not mistaken for hits by title: **db-SP**
(2511.23113) is DiT sequence parallelism but balances *sparsity* skew across
identical GPUs; **CoCoDiff** (2604.14561) targets asymmetric interconnect
*bandwidth*, not compute; **GENSERVE** (2604.04335) and **DDiT**
(2506.13497) mean heterogeneous *request types* and *temporal* scheduling
respectively.

### 5.4 What this changes

Nothing about the cut plan; everything about how it is described. The gap is
real but narrow and recently opened: **no production diffusion engine ships
capability-proportional splitting** (xDiT, ComfyUI, diffusers, TensorRT,
Ray Serve — all verified negative), while one 2025 paper describes it on an
axis this rig cannot afford. The engineering contribution stands; the
"nobody has thought of this" framing does not.

---

## 6. License

**Apache-2.0, confirmed by reading the file, not the badge.**
`LICENSE.txt` at repo root is the verbatim Apache License 2.0 text
(January 2004) with the appendix filled in as `Copyright [2024] [xDiT Team]`.
Source files carry `# Copyright 2024 xDiT team.` headers. This clears the
only-MIT/Apache/GPL rule, and it is the same license our vendored copy is
already redistributing under — `sp_shard_utils.py` opens with
`# SPDX-License-Identifier: Apache-2.0`.

Two derived obligations for any port: keep the xDiT copyright headers on
files derived from theirs (already done in the vendored tree), and record
the upstream commit a port was taken from.

---

## 7. Vehicle choice

### 7.1 Image DiT: FLUX.1-dev

The choice is forced by the PipeFusion support matrix, not by preference.
Upstream's own table marks PipeFusion for exactly six models: HunyuanDiT,
**Flux (1.x)**, PixArt-α, PixArt-Σ, SD3, SANA. It is **not** supported for
Flux.2, Qwen-Image, Z-Image, Krea2, or any video model — those live in the
newer `models/runner_models/` path, which has no pipeline stage split at
all. Qwen-Image (20B, 60 layers, Apache-2.0) would be the better *model*;
it is SP-only, and SP is the mechanism §3.3 rules out here.

FLUX.1-dev fixed-cost sheet, 3 cards, 72 GB total (32 + 20 + 20):

| Post | bf16 | Notes |
|---|---|---|
| Transformer 11.85B | 23.8 GB | **sharded** by PipeFusion: 13.0 / 5.4 / 5.4 GB at `[19,19,19]` |
| T5-XXL encoder 4.76B | 9.5 GB | replicated by default; `--use_fp8_t5_encoder` → 4.8 GB, or precompute embeddings and drop it |
| CLIP-L + VAE | 0.42 GB | |
| CUDA context | ~0.5 GB/card | #400 ledger line item |
| Stale-patch KV cache | 1.1 GB @1024px / 3.95 GB @2048px | ∝ stage block count × tokens; scales *with* the ratio knob |
| Activations | ~0.03 GB @1024px | negligible; flash attention, nothing materialised |
| Corridor | 0.4 GB/card | ours, arbiter-owned |

Verdicts: **1024px fits naively on all three cards** (5090 ~24.5 / 32 GB;
3080 ~16.3 / 20 GB). **2048px needs the T5 offload** (3080 lands at 20.25 GB
without it, 10.75 GB with). **4096px does not fit** — the KV cache alone is
15.4 GB per stage; that is the honest ceiling, and it is a cache-precision
problem, not a share problem.

Where the split pays: at 1024px FLUX is 4608 tokens and the attention is a
small share of a linear-dominated forward, so the yield is close to the pure
FLOP ratio of §3.2. At 2048px (16 896 tokens) attention grows quadratically
and the yield rises, but so does the KV cache, so the resolution sweep is
also the VRAM sweep. Run both.

### 7.2 Video DiT: Wan2.1-T2V-1.3B first, Wan2.2-A14B as the scale target

For the #333-M4 uneven-*sequence*-parallel axis (which is our own lane, not
xDiT — video has no PipeFusion anywhere), the sequence length is the point:

| Vehicle | Weights bf16 | Tokens @ target | Fits our cards? |
|---|---|---|---|
| **Wan2.1-T2V-1.3B**, 480p, 81 frames | 2.6 GB, replicated | 52 × 30 × 21 ≈ **32.8k** | yes, everywhere, with room |
| Wan2.2-I2V-A14B, 480p, fp8 | ~14 GB, replicated | ~32.8k | 3080 at ~14 + activations — tight |
| Wan2.2 @ 720p | ~14 GB fp8 | 80 × 45 × 21 ≈ **75.6k** | attention dominates; the real SP case |

Recommend **Wan2.1-T2V-1.3B at 480p/81f** as the mechanism vehicle: it fits
on every card with margin, gives a genuine 33k-token sequence axis for
`SGLANG_SP_CAPACITY_WEIGHTS`, and isolates the uneven-share question from
the VRAM question. Wan2.2-A14B fp8 at 720p is the scale target once M4's
`all_gather_v` lands. Both are supported in our vendored lane already.
Note that the §3.3 interconnect verdict applies to video SP on *this* rig
too — the video vehicle is for validating the *mechanism*, per the
rig-is-a-floor rule, not for claiming a deployment win here.

---

## 8. Cut plan

### Cut 1 — falsify the yield model with zero code

**Setup.** Throwaway xDiT checkout (not a maintained fork). FLUX.1-dev,
1024px and 2048px, 28 steps, fixed seed, `pp_degree=3`,
`num_pipeline_patch` swept over {3, 4, 6}. Rank→card pinned via
`CUDA_VISIBLE_DEVICES` built from an **NVML UUID/name resolution at run
time**, never a hardcoded index (AUDIT_331). Rank 0 = 5090.

**Arms.**

| Arm | Config | What it answers |
|---|---|---|
| A0 | solo 5090, no parallelism | the baseline every other arm must beat |
| A1 | pp=3, default `[19,19,19]` | the 2.4:1:1 accident of §3.4 |
| A2 | pp=3, `--attn_layer_num_for_pp 20 19 18` | ratio-tuned for r≈3.5 |
| A3 | pp=3, `--attn_layer_num_for_pp 19 19 19` with rank 0 = **3080** | the deliberately wrong placement — the control that proves placement is what moves the number |
| A4 | pp=3, `--attn_layer_num_for_pp 12 24 21` | a deliberately anti-proportional list; must be *worse* than A1 or the metric is not measuring what we think |

A3 and A4 are the falsifiers. A result where A1..A4 all land within noise
means the run is comm- or host-bound and the whole ratio thesis does not
apply to this vehicle — which is a valid and cheap outcome.

**Metric.** ms per denoising step, and ms per stage per step (the
per-rank compute-vs-wait split, per the "ms/Runde als Messlatte" rule), not
images per second. Noise floor established by an A-vs-A pair first
(benchmark-harness duty 5).

**Byte gate.** Changing `attn_layer_num_for_pp` moves *which device* runs a
block; it does not change `(M, N)`, so it does not change PipeFusion's
staleness schedule. The output should therefore be bit-identical across
A1/A2/A4 **on identical silicon**. Gate it that way: run the layer-list
sweep on the two 3080s (pp=2) and require byte-identical images. On the
mixed set, sm86 vs. sm120 kernels differ and byte equality is not
available — that is the known Hetero-Spec-Determinismus boundary, so the
mixed arms get a semantic/tolerance gate instead, and the initial latents
must be sampled **on CPU** and moved (the CUDA-randn cross-arch rule),
otherwise the arms do not even share a starting point.

**Fixed-cost check before any card time** (feasibility-before-measurement): §7.1
says 1024px fits naively and 2048px needs `--use_fp8_t5_encoder`. The 400
MiB corridor holds in both. No arm is scheduled that does not pass this on
paper first.

**Effort.** One card window. No build. No fork.

### Cut 2 — planner-driven layer list (~80 LOC)

Only if cut 1 shows a ratio response. A solver that takes per-card rates
from `compute_rates_for_cards` (#348b), per-block cost weights (double = 36,
single = 15 for FLUX; uniform for MMDiT-style stacks), and emits the
contiguous block list minimising `max_i (units_i / rate_i)`. This is the
same shape as `_apportion` but weighted and contiguity-constrained. Lives on
our side, drives an upstream flag — no xDiT change.

### Cut 3 — port PipeFusion into `multimodal_gen` (~1200 LOC)

Only if cuts 1-2 hold. Scope: `_split_transformer_blocks` equivalent against
our model registry, the displaced-patch async loop, the stale-patch KV cache
manager, and the pipeline P2P with per-segment shape negotiation. The layer
list is a first-class ratio parameter from day one, not a retrofit, and
`estimate()` reports per-card weights analytically so the #400 ledger can
admit or reject before boot.

### Exclusions, hard reasons only

| Excluded | Hard reason |
|---|---|
| Uneven Ring attention | The block schedule is provably optimal at equal shares (§2.3); ragged shares degrade a property currently obtained for free, and the buffer fix lives in a second repository. Revisit only if a fixed-size-sub-block ring schedule is designed, which is a research task, not a port. |
| Uneven CFG-parallel | The unit of work is one whole forward pass — indivisible. Additionally inapplicable: FLUX and Qwen-Image are guidance-distilled and use no CFG. |
| xDiT's TP as a ratio axis | ~20 LOC to make ragged, but it covers the FFN of one model (StepVideo) and replicates all attention weights. Not a reason to exclude the *technique*; it is a reason not to spend the window. |
| Maintained fork of xDiT | Two model paths, two USP implementations, hard diffusers-version coupling, no planner/identity/ledger surface to reuse, and a vendored xDiT-derived lane already in tree. |
| 4096px FLUX on this rig | 15.4 GB stale-patch KV per stage in bf16 exceeds both card classes. Physical, at current cache precision. |

---

## 9. What would change these verdicts

- **A measured bf16-dense GEMM score at DiT shapes.** Every number in §3
  rides on `r`, and `r` is currently interpolated from quantized-LLM lanes.
  If DiT-shape bf16 puts the 5090 at 2.5x rather than 3.5x, equal-split
  parallelism stops being a regression and cut 1's urgency drops.
- **P2P or a faster interconnect.** §3.3's 50:1 comm ratio is what
  eliminates SP here. On a rig with NVLink or a PCIe switch, USP is
  competitive and the uneven-Ulysses work (§2.2) becomes the priority
  instead — which is the rig-is-a-floor caveat made concrete.
- **PipeFusion landing for Qwen-Image or Wan in upstream.** That would move
  the vehicle choice immediately, since both are better models than FLUX.1
  and both are already in our lane.
- **A ragged-capable `RingComm` upstream.** Would reopen §2.3, though the
  schedule objection would remain.
- **STADI releasing code, or xDiT adopting a capability-aware split.**
  Either would move this from "build it" to "evaluate theirs first" — the
  standing PR-Check duty. Re-run the §5.3 searches before any card window,
  not once.

---

## 10. Sources

Read on 2026-08-01, no card time.

- `github.com/xdit-project/xDiT` at commit
  `175e0bfec974d0b44a0be9fa511df2f9f5ead080` (2026-07-14, "feat: flux2
  pipefusion (#736)"), full source checkout. Files cited by path and line
  throughout §2.
- `LICENSE.txt` (Apache-2.0, `Copyright [2024] [xDiT Team]`), read in full.
- `README.md` (supported-model / mechanism matrix, parallel-degree product
  rule), `docs/methods/pipefusion.md`, `docs/methods/usp.md`,
  `docs/methods/hybrid.md`, `docs/performance/flux.md`.
- `feifeibear/long-context-attention`, `yunchang/ring/utils.py` (`RingComm`).
- USP paper, arXiv 2405.07719; PipeFusion paper, arXiv 2405.14430;
  xDiT paper, arXiv 2411.01738; DistriFusion, arXiv 2402.19481;
  AsyncDiff, arXiv 2406.06911.
- Prior art (§5): **STADI**, arXiv 2509.04719 (abstract page read directly,
  v1 2025-09-05); HexiSeq, arXiv 2605.07569; db-SP, arXiv 2511.23113;
  CoCoDiff, arXiv 2604.14561; GENSERVE, arXiv 2604.04335; DDiT,
  arXiv 2506.13497.
- xDiT PR #727 "Add Head Balancing for AITER Sparge Backends" (merged
  2026-06-23); GitHub issue/PR searches for
  `heterogeneous|uneven|unbalanced|imbalance|unequal` over
  `repo:xdit-project/xDiT` (all zero results).
- `huggingface.co/docs/diffusers/en/training/distributed_inference`;
  `docs.comfy.org/built-in-nodes/MultiGPU_WorkUnits`;
  `docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/multi-device-inference.html`.
- This tree on `integration/r3-probe-next2`:
  `docs/dev/DESIGN_333_multimodal_classes.md`,
  `docs/dev/DESIGN_333_M3_diffusion_lane.md`,
  `docs/dev/DESIGN_348b_cost_model.md`,
  `docs/dev/AUDIT_331_card_identity.md`,
  `docs/advanced_features/uneven_kv_token_ratio.md`,
  `docs/EVAL_p2p_prefill_decode_split.md` §4.2,
  `python/sglang/multimodal_gen/runtime/distributed/sp_shard_utils.py`,
  `python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py`.
</content>
