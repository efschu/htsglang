# DESIGN #333-M3 — The diffusion lane (Class 2) as a registry tenant

Status: implementation note for milestone M3 of #333. Read
`docs/dev/DESIGN_333_multimodal_classes.md` first; this note assumes its
taxonomy (three classes, one arbiter), its ledger (§3.3), its lane contract
(§3.5), and its seam list (§5.4). Repository-relative paths throughout; base
worktree `/spinning/wt-333m3`, branch `feat/diffusion-lane-333m3`.

M3's charter, in priority order: (1) inventory the vendored diffusion runtime
and pin the three fork deltas to file-level insertion points; (2) make the
sequence-parallel split capacity-weighted for unlike cards, with a hermetic
coverage proof; (3) promote the Class-2 registry adapter from estimate-only to
launching, wired to the arbiter's ledger, ladder and honest-rejection path;
(4) design the preview tap. This note is the deliverable that survives even
where the code is designed-only, and it states plainly which is which.

---

## 1. Upstream inventory: what `multimodal_gen` already provides

`python/sglang/multimodal_gen/` is upstream SGLang-Diffusion, vendored. It is a
complete second inference runtime and M3 does not rebuild any of it. The parts
that matter to the lane:

| Concern | Where | Note |
|---|---|---|
| Server entry | `runtime/entrypoints/cli/main.py` (`serve` subcommand → `cli/serve.py::execute_serve_cmd`) | booted as `python -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve …` |
| HTTP app | `runtime/entrypoints/http_server.py::create_app` (line 372) | routers mounted: health, image, video, weights, … |
| Health | `http_server.py` `/health` (GET, line 143) | the adapter's readiness probe |
| Image gen | `runtime/entrypoints/openai/image_api.py` — `router = APIRouter(prefix="/v1/images")` (line 46), `POST /generations` (line 147) | the OpenAI surface #335-M0 routes into |
| Host offload | `runtime/entrypoints/post_training/weights_api.py` `POST /release_memory_occupation` (154) / `POST /resume_memory_occupation` (178) → `runtime/managers/memory_managers/memory_occupation_controller.py` (release 155, resume 177) | this is `WARM_HOST` already built |
| Process model | `runtime/launch_server.py::launch_server` (171) spawns `num_gpus` scheduler processes via `mp.Process` | the adapter drives the CLI, not this directly |
| Dynamic batching | `runtime/managers/scheduler.py::Scheduler.get_next_batch_to_run` (835), `runtime/managers/dynamic_batch_admission.py` | admission caps are model/resolution/memory-aware |
| Denoise loop | `runtime/pipelines_core/stages/denoising.py::DenoisingStage._denoise` (1566); step loop `for step_index, t_host in enumerate(timesteps_cpu)` (1605) | the only step-boundary yield/preview point |
| Default port | `runtime/server_args/server_args.py` `port = 30000` (335); SP degree `ulysses_degree` (210), `ring_degree` (211), world `num_gpus` (203) | |

**The SP mechanism.** Sequence parallelism is Ulysses-style, and its single
sharding decision lives in
`runtime/distributed/sp_shard_utils.py::build_shard_plan`. A `SpShard` records
one per-rank chunk length; `shard_like` slices the local chunk, `gather_seq`
all-gathers the sequence back, `tail_attn_meta` builds the varlen attention
metadata. Before M3 the split was **equal only**: `local_len =
ceil(seq_len / sp_size)` on every rank, remainder padded onto the last rank's
tail so the gathered sequence carries one contiguous pad block at its global
tail (which `tail_attn_meta` then skips for free). The weight tensor-parallel
split is separately equal: `runtime/layers/linear.py` uses
`divide(output_size, tp_size)` (lines 345, 350) and
`runtime/layers/vocab_parallel_embedding.py` uses
`divide(global_vocab_size, world_size)` (line 83).

**No VRAM budget model.** `grep mem_fraction` / `grep memory_saver` over the
subpackage return nothing; memory strategy is per-component CPU-offload flags
plus a post-hoc `GPUWorker.do_mem_analysis`. This is why the ledger must supply
the budget from outside (§3 below).

---

## 2. The three fork deltas, with insertion points

Per DESIGN_333 §5.4 the fork's genuine Class-2 delta is exactly three things.
M3 delivers the first as code and the second as wiring; the third's collective
half is scoped to M4 and pinned here.

### Delta 1 — Uneven sequence-parallel (BUILT, M3)

**File:** `runtime/distributed/sp_shard_utils.py`. This is an allowed seam
(§5.4). Delivered in M3:

- `SpShard` (line 48) gains `offsets`, `lens`, `uneven` fields plus
  `local_offset`. The equal scheme leaves them empty and is byte-identical to
  the pre-M3 dataclass.
- `_apportion(total, weights)` (98): largest-remainder (Hamilton) split; sums to
  `total` exactly; guarantees ≥1 token per rank when `total ≥ sp_size`.
- `capacity_weights_from_env(sp_size)` (135): reads
  `SGLANG_SP_CAPACITY_WEIGHTS`; raises on a wrong-length/malformed vector rather
  than silently reverting to equal.
- `build_shard_plan(seq_len, weights=None)` (163): `weights=None` → read env →
  if still unset, the classic equal split (default path, unchanged). A weight
  vector switches to `_build_uneven_plan` (192): contiguous per-rank slices,
  **zero padding** (real lengths already sum to the whole), faster card owns a
  longer slice.
- `shard_like` (219) and `gather_seq` (255) grew an uneven branch. `gather_seq`
  carries unequal chunks over the fixed-size all-gather by padding each to the
  common maximum and cutting each rank's real slice back out — correct, at a
  bandwidth cost on the shorter ranks.

**Rate source (reused from K1, not reinvented).** The capacity weights are the
same per-card GEMM rates the K1 uneven-TP planner uses:
`python/sglang/srt/uneven_perf.py::rank_gemm_scores` (2074) →
`gemm_tflops`, loaded from the boot-fingerprinted measured registry
`load_measured_registry` (2431). The diffusion DiT is a dense bf16/fp16
transformer, so the dense GEMM probe is the right lane — exactly the score
`rank_gemm_scores` returns for a checkpoint with no quantized lane table. The
registry's Class-2 adapter computes the vector and exports it into the child's
`SGLANG_SP_CAPACITY_WEIGHTS`; `sp_shard_utils` never imports the registry, so it
stays hermetic and CPU-testable.

**Correctness rule (honoured).** An uneven split must change *who does the
work*, never the result. At `sp_size == 1` the uneven and equal plans are the
identical no-op `SpShard`, which is where the identity is checkable without a
collective; the hermetic test asserts it. The coverage invariant — concatenating
every rank's real slice reproduces the sequence with no gap and no overlap — is
asserted directly (mirroring the #345 pool-geometry discipline).

### Delta 2 — VRAM budget via the ledger (BUILT, M3)

**File:** new code in `python/sglang/srt/registry/adapters/class2_diffusion.py`
(outside the vendored subtree, per §5.4). The adapter's `estimate()` turns the
declared §5.2 posts into a per-card `ResourceProfile` the ledger reserves
against. Single-card is the validated shape: `peak = Σ posts`,
`steady = peak − activation_peak` (the transient the corridor check watches).
Multi-card (opt-in) replicates weights per rank and splits the activation peak
by the capacity weights, because SP shards the sequence, not the weights.

### Delta 3 — Arbiter integration (BUILT for launch/ladder/rejection; the SP-collective half is M4)

The Class-2 adapter now launches (§3 below). What remains for M4, pinned here:

| Remaining seam | File:line | Why M4 |
|---|---|---|
| Per-rank **weight**-TP shard table | `runtime/layers/linear.py:345,350` (`divide(output_size, tp_size)`) | uneven weight TP is a second lever from uneven SP; needs a size table, not a divide |
| Vocab per-rank table | `runtime/layers/vocab_parallel_embedding.py:83` | same |
| Variable-length all-gather | `runtime/distributed/sp_shard_utils.py::gather_seq` uneven branch | replace max-pad+trim with `all_gather_v` to drop the bandwidth cost |
| Joint-attention varlen under uneven | `sp_shard_utils.py::tail_attn_meta` | the tail-pad layout assumes uniform `local_len`; uneven joint layout is unbuilt |
| Cooperative step-boundary yield | `denoising.py:1605` (loop head) | the only lossless preemption point; identified, not built |

M3's multi-card path is therefore gated behind `launch.enable_uneven_sp`; the
default single-card / equal-split path is what M3 validates.

---

## 3. Registry / arbiter wiring (BUILT, M3)

`Class2DiffusionAdapter` (was estimate-only in M1) is now a launching adapter on
the same process template as Class 1/3
(`python/sglang/srt/registry/adapters/process.py`). It implements the lane
contract (`adapter.py::LaneAdapter`): `estimate` / `promote` / `demote` /
`state` / `measured` / `health` / `pids` / `bind`.

**Residency ladder (§5.3).** Class 2 is the best-served class because the
upstream server already ships host offload:

- `HOT` — boot the diffusion server, pin `CUDA_VISIBLE_DEVICES` to the placed
  cards, wait `/health`.
- `WARM_HOST` — `POST /release_memory_occupation`; promote back with
  `POST /resume_memory_occupation`, no reload. `promotion_cost_ms` is measured
  by the arbiter across the transition.
- `COLD` — stop the process group. Weights are on disk in their original form,
  so a cold promotion is a normal load; no hibernate parking is needed.
- `WARM_GPU` — refused loudly. §5.3 names the rung (drop the BCG pool and VAE
  tiling while keeping DiT weights resident) but the upstream server exposes no
  endpoint for it, and faking it onto another rung would make the ladder a lie.

**Capture lock (§3.6).** The diffusion warmup captures breakable CUDA graphs,
which is a capture on that device. `_boot` takes the arbiter's per-card
`card_exclusive_lock` around the boot so co-located tenants are quiesced during
capture. Capture is a promotion-phase event only; a capture requested in steady
state is a bug the arbiter logs (that negative test is an M3 acceptance item,
GPU-gated).

**Honest rejection (§7.5).** The adapter never invents a budget: an
under-declared spec fails at registration (`EstimateError` naming the missing
post), and a spec the ledger cannot fit is rejected by the arbiter with
`PromotionRejected` (arbiter.py:79) — projected wait, eviction set, and
per-card shortfalls, not "busy". This is the shape #335-M0's
`/v1/images/generations` adapter rejects into: when the diffusion engine is not
registered or not `HOT`, the serving surface returns that structured rejection
rather than a hang. (No #335 file is touched here; the K2 lane is the thing it
routes to.)

**Estimate-vs-boot split.** `model_path` is required to *boot*, not to
*estimate*: registration and `GET /registry/plan` (§7.4, "validate without
booting") must answer from posts alone, so the check lives in `_boot`.

---

## 4. Preview tap (DESIGNED, build if time)

From #344 / ANALYSE_347 §5: the diffusion lane decodes a low-resolution latent
preview every N steps as the observation pattern, so a long generate is
observable before its terminal output (§2.2: Class 2 has no usable output until
the last step *except optional previews*).

**Design.**

- **Tap point:** `denoising.py::DenoisingStage._denoise`, inside the step loop
  at line 1605. Every `preview_every_n_steps` steps, snapshot the current
  latent (a cheap `.detach()`; no `.cpu()` on the hot path).
- **Cheap latent→RGB:** decode at reduced resolution — either a strided VAE
  decode or a lightweight approximate decoder (TAESD-class), producing a small
  thumbnail, not a full VAE pass. The preview's cost must be a small fraction of
  a denoise step or it defeats its purpose.
- **Never block the main loop.** Run the preview decode on a **separate CUDA
  stream** and emit it asynchronously; if the previous preview has not finished,
  **drop** this one rather than queue it. This is the Class-3 back-pressure
  discipline (§2.2: drop, do not grow a queue) applied to observability: a slow
  preview consumer must never slow the denoise.
- **Transport:** previews are stream events on the generation response,
  terminal image unchanged. Under the registry the preview cadence is a
  per-request field, bounded by a spec `declared_max` so a pathological
  `preview_every_n_steps=1` cannot turn every step into a VAE pass.
- **VRAM:** the preview decoder's workspace is a §5.2 post
  (`vae_decode_peak_bytes` shares the tap) reserved like any other, so a preview
  never pushes the card over the corridor mid-generate.

Not built in M3.

---

## 5. What is built vs designed-only

| Item | State |
|---|---|
| Uneven-SP plan (`build_shard_plan` capacity-weighted) + `shard_like`/`gather_seq` uneven branches | BUILT, CPU-proven |
| Hermetic coverage + SP=1 identity + equal-path-identity tests | BUILT (`test_sp_shard.py`) |
| Class-2 adapter: launch / `WARM_HOST` ladder / capture-lock / NVML measure / image probe | BUILT (structure; GPU boot not exercised — cards contended) |
| Multi-card uneven-SP estimate + capacity-weight export | BUILT, opt-in (`enable_uneven_sp`) |
| Uneven **weight**-TP tables (linear/vocab), `all_gather_v`, joint varlen, step yield | DESIGNED, M4 (§2 Delta 3 table) |
| Preview tap | DESIGNED (§4) |
| GPU acceptance gates (co-resident generate-while-decode, capture-lock live, jitter) | NOT RUN (desk-only; cards contended) |

---

## 6. Test results

`test/registered/registry/` — 109 passed (adapter/ledger/control-plane/registry),
including the rewritten Class-2 tests: estimate-from-posts, WARM_GPU refusal,
boot-needs-model_path, multi-card-needs-opt-in.

`multimodal_gen/test/unit/test_sp_shard.py` — extended with the uneven-split
group (apportionment coverage/proportionality/≥1 floor, env parsing, plan
coverage no-gap-no-overlap, SP=1 identity, short-sequence fallback, no-weights
byte-identity, uneven `shard_like` slice, shard→gather roundtrip). The mm unit
suite requires the full diffusion dependency stack (diffusers, imageio), which
this desk venv does not carry; the uneven-SP logic was proven by a hermetic
runner that stubs the two sibling modules and loads the real
`sp_shard_utils.py` from disk — all assertion groups pass. The committed test is
the canonical form for the diffusion CI env.
