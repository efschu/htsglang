# DESIGN #333 — Multimodal serving as a three-class architecture

Status: design document. No code in this branch. This is the document the
staged build-out of #333 sits on; every milestone below names the files it
touches and the gate it has to pass.

Written to be read without this conversation's context. Repository-relative
paths throughout (repository root = the `htsglang` fork worktree). Base
commit: `8cc836bb40`.

Companion document, read first for Class 3 background:
`docs/dev/ANALYSE_333_prior_art_vsgan.md` (prior-art reading of
`styler00dollar/VSGAN-tensorrt-docker`, already merged).

---

## 0. Scope, sources, and what is and is not established

### 0.1 What this document decides

1. A taxonomy of served workloads into **three classes**, with a boundary
   test that is a property of the workload rather than of the model family
   (§2).
2. The **co-tenancy substrate** that carries all three on one set of
   physical GPUs: a per-physical-GPU VRAM ledger plus a tick broker, rather
   than one merged scheduler (§3).
3. Per class: scheduler interface, memory posts in the planner, residency
   ladder placement, VRAM-regulator and staircase compatibility, graph
   strategy (§4, §5, §6).
4. The **engine registry** — N registered / M hot, idle default set,
   runtime add and remove — engine-agnostic across the three classes. This
   section is the build specification for #305-M1 (§7).
5. The **Class-3 first build-out** in full: the video-enhance stream server,
   its pipeline, its arithmetic, its named measurement posts (§8).
6. **Prior-art verdicts** per class with an explicit reuse choice under the
   order dependency > port > own build (§9).
7. A **staged build plan** M0-M5 with effort classes, feasibility gates, and
   an honest statement per stage of what it cannot do (§10).

### 0.2 Established facts this document rests on

Everything in this subsection was read out of the tree at `8cc836bb40` or
verified against a primary source. Section references point at where each
fact is used.

**The fork already carries a complete second inference runtime.**
`python/sglang/multimodal_gen/` is upstream SGLang's diffusion runtime
("SGLang Diffusion"), vendored into this fork, roughly 430 files under
`runtime/` alone. It has its own scheduler
(`python/sglang/multimodal_gen/runtime/managers/scheduler.py`), its own ZMQ
transport, its own worker processes
(`runtime/managers/gpu_worker.py`), its own FastAPI application
(`runtime/entrypoints/http_server.py`), its own server-args module
(`runtime/server_args/server_args.py`, ~2450 lines), its own distributed
parallel state (`runtime/distributed/parallel_state.py`), and its own
CLI. It supports approximately 22 pipeline families (FLUX, Qwen-Image,
Wan, Z-Image, SANA, Cosmos3, LTX-2, Hunyuan/Hunyuan3D, StableDiffusion3,
pi05 VLA, and others). This single fact reshapes Class 2 from a greenfield
build into an integration problem. See §5, §9.2.

**The two runtimes are process-exclusive today, but not import-isolated.**
`python/sglang/cli/serve.py` detects the model type
(`get_is_diffusion_model`, `--model-type {auto,llm,diffusion}`) and
dispatches to exactly one of `sglang.launch_server.run_server` (the
`srt` engine) or
`sglang.multimodal_gen.runtime.entrypoints.cli.serve.execute_serve_cmd`.
No code path runs both in one process. At the Python-import level, however,
`multimodal_gen` depends on `srt` in about forty places: breakable-CUDA-graph
primitives from
`sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph`,
quantization kernels from `sglang.srt.layers.quantization.*`, custom
all-reduce from `sglang.srt.distributed.device_communicators`, and
`runtime/distributed/parallel_state.py::_sync_srt_world_group()`, which
sets `srt`'s `_WORLD` group from `multimodal_gen`'s own if it is unset. The
real seam is the process and request lifecycle, not the import graph. See
§3.2.

**Diffusion dynamic batching is already present in the vendored copy.**
`ServerArgs.batching_max_size`, `batching_delay_ms`,
`enable_batching_metrics`, plus
`runtime/managers/dynamic_batch_admission.py::BatchAdmissionController`
with model/resolution/device-memory-aware caps, and signature-based request
merging in `Scheduler.get_next_batch_to_run()`. See §5.1.

**Diffusion sequence-parallel splits are equal-only.** The single place the
split is computed is
`python/sglang/multimodal_gen/runtime/distributed/sp_shard_utils.py::build_shard_plan()`:

```python
local_len = (seq_len + sp_size - 1) // sp_size    # every rank identical
num_pad   = local_len * sp_size - seq_len         # slack padded at the tail
```

Every rank gets the same `local_len` by ceil-division; the remainder is dead
compute padded onto the last rank. There is no per-rank size table anywhere
in the subpackage. Tensor-parallel linear layers
(`runtime/layers/linear.py`, `runtime/layers/vocab_parallel_embedding.py`)
likewise assume `output_size // tp_size`. This is the exact counterpart of
the problem the fork already solved for AR models with `--rank-tp-ratio`,
and it is the fork's genuine delta for Class 2. See §9.2, M4.

**Diffusion has no VRAM budget model.** `grep mem_fraction` over
`multimodal_gen/` returns nothing; `grep memory_saver` returns nothing.
Memory strategy is per-component CPU-offload flags (`--dit-cpu-offload`,
`--dit-layerwise-offload`, `--vae-cpu-offload`, `--text-encoder-cpu-offload`)
plus an advisory post-hoc analysis in `GPUWorker.do_mem_analysis()`.
Release/resume is `runtime/managers/memory_managers/memory_occupation_controller.py`,
which does `module.to("cpu")` plus `gc.collect()` and `empty_cache()` — not
an allocator-level pause. See §5.2, §5.3.

**The fork's own VRAM accounting is measured, per-boot, and single-tenant.**
`python/sglang/srt/uneven_perf.py::load_measured_registry(server_args)`
loads a JSON cache keyed by a boot fingerprint;
`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` builds the
per-rank component dictionary with keys `device_total_bytes`, `ranks_on_gpu`,
`residual_residency_bytes`, `ctx_overhead_bytes`, `weights_alloc_bytes`,
`weights_param_bytes`, `pools_bytes`, `kv_pool_bytes`, `kv_pool_tokens`,
`mamba_aux_pool_bytes`, `graphs_ws_bytes`, `draft_solo_pool_bytes`,
`graphs_ws_excl_draft_pool_bytes`, `rung_tags`, `frag_bytes`,
`boot_transient_bytes`, `safety_mib`, `required_free_bytes`,
`free_bytes_at_measure`, `max_total_num_tokens`. These are
`all_gather_object`'d across ranks and fed back on the next boot as a
correction. There is no tenant dimension and no pre-boot reservation. See
§3.3.

**Generation and embedding cannot be served from one process today.**
`ModelConfig.is_generation` (`python/sglang/srt/configs/model_config.py`)
is set once at boot from the `--is-embedding` server arg and read
process-globally in `tokenizer_manager.py` and `scheduler.py`. This is the
structural ceiling that M0 cannot lift and M1 exists to lift. See §4.3,
§10.

**The scheduler's extension point is a type dispatcher.**
`python/sglang/srt/managers/scheduler.py::Scheduler.init_request_dispatcher`
(around line 1643) builds a `TypeBasedDispatcher` keyed on the Python type
of the tokenized request struct: `(TokenizedGenerateReqInput,
self.handle_generate_request)`, `(TokenizedEmbeddingReqInput,
self.handle_embedding_request)`, and roughly forty control-plane types. A
new request class registers a `(Type, handler)` tuple here. There is no
`ForwardMode.EMBED`; an embedding request is a `ForwardMode.EXTEND` that
never transitions to `DECODE`. See §6.1.

**A third disaggregation role already exists.**
`python/sglang/srt/disaggregation/encode_server.py`,
`encode_receiver.py`, `encode_grpc_server.py` implement a standalone
multimodal encode service with cross-modality load balancing across
encoders. This is precedent inside the fork for "a stage of the pipeline is
its own service", and it is the closest existing thing to a Class-3 tenant.
See §6.

**Hibernate is two-state and GGUF-scoped.**
`python/sglang/srt/model_loader/hibernate.py` (task #89) parks
post-`process_weights_after_loading` GPU tensors to disk, keyed by
`current_gpu_nvml_uuid()` so restore lands on the same physical card;
`_assert_v1_scope()` refuses PP/DP/EP. The generic mechanism underneath is
`python/sglang/srt/utils/torch_memory_saver_adapter.py`, a tag-based
allocator pause/resume with tags `weights`/`kv`/`graphs`. Those tags are
**process-wide**, not lane-scoped. See §3.4, §4.4.

**`sgl-model-gateway/` is a router, not a residency manager.** The Rust
gateway (`--enable-igw`) routes traffic across independently launched,
fully resident worker processes and adds/removes workers via `POST
/workers`. It has no VRAM awareness, no promotion/demotion, no residency
state. It is the layer *above* the registry designed in §7, not a
substitute for it.

**Task numbers.** #274 (dual-group runtime) is real and partly landed;
its design document is `docs/DESIGN_121_dual_group_runtime.md` and its
server args live in `python/sglang/srt/server_args.py` under the comment
"Multi-group runtime (#121/#274)" (around line 3889). #287 (staircase /
dynamic rung switching between prefill-optimal and decode-optimal ratio
vectors) exists as analysis in `docs/dev/ANALYSE_321_nvfp4_asymmetry.md` §8
and `docs/dev/INTEGRATION_R3_VALIDATION.md`, not as code. **#305 and #330
do not appear anywhere in the tree.** They are forward references: #305 =
residency ladder (HOT/WARM/COLD plus hibernate), #330 = VRAM regulator
(corridor enforcement and waste accounting). This document specifies
against them and, for #305-M1, *is* their build specification.

### 0.3 Facts used but not verified in this pass

Flagged so a later reader does not treat them as settled.

| Claim | Confidence | How to settle it |
|---|---|---|
| RTX 5090 has 3 NVENC / 2 NVDEC engines enabled; RTX 3080 has 1 / 1 | high (NVIDIA support matrix) | re-read the matrix page raw, not through a summarizer |
| Consumer concurrent NVENC session cap is currently 12 | medium | fetch the matrix page's raw HTML or footnote text; the cap is driver-enforced and has moved 3 -> 5 -> 8 -> ? over time |
| NVDEC has no documented concurrent-session limit | medium | same source |
| `xDiT` upstream supports equal splits only | medium (docs, not a code audit) | irrelevant to our build — we patch the vendored copy in `multimodal_gen`, whose equal-split behaviour *is* code-verified above |
| Per-stage throughput at any resolution | none — nothing measured | §8.6 probe harness |

**Correction to a plausible misreading.** `efschu/vs-mlrt`'s commit message
mentions "standalone mode". Reading `vstrt/vs_tensorrt.cpp` in that fork,
the `standalone_mode` symbol means "run pure GPU Lanczos-3 resize without
loading a TensorRT engine". It does **not** mean "runs without VapourSynth".
No VapourSynth-free build target exists in either `AmusementClub/vs-mlrt`
or the fork today. §9.3 plans the extraction that would create one.

---

## 1. The problem

The fork's hardware target is a small number of unlike GPUs — the reference
rig is one RTX 5090 (32 GB, sm120) and two RTX 3080 (20 GB each, sm86), with
no NVLink and no CUDA peer-to-peer, all cross-GPU traffic host-staged, one
3080 on a PCIe Gen4 x4 slot at roughly 6.5 GB/s against 13-14 GB/s for the
other two. On such a rig the interesting question is not "how fast is one
model" but "how many different things can share these three cards without
any of them falling over".

Today the fork answers that question for exactly one workload shape:
autoregressive token generation, possibly with a second overlapping TP group
over the same weight bytes (#274). Everything else — image and video
generation, embeddings, rerankers, speech, and single-pass utility inference
such as video upscaling — is either a separate process with no coordination,
or not served at all.

Three constraints make a single merged scheduler the wrong answer:

- **Upstream churn.** `multimodal_gen` is ~430 files under active upstream
  development. Merging its request lifecycle into `srt`'s would create a
  permanent rebase tax against a codebase we do not control and do not want
  to diverge from.
- **Incompatible admission contracts.** An AR decode step does not know when
  the request ends. A diffusion request knows its exact step count at
  admission. A single-pass utility request knows its exact cost. These want
  three different admission policies, and forcing them through one policy
  degrades all three (§2.2).
- **The contended resource is not the event loop.** Two tenants on one card
  contend for VRAM bytes, for SM time, for CUDA graph capture windows, and
  for the copy engines. None of those are mediated by whose Python loop runs
  the request.

So the architecture is: **three schedulers, one arbiter**. The arbiter owns
what is actually shared.

---

## 2. The three classes

### 2.1 Definitions

A class is defined by the shape of a request's time axis and where its state
lives — not by modality and not by model family. A speech model can be
Class 1 or Class 3 depending on how it is served.

**Class 1 — Autoregressive token generation.** Iterative. State (KV cache,
SSM/GDN state) persists across iterations and belongs to the request.
Iteration count is data-dependent and unknown at admission. Output is
produced incrementally. Memory is dominated by a large persistent pool sized
once at boot and shared across all in-flight requests.
Exists today: `python/sglang/srt/managers/scheduler.py`.

**Class 2 — Diffusion / iterative-refinement generation.** Iterative. State
(the latent) persists across iterations and belongs to the request.
Iteration count is *known exactly at admission* (`num_inference_steps`).
No usable output until the last step, except optional previews. Memory is
dominated by a transient activation peak that scales with resolution,
batch, and CFG branch count; there is no cross-request pool.
Exists today as a separate runtime: `python/sglang/multimodal_gen/`.

**Class 3 — Single-pass / utility inference.** Non-iterative from the
scheduler's point of view: one forward per work unit. If there is sequential
state (successive video frames, a streaming session), it lives outside the
model, in the caller's stream. Work units are independent and of uniform,
statically known cost. Memory is a bounded workspace proportional to the
number of in-flight work units.
Does not exist today. Two disjoint populations belong here:
- **prefill-only model requests**: embeddings, rerankers, classifiers,
  scoring, standalone vision/audio encoding. Partly present in `srt`
  (`serving_embedding.py`, `serving_rerank.py`, `serving_classify.py`,
  `serving_score.py`, and the encode disaggregation role), but structurally
  locked out of coexisting with generation (§0.2).
- **non-LLM executors**: TensorRT engines for super-resolution, frame
  interpolation, resize, denoise, plus NVDEC/NVENC. Entirely new.

### 2.2 The boundary test

The operative question is: **what must the scheduler know at admission time,
and what may it do to a request already in flight?**

| | Class 1 | Class 2 | Class 3 |
|---|---|---|---|
| Cost known at admission | no | yes (steps x step cost) | yes, exactly |
| Completion time predictable | no | yes, within the step-cost error | yes |
| Preemption granularity | one token step, state must be preserved or recomputed | one denoise step, latent must be preserved | none needed — work unit is short; drop or delay instead |
| Output cadence | incremental, every step | terminal (plus optional previews) | terminal per work unit |
| Dominant memory term | persistent pool (KV/SSM) | transient activation peak | in-flight workspace |
| Correct back-pressure | queue plus KV admission plus preemption | quota and deadline scheduling | rate limiting, no queue growth |
| Failure mode when overloaded | latency cliff, then preemption thrash | latency cliff, requests still complete | frame drop or producer stall — must be explicit |

The three admission contracts follow directly:

- **Class 1** needs *continuous* batching and preemption because it cannot
  predict anything.
- **Class 2** needs *deadline and quota* scheduling because it can predict
  everything but cannot yield partial results.
- **Class 3** needs *rate control and back-pressure* because it can predict
  everything and its work units are individually cheap; queueing them is
  strictly worse than refusing or slowing the producer.

### 2.3 Consequences of the taxonomy

Two consequences worth stating up front because they drive §3.

**A class can host several engines.** Class 3 hosts both a BGE embedding
model and a TensorRT super-resolution engine. They share an admission
contract, not an implementation.

**A model family can move between classes.** A Qwen-Omni speech model served
as streaming speech-to-speech is Class 1 (autoregressive audio tokens). The
same model's audio encoder served standalone is Class 3. The registry in §7
therefore keys on *class per engine instance*, never on model architecture.

---

## 3. The substrate: co-tenancy over the #274 group runtime

### 3.1 What #274 established and what generalizes

`docs/DESIGN_121_dual_group_runtime.md` established that one process can run
two overlapping TP groups over one weight set: the 5090 holds full weights
exactly once, serving both rank 0 of the big uneven TP=3 group and a
standalone prefill lane. Its decision for in-process rather than
second-process co-location rests on five arguments, of which the load-bearing
ones are:

1. Two ranks of *the same NCCL communicator* on one physical GPU require
   runtime NCCL >= 2.30; the rig measures 2.28.9 and has no MPS daemon.
2. Sharing weights across processes would need CUDA IPC, already rejected in
   DESIGN_107 for lack of a single allocation point.
3. Group-internal collectives vanish: `all_gather` becomes `torch.cat`,
   `all_reduce` becomes `+`. The lane has no communicator, so it cannot
   collide with the group's.
4. A second process costs its own CUDA context, roughly 0.5-1 GiB per card —
   precisely the reserve the lane's KV was to be paid from.

**These arguments are specific to weight sharing.** Classes 2 and 3 do not
share weights with the LLM: a diffusion transformer and a TensorRT SR engine
have nothing in common with a Qwen checkpoint. Arguments 2 and 3 do not
apply. Argument 1 does not apply either, and this matters: it constrains two
ranks *within one communicator*, not two independent processes each holding
one rank of two different communicators. A diffusion tenant with its own
process group on the same card is one rank per communicator per card and
never crosses the NCCL 2.30 threshold.

Argument 4 survives in full and becomes a **ledger line item** rather than an
objection: each process-isolated tenant costs one CUDA context per card it
touches, and that cost must be reserved before the tenant boots.

**The generalization.** #274 is "two TP groups over one weight set in one
process". #333 is "N tenants over one VRAM budget across processes". The
mechanism that makes the first safe is nesting algebra
(`python/sglang/srt/distributed/dual_group.py::check_nesting`). The mechanism
that makes the second safe is a cross-process ledger with reservations.

### 3.2 Decision: three schedulers, one arbiter

**Decision.** Do not merge `multimodal_gen`'s scheduler into `srt`'s.
Instead define a narrow **lane contract** that every class adapter
implements, and put a per-physical-GPU **arbiter** underneath all of them.

```
      Class 1 adapter          Class 2 adapter          Class 3 adapter
   (srt Scheduler, in-proc)  (multimodal_gen Sched.,  (utility executors,
                              own processes)            own processes)
            |                        |                        |
            +------------------------+------------------------+
                                     |
                            lane contract (§3.5)
                                     |
                    per-physical-GPU ARBITER (one per card)
                    - VRAM ledger      (§3.3)
                    - capture lock     (§3.6)
                    - tick broker      (§3.7)
                    - corridor guard   (#330, §3.8)
```

Rationale, in the order that decided it:

1. **Rebase cost.** `multimodal_gen` is upstream code under heavy
   development. Every line we change inside it is paid for again at every
   rebase. The design therefore fixes a *named, minimal seam list* (§5.4)
   and treats everything else in that subpackage as read-only.
2. **The contracts genuinely differ** (§2.2). One scheduler serving all three
   would need three policy branches at every decision point — which is a
   merged scheduler in name only, with worse cohesion than three separate
   ones.
3. **The contended resources are all below the scheduler.** VRAM bytes, SM
   time, graph-capture windows, copy engines, NVDEC/NVENC engines. An
   arbiter that owns exactly those is smaller and more testable than a merged
   loop.
4. **Prior art agrees.** NVIDIA's TensorRT-LLM VisualGen keeps the diffusion
   pipeline deliberately separate from the LLM inference path. `vllm-omni`
   runs `OmniARScheduler` alongside `OmniGenerationScheduler` and its own
   RFC #5279 reports the shared-lifecycle code as the part that went wrong.
   Upstream sglang keeps `multimodal_gen` separate from `srt`. Three
   independent projects converged on sibling lanes; the one that tried to
   share lifecycle code is the one asking to refactor it.

**What this costs, stated honestly.** Cross-class scheduling decisions are
coarse. The arbiter can grant or withhold a tick and can move bytes between
tenants at tick boundaries, but it cannot interleave a diffusion step
*inside* an AR decode batch, and it cannot preempt a diffusion step
mid-flight. Fine-grained SM-level sharing between classes is out of scope
for #333 and would need MPS or green contexts, neither of which is available
on this rig (no MPS daemon; see DESIGN_121 §1.2).

### 3.3 The VRAM ledger: from measured to reserved

Today's registry (§0.2) is *measured, per-boot, single-tenant*. Multi-tenancy
needs a *declared, pre-boot, cross-process* reservation, because a tenant
that has not booted yet cannot have measured itself, and a tenant in another
process cannot be `all_gather_object`'d.

**Design.** Keep the measured registry exactly as it is; it remains the
source of truth for how much a tenant *actually* used, and remains the
feedback that converges pool sizing across boots. Add a reservation layer on
top of it.

- **Reservation store**: one file per physical GPU, keyed by NVML UUID, at
  `/run/htsglang/vram/<nvml_uuid>.json`. NVML UUID rather than CUDA index,
  for the same reason `hibernate.py::current_gpu_nvml_uuid()` uses it:
  enumeration order shifts between boots and driver states. The existing
  cross-session GPU arbitration convention under `/spinning/gpu-arb/`
  (published windows, holder plus heartbeat) is the same pattern and the
  store should follow it — holder identity, heartbeat, and a lease that
  expires so a crashed tenant does not hold bytes forever.
- **Entry schema** per tenant on that card:

  ```
  tenant_id        stable id from the registry (§7)
  klass            1 | 2 | 3
  state            HOT | WARM_GPU | WARM_HOST | COLD
  reserved_bytes   what the arbiter has promised this tenant
  measured_bytes   last observed peak, from the measured registry
  posts            {post_name: bytes}   -- see §4.2/§5.2/§6.2
  pid, heartbeat_ts, lease_expiry_ts
  ```

- **Invariant, enforced by the arbiter on every admission and every
  promotion**:

  ```
  sum(reserved_bytes for tenants in HOT or WARM_GPU on card C)
      + corridor_bytes(C)
      <= nvml_total_bytes(C)
  ```

  `corridor_bytes` is #330's rule: at least 400 MiB absolutely free per card
  before any boot. Under multi-tenancy the corridor is a property of the
  *card*, not of any tenant, so no tenant may account for it and the arbiter
  must own it.
- **Waste accounting**: the #330 rule that net waste above 1.5 GiB is a
  registered item becomes computable across tenants:
  `waste(C) = sum(reserved) - sum(measured)`. The arbiter reports it; it does
  not silently reclaim, because reclaiming a reservation a tenant has not
  used yet is how you get a runtime OOM three minutes later.
- **Relation to `--rank-gpu-memory-mib`**: unchanged for Class 1. That flag
  already declares a rank's entire absolute budget. Under the ledger it
  simply becomes that tenant's `reserved_bytes` on that card. The
  established rule — the value is the whole budget, no implicit ceiling, no
  safety factor, no rounding down, headroom is the operator's
  responsibility — carries over verbatim to Class 2 and Class 3 budgets.

**Why reservations rather than dynamic sharing.** Dynamic sharing across
processes requires either a shared allocator (does not exist across CUDA
contexts) or trust that each tenant releases on demand (does not survive a
tenant that is mid-`cudaMalloc`). Reservations are the only mechanism that
fails at plan time rather than at 03:00 with a fragmented heap.

### 3.4 The residency ladder (#305) and why process isolation earns it

**States.**

| State | Weights | Graphs / contexts | Pools | Promote cost from here |
|---|---|---|---|---|
| `HOT` | device | captured | full | 0 |
| `WARM_GPU` | device | released | shrunk to floor | graph recapture only |
| `WARM_HOST` | pinned host RAM | released | released | PCIe transfer of weights |
| `COLD` | disk, post-`process_weights_after_loading` | none | none | disk read plus transfer |

`COLD` is exactly what hibernate (#89,
`python/sglang/srt/model_loader/hibernate.py`) already implements — parked
final-form GPU tensors plus a manifest, NVML-UUID-locked to the same physical
card. Its current scope is GGUF-only and single-node-TP-only; widening it is
tracked with #89, not here.

**The gap that decides the process model.** The generic mechanism under
hibernate is `torch_memory_saver_adapter.py`, a tag-based allocator
pause/resume with tags `weights`/`kv`/`graphs`. Those tags are
**process-wide**. An in-process second lane therefore cannot be parked
independently of its host — parking the `weights` tag parks both. This is
already recorded as a known gap in `docs/dev/INTEGRATION_R3_VALIDATION.md`
("memory-saver tags ... generic at the tag level; missing lane-scoped tags
for the class `cold_lane`").

For process-isolated tenants the problem disappears: **the process boundary
is the tag scope.** Parking a Class-2 or Class-3 tenant is releasing its
memory and, at `COLD`, exiting its process. No new allocator work is
required.

This is the second independent argument for process isolation of Classes 2
and 3, and it is the stronger one: without it, #305's ladder needs
lane-scoped memory-saver tags — a change deep in the allocator adapter —
before any of it works. With it, the ladder for Classes 2 and 3 is
implementable in the registry alone.

**Per-class ladder support is uneven, and Class 2 is the easy one.**
`multimodal_gen` already ships
`runtime/managers/memory_managers/memory_occupation_controller.py` with
`/release_memory_occupation` and `/resume_memory_occupation` endpoints doing
`module.to("cpu")`. That is `WARM_HOST` already built. Class 1's
`WARM_HOST` is harder because its weights are sharded, quantized, and
post-processed. Class 3's ladder is trivial because TensorRT engines are
small; what is expensive there is the engine *build*, which is a disk-cache
problem (§6.4), not a residency problem.

### 3.5 The lane contract

Every class adapter implements this and nothing more. Deliberately narrow:
the arbiter must not need to understand a request.

```
class LaneAdapter(Protocol):
    klass: int                       # 1 | 2 | 3

    # planning, before anything boots
    def estimate(spec) -> ResourceProfile
        # {card_uuid: {post_name: bytes}}, plus context overhead,
        # plus a declared peak-vs-steady split

    # residency
    def promote(target: State) -> None
    def demote(target: State) -> None
    def state() -> State

    # execution
    def can_admit(request) -> Admission        # accept | defer(t) | reject(reason)
    def submit(request) -> Handle
    def tick_request() -> TickRequest | None   # what the lane wants next (§3.7)
    def grant(tick: TickGrant) -> None

    # exclusive-window negotiation
    def needs_capture_window() -> bool
    def capture(window) -> None

    # observability
    def measured() -> dict[str, int]           # feeds the ledger's measured_bytes
    def health() -> Health
```

`estimate()` is the contract's most important method: it is what makes the
registry engine-agnostic (§7). Everything class-specific — how a diffusion
activation peak scales with resolution, how a KV pool scales with tokens, how
a TRT workspace scales with in-flight streams — lives behind it.

### 3.6 The capture lock

CUDA graph capture is not thread-safe against other work on the same device:
during capture the capturing stream takes over, and other tenants' kernels on
that device can stall or fail. All three classes capture graphs:

- Class 1: decode graphs, plus breakable graphs
  (`python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph`).
- Class 2: breakable CUDA graphs over the DiT, at
  `python/sglang/multimodal_gen/runtime/breakable_cuda_graph/runner.py`,
  which reuses the `srt` primitives directly. Its documented rule is
  "serving never triggers a fresh capture" — all capture happens at warmup.
- Class 3: TensorRT builds graphs internally when the engine was built with
  `--useCudaGraph`; capture happens at engine build or first inference.

**Rule.** The arbiter holds one capture lock per physical GPU. A tenant that
wants to capture requests the lock, and the arbiter quiesces the other
tenants on that card first — for Class 1 that means finishing the in-flight
batch and not starting the next; for Class 2 finishing the current denoise
step. Capture windows are boot-phase and promotion-phase events, never
steady-state events. A tenant that requests a capture window during steady
state is a bug and the arbiter logs it as one.

This rule also inherits DESIGN_121's warning verbatim: capture time is
rank-locally visible, so no collective may be expected from that rank during
the window.

### 3.7 The tick broker

The arbiter does not schedule requests. It hands out *permission to run a
unit of work* on a card, and it does so with a deficit budget.

- Each tenant declares, via `tick_request()`, what it wants next and its
  estimated cost in milliseconds: an AR decode iteration, a diffusion step, a
  batch of Class-3 work units.
- The arbiter maintains a per-tenant deficit counter with a per-tenant weight
  and grants ticks in deficit order, capped by a per-card in-flight budget.
- A tenant may always run *within* an already-granted tick; the arbiter never
  interrupts.

The idea is imported from Sangam (arXiv 2607.04206), which uses a deficit
token-budget scheduler to colocate block-diffusion decoding with AR prefill
admission. Its domain is text diffusion, not image or video, so nothing is
ported — only the deficit-budget shape, which is the right shape here for the
same reason: it gives a predictable share to a lane whose work units are of
very different sizes, without needing preemption.

**What the tick broker deliberately does not do.** It does not try to hide
one class's latency behind another's. On this rig, with no MPS, two processes'
kernels on one card time-slice at the driver's discretion, and a 200 ms
diffusion step will add jitter to AR decode no matter how the ticks are
handed out. The broker's job is to bound that jitter (by bounding how much
work is in flight per card), not to eliminate it. §10 M3's gate measures it
rather than assuming it away.

### 3.8 Interaction with #287 (staircase) and #330 (VRAM regulator)

**#287, the staircase.** Switching a rank between a prefill-optimal and a
decode-optimal ratio vector re-shards weights, which changes
`weights_alloc_bytes`. Under multi-tenancy this is dangerous: a rung change
that grows a tenant's footprint can push a card over the corridor while
another tenant is mid-forward.

**Rule.** A rung may move freely *inside* the tenant's reservation. A rung
whose plan-time footprint exceeds the tenant's `reserved_bytes` is rejected
at plan time, not attempted and rolled back. Concretely: the staircase's
per-rung footprint must be computed for *all* rungs at boot and the tenant's
reservation set to the maximum across rungs. Reserving the maximum wastes
bytes at the non-maximal rungs, and that waste is a #330 registered item —
correctly, because it is real and the operator should see it.

`ANALYSE_321_nvfp4_asymmetry.md` §8.3 identifies the structural work #287
needs independently of this: `rank_gemm_scores(entries, fmt) -> List[float]`
returns one scalar per rank and must become
`Dict[family, List[float]]` for mlp/attn/moe/vocab. That stays #287's work.

**#330, the VRAM regulator.** Three items become the arbiter's
responsibility rather than any tenant's: the >= 400 MiB absolute free
corridor per card, the net-waste computation across tenants
(`sum(reserved) - sum(measured)`), and the "no tests on red" rule, which
under multi-tenancy means the arbiter refuses to promote a tenant onto a card
already below corridor rather than each tenant checking independently.

---

## 4. Class 1 — autoregressive token generation

Exists. This section records only what changes.

### 4.1 Scheduler interface

Unchanged. `python/sglang/srt/managers/scheduler.py` keeps its event loop,
its `TypeBasedDispatcher`, its continuous batching and preemption. The only
addition is that when the registry is enabled, the `Scheduler` is wrapped by
a Class-1 `LaneAdapter` that maps `tick_request()` onto "I want to run the
next batch" and `grant()` onto "run it".

**Backward compatibility rule, non-negotiable.** With the registry disabled
— which is the default — the adapter is not constructed and the scheduler's
loop is byte-for-byte the path it is today. The gate for every milestone
below includes an unchanged-default boot of the reference TP=3 command.

### 4.2 Memory posts

Already enumerated by the measured registry (§0.2). Under the ledger they are
namespaced under the tenant and gain nothing new except `tenant_id`. The
existing keys map straight across: `weights_alloc_bytes`, `kv_pool_bytes`,
`mamba_aux_pool_bytes`, `graphs_ws_bytes`, `draft_solo_pool_bytes`,
`ctx_overhead_bytes`, `frag_bytes`, `boot_transient_bytes`.

One addition: `rung_max_weights_alloc_bytes`, the maximum across #287 rungs
(§3.8).

### 4.3 Residency ladder

`HOT` and `COLD` exist (#89 hibernate, GGUF-scoped). `WARM_GPU` is
implementable today by releasing the `graphs` memory-saver tag and shrinking
the KV pool to a floor. `WARM_HOST` is the hard one — sharded, quantized,
post-processed weights are not a single `.to("cpu")` — and is explicitly
out of scope for #333. Class 1's practical ladder for this project is
`HOT` / `WARM_GPU` / `COLD`.

The `is_generation` process-global (§0.2) means a Class-1 generation engine
and a Class-3 embedding engine cannot be the same process. Under the registry
they are two tenants, which is the point: M1 resolves the constraint by not
fighting it.

### 4.4 Graph strategy

Unchanged, plus the capture lock (§3.6). Capture stays in the boot and
promotion phases.

### 4.5 VRAM regulator and staircase

Per §3.8. Class 1 is the only class where #287 currently applies at all,
since Classes 2 and 3 have no per-rank ratio vector today.

---

## 5. Class 2 — the diffusion lane

### 5.1 Scheduler interface

**Decision.** The "new scheduler class" for Class 2 is an *adapter that wraps
`multimodal_gen`'s existing scheduler*, not a new diffusion scheduler.
Writing one would duplicate a mature, upstream-maintained implementation that
already has dynamic batching with signature-based request merging, an
admission controller with model/resolution/device-memory-aware caps, batch
metrics, request splitting with sequential fallback, and a native
grouped-request path (§0.2).

The adapter:

- **Owns the process group** of `multimodal_gen` scheduler processes for one
  registered engine. `runtime/launch_server.py::launch_server` already spawns
  `num_gpus` processes via `mp.Process(target=run_scheduler_process, ...)`;
  the adapter drives that instead of the CLI.
- **Speaks the lane contract** (§3.5) on one side and `multimodal_gen`'s ZMQ
  `ROUTER` protocol on the other. `DiffGenerator`
  (`runtime/entrypoints/diffusion_generator.py`) is the existing in-process
  client for exactly this socket and is the model to follow.
- **Maps the tick contract onto steps, not requests.**
  `tick_request()` returns the cost of the next denoise step batch, computed
  as `steps_remaining x measured_step_ms`. Because Class 2's cost is known at
  admission (§2.2), this estimate is good, which is what makes the deficit
  broker work at all.
- **Translates admission.** `can_admit()` consults the ledger for the
  activation-peak post at the requested resolution and batch, then delegates
  to `BatchAdmissionController` for the batching decision. If the ledger says
  no, the request is deferred or rejected before it ever reaches
  `multimodal_gen` — which has no VRAM budget model of its own and would
  otherwise happily OOM (§0.2).

**The step-boundary preemption point.** `DenoisingStage._denoise()`
(`runtime/pipelines_core/stages/denoising.py`, main loop around line 1565)
iterates `for step_index, t_host in enumerate(timesteps_cpu)`. That loop head
is the only place a diffusion request can be yielded without losing work, and
it is where a future cooperative yield hook would go. Not in M3.

### 5.2 Memory posts

New posts, none of which exist today because `multimodal_gen` has no budget
model:

| Post | Scaling | Notes |
|---|---|---|
| `dit_weights_bytes` | model | the dominant static term |
| `text_encoder_weights_bytes` | model | offloadable independently (`--text-encoder-cpu-offload`) |
| `image_encoder_weights_bytes` | model | offloadable |
| `vae_weights_bytes` | model | small, kept resident by convention |
| `latent_activation_peak_bytes` | f(resolution, batch, sp_size, dtype) | the term that decides whether a request fits |
| `cfg_branch_bytes` | x2 when CFG is on and CFG-parallel is off | folded into the peak, listed separately because `--cfg-parallel-degree` moves it to another card |
| `bcg_graph_pool_bytes` | number of captured signatures | breakable-CUDA-graph pool |
| `vae_decode_peak_bytes` | f(output resolution, tiling) | tiling (`enable_tiling`, `tile_sample_min_height/width`) trades this against latency |
| `offload_staging_pinned_host_bytes` | host, not device | layerwise offload's pinned buffers; must be accounted against the 108 GB host budget |
| `tenant_ctx_overhead_bytes` | ~0.5-1 GiB per card | the process-isolation cost from DESIGN_121 argument 4 |

`estimate()` for Class 2 is therefore a real function, not a constant: it
must be evaluated per request shape, and the registry's reservation must be
set from the *declared maximum supported shape*, not from the first request.
An engine registered without a declared maximum resolution is rejected at
registration time.

### 5.3 Residency ladder

Class 2 is the best-served class. `MemoryOccupationController` already
implements `WARM_HOST` via `module.to("cpu")`, exposed as
`/release_memory_occupation` and `/resume_memory_occupation`. `WARM_GPU` maps
onto dropping the BCG pool and the VAE tiling buffers while keeping DiT
weights resident. `COLD` is process exit; the weights are already on disk in
their original form, so no hibernate-style parking is needed — the cost is a
normal cold load.

The per-component offload flags (`--dit-cpu-offload`,
`--dit-layerwise-offload`, `--vae-cpu-offload`, `--text-encoder-cpu-offload`)
give a *partial* residency ladder within `HOT`. The registry should treat
these as reservation-shaping inputs — the same engine registered with
`--text-encoder-cpu-offload` has a smaller device reservation and a larger
host reservation — rather than as runtime decisions.

**Idea worth importing, not porting.** `vllm-omni` implements distributed
layerwise offloading in which each DP rank stores only `1/dp_size` of the
weights in host memory and reconstructs full layers at runtime via
AllGather. On a host with 108 GB of RAM and three cards, that is a materially
better `WARM_HOST` than replicating host copies per rank. Registered as a
candidate for M4, not M3.

### 5.4 The seam list

To keep the rebase tax bounded, exactly these files in
`python/sglang/multimodal_gen/` may be modified by this project. Anything
else is read-only and any need to change it is escalated as a design
question.

| File | Change | Milestone |
|---|---|---|
| `runtime/distributed/sp_shard_utils.py` | `build_shard_plan()` gains a capacity-weighted split | M4 |
| `runtime/layers/linear.py` | per-rank shard sizes from a table instead of `// tp_size` | M4 |
| `runtime/layers/vocab_parallel_embedding.py` | same | M4 |
| `runtime/server_args/server_args.py` | accept a per-rank ratio vector and a ledger-supplied budget | M3/M4 |
| `runtime/managers/scheduler.py` | one hook to report step cost and accept an external admission veto | M3 |

Everything the adapter needs beyond that list goes into new files under
`python/sglang/srt/registry/` (§7), outside the vendored subtree.

### 5.5 Graph strategy

Reuse as-is. `runtime/breakable_cuda_graph/runner.py` captures the DiT per
input signature at explicit warmup and replays during serving, falling back
to eager for unseen signatures, with the documented property that serving
never triggers a fresh capture. That property is exactly what the capture
lock (§3.6) needs: the lock is only ever taken during promotion, never during
serving.

The consequence for the registry: **promotion of a Class-2 engine to `HOT`
includes a capture window**, so promotion latency is not just weight
transfer. The registry must publish that as part of the promotion cost so
callers can decide whether to wait.

### 5.6 VRAM regulator and staircase

#287 does not apply — there is no rung vector for diffusion today. #330
applies through the ledger: the activation peak is the term most likely to
blow the corridor, and it is request-shaped, so the corridor check happens at
`can_admit()` and not only at promotion.

---

## 6. Class 3 — single-pass and utility

Two populations (§2.1) sharing one admission contract.

### 6.1 Scheduler interface

**Population A, prefill-only model requests.** These plug into the existing
`srt` scheduler through the type dispatcher. A new
`TokenizedUtilityReqInput` registers a `(Type, handler)` tuple in
`Scheduler.init_request_dispatcher`; the handler builds a `Req` that runs one
`ForwardMode.EXTEND` and never enters `DECODE`. Mechanically this is what
`handle_embedding_request` already does. What is new is that under the
registry these live in their own tenant process, so `ModelConfig.is_generation`
being process-global stops being a ceiling.

**Population B, non-LLM executors.** These are not `srt` at all. A Class-3
executor tenant is a process that owns TensorRT engines and, optionally,
NVDEC/NVENC contexts, and exposes the lane contract over the same control
socket every tenant uses. Its `submit()` takes a work unit — a frame, a
tile, a tensor — not a token sequence.

**Common admission contract.** Rate control with explicit back-pressure. A
Class-3 tenant declares `max_in_flight` work units; when full, `can_admit()`
returns `defer(t)` with a real estimate, and the transport propagates that as
back-pressure to the producer rather than buffering (§8.4). No queue growth,
because a queue in front of a stage whose cost is exactly known is pure added
latency.

### 6.2 Memory posts

| Post | Scaling | Notes |
|---|---|---|
| `trt_engine_device_bytes` | per engine | engine weights, small for the compact SR nets |
| `trt_context_workspace_bytes` | per execution context = per in-flight stream | TensorRT's own scratch |
| `stage_intermediate_bytes` | per in-flight work unit per stage | the dominant term; see the arithmetic in §8.3 |
| `nvdec_surface_pool_bytes` | per decoder session x pool depth | |
| `nvenc_surface_pool_bytes` | per encoder session x pool depth | |
| `pooling_model_weights_bytes` | population A only | |
| `tenant_ctx_overhead_bytes` | ~0.5-1 GiB per card | |

The reservation formula for an executor tenant is fully determined at
configuration time, which is what makes Class 3 the easiest class to
reserve for and the reason it can be pulled forward ahead of the registry
(§10, M2):

```
reserved(C) = ctx_overhead
            + sum over engines: engine_device_bytes
            + sum over stages: streams_in_flight(stage)
                               * peak_intermediate(stage, resolution, dtype)
            + codec surface pools
```

### 6.3 Residency ladder

`HOT` / `WARM` / `COLD`, all cheap. `WARM` releases execution contexts and
surface pools while keeping engines resident — that is most of the bytes for
free, because the workspace and intermediates dominate the engines. `COLD` is
process exit.

The expensive transition is not residency but the **engine build**: `trtexec`
against an ONNX file takes minutes, and TensorRT engines are explicitly
non-portable across systems (the VSGAN prior-art document records upstream's
own statement: "Engines are system specific, don't use across multiple
systems"). Therefore engines are cached on disk keyed by the same identity
hibernate already uses — NVML UUID plus TensorRT version plus ONNX hash plus
shape triplet — reusing `hibernate.py`'s manifest pattern rather than
inventing a second cache-key scheme.

### 6.4 Graph strategy

TensorRT owns the graph. Engines are built with `--useCudaGraph` per the
recipe in `docs/dev/ANALYSE_333_prior_art_vsgan.md` §1. Capture happens
inside TensorRT at build or first inference, and it is still a capture on
that device, so it still takes the arbiter's capture lock (§3.6). In
practice this means an engine build or a first inference after promotion is
an exclusive window on that card, and the registry must publish it as part of
the promotion cost.

### 6.5 VRAM regulator and staircase

#287 does not apply. #330 applies statically: because Class 3's reservation
is exactly computable, a Class-3 tenant is the one tenant whose corridor
compliance can be proven at configuration time rather than measured. That
makes it a good first tenant for validating the ledger.

---

## 7. The engine registry — build specification for #305-M1

This section is the specification the #305-M1 implementation follows. It is
engine-agnostic: the registry knows classes and resource profiles, never
model architectures.

### 7.1 Entities

```
EngineSpec
    engine_id        stable, operator-supplied or derived from model path
    klass            1 | 2 | 3
    adapter          which LaneAdapter implementation
    launch           class-specific launch arguments (opaque to the registry)
    placement        [card_uuid, ...]  or  a placement policy name
    declared_max     class-specific shape ceiling  (Class 2: resolution/batch;
                     Class 3: resolution/streams; Class 1: max_model_len/seqs)
    pinned           bool  -- never demoted automatically
    priority         integer, used by the tick broker's weights and by eviction

EngineInstance
    spec, state (HOT|WARM_GPU|WARM_HOST|COLD), reservations per card,
    pid(s), health, last_used_ts, promotion_cost_ms (measured, EMA)

Slot
    a reservation of bytes on one physical card, held by one instance in one
    state, with a lease and a heartbeat
```

### 7.2 N registered, M hot

N `EngineSpec`s are registered. The number simultaneously `HOT` is **not a
configured constant**: the binding constraint is the ledger invariant of
§3.3. `M` is derived from what fits. A `--registry-max-hot` cap exists as a
blunt safety valve for operators who want to bound process count, not as the
primary mechanism.

The distinction matters. A count-based cap either wastes capacity (cap too
low) or fails at runtime (cap too high on a heterogeneous rig where "one
engine" means 2 GiB on one card and 18 GiB on another). Deriving M from bytes
is the only formulation that survives unlike cards.

### 7.3 Idle default set

`default_hot: [engine_id, ...]` names the set the arbiter returns to when no
request has been in flight for `idle_after_s`. On reaching idle the arbiter
demotes everything outside the set and promotes everything inside it. The set
is validated at registration time against the ledger invariant, so an
unsatisfiable default set is a startup error rather than an idle-time
surprise.

Rationale: the common case on a personal rig is "the LLM should be ready the
moment I ask, the video upscaler should not be holding 8 GiB while I sleep".
That is a declarative statement about the resting state, and it should be
configured as one.

### 7.4 Runtime add and remove

Control plane, on the registry's own HTTP surface (deliberately separate from
any engine's serving surface):

```
GET    /registry                       list specs, instances, slots, ledger per card
POST   /registry/engines               register a spec; validates feasibility;
                                       does NOT boot
DELETE /registry/engines/{id}          demote to COLD, release, deregister
POST   /registry/engines/{id}/state    {"target": "HOT"|"WARM_GPU"|"WARM_HOST"|"COLD"}
POST   /registry/engines/{id}/pin      {"pinned": true|false}
GET    /registry/cards                 per-card: total, reserved, measured, corridor, waste
GET    /registry/plan?spec=...         dry-run: would this fit, and what would be evicted
```

`POST /registry/engines` validating without booting is the important one: it
makes "does this configuration fit on this rig" answerable in milliseconds
and without spending a GPU window, which is the standing rule that a
fixed-cost calculation precedes any GPU booking.

`GET /registry/plan` is the dry-run form of the same check and is what the
existing offline planner package (`python/sglang/srt/planner/`, with
`capacity.py`, `feasibility.py`, `placement.py`, `wizard*.py`, `webui.py`)
should call. The planner already constructs the same `PerfCostModel` the
server constructs at parse time; extending it to multi-tenant plans is a
natural continuation rather than a new tool.

### 7.5 Admission and promotion

```
request for engine E arrives
  |
  +- E is HOT                  -> submit
  |
  +- E is WARM_* or COLD       -> promotion needed
        |
        +- ledger has room                  -> promote, then submit
        |
        +- ledger is short by K bytes
              |
              +- find eviction candidates on the needed cards:
              |     not pinned, not in default_hot (unless nothing else),
              |     lowest priority first, then least recently used
              |
              +- projected wait = promotion_cost_ms(E)
              |                 + sum(demotion_cost_ms(victims))
              |
              +- caller's max_promotion_wait_ms exceeded  -> reject with the
              |     projected wait and the eviction that would have happened
              |
              +- otherwise -> demote victims, promote E, submit
```

Three properties this gets right:

- **Rejection is informative.** The caller is told the projected wait and the
  eviction, not just "busy". A UI can then offer "wait" or "use the other
  model".
- **Pinning is absolute.** A pinned engine is never demoted automatically.
  This is what makes "the LLM is always up" expressible.
- **No thrash without a name.** If promotions and demotions of the same pair
  exceed a rate threshold, the arbiter logs a named thrash event with both
  engine ids rather than silently degrading.

### 7.6 Engine-agnosticism, concretely

The registry contains no reference to any model architecture, to KV caches,
to latents, or to TensorRT. It calls `adapter.estimate(spec)` and gets bytes
per card per post; it calls `adapter.promote/demote` and gets state changes;
it calls `adapter.measured()` and updates the ledger. Adding a fourth class
later means writing a fourth adapter, not touching the registry.

The three adapters at M1 are:

| Adapter | Wraps | Process model |
|---|---|---|
| `Class1SrtAdapter` | `srt` `Scheduler` | in-process (the registry runs in the srt process) or a child process |
| `Class2DiffusionAdapter` | `multimodal_gen` scheduler processes | child processes, ZMQ |
| `Class3UtilityAdapter` | pooling models and/or executor tenants | child processes |

### 7.7 What the registry does not do

- It does not route between replicas of the same model. That is
  `sgl-model-gateway`'s job and it already does it well. The registry sits
  *below* the gateway: the gateway picks a worker, the registry decides
  whether that worker's engine is resident.
- It does not migrate a running request between engines.
- It does not do cross-card load balancing of a single engine. Placement is a
  spec property, decided by the planner.

---

## 8. Class-3 first build-out: the video-enhance stream server

This is the concrete, pullable-forward deliverable. It is Class 3 population
B, it is process-isolated, its reservation is exactly computable, and it
needs nothing from the registry except a static reservation — so it can be
built before M1 lands (§10).

### 8.1 The chain

```
HTTP request (source URL or upload, target spec)
   |
 decode          NVDEC, output stays in VRAM
   |
 [colour]        YUV -> RGB, on GPU
   |
 SR              realesr-general-wdn-x4v3, TensorRT, x4
   |
 [resize]        Lanczos-3 on GPU, 8K -> target
   |
 RIFE            frame interpolation, TensorRT, engine built at post-resize size
   |
 [colour]        RGB -> YUV
   |
 encode          NVENC
   |
chunked HTTP response
```

Bracketed stages are conditional. Every arrow between `decode` and `encode`
is a device-to-device pointer hand-off with no host round-trip; that is the
single most important structural property of the chain and it is what
`efschu/vs-mlrt`'s "multi-engine pipeline (GPU-persistent intermediates)"
already implements.

**Why the resize stage is not optional.** `realesr-general-wdn-x4v3` is a
fixed x4 network. A 1080p source becomes 7680x4320. Nobody wants 8K out, and
RIFE at 8K does not fit. The resize is what makes the chain terminate at a
sane resolution, and putting it *after* SR and *before* RIFE is what lets
RIFE's TensorRT engine be built at the target size instead of at 8K.

**Why not pre-resize before SR.** The prior-art document (§6.2 of
`ANALYSE_333_prior_art_vsgan.md`) establishes that the RIFE plugin's
`trt_max_shape` default of 1920x1080 — not any architectural limit — is what
makes 4K fail out of the box, and that the maintainer's own recommended
mitigation for large input is RIFE's native `scale` parameter (valid values
0.25/0.5/1.0/2.0/4.0, with `scale=0.5` recommended for 4K), not a
pre-resize. Pre-resizing before SR would also throw away the detail SR exists
to recover. Order of preference, therefore: **raise `trt_max_shape` and
rebuild the RIFE engine at the target resolution; tune `scale`; pre-resize
only if VRAM genuinely cannot fit a whole frame in one pass** — and note that
the plugin has no tiling, so that ceiling is real, just VRAM-bound rather
than code-bound.

### 8.2 Regimes A and B

**Regime A — one card holds the whole chain.** Decode, SR, resize, RIFE, and
encode all on one physical GPU. Zero cross-GPU transfers. The chain's only
host interaction is the HTTP body in and out.

**Regime B — stages split across cards.** For example decode plus SR on the
5090, RIFE plus encode on a 3080. Every stage boundary that crosses a card is
a host-staged round trip, because this rig has no CUDA peer-to-peer (GeForce,
PHB topology). The transfer cost is therefore D2H plus H2D at the slower of
the two links.

**The decision rule is arithmetic, not preference.** Regime B is worth it
only when the offloaded stage's own time exceeds the transfer time it adds:

```
benefit(B) = t_stage_on_A - t_stage_on_B
cost(B)    = bytes_at_boundary / bw_D2H(A) + bytes_at_boundary / bw_H2D(B)
```

With frame sizes from §8.3 and measured bandwidths of roughly 6.5 GB/s for
the x4-slot 3080 and 13-14 GB/s for the other two, a single 8K fp16 frame
boundary (189.8 MiB) costs about 15 ms per hop on a 13.5 GB/s link and about
31 ms per hop on the x4 link — so a round trip between two fast cards is
roughly 30 ms and one involving the x4 card roughly 45 ms, per frame, of pure
transfer. That is large enough that **Regime B is only ever plausible at
boundaries carrying small frames** — after the resize, or in NV12 rather than
RGB float, where the same boundary is 11.9 MiB and the round trip drops to
single-digit milliseconds. Placing the split before the resize is
arithmetically excluded. This is the kind of conclusion the transfer
microbench (§8.6) must confirm before any Regime-B code is written.

**Codec engine facts that constrain placement.** Subject to the confidence
flags in §0.3: the RTX 5090 exposes 3 NVENC and 2 NVDEC engines; each RTX
3080 exposes 1 and 1. Decode and encode capacity is therefore strongly
asymmetric, and the natural Regime-B split for *multiple concurrent streams*
is not "one chain across cards" but "one whole chain per card, streams
distributed across cards" — which is Regime A replicated, and which the
prior-art document confirms is the granularity VSGAN also chose (frame-level
parallelism, static modulo-N round robin). Our delta over VSGAN is that the
distribution is capacity-weighted and derived from measured stage rates
rather than a hand-edited `cycle=N`.

### 8.3 The arithmetic — fixed costs before any GPU window

Frame byte sizes, exact:

| Resolution | NV12 8-bit | RGB fp16 | RGB fp32 |
|---|---|---|---|
| 960x540 | 0.74 MiB | 2.97 MiB | 5.93 MiB |
| 1280x720 | 1.32 MiB | 5.27 MiB | 10.55 MiB |
| 1920x1080 | 2.97 MiB | 11.87 MiB | 23.73 MiB |
| 3840x2160 | 11.87 MiB | 47.46 MiB | 94.92 MiB |
| 7680x4320 | 47.46 MiB | 189.84 MiB | 379.69 MiB |

`realesr-general-wdn-x4v3` is SRVGGNetCompact with 64 feature channels and 32
convolutions, operating at *input* resolution throughout and doing the x4
upscale only at the final pixel-shuffle. The activation tensor is therefore
`W x H x 64 x sizeof(dtype)`, and two must be live for the convolution
ping-pong:

| SR input | one activation, fp16 | two live | x4 output, fp16 | per-stream subtotal |
|---|---|---|---|---|
| 960x540 | 63.3 MiB | 126.6 MiB | 47.5 MiB | ~178 MiB |
| 1280x720 | 112.5 MiB | 225.0 MiB | 84.4 MiB | ~315 MiB |
| 1920x1080 | 253.1 MiB | 506.3 MiB | 189.8 MiB | ~708 MiB |

Adding TensorRT's own workspace and reformat buffers, budget **1.0 GiB per
in-flight 1080p frame** and **0.25 GiB per in-flight 540p frame** for the SR
stage alone. With `num_streams=2`, 1080p SR reserves 2 GiB before RIFE,
codecs, or context overhead are counted.

**What this settles.** On a 5090 shared with a HOT LLM under a
`--rank-gpu-memory-mib` budget, a 1080p-source chain at two in-flight frames
is affordable and a 4K-source chain at the same depth is not. The first
build-out therefore targets 1080p source with a 4K or 1080p target, and 4K
source is a later, separately budgeted configuration.

**RIFE's footprint is unknown and must be probed, not guessed.** RIFE 4.x
holds optical-flow pyramids at several scales and has no tiling. Its
per-frame-pair footprint at the post-resize resolution is registered as
measurement post P4 (§8.6) and no number is asserted here.

### 8.4 Transport: chunked HTTP and back-pressure

**Response.** `Transfer-Encoding: chunked`, one chunk per encoded muxed
segment, `Content-Type` per the negotiated container. The client sees output
while the tail of the source is still decoding. This is the whole point of a
stream server as opposed to a batch job, and it is exactly what the prior art
does not have — VSGAN is `vspipe | ffmpeg`, single-shot, no server, no
session (`ANALYSE_333_prior_art_vsgan.md` §4).

**Back-pressure, the part that is easy to get wrong.** The chain has a
producer (decode) that is fast and a consumer (the HTTP client) that may be
slow. Without back-pressure, VRAM fills with completed frames waiting to be
written and the tenant blows its reservation — the failure mode is an OOM
caused by a slow network client, which is the worst kind because it is
untraceable from the GPU side.

Rules:

1. **One bounded ring per stage boundary**, depth declared at configuration
   and counted in the reservation (§6.2). No unbounded queue anywhere in the
   chain.
2. **Back-pressure propagates upstream to the decoder**, which stops pulling
   from the source. The decoder, not the network buffer, is where the stream
   stalls.
3. **The socket write is the throttle.** When the ASGI send coroutine blocks
   because the client's TCP window is full, that block must reach the decoder
   within one ring depth. This requires the encode stage to await the socket
   write rather than fire-and-forget into an application buffer.
4. **Explicit policy on overload**, configured per request, never implicit:
   `stall` (default, correct for file-to-file enhancement) or
   `drop_frames` (correct for live sources, and only meaningful when the
   source is live). Silent dropping is prohibited; a dropped frame increments
   a counter that appears in the response trailer.
5. **In-flight cap is the reservation.** `max_in_flight` is derived from the
   §8.3 arithmetic and the tenant's reserved bytes, not configured
   independently, so it is impossible to configure a depth that does not fit.

### 8.5 Session and request model

- `POST /v1/video/enhance` — body carries source (URL or upload), target
  resolution, target frame rate multiplier, model selection per stage, and
  overload policy. Response is the chunked stream.
- `GET /v1/video/enhance/{id}` — progress: frames decoded, enhanced,
  encoded, current per-stage ms/frame, ring occupancies, dropped count.
- `DELETE /v1/video/enhance/{id}` — cancel; must release ring buffers and
  execution contexts synchronously so the reservation is honest.
- `GET /v1/video/engines` — which engines are built and cached for this
  card, which would need a build, and the estimated build time.

That last endpoint matters operationally: an engine build is minutes long and
exclusive on the card (§6.4), so a request that implies one must say so
before it starts rather than appearing to hang.

### 8.6 Named measurement posts

These are the measurements this build-out owes. Each is a named post so it
can be tracked, and each follows the standing benchmark rules: establish the
noise floor with an A-versus-A run first, interleave arms, fix the clock,
report nothing below the detection threshold, and report per-stage ms/frame
rather than aggregate frames per second.

| Post | What is measured | Why it decides something |
|---|---|---|
| **P1 — stage rate probe** | ms/frame per stage in isolation at 540p, 720p, 1080p input, at fp32 and fp16, on 5090 and on 3080 | the input to capacity-weighted stream distribution; replaces VSGAN's hand-picked `cycle=N` |
| **P2 — transfer microbench** | host-staged D2H+H2D round-trip at 0.74 / 2.97 / 11.87 / 47.46 / 189.84 MiB, on each card, against the known 6.5 and 13-14 GB/s links | the hop price in the Regime-A/B rule of §8.2; without it Regime B is speculation |
| **P3 — SR footprint validation** | measured peak device bytes per in-flight frame against the §8.3 predictions | validates the reservation formula; a large miss invalidates the ledger's Class-3 estimator |
| **P4 — RIFE footprint and rate** | peak bytes and ms/frame-pair at the post-resize resolution, at `scale` in {1.0, 0.5}, with `trt_max_shape` built to match | the only unknown large term in the chain |
| **P5 — codec ceiling** | NVDEC decode and NVENC encode ms/frame at 1080p and 4K per card; concurrent session count at which throughput degrades | settles the flagged session-cap uncertainty (§0.3) empirically rather than from documentation |
| **P6 — co-tenancy jitter** | AR decode ms/round on a card with and without the enhance chain running, at matched in-flight depth | the honest cost of co-tenancy; feeds the tick broker's weights |
| **P7 — 8-bit matrix** | see §8.7 | |

### 8.7 The 8-bit test matrix — compute separated from I/O

Two entirely different questions get conflated under "8-bit", and separating
them is the point of this matrix.

**Axis 1, compute precision.** The dtype the engine computes in. The prior-art
document establishes that INT8 post-training quantization of
super-resolution networks is a known, non-trivial quality problem — SR has no
softmax or argmax to absorb activation error, and the literature exists
specifically because naive PTQ degrades output visibly. VSGAN's entire
328-asset catalog is fp16/fp32/bf16 with zero int8 assets and no `--int8` in
any recipe. Verdict: **int8 compute is a research task requiring its own
calibration set and quality validation, and is deferred** under the standing
rule that lossy features come last.

**Axis 2, transport precision.** The dtype tensors are carried in *between*
stages and across the PCIe boundary. This is a different question with a
different answer: **the source video is 8-bit to begin with.** Carrying a
decoded 8-bit frame across PCIe as RGB fp32 wastes a factor of four of
bandwidth transporting precision that was never in the signal. Converting to
8-bit for transport and back is not a quality decision for the *input* side of
the chain at all. On the *output* side of a stage it is, because SR output
carries information the input did not.

The matrix therefore crosses three independent factors and reports two
different kinds of result:

| Factor | Levels |
|---|---|
| compute dtype | fp32, fp16 (bf16 where the card supports it), int8 (deferred arm, run only if the others are settled) |
| transport dtype at each boundary | fp32, fp16, uint8 / NV12 |
| boundary | decode->SR, SR->resize, resize->RIFE, RIFE->encode |

Reported per cell: **ms/frame** (the I/O question) and **quality against an
fp32 end-to-end reference** (the compute question), the latter as PSNR and
SSIM plus at least one perceptual metric, on a fixed clip set that includes
both high-detail and flat content.

The expected shape of the answer, stated in advance so the measurement can
falsify it: transport at 8-bit on the decode->SR boundary is free and worth
taking; transport at fp16 elsewhere is nearly free; int8 *compute* costs
visible quality and is not worth taking without calibration work. If the
measurement contradicts any of these, the measurement wins.

### 8.8 Deployment note inherited from prior art

VSGAN's image split into `latest`, `latest_no_avx512`, and `minimal` variants
exists because an AVX-512-less host produced "Illegal instruction (core
dumped)" from the full plugin set. Any container we build for this tenant
does host CPU feature detection before selecting a base image. Cheap lesson,
taken as-is.

---

## 9. Prior art and reuse verdicts

Reuse order: **dependency > port > own build.** Each verdict below states
which, and why the cheaper option was not taken.

### 9.1 Class 1

Nothing new. The AR path is the fork's existing work plus upstream sglang.

**Embeddings and rerankers — the gap list.** The starting assumption that
this area is largely missing is wrong; most of it is present and mature:

| Capability | State in the fork | File |
|---|---|---|
| `/v1/embeddings` incl. batch input and multimodal inputs | present | `python/sglang/srt/entrypoints/openai/serving_embedding.py` |
| Matryoshka dimension truncation | present, threaded end to end | `configs/model_config.py` (`is_matryoshka`, `matryoshka_dimensions`), `layers/pooler.py` |
| Cross-encoder reranking | present | `entrypoints/openai/serving_rerank.py` (dispatches `vl_decoder`/`text_decoder`/`cross_encoder`) |
| Sequence classification | present | `entrypoints/openai/serving_classify.py` |
| Scoring | present | `entrypoints/openai/serving_score.py` |
| Multi-item / delimiter-position pooling | present | `layers/pooler.py::pool_at_delimiter_positions` |
| **Mean/average and max pooling** | **missing** — `PoolingType` has only `LAST` and `CLS`; `pool_hidden_states` raises on anything else, despite a docstring in the same file claiming LAST/AVERAGE/MAX | `layers/pooler.py` |
| **Multi-vector / ColBERT late interaction** | **missing** — no hits anywhere in `python/sglang/srt/` | — |
| **Async batch API (`/v1/batches`)** | **missing** — a `BatchRequest`-shaped struct exists in `protocol.py` but is not wired to a handler | — |
| **Generation and embedding in one server** | **structurally blocked** — `ModelConfig.is_generation` is process-global | `configs/model_config.py` |

Verdict: M0 closes the pooling gap, which is genuinely XS and blocks common
BGE/E5-family models that expect mean pooling. ColBERT and `/v1/batches` are
real but discretionary. The fourth item is not an M0 item at all — it is
exactly what M1's registry resolves, by making them two tenants.

### 9.2 Class 2

**`sglang.multimodal_gen` — dependency.** It is already vendored in the tree,
it is upstream-maintained, it covers ~22 pipeline families, and it already
has the dynamic batching and the breakable-CUDA-graph integration. Writing a
diffusion runtime next to it would be indefensible. The cost of this choice
is the rebase tax, and it is bounded by the seam list in §5.4.

**HuggingFace `diffusers` — not a direct dependency.** Its Modular Diffusers
architecture is genuinely clean and separable — schedulers, VAE, transformer,
and pipeline glue are independently loadable and swappable — and if we were
starting from nothing, it would be the right base. We are not starting from
nothing: `multimodal_gen` already occupies that layer. Taking diffusers as a
second direct dependency would create a second model-loading path and a
second source of scheduler math for the same models. It stays an *indirect,
optional* dependency, reached only through `multimodal_gen`'s own
`diffusers_generic` pipeline config, which is the correct escape hatch for
models without a native pipeline.

Note also that diffusers ships no continuous batching and no scheduler — its
own server documentation is a thin HTTP wrapper. Its offload primitives
(`enable_model_cpu_offload`, `enable_group_offload` from PR #10503,
sequential offload) are the conceptual ancestors of both `multimodal_gen`'s
per-component flags and `vllm-omni`'s three tiers, and are worth reading for
the §5.3 ladder, but there is nothing to import.

**xDiT — do not port.** `multimodal_gen/runtime/distributed/` is already
adapted from FastVideo and xDiT and implements Ulysses, ring, CFG-parallel,
data-parallel, and a dedicated VAE-decode parallel group. Porting xDiT again
would duplicate code the fork already carries. xDiT upstream also appears not
to support uneven splits, so it would not solve the one thing we actually
need.

**The uneven split — our own work, at a named seam.** The genuine Class-2
delta for this fork is capacity-weighted sequence and tensor parallel, and
its location is precisely known:
`runtime/distributed/sp_shard_utils.py::build_shard_plan()` plus the two
layer files in §5.4. This mirrors what `--rank-tp-ratio` already does for AR
models and is the same argument: on unlike cards, an equal split makes every
card as slow as the slowest. Note the existing scheme is equal-with-tail-
padding, so the change is from "one `local_len` for everyone" to "a per-rank
size table"; the attention metadata already handles a ragged tail
(`shard_like`, `tail_attn_meta`), which is a helpful precedent for ragged
shards generally.

**`vllm-omni` — read, do not port.** Its dual `OmniARScheduler` /
`OmniGenerationScheduler` is the closest existing thing to what §3 designs,
and its own RFC #5279 reports duplicated lifecycle code and unused paths as
the problem. That is direct evidence for the three-schedulers-one-arbiter
decision. Its distributed layerwise offloading (each DP rank holds
`1/dp_size` of host weights, reconstructed by AllGather) is a good idea worth
importing for `WARM_HOST` on a RAM-limited host (§5.3), as an idea, at M4.

**TensorRT-LLM VisualGen — reference.** NVIDIA's own diffusion serving keeps
a separate pipeline abstraction from the LLM path, with its own denoising
loop and component loading, deliberately not sharing the LLM scheduler.
Third independent confirmation of the sibling-lane shape. Labelled Beta by
NVIDIA; nothing to import.

**Sangam (arXiv 2607.04206) — one idea imported.** Deficit token-budget
scheduling to colocate diffusion decoding with AR prefill admission. Wrong
domain (text diffusion), right shape. §3.7 takes the deficit-budget idea and
nothing else.

**ComfyUI — one idea noted.** Its execution model is a DAG with topological
scheduling and memoized, content-addressed caching, so an unchanged graph
re-executes nothing and a changed tail re-executes only the suffix. Not a
throughput answer, but the right pattern for the Class-3 multi-stage chain if
per-stage result caching ever becomes worthwhile. Not in scope for M2.

**`nunchaku` / SVDQuant — registered, deferred.** 4-bit weight-and-activation
PTQ for diffusion transformers, with an official diffusers integration and
claimed 3.5x memory reduction on 4090/5090-class hardware. Directly relevant
to fitting a diffusion tenant next to a HOT LLM on a 32 GB card. Deferred
under the lossy-features-last rule; registered so it is not rediscovered.

### 9.3 Class 3

**`efschu/vs-mlrt` — port by extraction.** There is no standalone build
target today, and the fork's `standalone_mode` means something else entirely
(§0.3). But the extraction is small and well-bounded:

| Piece | Action |
|---|---|
| `vstrt/trt_utils.h` — `InferenceInstance`, `Logger : ILogger`, profile-shape introspection, `getInstance()` | reuse as-is; it has no VapourSynth dependency |
| `vstrt/cuda_helper.h`, `cuda_utils.h` | reuse as-is |
| `vstrt/inference_helper.h` — the tiling and inference loop | rewrite one call: `vs_bitblt` becomes `cudaMemcpy2D`. That is the only VapourSynth coupling in the inference path |
| `vstrt/vs_tensorrt.cpp` — the plugin entry point | replace entirely with our own driver: engine path in, device buffer in, device buffer out |
| the fork's `lanczos3_kernel.cu` | reuse as-is — this is the §8.1 resize stage |
| the fork's multi-engine pipeline (GPU-persistent intermediates) | reuse — this is the §8.1 no-host-round-trip property |
| the fork's TensorRT-RTX support (`TRT_MAJOR_RTX`) | reuse |

Extraction rather than dependency because there is nothing to depend on;
extraction rather than own build because `trt_utils.h` is already the thing we
would write, and the fork's Lanczos-3 kernel and multi-engine pipeline are
work already done.

**`AmusementClub/vs-mlrt` upstream — the same extraction target, one layer
up.** Our fork is at the same layer as upstream, not a fork of the scripting
layer above it. No conflict.

**`styler00dollar/VSGAN-tensorrt-docker` — not a dependency; take four
things.** It is a set of copy-edited example scripts, not a library. From it
we take: the model catalog as a sourcing map, the `trtexec` engine-build
recipe (flags, tactic sources, shape-triplet convention), the RGBH/RGBS
fp16/fp32 colorspace convention and the general
decode->resize->infer->resize->encode chain shape, and the AVX-512 container
lesson. Its multi-GPU mechanism — a static equal-share modulo-N frame
round-robin with a hand-edited `cycle=N` — is explicitly *not* taken; §8.2's
capacity-weighted distribution is the delta.

**`HolyWu/vs-rife` — dependency at the model level, port at the execution
level.** The model version enum (4.0 through 4.26 with lite and heavy
variants), the modulo-32/64/128 padding rules, and the `scale` parameter
semantics all come from there and are worth mirroring exactly. The execution
goes through our extracted executor, not through the VapourSynth plugin.

**NVDEC/NVENC access — dependency.** `PyNvVideoCodec` (NVIDIA's official
Python binding) exposes device-memory output and DLPack interop, which is
exactly the "decode into VRAM, hand a tensor to TensorRT" seam §8.1 needs.
The alternative, `ffmpeg -hwaccel cuda -hwaccel_output_format cuda`, is the
fallback for container and codec coverage `PyNvVideoCodec` lacks. Both are
dependencies; neither is ported.

---

## 10. Staged build plan

Effort classes: XS (< 1 day), S (days), M (1-2 weeks), L (multiple weeks).
Every stage's gate includes, unconditionally, an **unchanged-default
regression**: the reference TP=3 launch boots and serves identically with the
new feature disabled.

Order note: the sequence below is M0 through M5 as scoped. **M2 is pullable
forward ahead of M1** and should be, if the video-enhance server is the
nearer user need — it is process-isolated, its reservation is exactly
computable at configuration time (§6.2), and it needs only a static budget,
not the registry. Doing M2 before M1 also has a design benefit: it validates
the ledger's reservation formula against the one class where the formula is
checkable by arithmetic, before the registry depends on it.

---

### M0 — Embedding and reranker gap closure

**Effort: XS to S.**

Scope: add `AVERAGE`/`MEAN` and `MAX` to `PoolingType` and implement them in
`python/sglang/srt/layers/pooler.py::pool_hidden_states`; fix the docstring
in that file that already claims they exist. Optionally: ColBERT-style
multi-vector output (S-M), `/v1/batches` (S).

**Feasibility gate**: none needed — CPU-only change.

**Acceptance gate**: pooled output matches a `sentence-transformers`
reference for a small mean-pooled model (BGE or E5 family) on CPU, to
floating-point tolerance. Existing LAST/CLS paths byte-identical before and
after. `ruff` and `codespell` clean.

**What it cannot do**: nothing about co-residency. A mean-pooled embedding
model still requires its own server process, because `is_generation` is
process-global. M0 makes more models *work*; it does not make them *share*.

---

### M1 — Engine registry, engine-agnostic

**Effort: M.**

Scope: §7 in full — `EngineSpec`/`EngineInstance`/`Slot`, the cross-process
VRAM ledger of §3.3 with NVML-UUID-keyed reservation files and leases, the
corridor guard, the capture lock, the control-plane endpoints, and the three
adapters (`Class1SrtAdapter`, `Class3UtilityAdapter` for pooling models,
and a stub `Class2DiffusionAdapter` that only estimates and does not yet
launch). New code under `python/sglang/srt/registry/`; no changes inside
`multimodal_gen`.

**Feasibility gate**: a fixed-cost table for the target demonstration — one
Qwen-class AR engine plus one embedding engine on the reference rig — showing
`sum(reserved) + 400 MiB corridor <= nvml_total` on every card, computed
before any GPU window is booked.

**Acceptance gate**:
1. Two engines registered, one AR generation and one embedding model; both
   reach `HOT`; both serve correctly; generation and embedding requests are
   served concurrently from one logical endpoint — the thing that is
   structurally impossible today.
2. A full promote/demote cycle observed on each, with `promotion_cost_ms`
   measured and reported.
3. The corridor invariant holds throughout, verified by an NVML sampler
   running independently of the registry.
4. `POST /registry/engines` for an infeasible spec is rejected at
   registration, without booting anything.
5. Registry disabled: the reference TP=3 launch is unchanged.

**What it cannot do**: no diffusion tenant yet (the Class-2 adapter only
estimates). No `WARM_HOST` for Class 1 — its ladder is `HOT`/`WARM_GPU`/
`COLD` only. No mid-request preemption of any kind: a promotion waits for
in-flight work to drain. Promotion latency for a `COLD` engine is a full
model load unless the engine is GGUF and hibernate applies.

---

### M2 — Class-3 video-enhance stream server

**Effort: M to L.**

Scope: §8 in full at Regime A on a single card — the extracted vs-mlrt
executor (§9.3), NVDEC decode to VRAM, the SR/resize/RIFE chain with
GPU-persistent intermediates, chunked HTTP with real back-pressure, the
engine disk cache keyed as in §6.3, and probes P1 through P5 and P7.

**Feasibility gate**: the §8.3 arithmetic, computed for the exact target
configuration, showing the chain's reservation plus the co-tenant LLM's
budget plus the corridor fits the chosen card. If it does not fit, reduce
source resolution or in-flight depth before booking a GPU window — do not
book one to find out.

**Acceptance gate**:
1. A 1080p source runs end to end to a 4K or 1080p target, output visually
   correct and byte-stable across two identical runs.
2. Sustained per-stage ms/frame recorded (P1), with the A-versus-A noise
   floor established first.
3. Back-pressure proven: an artificially slowed HTTP client stalls the
   decoder within one ring depth, and device memory stays flat. This is the
   gate that matters most — it is the failure mode that would otherwise
   appear only in production.
4. Measured peak device bytes within a declared tolerance of the §8.3
   prediction (P3). A large miss is a design failure of the Class-3
   estimator, not a tuning issue.
5. A co-tenant LLM on the same card is unaffected beyond its declared budget;
   AR decode ms/round jitter measured (P6) and reported, not assumed away.
6. P2 transfer microbench run and recorded, so the Regime-B decision has
   data behind it before any Regime-B code exists.

**What it cannot do**: single card only — no Regime B, no multi-card stage
split, no capacity-weighted stream distribution (P1 produces its *input*, not
the mechanism). No int8 compute (deferred by §8.7). No per-request model
selection beyond a fixed configured chain. No 4K source at the depth that
1080p source supports. If M2 runs before M1, its reservation is a static
configured budget rather than a registry slot, and co-tenancy safety rests on
the operator getting that number right.

---

### M3 — Diffusion lane core, images

**Effort: M to L.**

Scope: promote the Class-2 adapter from estimate-only to launching — process
group management, ZMQ client, the two `multimodal_gen` seam changes from
§5.4 (step-cost reporting, external admission veto), ledger integration for
the §5.2 posts, capture-lock participation, and the residency ladder using
the existing `MemoryOccupationController`.

**Feasibility gate**: fixed-cost table for one image model (FLUX-class or
Qwen-Image, at a declared maximum resolution) co-resident with a HOT AR
engine on the 5090, including both tenants' context overhead, showing the
corridor holds.

**Acceptance gate**:
1. A diffusion engine registered as a Class-2 tenant, promoted to `HOT`,
   generating images correctly while an AR engine on the same card serves
   tokens.
2. The corridor invariant holds across a full generate-while-decoding run,
   verified by an independent NVML sampler.
3. Capture-lock arbitration proven: a Class-2 promotion's capture window
   quiesces the AR tenant and neither crashes nor hangs. Include the negative
   test — a capture requested during steady state is refused and logged.
4. `WARM_HOST` demotion and re-promotion of the diffusion tenant works via
   the existing release/resume endpoints, with `promotion_cost_ms` measured.
5. AR decode ms/round with and without a concurrent diffusion tenant, at
   matched conditions (P6 extended). Report the jitter honestly; do not gate
   on it being small.

**What it cannot do**: images only, no video. Equal sequence-parallel splits
only — a diffusion tenant spanning unlike cards is as slow as its slowest
card. No cross-class preemption: an admitted diffusion request runs its full
step count. No cooperative yield at the step boundary (the hook location is
identified in §5.1 but not built). Fine-grained SM sharing is unavailable on
this rig regardless (no MPS).

---

### M4 — Video and uneven sequence parallel

**Effort: L.**

Scope: video pipelines (Wan-class) as Class-2 tenants, plus the fork's real
Class-2 delta — capacity-weighted `build_shard_plan()` and per-rank shard
tables in the two layer files (§5.4). Candidate: import `vllm-omni`'s
distributed layerwise offload idea for `WARM_HOST` (§5.3).

**Feasibility gate**: video activation-peak arithmetic per frame count and
resolution, against the 5090 plus two 3080s under the capacity-weighted
split. Video latents are large; this gate is likely to force a resolution or
frame-count ceiling, and finding that on paper is the point.

**Acceptance gate**:
1. Output equivalence: the capacity-weighted split produces output matching
   the equal-split reference at `sp_size=1` to a declared tolerance, on the
   same seed. Uneven splits must not change results, only who does the work.
2. A TP or SP=3 diffusion run across the 5090 and both 3080s, with per-rank
   ms/round reported — per the standing rule that the slowest rank sets the
   clock and the capacity split loads the weakest card most, so per-rank
   timing is mandatory, not optional.
3. Measured speedup of the weighted split over the equal split on the
   heterogeneous rig, above the noise floor.
4. `sp_shard_utils` unit tests extended to uneven cases; the existing
   equal-split tests still pass unchanged.

**What it cannot do**: no PipeFusion. No cross-rig sequence parallel. Video
lengths remain bounded by the activation peak, and there is no tiling or
spill escape hatch in the diffusion path. The uneven split helps throughput;
it does not reduce peak memory on any single card.

---

### M5 — Speech to speech

**Effort: L, and gated on model and scope decisions not yet made.**

Scope: streaming speech-in/speech-out. Note the class assignment from §2.3:
a streaming omni model producing audio tokens autoregressively is **Class 1**,
not a new class — it is an AR engine with an audio detokenizer. Its encoder,
served standalone, is Class 3. The existing encode-disaggregation role
(`python/sglang/srt/disaggregation/encode_server.py`) is the precedent for
splitting them. Upstream carries a separate `sglang-omni` repository whose
architecture was not examined for this document; that examination is M5's
first task and may change this scoping.

**Feasibility gate**: not yet formulable — depends on the model choice and on
whether `sglang-omni` is a dependency or a reference. M5 does not start
without its own prior-art pass.

**Acceptance gate**: to be specified at M5 planning.

**What it cannot do**: unknown until scoped. Stated as unknown rather than
guessed.

---

## 11. Open questions

Recorded so they are not rediscovered.

1. **Does the deficit tick broker actually bound cross-class jitter on a rig
   without MPS?** §3.7 argues it bounds in-flight work; whether that
   translates into bounded AR decode jitter is P6's question, and a negative
   answer would push toward time-slicing whole cards between classes rather
   than sharing them.
2. **Is `WARM_HOST` worth building for Class 1?** Sharded, quantized,
   post-processed weights make it expensive. If `COLD` via a widened
   hibernate is fast enough, `WARM_HOST` may never be worth it for Class 1.
3. **Does the capture lock need to be cross-process-robust against a tenant
   that dies mid-capture?** Probably yes; the lease mechanism in §3.3 covers
   reservations but a lock held by a dead process is a separate failure.
4. **How far can the `multimodal_gen` seam list (§5.4) hold?** If upstream
   restructures `sp_shard_utils.py`, the uneven-split patch moves. Worth a
   periodic check against upstream rather than a surprise at rebase time.
5. **Should the registry live in the `srt` process or its own?** §7.6 assumes
   the Class-1 adapter can be in-process. A separate registry process is
   cleaner but adds a hop to every admission decision.
6. **Regime B for Class 3: is it ever worth it on this rig?** §8.2's
   arithmetic suggests only at small-frame boundaries. P2 settles it. If the
   answer is no, Regime B should be cut rather than built and left unused.

---

## 12. References

Internal:
- `docs/dev/ANALYSE_333_prior_art_vsgan.md` — VSGAN prior art, the pinned
  Class-3 artifacts, the int8-SR literature finding
- `docs/DESIGN_121_dual_group_runtime.md` — #274, the in-process second lane
  and its five arguments
- `docs/dev/ANALYSE_321_nvfp4_asymmetry.md` §8 — #287 staircase coupling
- `docs/dev/INTEGRATION_R3_VALIDATION.md` — #274 slice validation, the
  memory-saver tag-scope gap
- `FEATURES_VS_UPSTREAM.md` — fork deltas, reference hardware, status legend
- `docs/wie_starte_ich/02_regeln_und_fallen.md` — the 400 MiB corridor rule

External, with the confidence notes of §0.3 applying:
- sglang diffusion continuous batching: `sgl-project/sglang` PR #28690
- sglang disaggregated diffusion: `sgl-project/sglang` PR #24200
- vLLM dLLM plugin RFC: `vllm-project/vllm` issue #36155
- vllm-omni scheduler refactor RFC: `vllm-project/vllm-omni` issue #5279
- diffusers group offloading: `huggingface/diffusers` PR #10503
- Sangam, deficit-budget colocation of diffusion and AR: arXiv 2607.04206
- SVDQuant / nunchaku: arXiv 2411.05007
- NVIDIA video encode/decode support matrix (NVENC/NVDEC engine counts,
  session cap)
