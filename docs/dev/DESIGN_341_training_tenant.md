# DESIGN #341 — Training/finetune as idle tenant

Status: design registered, implementation queued (entry step: LoRA-LLM, see
multi-day plan). This file is the single persistent record of the #341
analysis and decisions; the chat history is not authoritative.

## Goal

The rig trains or finetunes whenever it is not inferencing. Training is a
tenant of the same registry/arbiter that serves inference (#305-M1): it gets
VRAM leases, is preempted when serving demand arrives, and resumes when the
rig goes idle again. Produced adapters (LoRA etc.) flow back into serving as
loadable artifacts.

## Decisions

### D1 — Wrap existing tools, never rebuild them

Execution backends are existing training suites, driven as subprocesses /
optional dependencies:

| Backend | Scope | License |
|---|---|---|
| LLaMA-Factory | LLM SFT/LoRA/DPO/PPO, broadest method ladder | Apache-2.0 |
| Unsloth (OSS) | fast single-GPU LoRA/QLoRA | Apache-2.0 |
| kohya sd-scripts | diffusion LoRA/finetune (K2 models) | Apache-2.0 |

All Apache-2.0: wrapping as subprocess or optional dependency is
license-clean; no code is copied (if it ever is, port with provenance per
the OSS reuse rule).

### D2 — Feasibility is a formula, not a rig constant

Full finetune and pretrain are NOT excluded. Every job request passes a
feasibility gate computed against the ACTUAL machine's resources (VRAM sum,
RAM, disk, interconnect), never against hardcoded assumptions from the
development rig. A 27B full finetune is a normal request on larger hardware.
Rejections are informative and include the method ladder: which cheaper
method (freeze → LoRA → QLoRA → offloaded full-FT) WOULD fit, with the
resource math shown.

### D3 — Standard training API surface: OpenAI fine-tuning protocol
(user directive 2026-07-31)

External suites must be able to use the fork directly as their endpoint,
without our web frontend. This is an instance of a general project rule:
every feature exposes the existing de-facto standard protocol for its
domain so existing clients work unchanged; our own frontend is just one
client of the same API, and extensions stay namespaced/additive. Two inbound directions:

1. **Remote training jobs.** The fork exposes the OpenAI fine-tuning API
   surface as the de-facto standard protocol:
   - `POST/GET /v1/files` (training data upload, JSONL)
   - `POST/GET /v1/fine_tuning/jobs`, `.../jobs/{id}`, `.../jobs/{id}/events`,
     `.../jobs/{id}/checkpoints`, `POST .../jobs/{id}/cancel`
   Any client speaking this protocol (OpenAI SDKs, LangChain, custom
   scripts, suites with a configurable base URL) can submit training jobs.
   Jobs are scheduled by the idle-tenant scheduler and executed by the
   wrapped backends from D1. Job state maps naturally:
   `validating_files → queued → running → succeeded/failed/cancelled`,
   with our preemption showing up as extended `running` (checkpoint +
   resume is internal, not a protocol state).
   Reimplementing a compatible HTTP surface is standard practice (same
   pattern as the OpenAI-compatible inference API); no licensing obstacle.

2. **Inference callbacks during external local training.** Suites that use
   an OpenAI-compatible endpoint for evaluation, data synthesis, or reward
   scoring while training locally can point that endpoint at the fork's
   inference API. This costs nothing beyond #335-M0 (OpenAI full surface)
   and is documented as a supported configuration.

The fork's own web frontend (Training section of IA v2, #342) is a client
of the same API — one surface, two consumers. The frontend adds what the
protocol does not carry: method-ladder picker, feasibility preview,
live loss curves via the events stream.

Extensions beyond the OpenAI protocol (method selection LoRA/QLoRA/full,
target model paths, diffusion jobs) ride in the `hyperparameters` /
`metadata` fields or a namespaced `x-htsglang` block, so vanilla clients
remain compatible.

### D4 — Tenant semantics

- Training runs ONLY while no inference demand exists (idle detection via
  registry activity), with a configurable grace window.
- Preemption: checkpoint-and-release on serving demand; VRAM lease returned
  to the arbiter; resume from checkpoint on next idle window.
- Training holds WARM_GPU/COLD residence like any other tenant; hibernate
  (#89) applies to paused jobs.

## Order of work

1. M1: job store + OpenAI fine-tuning surface + LLaMA-Factory LoRA backend
   (LLM), idle detection + preempt/resume against the registry.
2. M2: diffusion LoRA via kohya (K2), Unsloth fast path.
3. M3: full-FT/pretrain path (offload-aware feasibility math, multi-card).

## Open items

- Idle-detection threshold and grace window defaults (measure, don't guess).
- Checkpoint cost per method (LoRA cheap, full-FT expensive) feeds the
  preemption policy.
- Whether LLaMA-Factory's own API mode can be reused as internal executor
  interface instead of raw subprocess CLI (check when M1 starts).
