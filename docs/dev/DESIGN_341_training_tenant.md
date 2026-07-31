# DESIGN #341 — Training/finetune as idle tenant

Status: **M1 implemented** (job store + OpenAI fine-tuning surface +
LLaMA-Factory backend + idle tenant with preempt/resume), M2/M3 open. This
file is the single persistent record of the #341 analysis and decisions; the
chat history is not authoritative.

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

## M1 as built (2026-07-31)

Modules, all under `python/sglang/srt/training/`, none of them importing
torch so the whole surface is testable on a card-less host:

| Module | Carries |
|---|---|
| `store.py` | uploaded files, job records, event log with cursor pagination and SSE subscribers, checkpoints |
| `feasibility.py` | D2: the formula, the machine probe, the model profiler, the method ladder |
| `backends/` | D1: the executor interface, the LLaMA-Factory subprocess wrapper, a mock executor |
| `tenant.py` | D4: idle detection, VRAM lease, checkpoint-and-release preemption, resume |
| `service.py` | the assembly the HTTP surface talks to |
| `activity.py` | this process's own serving-activity stamp |

The HTTP surface is `entrypoints/openai/serving_files.py` and
`serving_finetune.py`, wired into `http_server.py` as
`app.state.openai_serving_files` / `openai_serving_fine_tuning` in the same
place and the same style as the #335-M0 image and speech adapters.

Server flags: `--enable-training-tenant`, `--training-artifact-root`,
`--training-model-root`, `--training-idle-grace-seconds`,
`--training-poll-seconds`, `--training-preempt-timeout-s`,
`--training-save-steps`, `--training-default-backend`,
`--training-default-method`, `--training-event-stream-timeout-s`. Documented
in `docs/rig-runbook.md` section 10.

### Resolved: LLaMA-Factory is driven by CLI subprocess, not by its API mode

The open item asked whether LLaMA-Factory's own API mode could be the
internal executor interface. It cannot, because its API mode is not a
training API: `llamafactory-cli api` starts an OpenAI-compatible **chat**
server for an already-trained model. The only programmatic training entry
point is the in-process call `llamafactory.train.tuner.run_exp(args)`. So the
real choice was subprocess CLI versus importing LLaMA-Factory into the
serving process, and three properties decided it for the subprocess:

1. **Preemption.** D4 requires releasing VRAM on serving demand. A subprocess
   is signalled and the kernel returns its memory when it exits. An
   in-process trainer holds a CUDA context and an allocator arena inside the
   *serving* process; releasing that means unloading torch state the server
   itself uses.
2. **Dependency isolation.** LLaMA-Factory pins transformers, peft, trl,
   accelerate and bitsandbytes. In-process, the serving stack and the
   training stack become one constraint set.
3. **Blast radius.** A CUDA OOM or a segfault in a training step kills a
   subprocess; in-process it kills the server.

The cost is stated rather than hidden: preemption granularity is
`save_steps`, because the trainer's own checkpointing is the only place a
subprocess can be stopped cleanly. The event log says so when a preempt
happens.

LLaMA-Factory is not installed in this rig's venv and was deliberately not
installed: the backend probes for it and rejects by name with an install
remedy, and the mock executor covers the whole lifecycle in tests. Nothing
was pinned, nothing in the venv changed.

### Resolved: idle detection is policy, the ledger is safety

Two mechanisms, deliberately separate. `IdleMonitor` combines demand sources
(this process's own request stamp, the registry's `last_used_ts` over
class-1/2 engines) and a source that cannot answer contributes nothing rather
than vetoing — a policy that refused to train without a reachable control
plane would never train on most deployments. The VRAM lease is the actual
guard: an `acquire` against the ledger fails closed with the holders named if
the monitor was wrong.

`registry_view.RegisteredEngine` gained `last_used_ts` (additive, parsed from
the arbiter snapshot that already carried it).

### Resolved: preemption is not a protocol state

A preempted job stays `running` and reports `x-htsglang.tenant_state ==
"preempted"` with `preemptions`, `last_step` and `resume_from`. Adding a
state to OpenAI's vocabulary would break every client that switches on it.

## Open items

- Idle-detection threshold and grace window defaults: shipped as flags with
  120 s / 2 s defaults and no measurement behind them yet. Measure on a rig
  with real serving traffic before calling the defaults right.
- Checkpoint cost per method is now *estimated* by `checkpoint_bytes` and
  used for the disk post. It is not measured; a measured cost should feed the
  preemption policy (whether it is worth preempting at all for a short
  serving burst).
- Job records live in memory. A server restart loses them, which is honest
  today because a restart also kills the executor subprocess. Surviving a
  restart needs the executor to outlive the server — M3.
- Multi-card training is priced (`world_size`, sharded for `full_offload`)
  but only the single-process launch path is built. Multi-card execution is
  M3.
- The produced adapter is written to the artifact root and reported as
  `fine_tuned_model`, but is not yet registered as a loadable serving
  artifact. Closing that loop is the D4 sentence "produced adapters flow back
  into serving" and is M2.
