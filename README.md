# htsglang — sglang fork: asymmetric tensor parallelism (for heterogeneous / mismatched GPUs) + Qwen3.5/3.6 GGUF

under construction

## Docker

Pull:

```bash
docker pull ghcr.io/efschu/htsglang:cu130-nccl2307
```

Run (even TP=2 example, validated boot):

```bash
docker run -d --name htsglang \
  --security-opt apparmor=unconfined \
  --gpus '"device=<gpu-uuid-0>,<gpu-uuid-1>"' \
  --shm-size 16g --ipc host \
  -p 8021:30000 \
  -v <model-dir>:/model:ro \
  ghcr.io/efschu/htsglang:cu130-nccl2307 \
  python3 -m sglang.launch_server --model-path /model \
    --served-model-name <served-name> --tp-size 2 \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 \
    --context-length <ctx-len> --max-running-requests 8 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --trust-remote-code --host 0.0.0.0 --port 30000
```

`--gpus` takes NVML GPU UUIDs (`nvidia-smi -L`), not indices, to avoid
enumeration-order drift across driver/toolkit versions.

## Chat template

The checkpoint's own `chat_template.jinja` is used by default. For
Qwen3.6-class checkpoints add `--reasoning-parser qwen3
--tool-call-parser qwen3_coder`. Clients disable thinking via
`chat_template_kwargs: {"enable_thinking": false}`. Tool calling uses the
standard OpenAI `tools` / `tool_choice` request fields.

Caveat: the image entrypoint uses `${VAR:=default}` shell defaults, so an
*empty* env var (`-e RANK_GPU_ID=`) is treated as unset and the baked-in
default is re-applied, not omitted. To run without the image's defaults,
invoke `python3 -m sglang.launch_server ...` directly as the container
command (as in the example above) instead of relying on env vars.

NCCL: the image pins NCCL 2.30.7, required for multi-rank-per-GPU
co-location.

## Fork flags

- `--rank-gpu-id <id,id,...>` — one CUDA device index per tensor-parallel rank; duplicates co-locate ranks on one GPU, length must equal `--tp-size`.
- `--rank-gpu-memory-mib <MiB>|<MiB,MiB,...>` — absolute memory budget per rank; a scalar applies to every rank, a per-rank list is allowed with `--rank-tp-ratio`; no further utilization ceiling is applied.
- `--rank-tp-ratio auto|auto-performance|<w,w,...>` — uneven TP: integer shard weights per rank, derived from the memory budgets or NVML totals (`auto`) or additionally from a measured hardware profile (`auto-performance`).
- `--rank-auto-reserve-mib auto|<MiB>|<MiB,MiB,...>` — headroom subtracted from the NVML total when `--rank-tp-ratio auto` derives the budgets itself.
- `--rank-mlp-ratio <w,w,...>` — per-rank weights for the dense-MLP family only, rebalancing weight bytes so the min-synced KV pool grows.
- `--rank-moe-ratio <w,w,...>` — the same lever for the fused expert (MoE) weight family.
- `--mamba-checkpoint-interval <tokens>` — pin all radix-cached mamba/GDN checkpoints to absolute multiples of this token count.

**Not in the published Docker image — run from source or wait for the next image build:**

- `--rank-vocab-ratio auto|<w,w,...>` — ratio-weighted vocab sharding of the tied vocab layers, balancing lm_head read time instead of shard width.
- `--rank-kv-ratio coupled|capacity|auto|<t,t,...>` — uneven-DCP KV token ownership, decoupled from the weight split; `capacity` installs the measured per-rank optimum.
- `--rank-perf-tune both|dec|enc` — tuning target of `--rank-tp-ratio auto-performance`.
- `--rank-perf-loose-ctx-percent <percent>` — context floor for auto-performance candidates: predicted max context must stay at or above (100 - X)% of the VRAM-auto split.
- `--swa-pool-sizing ratio|cap` — SWA pool sizing on hybrid sliding-window models; `cap` pins the SWA pool at its window-bounded worst case and gives the rest to full attention.
- `--speculative-draft-placement split|solo` — draft model TP-sharded (default) or unsharded on one rank that broadcasts its draft token ids.
- `--speculative-draft-gpu <id>` — CUDA device index whose TP rank hosts the solo draft.
- `--speculative-adaptive-graph-memory auto|resident|offload|offload-scratch` — VRAM policy for the pre-built adaptive runtime states.
- `--speculative-cross-algorithm` — load the NEXTN/MTP and DFLASH drafts co-resident on one server.
- `--speculative-cross-algorithm-force nextn|dflash|schedule:N|auto|policy` — which rung serves batches: static pin, debug schedule, acceptance-driven bandit, or policy table.
- `--speculative-cross-algorithm-ctx-gate auto|off|<tokens>` — context threshold at or above which the DFLASH rung is ineligible.
- `--speculative-drafter-policy auto|<start_ctx:family:value,...>` — ordered context-threshold stage table used by `--speculative-cross-algorithm-force policy`.
- `--enable-fast-lane` — latency-priority scheduling class: requests tagged `lane='fast'` outrank batched heavy requests.
- `--fast-lane-priority <int>` — priority value seeded for fast-lane requests.
- `--fast-lane-reserved-heavy-slots <int>` — minimum number of running heavy requests that fast-lane preemption never drops below.
- `--fast-lane-heavy-aging-ms <ms>` — a heavy request queued longer than this is promoted ahead of the fast tier once; 0 disables aging.
- `--weightless-kv-fastlane` — experimental: one head rank holds all weights and runs as TP=1, the other ranks hold only a KV token shard.
- `--weightless-kv-head-rank <rank>` — the weight-bearing rank of the weightless-KV lane.
- `--weightless-kv-chunked-block-size <tokens>` — block-decode staging size for that lane; 0 keeps the single monolithic attention call.
- `--weightless-kv-host-spill-tokens <tokens>` — per-rank pinned-host KV overflow slots, so one sequence's KV can exceed the rank's VRAM.
- `--weightless-kv-spill-device-cap <tokens>` — debug cap on device-resident KV slots per rank, forcing the host-streaming path.
- `--enable-kv-session-offload` — experimental: spill the youngest running session's KV to host RAM and keep decoding it from host.
- `--kv-session-offload-block-size <tokens>` — per-rank streamed block size of the spill tick.
- `--kv-session-offload-tick-interval <iterations>` — minimum scheduler iterations between two spill ticks.
- `--kv-session-offload-tick-adaptive` — derive the tick cadence from measured device slack and tick cost instead of the static interval.
- `--kv-session-offload-tick-floor <iterations>` — anti-starvation bound on the adaptive interval, i.e. the guaranteed minimum progress rate of a spilled session.
- `--kv-session-offload-restore-margin-tokens <tokens>` — free-slot headroom required before a spilled session is restored.
- `--kv-session-offload-restore-hysteresis-steps <iterations>` — consecutive iterations the restore condition must hold.
- `--kv-session-offload-max-spills <n>` — maximum number of concurrently spilled sessions; sizes the pinned host pool linearly.
- `--determinism-logits-dump-dir <path>` — debug: every rank dumps its per-step next-token logits row into this directory.
- `--enable-weights-disk-backup` — hibernate (suspend-to-disk): the `/hibernate` endpoint parks each rank's post-load GPU weights for a fast restore.
- `--hibernate-dir <path>` — directory for the hibernate weight shards and manifest; required with `--enable-weights-disk-backup`.

*Upstream sglang README below.*

--------------------------------------------------------------------------------

<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## News
- [2026/06] 🔥 The next generation of speculative decoding: DFlash and Spec V2 ([blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).
- [2026/04] 🔥 DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles ([blog](https://lmsys.org/blog/2026-04-25-deepseek-v4/)).
- [2026/06] SGLang provides day-0 support for latest open models ([Nemotron 3 Ultra](https://lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/), [Nemotron 3 Super](https://lmsys.org/blog/2026-03-11-run-nvidia-nemotron-3-super/), [Higgs Audio v3 TTS](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)).
- [2026/02] 🔥 Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72 ([blog](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)).
- [2026/01] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).

<details>
<summary>More</summary>

- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark/5090), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend used for training many frontier models, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/), [GB300 long context](https://lmsys.org/blog/2026-02-19-gb300-longctx/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, Modal, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at [sglang@lmsys.org](mailto:sglang@lmsys.org).

Long-term active SGLang contributors are eligible for coding agent sponsorship, such as Cursor, Claude Code, or OpenAI Codex. Email [sglang@lmsys.org](mailto:sglang@lmsys.org) with your most important commits or pull requests.

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
