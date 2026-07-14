# htsglang — sglang fork: heterogeneous tensor parallelism for mismatched GPUs

**htsglang** ("split heterogeneous sglang") is a fork of
[sgl-project/sglang](https://github.com/sgl-project/sglang) that makes a
**single tensor-parallel group run well on mismatched GPUs** — cards with
different VRAM sizes, and multiple ranks co-located on one physical GPU. It is
the sglang sibling of the author's vLLM fork
[**shvllm**](https://github.com/efschu/shvllm): same heterogeneous-TP feature
set, but built around sglang's native **RadixAttention** prefix cache (the
reason a second fork exists rather than only patching vLLM).

Fork: [github.com/efschu/htsglang](https://github.com/efschu/htsglang) ·
branch `feature/uneven-tp`.

## 1. Explicit rank → GPU placement (`--rank-gpu-id` / `--rank-gpu-memory-mib`)

Pin each TP rank to a specific **physical** GPU, one entry per rank. Duplicate
an index and two ranks share that card:

```bash
# TP=4 on 3 GPUs — two ranks co-located on GPU 0, one absolute budget per rank
python -m sglang.launch_server \
  --model /models/Qwen3.6-27B-FP8 \
  --tp-size 4 \
  --rank-gpu-id 0,0,1,2 \
  --rank-gpu-memory-mib 13500
```

`--rank-gpu-memory-mib` is a single absolute MiB budget applied per rank (under
pure TP every rank holds a structurally equal shard). It converts to each
card's own utilization fraction, so heterogeneous totals are handled correctly.
Co-locating multiple ranks on one GPU needs **NCCL ≥ 2.30**; the fork
auto-sets `NCCL_MULTI_RANK_GPU_ENABLE=1` (torch 2.11 ships NCCL 2.28, which
rejects co-located communicators — use the Docker image below, which pins
NCCL 2.30.7).

## 2. Uneven tensor parallelism (`--rank-tp-ratio auto`)

The headline feature. Instead of splitting the model into equal TP shards,
htsglang splits **proportionally to each card's memory** so a 32 GB card carries
a bigger slice than a 20 GB card. The split is applied per dimension:

- attention heads,
- GDN (gated-delta-net) linear-attention heads,
- dense MLP columns and MoE expert partitions,
- and the KV-cache pool.

```bash
# TP=3, memory-proportional shards derived automatically per card
python -m sglang.launch_server \
  --model /models/Qwen3.6-27B-FP8 \
  --tp-size 3 \
  --rank-gpu-id 0,1,2 \
  --rank-tp-ratio auto \
  --rank-auto-reserve-mib 2048
```

`auto` fills each card's available VRAM (minus `--rank-auto-reserve-mib` for
CUDA context / workspaces) and derives the per-rank weights itself. The KV
split is **self-calibrating**: at startup the server logs a suggested
`SGLANG_UNEVEN_MLP_VECTOR=...` (and, for MoE models, `SGLANG_UNEVEN_MOE_VECTOR`)
hint; feeding it back on restart rebalances the MLP/MoE shards and grows the KV
pool. Manual override via `--rank-mlp-ratio` / `--rank-moe-ratio` is also
available.

**Validated:** Qwen3.6-27B FP8, **TP=3 on 1× RTX 5090 + 2× RTX 3080** — clean
boot, ~297k `max_total_num_tokens` @ 32k context, coherent output, greedy
decode **bit-identical cold vs. warm**. Hybrid GDN + full-attention models and
**NEXTN / MTP speculative decoding** both work under the uneven layout. With
NEXTN spec decode at batch 1, measured decode throughput is **~86 t/s (prose) /
~90 t/s (code)** and prefill ~1319 t/s on this tri-GPU box (ahead of the same
config on the vLLM fork).

## 3. HiCache (hierarchical KV cache)

sglang's tiered KV cache — host-RAM L1 pool plus a file storage backend for
L2/L3 — runs on top of the uneven-TP layout, so evicted RadixAttention prefixes
spill to host memory and disk and are restored across restarts. Enabled in the
Docker profile via `--enable-hierarchical-cache` with the `file` backend.

## Docker

Prebuilt runtime image on GitHub Packages:
[**`ghcr.io/efschu/htsglang:cu130-nccl2307`**](https://github.com/users/efschu/packages/container/package/htsglang)
— CUDA 13.0, built for **sm75–sm120** (Turing … Blackwell / RTX 5090),
**NCCL 2.30.7** (baked in for multi-rank-per-GPU), and the HiCache file backend.
The image ships an ENV-driven entrypoint, so launch flags are set via
environment variables.

Pull it:

```bash
docker pull ghcr.io/efschu/htsglang:cu130-nccl2307
```

Minimal `docker run`:

```bash
docker run --rm --gpus all \
  --ipc=host --shm-size=16g \
  --security-opt apparmor=unconfined \
  -p 8011:30000 \
  -v /models-cache:/models-cache:ro \
  -e MODEL_PATH=/models-cache/Qwen3.6-27B-FP8 \
  -e TP_SIZE=3 -e RANK_GPU_ID=0,1,2 -e RANK_TP_RATIO=auto \
  ghcr.io/efschu/htsglang:cu130-nccl2307
```

Or via `docker/htsglang.yml` (default profile: TP=3 uneven, 1 rank per GPU,
HiCache + NEXTN). For multi-rank-per-GPU co-location set an absolute budget,
which auto-disables the ratio flags:

```bash
# .env next to docker/htsglang.yml — two ranks on GPU 0
RANK_GPU_ID=0,0,1,2
RANK_GPU_MEMORY_MIB=13500
```

```bash
cd docker && docker compose -f htsglang.yml up
```

The compose file requires `apparmor=unconfined` (LXC/Proxmox host), `ipc: host`
+ `shm_size` (NCCL / shared-memory IPC), and the `nvidia` device reservation; it
maps host port **8011 → container 30000**.

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
