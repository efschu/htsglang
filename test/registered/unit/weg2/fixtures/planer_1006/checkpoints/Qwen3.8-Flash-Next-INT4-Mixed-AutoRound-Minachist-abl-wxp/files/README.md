---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- auto-round
- int4
- int6
---
# Qwen3.8-Flash-Next, INT4/INT6 mixed (AutoRound), for vLLM on 3x to 4x RTX 3090

[Article](https://minamism.com/posts/flash-next-vllm/)

This repository has a mixed-precision AutoRound 4.21bpw quantization of Qwen3.8-Flash-Next, and the vLLM patch that runs it. The model is a 125B-A6B mixture of experts, which means that only a few experts run for each token. It also carries a 51.2B PLE n-gram embedding table, which the model looks up once per token at layer 2. The quantization is sized so that 262,144 tokens of context and 4 concurrent requests fit on three 24 GB consumer cards.

Measured on 3x RTX 3090 with 125 GiB of RAM:

| Item | Measured |
|---|---|
| Weights resident on GPU | 65.05 GiB = 21.6 GiB x 3 (body 4.205 bpw) |
| KV cache | 1,067,300 tokens = 4.07 x 262,144 |
| Host RAM | ~63 GiB (47.5 GiB of it the pinned QSA K/V pool) |
| PLE n-gram table | 95.37 GiB, read from disk, never resident |
| Decode, 1 request | 60 tok/s, flat from 3.6k to 248k context |
| Decode, 4 requests | 150-156 tok/s total |
| Prefill at 248k | ~3,700 tok/s |

QSA is Qwen Sparse Attention, the attention that the 12 full attention layers use.

---

## Contents

```
config.json, *.safetensors, tokenizer.*, ...   quantized checkpoint
vllm-patch/flash-next-vllm.patch            the vLLM patch you need
vllm-patch/flash-next-decode-01-*.patch     decode speed, 45 -> 80 tok/s
vllm-patch/flash-next-decode-02-*.patch     the other half of the same change
vllm-patch/Dockerfile                       builds a patched vLLM image
vllm-patch/compose.yaml                     a working deployment
vllm-patch/LICENSE, NOTICE, LICENSE.MIT     licenses for the files above
```

---

## What is quantized

Everything below uses `compressed-tensors`, `pack-quantized`, and symmetric group quantization.

| Group | Scheme | What it covers |
|---|---|---|
| A | INT4, gs128 | MoE routed experts (512 per layer, top-10). 58.0 GiB, 92% of the body |
| B | INT6, gs64 | `linear_attn` in and out projections, QSA `q/k/v/o_proj`, shared expert |
| C | INT8, gs64 | hyper-connection low-rank mixers |
| D | INT8, gs128 | `lm_head`, `embed_tokens`, PLE `key_proj` and `value_proj`, indexer `index_qk_proj` |

Three parts stay in BF16:

- The PLE n-gram table (95.37 GiB). The patch streams it from disk instead.
- The MTP module for multi-token prediction (4.86 GiB). The vLLM setup here does not use speculative decoding. With four 3090 cards instead of three, you can turn it on.
- The vision tower, which `--language-model-only` never loads.

The router (`mlp.gate`) stays BF16, as usual for a mixture of experts.

---

## Requirements

- 72 GB or more of VRAM. The weights need 65 GB, so they do not fit on fewer than three cards. This setup uses pipeline parallelism, where each card holds a different set of layers. VRAM is too tight for tensor parallelism, where each card holds a slice of every layer. If you have four 3090, TP=4 with PP=1 might work, but I'm sure you need to write your own patches for it.
- 65 GB or more of free host RAM, for the offloaded QSA K/V pool at the settings below.
- 96 GB or more of fast local storage, for the PLE table file. The patch reads it with random 4 KiB accesses, so put it on NVMe, not on a hard disk and not on NFS.
- vLLM 0.29.1rc1.dev47+gdc36fcce9, or a nightly build close to it. The patch applies against `vllm/vllm-openai@sha256:43f13b4c624ab9e9e6753d0eeb5953268bff334f2a826239e6f4a2197d47bb96`.

---

## The patch

| File | What it does |
|---|---|
| `models/qwen4_exp/nvidia/model.py` | passes `quant_config` to `embed_tokens`, `lm_head` and the hyper-connections, so that vLLM reads the quantized tensors in this checkpoint |
| `models/qwen4_exp/nvidia/hyperconnection.py` | splits the merged `input_mix_weight_down_block_inject` back into two Linears. The low-rank half is INT8 here and the injection half is BF16, and one merged quant method cannot serve both |
| `models/qwen4_exp/nvidia/ngram_embedding.py` | adds the mmap PLE backend, described below |
| `models/qwen4_exp/nvidia/model_state.py` | replaces the `PP>1` refusal for PLE models with a test of where the PLE layer lands |
| `model_executor/models/config.py` | the same test on the engine side. Both refusals exist, and removing one is not enough |
| `v1/core/kv_cache_utils.py` | fixes `allocate_kv_cache`. A KV group mapped to a pipeline rank that owns none of its layers still carries the global spec dict, and vLLM then raises `StopIteration` |
| `models/qwen4_exp/nvidia/qsa.py` | puts the QSA main K/V in pinned host memory behind a UVA view |
| `models/qwen4_exp/nvidia/ops/qsa.py` | adds staged prefill for host-resident K/V |
| `platforms/interface.py` | picks an attention block size that matches the offloaded layout |

### How to apply it

Build an image. This is the recommended path.

```bash
cd vllm-patch
podman build -t flash-next-vllm:local -f Dockerfile .
```

Or apply the patch by hand, from the directory that holds the `vllm` package:

```bash
patch -p1 < flash-next-vllm.patch
```

### The PLE table

Qwen3.8-Flash-Next looks up a 95.37 GiB n-gram embedding table at layer 2. Upstream vLLM keeps that whole table in pinned host memory. This deployment does not need it in RAM, because it does not need high throughput.

The patch maps the same bytes from a file instead. Set `VLLM_PLE_MMAP_PATH` to the path of that file. If the file does not exist, the patch creates it and fills it while the checkpoint loads. On every later start, the patch maps the file copy-on-write and does not read the PLE shards at all, so startup is much faster. Set `VLLM_PLE_MMAP_REBUILD=1` to fill the file again.

The patch marks the mapping `MADV_RANDOM`. The lookup gathers 16 rows of 320 B per token. Each row sits on a different page, so readahead only reads pages that nobody uses. That one call takes the measured traffic from 922 KiB per token down to 39.5 KiB per token.

The gather runs on the host, so it cannot run inside a captured CUDA graph. The two decode patches move it out of the forward entirely. `Qwen4ExpModelState.prepare_inputs` fills a fixed device buffer before the forward starts, and the PLE layer only reads that buffer. The forward then contains no host work, so it can be captured whole, and `cudagraph_mode: FULL_DECODE_ONLY` becomes legal.

Without those two patches the gather stays inside the forward, and you must use `cudagraph_mode: NONE`, or `PIECEWISE` with `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. FULL modes are rejected at construction time. `VLLM_PLE_HOST_GATHER=0` restores that older behaviour on a patched build.

By applying these patches, you get from 45 to 80 tok/s of single-stream decode. Note that hoisting the gather is not itself the speedup: measured on its own, under PIECEWISE, it gives 45.1 tok/s, because the per-layer eager breaks still sit on the host critical path behind the sync. The full-graph capture makes the hoist possible.

---

## Why the (most of) KV cache can live in host RAM

Decode speed is a bandwidth problem. Each decode step produces one token, and to produce it the GPU reads every weight and every piece of attention state that the step needs. On a single stream the card spends most of the step waiting for memory, not computing. So the size of that per-step read sets the token rate.

This is why a normal hybrid model keeps its KV cache in VRAM. Take Qwen3.8-27B, which is built on the Qwen3-Next architecture and shares most of its properties with Qwen3.8-Flash-Next (`qwen4_exp`). It still has one full attention layer every few layers, and a full attention layer reads its entire KV cache on every step. That read grows with the context, so decode gets slower as the conversation gets longer. It also grows past what any host link can carry, so the cache has to sit next to the compute.

The numbers of this model show the size of the problem. One QSA layer holds 2 key/value heads of 256 dimensions, as K and as V, in 2 bytes each, which is 2,048 B per token. At 262,144 tokens that is 512 MiB for one layer, and 6 GiB for all 12 layers on every single step. A PCIe 4.0 x16 slot carries about 25 GiB/s, so a host-resident cache of that shape allows about 4 tokens per second.

Here's an interesting part, Qwen3.8-Flash-Next avoids this in two ways.

Only 12 of the 48 layers have a KV cache at all. The other 36 layers are gated delta-net layers, a linear attention whose recurrent state has a fixed size. That state does not grow with the context.

Those 12 layers also do not attend over the whole context. QSA runs a cheap indexer over a pooled, compressed key, where `indexer_head_dim=128` divided by `indexer_compress_ratio=4` gives the pooled width. The indexer selects at most `indexer_budget=2048` positions. The layer reads the main K/V rows only for the positions that the indexer selects.

So `indexer_budget` bounds the bytes that a decode step reads, and the context length does not:

```
2048 selected x 2 kv heads x 256 dim x 2 (K and V) x 2 B  =  4 MiB per layer
x 12 layers                                               = 48 MiB per token
```

At 80 tok/s that is about 3.9 GB/s across the link, and at the 45 tok/s of the pre-decode-patch build it was about 2.2 GB/s. Either way it is a small fraction of a PCIe 4.0 x16 slot, and it overlaps with compute. The measurement agrees with the arithmetic: on the 45 tok/s build, decode ran at 45.62 tok/s with 248,667 tokens of context and at 44.23 tok/s with 3,671 tokens. The per-token read does not grow with context, which is the point here; the decode patches raise the rate but do not change that.

Only what the selection itself needs stays on the GPU. That is a 2-byte slot plus the pooled index key, which is `1 x (128 / 4) x 2 B = 64 B`. Together they are 66 B per token per layer, against 2,048 B for a full row. GPU-resident KV therefore drops by 31x, and the selection never crosses PCIe.

### Block sizing

vLLM sizes a KV block from the largest per-group page in the model. Once the QSA group shrinks to 66 B per token per layer, the mamba group dominates, at 3,207,168 B per page. The mamba group holds the recurrent state of the 36 linear attention layers. The allocator then wastes 7.8x. The patch raises the attention block size, to 12,144 tokens here, so that the QSA page matches the mamba page. The same byte budget then holds 7.7x more tokens.

### Prefill

Decode reads few rows, but prefill reads rows times selection width, one row at a time, at random offsets. That collapses to 941 tok/s. The patch stages a page range into a GPU arena once and runs every row against it. The transfer volume then scales with the prefix length, instead of with rows times selections. A running softmax merges the partial results, so VRAM depends only on the arena size. Prefill at 248k then reaches 3,701 tok/s.

Every layer on a rank shares one arena. The layers of one forward pass run in sequence, so one buffer is enough, and a resident buffer for each layer does not fit.

---

## Running it

`vllm-patch/compose.yaml` is a working deployment. These are the settings that matter:

```
--pipeline-parallel-size 3 --tensor-parallel-size 1
--max-model-len 262144 --max-num-seqs 4
--max-num-batched-tokens 512
--gpu-memory-utilization 0.97
--kv-cache-memory 550000000
--no-enable-prefix-caching
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

And this is the environment:

| Variable | Value | Why |
|---|---|---|
| `VLLM_PLE_MMAP_PATH` | path to the table file | required, see above |
| `VLLM_PLE_MMAP_REBUILD` | unset | set it to `1` to fill the table file again |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` | `1` only if you build without the decode patches, with `PIECEWISE` |
| `VLLM_PLE_HOST_GATHER` | unset (`1`) | set to `0` to put the gather back in the forward; costs ~45% of decode |
| `VLLM_QSA_KV_OFFLOAD` | `1` | moves the QSA main K/V to host RAM |
| `VLLM_QSA_KV_OFFLOAD_MAX_GIB` | `56` | refuses to start rather than exhaust RAM |
| `VLLM_QSA_KVO_ARENA` | `100663296` | 96 MiB staging arena |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | see below |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `0` | see below |

### Why each memory setting is there

If you remove any one of these, long prefills die.

- `--kv-cache-memory` sets the budget. When you pass it, `--gpu-memory-utilization` no longer affects the KV size at all, because vLLM skips profiling. The value that vLLM recommends is also too large. Its estimate of peak activation is about half of what a real long prefill uses.
- The value is `550000000`, not a round number. The PyTorch caching host allocator rounds pinned allocations up to a power of two. At 560 MB the pool is 4.03 GiB per layer, which rounds to 8 GiB, and 12 layers then ask for 96 GiB of RAM. At 550 MB the pool is 3.96 GiB per layer, which rounds to 4 GiB. The same deployment then uses 63 GiB instead of 111 GiB. A 10 MB change on the GPU is worth 48 GiB of RAM.
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` stops a double count. vLLM adds the measured CUDA graph memory to a peak figure that already includes an estimate of it. Turning the estimate off is worth about 15% more KV cache and costs nothing.
- `expandable_segments:True` is a correctness requirement here. It's not an optimization. Without it, fragmentation makes the peak grow with the prompt length until a long prefill dies. With it, the peak is flat from 4k to 130k.
- `--no-enable-prefix-caching` needs an explicit `--max-num-batched-tokens`. With prefix caching on, the prefill chunk is the KV block size. With it off, the chunk becomes `--max-num-batched-tokens` literally. If you turn prefix caching off and set no value, even a 4k prompt runs out of memory. The value here is 512. 1568 costs 470 MiB of peak memory for no speed, and 256 costs 32% of prefill throughput.
- `cudagraph_capture_sizes: [1,2,4]` covers the real batch sizes. Under pipeline parallelism, the number of in-flight requests and the forward batch size are different things. If you capture only size 1, vLLM drops to eager mode whenever two requests land in the same forward pass. That costs about 4x in throughput at 4 concurrent requests.
- Put the whole compilation configuration in one JSON argument. If you mix `--compilation-config '{...}'` with `--compilation-config.cudagraph_capture_sizes`, vLLM resets the mode back to the default and says nothing.

---

## Limits

- 262,144 tokens should be considered as the real ceiling, because that is `max_position_embeddings`. You can go up to 1M context, but I don't recommend it. Memory is not the constraint, because 1,215,172 tokens of KV fit in 400 MB per GPU. 
- vLLM refuses speculative decoding while `VLLM_QSA_KV_OFFLOAD=1`, because nobody validated it against the host-resident K/V path. The bundled MTP module is therefore unused. Note also that `ngram` drafting does not work with this model on any setting. It forces the vLLM V1 model runner, which does not prepare the PLE inputs.
- `VLLM_QSA_KV_OFFLOAD=1` requires TP=1, and the mmap PLE backend requires ETP=1.
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1` disables torch.compile, because vLLM forces `CompilationMode.NONE` when you set it. With the decode patches you do not need it, so it is off. `FULL_DECODE_ONLY` is `(FULL, NONE)` and does not require piecewise compilation, so inductor is not needed either way.
- Per-step fixed cost limits single-stream decode, not bandwidth. About 12.6 ms of the step is GPU kernel time. The rest used to be host stall: the PLE host gather synchronised the host with the device on every step, which destroyed the run-ahead that vLLM V2 plus async scheduling depends on, and the per-layer eager breaks then surfaced as GPU gaps. With the decode patches and `FULL_DECODE_ONLY` the step is 12.5 ms and single-stream decode is 80 tok/s, against about 155 tok/s at 4 concurrent requests. The pipeline round-trip is only 1.1 ms of the step and was never the problem.
- Recording the decode graph costs 0.06-0.07 GiB per rank and does not shrink the KV cache. Chunked prefill runs eager under `FULL_DECODE_ONLY`; measured prefill did not regress (4 x 247,823 tokens took 368 s against 378 s under `PIECEWISE`).

---

## Making sure that a build is right

```bash
# expect 1,067,300 tokens / 4.07x
podman logs <container> | grep "GPU KV cache size"

# expect block size 12144 and 66 B/token/layer
podman logs <container> | grep "QSA host-KV offload"

# expect 171 blocks x 12144 tokens, 3.961 GiB pinned, once per QSA layer
podman logs <container> | grep "QSA host KV"
```

If you see `Setting attention block size to 1568`, the `platforms/interface.py` hunk is not applied.

---

## License

The model weights inherit the upstream Qwen Community License 1.0. See `LICENSE` in this directory.

The three patches in `vllm-patch/` are derivative works of vLLM, so they stay under the Apache License, Version 2.0, like vLLM itself. `vllm-patch/LICENSE` holds the full text, and `vllm-patch/NOTICE` names the nine vLLM files that they change.

`vllm-patch/Dockerfile` and `vllm-patch/compose.yaml` contain no vLLM code. They are under the MIT license, in `vllm-patch/LICENSE.MIT`.