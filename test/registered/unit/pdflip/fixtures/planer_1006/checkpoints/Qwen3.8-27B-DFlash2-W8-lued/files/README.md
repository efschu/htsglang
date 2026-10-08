---
license: apache-2.0
base_model: z-lab/Qwen3.8-27B-DFlash2
library_name: vllm
tags:
  - qwen3_5
  - dflash2
  - w8a16
  - int8
  - quantized
  - speculative-decoding
  - draft-model
base_model_relation: quantized
---

<div align="center">
  <h1>Qwen3.8-27B DFlash2 drafter · W8A16</h1>
  <p><strong>The DFlash2 speculative-decoding drafter, quantized W8A16 for Ampere-class GPUs.</strong></p>
</div>

> [!NOTE]
> This is a numerical **W8A16 quantization** of the [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter (mirror of [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)). Model credit belongs to the z-lab / incoai DFlash2 team and the Qwen base; this repository changes numerics only.

> [!WARNING]
> This is a **draft model, not a standalone LLM**. It is the speculative-drafting component of DFlash2 decoding: it proposes tokens that the target checkpoint verifies before any are emitted. It never emits unverified tokens. Serve it only as the `--speculative-config` drafter of a DFlash2-compatible target.

## What this is

The DFlash2 drafter ([PR vllm#52816](https://github.com/vllm-project/vllm/pull/52816), open as of 2026-08-19) is a 5-layer, block-diffusion draft model with a top-K candidate selector, trained to predict the target's next tokens at fixed layers. Quantizing it W8A16 drops it from 3.58 GiB (BF16) to **2.02 GiB** with no measurable effect on drafting (W8 vs BF16 drafter acceptance is equal: 2.65 vs 2.72/draft at the llama-benchy prose floor, same session).

| Property | Value |
|---|---|
| Architecture | `DFlash2DraftModel` (5 sliding-attention layers, block size 8) |
| Quantization | Data-free symmetric RTN W8A16, group size 128 |
| Packed | `mlp.*`, `self_attn.*`, `fc` projections: 36 tensors |
| Preserved BF16 | `candidate_selector.*` codebooks, `*_conv.*` kernels, norms: 45 tensors |
| Draft audit | `DRAFT AUDIT PASS` (dequant round-trip within INT8 bound; preserved bit-identical; no `mtp.*`/`lm_head`) |
| Worst error ratio | 0.749 |
| Size | 2.02 GiB (was 3.58 GiB BF16) |
| Source | `z-lab/Qwen3.8-27B-DFlash2` commit `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (verified unchanged 2026-08-19) |

The quantization was produced by the `int8-w8a16-dflash2` family toolkit
(`tools/quantize_draft_w8a16.py`); the exact target set and ignore list are
recorded in this repository's `recipe.yaml` and `config.json`.

## Usage

Serve it as the DFlash2 drafter of the matching quantized target
[`lued/Qwen3.8-27B-INT8-W8A16-DFlash2`](https://huggingface.co/lued/Qwen3.8-27B-INT8-W8A16-DFlash2)
by repo ID (both resolve from your HF cache):

```json
{"method": "dflash", "model": "lued/Qwen3.8-27B-DFlash2-W8", "num_speculative_tokens": 7}
```

- Spec window: 7 tokens (block 8), the drafter's trained design; shorter windows measure slower.
- vLLM: requires the vendored PR 52816 patch set from the [club-3090 repo](https://github.com/noonghunna/club-3090). See the target card's "Serving" section for the launch.
- Acceptance is workload-shaped, not a fixed number: ~6.0 tokens/draft on math, ~3.3 on long prose (mean acceptance length including the bonus token, same definition as the z-lab card). See the target card's "Speculative acceptance" section for the envelope and the z-lab comparison.
