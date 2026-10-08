---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - qwen3
  - quantized
  - int8
  - llmcompressor
---

# Qwen3.8-27B-INT8

INT8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [llm-compressor](https://github.com/vllm-project/llm-compressor).

Vision encoder, gated-delta attention (conv1d / linear_attn), `lm_head`, embeddings, and norm layers are kept in original precision.
