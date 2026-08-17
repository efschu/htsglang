# #724 — syv-ai/qwen38-27b-rtx3090 embed-quant harvest: CHECKS ONLY

Desk only, 2026-08-17. No boot, no model load, no serving contact. Checkpoint
inspected by reading `config.json`, `model.safetensors.index.json` and the
safetensors HEADERS only (8-byte length prefix plus the header JSON — 120 and
2624 bytes respectively). No shard was mmapped and no tensor data was touched.

Fixes are a follow-up decision. This note records verdicts, not changes.

## (a) Does our serving INT8-W8A8 Qwen3.8-27B carry embed_tokens / lm_head in BF16, untied?

**YES on all three counts, and the checkpoint says so in two independent
places.**

Checkpoint: `/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8`

From `config.json`:

- `tie_word_embeddings: False` — **untied**.
- `quantization_config.quant_method: compressed-tensors`, `format: int-quantized`
- `quantization_config.ignore` contains `'lm_head'` and `'re:.*embed_tokens.*'`
  (alongside vision, norm, conv1d and linear_attn) — so the quantizer was
  instructed to leave both alone.

From the safetensors headers, which is the check that does not rely on the
config being truthful:

| tensor | shard | dtype | shape |
|---|---|---|---|
| `lm_head.weight` | 00003-of-00018 | **BF16** | [248320, 5120] |
| `model.language_model.embed_tokens.weight` | 00018-of-00018 | **BF16** | [248320, 5120] |

**Consequence that closes an open question from #725.** That shape,
`[248320, 5120]`, is exactly NInfer's `lm_head` row — the single shape where
their measurement ("A16 always") points opposite to our shipped `N >= K`
aspect gate (which sees N/K = 48.5 and selects the fused lane). On THIS
checkpoint the divergence is **moot**: lm_head is BF16, so no fp8/int8
activation-quant gate ever sees it. The divergence would only become live on a
checkpoint that quantizes lm_head, and none of ours does.

## (b) Does our qwen3_5 code have the same unwired dequant-on-gather gap?

**NO — the kernel exists and IS wired. What we have is a narrower capability,
not an unwired one.** Two sub-verdicts:

**Main model — wired, GGUF-only.**
`python/sglang/srt/models/qwen3_5.py:1386-1405`. `embed_tokens` is built
QUANTIZED-RESIDENT (packed `qweight` via `GGUFEmbeddingMethod`) instead of the
dense BF16 materialization, saving ~1.1 GiB/rank on the 248k vocab. The gating
is explicit:

```
embedding_quant_config = None
if quant_config is not None and quant_config.get_name() == "gguf":
    ...
    embedding_quant_config = quant_config
```

So every non-GGUF quantization keeps `quant_config=None` for the embedding and
takes the dense path, by construction and with the comment saying so.
`SGLANG_GGUF_DENSE_VOCAB=1` restores the dense embed for GGUF too.

The dequant machinery it dispatches to is present:
`layers/quantization/gguf.py:1532` (`GGUFEmbeddingMethod`, extending
`GGUFLinearMethod`) over `ggml_dequantize`.

**NEXTN draft — no separate gap, because it does not build its own.**
`python/sglang/srt/models/qwen3_5_mtp.py:185-204`,
`set_embed_and_head_modules`: the draft is handed the target's embed/lm_head
MODULE OBJECTS, not their tensors, precisely because "a packed `qweight`
module has no `.weight` tensor to hand over". The draft's own never-loaded
modules are dropped and their storage freed. So whatever the main model built
— dense or quantized-resident — the draft inherits it, and there is no second
code path that could carry the gap independently. The tie invariant is
re-pointed explicitly for `tie_word_embeddings` (not our case here).

## Verdict

Their patch closes a gap we do not have in that form. Our quantized-resident
embedding path is wired and shared correctly into the draft; it is simply
scoped to GGUF. For the INT8 checkpoint we actually serve, embed and lm_head
are BF16 by the checkpoint's own ignore list, so there is nothing to
dequant-on-gather and the dense path is the correct one rather than a
shortfall.

The only real delta is a capability question, and it should be decided on its
own merits rather than as a bug fix: **should quantized-resident vocab weights
be extended beyond GGUF to compressed-tensors?** On our current checkpoint that
would require re-quantizing embed/lm_head, which the checkpoint's producer
deliberately excluded — so the ~1.1 GiB/rank prize is not available without
changing the checkpoint, and that trade (VRAM against whatever accuracy the
exclusion was protecting) is not measured here.
