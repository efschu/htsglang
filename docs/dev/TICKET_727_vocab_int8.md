# #727 -- INT8 vocab weights: requant, wiring, and the switchover A/B

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots, no GPU. One
offline artifact written.

## The prize, per stage rather than per model

`embed_tokens` and `lm_head` are each `[248320, 5120]` = 1,271,398,400
elements, and each is **BF16 by the producer's instruction**:
`quantization_config.ignore` carries `lm_head` and `re:.*embed_tokens.*`, and
the safetensors headers agree.

| | bytes | MiB |
|---|---|---|
| BF16 | 2,542,796,800 | **2425.0** |
| INT8 + per-row scale | 1,271,398,400 + 496,640 | **1213.0** |
| saving per tensor | | **1212.0** |

Under the serving geometry (`--tp-size 1 --pp-size 3`) the two do **not** land
on the same rank: `embed_tokens` is on the FIRST stage, `lm_head` on the LAST.
So this is 1212 MiB off PP0 and 1212 MiB off PP2 -- never 2424 MiB off one
card. Quoting it as a single per-rank number would overstate it by 2x.

## Why a flag could not have delivered it

#724 established the dequant-on-gather capability is wired but **GGUF-only**,
and that is still true at HEAD: `qwen3_5.py:1393` passes a `quant_config` to
`VocabParallelEmbedding` only when `quant_config.get_name() == "gguf"`, the
whole tree carried exactly two embedding methods
(`unquant.py:114`, `gguf.py:1532`), and compressed-tensors'
`get_quant_method` (`compressed_tensors.py:280`) answers for `LinearBase` and
`FusedMoE` and nothing else. An int8 vocab tensor had **no method able to load
it**. Closing #724 for compressed-tensors is a missing component, not a missing
flag.

And no wiring could have produced the saving anyway: the bytes are not
quantized in any checkpoint we hold. The checkpoint has to change first.

**Correction to prior art:** NOTE_724's shard table lists `lm_head` in
`00003-of-00018` and `embed_tokens` in `00018-of-00018`. Both the base
`Qwen3.8-27B-INT8` index and the yarn1.5 overlay have them the other way round
(`embed_tokens` -> 00003, `lm_head` -> 00018). Verified against both indices.
Flagged for the merge train; that note is not in this lineage.

## Built

**1. `tools/requant_vocab_int8.py`** -- offline requant into a new
models-cache dir. Format copied from the checkpoint, not invented: symmetric
per-output-channel int8, `weight` I8 `[out,in]` beside `weight_scale` BF16
`[out,1]`, matching `config_groups` (`strategy: channel`, `symmetric: true`,
`observer: memoryless_minmax`). Unchanged shards are HARD-LINKED, so the new
directory costs the rewritten shards alone. Quantization is row-blocked: a
whole-tensor fp32 upcast would spike ~8 GiB beside live serving, and per-row
scales make the blocked result bit-identical rather than approximate.
`--targets` keeps the two tensors separable.

**Artifact written and verified:**
`/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-vocabint8-embed`
-- `embed_tokens.weight` now `I8 [248320, 5120]` with `weight_scale`
`BF16 [248320, 1]`; `lm_head` untouched; the `re:.*embed_tokens.*` entry
dropped from `ignore` while `lm_head` stays; **1.2 GB on disk** for the whole
directory (17 shards hardlinked, 1 rewritten).

**2. `CompressedTensorsEmbeddingMethod`** (`ct_embedding.py`) -- gather first,
then scale. Dequantizing first would materialize the whole 2.4 GiB vocab in the
activation dtype, which is the cost this path exists to avoid. Wired into
`get_quant_method` and the `qwen3_5` gate, both **gated on
`vocab_is_quantized`**, which reads the checkpoint's own `ignore` list. Every
checkpoint we serve today lists `embed_tokens` there, so the dense BF16 path
runs byte-identically and this component is inert until the requantized
checkpoint is actually pointed at.

11 tests, red first, mutation-proven (neutering the `ignore` scan reds exactly
the two default-unchanged tests). One test round-trips the runtime against the
requant tool, because a tool and a loader that disagree make the checkpoint
unreadable in the only way that matters.

## The two halves carry different risk, and only one was requantized

* **`embed_tokens` -- LOW.** A gather. Per-row scales make dequant exact per
  row and cost one multiply on the rows a batch touches. The result feeds a
  layernorm, which absorbs a per-row scale error. **Requantized here.**
* **`lm_head` -- the risky half.** A GEMM producing logits directly. A ~0.4%
  per-channel error lands on logit DIFFERENCES, which is exactly what softmax
  and argmax read, so near-ties can flip. This is plausibly what the producer's
  ignore list was protecting. **Deliberately NOT requantized**; the tool takes
  `--targets lm_head` when someone decides to, and it needs its own A/B.

## Switchover A/B -- boot-gated, filed for the window list

**Arm name:** `vocab-int8-embed`. Control is today's serving checkpoint;
treatment swaps only `--model-path`. Nothing else changes -- same cut, same
token budget, same flags.

```
# control
--model-path .../Qwen3.8-27B-INT8-yarn1.5
# treatment
--model-path .../Qwen3.8-27B-INT8-vocabint8-embed
```

Note the treatment inherits yarn1.5's config (it was requantized FROM that
overlay), so the rope settings are identical and are not a confound.

**Acceptance, in order. Stop at the first failure.**

1. **GATE 0 -- it loads and generates.** The int8 vocab must be picked up by
   the new method, not silently fall back. Confirm the embedding parameter is
   int8 at runtime; a boot that quietly took the dense path proves nothing and
   would show a *worse* VRAM number, not a better one.
2. **GATE A -- VRAM.** PP0 resident must fall by ~1212 MiB. Less than that
   means the dense path ran; more means something else moved and the
   measurement is confounded.
3. **GATE B -- quality, and this is the point of the arm.** Greedy
   (`temperature 0`) on a fixed prompt set, control vs treatment, compared as
   text. The embedding half should be near-neutral by construction; the arm
   exists to confirm that rather than assume it. Report first-differing
   character, not a pass/fail feeling.
4. **GATE C -- TTFT/decode unchanged within the boot's own A-vs-A floor.**
   The gather is one extra multiply on a handful of rows; a measurable
   regression here would mean the dequant is running somewhere it should not.

**What would falsify the design:** a VRAM saving materially below 1212 MiB on
PP0 (the method did not engage), or a quality delta on GATE B that clears the
floor (the per-row scheme is not as benign for embeddings as the structure
predicts, which would also cast doubt on ever doing `lm_head`).

**Not in this arm:** `lm_head` requantization, and any prefill-graph or cut
change -- both would confound GATE C.
