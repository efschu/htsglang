# #763 — the int8 vocab was never selected, and the shard contract behind it was wrong

Desk only, 2026-08-18. Root found without a boot; the evidence is a serving log
line, two source locations, and a hermetic reproduction.

## The symptom

`Qwen3.8-27B-INT8-vocabint8-embed` (the #727 requantized artifact) generated
token soup under plain TP=3 uneven-DCP. Greedy `/generate` on "The capital of
France is" returned `" a a a a a a a…"`; the chat route returned empty. Swapping
only the checkpoint to `Qwen3.8-27B-INT8-yarn1.5`, with the identical code path,
flags and TP vector, restored fully coherent output at identical throughput
(473 vs 474 tok/s at bs=24) — so the defect cost correctness, not speed.

## The root: a gate that could never fire

`python/sglang/srt/models/qwen3_5.py:1444` (pre-fix) selected the #727 embedding
method with

```
and quant_config.get_name() == "compressed-tensors"
```

while `CompressedTensorsConfig.get_name`
(`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py:262-263`)
returns the UNDERSCORE spelling:

```
def get_name(self) -> str:
    return "compressed_tensors"
```

The hyphen spelling is what the checkpoint's `quant_method` field and the prose
use; the config names itself with an underscore. The `elif` therefore never
matched, `embedding_quant_config` stayed `None`, and `VocabParallelEmbedding`
was built on the DENSE method — which is why the int8 rows were copied into a
BF16 embedding with no scale applied, and the scale itself had nowhere to go.

Serving said so plainly, on every rank
(`/spinning/evidence-665-f1/bench_tp3.log:110-112`):

```
[TP2] Parameter model.embed_tokens.weight_scale not found in params_dict
[TP1] Parameter model.embed_tokens.weight_scale not found in params_dict
[TP0] Parameter model.embed_tokens.weight_scale not found in params_dict
```

Reproduced at the desk, both directions, no GPU:

| gate | selected method | parameters | weight dtype |
|---|---|---|---|
| old (`quant_config=None`) | `UnquantizedEmbeddingMethod` | `['weight']` | bf16 — int8 rows cast in unscaled |
| fixed | `CompressedTensorsEmbeddingMethod` | `['weight', 'weight_scale']` | int8 + per-row scale |

Everything else in the #727 lineage was correct: the ignore-list scan answers
`True` for every `embed_tokens` prefix on this checkpoint (measured), and
`get_quant_method`
(`compressed_tensors.py:300-317`) dispatches `VocabParallelEmbedding` properly.
The feature was simply never reached.

## The second defect, which the first one was hiding

`CompressedTensorsEmbeddingMethod.create_weights` registered `weight` and
`weight_scale` without `output_dim`. `VocabParallelEmbedding.weight_loader`
reads exactly that attribute
(`python/sglang/srt/layers/vocab_parallel_embedding.py:524`, branch at `:577`)
to decide whether to narrow the checkpoint tensor to the rank's row range; a
parameter without it takes the "copy onto all gpus" path meant for
shard-invariant tensors like gptq's `g_idx`, and asserts at `:584`.

This is invisible at `tp_size == 1` — the whole vocab IS the local shard, which
is why the PP=3 layout was never affected — and it bites the moment the vocab is
row-sharded. Fixing only the gate would therefore have converted token soup into
a hard boot crash under TP. Both halves are fixed together, and the per-row
scale now shards on dim 0 with the rows it belongs to; slicing one without the
other pairs each row with a stranger's scale.

## Why no test caught either half

`test_ct_embedding_int8_727.py` drove a single unsharded partition and handed
`create_weights` a dummy loader (`weight_loader=lambda *a, **k: None`), so the
real loader never ran against these parameters, and no test asserted that the
selection gate actually fires. Both axes are now covered:
`TestTheFamilyGateActuallyMatches` binds the predicate to the real config
object, and `TestItSurvivesTpVocabSharding` loads TP=3 shards through the real
`VocabParallelEmbedding.weight_loader` and requires the reduced result to equal
the TP=1 reference.

## Status

The gate now matches on the normalized family name via
`is_compressed_tensors_config`, so a second spelling cannot re-open this. 18/18
in the registered #727 suite; the TP-sharding test was confirmed RED before the
fix (AssertionError at `vocab_parallel_embedding.py:584`).

End-to-end confirmation that the requantized checkpoint now serves coherently
under TP=3 requires a boot and is NOT claimed here.
