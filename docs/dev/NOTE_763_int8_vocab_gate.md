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

## Boot-proof, 2026-08-18 23:12Z

Plain TP=3 uneven-DCP, `--model-path Qwen3.8-27B-INT8-vocabint8-embed`, on the
fixed tree (`1d1363cbc6`). All three acceptance criteria met:

1. **Coherent.** Greedy `'The capital of France is'` ->
   `' Paris.\nThe capital of Germany is Berlin.'`, and a prose prompt returned a
   correct Rayleigh-scattering sentence. This is the same checkpoint and the
   same layout that answered `' a a a a a a'` before the fix.
2. **The scale loads.** `weight_scale not found in params_dict` occurs **0**
   times (it occurred 3 times, once per rank, on the pre-fix boot).
3. **The saving materialized**, within 1 MiB of the arithmetic:

| rank | bf16 vocab (yarn1.5) | int8 vocab | saved |
|---|---|---|---|
| 0 | 14.930 GiB | 14.535 GiB | 404.5 MiB |
| 1 | 10.186 GiB | 9.793 GiB | 402.4 MiB |
| 2 | 9.504 GiB | 9.109 GiB | 404.5 MiB |
| | | **total** | **1211.4 MiB** |

Predicted 1212 MiB (BF16 2425.0 -> INT8 1212.5 + 0.5 scale). The split is even
across the three TP ranks because `--rank-vocab-ratio` is unset; a dense BF16
path cannot produce this reduction, which is what makes the number proof that
`CompressedTensorsEmbeddingMethod` is the method actually in use. Note that no
log line names the class -- the code never logs its embedding method, so the
absent warning plus the exact byte saving is the available evidence.

## The PP/TP contradiction, resolved

The open question was why the same checkpoint was reported coherent under PP=3.
The boot logs answer it: the defect is present under BOTH layouts and is not
layout-dependent at all.

`boot_735_v7tr_ctg.log` (vocabint8-embed, `tp_size=1, pp_size=3`) carries
`Parameter model.embed_tokens.weight_scale not found in params_dict` three
times -- exactly like the TP=3 bench. On PP the embedding lives only on stage 0,
and PP0's line is the SCALE alone (PP1/PP2 report the weight missing too simply
because they do not own the embedding at all -- the same benign pair appears in
`boot_knowngood_r14.log`, which is healthy). By contrast the yarn1.5 known-good
shows `weight_scale not found` **zero** times, because that checkpoint has no
scale tensor to place.

So the embed-owning rank behaved identically in both layouts: int8 rows loaded
into a BF16 dense parameter, scale discarded.

The generation artifacts close it. Every arm booted on `vocabint8-embed`
produced garbage, PP included -- `v7pp10`, `v7pp17`, `v7pp18`, `v7pp19`,
`v7pp20`, `v7tr_gap`. Every arm booted on `yarn1.5` produced correct answers --
`arm1`, `armA`, `armA2`, `armB2`, `comp4`, `disc`, `nohc`, and both of today's
`step1ctg` arms. There was never a coherent probe on the requantized checkpoint
under any layout, so the prediction holds without an exception to explain.

WHY IT LOOKED LIKE THERE WAS ONE, and this is the part worth keeping: the
garbage does not appear where a reader looks. `CONTENT` is the EMPTY STRING and
the token soup sits in `reasoning_content`:

    CONTENT: ''
    REASONING: ' The wordirikao...EventManager_pars...'
    FINISH: length

The model never emits a closing think marker, so the qwen3 reasoning parser
takes the entire degenerate stream as reasoning and leaves the content field
empty. A probe that prints `content` alone shows `''`, which reads as "no
output" or "empty answer" rather than "the model is broken" -- and `FINISH:
length` means it ran to max_tokens instead of stopping. Any check that judges
coherence from the content field alone is blind to exactly this failure, which
is the same blindness that let a GATE 0 the ticket had already specified go
unrun. Coherence probes must read `reasoning_content` too, or use the native
`/generate` route, which has no parser in front of it.
