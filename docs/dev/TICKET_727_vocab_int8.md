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

## The two halves, and which one is a HYPOTHESIS rather than a verdict

Both halves are now built as separate artifacts, because the difference between
them is a prediction that the arm exists to test -- not a reason to skip one.

* **`embed_tokens`.** A gather. Per-row scales make dequant exact per row and
  cost one multiply on the rows a batch touches, and the result feeds a
  layernorm that absorbs a per-row scale error. Expected near-neutral.
* **`lm_head`.** A GEMM producing logits directly. **HYPOTHESIS UNDER TEST:**
  a ~0.4% per-output-channel error lands on logit DIFFERENCES, which is exactly
  what softmax and argmax read, so near-ties can flip. This is plausibly what
  the producer's ignore list was protecting.

That hypothesis is stated so the suite can refute it. It is deliberately NOT
used to withhold the artifact: a desk argument about logit differences is not
evidence, and the three-arm suite produces numbers where the argument produces
only a prediction. If the suite shows flips, the numbers say so; if it does
not, both halves ship.

## Artifacts

| arm | path | vocab state | disk |
|---|---|---|---|
| baseline | `Qwen3.8-27B-INT8-yarn1.5` | both BF16 | (incumbent) |
| embed-only | `Qwen3.8-27B-INT8-vocabint8-embed` | embed I8, lm_head BF16 | 1.2 GB |
| both | `Qwen3.8-27B-INT8-vocabint8-both` | both I8 | 2.7 GB |

Each was verified from the safetensors headers after writing: I8
`[248320, 5120]` with `weight_scale` BF16 `[248320, 1]`, and the matching
`ignore` entries dropped (`embed` only for the first, both for the second).
`-both` was built FROM `-embed`, so the two are the same bytes except for
`lm_head` -- the arms differ in exactly one tensor each, which is what makes
the three-way comparison attributable.

Disk cost is small because unchanged shards are hard-linked: 18 shards, one
rewritten per artifact.

## Switchover A/B -- THREE arms, boot-gated, filed for the window list

**Arm name:** `vocab-int8`. Only `--model-path` changes between arms. Same
cut, same token budget, same flags, same rope (both artifacts inherit
yarn1.5's config, so it is not a confound).

```
A  baseline    --model-path .../Qwen3.8-27B-INT8-yarn1.5
B  embed-only  --model-path .../Qwen3.8-27B-INT8-vocabint8-embed
C  both        --model-path .../Qwen3.8-27B-INT8-vocabint8-both
```

Run A, B, C. B is what isolates `lm_head`: any quality delta present in C but
absent in B is attributable to `lm_head` alone, which is the entire reason the
middle arm exists rather than a two-arm baseline-vs-both.

**Acceptance, in order. Stop at the first failure.**

1. **GATE 0 -- each arm loads and generates.** The int8 vocab must be picked up
   by `CompressedTensorsEmbeddingMethod`, not silently fall back to dense.
   Confirm the embedding parameter is int8 at runtime; a boot that quietly took
   the dense path proves nothing and would show a *worse* VRAM number, not a
   better one.
2. **GATE A -- VRAM, per stage.** B: PP0 resident falls ~1212 MiB, PP2
   unchanged. C: PP0 AND PP2 each fall ~1212 MiB. A saving materially below
   that means the method did not engage on that stage.
3. **GATE B -- QUALITY, the point of the suite.** Per arm: the club-3090 suite
   plus determined-answer probes (questions with a single correct answer, where
   a flipped near-tie is visible as a wrong answer rather than as a style
   difference). Greedy, `temperature 0`, fixed prompt set, fixed order, fixed
   seed. Report per arm: suite score, determined-answer accuracy, and
   first-differing character against arm A. **A vs A first** -- without the
   baseline's own boot-to-boot floor, a B or C delta cannot be read, and this
   model is not deterministic across boots (the GDN prefill limit).
4. **GATE C -- TTFT / decode within each boot's own A-vs-A floor.** The gather
   is one extra multiply on a handful of rows. `lm_head` in arm C is a
   dequantized GEMM, so C is where a decode regression would appear if
   anywhere; report per rank.

**Decision rule, fixed before the run so it cannot be argued afterwards:**

* B clears GATE B within the A-vs-A floor -> **embed-only ships.**
* C also clears it -> **both ship**; the logit-difference hypothesis is
  refuted and should be recorded as such.
* C degrades while B does not -> the hypothesis is **confirmed**, `lm_head`
  stays BF16, and the 1212 MiB on PP2 is the priced cost of that.
* B degrades -> stop; the per-row scheme is not benign even for a gather,
  which would also retire the `lm_head` question.

**Not in this arm:** any prefill-graph or cut change -- both would confound
GATE C.

## #735 dependency: WHICH arm funds the GDN slot plan

`DESIGN_pp_layer_set.md` prices GDN slots 21-24 (nominal; 20-23 on the NVML
total) on the 5090 AFTER the #727 int8 vocab saving, and marks #727
**required**: on BF16 vocab the same card supports ~7 slots.
`DESIGN_family_fullplan.md` §2.1 places `lm_head` on the 5090 -- so the
tensor whose saving funds those slots is `lm_head`, and **the arm that
satisfies #735 is arm C (`vocabint8-both`)**. Arm B leaves `lm_head` BF16
and funds nothing on the 5090. Consequence, priced into the decision rule:
if C fails GATE B, the 21-24 slot plan loses its funder and the full-plan
ladder must be re-derived at ~7 slots or find another 1212 MiB.

(Artifact naming: the register may call the second artifact
`vocabint8-embed-lmhead`; the directory on disk is `vocabint8-both`, same
content -- doc-fix over rename, the verified artifact does not churn.)

## Second-artifact verification record (2026-08-18, independent re-check)

Verified from the shard headers and inodes, not from the build report:

* `-both/model-00003-of-00018`: `embed_tokens.weight` I8 `[248320, 5120]`
  beside `weight_scale` BF16 `[248320, 1]` (corrected shard table: embed in
  00003, lm_head in 00018);
* `-both/model-00018-of-00018`: `lm_head.weight` I8 + BF16 scale, same
  shapes;
* `config.json` `ignore` carries NO vocab entry (both dropped);
* hardlink economy: 17 of 18 shards inode-shared with `-embed` (including
  the I8-embed 00003), exactly one unique rewritten shard (00018) -- the
  arms differ in exactly one tensor, which is what makes the three-way
  comparison attributable;
* no upcast transient by construction: built by
  `tools/requant_vocab_int8.py`'s row-blocked path (bit-identical to
  whole-tensor per-row quantization, no ~8 GiB fp32 spike).

## Turnkey runner (desk-smoked, window-ready)

`tools/ab_vocab_int8_727.py` executes this protocol end to end: artifact
verification BEFORE the first boot (a window spent on a wrong checkpoint is
the most expensive way to find a build defect), then A1, A2 (the A-vs-A
floor), B, C, with GATES 0/A/B/C per arm and the decision rule above
applied mechanically -- including the #735 consequence in the verdict text.
GATE 0 is now observable: `ct_embedding.py` logs `INT8-VOCAB ENGAGED` per
loaded vocab tensor (0/1/2 lines for A/B/C).

The boot/suite/perf legs are command templates the window operator fills
(`--boot-cmd/--suite-cmd/--perf-cmd`, `{model_path}`/`{arm}` substituted);
`--mock DIR` replaces them with fixtures. Desk-written-never-executed:
`test_ab_runner_727.py` drives every gate in both directions, the floor
arithmetic, the abort-on-baseline-failure branch, the verdict table, and
two full mocked CLI runs (SHIP-BOTH and ABORT), plus the artifact verifier
against synthetic checkpoints broken in each way it must catch -- and
against the real artifacts on this rig.

## Head-chain closure (2026-08-18, second pass)

The module's "nothing here selects it" note for `lm_head` is retired: the
chain exists end to end on this lineage and is now PINNED in both
directions (`test_ct_lmhead_chain_727.py`, mutation-proven):

1. `qwen3_vl.py:1298` hands the compressed-tensors quant_config to
   `ParallelLMHead` unconditionally;
2. `get_quant_method` matches the head via
   `isinstance(layer, VocabParallelEmbedding)` gated on the checkpoint's
   own ignore list — the arm-B/C discriminator (`-embed` keeps `lm_head`
   ignored -> dense; `-both` drops it -> int8), both directions pinned;
3. `create_weights` fires the GATE-0 `INT8-VOCAB ENGAGED` line with the
   layer class name (0/1/2 lines for A/B/C);
4. `should_apply_lm_head_quant_method`'s default arm routes the head
   matmul through `apply()` — pinned against the silent-garbage refactor
   (listing the method as unquantized reds 2 tests), and `apply()` is
   bit-identical to the dense reference on exact-int8 rows.

The A/B runner additionally reads the MTP **accept length** per arm
(optional `accept_len` in the suite JSON, sourced from `meta_info` per the
acceptance-measurement rule): near-tie flips decay accept before the suite
score moves, which makes it the sharpest logit-sensitivity instrument the
serving stack already exposes. Absent on either baseline boot, the
comparison is skipped, never invented.
