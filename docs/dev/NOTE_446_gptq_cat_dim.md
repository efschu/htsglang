# NOTE 446 — `cat_dim` when fusing `q_a_proj` + `kv_a_proj_with_mqa`

Layout analysis behind the `fuse_q_kv_a_proj` helper in
`python/sglang/srt/models/deepseek_common/utils.py`. Everything below is read
off `create_weights` in this tree, not analogized from another project.

## What the fusion is

MLA models with a `q_lora_rank` build one `ReplicatedLinear` named
`fused_qkv_a_proj_with_mqa` with
`out = q_lora_rank + kv_lora_rank + qk_rope_head_dim`
(`deepseek_v2.py:1705`). The checkpoint stores it as two separate
projections — `q_a_proj` and `kv_a_proj_with_mqa` — so the weight loader
caches both and concatenates them along the OUTPUT axis before handing the
result to the destination parameter's `weight_loader`.

Which axis of the on-disk tensor that is depends on the quantization format.

## Copies of the enumeration (all four, at base 692992ec59)

| file | line |
| --- | --- |
| `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | 368 |
| `python/sglang/srt/models/longcat_flash.py` | 996 |
| `python/sglang/srt/models/longcat_flash_nextn.py` | 629 |
| `python/sglang/srt/models/bailing_moe_linear.py` | 1545 |

All four carried the identical body:

```python
cat_dim = 0
if self.quant_config is not None and (
    self.quant_config.get_name() == "awq"
    or self.quant_config.get_name() == "awq_marlin"
    or self.quant_config.get_name() == "moe_wna16"
):
    cat_dim = 1
```

(`lora.py` also computes a `cat_dim`, for merging LoRA A/B stacks. Unrelated.)

## Tensor layouts, per format

Read off `create_weights`; `pack` is the 32-bit pack factor (8 for 4-bit).

| format | parameter | shape | `output_dim` | source |
| --- | --- | --- | --- | --- |
| unquantized | `weight` | `[out, in]` | 0 | `unquant.py:165` |
| fp8 (block) | `weight` | `[out, in]` | 0 | `fp8.py:641` |
| fp8 (block) | `weight_scale_inv` | `[out/128, in/128]` | 0 | `fp8.py:666` |
| gguf | `qweight` | `[out, packed_in]` | 0 | `gguf.py:1131` |
| gguf | `qweight_type` | 0-d in checkpoint | — | `gguf.py:1147` |
| compressed-tensors wNa16 | `weight_packed` | `[out, in/pack]` | 0 | `compressed_tensors_wNa16.py:131` |
| compressed-tensors wNa16 | `weight_g_idx` | `[in]` | none | `compressed_tensors_wNa16.py:200` |
| awq | `qweight` | `[in, out/pack]` | 1 | `awq/schemes/awq_linear.py:54` |
| awq | `qzeros`, `scales` | `[in/group, out/pack]`, `[in/group, out]` | 1 | same |
| awq_marlin | `qweight`, `qzeros`, `scales` | as awq | 1 | `awq/schemes/awq_marlin.py:62` |
| **gptq** | `qweight` | `[in/pack, out]` | **1** | `gptq/schemes/gptq_linear.py:80` |
| **gptq** | `qzeros`, `scales` | `[·, out/pack]`, `[·, out]` | **1** | same |
| **gptq** | `g_idx` | `[in]` | **none** | `gptq/schemes/gptq_linear.py:93` |
| gptq_marlin | `qweight`, `qzeros`, `scales`, `g_idx` | as gptq | 1 / none | `gptq/schemes/gptq_marlin.py:94` |

Marlin variants keep the AutoGPTQ / AutoAWQ on-disk layout at load time; the
repack to Marlin's own format happens in `process_weights_after_loading`, so
the concatenation sees the same layout as the non-Marlin scheme.

## What the enumeration got wrong

GPTQ's `qweight` is `[in/pack, out]` — output features on axis 1, exactly like
AWQ. The list did not name it, so `cat_dim` stayed 0 and the concatenation ran
along the INPUT axis. For any real MLA geometry `q_lora_rank !=
kv_lora_rank + qk_rope_head_dim`, so `torch.cat(..., dim=0)` raises
`RuntimeError: Sizes of tensors must match except in dimension 0`. GPTQ-family
MLA checkpoints were therefore not loadable through this path at all — loud,
never silently wrong.

`moe_wna16` is in the list and stays effectively correct: it wraps an AWQ or a
GPTQ checkpoint, and both have output on axis 1.

## `g_idx` is not a concatenation

`g_idx[i]` is the quantization-group index of INPUT channel `i`. It has no
output axis (`RowvLLMParameter` with `input_dim=0` only). Both projections read
the same hidden state, so both vectors have length `hidden_size`, and the fused
layer holds exactly one of them.

* Concatenating along axis 0 doubles the length and describes input channels
  the layer does not have. The destination `weight_loader` then rejects it on
  the shape assert (`linear.py:453`) — again loud, not silent.
* `desc_act=False`: `g_idx` is the trivial ramp `i // group_size`, identical
  for both projections. The fused layer takes that one vector.
* `desc_act=True`: each projection was quantized independently, so each got
  its own activation-order permutation. The fused layer has one `g_idx` and
  cannot satisfy two different permutations at once — the fusion has no
  correct answer.

## Implemented

`fuse_q_kv_a_proj(param, param_name, q_a_proj_weight, kv_a_proj_weight)`:

1. 0-d pair → return one value (GGUF `qweight_type`, AutoFP8 per-tensor
   scale). Behavior carried over unchanged from the DeepSeek loader.
2. `output_dim is None and input_dim is not None` → per-input-channel vector.
   Return one copy if the two agree, otherwise raise
   `UnfusableAProjParameter` naming the parameter, both shapes and `desc_act`.
3. Otherwise `torch.cat(..., dim=output_dim)`, defaulting to 0 when the
   destination records no `output_dim`.

Reading `output_dim` off the destination parameter is exact for every format
present and cannot go stale when one is added, which is the same move #443
made for packed-vs-dense classification.

Two incidental changes came with unifying the four copies:

* `longcat_flash`, `longcat_flash_nextn` and `bailing_moe_linear` did not have
  the 0-d branch that `deepseek_weight_loader` had; they now do. Previously
  those three raised on a 0-d pair (`torch.cat` rejects 0-d tensors).
* In `bailing_moe_linear` the `param_name not in params_dict: continue` check
  now runs before the fusion instead of after it, because the fusion needs the
  destination parameter. A skipped parameter is no longer fused first.

## Open design fork, for the operator

**The `desc_act=True` refusal is a refusal, not support.** A GPTQ MLA
checkpoint quantized with `desc_act=True` is now rejected at load time with a
named error instead of failing on a shape assert somewhere downstream. The
alternative — actually supporting it — means not fusing at all for that case:
keep `q_a_proj` and `kv_a_proj_with_mqa` as two separate layers so each keeps
its own `g_idx`, and give `DeepseekV2AttentionMLA` a two-GEMM path for it. That
is a real feature, not a fix, and it costs the fused-GEMM win on every other
checkpoint if done by dropping fusion globally.

Recommendation: leave the refusal. `desc_act=True` is uncommon in published
MLA checkpoints and the alternative is a structural change to the attention
module. Revisit only if a real checkpoint shows up.

**Not addressed here:** the 0-d branch keeps `q_a_proj`'s value and discards
`kv_a_proj_with_mqa`'s. For a GGUF `qweight_type` that is right — one type per
fused layer. For an AutoFP8 per-tensor scale it is only right when the two
scales are equal; the correct fusion would be `max(...)` plus a rescale of the
smaller half. That behavior predates this task and is unchanged by it.

## Validation

Desk-only, no GPU. `test/registered/unit/models/test_glm_packed_layer_classification_446.py`
builds the real `fused_qkv_a_proj_with_mqa` for each quantization on the meta
device, splits every registered parameter back into the two checkpoint tensors
it is assembled from, and asserts the fusion reproduces the destination shape.
No real GPTQ MLA checkpoint was loaded — that step is BOOT-PENDING.
