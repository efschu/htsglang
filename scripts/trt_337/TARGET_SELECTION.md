# Task #337 -- target selection, and why these five

The question is a user program: build TensorRT engines for **individual parts**
of the LLM, on the theory that lower launch overhead and higher kernel
efficiency beat our own kernels at the same precision, so the win is
quality-neutral. This file records which parts were chosen to test that, and
what each one is supposed to falsify.

The deployed model is `Qwen3.6-27B-INT8-W8A8` (compressed-tensors W8A8),
TP=3 uneven on the decode vector `[30,17,17]`: rank 0 on the 5090 (sm120),
ranks 1 and 2 on the 3080s (sm86).

## 0. The constraint that shapes everything

TensorRT cannot express our activation quantization. This is read out of the
installed library, not recalled:

```
tensorrt_rtx 1.6.1.120
IQuantizeLayer.__doc__:
  "The subgraph which terminates with the scale tensor must be a build-time
   constant."
IDynamicQuantizeLayer.__doc__:
  output_type "must be either DataType::kFP4 (NVFP4 quantization) or
   DataType::kFP8 (MXFP8 quantization)"
  block_size  "Supports block sizes of 16 with NVFP4 ... and 32 with MXFP8"
```

Our scale is per token and computed at runtime
(`compressed_tensors_w8a8_int8.py`: `x_q, x_scale = per_token_quant_int8(x)`).
It is not a build-time constant, and it is neither FP4 nor FP8 at block 16/32.

Rather than accept a precision deviation, the engine boundary is drawn **at the
quant**: `per_token_quant_int8` runs in every arm, outside every engine.
Arithmetically nothing is lost, because

```
out[m,n] = (sum_k x_q[m,k] * W_q[n,k]) * x_scale[m] * w_scale[n]
```

is reproduced exactly by DQ(x_q, 1.0) + DQ(W_q, w_scale, axis=0) + MatMul +
elementwise-multiply by `x_scale` -- the per-token scale enters as an ordinary
runtime tensor, which the build-time-constant rule does not touch. The weight
side is the deployed quantization verbatim: the checkpoint stores
`*.weight_scale` as `[N,1]` bf16, i.e. per output channel, which is exactly what
`IDequantizeLayer` with `axis=0` takes.

The consequence is the property the whole comparison rests on: **both arms carry
equal total work.** TensorRT is not credited for skipping a kernel it cannot
express.

The alternatives were considered and rejected, and are named here so nobody has
to rediscover them:

| config | what it is | why not |
|---|---|---|
| static per-tensor activation scale | build-time constant, real INT8 GEMM | less accurate than the deployed per-token scale; breaks the quality-neutrality that is the point of the program |
| weight-only INT8, bf16 activations | DQ weights, bf16 matmul | *more* accurate than deployed, and TensorRT skips the quant we pay -- flatters TensorRT with work it did not do |
| chosen: quant outside, scale as runtime operand | exact | none; it is what the table below uses |

## 1. The dominant per-rank GEMM shapes

Derived from the checkpoint's `config.json` and the deployed shard plan,
cross-checked at runtime against
`sglang.srt.distributed.utils._partition_units_raw` (the harness refuses to emit
a table that disagrees with what the ranks would actually build). These match
the #368 shape table, which was independently validated against the same
checkpoint.

**rank 0 -- 5090, sm120** (12 q heads, 2 kv heads, 8/24 GDN heads, local
intermediate 8160)

| stage | N | K | per token | module |
|---|---:|---:|---:|---|
| `mlp_gate_up` | 16320 | 5120 | 64 | `Qwen2MoeMLP.gate_up_proj` |
| `mlp_down` | 5120 | 8160 | 64 | `Qwen2MoeMLP.down_proj` |
| `attn_qkv` | 7168 | 5120 | 16 | `Qwen3_5Attention.qkv_proj` |
| `attn_o` | 5120 | 3072 | 16 | `Qwen3_5Attention.o_proj` |

**rank 1 -- 3080, sm86** (6 q heads, 1 kv head, 4/12 GDN heads, local
intermediate 4624)

| stage | N | K | per token | module |
|---|---:|---:|---:|---|
| `mlp_gate_up` | 9248 | 5120 | 64 | `Qwen2MoeMLP.gate_up_proj` |
| `mlp_down` | 5120 | 4624 | 64 | `Qwen2MoeMLP.down_proj` |
| `attn_qkv` | 3584 | 5120 | 16 | `Qwen3_5Attention.qkv_proj` |
| `attn_o` | 5120 | 1536 | 16 | `Qwen3_5Attention.o_proj` |

The MLP pair is the whole matrix's centre of gravity: 64 of 64 layers run it,
every token. `attn_qkv` is included because it is the widest attention
projection and carries the output-gate doubling, so it is the shape a vendor
tactic table is least likely to have been tuned for -- if TensorRT's tactic
selection is the win, this is where it should show up largest.

## 2. The targets

Five by default. Small enough to run in minutes, wide enough that a null result
is a real null.

| # | target | chain | what it falsifies |
|---|---|---|---|
| T1 | `gemm_mlp_gate_up` | quant -> engine | "a TensorRT tactic beats CUTLASS `int8_scaled_mm` at the shape we run most" |
| T2 | `gemm_mlp_down` | quant -> engine | same, on the shard-dependent K dimension |
| T3 | `gemm_attn_qkv` | quant -> engine | same, on the shape least likely to be pre-tuned |
| T4 | `chain_mlp_block` | quant -> gate_up+SiLU-mul -> quant -> down | "fusion pays": the SiLU-gate multiply and both dual-scale epilogues are TensorRT's to fuse, while both quants stay ours |
| T5 | `chain_decode_layer` | qkv -> bridge -> attn_o -> gate_up+SiLU-mul -> down, four quants interleaved | "the win survives at the granularity a real integration would use" |
| T6 | `chain_gdn_conv` (optional, off) | quant -> gdn_in_qkvz | the #325 fusion candidate |

T4 is the one that matters most for the program's thesis. T1-T3 price the GEMM
alone; if the answer to the program is "yes", T4 is where the margin should be
visibly bigger than T1+T2 measured separately, because that difference *is* the
fusion.

### What T5 deliberately excludes, on both arms

- **The attention core.** Paged-KV attention is a flashinfer kernel with no
  stock-TensorRT expression. A placeholder would price a fiction. The chain
  therefore carries a `bridge` stage that narrows the qkv output (7168) to the
  width o_proj consumes (3072); it runs identically in both arms and cancels out
  of every ratio.
- **RMSNorms and residual adds.** They sit upstream of an activation quant that
  has to stay outside the engine anyway. Putting them inside one arm only would
  be exactly the asymmetry this design exists to avoid.

What remains is the INT8 linear chain a decode layer actually issues: four
GEMMs, four activation quants, one SiLU-multiply. That is the part a per-part
engine could replace **today**, and nothing it could not.

### Why T6 is off by default

#325 has not fixed the conv+gating fusion boundary. A microbench of a boundary
that may move is a number with a short shelf life. It is one flag away
(`--optional gdn_conv`) when #325 settles.

## 3. Regimes

`M in {1, 2, 4, 8}`. Decode at bs=1 is the operating point that matters most and
the one where launch overhead dominates; bs=4 is the deliverable's second named
profile; 2 and 8 bracket them so a crossover is visible rather than inferred.
One engine carries two optimization profiles (opt=1 and opt=4) and the harness
selects and records the one it used, so no measurement lands on a kernel
specialised for the wrong batch size.

## 4. Weights

Real values from `Qwen3.6-27B-INT8-W8A8/model.safetensors`, sliced to the
per-rank shard extent, in both the engines and the torch arm -- the same loader,
so a tolerance difference can only come from the kernels and never from the
operands.

What is *not* reproduced: the head-index permutation inside a shard. It changes
neither GEMM cost nor the comparison (both arms consume the identical tensor).
Recorded so the slice is not later mistaken for a faithful rank reconstruction.

`--random-weights` gives shape-faithful randoms instead, CPU-sampled and moved
per the rig's cross-arch rule, so two cards see identical bytes.

## 5. The reading, and the number that must not be quoted

TensorRT is not an alternative to CUDA graphs. TensorRT-RTX wraps itself in one:
the installed runtime exposes `CudaGraphStrategy.WHOLE_GRAPH_CAPTURE` as a
first-class knob. TensorRT is fusion + tactic selection + precision
optimization, and *then* a CUDA graph.

#368 measured what the graph alone is worth on this rig: `int8_quant` costs
0.0266 ms eager and 0.0012 ms under graph replay, ~21x, and the quant's share of
the fused path falls from 61 % to 11 %. A "TensorRT vs our eager kernels" number
would be that 21x wearing a TensorRT label.

| cell | ratio | status |
|---|---|---|
| verdict | `trt_outer_graph / torch_graph` | **the answer** |
| our launch axis | `torch_eager / torch_graph` | diagnostic |
| TensorRT's launch axis | `trt_enqueue / trt_native_graph` | diagnostic |
| — | `trt_outer_graph / torch_eager` | **DO NOT QUOTE** -- double counts the graph |

The harness computes all four and labels the last one in the JSON, so nobody has
to recompute it to see the trap.

A verdict ratio inside the A-vs-A floor at that operating point is not a result;
`inside_noise_floor` is emitted next to it.
