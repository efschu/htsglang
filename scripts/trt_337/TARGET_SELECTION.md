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

## 2b. The fold variants, and the stair

Everything above assumes the deployed INT8 storage. There is a second, more
interesting option: **fold the dequantize into the weights at build time.**
Store `w_q * w_scale` as bf16 or fp16 in the engine, and the per-token
activation quantization has nothing left to do -- so it disappears entirely.

That changes the structure, not just the numbers. The INT8 arms are per-stage
engine *islands* with our quant kernel between them, because TensorRT cannot
express a per-token dynamic scale. Folded, the constraint is gone: the whole
part becomes **one engine, one graph node, with nothing of ours inside it**.
`chain_decode_layer` folded is a single engine containing four GEMMs, the
SiLU-gate multiply and the attention-core bridge.

### Quality: measured, not asserted

The fold does not coarsen the weights -- `w_q * w_scale` is the checkpoint's
weight exactly, just in a wider container, and every value stays a representable
point of the original INT8 grid. bf16 carries 8 mantissa bits and fp16 carries
11, against a ~7-bit weight grid plus a per-channel exponent. And the fold
*removes* the per-token activation quantization error the deployed path pays.

So the harness measures every path against the same exact fp32 reference
computed from the dequantized weights, and gates a fold on being **at least as
accurate as what we deploy** -- not on a fixed tolerance. Comparing a fold to
the deployed *output* would measure the deployed path's own quantization error
and call it a fold defect.

Desk mock, random weights, `gemm_mlp_gate_up` M=1 (directional only -- CPU
stubs):

| path | max-abs error vs exact |
|---|---:|
| deployed INT8 | 0.0112 |
| fold bf16 | 0.0031 |
| fold fp16 | 0.0024 |

### The fp16 caveat the mock found

fp16 has the better mantissa (11 bits vs 8) but a much smaller exponent range --
5 bits, max representable 65504. bf16 carries fp32's exponent range. In the
4-GEMM `chain_decode_layer` the intermediate magnitudes compound, and the mock's
fp16 fold produced a **non-finite output** where bf16 did not.

The mock uses random weights, which make intermediates larger than the
checkpoint's would be, so this is a flag rather than a verdict -- the card run
with real weights decides. But it qualifies the "fp16 rounding 2^-11 beats the
INT8 grid" rationale precisely: fp16 wins on mantissa and loses on range, and
the deep chains are exactly where range matters. The harness detects non-finite
output per arm and records it as a result (`quality.arms.*.overflow`) instead of
letting a NaN pass as a number.

### The stair: fold is per-part, and that is the design

Full-model fold does not fit. Every fold engine is measured at **1.999x** the
INT8 weight bytes (2 bytes vs 1 byte plus the per-channel scale), and the 3080s
have 20 GB with a shard of a 27B model already in them. That is not an argument
against the fold -- it is the reason the fold is a **stair**: fold the hot parts,
leave the rest INT8, and the untouched parts stay offloadable exactly as they
are today.

Per-part cost, measured from the built engines (rank 0 = 5090, rank 1 = 3080):

| target | rank 0 fold MB | rank 0 INT8 MB | rank 1 fold MB | rank 1 INT8 MB |
|---|---:|---:|---:|---:|
| `gemm_mlp_gate_up` | 167.1 | 83.6 | 94.7 | 47.4 |
| `gemm_mlp_down` | 83.6 | 41.8 | 47.3 | 23.7 |
| `gemm_attn_qkv` | 73.4 | 36.7 | 36.7 | 18.4 |
| `chain_mlp_block` | 250.7 | 125.4 | 142.0 | 71.1 |
| `chain_decode_layer` | 355.5 | 177.8 | 194.5 | 97.3 |
| `gemm_mlp_gate_up` fp32_ref | 334.2 | 83.6 | 189.4 | 47.4 |

These are per **one** layer's worth of that part. The MLP parts run in all 64
layers and the attention parts in 16, so the planner can price residency for any
subset of the stair from this table plus the layer counts in section 1. Every
number is in `engines/manifest.json` as `weight_bytes_fold`,
`weight_bytes_int8` and `memory_multiplier_vs_int8`, per engine, so the pricing
does not depend on this table staying current.

### The crossover question

The fold trades **zero activation-quant kernels** against **2x the weight bytes
moved**. At M=1 the GEMM is memory-bound GEMV: bytes dominate and the fold
should lose on bandwidth while winning on kernel count. As M grows the weight
bytes amortise across more rows. #368 measured the quant at 78 % of M=1 *eager*
INT8 linear time -- but graph replay already cuts its share to ~11 %, so under
graphs the fold is spending 2x bandwidth to win back a much smaller slice than
the eager number suggests. Which way that lands, and **where it crosses**, is
what decides which parts get folded. That is why the fold arms are measured
under graphs against `torch_graph`, at M = 1, 2, 4, 8.

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
