# #325 — fusion audit of the GDN decode chain: measured and refused

Desk phase. Tree `/spinning/wt-602-slot2`, branch `fix/602-fill-side`.
**No GPU was touched.** Dims are read from the production checkpoint
`/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8/config.json`;
every code claim is a file:line in this tree.

---

## 0 — Verdict

**Refused, on a counted threshold.** The stated gate was: build the top fusion
only if it saves **>10 % of the chain's bytes**. The best remaining adjacency
saves **0.31 %**, and a *perfect* fusion of the entire remaining chain — every
intermediate materialization removed — saves **0.99 %**.

The reason is not that the code is clever. It is that **95.22 % of the chain's
bytes are the recurrent state, read once and written once per token**, which
is the GDN recurrence itself. No fusion can remove it: the algorithm is
defined as "load S, update S, store S". The premise of the ticket — "cut
memory round-trips" — is correct as a lever in general and simply has almost
no material left to cut here.

The premise that *no weight format can help* is also confirmed, and for the
same reason: on this chain the weights are 1.24 % of traffic.

## 1 — The actual chain on the decode path

Per `linear_attention` layer, one decode round. Entry is
`Qwen3GatedDeltaNet.forward` (`python/sglang/srt/models/qwen3_next.py:459`),
the linear-attention branch of `Qwen3HybridLinearDecoderLayer`.

| # | op | site | fused? |
|---|---|---|---|
| 1 | `in_proj_qkvz` (GEMM) | `qwen3_next.py:479` | — |
| 2 | `in_proj_ba` (GEMM) | `qwen3_next.py:480` | overlapped on `alt_stream` below 1024 tokens (`:474-481`) |
| 3 | `fused_qkvzba_split_reshape_cat` | `qwen3_next.py:469` | **yes** — split + reshape + cat in one kernel |
| 4 | `causal_conv1d_update` | `gdn_backend.py:510` | **yes** — activation is a kernel argument (`layer.activation`, `:515`) |
| 5 | `packed_decode` | `gdn_backend.py:522` | **yes** — see §3 |
| 6 | `FusedRMSNormGated` | `qwen3_next.py:514`, class from `fla/fused_norm_gate.py` | **yes** — norm + gate in one kernel |
| 7 | `out_proj` (GEMM) | `qwen3_next.py:518` | — |

**Launches per GDN layer: 7**, of which **4 are the memory-bound chain**
(3, 4, 5, 6); the other 3 are GEMMs. The checkpoint has **48
`linear_attention` layers of 64**, so the chain is **192 memory-bound kernel
launches per decode round**, plus 144 GEMMs.

## 2 — Bytes per GDN layer per token (decode, bs=1)

Dims: `hidden_size 5120`, `linear_num_key_heads 16 x 128`,
`linear_num_value_heads 48 x 128`, `linear_conv_kernel_dim 4`,
activations bf16 (2 B), `mamba_ssm_dtype float32` (4 B).
So `q_dim = k_dim = 2048`, `v_dim = 6144`, `conv_dim = 10240`,
state = `48 x 128 x 128 x 4 B = 3 MiB`.

| term | bytes | share |
|---|---:|---:|
| conv1d_update: read conv_state | 61,440 | 0.93 % |
| conv1d_update: write conv_state | 61,440 | 0.93 % |
| conv1d_update: read conv weights | 81,920 | 1.24 % |
| conv1d_update: read mixed_qkv | 20,480 | 0.31 % |
| conv1d_update: write mixed_qkv | 20,480 | 0.31 % |
| packed_decode: read mixed_qkv | 20,480 | 0.31 % |
| packed_decode: read a, b | 192 | 0.00 % |
| **packed_decode: READ ssm_state** | **3,145,728** | **47.61 %** |
| **packed_decode: WRITE ssm_state** | **3,145,728** | **47.61 %** |
| packed_decode: write core_attn_out | 12,288 | 0.19 % |
| norm_gate: read core_attn_out + z | 24,576 | 0.37 % |
| norm_gate: write out | 12,288 | 0.19 % |
| **total** | **6,607,040** | 100 % |

**Recurrent state: 6,291,456 B = 95.22 %. Everything else: 315,584 B =
4.78 %.**

## 3 — What is already fused (checked before proposing anything)

`packed_decode` (`gdn_backend.py:521-538`) is the one that matters. The code
states its own scope at `:519-520`, verbatim:

```
# Skip split + reshape + separate gating kernel by consuming
# the packed mixed_qkv directly in a single fused Triton kernel.
```

It dispatches to `fused_sigmoid_gating_delta_rule_update`
(`layers/attention/linear/kernels/gdn_triton.py:16-17`), called with
`use_qk_l2norm_in_kernel=True` (`:117`, `:131`, `:164`, `:197`, `:231`). So
one kernel already covers **split + reshape + sigmoid gating + q/k L2 norm +
the delta-rule scan + the state write**.

**It is live on this rig.** `gdn_triton.py:44`:
`supports_packed_decode: bool = not is_cpu() and not is_npu()` — CUDA sm86
takes it. This was checked specifically because a documented fusion that does
not reach our hardware would have been the actionable finding (the REACH
defect class); it is not the case here.

`python/sglang/srt/layers/attention/fla/` additionally carries
`fused_gdn_gating.py`, `fused_norm_gate.py`, `fused_recurrent.py`,
`fused_sigmoid_gating_recurrent.py`, `layernorm_gated.py`, `l2norm.py`.

## 4 — The fusable adjacencies, and what each would actually save

Two passes over the same tensor separable only by code structure:

| candidate | saved bytes | share of chain |
|---|---:|---:|
| conv1d -> packed_decode (drop the `mixed_qkv` write + re-read) | 40,960 | 0.62 % |
| packed_decode -> norm_gate (drop `core_attn_out` write + re-read) | 24,576 | 0.37 % |
| **both, i.e. a perfect single-kernel chain** | **65,536** | **0.99 %** |

The largest *single* move is 0.62 %; the ticket's "top fusion" is 0.31 % if
counted as one round-trip. Against a >10 % gate this is not close, and the gap
is a factor of ten to thirty, not a measurement-precision question.

Norm+scan and conv1d+activation, named in the ticket as candidates, are
**already fused** (§1 rows 4 and 6). There is no softmax in this chain — GDN
is linear attention; the gating is sigmoid and is already inside the scan
kernel.

## 5 — Why the verdict is robust

* **Batch size.** Each sequence carries its own `ssm_state`, so both the state
  traffic and `mixed_qkv` scale with bs; the ratio is bs-invariant. The conv
  *weights* (1.24 %) are the only per-layer constant, so they amortize as bs
  rises — larger batches make the state share **larger**, not smaller.
* **TP.** Heads shard across ranks, so state and activations shard together;
  the ratio is essentially TP-invariant. The weightless ranks drop the three
  GEMMs, which raises the memory-bound share and again strengthens the result.
* **State dtype.** If `mamba_ssm_dtype` were bf16 rather than fp32, the state
  share falls from 95.22 % to **90.9 %** — still an order of magnitude above
  everything fusion can touch. The verdict does not depend on that config
  value.

## 6 — The falsifier, and the arm nobody needs to run

This refusal rests on an algorithmic invariant, not on an inventory of
kernels, so it cannot be overturned by finding more fused kernels upstream.
What *would* overturn it:

* a formulation that does **not** read and write the full state per token —
  e.g. a low-rank or blocked state update where the touched slice is a
  fraction of S. That is an algorithm change, not a fusion, and it would not
  be lossless. Out of scope for a lossless-perf ticket.
* a measurement showing the chain is **launch-bound rather than byte-bound**
  at bs=1 — 192 memory-bound launches per round is not nothing. That is a
  different lever (fewer launches, e.g. CUDA-graph coverage or layer-batching)
  and a different ticket; this audit prices bytes, which is what #325 asked
  for. Flagging it as the one adjacent question worth its own count.

No A/B arm is defined, because nothing is being built. If a future arm is
wanted it must be gated on the launch-bound hypothesis above, and measured as
ms/round split compute vs wait — not as a byte-saving claim, which §2 has
already priced at under 1 %.
