# #325 — fusion audit of the GDN decode chain: measured and refused

Desk phase. Tree `/spinning/wt-602-slot2`, branch `fix/602-fill-side`.
**No GPU was touched.** Dims are read from the production checkpoint
`/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8/config.json`;
every code claim is a file:line in this tree.

---

## 0 — Verdict

**Fusion: refused, on a counted threshold.** The stated gate was: build the top
fusion only if it saves **>10 % of the chain's bytes**. The best remaining
adjacency saves **0.31 %**, and a *perfect* fusion of the entire remaining
chain — every intermediate materialization removed — saves **0.99 %**.

That is because **95.22 % of the chain's bytes are the recurrent state, read
once and written once per token**. No *fusion* can remove it: fusing adjacent
kernels does not change how often S is loaded and stored.

**But the ticket's real premise — cut memory round-trips — does have a live
answer, and it is already in this tree.** See §7. Revision 1 of this note
called the state traffic "irreducible"; that was **wrong**, and the mechanism
that refutes it is referenced in the very function this audit enumerated.
The fusion verdict stands; the "nothing left to cut" framing does not.

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

## 6 — The falsifier I named, and the one I missed

Revision 1 named the correct falsifier and then failed to check whether it was
already satisfied. It said: the verdict would be overturned by "a formulation
that does not read and write the full state per token — e.g. a low-rank or
blocked state update where the touched slice is a fraction of S", and dismissed
it as "an algorithm change, not a fusion, and it would not be lossless."

The first half is right and the second half was asserted without checking.
That formulation exists, it is in this tree, and §7 is it.

The other adjacent question stands and is still unpriced: at bs=1 the chain is
**192 memory-bound launches per round**, so it may be launch-bound rather than
byte-bound. This audit prices bytes, which is what #325 asked for. Note that
ReplaySSM's own headline is quoted at "batch >= 64", which is consistent with
bytes not being the binding constraint at bs=1.

## 7 — ReplaySSM: the lever that does clear the gate, already in the tree

`--enable-linear-replayssm` (`server_args.py:4145`, **default `False`**), with
`--linear-replayssm-cache-len` (`:4156`, default **16**).

**Mechanism**, from the kernel's own header
(`layers/attention/fla/fused_recurrent_linear_replayssm.py:14-23`): the plain
packed decode reads the full state S and writes it back *every* step.
ReplaySSM keeps a per-slot ring buffer of the last L steps' `(d, k, g)` and
only WRITES the full state every L steps; on non-flush steps it appends a tiny
record and reconstructs the readout from the checkpoint `S0` plus the buffer.
`S0` is still read every step, so per-step state traffic goes from read+write
to **read-only — "roughly halved"** in the header's own words.

Against §2's table that attacks the 47.61 % write term directly, i.e. the
single largest line in the chain — five times the >10 % gate that the fusion
candidates failed. The net is not the full 47.61 %: the ring buffer must be
appended each step and re-read on reconstruction, which is the offsetting term
that makes the header say "roughly halved" rather than "write eliminated". A
precise net is a count this audit has **not** done.

**Claimed benefit** (`server_args.py:4147-4149`): "~1.2-1.5x at batch >= 64",
GDN scalar-gate only. Explicitly **not recommended for KDA** — the per-K
`g_cache` is K times larger and the reconstruction refolds the per-K decay
every step, making KDA decode *slower* than the packed baseline.

**Preconditions**: the Triton linear-attn decode backend and
`--mamba-scheduler-strategy no_buffer` (the default).

**The open gate, and why it belongs to this queue.** The flag text claims the
kernel is "numerically correct". It does **not** claim byte-identity, and the
reconstruction sums buffered rank-1 updates in a different order than the
sequential update, so byte-identity should be assumed absent until measured.
#325 sits on the **lossless**-perf queue, and the standing rule is that lossy
features come last, after byte-identical wins. So the next step is not a new
kernel — it is:

1. establish byte-identity status of ReplaySSM on the GDN path (A-vs-A on
   CPU-sampled inputs, per the cuda-randn rule);
2. determine how far the wiring actually goes. `gdn_backend.py:497-507` passes
   `replayssm_d/k/g`, `replayssm_write_pos` and `replayssm_force_flush` into
   `packed_decode`, while the kernel header still says it is "NOT yet wired
   into the memory pool / radix cache / scheduler / backend dispatch". One of
   the two is stale. That contradiction must be resolved before any arm.

**Provenance**: upstream RFC `sgl-project/sglang#28511` covers the same
mechanism and reports only **~2.3 % end-to-end TPOT** on Qwen3.5-35B-A3B,
because a MoE-heavy model dilutes a GDN-kernel win. Our production checkpoint
is also MoE, so the end-to-end expectation here should be set from that number
and not from the 1.2-1.5x kernel figure.

**Arm, if and only if the byte-identity gate passes**: control
`--enable-linear-replayssm` off vs on at default L=16, GDN path, batch >= 64
(the regime where its own help text places the win), measured as ms/round
split compute vs wait. `layers/attention/fla/bench_gdn_replayssm_decode.py`
already exists and should be read before any harness is written.
