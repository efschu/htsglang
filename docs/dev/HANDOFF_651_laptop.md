# Handoff #651 — Qwen3.6-35B-A3B-UD-Q4_K_XL GGUF bring-up (laptop target)

Branch: `feat/gguf-q4-bringup-651` (off `integration/r3-probe-next2` @ `6c1e5cafb7`)
Author of this package: the #651 desk strand, 2026-08-07.

---

## 0. READ THIS FIRST — verification status

**No GPU ever ran any of this.** The strand that produced this package was
briefed onto the wrong machine (the main rig) and correctly never got cards;
the ticket targets the laptop. Everything below is therefore in exactly one of
two states, and the file says which for every claim:

| Marker | Meaning |
|---|---|
| **[DESK-PROVEN]** | Executed and checked on CPU against the real 22.85 GB checkpoint, or verified by reading the code in this tree at `6c1e5cafb7`. Reproduction command given. |
| **[UNVERIFIED]** | Never executed anywhere. Reasoning, arithmetic, or code-reading only. Treat as a hypothesis. |

In particular: **the model has never been loaded onto a GPU, has never
generated a token, and its coherence is UNVERIFIED.** Do not read "the fix is
proven" as "the model works" — the fix is proven at the tensor level, which is
a much smaller claim.

---

## 1. The checkpoint

```
/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf     22,853,663,008 B  = 21,795 MiB
  config.json  chat_template.jinja  tokenizer.json  generation_config.json
```

The Q3_K_M sibling named in the original brief **does not exist on disk**; only
this Q4 directory is present. There is no smaller fallback without a new
download.

### 1.1 Geometry — cross-checked, no sibling-config trap [DESK-PROVEN]

The GGUF header and `config.json` agree on **every** axis. This mattered: an
unvalidated sibling config once produced a 64-of-96-layer build that booted and
served nonsense.

| Quantity | GGUF header | config.json |
|---|---|---|
| blocks / layers | `block_count` 41 | 40 + `mtp_num_hidden_layers` 1 |
| hidden | `embedding_length` 2048 | `hidden_size` 2048 |
| attn heads / KV heads | 16 / 2 | 16 / 2 |
| head dim (K and V) | `key_length`/`value_length` 256 | `head_dim` 256 |
| experts / active | 256 / 8 | 256 / 8 |
| expert FFN | `expert_feed_forward_length` 512 | `moe_intermediate_size` 512 |
| full-attn interval | 4 | `layer_types` every 4th |
| SSM | inner 4096, state 128, groups 16 | 32 v-heads x 128 = 4096 |
| MTP | `nextn_predict_layers` 1 | `mtp_num_hidden_layers` 1 |

`general.architecture = qwen35moe`; `model_type = qwen3_5_moe`, text config
`qwen3_5_moe_text`. Block 40 is the NEXTN/MTP draft block. Reproduce with
`/tmp/gguf_probe.py`-style `GGUFReader` dump (see §8).

### 1.2 The file is TEXT-ONLY [DESK-PROVEN]

`config.json` declares a 27-layer vision tower, but the GGUF contains **zero**
vision tensors and there is **no `mmproj*.gguf`** beside it. The fork already
handles this: `model_config.py` force-disables multimodal for GGUF without an
mmproj, which closes the #52 NaN-contamination path (an uninitialized vision
tower fed by the automatic VLM image warmup poisons recycled mamba slots).
Confirmed at runtime on CPU: the adapter reports `mmproj: None`.

**But see §7.1** — the vision tower module is still *constructed*, costing
~818 MiB of never-used dense weights per rank. On a laptop that is real money.

---

## 2. The load-bearing fix: #647 on this checkpoint

Commit **`0155ff2c00`** — "[#651/#647] GGUF: restore the dense name for non-F32
MoE router gates".

### 2.1 What is wrong [DESK-PROVEN]

`weight_utils.py:1517`:

```python
if weight_type.name != "F32":
    name = gguf_quantized_name(name, "qweight")
```

The `.weight` -> `.qweight` rename is keyed on **tensor dtype**, which the code
treats as a proxy for "the destination module is quantized". Those are
different statements. A MoE router gate is never quantized —
`qwen2_moe.py:408` and `:459` build both `mlp.gate` and
`mlp.shared_expert_gate` with `quant_config=None`, so each owns a dense
`.weight` and has **no `.qweight` at all**. A non-F32 gate is therefore renamed
onto a parameter that does not exist.

Two things kept this invisible:

1. Published GGUFs almost always store router gates **F32**, and F32 is the one
   type the rename never touches.
2. BF16 arrives from `gguf-py` as **raw `uint8` with the last dimension
   doubled**. That also fails `Tensor.is_floating_point()`, so it slips past the
   dense-shard rescue in `gguf.py:_cast_dense_qweight`. **F16 survives the
   misroute** (it is a real float array and gets cast); **BF16 does not.** That
   asymmetry is the sting.

### 2.2 Why this checkpoint specifically [DESK-PROVEN]

Of its **753 tensors, exactly two are BF16** — and both are router gates in the
MTP block `blk.40`:

```
blk.40.ffn_gate_inp.weight        BF16  (2048, 256)   -> mtp.layers.0.mlp.gate.weight
blk.40.ffn_gate_inp_shexp.weight  BF16  (2048,)       -> mtp.layers.0.mlp.shared_expert_gate.weight
```

All 40 base-layer gates are F32 and unaffected. So on this file the **NEXTN
draft's router never loads**.

### 2.3 Why it matters more than a crash

A garbage router still routes every token to *some* expert. The model stays
fluent and grammatical and is quietly wrong. On the draft side the fork's
`raise_on_unloaded_draft_parameters` (#514/#505-A1-01) would likely have caught
it as a hard boot failure — but **on the target side an unloaded parameter is
only a `logger.warning`**. Had these two gates been in a base layer instead of
the MTP block, this would have been silent wrongness with no error at all.
Expected symptom if you run WITHOUT the fix: either a draft-parameter abort, or
an unexplained acceptance-rate collapse that looks like a bad draft head.

### 2.4 The fix, and its proof

Mirrors the identical fix already present for DeepSeek-V4
(`gguf_deepseek4.py:366-376`). In `Qwen35GGUFAdapter.transform_stream`:
restore the dense `.weight` name for the two anchored gate spellings, re-view
the BF16 bytes as `bfloat16`, and drop the stray `.qweight_type` the dense
module cannot receive. The branch **reassigns and falls through** rather than
yielding, so the 1-D `shared_expert_gate` still reaches the existing
`unsqueeze(0)`. Suffixes are anchored (`.mlp.gate.` / `.shared_expert_gate.`)
so the genuinely quantized `mlp.gate_proj` is not captured.

**[DESK-PROVEN]**
- New `test/registered/unit/quantization/test_gguf_qwen35_router_gate_dtype.py`: **7/7 pass**.
- **Can-fail proof**: with the suffix table neutralized, **3 of 7 fail**; the
  other 4 are inertness/documentation cases and correctly stay green.
- GGUF unit cluster (mtp expert names, moe expert-id sanitize, draft
  quantization, dflash sibling config, + the new file): **33 passed**.
- On the **real checkpoint**, through the real adapter and the real iterator,
  CPU only: both gates now arrive dense and correct —
  `mtp.layers.0.mlp.gate.weight` bf16 `(256, 2048)` and
  `mtp.layers.0.mlp.shared_expert_gate.weight` bf16 `(1, 2048)`, all finite,
  std `0.0096` / `0.0020` (plausible trained-router scale, not byte noise).

**[UNVERIFIED]** that the fixed draft actually improves acceptance on-GPU, or
that the model is coherent at all.

### 2.5 Full weight-stream audits [DESK-PROVEN]

Both streams were enumerated end-to-end on CPU against the real file.

**Target**: 63,841 tensors delivered, of which exactly **301** are dense
`.weight`. That number closes arithmetically:
`30 GDN layers x 4 + 10 attention layers x 2 + 40 layers x 4 + 1 final norm = 301`.
Every dense module gets `.weight` (`mlp.gate` f32 `(256,2048)`,
`shared_expert_gate` already unsqueezed to `(1,2048)`, norms, `conv1d`,
`in_proj_a/b`); every quantized module gets uint8 `.qweight`. **Zero orphans.**

**Draft**: 63,000 delivered; all **1,560** `mtp`-bound names present. The 61,440
dropped are the main model's experts
(`40 x 256 x 3 x 2`) — the generic iterator emits those by name pattern
regardless of the name map, and the draft loader discards them. Documented,
harmless, but it means **the draft load reads the whole 22 GB file to throw most
of it away** (boot-time cost, not correctness).

---

## 3. Parallelism: TP=2 is the natural width on BOTH axes

This is a planning fact with reach beyond this checkpoint.

### 3.1 MoE shard alignment [DESK-PROVEN by code reading]

`moe_intermediate_size = 512`. GGUF K-quant sharding must respect a **256-byte
block**, so the expert FFN splits into exactly **two** aligned units.

- **TP=2** -> `512 = 2 x 256`. Clean, block-aligned.
- **TP=3** -> cannot split 2 aligned units across 3 ranks. It does **not**
  crash: `qwen2_moe.py:_shared_expert_uneven_misaligned` detects this and builds
  the expert **REPLICATED** (full dimension on every rank, rank-0-only
  contribution). Its docstring uses `512 -> [229,141,141]` as the GGUF example.
  So TP=3 costs memory and yields no expert sharding.
- **TP=4** -> `128` per rank, not 256-aligned. Expect the same replication.

### 3.2 KV heads point the same way [UNVERIFIED, arithmetic]

`num_key_value_heads = 2`. TP>2 cannot split 2 KV heads across ranks without
replication or DCP. So both the MoE axis and the KV axis independently make
**TP=2 the clean tensor-parallel width for this model.**

### 3.3 Consequence for the "PP/TP" goal

Use **PP** to span more than two devices, and keep **TP at 2** (or 1). PP also
sidesteps both alignment problems entirely, since it splits by *layer*.
`qwen3_5.py` has full PP support (`pp_group`, `PPMissingLayer`, `make_layers`
with `pp_rank`/`pp_size`) and there is no GGUF-x-PP guard in the loader.
**[UNVERIFIED]** — PP has never been run with this checkpoint.

### 3.4 NEXTN/MTP prerequisites already satisfied [DESK-PROVEN by code reading]

- The MTP class unwraps `config.text_config` (`qwen3_5_mtp.py:64-66`), whose
  `model_type` is `qwen3_5_moe_text`, so the draft block builds **MoE** layers
  matching `blk.40` — not a dense MLP.
- The draft path is **mandatory** for this family: pass
  `--speculative-draft-model-path <the same .gguf>`. There is no auto-default.
- The GGUF vocab is **quantized-resident** by default (`embed_tokens` and
  `lm_head` arrive as uint8 `.qweight`). The NEXTN draft therefore shares the
  target's embedding/head as **modules** (`set_embed_and_head_modules`), not as
  raw `.weight` tensors. This is already wired; it is why a packed head works.
- `--model-path` must be the **`.gguf` FILE**, not its directory
  (`_prepare_weights` requires `os.path.isfile`; commit `d274bbe9ce`).

---

## 4. VRAM budget — the number that decides the configuration

All figures **[DESK-PROVEN]** (measured from the checkpoint / derived from the
verified geometry). Reproduce with `docs/dev/651/vram_budget.py`.

### 4.1 Weights, by category (total 21,784 MiB)

| Category | MiB | % | Shards under TP? |
|---|---:|---:|---|
| routed experts | 18,726 | 86.0 | yes (intermediate; see §3.1) |
| vocab (`token_embd` + `output`) | 1,031 | 4.7 | vocab-parallel |
| full attention | 787 | 3.6 | yes |
| GDN linear-attn | 529 | 2.4 | yes |
| **MTP draft block (blk.40)** | **504** | 2.3 | yes |
| shared expert | 128 | 0.6 | yes / replicated |
| router gates | 80 | 0.4 | replicated |
| norms | 0.3 | 0.0 | replicated |

**86% of the model is routed experts.** That single fact drives everything
below.

### 4.2 Per-token and per-sequence costs

Only **10 of 40** layers are full attention (indices 3, 7, ... 39); the other 30
are GDN linear attention and hold no KV cache.

| Item | Cost |
|---|---|
| KV per token, all ranks, **fp16/bf16** | **20.00 KiB** (160 MiB @ 8k ctx; 5,120 MiB @ 262k) |
| KV per token, all ranks, **fp8_e4m3** | **10.00 KiB** (80 MiB @ 8k ctx; 2,560 MiB @ 262k) |
| GDN state per **sequence**, float32 | **61.9 MiB** (ssm 60.0 + conv 1.9) |
| GDN state per **sequence**, bfloat16 | **31.9 MiB** (ssm 30.0 + conv 1.9) |
| Vision tower, **never used** | **818 MiB per rank** (see §7.1) |

Note the GDN state is charged **per concurrent sequence**, so
`--max-mamba-cache-size` / `--max-running-requests` multiply it. At fp32 and 8
concurrent sequences that is ~495 MiB. Set `SGLANG_MAMBA_SSM_DTYPE=bfloat16`
to halve it (the production rig does).

### 4.3 The decision rule

Add, for a candidate configuration:

```
per_rank = weights_total/ranks_that_shard        (~21,784 / N for TP or PP)
         + 818                                    (vision tower, until §7.1 is fixed)
         + KV_per_token * context * (1/kv_ranks)
         + GDN_per_seq * max_concurrent_seqs
         + CUDA context + activations + dequant scratch   (see §4.4)
```

Rough landing points **[UNVERIFIED]**:

| Laptop VRAM | Verdict |
|---|---|
| 1x 16 GiB | **Does not fit.** Weights alone are 21.3 GiB. Needs expert offload -> see §5. |
| 1x 24 GiB | Fits only just: 21.3 + 0.8 vision + small KV. Tiny context, no headroom. Drop the vision tower first (§7.1). |
| 1x 32 GiB | Comfortable at TP=1, room for spec and real context. |
| 2x >=12 GiB | Comfortable via TP=2 or PP=2 (~10.9 GiB weights/rank). This is the configuration the goal ("PP/TP") actually wants. |

### 4.4 Dequant scratch — the term I could not price [UNVERIFIED]

For K-quants this fork runs MMQ only up to `SGLANG_GGUF_MMQ_MAX_TOKENS`
(default **8**) and dequantizes to cuBLAS above that. The dequant workspace is
therefore a **prefill-batch-dependent transient**, not a constant, and I could
not derive it without running. If you hit OOM only during prefill and not at
load, this is the first suspect — lower `--max-num-batched-tokens` or
`SGLANG_GGUF_MMQ_MAX_TOKENS` before touching anything else.

---

## 5. #123 — does expert offload land on the critical path?

**Yes, if the laptop has a single GPU under ~24 GiB. Otherwise no.**

The reasoning is the 86% figure in §4.1: the routed experts are 18,726 MiB of a
21,784 MiB model. Any VRAM plan that has to shed more than ~3 GiB **must** shed
experts, because there is nothing else big enough to give.

And #123 says GGUF-MoE has **no expert-offload half**. This is not a flag: the
expert parameter is a `GGUFUninitializedParameter` that only takes real shape in
the loader postprocess, so the offload machinery has nothing to grab
(`expert_offload.py:1245`, `:1447`, `:2130` all name this wall explicitly, and
`planner/rejected.py:289` records it as a rejected shortcut). **Building it is a
real rebuild, not a configuration change.**

So the laptop side must answer *before starting*: is there ≥24 GiB on one
device, or ≥2 devices? If neither, the first task is not the bring-up — it is
#123.

**[UNVERIFIED]** I could not confirm the wall still stands as described; I read
it in this tree but did not test it. Ticket state in this project has been stale
repeatedly — verify before committing to a rebuild.

---

## 6. How to boot it — staged, one variable at a time

Scripts are in `docs/dev/651/`. Copy them to the laptop and adjust the paths at
the top. Staging is deliberate: each stage adds exactly **one** variable, so a
failure names its own cause instead of leaving three suspects.

| Stage | Config | Answers |
|---|---|---|
| **a** | TP=1, no spec, eager | loader / kernels / checkpoint |
| **b** | TP=1, **NEXTN spec** | the #647 router-gate fix on-card |
| **c** | TP=2, spec | tensor parallelism (§3.1) |
| **d** | PP=N, spec | pipeline parallelism |

```bash
STAGE=a TP=1 SPEC=0 DEVICES=5090 PORT=30040 ./boot.sh
STAGE=b TP=1 SPEC=1 ./boot.sh
STAGE=c TP=2 SPEC=1 DEVICES=all ./boot.sh
STAGE=d PP=2 SPEC=1 DEVICES=all ./boot.sh
```

### Every flag, justified

| Flag | Why |
|---|---|
| `--model-path <....gguf>` | the FILE, not the dir — `_prepare_weights` requires `os.path.isfile` (`d274bbe9ce`) |
| `--tokenizer-path <dir>` | tokenizer/config live beside the GGUF; the sibling `config.json` is also how the arch is resolved (transformers rejects the GGUF arch directly) |
| `--load-format gguf --quantization gguf` | both are needed; quantization alone is not enough |
| `--tp-size` / `--pp-size` | see §3 — keep TP at 2 or 1 |
| `--context-length 8192` | start small; KV is 20 KiB/token (§4.2), 262k would cost 5 GiB at fp16 |
| `--max-running-requests 1` | GDN state is charged per sequence, 62 MiB each at fp32 |
| `--mem-fraction-static 0.90` | leave headroom; raise only after a clean boot |
| `--disable-cuda-graph` (stages a/b) | faster boot and more forgiving; **turn graphs ON before believing any perf or stability claim** — graphs are where #113-class draft-capture bugs surface |
| `--disable-custom-all-reduce` (TP>1) | needed where the GPUs have no P2P; harmless warning otherwise |
| `--speculative-algorithm NEXTN` + `--speculative-draft-model-path <same .gguf>` | mandatory for this family, no auto-default (§3.4) |
| `SGLANG_MAMBA_SSM_DTYPE=bfloat16` | halves GDN state (§4.2) |
| devices by **NVML UUID** | torch's device order and NVML's diverge, and NVML order itself shifts across boots. `boot.sh` resolves UUIDs at runtime and never hardcodes an index. |

---

## 7. Coherence protocol — judge CONTENT, never HTTP 200

`docs/dev/651/probe.py`, run as `python probe.py 30040`.

This is not ceremony. The specific failure mode of this checkpoint — an
unloaded router gate or a misfiled expert shard — produces **fluent,
grammatical, wrong** text and a perfectly happy HTTP 200. So every probe has a
determined answer that is checked (Paris / 42 / 217 / Mars / 32 / ice density),
and the whole set runs **twice at temperature 0** to confirm greedy determinism.
The script exits non-zero unless every probe is content-correct in both rounds
**and** the two rounds are identical.

Additional gates worth running before declaring success:

1. **Turn CUDA graphs on** and re-probe. Eager only defers #113-class
   draft-capture bugs; it does not prove them absent.
2. **Check the acceptance rate** under NEXTN, and read it from `meta_info` —
   `spec_ema_accept_len` is NOT the acceptance length (known measurement trap).
   A near-zero acceptance rate with coherent output is the signature of a
   broken draft router, i.e. exactly what §2 fixes.
3. **Watch the boot log** for `logger.warning` about parameters not found. On
   the target side that warning is the *only* signal of an unloaded weight.

### 7.1 Separate item — the vision tower costs 818 MiB per rank for nothing

`qwen3_vl.py:1242` constructs `self.visual` **unconditionally**, with
`quant_config=None` (dense bf16), ~429M params = **818 MiB per rank**. On this
text-only checkpoint it is never fed: `model_config.py` force-disables
multimodal when a GGUF has no mmproj, so no image ever reaches it. It is pure
waste, and on a laptop fighting for VRAM it may be the difference between fits
and does-not-fit. Filed separately as `docs/dev/FINDING_651b_vision_tower_vram.md`.
Not fixed here — out of scope for the bring-up, and a change to a shared
multimodal path deserves its own review.

---

## 8. Reproducing the desk checks

All of these are CPU-only and need no GPU. From the worktree root with
`PYTHONPATH=<worktree>/python` (the worktree PYTHONPATH matters — without it you
silently test a different tree):

```bash
# unit tests + can-fail proof
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES="" python -m pytest \
  test/registered/unit/quantization/test_gguf_qwen35_router_gate_dtype.py -q

# VRAM budget table of §4
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES="" python docs/dev/651/vram_budget.py

# the real-file proof of §2.4: both draft gates dense, finite, right shape
#   builds the adapter with architectures[0]="Qwen3_5ForCausalLMMTP",
#   runs the REAL iterator + transform_stream over the real checkpoint
```

---

## 9. Open questions for the laptop side

Ordered by how much they change the plan.

1. **How many GPUs, and how much VRAM each?** This decides everything (§4.3),
   including whether #123 is a blocker (§5) and whether the "PP/TP" goal is even
   expressible — a single-GPU laptop cannot do PP or TP at all, and the goal
   would need restating.
2. **Is the 22 GB checkpoint present on the laptop**, and is there a smaller
   sibling there? There is no Q3_K_M on this rig despite the brief naming one.
3. **Does #123 still stand?** If the laptop is small-VRAM, verify before
   starting a rebuild (§5).
4. **Dequant scratch at the intended prefill batch size** (§4.4) — the one
   budget term I could not price without hardware.
5. **Does the laptop's GPU have the ggml K-quant kernels available?** On this
   rig they JIT fine on both consumer generations, and the "sm100-only" trap
   applies to AWQ/Marlin, not the ggml ops — but that is a rig observation, not
   a laptop one.
