# Task #447 — harvesting the llama.cpp DSV4 deltas for our DeepSeek-V4-Flash chain

Desk + download only, no GPU work. Branch `docs/447-llamacpp-dsv4-harvest`,
based on `integration/r3-probe-next2` at `8e4fe8d33a`.

Two trees are cited throughout:

* **ours** — `/spinning/wt-447-dspark` (this worktree).
* **llama.cpp** — throwaway shallow clone at
  `/root/.claude/jobs/1481bb40/tmp/llamacpp-447`, `master` at `221f0f6`
  (2026-08-02), which contains both PR #24162 (DSV4 support, merged
  2026-06-29) and PR #25784 (DSpark draft head, merged 2026-08-02). Paths
  below are relative to that clone.

Everything asserted here was read in one of those two trees or fetched from
the HuggingFace API in this session. Nothing is carried over from the task
brief unverified.

---

## 0 — Nomenclature, settled first

Three tokens that look alike and are not:

| token | meaning | defined at |
| --- | --- | --- |
| `DFLASH` | **our** spec-decode algorithm: a Qwen3-family draft head checkpoint | `python/sglang/srt/speculative/spec_info.py:44`, model `python/sglang/srt/models/dflash.py:1-4` |
| `dflash-draft` | **our** GGUF `general.architecture` string for that Qwen3 drafter | `python/sglang/srt/model_loader/gguf_dflash.py:53` |
| `dflash` | **llama.cpp's** model architecture for the DeepSeek-V4 *draft* family; "DSpark = DFlash + a semi-autoregressive Markov head and Confidence head" | `src/models/dflash.cpp:82`, `gguf-py/gguf/constants.py:1161` |

Our GGUF architecture string for the DeepSeek-V4 *target* is `deepseek4`
(`python/sglang/srt/model_loader/gguf_deepseek4.py:45-46, 163`); llama.cpp
calls the same target `deepseek4` too (`src/models/deepseek4.cpp`). The
collision is confined to the draft side, and it is lexical only: llama.cpp's
`dflash` and our `dflash-draft` are different checkpoint families that happen
to share a marketing name.

A second, worse trap, corrected in this task: **"DSpark layers 40-42"** was
used in our docs for two unrelated things. Resolved in §3.

---

## 1 — Item (a): the DSpark head

### 1.1 HF inspection, before downloading

`GET https://huggingface.co/api/models/am17an/DeepseekV4-Flash-20260731-DSpark?blobs=true`,
sha `9d79f20040120924bd2f7dc4f3a9f86c721b39f8`, lastModified 2026-08-02T14:45:06Z:

| file | bytes |
| --- | --- |
| `DeepseekV4-Flash-20260731-DSpark.gguf` | 10 896 057 568 (**10.15 GiB**) |
| `.gitattributes` | 1 593 |

GGUF header metadata reported by the API: `architecture = dflash`,
`total = 19 845 850 983` parameters, `context_length = 1048576`.

### 1.2 Download

Headroom check before: `df -h /spinning` → 2.2T size, **350G available**.
10.15 GiB <= 20 GiB and 350G − 10.15G = ~340G >= 100G, so the gate passes.

Downloaded with `hf download` to
`/spinning/llm_stuff/club-3090/models-cache/DeepseekV4-Flash-20260731-DSpark/`.
Result: `DeepseekV4-Flash-20260731-DSpark.gguf`, 10 896 057 568 bytes, size
matches the API exactly. **Download outcome: completed, 10.15 GiB.**

### 1.3 What is actually in the file

Read with `gguf-py`'s `GGUFReader`: 81 tensors, 62 KV entries.

Key metadata:

```
general.architecture      = dflash          general.size_label = 256x594M
dflash.block_count        = 3               dflash.embedding_length = 4096
dflash.block_size         = 5               dflash.target_layers = [41, 42, 43]
dflash.expert_count       = 256             dflash.expert_used_count = 6
dflash.expert_feed_forward_length = 2048    dflash.expert_shared_count = 1
dflash.attention.head_count = 64            dflash.attention.head_count_kv = 1
dflash.attention.key_length = 512           dflash.attention.sliding_window = 128
dflash.attention.compress_ratios = [0, 0, 0]
dflash.hyper_connection.count = 4           dflash.hash_layer_count = 0
```

Tensors, by dtype: **MXFP4 9.562 GiB** (the 9 routed-expert stacks, 3 stages x
gate/down/up, each `[4096, 2048, 256]`), **Q8_0 0.442 GiB** (attention + shared
experts + `fc`), **BF16 0.129 GiB** (`markov_w1`, `markov_w2` at
`[256, 129280]`; `conf_proj` at `[4352, 1]`; the 3 `ffn_gate_inp`), **F32
0.009 GiB** (norms, sinks, hyper-connection mixers).

Structurally decisive:

* **No `token_embd.weight`, no `output.weight`.** The `DFLASH` arch tensor
  list omits both (`gguf-py/gguf/constants.py:4377-4421`), and the graph
  asserts it borrows them from the target:
  `src/models/dflash.cpp:567` *"DSpark decoder requires the target model's
  token embeddings"*, `:662` *"... the target model's output projection"*.
  Our `DeepseekV4ForCausalLMDSpark` does exactly the same
  (`python/sglang/srt/models/deepseek_v4_dspark.py:632-633` sets
  `embed_tokens = None; lm_head = None`, `:642-647` `attach_shared_modules`).
  **Converged design, no gap.**
* `fc.weight` is `[12288, 4096]` = 3 x 4096, i.e. the concat of three target
  hidden states projected down — the EAGLE3-style multi-layer extract.
* `dflash.target_layers = [41, 42, 43]` is `dspark_target_layer_ids + 1`
  (`conversion/deepseek.py:983`: `add_target_layers([layer + 1 for layer in
  hparams["dspark_target_layer_ids"]])`), because llama.cpp extracts the
  *input* of layer k. Our `[40, 41, 42]` and their `[41, 42, 43]` name the
  same three tensors. **Converged.**
* `compress_ratios = [0, 0, 0]`: the draft runs **dense** attention with a
  128-token sliding window and no indexer. llama.cpp hard-refuses anything
  else (`src/models/dflash.cpp:46-50`); we hard-assert the same
  (`python/sglang/srt/models/deepseek_v4_dspark.py:100-102`
  `assert self.compress_ratio == 0`). **Converged.**
* `dflash.block_size = 5` = our `dspark_block_size = 5`; the draft is
  **semi-autoregressive**: one forward emits a whole 5-token block, and the
  Markov head chains position i from `argmax` of position i-1 inside the graph
  (`src/models/dflash.cpp:261-286`). The confidence head gives a per-position
  acceptance estimate used to truncate the block early
  (`common/speculative.cpp:1181-1189`, threshold `params.p_min`).

### 1.4 Can this file be attached as a draft to our UD-Q3_K_XL boot?

**No — four blockers, all named, none of them a design conflict.**

1. **Arch string rejected at the config peek.**
   `python/sglang/srt/utils/hf_transformers/config.py:74` is
   `return arch if arch in sibling_config_gguf_archs() else None`, and the
   accepted set is `("qwen35", "qwen35moe", "gemma4", "deepseek4",
   "dflash-draft")` (`python/sglang/srt/model_loader/gguf_registry.py:90-108`,
   `:55-57`). `"dflash"` is not in it, so `config.py:273-274` hands the file to
   transformers, which raises *"GGUF model with architecture dflash is not
   supported yet"*.
2. **No DSpark GGUF name map exists.** The only draft-GGUF map in the tree is
   `build_dflash_name_map` for the Qwen3 drafter
   (`python/sglang/srt/model_loader/gguf_dflash.py:91-110`).
   `gguf_deepseek4.py` maps backbone blocks only, and its unmapped-tensor
   audit **raises** rather than skipping
   (`python/sglang/srt/model_loader/gguf_deepseek4.py:200-204`), with a
   docstring at `:219-225` that anticipates exactly this case: *"A future export
   that adds blk.>=num_layers would need a draft name map (is_draft), not a
   silent skip, so do not widen this."*
3. **The `mtp.` prefix contract.**
   `DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name`
   (`python/sglang/srt/models/deepseek_v4_dspark.py:861-864`) accepts only
   `mtp.<stage>.*` and returns `None` for everything else, which the caller
   turns into a silent `continue` (`:786-788`). llama.cpp emits `blk.N.*` /
   `markov_w1` / `conf_proj`. A GGUF head would therefore load **zero**
   tensors and boot into the #290 accept-collapse failure mode rather than
   erroring.
4. **Quant format.** The file's experts are MXFP4. Our DSpark path expects the
   checkpoint's native fp8 `weight`/`scale` pairs and dequantises `wo_a`
   (`python/sglang/srt/models/deepseek_v4_dspark.py:771-773`, scale rewrite
   `.scale -> .weight_scale_inv` at `:889`); it has no `qweight` /
   `qweight_type` awareness at all (`grep -n gguf` over that file: zero hits),
   unlike the main model which was hardened for GGUF in #391
   (`python/sglang/srt/models/deepseek_v4.py:2614-2618`).

Not a blocker, worth banking: `--speculative-draft-model-path` pointing at a
`.gguf` is a supported, tested shape with auto `quantization="gguf"`
(`python/sglang/srt/configs/model_config.py:672-701`, the #290 comment), and
the arch-dispatched-drafter pattern already exists end to end
(`gguf_dflash.py` + `python/sglang/srt/model_loader/loader.py:2303-2313` +
`gguf_registry.py:55-57`).

### 1.5 The cheaper route this harvest actually found

The DSpark head **is not a llama.cpp artifact**. It ships inside the original
DeepSeek checkpoint, under the `mtp.` prefix — which is precisely the format
our loader already reads. llama.cpp's converter says so:
`conversion/deepseek.py:939-941` (`if not name.startswith("mtp."): return None`)
and `:944-959` (`mtp.<stage>.<rest>` -> `layers.<stage>.<rest>`), i.e. am17an's
GGUF is a *repack* of tensors we can fetch directly.

Verified against the HF index of `deepseek-ai/DeepSeek-V4-Flash-0731`
(`model.safetensors.index.json`, 72 317 tensors total):

* 4 705 tensors under `mtp.`, stages `0/1/2`.
* They live **exclusively** in shards 46, 47, 48 of 48 — those three shards
  contain 1 568 + 1 565 + 1 572 = 4 705 tensors and **no** non-`mtp` tensor.
* Combined size 3 610 455 184 + 3 560 111 960 + 3 692 775 244 =
  **10.12 GiB**, fp8 (`.weight` + `.scale` pairs, `quantization_config.fmt
  = e4m3`, `scale_fmt = ue8m0`, block `[128, 128]`).
* Per-stage pattern (from the index): `attn.{wq_a,wq_b,wkv,wo_a,wo_b}` +
  scales, `attn.attn_sink`, `attn.{q_norm,kv_norm}`, `hc_{attn,ffn}_{fn,base,scale}`,
  `ffn.experts.{0..255}.w{1,2,3}` + scales, `ffn.shared_experts.w{1,2,3}`,
  `ffn.gate.{weight,bias}`, plus the singletons `main_proj`, `main_norm`,
  `norm`, `hc_head_{fn,base,scale}`, `markov_head.markov_w{1,2}`,
  `confidence_head.proj`.

That is the exact namespace
`python/sglang/srt/models/deepseek_v4_dspark.py:861-889` maps to
`stages.{stage_id}.*`. **Effort to use it: a targeted 10.12 GiB download plus
the pristine `config.json` (which already carries `dspark_block_size = 5`,
`dspark_noise_token_id = 128799`, `dspark_target_layer_ids = [40, 41, 42]`,
`dspark_markov_rank = 256`). No loader change.** Compare against the GGUF
route's three new extension points plus an MXFP4 dequant path.

Note the local UD-Q3_K_XL target **does not** contain the head: read with
`GGUFReader` over all four shards, `deepseek4.block_count = 43`,
`split.tensors.count = 1328`, and a regex sweep for `blk\.4[3-9]|markov|conf_proj|^fc\.|enc\.`
returns **0 hits**. unsloth stripped it.

### 1.6 Boot-arm spec — BOOT-PENDING

Not run (this task is desk + download only). Specified so the next GPU window
can execute it without re-deriving anything.

Prerequisite fetch (not done here, ~10.12 GiB):

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --include 'model-000{46,47,48}-of-00048.safetensors' \
            'model.safetensors.index.json' 'config.json' \
            'tokenizer*.json' 'generation_config.json' \
  --local-dir "$MODEL_ROOT/DeepSeek-V4-Flash-0731-DSpark-head"
```

The index must then be filtered to the `mtp.*` entries, or the loader will
look for the 45 absent shards.

Boot arm, on top of the §4.5.4 rig-runbook recipe:

```
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$MODEL_ROOT/DeepSeek-V4-Flash-0731-DSpark-head" \
  --speculative-dspark-block-size 5 \
  --speculative-num-draft-tokens 6 \
  --speculative-num-steps 1 --speculative-eagle-topk 1
```

`num_draft_tokens = gamma + 1 = 6` and the `num_steps`/`topk` values are not
free choices — `python/sglang/srt/arg_groups/speculative_hook.py:367-430`
overrides or refuses anything else.

Known risks for that window, in order:

1. **fp8 on sm86.** The head is fp8 e4m3 block-scaled; boot 11 already died on
   sm86 with `type fp8e4nv not supported in this architecture`
   (`docs/dev/PLAN_417_dsv4_arch_paths.md`). The draft is only ~10 GiB, so the
   first attempt should keep the whole draft on the 5090 (sm120) rather than
   sharding it across the 3080s.
2. **The draft attention backend is pinned to `dsv4`**
   (`python/sglang/srt/speculative/dspark_components/dspark_config.py:22`,
   applied at `dspark_worker_v2.py:112-114`) — the same backend whose sm86/sm120
   routes #417 is still opening.
3. **Vocab identity.** `dspark_worker_v2.py:152-161` raises if the target has
   no `lm_head.weight`; `dspark_config.py:110-115` raises if
   `mask_token_id (128799) >= target_vocab_size`. Both hold for
   `vocab_size = 129280`, but they are the first two things that will fire if
   the GGUF target exposes its head differently.
4. **Compressor-state rollback on reject** — see §2.4. This is the one item
   where llama.cpp has machinery we appear to lack, and it is a *correctness*
   item, not a perf item.

**External acceptance ladder** (llama.cpp PR #25784, not measured by us):
decode 16.5 -> 23-30 t/s on DGX Spark, accept rate 0.49-0.77 by domain. Use
0.6-0.77 as the reference band; anything below ~0.45 on the same domain mix
means our block/Markov chaining is wrong, not merely slow. Our own floor must
still be an A-vs-A same-boot measurement before any delta is reported.

---

## 2 — Item (b): comp_plan / CSA prefill mechanics vs our indexer path

### 2.1 The headline answer

**Question 2 (is there a 4x-class prefill lever we lack?) — No. Both sides
score compressed tokens, and doing so is architecture, not implementation
freedom.**

The decisive evidence is that the checkpoint ships *separate learned
compressor weights for the indexer*:
`src/models/deepseek4.cpp:142-145` creates
`LLM_TENSOR_INDEXER_COMPRESSOR_{WKV,WGATE,APE,NORM}` with APE shape
`{2 * n_embd_indexer, ratio}` and `ratio == 4`, guarded by
`if (ratio == 4)` at `:136`. A weight tensor whose second dimension *is* the
compression ratio cannot be an implementation choice. Our side instantiates
the same second compressor: `python/sglang/srt/layers/attention/dsv4/indexer.py:997-1007`
builds the `C4Indexer`'s own `compress_ratio=4, rotate=True` compressor at
`head_dim = config.index_head_dim = 128`, distinct from the attention
compressor at head_dim 512.

Our indexer's KV axis is already `seq_len // 4`:
`python/sglang/srt/layers/attention/dsv4/metadata_kernel.py:41-42`
(`c4_seq_lens_raw = seq_len // 4`), `metadata.py:169-179`
(`c4_page_size = page_size // 4 = 64`), and the score matrix at
`indexer.py:342-347` is `[raw_query_tokens, ceil(seq_len/4)]`. llama.cpp's is
the same shape: `src/llama-kv-cache-dsv4.cpp:1999` builds `plans_lid` with
`DSV4_CSA_RATIO` (= 4, `:18`), and `src/models/deepseek4.cpp:651-671` scores
against `mctx->get_lid()->get_k()`.

Both sides also agree on the *overlap* semantics, which was the open fidelity
question: llama.cpp passes `overlap = true` for CSA and LID and `false` for
HCA (`src/llama-kv-cache-dsv4.cpp:1993-1999`) and sizes the compressor
projection with `coff = ratio == 4 ? 2 : 1`
(`src/models/deepseek4.cpp:129-133`); we compute `coff = 1 + overlap` with
`is_overlap_compress(r) := r == 4`
(`python/sglang/srt/layers/attention/dsv4/compressor.py:365-371`,
`compressor_v2.py:711-712`) and a `window_size = compress_ratio * (is_overlap
? 2 : 1)` in the kernel plan
(`python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh:521-522`). CSA reads an
8-token window and emits one row every 4 on both sides; HCA is disjoint 128:1
on both. **No fidelity gap.**

**Question 1 (does their comp_plan batching reduce indexer work relative to
our per-chunk loop?) — No, because comp_plan is not an indexer artifact.**
`comp_plan` (`src/llama-kv-cache-dsv4.h:273-319`,
built by `src/llama-kv-cache-dsv4.cpp:427-...`) is a per-ubatch, host-built
recipe of *row ids* for the compressor ring: `state_pos`, persist src/dst,
snapshot/restore src/dst, `state_read_idxs`, `state_write_idxs`,
`state_write_pos`, `n_visible`, `n_kv`. It plans compression **writes**. Our
equivalent exists and is built the same way — host loop into a pinned buffer,
one `cudaMemcpyAsync`, then a device finaliser:
`python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh:586-627` (host loop,
`should_compress = (position+1) % compress_ratio == 0` at `:596`), `:630-639`
(H2D), `:640-658` (`plan_compress_prefill_kernel_1`), reached via
`python/sglang/srt/layers/attention/dsv4/compressor_v2.py:757-780` and
`python/sglang/jit_kernel/dsv4/compress.py:195-233`. We additionally carry a
device-built twin for CUDA-graph capture (`c_plan.cuh:543-583`) that llama.cpp
has no analogue for.

So the comp_plan comparison is a wash. The real deltas are elsewhere and are
listed next, honestly separated into architecture and lever.

### 2.2 Architecture (checkpoint-defined; we must do it and we do)

| item | llama.cpp | ours |
| --- | --- | --- |
| CSA ratio 4, overlapping 8-token window | `llama-kv-cache-dsv4.cpp:18, 1993`; `deepseek4.cpp:129-133` | `compressor.py:365-371`; `c_plan.cuh:521-522` |
| HCA ratio 128, disjoint | `llama-kv-cache-dsv4.cpp:19, 1996` | same constants, `compressor.py:80-83` asserts `(4, 128)` only |
| indexer scores compressed rows via its own learned compressor | `deepseek4.cpp:136-145` | `indexer.py:997-1007` |
| indexer top-k = 512 over compressed rows | `dflash`/`deepseek4` KV `indexer.top_k = 512`, applied at `deepseek4.cpp:696-697` | `deepseek_v4_backend.py:89, 524-526` (`C4_TOPK = 512`, overridable by `index_topk`) |
| draft is dense (`compress_ratios = [0,0,0]`), SWA 128 | `dflash.cpp:46-57` | `deepseek_v4_dspark.py:100-102`; `deepseek_v4_backend.py:88` |
| draft borrows target embed + lm_head | `dflash.cpp:567, 662` | `deepseek_v4_dspark.py:632-647` |

### 2.3 Implementation freedom (real levers), with effort/yield

Ranked by yield-per-effort. No kill thresholds; gain and effort both stated.

**L1 — the B-fold duplicated KV gather at prefill.** Ours only.
`python/sglang/srt/layers/attention/dsv4/attn_metadata_kernels.py:370-376`
allocates the page table as `[num_q_tokens, num_pages]` (one row per *query
token*, from the causally-expanded lengths), and the indexer gathers per row at
`indexer.py:336`. For a single-sequence chunked prefill every row is identical,
so one chunk materialises B copies of the same KV span:
`B x chunk_pages x 64 x 132` bytes = **2.2 GB at B=2048, 8.8 GB at B=8192**,
on top of the `[B, chunk, H]` fp32 bmm intermediate (4.3 GB at B=2048).
llama.cpp has no analogue — its fused op reads one K view
(`deepseek4.cpp:651-658`) and broadcasts internally. *Yield:* the dominant
memory-traffic term of long-context DSV4 prefill on our cards; it is what
forces `--chunked-prefill-size 512` in the runbook. *Effort:* medium-large —
the page table is shared with the SWA path (`deepseek_v4_backend.py:1693-1726`),
so deduplicating rows means teaching the gather a per-request stride rather
than deleting a dimension. Already diagnosed and deferred at
`docs/dev/NOTE_440_c4_indexer_head_fold.md:257-273`; #447 adds no new argument
for doing it now beyond confirming that upstream-of-us nobody pays this cost.

**L2 — a fused indexer op for sm86/sm120.** llama.cpp has one:
`GGML_OP_LIGHTNING_INDEXER` (`ggml/include/ggml.h:573, 2597`) with a WMMA CUDA
kernel at `ggml/src/ggml-cuda/lightning-indexer.cu:18-...`, gated on
`TURING_MMA_AVAILABLE` and explicitly excluded for HIP/MUSA (`:6-7`) — i.e. it
covers exactly our sm86 + sm120 rig. It computes q.k, ReLU, weighted head sum
and mask add in one kernel (their non-fused fallback at
`deepseek4.cpp:675-693` is the same op graph we run). We run that as ~10
discrete torch kernels per chunk (`indexer.py:336-358`: gather, contiguous,
cast, contiguous, bmm, relu, mul, sum, mul, slice-copy), x
`ceil(c4_len / 8192)` chunks (`indexer.py:246-260`, env
`SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK`, `environ.py:1585`), **x 21 CSA layers**
(see §3) per prefill forward. At 256K raw context that is 8 chunks -> ~80
launches per layer -> ~1 700 launches per forward; at 1M, ~6 700.
*Yield:* removes the launch overhead and the fp32 intermediates entirely; this
is the single largest structural advantage llama.cpp holds on our hardware
class. *Effort:* large — a new CUDA/Triton kernel plus a reference twin, in the
same slot DeepGEMM occupies on Hopper (`indexer_arch.py:57-65` gates DeepGEMM
to cc major 9/10, so sm86/sm120 fall to `BACKEND_TORCH` at `:81-82`). It is,
however, the one place where a llama.cpp *file* is directly instructive: their
kernel is 588 lines and the op contract is documented at `ggml.h:2586-2596`.

**L3 — query-axis chunking with per-chunk top-k.** Neither theirs nor a
llama.cpp idea: it already exists in our own `dsa/` (V3.2) package
(`python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1081-1096, 1242-1299`)
and is simply not wired to `dsv4/`, where chunking is KV-axis only with a
single trailing top-k (`indexer.py:331-358` fills, `:905-931` selects).
Query-axis chunking bounds the B-fold gather of L1 without restructuring the
page table. *Yield:* a large fraction of L1 for a fraction of the effort.
*Effort:* small-medium; the deferral note is `NOTE_440:355-358` (upstream
#33288, "noted for orientation only"). **This is the best effort/yield item in
the list, and #447 raises its priority: it is the cheap 80 % of L1.**

**L4 — SWA window read from the checkpoint.** `deepseek_v4_backend.py:88`
hardcodes `SWA_WINDOW = 128` (and `:511, :1914`) while
`configs/deepseek_v4.py:102` has `window_size = 128` that only the DSpark draft
reads. llama.cpp reads `LLM_KV_ATTENTION_SLIDING_WINDOW` from the GGUF
(`deepseek4.cpp:31`) for the target and again for the draft
(`dflash.cpp:27`). *Yield:* zero today (both are 128), non-zero the first time
a V4 variant ships a different window; today it is a correctness-by-luck item.
*Effort:* trivial. Worth folding into whatever next touches that file.

### 2.4 The one llama.cpp mechanism we appear to lack, and it is correctness

**Compressor-state rollback for rejected draft tokens.** Because the CSA/HCA/LID
compressors are *stateful rings*, a speculative round that writes compressor
state for tokens the target then rejects must be able to undo those writes.
llama.cpp implements this explicitly:

* `comp_plan` carries `state_snapshot_src_idxs`/`state_snapshot_dst_idxs` and
  `state_restore_src_idxs`/`state_restore_dst_idxs`
  (`src/llama-kv-cache-dsv4.h:285-294`), described in the header as
  *"Device-side rollback restore copies snapshot planes back to the current
  compressor-state plane before the graph reads it."*
* Depth is `n_rs_seq`, allocated as extra state planes
  (`src/llama-memory-recurrent.cpp:99`, `n_rows = mem_size * (1 + n_rs_seq)`).
* `llama_kv_cache_dsv4::seq_rm` turns a partial removal into a rollback index
  rather than a clear (`src/llama-kv-cache-dsv4.cpp:1427-1441`).
* And it is wired to speculative decoding specifically:
  `common/common.h:386-391` — `need_n_rs_seq()` returns `draft.n_max` when the
  active spec type is MTP, EAGLE3, DFLASH or **DSPARK**, and 0 otherwise;
  applied at `common/common.cpp:1633`.

On our side a grep for `rollback|rewind|snapshot` across
`python/sglang/srt/layers/attention/dsv4/` and
`python/sglang/srt/speculative/dspark_components/` returns only
`_CeilingSnapshot` in the block-accept estimator — an unrelated statistics
struct. This does **not** prove a bug: our compressor may be reconstructible
from the raw KV rather than being a ring that must be unwound, and DSpark on
DSV4 has never booted here, so the path has never run. It does mean the
question must be answered before the §1.6 boot arm is trusted, and it is the
first thing to instrument in that window. *Effort to answer:* small (read
whether `compressor_v2.forward_unified`'s writes are idempotent under a
re-run at the same positions). *Effort to fix if the answer is bad:* large.

---

## 3 — Item (c): MTP-vs-DSpark 0731 config sweep

### 3.1 The trap, established from the checkpoints themselves

Fetched `config.json` and `model.safetensors.index.json` for three DeepSeek
repos in this session:

| repo | `num_nextn_predict_layers` | `dspark_*` keys | `mtp.` namespace |
| --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash` (June) | 1 | absent | 1 stage, `enorm`/`hnorm` = **real NextN** |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 1 | present | 3 stages, `markov_head.*`, `confidence_head.proj`, `main_proj` |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | 1 | present | 3 stages, same DSpark shape, **no `enorm`/`hnorm`** |

So `num_nextn_predict_layers: 1` is present on *every* V4 config and is a
reliable NextN signal only when the `dspark_*` keys are absent. The
distinguishing key is the `dspark_*` block — which is exactly what our runtime
already uses
(`python/sglang/srt/speculative/dspark_components/dspark_config.py:143-155`,
`checkpoint_bundles_dspark_draft`) and what llama.cpp's converter uses
(`conversion/deepseek.py:977-983`).

### 3.2 What silently mis-fires today, and the fixes made

**Bug 1 — `planner/flags.py` would emit a NEXTN preset for V4-Flash-0731.**
`model_has_mtp()` read `num_nextn_predict_layers` with no DSpark exclusion, and
the preset generator auto-enabled `speculative_algorithm="NEXTN"` on that
signal. `NEXTN` resolves to `EAGLE`
(`python/sglang/srt/speculative/spec_info.py:32`,
`arg_groups/speculative_hook.py:79-88`), which loads
`DeepseekV4ForCausalLMNextN` looking for `model.layers.43.*`
(`python/sglang/srt/models/deepseek_v4.py:2627-2653, 2739-2743`). Nothing
matches -> silently under-loaded draft, the #290 failure mode, not a clean
error.

*Fix:* new `model_bundles_dspark_draft()` next to `model_has_mtp()`, and
`model_has_mtp()` now returns `False` for a DSpark-bundled config, with the
three-checkpoint evidence in the docstring. `flags.py:189-231`.

**Bug 2 — the planner could not pick DSPARK at all.** The UI pick-list was
`("EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM")` while `server_args.py:3197`
accepts `DFLASH` and `DSPARK` too.

*Fix:* `DSPARK` added to `allowed`, with hover text naming the block-shape
requirement. `flags.py:~810`.

**Bug 3 — a misleading preset note.** With Bug 1 fixed, a DSpark-bundled model
fell through to *"no MTP head and no matching local draft model found -
speculative decoding unavailable"*, which is false: the head is right there in
the checkpoint.

*Fix:* a dedicated branch that names DSPARK and says why the generator does not
auto-enable it (the block shape — `num_steps 1`, `num_draft_tokens = gamma + 1`
— is not the NEXTN chain shape this generator emits, and DSpark on DSV4 is not
boot-proven here). Spec is still left off; emitting a preset that cannot boot
would be worse than emitting none.

**Bug 4 — a wrong layer count in two places, worth ~7x in cost math.**
`indexer_arch.py:3-4` and `PLAN_417_dsv4_arch_paths.md:9` both said the sparse
layers are *"the DSpark layers ... layers 40-42 of DeepSeek-V4-Flash"*. Read
from the shipped `config.json` (verified in this session):
`len(compress_ratios) == 46` for `num_hidden_layers == 43`; the trunk is
**21 CSA(4) layers** (even indices 2..42, each carrying an indexer), **20
HCA(128) layers** (odd 3..41), 2 SWA-only (0, 1); the trailing `[0, 0, 0]` are
the three DSpark draft stages. `40-42` is `dspark_target_layer_ids` — the
target layers whose hidden states feed the *draft head* — and has nothing to do
with compression. Both spots corrected; the plan doc gets a dated correction
block rather than a silent rewrite.

**Bug 5 — no runbook warning on the fork's own V4 recipe.** §4.5.4 of
`docs/rig-runbook.md` is the boot recipe we actually use and carried no
speculative guidance.

*Fix:* a named refusal note after the `o_groups` paragraph: NEXTN is wrong for
this checkpoint and why; the `UD-*` GGUF carries only the 43 backbone blocks
(verified: `deepseek4.block_count = 43`, 1328 tensors, zero `blk.43+` /
`markov*` hits across all four shards) so the head is not even in the model
directory; and where the head does live.

### 3.3 Checked and found clean — no change made

* `scripts/ci/slurm/recipes/mi355x-fp{4,8}/dsv4flash/1k1k/*-mtp.yaml` (4 files)
  say *"NextN head from the base model"*. They are upstream (`#29784`,
  `#30313`), the model comes from
  `scripts/ci/slurm/nightly-configs.yaml`, and the `-mtp` entries there point
  at `deepseek-ai/DeepSeek-V4-Flash` / `DeepSeek-V4-Pro` — the June releases,
  which **do** have a real NextN head (table in §3.1). The recipes are correct
  as written; they are also for MI355X hardware we do not have. Left alone
  (`eigene Bugs, nicht fremde`).
* `python/sglang/srt/configs/model_config.py:798-818` already branches on
  `checkpoint_bundles_dspark_draft` before falling back to
  `DeepseekV4ForCausalLMNextN`. Correct.
* `python/sglang/srt/arg_groups/speculative_hook.py:308-361, 367-430` already
  defaults the draft path to the target when bundled, and pins the block
  shape. Correct.
* `docs/dev/FEATURE_CATALOG.md` §4 does not mention DSpark, so it makes no
  false claim. It is *silent*, which is accurate today (DSpark on DSV4 is
  implemented and boot-pending); folding it in belongs with the boot, not with
  this desk pass.

---

## 4 — Adoption candidates, effort and yield

Gain and effort, no kill thresholds; the ratio is the reader's call.

| # | candidate | gain | effort | note |
| --- | --- | --- | --- | --- |
| A | Fetch the native DSpark head (shards 46-48, 10.12 GiB fp8) and boot the §1.6 arm | first DSpark-on-DSV4 boot in this fork; external ladder 0.6-0.77 accept, 1.4-1.8x decode | small (download + one GPU window) — **no loader change** | blocked only by GPU-window availability; §2.4 must be answered inside the window |
| B | Wire `dsa/`'s query-axis chunking into `dsv4/` (L3) | bounds the B-fold gather; the cheap 80 % of L1 | small-medium | code already exists in-tree at `dsa_indexer.py:1242-1299` |
| C | Deduplicate the per-query-token page table (L1) | the dominant prefill memory-traffic term at long context | medium-large | shared with the SWA path; `NOTE_440:257-273` |
| D | Fused sm86/sm120 indexer kernel (L2) | removes ~1 700-6 700 launches + all fp32 intermediates per long-context forward | large | llama.cpp's `lightning-indexer.cu` (588 lines) + `ggml.h:2586-2596` are a usable spec |
| E | GGUF DSpark drafter (`dflash` arch + name map + MXFP4) | lets am17an's file and future llama.cpp DSpark exports load directly | medium-large (3 extension points + dequant) | **dominated by A** for the 0731 head; only worth it if a DSpark head appears that is *not* in the target checkpoint |
| F | Read SWA window from config instead of `SWA_WINDOW = 128` (L4) | zero today, correctness-by-luck removed | trivial | fold into the next edit of `deepseek_v4_backend.py` |

Ordering note: A gates everything else on the spec side, B is the best pure
effort/yield item, and E is the one thing this task went looking for and found
**not** worth building — the head we wanted is already in a format we read.

---

## 5 — Provenance

* llama.cpp clone: `github.com/ggml-org/llama.cpp`, `--depth 200
  --filter=blob:none`, HEAD `221f0f6`, at
  `/root/.claude/jobs/1481bb40/tmp/llamacpp-447` (throwaway, outside
  `/spinning`).
* Downloaded artifact:
  `/spinning/llm_stuff/club-3090/models-cache/DeepseekV4-Flash-20260731-DSpark/DeepseekV4-Flash-20260731-DSpark.gguf`,
  10 896 057 568 bytes, HF sha `9d79f200`.
* HF API reads: `am17an/DeepseekV4-Flash-20260731-DSpark`,
  `deepseek-ai/DeepSeek-V4-Flash-0731`, `deepseek-ai/DeepSeek-V4-Flash`,
  `deepseek-ai/DeepSeek-V4-Flash-DSpark`.
* Local target inspected: `DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL` (4 shards,
  1328 tensors) and its pristine `config.json`.
* The Reddit prefill-regression thread is **not** re-litigated here: it was a
  llama.cpp CUB `DeviceTopK` regression, already ruled out for our stack
  (our top-k is the JIT `topk_v1`/`topk_v2` at
  `python/sglang/jit_kernel/dsv4/topk.py:79-172`, not CUB).
* No GPU was used. No `/spinning/gpu-arb/` window taken.
