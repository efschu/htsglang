# Qwen3.6-27B - Engine comparison matrix (final)

As of 2026-07-17. All numbers are measured (phase 3a-3c, M27a-M27e); no
estimated or interpolated values. Missing cells are explicitly marked
INFEASIBLE (with a short reason) or n/a.

## 1. Header: hardware, model, engines, common settings

### Hardware (this system)
- 1x NVIDIA RTX 5090, 32 GB VRAM (NVML 32607 MiB)
- 2x NVIDIA RTX 3080, 20 GB VRAM each (NVML 20480 MiB)
- One of the 3080s is attached to PCIe Gen4 x4 (narrower link). No NVLink,
  no GPU P2P (GeForce, `nvidia-smi -p2p` shows NS/CNS for all pairs -> PHB-only).
- NVML/PCI enumeration varies between boots/driver states; physical
  GPU IDs are resolved at runtime via NVML (the 5090 is not hard-wired).
  Container CUDA order is FASTEST_FIRST -> cuda:0 = 5090.

### Model
Qwen3.6-27B family (hybrid GDN / linear attention + embedded MTP/NEXTN draft,
blk.64). Quant variants:
- Original Qwen3.6-27B-FP8 (layers-*.safetensors + mtp.safetensors)
- Qwen3.6-27B-AWQ-BF16-INT4 (compressed-tensors, ~14 GB)
- unsloth UD-GGUF Q6_K_XL (26.0 GB, + mmproj-BF16 vision tower)
- unsloth UD-GGUF Q8_K_XL (35.8 GB) -- INFEASIBLE on both forks (see footnote 8)

### Engines (exact versions / images / commits)
- llama.cpp: `ghcr.io/ggml-org/llama.cpp:server-cuda`, upstream b10015 (id 297b3e6a71e1),
  model unsloth Qwen3.6-27B-MTP-GGUF (embedded MTP draft).
- shvllm (vLLM fork, uneven-TP / rank-gpu-id): image `shvllm-qwen35-gguf:cu129-uneven`
  (id 6ee0897f1157), repo HEAD f78ea433f, pip NCCL 2.30.7.
- htsglang (sglang fork, GGUF plugin + uneven-TP): repo HEAD 3e76cbbf1 (UNMODIFIED).
  TP=2/TP=3 as a local venv (torch 2.11+cu130, NCCL 2.28.9); TP=4 co-located as
  Docker image `htsglang-qwen35-gguf:cu130-3e76cbbf1` (NCCL 2.30.7 baked in -- co-located
  TP=4 requires NCCL >= 2.30, hence docker-only).

### Common settings (all cells)
- MTP / speculative decoding ACTIVE EVERYWHERE:
  - vLLM/shvllm: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  - sglang/htsglang: `--speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
  - llama.cpp: `--spec-type draft-mtp --spec-draft-n-max 3`
- KV-cache dtype fp8 (shvllm `--kv-cache-dtype fp8`, htsglang `fp8_e4m3`);
  llama.cpp has NO fp8 KV -> nearest analog q8_0 block-quant (footnote 1).
- UNCACHED by construction: unique-nonce token-id prompts, cached_tokens == 0 in
  every request of every cell (for llama.cpp benign field semantics, see JSON caveat).
- Thermal gate <= 80 C before every boot (cooldown.sh, fired on every boot).
- One warmed-up battery run per boot (no median-of-3); maxKV for
  llama.cpp by bisection, for the forks from the boot line.

### Metric definitions (columns)
- **Prefill20k** (P1): 1 request, 20000 exact unique input_ids, 1 output token -> tok/s.
- **Dec1k code / prose** (D1): 1 request, ~1000-token decode (ignore_eos) -> tok/s.
- **Par8-Prefill** (P8): 8 parallel requests x 6000 input_ids -> aggregated tok/s.
- **Par8-Dec code / prose** (D8): 8 parallel requests x ~400-token decode -> aggregated tok/s.
- **maxKV**: largest bootable+serving KV pool at the cell config (config-bound,
  see context section 3 for the distinction from the calibrated maximum context).
- **Accept**: MTP/NEXTN spec_accept_length (D1 code / D1 prose). Only sglang/htsglang
  exposes it; the vLLM and llama.cpp surfaces do not -> n/a there. The MTP speedup
  is already baked into the tok/s for all engines.

---

## 2. Main tables per scenario

All tok/s in tokens/second, maxKV in tokens.

### Scenario S1 -- Layer-split (3 GPUs, 5090 + 2x 3080, 72 GB)

llama.cpp only: true TP cannot span the 3 heterogeneous cards, hence
`-sm layer` (pipeline across all 3 GPUs). The forks do not run this scenario.

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| llama.cpp Q8 (UD-Q8_K_XL) | 1835.4 | 57.9 | 45.0 | 1914.4 | 150.1 | 156.8 | 355568 | n/a |
| llama.cpp Q6 (UD-Q6_K_XL) | 1688.9 | 74.5 | 50.8 | 1741.5 | 176.3 | 154.6 | 518096 | n/a |

### Scenario S2 -- TP=2 (2x RTX 3080, 5090 not involved)

True tensor parallelism across the two 3080s. htsglang TP=2 was measured with
the 5090 PHYSICALLY REMOVED (pure 2x3080), shvllm with CUDA_VISIBLE_DEVICES=0,2,
llama.cpp with `-sm tensor` (the planned `-sm row` is INFEASIBLE here, footnote 2).

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| llama.cpp Q6 (-sm tensor) | 1089.8 | 79.6 | 63.5 | 991.2 | 149.5 | 130.8 | 266856 | n/a |
| shvllm Q6 (GGUF) | 1128.1 | 44.6 | 36.3 | 1092.5 | 70.7 | 65.0 | 25600 | n/a |
| shvllm AWQ-INT4 | 1153.6 | 75.5 | 62.4 | 1121.9 | 195.0 | 187.8 | 94400 | n/a |
| htsglang Q6 (GGUF) | 1126.1 | 54.5 | 48.7 | 1098.7 | 65.5 | 49.8 | 32768 | 3.36 / 3.04 |
| htsglang AWQ-INT4 | 1169.2 | 86.5 | 63.9 | 1157.2 | 136.5 | 119.7 | 148864 | 3.22 / 2.38 |
| Q8 (any engine) | INFEASIBLE: does not fit on 2x20 GB (Q8=35.8 GB, excluded by plan); additionally a GGUF-Q8 loader bug (footnote 8) | | | | | | | |

Note: For htsglang TP=2 concurrency is hard-capped at 2 (the Mamba-state cache
is the bottleneck, not KV tokens) -> the Par8 columns are throughput with requests
queued 2-at-a-time, NOT 8-way batching (footnote 4). shvllm allows true 8-way batching
(D8 ~2.6x D1 for AWQ). maxKV efficiency Q6: shvllm serves 25600 @ GMU 0.88 unaided,
htsglang needs a manual KV cap of 32768 (footnote 7 / #63).

### Scenario S4 -- TP=3 uneven-auto (5090 + 2x 3080, weighted uneven-DCP)

VRAM-weighted uneven-TP split, rank0 = 5090. htsglang with BOTH variants:
- **V1 = max-KV** (`--rank-tp-ratio auto`, calibrated token vector, pure pool maximum).
- **V2 = max-perf @ >= 100k KV** (`auto-performance` with MLP concentration on the 5090;
  GGUF uses a PINNED-MLP approximation, since auto-performance is INFEASIBLE on GGUF, footnote 5).

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| shvllm Q6 (GGUF) | 1239.8 | 61.7 | 46.2 | 1215.2 | 180.3 | 155.3 | 1139318 | n/a |
| shvllm AWQ-INT4 | 1228.3 | 83.7 | 62.7 | 1211.6 | 288.1 | 266.2 | 1146573 | n/a |
| shvllm FP8 | 1218.7 | 73.9 | 60.7 | 1223.5 | 287.9 | 273.7 | 1046126 | n/a |
| shvllm Q8 (GGUF) | INFEASIBLE: UD-Q8_K_XL mixed-precision fused qkvz, GGUF plugin rejects early (footnote 8) | | | | | | | |
| htsglang FP8 -- V1 max-KV | 1123.7 | 98.4 | 80.8 | 1125.7 | 354.9 | 268.9 | 824896 (*M28) | 3.28 / 2.69 |
| htsglang FP8 -- V2 max-perf | 1202.7 | 84.4 | 61.3 | 1207.8 | 343.2 | 260.2 | 299968 | 3.12 / 2.25 |
| htsglang AWQ -- V1 max-KV | 1123.9 | 103.2 | 93.1 | 1135.2 | 346.0 | 299.2 | 857408 (*M28) | 3.14 / 2.86 |
| htsglang AWQ -- V2 max-perf | 1247.2 | 115.7 | 94.8 | 1261.6 | 345.3 | 305.2 | 441536 | 3.40 / 2.79 |
| htsglang Q6 -- V1 max-KV | 1102.7 | 65.9 | 54.6 | 1111.5 | 184.0 | 157.0 | 818880 (*M28) | 3.19 / 2.66 |
| htsglang Q6 -- V2 max-perf (pinned-MLP) | 1207.5 | 63.2 | 54.2 | 1213.4 | 221.7 | 164.1 | 241216 | 2.99 / 2.58 |
| htsglang Q8 -- V1/V2 | INFEASIBLE: loads further than shvllm, then crashes on mixed-dtype padding (footnote 8) | | | | | | | |

Note: Here concurrency is NOT mamba-capped (5090 in the mix, ~100 Mamba slots)
-> the htsglang Par8 columns are true 8-way batching (D8 >> D1). The V2 effect
depends on the quant (footnote 5): AWQ V2 = true double win (prefill +11% AND single-decode
+12%), FP8 V2 = prefill gain but single-decode drop (decode knee), GGUF V2 = only
a pinned-MLP approximation.

(*M28) maxKV recalibrated + config-aligned 2026-07-17 (audit M28): --max-running-requests 8
(mamba pool 50 instead of 100 slots) + reserve 1500 uniform + converged token vector
(FP8 32,15,17; AWQ 31,14,19; Q6 auto 30,17,17), HEAD 3e76cbbf1. The original
V1 boot values with mrr16/reserve bump (FP8 530944, AWQ 566912 [no-MTP vector, uncalibrated],
Q6 546560) are stored in the JSON as maxkv_v1_config. The vLLM number additionally includes
Mamba-block accounting in the unified pool -- the remaining gap (~1.27x, e.g. FP8 824896 vs 1046126)
is accounting semantics, not missing memory. Speed columns unchanged (V1 boots).

### Scenario S3 -- TP=4 co-located (5090 x2 + 2x 3080, via MPS)

4 equally sized ranks on 3 physical GPUs; ranks 0+1 share the 5090
via NVIDIA MPS. Absolute per-rank budget (no uneven ratio, no DCP -> V1/V2 n/a).
shvllm @ 14500 MiB/rank, htsglang @ 13500 MiB/rank (engine difference, footnote 3).

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| shvllm FP8 (@14500) | 1411.9 | 107.3 | 82.2 | 1442.9 | 323.4 | 308.0 | 429096 | n/a |
| shvllm AWQ-INT4 (@14500) | 1417.5 | 103.2 | 72.8 | 1443.7 | 306.9 | 296.9 | 514036 | n/a |
| shvllm Q6 (GGUF, @14500) | 1384.0 | 62.6 | 52.2 | 1424.4 | 220.7 | 175.0 | 443740 | n/a |
| shvllm Q8 (GGUF) | INFEASIBLE: identical to shvllm-Q8-TP=3 (mixed-precision fused qkvz, footnote 8) | | | | | | | |
| htsglang FP8 (@13500) | 1326.3 | 92.1 | 91.5 | 1341.4 | 244.6 | 244.6 | 330818 | 2.79 / 2.75 |
| htsglang AWQ-INT4 (@13500) | 1330.7 | 115.3 | 103.5 | 1351.9 | 259.0 | 216.2 | 345064 | 3.18 / 2.84 |
| htsglang Q6 (GGUF, @13500) | 1282.3 | 76.4 | 65.6 | 1311.9 | 166.0 | 164.3 | 322174 | 3.45 / 2.98 |
| htsglang Q8 (GGUF) | INFEASIBLE: no boot attempted, known loader bug (footnote 8) | | | | | | | |

Anomaly (open, escalated to the maintainer): For shvllm TP=4, FP8 beats AWQ-INT4 in
single-decode (107.3 vs 103.2); for htsglang it is the other way around (AWQ 115.3 vs FP8 92.1) --
engine-dependent INT4-dequant / fp8-path costs.

---

## 3. Context section: config-bound maxKV vs calibrated maximum contexts

The maxKV values in the tables above are CONFIG-BOUND pool sizes at the respective
benchmark config (context length, reserve, MTP active). They must NOT be read directly
as the "engine's maximum context" -- and certainly not compared 1:1 between the
engines, because they come from different boot lines with different
semantics:

- **shvllm-maxKV** = vLLM boot line "GPU KV cache size: N tokens".
- **htsglang-maxKV** = sglang boot line "max_total_num_tokens".
- **llama.cpp-maxKV** = largest serving `-c` via bisection.

This is why shvllm TP=3 (>1M) and htsglang TP=3 (~530k) lie so far apart despite
the same MTP activation: different engines count the DCP pool
differently, NO equivalence proof (apples/oranges).

### Calibrated maximum contexts (from HANDOFF, what was measured in each case)

**htsglang uneven-DCP calibration (single-node, VRAM-weighted, converged token vectors):**
These are the fork's actual KV ceilings per quant, determined by self-calibration
-- separated by MTP off/on:

| Context class | Value (tokens) | What was measured |
|---|---|---|
| AWQ, no-MTP | 886080 | converged uneven-DCP pool without draft KV (vector [31,15,18]) |
| GGUF-Q6, no-MTP | 840896 | converged uneven-DCP pool without draft KV |
| FP8, no-MTP | 804416 | converged uneven-DCP pool without draft KV (vector [32,15,17]) |
| FP8, MTP active (V1 boot config) | 530944 | with draft KV; mrr16 + RESERVE 3000,2200,2200 + TOKVEC 33,13,18 |
| GGUF-Q6, MTP active (V1 boot config) | 532224 | with draft KV; mrr16 + RESERVE 3000,2200,2200 |
| FP8, MTP active (M28 recalibrated) | 824896 | mrr8 (mamba 50 slots) + reserve 1500 + vector 32,15,17 |
| AWQ, MTP active (M28 recalibrated) | 857408 | mrr8 + reserve 1500 + vector 31,14,19 |
| GGUF-Q6, MTP active (M28 recalibrated) | 818880 | mrr8 + reserve 1500 + auto vector 30,17,17 (--skip-server-warmup) |

CORRECTION (M28, 2026-07-17): The jump previously described as FUNDAMENTAL, no-MTP -> MTP
(FP8 804416 -> 530944, -34%), was predominantly CONFIG, not draft KV: at mrr8 + reserve 1500,
FP8 WITH MTP reaches 824896 -- more than the no-MTP value 804416. The draft actually costs only
a little KV; the old -34% drop came from the oversized auto-mamba pool
(mrr16 -> 100 slots, ~5.4 GB on the 5090) plus the reserve bump. The old MTP rows remain
documented above as V1 boot config.

**shvllm TP=3 (from the boot logs, MTP active):** >1M class -- FP8 1046126, Q6 1139318,
AWQ 1146573. This is the maximized DCP pool that vLLM reports in its boot line;
because of the differing counting semantics it is NOT to be directly reconciled against
htsglang's max_total_num_tokens.

Section conclusion: Within one engine, maxKV values are comparable; across engines
only qualitatively. The calibrated htsglang numbers (886080 / 840896 / 804416) show
the pure fork capacity WITHOUT MTP; the matrix cells show the serving pool WITH MTP.

---

## Addendum (2026-07-17): GGUF after the performance overhaul (htsglang e25180447) -- PRELIMINARY

After the GGUF perf overhaul (htsglang commit e25180447: flat-byte-shard layout,
persistent dequant workspace, quantized-resident embed/lm_head + NEXTN module sharing,
batched-MMVQ ncols<=8) the GGUF data points were re-collected in a separate
validation battery. These numbers are PRELIMINARY and do NOT replace the matrix cells
above (those remain unchanged, the state before the overhaul).

| Data point | old (before overhaul) | new (e25180447) | Delta |
|---|---|---|---|
| Q6 TP=2, Dec1k code/prose | 54.5 / 48.7 | 66.0 / 54.0 | +21% code |
| Q6 TP=3-uneven+DCP, Dec1k code/prose | 65.9 / 54.6 | 91.1 / 79.1 | +37% / +47% |
| Q8_K_XL TP=3-uneven, Dec1k code/prose | INFEASIBLE | 91.1 / 82.3 | boots + coherent |
| Q4_K_M TP=2, Dec1k code/prose | 45.4 / 45.1 | 47.6 / 47.3 | +5% decode |
| Q4_K_M TP=2, maxKV (uncapped) | 153172 | 207824 | +36% |
| Q6 TP=2, weights/rank | 14.03 GB | 12.74 GB | -1.3 GB |

- **Q8 status change:** UD-Q8_K_XL was previously INFEASIBLE on BOTH forks (mixed-precision
  fused GDN `in_proj_qkvz`, footnote 8). After the overhaul (flat-byte-shard layout carries
  mixed dtypes) htsglang boots Q8_K_XL TP=3-uneven coherently -- making htsglang the
  only true TP>1 Q8 implementation on the rig besides the llama.cpp single-process layer-split.
- **Assessment:** Q6 TP=3-uneven decode (91.1 code) now sits ABOVE all llama.cpp cells
  on the rig (TP=2 `-sm tensor` 79.6, layer-split 74.5). Q6 TP=2 (66.0) stays behind llama.cpp
  TP=2 (79.6) but closes the gap substantially.
- **Methodology note (honest):** The numbers come from the T66 validation battery
  (`t66_decode_bench.py`, greedy, ignore_eos, 512 tok), NOT from a full
  matrix re-run. They were calibrated against the matrix TP=3 reference: the old boot yielded
  66.4 / 54.0 ≈ matrix cell 65.9 / 54.6 (Q6 TP=3), which supports the comparability of the internal
  A/B delta. Prefill (P1) and Par8 (P8/D8) were NOT remeasured. MTP accept
  remains unchanged (~3.35, Q6 TP=2). The kill-switch boot (flat layout + workspace, without
  batched-MMVQ/quantized vocab) is byte-identical to HEAD (3x1024 long-forms +
  logprob maxdelta 0.000000). Boot recipes needed, otherwise auto-sizing eats the gain (M29):
  TP=2 `--mem-fraction-static 0.80`; TP=3 `--rank-auto-reserve-mib 3500 --cuda-graph-max-bs-decode 8`;
  `SGLANG_GGUF_BATCHED_MMVQ=1`.

---

## 4. Footnotes / caveats (M27a-M27e)

1. **llama.cpp q8_0 KV instead of fp8** (M27a): llama.cpp has no fp8 KV. All cells use
   8-bit BLOCK-quant KV (`-ctk/-ctv q8_0`, + draft `-ctkd/-ctvd q8_0`, `-fa on`) as the nearest
   analog to `--kv-cache-dtype fp8` on the other engines. q8_0 (block-quant) != fp8 (per-tensor).

2. **llama.cpp -sm tensor CPU-sampling + -sm row INFEASIBLE** (M27a): The originally planned
   `-sm row` fails at model load ("device CUDA0 does not support split buffers" -- needs
   P2P/split buffers, not present on these GeForce cards). Replaced by `-sm tensor`
   (true TP). In doing so llama.cpp logs "backend sampling not supported with SPLIT_MODE_TENSOR;
   using CPU" -> sampling runs on the CPU (limitation, not an error). NCCL needs
   `--ipc=host --shm-size + NCCL_P2P_DISABLE=1 + NCCL_NVLS_ENABLE=0`.

3. **RANK_MIB 13500 instead of 14500 (htsglang TP=4)** (M27e F1): 14500 MiB/rank OOMs at
   draft CUDA-graph capture on the 5090 (both co-located ranks reach ~15.5 GiB = budget
   + ~1 GiB non-KV overhead). 13500 boots cleanly. shvllm fits at 14500 (vLLM's MiB budget
   covers total process usage incl. graphs) -- an engine difference, relevant to every
   cross-engine maxKV comparison in S3.

4. **htsglang TP=2 Mamba concurrency cap = 2 + manual Q6 tuning** (M27c): On 2x20 GB
   the Mamba/linear-attn state cache (73-75 MiB/slot/rank) is the bottleneck, not KV -> sglang
   automatically reduces max_running_requests 16 -> 2. All TP=2 Par8 values are therefore 2-at-a-time
   queued throughput, not 8-way batching. Both quants needed `--mem-fraction-static 0.90`
   (default 0.749 -> Mamba pool 0 slots -> boot RuntimeError). Q6 additionally a manual KV cap of
   32768 (at 0.90 uncapped it boots 102328, but OOMs on the first forward in the GGUF dequant scratch).

5. **V2 quant dependency** (M27d): AWQ V2 = true STRICT WIN (prefill +11%, single-decode +12%,
   maxKV 441536) -- AWQ has more MLP units (544), the 5090 absorbs the concentration without a
   decode knee. FP8 V2 = prefill +7% BUT single-decode drop (D1c 84.4 vs V1 98.4, decode knee),
   maxKV 299968. GGUF V2 = auto-performance INFEASIBLE (uneven_perf.py:531 open(model_path/'config.json')
   -> NotADirectoryError, because model_path for GGUF is the .gguf FILE); hence pinned-MLP approximation
   (`--rank-mlp-ratio 5,1,1`), no decode-knee guard.

6. **MPS restart lesson** (M27b/M27e): The documented check `ls /tmp/nvidia-mps/` is a
   FALSE POSITIVE -- stale sockets survive the daemon's death. The MPS daemon was actually
   DEAD; co-located ranks then merely time-slice. Check liveness ONLY via the control-daemon response
   (`echo get_default_active_thread_percentage | nvidia-cuda-mps-control` -> 100.0),
   never via ls. An initial shvllm-tp4 battery without live MPS was discarded and rerun.

7. **Q6 TP=2 memory-efficiency question (#63)**: shvllm serves q6_tp2 with maxKV 25600 @ GMU 0.88
   without special tuning; htsglang needs, on the same HW, a manual KV cap (32768) + mem-fraction 0.90,
   and even the natural pool (102328) does not serve. Suspected cause: GGUF on-the-fly dequant scratch,
   Mamba slots, mmproj load. Open investigation item for the maintainer.

8. **Q8 INFEASIBLE on both forks (different failure depths)**: Common root cause --
   unsloth UD-Q8_K_XL stores the fused GDN `in_proj_qkvz` in MIXED precision (fp16 + uint8).
   - shvllm rejects EARLY: `ValueError: Detected some but not all shards of ...in_proj_qkvz are
     quantized` (is_layer_skipped_gguf fused-shard check), identical for TP=3 and TP=4.
   - htsglang loads FURTHER (past the shard check), then crashes at model init:
     `AssertionError: Data container has mixed dtypes: {torch.float16, torch.uint8}`
     (gguf.py:475 _create_padded_weight_param). Q6_K_XL loads cleanly in both plugins ->
     Q8_K_XL-specific layout. Fix (mixed-dtype GGUF layer) filed to the maintainer as a TODO (#64), outside
     the measurement scope. Q8 is therefore only carried by llama.cpp (S1 layer-split).

---

## 5. Conclusion (honest, per discipline)

**Single-stream decode (D1):** htsglang wins clearly, driven by MTP/NEXTN + always-on
decode graphs. Peak values: AWQ V2 TP=3 (D1c 115.7) and AWQ TP=4 co-located (D1c 115.3);
AWQ V1 TP=3 (103.2). The vLLM side (shvllm) only tops in the TP=4 FP8 case (107.3). llama.cpp
sits structurally below (best Q6 TP=2 -sm tensor: 79.6). -> **htsglang takes the
single-decode crown via MTP + graphs.**

**Par8 / 8-way throughput (D8):** shvllm dominates where it allows true 8-way batching:
TP=3 AWQ 288.1 and FP8 287.9 code. htsglang TP=3 is on par at the code peak (FP8 V1 354.9 --
even higher here) but more inhomogeneous; the decisive point is that htsglang TP=2 is hard
mamba-capped (D8c only 65.5-136.5). -> **With genuine 8-way batching, shvllm-TP=3 and htsglang-TP=3
deliver the highest aggregates; htsglang-TP=2 falls behind due to the Mamba cap.**

**Prefill (P1/P8):** TP=4 co-located wins clearly -- shvllm FP8/AWQ are at 1411-1443 tok/s
(P1/P8), htsglang TP=4 at 1282-1352. Two ranks on the 5090 + one each on the 3080s maximize
the prefill compute. Layer-split (llama.cpp S1) is nominally high in single-prefill (Q8 1835,
Q6 1689), but that is server-native timing (without HTTP) and not 1:1 comparable with the wall-clock
numbers of the forks. -> **TP=4 co-located = prefill crown (among the comparable
wall-clock measurements).**

**Max context (config-maxKV):** shvllm holds the largest serving pools at TP=3 uneven-auto
(>1M: AWQ 1146573, Q6 1139318, FP8 1046126). htsglang-TP=3 is at ~530-567k per boot line
(MTP-active class), which, because of the differing counting semantics, must NOT be read as
"half as much context". -> **shvllm holds the largest config-maxKV at TP=3.**

### The 2-3 clearest overall statements
1. **htsglang wins single-stream decode** (MTP/NEXTN + decode graphs), strongest with AWQ.
2. **shvllm holds the largest config-maxKV at TP=3 uneven-auto** (>1M class) and delivers
   the most robust true 8-way batching (TP=3 AWQ/FP8 ~288 D8c).
3. **TP=4 co-located is the prefill crown** (shvllm FP8/AWQ ~1410-1443 tok/s), bought with
   MPS overhead and a tighter KV budget (13500/14500 MiB/rank).
4. **llama.cpp** is the pragmatic 3-card all-rounder (the only Q8-capable engine here, high
   layer-split prefill), but loses in decode and must use q8_0 KV instead of fp8; true TP
   on these GeForce cards is only possible via `-sm tensor` (with CPU sampling), not `-sm row`.

---

*Verification: All transcribed numbers were cross-checked against matrix_results/*.json (llamacpp,
shvllm, htsglang_tp2/tp3/tp4); calibrated context values against HANDOFF.md
(lines 620/623/1213/1242/1304). Spot checks confirmed, among others: shvllm awq_tp3 D8c 288.1,
htsglang awq_tp3_V2 D1c 115.7, htsglang fp8_tp4 maxKV 330818, llama.cpp q6_layersplit
maxKV 518096, shvllm fp8_tp4 D1c 107.3.*
