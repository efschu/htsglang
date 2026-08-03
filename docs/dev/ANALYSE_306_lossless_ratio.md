# ANALYSE #306 — lossless cold-tier compression: ratio falsification on real assets

WAVE 1 item 3 of `ROADMAP_456_matrix_execution.md`. Desk probe, no card, no
build. The question is narrow and was asked before the measurement: **is there
enough redundancy in this fork's cold-tier assets for lossless compression to
pay for itself on any link this rig has?** The build slice was deliberately
gated behind the answer.

Probe: `scripts/dev/306_ratio_probe/` (runnable, seed-fixed).
Raw data: `scripts/dev/306_ratio_probe/results/` (`samples.json`,
`results.jsonl`, `ceiling.jsonl` — one line per sample x layout x codec).

## 0. Verdict

**No cell on this rig is a win for a serial (transfer-then-decompress) mover,
on any asset class measured.** Parallel decompression saturates at
**4.3-4.8 GB/s** on this box (DRAM-write-bound past 4 workers, table I), which
puts the serial break-even ratio at **1.59-1.72 even against the slowest link
this rig has** (T3 NVMe, 1.8 GB/s). The best ratio achieved on any asset is
**1.211** (FP8 weights). The gap is not close, and it does not close with more
cores or a stronger codec.

**Under a pipelined mover** (chunked transfer overlapped with chunked
decompress) three cells clear 1.0 on the slow links:

| asset class | pipelined speedup on links <= 2.83 GB/s | status |
|---|---|---|
| Qwen3.6-27B **FP8-E4M3 weights** | **1.211x** | ALIVE |
| Qwen3.6-27B **INT8-W8A8 weights** | **1.131x** | ALIVE |
| DSV4-Flash **MXFP4** routed experts | 1.046-1.049x | ratio-alive, below the 1.08 kill line on the DSV4 mix as a whole |

**Everything else is DEAD**, including the asset this probe was commissioned
for. **The DeepSeek-V4-Flash cold-expert tier is DEAD for lossless
compression**: its IQ2_XS / IQ3_XXS / IQ3_S bytes compress **1.026-1.030** at
best, its order-0 entropy is **7.91-7.95 bits per byte**, and `zstd --ultra -22
--long` finds **1.000-1.007**. The k-quant family measured alongside it (Q3_K
1.014, Q4_K 1.052, Q6_K 1.029, IQ4_XS 1.028) is the same story. This is the
expected outcome — quantisation is itself a compressor — and it retires
`ANALYSE_456` §2.2 cell 2 for the expert asset.

Two findings the ratio table alone does not show:

* **The byte-plane split works, and is still useless.** Splitting the ggml
  block into planes raises the ratio on every GGUF class (e.g. IQ3_XXS
  1.0000 raw -> 1.0297 stride-split, MXFP4 1.0456 -> 1.0908) — but the split's
  own inverse permutation caps decompression at **0.5-0.9 GB/s**, below every
  link in the tier table. The layout that produces the best ratio is the one
  that can never deliver it.
* **On this box's storage, the disk-image win is already taken.** The `/spinning`
  ZFS dataset has transparent compression active, and its own allocated-vs-
  apparent accounting already delivers **1.1735x** on the #89 hibernate image,
  **1.2296x** and **1.1420x** on the INT8 and FP8 safetensors, and **0.9995x**
  on the DSV4-Flash GGUF. Application-level zstd on the hibernate image reaches
  1.1953x whole-file — i.e. **1.9 % beyond what the filesystem already gives
  for free**, in exchange for a decompress stage on every reload.

**Recommendation: do not build #306 as a codec.** One narrow piece of it is
worth taking, and it is not compression: **12.64 % of the #89 hibernate image
is all-zero 4 KiB pages**, worth **1.145x** through a sparse write with no
decompress stage at all (table J). That is a two-line change to the hibernate
writer, not a cold-tier compression feature.

---

## 1. Kill criterion, stated before the run

An asset class whose **median ratio stays below 1.08 across every method
tested** is DEAD for compression and is written down as such. 1.08 is the point
below which the saving (7.4 % of bytes) is smaller than the operational cost of
carrying a second representation of every cold asset — a compressed-vs-raw
branch in the fetch path, a second checksum domain, and a decompress stage that
must be scheduled somewhere.

Quantised weight bytes are high-entropy **by construction**: quantisation is
itself a lossy compressor, and a good one leaves little for a lossless coder to
find. A DEAD verdict here is a fully successful outcome of this probe, not a
failure of it — it retires a lever the #456 matrix listed as a conditional
multiplier, and it does so before anyone spends a build slice on it.

## 2. The break-even formulas

Let `S` = uncompressed bytes, `C` = compressed bytes, `r = S/C` the ratio,
`L` the link rate in B/s, and `D` the decompress rate measured **in
uncompressed output bytes per second** (this is the load-bearing rate: a
consumer needs the original bytes, not the compressed ones).

**Serial** — transfer completes, then decompress runs. The conservative case,
and the one a simple implementation gets:

```
T_compressed = C/L + S/D = S/(rL) + S/D
T_raw        = S/L
win  <=>  1/(rL) + 1/D  <  1/L
     <=>  D > L * r/(r-1)                     (break-even decompress rate)
     <=>  r > D/(D-L),  and impossible if D <= L   (required ratio r_min)
speedup_serial = 1 / (1/r + L/D)
```

The `D <= L` branch matters: if decompression alone is slower than sending the
payload uncompressed, **no ratio whatsoever** produces a win. That is the case
for most of the byte-plane arms below.

**Pipelined** — chunked transfer overlapped with chunked decompression, the
optimistic upper bound a well-engineered mover could reach:

```
T_compressed = max(C/L, S/D)
win  <=>  r > 1  AND  D > L
speedup_pipelined = min(r, D/L)
```

Both are reported. The serial formula is the gate; the pipelined one bounds how
much a better implementation could possibly recover. Neither formula rescues a
ratio that is not above 1 in the first place.

**Byte-plane split cost is charged to the decompress side.** A de-interleaved
asset is not usable until it is re-interleaved, so `D` for the non-`raw`
layouts includes the inverse permutation. The codec-only rate is recorded
separately (`decomp_codec_mbs`) so the permutation's share stays visible rather
than folded away. The two components are each a best-of-N and are then summed,
which understates the true combined time — the bias runs **toward** declaring
compression viable, so every DEAD verdict below survives it.

## 3. What was measured, and on what

### 3.1 Methods

Four codecs — `zstd -3`, `zstd -19`, `zlib -6`, `lzma` preset 1 ("lzma-fast") —
crossed with four byte layouts:

| layout | what it does | applies to |
|---|---|---|
| `raw` | bytes as stored | all |
| `plane` | semantic de-interleave into one plane per **ggml block-struct field** (scales/deltas separated from the packed quant bulk), transcribed field-by-field from `sgl-kernel/csrc/quantization/gguf/ggml-common.h` — see `scripts/dev/306_ratio_probe/blocks.py` for the per-type line citations | GGUF quant payloads |
| `stride<k>` | mechanical de-interleave: byte *j* of every *k*-byte group becomes one plane. `k` = the block size for GGUF payloads (so every byte position in the block gets its own plane — a finer split than `plane`, and one that needs no struct knowledge), `k` ∈ {2,4} for flat payloads | all |
| `nibble` | high/low nibble planes, packed two per byte. The sub-byte analogue of the plane split, and the only split available for 1-byte-element payloads: for FP8-E4M3 the high nibble carries sign + 3 exponent bits, the low nibble 1 exponent + 3 mantissa bits | flat payloads and even-length block payloads |

A layout is a pure **permutation** of the sample's bytes, compressed as a
single frame. One frame per plane would forfeit the shared window, which is the
entire point of grouping like-entropy bytes together.

Additional arms on the `raw` layout: `zstd -3` and `zstd -19` with 16 threads
(compression only — a single zstd frame decompresses on one core regardless of
how many are free), and a `zstd -3` **independent-4 MiB-frame** arm decompressed
across 8 threads, which is the only shape that gets aggregate decompress
bandwidth above one core.

Every `(sample, layout, codec)` triple asserts `restored == original`; a
mismatch aborts the run rather than being reported. All 8 528 triples verified
lossless.

### 3.2 Effort ceiling and entropy ceiling

Because a "not worth it" verdict invites the objection *you did not try hard
enough*, two further arms ran on 2 samples per class
(`scripts/dev/306_ratio_probe/entropy_ceiling.py`):

* **Maximum effort**: `zstd --ultra -22` with a 128 MiB window and long-distance
  matching, and `xz -9e`. These are the strongest generally available settings.
* **Information-theoretic ceiling**: the order-0 byte entropy `H0` of the
  payload, and of each plane. `ceil0 = 8/H0` bounds any memoryless coder; the
  size-weighted per-plane version bounds the byte-plane-split family
  specifically. The bound is one-sided — an LZ coder can beat it by modelling
  repeated substrings — so a measured ratio above `ceil0` is legitimate and is
  reported as such rather than treated as an error.

### 3.3 Host

AMD Ryzen 9 5950X, 32 threads, 104 GB RAM, no swap. `zstd` 1.5.7 via
`zstandard` 0.25.0, `zlib` and `lzma` from CPython 3.12 stdlib. Rates are
MB/s = 1e6 B/s throughout, to match the GB/s link figures. Compression and
decompression are best-of-N wall clock (N chosen to reach ≥0.2 s of work, capped
at 5 reps; the two slow codecs run once). The box was shared with unrelated
background work during the sweep (load average 3.6-4.6 of 32 cores); best-of-N
absorbs most of that, but every rate here should be read as a lower bound on an
idle machine, which only makes the DEAD verdicts more robust and would need
re-taking before any rate was used as a headline number.

## 4. Asset classes: present, and absent

### 4.1 Present — 112 samples, 8 per class, 16 MiB each

Fourteen classes, all extracted with `--seed 306`. The GGUF classes are read as
**raw quantised bytes at the tensor's file offset** — nothing is dequantised,
because the cold tier stores the quantised bytes and dequantising first would
measure the entropy of a representation the tier never holds.

One finding worth stating before the ratios, because it corrects an assumption
carried in `ANALYSE_456` §2.2: **DeepSeek-V4-Flash's routed experts are not
Q3_K**, despite the `UD-Q3_K_XL` directory name. Unsloth's dynamic mixes place
them as follows (`gguf-py` tensor-type enumeration over all four shards of each
variant):

| variant | `ffn_gate_exps` / `ffn_up_exps` | `ffn_down_exps` |
|---|---|---|
| `UD-IQ3_XXS` | IQ2_XS x25, IQ3_XXS x17, IQ3_S x1 (each of gate and up) | IQ3_XXS x41, MXFP4 x2 |
| `UD-Q3_K_XL` | IQ3_XXS x42, MXFP4 x1 (each) | MXFP4 x43 |

So the DSV4-Flash cold-expert asset is an **IQ2_XS / IQ3_XXS / IQ3_S / MXFP4**
payload, and MXFP4 — a 32-element block of one e8m0 scale byte plus 16 packed
bytes — carries the majority of the `UD-Q3_K_XL` expert bytes. Real Q3_K,
Q4_K, Q6_K and IQ4_XS expert samples come from
`Qwen3.6-35B-A3B-MTP-UD-Q3_K_M-GGUF`, the only checkpoint on this box that has
them in routed-expert tensors, so the k-quant family is covered by measurement
rather than by analogy.

### 4.2 SAMPLE-ABSENT — KV and GDN blobs

**No KV-cache or GDN/Mamba state dump exists on this box.** The complete
inventory of binary tensor artifacts under `/spinning/gpu-battery-results` is:

```
   2229 rankN_stepN.pt                       # #343/#345 determinism dumps, ~14 KB each
   1408 full_rankN_astepN_LN.out[N].pt       #   layer activations, logits, embeddings
    704 full_rankN_astepN_LN.o_proj[N].pt
    704 full_rankN_astepN_LN.mlp.pt
    704 full_rankN_astepN_LN.attn_shard.pt
     44 full_rankN_astepN_final_norm[N].pt
     22 full_rankN_astepN_logits.pt / _input_ids.pt / _embed.pt
      1 rankN_GPU-<uuid>.pt                  # the #89 hibernate image (7.18 GB)
```

Nothing matching `kv|gdn|mamba|ssm|conv|cache` in any of those names. The #343
dumps are per-step **activations**, not cache state, and at 14 KB they are not
a cold-tier asset in any case.

These classes are therefore reported **SAMPLE-ABSENT**, not estimated.
Synthesising a KV tensor would produce a synthetic entropy and therefore a
synthetic ratio — precisely the "number supplied from general knowledge" that
`DESIGN_407_memory_tier_registry.md` §1.1 forbids. `ANALYSE_456` §2.2 named
"fp8-KV blocks" and "GDN state blobs" as two of the four asset types this probe
should cover; the honest answer is that this box has never dumped either.

**What it would take to close them.** An fp8-KV block dump and a GDN state dump
off one real serving boot, written to `/spinning/gpu-battery-results` as raw
bytes with the pool geometry recorded next to them. That is a card-window task,
not a desk one, and it is the only thing standing between this document and a
complete asset table. Prior expectation, stated as an expectation and not a
result: fp8-KV is a dense 1-byte-per-element payload very like the FP8 weight
class measured below, so the FP8-weight row is the closest available proxy —
but a proxy is not a measurement and no verdict below is issued on it.

## 5. Results

### A. Ratio matrix -- median over 8 samples per class

Ratio = uncompressed / compressed; > 1 means the codec found something.

| asset class | layout | lzma-fast | zlib-6 | zstd-19 | zstd-19-mt16 | zstd-3 | zstd-3-chunk4M-x8 | zstd-3-mt16 |
|---|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | nibble | 0.9999 | 1.0005 | 1.0000 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | plane | 0.9999 | 1.0014 | 1.0000 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | raw | 0.9999 | 1.0011 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq2_xs` | stride74 | 1.0223 | 1.0294 | 1.0299 | -- | 1.0255 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | nibble | 0.9999 | 1.0007 | 1.0004 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | plane | 1.0011 | 1.0069 | 1.0092 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | raw | 0.9999 | 1.0031 | 1.0054 | 1.0054 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_s` | stride110 | 1.0163 | 1.0260 | 1.0285 | -- | 1.0175 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | nibble | 0.9999 | 1.0006 | 1.0000 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | plane | 1.0002 | 1.0044 | 1.0068 | -- | 1.0001 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | raw | 0.9999 | 1.0043 | 1.0064 | 1.0064 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | stride98 | 1.0207 | 1.0279 | 1.0276 | -- | 1.0289 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | plane | 1.0601 | 1.0671 | 1.0719 | -- | 1.0686 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | raw | 1.0345 | 1.0451 | 1.0494 | 1.0494 | 1.0487 | 1.0489 | 1.0491 |
| `dsv4f_ud_iq3xxs_mxfp4` | stride17 | 1.0772 | 1.0883 | 1.0934 | -- | 1.0877 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | nibble | 0.9999 | 1.0007 | 1.0001 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | plane | 1.0002 | 1.0049 | 1.0074 | -- | 1.0001 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | raw | 0.9999 | 1.0047 | 1.0071 | 1.0071 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_q3kxl_iq3_xxs` | stride98 | 1.0216 | 1.0287 | 1.0290 | -- | 1.0297 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | plane | 1.0387 | 1.0439 | 1.0478 | -- | 1.0456 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | raw | 1.0338 | 1.0433 | 1.0472 | 1.0472 | 1.0454 | 1.0456 | 1.0467 |
| `dsv4f_ud_q3kxl_mxfp4` | stride17 | 1.0751 | 1.0862 | 1.0908 | -- | 1.0846 | -- | -- |
| `hibernate_img` | nibble | 1.0322 | 1.0406 | 1.0456 | -- | 1.0434 | -- | -- |
| `hibernate_img` | raw | 1.0359 | 1.0352 | 1.0389 | 1.0389 | 1.0353 | 1.0356 | 1.0359 |
| `hibernate_img` | stride2 | 1.0367 | 1.0402 | 1.0456 | -- | 1.0410 | -- | -- |
| `hibernate_img` | stride4 | 1.0340 | 1.0408 | 1.0444 | -- | 1.0417 | -- | -- |
| `qwen27b_fp8` | nibble | 1.1399 | 1.1555 | 1.1811 | -- | 1.1806 | -- | -- |
| `qwen27b_fp8` | raw | 1.1559 | 1.2084 | 1.2104 | 1.2104 | 1.2108 | 1.2109 | 1.2109 |
| `qwen27b_fp8` | stride2 | 1.1556 | 1.2081 | 1.2101 | -- | 1.2105 | -- | -- |
| `qwen27b_fp8` | stride4 | 1.1553 | 1.2079 | 1.2096 | -- | 1.2099 | -- | -- |
| `qwen27b_int8` | nibble | 1.0993 | 1.1222 | 1.1239 | -- | 1.1302 | -- | -- |
| `qwen27b_int8` | raw | 1.1060 | 1.1295 | 1.1302 | 1.1302 | 1.1307 | 1.1307 | 1.1307 |
| `qwen27b_int8` | stride2 | 1.1057 | 1.1290 | 1.1301 | -- | 1.1306 | -- | -- |
| `qwen27b_int8` | stride4 | 1.1052 | 1.1286 | 1.1300 | -- | 1.1305 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | nibble | 1.0007 | 1.0015 | 1.0015 | -- | 1.0008 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | plane | 1.0012 | 1.0051 | 1.0073 | -- | 1.0013 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | raw | 1.0010 | 1.0050 | 1.0072 | 1.0072 | 1.0011 | 1.0012 | 1.0011 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | stride98 | 1.0196 | 1.0264 | 1.0258 | -- | 1.0261 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | nibble | 1.0039 | 1.0126 | 1.0081 | -- | 1.0016 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | plane | 1.0061 | 1.0128 | 1.0023 | -- | 1.0010 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | raw | 1.0056 | 1.0127 | 1.0016 | 1.0016 | 1.0016 | 1.0016 | 1.0014 |
| `qwen35ba3b_ud_q3km_iq4_xs` | stride136 | 1.0181 | 1.0275 | 1.0165 | -- | 1.0236 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | nibble | 0.9999 | 0.9997 | 1.0000 | -- | 1.0000 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | plane | 0.9999 | 1.0015 | 1.0022 | -- | 1.0000 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | raw | 0.9999 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `qwen35ba3b_ud_q3km_q3_k` | stride110 | 1.0089 | 1.0132 | 1.0139 | -- | 1.0107 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | nibble | 1.0158 | 1.0222 | 1.0264 | -- | 1.0234 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | plane | 1.0160 | 1.0220 | 1.0248 | -- | 1.0231 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | raw | 1.0155 | 1.0216 | 1.0242 | 1.0242 | 1.0231 | 1.0233 | 1.0237 |
| `qwen35ba3b_ud_q3km_q4_k` | stride144 | 1.0364 | 1.0488 | 1.0518 | -- | 1.0502 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | nibble | 0.9999 | 1.0018 | 1.0025 | -- | 1.0000 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | plane | 1.0100 | 1.0145 | 1.0162 | -- | 1.0159 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | raw | 0.9999 | 1.0024 | 1.0047 | 1.0047 | 1.0000 | 1.0000 | 1.0000 |
| `qwen35ba3b_ud_q3km_q6_k` | stride210 | 1.0209 | 1.0268 | 1.0288 | -- | 1.0281 | -- | -- |

### B. Decompress rate matrix -- median MB/s (1e6 B/s), inverse permutation included

| asset class | layout | lzma-fast | zlib-6 | zstd-19 | zstd-19-mt16 | zstd-3 | zstd-3-chunk4M-x8 | zstd-3-mt16 |
|---|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | nibble | 974 | 444 | 1147 | -- | 1155 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | plane | 1597 | 339 | 2114 | -- | 2161 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | raw | 4016 | 388 | 12830 | 11797 | 12881 | 15330 | 11213 |
| `dsv4f_ud_iq3xxs_iq2_xs` | stride74 | 194 | 357 | 693 | -- | 838 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | nibble | 953 | 435 | 1005 | -- | 1111 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | plane | 60 | 328 | 926 | -- | 2116 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | raw | 3906 | 383 | 1487 | 1468 | 12825 | 16524 | 12157 |
| `dsv4f_ud_iq3xxs_iq3_s` | stride110 | 36 | 270 | 594 | -- | 865 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | nibble | 819 | 406 | 944 | -- | 976 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | plane | 617 | 320 | 936 | -- | 1871 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | raw | 3131 | 370 | 1513 | 1541 | 11019 | 11326 | 9816 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | stride98 | 32 | 275 | 545 | -- | 542 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | plane | 27 | 280 | 788 | -- | 773 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | raw | 27 | 323 | 1460 | 1452 | 1420 | 4935 | 1436 |
| `dsv4f_ud_iq3xxs_mxfp4` | stride17 | 27 | 273 | 695 | -- | 664 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | nibble | 799 | 402 | 925 | -- | 961 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | plane | 655 | 327 | 951 | -- | 2176 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | raw | 3408 | 371 | 1478 | 1474 | 10936 | 11739 | 10393 |
| `dsv4f_ud_q3kxl_iq3_xxs` | stride98 | 32 | 274 | 542 | -- | 539 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | plane | 27 | 290 | 868 | -- | 841 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | raw | 27 | 338 | 1529 | 1504 | 1464 | 4812 | 1482 |
| `dsv4f_ud_q3kxl_mxfp4` | stride17 | 28 | 311 | 896 | -- | 838 | -- | -- |
| `hibernate_img` | nibble | 34 | 283 | 766 | -- | 781 | -- | -- |
| `hibernate_img` | raw | 34 | 368 | 1507 | 1467 | 1497 | 4802 | 1483 |
| `hibernate_img` | stride2 | 32 | 334 | 967 | -- | 1045 | -- | -- |
| `hibernate_img` | stride4 | 51 | 337 | 967 | -- | 1034 | -- | -- |
| `qwen27b_fp8` | nibble | 71 | 374 | 934 | -- | 937 | -- | -- |
| `qwen27b_fp8` | raw | 31 | 324 | 1341 | 1324 | 1336 | 4443 | 1348 |
| `qwen27b_fp8` | stride2 | 31 | 287 | 891 | -- | 908 | -- | -- |
| `qwen27b_fp8` | stride4 | 31 | 288 | 871 | -- | 876 | -- | -- |
| `qwen27b_int8` | nibble | 65 | 391 | 842 | -- | 905 | -- | -- |
| `qwen27b_int8` | raw | 30 | 329 | 1478 | 1475 | 1508 | 4891 | 1509 |
| `qwen27b_int8` | stride2 | 30 | 295 | 962 | -- | 976 | -- | -- |
| `qwen27b_int8` | stride4 | 30 | 290 | 926 | -- | 960 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | nibble | 461 | 399 | 926 | -- | 937 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | plane | 398 | 323 | 942 | -- | 1982 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | raw | 715 | 370 | 1454 | 1413 | 8728 | 9446 | 8921 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | stride98 | 33 | 294 | 621 | -- | 628 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | nibble | 25 | 265 | 738 | -- | 880 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | plane | 26 | 302 | 1568 | -- | 1637 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | raw | 26 | 349 | 6164 | 6069 | 6035 | 10321 | 6150 |
| `qwen35ba3b_ud_q3km_iq4_xs` | stride136 | 25 | 247 | 692 | -- | 598 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | nibble | 723 | 709 | 911 | -- | 915 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | plane | 1227 | 631 | 1347 | -- | 1776 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | raw | 3012 | 2302 | 9699 | 9863 | 10343 | 10588 | 9425 |
| `qwen35ba3b_ud_q3km_q3_k` | stride110 | 238 | 355 | 604 | -- | 697 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | nibble | 27 | 282 | 675 | -- | 684 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | plane | 27 | 327 | 1068 | -- | 1041 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | raw | 27 | 357 | 1486 | 1483 | 1430 | 4864 | 1482 |
| `qwen35ba3b_ud_q3km_q4_k` | stride144 | 27 | 265 | 634 | -- | 630 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | nibble | 1052 | 312 | 914 | -- | 1259 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | plane | 70 | 529 | 1653 | -- | 1649 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | raw | 4409 | 399 | 1542 | 1529 | 13895 | 17771 | 13659 |
| `qwen35ba3b_ud_q3km_q6_k` | stride210 | 69 | 424 | 832 | -- | 823 | -- | -- |

### C. Best achievable ratio per asset class (any method)

| asset class | n | best method | ratio median | ratio min-max | decompress MB/s | compress MB/s | kill criterion (< 1.08) |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 8 | stride74/zstd-19 | **1.0299** | 1.0295-1.0305 | 693 | 7 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_s` | 8 | stride110/zstd-19 | **1.0285** | 1.0283-1.0287 | 594 | 6 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 8 | stride98/zstd-3 | **1.0289** | 1.0283-1.0301 | 542 | 643 | **DEAD** |
| `dsv4f_ud_iq3xxs_mxfp4` | 8 | stride17/zstd-19 | **1.0934** | 1.0892-1.0999 | 695 | 5 | alive |
| `dsv4f_ud_q3kxl_iq3_xxs` | 8 | stride98/zstd-3 | **1.0297** | 1.0291-1.0306 | 539 | 633 | **DEAD** |
| `dsv4f_ud_q3kxl_mxfp4` | 8 | stride17/zstd-19 | **1.0908** | 1.0895-1.0915 | 896 | 6 | alive |
| `hibernate_img` | 8 | nibble/zstd-19 | **1.0456** | 1.0101-20189.1889 | 766 | 7 | **DEAD** |
| `qwen27b_fp8` | 8 | raw/zstd-3-mt16 | **1.2109** | 1.2032-1.2172 | 1348 | 1203 | alive |
| `qwen27b_int8` | 8 | raw/zstd-3 | **1.1307** | 1.1190-1.1438 | 1508 | 1149 | alive |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 8 | stride98/zlib-6 | **1.0264** | 1.0249-1.0435 | 294 | 55 | **DEAD** |
| `qwen35ba3b_ud_q3km_iq4_xs` | 8 | stride136/zlib-6 | **1.0275** | 1.0253-1.0294 | 247 | 51 | **DEAD** |
| `qwen35ba3b_ud_q3km_q3_k` | 8 | stride110/zstd-19 | **1.0139** | 1.0121-1.0144 | 604 | 5 | **DEAD** |
| `qwen35ba3b_ud_q3km_q4_k` | 8 | stride144/zstd-19 | **1.0518** | 1.0340-1.0521 | 634 | 7 | **DEAD** |
| `qwen35ba3b_ud_q3km_q6_k` | 8 | stride210/zstd-19 | **1.0288** | 1.0283-1.0292 | 832 | 7 | **DEAD** |

### D. Cell verdicts -- serial speedup per (asset class, link)

Each cell: the SERIAL speedup of the method that maximises it for that link (pipelined bound in brackets). The no-compression baseline is 1.000x by definition, so > 1.000x is a win and < 1.000x means storing the asset RAW is strictly faster. The method is re-chosen per link, so a cell is the best this probe can do there, not the best-ratio method forced onto a link it does not suit.

| asset class | best ratio (any method) | T3 local NVMe / disk image 1.80 GB/s | T4 remote rig-2 over 40G 2.07 GB/s | T4 remote rig-2 over 40G 2.83 GB/s | T2 host RAM -> card 6.40 GB/s | T2 host RAM -> card 13.00 GB/s | verdict |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 1.0299 | 0.895x [1.000x] raw/zstd-3-chunk4M-x8 | 0.881x [1.000x] raw/zstd-3-chunk4M-x8 | 0.844x [1.000x] raw/zstd-3-chunk4M-x8 | 0.705x [1.000x] raw/zstd-3-chunk4M-x8 | 0.541x [1.000x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_s` | 1.0285 | 0.902x [1.000x] raw/zstd-3-chunk4M-x8 | 0.889x [1.000x] raw/zstd-3-chunk4M-x8 | 0.854x [1.000x] raw/zstd-3-chunk4M-x8 | 0.721x [1.000x] raw/zstd-3-chunk4M-x8 | 0.560x [1.000x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 1.0289 | 0.863x [1.000x] raw/zstd-3-chunk4M-x8 | 0.845x [1.000x] raw/zstd-3-chunk4M-x8 | 0.800x [1.000x] raw/zstd-3-chunk4M-x8 | 0.639x [1.000x] raw/zstd-3-chunk4M-x8 | 0.466x [0.871x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_iq3xxs_mxfp4` | 1.0934 | 0.759x [1.049x] raw/zstd-3-chunk4M-x8 | 0.728x [1.049x] raw/zstd-3-chunk4M-x8 | 0.655x [1.049x] raw/zstd-3-chunk4M-x8 | 0.444x [0.771x] raw/zstd-3-chunk4M-x8 | 0.279x [0.380x] raw/zstd-3-chunk4M-x8 | no win |
| `dsv4f_ud_q3kxl_iq3_xxs` | 1.0297 | 0.867x [1.000x] raw/zstd-3-chunk4M-x8 | 0.850x [1.000x] raw/zstd-3-chunk4M-x8 | 0.806x [1.000x] raw/zstd-3-chunk4M-x8 | 0.647x [1.000x] raw/zstd-3-chunk4M-x8 | 0.474x [0.903x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_q3kxl_mxfp4` | 1.0908 | 0.752x [1.046x] raw/zstd-3-chunk4M-x8 | 0.721x [1.046x] raw/zstd-3-chunk4M-x8 | 0.647x [1.046x] raw/zstd-3-chunk4M-x8 | 0.437x [0.752x] raw/zstd-3-chunk4M-x8 | 0.273x [0.370x] raw/zstd-3-chunk4M-x8 | no win |
| `hibernate_img` | 1.0456 | 0.746x [1.036x] raw/zstd-3-chunk4M-x8 | 0.716x [1.036x] raw/zstd-3-chunk4M-x8 | 0.643x [1.036x] raw/zstd-3-chunk4M-x8 | 0.435x [0.750x] raw/zstd-3-chunk4M-x8 | 0.272x [0.369x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen27b_fp8` | 1.2109 | 0.812x [1.211x] raw/zstd-3-chunk4M-x8 | 0.774x [1.211x] raw/zstd-3-chunk4M-x8 | 0.684x [1.211x] raw/zstd-3-chunk4M-x8 | 0.441x [0.694x] raw/zstd-3-chunk4M-x8 | 0.267x [0.342x] raw/zstd-3-chunk4M-x8 | no win |
| `qwen27b_int8` | 1.1307 | 0.798x [1.131x] raw/zstd-3-chunk4M-x8 | 0.765x [1.131x] raw/zstd-3-chunk4M-x8 | 0.683x [1.131x] raw/zstd-3-chunk4M-x8 | 0.456x [0.764x] raw/zstd-3-chunk4M-x8 | 0.282x [0.376x] raw/zstd-3-chunk4M-x8 | no win |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 1.0264 | 0.841x [1.001x] raw/zstd-3-chunk4M-x8 | 0.821x [1.001x] raw/zstd-3-chunk4M-x8 | 0.770x [1.001x] raw/zstd-3-chunk4M-x8 | 0.597x [1.001x] raw/zstd-3-chunk4M-x8 | 0.421x [0.727x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_iq4_xs` | 1.0275 | 0.853x [1.002x] raw/zstd-3-chunk4M-x8 | 0.834x [1.002x] raw/zstd-3-chunk4M-x8 | 0.786x [1.002x] raw/zstd-3-chunk4M-x8 | 0.618x [1.002x] raw/zstd-3-chunk4M-x8 | 0.443x [0.794x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_q3_k` | 1.0139 | 0.855x [1.000x] raw/zstd-3-chunk4M-x8 | 0.836x [1.000x] raw/zstd-3-chunk4M-x8 | 0.789x [1.000x] raw/zstd-3-chunk4M-x8 | 0.623x [1.000x] raw/zstd-3-chunk4M-x8 | 0.449x [0.814x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_q4_k` | 1.0518 | 0.742x [1.023x] raw/zstd-3-chunk4M-x8 | 0.713x [1.023x] raw/zstd-3-chunk4M-x8 | 0.641x [1.023x] raw/zstd-3-chunk4M-x8 | 0.436x [0.760x] raw/zstd-3-chunk4M-x8 | 0.274x [0.374x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_q6_k` | 1.0288 | 0.908x [1.000x] raw/zstd-3-chunk4M-x8 | 0.896x [1.000x] raw/zstd-3-chunk4M-x8 | 0.863x [1.000x] raw/zstd-3-chunk4M-x8 | 0.735x [1.000x] raw/zstd-3-chunk4M-x8 | 0.578x [1.000x] raw/zstd-3-chunk4M-x8 | **DEAD** |

#### D.1 Required ratio `r_min = D/(D-L)` at the fastest decompress arm

`r_min` is the smallest ratio that could make a serial win, given the decompress rate actually measured. Compare it against the best ratio column above.

| asset class | fastest decompress arm | D (MB/s) | r_min @ 1.80 GB/s | r_min @ 2.07 GB/s | r_min @ 2.83 GB/s | r_min @ 6.40 GB/s | r_min @ 13.00 GB/s |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | raw/zstd-3-chunk4M-x8 | 15330 | 1.133 | 1.156 | 1.226 | 1.717 | 6.579 |
| `dsv4f_ud_iq3xxs_iq3_s` | raw/zstd-3-chunk4M-x8 | 16524 | 1.122 | 1.143 | 1.207 | 1.632 | 4.689 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | raw/zstd-3-chunk4M-x8 | 11326 | 1.189 | 1.224 | 1.333 | 2.299 | impossible |
| `dsv4f_ud_iq3xxs_mxfp4` | raw/zstd-3-chunk4M-x8 | 4935 | 1.574 | 1.722 | 2.344 | impossible | impossible |
| `dsv4f_ud_q3kxl_iq3_xxs` | raw/zstd-3-chunk4M-x8 | 11739 | 1.181 | 1.214 | 1.318 | 2.199 | impossible |
| `dsv4f_ud_q3kxl_mxfp4` | raw/zstd-3-chunk4M-x8 | 4812 | 1.598 | 1.755 | 2.428 | impossible | impossible |
| `hibernate_img` | raw/zstd-3-chunk4M-x8 | 4802 | 1.600 | 1.758 | 2.435 | impossible | impossible |
| `qwen27b_fp8` | raw/zstd-3-chunk4M-x8 | 4443 | 1.681 | 1.872 | 2.755 | impossible | impossible |
| `qwen27b_int8` | raw/zstd-3-chunk4M-x8 | 4891 | 1.582 | 1.734 | 2.373 | impossible | impossible |
| `qwen35ba3b_ud_q3km_iq3_xxs` | raw/zstd-3-chunk4M-x8 | 9446 | 1.235 | 1.281 | 1.428 | 3.101 | impossible |
| `qwen35ba3b_ud_q3km_iq4_xs` | raw/zstd-3-chunk4M-x8 | 10321 | 1.211 | 1.251 | 1.378 | 2.632 | impossible |
| `qwen35ba3b_ud_q3km_q3_k` | raw/zstd-3-chunk4M-x8 | 10588 | 1.205 | 1.243 | 1.365 | 2.528 | impossible |
| `qwen35ba3b_ud_q3km_q4_k` | raw/zstd-3-chunk4M-x8 | 4864 | 1.587 | 1.741 | 2.391 | impossible | impossible |
| `qwen35ba3b_ud_q3km_q6_k` | raw/zstd-3-chunk4M-x8 | 17771 | 1.113 | 1.132 | 1.189 | 1.563 | 3.725 |

### E. Multi-thread and chunked-frame arms (raw layout)

| asset class | zstd-3 1T comp MB/s | zstd-3 16T comp MB/s | zstd-19 1T comp MB/s | zstd-19 16T comp MB/s | zstd-3 1T decomp MB/s | zstd-3 4 MiB frames x8 decomp MB/s | frame-chunk ratio |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 1021 | 1077 | 6 | 6 | 12881 | 15330 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_s` | 1063 | 1149 | 6 | 6 | 12825 | 16524 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 1078 | 1025 | 5 | 5 | 11019 | 11326 | 1.0000 |
| `dsv4f_ud_iq3xxs_mxfp4` | 683 | 856 | 5 | 5 | 1420 | 4935 | 1.0489 |
| `dsv4f_ud_q3kxl_iq3_xxs` | 1023 | 1021 | 5 | 5 | 10936 | 11739 | 1.0000 |
| `dsv4f_ud_q3kxl_mxfp4` | 713 | 946 | 6 | 6 | 1464 | 4812 | 1.0456 |
| `hibernate_img` | 606 | 875 | 7 | 7 | 1497 | 4802 | 1.0356 |
| `qwen27b_fp8` | 1122 | 1203 | 5 | 5 | 1336 | 4443 | 1.2109 |
| `qwen27b_int8` | 1149 | 1211 | 6 | 6 | 1508 | 4891 | 1.1307 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 986 | 1040 | 4 | 4 | 8728 | 9446 | 1.0012 |
| `qwen35ba3b_ud_q3km_iq4_xs` | 626 | 792 | 5 | 5 | 6035 | 10321 | 1.0016 |
| `qwen35ba3b_ud_q3km_q3_k` | 2252 | 1361 | 5 | 5 | 10343 | 10588 | 1.0000 |
| `qwen35ba3b_ud_q3km_q4_k` | 583 | 867 | 7 | 7 | 1430 | 4864 | 1.0233 |
| `qwen35ba3b_ud_q3km_q6_k` | 1103 | 1126 | 7 | 7 | 13895 | 17771 | 1.0000 |

### F. Sample provenance

| asset class | n | bytes/sample | source file(s) | example tensor | format | block bytes |
|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 8 | 16777206 | 2 files, e.g. `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf` | `blk.18.ffn_up_exps.weight` | IQ2_XS | 74 |
| `dsv4f_ud_iq3xxs_iq3_s` | 8 | 16777200 | `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf` | `blk.26.ffn_up_exps.weight` | IQ3_S | 110 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 8 | 16777208 | 2 files, e.g. `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf` | `blk.36.ffn_down_exps.weight` | IQ3_XXS | 98 |
| `dsv4f_ud_iq3xxs_mxfp4` | 8 | 16777215 | 2 files, e.g. `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf` | `blk.26.ffn_down_exps.weight` | MXFP4 | 17 |
| `dsv4f_ud_q3kxl_iq3_xxs` | 8 | 16777208 | 3 files, e.g. `DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00002-of-00004.gguf` | `blk.32.ffn_up_exps.weight` | IQ3_XXS | 98 |
| `dsv4f_ud_q3kxl_mxfp4` | 8 | 16777215 | 3 files, e.g. `DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00002-of-00004.gguf` | `blk.0.ffn_down_exps.weight` | MXFP4 | 17 |
| `hibernate_img` | 8 | 16777216 | `rank0_GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d.pt` | `<image chunk 0>` | mixed-Q4_K/Q6_K/F32 | n/a (flat 1-byte elements) |
| `qwen27b_fp8` | 8 | 16777216 | 8 files, e.g. `layers-15.safetensors` | `model.language_model.layers.5.mlp.up_proj.weight` | F8_E4M3 | n/a (flat 1-byte elements) |
| `qwen27b_int8` | 8 | 16777216 | `model.safetensors` | `model.language_model.layers.2.linear_attn.in_proj_z.weight` | I8 | n/a (flat 1-byte elements) |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 8 | 16777208 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.4.ffn_gate_exps.weight` | IQ3_XXS | 98 |
| `qwen35ba3b_ud_q3km_iq4_xs` | 8 | 16777096 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.20.ffn_down_exps.weight` | IQ4_XS | 136 |
| `qwen35ba3b_ud_q3km_q3_k` | 8 | 16777200 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.40.ffn_gate_exps.weight` | Q3_K | 110 |
| `qwen35ba3b_ud_q3km_q4_k` | 8 | 16777152 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.40.ffn_down_exps.weight` | Q4_K | 144 |
| `qwen35ba3b_ud_q3km_q6_k` | 8 | 16777110 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.34.ffn_down_exps.weight` | Q6_K | 210 |

### G. Link-rate provenance (all MEASURED, none invented)

| link | rate | source |
|---|---|---|
| T3 local NVMe / disk image, cold read | 1.80 GB/s | `ANALYSE_389_nvme_expert_tier.md` §(b), `iflag=direct`, reproduced 3x; tier table `DESIGN_407_memory_tier_registry.md:135` |
| T4 remote rig-2 over 40G, NCCL-over-sockets | 2.07 GB/s | `NOTE_453_remote_expert_lane.md:9-10` / `INTEGRATION_R3_VALIDATION.md:5053`; tier table `DESIGN_407_memory_tier_registry.md:137` |
| T4 remote rig-2 over 40G, staged RDMA 1 MiB | 2.83 GB/s | tier table `DESIGN_407_memory_tier_registry.md:137` |
| T2 host RAM -> card, PCIe H2D pinned, gen4 x4 | 6.40 GB/s | `ANALYSE_393_ik_llama.md:301-304`; tier table `DESIGN_407_memory_tier_registry.md:136` |
| T2 host RAM -> card, PCIe H2D pinned, gen4 x8 | 13.00 GB/s | `ANALYSE_393_ik_llama.md:301-304`; tier table `DESIGN_407_memory_tier_registry.md:136` |

### H. Entropy ceiling and maximum-effort arms (2 samples per class)

`H0` is the order-0 byte entropy of the raw payload; `ceil0 = 8/H0` bounds any
memoryless byte coder. `ceil0-split` is the size-weighted per-plane version
(the bound on the byte-plane-split family). `zstd-22-long` is `--ultra -22`
with `windowLog=27` and long-distance matching; `xz -9e` is LZMA at maximum
effort. A measured ratio ABOVE `ceil0` is legitimate: LZ models order, which an
order-0 bound does not cover.

| asset class | H0 (bits/byte) | ceil0 | ceil0-split | zstd-22-long | xz -9e | best from the main sweep |
|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 7.9455 | 1.0069 | 1.0073 | 1.0000 | 0.9999 | 1.0299 |
| `dsv4f_ud_iq3xxs_iq3_s` | 7.9222 | 1.0098 | 1.0136 | 1.0054 | 0.9999 | 1.0285 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 7.9129 | 1.0110 | 1.0111 | 1.0056 | 0.9999 | 1.0289 |
| `dsv4f_ud_iq3xxs_mxfp4` | 7.5757 | 1.0560 | 1.0795 | 1.0505 | 1.0442 | 1.0934 |
| `dsv4f_ud_q3kxl_iq3_xxs` | 7.9102 | 1.0113 | 1.0121 | 1.0071 | 0.9999 | 1.0297 |
| `dsv4f_ud_q3kxl_mxfp4` | 7.5950 | 1.0533 | 1.0540 | 1.0474 | 1.0431 | 1.0908 |
| `hibernate_img` | 7.7754 | 1.0289 | 1.0334 | 1.0388 | 1.0463 | 1.0456 |
| `qwen27b_fp8` | 6.5366 | 1.2239 | 1.2239 | 1.2096 | 1.2076 | 1.2109 |
| `qwen27b_int8` | 6.9884 | 1.1448 | 1.1448 | 1.1329 | 1.1321 | 1.1307 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 7.9170 | 1.0105 | 1.0106 | 1.0077 | 1.0021 | 1.0264 |
| `qwen35ba3b_ud_q3km_iq4_xs` | 7.8621 | 1.0175 | 1.0176 | 1.0054 | 1.0106 | 1.0275 |
| `qwen35ba3b_ud_q3km_q3_k` | 7.9740 | 1.0033 | 1.0054 | 1.0004 | 1.0002 | 1.0139 |
| `qwen35ba3b_ud_q3km_q4_k` | 7.7786 | 1.0285 | 1.0289 | 1.0244 | 1.0215 | 1.0518 |
| `qwen35ba3b_ud_q3km_q6_k` | 7.9306 | 1.0088 | 1.0219 | 1.0048 | 0.9999 | 1.0288 |

### I. Decompress rate vs worker count (independent 4 MiB `zstd -3` frames)

The 8-worker figure used everywhere above is not a tuning choice that could be
improved on: aggregate decompress **saturates at 4 workers**, because past that
point the limit is DRAM write bandwidth for the decompressed output, not codec
CPU. `D_max` is therefore a property of the box, and it fixes `r_min` for every
link. The last row is a DEAD control (ratio 1.0000, i.e. stored blocks).

| asset class | ratio | 1T | 2T | 4T | 8T | 16T | 32T | D_max (MB/s) | r_min @ 1.8 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| `qwen27b_fp8` | 1.2070 | 1187 | 2314 | 4262 | 4238 | 4314 | 4249 | 4314 | 1.716 |
| `qwen27b_int8` | 1.1336 | 1335 | 2570 | 4838 | 4761 | 4731 | 4659 | 4838 | 1.592 |
| `dsv4f_ud_q3kxl_mxfp4` | 1.0456 | 1284 | 2500 | 4415 | 4473 | 4401 | 4503 | 4503 | 1.666 |
| `dsv4f_ud_q3kxl_iq3_xxs` | 1.0000 | 12790 | 15400 | 14608 | 18671 | 18878 | 18321 | 18878 | 1.105 |

### J. Whole-image measurement for the disk-image class

A 16 MiB chunk sample is the right unit for an expert tensor -- that is the
granule a cold-tier miss moves -- and the wrong one for a hibernate image,
which is written and read whole. The image is also not homogeneous: two of the
eight sampled chunks landed in an all-zero region and returned ratios of
**20 141x** and **19 174x**, while the other six returned 1.008-1.286. A median
of chunk ratios (1.046) and a pooled ratio over the same eight chunks (1.413)
therefore disagree by 35 %, and neither is the image's actual ratio. So the
whole 7.18 GB file was streamed (`image_whole.py`).

| measurement | value |
|---|---|
| image size | 7,177,133,323 B (6.68 GiB) |
| all-zero 4 KiB pages | 221,542 of 1,752,229 = **12.64%** |
| ratio from **sparse write alone**, no codec | **1.1447x** |
| order-0 entropy, whole image | 7.3823 bits/byte (ceiling 1.0837x) |
| `zstd -3`, 1 thread | ratio **1.1953**, compress 567 MB/s, decompress 915 MB/s |
| `zstd -3`, 16 threads | ratio 1.1947, compress 1565 MB/s, decompress 876 MB/s |
| `zstd -19`, 16 threads | ratio **1.2055**, compress 35 MB/s, decompress 754 MB/s |
| **already delivered by the ZFS dataset** (allocated vs apparent) | **1.1735x** |

The decomposition is the point. Of the 1.1953x that `zstd -3` achieves on the
whole image, **1.1447x is holes** — recoverable by a sparse write or
`fallocate(FALLOC_FL_PUNCH_HOLE)` at zero CPU and with no decompress stage on
the reload path at all. The residual codec gain on the non-hole part is
**1.0442x**. And the filesystem is already returning 1.1735x of the 1.1953x by
itself, so an application-level codec on this tier is worth **1.0186x — 1.9 %**.

The single-stream decompress rates in the table (915 MB/s) are below the
1.8 GB/s NVMe read they would have to beat; a chunked-frame layout would reach
the 4.80 GB/s of table E's `hibernate_img` row, which clears the link but leaves
`r_min = 1.600` against a 1.195 ratio. Serial: no win. Pipelined: 1.195x, of
which the filesystem already holds 1.174x.

Where the holes come from is worth naming for #89's benefit: the image is a
`torch.save` ZIP_STORED container for a Qwen3.5-9B Q4_K_M rank, and 12.6 % of
it is zero pages — pre-allocated buffers written out in full. Nothing about
that needs a codec to fix.

## 6. Verdicts per cell

The tables above give the arithmetic; this is what it decides, per cell of
`ANALYSE_456` §2.2's placement guidance.

### 6.1 (a) 40G remote tier — T4, 2.07 GB/s measured

**DEAD for the expert asset. ALIVE-under-a-pipelined-mover for FP8/INT8
weights, at 1.211x / 1.131x.**

* Serial: `r_min = D/(D-L)` with `D_max = 4.3-4.8 GB/s` gives **1.73-1.87**.
  Nothing measured reaches it — the best is FP8 at 1.211. Serial verdict is
  NO WIN for every class, and every serial cell in table D is **below 1.000x**,
  meaning storing raw is strictly faster.
* Pipelined: needs only `r > 1` and `D > L`, both satisfied for FP8 (1.211x),
  INT8 (1.131x) and marginally MXFP4 (1.049x). For the IQ/k-quant expert
  classes the pipelined bound is **1.000-1.002x**: nothing to overlap.
* The faster remote paths make it worse, not better: at the staged-RDMA
  2.83 GB/s the pipelined figures are unchanged (they are ratio-limited, not
  rate-limited) but `r_min` rises to 2.37-2.76, so serial recedes further.

### 6.2 (b) NVMe / disk image — T3, 1.8 GB/s measured cold

**DEAD as a codec on this box, because the filesystem already took the win.**

This is the one cell where the measurement changed the question. The
`/spinning` ZFS dataset has transparent compression active — its own
allocated-vs-apparent block accounting, read with `stat`/`du`, shows:

| file | apparent | allocated | filesystem ratio already delivered |
|---|---|---|---|
| `...UD-Q3_K_XL-00004-of-00004.gguf` | 29 317 659 776 | 29 332 652 544 | **0.9995** (nothing, as predicted) |
| `Qwen3.6-27B-FP8/layers-10.safetensors` | 383 865 472 | 336 126 464 | **1.1420** |
| `Qwen3.6-27B-INT8-W8A8/mtp.safetensors` | 849 400 424 | 690 782 720 | **1.2296** |
| #89 hibernate image `rank0_GPU-….pt` | 7 177 133 323 | 6 115 713 536 | **1.1735** |

Application-level `zstd -3` over the whole hibernate image reaches **1.1953x**
(measured streaming, table J). Against 1.1735x already delivered by the
filesystem, the marginal gain of building #306 for this tier is **1.9 %** —
paid for with a decompress stage on the reload path that the filesystem's own
decompression already covers, transparently, at block granularity.

Scope of this finding, stated because it is configuration-dependent: it is a
property of the `/spinning` dataset's compression setting, not of the assets.
On a filesystem without transparent compression the 1.1953x would be there to
take — but it would still be a pipelined-only win (`r_min` at 1.8 GB/s is
1.60), and 1.145x of it would still be free from hole-punching alone.

### 6.3 (c) PCIe H2D path — T2, 6.4 / 13 GB/s measured

**ABSENT — no measured GPU-side decompression rate exists, and CPU-side
decompression cannot help here by construction.**

`ANALYSE_456` §2.2 already states the structural half: CPU-decompress followed
by a raw H2D transfer moves the *decompressed* size across PCIe, so it saves
host RAM capacity and zero link bytes. This probe adds the arithmetic for the
other half, the GPU-side (nvcomp-class) case, and it is unfavourable
independently of any nvcomp rate. Using the host's own measured decompress
rates: at `L = 6.4 GB/s` the serial `r_min` is **1.563-3.101** where it is
finite at all, and **impossible** (`D <= L`) for **6 of the 14** classes; at
`L = 13 GB/s` it is **3.725-6.579** where finite and impossible for **11 of the
14**. A GPU decompressor would raise `D` and shrink those `r_min` values toward
1, but the ratios do not move — and the best ratio on the box, 1.211, is short
of what any of them asks.

The nvcomp-class rate itself is **ABSENT**, not estimated: nvcomp is not
installed on this box, no measurement of it exists anywhere in the tree
(`grep -rn nvcomp docs/ python/ scripts/` returns exactly one narrative
mention, `ANALYSE_456_dsv4f_matrix_sweep.md:174`), and this document does not
invent one. Should a GPU decompressor ever be priced, the numbers it has to
beat are in table D.1, and the ratio side of the inequality is already fixed by
this document: the best ratio any asset on this box offers is 1.211, so a GPU
decompressor only helps on the PCIe path if it is fast enough to make
`r_min < 1.211` — i.e. `D > L * 1.211/0.211 = 5.74 L`, meaning **36.8 GB/s at
the x4 link and 74.6 GB/s at x8**. That is the concrete bar; whether nvcomp
clears it is unmeasured here.

### 6.4 Summary table

| cell | expert asset (DSV4-Flash IQ/MXFP4) | k-quant experts (Qwen MoE) | FP8 / INT8 weights | hibernate image | KV / GDN |
|---|---|---|---|---|---|
| **40G remote, serial** | DEAD | DEAD | no win (r_min 1.73-1.87 vs 1.211) | no win | SAMPLE-ABSENT |
| **40G remote, pipelined** | DEAD (<=1.049x) | DEAD (<=1.002x) | **ALIVE 1.131-1.211x** | 1.036x, below kill line | SAMPLE-ABSENT |
| **NVMe / disk, serial** | DEAD | DEAD | no win (r_min 1.58-1.72) | no win | SAMPLE-ABSENT |
| **NVMe / disk, pipelined** | DEAD | DEAD | ALIVE on paper, but see 6.2: the FS already delivers 1.142-1.230 | **1.9 % beyond the FS** — not worth a codec; **1.145x free from holes** | SAMPLE-ABSENT |
| **PCIe H2D, CPU decompress** | N/A — moves the full size on the link | N/A | N/A | N/A | SAMPLE-ABSENT |
| **PCIe H2D, GPU decompress** | ABSENT (no nvcomp rate); ratio 1.03-1.09 vs r_min 1.56+ | ABSENT; ratio <=1.05 | ABSENT; ratio 1.211 vs r_min 1.68 | ABSENT | SAMPLE-ABSENT |

## 7. What this means for the #407 registry

`DESIGN_407_memtier_registry.md` §5 cut 9 pairs "#389 NVMe expert tier, #306
cold compression" and scopes both as "declare the tier and the capability flag
— both become a data change". This probe supplies the data for the second half.

**The rows to register**, each tagged `measured` with this document as the
source, on `TierDescriptor.properties`
(`python/sglang/srt/memtier/tiers.py`, the field whose docstring already says
"name-derived capability checks land here rather than in an `if name ==`"):

| tier / asset class | property | value | provenance |
|---|---|---|---|
| any tier holding **GGUF quantised expert bytes** (IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS, Q3_K, Q4_K, Q6_K) | `lossless_ratio` | **1.014-1.052** (median per format, table C) | `measured` — this document, `results.jsonl`, 8 samples x 16 MiB per format |
| same | `compressible` | **false** — below the 1.08 kill line under every method | `measured` |
| any tier holding **MXFP4 expert bytes** | `lossless_ratio` | **1.046 raw / 1.091 stride-split** | `measured` |
| same | `compressible` | **false** — the 1.091 arm decompresses at 0.70-0.90 GB/s, under every link | `measured` |
| any tier holding **FP8-E4M3 weight bytes** | `lossless_ratio` | **1.211** | `measured` |
| any tier holding **INT8 weight bytes** | `lossless_ratio` | **1.131** | `measured` |
| **T3 local NVMe / disk image** | `fs_transparent_compression_ratio` | **0.9995 (GGUF) / 1.142-1.230 (safetensors) / 1.1735 (hibernate image)** | `measured` — `stat`/`du` allocated-vs-apparent, §6.2 |
| **T3**, hibernate image specifically | `zero_page_fraction` | **0.12643** (221 542 of 1 752 229 4-KiB pages) | `measured` — `image_whole.py` full-file scan |
| any tier, **decompress rate ceiling** | `cpu_decompress_gbs` | **4.3-4.8** (saturates at 4 workers; DRAM-write-bound) | `measured` — table I |
| any tier holding **KV blocks** | `lossless_ratio` | — | **`absent`** — no KV dump exists on this box; probe is a serving-boot dump (§4.2) |
| any tier holding **GDN state blobs** | `lossless_ratio` | — | **`absent`** — same probe missing |
| **GPU-side decompress (nvcomp class)** | `gpu_decompress_gbs` | — | **`absent`** — not installed, never measured, one narrative mention in the tree (§6.3) |

The provenance rules from `DESIGN_407_memtier_registry.md` §3.2 apply
unchanged and are worth naming against this specific data:

* **A number is never re-labelled.** The ratio measured on DSV4-Flash IQ3_XXS
  expert bytes is a property of *that payload*, not of "the cold tier" or of
  "quantised weights" in general. A future INT4 GPTQ or NVFP4 asset needs its
  own row; the entropy of a quantisation format is not transferable across
  formats, and the spread measured here is the evidence for that — from **1.014
  (Q3_K) to 1.211 (FP8-E4M3)**, a **15x** difference in the surplus over 1.0
  between two formats that a single "quantised weights" row would have blended.
* **A row that did not succeed yields no value.** The KV and GDN rows stay
  `absent` naming their missing probe (a serving-boot dump), not `estimate`
  derived from the FP8-weight proxy.
* **A unit is never converted on a guess.** Ratios here are
  uncompressed/compressed on 16 MiB block-aligned slices. A whole-tensor or
  whole-checkpoint ratio is a different measurement (longer match distances,
  cross-expert redundancy) and must not be inferred from these numbers.

## 8. What this means for #126 (the lossy bucket)

`ROADMAP_456_matrix_execution.md`'s lossy bucket gates #126 on "**(a)** every
lossless gain above landed (WAVE 0-4, in particular #306)". This document
discharges the #306 half of gate (a) — not by landing a gain, but by
establishing that on the expert asset **there is no lossless gain to land**.

That is the correct way for the quality-last rule to be satisfied here. The
rule exists so that a lossy feature is never adopted while a byte-identical
alternative is still on the table. On DSV4-Flash routed experts the
byte-identical alternative has now been measured and is worth
**1.026-1.052x under a pipelined mover and negative under a serial one** — against a 1.59-1.72 serial break-even it does not reach, on a payload whose order-0 entropy is 7.91-7.97 of 8 bits per byte — it is not on the table. Whatever #126 is eventually judged
on, "you should have taken the lossless win first" is no longer an open
objection for this asset class.

Two things this does **not** do:

* It does not advance #126 past gate **(b)**, the quality gate. Gate (b) is
  independent and untouched by anything measured here.
* It does not advance the *other* items in gate (a). #439, heat migration
  (#302a), DSpark and the rest are unaffected; only the #306 line moves.

One measured fact does bear directly on #126's own arithmetic: the reason
lossless compression finds nothing is that the payload is already at its
entropy ceiling. A lossy tier gets its bytes back by *lowering the ceiling* —
re-quantising to fewer bits per weight — which is a different mechanism, not a
harder version of the same one. #126's bytes-per-miss estimate should therefore
be computed from the target bit width directly, and must not be stacked on top
of any lossless factor from this document.
