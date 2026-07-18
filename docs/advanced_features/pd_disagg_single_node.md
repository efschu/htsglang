# Single-Node PD-Disaggregation: Solo Prefill on the Big Card, Distributed Decode (Design)

Task #99. Status: **design APPROVED (main ruling 2026-07-18); Phase 2 in
progress (M0 done).** Ruling highlights: A3B-only v1, spec decode
auto-disabled+warn in disagg mode, local ~100-line proxy instead of
sglang_router (interface designed for later rig-dashboard folding: plain
HTTP, health passthrough, per-phase metric counters), M3 must include a
short-prompt TTFT check, M4 must prove 0-MiB teardown incl. hard-kill of
the prefill server mid-transfer (IPC leak check). Branch `feat/pd-disagg`
(worktree `/spinning/wt-pd-disagg`), based on `feat/gemma-draft-spec`
(`e2228882f` — strict superset of `feat/gemma-bringup`, adds the #100
uneven-TP MLP fix, #91 SWA pool sizing, #101 EAGLE3 head support).

## 0. TL;DR and headline measurement

On this rig (RTX 5090 + 2x RTX 3080, one 3080 on PCIe x4), prefill under
uneven TP=3 is communication-bound: the per-layer all-reduce payload grows
with batch tokens and crosses a PCIe Gen4 x4 link. Decode is
bandwidth/KV-bound and *wants* the distributed layout (3 cards' aggregate
bandwidth, big weighted-DCP KV pool). Quantized MoE models fit entirely on
the 5090 (A3B-AWQ ~19.2 GB weights in 32.6 GB), and MoE prefill activates
nearly all experts, so solo prefill wins twice.

**Measured (2026-07-18, this branch, zero code changes):** Qwen3.6-35B-A3B-AWQ,
fp8 KV, chunked prefill 2048, flashinfer, no cuda-graph. Both sides measured
warm on the SAME branch/config (M0 correction: the historical "177 s / 120k"
figure — ~680 tok/s — is stale; the current branch's TP=3+DCP baseline is
much faster, and it is the honest comparison):

| prompt tokens | TP=3+DCP baseline (this branch) | solo 5090 TP=1 | speedup |
|---|---|---|---|
| 8,192 (warm) | 2.04 s (4,022 tok/s) | 0.41 s (20,160 tok/s) | **5.0x** |
| 32,768 | 8.81 s (3,717 tok/s) | 2.03 s (16,100 tok/s) | **4.3x** |
| 81,920 | — | 7.79 s (10,520 tok/s) | — |
| 122,880 | 39.4 s (3,117 tok/s) | **15.2 s** (8,090 tok/s; runs 15.18/15.19) | **2.6x** |

TP=3 decode baseline (same no-graph config, bs=1): 15.9 tok/s @2k ctx,
15.6 tok/s @32k ctx — M3 decode comparisons stay config-matched.

The KV that must then be rescattered to the decode layout is 10.1 KiB/token
(fp8) for A3B — **1.2 GB for the whole 120k prompt**, i.e. ~0.2 s of PCIe
time against 24 s of prefill saved at 120k (and ~1.6 s saved for ~30 ms of
transfer at 8k). Net win at every measured length: inside the original 2–4x
hypothesis band at long context, above it at short lengths. **GO** for A3B;
27B is conditional (Section 8).

## 1. Rig and model facts (measured / from configs)

- GPUs (torch enumeration, fastest-first — `--rank-gpu-id` indexes THIS
  order, per the #102/T102 finding): 0 = RTX 5090 32.6 GB (PCIe Gen4 x8,
  ~13–14 GB/s), 1 = RTX 3080 20.5 GB (Gen4 **x4**, ~6.5 GB/s), 2 = RTX 3080
  20.5 GB (Gen4 x8, ~13 GB/s). HTCCL-measured host-staged DMA:
  14.3 / 6.5 / 13.2 GB/s. No NVLink, no CUDA P2P (GeForce, PHB topology):
  all cross-GPU traffic is host-staged.
- **A3B (Qwen3.6-35B-A3B-AWQ-4bit, primary):** 40 layers, 10 full-attention
  (interval 4), 30 GDN; 2 kv-heads x head_dim 256; 256 experts / 8 active;
  weights ~19.2 GB in VRAM. KV fp8: `10 layers x 2 heads x 256 x 2 (K+V) x 1 B
  = 10 KiB/token` (confirmed by boot log: 85,119 tokens = 0.82 GB).
  GDN state ~32 MB/request (boot log: ssm 0.70 GB + conv 0.03 GB for 23 slots).
  No MTP tensors in this checkpoint.
- **27B (Qwen3.6-27B, secondary):** 64 layers, 16 full-attention; 4 kv-heads
  x 256. KV fp8: 32 KiB/token (3.9 GB per 120k). GDN state ~75 MB/req.
  Weights: FP8 ~25 GB (solo fit marginal), GPTQ/AWQ-int4 ~15.5 GB (fits).
  MTP variant exists (draft = 1 full-attention layer).

## 2. What exists upstream, and what breaks here

sglang already ships full PD-disaggregation plumbing under
`python/sglang/srt/disaggregation/`:

**Reusable as-is**
- Roles + scheduler integration: `--disaggregation-mode prefill|decode`,
  `PrefillBootstrapQueue`, per-chunk KV send (`send_kv_chunk`,
  `prefill.py:1012`), inflight polling + radix release; decode
  `DecodePreallocQueue`/`DecodeTransferQueue`, commit into (optionally)
  radix-tracked pages (`--disaggregation-decode-enable-radix-cache`).
- Bootstrap/handshake: HTTP bootstrap server (`common/conn.py:1408`) + ZMQ
  per-rank registration; works on localhost (rewrites 0.0.0.0→127.0.0.1);
  roles need distinct ports only.
- Spec-decode metadata (topk + hidden states) via `MetadataBuffers`
  (`utils.py:222`); Mamba state transfer with TP-dim slicing
  (`mooncake/conn.py:1140` `_send_mamba_state_slice`).
- The GPU staging fast-path (`common/staging_buffer.py` /
  `staging_handler.py`, upstream #19890/#22536): gather-on-sender →
  one bulk transfer → scatter-on-receiver, ring allocator with watermark
  backpressure. This is exactly the right skeleton for our rescatter.

**Broken assumptions on this rig / this fork**
1. **Transport:** mooncake needs RDMA or NVLink custom mem pools (no plain
   `cudaMemcpy` intra-node path); nixl/UCX could do intra-node CUDA IPC but
   neither wheel is installed, and both are heavy for a 3-GPU localhost case.
2. **Uniform-head math:** every TP-mismatch reslice path
   (`send_kvcache_slice`, `compute_head_slice_params`) assumes
   `heads_per_rank = total // tp_size`. Our decode layout is neither: for
   A3B (kv=2 < tp=3) and for 27B under uneven-DCP, the fork **replicates all
   kv-heads on every rank and shards along the token axis** (weighted DCP,
   `distributed/utils.py:143-159`).
3. **Decode-side CP:** upstream receive path hard-asserts decode
   `attn_cp_size == 1` (`common/conn.py:501`). Fork DCP is a separate axis
   (registers as attn_tp=3, attn_cp=1), so we pass the assert — but nothing
   upstream understands token-vector ownership; that logic must come from
   the fork's own owner rule.
4. **Mamba slicing is uniform:** `_send_mamba_state_slice` divides the head
   dim by `tp`; the fork's GDN partition is prefix-sum uneven
   (`partition_sizes`/`partition_offsets`, `distributed/utils.py`).
5. **Same-GPU co-location:** upstream never intends prefill and a decode
   rank to share one GPU. VRAM budgeting (Section 5) and SM contention
   (Section 7) are ours to manage; the fork's `--rank-gpu-memory-mib`
   budgets both processes independently, which is sufficient.

## 3. Chosen architecture: two processes, standard PD roles, new `local` transfer backend

```
                    router (tiny local proxy / sglang-router)
                          |                     |
        +-----------------+          +-------------------------+
        | PREFILL server  |          | DECODE server            |
        | TP=1, 5090 only |          | TP=3 uneven + weighted   |
        | full weights    |  KV/state| DCP over 5090+3080+3080  |
        | own radix cache |  ------> | replicated kv-heads,     |
        | budget ~22 GB   |  local   | token-sharded KV         |
        +-----------------+  backend +-------------------------+
                 (5090 is shared: prefill instance + decode rank 0)
```

**Rejected: in-process "asymmetric prefill"** (rank 0 holds an extra full
model, ranks 1/2 idle during prefill). Reasons: (a) marlin/AWQ repacked
shards are not views of the full tensors — rank 0 would carry full 19.2 GB
*plus* its ~6.4 GB decode shard, blowing the 32.6 GB card once KV/workspace
are added; (b) divergent per-rank control flow inside one scheduler step is
deep surgery in scheduler + model_runner, impossible to gate cleanly;
(c) the two-process design reuses ~all existing disagg orchestration.

**Transport: `mooncake_tcp` (M1 empirical revision).** The design draft
proposed a custom `TransferBackend.LOCAL`; M1 found this unnecessary for
correctness: sglang ships a `mooncake_tcp` backend alias (mooncake with
`MC_FORCE_TCP=1`, `arg_groups/pd_disaggregation_hook.py:17-27`) that moves
KV over TCP loopback with no RDMA/NVLink requirement, and the
`mooncake-transfer-engine` wheel installs cleanly (0.3.11.post1, no dep
conflicts). M1 validated the full pipeline with it: prefill TP=1 + decode
TP=1 co-resident on the 5090 (17.0 GB combined, sequential boot required —
concurrent boots race each other's memory profiling), local proxy in front,
and **all four temp-0 oracle outputs bit-identical to a non-disagg server,
including a 28k-token needle retrieval** (KV per-chunk transfer + hybrid
GDN state transfer both correct). v1 therefore uses `mooncake_tcp` as the
byte mover; a custom LOCAL backend (CUDA-IPC same-GPU gather-scatter,
pinned-host staging cross-GPU, sketched below) is demoted to an M3+
optimization if TCP-loopback throughput proves limiting for 120k prompts:

- **Registration:** each decode rank exports its KV pool (and state pool)
  base allocation via `cudaIpcGetMemHandle` in the ZMQ registration payload
  (pools are single large allocations — IPC-friendly; requires
  non-expandable-segments allocator, see risks).
- **Same-GPU path (decode rank 0 = 5090):** prefill process maps rank 0's
  pool via IPC and runs a fused gather-scatter kernel (src pool rows →
  dst pool rows through both index lists) — device-to-device at memory
  bandwidth, no host bounce.
- **Cross-GPU path (3080 ranks):** reuse the upstream staging pipeline:
  gather kernel on the 5090 into a contiguous staging buffer → one
  `cudaMemcpyAsync` into a pinned host bounce → H2D into the decode rank's
  staging ring → existing scatter kernel into pool pages. Watermark
  backpressure inherited from `StagingAllocator`. Sequential/pipelined per
  chunk; bandwidth ceiling is the x4 card at ~6.5 GB/s, which per Section 6
  is never the bottleneck.

## 4. The core problem: KV rescatter to the uneven-DCP layout

Decisive simplification found in Phase 1: **for both target configs the
decode layout replicates all kv-heads per rank** (A3B: `attn_kv_replicated`
because kv=2 < tp=3, `model_config.py:1139`; 27B: uneven-DCP mode also
replicates heads and token-shards, `distributed/utils.py:143`). Therefore a
prefill (TP=1) KV row for token L is **byte-identical** to the row the
owning decode rank needs — same heads, same head_dim, same fp8 dtype. The
rescatter is a pure *token routing* problem, no head math:

- **Owner rule (weighted):** token slot L belongs to rank r iff
  `(L mod S) in [lo_r, hi_r)` where S = sum(token vector), with compact
  local slot `(L div S)*(hi_r-lo_r) + (L mod S) - lo_r`
  (`uneven_dcp_owner_bounds`, `cp_token_prefix`). **Owner rule (even):**
  round-robin `L mod dcp_size` (`layers/dcp/layout.py:29`). The sender-side
  filter must implement both; both formulas already exist in the fork.
- **Sender:** replace `filter_kv_indices_for_cp_rank` (upstream contiguous
  CP split) with a token-vector filter in the send path; each chunk of 2048
  tokens maps cleanly since chunk size is a multiple of S (S = gcd-reduced
  64-unit vector; enforce/pad at the boundary otherwise).
- **Receiver:** decode prealloc must allocate the *per-rank owned count*
  (`get_dcp_lens`) instead of full seq_len rows — the normal decode path
  already sizes pools this way; the disagg `pop_preallocated` path needs the
  same formula (verification item V1).
- **GDN/Mamba state:** solo prefill holds the full 32-head state; decode
  ranks hold prefix-sum uneven head shards. Extend
  `_send_mamba_state_slice`'s `dim // tp` math to fork `partition_offsets`.
  ~32 MB/req (A3B), sent once with the last chunk — negligible.
- **What is NOT supported in v1:** plain uneven TP *without* DCP for models
  with kv >= tp (27B without `SGLANG_UNEVEN_DCP`): that layout head-shards
  KV unevenly and would need general uneven head reslicing. Hard-error with
  a clear message: single-node PD requires the DCP (replicated-head) decode
  layout.

## 5. VRAM coexistence on the 5090

Measured: a solo A3B prefill instance boots at 26.2 GB with a 26,000 MiB
budget and 6.2 GB spare — i.e. **~20–21 GB floor** (19.2 weights + 0.7 GB
mamba + 0.8 GB KV + workspace/JIT/context). A *tuned* prefill instance
(KV pool sized to ~1.3 GB for one 131k prompt in flight + small radix
window, 8 mamba slots) needs **~22–23 GB**.

| option | verdict |
|---|---|
| (a) both resident, static budgets | **WORKS for A3B — chosen for v1.** Prefill ~22.5 GB + decode rank 0 ~9–10 GB (even weight ratio 1:1:1 → 6.4 GB shard + 2–3 GB KV/GDN/workspace) ≈ 32 GB. The token vector shifts KV ownership to the 3080s (each has ~13 GB free after its shard; A3B KV is tiny anyway — 4x131k ctx ≈ 5.2 GB *total*). Cost: decode ratio can't favor the 5090; expect some decode tok/s dip vs the decode-optimal ratio — measured in Phase 2 (M3). |
| (b) phase-switching via memory-saver | Reserved for 27B-FP8 / as an optimization. `release_memory_occupation`/`resume` (tags WEIGHTS/KV_CACHE/CUDA_GRAPH, `weight_updater.py:184`) exists in-tree; the #93/#102 pauseable-mempool work lives on `feat/adaptive-draft-len` (NOT in our base). Honest cost: resume re-imports weights, so a prefill burst pays ~19 GB H2D ≈ **1.5–2.5 s** — fine for 120k prompts, poison for short ones. Also requires idle-engine choreography (`assert is_fully_idle`). Not needed for A3B. |
| (c) cross-process weight sharing | **Rejected.** TP=1 and TP=3-uneven marlin shard layouts are different repacked tensors; there is no shareable representation. |

27B: int4 (~15.5 GB) fits pattern (a) with ~18 GB prefill instance; FP8
(~25 GB) requires (b) or is a no-go for coexistence.

## 6. Cost model and honest GO/NO-GO

Transfer cost per 10k tokens: A3B 101 MB (0.79 GB/s-equivalent spread:
rank 0 share D2D ~instant; worst path is the x4 3080's share at 6.5 GB/s →
**~8–25 ms per 10k tokens**, overlappable with the next chunk's compute).
27B: 327 MB per 10k → ~50–100 ms. GDN state: one-shot 32/75 MB.

| scenario | today (TP=3+DCP, this branch, warm) | disagg (measured solo + est. overheads) | net |
|---|---|---|---|
| (i) A3B short prompt 2k | ~0.5–1 s TTFT (at ~4k tok/s) | ~0.1–0.3 s prefill + ~30–80 ms transfer/handshake | **~2x TTFT, GO** — but M3 must verify proxy/handoff overhead doesn't eat it (ruling item 3a; fallback: length-threshold routing) |
| (i') A3B 32k | 8.8 s | 2.0 s + ~0.1 s | **~4x, GO** |
| (iii) A3B 120k deep prompt | **39.4 s** (historical stack: 177 s) | **15.2 s + ~0.2 s + prealloc** ≈ 16 s | **~2.5x TTFT, GO — headline** |
| (ii) 27B-int4 8k, no MTP | ~7 s TTFT (at 1,177 tok/s, older figure) | est. ~3–5 s (solo 27B **not yet measured**) + 0.1 s | ~1.5–2x TTFT, **but** v1 disables MTP (Section 7) → decode tok/s drops materially. **Conditional: NET LOSS for typical chat; only wins for very long prompts. Defer to v2.** |

Decode throughput: unchanged by design (same layout, same kernels); the only
risk is SM contention during prefill bursts (below). **Recommendation: GO,
scoped to A3B (and other all-experts-fit MoE hybrids) in v1; 27B+MTP as v2.**
The descoping fallbacks contemplated in the task (threshold-gated solo
prefill, 5090+x8-3080 prefill pair) are unnecessary on the evidence — solo
wins outright at all measured lengths.

## 7. Cross-cutting concerns

- **Radix continuity:** prefill keeps its own radix cache (5090-local; at
  10 KiB/token a few GB caches hundreds of k tokens). Multiturn: turn-N
  prompt = old prompt (radix hit on prefill) + previous answer (only ever
  seen by decode) → prefill recomputes just the answer tokens at 8–20k tok/s
  — cheap, and identical to upstream PD behavior. Decode-side radix via the
  existing flag, off by default in v1.
- **MTP/draft KV:** the draft pool is deliberately NOT DCP-token-sharded
  (plain uneven head-sharded, full token space,
  `model_runner_kv_cache_mixin.py:1465`), so draft KV transfer would need
  general uneven head reslicing. **v1: spec decode hard-disabled in disagg
  mode** (A3B has no MTP anyway). v2 options: teach the draft pool
  DCP-token-sharding too, or a decode-side draft-prefill pass; the
  hidden-state/topk metadata path already exists.
- **Chunked prefill:** per-chunk send is upstream behavior; transfer
  overlaps compute. Chunk/page boundary vs S alignment handled at the
  filter (Section 4).
- **Scheduling:** standard PD flow — every request goes prefill → decode;
  no dual-role scheduling decisions needed. Router: `sglang_router` wheel or
  a ~100-line local asyncio proxy (test fixtures show the two-call pattern).
- **SM contention on the 5090:** prefill bursts time-share with decode
  rank 0. Expected: decode step inflation during bursts, no correctness
  impact (NCCL timeouts are minutes-scale). Measured in M3; knobs if bad:
  `pause_generation` during bursts, MPS, or chunk-size reduction.
- **fp8 KV scales:** both sides run scale=1.0 (same warning today) —
  byte-compatible.

## 8. Risks (ranked)

1. **CUDA IPC vs torch allocator:** pools must be plain cudaMalloc regions
   (no expandable segments; no `torch.cuda.MemPool`). Mitigation: assert at
   boot; allocate pools with explicit allocator config. (M1 proves this.)
   Approved fallback (ruling 5): if IPC on allocator-owned memory proves
   fragile, use a dedicated `cudaMalloc`'d transfer buffer per rank — one
   extra D2D copy, still trivially cheap vs the seconds saved.
2. **Decode prealloc under DCP** (V1, Section 4) — disagg prealloc path may
   assume full-seq_len rows; must use owned-count. Found early by M2 oracle.
3. **MambaRadixCache on decode** receiving externally-produced state —
   checkpoint-grid bookkeeping (fork's align machinery) must accept a state
   injected at seq_len rather than grown locally. (M3; the `StateType.MAMBA`
   landing already writes `dst_mamba_index` rows, but radix metadata is ours.)
4. **Contention/thermals** on the shared 5090 (M3 measurement; knobs listed).
5. **Staging-ring sizing** on 20 GB cards (watermark backpressure exists;
   size ring at ~256 MB).
6. **Overlap scheduler + disagg + hybrid** interactions — upstream runs this
   combination, but not with our DCP pools; temp-0 oracle gates every step.

## 9. Phase-2 plan (small commits, bugs-first, each step gated by an oracle)

- **M0** — DONE (2026-07-18): A3B TP=3+DCP baseline on this branch measured
  warm — prefill 4,022 / 3,717 / 3,117 tok/s at 8k / 32k / 120k (i.e. the
  historical 177 s / 120k figure is stale; today's baseline is 39.4 s);
  decode bs=1 15.9 tok/s @2k, 15.6 @32k (no-cuda-graph config). Temp-0
  oracle references + 32k-needle (retrieved: 483729) captured to
  `/tmp/pd99/oracle_tp3.json`; scripts in `/tmp/pd99/`.
- **M1** — DONE (2026-07-18): same-layout smoke via `mooncake_tcp` (no new
  transfer backend needed — see Section 3). Qwen3.5-2B (hybrid GDN
  miniature of A3B: 24 layers, 6 full-attn, kv=2) prefill TP=1 + decode
  TP=1 co-resident on the 5090, `local_proxy.py` in front. Oracle: 4/4
  temp-0 outputs bit-identical to non-disagg (3 text prompts + 28k needle);
  per-chunk KV transfer and StateType.MAMBA hybrid state transfer both
  exercised. Lesson: boot the two servers sequentially (concurrent memory
  profiling races); decode needs a distinct `--engine-info-bootstrap-port`.
- **M2** — token-vector rescatter: sender filter (even + weighted), decode
  prealloc owned-count, staging pipeline to the 3080s. Prefill TP=1 →
  decode TP=3+DCP. Oracle: temp-0 == non-disagg TP=3+DCP oracle; needle at
  32k/80k; kv-canary if available.
- **M3** — PARTIAL (2026-07-18): two real fixes landed, two blockers filed.
  * FIXED: the uneven-budget memory profiler double-charged co-resident
    FOREIGN processes (`_profile_available_bytes`: the legacy
    `pre_load*(1-frac)` slack assumes a fresh GPU). Now delta-based
    absolute-budget accounting under `--rank-gpu-memory-mib` —
    co-residency-proof (a static sibling cancels in the delta).
  * FIXED (ops): decode rank0's fixed 384 MiB flashinfer workspace via
    `SGLANG_FLASHINFER_WORKSPACE_SIZE=134217728`. With both fixes the full
    pair reached coexistence on the 5090: 31.99/32.6 GB (prefill 25.6 +
    decode rank0 6.4), decode up, per-rank pools sized correctly.
  * BLOCKER 1 — RESOLVED (task #106, follow-up session): root cause is an
    UPSTREAM bug (PR #19746, decode-side radix cache) affecting
    fake-bootstrap (warmup) requests longer than `chunked_prefill_size`:
    `Req.skip_radix_cache_insert` (set iff `bootstrap_host ==
    FAKE_BOOTSTRAP_HOST`) made `maybe_cache_unfinished_req` skip
    `cache_unfinished_req` entirely — but for a chunked prefill that call
    is ALSO the chunk→prefix conversion (`prefix_indices` advance). With it
    skipped, `PrefillAdder.add_chunked_req` re-plans the SAME first chunk
    forever, each pass allocating fresh KV rows until the pool is exhausted
    (fail-loud OOM crash). The "~pool/4" reading was an artifact of the
    32,768-token warm filling the 131,072-row pool after 64 re-prefills of
    chunk 1 — the multiplier is pool/chunk, not 4. Real-bootstrap requests
    were never affected (flag False; M2 oracle bit-identical). Fix in
    `mem_cache/common.py`: when the flag is set, perform the minimal
    ChunkCache-equivalent `prefix_indices` advance without touching the
    tree; rows stay request-owned and are freed at completion via
    `cache_finished_req(is_insert=False)`. Falsified live on A3B
    (ctx 65,536, pool 219,948): before — `#pending-token` frozen at
    61,440, usage → 1.00 after 108 re-prefill batches, RuntimeError OOM;
    after — pending 61,440 → 0 monotone, usage peaks at 0.29
    (= 63,488/219,948 exactly), 200 OK in 5.13 s, pool fully released
    (second 16k warm starts at usage 0.07). This unblocks the eager-warm
    boot protocol required by Blocker 2.
  * BLOCKER 2: prefill budget overshoot: AWQ/marlin load transients stay in
    the torch cache (charged by the delta at profile time — fix:
    `torch.cuda.empty_cache()` before profiling) and ~1.5-2.5 GB of
    flashinfer/JIT buffers allocate lazily on the FIRST real request
    (crashed a boot-time-coexistent pair — fix: eager warm via the
    fake-bootstrap path, `bootstrap_host=2.2.2.2`, before decode boots).
  * Fixed-cost accounting (5090, torch-visible 31.34 GiB): warmed prefill =
    19.7 weights + ~2.6 JIT/workspaces + pools + ~0.6 ctx; decode rank0
    ~6.4 → pool window ~1.4 GB, which Blocker 1 shrinks 4x. Fix Blocker 1
    first; then the 32k+ static table is straightforward.
  * Perf signal: the co-resident solo prefill sustains **~25,000 tok/s per
    2048-chunk** (fake-path warm, from the batch log).
  * Remaining for follow-up: TTFT/perf table (target: beat 39.4 s/120k by
    >2x; short-prompt TTFT gate per ruling 3a), contention measurement,
    mamba-radix acceptance.
- **M4** — regression: disagg OFF byte-identical on the reference launch
  commands; teardown hygiene per ruling 3b — both servers stopped leaves
  every GPU at 0 MiB, AND a hard-kill (SIGKILL) of the prefill server
  mid-transfer must not orphan IPC handles / pinned buffers or wedge the
  decode server; docs; push `feat/pd-disagg` to efschu/htsglang.

## Appendix: measurement provenance

Solo numbers from `/tmp/pd99/` scripts (launch_solo_a3b.sh, bench_prefill.py,
bench120.py), server boot logs `/tmp/pd99/solo{,2}.log`; budget 26,000 MiB
(85k-token pool) for ≤80k runs, 30,000 MiB (294k-token pool) for 120k runs;
`--disable-cuda-graph` (irrelevant for prefill), chunked_prefill_size 2048,
max_prefill_tokens 16384, flashinfer attention, triton GDN. Thermal gate
respected (≤77 °C peak); all GPUs at 0 MiB before/after.
