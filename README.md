# htsglang: heterogeneous, tier-aware sglang — uneven compute + VRAM pooling, session KV spill to system RAM, adaptive drafter routing, and improved GGUF support

**htsglang** ("split heterogeneous sglang") is a fork of
[sgl-project/sglang](https://github.com/sgl-project/sglang) that makes a
**single tensor-parallel group run well on mismatched GPUs** — cards with
different VRAM sizes, and multiple ranks co-located on one physical GPU. It is
the sglang sibling of the author's vLLM fork
[**shvllm**](https://github.com/efschu/shvllm): the same heterogeneous-TP
feature set, built around sglang's native **RadixAttention** prefix cache.

Beyond the core uneven-TP layout, the fork adds speculative-decoding extensions
(solo draft placement, co-resident cross-algorithm drafting), a weightless-KV
fast lane, MoE expert offloading, suspend-to-disk hibernation, and Qwen3.5/3.6 +
Gemma-4 GGUF loading. This document is the **reference for every fork-specific
command-line flag and environment variable**; for a feature-by-feature
comparison against upstream sglang and vLLM see
[`FEATURES_VS_UPSTREAM.md`](./FEATURES_VS_UPSTREAM.md), and for validated
hardware layouts see [`TOPOLOGIES.md`](./TOPOLOGIES.md).

Fork: [github.com/efschu/htsglang](https://github.com/efschu/htsglang).

> **Scope.** All fork flags below target **pure single-node tensor parallelism**.
> Combination with pipeline (`--pp-size`), data (`--dp-size`), or expert
> (`--ep-size`) parallelism is rejected at startup with an explicit error rather
> than silently attempted. Every flag is validated fail-fast at argument-parse
> time; an invalid mapping aborts the launch instead of hanging a collective at
> runtime. When none of these flags is set the default sglang path is unchanged.

---

## Fork flag reference

### Heterogeneous tensor parallelism and DCP

Explicit rank placement and memory-proportional ("uneven") sharding of a single
TP group across mismatched cards.

**`--rank-gpu-id <int,int,...>`** — Comma-separated CUDA device indices, one per
TP rank (torch.cuda ordering, which is *not* necessarily the nvidia-smi/NVML
order; physical GPUs are resolved via NVML PCI-bus mapping internally).
Duplicating an index co-locates several ranks on one physical GPU. Length must
equal `--tp-size`. Replaces the `--base-gpu-id`/`--gpu-id-step` formula.
*Requires* `--rank-gpu-memory-mib` (or `--rank-tp-ratio auto`, which derives the
budgets). *Conflicts with* a non-default `--base-gpu-id`/`--gpu-id-step`.
Co-locating ranks needs **NCCL ≥ 2.30** and a running **CUDA MPS** server.
Example: `--tp-size 4 --rank-gpu-id 0,0,1,2` (two ranks share GPU 0).

**`--rank-gpu-memory-mib <int | int,int,...>`** — Absolute per-rank memory
budget in MiB. A single scalar applies uniformly to every rank (under even TP
all shards are structurally equal). A per-rank list is permitted only together
with `--rank-tp-ratio` (unequal shards) or the weightless-KV lane. The value is
each rank's **entire** budget — it is converted once to that rank's physical-GPU
fraction (`budget / nvml_total(gpu)`) and applied as `mem_fraction_static` with
**no** additional utilization ceiling or safety margin. Leaving headroom for the
CUDA context is the user's responsibility; the only enforced check is physical
impossibility (`sum of co-located budgets ≤ NVML total`). *Required whenever*
`--rank-gpu-id` is set (except under `--rank-tp-ratio auto`). *Conflicts with*
an explicit `--rank-gpu-memory-mib` value combined with `--mem-fraction-static`
(mutually exclusive; the MiB budget replaces the global fraction in this mode).

**`--rank-tp-ratio <int,int,... | auto | auto-performance>`** — Uneven TP: split
every sharded dimension (attention heads, GDN heads, dense-MLP columns, MoE
expert partitions, KV pool) proportionally to per-rank integer weights instead
of evenly. `sum(weights)` must divide every sharded dimension; identical entries
are rejected (that is the even split — omit the flag). `auto` derives the
weights from `--rank-gpu-memory-mib`, or, when that is omitted, from the NVML
totals of the assigned GPUs minus `--rank-auto-reserve-mib`. `auto-performance`
starts from the same VRAM-auto split and additionally derives a dense-MLP family
vector from a measured hardware micro-probe. *Requires* `--rank-gpu-id`.
Example: `--rank-gpu-id 0,1,2 --rank-tp-ratio 2,1,1`.

**`--rank-auto-reserve-mib <int | int,int,... | auto>`** (default `auto`) —
Headroom in MiB subtracted from each NVML total when `--rank-tp-ratio auto`
derives the budgets itself: a single value for every GPU, one value per rank, or
`auto` (derive from the stock runtime reserve plus the CUDA-graph capture
demand). For co-located ranks the largest reserve on a GPU applies. *Only valid
with* `--rank-tp-ratio auto`.

**`--rank-perf-loose-ctx-percent <float>`** (default `0.0`, range `[0, 100)`) —
Context floor for `auto-performance`: only candidate splits whose predicted max
context stays ≥ `(100 − X)%` of the VRAM-auto prediction are considered. `0`
takes only the free MLP-vector gains that do not reduce predicted context.
*Only valid with* `--rank-tp-ratio auto-performance`.

**`--rank-perf-tune <both | dec | enc>`** (default `both`) — Tuning target of
`auto-performance`: `enc` maximizes prefill throughput, `both` targets prefill +
concurrent throughput, `dec` is a near-no-op (the VRAM-auto split is already at
the decode optimum). *Only valid with* `--rank-tp-ratio auto-performance`.

**`--rank-mlp-ratio <int,int,...>`** — Per-rank integer weights for the
**dense-MLP / shared-expert** weight family only; attention/KV keep following
`--rank-tp-ratio`. Shifting MLP mass off memory-tight ranks grows the
min-synced KV pool. The server logs a suggested restart vector when profiling
detects a rebalancing gain > 10 %. *Requires* an active `--rank-tp-ratio` plan.
`SGLANG_UNEVEN_MLP_VECTOR` overrides this flag.

**`--rank-moe-ratio <int,int,...>`** — As `--rank-mlp-ratio`, but for the
**fused expert-weight (MoE)** family — the main shiftable mass on MoE models.
*Requires* an active `--rank-tp-ratio` plan. `SGLANG_UNEVEN_MOE_VECTOR`
overrides this flag.

**`--rank-vocab-ratio <int,int,... | auto>`** — Ratio-weighted sharding of the
tied vocab layers (`VocabParallelEmbedding` / `ParallelLMHead`, shared with
NEXTN/EAGLE drafts): weight the shard widths by memory bandwidth so the lm_head
read time balances across heterogeneous cards instead of being bounded by the
slowest one. `auto` derives weights from the cached hardware profile.
*Requires* an active `--rank-tp-ratio` plan. Default **off** — without the flag
vocab sharding stays even (unchanged). `SGLANG_UNEVEN_VOCAB_VECTOR` overrides it.

**`--rank-kv-ratio <coupled | capacity | auto | int,int,...>`** (default
`coupled`) — Uneven-DCP KV-token ownership vector, decoupled from the weight
split (moving a token's home rank only moves where its attention math runs).
`coupled` keeps the previous env-gated behavior byte-identical. `capacity`
(alias `auto`) installs the measured capacity-optimal vector after post-load
profiling — **two-boot convergence** that maximizes `max_total_num_tokens` at
the cost of shifting deep-context decode work toward higher-capacity cards. An
explicit integer vector pins ownership (the capacity-vs-depth-speed slider). Any
non-`coupled` value *requires* `--rank-gpu-id` with a **non-uniform**
`--rank-tp-ratio` plan. `SGLANG_UNEVEN_TOKEN_VECTOR` overrides this flag.

### Speculative decoding extensions

**`--speculative-draft-placement <split | solo>`** (default `split`) — `split`
is byte-identical to stock (the draft is TP-sharded across all ranks). `solo`
runs the draft **unsharded on one rank** and broadcasts its `k` draft-token ids
once per round; the other ranks build the draft on the meta device (no draft
weights/KV/graphs) — replacing `k` per-round host-staged all-reduces with one
small broadcast on no-P2P rigs. *Solo scope:* EAGLE/EAGLE3/NEXTN family or
DFLASH; `--speculative-eagle-topk 1`; no rejection sampling; no
`--enable-multi-layer-eagle`; not `FROZEN_KV_MTP`; `--tp-size ≥ 2`; pure
single-node TP; no PD disaggregation. DFLASH+solo additionally rejects
`--speculative-adaptive`.

**`--speculative-draft-gpu <int>`** — CUDA device index (same space as
`--rank-gpu-id`) whose TP rank hosts the solo draft; unset ⇒ rank 0. *Only valid
with* `--speculative-draft-placement solo`.

**Adaptive drafter routing** — switching the speculative draft model at runtime by context length (`policy` mode) or measured acceptance (`auto` / bandit mode) — is configured by the cross-algorithm flags below.

**`--speculative-cross-algorithm`** (flag) — Co-resident cross-algorithm
speculative decoding: load **both** a NEXTN/MTP draft and a DFLASH draft and
capture both CUDA-graph sets at boot. The NEXTN rung runs TP-sharded, the DFLASH
rung runs solo on rank 0. *Requires* `--speculative-algorithm NEXTN` plus
`--speculative-draft-model` pointing at the DFLASH checkpoint, and
`--speculative-cross-algorithm-force`. Default off. *(Experimental — the runtime
per-batch switch and bandit controller are implemented; the ctx-gate-from-config
sub-component is still in progress, see `FEATURES_VS_UPSTREAM.md` row 5.)*

**`--speculative-cross-algorithm-force <nextn | dflash | schedule:N | auto | policy>`**
— Active-rung policy under `--speculative-cross-algorithm` (**required** when it
is set). `nextn`/`dflash` statically pin one rung; `schedule:N` switches every N
verify rounds (debug); `auto` is an acceptance-driven bandit over the rungs
(rank 0 decides, rung id broadcast; tunables via `SGLANG_CROSS_BANDIT_*`);
`policy` is a deterministic context-threshold lookup from
`--speculative-drafter-policy`.

**`--speculative-cross-algorithm-ctx-gate <auto | off | int>`** (default `auto`)
— Context threshold above which the short-context DFLASH rung is ineligible
(never selected or probed). `auto` derives it from the DFLASH drafter's
`config.json` (factor × sliding-window, `SGLANG_CROSS_CTX_GATE_FACTOR`); an
integer sets it in tokens; `off` disables the gate. Only meaningful under
`--speculative-cross-algorithm-force auto|policy` (under `policy` it acts as a
safety filter on the policy table).

**`--speculative-drafter-policy <"start_ctx:family:value,...">`** — Ordered
deterministic drafter-policy table for `--speculative-cross-algorithm-force
policy`, e.g. `0:dflash:16,4096:nextn:3` (`family` = `nextn`|`dflash`; nextn
value = k, dflash value = block size). Stages must reference configured arms,
start contexts strictly ascending, first stage at 0. `auto`/unset derives two
stages from the drafter training context. *Only used with* force `policy`.

**`--speculative-adaptive-graph-memory <auto | resident | offload | offload-scratch>`**
(default `auto`) — VRAM policy for the pre-built adaptive-speculative runtime
states. `resident` keeps all candidate states materialized (reserve = sum);
`offload` physically unmaps inactive states' scratch/capture pools via
`torch_memory_saver` and remaps on swap (reserve = max, KV capacity recovered) —
*requires* CUDA + flashinfer + the `full` decode cuda-graph backend;
`offload-scratch` offloads scratch buffers only. `auto` degrades
offload → offload-scratch → resident by prerequisite.

> **Modified upstream flag.** `--speculative-adaptive-config` (upstream) gains a
> built-in profile name: `high-accept` adds k=4/5 ladder rungs with upward
> hysteresis for workloads with per-position accept probability ≳ 0.85.

### Memory / VRAM management

**`--swa-pool-sizing <ratio | cap>`** (default `ratio`) — How the sliding-window
(SWA) KV pool of hybrid-SWA models (Gemma-style) is sized. `ratio` grows the SWA
pool with the context budget; `cap` pins it at its window-bounded worst case and
routes the entire remaining KV budget to the full-attention pool, unlocking
long-context capacity on models whose SWA layers dominate KV bytes (e.g. Gemma-4
31B). `cap` *requires* `--max-running-requests` and chunked prefill.

**`--mamba-checkpoint-interval <int>`** — Pin all radix-cached mamba/GDN
checkpoints to absolute multiples of this token count so the checkpoint grid is
a pure function of token history (removing traffic-dependent greedy/temp-0 drift
under mixed traffic). Must be a multiple of the page size and the model's
mamba/FLA chunk size; overrides `--mamba-track-interval`. Off by default;
recommended 2048 or the chunked-prefill size. Companion tuning envs:
`SGLANG_MAMBA_CKPT_WINDOW`, `SGLANG_MAMBA_CKPT_STRICT_RESUME`.

### Fast-lane scheduling

An opt-in latency-priority scheduling class layered on the existing
priority-scheduling subsystem; the default path is unchanged.

**`--enable-fast-lane`** (flag) — Enable the fast/heavy two-tier class:
`lane='fast'` requests are seeded with a high priority so they preempt batched
`lane='heavy'` (default) requests. Implies priority scheduling in two-tier mode.

**`--fast-lane-priority <int>`** (default `1000000`) — Priority seeded for
`lane='fast'` requests. *Only used with* `--enable-fast-lane`.

**`--fast-lane-reserved-heavy-slots <int>`** (default `1`) — Anti-starvation
floor: minimum running `lane='heavy'` requests fast-lane preemption never drops
below. *Only used with* `--enable-fast-lane`.

**`--fast-lane-heavy-aging-ms <float>`** (default `10000`) — A heavy request that
has waited longer than this is promoted ahead of the fast tier for one
admission (`0` disables aging). *Only used with* `--enable-fast-lane`.

**Request-side usage.** The lane is enabled server-side with `--enable-fast-lane`
but selected **per request** via the top-level `"lane"` field on the native
`/generate` request (`GenerateReqInput.lane`; the scheduler tags the request
when `recv_req.lane == "fast"`). Set `"lane": "fast"` for an interactive request;
omitting it (or `"heavy"`) keeps the default batchable class.

```bash
curl http://HOST:PORT/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "...", "sampling_params": {"max_new_tokens": 128}, "lane": "fast"}'
```

Fast-lane requests are admitted with the configured `--fast-lane-priority`
(default `1000000`) in the priority-scheduling path;
`--fast-lane-reserved-heavy-slots` keeps slots for regular requests and
`--fast-lane-heavy-aging-ms` prevents starvation of aged heavy requests.

The `"lane"` field is honored on both the native `/generate` endpoint (top-level
field, as above) and the **OpenAI-compatible endpoints** — Chat Completions and
Completions — where clients pass it in `extra_body`, exactly like `priority`. It
is forwarded to `GenerateReqInput.lane`; an invalid value is rejected with HTTP
`400` (`lane must be "fast" or omitted`).

```python
from openai import OpenAI
client = OpenAI(base_url="http://HOST:PORT/v1", api_key="none")
client.chat.completions.create(
    model="...", messages=[{"role": "user", "content": "..."}],
    extra_body={"lane": "fast"},
)
```

### Offloading and the weightless-KV lane

MoE expert offloading is env-only (see the environment-variable table). The
weightless-KV fast lane is *experimental*.

**`--weightless-kv-fastlane`** (flag, **experimental**) — One head rank holds the
full model weights and runs Q/O-proj + FFN + GDN as collective-free TP=1; the
other DCP ranks are **weightless**, holding only a token-shard of the KV cache
and computing attention over it — so a large KV capacity lives on the smaller
cards while single-stream speed stays near the head card's TP=1 rate. *Requires*
`--tp-size ≥ 2`, `--dcp-size == --tp-size`, and a flashinfer/fa3 attention
backend. Chunked prefill is supported; speculative decoding,
`--enable-mixed-chunk`, and `--speculative-eagle-topk > 1` are hard-rejected.

**`--weightless-kv-head-rank <int>`** (default `0`) — The rank that holds all
weights/heads; every other rank is weightless. Must be `0 ≤ rank < tp_size`.
*Requires* `--weightless-kv-fastlane`.

**`--weightless-kv-chunked-block-size <int>`** (default `0` = off,
**experimental**) — `> 0` restructures the per-rank decode attention into a
block loop (bounded device staging region, split-KV partial decodes online-LSE
merged), CUDA-graph-capable as a bucketed ladder. *Requires*
`--weightless-kv-fastlane`.

**`--weightless-kv-host-spill-tokens <int>`** (default `0` = off,
**experimental**) — Attach a pinned-host overflow tier of this many token slots
to each rank's full-attention KV pool, streamed H2D through the block-decode
loop so a single sequence's KV can exceed a rank's VRAM. *Requires*
`--weightless-kv-fastlane`, `--weightless-kv-chunked-block-size > 0`,
`page_size == 1`, the even-modulo DCP owner rule (no `--rank-tp-ratio`); rejects
hierarchical/unified cache and PD disaggregation. Streaming is PCIe-bound
(correctness-first). Tuning envs: `SGLANG_WL_GRAPH_MAX_BS`,
`SGLANG_WL_H2D_PREFETCH`.

**`--weightless-kv-spill-device-cap <int>`** (default `0`, **debug/test**) — Cap
the allocatable device-resident KV slots per rank to force the host-streaming
path at small contexts (the all-resident-vs-streamed byte-parity gate). A server
arg (not an env) so every rank sees an identical slot→tier map.

#### Per-session KV offload (S1, experimental)

A distinct host-tier mechanism from the weightless-KV lane. When the decode
batch runs out of KV memory (after tree eviction), the **youngest** running
session is spilled — its full-attention KV shard is backed up per rank into a
pinned host pool and it keeps decoding from host via a separate eager `bs=1`
tick interleaved between device-batch iterations, while its GDN/Mamba state
stays resident. The oldest session is never evicted; a spilled session is
restored FIFO with hysteresis once device KV frees up. The spilled session's
decode is PCIe-bound (each step streams its whole context H2D; token-sharded
uneven DCP splits that volume across ranks). Default off ⇒ every other path is
byte-identical.

**`--enable-kv-session-offload`** (flag, **experimental**) — Enable per-session
KV offload to host RAM with FCFS eviction. *S1 scope, enforced fail-fast:*
flashinfer attention backend, `page_size == 1`, exactly **one** spilled session
at a time (further pressure falls back to stock retraction), single-node pure
TP/DCP; rejects speculative decoding, PD disaggregation, `--enable-mixed-chunk`,
`--weightless-kv-fastlane`, `--enable-hierarchical-cache` / `--enable-unified-memory`,
and `--enable-hisparse`.

**`--kv-session-offload-block-size <int>`** (default `8192`) — Streamed-block
size in per-rank token slots: each spill-tick attention layer pulls the
host-resident KV shard H2D through a staging buffer of this many slots per block,
runs a partial flashinfer decode, and online-LSE-merges the partials (the same
block loop as the weightless-KV lane). Also sizes the staging buffer. *Requires*
`--enable-kv-session-offload`.

**`--kv-session-offload-tick-interval <int>`** (default `1`, min `1`) — Minimum
scheduler iterations between two spill ticks while device sessions run (`1` =
alternate device/spill tick); larger values keep more device throughput and slow
the spilled session. When no device batch runs, the spilled session ticks every
iteration. *Requires* `--enable-kv-session-offload`.

**`--kv-session-offload-restore-margin-tokens <int>`** (default `4096`, min `0`)
— Restore the spilled session only once the allocator has `(session tokens +
this margin)` free slots (anti-flutter headroom so a restored session does not
immediately re-trigger a spill). *Requires* `--enable-kv-session-offload`.

**`--kv-session-offload-restore-hysteresis-steps <int>`** (default `4`, min `1`)
— The restore memory condition must hold for this many consecutive scheduler
iterations before the session is copied back to device. *Requires*
`--enable-kv-session-offload`.

### Persistence / hibernate

**`--enable-weights-disk-backup`** (flag) — Suspend-to-disk hibernation
(GGUF-scoped). The `/hibernate` endpoint parks each rank's final
post-`process_weights_after_loading` GPU tensors to `--hibernate-dir`; a
subsequent matching boot restores them via a raw byte copy, skipping the GGUF
parse + weight-map + flat-assembly (~44 s → seconds on Qwen3.6-27B Q3_K_M).
*Requires* `--hibernate-dir`.

**`--hibernate-dir <path>`** — Directory for the per-rank hibernate shards and
manifest. On boot, a manifest here matching the launch args (model, quant,
tp/dcp/ratio, rank-gpu-id, per-rank NVML UUID) triggers the fast restore; any
mismatch is a hard error (no silent cold-load fallback). *Required whenever*
`--enable-weights-disk-backup` is set.

### Diagnostics

**`--determinism-logits-dump-dir <path>`** (**debug**) — Exports each TP rank's raw next-token logits row (single-sequence batches only, in the exact served dtype, captured before logit post-processing and sampling) to sequentially numbered, atomically written `torch.save` files tagged by rank and decode step. This is the capture surface for the determinism harness (`tests/determinism`), which compares these rows across repeated runs and across execution configurations to confirm that the fork's heterogeneous execution paths — uneven TP/DCP token/weight sharding, mixed GPU architectures (differing floating-point reduction order), speculative-decode verify, and CUDA-graph replay — leave the emitted token identical, and to localize any divergence to a specific rank and step. Default off; the default serving path is untouched.

---

## Environment variables

Fork-specific environment variables (upstream sglang envs are unchanged and not
listed). Operational variables tune shipped features; the advanced/debug group
holds diagnostic levers used during the determinism and offload bring-up.

### Operational

| Variable | Default | Purpose |
|---|---|---|
| `SGLANG_UNEVEN_MLP_VECTOR` | unset | Per-rank dense-MLP weight vector; overrides `--rank-mlp-ratio`. Emitted as a restart hint by the KV-pool self-calibration. |
| `SGLANG_UNEVEN_MOE_VECTOR` | unset | Per-rank fused-expert (MoE) weight vector; overrides `--rank-moe-ratio`. |
| `SGLANG_UNEVEN_VOCAB_VECTOR` | unset | Per-rank vocab-shard vector; overrides `--rank-vocab-ratio`. Never falls back to the base plan — unset ⇒ even vocab. |
| `SGLANG_UNEVEN_TOKEN_VECTOR` | unset | Per-rank DCP KV-token ownership vector; overrides `--rank-kv-ratio`. |
| `SGLANG_UNEVEN_DCP` / `SGLANG_UNEVEN_DCP_WEIGHTED` | `0` | Legacy env gate for the weighted-DCP token path; superseded by `--rank-kv-ratio` (kept for the `coupled` byte-identical path). |
| `SGLANG_PERF_REPROBE` | `0` | Force a fresh `auto-performance` hardware micro-probe, ignoring the `~/.cache/sglang` profile. |
| `SGLANG_MEASURED_KV_BUDGET` | `0` | Two-boot measured KV-budget correction: persist each rank's measured leftover VRAM and add it to the next boot's KV budget. |
| `SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB` | `400` | Safety MiB subtracted from the measured leftover (scalar or per-rank list). |
| `SGLANG_MAMBA_CKPT_WINDOW` | `2` | With `--mamba-checkpoint-interval`: how many deepest on-grid checkpoints per radix path to keep live. |
| `SGLANG_MAMBA_CKPT_STRICT_RESUME` | `0` | Resume only at the deepest interval boundary of the full-KV match (else recompute from 0). |
| `SGLANG_MOE_RESIDENT_EXPERT_FRACTION` | `1.0` | Fraction of each layer's routed experts kept GPU-resident; `< 1.0` activates the pinned-host expert pool + LRU H2D fetch (offload). |
| `SGLANG_MOE_HOT_RESIDENCY` | `0` | Freeze the resident set to the R most-frequently-routed experts per layer (calibrated then frozen; byte-identical to static residency). |
| `SGLANG_MOE_HOT_CALIB_STEPS` | `1` | Offload forwards observed before the hot-set is computed and frozen. |
| `SGLANG_MOE_OFFLOAD_CUDA_GRAPH` | `0` | CUDA-graph-capturable decode offload (on-device index math + captured gather); needs a residency layout frozen before capture. |
| `SGLANG_MOE_HOTSET_FILE` | unset | Path to an offline per-layer frozen hot-set file (enables hot-residency under CUDA-graph capture). |
| `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` | `0` | Max decode batch size eligible for the captured offload path (`0` = no cap). |
| `SGLANG_MOE_OFFLOAD_TRACE` | unset | Path to log per-layer routed expert ids for offline locality/cache-hit simulation. |
| `SGLANG_WL_GRAPH_MAX_BS` | `1` | Weightless-KV streaming: max decode capture bucket (host-spill graph path supports bs=1; larger batches fall back to the eager block loop). |
| `SGLANG_WL_H2D_PREFETCH` | `1` | Weightless-KV streaming: double-buffered H2D prefetch on a side stream (PCIe hidden behind compute); `0` restores single-buffer serial copy. |
| `SGLANG_ADAPTIVE_SERVING_MARGIN_MIB` | `512` | Adaptive graph-memory offload: min free VRAM that must remain after mapping the largest candidate state (boot-enforced). |
| `SGLANG_ADAPTIVE_CAPTURE_CPU_BACKUP` | `0` | Back per-state capture pools to host RAM on pause and restore exact bytes on resume (fallback for the offload swap path). |
| `SGLANG_CROSS_CTX_GATE_FACTOR` | `4.0` | Cross-algorithm `auto` ctx-gate: factor × the DFLASH drafter's sliding window when deriving the gate threshold. |
| `SGLANG_CROSS_CTX_GATE_NEAR_FRAC` | `0.8` | Nearness fraction for pre-empting requests whose remaining budget can cross the ctx gate. |
| `SGLANG_CROSS_POLICY_CTX_FACTOR` | — | `policy`-mode derived-stage factor (× drafter sliding window) for the auto policy table. |
| `SGLANG_CROSS_BANDIT_*` | — | Cross-algorithm `auto` bandit tunables (e.g. `SGLANG_CROSS_BANDIT_MIN_DWELL_ROUNDS`) guarding against rung flapping (dwell/deadzone/burn-in/probing). |
| `SGLANG_MODEL_ROOTS` | `~/.cache/huggingface/hub:./models` | Planner/rig-dashboard: colon-separated model-discovery roots. |
| `SGLANG_PLANNER_PROFILES` | store default | Planner: path to the profiles store. |
| `SGLANG_PLANNER_VALIDATION_MODEL` | unset | Planner: validation model override. |
| `SGLANG_PLANNER_GRAPH_ANCHORS` | unset | Planner: graph-memory anchor configuration. |
| `SGLANG_PLANNER_PYTHONPATH` | unset | Planner: PYTHONPATH override for launched server subprocesses. |

Two determinism fixes ship **on by default** and change behavior (they make
output a fixed function of request history rather than request ordinal); set to
`0` only to restore the old nondeterministic-across-requests behavior:

| Variable | Default | Purpose |
|---|---|---|
| `SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST` | `1` | Zero every flashinfer float workspace when a request finishes (the fa2 split-KV kernels otherwise read a previous request's partials). |
| `SGLANG_FLUSH_ZERO_KV` | `1` | Zero the attention KV data buffers on `/flush_cache` so a flushed server matches a fresh boot bit-for-bit. |

### Advanced / debug

Diagnostic levers from the determinism (#50) and offload bring-up campaigns;
not needed for normal operation.

| Variable | Default | Purpose |
|---|---|---|
| `SGLANG_FLUSH_SCRUB_FREE_MEMORY` | `0` | After `/flush_cache`, claim + zero + release free VRAM so recycled pages read as fresh-boot zeros. |
| `SGLANG_POISON_POOL_DATA` | `0` | Fill pool data buffers with NaN at boot so any kernel reading un-written pool bytes surfaces immediately. |
| `SGLANG_POISON_GRAPH_PAD` | `0` | Poison padded/tail regions of persistent input and graph-replay buffers to localize stale-tail reads. |
| `SGLANG_SPEC_STATE_HASH` | `0` | Hash all persistent worker tensors/attributes after each request to pinpoint cross-request state mutation. |
| `SGLANG_SPEC_STATE_HASH_MAX_MB` | `0` | Tensors above this size are fingerprinted from a strided sample (`0` = hash fully). |
| `SGLANG_SPEC_RESET_PROBE` | `""` | Comma-separated families (`flashinfer`, `registry`) of persistent state to hard-reset after each request (bisection probe). |
| `SGLANG_SPEC_RESET_PROBE_FILTER` | `""` | fnmatch globs on flashinfer wrapper attribute names for the reset probe. |
| `SGLANG_MAMBA_CKPT_DEBUG` | `0` | Per-request mamba-checkpoint match/resume/slot diagnostics. |
| `SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC` | `0` | All-gather + assert equal (swap ordinal, target steps) on every adaptive state swap (turns a rank-divergent swap into an immediate failure). |
| `SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL` | `0` | Force an adaptive runtime-state swap every N verify completions (stress the offload swap path). |
| `SGLANG_CROSS_ALGO_TRACE` / `SGLANG_CROSS_BANDIT_TRACE` | unset | Trace files for the cross-algorithm rung selection and bandit decisions. |
| `SGLANG_CROSS_WARMKEEP_TIMING` | unset | Timing trace for the cross-algorithm warm-keep path. |

---

## Qwen3.5/3.6 and Gemma-4 GGUF loading

The GGUF format and its K-quant dequant / MMQ / MMVQ compute kernels come from **ggml / llama.cpp**, integrated into the engine via vLLM / upstream sglang (the `sgl-kernel` GGUF kernels are ported ggml ops — `mmq.cuh` is copied from llama.cpp's `ggml-cuda/mmq.cu` — and the quant layer is adapted from vLLM). The fork's contribution is running GGUF under uneven TP/DCP (256-superblock alignment, MLP-unit coarsening, the MMQ out-of-bounds fix under expert sharding), the bespoke Qwen3.5/3.6 + Gemma-4 architecture adapters, and dispatch tuning (per-device MMVQ↔MMQ crossover, K-split MMVQ).

The fork adds dedicated GGUF adapters for the **Qwen3.5 / 3.6 hybrid-GDN models**
and **Gemma-4** — architectures upstream sglang's generic GGUF path cannot load.
On conversion `llama.cpp` rewrites several tensors (norm offset,
`A_log = log(-ssm_a)`, conv1d unsqueeze, GDN value-head retiling, block-aligned
`out_proj` permute); the adapters invert every transform. All K-quants
(Q4_K_M … Q8_0) load coherent and greedy-deterministic. **NEXTN / MTP** is loaded
straight from the same `.gguf` (`blk.<num_layers>` / `nextn.*`) as the draft
model — no separate checkpoint — and composes with uneven TP (the 256-element
K-quant superblock boundary is respected on every per-rank split). These use the
upstream `--load-format gguf` / `--quantization gguf` flags; the fork delta is
the adapter, so no new flag is introduced. The MMVQ↔MMQ decode-kernel crossover
is picked per physical GPU (`SGLANG_GGUF_MMVQ_SAFE` to override).

```bash
# GGUF Q6_K, uneven TP=3, MTP from the same file
python -m sglang.launch_server \
  --model-path     /models/Qwen3.6-27B-...-Q6_K.gguf \
  --tokenizer-path /models/Qwen3.6-27B-...-GGUF \
  --load-format gguf --quantization gguf \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto --rank-auto-reserve-mib 2048 \
  --kv-cache-dtype fp8_e4m3 --disable-custom-all-reduce \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /models/Qwen3.6-27B-...-Q6_K.gguf \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --context-length 8192
```

**Validated:** Qwen3.6-27B FP8, **TP=3 on 1× RTX 5090 + 2× RTX 3080** — clean
boot, ~297k `max_total_num_tokens` @ 32k context, coherent output, greedy decode
bit-identical cold vs. warm. Hybrid GDN + full-attention models and NEXTN / MTP
speculative decoding both work under the uneven layout.

> **Modified upstream flags.** `--load-format gguf` / `--quantization gguf` are
> extended to the Qwen3.5/3.6 and Gemma-4 arches; `--enable-hierarchical-cache`
> (HiCache) is made correct under the uneven-TP/DCP layouts (global→owned-compact
> KV index translation); the CUDA `--dcp-size > 1` + speculative-decoding guard
> is relaxed for the uneven-hybrid weighted-DCP path.

## Docker

> **Image status.** The previously published GHCR runtime images were built from
> **an earlier snapshot of this fork** and do **not** contain the full set of
> flags documented in this README — in particular the later cross-algorithm
> speculative-decoding, drafter-policy, and VRAM/measured-budget changes on
> `integration/r2` are absent. Their concrete tags have been removed below so
> they are not copied as the current path. **Updated images will follow;** until
> then the supported way to run the current fork is a **source build**.

**Source build (supported path).** Follow the standard sglang from-source
install (see `docs/get_started/install.md`, "Method 2: From source"), but clone
this fork and check out the feature branch before installing into your venv:

```bash
git clone https://github.com/efschu/htsglang.git
cd htsglang && git checkout integration/r2
pip install -e "python"
```

**Runtime container (reference; rebuild required).** The image is CUDA 13.0,
built for **sm75–sm120** (Turing … Blackwell / RTX 5090), with **NCCL 2.30.7**
baked in (required for multi-rank-per-GPU), a HiCache file backend, and an
ENV-driven entrypoint. Two overlays exist: a base heterogeneous-TP image
(FP8 / safetensors) and a thin overlay adding the Qwen3.5/3.6 GGUF adapter +
ffmpeg (use the latter for GGUF models). Build them from the Dockerfiles in
[`docker/`](./docker/) (`htsglang.Dockerfile`, `htsglang-qwen35-gguf.Dockerfile`)
and publish under your own registry path; the tags below are placeholders.

Pull and run the GGUF image (GGUF + uneven TP=3 + MTP):

```bash
docker pull ghcr.io/<owner>/<gguf-image>:<tag>

GD=/models-cache/Qwen3.6-27B-...-GGUF
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --security-opt apparmor=unconfined \
  -p 8014:30000 \
  -v /models-cache:/root/.cache/huggingface \
  -e MODEL_PATH=$GD/Qwen3.6-27B-...-Q6_K.gguf \
  -e TOKENIZER_PATH=$GD \
  -e LOAD_FORMAT=gguf -e QUANTIZATION=gguf \
  -e SPECULATIVE_DRAFT_MODEL_PATH=$GD/Qwen3.6-27B-...-Q6_K.gguf \
  -e DISABLE_CUSTOM_ALL_REDUCE=1 \
  -e TP_SIZE=3 -e RANK_GPU_ID=0,1,2 -e RANK_TP_RATIO=auto \
  ghcr.io/<owner>/<gguf-image>:<tag>
```

Empty ENV ⇒ the flag is omitted, so the same image also serves the plain
FP8/safetensors path (drop the four `*gguf*` / draft vars). For multi-rank-per-GPU
co-location set an absolute budget, which auto-disables the ratio flags
(`RANK_GPU_ID=0,0,1,2`, `RANK_GPU_MEMORY_MIB=13500`). The container needs
`apparmor=unconfined` (LXC/Proxmox host), `ipc: host` + `shm_size`
(NCCL / shared-memory IPC), and the `nvidia` device reservation.

*Upstream sglang README below.*


--------------------------------------------------------------------------------

<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## News
- [2026/06] 🔥 The next generation of speculative decoding: DFlash and Spec V2 ([blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).
- [2026/04] 🔥 DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles ([blog](https://lmsys.org/blog/2026-04-25-deepseek-v4/)).
- [2026/06] SGLang provides day-0 support for latest open models ([Nemotron 3 Ultra](https://lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/), [Nemotron 3 Super](https://lmsys.org/blog/2026-03-11-run-nvidia-nemotron-3-super/), [Higgs Audio v3 TTS](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)).
- [2026/02] 🔥 Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72 ([blog](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)).
- [2026/01] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).

<details>
<summary>More</summary>

- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark/5090), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend used for training many frontier models, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/), [GB300 long context](https://lmsys.org/blog/2026-02-19-gb300-longctx/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, Modal, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at [sglang@lmsys.org](mailto:sglang@lmsys.org).

Long-term active SGLang contributors are eligible for coding agent sponsorship, such as Cursor, Claude Code, or OpenAI Codex. Email [sglang@lmsys.org](mailto:sglang@lmsys.org) with your most important commits or pull requests.

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
