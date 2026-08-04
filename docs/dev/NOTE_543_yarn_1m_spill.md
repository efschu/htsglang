# NOTE 543 — YaRN long-context lane and KV-session spill layout

Status: **desk analysis complete and config route proven off-GPU. Rig validation
pending a serving-restart window.**
Branch `feat/yarn-1m-543`, worktree `/spinning/wt-543-yarn-1m`.

What was asked for:

* Variant A — two sessions, one may grow to 1 000 000 tokens, the other capped
  at 262 144, the second one's KV spilling to host RAM while the big one grows.
* Variant B — four pool sessions at 262 144 **plus** one 1 000 000 session, all
  guaranteed schedulable via spill.

Verdict up front: **neither variant is reachable at 1 000 000 tokens on this
rig, and the reason is not the spill layout — it is that a request can never
exceed the VRAM KV pool.** Section 3 shows the pool tops out near 925 000 tokens
in the most aggressive configuration, and section 5 shows the admission gate
that makes that a hard cap rather than a soft one. Section 4 shows the second,
independent wall: the pinned host spill pool. Section 9 gives the two shapes
that do fit.

---

## 1. Model geometry (measured, not assumed)

`Qwen3.6-27B-INT8-W8A8`, `model_type qwen3_5_text`, 64 layers,
`full_attention_interval 4` — **16 full-attention layers carry RoPE and paged
KV**, the other **48 are Gated DeltaNet** and carry a constant-size recurrent
state per session. `num_key_value_heads 4`, `head_dim 256`,
`partial_rotary_factor 0.25` (so `rotary_dim` 64), `max_position_embeddings
262144`.

Live boot (PID 1236, `--kv-cache-dtype fp8_e4m3`, TP=3 uneven,
`--rank-tp-ratio auto-performance` → `[19607, 16280, 16280]`, `dcp_size 1`):

| | rank 0 (5090) | rank 1 (3080) | rank 2 (3080) | node |
|---|---|---|---|---|
| KV heads | 2 | 1 | 1 | 4 |
| main KV bytes/token | 16 384 | 8 192 | 8 192 | **32 768** |
| NEXTN draft KV bytes/token | 1 024 | 512 | 512 | 2 048 |
| total with MTP | 17 408 | 8 704 | 8 704 | **34 816** |

Cell size is `kv_heads_rank × (head_dim + v_head_dim) × 16 layers × 1 B`
(`model_executor/pool_configurator.py:429-435`, layer count from `:248-258`
counting only `full_attention_layer_ids`). Cross-checked against the boot log:
`KV Cache is allocated … #tokens: 333254, K size: 2.54 GB, V size: 2.54 GB` on
TP0 and `1.27 GB` on TP1/TP2 — exactly 16 384 / 8 192 bytes per token.

KV heads are **split 2/1/1, never replicated**: plain even TP=3 is impossible
for this model (`models/qwen3_5.py:881-889` asserts `4 % 3 == 0` or `3 % 4 == 0`,
both false), `--rank-tp-ratio` is required, and `attn_kv_replicated(3, 4)` is
false because replication needs `kv < tp`. The 4 heads are indivisible units
distributed by largest remainder. `max_total_num_tokens` is derived per rank and
then MIN-reduced, so the 2-head rank binds unless its budget is twice the others'.

## 2. GDN recurrent state — the second, separate axis

`Mamba Cache is allocated. max_mamba_cache_size: 37`, confirmed live
(`mamba_available_tokens 3` + `mamba_evictable_tokens 30` + `mamba_used_tokens 4`
= 37, `mamba_usage` = 4/37).

| | rank 0 | rank 1 | rank 2 | node |
|---|---|---|---|---|
| conv + ssm per slot | 56.5 MiB | 47.0 MiB | 47.0 MiB | **150.6 MiB / session** |
| pool, 37 slots | 2.04 GiB | 1.70 GiB | 1.70 GiB | 5.44 GiB |
| speculative intermediate | 1.05 GiB | 0.88 GiB | 0.88 GiB | 2.81 GiB |
| **GDN total** | 3.10 GiB | 2.59 GiB | 2.59 GiB | **8.25 GiB** |

Formula `configs/mamba_utils.py:177-187`, shapes from `configs/qwen3_next.py:352`:
48 GDN layers × (conv `10240/tp × 3` at bf16 + ssm `48/tp × 128 × 128` at fp32).
The unsharded total is 146.81 MiB per session, split by whole units of the
16-unit GDN grid (`models/qwen3_5.py:198-206`), here 6/5/5.
`SGLANG_MAMBA_SSM_DTYPE=bfloat16` halves the ssm term; fp8 is refused
(`kv_session_offload.py:1351` rejects an itemsize below 2).
**Independent of `context_length`.**

Concurrency: `mamba_ratio = 3 + 2` (overlap plus the `extra_buffer` radix
strategy) `= 5`, `max_num_reqs = min(--max-running-requests, …, 37 // 5 = 7)`.
The existing 37-slot pool **already admits 7 sessions**, so for five sessions only
`--max-running-requests` has to be raised. Trimming to 25 slots frees 1.77 GiB
into the KV pool and still admits 5.

### Can the GDN state be offloaded as well?

The requirement that idle sessions' GDN states also vacate VRAM was checked
against the tree. Three mechanisms exist; only one moves bytes.

| Mechanism | Verdict |
|---|---|
| kvso session spill | **KV only.** `bundle_spillable_sizes` returns `[("kv", kv)]` and nothing else (`managers/kv_session_offload.py:111-125`); a spilled session keeps its GDN slot until it finishes (`:5572 free_mamba_cache`). The bundle seam has **no production caller** — `"kv"` is hardcoded downstream. NEEDS-CODE. |
| `offload_gdn_states.py` behind `SGLANG_OFFLOAD_REGISTER=1` | Registers and plans only. Module docstring `:70`: "Nothing in this module touches torch.cuda." INERT. |
| **`--gdn-resident-state-slots N`** | **Works today, no env gate.** `managers/scheduler.py:3368` → `managers/gdn_slot_runtime.py:231` → `mem_cache/gdn_slot_executor.py:352` → `mem_cache/memory_pool.py:918 export_state_blob` → `t[:, idx].to("cpu", copy=True)` after a device synchronize. A real D2H. |

Two constraints on the working one:

* `va_stable_required=True` (`offload_gdn_states.py:360`): a state set is a stride
  slice of the pool tensors, not a page range. Vacating frees the **slot** for
  reuse; it never returns the pool's VRAM to the KV pool
  (`model_runner_kv_cache_mixin.py:2017-2020` states this explicitly). The lever
  raises sessions-per-slot-budget; it does not enlarge the KV pool at runtime.
* **Do not combine it with kvso in this boot.** `GdnSlotRuntime.on_round` treats
  only `running_batch` members as active (`gdn_slot_runtime.py:174`), while
  `idle_holders()` (`:247-256`) explicitly offers **kvso-spilled sessions** as
  vacate candidates. A spilled session that is still host-ticking would get
  `req.mamba_pool_idx = None` set under it while its tick must still forward 48
  GDN layers. No guard excludes it. This is a correctness bug, not a tuning
  question, and it is why the boot below leaves `--gdn-resident-state-slots` unset.

The blob store behind the working path is `LocalGdnBlobStore`
(`gdn_slot_executor.py:121-146`) — a plain unpinned, unbounded, unmetered host
dict. `TieredGdnBlobStore` exists but nothing constructs it.

For five sessions the GDN pool is **not** binding (37 slots admit 7, at
150.6 MiB each). It becomes binding above roughly 7 concurrent sessions, and
that is the point at which the kvso×GDN work above has to be done.

## 3. VRAM ceiling

Free today (`nvidia-smi`): rank 0 (5090) 3 461 MiB, ranks 1/2 (3080) 2 955 MiB
each. The 5090 additionally carries the translator tenant (5 916 MiB, PID 30439).
Corridor rule: ≥ 400 MiB free on every card at all times.

New context-dependent fixed costs at a long context, per rank:

* rope cos/sin cache `context × 64 × 4 B` kept fp32 on CUDA
  (`rotary_embedding/base.py:102-104`) — 96 MiB at 393 216, 256 MiB at 1 048 576,
  against 64 MiB today;
* CUDA-graph KV index and mask buffers `max_bs × context × 5 B`
  (`layers/attention/flashinfer_backend.py:1703-1763`) — these scale with
  `--cuda-graph-max-bs`, currently 24;
* `req_to_token` `(max_running_requests + 1) × (context + 4 + draft_tokens) × 4 B`
  (`mem_cache/memory_pool.py:352-363`) — small, ~23 MiB.

Resulting pool (mamba trimmed to 25 slots, `--max-running-requests 6`,
`--cuda-graph-max-bs 8`, 400 MiB corridor, all fixed costs subtracted):

| context | keep tenant + MTP | keep tenant, no MTP | stop tenant + MTP | stop tenant, no MTP |
|---|---|---|---|---|
| 393 216 | **502 502** | 702 899 | 627 589 | 950 400 |
| 524 288 | 500 002 | **700 243** | 622 590 | 945 088 |
| 786 432 | 495 002 | 694 931 | 612 590 | **934 464** |
| 1 000 000 | 490 929 | 690 603 | 604 444 | 925 809 |

**Even the most aggressive column — no tenant, no speculative decoding, trimmed
GDN pool, zero transient margin — tops out at 925 809 tokens, below 1 000 000.**

## 4. Host spill pool — the second wall

`--kv-session-offload-host-ram-gib 0.0` (the default) does **not** mean "off, no
RAM". It means the pinned pool is auto-sized from `--context-length`
(`model_executor/model_runner_kv_cache_mixin.py:2549-2670`):

```
region_tokens   = (context_len // S + 2) * max_ratio     # S = 1, max_ratio = 1 at dcp_size 1
need_tokens     = region_tokens * max_spills
per_token_bytes = head_num * head_dim * layer_num * itemsize * 2
```

The pool is `pin_memory=True` and allocated **eagerly at boot**
(`mem_cache/pool_host/mha.py:151-187`). Regions are uniform and sized to the
**maximum** context, so a 262 144-token session occupies a full region.

Host RAM truth: **98 GB total, no swap**; the container's cgroup
`memory.current` is 23.4 GB. `/proc/meminfo` and `free` are lxcfs-distorted (they
claim 120 GB), and the launcher's own plausibility check reads `psutil.available`
— **so on this box that guard is blind and will not stop an oversized pool.**
Size it by hand. Budget after a 10 GB reserve: **64.6 GB**.

Node-wide pinned bytes, `region × max_spills × 32 768 B`:

| context | 1 spill | 2 | 3 | 4 |
|---|---|---|---|---|
| 393 216 | 12.9 GB | 25.8 GB | 38.7 GB | **51.5 GB** |
| 524 288 | 17.2 GB | 34.4 GB | **51.5 GB** | 68.7 GB |
| 786 432 | 25.8 GB | **51.5 GB** | 77.3 GB | 103.1 GB |
| 1 000 000 | **32.8 GB** | 65.5 GB | 98.3 GB | 131.1 GB |

Bold = the largest value fitting 64.6 GB. At a 1 000 000 context only **one**
session can be spilled at a time; Variant B's four concurrent spills would need
131 GB.

**Leave `--kv-session-offload-host-ram-gib` at 0 (auto).** As an explicit budget
it is divided uniformly by `tp_size`, but rank 0 needs twice what ranks 1/2 need,
so a budget large enough for rank 0 wastes a third of itself on the other two:
`max_spills 4` at 393 216 would need a 77.4 GB node budget to deliver the 51.5 GB
actually used. The auto path sizes each rank from its own cell size and is exact.

## 5. Why the context number is nominal above the pool size

`managers/tp_worker.py:484-494`:

```python
max_req_len = min(self.model_config.context_len - 1,
                  self.model_runner.max_token_pool_size - 1)
return (..., max_req_len, max_req_len - 5, ...)   # max_req_input_len
```

`max_token_pool_size` is `max_total_num_tokens` for a non-SWA model. Enforced in
`managers/utils.py:189-217` and at `scheduler.py:2787`, `:3074`. **A request is
refused above `max_total_num_tokens - 6` no matter what `--context-length`
says**, and there is no per-request or per-lane override. Combined with section 3
this is the decisive constraint: setting `--context-length 1000000` would produce
a server that advertises 1M and refuses anything past roughly 500 000.

Consequence for the design: **choose the YaRN factor so that the resulting
context is at or below the achievable pool**, not above it.

## 6. YaRN configuration — the working route

The rope block is transformers-v5 style: `text_config.rope_parameters`, aliased
onto `rope_scaling` by `configs/qwen3_5.py:19-34`. The model is **M-RoPE**
(`mrope_section [11, 11, 10]`, `mrope_interleaved true`), and `sum([11,11,10]) =
32 = rotary_dim // 2`, i.e. the sections are already sized for the partial
rotary dim of 64.

Three candidate mechanisms, measured:

| route | result |
|---|---|
| `--json-model-override-args '{"rope_scaling": …}'` | **inert.** `config.update()` is a flat setattr loop (`utils/hf_transformers/config.py:297-298`); the parser copies text-config attrs up before the override, so the text config never sees it. Measured: `rope_type` stays `default`. |
| `--json-model-override-args '{"text_config": …}'` | **destructive.** Replaces `Qwen3_5TextConfig` with a plain dict; measured `AttributeError`. Even a complete dict loses `layers_block_type` and fails the `hybrid_gdn_config` isinstance check, so the model stops being a GDN hybrid. |
| `--decrypted-config-file config_yarn.json` | **no effect on this tree.** Measured: sibling file ignored, `rope_type` stays `default`, derived context 262 144. |
| **overlay model directory** | **works.** |

`scripts/dev/543_yarn/make_yarn_overlay.py` builds a directory of symlinks to the
real checkpoint with one patched `config.json`. No weights are copied. Measured:

```
baseline    ctx=262144  rope=default f=None  mrope=[11,11,10] cfg=Qwen3_5Config fullattn=16 kv=4
yarn1.5     ctx=393216  rope=yarn    f=1.5   mrope=[11,11,10] cfg=Qwen3_5Config fullattn=16 kv=4
```

Two rules the script encodes:

* **Keep `mrope_section` and `mrope_interleaved`.** The factory routes to
  `YaRNScalingMRotaryEmbedding` only when `mrope_section` is present
  (`rotary_embedding/factory.py:234-272`); `model_runner.model_is_mrope` is
  derived from the config either way, so dropping them desynchronises the two.
* **Omit `original_max_position_embeddings`.** `get_context_length`
  (`utils/hf_transformers/common.py:419-436`) forces the scaling factor to 1 when
  that key is present, collapsing the derived context back to 262 144 so
  `--context-length` raises. Omitting it makes the factory default
  `original_max_position` to `max_position_embeddings` = 262 144, which is the
  wanted value; the rope math is identical, only the context gate differs.

Verified as unaffected: the 48 GDN layers hold no rotary embedding at all
(`ALL_DECODER_LAYER_TYPES`, `models/qwen3_5.py:1274-1277`; grepping `rotary` in
the model file hits only `Qwen3_5AttentionDecoderLayer`), and the GDN state
manager has no `context_length` term.

YaRN at `rotary_dim 64` is well-formed: `_compute_inv_freq` uses `self.rotary_dim`
throughout, and at `base 1e7`, `original 262144` the correction range is
`low 14, high 22`, clamped inside `[0, 63]` — no degeneracy. The fused Triton
path this model actually takes (`models/qwen3_5.py:1074-1103` →
`layers/fused_qk_rmsnorm_rope_gate.py:143-213`) supports partial rotary via
`HAS_PASS` and is scaling-agnostic, so it picks up the YaRN cache including the
baked-in `mscale`.

**Cliff to avoid: do not set `--context-length` at or above 1 048 320.**
`reserve_rope_cache_for_long_sequences` (`utils/common.py:5049-5086`) extends the
cache via `_ensure_cos_sin_cache_length`, which calls
`self._compute_inv_freq(self.base)` — but the YaRN subclasses override
`_compute_inv_freq(scaling_factor)`, so `base = 10_000_000` would be passed as
the scaling factor and the appended rows would also miss the `mscale` multiply.
Below the cache length the extension is a no-op, so staying under the cliff is
sufficient.

**Quality caveat:** static YaRN is boot-global. Every request, including the
262 144 pool sessions and ordinary short chats, runs with the scaled rope. Static
YaRN is known to cost some short-context quality. This is not measurable from the
config and needs the short-context sanity probe in section 10.

## 7. Spill gates under speculative decoding

The live boot runs `--speculative-algorithm NEXTN`. Four gates, in order of impact:

1. **Boot refusal.** kvso × any speculative algorithm raises unless
   `KVSO_ALLOW_SPEC=1` (`server_args.py:6606-6639`). Documented as deliberate
   opt-in, not unimplemented; the named unobserved case is a spill landing in the
   same round as a drafter-in-tick step.
2. **`KVSO_RESUME` is unset by default**, and with a spec algorithm active
   `_maybe_restore_flow` returns immediately (`kv_session_offload.py:503-509`,
   `:4401-4407`). **A spilled session under MTP never returns to device** — it
   finishes on host at tick rate. This is a larger hazard than gate 1 and is not
   mentioned in the refusal text.
3. **Only the back-most request is spillable under spec**
   (`spec_decline_non_back_spill:1103-1121`). The documented FCFS youngest-first
   victim order — key `(spill_class_rank, is_fast_lane, -kv_arrival_seq)`,
   `:867-887` — is largely bypassed, and a declined spill falls back to **stock
   retraction**, i.e. discard and re-prefill.
4. **A lone session never self-spills.** Under decode OOM the oldest normal
   session is tabu (`:959-967`), and a single request larger than the device
   budget truncates rather than spilling (`:932-940`).

**Recommendation: drop MTP for this boot layout rather than setting
`KVSO_ALLOW_SPEC=1`.** It removes gates 1-3 at once instead of opting into an
unobserved round plus a silent never-resume behaviour, and it is worth roughly
200 000 extra resident tokens (section 3, column 2 against column 1) — the
single largest VRAM lever available. The cost is decode throughput on the short
pool sessions; on a long lane the decode step is dominated by attention over the
resident KV, where MTP's relative contribution is smaller.

If MTP must stay, the boot needs `KVSO_ALLOW_SPEC=1` **and** `KVSO_RESUME=1`,
and `--draft-kv-layout replicated` (the current value; `dcp` is refused outright,
`server_args.py:7623`).

## 8. Other kvso settings that must not stay at their defaults

| flag | default | why it must change |
|---|---|---|
| `--kv-session-offload-max-spills` | **1** | only one session may be spilled at a time; further pressure falls back to retraction. Concurrent multi-spill is unit-tested but **never shown on hardware**. |
| `--kv-session-offload-wave-back-min-free-tokens` | **0** | 0 = wave a block back whenever any slot is free, which drains the host tail faster than the tick refills it and caps reachable depth. |
| `--kv-session-offload-restore-margin-tokens` | 4096 | scale with context. |
| `--kv-session-offload-spill-progress-lock-tokens` | 0 | 0 = no anti-pendulum lock. |
| `--kv-session-offload-mtp-resident-slices` | 0 | 0 **disarms** `spec-in-tick`; a positive value is permanently subtracted from the KV allocator. Only relevant if MTP is kept. |
| `--kv-session-offload-host-ram-gib` | 0.0 | **leave at 0** — see section 4. |
| all `--kv-session-offload-budget-*` | 0 | 0 = regulator off. Fine to leave off. |

Per-request control is `spill_class` ∈ `preferred | normal | never`, sent as a
top-level field on `/generate` or `extra_body={"spill_class": …}` on the OpenAI
endpoints (`entrypoints/openai/protocol.py:472-505`, `:876-919`). A "session"
here is **one in-flight request** keyed by `req.kv_arrival_seq`; it is unrelated
to `--enable-session-radix-cache` or `--enable-streaming-session`.

**There is no per-request context cap.** The only length gate is the global
`max_req_input_len` of section 5. `kv_session_offload_budget_session_tokens`
bounds a session's host-resident spill volume, not its context. So the 262 144
cap on pool sessions must be enforced **client-side**.

kvso is mutually exclusive with `--enable-hierarchical-cache`, unified memory,
hisparse, mixed-chunk, weightless-KV fastlane, PD disagg, `pp>1`, `dp>1`, and
requires `page_size 1` and the flashinfer backend (`server_args.py:6640-6685`).

Observability is thin: the only Prometheus series are
`sglang:spill_tier_used_bytes{spill_tier="kv_session_host_ram"}` and
`sglang:spill_tier_total_bytes{…}` (`observability/metrics_collector.py:733-750`),
region-quantized, so a spill shows as a one-region step, not a counter.
`sglang:num_retracted_requests_total` climbing is the signal that a spill was
**declined**.

## 9. The two shapes that fit

**Option 1 — Variant B geometry, 1.5× native lane (recommended).**
YaRN factor 1.5 → context 393 216. Four concurrent spills, pinned host pool
51.5 GB. Pool 502 502 with the tenant and MTP kept, 702 899 without MTP. Delivers
the five simultaneously schedulable sessions the ticket is actually about, with
the long lane at 1.5 × native. Dropping MTP is recommended for the gate reasons
in section 7, and is not needed for capacity here.

**Option 2 — biggest single lane, 3× native.**
YaRN factor 3.0 → context 786 432, requires stopping the translator tenant **and**
dropping MTP (pool 934 464). Only two concurrent spills fit (51.5 GB pinned), so
three sessions are guaranteed, not five.

Recommendation: **Option 1.** The requirement that motivated the ticket is five
simultaneously schedulable sessions; Option 2 cannot spill more than two of them,
and it costs the translator tenant.

Proposed boot, as a delta on the live command (PID 1236):

```
  --model-path   <overlay dir from make_yarn_overlay.py --factor 1.5>
  --tokenizer-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8
  --context-length 393216
  --max-running-requests 6
  --max-mamba-cache-size 25
  --cuda-graph-max-bs 8
  --enable-kv-session-offload
  --kv-session-offload-max-spills 4
  --kv-session-offload-restore-margin-tokens 16384
  --kv-session-offload-spill-progress-lock-tokens 256
  --kv-session-offload-wave-back-min-free-tokens 8192
  # drop: --speculative-algorithm NEXTN and its four companions
  # unchanged: --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
  #            --rank-auto-reserve-mib 13000,4200,4200 --kv-cache-dtype fp8_e4m3
  #            --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  #            --enable-fast-lane --retraction-policy priority --enable-metrics
```

`--rank-auto-reserve-mib` has to be re-solved for the new pool target; the
numbers in section 3 assume the reserve is spent down to the 400 MiB corridor.

## 10. Validation plan (not yet executed)

Under the `/spinning/gpu-arb` protocol, holder plus heartbeat, heartbeat stopped
before release.

1. Boot proof: `YaRNScalingMRotaryEmbedding` in the log (not `MRotaryEmbedding`),
   `rotary_dim=64`, `max_total_num_tokens` at or above 393 216, and the
   `kv-session-offload (P2 budget): attached %d-token pinned host pool` line with
   the effective `max_spills`.
2. Short-context sanity: a plain request answers correctly under the YaRN boot —
   the static-YaRN quality caveat of section 6.
3. Long lane: one session driven past 262 144 (300 000+) to prove past-native
   correctness, with four pool sessions active.
4. Spill observed: `sglang:spill_tier_used_bytes{spill_tier="kv_session_host_ram"}`
   steps up while `sglang:num_retracted_requests_total` stays flat (a climbing
   retraction counter means the spill was declined).
5. Corridor held: ≥ 400 MiB free on all three cards, sampled at 100 ms to catch
   transients.
6. Five-session concurrency smoke.
7. `feat/thinking-budget-540` riding along: one budgeted request.

## 11. Collision with the bundled restart requests

* **`--enable-hierarchical-cache` is mutually exclusive with
  `--enable-kv-session-offload`** (`server_args.py:6664-6667`). The 100 GB disk
  HiCache tier requested for the bundled window **cannot** be in the same boot as
  the spill lane this task needs. Branch `feat/hicache-runtime-544` currently has
  no commits ahead of the integration tip, so nothing is lost by giving it its own
  window. Note also that the disk tier cannot substitute for kvso here: HiCache is
  a prefix cache, it rehydrates on prefix match between turns, it does not hold a
  live session's KV out of the way while another session runs.
* `feat/thinking-budget-540` is ready (one commit) and orthogonal; it can ride
  this boot.
