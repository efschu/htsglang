# Axis C — numeric defaults that bound nothing (binds-proof backlog)

Desk audit, nothing executed, no GPU, no boot. Worktree `/spinning/wt-505-silent`,
branch `docs/silent-wrongness-505`, base `d6534052231276171daf3a844476812ec702ccf3`.
Upstream reference for the fork-delta: `upstream/main` = `ec741e4161` (2026-08-02).

**The law under audit** (CLAUDE.md:23-29, verbatim): *"REACH INCLUDES PARAMETERS
(#493 lesson): a cap/threshold/budget that never BINDS at the served geometry has
reach zero — the #449 query-chunk cap existed and was correct, but shipped at a
desk-picked 2048 MiB above the real peak, so it protected nothing for weeks. Any
shipped numeric default that exists to bound something needs a binds-proof at the
target geometry (a falsifier where the default measurably changes behavior); 'the
mechanism exists' is not evidence that it acts."*

**Target geometry** (what "binds at the served geometry" means below): 1x RTX 5090
32 GiB sm120 + 2x RTX 3080 20 GiB sm86, single node, no NVLink, no GPUDirect P2P,
all PHB, GPU0 on x4. Standing recipe: uneven TP=3 (`--rank-tp-ratio`), uneven DCP
(`--rank-kv-ratio`), NEXTN speculation, barlink transport, 27B-35B models plus a
122B-A10B offloaded MoE and DeepSeek-V4-Flash GGUF.

**Evidence ladder used for the verdicts** (descending strength): (a) a test that
fails when the default is raised/removed — a real falsifier; (b) a recorded
MEASUREMENT in `docs/dev/*.md` or a commit message naming the value and the
observed peak/rate; (c) a NOTE at the constant stating how the value was derived
from a measurement.

---

## Coverage

| surface | fork | upstream/main | fork-added | numeric | bounding-worded (this audit's set) |
|---|---|---|---|---|---|
| `python/sglang/srt/environ.py` `class Envs` | 572 | 519 | **115** | 47 | **31** |
| `python/sglang/srt/server_args.py` `class ServerArgs` | 598 | 457 | **166** | 103 | **75** |
| **total** | | | 281 | 150 | **106** |

Enumeration is AST-based, not regex: `ast.parse` over both files, walking
`ClassDef("Envs")` `Assign` nodes and `ClassDef("ServerArgs")` `AnnAssign` nodes,
with the preceding comment block / the `A[type, Arg(help=...)]` string literals
attached to each entry, and set-differenced against the same walk over
`git show upstream/main:<file>`. The 166 fork-added `ServerArgs` fields reproduce
audit #500's count exactly, which is the cross-check that the walk is right. The
briefing said 573 `Env*` entries; the AST finds 572 assignments inside `class Envs`
— the difference is not chased further, it does not change any verdict.

"Bounding-worded" = the NAME or the comment/help text contains one of
`cap|budget|threshold|limit|max|min|timeout|reserve|margin|quota|headroom|ceiling|floor|chunk|watermark|interval|retries|rounds|bound|allowance|safety|above which|below which`.
`--rank-tp-ratio` / `--rank-mlp-ratio` / `--rank-vocab-ratio` / `--rank-moe-ratio`
match the filter only through "length must equal `tp_size`" and are **not** bounds;
they are excluded from the tables.

**Out of scope, counted not audited:** 519 upstream `Env*` entries and 457 upstream
`ServerArgs` fields. Per the standing rule ("fix bugs in OUR features, not all of
sglang") they were enumerated for the set difference and then dropped. One upstream
observation is recorded in §5 because it surfaced from the same script and is a
40x default drift.

**Opened individually** (consumer site read, gate predicate read at its source,
`docs/dev/` + `git log` searched for the value): **42** of 106. **Not opened: 64** —
listed in §6 with the reason. Nothing in §6 is claimed to be fine; it is unexamined.

### What AUDIT_434 already discharged (so this audit's delta is visible)

`docs/dev/AUDIT_434_planner_constants.md` was an exhaustive sweep of **module-level
numeric constants and in-function numeric literals** in exactly two places:
`python/sglang/srt/uneven_perf.py` (6125 lines) and `python/sglang/srt/planner/**`
(59 modules). 764 candidate literals, triaged into `PROBE-FED` (11) / `STRUCTURAL`
(17) / `RIG-FITTED` (19) / `POLICY` (15) / `UNKNOWN-PROVENANCE` (1), with 16
follow-up tasks FU-434-1..16. Its question was *generality* ("is this number fitted
on the reference rig and applied unconditionally elsewhere"), not *reach* ("does
this number ever bind here").

It did **not** touch `environ.py` or the `ServerArgs` dataclass — the two surfaces
this audit enumerates. The only overlap is four rows, and they are not re-reported
here: `SGLANG_PERF_DECODE_GEMV_RESIDUAL_EXP` / `_PEAK_COMPRESSION_EXP` /
`_NONWEIGHT_FRACTION` / `_PREFILL_INVARIANT_FRACTION` (`environ.py:611-614`) are the
declared *seams* for AUDIT_434's four calibration scalars; #434 established that the
seam exists, is documented, and that **nothing populates it automatically**
(FU-434-1/2/3). They are `EnvFloat(None)` — absent-markers, not bounds — so they
fall outside this axis anyway. `SGLANG_PLANNER_CORRIDOR_MIB` (`environ.py:621`) is
#434's `POLICY` row for the #330 corridor and is likewise not re-litigated.

Everything else below is new ground.

---

## 1. Table INERT (no consumer, or consumer behind a gate that is off by default)

**No fork-added bounding default was found with zero consumers.** The first pass
produced 14 apparent zero-consumer fields; every one resolved to a real consumer on
a second look — the #236 spill budgets are read through a string-built
`getattr(sa, "kv_session_offload_" + name, default)`
(`managers/kv_session_offload.py:1305-1321`), `pp_stage_ratio` is consumed inside
its own declaring file (`server_args.py:12847`), and so on. Recorded because
"accepted-then-inert with no consumer at all" was the strongest finding this axis
could have produced, and it is **not** present. What *is* present is the weaker but
much broader form: consumers behind a gate that is off in the standing recipe.

| posten | file:line | claimed protection (verbatim) | why inert (gate file:line) |
|---|---|---|---|
| `admission_throttle_high` = 0.9 | `server_args.py:1099` | "Pool-occupancy fraction (0..1] at or above which the dynamic admission limit (#287) is lowered." | `admission_limiter.py:209` `if not self.auto: return False` in `observe()`; armed only by `scheduler.py:2323` `auto=sa.max_running_requests_ceiling is not None`, default `None` (`server_args.py:1061`). The limiter object is always built, so it *looks* live in a snapshot; `observe()` returns before reading either mark. |
| `admission_release_low` = 0.7 | `server_args.py:1109` | "Pool-occupancy fraction at or below which the dynamic admission limit (#287) may be raised again." | same gate |
| `admission_release_hysteresis` = 8 | `server_args.py:1119` | "Consecutive samples at or below --admission-release-low required before the dynamic admission limit is raised" | same gate |
| `admission_floor` = 1 | `server_args.py:1089` | "Lowest value the dynamic admission limit (#287) may float down to." | same gate; also `min(sa.admission_floor, ceiling)` at `scheduler.py:2319` makes 1 unreachable as a bound except at ceiling 1 |
| `fast_lane_reserved_heavy_slots` = 1 | `server_args.py:1223` | "Anti-starvation floor: the minimum number of running lane='heavy' requests that fast-lane preemption may not go below" | `schedule_policy.py:1410` `if getattr(server_args, "enable_fast_lane", False):` — `max_heavy_preemptible` stays `None` otherwise; `enable_fast_lane` default `False` (`server_args.py:1210`) |
| `kv_pressure_ascend_threshold` 0.85, `_ascend_window` 4, `_descend_threshold` 0.55, `_descend_window` 64, `_pre_stage_threshold` 0.7, `_pre_stage_window` 3, `_abort_stage_window` 32, `_horizon_rounds` 32, `_external_hysteresis_rounds` 512, `_consensus_interval` 8 | `server_args.py:4874-4962` | e.g. "Flip mark of the KV pressure ladder: occupancy fraction (0..1] at or above which the ladder climbs." | `kv_pressure_ladder.py:1944` `spec = parse_kv_pressure_ladder(getattr(server_args, "kv_pressure_ladder", None))` → controller is `None`; `--kv-pressure-ladder` default `None` (`server_args.py:4833`). Docstring at `:1928-1931`: "or ``None`` when the flag is unset (= today's behavior, byte-identical: nothing is constructed, no hook is attached, no sample is taken)." |
| the ten #236 spill reglers: `kv_session_offload_budget_total_tokens` 0, `_session_tokens` 0, `_prefill_tokens` 0, `_decode_tokens` 0, `_rate_tokens_per_s` 0.0, `_episode_seconds` 0.0, `_max_sessions` 0, `_spill_progress_lock_tokens` 0, `_spill_hysteresis_steps` 0, `_spill_cooldown_seconds` 0.0 | `server_args.py:1977-2092` | e.g. "maximum host-resident spill volume in TOKENS across ALL spilled sessions" | DOUBLE gate: `enable_kv_session_offload` default `False` (`server_args.py:1763`), and inside the feature `SpillBudgetConfig.armed` (`managers/kv_session_offload.py:1324-1338`) is False while every regler is zero — "All-zero (the default) -> every hook is skipped, byte-identical." So even with the feature ON, the whole #236 budget is disarmed by default. |
| `kv_session_offload_budget_demote_grace_iters` = 256 | `server_args.py:2101` | "scheduler iterations a DEMOTED session may wait for its host tail before it falls back to a host finish" | as above, and additionally: 256 is the one #236 regler that is NOT zero, yet it is not a member of the `armed` disjunction (`kv_session_offload.py:1327-1338`) — so it cannot arm the machinery on its own and never binds unless some *other* regler is set. |
| `kv_session_offload_restore_margin_tokens` 4096, `_tick_interval` 1, `_tick_floor` 8, `_max_spills` 1, `_host_ram_gib` 0.0, `_mtp_resident_slices` 0, `_park_timeout_iters` 512, `_wave_back_min_free_tokens` 0 | `server_args.py:1796-2181` | e.g. "restore the spilled session only when the allocator has (session tokens + this margin)" | `enable_kv_session_offload` default `False` (`server_args.py:1763`); `_handle_kv_session_offload` (`server_args.py:6347`) refuses each standalone by name, so they are *validated* and then unreachable |
| `dual_group_lane_budget_mib` None, `_admission_ms` 2.0, `_pairing_sat_rows` 64, `_pairing_max_defer_ms` 500.0, `_lend_mib` 0, `_lend_threshold_s` 5.0, `_spec_adaptive_hysteresis` 4, `_share_window_s` 0.0, `_share_min` None, `_share_min_windows` 5, `_prefill_chunk` None | `server_args.py:4365-4704` | e.g. "Starvation cap for --dual-group-lane-pairing: a queue head skipped in favour of better-pairing jobs" | `dual_group_lane` default `False` (`server_args.py:4324`) |
| `weightless_kv_chunked_block_size` 0, `_host_spill_tokens` 0, `_spill_device_cap` 0 | `server_args.py:1706-1750` | e.g. "cap the ALLOCATABLE device-resident KV slots" | `weightless_kv_fastlane` default `False` (`server_args.py:1655`); each of the three is additionally its own on/off value at 0 |
| `SGLANG_WL_GRAPH_MAX_BS` = 1 | `environ.py:1243` | "Weightless-KV streaming block-decode graphs (#136a): max decode capture bucket." | `decode_cuda_graph_runner.py:403-408` requires `model_runner.is_weightless_head or .is_weightless_worker` AND `_wl_chunk_block_size` — both off without the lane |
| `SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB` = "400" | `environ.py:376` | "Scalar MiB or a comma list with one value per TP rank (roles differ: the draft-solo host carries prompt-length-scaled serving transients)." | `SGLANG_MEASURED_KV_BUDGET` default `False` (`environ.py:373`); guard `model_runner_kv_cache_mixin.py:896` `if not envs.SGLANG_MEASURED_KV_BUDGET.get():`. See §4 finding #505-C-03 — when it IS on, the value contradicts a measurement recorded ten lines above it. |
| `SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB` = 1024 | `environ.py:383` | "how many MiB of this rank's device share may be used by things outside its own allocator reservation (CUDA context, NCCL buffers) before the leftover measurement is reported as contaminated by a FOREIGN consumer" | same gate; consumer `model_runner_kv_cache_mixin.py:1196` sits below `if not envs.SGLANG_MEASURED_KV_BUDGET.get(): return` at `:1098` |
| `SGLANG_MOE_HEAT_DECAY` 0.5, `_MIN_GAIN` 8.0, `_MAX_SWAPS` 4, `_MIN_OBS` 32 | `environ.py:1057-1072` | e.g. "Upper bound on swaps per layer per round; the burst is swaps x expert bytes" | `SGLANG_MOE_HEAT_MIGRATION` default `False` (`environ.py:1051`) |
| `SGLANG_MOE_COLD_TIER_MANIFEST_TIMEOUT_S` = 30.0 | `environ.py:1210` | "Bounded wait for a peer's cold-tier manifest at the FIRST fetch" | `SGLANG_MOE_COLD_TIER_SHM` default `False` (`environ.py:1206`) |
| `SGLANG_EXPERT_STATS_INTERVAL_SEC` = 0.0 | `environ.py:1169` | "Additionally dump every N seconds (0 = only on exit / SIGUSR2)." | `SGLANG_EXPERT_STATS` default `False` (`environ.py:1146`), and 0.0 is itself "off" |
| `SGLANG_VRAM_DIAL_CHUNK_MIB` 16, `vram_dial_consensus_interval` 8 | `environ.py:365`, `server_args.py:5097` | "physical commit chunk of the VMM-backed KV pool in MiB" | `enable_vram_dial` default `False` (`server_args.py:5067`) |
| `kv_reshard_consensus_interval` = 8 | `server_args.py:5056` | "Scheduler rounds between two consensus boundaries of the #297 KV reshard runtime" | `kv_reshard_vectors` default `None` (`server_args.py:5034`) |
| `gdn_state_set_ladder_hysteresis` = 2 | `server_args.py:4823` | "Lowering hysteresis of --gdn-state-set-ladder, in admission cycles" | `gdn_state_set_ladder` default `None` (`server_args.py:4780`) AND the #500-B11 register gate `SGLANG_OFFLOAD_REGISTER` default `False` (`environ.py:1079`, `model_executor/offload_gdn_states.py:344`) |
| `SGLANG_LOGITS_PROCESSER_CHUNK_SIZE` = 2048 | `environ.py:1513` | (no comment; the flag it belongs to is `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK`) | `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK` default `False` (`environ.py:1512`), read at `layers/logits_processor.py:418` |
| `SGLANG_MAMBA_CKPT_WINDOW` = 2 | `environ.py:394` | "how many of the deepest on-grid mamba checkpoints per radix path evict_mamba keeps live" | `mamba_checkpoint_interval` default `None` (`server_args.py:3812`); `mem_cache/mamba_radix_cache.py:447` "None = upstream behavior, byte-identical" |
| `SGLANG_GGUF_STREAM_TRIM_TARGET_GIB` = 0.0 | `environ.py:1800` | "Reclaim down to about here once the soft watermark is crossed." | the soft watermark itself is 0.0 = off (`environ.py:1798`) — see §4 finding #505-C-02 |
| `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` = 0 | `environ.py:1110` | "Max decode batch size eligible for the captured offload path. Buckets with bs*top_k > scratch (would need >1 wave) cannot be captured" | self-gated: `decode_cuda_graph_runner.py:380` `if _moe_offload_graph_bs > 0:` — the default disables the cap. **Low severity, deliberately**: the invariant it names is separately enforced by a hard raise at capture (`layers/moe/fused_moe_triton/layer.py:2276-2286`, "worst-case unique spill must fit the scratch region, or a captured step could silently drop spill experts (wrong output, not epsilon)"). This is the one place in the sweep where a disarmed cap is demonstrably backstopped. |
| `SGLANG_SPEC_STATE_HASH_MAX_MB` = 0 | `environ.py:429` | "0 = hash every tensor fully. >0 = tensors above this many MiB are fingerprinted from a strided sample" | `SGLANG_SPEC_STATE_HASH` default `False` (`environ.py:425`) |
| `SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL` = 0 | `environ.py:1425` | "TEST-ONLY: force an adaptive runtime-state swap every N verify completions" | self-gated at 0; test-only by its own text |
| `SGLANG_PP_BOUNDARY_STATS` = 0 | `environ.py:523` | "log the stage-boundary traffic every N crossings (0 = off)" | self-gated at 0 |
| `SGLANG_BARLINK_SLOT_MIB` 64, `_CHUNK_MIB` 8 | `environ.py:652-654` | "Per-rank shared-memory slot size (MiB) for payload staging" / "Chunk size (MiB) of the gloo data-plane pipeline" | belong to the shm/gloo data planes; `SGLANG_BARLINK_TRANSPORT` default `"device"` (`environ.py:645`). (`_SLOT_BYTES` is still consulted by the device transport's shm segment, `barlink_device.py:1254`; the *chunk* is not.) |
| `SGLANG_BARLINK_HOST_P2P_MIB` 4, `_HOST_BLOCKS` 32, `_HOST_SLOT_MIB` None | `environ.py:672-680` | "Grid width of the host transport's two data kernels … more blocks buy nothing below ~1 MiB and cost tail latency" | host transport only; gate as above |
| `SGLANG_BARLINK_UCX_CHUNK_MIB` 4, `_UCX_RING_KIB` 24, `_UCX_AG_RING_KIB` 32, `_UCX_GRAIN_ELEMS` 32768, `_UCX_TIMEOUT_S` 300, `_UCX_RING_MIB` None | `environ.py:694-721` | "all_reduce payload (KiB) at or above which the one-step flat exchange gives way to a ring" etc. | UCX/RDMA data plane only; gate as above, and the target rig is single-node with no RDMA. Their derivations ARE measured (see §3) — measured on a cross-rig world-4 UCX link that this geometry cannot reach. |
| `disaggregation_prefill_budget_mib` None, `disaggregation_prefill_lane_interval` 1 | `server_args.py:4294-4305` | "Prefill-side activation/scratch budget in MiB PER CARD, an explicit item of the boot-time VRAM check" | PD disaggregation off by default (`disaggregation_mode`/`disaggregation_topology` `None`, `server_args.py:4237`) |
| `training_idle_grace_seconds` 120.0, `_poll_seconds` 2.0, `_preempt_timeout_s` 120.0, `_save_steps` 50, `_event_stream_timeout_s` 120.0 | `server_args.py:5479-5523` | "How long a preempted trainer may take to checkpoint and exit before it is killed. Bounded on purpose" | training tenant off by default |
| `workbench_preempt_timeout_s` 60.0, `_segment_timeout_s` 1800.0, `_probe_max_age_s` 604800.0 | `server_args.py:5577-5617` | "Hard bound on one segment. A tenant whose work does not fit inside this must cut it into smaller iterations." | `enable_idle_workbench` default `False` (`server_args.py:5537`), `workbench_tenants` default `None` (`:5557`) |

**Total INERT rows: 24 groups covering 71 individual postens.** Of the 106
bounding-worded fork-added defaults, **71 (67 %) cannot act in the standing recipe
at all.** That is not by itself a defect — most are honest opt-in features that say
so at their site — but it is the denominator that matters for the next table: only
~35 of 106 shipped bounding defaults are even *reachable* on the served geometry.

---

## 2. Table UNPROVEN, ranked by damage potential

Reachable in the standing recipe, consumer read at its source, no evidence of class
(a), (b) or (c) that the value binds here.

| posten | default | consumer file:line | claims to protect | what happens if it never fires (or fires wrongly) | proposed binds-proof (concrete falsifier) |
|---|---|---|---|---|---|
| `DEFAULT_TIMEOUTS_S[LLM_STREAM]` | **90.0 s** | `liveness/classes.py:84`, resolved at `liveness/watchdog.py:121`, enforced at `watchdog.py:321-323` `if silent >= timeout: await self._declare_dead(...)`, which calls `tokenizer_manager.abort_request(rid)` (`liveness/stream.py:169-175`) | "Seconds of silence tolerated per class before the consumer is declared dead." The table is self-labelled at `classes.py:81-82`: **"Unmeasured; see :data:`DEFAULT_TIMEOUT_RATIONALE` for why each is what it is"**, and `classes.py:98-99`: "the numbers encode an argument about the consumer, not a measurement of the server". | This one fires **too early**, not too late. `last_progress_at` is set at watchdog construction (`watchdog.py:204-210`) and only advances when the transport ACCEPTS bytes (`note_progress`, `:217-224`). A stream that emits no bytes for 90 s — a request queued behind a long prefill, or a first-token latency on the 122B-A10B offloaded MoE at high context — is declared dead and **aborted while healthy**. Presents to the user as a randomly dropped stream with a `WARNING … releasing` line, not as an error. It is live by default on every OpenAI-shaped streaming endpoint: `serving_base.py:119-124` wraps every `request.stream` response in `guard_generate_stream(..., endpoint_class=self._liveness_endpoint_class())`, no feature flag. | Two parts. (i) **Does it bind?** Hermetic: construct a `ConsumerWatchdog` with `LLM_STREAM` policy and a fake clock, feed no `note_progress`, assert `_declare_dead` at 90 s and that `release()` calls `abort_request`. (ii) **Does it bind wrongly?** On the rig: measure TTFT (first byte accepted by the transport, not first scheduler token) for the 122B-A10B offloaded MoE at the longest supported context, and for a request queued behind one, and compare against 90 s. If max observed TTFT + queueing is within 2x of 90 s the default is unsafe on this geometry. NOTE: I did NOT verify whether the chat/completions generator emits an early role/keep-alive chunk that would restart the clock — establishing that is step 0 of the falsifier, and it decides whether the finding is severe or moot. |
| `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB` | **0.0 = OFF** | `environ.py:1798`, consumed in the GGUF weight stream (`SGLANG_GGUF_STREAM_DROP_CACHE` path) | "Synchronous cgroup reclaim during the GGUF stream, in GiB of memory.current. 0 (default) = off, behaviour byte-identical to before." | The comment at `environ.py:1804-1808` records the measurement that motivated the mechanism: *"on a swapless box that gap is the whole budget (#391). An external sampler chasing it on a wall-clock interval can be outrun -- window 3 saw memory.current move 88 -> 102 GiB inside one 15 s window."* The watermark built to stop that ships **disabled**, so the host-RAM wall it protects against is unprotected on every GGUF boot. The standing recipe serves DeepSeek-V4-Flash GGUF and 27B GGUF on a swapless box. | The measurement already exists (14 GiB in 15 s, window 3). What is missing is a value: boot the GGUF stack with `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB` at candidate watermarks derived from `memory.max` minus the observed 15 s slew, and show a boot that OOMs at 0.0 and survives at the candidate. A falsifier that fails with the default and passes with a value IS the binds-proof. |
| `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` | **2048** | `layers/attention/dsv4/indexer.py:337` `budget_mib = envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get()`, `:346` `rows = (int(budget_mib) * 1024 * 1024) // step_bytes` | "Bounds the per-query-token duplication of the KV gather described in ANALYSE_447 section 2.3 L1. 0 disables it (one pass over the whole query axis, the pre-#449 shape). See #449." (`environ.py:1631-1638`) | **This is the #493 lesson's own posten, and it is still at the desk value.** `docs/dev/NOTE_449_dsv4_indexer_query_chunk.md:213-226` says it outright: *"the default is a ceiling picked at desk, not a tuned value"*, and §5 of that note is headed **"GPU measurement arm — BOOT-PENDING, not run"**. Reachable at the target geometry: `server_args.py:11355` sets `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` to True for DeepSeek-V4 on sm120, so the chunked torch indexer is the path the 5090 runs. If 2048 MiB is above the real per-rank peak the loop runs exactly once and #449 protects nothing — the shape the law was written about. | NOTE_449 §5 already specifies it, step by step, and it has not been run: one boot of DeepSeek-V4-Flash-0731 in the `BENCH_394_v4flash_club3090.md` configuration, A-vs-A floor first, then interleaved A/B at 8K and 32K context with the budget at `0` vs `2048`, **peak allocated VRAM per rank as the primary result**. The binds-proof is: at the served context the default must measurably lower the peak. If it does not, the default is above the peak and must come down. |
| `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK` | **8192** | `layers/attention/dsv4/indexer.py:256` `chunk_positions = envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.get()` | "Sequence-axis chunk (in KV positions) … Bounds its peak intermediate at O(batch x chunk x heads) instead of O(batch x context x heads); see #426 / upstream #33246." (`environ.py:1622-1629`) | Same family, same path, same exposure as the row above — and the sibling that #449's own note calls the axis it "composes with rather than replaces". At 8192 KV positions the bound is inactive for every request whose context is below 8K, i.e. for a large part of the served traffic; whether it binds at the served 32K+ contexts is unrecorded. | Fold into the same boot as the row above: sweep `SEQ_CHUNK` at 0 / 4096 / 8192 and record peak allocated VRAM per rank at 8K and 32K. The pair (query-MiB, seq-chunk) must be swept together, because the per-row byte cost `_indexer_logits_step_bytes(chunk_seq, …)` is a function of the seq chunk — changing one changes what the other bounds. |
| `SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S` | **600.0** | `mem_cache/unified_radix_cache.py:419` `self.collective_timeout_s = envs.SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S.get()`, enforced in `_wait_bounded` (`:455-467`) | "Deadline for every cross-rank control collective issued from this cache … without it a dead peer parks this rank in all_reduce until the two-hour gloo group timeout expires." | A hang guard of the rank-local-condition-before-a-collective family the fork has hit four times (`CLAUDE.md` / `rank-lokaler-test-vor-kollektiv`). 600 s is argued only relative to the thing it replaces (7200 s), never against how long a legitimate HiCache control collective takes here. Too high: a wedged boot burns 10 minutes per collective before the named error. Too low: a legitimate slow collective aborts a healthy server. Neither direction is measured. | Instrument `_wait_bounded` to record the observed completion time of every control collective for one full serving window (it already loops on `work.is_completed()`), and set the bound at a stated multiple of the observed max. The falsifier: a test that patches the clock and asserts `HiCacheCollectiveTimeoutError` at exactly the configured bound already half-exists (`test/registered/unit/mem_cache/test_hicache_collective_wedge.py:102` asserts the message names the env) — it must additionally assert the *value*, which it does not. Reachable only on the hierarchical-cache path; rank the campaign accordingly. |
| `SGLANG_PERF_PROBE_LINK_TIMEOUT_S` | **45.0** | `uneven_perf.py:1365` `return float(envs.SGLANG_PERF_PROBE_LINK_TIMEOUT_S.get())`, expiry branch `uneven_perf.py:1410-1425` | "Wall-clock cap (seconds) on the NETWORK phase of the stage-0 probe (the pairwise NCCL link matrix) … without a cap it inherits torch's 600 s default process-group timeout and charges it to every boot." (`environ.py:594-601`) | Inverted damage: firing too EARLY is the harm. On expiry the probe "keeps the per-card measurements, stores the reason next to the empty link table, and returns" — and per the message at `:1420-1421` **"the plan falls back to its uniform link assumption"**. On this rig the links are emphatically not uniform (no NVLink, no P2P, all PHB, GPU0 on x4). A 45 s budget that is occasionally short would silently hand the planner a uniform-link model on the one rig where the link asymmetry is the point, on some boots and not others. | Record the wall time of the link-matrix phase across N cold boots of the standing TP=3 recipe (the phase already reports its own reason string) and compare the distribution against 45 s. A binds-proof here is the opposite of the usual one: the default must be shown to be *comfortably above* the observed max, not below it. Cheap: the number is already printed at every `auto-performance` boot. |
| `SGLANG_PERF_PROBE_TIMEOUT_S` | **600.0** | `uneven_perf.py:1624` `return float(envs.SGLANG_PERF_PROBE_TIMEOUT_S.get())` | "Wall-clock cap (seconds) on the WHOLE stage-0 probe subprocess." (`environ.py:592-593`) | Same shape, coarser. No derivation anywhere; a round 10 minutes. If short, a slow first probe is killed and the plan runs on no profile at all; if long, a wedged probe costs 10 minutes of every boot. | Same instrument as the row above — total probe wall time across cold boots, on the same run. |
| `SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES` | **8** | `managers/schedule_batch.py:2819` `max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()`, branch `:2820` `if last_req.solo_oom_count <= max_retries:` | "how many times in a row a request may be the sole survivor of retract_decode and still not fit before it is failed instead of re-queued again. Ordinary extreme pressure … resolves within a couple of scheduler iterations; a request still solo-OOMing past this many retries is structurally too large for the pool" (`environ.py:480-486`) | **Mechanism-proven, value-unproven — the cleanest example of the distinction this axis is about.** `test/registered/unit/managers/test_retract_decode_fcfs.py:219-265` proves the guard fires and that the failure is a clean 503, but it reads the default (`:232 max_retries = envs.….get()`) and loops `range(1, max_retries + 2)` — it passes for ANY value of 8. So nothing pins 8. If 8 is too low, a transiently contended request is failed with a 503 that the comment itself says should not happen ("transient, not a sign this request is unfittable"). If too high, an unfittable request occupies retract cycles for longer. | Measure the empirical distribution of `solo_oom_count` at which pressure actually resolves, under the load the comment names (kv-session-offload spill budget exhausted / extreme concurrency), and pin the default at a stated quantile. The unit test then gains a second case that fails when the default is halved. |
| `DEFAULT_ATTN_SCRATCH_BUDGET_MIB` / `--attn-scratch-budget-mib` | **640** (flag default `None` → falls back to 640) | `models/deepseek_common/attention_forward_methods/forward_mha.py:210-218` `budget_mib = get_server_args().attn_scratch_budget_mib; if budget_mib is None: budget_mib = DEFAULT_ATTN_SCRATCH_BUDGET_MIB` | "Per-rank MiB budget (#395) for DeepSeek's chunked-prefix / attention-scratch strategy switch" (`server_args.py:1331`) | Has a derivation note (`forward_mha.py:77-85`) but it is a **back-derivation, not a measurement**: 640 MiB is the value that reproduces upstream's old 8192-token threshold *bit-for-bit on DeepSeek-V3 at TP=1* (`num_local_heads=128`). The note says so and says the derived token threshold differs "by design" on every other geometry. Under uneven TP=3 on this rig no rank has 128 local heads, so the threshold every rank runs is an extrapolation of an upstream token count nobody measured a peak for. | Measure the actual per-rank scratch peak for the served DeepSeek geometry (`attn_scratch_bytes_per_token` is already a named function) and check whether 640 MiB is above or below it per rank. Falsifier: a boot at the derived threshold vs one at half of it, comparing peak allocated VRAM per rank. This posten is the sibling that #449's own comment cites as its model (`environ.py:1634-1636`), so proving one and not the other leaves the pattern unproven. |
| `SGLANG_BARLINK_PEER_TIMEOUT_S` | **120.0** | `barlink_liveness.py:107` `ENV_TIMEOUT_S = "SGLANG_BARLINK_PEER_TIMEOUT_S"`, policy applied in `barlink_shm.py:168` | "Seconds a host-side wait may make no progress before it gives up. Scaled by SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT…" (`environ.py:757-760`) | Live in the standing recipe (barlink is the standing transport, `SGLANG_BARLINK_PEER_LIVENESS` defaults True, `environ.py:756`). This is the guard against the wedge family; 120 s is a round number with no derivation. Too low and a legitimately slow cold-build barrier is declared a dead peer; too high and a real wedge costs 2 minutes per collective before anyone learns. | The scaling seam (`SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT`) proves the authors knew 120 s is too short for a cold build. Measure the longest legitimate host-side barrier wait across a cold-cache boot and a warm boot of the TP=3 recipe, and set the base from that; the falsifier is a test that patches the clock and asserts the named error fires at the configured bound, plus an arm proving a cold build does NOT trip it. |
| `client_liveness_grace_fraction` 0.25, `_poll_interval_s` 1.0, `_teardown_timeout_s` 30.0 | as shown | `entrypoints/http_server.py:291-296` → `LivenessConfig.parse` | "Fraction of a class's timeout after which a quiet consumer enters the grace window" | Live by default (same path as the 90 s row). Grace at 22.5 s for LLM streams puts claims on the reclaim ladder (`watchdog.py:328-336`) for any stream that is quiet for 22.5 s — well within a normal long prefill. No derivation. | Same boot as the LLM_STREAM row: record the observed distribution of inter-byte gaps for real traffic and check what fraction of healthy streams cross the grace mark. |
| the other 11 reachable `DEFAULT_TIMEOUTS_S` entries (`VIDEO_STREAM` 300, `PREVIEW_TAP` 15, `IMAGE_GENERATION` 900, `AUDIO_SPEECH` 300, `AUDIO_TRANSCRIPTION` 120, `REALTIME_SESSION` 60, `CONTROL` 60, `REGISTRY_LEASE` 120, `DASHBOARD_SSE` 60, `EMBEDDING` 60, `TRAINING_EVENTS` 120) | as shown | `liveness/classes.py:83-96` | same table, same self-label "Unmeasured" | Each aborts a live consumer of its class on expiry. `PREVIEW_TAP` at 15 s is the tightest and sits on the video-enhance chain, an asset class the ONE-RUNTIME law puts inside this server. | One table, one campaign: instrument `note_progress` to record inter-byte gap percentiles per class over a representative window, then set each default from its own distribution. Until then the table is 12 desk numbers that can each abort a healthy client. |

---

## 3. Table BOUND-PROVEN (the discriminating half)

**It is not empty — but only just, and not one row is evidence class (a).** No
shipped bounding default in this fork has a test that fails when the default is
raised or removed. Four rows have evidence class (c) — a note at the constant
naming the measurement it came from — and one has class (b). Every one of them
carries a caveat, given verbatim.

| posten | default | evidence file:line | class | caveat |
|---|---|---|---|---|
| `SGLANG_ADAPTIVE_SERVING_MARGIN_MIB` | 512 | `environ.py:1430-1432`: "Measured on the T102 rig: 148 MiB post-map free OOM'd at KV-full deep prefill, 1367 MiB survived; 512 is the enforced floor between them." Repeated at the enforcement site `speculative/adaptive_graph_memory.py:938-943` with the failing kernel named (`fla/wy_fast recompute_w_u_fwd`), enforced at `:947` `if free_bytes < max_bytes + margin_bytes: raise RuntimeError` | (c) | 512 is an **interpolation between an OOM point and a survival point**, not a measured threshold — the true boundary is somewhere in [148, 1367] and nobody narrowed it. The enforcement is a boot-time refusal, which is the right shape (fails fast rather than late), and the error text quotes the measured numbers. Strongest row in the sweep. |
| `SGLANG_BARLINK_UCX_RING_KIB` | 24 | `environ.py:695-701`: "Measured crossover on a cross-rig world-4 group is ~22 KiB (task #244), so a speculative verify all-reduce sits on the ring side and a bs=1 decode all-reduce on the flat side." | (c) | Measured on a geometry **this rig cannot reach** (cross-rig world-4 over RDMA). At the target geometry the UCX plane is not selected at all (`SGLANG_BARLINK_TRANSPORT="device"`), so the row is proven elsewhere and inert here. It is in this table because it shows the sweep can discriminate, not because it acts. |
| `SGLANG_BARLINK_UCX_AG_RING_KIB` | 32 | `environ.py:709-711`: "Measured crossover cross-rig at world 4 is ~32 KiB (task #263), so a bs=1 decode gather stays flat and a 4-token verify gather rings." | (c) | same caveat |
| `SGLANG_BARLINK_UCX_GRAIN_ELEMS` | 32768 | `environ.py:713-717`: "Co-located TP ranks enter their host passes together, so the OpenMP region's join lands on a descheduled thread and the 128 -> 256 KiB step cost milliseconds (task #263)." | (c) | same caveat; the number 32768 elements is one step below the named 256 KiB knee at 8 bytes/element, i.e. derived from the measurement rather than measured directly |
| `SGLANG_BARLINK_PIPE_CHUNK_MIB` | `None` → **calibrated at boot** | `barlink_device.py:1070-1140` `_resolve_pipe_chunk`: sweeps candidates `[1, 2, 4, 8]` MiB over the REAL `all_reduce` path with barriers, gathers per-rank times, picks the summed minimum, logs it | (b), and the right pattern | The one posten in the sweep that does not ship a value at all: the default is a measurement. Two residues: the pre-calibration seed is a **duplicated literal** `os.environ.get("SGLANG_BARLINK_PIPE_CHUNK_MIB", "4")` at `barlink_device.py:989` where `environ.py:658` declares `EnvStr(None)`; and the candidate grid `[1,2,4,8]` is itself desk-picked — an optimum at 16 MiB is unfindable. |

---

## 4. Top findings (ranked)

**#505-C-01 — the client-liveness timeout table is live by default, self-labelled
"Unmeasured", and aborts healthy streams.**
`liveness/classes.py:81-96` ships twelve silence budgets, `LLM_STREAM = 90.0` among
them, under the comment *"Unmeasured; see :data:`DEFAULT_TIMEOUT_RATIONALE` for why
each is what it is"* and *"the numbers encode an argument about the consumer, not a
measurement of the server"*. Unlike almost everything else in this sweep it is
**not behind a feature gate**: `serving_base.py:119-124` wraps every OpenAI-shaped
streaming response in `guard_generate_stream`, and on expiry
`watchdog.py:336-350` → `stream.py:169-175` calls `abort_request` on a live request.
The clock starts at watchdog construction and advances only on bytes accepted by
the transport, so a long first-token latency counts in full. On the standing recipe
(122B-A10B offloaded MoE, long contexts, single-stream serving) 90 s is not
obviously above the worst legitimate TTFT — and nobody has checked. Highest damage
of anything found: silent, user-visible, on the default path.
*Task:* `#505-C-01: measure per-class inter-byte gaps and derive the liveness timeout table, starting with LLM_STREAM=90s vs real TTFT`

**#505-C-02 — the GGUF host-RAM watermark ships at 0 (off) although the measurement
that motivated it is recorded at the flag.**
`environ.py:1798` `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB = EnvFloat(0.0)`, four lines
under *"window 3 saw memory.current move 88 -> 102 GiB inside one 15 s window"*.
The rig is swapless and the standing recipe streams GGUF weights. A guard whose
motivating measurement is written next to it and whose default is "off" is the #449
shape with the sign flipped: not a bound too high to bind, a bound never armed.
*Task:* `#505-C-02: derive and arm SGLANG_GGUF_STREAM_TRIM_SOFT_GIB from the measured stream slew, or state at the flag why off is correct`

**#505-C-03 — the measured-KV-budget safety margin (400 MiB) contradicts a
measurement recorded in its own consumer.**
`environ.py:376` ships `SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB = EnvStr("400")` as a
scalar for all ranks. Its consumer's docstring
(`model_runner_kv_cache_mixin.py:809-815`) records: *"the draft-solo host carries
the dual-prefill / draft-append serving transients, which scale with prompt length
— measured 2026-07-22: 10k prefill needs ~1 GiB, 50k ~2-3.5 GiB on the host, while
shadow ranks served everything with ~1.6 GiB"*. Every one of those measured numbers
is above 400 MiB, on every rank. The surrounding comment says *"the only ASSUMED
number left is the safety margin itself"* (`:786-788`). The feature is opt-in
(`SGLANG_MEASURED_KV_BUDGET` default False, `environ.py:373`), which is why this is
finding 3 and not finding 1 — but the moment it is switched on, the shipped default
is known-too-small by the file's own evidence, and under NEXTN speculation the
draft-solo host is exactly the rank it is too small for.
*Task:* `#505-C-03: derive the measured-KV safety margin per rank ROLE from the recorded 2026-07-22 numbers instead of one scalar 400`

**#505-C-04 — #449's own default is still the desk number the law was written
about, and its measurement arm is still not run.**
`environ.py:1638` `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB = EnvInt(2048)`.
`NOTE_449_dsv4_indexer_query_chunk.md:226` — *"the default is a ceiling picked at
desk, not a tuned value"* — and §5 of the same note is titled **"GPU measurement arm
— BOOT-PENDING, not run"**. There is no `NOTE_493` or later commit in this tree that
retunes it (`git log --grep` over 449/493 finds only the four #449 commits), and
`FEATURE_CATALOG.md` §15 says the same in its own words: *"it does not remove it,
and the speed effect is unmeasured (no GPU window taken)"*
(`docs/dev/FEATURE_CATALOG.md:1400-1402`). The posten that supplied the lesson in
`CLAUDE.md:23-29` has not itself been discharged.
Its sibling `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK = 8192` (`environ.py:1629`) is in
the same state and must be swept jointly, because the query-MiB budget is converted
to rows through a per-row byte cost that is a function of the seq chunk
(`indexer.py:341-346`).
*Task:* `#505-C-04: run NOTE_449 §5 — peak-VRAM-per-rank A/B for the DSV4 indexer query and seq chunks at 8K/32K`

**#505-C-05 — two thirds of the fork's bounding defaults cannot act in the standing
recipe, and no shipped bounding default anywhere has a value-pinning falsifier.**
71 of 106 (67 %) are behind a gate that is off in the served configuration (§1).
Of the ~35 reachable ones, the BOUND-PROVEN table has five rows, four of them
evidence class (c) and three of those measured on a geometry this rig cannot reach.
**Zero rows are evidence class (a).** The pattern is visible in
`test_retract_decode_fcfs.py:232`, which reads the default it is testing
(`max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()`) and therefore
passes for every possible value: the fork tests that its guards FIRE, never that
its numbers are RIGHT. That is a systematic instrument gap, not a per-posten
oversight, and it is the reason #449 could ship inert for weeks.
*Task:* `#505-C-05: add a value-pinning convention — every shipped bounding default gets a test that fails when it is doubled or removed`

**#505-C-06 — barlink's knobs are read through raw `os.environ` at import time, so
their declarations in `environ.py` are decorative.**
`barlink.py:52` `_CHUNK_BYTES = int(os.environ.get("SGLANG_BARLINK_CHUNK_MIB", "8"))`,
`:70` `_SLOT_BYTES = … "64"`, `barlink_host.py:127` `_BLOCKS = … "32"`,
`barlink_ucx.py:91` `_RING_BYTES = … "24"`, `:121` `_TIMEOUT_S = … "300"`,
`barlink_device.py:989` `… "4"`. Three consequences: (i) each default exists twice
and can drift silently — `barlink_device.py:989`'s `"4"` already disagrees with
`environ.py:658`'s declared `None`; (ii) they are read **once at module import**, so
`envs.X.set()` after import — which `server_args.py` does for other envs at
`:11350-11366` — has no effect, the #500-B20 shape; (iii)
`envs.SGLANG_BARLINK_*.override(...)` in a test changes nothing, so any test written
against the declared env is silently inert. This blocks the falsifiers proposed for
`SGLANG_BARLINK_PEER_TIMEOUT_S` above.
*Task:* `#505-C-06: route the barlink env reads through envs.* at call time and delete the duplicated literal defaults`

**#505-C-07 — the KV-pressure ladder's defaults exist twice with no single source
of truth.**
`kv_pressure_ladder.py:244-255` defines `DEFAULT_ASCEND_THRESHOLD = 0.85`,
`DEFAULT_ASCEND_WINDOW = 4`, `DEFAULT_DESCEND_THRESHOLD = 0.55`,
`DEFAULT_DESCEND_WINDOW = 64`, `DEFAULT_PRE_STAGE_THRESHOLD = 0.70`,
`DEFAULT_PRE_STAGE_WINDOW = 3`, `DEFAULT_ABORT_STAGE_WINDOW = 32`,
`DEFAULT_HORIZON_ROUNDS = 32`, `DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS = 512`; the
identical nine values are declared again as `ServerArgs` defaults at
`server_args.py:4874-4962` and joined by `getattr(server_args, "<name>", DEFAULT_*)`
at `kv_pressure_ladder.py:1880-1922`. They agree today. Nothing keeps them agreeing,
and the `getattr` fallback is exactly the construct that hides the disagreement when
they stop. The same duplication-with-fallback appears at
`managers/admission_limiter.py:52-55` vs `server_args.py:1089-1119`. Low severity
while the ladder is off by default (§1), which is why it is last.
*Task:* `#505-C-07: make the ServerArgs field the single source of the kv-pressure and admission defaults, or assert equality at import`

---

## 5. One upstream observation (out of scope, recorded not audited)

`SGLANG_VLM_CACHE_SIZE_MB` is declared `EnvInt(100)` (`environ.py:1465`, identical in
`upstream/main`) and read as
`int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))`
(`disaggregation/encode_server.py:330`) — a 40x drift between the declared default
and the effective one. Upstream code, upstream declaration; reported for the
upstream-PR pile, not audited further here.

---

## 6. Not opened (64 of 106) — honest gap

Enumerated by the AST walk and filtered into the bounding set, but not individually
traced to a consumer and a gate. **No verdict is implied for any of them.**

- The remaining `dual_group_lane_*`, `workbench_*`, `training_*` and
  `disaggregation_*` numeric fields beyond the ones tabled in §1: the feature gate
  was confirmed off by default, so they were placed in §1 by their group gate
  without reading each consumer individually.
- `rank_perf_loose_ctx_percent` (0.0), `rank_auto_reserve_mib` ("auto"),
  `rank_gpu_memory_mib` (None), `gdn_resident_state_slots` (None),
  `mamba_checkpoint_interval` (None), `max_running_requests_ceiling` (None): all
  reachable in the standing recipe, all with `None`/`auto`/`0` defaults that mean
  "derive it" rather than "bound it at N". They need a different question than this
  axis asks (does the DERIVATION bind), and they are the natural first cut of a
  follow-up.
- The `SGLANG_KV_CANARY_*`, `SGLANG_TEST_*` and other test-only numeric envs: named
  test-only at their site.
- The 32 fork-added `ServerArgs` fields whose help text is bounding-worded but whose
  default is a string/enum rather than a number (`rank_kv_ratio` "coupled",
  `rank_auto_reserve_mib` "auto", `swa_pool_sizing` "ratio",
  `speculative_cross_algorithm_ctx_gate` "auto", `lane_offload_profile` "latency",
  `regime_controller` "off", …): out of this axis by construction, but several are
  policy selectors that resolve to numeric bounds downstream and would belong to a
  follow-up sweep.

**PROGRESS MARKER: reached a complete enumeration of both surfaces (106 bounding
fork-added numeric defaults), 42 opened individually with consumer + gate read at
source; remaining 64 listed above with the reason, unexamined.**
