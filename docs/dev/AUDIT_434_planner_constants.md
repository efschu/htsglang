# AUDIT #434 — hardcoded, rig-fitted numeric constants in the planner/solver

Task #434 cut 2 ("generality audit"). Desk audit, no GPU.

**The directive being audited.** *"The optimum per task must ALWAYS be selected
automatically by the auto-planner — permanently, for every hardware
combination, model and quant diversity; everything transferable; nothing
tailored to this rig; general and automatic, always."*

The reference rig is 1x RTX 5090 (32607 MiB) + 2x RTX 3080 20GB (20480 MiB),
PCIe, no GPUDirect P2P. **Any number fitted on THAT rig and applied
unconditionally on foreign hardware is a generality defect.**

## Scope

Exhaustive, not sampled:

- `python/sglang/srt/uneven_perf.py` (6125 lines)
- `python/sglang/srt/planner/**/*.py` (59 modules, 66 289 lines)

## Method

1. `/tmp/a434/scan.py` walks the `ast` of all 60 files and collects every
   module-level numeric constant plus every numeric literal used inside a
   function as a threshold, coefficient, weight, exponent, margin, clamp
   bound, keyword argument or default argument. 764 candidate rows after
   filtering trivially structural literals (0, 1, ±2, small indices, unit
   conversions `1024` / `2**20` / `1e9`, percent-for-formatting, HTTP status
   codes, pure-I/O timeouts).
2. `/tmp/a434/fingerprint.py` tokenizes each file and separates
   reference-rig fingerprints (`5090`, `3080`, `RTX`, `32760`, `24564`,
   `20480`, `sm_120`, `sm_86`, TFLOPS figures, split vectors) that appear in
   **executable code** from those in **comments and docstrings**. 87
   executable-line hits, triaged individually below.
3. Every surviving constant was read in context — surrounding code, comment
   block, consumers, and whether an override seam exists and whether anything
   populates it automatically.

## Verdict vocabulary

| verdict | meaning |
| --- | --- |
| `PROBE-FED` | the value used at runtime comes from a measurement, probe or stored hardware profile; the literal is only a default or absent-marker. **OK.** Also used for a measured table that is *gated on detected hardware* and therefore never reaches a foreign rig — the source column says which. |
| `STRUCTURAL` | the value follows from the model, the algorithm, a public datasheet or hardware-independent arithmetic (bytes per element, a grid step, a convergence tolerance, a software version threshold). **OK**, with the reason given. |
| `RIG-FITTED` | fitted or measured on the reference rig and applied **unconditionally** on foreign hardware. **Forbidden by the directive.** |
| `POLICY` | a deliberate user-facing default that is a choice, not a measurement. **OK** where an override exists; the override is named. |
| `UNKNOWN-PROVENANCE` | the code and comments do not establish where the number came from. Treated as a follow-up, never guessed. |

## Summary counts

Counted over the constants that survived filtering and are individually
tabled below (grouped rows count once). One row was added after the first
pass (#437's corridor floor), which is why `POLICY` is 15 rather than 14.

| verdict | count |
| --- | --- |
| `PROBE-FED` | 11 |
| `STRUCTURAL` | 17 |
| `RIG-FITTED` | **19** |
| `POLICY` | 15 |
| `UNKNOWN-PROVENANCE` | 1 |

The 19 `RIG-FITTED` findings split into:

- **7 in `uneven_perf.py`** — the decode/prefill cost-model calibration. All
  have a refit seam (`PerfCalibration` + `SGLANG_PERF_*`), **nothing
  populates it automatically**, so a foreign rig silently runs the reference
  fit. Not owned by this agent; proposed edits at the end.
- **12 in `planner/`** — 5 fixed in this branch, 7 carried as follow-up tasks
  because they need a probe or a measurement campaign.

---

## `python/sglang/srt/uneven_perf.py` (NOT owned by this agent)

### The four calibration scalars and their seam

| constant | file:line | value | source (how the number came to exist) | verdict | action taken / follow-up |
| --- | --- | --- | --- | --- | --- |
| `_PREDICT_DECODE_GEMV_RESIDUAL` | `uneven_perf.py:340` | `0.50` | Exponent on the measured GEMV rate. Fitted on 4 measured ms-per-speculative-step points (#216 follow-up + #264 A/B), refit #265. Comment at :336 states the sample width honestly: "ONE rig (5090 + 2x 3080, PCIe, no P2P) and one checkpoint family … this particular value is not claimed to be [general]". | `RIG-FITTED` | Seam: `PerfCalibration.decode_gemv_residual_exp` / `SGLANG_PERF_DECODE_GEMV_RESIDUAL_EXP`. **Nothing populates it automatically** — grep over the tree finds the env var declared at `environ.py:611`, read at `uneven_perf.py:481`, and set only by `test/registered/unit/planner/test_prefill_calibration.py:184`. FU-434-1. |
| `_PREDICT_DECODE_BW_COMPRESSION` | `uneven_perf.py:352` | `0.45` | Fallback exponent applied to the STREAMING peak when no usable GEMV rate exists. Same 4 points, same rig. Held at `2.131**0.50 == 2.319**0.45` so the fallback reproduces *the reference rig's* achieved decode ratio. | `RIG-FITTED` | Seam `SGLANG_PERF_DECODE_PEAK_COMPRESSION_EXP`, unpopulated. Reachable on foreign hardware — see the #231 answer below. FU-434-1. |
| `_PREDICT_DECODE_NONWEIGHT_FRACTION` | `uneven_perf.py:387` | `0.35` | Share of a bs=1 decode step that is not weight streaming. Comment at :376 states it is NOT independently identified: it and the exponent trade off along a valley (BETA 0.50-0.57 × f 0.36-0.37, all rms 1.34). | `RIG-FITTED` | Seam `SGLANG_PERF_DECODE_NONWEIGHT_FRACTION`, unpopulated. Separating it needs a profiled decode step (device time split between weight-reading kernels and everything else). FU-434-2. |
| `_PREDICT_PREFILL_INVARIANT_FRACTION` | `uneven_perf.py:420` | `0.35` | Share of a prefill step invariant under the weight split. Fitted on three #216 prefill slope gains, **eager (`prefill.backend='disabled'`) prefill only**. Comment at :417 says explicitly: "Sample width: ONE rig, one checkpoint family, eager (graphless) prefill; a graph-captured prefill path has a smaller invariant share, so refit rather than reuse when that lands." | `RIG-FITTED` | Seam `SGLANG_PERF_PREFILL_INVARIANT_FRACTION`, unpopulated. The constant carries its own refit formula (`f = (r_model − r_meas) / (r_meas · (r_model − 1))`) at :414-416 but no code runs it. This is task #230's open item. FU-434-3. |
| `PerfCalibration` | `uneven_perf.py:429-527` | — | The seam itself: 4 optional fields, `from_env()`, resolved properties, and `overridden_fields()` so a refit in effect is visible in the plan log (`uneven_perf.py:5392`). | `STRUCTURAL` | The seam is well built and honest. Its defect is that it is **manual only**: four env vars a human must measure and set. Making it automatic is FU-434-1/2/3. |

### The GEMV basis guards

| constant | file:line | value | source | verdict | action taken / follow-up |
| --- | --- | --- | --- | --- | --- |
| `_PROBE_GEMV_SATURATION_FLOOR` | `uneven_perf.py:361` | `0.25` | Floor on GEMV/streaming ratio below which the GEMV is not bandwidth-bound. Comment: measured 0.92 on the 5090, 1.00 on the 3080, "so the floor is set where only a broken kernel can reach it". | `STRUCTURAL` | The number is a *validity* bound on a measurement, deliberately set far from any real value, not a fit of a physical quantity. Its purpose is to reject a pathological kernel on an unseen architecture. OK. |
| `_PROBE_GEMV_CACHE_CEILING` | `uneven_perf.py:368` | `1.05` | Ceiling: a GEMV reading faster than a pure stream is cache-served. | `STRUCTURAL` | Same reasoning: a physical impossibility bound (DRAM read cannot exceed DRAM peak), plus 5 % measurement slack. OK. |
| `_PREDICT_KNEE_COARSE_HEADROOM_UNITS` | `uneven_perf.py:252` | `2` | Unit-steps of byte-share headroom required below the knee on a coarse MLP grid. Comment at :233: "Calibrated on the M27d rig (5090 + 2x3080, membw share 51.9 %): this rejects the FP8 4,1,1 knee overshoot (picking the 3,1,1 class)". | `RIG-FITTED` | Applied unconditionally. **No seam at all** — not a `PerfCalibration` field, no env var. The comment at :237-251 already names the parameter-free replacement (sum per-sync maxima instead of max-of-sums, which "fixes the SIGN with no exponent at all") and says why it was not adopted. FU-434-4. |
| `_PREDICT_KNEE_COARSE_UNITS` | `uneven_perf.py:229` | `256` | Grid-fineness boundary above which the strict byte-share test is trusted alone. Derived from real grid sizes (FP8 dense ~136 units, GGUF K-quant ~68, AWQ 544). | `RIG-FITTED` | A model-geometry boundary, but the "~136 / ~68 / 544" figures are this checkpoint family's. No seam. FU-434-4. |
| `_PREDICT_DECODE_KNEE_TOL` | `uneven_perf.py:219` | `0.02` | Decode-knee guard tolerance, argued from M20/M22/M23 measurements on the reference rig. | `RIG-FITTED` | No seam. FU-434-4. |

### Capacity-prediction constants

| constant | file:line | value | source | verdict | action taken / follow-up |
| --- | --- | --- | --- | --- | --- |
| `_PREDICT_OVERHEAD_MIB` | `uneven_perf.py:184` | `1280` | Per-rank overhead subtracted from the byte budget. Header at :177: "calibrated against the M22/M23 boot logs of this fork's uneven-TP pipeline". | `RIG-FITTED` | Comment argues only candidate-relative and rank-relative fidelity matters, which limits but does not remove the harm: it also sets the absolute token capacity. No seam. FU-434-5. |
| `_SOLO_HOST_WORKSPACE_MIB`, `_SOLO_HOST_GRAPH_MIB_PER_REQ` | `uneven_perf.py:196-197` | `512`, `512` | Explicitly: "Calibrated on the reference rig, where the host allocated 634 MiB past its budget at `max_running_requests=2` and 2266 MiB at 4; the values below bound both with margin". | `RIG-FITTED` | Deliberately over-reserving (comment: "under-reserving here OOMs the card mid-decode while over-reserving only costs some KV") — a conservative bound, not a point estimate. Lower harm. No seam. FU-434-5. |
| `_PREDICT_MAMBA_ACT_RESERVE_MIB` | `uneven_perf.py:186` | `1024` | Mirrors the engine's `MAMBA_AUTO_ACTIVATION_RESERVE_MIB`. | `STRUCTURAL` | Mirrors an engine constant; keep in sync. OK. |
| `_PREDICT_MIN_RANK_TOKENS` | `uneven_perf.py:201` | `4096` | Below this the weighted-DCP owner rule degenerates (a rank must own ≥1 of every virtual block). | `STRUCTURAL` | An algorithmic floor of the owner rule, not a hardware measurement. OK. |
| `_PREDICT_TOKEN_UNITS` | `uneven_perf.py:203` | `64` | Token-vector granularity of the weighted-DCP optimum. | `STRUCTURAL` | Grid step of the algorithm. OK. |
| `planner_corridor_mib()` -> `registry.ledger.DEFAULT_CORRIDOR_BYTES` | `ledger.py:69`, read at `uneven_perf.py` | `400` MiB | #330's absolutely-free corridor: VRAM that must stay unallocated on every card after a boot. The planner's fundability gate prices it (#437). | `POLICY` | Added after this audit's first pass. Not a rig fit -- the same number on a 3080 and on a 5090, and a statement about how much room a boot must leave rather than a measurement of anything. One definition on the rig (`ledger.py:69`, #330), two named overrides: `--corridor-mib` on the ledger daemon and `SGLANG_PLANNER_CORRIDOR_MIB` for the planner. OK. |
| `_PREDICT_GEMM_EFF` | `uneven_perf.py:206` | `0.6` | GEMM efficiency for converting probe TFLOPS to per-token prefill time. | `POLICY` | Comment states it "cancels in candidate ratios; kept for logging" — it does not affect ranking. OK. |
| `_PREDICT_LINK_ALPHA` | `uneven_perf.py:209` | `0.25` | Exponent of the link-bandwidth penalty in the prefill score. | `UNKNOWN-PROVENANCE` | The comment states what it does, not where `0.25` came from. No fit, no measurement, no seam is cited anywhere in the file. FU-434-6. |
| `_TP_DROP_LINK_FRACTION`, `_TP_DROP_FIT_FACTOR` | `uneven_perf.py:423,426` | `0.7`, `0.85` | Thresholds for recommending a GPU be dropped from the TP group. | `POLICY` | Advisory heuristics for a *recommendation*, expressed as fractions of this rig's own best link (i.e. relative, not absolute), so they transfer in form. No seam; low harm. |
| `_PROBE_GEMM_M/K/N`, `_PROBE_GEMV_ROWS/K` | `uneven_perf.py:162-163` | `2048,5120,17408` / `131072,5120` | Probe shapes chosen as "the lm_head-class shape of m20", i.e. the reference *model*, not the rig. | `STRUCTURAL` | These define what is measured, identically on every card, and the model consumes ratios between cards. A shape is not a fitted value. Noted: they are tuned to a 5120-hidden model family — see FU-434-7. |
| `_PROBE_FP8_BLOCK` | `uneven_perf.py:174` | `(128, 128)` | The `weight_block_size` the fp8 checkpoints on this fork's rigs ship. | `STRUCTURAL` | Read off the checkpoint format, not the hardware. Would need to follow the checkpoint if a family with another block size lands — FU-434-7. |
| `_PROBE_GEMM_WARMUP`, `_PROBE_GEMM_ITERS` | `uneven_perf.py:168` | `10`, `60` | Timing budget, deliberately identical across lanes so ratios are comparable. | `STRUCTURAL` | Measurement hygiene. OK. |
| `PROFILE_VERSION` | `uneven_perf.py:91` | `3` | Cache schema version. | `STRUCTURAL` | OK. |

---

## `python/sglang/srt/planner/` — fixed in this branch

| constant | file:line | value | source | verdict | action taken / follow-up |
| --- | --- | --- | --- | --- | --- |
| `_match_calibration` rig gate | `flags.py:2215` (pre-edit) | tokvec `[33,13,18]`, reserve `"3000,2200,2200"` (fp8); `[31,15,18]`, `"1500"` (awq) | Measured on the reference rig (MATRIX_PLAN 3.3). Gate was **name-only**: one card matching `"5090"` + two matching `"3080"`. | `RIG-FITTED` (was) → `PROBE-FED` (gated) | **FIXED.** The reference rig's 3080s are the 20 GB variant; the *stock* RTX 3080 is a 10 GB card (`card_library.py:149`, `total_mib=10240`, next to `card_library.py:133` `"RTX 3080 20GB"`, `20480`). A name-only gate handed a stock 5090 + 2x 3080 10 GB rig a KV token vector and a 3000 MiB reserve solved against cards with twice the memory — a boot-time OOM presented as a calibration. Added `_CALIBRATED_RIG_TOTALS_MIB` + a ±5 % NVML-total gate (`flags.py:2215-2254`); a mismatch now falls to the existing honest "NO measured calibration for this rig/quant combination" branch at `flags.py:2632`. |
| `_SPEED_TIE_TOL_PCT` | `lever_profiles.py:303` (pre-edit) | `1.0` | "the boot-to-boot noise floor measured on this rig for the decode step (#210: 1.07 %)". Gated every speed-objective vector choice, and the user-facing reason line at the old `:546` said *"stays inside **this rig's measured** boot-to-boot noise floor"* — false on any foreign rig. | `RIG-FITTED` (was) → `PROBE-FED` | **FIXED.** Renamed to `_SPEED_TIE_TOL_PCT_FALLBACK` and added `_speed_tie_tol(crossover)` (`lever_profiles.py:298-352`), which prefers this rig's own A-vs-A spread from `crossover.CrossoverFinding.noise_floor_pct["ms_per_spec_step"]` — but only for a finding `usable_for_advice()` accepts (measured HERE, cache-bypass proven, not stale). The `(value, provenance)` pair is threaded into `_choose` and printed, so a borrowed threshold is now labelled "the REFERENCE rig's 1 % boot-to-boot noise floor … Run the crossover study to replace it". The store is already populated by the existing crossover study; no new probe needed. |
| GPU-arch tag table | `webui.py:4122-4126` (pre-edit) | `"2080"/"titan rtx"/" t4" → sm75`; `"3080"/"3090"/"a6000" → sm86` | A second, narrower copy of the arch table, listing the reference rig's cards and their neighbours. Feeds `config_tags`, which selects which measured evidence a wizard family may cite (`wizard.py:817`, `rejected.check_combination`). | `RIG-FITTED` (was) → `PROBE-FED` | **FIXED.** Now delegates to the single arch table `rig_coupling.arch_of_card_name` (`rig_coupling.py:206-217`, 10 patterns across sm75/sm80/sm86/sm89/sm90/sm120/gfx900/gfx942), and prefers an explicit `arch`/`sm_arch`/`compute_cap` field on the card when the inventory carries one. A card the table does not know keeps no arch tag — the honest answer. Previously an A40, an A5000 or a 3070 (all sm86) silently lost the sm86 evidence that applies to them. |
| `NOISE_FLOOR_PCT` | `key_solver.py:465` | `4.2` | "Benchmark noise floor of this project's harness". Measured on the reference rig (2.7-4.2 % per arm, per `REGRESSION_ANCHORS[0].tolerance_reason` at `key_solver.py:4384`). Emitted unlabelled by `solver_api.py:456`. | `RIG-FITTED` | **PARTIALLY FIXED** (provenance only). Added `NOISE_FLOOR_SOURCE` (`key_solver.py:475-482`) and emitted it beside the value (`solver_api.py:459`). Deliberately **not** substituted with the crossover store's floor: that one is a cold-boot `ms_per_spec_step` spread, a different harness from the split_probe/bench arms, and silently swapping them would change the meaning of the number. Measuring it locally needs an A-vs-A campaign on the same harness — `runner.noise_floor_from_points` (`runner.py:852`) already exists to run it. FU-434-8. |
| `DEFAULT_RESERVE_MIB` stretch | `split_probe.py:405-407` (pre-edit) | `(3000, 2700, 2700)`, extended by repeating the last element | Reference-rig runbook 4.1 reserve. `reserve_for_candidate` silently padded it to any `tp_size` by repeating `2700`, so a TP=8 rig of H100s got `[3000, 2700, 2700, 2700, …]` with no indication that ranks 3-7 were unmeasured. | `RIG-FITTED` | **PARTIALLY FIXED** (silent → named). Added `DEFAULT_RESERVE_TP_SIZE` / `DEFAULT_RESERVE_FILL_MIB` and a `stretch_note` appended to every return path of `reserve_for_candidate` (`split_probe.py:113-131`, `418-495`), naming the rank range that was filled with an unmeasured value and pointing at `--rank-auto-reserve-mib`. The fill now uses the *smallest* measured entry rather than the last, on the argument that too-small OOMs loudly and is retried while too-large silently eats KV forever. Deriving a real per-card reserve is FU-434-9. |
| stale `FASTEST_FIRST` docstring | `live_metrics.py:483-488` | — | `_annotate_cuda_indices` claimed an injected test rig "deterministically exercises the documented FASTEST_FIRST-emulation path". #397 deleted that path; `build_device_map` now raises `DeviceOrderUnresolvedError` and the bare `except` leaves the cards unbridged. | `RIG-FITTED` (doc) | **FIXED.** Docstring rewritten to state that there is no emulated order behind this any more and that unbridged (`cuda_index=None`) is the outcome and the honest one. |

## `python/sglang/srt/planner/` — RIG-FITTED, follow-up required

| constant | file:line | value | source | verdict | action taken / follow-up |
| --- | --- | --- | --- | --- | --- |
| `FIXED_PROCESS_POST_MIB` | `key_solver.py:390` | `1536.0` | "Reference values from the 28.5 GiB lesson (CUDA context + graphs + activations per extra process)" — a reference-rig incident. CUDA context size is card-, driver- and arch-dependent. | `RIG-FITTED` | No seam. Prices whether a #82-style co-location process fits. FU-434-10: measure the per-process post per card (a one-boot probe: allocate a context, read NVML delta). |
| `IDLE_FRACTION_OF_TDP` | `roofline.py:145` | `0.10` | "Calibrated loosely against the measured #146 board power… the RTX 5090's absolute watts come out ~30 % high". | `PROBE-FED` | **Already has an automatic override**: `_measured_power_by_card` (`roofline.py:557-560`) replaces it per card from `power_calibration`'s NVML board-power measurements, and the caveat list says which cards were measured. The literal is the fallback. OK. |
| `_MOE_HOTSET_HIT_RATE` | `roofline.py:196` | `0.85` | "Calibrated against our real-rig MEASURED point: 122B-A10B TP=3 on the live 5090+2x3080 (no NVLink, GPU0 on a x4 slot)". | `RIG-FITTED` | Applied unconditionally, no seam. Bounded harm: only reaches the roofline *estimate*, which carries `ROOFLINE_PROVENANCE = "planner-estimate"` (`roofline.py:112`) and is rejected by the measured store by construction. FU-434-11. |
| `_PCIE_STAGING_EFFICIENCY` | `roofline.py:178` | `0.60` | "real H2D copies under no-P2P staging land well below theoretical" — a no-P2P PCIe observation from this rig. | `RIG-FITTED` | Same bounded harm. Directly measurable by the existing `comm_suite` H2D arms. FU-434-11. |
| `_NVLINK_DISCOUNT`, `_PCIE_P2P_DISCOUNT`, `_PCIE_NOP2P_BY_CROSS_CARDS`, `_PCIE_NOP2P_MANY` | `roofline.py:161-164` | `0.90`, `0.70`, `{2:0.50, 3:0.35}`, `0.28` | Self-described as "Interconnect discount tiers (crude heuristics — shown to the user)". This rig has none of the NVLink / P2P cases, so those tiers were never measured anywhere. | `RIG-FITTED` | The no-P2P tiers are this rig's; the NVLink and P2P tiers are unmeasured guesses shipped as numbers. The measured alternative already exists — `cost_model.PairMatrix` from `comm_suite`. FU-434-11: consume the pair matrix when present instead of the tier table. |
| `_KV_DONOR_MAX_KNOCKDOWN`, `_DONOR_DIVERGENCE_NOTE` | `roofline.py:172,174` | `0.40`, `0.25` | KV-donor surcharge grading. No provenance cited. | `RIG-FITTED` | Estimate-only path. FU-434-11. |
| `_GGUF_MARLIN_DECODE_DISCOUNT` | `roofline.py:150` | `0.90` | "A documented, crude knock-down" for GGUF/marlin dequant. Quant-lane specific and card-specific (native fp8 on sm_120 vs Marlin upconvert on sm_86 behave differently). | `RIG-FITTED` | The per-lane GEMM probe in `uneven_perf.py` already measures exactly this for prefill; decode has no equivalent. FU-434-11 / FU-434-2. |
| `EFF_DECODE`, `EFF_PREFILL` | `roofline.py:129-130` | `0.75`, `0.60` | "ASSUMED, not measured… We take the middle of each band." | `POLICY` | Explicitly parameters of `estimate_roofline` (`roofline.py:981-982`) so a measured value can replace them without a code edit — the override name is the `eff_decode` / `eff_prefill` keyword. Documented assumption, not a rig fit. OK. |
| `graphmem` fit: `BASE_MIB`, `SLOPE_MIB_PER_K_PER_BS`, `TOK_ALPHA`, `DRAFT_DECODE_MIB`, `DRAFT_EXTEND_MIB`, `DRAFT_FIRST_MIB_PER_BS` | `graphmem.py:104-117` | `85.0`, `1.5e-4`, `0.44`, `85.0`, `70.0`, `45.0` | "calibrated 2026-07 against the real /tmp boot logs" of this rig. | `PROBE-FED` | **This is the pattern the rest of the tree should follow.** `graphmem` is a three-layer design (`graphmem.py:17-43`): a measured-anchor store keyed on config shape, self-populating from every boot log on disk with no explicit calibration step (`graphmem.py:378-382`), and the fit is only the fallback when no anchor matches, shipped with a stated `ERROR_BAND_PCT = 30`. The literals are defaults, not the values used on a rig that has booted. OK. |
| `wizard` `ANCHOR_*` block | `wizard.py:169-196` | `0.257`, `0.604`, `6.54`, `3.22`, `3.52`, `3.18`, `0.136`, `2385.0`, `1.93`, `0.105`, `10850/25400` | "Every figure below was measured on THIS rig pair in the #212 satellite study (Qwen3.5-2B fp16, rig 1 = RTX 5090 decode, rig 2 = RTX 2080 Ti prefill, 40G RoCE)." | `PROBE-FED` | Consumed exclusively through `WizardContext.rate(key, default)` (`wizard.py:876-886`), whose last rung is the module anchor and whose `rate_source()` reports which rung answered. A rig that measured its own line is priced on it. The anchors are labelled at every use via `ANCHOR_STUDY`. OK — but note the *default* is still the reference pair, so FU-434-12: make the rate report surface "borrowed" as loudly as `lever_profiles` now does. |
| `crossover.REFERENCE_FINDING` | `crossover.py:559-606` | full measured finding for `5090 + 2x3080` | The reference rig's crossover study, verbatim. | `STRUCTURAL` | **The gold-standard pattern in this tree.** Provenance `MEASURED_ELSEWHERE`, so `usable_for_advice()` (`crossover.py:368-378`) returns False for it by construction — it can never select a vector on any rig. Comment at :556: "It is here so a reader can see the size and the shape of the effect and so the arithmetic has a fixture. It is NOT this machine's crossover." OK. |
| `spread.MEASURED_PAIR_E` | `spread.py:64-67` | `{0: 1.440, 1: 1.130}` | "Card equivalents measured for TWO lanes on one card… (DESIGN_121 §11.5, slice C, 27B-Q3 GGUF TP=3 uneven, lane on the 5090)". | `RIG-FITTED` | Mitigated but not removed: the comment argues "what the decider uses is their ORDER, and the order is what was measured", and `SM_PAIR_UPPER_BOUND` refuses to report the unmeasured two-saturating-lane case as an expected E. The values are still numeric inputs on foreign hardware. FU-434-13. |
| `key_solver.COLLECTIVE_EFFICIENCY` | `key_solver.py:426` | `1.0` | Ring efficiency for the pair-matrix collective term. | `POLICY` | `1.0` is the neutral / absent marker (no efficiency loss assumed), and the constant carries its own refit recipe in the comment. A neutral default is not a rig fit. OK. |
| `bench_factors.PREFILL_RATIO_THRESHOLD` / `DECODE_RATIO_THRESHOLD` | `bench_factors.py:954,958` | `8.0`, `1.5` | Prompt-to-output ratios above/below which a working point is proposed. "Deliberately high: the prefill lever is the weaker-calibrated of the two terms". | `POLICY` | A deliberate conservatism choice about *when to volunteer advice*, not a measurement. OK. |
| `bench_suite.TOKENS_PER_FILLER_SCALE` | `bench_suite.py:135` | `65` | "verify-stress.sh fallback for Qwen tokenizers; the script's live calibration probe is not replicated here". | `RIG-FITTED` (model-family) | Not rig-fitted but **model-family-fitted**, which the directive covers equally ("model and quant diversity"). The comment names the fix: the live calibration probe already exists in the shell script. FU-434-14. |
| `key_solver.REGRESSION_ANCHORS`, `ADDITIVE_ANCHOR`, and the `32607 − 3000, 20480 − 2700, …` budget literals | `key_solver.py:4370-4460`, `4492-4547`, `4711-4715` | reference-rig measurements | Measured points the cost model must reproduce. | `STRUCTURAL` | These are **regression fixtures**, not decision inputs. Consumed only by `check_regressions` / `check_additive_regression`, which are self-check paths reachable from the tests and from `solver_api.py:435,444` — never from `feasibility.plan` or the solver. Docstring at `key_solver.py:4701`: "because a calibration nobody can see is a calibration nobody can refute". OK. |
| `split_probe._ROWS_264`, `_TS_264` | `split_probe.py:652-740` | reference-rig measured rows | Recorded #264 A/B result rows. | `STRUCTURAL` | Stored measurement records with a timestamp, not thresholds. OK. |
| `crossover.ConcentrationPoint` values in `REFERENCE_FINDING` | `crossover.py:574-591` | `0.0473`, `0.648`, `7.2`, `6.0`, … | see `REFERENCE_FINDING` above | `STRUCTURAL` | Gated by `usable_for_advice()`. OK. |
| `wizard_offload.EVIDENCE` | `wizard_offload.py:92-118` | `0.25` fractions, `"one RTX 3080, 20.00 GiB"`, `20480` | Recorded offload evidence records with their hardware named in the record itself. | `STRUCTURAL` | Evidence records, displayed with their hardware. Not decision inputs. OK. |

## `python/sglang/srt/planner/` — structural / policy, no action

| constant | file:line | value | source | verdict | note |
| --- | --- | --- | --- | --- | --- |
| `card_library.SEED_CARDS` peaks | `card_library.py:126-173` | e.g. `peak_membw_gbs=1792.0`, `peak_gemm_tflops_fp16=419.0` | Public datasheet nameplate figures, dense tensor peak, no 2:4 sparsity. | `STRUCTURAL` | **Not a fallback for unknown cards.** `CardLibrary.get` raises `KeyError` naming the known cards (`card_library.py:196-202`); `compose_rig` only ever resolves names the caller asked for. The library is used solely for *composed hypothetical* rigs, marked `source="library-composition"` with `free_mib=None` on every card so the explorer labels the result an estimate (`card_library.py:41-48`). Measured probe fields (`gemm_tflops`/`membw_gbs`) win over nameplate when present (`card_library.py:82-86`). Nothing hands reference-rig numbers to an unknown card. |
| `rig_coupling._ARCH_PATTERNS` | `rig_coupling.py:206-217` | 10 name→arch regexes | "Only the walls this project has actually hit are listed: a table of every GPU ever made would be a maintenance burden whose wrong rows would read as facts." | `STRUCTURAL` | Keyed lookup with an explicit unknown state (`arch=None` → every gate row reports ABSENT), and `CardFacts.from_json` prefers a real `arch`/`compute_cap` field over the inference. Correct pattern. |
| `rig_coupling.COLOCATION_NCCL_MIN` | `rig_coupling.py:120` | `(2, 30)` | NCCL version required for multiple ranks per GPU. | `STRUCTURAL` | A software version threshold, deliberately "encoded as the threshold rather than as the installed version so a rig with a newer NCCL passes without a code change". |
| `mrr_balance.MAMBA_RATIO`, `MAMBA_SAFETY_MARGIN`, `PREDICTOR_TARGET_CLAMP` | `mrr_balance.py:124-131` | `5`, `1.25`, `48` | Mirror `PerfCostModel._mamba_pool_bytes` and the engine's `MAMBA_CACHE_*` ratios. | `STRUCTURAL` | Engine-mirroring constants; comment says "Keep the three in sync". |
| `roofline._PCIE_GBS_PER_LANE`, `live_metrics._PCIE_LANE_GBS` | `roofline.py:198`, `live_metrics.py:222` | PCIe gen → GB/s per lane | PCIe specification figures. | `STRUCTURAL` | Standards numbers. |
| `roofline._PCIE_DEFAULT_GEN/_WIDTH` | `roofline.py:199` | `4`, `8` | "when the topology is unknown" | `POLICY` | A conservative default for an unreadable topology, not a rig measurement. The live path reads the negotiated width. |
| `roofline._VOCAB_STREAM_FRACTION` | `roofline.py:158` | `0.5` | "Only the lm_head half of the `vocab` family (embed + lm_head) is a per-token matmul; the embedding table is a gather". | `STRUCTURAL` | Follows from the model structure. |
| `comm_suite._arm_noise_floor` gate | `comm_suite.py:850,854` | `15.0` | A-vs-A spread above which the box is called busy. | `POLICY` | The floor itself is **measured** by the arm; 15 % is a "was the machine quiet" gate, deliberately wide. |
| `comparison.CONDITION_TOLERANCE` | `comparison.py:231` | `0.02` | "Accept length is measured, so it is never bit-identical between two runs; a percent of drift is not a different experiment, ten is." | `POLICY` | A comparability choice. |
| `lever_profiles._CAPACITY_TIE_TOL` | `lever_profiles.py:355` | `1e-9` | "Capacity is deterministic arithmetic rather than a timing, so there is no noise floor to hide behind: only an exact tie falls back". | `STRUCTURAL` | Float epsilon on exact arithmetic. |
| `lever_profiles` `loose_ctx_percent` | `lever_profiles.py:187,201` | `10.0`, `15.0` | Axis positions of the four profiles. | `POLICY` | User-facing stops on the context-for-speed slider; the expert view keeps every individual knob. |
| `key_solver.MAX_ROLE_SEARCH_RANKS`, `spread.MAX_PLACEMENTS`, `key_solver` `iterations`/`max_moves`/`segment_steps`/`grid_steps` | `key_solver.py:538,953,1059,2464-2465`; `spread.py:76` | `6`, `20000`, `200`, `64`, `40`, `8` | Search budgets, each with a stated cost argument (e.g. "3 roles ^ 6 ranks = 729 solves, each a bisection — milliseconds"). | `STRUCTURAL` | Compute budgets, hardware-independent. |
| `mrr_balance.DEFAULT_TARGET_CONTEXTS`, `lever_profiles.DEFAULT_SESSION_CONTEXT`, `kv_ladder_table.DEFAULT_DEPTH_BINS`, `graphmem._DEFAULT_DECODE_BS`, `webui._CONCURRENCY_LADDER` | various | context / batch ladders | Presentation grids. | `POLICY` | Report axes, overridable by the caller. |
| `kv_ladder_table.RigModelProfile.budget_fraction` | `kv_ladder_table.py:261` | `0.9` | Default budget fraction. | `POLICY` | A dataclass default the caller sets. |
| `energy` / `hicache_savings` / `jtok_counter` `JOULES_PER_KWH`, `3.6` | various | `3.6e6`, `3.6` | Unit conversions. | `STRUCTURAL` | Physics. |
| `energy` `price_ct_per_kwh=30.0` | `energy.py:1563,1683,1804` | `30.0` | Electricity price default. | `POLICY` | CLI-overridable (`--price-ct-per-kwh`). |
| `power_calibration._DEFAULT_*` windows | `power_calibration.py:111-114` | `50.0` ms, `2.0`/`1.0`/`4.0` s | NVML sampling windows. | `POLICY` | Measurement-hygiene defaults, all keyword-overridable. |
| `rig_artifact._RAM_CLASSES`, `RIG_FP_VERSION`, `MAX_ARTIFACT_BYTES` | `rig_artifact.py:123-257` | classes, `1`, `100_000` | Anonymization buckets and a size cap. | `STRUCTURAL` | Privacy/serialization mechanics. |
| `crossover.STALE_AFTER_S` | `crossover.py:132` | 60 days | Freshness limit on a finding. | `POLICY` | Staleness policy, reported with the age. |
| `self_update.RESTART_EXIT_CODE`, `DATA_SCHEMA_VERSION` | `self_update.py:105-108` | `43`, `1` | Protocol constants. | `STRUCTURAL` | OK. |
| `split_probe` timing/`DECODE_TOKENS_PER_SECOND`/`DEFAULT_*` | `split_probe.py:121-147` | `20000`, `25`, `60`, timeouts | Harness shape: "the window only has to land in the 20-30 s band". | `POLICY` | Harness parameters, request-overridable. |
| `scenarios`/`crossover` `est_runtime_min` | various | `10`…`270` | Estimated study durations shown to the user. | `POLICY` | Display estimates. |
| `graphmem.ERROR_BAND_PCT` | `graphmem.py:82` | `30` | "Stated heuristic error band — verified against the calibration logs in the unit tests; shown verbatim in the UI tooltip." | `POLICY` | An honesty label on the fallback, shown to the reader. |

---

## Named question 1 — did #231's probe FULLY replace the fitted exponent?

**No. #231 replaced the fitted *divisor*, not the fitted *exponent*, and the
fallback to the second fitted exponent remains reachable on foreign
hardware.** There is, however, **no `max()` left** — that specific defect is
gone.

The whole decision lives in `PerfCostModel.decode_bw_basis`
(`uneven_perf.py:4414-4469`) and `effective_decode_bw`
(`uneven_perf.py:4471-4483`):

```python
peak = [max(float(b), 1e-9) for b in membw_gbs]
beta_peak = self.calibration.peak_compression          # 0.45
if gemv_gbs is None or len(gemv_gbs) != len(peak):
    return (peak, beta_peak, "streaming peak (no GEMV rate in the hardware "
            "profile; re-probe with SGLANG_PERF_REPROBE=1 to measure it)")
gemv = [float(g or 0.0) for g in gemv_gbs]
for r, (g, p) in enumerate(zip(gemv, peak)):
    frac = g / p
    if frac < _PROBE_GEMV_SATURATION_FLOOR:            # 0.25
        return (peak, beta_peak, f"streaming peak (rank {r}: ...)")
    if frac > _PROBE_GEMV_CACHE_CEILING:               # 1.05
        return (peak, beta_peak, f"streaming peak (rank {r}: ...)")
return gemv, self.calibration.gemv_residual, "measured decode GEMV rate"
```

and

```python
rates, beta, _ = self.decode_bw_basis(membw_gbs, gemv_gbs)
return [max(float(b), 1e-9) ** beta for b in rates]
```

Three findings, precisely:

1. **The `max()` is gone.** The docstring at `uneven_perf.py:4426-4427` states
   it: "Falling back to the streaming peak is a real loss of information, so
   every reason to fall back is a named, reported condition — never a silent
   `max()`." Every fallback returns a `basis` string that names the rank, the
   two rates and the reason, and the caller prints it into the plan log
   (`uneven_perf.py:5377-5390`). The only `max()` remaining in
   `effective_decode_bw` is the `max(float(b), 1e-9)` divide-by-zero clamp.
2. **The probe path still applies a reference-rig-fitted exponent.** The
   success return is `gemv, self.calibration.gemv_residual` — i.e.
   `bw_eff = gemv_gbs ** 0.50`, with `0.50` fitted on four points on the
   reference rig. The probe supplied the *divisor*; the exponent is still the
   fit. `uneven_perf.py:264-272` says this outright: the measurement is "a
   fifth of the way (22 % in log terms) to the implied 1.46", and "the
   residual exponent below, for the rest".
3. **The fallback to `0.45` is reachable on foreign hardware**, by three
   routes, two of which are precisely foreign-architecture conditions:
   - profile predates `PROFILE_VERSION 2` (`uneven_perf.py:4440`) — a rig
     upgrading from an older release before its profile is re-probed;
   - `gemv/peak < 0.25` (`:4450`) — a card whose GEMV kernel is not
     bandwidth-bound, which is exactly "an architecture we have not seen"
     (`:355-356`);
   - `gemv/peak > 1.05` (`:4461`) — a cache-resident read; the comment at
     `:364-366` notes a 64 MiB matrix already reads at 2.2 TB/s on a 5090's
     96 MiB L2, so a future card with a larger L2, or a smaller probe shape,
     trips this.

   And per `uneven_perf.py:347-351` the two exponents are pinned to each other
   *at the reference rig's ratio* (`2.131**0.50 == 2.319**0.45 == 1.46`), so
   the fallback deliberately reproduces the reference rig's decode advantage
   ratio rather than the local one.

**Net:** #231 moved one of two fitted quantities onto the probe. Task #230's
"refit the exponent on foreign hardware" is still open, and #231's precedent
("a GEMV probe beats a fitted exponent") has not yet been applied to the
exponent itself. Closing it needs a **decode-shaped quantized probe** — the
file names this and explains why the current dense GEMV cannot substitute
(`uneven_perf.py:314-334`): a bs=1 decode step reads *quantized* weights
through dequantising kernels (native fp8 on sm_120, Marlin upconvert on
sm_86, MMVQ for GGUF) at M=1, and "a decode-shaped quantized probe plus a
refit is the way to close it, and it is its own campaign". That is FU-434-1.

## Named question 2 — is #397 (FASTEST_FIRST → named error) complete?

**Complete inside `planner/`. One residue outside it, in
`python/sglang/srt/utils/common.py`, which is bounded and documented but is
still a capability-order emulation that can be wrong on foreign hardware.**

Verified clean:

- `planner/device_map.py` — the emulation is gone. `build_device_map`
  (`:192-222`) resolves through the #331 identity map and calls
  `require_cuda_ordinals`, which raises; `device_map` (`:229-246`) catches,
  logs once and caches the **empty** map. Docstring at `:38-46`: "It was
  labelled 'heuristic' and callers were asked to surface that label, but a
  label does not make an ordering right… There is now no fallback."
  `IDENTITY_MAP_SOURCE` is the only non-`None` source; `"torch"` and
  `"heuristic"` are gone (`:71-73`).
- `planner/flags.py:1760-1790` — `_cuda_index_at` has three resolution rungs
  (explicit `cuda_index`, UUID against the identity map, and the *declared*
  order for an inventory with no card identity at all, i.e. offline
  `--gpu NAME:MIB` specs). A card that **carries** identity and still cannot
  be placed raises `DeviceOrderUnresolvedError`. Callers omit the pin instead
  of guessing it.
- `planner/flags.py:2216-2254` — `_match_calibration` is name-multiset gated,
  order-insensitive, and never remaps rank-space vectors onto an unmeasured
  rig. (Its *other* defect — the missing VRAM check — is fixed above.)
- `planner/flags.py:2977-2990` — `ProfileStore` resolves saved `--rank-gpu-id`
  ordinals to NVML UUIDs on save and back on load, rewriting when they moved
  and raising a named error when a card is gone.
- `planner/hardware.py:62-80` — `GpuDescriptor.cuda_index` is documented as
  "None when it cannot be resolved… never a guessed value, #397", and
  `HardwareSpec.cuda_index_source` is `"identity-map"` or `None`, "never a
  guess".
- `planner/live_metrics.py:482` — behaviour was already correct; only the
  docstring still described the removed path. Fixed in this branch.
- `server_args.py:554-583` — `_resolve_rank_gpu_cards` goes through the
  identity map and states that "an ordinal the map cannot place is an error,
  never a fallback to the NVML index of the same number", citing the #349
  sweep-3 arm L OOM as the field incident.

**The residue** — `python/sglang/srt/utils/common.py:329-341`,
`_nvml_devices_in_cuda_order`:

```python
pci_order = os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST") == "PCI_BUS_ID"
if pci_order:
    devices.sort(key=lambda d: d[3])
else:
    devices.sort(key=lambda d: d[0], reverse=True)   # by compute capability
```

This still emulates FASTEST_FIRST by sorting on compute capability. The two
consumers are honest about it and neither hands out a guessed *identity*:

- `min_visible_cuda_capability_no_init` (`common.py:395-421`) returns the
  floor over **every** NVML card for an unresolvable index on a mixed rig,
  arguing that a bound over a superset can only be lower than the true floor,
  so it arms a safe kernel variant. That is a bound, not an identity — fine.
- `_nvml_cuda_device0` (`common.py:424-471`) **returns `None`** for an index
  past 0 on a mixed rig: "FASTEST_FIRST specifies only position 0; an index
  into the unspecified remainder of a mixed rig cannot be emulated." That is
  the named refusal — fine.

What remains wrong is **position 0 itself on foreign hardware**. The docstring
at `common.py:429-433` admits it: "Exotic mixes where a lower-capability card
is faster would be mis-ordered there". This is not exotic in practice — an
A100 (sm_80) alongside an RTX 4060 (sm_89) puts the 4060 at position 0 by
capability, while CUDA would pick the A100. The reference rig hides this
because there the fastest card (5090, sm_120) *is* the highest capability.
That is a reference-rig-shaped assumption. It is outside `planner/` and
outside my write scope; proposed edit below.

**Verdict: #397 is complete for the planner's device-identity surface. It is
not complete for `utils/common.py`'s capability-ordered position-0
emulation.** No reachable path in `planner/` silently emulates a device order.

## Follow-up tasks

| id | task | why it is not a cheap fix |
| --- | --- | --- |
| FU-434-1 | Decode-shaped **quantized** GEMV probe (per lane, per checkpoint) to replace `_PREDICT_DECODE_GEMV_RESIDUAL` / `_PREDICT_DECODE_BW_COMPRESSION`. | A new probe plus a refit campaign; `uneven_perf.py:331-334` names it as its own campaign. |
| FU-434-2 | Profile one decode step and split device time between weight-reading kernels and everything else, to identify `_PREDICT_DECODE_NONWEIGHT_FRACTION` independently of the exponent. | A direct measurement the tree has no instrument for yet. The two scalars trade off along a valley and cannot be separated by more of the same A/B points. |
| FU-434-3 | Auto-refit `_PREDICT_PREFILL_INVARIANT_FRACTION` on first boot: the formula is already at `uneven_perf.py:414-416`, and the inputs (base split + one concentration vector, cached-token 0 proven) are a two-boot campaign. Task #230's open item. | Two cold boots plus cache-bypass proof; also needs a graph-captured-prefill variant. |
| FU-434-4 | Replace the decode-knee guard's hand-set margins (`_PREDICT_KNEE_COARSE_HEADROOM_UNITS`, `_PREDICT_KNEE_COARSE_UNITS`, `_PREDICT_DECODE_KNEE_TOL`) with the parameter-free per-sync-maximum formulation already described at `uneven_perf.py:237-251`. | The comment says why it was deferred: "it is a second change to the same model, it needs its own campaign (per-block testing rejects 3,1,1 too, which does cost +6.0 %)". |
| FU-434-5 | Measure per-rank fixed overhead (`_PREDICT_OVERHEAD_MIB`, `_SOLO_HOST_*`) from the boot logs the way `graphmem` already does for graph memory. | Needs a boot-log parser for the allocator/workspace lines and an anchor store keyed by card + config shape. |
| FU-434-6 | Establish provenance for `_PREDICT_LINK_ALPHA = 0.25` or refit it. | Nothing in the tree records where it came from. `UNKNOWN-PROVENANCE`. |
| FU-434-7 | Make the probe shapes (`_PROBE_GEMM_*`, `_PROBE_GEMV_*`, `_PROBE_FP8_BLOCK`) follow the *checkpoint being planned* rather than the m20 reference model. | Changes what the cached hardware profile means; needs a cache-key change and a migration. |
| FU-434-8 | Measure this rig's harness noise floor with an A-vs-A run and feed it to `key_solver.NOISE_FLOOR_PCT`. `runner.noise_floor_from_points` (`runner.py:852`) and `RunPolicy.noise_floor_boots` already exist; what is missing is persisting the result and reading it back. | A measurement campaign (≥3 boots) plus a store. Provenance labelling landed in this branch as the interim step. |
| FU-434-9 | Derive `split_probe.DEFAULT_RESERVE_MIB` per card from the checkpoint and the card rather than shipping three measured numbers. | Needs the FU-434-5 overhead model. Interim: the stretch is now named, not silent. |
| FU-434-10 | Measure the per-extra-process CUDA-context post (`key_solver.FIXED_PROCESS_POST_MIB`) per card with a one-boot NVML-delta probe. | New probe, but a small one — a good next candidate. |
| FU-434-11 | Feed `roofline.py`'s interconnect and MoE-offload knobs (`_MOE_HOTSET_HIT_RATE`, `_PCIE_STAGING_EFFICIENCY`, `_NVLINK_DISCOUNT`, `_PCIE_P2P_DISCOUNT`, `_PCIE_NOP2P_*`, `_KV_DONOR_*`, `_GGUF_MARLIN_DECODE_DISCOUNT`) from the measured `cost_model.PairMatrix` / `comm_suite` results when they exist. | Several knobs, each with its own measurement; the pair-matrix consumption alone is a real refactor of `_interconnect_discount`. Bounded harm meanwhile: all of it is `ROOFLINE_PROVENANCE = "planner-estimate"`. |
| FU-434-12 | Make `wizard`'s rate report surface "this figure is the #212 reference pair's, not yours" as loudly as `lever_profiles` now does for the tie tolerance. | Mostly presentation, but touches every family's caveat rendering. |
| FU-434-13 | Measure `spread.MEASURED_PAIR_E` on the rig, or make the decider consume only the *order* it claims to use. | Needs a co-location lane measurement; the two-saturating-lane cell was never measured anywhere. |
| FU-434-14 | Replace `bench_suite.TOKENS_PER_FILLER_SCALE = 65` with the live tokenizer calibration probe the shell harness already runs. | Model-family fitted (Qwen tokenizers). Cheap-ish, but it belongs to the bench harness owner. |
| FU-434-15 | Fix `utils/common.py`'s capability-sort emulation of CUDA position 0 (see the proposed edit below). | Outside `planner/`; needs the owner of the device-order surface. |
| FU-434-16 | Derive `split_probe.LADDER` from the rig's own solver candidates (`uneven_perf._mlp_candidates` over the measured profile) instead of shipping a fixed set of THREE-rank vectors. Today every concentrated row is unmeasurable on a rig that is not TP=3 -- the vector length does not match `--tp-size`, so the boot the "not measured" reason invites is rejected at parse time -- and the candidate set does not follow the profile at all. | The dashboard consumes the table; deriving the ladder needs a profile at render time and a decision about what to show when there is none. Interim (this branch): the limitation is stated at the constant and callers on other rank counts are pointed at the existing `ladder=` override. |

---

### Proposed edits in files owned by the orchestrator

**Status: P1, P3 and P4 are APPLIED in this branch; P2 is carried as FU-434-4's
seam half.** P1 landed as `PerfCalibration.borrowed_fields()` plus a
`calibration BORROWED (not measured here)` line in the plan log, worded to say
what the borrowed scalars do and do not decide (they set how much a candidate
is predicted to gain, not which cards are fast -- that comes from the profile).
P3 landed as the comment on `utils/common.py`'s capability sort; the structural
fix stays FU-434-15. P4 landed as
`test/registered/unit/planner/test_borrowed_calibration_434.py`, which covers
both fixed gates and the new calibration reporting, with the shipped
`crossover.REFERENCE_FINDING` pinned as the pattern the rest of the file argues
for; 10 of its 12 tests fail on the pre-#434 tree (the two that pass are the
anti-vacuity halves, which is what makes them anti-vacuity halves).

One finding is added to the table below by the orchestrator rather than by this
audit: `split_probe.LADDER` (FU-434-16), a fixed set of THREE-rank candidate
vectors that neither follows the profile nor matches any rig of another rank
count. Documented at the constant in this branch, derivation deferred.


These are outside this agent's write scope (`python/sglang/srt/uneven_perf.py`,
`python/sglang/srt/server_args.py`, `python/sglang/srt/utils/common.py`,
`test/**`). Each is written as an exact anchor + replacement.

#### P1 — `python/sglang/srt/uneven_perf.py`: make the calibration seam self-describing when unrefit

The `SGLANG_PERF_*` seam exists and is well documented, but a foreign rig that
sets none of the four vars gets the reference fit with **no statement in the
plan log that it is borrowed**. `overridden_fields()` reports the *overridden*
case (`uneven_perf.py:5392-5401`); the *un*overridden case is silent. Invert
it so the borrowed case is the loud one.

Anchor (`uneven_perf.py:519-527`):

```python
    def overridden_fields(self) -> List[str]:
        """Names of the fields explicitly set (for the plan log: a refit in
        effect must be visible, or a foreign value silently poses as the
        reference fit)."""
        return [
            f.name
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        ]
```

Replacement:

```python
    def overridden_fields(self) -> List[str]:
        """Names of the fields explicitly set (for the plan log: a refit in
        effect must be visible, or a foreign value silently poses as the
        reference fit)."""
        return [
            f.name
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        ]

    def borrowed_fields(self) -> List[str]:
        """Names of the fields still at the REFERENCE-RIG fit (#434).

        The mirror of :meth:`overridden_fields`, and the more important half
        on any machine that is not the reference rig: these four scalars were
        fitted on one rig, one checkpoint family and one prefill mode, and a
        plan that does not say so presents a hypothesis as a measurement.
        """
        return [
            f.name
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is None
        ]
```

and at `uneven_perf.py:5392`, anchor:

```python
    overridden = model.calibration.overridden_fields()
    if overridden:
```

Replacement:

```python
    borrowed = model.calibration.borrowed_fields()
    if borrowed:
        lines.append(
            "calibration BORROWED from the reference rig (RTX 5090 + 2x RTX "
            "3080, PCIe without P2P, Qwen3.6-27B-FP8, uneven TP=3): "
            + ", ".join(borrowed)
            + " -- these are fitted scalars, not measurements of this "
            "machine. Refit them via the matching SGLANG_PERF_* env vars "
            "(recipes at each constant's definition in uneven_perf.py)."
        )
    overridden = model.calibration.overridden_fields()
    if overridden:
```

#### P2 — `python/sglang/srt/uneven_perf.py`: give the knee-guard margins a seam

`_PREDICT_KNEE_COARSE_HEADROOM_UNITS`, `_PREDICT_KNEE_COARSE_UNITS` and
`_PREDICT_DECODE_KNEE_TOL` are reference-rig fits with **no** override at all,
unlike the four in `PerfCalibration`. Adding them as three more optional
`PerfCalibration` fields plus three `SGLANG_PERF_*` vars in `environ.py` is a
mechanical change and would make FU-434-4 refittable without a code edit.

Anchor (`uneven_perf.py:469-472`):

```python
    decode_gemv_residual_exp: Optional[float] = None
    decode_peak_compression_exp: Optional[float] = None
    decode_nonweight_fraction: Optional[float] = None
    prefill_invariant_fraction: Optional[float] = None
```

Replacement:

```python
    decode_gemv_residual_exp: Optional[float] = None
    decode_peak_compression_exp: Optional[float] = None
    decode_nonweight_fraction: Optional[float] = None
    prefill_invariant_fraction: Optional[float] = None
    #: Decode-knee guard margins (#434). Fitted on the M20/M22/M23/M27d runs
    #: of the reference rig and, until this seam, not overridable at all.
    decode_knee_tol: Optional[float] = None
    knee_coarse_units: Optional[int] = None
    knee_coarse_headroom_units: Optional[int] = None
```

with matching `from_env` entries, resolved properties following the existing
`gemv_residual` pattern, and `EnvFloat(None)` / `EnvInt(None)` declarations
for `SGLANG_PERF_DECODE_KNEE_TOL`, `SGLANG_PERF_KNEE_COARSE_UNITS` and
`SGLANG_PERF_KNEE_COARSE_HEADROOM_UNITS` after `environ.py:614`.

#### P3 — `python/sglang/srt/utils/common.py`: stop emulating CUDA position 0 by compute capability

`_nvml_devices_in_cuda_order` sorts by compute capability descending to
emulate FASTEST_FIRST. On the reference rig the fastest card is also the
highest-capability one, which is why this has never been wrong here. It is
wrong on, e.g., an A100 (sm_80) + RTX 4060 (sm_89) rig, where it names the
4060 as CUDA device 0.

Anchor (`common.py:336-341`):

```python
    pci_order = os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST") == "PCI_BUS_ID"
    if pci_order:
        devices.sort(key=lambda d: d[3])
    else:
        devices.sort(key=lambda d: d[0], reverse=True)
    return devices, pci_order
```

Replacement:

```python
    pci_order = os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST") == "PCI_BUS_ID"
    if pci_order:
        devices.sort(key=lambda d: d[3])
    else:
        # FASTEST_FIRST is not a capability sort. CUDA ranks by device
        # performance, and capability is only a proxy for it -- a proxy that
        # holds on the reference rig (the 5090 is both the fastest card and
        # the highest sm_) and fails wherever the two disagree, e.g. an A100
        # (sm_80) beside an RTX 4060 (sm_89) (#434). Position 0 is therefore
        # a GUESS here whenever the visible cards do not share a capability;
        # the flag below already tells every caller not to trust positions
        # past 0, and ``_nvml_cuda_device0`` refuses outright on a mixed rig.
        devices.sort(key=lambda d: d[0], reverse=True)
    return devices, pci_order
```

The stronger fix — resolving position 0 through the #331 identity map the way
`planner/device_map.py` does, and returning `None` when the visible cards
disagree on capability — is FU-434-15; it changes what
`min_visible_cuda_capability_no_init` can answer on a mixed rig and needs the
owner of that surface.

#### P4 — test coverage for the two fixed gates

Falsifier-first, per `CLAUDE.md`. Both belong in
`test/registered/unit/planner/`:

- `test_flags.py`: a rig of `["NVIDIA GeForce RTX 5090" @ 32607,
  "NVIDIA GeForce RTX 3080" @ 10240, "NVIDIA GeForce RTX 3080" @ 10240]` with
  `quant="fp8"` must get `_match_calibration(...) is None` (fails before the
  fix, passes after), while the 20480 variant still gets the calibration.
- `test_lever_profiles.py`: `_speed_tie_tol(None)` returns
  `(1.0, <text containing "REFERENCE rig">)`; `_speed_tie_tol(finding)` with a
  `MEASURED_HERE`, cache-bypass-proven, fresh finding carrying
  `noise_floor_pct={"ms_per_spec_step": (0.4, 2.6)}` returns
  `(2.6, <text containing "this rig's measured">)`; and
  `crossover.REFERENCE_FINDING` (provenance `MEASURED_ELSEWHERE`) must NOT be
  accepted.

---

## Validation

- `ruff check` on all 7 changed files: clean except one pre-existing
  `F401` (`live_metrics.py:80`, `LiveSnapshot` unused) that is present
  unchanged in `HEAD` and is not on a line this branch touched.
- `codespell` on all 7 changed files: one pre-existing hit
  (`webui.py:1786`, a misspelling of "unparsable" in a string literal),
  unchanged in `HEAD`.
- `CUDA_VISIBLE_DEVICES=99 … pytest -q test/registered/unit/planner/` —
  **2050 passed, 1 skipped, 184 subtests passed** in 76 s.
