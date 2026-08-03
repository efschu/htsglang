# Audit #505 part A1

Desk audit, nothing executed, no GPU. Worktree `/spinning/wt-505-silent`,
branch `docs/silent-wrongness-505`, base `d6534052231276171daf3a844476812ec702ccf3`.
Method copied from `docs/dev/AUDIT_500_mechanism_reach.md` §§1-6: every row
cites `file:line`, quotes the operative line verbatim, and a row that rests on a
comment or docstring rather than executed code says so.

## Axis A1 — warning-instead-of-error: loader / weights / spec / MoE-offload

The defect class, from the occasion: a site DETECTS an anomaly, logs a warning,
and CONTINUES, where the resulting state is silently WRONG rather than merely
degraded. The bar for DANGEROUS is a concrete answer to *what exactly would be
wrong, and through which observable would a user not notice it*. Rows that
cannot answer that are BENIGN, and are marked so without argument.

The tree already knows this class well. Three sites in scope are the CORRECT
shape and serve as the fix templates:

- `models/deepseek_v4_dspark.py:868` `_assert_required_params_loaded` — the
  #496-(b) completeness check, raising by name on unwritten parameters.
- `models/deepseek_v4.py:3074-3076` `raise KeyError(_unmatched_gguf_tensor_message(name))`
  — an unmatched tensor is fatal on the GGUF route because the adapter's name
  table already refused everything it did not recognise.
- `layers/moe/token_dispatcher/bar1ep.py:879` `byte_proof()` — a proof whose
  failure makes the dispatcher DECLINE (`raise Bar1EPUnavailable` at `:877`),
  never warn.

Almost every DANGEROUS row below is a place where one of those three patterns
exists elsewhere in the tree and does not reach here.

### Coverage

Grep command as briefed, per directory:

| surface | grep total | opened in source | not opened |
|---|---|---|---|
| `python/sglang/srt/model_loader/` | 56 | 13 | 43 |
| `python/sglang/srt/speculative/` | 30 | 7 | 23 |
| `python/sglang/srt/layers/moe/` | 34 | 7 | 27 |
| **subtotal (briefed dirs)** | **120** | **27** | **93** |
| C1 string family in `models/` + `layers/quantization/` (`unexpected weight`, `not found in params_dict`, `no parameter`, `skipping/ignoring weight`) | 66 | 19 | 47 |
| **total** | **186** | **46** | **140** |

"Opened in source" means I read the executed branch and its consequence. The
other 140 were read at grep + 3-6 lines of context, which is enough to classify
a deprecation notice or a perf-config fallback but not enough to call something
DANGEROUS — so no DANGEROUS row below rests on a context-only read.

Two further greps were run for the shapes that carry NO warning at all
(`except Exception:` + `pass`/`continue`/`return None`, bare `continue` in a
weight loop) across `layers/moe/expert_offload.py`, `breakable_offload.py`,
`cold_tier_fetch.py`, `cold_tier_shm.py`, `offload_capture_gate.py`,
`expert_heat_migration.py`, `resident_fraction.py`, `router.py`,
`layers/quantization/gguf.py` and `speculative/`. Three rows below come from
that grep rather than the warning grep, and they are the worst of the set
because they carry no message at all: `speculative/eagle_utils.py:193-194`
(A1-02), `models/qwen3_5_mtp.py:404-405` (A1-01), and
`speculative/dflash_utils.py:44-47` (A1-05).

**Not reached, named honestly:**

- `model_loader/ci_weight_validation.py` — 18 of the 56 model_loader sites.
  Read at context only. It is a CI cache-hygiene utility whose warnings all
  precede a `return False` / re-download; no serving path consumes it. Not
  swept in depth, and not claimed clean.
- `layers/moe/token_dispatcher/moriep.py` (4 sites) and `deepep.py` (3 sites) —
  context only. NPU/DeepEP dispatchers, not on this rig's path.
- The remaining 94 `logger.warning` sites in `models/` outside the C1 string
  family, and 40 of the 46 in `layers/quantization/`. Only the C1 family was
  swept there, as briefed. A second pass over `layers/quantization/` is the
  largest single gap in this part.
- `layers/moe/expert_compute_placement.py` (4), `layers/moe/utils.py` (6),
  `moe_runner/triton_utils/fused_moe_triton_config.py` (6) — context only; all
  six of the `utils.py` ones are `"X is not initialized, using <default>"`
  getters and all six config ones are perf-config fallbacks.
- `model_loader/hibernate.py` (5) — context only. Every one of them ends
  `-> cold load`, i.e. falls back to the full, correct load. Benign by
  construction.

### Table

| file:line | site (symbol) | classification | what would be silently wrong | fix pattern |
|---|---|---|---|---|
| `model_loader/weight_utils.py:2058` | `raise_on_unloaded_draft_parameters`, `if loaded_params is None:` / `return` | **DANGEROUS** | The #290/#318 guard against a draft that loaded nothing is a no-op for 25 of 27 draft classes (see A1-01). The draft runs on `torch.empty`, accept length pins at ~1.0, and no error is raised anywhere. | (ii) — the guard IS the #496 shape; it needs reach, not redesign. Make the report mandatory for `is_draft_model`, i.e. turn `loaded_params is None` from "skip" into "this draft class does not report — refuse". |
| `model_loader/loader.py:2409` | `GGUFModelLoader.load_model`, `model.load_weights(_timed(iter(weights_iterator)))` | **DANGEROUS** | Return value discarded, so the draft guard at `loader.py:903` never runs on a GGUF boot. A GGUF MTP draft with a name-table defect (the #113 family) loads zero tensors and reports success. | (ii) — capture the return and call `raise_on_unloaded_draft_parameters` here too. |
| `models/qwen3_next_mtp.py:149` | `Qwen3NextForCausalLMMTP.load_weights`, `super().load_weights(weights, is_mtp=True)` | **DANGEROUS** | The base `qwen3_next.py:1407` DOES `return loaded_params`; the MTP wrapper drops it, so this draft reports `None` and the guard returns at `weight_utils.py:2058`. One missing `return` disables the check for the whole class. | (i)+(ii) — `return super().load_weights(...)`. Same one-line audit for every `*_nextn.py` / `*_mtp.py` wrapper. |
| `models/qwen3_5_mtp.py:404-405` | `Qwen3_5ForCausalLMMTP.load_weights`, `if name_mapped not in params_dict:` / `break` | **DANGEROUS** | No warning at all. A local expert whose packed/dense spelling does not match (`.qweight` vs `.weight`) is dropped in total silence, and the checkpoint name is then recorded as loaded at `:437`. The rank-local "not my expert" case and the quant-mismatch case are indistinguishable here. | (ii) — the parameter-side check is the only thing that separates the two cases; ensure it runs (A1-01) and stop polluting the report (`:437`). |
| `models/qwen3_5_mtp.py:437` | same, `loaded_params.add(name)` at loop-body level | DEGRADED-LOUD | Executed unconditionally, including in the `logger.warning_once(...)` skip branch at `:433`, so `loaded_params` mixes checkpoint names with parameter names. It does not defeat the guard (a skipped name is by construction not in `params_dict`) but it makes the report untrustworthy as evidence. | (iii) — the set is the state probe; it must contain only names actually written. |
| `models/gemma4_causal.py:1437-1442` | `Gemma4ForCausalLM.load_weights`, `unloaded_params = params_dict.keys() - loaded_params` → `logging.WARNING: ("Some weights are not initialized from checkpoints", ...)` | **DANGEROUS** | The check is present and correct; its verdict is a log line. A TARGET model with unwritten real parameters serves uninitialised weights. Observable: a WARNING among hundreds of boot lines. Output is fluent text — nothing downstream flags it. This is the same evidence the draft guard RAISES on. | (i) — the `logging.WARNING` bucket (real parameters, as opposed to the INFO/DEBUG buffer buckets, which are correctly informational) must raise, with the same escape env as `SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS`. |
| `models/qwen3_5.py:1593` | `Qwen3_5ForCausalLM.load_weights`, `logger.warning(f"Parameter {name} not found in params_dict")` + `continue` | **DANGEROUS** on the GGUF route | On GGUF, `deepseek_v4.py:3209` argues the case verbatim: every tensor was mapped by an explicit table, so an unmatched name is a mapping defect and the parameter it should have filled stays uninitialised (#391 walls 10-12: "the server came up on uninitialized weights"). Qwen3.5 is the fork's flagship GGUF bring-up and still only warns. Same line at `:1815` (`Qwen3_5MoeForCausalLM`), `:1957`, `:2340` (the two VL classes). | (i) — hoist `_unmatched_gguf_tensor_message` out of `deepseek_v4.py` and raise on `is_gguf` in every loader, not per model. Only `deepseek_v2.py`, `deepseek_v4.py`, `llama4.py`, `deepseek_common/utils.py` have it today. |
| `speculative/eagle_utils.py:193-194` | `decide_spec_kernel_backend`, `except Exception:  # noqa: BLE001 -- single-process/unit-test contexts` / `tp_group = None` | **DANGEROUS** | The function's own log line at `:220` states the hazard: "one rank without the native ops puts every rank on Triton, because verify decides accept counts and a mixture would desync the group silently." The bare except restores exactly the per-rank answer. A rank that takes this arm skips the `all_reduce` MIN, keeps `group_ok = local_ok`, and can end up on a different verify kernel than its peers. Observable: none — accept counts diverge, output stays fluent. The `# single-process/unit-test contexts` comment is an ASSERTED invariant with no test behind it (CLAUDE.md law 3). | (i) — `get_tp_group()` raising inside a serving process must propagate; gate the swallow on an explicit single-process marker instead of on the exception type. |
| `model_loader/loader.py:1257` | `QuantizedRLModelLoader._load_scale_param`, `logger.warning(f"Scale param shape {scale_param.data.shape[-1]} not divisible by {len(shard_names)}")` | **DANGEROUS** | Warns and then falls straight into `offset = 0` and the per-shard write loop using the truncated `rows_per_shard`. The fp8 scale shards land at wrong offsets — every subsequent dequant of that fused linear uses another shard's scale. Observable: none; logits shift, no shape error. | (i) — the condition is already computed; `raise ValueError` on it instead of logging. |
| `model_loader/loader.py:1223` | same, `logger.warning("[QuantizedRL] Scale parameter not found: %s", scale_param_name)` / `return` | **DANGEROUS** | The weight was requantised; its scale keeps its construction value. Dequant with a stale scale is a silent numeric error on an RL weight-update path, where the model is expected to change between rollouts, so drifting output is the EXPECTED signal — the one observable a user would have is preempted. | (i) — a scale name derived mechanically from a param name that is present is not optional; raise. |
| `model_loader/loader.py:1232` | same, `logger.warning("[QuantizedRL] Scale shape mismatch for %s: expected %s, got %s", ...)` | **DANGEROUS** | Skips the `copy_`, same stale-scale consequence as `:1223`. | (i) |
| `model_loader/weight_utils.py:1929` | `kv_cache_scales_loader`, `"Defaulting to KV cache scaling factors = 1.0 for all layers in TP rank %d as an error occurred during loading."` | **DANGEROUS** | Reached from three `except` arms (`FileNotFoundError`, `JSONDecodeError`, bare `Exception`) after the user explicitly supplied a calibration file. With fp8 KV, scale 1.0 means unscaled e4m3 casts: values past the e4m3 range saturate. Observable: attention quality only. The user asked for calibrated scales and silently got none. | (i) — an explicitly supplied path that fails to load must raise. The fallback is only defensible when no path was given. |
| `model_loader/weight_utils.py:1766`, `:1796` | `maybe_remap_kv_scale_name`, `"... but not found the expected name in the model (e.g. {remapped_name}). {scale_name} is not loaded."` → `return None` | **DANGEROUS** | Callers treat `None` as `continue`. Same end state as `:1929` — fp8 KV on default scales — but per tensor and per name, so no single line says "no scales were loaded at all". `print_warning_once` deduplicates. | (ii) — a completeness check on the attention side after load: if the model declares `k_scale`/`v_scale` parameters and none were written, refuse. |
| `speculative/dflash_worker_v2.py:1859` | `_validate_phase1_sampling_support`, `"DFLASH non-greedy verification is unavailable on this build/device; falling back to greedy argmax verification."` | **DANGEROUS** | A function named `_validate_...` that only warns. A request with `temperature`/`top_p` is verified greedily, so the served distribution is not the requested one. Warned ONCE, on `tp_rank == 0` only, at the first non-greedy request — long after boot. Observable: output is plausible and merely more deterministic. | (i) — refuse the request (or refuse the boot when `is_dflash_sampling_verify_available()` is False and sampling is reachable), rather than substituting a different sampler. |
| `speculative/dflash_utils.py:44-47` | `except Exception:` → `top_k_renorm_prob = None; top_p_renorm_prob = None; tree_speculative_sampling_target_only = None` | **DANGEROUS** (root of the row above) | `_DFLASH_SAMPLING_VERIFY_AVAILABLE` stays `False` with NO log of the exception. An `sgl_kernel` import that fails for an unrelated reason — an ABI/wheel mismatch, the known dual-dist wheel trap — is indistinguishable from "this device has no kernel", and silently converts the whole server to greedy verification. | (iii) — log the exception at ERROR with the resolved wheel path; the availability flag is a state claim and needs a probe, not a swallowed import. |
| `speculative/adaptive_graph_memory.py:1032` | `"Adaptive graph memory: could not attribute tagged buffer of %s (ptr 0x%x) to a build-window segment; physical-isolation audit incomplete for it."` | **DANGEROUS** | Twelve lines above, the SAME audit raises `RuntimeError` when a buffer is attributed to another tag's window ("Pausing one state would unmap another state's memory"). An UNATTRIBUTABLE buffer is the case where the audit could not decide — and it proceeds. If that buffer does belong to another state's segment, a pause unmaps live graph memory and replay reads reclaimed pages: wrong logits, no fault. The in-code comment "Not fatal by itself (snapshot attribution can miss a segment)" is an asserted invariant, unpinned. | (iii) — the audit is the only evidence the mode is safe; an incomplete audit is not a pass. Either raise, or count unattributed buffers and refuse the offload mode above zero. |
| `speculative/adaptive_graph_memory.py:1218-1222` | `_maybe_verify_rank_sync`, `except Exception:` → `logger.warning("SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC check skipped", exc_info=True)` | DEGRADED-LOUD | The instrument for #50/G5 rank-divergent swaps silently does not run; the swap proceeds. Downgraded from DANGEROUS only because the whole check is opt-in (`SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC` is off by default), so it protects nothing by default anyway — a user who switched it ON is precisely the user who must be told it did not run. `RuntimeError` is correctly re-raised at `:1216`. | (i) — a check the user explicitly enabled must raise when it cannot execute. |
| `layers/quantization/npu_mxfp4.py:115-122` | `NPUMXFP4Config.get_quant_method`, `"MXFP4 W4A8 quantization is not yet supported for FusedMoE layers (prefix=%s). Falling back to unquantized MoE — MoE weights will run in full precision (BF16/FP16)."` → `return UnquantizedFusedMoEMethod(...)` | **DANGEROUS** | The message is a false SUCCESS CLAIM about state: an MXFP4 checkpoint has no bf16 MoE tensors to "run in full precision". The layer declares dense `w13_weight`/`w2_weight`, the checkpoint's packed tensors match nothing, and they are dropped by the model's own `not found in params_dict` warning — the exact compound failure of the occasion bug. Reach is NPU-only, so low on this rig, but the shape is textbook. | (i) — `raise NotImplementedError`, exactly as the sibling linear branch at `:108` already does for the same missing kernel. |
| `layers/quantization/modelslim/modelslim.py:212`, `:240` | `get_scheme` / `get_moe_scheme`, `logger.warning(f"Unsupported Linear modelslim scheme: ...")` → `return None` | **DANGEROUS** (NPU-only reach) | `None` routes the layer to the unquantized method over a quantized checkpoint; same packed-vs-dense drop as above. | (i) — refuse by scheme name. |
| `layers/quantization/fp8.py:582-585` | `Fp8LinearMethod._maybe_pad_weight` block check, `if skip_block_quant_check:` → `print_warning_once("Skipping block quantization checks for weight partition.")` | BENIGN | The two callers are `models/mimo_v2.py:492` and `layers/linear.py:2350`, both passing a literal `True` for a known-unshardable layout; the checks skipped are divisibility guards that would raise on a partition the caller has already established is not TP-split. One clause: the guard is skipped where its premise (`tp_size > 1 and input_size // input_size_per_partition == tp_size`) does not hold. | — |
| `layers/quantization/compressed_tensors/compressed_tensors.py:1030-1039` | NVFP4 lane fallback, `"NVFP4: layer '%s' has no form this rank's FP4 lane can serve (%s), so the checkpoint's packed weight is DEQUANTISED to dense %s at load"` | DEGRADED-LOUD | Model of what this class should look like: the change is named, the cost is named ("costs bf16 VRAM instead of 4 bits"), the numerics are argued exactly ("No precision is lost"), and VRAM is the observable. | — |
| `layers/quantization/compressed_tensors/compressed_tensors.py:966-971` | `"Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod"` → `return None` | BENIGN | Branch condition is `elif weight_quant is None:` — the layer is genuinely unquantized in the config; nothing packed to drop. | — |
| `model_loader/gguf_registry.py:347-357` | depth reconciliation, `"GGUF depth reconciliation: ... Using the file's %d (the rest of the geometry matches, ...)"` | BENIGN | Every non-depth geometry disagreement already raised at `:325`, and a `layer_types` pattern that cannot be extended raises at `:339`. Only the file-wins depth case warns, and the model is BUILT at the file's depth, so no layer is left unwritten. | — |
| `model_loader/loader.py:1660-1666`, `:2822-2828` | `ShardedStateLoader` / `RemoteModelLoader`, `"loading tensor of shape %s into parameter '%s' of shape %s"` | DEGRADED-LOUD | A short tensor loads into a narrowed view and the tail of the parameter keeps its construction value. Named per key, and the loop's own `if state_dict: raise ValueError(f"Missing keys ...")` covers the wholly-absent direction. The intended case (LoRA padding) is documented at the site. | — |
| `layers/moe/token_dispatcher/bar1ep.py:853-857` | `_selftest_if_needed`, `"bar1ep: byte proof skipped via SGLANG_BAR1EP_SELFTEST=0. With that, no number from this run carries any statement about whether the bytes actually arrive."` | BENIGN | Opt-out, off-by-default-safe, and the message states the epistemic consequence exactly. `:919`/`:930` both end in `return False` → `raise Bar1EPUnavailable` at `:876`. Template row. | — |
| `layers/moe/offload_capture_gate.py:412-413` | `resolved_backend`, `except Exception:` / `return None` | **DANGEROUS** — already filed | Feeds `validate_breakable_boot`'s `if backend is None: return` bypass at `:358`. Identical to audit #500's **#500-B8**; recorded here for cross-reference, not claimed as a new find. | (i), per #500-B8 |
| `layers/moe/cold_tier_shm.py:536-541` | `_register_host_memory`, `"could not pin peer segment %s (cudaHostRegister -> %s); fetches from it will be pageable and slower, but correct."` | BENIGN | Pinning is a bandwidth optimisation; the "but correct" claim is structural (the mapping is used either way). | — |
| `layers/moe/expert_offload.py:1232`, `:3930`, `:1652` | `except Exception:` / `pass` around `malloc_trim`, `gc`, and host-shard instrumentation | BENIGN | Host-memory hygiene and an instrument row; neither is on a correctness path, and each carries a `# noqa: BLE001` naming that. | — |
| `layers/moe/expert_stats.py:642-647` | `"SGLANG_EXPERT_STATS=1 but the MoE expert offload is inactive ...: nothing will be recorded."` | DEGRADED-LOUD | The instrument states it will produce nothing — the correct handling of a measurement that cannot run. | — |
| `speculative/dflash_solo_pool.py:262-268`, `:334-340` | zero-KV holes / LRU reclaim of draft slots | DEGRADED-LOUD | Draft KV holes degrade accept rate; the target verify still decides every emitted token, so output correctness is unaffected, and both messages name the accept-rate observable and the knob. | — |
| `speculative/eagle_info.py:224-235` | `filter_batch`, `error_msg = f"length of new_indices: ... != length of topk_p: ..., this should not happen"` → `raise` if strict else `logger.warning(error_msg)` then positional truncation | DEGRADED-LOUD | Would be DANGEROUS (positional truncation misassigns `draft_probs`, which breaks the rejection-sampling correctness argument, not just the accept rate) — except `SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)` (`environ.py:1408`), so the default path RAISES. The warning arm is reachable only for a user who opted out. Recorded because the opt-out is the wrong default direction for a "this should not happen" invariant. | (i) if the flag is ever flipped |
| `speculative/dflash_worker_v2.py:254-258` | `"DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s."` | DEGRADED-LOUD | Runs the draft at a block size its checkpoint was not trained for; accept length is the observable and the message names both numbers. Same shape at `dspark_components/dspark_config.py:101-107`. | — |
| `speculative/dflash_worker_v2.py:935`, `:1630`, `:1668`, `:2113`, `:2435` | fused-KV / Triton fast paths, `"... failed, falling back to sequential path: %s"` then `self._use_fused_kv_materialize = False` | DEGRADED-LOUD | Each latches the flag off and recomputes via the eager path. The residual question — whether the failed fused call left a partial KV write that the fallback then overwrites completely — is a COMMENT-level claim at these sites, untested. Flagged as a testable-claim, not scored DANGEROUS. | (iii) if pursued: a falsifier that injects the exception mid-write and compares KV bytes against the never-fused reference |
| `speculative/draft_worker_common.py:45-51` | `"%s draft worker only supports attention_backend in %s for now, but got %r. Falling back to '%s'."` | BENIGN | Backend substitution, both names printed, no state left unwritten. | — |
| `speculative/dspark_components/dspark_planner.py:194-198` | `"DSpark SPS table is uninitialized (flat): the verify budget degenerates to verify-all (zero scheduling gain)."` | DEGRADED-LOUD | Verify-all is the conservative direction — more verification, not less; correctness unaffected, and the perf loss is named with its knob. | — |
| `speculative/spec_registry.py:112-118`, `layers/moe/utils.py:244-248` | deprecation notices | BENIGN | Forward-compat, behaviour named. | — |
| `layers/moe/utils.py:322`, `:332`, `:342`, `:350`, `:400` | `"X is not initialized, using <default>"` getters | BENIGN | Read-before-publish of a flags singleton; each names the default it installs, and the defaults are the documented ones. | — |
| `layers/quantization/gguf.py:273`, `:367`, `:414`, `:464`, `:623`, `:789` | wheel/capability probes, `except Exception:` → `False` / `0` / `(0, 0)` | BENIGN | Every one selects a SLOWER but correct kernel route (MXFP4 repack, #72 reroute, MMVQ instead of MMQ, no-`out=` dequant). No numeric path changes; `gguf.py:612-616` even documents deliberately not latching. Perf reach, not correctness. | — |
| `model_loader/utils.py:134`, `:144`, `:171`, `:186` | Transformers-impl compatibility gate skips | BENIGN (context-only read) | All four warn on an explicitly requested `--model-impl=transformers`, i.e. the user asked to bypass. Classified from the message text plus its condition; not opened in depth. | — |
| `model_loader/weight_utils.py:727-730` | `"Found mtp.safetensors but it's not referenced in {index_file}. This is a checkpoint packaging bug. Auto-adding it for loading."` | BENIGN | Adds a file that would otherwise be missed; the failure direction is toward loading MORE, and the guard in A1-01 covers the result. | — |

### Top findings

**#505-A1-01 — the draft-load completeness guard reaches 2 of 27 draft classes,
and no GGUF boot at all.** `raise_on_unloaded_draft_parameters`
(`weight_utils.py:2032`) is the tree's answer to exactly the occasion bug, and
its docstring states the ambition: *"Hoisting the check to the loader makes it a
property of loading A DRAFT, not of one model class."* Its actual reach is set
by three lines. First, `if loaded_params is None: return` (`:2058`): of the 27
draft classes in `models/`, only `qwen3_5_mtp.py` and `step3p5_mtp.py` return a
loaded set at all — `gemma4_mtp.py:382` returns its super's, so three in effect.
`deepseek_v4_nextn.py`, `qwen3_next_mtp.py`, `llama_eagle3.py`,
`kimi_k25_eagle3.py`, `qwen3_moe_mtp.py`, all `*_nextn.py` and the rest report
`None` and are skipped. `qwen3_next_mtp.py:149` is the sharpest case: the base
class DOES `return loaded_params` (`qwen3_next.py:1407`) and the MTP wrapper
drops it with a missing `return`. Second, the only call site is `loader.py:903`,
inside `DefaultModelLoader` — `GGUFModelLoader` discards the return value
outright (`loader.py:2409` `model.load_weights(_timed(iter(weights_iterator)))`),
so a GGUF MTP draft, the fork's own #113 territory, is unguarded. Third,
`QuantizedRLModelLoader`'s `load_weights_proxy` (`loader.py:1095-1102`) also
returns `None`, disabling the guard for anything it wraps. This is the
REACH-INCLUDES-PARAMETERS law applied to a guard rather than a threshold: the
mechanism exists, is correct, is tested
(`test/registered/unit/model_loader/test_draft_quantization_namespace.py`), and
does not act on the configuration where today's bug happened.
*Task:* `#505-A1-01: make the draft-load completeness check reach every draft class and every loader (GGUF, QuantizedRL), not only DefaultModelLoader x self-reporting models`

**#505-A1-02 — the spec-kernel-backend collective degrades to a per-rank answer
on any exception.** `speculative/eagle_utils.py:193-194`:
`except Exception:  # noqa: BLE001 -- single-process/unit-test contexts` /
`tp_group = None`. The group `all_reduce(MIN)` is then skipped and
`group_ok = local_ok` stands. The function's own log line at `:220` names the
consequence: *"one rank without the native ops puts every rank on Triton,
because verify decides accept counts and a mixture would desync the group
silently."* A rank taking this arm can end up on a different verify kernel from
its peers, with no warning and no observable — accept counts diverge, output
stays fluent. The comment naming the intended contexts is an asserted invariant
with nothing pinning it. This is the rank-local-before-collective family
inverted: the rank-local fallback is what breaks the group.
*Task:* `#505-A1-02: decide_spec_kernel_backend must not fall back to a per-rank verify kernel when get_tp_group() raises`

**#505-A1-03 — the QuantizedRL scale reload has three warn-and-proceed sites,
one of which writes scales at wrong offsets.** `loader.py:1223` (scale parameter
not found → `return`), `:1232` (shape mismatch → skip the `copy_`), and worst,
`:1257`: the divisibility warning does not guard anything — control falls
straight through to `offset = 0` and the per-shard write loop using the
truncated `rows_per_shard`, so the fp8 scales of a fused linear land on the
wrong shards. All three leave a requantised weight paired with a wrong or stale
scale. The observable that would normally catch this — changed output — is
preempted, because this path exists to change the weights between RL rollouts.
*Task:* `#505-A1-03: QuantizedRL scale reload must raise on a missing / mis-shaped / non-divisible scale instead of warning and writing`

**#505-A1-04 — an explicitly supplied KV-scale calibration file that fails to
load defaults every layer to 1.0.** `weight_utils.py:1929`, reached from three
`except` arms including a bare `except Exception`. With fp8 KV that is unscaled
e4m3, i.e. saturation past the representable range, with no observable but
answer quality. The per-tensor sibling is `maybe_remap_kv_scale_name`
(`:1766`, `:1796`), which returns `None` — read as `continue` by callers — so
individual `k_scale`/`v_scale` tensors are dropped one deduplicated warning at a
time and nothing states that zero scales were loaded in total.
*Task:* `#505-A1-04: a supplied --quantization-param-path that fails to load must raise; add a loaded-scale completeness check for fp8 KV`

**#505-A1-05 — DFLASH silently substitutes greedy verification for the
requested sampler.** `dflash_worker_v2.py:1859`, inside a method named
`_validate_phase1_sampling_support`, warns once on rank 0 at the first
non-greedy request and then verifies with argmax. The served distribution is not
the requested one and the output is plausible either way. The root is
`dflash_utils.py:44-47`: a bare `except Exception` around the `sgl_kernel`
import sets `_DFLASH_SAMPLING_VERIFY_AVAILABLE = False` with no log at all, so an
ABI/wheel mismatch (a known trap in this tree) is indistinguishable from a
device that genuinely has no kernel.
*Task:* `#505-A1-05: refuse non-greedy DFLASH requests when the sampling-verify kernels are absent, and log why the sgl_kernel import failed`

**#505-A1-06 — the "unmatched GGUF tensor is fatal" argument is written down
once and enforced in four files.** `deepseek_v4.py:3209-3231` states the general
principle — every GGUF tensor is mapped by an explicit table, so an unmatched
name is a mapping defect, and #391 walls 10-12 are what it costs when it only
warns — and then applies it only in `deepseek_v2.py`, `deepseek_v4.py`,
`llama4.py` and `deepseek_common/utils.py`. `qwen3_5.py:1593` / `:1815` /
`:1957` / `:2340`, the fork's flagship GGUF bring-up across four classes, still
does `logger.warning(...)` + `continue`. So does `gemma4_causal.py`, whose
parameter-side check exists but only logs (`:1437-1442`).
*Task:* `#505-A1-06: hoist _unmatched_gguf_tensor_message to the loader so an unmatched GGUF tensor is fatal for every model, not four`

**#505-A1-07 — an incomplete physical-isolation audit is reported as a
warning.** `adaptive_graph_memory.py:1032`. Twelve lines above, the same audit
raises `RuntimeError` when a tagged buffer is attributed to a FOREIGN build
window, with the reason spelled out: "Pausing one state would unmap another
state's memory." The unattributable case — where the audit could not decide — is
logged and the run continues under the mode the audit exists to license. Per the
SUCCESS-CLAIMS law, an instrument that could not discriminate has not returned a
pass.
*Task:* `#505-A1-07: an unattributable tagged buffer must fail the adaptive-graph-memory isolation audit, not warn`

**#505-A1-08 — a target model that loaded nothing is a warning; a draft that
loaded nothing is an error.** `gemma4_causal.py:1437-1442` computes exactly the
right set (`params_dict.keys() - loaded_params`) and routes real, unwritten
parameters to `logging.WARNING`. The argument the draft guard makes at
`weight_utils.py:2039-2046` — "the drafter then runs on `torch.empty` … the only
symptom is an accept rate that looks like a weak drafter" — transfers with the
symptom changed, not the mechanism: a target model with unwritten parameters
produces wrong logits and there is no accept rate to look weak. The escape hatch
already has a pattern (`SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS`, `environ.py:243`).
*Task:* `#505-A1-08: unwritten real parameters must fail the target load too, with the same named escape env as the draft guard`

**#505-A1-09 — an unsupported quant scheme returns an unquantized method over a
quantized checkpoint.** `npu_mxfp4.py:115-122` is the clearest instance and its
message is a false state claim ("MoE weights will run in full precision
(BF16/FP16)" — there are no bf16 MoE weights in an MXFP4 checkpoint). The layer
then declares dense parameters, the packed tensors match nothing, and they are
dropped by the model's own `not found in params_dict` warning: the occasion bug
assembled from two sites, neither of which is wrong on its own. Same shape at
`modelslim.py:212` and `:240`. Reach is NPU-only, which is why it is ranked
here rather than higher. The correct handling is three lines above it, at
`npu_mxfp4.py:108`: `raise NotImplementedError` for the same missing kernel on
the linear branch.
*Task:* `#505-A1-09: an unsupported quant scheme must refuse, not hand a packed checkpoint to an unquantized method`

### Note on what this part does NOT claim

The 140 unopened sites are not certified benign. The largest coherent gap is
`layers/quantization/` outside the C1 string family (40 sites) — that is where
the packed-vs-dense family of `#443`/`#446` lives, and it deserves the same
sweep. `model_loader/ci_weight_validation.py` (18 sites) is a CI utility and was
deliberately deprioritised. No DANGEROUS row above rests on a context-only read,
a docstring, or a comment; where a comment is the only evidence for an
invariant, the row says so explicitly (`eagle_utils.py:193`,
`adaptive_graph_memory.py:1032`, the `dflash_worker_v2` fused-KV fallbacks).
