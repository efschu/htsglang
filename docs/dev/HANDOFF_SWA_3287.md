# HANDOFF — #3287, the SWA target-verify double translate + the #629 leftovers

Branch `fix/swa-target-verify-3287`, worktree `/spinning/wt-swa-fix`.

**MERGE ORDER: this branch STACKS on the #629 bundle.** It is based on
`fix/replay-mirror-dequant-629` @ `0a55facc74`, not on `feat/route-a-631`, and
it edits the same two files (`triton_backend.py`, `flashinfer_backend.py`) and
builds on the host-mirror channel that bundle introduced. Merge #629 first;
merging this alone will not apply. Not merged here — operator sequences.

Desk + hermetic only. No GPU taken, no arbitration held, serving (30030) and
the router (30099) untouched. Other worktrees untouched.

Errors first: section 1 is what is still open, section 2 what was fixed,
section 3 what dissolved.

---

## 1. OPEN

### 1a. GPU-gated, recipe only: does the shipping kernel keep `2^-127`?

Inherited from the #629 handoff (its 1d), still undecided and still not
decidable at desk. `dequant_k_cache.py:139` lowers `tl.exp2` to NVPTX
`ex2.approx`; the `.ftz` form flushes subnormal RESULTS to zero, the plain form
does not, and only byte `0x00` decodes to a subnormal. If the shipping build
emits the ftz form, hardware coincidentally reproduces the pre-#418 (wrong)
reference.

**The recipe in the #629 handoff does not run.** `JITFunction` in the installed
triton 3.6.0 has no `cache` attribute; the compiled-kernel dict moved to
`device_caches[device][0]` (`triton/runtime/jit.py:770-771`, tuple layout at
`:707`). The old recipe also read the cache at import time, before anything is
compiled. Corrected:

```
CUDA_VISIBLE_DEVICES=<any> PYTHONPATH=/spinning/wt-swa-fix/python \
/spinning/htsglang-gpu/.venv/bin/python - <<'EOF'
import torch
import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc
k = dqc._dequantize_k_cache_paged_kernel
# Compile first: call dqc.dequantize_k_cache_paged(...) on a tiny cache, or
# k.warmup(*args, grid=(1,)). warmup returns the CompiledKernel and launches
# nothing (jit.py:731-746).
dev = torch.cuda.current_device()
ck = next(iter(k.device_caches[dev][0].values()))   # NOT k.cache[0]
if hasattr(ck, "result"):
    ck = ck.result()                                 # async-compile mode
print([ln for ln in ck.asm["ptx"].splitlines() if "ex2" in ln])
EOF
```

Pass: no `ex2.approx.ftz.f32` in the PTX, and byte `0x00` × fp8 `0x7E` gives
`448 * 2**-127 = 2.633e-36` on device. If `.ftz` IS emitted, the KERNEL is the
side to fix — the reference stays as #418 left it.

**One dump now answers both sites**, see 3a: the second decode site does not
have a different lowering.

### 1b. Byte 255: pinned, not repaired, because it is unreachable

The `flash_mla_sm120` decode pair genuinely diverges at `0xFF` (torch → NaN,
spec-correct; Triton `exp2(128)` → inf). It is now pinned hermetically rather
than left silent, and deliberately NOT repaired: the fork's own k-cache
quantizer cannot emit that byte (see 3c), so a numerics change there could not
be validated against anything.

RECIPE for any GPU window, in `test/registered/kernels/test_flash_mla_backends.py`
(SM120-gated, cannot run at desk): its cache generator draws
`torch.randint(120, 131, ...)` — 11 of the 156 encodings the quantizer can
actually emit, narrow by accident ("keep exponents in a sane range") rather than
by argument. Widen to the reachable `[92, 248)` and run
`TestSparseDecodeTritonVsTorch` on the 5090. The decode half of that widening is
already proven safe hermetically
(`test_ue8m0_decode_divergence_3287.py::test_the_two_decoders_agree_over_every_reachable_scale_byte`);
what a card adds is the gather/kernel machinery around it.

### 1c. The eager `update_sliding_window_buffer` callers still take the device read

`triton_backend.py` has three eager callers (the `init_forward_metadata` decode
and verify sites, and the draft site). They still bound the in-buffer translate
with `window_kv_indptr[-1]`, i.e. still sync. Left deliberately: they are
out-of-graph, outside the replay-prep collective window that makes the sync
fatal, and each needs its own mirror argument. The channel is now there —
`update_sliding_window_buffer(..., window_lens_sum=...)` — so wiring them is a
small follow-up, not a redesign.

### 1d. `update_single_wrapper`'s sizing exposure, unchanged

Inherited open item from the #629 bundle's 2c, restated so it is not lost: that
arm has the same "mirror accepted unchecked for SIZING" exposure the freshness
guard removes elsewhere, and was left as #623 wrote it. Untouched here.

---

## 2. FIXED

### 2a. S-class: the target-verify SWA graph fill translated the window TWICE

`triton_backend.py`, `_update_target_verify_buffers`. The finding was reported
as "appears to translate twice". It does. Turned into a verdict by a counting
test written and run BEFORE any edit: on the unmodified tree the window is
translated **2×**, and its values equal the double-translated expectation
exactly.

Mechanism. `update_sliding_window_buffer` ends with an optional full→swa
translate guarded by `skip_full_to_swa_translation`. Under the unified pool that
translate is supposed to be DEFERRED — `_translate_cuda_graph_shared_pool_locs`
runs it later, once, out of graph, over the same static
`cuda_graph_window_kv_indices`. The DECODE graph fill defers correctly
(`skip_full_to_swa_translation=(self._translate_kv_loc is not None)`). Its
target-verify sibling passed nothing, so the flag defaulted `False`, the fill
translated in place, and the deferred pass translated the same buffer again — on
both the capture and the replay leg.

Neither translator is idempotent: both are table gathers
(`UnifiedSWAKVPool.translate_loc_from_full_to_swa` indexes `virtual_to_physical`,
`unified_memory_pool.py:1098-1110`; `SWAKVPool`'s indexes
`full_to_swa_index_mapping`, `swa_memory_pool.py:172-175`). So the verify window
became `v2p[v2p[x]]` — in range, correctly shaped, wrong. Silent wrongness on the
path that decides which drafted tokens are accepted.

Reachability is concrete, not hypothetical: allocator
`UnifiedSWATokenToKVPoolAllocator` (`multi_ended_allocator.py:1968`, exposes
`translate_kv_loc` at `:2204`) over pool `UnifiedSWAKVPool`
(`unified_memory_pool.py:996`, exposes `translate_loc_from_full_to_swa`) makes
both guard conditions true at once.

Fix: pass the flag, mirroring the decode sibling.

### 2b. …and it took a blocking D2H on THREE of the four graph legs

The same call bounded its slice with `window_kv_indptr[-1]`, a 0-dim device
tensor whose `__index__` is an unbounded blocking device-to-host read inside the
replay-prep window (#616h class). The handoff attributed this to the verify
graph path. The counting test measured it on all four legs, and it is broader:

| leg | syncs on unmodified tree |
|---|---|
| unified, decode | 0 (defers, so the block is skipped) |
| unified, verify | 2 |
| baseline SWA, decode | 2 |
| baseline SWA, verify | 2 |

Baseline SWA takes it on BOTH legs, because there the deferral does not apply
and the in-buffer translate is the real one. Fixed by putting the bound on the
#629 host mirror: new `_swa_window_host_sum` (`triton_backend.py`, beside
`_verify_host_mirror`), plumbed as `window_lens_sum` from both graph fills.

`_translate_cuda_graph_shared_pool_locs` now derives its own bound from that
same helper instead of its own inline `clamp(max=W).sum()`. Same discipline as
`_verify_host_mirror`: the fill and the deferred translate address one buffer, so
if they ever disagreed on the prefix length the tail would be left
half-translated — silently, since neither shape nor dtype changes.

### 2c. The two `.update()` callers that promised a prefix mirror and sent none

`flashinfer_backend.py`, `init_forward_metadata_out_graph`. Of six
`indices_updater_prefill.update(...)` callsites, the two that pass
`spec_info=None` under a cuda graph reached the consuming branch
(`spec_info is None and self.attn_backend.uneven_dcp`) without
`extend_prefix_lens_cpu`, so both fell back to the device read the channel exists
to remove — `int(full_indptr[bs].item())` (weighted) / `int(dcp_lens.sum().item())`
(even):

* the dLLM extend graph replay
* plain EXTEND under the full prefill CUDA graph

Both reachable, both only via DCP/weightless — a plain flashinfer boot never
enters the consuming branch, which is why this survived normal use.

**Mirror the expression, not the name.** The branch indexes over `prefix_lens`,
so the mirror must describe whatever `prefix_lens` IS at that callsite. At the
full-CG site that is `forward_batch.extend_prefix_lens`, whose own mirror is
right. At the dLLM site it is NOT — `prefix_lens` there is
`seq_lens - block_size`, so forwarding `extend_prefix_lens_cpu` would have sized
the buffer from a different vector: a silent mis-size, strictly worse than the
stall. The dLLM mirror is derived by the same subtraction on the host.

Note for readers of the consumer: the comment at the weighted call site says
"prefix_lens IS forward_batch.extend_prefix_lens, so this host mirror is the same
sum". That is true of the eager caller it was written for and false of the dLLM
one.

### 2d. The census that can catch 2c, one layer out

The #629 handoff's 1c was right: `test_dcp_index_host_sum_623.py` asserts each
builder callsite *names* `total_tokens=` — a source-text scan whose own docstring
concedes "even if it evaluates to None at runtime". By construction it cannot
catch a caller that supplies nothing, which is the #616h failure mode itself.

`test/srt/distributed/test_dcp_update_caller_census_3287.py` is BEHAVIOURAL: it
drives the real `init_forward_metadata_out_graph` and inspects the kwargs that
arrive. It asserts the mirror's VALUE equals the host twin of the device
`prefix_lens` handed to the same call, so a wrong-but-non-None vector fails too —
a presence-only check would have accepted exactly the dLLM mistake described
above.

---

## 3. DISSOLVED under scrutiny

Booked with evidence rather than repeated. Bundled appearances split into
separate facts, again.

### 3a. "The two ue8m0 decode sites do not share a lowering"

They do. On the installed triton 3.6.0, `tl.exp2 is tl.math.exp2` is `True` —
one function object, defined once in `triton/language/math.py:107-109` and
re-exported. There is no libdevice `__nv_exp2f` path in play. The ftz question
(1a) is ONE question, one PTX dump settles both sites, and the handoff's
"resolve the asymmetry at the same time" item has nothing to resolve. Pinned as
a test so a future Triton that splits them re-raises it.

### 3b. "The full-CG prefix mirror must be padded or it UNDER-sizes"

It must not. Both consumers take only a SUM
(`dcp_host_total_tokens` → `int(lens.sum())`, `dcp_host_even_total` →
`int(get_dcp_lens(lens,...).sum())`), and `replay_prepare` zeroes the device
tail (`prefill_cuda_graph_runner.py`, `extend_prefix_lens[bs:r].zero_()`) before
this runs, with `get_dcp_lens(0, ...) == 0`. So the unpadded real-`bs` list and
the padded device vector have equal sums; neither call passes an `expected_sum`,
so there is no staleness refusal either. Padding would be harmless and is not
required. (The reason `_full_cg_seq_lens_cpu` IS padded is different and does not
apply to this arm: it is cumsum'd into a length-`bs+1` host indptr on the
`fast_prefill_plan` path, a shape requirement — and that path additionally needs
`num_tokens_per_req` from a `spec_info` that is `None` here.)

### 3c. "The dLLM site needs a new field on the decode replay view"

It does not. `build_replay_fb_view` indeed carries no `extend_prefix_lens_cpu` —
that part of the report is exact — but the callsite does not need it, because its
prefix is DERIVED (`seq_lens - block_size`) rather than carried, and `seq_lens_cpu`
is already on the view. Adding the field would also have been the wrong fix: it
would have supplied a mirror of a vector this callsite never uses (see 2c).

### 3d. "The flash_mla comparison at `:224`/`:238` is poisoned"

That comparison is torch-against-torch: `_build_kvcache` computes its reference
with the SAME `.view(torch.float8_e8m0fnu).float()` the code under test uses, so
no Triton decoder participates and byte 255 could not poison it — both sides
would produce NaN. The genuinely exposed comparison is
`TestSparseDecodeTritonVsTorch._run`. The masking claim itself stands:
`randint(120, 131)` has an exclusive high, so the drawn range is 120..130 and
255 is genuinely excluded — as is `0x00` and everything else outside that window.

### 3e. Byte 255 is unreachable through the fork's quantizer

`quant_k_cache.py` derives the byte as
`ceil(log2(max(max_abs, EPS) / FP8_MAX)) + 127` with `EPS=1e-8`, `FP8_MAX=448`,
deliberately unclamped and reasoned in place. For any finite bf16 input that
lands in `[92, 247]`. So the 255 divergence is a conformance gap in a decoder,
not a live wrong answer — which is what turns 1b from "fix" into "pin + recipe".
Asserted from the quantizer's own formula, so a change to `EPS`, `FP8_MAX` or the
clamping decision re-opens it.

---

## 4. TESTS

All three new files hermetic — no CUDA, no process group, no model, no Triton
launch (the index kernel is replaced by a CPU stand-in with the same fill
semantics).

| file | result |
|---|---|
| `test/registered/unit/layers/attention/test_swa_target_verify_translate_3287.py` | **7 passed, 6 subtests** |
| `test/srt/distributed/test_dcp_update_caller_census_3287.py` | **6 passed** |
| `test/registered/unit/layers/attention/test_ue8m0_decode_divergence_3287.py` | **5 passed** |

### Red-first, measured before the fix existed

| pin | on unmodified tree |
|---|---|
| verify window translate count | **2**, expected 1 |
| verify window values | equal to `v2p[v2p[x]]`, i.e. double-translated |
| decode vs verify window agree | differ |
| blocking host reads, 3 of 4 graph legs | 2 each |
| `.update()` mirror present (dllm, extend) | `None` |
| `.update()` mirror value (dllm, extend) | `None` |

6 failed → 7 passed for the SWA file, 6 failed → 6 passed for the census.

### The corpus catches both regression directions

Both a DROPPED translate and a DOUBLED one are silent, so the SWA corpus asserts
that virtual / translated-once / translated-twice are three pairwise-distinct
vectors, and that distinctness is itself a test
(`test_the_corpus_distinguishes_all_three_outcomes`). An identity table would
make "twice" indistinguishable from "once" and the pin would be blind to the very
regression it exists for. `test_baseline_swa_still_translates_the_window_exactly_once`
covers the drop direction: with no unified allocator there is no deferred pass, so
skipping the in-buffer translate would leave the window holding virtual ids.

The census likewise asserts the mirror's value, not its presence, and
`test_the_dllm_mirror_is_not_the_extend_prefix_vector` fails specifically on the
plausible-but-wrong fix.

### Suites

Measured with `PYTHONPATH=<tree>/python` per tree, base runs on a detached
worktree at `0a55facc74` so the comparison is same-box and same-command. Tree
md5-frozen across each run and verified unchanged afterwards (the #629 shift's
`inspect.getsource` lesson).

| suite | base `0a55facc74` | branch |
|---|---|---|
| `test/registered/unit/layers/attention` | 199 passed, 1121 subtests | 206 passed, 1127 subtests |
| `test/srt/distributed` + `registered/unit/distributed` + `registered/dcp` + `registered/cp` | 23 failed, 2757 passed, 6 skipped, 52 errors, 1019 subtests | 23 failed, **2763** passed, 6 skipped, 52 errors, 1019 subtests |
| `scripts/run_631_flip_family.sh` | **1116 passed** | **1116 passed** |

Every delta is accounted for by the new files and nothing else:

* attention dir: +7 tests, +6 subtests = the two new files in that dir.
* distributed set: +6 passed = the new census file, which lives in
  `test/srt/distributed`. Failed and error counts are unchanged at 23 / 52, and
  the FAILED/ERROR **name lists** are identical between base and branch — so the
  equal counts are not an equal-sized swap. Those 23+52 are pre-existing on base
  (missing `sglang` binary, UCX device warnings, process-table-sensitive bar1
  tests) and none of them touch anything in this diff.
* flip family: byte-for-byte the same 1116, before and after.

ruff `check`: `flashinfer_backend.py` 30 / 30 (all pre-existing at base),
`triton_backend.py` 0 / 0, all three new files 0. `ruff format`:
`triton_backend.py` was already unformatted at base in the same 6 hunks, none of
them in the edited regions; all three new files formatted.

---

## 5. NEXT

1. Run the 1a PTX recipe (corrected above) in any GPU window — seconds, one
   card, no model. It settles #418 on-device for both ue8m0 sites at once.
2. Widen the SM120 generator per 1b and run `TestSparseDecodeTritonVsTorch` on
   the 5090.
3. Wire the three eager `update_sliding_window_buffer` callers to
   `window_lens_sum` (1c) — the channel exists now.
4. 1d, `update_single_wrapper`'s unchecked sizing mirror, is still open from the
   #629 bundle.
