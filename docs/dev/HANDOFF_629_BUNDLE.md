# HANDOFF — small-bug bundle #629 + #418

Branch `fix/replay-mirror-dequant-629`, worktree `/spinning/wt-629-bundle`,
based on `origin/feat/route-a-631` @ `c2ceac7f31`. Desk + hermetic only; no
GPU was taken, no arbitration held, serving and the router untouched.
Not merged — operator sequences.

Errors first: section 1 is what is still broken, section 2 what was fixed.

---

## 1. OPEN — found during this work, NOT fixed

### 1a. S-class: the target-verify SWA graph fill takes a blocking D2H, and translates twice

`triton_backend.py:3287-3292`, inside `update_sliding_window_buffer`:

```python
kv_last_index = window_kv_indptr[-1]
window_kv_indices[:kv_last_index] = (
    token_to_kv_pool.translate_loc_from_full_to_swa(window_kv_indices[:kv_last_index])
)
```

`window_kv_indptr[-1]` is a **CUDA 0-dim tensor** used as a Python slice bound,
so `__index__` forces an unbounded blocking device-to-host read — the same
fatal class as #616h, and inside the replay-prep collective window.

It is reached because the **verify** graph fill calls
`update_sliding_window_buffer` **without** `skip_full_to_swa_translation`
(`triton_backend.py:1302`, args end `:1315`), so it defaults `False`
(`:3248`). Its **decode** sibling does pass it (`:1237`,
`skip_full_to_swa_translation=(self._translate_kv_loc is not None)`).

There is a second, worse consequence than the sync. `_translate_cuda_graph_shared_pool_locs`
translates the window buffer again after the fill (`:1497-1511`), on both the
capture and replay legs. So when `_translate_kv_loc is not None` **and** SWA is
on **and** the pool exposes `translate_loc_from_full_to_swa`, the verify graph
path translates the window **twice** — silently wrong window indices, not a
crash. The decode path avoids exactly this by skipping the in-buffer translate.

Candidate fix is one line, mirroring the decode sibling at `:1237`. **Not
applied**: it changes translation behaviour on a path I could not exercise, and
the double-translate reading should be confirmed by a counting test before the
change lands. That test is hermetic and cheap — fake `token_to_kv_pool` whose
`translate_loc_from_full_to_swa` increments a counter, drive
`_update_target_verify_buffers` then `_translate_cuda_graph_shared_pool_locs`,
assert the count. Recommend doing that first.

### 1b. Two `.update()` callers still reach `owner.py:548`

The two PrefillWrapper *updater arms* are now wired (section 2b), but one layer
out, `FlashInferAttnBackend` calls `indices_updater_prefill.update(...)` at six
places and only one forwards `extend_prefix_lens_cpu`:

| callsite | mode | reaches | net |
|---|---|---|---|
| `flashinfer_backend.py:1497` | target_verify | `:7246` (uses `paged_kernel_lens_cpu`, passed `:1500`) | covered |
| `flashinfer_backend.py:1509` | dllm_extend | **`:7031`, needs `extend_prefix_lens_cpu`** | **UNWIRED** |
| `flashinfer_backend.py:1521` | draft_extend_v2 | `:7145` | covered |
| `flashinfer_backend.py:1539` | extend, full prefill CG | **`:7031`** | **UNWIRED** |
| `flashinfer_backend.py:1675` | eager verify | `:7246` | covered |
| `flashinfer_backend.py:1719` | eager extend | `:7031`, forwards it at `:1732` | wired (#616h) |

For `:1539` the mirror is in scope — the padded view is a `copy.copy(forward_batch)`
(`prefill_cuda_graph_runner.py:595`) so `extend_prefix_lens_cpu` survives — but
it has length `bs`, not the padded `r`, so a correct plumb must pad it the way
`_full_cg_seq_lens_cpu` is padded at `prefill_cuda_graph_runner.py:593-598`;
forwarding it unpadded would UNDER-size the index buffer. For `:1509` the decode
replay view (`build_replay_fb_view`, `decode_cuda_graph_runner.py:182-209`)
carries no `extend_prefix_lens_cpu` at all — a new field on the view, sourced
from `forward_batch_info.py:1328`.

### 1c. The #623 census pin cannot catch 1b, by construction

`test_dcp_index_host_sum_623.py:671-711` asserts each builder callsite *names*
`total_tokens=`. It says nothing about whether the argument can ever be
non-`None`. That is the #616h failure mode — "a wired channel with no caller
supplying it" — recurring one layer out, and it is why 1b survived a green
census. A pin over the `.update()` callers would fail today on `:1509`/`:1539`;
it should be added together with their fix, not before it.

### 1d. GPU-gated, recipe only: does the shipping kernel actually keep `2^-127`?

The #418 fix (section 2c) makes the reference spec-conformant. Whether the
**kernel** achieves the same on a real card is not decidable hermetically.
`dequant_k_cache.py:139` uses `tl.exp2`, which lowers to NVPTX `ex2.approx`.
PTX `ex2.approx.ftz.f32` flushes subnormal **results** to zero; plain
`ex2.approx.f32` does not. Which form Triton emits is a codegen/build detail
the interpreter never exercises. If the shipping build emits the ftz form,
hardware would coincidentally reproduce the old (wrong) reference.

RECIPE — no model, no serving, one card for seconds:

```
CUDA_VISIBLE_DEVICES=<any> PYTHONPATH=/spinning/wt-629-bundle/python \
/spinning/htsglang-gpu/.venv/bin/python - <<'EOF'
import torch, sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc
# 1. dump the PTX of the compiled kernel and grep the exp2 lowering
k = dqc._dequantize_k_cache_paged_kernel
print([ln for ln in list(k.cache[0].values())[0].asm["ptx"].splitlines() if "ex2" in ln])
# 2. direct check: scale byte 0x00, nope byte 0x7E must give 448 * 2**-127
#    (build the cache with test_dequant_k_cache_subnormal_scale_418._build_cache)
EOF
```

Expected pass: no `ex2.approx.ftz.f32` in the PTX, and the on-device value
equals `448 * 2**-127 = 2.633e-36`. If `.ftz` IS emitted, the KERNEL is the
side to fix (force the non-ftz path or decode the exponent by integer bit
construction) — the reference stays as it is now, because OCP MX v1.0 §5.4.1
and torch's `Float8_e8m0fnu.h:132-134` both make `2^-127` the correct value.

Note an asymmetry worth resolving at the same time: the fork's two ue8m0 decode
sites do not even share a lowering. `dequant_k_cache.py:139` uses `tl.exp2`
(NVPTX `ex2.approx`), `flash_mla_sm120_triton.py:172` uses `tl.math.exp2`
(libdevice `__nv_exp2f`).

### 1e. Byte 255: both sides non-conformant, but they AGREE

Spec makes `0xFF` the sole NaN encoding. Both the reference and the Triton
kernel compute `exp2(128) == inf`. Because they agree, the oracle is not
poisoned and #418 does not cover it — but it is a real non-conformance and is
now **pinned as such**, not left silent, in
`test_dequant_k_cache_subnormal_scale_418.py::test_scale_byte_255_is_a_known_shared_non_conformance`.

A separate, genuinely poisoned pair exists at the same byte:
`flash_mla_sm120.py:115,119-126` decodes via `.view(torch.float8_e8m0fnu).float()`
→ **NaN** (spec-correct), while `flash_mla_sm120_triton.py:167-173` gives
**inf**. Their comparison at `test/registered/kernels/test_flash_mla_backends.py:224,238`
is masked by a generator drawing `randint(120, 131)` (`:105-111`). Same family
as #418, different pair, not fixed here.

### 1f. fp32-subnormal PRODUCTS remain a divergence, deliberately unasserted

With the scale decode fixed, reference and kernel agree **exactly** on every
`(fp8 code, scale byte)` pair whose fp32 product is normal. Where the product
itself lands subnormal (scale `2^-127` times a small fp8), torch-CPU and the
Triton interpreter round differently — 120 of 256 fp8 codes at scale `0x00`.
That is fp32 subnormal arithmetic, not ue8m0 decoding, and the interpreter is
not evidence about the card either way (same ftz question as 1d).

It is bounded rather than ignored:
`test_subnormal_products_are_the_only_remaining_divergence` asserts the
divergence never escapes that regime, so a future scale-decode defect fires
even though the known regime is tolerated.

---

## 2. FIXED

### 2a. #629 — the Triton cuda-graph replay-prep fills now take the host mirror

#623 left these deliberately, recording: *"the Triton cuda-graph buffer fills,
which have no host mirror in scope"*. That was true of the signatures of
`_update_decode_kv_buffers` / `_update_target_verify_buffers`, and false of the
path that calls them — `init_forward_metadata_out_graph` holds the ForwardBatch.

Plumb: `init_forward_metadata_out_graph` (`triton_backend.py:1407`) →
`_apply_cuda_graph_metadata` (`:2102`, `:2119` — both the capture and the
replay leg) → `_update_decode_kv_buffers` (`:1210`) /
`_update_target_verify_buffers` (`:1282`) → `_dcp_kv_indices`.

**Precisely which read is removed.** Under the EVEN owner rule the graph fills
hand the builder an address-stable buffer, so the `torch.empty(int(dcp_lens.sum().item()))`
sizing branch is never entered — nothing to remove there. Under the WEIGHTED
rule `build_dcp_weighted_kv_indices` derives `total_tokens` from
`int(full_indptr[bs].item())` whenever it is not supplied, buffer or no buffer.
So the weighted replay-prep fill is the site with the sync, and weighted is the
rule this rig runs. Claiming more than that would overstate it.

`_update_draft_extend_buffers` has no DCP branch and no host read — not a
defect, not touched.

The verify mirror goes through `_verify_host_mirror` (`triton_backend.py:97`),
a new shared helper, so the eager site and its graph twin cannot derive the
verify paged length by two different expressions.

### 2b. #629 — the two unwired PrefillWrapper arms

`FlashInferIndicesUpdaterPrefill` has three arms of one exclusive dispatch.
`update_single_wrapper` was wired by #623; `update_sliding_window` (`:6954-6956`)
and `update_cross_attention` (`:7063-7065`) forwarded nothing — both already
*took* `seq_lens_cpu` as a parameter and dropped it.

Two consequences, not one: the weighted-DCP branch fell back to the unbounded
`int(full_indptr[bs].item())`, **and** `call_begin_forward` hard-asserts
`seq_lens_cpu is not None` once `fast_prefill_plan` is installed
(`flashinfer_backend.py:7529-7531`), so either arm would trip that assert under
a captured prefill graph.

Mirrors supplied only where exact, never invented:
- cross-attn wrapper 0 reads `seq_lens` → `seq_lens_cpu`; wrapper 1 reads
  `encoder_lens`, which has no host mirror → `None` (old read kept).
- SWA wrapper 1 (full attention) reads `seq_lens` → `seq_lens_cpu`.
- SWA wrapper 0 ragged: `prefix - clamp(prefix - W, 0)` **is** `min(prefix, W)`
  by identity, computed by `_host_clamp_max` (`:6274`); this also removes the
  `paged_kernel_lens.sum().item()` that sat there.
- SWA wrapper 0 non-ragged: `min(seq, W + seq - prefix)` needs both vectors —
  `_host_swa_paged_lens` (`:6289`) returns `None` if either is missing.
- When `prefix_lens` arrives `None` it is re-derived on the device, so the
  incoming prefix mirror no longer describes it and is dropped (`:6863`).

### 2c. #629 — a second defect in the #623 wiring itself: unchecked mirrors

`dcp_host_lens(mirror, None)` **accepts** the mirror unchecked. `seq_lens_cpu`
is a non-None but STALE slice exactly when `seq_lens_sum` is None (gpu_only
batches; the invariant is stated at `triton_backend.py:1481-1487` and again at
`:1573-1580`). So the eager sites #623 wired passed the pair raw and would size
an index buffer from a stale vector — a silent **mis-size**, which is a worse
failure than the stall it replaced.

New `dcp_fresh_host_lens` (`layers/dcp/layout.py:105`, exported at
`layers/dcp/__init__.py:50`) makes "no sum" degrade to "no mirror" — the old
device read — instead of to "unchecked mirror". Applied at the eager decode
site (`:1582`), the eager verify site (`:1715`) and the graph entry point
(`:1407`). Deliberately NOT applied to the `extend_prefix_lens_cpu` site
(`:1794`): that mirror is unset rather than stale on gpu_only batches, and its
comment already says no second sum exists to check it against.

`layout.dcp_host_lens` semantics are unchanged — the guard is additive, so no
#623 caller shifted underneath.

The same guard is applied to the mirrors the two newly wired prefill arms use
for SIZING (`flashinfer_backend.py:6870`, `:7026`), so 2b does not introduce a
fresh instance of the hazard 2c removes. The mirror those arms FORWARD stays
raw, matching `update_single_wrapper`: `fast_prefill_plan` asserts it is not
None (`:7529-7531`), and dropping it there would convert a slow path into a
hard failure. `update_single_wrapper` has the identical sizing exposure and is
left as #623 wrote it — a small open item, listed here rather than changed
under an unrelated ticket.

### 2d. #418 — the dequant reference no longer flushes ue8m0 `0x00`

`dequantize_k_cache_paged_ref` (`dsv4/dequant_k_cache.py:191-195` before the
fix) decoded the scale and then ran an explicit FTZ emulation:

```python
scale_pow2 = torch.where(scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2)
```

`exp2(e-127)` falls below `2**-126` for exactly one encoding, `e == 0`, so the
clause fired on byte `0x00` and nothing else. The kernel (`:138-139`) has no
counterpart, so the two disagreed on every 0x00-scale input — the oracle lying
in exactly one regime, #380-class.

**The reference is the wrong side, established against the spec and not against
the kernel.** ue8m0 is OCP MX v1.0 §5.4.1 E8M0: unsigned exponent only, no
zero, no subnormals; 0..254 are `2**(e-127)`, 255 alone is NaN. `e == 0` is an
ordinary legal code for `2^-127`; only the fp32 value it decodes to is
subnormal, which is what the clause confused it with. torch's own decoder
special-cases it for that reason — `torch/include/torch/headeronly/util/Float8_e8m0fnu.h:132-134`:

```cpp
// if exponent is zero, need to special case to return 2^-127 instead of zero
if (x == 0) { return c10::detail::fp32_from_bits(0x00400000); }
```

Verified at runtime on the installed torch 2.11.0: byte 0 → `5.877471754111438e-39`
(`== 2**-127`), byte 255 → `nan`. Fix = delete the clause; provenance of the
clause is upstream (`93173b27e8a`), not a fork edit.

### 2e. #418 — the instrument that had been narrowed to hide it

`test_dsv4_fp8_triton_compat_417.py:212-218` drew scale bytes from `[110, 141)`
and its comment named the reason: *"the reference flushes subnormal scales to
zero, the kernel does not"*. With the defect fixed that bound is gone; the
window is now `[10, 246)` — 236 of 256 encodings, up from 31 — bounded by fp32
range instead of by a defect, with the arithmetic for both ends written out at
the generator. Leaving the narrow window would have meant fixing the bug and
keeping the blindfold.

Also updated: `test_dcp_index_host_sum_623.py:578` asserted by name that *"the
cuda-graph callers pass no mirror"*, which 2a makes false. It now pins the
DEFAULT (absent a mirror, the device read stands), which is still true.

---

## 3. TESTS

New, both hermetic — no CUDA, no process group, no model:

- `test/srt/distributed/test_dcp_replay_prep_mirror_629.py` — **20 passed**
- `test/registered/unit/layers/attention/test_dequant_k_cache_subnormal_scale_418.py`
  — **8 passed, 641 subtests** (registered CPU CI, `base-a-test-cpu`)

Throughout the #629 suite the device vector and the host mirror carry
DIFFERENT numbers (41214 vs 39166) and the assertion is that the result follows
the MIRROR. Had they been equal, every test would pass on the unfixed tree by
coincidence.

### Can-fail proof — every fix reverted, each against the test that pins it

| revert | result |
|---|---|
| R1 entry point drops the mirror | 2 failed |
| R2 decode fill drops `lens_cpu` | 3 failed |
| R3 verify fill drops `lens_cpu` | 2 failed |
| R4 freshness guard neutered | 2 failed |
| R5 #418 flush restored | 132 failed |
| R6 prefill-arm sizing guard removed | 1 failed |

All six detected; files byte-identical after restore (`diff -q`). The #418
falsifier was additionally run RED before the fix existed: `0.0 != 5.877471754111438e-39`.

The two census pins carry their own can-fail proofs in-file
(`test_the_census_can_actually_fail`, and the `_subnormal_product_codes`
exclusion is guarded by `assertGreater(checked, 120)` so it cannot swallow the
corpus).

### Suites

| suite | result |
|---|---|
| #631 flip family (`scripts/run_631_flip_family.sh`) | **1116 passed / 0 failed** — identical before and after |
| `test_dcp_index_host_sum_623.py` (#623/#616h) | 23 passed, unmoved |
| `test_dsv4_fp8_triton_compat_417.py` (widened) | 12 passed, 262 subtests |
| `test/registered/unit/layers/attention` (whole dir) | 190 passed, 9 skipped, 1121 subtests |

`test/registered/unit/distributed` + `test/registered/dcp` + `test/srt/distributed`
+ `test/registered/cp`, same command on base (`c2ceac7f31`) and branch:

    base:   23 failed, 2727 passed, 16 skipped, 52 errors, 971 subtests
    branch: 21 failed, 2749 passed, 16 skipped, 52 errors, 971 subtests

The FAILED/ERROR **name lists** differ by exactly two entries, both present on
base and absent on branch:
`test_bar1_host_cleanup.py::TestPgrepSelfMatchTrap::test_the_checking_shell_does_not_count_itself`
and `::TestStaleBerichtDoesNotSurviveACleanPass::test_a_stale_report_does_not_survive_a_clean_pass`.
Both shell out and inspect the live process table, so they are sensitive to
what else is running on the box; nothing in this diff touches bar1 host
cleanup. **No new failure, in either direction, from this branch.** The +22
passed is the 20 new #629 tests plus those two.

MEASUREMENT NOTE, recorded because it cost a cycle and will recur: a first
branch run showed 7 "new" failures, all of them `inspect.getsource`-based
source-scanning tests returning the body of the WRONG function. Cause was mine,
not the code — I edited `flashinfer_backend.py` while that run was in flight,
so `co_firstlineno` no longer matched the file on disk. Source-scanning pins
(this tree has many) make a test run non-reproducible if the tree moves under
it. Freeze the tree for the duration, and md5 it before and after to prove it
stayed frozen.

ruff, per changed file, base vs branch: `flashinfer_backend.py` 30 / 30 (all
pre-existing), every other changed file 0 / 0, both new files 0.
`ruff format`: `layout.py` and `dequant_k_cache.py` were formatted at base and
still are; the four files unformatted at base are unchanged in that respect;
both new files are formatted.

---

## 4. NEXT

1. **1a is the next shift's red-first job, and the only silent-wrongness item
   in this handoff.** Everything else here is a stall or a lying oracle; 1a can
   produce WRONG WINDOW INDICES on the target-verify graph path. Order: write
   the translate-counting test FIRST and watch it go red (fake
   `token_to_kv_pool.translate_loc_from_full_to_swa` incrementing a counter,
   drive `_update_target_verify_buffers` then
   `_translate_cuda_graph_shared_pool_locs`, assert the window is translated
   ONCE); only then apply the one-line `skip_full_to_swa_translation` fix
   mirroring `triton_backend.py:1237`. It was deliberately left unfixed this
   shift — a one-line change to translation behaviour with no failing test
   under it is exactly the move that turns a stall into a wrong answer.
2. Wire `:1539` (pad the mirror first) and `:1509` (new field on the replay
   view), then add the `.update()`-layer census that 1c describes.
3. Run the 1d PTX recipe in any GPU window; it costs seconds and settles
   whether the kernel is spec-conformant on-device.
4. 1e's `flash_mla_sm120` pair is a clean, self-contained follow-up ticket.
