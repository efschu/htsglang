# Task #417 — sm86 and sm120 paths for the DSV4 attention backend

Branch `feat/dsv4-sm86-sm120-417`, based on `integration/r3-probe-next2` at
`a56f33aafd`. Desk-only phase: no GPU, every python invocation under
`CUDA_VISIBLE_DEVICES=99`.

Ground truth is boot 11
(`/spinning/gpu-battery-results/2026-08-01_391_dsv4flash11/WALL.txt`), running
DeepSeek-V4-Flash GGUF, 43 layers, on the 5090 (sm120) + 2x 3080 (sm86) rig:

> **Correction (#447).** Earlier revisions of this plan called the sparse
> layers "DSpark sparse layers 40-42". Both halves were wrong and the error
> understates the indexer's share of a forward pass by ~7x. `compress_ratios`
> in the shipped `config.json` of DeepSeek-V4-Flash-0731 is
> `[0, 0, 4, 128, 4, 128, ..., 128, 4]` + `[0, 0, 0]` (46 entries: 43 trunk
> layers plus the 3 DSpark draft stages). So the trunk has **21 CSA(4) layers**
> (the even indices 2..42, each with an indexer), **20 HCA(128) layers** (odd
> indices 3..41) and only layers 0-1 SWA-only. `40-42` is
> `dspark_target_layer_ids` — the layers whose hidden states feed the DSpark
> draft head — and has nothing to do with compression. Wording below is left
> as written; read "DSpark sparse layers 40-42" as "the 41 compressed layers".

* TP0 / sm120 — `RuntimeError: Sparse Attention Forward Kernel is only
  supported on SM90a and SM100f architectures.`
* TP1,TP2 / sm86 — `triton.compiler.errors.CompilationError: type fp8e4nv not
  supported in this architecture` out of `dsv4/dequant_k_cache.py`.

---

## Phase 0 — upstream PR check (standing rule: before any code)

Searched `sgl-project/sglang` (open + merged PRs and issues) for DSV4 on
non-Hopper, sparse-attention fallbacks and `dequant_k_cache` portability.

### The sm120 half is already solved upstream — adopt, do not rebuild

| Ref | State | Bearing |
|---|---|---|
| issue #31578 | open | Exactly our TP0 failure, same error text, `--attention-backend dsv4` on SM120. |
| PR #30272 | **merged** | Adds `and not _is_sm120` to the sparse-prefill predicate in `deepseek_v4_backend.py`, so SM120 falls through to the dense `flash_mla_with_kvcache_sm120` path. **Our base predates it** (`git log -S"not _is_sm120"` on that file is empty — this is not a fork regression, just an old base). |
| PR #32464 | open | Test-side counterpart: the registered DSV4 attention test must expect the dense path on SM120. Confirms the dense fallback is the *intended production behaviour* on SM120, not a workaround. |
| PR #24692 | merged | Origin of `flash_mla_with_kvcache_sm120` (already in our base). |
| PR #29927, #31481, #32779, #33075 | open | Further SM120 DSA/DSV4 work (flashinfer packed sparse, Triton sparse-MLA prefill). Adjacent, none of it required for our TP0 fix. |

**Verdict for sm120: no parallel build.** Our delta is to take upstream
#30272's gate, generalised (see Cut 2) so it also covers sm86 instead of
naming one architecture.

### The sm86 half has no upstream implementation, only a converging bug report

| Ref | State | Bearing |
|---|---|---|
| issue #33194 | open, filed 2026-08-01 | DeepSeek-V4-Flash-0731 on 8x A800 (SM80). Same failure chain as our TP1/TP2, reported independently. Three blockers: (A) `topk_v2` JIT uses `cg::this_cluster()` (SM90+), workaround `SGLANG_OPT_USE_TOPK_V2=0`; (B) FlashMLA decode has no Ampere path though SM120 is in the identical position; (C) `fp8e4nv` unavailable to Triton on SM80. |
| PR #28620 | open | SM89/L20 FP8 indexer fallback. Nearest precedent in shape (explicit source-level capability gating rather than env-var workarounds), but SM89 *has* `fp8e4nv`, so it does not solve (C). |
| PR #31480 | open | Arch-independent torch paged-MQA-logits backend for the generic `dsa/` path. Same idea one package over; the `dsv4/` indexer already has `fp8_paged_mqa_logits_torch_sm120`. |
| PR #32431 | **closed** | "SM86/SM80 Ampere GPU support" — a patch-file bundle (SM86→SM90 binary fallback for `sgl-kernel`) unrelated to the V4 attention kernels. Closed without merge. |

**Verdict for sm86: build, no upstream path exists.** #33194 is a bug report
with a throwaway local patch, not a PR. Two things from it are worth taking
verbatim as design input:

1. The compile failure is triggered by the *kernel argument*, not the load —
   handing a `torch.float8_e4m3fn` tensor to a `@triton.jit` kernel fails on
   SM80/86 even when the pointer is never dereferenced, and the error points
   at the `def` line. An in-kernel branch cannot fix this; the argument has
   to be a `uint8` view.
2. The reporter's chain ends at `is_fp4_experts=True` → `Hidden size
   mismatch` in `fused_moe`, i.e. FP4 routed experts that Ampere cannot run.
   **That terminal blocker does not apply to us**: we serve a GGUF checkpoint
   through the fork's GGUF/MoE-offload path, not the official FP4 one. This
   is the single reason our rig can get further than #33194's, and it is the
   thing to re-check first if the GPU window ends in a MoE assert.

Also relevant from the fork's own history: #340 hit `type fp8e4nv not
supported in this architecture` on the dual-group lane host, and #343
(`c3c019ab7f`) fixed the *capability-answering* half of it — every probe is
now per-device (`per_device_gate`, `resolve_capability_device_id`). #343 made
the right architecture reachable; #417 is what that architecture then needs
to actually have.

---

## Per-architecture coverage matrix

Rows are the kernels the DSV4 backend can reach. "n/a" means the path is not
selected on that architecture.

| # | Site | SM90a | SM100f | sm120 | sm86 (before) | sm86 (after) |
|---|---|---|---|---|---|---|
| 1 | `deepseek_v4_backend.py:1660` sparse-prefill gate | taken | taken | taken → crash | taken → crash | **not taken** |
| 2 | `_forward_prefill_sparse` → `flash_mla_sparse_fwd` | ok | ok | RuntimeError | RuntimeError | n/a |
| 3 | `_forward_prefill_sparse` → `dsv4/dequant_k_cache.py` | ok | ok | ok | **CompilationError** | fixed (Cut 1) |
| 4 | `deepseek_v4_backend.py:1674` dense gate | else → CUDA FlashMLA | else → CUDA FlashMLA | `flash_mla_with_kvcache_sm120` | else → CUDA FlashMLA, `Unsupported architecture` | **`flash_mla_with_kvcache_sm120`** |
| 5 | `flash_mla_sm120.py` backend choice | n/a | n/a | flashinfer (env default) | n/a | **triton** (flashinfer is cc-gated to 120/121) |
| 6 | `flash_mla_sm120_triton.py:_tiled_sparse_decode_kernel` | n/a | n/a | ok | would CompilationError | fixed (Cut 1) |
| 7 | `dsv4/quant_k_cache.py` KV write (fp8 encode) | ok | ok | ok | **CompilationError** | fixed (Cut 1b) |
| 8 | `dsv4/index_buf_accessor.py` KV write (fp8 byte copy) | ok | ok | ok | **CompilationError** | fixed (Cut 1b) |
| 9 | `dsv4/indexer.py` paged-MQA-logits selection | deepgemm | deepgemm | deepgemm → **crash** | deepgemm → crash | **torch** (Cut 3) |
| 10 | `dsv4/metadata.py` deepgemm indexer schedule | ok | ok | built → crash | built → crash | **skipped** (Cut 3) |
| 9b | `dsv4/indexer.py` non-paged indexer (`fp8_mqa_logits`) | ok | ok | eligible → crash | eligible → crash | **not eligible** (Cut 3) |
| 9c | `dsv4/indexer.py` FP4 indexer | ok | ok | crash | crash | **named refusal** (Cut 3) |
| 11 | MoE `topk_v2` JIT (`cg::this_cluster()`) | ok | ok | ok | JIT compile error | `SGLANG_OPT_USE_TOPK_V2=0`, launch-side |

Rows 3, 6, 7, 8 are one bug with four call sites — the V4 KV layout is FP8
E4M3 **by architecture** (448 fp8 nope + 64 bf16 rope + 7 ue8m0 scale bytes
per token, all inside a `uint8` pool). `--kv-cache-dtype` cannot move it:
`deepseek_v4_memory_pool.py:117` asserts `store_dtype == torch.uint8`, which
is also why `--kv-cache-dtype bfloat16` (admitted by the whitelist in
`arg_groups/overrides.py::_deepseek_v4_kv_cache_dtype`) fails on every GPU,
not just Ampere.

Note that rows 3 and 6 are *alternatives*: after Cut 2, sm86 leaves the
sparse-prefill branch, so row 3 stops being on the default sm86 path and row
6 becomes the load-bearing one. Row 3 is still fixed, because it is reachable
whenever sparse prefill is deliberately re-enabled and because both sites
share one decode helper.

### Why not the existing Triton MLA backend (the V2/V3-on-Ampere precedent)

Checked and rejected as the sm86 route. `TritonAttnBackend`
(`layers/attention/triton_backend.py:614`, MLA is a mode of it via
`use_mla`) contains no `kv_cache_dtype` handling at all — it hands
`get_key_buffer()` straight to `tl.dot(q.to(k.dtype), k)`. It expects the
`MLATokenToKVPool` bf16 layout (512 + 64), not V4's packed 584-byte token.
Routing sm86 there would mean re-implementing the V4 pool for one
architecture. The `flash_mla_sm120` fallback family already speaks the V4
layout and already has a Triton and a pure-torch backend; sm86 is in exactly
the position SM120 was in (no FlashMLA CUDA kernel), which is the same
reading #33194 arrived at independently. That is the route.

Also checked: there is **no per-rank attention-backend selection** anywhere
in the tree — `ServerArgs.get_attention_backends()` splits prefill vs decode,
never rank vs rank, and `ModelRunner._get_attention_backend` applies one
string fleet-wide. So the mixed-group case must be solved *inside* the one
`dsv4` backend by per-device dispatch, not by giving rank 0 a different
backend name. This is what #343's `per_device_gate` was built for.

### Precedents reused

* **#262** (`test/registered/unit/layers/attention/test_fp8_e4m3_kv_decode.py`,
  commit `8a43310d23`) — proves fp8-e4m3 KV bytes decode bit-exactly on sm86
  through torch, and pins that Triton's NVIDIA backend admits `fp8e4nv` only
  from capability 8.9. Two things are taken from it: the reference is a
  **pure-Python IEEE-754 decoder that never touches a torch fp8 type** (a
  torch-based reference would be wrong in the same way as the code under
  test and pass vacuously), and every assertion is paired with a **cross
  probe** that the same bytes read as e5m2 deviate, so a vacuous pass is
  impossible.
* **#192** (`fp8_utils.py:285,339`) — the pairing invariant: a gate that
  switches a fast path *off* must arm its fallback in the same call, or a
  checkpoint is left with no usable path at all. Cut 1 follows it: the
  capability gate returns the tensor view and the `tl.constexpr` flag
  together (`nope_cache_view`), so a caller cannot take one without the
  other.
* **#401** (`test/registered/unit/models/test_deepseek_gguf_construction.py`,
  commit `f6b752a520`) — hermetic construction smoke: `register_cpu_ci`,
  `ServerArgs(device="cpu")` published through `get_context()`, a single-rank
  gloo world, and every construction inside `with torch.device("meta")`.
  Cut 2's backend-construction smoke follows it.

---

## Cut 1 — architecture-portable FP8 E4M3 *decode*

New module `python/sglang/srt/layers/attention/dsv4/fp8_triton_compat.py`:

* `triton_fp8e4nv_supported(device_id)` — `per_device_gate`, true from
  capability (8, 9); non-CUDA vendors stay native (ROCm stores E4M3-**FNUZ**,
  bias 8, a different encoding this module does not decode).
* `nope_cache_view(cache_uint8, fp8_dtype) -> (tensor, fp8_native)` — the
  #192-style pairing: returns the fp8 view **and** True, or the uint8 view
  **and** False. Never one without the other.
* `e4m3fn_bits_to_f32(raw)` — `@triton.jit` device function, pure
  integer/float arithmetic.

Applied to:

* `dsv4/dequant_k_cache.py` — `FP8_NATIVE: tl.constexpr` selects
  `tl.load(...).to(tl.float32)` or `e4m3fn_bits_to_f32(tl.load(...))`; the
  host passes `buf_uint8` in the nope slot on the fallback.
* `flash_mla_sm120_triton.py` — same treatment for
  `_tiled_sparse_decode_kernel`'s `cache_fp8_ptr`.

Falsifier (hermetic, `CUDA_VISIBLE_DEVICES=99`, Triton interpreter mode):
all 256 byte values decoded by the actual `@triton.jit` device function and
compared to the #262-style pure-Python IEEE-754 reference, bit-for-bit on the
float32 pattern, with NaN codes compared as NaN and `-0.0` at 0x80 required
to stay signed. Cross-probe: the same bytes read as e5m2 must deviate.
Can-fail proof: perturbing one mantissa bit must fail the test.

## Cut 1b — the *write* side, same bug class

`_forward_prefill_sparse` is not the only fp8-typed kernel on the V4 path;
the KV writers are too, and they run *before* any attention. Fixing only the
read side moves the crash rather than removing it.

* `dsv4/index_buf_accessor.py::_set_k_and_s_triton_kernel` — pure byte copy,
  loads fp8 and stores fp8 with no arithmetic in between. Fixed
  unconditionally by viewing both source and destination as `uint8`: fp8 and
  uint8 are both one byte, all offset arithmetic is already in element units,
  and the result is byte-identical on every architecture. No gate needed, so
  none is added. (The file's `_set_k_and_s_torch` fallback is dead code —
  `SetKAndS.execute` hardcodes `.triton` — and stays dead.)
* `dsv4/quant_k_cache.py::_quant_k_cache_fused_kernel` — a real encode
  (`.to(fp8e4nv)` after clamping to ±448). Rather than hand-roll
  round-to-nearest-even in Triton, the fallback stages the already-clamped
  **float32** value into a temporary and lets torch do the cast
  (`tmp.to(fp8_dtype)`), which on Ampere is the software conversion #262
  already proved bit-exact. Cost is one extra `num_tokens x 448` fp32 buffer
  and one cast pass, on the fallback path only. f32 (not bf16) staging is
  deliberate: bf16 has 8 mantissa bits, so bf16 → e4m3 would double-round.

## Cut 2 — dispatch

Three changes in `deepseek_v4_backend.py` / `flash_mla_sm120.py`, all
expressed as "which kernel exists on this device", never as a list of
architecture names:

1. **Sparse-prefill gate** (`:1660`) — add
   `flash_mla_sparse_fwd_supported(q.device.index)`. The kernel states its
   own domain in its error text (SM90a and SM100f), so the gate is
   `major in (9, 10)`. This is upstream #30272 generalised: it fixes TP0
   (sm120) exactly as upstream does, and covers sm86 with the same line.
2. **Dense gate** (`:1674`) — replace `is_sm120_supported(...)` with
   `not flash_mla_cuda_kernel_supported(...)`, same `major in (9, 10)`
   domain. sm86, sm89 and sm12x all take `flash_mla_with_kvcache_sm120`.
   SM90/SM100 keep the CUDA kernel, XPU keeps its own import. No
   architecture that works today changes branch.
3. **Fallback backend choice** (`flash_mla_sm120.py:202`) —
   `_sm120_default_backend` is read once at import into a module global,
   which is the exact #343 anti-pattern on a mixed rig, and its default
   (`flashinfer`) resolves to a kernel flashinfer gates to compute capability
   120/121. Make the choice per device: on sm12x keep the env value; off
   sm12x, `flashinfer` is not selectable and the resolution falls to
   `triton`. An explicit `SGLANG_SM120_FLASHMLA_BACKEND=torch|triton` still
   applies everywhere — that is a statement about the launch, not a probe
   (#343's rule for `--fp8-gemm-backend`).

Cut 2 also has a test-side half, the same one upstream #32464 makes for
SM120 alone. `dsv4_attention.py`'s compact runner pins
`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` and then asserts `sparse_prefill_cache`
was populated -- an assertion that is now false on every device without
`flash_mla_sparse_fwd`, and that fires *before* the output is compared with
the reference, so a real regression would hide behind a known false failure.
The runner now seeds and asserts against the same predicate production uses.
The env override deliberately stays on, so removing the production gate turns
this test red rather than letting it follow along.

**Is sparse-off lossless?** Both branches consume the *same* indexer output —
the dense branch passes `indices=swa_page_indices, topk_length=...` to
`flash_mla_with_kvcache` just as the sparse branch passes them to
`flash_mla_sparse_fwd`. The difference is the kernel, not the attention
support: no token is dropped and no approximation is introduced. Two
independent confirmations: upstream #32464 calls the SM120 dense route
"production intentionally stays on the dense path", and the fork's own test
kit (`python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`)
compares **both** paths against the same `_pure_torch_dsv4_combined_reference`.
So this is not a refusal case and needs no user-facing warning. It is,
however, listed as a GPU-window gate below rather than asserted from reading
alone.

**DSpark sparse layers 40-42.** The sparse-vs-dense *layer* decision is not
architecture-gated and is not in `dsv4/` at all: `models/deepseek_v4.py:422`
reads `config.compress_ratios[layer_id]`, and `compress_ratio == 0` is the
SWA-only branch that instantiates no `C4Indexer`. A dense equivalent
therefore already exists and is exercised today — `deepseek_v4_dspark.py:92`
asserts `compress_ratio == 0` for the draft model. Forcing `compress_ratio=0`
on layers 40-42 would bypass the indexer entirely, but **that would be a real
approximation** (the layer would stop attending to its compressed history),
so it is deliberately *not* part of this task. The indexer itself is given a
route instead — Cut 3 below.

---

## Cut 3 — an indexer route on cards without DeepGEMM

Cuts 1 and 2 get sm86 through attention; the DSpark layers (40-42, any layer
with `compress_ratio in (4, 128)`) then run an indexer whose paged-MQA-logits
step defaults to DeepGEMM. Matrix rows 9 and 10. Without this cut boot 12
would stop at a wall we already knew about, which is not worth a card window.

Note this is **not an Ampere-only cut**. DeepGEMM declined SM12x, so the 5090
has no paged-MQA-logits kernel either: after Cut 2 gets TP0 through attention,
TP0 would have hit rows 9/10 exactly like the 3080s. All three ranks of this
rig need Cut 3.

Upstream first, as always. **PR #31480** solves exactly this for the sibling
`dsa/` package, and its structure is what is adopted: keep the explicit
backend selections authoritative, add the capability-driven choice under
them, and skip the DeepGEMM schedule metadata for any backend that does not
consume it. The parts that PR had to *build* — a torch implementation and a
metadata bypass — already exist here from #24692, so our delta is the
dispatch half only. That is the honest framing: this cut writes no kernel.

New module `dsv4/indexer_arch.py`:

* `deepgemm_indexer_supported(device_id)` — `per_device_gate`, `major in
  (9, 10)`. DeepGEMM upstream declined SM12x (deepgemm PR #318), and it is
  Hopper-and-up, so Ampere, Ada and consumer Blackwell are all out.
* `resolve_paged_mqa_logits_backend(device_id)` — tilelang / aiter / torch /
  deepgemm. The three explicit environment selections are checked first and
  win everywhere; only the otherwise-default DeepGEMM choice is overridden,
  and only where DeepGEMM cannot run at all.
* `deepgemm_indexer_metadata_needed(device_id)` — the companion for
  `PagedIndexerMetadata.__post_init__`. The two decisions must not be able to
  disagree: a rank with a DeepGEMM schedule and no DeepGEMM, or a DeepGEMM
  kernel and no schedule, is a crash either way. Pinned as an invariant over
  the whole env × capability matrix in the test.
* `warn_torch_indexer_substitution_once(device_id)` — the named warning
  (see "Open risks"; this substitution is *not* bit-identical).

Call-site changes:

* `indexer.py` — the `fn` selection is lifted out of `_forward_paged_indexer`
  into module-level `select_paged_mqa_logits_fn(device, use_fp4_indexer)` so
  the decision is testable on its own. It is the whole of this cut, and it is
  the kind of decision that goes quietly wrong on a heterogeneous group.
* `indexer.py::_can_use_nonpaged_indexer` — the non-paged branch calls
  `deep_gemm.fp8_mqa_logits` directly and has **no** torch twin, so a card
  without DeepGEMM is sent back to the paged path (which is the fallback).
* `metadata.py::__post_init__` — schedule skipped via the companion gate,
  device taken from `c4_seq_lens` rather than device 0 (#343).

Two findings worth recording, both from writing the test:

1. **The FP4 indexer has no fallback and now refuses by name.**
   `fp8_fp4_paged_mqa_logits` is DeepGEMM-only and the FP4 index cache has a
   different layout (68 bytes/head vs 132). Routing an FP4 checkpoint at the
   FP8 torch path would read it wrongly and return plausible numbers, which
   is worse than stopping. The error names the device, its capability, and
   the missing kernel. Our GGUF checkpoint is not affected.
2. **The two torch implementations are not interchangeable, and the
   difference is not architectural.** The paged call site passes `seq_lens`
   with a trailing dim of 1; only `fp8_paged_mqa_logits_torch_sm120` squeezes
   it, while `fp8_paged_mqa_logits_torch` asserts `shape == (batch_size,)`.
   So the old `is_sm120_supported()` split at `indexer.py:667` would have
   sent every non-SM120 card at an implementation that *asserts* — including
   anyone who set `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` on a Hopper box today.
   The dispatch now always selects the trimmed variant, which is also the one
   that walks `ceil(max_seq_len / 64)` pages instead of the full capture-time
   page-table width. `fp8_paged_mqa_logits_torch` stays as the reference
   implementation `test_sm120_paged_mqa_logits.py` compares against.
3. **Consequence of making it the production path (#426).** Because every
   non-DeepGEMM card now lands here, the `[B, S, H]` bmm intermediate that
   upstream sgl-project/sglang#33246 reports (~15 GiB/rank at a 1M context)
   stopped being a fallback-only concern; `fp8_paged_mqa_logits_torch_sm120`
   walks the sequence axis in `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK`-wide
   chunks, bit-identically, since `test_dsv4_indexer_seq_chunk_426.py`.

**Rank uniformity.** TP0 will run `flash_mla_with_kvcache_sm120` while TP1/2
run the same entry point with a different inner backend. Every gate added
here sits inside `forward()` on a per-tensor device question and returns a
tensor of identical shape and dtype on both sides; none of them can
early-return before a collective. Audited against the
rank-local-test-before-collective rule: no `return`, `raise` or `continue`
is introduced upstream of a group collective, and no gate participates in a
control-flow decision that differs in *shape* across ranks — only in which
kernel produces the same-shaped `o`.

---

## Gates for the later GPU window (defined here, not run)

**Request: all three cards (5090 + both 3080s), ~90 minutes.** The window
assumes Cuts 1, 1b, 2 and 3 all in path — every named wall we know of is
closed desk-side, so this window is spent finding the *unknown* one, not
re-confirming a known one.

Prerequisites for every arm:

* resolve the physical index of each card via NVML at runtime; never assume
  0/1/2;
* launch with **`SGLANG_OPT_USE_TOPK_V2=0`** on any boot that includes an
  sm86 rank. Matrix row 11: the `topk_v2` JIT kernel uses
  `cg::this_cluster()` (SM90+) and has no architecture gate, but a working V1
  path is selectable by this flag. Launch-side deliberately — a flag exists
  and works, so this does not need code;
* expect the Cut 3 substitution warning in the log on all three ranks. Its
  *absence* means the DeepGEMM path was taken and the boot is about to fail.

1. **A-vs-A noise floor, same boot, first.** Two identical runs in one boot
   before any comparison is reported. Nothing below this floor gets stated.
2. **sm120 solo reference.** 5090 alone, `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0`,
   dense path. Establishes the reference outputs and the ms/verify +
   ms/prefill baseline. This arm must pass *before* any sm86 arm runs
   (component-alone-before-composite rule).
3. **Sparse-off equivalence on sm120.** Same boot, same prompts, sparse
   prefill on vs off. Confirms the "lossless" claim of Cut 2 on hardware that
   can run both branches. If this diverges beyond the A-vs-A floor, Cut 2's
   gate is a quality change and must be re-framed as such.
4. **sm86 dequant byte gate.** Identical KV bytes through
   `dequantize_k_cache_paged` on the 5090 and on a 3080; the outputs must be
   byte-identical, not close. Same for `quant_k_cache` in the encode
   direction (Cut 1b's torch-staging claim). This is the one gate that
   cannot be run at the desk.
5. **Triton-kernel smoke on sm86.** `flash_mla_sparse_decode_triton` must
   compile and run on a 3080 — the desk falsifier proves the decode
   arithmetic, not that the kernel compiles.
6. **Indexer substitution, sized (Cut 3).** On the 5090, same boot: torch
   paged-MQA-logits vs DeepGEMM cannot be compared (DeepGEMM does not run
   there either), so the measurable question is instead *how far the top-k
   selection moves between two ranks of different architecture on the same
   prompt*. Capture the indexer top-k on TP0 and TP1 for identical inputs and
   report the overlap. This is the number that says whether the warning is
   pedantry or a real quality axis. Also report ms/layer-call for the torch
   path — #31480 measured the equivalent pure-torch kernel at 1.476 ms/call
   at page-table width 131072 because it pays the full capture-time width;
   the trimmed variant we select should not, and that is worth confirming.
7. **TP=3 coherence.** Vector 30,17,17 (or auto), determined-answer probes,
   per-rank ms/verify and ms/prefill (the slowest rank sets the clock).
8. **VRAM corridor** per the standing rule: free >= 400 MiB, no net waste
   > 1.5 GiB. No arm runs on red.

## Desk test results

All runs `CUDA_VISIBLE_DEVICES=99`, venv `/spinning/htsglang-gpu/.venv`,
`PYTHONPATH` pinned to this worktree.

* `test/registered/unit/layers/attention/test_dsv4_fp8_triton_compat_417.py`
  — 12 tests, 262 subtests, green. Runs the *production* `@triton.jit` decode
  under `TRITON_INTERPRET=1`, plus the whole paged dequant kernel against
  `dequantize_k_cache_paged_ref`.
  **Can-fail proven**: changing `bits & 7` to `bits & 6` in
  `e4m3fn_bits_to_f32` (one mantissa bit) turns it into 128 failed / 10
  passed.
* `test/registered/unit/layers/attention/test_dsv4_arch_dispatch_417.py`
  — 16 tests, 28 subtests, green.
  **Can-fail proven**: widening `_FLASH_MLA_CUDA_MAJORS` to `(8, 9, 10)`
  turns it into 7 failed / 15 passed.
* `test/registered/unit/layers/attention/test_dsv4_indexer_arch_417.py`
  — 21 tests, 68 subtests, green. Gate matrix, backend resolution, the
  metadata/kernel agreement invariant over the whole env × capability matrix,
  the named warning, the extracted dispatch, the FP4 refusal, the non-paged
  eligibility, and `PagedIndexerMetadata` constructed on a mocked Ampere card
  without importing DeepGEMM.
  **Can-fail proven, three ways**: `_DEEPGEMM_MAJORS` → `(8, 9, 10)` gives
  22 failed / 17 passed; returning `fp8_paged_mqa_logits_torch` from the
  dispatch gives 8 failed / 21 passed; making the FP4 branch fall back
  silently instead of refusing gives 1 failed / 20 passed.
* Regression: `test/registered/unit/layers/` +
  `test/registered/unit/utils/` — 47 failed / 731 passed / 42 skipped on this
  branch against 47 failed / 682 passed / 42 skipped on the base
  (`a56f33aafd`, checked out into a throwaway detached worktree). The failing
  set is **byte-identical** (`diff` empty); all 47 are pre-existing
  `RuntimeError: No CUDA GPUs are available`. The +49 passes are this task's
  new tests. This comparison earned its keep: it caught Cut 3 breaking
  `test_dsv4_nonpaged_indexer.py::test_eligibility_is_fail_closed`, because
  the first version of the DeepGEMM check reached into a metadata field that
  test's stub does not provide. Asking the *current device* instead is both
  what that test expects and the more correct question after `set_device`.
* ruff: identical findings on the touched files at branch and base
  (`F841 seq_lens_sum` in `deepseek_v4_backend.py`, `E712 clean_logits ==
  False` ×3 in `indexer.py`), all pre-existing; new files clean. codespell,
  black and isort clean. mypy clean on all three new modules
  (`--follow-imports=skip`; the repo-wide run is 10k pre-existing errors).

Not proved at the desk, by construction: that any of these kernels *compiles*
on a real sm86 card. Triton's interpreter never invokes the NVIDIA backend,
which is where `fp8e4nv` is rejected. That is GPU-window gate 5.

## Open risks

* **The indexer substitution is not bit-identical** (Cut 3). Named at boot,
  once per device, and repeated here: DeepGEMM computes the indexer logits as
  an FP8 tensor-core GEMM, the torch implementation dequantises to float32
  and accumulates with `torch.bmm`. Same quantity, different arithmetic. The
  logits feed a top-k, so a near-tie between two KV positions can be broken
  differently, and output on this rig can differ from a Hopper run of the
  same checkpoint. This is the one place in #417 where an architecture is not
  merely given a different kernel for the same numbers.
* **`flash_mla_with_kvcache_sm120` is now a misnomer** — it serves sm86 after
  Cut 2. The name is kept and documented rather than renamed (the fork's
  standing preference for a doc fix over a rename with this blast radius).
* **The torch-staging cast in Cut 1b assumes** torch's f32 → e4m3fn
  conversion is the same round-to-nearest-even-with-saturate as Triton's
  `cvt.rn.satfinite`. Both are documented as such and both go through the
  same CUDA conversion intrinsic, but the claim is only *proved* by GPU-window
  gate 4.
* **Pre-existing kernel/reference divergence at extreme ue8m0 scale bytes,
  found while building the desk test and deliberately not fixed here.**
  `dequantize_k_cache_paged_ref` flushes a subnormal scale to zero
  (`torch.where(scale_pow2 < 2**-126, 0, ...)`, `dequant_k_cache.py:179`);
  the Triton kernel does not. So scale byte 0x00 (2**-127) makes the two
  disagree, and 0xFF (2**128) overflows to inf in both. The file's own
  `__main__` self-test draws scale bytes uniformly from 0..255 and asserts
  `atol=0`, so it would fail on a GPU today for reasons that have nothing to
  do with #417. The new test therefore draws the scale section from the range
  real data occupies (110..140) and leaves the nope bytes fully random. Worth
  a separate ticket; touching it inside an architecture change would make the
  byte gate ambiguous.
* **Perf on sm86 is unmeasured and expected to be poor.** The manual decode
  adds ~8 ALU ops per KV element on a card that is already the slowest in the
  group, and the langsamster-Rang rule says the slowest rank sets the clock.
  This task is about reachability, not throughput; a per-rank ms/verify split
  is the first thing to look at afterwards.

---

## Ticket note — ue8m0 scale divergence in `dsv4/dequant_k_cache.py`

Registered as its own task; **not** to be fixed on this branch.

1. `dequantize_k_cache_paged_ref` flushes a subnormal ue8m0 scale to zero
   (`torch.where(scale_pow2 < 2**-126, 0, ...)`, line 179); the Triton kernel
   at line 125 does not, so scale byte `0x00` (2**-127) yields `0` from the
   reference and ~2.6e-36 from the kernel — representable in bf16, so it
   survives the cast.
2. Scale byte `0xFF` (2**128) overflows to `inf` in both, then `inf * 0`
   gives `NaN`, so the extreme end is ill-defined in both directions.
3. The file's own `__main__` self-test draws scale bytes uniformly from
   0..255 and asserts `atol=0, rtol=0`, so it would fail on a GPU today for
   reasons that have nothing to do with #417. It has evidently not been run
   recently.
4. Nothing here is architecture-specific and nothing was introduced by #417 —
   it is reachable on Hopper exactly as on Ampere.
5. Decide which side is authoritative (the kernel's un-flushed 2**-127 is
   arguably the more faithful decode; the reference's flush matches what the
   ue8m0 *encoder* in `quant_k_cache.py` can actually emit, since it derives
   the exponent from a `max(abs, 1e-8)`-clamped scale and cannot reach 0x00
   in practice), then make both sides agree and fix the self-test's data
   range or its tolerance to match the decision.
