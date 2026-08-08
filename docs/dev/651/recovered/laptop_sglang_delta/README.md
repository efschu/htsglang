# Recovered: the laptop's sglang_src delta beyond gguf.py

**Status: RECOVERED VERBATIM, NOT YET RECONCILED.** Companion to
`../gguf.py.laptop.patch` and `sgl-kernel/recovered-rocm-gguf/`. The first
recovery pass captured only `gguf.py`; this directory completes it.

## Provenance

Recovered 2026-08-08 (phase 2 of #651) from `/root/lh/sglang_src` on the
laptop `efeu-TP14` (192.168.0.116), which is **not a git repository**. Laptop
originals untouched; every `.laptop` copy here verified byte-identical against
the laptop by SHA-256 (`SHA256SUMS.laptop`).

How the set was determined: the tree was extracted from a tgz at
2026-08-07 10:53:59, so any file with a later mtime was touched afterwards.
A batch of 12 files stamped 12:05:17 is an overlay copy whose blobs all exist
verbatim in this repository's history (verified via `git hash-object` /
`git cat-file -e`) — an older branch state, no unique work. The remaining
individually-edited files are this set, plus the already-recovered `gguf.py`.
`sglang/_version.py` is generated, carries `commit_id = None`, and confirms
the tree has no baked commit id.

## The seven unique files and what each edit does

| File | Laptop edit |
|---|---|
| `model_config.py.laptop` | adds `"gguf"` to the ROCm quantization allow-list; the functional refusal moves to `GGUFConfig.supports_current_device()` |
| `loader.py.laptop` | `_release_gguf_loader_arena`: `malloc_trim(0)` after the GGUF load returns freed host pages to the kernel. Measured: 9,455 MiB of 11,926 MiB RSS released. On unified memory this is the same DRAM the GPU allocates from |
| `layer.py.laptop` | GGUF MoE **early expert-stack materialization** (`_gguf_try_early_materialize` + `_gguf_fill_expert_stack` + lock): materialize each `[E,...]` parameter the moment its last expert arrives, bounding host residency to one parameter's experts. Without it Q4_K_M (21.11 GiB) was OOM-killed inside `load_weights` at ~15.5 GB RSS before the post-load hook ran. Narrow: plain fully-resident path only (#82/#123/#391c keep old timing) |
| `topk.py.laptop` | `_aot_topk_available()` probe: ROCm sgl-kernel build has no MoE routing kernels, so `fused_topk` would raise `ModuleNotFoundError` on the FIRST forward; falls back to `torch_native` routing |
| `moe_align_block_size.py.laptop` | same probe pattern for `moe.moe_align_block_size` AOT kernel; falls back to the JIT kernel, which does build for HIP |
| `moe_align_kernel.cu.laptop` | HIP compile fix for that JIT kernel: `__shfl_*_sync` lane mask is 64-bit on HIP (`static_assert` otherwise); `sgl_shfl_mask_t` + `SGL_FULL_WARP_MASK`, CUDA keeps its 32-bit mask |
| `weight_utils.py.laptop` | the laptop-side variant of the #647 router-gate dense rescue (`gguf_is_dense_unquantized_target`, suffix-matched `ALWAYS_DENSE` set, dense F16/BF16 pass-2 yield). NOTE: this branch already carries an in-tree #647 fix (`0155ff2c00`); the two implementations overlap and MUST be reconciled deliberately, not merged blindly |

## Reconciliation

Tracked with #651 work-queue items 5-6 (HANDOFF §7). The boots of phase 2 run
against the laptop tree, i.e. against these versions. `weight_utils.py` is the
one file where laptop and branch have independently-written fixes for the same
bug; the other six are pure additions relative to the branch.

Patches (`*.patch`) are against branch state 9b0de1f349; the `.laptop` files
are the byte-exact originals and take precedence if a patch fails to apply.
