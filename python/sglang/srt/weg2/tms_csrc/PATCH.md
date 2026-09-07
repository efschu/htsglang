# Vendored torch_memory_saver 0.0.9.post1 csrc (MIT, fzyzcjy) with TWO patches

Source: the PyPI sdist `torch_memory_saver-0.0.9.post1.tar.gz` (the version
installed in /spinning/htsglang-gpu/.venv, a binary wheel that ships no csrc).
Only the CUDA preload hook is built from this copy (`hardware_amd_support.h`
is not vendored: `USE_CUDA` only).

## Patch 1 -- #1233, one backup

(core.cpp, `TorchMemorySaver::resume`): after the H2D restore of a
cpu-backed allocation the pinned host image is `cudaFreeHost`-ed and the
pointer nulled.  Upstream keeps it ("currently keep it there to reduce re-alloc
time"), which is DR-1: every group that has slept once holds its whole weights
image in host RAM for the rest of its life, so two groups hold two images at
the flip (boot weg2ls1b2 2026-09-07: host OOM).  `pause()` already handles a
null pointer (it re-allocates), so nothing else changes.

Built by scripts/weg2/tms/build_tms_preload.sh into a file whose name carries
`torch_memory_saver` (HookUtilModePreload.get_path_binary asserts that on the
LD_PRELOAD value) and the sha256 of these sources; the launcher points
SGLANG_WEG2_TMS_PRELOAD_SO at it and the adapter's configure_subprocess()
honours that variable.  Unset, the stock wheel's hook is preloaded unchanged.

## Patch 2 -- #1235, the shared host granule ring (WEG2_FLIPCOST_SPEC_0907 C1-C7)

New file `host_ring.{h,cpp}`; `core.{h,cpp}` and `entrypoint.cpp` changed.

`cudaMallocHost` / `cudaFreeHost` per allocation is REPLACED (not wrapped) by
`acquire` / `release` against ONE per-card region of `H(c) = max_g image_g(c)`
bytes, shared by both co-located rank processes.  Two groups flipping weights in
opposite directions on the same card no longer need two private pinned images
(61.4 GiB, refused in record 1h) -- they need one region, and `acquire` BLOCKS
on the peer rather than falling back to a second allocator the ledger cannot
price (spec R4/R5).  The copies move from a per-allocation `cudaMemcpy` on the
legacy default stream to `cudaMemcpyAsync` on ONE `backup_stream_` with ONE
`cudaStreamSynchronize` per leg; that stream is created with
`cudaStreamCreate` -- DEFAULT/blocking flags, never `cudaStreamNonBlocking`,
which would silently drop the ordering against PyTorch's default stream (R12).

Two C entrypoints are added: `tms_tag_bytes(tag)` (the per-tag byte instrument,
because the previous one -- an RssShmem delta -- is dead by construction once
the granules are shared pages, R8) and `tms_ring_stats(...)`.

The switch is the presence of `TMS_HOST_RING_DIR` / `_MAP` / `_EPOCH` / `_FORM`,
all four published by `weg2/launcher.py` and none of them operator-facing; unset,
this file behaves exactly as patch 1 left it.  `_FORM` selects the cross-process
registration form (`MAP_SHARED` file, or `memfd_create` + fd inheritance) --
whether `cudaHostRegister` accepts either is a metal fact, so a ring directory
published without a proven form is refused by name (W33 Weg2RingFormUnproven)
rather than guessed at.
