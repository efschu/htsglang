# Vendored torch_memory_saver 0.0.9.post1 csrc (MIT, fzyzcjy) with ONE patch

Source: the PyPI sdist `torch_memory_saver-0.0.9.post1.tar.gz` (the version
installed in /spinning/htsglang-gpu/.venv, a binary wheel that ships no csrc).
Only the CUDA preload hook is built from this copy (`hardware_amd_support.h`
is not vendored: `USE_CUDA` only).

The patch (core.cpp, `TorchMemorySaver::resume`): after the H2D restore of a
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
