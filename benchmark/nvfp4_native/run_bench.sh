#!/usr/bin/env bash
# N4B 5090 NVFP4/FP8 micro-bench runner. All JIT caches redirected into the worktree's .jit (never ~/.cache).
set -euo pipefail
WT=/spinning/wt-27b-nvfp4-native
J=$WT/.jit
export PYTHONPATH=$WT/python
export TVM_FFI_CACHE_DIR=$J/tvm-ffi TRITON_CACHE_DIR=$J/triton TORCH_EXTENSIONS_DIR=$J/torch_ext
export FLASHINFER_WORKSPACE_BASE=$J/fi CUTE_DSL_CACHE_DIR=$J/cutedsl TMPDIR=$J/tmp
export TORCHINDUCTOR_CACHE_DIR=$J/inductor XDG_CACHE_HOME=$J/xdg CUDA_CACHE_PATH=$J/nvcache
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
exec /spinning/htsglang-gpu/.venv/bin/python $WT/benchmark/nvfp4_native/bench_5090_nvfp4.py "$@"
