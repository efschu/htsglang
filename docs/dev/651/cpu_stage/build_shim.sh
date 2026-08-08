#!/usr/bin/env bash
# Build recipe for the qmatmul shim (CPU-only, portable to the APU laptop).
#
# Prereq: llama.cpp cloned into ./llama.cpp and built CPU-only:
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
#   cmake -S llama.cpp -B llama.cpp/build \
#       -DGGML_CUDA=OFF -DGGML_HIP=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON
#   cmake --build llama.cpp/build -j8 --target ggml
# Produces llama.cpp/build/bin/{libggml.so,libggml-cpu.so,libggml-base.so}.
#
# The rpath below makes libqmatmul_shim.so self-contained relative to this
# directory; alternatively export LD_LIBRARY_PATH=$PWD/llama.cpp/build/bin.
set -euo pipefail
cd "$(dirname "$0")"

gcc -O2 -shared -fPIC qmatmul_shim.c \
    -I llama.cpp/ggml/include \
    -L llama.cpp/build/bin \
    -lggml -lggml-cpu -lggml-base \
    -Wl,-rpath,"$PWD/llama.cpp/build/bin" \
    -o libqmatmul_shim.so

echo "built $(realpath libqmatmul_shim.so)"
