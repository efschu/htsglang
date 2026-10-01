#!/bin/bash
# efeu-TP14: llama.cpp with the HIP backend for the 780M, as the PP2 test vehicle.
# llama.cpp's partial offload (-ngl k) IS a two-stage pipeline over iGPU and CPU
# (layers [0, 41-k) on the CPU, the rest on the iGPU, run in sequence for one
# request) with mature CPU kernels for every op of this model, including the
# gated delta net. Built for gfx1100 (system rocBLAS ships gfx1100, run with
# HSA_OVERRIDE_GFX_VERSION=11.0.0) and WITHOUT real-true16 -- the clang-21
# miscompile found today applies to ggml's HIP kernels just as to ours.
set -eu
SRC=/root/651-p2/llama.cpp
B=/root/efeu35q3/llama-hip-build
cmake -S $SRC -B $B -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_FLAGS="-Xclang -target-feature -Xclang -real-true16" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON > /root/efeu35q3/logs/llama_hip_cmake.log 2>&1
nice -n 5 cmake --build $B -j 14 --target llama-bench llama-server > /root/efeu35q3/logs/llama_hip_build.log 2>&1
echo "LLAMA-HIP BUILD rc=$?"
ls -la $B/bin/llama-bench $B/bin/llama-server
