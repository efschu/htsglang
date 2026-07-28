#!/bin/bash
# SPDX-License-Identifier: MIT
# Baut ramrelay fuer die drei Architekturen dieses Verbunds:
#   sm_75 Turing (2080 Ti, Rig 2), sm_86 Ampere (3080), sm_120 Blackwell (5090).
# Laeuft im Container -- dort liegt CUDA; /dev/infiniband braucht diese Sonde
# nicht, sie fasst die NIC nicht an.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
OUT="$HERE/bin/ramrelay"
mkdir -p "$HERE/bin"

ARCHS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86"
# sm_120 kennt erst CUDA 12.8+; aelteren nvcc nicht daran scheitern lassen.
if "$NVCC" --list-gpu-arch 2>/dev/null | grep -q compute_120; then
  ARCHS="$ARCHS -gencode arch=compute_120,code=sm_120"
else
  echo "[warn] nvcc kennt compute_120 nicht -- 5090 laeuft ueber PTX-JIT"
  ARCHS="$ARCHS -gencode arch=compute_86,code=compute_86"
fi

"$NVCC" -O3 -std=c++14 $ARCHS -o "$OUT" "$HERE/ramrelay.cu"
echo "[ok] $OUT"
"$NVCC" --version | tail -1
