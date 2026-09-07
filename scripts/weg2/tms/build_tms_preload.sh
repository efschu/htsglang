#!/bin/bash
# Build the Weg-2 one-backup torch_memory_saver preload hook (cu13) from the
# vendored, patched csrc (python/sglang/srt/weg2/tms_csrc/PATCH.md).
#
#   scripts/weg2/tms/build_tms_preload.sh [--out-dir DIR] [--venv VENV]
#
# Prints the path of the built .so on stdout.  Idempotent: the output name
# carries the sha256 of the sources, an existing file is reused.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
SRC="$TREE/python/sglang/srt/weg2/tms_csrc"
VENV=/spinning/htsglang-gpu/.venv
OUT_DIR=/spinning/gpu-arb/weg2/tms
while [ $# -gt 0 ]; do
  case "$1" in
    --out-dir) OUT_DIR=$2; shift 2;;
    --venv) VENV=$2; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"
[ -f "$CU13/include/cuda_runtime_api.h" ] || { echo "no cu13 headers at $CU13/include" >&2; exit 1; }
[ -f "$CU13/lib/libcudart.so.13" ] || { echo "no libcudart.so.13 at $CU13/lib" >&2; exit 1; }
STUBS=""
for d in /usr/local/cuda/lib64/stubs /usr/local/cuda-12.9/lib64/stubs "$CU13/lib/stubs"; do
  [ -f "$d/libcuda.so" ] && { STUBS=$d; break; }
done
[ -n "$STUBS" ] || { echo "no libcuda.so stub found for linking" >&2; exit 1; }
SHA=$(cat "$SRC"/core.cpp "$SRC"/core.h "$SRC"/entrypoint.cpp "$SRC"/api_forwarder.cpp "$SRC"/api_forwarder.h "$SRC"/utils.h "$SRC"/macro.h | sha256sum | cut -c1-12)
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/torch_memory_saver_hook_mode_preload_cu13_onebackup_$SHA.so"
if [ -f "$OUT" ]; then echo "$OUT"; exit 0; fi
TMP=$(mktemp -d)
g++ -std=c++17 -O3 -fPIC -shared -DUSE_CUDA=1 -DTMS_HOOK_MODE_PRELOAD=1 \
  -I"$CU13/include" "$SRC/api_forwarder.cpp" "$SRC/core.cpp" "$SRC/entrypoint.cpp" \
  -L"$CU13/lib" -L"$STUBS" -lcudart -lcuda -ldl -o "$TMP/out.so"
mv "$TMP/out.so" "$OUT"; rmdir "$TMP"
echo "$OUT"
