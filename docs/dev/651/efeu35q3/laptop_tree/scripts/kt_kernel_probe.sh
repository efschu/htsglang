#!/bin/bash
# #655: does kt_kernel offer a usable CPU-expert path on this Zen4 APU laptop?
#
# The open question from the handoff was whether kt_kernel is AMX-gated. It is
# not, but the reason matters: kt_kernel ships SIX prebuilt variants and picks
# one at runtime (amx -> avx512_bf16 -> avx512_vbmi -> avx512_vnni ->
# avx512_base -> avx2). Only the DEFAULT method AMXINT4 is Intel-only. This
# CPU (Ryzen 7 PRO 8840HS, Zen4) has avx512_bf16/vbmi/vnni and no amx_tile, so
# the loader selects avx512_bf16 -- the highest non-AMX tier.
#
# Two packaging constraints decide whether this is reachable at all, and both
# were nearly fatal:
#
#   1. kt-kernel publishes cp311 and cp312 wheels and NO source distribution.
#      The laptop's SYSTEM python is 3.14, which is the same wall that blocked
#      aider. The serving venv, however, is python 3.12 -- so the wheel fits
#      the interpreter that actually matters.
#
#   2. Installing normally drags in a CUDA torch 2.9.1, which would SHADOW the
#      venv's ROCm torch 2.10 the moment the target dir joined PYTHONPATH and
#      break serving outright. --no-deps is therefore not tidiness, it is the
#      difference between a probe and an outage. Keep it.
#
# Weights: the AMX methods need weights converted by convert_cpu_weights.py,
# but LLAMAFILE consumes GGUF directly -- which is the format this checkpoint
# already is, so that conversion step does not apply here.
set -u

VENV_PY=${VENV_PY:-/root/lh/venv/bin/python}
KTK_DIR=${KTK_DIR:-/opt/ktk}

echo "=== interpreter ==="
"$VENV_PY" -V
echo "=== CPU features that decide the variant ==="
grep -o -m1 -E "avx512f|avx512bw|avx512_bf16|avx512vbmi|avx512_vnni|amx_tile" /proc/cpuinfo | sort -u | tr '\n' ' '
echo
echo "=== kt_kernel import + selected variant ==="
cd /tmp || exit 1
PYTHONPATH="$KTK_DIR" "$VENV_PY" - <<'PY'
import kt_kernel
print("version      =", getattr(kt_kernel, "__version__", "?"))
print("cpu_variant  =", kt_kernel.__cpu_variant__)
from kt_kernel import KTMoEWrapper
print("KTMoEWrapper = importable")
from kt_kernel.experts import INFERENCE_METHODS
print("inference methods =", sorted(INFERENCE_METHODS))
PY
