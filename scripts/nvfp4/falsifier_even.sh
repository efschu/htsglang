#!/usr/bin/env bash
# Falsifier for the ARM 1 stopper, run only if the window has budget left.
#
# THE CLAIM ARM 1 PRODUCED
#   TP0 is the 5090 and takes the NATIVE NVFP4 cutlass lane. It died at the
#   first forward with
#     nvfp4_scaled_mm_kernels.cuh:81: Expected n to be divisible by 32, got 42
#   in in_proj_ba. The arithmetic: linear_num_value_heads = 48, the GDN b/a
#   gate carries two scalars per value head, so the unsharded projection is
#   96 rows. The uneven plan gives rank 0 twenty-one of the 48 heads = 42 rows,
#   and 42 % 32 != 0.
#
#   The #332 posten-1 guard did fire, and correctly: 96 DEQUANTISED lines,
#   48 on TP1 and 48 on TP2, ZERO on TP0. That is right, not a miss -- the
#   guard asks whether the UNSHARDED width has a MARLIN form, and the native
#   rank does not use Marlin. The hole is that no guard asks the native lane's
#   own question, which is about the SHARDED width and the number 32.
#
# THE PREDICTION, REGISTERED BEFORE THE RUN
#   Even TP=3 gives every rank 48/3 = 16 heads = 32 rows, and 32 % 32 == 0.
#   So this boot MUST get past the point where ARM 1 died.
#   The Marlin ranks must STILL emit 96 DEQUANTISED lines, because the guard
#   reads the unsharded 96 and 32 is no multiple of the 64 tile either.
#
#   If it boots  -> the stopper is the uneven plan's row count meeting the
#                   native kernel, not the checkpoint. A width guard on the
#                   native lane fixes it; the dequant lane already exists.
#   If it fails with the SAME n=42 -> the plan is not the source and the
#                   diagnosis above is wrong.
#   If it fails with a DIFFERENT n -> the constraint is real but my row
#                   arithmetic is wrong; the new n names the true geometry.
set -uo pipefail
OUT=/spinning/gpu-battery-results/2026-07-31_332_fam_beleg
source "$OUT/env.sh"

# No --rank-gpu-id and no --rank-tp-ratio: that is what makes the split even.
# Reserve stays off the flag list for the same reason -- --rank-auto-reserve-mib
# is part of the uneven placement path this arm is deliberately not using.
bash "$OUT/arm.sh" v4_tp3_even "$V4_MODEL" "" -- \
  --tp-size 3 \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768
