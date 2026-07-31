#!/usr/bin/env bash
# Falsifier v2 for the ARM 1 stopper. v1 (plain even --tp-size 3) never got to
# ask the question: stock sglang refuses an even split across mismatched cards
# ("The memory capacity is unbalanced", pre_model_load 19.23 vs local 30.55 GB)
# — which is the very thing this fork's uneven TP exists to avoid. So v2 stays
# on the uneven placement path and varies ONLY the ratio, which is the single
# variable the diagnosis is about.
#
# THE MECHANISM UNDER TEST
#   linear_num_value_heads = 48; the GDN b/a gate carries two scalars per value
#   head, so in_proj_ba is 96 rows unsharded. The native NVFP4 cutlass kernel
#   requires the SHARDED output width n to satisfy n % 32 == 0. ARM 1's `auto`
#   plan gave rank 0 twenty-one heads = 42 rows, and it died with n: 42.
#
# TWO PREDICTIONS, BOTH REGISTERED BEFORE THE RUN. The point of running both is
# that one predicts a PASS and the other predicts a SPECIFIC failure number, so
# the pair identifies the mechanism instead of merely being consistent with it.
#
#   A) ratio 2,1,1 -> rank 0 gets 48 * 2/4 = 24 heads = 48 rows. 48 % 32 != 0,
#      so this MUST still die, and it must die with "got n: 48", NOT 42.
#      A different number here means my row arithmetic is wrong.
#      A boot here falsifies the whole diagnosis.
#
#   B) ratio 4,1,1 -> rank 0 gets 48 * 4/6 = 32 heads = 64 rows. 64 % 32 == 0,
#      so this MUST get past the point where ARM 1 died. The Marlin ranks must
#      still emit 96 DEQUANTISED lines, because that guard reads the unsharded
#      96 and neither 16 nor 64 is a multiple of the Marlin tile.
#      B is deliberately still UNEVEN. The even split cannot be used as the
#      control on this rig at all: --rank-tp-ratio 1,1,1 is refused ("identical
#      entries is the even split -- omit the flag instead") and omitting the
#      flag runs into stock sglang's balance guard on mismatched cards. So the
#      control has to be an uneven ratio that happens to satisfy the kernel's
#      constraint, which is exactly what 4,1,1 is.
#
# Together: A pins the constraint to the sharded row count, B shows the uneven
# plan is the only thing standing between this checkpoint and a TP=3 boot.
set -uo pipefail
OUT=/spinning/gpu-battery-results/2026-07-31_332_fam_beleg
source "$OUT/env.sh"

ARM_A="${ARM_A:-1}"
ARM_B="${ARM_B:-1}"

if [ "$ARM_A" = "1" ]; then
  bash "$OUT/arm.sh" v4_tp3_ratio211 "$V4_MODEL" "" -- \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio 2,1,1 \
    --rank-gpu-memory-mib 29607,17780,17780 \
    --kv-cache-dtype fp8_e4m3 --context-length 32768
fi

if [ "$ARM_B" = "1" ]; then
  bash "$OUT/arm.sh" v4_tp3_ratio411 "$V4_MODEL" "" -- \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio 4,1,1 \
    --rank-gpu-memory-mib 29607,17780,17780 \
    --kv-cache-dtype fp8_e4m3 --context-length 32768
fi
