#!/usr/bin/env bash
# #704 slice 1a-i -- BOOT TICKET. Prepared desk-side; NOT self-arming.
#
# WHAT THIS IS
#   A one-variable boot of the PP cut [29,19,16] against the incumbent
#   [28,20,16]. Only --pp-layer-ratio changes. Everything else -- argv, env,
#   commit, GPU mapping, memory pins -- is inherited from the currently
#   serving configuration, so any difference observed is attributable to the
#   cut and to nothing else.
#
# WHY THIS PAIR
#   [28,20,16] and [29,19,16] both have attention profile (7,5,4): layer 28 is
#   a LINEAR (GDN) layer under full_attention_interval 4. So the cut moves one
#   layer of weights from rank1 to rank0 and moves NO attention layer and NO KV
#   row. It is the cheapest possible probe of the layer boundary.
#
#   It is deliberately NOT the [33,15,16] arm, which is retracted: that one was
#   recommended on a pool number (457,604) that exceeded an unchanged rank's
#   measured capacity. See DESIGN_704 E3.
#
# THE FALSIFIABLE PREDICTION, WRITTEN BEFORE THE WINDOW
#   The review gate requires a prediction recorded before booting, that can
#   fail. Ours, from measured quantities only:
#
#     P1. max_total_num_tokens == 436,766, within boot noise (~2k).
#     P2. PP2 is the binding rank, as it is at the incumbent.
#     P3. Per-rank K sizes 2.92 / 2.08 / 1.67 GB, UNCHANGED, because the
#         attention profile (7,5,4) is unchanged.
#
#   Reasoning, and it is structural rather than a rung-table re-derivation:
#   rank2 keeps layers 48-63 byte-identically, so cap2 is unchanged at 436,766.
#   rank1 sheds a layer, so its cap rises. rank0 gains one linear layer, about
#   451 MiB of weights plus ~51 MiB of GDN residency, which lowers cap0 by
#   ~502 MiB / (7 attn x 2048 B) ~ 36,600 tokens. cap0 was clipped at >=550,000,
#   so it lands >=513,000 -- still far above 436,766. PP2 therefore still binds
#   and the pool does not move.
#
#   IF P1 FAILS the corrected pool model is wrong in a way that matters, and
#   that is a more valuable result than a pool win. Record it and stop; do not
#   re-tune and re-boot in the same window.
#
# THE SECOND, INDEPENDENT PRIZE
#   DESIGN_704 flags that PrefillTiming.fixed_ms defaults to zero, the
#   optimistic end of the family, so EVERY speedup in that document is an upper
#   bound rather than a prediction until a second measured cut exists. This
#   boot is that second cut. Capturing the three stage times at [29,19,16]
#   against the incumbent's 49.2 / 154.8 / 116.4 ms pins the per-layer slope
#   against the fixed per-stage intercept for the first time. That result is
#   worth the window even if P1..P3 all hold and nothing else is learned.
#
# WHAT THIS TICKET DOES *NOT* DO
#   It does not exercise the union arena or any runtime rung change. Slice 1a-i
#   is two STATIC boots. The flip actuator (slice 1a-ii) needs runtime wiring
#   that does not exist yet -- see the REQUIRES block below -- and booting a
#   static cut is the precondition that de-risks it, not a substitute for it.
#
# PRECONDITIONS -- ALL REQUIRED, and this script refuses without them
#   1. Slot-2's retro-prediction gate CLOSED: the corrected model reproduces
#      434,878 / 435,822 / 436,766 ([28,20,16]) and 416,796 ([32,16,16]) within
#      boot noise, with the correct binding rank each time.
#   2. Review-gate GO for this arm.
#   3. F4-r4 has the window: he owns the boot queue (front item #703/#706).
#   4. A GPU claim in /spinning/gpu-arb/ with a live heartbeat.
#
#   This script is NOT self-arming. It requires GO=1 in the environment and it
#   verifies the gate markers below. Slice 1a-i is a cheap boot, and cheap is
#   exactly when an unreviewed boot is most tempting.
#
set -uo pipefail

OUT=${OUT:-/spinning/evidence-665-f1}
WT=${WT:-/spinning/wt-704-ladder}
LOG=${LOG:-$OUT/boot_704_1a_29_19_16.log}
RATIO_NEW=${RATIO_NEW:-29,19,16}
RATIO_OLD=${RATIO_OLD:-28,20,16}
PRED_POOL=${PRED_POOL:-436766}
PRED_NOISE=${PRED_NOISE:-2500}

say() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# ---- gate 1: explicit arming -------------------------------------------------
if [ "${GO:-0}" != "1" ]; then
  cat <<'MSG'
REFUSE: this ticket is not self-arming.

  Set GO=1 only after ALL of:
    - Slot-2's retro-prediction gate has CLOSED (four measured points
      reproduced, correct binding rank each time);
    - the review gate has said GO for this specific arm;
    - F4-r4 has released a window (he owns the boot queue);
    - a /spinning/gpu-arb/ claim is held with a live heartbeat.

  Booting a pool-gated arm before the model that predicts it has been
  validated is the exact sequence that cost three boots on 2026-08-16.
MSG
  exit 2
fi

# ---- gate 2: GPU arbitration -------------------------------------------------
CLAIM=${CLAIM:-/spinning/gpu-arb/holder-704-1a}
if [ ! -f "$CLAIM" ]; then
  say "REFUSE: no GPU claim at $CLAIM. Claim the cards before taking serving down."
  exit 2
fi

# ---- gate 3: tree identity ---------------------------------------------------
HAVE=$(cd "$WT" && git rev-parse --short=10 HEAD)
if [ -n "$(cd "$WT" && git status --porcelain)" ]; then
  say "REFUSE: $WT is dirty; a boot must name a commit, not a working tree."
  exit 2
fi
say "tree $WT @ $HAVE"

# ---- capture the CURRENT serving argv, so exactly one variable moves ---------
ARGV_SRC=${ARGV_SRC:-$OUT/armB_argv.txt}
if [ ! -f "$ARGV_SRC" ]; then
  say "REFUSE: no argv source at $ARGV_SRC; refusing to reconstruct a command line."
  exit 2
fi

NEW_ARGV=/tmp/boot_704_1a_argv.txt
: > "$NEW_ARGV"
SWAPPED=0
PREV=""
while IFS= read -r line; do
  if [ "$PREV" = "--pp-layer-ratio" ]; then
    printf '%s\n' "$RATIO_NEW" >> "$NEW_ARGV"
    SWAPPED=$((SWAPPED+1))
    PREV="$line"
    continue
  fi
  printf '%s\n' "$line" >> "$NEW_ARGV"
  PREV="$line"
done < "$ARGV_SRC"

if [ "$SWAPPED" -ne 1 ]; then
  say "REFUSE: expected exactly one --pp-layer-ratio in $ARGV_SRC, swapped $SWAPPED."
  say "        A boot that changes zero or two variables proves nothing."
  exit 2
fi
say "argv prepared: --pp-layer-ratio $RATIO_OLD -> $RATIO_NEW (one variable)"

# ---- record the prediction BEFORE the boot ----------------------------------
PRED=$OUT/PREDICTION_704_1a_${RATIO_NEW//,/_}.txt
{
  echo "arm            : --pp-layer-ratio $RATIO_NEW (from $RATIO_OLD)"
  echo "tree           : $HAVE"
  echo "written        : $(date -u +%Y-%m-%dT%H:%M:%SZ)  (BEFORE the boot)"
  echo "P1 pool        : $PRED_POOL +- $PRED_NOISE tokens"
  echo "P2 binding rank: PP2"
  echo "P3 K sizes     : 2.92 / 2.08 / 1.67 GB unchanged (attn profile 7,5,4)"
  echo "falsifier      : any of P1..P3 missing => the corrected pool model is"
  echo "                 wrong where it matters; record and STOP, do not retune"
  echo "                 and re-boot inside the same window."
} > "$PRED"
say "prediction recorded at $PRED"

# ---- stop serving (targeted, never a broad pkill) ---------------------------
serving_up() { ss -ltn 2>/dev/null | grep -q ':30030 '; }
if serving_up; then
  OLD=$(ss -ltnp 2>/dev/null | grep ':30030 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  say "stopping serving pid ${OLD:-unknown}"
  [ -n "$OLD" ] && kill -TERM "$OLD" 2>/dev/null
  for _ in $(seq 1 30); do serving_up || break; sleep 2; done
  if serving_up; then
    PGID=$(awk '{print $5}' /proc/"$OLD"/stat 2>/dev/null)
    say "drain incomplete; killing process group $PGID"
    [ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
    for _ in $(seq 1 20); do serving_up || break; sleep 2; done
  fi
  serving_up && { say "port 30030 still held; refusing to double-boot"; exit 1; }
fi

say "READY TO LAUNCH. This ticket stops here by design."
cat <<MSG

  The launch line itself is intentionally NOT executed by this script.
  F4-r4 owns the boot queue and the launcher; hand him:

      argv      : $NEW_ARGV
      log       : $LOG
      prediction: $PRED

  ACCEPTANCE (all three, checked against $PRED):
    A1  max_total_num_tokens within $PRED_NOISE of $PRED_POOL
    A2  the binding rank is PP2
    A3  per-rank K sizes unchanged at 2.92 / 2.08 / 1.67 GB

  ALSO CAPTURE (the independent prize -- the timing intercept):
    T1  the three per-stage prefill times at $RATIO_NEW, against the
        incumbent's 49.2 / 154.8 / 116.4 ms. Two measured cuts pin the
        per-layer slope against the fixed per-stage cost, which converts every
        speedup in DESIGN_704 from an UPPER BOUND into a prediction.

        WHAT THIS PAIR CAN AND CANNOT PIN -- read before planning the run.
        Solved by planner/timing_calibration.py, not by hand:

          rank0: 28 -> 29  (dn=+1)   expected dt ~ 1.76 ms
          rank1: 20 -> 19  (dn=-1)   expected dt ~ 7.74 ms
          rank2: 16 -> 16  UNCHANGED -- NOT CALIBRATED BY THIS PAIR

        rank2 keeps 16 layers in both cuts, so the pair carries no information
        about its intercept. Do NOT report one for it; the solver refuses by
        default rather than emitting the optimistic fixed_ms=0 fallback, which
        would be indistinguishable from a measurement.

        SAMPLE COUNT IS A PRECONDITION, NOT AN AFTERTHOUGHT. The slope is a
        difference of two means over the layer delta, so its standard error is
        sqrt(2)*sigma/|dn|, and dn=1 puts the entire per-stage noise onto the
        slope. That bites unevenly, because rank0's slope is small:

          per-chunk SD | chunks needed for 10% slope precision
                       |   rank0 (1.757)      rank1 (7.740)
                  1 ms |        65                  4
                  2 ms |       260                 14
                  3 ms |       584                 31
                  5 ms |      1620                 84

        So rank1 is cheap to calibrate and rank0 is the binding cost. Convenient
        arithmetic: one max-length prompt is 327680/512 = 640 chunks, which
        clears the 584 needed at SD=3 ms. ONE full-length prefill per arm is
        therefore sufficient for rank0 at 10% -- but MEASURE the per-chunk SD
        and report it, do not assume 3 ms.

        Report per stage: mean, per-chunk SD, N, and the standard error of the
        mean. Without the SE the intercept cannot be gated and must not be
        published.

    T2  the per-rank arming floors at this layout (rev5 consumes them per
        layout; an unbooted cut has no solved floor).

  ROLLBACK: re-run with RATIO_NEW=$RATIO_OLD, or relaunch the incumbent argv
  unmodified. Rank2 is untouched by this arm, so rollback is a plain restart
  with no state migration.

MSG
exit 0
