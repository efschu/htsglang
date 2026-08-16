#!/usr/bin/env bash
# #441(c): the #261 gate, SHORT RUN, with NO harness shim. PREP ONLY.
#
# This script is not run from a desk session. It is GPU-gated end to end and
# exists so a window operator can execute one command instead of reassembling
# the sequence under time pressure. It composes `live_handover_gate.sh` rather
# than reimplementing it -- the gate's order of proof is the thing being
# trusted, and a second copy of it would drift.
#
# WHY "NO SHIM" IS THE POINT. The hermetic suites stand a harness in for the
# HiCache host tier, which is exactly the component this run has to prove. A
# short run that still went through the shim would be a proof about the shim.
# So this script REFUSES to start if the shim's env switch is set, rather than
# unsetting it silently: if something in the environment wanted the shim, the
# operator should know before the window is spent, not after.
#
# WHAT "SHORT" MEANS, and what it costs. The full gate runs the A-vs-A floor
# twice and then the whole cross-server claim. The short run keeps EVERY step
# -- dropping one would make the remaining ones unattributable -- and shortens
# only the CONTINUATION LENGTH (`GATE_MAX_TOKENS`, default 64 instead of the
# gate's default). Byte-identity is the assertion; it holds at 64 tokens or it
# does not hold at all, and a shorter continuation makes a failure cheaper to
# repeat, not weaker.
#
# PRECONDITIONS the operator owns (the gate refuses without them):
#   * BOTH servers already up and staying up: source A on $PORT_A, destination
#     B on $PORT_B, same checkpoint and dtype/kv-dtype.
#   * Both booted with at least:
#       --page-size 1 --enable-hierarchical-cache
#       --hicache-storage-backend file --hicache-write-policy write_through
#       --hicache-mem-layout page_first_direct
#     A hybrid-GDN model has no choice: MambaPoolHost accepts
#     page_first_direct only.
#   * SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR set per server to its own store.
#   * A held /spinning/gpu-arb claim. This script checks for one and refuses
#     without it; it never creates or steals a claim.
#
# EXIT CODES: 0 gate passed. 2 precondition missing (nothing was run).
#             Anything else is the gate's own verdict, passed through.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="${GATE:-$HERE/live_handover_gate.sh}"

# --- 1. no shim, and say so if one was asked for -------------------------
: "${SGLANG_HICACHE_HARNESS_SHIM:=}"
if [[ -n "${SGLANG_HICACHE_HARNESS_SHIM}" ]]; then
  echo "REFUSING: SGLANG_HICACHE_HARNESS_SHIM=${SGLANG_HICACHE_HARNESS_SHIM} is set." >&2
  echo "  This run exists to prove the HiCache host tier. With the harness" >&2
  echo "  shim in place it would prove the shim instead. Unset it deliberately" >&2
  echo "  and re-run, so the choice is on the record." >&2
  exit 2
fi

# --- 2. a GPU window must already be held --------------------------------
ARB_DIR="${ARB_DIR:-/spinning/gpu-arb}"
ARB_HOLDER="${ARB_HOLDER:-}"
if [[ -z "${ARB_HOLDER}" ]]; then
  # Any live holder file is enough to prove a window exists; naming which one
  # is the operator's job, not this script's guesswork.
  if ! ls "${ARB_DIR}"/holder-* >/dev/null 2>&1; then
    echo "REFUSING: no holder file under ${ARB_DIR}." >&2
    echo "  This run drives two live servers on the shared cards. Claim a" >&2
    echo "  window first; this script will not create or steal one." >&2
    exit 2
  fi
fi

# --- 3. the gate's own required inputs, checked before anything starts ----
missing=()
for v in PORT_A PORT_B STORE_A STORE_B TOKENIZER TARGET_TP TARGET_RATIOS MODEL_CONFIG; do
  [[ -n "${!v:-}" ]] || missing+=("$v")
done
if (( ${#missing[@]} )); then
  echo "REFUSING: unset required variable(s): ${missing[*]}" >&2
  echo "  Checked here rather than letting the gate fail on the first one," >&2
  echo "  so a window is not spent discovering them one at a time." >&2
  exit 2
fi

if [[ ! -x "${GATE}" && ! -r "${GATE}" ]]; then
  echo "REFUSING: gate script not found at ${GATE}" >&2
  exit 2
fi

# --- 4. short-run knobs --------------------------------------------------
# Only the continuation length moves. Every step of the gate still runs.
export GATE_MAX_TOKENS="${GATE_MAX_TOKENS:-64}"
export STATE="${STATE:-/tmp/handover_shortrun_state.json}"
export MANIFEST="${MANIFEST:-/tmp/handover_shortrun_manifest.json}"

echo "#441(c) #261 gate SHORT RUN -- no shim"
echo "  gate            : ${GATE}"
echo "  A / B           : ${PORT_A} -> ${PORT_B}"
echo "  stores          : ${STORE_A} -> ${STORE_B}"
echo "  target tp/ratios: ${TARGET_TP} / ${TARGET_RATIOS}"
echo "  continuation    : ${GATE_MAX_TOKENS} tokens (short; every step still runs)"
echo "  shim            : none (refused above if requested)"
echo

exec bash "${GATE}" "$@"
