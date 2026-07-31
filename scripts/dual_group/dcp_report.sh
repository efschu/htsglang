#!/usr/bin/env bash
# Effective DCP geometry of a booted arm -- print it, per arm, always.
#
# WHY THIS EXISTS (#340 -> #343 -> #345). SGLANG_UNEVEN_DCP /
# SGLANG_UNEVEN_DCP_WEIGHTED live in the shared harness environment, but they
# only BITE once a --rank-tp-ratio plan is installed (uneven_dcp_kv_replicated
# keys on the plan). So an arm matrix built on that environment silently ran
# every ratio arm at dcp_size=2 and every control at dcp_size=1: the ratio flag
# was carrying a second, undeclared change, and #340 published "the deviation
# is uneven-TP-specific" from it. #343 overturned that verdict with one extra
# arm. The cost of never repeating it is these four lines per arm.
#
# Usage (after the server is up, before teardown):
#   source "$(dirname "$0")/../dcp_report.sh"
#   report_dcp <label> <server-log-path> [<note-fn>]
#
# `note_fn` defaults to `note` when the caller defines one, else `echo`.
# Reads the server log only -- no endpoint, no extra request, works even when
# the arm's probe failed.

report_dcp() {
  local label="${1:?report_dcp needs a label}"
  local log="${2:?report_dcp needs the server log path}"
  local emit="${3:-}"
  if [ -z "$emit" ]; then
    if declare -F note >/dev/null 2>&1; then emit=note; else emit=echo; fi
  fi

  local dcp tp ratio vector weighted env_dcp env_weighted
  if [ ! -f "$log" ]; then
    "$emit" "$label: dcp geometry UNKNOWN (no server log at $log)"
    return 0
  fi

  # The scheduler prints the resolved geometry itself; take it from there
  # rather than re-deriving it from the launch flags, which is exactly the
  # inference that was wrong.
  dcp="$(grep -oE 'dcp_size=[0-9]+' "$log" | head -1 | cut -d= -f2)"
  tp="$(grep -oE 'tp_size=[0-9]+' "$log" | head -1 | cut -d= -f2)"
  ratio="$(grep -oE 'rank_tp_ratio=(None|\[[0-9, ]*\])' "$log" | head -1 | cut -d= -f2-)"
  vector="$(grep -oE 'active vector \[[0-9, ]*\]|vector \[[0-9, ]*\]' "$log" | head -1)"
  env_dcp="${SGLANG_UNEVEN_DCP:-unset}"
  env_weighted="${SGLANG_UNEVEN_DCP_WEIGHTED:-unset}"
  if grep -q 'Uneven DCP: auto-set dcp_size' "$log"; then
    weighted="token-sharded"
  else
    weighted="not engaged"
  fi

  "$emit" "$label: effective dcp_size=${dcp:-?} tp_size=${tp:-?} rank_tp_ratio=${ratio:-?} ${vector:-no-token-vector} [$weighted] (env SGLANG_UNEVEN_DCP=$env_dcp WEIGHTED=$env_weighted)"
}
