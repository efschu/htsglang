#!/usr/bin/env bash
# Apply and verify the NCCL tuning package (#599).
# =====================================================================
#
# WHAT THIS SCRIPT IS FOR
# -----------------------
# deploy/release/nccl-tuning.env is config plus evidence. It is deliberately
# NOT sourced wholesale: most of its content is either already set by the fork
# at runtime ([AUTO]) or conditioned on this rig's weaknesses ([RIG]), and
# both categories do damage when handed to a stranger's machine.
#
# So "applying" the package is mostly a refusal exercise, and that is what this
# script automates:
#
#   * it emits the settings the package is willing to own -- and the honest
#     answer today is ZERO environment variables (see EMIT below);
#   * it emits the one boot-blocking `docker run` flag pair, which is the
#     package's only universal item;
#   * it VERIFIES, by reading back rather than assuming, that the environment
#     a boot actually got is the one the package intends.
#
# THE READ-BACK RULE
# ------------------
# Exporting a variable proves the shell exported it. It does NOT prove NCCL
# honoured it: NCCL ignores unknown names silently, and a value set after
# ncclCommInitRank never takes effect at all. The only authority on what NCCL
# actually consumed is NCCL itself, which prints one line per consumed
# variable under NCCL_DEBUG=INFO / NCCL_DEBUG_SUBSYS=ENV:
#
#     <host>:<pid>:<tid> [0] NCCL INFO NCCL_MAX_CTAS set by environment to 4
#
# `verify-log` parses exactly those lines. A variable we intended to set that
# does not appear in them was NOT applied, whatever the shell says.
#
# EXIT CODES
#   0  every check PASSed (and, under --strict, none were SKIPped)
#   1  at least one check FAILed (or SKIPped under --strict)
#   2  usage error
#
# A check that cannot be shown to fail is unvalidated, so every check here has
# a red case in test/registered/unit/release/test_nccl_tuning_apply_599.py.

set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_FILE="${PACKAGE_DIR}/nccl-tuning.env"

# --- the package's own classification, kept in sync with nccl-tuning.env -----
#
# [AUTO] -- the fork sets these itself at runtime. Setting them by hand either
# loses to the automatic value or fights it, so a hand-set one is an ERROR,
# not a preference. Sources are cited per line in nccl-tuning.env section 3.
NCCL_AUTO_VARS=(
  NCCL_MULTI_RANK_GPU_ENABLE   # engine.py:1624-1625, on duplicate --rank-gpu-id
  NCCL_MAX_CTAS                # engine.py:1638-1640, co-located case [UNMEASURED]
  NCCL_NVLS_ENABLE             # engine.py:1611-1615, co-located case
  NCCL_CUMEM_ENABLE            # engine.py:1358-1359, follows --enable-symm-mem
  NCCL_GRAPH_MIXING_SUPPORT    # engine.py:1368-1372, DEFECT CANDIDATE, see checklist 9
  NCCL_ALGO                    # server_args.py:15992, deterministic inference
)

# [RIG] -- measured on THIS rig and conditioned on its weaknesses (no P2P, no
# NVLink, all pairs PHB, one card on x4). Handing these to a machine that HAS
# working P2P is a performance regression, not a fix, so they require explicit
# consent via --allow-rig-conditioned.
NCCL_RIG_VARS=(
  NCCL_P2P_DISABLE             # nccl-tuning.env section 2 [RIG]
  NCCL_IB_DISABLE              # scripts/pp/pp_crossrig_rank.sh:66-92, foreign RoCE bug
)

# Upstream-inherited IB/RoCE recipes with no fork measurement behind them.
# Re-shipping someone else's claim as ours is the #251 defect by definition.
NCCL_UNBACKED_VARS=(
  NCCL_IB_GID_INDEX NCCL_IB_HCA NCCL_IB_TC NCCL_IB_SL
  NCCL_IB_QPS_PER_CONNECTION NCCL_NET_PLUGIN NCCL_MIN_NCHANNELS
)

# The version floor is a hard requirement, not tuning: 2.28.9 REJECTS a
# co-located communicator, 2.30.7 builds it and serves (engine.py:1604-1610).
NCCL_VERSION_FLOOR_DEFAULT=2.30

# Docker's default /dev/shm is 64 MB and NCCL aborts at ncclGroupEnd() under
# it. This is a property of the container runtime, not of the cards, so it is
# the package's one universal item.
SHM_MIN_BYTES=$((4 * 1024 * 1024 * 1024))

STRICT=0
ALLOW_RIG=0
FAILURES=0
SKIPS=0

log_pass() { printf 'PASS  %-24s %s\n' "$1" "$2"; }
log_fail() { printf 'FAIL  %-24s %s\n' "$1" "$2"; FAILURES=$((FAILURES + 1)); }
log_skip() { printf 'SKIP  %-24s %s\n' "$1" "$2"; SKIPS=$((SKIPS + 1)); }
log_info() { printf 'INFO  %-24s %s\n' "$1" "$2"; }

usage() {
  cat <<'EOF'
usage: apply-nccl-tuning.sh <command> [options]

commands:
  check                 Read-back verification of the current environment.
                        Mutates nothing. This is the safe default to run.
  env                   Print the env assignments the package owns, for
                        `eval "$(apply-nccl-tuning.sh env)"`. Today this is
                        empty by design -- see EMIT in the script header.
  run-flags             Print the docker run flags the package requires.
  verify-log <path>     Parse an NCCL_DEBUG=INFO log and assert that the
                        variables NCCL actually consumed match intent.
                        This is the only true bind proof.
  selftest              Run every check against fabricated fixtures and show
                        each one going red. Proves the checks can fail.

options:
  --strict                  Treat SKIP as failure. Use in the release gate:
                            an unrunnable check must not read as a green one.
  --allow-rig-conditioned   Permit [RIG] variables. Requires that you have
                            confirmed the TARGET machine also reports no peer
                            access (scripts/p2p_readiness/capability_matrix.py).
  --version-floor X.Y       Override the NCCL version floor (default 2.30).
EOF
}

# ---------------------------------------------------------------------------
# check: classify every NCCL_* variable currently set in the environment.
#
# The failure this catches is the realistic one: an operator reads the package,
# copies a line that was labelled [AUTO] or [RIG], and exports it by hand.
# ---------------------------------------------------------------------------
check_env_classification() {
  local name="env-classification"
  local set_vars=() v
  # Only names actually present in the environment, however they got there.
  while IFS='=' read -r v _; do
    case "${v}" in NCCL_*) set_vars+=("${v}") ;; esac
  done < <(env)

  if [ ${#set_vars[@]} -eq 0 ]; then
    log_pass "${name}" "no NCCL_* variables set by hand; the fork owns them all"
    return
  fi

  local bad=0
  for v in "${set_vars[@]}"; do
    if _in_list "${v}" "${NCCL_AUTO_VARS[@]}"; then
      log_fail "${name}" "${v} is [AUTO] -- the fork sets it at runtime; a hand-set value fights the automatic one"
      bad=1
    elif _in_list "${v}" "${NCCL_RIG_VARS[@]}"; then
      if [ "${ALLOW_RIG}" = "1" ]; then
        log_info "${name}" "${v} is [RIG], permitted by --allow-rig-conditioned; confirm the TARGET reports no peer access"
      else
        log_fail "${name}" "${v} is [RIG] -- conditioned on this rig having no P2P/NVLink; pass --allow-rig-conditioned only after probing the target"
        bad=1
      fi
    elif _in_list "${v}" "${NCCL_UNBACKED_VARS[@]}"; then
      log_fail "${name}" "${v} is upstream-inherited with no fork measurement behind it (#251: someone else's claim re-shipped as ours)"
      bad=1
    else
      log_info "${name}" "${v} set, not classified by the package -- you own this one"
    fi
  done
  [ "${bad}" = "0" ] && log_pass "${name}" "${#set_vars[@]} NCCL_* variable(s) set, none of them forbidden"
  return 0
}

_in_list() {
  local needle="$1"; shift
  local item
  for item in "$@"; do [ "${item}" = "${needle}" ] && return 0; done
  return 1
}

# ---------------------------------------------------------------------------
# check: /dev/shm large enough. Read back the actual mount, not the run flags
# we hoped were passed -- the flag pair is what SETS this, the size is what
# PROVES it.
# ---------------------------------------------------------------------------
check_shm() {
  local name="shm-size"
  local shm_path="${HTSGLANG_SHM_PATH:-/dev/shm}"
  if [ ! -d "${shm_path}" ]; then
    log_fail "${name}" "${shm_path} does not exist"
    return
  fi
  local bytes
  bytes="$(df -B1 --output=size "${shm_path}" 2>/dev/null | tail -1 | tr -dc '0-9')" || bytes=""
  if [ -z "${bytes}" ]; then
    log_skip "${name}" "could not read the size of ${shm_path}"
    return
  fi
  if [ "${bytes}" -lt "${SHM_MIN_BYTES}" ]; then
    log_fail "${name}" "${shm_path} is $((bytes / 1024 / 1024)) MB, below the 4096 MB floor; NCCL aborts at ncclGroupEnd() -- add --ipc=host --shm-size=4g"
  else
    log_pass "${name}" "${shm_path} is $((bytes / 1024 / 1024)) MB (>= 4096 MB floor)"
  fi
}

# ---------------------------------------------------------------------------
# check: NCCL version floor.
#
# Read back from the library actually present. The Dockerfile's build-time
# assert greps `strings` for the version, which also matches unrelated text;
# here we prefer the versioned filename or soname, and fall back to the
# substring only with the weakness stated out loud.
# ---------------------------------------------------------------------------
check_nccl_version() {
  local name="nccl-version-floor"
  local floor="${VERSION_FLOOR:-${NCCL_VERSION_FLOOR_DEFAULT}}"
  local lib="${HTSGLANG_NCCL_LIB:-}"

  if [ -z "${lib}" ]; then
    # Bounded search only. An unqualified `find /` walks the whole volume --
    # on the build box that is a multi-terabyte pool and the check appears to
    # hang. Look where the bundled wheel actually puts it: the interpreter's
    # own site-packages first, then the usual system paths.
    local roots=() r site
    site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    [ -n "${site}" ] && roots+=("${site}")
    roots+=(/usr/local/lib/python3*/dist-packages /usr/lib/python3*/dist-packages
            /usr/local/lib /usr/lib/x86_64-linux-gnu)
    for r in "${roots[@]}"; do
      [ -d "${r}" ] || continue
      lib="$(find "${r}" -maxdepth 6 -path '*nccl/lib/libnccl.so.2*' 2>/dev/null | head -1)"
      [ -n "${lib}" ] && break
    done
  fi
  if [ -z "${lib}" ] || [ ! -e "${lib}" ]; then
    log_skip "${name}" "no bundled libnccl found; run this inside the image, or set HTSGLANG_NCCL_LIB"
    return
  fi

  # Preferred: a fully versioned file (libnccl.so.2.30.7) reached directly or
  # through the .so.2 symlink. This is NCCL's own naming, not a text match.
  local target ver=""
  target="$(readlink -f "${lib}" 2>/dev/null || echo "${lib}")"
  ver="$(printf '%s' "${target}" | sed -n 's/.*libnccl\.so\.\([0-9][0-9.]*\)$/\1/p')"

  if [ -z "${ver}" ] || [ "${ver}" = "2" ]; then
    # The wheel ships libnccl.so.2 with no versioned filename, so the naming
    # route is unavailable. Fall back to NCCL's own embedded banner, which has
    # the distinctive form "2.30.7+cuda13.0". This is a strings(1) match and
    # therefore weaker than a symbol read -- but anchoring on the "+cuda"
    # suffix makes it far more specific than the Dockerfile's bare version
    # grep, which also matches paths and log text anywhere in the binary.
    ver="$(strings "${target}" 2>/dev/null \
           | sed -n 's/^\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)+cuda[0-9.]*$/\1/p' \
           | sort -V | tail -1)"
    if [ -n "${ver}" ]; then
      log_info "${name}" "no versioned soname; using NCCL's embedded '<ver>+cuda' banner instead"
    else
      log_skip "${name}" "libnccl at ${target} carries neither a versioned soname nor a readable '+cuda' banner; use verify-log against a real boot, where NCCL reports its own version"
      return
    fi
  fi

  if _version_ge "${ver}" "${floor}"; then
    log_pass "${name}" "libnccl ${ver} >= ${floor} (from ${target})"
  else
    log_fail "${name}" "libnccl ${ver} is below the ${floor} floor; a co-located communicator (--rank-gpu-id with duplicates) is REJECTED by NCCL below 2.30"
  fi
}

_version_ge() { # _version_ge HAVE WANT
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]
}

# ---------------------------------------------------------------------------
# verify-log: the only true bind proof.
#
# NCCL prints one line per variable it actually consumed. Anything we intended
# to set that is absent from those lines was NOT applied.
# ---------------------------------------------------------------------------
verify_log() {
  local logfile="$1"; shift
  local name="env-bind"

  if [ ! -f "${logfile}" ]; then
    log_fail "${name}" "log ${logfile} does not exist"
    return
  fi

  # NCCL's own report of what it consumed.
  local consumed
  consumed="$(sed -n 's/.*NCCL INFO \(NCCL_[A-Z0-9_]*\) set by environment to \(.*\)$/\1=\2/p' "${logfile}" | tr -d '\r')"

  if [ -z "${consumed}" ]; then
    if grep -q 'NCCL INFO' "${logfile}"; then
      log_pass "${name}" "NCCL logged INFO but consumed no environment variables -- consistent with the package owning none"
    else
      log_skip "${name}" "no 'NCCL INFO' lines in ${logfile}; the boot did not run with NCCL_DEBUG=INFO, so nothing can be read back"
    fi
  else
    local line var val
    while IFS= read -r line; do
      [ -z "${line}" ] && continue
      var="${line%%=*}"; val="${line#*=}"
      if _in_list "${var}" "${NCCL_RIG_VARS[@]}" && [ "${ALLOW_RIG}" != "1" ]; then
        log_fail "${name}" "NCCL consumed ${var}=${val}, a [RIG] setting, without --allow-rig-conditioned"
      elif _in_list "${var}" "${NCCL_UNBACKED_VARS[@]}"; then
        log_fail "${name}" "NCCL consumed ${var}=${val}, which has no fork measurement behind it (#251)"
      else
        log_pass "${name}" "NCCL consumed ${var}=${val}"
      fi
    done <<< "${consumed}"
  fi

  # The version floor, as NCCL reports it about itself. Strongest available
  # form of the check: not the filename, not a strings match, but NCCL's own
  # banner from the process that actually initialised.
  local banner
  banner="$(sed -n 's/.*NCCL version \([0-9][0-9.]*\).*/\1/p' "${logfile}" | head -1)"
  if [ -z "${banner}" ]; then
    log_skip "nccl-version-runtime" "no 'NCCL version' banner in ${logfile}"
  elif _version_ge "${banner}" "${VERSION_FLOOR:-${NCCL_VERSION_FLOOR_DEFAULT}}"; then
    log_pass "nccl-version-runtime" "NCCL reports ${banner} >= ${VERSION_FLOOR:-${NCCL_VERSION_FLOOR_DEFAULT}}"
  else
    log_fail "nccl-version-runtime" "NCCL reports ${banner}, below the floor; co-location will be rejected"
  fi
}

emit_env() {
  cat <<'EOF'
# NCCL tuning package (#599) -- environment assignments.
#
# INTENTIONALLY EMPTY. This is a result, not an omission.
#
# Every performance setting the package examined fell into one of three
# buckets, and none of them belongs in a shipped export:
#
#   [AUTO]  the fork already sets it at runtime, from the actual rank layout.
#           A hand-set value here loses to it or fights it.
#   [RIG]   measured on a machine with no P2P and no NVLink. Exporting it for
#           a user who HAS NVLink is a performance regression.
#   [none]  the remaining candidates had no measurement behind them at all
#           (see nccl-tuning.env section 4 for each one and why it was cut).
#
# What the package DOES require is not an env var: it is the container run
# flag pair below, plus an NCCL >= 2.30 floor. Get those from `run-flags`.
EOF
}

emit_run_flags() {
  # The one boot-blocking, universal item in the whole package.
  echo "--ipc=host --shm-size=4g"
}

# ---------------------------------------------------------------------------
# selftest: drive every check into its red state with fabricated fixtures.
# A check that has never been observed failing is not a check.
# ---------------------------------------------------------------------------
selftest() {
  local rc red=0 total=0
  SELFTEST_TMP="$(mktemp -d)"
  local tmp="${SELFTEST_TMP}"
  trap 'rm -rf "${SELFTEST_TMP:-}"' RETURN

  echo "== selftest: each case below MUST go red =="

  total=$((total + 1))
  echo "-- case 1: an [AUTO] variable set by hand"
  rc=0
  NCCL_MAX_CTAS=4 "${BASH_SOURCE[0]}" check >/dev/null 2>&1 || rc=$?
  if [ "${rc}" != "0" ]; then echo "   red as required (exit ${rc})"; red=$((red + 1)); else echo "   GREEN -- check is broken"; fi

  total=$((total + 1))
  echo "-- case 2: a [RIG] variable without consent"
  rc=0
  NCCL_P2P_DISABLE=1 "${BASH_SOURCE[0]}" check >/dev/null 2>&1 || rc=$?
  if [ "${rc}" != "0" ]; then echo "   red as required (exit ${rc})"; red=$((red + 1)); else echo "   GREEN -- check is broken"; fi

  total=$((total + 1))
  echo "-- case 2b: the same [RIG] variable WITH consent must go green"
  rc=0
  NCCL_P2P_DISABLE=1 "${BASH_SOURCE[0]}" check --allow-rig-conditioned >/dev/null 2>&1 || rc=$?
  if [ "${rc}" = "0" ]; then echo "   green as required"; red=$((red + 1)); else echo "   RED -- consent flag does not work (exit ${rc})"; fi

  total=$((total + 1))
  echo "-- case 3: an undersized /dev/shm"
  mkdir -p "${tmp}/shm"
  # A plain directory inherits the host filesystem's size and would NOT
  # exercise the floor, so mount a real 64 MB tmpfs -- Docker's actual default
  # -- inside a PRIVATE mount namespace. The namespace dies with the command,
  # so nothing is left mounted on the host and no cleanup can be forgotten.
  rc=0
  if unshare -m --map-root-user true 2>/dev/null || unshare -m true 2>/dev/null; then
    unshare -m bash -c "
      mount -t tmpfs -o size=64m tmpfs '${tmp}/shm' || exit 97
      HTSGLANG_SHM_PATH='${tmp}/shm' '${BASH_SOURCE[0]}' check >/dev/null 2>&1
    " || rc=$?
    if [ "${rc}" = "97" ]; then
      echo "   NOT EXERCISED: could not mount tmpfs in the namespace -- case counted as FAILED"
    elif [ "${rc}" != "0" ]; then
      echo "   red as required (exit ${rc})"; red=$((red + 1))
    else
      echo "   GREEN -- check is broken"
    fi
  else
    echo "   NOT EXERCISED: no usable mount namespace here -- case counted as FAILED, not waved through"
  fi

  total=$((total + 1))
  echo "-- case 4: a log where NCCL consumed a [RIG] variable"
  printf 'host:1:1 [0] NCCL INFO NCCL_P2P_DISABLE set by environment to 1\n' > "${tmp}/rig.log"
  rc=0
  "${BASH_SOURCE[0]}" verify-log "${tmp}/rig.log" >/dev/null 2>&1 || rc=$?
  if [ "${rc}" != "0" ]; then echo "   red as required (exit ${rc})"; red=$((red + 1)); else echo "   GREEN -- check is broken"; fi

  total=$((total + 1))
  echo "-- case 5: a log whose NCCL banner is below the floor"
  printf 'host:1:1 [0] NCCL INFO NCCL version 2.28.9+cuda13.0\n' > "${tmp}/old.log"
  rc=0
  "${BASH_SOURCE[0]}" verify-log "${tmp}/old.log" >/dev/null 2>&1 || rc=$?
  if [ "${rc}" != "0" ]; then echo "   red as required (exit ${rc})"; red=$((red + 1)); else echo "   GREEN -- check is broken"; fi

  total=$((total + 1))
  echo "-- case 6: a log with no NCCL_DEBUG output at all is a SKIP, and --strict must turn it red"
  printf 'nothing to see here\n' > "${tmp}/quiet.log"
  rc=0
  "${BASH_SOURCE[0]}" verify-log "${tmp}/quiet.log" --strict >/dev/null 2>&1 || rc=$?
  if [ "${rc}" != "0" ]; then echo "   red as required (exit ${rc})"; red=$((red + 1)); else echo "   GREEN -- --strict does not escalate SKIP"; fi

  echo "== selftest: ${red}/${total} cases behaved as required =="
  [ "${red}" = "${total}" ]
}

main() {
  local cmd="${1:-}"; shift || true
  local logfile=""

  # Collect options; verify-log takes a positional path.
  local args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --strict) STRICT=1 ;;
      --allow-rig-conditioned) ALLOW_RIG=1 ;;
      --version-floor) VERSION_FLOOR="${2:-}"; shift ;;
      -h|--help) usage; return 0 ;;
      *) args+=("$1") ;;
    esac
    shift
  done

  case "${cmd}" in
    check)
      log_info "package" "${PACKAGE_FILE}"
      check_env_classification
      check_shm
      check_nccl_version
      ;;
    verify-log)
      logfile="${args[0]:-}"
      [ -n "${logfile}" ] || { echo "verify-log needs a log path" >&2; usage >&2; return 2; }
      verify_log "${logfile}"
      ;;
    env) emit_env; return 0 ;;
    run-flags) emit_run_flags; return 0 ;;
    selftest) selftest; return $? ;;
    ""|-h|--help) usage; return 0 ;;
    *) echo "unknown command: ${cmd}" >&2; usage >&2; return 2 ;;
  esac

  if [ "${FAILURES}" -gt 0 ]; then
    echo "-- ${FAILURES} check(s) FAILED"
    return 1
  fi
  if [ "${STRICT}" = "1" ] && [ "${SKIPS}" -gt 0 ]; then
    echo "-- ${SKIPS} check(s) SKIPPED and --strict is set; a check that could not run is not a passing check"
    return 1
  fi
  echo "-- all checks passed${SKIPS:+ (${SKIPS} skipped)}"
  return 0
}

main "$@"
