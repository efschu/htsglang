#!/bin/bash
# Continuous page-cache trim during a DeepSeek V4 Flash GGUF load (#391).
#
# WHY THIS EXISTS -- THE DROPPER ALONE IS NOT ENOUGH
# ---------------------------------------------------------------------------
# The loader's ConsumedPageDropper (SGLANG_GGUF_STREAM_DROP_CACHE, runbook
# 4.5.5) releases the checkpoint's page cache BEHIND the weight stream. Boot 10
# attempt B (2026-08-01, .../2026-08-01_391_dsv4flash10/ram10b.log) showed what
# that leaves on the table: the dropper was working -- 40.68 GiB released in 76
# advice calls by tensor 565 of 1328 -- and rammon's 92.6 GiB guard still fired
# at memory.current 95.8 GiB, with anon at only 12.7 and `file` at 79.3. The
# dropper can only drop behind the consumer; the kernel's readahead pulls pages
# in AHEAD of it faster than the consumer retires them, and on a swapless box
# that difference is the whole budget.
#
# Run alongside rammon.sh, this converted a guard trip into a completed load:
# attempts C, D and E all held at 83-85 GiB against the same 92.6 GiB guard and
# streamed all 1328 tensors, all 43 layers, on all 3 ranks.
#
# WHY memory.reclaim IS SAFE HERE, AND WHY THE SWAP CHECK IS NOT DECORATION
# ---------------------------------------------------------------------------
# With no swap, cgroup reclaim has nowhere to put anonymous pages, so it CANNOT
# evict them: the pinned host expert pool, the CUDA host allocations and the
# Python heap are all structurally out of reach. The only thing it can take is
# page cache and reclaimable slab -- which is exactly the term that over-runs
# the guard, and which the loader re-reads from disk if it ever needs it again.
# Worst case is a slower load, not a wrong one. It is the same instrument
# preboot_cache_reset.sh already uses, applied continuously instead of once.
#
# That argument is a property of the BOX, not of the script, so the script
# checks it instead of assuming it: with swap configured, reclaim could page
# out the loader's own anon and turn a fast load into a thrashing one. Then
# this refuses to run unless --allow-swap says the operator owns that choice.
#
# Harness script: it patches no fork code and changes no load path.
#
# Measured, boot 10 attempt C: one pass took memory.current 51.91 -> 39.91 GiB
# with `file` giving up the full 12 GiB and anon untouched. `--self-test`
# re-checks the mechanism (acts when over, quiet when under, exits when the
# server is gone, refuses with swap) without needing a load.
set -u

PIDFILE=""
OUT=""
RUN_DIR=""
SOFT_GIB=66
TARGET_GIB=58
MAX_ASK_GIB=12
INTERVAL=5
ONCE=0
ALLOW_SWAP=0
SELF_TEST=0
# Stop by itself once the server is serving. The trim exists to manage the
# LOAD-time page-cache race; that race ends at ready. Window 4 vs 5 of #391
# measured what leaving it running costs: the A-vs-A floor was 39.91% with it
# alive during serving and 2.55% with it stopped, and both stopped-runs were
# FASTER than either running-run -- reclaiming page cache out from under the
# host-pinned expert pool is not free. Leaving that to the operator is a
# 37-point throughput trap with no upside.
READY_URL=""
READY_MARKER=""

usage() {
  cat >&2 <<'EOF'
usage: cachetrim.sh [--pidfile FILE] [--run-dir DIR] [--out FILE]
                    [--soft-gib N] [--target-gib N] [--max-ask-gib N]
                    [--interval SECONDS] [--once] [--allow-swap]
                    [--ready-url URL] [--ready-marker FILE]
       cachetrim.sh --self-test

  --pidfile      PID of the launched server; the trim ends when it is gone
  --run-dir      directory for the log (default: the pidfile's directory, or .)
  --out          log path (default: <run-dir>/cachetrim.log)
  --soft-gib     start trimming above this memory.current (default 66)
  --target-gib   trim back down to about here (default 58)
  --max-ask-gib  largest single reclaim ask (default 12)
  --interval     seconds between samples (default 5)
  --once         one pass, then exit (the smoke; also useful by hand)
  --allow-swap   run even though this box has swap -- see the header
  --ready-url    stop once this URL answers 200 (the server is serving and the
                 load-time race is over; see the note at READY_URL above)
  --ready-marker stop once this file exists (alternative to --ready-url)
  --self-test    exercise both branches against a fake cgroup and exit
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --pidfile) PIDFILE="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --soft-gib) SOFT_GIB="$2"; shift 2 ;;
    --target-gib) TARGET_GIB="$2"; shift 2 ;;
    --max-ask-gib) MAX_ASK_GIB="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --allow-swap) ALLOW_SWAP=1; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    --ready-url) READY_URL="$2"; shift 2 ;;
    --ready-marker) READY_MARKER="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "cachetrim.sh: unknown argument '$1'" >&2; usage ;;
  esac
done

# Overridable so the self-test can drive both branches without a load, in the
# same shape preboot_cache_reset.sh already uses.
CG=${DSV4_CGROUP_ROOT:-/sys/fs/cgroup}
GIB=$((1024 * 1024 * 1024))

swap_total_kb() {
  if [ -n "${DSV4_SWAP_TOTAL_KB:-}" ]; then
    echo "$DSV4_SWAP_TOTAL_KB"
    return
  fi
  awk '/^SwapTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0
}

stat_gib() {  # $1 = memory.stat key
  awk -v k="$1" '$1==k{printf "%.2f", $2/1073741824}' "$CG/memory.stat" \
    2>/dev/null || echo "?"
}

# Is the server past load and serving? Checked before every sample, so the
# trim retires on its own the moment its job is done.
server_ready() {
  if [ -n "$READY_MARKER" ] && [ -e "$READY_MARKER" ]; then
    return 0
  fi
  if [ -n "$READY_URL" ]; then
    local code
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$READY_URL" 2>/dev/null)
    [ "$code" = "200" ] && return 0
  fi
  return 1
}

# One sample. Returns 0 normally, 10 when the watched server is gone,
# 11 once the server is ready and this script has no further job.
trim_pass() {
  if server_ready; then
    echo "$(date -u +%H:%M:%S) server ready -- load-time race is over, exiting" \
      >> "$OUT"
    return 11
  fi
  if [ -n "$PIDFILE" ] && [ -f "$PIDFILE" ]; then
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || {
      echo "$(date -u +%H:%M:%S) server gone, exiting" >> "$OUT"
      return 10
    }
  fi
  local cur cur_gib ask file_before file_after anon_before anon_after after rc
  cur=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
  cur_gib=$((cur / GIB))
  [ "$cur_gib" -gt "$SOFT_GIB" ] || return 0

  ask=$((cur_gib - TARGET_GIB))
  [ "$ask" -gt "$MAX_ASK_GIB" ] && ask=$MAX_ASK_GIB
  [ "$ask" -ge 1 ] || return 0
  file_before=$(stat_gib file)
  anon_before=$(stat_gib anon)
  if printf '%sG' "$ask" > "$CG/memory.reclaim" 2>/dev/null; then rc=ok; else rc=partial; fi
  after=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
  file_after=$(stat_gib file)
  anon_after=$(stat_gib anon)
  # anon is printed on purpose: it is the term that must NOT move. If it ever
  # does, the swapless assumption above has stopped holding on this box.
  printf '%s trim ask=%sG (%s) current %.2f -> %.2f GiB, file %s -> %s GiB, anon %s -> %s GiB\n' \
    "$(date -u +%H:%M:%S)" "$ask" "$rc" \
    "$(awk -v a="$cur" 'BEGIN{print a/1073741824}')" \
    "$(awk -v a="$after" 'BEGIN{print a/1073741824}')" \
    "$file_before" "$file_after" "$anon_before" "$anon_after" >> "$OUT"
  return 0
}

# --- self-test -------------------------------------------------------------
# Both branches, plus the two refusals. Fails loudly if the trim ever fires
# under the ceiling (which would make it a blind reclaim loop) or never fires
# above it (which would make it a no-op the boot recipe cannot rely on).
self_test() {
  local root pass fail me rc asked askok quiet gone
  me=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  pass=0
  fail=0
  root=$(mktemp -d)
  _SELFTEST_ROOT="$root"
  trap 'rm -rf "${_SELFTEST_ROOT:-}"' EXIT
  mkdir -p "$root/cg"
  printf '0\n' > "$root/cg/memory.reclaim"
  printf 'anon 1073741824\nfile 75161927680\n' > "$root/cg/memory.stat"

  check() {  # name, expected-rc, actual-rc, condition-description, condition-rc
    if [ "$2" = "$3" ] && [ "${5:-0}" = "0" ]; then
      echo "  ok   $1"
      pass=$((pass + 1))
    else
      echo "  FAIL $1 (rc=$3, expected $2; $4)"
      fail=$((fail + 1))
    fi
  }

  # 1) over the ceiling: it asks, and it asks for the difference (capped).
  awk 'BEGIN{printf "%d\n", 80*1073741824}' > "$root/cg/memory.current"
  : > "$root/cg/memory.reclaim"
  DSV4_CGROUP_ROOT="$root/cg" DSV4_SWAP_TOTAL_KB=0 bash "$me" \
    --once --soft-gib 66 --target-gib 58 --out "$root/over.log" >/dev/null 2>&1
  rc=$?
  asked=$(cat "$root/cg/memory.reclaim")
  askok=1
  [ "$asked" = "12G" ] && grep -q "trim ask=12G" "$root/over.log" && askok=0
  check "acts when over the ceiling (asked '$asked', want 12G)" 0 "$rc" \
    "log: $(tr -d '\n' < "$root/over.log")" "$askok"

  # 2) under the ceiling: it must be silent, not merely small.
  awk 'BEGIN{printf "%d\n", 40*1073741824}' > "$root/cg/memory.current"
  : > "$root/cg/memory.reclaim"
  DSV4_CGROUP_ROOT="$root/cg" DSV4_SWAP_TOTAL_KB=0 bash "$me" \
    --once --soft-gib 66 --target-gib 58 --out "$root/under.log" >/dev/null 2>&1
  rc=$?
  quiet=0
  [ -s "$root/cg/memory.reclaim" ] && quiet=1
  grep -q "trim ask=" "$root/under.log" && quiet=1
  check "quiet when under the ceiling" 0 "$rc" "it reclaimed anyway" "$quiet"

  # 3) the watched server is gone -> clean exit, no reclaim.
  awk 'BEGIN{printf "%d\n", 80*1073741824}' > "$root/cg/memory.current"
  : > "$root/cg/memory.reclaim"
  echo 2147483646 > "$root/dead.pid"   # a pid that cannot exist
  DSV4_CGROUP_ROOT="$root/cg" DSV4_SWAP_TOTAL_KB=0 bash "$me" \
    --pidfile "$root/dead.pid" --soft-gib 66 --target-gib 58 \
    --out "$root/gone.log" >/dev/null 2>&1
  rc=$?
  gone=0
  grep -q "server gone" "$root/gone.log" || gone=1
  [ -s "$root/cg/memory.reclaim" ] && gone=1
  check "exits when the server is gone" 0 "$rc" "no 'server gone' line" "$gone"

  # 4) swap present -> refuse by name rather than thrash the loader's anon.
  awk 'BEGIN{printf "%d\n", 80*1073741824}' > "$root/cg/memory.current"
  DSV4_CGROUP_ROOT="$root/cg" DSV4_SWAP_TOTAL_KB=4194304 bash "$me" \
    --once --out "$root/swap.log" >/dev/null 2>&1
  check "refuses on a box with swap" 3 "$?" "it ran anyway"

  # 5) the real instrument, when this container can use it at all. Not a
  #    branch of the script -- proof that memory.reclaim does something here.
  if [ -w /sys/fs/cgroup/memory.reclaim ]; then
    local rf_before rf_after
    rf_before=$(awk '/^file /{print $2}' /sys/fs/cgroup/memory.stat)
    printf '256M' > /sys/fs/cgroup/memory.reclaim 2>/dev/null || true
    rf_after=$(awk '/^file /{print $2}' /sys/fs/cgroup/memory.stat)
    awk -v b="$rf_before" -v a="$rf_after" \
      'BEGIN{printf "  info live memory.reclaim 256M: file %.2f -> %.2f GiB\n", b/1073741824, a/1073741824}'
  else
    echo "  info live memory.reclaim not writable here; mechanism untested"
  fi

  echo "cachetrim self-test: $pass passed, $fail failed"
  [ "$fail" -eq 0 ]
}

if [ "$SELF_TEST" = "1" ]; then
  self_test
  exit $?
fi

swap_kb=$(swap_total_kb)
if [ "${swap_kb:-0}" -gt 0 ] && [ "$ALLOW_SWAP" = "0" ]; then
  cat >&2 <<EOF
cachetrim.sh: this box has ${swap_kb} kB of swap.
  The safety argument for a continuous memory.reclaim is that a swapless
  cgroup CANNOT evict anon, so the pinned expert pool, the CUDA host
  allocations and the Python heap are out of reach and only page cache can be
  taken. With swap that stops being true and the trim can page out the load
  itself. Pass --allow-swap to run anyway.
EOF
  exit 3
fi

[ -n "$RUN_DIR" ] || { [ -n "$PIDFILE" ] && RUN_DIR=$(dirname "$PIDFILE"); }
[ -n "$RUN_DIR" ] || RUN_DIR=.
[ -n "$OUT" ] || OUT="$RUN_DIR/cachetrim.log"

if [ ! -r "$CG/memory.current" ]; then
  echo "cachetrim.sh: $CG/memory.current is not readable; refusing to run blind" >&2
  exit 4
fi

printf 'cachetrim: soft=%s GiB target=%s GiB max-ask=%s GiB interval=%ss cgroup=%s\n' \
  "$SOFT_GIB" "$TARGET_GIB" "$MAX_ASK_GIB" "$INTERVAL" "$CG" > "$OUT"
if [ -n "$READY_URL" ] || [ -n "$READY_MARKER" ]; then
  printf 'cachetrim: will stop itself at ready (url=%s marker=%s)\n' \
    "${READY_URL:-none}" "${READY_MARKER:-none}" >> "$OUT"
else
  printf 'cachetrim: NO ready signal given -- it will run until the server exits, which costs throughput during serving (#391 w4 vs w5: floor 39.91%% vs 2.55%%). Pass --ready-url or --ready-marker.\n' >> "$OUT"
fi

while :; do
  trim_pass
  rc=$?
  # 10 = the watched server is gone; 11 = it is ready and this script is done.
  [ "$rc" = "10" ] && exit 0
  [ "$rc" = "11" ] && exit 0
  [ "$ONCE" = "1" ] && exit 0
  sleep "$INTERVAL"
done
