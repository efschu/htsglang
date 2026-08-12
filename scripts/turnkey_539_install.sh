#!/usr/bin/env bash
# Install the #539 turnkey units. Idempotent, dry-run by default.
#
# WHAT THIS DOES NOT DO, and it is the most important line in the file:
# it never ENABLES anything and never starts, stops or restarts serving.
# Installing a unit file and enabling it are separate acts, and only the
# second one changes what the machine does. Two standing reasons:
#
#   1. /spinning/GPU_WINDOWS.md:15-18 and rule 6 at :71 carry a live user
#      order -- "Do not restore production and do not restart the watchdog."
#      Enabling htsglang.target would reverse that order, which is an
#      operator's decision to make, not an installer's.
#   2. The cutover from the existing hand-rolled boot to these units is a
#      scheduled event that wants a human watching it.
#
# So: this script leaves the machine able to run the stack, and unchanged in
# what it is currently running.
#
# THE UNITS ARE RENDERED, NOT COPIED. Until 2026-08-12 this script copied
# them byte for byte, so five units carried a literal /spinning/htsglang-gpu
# for PYTHONPATH and for the interpreter no matter what [stack].repo said. On
# this rig that checkout predated the turnkey merge and every unit died with
# "No module named sglang.srt.turnkey". [stack].repo now decides, which is
# what it always claimed to do.
#
# Usage:
#   scripts/turnkey_539_install.sh                 # dry-run, prints a diff
#   scripts/turnkey_539_install.sh --apply         # write unit files
#   scripts/turnkey_539_install.sh --apply --config-too   # also seed stack.toml
#
set -uo pipefail

CHECKOUT="${CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="${SRC:-$CHECKOUT/deploy/turnkey}"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
CONF_DIR="${CONF_DIR:-/etc/htsglang}"

APPLY=0
CONFIG_TOO=0
for a in "$@"; do
    case "$a" in
        --apply) APPLY=1 ;;
        --config-too) CONFIG_TOO=1 ;;
        --help|-h) sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

UNITS=(
    htsglang.target
    htsglang-preflight.service
    htsglang-planner.service
    "htsglang-serving@.service"
    "htsglang-watchdog@.service"
)

say() { printf '%s\n' "$*"; }
note() { printf '  %s\n' "$*"; }

[ "$APPLY" = 1 ] || say "== DRY RUN == (pass --apply to write; nothing is enabled either way)"

# --- render the units from the config that will run them --------------------
# WHICH CONFIG: the installed one if it exists, because that is what the units
# will read at runtime; otherwise the shipped template, because a first
# install has nothing else. Printed either way -- this file decides every path
# in every unit, and picking it silently is the defect one level up.
dstconf="$CONF_DIR/stack.toml"
seed="$SRC/stack.rig3.toml"
if [ -n "${CONFIG:-}" ]; then
    RENDER_FROM="$CONFIG"
elif [ -f "$dstconf" ]; then
    RENDER_FROM="$dstconf"
else
    RENDER_FROM="$seed"
fi
PY="${PY:-$CHECKOUT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
RENDERED="$(mktemp -d)"
trap 'rm -rf "$RENDERED"' EXIT
say "render from: $RENDER_FROM"
if ! PYTHONPATH="$CHECKOUT/python:${PYTHONPATH:-}" "$PY" \
        "$CHECKOUT/scripts/turnkey_539_render_units.py" \
        --config "$RENDER_FROM" --src "$SRC" --dst "$RENDERED" \
        --config-path "$dstconf"; then
    say "REFUSED: could not render the units from $RENDER_FROM"
    say "  Nothing was written. Fix [stack].repo/.venv/.log_dir and re-run."
    exit 1
fi
# LOG_DIR follows the config too, so the directory this script creates is the
# one the rendered units write into.
LOG_DIR="${LOG_DIR:-$(PYTHONPATH="$CHECKOUT/python:${PYTHONPATH:-}" "$PY" -c '
import sys
from sglang.srt.turnkey import config as C
print(C.load(sys.argv[1]).log_dir)' "$RENDER_FROM")}"
REPO="${REPO:-$(PYTHONPATH="$CHECKOUT/python:${PYTHONPATH:-}" "$PY" -c '
import sys
from sglang.srt.turnkey import config as C
print(C.load(sys.argv[1]).repo)' "$RENDER_FROM")}"

say "source:  $SRC (rendered)"
say "target:  $UNIT_DIR"
say ""

rc=0
for f in "${UNITS[@]}"; do
    src="$RENDERED/$f"
    dst="$UNIT_DIR/$f"
    if [ ! -f "$src" ]; then
        say "MISSING SOURCE $src"; rc=1; continue
    fi
    if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
        say "unchanged  $f"
        continue
    fi
    if [ -f "$dst" ]; then
        say "UPDATE     $f"
        diff -u "$dst" "$src" | sed 's/^/    /' | head -40
    else
        say "NEW        $f"
    fi
    if [ "$APPLY" = 1 ]; then
        install -m 0644 "$src" "$dst" || { say "  install failed"; rc=1; }
    fi
done

say ""
# Directories the units assume. Creating a directory is not a behaviour
# change, so it happens in both modes only under --apply for symmetry.
for d in "$CONF_DIR" "$LOG_DIR"; do
    if [ -d "$d" ]; then
        say "dir exists $d"
    else
        say "MKDIR      $d"
        [ "$APPLY" = 1 ] && mkdir -p "$d"
    fi
done

# The stack config is DATA, not a unit, and it encodes the ship parity. It is
# seeded once and never overwritten: silently replacing an operator's tuned
# config during a routine unit update is exactly the class of surprise this
# feature exists to remove. ($seed and $dstconf are set above, where they also
# decide which config the units are rendered from.)
say ""
if [ -f "$dstconf" ]; then
    say "config exists, NOT overwritten: $dstconf"
    if [ -f "$seed" ] && ! cmp -s "$seed" "$dstconf"; then
        note "differs from the shipped template; review by hand:"
        note "diff -u $dstconf $seed"
    fi
elif [ "$CONFIG_TOO" = 1 ]; then
    say "SEED       $dstconf  (from $(basename "$seed"))"
    [ "$APPLY" = 1 ] && install -m 0644 "$seed" "$dstconf"
else
    say "config absent: $dstconf"
    note "pass --config-too to seed it from $(basename "$seed")"
fi

if [ "$APPLY" = 1 ]; then
    say ""
    if [ "$UNIT_DIR" = "/etc/systemd/system" ]; then
        say "systemctl daemon-reload"
        systemctl daemon-reload || rc=1
    else
        # Reloading systemd after writing units somewhere systemd does not
        # read is a no-op that touches the live manager for nothing. This is
        # also what lets the rendering be exercised end to end in a test.
        say "UNIT_DIR is $UNIT_DIR, not /etc/systemd/system: no daemon-reload"
    fi
fi

# Verify the unit files parse. systemd-analyze verify resolves the units by
# name once installed; before that it can still check the source files.
# It runs on the RENDERED units: a template full of @@REPO@@ would be checked
# for a syntax nobody installs, and would report an absolute-path error for
# every ExecStart.
say ""
say "== systemd-analyze verify =="
for f in "${UNITS[@]}"; do
    out=$(systemd-analyze verify "$RENDERED/$f" 2>&1)
    if [ -n "$out" ]; then
        # Missing-dependency notes are expected for a template's %i instances
        # and for units not yet installed; print them rather than judging.
        printf '  %s:\n' "$f"; printf '%s\n' "$out" | sed 's/^/    /'
    else
        printf '  %s: ok\n' "$f"
    fi
done

say ""
say "== state (nothing was enabled) =="
for f in "${UNITS[@]}"; do
    case "$f" in *@.service) continue ;; esac
    printf '  %-32s %s / %s\n' "$f" \
        "$(systemctl is-enabled "$f" 2>&1 | head -1)" \
        "$(systemctl is-active "$f" 2>&1 | head -1)"
done

say ""
say "Next steps are the OPERATOR's, deliberately not automated:"
say "  1. review $dstconf against the running ship config"
say "  2. $REPO/.venv/bin/python -m sglang.srt.turnkey --config $dstconf preflight"
say "  3. ... boot ship --dry-run     # prove argv/env parity before cutover"
say "  4. systemctl enable --now htsglang.target   # REVERSES the standing"
say "     'do not restore production' order in /spinning/GPU_WINDOWS.md:71"
exit $rc
