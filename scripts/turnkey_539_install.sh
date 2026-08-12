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
# Usage:
#   scripts/turnkey_539_install.sh                 # dry-run, prints a diff
#   scripts/turnkey_539_install.sh --apply         # write unit files
#   scripts/turnkey_539_install.sh --apply --config-too   # also seed stack.toml
#
set -uo pipefail

REPO="${REPO:-/spinning/htsglang-gpu}"
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deploy/turnkey}"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
CONF_DIR="${CONF_DIR:-/etc/htsglang}"
LOG_DIR="${LOG_DIR:-/var/log/htsglang}"

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
say "source:  $SRC"
say "target:  $UNIT_DIR"
say ""

rc=0
for f in "${UNITS[@]}"; do
    src="$SRC/$f"
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
# feature exists to remove.
seed="$SRC/stack.rig3.toml"
dstconf="$CONF_DIR/stack.toml"
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
    say "systemctl daemon-reload"
    systemctl daemon-reload || rc=1
fi

# Verify the unit files parse. systemd-analyze verify resolves the units by
# name once installed; before that it can still check the source files.
say ""
say "== systemd-analyze verify =="
for f in "${UNITS[@]}"; do
    out=$(systemd-analyze verify "$SRC/$f" 2>&1)
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
