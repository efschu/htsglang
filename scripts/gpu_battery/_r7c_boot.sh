#!/usr/bin/env bash
# Shared wrapper for the four r7c boot recipes.
#
# The recipes are NOT edited and NOT copied. They are run exactly as they are,
# with the environment they already read (WT, VENV, MODEL_ROOT, REPO_ROOT), and
# their output is collected afterwards. A recipe that has been touched is a
# recipe whose result cannot be compared with the queue's expectations.
#
# What the wrapper adds, and only this:
#   * pins WT to the battery's worktree so the recipe cannot fall back to a
#     different checkout than the one under test,
#   * registers the recipe's server pid for run_step.sh's py-spy-before-kill,
#   * copies the recipe's /tmp output into the run directory, so a result
#     survives the next boot writing to the same /tmp path,
#   * emits the reference column, which is a mechanical join, not a judgement.
#
# NOT executable on its own.

set -uo pipefail

run_r7c_boot() {  # $1 = letter (a|b|c|d), $2 = recipe file name
    local letter="$1" recipe="$2"
    local dir="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR fehlt -- ueber run_step.sh starten}"
    local tmp_out="/tmp/r7c-boot-$letter"
    local tmp_log="/tmp/r7c-boot-$letter.server.log"
    local pidfile="/tmp/r7c-boot-$letter.pid"

    # A stale directory from an earlier attempt would be copied out as if it
    # were this run's result. Move it aside rather than delete it: a previous
    # attempt is evidence too.
    if [ -d "$tmp_out" ]; then
        mv "$tmp_out" "$tmp_out.prev.$(date +%s)" 2>/dev/null
    fi
    rm -f "$tmp_log"

    battery_harvest_pidfile "$pidfile" "$dir/pids"

    echo "== Rezept $recipe (unveraendert) mit WT=$WT =="
    WT="$WT" REPO_ROOT="$REPO_ROOT" VENV="$VENV" MODEL_ROOT="$MODEL_ROOT" \
        bash "$WT/scripts/dual_group/r7c/$recipe"
    local rc=$?
    battery_stop_harvest

    echo "== Ergebnisse einsammeln =="
    if [ -d "$tmp_out" ]; then
        cp -a "$tmp_out/." "$dir/" 2>/dev/null
    else
        echo "WARNUNG: $tmp_out existiert nicht -- das Rezept hat nichts geschrieben"
    fi
    [ -f "$tmp_log" ] && cp "$tmp_log" "$dir/server.log"

    # The card order the recipe resolved at runtime, kept next to the result:
    # every --rank-gpu-id in the log is only readable against it.
    [ -f "$dir/cards.txt" ] && echo "Kartenreihenfolge:" && cat "$dir/cards.txt"

    ls -la "$dir"
    echo "Rezept rc=$rc"
    return "$rc"
}

emit_reference_column() {  # $1 = boot label
    local dir="${BATTERY_STEP_DIR}"
    "$PY" "$BATTERY_DIR/emit_reference_column.py" \
        --accept "$dir/accept.json" \
        --out "$dir/reference_column.json" \
        --boot "$1"
}
