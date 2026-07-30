#!/usr/bin/env bash
# S13 -- #293 step 2: what each lever against the multi-session prefill ceiling
# is actually worth, one number per lever.
#
# Step 1 established the ceiling and ruled out the obvious causes: collective
# size constant at 20 MiB, round count constant at 1, both transports ending on
# the same absolute level from four sessions on, over 96 % of the growth in
# `wait` rather than `compute`. What it could NOT say is what the ceiling IS.
# The s12 run that produced those numbers had the pipe off, the direct mode off
# and the prefill graph off -- three levers that were never once measured.
#
# This step measures them. One boot per arm, the arms differing in exactly the
# environment variables and server arguments that name the lever, everything
# else byte-identical through the shared boot template.
#
# ROUNDS, NOT BLOCKS. Every round walks the whole arm list in the same order,
# so a drifting clock or a warming card hits all arms alike, and the repeated
# rounds are the A-vs-A spread that decides which differences may be reported
# at all. Two rounds of the same arm are the noise floor; anything inside it is
# not a finding.
#
# TWO POINTS PER BOOT, sessions=1 first and sessions=8 last. The order is not
# cosmetic: s12_log_analyse takes the last N large batches of a log as the
# measured window, so the point that gets the compute/wait split is whichever
# ran last. That is the primary point, and it runs last on purpose.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./battery_host.sh
source ./_bar1_host_boot.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR fehlt}"
PORT="${BAR1_PORT}"
POINT_S="${S13_POINT_SECONDS:-15}"
WARMUP_S="${S13_WARMUP_SECONDS:-8}"
PROMPT_TOKENS="${S13_PROMPT_TOKENS:-2048}"
RUNDEN="${S13_RUNDEN:-2}"
# The round number goes into the arm label, so a run that continues an earlier
# one must not restart the count and overwrite its points.
RUNDE_START="${S13_RUNDE_START:-1}"
# sessions:with_decode, in the order they run inside one boot.
PUNKTE="${S13_PUNKTE:-1:0 8:1}"

# name | transport arm | extra env | extra server args
#
# The transport arm is the ONLY thing that decides bar1 vs. host path; every
# lever on top of it is an environment variable or a server argument, so that
# an arm can never accidentally change the transport as a side effect.
#
# `bar1pipe` carries SGLANG_HTCCL_BAR1_PIPE_DIREKT=0 on purpose. The bare
# SGLANG_HTCCL_BAR1_PIPE=1 -- pipe with the direct mode at its default 1 and
# the result ring at its default 2 -- DOES NOT BOOT this model: the decode
# graph capture warmup runs several all_reduce call sites back to back, the
# two eager result slots are still held, and `_erg_platz` raises
# Bar1Unverfuegbar rather than overwrite a buffer the caller still owns. That
# is the ring doing its job, not a bug, but it means the pipe lever can only
# be measured either without the direct mode (this arm) or with a ring big
# enough for the graph (the next one). Evidence:
# 2026-07-30_hebel/befunde/bar1pipe_bootfehler.txt
ARME_TABELLE=(
    "nccl|grundlinie||"
    "bar1|bar1||"
    "bar1pipe|bar1|SGLANG_HTCCL_BAR1_PIPE=1 SGLANG_HTCCL_BAR1_PIPE_DIREKT=0|"
    "bar1direkt|bar1|SGLANG_HTCCL_BAR1_PIPE=1 SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH=1 SGLANG_HTCCL_BAR1_PIPE_ERG_RING=5|"
    "bar1cp4096|bar1||--chunked-prefill-size 4096"
    "ncclpg|grundlinie||--cuda-graph-backend-prefill breakable"
    "bar1pg|bar1||--cuda-graph-backend-prefill breakable"
    # --- Prefill-Graph mit einer Reserve, die ihn auch bezahlt --------------
    # Die vier Arme oben teilen sich `--rank-auto-reserve-mib 3000,2700,2700`.
    # Diese Pins stammen aus einem Rezept OHNE Prefill-Graphen, und der Graph
    # laesst sich zwar einschalten, aber nicht daraus bezahlen: beide
    # pg-Arme kommen bis `Capture target prefill CUDA graph` und sterben dort
    # an einem OOM auf einer 20-GB-Karte. Eine hoehere Reserve verschiebt
    # Speicher aus dem KV-Pool in den Laufzeitanteil und ist damit NICHT mit
    # den Armen oben vergleichbar -- deshalb bringt dieser Block seine EIGENEN
    # zwei Kontrollarme mit, die sich von den pg-Armen in nichts als dem
    # Prefill-Graphen unterscheiden. Verglichen wird innerhalb des Blocks.
    # Das zweite Vorkommen von --rank-auto-reserve-mib gewinnt (argparse).
    "nccl_hi|grundlinie||--rank-auto-reserve-mib 4500,4200,4200"
    "bar1_hi|bar1||--rank-auto-reserve-mib 4500,4200,4200"
    "ncclpg_hi|grundlinie||--rank-auto-reserve-mib 4500,4200,4200 --cuda-graph-backend-prefill breakable"
    "bar1pg_hi|bar1||--rank-auto-reserve-mib 4500,4200,4200 --cuda-graph-backend-prefill breakable"
    # Der Falsifikator zum Befund des pg-Blocks. `bar1pg_hi` verliert den
    # ganzen BAR1-Vorsprung, und sein Log nennt den Grund selbst: waehrend
    # einer Graph-Aufzeichnung wird der gitter-Kern auf die 1blk-Variante
    # gelegt, weil ein Vorbehalt sagt, cudaLaunchCooperativeKernel sei
    # aufgezeichnet ungemessen. Phase A DIESES Laufs hat genau das gemessen
    # (Fall 'gitter' bestanden). Faellt der Vorbehalt, muss der Vorsprung
    # zurueckkommen -- und wenn nicht, ist die Erklaerung falsch.
    "bar1pggitter_hi|bar1|SGLANG_HTCCL_BAR1_GRAPH_GITTER=1|--rank-auto-reserve-mib 4500,4200,4200 --cuda-graph-backend-prefill breakable"
)
# A caller may narrow the list; the names must match column 1.
S13_NUR="${S13_NUR:-}"

mkdir -p "$DIR/belege" "$DIR/logs" "$DIR/wait"
DIR_HOST="$(host_path "$DIR")" || exit 2
DRIVER_HOST="$(host_path "$BATTERY_DIR/s12_prefill_kurve.py")" || exit 2
ANALYSE_HOST="$(host_path "$BATTERY_DIR/s12_log_analyse.py")" || exit 2

if ! host_reachable; then
    echo "STOP: Host $BAR1_HOST nicht erreichbar"; exit 2
fi
if ! bar1_require_integration; then exit 2; fi
if ! host_locks_acquire "${BATTERY_STEP:-s13}"; then
    echo "STOP: Host-Locks nicht zu bekommen"
    echo "Host-Locks fremd gehalten -- nicht gebrochen" > "$DIR/blocked.txt"
    exit 2
fi

SERVER_PID=""
HOSTPID=""

# host_wait_for_server only ever asks /health, so a server that DIED during
# startup keeps it polling for the whole budget -- 15 minutes of an arm's
# measurement window spent waiting for a process that is already gone. That is
# what the first attempt of this step spent on the bare-pipe arm. This wait
# asks both questions, and the dead process is the faster answer.
s13_warte_auf_server() {  # $1 = port, $2 = host pid, $3 = budget_s
    local port="$1" pid="$2" budget="${3:-600}" t0
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$budget" ]; do
        if host_ssh_for 40 "curl -sf -m 5 http://127.0.0.1:$port/health >/dev/null" \
            >/dev/null 2>&1; then
            echo "Host-Server oben nach $(( $(date +%s) - t0 ))s"
            return 0
        fi
        if ! host_ssh_for 40 "kill -0 $pid 2>/dev/null" >/dev/null 2>&1; then
            echo "Host-Server $pid ist gestorben, bevor er antwortete " \
                 "(nach $(( $(date +%s) - t0 ))s)" >&2
            return 2
        fi
        sleep 10
    done
    echo "Host-Server nicht oben in ${budget}s" >&2
    return 1
}

cleanup() {
    bar1_kill_host_server "$SERVER_PID" "${HOSTPID:-}" \
        "$DIR/pyspy-host-cleanup.txt" || true
    SERVER_PID=""
    host_locks_release
}
trap cleanup EXIT INT TERM

cat > "$DIR/remote_messen.sh" <<EOF
#!/usr/bin/env bash
# GENERATED by s13_hebel_messung.sh. Runs on the PVE host.
set -uo pipefail
/spinning/miniforge3_local_install/bin/python3.12 $DRIVER_HOST \\
  --mode messen --port $PORT --out-dir $DIR_HOST \\
  --point-seconds $POINT_S --warmup-seconds $WARMUP_S \\
  --prompt-tokens $PROMPT_TOKENS \\
  --arm "\$1" --sessions "\$2" --folge "\$3" --server-log "\$4" \\
  --with-decode "\$5"
EOF
chmod +x "$DIR/remote_messen.sh"

FOLGE=0
ABBRUCH=""

for RUNDE in $(seq "$RUNDE_START" $((RUNDE_START + RUNDEN - 1))); do
    for ZEILE in "${ARME_TABELLE[@]}"; do
        NAME="${ZEILE%%|*}"
        REST="${ZEILE#*|}"
        TARM="${REST%%|*}"
        REST="${REST#*|}"
        EENV="${REST%%|*}"
        EARGS="${REST#*|}"

        if [ -n "$S13_NUR" ]; then
            case " $S13_NUR " in *" $NAME "*) ;; *) continue ;; esac
        fi

        ARM="${NAME}_r${RUNDE}"
        FOLGE=$((FOLGE + 1))
        HOSTLOG="$BAR1_HOST_LOGDIR/s13.$ARM.log"
        HOSTPID="$BAR1_HOST_LOGDIR/s13.$ARM.pid"
        echo "== [$FOLGE] Runde $RUNDE, Arm $NAME (Transport $TARM) =="

        # No run against the leftovers of the previous boot -- with this many
        # boots a stale server would silently answer for several arms.
        if ! bar1_altlast_pruefen "$PORT" "$DIR/blocked.txt"; then
            ABBRUCH="Altlast vor $ARM"; break
        fi

        BAR1_EXTRA_ENV="$EENV" BAR1_EXTRA_ARGS="$EARGS" \
            bar1_write_boot_script "$DIR/remote_boot_${ARM}.sh" "$TARM" \
                "$HOSTLOG" "$HOSTPID" "$PORT" \
            || { ABBRUCH="Bootskript $ARM"; break; }

        SERVER_PID="$(bar1_boot_start "$DIR/remote_boot_${ARM}.sh" "$HOSTPID")" \
            || SERVER_PID=""
        if ! bar1_pid_ok "$SERVER_PID"; then
            SERVER_PID=""
            host_tail_into "$HOSTLOG" "$DIR/logs/${ARM}.tail.txt" 200
            echo "ohne pid" > "$DIR/logs/${ARM}.bootfehler.txt"
            echo "  ARM UEBERSPRUNGEN: Server-Start ohne pid"
            continue
        fi
        echo "$SERVER_PID" >> "$DIR/host_pids"

        # AN ARM THAT DOES NOT BOOT IS A RESULT, NOT AN ABORT. A configuration
        # this rig refuses is exactly what a lever survey is looking for, and
        # letting it take the remaining arms and the second round down with it
        # would trade six answers for one. The reason is harvested, the arm is
        # skipped, the run goes on.
        if ! s13_warte_auf_server "$PORT" "$SERVER_PID" 600; then
            host_grep_into "$HOSTLOG" "$DIR/logs/${ARM}.bootfehler.txt" \
                "Bar1Unverfuegbar" "ColdBuildWindowError" "CUDA out of memory" \
                "torch.OutOfMemoryError" "NCCL error" "Capture cuda graph failed" \
                "Received sigquit"
            host_tail_into "$HOSTLOG" "$DIR/logs/${ARM}.tail.txt" 200
            bar1_kill_host_server "$SERVER_PID" "$HOSTPID" \
                "$DIR/pyspy-host-$SERVER_PID.txt" || true
            SERVER_PID=""
            echo "  ARM UEBERSPRUNGEN: Server nicht oben (Grund in logs/${ARM}.bootfehler.txt)"
            sleep 20
            continue
        fi

        # ERREICHT= per group BEFORE any number: an arm whose second
        # communicator group quietly fell back is a mixed point, not a point.
        # The prefill-graph lines go into the same file -- an arm that asks for
        # the prefill graph and silently does not get it would otherwise be
        # reported as a prefill-graph measurement.
        host_grep_into "$HOSTLOG" "$DIR/belege/${ARM}.txt" \
            "HTCCL enabled for group" \
            "ERREICHT=" \
            "HTCCL-BAR1: Aufbau in" \
            "HTCCL-BAR1-PIPE:" \
            "Disable prefill CUDA graph" \
            "disabling prefill CUDA graph" \
            "prefill CUDA graph begin" \
            "prefill CUDA graph end" \
            "waehrend einer CUDA-Graph-Aufzeichnung"

        MRC=0
        for P in $PUNKTE; do
            N="${P%%:*}"
            WD="${P##*:}"
            echo "   Punkt sessions=$N decode=$WD"
            host_run_script 900 "$DIR/remote_messen.sh" "$ARM" "$N" "$FOLGE" \
                "$HOSTLOG" "$WD" >> "$DIR/messen.log" 2>&1
            MRC=$?
            [ "$MRC" != 0 ] && break
        done
        echo "  Messung rc=$MRC"

        host_grep_into "$HOSTLOG" "$DIR/logs/${ARM}.fatal.txt" \
            "CUDA out of memory" "torch.OutOfMemoryError" "NCCL error" \
            "Traceback (most recent call last)"

        # The compute/wait split, computed ON THE HOST where the log lives.
        # What comes back into the run directory is the aggregate, never the
        # log.
        host_ssh_for 300 "/spinning/miniforge3_local_install/bin/python3.12 \
            $ANALYSE_HOST --log '$ARM:8:$HOSTLOG' --punkte $DIR_HOST/punkte.jsonl \
            --json $DIR_HOST/wait/${ARM}.json" > "$DIR/wait/${ARM}.txt" 2>&1 || true

        host_tail_into "$HOSTLOG" "$DIR/logs/${ARM}.tail.txt" 120

        bar1_kill_host_server "$SERVER_PID" "$HOSTPID" \
            "$DIR/pyspy-host-$SERVER_PID.txt" || true
        SERVER_PID=""
        sleep 20

        # Same reasoning as the boot path: a point this configuration cannot
        # produce is an answer about that configuration, not a reason to stop
        # asking the others. `bar1cp4096` is the concrete case -- its
        # sessions=1 point measures fine and its sessions=8 point runs the
        # 3080s out of memory, because the pinned reserve was sized for
        # chunked_prefill_size=2048 and the GDN prefill scratch scales with the
        # chunk. Both halves of that are results.
        if [ "$MRC" != 0 ]; then
            printf 'Messung rc=%s\n' "$MRC" > "$DIR/logs/${ARM}.messfehler.txt"
            echo "  PUNKT FEHLT: Messung rc=$MRC (Grund in logs/${ARM}.fatal.txt)"
        fi
    done
    [ -n "$ABBRUCH" ] && break
    echo "== Runde $RUNDE fertig =="
done

cleanup
trap - EXIT INT TERM

if [ -n "$ABBRUCH" ]; then
    echo "abgebrochen: $ABBRUCH" | tee "$DIR/abbruch.txt"
    exit 1
fi
exit 0
