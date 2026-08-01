#!/usr/bin/env bash
# Sample CUDA-graph capture progression from a host-side container log.
#
# Why sampling and not log parsing: the capture progress is a tqdm bar written
# with \r and carries no timestamps of its own, so the only way to get a cost
# model out of it is to observe it from outside at a known wall clock. Every
# tick records (wall seconds since start, current graph batch size, percent),
# which turns "capture took N minutes" into ms-per-graph -- the number that
# decides whether capturing fewer sizes is a viable mitigation or whether the
# per-graph cost itself has to be fixed.
#
# Usage: capture_sampler.sh <host-log-path> <out-csv> <interval-s> <max-s>
set -u
LOG="${1:?host log}"; OUT="${2:?out csv}"; IV="${3:-10}"; MAX="${4:-2400}"
HOST="${BAR1_HOST:-192.168.0.1}"; KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
T0=$(date +%s)
echo "elapsed_s,bs,pct,phase" > "$OUT"
while : ; do
    NOW=$(date +%s); EL=$(( NOW - T0 ))
    [ "$EL" -gt "$MAX" ] && break
    LINE=$(timeout 20 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=8 "root@$HOST" \
        "tail -c 4000 '$LOG' 2>/dev/null | tr '\r' '\n' | grep -oE 'Capturing batches \(bs=[0-9]+[^)]*\): +[0-9]+%' | tail -1" 2>/dev/null)
    PHASE=$(timeout 20 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=8 "root@$HOST" \
        "grep -oE 'Capture (target verify|draft [a-z]*|[a-z ]*) CUDA graph (begin|end)' '$LOG' 2>/dev/null | tail -1" 2>/dev/null)
    BS=$(echo "$LINE" | grep -oE 'bs=[0-9]+' | cut -d= -f2)
    PCT=$(echo "$LINE" | grep -oE '[0-9]+%' | tr -d '%')
    echo "$EL,${BS:-},${PCT:-},${PHASE:-}" >> "$OUT"
    # Stop as soon as the server answers: capture is over.
    if timeout 12 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=8 "root@$HOST" \
        "curl -s -m 5 http://127.0.0.1:${PORT:-30370}/health_generate >/dev/null 2>&1"; then
        echo "$EL,,,SERVER_READY" >> "$OUT"; break
    fi
    sleep "$IV"
done
