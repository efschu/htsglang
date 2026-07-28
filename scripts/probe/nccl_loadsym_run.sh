#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Container-Seite des symmetrischen Lastvergleichs (04_OFFEN 1a): startet das
# Mess-Paar und -- bei "mit" -- das Last-Paar aus nccl_loadsym.py auf demselben
# Kartenpaar. Laeuft im CONTAINER (torch nur dort); die NIC-Seite desselben
# Vergleichs faehrt Phase loadsym von crossrig_ladder_run.sh auf dem PVE-Host.
#
# Dieses Skript nimmt KEINE Kartenlocks -- der Aufrufer haelt das Fenster
# (Locks + /spinning/gpu-arb/holder) und startet erst dann.
#
# Aufruf: nccl_loadsym_run.sh <ord_a> <ord_b> <ohne|mit> <out.json>
#   ord_a/ord_b  CUDA-Ordinale bei CUDA_DEVICE_ORDER=PCI_BUS_ID
#                (= nvidia-smi-Index); Rang r bindet Karte r der Liste.
#   Last-Protokoll landet in <out ohne .json>.load.log (nur Rang 0 schreibt).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="${LS_PYTHON:-/spinning/htsglang-gpu/.venv/bin/python}"
ORD_A=${1:?ord_a}; ORD_B=${2:?ord_b}; LOAD=${3:?ohne|mit}; OUT=${4:?out.json}
SECS="${LS_SECS:-5}"

STOP="/tmp/nccl_loadsym_stop.$$"
rm -f "$STOP"
LOADLOG="${OUT%.json}.load.log"

run_rank() { # <role> <rank> <port> <logfile> [extra...]
  local role="$1" rank="$2" port="$3" logf="$4"; shift 4
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$ORD_A,$ORD_B" TORCH_NCCL_BLOCKING_WAIT=1 \
    "$PY" "$HERE/nccl_loadsym.py" --role "$role" --rank "$rank" --world 2 "$@" \
    > "$logf" 2>&1 &
}

lpids=()
if [ "$LOAD" = "mit" ]; then
  PORT_L=$((27500 + RANDOM % 2000))
  # Obergrenze grosszuegig; das echte Ende setzt die Stop-Datei.
  run_rank load 0 "$PORT_L" "$LOADLOG"      --secs $((SECS * 4 + 90)) --stop-file "$STOP"
  lpids+=($!)
  run_rank load 1 "$PORT_L" "$LOADLOG.r1"   --secs $((SECS * 4 + 90)) --stop-file "$STOP"
  lpids+=($!)
  sleep 6   # Last anlaufen lassen, damit die Messung nicht in die Rampe faellt
fi

PORT_M=$((29500 + RANDOM % 2000))
mpids=()
run_rank measure 0 "$PORT_M" "${OUT%.json}.m0.log" --secs "$SECS" --out "$OUT"
mpids+=($!)
run_rank measure 1 "$PORT_M" "${OUT%.json}.m1.log" --secs "$SECS"
mpids+=($!)

rc=0
wait "${mpids[@]}" || rc=1
touch "$STOP"
if [ "${#lpids[@]}" -gt 0 ]; then
  wait "${lpids[@]}" 2>/dev/null || true
fi
rm -f "$STOP"

if [ "$rc" != "0" ] || [ ! -r "$OUT" ]; then
  echo "[FAIL] Messung ohne Ergebnis -- letzte Zeilen:" >&2
  tail -3 "${OUT%.json}.m0.log" >&2
  exit 1
fi
grep -h '^LOADSUM' "$LOADLOG" 2>/dev/null || true
exit 0
