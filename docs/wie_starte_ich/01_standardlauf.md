# 01 — Standardlauf starten (erprobtes Rezept, lief mehrfach durch)

## 0. Vorbedingungen pruefen (STOP statt Boot, wenn eine reisst)

```bash
# Karten frei? (jede Karte < 400 MiB belegt, sonst STOP)
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Locks frei? (Verzeichnisse! fremde Locks NIE brechen)
ls -d /tmp/gpu-card-*.lock 2>/dev/null && echo "BELEGT -> STOP" || echo frei

# Locks nehmen (atomar, je Karte 0..2):
for n in 0 1 2; do mkdir /tmp/gpu-card-$n.lock || exit 1; \
  echo "holder=<dein-agent-name> seit $(date -Is)" > /tmp/gpu-card-$n.lock/info; done
```

## 1. Umgebung (exakt so, keine Abweichung)

```bash
cd /spinning/htsglang          # oder der zugewiesene Worktree
source /root/rig-env.sh        # PFLICHT: echte Hosts/Pfade (Repo hat nur Platzhalter)

NVRTC=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$NVRTC:${LD_LIBRARY_PATH:-}"   # libnvrtc.so.13 fuer deep_gemm
export PYTHONPATH=$PWD/python
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
```

## 2. Start (setsid + Logfile, NIE im Vordergrund)

```bash
LOG=/root/.claude/jobs/<deine-job-id>/tmp/std_run.log   # eigener tmp, NICHT /tmp
setsid /spinning/htsglang-gpu/.venv/bin/python -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8 \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 3000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port 30030 \
  > "$LOG" 2>&1 &
echo $! > /root/wie_starte_ich/.server_pid   # eigene PID merken
```

Feste Fakten dazu:
- `cuda:0` = 5090, `cuda:1`/`cuda:2` = 3080 (NVML-Reihenfolge weicht ab —
  nie ueber nvidia-smi-Indizes auf torch-Indizes schliessen).
- `--rank-auto-reserve-mib`: **2700 auf den 3080ern ist die belegte
  Untergrenze.** 2200 kippt (OOM im GDN-Prefill-Scratch beim ersten
  2048er-Chunk; der 80-Token-Warmup ueberlebt und taeuscht Erfolg vor).
- `--enable-metrics` ist PFLICHT in jedem Boot (Dashboard sonst blind).
- Boot-Dauer bis READY: normal 60-120 s (JIT-Kaltbau kann laenger sein).

## 3. Ready-Check (pollen mit Frist, nie blind warten)

```bash
for i in $(seq 1 60); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)
  [ "$code" = 200 ] && echo READY && break
  sleep 5
done
# nach 300 s ohne READY: py-spy dump --pid $(cat .server_pid), dann Abbruch melden
```

## 4. Smoke-Request (Funktion + Spec pruefen)

```bash
curl -s -m 60 http://127.0.0.1:30030/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Zaehle von 1 bis 20."}],
    "max_tokens": 128, "temperature": 0
  }' > /root/wie_starte_ich/.smoke.json
python3 -c "import json;d=json.load(open('/root/wie_starte_ich/.smoke.json'));\
print(d['choices'][0]['message']['content'][:200]);\
print('accept:',d['choices'][0].get('meta_info',{}).get('spec_accept_length'))"
```
Gesund: kohaerenter Text UND `spec_accept_length` als Zahl (~2,5-3,5 bei
diesem Prompt). Fehlt das Feld oder ist der Text Muell -> Lauf ist NICHT
gesund, melden statt weitermachen.

## 5. Sauber beenden + freigeben (immer, auch nach Fehlern)

```bash
PID=$(cat /root/wie_starte_ich/.server_pid)
py-spy dump --pid "$PID" > /dev/null 2>&1 || true   # nur bei Haenger relevant
kill "$PID"; sleep 5; kill -9 "$PID" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used --format=csv   # muss auf ~0 zurueck
for n in 0 1 2; do rm -rf /tmp/gpu-card-$n.lock; done
# Freigabe AUSDRUECKLICH im Bericht melden.
```
