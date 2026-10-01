#!/bin/bash
# Measure the coding agent's real prompt size per flag combination, offline.
#
# Tokenised with the SAME tokenizer the server uses (/root/lh/models), not an
# estimate -- the whole point is to compare against a wedge envelope measured
# in server-side tokens.
set -u
OUT=/root/651-p2/results/omp_promptsize_$(date +%H%M%S).txt
exec > >(tee -a "$OUT") 2>&1

rm -f /root/651-p2/results/omp_capture_*.json

run_combo() {
  local label="$1"; shift
  local flags="$*"
  rm -f /root/651-p2/results/omp_capture_*.json
  su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd /tmp
timeout 90 omp --model probe/qwen36-35b-a3b $flags -p 'say hi' >/dev/null 2>&1
" || true
  sleep 1
  local cap
  cap=$(ls -t /root/651-p2/results/omp_capture_*.json 2>/dev/null | head -1)
  if [ -z "$cap" ]; then
    echo "$label -> NO CAPTURE (client refused before sending)"
    return
  fi
  /root/lh/venv/bin/python - "$cap" "$label" <<'PY'
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/lh/models", trust_remote_code=True)
body = json.load(open(sys.argv[1]))
parts = []
for m in body.get("messages", []):
    c = m.get("content")
    parts.append(c if isinstance(c, str) else json.dumps(c))
tools = body.get("tools", [])
msg_tokens = len(tok.encode("\n".join(parts)))
tool_tokens = len(tok.encode(json.dumps(tools))) if tools else 0
print("%-38s messages=%6d tools=%2d tool_tokens=%6d TOTAL=%6d"
      % (sys.argv[2], msg_tokens, len(tools), tool_tokens, msg_tokens + tool_tokens))
PY
}

echo "=== omp prompt size by flag combination $(date -Is) ==="
run_combo "default"                     "--no-lsp"
run_combo "no-lsp,no-skills"            "--no-lsp --no-skills"
run_combo "4 tools"                     "--no-lsp --no-skills --tools=read,write,edit,bash"
run_combo "2 tools"                     "--no-lsp --no-skills --tools=write,bash"
run_combo "4 tools + tiny sysprompt"    "--no-lsp --no-skills --no-rules --tools=read,write,edit,bash --system-prompt='You are a terse coding assistant. Use the tools to edit files and run commands. Keep replies short.'"
run_combo "2 tools + tiny sysprompt"    "--no-lsp --no-skills --no-rules --tools=write,bash --system-prompt='You are a terse coding assistant. Use the tools to edit files and run commands. Keep replies short.'"
echo "=== done -> $OUT ==="
