#!/bin/bash
# Post-cleanup smoke: does omp still resolve its config and drive a tool call
# as user efeu, after the probe provider was removed from models.yml?
OUT=/root/651-p2/results/smoke655_$(date +%H%M%S).txt
exec > >(tee -a "$OUT") 2>&1
W=/home/efeu/omp655smoke-$$
mkdir -p "$W"; chown efeu:efeu "$W"
su - efeu -c "
export PATH=\$HOME/.local/bin:\$PATH
cd 
timeout 900 omp --model local/qwen36-35b-a3b --no-lsp --no-skills --tools=write,bash --auto-approve -p \
  Write a file hello.py that prints exactly SMOKE_OK, then run it with python3 and report the output.
"
echo "FILES: $(ls -A "$W")"
R=$(cd "$W" && timeout 30 python3 hello.py 2>&1)
echo "independent run: $R"
case "$R" in *SMOKE_OK*) echo "SMOKE655: PASS";; *) echo "SMOKE655: FAIL";; esac
echo "resets: $(dmesg -T | grep -c "GPU reset(")"
