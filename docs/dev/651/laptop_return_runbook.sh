#!/bin/bash
# #651: laptop efeu-TP14 return runbook. Run FROM THE RIG when the return
# watch fires. Order is load-bearing (poisoned-state law, HANDOFF 12.3).
#
#   docs/dev/651/laptop_return_runbook.sh          # full sequence
#   docs/dev/651/laptop_return_runbook.sh --no-reboot   # if it just rebooted
set -u
SSH="ssh -i /root/.ssh/id_ed25519_root@192.168.0.116 -o BatchMode=yes root@192.168.0.116"
TREE=/spinning/wt-gguf-q4-651

if [ "${1:-}" != "--no-reboot" ]; then
  echo "== 1. reboot (poisoned-state law: never measure after suspend/resume) =="
  $SSH 'systemd-run --on-active=3 --timer-property=AccuracySec=1s /usr/bin/systemctl reboot -i' || exit 1
  old_boot=$($SSH 'uptime -s' 2>/dev/null)
  sleep 45
  for i in $(seq 1 30); do
    b=$($SSH 'uptime -s' 2>/dev/null) && [ -n "$b" ] && [ "$b" != "$old_boot" ] && break
    sleep 12
  done
  echo "rebooted: $($SSH 'uptime -s')"
fi

echo "== 2. mask sleep targets: a headless target must NEVER auto-suspend =="
$SSH 'systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target && mkdir -p /etc/systemd/logind.conf.d && printf "[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nIdleAction=ignore\n" > /etc/systemd/logind.conf.d/651-no-sleep.conf && systemctl restart systemd-logind && echo SLEEP-MASKED'

echo "== 3. GPU sanity guard (must PASS on the fresh boot) =="
$SSH 'cd /root/lh/ggufbuild && source /root/lh/venv/bin/activate && HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard.py' | grep -v libdrm || exit 1

echo "== 4. recover the #644 _host_verify pyc (unversioned bytecode) =="
scp -q -i /root/.ssh/id_ed25519_root@192.168.0.116 \
  root@192.168.0.116:/root/lh/sglang_src/python/sglang/srt/__pycache__/_host_verify.cpython-312.pyc \
  "$TREE/docs/dev/651/recovered/laptop_sglang_delta/_host_verify.cpython-312.pyc.laptop" \
  && echo "pyc recovered into the worktree (commit it)"

echo "== 5. queued falsifiers, in order =="
cat << 'EOF'
  a) Device-weight byte verification (early-materialization suspect):
     scp docs/dev/651/verify_device_weights.py to /root/651-p2/scripts/, then
     on the laptop: cd /root/651-p2 && source /root/lh/venv/bin/activate &&
       HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/verify_device_weights.py \
         /root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf /root/lh/models \
         2>&1 | tee results/verify_device_weights_$(date +%H%M%S).txt
  b) GDN triton kernels vs torch-naive reference (prime forward suspect):
     scp test/registered/cpu/test_mamba.py + test utils to the laptop, run the
     three GPU tests under HSA_OVERRIDE_GFX_VERSION=11.0.0 with the laptop
     tree's PYTHONPATH; add a parametrization matching the 35B GDN geometry.
  c) If (a) and (b) are both clean: activation bisection
     (llama-eval-callback reference vs debug_tensor_dump).
EOF
