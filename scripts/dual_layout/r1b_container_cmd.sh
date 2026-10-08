#!/bin/bash
# DUAL-TP3PP3 risk 1b IN THE RELEASE IMAGE (container, like host_acceptance's bar1probe): same bench, image tree src-27b @ 2c3eb0ab.
ssh -o BatchMode=yes root@proxmox 'docker run --rm --name dual-r1b-bench --gpus all --security-opt apparmor=unconfined \
  -v /sys/devices:/sys/devices --shm-size=4g --ulimit memlock=-1:-1 --init --device /dev/dmabuf_holder \
  -v /spinning/subvol-999-disk-0/root/.claude/jobs/1ab4cd30/tmp/dual-notes:/out -v /spinning/subvol-999-disk-0/spinning/nvidia-open-595:/opt/nvidia-open-595:ro \
  -e PY=/opt/venv/bin/python3 -e DUR=6 -e TO=200 -e FLLIPER_BARLINK_BAR1_NV_SOURCE=/opt/nvidia-open-595 -e HTSGLANG_TRANSPORT=bar1 --cap-add SYS_PTRACE \
  --entrypoint bash -w /opt/htsglang/src-27b htsglang:cu130-weg2-rc12z30y3dual-27b-nf \
  -c "bash scripts/dual_layout/run_bar1_two_groups.sh /out/r1b_docker_$(date -u +%H%M)"'
