#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Nordstern L1 -- drive scripts/nordstern/l1_ucx_crossrig.py on both rigs.
#
# Stages the two transport modules plus the driver to each host and starts one
# rank per host: rank 0 on the main rig (<RDMA_NET>.1), rank 1 on the second rig
# (<RDMA_NET>.2). CPU tensors only -- no GPU is touched, so this is safe to run
# while a GPU window is busy on either machine.
#
# Control plane and data plane are deliberately separate:
#   * gloo rendezvous  -> the 1 GbE LAN (<RIG_LAN>.x)
#   * UCX collectives  -> the 40G RoCE link, UCX_TLS=rc with no tcp fallback
# so a run that passes has demonstrably moved its bytes over RDMA. If RDMA is
# broken the run fails rather than quietly degrading to TCP.
#
# VERSION PARITY is mandatory: UCX peers must run the same release or endpoint
# creation fails with the useless 'invalid bandwidth 0.00'. The main rig ships
# 1.18.1 while the second rig ships 1.16.0, so rank 0 is pointed at the
# side-by-side 1.16.0 build via SGLANG_HTCCL_UCX_LIB. The transport checks this
# itself at rendezvous and refuses with instructions; --mismatch exercises that.
#
# Hosts, keys and interpreter paths are read from the environment
# (RIG1_HOST, RIG2_HOST, RIG2_KEY, RIG2_VENV, PVE_MINIFORGE, MASTER_ADDR).
# Source your local rig env file first; unset variables fall back to
# placeholders and the run fails at the first ssh rather than silently
# reaching some other machine.
#
# Usage:
#   ./l1_ucx_crossrig.sh              # correctness + throughput over RDMA
#   ./l1_ucx_crossrig.sh --mismatch   # prove the parity check rejects 1.18 vs 1.16
#   ./l1_ucx_crossrig.sh --reps 5     # extra args go to both ranks
#   EXTRA_ENV=SGLANG_HTCCL_UCX_PIPELINE=0 ./l1_ucx_crossrig.sh   # A/B control
set -uo pipefail

# Site-specific values come from the environment, never from a default baked
# into the repo: `source` your local rig env file before running this. The
# fallbacks below are placeholders, so an unsourced run fails visibly instead
# of talking to whatever host happened to be hard-coded here.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
COMM_SRC="$REPO_ROOT/python/sglang/srt/distributed/device_communicators"
DRIVER="$REPO_ROOT/scripts/nordstern/l1_ucx_crossrig.py"

# --- rank 0: main rig ------------------------------------------------------
R0_ADDR="${RIG1_HOST:-<RIG1_IP>}"
R0_SSH=(ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "root@$R0_ADDR")
R0_SCP=(-o BatchMode=yes -o StrictHostKeyChecking=no)
R0_HOST="root@$R0_ADDR"
R0_PY="${PVE_MINIFORGE:-<PVE_MINIFORGE>}/bin/python3"
R0_LAN_IF=vmbr0
R0_IB=rocep4s0f1:1
# The parity workaround: 1.16.0 side-by-side install matching the second rig.
R0_UCX_LIB=/opt/ucx116/lib/libucp.so.0

# --- rank 1: second rig ----------------------------------------------------
R1_ADDR="${RIG2_HOST:-<RIG2_IP>}"
R1_KEY="${RIG2_KEY:-<RIG2_SSH_KEY>}"
R1_SSH=(ssh -n -i "$R1_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no "root@$R1_ADDR")
R1_SCP=(-i "$R1_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no)
R1_HOST="root@$R1_ADDR"
R1_PY="${RIG2_VENV:-<RIG2_VENV>}/bin/python"
R1_LAN_IF=enp7s0
R1_IB=rocep1s0f1:1

MASTER_ADDR="${MASTER_ADDR:-<MASTER_ADDR>}"
PORT="${PORT:-$((29700 + RANDOM % 200))}"
STAGE="${L1_STAGE_DIR:-/root/htccl-ucx-l1}"
MODE_FLAG="--bench"
if [[ "${1:-}" == "--mismatch" ]]; then
  MODE_FLAG="--expect-version-mismatch"
  R0_UCX_LIB=""   # let rank 0 load its native 1.18.1 -> a real mismatch
  shift
fi
# Anything left on the command line goes to BOTH ranks (e.g. --reps 5).
EXTRA_ARGS=("$@")
# EXTRA_ENV is applied to both ranks identically -- the A/B control for the
# pipelining measurement is EXTRA_ENV="SGLANG_HTCCL_UCX_PIPELINE=0". Both
# sides must agree: the two paths post the same tags in the same order, but
# only a group that is uniformly configured is a meaningful measurement.
EXTRA_ENV="${EXTRA_ENV:-}"

echo "== staging to both rigs (port $PORT) =="
for spec in "R0" "R1"; do
  host_var="${spec}_HOST"; scp_var="${spec}_SCP[@]"; ssh_var="${spec}_SSH[@]"
  "${!ssh_var}" "mkdir -p $STAGE" || { echo "FATAL: cannot reach ${!host_var}"; exit 1; }
  scp "${!scp_var}" "$COMM_SRC/htccl_ucx.py" "$COMM_SRC/htccl_ucx_bindings.py" \
      "$DRIVER" "${!host_var}:$STAGE/" >/dev/null || { echo "FATAL: stage failed"; exit 1; }
done

echo "== launching rank 1 (second rig, <RDMA_NET>.2) =="
# `timeout` is load-bearing, not defensive. ssh holds the session open until
# the channel closes, and a backgrounded setsid does not close it even with
# all three fds redirected -- so this call never returns on its own. The rank
# survives the ssh teardown precisely because setsid detached it, so a
# timeout here is the normal, expected exit path.
timeout 25 "${R1_SSH[@]}" "cd $STAGE && setsid env \
  GLOO_SOCKET_IFNAME=$R1_LAN_IF UCX_TLS=rc,self,sm UCX_IB_GID_INDEX=3 UCX_NET_DEVICES=$R1_IB \
  $EXTRA_ENV \
  $R1_PY l1_ucx_crossrig.py --rank 1 --world 2 --master-addr $MASTER_ADDR \
  --master-port $PORT --comm-dir $STAGE $MODE_FLAG ${EXTRA_ARGS[*]} \
  </dev/null >$STAGE/r1.log 2>&1 & echo rank1-launched" || true
sleep 4

echo "== rank 0 (main rig, <RDMA_NET>.1) =="
R0_LIB_ENV=""
[[ -n "$R0_UCX_LIB" ]] && R0_LIB_ENV="SGLANG_HTCCL_UCX_LIB=$R0_UCX_LIB"
"${R0_SSH[@]}" "cd $STAGE && env \
  GLOO_SOCKET_IFNAME=$R0_LAN_IF UCX_TLS=rc,self,sm UCX_IB_GID_INDEX=3 UCX_NET_DEVICES=$R0_IB \
  $R0_LIB_ENV $EXTRA_ENV \
  $R0_PY l1_ucx_crossrig.py --rank 0 --world 2 --master-addr $MASTER_ADDR \
  --master-port $PORT --comm-dir $STAGE $MODE_FLAG ${EXTRA_ARGS[*]}"
RC=$?

echo "== rank 1 log =="
"${R1_SSH[@]}" "cat $STAGE/r1.log"

echo "== rank 0 exit: $RC =="
exit $RC
