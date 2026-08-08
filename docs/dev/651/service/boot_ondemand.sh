#!/bin/bash
# #651: the boot the on-demand service uses. One operating point, settled by
# measurement elsewhere in this strand -- this script exists so that point is
# written down once instead of living in a systemd Environment= line.
#
# THE OPERATING POINT AND WHY:
#   speculation OFF   NEXTN draft cost is structural on this GPU, not a defect
#                     (falsified in ef8e66d); it lost throughput on every shape.
#   decode graphs ON  +12.8% decode (15.5 tok/s). The capture costs ~2.5 s of
#                     the load and 0.1 GB, both of which the service pays once.
#   memfrac 0.99      NOT a safety margin being spent. The KV budget is
#                     mem_fraction_static x total MINUS the weights, so on this
#                     APU the fraction is what CONVERTS free GTT into context:
#                     0.97 -> 7254 tokens, 0.99 -> 15070. The loader refuses
#                     anything that leaves no KV at all, which puts a hard floor
#                     just under this value; see FINAL_651.md.
#   chunked prefill   capped at 256 by wedge_policy.py -- the amdgpu MES wedge
#                     is a firmware hang triggered by the prefill-chunk-sized
#                     bf16 GEMM. This is a MITIGATION, not a fix.
#
# Both guards run on EVERY load, which is the point of routing the service
# through this script rather than through a saved command line: an on-demand
# service reloads the model many times a day, and a guard that only ran at
# first boot would be a guard that mostly does not run.
set -u

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12

source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_src/python

# HiCache: the file backend's staging tier. The pinned-host guard reserves
# 10 GiB by default -- a figure sized for the rig, where RAM is plentiful and
# there is no swap. This laptop has 29.5 GiB total with ~22.7 GiB held by GGUF
# weights through GTT, so that default refuses a 0.15 GB staging pool the
# machine can trivially afford. The tier is a STAGING layer for the disk
# backend, NOT a RAM cache: it is the smallest the code can express.
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-/var/lib/hicache/kv}
export SGLANG_PINNED_HOST_RESERVE_MIB=${SGLANG_PINNED_HOST_RESERVE_MIB:-512}

# Give the page cache back before asking for 22.7 GiB. On this APU the GPU
# allocates from host RAM through GTT, so cache the kernel is holding is memory
# the model cannot have -- and the margin here is thin enough that it decides
# whether a load succeeds: the SAME memfrac has both produced a 15070-token
# pool and been refused outright, run to run, purely on how much RAM happened
# to be free. Reading a 21.6 GiB checkpoint fills that cache every load, so an
# on-demand service that reloads all day has to drop it every time.
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || \
  echo "note: could not drop caches (not root?); load margin will be tighter"

PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard_v2.py || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

CHUNKED_PREFILL=${CHUNKED_PREFILL:-256}
python /root/651-p2/scripts/wedge_policy.py "$CHUNKED_PREFILL" || {
  echo "Wedge policy refused this configuration"; exit 1; }

MODEL=/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
MEMFRAC=${MEMFRAC:-0.99}
PORT=${PORT:-31661}
CTX=${CTX:-8192}

# HiCache is OFF by default on this machine, and the reason is physical rather
# than a policy choice. The pools build fine -- the GDN hybrid passes the host
# pool's model check, which was the thing in doubt -- but the host tier needs
# PINNED RAM that this laptop does not reliably have while 22.7 GiB of weights
# sit in GTT. Two upstream defects were fixed getting this far (the 10 GiB
# pinned-host reserve is now configurable; the ROCm layout collision is pinned
# to page_first_direct), and the remaining wall is the machine, not the code.
# HICACHE=1 re-arms it for a host with RAM to spare; see FINAL_651.md.
HICACHE_ARGS=()
if [ "${HICACHE:-0}" = "1" ]; then
  HICACHE_ARGS=(
    --enable-hierarchical-cache
    --hicache-storage-backend file
    # --hicache-size is an int with a 1 GB minimum, so 0 falls through to the
    # ratio, which is a float of the device pool -- the only way to express a
    # staging tier smaller than a gigabyte.
    --hicache-size 0
    # 0.1, not the 0.5 default. The host tier has TWO posts on a GDN
    # hybrid, and the mamba one is chunky: a single mamba slot is ~64 MB,
    # so ratio 0.5 asked for 0.58 GB of mamba pool alone against ~1.08 GB
    # of free host RAM. 0.1 buys the one slot the staging path needs and
    # nothing more, which is the intent -- this is a STAGING tier for the
    # disk backend, so its job is to be big enough to hand pages through,
    # not to cache anything.
    --hicache-ratio "${HICACHE_RATIO:-0.1}"
    --hicache-write-policy write_through
    # The layout is forced, not defaulted, and only one value works here.
    # Two constraints meet: MambaPoolHost (this is a GDN hybrid) accepts ONLY
    # page_first_direct, while on ROCm the default page_first+kernel pair is
    # auto-downgraded to layer_first because the page_first write-back needs a
    # CUDA-only JIT kernel. Left to the defaults those two rules collide and
    # the boot dies with "MambaPoolHost only supports layout='page_first_direct',
    # got 'layer_first'". Naming page_first_direct also switches the io backend
    # to direct, which is the pairing the layout requires.
    --hicache-mem-layout page_first_direct
  )
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-qwen36-35b-a3b}" \
  --tokenizer-path /root/lh/models \
  --load-format gguf \
  --quantization gguf \
  --device cuda \
  --tp-size 1 \
  --context-length "$CTX" \
  --max-running-requests 1 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule \
  --page-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  "${HICACHE_ARGS[@]}" \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
