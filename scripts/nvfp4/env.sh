#!/usr/bin/env bash
# Shared environment for the #332 + families follow-up proof run.
source /root/rig-env.sh 2>/dev/null || true
export WT=/spinning/wt-beleg-332fam
export OUT=/spinning/gpu-battery-results/2026-07-31_332_fam_beleg
export PORT=30332
export V4_MODEL="$MODEL_ROOT/Qwen3.6-27B-NVFP4"
export PY="$VENV/bin/python"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Card identity is resolved to UUIDs, never to indices: CUDA_VISIBLE_DEVICES is
# read in CUDA enumeration order, which is not NVML order on this rig (NVML 1 is
# the 5090, CUDA 0 is). A UUID names the same card in both order systems.
resolve_uuids() {
  python3 - <<'PY'
import pynvml
pynvml.nvmlInit()
five, threes = None, []
for i in range(pynvml.nvmlDeviceGetCount()):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    uuid = pynvml.nvmlDeviceGetUUID(h)
    name = name.decode() if isinstance(name, bytes) else name
    uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
    print(f"# nvml[{i}] {name} {uuid}", flush=True)
    if "5090" in name:
        five = uuid
    elif "3080" in name:
        threes.append(uuid)
pynvml.nvmlShutdown()
assert five is not None, "no RTX 5090 found via NVML"
print(f"FIVE={five}")
print(f"THREES={','.join(threes)}")
PY
}
