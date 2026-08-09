#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#652 probe: why does the 5090's CUDA context see less memory than NVML?

THE OBSERVATION THIS EXISTS TO ISOLATE. PP pool sizing calls
``_profile_available_bytes``, which computes reachable memory as
``used + free`` from ``torch.cuda.mem_get_info``. On this rig the 5090
reports only ~21.9 GiB reachable while ``nvidia-smi`` shows 32607 MiB
total -- roughly 13 GiB the sizer cannot see, and the PP pool is sized
BEFORE the TP stack exists, so that shortfall caps max_total_num_tokens
for the whole instance.

WHAT THE PROBE SEPARATES. Run it in a BARE process (no serving) and again
while serving is up. ``mem_get_info``'s SECOND element is the device total
as the CUDA context sees it; NVML's total is the card's physical total.
If they disagree in a bare process, the cause is boot- or driver-level
(carve-out, open-driver module state, BAR/MIG config) and no amount of
scheduler work fixes it. If they agree in a bare process and diverge only
inside a serving rank, the cause is process-level (visible-device mapping,
an env var, allocator state) and is ours to fix.

Deliberately reports per-card and does NOT create a context on cards it
was not asked about: a context costs a few hundred MiB and the rig runs a
1024 MiB free-memory corridor.

Usage:
    python scripts/probe_652_device_total.py            # all visible cards
    python scripts/probe_652_device_total.py 0          # one card
"""

import os
import sys

MIB = 1024 * 1024


def nvml_totals():
    """Physical totals and UUIDs straight from NVML, by PHYSICAL index."""
    try:
        import pynvml
    except ImportError:
        return None
    pynvml.nvmlInit()
    out = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        uuid = pynvml.nvmlDeviceGetUUID(h)
        out.append(
            {
                "index": i,
                "name": name if isinstance(name, str) else name.decode(),
                "uuid": uuid if isinstance(uuid, str) else uuid.decode(),
                "total_mib": info.total // MIB,
                "used_mib": info.used // MIB,
                "free_mib": info.free // MIB,
            }
        )
    pynvml.nvmlShutdown()
    return out


def main() -> int:
    import torch

    print("=== ENVIRONMENT ===")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_MPS_PIPE_DIRECTORY",
        "PYTORCH_CUDA_ALLOC_CONF",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        print(f"  {var}={os.environ.get(var, '<unset>')}")
    print(f"  torch={torch.__version__}  cuda={torch.version.cuda}")
    print(f"  driver_visible_device_count={torch.cuda.device_count()}")

    print("\n=== NVML (physical, by NVML index) ===")
    nv = nvml_totals()
    if nv is None:
        print("  pynvml unavailable")
    else:
        for d in nv:
            print(
                f"  [{d['index']}] {d['name']:<24} total={d['total_mib']:>6} MiB "
                f"used={d['used_mib']:>6} free={d['free_mib']:>6}  {d['uuid']}"
            )

    by_uuid = {d["uuid"]: d for d in (nv or [])}

    want = [int(a) for a in sys.argv[1:]] or list(range(torch.cuda.device_count()))
    print("\n=== TORCH CUDA CONTEXT (per visible device) ===")
    print(
        "  visible  name                      mem_get_info_total  "
        "props.total  nvml_total   SHORTFALL"
    )
    worst = 0
    for idx in want:
        torch.cuda.set_device(idx)
        # Force a real context; mem_get_info on a lazy device can report
        # the driver's view rather than the context's.
        torch.zeros(1, device=f"cuda:{idx}")
        free_b, total_b = torch.cuda.mem_get_info(idx)
        props = torch.cuda.get_device_properties(idx)
        uuid = f"GPU-{props.uuid}" if not str(props.uuid).startswith("GPU-") else str(props.uuid)
        phys = by_uuid.get(uuid)
        nvml_total = phys["total_mib"] if phys else -1
        ctx_total = total_b // MIB
        prop_total = props.total_memory // MIB
        short = nvml_total - ctx_total if nvml_total > 0 else 0
        worst = max(worst, short)
        print(
            f"  cuda:{idx}   {props.name:<24} {ctx_total:>10} MiB "
            f"{prop_total:>10} MiB {nvml_total:>10} MiB {short:>+9} MiB"
        )
        print(
            f"           free={free_b // MIB} MiB   uuid={uuid}   "
            f"nvml_index={phys['index'] if phys else '?'}"
        )

    print("\n=== VERDICT ===")
    if worst <= 64:
        print(
            "  Context total matches NVML total on every probed card "
            "(<=64 MiB slack). No #652 shortfall in THIS process."
        )
    else:
        print(
            f"  SHORTFALL PRESENT: up to {worst} MiB invisible to the CUDA "
            "context. Compare a bare run against a run inside a serving "
            "rank to place the cause at process or boot level."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
