"""Memory budget for the #651 bring-up on the APU laptop (ThinkPad P14s Gen5 AMD).

Every constant here is MEASURED on the target machine (2026-08-08) or measured
from the checkpoint by vram_budget.py / dequant_cost.py -- none is assumed.
Re-run the PROBE block on the laptop if the machine changes.

    python3 docs/dev/651/apu_budget.py

PROBE (on the laptop, read-only):
    grep -E '^(MemTotal|MemAvailable)' /proc/meminfo
    cat /sys/class/drm/card*/device/mem_info_vram_total
    cat /sys/class/drm/card*/device/mem_info_gtt_total
    cat /proc/cmdline            # amdgpu.gttsize=, ttm.pages_limit=
"""

MiB = 1048576

# --- measured on the laptop, 2026-08-08 -------------------------------------
MEM_TOTAL_KB = 30935848      # /proc/meminfo MemTotal
MEM_AVAIL_KB = 29253260      # /proc/meminfo MemAvailable, desktop mostly idle
VRAM_B = 1073741824          # mem_info_vram_total: BIOS UMA carve-out.
                             # 1 GiB is the BIOS MINIMUM on this board (user,
                             # 2026-08-08) -- it cannot be set lower. Good:
                             # a small carve-out leaves more to GTT + OS.
GTT_B = 25769803776          # mem_info_gtt_total, pinned by amdgpu.gttsize=24576
                             # (ttm.pages_limit=6291456 pages x 4 KiB agrees)

# --- checkpoints ------------------------------------------------------------
CKPT = {
    "Q4_K_M  (laptop /root/lh/models)": 22663387424,
    "Q2_K_XL (laptop /root/lh/models)": 12574128416,
    "Q4_K_XL (rig, not on laptop)":     22853663008,
}

# --- from the checkpoint (vram_budget.py / dequant_cost.py) -----------------
VISION_TOWER_MIB = 818.0     # #651b: constructed unconditionally, never fed
KV_FP16_KIB_TOK = 20.0       # 10 full-attention layers of 40
KV_FP8_KIB_TOK = 10.0
GDN_PER_SEQ_FP32 = 61.9
GDN_PER_SEQ_BF16 = 31.9      # SGLANG_MAMBA_SSM_DTYPE=bfloat16
DENSE_EXPANSION = 3.17       # measured: 506.2 -> 1604.2 MiB per layer

# Reserve inside the GPU-addressable window for activations, the K-quant
# dequant scratch (prefill-batch dependent, unpriced -- see HANDOFF section 4.4)
# and allocator slack. A working assumption, not a measurement.
RUNTIME_RESERVE_MIB = 1500.0


def main() -> None:
    mem_total = MEM_TOTAL_KB / 1024
    mem_avail = MEM_AVAIL_KB / 1024
    vram = VRAM_B / MiB
    gtt = GTT_B / MiB
    ceiling = vram + gtt

    print("=== machine (measured) ===")
    print(f"  MemTotal            {mem_total:9.0f} MiB  ({mem_total/1024:5.2f} GiB)")
    print(f"  MemAvailable        {mem_avail:9.0f} MiB  ({mem_avail/1024:5.2f} GiB)")
    print(f"  VRAM (BIOS UMA min) {vram:9.0f} MiB")
    print(f"  GTT  (gttsize)      {gtt:9.0f} MiB")
    print(f"  GPU-addressable     {ceiling:9.0f} MiB  <- the binding ceiling")
    print()
    print("  GTT is backed by the SAME DDR5 as system RAM. The ceiling caps how")
    print("  much the iGPU may pin; it is not memory in addition to MemTotal.")
    print()

    print("=== the CPU-only fallback is physically impossible ===")
    dense = (CKPT["Q4_K_M  (laptop /root/lh/models)"] / MiB) * DENSE_EXPANSION
    print(f"  dense bf16 of this model  {dense:9.0f} MiB  ({dense/1024:5.1f} GiB)")
    print(f"  MemTotal                  {mem_total:9.0f} MiB  ({mem_total/1024:5.1f} GiB)")
    print(f"  overshoot                 {dense/mem_total:9.2f}x")
    print("  Dense bf16 is quant-INDEPENDENT (same weights, unpacked), so")
    print("  starting from Q2_K_XL lands on the same ~67 GiB. There is no CPU")
    print("  K-quant kernel in this tree, so a CPU stage must materialize dense")
    print("  -- and dense does not fit. See HANDOFF section 2.")
    print()

    print("=== resident budget, TP=1 PP=1, iGPU via ROCm ===")
    hdr = f"  {'checkpoint':34s} {'vision':6s} {'weights':>8s} {'room':>8s} {'fp16 ctx':>10s} {'fp8 ctx':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, size in CKPT.items():
        if "rig" in name:
            continue
        w = size / MiB
        for vis_on in (True, False):
            vis = VISION_TOWER_MIB if vis_on else 0.0
            room = ceiling - w - vis - GDN_PER_SEQ_BF16
            usable = max(0.0, room - RUNTIME_RESERVE_MIB)
            ctx16 = usable * 1024 / KV_FP16_KIB_TOK
            ctx8 = usable * 1024 / KV_FP8_KIB_TOK
            print(
                f"  {name:34s} {'yes' if vis_on else 'no':6s} {w:8.0f} {room:8.0f}"
                f" {ctx16/1024:9.0f}k {ctx8/1024:9.0f}k"
            )
    print()
    print(f"  room    = {ceiling:.0f} MiB ceiling - weights - vision - GDN({GDN_PER_SEQ_BF16} MiB, 1 seq, bf16)")
    print(f"  ctx     = (room - {RUNTIME_RESERVE_MIB:.0f} MiB runtime reserve) / KV per token")
    print("  'vision no' assumes #651b is fixed; it is NOT fixed today.")
    print()
    print("  Context is NOT the binding constraint here -- the weights are.")
    print("  Start at 8k and raise; do not chase the ceiling figure on the")
    print("  first boot, because the dequant scratch term is still unpriced.")


if __name__ == "__main__":
    main()
