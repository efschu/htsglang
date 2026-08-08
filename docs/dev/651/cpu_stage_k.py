"""#651 -- how many prefill layers can a CPU PP stage actually hold?

The user's design (2026-08-08): prefill runs PIPELINE-PARALLEL across the APU
with two workers -- the CPU part and the iGPU part are the two PP stages --
and decode runs on the iGPU alone, with a reshard between the phases.

So the CPU stage holds a LAYER SHARE k, not the whole model. The earlier
"CPU is impossible" verdict priced k=1 and is only valid there. This script
prices k properly, for both implementation routes.

    python3 docs/dev/651/cpu_stage_k.py

ROUTE B (PRIMARY) -- CPU-native quantized compute. llama.cpp computes directly
  on quantized blocks, so Q4 stays ~22 GB; there is NO 3.17x. The CPU stage
  costs ~k * quant-size and k is bounded by the BALANCE POINT, not by RAM.
  Needs only the quant types this checkpoint uses on the layers the CPU owns --
  far less than the full GGUF surface. Also sidesteps the gfx1103 Q6_K bug,
  since CPU dequant is the already-validated-correct reference.
ROUTE A (STOPGAP COMPARATOR) -- dense-materialize the CPU stage's layers (bf16)
  because THIS FORK has no CPU K-quant kernel. Costs the measured 3.17x on the
  CPU share only. Caps k hard and forces the debug checkpoint. Not the
  destination; useful only to get co-run numbers sooner.

CHECKPOINT POLICY (user, 2026-08-08): Q4_K_M is the TARGET. Q2 is "zu dumm" and
is a DEBUG vehicle only -- no Q2 number may be quoted as a result. Fallback if a
config stops fitting is a Q3 variant (needs a download), never Q2.

NOTE: solo stage speeds do NOT predict co-run -- CPU and iGPU throttle each
other through shared TDP, DDR5 bandwidth and memory-controller contention.
R here is a PARAMETER to sweep, not a measurement.

The split is set by the equal-co-run-ms condition, NOT by this table and NOT by
throughput: measure ms per ROUND per stage under co-run, and choose the layer
split so both stages' co-run ms per pipeline round come out EQUAL. Clocks and
power are deliberately NOT recorded -- the throttling interaction is already
fully captured in the ms times. See HANDOFF section 6.0.2.

Everything lives in ONE physical DDR5 pool, so both stages plus the OS are
charged against MemTotal. The iGPU share is additionally capped by the GTT
window.
"""

MiB = 1048576

# --- measured on the laptop --------------------------------------------------
MEM_TOTAL = 30935848 / 1024        # 30,211 MiB
GTT_CEILING = (1073741824 + 25769803776) / MiB   # 25,600 MiB
OS_RESIDENT = 1674.0               # measured `free -m` used, desktop quiesced
N_LAYERS = 40                      # backbone depth (NOT block_count 41)

# --- measured from the checkpoints (dequant_cost.py / file sizes) -----------
DENSE_PER_LAYER = 1604.2           # bf16, quant-independent
Q4KXL_PACKED_PER_LAYER = 506.2     # measured on the rig's Q4_K_XL
Q4KXL_TOTAL = 22853663008 / MiB

# Per-layer packed size for the laptop's files, scaled by total file size from
# the one file whose per-layer figure was measured directly. Approximate --
# UD quants are mixed-precision -- so treat k to two significant figures.
_LAYER_FRACTION = (N_LAYERS * Q4KXL_PACKED_PER_LAYER) / Q4KXL_TOTAL

CKPT = {
    "Q4_K_M (TARGET)": 22663387424 / MiB,
    "Q2_K_XL (debug only)": 12574128416 / MiB,
}

# Runtime overhead: KV at ctx 8192 (20.00 KiB/token fp16) + activations +
# dequant scratch + allocator slack. Same reserve apu_budget.py uses.
KV_CTX8K_FP16 = 8192 * 20.0 / 1024
RUNTIME_RESERVE = 1500.0


def parts(total: float):
    layers = total * _LAYER_FRACTION
    return layers / N_LAYERS, total - layers      # per-layer packed, non-layer


def max_k_route_a(total: float):
    """Largest CPU layer share whose dense weights still fit in MemTotal."""
    packed_per_layer, non_layer = parts(total)
    overhead = OS_RESIDENT + KV_CTX8K_FP16 + RUNTIME_RESERVE + non_layer
    # k*N*DENSE + (1-k)*N*packed + overhead <= MEM_TOTAL
    budget = MEM_TOTAL - overhead - N_LAYERS * packed_per_layer
    slope = N_LAYERS * (DENSE_PER_LAYER - packed_per_layer)
    return max(0.0, budget / slope)


def main() -> None:
    print("=== the split the CPU stage must fit into ===")
    print(f"  MemTotal {MEM_TOTAL:,.0f} MiB | GTT ceiling {GTT_CEILING:,.0f} MiB"
          f" | OS {OS_RESIDENT:,.0f} MiB | backbone {N_LAYERS} layers")
    print(f"  dense/layer {DENSE_PER_LAYER:,.1f} MiB (bf16, quant-independent)")
    print()

    print("=== ROUTE A: dense-materialize the CPU stage's layers ===")
    print(f"  {'ckpt':9s} {'packed/layer':>13s} {'max k':>7s} {'max CPU layers':>15s}"
          f" {'iGPU layers':>12s} {'iGPU MiB':>10s}")
    results = {}
    for name, total in CKPT.items():
        ppl, non_layer = parts(total)
        k = max_k_route_a(total)
        results[name] = k
        l_cpu = k * N_LAYERS
        igpu = (N_LAYERS - l_cpu) * ppl + non_layer
        print(f"  {name:9s} {ppl:13,.1f} {k:7.3f} {l_cpu:15.1f} "
              f"{N_LAYERS - l_cpu:12.1f} {igpu:10,.0f}")
    print()
    print("  iGPU MiB is comfortably under the 25,600 GTT ceiling in both rows,")
    print("  so RAM -- not the GTT window -- is what binds route A.")
    print()

    print("=== is that share ENOUGH to be worth it? ===")
    print("  Balanced 2-stage split: L_cpu = N/(R+1), R = prefill_igpu/prefill_cpu.")
    print("  Max prefill speedup = 1 + 1/R. R MUST BE MEASURED (see plan).")
    print()
    print(f"  {'R':>5s} {'balanced L_cpu':>15s} {'max speedup':>12s}"
          f" {'fits Q2(dbg)?':>14s} {'fits Q4(TGT)?':>13s}")
    for R in (2, 3, 4, 5, 8, 10, 20):
        l_bal = N_LAYERS / (R + 1)
        speedup = 1.0 + 1.0 / R
        ok2 = "YES" if l_bal <= results["Q2_K_XL (debug only)"] * N_LAYERS else "no"
        ok4 = "YES" if l_bal <= results["Q4_K_M (TARGET)"] * N_LAYERS else "no"
        print(f"  {R:5d} {l_bal:15.1f} {speedup:11.0%} {ok2:>14s} {ok4:>13s}")
    print()
    print("  Read: route A is viable on Q2_K_XL for any R >= ~3, and on Q4_K_M")
    print("  only for R >= ~8 (where the payoff has shrunk to ~+12%).")
    print()

    print("=== ROUTE B: CPU-native K-quant (no dense penalty) ===")
    for name, total in CKPT.items():
        ppl, non_layer = parts(total)
        overhead = OS_RESIDENT + KV_CTX8K_FP16 + RUNTIME_RESERVE + non_layer
        headroom = MEM_TOTAL - overhead - N_LAYERS * ppl
        print(f"  {name:9s} whole model packed in RAM, {headroom:>7,.0f} MiB spare"
              f" -> k bounded by the BALANCE POINT, not by memory")
    print()
    print("  Route B removes the memory constraint entirely: the CPU stage costs")
    print("  its packed share, which the machine already holds. k is then chosen")
    print("  purely to balance the pipeline, and Q4_K_M stays on the table.")


if __name__ == "__main__":
    main()
