#!/usr/bin/env python3
"""phi0 lane microbench -- step 1 of ANALYSE_321 §9.2. Seconds of card time.

WHAT IT ANSWERS
---------------
`phi0` is the only number ANALYSE_321 had to band instead of measure: the
speed-up of the 5090's NATIVE W4A4 NVFP4 GEMM over its current `fp8_native`
lane. Everything else in that document is derived from measurements already in
the tree; this one was bounded at `phi0 in [1.3, 2.0]` on Blackwell datasheet
reasoning alone.

    phi0 = nvfp4_native_tflops / fp8_native_tflops        (on the 5090)

STOP RULE (ANALYSE_321 §5.2, §9.2)
----------------------------------
The MLP corner `[136, 0, 0]` binds -- i.e. the whole dense-MLP family wants to
sit on the 5090 -- exactly when

    a_0 / phi0 + n_0  <=  max(n_1, n_2)
    193.2  / phi0 + 64.0  <=  208.9      ->      phi0 >= 1.3333

with `a_0` the 5090's 100 %-of-MLP prefill cost and `n_r` each rank's fixed
non-MLP residual, both from the calibrated #299 model (ANALYSE_321 §1.2).

    phi0 <  1.333  ->  the placement thesis is dead on arithmetic. NVFP4 must
                       then be justified purely on the VRAM/decode axis of §6
                       (-27 % decode step, 1.57x context), which does not
                       depend on phi0 at all and stands regardless.
    phi0 >= 1.333  ->  the interior optimum ceases to exist; the answer to
                       "how far 5090-wards" is "all the way", and the binding
                       term becomes the 3080s' weight-free GDN/attention
                       residual, which no weight format can touch. Ceiling of
                       the entire thesis: 3.6 % of the prefill window against a
                       3.18 % noise floor.

Either way this run is cheap and settles it. It also settles the §3.3 question
of which backend `auto` actually lands on per rank.

WHAT IT MEASURES, PER CARD
--------------------------
  dense bf16      reference, same helper the cached hw_profile uses
  fp8_native      today's 5090 lane          (cached: 568.48 TFLOPS)
  fp8_marlin      today's 3080 lane          (cached: 58.44 / 59.15)
  nvfp4_native    the fork's OWN sm_120a JIT CUTLASS kernel, called directly
                  (NOT through `fp4_gemm`, so the `auto` dispatch cannot
                  silently redirect the measurement to flashinfer -- §3.3)
  nvfp4_marlin    `prepare_nvfp4_layer_for_marlin` + `apply_fp4_marlin_linear`,
                  the real serving helpers, so the number measured is the
                  number served

A lane that cannot run on a card records its REASON, never a substitute
number -- same contract as `uneven_perf._LANE_PROBES`. On sm_86 the native
NVFP4 lane is expected to record "NVFP4 JIT kernels require compute capability
>= 10.0, got 8.6"; that is a fact about the card, not a failure of the run.

SHAPES
------
Two sets, both real:

  probe     M,K,N = 2048, 5120, 17408 -- identical to the cached lane table, so
            the new numbers are directly comparable to 568.48 / 58.44 / 59.15.
  shards    the actual per-rank Qwen3.6-27B MLP shards under the uneven plan
            after the #323a coarsening -- gate_up and down_proj separately,
            for the [7936, 4736, 4736] split and for the [136, 0, 0] corner
            the analysis predicts NVFP4 moves the optimum to.

USAGE
-----
    # every visible card
    python3 scripts/nvfp4/phi0_lane_microbench.py

    # one card, NVML-resolved by name (never assume physical index 0 is the
    # 5090 -- NVML/torch enumeration order can and does diverge)
    python3 scripts/nvfp4/phi0_lane_microbench.py --card 5090

    python3 scripts/nvfp4/phi0_lane_microbench.py --json out.json

No model, no checkpoint download, no server.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import torch

# --- the #299 calibrated prefill model, as published in ANALYSE_321 §1.2 ----
A_MLP_MS = [193.2, 1202.8, 1282.4]  # 100 % of the dense-MLP family, per rank
N_RESIDUAL_MS = [64.0, 208.9, 205.0]  # fixed non-MLP residual, per rank (V1/V3)
PHI0_STOP_RULE = 1.3333

# Probe shape of the cached hw_profile (v3 `gemm_lanes`).
PROBE_M, PROBE_K, PROBE_N = 2048, 5120, 17408
CACHED_TFLOPS = {  # for the side-by-side print
    "5090": {"bf16": 232.97, "fp8_native": 568.48, "fp8_marlin": 216.34},
    "3080": {"bf16": 62.72, "fp8_marlin": 58.44},
}

HIDDEN = 5120
INTERMEDIATE = 17408
# Post-#323a coarsened uneven split (block 128) and the predicted corner.
UNEVEN_MLP_SHARDS = [7936, 4736, 4736]
CORNER_MLP_SHARDS = [17408, 0, 0]

WARMUP = 5
ITERS = 20

NVFP4_GROUP = 16


# ---------------------------------------------------------------------------
# device identity -- NVML, never torch enumeration order
# ---------------------------------------------------------------------------


def nvml_inventory() -> List[Tuple[int, str, int]]:
    """``[(physical_index, name, total_mib)]`` straight from NVML.

    torch's device order and NVML's physical order can diverge (and do on this
    rig), so anything that has to name a card resolves it here and never from
    a hardcoded index.
    """
    import pynvml

    pynvml.nvmlInit()
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
            out.append((i, name, total))
        return out
    finally:
        pynvml.nvmlShutdown()


def torch_devices(card_filter: Optional[str]) -> List[Tuple[int, str]]:
    devices = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        if card_filter and card_filter.lower() not in name.lower():
            continue
        devices.append((index, name))
    return devices


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def time_tflops(device, fn, m: int, k: int, n: int) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    for _ in range(ITERS):
        fn()
    end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    ms = start.elapsed_time(end) / ITERS
    return (2.0 * m * k * n) / (ms / 1e3) / 1e12


# ---------------------------------------------------------------------------
# lanes
# ---------------------------------------------------------------------------


def lane_bf16(device, m, k, n) -> Tuple[Optional[float], str]:
    a = b = None
    try:
        a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
        return time_tflops(device, lambda a=a, b=b: a @ b, m, k, n), ""
    except Exception as ex:  # pragma: no cover - card-window path
        return None, f"dense bf16 did not run: {type(ex).__name__}: {ex}"
    finally:
        del a, b
        torch.cuda.empty_cache()


def lane_fp8(device, m, k, n) -> Dict[str, Tuple[Optional[float], str]]:
    """Reuse the exact probes that produced the cached lane table."""
    from sglang.srt import uneven_perf

    out = {}
    saved = (
        uneven_perf._PROBE_GEMM_M,
        uneven_perf._PROBE_GEMM_K,
        uneven_perf._PROBE_GEMM_N,
    )
    uneven_perf._PROBE_GEMM_M, uneven_perf._PROBE_GEMM_K, uneven_perf._PROBE_GEMM_N = (
        m,
        k,
        n,
    )
    try:
        for lane, probe in uneven_perf._LANE_PROBES.items():
            out[lane] = probe(device)
    finally:
        (
            uneven_perf._PROBE_GEMM_M,
            uneven_perf._PROBE_GEMM_K,
            uneven_perf._PROBE_GEMM_N,
        ) = saved
    return out


def lane_nvfp4_native(device, m, k, n) -> Tuple[Optional[float], str]:
    """The fork's own sm_120a / sm_100a CUTLASS block-scaled NVFP4 GEMM.

    Called DIRECTLY, not through `modelopt_quant.fp4_gemm`: the `auto` dispatch
    of §3.3 would otherwise hand the measurement to flashinfer and the number
    would not be the fork's kernel at all.
    """
    a = b = a_sf = b_sf = alpha = None
    try:
        from sglang.jit_kernel.nvfp4 import (
            cutlass_scaled_fp4_mm,
            scaled_fp4_quant,
            suggest_nvfp4_global_scale,
        )
    except Exception as ex:
        return None, f"NVFP4 JIT kernels unavailable: {type(ex).__name__}: {ex}"
    try:
        if k % 32 or n % 32:
            return None, f"kernel needs k%32==0 and n%32==0, got k={k} n={n}"
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        gs_a = suggest_nvfp4_global_scale(x)
        gs_b = suggest_nvfp4_global_scale(w)
        a, a_sf = scaled_fp4_quant(x, gs_a)
        b, b_sf = scaled_fp4_quant(w, gs_b)
        alpha = (1.0 / (gs_a * gs_b)).to(torch.float32)
        del x, w

        # Default-bound so the `del` in `finally` cannot reach into the
        # closure -- same idiom as uneven_perf's lane probes.
        def fn(a=a, b=b, a_sf=a_sf, b_sf=b_sf, alpha=alpha):
            return cutlass_scaled_fp4_mm(a, b, a_sf, b_sf, alpha, torch.bfloat16)

        return time_tflops(device, fn, m, k, n), ""
    except Exception as ex:
        return None, f"NVFP4 native GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del a, b, a_sf, b_sf, alpha
        torch.cuda.empty_cache()


def lane_nvfp4_marlin(device, m, k, n) -> Tuple[Optional[float], str]:
    """The real serving helpers, so the number measured is the number served.

    `prepare_nvfp4_layer_for_marlin` demands group_size == 16 and
    `n % GPTQ_MARLIN_MIN_THREAD_N == 0`; both hold for every shard the
    post-#323a plan produces.
    """
    layer = x = None
    try:
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_nvfp4_layer_for_marlin,
        )
    except Exception as ex:
        return None, f"NVFP4 Marlin kernels unavailable: {type(ex).__name__}: {ex}"
    try:
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.weight = torch.nn.Parameter(
            torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones((n, k // NVFP4_GROUP), dtype=torch.bfloat16, device=device).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        prepare_nvfp4_layer_for_marlin(layer)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)

        def fn(layer=layer, x=x, n=n, k=k):
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=layer.workspace,
                size_n=n,
                size_k=k,
                bias=None,
            )

        return time_tflops(device, fn, m, k, n), ""
    except Exception as ex:
        return None, f"NVFP4 Marlin GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del layer, x
        torch.cuda.empty_cache()


def resolved_auto_backend() -> str:
    """What `--fp4-gemm-backend auto` resolves to on THIS device (§3.3).

    Part of the deliverable: the analysis flagged that on sm_120 `auto` never
    chose the fork's own kernel. Print the answer next to the numbers so the
    routing claim is checked by the same run that measures the lanes.
    """
    from sglang.srt.layers.quantization import fp4_utils

    saved = fp4_utils.FP4_GEMM_RUNNER_BACKEND
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = None
    try:

        class _Args:
            fp4_gemm_runner_backend = "auto"

        fp4_utils.initialize_fp4_gemm_config(_Args())
        return fp4_utils.get_fp4_gemm_runner_backend().value
    finally:
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = saved


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------


def shape_set(include_shards: bool) -> List[Tuple[str, int, int, int]]:
    shapes = [("probe", PROBE_M, PROBE_K, PROBE_N)]
    if not include_shards:
        return shapes
    for label, shards in (("uneven", UNEVEN_MLP_SHARDS), ("corner", CORNER_MLP_SHARDS)):
        for rank, shard in enumerate(shards):
            if shard == 0:
                continue
            shapes.append((f"{label}-r{rank}-gate_up", PROBE_M, HIDDEN, 2 * shard))
            shapes.append((f"{label}-r{rank}-down", PROBE_M, shard, HIDDEN))
    # De-duplicate identical shapes (r1 and r2 are equal by construction).
    seen, unique = set(), []
    for entry in shapes:
        key = entry[1:]
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def measure_card(index: int, name: str, shapes) -> Dict:
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(index)
    record = {
        "torch_index": index,
        "name": name,
        "capability": f"{capability[0]}.{capability[1]}",
        "auto_fp4_backend": resolved_auto_backend(),
        "shapes": {},
    }
    for label, m, k, n in shapes:
        lanes: Dict[str, object] = {}
        notes: Dict[str, str] = {}

        value, note = lane_bf16(device, m, k, n)
        (lanes if value is not None else notes)["bf16"] = (
            round(value, 2) if value is not None else note
        )
        for lane, (value, note) in lane_fp8(device, m, k, n).items():
            (lanes if value is not None else notes)[lane] = (
                round(value, 2) if value is not None else note
            )
        for lane, fn in (
            ("nvfp4_native", lane_nvfp4_native),
            ("nvfp4_marlin", lane_nvfp4_marlin),
        ):
            value, note = fn(device, m, k, n)
            (lanes if value is not None else notes)[lane] = (
                round(value, 2) if value is not None else note
            )

        entry = {"m": m, "k": k, "n": n, "lanes": lanes, "notes": notes}
        if "nvfp4_native" in lanes and "fp8_native" in lanes:
            entry["phi0"] = round(lanes["nvfp4_native"] / lanes["fp8_native"], 4)
        record["shapes"][label] = entry
    return record


def verdict(phi0: Optional[float]) -> str:
    if phi0 is None:
        return (
            "phi0 NOT MEASURED -- no card ran both nvfp4_native and fp8_native. "
            "The VRAM/decode case of ANALYSE_321 §6 is unaffected and still stands."
        )
    binding = A_MLP_MS[0] / phi0 + N_RESIDUAL_MS[0]
    pacer = max(N_RESIDUAL_MS[1], N_RESIDUAL_MS[2])
    if phi0 < PHI0_STOP_RULE:
        return (
            f"phi0 = {phi0:.3f} < {PHI0_STOP_RULE} -> STOP RULE FIRES. The "
            f"placement thesis is dead on arithmetic: the 5090 at 100 % of the "
            f"MLP family still costs {binding:.1f} ms against the 3080s' "
            f"{pacer:.1f} ms residual, so an interior optimum survives and the "
            f"corner never binds. Re-justify steps 3-5 of ANALYSE_321 §9.2 on "
            f"the VRAM/decode axis of §6 alone -- which does not depend on "
            f"phi0 and is a 9-10x margin over its noise floor."
        )
    return (
        f"phi0 = {phi0:.3f} >= {PHI0_STOP_RULE} -> the MLP corner [136, 0, 0] "
        f"binds ({binding:.1f} ms <= {pacer:.1f} ms). The interior optimum "
        f"ceases to exist and the pacer becomes the 3080s' weight-free "
        f"GDN/attention residual, which no weight format can touch. Ceiling of "
        f"the whole placement thesis: 3.6 % of the prefill window against a "
        f"3.18 % s=8 noise floor -- i.e. still not the reason to do this. The "
        f"reason remains §6."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--card",
        default=None,
        help="substring of the card name (e.g. 5090, 3080); default all",
    )
    parser.add_argument(
        "--shards",
        action="store_true",
        default=True,
        help="also measure the real per-rank Qwen3.6-27B MLP shard shapes",
    )
    parser.add_argument("--probe-only", action="store_true", help="probe shape only")
    parser.add_argument("--json", default=None, help="write the full record here")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device visible.", file=sys.stderr)
        return 2

    print("NVML inventory (physical index -> card, the ONLY authority on ids):")
    try:
        for index, name, total in nvml_inventory():
            print(f"  [{index}] {name}  {total} MiB")
    except Exception as ex:  # pragma: no cover - card-window path
        print(f"  (NVML unavailable: {type(ex).__name__}: {ex})")
    print()

    shapes = shape_set(include_shards=args.shards and not args.probe_only)
    records = [
        measure_card(index, name, shapes) for index, name in torch_devices(args.card)
    ]
    if not records:
        print("No matching card.", file=sys.stderr)
        return 2

    phi0_best: Optional[float] = None
    for record in records:
        print(
            f"=== torch:{record['torch_index']}  {record['name']}  "
            f"sm_{record['capability'].replace('.', '')}  "
            f"auto -> {record['auto_fp4_backend']}"
        )
        cached = None
        for key, values in CACHED_TFLOPS.items():
            if key in record["name"]:
                cached = values
        for label, entry in record["shapes"].items():
            print(f"  {label}  M,K,N = {entry['m']},{entry['k']},{entry['n']}")
            for lane, value in entry["lanes"].items():
                suffix = ""
                if label == "probe" and cached and lane in cached:
                    suffix = f"   (cached {cached[lane]})"
                print(f"      {lane:<14} {value:>9.2f} TFLOPS{suffix}")
            for lane, note in entry["notes"].items():
                print(f"      {lane:<14} --  {note}")
            if "phi0" in entry:
                print(f"      phi0 (nvfp4_native / fp8_native) = {entry['phi0']:.4f}")
                if label == "probe":
                    phi0_best = entry["phi0"]
        print()

    print(verdict(phi0_best))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "records": records,
                    "phi0": phi0_best,
                    "stop_rule": PHI0_STOP_RULE,
                    "verdict": verdict(phi0_best),
                },
                handle,
                indent=2,
            )
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
