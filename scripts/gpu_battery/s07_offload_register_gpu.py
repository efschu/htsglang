#!/usr/bin/env python3
"""Offload register on real silicon: CudaDeviceOps, real item sizes, retrieval
latency per class.

The #286 GPU restlist, items 1 and 4. Everything else about the register was
built and tested CPU-hermetically behind an injectable device layer; this is
the one thing a fake cannot answer -- whether the three real movement routes
move real bytes, and what getting them back costs.

THREE ROUTES, because the register has three and a validation of one is not a
validation of the register:

  tensor   pinned host pool + async H2D behind compute (the
           MoEExpertOffloadCache._fetch pattern, wait_stream in both
           directions). Carries lane_workspaces, kv_shadow, experts.
  tag      #93 tag pools / VMM through the memory saver. The route for
           va_stable items -- graph rungs and GDN state sets are addressed by
           kernels and graphs, so their virtual addresses have to survive the
           park.
  suspend  #89 suspend, for cold_lane.

WHY THE LATENCY NUMBERS MATTER MORE THAN THE PASS: every auto/ram default in
the register carries a measurement obligation ("Messpflicht vor jedem
auto/ram-Default"). Until a class has a measured retrieval latency, its
latency term for the #279 dispatcher is a guess, and a guessed latency term
silently decides placements. This step produces those numbers; it does not
decide the defaults.

The park is timed too, but the retrieval is the number that matters: a park
happens when there is slack, a wave-in happens when something is waiting.

Usage:
    python s07_offload_register_gpu.py --out <dir>/offload_register_gpu.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

MIB = 1024 * 1024

# One size for every route, chosen to be big enough that the copy dominates the
# call overhead and small enough that three of them fit anywhere on this rig.
ITEM_BYTES = 256 * MIB
CYCLES = 5


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=CYCLES)
    ap.add_argument("--item-mib", type=int, default=ITEM_BYTES // MIB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(
            "Plan: eine Karte (die groesste, zur Laufzeit per PCI aufgeloest), "
            "CudaDeviceOps + RealMovementBackend; je Route (tensor|tag|suspend) "
            f"ein Posten von {args.item_mib} MiB, {args.cycles} park/wave_in-Zyklen, "
            "p50/p99 je Klasse; echte Groessen ueber resolve_size_bytes."
        )
        return 0

    repo_python = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
    )
    sys.path.insert(0, repo_python)

    import torch

    from sglang.srt.model_executor.offload_movement import (
        CudaDeviceOps,
        RealMovementBackend,
        SuspendPayload,
        TagPayload,
        TensorPayload,
    )
    from sglang.srt.model_executor.offload_register import (
        OffloadRegister,
        resolve_class_policies,
    )
    from sglang.srt.model_executor.offload_sizes import resolve_size_bytes

    if not torch.cuda.is_available():
        print("torch.cuda nicht verfuegbar", file=sys.stderr)
        return 2

    # The card is resolved, never assumed. The biggest one carries the test
    # because it is the only one with room for three parked items at once.
    big = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.get_device_properties(i).total_memory,
    )
    torch.cuda.set_device(big)
    props = torch.cuda.get_device_properties(big)
    device = torch.device(f"cuda:{big}")

    payload = {
        "kind": "offload_register_gpu",
        "schema_version": 1,
        "timestamp": datetime.datetime.now().isoformat(),
        # Filled from the object that is actually built further down. A literal
        # here would make the check that guards against a FakeDeviceOps fallback
        # assert its own constant.
        "device_ops": None,
        "device": {
            "cuda_index": big,
            "name": props.name,
            "pci_bus_id": "%08x:%02x:%02x.0"
            % (
                getattr(props, "pci_domain_id", 0),
                getattr(props, "pci_bus_id", 0),
                getattr(props, "pci_device_id", 0),
            ),
            "total_mib": props.total_memory // MIB,
        },
        "item_bytes": args.item_mib * MIB,
        "cycles": args.cycles,
        "routes": {},
        "rows": [],
        "stats": {},
    }

    # The memory saver is what makes the tag and suspend routes real. Without
    # it those routes are not "fine", they are untested -- the check turns that
    # into a STOP rather than a green run with two thirds of the register
    # unexercised.
    saver = None
    try:
        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

        saver = TorchMemorySaverAdapter.create(enable=True)
        payload["memory_saver"] = "real"
    except Exception as exc:
        payload["memory_saver"] = f"unavailable: {exc!r}"

    device_ops = CudaDeviceOps(memory_saver_adapter=saver)
    payload["device_ops"] = type(device_ops).__name__
    policies = resolve_class_policies("auto")
    backend = RealMovementBackend(
        device_ops=device_ops,
        target_order=("host_ram",),
        class_policies=policies,
    )
    register = OffloadRegister(policies=policies, backend=backend)

    nelem = (args.item_mib * MIB) // 2  # float16

    def cycle(item_id, row):
        """park -> settle -> wave_in, timed, with the state sequence recorded.

        The states are recorded rather than assumed: a park that silently
        no-ops and a park that moves 256 MiB both return None, and only the
        state machine tells them apart.
        """
        park_ms, wave_ms, states = [], [], []
        for _ in range(args.cycles):
            states.append(backend.state_of(item_id))
            t0 = time.perf_counter()
            register.park(item_id)
            backend.settle(item_id)
            park_ms.append((time.perf_counter() - t0) * 1e3)
            states.append(backend.state_of(item_id))

            t0 = time.perf_counter()
            register.wave_in(item_id)
            backend.settle(item_id)
            wave_ms.append((time.perf_counter() - t0) * 1e3)
            states.append(backend.state_of(item_id))
        row["park_ms_p50"] = round(percentile(park_ms, 0.50), 3)
        row["park_ms_p99"] = round(percentile(park_ms, 0.99), 3)
        row["wave_in_ms_p50"] = round(percentile(wave_ms, 0.50), 3)
        row["wave_in_ms_p99"] = round(percentile(wave_ms, 0.99), 3)
        row["iters"] = len(wave_ms)
        p50 = row["wave_in_ms_p50"]
        row["wave_in_gb_per_s"] = (
            round(row["size_bytes"] / (p50 / 1e3) / 1e9, 3) if p50 > 0 else None
        )
        row["state_sequence"] = states[:9]
        row["status"] = "ok"

    # ---------------------------------------------------------------- tensor
    tensor_classes = (
        ("lane_workspaces", "phase"),
        ("kv_shadow", "turn"),
        ("experts", "wave"),
    )
    tensor_status = "ok"
    for offload_class, tier in tensor_classes:
        item_id = f"battery:{offload_class}"
        row = {"offload_class": offload_class, "route": "tensor", "item_id": item_id}
        try:
            tensor = torch.empty(nelem, dtype=torch.float16, device=device)
            resolved = resolve_size_bytes(tensor)
            row["size_bytes"] = resolved
            row["size_source_matches"] = (
                resolved == tensor.numel() * tensor.element_size()
            )
            register.register(
                item_id=item_id,
                offload_class=offload_class,
                size_bytes=resolved,
                restore_cost_ms=0.0,
                hot=lambda: False,
                time_constant_tier=tier,
            )
            backend.bind(item_id, TensorPayload(tensors=(tensor,)), source_device=big)
            cycle(item_id, row)
            del tensor
            torch.cuda.empty_cache()
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()[-2000:]
            tensor_status = "error"
        payload["rows"].append(row)
    payload["routes"]["tensor"] = tensor_status

    # ------------------------------------------------------------------- tag
    if saver is None:
        payload["routes"]["tag"] = "unavailable"
        payload["routes"]["suspend"] = "unavailable"
    else:
        tag_status = "ok"
        for offload_class in ("graph_rungs", "gdn_state_sets"):
            item_id = f"battery:{offload_class}"
            tag = f"battery_{offload_class}"
            row = {
                "offload_class": offload_class,
                "route": "tag",
                "item_id": item_id,
                "tag": tag,
            }
            try:
                with saver.region(tag=tag, enable_cpu_backup=True):
                    tagged = torch.empty(nelem, dtype=torch.float16, device=device)
                resolved = resolve_size_bytes(tagged)
                row["size_bytes"] = resolved
                row["size_source_matches"] = (
                    resolved == tagged.numel() * tagged.element_size()
                )
                register.register(
                    item_id=item_id,
                    offload_class=offload_class,
                    size_bytes=resolved,
                    restore_cost_ms=0.0,
                    hot=lambda: False,
                    va_stable_required=True,
                    time_constant_tier="turn",
                )
                backend.bind(
                    item_id, TagPayload(tag=tag, cpu_backup=True), source_device=big
                )
                cycle(item_id, row)
                del tagged
                torch.cuda.empty_cache()
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()[-2000:]
                tag_status = "error"
            payload["rows"].append(row)
        payload["routes"]["tag"] = tag_status

        # --------------------------------------------------------- suspend
        item_id = "battery:cold_lane"
        tag = "battery_cold_lane"
        row = {
            "offload_class": "cold_lane",
            "route": "suspend",
            "item_id": item_id,
            "tag": tag,
        }
        try:
            with saver.region(tag=tag, enable_cpu_backup=True):
                cold = torch.empty(nelem, dtype=torch.float16, device=device)
            resolved = resolve_size_bytes(cold)
            row["size_bytes"] = resolved
            row["size_source_matches"] = resolved == cold.numel() * cold.element_size()
            register.register(
                item_id=item_id,
                offload_class="cold_lane",
                size_bytes=resolved,
                restore_cost_ms=0.0,
                hot=lambda: False,
                time_constant_tier="turn",
            )
            backend.bind(item_id, SuspendPayload(tags=(tag,)), source_device=big)
            cycle(item_id, row)
            del cold
            torch.cuda.empty_cache()
            payload["routes"]["suspend"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()[-2000:]
            payload["routes"]["suspend"] = "error"
        payload["rows"].append(row)

    stats = getattr(backend, "stats", None)
    if stats is not None:
        payload["stats"] = {
            "parks": getattr(stats, "parks", None),
            "wave_ins": getattr(stats, "wave_ins", None),
            "park_failures": getattr(stats, "park_failures", None),
            "wave_in_failures": getattr(stats, "wave_in_failures", None),
            "peer_degradations": getattr(stats, "peer_degradations", None),
            "chunked_transfers": getattr(stats, "chunked_transfers", None),
            "bytes_by_target": dict(getattr(stats, "bytes_by_target", {}) or {}),
        }

    # The latency term is what the #279 dispatcher will actually read. Recording
    # it here closes the loop: the number the cost model sees comes from this
    # measurement and not from a constant.
    payload["latency_term_ms"] = {
        row["offload_class"]: register.latency_term_ms(row["offload_class"])
        for row in payload["rows"]
        if row.get("status") == "ok"
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(
        f"{'Klasse':<18} {'Route':<8} {'MiB':>6} {'park p50':>9} {'wave p50':>9} {'GB/s':>7}"
    )
    for row in payload["rows"]:
        print(
            f"{row['offload_class']:<18} {row['route']:<8} "
            f"{(row.get('size_bytes') or 0) // MIB:>6} "
            f"{str(row.get('park_ms_p50')):>9} {str(row.get('wave_in_ms_p50')):>9} "
            f"{str(row.get('wave_in_gb_per_s')):>7}  {row.get('status')}"
        )
    print(f"geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
