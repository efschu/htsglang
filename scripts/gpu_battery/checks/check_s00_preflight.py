#!/usr/bin/env python3
"""S0 check -- is the rig in a state where the rest of the battery means
anything?

Everything here is a STOP, never a FAIL: nothing has been tested yet, so
nothing can have failed. A red corridor, a foreign lock or a missing model is
a reason not to start, not a defect in the code.

The device-order join is checked explicitly. If NVML and CUDA cannot be joined
by PCI address, every later step that names a card is quoting a number that
may mean a different piece of silicon, and the battery must not run.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import CheckStop, load_json, require_envelope, run_check  # noqa: E402

STEP = "s00_preflight"


def check(step_dir: str) -> None:
    payload = load_json(os.path.join(step_dir, "preflight.json"), "preflight.json")
    require_envelope(payload, "gpu_battery_preflight", "preflight.json", 1)

    errors = payload.get("inventory_errors") or []
    if errors:
        raise CheckStop(f"Karten-Inventar meldet {len(errors)} Fehler: {errors[0]}")

    cards = payload.get("cards") or []
    if len(cards) < 2:
        raise CheckStop(
            f"nur {len(cards)} Karte(n) sichtbar, die Batterie braucht >= 2"
        )

    min_free = int(payload.get("min_free_mib") or 400)
    for card in cards:
        for field in ("pci_bus_id", "uuid", "nvml_index", "name"):
            if not card.get(field) and card.get(field) != 0:
                raise CheckStop(f"Karte ohne {field}: {card}")
        if card.get("cuda_index") is None:
            raise CheckStop(
                f"Karte {card.get('pci_bus_id')} hat keinen CUDA-Index -- "
                "der NVML<->CUDA-Join per PCI ist fehlgeschlagen"
            )
        free = card.get("vram_free_mib")
        if not isinstance(free, int):
            raise CheckStop(f"Karte {card.get('pci_bus_id')} ohne freie MiB")
        if free < min_free:
            raise CheckStop(
                f"Karte {card.get('nvml_index')} ({card.get('name')}) nur {free} MiB "
                f"frei, Korridor verlangt >= {min_free}"
            )

    cuda_indices = [c["cuda_index"] for c in cards]
    if len(set(cuda_indices)) != len(cuda_indices):
        raise CheckStop(f"CUDA-Indizes nicht eindeutig: {cuda_indices}")

    held = payload.get("locks_held") or []
    if held:
        raise CheckStop(
            f"{len(held)} Karten-Lock(s) fremd gehalten, erstes: {held[0].get('info')}"
        )

    missing = [p for p, ok in (payload.get("required_files") or {}).items() if not ok]
    if missing:
        raise CheckStop(f"{len(missing)} Pflichtdatei(en) fehlen, erste: {missing[0]}")

    tools = payload.get("tools") or {}
    for tool in ("nvidia-smi", "curl", "py-spy"):
        if not tools.get(tool):
            raise CheckStop(
                f"Werkzeug {tool} fehlt -- ohne es ist kein Schritt sauber fahrbar"
            )

    if not payload.get("driver"):
        raise CheckStop("Treiberversion nicht ermittelbar")
    if not payload.get("torch") or not payload.get("nccl"):
        raise CheckStop("torch/NCCL-Version nicht ermittelbar")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
