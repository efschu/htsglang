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

# Two arbitration mechanisms exist on this rig and they are not the same: the
# battery takes /tmp/gpu-card-N.lock, while /spinning/gpu-arb/holder is the
# operator-level file the r7c recipes write. Only the operator's own window
# authorises the battery to run; anyone else's holder is a foreign session
# whose boot the battery would run straight into.
ARB_OWN_SESSION = "operator"


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

    _check_arbitration(payload)

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


def _check_arbitration(payload: dict) -> None:
    """The cross-session holder in /spinning/gpu-arb/holder.

    The card locks only cover sessions that take them; the holder file is what
    an r7c boot in another session writes. It was recorded by the preflight and
    then read by nobody, so a foreign session's window did not stop the
    battery -- and both would be on the cards at once.

    Line shape: ``session=<name>  cards=0,1,2  purpose=...  since=...``.
    """
    holder = payload.get("arb_holder")
    if not holder:
        return
    fields = dict(tok.split("=", 1) for tok in str(holder).split() if "=" in tok)
    session = fields.get("session")
    if session == ARB_OWN_SESSION:
        return
    if session is None:
        raise CheckStop(
            f"/spinning/gpu-arb/holder ohne session=-Feld: {str(holder)[:120]} -- "
            "unklarer Halter wird nicht ueberfahren"
        )
    raise CheckStop(
        f"/spinning/gpu-arb/holder haelt session={session} "
        f"(cards={fields.get('cards')}, purpose={fields.get('purpose')}) -- "
        "fremdes Fenster, die Batterie startet nicht hinein"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
