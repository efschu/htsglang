#!/usr/bin/env python3
"""s10 check -- is the patched driver really the loaded one, and is the holder up?

Verified, in this order:

  * the host answered at all. An unreachable host measured nothing (STOP),
  * nothing blocked the step: no compute process on the cards, no viewer
    holding the modules without permission (STOP -- the environment, not the
    code),
  * /proc/driver/nvidia/params carries the regkey. Without it the peer-BAR1
    branch is compiled in and switched off,
  * the .ko is the FULL patch: 37 SMALLBAR_P2P strings, not the 1 of the
    minimal one,
  * the LOADED module is that .ko -- srcversion of /sys/module/nvidia against
    modinfo of the file. The regkey line alone proves only that SOMETHING was
    loaded with a parameter, and a stale module with a matching parameter looks
    exactly like a fresh patched one,
  * dmabuf_holder is loaded AND /dev/dmabuf_holder exists. The module without
    the node is the case that fails later, inside the first export,
  * nvidia_uvm is loaded (no UVM, no CUDA),
  * all three cards enumerate. Two cards after a reload means a card did not
    come back from its PCI reset, and every later "which card" statement would
    be wrong.

NOT judged: whether a reload happened. The step is idempotent, and finding the
desired state already in place is the cheaper way to reach it, not a lesser
result.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    load_json,
    require_envelope,
    run_check,
)

STEP = "s10_bar1_driver"
KIND = "bar1_driver_state"
STRINGS_FULL = 37
MIN_CARDS = 3


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "driver_state.json")
    if not os.path.exists(path):
        raise CheckStop(
            f"driver_state.json fehlt ({path}) -- der Schritt ist nicht gelaufen"
        )
    payload = load_json(path, "driver_state.json")
    require_envelope(payload, KIND, "driver_state.json", 1)

    if not payload.get("reachable"):
        raise CheckStop(
            f"Host {payload.get('host') or '?'} nicht erreichbar -- ueber ssh wurde "
            "nichts erhoben, also ist auch nichts geprueft"
        )

    blocked = payload.get("blocked")
    if blocked:
        first = " ".join(str(blocked).split())[:180]
        raise CheckStop(
            f"Eingriff blockiert, nichts angefasst: {first} "
            "(Betrachter beenden ist eine Nutzer-Entscheidung: BAR1_VIEWER_KILL_OK=1)"
        )

    apps = payload.get("compute_apps") or []
    if apps:
        raise CheckStop(f"{len(apps)} Rechenprozess(e) auf den Karten: {apps[0][:120]}")

    if not payload.get("regkey_present"):
        raise CheckFail(
            f"Regkey {payload.get('regkey_expected')!r} steht nicht in "
            f"/proc/driver/nvidia/params (Zeile: {payload.get('regkey_line')!r}) -- "
            "der Peer-BAR1-Zweig ist aus"
        )

    level = payload.get("patch_level")
    if level != "voll":
        raise CheckFail(
            f"Patch-Stand {level!r} (strings SMALLBAR_P2P = "
            f"{payload.get('strings_smallbar')!r}, erwartet {STRINGS_FULL}) -- "
            "der minimale Patch traegt den Direktpfad nicht"
        )

    if not payload.get("module_identity_matches"):
        raise CheckFail(
            f"srcversion des geladenen Moduls {payload.get('srcversion_loaded')!r} "
            f"passt nicht zur .ko {payload.get('srcversion_file')!r} -- geladen ist "
            "ein anderes Modul als das gepatchte, die Regkey-Zeile allein belegt "
            "nur einen Parameter"
        )

    if not payload.get("dmabuf_holder_loaded"):
        raise CheckFail("dmabuf_holder ist nicht geladen -- ohne Halter kein Export")
    if not payload.get("dmabuf_dev_present"):
        raise CheckFail(
            "/dev/dmabuf_holder fehlt, obwohl das Modul geladen ist -- der Export "
            "scheitert erst spaeter, mitten im Aufbau"
        )

    modules = payload.get("modules") or {}
    # The field carries lsmod's use count: "" means absent, "0" means loaded
    # with nobody using it yet. Testing truthiness would call a fresh insmod a
    # missing module.
    if modules.get("nvidia_uvm", "") == "":
        raise CheckFail("nvidia_uvm ist nicht geladen -- ohne UVM kein CUDA")

    cards = payload.get("cards") or []
    if len(cards) < MIN_CARDS:
        raise CheckFail(
            f"nur {len(cards)} von {MIN_CARDS} Karten enumerieren -- eine Karte ist "
            "nach dem PCI-Reset nicht zurueckgekommen"
        )
    for card in cards:
        if not card.get("uuid") or not card.get("pci_bus_id"):
            raise CheckFail(
                f"Karte ohne UUID oder PCI-Adresse: {card!r} -- jede spaetere "
                "Kartenaussage waere unbelegt"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
