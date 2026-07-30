#!/usr/bin/env python3
"""s10 -- turn the raw host probe files into one machine-readable state.

Two jobs, one file, on purpose:

  --gate-only   decide whether the desired state is ALREADY there. Exit 0 =
                yes, nothing has to be unloaded; exit 3 = a reload is needed.
                The shell asks this before it touches anything.
  (default)     write driver_state.json from whatever the probe collected.

The identity question is the interesting one. "The regkey is in
/proc/driver/nvidia/params" says a module was loaded with the parameter -- it
does NOT say the LOADED module is the patched build. So srcversion of
/sys/module/nvidia is compared against modinfo of the .ko that would be
inserted. That is the same lesson as "the transport name in the log lies",
applied to a kernel module: a name is not a proof of identity.

CPU-only, no ssh, no card. The shell does the talking, this does the reading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

KIND = "bar1_driver_state"
SCHEMA_VERSION = 1

# strings nvidia.ko | grep -c SMALLBAR_P2P: 37 = full patch, 1 = minimal.
STRINGS_FULL = 37
STRINGS_MINIMAL = 1


def read_kv(path: str) -> dict:
    out: dict = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def read_lines(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_first(path: str, default: str = "") -> str:
    lines = read_lines(path)
    return lines[0].strip() if lines else default


def parse_cards(path: str) -> list:
    cards = []
    for line in read_lines(path):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        entry = {
            "nvml_index": parts[0],
            "name": parts[1],
            "uuid": parts[2],
            "pci_bus_id": parts[3].lower(),
        }
        if len(parts) >= 6:
            try:
                entry["vram_total_mib"] = int(parts[4])
                entry["vram_used_mib"] = int(parts[5])
            except ValueError:
                pass
        cards.append(entry)
    return cards


def patch_level(count: object) -> str:
    # The returned tokens are a data contract: check_s10_bar1_driver.py and
    # test_gpu_battery_checks_bar1.py compare against "voll"/"minimal"
    # verbatim, so the enumeration stays German.
    try:
        n = int(str(count).strip())
    except (TypeError, ValueError):
        return "unbekannt"
    if n >= STRINGS_FULL:
        return "voll"
    if n >= STRINGS_MINIMAL:
        return "minimal"
    return "unpatched"


def compose(step_dir: str, phase: str = "after") -> dict:
    host = os.path.join(step_dir, "host")
    state = read_kv(os.path.join(host, f"state_{phase}.txt"))
    if not state and phase == "after":
        state = read_kv(os.path.join(host, "state_before.txt"))
    regkey_line = state.get("regkey_line", "")
    strings_count = state.get("strings_smallbar")
    loaded = state.get("srcversion_loaded", "")
    from_file = state.get("srcversion_file", "")

    payload = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "host": os.environ.get("BAR1_HOST", ""),
        "reachable": read_first(os.path.join(host, "reach.txt")) == "ok",
        "reload_performed": read_first(os.path.join(host, "reload_performed.txt"), "0")
        == "1",
        "reload_rc": read_first(os.path.join(host, "reload_rc.txt"), ""),
        "blocked": "\n".join(read_lines(os.path.join(host, "blocked.txt"))) or None,
        "viewers_blocking": read_lines(os.path.join(host, "holders_before.txt")),
        "viewers_killed": read_lines(os.path.join(host, "viewers_killed.txt")),
        "compute_apps": read_lines(os.path.join(host, f"compute_apps_{phase}.csv")),
        "kernel": state.get("kernel", ""),
        "driver_version": state.get("driver_version", ""),
        "regkey_expected": os.environ.get("BAR1_REGKEY", "RMSmallBarP2PPeerBar1=1"),
        "regkey_line": regkey_line,
        "strings_smallbar": strings_count,
        "patch_level": patch_level(strings_count),
        "srcversion_loaded": loaded,
        "srcversion_file": from_file,
        "module_identity_matches": bool(loaded) and loaded == from_file,
        "modules": {
            "nvidia": state.get("mod_nvidia", ""),
            "nvidia_uvm": state.get("mod_nvidia_uvm", ""),
            "nvidia_modeset": state.get("mod_nvidia_modeset", ""),
            "dmabuf_holder": state.get("mod_dmabuf_holder", ""),
        },
        # "ja" is what the host probe writes and what the fixtures feed.
        "dmabuf_dev_present": state.get("dmabuf_dev") == "ja",
        "dmabuf_major": state.get("dmabuf_major", ""),
        "cards": parse_cards(os.path.join(host, f"cards_{phase}.csv")),
        "pci_nvidia": read_lines(os.path.join(host, "pci.txt")),
        "container_cards": read_lines(os.path.join(host, "container_cards.txt")),
        "reload_log_tail": read_lines(os.path.join(host, "reload.log"))[-20:],
    }
    payload["regkey_present"] = payload["regkey_expected"] in regkey_line
    # The module fields hold the USE COUNT from lsmod, and a loaded module with
    # nobody using it reports "0". Absent means the awk printed nothing, so the
    # empty string -- not the zero -- is what "not loaded" looks like.
    payload["dmabuf_holder_loaded"] = payload["modules"]["dmabuf_holder"] != ""
    return payload


def desired_state_reached(payload: dict) -> tuple:
    """What s11 and s12 actually need. Returns (ok, list of missing pieces)."""
    missing = []
    if not payload.get("reachable"):
        missing.append("host not reachable")
    if not payload.get("regkey_present"):
        missing.append("Regkey not in /proc/driver/nvidia/params")
    if payload.get("patch_level") != "voll":
        # "Patch-Stand" is asserted on by test_gpu_battery_checks_bar1.py.
        missing.append(f"Patch-Stand {payload.get('patch_level')!r}, expected 'voll'")
    if not payload.get("module_identity_matches"):
        missing.append("srcversion of the loaded module does not match the .ko")
    if not payload.get("dmabuf_holder_loaded"):
        missing.append("dmabuf_holder not loaded")
    if not payload.get("dmabuf_dev_present"):
        missing.append("/dev/dmabuf_holder is missing")
    # "" = not loaded, "0" = loaded and unused. See the note in compose().
    if payload["modules"].get("nvidia_uvm", "") == "":
        missing.append("nvidia_uvm not loaded")
    if len(payload.get("cards") or []) < 3:
        missing.append(f"only {len(payload.get('cards') or [])} cards enumerated")
    return (not missing, missing)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="only check whether the desired state is already there (0 = yes, 3 = no)",
    )
    args = ap.parse_args()

    if args.gate_only:
        payload = compose(args.step_dir, phase="before")
        ok, missing = desired_state_reached(payload)
        for item in missing:
            print(f"  missing: {item}")
        return 0 if ok else 3

    payload = compose(args.step_dir, phase="after")
    ok, missing = desired_state_reached(payload)
    payload["desired_state"] = ok
    payload["missing"] = missing
    out = os.path.join(args.step_dir, "driver_state.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"driver_state.json written "
        f"({'desired state' if ok else 'incomplete'})"
    )
    for item in missing:
        print(f"  missing: {item}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
