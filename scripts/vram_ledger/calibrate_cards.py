#!/usr/bin/env python3
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GPU gate (a): measure the VRAM ledger's hardware residuals on every card.

    python scripts/vram_ledger/calibrate_cards.py            # measure + cache
    python scripts/vram_ledger/calibrate_cards.py --show     # read the cache
    python scripts/vram_ledger/calibrate_cards.py --dry-run  # no GPU touched

WHY EACH CARD GETS ITS OWN PROCESS. Two reasons, and neither is style.

*Isolation.* A residual is a PER-PROCESS quantity: the CUDA primary context,
the allocator's segment granularity and the lazily-created kernel workspaces
are all things a process pays once. Measuring three cards from one process
would measure the first card's context plus the marginal cost of the next two,
which is not what a rank pays and not what the ledger charges.

*The device-order trap.* The child is pinned with
``CUDA_VISIBLE_DEVICES=GPU-<uuid>``, the UUID form, never an index. On this rig
CUDA's FASTEST_FIRST enumeration and NVML's PCI-bus order disagree -- the RTX
5090 is CUDA ordinal 0 and NVML index 1 -- and #349 sweep-3 arm L is that
disagreement in the field: a budget vector accepted against the wrong card,
then OOM. A UUID means the same card under both orders and under a reboot.
Inside the child exactly one device is visible, so ``cuda:0`` is unambiguous.

TIME-BOX. Each child gets ``--timeout`` seconds. A CUDA call can block
uninterruptibly (a wedged context, an ECC event, a card another process is
resetting), and a calibration that hangs is worse than one that fails: it
holds a GPU window open with nothing to show. On expiry the child is killed
and the card is reported as a NAMED failure. The run then refuses to write a
partial profile -- a cache file missing a card would silently make that card's
ledger term unbounded later, which looks like a bug in the ledger rather than
like the incomplete measurement it is.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
)
if REPO_PYTHON not in sys.path:
    sys.path.insert(0, REPO_PYTHON)

DEFAULT_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Card resolution: UUID and PCI BDF, never a bare index
# ---------------------------------------------------------------------------


def resolve_cards(allow_cuda_init: bool = True) -> Tuple[List[dict], Optional[str]]:
    """``(cards, driver)`` where each card carries uuid, name and PCI BDF.

    THE CARD LIST AND THE DRIVER COME FROM ``card_probe._inventory()``, which is
    the same source :func:`calibration.live_fingerprint` reads at boot. That is
    not an implementation detail: the fingerprint is a hash of the card UUID set
    plus the driver string, so a script that assembled either from a different
    source could write a perfectly good calibration under a key the boot never
    looks up. The cache would then miss forever and the ledger would refuse
    every boot while a valid-looking file sat next to it.

    The PCI BDF is added on top from the #331/#392 identity map, for display and
    for naming a card in an error. A card the map cannot place keeps a "?" BDF
    rather than failing the run: the BDF is how a human recognises the card, not
    how this script addresses it.
    """
    from sglang.srt.rigmon.card_probe import _inventory

    gpus, driver = _inventory()
    if not gpus:
        raise RuntimeError(
            "NVML reports no cards; nothing to calibrate. (This is a named "
            "failure, not an empty success: a profile without cards would "
            "make every ledger term unbounded at the next boot.)"
        )

    bdf_by_uuid: Dict[str, str] = {}
    nvml_by_uuid: Dict[str, int] = {}
    try:
        from sglang.srt.registry import nvml as registry_nvml

        imap = registry_nvml.identity_map(allow_cuda_init=allow_cuda_init)
        for card in imap.cards:
            bdf_by_uuid[card.uuid] = card.pci_bus_id
            nvml_by_uuid[card.uuid] = card.nvml_index
    except Exception as e:
        print(f"  (identity map unavailable, BDFs will show as '?': {e})")

    cards = []
    for gpu in gpus:
        uuid = str(gpu["uuid"])
        cards.append(
            {
                "uuid": uuid,
                "name": str(gpu.get("name", "")),
                "pci_bus_id": bdf_by_uuid.get(uuid, "?"),
                "nvml_index": nvml_by_uuid.get(uuid, "?"),
                "cuda_ordinal": gpu.get("cuda_index", "?"),
            }
        )
    return cards, driver


# ---------------------------------------------------------------------------
# The child: one card, one process, JSON on stdout
# ---------------------------------------------------------------------------


def measure_child(uuid: str) -> int:
    """Measure the visible card and print one JSON object. Runs in the child."""
    from sglang.srt.mem_ledger.calibration import _measure_one_card

    try:
        import torch

        if torch.cuda.device_count() != 1:
            print(
                json.dumps(
                    {
                        "uuid": uuid,
                        "error": (
                            f"child sees {torch.cuda.device_count()} devices, "
                            "expected exactly 1; CUDA_VISIBLE_DEVICES pinning "
                            "did not take, refusing to attribute a "
                            "measurement to a card it may not belong to"
                        ),
                    }
                )
            )
            return 2
        name = torch.cuda.get_device_name(0)
        ctx, gran, ws, note = _measure_one_card(0, uuid)
    except Exception as e:
        print(json.dumps({"uuid": uuid, "error": f"{type(e).__name__}: {e}"}))
        return 3
    print(
        json.dumps(
            {
                "uuid": uuid,
                "name": name,
                "cuda_context_bytes": ctx,
                "allocator_granularity_bytes": gran,
                "lazy_workspace_bytes": ws,
                "note": note,
            }
        )
    )
    return 0


def measure_one_in_subprocess(
    card: dict, timeout_s: int, python: Optional[str] = None
) -> Tuple[Optional[dict], Optional[str]]:
    """``(residual dict, None)`` or ``(None, named error)``. Never raises."""
    env = dict(os.environ)
    # The UUID form. An index here would be the #349 defect.
    env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
    env["PYTHONPATH"] = REPO_PYTHON + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        python or sys.executable,
        os.path.abspath(__file__),
        "--measure-child",
        card["uuid"],
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env
        )
    except subprocess.TimeoutExpired:
        return None, (
            f"TIMEOUT after {timeout_s}s on {card['name']} "
            f"(uuid {card['uuid']}, bdf {card['pci_bus_id']}). The child was "
            "killed. A CUDA call blocked uninterruptibly -- check for a wedged "
            "context, an ECC event, or another process resetting the card. "
            "Raise --timeout only after ruling those out; a longer timeout "
            "does not unwedge a context."
        )
    elapsed = time.time() - started
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        return None, (
            f"FAILED on {card['name']} (uuid {card['uuid']}, bdf "
            f"{card['pci_bus_id']}) after {elapsed:.1f}s, exit "
            f"{proc.returncode}: {detail[-800:] or '(no output)'}"
        )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except ValueError as e:
        return None, f"FAILED to parse the child's output on {card['name']}: {e}"
    if "error" in payload:
        return None, f"FAILED on {card['name']}: {payload['error']}"
    payload.setdefault("name", card["name"])
    payload["_elapsed_s"] = round(elapsed, 1)
    return payload, None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_profile(profile, cards_by_uuid: Optional[Dict[str, dict]] = None) -> None:
    """The calibrated terms, with the provenance the ledger will carry."""
    cards_by_uuid = cards_by_uuid or {}
    print()
    print(
        f"VRAM calibration {profile.fingerprint}  "
        f"(driver {profile.driver}, build {profile.build})"
    )
    print(
        "  provenance: CALIBRATED -- measured on this hardware, cached under "
        "the fingerprint above."
    )
    print(
        "  invalidated by: a change of card set, driver version or torch/CUDA "
        "build. Never adjusted, never extrapolated."
    )
    print()
    width = max([len(c.name) for c in profile.cards] + [16])
    print(
        f"  {'card':<{width}}  {'context':>11}  {'granularity':>13}  "
        f"{'workspace':>11}  {'TOTAL/proc':>11}  identity"
    )
    for card in profile.cards:
        ident = cards_by_uuid.get(card.uuid, {})
        bdf = ident.get("pci_bus_id", "?")
        print(
            f"  {card.name:<{width}}  "
            f"{card.cuda_context_bytes // (1 << 20):>7} MiB  "
            f"{card.allocator_granularity_bytes // (1 << 20):>9} MiB  "
            f"{card.lazy_workspace_bytes // (1 << 20):>7} MiB  "
            f"{card.total_mib:>7} MiB  {card.uuid} @ {bdf}"
        )
    print()
    print(
        "  The TOTAL column is what one RANK PROCESS pays on that card. A card "
        "hosting N co-located ranks is charged N x this value, plus one more "
        "context if a parent/tokenizer process binds the card (#237/#403)."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure and cache the VRAM ledger's per-card hardware residuals. "
            "One subprocess per card, pinned by UUID, time-boxed."
        )
    )
    parser.add_argument("--measure-child", metavar="UUID", help=argparse.SUPPRESS)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-card time box in seconds (default {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the cached calibration for this rig and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and print the cards and the cache path without touching "
            "a GPU. Use this to verify the plan before a GPU window."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the cache directory (tests and dry runs).",
    )
    args = parser.parse_args(argv)

    if args.measure_child:
        return measure_child(args.measure_child)

    from sglang.srt.mem_ledger.calibration import (
        CalibrationProfile,
        CardResidual,
        _build_id,
        calibration_cache_path,
        calibration_fingerprint,
        load_calibration,
        save_calibration,
    )

    if args.show:
        profile = load_calibration(cache_dir=args.cache_dir)
        if profile is None:
            print(
                "No VRAM calibration matches this rig. Run this script without "
                "--show to measure one."
            )
            return 1
        print_profile(profile)
        return 0

    # --dry-run must not initialise CUDA; NVML alone can enumerate.
    # A resolution failure is an operator-facing message, not a traceback: the
    # commonest cause is a CUDA_VISIBLE_DEVICES pin that hides the cards (a
    # hermetic shell exports 99), and a stack trace buries that under noise.
    try:
        cards, driver = resolve_cards(allow_cuda_init=not args.dry_run)
    except Exception as e:
        print(f"\nCannot resolve the cards: {e}")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            print(
                f"CUDA_VISIBLE_DEVICES is set to {visible!r}. Unset it to see "
                "the rig's real cards; this script does its own per-card "
                "pinning and must start from the full inventory."
            )
        return 1
    fingerprint = calibration_fingerprint([c["uuid"] for c in cards], driver)
    cache_path = calibration_cache_path(fingerprint, args.cache_dir)

    print("Cards resolved via NVML UUID / PCI BDF (never a bare index):")
    for c in cards:
        print(
            f"  {c['name']:<12}  uuid {c['uuid']}  bdf {c['pci_bus_id']}  "
            f"nvml {c['nvml_index']}  cuda {c['cuda_ordinal']}"
        )
    print(f"\nDriver {driver}, build {_build_id()}")
    print(f"Fingerprint {fingerprint}")
    print(f"Cache path  {cache_path}")

    if args.dry_run:
        print(
            "\n--dry-run: no GPU was touched. Remove the flag inside the "
            "window to measure."
        )
        return 0

    residuals: List[CardResidual] = []
    failures: List[str] = []
    for card in cards:
        print(f"\nMeasuring {card['name']} ({card['uuid']}), time box {args.timeout}s")
        payload, error = measure_one_in_subprocess(card, args.timeout)
        if error:
            print(f"  {error}")
            failures.append(error)
            continue
        residuals.append(
            CardResidual(
                uuid=payload["uuid"],
                name=payload.get("name", card["name"]),
                cuda_context_bytes=int(payload["cuda_context_bytes"]),
                allocator_granularity_bytes=int(payload["allocator_granularity_bytes"]),
                lazy_workspace_bytes=int(payload.get("lazy_workspace_bytes", 0)),
                note=str(payload.get("note", "")),
            )
        )
        print(f"  ok in {payload['_elapsed_s']}s")

    if failures:
        print("\nREFUSING to write a partial calibration.")
        print(
            "A cache file missing a card makes that card's hardware-residual "
            "term UNBOUNDED at the next boot, which the ledger reports as a "
            "refusal -- and that refusal would look like a ledger bug instead "
            "of like this incomplete measurement. Fix the cards below and "
            "re-run; nothing was written."
        )
        for f in failures:
            print(f"  - {f}")
        return 1

    profile = CalibrationProfile(
        fingerprint=fingerprint,
        driver=driver or "",
        build=_build_id(),
        cards=tuple(residuals),
        measured_at=time.time(),
    )
    save_calibration(profile, cache_dir=args.cache_dir)
    print_profile(profile, {c["uuid"]: c for c in cards})
    print(f"\nWritten to {cache_path}")
    print(
        "Gate (a) is met when every card above has a TOTAL and this file "
        "exists; --enable-vram-ledger will now boot instead of refusing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
