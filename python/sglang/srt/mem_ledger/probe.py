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
"""``python -m sglang.srt.mem_ledger.probe`` -- calibrate the hardware residuals.

Run once per rig. The result is cached under the card set, driver and torch
build, and a boot uses it until one of those three changes. This is the ONLY
GPU-touching entry point of the ledger, deliberately behind its own command so
that no read path can spend GPU seconds by accident.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sglang.srt.mem_ledger.calibration import (
    calibration_cache_path,
    load_calibration,
    measure_calibration,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sglang.srt.mem_ledger.probe",
        description=(
            "Measure the per-card VRAM hardware residuals (CUDA context, "
            "allocator granularity, lazy kernel workspaces) and cache them "
            "under this rig's fingerprint."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-measure even when a calibration for this fingerprint exists.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the cached calibration and exit without measuring.",
    )
    parser.add_argument(
        "--from-flight-recorder",
        metavar="DIR",
        nargs="?",
        const="",
        default=None,
        help=(
            "Read the flight recorder's boot history instead of touching the "
            "GPU, and report which residuals it can calibrate. Only posts that "
            "are stable across several boots are offered; a post whose spread "
            "is wide is listed as DECLINED, because a median of a wide "
            "distribution is variance wearing a constant's clothes. Defaults "
            "to $SGLANG_VRAM_FLIGHT_DIR."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.from_flight_recorder is not None:
        from sglang.srt.mem_ledger.measured import describe, residual_overrides

        directory = args.from_flight_recorder or None
        print(describe(directory))
        overrides = residual_overrides(directory, force=True)
        if not overrides:
            print("No residual is stable enough to calibrate from this history.")
            return 1
        print("residuals this history supports:")
        for uuid in sorted(overrides):
            for field, value in sorted(overrides[uuid].items()):
                print(f"  {uuid}  {field} = {value // (1 << 20)} MiB")
        return 0

    existing = load_calibration()
    if args.show:
        if existing is None:
            print("No VRAM calibration matches this rig.")
            return 1
        _print(existing)
        return 0
    if existing is not None and not args.force:
        print("A calibration already matches this rig; pass --force to redo it.")
        _print(existing)
        return 0

    profile = measure_calibration()
    _print(profile)
    print(f"\nCached at {calibration_cache_path(profile.fingerprint)}")
    return 0


def _print(profile) -> None:
    print(
        f"VRAM calibration {profile.fingerprint} "
        f"(driver {profile.driver}, build {profile.build})"
    )
    width = max([len(c.name) for c in profile.cards] + [16])
    print(
        f"  {'card':<{width}}  {'context':>9}  {'granularity':>12}  "
        f"{'workspace':>10}  {'total':>8}"
    )
    for card in profile.cards:
        print(
            f"  {card.name:<{width}}  "
            f"{card.cuda_context_bytes // (1 << 20):>7} MiB  "
            f"{card.allocator_granularity_bytes // (1 << 20):>10} MiB  "
            f"{card.lazy_workspace_bytes // (1 << 20):>8} MiB  "
            f"{card.total_mib:>6} MiB"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
