#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Fetch the Practical-RIFE flownet checkpoints the ladder (#460) can run.

The SR ONNX is sha256-pinned (``sr.REALESR_GENERAL_WDN_X4V3.sha256``) and the
RIFE weights are held to the same standard: a version whose hash is already in
``rife.KNOWN_WEIGHT_SHA256`` is verified against it and a mismatch is a hard
refusal, because the upstream release tag is mutable on GitHub. A version with
no pin yet is refused too, unless ``--record-new-pin`` is passed -- that flag
is the one-time act of *establishing* a pin, and it prints the line to paste
into ``KNOWN_WEIGHT_SHA256``. There is deliberately no mode in which an
unpinned artifact is downloaded twice.

Usage::

    python scripts/video_enhance/fetch_rife_weights.py --list
    python scripts/video_enhance/fetch_rife_weights.py --all-vendored
    python scripts/video_enhance/fetch_rife_weights.py 4.17 4.17.lite \\
        --record-new-pin --dir /spinning/llm_stuff/k3-models/rife

Progress is printed per file, so a run in a log file shows what it is doing
rather than going silent for a minute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(REPO_PYTHON) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_PYTHON))

from sglang.srt.video_enhance.rife import (  # noqa: E402
    KNOWN_WEIGHT_SHA256,
    SUPPORTED_VERSIONS,
    WEIGHT_URL_TEMPLATE,
    default_weight_dir,
    download_weights,
    require_known,
    sha256_file,
    weight_filename,
    weights_are_cached,
)


class UnpinnedRefusal(RuntimeError):
    """A version with no recorded sha256 was requested without the opt-in."""


def _describe(version: str, directory: Path) -> str:
    pin = KNOWN_WEIGHT_SHA256.get(version)
    path = directory / weight_filename(version)
    if weights_are_cached(version, directory):
        state = "present"
    elif pin is not None:
        state = "fetchable (pinned)"
    else:
        state = "UNPINNED"
    vendored = "vendored" if version in SUPPORTED_VERSIONS else "arch not vendored"
    return f"{version:<12} {state:<20} {vendored:<18} {path}"


def fetch_one(
    version: str, directory: Path, *, record_new_pin: bool, force: bool
) -> tuple[str, str]:
    """Download one checkpoint. Returns ``(version, sha256)``."""
    require_known(version)
    pin = KNOWN_WEIGHT_SHA256.get(version)
    if pin is None and not record_new_pin:
        raise UnpinnedRefusal(
            f"{version}: no sha256 in rife.KNOWN_WEIGHT_SHA256. Re-run with "
            "--record-new-pin to establish one from a verified download, then "
            "paste the printed line into KNOWN_WEIGHT_SHA256. An unpinned "
            "re-download is refused on purpose: the upstream release tag is "
            "mutable, so without a pin a second fetch cannot be shown to have "
            "produced the same bytes."
        )
    url = WEIGHT_URL_TEMPLATE.format(version=version)
    print(f"  fetching {version} from {url}", flush=True)
    path = download_weights(version, directory, force=force)
    digest = sha256_file(path)
    size_mib = path.stat().st_size / (1 << 20)
    print(f"  {version}: {size_mib:.1f} MiB, sha256 {digest}", flush=True)
    if pin is None:
        print(f'    NEW PIN -> "{version}": "{digest}",', flush=True)
    return version, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("versions", nargs="*", help="RIFE versions, e.g. 4.17.lite")
    parser.add_argument(
        "--all-vendored",
        action="store_true",
        help="every version whose IFNet architecture is vendored in this tree",
    )
    parser.add_argument("--dir", default=None, help="weight cache directory")
    parser.add_argument(
        "--list", action="store_true", help="print the on-disk state and exit"
    )
    parser.add_argument(
        "--record-new-pin",
        action="store_true",
        help="allow a first download of a version with no recorded sha256, and "
        "print the pin line to add",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the cache validates"
    )
    args = parser.parse_args(argv)

    directory = Path(args.dir) if args.dir else default_weight_dir()
    versions = list(args.versions)
    if args.all_vendored:
        versions = sorted(SUPPORTED_VERSIONS)
    if args.list or not versions:
        print(f"weight directory: {directory}")
        for version in sorted(SUPPORTED_VERSIONS):
            print("  " + _describe(version, directory))
        if not versions:
            return 0
        return 0

    directory.mkdir(parents=True, exist_ok=True)
    print(f"weight directory: {directory}", flush=True)
    new_pins: list[tuple[str, str]] = []
    failures: list[str] = []
    for version in versions:
        try:
            got_version, digest = fetch_one(
                version,
                directory,
                record_new_pin=args.record_new_pin,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 - a per-file report, not a stack
            print(f"  {version}: FAILED {type(exc).__name__}: {exc}", flush=True)
            failures.append(version)
            continue
        if KNOWN_WEIGHT_SHA256.get(got_version) is None:
            new_pins.append((got_version, digest))
    if new_pins:
        print("\nAdd to rife.KNOWN_WEIGHT_SHA256:", flush=True)
        for version, digest in new_pins:
            print(f'    "{version}": "{digest}",', flush=True)
    if failures:
        print(f"\nfailed: {', '.join(failures)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
