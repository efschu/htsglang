#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #337 -- resolve a card by architecture and print a UUID for CUDA_VISIBLE_DEVICES.

WHY THIS EXISTS
===============
Window 6 ran an arm labelled ``sm120`` on a 3080.

The run sheet said to read the index from ``nvidia-smi --query-gpu=index`` and
feed it to ``CUDA_VISIBLE_DEVICES``. Those are two different orderings.
``nvidia-smi`` reports NVML order; ``CUDA_VISIBLE_DEVICES`` consumes CUDA order,
which by default is CUDA's own enumeration and can differ. On this rig NVML
index 1 is the 5090, and ``CUDA_VISIBLE_DEVICES=1`` yields an sm86 card. The
prose was followed exactly and produced the wrong card.

``CUDA_VISIBLE_DEVICES`` also accepts ``GPU-<uuid>``, which is unambiguous under
both orderings and stable across reboots and driver reloads. So the run sheet no
longer prints an index at all:

    export CUDA_VISIBLE_DEVICES=$(python3 scripts/trt_337/select_card.py --arch sm120)

and the harness re-checks it with ``--expect-arch sm120`` after CUDA has
initialised, because a resolver and an assertion protect against different
mistakes: this script can pick the wrong card if the architecture map is wrong,
and only the harness can see what CUDA actually opened.

USAGE
=====
    select_card.py --list                 # NVML index, name, arch, UUID
    select_card.py --arch sm120           # one UUID, for CUDA_VISIBLE_DEVICES
    select_card.py --arch sm86 --index 1  # the second sm86 card
"""

from __future__ import annotations

import argparse
import sys

#: Compute capability by marketing name fragment. NVML reports the name; the
#: architecture is what the arm labels refer to.
NAME_TO_ARCH = {
    "RTX 5090": "sm120",
    "RTX 5080": "sm120",
    "RTX 3080": "sm86",
    "RTX 3090": "sm86",
    "RTX 4090": "sm89",
}


def enumerate_cards() -> list:
    import pynvml

    pynvml.nvmlInit()
    out = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        uuid = pynvml.nvmlDeviceGetUUID(h)
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        arch = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            arch = f"sm{major}{minor}"
        except Exception:
            for frag, a in NAME_TO_ARCH.items():
                if frag in name:
                    arch = a
                    break
        out.append(
            {"nvml_index": i, "name": name, "uuid": uuid, "arch": arch or "unknown"}
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", default="", help="e.g. sm120, sm86")
    ap.add_argument(
        "--index", type=int, default=0, help="which card, when several match"
    )
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    cards = enumerate_cards()
    if a.list or not a.arch:
        w = max(len(c["name"]) for c in cards)
        for c in cards:
            print(
                f"nvml_index={c['nvml_index']}  {c['name']:{w}s}  "
                f"{c['arch']:8s}  {c['uuid']}"
            )
        print(
            "\nUse the UUID, never the index: CUDA_VISIBLE_DEVICES consumes "
            "CUDA order and nvidia-smi prints NVML order.",
            file=sys.stderr,
        )
        return 0

    match = [c for c in cards if c["arch"] == a.arch]
    if not match:
        have = ", ".join(sorted({c["arch"] for c in cards}))
        raise SystemExit(
            f"no card with arch {a.arch}; this host has: {have}"
        )
    if a.index >= len(match):
        raise SystemExit(
            f"--index {a.index} but only {len(match)} card(s) with arch {a.arch}"
        )
    print(match[a.index]["uuid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
