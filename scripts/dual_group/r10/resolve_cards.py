#!/usr/bin/env python3
"""Resolve this rig's cards ONCE, in the order the launch flags mean.

Two different orders are in play on this box and they do not agree:

  * CUDA enumeration order -- what ``--rank-gpu-id`` indexes and what
    ``CUDA_VISIBLE_DEVICES`` numbers refer to.
  * NVML / nvidia-smi order -- what the inventory prints.

Nothing here is hardcoded to either. The "big" card is defined as the one with
the largest ``total_memory`` and the "smalls" are the rest, which is the same
idiom the #336 arms used. The output is shell-eval-able:

    CUDA_BIG / CUDA_SMALL0 / CUDA_SMALL1   CUDA indices, for --rank-gpu-id
    CVD_BIG / CVD_SMALL0 / CVD_SMALL1      strings for CUDA_VISIBLE_DEVICES
                                           (GPU-<uuid> where torch exposes it,
                                           otherwise the CUDA index)

Comment lines carry the full inventory for the record. Must run with
CUDA_VISIBLE_DEVICES unset, or it will only see a subset of the rig.
"""

from __future__ import annotations

import os
import sys


def _cvd_token(props, index: int) -> str:
    """A CUDA_VISIBLE_DEVICES token that survives an enumeration change.

    UUIDs are preferred because they are stable across driver states; the
    numeric index is only a fallback for a torch build that does not expose
    ``uuid`` on the device properties.
    """
    uuid = getattr(props, "uuid", None)
    if uuid:
        text = str(uuid)
        return text if text.startswith("GPU-") else f"GPU-{text}"
    return str(index)


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, ""):
        print(
            f"# WARNING: CUDA_VISIBLE_DEVICES={visible} is set; the resolution "
            "below describes only the visible subset.",
            file=sys.stderr,
        )

    import torch

    count = torch.cuda.device_count()
    if count < 2:
        print(f"# ERROR: need at least 2 CUDA devices, found {count}", file=sys.stderr)
        return 1

    props = [torch.cuda.get_device_properties(i) for i in range(count)]
    for i, p in enumerate(props):
        print(
            f"# cuda:{i}  {p.name}  {p.total_memory // (1024 * 1024)} MiB  "
            f"{_cvd_token(p, i)}"
        )

    big = max(range(count), key=lambda i: props[i].total_memory)
    smalls = [i for i in range(count) if i != big]

    print(f"CUDA_BIG={big}")
    print(f"CVD_BIG={_cvd_token(props[big], big)}")
    print(f"NAME_BIG={props[big].name.replace(' ', '_')}")
    print(f"MIB_BIG={props[big].total_memory // (1024 * 1024)}")
    for n, idx in enumerate(smalls):
        print(f"CUDA_SMALL{n}={idx}")
        print(f"CVD_SMALL{n}={_cvd_token(props[idx], idx)}")
        print(f"NAME_SMALL{n}={props[idx].name.replace(' ', '_')}")
        print(f"MIB_SMALL{n}={props[idx].total_memory // (1024 * 1024)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
