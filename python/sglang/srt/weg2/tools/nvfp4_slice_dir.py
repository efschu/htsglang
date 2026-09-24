# SPDX-License-Identifier: Apache-2.0
"""H68b: an N-layer VIEW of a Qwen4-Exp (Qwen3.8-Flash-Next) checkpoint for slice smokes.

Why a directory and not ``--json-model-override-args``: sglang applies the JSON
override with ``PretrainedConfig.update`` (``utils/hf_transformers/config.py``),
which REPLACES a nested key -- ``{"text_config": {"num_hidden_layers": 4}}``
would turn ``text_config`` into a bare dict. So the slice is a sibling
directory: every checkpoint file as a SYMLINK (no weight byte is copied or
written) and one rewritten ``config.json`` with ``text_config.num_hidden_layers
= N`` and ``layer_types[:N]``. The loader skips every tensor of a layer >= N by
NAME before reading it (``qwen4_exp.weight_name_needed`` /
``weight_layer_is_owned``); measured on nvidia/Qwen3.8-Flash-Next-NVFP4 with
N=4: 24,679 of 299,545 tensors read (8.28 GiB), the PLE table mapped (not
read), layers 0-3 only.

CLI: ``python <tree>/python/sglang/srt/weg2/tools/nvfp4_slice_dir.py <checkpoint> <slice_dir> [N]``
-- run as a FILE: this module imports nothing but the stdlib, while ``-m``
would import the ``sglang.srt.weg2`` package (and torch) first.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

#: The PLE sits at layer index 1 (``ple_layer_ids`` [2], 1-based), so a slice
#: needs at least two layers to carry it.
MIN_LAYERS = 2


def make_slice_dir(src: str, dst: str, n: int = 4) -> List[str]:
    """Build (idempotently) the N-layer view; returns the text layer types."""
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if name.startswith(".") or name == "config.json":
            continue
        link = os.path.join(dst, name)
        target = os.path.join(src, name)
        if os.path.islink(link):
            if os.readlink(link) != target:
                raise RuntimeError(f"{link} points to {os.readlink(link)}, not {target}")
            continue
        if os.path.exists(link):
            raise RuntimeError(f"{link} exists and is not a symlink; not touching it")
        os.symlink(target, link)
    with open(os.path.join(src, "config.json")) as fh:
        cfg = json.load(fh)
    tc = cfg["text_config"]
    full = int(tc["num_hidden_layers"])
    if not (MIN_LAYERS <= int(n) <= full):
        raise ValueError(f"N={n} outside [{MIN_LAYERS}, {full}]")
    tc["num_hidden_layers"] = int(n)
    tc["layer_types"] = list(tc["layer_types"][: int(n)])
    cfg["_h68b_slice"] = {"of": src, "layers": int(n), "full_layers": full}
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return tc["layer_types"]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    n = int(argv[2]) if len(argv) > 2 else 4
    types = make_slice_dir(argv[0], argv[1], n)
    print(f"NVFP4-SLICE-DIR {argv[1]}: {n} layers {types}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
