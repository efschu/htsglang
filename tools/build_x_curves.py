#!/usr/bin/env python3
"""X-CURVES 1006: build one curve file (model x form x hardware) from a
``calib_x_<Boot>.jsonl`` harvest (CPU only).

Thin wrapper of ``python -m sglang.srt.weg2.x_curves_build``:

    tools/build_x_curves.py calib_x_s9wwu9.jsonl --out nf-int4-h6-abl.xcurves.json \\
        --model <checkpoint dir name> --form arch=moe,experts=...,draft=...,kv=...,flip=... \\
        --hardware RTX5090,RTX3080,RTX3080 --p-chunk-tokens 4096 --d-chunk-tokens 4096 --cap 12288

Prints the plot check as text (every curve row, the flip price, X by depth x k).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2.x_curves_build import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
