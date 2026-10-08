#!/usr/bin/env python3
"""HW-AP0 1525: the progress meter of the hardware-generic work (CPU only).

Thin wrapper of ``python -m flliper.srt.pdflip.hw_progress``:

    tools/hw_progress.py                             # every example configuration x release model
    tools/hw_progress.py --models NF,27B-INT8
    tools/hw_progress.py --configs "3x3070,3x5070Ti" --kv-tokens 131072
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip.hw_progress import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
