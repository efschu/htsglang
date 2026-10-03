#!/usr/bin/env python3
"""HW-P1b 1003: the hardware simulation harness (CPU only, no GPU, no NVML).

Thin wrapper of ``python -m sglang.srt.weg2.hw_sim`` that finds the tree's
``python/`` itself:

    tools/hw_sim.py                                  # grid 1..6 cards x arch x model
    tools/hw_sim.py --n 1,2,3,4,5,6,7,8 --wide       # + argv shape and notes
    tools/hw_sim.py --inventory 3080-20G,5090,3080-20G --cards 1,0
    tools/hw_sim.py --catalog
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2.hw_sim import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
