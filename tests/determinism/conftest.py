# SPDX-License-Identifier: Apache-2.0
"""Make the determinism_harness package importable when pytest is invoked
from the repo root (mirrors the tests/moe_offload sys.path convention)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
