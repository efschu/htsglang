"""xsn345 (18.09.2026): P's write-through into the arena (card -> page slot)
ran the 1-KiB-cell all-layer kernel with the kernel's default block quota (2
blocks): ~1 GB/s per rank, so the pages of the LAST 100k request of a phase
were complete in the arena only ~8 s after its retain, D's dormant read found
them 4 s after the flip began and the wake paid 2.9 s. The loader had the same
shape and was rebuilt as whole-page transfers (LOADBACK-SEITENKERNEL).

Run mode: this rank's K extents of a page are ONE contiguous run (layer l at
k_off + l * cell), so are its V extents; the card gathers its layers into a
run-shaped stage and the pointer/stride kernel writes two runs per page
instead of 2 x L cells. Pure decisions live here so the desk can test them."""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Tuple

MODE_ENV = "SGLANG_WEG2_ARENA_WRITE_MODE"          # run (default) | cell
QUOTA_ENV = "SGLANG_WEG2_ARENA_WRITE_BLOCK_QUOTA"  # kernel blocks; default 16 (the loader's)


def write_mode(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    v = str(env.get(MODE_ENV, "run")).strip().lower()
    return v if v in ("run", "cell") else "run"


def write_block_quota(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """None = the kernel default (2)."""
    env = os.environ if env is None else env
    v = str(env.get(QUOTA_ENV, "16")).strip()
    try:
        q = int(v)
    except ValueError:
        return 16
    return q if q > 0 else None


def contiguous_runs(k_offs: Sequence[int], v_offs: Sequence[int], cell: int) -> Optional[Tuple[int, int, int]]:
    """(k_off, v_off, run_bytes) when both extent lists are one run each
    (layer l at off + l * cell); None otherwise (per-layer interleaved
    extents, e.g. the draft page) -- the caller then keeps the cell kernel."""
    L = len(k_offs)
    if L == 0 or len(v_offs) != L or cell <= 0:
        return None
    k0, v0 = int(k_offs[0]), int(v_offs[0])
    for l in range(L):
        if int(k_offs[l]) != k0 + l * cell or int(v_offs[l]) != v0 + l * cell:
            return None
    run = L * cell
    if k0 < v0 and k0 + run > v0:
        return None
    if v0 < k0 and v0 + run > k0:
        return None
    return k0, v0, run


def run_pointers(data_base: int, k_off: int, v_off: int, run_bytes: int, stage_ptr: int):
    """dst pointer per pseudo-layer (K run, V run) into the arena data region,
    src pointer per pseudo-layer into the (b, 2 * run_bytes) device stage,
    and the stage's row stride."""
    dst = [int(data_base) + int(k_off), int(data_base) + int(v_off)]
    src = [int(stage_ptr), int(stage_ptr) + int(run_bytes)]
    return dst, src, 2 * int(run_bytes)
