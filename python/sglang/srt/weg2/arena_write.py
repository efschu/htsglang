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


#: fnFL2 H47: bytes per kernel element of the run-mode write. The pointer/stride
#: kernel (hicache.cuh load_vec/store_vec) holds ONE element per worker in a
#: per-thread LocalStorage of element / (32 / unroll) bytes; a whole run as the
#: element (unroll 1 above 1 KiB) spills that array to local memory and the
#: driver grows the card's LMEM reservation to it for every resident thread --
#: and keeps it. Measured (fnFL2x151, d6b7d4a1d3): stack = run/32 - 64 B, i.e.
#: PP0 run 229376 B -> 7104 B/thread x 261120 threads = 1769 MiB (base 1248 B =
#: 311 MiB, +1458 MiB), PP1 98304 -> 3008 B, PP2 65536 -> 1984 B. On fnFL2x149
#: (FR_P 0.351) that +1458 MiB landed between chunk 0 and chunk 1 of the 97k
#: prompt (cap 28951 -> 27494) and the GDN extend of chunk 1 died. 1024 B is the
#: element the mamba write already launches (same module, same block quota: no
#: new JIT build) and it leaves the stack at the base (x146/x150: 1248 B after
#: the mamba write).
RUN_ELEMENT_BYTES = 1024


def kernel_thread_bytes(element: int, unroll: int) -> int:
    """Per-thread LocalStorage of the pointer/stride kernel for one element
    (hicache.cuh: kNumThreads = 32 / unroll threads share one element)."""
    return int(element) * int(unroll) // 32


def run_split_elements(run_bytes: int, page_bytes: int, element: int = RUN_ELEMENT_BYTES) -> Optional[int]:
    """Elements per run when the run is written in ``element``-byte items
    (fnFL2 H47), or None when run or page is not a whole number of elements
    (the caller then keeps the cell/paged path -- never the whole-run element)."""
    run_bytes, page_bytes, element = int(run_bytes), int(page_bytes), int(element)
    if element <= 0 or run_bytes <= 0 or page_bytes <= 0:
        return None
    if run_bytes % element or page_bytes % element:
        return None
    return run_bytes // element


def run_split_indices(slots, n: int, page_bytes: int, element: int = RUN_ELEMENT_BYTES):
    """Index plan of the split run write: item (page p, element j) of the two
    pseudo-layers (K run, V run; pointers from :func:`run_pointers`) goes

      dst = ptr_dst[side] + idx_dst * element = base + off + slot_p * page_bytes + j * element
      src = ptr_src[side] + idx_src * element = stage + side * run + p * 2 * run + j * element

    so the kernel strides are ``element`` on both sides. Returns
    ``(idx_dst, idx_src)`` as int64 CPU tensors of ``len(slots) * n``."""
    import torch

    n, page_elems = int(n), int(page_bytes) // int(element)
    sl = torch.as_tensor(slots, dtype=torch.int64).reshape(-1, 1)
    j = torch.arange(n, dtype=torch.int64)
    idx_dst = (sl * page_elems + j).reshape(-1)
    idx_src = (torch.arange(sl.shape[0], dtype=torch.int64).reshape(-1, 1) * (2 * n) + j).reshape(-1)
    return idx_dst, idx_src


MAMBA_MODE_ENV = "SGLANG_WEG2_MAMBA_WRITE_MODE"   # kernel (default) | copy


def mamba_write_mode(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    v = str(env.get(MAMBA_MODE_ENV, "kernel")).strip().lower()
    return v if v in ("kernel", "copy") else "kernel"


def group_pieces(pieces):
    """xsn351: the mamba state's D2H pieces (temporal row per layer, conv
    channel segments per layer) grouped by element size -- one pointer/stride
    kernel launch per size. pieces: (element_size, dst_ptr, src_ptr, src_stride).
    Returns {(element_size, src_stride): ([dst_ptrs], [src_ptrs])} in first-seen order."""
    out = {}
    for es, dp, sp, ss in pieces:
        key = (int(es), int(ss))
        d, s = out.setdefault(key, ([], []))
        d.append(int(dp)); s.append(int(sp))
    return out


MAMBA_ELEMENT_ENV = "SGLANG_WEG2_MAMBA_WRITE_ELEMENT"   # bytes per kernel element, default 1024


def mamba_element_bytes(env: Optional[Mapping[str, str]] = None) -> int:
    """xsn353: a NEW element size means a NEW JIT module -- the first mamba
    write of the boot compiled kernels for 1.5-MB temporal rows in the
    scheduler thread of all three P ranks (ninja, minutes) and the PP ring
    stood (FLIP STALL). 1024 B is the cell size the KV write always used, so
    its module is in every rank's cache; a piece is split into bytes/1024
    pseudo-layers instead."""
    env = os.environ if env is None else env
    try:
        v = int(env.get(MAMBA_ELEMENT_ENV, "1024"))
    except ValueError:
        return 1024
    return v if v > 0 else 1024


def split_pieces(pieces, element: int):
    """(bytes, dst_ptr, src_ptr, src_stride) -> kernel pieces of `element` bytes
    (dst/src advanced by j*element) and the remainder pieces that are not a
    whole number of elements (the caller copies those synchronously)."""
    kern, rest = [], []
    for nbytes, dp, sp, ss in pieces:
        nbytes = int(nbytes)
        if nbytes <= 0:
            continue
        if nbytes % element:
            rest.append((nbytes, dp, sp, ss))
            continue
        for j in range(nbytes // element):
            kern.append((element, int(dp) + j * element, int(sp) + j * element, int(ss)))
    return kern, rest
