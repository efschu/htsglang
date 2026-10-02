"""Owner-lane DMA for the Form-A worker's arena loadback (NF, uneven DCP-KV).

MEASURED (boot y6o, 01.10. 21:36-21:59Z, NF P->D): after the D wake the
expert workers TP1/TP2 load their owner rows back with
``ArenaMHAHostPool._owner_page_prefix_load`` -> ``_load_pages_all_layers``
(mode=dma) and copy WHOLE pages into the device stage -- 3601 pages = 2.83 GB
per worker -- although TP1 owns 40 and TP2 24 of the 64 lanes of a page. TP1
sits on the x4 link (~6.5 GB/s, ~440 ms of DMA) and TP0 waits on it in the
first collectives after the wake, so it bounds the first and second token.

A page is K/V-major, layer-major, TOKEN-MINOR (``ArenaMHAHostPool.bind``):
2L blocks of ``P * cell`` bytes. The owner rule gives token ``t`` of a page to
the rank with ``t % S`` in ``[lo, hi)`` (``owner_token_runs``), the same
offsets in every page. So a rank's owned cells of a run of consecutive pages
are ROWS of ``(hi - lo) * cell`` bytes, ``S * cell`` apart, starting at
``lo * cell`` -- uniformly across block and page borders, because ``P`` is a
multiple of ``S`` and the blocks tile the page. One cudaMemcpy2DAsync per run
of consecutive slots (src pitch = S * cell, width = (hi - lo) * cell) lands
exactly the owned cells in a compact device stage, in (page, block, lane)
order; the per-layer scatter then reads a layer's owned cells of every staged
page as a strided view (no device copy) and puts them into the compact device
rows, one kernel per layer and K/V.

Bytes per page drop from ``2L * P * cell`` to ``2L * m * cell`` (m owned
lanes): TP1 2.83 -> 1.77 GB, TP2 -> 1.06 GB on y6o's 3601 pages. The compact
stage keeps the whole-page stage's BYTES, so it holds P/m times the pages
and the block count (one scatter round each) drops by the same factor.

Pure functions; the pool keeps the dispatch and the fallback
(``ArenaMHAHostPool._owner_lane_load``).
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Optional, Sequence

import msgspec
import torch

_CUDA_MEMCPY_HOST_TO_DEVICE = 1


class OwnerLaneGeometry(msgspec.Struct, frozen=True):
    """Where a rank's owned cells sit inside one arena page, as 2D rows."""

    lo_b: int                  # byte offset of the first owned cell of a block
    spitch: int                # source row pitch: the owner split S x cell
    width: int                 # one owned run: (hi - lo) x cell
    rows_per_page: int         # 2L x P / S source rows per page
    page_bytes: int            # the whole page (2L x P x cell)
    compact_page_bytes: int    # the owned cells of a page (2L x m x cell)
    lanes_per_page: int        # m
    k_offs_c: tuple            # each layer's K block in the compact page
    v_offs_c: tuple            # each layer's V block in the compact page


def _lane_runs(lanes: Sequence[int]) -> list:
    runs: list = []
    for t in lanes:
        if runs and t == runs[-1][1]:
            runs[-1][1] = t + 1
        else:
            runs.append([t, t + 1])
    return runs


def owner_lane_geometry(*, lanes, page_tokens: int, cell: int, k_offs_b, v_offs_b,
                        page_bytes: int) -> Optional[OwnerLaneGeometry]:
    """The 2D-row form of the owned ``lanes`` (token offsets in a page), or
    None when the lanes or the page layout do not have it -- the caller then
    keeps the whole-page load. Checked, not assumed: equal-width runs of
    consecutive lanes, one every S tokens, P/S of them; blocks of ``P * cell``
    that tile the page exactly."""
    ls = [int(x) for x in (lanes.tolist() if torch.is_tensor(lanes) else lanes)]
    P, cell, pb = int(page_tokens), int(cell), int(page_bytes)
    m = len(ls)
    if m == 0 or P <= 1 or cell <= 0 or ls != sorted(set(ls)) or ls[0] < 0 or ls[-1] >= P:
        return None
    runs = _lane_runs(ls)
    w = runs[0][1] - runs[0][0]
    if any(b - a != w for a, b in runs):
        return None
    stride = P if len(runs) == 1 else runs[1][0] - runs[0][0]
    if any(runs[i + 1][0] - runs[i][0] != stride for i in range(len(runs) - 1)):
        return None
    lo = runs[0][0]
    if stride * len(runs) != P or lo + w > stride:
        return None
    block = P * cell
    L = len(k_offs_b)
    offs = [int(o) for o in k_offs_b] + [int(o) for o in v_offs_b]
    if len(v_offs_b) != L or sorted(offs) != [j * block for j in range(2 * L)] or pb != 2 * L * block:
        return None
    cblock = m * cell
    return OwnerLaneGeometry(
        lo_b=lo * cell,
        spitch=stride * cell,
        width=w * cell,
        rows_per_page=2 * L * len(runs),
        page_bytes=pb,
        compact_page_bytes=2 * L * cblock,
        lanes_per_page=m,
        k_offs_c=tuple((int(o) // block) * cblock for o in k_offs_b),
        v_offs_c=tuple((int(o) // block) * cblock for o in v_offs_b),
    )


def owner_lane_dma_plan(*, geom: OwnerLaneGeometry, runs) -> list:
    """``page_dma_runs`` output -> one 2D copy per run as ``(dst_off,
    src_off, height)``: stage bytes from ``dst_off`` (rows ``width`` apart),
    arena bytes from ``src_off`` (rows ``spitch`` apart)."""
    return [
        (int(row) * geom.compact_page_bytes,
         int(first) * geom.page_bytes + geom.lo_b,
         int(count) * geom.rows_per_page)
        for row, first, count in runs
    ]


def lane_stage_pages(*, whole_stage_pages: int, geom: OwnerLaneGeometry) -> int:
    """Pages per compact stage at the whole-page stage's bytes (never more)."""
    return max(1, (int(whole_stage_pages) * geom.page_bytes) // geom.compact_page_bytes)


_CUDART = None


def _torch_cudart_path() -> str:
    """The libcudart torch itself runs on (its CUDA major), read from this
    process's mappings -- a second runtime copy would keep its own current
    device. The transport's search is the fallback."""
    want = "libcudart.so." + str(torch.version.cuda or "").split(".")[0]
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split()[-1] if line.strip() else ""
                if os.path.basename(path) == want:
                    return path
    except OSError:
        pass
    from sglang.srt.weg2.weight_exchange_transport import find_libcudart

    return find_libcudart()


def _cudart():
    global _CUDART
    if _CUDART is None:
        path = _torch_cudart_path()
        lib = ctypes.CDLL(path)
        lib.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_size_t,
                                          ctypes.c_int, ctypes.c_void_p]
        lib.cudaMemcpy2DAsync.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaGetLastError.restype = ctypes.c_int
        _CUDART = (lib, path)
    return _CUDART


def issue_2d(*, lib, path: str, plan, geom: OwnerLaneGeometry, src_base: int, dst_base: int,
             stream: int) -> int:
    """One cudaMemcpy2DAsync per plan entry (dst pitch = width: the compact
    stage). A refused copy raises with the runtime's own message."""
    n = 0
    s = ctypes.c_void_p(int(stream))
    for dst_off, src_off, height in plan:
        rc = lib.cudaMemcpy2DAsync(
            ctypes.c_void_p(dst_base + dst_off), geom.width,
            ctypes.c_void_p(src_base + src_off), geom.spitch,
            geom.width, height, _CUDA_MEMCPY_HOST_TO_DEVICE, s)
        if int(rc) != 0:
            msg = lib.cudaGetErrorString(int(rc))
            lib.cudaGetLastError()  # #1378: clear the latched code, report it once, here
            raise RuntimeError(
                f"cudaMemcpy2DAsync rc={int(rc)} {msg.decode() if msg else '?'} "
                f"(width={geom.width} spitch={geom.spitch} height={height} libcudart={path})")
        n += geom.width * height
    return n


def copy_plan(*, plan, geom: OwnerLaneGeometry, page_view: torch.Tensor, stage: torch.Tensor) -> int:
    """Issue the plan's 2D copies arena -> stage on the current stream (CUDA:
    cudaMemcpy2DAsync from the registered arena; CPU desk: the same rows as a
    strided torch copy). Returns the bytes copied."""
    if stage.device.type == "cuda":
        lib, path = _cudart()
        with torch.cuda.device(stage.device):
            return issue_2d(lib=lib, path=path, plan=plan, geom=geom,
                            src_base=int(page_view.data_ptr()), dst_base=int(stage.data_ptr()),
                            stream=int(torch.cuda.current_stream(stage.device).cuda_stream))
    n = 0
    flat = page_view.reshape(-1)
    sflat = stage.view(-1)
    for dst_off, src_off, height in plan:
        src = flat.as_strided((height, geom.width), (geom.spitch, 1), flat.storage_offset() + src_off)
        sflat[dst_off:dst_off + height * geom.width].view(height, geom.width).copy_(src)
        n += geom.width * height
    return n


def _word_view(t: torch.Tensor):
    """A device KV buffer as the widest integer words its rows allow (byte
    moves only; a dtype-agnostic scatter)."""
    e = t.element_size()
    last = int(t.shape[-1]) * e
    for wd in (torch.int64, torch.int32, torch.int16):
        w = wd.itemsize
        if last % w == 0 and (int(t.storage_offset()) * e) % w == 0 and t.stride(-1) == 1:
            try:
                return t.view(torch.uint8).view(wd), wd
            except RuntimeError:
                continue
    return t.view(torch.uint8), torch.uint8


def scatter_lanes(*, stage: torch.Tensor, b: int, geom: OwnerLaneGeometry, dst: torch.Tensor,
                  k_buffers, v_buffers) -> int:
    """Put every layer's owned cells of the ``b`` staged pages into the
    compact device rows ``dst`` (``m`` per page, in lane order). A layer's
    block of the compact stage is a strided view ``(b, m, ...)``; one
    ``index_put_`` per layer and K/V. Returns the kernels launched."""
    m = geom.lanes_per_page
    dst2 = dst.view(b, m)
    cblock = geom.compact_page_bytes // (2 * len(geom.k_offs_c))  # m x cell
    launched = 0
    for bufs, offs in ((k_buffers, geom.k_offs_c), (v_buffers, geom.v_offs_c)):
        for buf, off in zip(bufs, offs):
            tgt, wd = _word_view(buf)
            w = wd.itemsize
            words = stage[:b].view(wd)
            o = off // w
            vals = words[:, o:o + cblock // w].view(b, m, *tgt.shape[1:])
            tgt.index_put_((dst2,), vals)
            launched += 1
    return launched


class LaneLoadStats(msgspec.Struct):
    pages: int = 0
    blocks: int = 0
    copies: int = 0
    bytes: int = 0
    kernels: int = 0
    stage_pages: int = 0
    cpu_ms: float = 0.0


class LaneDmaFailed(RuntimeError):
    """A 2D copy was refused after ``done`` pages were fully loaded."""

    def __init__(self, done: int, exc: BaseException):
        super().__init__(f"{type(exc).__name__}: {exc}")
        self.done = int(done)


def load_owner_lanes(*, page_view: torch.Tensor, piece_pages: int, slots: torch.Tensor,
                     device_indices: torch.Tensor, k_buffers, v_buffers,
                     geom: OwnerLaneGeometry, stage_pages: int, runs_of) -> LaneLoadStats:
    """The worker loadback of ``slots`` (whole owner groups) with only the
    owned lanes crossing the link. ``runs_of(block_slots, piece_pages)`` is
    the pool's ``page_dma_runs`` (a run never leaves one registration).
    Raises ``LaneDmaFailed`` with the pages already loaded."""
    t0 = time.perf_counter()
    n = int(slots.numel())
    m = geom.lanes_per_page
    dev = k_buffers[0].device
    B = max(1, min(int(stage_pages), n))
    stage = torch.empty((B, geom.compact_page_bytes), dtype=torch.uint8, device=dev)
    cpu_slots = slots.to("cpu", dtype=torch.int64)
    if dev.type == "cuda" and device_indices.device.type != "cuda":
        dst_all = device_indices.to("cpu", dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
    else:
        dst_all = device_indices.to(device=dev, dtype=torch.int64)
    st = LaneLoadStats(pages=n, stage_pages=B)
    for start in range(0, n, B):
        b = min(B, n - start)
        plan = owner_lane_dma_plan(geom=geom, runs=runs_of(cpu_slots[start:start + b].tolist(), piece_pages))
        try:
            st.bytes += copy_plan(plan=plan, geom=geom, page_view=page_view, stage=stage)
        except Exception as exc:  # noqa: BLE001 -- the pool falls back for the rest, by name
            raise LaneDmaFailed(start, exc) from exc
        st.copies += len(plan)
        st.kernels += scatter_lanes(stage=stage, b=b, geom=geom, dst=dst_all[start * m:(start + b) * m],
                                    k_buffers=k_buffers, v_buffers=v_buffers)
        st.blocks += 1
    st.cpu_ms = (time.perf_counter() - t0) * 1000.0
    return st
