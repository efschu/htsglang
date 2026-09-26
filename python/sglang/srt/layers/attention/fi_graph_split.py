"""KV split INSIDE the full prefill CUDA graph (27B line, group P, 2026-09-24).

Default OFF: nothing here runs unless ``SGLANG_FI_PREFILL_GRAPH_SPLIT=N`` (N =
the largest KV-chunk count, launcher ``--p-prefill-graph-split N``) is in the
rank's environment. Unset, the full prefill graph's plan is flashinfer's own,
byte for byte.

WHY FLASHINFER CANNOT DO IT (0.6.14, data/include/flashinfer/attention/):
* scheduler.cuh:717 ``int num_blocks_per_sm = 2;`` -- the prefill planner
  ASSUMES two CTAs per SM (the decode planners ask
  cudaOccupancyMaxActiveBlocksPerMultiprocessor): max_grid = 2 x 170 = 340 on
  the 5090, ``max_batch_size_if_split`` = 340 / 4 KV heads = 85.
* The real occupancy of this kernel is ONE CTA per SM: prefill.cuh:3465-3490,
  fp8 KV + head_dim 256 + CTA_TILE_Q 64 -> kUseRepack, kKVSmemPerMmaKV = 16 KiB,
  kMinValidMmaKV = 2, so two CTAs would need 2 x (32 + 32) KiB = 128 KiB > the
  SM's 100 KiB -> ``num_ctas_per_sm = 1``.
* ``PrefillSplitQOKVIndptr`` (scheduler.cuh ~520-600), graph mode: 512 new
  tokens x gqa 6 / CTA_TILE_Q 64 = 48 q tiles; the binary search keeps the
  smallest KV chunk with 48 x ceil(kv / chunk) <= 85 -- only chunk = kv fits
  (two chunks are 96), so ONE chunk; ``padded_batch_size = max(85, 48) = 85``.
  A fixed split of 5 (240 work items) hits ``FLASHINFER_CHECK(new_batch_size <=
  padded_batch_size, ... "consider disabling cuda graph")``. With the correct
  occupancy (1 CTA/SM: 170 / 4 = 42) it would not split either: 48 tiles are
  already more than one wave, and the heuristic only splits to FILL one wave --
  it has no notion of wave quantization (192 CTAs on 170 SMs = 2 rounds, 56 %).

WHAT THIS DOES, without touching the shared venv header or the JIT module:
flashinfer's ``run()`` takes the plan as a VECTOR (``wrapper._plan_info``,
``PrefillPlanInfo::FromVector``) and reads every per-work-item array at the
offsets it names (batch_prefill.cu, BatchPrefillWithPagedKVCacheRun): grid
``(padded_batch_size, 1, num_kv_heads)``, request / qo-tile / kv-tile indices,
o_indptr, kv_chunk_size (a DEVICE scalar), merge_indptr, block_valid_mask and
the split partials tmp_v / tmp_s. So after the stock ``plan()`` this module
writes its OWN arrays into a private region of the wrapper's int workspace
(async, from pinned host memory) and hands ``run()`` a vector pointing there:
* at CAPTURE: ``padded = q_tiles_max x N`` (fixed grid, e.g. 48 x 7 = 336 per
  KV head) and split_kv on, so the captured kernel + merge have room for N
  chunks;
* at every REPLAY: the arrays for THIS chunk -- the wave chooser
  (fi_prefill_wave_split.choose_prefill_kv_split) picks n <= N from the
  prefix, kv_chunk_size = ceil(kv / n), the first q_tiles x n work items valid,
  the rest masked (block_valid_mask, the same early exit the stock graph plan
  uses for its own padding).
Partials are sized by the rows the kernel really writes (rows x n x heads x
head_dim x sizeof(out)), not by flashinfer's GQA over-reservation, so N = 7 (99 %
wave fill at 192 x 7 = 1344 CTAs on 170 SMs) costs 44 MB of the float
workspace, placed at its offset 0 (the stock plan only computes offsets there;
nothing else lives in it during one attention call).

CONTRACT, CHECKED rather than assumed: flashinfer 0.6.14, fa2 backend,
page_size 1, a 15-field plan vector in the 0.6.14 order, one full-attention
wrapper. Anything else at capture -> this module stands down for that wrapper
and the stock plan stays (named once in the log). The GPU proof (graph split
vs eager stock, and our arrays vs the C++ planner's own fixed-split arrays) is
test/registered/unit/layers/attention/test_fi_graph_split_gpu_0924.py.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

#: Largest KV-chunk count the captured grid is sized for; 0/unset = off.
GRAPH_SPLIT_ENV = "SGLANG_FI_PREFILL_GRAPH_SPLIT"
#: Smallest KV chunk the chooser may pick inside the graph, tokens.
GRAPH_SPLIT_MIN_CHUNK_ENV = "SGLANG_FI_PREFILL_GRAPH_SPLIT_MIN_CHUNK"
GRAPH_SPLIT_MIN_CHUNK_DEFAULT = 512
#: flashinfer version whose plan vector and kernel reads this module mirrors.
FLASHINFER_VERSION = "0.6.14"
#: PrefillPlanInfo::ToVector order (scheduler.cuh, 0.6.14).
PLAN_FIELDS = (
    "padded_batch_size",
    "total_num_rows",
    "total_num_rows_offset",
    "cta_tile_q",
    "request_indices_offset",
    "qo_tile_indices_offset",
    "kv_tile_indices_offset",
    "merge_indptr_offset",
    "o_indptr_offset",
    "kv_chunk_size_ptr_offset",
    "v_offset",
    "s_offset",
    "block_valid_mask_offset",
    "enable_cuda_graph",
    "split_kv",
)
_F = {name: i for i, name in enumerate(PLAN_FIELDS)}
#: Our int arrays start here inside the wrapper's int workspace (8 MiB); the
#: stock planner's own arrays for one request slot stay below 4 KiB.
INT_REGION_BASE = 1 << 20
ALIGN = 16


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "") or default).strip())
    except ValueError:
        return default


def graph_split_max_chunks() -> int:
    return max(0, _env_int(GRAPH_SPLIT_ENV, 0))


def graph_split_min_chunk() -> int:
    return max(1, _env_int(GRAPH_SPLIT_MIN_CHUNK_ENV, GRAPH_SPLIT_MIN_CHUNK_DEFAULT))


def launcher_env_p_graph_split(max_chunks: int) -> Dict[str, str]:
    """Group P's environment for ``--p-prefill-graph-split N``; {} when off."""
    n = int(max_chunks or 0)
    if n <= 1:
        return {}
    return {GRAPH_SPLIT_ENV: str(n)}


def _align(x: int) -> int:
    return (int(x) + ALIGN - 1) // ALIGN * ALIGN


@dataclasses.dataclass(frozen=True)
class GraphSplitLayout:
    """The fixed shape one capture commits to. Offsets are BYTES into the
    wrapper's int workspace (``int_*``) and float workspace (``v``/``s``)."""

    padded: int
    max_rows: int
    slots: int
    max_chunks: int
    cta_tile_q: int
    num_qo_heads: int
    head_dim_vo: int
    out_bytes: int
    int_base: int = INT_REGION_BASE
    float_base: int = 0

    @property
    def offsets(self) -> Dict[str, int]:
        o: Dict[str, int] = {}
        at = self.int_base
        for name, nbytes in (
            ("request_indices", 4 * self.padded),
            ("qo_tile_indices", 4 * self.padded),
            ("kv_tile_indices", 4 * self.padded),
            ("o_indptr", 4 * (self.slots + 1)),
            ("kv_chunk_size", 4),
            ("merge_indptr", 4 * (self.max_rows + 1)),
            ("block_valid_mask", self.padded),
        ):
            o[name] = at
            at = _align(at + nbytes)
        o["int_end"] = at
        v = _align(self.float_base)
        o["v"] = v
        s = _align(v + self.max_rows * self.max_chunks * self.num_qo_heads * self.head_dim_vo * self.out_bytes)
        o["s"] = s
        o["float_end"] = _align(s + self.max_rows * self.max_chunks * self.num_qo_heads * 4)
        return o

    @property
    def int_bytes(self) -> int:
        return self.offsets["int_end"] - self.int_base

    def plan_vector(self, stock: Sequence[int]) -> List[int]:
        """The vector ``run()`` receives: the stock plan's live-row fields
        (``total_num_rows`` = the captured maximum, its device slot) and
        ``cta_tile_q``, everything else pointing at this layout."""
        o = self.offsets
        v = [int(x) for x in stock]
        v[_F["padded_batch_size"]] = self.padded
        v[_F["cta_tile_q"]] = self.cta_tile_q
        v[_F["request_indices_offset"]] = o["request_indices"]
        v[_F["qo_tile_indices_offset"]] = o["qo_tile_indices"]
        v[_F["kv_tile_indices_offset"]] = o["kv_tile_indices"]
        v[_F["merge_indptr_offset"]] = o["merge_indptr"]
        v[_F["o_indptr_offset"]] = o["o_indptr"]
        v[_F["kv_chunk_size_ptr_offset"]] = o["kv_chunk_size"]
        v[_F["v_offset"]] = o["v"]
        v[_F["s_offset"]] = o["s"]
        v[_F["block_valid_mask_offset"]] = o["block_valid_mask"]
        v[_F["enable_cuda_graph"]] = 1
        v[_F["split_kv"]] = 1
        return v


def layout_from_stock(
    stock: Sequence[int],
    *,
    slots: int,
    max_chunks: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_vo: int,
    out_bytes: int,
    int_workspace_bytes: int,
    float_workspace_bytes: int,
) -> Tuple[Optional[GraphSplitLayout], str]:
    """The capture's layout, or ``(None, reason)`` -- never a guess."""
    if len(stock) != len(PLAN_FIELDS):
        return None, "plan vector has %d fields, 0.6.14 has %d" % (len(stock), len(PLAN_FIELDS))
    if not int(stock[_F["enable_cuda_graph"]]):
        return None, "stock plan is not a cuda-graph plan"
    max_rows = int(stock[_F["total_num_rows"]])
    tile = int(stock[_F["cta_tile_q"]])
    if max_rows <= 0 or tile <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        return None, "bad geometry rows=%d tile=%d heads=%d/%d" % (max_rows, tile, num_qo_heads, num_kv_heads)
    gqa = num_qo_heads // num_kv_heads
    tiles_max = -(-max_rows * gqa // tile) + int(slots) - 1  # the C++ graph bound
    lay = GraphSplitLayout(
        padded=tiles_max * int(max_chunks),
        max_rows=max_rows,
        slots=int(slots),
        max_chunks=int(max_chunks),
        cta_tile_q=tile,
        num_qo_heads=int(num_qo_heads),
        head_dim_vo=int(head_dim_vo),
        out_bytes=int(out_bytes),
    )
    o = lay.offsets
    stock_int_top = max(
        int(stock[_F[k]])
        for k in (
            "request_indices_offset",
            "qo_tile_indices_offset",
            "kv_tile_indices_offset",
            "merge_indptr_offset",
            "o_indptr_offset",
            "kv_chunk_size_ptr_offset",
            "block_valid_mask_offset",
            "total_num_rows_offset",
        )
    )
    if stock_int_top >= lay.int_base:
        return None, "stock int arrays reach %d >= our base %d" % (stock_int_top, lay.int_base)
    if o["int_end"] > int(int_workspace_bytes):
        return None, "int region ends at %d > workspace %d" % (o["int_end"], int_workspace_bytes)
    if o["float_end"] > int(float_workspace_bytes):
        return None, "partials need %d B > float workspace %d" % (o["float_end"], float_workspace_bytes)
    return lay, "ok"


def split_arrays(
    qo_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    gqa: int,
    cta_tile_q: int,
    kv_chunk: int,
    page_size: int = 1,
) -> Dict[str, List[int]]:
    """flashinfer's own work-item layout for a FIXED KV chunk -- a line-by-line
    port of PrefillSplitQOKVIndptr step 3 (scheduler.cuh, 0.6.14, non-graph,
    no window, page_size 1): for each request, each q tile, each KV chunk one
    work item; o_indptr advances by qo_len x chunks, merge_indptr by chunks
    per row. The kernel's own chunk count, ceil(min(kv, kv + CTA_TILE_Q) /
    chunk), equals the planner's here because every page is full (the #5502
    mismatch needs trailing empty pages)."""
    if int(page_size) != 1:
        raise ValueError("page_size 1 only (every page full)")
    req: List[int] = []
    qt: List[int] = []
    kt: List[int] = []
    merge = [0]
    oind = [0]
    chunk = max(1, int(kv_chunk))
    for r, (q, k) in enumerate(zip(qo_lens, kv_lens)):
        q, k = int(q), max(1, int(k))
        tiles = -(-q * int(gqa) // int(cta_tile_q))
        n = -(-k // chunk)
        for t in range(tiles):
            for c in range(n):
                req.append(r)
                qt.append(t)
                kt.append(c)
        for _ in range(q):
            merge.append(merge[-1] + n)
        oind.append(oind[-1] + q * n)
    return {
        "request_indices": req,
        "qo_tile_indices": qt,
        "kv_tile_indices": kt,
        "merge_indptr": merge,
        "o_indptr": oind,
        "kv_chunk_size": [chunk * int(page_size)],
    }


def int_region_bytes(layout: GraphSplitLayout, arrays: Dict[str, List[int]]) -> bytearray:
    """The whole private int region as bytes (little-endian int32 arrays at the
    layout's offsets, the valid-work-item mask as bytes), ready for one copy."""
    import struct

    o = layout.offsets
    base = layout.int_base
    buf = bytearray(layout.int_bytes)
    items = len(arrays["request_indices"])
    if items > layout.padded:
        raise ValueError("%d work items > captured grid %d" % (items, layout.padded))
    rows = len(arrays["merge_indptr"]) - 1
    if rows > layout.max_rows or len(arrays["o_indptr"]) > layout.slots + 1:
        raise ValueError("rows %d / slots %d exceed the capture" % (rows, len(arrays["o_indptr"]) - 1))

    def put(name: str, values: Sequence[int]) -> None:
        at = o[name] - base
        struct.pack_into("<%di" % len(values), buf, at, *[int(v) for v in values])

    for name in ("request_indices", "qo_tile_indices", "kv_tile_indices"):
        put(name, arrays[name])
    oind = list(arrays["o_indptr"]) + [arrays["o_indptr"][-1]] * (layout.slots + 1 - len(arrays["o_indptr"]))
    put("o_indptr", oind)
    put("kv_chunk_size", arrays["kv_chunk_size"])
    put("merge_indptr", arrays["merge_indptr"])
    mask_at = o["block_valid_mask"] - base
    for i in range(items):
        buf[mask_at + i] = 1
    return buf


def choose_chunk(
    qo_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    layout: GraphSplitLayout,
    num_kv_heads: int,
    num_sm: int,
) -> Tuple[int, object]:
    """The KV chunk for this replay: the wave chooser bounded by the captured
    ``max_chunks``; one chunk (= the longest KV) when it finds no gain."""
    from sglang.srt.layers.attention.fi_prefill_wave_split import choose_prefill_kv_split

    kv_max = max(max(1, int(k)) for k in kv_lens) if kv_lens else 1
    choice = choose_prefill_kv_split(
        qo_lens,
        kv_lens,
        num_qo_heads=layout.num_qo_heads,
        num_kv_heads=int(num_kv_heads),
        head_dim=layout.head_dim_vo,
        num_sm=int(num_sm),
        # the partials are pre-sized for max_chunks; the 0.6.14 over-reservation
        # the eager chooser models does not apply to this region
        float_workspace_bytes=1 << 62,
        max_chunks=layout.max_chunks,
        min_chunk_tokens=graph_split_min_chunk(),
    )
    if choice.fixed_split_size is None:
        return kv_max, choice
    return int(choice.fixed_split_size), choice


class GraphSplitState:
    """Per wrapper: the capture's layout, double-buffered pinned staging and
    their copy events (a staging buffer is rewritten only after its previous
    host->device copy has run -- P-NOSYNC plans one forward ahead)."""

    def __init__(self, layout: GraphSplitLayout, stock: Sequence[int]):
        import torch

        self.layout = layout
        self.stock = [int(x) for x in stock]
        pin = torch.cuda.is_available()
        self.staging = [
            torch.zeros(layout.int_bytes, dtype=torch.uint8, pin_memory=pin) for _ in range(2)
        ]
        self.events: List[Optional[object]] = [None, None]
        self.turn = 0
        self.last_chunks = 1

    def write(self, wrapper, arrays: Dict[str, List[int]]) -> None:
        import torch

        data = int_region_bytes(self.layout, arrays)
        b = self.turn
        self.turn ^= 1
        ev = self.events[b]
        if ev is not None:
            ev.synchronize()
        stage = self.staging[b]
        stage.copy_(torch.frombuffer(data, dtype=torch.uint8))
        dst = wrapper._int_workspace_buffer
        base = self.layout.int_base
        dst[base : base + self.layout.int_bytes].copy_(stage, non_blocking=True)
        if torch.cuda.is_available() and dst.is_cuda:
            e = torch.cuda.Event()
            e.record()
            self.events[b] = e


def apply(
    wrapper,
    state: Optional[GraphSplitState],
    qo_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    num_kv_heads: int,
    num_sm: int,
) -> Tuple[int, object]:
    """Write this replay's arrays and point the wrapper's plan vector at them.
    Returns ``(kv_chunk, choice)``."""
    lay = state.layout
    gqa = lay.num_qo_heads // int(num_kv_heads)
    chunk, choice = choose_chunk(qo_lens, kv_lens, layout=lay, num_kv_heads=num_kv_heads, num_sm=num_sm)
    arrays = split_arrays(qo_lens, kv_lens, gqa=gqa, cta_tile_q=lay.cta_tile_q, kv_chunk=chunk)
    state.write(wrapper, arrays)
    state.last_chunks = max(arrays["kv_tile_indices"] or [0]) + 1
    wrapper._plan_info = _as_plan_vector(wrapper._plan_info, lay.plan_vector(state.stock))
    return chunk, choice


def _as_plan_vector(like, values: List[int]):
    """The same container type flashinfer's plan() returned (a tvm_ffi Array
    in 0.6.14), so ``run()`` receives what it always receives."""
    try:
        import tvm_ffi

        if isinstance(like, tvm_ffi.Array) or type(like).__module__.startswith("tvm_ffi"):
            return tvm_ffi.convert(values)
    except Exception:  # noqa: BLE001 - no tvm_ffi: plain list (tests)
        pass
    return list(values)


def flashinfer_contract_ok(wrapper) -> Tuple[bool, str]:
    """The checks a capture runs before committing to this layout."""
    try:
        import flashinfer

        ver = str(getattr(flashinfer, "__version__", ""))
    except Exception:  # noqa: BLE001
        ver = ""
    if not ver.startswith(FLASHINFER_VERSION):
        return False, "flashinfer %r, this module mirrors %s" % (ver, FLASHINFER_VERSION)
    if getattr(wrapper, "_backend", None) != "fa2":
        return False, "backend %r, not fa2" % (getattr(wrapper, "_backend", None),)
    if not getattr(wrapper, "is_cuda_graph_enabled", False):
        return False, "wrapper is not a cuda-graph wrapper"
    if getattr(wrapper, "_custom_mask_buf", None) is not None:
        return False, "custom mask"
    window_left = getattr(wrapper, "_window_left", -1)
    if window_left is not None and int(window_left) >= 0:
        # the kernel's chunk count is ceil(min(kv, window_left + CTA_TILE_Q) /
        # chunk) (prefill.cuh:3343); split_arrays counts ceil(kv / chunk), so a
        # sliding window would misplace the partials. The one-wrapper rule
        # already keeps SWA models out; this names it.
        return False, "sliding window (window_left=%d)" % int(window_left)
    return True, "ok"


def stock_eager_split_float_bytes(num_qo_heads: int, work_items: int, cta_tile_q: int, head_dim_vo: int) -> int:
    """The float workspace flashinfer 0.6.14's EAGER plan demands once it
    splits (PrefillPlan, scheduler.cuh:772-776): tmp_v = heads x work items x
    CTA_TILE_Q x head_dim x sizeof(float), then tmp_s, 16-aligned -- the GQA
    over-reservation #5177 removes upstream. Sizes the GPU parity test's
    reference plan: at 98k / 7 chunks it is 506 MiB, and a 384 MiB workspace
    makes the C++ planner refuse ("Buffer overflow ... batch_prefill_tmp_v").
    The graph path never plans this way (its partials are sized by rows)."""
    rows = int(num_qo_heads) * int(work_items) * int(cta_tile_q)
    return _align(rows * int(head_dim_vo) * 4) + rows * 4
