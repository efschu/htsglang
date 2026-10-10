# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram table out of a GGUF file (NF-GGUF AP G2).

The safetensors checkpoints of Qwen3.8-Flash-Next carry the PLE table as 128
``ngram_embedding.shard_N`` tensors of ``[2500012, 160]`` (bf16 or fp8); the
``checkpoint`` PLE offload backend (``qwen4_exp_ple_table.CheckpointMappedPleTable``)
maps those files read-only and the gather kernel reads rows through HMM. The
unsloth GGUF export (llama.cpp ``qwen4exp``, converter ``_place_ple_shard``)
holds the same table as ONE tensor ``per_layer_token_embd.weight``, ggml ne =
``[160, 320001536]`` (torch ``[320001536, 160]``), type IQ4_NL (20): 160
elements = 5 blocks of 32, 18 bytes per block (fp16 ``d`` + 16 nibble bytes),
so one row is **90 bytes**, the table 28 800 138 240 bytes (real header of
``Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf``).

This module is the GGUF-side twin of ``CheckpointMappedPleTable``:

* :func:`locate_gguf_tensor` -- offset and length of the table taken from the
  GGUF header alone (own stdlib parser; nothing but the header bytes is read,
  the 49 GB part is never loaded);
* :class:`GgufMappedPleTable` -- the table's byte span mapped lazily (read-only
  ``np.memmap``), row lookup returning the IQ4_NL blocks of a row, a CPU
  reference lookup, and the device dequantisation of gathered rows;
* the dequantisation itself is the EXISTING sgl-kernel ``ggml_dequantize``
  (``csrc/quantization/gguf/dequantize.cuh`` ``dequantize_block_iq4_nl``) -- no
  kernel is added. The GPU side of a gather is two existing pieces: the PLE
  shard-gather kernel copies the 90-byte rows (as 45 bf16 *bit patterns*) out of
  the HMM-visible mapping into a small device buffer, then ``ggml_dequantize``
  turns that buffer into bf16 rows. Both are capturable in a CUDA graph.

Row-count rule of the sgl-kernel IQ4_NL dequantiser: it works in superblocks of
256 elements (``dequantize_row_iq4_nl_cuda`` rounds ``k = m * n`` UP to a
multiple of 256 and the kernel then touches all 8 blocks of the last
superblock). For 160-element rows that means ``m`` must be a multiple of 8, or
the kernel reads and writes beyond its buffers. :meth:`GgufMappedPleTable.padded_rows`
gives the padded row count; the padding rows are zero bytes (``d = 0``, which
dequantises to exact zeros).

Every other ggml type of the table (Q8_0, Q4_K, BF16, F16, ...) is refused by
name: :func:`check_ple_table_supported`.

No Triton / CUDA import at module level, so the parser, the lookup and the
reference dequantisation are unit-tested on CPU.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

#: GGUF tensor name of the table (llama.cpp ``qwen4exp`` ``PER_LAYER_TOKEN_EMBD``)
PLE_TABLE_GGUF_NAME = "per_layer_token_embd.weight"

#: leaf of the marker tensor the GGUF adapter yields for the table; the marker's
#: uint8 payload is the table's location (``encode_ple_table_marker``)
PLE_TABLE_MARKER_LEAF = "gguf_table"

# ggml type ids (ggml.h enum ggml_type); only the names the refusal prints
_GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4",
}

IQ4_NL_TYPE = 20
QK4_NL = 32  # elements per IQ4_NL block (ggml QK4_NL)
IQ4_NL_BLOCK_BYTES = 18  # sizeof(block_iq4_nl): ggml_half d + uint8 qs[16]

#: ggml ``kvalues_iq4nl`` (ggml-common.h): the 16 non-linear levels the nibbles
#: index; ``y = d * kvalues[nibble]``
KVALUES_IQ4NL: Tuple[int, ...] = (
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
)

# GGUF value types (gguf spec)
_GGUF_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_DEFAULT_ALIGNMENT = 32


def ggml_type_name(t: int) -> str:
    return _GGML_TYPE_NAMES.get(int(t), f"type{int(t)}")


# ---------------------------------------------------------------------------
# GGUF header: offset / length of one tensor, header bytes only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufTensorLoc:
    """Where one tensor's payload lives (absolute byte offset in ``path``)."""

    path: str
    name: str
    ggml_type: int
    ne: Tuple[int, ...]  # ggml order: ne[0] is the contiguous (innermost) dim
    data_offset: int  # absolute, from file start
    n_bytes: int
    file_size: int

    @property
    def rows(self) -> int:
        """torch dim 0 of a 2-D tensor (ggml ``ne[1]``)."""
        return int(self.ne[1]) if len(self.ne) >= 2 else 1

    @property
    def dim(self) -> int:
        """torch dim 1 of a 2-D tensor (ggml ``ne[0]``)."""
        return int(self.ne[0])


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError(f"{f.name}: truncated GGUF header (wanted {n} bytes, got {len(b)})")
    return b


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_value(f, vtype: int) -> Optional[Any]:
    """Consume one KV value; returns it only for the scalar/string types the
    caller may need (``general.alignment``)."""
    fmt = _GGUF_SCALAR_FMT.get(vtype)
    if fmt is not None:
        return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]
    if vtype == _GGUF_STRING:
        return _read_string(f)
    if vtype == _GGUF_ARRAY:
        (etype,) = struct.unpack("<I", _read_exact(f, 4))
        (count,) = struct.unpack("<Q", _read_exact(f, 8))
        efmt = _GGUF_SCALAR_FMT.get(etype)
        if efmt is not None:
            f.seek(struct.calcsize(efmt) * count, os.SEEK_CUR)
        elif etype == _GGUF_STRING:
            for _ in range(count):
                (n,) = struct.unpack("<Q", _read_exact(f, 8))
                f.seek(n, os.SEEK_CUR)
        else:
            raise ValueError(f"{f.name}: GGUF array of unsupported element type {etype}")
        return None
    raise ValueError(f"{f.name}: unsupported GGUF value type {vtype}")


def parse_gguf_tensor_infos(path: str) -> Tuple[List[GgufTensorLoc], int]:
    """All tensor infos of one GGUF file with absolute payload offsets.

    Reads the header only (KV block + tensor info block); the payload is never
    touched. ``data_offset`` follows the GGUF spec: the tensor's relative offset
    plus the start of the data section, which is the end of the tensor-info
    block rounded up to ``general.alignment`` (default 32).

    Returns ``(tensors, data_start)``.
    """
    size = os.path.getsize(path)
    with open(path, "rb", buffering=1 << 20) as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (bad magic)")
        (version,) = struct.unpack("<I", _read_exact(f, 4))
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} (only v2/v3 are supported)")
        n_tensors, n_kv = struct.unpack("<QQ", _read_exact(f, 16))
        alignment = _GGUF_DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", _read_exact(f, 4))
            value = _skip_value(f, vtype)
            if key == "general.alignment":
                alignment = int(value)
        if alignment <= 0:
            raise ValueError(f"{path}: general.alignment {alignment}")
        raw: List[Tuple[str, Tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
            ne = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
            ggml_type, rel = struct.unpack("<IQ", _read_exact(f, 12))
            raw.append((name, tuple(int(d) for d in ne), int(ggml_type), int(rel)))
        end = f.tell()
    data_start = (end + alignment - 1) // alignment * alignment
    out: List[GgufTensorLoc] = []
    for name, ne, ggml_type, rel in raw:
        out.append(
            GgufTensorLoc(
                path=path,
                name=name,
                ggml_type=ggml_type,
                ne=ne,
                data_offset=data_start + rel,
                n_bytes=_tensor_nbytes(ggml_type, ne),
                file_size=size,
            )
        )
    return out, data_start


def _tensor_nbytes(ggml_type: int, ne: Sequence[int]) -> int:
    """Payload bytes of a tensor; only the types the PLE table can have are sized."""
    n_elem = 1
    for d in ne:
        n_elem *= int(d)
    if ggml_type == IQ4_NL_TYPE:
        if n_elem % QK4_NL:
            raise ValueError(f"IQ4_NL tensor of {n_elem} elements (not a multiple of {QK4_NL})")
        return n_elem // QK4_NL * IQ4_NL_BLOCK_BYTES
    # other types: size only matters for the refusal path, which never maps them
    sizes = {0: 4, 1: 2, 30: 2, 28: 8, 24: 1, 25: 2, 26: 4, 27: 8}
    if ggml_type in sizes:
        return n_elem * sizes[ggml_type]
    return 0  # unknown here; check_ple_table_supported refuses before any use


def locate_gguf_tensor(gguf_paths: Sequence[str], name: str) -> GgufTensorLoc:
    """Find tensor ``name`` over the parts of a split export (header reads only).

    A name that occurs in two parts, or in none, is an error.
    """
    found: List[GgufTensorLoc] = []
    for p in gguf_paths:
        tensors, _ = parse_gguf_tensor_infos(p)
        found.extend(t for t in tensors if t.name == name)
    if not found:
        raise KeyError(f"GGUF tensor {name!r} is in none of {len(gguf_paths)} part(s): {list(gguf_paths)[:3]}")
    if len(found) > 1:
        raise ValueError(f"GGUF tensor {name!r} occurs in {len(found)} parts: {[t.path for t in found]}")
    loc = found[0]
    if loc.n_bytes and loc.data_offset + loc.n_bytes > loc.file_size:
        raise ValueError(
            f"{loc.path}: tensor {name!r} spans bytes {loc.data_offset}..{loc.data_offset + loc.n_bytes} "
            f"but the file has {loc.file_size} (truncated download?)"
        )
    return loc


def check_ple_table_supported(loc: GgufTensorLoc) -> None:
    """Refuse every table format but IQ4_NL by name (G2 scope)."""
    if int(loc.ggml_type) != IQ4_NL_TYPE:
        raise NotImplementedError(
            f"GGUF PLE table {loc.name!r} in {os.path.basename(loc.path)} is ggml type "
            f"{ggml_type_name(loc.ggml_type)} ({int(loc.ggml_type)}); only IQ4_NL (20) is "
            "supported for the PLE n-gram table (NF-GGUF G2). BF16/F16 and the other "
            "quantised formats are not implemented -- re-export the model with "
            "per_layer_token_embd in IQ4_NL or use the safetensors checkpoint"
        )
    if len(loc.ne) != 2:
        raise ValueError(f"GGUF PLE table {loc.name!r}: expected 2 dims, got ne={loc.ne}")
    if loc.dim % QK4_NL:
        raise ValueError(f"GGUF PLE table {loc.name!r}: row of {loc.dim} elements is not a whole number of IQ4_NL blocks")


# ---------------------------------------------------------------------------
# marker: how the adapter hands the table's location to the model
# ---------------------------------------------------------------------------


def encode_ple_table_marker(loc: GgufTensorLoc) -> torch.Tensor:
    """A small uint8 tensor (JSON) carrying the table's location through the
    GGUF weight stream; the 28.8 GB payload itself never goes through it."""
    return torch.tensor(list(json.dumps(asdict(loc)).encode("utf-8")), dtype=torch.uint8)


def decode_ple_table_marker(marker: torch.Tensor) -> GgufTensorLoc:
    d = json.loads(bytes(marker.to(torch.uint8).tolist()).decode("utf-8"))
    d["ne"] = tuple(int(x) for x in d["ne"])
    return GgufTensorLoc(**d)


# ---------------------------------------------------------------------------
# IQ4_NL: reference dequantisation (ggml spec) and the table
# ---------------------------------------------------------------------------


def iq4_nl_row_bytes(dim: int) -> int:
    if dim % QK4_NL:
        raise ValueError(f"row of {dim} elements is not a whole number of IQ4_NL blocks of {QK4_NL}")
    return dim // QK4_NL * IQ4_NL_BLOCK_BYTES


def dequantize_iq4_nl_reference(raw, dim: int):
    """ggml ``dequantize_row_iq4_nl`` written out (ggml-quants.c): per block of
    18 bytes, ``d`` = fp16 at bytes 0..1, ``qs`` = bytes 2..17; element ``j`` of
    the first half is ``d * kvalues[qs[j] & 0xF]``, element ``j + 16`` is
    ``d * kvalues[qs[j] >> 4]``.

    ``raw``: ``uint8`` numpy array ``[n, row_bytes]``; returns ``float32``
    ``[n, dim]``. ``d * kvalue`` is exact in float32 (11-bit mantissa times a
    7-bit integer), so the only rounding of a lookup is the final bf16 cast.
    """
    import numpy as np

    rb = iq4_nl_row_bytes(dim)
    raw = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1, rb)
    n = raw.shape[0]
    nb = dim // QK4_NL
    blocks = raw.reshape(n, nb, IQ4_NL_BLOCK_BYTES)
    d = np.ascontiguousarray(blocks[:, :, :2]).view("<f2").reshape(n, nb).astype(np.float32)
    qs = blocks[:, :, 2:]
    kv = np.asarray(KVALUES_IQ4NL, dtype=np.float32)
    lo = kv[qs & 0x0F]
    hi = kv[qs >> 4]
    out = np.empty((n, nb, QK4_NL), dtype=np.float32)
    out[:, :, :16] = d[:, :, None] * lo
    out[:, :, 16:] = d[:, :, None] * hi
    return out.reshape(n, dim)


class GgufMappedPleTable:
    """The IQ4_NL PLE table as a lazy read-only mapping of its byte span.

    Interface parallel to ``CheckpointMappedPleTable`` where the gather kernel
    needs it: ``bases_on(device)`` (one base pointer) and ``shard_rows``
    (= ``total_rows``, one "shard"). ``row_bytes`` is 90, NOT the dequantised
    row; ``embedding_dim`` (160) is the dequantised width.

    Deliberately NOT a subclass of ``CheckpointMappedPleTable``: the pread gather,
    the file prefetcher, the decode stage and the D-side warm of that class all
    assume ``dtype x embedding_dim`` rows, and would copy IQ4_NL bytes into a
    bf16 buffer. They are not wired to this table (see
    ``Qwen4ExpPinnedHostEmbedding.attach_gguf_table``).
    """

    is_gguf = True

    def __init__(self, loc: GgufTensorLoc, *, total_rows: Optional[int] = None,
                 embedding_dim: Optional[int] = None, madvise_random: bool = True) -> None:
        import numpy as np

        check_ple_table_supported(loc)
        self.loc = loc
        self.path = loc.path
        self.embedding_dim = int(loc.dim)
        self.total_rows = int(loc.rows)
        if embedding_dim is not None and int(embedding_dim) != self.embedding_dim:
            raise ValueError(
                f"GGUF PLE table has {self.embedding_dim} columns, the embedding expects {embedding_dim}"
            )
        if total_rows is not None and int(total_rows) != self.total_rows:
            raise ValueError(
                f"GGUF PLE table has {self.total_rows} rows, the embedding expects {total_rows} "
                "(padded n-gram vocabulary of the sibling config.json)"
            )
        self.row_bytes = iq4_nl_row_bytes(self.embedding_dim)
        if loc.n_bytes != self.total_rows * self.row_bytes:
            raise ValueError(
                f"GGUF PLE table byte span {loc.n_bytes} != {self.total_rows} rows x {self.row_bytes} B"
            )
        self.shard_rows = self.total_rows
        self.out_dtype = torch.bfloat16
        self.data_offset = int(loc.data_offset)
        # the span only: np.memmap maps lazily, nothing is read here
        self._mm = np.memmap(
            loc.path, dtype=np.uint8, mode="r", offset=self.data_offset, shape=(loc.n_bytes,)
        )
        self.base = int(self._mm.ctypes.data)
        self.bases = (self.base,)
        self._rows_view = self._mm.reshape(self.total_rows, self.row_bytes)
        self._device_bases: dict = {}
        if madvise_random:
            try:
                from sglang.srt.models.qwen4_exp_ple_table import _MADV_RANDOM, _madvise

                # madvise wants a page-aligned address: the span starts mid-page
                # (np.memmap maps from the page below the offset)
                page = mmap.PAGESIZE
                start = self.base - self.base % page
                _madvise(start, int(loc.n_bytes) + (self.base - start), _MADV_RANDOM)
            except Exception:  # noqa: BLE001 - a hint only
                pass

    # ---- interface the kernel launch needs ---------------------------------

    @property
    def kernel_row_elems(self) -> int:
        """Row width in bf16 elements when the 90 bytes are copied as bit
        patterns (45)."""
        return self.row_bytes // 2

    def bases_on(self, device) -> torch.Tensor:
        key = str(device)
        t = self._device_bases.get(key)
        if t is None:
            t = torch.tensor(self.bases, dtype=torch.int64, device=device)
            self._device_bases[key] = t
        return t

    def row_ptr(self, row: int) -> int:
        return self.base + int(row) * self.row_bytes

    # ---- lookup ------------------------------------------------------------

    def read_rows(self, ids) -> "Any":
        """IQ4_NL blocks of rows ``ids``: ``uint8`` numpy ``[n, row_bytes]``.
        Only these rows' pages are read."""
        import numpy as np

        idx = np.asarray(ids, dtype=np.int64).reshape(-1)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.total_rows):
            raise IndexError(f"PLE row id out of 0..{self.total_rows - 1}")
        return self._rows_view[idx]

    def lookup_cpu(self, ids, vocab_start: int = 0, vocab_end: Optional[int] = None) -> torch.Tensor:
        """Reference lookup on the CPU: bf16 ``[n, dim]``; ids outside
        ``[vocab_start, vocab_end)`` give zero rows (the gather kernel's rule)."""
        import numpy as np

        ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
        end = self.total_rows if vocab_end is None else min(int(vocab_end), self.total_rows)
        ok = (ids_np >= int(vocab_start)) & (ids_np < end)
        out = np.zeros((ids_np.size, self.embedding_dim), dtype=np.float32)
        if ok.any():
            out[ok] = dequantize_iq4_nl_reference(self.read_rows(ids_np[ok]), self.embedding_dim)
        return torch.from_numpy(out).to(self.out_dtype)

    # ---- device side -------------------------------------------------------

    def padded_rows(self, n: int) -> int:
        """Smallest ``m >= n`` with ``m * dim`` a multiple of 256 (see the
        module docstring: the sgl-kernel dequantiser's superblock)."""
        m = max(int(n), 1)
        step = 256 // _gcd(256, self.embedding_dim)
        return (m + step - 1) // step * step

    def dequantize_device(self, raw: torch.Tensor, m: int) -> torch.Tensor:
        """``raw`` = uint8 ``[m, row_bytes]`` on the device -> bf16 ``[m, dim]``
        through the existing sgl-kernel ``ggml_dequantize``."""
        if m * self.embedding_dim % 256:
            raise ValueError(
                f"IQ4_NL device dequantisation of {m} rows x {self.embedding_dim} is not a whole "
                "number of 256-element superblocks (the kernel would overrun); pad with padded_rows()"
            )
        if raw.dtype != torch.uint8 or tuple(raw.shape) != (m, self.row_bytes):
            raise ValueError(f"expected uint8 [{m}, {self.row_bytes}], got {raw.dtype} {tuple(raw.shape)}")
        try:
            from sgl_kernel.quantization import ggml_dequantize
        except ImportError as e:  # pragma: no cover - needs the CUDA wheel
            raise RuntimeError(
                "GGUF PLE table: sgl_kernel.quantization.ggml_dequantize is not available "
                "(no CUDA sgl-kernel wheel); the IQ4_NL table cannot be dequantised"
            ) from e
        return ggml_dequantize(raw, IQ4_NL_TYPE, m, self.embedding_dim, self.out_dtype)

    def close(self) -> None:
        self._rows_view = None
        self._mm = None


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def map_ple_table_from_gguf(
    loc: GgufTensorLoc, *, total_rows: int, embedding_dim: int
) -> GgufMappedPleTable:
    """Map the table the marker names; row and column counts are checked against
    the embedding so a mismatched export refuses instead of gathering garbage."""
    table = GgufMappedPleTable(loc, total_rows=total_rows, embedding_dim=embedding_dim)
    logger.info(
        "PLE table: mapped GGUF tensor %s of %s read-only (%.1f GiB, IQ4_NL, %d rows x %d B, "
        "data offset %d) -- no copy; gathered rows are dequantised on the GPU (ggml_dequantize)",
        loc.name,
        os.path.basename(loc.path),
        loc.n_bytes / 2**30,
        table.total_rows,
        table.row_bytes,
        table.data_offset,
    )
    return table
