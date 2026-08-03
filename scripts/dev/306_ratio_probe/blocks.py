"""Byte-plane layouts for the #306 lossless-compression ratio probe.

Every GGUF block layout below is transcribed from the fork's own copy of the
ggml block structs, `sgl-kernel/csrc/quantization/gguf/ggml-common.h`, with the
line numbers cited per entry. `QK_K = 256` (ggml-common.h:4) and
`K_SCALE_SIZE = 12` (ggml-common.h:7).

A "plane" is one field of the block struct, gathered across every block of the
sample: field `f` of block `i` for all `i`, contiguously. This is the semantic
de-interleave the #456 analysis calls the "byte-plane split" -- it separates the
small, low-entropy scale/delta fields from the high-entropy packed quant bulk.

MXFP4 is NOT in this header (the fork's vendored ggml predates it), so its
layout is derived from `gguf.constants.GGML_QUANT_SIZES[MXFP4] == (32, 17)`:
32 elements in 17 bytes = one e8m0 byte scale plus 16 bytes of packed 4-bit
values. That derivation is marked as such in the layout table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockLayout:
    """One ggml block format: its byte size and its field decomposition."""

    name: str
    block_elems: int
    block_bytes: int
    # (field_name, start_byte, end_byte) within one block, in struct order.
    fields: tuple[tuple[str, int, int], ...]
    source: str

    def __post_init__(self) -> None:
        covered = 0
        pos = 0
        for fname, start, end in self.fields:
            if start != pos:
                raise ValueError(f"{self.name}: field {fname} starts at {start}, expected {pos}")
            if end <= start:
                raise ValueError(f"{self.name}: field {fname} is empty")
            covered += end - start
            pos = end
        if covered != self.block_bytes:
            raise ValueError(f"{self.name}: fields cover {covered} of {self.block_bytes} bytes")


# ggml-common.h line numbers are from the fork tree at the probe's base commit.
LAYOUTS: dict[str, BlockLayout] = {
    # typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12];
    #                  half d; } block_q3_K;   -- ggml-common.h:78-83
    "Q3_K": BlockLayout(
        "Q3_K",
        256,
        110,
        (("hmask", 0, 32), ("qs", 32, 96), ("scales", 96, 108), ("d", 108, 110)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:78-83",
    ),
    # typedef struct { half2 dm; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
    #   -- ggml-common.h:87-91  (3 * QK_K / 64 == 12)
    "Q4_K": BlockLayout(
        "Q4_K",
        256,
        144,
        (("dm", 0, 4), ("scales", 4, 16), ("qs", 16, 144)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:87-91",
    ),
    # typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16];
    #                  half d; } block_q6_K;   -- ggml-common.h:104-109
    "Q6_K": BlockLayout(
        "Q6_K",
        256,
        210,
        (("ql", 0, 128), ("qh", 128, 192), ("scales", 192, 208), ("d", 208, 210)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:104-109",
    ),
    # typedef struct { half d; uint16_t qs[32]; uint8_t scales[8]; } block_iq2_xs;
    #   -- ggml-common.h:120-124
    "IQ2_XS": BlockLayout(
        "IQ2_XS",
        256,
        74,
        (("d", 0, 2), ("qs", 2, 66), ("scales", 66, 74)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:120-124",
    ),
    # typedef struct { half d; uint8_t qs[96]; } block_iq3_xxs;
    #   -- ggml-common.h:137-140  (3 * (QK_K / 8) == 96)
    "IQ3_XXS": BlockLayout(
        "IQ3_XXS",
        256,
        98,
        (("d", 0, 2), ("qs", 2, 98)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:137-140",
    ),
    # typedef struct { half d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32];
    #                  uint8_t scales[4]; } block_iq3_s;   -- ggml-common.h:145-151
    "IQ3_S": BlockLayout(
        "IQ3_S",
        256,
        110,
        (("d", 0, 2), ("qs", 2, 66), ("qh", 66, 74), ("signs", 74, 106), ("scales", 106, 110)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:145-151",
    ),
    # typedef struct { half d; uint16_t scales_h; uint8_t scales_l[4];
    #                  uint8_t qs[128]; } block_iq4_xs;   -- ggml-common.h:187-192
    "IQ4_XS": BlockLayout(
        "IQ4_XS",
        256,
        136,
        (("d", 0, 2), ("scales_h", 2, 4), ("scales_l", 4, 8), ("qs", 8, 136)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:187-192",
    ),
    # typedef struct { half d; int8_t qs[32]; } block_q8_0;  -- ggml-common.h:55-58
    "Q8_0": BlockLayout(
        "Q8_0",
        32,
        34,
        (("d", 0, 2), ("qs", 2, 34)),
        "sgl-kernel/csrc/quantization/gguf/ggml-common.h:55-58",
    ),
    # NOT in the fork's ggml-common.h. Derived from
    # gguf.constants.GGML_QUANT_SIZES[MXFP4] == (32, 17): one e8m0 scale byte
    # plus 16 bytes holding 32 packed 4-bit values.
    "MXFP4": BlockLayout(
        "MXFP4",
        32,
        17,
        (("e", 0, 1), ("qs", 1, 17)),
        "DERIVED from gguf.constants.GGML_QUANT_SIZES[MXFP4] == (32, 17)",
    ),
}


def plane_split(buf: bytes, layout: BlockLayout) -> list[tuple[str, bytes]]:
    """Semantic de-interleave: one output plane per block-struct field."""
    a = np.frombuffer(buf, dtype=np.uint8)
    if a.size % layout.block_bytes:
        raise ValueError(f"buffer of {a.size} B is not a whole number of {layout.name} blocks")
    m = a.reshape(-1, layout.block_bytes)
    return [(fname, m[:, start:end].tobytes()) for fname, start, end in layout.fields]


def plane_join(planes: list[tuple[str, bytes]], layout: BlockLayout, n_bytes: int) -> bytes:
    """Inverse of plane_split. Counted in the plane arms' decompress cost."""
    nblocks = n_bytes // layout.block_bytes
    out = np.empty((nblocks, layout.block_bytes), dtype=np.uint8)
    for (fname, start, end), (pname, pbuf) in zip(layout.fields, planes, strict=True):
        if fname != pname:
            raise ValueError(f"plane order mismatch: {fname} != {pname}")
        out[:, start:end] = np.frombuffer(pbuf, dtype=np.uint8).reshape(nblocks, end - start)
    return out.tobytes()


def stride_split(buf: bytes, k: int) -> list[tuple[str, bytes]]:
    """Cheap, format-agnostic de-interleave: byte j of every k-byte group."""
    a = np.frombuffer(buf, dtype=np.uint8)
    if a.size % k:
        raise ValueError(f"buffer of {a.size} B is not a multiple of stride {k}")
    m = a.reshape(-1, k)
    return [(f"s{j}", np.ascontiguousarray(m[:, j]).tobytes()) for j in range(k)]


def stride_join(planes: list[tuple[str, bytes]], k: int, n_bytes: int) -> bytes:
    n = n_bytes // k
    out = np.empty((n, k), dtype=np.uint8)
    for j, (_, pbuf) in enumerate(planes):
        out[:, j] = np.frombuffer(pbuf, dtype=np.uint8)
    return out.tobytes()


def nibble_split(buf: bytes) -> list[tuple[str, bytes]]:
    """High/low nibble planes, packed two-per-byte.

    For 1-byte-element payloads (INT8 weights, FP8-E4M3 weights) there is no
    byte plane to separate, but the fields inside the byte still differ in
    entropy: for E4M3 the high nibble carries sign + 3 exponent bits, the low
    nibble 1 exponent + 3 mantissa bits. This is the byte-plane split's
    sub-byte analogue and costs two shifts plus an or.
    """
    a = np.frombuffer(buf, dtype=np.uint8)
    if a.size % 2:
        raise ValueError("odd-length buffer")
    hi = a >> 4
    lo = a & 0x0F
    return [
        ("hi", ((hi[0::2] << 4) | hi[1::2]).tobytes()),
        ("lo", ((lo[0::2] << 4) | lo[1::2]).tobytes()),
    ]


def nibble_join(planes: list[tuple[str, bytes]], n_bytes: int) -> bytes:
    hi_p = np.frombuffer(planes[0][1], dtype=np.uint8)
    lo_p = np.frombuffer(planes[1][1], dtype=np.uint8)
    out = np.empty(n_bytes, dtype=np.uint8)
    out[0::2] = ((hi_p >> 4) << 4) | (lo_p >> 4)
    out[1::2] = ((hi_p & 0x0F) << 4) | (lo_p & 0x0F)
    return out.tobytes()
