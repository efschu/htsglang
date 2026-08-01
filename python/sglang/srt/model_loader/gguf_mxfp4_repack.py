# SPDX-License-Identifier: Apache-2.0
"""Lossless load-time repack of GGUF MXFP4 (ggml type 39) to Q5_0 (type 6).

#391 blocker 1. The published DeepSeek V4 Flash GGUF stores 45 of its expert
tensors -- every routed ``down`` projection plus layer 26's ``gate``/``up`` --
as MXFP4, 47.8 GiB of the 119.4 GiB file. MXFP4 is in none of the GGUF type
sets and in no dequantize / MMVQ / MMQ kernel case, so those tensors are
unreachable and the checkpoint cannot boot.

The repair does not need a new kernel, because the MXFP4 lattice is a SUBSET of
Q5_0's:

* an MXFP4 block is 32 values in 17 bytes -- byte 0 is an e8m0 exponent, bytes
  1..16 hold 4-bit codes in llama.cpp's split-half order (element ``j`` is the
  low nibble of byte ``j``, element ``j + 16`` the high nibble). The value is
  ``kvalues_mxfp4[code] * 2**(e - 128)``, where ``kvalues_mxfp4`` is the E2M1
  lattice DOUBLED so it fits in int8: ``0, 1, 2, 3, 4, 6, 8, 12`` and negations.
* a Q5_0 block is 32 values in 22 bytes -- an fp16 scale ``d``, a 4-byte
  high-bit mask and 16 bytes of low nibbles, in the same split-half order. The
  value is ``(q5 - 16) * d`` with ``q5`` a 5-bit unsigned code.

Every doubled-E2M1 code lies in ``[-12, 12]``, i.e. inside Q5_0's ``[-16, 15]``
integer range, so ``q5 = kvalues_mxfp4[code] + 16`` and ``d = 2**(e - 128)``
reproduce each element EXACTLY -- not approximately, not within a tolerance.
The one place the mapping can fail is the scale: Q5_0 stores ``d`` as fp16,
which holds ``2**e`` exactly only for ``e`` in ``[-24, 15]``, while e8m0 spans
``[-127, 127]``. A block outside that window is refused by name rather than
silently rounded (:func:`_blocks_mxfp4_to_q5_0`).

The price is bytes: 22 per block instead of 17, a factor 22/17 = 1.294 on the
repacked tensors only. That is real RAM and real VRAM, so the plan is logged
once at load time (:func:`log_gguf_repack_plan`).

Placement: the repack runs inside ``weight_utils.gguf_quant_weights_iterator``,
at the two points where that iterator reads a tensor's type and payload. Every
consumer -- dense linear, MoE experts (before and after the per-expert split of
the stacked ``ffn_*_exps`` tensors), embedding -- therefore sees Q5_0 and
nothing downstream of the iterator knows MXFP4 existed. Q5_0 blocks are
self-contained and the split runs on whole blocks, so a per-expert slice of a
repacked tensor is a valid Q5_0 payload on its own.

``SGLANG_GGUF_MXFP4_REPACK=0`` turns the repack off. That restores the loud
refusal (the family adapter's executability gate, and this module's own backstop
for families without one) -- never a silent fallback to an unexecutable type.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: llama.cpp's ``kvalues_mxfp4`` (ggml-common.h), indexed by the raw 4-bit code:
#: the E2M1 lattice doubled so it fits in int8. The doubling is paid back by
#: halving the scale to ``2**(e - 128)``.
_MXFP4_KVALUES = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8
)

_BLOCK_SIZE = 32
_MXFP4_TYPE_SIZE = 17
_Q5_0_TYPE_SIZE = 22

#: fp16 represents ``2**e`` exactly from the smallest subnormal (``2**-24``) up
#: to ``2**15``. Outside that window a Q5_0 scale cannot carry an e8m0 exponent.
_FP16_MIN_EXP = -24
_FP16_MAX_EXP = 15

_ENV_VAR = "SGLANG_GGUF_MXFP4_REPACK"


def repack_enabled() -> bool:
    """Whether the MXFP4 -> Q5_0 repack is active (default: yes)."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_GGUF_MXFP4_REPACK.get())


def _type_map() -> Dict[Any, Any]:
    """``{source ggml type: target ggml type}`` this module can repack."""
    import gguf

    return {gguf.GGMLQuantizationType.MXFP4: gguf.GGMLQuantizationType.Q5_0}


def repack_source_types() -> set:
    """GGML types a load-time repack makes executable, as a set of
    ``gguf.GGMLQuantizationType``. Empty when the repack is switched off, so an
    executability gate built on it refuses exactly what it used to."""
    if not repack_enabled():
        return set()
    return set(_type_map())


def _refuse(tensor_name: str, type_name: str) -> "RuntimeError":
    return RuntimeError(
        f"GGUF tensor {tensor_name!r} is {type_name}, which no GGUF kernel in "
        f"this build dispatches on, and {_ENV_VAR}=0 disabled the lossless "
        f"load-time repack that would make it executable. Unset {_ENV_VAR} (or "
        "set it to 1) to repack, or use a quantization without this type. "
        "There is no fallback: loading it as-is would fail inside a CUDA kernel "
        "after the whole file has been read."
    )


def _blocks_mxfp4_to_q5_0(blocks: np.ndarray, tensor_name: str) -> np.ndarray:
    """``(n, 17)`` MXFP4 blocks -> ``(n, 22)`` Q5_0 blocks, value-exact.

    ``value = kvalues_mxfp4[code] * 2**(e - 128)`` maps onto Q5_0's
    ``value = (q5 - 16) * d`` with ``q5 = kvalues_mxfp4[code] + 16`` and
    ``d = 2**(e - 128)``. Both formats store the 32 codes in split-half order,
    so the nibble halves are carried across unchanged and only the extra fifth
    bit has to be extracted into Q5_0's ``qh`` mask.

    A block whose codes are all zero is emitted with ``d = 0``: exact for it
    (every element is zero either way) and it sidesteps the e8m0 sentinel the
    quantizer writes for a fully-zero block, which would otherwise trip the
    range check below for no reason.
    """
    e8m0 = blocks[:, 0]
    packed = blocks[:, 1:]
    # Split-half de-interleave: element j is byte j's low nibble, element j+16
    # its high nibble. Concatenating the two halves yields element order.
    codes = np.concatenate(
        [packed & np.uint8(0x0F), (packed >> np.uint8(4)) & np.uint8(0x0F)], axis=-1
    )
    signed = _MXFP4_KVALUES[codes]
    q5 = (signed.astype(np.int16) + 16).astype(np.uint8)

    exponent: np.ndarray = e8m0.astype(np.int32) - 128
    all_zero = (signed == 0).all(axis=-1)
    unrepresentable = (~all_zero) & (
        (exponent < _FP16_MIN_EXP) | (exponent > _FP16_MAX_EXP)
    )
    if unrepresentable.any():
        first = int(np.flatnonzero(unrepresentable)[0])
        raise ValueError(
            f"GGUF tensor {tensor_name!r}: {int(unrepresentable.sum())} of "
            f"{blocks.shape[0]} MXFP4 blocks carry an e8m0 exponent outside the "
            f"range fp16 holds exactly (2**{_FP16_MIN_EXP} .. 2**{_FP16_MAX_EXP}); "
            f"Q5_0 stores its scale as fp16 and cannot represent them. First "
            f"offender: block {first}, e8m0 byte {int(e8m0[first])} = "
            f"2**{int(exponent[first])}. Repacking it would change the weights, "
            f"so it is refused instead."
        )
    d = np.where(
        all_zero,
        np.float32(0.0),
        np.exp2(np.clip(exponent, _FP16_MIN_EXP, _FP16_MAX_EXP).astype(np.float32)),
    )

    # Low four bits stay in the same split-half layout Q5_0 uses.
    qs = (q5[:, 0:16] & np.uint8(0x0F)) | (
        (q5[:, 16:32] & np.uint8(0x0F)) << np.uint8(4)
    )
    # qh is a 32-bit little-endian mask, bit i = the fifth bit of element i.
    qh = np.packbits(
        (q5 >> np.uint8(4)).reshape(-1, 1, _BLOCK_SIZE), axis=-1, bitorder="little"
    ).reshape(-1, 4)
    # GGUF is little-endian on disk and the reader hands the payload back in
    # host order; spell the scale's byte order out rather than inherit it.
    d_bytes = d.astype(np.dtype("<f2")).view(np.uint8).reshape(-1, 2)
    return np.concatenate([d_bytes, qh, qs], axis=-1)


def repacked_gguf_type(tensor_type: Any, tensor_name: str) -> Any:
    """The GGML type a consumer will see for this tensor.

    Identity for everything the repack does not cover. Raises when the tensor
    needs a repack that the opt-out switched off, so the refusal is loud even
    for a family whose adapter carries no executability gate.
    """
    target = _type_map().get(tensor_type)
    if target is None:
        return tensor_type
    if not repack_enabled():
        raise _refuse(tensor_name, tensor_type.name)
    return target


def repacked_gguf_bytes(
    tensor_type: Any, data: np.ndarray, tensor_name: str
) -> np.ndarray:
    """The block payload a consumer will see for this tensor (or slice of it).

    ``data`` is the raw block bytes with the LAST axis holding whole blocks --
    which is how ``gguf.GGUFReader`` presents a quantized tensor, and which
    survives indexing a stacked expert tensor's leading axis. Any leading axes
    are preserved, so this can be applied to a whole ``ffn_down_exps`` tensor or
    to one expert's slice of it with the same result.
    """
    target = _type_map().get(tensor_type)
    if target is None:
        return data
    if not repack_enabled():
        raise _refuse(tensor_name, tensor_type.name)

    row_bytes = data.shape[-1]
    if row_bytes % _MXFP4_TYPE_SIZE:
        raise ValueError(
            f"GGUF tensor {tensor_name!r}: {row_bytes} bytes on the last axis is "
            f"not a whole number of {_MXFP4_TYPE_SIZE}-byte MXFP4 blocks, so the "
            "row length is not a multiple of the 32-element block size."
        )
    n_blocks = row_bytes // _MXFP4_TYPE_SIZE
    blocks = np.ascontiguousarray(data).reshape(-1, _MXFP4_TYPE_SIZE)
    out = _blocks_mxfp4_to_q5_0(blocks, tensor_name)
    return out.reshape(data.shape[:-1] + (n_blocks * _Q5_0_TYPE_SIZE,))


def log_gguf_repack_plan(tensors: Iterable[Any]) -> None:
    """One line at load time naming the memory the repack costs.

    22/17 is not free: the DeepSeek V4 Flash UD-Q3_K_XL export grows by 14 GiB,
    which lands in host RAM and (for the resident share) in VRAM. Nobody should
    have to derive that from a type table.
    """
    counts: Dict[Any, Tuple[int, int]] = {}
    for tensor in tensors:
        target = _type_map().get(tensor.tensor_type)
        if target is None:
            continue
        n, nbytes = counts.get(tensor.tensor_type, (0, 0))
        counts[tensor.tensor_type] = (n + 1, nbytes + int(tensor.n_bytes))
    if not counts:
        return
    if not repack_enabled():
        # The family adapter's gate normally fires first; this keeps the
        # message from being a surprise when it does not exist.
        logger.warning(
            "GGUF: %s=0 and this checkpoint uses %s. Nothing will be repacked "
            "and the load will be refused.",
            _ENV_VAR,
            ", ".join(sorted(t.name for t in counts)),
        )
        return
    gib = float(1 << 30)
    for source, (n, nbytes) in sorted(counts.items(), key=lambda kv: kv[0].name):
        target = _type_map()[source]
        grown = nbytes * _Q5_0_TYPE_SIZE / _MXFP4_TYPE_SIZE
        logger.info(
            "GGUF %s->%s load-time repack: %d tensor(s), %.2f -> %.2f GiB "
            "(+%.2f GiB, x%.3f). Lossless -- every element dequantizes to the "
            "same fp32 value. Set %s=0 to refuse %s instead.",
            source.name,
            target.name,
            n,
            nbytes / gib,
            grown / gib,
            (grown - nbytes) / gib,
            _Q5_0_TYPE_SIZE / _MXFP4_TYPE_SIZE,
            _ENV_VAR,
            source.name,
        )
