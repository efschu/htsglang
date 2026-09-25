"""fp8 PLE table reads for the Triton PLE gathers (H68d).

The nvidia NVFP4 export of Qwen3.8-Flash-Next stores the PLE n-gram table as
``float8_e4m3fn`` (128 shards [2500012, 160] plus ONE per-tensor
``weight_scale``, applied after the gather). Every PLE gather kernel typed
the table -- and the H40/H69 decode stage -- as ``tl.float8e4nv``, a type
Triton refuses below sm89: the 3080 of the NVFP4 slice smoke died at its
first forward with ``type fp8e4nv not supported in this architecture. The
supported fp8 dtypes are ('fp8e4b15', 'fp8e5')`` (h68b_slice_x172s_card0,
2026-09-24). The INT4 checkpoint's PLE table is bf16, so no boot met it.

Below sm90 the kernels load the table bytes as ``uint8`` and decode them in
the kernel with the QSA decoders of fnFL2 H65 (layers/attention/qsa/
sparse_attn.py, one authority, not a second copy). An e4m3fn value is exact
in bf16, so every decode gives the same bf16 as the native conversion for
all 254 non-NaN codes, with one sign exception: 0x80 comes out +0.0 (the
native path gives -0.0) -- the gather's output is multiplied by the scale and
added into the residual stream, where +0.0 and -0.0 are the same number. The
two NaN codes stay NaN. (sm89 types fp8e4nv, but Triton 3.6's fp8e4nv ->
bf16 conversion needs sm90's ``cvt.bf16.f16``: see NATIVE_FP8_MIN_ARCH.)

``SGLANG_WEG2_PLE_FP8_DECODE`` picks the decode per architecture, grammar
``[smXX:]MODE[;[smYY:]MODE]`` with MODE = native | exp2 | bits | ptx (the
H65 grammar plus ``native``); an arch group wins over a generic one. Unset:
``native`` from sm90 on (the kernels compile to the pre-H68d code), ``bits``
below. ``native`` below sm90 and ``ptx`` below sm80 are refused by name.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _fp8_e4m3_bytes_to_bf16_ptx,
    _fp8_e4m3_bytes_to_f32,
    _fp8_e4m3_bytes_to_f32_bits,
)

logger = logging.getLogger(__name__)

# The FP8_DECODE constexpr of the PLE gathers. 0..2 are the QSA numbers
# (FP8_DECODE_EXP2/BITS/PTX), 3 keeps the native fp8e4nv pointer.
PLE_FP8_EXP2 = 0
PLE_FP8_BITS = 1
PLE_FP8_PTX = 2
PLE_FP8_NATIVE = 3
_MODES = {"exp2": PLE_FP8_EXP2, "bits": PLE_FP8_BITS, "ptx": PLE_FP8_PTX, "native": PLE_FP8_NATIVE}
_NAMES = {v: k for k, v in _MODES.items()}

#: the first architecture on which the native read BUILDS: Triton types
#: fp8e4nv from sm89 on, but its fp8e4nv -> bf16 conversion emits
#: ``cvt.bf16.f16``, which ptxas accepts only from sm_90 on (offline compile
#: of these kernels for sm89 with Triton 3.6: "Feature 'cvt with .bf16.f16'
#: requires .target sm_90 or higher"). The rig runs sm86 and sm120.
NATIVE_FP8_MIN_ARCH = 90


def default_mode(arch: Optional[int]) -> int:
    """native from sm90 on, the bit decode below (and off-GPU, e.g. the
    Triton interpreter, where no architecture exists)."""
    if arch is not None and int(arch) >= NATIVE_FP8_MIN_ARCH:
        return PLE_FP8_NATIVE
    return PLE_FP8_BITS


def parse_ple_fp8_decode(raw: str, arch: Optional[int]) -> int:
    """The FP8_DECODE constexpr ``SGLANG_WEG2_PLE_FP8_DECODE`` names for
    ``arch`` (86, 120, ...; None = no GPU), :func:`default_mode` when it names
    none. Raises ValueError for an unknown mode and for a mode the arch cannot
    compile (native below sm90, ptx below sm80)."""
    generic = chosen = None
    for group in (g.strip() for g in str(raw or "").split(";")):
        if not group:
            continue
        target = None
        if group.startswith("sm") and ":" in group:
            head, group = group.split(":", 1)
            target = int(head[2:])
        mode = group.strip().lower()
        if mode not in _MODES:
            raise ValueError(
                f"SGLANG_WEG2_PLE_FP8_DECODE group {group!r}: mode must be one of "
                f"{sorted(_MODES)}"
            )
        if target is None:
            generic = _MODES[mode]
        elif arch is not None and target == int(arch):
            chosen = _MODES[mode]
    value = chosen if chosen is not None else generic if generic is not None else default_mode(arch)
    if arch is not None:
        if value == PLE_FP8_NATIVE and int(arch) < NATIVE_FP8_MIN_ARCH:
            raise ValueError(
                f"SGLANG_WEG2_PLE_FP8_DECODE={raw!r}: 'native' needs sm{NATIVE_FP8_MIN_ARCH}+ "
                f"(no fp8e4nv below sm89, no fp8e4nv -> bf16 conversion below sm90), "
                f"this device is sm{int(arch)}"
            )
        if value == PLE_FP8_PTX and int(arch) < 80:
            raise ValueError(
                f"SGLANG_WEG2_PLE_FP8_DECODE={raw!r}: 'ptx' needs sm80+ "
                f"(fma.rn.bf16x2), this device is sm{int(arch)}"
            )
    return value


_ARCH_OF: Dict[int, int] = {}
_MODE_CACHE: Dict[Tuple[str, Optional[int]], int] = {}
_NOTED: set = set()


def _arch_of(device) -> Optional[int]:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    if index not in _ARCH_OF:
        major, minor = torch.cuda.get_device_capability(index)
        _ARCH_OF[index] = major * 10 + minor
    return _ARCH_OF[index]


def ple_fp8_decode_for(device) -> int:
    """FP8_DECODE for a gather on ``device``; one ``PLE-FP8-DECODE`` line per
    process and (arch, mode) -- the proof in the rank's own log that the arm's
    choice arrived (an env var in the launcher is not an env var in the rank)."""
    arch = _arch_of(device)
    raw = envs.SGLANG_WEG2_PLE_FP8_DECODE.get() or ""
    key = (raw, arch)
    if key not in _MODE_CACHE:
        _MODE_CACHE[key] = parse_ple_fp8_decode(raw, arch)
    mode = _MODE_CACHE[key]
    if (arch, mode) not in _NOTED:
        _NOTED.add((arch, mode))
        logger.info(
            "PLE-FP8-DECODE arch=%s mode=%s (SGLANG_WEG2_PLE_FP8_DECODE=%r; fp8 PLE "
            "table read %s; one line per arch and mode)",
            f"sm{arch}" if arch is not None else "none",
            _NAMES[mode],
            raw,
            "through the native fp8e4nv pointer" if mode == PLE_FP8_NATIVE
            else "as uint8 bytes, decoded in the kernel",
        )
    return mode


def ple_fp8_decode_arg(table_dtype: torch.dtype, device) -> int:
    """The FP8_DECODE launch argument: the device's mode for an fp8 table,
    PLE_FP8_NATIVE (inert: the bf16 branch never reads it) otherwise."""
    if table_dtype == torch.float8_e4m3fn:
        return ple_fp8_decode_for(device)
    return PLE_FP8_NATIVE


@triton.jit
def ple_fp8_bytes_to_bf16(raw, FP8_DECODE: tl.constexpr):
    """uint8 e4m3fn bytes -> bf16 by the decode FP8_DECODE names (0..2).

    The kernels call it ONLY in their byte branch (``is_fp8 and FP8_DECODE !=
    PLE_FP8_NATIVE``); their bf16 and native branches are the pre-H68d code,
    verbatim -- same PTX, and no nested jit call for Triton's interpreter to
    trip over in the bf16 tests of H40/H69."""
    if FP8_DECODE == 2:
        out = _fp8_e4m3_bytes_to_bf16_ptx(raw)
    elif FP8_DECODE == 1:
        out = _fp8_e4m3_bytes_to_f32_bits(raw).to(tl.bfloat16)
    else:
        out = _fp8_e4m3_bytes_to_f32(raw).to(tl.bfloat16)
    return out
