# SPDX-License-Identifier: Apache-2.0
"""``--fp4-gemm-backend native-mixed``: ONE native NVFP4 byte layout on every
rank, the kernel chosen per rank by its compute capability (Backlog #38).

The contract is docs/NVFP4_NATIVE_LAYOUT_CONTRACT.md. In short: every rank keeps
exactly the tensors ``ModelOptFp4LinearMethod.process_weights_after_loading``
builds for the native sm_120 path -- ``weight`` uint8 [N, K/2] (E2M1, element 2j
in the LOW nibble), ``weight_scale`` float8_e4m3fn [ceil128(N), ceil4(K/16)] in the
128x4 swizzle, the replicated scalars ``weight_scale_2`` / ``input_scale`` /
``alpha`` / ``input_scale_inv`` -- so a flip moves bytes and never reshapes them.
Only the GEMM differs:

* sm_12x (5090)   -> ``cutlass`` (W4A4 on the FP4 tensor cores, the path ``auto``
                     already takes there);
* sm_10x          -> ``flashinfer_cutedsl`` (what ``auto`` takes there);
* sm_8x  (3080)   -> ``marlin_native_inplace`` (default): Marlin W4A16, the
                     measured best on sm_86 (decode 635 vs 233 GB/s; prefill
                     W4A8-g16 74 TOPS barely beats W4A16 55 TF). Marlin wants
                     its own layout, so the CONTENT of this rank's NVFP4
                     parameters is permuted in place -- same byte counts, same
                     Parameter objects and shapes -- and back to native around
                     every flip (nvfp4_marlin_inplace.py). The exchange only
                     ever sees native bytes.
                  -> ``w4a8_int8`` with ``SGLANG_FP4_NATIVE_MIXED_SM8X=w4a8``:
                     the registered W4A8 INT8 kernel (agent N4A's seam, kept).
* sm_12x kernel choice is ONE replaceable function per arch:
  :func:`register_sm12x_apply` lets the FlashInfer-next strand hook its own
  per-call choice (cute-dsl-native W4A16 at small M, W4A4 otherwise) without
  touching this module; unregistered, the rank runs the fork CUTLASS W4A4.

Default OFF: nothing here runs unless ``--fp4-gemm-backend native-mixed`` is
passed; every other backend value resolves exactly as before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

NATIVE_MIXED = "native-mixed"

#: Swizzle tile of the native scale layout: 128 rows x 4 scale columns
#: (= 64 K elements). Every rank shard and every fused component boundary must
#: land on it, or the per-rank tensor needs padding (see check_shard_alignment).
SF_TILE_ROWS = 128
SF_TILE_COLS = 4
SF_GROUP = 16

#: Attribute stamped on the swizzled scale Parameter in native-mixed mode, so the
#: weight exchange can tell a 128x4-swizzled scale from a row-major one by the
#: tensor, never by its name (the name is ``weight_scale`` for both: see the
#: contract, section 2.3).
SF_LAYOUT_ATTR = "nvfp4_sf_layout"
SF_LAYOUT_128X4 = "128x4"


@dataclass(frozen=True)
class RankKernelChoice:
    """The resolved FP4 GEMM backend of ONE rank under native-mixed."""

    backend: str  # a Fp4GemmRunnerBackend value
    capability: Tuple[int, int]
    shared_layout: bool  # the exchange sees the native layout (True on every branch today)
    reason: str


class NativeMixedUnsupported(RuntimeError):
    """native-mixed cannot serve this rank with the shared native layout."""


# ---------------------------------------------------------------------------
# W4A8 kernel seam (filled by N4A's kernel module).
# ---------------------------------------------------------------------------

#: Signature of the W4A8 kernel this seam accepts:
#:   fn(x, weight, weight_scale_swizzled, weight_global_scale, out_features) -> out
#: x                      bf16/fp16 [M, K]  (the kernel quantizes to INT8 per token)
#: weight                 uint8 [N, K/2]    native E2M1, low nibble = even element
#: weight_scale_swizzled  float8_e4m3fn [ceil128(N), ceil4(K/16)], 128x4 swizzle
#: weight_global_scale    float32 0-dim     max(weight_scale_2); bound as the
#:                        Parameter ``weight_global_scale`` on EVERY native-mixed
#:                        rank (uniform parameter set for the exchange)
#: out_features           int               N before any padding
#: out                    x.dtype [M, out_features]
W4A8Kernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]

_W4A8_KERNEL: Optional[W4A8Kernel] = None
_W4A8_KERNEL_NAME: str = ""


def register_w4a8_kernel(fn: W4A8Kernel, name: str) -> None:
    global _W4A8_KERNEL, _W4A8_KERNEL_NAME
    _W4A8_KERNEL = fn
    _W4A8_KERNEL_NAME = str(name)


def unregister_w4a8_kernel() -> None:
    global _W4A8_KERNEL, _W4A8_KERNEL_NAME
    _W4A8_KERNEL = None
    _W4A8_KERNEL_NAME = ""


def w4a8_kernel() -> Optional[W4A8Kernel]:
    return _W4A8_KERNEL


def w4a8_kernel_name() -> str:
    return _W4A8_KERNEL_NAME


def _try_autoload_w4a8_kernel() -> None:
    """Import N4A's kernel module if it exists in this tree; it registers
    itself. Absent module = no kernel, never an error here."""
    if _W4A8_KERNEL is not None:
        return
    try:
        import importlib

        importlib.import_module("sglang.srt.layers.quantization.nvfp4_w4a8_int8")
    except ImportError:
        return


# ---------------------------------------------------------------------------
# sm_12x kernel-choice seam (filled by the FlashInfer-next strand).
# ---------------------------------------------------------------------------

#: fn(layer, x, bias) -> out, or None to fall through to the fork CUTLASS W4A4
#: path. It reads the NATIVE parameters only (weight u8 [N, K/2],
#: weight_scale_interleaved 128x4, alpha, input_scale_inv) -- e.g. FlashInfer
#: ``mm_bf16_fp4(backend="cute-dsl-native")`` for small M, which needs no
#: preparation and no second weight copy. One function per arch: sm_8x has its
#: own (nvfp4_marlin_inplace.apply), sm_12x this one.
Sm12xApply = Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor]], Optional[torch.Tensor]]

_SM12X_APPLY: Optional[Sm12xApply] = None
_SM12X_APPLY_NAME: str = ""


def register_sm12x_apply(fn: Sm12xApply, name: str) -> None:
    global _SM12X_APPLY, _SM12X_APPLY_NAME
    _SM12X_APPLY = fn
    _SM12X_APPLY_NAME = str(name)


def unregister_sm12x_apply() -> None:
    global _SM12X_APPLY, _SM12X_APPLY_NAME
    _SM12X_APPLY = None
    _SM12X_APPLY_NAME = ""


def sm12x_apply() -> Optional[Sm12xApply]:
    return _SM12X_APPLY


def sm12x_apply_name() -> str:
    return _SM12X_APPLY_NAME


def _try_autoload_sm12x_apply() -> None:
    """Import the FlashInfer-next strand's sm_12x choice module if this tree has
    it; it registers itself. Absent module = fork CUTLASS W4A4, never an error."""
    if _SM12X_APPLY is not None:
        return
    try:
        import importlib

        importlib.import_module("sglang.srt.layers.quantization.nvfp4_sm12x_choice")
    except ImportError:
        return


# ---------------------------------------------------------------------------
# Per-rank resolution (pure; the desk tests drive it with a fake capability).
# ---------------------------------------------------------------------------


def resolve_rank_backend(
    capability: Tuple[int, int],
    *,
    w4a8_available: bool,
    sm8x_choice: str = "marlin",
    native_sm120_available: bool = True,
) -> RankKernelChoice:
    """The per-arch table. Every branch keeps the SHARED native layout at the
    exchange; the sm_8x Marlin rank does so by permuting its content in place
    around each flip (nvfp4_marlin_inplace)."""
    major, minor = int(capability[0]), int(capability[1])
    cap = (major, minor)
    if major == 12:
        if not native_sm120_available:
            raise NativeMixedUnsupported(
                f"native-mixed: sm_{major}{minor} rank has no native NVFP4 CUTLASS "
                "kernel (sglang.jit_kernel.nvfp4 not importable); refusing rather "
                "than falling back to a different byte layout."
            )
        return RankKernelChoice("cutlass", cap, True, "sm_12x: native W4A4 FP4 tensor cores")
    if major == 10:
        return RankKernelChoice(
            "flashinfer_cutedsl", cap, True, "sm_10x: native W4A4 (auto's choice there)"
        )
    if major == 8:
        choice = str(sm8x_choice or "marlin").strip().lower()
        if choice == "w4a8":
            if not w4a8_available:
                raise NativeMixedUnsupported(
                    f"native-mixed: SGLANG_FP4_NATIVE_MIXED_SM8X=w4a8 on sm_{major}{minor}, "
                    "but no W4A8 INT8 kernel is registered "
                    "(sglang.srt.layers.quantization.nvfp4_w4a8_int8)."
                )
            return RankKernelChoice(
                "w4a8_int8", cap, True, "sm_8x: W4A8 on INT8 tensor cores, native layout"
            )
        if choice == "marlin":
            return RankKernelChoice(
                "marlin_native_inplace",
                cap,
                True,
                "sm_8x: Marlin W4A16 on the shared native layout (content permuted "
                "in place around every flip, N-bands bound the transient)",
            )
        raise NativeMixedUnsupported(
            f"native-mixed: SGLANG_FP4_NATIVE_MIXED_SM8X={sm8x_choice!r} is neither "
            "'marlin' nor 'w4a8'."
        )
    raise NativeMixedUnsupported(
        f"native-mixed: compute capability {major}.{minor} has neither FP4 nor a "
        "supported W4A16/W4A8 path."
    )


def resolve_this_rank() -> RankKernelChoice:
    """Resolve for the current CUDA device (called once per scheduler process)."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.fp4_utils import has_fork_nvfp4_cutlass_kernel
    from sglang.srt.utils.common import get_device_capability

    _try_autoload_w4a8_kernel()
    _try_autoload_sm12x_apply()
    cap = get_device_capability()
    if cap is None or cap[0] is None:
        raise NativeMixedUnsupported("native-mixed needs a CUDA device.")
    native = True
    if int(cap[0]) == 12:
        native = has_fork_nvfp4_cutlass_kernel()
    choice = resolve_rank_backend(
        (int(cap[0]), int(cap[1])),
        w4a8_available=w4a8_kernel() is not None,
        sm8x_choice=str(envs.SGLANG_FP4_NATIVE_MIXED_SM8X.get()),
        native_sm120_available=native,
    )
    log = logger.info if choice.shared_layout else logger.warning
    log(
        "NVFP4 native-mixed: sm_%d%d -> %s (%s)%s",
        choice.capability[0],
        choice.capability[1],
        choice.backend,
        choice.reason,
        f" [kernel {w4a8_kernel_name()}]" if choice.backend == "w4a8_int8"
        else (
            f" [sm12x hook {sm12x_apply_name()}]"
            if sm12x_apply() is not None and int(cap[0]) == 12
            else ""
        ),
    )
    return choice


# ---------------------------------------------------------------------------
# Layout checks and the exchange view.
# ---------------------------------------------------------------------------


def shard_needs_padding(n_rows: int, k_in: int) -> bool:
    """Would the native path pad this per-rank shard (weight N/K to 32, scale
    rows to 128, scale cols to 4)? Padding means a second scale tensor and a
    rank-local shape that no other rank's slice reproduces."""
    return (
        n_rows % SF_TILE_ROWS != 0
        or (k_in // SF_GROUP) % SF_TILE_COLS != 0
        or k_in % 32 != 0
    )


def check_shard_alignment(name: str, n_rows: int, k_in: int, components=()) -> None:
    """Refuse (by name) a shard the shared layout cannot carry unpadded."""
    if k_in % SF_GROUP:
        raise NativeMixedUnsupported(f"{name}: K={k_in} is not a multiple of 16.")
    if shard_needs_padding(n_rows, k_in):
        raise NativeMixedUnsupported(
            f"native-mixed: {name} shard N={n_rows}, K={k_in} is not aligned to the "
            f"128x4 scale tile (N % 128, (K/16) % 4, K % 32 must all be 0). The native "
            f"path would pad it, keep a second (raw) scale tensor next to the swizzled "
            f"one, and the flip could not move it as a slice of another rank's layout. "
            f"Use a TP ratio whose shards are whole 128-element units."
        )
    for i, c in enumerate(components or ()):
        if int(c) % SF_TILE_ROWS:
            raise NativeMixedUnsupported(
                f"native-mixed: {name} fused component {i} has {c} rows, not a whole "
                f"number of 128-row scale tiles; a swizzle tile would straddle two "
                f"components."
            )


def sf_tile_view_shape(scale: torch.Tensor) -> Tuple[int, int]:
    """(rows, cols) in BYTES of the swizzled scale seen as one row per 128-row
    tile. In this view a K-axis (row-parallel) shard of the swizzled scale IS a
    plain column slice -- columns [kb0*128, (kb0+kb_r)*128) -- and a 128-aligned
    N-axis shard is a plain row slice (contract section 4)."""
    rows, cols = int(scale.shape[-2]), int(scale.shape[-1])
    if rows % SF_TILE_ROWS or cols % SF_TILE_COLS:
        raise NativeMixedUnsupported(
            f"swizzled scale {tuple(scale.shape)} is not a whole number of 128x4 tiles"
        )
    return rows // SF_TILE_ROWS, cols * SF_TILE_ROWS * scale.element_size()


def sf_tile_view(scale: torch.Tensor) -> torch.Tensor:
    r, c = sf_tile_view_shape(scale)
    return scale.contiguous().view(torch.uint8).view(r, c)


def swizzle_128x4(scale: torch.Tensor) -> torch.Tensor:
    """The native swizzle of modelopt_quant.py (pad to 128x4, then
    reshape(M/128,4,32,K/4,4).permute(0,1,4,3,2,5)); CPU-safe, for tests and
    for re-deriving a shard's expected bytes."""
    m, k = scale.shape
    mp = (m + SF_TILE_ROWS - 1) // SF_TILE_ROWS * SF_TILE_ROWS
    kp = (k + SF_TILE_COLS - 1) // SF_TILE_COLS * SF_TILE_COLS
    padded = torch.zeros((mp, kp), dtype=scale.dtype, device=scale.device)
    padded[:m, :k] = scale
    return (
        padded.reshape(mp // 128, 4, 32, kp // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(mp, kp)
    )


#: Stamped IN ADDITION on the swizzled scale of a K-sharded (row-parallel)
#: layer: the weight exchange then describes it in the 128-row tile view
#: (weight_exchange.StorageGeom._nvfp4_sf_tile_view), where a K cut is a plain
#: column slice. N-sharded layers (column-parallel, lm_head) keep the element
#: view: a 128-aligned row cut is already layout-neutral there, and the vocab
#: pad arithmetic of the join counts rows, not tiles.
SF_TILE_VIEW_ATTR = "nvfp4_sf_tile_view"


def mark_swizzled(param: torch.Tensor, *, k_sharded: bool = False) -> None:
    setattr(param, SF_LAYOUT_ATTR, SF_LAYOUT_128X4)
    if k_sharded:
        setattr(param, SF_TILE_VIEW_ATTR, True)


def is_swizzled(param) -> bool:
    return getattr(param, SF_LAYOUT_ATTR, None) == SF_LAYOUT_128X4


# ---------------------------------------------------------------------------
# apply() on a W4A8 rank.
# ---------------------------------------------------------------------------


def apply_w4a8(layer: torch.nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
    fn = w4a8_kernel()
    if fn is None:
        raise NativeMixedUnsupported(
            "native-mixed resolved w4a8_int8 for this rank but no kernel is registered"
        )
    x2 = x.reshape(-1, x.shape[-1])
    out = fn(
        x2,
        layer.weight,
        layer.weight_scale_interleaved,
        layer.weight_global_scale,
        int(layer.output_size_per_partition),
    )
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], int(layer.output_size_per_partition))
