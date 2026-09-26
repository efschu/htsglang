# SPDX-License-Identifier: Apache-2.0
"""Marlin W4A16 on the SHARED native NVFP4 byte layout (Backlog #38, gap L8).

Under ``--fp4-gemm-backend native-mixed`` an sm_8x rank (RTX 3080) runs the
NVFP4 GEMMs on Marlin W4A16 (the measured best on sm_86: decode 635 vs 233 GB/s,
prefill W4A8 barely ahead of W4A16), while the 5090 runs native W4A4. The flip
moves NATIVE bytes (docs/NVFP4_NATIVE_LAYOUT_CONTRACT.md). Marlin wants its own
layout, so this module converts the CONTENT of the 3080's NVFP4 parameters, in
place, between the two layouts:

* native: ``weight`` uint8 [N, K/2] (element 2j in the low nibble),
  ``weight_scale`` float8_e4m3fn [N, K/16] in the 128x4 swizzle;
* Marlin: per N-band of R rows, ``weight`` int32 [K/16, 2R] (gptq_marlin_repack)
  and ``weight_scale`` fp8 [K/16, R] (marlin_permute_scales +
  nvfp4_marlin_process_scales).

EVERY BYTE COUNT IS EQUAL, so the Parameter objects, their shapes, dtypes and
storages never change -- the exchange descriptors, the join, the CUDA graphs and
the coverage census see exactly the native parameter set. Only the content is
permuted. Nothing is duplicated (``kein-dauer-hostram``: one byte exists once).

BANDS, AND WHY. Marlin's weight is [K/16, 2N] row-major over K-tiles; the native
weight is row-major over N. A whole-tensor in-place conversion therefore needs a
full-tensor scratch (up to 636 MB for the P lm_head [248320, 2560] u8 on the last 3080 stage), and
the house law forbids a reserve for it (``keine-korridor-reserve-nie``). So the
tensor is cut into N-bands of R rows (R a multiple of 128, band bytes
<= BAND_MAX_BYTES): the native band is the contiguous byte range
[r0*K/2, r1*K/2), exactly the size of a Marlin tensor with size_n = R, and each
band is converted in place into ITS OWN Marlin tensor. The GEMM runs once per
band and concatenates along N (exact: no reduction crosses N). Transient per
step: one band clone + one K-chunk of intermediates, bounded by
``transient_bound_bytes()`` (<= 32 MiB with the defaults).

State: every converted Parameter carries ``CONTENT_ATTR`` ("native"/"marlin"),
so each conversion is idempotent. The flip hooks (weight_updater):

* before the first pause of a sleep (before the seam digest and the deposit):
  :func:`model_to_native` -- the deposit and the digest read native bytes;
* after the wake, after the seam digest: :func:`model_to_marlin` -- when the
  exchange carried the bytes they are native whatever the stamp said.

Default OFF: nothing here runs unless native-mixed resolved this rank to
``marlin_native_inplace``.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

#: The backend value native-mixed resolves an sm_8x rank to (never a CLI value).
BACKEND = "marlin_native_inplace"

#: Stamp on the ``weight`` and ``weight_scale`` Parameters: which layout the
#: bytes currently hold. Set on the Parameter OBJECT (not via ``.data``).
CONTENT_ATTR = "nvfp4_content"
NATIVE = "native"
MARLIN = "marlin"

#: Layer flag: this linear runs Marlin on the shared native layout.
LAYER_FLAG = "_nvfp4_marlin_inplace"
#: Layer attribute: the band table [(r0, r1), ...] (python ints, shape-derived).
BANDS_ATTR = "_nvfp4_marlin_bands"

#: The processed Marlin global scale. A PARAMETER bound on EVERY native-mixed
#: rank (the value is replicated: max(weight_scale_2) of the layer, processed in
#: params_dtype), so the exchange joins identical parameter sets and carries it
#: as a replicated scalar -- no tensor outside the plan, nothing to recompute.
GSCALE_PARAM = "weight_global_scale_w4a16"

MIB = 1 << 20
_DEF_BAND_MIB = 24
_DEF_CHUNK_MIB = 1

SF_TILE_ROWS = 128
MARLIN_TILE_K = 16
MARLIN_TILE_N = 64


def band_max_bytes() -> int:
    v = os.environ.get("SGLANG_FP4_NATIVE_MIXED_BAND_MIB", "").strip()
    mib = int(v) if v else _DEF_BAND_MIB
    return max(1, mib) * MIB


def chunk_max_bytes() -> int:
    v = os.environ.get("SGLANG_FP4_NATIVE_MIXED_CHUNK_MIB", "").strip()
    mib = int(v) if v else _DEF_CHUNK_MIB
    return max(1, mib) * MIB


def transient_bound_bytes(band_bytes: Optional[int] = None, chunk_bytes: Optional[int] = None) -> int:
    """Upper bound of the conversion's working set per step, stated as what it
    is: the band clone (the source must survive while the band is rewritten)
    plus ~6 chunk-sized intermediates (unpack to one byte per nibble, the
    permutation gather, the repack). The scale pass runs after the weight pass
    of the same band (weight temporaries freed) and is band/8 x ~8."""
    b = band_max_bytes() if band_bytes is None else int(band_bytes)
    c = chunk_max_bytes() if chunk_bytes is None else int(chunk_bytes)
    return b + 6 * c


def band_table(n: int, k: int, max_bytes: Optional[int] = None) -> List[Tuple[int, int]]:
    """N-bands [(r0, r1)] of R rows each: R a multiple of 128 (whole scale tiles,
    whole Marlin 64-column tiles), band weight bytes R*K/2 <= max_bytes, at least
    one 128-row tile per band. A pure function of the shape, so load and every
    flip cut the same bands."""
    if n % SF_TILE_ROWS or k % (4 * MARLIN_TILE_K):
        raise ValueError(
            f"marlin-inplace needs N % 128 == 0 and K % 64 == 0, got N={n}, K={k}"
        )
    mb = band_max_bytes() if max_bytes is None else int(max_bytes)
    row_bytes = k // 2
    tiles = n // SF_TILE_ROWS
    tiles_per_band = max(1, mb // (row_bytes * SF_TILE_ROWS))
    nb = -(-tiles // tiles_per_band)
    # even split in tiles, so the bands are as equal as the tiles allow
    base, extra = divmod(tiles, nb)
    out, r0 = [], 0
    for b in range(nb):
        t = base + (1 if b < extra else 0)
        out.append((r0, r0 + t * SF_TILE_ROWS))
        r0 += t * SF_TILE_ROWS
    assert r0 == n
    return out


def _k_chunk(rows: int, k: int, max_bytes: Optional[int] = None) -> int:
    """K elements per conversion step for a band of ``rows`` rows (multiple of 16)."""
    mb = chunk_max_bytes() if max_bytes is None else int(max_bytes)
    kc = (2 * mb // max(1, rows)) // MARLIN_TILE_K * MARLIN_TILE_K
    return max(MARLIN_TILE_K, min(k, kc))


# ---------------------------------------------------------------------------
# The Marlin weight permutation (pure torch; identical to
# sglang.test.test_marlin_utils.marlin_weights, which the upstream repack test
# pins against gptq_marlin_repack).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _weight_perm_cpu() -> torch.Tensor:
    perm_list: List[int] = []
    for i in range(32):
        perm1: List[int] = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend(p + 256 * j for p in perm1)
    perm = torch.tensor(perm_list, dtype=torch.long)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.long)
    return perm.reshape(-1, 8)[:, interleave].reshape(-1)


@lru_cache(maxsize=None)
def _weight_perms(device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = _weight_perm_cpu()
    inv = torch.argsort(perm)
    return perm.to(device_str), inv.to(device_str)


def native_rows_to_nibbles_kn(w_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [R, Kc/2] native -> uint8 nibbles [Kc, R] (q_w[k, n] of GPTQ)."""
    lo = w_u8 & 0xF
    hi = w_u8 >> 4
    kn = torch.stack((lo, hi), dim=-1).reshape(w_u8.shape[0], -1)  # [R, Kc]
    return kn.t().contiguous()


def nibbles_kn_to_native_rows(q_kn: torch.Tensor) -> torch.Tensor:
    """uint8 nibbles [Kc, R] -> native uint8 [R, Kc/2]."""
    nk = q_kn.t()
    return (nk[:, 0::2] | (nk[:, 1::2] << 4)).contiguous()


def marlin_pack_ref(q_kn: torch.Tensor) -> torch.Tensor:
    """Reference Marlin repack of 4-bit values q_kn [K, N] -> int32 [K/16, 2N]."""
    k, n = q_kn.shape
    perm, _ = _weight_perms(str(q_kn.device))
    t = MARLIN_TILE_K
    q = q_kn.reshape(k // t, t, n // t, t).permute(0, 2, 1, 3).reshape(k // t, n * t)
    q = q.reshape(-1, perm.numel())[:, perm].reshape(k // t, n * t)
    # pack 8 consecutive nibbles into one int32, nibble i at bits 4i:
    # byte b of the little-endian int32 = nibble 2b | nibble 2b+1 << 4
    q = q.to(torch.uint8)
    by = q[:, 0::2] | (q[:, 1::2] << 4)
    return by.contiguous().view(torch.int32)


def marlin_unpack_ref(q_i32: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`marlin_pack_ref`: int32 [T, 2N] -> nibbles uint8 [16T, N]."""
    tk = q_i32.shape[0]
    by = q_i32.contiguous().view(torch.uint8)  # [T, 8N]
    q = torch.stack((by & 0xF, by >> 4), dim=-1).reshape(tk, -1)  # [T, 16N]
    n = q.shape[1] // MARLIN_TILE_K
    _, inv = _weight_perms(str(q.device))
    q = q.reshape(-1, inv.numel())[:, inv].reshape(tk, n * MARLIN_TILE_K)
    t = MARLIN_TILE_K
    return q.reshape(tk, n // t, t, t).permute(0, 2, 1, 3).reshape(tk * t, n)


# ---------------------------------------------------------------------------
# Scales (all exact for non-negative E4M3; the checkpoint's NVFP4 block scales
# are non-negative and the load refuses otherwise).
# ---------------------------------------------------------------------------


def unswizzle_128x4(sw: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of nvfp4_native_mixed.swizzle_128x4 for an unpadded [rows, cols]."""
    return (
        sw.reshape(rows // 128, cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )


def swizzle_128x4_exact(s: torch.Tensor) -> torch.Tensor:
    rows, cols = s.shape
    return (
        s.reshape(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )


@lru_cache(maxsize=None)
def _scale_perms(device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend(i + 8 * j for j in range(8))
    p = torch.tensor(scale_perm, dtype=torch.long)
    return p.to(device_str), torch.argsort(p).to(device_str)


_I4 = [0, 2, 1, 3]  # nvfp4_marlin_process_scales' group-of-4 swap (self-inverse)


def scales_native_to_marlin(sw_band: torch.Tensor, rows: int, kb: int, dtype: torch.dtype) -> torch.Tensor:
    """swizzled e4m3 [R, K/16] -> Marlin fp8 [K/16, R] (bytes), same expressions
    as marlin_utils.marlin_permute_scales + marlin_utils_fp4.nvfp4_marlin_process_scales."""
    s = unswizzle_128x4(sw_band.view(torch.float8_e4m3fn), rows, kb)
    s = s.t().contiguous().to(dtype)  # [K/16, R]
    perm, _ = _scale_perms(str(s.device))
    s = s.reshape(-1, perm.numel())[:, perm].reshape(kb, rows)
    s = s.to(torch.half)
    s = s.view(-1, 4)[:, _I4].view(kb, rows)
    s = (s * (2**7)).view(torch.int16) << 1
    return s.view(torch.float8_e4m3fn)[:, 1::2].contiguous()


def scales_marlin_to_native(m_band: torch.Tensor, rows: int, kb: int) -> torch.Tensor:
    """Inverse: Marlin fp8 [K/16, R] -> swizzled e4m3 [R, K/16]."""
    hb = m_band.view(torch.uint8).reshape(kb, rows).to(torch.int16)
    h = ((hb << 8) >> 1) & 0x7FFF  # arithmetic shift of a positive value
    s = h.to(torch.int16).view(torch.half) / (2**7)
    s = s.view(-1, 4)[:, _I4].view(kb, rows)
    _, inv = _scale_perms(str(s.device))
    s = s.reshape(-1, inv.numel())[:, inv].reshape(kb, rows)
    s = s.to(torch.float8_e4m3fn).t().contiguous()  # [R, K/16], exact
    return swizzle_128x4_exact(s.view(torch.uint8)).view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Band conversion (in place).
# ---------------------------------------------------------------------------

RepackFn = Callable[[torch.Tensor, int, int], torch.Tensor]


def _gpu_repack(q_k8n: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

    perm = torch.empty(0, dtype=torch.int, device=q_k8n.device)
    return gptq_marlin_repack(q_k8n, perm, size_k, size_n, 4)


def _ref_repack(q_k8n: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    by = q_k8n.contiguous().view(torch.uint8)  # [K/8, 4N] -> nibbles along K
    kn = torch.stack((by & 0xF, by >> 4), dim=-1)  # [K/8, 4N, 2]
    # int32 [K/8, N] little-endian: byte b of column n holds k = 8*k8 + 2b (+1)
    kn = kn.reshape(size_k // 8, size_n, 8).permute(0, 2, 1).reshape(size_k, size_n)
    return marlin_pack_ref(kn)


def default_repack() -> RepackFn:
    return _gpu_repack if torch.cuda.is_available() else _ref_repack


def weight_band_to_marlin_(band_u8: torch.Tensor, k: int, repack: Optional[RepackFn] = None) -> None:
    """In place: native uint8 [R, K/2] -> Marlin int32 [K/16, 2R] in the same bytes."""
    repack = repack or default_repack()
    rows = band_u8.shape[0]
    src = band_u8.clone()
    dst = band_u8.view(-1).view(torch.int32).view(k // MARLIN_TILE_K, 2 * rows)
    kc = _k_chunk(rows, k)
    for k0 in range(0, k, kc):
        k1 = min(k, k0 + kc)
        # GPTQ packing along K: int32 [Kc/8, R], nibble i = element 8*k8 + i
        q = src[:, k0 // 2 : k1 // 2].contiguous().view(torch.int32).t().contiguous()
        dst[k0 // MARLIN_TILE_K : k1 // MARLIN_TILE_K].copy_(repack(q, k1 - k0, rows))
        del q
    del src


def weight_band_to_native_(band_u8: torch.Tensor, k: int) -> None:
    """In place: Marlin int32 [K/16, 2R] -> native uint8 [R, K/2] in the same bytes."""
    rows = band_u8.shape[0]
    src = band_u8.view(-1).clone().view(torch.int32).view(k // MARLIN_TILE_K, 2 * rows)
    kc = _k_chunk(rows, k)
    for k0 in range(0, k, kc):
        k1 = min(k, k0 + kc)
        q_kn = marlin_unpack_ref(src[k0 // MARLIN_TILE_K : k1 // MARLIN_TILE_K])
        band_u8[:, k0 // 2 : k1 // 2].copy_(nibbles_kn_to_native_rows(q_kn))
        del q_kn
    del src


def scale_band_to_marlin_(band: torch.Tensor, k: int, dtype: torch.dtype) -> None:
    rows, kb = band.shape[0], k // 16
    m = scales_native_to_marlin(band.clone(), rows, kb, dtype)
    band.view(torch.uint8).view(-1).copy_(m.view(torch.uint8).view(-1))


def scale_band_to_native_(band: torch.Tensor, k: int) -> None:
    rows, kb = band.shape[0], k // 16
    s = scales_marlin_to_native(band.view(torch.uint8).view(-1).clone().view(torch.float8_e4m3fn), rows, kb)
    band.view(torch.uint8).copy_(s.view(torch.uint8))


# ---------------------------------------------------------------------------
# Layer level.
# ---------------------------------------------------------------------------


def _content(p) -> str:
    return getattr(p, CONTENT_ATTR, NATIVE)


def _stamp(p, value: str) -> None:
    setattr(p, CONTENT_ATTR, value)


def layer_dims(layer) -> Tuple[int, int]:
    n = int(layer.weight.shape[0])
    k = int(layer.weight.shape[1]) * 2
    return n, k


def layer_to_marlin(layer, *, repack: Optional[RepackFn] = None) -> int:
    """Native -> Marlin content for one flagged layer; returns bytes converted."""
    w, s = layer.weight, layer.weight_scale
    if _content(w) == MARLIN and _content(s) == MARLIN:
        return 0
    n, k = layer_dims(layer)
    dtype = getattr(layer, "params_dtype", None) or torch.bfloat16
    with torch.no_grad():
        for r0, r1 in getattr(layer, BANDS_ATTR):
            if _content(w) != MARLIN:
                weight_band_to_marlin_(w.data[r0:r1], k, repack)
            if _content(s) != MARLIN:
                scale_band_to_marlin_(s.data[r0:r1], k, dtype)
    _stamp(w, MARLIN)
    _stamp(s, MARLIN)
    return n * k // 2 + n * k // 16


def layer_to_native(layer) -> int:
    w, s = layer.weight, layer.weight_scale
    if _content(w) == NATIVE and _content(s) == NATIVE:
        return 0
    n, k = layer_dims(layer)
    with torch.no_grad():
        for r0, r1 in getattr(layer, BANDS_ATTR):
            if _content(w) != NATIVE:
                weight_band_to_native_(w.data[r0:r1], k)
            if _content(s) != NATIVE:
                scale_band_to_native_(s.data[r0:r1], k)
    _stamp(w, NATIVE)
    _stamp(s, NATIVE)
    return n * k // 2 + n * k // 16


def flagged_layers(models: Iterable, seen: Optional[set] = None) -> List[torch.nn.Module]:
    """Every flagged layer reachable from ``models``, each ONCE (by ``id``).

    ``seen``: a caller-owned id set shared across SEVERAL calls. A layer that
    two models hold (the DFlash2 draft's ``lm_head`` IS the target's module,
    dflash_worker_v2 ``self.draft_model.lm_head = lm_head``) must be converted
    once per flip; a per-model loop without a shared set converted it twice
    (rc9meas n4old 26.09.: 149 layers to native at sleep, 129 + 21 = 150 back
    to Marlin at wake -> the shared lm_head permuted twice, needle MISS)."""
    out = []
    seen = set() if seen is None else seen
    for m in models or ():
        walk = getattr(m, "modules", None)
        if walk is None:
            continue
        for mod in walk():
            if getattr(mod, LAYER_FLAG, False) and id(mod) not in seen:
                seen.add(id(mod))
                out.append(mod)
    return out


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def model_to_native(models: Sequence, *, log: Optional[Callable[[str], None]] = None) -> int:
    """Flip hook, SLEEP side: before the seam digest and the first deposit."""
    layers = flagged_layers(models)
    if not layers:
        return 0
    t0 = time.perf_counter()
    nbytes = sum(layer_to_native(l) for l in layers)
    _sync()
    (log or logger.info)(
        "NVFP4-MARLIN-INPLACE to_native layers=%d bytes=%.1fMiB ms=%.1f "
        "transient_bound=%.1fMiB" % (
            len(layers), nbytes / MIB, (time.perf_counter() - t0) * 1000,
            transient_bound_bytes() / MIB)
    )
    return nbytes


def model_to_marlin(
    models: Sequence,
    *,
    delivered_native: bool,
    log: Optional[Callable[[str], None]] = None,
    repack: Optional[RepackFn] = None,
    seen: Optional[set] = None,
) -> int:
    """Flip hook, WAKE side: after the seam digest, before any forward.

    ``delivered_native``: the exchange wrote this wake's bytes, so every
    flagged parameter holds native content whatever its stamp said (the stamp
    from the last sleep, or from the load). Otherwise (TMS backup restore) the
    stamp set before the pause is the truth.

    ``seen``: pass ONE set to every call of the same wake (see
    :func:`flagged_layers`). Without it a layer shared by two models is
    re-stamped native by the second call AFTER the first converted it, and
    permuted a second time."""
    layers = flagged_layers(models, seen)
    if not layers:
        return 0
    if delivered_native:
        for l in layers:
            _stamp(l.weight, NATIVE)
            _stamp(l.weight_scale, NATIVE)
    t0 = time.perf_counter()
    nbytes = sum(layer_to_marlin(l, repack=repack) for l in layers)
    _sync()
    (log or logger.info)(
        "NVFP4-MARLIN-INPLACE to_marlin layers=%d bytes=%.1fMiB ms=%.1f "
        "delivered_native=%s transient_bound=%.1fMiB" % (
            len(layers), nbytes / MIB, (time.perf_counter() - t0) * 1000,
            bool(delivered_native), transient_bound_bytes() / MIB)
    )
    return nbytes


# ---------------------------------------------------------------------------
# Load and apply.
# ---------------------------------------------------------------------------


def processed_global_scale(weight_global_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
    )

    return nvfp4_marlin_process_global_scale(weight_global_scale.to(dtype)).reshape(1)


def bind_global_scale(layer, dtype: torch.dtype) -> None:
    """Bind GSCALE_PARAM on EVERY native-mixed rank (uniform parameter set)."""
    from sglang.srt.layers.utils.common import copy_or_rebind_param

    copy_or_rebind_param(
        layer, GSCALE_PARAM, processed_global_scale(layer.weight_global_scale, dtype)
    )


def prepare_layer(layer, *, repack: Optional[RepackFn] = None, make_workspace: bool = True) -> None:
    """After the native load (swizzle done, stamps set): flag the layer, cut its
    bands, give it a Marlin workspace (local scratch, zeroed at every wake) and
    convert the content to Marlin."""
    n, k = layer_dims(layer)
    if not (layer.weight_scale is getattr(layer, "weight_scale_interleaved", layer.weight_scale)):
        raise RuntimeError(
            "marlin-inplace: weight_scale_interleaved is not weight_scale "
            "(a padded native load); native-mixed refuses padding, so this is a bug"
        )
    if tuple(layer.weight_scale.shape) != (n, k // 16):
        raise RuntimeError(
            f"marlin-inplace: scale {tuple(layer.weight_scale.shape)} is not [N, K/16] = {(n, k // 16)}"
        )
    if (layer.weight_scale.view(torch.uint8) & 0x80).any():
        raise RuntimeError(
            "marlin-inplace: negative NVFP4 block scales; the Marlin scale format "
            "drops the sign, so the flip round trip would not be exact. Refusing."
        )
    if getattr(layer, "params_dtype", torch.bfloat16) == torch.bfloat16:
        # vLLM #34694 (V50 25.09.): Marlin's dequant_fp8_scales<bf16> widens an
        # E4M3 SUBNORMAL block scale (exponent bits 0, value < 2^-6) to ~2^-112,
        # zeroing that block. The kernel is shared with upstream and unfixed; the
        # upstream fix (#34577) rescales the stored scales, which would change
        # the shared native bytes. Our 27B checkpoints have none (min scale 0.94);
        # name it loudly for any other checkpoint instead of refusing a model
        # over a few near-zero blocks.
        u8 = layer.weight_scale.view(torch.uint8)
        n_sub = int(((u8 & 0x78) == 0).logical_and((u8 & 0x07) != 0).sum())
        if n_sub:
            logger.warning(
                "marlin-inplace: %d subnormal E4M3 block scales (< 2^-6) in a bf16 "
                "layer [%d x %d]; Marlin bf16 dequant reads them as ~0 (vLLM #34694), "
                "those blocks contribute nothing.",
                n_sub, n, k,
            )
    setattr(layer, BANDS_ATTR, band_table(n, k))
    setattr(layer, LAYER_FLAG, True)
    if make_workspace and getattr(layer, "workspace", None) is None:
        # Once per layer: a reload (draft disk refill) keeps the workspace a
        # captured CUDA graph already addresses.
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        layer.workspace = marlin_make_workspace(layer.weight.device)
    _stamp(layer.weight, NATIVE)
    _stamp(layer.weight_scale, NATIVE)
    layer_to_marlin(layer, repack=repack)


def apply(layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None, *, gemm=None) -> torch.Tensor:
    """Marlin W4A16, once per band, concatenated along N."""
    if gemm is None:
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear as gemm,
        )
    n, k = layer_dims(layer)
    w_u8 = layer.weight
    s = layer.weight_scale
    gs = getattr(layer, GSCALE_PARAM)
    outs = []
    for r0, r1 in getattr(layer, BANDS_ATTR):
        rows = r1 - r0
        outs.append(
            gemm(
                input=x,
                weight=w_u8[r0:r1].view(-1).view(torch.int32).view(k // MARLIN_TILE_K, 2 * rows),
                weight_scale=s[r0:r1].view(torch.uint8).view(-1).view(torch.float8_e4m3fn).view(k // 16, rows),
                weight_global_scale=gs,
                workspace=layer.workspace,
                size_n=rows,
                size_k=k,
                bias=None,
            )
        )
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
    n_out = int(getattr(layer, "output_size_per_partition", n))
    if n_out != n:
        out = out[..., :n_out]
    if bias is not None:
        out = out + bias
    return out
