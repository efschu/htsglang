"""compressed-tensors MoE, int4 expert weights x int8 activations (W4A8), H88-B.

Sits next to ``compressed_tensors_wNa16_moe.py`` (the W4A16 Marlin scheme, left
untouched) and is selected by ``CompressedTensorsConfig.get_moe_scheme`` only
behind the switch ``moe_act_int8_requested()`` (layers/quantization/moe_act_int8.py,
H88-E: flag ``--moe-act-int8 on`` OR env ``SGLANG_MOE_ACT_INT8=1``, default OFF) on CUDA. With the switch off nothing in
this module runs.

What it does differently from ``CompressedTensorsWNA16MoE`` (whose
``create_weights`` it inherits: same checkpoint tensors, same loader contract,
``is_transposed=True``):

* the expert weights are repacked to the Marlin W4A8 layout (32x32 tiles, nibble
  order of the int8 dequant) with ``gptq_marlin_moe_repack_w4a8`` (H88-A);
* the group scales are permuted with the "single" permutation and converted to
  ``int16 = round(s / s.max() * 4096)`` (bit pattern viewed as the model dtype);
  ``s.max() / 4096`` is kept as a float32 0-d tensor per layer tensor
  (``layer.w13_act_scale_factor`` / ``layer.w2_act_scale_factor``) and is
  multiplied into the per-token activation scale at run time. Channelwise scales
  (one group) stay plain and have no factor;
* zero points (asymmetric checkpoints; prepared, no NF checkpoint uses them) are
  permuted without the column interleave of the A16 layout;
* ``apply_weights`` quantises the activations per token to int8 and runs
  ``fused_marlin_moe_w4a8``.

H88-C (loader / offload / store / flip) -- what the scheme does with the rows other processes share:

* the layer's tensors are exactly the A16 names (``w13_weight_packed`` ... ``*_weight_scale``, zero points): the
  offload cache stages them with the unchanged ``EXPERT_TENSOR_ATTRS``; the scheme verifies that against
  ``moe_w4a8_layout`` and only then marks the layer ``_moe_offload_w4a8_covered`` (the offload guard admits
  this scheme on such a layer only);
* the two per-tensor factors are registered non-persistent BUFFERS (not plain attributes): the flip exchange books a
  buffer as static state and carries it across sleep/wake in its own process, and refuses an unregistered attribute
  tensor as UNCOVERED (W84);
* a store row is a function of (expert, factor) in this layout. Where processes share rows (the expert store, the
  D-store-adopt veto, the flip exchange between P and D) the factor of a (layer, tensor) is agreed through the store
  (``expert_store.claim_scale_factor``: first converting rank publishes, everybody else adopts); a rank that sees only
  a subset of the rows (vetoed rows / h2d cut) or holds placeholder weights can only ADOPT. A two-group boot without a
  store directory has no agreement channel and is refused by name. The factor is checked against the int16 range of
  this rank's own scales (:data:`W4A8_MAX_INT_BAND`), never clipped silently;
* the layout tag goes into the store identity (``expert_store.compute_identity(layout=...)``, launcher) and P and D
  must run the same layout (``moe_w4a8_layout.check_one_layout_per_boot``).

Still refused (loud, at load time): act-order (g_idx), expert parallelism / a2a backends (the kernel has no -1 expert
ids), group sizes other than -1/32/64/128, K/N not divisible by 64.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
    _rang_karte,
)

if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs

    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

__all__ = [
    "CompressedTensorsWNA16A8MoE",
    "W4A8FactorOutOfBand",
    "W4A8_MAX_INT_BAND",
    "w4a8_scale_proposal",
    "validate_w4a8_moe_dims",
    "w4a8_process_moe_scales",
    "w4a8_moe_zero_points",
    "w4a8_repack_moe_weights",
]

logger = logging.getLogger(__name__)

#: group sizes the int8 kernels are built for (-1 = channelwise)
W4A8_GROUP_SIZES = (-1, 32, 64, 128)
#: kernel thread tile: K and N of every GEMM must be a multiple of this
#: (marlin_a8: min_thread_k = min_thread_n = 64)
W4A8_DIM_MULTIPLE = 64

MARKER = "MOE-ACT-INT8"

#: the int16 scale of a group is ``round(s / factor)``. With the factor ``max(s) / 4096`` of one rank's own view the
#: largest scale is 4096 (8x headroom to int16). A factor adopted from another rank (H88-C: the first converting rank
#: of the store publishes) maps THIS rank's largest scale to ``max(s) / factor``: it must stay inside this band --
#: above the upper end the int16 saturates (silent clipping), below the lower end fewer than ~6 bits of the largest
#: scale survive. Outside the band the rank REFUSES by name (:class:`W4A8FactorOutOfBand`).
W4A8_MAX_INT_BAND = (64, 32767)


class W4A8FactorOutOfBand(RuntimeError):
    """The adopted scale factor does not fit this rank's own scales (saturation or too coarse)."""


# ---------------------------------------------------------------------------
# boot marker
# ---------------------------------------------------------------------------

_MARK = {"layers": 0, "groups": set(), "lock": threading.Lock()}


def _mark_scheme_built(group_size: int) -> Tuple[int, str]:
    """Count one W4A8 MoE layer scheme; returns (layers so far, groups text)."""
    with _MARK["lock"]:
        _MARK["layers"] += 1
        _MARK["groups"].add(int(group_size))
        return _MARK["layers"], ",".join(str(g) for g in sorted(_MARK["groups"]))


def _reset_marker_for_tests() -> None:
    with _MARK["lock"]:
        _MARK["layers"] = 0
        _MARK["groups"].clear()


# ---------------------------------------------------------------------------
# pure helpers (CPU-testable; the kernel-bound repack is injected)
# ---------------------------------------------------------------------------


def validate_w4a8_moe_dims(
    hidden_size: int, intermediate_size: int, group_size: int
) -> None:
    """The shape contract of the two grouped GEMMs of one MoE layer.

    GEMM 1: K = hidden, N = 2 * intermediate; GEMM 2: K = intermediate,
    N = hidden. The kernel picks a thread tile (k, n) in {(128,128), (64,128),
    (128,64), (64,256)} that divides the problem, so K and N must be multiples of
    64; a group size must divide K."""
    if group_size not in W4A8_GROUP_SIZES:
        raise ValueError(
            f"{MARKER}: group_size {group_size} has no int8 kernel "
            f"(supported: {W4A8_GROUP_SIZES})"
        )
    dims = {
        "hidden (K of GEMM 1, N of GEMM 2)": hidden_size,
        "2*intermediate (N of GEMM 1)": 2 * intermediate_size,
        "intermediate (K of GEMM 2)": intermediate_size,
    }
    for name, v in dims.items():
        if v % W4A8_DIM_MULTIPLE != 0:
            raise ValueError(
                f"{MARKER}: {name} = {v} is not a multiple of {W4A8_DIM_MULTIPLE} "
                f"(hidden={hidden_size}, intermediate per partition={intermediate_size}); "
                f"the Marlin W4A8 kernel needs it"
            )
    if group_size != -1:
        for name, k in (("hidden", hidden_size), ("intermediate", intermediate_size)):
            if k % group_size != 0:
                raise ValueError(
                    f"{MARKER}: {name} = {k} is not divisible by group_size {group_size}"
                )


def w4a8_scale_proposal(s: torch.Tensor, rows=None) -> Optional[float]:
    """The factor a rank that sees ``rows`` of the group scales ``s`` [E, G, N] would choose: ``max / 4096`` (exactly
    representable in float32 -- a power-of-two quotient). None for channelwise scales (G == 1: no factor) and when the
    viewed rows are all zero (nothing to scale)."""
    from sglang.jit_kernel import marlin_w4a8_utils as U

    E, G, N = s.shape
    if G == 1 or E == 0:
        return None
    view = s if rows is None else s[torch.as_tensor(list(rows), dtype=torch.long, device=s.device)]
    if view.numel() == 0:
        return None
    smax = float(view.float().abs().max())
    if smax == 0.0 or smax != smax:
        return None
    return smax / U.W4A8_SCALE_INT_RANGE


def w4a8_process_moe_scales(
    s: torch.Tensor,
    *,
    factor: Optional[float] = None,
    rows=None,
    what: str = "",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Group scales [E, G, N] (model dtype) -> kernel scales.

    Permutation "single" (8-bit activations use it for every group size), then,
    for G > 1, the int16 x4096 trick over the WHOLE tensor (one factor per layer
    tensor, all experts, as vLLM). Returns (scales [E, G, N] in s.dtype -- for
    G > 1 the int16 bit pattern --, factor float32 0-d or None for G == 1).

    ``factor=None`` (H88-B, one process sees the whole tensor): the factor is ``max / 4096`` of ``s`` itself.
    ``factor=<float>`` (H88-C, a factor agreed with other processes through the store): the int16 is
    ``round(s / factor)``; the rows in ``rows`` (None = all) are the ones this rank actually holds -- the others are
    never read and are ignored by the range check; the largest viewed scale must map into
    :data:`W4A8_MAX_INT_BAND`, else :class:`W4A8FactorOutOfBand`."""
    from sglang.jit_kernel import marlin_w4a8_utils as U

    E, G, N = s.shape
    if E == 0:
        return torch.empty_like(s), None
    flat = s.reshape(E * G, N)
    # size_k is only used by the A16 branch of the permutation; the a8 branch
    # ignores it and group_size only has to differ from "channelwise" cases
    perm = U.marlin_permute_scales(
        flat, size_k=G, size_n=N, group_size=-1 if G == 1 else 1, is_a_8bit=True
    ).reshape(E, G, N)
    if G == 1:
        return perm.contiguous(), None
    if factor is None:
        if float(perm.float().abs().max()) == 0.0:
            # all-zero scales (e.g. a layer of pad experts): nothing to scale, avoid 0/0
            return torch.zeros_like(perm), torch.ones((), dtype=torch.float32, device=s.device)
        out, fac = U.marlin_act_int8_process_scales(perm)
        return out.contiguous(), fac.to(torch.float32).reshape(())
    fac32 = torch.tensor(float(factor), dtype=torch.float32, device=s.device)
    if not (float(fac32) > 0.0):
        raise W4A8FactorOutOfBand(f"{MARKER}: the scale factor {factor!r} of {what or 'a layer tensor'} is not positive")
    view = perm if rows is None else perm[torch.as_tensor(list(rows), dtype=torch.long, device=s.device)]
    vmax = float(view.float().abs().max()) if view.numel() else 0.0
    if vmax > 0.0:
        max_int = vmax / float(fac32)
        lo, hi = W4A8_MAX_INT_BAND
        if not (lo <= max_int <= hi):
            raise W4A8FactorOutOfBand(
                f"{MARKER} FACTOR OUT OF BAND: {what or 'a layer tensor'} -- the agreed scale factor {float(fac32):.6g} "
                f"maps this rank's largest group scale {vmax:.6g} to {max_int:.1f}, outside the int16 band "
                f"[{lo}, {hi}] ("
                + ("above: the int16 would saturate and clip scales silently" if max_int > hi else
                   "below: fewer than ~6 bits of the largest scale would survive")
                + "). The factor came from another rank's view of the same tensor; remove the factor sidecars "
                f"(*.factor.json in the expert store directory) and boot again with the group that holds the "
                f"whole layer (P) converting first."
            )
    out = (perm.float() / fac32).round().clamp_(-32768, 32767).to(torch.int16).view(s.dtype)
    return out.contiguous(), fac32.reshape(())


def w4a8_moe_zero_points(q_zp_packed: torch.Tensor, size_n: int) -> torch.Tensor:
    """AWQ-style packed zero points [E, G, N/8] int32 -> kernel layout.

    unpack, undo the AWQ column interleave, apply the scale permutation, repack
    WITHOUT the A16 column interleave (the int8 dequant reads nibbles in natural
    order). Pure torch on the tensor's own device; matches
    ``marlin_w4a8_utils.marlin_zero_points`` (tested)."""
    import numpy

    from sglang.srt.layers.quantization.marlin_utils import (
        _pack_cols_torch,
        _unpack_cols_torch,
        get_scale_perms,
    )

    E, G, W = q_zp_packed.shape
    if E == 0:
        return torch.empty_like(q_zp_packed)
    assert W * 8 == size_n, f"zero-point width {W}*8 != size_n {size_n}"
    dev = q_zp_packed.device
    flat = q_zp_packed.reshape(E * G, W)
    q = _unpack_cols_torch(flat, 4, E * G, size_n)
    undo = torch.tensor(numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7])), device=dev)
    q = q.reshape((-1, undo.numel()))[:, undo].reshape((-1, size_n)).contiguous()
    scale_perm, _ = get_scale_perms()
    perm = torch.tensor(scale_perm, device=dev)
    zp = q.reshape((-1, perm.numel()))[:, perm].reshape((-1, size_n)).contiguous()
    out = _pack_cols_torch(zp, 4, E * G, size_n)
    return out.reshape(E, G, W).contiguous()


def w4a8_repack_moe_weights(
    packed: torch.Tensor,
    packed_factor: int,
    num_bits: int,
    repack_fn: Optional[Callable] = None,
    rows=None,
) -> torch.Tensor:
    """GPTQ/CT-packed expert stack int32 [E, K/8, N] -> Marlin W4A8 layout
    int32 [E, K/16, N*16/8] (``gptq_marlin_moe_repack_w4a8``; ``repack_fn`` lets
    a test inject the CPU reference).

    ``rows`` (H88-C, ``store_adopt.repack_rows``): only these expert rows are repacked, the others are left as
    ``torch.empty`` leaves them (rows nobody reads: the D-store-adopt veto, same contract as
    ``gptq_marlin_moe_repack(rows=...)`` of the A16 layout). None = all."""
    if repack_fn is None:
        from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_moe_repack_w4a8

        repack_fn = gptq_marlin_moe_repack_w4a8
    size_k = packed.shape[1] * packed_factor
    size_n = packed.shape[2]
    if rows is None:
        return repack_fn(packed, size_k, size_n, num_bits)
    rows = [int(r) for r in rows]
    out = None
    chunk = 32  # bound the gathered copy; the kernel loops over the leading dimension anyway
    for i in range(0, len(rows), chunk):
        sel = torch.as_tensor(rows[i : i + chunk], dtype=torch.long, device=packed.device)
        sub = repack_fn(packed.index_select(0, sel), size_k, size_n, num_bits)
        if out is None:
            out = torch.empty((packed.shape[0],) + tuple(sub.shape[1:]), dtype=sub.dtype, device=sub.device)
        out[sel] = sub
    if out is None:  # no rows at all: the shape contract still holds
        probe = repack_fn(packed[:0], size_k, size_n, num_bits)
        out = torch.empty((packed.shape[0],) + tuple(probe.shape[1:]), dtype=probe.dtype, device=probe.device)
    return out


# ---------------------------------------------------------------------------
# the scheme
# ---------------------------------------------------------------------------


class CompressedTensorsWNA16A8MoE(CompressedTensorsWNA16MoE):
    """int4 (g32/g64/g128/channel, sym or asym) experts x dynamic per-token int8
    activations on the Marlin W4A8 kernels (CUDA, compute capability >= 8.0)."""

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",
        weight_quant: "QuantizationArgs",
        num_gpu_experts: int = -1,
    ):
        super().__init__(quant_config, weight_quant, num_gpu_experts)
        if self.num_bits != 4:
            raise ValueError(
                f"{MARKER}: only int4 expert weights have an int8-activation Marlin "
                f"kernel, got num_bits={self.num_bits}"
            )
        if self.actorder == "group":
            raise NotImplementedError(
                f"{MARKER}: group act-order (g_idx) is not supported by the "
                f"Marlin W4A8 kernel (no g_idx path)"
            )
        if self.strategy != "channel" and self.group_size not in W4A8_GROUP_SIZES:
            raise ValueError(
                f"{MARKER}: group_size {self.group_size} has no int8 kernel "
                f"(supported: {W4A8_GROUP_SIZES})"
            )
        self.w4a8_group_size = -1 if self.strategy == "channel" else int(self.group_size)
        layers, groups = _mark_scheme_built(self.w4a8_group_size)
        logger.info(
            "%s active layers=%d groups=%s (int4 experts x int8 activations, Marlin W4A8; "
            "layers = MoE layer schemes built so far in this process, groups = group sizes seen)",
            MARKER,
            layers,
            groups,
        )

    # -- guards ---------------------------------------------------------

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    # -- H88-C: what the layer shares with other processes ------------------

    def _factor_key(self, layer) -> str:
        """The name of the layer in the factor sidecar: the module prefix (unique: a target layer and an MTP draft
        layer can share a layer_id), else the store's layer key."""
        prefix = str(getattr(layer, "_sglang_prefix", "") or "")
        if prefix:
            return prefix
        from sglang.srt.layers.moe.cold_tier_fetch import layer_key_for

        return layer_key_for(layer)

    def _resolve_factor(
        self, layer, scale_attr: str, scale: torch.Tensor, rows, placeholder: bool
    ) -> Tuple[Optional[float], str]:
        """The factor of one scale tensor and where it came from: ``(factor | None, source)``.

        ``None`` = channelwise (no factor). Sources: ``local`` (one process, whole tensor: H88-B), ``published`` /
        ``adopted`` (agreed through the store), ``placeholder`` (single-group dummy load: nothing to agree on).

        Two-group boot (``SGLANG_WEG2_GROUP``): P and D share store rows and the flip moves bytes from one to the
        other, so the factor MUST be agreed -- without a store directory there is no channel and the boot is refused."""
        from sglang.srt.layers.moe import expert_store
        from sglang.srt.layers.moe.moe_w4a8_layout import GROUP_ENV, LAYOUT_W4A8

        E, G, N = scale.shape
        if G == 1:
            return None, "channelwise"
        proposal = None if placeholder else w4a8_scale_proposal(scale, rows)
        two_group = bool(os.environ.get(GROUP_ENV, "").strip())
        if expert_store.store_enabled():
            try:
                factor, source = expert_store.claim_scale_factor(
                    expert_store.store_dir(),
                    self._factor_key(layer),
                    scale_attr,
                    proposal,
                    group_size=self.w4a8_group_size,
                    layout=LAYOUT_W4A8,
                    writer=f"{os.environ.get(GROUP_ENV, '-') or '-'}:rank{getattr(layer, 'moe_tp_rank', '?')}",
                )
            except expert_store.W4A8FactorUnavailable:
                if not placeholder and proposal is None:
                    return 1.0, "local"  # this rank's view is all-zero (pad experts only): any factor scales zeros
                raise
            return factor, source
        if two_group:
            raise RuntimeError(
                f"{MARKER} REFUSED: group {os.environ.get(GROUP_ENV)!r} of a two-group boot runs the W4A8 expert layout "
                f"without SGLANG_MOE_EXPERT_STORE_DIR. P and D must use the same int16 scale factor for "
                f"{self._factor_key(layer)}/{scale_attr} (the flip moves their weight bytes into each other) and the "
                f"expert store directory is the channel that agrees it. Set the store directory for both groups, or "
                f"switch SGLANG_MOE_ACT_INT8 off."
            )
        if placeholder:
            return 1.0, "placeholder"
        return (proposal if proposal is not None else 1.0), "local"

    # -- weights --------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"{MARKER}: the kernel writes in the scale dtype, which must be "
                f"float16 or bfloat16, got {params_dtype}"
            )
        validate_w4a8_moe_dims(
            hidden_size, intermediate_size_per_partition, self.w4a8_group_size
        )
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        # BUFFERS, not attributes (H88-C, flip): the weight exchange walks every tensor of the model and refuses a
        # plain attribute tensor that is in no plan as UNCOVERED (W84); a registered buffer is booked as static
        # state and carried across sleep/wake by _export_static_state / _import_static_state in its own process.
        from sglang.srt.layers.moe.moe_w4a8_layout import W4A8_LAYER_GLOBAL

        for name in W4A8_LAYER_GLOBAL:
            if hasattr(layer, "_buffers") and name not in layer._buffers:
                layer.register_buffer(name, None, persistent=False)
            else:
                setattr(layer, name, None)

    def _repack_to_marlin(self, layer: torch.nn.Module) -> None:
        import time as _time

        from sglang.srt.layers.moe.expert_offload import (
            _STORE_CLOCK,
            MoEExpertOffloadCache,
            presplit_expert_offload_after_repack,
        )
        from sglang.srt.layers.moe.moe_w4a8_layout import (
            OFFLOAD_COVERED_ATTR,
            assert_offload_covers_w4a8,
        )
        from sglang.srt.layers.moe.store_adopt import repack_rows
        from sglang.srt.managers.weg2_memory_saver import back_into_tag_pool

        # #323b class: the offload cache must know every expert-major tensor the W4A8 kernel reads. Checked
        # BEFORE a byte is converted, and the layer is marked only after it passed (the offload guard reads the mark).
        assert_offload_covers_w4a8(MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS, self.sym)

        zero_pad = getattr(layer, "zero_expert_shard_pad", None)
        if callable(zero_pad):
            zero_pad()

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        def replace_tensor(name, new_t):
            target_attr = getattr(layer, name)
            if name not in layer._original_shapes:
                layer._original_shapes[name] = tuple(target_attr.shape)
            with back_into_tag_pool():
                target_attr.resize_(new_t.shape)
            target_attr.copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device
        # act-order is refused in __init__: g_idx stay empty, as in the A16 scheme
        for name in (
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        ):
            setattr(
                layer,
                name,
                torch.nn.Parameter(
                    torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                    requires_grad=False,
                ),
            )

        # ---- placeholder weights (D-first-flip adoption / dummy load) -------------------------------------
        # #112 of the A16 scheme: no repack on placeholders (random bytes, and the real ones arrive already in
        # the layout from the store / the legs). The W4A8 buffers keep the A8 shapes (the same as A16's), the
        # scales are only moved to the card; the factor comes from the store (adopt-only).
        try:
            from sglang.srt.weg2 import adopt as _weg2_adopt

            placeholder = bool(_weg2_adopt.weights_are_placeholder())
        except ImportError:
            placeholder = False
        # ---- rows this rank actually holds (D-store-adopt veto / h2d cut) -----------------------------------
        rows = None if placeholder else repack_rows(layer, num_experts)

        def packed_shape(name):
            t = getattr(layer, name)
            return (t.shape[0], t.shape[1] * self.packed_factor // 16, t.shape[2] * (self.num_bits // 2))

        _t_marlin = _time.perf_counter()
        for name in ("w13_weight_packed", "w2_weight_packed"):
            if placeholder:
                _dev = getattr(layer, name).device
                # a real rank always runs CUDA (the W4A8 dispatch refuses non-CUDA); a CPU process (unit test,
                # tooling) keeps its tensors where they are instead of dying on torch.cuda.current_device()
                if _dev.type != "cuda" and torch.cuda.is_available():
                    _dev = torch.device("cuda", torch.cuda.current_device())
                replace_tensor(
                    name,
                    torch.empty(packed_shape(name), device=_dev, dtype=getattr(layer, name).dtype),
                )
            else:
                replace_tensor(
                    name,
                    w4a8_repack_moe_weights(
                        getattr(layer, name), self.packed_factor, self.num_bits, rows=rows
                    ),
                )
        _STORE_CLOCK["marlin_s"] += _time.perf_counter() - _t_marlin
        _STORE_CLOCK["rows_repacked"] += num_experts if rows is None else len(rows)
        _STORE_CLOCK["rows_total"] += num_experts

        if placeholder and torch.cuda.is_available():
            _dev2 = torch.device("cuda", torch.cuda.current_device())
            for _attr in (
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_zero_point",
                "w2_weight_zero_point",
            ):
                _t = getattr(layer, _attr, None)
                if _t is not None and _t.device.type != "cuda":
                    replace_tensor(_attr, _t.data.to(_dev2))

        # ---- scales -> single permutation, int16, ONE factor per layer tensor (agreed, H88-C) ---------------
        _t_scales = _time.perf_counter()
        factors = {}
        for scale_attr, factor_attr in (
            ("w13_weight_scale", "w13_act_scale_factor"),
            ("w2_weight_scale", "w2_act_scale_factor"),
        ):
            scale = getattr(layer, scale_attr)
            factor, source = self._resolve_factor(layer, scale_attr, scale, rows, placeholder)
            if placeholder:
                # the bytes are the origin's (already int16 under its factor): keep them, take its factor only
                new_scale, f_t = scale.data, (
                    None if factor is None else torch.tensor(float(factor), dtype=torch.float32, device=scale.device)
                )
            elif source in ("published", "adopted"):
                new_scale, f_t = w4a8_process_moe_scales(
                    scale.data,
                    factor=factor,
                    rows=rows,
                    what=f"{self._factor_key(layer)}/{scale_attr} (factor {source})",
                )
            elif rows is None:  # "local" / channelwise with every row: H88-B arithmetic, byte for byte
                new_scale, f_t = w4a8_process_moe_scales(scale.data)
            else:
                # "local" but only the KEPT rows are loaded (H2 store-adopt veto): the vetoed rows' scale
                # entries were never written and must not set the tensor's factor. The factor of the kept
                # rows (:func:`w4a8_scale_proposal` over ``rows``) is applied exactly like an adopted one.
                new_scale, f_t = w4a8_process_moe_scales(
                    scale.data,
                    factor=factor,
                    rows=rows,
                    what=f"{self._factor_key(layer)}/{scale_attr} (factor local, vetoed rows discounted)",
                )
            if not placeholder:
                replace_tensor(scale_attr, new_scale)
            factors[factor_attr] = f_t
            if factor is not None:
                logger.info(
                    "%s SCALE-FACTOR layer=%s tensor=%s factor=%.9g source=%s group=%s",
                    MARKER,
                    self._factor_key(layer),
                    scale_attr,
                    float(factor),
                    source,
                    os.environ.get("SGLANG_WEG2_GROUP", "-") or "-",
                )
        _STORE_CLOCK["scales_s"] += _time.perf_counter() - _t_scales
        with back_into_tag_pool():
            for factor_attr, f_t in factors.items():
                setattr(layer, factor_attr, None if f_t is None else f_t.clone().reshape(()))

        if not self.sym and not placeholder:
            replace_tensor(
                "w13_weight_zero_point",
                w4a8_moe_zero_points(
                    layer.w13_weight_zero_point,
                    layer.w13_weight_zero_point.shape[2] * self.packed_factor,
                ),
            )
            replace_tensor(
                "w2_weight_zero_point",
                w4a8_moe_zero_points(
                    layer.w2_weight_zero_point,
                    layer.w2_weight_zero_point.shape[2] * self.packed_factor,
                ),
            )

        with back_into_tag_pool():
            from sglang.srt.layers.quantization.marlin_utils import (
                marlin_make_workspace,
            )

            layer.workspace = marlin_make_workspace(_rang_karte(layer), 4)
        layer.is_marlin_converted = True
        setattr(layer, OFFLOAD_COVERED_ATTR, True)

        _t_presplit = _time.perf_counter()
        presplit_expert_offload_after_repack(layer)
        _STORE_CLOCK["presplit_s"] += _time.perf_counter() - _t_presplit

    def restore_weights_before_loading(self, layer: torch.nn.Module):
        super().restore_weights_before_loading(layer)
        from sglang.srt.layers.moe.moe_w4a8_layout import (
            OFFLOAD_COVERED_ATTR,
            W4A8_LAYER_GLOBAL,
        )

        for name in W4A8_LAYER_GLOBAL:
            setattr(layer, name, None)
        if getattr(layer, OFFLOAD_COVERED_ATTR, False):
            setattr(layer, OFFLOAD_COVERED_ATTR, False)

    # -- runtime --------------------------------------------------------

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        from sglang.srt.layers.moe import get_moe_a2a_backend

        if not get_moe_a2a_backend().is_none():
            raise NotImplementedError(
                f"{MARKER}: the Marlin W4A8 MoE kernel has no expert-parallel / a2a "
                f"path (expert ids must not contain -1); got a2a backend "
                f"{get_moe_a2a_backend()}"
            )
        # the runner object only carries the config: apply_weights below calls
        # fused_marlin_moe_w4a8 directly, like the A16 scheme calls fused_marlin_moe
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def get_marlin_quant_info(self, layer):
        raise NotImplementedError(
            f"{MARKER}: MarlinMoeQuantInfo describes the W4A16 layout; the W4A8 "
            f"layer is served by fused_marlin_moe_w4a8 only"
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe_w4a8 import (
            fused_marlin_moe_w4a8,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation in (
            "silu",
            "gelu",
        ), "Only SiLU/GeLU activations are supported."

        expert_map = None
        if hasattr(layer, "dispatcher") and hasattr(
            layer.dispatcher, "local_expert_mapping"
        ):
            expert_map = layer.dispatcher.local_expert_mapping
        if expert_map is not None:
            raise NotImplementedError(
                f"{MARKER}: expert parallelism (expert_map) is not supported by the "
                f"Marlin W4A8 MoE kernel"
            )

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _router_logits = dispatch_output.topk_output

        output = fused_marlin_moe_w4a8(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            w1_act_factor=layer.w13_act_scale_factor,
            w2_act_factor=layer.w2_act_scale_factor,
            w1_zeros=layer.w13_weight_zero_point if not self.sym else None,
            w2_zeros=layer.w2_weight_zero_point if not self.sym else None,
            workspace=layer.workspace,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
            clamp_limit=self.moe_runner_config.swiglu_limit,
            activation=self.moe_runner_config.activation,
            is_gated=self.moe_runner_config.is_gated,
        )
        return StandardCombineInput(hidden_states=output)
