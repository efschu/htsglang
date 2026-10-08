"""H88-C (PLAN-H88-W4A8-1007, AP H88-C): the weight LAYOUT of the int4 MoE experts as a boot-wide fact.

Two layouts of the same checkpoint exist behind ``--moe-act-int8`` / ``SGLANG_MOE_ACT_INT8`` (H88-E):

``marlin_w4a16``  the default: CompressedTensorsWNA16MoE, A16 Marlin rows, group scales permuted with the
                  A16 permutation, no per-tensor factor.
``marlin_w4a8``   CompressedTensorsWNA16A8MoE (H88-B): W4A8 Marlin rows (32x32 tiles), group scales
                  permuted with the "single" permutation and stored as int16 (bit pattern in the model dtype),
                  ONE float32 factor per layer tensor (``w13_act_scale_factor`` / ``w2_act_scale_factor``).

The two layouts have the SAME tensor names and the SAME shapes -- that is exactly why mixing them is silent: the
shared expert store (``expert_store``), the D-store-adopt veto (``store_adopt``), the offload cache
(``expert_offload``) and the flip exchange would move W4A16 rows into a W4A8 layer and nothing would raise.
This module is the one place that says

* which layout a process / a launch is in (:func:`layout_tag`, :func:`launch_layout`),
* that P and D of one boot are in the SAME layout, else a named refusal (:func:`check_one_layout_per_boot`),
* which tensors a W4A8 layer owns and how the offload cache treats each (:data:`W4A8_EXPERT_MAJOR`,
  :data:`W4A8_LAYER_GLOBAL`, :func:`assert_offload_covers_w4a8`) -- the #323b class: a new scheme whose
  tensors the offload tuple does not know pairs expert rows with the wrong scales,
* the byte sizes per tensor of both layouts from the repack shapes (:func:`layer_tensor_bytes`,
  :func:`census_compare`) -- the W71 census question of D6.

Pure python on purpose (no torch at import): the launcher imports it, and the launcher must stay light.
"""

from __future__ import annotations

import os
import shlex
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MARKER = "H88C"

LAYOUT_W4A16 = "marlin_w4a16"
LAYOUT_W4A8 = "marlin_w4a8"
LAYOUTS = (LAYOUT_W4A16, LAYOUT_W4A8)

#: the env switch (H88-E); the flag spelling is ``--moe-act-int8 on``
SWITCH_ENV = "SGLANG_MOE_ACT_INT8"
SWITCH_FLAG = "--moe-act-int8"

#: the launcher publishes the group of a rank (``SGLANG_WEG2_GROUP=P|D``) in the rank env; a process
#: without it is a plain single-group server
GROUP_ENV = "SGLANG_WEG2_GROUP"

REFUSAL = "H88C MoeLayoutMismatch"

_TRUE = ("true", "1", "yes", "y")
_FALSE = ("false", "0", "no", "n")


class MoeLayoutMismatch(RuntimeError):
    """P and D of one launch would run different expert layouts (or a layout the store does not carry)."""


def layout_tag(server_args: Optional[Any] = None) -> str:
    """The layout THIS process runs: ``marlin_w4a8`` iff the H88-E switch asks for int8 activations."""
    from sglang.srt.layers.quantization.moe_act_int8 import moe_act_int8_requested

    return LAYOUT_W4A8 if moe_act_int8_requested(server_args) else LAYOUT_W4A16


# ---------------------------------------------------------------------------
# launch side: what the launcher can know before any rank exists
# ---------------------------------------------------------------------------


def _env_switch_value(spec: str, base_env: Mapping[str, str]) -> Optional[bool]:
    """The value of SWITCH_ENV in a ``K=V;K=V`` group env spec, else in ``base_env``; None = not set anywhere."""
    for item in [x for x in str(spec or "").split(";") if x.strip()]:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if k.strip() == SWITCH_ENV:
            return _parse_bool(v.strip(), where=f"{SWITCH_ENV} in the group env")
    raw = base_env.get(SWITCH_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    return _parse_bool(str(raw).strip(), where=f"{SWITCH_ENV} in the launcher env")


def _parse_bool(value: str, *, where: str) -> bool:
    v = value.lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise MoeLayoutMismatch(f'{REFUSAL}: "{value}" ({where}) is not a boolean (1/0, true/false, yes/no)')


def _flag_value(extra: str) -> Optional[bool]:
    """``--moe-act-int8 on|off`` / ``--moe-act-int8=on`` in an ``--extra-p`` / ``--extra-d`` string; the LAST one
    wins, as in argparse; None = not present."""
    try:
        toks = shlex.split(str(extra or ""))
    except ValueError:
        return None
    out: Optional[bool] = None
    i = 0
    while i < len(toks):
        t = toks[i]
        val: Optional[str] = None
        if t == SWITCH_FLAG and i + 1 < len(toks):
            val = toks[i + 1]
            i += 1
        elif t.startswith(SWITCH_FLAG + "="):
            val = t.split("=", 1)[1]
        if val is not None:
            out = val.strip().lower() == "on"
        i += 1
    return out


def group_layout(env_spec: str = "", extra_args: str = "", base_env: Optional[Mapping[str, str]] = None) -> str:
    """The layout one rank group will run, from what the launcher hands it: the group env (``--env-p`` /
    ``--env-d``), the launcher's own env (a rank inherits it), and the group's extra args (``--extra-p`` /
    ``--extra-d``). Either spelling switches it ON (as ``moe_act_int8_requested`` does); a flag ``off`` does not
    undo an env ``1`` there either -- the same OR the rank applies."""
    env = os.environ if base_env is None else base_env
    on_env = _env_switch_value(env_spec, env)
    on_flag = _flag_value(extra_args)
    return LAYOUT_W4A8 if (on_env or on_flag) else LAYOUT_W4A16


def launch_layout(ns: Any, base_env: Optional[Mapping[str, str]] = None) -> Tuple[str, str, str]:
    """``(layout, layout_p, layout_d)`` of a launch namespace; raises :class:`MoeLayoutMismatch` when P != D.

    One layout for all cards of a boot: P and D share the expert store file per (layer, tensor) and the flip moves
    their bytes into each other, so a boot with P on W4A8 and D on W4A16 (or the reverse) cannot be served --
    refused here, by name, before any rank loads a byte (never a late OOM or silently wrong rows)."""
    lp = group_layout(getattr(ns, "env_p", ""), getattr(ns, "extra_p", ""), base_env)
    ld = group_layout(getattr(ns, "env_d", ""), getattr(ns, "extra_d", ""), base_env)
    check_one_layout_per_boot(lp, ld)
    return lp, lp, ld


def check_one_layout_per_boot(layout_p: str, layout_d: str) -> None:
    if layout_p != layout_d:
        raise MoeLayoutMismatch(
            f"{REFUSAL}: group P runs the expert layout {layout_p!r} and group D {layout_d!r}. The groups share one "
            f"expert store file per (layer, tensor) and the flip moves their weight bytes into each other; the two "
            f"layouts have the same tensor names and shapes but different bytes (W4A8 rows, int16 scales, a "
            f"per-tensor scale factor), so a mixed boot would serve W4A16 rows through the W4A8 kernel or the reverse. "
            f"Set {SWITCH_ENV}=1 (or {SWITCH_FLAG} on) for BOTH groups -- NF_ENV_P and NF_ENV_D of the profile "
            f"carry it, nf-int4-w4a8.env -- or for neither."
        )


# ---------------------------------------------------------------------------
# tensors of the W4A8 layer (the #323b guard)
# ---------------------------------------------------------------------------

#: expert-major tensors (dim 0 = expert) of a W4A8 MoE layer that the Marlin W4A8 kernel reads. Each must be a member
#: of ``MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS`` -- else the offload cache stages a subset of what the kernel reads
#: and pairs expert rows with another expert's scales (#323b).
W4A8_EXPERT_MAJOR: Tuple[str, ...] = (
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
)
#: only asymmetric checkpoints (no NF checkpoint): read per expert by the kernel
W4A8_EXPERT_MAJOR_ASYM: Tuple[str, ...] = ("w13_weight_zero_point", "w2_weight_zero_point")
#: layer-global tensors (NOT expert-major): one float32 0-d factor per layer tensor, ``None`` for channelwise scales.
#: They are registered BUFFERS (non-persistent): the flip exchange (W84 coverage walk) books a buffer as static state,
#: an unregistered attribute tensor as UNCOVERED and refuses the boot; ``_export_static_state`` /
#: ``_import_static_state`` carry a buffer across sleep/wake in its own process.
W4A8_LAYER_GLOBAL: Tuple[str, ...] = ("w13_act_scale_factor", "w2_act_scale_factor")
#: the scale attribute each factor belongs to
FACTOR_OF_SCALE: Mapping[str, str] = {
    "w13_weight_scale": "w13_act_scale_factor",
    "w2_weight_scale": "w2_act_scale_factor",
}

#: the marker the scheme sets on a layer once its tensors are verified against the offload tuple; the offload guard
#: (``assert_expert_offload_quant_supported``) admits ``CompressedTensorsWNA16A8MoE`` only on such a layer
OFFLOAD_COVERED_ATTR = "_moe_offload_w4a8_covered"


def expert_major_names(sym: bool) -> Tuple[str, ...]:
    return W4A8_EXPERT_MAJOR + (() if sym else W4A8_EXPERT_MAJOR_ASYM)


def assert_offload_covers_w4a8(expert_tensor_attrs: Iterable[str], sym: bool) -> None:
    """Every expert-major tensor of the W4A8 layer is in the offload tuple, and no layer-global tensor is (a
    layer-global tensor in the tuple would be sliced per expert). Raises ``RuntimeError`` naming the tensor."""
    attrs = set(expert_tensor_attrs)
    missing = [n for n in expert_major_names(sym) if n not in attrs]
    if missing:
        raise RuntimeError(
            f"{MARKER} W4A8 OFFLOAD-GUARD (#323b class): the expert-major tensors {missing} of the W4A8 MoE layer are "
            f"not in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS, so the offload cache would stage a strict subset of "
            f"what the Marlin W4A8 kernel reads and pair experts with another expert's scales (silently wrong "
            f"output). Add them to the tuple and to presplit_expert_offload_after_repack."
        )
    wrong = [n for n in W4A8_LAYER_GLOBAL if n in attrs]
    if wrong:
        raise RuntimeError(
            f"{MARKER} W4A8 OFFLOAD-GUARD: {wrong} are layer-global factors (0-d), not expert-major tensors; the "
            f"offload tuple must not slice them per expert."
        )


# ---------------------------------------------------------------------------
# bytes per tensor of both layouts (D6, W71 census)
# ---------------------------------------------------------------------------

_INT32 = 4
_FACTOR_BYTES = 4  # float32 0-d


def layer_tensor_bytes(
    layout: str,
    *,
    num_experts: int,
    hidden: int,
    intermediate: int,
    group_size: int,
    sym: bool = True,
    scale_bytes: int = 2,
) -> Dict[str, int]:
    """Byte size of every tensor of ONE MoE layer's expert weights, from the repack shapes.

    Shapes (K = reduction, N = output of the grouped GEMM; w13: K = hidden, N = 2 * intermediate; w2: K =
    intermediate, N = hidden; G = K / group_size, or 1 for channelwise):

    * packed weights: int32 ``[E, K/16, 2N]`` in BOTH layouts (``gptq_marlin_moe_repack``: ``N * (num_bits // 2)``;
      ``gptq_marlin_moe_repack_w4a8``: ``N * 16 / 8``);
    * scales: ``[E, G, N]`` of the model dtype in BOTH layouts (a permutation, plus for W4A8 an int16 bit pattern
      viewed as the model dtype -- the element size does not change);
    * zero points (asymmetric only): int32 ``[E, G, N/8]`` in BOTH layouts;
    * W4A8 only: two float32 0-d factors per layer (``None``, i.e. no bytes, for channelwise scales).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown expert layout {layout!r} (known: {LAYOUTS})")
    E, H, I = int(num_experts), int(hidden), int(intermediate)
    out: Dict[str, int] = {}
    for tag, K, N in (("w13", H, 2 * I), ("w2", I, H)):
        G = 1 if group_size == -1 else K // int(group_size)
        out[f"{tag}_weight_packed"] = E * (K // 16) * (2 * N) * _INT32
        out[f"{tag}_weight_scale"] = E * G * N * int(scale_bytes)
        if not sym:
            out[f"{tag}_weight_zero_point"] = E * G * (N // 8) * _INT32
        if layout == LAYOUT_W4A8 and G > 1:
            out[f"{tag}_act_scale_factor"] = _FACTOR_BYTES
    return out


def census_compare(**kw) -> List[Dict[str, Any]]:
    """Rows ``{tensor, a16_bytes, a8_bytes, delta}`` over the union of both layouts' tensors (same kwargs as
    :func:`layer_tensor_bytes`, without ``layout``)."""
    a16 = layer_tensor_bytes(LAYOUT_W4A16, **kw)
    a8 = layer_tensor_bytes(LAYOUT_W4A8, **kw)
    rows = []
    for name in list(a16) + [n for n in a8 if n not in a16]:
        b16, b8 = a16.get(name, 0), a8.get(name, 0)
        rows.append({"tensor": name, "a16_bytes": b16, "a8_bytes": b8, "delta": b8 - b16})
    return rows
