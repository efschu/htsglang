"""27b-gdn-form-1028: the form of the GDN (Gated DeltaNet) layers this rank built, read from the real modules, and whether NF's two GDN
fusions would apply to them -- one census line per rank (``27B-GDN-FORM``). Nothing in the forward changes.

NF built two fusions for the same class (``Qwen3_5GatedDeltaNet``): FORMHP-GDN-IN (in_proj_qkvz W8A16 GEMV + split as one kernel,
b166989d8) and FORMHP-GDN-OUT (gated RMSNorm + out_proj W8A16 GEMV as one kernel, 67c1f8242). Their first detection refused the real NF
module on ``tp_size != 1`` (formb169 21:46:32 TP0: Form B builds the GDN layers with tp_size 3 and the split [all, 0, 0]); 14a71cc5e
replaced it with "the projection's forward IS ``quant_method.apply`` on this rank". The same rule is applied here, structure first and
the lane last, so the line says whether the 27B's form would pass everything except the lane:

  * out_proj: input already parallel, no all-reduce inside (``reduce_results`` False or TP 1), no bias, K == local v heads x head dim;
  * in_proj_qkvz: no all-gather (``gather_output`` False or TP 1), no bias, v/k head ratio one the split kernel takes;
  * the norm: ``RMSNormGated`` without bias, norm before the gate, no group size, activation swish / silu / sigmoid;
  * heads > 0 on this rank (a P stage or a rank without GDN heads has none);
  * the lane: both NF kernels embed the formhp per-channel e4m3 W8A16 GEMV tile loop (``FormhpDenseFp8LinearMethod``). The 27B forms run
    other methods (INT8: CompressedTensors W8A8 int8 -> per-token quant + cutlass/sm120 int8 GEMM; NVFP4-RadixArk GDN: ModelOpt FP8;
    FP8: block-128 Fp8LinearMethod; INT4-AR: CompressedTensors W4A16), so a port would REPLACE the production GEMM, not fuse around it
    (A5 1028: skipped -- the bound of the gain is below 0.3 ms a round, see the commit).

Switch ``FLLIPER_GDN_FORM_CENSUS`` (default on).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, NamedTuple, Tuple

import torch

logger = logging.getLogger(__name__)

#: v/k head ratios ``fused_qkvzba_split_reshape_cat_contiguous`` takes on CUDA (qwen3_5._GDN_FUSED_QKVZBA_RATIOS)
SPLIT_RATIOS = (1, 2, 3, 4)
#: the only lane NF's fused kernels are written for
NF_LANE = "FormhpDenseFp8LinearMethod"
NORM_ACTIVATIONS = ("swish", "silu", "sigmoid")


class GdnForm(NamedTuple):
    """One GDN layer's form on this rank (everything but the layer id; equal forms are counted together)."""

    cls: str
    tp: int
    tp_rank: int
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    hidden: int
    qkvz: str
    qkvz_n: int
    qkvz_gather: bool
    qkvz_bias: bool
    ba: str
    out: str
    out_k: int
    out_in_parallel: bool
    out_reduce: bool
    out_bias: bool
    norm: str
    norm_act: str
    norm_before_gate: bool
    norm_group: int
    norm_bias: bool
    eps: float


def method_name(layer: torch.nn.Module) -> str:
    """The quant method's class, with the scheme for a compressed-tensors layer (its apply dispatches to ``layer.scheme``)."""
    from flliper.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod

    name = type(layer.quant_method).__name__
    if isinstance(layer.quant_method, CompressedTensorsLinearMethod):
        return f"{name}/{type(layer.scheme).__name__}"
    return name


def read_form(mod: torch.nn.Module) -> GdnForm:
    """The form of one ``Qwen3_5GatedDeltaNet`` as built on this rank (attributes of the real projections and norm)."""
    qkvz, out, norm = mod.in_proj_qkvz, mod.out_proj, mod.norm
    return GdnForm(
        cls=type(mod).__name__,
        tp=int(out.tp_size),
        tp_rank=int(out.tp_rank),
        k_heads=int(mod.local_num_k_heads),
        v_heads=int(mod.local_num_v_heads),
        k_dim=int(mod.head_k_dim),
        v_dim=int(mod.head_v_dim),
        hidden=int(mod.hidden_size),
        qkvz=method_name(qkvz),
        qkvz_n=int(qkvz.output_size_per_partition),
        qkvz_gather=bool(qkvz.gather_output),
        qkvz_bias=qkvz.bias is not None,
        ba=method_name(mod.in_proj_ba),
        out=method_name(out),
        out_k=int(out.input_size_per_partition),
        out_in_parallel=bool(out.input_is_parallel),
        out_reduce=bool(out.reduce_results),
        out_bias=out.bias is not None,
        norm=type(norm).__name__,
        norm_act=str(norm.activation),
        norm_before_gate=bool(norm.norm_before_gate),
        norm_group=0 if norm.group_size is None else int(norm.group_size),
        norm_bias=norm.bias is not None,
        eps=float(norm.eps),
    )


def gdn_out_why(f: GdnForm) -> str:
    """"" when NF's GDN-OUT (gnorm + out_proj) would apply to this form, else the first reason it would not (rule of 14a71cc5e)."""
    if f.v_heads <= 0:
        return "no v heads on this rank"
    if not f.out_in_parallel:
        return "out_proj slices its input"
    if f.tp > 1 and f.out_reduce:
        return f"out_proj all-reduces (tp {f.tp})"
    if f.out_bias:
        return "out_proj has a bias"
    if f.norm_bias or not f.norm_before_gate or f.norm_group:
        return "norm form (bias / gate before norm / group size)"
    if f.norm_act not in NORM_ACTIVATIONS:
        return f"norm activation {f.norm_act}"
    if f.out_k != f.v_heads * f.v_dim:
        return f"out_proj K {f.out_k} != {f.v_heads} x {f.v_dim}"
    if f.out != NF_LANE:
        return f"lane {f.out}"
    return ""


def gdn_in_why(f: GdnForm) -> str:
    """"" when NF's GDN-IN (in_proj_qkvz + split) would apply to this form, else the first reason it would not (rule of 14a71cc5e)."""
    if f.k_heads <= 0 or f.v_heads <= 0:
        return "no heads on this rank"
    if f.tp > 1 and f.qkvz_gather:
        return f"in_proj_qkvz all-gathers (tp {f.tp})"
    if f.qkvz_bias:
        return "in_proj_qkvz has a bias"
    if f.v_heads % f.k_heads or f.v_heads // f.k_heads not in SPLIT_RATIOS:
        return f"v/k head ratio {f.v_heads}/{f.k_heads}"
    if f.qkvz_n != 2 * f.k_heads * f.k_dim + 2 * f.v_heads * f.v_dim:
        return f"in_proj_qkvz N {f.qkvz_n} != q k v z of the local heads"
    if f.qkvz != NF_LANE:
        return f"lane {f.qkvz}"
    return ""


def _verdict(why: str) -> str:
    return "fuse" if not why else "skip:" + why.replace(" ", "_")


def form_text(f: GdnForm) -> str:
    """One form, as the census prints it."""
    return (
        f"cls={f.cls} tp={f.tp} rank={f.tp_rank} heads_k/v={f.k_heads}/{f.v_heads} dim_k/v={f.k_dim}/{f.v_dim} hidden={f.hidden} "
        f"qkvz={f.qkvz} N={f.qkvz_n} gather={f.qkvz_gather} bias={f.qkvz_bias} ba={f.ba} "
        f"out={f.out} K={f.out_k} in_parallel={f.out_in_parallel} reduce={f.out_reduce} bias={f.out_bias} "
        f"norm={f.norm}({f.norm_act},before_gate={f.norm_before_gate},group={f.norm_group},bias={f.norm_bias},eps={f.eps:g}) "
        f"gdn_in={_verdict(gdn_in_why(f))} gdn_out={_verdict(gdn_out_why(f))}"
    )


def census_line(*, forms: Iterable[Tuple[int, GdnForm]], pp_rank: int, pp_size: int) -> str:
    """``27B-GDN-FORM`` over (layer id, form) pairs: the layer count, then each distinct form with its count and layer range."""
    by_form: Counter = Counter()
    ids: dict = {}
    for layer_id, f in forms:
        by_form[f] += 1
        ids.setdefault(f, []).append(int(layer_id))
    parts = [
        f"[n={n} layers={min(ids[f])}..{max(ids[f])} {form_text(f)}]" for f, n in sorted(by_form.items(), key=lambda kv: min(ids[kv[0]]))
    ]
    return f"27B-GDN-FORM pp={pp_rank}/{pp_size} gdn={sum(by_form.values())} forms={len(by_form)} " + " ".join(parts)


def collect(*, layers: torch.nn.Module, gdn_cls: type) -> list:
    """(layer id, form) of every ``gdn_cls`` module under ``layers`` (placeholders of other stages / workers are other classes)."""
    return [(int(m.layer_id), read_form(m)) for m in layers.modules() if isinstance(m, gdn_cls)]


def log_census(*, layers: torch.nn.Module, gdn_cls: type, pp_rank: int, pp_size: int) -> str:
    """The construction-time hook (``Qwen3_5ForCausalLM.__init__``): one ``27B-GDN-FORM`` line when this stage built GDN layers.
    A census must never stop a boot: a failure is logged as ``27B-GDN-FORM FAILED`` and the model is built as before."""
    from flliper.srt.environ import envs

    if not envs.FLLIPER_GDN_FORM_CENSUS.get():
        return ""
    try:
        forms = collect(layers=layers, gdn_cls=gdn_cls)
        if not forms:
            return ""
        line = census_line(forms=forms, pp_rank=pp_rank, pp_size=pp_size)
    except Exception as exc:  # noqa: BLE001 -- census only
        line = f"27B-GDN-FORM FAILED {type(exc).__name__}: {exc}"
        logger.error("%s", line)
        return line
    logger.info("%s", line)
    return line
