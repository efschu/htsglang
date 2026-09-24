# SPDX-License-Identifier: Apache-2.0
"""NVFP4 checkpoint FORMAT KEY and a METADATA-ONLY mixed-precision dry run (H68).

Two questions have to be answered before an NVFP4 checkpoint is booted on this
rig, and both are answerable from files alone -- no torch tensor, no GPU, no
model load:

1. **Which NVFP4 is it?**  "NVFP4" names at least two incompatible families:

   * ``modelopt`` -- NVIDIA ModelOpt exports (nvidia/*, RadixArk/*): a
     ``quant_algo`` (``NVFP4``, ``W4A16_NVFP4`` or ``MIXED_PRECISION`` with a
     per-module ``quantized_layers`` map), per-tensor ``weight_scale_2`` and
     ``input_scale``, loaded by ``ModelOpt*Config``;
   * ``compressed-tensors`` -- llm-compressor exports (unsloth "Dynamic",
     ``format: nvfp4-pack-quantized`` / ``mixed-precision``), ``weight_packed``
     + ``weight_global_scale``, loaded by ``CompressedTensorsConfig``.

   The repo name does not decide it and neither does the word "NVFP4"; the
   quantization sections do. :func:`detect_nvfp4_format` reads them and names
   the checkpoint with one greppable key, e.g.::

       modelopt:MIXED_PRECISION|draft.moe.routed=fp8blk128;moe.routed=nvfp4a4g16;ple.table=fp8pt;*=bf16

2. **Does every tensor fit the plan?**  A MIXED_PRECISION export mixes
   algorithms per module, and the two quantization files of one export can
   disagree (nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54fb: ``config.json`` says
   ``FP8_PB_WO`` for the MTP experts, ``hf_quant_config.json`` says
   ``FP8_BLOCK_SCALES`` -- aliases for the same 128x128 block layout, vLLM
   PR #55513).  :func:`plan_from_checkpoint` walks
   ``model.safetensors.index.json`` and the safetensors HEADERS of every shard,
   resolves each checkpoint module to the algorithm the config declares,
   derives the algorithm its tensors actually carry (dtype, shape, companion
   scales), and reports every disagreement -- plus the byte census per (role,
   algorithm) and the per-expert row size of the routed experts, which is the
   unit the expert pool, the host store and the Platztausch are sized in.

A shard is read only when it is COMPLETE: its size must reach
``8 + header_len + last data_offset``; a shorter file is refused
(:class:`IncompleteShard`), never measured.  The only tensor BYTES ever read
are F32 scalars (``weight_scale_2``, 4 bytes each), and only with
``read_scalars=True``.

Model-agnostic on purpose (the 27B and the Next-Flash line pick the same
file): the roles below are name heuristics that fall back to ``other``;
nothing in here carries a rig or model constant.

CLI::

    python -m sglang.srt.weg2.nvfp4_ckpt_plan <model_dir> [--scalars] [--json]
    python -m sglang.srt.weg2.nvfp4_ckpt_plan <model_dir> --format-only
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import struct
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Algorithms (normalized)
# ---------------------------------------------------------------------------

#: E2M1 weights AND E2M1 activations, E4M3 group scales + F32 per-tensor scale.
ALGO_NVFP4 = "NVFP4"
#: Weight-only NVFP4 (activations stay 16-bit).
ALGO_NVFP4_A16 = "W4A16_NVFP4"
#: FP8 E4M3 weight with ONE per-tensor scale (``weight_scale`` scalar).
ALGO_FP8 = "FP8"
#: FP8 E4M3 weight with 2-D block scales (``weight_scale_inv``).
ALGO_FP8_BLOCK = "FP8_BLOCK"
ALGO_MXFP8 = "MXFP8"
#: Not quantized: excluded, or simply not listed in ``quantized_layers``.
ALGO_BF16 = "BF16"
#: Modules that carry no ``weight`` at all (buffers, A_log, offsets ...).
ALGO_BUFFER = "BUFFER"

#: Raw ModelOpt / checkpoint spellings -> normalized algorithm.  ``FP8_PB_WO``
#: is ModelOpt's canonical 2-D block-FP8 name, ``FP8_BLOCK_SCALES`` the
#: spelling of early composed Qwen3.8-Flash-Next exports (vLLM PR #55513).
_ALGO_ALIASES: Dict[str, str] = {
    "NVFP4": ALGO_NVFP4,
    "W4A4_NVFP4": ALGO_NVFP4,
    "NVFP4_AWQ": ALGO_NVFP4,
    "W4A16_NVFP4": ALGO_NVFP4_A16,
    "NVFP4_A16": ALGO_NVFP4_A16,
    "FP8": ALGO_FP8,
    "FP8_PB_WO": ALGO_FP8_BLOCK,
    "FP8_BLOCK_SCALES": ALGO_FP8_BLOCK,
    "MXFP8": ALGO_MXFP8,
    "BF16": ALGO_BF16,
    "NONE": ALGO_BF16,
}

FLAVOUR_MODELOPT = "modelopt"
FLAVOUR_COMPRESSED_TENSORS = "compressed-tensors"

FORMAT_LINE_PREFIX = "NVFP4-FORMAT"
PLAN_LINE_PREFIX = "NVFP4-PLAN"
ROW_LINE_PREFIX = "NVFP4-EXPERT-ROW"


def normalize_algo(raw: Optional[str]) -> str:
    """ModelOpt ``quant_algo`` spelling -> normalized algorithm.

    Unknown spellings come back as ``UNKNOWN:<raw>`` -- never mapped to a
    neighbour, because a guessed algorithm is exactly the silent-wrong-layout
    failure this module exists to catch."""
    a = str(raw or "").strip().upper()
    if not a:
        return ALGO_BF16
    return _ALGO_ALIASES.get(a, f"UNKNOWN:{a}")


def algo_tag(algo: str, group_size: Optional[int] = None) -> str:
    """Short tag used inside the format key (``nvfp4a4g16``, ``fp8blk128``)."""
    g = int(group_size) if group_size else None
    if algo == ALGO_NVFP4:
        return f"nvfp4a4g{g or 16}"
    if algo == ALGO_NVFP4_A16:
        return f"nvfp4a16g{g or 16}"
    if algo == ALGO_FP8:
        return "fp8pt"
    if algo == ALGO_FP8_BLOCK:
        return f"fp8blk{g or 128}"
    if algo == ALGO_MXFP8:
        return "mxfp8"
    if algo == ALGO_BF16:
        return "bf16"
    return algo.lower()


# ---------------------------------------------------------------------------
# Roles (name heuristics, model-agnostic; unknown -> "other")
# ---------------------------------------------------------------------------

#: Draft/MTP namespaces: the drafter is a SEPARATE model whose format never
#: describes the target's families (ANALYSE_321 sec. 7 e 1).
_DRAFT_RE = re.compile(r"(^|\.)(mtp|nextn|draft)(\.|$)")


def module_role(module: str) -> str:
    """Coarse role of a CHECKPOINT module path (or a compressed-tensors target).

    Order is load-bearing: routed experts live under ``mlp.experts`` and the
    shared expert under ``mlp.shared_expert``, so both expert roles are decided
    before the generic ``mlp`` one; ``ngram_embedding`` before ``ple``."""
    m = module.lower().replace("\\", "")
    if "shared_expert" in m:
        base = "moe.shared"
    elif re.search(r"\.experts(\.|$)", m) or m.endswith("experts"):
        base = "moe.routed"
    elif re.search(r"\.(mlp|block_sparse_moe)\.gate$", m) or m.endswith(".router"):
        base = "moe.router"
    elif "ngram_embedding" in m:
        base = "ple.table"
    elif ".ple." in m or m.endswith(".ple"):
        base = "ple"
    elif "hyper_connection" in m:
        base = "hyper"
    elif "linear_attn" in m:
        base = "attn.gdn"
    elif "self_attn" in m or "self_attention" in m:
        base = "attn.full"
    elif "embed_tokens" in m or "word_embeddings" in m:
        base = "vocab.embed"
    elif "lm_head" in m or "output_layer" in m:
        base = "vocab.lm_head"
    elif "visual" in m or "vision" in m:
        base = "vision"
    elif re.search(r"\.(mlp|feed_forward|ffn)(\.|$)", m):
        base = "mlp.dense"
    else:
        base = "other"
    return f"draft.{base}" if _DRAFT_RE.search(m) else base


# ---------------------------------------------------------------------------
# Quantization sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSections:
    """The two places a checkpoint declares its quantization."""

    #: ``config.json`` ``quantization_config`` (top level or ``text_config``);
    #: this is what sglang's ``ModelConfig`` reads first.
    config: Mapping
    #: ``hf_quant_config.json`` ``quantization`` section (ModelOpt legacy).
    hf: Mapping
    #: producer (``{"name": "modelopt", "version": ...}``) if declared.
    producer: Mapping


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def read_quant_sections(model_dir: str) -> QuantSections:
    cfg = _load_json(os.path.join(model_dir, "config.json")) or {}
    qc = cfg.get("quantization_config")
    if not qc and isinstance(cfg.get("text_config"), dict):
        qc = cfg["text_config"].get("quantization_config")
    hf_payload = _load_json(os.path.join(model_dir, "hf_quant_config.json")) or {}
    hf = hf_payload.get("quantization") if isinstance(hf_payload, dict) else None
    producer = (qc or {}).get("producer") or hf_payload.get("producer") or {}
    return QuantSections(config=qc or {}, hf=hf or {}, producer=producer or {})


# ---------------------------------------------------------------------------
# Format key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemeUse:
    """One (role, algorithm, group) the checkpoint declares, with a count."""

    role: str
    algo: str
    group_size: Optional[int]
    modules: int


@dataclass(frozen=True)
class Nvfp4FormatKey:
    flavour: str
    quant_algo: str
    producer: str
    #: NVFP4 group size (16 for every export seen so far).
    group_size: Optional[int]
    #: 4 for W4A4 (``NVFP4``), 16 for weight-only (``W4A16_NVFP4``): whether a
    #: native FP4 lane is even reachable.  The Marlin lane serves both as W4A16.
    activation_bits: int
    kv_cache_quant_algo: Optional[str]
    schemes: Tuple[SchemeUse, ...]
    #: Algorithm of the draft/MTP routed experts, ``None`` = not quantized.
    draft_algo: Optional[str]
    #: Algorithm of the PLE n-gram table, ``None`` = no PLE entry.
    ple_algo: Optional[str]
    #: Places where config.json and hf_quant_config.json disagree.
    inconsistencies: Tuple[str, ...]
    key: str

    def line(self) -> str:
        inc = (
            f" inconsistencies={len(self.inconsistencies)}"
            if self.inconsistencies
            else ""
        )
        return (
            f"{FORMAT_LINE_PREFIX} key={self.key} producer={self.producer or '?'} "
            f"act_bits={self.activation_bits} kv={self.kv_cache_quant_algo or 'none'}{inc}"
        )


def _layers_map(section: Mapping) -> Dict[str, Mapping]:
    ql = section.get("quantized_layers") if isinstance(section, Mapping) else None
    return dict(ql) if isinstance(ql, Mapping) else {}


def _diff_quantized_layers(cfg: Mapping, hf: Mapping) -> List[str]:
    """Human-readable disagreements between the two ``quantized_layers`` maps."""
    a, b = _layers_map(cfg), _layers_map(hf)
    if not a or not b:
        return []
    out: List[str] = []
    for key in sorted(set(a) | set(b)):
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None:
            where = "config.json" if ra is not None else "hf_quant_config.json"
            out.append(f"{key}: listed only in {where}")
            continue
        aa = str((ra or {}).get("quant_algo") or "")
        ab = str((rb or {}).get("quant_algo") or "")
        ga, gb = (ra or {}).get("group_size"), (rb or {}).get("group_size")
        if aa.upper() == ab.upper() and ga == gb:
            continue
        if normalize_algo(aa) == normalize_algo(ab) and ga == gb:
            verdict = f"aliases -> {normalize_algo(aa)}" + (f" g{ga}" if ga else "")
        else:
            verdict = "CONFLICT"
        out.append(
            f"{key}: config.json={aa}{'/g' + str(ga) if ga else ''} "
            f"hf_quant_config.json={ab}{'/g' + str(gb) if gb else ''} ({verdict})"
        )
    return out


def _producer_str(producer: Mapping) -> str:
    if not producer:
        return ""
    return f"{producer.get('name', '?')} {producer.get('version', '?')}".strip()


def _group_or_none(value) -> Optional[int]:
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def detect_nvfp4_format(
    model_dir: Optional[str] = None,
    *,
    sections: Optional[QuantSections] = None,
) -> Optional[Nvfp4FormatKey]:
    """Name the NVFP4 flavour of a checkpoint, or ``None`` when it declares no
    NVFP4 module at all.  Reads the two JSON files only."""
    if sections is None:
        if model_dir is None:
            raise ValueError("detect_nvfp4_format needs model_dir or sections")
        sections = read_quant_sections(model_dir)
    cfg, hf = sections.config or {}, sections.hf or {}
    method = str(cfg.get("quant_method") or "").lower()

    if method in ("compressed-tensors", "compressed_tensors"):
        return _detect_compressed_tensors(cfg, sections)

    quant_algo = str(cfg.get("quant_algo") or hf.get("quant_algo") or "").upper()
    is_modelopt = (
        method.startswith("modelopt")
        or str(sections.producer.get("name", "")).lower() == "modelopt"
        or (not method and bool(quant_algo))
    )
    if not is_modelopt:
        return None

    layers = _layers_map(cfg) or _layers_map(hf)
    kv = cfg.get("kv_cache_quant_algo", hf.get("kv_cache_quant_algo"))
    uses: Dict[Tuple[str, str, Optional[int]], int] = {}
    if quant_algo == "MIXED_PRECISION":
        for module, info in layers.items():
            info = info or {}
            k = (
                module_role(str(module)),
                normalize_algo(info.get("quant_algo")),
                _group_or_none(info.get("group_size")),
            )
            uses[k] = uses.get(k, 0) + 1
    else:
        g = _group_or_none(cfg.get("group_size") or hf.get("group_size"))
        uses[("*", normalize_algo(quant_algo), g)] = 1

    fp4 = [k for k in uses if k[1] in (ALGO_NVFP4, ALGO_NVFP4_A16)]
    if not fp4:
        return None
    groups = sorted({k[2] for k in fp4 if k[2]})
    schemes = tuple(
        SchemeUse(role=r, algo=a, group_size=g, modules=n)
        for (r, a, g), n in sorted(uses.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    )
    draft = sorted(
        {s.algo for s in schemes if s.role.startswith("draft.") and s.role.endswith("moe.routed")}
    )
    ple = sorted({s.algo for s in schemes if s.role.endswith("ple.table")})
    parts = [f"{s.role}={algo_tag(s.algo, s.group_size)}" for s in schemes]
    if quant_algo == "MIXED_PRECISION":
        parts.append("*=bf16")
    return Nvfp4FormatKey(
        flavour=FLAVOUR_MODELOPT,
        quant_algo=quant_algo,
        producer=_producer_str(sections.producer),
        group_size=groups[0] if len(groups) == 1 else None,
        activation_bits=4 if any(k[1] == ALGO_NVFP4 for k in fp4) else 16,
        kv_cache_quant_algo=(str(kv) if kv else None),
        schemes=schemes,
        draft_algo=",".join(draft) or None,
        ple_algo=",".join(ple) or None,
        inconsistencies=tuple(_diff_quantized_layers(cfg, hf)),
        key=f"{FLAVOUR_MODELOPT}:{quant_algo or '?'}|" + ";".join(parts),
    )


def _detect_compressed_tensors(
    cfg: Mapping, sections: QuantSections
) -> Optional[Nvfp4FormatKey]:
    """compressed-tensors: an NVFP4 group is ``weights: {num_bits: 4, type:
    float}``; the activation group decides W4A4 vs W4A16."""
    fmt = str(cfg.get("format") or "").lower()
    uses: Dict[Tuple[str, str, Optional[int]], int] = {}
    for group in (cfg.get("config_groups") or {}).values():
        group = group or {}
        w = group.get("weights") or {}
        a = group.get("input_activations") or {}
        try:
            wbits, abits = int(w.get("num_bits") or 0), int(a.get("num_bits") or 0)
        except (TypeError, ValueError):
            continue
        wtype = str(w.get("type") or "").lower()
        if wbits == 4 and wtype == "float":
            algo = ALGO_NVFP4 if abits == 4 else ALGO_NVFP4_A16
        elif wbits == 8 and wtype == "float":
            algo = ALGO_FP8_BLOCK if w.get("block_structure") else ALGO_FP8
        elif wbits:
            algo = f"INT{wbits}"
        else:
            continue
        g = _group_or_none(w.get("group_size"))
        for target in group.get("targets") or ["*"]:
            t = str(target)
            role = module_role(t) if "." in t else t
            uses[(role, algo, g)] = uses.get((role, algo, g), 0) + 1
    fp4 = [k for k in uses if k[1] in (ALGO_NVFP4, ALGO_NVFP4_A16)]
    if not fp4:
        return None
    groups = sorted({k[2] for k in fp4 if k[2]})
    schemes = tuple(
        SchemeUse(role=r, algo=a, group_size=g, modules=n)
        for (r, a, g), n in sorted(uses.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    )
    return Nvfp4FormatKey(
        flavour=FLAVOUR_COMPRESSED_TENSORS,
        quant_algo=fmt,
        producer=_producer_str(sections.producer),
        group_size=groups[0] if len(groups) == 1 else None,
        activation_bits=4 if any(k[1] == ALGO_NVFP4 for k in fp4) else 16,
        kv_cache_quant_algo=None,
        schemes=schemes,
        draft_algo=None,
        ple_algo=None,
        inconsistencies=(),
        key=f"{FLAVOUR_COMPRESSED_TENSORS}:{fmt or '?'}|"
        + ";".join(f"{s.role}={algo_tag(s.algo, s.group_size)}" for s in schemes),
    )


# ---------------------------------------------------------------------------
# Safetensors headers (metadata only)
# ---------------------------------------------------------------------------


class IncompleteShard(RuntimeError):
    """A shard is shorter than its own header says: a download in flight or a
    truncated copy.  Refused -- a half file is never a finding."""


def _read_header(path: str, verify_complete: bool) -> Tuple[Dict[str, dict], int]:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            raise IncompleteShard(f"{path}: {size} bytes, no header length")
        n = struct.unpack("<Q", head)[0]
        raw = fh.read(n)
    if len(raw) < n:
        raise IncompleteShard(f"{path}: header of {n} bytes truncated at {len(raw)}")
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    if verify_complete:
        end = max((int(m["data_offsets"][1]) for m in hdr.values()), default=0)
        need = 8 + n + end
        if size < need:
            raise IncompleteShard(f"{path}: {size} bytes on disk, header needs {need}")
    return hdr, n


def read_safetensors_header(path: str, *, verify_complete: bool = True) -> Dict[str, dict]:
    """The JSON header of one ``.safetensors`` file (``__metadata__`` dropped).

    With ``verify_complete`` the file must reach ``8 + header_len +
    max(data_offsets end)``, else :class:`IncompleteShard`."""
    return _read_header(path, verify_complete)[0]


@dataclass
class TensorMeta:
    name: str
    file: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    #: absolute byte offset of the tensor data inside ``file``
    offset: int


def read_f32_scalar(path: str, meta: TensorMeta) -> float:
    """ONE F32 scalar tensor (4 bytes) -- the only tensor bytes this module
    ever reads, and only on request."""
    if meta.dtype != "F32" or meta.nbytes != 4:
        raise ValueError(f"not an F32 scalar: {meta}")
    with open(path, "rb") as fh:
        fh.seek(meta.offset)
        return struct.unpack("<f", fh.read(4))[0]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

#: Param suffixes, longest first so ``weight_scale_2`` is not read as
#: ``weight_scale`` + ``_2``.
_PARAM_SUFFIXES = (
    "weight_scale_inv",
    "weight_scale_2",
    "weight_scale",
    "input_scale_inv",
    "input_scale",
    "weight_global_scale",
    "input_global_scale",
    "weight_packed",
    "weight_shape",
    "weight",
    "bias",
)
_SHARD_RE = re.compile(r"^(?P<module>.+)\.shard_(?P<idx>\d+)\.weight$")
_EXPERT_RE = re.compile(
    r"^(?P<layer>.+\.layers\.(?P<lid>\d+))\..*\.experts\.(?P<eid>\d+)\.(?P<proj>[^.]+)$"
)


def split_param(name: str) -> Tuple[str, str]:
    """``module.path.<param>`` -> (module, param).  A PLE table shard
    ``...ngram_embedding.shard_7.weight`` is module ``...ngram_embedding``,
    param ``shard``."""
    m = _SHARD_RE.match(name)
    if m:
        return m.group("module"), "shard"
    for suf in _PARAM_SUFFIXES:
        if name.endswith("." + suf):
            return name[: -len(suf) - 1], suf
    head, _, last = name.rpartition(".")
    return (head, last) if head else (name, name)


def _glob_match(patterns: Sequence[str], module: str) -> bool:
    """ModelOpt ``exclude_modules`` / ``ignore`` are fnmatch-style globs on the
    CHECKPOINT module path; a module is excluded when it or an ancestor
    matches."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        for pat in patterns:
            if fnmatch.fnmatchcase(cand, pat):
                return True
    return False


def declared_algo(
    module: str,
    quantized_layers: Mapping[str, Mapping],
    exclude: Sequence[str],
    whole_model_algo: Optional[str] = None,
) -> Tuple[str, Optional[int], str]:
    """(normalized algo, group size, source) the CONFIG declares for a
    checkpoint module.  The first ancestor listed in ``quantized_layers`` wins
    (ModelOpt keys fused routed experts by their parent ``...mlp.experts``)."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        info = quantized_layers.get(cand)
        if info is not None:
            info = info or {}
            return (
                normalize_algo(info.get("quant_algo")),
                _group_or_none(info.get("group_size")),
                cand,
            )
    if _glob_match(exclude, module):
        return ALGO_BF16, None, "excluded"
    if whole_model_algo:
        return whole_model_algo, None, "whole-model"
    return ALGO_BF16, None, "unlisted"


def observed_algo(params: Mapping[str, TensorMeta]) -> str:
    """The algorithm a module's TENSORS carry, from dtypes and companions."""
    w = params.get("weight") or params.get("shard")
    if w is None:
        return "CT_PACKED" if "weight_packed" in params else ALGO_BUFFER
    if w.dtype == "U8" and "weight_scale" in params:
        ws = params["weight_scale"]
        if ws.dtype == "F8_E4M3" and "weight_scale_2" in params:
            return ALGO_NVFP4 if "input_scale" in params else ALGO_NVFP4_A16
        return f"UNKNOWN:U8+{ws.dtype}"
    if w.dtype in ("F8_E4M3", "F8_E5M2"):
        if "weight_scale_inv" in params:
            return ALGO_FP8_BLOCK
        if "weight_scale" in params:
            return ALGO_FP8
        return f"UNKNOWN:{w.dtype}-unscaled"
    if w.dtype in ("BF16", "F16", "F32"):
        return ALGO_BF16
    return f"UNKNOWN:{w.dtype}"


def _scalar_like(shape: Sequence[int]) -> bool:
    return tuple(shape) in ((), (1,))


def check_shapes(
    module: str, algo: str, group: Optional[int], params: Mapping[str, TensorMeta]
) -> List[str]:
    """Shape/dtype contract per algorithm; returns human-readable violations."""
    bad: List[str] = []
    if algo in (ALGO_NVFP4, ALGO_NVFP4_A16):
        g = group or 16
        w, s = params.get("weight"), params.get("weight_scale")
        if w is None or s is None:
            return [f"{module}: {algo} without weight/weight_scale"]
        if len(w.shape) != 2 or len(s.shape) != 2:
            return [f"{module}: {algo} weight {w.shape} / weight_scale {s.shape} not 2-D"]
        n, k = w.shape[0], 2 * w.shape[1]
        if k % g or s.shape != (n, k // g):
            bad.append(f"{module}: weight_scale {s.shape} != ({n}, {k}//{g}) for weight {w.shape}")
        if s.dtype != "F8_E4M3":
            bad.append(f"{module}: weight_scale dtype {s.dtype} != F8_E4M3")
        s2 = params.get("weight_scale_2")
        if s2 is None or not _scalar_like(s2.shape) or s2.dtype != "F32":
            bad.append(f"{module}: weight_scale_2 missing or not an F32 scalar")
        if algo == ALGO_NVFP4:
            i = params.get("input_scale")
            if i is None or not _scalar_like(i.shape):
                bad.append(f"{module}: W4A4 without a scalar input_scale")
    elif algo == ALGO_FP8_BLOCK:
        b = group or 128
        w, s = params.get("weight"), params.get("weight_scale_inv")
        if w is None or s is None or len(w.shape) != 2:
            return [f"{module}: FP8 block without 2-D weight / weight_scale_inv"]
        want = (-(-w.shape[0] // b), -(-w.shape[1] // b))
        if s.shape != want:
            bad.append(
                f"{module}: weight_scale_inv {s.shape} != {want} for weight {w.shape} block {b}"
            )
    elif algo == ALGO_FP8:
        s = params.get("weight_scale")
        if s is None or not _scalar_like(s.shape):
            bad.append(f"{module}: per-tensor FP8 without a scalar weight_scale")
    return bad


@dataclass
class ExpertRow:
    """Routed-expert bytes for ONE (layer, expert): the unit the expert pool,
    the host store and the Platztausch move."""

    algo: str
    layers: int
    experts_per_layer: int
    #: bytes of one expert in one layer as stored in the checkpoint
    ckpt_bytes: int
    #: the same expert after the Marlin repack of the W4A16 path
    #: (``marlin_utils_fp4.prepare_moe_nvfp4_layer_for_marlin``): packed E2M1
    #: and the E4M3 group scales keep their byte count, ``input_scale`` is not
    #: per slot, the global scales become one 16-bit value each for w13
    #: (gate == up collapsed) and for w2.  ``None`` for non-NVFP4 experts.
    marlin_slot_bytes: Optional[int]
    uniform: bool
    #: (equal, checked): gate vs up ``weight_scale_2`` -- Marlin keeps ONE
    #: w13 global scale, so anything but equal costs accuracy.
    gate_up_scale2_equal: Optional[Tuple[int, int]] = None

    @property
    def row_mib_all_layers(self) -> float:
        per = self.marlin_slot_bytes or self.ckpt_bytes
        return per * self.layers / 2**20

    def line(self) -> str:
        eq = (
            f" gate_up_scale2_equal={self.gate_up_scale2_equal[0]}/{self.gate_up_scale2_equal[1]}"
            if self.gate_up_scale2_equal
            else ""
        )
        return (
            f"{ROW_LINE_PREFIX} algo={self.algo} layers={self.layers} "
            f"experts_per_layer={self.experts_per_layer} ckpt_bytes={self.ckpt_bytes} "
            f"marlin_slot_bytes={self.marlin_slot_bytes} uniform={self.uniform} "
            f"row_all_layers_mib={self.row_mib_all_layers:.2f}{eq}"
        )


@dataclass
class MixedPrecisionPlan:
    fmt: Optional[Nvfp4FormatKey]
    files: int
    tensors: int
    modules: int
    total_bytes: int
    #: (role, declared algo) -> [modules, bytes]
    census: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    expert_row: Optional[ExpertRow] = None
    draft_expert_row: Optional[ExpertRow] = None
    #: sglang module prefix -> declared algo (what the loader resolves against)
    loader_view: Dict[str, str] = field(default_factory=dict)

    def bytes_for(self, role: str, algo: Optional[str] = None) -> int:
        return sum(
            b for (r, a), (_n, b) in self.census.items() if r == role and (algo is None or a == algo)
        )

    def lines(self) -> List[str]:
        out = []
        if self.fmt is not None:
            out.append(self.fmt.line())
            for inc in self.fmt.inconsistencies:
                out.append(f"{FORMAT_LINE_PREFIX} inconsistency {inc}")
        out.append(
            f"{PLAN_LINE_PREFIX} files={self.files} tensors={self.tensors} "
            f"modules={self.modules} bytes={self.total_bytes} "
            f"({self.total_bytes / 2**30:.2f} GiB) anomalies={len(self.anomalies)}"
        )
        for (role, algo), (n, b) in sorted(self.census.items(), key=lambda kv: -kv[1][1]):
            out.append(
                f"{PLAN_LINE_PREFIX} role={role} algo={algo} modules={n} "
                f"bytes={b} ({b / 2**30:.3f} GiB)"
            )
        for row in (self.expert_row, self.draft_expert_row):
            if row is not None:
                out.append(row.line())
        for a in self.anomalies[:50]:
            out.append(f"{PLAN_LINE_PREFIX} ANOMALY {a}")
        if len(self.anomalies) > 50:
            out.append(f"{PLAN_LINE_PREFIX} ANOMALY ... {len(self.anomalies) - 50} more")
        return out


#: Checkpoint projection names that sglang loads into ONE fused module.
_STACKED = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "in_proj_qkv": "in_proj_qkvz",
    "in_proj_z": "in_proj_qkvz",
    "in_proj_b": "in_proj_ba",
    "in_proj_a": "in_proj_ba",
}


def sglang_module_prefix(module: str) -> str:
    """The prefix sglang's loader builds for a checkpoint module: the VL text
    stack ``model.language_model.*`` -> ``model.*``, routed experts -> their
    fused ``FusedMoE`` parent, stacked projections -> the fused name.  Only what
    ``ModelOpt*Config.get_quant_method`` resolves against."""
    m = module
    if m.startswith("model.language_model."):
        m = "model." + m[len("model.language_model."):]
    ex = re.match(r"^(.*\.experts)\.\d+\.[^.]+$", m)
    if ex:
        return ex.group(1)
    head, _, last = m.rpartition(".")
    if head and last in _STACKED:
        return f"{head}.{_STACKED[last]}"
    return m


def _expert_rows(
    modules: Mapping[str, Mapping[str, TensorMeta]],
    declared: Mapping[str, Tuple[str, Optional[int], str]],
    draft: bool,
    scalar_reader: Optional[Callable[[TensorMeta], float]],
) -> Optional[ExpertRow]:
    want_role = "draft.moe.routed" if draft else "moe.routed"
    per: Dict[Tuple[str, str], int] = {}
    per_algo: Dict[Tuple[str, str], str] = {}
    marlin: Dict[Tuple[str, str], int] = {}
    s2: Dict[Tuple[str, str], Dict[str, TensorMeta]] = {}
    for module, params in modules.items():
        if module_role(module) != want_role:
            continue
        m = _EXPERT_RE.match(module)
        if not m:
            continue
        key = (m.group("layer"), m.group("eid"))
        per[key] = per.get(key, 0) + sum(t.nbytes for t in params.values())
        algo = declared[module][0]
        per_algo[key] = algo
        if algo in (ALGO_NVFP4, ALGO_NVFP4_A16):
            body = sum(params[p].nbytes for p in ("weight", "weight_scale") if p in params)
            marlin[key] = marlin.get(key, 0) + body
            if "weight_scale_2" in params:
                s2.setdefault(key, {})[m.group("proj")] = params["weight_scale_2"]
    if not per:
        return None
    sizes = set(per.values())
    algos = sorted(set(per_algo.values()))
    experts: Dict[str, int] = {}
    for lyr, _eid in per:
        experts[lyr] = experts.get(lyr, 0) + 1
    marlin_bytes = None
    if marlin:
        # two 16-bit global scales per expert slot: w13 (gate == up) and w2
        vals = {v + 2 * 2 for v in marlin.values()}
        marlin_bytes = vals.pop() if len(vals) == 1 else None
    eq = None
    if scalar_reader is not None and s2:
        n_eq = n_all = 0
        for projs in s2.values():
            if "gate_proj" in projs and "up_proj" in projs:
                n_all += 1
                n_eq += int(scalar_reader(projs["gate_proj"]) == scalar_reader(projs["up_proj"]))
        eq = (n_eq, n_all)
    return ExpertRow(
        algo=",".join(algos),
        layers=len(experts),
        experts_per_layer=max(experts.values()),
        ckpt_bytes=max(sizes),
        marlin_slot_bytes=marlin_bytes,
        uniform=len(sizes) == 1 and len(set(experts.values())) == 1,
        gate_up_scale2_equal=eq,
    )


def plan_from_checkpoint(
    model_dir: str,
    *,
    read_scalars: bool = False,
    verify_complete: bool = True,
    resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> MixedPrecisionPlan:
    """Dry run of the mixed-precision load plan over the index and headers.

    ``resolver`` optionally maps an sglang module prefix to the raw algorithm
    the REAL quant config resolves (e.g.
    ``ModelOptMixedPrecisionConfig.from_config(...).resolve_quant_algo``);
    every prefix where it disagrees with this dry run is an anomaly -- the
    drift guard between the dry run and the loader."""
    sections = read_quant_sections(model_dir)
    fmt = detect_nvfp4_format(sections=sections)
    cfg, hf = sections.config or {}, sections.hf or {}
    quantized_layers = _layers_map(cfg) or _layers_map(hf)
    exclude = list(cfg.get("ignore") or hf.get("exclude_modules") or [])
    qa = str(cfg.get("quant_algo") or hf.get("quant_algo") or "").upper()
    whole = None if qa in ("", "MIXED_PRECISION") else normalize_algo(qa)

    idx = _load_json(os.path.join(model_dir, "model.safetensors.index.json"))
    if idx is None:
        single = "model.safetensors"
        if not os.path.exists(os.path.join(model_dir, single)):
            raise FileNotFoundError(f"{model_dir}: no index and no {single}")
        hdr = read_safetensors_header(os.path.join(model_dir, single), verify_complete=verify_complete)
        weight_map = {n: single for n in hdr}
    else:
        weight_map = dict(idx["weight_map"])

    by_file: Dict[str, List[str]] = {}
    for name, f in weight_map.items():
        by_file.setdefault(f, []).append(name)

    metas: Dict[str, TensorMeta] = {}
    anomalies: List[str] = []
    for f in sorted(by_file):
        hdr, n = _read_header(os.path.join(model_dir, f), verify_complete)
        wanted = set(by_file[f])
        for name in by_file[f]:
            meta = hdr.get(name)
            if meta is None:
                anomalies.append(f"{name}: in the index for {f} but not in its header")
                continue
            a, b = (int(x) for x in meta["data_offsets"])
            metas[name] = TensorMeta(
                name, f, meta["dtype"], tuple(int(x) for x in meta["shape"]), b - a, 8 + n + a
            )
        extra = len(set(hdr) - wanted)
        if extra:
            anomalies.append(f"{f}: {extra} header tensors missing from the index")
        del hdr

    modules: Dict[str, Dict[str, TensorMeta]] = {}
    shards: Dict[str, List[TensorMeta]] = {}
    for name, meta in metas.items():
        mod, param = split_param(name)
        if param == "shard":
            shards.setdefault(mod, []).append(meta)
            modules.setdefault(mod, {}).setdefault("shard", meta)
        else:
            modules.setdefault(mod, {})[param] = meta

    declared: Dict[str, Tuple[str, Optional[int], str]] = {}
    census: Dict[Tuple[str, str], List[int]] = {}
    loader_view: Dict[str, str] = {}
    for mod, params in modules.items():
        algo, group, source = declared_algo(mod, quantized_layers, exclude, whole)
        seen = observed_algo(params)
        if source == "whole-model" and seen in (ALGO_BF16, ALGO_BUFFER):
            # a whole-model export cannot name its unquantized modules
            algo, group, source = ALGO_BF16, None, "whole-model:not-quantized"
        declared[mod] = (algo, group, source)
        if seen != ALGO_BUFFER and seen != algo:
            anomalies.append(f"{mod}: declared {algo} ({source}) but tensors carry {seen}")
        elif seen == algo:
            anomalies.extend(check_shapes(mod, algo, group, params))
        nbytes = sum(t.nbytes for p, t in params.items() if p != "shard")
        tbl = shards.get(mod)
        if tbl:
            nbytes += sum(t.nbytes for t in tbl)
            dtypes = {t.dtype for t in tbl}
            if len(dtypes) != 1:
                anomalies.append(f"{mod}: table shards mix dtypes {sorted(dtypes)}")
        cell = census.setdefault((module_role(mod), algo), [0, 0])
        cell[0] += 1
        cell[1] += nbytes
        if seen != ALGO_BUFFER:
            loader_view.setdefault(sglang_module_prefix(mod), algo)

    if resolver is not None:
        for prefix, algo in sorted(loader_view.items()):
            raw = resolver(prefix)
            got = normalize_algo(raw) if raw else ALGO_BF16
            if got != algo:
                anomalies.append(f"loader drift: {prefix}: dry run {algo}, quant config {got}")

    reader: Optional[Callable[[TensorMeta], float]] = None
    if read_scalars:
        def reader(meta: TensorMeta) -> float:  # noqa: E306
            return read_f32_scalar(os.path.join(model_dir, meta.file), meta)

    return MixedPrecisionPlan(
        fmt=fmt,
        files=len(by_file),
        tensors=len(metas),
        modules=len(modules),
        total_bytes=sum(t.nbytes for t in metas.values()),
        census=census,
        anomalies=anomalies,
        expert_row=_expert_rows(modules, declared, False, reader),
        draft_expert_row=_expert_rows(modules, declared, True, None),
        loader_view=loader_view,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NVFP4 format key + metadata dry run")
    ap.add_argument("model_dir")
    ap.add_argument("--scalars", action="store_true", help="read gate/up weight_scale_2 (4 bytes each)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--format-only", action="store_true", help="only the format key (JSON files)")
    args = ap.parse_args(argv)
    if args.format_only:
        fmt = detect_nvfp4_format(args.model_dir)
        print(fmt.line() if fmt else f"{FORMAT_LINE_PREFIX} key=none (no NVFP4 module declared)")
        for inc in fmt.inconsistencies if fmt else ():
            print(f"{FORMAT_LINE_PREFIX} inconsistency {inc}")
        return 0 if fmt else 1
    plan = plan_from_checkpoint(args.model_dir, read_scalars=args.scalars)
    if args.json:
        payload = asdict(plan)
        payload["census"] = {f"{r}|{a}": v for (r, a), v in plan.census.items()}
        payload.pop("loader_view", None)
        json.dump(payload, sys.stdout, indent=1, default=str)
        print()
    else:
        print("\n".join(plan.lines()))
    return 0 if not plan.anomalies else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
