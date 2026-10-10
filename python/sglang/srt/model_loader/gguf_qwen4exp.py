# SPDX-License-Identifier: Apache-2.0
"""GGUF loading support for Qwen3.8-Flash-Next (GGUF arch ``qwen4exp``).

G1 of the NF-GGUF migration (deskq/PLAN-GGUF-NF-1009.md, section 2 row G1).
Template: ``gguf_qwen35.py`` (the GDN math is reused through subclassing, the
qwen4exp-specific tensors are handled here).

Why a bespoke table. The installed gguf-py (0.19.0) does not know the arch
``qwen4exp`` (``gguf.MODEL_ARCH_NAMES`` has no entry), so
``GGUFAdapterBase.build_name_map`` -- which asks ``gguf.get_tensor_name_map`` --
refuses (``gguf_adapter_base.py:174-180``). The table below is therefore copied,
not looked up. It is the GGUF -> HF direction of llama.cpp's converter
``conversion/qwen4exp.py`` plus ``gguf-py/gguf/tensor_mapping.py`` /
``constants.py``.

Source (all fetched 2026-10-09; llama.cpp master commit
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, PR #27742 "qwen4exp", b10660):

* ``conversion/qwen4exp.py``  227 lines: class ``Qwen4ExpTextModel`` L18-L221
  (``modify_tensors`` L130-L171, ``set_gguf_parameters`` L72-L115,
  ``_place_ple_shard`` L175-L202).
* ``conversion/qwen.py``  ``Qwen3NextModel.modify_tensors`` L395-L435
  (``A_log -> -exp``, ``dt_bias -> dt_proj.bias``, ``conv1d`` squeeze,
  ``*norm.weight`` ``+1`` except ``linear_attn.norm``),
  ``_LinearAttentionVReorderBase`` L454-L630 (V-head grouped -> tiled reorder).
* ``gguf-py/gguf/tensor_mapping.py`` ``MODEL_ARCH.QWEN4EXP`` block L2897-L2962
  (HC, indexer norms, PLE) and the arch-independent rows quoted per entry below.
* ``gguf-py/gguf/constants.py`` ``MODEL_TENSORS[QWEN4EXP]`` L3057-L3113, name
  strings L1466-L1565 and L1628-L1641, L1717-L1723.

Inverse transforms applied at load time (llama.cpp -> sglang HF layout):

=======================  =======================================================
GGUF role                 transform (converter line  ->  inverse here)
=======================  =======================================================
``ssm_a``                 ``-exp(A_log)`` (qwen.py L396-397)  ->  ``log(-x)``,
                          then V-head un-tile                       [qwen35 base]
``ssm_dt.bias``           renamed ``dt_proj.bias`` (qwen.py L398-399), V-head
                          reordered (qwen.py L610-L617)         ->  V-head un-tile
``ssm_conv1d``            ``squeeze`` + V-channel reorder (qwen.py L400-401,
                          L619-L626)  ->  ``unsqueeze(1)`` + un-tile of the V
                          channels (rows 4096..10239)               [qwen35 base]
``attn_qkv/attn_gate/ssm_alpha/ssm_beta/ssm_out``
                          V rows / columns reordered (qwen.py L594-L625)
                          ->  un-tile, byte-granular for ``ssm_out`` [qwen35 base]
``*norm.weight``          ``+1`` baked in (qwen.py L402-403 for ``q_norm``,
                          ``k_norm``, every ``hc_norm``; qwen4exp.py L163-L166
                          for ``ple.norm_{key,query,conv}`` and the indexer
                          ``q_layernorm`` / ``k_layernorm``)  ->  ``-1``
                          (every one of them is Gemma-style ``1 + w`` in the
                          sglang model: ``GroupedGemmaRMSNorm``,
                          ``Qwen4ExpPLEGroupedNorm``, ``GemmaRMSNorm``)
                          ``ssm_norm`` (``linear_attn.norm``) is NOT touched.
``indexer.q_proj`` +      one HF tensor ``index_qk_proj`` [640, 2560] split at row
``indexer.k_proj``        ``indexer_n_heads * indexer_head_dim`` = 512
                          (qwen4exp.py L153-L161)  ->  fused back, q rows first,
                          then k rows
``ple_conv1d``            ``squeeze`` (qwen4exp.py L168-L169) -> ``unsqueeze(1)``
                          (NO V-head reorder: it is not a GDN conv)
``hc_*_inject``           stored as is: GGUF ne = (10240, 4) is torch [4, 10240]
                          = HF ``block_inject_weight``; only F32 -> bf16
``ffn_gate_inp``          router, F32 -> bf16 (module dtype)
``ffn_gate_inp_shexp``    [2560] -> [1, 2560] (``unsqueeze(0)``)    [qwen35 base]
``hc_*_down/up``          Q8_0 -> dequantised to bf16 (the mixers are plain
                          ``nn.Linear`` in sglang)
``per_layer_token_embd``  ONE tensor [320001536, 160] instead of the 128 HF
                          ``ngram_embedding.shard_N`` tensors (qwen4exp.py
                          L173-L202). Mapped here to
                          ``model.layers.<ple layer>.ple.ple_embedding.
                          ngram_embedding.weight`` (quantized, IQ4_NL in the
                          unsloth export); the model side (AP G2) consumes it.
=======================  =======================================================

The three int64 PLE hash constants are NOT tensors in the GGUF but UINT64 KV
arrays (``qwen4exp.ple.layer_multipliers`` / ``head_offsets`` /
``head_vocab_sizes``, converter L110-L115, L131-L140). ``transform_stream`` emits
them at the end as int64 tensors named like the HF buffers
(``...ple.ple_embedding.layer_multipliers`` / ``ngram_heads_offsets`` /
``ngram_heads_vocab_sizes``) so the model's own buffer loader
(``qwen4_exp.py _load_qwen4_exp_ple_buffer``) takes them.

PLE table consumer (AP G2). The 28.8 GB ``per_layer_token_embd`` payload must
NOT go through the generic weight stream: ``gguf_quant_weights_iterator`` copies
every mapped tensor (``torch.tensor(...)``). The loader therefore builds the
iterator from :meth:`Qwen4ExpGGUFAdapter.stream_name_map` (the name map without
the table), and :meth:`transform_stream` yields ONE small marker tensor first,
``model.layers.<ple layer>.ple.ple_embedding.ngram_embedding.gguf_table``, whose
uint8 payload is the table's location (file, absolute offset, type, shape;
``qwen4_exp_ple_gguf.encode_ple_table_marker``). The model maps the span lazily
(``Qwen4ExpPinnedHostEmbedding.attach_gguf_table``). Any table format but IQ4_NL
is refused by name before a single tensor is streamed.

Not in G1: MoE/PLE kernels (AP G6).

G3 (NF-GGUF AP G3, 2026-10-10): the NEXTN/MTP draft head from its own GGUF
(``mtp-Qwen3.8-Flash-Next-{Q8_0,shared-Q8_0}.gguf``; llama.cpp PR #29761 "Qwen4Exp:
add MTP", unsloth ``nextn_shared_target_tensors`` variant). The draft block
``blk.<block_count - nextn_predict_layers>`` (48 for the real files) carries the
SAME roles as a full-attention backbone layer (attention, QSA indexer, hyper
connections, MoE; no GDN, no PLE) plus six ``nextn.*`` tensors. HF side is the
albucino/Minachist draft layout (``mtp.layers.0.*``, ``mtp.fc_embedding``,
``mtp.fc_hidden``, ``mtp.pre_fc_norm_*``, ``mtp.hyper_connection_mixer.*``;
checked against the safetensors header of the albucino MTP draft). Roles:

=======================  =======================================================
GGUF role (blk.<N>.)      HF name / transform
=======================  =======================================================
``nextn.eh_proj``         [2H, H] in ggml order = torch [H, 2H]. The converter
                          fuses the two HF projections (conversion/qwen4exp.py,
                          ``modify_tensors``): ``eh = torch.cat([fc_embedding,
                          fc_hidden], dim=1)``, comment ``eh_proj([e ; h_s]) =
                          fc_embedding(e) + fc_hidden(h_s) for every hc stream
                          s``  ->  columns [:H] = ``mtp.fc_embedding``, columns
                          [H:] = ``mtp.fc_hidden`` (dequantised first: both are
                          plain ``nn.Linear`` in the draft)
``nextn.enorm``           ``mtp.pre_fc_norm_embedding`` ([H]); Gemma gamma, ``-1``
``nextn.hnorm``           ``mtp.pre_fc_norm_hidden`` ([hc*H]); Gemma gamma, ``-1``
``nextn.hc_head_norm``    ``mtp.hyper_connection_mixer.hc_norm``; ``-1``
``nextn.hc_head_down``    ``mtp.hyper_connection_mixer.input_mix_weight_down``
``nextn.hc_head_up``      ``mtp.hyper_connection_mixer.input_mix_weight_up``
all backbone-layer roles  ``mtp.layers.0.<same HF suffix as LAYER_TABLE>``; the
                          ``-1`` of every ``*norm.weight`` follows from qwen.py
                          ``modify_tensors`` (``endswith("norm.weight")``, the
                          MTP names are renamed to enorm/hnorm/layers.N BEFORE it
                          runs: ``_QwenMtpMixin.filter_tensors``)
``token_embd/output``     NOT loaded (full variant only): the draft shares the
                          target's vocabulary modules (eagle_worker_v2.
                          ``init_lm_head`` -> ``set_embed_and_head_modules`` for a
                          quantised-resident target, tensor share otherwise), and
                          ``Qwen3_5ForCausalLMMTP.load_weights`` has never read a
                          draft's own vocabulary tensors. The shared variant has
                          none; the full variant's two Q8_0 tables are skipped
                          unread (the iterator touches only mapped tensors)
=======================  =======================================================

The hc_*_inject / hc_*_down / hc_*_up tensors are Q8_0 in the draft (F32 in the
main model); the same ``hc_dense`` dequantisation as the main model applies.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.model_loader.gguf_qwen35 import Qwen35GGUFAdapter

logger = logging.getLogger(__name__)

QWEN4EXP_GGUF_ARCH = "qwen4exp"

#: HF ``model_type`` of the wrapper / the text config -> GGUF arch.
_MODEL_TYPE_TO_GGUF_ARCH = {
    "qwen4_exp": QWEN4EXP_GGUF_ARCH,
    "qwen4_exp_text": QWEN4EXP_GGUF_ARCH,
}

# ---------------------------------------------------------------------------
# The name table (47 roles). (gguf name, HF name, llama.cpp source lines)
#
# Source column: "tm" = gguf-py/gguf/tensor_mapping.py, "cs" = constants.py,
# "cv" = conversion/qwen4exp.py, all llama.cpp master 79e2e74eb110. The HF
# spelling is the one sglang's Qwen4ExpForConditionalGeneration.load_weights
# resolves (``.self_attn`` is stripped there, ``model.language_model.`` is
# optional).
# ---------------------------------------------------------------------------

#: Global tensors: gguf name -> HF name.
GLOBAL_TABLE: Tuple[Tuple[str, str], ...] = (
    ("token_embd.weight", "model.embed_tokens.weight"),  # tm 16, cs 1466
    ("output.weight", "lm_head.weight"),  # tm 81, cs 1473
    ("output_hc_norm.weight", "model.hyper_connection_mixer.hc_norm.weight"),  # tm 2922, cs 1480
    (
        "output_hc_down.weight",
        "model.hyper_connection_mixer.input_mix_weight_down.weight",
    ),  # tm 2925, cs 1481
    (
        "output_hc_up.weight",
        "model.hyper_connection_mixer.input_mix_weight_up.weight",
    ),  # tm 2928, cs 1482
    # not in tm: the converter writes it under TENSOR_NAMES[PER_LAYER_TOKEN_EMBD]
    # (cv 201-202, cs 1530); the HF name is filled in per file (the PLE layer)
    ("per_layer_token_embd.weight", "<ple>"),
)

#: Per-layer tensors: gguf suffix after ``blk.N.`` -> HF suffix after
#: ``model.layers.N.``.
LAYER_TABLE: Tuple[Tuple[str, str], ...] = (
    # --- GDN linear attention (36 layers) -------------------------------
    ("attn_qkv.weight", "linear_attn.in_proj_qkv.weight"),  # tm 256, cs 1488
    ("attn_gate.weight", "linear_attn.in_proj_z.weight"),  # tm 400, cs 1495
    ("ssm_alpha.weight", "linear_attn.in_proj_a.weight"),  # tm 923, cs 1557
    ("ssm_beta.weight", "linear_attn.in_proj_b.weight"),  # tm 951, cs 1564
    ("ssm_a", "linear_attn.A_log"),  # tm 879, cs 1551 (bare param)
    ("ssm_dt.bias", "linear_attn.dt_bias"),  # tm 862 (dt_proj), cs 1549 (bare param)
    ("ssm_conv1d.weight", "linear_attn.conv1d.weight"),  # tm 846, cs 1547
    ("ssm_norm.weight", "linear_attn.norm.weight"),  # tm 906, cs 1555
    ("ssm_out.weight", "linear_attn.out_proj.weight"),  # tm 917, cs 1556
    # --- full attention (12 layers) --------------------------------------
    ("attn_q.weight", "self_attn.q_proj.weight"),  # tm 263, cs 1489 ([q|gate] per head)
    ("attn_k.weight", "self_attn.k_proj.weight"),  # tm 284, cs 1490
    ("attn_v.weight", "self_attn.v_proj.weight"),  # tm 306, cs 1491
    ("attn_output.weight", "self_attn.o_proj.weight"),  # tm 332, cs 1492
    ("attn_q_norm.weight", "self_attn.q_norm.weight"),  # tm 715, cs 1496
    ("attn_k_norm.weight", "self_attn.k_norm.weight"),  # tm 732, cs 1497
    # --- QSA indexer (12 full-attention layers) --------------------------
    # q_proj / k_proj are the two halves of ONE HF tensor (cv 153-161); the
    # ``__q`` / ``__k`` segments are internal and never reach the model
    ("indexer.q_proj.weight", "self_attn.indexer.index_qk_proj.__q.weight"),  # cs 1721, cv 159
    ("indexer.k_proj.weight", "self_attn.indexer.index_qk_proj.__k.weight"),  # cs 1722, cv 160
    ("indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight"),  # tm 2941, cs 1723
    ("indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight"),  # tm 2944, cs 1717
    # --- hyper connections (every layer) ---------------------------------
    ("hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight"),  # tm 2898, cs 1628
    ("hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight"),  # tm 2901, cs 1629
    ("hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight"),  # tm 2904, cs 1630
    ("hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight"),  # tm 2907, cs 1631
    ("hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight"),  # tm 2910, cs 1632
    ("hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight"),  # tm 2913, cs 1633
    ("hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight"),  # tm 2916, cs 1634
    ("hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight"),  # tm 2919, cs 1635
    # --- PLE (one layer) --------------------------------------------------
    ("ple_key.weight", "ple.key_proj.weight"),  # tm 2947, cs 1636
    ("ple_value.weight", "ple.value_proj.weight"),  # tm 2950, cs 1637
    ("ple_conv1d.weight", "ple.conv1d.weight"),  # tm 2962, cs 1641
    ("ple_norm_conv.weight", "ple.norm_conv.weight"),  # tm 2959, cs 1640
    ("ple_norm_key.weight", "ple.norm_key.weight"),  # tm 2953, cs 1638
    ("ple_norm_query.weight", "ple.norm_query.weight"),  # tm 2956, cs 1639
    # --- MoE (every layer) ------------------------------------------------
    ("ffn_gate_inp.weight", "mlp.gate.weight"),  # tm 469, cs 1500
    ("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"),  # tm 488, cs 1501
    ("ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"),  # tm 615, cs 1511
    ("ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"),  # tm 562, cs 1513
    ("ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"),  # tm 695, cs 1512
    # stacked [E, out, in] tensors: split per expert by the generic GGUF
    # iterator (weight_utils.gguf_quant_weights_iterator), which does not use
    # this name; the collective HF names only satisfy the name-map audit
    ("ffn_gate_exps.weight", "mlp.experts.gate_proj.weight"),  # tm 607, cs 1519
    ("ffn_up_exps.weight", "mlp.experts.up_proj.weight"),  # tm 553, cs 1521
    ("ffn_down_exps.weight", "mlp.experts.down_proj.weight"),  # tm 684, cs 1520
)

#: The 47 tensor roles of the format: 6 global + 41 per layer.
QWEN4EXP_ROLES: Tuple[str, ...] = tuple(g for g, _ in GLOBAL_TABLE) + tuple(
    s for s, _ in LAYER_TABLE
)

_GDN_ROLES = frozenset(
    {
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_a",
        "ssm_dt.bias",
        "ssm_conv1d.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
    }
)
_ATTN_ROLES = frozenset(
    {
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "indexer.q_proj.weight",
        "indexer.k_proj.weight",
        "indexer.q_norm.weight",
        "indexer.k_norm.weight",
    }
)
_PLE_LAYER_ROLES = frozenset(s for s, _ in LAYER_TABLE if s.startswith("ple_"))
_EVERY_LAYER_ROLES = frozenset(
    s
    for s, _ in LAYER_TABLE
    if s not in _GDN_ROLES and s not in _ATTN_ROLES and s not in _PLE_LAYER_ROLES
)

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

#: HF buffer names of the PLE hash constants and the KV key each one comes from
#: (converter L110-L115). All three are UINT64 arrays in the GGUF.
_PLE_CONSTANTS: Tuple[Tuple[str, str], ...] = (
    ("layer_multipliers", "ple.layer_multipliers"),
    ("ngram_heads_offsets", "ple.head_offsets"),
    ("ngram_heads_vocab_sizes", "ple.head_vocab_sizes"),
)

#: Marks a tensor the qwen4exp pre-stage already finished, so the inherited
#: qwen35 stream (which matches by NAME SUFFIX, e.g. ``conv1d.weight``) leaves
#: it alone. Stripped again before the loader sees it.
_FINAL = "\x00final"

# the marker leaf and the table's GGUF name live with the table code
from sglang.srt.models.qwen4_exp_ple_gguf import (  # noqa: E402
    PLE_TABLE_GGUF_NAME,
    PLE_TABLE_MARKER_LEAF,
)


def _split_blk(name: str) -> Optional[Tuple[int, str]]:
    m = _BLK_RE.match(name)
    return (int(m.group(1)), m.group(2)) if m else None


# ---------------------------------------------------------------------------
# Pure helpers (header level; no tensor data)
# ---------------------------------------------------------------------------


def build_qwen4exp_name_map(
    file_tensors: Iterable[str],
    num_layers: int,
    ple_layer: Optional[int],
) -> Tuple[Dict[str, str], List[str]]:
    """``({gguf name: HF name}, [file tensors the table does not know])``.

    Only names present in ``file_tensors`` are emitted (the layer kind falls out
    of which roles exist, as in the qwen35 adapter). Blocks at or beyond
    ``num_layers`` (an MTP block of a combined file) and vision tensors are
    neither mapped nor reported.
    """
    present = set(file_tensors)
    out: Dict[str, str] = {}
    for gname, hf in GLOBAL_TABLE:
        if gname not in present:
            continue
        if hf == "<ple>":
            if ple_layer is None:
                raise RuntimeError(
                    "qwen4exp GGUF: per_layer_token_embd.weight is present but the "
                    "file names no PLE layer (neither the KV qwen4exp.ple.layers "
                    "nor a blk.N.ple_key.weight tensor)"
                )
            hf = f"model.layers.{ple_layer}.ple.ple_embedding.ngram_embedding.weight"
        out[gname] = hf
    for layer in range(num_layers):
        for suffix, hf_suffix in LAYER_TABLE:
            gname = f"blk.{layer}.{suffix}"
            if gname in present:
                out[gname] = f"model.layers.{layer}.{hf_suffix}"
    unknown: List[str] = []
    for name in sorted(present):
        if name in out or name.startswith(("v.", "mm.")):
            continue
        parts = _split_blk(name)
        if parts is not None and parts[0] >= num_layers:
            continue  # NEXTN/MTP block: AP G3
        unknown.append(name)
    return out, unknown


# ---------------------------------------------------------------------------
# G3: the NEXTN/MTP draft head
# ---------------------------------------------------------------------------

#: Internal HF-side name of the fused ``nextn.eh_proj``; the stream splits it into
#: ``mtp.fc_embedding`` + ``mtp.fc_hidden`` (never reaches the model).
DRAFT_EH_PROJ_INTERNAL = "mtp.eh_proj"

#: ``blk.<N>.<suffix>`` -> HF name, the six NEXTN-only roles of the draft block.
DRAFT_NEXTN_TABLE: Tuple[Tuple[str, str], ...] = (
    ("nextn.eh_proj.weight", DRAFT_EH_PROJ_INTERNAL + ".weight"),  # tm 2853-2866, cv 145-151
    ("nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),  # tm 2853-2866
    ("nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),  # tm 2853-2866
    ("nextn.hc_head_norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight"),  # tm 2931-2938
    (
        "nextn.hc_head_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    ),  # tm 2931-2938
    (
        "nextn.hc_head_up.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    ),  # tm 2931-2938
)

#: HF prefix of the (single) draft decoder layer.
DRAFT_LAYER_PREFIX = "mtp.layers.0."

#: The roles a full-attention backbone layer has and the draft block carries.
_DRAFT_LAYER_ROLES = frozenset(_ATTN_ROLES | _EVERY_LAYER_ROLES)

#: Vocabulary tensors of the self-contained draft file (skipped, see module doc).
DRAFT_VOCAB_ROLES: Tuple[str, ...] = ("token_embd.weight", "output.weight")

#: Global names of a BACKBONE (combined export): never the draft's.
_BACKBONE_GLOBALS = frozenset(g for g, _ in GLOBAL_TABLE) | frozenset(DRAFT_VOCAB_ROLES)


def qwen4exp_draft_blocks(file_tensors: Iterable[str]) -> List[int]:
    """Block indices that carry ``nextn.eh_proj.weight`` (the NEXTN marker)."""
    out = set()
    for name in file_tensors:
        parts = _split_blk(name)
        if parts is not None and parts[1] == "nextn.eh_proj.weight":
            out.add(parts[0])
    return sorted(out)


def qwen4exp_draft_block_index(
    file_tensors: Iterable[str], kv: Mapping[str, Any], gguf_file: str = "<file>"
) -> int:
    """The draft block index of a qwen4exp MTP file, from the file itself.

    NOT ``num_hidden_layers``: the draft ModelConfig rewrites that to 1
    (``configs/model_config.py``), and the real files hold the block at
    ``block_count - nextn_predict_layers`` = 48.
    """
    blocks = qwen4exp_draft_blocks(file_tensors)
    nextn = kv.get("nextn_predict_layers")
    if not blocks:
        raise RuntimeError(
            f"qwen4exp GGUF MTP draft {gguf_file}: no blk.<N>.nextn.eh_proj.weight "
            "tensor -- this is not an MTP/NEXTN head (a plain backbone export has "
            "none; an MTP head is the separate mtp-*.gguf of the export)"
        )
    if len(blocks) != 1 or (nextn is not None and int(nextn) != 1):
        raise RuntimeError(
            f"qwen4exp GGUF MTP draft {gguf_file}: {len(blocks)} NEXTN block(s) "
            f"{blocks} and nextn_predict_layers={nextn}; the draft model is built "
            "with exactly one MTP layer"
        )
    block = blocks[0]
    count = kv.get("block_count")
    if count is not None and nextn is not None and block != int(count) - int(nextn):
        raise RuntimeError(
            f"qwen4exp GGUF MTP draft {gguf_file}: the NEXTN tensors are at blk."
            f"{block} but block_count={count} - nextn_predict_layers={nextn} = "
            f"{int(count) - int(nextn)}"
        )
    return block


def qwen4exp_draft_is_shared(kv: Mapping[str, Any]) -> bool:
    """``qwen4exp.nextn_shared_target_tensors`` (unsloth shared-Q8_0 export): the
    file carries no ``token_embd`` / ``output``."""
    return bool(kv.get("nextn_shared_target_tensors"))


def build_qwen4exp_draft_name_map(
    file_tensors: Iterable[str], block: int
) -> Tuple[Dict[str, str], List[str]]:
    """``({gguf name: HF name}, [file tensors the table does not know])`` for the
    draft block ``block``. Backbone blocks and the backbone globals of a combined
    export, and the vocabulary tensors of the self-contained draft, are neither
    mapped nor reported."""
    present = set(file_tensors)
    out: Dict[str, str] = {}
    for suffix, hf in DRAFT_NEXTN_TABLE:
        gname = f"blk.{block}.{suffix}"
        if gname in present:
            out[gname] = hf
    for suffix, hf_suffix in LAYER_TABLE:
        if suffix not in _DRAFT_LAYER_ROLES:
            continue
        gname = f"blk.{block}.{suffix}"
        if gname in present:
            out[gname] = DRAFT_LAYER_PREFIX + hf_suffix
    unknown: List[str] = []
    for name in sorted(present):
        if name in out or name.startswith(("v.", "mm.")) or name in _BACKBONE_GLOBALS:
            continue
        parts = _split_blk(name)
        if parts is not None and parts[0] != block:
            continue  # a backbone block of a combined export: the target's
        unknown.append(name)
    return out, unknown


def qwen4exp_draft_missing_roles(
    file_tensors: Iterable[str], block: int, kv: Mapping[str, Any]
) -> List[str]:
    """What a complete qwen4exp MTP head has and this file lacks (empty when
    complete), plus the shared-flag contradictions. A GDN tensor in the draft block
    (the draft model is a full-attention layer) is not in the draft table and is
    refused earlier as an unmapped tensor."""
    present = set(file_tensors)
    problems: List[str] = []
    want = [f"blk.{block}.{s}" for s, _ in DRAFT_NEXTN_TABLE]
    want += [f"blk.{block}.{s}" for s, _ in LAYER_TABLE if s in _DRAFT_LAYER_ROLES]
    gap = [n for n in want if n not in present]
    if gap:
        problems.append(f"blk.{block} missing {gap}")
    vocab = [n for n in DRAFT_VOCAB_ROLES if n in present]
    if qwen4exp_draft_is_shared(kv) and vocab:
        problems.append(
            f"the header says nextn_shared_target_tensors=True but the file carries {vocab}"
        )
    return problems


def qwen4exp_draft_shape_problems(
    shapes: Mapping[str, Sequence[int]], block: int, hidden: int, hc_count: int
) -> List[str]:
    """Shape facts of the NEXTN tensors in ggml dim order (``ne0`` first)."""
    problems: List[str] = []

    def shape(suffix: str) -> Optional[Tuple[int, ...]]:
        s = shapes.get(f"blk.{block}.{suffix}")
        return None if s is None else tuple(int(d) for d in s)

    for suffix, want in (
        ("nextn.eh_proj.weight", (2 * hidden, hidden)),
        ("nextn.enorm.weight", (hidden,)),
        ("nextn.hnorm.weight", (hc_count * hidden,)),
        ("nextn.hc_head_norm.weight", (hc_count * hidden,)),
    ):
        got = shape(suffix)
        if got is not None and got != want:
            problems.append(f"blk.{block}.{suffix}: ggml shape {got}, expected {want}")
    return problems


def qwen4exp_missing_roles(file_tensors: Iterable[str], num_layers: int) -> List[str]:
    """What a complete qwen4exp backbone has and this file lacks, one line each
    (empty when complete). A file that maps cleanly but misses e.g. the
    hyper-connection of one layer would otherwise load with that module
    uninitialised."""
    present = set(file_tensors)
    problems: List[str] = []
    for name in ("token_embd.weight", "output.weight") + tuple(
        g for g, _ in GLOBAL_TABLE if g.startswith("output_hc")
    ):
        if name not in present:
            problems.append(f"missing {name}")
    has_ple_table = "per_layer_token_embd.weight" in present
    for layer in range(num_layers):
        have = {
            parts[1]
            for n in present
            if (parts := _split_blk(n)) is not None and parts[0] == layer
        }
        is_gdn = "attn_qkv.weight" in have or "ssm_a" in have
        is_attn = "attn_q.weight" in have
        if is_gdn == is_attn:
            problems.append(
                f"blk.{layer}: neither (or both) a GDN and a full-attention tensor set"
            )
            continue
        want = set(_EVERY_LAYER_ROLES) | (set(_GDN_ROLES) if is_gdn else set(_ATTN_ROLES))
        gap = sorted(want - have)
        if gap:
            problems.append(f"blk.{layer}: missing {gap}")
        has_ple = bool(have & _PLE_LAYER_ROLES)
        if has_ple and (_PLE_LAYER_ROLES - have):
            problems.append(f"blk.{layer}: PLE set incomplete: {sorted(_PLE_LAYER_ROLES - have)}")
    if has_ple_table and not any(
        f"blk.{layer}.ple_key.weight" in present for layer in range(num_layers)
    ):
        problems.append("per_layer_token_embd.weight without any blk.N.ple_* tensors")
    return problems


def read_qwen4exp_kv(gguf_file: str) -> Dict[str, Any]:
    """The ``qwen4exp.*`` KV block of ``gguf_file``'s metadata part, keys without
    the ``qwen4exp.`` prefix; scalars as ``int``/``float``, arrays as lists of
    ``int``/``float`` (UINT64 stays exact). Header only."""
    import gguf

    from sglang.srt.model_loader.gguf_shards import gguf_metadata_path

    reader = gguf.GGUFReader(gguf_metadata_path(gguf_file), "r")
    out: Dict[str, Any] = {}
    prefix = QWEN4EXP_GGUF_ARCH + "."
    for key, field in reader.fields.items():
        if not key.startswith(prefix):
            continue
        try:
            value = field.contents()
        except Exception:  # noqa: BLE001 - an unreadable field is simply absent
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        out[key[len(prefix):]] = value
    return out


#: (sibling-config key, GGUF KV key without prefix, transform applied to the
#: GGUF value). The config side is read from ``text_config`` (TOP LEVEL first,
#: then ``text_config``, the server's own probe order).
_META_CHECKS: Tuple[Tuple[str, str, Any], ...] = (
    ("num_experts", "expert_count", None),
    ("num_experts_per_tok", "expert_used_count", None),
    ("moe_intermediate_size", "expert_feed_forward_length", None),
    ("shared_expert_intermediate_size", "expert_shared_feed_forward_length", None),
    ("hidden_size", "embedding_length", None),
    ("num_attention_heads", "attention.head_count", None),
    ("num_key_value_heads", "attention.head_count_kv", None),
    ("head_dim", "attention.key_length", None),
    ("hc_count", "hyper_connection.count", None),
    ("hc_lowrank", "hyper_connection.low_rank", None),
    ("indexer_n_heads", "attention.indexer.head_count", None),
    ("indexer_head_dim", "attention.indexer.key_length", None),
    ("indexer_budget", "attention.indexer.top_k", None),
    # config ple_layer_ids are 1-based, the GGUF's 0-based (converter L94)
    ("ple_layer_ids", "ple.layers", lambda v: [int(i) + 1 for i in v]),
    ("ngram_size", "ple.ngram_size", None),
    ("heads_per_ngram", "ple.heads_per_ngram", None),
    ("ple_conv_kernel_size", "ple.conv_kernel", None),
    ("linear_conv_kernel_dim", "ssm.conv_kernel", None),
    ("linear_key_head_dim", "ssm.state_size", None),
    ("linear_num_key_heads", "ssm.group_count", None),
    ("linear_num_value_heads", "ssm.time_step_rank", None),
    ("full_attention_interval", "full_attention_interval", None),
)


#: The sibling-config keys :func:`qwen4exp_meta_mismatches` reads.
META_CONFIG_KEYS: Tuple[str, ...] = tuple(c for c, _, _ in _META_CHECKS) + (
    "linear_value_head_dim",
    "num_hidden_layers",
    "layer_types",
    "indexer_compress_ratio",
    "vocab_size",
    "split_ngram_parts",
)


def qwen4exp_meta_mismatches(
    config: Mapping[str, Any],
    kv: Mapping[str, Any],
    *,
    n_blocks_backbone: Optional[int] = None,
    token_embd_rows: Optional[int] = None,
    ple_table_rows: Optional[int] = None,
) -> List[str]:
    """Every disagreement between a sibling ``config.json`` text config
    (``config``, a dict) and the GGUF header (``kv`` from
    :func:`read_qwen4exp_kv`), one line each. Empty means consistent.

    A field one side does not carry is not checked (best effort, as
    ``gguf_registry.reconcile_sibling_config``). The sibling config of this
    format is borrowed from the safetensors original, so these are exactly the
    numbers a different checkpoint would change.
    """
    out: List[str] = []

    def cfg(key: str) -> Any:
        return config.get(key)

    for ckey, kkey, fn in _META_CHECKS:
        have = cfg(ckey)
        want = kv.get(kkey)
        if have is None or want is None:
            continue
        if fn is not None:
            want = fn(want)
        if isinstance(want, list) and not isinstance(have, list):
            have = [have]
        if isinstance(have, list):
            have = [int(x) for x in have]
        if have != want:
            out.append(f"{ckey}: config.json says {have}, GGUF {kkey} says {want}")
    # inner size = value head dim * value heads
    v_dim, v_heads = cfg("linear_value_head_dim"), cfg("linear_num_value_heads")
    inner = kv.get("ssm.inner_size")
    if None not in (v_dim, v_heads, inner) and int(v_dim) * int(v_heads) != int(inner):
        out.append(
            f"linear_value_head_dim*linear_num_value_heads: config.json says "
            f"{int(v_dim) * int(v_heads)}, GGUF ssm.inner_size says {inner}"
        )
    if n_blocks_backbone is not None and cfg("num_hidden_layers") is not None:
        if int(cfg("num_hidden_layers")) != int(n_blocks_backbone):
            out.append(
                f"num_hidden_layers: config.json says {cfg('num_hidden_layers')}, "
                f"GGUF block_count (minus nextn_predict_layers) says {n_blocks_backbone}"
            )
    # layer kinds: a full-attention layer carries a QSA indexer, i.e. a nonzero
    # compress ratio in the GGUF
    ratios, kinds = kv.get("attention.compress_ratios"), cfg("layer_types")
    if isinstance(ratios, list) and isinstance(kinds, list):
        if len(ratios) >= len(kinds):
            want_kinds = ["full_attention" if r else "linear_attention" for r in ratios[: len(kinds)]]
            if want_kinds != list(kinds):
                bad = [i for i, (a, b) in enumerate(zip(want_kinds, kinds)) if a != b]
                out.append(
                    f"layer_types: config.json disagrees with the GGUF "
                    f"attention.compress_ratios at layers {bad[:8]}"
                )
        ratio_cfg = cfg("indexer_compress_ratio")
        nz = sorted({int(r) for r in ratios if r})
        if ratio_cfg is not None and nz and nz != [int(ratio_cfg)]:
            out.append(
                f"indexer_compress_ratio: config.json says {ratio_cfg}, GGUF "
                f"attention.compress_ratios says {nz}"
            )
    vocab = cfg("vocab_size")
    if vocab is not None and token_embd_rows is not None and int(vocab) != int(token_embd_rows):
        out.append(f"vocab_size: config.json says {vocab}, GGUF token_embd has {token_embd_rows} rows")
    # PLE table: the three hash arrays describe the heads, the table must hold
    # their vocabularies and split into the config's shard count
    sizes, offs = kv.get("ple.head_vocab_sizes"), kv.get("ple.head_offsets")
    if isinstance(sizes, list) and isinstance(offs, list) and len(sizes) != len(offs):
        out.append(f"ple.head_vocab_sizes has {len(sizes)} entries, ple.head_offsets {len(offs)}")
    if ple_table_rows is not None:
        if isinstance(sizes, list) and sizes and int(ple_table_rows) < sum(sizes):
            out.append(
                f"per_layer_token_embd has {ple_table_rows} rows, fewer than the sum "
                f"of ple.head_vocab_sizes ({sum(sizes)})"
            )
        parts = cfg("split_ngram_parts")
        if parts and int(ple_table_rows) % int(parts) != 0:
            out.append(
                f"per_layer_token_embd rows {ple_table_rows} are not a multiple of "
                f"split_ngram_parts={parts}"
            )
    return out


def sibling_config_text(config_path: str) -> Dict[str, Any]:
    """``text_config`` of a sibling ``config.json`` (top level for a flat one)."""
    import json

    with open(config_path) as f:
        cfg = json.load(f)
    text = cfg.get("text_config")
    return dict(text) if isinstance(text, dict) else dict(cfg)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def _hc_mixer_int8_requested() -> bool:
    v = str(os.environ.get("SGLANG_HC_MIXER_INT8", "0")).strip().lower()
    return v not in ("", "0", "off", "false")


class Qwen4ExpGGUFAdapter(Qwen35GGUFAdapter):
    """Name map + llama.cpp inverse transforms for one ``qwen4exp`` checkpoint.

    Inherits the GDN un-tiling, ``out_proj`` byte-granular un-tile, norm ``-1``
    for ``q_norm``/``k_norm`` and the F32 carve-out from the qwen35 adapter
    (same converter base class upstream, ``_LinearAttentionVReorderBase``), and
    adds what only qwen4exp has: hyper connections, the QSA indexer fusion, PLE
    and the router cast. The NEXTN/MTP draft (``is_draft``, the architecture
    ``Qwen4ExpForCausalLMMTP``) is G3: its own name map (``_build_draft_name_map``),
    the ``eh_proj`` split and the draft-block expert rename in ``_draft_pre``.
    """

    FAMILY = "qwen4exp"
    MODEL_TYPE_TO_ARCH = _MODEL_TYPE_TO_GGUF_ARCH
    # the tables above replace the gguf-py driven emit tables
    GLOBAL_ENTRIES = ()
    LAYER_ENTRIES = ()

    def _resolve_arch(self, config, text_config) -> str:
        return self.MODEL_TYPE_TO_ARCH[text_config.model_type]

    def _post_init(self, config) -> None:
        archs = getattr(config, "architectures", None) or []
        # the draft's architecture name is qwen4_exp_mtp.py:90 (configs/model_config.py
        # rewrites the draft ModelConfig to it)
        self.is_draft = archs[:1] == ["Qwen4ExpForCausalLMMTP"]
        # the unsloth export ships no mmproj (plan 1, 13); never route a
        # qwen4exp load through qwen35's clip name map
        self.vision_config = None
        self.mmproj_path = None
        self.num_k = int(getattr(self.config, "linear_num_key_heads", 0) or 0)
        self.num_v = int(getattr(self.config, "linear_num_value_heads", 0) or 0)
        self.head_k_dim = int(getattr(self.config, "linear_key_head_dim", 0) or 0)
        self.head_v_dim = int(getattr(self.config, "linear_value_head_dim", 0) or 0)
        self._kv_cache: Optional[Dict[str, Any]] = None
        self._draft_block_cache: Optional[int] = None

    # ------------------------------------------------------------------
    # Header facts
    # ------------------------------------------------------------------

    def _kv(self) -> Dict[str, Any]:
        if self._kv_cache is None:
            self._kv_cache = read_qwen4exp_kv(self.gguf_file)
        return self._kv_cache

    def _ple_layer(self, file_tensors: Sequence[str]) -> Optional[int]:
        layers = self._kv().get("ple.layers")
        if isinstance(layers, list) and layers:
            return int(layers[0])
        for n in file_tensors:
            parts = _split_blk(n)
            if parts is not None and parts[1] == "ple_key.weight":
                return parts[0]
        return None

    # ------------------------------------------------------------------
    # Name map
    # ------------------------------------------------------------------

    def build_name_map(self) -> Dict[str, str]:
        if _hc_mixer_int8_requested():
            raise RuntimeError(
                "qwen4exp GGUF: SGLANG_HC_MIXER_INT8 builds the hyper-connection "
                "mixers as quantized layers, but the GGUF loader dequantizes the "
                "Q8_0 mixer tensors to bf16 nn.Linear weights; unset "
                "SGLANG_HC_MIXER_INT8 for a GGUF boot"
            )
        if self.is_draft:
            return self._build_draft_name_map()
        file_tensors = self._file_tensors()
        gguf_to_hf, unknown = build_qwen4exp_name_map(
            file_tensors, self.num_layers, self._ple_layer(sorted(file_tensors))
        )
        if unknown:
            raise RuntimeError(
                f"qwen4exp GGUF: {len(unknown)} tensors not mapped: {unknown[:12]}"
            )
        problems = qwen4exp_missing_roles(file_tensors, self.num_layers)
        if problems:
            raise RuntimeError(
                f"qwen4exp GGUF {self.gguf_file}: incomplete tensor set "
                f"({len(problems)} problems): {problems[:6]}"
            )
        logger.info(
            "qwen4exp GGUF name map: %d tensors for %d layers (arch %s)",
            len(gguf_to_hf),
            self.num_layers,
            self.arch,
        )
        return gguf_to_hf

    # ------------------------------------------------------------------
    # G3: the NEXTN/MTP draft
    # ------------------------------------------------------------------

    def draft_block_index(self) -> int:
        """Block index of the draft layer, from the file (see
        :func:`qwen4exp_draft_block_index`)."""
        if self._draft_block_cache is None:
            self._draft_block_cache = qwen4exp_draft_block_index(
                self._file_tensors(), self._kv(), self.gguf_file
            )
        return self._draft_block_cache

    def draft_shares_target_vocab(self) -> bool:
        """True for the unsloth ``shared-*`` head (no token_embd / output in the
        file). For the self-contained head the two vocabulary tables are skipped
        and the target's are shared in all the same (module docstring)."""
        return qwen4exp_draft_is_shared(self._kv())

    def _build_draft_name_map(self) -> Dict[str, str]:
        """``{gguf name: HF name}`` of the MTP head: ``blk.<N>.*`` -> ``mtp.*`` of
        ``Qwen4ExpForCausalLMMTP`` (module docstring, G3 table)."""
        from sglang.srt.model_loader.gguf_shards import iter_gguf_tensors

        file_tensors = self._file_tensors()
        kv = self._kv()
        block = self.draft_block_index()
        gguf_to_hf, unknown = build_qwen4exp_draft_name_map(file_tensors, block)
        if unknown:
            raise RuntimeError(
                f"qwen4exp GGUF MTP draft {self.gguf_file}: {len(unknown)} tensors "
                f"not mapped: {unknown[:12]}"
            )
        problems = qwen4exp_draft_missing_roles(file_tensors, block, kv)
        hidden = int(getattr(self.config, "hidden_size", 0) or 0)
        hc = int(getattr(self.config, "hc_count", 0) or 0)
        if hidden and hc:
            shapes = {str(t.name): tuple(int(d) for d in t.shape)
                      for t in iter_gguf_tensors(self.shard_paths())}
            problems += qwen4exp_draft_shape_problems(shapes, block, hidden, hc)
        if problems:
            raise RuntimeError(
                f"qwen4exp GGUF MTP draft {self.gguf_file}: incomplete or "
                f"inconsistent head ({len(problems)} problems): {problems[:6]}"
            )
        skipped = [n for n in DRAFT_VOCAB_ROLES if n in file_tensors]
        logger.info(
            "qwen4exp GGUF MTP draft name map: %d tensors (blk.%d, %s variant%s); "
            "the draft shares the target's vocabulary modules",
            len(gguf_to_hf),
            block,
            "shared" if self.draft_shares_target_vocab() else "self-contained",
            f", {skipped} not read" if skipped else "",
        )
        return gguf_to_hf

    def _draft_pre(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """The draft-only stream rules, ahead of ``_pre_stream``:

        * the routed experts the generic GGUF iterator emits under the MAIN-model
          spelling ``model.layers.<N>.mlp.experts.<e>.<proj>.qweight(_type)``
          (``N`` = the draft block) go to ``mtp.layers.0.mlp.experts...``, where
          ``Qwen3_5ForCausalLMMTP.load_weights`` keeps them (the qwen35 adapter
          keys this on ``num_hidden_layers``, which is 1 for this draft);
          experts of any other block of a combined export are dropped;
        * ``nextn.eh_proj`` is dequantised and split: columns ``[:H]`` are
          ``mtp.fc_embedding``, columns ``[H:]`` are ``mtp.fc_hidden``
          (converter: ``torch.cat([fc_embedding, fc_hidden], dim=1)``).
        """
        block = self.draft_block_index()
        hidden = int(getattr(self.config, "hidden_size", 0) or 0)
        main_exp = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.")
        eh_type: Optional[int] = None
        for name, weight in weights:
            m = main_exp.match(name)
            if m is not None:
                if int(m.group(1)) == block:
                    yield "mtp.layers.0.mlp.experts." + name[m.end():], weight
                continue
            base, leaf = self._split_leaf(name)
            if base != DRAFT_EH_PROJ_INTERNAL:
                yield name, weight
                continue
            if leaf == "qweight_type":
                eh_type = int(weight.item())
                continue
            if leaf == "qweight":
                if eh_type is None:
                    raise RuntimeError(
                        f"qwen4exp GGUF MTP draft: {name}: type marker not seen yet"
                    )
                dense = self._dequantize(weight, eh_type)
            else:  # an F32 eh_proj arrives as a plain .weight
                dense = weight.to(self._param_dtype())
            rows, cols = int(dense.shape[0]), int(dense.shape[1])
            if cols != 2 * rows or (hidden and rows != hidden):
                raise RuntimeError(
                    f"qwen4exp GGUF MTP draft: eh_proj is {tuple(dense.shape)} "
                    f"(torch [out, in]); expected [H, 2H] with H = hidden_size "
                    f"{hidden or rows}"
                )
            yield "mtp.fc_embedding.weight" + _FINAL, dense[:, :rows].contiguous()
            yield "mtp.fc_hidden.weight" + _FINAL, dense[:, rows:].contiguous()

    def _module_prefix_spelling(self, base: str) -> str:
        # the two indexer halves are ONE module in the model
        for half in (".__q", ".__k"):
            if base.endswith(half):
                return base[: -len(half)]
        return base

    # ------------------------------------------------------------------
    # Weight-value transforms
    # ------------------------------------------------------------------

    def _param_dtype(self) -> torch.dtype:
        dt = getattr(self.config, "dtype", None) or getattr(self.config, "torch_dtype", None)
        if isinstance(dt, str):
            dt = getattr(torch, dt.replace("torch.", ""), None)
        return dt if isinstance(dt, torch.dtype) else torch.bfloat16

    @staticmethod
    def _dense_from_bytes(weight: torch.Tensor, qtype: int) -> torch.Tensor:
        """A ``.qweight`` payload of an UNQUANTIZED ggml type (gguf-py hands
        BF16/F16 out as uint8 with a doubled last dim) as real values."""
        import gguf

        gt = gguf.GGMLQuantizationType(qtype)
        if weight.dtype != torch.uint8:
            return weight
        if gt == gguf.GGMLQuantizationType.BF16:
            return weight.contiguous().view(torch.bfloat16)
        if gt == gguf.GGMLQuantizationType.F16:
            return weight.contiguous().view(torch.float16)
        if gt == gguf.GGMLQuantizationType.F32:
            return weight.contiguous().view(torch.float32)
        raise ValueError(f"not an unquantized ggml type: {gt.name}")

    def _dequantize(self, weight: torch.Tensor, qtype: int) -> torch.Tensor:
        import gguf
        from gguf.quants import dequantize

        gt = gguf.GGMLQuantizationType(qtype)
        if gt in (
            gguf.GGMLQuantizationType.BF16,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.F32,
        ):
            return self._dense_from_bytes(weight, qtype).to(self._param_dtype())
        return torch.from_numpy(dequantize(weight.numpy(), gt).copy()).to(self._param_dtype())

    @staticmethod
    def _split_leaf(name: str) -> Tuple[str, str]:
        """``(base, leaf)`` for the three leaves a streamed GGUF name can have."""
        for leaf in ("qweight_type", "qweight", "weight"):
            if name.endswith("." + leaf):
                return name[: -len(leaf) - 1], leaf
        return name, ""

    @staticmethod
    def _kind(base: str, leaf: str) -> Optional[str]:
        """Which qwen4exp pre-stage rule (if any) a streamed tensor falls under."""
        if re.search(r"\.index_qk_proj\.__[qk]$", base):
            return "indexer"
        if re.search(r"(^|\.)(\w+_hyper_connection|hyper_connection_mixer)\.input_mix_weight_(down|up)$", base):
            return "hc_dense"
        if base.endswith(".block_inject_weight"):
            return "hc_dense"
        if leaf != "weight":
            return None
        if base.endswith(
            (
                ".hc_norm",
                ".indexer.q_layernorm",
                ".indexer.k_layernorm",
                ".ple.norm_key",
                ".ple.norm_query",
                ".ple.norm_conv",
            )
        ):
            return "gemma_norm"
        if base.endswith(".ple.conv1d"):
            return "ple_conv"
        if base.endswith(".mlp.gate"):
            return "router"
        return None

    def _pre_stream(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """The qwen4exp-only tensors. Whatever is finished here is yielded under
        ``name + _FINAL``; everything else passes through to the inherited
        qwen35 stream unchanged."""
        dense_types: Dict[str, int] = {}
        idx_types: Dict[str, Dict[str, Tuple[str, torch.Tensor]]] = {}
        idx_data: Dict[str, Dict[str, torch.Tensor]] = {}
        param_dtype = self._param_dtype()

        for name, weight in weights:
            base, leaf = self._split_leaf(name)
            kind = self._kind(base, leaf)
            if kind is None:
                yield name, weight
                continue

            if kind == "hc_dense":
                # plain nn.Linear / nn.Linear-like parameters in the model: a
                # quantized GGUF tensor (Q8_0 mixers) is dequantised to the
                # module dtype, an F32 one (inject) is cast
                if leaf == "qweight_type":
                    dense_types[base] = int(weight.item())
                    continue
                if leaf == "qweight":
                    if base not in dense_types:
                        raise RuntimeError(f"qwen4exp GGUF: {name}: type marker not seen yet")
                    yield base + ".weight" + _FINAL, self._dequantize(weight, dense_types[base])
                    continue
                yield name + _FINAL, weight.to(param_dtype)
                continue

            if kind == "router":
                yield name + _FINAL, weight.to(param_dtype)
                continue

            if kind == "gemma_norm":
                # the converter baked +1 into the zero-centred gamma
                yield name + _FINAL, weight.float() - 1.0
                continue

            if kind == "ple_conv":
                # converter L168-L169 squeezed [C, 1, K]; NOT a GDN conv, so no
                # V-head un-tile (the inherited stream would apply one by name)
                yield name + _FINAL, (weight.unsqueeze(1) if weight.dim() == 2 else weight)
                continue

            # --- indexer: fuse q|k back into index_qk_proj ------------------
            half = "q" if base.endswith(".__q") else "k"
            fused_base = base[: -len(".__q")]
            if leaf == "qweight_type":
                slot = idx_types.setdefault(fused_base, {})
                slot[half] = (name, weight)
                if len(slot) == 2:
                    tq, tk = int(slot["q"][1].item()), int(slot["k"][1].item())
                    if tq != tk:
                        raise RuntimeError(
                            f"qwen4exp GGUF: {fused_base}: indexer q_proj and k_proj "
                            f"carry different ggml types ({tq} vs {tk}); they are one "
                            "HF tensor"
                        )
                    yield fused_base + ".qweight_type" + _FINAL, slot["q"][1]
                    del idx_types[fused_base]
                continue
            key = fused_base + "." + leaf
            slot = idx_data.setdefault(key, {})
            slot[half] = weight
            if len(slot) == 2:
                # converter L156-L157: q = rows [:n_q], k = rows [n_q:]
                yield key + _FINAL, torch.cat([slot["q"], slot["k"]], dim=0)
                del idx_data[key]

        if idx_types or idx_data:
            raise RuntimeError(
                "qwen4exp GGUF: indexer q_proj/k_proj without their other half: "
                f"{sorted(idx_types) + sorted(idx_data)}"
            )

    # ------------------------------------------------------------------
    # PLE table (G2)
    # ------------------------------------------------------------------

    def _ple_table_hf_name(self) -> Optional[str]:
        """HF name the name map gives the table (``...ngram_embedding.weight``)."""
        layers = self._kv().get("ple.layers")
        layer = int(layers[0]) if isinstance(layers, list) and layers else None
        if layer is None:
            layer = self._ple_layer(sorted(self._file_tensors()))
        if layer is None:
            return None
        return f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight"

    def stream_name_map(self, name_map: Dict[str, str]) -> Dict[str, str]:
        """The map the weight ITERATOR is built from: ``name_map`` without the PLE
        table, whose payload is mapped by the model instead of copied
        (``gguf_quant_weights_iterator`` only streams tensors named in its map)."""
        if self.is_draft:
            return name_map
        return {g: h for g, h in name_map.items() if g != PLE_TABLE_GGUF_NAME}

    def ple_table_marker(self) -> Optional[Tuple[str, torch.Tensor]]:
        """``(marker name, marker tensor)`` for the table, or None when the file
        has none. Reads the GGUF headers only; refuses every format but IQ4_NL."""
        from sglang.srt.models.qwen4_exp_ple_gguf import (
            check_ple_table_supported,
            encode_ple_table_marker,
            locate_gguf_tensor,
        )

        if PLE_TABLE_GGUF_NAME not in self._file_tensors():
            return None
        hf = self._ple_table_hf_name()
        if hf is None:
            raise RuntimeError(
                f"qwen4exp GGUF {self.gguf_file}: has {PLE_TABLE_GGUF_NAME} but names no PLE layer"
            )
        loc = locate_gguf_tensor(self.shard_paths(), PLE_TABLE_GGUF_NAME)
        check_ple_table_supported(loc)
        base = hf[: -len(".weight")]
        return f"{base}.{PLE_TABLE_MARKER_LEAF}", encode_ple_table_marker(loc)

    def _ple_constant_tensors(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """The UINT64 KV arrays as the int64 buffers the model loads."""
        kv = self._kv()
        layers = kv.get("ple.layers")
        if not (isinstance(layers, list) and layers):
            return
        layer = int(layers[0])
        for buf, key in _PLE_CONSTANTS:
            values = kv.get(key)
            if not isinstance(values, list):
                continue
            yield (
                f"model.layers.{layer}.ple.ple_embedding.{buf}",
                torch.tensor([int(v) for v in values], dtype=torch.int64),
            )

    def transform_stream(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        if self.is_draft:
            weights = self._draft_pre(weights)
        else:
            # G2: the table's location goes first, so a refused format (or a
            # model without the checkpoint offload backend) fails before the
            # weights load (the draft has no PLE layer)
            marker = self.ple_table_marker()
            if marker is not None:
                yield marker
        for name, weight in super().transform_stream(self._pre_stream(weights)):
            if name.endswith(_FINAL):
                name = name[: -len(_FINAL)]
            yield name, weight
        if not self.is_draft:  # the draft has no PLE layer (config.ple_layer_ids = [])
            yield from self._ple_constant_tensors()
