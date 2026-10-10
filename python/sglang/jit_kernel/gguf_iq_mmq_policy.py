"""GGUF-NF G6 (2026-10-09): policy of the IQ-type MMQ / MoE-MMQ kernels -- pure Python, no torch, no CUDA import.

What lives here (so desk tests drive every branch without a GPU and ``gguf.py`` keeps its import surface):

* the ggml type ids the kernels serve and their K alignment,
* the thresholds, each with its source,
* the nvcc / Blackwell gate (llama.cpp #28581, fix #28784),
* the shape rules that decide MMQ vs. the old path (``moe_vec`` / dequant + cuBLAS).

The kernels themselves are ``csrc/gguf_iq_mmq/`` (sglang PR #36122, vendored, see the file heads) and the JIT glue is
``gguf_iq_mmq.py``.

Switches (both new, listed in the G6 report):

    SGLANG_GGUF_IQ_MMQ                       default ON  -- ``0`` switches the IQ MMQ kernels off (every shape then takes
        the pre-G6 path: ``moe_vec`` for the MoE, MMVQ / dequant + cuBLAS for dense). The rollback lever.
    SGLANG_GGUF_IQ_MMQ_NVCC132_FIX_VERIFIED  default off -- the Blackwell gate below refuses IQ1_S / IQ2_S / IQ3_S on sm_12x
        when the JIT toolchain is nvcc 13.2.0 / 13.2.1. The vendored sources carry the #28784 byte-extraction workaround,
        but it has not run on a card of this rig yet; ``1`` lifts the refusal for the metal test that proves it.
"""

from __future__ import annotations

import os
import re
from typing import Dict, FrozenSet, Optional, Tuple

ENV_ENABLE = "SGLANG_GGUF_IQ_MMQ"
ENV_FIX_VERIFIED = "SGLANG_GGUF_IQ_MMQ_NVCC132_FIX_VERIFIED"

#: ggml type ids (gguf.GGMLQuantizationType) the vendored kernels serve. Pinned against the gguf package by the test.
IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS = 16, 17, 18, 19, 20, 21, 22, 23
IQ_TYPE_NAMES: Dict[int, str] = {
    IQ2_XXS: "IQ2_XXS",
    IQ2_XS: "IQ2_XS",
    IQ3_XXS: "IQ3_XXS",
    IQ1_S: "IQ1_S",
    IQ4_NL: "IQ4_NL",
    IQ3_S: "IQ3_S",
    IQ2_S: "IQ2_S",
    IQ4_XS: "IQ4_XS",
}
IQ_MMQ_TYPES: FrozenSet[int] = frozenset(IQ_TYPE_NAMES)

#: Required input size (K) multiple. 256 = one QK_K super block (PR #36122 uses 256 for all eight types). IQ4_NL has
#: 32-element blocks; its tile loader zero-fills the blocks past the row end (csrc/gguf_iq_mmq/iq_mmq_tiles.cuh), which
#: makes K = 128 * n safe -- the NF expert ffn_down is K = 640 = 5 * 128 (GGUF_NF_IST_1009.md: expert_ff 640).
K_ALIGNMENT: Dict[int, int] = {t: 256 for t in IQ_MMQ_TYPES}
K_ALIGNMENT[IQ4_NL] = 128

#: Source: sglang PR #36122 python/sglang/srt/layers/quantization/gguf.py (IQ_MOE_MMQ_MIN_TOKENS = 128, measured on GB10:
#: MoE IQ3_S M=128 1.51x, M=256 2.56x, M=2050 3.84x over dequant + BF16 -- below 128 tokens the PR keeps the old path).
IQ_MOE_MMQ_MIN_TOKENS = 128
#: Source: PR #36122 (IQ_MOE_MMQ_MIN_ASSIGNMENTS_PER_EXPERT = 2): the batched kernel needs enough work per expert.
IQ_MOE_MMQ_MIN_ASSIGNMENTS_PER_EXPERT = 2
#: ggml_moe_get_block_size() of the IQ MoE kernels on CUDA (MOE_X_IQ* = 4 in iq_mmq_kernels.cuh; static_assert in the host).
IQ_MOE_MMQ_BLOCK_SIZE = 4
#: CUDA grid.y limit; the IQ MoE kernels launch one four-assignment block per grid.y step (PR #36122 CUDA_MAX_GRID_Y).
CUDA_MAX_GRID_Y = 65535
#: Source: PR #36122 (IQ_MMQ_MAX_BATCH_SIZE = 16): the dense IQ MMQ is flat-priced for small M only (GB10: M=16 IQ3_S 1.66x,
#: IQ4_NL 1.48x, IQ4_XS 1.26x over dequant + BF16); above 16 tokens dequant + cuBLAS wins.
IQ_MMQ_MAX_BATCH_SIZE = 16
#: The PR's literal cap IQ_MOE_MMQ_MAX_EXPERTS = 256 came from the `exp_idx > 255` guard of the upstream moe_q. This fork's
#: iq_moe_q guards with the LOCAL expert count (fork fix #109/#112), so the cap does not apply: NF has 512 experts.

#: Types affected by the nvcc 13.2.0 / 13.2.1 byte-mask miscompile on sm_120 (llama.cpp #28581: IQ1_S / IQ2_S / IQ3_S fail,
#: IQ4_XS and the K-quants are OK; IQ4_NL not reported broken). Our IQ4_NL / IQ4_XS kernels are therefore never gated.
BLACKWELL_AFFECTED_TYPES: FrozenSet[int] = frozenset({IQ1_S, IQ2_S, IQ3_S})
#: nvcc builds of the broken CUDA 13.2.0 / 13.2.1 toolkits (llama.cpp #28581 comments: V13.2.51 = 13.2.0, V13.2.78 = 13.2.1;
#: V13.2.86 = 13.2.2 clean). Anything in the 13.2 line below build 86 is treated as broken.
NVCC_BAD_RELEASE = (13, 2)
NVCC_FIRST_CLEAN_BUILD = 86

_NVCC_RE = re.compile(r"release\s+(\d+)\.(\d+),\s+V(\d+)\.(\d+)\.(\d+)")


def parse_nvcc_version(text: str) -> Optional[Tuple[int, int, int]]:
    """``nvcc --version`` text -> (major, minor, build), e.g. (13, 2, 78); None when it does not parse."""
    m = _NVCC_RE.search(text or "")
    if not m:
        return None
    major, minor, vmajor, vminor, build = (int(g) for g in m.groups())
    if (major, minor) != (vmajor, vminor):
        return None
    return (major, minor, build)


def nvcc_is_broken_132(version: Optional[Tuple[int, int, int]]) -> bool:
    """True for CUDA 13.2.0 / 13.2.1 (nvcc 13.2.x with build < 86)."""
    if version is None:
        return False
    return (version[0], version[1]) == NVCC_BAD_RELEASE and version[2] < NVCC_FIRST_CLEAN_BUILD


def env_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get(ENV_ENABLE, "1")).strip().lower() not in ("0", "false", "off", "no", "n", "f")


def fix_verified(env: Optional[Dict[str, str]] = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get(ENV_FIX_VERIFIED, "")).strip().lower() in ("1", "true", "on", "yes", "y", "t")


def blackwell_refusal(
    type_id: int,
    sm_major: int,
    nvcc_version: Optional[Tuple[int, int, int]],
    verified: bool = False,
) -> Optional[str]:
    """Named refusal of the IQ1_S / IQ2_S / IQ3_S MMQ on sm_12x built with nvcc 13.2.0 / 13.2.1, else None.

    The caller falls back to the pre-G6 path (``moe_vec`` / MMVQ / dequant); nothing is silently wrong.
    """
    if type_id not in BLACKWELL_AFFECTED_TYPES:
        return None
    if sm_major != 12:
        return None
    if not nvcc_is_broken_132(nvcc_version):
        return None
    if verified:
        return None
    name = IQ_TYPE_NAMES[type_id]
    ver = ".".join(str(p) for p in nvcc_version)  # type: ignore[arg-type]
    return (
        f"IQ-MMQ-BLACKWELL-NVCC: {name} MMQ refused on sm_12x: nvcc {ver} (CUDA 13.2.0/13.2.1) miscompiles the "
        f"byte extraction of the IQ1_S/IQ2_S/IQ3_S kernels (llama.cpp #28581); CUDA >= 13.2.2 is clean. The vendored "
        f"sources carry the #28784 __byte_perm workaround, not yet proven on this rig; set {ENV_FIX_VERIFIED}=1 to "
        f"run the metal test, otherwise the MoE stays on moe_vec"
    )


def k_aligned(type_id: int, input_size: int) -> bool:
    align = K_ALIGNMENT.get(int(type_id))
    return align is not None and input_size % align == 0


def moe_mmq_shape_ok(num_tokens: int, num_experts: int, top_k: int) -> bool:
    """PR #36122 `_is_iq_moe_mmq_shape` minus the 256-expert cap (see above)."""
    max_num_tokens_padded = num_tokens * top_k + (num_experts + 1) * (IQ_MOE_MMQ_BLOCK_SIZE - 1)
    return (
        num_tokens >= IQ_MOE_MMQ_MIN_TOKENS
        and num_tokens * top_k >= IQ_MOE_MMQ_MIN_ASSIGNMENTS_PER_EXPERT * num_experts
        and max_num_tokens_padded <= CUDA_MAX_GRID_Y * IQ_MOE_MMQ_BLOCK_SIZE
    )


def dense_mmq_shape_ok(num_tokens: int, mmvq_safe: int) -> bool:
    """The dense IQ MMQ takes the window the MMVQ leaves: mmvq_safe < M <= IQ_MMQ_MAX_BATCH_SIZE."""
    return mmvq_safe < num_tokens <= IQ_MMQ_MAX_BATCH_SIZE
