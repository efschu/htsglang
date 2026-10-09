"""GGUF-NF G6 (2026-10-09): JIT glue of the IQ-type MMQ / MoE-MMQ kernels (sglang PR #36122, vendored).

Sources: ``csrc/gguf_iq_mmq/`` -- kernels from sgl-project/sglang PR #36122 (open, unreviewed), host side new (tvm-ffi,
torch-free). One JIT module per ggml type (a cold cache costs ~15 s of nvcc per type instead of ~8x that); the module
exports ``mul_mat_a8`` (dense, the ``ggml_mul_mat_a8`` analogue) and ``moe_a8`` (the ``ggml_moe_a8`` analogue).

Lifecycle -- the important part: a JIT build must NEVER run inside a forward or a CUDA-graph capture. ``prepare`` builds (or
refuses, with a named reason) at weight-load time; ``is_ready`` is a dictionary lookup and is the only thing the dispatch
in ``srt/layers/quantization/gguf.py`` calls per forward. A type that was not prepared, was refused (env off, Blackwell
nvcc gate) or failed to build is simply "not ready" and the dispatch keeps the pre-G6 path (``moe_vec`` / MMVQ / dequant).

Policy (thresholds with sources, nvcc / Blackwell gate, env switches): ``gguf_iq_mmq_policy.py``.
"""

from __future__ import annotations

import functools
import logging
import os
import pathlib
import shutil
import subprocess
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from sglang.jit_kernel import gguf_iq_mmq_policy as policy
from sglang.jit_kernel.utils import get_jit_cuda_arch, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_READY = "ready"
#: (arch target, ggml type) -> "ready" | named refusal / failure text
_STATE: Dict[Tuple[str, int], str] = {}
_MODULES: Dict[Tuple[str, int], "Module"] = {}


def _nvcc_path() -> Optional[str]:
    """The nvcc the JIT build uses -- the same search order as tvm_ffi.cpp (CUDA_HOME / CUDA_PATH, PATH, /usr/local/cuda)."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        found = shutil.which("nvcc")
        if found is not None:
            return found
        cuda_home = "/usr/local/cuda"
    cand = pathlib.Path(cuda_home) / "bin" / "nvcc"
    return str(cand) if cand.exists() else None


@functools.lru_cache(maxsize=None)
def nvcc_version() -> Optional[Tuple[int, int, int]]:
    """(major, minor, build) of the JIT toolchain, e.g. (13, 0, 88); None when nvcc is missing or does not parse."""
    nvcc = _nvcc_path()
    if nvcc is None:
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60, check=False).stdout
    except Exception:
        return None
    return policy.parse_nvcc_version(out)


def refusal(type_id: int) -> Optional[str]:
    """Named reason why the IQ MMQ kernel of ``type_id`` must not be used on THIS device (None = may be used)."""
    if not policy.env_enabled():
        return f"IQ-MMQ-OFF: {policy.ENV_ENABLE}=0"
    if type_id not in policy.IQ_MMQ_TYPES:
        return f"IQ-MMQ-TYPE: ggml type {type_id} is not an IQ MMQ type"
    arch = get_jit_cuda_arch()
    return policy.blackwell_refusal(type_id, arch.major, nvcc_version(), policy.fix_verified())


def _module(type_id: int) -> "Module":
    arch = get_jit_cuda_arch().target_name
    key = (arch, type_id)
    mod = _MODULES.get(key)
    if mod is None:
        mod = load_jit(
            "gguf_iq_mmq",
            f"t{type_id}",
            cuda_files=["gguf_iq_mmq/gguf_iq_mmq.cuh"],
            cuda_wrappers=[
                ("mul_mat_a8", "gguf_iq_mmq::mul_mat_a8"),
                ("moe_a8", "gguf_iq_mmq::moe_a8"),
            ],
            extra_cuda_cflags=[f"-DGGUF_IQ_MMQ_TYPE_MASK={1 << (type_id - 16)}"],
        )
        _MODULES[key] = mod
    return mod


def prepare(type_id: int) -> bool:
    """Build (or refuse) the IQ MMQ module of ``type_id`` for the current device. Call at weight-load time, never in a forward."""
    type_id = int(type_id)
    if type_id not in policy.IQ_MMQ_TYPES:
        return False
    key = (get_jit_cuda_arch().target_name, type_id)
    state = _STATE.get(key)
    if state is not None:
        return state == _READY
    why = refusal(type_id)
    if why is not None:
        _STATE[key] = why
        logger.warning("%s -- %s keeps the pre-G6 path on this rank (moe_vec / MMVQ / dequant)", why, policy.IQ_TYPE_NAMES[type_id])
        return False
    try:
        _module(type_id)
    except Exception as exc:  # noqa: BLE001 -- a failed build must degrade to the old path, loudly
        _STATE[key] = f"IQ-MMQ-BUILD-FAILED: {type(exc).__name__}: {exc}"
        logger.warning("%s -- %s keeps the pre-G6 path on this rank", _STATE[key], policy.IQ_TYPE_NAMES[type_id])
        return False
    _STATE[key] = _READY
    logger.info(
        "IQ-MMQ ready: %s on %s (nvcc %s)",
        policy.IQ_TYPE_NAMES[type_id],
        key[0],
        nvcc_version(),
    )
    return True


def is_ready(type_id: int) -> bool:
    """Per-forward check: a dictionary lookup, never a build."""
    if not _STATE:
        return False
    return _STATE.get((get_jit_cuda_arch().target_name, int(type_id))) == _READY


def _padded_k(col: int) -> int:
    return (col + 512 - 1) // 512 * 512


def mul_mat_a8(W: torch.Tensor, X: torch.Tensor, type_id: int, row: int) -> torch.Tensor:
    """Dense IQ MMQ: Y[M, row] = X[M, K] @ dequant(W)[row, K].T through the q8_1 activation. Same contract as ggml_mul_mat_a8."""
    X = X.contiguous()
    mod = _module(int(type_id))
    m, col = X.shape
    Y = torch.empty((m, row), dtype=X.dtype, device=W.device)
    quant_x = torch.empty((m, _padded_k(col) // 32 * 9), dtype=torch.int32, device=W.device)
    mod.mul_mat_a8(W, X, Y, quant_x, int(type_id), int(row))
    return Y


def moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    type_id: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MoE IQ MMQ. Same contract as ggml_moe_a8 (sgl_kernel): returns Y[tokens * top_k, row].

    ``sorted_token_ids`` / ``expert_ids`` / ``num_tokens_post_padded`` come from moe_align_block_size with block size
    ``policy.IQ_MOE_MMQ_BLOCK_SIZE`` (4); ``expert_ids`` must already carry -1 for blocks that are not a valid LOCAL expert
    (the caller sanitises, exactly as for ggml_moe_a8 -- see fused_moe_gguf).
    """
    X = X.contiguous()
    mod = _module(int(type_id))
    col = X.shape[1]
    Y = torch.empty((tokens * top_k, row), dtype=X.dtype, device=W.device)
    quant_x = torch.empty((tokens, _padded_k(col) // 32 * 9), dtype=torch.int32, device=W.device)
    mod.moe_a8(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, Y, quant_x, int(type_id), int(row), int(top_k), int(tokens))
    return Y
