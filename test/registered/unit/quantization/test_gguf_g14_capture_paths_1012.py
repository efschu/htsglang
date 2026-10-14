# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF G14 (2026-10-12): what of the CUDA-graph capture of the GGUF decode forms is checkable WITHOUT a GPU.

Stage A of the metal run died in the first capture warm-up (bs=6, M=24, MTP verify 6 x 4) on layer 3 (QSA indexer, fixed by
``test_gguf_g14_bf16_dense_payload_1012.py``). Whether the graph can then be CAPTURED and REPLAYED (moe_a8 at M = 1..8, the JIT
IQ3_S MMQ under capture) can only be answered by a card; this file pins the part that is a property of the SOURCE and of the
Python wrappers, so a regression of it is red at the desk:

* nothing reachable from the capture (dispatch, wrappers, torch IQ3_S dequant) calls a host-synchronising op
  (``.item() .tolist() .cpu() .numpy() .nonzero() synchronize``) or compiles (``load_jit``) outside the cache-miss branch of
  ``gguf_iq_mmq._module`` -- that branch is only reachable for a type ``prepare`` did not build at LOAD time;
* after ``prepare`` a forward never builds: ``mul_mat_a8`` / ``moe_a8`` take the cached module (call counting on ``load_jit``);
* the argument contract of ``moe_a8`` for the decode token counts M = 1..8 of the production routing (E = 512, top_k = 10): the
  routing buffer is a multiple of the 4-wide MoE tile BEFORE the slice (``moe_align_block_size`` allocates ``numel * block`` for
  fewer than E + 1 assignments), so the G9 slice is the identity there and the launch shape is a pure function of M (graph-static);
* every ``tensor`` the wrappers create sits on the weight's device with a shape that depends on M only.

NOT checked here (metal only, listed in the G14 report): the capture itself, the graph pool size, replay numerics at M = 1..8.
"""

from __future__ import annotations

import ast
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pathlib

import pytest
import torch

from sglang.jit_kernel import gguf_iq_mmq as K
from sglang.jit_kernel import gguf_iq_mmq_policy as P

SRC = pathlib.Path(__file__).resolve()
while SRC.name != "test" and SRC.parent != SRC:
    SRC = SRC.parent
PY = SRC.parent / "python" / "sglang"

HOST_SYNC = {"item", "tolist", "cpu", "numpy", "nonzero", "synchronize", "wait_stream", "record_stream"}
COMPILE = {"load_jit", "compile"}

# capture-reachable on the GGUF decode forms of the NF checkpoint (dense Q8_0 / IQ4_XS / BF16 / IQ3_S, MoE IQ3_S + IQ4_NL)
GGUF_PY = PY / "srt" / "layers" / "quantization" / "gguf.py"
GGUF_FUNCS = {
    "fused_mul_mat_gguf",
    "fused_moe_gguf",
    "_fused_moe_gguf_iq_mmq",
    "_moe_mm_a8",
    "_iq_dense_mmq_selected",
    "_iq_moe_mmq_selected",
    "_mul_mat_iq3s_dequant",
    "_iq3s_dequant_rows",
    "_mul_mat_dequant_chunked",
    "_ggml_dequantize_ws",
    "apply_gguf_embedding",
}
JIT_DIR = PY / "jit_kernel"
JIT_FUNCS = {
    "gguf_iq_mmq.py": {"mul_mat_a8", "moe_a8", "is_ready"},
    "gguf_iq_mmq_policy.py": None,  # every function
    "gguf_iq3s_dequant.py": {"dequantize_iq3_s"},
}


def _functions(path: pathlib.Path, names):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
            yield node


def _calls(node: ast.FunctionDef):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                yield f.attr, n.lineno
            elif isinstance(f, ast.Name):
                yield f.id, n.lineno


def test_the_scan_sees_the_functions_it_claims_to_scan():
    found = {n.name for n in _functions(GGUF_PY, GGUF_FUNCS)}
    assert found == GGUF_FUNCS, GGUF_FUNCS - found
    for fn, names in JIT_FUNCS.items():
        got = {n.name for n in _functions(JIT_DIR / fn, names)}
        assert got and (names is None or got == names), (fn, got)


def test_no_capture_reachable_function_synchronises_the_host_or_compiles():
    bad = []
    for node in _functions(GGUF_PY, GGUF_FUNCS):
        bad += [(GGUF_PY.name, node.name, c, ln) for c, ln in _calls(node) if c in HOST_SYNC | COMPILE]
    for fn, names in JIT_FUNCS.items():
        for node in _functions(JIT_DIR / fn, names):
            bad += [(fn, node.name, c, ln) for c, ln in _calls(node) if c in HOST_SYNC | COMPILE]
    assert bad == [], bad


def test_the_only_compile_site_is_the_cache_miss_branch_of_module():
    sites = []
    for fn in ("gguf_iq_mmq.py", "gguf_iq3s_dequant.py", "gguf_iq_mmq_policy.py"):
        for node in _functions(JIT_DIR / fn, None):
            sites += [(fn, node.name) for c, _ in _calls(node) if c in COMPILE]
    assert sites == [("gguf_iq_mmq.py", "_module")], sites
    src = (JIT_DIR / "gguf_iq_mmq.py").read_text()
    # ... and it is guarded by the module cache, which prepare() fills at load time
    mod = [n for n in _functions(JIT_DIR / "gguf_iq_mmq.py", {"_module"})][0]
    text = ast.get_source_segment(src, mod)
    assert "mod = _MODULES.get(key)" in text and "if mod is None:" in text


class _Rec:
    """Stands for the tvm-ffi module: records the launch."""

    def __init__(self):
        self.calls = []

    def mul_mat_a8(self, *a):
        self.calls.append(("mul_mat_a8",) + a)

    def moe_a8(self, *a):
        self.calls.append(("moe_a8",) + a)


@pytest.fixture
def armed(monkeypatch):
    """A prepared module for IQ3_S (16+... ids) on a fake arch; ``load_jit`` counts instead of compiling."""
    loads = []
    rec = _Rec()
    monkeypatch.setattr(K, "get_jit_cuda_arch", lambda: type("A", (), {"target_name": "sm_test", "major": 8})())
    monkeypatch.setattr(K, "load_jit", lambda *a, **k: loads.append((a, k)) or rec)
    monkeypatch.setattr(K, "_MODULES", {})
    monkeypatch.setattr(K, "_STATE", {})
    # prepare() at LOAD time: the one compile per type
    monkeypatch.setattr(K, "refusal", lambda t: None)
    for t in (P.IQ3_S, P.IQ4_NL):
        assert K.prepare(t) is True
    n_load = len(loads)
    assert n_load == 2
    return rec, loads, n_load


def test_a_forward_after_prepare_never_builds(armed):
    rec, loads, n_load = armed
    E, rows, k = 8, 16, 512
    for qtype in (P.IQ3_S, P.IQ4_NL):
        for m in range(1, 9):
            x = torch.zeros(m, k, dtype=torch.bfloat16)
            w = torch.zeros(rows, k // 256 * 110, dtype=torch.uint8)
            K.mul_mat_a8(w, x, qtype, rows)
            sorted_ids = torch.zeros(m * 4 * 4, dtype=torch.int32)
            K.moe_a8(x, torch.zeros(E, rows, 8, dtype=torch.uint8), sorted_ids, torch.zeros(m * 4, dtype=torch.int32),
                     torch.zeros(1, dtype=torch.int32), qtype, rows, 4, m)
    assert len(loads) == n_load, "a forward compiled a kernel (would be inside the capture)"
    assert len(rec.calls) == 2 * 8 * 2


def test_unprepared_type_is_the_only_one_that_would_compile(armed):
    rec, loads, n_load = armed
    K._module(P.IQ3_S)
    assert len(loads) == n_load
    other = [t for t in P.IQ_MMQ_TYPES if t not in (P.IQ3_S, P.IQ4_NL)][0]
    K._module(other)
    assert len(loads) == n_load + 1  # the cache-miss branch, i.e. exactly what a missed prepare() would do inside a forward


def _alloc_len(numel: int, num_experts: int, block: int) -> int:
    """moe_align_block_size's routing buffer length (pinned against the source below)."""
    if numel < num_experts + 1:
        return numel * block
    return numel + (num_experts + 1) * (block - 1)


def test_the_alloc_formula_used_here_is_the_one_of_moe_align_block_size():
    src = (PY / "srt" / "layers" / "moe" / "moe_runner" / "triton_utils" / "moe_align_block_size.py").read_text()
    assert "if topk_ids.numel() < num_experts + 1:" in src
    assert "max_num_tokens_padded = topk_ids.numel() * block_size" in src
    assert "max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)" in src


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
def test_decode_routing_buffer_of_the_nf_checkpoint_needs_no_slice(m):
    """E = 512 experts, top_k = 10: M * 10 < 513 for every decode M <= 51, so the buffer is numel * 4 -- a multiple of the tile.
    The G9 slice (floor to the tile) is the identity there, i.e. the launch shape is graph-static in M."""
    E, top_k = 512, 10
    n = _alloc_len(m * top_k, E, P.IQ_MOE_MMQ_BLOCK_SIZE)
    assert n == m * top_k * 4 and n % P.IQ_MOE_MMQ_BLOCK_SIZE == 0
    assert P.moe_routing_len(n) == n


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 24])
def test_moe_a8_wrapper_contract_for_decode_m(armed, m):
    rec, loads, n_load = armed
    E, top_k, rows, k = 512, 10, 64, 512
    n = _alloc_len(m * top_k, E, P.IQ_MOE_MMQ_BLOCK_SIZE)
    x = torch.zeros(m, k, dtype=torch.bfloat16)
    w = torch.zeros(2, rows, 8, dtype=torch.uint8)  # content irrelevant: the launch is recorded, not run
    ids = torch.arange(n, dtype=torch.int32)
    eids = torch.zeros(n // P.IQ_MOE_MMQ_BLOCK_SIZE, dtype=torch.int32)
    y = K.moe_a8(x, w, ids, eids, torch.zeros(1, dtype=torch.int32), P.IQ3_S, rows, top_k, m)
    (call,) = rec.calls
    _, X, W, sorted_ids, expert_ids, ntpp, Y, quant_x, tid, row, tk, tokens = call
    assert sorted_ids.shape[0] % P.IQ_MOE_MMQ_BLOCK_SIZE == 0 and sorted_ids.shape[0] == n
    assert sorted_ids.data_ptr() == ids.data_ptr()  # the slice is a zero-copy view
    assert tuple(Y.shape) == (m * top_k, rows) and Y is y
    assert tuple(quant_x.shape) == (m, K._padded_k(k) // 32 * 9) and quant_x.dtype == torch.int32
    assert Y.device == W.device and quant_x.device == W.device
    assert (tid, row, tk, tokens) == (P.IQ3_S, rows, top_k, m)
    assert len(loads) == n_load
