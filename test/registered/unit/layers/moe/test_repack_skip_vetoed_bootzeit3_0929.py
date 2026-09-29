"""BOOTZEIT 3 (29.09., z30r3): D's Marlin repack only over the rows it read.

z30r3 D (H2 on): TP0 read 3171 of 9040 owned expert rows on 40 layers, TP1
3320/6096, TP2 3120/7632 -- the rest were vetoed (P's bytes in the store) and
never read, yet the repack ran one JIT launch per row over the whole [E]
window, ~30 s per D rank. SGLANG_MOE_REPACK_SKIP_VETOED=1 hands the repack the
kept rows only; every row a reader sees (residents, rows written to the
store) is repacked exactly as before.
"""

import sys
import types
from unittest import mock

import torch

# gptq_kernels only through the quantization registry: importing it first
# trips a pre-existing import cycle (see tests/moe_offload/test_repack_adoption_0922.py)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # noqa: F401
    compressed_tensors_wNa16_moe,
)
from sglang.srt.layers.moe import store_adopt as sa

gk = sys.modules["sglang.srt.hardware_backend.gpu.quantization.gptq_kernels"]


def _layer(vetoed):
    lay = types.SimpleNamespace(layer_id=0)
    lay._moe_store_adopt_vetoed_global = frozenset(vetoed)
    return lay


def _window(lo, pad):
    return mock.patch("sglang.srt.layers.moe.expert_offload._layer_expert_window",
                      return_value=(lo, pad))


def test_off_by_default_means_all_rows(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_REPACK_SKIP_VETOED", raising=False)
    with _window(6, True):
        assert sa.repack_rows(_layer({8, 10, 11}), 7) is None


def test_on_keeps_pad_and_every_unvetoed_row(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_REPACK_SKIP_VETOED", "1")
    # window lo=6 with pad: local e>=1 is global 6+e-1 -> 8,10,11 are local 3,5,6
    with _window(6, True):
        assert sa.repack_rows(_layer({8, 10, 11}), 7) == [0, 1, 2, 4]
    # no pad: local e is global lo+e
    with _window(6, False):
        assert sa.repack_rows(_layer({8, 10, 11}), 6) == [0, 1, 3]


def test_on_but_nothing_vetoed_or_no_window_means_all(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_REPACK_SKIP_VETOED", "1")
    with _window(6, True):
        assert sa.repack_rows(_layer(set()), 7) is None
        assert sa.repack_rows(types.SimpleNamespace(), 7) is None
        # vetoed ids outside this window change nothing
        assert sa.repack_rows(_layer({0, 1}), 7) is None
    with mock.patch("sglang.srt.layers.moe.expert_offload._layer_expert_window",
                    return_value=None):
        assert sa.repack_rows(_layer({8}), 7) is None


class _Module:
    def __init__(self):
        self.rows = []

    def gptq_marlin_repack(self, w, perm, out, size_k, size_n, num_bits):
        self.rows.append(int(w[0, 0]))
        out.fill_(int(w[0, 0]) + 100)


def test_kernel_loop_touches_exactly_the_given_rows():
    E, k, n = 5, 16, 4
    w = torch.arange(E, dtype=torch.int32).view(E, 1, 1).expand(E, k // 8, n).contiguous()
    perm = torch.empty((E, 0), dtype=torch.int32)
    mod = _Module()
    with mock.patch.object(gk, "_jit_gptq_marlin_repack_module", return_value=mod):
        out = gk.gptq_marlin_moe_repack(w, perm, k, n, 4, rows=[0, 2, 3])
        assert mod.rows == [0, 2, 3]
        for e in (0, 2, 3):
            assert bool((out[e] == e + 100).all())
        mod.rows.clear()
        gk.gptq_marlin_moe_repack(w, perm, k, n, 4)
        assert mod.rows == [0, 1, 2, 3, 4]
