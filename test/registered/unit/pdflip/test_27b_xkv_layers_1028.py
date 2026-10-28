# SPDX-License-Identifier: Apache-2.0
"""27b-xkv-1028: FORMHP-XKV-LAYERS on the 27B line -- the DFlash2 draft-KV commit of all 5 draft layers in one cast + one launch.

Port of NF's b64dd98df (formhp-devdraft-xkv-1028), adapted to the 27B tree and tightened where the 27B draft differs: the 27B draft
(Qwen3.8-27B-DFlash2, 5 sliding layers, head_dim 128, 8 KV heads sharded over TP, bf16 K/V cast into the pool's dtype) reaches the
pool with k_scale / v_scale possibly 1.0 instead of None (identity: the batched cast may skip the division), and its draft pool may be
a page-major pool whose own prefix-valid writer refuses (the batched writer must not bypass that).

Interpreter tests (TRITON_INTERPRET=1) show the kernel's results; they cannot show that the COMPILER accepts it (formb164: DFAST was green
in the interpreter and failed on the card with ``NameError('NGK is not defined')``). So the last test compiles every launch the wrapper can
make for sm_120 and sm_86 with ``triton.compile`` (no GPU), debug on AND off, on the 27B geometries."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(_HERE, "..", "..", "..", "..", "python")

_MK = """
import torch
from flliper.srt.mem_cache import memory_pool as mp

class FakePool(mp.MHATokenToKVPool):
    def __init__(self, n_layers, slots, row_dim, dtype, store_dtype, seed):
        g = torch.Generator().manual_seed(seed)
        nbytes = row_dim * torch.empty((), dtype=store_dtype).element_size()
        self.k_buffer = [torch.randint(0, 256, (slots, nbytes), generator=g, dtype=torch.uint8).view(store_dtype).view(slots, row_dim)
                         for _ in range(n_layers)]
        self.v_buffer = [torch.randint(0, 256, (slots, nbytes), generator=g, dtype=torch.uint8).view(store_dtype).view(slots, row_dim)
                         for _ in range(n_layers)]
        self.dtype, self.store_dtype, self.row_dim, self.size, self.page_size = dtype, store_dtype, row_dim, slots - 1, 1

    def local_slot(self, layer_id):
        return layer_id

class Attn:
    def __init__(self, i, k_scale=None, v_scale=None):
        self.layer_id, self.k_scale, self.v_scale = i, k_scale, v_scale
"""


def _run(code: str) -> str:
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    p = subprocess.run([sys.executable, "-c", _MK + textwrap.dedent(code)], env=env, capture_output=True, text=True, timeout=1500)
    out = [line for line in p.stdout.splitlines() if line.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-3000:]
    return out[0][4:]


def test_selftest_passes_under_the_interpreter_and_has_teeth():
    res = _run("""
        r = []
        # 27B geometries: 3 KV heads x 128 per rank (the uneven TP3 shard) in fp8 -> 384 B, and bf16 pool 2 heads -> 512 B
        for src, dt, st, rd in ((torch.bfloat16, torch.float8_e4m3fn, torch.uint8, 384), (torch.bfloat16, torch.bfloat16, torch.bfloat16, 256)):
            ok, why = mp.prefix_valid_layers_selftest("cpu", src_dtype=src, dtype=dt, store_dtype=st, row_dim=rd, n_layers=5)
            r.append(ok)
        real = mp.layer_buffer_offsets
        mp.layer_buffer_offsets = lambda k, v, d: tuple(torch.zeros_like(t) for t in real(k, v, d))  # every layer into layer 0
        ok, why = mp.prefix_valid_layers_selftest("cpu", src_dtype=torch.bfloat16, dtype=torch.float8_e4m3fn, store_dtype=torch.uint8,
                                                  row_dim=384, n_layers=5)
        r.append(ok)
        print("RES", r)
    """)
    assert res == "[True, True, False]", res


def test_pool_method_equals_the_per_layer_method_byte_for_byte():
    res = _run("""
        mp._is_cuda = True  # the Triton paths (the interpreter runs them on the CPU)
        L, bs, block, slots, rd = 5, 3, 8, 96, 384
        g = torch.Generator().manual_seed(7)
        k = (torch.randn(L, bs * block, 3, rd // 3, generator=g) * 3).to(torch.bfloat16)
        v = (torch.randn(L, bs * block, 3, rd // 3, generator=g) * 3).to(torch.bfloat16)
        loc = (torch.randperm(slots - 2, generator=g)[: bs * block] + 1).view(bs, block).to(torch.int64)
        commit = torch.tensor([3, 8, 1], dtype=torch.int32)
        out = []
        for scale in (None, 1.0):  # 27B: the quant method may leave k/v_scale = 1.0 -- identity, still handled
            a = FakePool(L, slots, rd, torch.float8_e4m3fn, torch.uint8, 1)
            b = FakePool(L, slots, rd, torch.float8_e4m3fn, torch.uint8, 1)
            attns = [Attn(i, scale, scale) for i in range(L)]
            for i in range(L):
                a.set_kv_buffer_prefix_valid(attns[i], loc, commit, k[i].clone(), v[i].clone(), scale, scale)
            handled = b.set_kv_buffer_prefix_valid_layers(layers=attns, loc_2d=loc, commit_lens=commit,
                                                          cache_k_all=k.clone(), cache_v_all=v.clone())
            same = all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a.k_buffer + a.v_buffer, b.k_buffer + b.v_buffer))
            out += [handled, b._xkv_layers_ok, same]
        print("RES", out)
    """)
    assert res == "[True, True, True, True, True, True]", res


def test_pool_method_equals_the_per_layer_method_without_a_cast():
    res = _run("""
        mp._is_cuda = True
        L, bs, block, slots, rd = 5, 2, 8, 64, 256
        g = torch.Generator().manual_seed(11)
        k = (torch.randn(L, bs * block, 2, rd // 2, generator=g) * 3).to(torch.bfloat16)
        v = (torch.randn(L, bs * block, 2, rd // 2, generator=g) * 3).to(torch.bfloat16)
        loc = (torch.randperm(slots - 2, generator=g)[: bs * block] + 1).view(bs, block).to(torch.int64)
        commit = torch.tensor([5, 8], dtype=torch.int32)
        a = FakePool(L, slots, rd, torch.bfloat16, torch.bfloat16, 3)
        b = FakePool(L, slots, rd, torch.bfloat16, torch.bfloat16, 3)
        attns = [Attn(i) for i in range(L)]
        # the draft workspace slice is NOT contiguous across layers when total_ctx < capacity: use such a view
        big_k = torch.zeros(L, bs * block + 5, 2, rd // 2, dtype=torch.bfloat16); big_k[:, : bs * block] = k
        big_v = torch.zeros(L, bs * block + 5, 2, rd // 2, dtype=torch.bfloat16); big_v[:, : bs * block] = v
        for i in range(L):
            a.set_kv_buffer_prefix_valid(attns[i], loc, commit, k[i].clone(), v[i].clone(), None, None)
        handled = b.set_kv_buffer_prefix_valid_layers(layers=attns, loc_2d=loc, commit_lens=commit,
                                                      cache_k_all=big_k[:, : bs * block], cache_v_all=big_v[:, : bs * block])
        same = all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a.k_buffer + a.v_buffer, b.k_buffer + b.v_buffer))
        print("RES", [handled, same])
    """)
    assert res == "[True, True]", res


def _gate_pool(mp):
    pool = object.__new__(mp.MHATokenToKVPool)
    pool.dtype, pool.store_dtype, pool.row_dim = torch.float8_e4m3fn, torch.uint8, 8
    pool.k_buffer = [torch.zeros(4, 8, dtype=torch.uint8) for _ in range(2)]
    pool.v_buffer = [torch.zeros(4, 8, dtype=torch.uint8), torch.zeros(5, 8, dtype=torch.uint8)]  # not uniform
    pool.local_slot = lambda i: i
    return pool


def test_pool_method_gates(monkeypatch):
    from flliper.srt.mem_cache import memory_pool as mp

    monkeypatch.setattr(mp, "_is_cuda", True)
    pool = _gate_pool(mp)
    attns = [types.SimpleNamespace(layer_id=i, k_scale=None, v_scale=None) for i in range(2)]
    loc, commit = torch.zeros(1, 2, dtype=torch.int64), torch.ones(1, dtype=torch.int32)
    k = torch.zeros(2, 2, 8, dtype=torch.bfloat16)

    def call(p):
        return p.set_kv_buffer_prefix_valid_layers(layers=attns, loc_2d=loc, commit_lens=commit, cache_k_all=k, cache_v_all=k)

    monkeypatch.setenv(mp.XKV_LAYERS_ENV, "0")
    assert call(pool) is False and pool._xkv_layers_ok is None  # switch off: not even the verdict is touched
    monkeypatch.delenv(mp.XKV_LAYERS_ENV)
    attns[1].k_scale = 0.5  # a scale that is not 1.0 with a cast: the per-layer division stays
    assert call(pool) is False and pool._xkv_layers_ok is None
    attns[1].k_scale = 1.0  # identity: passes the scale gate, then the non-uniform buffers stop it
    assert call(pool) is False and pool._xkv_layers_ok is False
    attns[1].k_scale = None

    pool_off = _gate_pool(mp)
    pool_off._xkv_layers_ok = False
    assert call(pool_off) is False

    class OwnWriter(mp.MHATokenToKVPool):  # a pool class with its own prefix-valid writer decides itself (page-major: refuses)
        def set_kv_buffer_prefix_valid(self, *args, **kwargs):
            raise NotImplementedError

    own = object.__new__(OwnWriter)
    own.__dict__.update(pool.__dict__)
    own._xkv_layers_ok = None
    assert call(own) is False and own._xkv_layers_ok is None

    monkeypatch.setattr(mp, "_is_cuda", False)
    monkeypatch.setattr(mp, "_is_hip", False)
    assert call(object.__new__(mp.MHATokenToKVPool)) is False


def test_page_major_pool_keeps_its_refusing_writer():
    from flliper.srt.mem_cache import memory_pool as mp

    cls = mp.PageMajorMHATokenToKVPool
    assert cls.set_kv_buffer_prefix_valid is not mp.MHATokenToKVPool.set_kv_buffer_prefix_valid


def test_materialize_takes_the_all_layers_writer_and_falls_back():
    from flliper.kernels.ops.speculative import fused_kv_materialize as fk

    h = object.__new__(fk.FusedKVMaterializeHelper)
    h.device = torch.device("cpu")
    h.flat_kv_weight_t = torch.zeros(4, 6, dtype=torch.bfloat16)  # [hidden, layers * layer_out_dim]
    h.max_position_hint = 100
    h.n_layers = 3
    h.layer_out_dim = 2
    h.num_kv_heads, h.head_dim, h.rotary_dim = 1, 2, 2
    h.k_norm_weights, h.eps_values = None, None
    h._mm_out_supported = True
    h._proj_workspace = torch.zeros(8, 6, dtype=torch.bfloat16)
    h._k_workspace = torch.zeros(3, 8, 1, 2, dtype=torch.bfloat16)
    h._v_workspace = torch.zeros(3, 8, 1, 2, dtype=torch.bfloat16)
    h._ensure_rope_cache = lambda m: None
    h._ensure_workspace = lambda n, dt: None
    ks = torch.arange(3 * 2).view(3, 2).float()
    orig = fk._fused_norm_rope_stacked
    fk._fused_norm_rope_stacked = lambda *a, **kw: (ks, ks + 100)
    try:
        per_layer, got = [], []
        x = torch.zeros(2, 4, dtype=torch.bfloat16)
        pos = torch.zeros(2, dtype=torch.int64)
        h.materialize(x, pos, write_layer_kv=lambda i, k, v: per_layer.append(i),
                      write_all_layers_kv=lambda k, v: got.append((k, v)) or True)
        assert per_layer == [] and len(got) == 1 and got[0][0] is ks
        h.materialize(x, pos, write_layer_kv=lambda i, k, v: per_layer.append(i), write_all_layers_kv=lambda k, v: False)
        assert per_layer == [0, 1, 2]
        per_layer.clear()
        h.materialize(x, pos, write_layer_kv=lambda i, k, v: per_layer.append(i))  # no writer given: the loop as before
        assert per_layer == [0, 1, 2]
    finally:
        fk._fused_norm_rope_stacked = orig


def test_worker_passes_the_all_layers_writer_only_for_prefix_valid_and_never_via_getattr():
    src = open(os.path.join(_PY, "flliper", "srt", "speculative", "dflash_worker_v2.py"), encoding="utf-8").read()
    assert "write_all_layers_kv=_write_all_layers_kv if prefix_valid else None" in src
    assert "isinstance(token_to_kv_pool, MHATokenToKVPool)" in src
    body = src[src.index("def _write_all_layers_kv"): src.index("self._fused_kv_helper.materialize(", src.index("def _write_all_layers_kv"))]
    assert "getattr(" not in body


_COMPILE = textwrap.dedent(
    """
    import torch
    from flliper.srt.mem_cache import memory_pool as mp
    from flliper.test import triton_compile_check as tcc
    # row bytes of the 27B draft pool per rank: 8 KV heads x 128 over TP (1 head 128 .. 8 heads 1024 elements), fp8 -> uint8 store or bf16
    GEOMS = [(128, torch.uint8), (384, torch.uint8), (1024, torch.uint8), (256, torch.bfloat16), (384, torch.bfloat16), (1024, torch.bfloat16),
             (2048, torch.bfloat16), (64, torch.bfloat16)]
    with tcc.capture(mp, "set_kv_buffer_prefix_valid_layers_tiled") as rec:
        for debug_env in ("0", "1"):  # FLLIPER_DISABLE_KV_MASKED_BOUND_CHECK: 0 = assert compiled in (default), 1 = off
            import os
            os.environ["FLLIPER_DISABLE_KV_MASKED_BOUND_CHECK"] = debug_env
            for rd, st in GEOMS:
                L, bs, block, slots = 5, 2, 8, 64
                kb = [torch.zeros(slots, rd, dtype=st) for _ in range(L)]
                vb = [torch.zeros(slots, rd, dtype=st) for _ in range(L)]
                k = torch.zeros(L, bs * block, rd, dtype=st)
                loc = torch.arange(bs * block, dtype=torch.int64).view(bs, block)
                commit = torch.full((bs,), block, dtype=torch.int32)
                mp._set_kv_buffer_prefix_valid_layers_impl(k, k, kb, vb, loc, commit, row_dim=rd, store_dtype=st, size_limit=slots)
    n_debug = sum(1 for ln in rec if ln.kwargs.get("debug") is True)
    out = []
    for cc in (120, 86):
        res = tcc.compile_all(rec, cc=cc)
        out.append([len(rec), n_debug, len(res), [kk + " -> " + e for kk, e in res if e is not None][:3]])
    print("RES", out)
    """
)


def test_the_layers_kernel_compiles_for_sm120_and_sm86_without_a_gpu():
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY)
    p = subprocess.run([sys.executable, "-c", _COMPILE], env=env, capture_output=True, text=True, timeout=1800)
    out = [line for line in p.stdout.splitlines() if line.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-3000:]
    for n_launch, n_debug, n_variants, bad in ast.literal_eval(out[0][4:]):
        assert n_launch == 16 and n_debug == 8, (n_launch, n_debug)  # every wrapper start was recorded, half of them with the assert
        assert n_variants >= 12, n_variants  # distinct (signature, constexpr, option) variants, each compiled
        assert bad == [], bad
