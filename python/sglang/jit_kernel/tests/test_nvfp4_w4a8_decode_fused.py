"""Fused activation quantisation in the N4D W4A8 decode GEMV (release table row 25, SGLANG_W4A8_DECODE_FUSED_QUANT).

Desk (no GPU):
  * the switch routes nvfp4_w4a8_decode_linear to ONE fused launch, default stays quant + GEMM;
  * the semaphore is keyed by (device, weight, stream): two streams never share one;
  * the hand-over protocol of the .cuh (rows CLAIMED through an atomic counter), simulated with a bounded number of
    resident CTAs, legal but adversarial dispatch orders and random interleavings: it never deadlocks, quantises
    every row exactly once before any read and leaves the semaphore at zero -- while static row owners (by blockIdx
    or by arrival ticket, the first design) DO deadlock, which is the danger the claim counter removes;
  * source guards: the fused kernels read the activation only through the coherent loader, wait before the main
    loop, and the default module is not built with the fused define.
GPU (TEST27B_GPU=1, own gpuq window, sm_86): fused == quant + GEMM bit for bit for every config, repeated launches and
CUDA-graph replays (the semaphore resets itself), and the int8/scale workspace equals the separate quantiser.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import random
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_nvfp4_w4a8 as T4  # noqa: E402

CUH = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "gemm" / "nvfp4_w4a8_decode_sm86.cuh"


# ------------------------------------------------------------------------------------------------------------------
# desk: routing + semaphore keying
# ------------------------------------------------------------------------------------------------------------------
def test_switch_routes_to_one_fused_launch(monkeypatch):
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    calls = []
    monkeypatch.setattr(dec, "decode_fused", lambda x, w, s, gs, n, cfg: calls.append(("fused", cfg)) or "F")
    monkeypatch.setattr(dec, "quantize_activation", lambda x, permuted: calls.append(("quant", permuted)) or (1, 2))
    monkeypatch.setattr(dec, "decode_gemm", lambda *a, **k: calls.append(("gemm",)) or "G")
    monkeypatch.delenv("SGLANG_W4A8_DECODE_CFG", raising=False)
    dec.config_for.cache_clear()
    x = torch.zeros((8, 5120), dtype=torch.bfloat16)
    w = torch.zeros((8192, 2560), dtype=torch.uint8)
    monkeypatch.delenv("SGLANG_W4A8_DECODE_FUSED_QUANT", raising=False)
    assert not dec.fused_quant_enabled()
    assert dec.nvfp4_w4a8_decode_linear(x, w, None, None, 8192) == "G"
    assert calls == [("quant", True), ("gemm",)]
    calls.clear()
    for v in ("1", "on", "true"):
        monkeypatch.setenv("SGLANG_W4A8_DECODE_FUSED_QUANT", v)
        assert dec.nvfp4_w4a8_decode_linear(x, w, None, None, 8192) == "F"
    assert calls == [("fused", dec.config_for(8, 8192, 5120))] * 3
    monkeypatch.setenv("SGLANG_W4A8_DECODE_FUSED_QUANT", "0")
    assert not dec.fused_quant_enabled()
    dec.config_for.cache_clear()


def test_semaphore_keyed_by_weight_and_stream(monkeypatch):
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    monkeypatch.setattr(dec, "_FUSED_SEMS", {})
    stream = {"h": 11}
    monkeypatch.setattr(dec, "_stream_handle", lambda device: stream["h"])
    w1 = torch.zeros((16, 64), dtype=torch.uint8)
    w2 = torch.zeros((16, 64), dtype=torch.uint8)
    a = dec.fused_semaphore(w1)
    assert a.dtype == torch.int32 and a.numel() >= 3 and int(a.abs().sum()) == 0
    assert dec.fused_semaphore(w1) is a  # same weight, same stream: stream order serialises the launches
    assert dec.fused_semaphore(w2) is not a  # another weight
    stream["h"] = 12
    assert dec.fused_semaphore(w1) is not a  # another stream may run concurrently: never share


# ------------------------------------------------------------------------------------------------------------------
# desk: the ticket protocol, simulated
# ------------------------------------------------------------------------------------------------------------------
class _Cta:
    """One CTA of the fused kernel as a generator of scheduling points (the .cuh control flow, step by step).

    scheme "claim"  -- the kernel: rows claimed through sem[0] by whoever arrives (fused_quant_claimed_rows);
    scheme "ticket" -- rejected: static owner = arrival ticket (rows t, t + nctas, ...);
    scheme "bid"    -- rejected: static owner = blockIdx.
    """

    def __init__(self, bid, sim):
        self.bid, self.sim = bid, sim
        self.gen = self._run()

    def _rows_static(self, owner):
        return list(range(owner, self.sim.M, self.sim.nctas))

    def _run(self):
        s = self.sim
        yield  # first weight loads in flight
        rows = 0
        if s.scheme == "claim":
            while True:
                r = s.M
                if s.sem[0] < s.M:  # relaxed peek, then atomicAdd
                    yield
                    r = s.sem[0]
                    s.sem[0] += 1
                if r >= s.M:
                    break
                yield  # quantising row r takes time
                s.quantised[r] += 1
                rows += 1
        else:
            if s.scheme == "ticket":
                owner = s.sem[0]
                s.sem[0] += 1
            else:
                owner = self.bid
            for r in self._rows_static(owner):
                yield
                s.quantised[r] += 1
                rows += 1
        if rows:
            s.sem[1] += rows  # red.release.gpu
        yield
        while s.sem[1] < s.M:  # fused_wait
            yield "spin"
        assert all(q == 1 for q in s.quantised), "activation read before every row was quantised"
        s.tiles.append(self.bid)
        yield
        d = s.sem[2]
        s.sem[2] += 1  # atomicAdd(sem + 2, 1)
        if d == s.nctas - 1:
            s.sem[0] = s.sem[1] = s.sem[2] = 0


class _Sim:
    def __init__(self, M, nctas, slots, scheme, rng, order):
        self.M, self.nctas, self.slots, self.scheme = M, nctas, slots, scheme
        self.sem = [0, 0, 0]
        self.quantised = [0] * M
        self.tiles = []
        self.rng, self.order = rng, order

    def run(self, max_steps=400000):
        pending = list(range(self.nctas))
        if self.order == "reverse":  # CUDA promises no dispatch order: highest blockIdx first is legal
            pending.reverse()
        elif self.order == "random":
            self.rng.shuffle(pending)
        resident = []
        for _ in range(max_steps):
            while pending and len(resident) < self.slots:
                resident.append(_Cta(pending.pop(0), self))
            if not resident:
                return True
            progressed = False
            order = list(resident)
            self.rng.shuffle(order)  # arbitrary interleaving of the resident CTAs
            for c in order:
                try:
                    progressed |= next(c.gen) != "spin"
                except StopIteration:
                    resident.remove(c)
                    progressed = True
            if not progressed:
                return False  # every resident CTA spins: the rows it waits for can never be quantised
        return False


@pytest.mark.parametrize("M", [1, 8, 16, 48])
@pytest.mark.parametrize("nctas,slots", [(512, 68 * 7), (320, 68), (40, 4), (6, 1), (3, 2), (1, 1)])
def test_claim_protocol_never_deadlocks_and_resets(M, nctas, slots):
    for seed, order in itertools.product(range(3), ("forward", "reverse", "random")):
        sim = _Sim(M, nctas, slots, "claim", random.Random(seed), order)
        assert sim.run(), (M, nctas, slots, seed, order)
        assert sim.sem == [0, 0, 0]
        assert sim.quantised == [1] * M
        assert sorted(sim.tiles) == list(range(nctas))


@pytest.mark.parametrize("scheme,M,nctas,slots,order", [
    ("bid", 8, 64, 16, "reverse"),  # blockIdx owners 0..7 never get a slot: 16 spinning CTAs hold them all
    ("ticket", 8, 6, 1, "forward"),  # ticket 0 owns rows 0 and 6, waits for rows 1..5, 7 -- their owners wait for its slot
    ("ticket", 48, 40, 4, "forward"),
])
def test_static_row_owners_deadlock_where_claiming_does_not(scheme, M, nctas, slots, order):
    """The danger the claim counter removes (found by this simulation on the first design, which used tickets)."""
    assert not _Sim(M, nctas, slots, scheme, random.Random(0), order).run(max_steps=5000)
    assert _Sim(M, nctas, slots, "claim", random.Random(0), order).run()


# ------------------------------------------------------------------------------------------------------------------
# desk: source guards on the .cuh
# ------------------------------------------------------------------------------------------------------------------
def _kernel_src(name):
    src = CUH.read_text()
    i = src.index(f"w4a8_dec_{name}_kernel(const DecParamsT<FUSED> p)")
    return src[i : src.index("\n}\n", i)]


@pytest.mark.parametrize("name", ["diag", "xpose"])
def test_fused_kernels_read_activation_coherently_after_the_wait(name):
    body = _kernel_src(name)
    assert "ldg_keep16(" not in body  # activation loads go through ldg_act16<FUSED> only
    assert body.count("ldg_act16<FUSED>(") >= 1
    wait = body.index("    fused_wait(p);")
    assert body.index("for (int c = warp;") > wait  # main loop (first activation read) after the wait
    assert "stagger_of(blockIdx.x," in body and "T0 = blockIdx.x * p.rw" in body  # the unfused tile mapping
    claim = body.index("fused_quant_claimed_rows<")
    assert body.index("load_seg(warp * U + u, cur[u]);") < claim < wait  # weights in flight during the claim
    assert "fused_done(p," in body
    src = CUH.read_text()
    act = src[src.index("__device__ __forceinline__ uint4 ldg_act16") :][:900]
    fused_branch = act[: act.index("} else {")]
    assert ".nc" not in fused_branch and "asm volatile" in fused_branch and '"memory"' in fused_branch


def test_default_module_is_built_without_the_fused_define():
    import inspect

    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    src = inspect.getsource(dec._jit_module)
    assert "SGL_W4A8_DEC_FUSED" not in src
    assert "-DSGL_W4A8_DEC_FUSED=1" in inspect.getsource(dec._jit_fused_module)
    cuh = CUH.read_text()
    # the default entry points are compiled out of the fused module and vice versa
    assert cuh.count("#ifndef SGL_W4A8_DEC_FUSED") == 2 and cuh.count("#ifdef SGL_W4A8_DEC_FUSED") == 1


# ------------------------------------------------------------------------------------------------------------------
# GPU (prepared; runs only with TEST27B_GPU=1 in an own gpuq window on a 3080)
# ------------------------------------------------------------------------------------------------------------------
gpu = T4.gpu


def _all_cfgs(m):
    modes = (0, 1) if m <= 8 else (1,)
    pairs = [(kw, rw) for kw in (1, 2, 4, 8) for rw in (1, 2, 4, 8) if kw * rw <= 8]
    return [(md, kw, rw, u) for md in modes for (kw, rw) in pairs for u in (1, 2, 4)]


@gpu
@pytest.mark.parametrize("m", [1, 2, 4, 5, 8, 9, 16, 17, 33, 48])
@pytest.mark.parametrize("n,k", [(256, 1024), (224, 640), (1024, 384), (8192, 5120)])
def test_fused_equals_quant_plus_gemm_bitwise(m, n, k):
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    gen = torch.Generator().manual_seed(m * 131 + n + k)
    w, ws, ws2 = T4._random_nvfp4(n, k, gen)
    w = w.cuda()
    swz = T4._swizzle_like_modelopt(ws).cuda()
    ws2 = ws2.cuda()
    x = (torch.randn((m, k), generator=gen) * 3).to(torch.bfloat16).cuda()
    sem = torch.zeros(4, dtype=torch.int32, device="cuda")
    cfgs = _all_cfgs(m) if n * k <= 1024 * 1024 else [dec.config_for(m, n, k)]
    for cfg in cfgs:
        xq, xs = dec.quantize_activation(x, permuted=(cfg[0] == 1))
        ref = dec.decode_gemm(xq, xs, w, swz, ws2, n, cfg)
        for _ in range(3):  # the semaphore must be back at zero after every launch
            y = dec.decode_fused(x, w, swz, ws2, n, cfg, sem=sem)
            assert torch.equal(y, ref), cfg
        torch.cuda.synchronize()
        assert int(sem[:3].abs().sum()) == 0, (cfg, sem.tolist())


@gpu
def test_fused_under_cuda_graph_replay_and_strided_x():
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    gen = torch.Generator().manual_seed(25)
    n, k, m = 5120, 4096, 8
    w, ws, ws2 = T4._random_nvfp4(n, k, gen)
    w, swz, ws2 = w.cuda(), T4._swizzle_like_modelopt(ws).cuda(), ws2.cuda()
    big = (torch.randn((m, k + 64), generator=gen)).to(torch.bfloat16).cuda()
    x = big[:, :k]  # row stride k + 64 (a view, as a fused-add-norm output slice may be)
    xq, xs = dec.quantize_activation(x.contiguous(), permuted=True)
    ref = dec.decode_gemm(xq, xs, w, swz, ws2, n)
    os.environ["SGLANG_W4A8_DECODE_FUSED_QUANT"] = "1"
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            dec.nvfp4_w4a8_decode_linear(x, w, swz, ws2, n)  # warm-up creates the semaphore of the capture stream
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                y = dec.nvfp4_w4a8_decode_linear(x, w, swz, ws2, n)
        torch.cuda.current_stream().wait_stream(s)
        for _ in range(5):
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, ref)
    finally:
        os.environ.pop("SGLANG_W4A8_DECODE_FUSED_QUANT", None)
