# SPDX-License-Identifier: Apache-2.0
"""H101b -- the H101 boot prewarm warmed nothing on any rank of fnFL2h91bb2.

DER BEFUND (Boot fnFL2h91bb2, Baum e17bd548b5, 26.09.):
D-Log Z. 7739, TP0 14:29:54: 'H101 QSA-ROWS-PREWARM skipped: no rows launch
recorded on this rank ... forms=[]'; PP0/PP1/PP2 im P-Log Z. 2545-2569
dasselbe. Drei Sekunden VOR dieser Zeile hatte D-TP0 den Rows-Kern im
Draft-Decode-Capture gestartet (Z. 7646 'QSA-ROWS-LAUNCH arch=sm120 kv=fp8
... cfg=32/8/2 first_total_q=6'). Die Aufzeichnung hielt den Pool nur schwach,
und ``MHATokenToKVPool.get_key_buffer`` gibt bei fp8-KV
``k_buffer[slot].view(float8_e4m3fn)`` zurueck -- ein NEUES Tensor-Objekt je
Aufruf, das nach dem Launch niemand mehr haelt: die Referenz war tot, bevor
der Prewarm sie las. Die P-Stufen fangen gar keinen Graphen. Die erste echte
Extend einer Form zahlte ihren Cold-Load (und ein LMEM-Wachstum) mitten im
Betrieb (Z. 11563 'cfg=64/8/2 first_total_q=59').

Der Fix, hier CPU-hermetisch gepinnt: die Formen kommen aus der Rows-Tabelle
(Architektur + Env-Tisch, rows_launch_forms), die Spezialisierung nennt jedes
QwenSparseAttnBackend selbst aus Modell und Pool (Heads, head_dim, dtype,
K = indexer_budget + compress_ratio - 1, die Pool-Tensoren wie
get_key_buffer sie ausgibt) -- ohne dass vorher irgendetwas gestartet hat.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qbk
from sglang.srt.layers.attention.qsa import rows_prewarm as rp
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import lmem_census as lc
from sglang.srt.weg2 import sleep_lmem as sl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THREADS_5090 = 170 * 1536
#: NF geometry (weg2/tools/qsa_rows_bench.py): 24 q heads, 2 kv heads; head_dim
#: shrunk from 256 to 8 (CPU), the rows width is the real one.
HQ, HKV, D = 24, 2, 8
BUDGET, RATIO = 2048, 4
K_NF = BUDGET + RATIO - 1  # 2051, the 'K 2051' of '[qsa-rows] fused ... armed'


@pytest.fixture(autouse=True)
def _clean_state():
    def _reset():
        lc._reset_for_test()
        sa._ROWS_CONFIG_CACHE.clear()
        sa._ROWS_LAUNCH_SEEN.clear()
        sa._ROWS_PREWARM_SIG["sigs"].clear()
        sa._ROWS_PREWARM_SIG["done"] = False
        getattr(sa, "_ROWS_PREWARM_PROVIDERS", []).clear()

    _reset()
    yield
    _reset()


class _Fp8ViewPool:
    """The metal's pool shape: stores uint8, hands out a FRESH fp8 view per
    get_key_buffer call (MHATokenToKVPool._get_key_buffer, store_dtype !=
    dtype); a layer id it does not hold raises (HybridLinearKVPool)."""

    def __init__(self, layer_ids, n=64):
        self.k = {i: torch.zeros(n, HKV, D, dtype=torch.uint8) for i in layer_ids}
        self.v = {i: torch.zeros(n, HKV, D, dtype=torch.uint8) for i in layer_ids}

    def _get(self, store, layer_id):
        if layer_id not in store:
            raise ValueError(f"{layer_id=} not in full attention layers")
        return store[layer_id].view(torch.float8_e4m3fn)

    def get_key_buffer(self, layer_id):
        return self._get(self.k, layer_id)

    def get_value_buffer(self, layer_id):
        return self._get(self.v, layer_id)


class _Model(nn.Module):
    """A hybrid stage: linear-attention layers (no RadixAttention) and the
    full-attention layers this rank holds."""

    def __init__(self, full_ids, linear_ids=(0, 1, 2)):
        super().__init__()
        self.linear = nn.ModuleList(nn.Linear(2, 2) for _ in linear_ids)
        self.attn = nn.ModuleList(
            RadixAttention(num_heads=HQ, head_dim=D, scaling=D ** -0.5, num_kv_heads=HKV,
                           layer_id=i)
            for i in full_ids
        )


def _runner(pool, full_ids, budget=BUDGET):
    cfg = types.SimpleNamespace(indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=8,
                                indexer_budget=budget, indexer_compress_ratio=RATIO)
    model_config = types.SimpleNamespace(hf_text_config=cfg, context_len=262144,
                                         dtype=torch.bfloat16)
    return types.SimpleNamespace(token_to_kv_pool=pool, model=_Model(full_ids),
                                 model_config=model_config, dtype=torch.bfloat16,
                                 device="cpu", req_to_token_pool=None,
                                 is_draft_worker=False)


class _RecordedKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class _Driver:
    def __init__(self, stack):
        self.stack = stack

    def get_stack_bytes(self):
        return self.stack

    def set_stack_bytes(self, value):
        self.stack = value


def _device(monkeypatch, capability=(12, 0), name="NVIDIA GeForce RTX 5090", stack=1024):
    """Everything run_boot_prewarm touches on CUDA, mocked the way the H101
    tests mock it: the device, its properties, the stack-limit driver, the
    rows kernel itself, and an attending (not Form-A) rank."""
    from sglang.srt import rank_role

    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    monkeypatch.setattr(sl, "CudaDriverStackLimit", lambda: _Driver(stack))
    props = types.SimpleNamespace(multi_processor_count=170, max_threads_per_multi_processor=1536)
    monkeypatch.setattr(sa.torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(sa.torch.cuda, "get_device_name", lambda *a: name)
    monkeypatch.setattr(sa.torch.cuda, "get_device_properties", lambda *a: props)
    monkeypatch.setattr(sa.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(sa.torch.cuda, "is_available", lambda: False)
    kern = _RecordedKernel()
    monkeypatch.setattr(sa, "_sparse_attn_rows_fwd", kern)
    return kern


def _launched(kern):
    """(total_q, BLOCK_N, warps, stages, USE_COUNTS, K, heads) per launch."""
    out = []
    for grid, args, kw in kern.calls:
        q, k_pool, rows = args[0], args[1], args[5]
        out.append((grid[0], kw["BLOCK_N"], kw["num_warps"], kw["num_stages"],
                    kw["USE_COUNTS"], int(rows.shape[-1]), int(q.shape[1]),
                    kw["KV_FP8"], k_pool.dtype))
    return out


# -- the metal: a capture launch on an fp8 pool leaves no usable record --------


def test_rc_fnfl2h91bb2_a_capture_launch_on_an_fp8_pool_records_nothing_usable(monkeypatch):
    kern = _device(monkeypatch)
    pool = _Fp8ViewPool([3])
    q = torch.zeros(6, HQ, D, dtype=torch.bfloat16)
    rows = torch.full((6, K_NF), -1, dtype=torch.int32)
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""):
        # D-TP0's draft decode capture, bs=6 (Z. 7646): the launch happens ...
        sa.sparse_attn_rows_triton(q, pool.get_key_buffer(3), pool.get_value_buffer(3), rows, 0.1)
    assert len(kern.calls) == 1
    # ... and the weak record is dead by the time the prewarm reads it
    assert sa.rows_prewarm_signatures() == []


# -- the fix: the prewarm warms the table's forms without any recorded launch ---


def test_prewarm_without_a_recorded_launch_warms_every_table_form_on_d_tp0(monkeypatch):
    kern = _device(monkeypatch)
    pool = _Fp8ViewPool([3, 7, 11])
    backend = qbk.QwenSparseAttnBackend(_runner(pool, [3, 7, 11]))
    assert sa.rows_prewarm_signatures() == []  # nothing launched, nothing recorded
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""), \
            envs.SGLANG_WEG2_QSA_FP8_DECODE.override(""):
        res = rp.run_boot_prewarm()
    assert res is not None and res.status == "ok", res
    # the H101 sm120 table: four builds, each with and without the counts
    assert res.forms == ("32/8/2@1", "32/8/2+counts@1", "64/8/2@33", "64/8/2+counts@33",
                         "64/4/2@65", "64/4/2+counts@65", "32/4/2@129", "32/4/2+counts@129")
    got = _launched(kern)
    assert [(t, bn, w, s, c) for t, bn, w, s, c, *_ in got] == [
        (t, bn, w, s, c)
        for t, (bn, w, s) in ((1, (32, 8, 2)), (33, (64, 8, 2)), (65, (64, 4, 2)), (129, (32, 4, 2)))
        for c in (False, True)
    ]
    # the specialization serving launches: 24 heads, K 2051, the fp8 pool
    # decoded in-kernel (the uint8 view of the pool's storage)
    assert {(k, h, fp8, dt) for *_, k, h, fp8, dt in got} == {(K_NF, HQ, True, torch.uint8)}
    for _grid, args, _kw in kern.calls:
        assert args[1].untyped_storage().data_ptr() == pool.k[3].untyped_storage().data_ptr()
    line = res.line(THREADS_5090)
    assert line.startswith("H101 QSA-ROWS-PREWARM ok forms=[32/8/2@1, 32/8/2+counts@1,")
    assert "stack 1024->1024 B lmem 255->255 MiB" in line
    assert f"sigs=[{HQ}x{D} bfloat16 K{K_NF} kv=float8_e4m3fn (derived)]" in line
    del backend


def test_the_p_stage_warms_its_arm_form_on_sm86(monkeypatch):
    """PP1/PP2 (3080, sm86): the arm's env table inf=64/8/2 is one build; the
    stage's pool holds only its own full-attention layers."""
    kern = _device(monkeypatch, capability=(8, 6), name="NVIDIA GeForce RTX 3080")
    pool = _Fp8ViewPool([35, 39])
    backend = qbk.QwenSparseAttnBackend(_runner(pool, [35, 39]))
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("inf=64/8/2"), \
            envs.SGLANG_WEG2_QSA_FP8_DECODE.override("ptx"):
        res = rp.run_boot_prewarm()
    assert res.status == "ok" and res.forms == ("64/8/2@1", "64/8/2+counts@1")
    assert [kw["FP8_DECODE"] for *_, kw in kern.calls] == [sa.FP8_DECODE_PTX] * 2
    del backend


def test_target_and_draft_with_one_build_key_warm_once(monkeypatch):
    kern = _device(monkeypatch)
    target = qbk.QwenSparseAttnBackend(_runner(_Fp8ViewPool([3]), [3]))
    draft = qbk.QwenSparseAttnBackend(_runner(_Fp8ViewPool([48]), [48]))  # MTP layer, own pool
    sigs, errors = sa.derived_rows_prewarm_signatures()
    assert errors == [] and len(sigs) == 1  # same heads/dim/dtype/K/pool strides
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("inf=64/8/2"):
        res = rp.run_boot_prewarm()
    assert res.forms == ("64/8/2@1", "64/8/2+counts@1") and len(kern.calls) == 2
    del target, draft


def test_mtp_index_sharing_adds_the_shared_width_as_a_second_build(monkeypatch):
    _device(monkeypatch)
    backend = qbk.QwenSparseAttnBackend(_runner(_Fp8ViewPool([48]), [48]))
    backend.set_mtp_shared_sparse_indices(
        types.SimpleNamespace(indices=torch.zeros(1, 2, K_NF + 3 + 1, dtype=torch.int32)))
    sigs, _ = sa.derived_rows_prewarm_signatures()
    assert [s["k"] for s in sigs] == [K_NF, K_NF + 4]
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("inf=64/8/2"):
        res = rp.run_boot_prewarm()
    assert res.forms == ("s0:64/8/2@1", "s0:64/8/2+counts@1", "s1:64/8/2@1", "s1:64/8/2+counts@1")


def test_the_backend_derives_heads_under_dcp_and_skips_layers_its_pool_lacks():
    pool = _Fp8ViewPool([7])
    backend = qbk.QwenSparseAttnBackend(_runner(pool, [3, 7]))  # layer 3 is another stage's
    (sig,) = backend.rows_prewarm_signatures()
    assert (sig["heads"], sig["head_dim"], sig["dtype"], sig["k"]) == (HQ, D, torch.bfloat16, K_NF)
    assert sig["k_pool"].dtype == torch.float8_e4m3fn and sig["source"] == "derived"
    # under DCP _attend_rows gathers the group's q heads: the build sees their sum
    backend.dcp_size = 3
    with mock.patch.object(backend, "_dcp_group_q_head_counts", lambda local: [8, 10, 6]):
        (sig,) = backend.rows_prewarm_signatures()
    assert sig["heads"] == 24
    # no QSA profile / no pool / no layer of this pool: nothing named
    assert qbk.QwenSparseAttnBackend(_runner(_Fp8ViewPool([99]), [3])).rows_prewarm_signatures() == []
    assert qbk.QwenSparseAttnBackend(None).rows_prewarm_signatures() == []


def test_a_rank_without_qsa_attention_is_still_a_named_skip(monkeypatch):
    _device(monkeypatch)
    res = rp.run_boot_prewarm()
    assert res.status.startswith("skipped: no rows launch recorded and no QSA backend")
    assert res.forms == ()


def test_a_provider_that_raises_is_named_and_the_others_still_warm(monkeypatch):
    kern = _device(monkeypatch)
    good = qbk.QwenSparseAttnBackend(_runner(_Fp8ViewPool([3]), [3]))

    class _Broken:
        def rows_prewarm_signatures(self):
            raise RuntimeError("pool not built")

    broken = _Broken()
    sa.register_rows_prewarm_provider(broken)
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override("inf=64/8/2"):
        res = rp.run_boot_prewarm()
    assert len(kern.calls) == 2 and res.forms == ("64/8/2@1", "64/8/2+counts@1")
    assert any(e.startswith("derive _Broken: RuntimeError: pool not built") for e in res.errors)
    del good, broken


def test_the_warmed_census_is_what_the_wake_books(monkeypatch):
    """The prewarm's loads feed the census at the #1056 chokepoint; the first
    sleep's line carries it and every wake restores it (H101 c) -- the
    spilling build's growth happens at boot, never inside a serving launch."""
    kern = _device(monkeypatch, capability=(8, 6), name="NVIDIA GeForce RTX 3080")
    loaded = []

    class _LoadingKernel(_RecordedKernel):
        def __getitem__(self, grid):
            inner = super().__getitem__(grid)

            def launch(*args, **kwargs):
                build = (kwargs["BLOCK_N"], kwargs["num_warps"], kwargs["num_stages"],
                         kwargs["USE_COUNTS"])
                if build not in loaded:  # cold load: the chokepoint's census
                    loaded.append(build)
                    lc.record("_sparse_attn_rows_fwd", 2320 if build[:3] == (16, 1, 2) else 0)
                inner(*args, **kwargs)

            return launch

    monkeypatch.setattr(sa, "_sparse_attn_rows_fwd", _LoadingKernel())
    backend = qbk.QwenSparseAttnBackend(_runner(_Fp8ViewPool([3]), [3]))
    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(""):
        res = rp.run_boot_prewarm()
    # sm86 keeps the L20 table: its >512 band is the spilling (16, 1, 2)
    assert "16/1/2@513" in res.forms and res.census_max_bytes == 2320
    assert "census_max=2320(_sparse_attn_rows_fwd)" in res.line()
    drv = _Driver(1024)
    park = sl.park_lmem(driver=drv, threads=104448, nvml_bytes=lambda: 0, base_stack_bytes=1024)
    booked, kernel = lc.census_max()
    rec = sl.restore_lmem(driver=drv, park=park, nvml_bytes=lambda: 0,
                          booked_stack_bytes=booked, booked_kernel=kernel)
    assert drv.stack == 2320 and rec.booked_stack_bytes == 2320
    del backend, kern


if __name__ == "__main__":
    unittest.main()
