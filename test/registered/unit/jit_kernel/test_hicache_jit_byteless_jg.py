# SPDX-License-Identifier: Apache-2.0
"""JG -- a byteless host pool (0 bytes per element) never builds a JIT HiCache
kernel and never launches a copy.

fnFL2h91bb1/bb2/v1: on the NF D group the Form A expert workers TP1/TP2 hold
0 kv-heads, so ``MHATokenToKVPoolHost 0.00 GB`` asked
``can_use_hicache_jit_kernel(element_size=0)``. ``0 % 128 == 0`` let it
through, nvcc instantiated ``HiCacheKernel<0, 4, 2, 1024>`` and failed on
``zero-sized variable "vec"``; the next boot then discarded the half-built
cache entry. The D group waited 6-7 s at init for the two failed builds.
All CPU, the JIT is mocked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel import hicache as hc
from sglang.srt.mem_cache.pool_host import base as host_base
from sglang.srt.mem_cache.pool_host import mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def infos(monkeypatch):
    lines = []
    monkeypatch.setattr(hc, "_byteless_sites_logged", set(), raising=False)
    monkeypatch.setattr(
        hc.logging.getLogger(hc.__name__), "info", lambda fmt, *a: lines.append(fmt % a)
    )
    return lines


def _recorder(monkeypatch, name):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return object()

    monkeypatch.setattr(hc, name, fake)
    return calls


def test_zero_element_builds_no_hicache_module(monkeypatch, infos):
    built = _recorder(monkeypatch, "_jit_hicache_module")
    assert hc.can_use_hicache_jit_kernel(element_size=0) is False
    assert built == []
    # control: a real element size still asks for its module
    assert hc.can_use_hicache_jit_kernel(element_size=512) is True
    assert [c["element_size"] for c in built] == [512]


def test_zero_element_builds_no_staged_module(monkeypatch, infos):
    built = _recorder(monkeypatch, "_jit_hicache_staged_module")
    assert hc.can_use_write_back_jit_kernel(element_size=0) is False
    assert built == []
    assert hc.can_use_write_back_jit_kernel(element_size=512) is True
    assert [c["element_size"] for c in built] == [512]


def test_one_named_info_line_per_site(monkeypatch, infos):
    _recorder(monkeypatch, "_jit_hicache_module")
    for _ in range(3):
        hc.can_use_hicache_jit_kernel(element_size=0)
    tagged = [l for l in infos if l.startswith("HICACHE-JIT BYTELESS ")]
    assert len(tagged) == 1
    assert "site=can_use_hicache_jit_kernel element_size=0" in tagged[0]


@pytest.mark.parametrize(
    "module_fn", ["_jit_hicache_module", "_jit_hicache_staged_module"]
)
def test_zero_element_never_reaches_nvcc(monkeypatch, module_fn):
    """The guard behind the gate: a 0-byte element that slips past every pool
    gate dies named before load_jit (nvcc)."""
    built = []
    monkeypatch.setattr(hc, "load_jit", lambda *a, **kw: built.append(a) or object())
    fn = getattr(getattr(hc, module_fn), "__wrapped__", None)
    assert fn is not None
    with pytest.raises(hc.HiCacheJitZeroElementError, match="HICACHE-JIT ZERO-ELEMENT"):
        fn(element_size=0, unroll=4, block_quota=2)
    assert built == []


def _host(monkeypatch, *, head_num, layout):
    """A real MHATokenToKVPoolHost.__init__ over a CPU stand-in for the base
    allocation (HostKVCache.__init__), with the CUDA branch forced on."""
    layer_num, size, head_dim = 2, 128, 128
    device_pool = SimpleNamespace(
        head_num=head_num, head_dim=head_dim, layer_num=layer_num, device="cpu"
    )

    def fake_base_init(self, device_pool, *a, **kw):
        self.device_pool = device_pool
        self.dtype = torch.bfloat16
        self.layout = layout
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.page_size = 64
        self.page_num = size // 64
        self.size = size
        if layout == "page_first":
            dims = (2, size, layer_num, head_num, head_dim)
        else:
            dims = (2, layer_num, size, head_num, head_dim)
        self.kv_buffer = torch.empty(dims, dtype=self.dtype)
        self.token_stride_size = head_num * head_dim * 2
        self.layout_dim = self.token_stride_size * layer_num

    monkeypatch.setattr(host_base.HostKVCache, "__init__", fake_base_init)
    monkeypatch.setattr(mha, "_is_cuda", True)
    gates = {"jit": [], "write_back": []}
    monkeypatch.setattr(
        mha,
        "can_use_hicache_jit_kernel",
        lambda **kw: gates["jit"].append(kw["element_size"]) or True,
    )
    monkeypatch.setattr(
        mha,
        "can_use_write_back_jit_kernel",
        lambda **kw: gates["write_back"].append(kw["element_size"]) or False,
    )
    host = mha.MHATokenToKVPoolHost(device_pool, 1.0, 0, 64, layout)
    return host, device_pool, gates


@pytest.mark.parametrize("layout", ["layer_first", "page_first"])
def test_byteless_pool_asks_no_gate(monkeypatch, infos, layout):
    host, _, gates = _host(monkeypatch, head_num=0, layout=layout)
    assert gates == {"jit": [], "write_back": []}
    assert not host.can_use_jit
    assert not host.can_use_write_back_jit
    assert any(
        l.startswith("HICACHE-JIT BYTELESS site=MHATokenToKVPoolHost element_size=0")
        for l in infos
    )


def test_real_pool_still_asks_the_gate(monkeypatch, infos):
    host, _, gates = _host(monkeypatch, head_num=2, layout="page_first")
    assert gates == {"jit": [512], "write_back": [512]}
    assert host.can_use_jit
    assert not [l for l in infos if "HICACHE-JIT BYTELESS" in l]


@pytest.mark.parametrize("layout", ["layer_first", "page_first"])
@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
def test_byteless_pool_launches_no_copy(monkeypatch, infos, layout, io_backend):
    host, device_pool, _ = _host(monkeypatch, head_num=0, layout=layout)
    launched = []
    for name in (
        "jit_transfer_hicache_one_layer",
        "jit_transfer_hicache_all_layer",
        "jit_transfer_hicache_all_layer_staged_lf_pf",
        "transfer_kv_per_layer",
        "transfer_kv_per_layer_pf_lf",
        "transfer_kv_all_layer",
        "transfer_kv_all_layer_lf_pf",
        "transfer_kv_direct",
    ):
        monkeypatch.setattr(
            mha, name, lambda *a, _n=name, **kw: launched.append(_n), raising=False
        )
    monkeypatch.setattr(mha, "_guard_kv_transfer", lambda *a, **kw: None)
    device_pool.k_buffer = [torch.empty(0)] * 2
    device_pool.v_buffer = [torch.empty(0)] * 2
    device_pool.k_data_ptrs = torch.zeros(2, dtype=torch.uint64)
    device_pool.v_data_ptrs = torch.zeros(2, dtype=torch.uint64)
    idx = torch.arange(64, dtype=torch.int64)
    # a byteless pool returns before the layout dispatch (so even the
    # unsupported "direct" + page_first pair is a no-op, not a ValueError)
    host.backup_from_device_all_layer(device_pool, idx, idx, io_backend)
    for layer_id in range(2):
        host.load_to_device_per_layer(device_pool, idx, idx, layer_id, io_backend)
    assert launched == []
