"""Exact-size page-locked spill pools (host-RAM peak of the load-time presplit).

Desk tests, no CUDA context: the helper must (a) hand back a tensor of the
requested shape/dtype whose byte size is EXACT, (b) route through
mmap + cudaHostRegister when a CUDA context exists (faked here), never through
``pin_memory()`` (whose CachingHostAllocator rounds to a power of two), and
(c) be what the presplit and the spill-pool allocation actually call.
"""

import math
import mmap

import pytest
import torch

from sglang.srt.layers.moe import expert_offload as eo


def _fake_cuda(monkeypatch, calls):
    class _Cudart:
        def cudaHostRegister(self, ptr, nbytes, flags):
            calls.append(("register", int(ptr), int(nbytes), int(flags)))
            return 0

        def cudaHostUnregister(self, ptr):
            calls.append(("unregister", int(ptr)))
            return 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())


def test_exact_bytes_and_shape_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t = eo.pinned_exact_empty((165, 800, 1024), torch.uint8)
    assert t.shape == (165, 800, 1024) and t.dtype == torch.uint8
    assert t.numel() * t.element_size() == 165 * 800 * 1024
    assert t.is_contiguous()


def test_registers_exact_page_aligned_range(monkeypatch):
    calls = []
    _fake_cuda(monkeypatch, calls)
    rows, cols = 165, 1_638_400 // 2  # 165 spill rows of 1.6384 MB (TP0, 32/16/16)
    t = eo.pinned_exact_empty((rows, cols), torch.bfloat16)
    nbytes = rows * cols * 2
    assert t.shape == (rows, cols) and t.dtype == torch.bfloat16
    assert calls == [("register", t.data_ptr(), nbytes, eo._CUDA_HOST_REGISTER_MAPPED)]
    assert t.data_ptr() % mmap.PAGESIZE == 0
    # The whole point: NOT the CachingHostAllocator's power-of-two size.
    assert nbytes != 1 << math.ceil(math.log2(nbytes))
    assert eo.pinned_exact_bytes() >= nbytes
    t[0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    assert t[0, :4].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert eo.pinned_exact_release(t) is True
    assert calls[-1] == ("unregister", t.data_ptr())
    assert eo.pinned_exact_release(t) is False


def test_register_failure_is_loud(monkeypatch):
    calls = []
    _fake_cuda(monkeypatch, calls)

    class _Bad:
        def cudaHostRegister(self, ptr, nbytes, flags):
            return 2  # cudaErrorMemoryAllocation

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Bad())
    with pytest.raises(RuntimeError, match="cudaHostRegister"):
        eo.pinned_exact_empty((4, 4), torch.float32)


def test_zero_rows_is_plain(monkeypatch):
    calls = []
    _fake_cuda(monkeypatch, calls)
    t = eo.pinned_exact_empty((0, 16), torch.uint8)
    assert t.shape == (0, 16) and calls == []


def test_presplit_and_spill_pool_use_exact_pins(monkeypatch):
    """No ``pin_memory()`` left on the presplit / spill-pool allocation paths."""
    import ast
    import inspect
    import textwrap

    def code_of(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        return ast.unparse(node)  # comments and docstrings gone: code only

    for fn in (eo.presplit_expert_offload_after_repack, eo.allocate_spill_pool):
        src = code_of(fn)
        assert "pin_memory()" not in src and "pinned_exact_empty(" in src, fn.__name__
