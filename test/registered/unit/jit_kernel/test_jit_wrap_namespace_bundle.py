"""The #37500 bundle's JIT kernels (grouped_gemma_rmsnorm, hc_combine,
fast_topk, qsa_indexer) build against upstream's newer sgl_kernel headers,
shipped here as ``sgl_kernel_next``, through the ``sglang.kernels.jit``
alias: it adds that include root and asks load_jit to wrap the exports in
``namespace sglang`` (upstream's form). This line's own kernels keep the
bare wrapper and therefore their build hashes (fn1r boot 2026-09-16: the
first forward's nvcc build failed on CHECK_HOST / bf16_t / DTypeTrait)."""

import inspect
import os

from sglang.jit_kernel import utils as ours
from sglang.kernels.jit import utils as alias


def test_default_wrapper_is_the_bare_export_line():
    line = ours._make_wrapper(("f", "Kernel<1>::run"))
    assert line == "TVM_FFI_DLL_EXPORT_TYPED_FUNC(f, (Kernel<1>::run));"
    assert ours._wrap_in_namespace([line], None) == [line]
    assert ours._wrap_in_namespace([], "sglang") == []
    wrapped = ours._wrap_in_namespace([line], "sglang")
    assert wrapped[0] == "namespace sglang {" and wrapped[-1].startswith("}")
    assert "wrap_namespace" in inspect.signature(ours.load_jit).parameters


def test_alias_adds_the_next_include_root_and_the_namespace(monkeypatch):
    seen = {}

    def fake_load_jit(*args, **kwargs):
        seen.update(kwargs)
        return "module"

    monkeypatch.setattr(ours, "load_jit", fake_load_jit)
    assert alias.load_jit("k", cuda_files=["x.cuh"]) == "module"
    assert seen["wrap_namespace"] == "sglang"
    assert os.path.isdir(os.path.join(seen["extra_include_paths"][0], "sgl_kernel_next"))
    assert alias.cache_once is ours.cache_once and alias.make_cpp_args is ours.make_cpp_args


def test_bundle_kernels_include_the_next_headers_only():
    root = os.path.join(os.path.dirname(ours.__file__), "csrc")
    for rel in (
        "elementwise/grouped_gemma_rmsnorm.cuh",
        "elementwise/hc_combine.cuh",
        "elementwise/fast_topk.cuh",
        "attention/qsa_indexer.cuh",
    ):
        src = open(os.path.join(root, rel)).read()
        assert "<sgl_kernel/" not in src and "<sgl_kernel_next/" in src, rel
    nxt = os.path.join(os.path.dirname(alias.__file__), "include", "sgl_kernel_next")
    for dirpath, _, files in os.walk(nxt):
        for f in files:
            assert "<sgl_kernel/" not in open(os.path.join(dirpath, f), errors="ignore").read(), f
