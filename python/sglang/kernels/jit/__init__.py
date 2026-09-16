"""Alias of this line's JIT kernel infrastructure under upstream's package
name. Upstream moved ``sglang.jit_kernel`` to ``sglang.kernels.jit``; the
#37500 bundle's ops (hyper-connection norm/combine, ...) import the new name
(fn1q boot 2026-09-16: ModuleNotFoundError on the first forward). The
kernels themselves (csrc/elementwise/*.cuh) live in ``sglang.jit_kernel``."""

from sglang import jit_kernel as _jit_kernel


def __getattr__(name):
    return getattr(_jit_kernel, name)
