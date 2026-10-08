"""Alias of this line's JIT kernel infrastructure under upstream's package
name. Upstream moved ``flliper.jit_kernel`` to ``flliper.kernels.jit``; the
#37500 bundle's ops (hyper-connection norm/combine, ...) import the new name
(fn1q boot 2026-09-16: ModuleNotFoundError on the first forward). The
kernels themselves (csrc/elementwise/*.cuh) live in ``flliper.jit_kernel``."""

from flliper import jit_kernel as _jit_kernel
from flliper.kernels.jit import utils  # noqa: F401  -- bind the submodule BEFORE
# the fallback below: `from flliper.kernels.jit import utils` resolves through
# the package attribute first, and the fallback would hand back
# flliper.jit_kernel.utils instead of the alias with the sgl_kernel_next root.


def __getattr__(name):
    return getattr(_jit_kernel, name)
