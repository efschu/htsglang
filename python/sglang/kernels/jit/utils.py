"""``sglang.kernels.jit.utils`` -> this line's ``sglang.jit_kernel.utils``.

The #37500 bundle's kernels were written against upstream's newer
``sgl_kernel`` headers (CHECK_HOST, DTypeTrait, ...). Those headers live in
this tree as ``sgl_kernel_next`` (kernels/jit/include), and the bundle's
.cuh files include them under that name, so nothing of this line's own JIT
kernels (built against ``jit_kernel/include/sgl_kernel``) changes. ``load_jit``
here adds that include root."""

import pathlib

from sglang.jit_kernel import utils as _utils
from sglang.jit_kernel.utils import (  # noqa: F401  (the names the ops import)
    cache_once,
    is_arch_support_pdl,
    make_cpp_args,
)

NEXT_INCLUDE = str(pathlib.Path(__file__).resolve().parent / "include")


def load_jit(*args, extra_include_paths=None, **kwargs):
    paths = [NEXT_INCLUDE] + list(extra_include_paths or [])
    kwargs.setdefault("wrap_namespace", "sglang")
    return _utils.load_jit(*args, extra_include_paths=paths, **kwargs)


def __getattr__(name):
    return getattr(_utils, name)
